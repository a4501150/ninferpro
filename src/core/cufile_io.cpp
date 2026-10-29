#include "core/cufile_io.h"

#include <cuda_runtime.h>
#include <cufile.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <mutex>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <unordered_map>

namespace infernix {
namespace {

std::string cufile_error(const char* what, CUfileError_t status) {
    std::string out = std::string(what) + ": ";
    if (status.err == CU_FILE_CUDA_DRIVER_ERROR) {
        out += "CUDA driver error " + std::to_string(static_cast<int>(status.cu_err));
    } else if (IS_CUFILE_ERR(status.err)) {
        out += CUFILE_ERRSTR(status.err);
    } else {
        out += "cuFile error " + std::to_string(static_cast<int>(status.err));
    }
    return out;
}

void check(CUfileError_t status, const char* what) {
    if (status.err != CU_FILE_SUCCESS) { throw std::runtime_error(cufile_error(what, status)); }
}

// cuFile 1.19 does not survive a driver close followed by a reopen in one process (its POSIX
// pools are freed twice, or a later batch status call faults), and every Engine opens a queue.
// The driver opens once per process and stays open until exit.
void open_driver() {
    static std::once_flag once;
    static std::string failure;
    std::call_once(once, [] {
        const CUfileError_t status = cuFileDriverOpen();
        if (status.err != CU_FILE_SUCCESS) { failure = cufile_error("cuFile driver open", status); }
    });
    if (!failure.empty()) { throw std::runtime_error(failure); }
}

bool final_status(CUfileStatus_t status) noexcept {
    return status != CUFILE_WAITING && status != CUFILE_PENDING;
}

} // namespace

struct CuFileQueue::Impl {
    struct Part {
        std::uint64_t parent      = 0;
        Op op                     = Op::Read;
        std::uint64_t offset      = 0;
        std::size_t bytes         = 0;
        void* buffer              = nullptr;
        std::size_t buffer_offset = 0;
        bool device               = false;
    };
    struct Parent {
        std::uint64_t tag       = 0;
        std::uint32_t remaining = 0;
        bool ok                 = true;
        std::string error;
    };
    struct Batch {
        CUfileBatchHandle_t handle = nullptr;
        std::uint32_t outstanding  = 0;
        std::vector<Part> parts;
        std::vector<CUfileIOParams_t> params;
        std::vector<CUfileIOEvents_t> events;
    };

    Options options;
    int fd              = -1;
    std::uint64_t bytes = 0;
    CUfileHandle_t handle{};
    bool handle_registered = false;
    Mode mode              = Mode::Compatibility;
    std::vector<void*> registered;
    std::vector<Batch> batches;
    std::deque<Part> queued;
    std::deque<Part> urgent_queued; // issued first
    // Host memory moves by pread/pwrite on these threads: cuFile 1.19's compatibility mode
    // faults now and then inside cuFileBatchIOGetStatus after batch reads into host memory
    // (pinned or pageable, registered or not), and GPUDirect Storage has no direct path for host
    // memory to offer.
    struct HostDone {
        Part part;
        bool ok = true;
        std::string error;
    };
    std::vector<std::thread> host_threads;
    std::mutex host_mutex;
    std::condition_variable host_wake;
    std::deque<Part> host_queued;      // guarded by host_mutex; its first host_urgent are urgent
    std::size_t host_urgent = 0;       // guarded by host_mutex
    std::vector<HostDone> host_done;   // guarded by host_mutex
    std::uint32_t host_outstanding = 0; // submitted to the threads, not yet reaped
    bool host_stopping             = false;
    std::unordered_map<std::uint64_t, Parent> parents;
    std::uint64_t next_parent = 0;
    std::vector<Completion> completed;
    std::vector<Part> host_parts; // submit's scratch

    void finish(const Part& part, bool ok, const std::string& error) {
        Parent& parent = parents.at(part.parent);
        if (!ok && parent.ok) {
            parent.ok    = false;
            parent.error = error;
        }
        if (--parent.remaining != 0) { return; }
        completed.push_back(Completion{parent.tag, parent.ok, parent.error});
        parents.erase(part.parent);
    }

    void reap(Batch& batch) {
        unsigned count = static_cast<unsigned>(batch.events.size());
        timespec now{0, 0};
        const CUfileError_t status =
            cuFileBatchIOGetStatus(batch.handle, 0, &count, batch.events.data(), &now);
        if (status.err != CU_FILE_SUCCESS) {
            // The batch's state is unknown: cancel it and fail what it still holds.
            (void)cuFileBatchIOCancel(batch.handle);
            const std::string error = cufile_error("cuFile batch status", status);
            for (const Part& part : batch.parts) { finish(part, false, error); }
            batch.parts.clear();
            batch.outstanding = 0;
            return;
        }
        for (unsigned i = 0; i < count; ++i) {
            const CUfileIOEvents_t& event = batch.events[i];
            if (!final_status(event.status)) { continue; }
            const auto index = static_cast<std::size_t>(reinterpret_cast<std::uintptr_t>(event.cookie));
            const Part& part = batch.parts.at(index);
            const auto done  = static_cast<std::int64_t>(event.ret);
            if (event.status == CUFILE_COMPLETE && done == static_cast<std::int64_t>(part.bytes)) {
                finish(part, true, {});
            } else if (event.status == CUFILE_COMPLETE) {
                finish(part, false,
                       done < 0 ? "cuFile transfer failed: " + std::string(CUFILE_ERRSTR(static_cast<int>(-done)))
                                : "cuFile short transfer of " + std::to_string(done) + " of " +
                                      std::to_string(part.bytes) + " bytes");
            } else {
                finish(part, false, "cuFile transfer ended with status " + std::to_string(static_cast<int>(event.status)));
            }
            --batch.outstanding;
        }
        if (batch.outstanding == 0) { batch.parts.clear(); }
    }

    void issue(Batch& batch) {
        batch.parts.clear();
        for (std::deque<Part>* from : {&urgent_queued, &queued}) {
            while (!from->empty() && batch.parts.size() < options.batch_depth) {
                batch.parts.push_back(from->front());
                from->pop_front();
            }
        }
        const std::size_t count = batch.parts.size();
        batch.params.resize(count);
        for (std::size_t i = 0; i < count; ++i) {
            const Part& part              = batch.parts[i];
            CUfileIOParams_t& params      = batch.params[i];
            params                        = CUfileIOParams_t{};
            params.mode                   = CUFILE_BATCH;
            params.u.batch.devPtr_base   = part.buffer;
            params.u.batch.file_offset   = static_cast<off_t>(part.offset);
            params.u.batch.devPtr_offset = static_cast<off_t>(part.buffer_offset);
            params.u.batch.size           = part.bytes;
            params.fh                     = handle;
            params.opcode                 = part.op == Op::Read ? CUFILE_READ : CUFILE_WRITE;
            params.cookie                 = reinterpret_cast<void*>(static_cast<std::uintptr_t>(i));
        }
        const CUfileError_t status =
            cuFileBatchIOSubmit(batch.handle, static_cast<unsigned>(count), batch.params.data(), 0);
        if (status.err != CU_FILE_SUCCESS) {
            const std::string error = cufile_error("cuFile batch submit", status);
            for (const Part& part : batch.parts) { finish(part, false, error); }
            batch.parts.clear();
            return;
        }
        batch.outstanding = static_cast<std::uint32_t>(count);
    }

    void host_main() noexcept {
        for (;;) {
            Part part;
            {
                std::unique_lock lock(host_mutex);
                host_wake.wait(lock, [&] { return host_stopping || !host_queued.empty(); });
                if (host_queued.empty()) { return; }
                part = host_queued.front();
                host_queued.pop_front();
                if (host_urgent != 0) { --host_urgent; }
            }
            HostDone done{part, true, {}};
            auto* memory     = static_cast<std::byte*>(part.buffer) + part.buffer_offset;
            std::size_t moved = 0;
            while (moved < part.bytes) {
                const auto offset = static_cast<off_t>(part.offset + moved);
                const ssize_t n   = part.op == Op::Read ? ::pread(fd, memory + moved, part.bytes - moved, offset)
                                                        : ::pwrite(fd, memory + moved, part.bytes - moved, offset);
                if (n < 0 && errno == EINTR) { continue; }
                if (n <= 0) {
                    done.ok    = false;
                    done.error = n < 0 ? std::string("Disk tier transfer failed: ") + std::strerror(errno)
                                       : "Disk tier short transfer of " + std::to_string(moved) + " of " +
                                             std::to_string(part.bytes) + " bytes";
                    break;
                }
                moved += static_cast<std::size_t>(n);
            }
            std::lock_guard lock(host_mutex);
            host_done.push_back(std::move(done));
        }
    }

    void reap_host() {
        std::vector<HostDone> done;
        {
            std::lock_guard lock(host_mutex);
            done.swap(host_done);
        }
        for (const HostDone& d : done) { finish(d.part, d.ok, d.error); }
        host_outstanding -= static_cast<std::uint32_t>(done.size());
    }

    void progress() {
        if (host_outstanding != 0) { reap_host(); }
        for (Batch& batch : batches) {
            if (batch.outstanding != 0) { reap(batch); }
        }
        for (Batch& batch : batches) {
            if (queued.empty() && urgent_queued.empty()) { break; }
            if (batch.outstanding == 0) { issue(batch); }
        }
    }

    [[nodiscard]] bool in_flight() const noexcept {
        if (host_outstanding != 0) { return true; }
        for (const Batch& batch : batches) {
            if (batch.outstanding != 0) { return true; }
        }
        return false;
    }

    void close() noexcept {
        while (in_flight()) {
            progress();
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
        for (Batch& batch : batches) {
            if (batch.handle != nullptr) { cuFileBatchIODestroy(batch.handle); }
        }
        batches.clear();
        {
            std::lock_guard lock(host_mutex);
            host_stopping = true;
        }
        host_wake.notify_all();
        for (std::thread& thread : host_threads) { thread.join(); }
        host_threads.clear();
        for (void* base : registered) { (void)cuFileBufDeregister(base); }
        registered.clear();
        if (handle_registered) { cuFileHandleDeregister(handle); }
        handle_registered = false;
        if (fd >= 0) { ::close(fd); }
        fd = -1;
    }
};

CuFileQueue::CuFileQueue(const std::filesystem::path& path, std::uint64_t bytes, Options options)
    : impl_(std::make_unique<Impl>()) {
    if (bytes == 0 || bytes % kAlignment != 0 || options.batch_depth == 0 || options.batches == 0 ||
        options.host_threads == 0 || options.max_part == 0 || options.max_part % kAlignment != 0) {
        throw std::invalid_argument("cuFile queue needs a 4 KiB-aligned size and positive depths");
    }
    Impl& impl   = *impl_;
    impl.options = options;
    impl.bytes   = bytes;
    try {
        impl.fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC | O_DIRECT, 0644);
        if (impl.fd < 0) {
            throw std::system_error(errno, std::generic_category(), "cannot open " + path.string());
        }
        struct stat info {};
        if (::fstat(impl.fd, &info) != 0) {
            throw std::system_error(errno, std::generic_category(), "cannot stat " + path.string());
        }
        if (static_cast<std::uint64_t>(info.st_size) != bytes &&
            ::ftruncate(impl.fd, static_cast<off_t>(bytes)) != 0) {
            throw std::system_error(errno, std::generic_category(), "cannot size " + path.string());
        }
        if (const int error = ::posix_fallocate(impl.fd, 0, static_cast<off_t>(bytes)); error != 0) {
            throw std::system_error(error, std::generic_category(), "cannot allocate " + path.string());
        }
        open_driver();
        CUfileDrvProps_t properties{};
        check(cuFileDriverGetProperties(&properties), "cuFile driver properties");
        impl.mode = (properties.nvfs.dstatusflags & (1U << CU_FILE_NVME_SUPPORTED)) != 0 ? Mode::Direct
                                                                                          : Mode::Compatibility;
        CUfileDescr_t descriptor{};
        descriptor.type      = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
        descriptor.handle.fd = impl.fd;
        check(cuFileHandleRegister(&impl.handle, &descriptor), "cuFile handle registration");
        impl.handle_registered = true;
        impl.batches.resize(options.batches);
        for (Impl::Batch& batch : impl.batches) {
            check(cuFileBatchIOSetUp(&batch.handle, options.batch_depth), "cuFile batch setup");
            batch.events.resize(options.batch_depth);
        }
        for (std::uint32_t i = 0; i < options.host_threads; ++i) {
            impl.host_threads.emplace_back([&impl] { impl.host_main(); });
        }
    } catch (...) {
        impl.close();
        throw;
    }
}

CuFileQueue::~CuFileQueue() { impl_->close(); }

CuFileQueue::Mode CuFileQueue::mode() const noexcept { return impl_->mode; }

std::uint64_t CuFileQueue::file_bytes() const noexcept { return impl_->bytes; }

bool CuFileQueue::register_buffer(void* base, std::size_t bytes) {
    if (base == nullptr || bytes == 0) { throw std::invalid_argument("cuFile buffer is empty"); }
    cudaPointerAttributes attributes{};
    const bool device = cudaPointerGetAttributes(&attributes, base) == cudaSuccess &&
                        attributes.type == cudaMemoryTypeDevice;
    (void)cudaGetLastError();
    if (!device) { return true; } // host memory never goes through cuFile
    if (cuFileBufRegister(base, bytes, 0).err != CU_FILE_SUCCESS) { return false; }
    impl_->registered.push_back(base);
    return true;
}

void CuFileQueue::submit(const Transfer& transfer) {
    Impl& impl = *impl_;
    if (transfer.bytes == 0 || transfer.buffer == nullptr || transfer.offset % kAlignment != 0 ||
        transfer.bytes % kAlignment != 0 || transfer.buffer_offset % kAlignment != 0 ||
        transfer.offset > impl.bytes || transfer.bytes > impl.bytes - transfer.offset) {
        throw std::invalid_argument("cuFile transfer is empty, misaligned or outside the file");
    }
    cudaPointerAttributes attributes{};
    const bool device = cudaPointerGetAttributes(&attributes, transfer.buffer) == cudaSuccess &&
                        attributes.type == cudaMemoryTypeDevice;
    (void)cudaGetLastError();
    const std::uint64_t parent = impl.next_parent++;
    const auto parts = static_cast<std::uint32_t>((transfer.bytes + impl.options.max_part - 1) / impl.options.max_part);
    impl.parents.emplace(parent, Impl::Parent{transfer.tag, parts, true, {}});
    for (std::size_t done = 0; done < transfer.bytes; done += impl.options.max_part) {
        const Impl::Part part{
            .parent        = parent,
            .op            = transfer.op,
            .offset        = transfer.offset + done,
            .bytes         = std::min(impl.options.max_part, transfer.bytes - done),
            .buffer        = transfer.buffer,
            .buffer_offset = transfer.buffer_offset + done,
            .device        = device,
        };
        if (device) {
            (transfer.urgent ? impl.urgent_queued : impl.queued).push_back(part);
        } else {
            impl.host_parts.push_back(part);
        }
    }
    if (!impl.host_parts.empty()) {
        {
            // An urgent transfer goes behind the urgent ones already queued, before the others.
            std::lock_guard lock(impl.host_mutex);
            const auto at = transfer.urgent ? impl.host_queued.begin() + static_cast<std::ptrdiff_t>(impl.host_urgent)
                                            : impl.host_queued.end();
            impl.host_queued.insert(at, impl.host_parts.begin(), impl.host_parts.end());
            if (transfer.urgent) { impl.host_urgent += impl.host_parts.size(); }
        }
        impl.host_outstanding += static_cast<std::uint32_t>(impl.host_parts.size());
        impl.host_parts.clear();
        impl.host_wake.notify_all();
    }
}

std::size_t CuFileQueue::poll(std::vector<Completion>& out) {
    Impl& impl = *impl_;
    impl.progress();
    const std::size_t count = impl.completed.size();
    out.insert(out.end(), std::make_move_iterator(impl.completed.begin()),
               std::make_move_iterator(impl.completed.end()));
    impl.completed.clear();
    return count;
}

void CuFileQueue::wait(std::vector<Completion>& out) {
    while (pending() != 0) {
        if (poll(out) == 0 && pending() != 0) { std::this_thread::sleep_for(std::chrono::microseconds(50)); }
    }
}

std::size_t CuFileQueue::pending() const noexcept { return impl_->parents.size() + impl_->completed.size(); }

void CuFileQueue::sync_data() const {
    if (::fdatasync(impl_->fd) != 0) {
        throw std::system_error(errno, std::generic_category(), "cannot sync the Disk tier file");
    }
}

const char* cufile_mode_name(CuFileQueue::Mode mode) noexcept {
    return mode == CuFileQueue::Mode::Direct ? "GDS" : "compat";
}

} // namespace infernix
