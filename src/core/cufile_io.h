#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace infernix {

// Asynchronous unbuffered reads and writes of one file through cuFile batch I/O, for the prefix
// cache's Disk tier. GPUDirect Storage moves the bytes between the file and device memory when
// the file's device is served directly (the nvidia-fs driver, or PCI P2PDMA); otherwise cuFile's
// compatibility mode does POSIX I/O through its own pinned bounce buffers. Buffers may be device
// or host memory; a registered device buffer (register_buffer) is used without a bounce where
// possible. Host memory moves by pread/pwrite on the queue's own threads, not through cuFile.
//
// On the T710 in compatibility mode, batch reads at depth 4 reached 10.5 GiB/s from one thread
// while the stream-ordered API (cuFileReadAsync) reached 2.9-3.6 GiB/s, so transfers are batched
// and polled rather than stream-ordered.
//
// submit, poll, wait and register_buffer are used by one thread. sync_data may run on any thread.
// Linux only: cuFile has no Windows build.
class CuFileQueue {
public:
    static constexpr std::size_t kAlignment = 4096; // file offsets, sizes and buffer offsets

    enum class Mode : std::uint8_t {
        Direct,        // GPUDirect Storage serves the file's device
        Compatibility, // POSIX I/O through cuFile's bounce buffers
    };
    enum class Op : std::uint8_t { Read, Write };

    struct Options {
        // Transfers per cuFile batch and batches in flight.
        std::uint32_t batch_depth = 16;
        std::uint32_t batches     = 4;
        // Larger transfers are issued as parts of at most this many bytes (cuFile's default
        // max_direct_io_size).
        std::size_t max_part = 16U << 20U;
        // Threads moving host memory with pread/pwrite.
        std::uint32_t host_threads = 4;
    };

    // `buffer + buffer_offset` is the transfer's memory; `buffer` is the registered base when it
    // was registered.
    struct Transfer {
        Op op                     = Op::Read;
        std::uint64_t offset      = 0;
        std::size_t bytes         = 0;
        void* buffer              = nullptr;
        std::size_t buffer_offset = 0;
        std::uint64_t tag         = 0; // returned with the completion
        bool urgent               = false; // issued before every queued transfer that is not
    };

    struct Completion {
        std::uint64_t tag = 0;
        bool ok           = false;
        std::string error; // set unless ok
    };

    // Opens the file at `path` (created when missing) for unbuffered I/O and sets its size to
    // `bytes`, allocating every block. Throws std::system_error or std::runtime_error.
    CuFileQueue(const std::filesystem::path& path, std::uint64_t bytes, Options options);
    CuFileQueue(const std::filesystem::path& path, std::uint64_t bytes)
        : CuFileQueue(path, bytes, Options{}) {}
    ~CuFileQueue(); // waits for transfers in flight

    CuFileQueue(const CuFileQueue&)            = delete;
    CuFileQueue& operator=(const CuFileQueue&) = delete;

    [[nodiscard]] Mode mode() const noexcept;
    [[nodiscard]] std::uint64_t file_bytes() const noexcept;

    // Registers device memory for cuFile I/O until destruction (host memory needs none). False
    // when cuFile refuses (the memory is then still usable, through bounce buffers).
    bool register_buffer(void* base, std::size_t bytes);

    // Queues a transfer; throws std::invalid_argument for an empty, misaligned or out-of-file
    // range.
    void submit(const Transfer& transfer);
    // Issues queued transfers and appends every transfer completed since the last call, without
    // waiting. Returns the number appended.
    std::size_t poll(std::vector<Completion>& out);
    // Polls until every submitted transfer has completed.
    void wait(std::vector<Completion>& out);
    [[nodiscard]] std::size_t pending() const noexcept; // transfers not yet returned by poll

    // Makes every write completed so far durable (fdatasync). Throws std::system_error.
    void sync_data() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

[[nodiscard]] const char* cufile_mode_name(CuFileQueue::Mode mode) noexcept;

} // namespace infernix
