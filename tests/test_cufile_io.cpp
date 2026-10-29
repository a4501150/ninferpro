// Core CuFileQueue: unbuffered writes and reads of host memory (its POSIX threads) and device
// memory (cuFile batches) against a byte oracle, many host reads in flight at once, transfers split into parts, a reopened file keeping its bytes, and the argument
// checks. `INFERNIX_TEST_DISK_DIR` puts the file on a given filesystem (GPUDirect Storage needs a
// local NVMe one); otherwise it lives in the system temp directory (compatibility mode).
#include "test_temp_path.h"
#include "core/cufile_io.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using infernix::CuFileQueue;

constexpr std::size_t kBlock     = CuFileQueue::kAlignment;
constexpr std::uint64_t kFile    = 8ULL << 20U;
constexpr std::size_t kTransfer  = 1U << 20U;

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        ++failures;
        std::cerr << "FAIL: " << what << '\n';
    }
}

void cuda(cudaError_t status, const char* what) {
    if (status != cudaSuccess) { throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status)); }
}

std::uint8_t oracle(std::uint64_t at, std::uint32_t seed) {
    return static_cast<std::uint8_t>(((at + seed) * 2654435761ULL >> 11) ^ (at >> 17) ^ seed);
}

struct Aligned {
    std::byte* p      = nullptr;
    std::size_t bytes = 0;
    explicit Aligned(std::size_t n)
        : p(static_cast<std::byte*>(::operator new(n, std::align_val_t{kBlock}))), bytes(n) {}
    ~Aligned() { ::operator delete(p, std::align_val_t{kBlock}); }
    Aligned(const Aligned&)            = delete;
    Aligned& operator=(const Aligned&) = delete;
};

void fill(std::byte* data, std::size_t bytes, std::uint32_t seed) {
    for (std::size_t i = 0; i < bytes; ++i) { data[i] = static_cast<std::byte>(oracle(i, seed)); }
}

bool matches(const std::byte* data, std::size_t bytes, std::uint32_t seed) {
    for (std::size_t i = 0; i < bytes; ++i) {
        if (static_cast<std::uint8_t>(data[i]) != oracle(i, seed)) { return false; }
    }
    return true;
}

bool all_ok(const std::vector<CuFileQueue::Completion>& completions, std::size_t expected) {
    if (completions.size() != expected) { return false; }
    for (const auto& completion : completions) {
        if (!completion.ok) {
            std::cerr << "transfer " << completion.tag << ": " << completion.error << '\n';
            return false;
        }
    }
    return true;
}

std::filesystem::path test_file() {
    const char* dir = std::getenv("INFERNIX_TEST_DISK_DIR");
    if (dir != nullptr && *dir != '\0') {
        return std::filesystem::path(dir) /
               infernix::test::unique_temp_path("cufile-io.bin").filename();
    }
    return infernix::test::unique_temp_path("cufile-io.bin");
}

void test_round_trips(const std::filesystem::path& path) {
    CuFileQueue::Options options;
    options.batch_depth = 4;
    options.batches     = 2;
    options.max_part    = 64U << 10U; // a 1 MiB transfer is 16 parts
    CuFileQueue queue(path, kFile, options);
    check(std::filesystem::file_size(path) == kFile, "the file is sized");
    std::cout << "cuFile mode: " << infernix::cufile_mode_name(queue.mode()) << '\n';

    // Host memory: two writes, read back into one buffer at an offset.
    Aligned host(2 * kTransfer);
    fill(host.p, kTransfer, 1);
    fill(host.p + kTransfer, kTransfer, 2);
    (void)queue.register_buffer(host.p, host.bytes);
    queue.submit({.op = CuFileQueue::Op::Write, .offset = 0, .bytes = kTransfer, .buffer = host.p, .tag = 1});
    queue.submit({.op            = CuFileQueue::Op::Write,
                  .offset        = 3 * kTransfer,
                  .bytes         = kTransfer,
                  .buffer        = host.p,
                  .buffer_offset = kTransfer,
                  .tag           = 2});
    check(queue.pending() == 2, "two transfers are pending");
    std::vector<CuFileQueue::Completion> done;
    queue.wait(done);
    check(all_ok(done, 2), "host writes complete, one completion per transfer");
    queue.sync_data();

    Aligned back(2 * kTransfer);
    queue.submit({.op = CuFileQueue::Op::Read, .offset = 3 * kTransfer, .bytes = kTransfer, .buffer = back.p, .tag = 3});
    queue.submit({.op            = CuFileQueue::Op::Read,
                  .offset        = 0,
                  .bytes         = kTransfer,
                  .buffer        = back.p,
                  .buffer_offset = kTransfer,
                  .tag           = 4});
    done.clear();
    queue.wait(done);
    check(all_ok(done, 2), "host reads complete");
    check(matches(back.p, kTransfer, 2) && matches(back.p + kTransfer, kTransfer, 1), "host reads return the written bytes");

    // Device memory: write from the Device, read into the Device and the Host.
    void* device = nullptr;
    cuda(cudaMalloc(&device, 2 * kTransfer), "cudaMalloc");
    std::vector<std::byte> staged(kTransfer);
    fill(staged.data(), kTransfer, 3);
    cuda(cudaMemcpy(device, staged.data(), kTransfer, cudaMemcpyHostToDevice), "upload");
    (void)queue.register_buffer(device, 2 * kTransfer);
    queue.submit({.op = CuFileQueue::Op::Write, .offset = 5 * kTransfer, .bytes = kTransfer, .buffer = device, .tag = 5});
    done.clear();
    queue.wait(done);
    check(all_ok(done, 1), "a device write completes");
    queue.submit({.op            = CuFileQueue::Op::Read,
                  .offset        = 0,
                  .bytes         = kTransfer,
                  .buffer        = device,
                  .buffer_offset = kTransfer,
                  .tag           = 6});
    queue.submit({.op = CuFileQueue::Op::Read, .offset = 5 * kTransfer, .bytes = kTransfer, .buffer = back.p, .tag = 7});
    done.clear();
    queue.wait(done);
    check(all_ok(done, 2), "device and host reads complete");
    check(matches(back.p, kTransfer, 3), "a device write reads back on the Host");
    cuda(cudaMemcpy(staged.data(), static_cast<std::byte*>(device) + kTransfer, kTransfer, cudaMemcpyDeviceToHost),
         "download");
    check(matches(staged.data(), kTransfer, 1), "a device read returns the written bytes");

    // Argument checks.
    const auto throws = [&](const CuFileQueue::Transfer& transfer) {
        try {
            queue.submit(transfer);
        } catch (const std::invalid_argument&) { return true; }
        return false;
    };
    check(throws({.op = CuFileQueue::Op::Read, .offset = 1, .bytes = kBlock, .buffer = back.p}), "a misaligned offset");
    check(throws({.op = CuFileQueue::Op::Read, .offset = 0, .bytes = 100, .buffer = back.p}), "a misaligned size");
    check(throws({.op = CuFileQueue::Op::Read, .offset = kFile, .bytes = kBlock, .buffer = back.p}), "past the file");
    check(throws({.op = CuFileQueue::Op::Read, .offset = 0, .bytes = 0, .buffer = back.p}), "an empty transfer");
    check(throws({.op = CuFileQueue::Op::Read, .offset = 0, .bytes = kBlock, .buffer = nullptr}), "no buffer");
    check(queue.pending() == 0, "refused transfers are not queued");
    cuda(cudaFree(device), "cudaFree");
}

void test_many_in_flight(const std::filesystem::path& path) {
    // Rounds of host and device reads, many in flight at once (cuFile 1.19's compatibility mode
    // faulted in its batch status call on such host reads).
    constexpr std::size_t kPiece = 64U << 10U;
    constexpr std::size_t kPieces = kFile / kPiece;
    CuFileQueue queue(path, kFile);
    Aligned source(kFile);
    fill(source.p, kFile, 9);
    std::vector<CuFileQueue::Completion> done;
    queue.submit({.op = CuFileQueue::Op::Write, .offset = 0, .bytes = kFile, .buffer = source.p, .tag = 0});
    queue.wait(done);
    check(all_ok(done, 1), "the file is written");
    void* device = nullptr;
    cuda(cudaMalloc(&device, kFile), "cudaMalloc");
    (void)queue.register_buffer(device, kFile);
    Aligned back(kFile);
    bool same = true;
    for (int round = 0; round < 20 && same; ++round) {
        std::memset(back.p, 0, kFile);
        for (std::size_t i = 0; i < kPieces; ++i) {
            // Piece i lands at slot (i * 7) % kPieces, so a misplaced read shows.
            const std::size_t at = (i * 7U) % kPieces * kPiece;
            queue.submit({.op = CuFileQueue::Op::Read, .offset = i * kPiece, .bytes = kPiece, .buffer = back.p,
                          .buffer_offset = at, .tag = i});
            queue.submit({.op = CuFileQueue::Op::Read, .offset = i * kPiece, .bytes = kPiece, .buffer = device,
                          .buffer_offset = at, .tag = kPieces + i});
        }
        done.clear();
        queue.wait(done);
        same = all_ok(done, 2 * kPieces);
        for (std::size_t i = 0; same && i < kPieces; ++i) {
            same = std::memcmp(back.p + (i * 7U) % kPieces * kPiece, source.p + i * kPiece, kPiece) == 0;
        }
    }
    check(same, "host reads in flight together land at their offsets");
    std::vector<std::byte> staged(kFile);
    cuda(cudaMemcpy(staged.data(), device, kFile, cudaMemcpyDeviceToHost), "download");
    same = true;
    for (std::size_t i = 0; same && i < kPieces; ++i) {
        same = std::memcmp(staged.data() + (i * 7U) % kPieces * kPiece, source.p + i * kPiece, kPiece) == 0;
    }
    check(same, "device reads in flight together land at their offsets");
    cuda(cudaFree(device), "cudaFree");
}

void test_reopen(const std::filesystem::path& path) {
    // A reopened file of the same size keeps its bytes.
    CuFileQueue queue(path, kFile);
    Aligned back(kTransfer);
    queue.submit({.op = CuFileQueue::Op::Read, .offset = 3 * kTransfer, .bytes = kTransfer, .buffer = back.p, .tag = 1});
    std::vector<CuFileQueue::Completion> done;
    queue.wait(done);
    check(all_ok(done, 1) && matches(back.p, kTransfer, 2), "a reopened file keeps its bytes");
}

} // namespace

int main() {
    const std::filesystem::path path = test_file();
    try {
        test_round_trips(path);
        test_reopen(path);
        test_many_in_flight(path);
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        ++failures;
    }
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
    if (failures != 0) { return 1; }
    std::cout << "cuFile queue tests passed\n";
    return 0;
}
