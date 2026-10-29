#include "core/cufile_io.h"

#include <stdexcept>

namespace infernix {

// cuFile has no Windows build: the Disk tier is refused before a queue is made.

struct CuFileQueue::Impl {};

CuFileQueue::CuFileQueue(const std::filesystem::path&, std::uint64_t, Options) {
    throw std::runtime_error("cuFile is not available on this platform");
}
CuFileQueue::~CuFileQueue() = default;
CuFileQueue::Mode CuFileQueue::mode() const noexcept { return Mode::Compatibility; }
std::uint64_t CuFileQueue::file_bytes() const noexcept { return 0; }
bool CuFileQueue::register_buffer(void*, std::size_t) { return false; }
void CuFileQueue::submit(const Transfer&) { throw std::logic_error("cuFile is not available on this platform"); }
std::size_t CuFileQueue::poll(std::vector<Completion>&) { return 0; }
void CuFileQueue::wait(std::vector<Completion>&) {}
std::size_t CuFileQueue::pending() const noexcept { return 0; }
void CuFileQueue::sync_data() const {}

const char* cufile_mode_name(CuFileQueue::Mode mode) noexcept {
    return mode == CuFileQueue::Mode::Direct ? "GDS" : "compat";
}

} // namespace infernix
