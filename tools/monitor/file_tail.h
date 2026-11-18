#pragma once

// Follows a growing text file by path, as `tail -F` does: a replaced file (log rotation, or a
// symlink pointed at a new log) or a truncated one is read again from its start.

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace infernix::monitor {

class FileTail {
public:
    // On first open, only the file's last `backlog_bytes` are read; the partial line they start
    // inside is dropped.
    FileTail(std::filesystem::path path, std::uint64_t backlog_bytes);

    // The complete lines appended since the last call. A line still being written stays buffered
    // until its newline arrives.
    std::vector<std::string> read_lines();

    // True when the last read_lines() call opened a different file than the one before it.
    bool reopened() const noexcept { return reopened_; }
    const std::filesystem::path& path() const noexcept { return path_; }

private:
    std::filesystem::path path_;
    std::uint64_t backlog_bytes_;
    std::ifstream file_;
    std::uint64_t device_ = 0, inode_ = 0, offset_ = 0;
    bool opened_once_ = false, reopened_ = false;
    std::string partial_;
};

} // namespace infernix::monitor
