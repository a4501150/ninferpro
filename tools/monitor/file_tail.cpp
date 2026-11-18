#include "file_tail.h"

#include <sys/stat.h>

#include <algorithm>

namespace infernix::monitor {

FileTail::FileTail(std::filesystem::path path, std::uint64_t backlog_bytes)
    : path_(std::move(path)), backlog_bytes_(backlog_bytes) {}

std::vector<std::string> FileTail::read_lines() {
    reopened_ = false;
    std::vector<std::string> lines;
    struct stat info {};
    if (::stat(path_.c_str(), &info) != 0) { return lines; } // follows a symlink to its target
    const auto size = static_cast<std::uint64_t>(info.st_size);
    const bool replaced = !file_.is_open() || static_cast<std::uint64_t>(info.st_dev) != device_ ||
                          static_cast<std::uint64_t>(info.st_ino) != inode_;
    if (replaced || size < offset_) {
        file_.close();
        file_.clear();
        file_.open(path_, std::ios::binary);
        if (!file_) { return lines; }
        device_  = static_cast<std::uint64_t>(info.st_dev);
        inode_   = static_cast<std::uint64_t>(info.st_ino);
        partial_.clear();
        offset_ = 0;
        if (!opened_once_ && size > backlog_bytes_) {
            offset_ = size - backlog_bytes_;
            // Drop the line the backlog starts inside.
            file_.seekg(static_cast<std::streamoff>(offset_));
            std::string skipped;
            std::getline(file_, skipped);
            if (!file_) { return lines; }
            offset_ += skipped.size() + 1;
        }
        reopened_    = opened_once_;
        opened_once_ = true;
    }
    if (size <= offset_) { return lines; }
    std::string chunk(static_cast<std::size_t>(size - offset_), '\0');
    file_.clear();
    file_.seekg(static_cast<std::streamoff>(offset_));
    file_.read(chunk.data(), static_cast<std::streamsize>(chunk.size()));
    chunk.resize(static_cast<std::size_t>(std::max<std::streamsize>(file_.gcount(), 0)));
    offset_ += chunk.size();
    std::size_t start = 0;
    while (true) {
        const std::size_t newline = chunk.find('\n', start);
        if (newline == std::string::npos) {
            partial_.append(chunk, start, std::string::npos);
            break;
        }
        partial_.append(chunk, start, newline - start);
        if (!partial_.empty() && partial_.back() == '\r') { partial_.pop_back(); }
        lines.push_back(std::move(partial_));
        partial_.clear();
        start = newline + 1;
    }
    return lines;
}

} // namespace infernix::monitor
