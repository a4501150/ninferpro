#include "models/qwen4_exp/program/prefix/disk_store.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <map>
#include <stdexcept>
#include <system_error>
#include <unordered_map>
#include <utility>

#ifndef _WIN32
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace infernix::models::qwen4_exp::prefix {

namespace pc = infernix::models::qwen4_exp::prefix::cache_index;

#ifdef _WIN32

DiskStore::DiskStore(const std::filesystem::path&, std::string_view, const DiskGeometry&) {
    throw std::runtime_error("the prefix cache Disk tier needs cuFile, which has no Windows build");
}
DiskStore::~DiskStore() = default;
void DiskStore::compact(std::span<const DiskBlockRecord>, std::span<const DiskSnapshotRecord>) {}
void DiskStore::append_block(const DiskBlockRecord&) {}
void DiskStore::append_snapshot(const DiskSnapshotRecord&) {}
void DiskStore::append_removal(std::uint32_t, std::span<const std::uint32_t>) {}
void DiskStore::append_hits(std::uint32_t, std::uint32_t) {}
bool DiskStore::writable(std::span<const std::uint32_t>) const noexcept { return false; }
bool DiskStore::failed() const noexcept { return true; }
void DiskStore::flush() {}

#else

namespace {

constexpr std::array<char, 8> kManifestMagic = {'N', 'I', 'N', 'F', 'Q', '4', 'D', 'K'};
constexpr std::uint32_t kManifestVersion     = 1;

enum class RecordType : std::uint8_t { Block = 1, Snapshot = 2, Removal = 3, Hits = 4 };

std::uint32_t crc32(std::span<const std::byte> bytes) noexcept {
    static const std::array<std::uint32_t, 256> table = [] {
        std::array<std::uint32_t, 256> out{};
        for (std::uint32_t i = 0; i < 256; ++i) {
            std::uint32_t c = i;
            for (int k = 0; k < 8; ++k) { c = (c & 1U) != 0 ? 0xEDB88320U ^ (c >> 1U) : c >> 1U; }
            out[i] = c;
        }
        return out;
    }();
    std::uint32_t c = 0xFFFFFFFFU;
    for (const std::byte b : bytes) { c = table[(c ^ static_cast<std::uint8_t>(b)) & 0xFFU] ^ (c >> 8U); }
    return c ^ 0xFFFFFFFFU;
}

class Writer {
public:
    template <class T>
    void value(const T& v) {
        const auto* p = reinterpret_cast<const std::byte*>(&v);
        bytes_.insert(bytes_.end(), p, p + sizeof(T));
    }
    template <class T>
    void array(std::span<const T> values) {
        for (const T& v : values) { value(v); }
    }
    [[nodiscard]] std::vector<std::byte>& bytes() noexcept { return bytes_; }

private:
    std::vector<std::byte> bytes_;
};

class Reader {
public:
    explicit Reader(std::span<const std::byte> bytes) : bytes_(bytes) {}
    template <class T>
    [[nodiscard]] bool value(T& out) {
        if (bytes_.size() - at_ < sizeof(T)) { return false; }
        std::memcpy(&out, bytes_.data() + at_, sizeof(T));
        at_ += sizeof(T);
        return true;
    }
    [[nodiscard]] bool done() const noexcept { return at_ == bytes_.size(); }

private:
    std::span<const std::byte> bytes_;
    std::size_t at_ = 0;
};

std::vector<std::byte> frame(RecordType type, Writer&& payload) {
    std::vector<std::byte>& body = payload.bytes();
    Writer out;
    out.value(static_cast<std::uint8_t>(type));
    out.value(static_cast<std::uint32_t>(body.size()));
    out.bytes().insert(out.bytes().end(), body.begin(), body.end());
    out.value(crc32(out.bytes()));
    return std::move(out.bytes());
}

void write_meta(Writer& out, const StateImageHeader& meta) {
    out.value(meta.frontier);
    out.value(static_cast<std::uint8_t>(meta.mtp_written));
    out.value(meta.mtp_next);
    out.value(static_cast<std::uint8_t>(meta.lineage_echo));
    out.value(static_cast<std::uint8_t>(meta.opener));
    out.value(static_cast<std::uint8_t>(meta.head_residual));
    out.array(std::span<const double>(meta.mtp_accept));
    out.value(meta.draft_frontier);
}

bool read_flag(Reader& in, bool& out) {
    std::uint8_t byte = 0;
    if (!in.value(byte) || byte > 1U) { return false; }
    out = byte != 0;
    return true;
}

bool read_meta(Reader& in, StateImageHeader& meta) {
    if (!in.value(meta.frontier) || !read_flag(in, meta.mtp_written) || !in.value(meta.mtp_next) ||
        !read_flag(in, meta.lineage_echo) || !read_flag(in, meta.opener) || !read_flag(in, meta.head_residual)) {
        return false;
    }
    for (double& accept : meta.mtp_accept) {
        if (!in.value(accept)) { return false; }
    }
    return in.value(meta.draft_frontier);
}

std::vector<std::byte> encode_block(const DiskBlockRecord& record) {
    Writer out;
    out.value(record.slot);
    out.value(record.parent_slot);
    out.value(record.lookup_hash);
    out.value(record.extra);
    out.value(record.mtp_next);
    out.array(std::span<const TokenId>(record.tokens));
    return frame(RecordType::Block, std::move(out));
}

std::vector<std::byte> encode_snapshot(const DiskSnapshotRecord& record) {
    Writer out;
    out.value(static_cast<std::uint32_t>(record.slots.size()));
    out.array(std::span<const std::uint32_t>(record.slots));
    out.value(record.anchor_slot);
    out.value(record.frontier);
    out.value(static_cast<std::uint32_t>(record.tail.size()));
    out.array(std::span<const TokenId>(record.tail));
    out.value(static_cast<std::uint8_t>(record.kind));
    out.value(record.hits);
    write_meta(out, record.meta);
    return frame(RecordType::Snapshot, std::move(out));
}

void write_all(int fd, std::span<const std::byte> bytes, const char* what) {
    while (!bytes.empty()) {
        const ssize_t written = ::write(fd, bytes.data(), bytes.size());
        if (written < 0) {
            if (errno == EINTR) { continue; }
            throw std::system_error(errno, std::generic_category(), what);
        }
        bytes = bytes.subspan(static_cast<std::size_t>(written));
    }
}

std::vector<std::byte> read_file(const std::filesystem::path& path) {
    std::vector<std::byte> out;
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) { return out; }
    std::byte buffer[1 << 16];
    for (;;) {
        const ssize_t got = ::read(fd, buffer, sizeof(buffer));
        if (got < 0 && errno == EINTR) { continue; }
        if (got <= 0) { break; }
        out.insert(out.end(), buffer, buffer + got);
    }
    ::close(fd);
    return out;
}

void sync_directory(const std::filesystem::path& directory) {
    const int fd = ::open(directory.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (fd < 0) { return; }
    (void)::fsync(fd);
    ::close(fd);
}

// Writes `bytes` to `path` through a temporary file, durable once this returns.
void replace_file(const std::filesystem::path& path, std::span<const std::byte> bytes) {
    std::filesystem::path temporary = path;
    temporary += ".tmp";
    const int fd = ::open(temporary.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (fd < 0) { throw std::system_error(errno, std::generic_category(), "cannot create " + temporary.string()); }
    try {
        write_all(fd, bytes, "cannot write the prefix cache Disk tier");
        if (::fdatasync(fd) != 0) {
            throw std::system_error(errno, std::generic_category(), "cannot sync " + temporary.string());
        }
    } catch (...) {
        ::close(fd);
        throw;
    }
    ::close(fd);
    std::filesystem::rename(temporary, path);
    sync_directory(path.parent_path());
}

std::vector<std::byte> encode_manifest(std::string_view fingerprint, const DiskGeometry& geometry) {
    Writer out;
    out.array(std::span<const char>(kManifestMagic));
    out.value(kManifestVersion);
    out.value(static_cast<std::uint32_t>(fingerprint.size()));
    out.array(std::span<const char>(fingerprint.data(), fingerprint.size()));
    out.value(geometry.slot_bytes);
    out.value(geometry.slots);
    out.value(geometry.image_slots);
    out.value(geometry.page_stride);
    out.value(geometry.image_fingerprint);
    return std::move(out.bytes());
}

} // namespace

DiskStore::DiskStore(const std::filesystem::path& directory, std::string_view fingerprint, const DiskGeometry& geometry)
    : directory_(directory), geometry_(geometry) {
    if (geometry.slots == 0 || geometry.slot_bytes == 0 || geometry.slot_bytes % CuFileQueue::kAlignment != 0 ||
        geometry.image_slots == 0) {
        throw std::invalid_argument("prefix cache Disk tier geometry is empty or misaligned");
    }
    std::filesystem::create_directories(directory);
    const std::filesystem::path lock = directory / "lock";
    lock_fd_                         = ::open(lock.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
    if (lock_fd_ < 0) { throw std::system_error(errno, std::generic_category(), "cannot open " + lock.string()); }
    if (::flock(lock_fd_, LOCK_EX | LOCK_NB) != 0) {
        const int error = errno;
        ::close(lock_fd_);
        lock_fd_ = -1;
        if (error == EWOULDBLOCK) {
            throw std::runtime_error("prefix cache Disk tier " + directory.string() + " is in use by another server");
        }
        throw std::system_error(error, std::generic_category(), "cannot lock " + lock.string());
    }
    try {
        const std::vector<std::byte> manifest = encode_manifest(fingerprint, geometry);
        const std::vector<std::byte> stored   = read_file(directory / "manifest");
        if (stored.empty()) {
            reset_reason_ = "new Disk tier";
        } else if (stored != manifest) {
            reset_reason_ = "the Disk tier was written for another model, build or size";
        }
        if (!reset_reason_.empty()) { replace_file(directory / "manifest", manifest); }
        io_ = std::make_unique<CuFileQueue>(directory / "slots.bin",
                                            static_cast<std::uint64_t>(geometry.slots) * geometry.slot_bytes);
        if (reset_reason_.empty()) { replay(read_file(directory / "catalog.log")); }
        freed_at_.assign(geometry.slots, 0);
    } catch (...) {
        ::close(lock_fd_);
        lock_fd_ = -1;
        throw;
    }
}

DiskStore::~DiskStore() {
    {
        std::lock_guard lock(mutex_);
        stopping_ = true;
    }
    wake_.notify_all();
    if (journal_.joinable()) { journal_.join(); }
    io_.reset();
    if (journal_fd_ >= 0) { ::close(journal_fd_); }
    if (lock_fd_ >= 0) { ::close(lock_fd_); }
}

void DiskStore::replay(std::span<const std::byte> journal) {
    // A block's parent and a snapshot's anchor are the block record at that slot when the record
    // is read: a slot freed and rewritten later holds another block, which the link must not reach.
    // Each record keeps the position (epoch) of the record it links to.
    struct Linked {
        std::uint64_t epoch = 0;
        std::uint64_t link  = 0; // the parent's or anchor's epoch; 0: root or unresolved
        bool resolved       = false;
    };
    std::map<std::uint32_t, std::pair<DiskBlockRecord, Linked>> blocks;
    std::map<std::uint32_t, std::pair<DiskSnapshotRecord, Linked>> snapshots;
    std::uint64_t epoch = 0;
    const auto link = [&](std::uint32_t slot) {
        Linked out{.epoch = ++epoch};
        if (slot == pc::kNoId) {
            out.resolved = true;
        } else if (const auto found = blocks.find(slot); found != blocks.end()) {
            out.link     = found->second.second.epoch;
            out.resolved = true;
        }
        return out;
    };
    std::size_t at = 0;
    while (at < journal.size()) {
        Reader head(journal.subspan(at));
        std::uint8_t type    = 0;
        std::uint32_t length = 0;
        if (!head.value(type) || !head.value(length) || journal.size() - at < 5U + length + 4U) {
            damaged_ = true;
            break;
        }
        std::uint32_t stored = 0;
        std::memcpy(&stored, journal.data() + at + 5U + length, sizeof(stored));
        if (crc32(journal.subspan(at, 5U + length)) != stored) {
            damaged_ = true;
            break;
        }
        Reader in(journal.subspan(at + 5U, length));
        at += 5U + length + 4U;
        bool ok = false;
        switch (static_cast<RecordType>(type)) {
        case RecordType::Block: {
            DiskBlockRecord record;
            ok = in.value(record.slot) && in.value(record.parent_slot) && in.value(record.lookup_hash) &&
                 in.value(record.extra) && in.value(record.mtp_next);
            for (TokenId& token : record.tokens) { ok = ok && in.value(token); }
            ok = ok && in.done() && record.slot < geometry_.slots;
            if (ok) {
                const Linked linked = link(record.parent_slot);
                blocks[record.slot] = {record, linked};
            }
            break;
        }
        case RecordType::Snapshot: {
            DiskSnapshotRecord record;
            std::uint32_t slots = 0, tail = 0;
            std::uint8_t kind = 0;
            ok = in.value(slots) && slots >= geometry_.image_slots && slots <= geometry_.image_slots + 1U;
            record.slots.resize(ok ? slots : 0U);
            for (std::uint32_t& slot : record.slots) { ok = ok && in.value(slot) && slot < geometry_.slots; }
            ok = ok && in.value(record.anchor_slot) && in.value(record.frontier) && in.value(tail) &&
                 tail < pc::kBlockTokens && (tail != 0) == (slots == geometry_.image_slots + 1U);
            record.tail.resize(ok ? tail : 0U);
            for (TokenId& token : record.tail) { ok = ok && in.value(token); }
            ok = ok && in.value(kind) && kind <= static_cast<std::uint8_t>(pc::SnapshotKind::Boundary) &&
                 in.value(record.hits) && read_meta(in, record.meta) && in.done();
            record.kind = static_cast<pc::SnapshotKind>(kind);
            if (ok) {
                const Linked linked       = link(record.anchor_slot);
                const std::uint32_t first = record.slots.front();
                snapshots[first]          = {std::move(record), linked};
            }
            break;
        }
        case RecordType::Removal: {
            std::uint32_t first = 0;
            ok = in.value(first) && in.done();
            if (ok) {
                blocks.erase(first);
                snapshots.erase(first);
            }
            break;
        }
        case RecordType::Hits: {
            std::uint32_t first = 0, hits = 0;
            ok = in.value(first) && in.value(hits) && in.done();
            if (ok) {
                if (const auto found = snapshots.find(first); found != snapshots.end()) {
                    found->second.first.hits = hits;
                }
            }
            break;
        }
        }
        if (!ok) {
            damaged_ = true;
            break;
        }
    }
    // Blocks parents first; a block whose parent record was not live when it was written cannot be
    // rebuilt, nor can what lies below it. Children are keyed by their parent's epoch (0: root).
    std::unordered_multimap<std::uint64_t, std::uint32_t> children;
    for (const auto& [slot, entry] : blocks) {
        if (entry.second.resolved) { children.emplace(entry.second.link, slot); }
    }
    std::vector<std::uint64_t> frontier{0};
    std::unordered_map<std::uint64_t, bool> kept; // by epoch
    while (!frontier.empty()) {
        const std::uint64_t parent = frontier.back();
        frontier.pop_back();
        const auto [begin, end] = children.equal_range(parent);
        for (auto it = begin; it != end; ++it) {
            const auto& [record, linked] = blocks.at(it->second);
            if (kept.contains(linked.epoch)) { continue; }
            kept[linked.epoch] = true;
            blocks_.push_back(record);
            frontier.push_back(linked.epoch);
        }
    }
    for (auto& [first, entry] : snapshots) {
        if (entry.second.resolved && (entry.second.link == 0 || kept.contains(entry.second.link))) {
            snapshots_.push_back(std::move(entry.first));
        }
    }
}

void DiskStore::open_journal(bool truncate) {
    const std::filesystem::path path = directory_ / "catalog.log";
    journal_fd_ = ::open(path.c_str(), O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC | (truncate ? O_TRUNC : 0), 0644);
    if (journal_fd_ < 0) { throw std::system_error(errno, std::generic_category(), "cannot open " + path.string()); }
}

void DiskStore::compact(std::span<const DiskBlockRecord> blocks, std::span<const DiskSnapshotRecord> snapshots) {
    if (journal_.joinable()) {
        // Every removal is durable before the journal is replaced, so every freed slot stays
        // writable; the thread restarts on the new journal.
        flush();
        {
            std::lock_guard lock(mutex_);
            stopping_ = true;
        }
        wake_.notify_all();
        journal_.join();
        stopping_ = false;
        ::close(journal_fd_);
        journal_fd_ = -1;
    }
    std::vector<std::byte> bytes;
    for (const DiskBlockRecord& record : blocks) {
        const std::vector<std::byte> encoded = encode_block(record);
        bytes.insert(bytes.end(), encoded.begin(), encoded.end());
    }
    for (const DiskSnapshotRecord& record : snapshots) {
        const std::vector<std::byte> encoded = encode_snapshot(record);
        bytes.insert(bytes.end(), encoded.begin(), encoded.end());
    }
    replace_file(directory_ / "catalog.log", bytes);
    blocks_.clear();
    snapshots_.clear();
    open_journal(false);
    journal_ = std::thread([this] { journal_main(); });
}

std::uint64_t DiskStore::enqueue(std::vector<std::byte>&& bytes, bool data) {
    if (!journal_.joinable()) { throw std::logic_error("prefix cache Disk tier appended before compaction"); }
    std::uint64_t sequence = 0;
    {
        std::lock_guard lock(mutex_);
        sequence = next_sequence_++;
        pending_.push_back(Pending{std::move(bytes), sequence, data});
    }
    wake_.notify_one();
    return sequence;
}

void DiskStore::append_block(const DiskBlockRecord& record) { (void)enqueue(encode_block(record), true); }

void DiskStore::append_snapshot(const DiskSnapshotRecord& record) { (void)enqueue(encode_snapshot(record), true); }

void DiskStore::append_removal(std::uint32_t first_slot, std::span<const std::uint32_t> slots) {
    Writer out;
    out.value(first_slot);
    const std::uint64_t sequence = enqueue(frame(RecordType::Removal, std::move(out)), false);
    for (const std::uint32_t slot : slots) {
        if (slot < freed_at_.size()) { freed_at_[slot] = sequence; }
    }
}

void DiskStore::append_hits(std::uint32_t first_slot, std::uint32_t hits) {
    Writer out;
    out.value(first_slot);
    out.value(hits);
    (void)enqueue(frame(RecordType::Hits, std::move(out)), false);
}

bool DiskStore::failed() const noexcept {
    std::lock_guard lock(mutex_);
    return !failure_.empty();
}

bool DiskStore::writable(std::span<const std::uint32_t> slots) const noexcept {
    const std::uint64_t durable = durable_.load(std::memory_order_acquire);
    return std::all_of(slots.begin(), slots.end(),
                       [&](std::uint32_t slot) { return slot < freed_at_.size() && freed_at_[slot] <= durable; });
}

void DiskStore::flush() {
    std::unique_lock lock(mutex_);
    const std::uint64_t last = next_sequence_ - 1U;
    durable_changed_.wait(lock, [&] { return durable_.load() >= last || !failure_.empty(); });
    if (!failure_.empty()) { throw std::runtime_error(failure_); }
}

void DiskStore::journal_main() noexcept {
    for (;;) {
        std::deque<Pending> batch;
        {
            std::unique_lock lock(mutex_);
            wake_.wait(lock, [&] { return stopping_ || !pending_.empty(); });
            if (pending_.empty()) { return; }
            batch.swap(pending_);
        }
        try {
            if (std::any_of(batch.begin(), batch.end(), [](const Pending& p) { return p.data; })) { io_->sync_data(); }
            for (const Pending& pending : batch) {
                write_all(journal_fd_, pending.bytes, "cannot write the prefix cache Disk tier journal");
            }
            if (::fdatasync(journal_fd_) != 0) {
                throw std::system_error(errno, std::generic_category(), "cannot sync the prefix cache Disk tier journal");
            }
            durable_.store(batch.back().sequence, std::memory_order_release);
        } catch (const std::exception& error) {
            // Nothing more is recorded: slots freed after this never become writable, and the records
            // not written are lost at the next start (their slots read as free).
            std::lock_guard lock(mutex_);
            failure_ = error.what();
            durable_changed_.notify_all();
            return;
        }
        durable_changed_.notify_all();
    }
}

#endif

} // namespace infernix::models::qwen4_exp::prefix
