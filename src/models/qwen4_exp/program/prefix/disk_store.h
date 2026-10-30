#pragma once

// The Disk tier's directory for Qwen4Exp's prefix cache (--prefix-cache-disk): the slot file the
// index's Disk slots name, and the journal of what those slots hold. The index decides which slot
// holds what; this class keeps the record of it across restarts and orders that record after the
// bytes it describes.
//
// - `slots.bin`: Disk slot s at byte s * slot_bytes, written and read through cuFile.
// - `catalog.log`: one record per Disk entry (a block, a snapshot, its removal, a snapshot's hits),
//   each with a CRC. A record is written only after the slots it names are durable, and a slot is
//   rewritten only after the removal of its previous record is durable, so a crash leaves at most
//   records that were never written (their slots read as free). A block record follows its
//   parent's and a snapshot record its anchor's: replay links each to the record live at that
//   slot when it is read, so a link never reaches a block written later into a freed slot. Replay
//   stops at the first damaged record; startup then rewrites the journal with the live records.
// - `manifest`: the fingerprint and slot geometry. A mismatch clears the journal: nothing in the
//   slot file is trusted.
// - `lock`: held while the store is open, so two servers never share a directory.

#include "core/cufile_io.h"
#include "models/qwen4_exp/program/prefix/state_image.h"
#include "runtime/prefix_cache/block_hash.h"
#include "models/qwen4_exp/program/prefix/prefix_index.h"

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace infernix::models::qwen4_exp::prefix {

struct DiskGeometry {
    std::uint64_t slot_bytes        = 0; // one Host slab
    std::uint32_t slots             = 0;
    std::uint32_t image_slots       = 0;
    std::uint64_t page_stride       = 0;
    std::uint64_t image_fingerprint = 0;

    [[nodiscard]] friend bool operator==(const DiskGeometry&, const DiskGeometry&) = default;
};

struct DiskBlockRecord {
    std::uint32_t slot        = prefix::cache_index::kNoId;
    std::uint32_t parent_slot = prefix::cache_index::kNoId; // the parent block's record; kNoId: root
    std::uint64_t lookup_hash = 0;
    std::uint64_t extra       = 0;
    std::int32_t mtp_next     = -1;
    std::array<TokenId, runtime::prefix_cache::kBlockTokens> tokens{};
};

struct DiskSnapshotRecord {
    std::vector<std::uint32_t> slots; // image, then the tail's
    std::uint32_t anchor_slot = prefix::cache_index::kNoId; // the anchor block's record; kNoId: root
    std::uint32_t frontier    = 0;
    std::vector<TokenId> tail;
    prefix::cache_index::SnapshotKind kind = prefix::cache_index::SnapshotKind::Tap;
    std::uint32_t hits = 0;
    StateImageHeader meta;
};

class DiskStore {
public:
    // Opens or creates the store in `directory` (created when missing) for `geometry`. Throws when
    // the directory is in use by another store or cannot be created.
    DiskStore(const std::filesystem::path& directory, std::string_view fingerprint, const DiskGeometry& geometry);
    ~DiskStore(); // makes every appended record durable

    DiskStore(const DiskStore&)            = delete;
    DiskStore& operator=(const DiskStore&) = delete;

    [[nodiscard]] CuFileQueue& io() noexcept { return *io_; }
    [[nodiscard]] const DiskGeometry& geometry() const noexcept { return geometry_; }
    [[nodiscard]] std::uint64_t slot_offset(std::uint32_t slot) const noexcept {
        return static_cast<std::uint64_t>(slot) * geometry_.slot_bytes;
    }

    // The live records replay found, blocks parents first; empty after a reset. `reset_reason` says
    // why the journal was cleared (empty when it was trusted), `damaged` whether replay stopped at
    // a damaged record.
    [[nodiscard]] const std::vector<DiskBlockRecord>& replayed_blocks() const noexcept { return blocks_; }
    [[nodiscard]] const std::vector<DiskSnapshotRecord>& replayed_snapshots() const noexcept { return snapshots_; }
    [[nodiscard]] const std::string& reset_reason() const noexcept { return reset_reason_; }
    [[nodiscard]] bool damaged() const noexcept { return damaged_; }
    // Drops the replayed records (after the caller rebuilt the index from them) and rewrites the
    // journal with `blocks` and `snapshots`, the ones the index kept; appends follow. A later call
    // (the cache was emptied) first makes every appended record durable; it throws when that fails.
    void compact(std::span<const DiskBlockRecord> blocks, std::span<const DiskSnapshotRecord> snapshots);

    // Records whose slots have completed their writes. The record is written after those writes are
    // durable, on the journal thread.
    void append_block(const DiskBlockRecord& record);
    void append_snapshot(const DiskSnapshotRecord& record);
    // The index released the record at `first_slot`, which spans `slots`; they may be rewritten once
    // `writable` says so.
    void append_removal(std::uint32_t first_slot, std::span<const std::uint32_t> slots);
    void append_hits(std::uint32_t first_slot, std::uint32_t hits);
    // Whether every slot's previous record removal is durable, so a write cannot leave a record that
    // describes other bytes.
    [[nodiscard]] bool writable(std::span<const std::uint32_t> slots) const noexcept;
    // Waits until every appended record is durable. Throws when the journal thread failed.
    void flush();
    // The journal could not be written: nothing appended since is recorded, and no freed slot
    // becomes writable again.
    [[nodiscard]] bool failed() const noexcept;

private:
    struct Pending {
        std::vector<std::byte> bytes;
        std::uint64_t sequence = 0;
        bool data              = false; // needs the slot file synced first
    };

    void open_journal(bool truncate);
    void replay(std::span<const std::byte> journal);
    std::uint64_t enqueue(std::vector<std::byte>&& bytes, bool data);
    void journal_main() noexcept;

    std::filesystem::path directory_;
    DiskGeometry geometry_;
    int lock_fd_    = -1;
    int journal_fd_ = -1;
    std::unique_ptr<CuFileQueue> io_;
    std::vector<DiskBlockRecord> blocks_;
    std::vector<DiskSnapshotRecord> snapshots_;
    std::string reset_reason_;
    bool damaged_ = false;

    // Sequence of the removal that last freed each slot (0: none since startup).
    std::vector<std::uint64_t> freed_at_;
    std::uint64_t next_sequence_ = 1;
    std::atomic<std::uint64_t> durable_{0};

    mutable std::mutex mutex_;
    std::condition_variable wake_;
    std::condition_variable durable_changed_;
    std::deque<Pending> pending_;
    bool stopping_ = false;
    std::string failure_;
    std::thread journal_;
};

} // namespace infernix::models::qwen4_exp::prefix
