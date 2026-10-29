// The prefix cache Disk tier's store (disk_store.h): the journal replays blocks parents first and
// snapshots with their meta, a link reaches only the record live when it was written, removals
// and hits apply, a damaged tail stops replay, a manifest
// mismatch clears the journal, the directory is held by one store, and a freed slot becomes
// writable only once its removal is durable. `INFERNIX_TEST_DISK_DIR` puts the store on a given
// filesystem; otherwise it lives in the system temp directory.
#include "../../test_temp_path.h"
#include "models/qwen4_exp/program/prefix/disk_store.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace infernix::models::qwen4_exp::prefix;
using infernix::TokenId;
namespace pc = infernix::runtime::prefix_cache;

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        ++failures;
        std::cerr << "FAIL: " << what << '\n';
    }
}

const DiskGeometry kGeometry{
    .slot_bytes = 8192, .slots = 64, .image_slots = 3, .page_stride = 8000, .image_fingerprint = 42};

DiskBlockRecord block(std::uint32_t slot, std::uint32_t parent, std::int32_t seed) {
    DiskBlockRecord record;
    record.slot        = slot;
    record.parent_slot = parent;
    record.lookup_hash = 1000U + slot;
    record.extra       = slot * 3U;
    record.mtp_next    = seed;
    for (std::size_t i = 0; i < record.tokens.size(); ++i) { record.tokens[i] = static_cast<TokenId>(seed + i); }
    return record;
}

DiskSnapshotRecord snapshot(std::vector<std::uint32_t> slots, std::uint32_t anchor, std::uint32_t frontier,
                            std::vector<TokenId> tail) {
    DiskSnapshotRecord record;
    record.slots                = std::move(slots);
    record.anchor_slot          = anchor;
    record.frontier             = frontier;
    record.tail                 = std::move(tail);
    record.kind                 = pc::SnapshotKind::Endpoint;
    record.hits                 = 2;
    record.meta.frontier        = frontier;
    record.meta.mtp_written     = true;
    record.meta.mtp_next        = 77;
    record.meta.head_residual   = true;
    record.meta.mtp_accept[3]   = 0.25;
    record.meta.draft_frontier  = frontier - 1U;
    return record;
}

bool same(const DiskBlockRecord& a, const DiskBlockRecord& b) {
    return a.slot == b.slot && a.parent_slot == b.parent_slot && a.lookup_hash == b.lookup_hash &&
           a.extra == b.extra && a.mtp_next == b.mtp_next && a.tokens == b.tokens;
}

bool same(const DiskSnapshotRecord& a, const DiskSnapshotRecord& b) {
    return a.slots == b.slots && a.anchor_slot == b.anchor_slot && a.frontier == b.frontier && a.tail == b.tail &&
           a.kind == b.kind && a.hits == b.hits && a.meta.frontier == b.meta.frontier &&
           a.meta.mtp_written == b.meta.mtp_written && a.meta.mtp_next == b.meta.mtp_next &&
           a.meta.head_residual == b.meta.head_residual && a.meta.mtp_accept == b.meta.mtp_accept &&
           a.meta.draft_frontier == b.meta.draft_frontier && a.meta.opener == b.meta.opener &&
           a.meta.lineage_echo == b.meta.lineage_echo;
}

std::filesystem::path test_directory() {
    const char* dir = std::getenv("INFERNIX_TEST_DISK_DIR");
    if (dir != nullptr && *dir != '\0') {
        return std::filesystem::path(dir) / infernix::test::unique_temp_path("disk-store").filename();
    }
    return infernix::test::unique_temp_path("disk-store");
}

void run(const std::filesystem::path& directory) {
    const auto root  = block(0, pc::kNoId, 10);
    const auto child = block(5, 0, 20);
    const auto orphan = block(9, 30, 40); // its parent has no record
    const auto early  = block(12, 13, 50); // written before its parent's record
    const auto early_parent = block(13, pc::kNoId, 60);
    const auto reused_parent = block(20, pc::kNoId, 70);
    const auto stale_child   = block(21, 20, 80);
    const auto reused        = block(20, pc::kNoId, 90);
    const auto tap   = snapshot({6, 7, 8, 11}, 5, 140, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    {
        DiskStore store(directory, "fingerprint-a", kGeometry);
        check(store.reset_reason() == "new Disk tier" && store.replayed_blocks().empty(), "a new store is empty");
        check(std::filesystem::file_size(directory / "slots.bin") == kGeometry.slots * kGeometry.slot_bytes,
              "the slot file spans every slot");
        bool refused = false;
        try {
            DiskStore second(directory, "fingerprint-a", kGeometry);
        } catch (const std::runtime_error&) { refused = true; }
        check(refused, "a second store on the same directory is refused");
        store.compact({}, {});
        store.append_block(root);
        store.append_block(child);
        store.append_block(orphan);
        store.append_block(early);
        store.append_block(early_parent);
        store.append_snapshot(tap);
        // A block whose parent's slot was freed and rewritten does not hang below the new block.
        store.append_block(reused_parent);
        store.append_block(stale_child);
        store.append_removal(reused_parent.slot, std::vector<std::uint32_t>{reused_parent.slot});
        store.append_block(reused);
        store.flush();
    }
    {
        DiskStore store(directory, "fingerprint-a", kGeometry);
        check(store.reset_reason().empty() && !store.damaged(), "a matching store replays its journal");
        const auto& blocks = store.replayed_blocks();
        bool parents_first = blocks.size() == 4;
        for (std::size_t i = 0; i < blocks.size(); ++i) {
            parents_first = parents_first && (blocks[i].parent_slot == pc::kNoId ||
                                              std::any_of(blocks.begin(), blocks.begin() + static_cast<std::ptrdiff_t>(i),
                                                          [&](const DiskBlockRecord& b) {
                                                              return b.slot == blocks[i].parent_slot;
                                                          }));
        }
        const auto has = [&](const DiskBlockRecord& wanted) {
            return std::any_of(blocks.begin(), blocks.end(), [&](const DiskBlockRecord& b) { return same(b, wanted); });
        };
        check(parents_first && has(root) && has(child) && has(early_parent), "blocks replay parents first");
        check(std::none_of(blocks.begin(), blocks.end(),
                           [&](const DiskBlockRecord& b) {
                               return b.slot == orphan.slot || b.slot == early.slot || b.slot == stale_child.slot;
                           }),
              "a block without a live parent record when it was written is dropped");
        check(has(reused), "a rewritten slot holds its new block");
        std::vector<DiskBlockRecord> rewritten;
        for (const DiskBlockRecord& b : blocks) {
            if (b.slot == root.slot || b.slot == child.slot) { rewritten.push_back(b); }
        }
        check(store.replayed_snapshots().size() == 1 && same(store.replayed_snapshots()[0], tap),
              "a snapshot replays with its tail and meta");
        const std::vector<DiskSnapshotRecord> kept_snapshots = store.replayed_snapshots();
        store.compact(rewritten, kept_snapshots);
        check(store.replayed_blocks().empty(), "compaction drops the replayed records");

        // A freed slot is writable again only once its removal is durable.
        const std::vector<std::uint32_t> freed{5};
        check(store.writable(freed), "a slot never freed is writable");
        store.append_hits(6, 9);
        store.append_removal(5, freed);
        store.flush();
        check(store.writable(freed), "a durable removal makes its slot writable");
    }
    {
        DiskStore store(directory, "fingerprint-a", kGeometry);
        check(store.replayed_blocks().size() == 1 && same(store.replayed_blocks()[0], root),
              "a removed block is gone after compaction");
        check(store.replayed_snapshots().empty(), "a snapshot whose anchor was removed is dropped");
        store.compact(store.replayed_blocks(), {});
        auto with_hits = tap;
        with_hits.anchor_slot = 0;
        store.append_snapshot(with_hits);
        store.append_hits(6, 9);
        store.flush();
    }
    {
        // A torn tail: replay keeps the records before it.
        std::ofstream journal(directory / "catalog.log", std::ios::binary | std::ios::app);
        journal.write("\x02\x40\x00\x00\x00garbage", 12);
    }
    {
        DiskStore store(directory, "fingerprint-a", kGeometry);
        check(store.damaged(), "a damaged tail is reported");
        check(store.replayed_blocks().size() == 1 && store.replayed_snapshots().size() == 1 &&
                  store.replayed_snapshots()[0].hits == 9,
              "records before the damage replay, hits applied");
        store.compact(store.replayed_blocks(), store.replayed_snapshots());
    }
    {
        DiskStore store(directory, "fingerprint-b", kGeometry);
        check(!store.reset_reason().empty() && store.replayed_blocks().empty() && store.replayed_snapshots().empty(),
              "another fingerprint clears the journal");
        store.compact({}, {});
    }
    {
        DiskGeometry larger = kGeometry;
        larger.slots        = 128;
        DiskStore store(directory, "fingerprint-b", larger);
        check(!store.reset_reason().empty(), "another size clears the journal");
        check(std::filesystem::file_size(directory / "slots.bin") == larger.slots * larger.slot_bytes,
              "the slot file follows the size");
    }
}

} // namespace

int main() {
    const std::filesystem::path directory = test_directory();
    try {
        run(directory);
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        ++failures;
    }
    std::error_code ignored;
    std::filesystem::remove_all(directory, ignored);
    if (failures != 0) { return 1; }
    std::cout << "prefix disk store tests passed\n";
    return 0;
}
