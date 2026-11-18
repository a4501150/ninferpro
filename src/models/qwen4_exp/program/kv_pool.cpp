// The elastic KV pool: mapping, growing and shrinking its pages.

#include "models/qwen4_exp/program/program_impl.h"

namespace infernix::models::qwen4_exp::detail {

bool ProgramImpl::kv_idle_due() const noexcept {
    return kv_grown_.load() && std::chrono::steady_clock::now().time_since_epoch().count() - kv_busy_at_.load() >=
                                   std::chrono::duration_cast<std::chrono::steady_clock::duration>(kKvIdleShrink).count();
}

bool ProgramImpl::map_kv(std::uint32_t pages) {
    const auto need = kv_chunks(plan_, pages);
    for (std::size_t c = 0; c < need.size(); ++c) {
        if (!need[c] || kv_vmm_->mapped(c)) { continue; }
        if (!kv_vmm_->map(c)) { return false; }
        CUDA_CHECK(cudaMemsetAsync(static_cast<std::byte*>(kv_vmm_->base()) + c * kKvChunkBytes, 0, kKvChunkBytes,
                                   device_.stream));
    }
    return true;
}

bool ProgramImpl::grow_kv(std::uint32_t need) {
    if (!kv_vmm_ || pool_->available_pages() >= need) { return pool_->available_pages() >= need; }
    const std::uint32_t backed = pool_->backed_pages();
    std::uint32_t target       = backed + (need - pool_->available_pages());
    target = std::min(pool_->capacity_pages(), (target + kKvGrowPages - 1) / kKvGrowPages * kKvGrowPages);
    if (target - backed < need - pool_->available_pages()) { return false; }
    const std::lock_guard<std::recursive_mutex> lock(vram_mutex_);
    const std::size_t added    = kv_mapped_bytes(plan_, target) - kv_vmm_->mapped_bytes();
    const std::uint64_t stride = residency_->frame_bytes();
    const std::uint32_t frames = residency_->frames();
    // Memory the sizing function would give the frames anyway comes first; the rest is taken from
    // the frames, one chunk per pool over so the pools unmap at least that much.
    const std::uint32_t spare =
        residency_->elastic()
            ? std::max<std::uint32_t>(frame_target(spare_demand_, offload::ExpertResidency::kChunkBytes,
                                                   vram_->query(), residency_->pool_bytes(), usage_at_sizing_,
                                                   pool_at_sizing_),
                                     frames) -
                  frames
            : 0;
    const std::uint64_t spare_bytes = static_cast<std::uint64_t>(spare) * stride;
    std::uint32_t give              = 0;
    if (added > spare_bytes) {
        give = static_cast<std::uint32_t>((added - spare_bytes + residency_->pools() * offload::ExpertResidency::kChunkBytes + stride - 1) / stride);
    }
    if (give > frames) { return false; }
    device_.synchronize();
    if (give != 0) {
        const auto resized = residency_->resize(frames - give, device_.stream, *vram_);
        if (resized.frames > frames - give) { return false; } // a lease holds the frames
    }
    if (!map_kv(target)) {
        if (give != 0) { (void)residency_->resize(frames, device_.stream, *vram_); }
        return false;
    }
    pool_->set_backed_pages(target);
    kv_grown_.store(true);
    if (residency_->elastic()) {
        usage_at_sizing_ = static_cast<std::uint64_t>(static_cast<std::int64_t>(usage_at_sizing_) +
                                                      static_cast<std::int64_t>(added));
    }
    diagnostic("KV pool grew to " + std::to_string(target) + " pages (" +
               std::to_string(kv_vmm_->mapped_bytes() >> 20) + " MiB); expert cache " + std::to_string(frames) +
               " -> " + std::to_string(residency_->frames()) + " frames");
    return true;
}

void ProgramImpl::shrink_kv(bool evict) {
    if (!kv_vmm_ || pool_->backed_pages() <= plan_.kv_base_pages) { return; }
    if (evict && prefix_) {
        prefix::cache_index::PrefixCacheIndex& index = prefix_->index();
        for (std::uint32_t budget = pool_->backed_pages() - plan_.kv_base_pages;
             budget != 0 && !pool_->can_back(plan_.kv_base_pages) && index.evict_backed_device_blocks(1) != 0; --budget) {}
    }
    std::uint32_t target = pool_->backed_pages();
    while (target > plan_.kv_base_pages && pool_->can_back(target - std::min(target, kKvGrowPages))) {
        target -= std::min(target, kKvGrowPages);
    }
    target = std::max(target, plan_.kv_base_pages);
    if (target == pool_->backed_pages()) { return; }
    const std::lock_guard<std::recursive_mutex> lock(vram_mutex_);
    device_.synchronize();
    const std::size_t before = kv_vmm_->mapped_bytes();
    pool_->set_backed_pages(target);
    const auto need = kv_chunks(plan_, target);
    for (std::size_t c = 0; c < need.size(); ++c) {
        if (!need[c] && kv_vmm_->mapped(c)) { kv_vmm_->unmap(c); }
    }
    kv_grown_.store(target > plan_.kv_base_pages);
    const std::size_t freed   = before - kv_vmm_->mapped_bytes();
    const std::uint64_t stride = residency_->frame_bytes();
    const std::uint32_t frames = residency_->frames();
    (void)residency_->resize(frames + static_cast<std::uint32_t>(freed / stride), device_.stream, *vram_);
    if (residency_->elastic()) {
        usage_at_sizing_ = static_cast<std::uint64_t>(static_cast<std::int64_t>(usage_at_sizing_) -
                                                      static_cast<std::int64_t>(freed));
    }
    diagnostic("KV pool shrank to " + std::to_string(target) + " pages; expert cache " + std::to_string(frames) +
               " -> " + std::to_string(residency_->frames()) + " frames");
}

std::uint32_t ProgramImpl::kv_grow_pages() const noexcept {
    if (!kv_vmm_ || !residency_ || !residency_->elastic() || residency_->frames() == 0 ||
        (residency_->stats().lent_frames != 0 && !leases_returnable())) {
        return 0;
    }
    const std::uint32_t room = pool_->capacity_pages() - pool_->backed_pages();
    const std::uint64_t page_bytes =
        std::max<std::uint64_t>(1, kv_span_.bytes / std::max<std::uint32_t>(1, pool_->capacity_pages()));
    const std::uint64_t frame_room = static_cast<std::uint64_t>(residency_->frames()) * residency_->frame_bytes();
    const std::uint64_t slack      = residency_->pools() * offload::ExpertResidency::kChunkBytes + 2 * kKvChunkBytes * plan_.pool.planes.size();
    const std::uint64_t usable     = frame_room > slack ? frame_room - slack : 0;
    return static_cast<std::uint32_t>(std::min<std::uint64_t>(room, usable / page_bytes));
}

} // namespace infernix::models::qwen4_exp::detail
