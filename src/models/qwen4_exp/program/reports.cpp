// Operational reports, diagnostics and round traces.

#include "models/qwen4_exp/program/program_impl.h"

namespace infernix::models::qwen4_exp::detail {

void ProgramImpl::trace_round(offload::RouteTraceKind kind, std::int32_t rows, std::int32_t width,
                              std::uint32_t tokens, std::size_t budget, std::uint32_t position,
                              std::span<const std::uint8_t> live, std::span<const std::int32_t> lanes) {
    if (!trace_) { return; }
    trace_->append({.kind     = kind,
                    .rows     = static_cast<std::uint32_t>(rows),
                    .width    = static_cast<std::uint32_t>(width),
                    .tokens   = tokens,
                    .budget   = static_cast<std::uint32_t>(std::min<std::size_t>(budget, 0xFFFFFFFFU)),
                    .position = position,
                    .frames   = residency_->frames()},
                   lanes.empty() ? std::span<const std::int32_t>(host_lanes_.data(), static_cast<std::size_t>(rows))
                                : lanes,
                   live, residency_->route_host(), residency_->route_stride());
}

void ProgramImpl::report_cache(const Lane& lane) noexcept {
    try {
        const auto& at_admission = lane.cache_at_admission;
        const auto& s       = residency_->stats();
        const auto routed   = s.routed - at_admission.routed;
        const auto hits     = s.hits - at_admission.hits;
        const auto promoted = s.promotions - at_admission.promotions;
        const auto landed   = s.landed - at_admission.landed;
        const auto cpu      = cpu_service_ ? cpu_service_->served_experts() - lane.cpu_served_at_admission : 0;
        std::size_t free_vram = 0, total_vram = 0;
        if (cudaMemGetInfo(&free_vram, &total_vram) != cudaSuccess) { free_vram = 0; }
        char text[512];
        std::snprintf(text, sizeof(text),
                      "expert cache: %u frames; request %.1f%% of %llu routed experts hit, %llu promotions, "
                      "%llu landed, %llu misses CPU-served; since start %.1f%%; VRAM free %zu MiB",
                      residency_->frames(), routed ? 100.0 * static_cast<double>(hits) / static_cast<double>(routed) : 0.0,
                      static_cast<unsigned long long>(routed), static_cast<unsigned long long>(promoted),
                      static_cast<unsigned long long>(landed),
                      static_cast<unsigned long long>(cpu),
                      s.routed ? 100.0 * static_cast<double>(s.hits) / static_cast<double>(s.routed) : 0.0,
                      free_vram >> 20);
        diagnostic(text, DiagnosticLevel::Debug);
        if (tier_) {
            // The tier's counters as of the last round boundary.
            const auto& t = tier_->stats();
            const auto& a = lane.tier_at_admission;
            const auto reads = t.demand_reads - a.demand_reads;
            std::snprintf(text, sizeof(text),
                          "SSD expert tier: request %llu expert reads (%.2f GiB, mean %.2f ms), %llu failed; %llu "
                          "read for the prefill stream (%.2f GiB); %llu "
                          "fetched (%llu from RAM slots, %llu copied from unpinned records); %llu served in place "
                          "from unpinned records; %llu admitted to RAM, %llu demoted from VRAM, %llu promoted from unpinned records",
                          static_cast<unsigned long long>(reads),
                          static_cast<double>(reads) * static_cast<double>(residency_->frame_bytes()) / (1ULL << 30),
                          reads ? 1e-6 * static_cast<double>(t.read_ns - a.read_ns) / static_cast<double>(reads) : 0.0,
                          static_cast<unsigned long long>(t.demand_failures - a.demand_failures),
                          static_cast<unsigned long long>(t.stream_reads - a.stream_reads),
                          static_cast<double>(t.stream_reads - a.stream_reads) *
                              static_cast<double>(residency_->frame_bytes()) / (1ULL << 30),
                          static_cast<unsigned long long>(t.fetch_records - a.fetch_records),
                          static_cast<unsigned long long>(t.fetch_from_ram - a.fetch_from_ram),
                          static_cast<unsigned long long>(t.unpinned_copies - a.unpinned_copies),
                          static_cast<unsigned long long>(t.unpinned_served - a.unpinned_served),
                          static_cast<unsigned long long>(t.admitted - a.admitted),
                          static_cast<unsigned long long>(s.demotions - at_admission.demotions),
                          static_cast<unsigned long long>(s.unpinned_promotions - at_admission.unpinned_promotions));
            diagnostic(text, DiagnosticLevel::Debug);
        }
    } catch (...) {}
}

void ProgramImpl::report_lane_cost(std::int32_t lanes, std::uint64_t record_stride) noexcept {
    try {
        const auto n         = static_cast<std::size_t>(lanes);
        const bool fixed_kv  = options_.kv_capacity_tokens != 0;
        const std::size_t kv = fixed_kv ? 0 : kv_span_.bytes / n;
        // The feature buffer's call region (one window) is shared; its round region is per lane.
        const std::size_t dflash_round =
            plan_.dflash2 ? 2ULL * plan_.dflash2_config->feature_width(c_.hidden_size) *
                                static_cast<std::size_t>(plan_.dflash_round_columns)
                          : 0;
        const std::size_t state =
            (state_backing_.bytes + ple_backing_.bytes + tails_backing_.bytes + logits32_.bytes + logits16_.bytes +
             token_counts_.bytes + records_backing_.bytes + ple_records_.bytes + qsa_records_.bytes +
             mtp_residuals_.bytes + mtp_saved_.bytes + mtp_chain_.bytes + mtp_records_.bytes +
             dflash_ring_backing_.bytes + dflash_round) /
            n;
        // The workspace grows with the lane count by its sampling, acceptance and drafter rows:
        // the mean step from one lane to this count, or the second lane's at one lane.
        const std::size_t wide   = workspace_capacity(lanes > 1 ? lanes : 2);
        const std::size_t narrow = workspace_capacity(1);
        const std::size_t workspace = wide > narrow ? (wide - narrow) / (lanes > 1 ? n - 1 : 1) : 0;
        const double lane_frames = static_cast<double>(kv + state + workspace) /
                                   static_cast<double>(std::max<std::uint64_t>(record_stride, 1));
        char kv_text[64];
        if (fixed_kv) {
            std::snprintf(kv_text, sizeof(kv_text), "KV in the fixed --kv-capacity pool");
        } else {
            std::snprintf(kv_text, sizeof(kv_text), "KV %zu MiB", kv >> 20);
        }
        char text[320];
        std::snprintf(text, sizeof(text),
                      "expert cache: %u frames of %.2f MiB at %d lane%s; each lane holds %.1f frames (%s; "
                      "recurrent state, records and workspace %zu MiB)",
                      residency_->frames(), static_cast<double>(record_stride) / 1048576.0, lanes,
                      lanes == 1 ? "" : "s", lane_frames, kv_text, (state + workspace) >> 20);
        diagnostic(text, DiagnosticLevel::Debug);
    } catch (...) {}
}

void ProgramImpl::report_ngram(const Lane& lane) noexcept {
    try {
        const auto& n = lane.ngram;
        char text[192];
        std::snprintf(text, sizeof(text),
                      "n-gram rows: %llu requested, %.1f%% host-cache hits, %llu NVMe reads, %.1f ms of reads",
                      static_cast<unsigned long long>(n.rows),
                      n.rows ? 100.0 * static_cast<double>(n.hits) / static_cast<double>(n.rows) : 0.0,
                      static_cast<unsigned long long>(n.reads), static_cast<double>(n.read_ns) * 1e-6);
        std::string line = text;
        if (const auto& g = lane.gate; g.rounds > 0) {
            std::uint64_t waits[2] = {};
            CUDA_CHECK(cudaMemcpyAsync(waits, static_cast<const std::byte*>(gate_stats_.p) + kGateStatsBytes * lane_index(lane),
                                       sizeof(waits), cudaMemcpyDeviceToHost, device_.stream));
            device_.synchronize();
            std::snprintf(text, sizeof(text),
                          "; %llu gated rounds: %llu NVMe reads (%.1f ms) behind the gate, the GPU waited in %llu "
                          "(mean %.1f us)",
                          static_cast<unsigned long long>(g.rounds), static_cast<unsigned long long>(g.reads),
                          static_cast<double>(g.read_ns) * 1e-6, static_cast<unsigned long long>(waits[1]),
                          waits[1] ? static_cast<double>(waits[0]) * 1e-3 / static_cast<double>(waits[1]) : 0.0);
            line += text;
        }
        diagnostic(line, DiagnosticLevel::Debug);
    } catch (...) {}
}

void ProgramImpl::report_verify_rounds(const Lane& lane) noexcept {
    try {
        const auto& a = lane.verify_at_admission;
        const auto rounds = verify_rounds_ - a[0];
        if (rounds == 0) { return; }
        const auto columns = verify_columns_ - a[1];
        const auto padded  = verify_padded_ - a[2];
        char text[224];
        std::snprintf(text, sizeof(text),
                      "verify rounds: request %llu rounds, %llu of %llu columns past their row's drafts "
                      "(%.1f%%); since start %llu of %llu (%.1f%%)",
                      static_cast<unsigned long long>(rounds),
                      static_cast<unsigned long long>(padded), static_cast<unsigned long long>(columns),
                      columns ? 100.0 * static_cast<double>(padded) / static_cast<double>(columns) : 0.0,
                      static_cast<unsigned long long>(verify_padded_),
                      static_cast<unsigned long long>(verify_columns_),
                      verify_columns_
                          ? 100.0 * static_cast<double>(verify_padded_) / static_cast<double>(verify_columns_)
                          : 0.0);
        diagnostic(text);
        std::string widths = "verify rounds by width, since start:";
        for (std::size_t w = 2; w < verify_by_width_.size(); ++w) {
            const auto& by_width = verify_by_width_[w];
            if (by_width[0] == 0) { continue; }
            std::snprintf(text, sizeof(text), " %zu: %llu rounds, %.1f%% padded", w,
                          static_cast<unsigned long long>(by_width[0]),
                          by_width[1] ? 100.0 * static_cast<double>(by_width[2]) / static_cast<double>(by_width[1])
                                      : 0.0);
            widths += text;
        }
        if (widths.back() != ':') { diagnostic(std::move(widths), DiagnosticLevel::Debug); }
    } catch (...) {}
}

void ProgramImpl::diagnostic(const std::string& text, DiagnosticLevel level) const noexcept {
    try {
        if (options_.diagnostics.callback) {
            options_.diagnostics.callback(Diagnostic{.level = level, .message = text});
        } else {
            std::fprintf(stderr, "[engine] %s\n", text.c_str());
        }
    } catch (...) {}
}

void ProgramImpl::trace(DebugCategory category, const char* format, ...) const noexcept {
    if (!options_.diagnostics.tracing(category)) [[likely]] { return; }
    std::va_list args;
    va_start(args, format);
    runtime::publish_diagnostic_v(options_.diagnostics, DiagnosticLevel::Debug, category, format, args);
    va_end(args);
}

void ProgramImpl::report_cache_layers() noexcept {
    if (!residency_) { return; }
    try {
        const std::span<const std::uint64_t> routed = residency_->routed_by_layer();
        const std::span<const std::uint64_t> hits   = residency_->hits_by_layer();
        if (routed.empty()) { return; }
        std::string text = "expert cache per layer (hit% of routed uses):";
        char part[64]; // one layer term plus the run term fit their worst case
        std::uint64_t total_routed = 0, total_hits = 0;
        for (std::size_t layer = 0; layer < routed.size(); ++layer) {
            total_routed += routed[layer];
            total_hits += hits[layer];
            std::snprintf(part, sizeof(part), " %zu:%.1f%%(%llu)", layer,
                          routed[layer] ? 100.0 * static_cast<double>(hits[layer]) / static_cast<double>(routed[layer]) : 0.0,
                          static_cast<unsigned long long>(routed[layer]));
            text += part;
        }
        std::snprintf(part, sizeof(part), "; run %.1f%% of %llu",
                      total_routed ? 100.0 * static_cast<double>(total_hits) / static_cast<double>(total_routed) : 0.0,
                      static_cast<unsigned long long>(total_routed));
        text += part;
        diagnostic(text, DiagnosticLevel::Debug);
    } catch (...) {}
}

} // namespace infernix::models::qwen4_exp::detail
