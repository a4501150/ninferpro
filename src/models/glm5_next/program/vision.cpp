// The Program side of the vision tower (docs/maintainer/glm5_3-flash-design.md §15.2): binding
// plans the segments past the reused prefix, the lane's first prefill call encodes them in one
// pass into the handoff (with vision offload, streaming the tower's weights through the same
// memory), every call scatters its columns of it over the token embeddings, and the handoff goes
// back once the prompt's last call has completed.

#include "models/glm5_next/program/program_impl.h"

#include "models/common/vision/weight_stream.h"
#include "models/glm5_next/execution/vision.h"

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <string>

namespace infernix::models::glm5_next::detail {

void ProgramImpl::vision_plan(Lane& lane, const PreparedPromptData& prompt, std::uint32_t reused) {
    lane.vision.reset();
    if (!prompt.has_media()) { return; }
    if (!parameters_.vision) {
        throw std::invalid_argument(
            "Glm5Next: images and video need an engine started with --vision");
    }
    const VisionConfig& config = *parameters_.model.config().vision;
    const std::size_t width    = config.patch_width();
    auto vision                = std::make_unique<LaneVision>();
    for (std::size_t i = 0; i < prompt.vision_items.size(); ++i) {
        const VisionItem& item = prompt.vision_items[i];
        const auto pair        = static_cast<std::size_t>(item.grid.height) * item.grid.width;
        const auto& payload    = prompt.media_payloads.at(i);
        if (!payload || payload->patch_elements != item.patch_count * width ||
            item.patch_count != pair * item.token_spans.size()) {
            throw std::invalid_argument("Glm5Next: a media item's patches do not match its grid");
        }
        bool used = false;
        for (std::size_t t = 0; t < item.token_spans.size(); ++t) {
            const TokenSpan& span = item.token_spans[t];
            if (span.begin + span.count <= reused) { continue; }
            if (span.count * config.merge_area() != pair) {
                throw std::invalid_argument("Glm5Next: a media span does not match its grid");
            }
            vision->segments.push_back(
                {.patches   = payload->span().subspan(t * pair * width, pair * width),
                 .positions = execution::vision_patch_positions(
                     static_cast<std::uint32_t>(item.grid.height),
                     static_cast<std::uint32_t>(item.grid.width), config.spatial_merge_size),
                 .token_begin = static_cast<std::uint32_t>(span.begin),
                 .tokens      = static_cast<std::uint32_t>(span.count)});
            for (std::size_t k = 0; k < span.count; ++k) {
                vision->visual.push_back(static_cast<std::uint32_t>(span.begin + k));
            }
            used = true;
        }
        if (used) { vision->payloads.push_back(payload); }
    }
    if (vision->segments.empty()) { return; } // every item lies in the reused prefix
    if (!std::is_sorted(vision->visual.begin(), vision->visual.end())) {
        throw std::logic_error("Glm5Next: the media spans are not in prompt order");
    }
    lane.vision = std::move(vision);
}

void ProgramImpl::vision_encode(Lane& lane) {
    LaneVision& v              = *lane.vision;
    const VisionConfig& config = *parameters_.model.config().vision;
    const cudaStream_t s       = device_.stream;
    const auto O               = dim(config.out_hidden_size);
    std::vector<std::size_t> patches;
    for (const auto& segment : v.segments) { patches.push_back(segment.positions.size() / 2); }
    const execution::VisionPassLayout layout =
        execution::plan_vision_pass(config, *parameters_.vision, patches);
    const auto& overlay       = parameters_.model.overlay_vision();
    const std::size_t handoff = align(2ULL * O * v.visual.size());
    const std::size_t staging =
        overlay ? common::vision_staging_align(overlay->layout.staging_bytes) : 0;
    const std::size_t bytes = handoff + staging + layout.bytes;
    // The weight stream's transfer stream writes the staging: it waits for promotions into the run.
    const cudaStream_t writers[] = {device_.transfer_stream};
    if (residency_->can_lend(bytes)) { v.lease = residency_->lend(bytes, s, writers); }
    DeviceSpan memory{};
    if (v.lease.valid()) {
        memory = v.lease.memory;
    } else {
        // The resident placement lends no frame, and a pass larger than what may be lent cannot be
        // lent either: the handoff and pass are allocated for the encode instead (a failed
        // allocation fails this prompt's request, not the Engine).
        try {
            v.fallback = DeviceBuffer(bytes);
        } catch (const std::exception& error) {
            throw runtime::RecoverableExecutionError(
                "the vision encode (" + std::to_string(bytes >> 20) +
                " MiB) could neither be lent from the expert cache nor allocated (" + error.what() +
                ")");
        }
        memory = {v.fallback.p, bytes};
    }
    std::byte* const base = static_cast<std::byte*>(memory.data);
    std::optional<common::VisionWeightStream> stream;
    std::optional<execution::VisionParameters> streamed;
    if (overlay) {
        stream.emplace(device_, *overlay, base + handoff);
        streamed = execution::vision_window_weights(*stream, *parameters_.vision);
    }
    v.handoff = Tensor(memory.data, DType::BF16, {O, static_cast<std::int32_t>(v.visual.size())});
    std::vector<execution::VisionTowerItem> items;
    std::size_t column = 0;
    for (const auto& segment : v.segments) {
        items.push_back({.patches   = segment.patches,
                         .positions = segment.positions,
                         .output    = v.handoff.slice(1, static_cast<std::int32_t>(column),
                                                      static_cast<std::int32_t>(segment.tokens))});
        column += segment.tokens;
    }
    execution::VisionTowerPass pass(
        device_, config, streamed ? *streamed : *parameters_.vision, std::move(items),
        DeviceSpan{base + handoff + staging, layout.bytes}, layout, stream ? &*stream : nullptr);
    pass.advance(execution::VisionTowerPass::stages(config.depth));
    if (stream) { stream->finish(s); }
    if (v.lease.valid()) {
        // Only the handoff stays lent; the pass's frames go back once compute reaches here.
        auto tail = residency_->split(v.lease, handoff);
        if (tail.valid()) { residency_->give_back(tail, s); }
    }
    diagnostic("vision encode: " + std::to_string(v.segments.size()) + " items, " +
                   std::to_string(v.visual.size()) + " tokens, " + std::to_string(bytes >> 20) +
                   " MiB " + (v.lease.valid() ? "lent" : "allocated") +
                   (stream ? ", " + std::to_string(stream->uploaded_bytes() >> 20) +
                                 " MiB of weights streamed"
                           : std::string()),
               DiagnosticLevel::Debug);
    v.segments.clear();
    v.payloads.clear(); // the patches were copied before the pass returned
    v.encoded = true;
}

std::pair<std::uint32_t, std::uint32_t> ProgramImpl::vision_columns(const Lane& lane,
                                                                    std::uint32_t begin,
                                                                    std::uint32_t width,
                                                                    std::int32_t* columns) const {
    if (!lane.vision) { return {0, 0}; }
    const auto& visual = lane.vision->visual;
    const auto first   = std::lower_bound(visual.begin(), visual.end(), begin);
    const auto last    = std::lower_bound(first, visual.end(), begin + width);
    for (auto at = first; at != last; ++at) { *columns++ = static_cast<std::int32_t>(*at - begin); }
    return {static_cast<std::uint32_t>(first - visual.begin()),
            static_cast<std::uint32_t>(last - first)};
}

void ProgramImpl::vision_batch(const Lane& lane, std::uint32_t first, std::uint32_t count,
                               std::byte* columns, execution::ForwardBatch& batch) const {
    if (count == 0) { return; }
    batch.vision_embeddings = lane.vision->handoff.slice(1, static_cast<std::int32_t>(first),
                                                         static_cast<std::int32_t>(count));
    batch.vision_columns    = Tensor(columns, DType::I32, {static_cast<std::int32_t>(count)});
}

void ProgramImpl::vision_release(Lane& lane) noexcept {
    if (!lane.vision) { return; }
    try {
        LaneVision& v = *lane.vision;
        if (v.lease.valid()) { residency_->give_back(v.lease, device_.stream); }
        if (v.fallback.p != nullptr) {
            // The free is a blocking cudaFree: it waits for the enqueued calls that read the
            // handoff.
            device_.synchronize();
            v.fallback = DeviceBuffer{};
        }
    } catch (...) {}
    lane.vision.reset();
}

} // namespace infernix::models::glm5_next::detail
