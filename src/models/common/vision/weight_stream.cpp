#include "models/common/vision/weight_stream.h"

#include <stdexcept>

namespace infernix::models::common {
namespace {

// Rebases a pointer inside a group's pinned bytes into the group's staged slot, through the
// segment that contains it. A group's objects are not necessarily adjacent in the pinned block,
// so each object is staged independently.
const void* stage_pointer(const VisionOverlayGroup& group, std::byte* staging, const std::byte* block,
                          const void* pointer) {
    if (pointer == nullptr) { return nullptr; }
    const std::size_t pinned = static_cast<std::size_t>(static_cast<const std::byte*>(pointer) - block);
    for (const VisionOverlaySegment& segment : group.segments) {
        if (pinned >= segment.pinned_offset && pinned < segment.pinned_offset + segment.bytes) {
            return staging + segment.staging_offset + (pinned - segment.pinned_offset);
        }
    }
    throw std::logic_error("overlay weight is outside its staged group");
}

Tensor stage_tensor(const VisionOverlayGroup& group, std::byte* staging, const std::byte* block, const Tensor& source) {
    Tensor out = source;
    out.data   = const_cast<std::byte*>(static_cast<const std::byte*>(stage_pointer(group, staging, block, source.data)));
    return out;
}

Weight stage_weight(const VisionOverlayGroup& group, std::byte* staging, const std::byte* block, const Weight& source) {
    Weight out  = source;
    out.payload = static_cast<const std::byte*>(stage_pointer(group, staging, block, source.payload));
    out.qdata   = stage_pointer(group, staging, block, source.qdata);
    out.qhigh   = stage_pointer(group, staging, block, source.qhigh);
    out.scales  = stage_pointer(group, staging, block, source.scales);
    return out;
}

// Copies a group's pinned segments into its device slot on `stream`.
void stage_group(cudaStream_t stream, std::byte* staging, const VisionOverlayGroup& group, const std::byte* block) {
    for (const VisionOverlaySegment& segment : group.segments) {
        CUDA_CHECK(cudaMemcpyAsync(staging + segment.staging_offset, block + segment.pinned_offset, segment.bytes,
                                   cudaMemcpyHostToDevice, stream));
    }
}

} // namespace

VisionWeightStream::VisionWeightStream(DeviceContext& device, const VisionOverlayAssets& assets, std::byte* staging,
                                       std::span<const cudaEvent_t> wait_before_first_upload)
    : device_(device), assets_(assets) {
    const VisionOverlayLayout& layout = assets.layout;
    prelude_ = staging;
    merger_  = prelude_ + vision_staging_align(layout.prelude.bytes);
    slot_[0] = merger_ + vision_staging_align(layout.merger.bytes);
    slot_[1] = slot_[0] + vision_staging_align(layout.slot_bytes);
    for (cudaEvent_t& event : uploaded_) { CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming)); }
    for (cudaEvent_t* event : {&prelude_event_, &merger_event_, &compute_fence_, &final_event_}) {
        CUDA_CHECK(cudaEventCreateWithFlags(event, cudaEventDisableTiming));
    }
    // Nothing reaches the staging memory before its previous users are done.
    CUDA_CHECK(cudaEventRecord(compute_fence_, device_.stream));
    CUDA_CHECK(cudaStreamWaitEvent(copy_stream(), compute_fence_, 0));
    for (const cudaEvent_t event : wait_before_first_upload) { CUDA_CHECK(cudaStreamWaitEvent(copy_stream(), event, 0)); }
    stage_group(copy_stream(), prelude_, layout.prelude, assets.pinned_block);
    CUDA_CHECK(cudaEventRecord(prelude_event_, copy_stream()));
    upload_bytes_ = layout.prelude.bytes;
    next_upload_  = 0;
    upload_next_layer();
    upload_next_layer();
}

VisionWeightStream::~VisionWeightStream() {
    if (!finished_) { (void)cudaStreamSynchronize(copy_stream()); }
    for (cudaEvent_t event : uploaded_) { (void)cudaEventDestroy(event); }
    for (cudaEvent_t event : {prelude_event_, merger_event_, compute_fence_, final_event_}) {
        (void)cudaEventDestroy(event);
    }
}

Tensor VisionWeightStream::prelude(const Tensor& host) const {
    return stage_tensor(assets_.layout.prelude, prelude_, assets_.pinned_block, host);
}

Weight VisionWeightStream::prelude(const Weight& host) const {
    return stage_weight(assets_.layout.prelude, prelude_, assets_.pinned_block, host);
}

Tensor VisionWeightStream::layer(std::uint32_t layer, const Tensor& host) const {
    return stage_tensor(assets_.layout.layers.at(layer), slot_[layer % 2], assets_.pinned_block,
                        host);
}

Weight VisionWeightStream::layer(std::uint32_t layer, const Weight& host) const {
    return stage_weight(assets_.layout.layers.at(layer), slot_[layer % 2], assets_.pinned_block,
                        host);
}

Tensor VisionWeightStream::merger(const Tensor& host) const {
    return stage_tensor(assets_.layout.merger, merger_, assets_.pinned_block, host);
}

Weight VisionWeightStream::merger(const Weight& host) const {
    return stage_weight(assets_.layout.merger, merger_, assets_.pinned_block, host);
}

void VisionWeightStream::reset(cudaStream_t compute) {
    CUDA_CHECK(cudaEventRecord(compute_fence_, compute));
    CUDA_CHECK(cudaStreamWaitEvent(copy_stream(), compute_fence_, 0));
    next_upload_ = 0;
    upload_next_layer();
    upload_next_layer();
}

void VisionWeightStream::prelude_ready(cudaStream_t compute) {
    CUDA_CHECK(cudaStreamWaitEvent(compute, prelude_event_, 0));
}

void VisionWeightStream::merger_ready(cudaStream_t compute) {
    if (!merger_uploaded_) { throw std::logic_error("Vision weight stream: the merger waits for the last layer's upload"); }
    CUDA_CHECK(cudaStreamWaitEvent(compute, merger_event_, 0));
}

void VisionWeightStream::arrive(std::uint32_t layer, cudaStream_t compute) {
    CUDA_CHECK(cudaStreamWaitEvent(compute, uploaded_[layer % 2], 0));
    const auto layers = static_cast<std::uint32_t>(assets_.layout.layers.size());
    if (next_upload_ == layer + 1 && next_upload_ < layers) {
        CUDA_CHECK(cudaEventRecord(compute_fence_, compute));
        CUDA_CHECK(cudaStreamWaitEvent(copy_stream(), compute_fence_, 0));
        upload_next_layer();
    }
}

void VisionWeightStream::finish(cudaStream_t compute) {
    CUDA_CHECK(cudaEventRecord(final_event_, copy_stream()));
    CUDA_CHECK(cudaStreamWaitEvent(compute, final_event_, 0));
    finished_ = true;
}

void VisionWeightStream::upload_next_layer() {
    const auto layers = static_cast<std::uint32_t>(assets_.layout.layers.size());
    if (next_upload_ >= layers) { return; }
    const std::uint32_t layer       = next_upload_++;
    const VisionOverlayGroup& group = assets_.layout.layers[layer];
    stage_group(copy_stream(), slot_[layer % 2], group, assets_.pinned_block);
    CUDA_CHECK(cudaEventRecord(uploaded_[layer % 2], copy_stream()));
    upload_bytes_ += group.bytes;
    // The merger follows the last layer, so it never delays layer 0.
    if (next_upload_ == layers && !merger_uploaded_) {
        stage_group(copy_stream(), merger_, assets_.layout.merger, assets_.pinned_block);
        CUDA_CHECK(cudaEventRecord(merger_event_, copy_stream()));
        upload_bytes_ += assets_.layout.merger.bytes;
        merger_uploaded_ = true;
    }
}

} // namespace infernix::models::common
