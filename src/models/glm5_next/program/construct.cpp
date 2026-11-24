// The device plan and the Program's construction: every fixed device allocation is computed once
// from the options and the configuration (plan_device) and the constructor allocates exactly it;
// the expert frames take what remains (experts.cpp).

#include "models/glm5_next/program/program_impl.h"

#include "core/weight_view.h"
#include "infernix/ops/speculative_round.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace infernix::models::glm5_next::detail {

namespace {

// Misses staged per pass of a layer's experts: frame-sized slots, so each costs a cached expert's
// VRAM (glm5_3-flash-design.md §10.3). A decode or verify call keeps the fork route of
// ops::moe_experts (staging beside the resident experts, staged misses landing in the cache) only
// while its top_k x columns jobs fit the slots, and that route covers calls of at most
// kForkColumns columns.
constexpr std::int32_t kStagingSlots = 32;
constexpr std::int32_t kForkColumns  = 8;

} // namespace

DeviceLayout ProgramImpl::plan_device(const ProgramOptions& o, const Config& config, std::int32_t public_tokens) {
    const TextConfig& c = config.text;
    if (o.max_concurrency == 0 || o.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("Glm5Next: max_concurrency must be in [1, " + std::to_string(kMaximumConcurrency) + "]");
    }
    if (o.max_context == 0 || o.max_context > c.context_ceiling()) {
        throw std::invalid_argument("Glm5Next: max_context must be in [1, " + std::to_string(c.context_ceiling()) + "]");
    }
    if (o.prefill_chunk == 0) { throw std::invalid_argument("Glm5Next: prefill_chunk must be positive"); }
    if (public_tokens <= 0 || static_cast<std::uint32_t>(public_tokens) > c.vocab_size) {
        throw std::invalid_argument("Glm5Next: the public token domain exceeds the vocabulary");
    }
    if (o.kv_cache != KvCacheStorage::BFloat16 && o.kv_cache != KvCacheStorage::Int8Group64) {
        throw std::invalid_argument("Glm5Next: the latent cache is BF16 or INT8-G64");
    }
    if (o.kv_capacity_tokens < o.max_context) {
        throw std::invalid_argument("Glm5Next: the KV capacity cannot hold one max_context request");
    }
    if (o.draft_tokens != 0 && (!config.dflash2 || o.causal_scoring || o.draft_tokens > 15)) {
        throw std::invalid_argument(
            "Glm5Next: DFlash2 drafts 1..15 tokens a round, with its drafter bound, "
            "while generating");
    }
    DeviceLayout d;
    d.lanes         = static_cast<std::int32_t>(o.max_concurrency);
    d.token_domain  = public_tokens;
    d.chunk         = static_cast<std::int32_t>(std::min(o.prefill_chunk, o.max_context));
    d.draft         = static_cast<std::int32_t>(o.draft_tokens);
    d.width         = d.draft + 1;
    // The widest call: a prefill chunk, a (verify) round of every lane or a scoring tile's head.
    d.columns = std::max({d.chunk, d.lanes * d.width, o.causal_scoring ? kScoreTileColumns : 1});
    d.pages_per_row = static_cast<std::int32_t>((o.max_context + kPagedKVPageSize - 1) / kPagedKVPageSize);
    d.kv_pages      = (o.kv_capacity_tokens + kPagedKVPageSize - 1) / kPagedKVPageSize;
    const auto lanes = static_cast<std::size_t>(d.lanes);

    // KDA: per KDA block, the q | k | v convolution windows and the FP32 recurrence, one slot a lane.
    {
        LayoutBuilder builder;
        d.kda       = plan_linear_attention_state_pool(builder, {.layers         = c.kda_layers,
                                                                 .conv_channels  = dim(3ULL * c.kda.width()),
                                                                 .conv_width     = dim(c.kda.conv_kernel - 1),
                                                                 .value_heads    = dim(c.kda.heads),
                                                                 .value_head_dim = dim(c.kda.head_dim),
                                                                 .key_head_dim   = dim(c.kda.head_dim),
                                                                 .slot_count     = d.lanes});
        d.kda_bytes = builder.finish(256);
    }
    // Per MLA block and lane the indexer's ring of its 64 most recent keys and gates (BF16).
    d.index_ring = align(2ULL * 2 * c.indexer.head_dim * 64 * lanes * c.mla_layers);
    // Per MLA block the latent (BF16, or INT8 codes and their FP16 group scales), then the
    // indexer's pooled keys.
    const bool int8 = o.kv_cache == KvCacheStorage::Int8Group64;
    for (std::uint32_t l = 0; l < c.mla_layers; ++l) {
        if (int8) {
            d.geometry.planes.push_back({DType::I8, dim(c.mla.kv_lora), 1});
            d.geometry.planes.push_back({DType::FP16, dim(c.mla.kv_lora / 64), 1});
        } else {
            d.geometry.planes.push_back({DType::BF16, dim(c.mla.kv_lora), 1});
        }
        d.geometry.planes.push_back({DType::BF16, dim(c.indexer.head_dim / c.indexer.pool), 1});
    }
    {
        LayoutBuilder builder;
        d.pool   = plan_device_kv_page_pool(builder, {.page_group_count = d.kv_pages, .geometry = d.geometry});
        d.tables = plan_kv_execution_tables(builder, {.logical_page_capacity = static_cast<std::uint32_t>(d.pages_per_row),
                                                      .table_rows            = d.lanes});
        d.kv     = builder.finish(256);
    }

    // Per-call io: ids, positions and logit columns of the widest call, slots and table rows.
    const auto columns = static_cast<std::size_t>(d.columns);
    std::size_t at     = 0;
    const auto take    = [&](std::size_t bytes) {
        const std::size_t begin = at;
        at += align(bytes);
        return begin;
    };
    d.io.ids       = take(4 * columns);
    d.io.positions = take(4 * columns);
    d.io.slots     = take(4 * lanes);
    d.io.rows      = take(4 * lanes);
    d.io.columns   = take(4 * columns);
    d.io.bytes     = at;
    d.vision_io    = config.vision ? align(4 * columns) : 0; // a call's media columns

    const auto vocab = static_cast<std::size_t>(c.vocab_size);
    const auto round     = lanes * static_cast<std::size_t>(d.width); // a round's logit columns
    d.logits32           = align(4 * vocab * round);
    d.logits16           = o.bf16_logits ? align(2 * vocab * round) : 0;
    d.token_counts   = align(4 * static_cast<std::size_t>(d.token_domain) * lanes);
    d.constraint     = align(constraint_bytes(d.lanes));
    d.grammar            = align(grammar_mask_bytes(d.token_domain, d.lanes, d.width));
    std::size_t sampling = ops::sampling_workspace_capacity_bytes(d.token_domain, 1, d.lanes);
    d.work               = execution::Forward::workspace_bytes(c, d.columns, dim(o.max_context));
    if (d.draft > 0) {
        // The drafter runs between Forward calls, in the same arena; the acceptance beside
        // sampling.
        const DFlash2Config& f = *config.dflash2;
        d.work   = std::max(d.work, execution::DFlash2::workspace_bytes(
                                        c, f, d.lanes, d.width, dim(f.window), o.max_context));
        sampling = std::max({sampling,
                             ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(
                                 d.token_domain, 1, d.draft, 1, d.lanes),
                             ops::speculative_accept_sparse_drafts_workspace_capacity_bytes(
                                 d.token_domain, ops::SpeculativeAcceptExecutionEnvelope{}, 1,
                                 d.draft, 1, d.lanes)});
        d.feature_columns = std::max(d.lanes * d.width, dim(f.window));
        d.features        = align(2ULL * f.feature_width(c.hidden_size) *
                                  static_cast<std::size_t>(d.feature_columns));
        LayoutBuilder ring;
        d.ring_layout = plan_cyclic_kv_cache(ring, f.layers, f.window, dim(f.kv_heads),
                                             dim(f.head_dim), d.lanes);
        d.ring        = ring.finish(256);
        // Per KDA block: the convolution input, k, v, g and beta of a round's columns.
        const std::size_t W = c.kda.width();
        d.records           = c.kda_layers * (align(2 * 3 * W * round) + 3 * align(2 * W * round) +
                                              align(2ULL * c.kda.heads * round));
        const std::size_t L = lanes, K = static_cast<std::size_t>(d.draft), R = round;
        const std::size_t C = ops::kSparseSpeculativeCandidates;
        auto& io            = d.dflash_io;
        auto take           = [at = std::size_t{0}](std::size_t words) mutable {
            const std::size_t begin = at;
            at += (words + 63) / 64 * 64;
            return begin;
        };
        io.anchors           = take(L);
        io.frontiers         = take(L);
        io.valid             = take(L);
        io.slots             = take(L);
        io.ids               = take(R);
        io.positions         = take(R);
        io.drafts            = take(K * L);
        io.candidates        = take(C * K * L);
        io.q                 = take(C * K * L);
        io.context_positions = take(f.window + 2); // then the call's count and lane
        io.context_count     = io.context_positions + f.window;
        io.context_slot      = io.context_count + 1;
        io.ones              = take(L);
        io.target            = take(R);
        io.extents           = take(L);
        io.lengths           = take(L);
        io.round_anchors     = take(L);
        io.licensed          = take(R + 2 * L); // then the licensed counts and the accepted drafts
        io.counts            = io.licensed + R;
        io.accepted          = io.counts + L;
        io.commit            = take(L);
        io.words             = take(0);
    }
    d.work += sampling;
    if (o.prefix_cache) {
        if (o.prefill_chunk % kPagedKVPageSize != 0) {
            throw std::invalid_argument("Glm5Next: the prefix cache snapshots only chunk-grid "
                                        "state (--prefill-chunk must be a "
                                        "multiple of 64)");
        }
        // The prefix cache's Device staging: the prompt-end captures' fixed slots (the packed state
        // image, min(lanes, 2) of them) and every lane's saved head residual. The image's layout
        // (and its size) comes from the state spec alone.
        prefix::DraftRingGeometry ring;
        if (d.draft > 0) {
            const auto lanes32 = static_cast<std::uint32_t>(d.lanes);
            ring.layers        = static_cast<std::uint32_t>(d.ring_layout.k.size());
            ring.k_bytes       = d.ring_layout.k.at(0).region.bytes / lanes32;
            ring.v_bytes       = d.ring_layout.v.at(0).region.bytes / lanes32;
        }
        const prefix::StateImageLayout image = prefix::plan_state_image(
            prefix::state_image_spec(c, ring, dim(c.residual_width()) * 2U));
        d.prefix_staging = (std::min<std::size_t>(lanes, 2U)) * image.image_bytes;
        d.head_saved     = align(dim(c.residual_width()) * 2U);
    }
    if (o.causal_scoring) {
        d.score_residuals = align(2ULL * c.residual_width() * o.max_context);
        d.score_logits    = align(4 * vocab * kScoreTileColumns);
        d.score_logits16  = o.bf16_logits ? align(2 * vocab * kScoreTileColumns) : 0;
        d.score_io        = align(4 * kScoreIoWords);
    }

    // The expert records: every bank is NVFP4 rg16 at the family's geometry.
    const std::uint64_t shape[] = {c.moe.experts, c.hidden_size, c.moe.intermediate};
    const std::uint64_t record_stride =
        weight_geometry(QType::NVFP4_MUL, QuantLayout::ExpertRg16, shape).record_stride;
    // The widest decode or verify call: a plain round of every lane, or a one-row verification
    // (only one-row rounds draft, execution.cpp).
    const std::int32_t round_columns = std::min(std::max(d.lanes, d.width), kForkColumns);
    d.staging_slots =
        std::max(kStagingSlots, static_cast<std::int32_t>(c.moe.top_k) * round_columns);
    d.staging       = static_cast<std::size_t>(d.staging_slots) * record_stride;

    auto& b     = d.bytes;
    b.kv        = d.kv;
    b.workspace = d.work;
    b.staging   = d.staging;
    b.state     = d.kda_bytes + d.index_ring;
    b.draft     = d.features + d.ring + d.records;
    b.prefix    = d.prefix_staging + d.head_saved * lanes;
    b.io = d.io.bytes + d.vision_io + d.logits32 + d.logits16 + align(4 * lanes) * 2 +
           4 * d.dflash_io.words + align(sizeof(ops::SamplingConfig) * lanes) + d.token_counts +
           d.constraint + d.grammar + d.score_residuals + d.score_logits + d.score_logits16 +
           d.score_io;
    b.residency            = offload::ExpertResidency::table_bytes(bank_geometry(c), d.columns);
    b.expert_record_stride = record_stride;
    b.max_frames           = offload::ExpertResidency::max_frames(bank_geometry(c));
    // One decode graph per batch size; with the drafter, a proposal, verify and context graph per
    // batch size and verify width.
    b.graph_bound = o.decode_graphs ? o.max_concurrency * (1U + 3U * o.draft_tokens) : 0U;
    return d;
}

void ProgramImpl::allocate(DeviceBuffer& buffer, std::size_t bytes, std::uint64_t& tally) {
    if (bytes == 0) { return; }
    buffer = DeviceBuffer(bytes);
    tally += bytes;
}

ProgramImpl::ProgramImpl(const execution::Parameters& parameters, DeviceContext& device, ProgramOptions options)
    : parameters_(parameters), c_(parameters.model.config().text), device_(device), options_(std::move(options)) {
    const auto public_tokens = static_cast<std::int32_t>(parameters_.model.resources().public_token_count);
    plan_          = plan_device(options_, parameters_.model.config(), public_tokens);
    token_domain_  = plan_.token_domain;
    vocab_         = dim(c_.vocab_size);
    width_         = dim(c_.residual_width());
    chunk_         = plan_.chunk;
    columns_       = plan_.columns;
    beside_decode_chunk_ = options_.prefill_chunk_beside_decode;
    lanes_.resize(options_.max_concurrency);
    const auto lanes = static_cast<std::size_t>(plan_.lanes);

    std::uint64_t allocated = 0;
    allocate(kda_backing_, plan_.kda_bytes, allocated);
    kda_backing_.fill(0);
    kda_ = std::make_unique<LinearAttentionStatePool>(DeviceSpan{kda_backing_.p, kda_backing_.bytes}, plan_.kda);
    allocate(index_ring_, plan_.index_ring, allocated);
    index_ring_.fill(0);
    allocate(kv_backing_, plan_.kv, allocated);
    kv_backing_.fill(0);
    const DeviceSpan kv_span{kv_backing_.p, kv_backing_.bytes};
    pool_   = std::make_unique<DeviceKVPagePool>(kv_span, plan_.pool);
    tables_ = std::make_unique<KVExecutionTablePool>(kv_span, plan_.tables, *pool_);

    if (plan_.draft > 0) { allocate_dflash2(allocated); }
    if (plan_.head_saved != 0) { allocate(head_saved_, plan_.head_saved * lanes, allocated); }
    allocate(io_device_, plan_.io.bytes, allocated);
    io_host_ = PinnedHostBuffer(plan_.io.bytes);
    allocate(vision_io_device_, plan_.vision_io, allocated);
    if (plan_.vision_io != 0) { vision_io_host_ = PinnedHostBuffer(plan_.vision_io); }
    allocate(logits32_, plan_.logits32, allocated);
    allocate(logits16_, plan_.logits16, allocated);
    allocate(sampled_, align(4 * lanes), allocated);
    allocate(sample_pos_, align(4 * lanes), allocated);
    allocate(configs_, align(sizeof(ops::SamplingConfig) * lanes), allocated);
    allocate(token_counts_, plan_.token_counts, allocated);
    token_counts_.fill(0);
    host_sampled_ = PinnedHostBuffer(4 * lanes);
    host_configs_ = PinnedHostBuffer(sizeof(ops::SamplingConfig) * lanes);
    allocate(constraint_, plan_.constraint, allocated);
    allocate(grammar_masks_, plan_.grammar, allocated);
    grammar_host_ = PinnedHostBuffer(grammar_mask_bytes(token_domain_, plan_.lanes, plan_.width));
    constraint_host_.assign(lanes, -1);
    if (options_.causal_scoring) {
        allocate(score_residuals_, plan_.score_residuals, allocated);
        allocate(score_logits_, plan_.score_logits, allocated);
        allocate(score_logits16_, plan_.score_logits16, allocated);
        allocate(score_io_, plan_.score_io, allocated);
        score_host_ = PinnedHostBuffer(plan_.score_io);
    }
    std::memset(nonfinite_flags_.data(), 0, nonfinite_flags_.size());
    std::memset(device_error_.data(), 0, device_error_.size());

    execution::ForwardState state{.kda = kda_.get()};
    execution::ForwardKV kv;
    kv.block_tables   = tables_->matrix();
    std::size_t plane = 0;
    const std::int32_t index_dim = dim(c_.indexer.head_dim);
    const std::size_t ring_elements = static_cast<std::size_t>(index_dim) * 64 * lanes;
    auto* ring = static_cast<std::uint16_t*>(index_ring_.p);
    for (std::uint32_t l = 0; l < c_.mla_layers; ++l) {
        execution::MlaLayerCache layer;
        layer.latent.pages = pool_->plane(plane++);
        if (options_.kv_cache == KvCacheStorage::Int8Group64) {
            layer.latent.scales = pool_->plane(plane++);
        }
        layer.index.k_recent = Tensor(ring + (2 * l) * ring_elements, DType::BF16, {index_dim, 64, plan_.lanes});
        layer.index.z_recent = Tensor(ring + (2 * l + 1) * ring_elements, DType::BF16, {index_dim, 64, plan_.lanes});
        layer.index.pooled_pages = pool_->plane(plane++);
        kv.layers.push_back(layer);
    }
    if (plane != plan_.geometry.planes.size()) {
        throw std::logic_error("Glm5Next: the KV planes do not match the page geometry");
    }
    kv_geometry_ = prefix::kv_page_geometry(c_, options_.kv_cache);
    work_ = std::make_unique<WorkspaceArena>(plan_.work);
    allocated += plan_.work;

    execution::ForwardExperts experts;
    create_experts(experts, allocated);
    forward_ = std::make_unique<execution::Forward>(parameters_, device_, *work_, state, std::move(kv),
                                                    std::move(experts), dim(options_.max_context));
    if (plan_.draft > 0) {
        std::vector<CyclicKVCacheLayerView> ring;
        for (std::uint32_t l = 0; l < ring_->layer_count(); ++l) {
            ring.push_back(ring_->layer_view(l));
        }
        drafter_ = std::make_unique<execution::DFlash2>(
            parameters_, device_, *work_, std::move(ring), options_.max_context,
            static_cast<std::uint32_t*>(device_error_.data()));
    }
    create_prefix_cache(allocated);
    if (allocated != plan_.bytes.fixed_bytes()) {
        throw std::logic_error("Glm5Next: the Program's device allocations (" +
                               std::to_string(allocated) + " bytes) differ from its device plan (" +
                               std::to_string(plan_.bytes.fixed_bytes()) + " bytes)");
    }
    device_.synchronize();
}

ProgramImpl::~ProgramImpl() {
    try {
        device_.synchronize();
    } catch (...) {}
    forward_.reset();
    placement_.reset();
    if (overlap_stream_ != nullptr) {
        for (cudaEvent_t event : overlap_events_) {
            if (event != nullptr) { (void)cudaEventDestroy(event); }
        }
        (void)cudaStreamDestroy(overlap_stream_);
    }
}

void ProgramImpl::diagnostic(const std::string& text, DiagnosticLevel level) const {
    runtime::publish_diagnostic(options_.diagnostics, level, "%s", text.c_str());
}

} // namespace infernix::models::glm5_next::detail
