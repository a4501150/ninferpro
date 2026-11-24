// ProgramImpl construction and the static device plan: every fixed device allocation, the
// forward state, the expert tiers and services, and the plan's byte counts.

#include "models/qwen4_exp/program/program_impl.h"

namespace infernix::models::qwen4_exp::detail {

DeviceLayout ProgramImpl::plan_device(const ProgramOptions& o, const Config& config, std::int32_t public_tokens) {
    const TextConfig& c = config.text;
    if (o.max_concurrency == 0 || o.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("Qwen4Exp: max_concurrency must be in [1,8]");
    }
    if (o.max_context == 0 || o.prefill_chunk == 0) {
        throw std::invalid_argument("Qwen4Exp: max_context and prefill_chunk must be nonzero");
    }
    if (o.max_context > c.context_ceiling()) {
        throw std::invalid_argument("Qwen4Exp: max_context " + std::to_string(o.max_context) + " exceeds the " +
                                    std::to_string(c.context_ceiling()) +
                                    " positions of the model (native positions x the YaRN factor)");
    }
    if (o.ngram_draft_tokens > 15 ||
        (o.ngram_draft_tokens > 0 && (o.ngram_min_match < 4 || o.ngram_min_match > 64))) {
        throw std::invalid_argument("Qwen4Exp: n-gram drafts must be 0..15 with a minimum match of 4..64");
    }
    if (config.mtp && (o.mtp_draft_tokens == 0 || o.mtp_draft_tokens > 7)) {
        throw std::invalid_argument("Qwen4Exp: MTP needs 1..7 draft tokens");
    }
    // Verification widths up to 8 keep the dense routes column-invariant (design 11.1).
    if (config.dflash2 && (o.dflash2_draft_tokens == 0 || o.dflash2_draft_tokens > 7 || config.mtp)) {
        throw std::invalid_argument("Qwen4Exp: DFlash2 needs 1..7 draft tokens and no MTP drafter");
    }
    // Its drafts are verified with their laws (sparse acceptance, which serves one token domain).
    if (config.dflash2 &&
        !sparse_acceptance(public_tokens, 1 + std::max(static_cast<std::int32_t>(o.ngram_draft_tokens),
                                                      static_cast<std::int32_t>(o.dflash2_draft_tokens)),
                           static_cast<std::int32_t>(o.max_concurrency))) {
        throw std::invalid_argument("Qwen4Exp: DFlash2 needs the sparse acceptance's token domain");
    }
    DeviceLayout d;
    d.lanes         = static_cast<std::int32_t>(o.max_concurrency);
    d.token_domain  = public_tokens;
    d.mtp           = config.mtp;
    d.head_rows     = config.proposal_rows != 0 ? dim(config.proposal_rows) : dim(c.vocab_size);
    d.chunk         = static_cast<std::int32_t>(std::min(o.prefill_chunk, o.max_context));
    // Without free cache resume every prefill call starts on the GDN prefill's 16-token grid.
    if (!o.free_cache_resume) {
        constexpr auto grid = static_cast<std::int32_t>(kResumeGrid);
        d.chunk             = std::max(grid, d.chunk / grid * grid);
    }
    d.mtp_k         = d.mtp ? static_cast<std::int32_t>(o.mtp_draft_tokens) : 0;
    d.dflash2       = config.dflash2.has_value();
    d.dflash_k      = d.dflash2 ? static_cast<std::int32_t>(o.dflash2_draft_tokens) : 0;
    d.dflash2_config = config.dflash2;
    d.max_width     = 1 + std::max({static_cast<std::int32_t>(o.ngram_draft_tokens), d.mtp_k, d.dflash_k});
    d.columns       = std::max(d.chunk, d.lanes * d.max_width);
    d.pages_per_row = (dim(o.max_context) + kPagedKVPageSize - 1) / kPagedKVPageSize;
    const auto kv_tokens = o.kv_capacity_tokens != 0 ? o.kv_capacity_tokens : o.max_context * o.max_concurrency;
    d.kv_pages      = (kv_tokens + kPagedKVPageSize - 1) / kPagedKVPageSize;
    const std::int32_t lanes = d.lanes, W = d.max_width;
    // One copy-on-write page per lane, so a whole-extent resume always fits (design §15.4).
    if (o.prefix_cache) { d.kv_pages += static_cast<std::uint32_t>(lanes); }
    const std::size_t width = c.residual_width(), di = c.qsa.index_head_dim, r = c.qsa.compress_ratio;

    // Recurrent state: one slot per lane; the PLE history; one QSA tail slab per attention
    // layer, and one more for the MTP block.
    LayoutBuilder state_builder;
    d.gdn       = plan_linear_attention_state_pool(state_builder, {
                                                                .layers         = c.gdn_layers,
                                                                .conv_channels  = dim(c.gdn.conv_channels()),
                                                                .conv_width     = dim(c.gdn.conv_kernel - 1),
                                                                .value_heads    = dim(c.gdn.value_heads),
                                                                .value_head_dim = dim(c.gdn.value_head_dim),
                                                                .key_head_dim   = dim(c.gdn.key_head_dim),
                                                                .slot_count     = lanes,
                                                            });
    d.gdn_bytes = state_builder.finish(256);
    d.ple       = width * c.ple.conv_span() * lanes * 2;
    d.kv_layers = c.attention_layers + (d.mtp ? 1U : 0U);
    d.tails     = d.kv_layers * di * (r - 1) * lanes * 2;
    d.window    = prefix::kv_window_geometry(c, o.kv_cache, d.mtp, static_cast<std::uint32_t>(lanes)).bytes();

    // Paged KV: one page group holds 64 positions of every attention layer.
    // The page group's planes, in the order the prefix cache's records also use.
    const prefix::KvPageGeometry kv_geometry = prefix::kv_page_geometry(c, o.kv_cache, d.mtp);
    LayoutBuilder kv_builder;
    d.pool   = plan_device_kv_page_pool(kv_builder, {.page_group_count = d.kv_pages, .geometry = kv_geometry.geometry});
    d.tables = plan_kv_execution_tables(
        kv_builder, {.logical_page_capacity = static_cast<std::uint32_t>(d.pages_per_row), .table_rows = lanes});
    d.kv = kv_builder.finish(256);
    // Pages [0, n) are a prefix of each page-major plane, so growth maps whole chunks at each plane's end.
    // A resident placement holds every expert in the frame pool: the KV pool is fully mapped
    // at startup instead, so its growth never takes frames (grow_kv is inert).
    d.kv_elastic = !o.resident_experts && d.pool.spec.geometry.device_plane_order == PagedKVPlaneOrder::PageMajor;
    d.kv_base_pages = std::min<std::uint32_t>(d.kv_pages, kKvBaseTokens / kPagedKVPageSize);

    // Per-call inputs, logits and sampling counts.
    const std::size_t heads     = NgramHash(c.ple.ngram, 0).heads();
    const std::size_t row_bytes = (o.ngram_table ? *o.ngram_table : c.ple.table).row_bytes;
    const std::size_t columns   = static_cast<std::size_t>(d.columns);
    const std::size_t mtp_subchunks = (columns + 1 + execution::kMtpChunkColumns - 1) / execution::kMtpChunkColumns;
    d.io.ids            = 0;
    d.io.positions      = align(d.io.ids + 4ULL * columns);
    d.io.rope           = align(d.io.positions + 4ULL * columns);
    d.io.block_rope     = align(d.io.rope + 12ULL * columns);
    d.io.slots          = align(d.io.block_rope + 12ULL * lanes);
    d.io.rows           = align(d.io.slots + 4ULL * lanes);
    d.io.columns        = align(d.io.rows + 4ULL * lanes);
    d.io.gate           = align(d.io.columns + 4ULL * columns);
    d.io.divisor        = d.io.gate + 4; // I32: plain decode's one-column CPU divisor this round
    d.io.ngram          = align(d.io.divisor + 4);
    d.io.mtp_ids        = align(d.io.ngram + row_bytes * heads * columns);
    d.io.mtp_cells      = align(d.io.mtp_ids + 4ULL * (columns + 1));
    d.io.mtp_rope       = align(d.io.mtp_cells + 4ULL * (columns + 1));
    d.io.mtp_block_rope = align(d.io.mtp_rope + 12ULL * (columns + 1));
    d.io.vision_columns     = align(d.io.mtp_block_rope + 12ULL * mtp_subchunks);
    d.io.mtp_vision_columns = align(d.io.vision_columns + 4ULL * columns);
    d.io.bytes              = align(d.io.mtp_vision_columns + 4ULL * (columns + 1));
    if (d.io.block_rope + 12ULL * lanes > d.io.ngram || d.io.columns + 4ULL * columns > d.io.ngram) {
        throw std::logic_error("Qwen4Exp: decode io must precede the n-gram rows");
    }
    // The head's FP32 logits, which the sampler, verification, constraints and the prompt
    // readout read; with BF16 logits their BF16 rounding beside them.
    d.logits32     = sizeof(float) * c.vocab_size * lanes * W;
    d.logits16     = o.bf16_logits ? 2ULL * c.vocab_size * lanes * W : 0;
    d.token_counts = 4ULL * static_cast<std::size_t>(d.token_domain) * lanes;
    if (o.causal_scoring) {
        d.score_residuals = align(2ULL * width * o.max_context);
        d.score_logits    = align(sizeof(float) * c.vocab_size * kScoreTileColumns);
        d.score_logits16  = o.bf16_logits ? align(2ULL * c.vocab_size * kScoreTileColumns) : 0;
        d.score_io        = align(4ULL * kScoreIoWords);
    }

    // The workspace arena (the forward pass, sampling, draft acceptance and the drafter) covers
    // decode, verification and calls of up to kStaticWorkColumns columns; a wider prefill call
    // swaps in an arena of the chunk's size lent from expert frames (design §13.1, VRAM item 1).
    d.static_columns = std::max(lanes * W, std::min(d.chunk, kStaticWorkColumns));
    d.work           = workspace_bytes(c, o, d, lanes, d.static_columns);
    d.wide_work      = d.chunk > d.static_columns ? workspace_bytes(c, o, d, lanes, d.chunk) : 0;

    // Verification (W > 1): GDN replay records, PLE and QSA records, the round's I32 arrays.
    if (W > 1) {
        LayoutBuilder records_builder;
        d.records       = ops::plan_gdn_replay_records(records_builder,
                                                       {.layers          = dim(c.gdn_layers),
                                                        .record_capacity = lanes,
                                                        .width           = W,
                                                        .conv_channels   = dim(c.gdn.conv_channels()),
                                                        .qk_heads        = dim(c.gdn.key_heads),
                                                        .value_heads     = dim(c.gdn.value_heads),
                                                        .key_dim         = dim(c.gdn.key_head_dim),
                                                        .value_dim       = dim(c.gdn.value_head_dim)});
        d.records_bytes = records_builder.finish(256);
        d.ple_records   = 2ULL * width * W * lanes;
        d.qsa_records   = 2ULL * di * W * lanes * c.attention_layers;
        const std::size_t cells = static_cast<std::size_t>(lanes) * W;
        std::size_t at          = 0;
        const auto take         = [&](std::size_t words) {
            const std::size_t begin = at;
            at += (words + 63) / 64 * 64;
            return begin;
        };
        d.spec.target   = take(cells);
        d.spec.drafts   = take(cells);
        d.spec.extents  = take(lanes);
        d.spec.lengths  = take(lanes);
        d.spec.anchors  = take(lanes);
        d.spec.licensed = take(cells);
        d.spec.counts   = take(lanes);
        d.spec.accepted = take(lanes);
        d.spec.commit   = take(lanes);
        d.spec.words    = at;
    }

    // The MTP drafter (design 11.2): residual columns, the saved and chain columns, its QSA
    // records, the HC ones and its io.
    if (d.mtp) {
        d.mtp_columns      = std::max(lanes * W, std::min(d.chunk, 512));
        d.mtp_column_bytes = 2ULL * width;
        d.mtp_ones         = sizeof(float) * c.hc.streams * static_cast<std::size_t>(d.mtp_columns);
        const auto k       = static_cast<std::size_t>(d.mtp_k);
        d.mtp_io.ids              = 0;
        d.mtp_io.cells            = d.mtp_io.ids + lanes;
        d.mtp_io.rope_cells       = d.mtp_io.cells + lanes * k;
        d.mtp_io.block_rope_cells = d.mtp_io.rope_cells + 3 * lanes * k;
        d.mtp_io.drafts           = d.mtp_io.block_rope_cells + 3 * lanes * k;
        d.mtp_io.up_ids    = d.mtp_io.drafts + lanes * k;
        d.mtp_io.gather    = d.mtp_io.up_ids + static_cast<std::size_t>(lanes) * W;
        d.mtp_io.up_cells  = d.mtp_io.gather + lanes;
        d.mtp_io.logprobs  = d.mtp_io.up_cells + static_cast<std::size_t>(lanes) * W;
        // Sampled drafts (sampled-mtp-drafts; planned whether or not it runs, a few KiB).
        const auto ranks      = static_cast<std::size_t>(kSampledDraftRanks);
        const auto candidates = static_cast<std::size_t>(ops::kSparseSpeculativeCandidates);
        const auto verified   = static_cast<std::size_t>(lanes) * static_cast<std::size_t>(W - 1);
        d.mtp_io.sample_rows       = d.mtp_io.logprobs + lanes * k;
        d.mtp_io.sample_logprobs   = d.mtp_io.sample_rows + ranks * lanes * k;
        d.mtp_io.draft_candidates  = d.mtp_io.sample_logprobs + ranks * lanes * k;
        d.mtp_io.draft_q           = d.mtp_io.draft_candidates + candidates * lanes * k;
        d.mtp_io.verify_candidates = d.mtp_io.draft_q + candidates * lanes * k;
        d.mtp_io.verify_q          = d.mtp_io.verify_candidates + candidates * verified;
        d.mtp_io.words             = d.mtp_io.verify_q + candidates * verified;
    }

    // The DFlash2 drafter: the target features of every verification column or a call's last
    // window, every lane's context ring and the drafter's io.
    if (d.dflash2) {
        const DFlash2Config& f = *config.dflash2;
        d.dflash_round_columns = lanes * W;
        d.dflash_columns       = d.dflash_round_columns + dim(f.window);
        d.dflash_features      = align(2ULL * f.feature_width(c.hidden_size) * static_cast<std::size_t>(d.dflash_columns));
        LayoutBuilder ring_builder;
        d.dflash_ring_layout = plan_cyclic_kv_cache(ring_builder, f.layers, f.window, dim(f.attention.kv_heads),
                                                    dim(f.attention.head_dim), lanes);
        d.dflash_ring        = ring_builder.finish(256);
        const auto L = static_cast<std::size_t>(lanes), K = static_cast<std::size_t>(d.dflash_k);
        const auto C = static_cast<std::size_t>(ops::kSparseSpeculativeCandidates);
        auto take = [at = std::size_t{0}](std::size_t words) mutable {
            const std::size_t begin = at;
            at += (words + 63) / 64 * 64;
            return begin;
        };
        d.dflash_io.anchors           = take(L);
        d.dflash_io.frontiers         = take(L);
        d.dflash_io.valid             = take(L);
        d.dflash_io.slots             = take(L);
        d.dflash_io.ids               = take((K + 1) * L);
        d.dflash_io.positions         = take((K + 1) * L);
        d.dflash_io.drafts            = take(K * L);
        d.dflash_io.confidence        = take(K * L);
        d.dflash_io.candidates        = take(C * K * L);
        d.dflash_io.q                 = take(C * K * L);
        d.dflash_io.verify_candidates = take(C * static_cast<std::size_t>(W - 1) * L);
        d.dflash_io.verify_q          = take(C * static_cast<std::size_t>(W - 1) * L);
        d.dflash_io.context_positions = take(f.window);
        d.dflash_io.context_count     = take(1);
        d.dflash_io.context_slot      = take(1);
        d.dflash_io.words             = take(0);
    }

    // Staging slots for each layer call's misses (design 8.5), one expert record each.
    const std::uint64_t shape[] = {c.moe.experts, c.hidden_size, c.moe.intermediate};
    // The largest record (staging, stream-ring and RAM slots): the NVFP4 record's, or an exact
    // bank's largest per-layer record; and the expert cache's average frame (its sizing unit).
    d.record_stride = c.expert_frame_stride != 0
                          ? c.expert_frame_stride
                          : weight_geometry(QType::NVFP4_MUL, QuantLayout::ExpertRg16, shape).record_stride;
    d.frame_bytes   = c.expert_frame_bytes != 0 ? c.expert_frame_bytes : d.record_stride;
    // Without the SSD tier, a verification call's misses (up to top_k per column) stage in one
    // pass, so it keeps the fork route (ops::moe_experts): eight columns (seven drafts) route up
    // to 80 experts, and the larger slot count measured faster without the tier; with the tier the
    // larger staging and its ring took RAM and frames from the cache, so it keeps 64 (the
    // per-count measurements are design §8.5).
    d.staging_slots = o.expert_ram_bytes != 0
                          ? kStagingSlots
                          : std::clamp(static_cast<std::int32_t>(c.moe.top_k) * d.max_width, kStagingSlots,
                                       kMaxStagingSlots);
    d.staging       = static_cast<std::size_t>(d.staging_slots) * d.record_stride;

    auto& b     = d.bytes;
    b.kv_pages  = d.kv_pages;
    b.kv        = d.kv_elastic ? kv_mapped_bytes(d, d.kv_base_pages) : d.kv;
    b.kv_max    = d.kv;
    b.workspace = d.work;
    b.staging   = d.staging;
    b.state     = d.gdn_bytes + d.ple + d.tails + d.window + d.records_bytes + d.ple_records + d.qsa_records +
              (d.mtp ? d.mtp_column_bytes * lanes * (W + 2) + 2ULL * di * W * lanes
                     : 2ULL * width * lanes) + // the saved head residual (the zero-token resume)
              d.dflash_features + d.dflash_ring;
    b.io = d.io.bytes + d.logits32 + d.logits16 + 4ULL * lanes + 4ULL * lanes +
           sizeof(ops::SamplingConfig) * lanes + d.token_counts + 4ULL * d.spec.words + d.mtp_ones +
           4ULL * d.mtp_io.words + 4ULL * d.dflash_io.words + kGateStatsBytes * lanes +
           (o.ngram_table && o.ngram_table->fp8_scale ? sizeof(std::uint16_t) : 0) + constraint_bytes(lanes, W) +
           grammar_mask_bytes(d.token_domain, lanes, W) + d.score_residuals + d.score_logits + d.score_io;
    b.residency            = offload::ExpertResidency::table_bytes(bank_geometry(c), d.columns);
    b.expert_record_stride = d.record_stride;
    b.expert_frame_bytes   = d.frame_bytes;
    b.max_frames           = offload::ExpertResidency::max_frames(bank_geometry(c));
    b.graph_bound          = graph_bound(o.max_concurrency, static_cast<std::uint32_t>(W),
                                         static_cast<std::uint32_t>(d.mtp_k), o.sampled_mtp_drafts && d.mtp,
                                         static_cast<std::uint32_t>(d.dflash_k));
    return d;
}

std::vector<std::uint8_t> ProgramImpl::kv_chunks(const DeviceLayout& d, std::uint32_t pages) {
    std::vector<std::uint8_t> need((d.kv + kKvChunkBytes - 1) / kKvChunkBytes, 0);
    const auto mark = [&](std::size_t offset, std::size_t bytes) {
        if (bytes == 0) { return; }
        for (std::size_t c = offset / kKvChunkBytes; c <= (offset + bytes - 1) / kKvChunkBytes; ++c) { need[c] = 1; }
    };
    for (const auto& plane : d.pool.planes) { // page-major: the page is the outermost dimension
        const auto total = static_cast<std::size_t>(d.pool.spec.page_group_count);
        mark(plane.storage.region.offset, plane.storage.region.bytes / total * pages);
    }
    mark(d.tables.block_tables.region.offset, d.tables.block_tables.region.bytes);
    return need;
}

std::size_t ProgramImpl::kv_mapped_bytes(const DeviceLayout& d, std::uint32_t pages) {
    const auto need = kv_chunks(d, pages);
    return static_cast<std::size_t>(std::count(need.begin(), need.end(), std::uint8_t{1})) * kKvChunkBytes;
}

ops::offloaded_moe::WideA16 ProgramImpl::wide_a16(const ProgramOptions& o) noexcept {
    if (o.fast_paths.expert_a4_residual) { return ops::offloaded_moe::WideA16::kA4Residual; }
    return ops::has_route(o.fast_paths.dense_routes, ops::LinearRoutes::GgmlBf16Wide) ? ops::offloaded_moe::WideA16::kGgmlBf16
                                                                                      : ops::offloaded_moe::WideA16::kExact;
}

std::size_t ProgramImpl::workspace_bytes(const TextConfig& c, const ProgramOptions& o, const DeviceLayout& d,
                                         std::int32_t lanes, std::int32_t columns) {
    const std::int32_t W = d.max_width;
    std::size_t bytes =
        execution::Forward::workspace_bytes(c, std::max(columns, lanes * W), dim(o.max_context), o.kv_cache,
                                            wide_a16(o)) +
        ops::sampling_workspace_capacity_bytes(d.token_domain, 1, lanes);
    if (W > 1) {
        std::size_t accept =
            ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(d.token_domain, 1, W - 1, 1, lanes);
        if (((o.sampled_mtp_drafts && d.mtp) || d.dflash2) && sparse_acceptance(d.token_domain, W, lanes)) {
            accept = std::max(accept, ops::speculative_accept_sparse_drafts_workspace_capacity_bytes(
                                          d.token_domain, ops::SpeculativeAcceptExecutionEnvelope{}, 1, W - 1, 1,
                                          lanes));
        }
        bytes += accept;
    }
    if (d.mtp) {
        bytes += execution::Forward::mtp_workspace_bytes(c, std::max(lanes * W, std::min(d.chunk, 512)), lanes,
                                                         d.head_rows, dim(o.max_context), o.kv_cache);
    }
    // The DFlash2 drafter runs between Forward calls, in the same arena.
    if (d.dflash2) {
        bytes = std::max(bytes, execution::DFlash2::workspace_bytes(c, *d.dflash2_config, lanes, d.dflash_k + 1,
                                                                    dim(d.dflash2_config->window), o.max_context));
    }
    return bytes;
}

bool ProgramImpl::sparse_acceptance(std::int32_t token_domain, std::int32_t W, std::int32_t lanes) {
    try {
        (void)ops::speculative_accept_sparse_drafts_workspace_capacity_bytes(
            token_domain, ops::SpeculativeAcceptExecutionEnvelope{}, 1, W - 1, 1, lanes);
        return true;
    } catch (const std::invalid_argument&) {
        return false;
    }
}

void ProgramImpl::allocate(DeviceBuffer& buffer, std::size_t bytes, std::uint64_t& tally) {
    buffer = DeviceBuffer(bytes);
    tally += bytes;
}

std::string ProgramImpl::spill_message(std::uint64_t bytes) {
    return std::to_string(bytes >> 20) +
           " MiB of Infernix's VRAM allocations were placed in shared system memory by the driver's sysmem "
           "fallback: free VRAM, or lower --max-context, --max-concurrency or --prefill-chunk. Setting the NVIDIA "
           "Control Panel 'CUDA - Sysmem Fallback Policy' to 'Prefer No Sysmem Fallback' for this program makes "
           "this fail at allocation instead.";
}

ProgramImpl::ProgramImpl(const execution::Parameters& parameters, DeviceContext& device, ProgramOptions options) : parameters_(parameters), device_(device), c_(parameters.model.config().text), options_(std::move(options)),
      volume_(options_.ngram_volume, ngram_table(), options_.ngram_in_ram), hash_(c_.ple.ngram, 0),
      plan_(plan_device(options_, parameters.model.config(), dim(parameters.model.resources().public_token_count))) {
    const auto& head = parameters_.draft_head;
    mtp_             = parameters_.mtp.has_value();
    dflash2_         = parameters_.dflash2.has_value();
    if (dflash2_ != plan_.dflash2) { throw std::logic_error("Qwen4Exp: the bound DFlash2 drafter differs from its configuration"); }
    if (mtp_ != plan_.mtp || (mtp_ && (head.rows ? dim(head.rows->weight.n) : dim(c_.vocab_size)) != plan_.head_rows)) {
        throw std::logic_error("Qwen4Exp: the bound drafter differs from its configuration");
    }
    const std::int32_t lanes = plan_.lanes;
    vocab_         = dim(c_.vocab_size);
    token_domain_  = plan_.token_domain;
    chunk_         = plan_.chunk;
    beside_decode_chunk_ = options_.prefill_chunk_beside_decode;
    mtp_k_         = plan_.mtp_k;
    dflash_k_      = plan_.dflash_k;
    max_width_     = plan_.max_width;
    columns_       = plan_.columns;
    pages_per_row_ = plan_.pages_per_row;
    kv_pages_      = plan_.kv_pages;
    width_         = dim(c_.residual_width());
    span_          = dim(c_.ple.conv_span());
    di_            = dim(c_.qsa.index_head_dim);
    r_             = dim(c_.qsa.compress_ratio);
    draft_min_logprob_ = options_.draft_min_p > 0.0 ? std::log(options_.draft_min_p)
                                                    : -std::numeric_limits<double>::infinity();
    sampled_drafts_ = options_.sampled_mtp_drafts && plan_.mtp && max_width_ > 1 &&
                      sparse_acceptance(plan_.token_domain, max_width_, plan_.lanes);
    if (options_.sampled_mtp_drafts && plan_.mtp && !sampled_drafts_) {
        diagnostic("sampled MTP drafts need the sparse acceptance's token domain; drafts stay one-hot",
                   DiagnosticLevel::Warning);
    }

    // Every fixed allocation below is checked against the plan and against device free memory
    // (an allocation the driver placed in system memory fails startup, design §9.7).
    vram_ = open_vram_budget_source(device_.device, options_.vram_past_budget);
    SpillGuard guard(*vram_);
    guard.begin();
    std::uint64_t allocated = 0;

    allocate(state_backing_, plan_.gdn_bytes, allocated);
    state_backing_.fill(0);
    gdn_ = std::make_unique<LinearAttentionStatePool>(DeviceSpan{state_backing_.p, state_backing_.bytes}, plan_.gdn);
    allocate(ple_backing_, plan_.ple, allocated);
    ple_backing_.fill(0);
    allocate(tails_backing_, plan_.tails, allocated);
    tails_backing_.fill(0);
    // The zero-token resume's saved head residual: one residual column per lane (the MTP saved
    // column's counterpart for a non-MTP Program); the prompt-end snapshot carries it.
    if (!mtp_) { allocate(head_saved_, 2ULL * static_cast<std::size_t>(width_) * lanes, allocated); }
    window_geometry_ = prefix::kv_window_geometry(c_, options_.kv_cache, mtp_,
                                                  static_cast<std::uint32_t>(options_.max_concurrency));
    if (window_geometry_.present()) {
        allocate(window_backing_, plan_.window, allocated);
        window_backing_.fill(0); // zero tags never match a row's (tags are odd)
    }

    kv_geometry_ = prefix::kv_page_geometry(c_, options_.kv_cache, mtp_);
    if (plan_.kv_elastic) {
        kv_vmm_  = std::make_unique<VmmRange>(device_.device, plan_.kv, kKvChunkBytes);
        kv_span_ = DeviceSpan{kv_vmm_->base(), plan_.kv};
        if (!map_kv(plan_.kv_base_pages)) { throw std::runtime_error("Qwen4Exp: the KV pool's base pages do not fit"); }
        allocated += kv_vmm_->mapped_bytes();
    } else {
        allocate(kv_backing_, plan_.kv, allocated);
        kv_backing_.fill(0);
        kv_span_ = DeviceSpan{kv_backing_.p, kv_backing_.bytes};
    }
    pool_   = std::make_unique<DeviceKVPagePool>(kv_span_, plan_.pool);
    tables_ = std::make_unique<KVExecutionTablePool>(kv_span_, plan_.tables, *pool_);
    if (plan_.kv_elastic) { pool_->set_backed_pages(plan_.kv_base_pages); }

    io_layout_ = plan_.io;
    allocate(io_device_, io_layout_.bytes, allocated);
    io_host_ = PinnedHostBuffer(io_layout_.bytes);
    allocate(logits32_, plan_.logits32, allocated);
    allocate(logits16_, plan_.logits16, allocated);
    allocate(sampled_, 4ULL * lanes, allocated);
    allocate(sample_pos_, 4ULL * lanes, allocated);
    allocate(configs_, sizeof(ops::SamplingConfig) * lanes, allocated);
    allocate(token_counts_, plan_.token_counts, allocated);
    token_counts_.fill(0);
    host_sampled_ = PinnedHostBuffer(4ULL * lanes);
    host_configs_ = PinnedHostBuffer(sizeof(ops::SamplingConfig) * lanes);
    allocate(gate_stats_, kGateStatsBytes * lanes, allocated);
    gate_stats_.fill(0);
    // Another FP8 volume's per-tensor scale (resolve_ngram_volume): the artifact's ngram_scale serves only
    // its own table.
    if (const auto scale = ngram_table().fp8_scale) {
        allocate(ngram_scale_, sizeof(*scale), allocated);
        ngram_scale_.copy_from_host(&*scale, sizeof(*scale), 0);
    }
    allocate(constraint_, constraint_bytes(lanes, max_width_), allocated);
    allocate(grammar_masks_, grammar_mask_bytes(token_domain_, lanes, max_width_), allocated);
    grammar_host_ = PinnedHostBuffer(grammar_mask_bytes(token_domain_, lanes, max_width_));
    if (options_.causal_scoring) {
        allocate(score_residuals_, plan_.score_residuals, allocated);
        allocate(score_logits_, plan_.score_logits, allocated);
        allocate(score_logits16_, plan_.score_logits16, allocated);
        allocate(score_io_, plan_.score_io, allocated);
        score_host_ = PinnedHostBuffer(plan_.score_io);
    }
    constraint_host_.assign(static_cast<std::size_t>(lanes * max_width_), -1);
    publish_pinned_word(gate_ready(), 0); // pinned memory starts undefined; 0 is never a round's word

    execution::ForwardState state;
    state.gdn      = gdn_.get();
    state.ple_conv = Tensor(ple_backing_.p, DType::BF16, {width_, span_, lanes});
    for (std::uint32_t l = 0; l < c_.attention_layers; ++l) {
        state.qsa_tails.push_back(Tensor(static_cast<std::byte*>(tails_backing_.p) +
                                             static_cast<std::size_t>(l) * di_ * (r_ - 1) * lanes * 2,
                                         DType::BF16, {di_, r_ - 1, lanes}));
    }
    const auto layout   = paged_kv_storage_layout(options_.kv_cache, dim(c_.attention.head_dim));
    const auto kv_heads = dim(c_.attention.kv_heads);
    execution::ForwardKV kv;
    kv.block_tables  = tables_->matrix();
    for (std::uint32_t l = 0; l < plan_.kv_layers; ++l) {
        const prefix::PlaneRange planes = prefix::kv_layer_planes(kv_geometry_, l);
        std::size_t plane               = planes.begin;
        ops::QsaKVLayer layer;
        layer.kv.storage      = options_.kv_cache;
        layer.kv.head_dim     = dim(c_.attention.head_dim);
        layer.kv.num_kv_heads = kv_heads;
        layer.kv.k_pages      = pool_->plane(plane++);
        if (layout.key.has_scale()) { layer.kv.k_scale_pages = pool_->plane(plane++); }
        layer.kv.v_pages = pool_->plane(plane++);
        if (layout.value.has_scale()) { layer.kv.v_scale_pages = pool_->plane(plane++); }
        layer.pooled_pages = pool_->plane(plane++);
        if (window_geometry_.present()) { layer.kv.window = prefix::kv_window_view(window_geometry_, window_backing_.p, l); }
        if (plane != planes.end) { throw std::logic_error("Qwen4Exp: KV planes do not match the page geometry"); }
        if (l < c_.attention_layers) {
            kv.layers.push_back(layer);
        } else {
            kv.mtp = layer;
        }
    }
    if (mtp_) {
        state.mtp_tails = Tensor(static_cast<std::byte*>(tails_backing_.p) +
                                     static_cast<std::size_t>(c_.attention_layers) * di_ * (r_ - 1) * lanes * 2,
                                 DType::BF16, {di_, r_ - 1, lanes});
        allocate_mtp(lanes, allocated);
        state.mtp_ones = Tensor(mtp_ones_.p, DType::FP32, {dim(c_.hc.streams), mtp_columns_});
    }
    execution::ForwardExperts experts;
    experts.frames.assign(c_.num_hidden_layers, nullptr);
    if (max_width_ > 1) { allocate_verification(lanes, allocated); }
    if (plan_.dflash2) { allocate_dflash2(lanes, allocated); }
    work_capacity_ = plan_.work;
    wide_work_     = plan_.wide_work;
    work_          = std::make_unique<WorkspaceArena>(work_capacity_);
    allocated += work_capacity_;

    // The promotions' copy sources: the banks where the device reads them (cudaMemcpyBatchAsync
    // faults on a registered block's host address where the device maps it elsewhere: WDDM).
    std::vector<const std::uint8_t*> banks;
    std::vector<std::uint64_t> host_strides; // each bank's own record stride (a GGUF import's vary)
    std::uint64_t record_stride = 0;
    for (const auto& layer : parameters_.layers) {
        banks.push_back(reinterpret_cast<const std::uint8_t*>(layer.moe.bank->planes.device_records));
        host_strides.push_back(layer.moe.bank->planes.record_stride);
        record_stride = std::max(record_stride, layer.moe.bank->planes.record_stride);
    }
    host_strides_ = host_strides; // the expert stream's retarget reads the member
    if (record_stride != plan_.record_stride ||
        (!c_.exact_experts && std::any_of(host_strides.begin(), host_strides.end(),
                                         [&](std::uint64_t s) { return s != record_stride; }))) {
        throw std::logic_error("Qwen4Exp: the expert banks' record stride differs from their format's");
    }
    // The shared expert placement runtime (models/offload/expert_placement_runtime.h): it owns
    // the residency, the RAM tier and its fetch channel, the VRAM monitor, the warm start and
    // the placement orchestration; the hooks carry what stays with the family (diagnostics, the
    // calibrated order, the tier's consumers, the round-cost reset, the elastic KV interplay).
    offload::ExpertPlacementRuntime::Options placement;
    placement.label                   = "Qwen3.8-Flash-Next";
    placement.geometry                = bank_geometry(c_);
    placement.expert_state            = options_.expert_state;
    placement.expert_state_identity   = options_.expert_state_identity;
    placement.expert_cache            = options_.expert_cache;
    placement.resident_experts        = options_.resident_experts;
    placement.expert_frame_target     = options_.expert_frame_target;
    placement.expert_frame_ceiling    = options_.expert_frame_ceiling;
    placement.expert_pinned_keys      = options_.expert_pinned_keys;
    placement.expert_tier_order       = options_.expert_tier_order;
    placement.expert_swap             = options_.expert_swap;
    placement.vram_monitor            = options_.vram_monitor;
    placement.vram_headroom           = options_.vram_headroom;
    placement.expert_ram_bytes        = options_.expert_ram_bytes;
    placement.expert_unpinned_bytes   = options_.expert_unpinned_bytes;
    placement.later_pinned_bytes      = options_.later_pinned_bytes;
    placement.expert_stripe           = options_.expert_stripe;
    placement.stripe_artifact_id      = parameters_.model.info().artifact_id;
    placement.prefill_stream          = options_.prefill_stream;
    placement.staging_slots           = static_cast<std::uint32_t>(plan_.staging_slots);
    placement.cpu_workers             = options_.cpu_expert_workers;
    placement.cpu_jobs                = options_.cpu_expert_jobs;
    placement.cpu_assist_jobs         = options_.cpu_assist_jobs;
    placement.layer_walk_tokens       = kWalkMaxTokens;
    placement.chunk_columns           = static_cast<std::uint32_t>(chunk_);
    placement.stream_read_slots       = kStreamReadSlots;
    placement.min_tier_slots          = kMinTierSlots;
    // The prefill stream's bounce per part: a wide chunk streams about a layer's unpinned share
    // of its experts (the stream copies two halves of a layer's unpinned records per call).
    if (options_.prefill_stream && options_.expert_cache) {
        if (const auto& store = parameters_.model.expert_store(); store.has_value()) {
            const std::uint64_t keys = static_cast<std::uint64_t>(store->layers()) * store->experts();
            const double share = std::min(1.0, static_cast<double>(options_.expert_unpinned_bytes / record_stride) /
                                                   static_cast<double>(keys));
            placement.stream_bounce_per_part = std::clamp<std::uint32_t>(
                (static_cast<std::uint32_t>(std::ceil(share * store->experts())) + 15U) / 16U * 16U,
                kStreamBounceMin, kStreamBounceMax);
            placement.stream_bounce_parts = execution::ExpertStream::kBounceParts;
        }
    }
    offload::ExpertPlacementRuntime::Hooks placement_hooks;
    placement_hooks.diagnostic = [this](const std::string& text, DiagnosticLevel level) {
        diagnostic(text, level);
    };
    placement_hooks.fallback_seed_order = [this]() -> std::optional<offload::expert_cache::SavedState> {
        if (!has_calibrated_order()) { return std::nullopt; }
        return qwen4_exp::calibrated_expert_order();
    };
    placement_hooks.on_tier_changed = [this](offload::HostExpertTier* tier, bool rebuilt) {
        on_tier_changed(tier, rebuilt);
    };
    placement_hooks.on_switch_settled = [this](const char* reason) { reset_round_costs(reason); };
    placement_hooks.all_lanes_idle = [this] {
        return std::all_of(lanes_.begin(), lanes_.end(), [](const Lane& l) { return l.phase == Phase::Free; });
    };
    placement_hooks.kv_idle_due = [this] { return kv_idle_due(); };
    placement_hooks.kv_bytes = [this] { return kv_vmm_ != nullptr ? kv_vmm_->mapped_bytes() : plan_.kv; };
    placement_hooks.capture_sizing = [this](const VramDemand& demand, const VramSnapshot& snapshot) {
        capture_sizing_inputs(demand, snapshot);
    };
    placement_runtime_ = std::make_unique<offload::ExpertPlacementRuntime>(device_, *vram_, std::move(placement),
                                                                           std::move(placement_hooks));
    // Staging slots for each layer call's misses (copied with a compact read window, design
    // 8.6), allocated before the frames take the remaining memory.
    allocate(staging_, plan_.staging, allocated);
    experts.staging_base  = static_cast<std::uint8_t*>(staging_.p);
    experts.staging_slots = plan_.staging_slots;
    // The expert cache's tables and route log; its frames come last (below).
    residency_ = &placement_runtime_->create_residency(std::move(banks), record_stride, columns_, host_strides);
    allocated += plan_.bytes.residency;
    if (allocated != plan_.bytes.fixed_bytes()) {
        throw std::logic_error("Qwen4Exp: the Program's device allocations differ from its device plan");
    }
    if (const std::uint64_t spilled = guard.end(allocated); spilled != 0) {
        throw std::runtime_error(spill_message(spilled));
    }
    {
        // A device fault (the kernel log's Xid 31 address) maps to its buffer, and inside the
        // frame pools to a frame index, from this line.
        char line[256];
        std::snprintf(line, sizeof(line), "; workspace %p +%zu, staging %p +%zu, KV %p +%zu, GDN state %p +%zu",
                      work_->base(), work_capacity_, staging_.p, staging_.bytes,
                      kv_vmm_ ? kv_vmm_->base() : kv_backing_.p, static_cast<std::size_t>(plan_.kv),
                      state_backing_.p, state_backing_.bytes);
        diagnostic("device address map: " + residency_->address_map() + line, DiagnosticLevel::Info);
    }
    CUDA_CHECK(cudaStreamCreateWithFlags(&overlap_.stream, cudaStreamNonBlocking));
    for (auto& event : overlap_.events) { CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming)); }
    experts.overlap_stream = overlap_.stream;
    experts.overlap_events = overlap_.events;
    CUDA_CHECK(cudaStreamCreateWithFlags(&overlap_.shared_stream, cudaStreamNonBlocking));
    for (auto& event : overlap_.shared_events) { CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming)); }
    experts.shared_stream = overlap_.shared_stream;
    experts.shared_events = overlap_.shared_events;
    // The prefill chunk the expert cache can lend a wide workspace for, from the frames the
    // cache will start with (sized below with the same demand); before the SSD tier, whose
    // tickets are sized by the chunk. A cache of several frame pools lends from one, which the
    // check after the cache is sized accounts for.
    if (options_.expert_cache) {
        const std::uint64_t frames = size_expert_frames(vram_->query(), expert_frame_demand(record_stride)).frames;
        fit_wide_work_to_cache(frames - frames / 4, record_stride);
    }
    // The SSD expert tier (design §9.5): the banks were not pinned; RAM slots hold the
    // records the tier keeps and the rest are read from the artifact when a round needs them.
    // The tier's sizing and construction are the shared placement runtime's (create_tier).
    if (const auto& store = parameters_.model.expert_store(); store.has_value()) {
        for (std::uint32_t l = 0; l < store->layers(); ++l) {
            if (store->record_bytes(l) != host_strides[l]) {
                throw std::logic_error("Qwen4Exp: the expert store's records differ from the banks' strides");
            }
        }
        placement_runtime_->create_tier(*store);
        tier_ = placement_runtime_->tier();
        fetch_channel_ = placement_runtime_->fetch_channel();
    }
    // After the tier (whose pinned slots are the probe's source when the banks stay in the
    // artifact), before the CPU service and the split take the measured rate.
    measure_link();
    device_error_ = PinnedHostBuffer(64);
    std::memset(device_error_.data(), 0, 64);
    experts.error = static_cast<std::uint32_t*>(device_error_.data());
    // The prefill stream copies records into ring slots of the frame pitch; a GGUF import's
    // banks keep their own (smaller) strides.
    if (options_.prefill_stream && options_.expert_cache) {
        StreamBindings bindings = stream_bindings();
        expert_stream_ = std::make_unique<execution::ExpertStream>(c_.num_hidden_layers, c_.moe.experts, record_stride,
                                                                   std::move(bindings.record_of), bindings.one_allocation,
                                                                   std::move(bindings.hold), host_strides_,
                                                                   std::move(bindings.ssd));
        expert_stream_->set_staging(std::move(bindings.staging));
        experts.stream = expert_stream_.get();
    }
    // CPU-served misses for decode and verify calls, and for every call of at most
    // kMaxCpuColumns columns (short prompts, chunk remainders, forced tokens), whose experts are
    // all thin and so give the same bits on either device (design 17.2); wider prefill chunks
    // stay on the GPU.
    // A GGUF import's records are computed by the ggml CPU route (ops/offloaded_sparse_moe/cpu/ggml_expert.h).
    const std::uint32_t cpu_workers = this->cpu_workers();
    const std::uint32_t cpu_jobs    = options_.cpu_expert_jobs;
    const std::uint32_t assist_jobs = options_.cpu_assist_jobs;
    if (cpu_workers > 0 && cpu_jobs > 0) {
        // The team's rate sets the decode miss split (design §10.5): the service starts on the
        // reference rates, and its channels take the measured split after the warm start (below).
        // Decode and verification calls are CPU-served up to decode_columns; prefill calls up to
        // kAssistMaxColumns take the assist cap (wider chunks stream or stage on the GPU).
        const int decode_columns = std::max(ops::offloaded_moe::kMaxCpuColumns, lanes * max_width_);
        const bool assist        = assist_jobs > 0 && decode_columns < kAssistMaxColumns;
        std::vector<ops::offloaded_moe::CpuMissService::Layer> service_layers;
        for (const auto& layer : parameters_.layers) {
            service_layers.push_back({.records       = reinterpret_cast<const std::uint8_t*>(layer.moe.bank->planes.records),
                                      .record_stride = layer.moe.bank->planes.record_stride,
                                      .scales        = layer.moe.bank->scales.data(),
                                      .ggml          = ggml_format(*layer.moe.bank)});
        }
        cpu_service_ = std::make_unique<ops::offloaded_moe::CpuMissService>(
            std::move(service_layers),
            ops::offloaded_moe::CpuMissService::Options{
                .workers     = static_cast<int>(cpu_workers),
                .max_jobs    = static_cast<int>(std::min<std::uint32_t>(cpu_jobs, ops::offloaded_moe::kMaxCpuJobs)),
                .max_columns = assist ? kAssistMaxColumns : decode_columns,
                .pcie_divisor = pcie_divisor(),
                .pcie_share   = pcie_share(),
                .wide_share      = wide_link_share().first,
                .wide_share_from = wide_link_share().second,
                .one_column_divisor = one_column_divisor(),
                .wide_from   = assist ? decode_columns + 1 : 0,
                .wide_jobs   = assist ? static_cast<int>(std::min<std::uint32_t>(assist_jobs, ops::offloaded_moe::kMaxCpuJobs)) : 0,
                .cpus        = {},
                .activation  = parameters_.layers.front().moe.bank->activation,
                .records     = tier_}); // tier mode: SSD-only jobs read through the tier
        for (std::uint32_t l = 0; l < c_.num_hidden_layers; ++l) {
            experts.cpu.push_back(cpu_service_->channel(static_cast<int>(l)));
            // Plain decode's divisor follows the context (plain_divisor): the calls read it from
            // the io word each round uploads.
            if (one_column_divisor() != 0) {
                experts.cpu.back().one_column_divisor_device =
                    reinterpret_cast<const std::int32_t*>(static_cast<std::byte*>(io_device_.p) + io_layout_.divisor);
            }
        }
        if (assist && expert_stream_ && options_.prefill_cpu_split) {
            split_policy_ = std::make_unique<SplitPolicy>(*this);
            experts.split = split_policy_.get();
        }
    }
    create_prefix_cache();

    // The VRAM expert cache takes what the fixed allocations, the reserve for graph
    // executables and the display headroom leave (design §9.7). Its frames are the one
    // elastic allocation: a chunk the driver places in system memory (another program took
    // memory since the reading) is given back and the cache stops growing there.
    if (options_.expert_cache) {
        const VramDemand demand = expert_frame_demand(residency_->frame_bytes());
        // The sizing, the resize and the monitor's start are the shared placement runtime's
        // (start_expert_frames); the workspace's lending follows the frames it started with.
        const auto started = placement_runtime_->start_expert_frames(demand);
        sizing_ = started.sizing;
        // The pool the workspace is lent from (one of several for a GGUF import's record sizes).
        if (started.resized.frames > 0) {
            fit_wide_work_to_cache(residency_->lend_capacity_bytes() / residency_->lend_pitch(), residency_->lend_pitch());
        }
        if (options_.resident_experts) {
            // No frame lends: the wide workspace comes from what startup left free.
            fit_wide_work_to_free(vram_->query().device_free);
        }
    }
    for (std::uint32_t l = 0; l < c_.num_hidden_layers; ++l) {
        experts.frame_bases.push_back(residency_->frame_base(l));
        experts.frame_pitches.push_back(residency_->frame_pitch(l));
    }
    experts.route_log    = residency_->route_log();
    experts.route_stride = residency_->route_stride();
    for (std::uint32_t l = 0; l < c_.num_hidden_layers; ++l) { experts.frames[l] = residency_->table(l); }
    experts.landing       = residency_->landing_table();
    experts.landed        = residency_->landed_log();
    experts.landing_slots = static_cast<std::int32_t>(offload::ExpertResidency::kLandingSlots);
    experts.tail_skip_rank = static_cast<std::int32_t>(options_.expert_tail_skip);
    if (tier_) {
        for (std::uint32_t l = 0; l < c_.num_hidden_layers; ++l) {
            experts.host_tables.push_back(residency_->record_table(l));
            experts.fetch.push_back(fetch_channel_->channel(static_cast<int>(l)));
        }
    }
    placement_runtime_->warm_start_experts();
    // The CPU team's rates, read until they settle (cpu_rates.cpp) on the pinned banks or the SSD
    // tier's RAM copies (it has none before its fill), so the decode miss split follows this record
    // size and CPU, not the reference. Both layouts measure here, after the warm start; the service
    // starts on the reference rates and its channels take the measured divisor.
    if (cpu_service_) {
        measure_cpu_rate_settled(record_stride);
        for (auto& channel : experts.cpu) { channel.pcie_divisor = pcie_divisor(); }
    }
    if (!options_.route_trace.empty()) {
        trace_ = std::make_unique<offload::RouteTrace>(
            options_.route_trace,
            offload::RouteTraceSetup{.layers             = c_.num_hidden_layers,
                            .experts            = c_.moe.experts,
                            .top_k              = c_.moe.top_k,
                            .frames             = residency_->frames(),
                            .max_columns        = static_cast<std::uint32_t>(columns_),
                            .lanes              = options_.max_concurrency,
                            .max_width          = static_cast<std::uint32_t>(max_width_),
                            .mtp_draft_tokens   = static_cast<std::uint32_t>(mtp_k_),
                            .ngram_draft_tokens = options_.ngram_draft_tokens,
                            .prefill_chunk      = static_cast<std::uint32_t>(chunk_)});
    }
    forward_ = std::make_unique<execution::Forward>(parameters_, device_, *work_, std::move(state), std::move(kv),
                                                    std::move(experts), dim(options_.max_context), options_.fast_paths);
    if (dflash2_) { create_dflash2_drafter(); }
    device_.synchronize();
    report_lane_cost(lanes, residency_->frame_bytes());
}

} // namespace infernix::models::qwen4_exp::detail
