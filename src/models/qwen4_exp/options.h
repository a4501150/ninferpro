#pragma once

// The Qwen4Exp (Qwen3.8-Flash-Next) family's option defaults and refusals: the family's branches
// of the shared runtime::validate_engine_options / runtime::normalize_engine_options
// (src/runtime/engine/engine_options.cpp), which calls them at their original points.

#include "infernix/types.h"

namespace infernix::models::qwen4_exp {

// The model's default prefill chunk. Its exact route split at 1,024, 4,096 and 12,288 tokens gave
// identical bits (36 perplexity windows of 14,336 tokens; its calls split on the 16-token GDN grid
// and its routes are column-invariant), and wide chunks prefill faster with no decode cost:
// Dense8, INT8 KV, RTX 5090,
// pp32768 9,639 -> 11,211 tok/s from 4,096 to 16,384 (pp65536 9,887 -> 11,268; 26 fewer expert
// frames of ~8,560).
constexpr std::uint32_t kPrefillChunk = 16384U;

// While another request decodes, the model prefills in narrower calls: the engine runs one decode
// round per prefill call (a mixed round runs both in one Forward). Production's flags (NVFP4,
// FP8 KV, DFlash2, RTX PRO 6000), a 15.6k-token prompt arriving beside a stream of 137 deltas/s,
// with mixed rounds: 2,048-token calls (less the round's columns) left it 10.8 deltas/s with
// 165 ms gaps and a time to first token of 1.5 s (1.3 s alone); 1,024 gave 11.5 and 104 ms for
// 1.6 s; 512 gave 15.7 and 71 ms for 2.1 s.
constexpr std::uint32_t kPrefillChunkBesideDecode = 1024U;

// An automatic draft length chooses up to the model's most drafts unless given a ceiling: it
// verifies 8 columns column-invariantly. Its DFlash2 drafter takes the same per-round policy.
constexpr std::uint32_t kDraftTokensAuto = 7U;

// --draft-tokens auto drafts with --spec mtp or dflash2 here.
void validate_draft_backend(const EngineOptions& options);

// The draft-confidence gate applies to this model's one-row MTP rounds with --spec mtp or dflash2.
void validate_draft_min_p(const EngineOptions& options);

// The family's prefix-cache tier rules and defaults: its snapshots are born in the Host tier, or
// without one written through to the Disk tier (design §19.3.1); it has no Device snapshot slots,
// no tool-call holds, and the Disk tier's default capacity (the shared normalize filled nothing
// it owns here, so this runs at the cache block's family point).
void resolve_cache(ContextCacheOptions& cache, const EngineOptions& options, std::uint32_t concurrency,
                    bool host_tier);

// The chunk spacing its tap ladder keeps: the model splits its prefill calls at taps instead
// (free-cache-resume), so the ladder keeps the 4,096-token chunk's spacing whatever wider chunk it
// prefills with.
[[nodiscard]] std::uint32_t tap_ladder_chunk(std::uint32_t prefill_chunk) noexcept;

} // namespace infernix::models::qwen4_exp
