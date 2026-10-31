#pragma once

// The Qwen3.5 family's option defaults and refusals: the family's branches of the shared
// runtime::validate_engine_options / runtime::normalize_engine_options
// (src/runtime/engine/engine_options.cpp), which calls them at their original points. The
// messages name the families because the flags are shared vocabulary across them.

#include "infernix/types.h"

namespace infernix::models::qwen3_5 {

// The family's default prefill chunk. Its prefill bits follow the chunk grid, and idle-wide-prefill
// already widens its idle steps (16,384 measured 3.5 % slower at pp16384 and pp32768 on the 27B).
constexpr std::uint32_t kPrefillChunk = 1024U;

// An automatic draft length chooses up to the model's most drafts unless given a ceiling: the
// Qwen3.5 planner captures 5 columns.
constexpr std::uint32_t kDraftTokensAuto = 5U;

// --vram-headroom-mib is what automatic KV capacity leaves; an explicit capacity has none
// (Qwen3.8-Flash-Next's is what its expert cache leaves, with any KV capacity).
void validate_headroom(const EngineOptions& options);

// --expert-tail-skip belongs to the offloaded-experts model; this family refuses it rather than
// ignoring a tuning value.
void validate_expert_tail_skip(const EngineOptions& options);

// --draft-tokens auto drafts with --spec mtp only here.
void validate_draft_backend(const EngineOptions& options);

// The draft-confidence gate exists only on Qwen3.8-Flash-Next's one-row MTP rounds; this family
// refuses it rather than ignoring a tuning value.
void validate_draft_min_p(const EngineOptions& options);

// The family's prefix-cache tier defaults and rules: the Device snapshot slots (one resident
// snapshot per request lane keeps every live conversation's latest endpoint restorable without
// PCIe traffic; one more slot stages taps and endpoints while their Host copies are written —
// without a Host tier these slots are the only snapshot storage, so one more is kept for shared
// prefixes), and the default tool-call hold.
void resolve_cache(ContextCacheOptions& cache, const EngineOptions& options, std::uint32_t concurrency,
                    bool host_tier);

} // namespace infernix::models::qwen3_5
