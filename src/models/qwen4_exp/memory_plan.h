#pragma once

// Qwen3.8-Flash-Next's startup memory plans: the family-specific CUDA-graph executable bound
// (graph_bound, its drafter graphs' counts). The host RAM ledger and the VRAM sizing are the
// shared offload tier's (models/offload/memory_plan.h); aliased here under the family's names.

#include "models/offload/memory_plan.h"

#include <cstdint>

namespace infernix::models::qwen4_exp {

using offload::kLedgerGiB;
using offload::HostMemoryDemand;
using offload::kPinRoomMarginBytes;
using offload::pin_room;
using offload::ExpertPlacement;
using offload::HostMemoryLedger;
using offload::plan_host_memory;
using offload::kLedgerMiB;
using offload::kVramBudgetMarginBytes;
using offload::kGraphExecutableBytes;
using offload::VramDemand;
using offload::VramSizing;
using offload::display_headroom;
using offload::internal_reserve;
using offload::size_expert_frames;
using offload::frame_target;

// CUDA graph executables a Program instantiates at most: one decode graph per batch size, and with
// speculation a verification graph per batch size and width, and the MTP drafter's catch-up (per
// batch size and width) and draft graphs (per batch size and step), twice with sampled drafts
// (sampled-mtp-drafts: rounds with a sampled row replay a sampling draft graph), or the DFlash2
// drafter's proposal (per batch size) and context catch-up (per batch size and width) graphs.
[[nodiscard]] std::uint32_t graph_bound(std::uint32_t lanes, std::uint32_t max_width, std::uint32_t mtp_draft_tokens,
                                        bool sampled_drafts = false, std::uint32_t dflash2_draft_tokens = 0);

} // namespace infernix::models::qwen4_exp
