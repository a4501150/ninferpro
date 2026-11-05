#pragma once

// The mapped-memory fetch channel through which the GPU asks the host for the records of a layer
// call's SSD-only experts that no CPU job serves (docs/maintainer/moe-offload-design.md
// §4.7, the SSD expert tier: prefill, calls past the CPU cap, CPU serving off). Shared by the
// device kernels (moe_layer.cu) and the host side (fetch_channel.h); plain data, no host-only types.
//
// Protocol. The device writes the request (layer, count, experts in job order), a system-scope
// fence, then `sequence` (never 0). The host answers that sequence: it writes `landed = 0` and
// `status = 0`, then `sequence`; then, as records land in job order, record[i] followed by
// `landed = i + 1` (release stores). The device copies record i once it reads landed > i for its
// sequence, and after a pass has copied records [0, b) it stores consumed = sequence << 32 | b, so
// the host may reuse the slots holding them. A nonzero `status` fails the request: the device
// copies nothing more and reports it in the call's error word. A host whose heartbeat word stops
// changing for kHeartbeatTimeoutNs is treated as gone.

#include <cstdint>

namespace infernix::ops::offloaded_moe {

inline constexpr int kMaxFetch = 512; // fetch-served experts per layer call (a layer's experts)

// Device-detected failure codes of a call's error word (layer << 16 | code). Host-reported failures
// carry an errno value, all below 0x8000.
inline constexpr std::uint32_t kErrorUnservedRecord  = 0xFF01; // an SSD-only expert with no path to it
inline constexpr std::uint32_t kErrorHostSilent      = 0xFF02; // the CPU service's heartbeat stopped during a wait
inline constexpr std::uint32_t kErrorFetchSilent     = 0xFF03; // the fetch responder's heartbeat stopped during a wait
inline constexpr std::uint32_t kErrorRowsWaitTimeout = 0xFF04; // the rows down pass's per-job wait passed its bound

// The bound of the rows kernels' per-job wait (moe_layer.cu, wait_job_h): a gate/up job counts its
// h elements in microseconds, so a count that is still short of its target after this long (GPU
// time) never completes — a producer bug (the gate/up pass no longer triggering its dependents
// before it works, a lost clear of the counts), not a slow round — and the wait reports
// kErrorRowsWaitTimeout and gives up instead of hanging the GPU. Like the heartbeat bound it
// leaves room for OS-level stalls that delay a live grid: a whole layer call runs in milliseconds,
// so no healthy round comes near it.
inline constexpr std::uint64_t kRowsWaitTimeoutNs = 2000000000ULL;

// A wait gives up when the host's heartbeat has not changed for this long (GPU time). A host thread
// that died is reported at once through its own failure (CpuMissService::failure, the tier agent's);
// this bound only ends a wait on a host that stopped without one. It is long enough for the OS to
// stall a live thread: under RAM pressure (Windows trimming and compressing working sets) 1 s failed
// FP8 requests on a healthy service (2026-10-16: 8 of 34 SSD-tier runs at ~2 GiB available).
inline constexpr std::uint64_t kHeartbeatTimeoutNs = 10000000000ULL;

struct alignas(64) FetchRequest {
    std::uint32_t sequence;
    std::int32_t layer;
    std::int32_t count;
    std::int32_t reserved;
    std::int32_t expert[kMaxFetch];
};

struct alignas(64) FetchResponse {
    std::uint32_t sequence;
    std::uint32_t landed;
    std::uint32_t status;
    std::uint32_t reserved;
    std::uint64_t record[kMaxFetch]; // host addresses the device can read (mapped pinned memory)
};

} // namespace infernix::ops::offloaded_moe
