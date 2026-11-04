# Offloaded MoE families: shared design

This document is the design authority for the offload tier that every offloaded MoE family
composes: the VRAM expert cache, the RAM and SSD tiers, expert placement, the CPU expert
engine, and the miss and fetch channels between the device and the host. It states the design
in the present tense and keeps only what is true of the code today. A family document — today
[qwen3_8-flash-next-design.md](qwen3_8-flash-next-design.md) — keeps that family's constants,
kernels, recipes and measurements, and cites this document as "MoE §N" for the shared
machinery.

## Contents

1. [Scope and use](#1-scope-and-use)
2. [Principles](#2-principles)
3. [Expert banks](#3-expert-banks)
4. [Residency](#4-residency)
5. [CPU expert engine](#5-cpu-expert-engine)
6. [Decode on an offloaded MoE](#6-decode-on-an-offloaded-moe)
7. [Prefill on an offloaded MoE](#7-prefill-on-an-offloaded-moe)
8. [Performance method](#8-performance-method)
9. [Adding a family](#9-adding-a-family)

---

## 1. Scope and use

One tier, owned by no family. What it is:

| Piece | Code |
|---|---|
| Bank geometry (`ExpertBankGeometry`: banks, experts per bank, top-k) | `src/models/offload/geometry.h` |
| VRAM frame pools, device tables, route log, landing, frame leases (`ExpertResidency`; `expert_cache::LfruPolicy`, `FramePool`, `CacheController`) | `src/models/offload/expert_residency.*`, `expert_cache.*` |
| Placement spec, placement file, runtime switch request (`ExpertPlacementSpec`, `ExpertsSpec`) | `src/models/offload/expert_placement.*` |
| In-place expert records and the expert stripe (`ExpertStore`, `ExpertStripe`) | `src/models/offload/expert_store.*` |
| RAM level: decayed-LFU ranking, pinned slots, pins, landings (`HostTier`); the read agent, tickets, stream slots (`HostExpertTier`) | `src/models/offload/host_tier.*`, `host_expert_tier.*` |
| Saved cache state and the shipped calibrated order | `src/models/offload/expert_state.*`, `calibrated_order.*` |
| Route trace for offline replay | `src/models/offload/route_trace.*` |
| VRAM monitor | `src/models/offload/vram_monitor.*` |
| Routed-expert MoE op: routing, dispatch, narrow and wide GPU kernels, staging, CPU engine, miss and fetch channels | `include/infernix/ops/offloaded_sparse_moe.h`; `src/ops/offloaded_sparse_moe/` (`cuda/`, `cpu/`) |
| Canonical expert arithmetic shared by CPU and GPU | `src/ops/common/canonical_math.h` |
| Placement admin over the serving API | `src/serve/` (`GET`/`POST /admin/expert-placement`) |

What stays with a family: its mathematics, config, binding, recipes and stored formats, its
Program (which owns the tier's instances and calls them at its own round boundaries), its
memory-plan constants, its KV and prefix machinery, and its measurements.

**Composition.** A family's Program constructs the tier's pieces directly — no base class, no
virtual model interface, no plugin discovery — hands each its `ExpertBankGeometry` and record
strides, and drives them at its own round boundaries (`before_round` before the round's
kernels, `after_round` on the downloaded route log, `tier_step` inside a layer walk, the
placement switch at a maintenance boundary). The public contract the MoE kernels consume is
`MoeExpertSource` ([op development](op-development.md)); the tier is an execution resource of
the op, like a stream or a workspace.

**Neighbouring authorities.** [Engine architecture](engine-architecture.md) owns ownership and
boundaries; [op development](op-development.md) owns op qualification and the
placement-invariance exception; [model quality](model-quality.md#6-numerical-acceptance-policy-and-fast-paths)
§6 owns the acceptance policy and the fast-path gate; [hybrid prefix cache](hybrid-prefix-cache-spec.md)
owns the prefix cache; [paged KV cache](paged-kv-cache.md) owns KV. This document links them
and does not copy them. [Tensor formats](tensor-formats.md) and [storage layouts](storage-layouts.md)
own the record formats' representations.

## 2. Principles

1. **Mechanism in the engine, policy in the client.** The engine places experts exactly where
   a placement says, moves only the experts whose tier changes, reports what it did, and
   refuses or reports a placement it cannot reach. It does not choose tier shares, react to
   other programs' memory use, or keep headroom heuristics for the placement. Policies such as
   "leave N GB of VRAM and M GB of RAM free" live in the client. Which expert sits in which
   tier inside the client's counts is mechanism and stays in the engine.
2. **Exactness.** The exact route is bit-exact from release to release. Placement and tier
   never change bits: a routed expert's output is the same whether a VRAM frame, a staging
   slot, the CPU engine or a streamed copy computed it, so the tiers can move an expert at any
   boundary and a promotion landing mid-round changes no result. This placement invariance is
   a deliberate exception to the rule that a contract must not freeze bitwise equality
   ([op development §3](op-development.md#3-contract-headers)); without it the output would
   depend on cache history and CPU load. Where the artifact's formats allow, the arithmetic is
   also construction-exact and column-invariant, so a position's bits do not depend on how its
   work was split into calls and rounds. A faster but inexact route ships only as a fast path,
   qualified per configuration — the set of kernels an artifact runs — under the
   [acceptance policy](model-quality.md#6-numerical-acceptance-policy-and-fast-paths).
3. **Zero-cost sharing.** Sharing never costs runtime. The geometry is a plain struct of
   integers, the record format a compile-time kernel trait; there are no base classes, virtual
   model interfaces, plugin discovery or string-driven execution. Duplication between families
   is preferred over wrong sharing; only zero-cost primitives (core, ops, runtime contracts,
   this tier) are shared, and one family's fast path is never slowed to accommodate another
   ([AGENTS.md](../../AGENTS.md)).
4. **Measure at the claimed scope.** A speed claim holds for the placement it was measured
   under; the online cost fits are placement-scoped and reset on a placement switch. A
   measurement kept in this document decides a generic choice and names its hardware, date and
   family; family-tuned numbers stay in the family document.
5. **One authority per fact.** This document owns the shared machinery's design; the family
   documents own their constants, kernels and measurements, and cite it as "MoE §N". No fact
   is explained in both.

## 3. Expert banks

**Geometry as data.** `ExpertBankGeometry` carries three integers — `banks` (the routed
layers, one bank of expert records each), `experts_per_bank`, and `top_k` (the experts one
token selects per layer, the route log's width) — and every offload component is parameterized
by it. A family composes it from its own config. The record format's stride stays a constructor
argument (a bank's records carry it); families that need a frame-side entry or inner
dimensions extend their own composition, not the struct.

**Keys.** An expert's cache key is `layer * experts_per_bank + expert`, over every bank; the
policies, the tiers and the placement file all speak it.

**Record layouts.** One expert is one contiguous record, and every consumer reads it directly:
DMA into a VRAM frame, the narrow GPU kernels, the wide prefill route, and the CPU engine.
Only one copy of a bank fits in host RAM and runtime repacking is excluded, so the stored
record must serve every consumer — no tensor-core layout leaks into the record, no second
encoding for the host. Rows, blocks and scale tails follow the bank's stored format
([storage layouts](storage-layouts.md)); a bank's records sit a fixed stride apart, and
layers whose records share a stride share one frame pool (§4.2).

**Formats as kernel traits.** The record's stored format decides the kernel: `nvfp4_mul`
(W4A4 with stored input scales, or W4A16 without them), `fp8_e4m3fn_b128` (A16), the ggml
block formats, each with its exact decode in `src/ops/common/canonical_math.h` — one header
the CPU and the GPU compile, so both read the same words and the epilogue multiplier is
identical on both. A new stored format adds a trait and its kernels beside the existing ones
(`src/ops/offloaded_sparse_moe/`), not a new engine; a loader check keeps every layer's bank
one kind and its activation policy consistent with the presence of activation scales.

**The artifact is the SSD tier's source.** `ExpertStore` describes every record as file
segments — one, or two for a record straddling part files — each 4 KiB-aligned in offset and
size so a tier read is one unbuffered operation; each layer's scale tail is read into Model
memory at load. An **expert stripe** file (`tools/` per family) puts the tail of every record
on another drive, so the SSD tier reads each record from two drives at once without a second
copy of the experts; the stripe carries the artifact's id and is checked against it.

## 4. Residency

### 4.1 Tiers

Every routed expert lives in exactly one of three tiers: a frame in a VRAM pool, a slot (or an
unpinned record) in host RAM, or the artifact on SSD, read in place. Residency is exclusive; a
VRAM resident keeps its RAM copy only as a shadow. The device sees residency through three
fixed tables plus the route log: the per-layer frame table (`frame or -1`; -1 means the
record is read from host memory), in tier mode beside it the per-layer table of host record
pointers (null: SSD-only), and the landing table (§4.4).

### 4.2 Frame pools and device tables

**Pools by record size.** The layers whose expert records share a stride share one pool, with
its own LFRU. An artifact with one record size has one pool; an import whose layers have
several record sizes has one pool per size. Frames are sized as **average frames** (the mean
stride rounded up), and `resize(n)` gives every layer `n / layers` frames, so the cache holds
as many experts as its bytes allow (replay on Flash-Next traces: static size classes within
0.14 points of a byte-exact LFRU; a byte-exact per-size pool is rejected).

**Backing.** Each pool's frames live in a CUDA VMM arena whose base never moves (64 MiB
chunks): `resize()` maps chunks at the top to grow, and shrinks after evicting the
lowest-score experts and relocating the survivors below the new top, so kernels and captured
graphs keep their addresses. Without VMM the frames are one allocation of the first size.

**Device tables.** The frame tables are uploaded at each boundary; a promoted expert appears
in its table only after its copy has completed. Beside the tables of a round: the landing
table (§4.4) and the route log (each round's routed ids, downloaded with the round).

**Ordering.** Rounds run on the compute stream and the Program synchronizes at the end of
each. At the boundary the policy's evictions are applied to the device tables on the compute
stream before the next round; the promotions' copies run on a copy stream that first waits for
that table update, so no kernel can read a frame while it is overwritten. Outputs are
placement-invariant (§2), so the round a promotion lands in does not change any result.

### 4.3 Replacement and admission: LFRU

**Score.** `f / (now − last + 1)`, with `f` a global per-key use count kept for every routed
key, never reset on eviction. The clock advances once per routed layer call. Scores compare as
IEEE binary64 and ties evict the lower key, so the engine and the offline replay tool decide
identically step for step. `f` halves exactly every `halving_period` ticks (a pool's period is
scaled by its share of layers; counts below a floor become 0), so `f` weighs recent use; the
period is a family-tuned constant. Replay on Qwen3.8-Flash-Next traces (greedy decode of 14
mixed requests, 8,000 frames, 2026-10-10) measured halving 9-12 % better than never halving,
and never-halved LFRU loses to plain LRU by 4-5 % on a topic-changing trace.

**Victim search.** A budgeted round picks its victims through a low set (`LfruPolicy::build_low`):
the unpinned residents that can reach the 256th-lowest score or less within 64 ticks answer each
search exactly (a resident's score only falls until its next use, so one left out stays above the
bound until then; a use, credit, insert, halving or pin re-tests or empties the set), and a set that
cannot answer falls back to the full scan. The scan ordered every resident for each of ~75 victims
a round — upstream measured 992 µs a round on its 9,125-resident Dense8 struct fixture, against
155 µs with the set (2026-10-29). A step without an admission budget (the swap disabled) skips
ordering its misses.

**Evidence.** Replay of routing traces (`tools/expert_cache_replay/replay.py`; FreeToken
community traces of Qwen3.8-Flash-Next, 383 decode tokens each, batch 1, event-atomic — a
layer's experts must be resident together; recorded 2026-10-03), misses per token:

| Policy | 16.3% capacity | 32.6% | 48.8% |
|---|---:|---:|---:|
| LRU | 104.6 | 30.2 | 8.5 |
| **LFRU** | **88.4 (−15%)** | **25.0 (−17%)** | **6.6 (−22%)** |
| ARC | 81.1 | 27.9 | 7.2 |
| S3-FIFO | 94.2 | 26.6 | 7.2 |
| Belady MIN (offline optimum) | 46.0 | 11.8 | 3.2 |

Of ten policies replayed, LFRU is the only one better than LRU in every replay, method and
capacity (Zhang, arXiv 2608.07911, agrees on Qwen3-30B-A3B traces). Admission filtering alone
is worth ~0 at batch 1; the gain lies in choosing victims. Belady's gap is the headroom;
learned next-use predictors measured below LFRU in the same replay study and stay research
options. Hard per-layer quotas lost 3-34 % in replay; per-layer routing entropy is high but
static, while the order the cache exploits is temporal, which one global LFRU already tracks.
Host-shadow replays of the policies with parameter dueling were planned and never built;
replay stays offline.

**Admission budget.** One promotion per layer call per decode round, the interval scaled to
the measured link (a promotion moves as many PCIe bytes as serving the expert once
zero-copy). Until the session has promoted or landed as many experts as there are frames (the
fill phase), every round promotes at that rate. In tier mode only keys with a host copy are
admitted — SSD-only experts reach VRAM through landing (§4.4) — and the RAM tier's eviction
gate can refuse a victim, which stops that layer's admissions for the round.

**Pins and seeds.** An explicit placement's VRAM set is pinned: pinned keys are never
eviction victims, so exactly the placed set stays resident. The cache starts from the saved
state of the previous run (`--expert-state`, identity-checked against the artifact; counts
capped so a stale expert yields to new uses), else from a shipped calibrated order for the
architecture's shape, else spread across the layers. The seed fills a share of the frames,
leaving the rest free for the fill phase's landings. A static profile alone loses badly on a
differing workload, which is why the ranking is saved per session.

**Prompt credit.** At a prompt's end every routed key gains LFRU count proportional to the
prompt's uses; nothing is copied — the credit only ranks later admissions and evictions.

### 4.4 Landing and safe frame reuse

**`before_round`.** Publishes the promotion copies that completed (each batch of 64 loads
carries a completion event, so a large promotion publishes in steps). A load is published only
if the policy still wants it — same key, frame and load serial — closing the ABA case where a
key is evicted and re-admitted into the same frame while an older copy is still in flight. In
tier mode it finishes completed demotions (§4.6), closes the tier's round, admits its
landings, and uploads the host pointers that changed. For decode and verification rounds it
reserves landing frames per layer and uploads the landing table; the round's forked MoE calls
copy their first staged misses into them instead of staging slots, and the settle adopts them
as residents (a landed expert uses up its layer's promotion budget first).

**`after_round`.** The round's route log is downloaded with the round; the policies run over
it on the host. A verification round's rejected draft columns are flagged `live = 0`: they
neither count as uses nor start promotions. Promotions follow the tokens a round advances, so
a speculative round warms the cache per token as plain decode does.

**Safe frame reuse.** The frame pool tracks free, held (demoting), pending (retired),
unbacked and loaned frames. A retired frame returns when `round_done ≥ round_started_sample +
D`, `D` the rounds that can be in flight; the composing Program decides `D` (a synchronous
Program frees every retired frame at its boundary and runs with no slack frames). An admitted
expert that finds no free frame waits in a FIFO queue, stays CPU-served, and loads as soon as
a frame retires; if the policy evicts it first it leaves the queue.

**Failures.** Every MoE call has an error word. A failed record read answers an errno in the
status word; the Program checks the word after each round's synchronization and fails the
round's requests while keeping the engine serving (the prefix cache is cleared, since the
failed round may have published state it never computed).

**Rejected:** the packed u32 residency entry (state, generation, frame bits) with a LOADING
state and expected-landing-time gating — superseded by the I32 frame table with
publish-after-copy and the landing table; slack frames for an asynchronous agent (0 as built).

### 4.5 Frame leases

A contiguous run of one pool's frames can serve another user of device memory and come back
later. `lend()` picks the run with the lowest summed LFRU score (free frames score 0), never
loaned frames, avoiding frames that in-flight promotions target while a run without them
exists; it evicts the run's experts (no write-back: every cached expert is clean), relocates
the survivors into free frames below the new top by device copies, uploads one table, and
makes the compute and writer streams wait for each in-flight promotion batch that targets the
run. Graphs address experts through the tables, so a lease invalidates none. `give_back()`
restores the capacity and refills the run best-first while the policy has room, so a prefill's
leases do not leave decode a cache short of what it evicted. Each pool always keeps a quarter
of its frames; a lease is one contiguous run, so an earlier lease in the middle of a pool
splits what remains. Under a resident placement (§4.8) `lend()` refuses.

### 4.6 The RAM tier

The RAM tier (`HostExpertTier` with the `HostTier` controller) holds expert records in pinned
host slots, read in place from the artifact. An agent thread owns the unbuffered read queue
(§4.7) and spins from `begin_round` to `end_round`, so a request waits for no timer.

**Ranking: decayed LFU.** Score `Σ w · 2^−(t − t_use)/h`, kept per key as the time-invariant
`v = Σ w · 2^((t_use − base)/h)` — one multiply-add per use by the clock's growth, with a rebase
dividing every `v` by the same power of two once the clock is 256 half-lives past the base — so
ranking by `v` equals ranking by the current score; victims come from indexed min-heaps (each
eligible slot once; a resident's entry re-ranked lazily at the top, a shadow's at each use). The
log2 form (`v = log2 s + t/h`, two transcendentals a use) and lazy heaps that grew an entry a
use are gone. The clock advances by decode tokens, accepted verification
tokens and `min(columns, 16)` per prefill chunk; the half-life is a family constant. A demand
landing is admitted when it outranks the lowest resident; a prefetch landing at a margin
above the victim's score; a VRAM victim is demotion-worthy at a margin above the RAM victim.
Shadows leave most-used first, not in the residents' least-used order, and a use re-ranks the
shadow's heap entry (`HostTier::victim_value`): a shadow saves a demotion only when VRAM evicts
its expert, and VRAM evicts the least-used experts first, so dropping the least-used experts'
shadows left nearly every VRAM victim without a copy — upstream measured 24,318 → 2,057 demotions
a request on its Dense8 struct fixture (RTX 5090, 2026-10-29).

**Slots and lists.** Pinned slots are carved into a demand ring, prefetch slots, a demotion
list, bounce slots (with unpinned records) and the prefill stream's bounce. A slot is never a
victim and never rewritten while any pin is held (H2D, Queue, D2H, Use, Prefetch, Stream), and
every landing carries a serial so a stale one is discarded. A layer walk's span borrows
resident slots for stream reads of SSD-only records (§4.7) and returns them per span.

**Demotions.** A VRAM eviction whose key has a host copy keeps it (a shadow becomes
resident); one without is demoted when it outranks the RAM victim and the boundary's
allowance lasts: its frame is held and readable while a D2H stream copies it to a demotion
slot, and the next `before_round` publishes the slot and frees the frame. When the allowance
is spent, VRAM admissions wait instead of dropping the victim. Lending and resizing finish
pending demotions first and drop, not demote, their own victims.

### 4.7 The SSD tier and the expert stripe

The SSD tier is the artifact itself, read in place. An agent thread answers, while a tier
round is open: the CPU service's demands for SSD-only records (through `RecordProvider`:
demand, wait, done; one ticket per key per round), and the device's fetch requests
(`MoeFetchChannel`: request, response as records land in job order, a `consumed` word that
returns copied slots; one request outstanding per layer, the SSD-only experts no CPU job
takes moved to the end of the job list by a stable partition, so no bit changes). Reads are
unbuffered (`O_DIRECT` on POSIX, IOCP on Windows) through a read queue with one retry and
transient-error backoff, demand reads before prefetch. A fetch of at least 64 records queues
up to 32 read-aheads of the next layer's records without a host copy. Records whose bytes
follow each other in a file coalesce into one scatter read. A layer walk's span reads each
SSD-only expert once through lent stream slots. An agent that stops fails its pending and
later demands at once and the tier is treated as failed.

**Rejected:** an OS mapped-file expert tier (measured slower than the read queue with
identical flags, Strata); any SSD expert cache, spill or swap file (the artifact already holds
the exact bytes; writes only add wear); inclusive or independent tier residency (2-2.5× more
SSD reads in replay); long-half-life LFU or LFRU for the RAM ranking (wrong order for a
time-varying ranking); a fixed pool of stream-read slots (−8 % decode; per-span lending
streams them all).

### 4.8 Placement forms and the runtime switch

**Startup forms**, resolved by the family's instance construction:

| Form | Effect |
|---|---|
| default (no flags) | Probe: the device plan with the KV pool fully mapped must hold every expert (plus a small margin). Where it fits, every expert is resident, nothing is lent or evicted, and the RAM tier stays at its floor. A smaller card falls back to the maximal RAM tier. |
| shares or counts (`vram=…,ram=…[,ssd=…]`) | Shares or counts over every routed key, ranked by the saved state, else the shipped order, else spread across the layers. Adaptive within the counts; the vram count is the frame ceiling. `vram=all`/`100%` is the resident placement and fails the start if it does not fit. |
| placement file | CSV of `layer,expert,tier` rows; `vram` rows are pinned exactly, the pool is sized to hold them, `ram` rows fill the tier, the rest is SSD. |
| byte-level RAM flags | Byte-level control of the RAM tier (pinned and pageable expert bytes); cannot be combined with the counts or the file. |

The client owns the tier shares; the engine does not choose them, does not react to other
programs' memory use, and keeps no headroom heuristics for the placement. Within the client's
counts, which expert sits in which tier is mechanism and stays in the engine: the per-round
LFRU swaps hot experts in and cold ones out, and `--expert-swap off` freezes the pools'
content at what the seeds placed (misses still read and compute).

**Runtime switch** (`POST /admin/expert-placement`): counts or a placement file;
`dry_run` previews the placement's own numbers without applying. The switch is validated
before anything moves (frame limit, every pool's pinned keys fit, the RAM target plus list
slots fit available host memory) and applied at the next maintenance boundary — an idle
engine is woken for it. A shrink sets up the RAM tier first so its demotions land in slots,
evicts each pool's lowest-score residents (pinned keys never), demotes what the tier has room
for and drops the rest, then releases the freed VMM chunks. A grow resizes and then walks the
usage-rank order into the new frames as a resident refill, batched per maintenance tick,
reading through the tier (a reader-sized tier is temporarily grown to stage the walk). A
switch that names no RAM tier while holding every key releases the tier. `GET
/admin/expert-placement` reports the placement's counts, bytes, KV, free device and host
memory, the fill's progress, the swap counters and the last switch's outcome; `?export=PATH`
writes the current placement as a loadable file whose row order reproduces the ranking. The
boot warm start fills the frames from the saved state and the tier fill tops the RAM tier up
between requests.

**Rejected:** a runtime VRAM shrink/regrow law inside the engine (contradicted the
mechanism/policy split; removed 2026-10-30); two-phase pool shrink in full-RAM mode (the
shrink completes at the boundary in 1-3 ms).

### 4.9 Device memory sizing law

One sizing function, used at startup before the weights are read:

```text
frame_bytes = min(F − fixed − headroom, B − fixed − 64 MiB) − reserve
frames      = clamp(frame_bytes / average frame bytes, 0, max_frames)
```

with `F` the device-wide free VRAM, `B` the OS budget less this process's usage (no budget
source: `F`'s bound alone), `fixed` the dense weights plus every fixed Program allocation
(computed by the family's device plan, the single authority for constructor sizes), `headroom`
the display allowance, and `reserve` the allowance for CUDA graph executables instantiated
after sizing (a measured per-executable term; the formula is the family plan's). A negative
budget fails with the ledger and its largest contributors; a small frame count warns that
decode will be slow. A spill guard wraps every startup device
allocation: an allocation that does not lower device free by at least its size minus a slack
was silently placed in system memory and fails the start (Linux allocations are
VRAM-or-fail; there is no spill detection without a WDDM budget).

The pool is elastic through the VMM arena (VMM reads equal `cudaMalloc` reads); resizes run
on the engine thread at a boundary. At runtime the monitor thread wakes on the OS
budget-change event or once a second, keeps the latest snapshot, and does nothing else with
CUDA state: it hands the snapshot to the Program, which wakes an idle engine to return idle
KV pages to the expert frames and reports pressure below the headroom, naming the placement
lever. KV growth spends the sizing law's current allowance before it takes frames; the
placement names the pool's ceiling, which the target never exceeds. Never keep VRAM the
display needs: the sizing leaves headroom instead (a frame floor for the display is
rejected).

### 4.10 Host RAM ledger

A pure, unit-tested ledger plans every host allocation against one reserve (a family
constant), and each pin is re-checked against memory available at the moment it is made, so
other programs' growth since the snapshot is seen. Terms: the expert banks, other pinned
weights, program pinned and mapped buffers, the prefix-cache host tier, load staging,
pageable growth, lock overhead on every pinned byte, a full-mode margin. The experts take
what is left of physical memory and the commit limit, never more than full mode needs. The
ledger then splits the experts' RAM into pinned bytes inside the pin room (§4.11) and
pageable records beyond it. The tier refuses to start below its list slots plus a floor of
pinned slots, with the reason in the message. An explicit byte budget above what the ledger
leaves is an error, not a silent clamp.

### 4.11 Windows page-locking and unpinned records

Windows page-locks memory for the device only up to the adapter's shared system memory
(measured on an RTX 5090 host, 95.8 GiB RAM: shared 76.61 GiB, pins fail at 76.07 GiB with
5-6 GiB still free), so the RAM past `pin room = shared − usage − 1 GiB` holds a fixed set of
records in pageable memory kept resident (`ResidentHostBuffer`: hard minimum working set plus
`VirtualLock`/`mlock`), filled at startup after the slots and never replaced. They are never
admitted to a slot, are absent from the device's record table (the device treats them as
SSD-only), and are served in place to CPU jobs. Fetches and promotions copy them through
pinned bounce slots on copy threads: a pageable `cudaMemcpyBatchAsync` would hold the calling
thread at ~13 GB/s, while memcpy pageable-to-pinned runs at 20 GB/s on one thread, 36 on four,
41 on eight (i9-13900K), against ~50 GB/s on a Gen5 x16 link. Promotions of unpinned records
use the bounce slots in two halves — the CPU fills one while the link drains the other — and
the prefill stream has its own bounce beside them.

Linux has no such limit: without a WDDM shared-memory figure the pin room is unlimited, so the
ledger pins every RAM-tier expert and no unpinned records exist; only `--expert-pin-mib`
(a family option) caps the pinned share there.

## 5. CPU expert engine

### 5.1 Role and limits

The engine computes a call's non-resident routed experts on the CPU, in place from their
pinned host records, while the PCIe stage copies the rest into staging slots. It pays because
an expert with few columns is cheaper to compute than to move: a desktop CPU's DRAM reads
several records in the time an x8 link carries one (Qwen3.8-Flash-Next NVFP4 records, host
i9-13900K, DDR5-5800: one column ~52-55 µs per expert at 8 workers, ~50 GB/s, against ~115 µs
staged over a Gen5 x8 link). Only the narrow route (at most 8 columns per job) is
CPU-served — it is the placement-invariant route, and its sums are exact int64, so a CPU
output equals the GPU's bit for bit and greedy ids never depend on the split.

Limits fixed in code (`miss_request.h`): at most 512 jobs per layer call, at most 8 columns
per job, at most 4,096 columns per CPU-served call. The caps a family sets below them
(`max_jobs`, `max_columns`, `wide_jobs` for prefill assist) are family constants.

### 5.2 Kernels

Four ISA variants, chosen by CPUID at first use: scalar, AVX2, AVX-VNNI, AVX-512 VNNI. Each
kernel is a template on the column count: a record's codes are decoded once and reused for
all columns. Software prefetch runs a configurable distance ahead along the unit stream (a
host probe sweeps it; the default is a starting point, the tuned value a family constant).
The CPU sources compile with `-ffp-contract=off -fno-fast-math`. The per-format decode paths
(W4A4 quads, W4A16 byte limbs, ggml block planes, FP8 tiles) live with the format trait in
`src/ops/offloaded_sparse_moe/cpu/`; the family document records them. Measured capacity of
the engine on Qwen3.8-Flash-Next NVFP4 records (i9-13900K AVX-VNNI, DDR5-5800, 2026-10-04):

| | Rate |
|---|---|
| Host DRAM | 82-85 GB/s |
| W4A4 expert, 1-4 columns, cold | ~30,000 experts/s (DRAM-bound) |
| 8 columns, cold | ~19,000-21,000 experts/s (compute-bound) |
| 8 columns, warm | ~150,000 expert-columns/s |
| PCIe 5.0 x8 link, for comparison | ~10,000 records/s |

### 5.3 Worker team and scheduling

The service thread is worker 0 and starts computing at once; the other workers spin on an
epoch word for a bounded iteration count, then park on `std::atomic::wait` (a futex). Worker
pinning is supported but a Program that starts the service unpinned gets unpinned workers.
Work is handed out by atomic counters, not static ownership: x quantization (8 block slices
per column; gate and up scales separately when distinct), a barrier, the gate/up
intermediate units with SwiGLU and the activation quantization of h, a barrier, then the down
row groups. A16 rounds add an h-encoding phase; ggml rounds run four phases of their own.
The per-column phases hand out only the round's real columns, published with the epoch (a
one-column job no longer walks its eight slots a phase; a contended counter increment each).
SwiGLU runs as AVX2 canonical vector code on every vector ISA (`swiglu_bf16_n`): exp_c, silu_c
and the BF16 rounding on eight lanes, the scalar functions' IEEE operations in the same
order, so every element equals `canon::swiglu_bf16` bit for bit (the test checks it
exhaustively over BF16 gates; 640 elements 5.5 → 1.0 µs on this host, scalar → vector).
Faster cores take more items. Sums are exact, so the item split changes no bit: the team test
checks 1-40 workers against the single-threaded reference.

### 5.4 The miss-service protocol

GPU↔host signalling uses mapped pinned memory and sequence words: the producer writes the
payload, executes `__threadfence_system()`, then stores a never-zero sequence; the consumer
polls the word and reads the payload after it matches. Host memory the host rewrites between
rounds is read with `.cv` loads so a stale L2 line is never used.

The device publishes a call's CPU plan as one `MissRequest`: jobs, experts, record pointers
(the tier's host table entries) and per-job columns, the x columns compacted to those the
chosen jobs read. The host team computes, writes the mapped outputs, answers an errno in the
status word for a record it could not read, and publishes the answered sequence. Tiered jobs
(no host record) are demanded from the tier before the other jobs run and computed after
them. The device's wait lasts while the service's heartbeat word keeps changing — the service
thread beats on every poll, worker 0 after each completed item, so a multi-second tier read
stays alive while a hung worker stops the beat; after 10 s of silence the call writes
`kErrorHostSilent` (a wait without an error word or heartbeat traps after 2 s). After 20 ms
without a request the service idles in short waits; `wake()` ends one when a round that may
hand it requests is enqueued. A stopped service fails every later call. The fetch channel
(§4.7) carries the same discipline for SSD-only experts, with one elected poller CTA per call.

### 5.5 Startup rate probes

Two startup measurements make every bandwidth-dependent choice follow the machine. The link
probe times best-of-3 copies of expert records into the staging slots **while the SMs are
busy**: quiet copies measured a fraction of the Gen5 rate in about one startup in seven on a
Gen5 x8 link (3.0 against 12.0 GB/s, and the Gen2/Gen4 rates, 0 of 60 with the SMs busy), so
the warm-up keeps the device busy while the link copies. The CPU probe times the configured
workers on one-column and eight-column jobs while the link copies records (the staging's
DRAM reads compete), best batch per width; an attempt during which other processes used a
quarter or more of the CPU repeats — a busy host halves the rate, which moves the decode
divisor and costs a fifth of decode speed for a session. The measured link rescales the
family's bandwidth-dependent constants (the promotion interval, the split's width limit,
restore rates); until records exist in RAM the family's reference rates stand. A fixed rate
is injectable in tests.

### 5.6 The decode miss divisor

A call's host-memory misses are shared between the CPU and the PCIe stage: `misses / d` stay
on the stage, the CPU serves the rest up to its cap. `d = floor(1.65 + C·t)`, with `C` the
measured one-column expert rate and `t` a record's link time; the step from 2 to 3 lands at
`C·t = 1.35`, midway between the measured fastest divisors (Qwen3.8-Flash-Next, i9-13900K
host, Gen5 x8, 2026-10-09: at 6 workers `C·t = 1.5-1.7`, d = 3 best; at 3 and 2 workers
`C·t ≈ 1.0` and 0.7-0.8, d = 2 best), so the ±7 % startup noise stays clear of the switch. A
positive fixed divisor overrides the derivation, and one-column calls can take their own
divisor — a device word the captured graphs read, which a Program rewrites between rounds.

NVFP4 banks take a fixed **share form** instead of the divisor rule: W4A16 banks keep
67/256 of a decode call's misses on the stage, and every NVFP4 bank keeps 51/256 in calls of
at least eight columns — a CPU expert's cost past its first column is its compute, which the
one-column rate the rules take hides, so wide verification rounds give the CPU more. The
shares are fixed family constants, not measured rates; they are upstream's, derived on its
host (i9-13900K, RTX 5090 over Gen5 x8, DFlash2 auto), and need re-deriving on this machine
(9950X3D2, Gen5 x16) at T3 in a RAM/SSD placement.

**Rejected:** a MEASURED fractional share `1/(1+C·t)` (floor flips between startups);
`round(1 + C·t)` (the switch lands inside the startup noise); a divisor following the expert
hit rate over eight-token windows (below the best fixed divisor at every length); a
cost-based CPU/link split chosen per call while a cost model says it pays.

### 5.7 The prefill CPU split

A streamed narrow prefill call gates its copies on each layer's routing: after the dispatch
the host reads the per-expert column counts, marks the narrow non-resident experts — fewest
columns first, as many as keep the CPU's predicted time under the link's time for the rest —
and the stream copies only the others, served with divisor 0 and the chosen count as their
cap. The marks change no output bit. The policy plans at a **share** of the measured rates,
not the full rates: the stream copies most experts ahead, so the GPU finishes its side far
sooner and full rates overload the CPU (traced: a 256-token chunk's layer 0 ran the CPU
13.1 ms against an 11.8 ms GPU wait). The split takes calls up to a width cap scaled
inversely to the measured link (the bytes the gating saves cost less on a faster link, while
the idle link between a layer's routing and its copies does not shrink); wider calls take the
assist cap instead (calls from `wide_from` columns take `wide_jobs` as their cap, still
narrow-route jobs), and wider chunks stream or stage on the GPU (§7). The share, the width
cap and the assist caps are family constants.

### 5.8 Rejected

- Static worker ownership (E-cores held P-cores back; dynamic counters, cold 41 vs 52 µs per
  expert at 12+ workers, bits unchanged, i9-13900K).
- An AVX-VNNI-INT8 kernel variant (never built).
- CPU L2 warming of predicted misses (the GPU warms its own L2 during the wait instead).
- Full or fitted rates for the split (overload the CPU; a fitted line starved it, pp256
  +13 %, 2026-10-10); a split width cap above the measured optimum (pp2048 +2.4 %).

## 6. Decode on an offloaded MoE

### 6.1 Miss service and arbitration

The device only looks up residency; the host owns policy. Hits never involve the host: the
residency plan is made on the device inside the round (no per-layer handshake), and the host
hears about a layer only when it has a CPU-served miss. A non-resident job's record is first
copied into a staging slot by the stage kernels beside the hit compute — on the fork stream
for decode and verification rounds, double-buffered on the overlap stream for prefill chunks —
or served by the CPU team through the mapped channel; the first misses in job order land in
the reserved landing frames instead (§4.4). Tier records live in host slots the host
rewrites, so the stage reads them with `.cv` loads. Records the prefill stream already
copied resolve like resident ones: never staged, never CPU-served.

The CPU plan chooses the host-served jobs of a call from the dispatch alone:

- SSD-only misses first — only the CPU can serve them without a fetch;
- then host-memory misses by width, the fewest-column ones first, in job order;
- up to the call's cap: the decode/verification job cap, or the prefill assist cap (§5.7);
- the host share is `misses − kept`, so the CPU and the PCIe stage share a
  call's misses (`kept` is the divisor rule's `misses / d`, or the share
  form's fixed share — §5.6).

The fetch channel serves the SSD-only experts no CPU job takes (§4.7), moving them to the
end of the job list by a stable partition, so no bit changes. While the device waits for the
host's answer, the wait kernel's extra CTAs warm the next layer's dense weights into L2 in
reading order — the family decides the spans. A call that cannot serve an expert writes the
error word and fails the round.

### 6.2 Column limits

The fork route (staging beside hit compute, capturable into a round's graph) needs calls of
at most 8 columns and at most 512 jobs, within the staging slots; the fused
route/dispatch/plan kernel needs at most 32 columns and 1,024 routed entries; a CPU-served
call at most 4,096 columns, a CPU job at most 8; one call lands at most 32 experts. The
dispatch unions the routed entries, so each distinct expert is fetched and computed once for
all its columns; CPU cost scales with distinct misses, not with columns. A family's batching
and draft widths must stay inside these limits (a family's wide experts take the wide route,
which no fork serves).

### 6.3 Verify columns and speculation

Verification columns cost misses twice over: more columns touch more distinct experts (a
round of B rows at W columns costs more than B plain rounds' experts), and rows drafted as if
alone become recurring uncredited misses unless the policy accounts them. The rules the tier
gives a family's speculation:

- **Promotions follow the tokens a round advances** (draft-implied prefetch): a speculative
  round warms the cache per token as plain decode does, and rejected draft columns are
  flagged `live = 0`, so they neither count as uses nor start promotions (§4.4).
- **The drafter never waits on the host**: misses are staged and CPU-served beside hit
  compute, and a speculative round that outruns the miss service slows every row, so draft
  length is chosen from acceptance and the measured round cost, not from a fixed schedule.
- **The online cost fits are placement-scoped.** A fit measured under one placement prices
  another wrongly (a plain round timed with half the experts on the SSD priced drafts at a
  twentieth of the all-VRAM round after a switch back, and six-row decode fell 444 → 351
  tok/s until the reset was added; Qwen3.8-Flash-Next, RTX PRO 6000, 2026-10-22), so the
  fits reset on every placement switch or tier refill.
- A prompt's prefill credits its experts (§4.3), so the first decode rounds after a prompt
  find the answer's experts ranked, not the prompt's alone.

**Rejected:** lookahead prefetch of next-layer experts by the router (never built — the host
owns expert movement, and the CPU wait's L2 warming covers the same window); victim landing
(one-off misses push out residents that still earn hits; promoting less gained more);
per-layer cache quotas.

## 7. Prefill on an offloaded MoE

### 7.1 Wide calls stream experts

A call of at least a few hundred columns routes nearly every expert of every layer
(P(unused) = (1 − top_k/experts)^T: 0.7 % at 256 columns, ~10⁻⁹ at 1,024), so a layer's
copies need not wait for its router. The prefill expert stream copies each layer's
non-resident experts, in ascending id, on its own copy stream two layers ahead of compute,
into half a ring of borrowed frames (§4.5), with a `landed` event the compute waits for and
a `consumed` event the next layer's copies wait for. Runs of consecutive experts coalesce
into one copy; all of a layer's copies are enqueued as one batch, so the host never waits on
a full copy queue. An expert without a host copy is streamed too when the tier lends stream
slots: its record is read from the artifact ahead of its copy, so a layer-walk span reads
each SSD-only expert once. A call whose layer has every non-resident expert in the ring and
every record in RAM stages nothing and runs as one pass over its jobs. An expert a decode
round evicted meanwhile is read in place from its host record — the same bytes. Chunks below
the streaming threshold keep decode's staging and the gated CPU split (§5.7).

### 7.2 Chunk sizing against the cache

The prefill chunk trades link bytes against kernel width; a family measures its optimum and
the wider-call routes it selects. The wide workspace is lent from the frame pools in order —
the stream lease's own part, a frame lease beside it, or, when nothing is lendable, a
temporary allocation with a warning (a resident placement lends nothing). A chunk whose wide
workspace the cache could never lend is halved repeatedly until it fits, with a warning; the
output is the same. Leases taken for a prefill return before decode: `give_back` refills
best-first while the policy has room (§4.5).

### 7.3 No promotions during prefill

Wide calls and layer-walk spans promote nothing; their routes still count toward the cache's
scores and seed the prompt credit (§4.3). Calls of at most the served width (the CPU-served
openers and tiny suffixes) promote at the decode rate. Promotions during a walk would churn
through the frames the lease does not hold and, after the prompt, fill the returned frames
with the prompt's experts while the answer's staged misses land there anyway: promoting
during prefill measured slower than none in every regime tried (Qwen3.8-Flash-Next, RTX
5090, 2026-10-06; the family document holds the numbers). What promotions remain (decode)
are enqueued through copy batches in groups of 64 records with a completion event each, so
the host never waits on a full copy queue (the group size and the event cadence are measured
per family).

## 8. Performance method

### 8.1 The bytes-per-token bound

Two ceilings bound the miss traffic of a partially resident round:

```text
host DRAM:  CPU-served + DMA'd bytes  ≤  BW_dram · t_round
PCIe:       DMA'd bytes              ≤  BW_pcie · t_round
```

The CPU and PCIe paths draw host DRAM in parallel and share its bandwidth (six workers beat
eight; keeping a third of the misses on PCIe beat giving them all to the CPU — both measured,
Qwen3.8-Flash-Next, i9-13900K host, 2026-10-04). The dense path, not the experts, sets the
decode ceiling: the dense bytes decide it, the dense kernels run at the bandwidth roofline,
and speculation amortizes the dense read over accepted tokens. The dense recipe is a family
decision.

### 8.2 Tier bandwidths

RAM is worth far more than SSD for experts that leave VRAM. Qwen3.8-Flash-Next on the RTX
PRO 6000 (2026-10-30, one request, DFlash2, BF16 n-gram, FP8 KV): half the experts in RAM
decode at 263 tok/s, half on SSD at 68; the same boot's all-VRAM placement at 380. Whole
SSD-record reads of Qwen3.8-Flash-Next's 2,764,800 B records from the artifact
(`infernix_expert_read_probe --only reads`, unbuffered, machine idle):

| Drive (machine) | QD 1 | p50 at QD 1 | QD 8 |
|---|---:|---:|---:|
| Crucial T710, Gen5 (RTX PRO 6000, Linux, 2026-11-02) | 6.36 GB/s | 0.43 ms | 13.35 GB/s (4,830 records/s) |
| Samsung 990 PRO, Gen4 (RTX 5090 machine) | 4.68 GB/s | 0.55 ms | 6.71 GB/s (~2,100 records/s) |

The T710 saturates by depth 8 (13.11 GB/s at depth 16). A fetch of at least 64 records queues
read-aheads of the next layer. A switch round trip costs seconds and leaves
the engine whole.

### 8.3 How to measure and report

- One boot per binary; placements switched at runtime, so a matrix shares one warm cache.
- Concurrent requests from a thread pool; short polls instead of fixed sleeps; unique prompt
  prefixes so the prefix cache does not hide prefill.
- Every speed read from the server's `req#N done` log lines, not client timing
  (`bench_infernix.py`).
- Report the per-tier expert counts and the process VRAM (experts / KV / other) — and host
  RAM where the RAM tier is in play — with every speed table, from `GET
  /admin/expert-placement` and the process ledger.
- State the placement, hardware, date and family with every number. Placement-scoped claims
  only: a fully resident card's cost cannot be extrapolated from a partially resident
  machine, and cost fits reset on a switch (§6.3).

## 9. Adding a family

The checklist, in order:

1. **Copy the closest family** directory under `src/models/<family>/` and specialize it —
   its own mathematics, binding, Program, kernels and tuning. Duplicated code between
   families is acceptable; wrong sharing is not ([AGENTS.md](../../AGENTS.md)).
2. **Geometry**: compose `ExpertBankGeometry` from the config (banks, experts per bank,
   top-k); the record strides come from the artifact's banks.
3. **Recipe and format traits**: which stored formats the artifact's expert banks use; the
   kernel trait for each (an existing one, or a new trait with its exact decode beside the
   existing kernels); the exact-import rule for the publisher's quantized tensors.
4. **Binding**: loader checks — every layer's bank one kind, activation policy consistent
   with stored activation scales, strides as stored.
5. **Program composition of the tier**: construct the residency, the tiers, the CPU service,
   the fetch channel and the monitor; call them at the round boundaries; mark the resident
   placement; size frames and host RAM by the laws of §4.9-§4.10 with the family's constants.
6. **Family constants to measure** (each labeled with hardware and date): CPU worker count
   per bank format; the decode miss divisor's inputs (the startup probes measure them); the
   prefill split share and width cap; the prefill chunk; the assist caps; the LFRU halving
   period; the seed share; a warm start and a calibrated order (or none).
7. **Qualification**: op oracles for each format's exact arithmetic against the independent
   FP32/FP64 oracle ([op development](op-development.md)); round hashes at the family's T1;
   every faster-but-inexact route under the
   [acceptance policy](model-quality.md#6-numerical-acceptance-policy-and-fast-paths), qualified
   per configuration; real-artifact tests through the public Engine, including a placement
   switch and back with greedy ids unchanged across placements.
8. **Docs**: a family design document citing this document as "MoE §N" for the shared
   machinery and keeping the family's constants, kernels and measurements; a user guide
   (flags, placements, measured speeds) linked from `docs/README.md`.
9. **Tier extensions**: a family that needs the tier to grow names each addition, where it
   lands and why it costs the existing families nothing. Two planned families' additions are
   designed but not built:
   [GLM-5.3-Flash §8.2](glm5_3-flash-design.md#82-the-shared-offload-tier-and-what-it-gains-here)
   and [DeepSeek-V4.1-Flash §8.2](deepseek_v4_1-flash-design.md#82-the-shared-offload-tier-what-exists-what-this-family-adds).
   Both need expert records striped over several drives (more segments per record, one read
   queue per drive), designed in GLM §11. An addition moves into this document when it is built.
