# Qwen3.8-Flash-Next on the RTX PRO 6000 and RTX 5090: current design

This document is the current design authority for the `qwen4_exp` model family
(`Qwen4ExpForCausalLM`, Qwen3.8-Flash-Next) as built and in production today. It states the design
by subsystem, in the present tense, and keeps only what is true of the code. Superseded design
proposals and the dated implementation log were removed on 2026-11-03 and remain in git at base
commit `0273fb5`.

The offload tier this family composes — the VRAM expert cache, residency, placement, the RAM and
SSD tiers, the CPU expert engine — is designed once for every offloaded MoE family and lives in
[MoE offload design](moe-offload-design.md), cited below as "MoE §N". This document keeps
Qwen3.8-Flash-Next's constants, kernels, recipes and measurements and points to the MoE document
for the shared machinery.

## Contents

1. [Goals, hardware and status](#1-goals-hardware-and-status)
2. [Model facts](#2-model-facts)
3. [Lessons from other implementations](#3-lessons-from-other-implementations)
4. [Performance model](#4-performance-model)
5. [Design overview](#5-design-overview)
6. [Artifact and conversion](#6-artifact-and-conversion)
7. [Ownership](#7-ownership)
8. [Decode pipeline](#8-decode-pipeline)
9. [Expert residency: tiers, cache, placement and memory budget](#9-expert-residency-tiers-cache-placement-and-memory-budget)
10. [CPU expert engine](#10-cpu-expert-engine)
11. [Speculative decoding](#11-speculative-decoding)
12. [PLE n-gram table](#12-ple-n-gram-table)
13. [Prefill](#13-prefill)
14. [KV, attention and long context](#14-kv-attention-and-long-context)
15. [Prefix cache, preemption and scheduling](#15-prefix-cache-preemption-and-scheduling)
16. [Vision](#16-vision)
17. [Numerics and qualification](#17-numerics-and-qualification)
18. [Calibration and configuration](#18-calibration-and-configuration)
19. [Sources](#19-sources)

---

## 1. Goals, hardware and status

### 1.1 What the product supports

`Qwen4ExpForCausalLM` runs through the public Engine, the CLI (`infernix`), `infernix-serve`, and
offline teacher-forced scoring (`infernix-perplexity`, `EnginePurpose::CausalScoring`). Delivered:

- Text generation with 1-8 resident lanes fixed at startup.
- Speculative decoding: MTP (`--spec mtp`), DFlash2 (`--spec dflash2`, the production setting) with
  a batched online cost fit, and n-gram copy proposals (`--ngram-draft-tokens`), combinable with a
  neural drafter (§11).
- QSA with every engine KV profile: `bf16`, `int8`, `fp8`, `nvfp4`, `k8v4`, `vq2`, `k4v2`
  (`--kv-dtype`; §14).
- Vision, selected at startup (`--vision`), with the tower kept in pinned host RAM by default
  (`--vision-offload auto|on|off`, §16).
- Three-tier expert placement — every routed expert in exactly one of VRAM, RAM or SSD — with
  runtime switching over the serving API (§9).
- The prefix cache with a pinned host tier and a Disk (SSD) tier moved through cuFile (§15).
- Prefill beside decode: a long prompt's calls ride inside the decode round's Forward (mixed
  rounds, §13).

Fidelity to the source checkpoint is still a requirement: tensors the source stores at 8 bits or
less (the NVFP4 routed experts, the block-FP8 MTP experts, the FP8 n-gram table) are imported
bit-exactly and run in the source's activation formats; tensors kept in BF16 stay BF16, or move to
8 bits only where that passes the quality gate (§6, §17).

### 1.2 Hardware

- Production and development run on one RTX PRO 6000 Blackwell: 96 GB, `sm_120a`, PCIe Gen5 x16,
  Linux, 89 GB of host RAM, one Gen5 NVMe holding the models. Every routed expert fits in VRAM on
  this card; a placement decides how much of it the engine keeps (§9).
- The RTX 5090 path (32 GB, Windows, PCIe Gen5 x8) must keep working. It is not measured on this
  machine; every 5090 number in this document is dated and comes from the implementation log at
  the base commit.
- Every measured number names the hardware it came from.

### 1.3 Status and measured speeds

The engine is in production on the RTX PRO 6000 (commit `e984e41`, 2026-10-30, `make t3`;
production flags: DFlash2 drafter, BF16 n-gram volume, FP8 KV, 6 lanes, 262,144-token context,
a RAM-only prefix cache — an 8 GiB Host tier, the Disk tier off since 2026-11-01). One server boot,
placements switched at runtime; tok/s from the server's own request log. Workload definitions are
in [performance.md](../performance.md).

| Placement | Experts VRAM / RAM / SSD | Decode, 1 request | Decode, 30K context | Prefill, 30K | 6 requests, total |
|---|---|---:|---:|---:|---:|
| `vram=100%` | 24,576 / 0 / 0 | 380 | 333 | 13,700 | 541 |
| `vram=65%,ram=15%,ssd=20%` | 15,974 / 3,658 / 4,944 | 232 | 220 | 9,750 | 380 |
| `vram=50%,ram=50%` | 12,288 / 12,288 / 0 | 263 | 208 | 8,160 | 388 |
| `vram=50%,ssd=50%` | 12,288 / 0 / 12,288 | 68 | 51 | 3,990 | 193 |

Switching back to `vram=100%` at the end of the same boot measured 379 / 287 / 13,600 / 525, so a
switch round trip leaves the engine whole. Switches take 1.7-8.6 s. RAM is worth far more than SSD
for experts that leave VRAM: half the experts in RAM decode at 263 tok/s, half on SSD at 68.

### 1.4 Still-open risks

- The RTX 5090 path is not measured on this machine. Changes are checked for it by review only.
- The GPU also drives the desktop, so a benchmark row can measure disturbed; the benchmark flags
  such rows and they are not comparable ([performance.md](../performance.md)).
- Verification rounds wider than 8 columns take MMA tiles that round BF16 dense outputs
  differently, so output at C > 1 can differ from C = 1 at near-ties. This is documented
  behaviour, and the construction-exact route on recipe C removes it (§17).
- Routing locality on long, diverse sessions is weaker than in the short replay traces the cache
  policy was tuned on (§9), so speeds at reduced placements fall with the hit rate; the measured
  rows above are the current evidence.
- The speculation cost fits are placement-scoped: a fully resident card's width cost cannot be
  extrapolated from a partially resident machine, and the fits reset on a placement switch (§4).
- VRAM headroom at `vram=100%` is small (90.3 GiB of process VRAM measured). A grow needs other
  programs to release memory first; a switch that cannot be reached stops where it can and
  reports it.

### 1.5 Non-goals

- Copying Strata or ninfer-ext code. Both were studied for mechanisms and measurements only (§3).
- Any tensor below its source precision, or below 8 bits where the source keeps BF16 (§6).
- Multi-GPU, AMD, or GPUs other than `sm_120a`.
- A generic model-independent offloading framework. The offload tier is shared between flash
  families through a plain geometry struct — no base class, no virtual interface (§7, MoE §2) —
  and stays specialized to routed-expert MoE families.

---

## 2. Model facts

The mathematics is defined by the upstream Transformers `qwen4_exp` model and SGLang's
`qwen4_exp_mtp`. The quantities below are verified against the checkpoint config
(`text_config` of the production NVFP4 checkpoint; values parse in
`src/models/qwen4_exp/config.cpp`).

| Quantity | Value |
|---|---|
| Layers | 48: 36 GDN (`linear_attention`) and 12 QSA (`qwen_sparse_attention`, every 4th from layer 3) |
| Hidden width | H = 2560 |
| Hyper-connection residual | 4 streams (10,240 wide); two HC mixers per layer, rank 320 |
| GDN | 16 key heads × 128, 48 value heads × 128, conv kernel 4, FP32 state |
| QSA | 24 Q heads, 2 KV heads, head dim 256; indexer 4×128 query, 1×128 key; budget 2048 tokens in 4-token blocks; partial rotary 0.25 |
| MoE | 512 routed experts, top-10, width 640; gated shared expert of width 640 |
| PLE | One injection before GDN layer 1 (`ple_layer_ids: [2]`, 1-based); 16 rows of 160 values per token (2560); n-gram size 3, 8 heads per n-gram; conv kernel 4 |
| MTP | One block: its own attention layer and its own 512-expert MoE |
| Vocabulary | 248,320; untied embedding and head, no final norm |
| Context | 262,144 positions |

Parameter counts are the upstream model card's: ~176B total — ~125B main model, ~51B n-gram table,
~4B MTP; ~6B active per token.

**Checkpoint composition** (ModelOpt `MIXED_PRECISION`): only three tensor groups are quantized.

- Routed experts: NVFP4, W4A4, with MSE-calibrated weight scales and each expert's own FP32
  `input_scale` (§6, §17).
- MTP routed experts: FP8 with 128×128 block scales.
- The n-gram table: per-tensor FP8 (a BF16 table volume is the exact alternative; §12).

Everything else, including the shared experts, is BF16.

**Byte budget**, from the shapes above:

| Item | Size |
|---|---|
| One routed expert (4,915,200 E2M1 weights + E4M3 scales per 16) | 2,764,800 B (2.64 MiB) |
| One MTP routed expert (FP8 + 300 FP32 block scales) | 4,916,400 B |
| Routed expert weights read per token (48 layers × top-10) | 1.33 GB |
| All routed experts (48 × 512 main + 512 MTP) | 67.9 GB + 2.5 GB |
| Dense text weights | 8.6 GB BF16 (recipe A); about 5.1 GB at 8-bit (recipe B); §6 for the recipes in use |
| Token embedding (248,320 × 2560) | 1.27 GB BF16; one row read per token |
| N-gram table (320,001,536 rows × 160 values) | 51.2 GB FP8 or 102.4 GB BF16 |
| KV per context token (13 attention layers, 2 KV heads, D = 256) | 27.5 KB `bf16`, 14.6 KB `int8`, 14.2 KB `fp8`, 8.3 KB `nvfp4`, 4.3 KB `vq2` (§14) |
| GDN recurrent state per sequence (48 heads × 128 × 128 FP32 × 36 layers) | 113 MB |
| GDN conv state per layer | 3 × 10,240 values (the trailing width-3 window) |

What follows from these facts:

- On the RTX PRO 6000 every routed expert fits in VRAM (67.9 GB of 96 GB). Placement decides how
  much of the card the engine keeps; at `vram=100%` the process measured 90.3 GiB of VRAM
  (63.3 experts / 19.3 KV / 7.7 other).
- On the RTX 5090 (32 GB) the experts do not fit: a third to two fifths are resident and every
  miss costs host DRAM bandwidth, whether the CPU computes it in place or PCIe stages it (§9, §10).
- The n-gram table (51.2 GB FP8, 102.4 GB BF16) is read row by row while the engine runs: from
  its NVMe volume by default, or pinned in host RAM where `--ngram-residency` and the host-memory
  ledger allow (§12, §12.5).
- The dense path, not the experts, sets the decode ceiling (§4.3).

---

## 3. Lessons from other implementations

The design was shaped against the engines that also run this model off one GPU: Strata (upstream
and its forks architectds/Strata, sergqwer/strata-nvfp4), Infernix and its forks (giveen/ninfer-ext,
kido5217/Infernix), NInfer dev, FreeToken, SGLang, vLLM, llama.cpp, TensorSharp, the
jpezzulli/sglang-rtxpro6000 kernel work, and the MoE-offload literature. This section keeps only
what still matters: the ideas adopted (with the section where each lives now), the ideas rejected
with the measurement that decided, the baselines still in use, and the standing lessons. Engine
surveys and dated review narratives are in git history. No external code is copied; the one copied
asset is the calibrated expert order (Apache 2.0, recorded in its file).

### 3.1 Adopted ideas

| Idea | From | Decided by | Now |
|---|---|---|---|
| Codebook decode without a local-memory table | Strata (codebooks in registers) | Strata SYCL port: expert call 80.8 -> 43.1 µs | §8: `e2m1_x2` (`__byte_perm`), `nvfp4_scaled_pair_bf16` (PTX `cvt`) |
| QSA selection that scales with context (multi-CTA scoring, histogram top-k) | Strata `cce52db` | vs one CTA per column: decode +4.9 / +10.6 / +34.6 % at 4K / 8K / 32K; 32K at 89 % of 4K (RTX 5090, 2026-10-05) | §14: `qsa_score_kernel`, `qsa_score_wide_kernel` (32+ columns) |
| Warm start from a saved LFRU state, plus a built-in calibrated order when none is saved | NInfer `kFlashNextExpertPriority` (f8a774f73) | first request vs empty cache: decode 147 -> 162.6 tok/s (code prompt), prefill 52 -> 66 tok/s (RTX 5090, 2026-10-17) | §9: `expert_state`, `calibrated_order.cpp` |
| Residual-A4 fast path for W4A16 wide experts (recipe C) | strata-nvfp4 | three terms are 35-43 % faster than the exact limb route at 4K-32K chunks; two terms left ~0.8 % row error and half the FP8-KV decode excess (RTX PRO 6000, 2026-11-02) | §13: `WideA16::kA4Residual`, three residual terms |
| Fused HC up projection + collapse (bit-exact Q8 MMA epilogue) | strata-nvfp4 | same output hash at 13 widths; 540 vs 580 µs per 4,096-column mixer call (RTX 5090, 2026-10-14) | §13: `hyper_connection_up_collapse.cu` |
| v4 HC read scheduling for mixer calls of 2+ columns | strata-nvfp4 `fused_gr.cu` | mixer 15.9 -> 13.1 µs at T 5, decode MTP +0.9 % (RTX 5090, 2026-10-16) | §8: `hc_down_partials_v4`, `hc_up_collapse_v4` |
| Prefill chunk 16,384 | kido5217 #76 | pp32768 9,639 -> 11,211 tok/s from chunk 4,096; the exact route is chunk-invariant (RTX 5090, 2026-10-15) | §13: `kPrefillChunk` |
| Halve a chunk whose wide workspace could never be lent | GitHub #5 | a 65,536-token chunk wanted 27 GiB and failed `cudaMalloc` after admission | §13: `fit_wide_work_to_free` |
| Admission re-check after each prefill walk span | GitHub #24 | a request waiting on KV growth: HTTP 503 after 8.3 s before, admitted after 3.6 s with the re-check | §15 |
| Branch tap where a prompt leaves its cached path | Strata PR #2843 | re-sent 40K transcript: 0.6-1.1 s vs 2.1-2.2 s to first token | §15: `kBranchTapMinTokens` |
| Sampled MTP drafts from the head's top four ranks | NInfer (sparse rejection) | chi-square exact against plain sampling; sampled decode geomean +2.7 % over six cells (RTX 5090, 2026-10-17) | §11: `kSampledDraftRanks = 4` |
| CPU computes misses in place, balanced against the PCIe copy | Fiddler, KTransformers, FreeToken `hybrid` | worker and split defaults measured on the i9-13900K host | §10 |
| LFRU, f / (age + 1), as the cache policy | Zhang 2608.07911 (replay study) | 15-22 % fewer misses than LRU at every capacity tested on this model's traces; pure frequency loses | §9: `LfruPolicy` |
| Promotions follow the tokens a round advances (draft-implied prefetch) | SP-MoE | replay: verify windows of 1, 2 and 4 tokens have the same misses per accepted token | §9, §11 |
| Token-determined n-gram rows prefetched off the critical path | DeepSeek Engram | a column's 2,560 B crosses PCIe in ~1-2 µs; the host reads rows hidden behind drafting | §12 |
| The Gold-Star-Infer runtime contract: hybrid sources, snapshot prefetch, coalescing, boundary snapshots, `/v1/decide`, no preemption | Gold-Star-Infer 5fbfc3dbe | its result checks exposed four latent faults at the port (2026-10-05) | §15, §18 |

### 3.2 Rejected ideas

- Victim landing (Strata `STRATA_ADAPT_FETCH=2`): one-off misses push out residents that still earn
  hits — plain decode −0.8-1.6 %, hit rate −0.4-0.7 points; promoting less gained more (191.3 vs
  184.3 tok/s MTP). RTX 5090, 2026-10-14.
- Promotion interval 8 / 16: +3.3 % MTP on one natural text, +0.4 % (drift-level) on a
  changing-topic corpus; the cache stops adapting when the text moves. Interval stays 4 (today
  scaled by the measured link rate, §9). RTX 5090, 2026-10-15.
- INT8 tensor cores for QSA prompt scores: the FP16 kernel already sits near the tensor-core bound;
  the int8-limb prototype was 12.5-13.1 % slower per call. RTX 5090, 2026-10-14.
- Draft cost from raw per-width round times (Strata `DraftPolicy`): round times carry the expert
  misses — decode 140.5 -> 133.8 tok/s, accepted drafts 2.10 -> 1.29. §11 instead fits one-row
  start-to-start times, gap-filtered, with `kWidthCost` as the prior. RTX 5090, 2026-10-15.
- 32,768-token prefill chunk: pp32768 +3.8 % but pp65536 −5.7 %. RTX 5090, 2026-10-16.
- Equal-width prefill calls: −0.6 % at 18K-20K prompts, bits unchanged. RTX 5090, 2026-10-16.
- QSA selection shared across chained MTP draft steps: within run spread (4K 127.9 -> 128.6 tok/s).
  RTX 5090, 2026-10-16.
- RMSNorm folded into the GDN `out_proj` epilogue: every CTA normalizes the whole activation —
  Q8 T 5 15.0 -> 51.8 µs per call. RTX 5090, 2026-10-16.
- NInfer's BF16 HC kernels: 1.11-1.55× at the op, but normwise error 2.45e-3 against 1.70e-3, below
  the mixer's criterion at decode widths. RTX 5090, 2026-10-16.
- Strata's multi-column narrow expert kernel: mean 1.19-1.50 columns per expert call at width 2-5;
  sharing saves only code and scale conversion, est. +0.5-1 %. Deferred. RTX 5090, 2026-10-17.
- Two QSA cells per step and empty-chunk merge skipping: decode attention is a few % of a round;
  the ceiling is below 1 %. RTX 5090, 2026-10-17.
- MTP catch-up folded into the first draft step (Strata `3fd0460ba2`): the catch-up costs 104 µs of
  a 10.1 ms round, so at most ~0.5-0.9 %. RTX 5090, 2026-10-16.
- Draft-column n-gram reads: the GPU waited at the n-gram gate in 0 of 210 gated
  rounds at C = 2 on first-time text. Parked on its trigger. RTX 5090, 2026-10-16.
- Router lookahead (Eliseev & Mazur, FATE): recall@10 65 %, and no file tier to feed it.
- HOBBIT low-precision copies for missed experts: changes the vendor's weights — the fidelity
  requirement (§1) forbids it.
- Per-layer cache quotas (AdapMoE, FATE): 3-34 % worse in replay; one global pool (§9).
- Learned next-use predictors (2608.07911) and SeqMoE's sequence predictor: the predictor scored
  below LFRU in the same replay study. Research option only.
- VMM-segment elastic VRAM resize (Strata #533): a capability without a product requirement.

### 3.3 Baselines still in use

- Strata, teacher-forced on the same token ids: `tools/flash_next/strata_compare.py` scores paired
  NLL, top-token agreement and KL over Strata's top 20 (Strata's `UNSLOTH_Q4.md` method). The
  quality reference for §17.
- Cache policies on this model's routing traces: `tools/expert_cache_replay/replay.py` replays the
  FreeToken community traces event-atomically — LRU, SLRU, ARC, LRU-2, S3FIFO, W-TinyLFU, decayed
  LFU, equal per-layer partitions, Strata's decayed-LFU exchange, Belady and LFRU. The policy
  evidence for §9.
- Strata's and its forks' own numbers, measured on their cards: references for a mechanism, not
  predictions for this machine (the production card is an RTX PRO 6000 behind a PCIe x8 link).
- SGLang all-resident on an RTX PRO 6000, one request: TPOT 11.4 ms plain and 6.0 ms with MTP
  accepting about 3 of 4 drafts (about 87 and 165 tok/s per request). The all-resident reference
  point on the same card.

### 3.4 Standing lessons

1. Hits never involve the host: the residency plan is made on the device inside the round, and the
   host hears about a layer only when it has a CPU-served miss (Strata's per-layer handshake cost
   48 serialized round trips per window). §8.
2. A miss is a host-DRAM bandwidth problem, not PCIe-versus-CPU: DRAM must stay busy during the
   GPU-only phases (prefetch) as well as the MoE phase (CPU compute). §8, §10.
3. The dense path sets the decode ceiling: dense tensors are 8-bit where quality allows, kernels
   run at the bandwidth roofline, and a layer costs few kernel launches (all three surveyed engines
   spent most of their token time on dense BF16 and launch overhead). §8, §17.
4. Recency × frequency (LFRU) over LRU; Belady's gap is the headroom. §9.
5. Speculation must not multiply misses per accepted token, and the drafter never waits on the host
   (ninfer-ext's MTP ran slower than plain decode: 82 vs 98.5 tok/s). §11.
6. NVMe reads stay off the critical path: every token is on the host before its rows are needed.
   §12.
7. Scale handling decides fidelity: engines that re-derive the vendor's scales silently run a
   different model; every quantized tensor is imported bit-exactly. §6, §17.

## 4. Performance model

### 4.1 Round time and the online cost fits

One decode round of T token columns costs, approximately:

```text
t_round ≈ t_gpu(T) + Σ_layers t_exposed(l) + t_draft
```

The design no longer projects these terms; it measures them online and uses the fits to choose
draft lengths (§11):

- A plain round and a drafting round of K extra columns are fitted as t = a + b·K by exponentially
  weighted least squares over consecutive one-row rounds, gap-filtered; until the fit is
  determined, the prior `kWidthCost` = 0.38 plain rounds per draft stands
  (`src/models/qwen4_exp/program/program_impl.h`). 0.38 was measured on the RTX 5090
  (2026-10-10): slightly high at 8 % expert misses, about right at 20 %.
- Batched DFlash2 rounds fit the plain round time, the drafting overhead and the marginal column
  cost separately; a single line through both underpriced wide rounds and drafted where it lost
  (§11.5).
- The fits describe the placement they were measured under (a rule of the offload tier, MoE
  §6.3). After a placement switch the 24 ms
  six-row SSD round priced drafts at a twentieth of the 11 ms all-VRAM round, and the plain EMA
  needed ~800 rounds to recover: C6 fell 444 → 351 tok/s. The fits now reset on every switch
  (`reset_round_costs`); the same matrix then measured 351 → 436 (2026-10-22, RTX PRO 6000).
- As a one-off reference fit on the RTX 5090 (2026-10-10): t = 7.2 ms + 1.95 ms·(W−1) + 0.04 ms
  per miss, R² 0.96-0.98 — but its zero-miss slope was not stable across cache sizes (0.27 vs 0.19
  plain rounds per column), which is why the cost is fitted, not fixed.

### 4.2 Ceilings and measured rates

Two ceilings bound the miss traffic of a partially resident round:

```text
host DRAM:  CPU-served + DMA'd bytes  ≤  BW_dram · t_round
PCIe:       DMA'd bytes               ≤  BW_pcie · t_round
```

Measured rates, RTX 5090 machine (2026-10-04, i9-13900K, DDR5-5800, PCIe Gen5 x8 link):

| Path | Rate |
|---|---|
| Copy-engine H2D DMA | 27.6 GB/s |
| SM zero-copy reads of pinned host memory, 16 KiB interleave in a ≤ 0.5 MiB window | 23.5-24.0 GB/s |
| SM zero-copy reads, the expert kernels' own pattern (40 CTAs × 46 KiB slices) | 11.9-13.9 GB/s |
| Staged misses (one PCIe crossing per miss at the interleave above) | ~24 GB/s per miss |
| CPU expert engine, 8 workers, cold 2.76 MB records | 52-55 µs per expert (~50 GB/s) |
| One staged miss over PCIe | ~115 µs |

The CPU and PCIe paths draw host DRAM in parallel and share its bandwidth: six workers beat eight,
and keeping a third of the misses on PCIe beat giving them all to the CPU (measured, tg512:
49.80 tok/s at 6 workers / 6 jobs / divisor 3, 2026-10-04). On the RTX PRO 6000 the effective cost
of each tier is the measured placement matrix of §1.3.

### 4.3 The dense path sets the decode ceiling

Measured round profiles:

- RTX 5090, recipe A, warm, 14.9 ms per round: dense GEMV 5.5 ms, experts 4.2 ms, LM head and
  router 1.8 ms, small kernels 2.2 ms, gaps 1.2 ms (2026-10-04).
- RTX PRO 6000, one-column round 5.49 ms: about 3.4 ms of it reading dense weights, and 2.1 ms of
  DRAM bandwidth left unused across kernel boundaries (2026-10-25).

So the dense bytes decide the ceiling: 8-bit dense recipes halve them where the quality gate
passes (§6), the dense kernels run at the bandwidth roofline (§8), and speculation amortizes the
dense read over accepted tokens (§11).

### 4.4 The levers, as built

| Lever | Mechanism | Section |
|---|---|---|
| Tokens per round | DFlash2/MTP with the online cost fits choosing K | §11 |
| The miss stall | Tier placement; staging overlapped with hit compute; CPU service of misses | §9, §10 |
| t_gpu | Dense recipes; one-column kernels; composed launches; PDL | §6, §8 |
| VRAM given to experts | The placement (client policy) and the LFRU swaps within it | §9 |

---

## 5. Design overview

```text
┌───────────────────── one GPU: RTX PRO 6000 96 GB (or RTX 5090 32 GB) ─────────────────────┐
│ Dense text weights (per recipe, §6) · drafter weights and KV · token embedding            │
│ GDN and conv state · decode workspace · prefill arena · vision encode window (§16)        │
│ ┌─ VRAM expert frames: the placement's VRAM share (24,576 at vram=100%) ──────────────┐    │
│ │ resident expert records · staging slots · per-layer frame tables · prefix blocks   │    │
│ └────────────────────────────────────────────────────────────────────────────────────┘    │
│ KV pages · residency tables (frame or −1; host pointer or null) · route log                 │
└────▲───────────────────▲───────────────────────────────▲────────────────────────▲─────────┘
     │ copy stream:      │ zero-copy mailboxes            │ pinned uploads        │ pinned slots,
     │ promotions        │ (CPU miss service)             │ (upload_pinned)       │ read in place
┌────┴───────────────────┴───────────────────────────────┴───────────────▲──────┴─────────────┐
│ host RAM (89 GB here; 96 GB on the 5090 machine)                          │   NVMe            │
│  RAM expert tier: pinned slots holding expert records (§9)                │  expert records   │
│  prefix-cache host tier, 8 GiB by default (§15)                           │  (SSD tier)       │
│  n-gram row cache: direct-mapped, batched reads (§12)                     │  n-gram volume   │
│  CPU expert engine: spin workers, exact W4A4/W4A16 arithmetic (§10)        │  (FP8 or BF16)    │
│  engine thread: cache policy, tiers, promotions, residency updates        │  Disk prefix tier │
│                                                                            │  (cuFile, §15)    │
└────────────────────────────────────────────────────────────────────────────┴──────────────────┘
```

A decode round at C = 1 with the DFlash2 drafter. The host loop is off the critical path; the
engine thread acts at round boundaries:

```text
embed (host row) ─ per layer: HC mixers ─ GDN or QSA ─ out_proj + inject ─ HC mlp
  router → route plan (in-place when a call stages nothing)
  routed experts: resident frames; misses staged into slots beside the hit compute
    (side stream) or served by the CPU team through mapped mailboxes; wide route for
    many columns; shared expert on its own stream; combine + inject
head (projection_fp32) ─ sample ─ DFlash2 proposal ─ verify (CUDA graph per B, W)
commit: GDN replay fold, ple_conv_commit, qsa_commit_tails, drafter catch-up
after_round (deferred): route log, LFRU accounting, promotions, tier boundaries
```

Four ideas carry the design as built:

- **Exact expert arithmetic makes placement free.** A routed expert's output bits are the
  same wherever it ran (§17; the principle is MoE §2), so the tiers move an expert between
  VRAM, RAM and SSD at any boundary and a promotion landing mid-round changes no result.
- **The device only looks up residency; the host owns policy.** Hits never involve the host.
  The engine thread applies the LFRU and tier decisions at round boundaries; evictions update the
  device tables on the compute stream, promotions copy on a copy stream that waits for them
  (§9, MoE §6.1).
- **The dense path runs at the bandwidth roofline.** One-column kernels, composed launches of
  independent projections, PDL chaining; every miss path overlaps hit compute (§8).
- **Memory is tiered by name, not one pool.** A placement names the VRAM / RAM / SSD expert counts
  and can be switched at runtime; within the counts the engine swaps hot experts in and cold ones
  out. Frames, KV pages, prefix blocks and the prefill arena share the device memory that is left
  (§9, §15).

---

## 6. Artifact and conversion

Conversion turns one local checkpoint into an Infernix v3 artifact (`.infernix`,
[artifact-container.md](artifact-container.md)) plus side files: the PLE n-gram volume (`*.ngram`,
§6.4) and, for the DFlash2 drafter, a companion artifact (§6.2). The tool is
`python -m tools.convert`; commands and current sizes are in the
[user guide](../qwen3_8-flash-next.md#convert). Recipes live in `tools/convert/official_recipes.py`,
the model adapter in `tools/convert/qwen4_exp.py`, the GGUF adapter in `tools/convert/qwen4_exp_gguf.py`.

The rule every recipe follows: a tensor the publisher quantized is imported bit-exactly — codes,
scales and multipliers as stored — and the artifact runs the publisher's arithmetic. Re-quantization
happens in two places only: the drafter, whose precision changes acceptance, never output, because
the main model verifies every draft (§11); and the dense classes recipe B moves to 8 bits for speed,
held by the quality gates of §6.6 and §17. The routed experts dominate the weight bytes (24,576
experts, 63.3 GiB of banks, §2), so their fidelity is the artifact's fidelity. The dense8 classes
are 97 % of the dense bytes; their cost is speed, not correctness (measured quality in §17).

### 6.1 Sources and the model adapter

- The adapter `tools/convert/qwen4_exp.py` builds the logical model — parameter names `text/`,
  `mtp/`, `vision/`, packing groups for the fused projections — from the one checkpoint given with
  `--model`. `validate_source` refuses every substitute tensor source. The checkpoint's PLE hash
  buffers (`layer_multipliers`, `ngram_heads_*`) must equal the config's derivation, or conversion
  fails.
- `tools/convert/qwen4_exp_gguf.py` builds the same logical model from a llama.cpp GGUF (`--gguf`,
  all shards of one split). Quantized tensors expose their stored block rows; recipes copy them
  verbatim. llama.cpp's export transforms are undone exactly and checked per tensor: the `+1` on
  zero-centred norm gammas (g − 1 must be a BF16 value), the GDN value heads in ggml's tiled order
  (rows permuted back; the output projection's columns cannot be, their K-blocks span two heads, so
  the config records `gdn_output_order: ggml_tiled` and the runtime reads through the permutation),
  `A_log` stored as `−exp(A_log)` (kept as FP32 A, `gdn_a: direct`), the indexer's fused q/k split
  concatenated again. The GGUF metadata must agree with the checkpoint config on every structural
  key.
- The GGUF holds no MTP head. `--model` names Qwen's checkpoint for the config, tokenizer and, with
  the `mtp` component, the MTP tensors; `tools/convert/fetch_tensors.py` HTTP range-reads just those
  tensors (about 2.7 GB of a 180 GB repository) at a pinned revision. With `--mmproj` the vision
  tower comes from the GGUF repository's BF16 mmproj: every value must be a BF16 value, and its
  metadata must match the checkpoint's `vision_config`.
- Options: `--components text,vision,mtp` (or `dflash2` alone, §6.2), `--proposal` (§6.2),
  `--ngram-out`/`--ngram-reuse` (§6.4), `--source NAME=PATH`, `--device cpu --workers N`. The report
  `OUT.conversion.json` records per-parameter formats and the provenance (source paths, recipe,
  GGUF/mmproj paths, ranking). The container splits at 32 GB into `.part-NNNN` files.

### 6.2 Recipes

| Recipe | Source | Exact import | Re-quantized |
|---|---|---|---|
| A `qwen3_8_flash_next_nvfp4` | a ModelOpt W4A4 export | everything: routed experts as `nvfp4_mul` banks with their per-matrix input scales; all BF16/FP32 tensors in their dtype | the MTP drafter |
| B `qwen3_8_flash_next_nvfp4_dense8` | A's source | as A | drafter, plus the dense8 classes below in `q8_g32_fp16` |
| C `qwen3_8_flash_next_nvfp4_orcarouter` | a weight-only (compressed-tensors) export | experts (codes, block scales, reciprocal global scale), FP8 row projections | drafter, plus B's 8-bit classes where the checkpoint is BF16 |
| `qwen3_8_flash_next_fp8` | Qwen's official FP8 checkpoint | experts (E4M3FN codes, BF16 128 × 128 tile scales), all BF16/FP32 tensors | drafter (its experts in `q2_g64_fp16`) |
| `qwen3_8_flash_next_gsq_rco_iq3s` | ISTA-DASLab GSQ-RCO IQ3_S GGUF | every GGUF block, the n-gram table byte for byte | drafter (experts `q2_g64_fp16`) |
| `qwen3_8_flash_next_dflash2` | a DFlash2 drafter checkpoint | — (a companion artifact; see below) | its projections in `q8_g32_fp16`, with the exceptions below |

Dense classes, recipes A/B/C (48 layers unless noted):

| Class | Source form | A | B | C |
|---|---|---|---|---|
| Routed experts (48 × 512) | ModelOpt NVFP4: E2M1 codes, E4M3FN per 16, FP32 `weight_scale_2` and `input_scale` per matrix | `nvfp4_mul`, W4A4 | as A | `nvfp4_mul`, W4A16 (no activation scales exist) |
| GDN query/key/value/z, `out_proj`; shared experts | BF16 (ModelOpt and BF16 exports), per-row FP8 (weight-only exports) | BF16 | `q8_g32_fp16` | exact `fp8_e4m3fn_row_bf16`, W8A16 |
| QSA query/gate/key/value | BF16, or per-row FP8 in a weight-only export | BF16 | BF16 | exact `fp8_e4m3fn_row_bf16`, W8A16 |
| QSA `o_proj` | BF16 | BF16 | `q8_g32_fp16` | exact `fp8_e4m3fn_row_bf16`, W8A16 |
| HC mixers (down/inject/up), PLE key/value projections, final mixer, `lm_head` | BF16 | BF16 | `q8_g32_fp16` | `q8_g32_fp16` |
| Router, shared-expert gate, GDN a/b, QSA indexer | BF16 | BF16 | BF16 | BF16 |
| Token embedding | BF16 | BF16, host-resident (§6.5) | same | same |
| Vision tower | BF16 | BF16 | BF16 | BF16 |
| MTP dense | BF16 | `q8_g32_fp16`; router and shared-expert gate stay BF16 | same | same |
| MTP routed experts (512) | block-FP8 per expert, or a BF16 fused export | `q4_g64_fp16`, MSE-chosen group scales (1.34 GB, resident) | same | same |
| N-gram table | FP8 shards, or BF16 shards | the checkpoint's FP8 words (§6.4) | same | per-tensor quantization of its BF16 table, which reproduces NVIDIA's FP8 words |

Notes:

- Recipe B's dense8 classes (629 tensors, 97 % of the dense bytes): the GDN projections, the QSA
  output projection, the shared experts, the PLE projections, the HC mixers, the final mixer and
  `lm_head`. Dense weight bytes read per token fall from about 8.6 GB (A, BF16) to about 5.1 GB.
  Every class takes `q8_g32_fp16` (`grouped_absmax`) directly: a 32-element group scale and an 8-bit
  integer give finer resolution than an FP8 row at 1.06 bytes per weight.
- Recipe C's bank multiplier is `fl32(1 / weight_global_scale)` — one IEEE FP32 division, the
  import's only rounding. The QSA query/gate/key/value projections (FP8) and the indexer (BF16) form
  two parents, which the model runs as two projections.
- The FP8 checkpoint has no quantized dense tensors; its experts are stored verbatim as
  `fp8_e4m3fn_b128` and run W8A16 with exact arithmetic (§17). Its BF16 n-gram table binds a BF16
  or FP8 volume (§6.4).
- The GGUF recipe stores each block format verbatim: the tiled formats (Q4_K, Q5_K, Q6_K, IQ4_XS)
  in `ggml_tiles16_v1`, row-major formats and the token embedding (a row gather) in `ggml_rows_v1`,
  the one Q8_0 tensor as identical `q8_g32_fp16` words (no code is −128), the experts as
  `ggml_expert_*_v1` banks (§6.3).
- Refusals keep the recipes on their sources: A and B refuse a weight-only bank ("weight-only
  experts take recipe C") and a block-FP8 bank ("take the FP8 recipe"); C refuses a ModelOpt bank;
  the FP8 recipe requires Qwen's block-FP8 experts; the GGUF recipe requires a GGUF-built model.
- The drafter (`_flash_next_drafter`, shared by A-C and the FP8/GGUF recipes): projections
  `q8_g32_fp16` except the router and shared-expert gate (BF16, as NVIDIA stores them); routed
  experts `q4_g64_fp16` (MSE, 1.34 GB) in A-C, `q2_g64_fp16` (0.71 GB) in the FP8 and GGUF recipes,
  the 0.63 GB difference going to expert-cache frames. Strata keeps its drafter at 2.25 bits; the
  Q2-versus-Q4 measurement is §11.2's (Q2 decoded faster).
- The proposal head (`--proposal`, for `--lm-head-draft`, §11): `lm_head` rows of the 131,072 most
  frequent tokens (a frequency-ranking fixture; special tokens forced in), `q4_g64_fp16`, 178 MB,
  plus its INT32 token ids, which the loader checks for uniqueness.
- The DFlash2 companion (`--components dflash2 --source dflash2=PATH`): a separate artifact holding
  only the drafter, loaded beside any artifact of that model with `--draft-model`. Every projection
  is `q8_g32_fp16` except the router, the shared-expert gate, the dynamic-convolution kernel
  projections and the selector's hidden projection, which stay BF16 with the norms, base kernels
  and codebooks. The drafter reads the target's hidden features and declares external Uses of the
  target's output or proposal head; its config records `target_config`, which the loader checks
  (§6.5).
- Artifacts on this workstation (2026-10-20, RTX PRO 6000, `--device cpu`): recipe C of the
  orcarouter checkpoint (72 GB, 8m40s, 27 GB peak), recipe A of a ModelOpt W4A4 export with a BF16
  PLE table (76 GB, 10m52s, 4 GB peak, reusing C's volume), the DFlash2 companion (741 MB, 4 s), and
  two volumes: FP8 (49 GB) and BF16 (109 GB).

### 6.3 Expert banks and model-role tensors

- **`nvfp4_mul`** ([tensor formats](tensor-formats.md#35-nvfp4_mul)): the words of `nvfp4` (E2M1
  codes, one E4M3FN scale per 16) with an FP32 **multiplier** `m_w` per matrix:
  `W[n,k] = e2m1(c) · e4m3fn(s) · m_w`. The engine's `nvfp4` stores a divisor; NVIDIA's
  `weight_scale_2` is a multiplier, and a divisor equals it only when the scale is a power of two.
  The routed experts are its only producer.
- Model-role tensors beside each layer's bank: the multiplier plane (FP32 `[experts, 3]`, gate, up,
  down order, at the end of the record region) and, for W4A4 banks, `expert_input_scales` (FP32
  `[experts, 3]`, the checkpoint's `input_scale` words as stored). The loader forms
  `α = fl32(m_w · g)` once per matrix on the host — one IEEE binary32 product per matrix, compiled
  without FP contraction, positive-finite checked (`src/models/qwen4_exp/load.cpp:354`). CPU and GPU
  read the same words, so the epilogue multiplier is identical on both (§10). A W4A16 bank has no
  input scales; `α` is the stored multipliers alone.
- **`nvfp4_expert_rg16_v1`** ([storage layouts](storage-layouts.md#6-nvfp4_expert_rg16_v1)): one
  contiguous record per expert, 2,764,800 B = 675 × 4 KiB, that the GPU and the CPU both read
  directly. Only one copy of the bank fits in host RAM and runtime repacking is excluded, so the
  stored record must serve every consumer: DMA into a frame; the narrow GPU kernels, where a warp's
  32 four-byte loads cover one unit's 128 code bytes and a thread's word decodes to one register
  pair of the `mma.sync.m16n8k32.s8` A fragment, or feeds `dp4a` (§8.5); the wide prefill route,
  which converts units into the `mxf4nvf4` operand order in shared memory, so no tensor-core layout
  leaks into the record; the CPU engine, where one 32-byte quad becomes one AVX-512 vector after a
  broadcast load, a masked shift and an AND, decoded by `vpshufb` and multiplied by `vpdpbusd`
  (§10.2). Rows form groups of 16; each row group and 16-element block is one 144-byte unit (128
  code bytes, then 16 scale bytes). The gate/up matrix (1,280 rows, K = 2,560, rows interleaved
  gate/up) takes 1,843,200 B; the down matrix (2,560 rows, K = 640) 921,600 B.
- The drafter's experts are ordinary `row_split_k128_v1` tensors: they are resident and only the
  GPU reads them.
- **`fp8_e4m3fn_b128` in `fp8_expert_b128_v1`** ([storage layouts](storage-layouts.md#9-fp8_expert_b128_v1)):
  E4M3FN codes row-major, then the BF16 tile-scale words, verbatim; 4.69 MiB per expert, 112.6 GiB
  for 24,576 experts. A16 only; the exact W8A16 arithmetic is §17's.
- **GGUF banks `ggml_expert_iq4_nl_v1` / `ggml_expert_q2_0_v1`** ([storage
  layouts](storage-layouts.md#8-gguf-expert-banks-ggml_expert_iq4_nl_v1-ggml_expert_q2_0_v1)): gate
  and up share one format (IQ2_S, IQ3_XXS, IQ3_S or IQ4_XS), the down format is IQ4_NL or Q2_0 and
  names the layout. One record per expert: gate rows, up rows, down rows, zero padding to 4 KiB.
  Record sizes vary by layer.
- Loader checks (`src/models/qwen4_exp/load.cpp:523`): every layer's bank is one kind;
  `nvfp4_expert_rg16_v1` requires `nvfp4_mul`; ggml banks belong to a GGUF import and only there;
  exact banks are A16-only; a bank's activation policy must agree with the presence of activation
  scales.

### 6.4 The n-gram volume

- The table is its own file (`*.ngram`, placeable on another NVMe drive): 4 KiB blocks, a header
  with the geometry and a 16-byte volume id, and — since 2026-10-18 — the FP8 rows' per-tensor scale
  (BF16 bits) after the fixed fields, which lets one volume serve every artifact of the model: the
  engine knows a published volume's scale by its id.
- 320,001,536 rows of 160 bytes, 25 per block. FP8 rows are the checkpoint's words unchanged. A
  BF16 table is quantized per tensor by ModelOpt's own rule — `s = bf16(amax / 448)`,
  `code = e4m3_rn_satfinite(bf16(v / s))` — which reproduces NVIDIA's FP8 table of the same base
  model word for word; recipe C therefore reuses the volume A or B wrote. One scalar scale, no
  per-row plane: one I/O per row instead of two.
- `--ngram-reuse VOLUME` binds a new artifact to an existing volume instead of writing one. The
  converter checks the header geometry, the file size and 512 sampled rows against the checkpoint,
  then records the volume id in the artifact's config. A volume of another checkpoint is refused.
- A BF16 volume (`tools/flash_next/ngram_bf16.py`) holds the unquantized rows: 320 bytes each, 12
  per block, about 109 GB. Any artifact runs with it through `--ngram-volume`; the PLE reads the
  BF16 rows as its embedding, with no dequantization. `--dequantize` writes an FP8 table's values
  as BF16 to qualify that route; it adds no precision. Runtime behavior is §12's.

### 6.5 Residency and materialization

Residency is a property of a Use and the startup options, not of the stored bytes. The same
artifact loads all-device on a larger card.

| Residency | Backing | Users (this family) |
|---|---|---|
| `device` | uploaded at load | dense weights, the drafter, the vision tower without offload |
| `values` | read into Model memory at load | expert input scales, proposal token ids |
| `host_pinned` | one registered host block | routed expert banks, the token embedding, the vision tower under offload |
| `streamed` | never materialized; the artifact is read in place | expert banks behind the SSD tier |

- Pinned loading (`src/artifact/materializer.cpp`): available host memory is checked first (pinned
  bytes, later pins and the `--ram-headroom-mib` reserve against memory available now, not an
  earlier snapshot). One committed block (`RegisteredHostBuffer`). A worker team (up to 8 threads,
  64 MiB chunks) zeroes pages ahead of the reads, so the OS's page zeroing runs beside the disk.
  The reads are unbuffered (`O_DIRECT`; an aligned bounce buffer of at most 8 MiB per thread covers
  unaligned tails) straight into the block: never a second host copy, never the page cache. The
  block is registered whole once; device code reads it at the host-mapped device address, which
  differs from the host address under WDDM.
- Streamed banks (the SSD tier, §9): the records stay in the artifact file and are read in place;
  only each bank's scale tail is read into Model memory at load.
- Vision offload (`vision && vision_offload`; `auto` by default in serving) pins the BF16 tower and
  streams it per encode window through borrowed device memory (§16).
- The DFlash2 companion has its own reader and materialization beside the main artifact. It declares
  its Use of the target's output or proposal head; the loader takes the head's row count and
  fused-head flag from the main artifact's stored format.
- The 16-bit-activation reference (`a16_activations`) binds every Use A16Only, so W4A4 banks run
  W4A16 and their input scales stay unused. It is the quality reference: causal scoring only, on the
  exact route, no fast paths (§17).

### 6.6 Verification and qualification

- `python -m tools.flash_next.verify_artifact` compares every exact parameter with the checkpoint:
  each expert's codes, block scales and multiplier decoded from its record (or each
  `fp8_expert_b128_v1` record byte for byte), every FP8 row projection's codes and row scales,
  every BF16 tensor bit for bit, every FP32 widening of a BF16 tensor, the input-scale words, and
  the n-gram volume's header, geometry, id and rows (`--ngram-rows all` or a sample; BF16 volumes
  through `bf16_rows_match`). Re-quantized parameters are listed per component; proposal bindings
  are counted apart. Source names and axes come from the converter's logical model, so a pass covers
  the writer, the recipe and the bindings.
- `python -m tools.flash_next.verify_gguf_artifact` compares a GGUF import with the GGUF: expert
  records, ggml rows (a tiled matrix after untiling; padding rows must be zero blocks), tensors, the
  n-gram volume, and the vision tower against the mmproj — optionally bit for bit against another
  artifact's vision.
- `python -m tools.flash_next.inspect_checkpoint` reads `config.json`, `hf_quant_config.json` and
  the safetensors headers: the source-form audit before a recipe is trusted.
- Recipe B is qualified against recipe A on the fixed corpus: mean KL ≤ 0.005 nats, p99 per-token
  KL ≤ 0.08, top-1 agreement ≥ 98 %, perplexity increase ≤ 0.5 %, and each 8-bit class alone mean
  KL ≤ 0.002. Recipe A is qualified by word equality alone; its reference agreement against an
  independent runtime serving the same checkpoint (SGLang, vLLM) is expected to be noise from
  activation rounding. The gates and measured values are §17's.
- Word-equality results: 2026-10-08 recipes A (24,576 experts, 1,616 tensors, 320,001,536 n-gram
  rows, no mismatch) and B (987 tensors, no mismatch, 629 parameters re-quantized); 2026-10-09
  recipe C (1,323 tensors, every row equal, teacher-forced perplexity 4.772 against Strata
  UD-Q4_K_XL's 4.844); 2026-10-17 the GGUF (385 GGUF and 1,218 plain tensors) and FP8 (1,615
  tensors) imports; 2026-10-20 recipe C on this workstation (0 expert, tensor, input-scale and
  n-gram-row mismatches, 10 minutes).

**Rejected**

- The first recipe: BF16 dense moved to NVFP4 and FP8, MTP experts re-quantized to NVFP4; withdrawn
  for exactness — NVIDIA's weights and calibration are kept instead (2026-10-03).
- `fp8_e4m3fn_block128_f32` + `block128_scale_v1` for the MTP experts: registered, never produced
  by any recipe, removed 2026-10-08 in a review of registered-but-unused formats; the drafter takes
`q4_g64_fp16`.
- A per-class try order for recipe B (FP8 rows, then Q8, then BF16): every class is `q8_g32_fp16` —
  finer resolution at 1.06 bytes per weight.
- Importing Qwen's FP8 checkpoint for recipe B's dense classes: that checkpoint has no quantized
  dense tensors; the import became its own exact recipe instead (2026-10-12).
- Recipe F-B (B's 8-bit dense over the FP8 checkpoint's experts): built and measured (perplexity
  3.4863 against 3.4870), removed 2026-10-12 to publish the exact artifact only.
- Exact MTP experts: the drafter only changes acceptance; Q2 experts decoded faster (§11.2).
- The engine's `nvfp4` divisor for NVIDIA's weights: it equals the stored multiplier only for powers
  of two; `nvfp4_mul` keeps the multiplier.
- Loading the GGUF in place, or dequantizing to FP32 and re-quantizing: the v3 container is the only
  artifact, runtime repacking is excluded, and a second quantization loses exactness.

## 7. Ownership

The boundaries are those of the [engine architecture](engine-architecture.md), which is the
authority; read its §2 before changing one. This family adds three model-side layers: the family
itself, a shared offload tier, and a shared Qwen frontend.

| Component | Owner | Location |
|---|---|---|
| Mathematics, config, binding, block order, MTP/DFlash2 alignment, tuned shapes | Model | `src/models/qwen4_exp/` (`config.*`, `load.*`, `instance.*`, `model.h`; `execution/` forward and parameters; `shapes/`) |
| Program: mutable state, placement, residency, tiers, CUDA graphs, prefix cache, n-gram volume, vision window, scoring, draft-length policy, fast paths | Program (allocated at startup) | `src/models/qwen4_exp/program/` (`program_impl.h`, `prefix/`, `ngram_volume.*`, `vision_window.*`, `score.cpp`, `dflash2_program.cpp`, `draft_length.h`, `fast_paths.*`, `prefill_walk.cpp`, `rope_positions.*`, `constraint.cpp`) |
| Expert residency machinery shared by the flash families: LFRU policy, VRAM frames and tables, placement, SSD records, RAM/SSD tier controllers, warm-start state, calibrated order, route trace, VRAM monitor | Offload tier (composed by each Program; no base class, no virtual interface; [MoE offload design](moe-offload-design.md) §1) | `src/models/offload/` (`expert_cache.*`, `expert_residency.*`, `expert_placement.*`, `expert_store.*`, `host_tier.*`, `host_expert_tier.*`, `expert_state.*`, `calibrated_order.*`, `route_trace.*`, `vram_monitor.*`, `geometry.h`) |
| Frontend shared by the Qwen families: tokenizer, chat template, processor, prompt layout, tool grammar, reasoning loop, media cache, vision tower and weight stream, n-gram proposer, prefix block keys | Shared Qwen frontend | `src/models/qwen_common/` (`frontend/`, `execution/`, `program/`, `load/`) |
| Routed-expert MoE: routing, residency classification, narrow and wide GPU kernels, CPU engine, placement-invariant contract (output = MoE(x) independent of residency) | Op family `offloaded_sparse_moe` | `include/infernix/ops/offloaded_sparse_moe.h`; `src/ops/offloaded_sparse_moe/` (`cuda/moe_layer.cu`, `cuda/narrow_expert.cu`, `cuda/wide_expert.cu`, `cpu/` the expert team and miss service) |
| Canonical routed-expert arithmetic (A4 quantizer, E2M1/E4M3 integer decode, int64 block sums; one header for CPU and GPU) | Op common code | `src/ops/common/canonical_math.h` |
| Drafter's resident experts | Op `resident_moe` | `include/infernix/ops/resident_moe.h`, `src/ops/resident_moe/` |
| HC mixer and inject; QSA prep, pooling, select, sparse attention; PLE gather, projections, gate, conv; LM head and router projections; GDN | Ops `hyper_connection`, `qsa`, `ple`, `projection_fp32`, `gated_delta_net`, `causal_conv1d_silu` | `include/infernix/ops/`, `src/ops/<op>/` |
| N-gram row ids (exact integer oracle in `infernix_qwen4_exp_ngram_hash_test`) | Model frontend, called by the Program on the host | `src/models/qwen4_exp/frontend/ngram_hash.*` |
| Pinned arenas, huge pages, unbuffered batched file reads, `upload_pinned`, cuFile I/O, VMM arenas, PDL | Core (model-independent physical and transfer primitives) | `src/core/` (`arena.*`, `vmm_arena.*`, `read_only_file*`, `device.*`, `cufile_io.*`, `pdl.cuh`) |
| Compile-time launch composition of Op kernel bodies (fused decode launches) | Op common code | `src/ops/common/blocks.cuh` |
| Weight formats (`nvfp4_mul`, `fp8_e4m3fn_block128_f32`, `q8_g32_fp16`, …) and layouts (`nvfp4_expert_rg16_v1`, …); ModelOpt / GGUF source readers | Artifact registries; converter | `src/artifact/formats.*`, `layouts.*`; `tools/convert/sources/` ([tensor formats](tensor-formats.md), [storage layouts](storage-layouts.md), §6) |

Residency is an **execution resource** for `offloaded_sparse_moe`, like a stream or a workspace:
by the [Op rules](op-development.md#2-op-admission-and-semantic-boundary) it may choose the
implementation but must not change the result. Placement invariance is a deliberate exception to
the rule that a contract must not freeze bitwise equality
([op development §3](op-development.md#3-contract-headers)): residency changes every round, so
without it the output would depend on cache history and CPU load. The exception covers only the
arithmetic of the routed-expert narrow route (§17); every other Op keeps the ordinary
oracle-and-tolerance contract.

---

## 8. Decode pipeline

Code: `src/models/qwen4_exp/program/` (round, graphs, io), `src/models/qwen4_exp/execution/`
(per-layer composition), `src/ops/offloaded_sparse_moe/` (routing, dispatch, expert kernels,
CPU channel). This section describes one decode round as built; §9 owns frames and placement,
§10 the CPU expert arithmetic, §11 the drafters and verify widths, §12 the n-gram gate, §14 the
QSA attention and KV appends, §17 the expert numerics.

### 8.1 Threads and streams

| Thread | Role |
|---|---|
| Engine thread | Runs `ProgramImpl::decode` and `commit` (the runtime's worker; §7). Stages io, replays the round's graph, synchronizes, reads the sampled tokens, settles the round. |
| CPU miss service | `ops::offloaded_moe::CpuMissService`: one service thread (worker 0) plus a worker team. Default 6 workers (16 for a GGUF import's banks, `ProgramOptions`). Spins while requests arrive; after 20 ms idle it waits in ~0.5 ms steps and `wake()` revives it at round submission. A stopped service fails every later call (`kErrorHostSilent`). |
| SSD tier read agent | `HostExpertTier`'s agent thread answers fetch requests (§9). |
| VRAM monitor | Wakes an idle engine so it returns KV pages; reports free-VRAM pressure but never resizes the expert cache itself (§9). |

Host threads are not pinned. On Windows the process is exempted from EcoQoS power throttling once,
when an Engine is constructed (`core/power_throttling`): the exemption is the whole gain
(+69-162 % on tg8k workloads, i9-13900K, 2026-10-07), a P-core affinity set adds nothing and pinning
each thread costs 3-4 %. No placement code is kept.

GPU streams of a round: the compute stream (captured into the round's graph), `ForwardExperts::
shared_stream` (the shared expert, forked at the layer's MoE input and joined before the combine),
and `ForwardExperts::overlap_stream` — the fork stream for decode and verification staging, and the
double-buffered staging stream of prefill chunks (§13; not captured). The SSD tier owns its own copy
streams (§9).

### 8.2 Host-device protocol

GPU↔host signalling uses mapped pinned memory and sequence words. The producer writes the payload,
executes `__threadfence_system()`, then stores a never-zero sequence; the consumer polls the word
and reads the payload after it matches. Host memory that the host rewrites between rounds (CPU
outputs, tier slots) is read with `.cv` loads (`__ldcv`) so a stale L2 line is never used.

| Channel | Direction | Content |
|---|---|---|
| Round io | host → device | ids, positions, RoPE positions, block starts, slots, rows. One `upload_pinned` copy, read by the SMs through the mapping: small per-round inputs never queue behind bulk copy-engine transfers of other streams (§13's promotions share that engine in FIFO order). |
| N-gram row gate | host → device | `upload_pinned_when`: the copy waits in stream order until the pinned ready word equals the device word; the host publishes the rows, then `publish_pinned_word`. A wait longer than 120 s stores its fault in the device error word and skips the copy (§12). |
| CPU miss channel | device → host → device | `MoeCpuChannel`: `MissRequest` (jobs, experts, record pointers, per-job columns), the compacted x columns, mapped y outputs, `done` (the answered sequence), `status`, `heartbeat`, and a device sequence counter. Owned by the miss service (§10). |
| Fetch channel | device → host → device | `MoeFetchChannel`: `FetchRequest`/`FetchResponse`/`consumed` plus heartbeat; serves SSD-only experts (§9). |
| Error word | device → host | Mapped u32, `layer << 16 \| code`. A call that cannot serve an expert writes it, and so do the Forward's own kernels' faults — the token embedding's id outside the vocabulary (`kErrorTokenId`) and the n-gram gate past its bound (`kErrorRowGateTimeout`); the Program checks it after every synchronization and fails the round. |
| Sampler non-finite flags | device → host | Per lane: `ops::SamplingConfig::nonfinite`, a mapped i32 the sampling and acceptance kernels set when a row's selection (or a verification column's target) has no finite value; the token they publish in its place is a placeholder (0). The Program folds the flag into the row's `runtime::PendingRowFailure::NonFiniteLogits` after the round, and the Engine fails only that request — it commits nothing, publishes no prefix-cache entry from the failing round and is released FailedReleased — while the other rows commit and the Engine keeps serving (port of upstream e053f0d5's mechanism; valid inputs keep their bits). |
| Route log | device → host | The round's routed ids, downloaded after the round for the cache policy (§9). |
| Landing tables | host → device, device | I32 [layers][16], uploaded by `before_round`; the stage kernels write each landed expert (§9). |

Polling discipline: one thread per CTA polls a host word, with `__nanosleep(200)` backoff. The
fetch channel elects one poller CTA per call (`atomicCAS` token) that mirrors the host's answer into
device memory; the CPU wait runs `min(8, jobs)` waiting CTAs. Every wait is bounded: a heartbeat that
stands still for 10 s (`kHeartbeatTimeoutNs`) fails the call through the error word; a channel
without a heartbeat is bounded at 2 s. No wait traps (port of upstream 3b34b6e7): a trap loses the
CUDA context and every lane with it, and on sm_120 one has hung the driver until a power cycle; a
channel without an error word goes unreported instead.

### 8.3 Per-layer kernel sequence

A GDN decode layer, in launch order (a QSA layer replaces its middle with the projections, the
fused prepare, attention and the output gate; §14):

1. The attention mixer `hc_down_partials` + `hc_up_collapse` (`hyper_connection_mix`): two kernels,
   every intermediate FP32, weights loaded before the PDL wait. At two or more columns the same
   arithmetic runs Strata v4's read scheduling (token groups sized to the call). The fused route
   serves up to 28 columns (16 when the norm weight is not 16-byte aligned); wider calls run the
   composed steps.
2. The GDN input projection `gdn_input_proj` — the Q8 [16384, 2560] form writes q/k/v and z to
   their destinations itself, no split copies — and the control projection [96, 2560]. The
   `gdn-conv-fused` fast path runs the causal convolution in the projection's epilogue (Q8 weights,
   one-column decode rounds).
3. `gated_delta_net_batch_update_gating`: the gates are evaluated inside the recurrent update at
   decode width, so no gating kernel runs between the convolution and the update.
4. `gated_rmsnorm`, then the output projection [2560, 6144].
5. The MLP mixer: at one column the inject of the mixer output is folded into K1a
   (`hyper_connection_inject_mix`); at wider rounds one inject kernel runs before the mix pair.
6. The MoE: the shared expert on its own stream from the layer's input; the router projection; the
   fused route/dispatch/plan kernel; the expert passes; the CPU wait; `moe_combine_inject` (§8.4-8.6).

Per layer the chain is about 25 kernels — 1,139 per 6 x 1 round (measured, RTX PRO 6000, 2026-10-22).
Every launch in a captured graph is a programmatic dependent (`pdl::launch_consumer`): a kernel loads
its weights, then `pdl::wait_for_dependencies()`, and `trigger_dependents()` once its own streaming
is done (`pdl.cuh`). Eager launches (prefill, forced tokens) stay stream-ordered. Kernel boundaries
therefore do not appear as idle time: under PDL a kernel is always running (no kernel ran for 47 µs
of an 11.5 ms round, 0.4 %). The head's Q8 weights are read once per call (636 MB) with an
L2 evict-first hint, so the stream does not flush the lines the drafter reuses.

Launch-boundary fusions, measured end to end on the RTX 5090 (Dense8, INT8 KV, tg512, ABBA,
2026-10-12): the router projection as a PDL consumer (+0.4 % plain), the gates inside the recurrent
update (+0.3 % plain), route and dispatch in one kernel (within noise alone), the combine's inject
form (+0.5 % MTP), the CPU plan inside the route kernel (+0.3 % MTP), the remaining small kernels
and RoPE as programmatic dependents (+0.25 / +0.17 % MTP). The last four together took plain tg512
from 128.60 to 130.51 tok/s (+1.5 %) and MTP from 177.86 to 179.50 (+0.9 %). The same week: K1b
fetching its weights during K1a's reductions (+2.2 % plain, +1.3 % MTP; the mixer chain measured
10.58 → 10.07 µs at T = 1), the GDN chain's remaining plain launches as dependents (+0.3-0.4 %), the
head's evict-first policy (+0.3 % MTP). The shared small kernels also serve Qwen3.8-27B, where the
same changes measured +0.1-0.2 %.

The GDN front end writes no split copies (2026-10-12, RTX 5090): plain 118.64 → 120.81 tok/s (+1.8 %),
MTP K = 4 170.32 → 172.13 (+1.1 %, identical acceptance), pp4096 unchanged.

**Where a 6 x 1 round goes** (RTX PRO 6000, 188 SMs, 1.79 TB/s; the Uncensored NVFP4 artifact, T1 at
context 1024, 2026-10-22). Of 11.0 ms device-busy time, 3.9 ms (35 %) leaves DRAM bandwidth unused;
streaming kernels top out near 90 % of DRAM, so about 1.1 ms of that is not recoverable and 2.8 ms is.

| Kernel | Time per round | DRAM |
|---|---:|---:|
| `gate_up_rows_kernel_a16` (routed experts, gate/up) | 3.02 ms | 89.9 % |
| `fp8_a16_simt_kernel` (dense FP8 projections) | 2.15 ms | 63.5 % |
| `down_rows_kernel_a16` (routed experts, down) | 1.92 ms | 71.6 % |
| `hc_up_collapse_v4_kernel` | 0.86 ms | 41.3 % |
| `hc_down_partials_v4_kernel` | 0.48 ms | 28.1 % |
| `recurrent_batch_update_kernel` (GDN state) | 0.52 ms | 63.0 % |
| `projection_q8_kernel` (head) | 0.52 ms | 73.5 % |
| `attention_kernel` (QSA decode) | 0.27 ms | 11.2 % |
| `projection_kernel` (router) | 0.28 ms | 22.0 % |
| ~20 small kernels (inject, combine, norms, RoPE, KV append, QSA index) | ~0.7 ms | 1-60 % |

The unused-bandwidth share moves little across shapes — 34.2 % at 1 x 1, 31.7 % at 6 x 1, 36.0 % at
1 x 4, 35.5 % at 6 x 2, 38.8 % at 6 x 3 (2026-10-31) — but the largest pool moves with the width: at
6 x 3 the hyper-connection pair holds 2.1 ms of the 7.5 ms unused beside the wide projections; at
1 x 1 the dense FP8 GEMV and the composed GDN kernel lead, and the routed down pass falls to 41.8 %
DRAM (40 CTAs per job at low occupancy, the per-CTA dependency chain exposed). `make t1-kernels
KSHAPE=S METRICS=1` publishes this table for any shape.

### 8.4 Router, route and CPU plan

The router projection (`projection_fp32`, the [513, 2560] BF16 router row plus the shared-expert
gate row) loads its weight rows into registers before staging x, and is a PDL consumer so its DRAM
trip overlaps the mixer's tail.

Calls of at most 32 columns and 1024 routed entries route, dispatch and plan in one 1024-thread CTA
(`route_dispatch_plan_kernel`), launched as a programmatic dependent of the projection:

1. **Route.** One warp per column: exact top-k by repeated arg-max over order-preserving keys, the
   lower expert id winning exact ties; route weights are the softmax over the selected logits,
   summed in selection order; the shared gate is the row's sigmoid. The selection and its order are
   fixed, so the route is deterministic.
2. **Dispatch.** Shared-atomic counts (no memset node in the graph), one shuffle scan of counts and
   used flags, the scatter, the job list in ascending expert id, and the route-log copy. Every
   distinct expert of the call becomes one job with its entries grouped.
3. **CPU plan.** After a barrier of the whole CTA, its first 256 threads run the plan body of
   `cpu_plan_kernel` (§8.6) on a named barrier, so the host request follows the dispatch with no
   kernel boundary. `moe_experts` then skips its own plan.

Wider calls run `route_kernel` (a warp per column) and the separate count/scan/scatter kernels. A
call that stages nothing and reads every record where it lies (a fully resident placement of an NVFP4
A16 bank) takes the in-place form `route_dispatch_resolve_kernel`: CTA 0 routes and dispatches while
resolving every job's record, CTA 1 + t encodes x column t — the encode and stage kernels do not
launch at all.

There is no device-side prediction of the next layer's route. A lookahead prefetch of the next
layer's experts was designed and never built: expert movement is host policy (§9), and the CPU
wait warms the next layer's dense weights instead (§8.6). A placement that holds every expert sets
`ForwardExperts::resident`: calls then run one pass over every job with no staging, fork, landing,
CPU or fetch channel — that machinery costs 0.37 ms of a 6 x 1 round in which no call misses
(0.57 ms of a 6 x 2) — and the graphs are captured again when it changes.

### 8.5 Expert kernels

The narrow route serves a job of at most 8 columns one column at a time, one pass over the job's
weights per column. It beat an eight-column form at op level at every width from 1 to 64 columns
per expert (RTX 5090, 2026-10-11; the eight-column form was removed).

**W4A4** (`gate_up_kernel` / `down_kernel`): a gate/up CTA owns two row groups of one expert (one
A4 block of h) and stages its 46 KB slice into shared memory (two CTAs per SM); a down CTA owns four
output row groups. A unit's code byte is converted with `cvt.rn.f16x2.e2m1x2` into a half2 pair and
the products are HMUL2/HFMA2: the four-term sums are exact in FP16, and four times them is the
lane's integer share of the block product, accumulated into exact int64 row sums (§17). The outputs
equal the CPU engine's bits wherever the expert ran. Instructions per layer call fell from 12.3 to
8.1 M (gate/up) and 6.8 to 3.6 M (down); op bench cold T = 1 47.7 → 41.5 µs; end to end plain
+3.74 %, MTP +4.55 % (RTX 5090, 2026-10-10). `down_kernel` launches as a programmatic dependent of
the gate/up pass and stages its weights before the PDL wait — only h, the pass's output, is read
after it — for another +3.3 % plain, +2.3 % MTP.

**W4A16** (`gate_up_rows_kernel_a16` / `down_rows_kernel_a16`, banks without stored activation
scales): x is encoded once per call (`encode_x_acts_kernel`), a gate/up CTA takes one row group with
its weights in registers (three CTAs per SM, 80 registers), and the products are exact dp4a over
byte limbs with wrapping int32 partials (§17). One-column passes beat multi-column forms at every
width measured — 1-3 columns per expert, 6-48 per call (RTX PRO 6000, 2026-10-21) — at 33 / 109 /
191 µs per layer call at T = 1 / 6 / 12 (2026-10-24).

**Wide route** (a call of more than 8 columns, per-expert column count above 8): a block-scaled
tensor-core grouped GEMM (`mma.kind::mxf4nvf4.block_scale`) with the same canonical A4 inputs; the
tensor core accumulates in its own FP32 order, so the route is qualified against the FP64 oracle
(§17). The route depends only on the column count and the expert's stored scales, never on
placement, so outputs stay placement-invariant. Experts whose gate and up input scales differ keep
the narrow route at every width.

**Staging.** The staging mechanism — a miss copied into a staging slot beside the hit compute,
`.cv` loads of tier slots, the first misses in job order landing in reserved frames — is MoE
§6.1, the fork and overlap streams §8.1's. `stage_kernel` here runs 16 CTAs of 16 KiB chunks in
one compact read window: pinned host memory read by many CTAs at scattered offsets, as the expert
kernels would, runs at under half the copy rate (RTX 5090). `--expert-tail-skip` (§18) is an
approximate fast path that skips the smallest-rank routed experts of a generation round and
zeroes their outputs.

### 8.6 Miss service and arbitration

The arbitration rules — SSD-only misses first (only the CPU serves them without a fetch), then
host-memory misses fewest columns first in job order, up to the call's cap, with the host share
`misses − misses / divisor` — and the wait kernel's heartbeat contract are MoE §6.1. The divisor
is derived at startup from the measured link and CPU rates (§10.5; the rule and its step
evidence are MoE §5.6's), or fixed by `--cpu-pcie-divisor`. A GGUF import's plain decode
reads a per-context divisor from a device word its graph follows (§10.5).

While it waits, `cpu_wait_kernel`'s extra CTAs warm the next layer's dense chain into L2 in
reading order: a one-column call warms the whole chain up to 40 MiB (tg512 plain 121.8 → 123.3
tok/s; 32 MiB +0.7 %, 48-56 MiB alike, 64-80 MiB less; RTX 5090, 2026-10-12), a wider call
keeps the smaller set of both mixer projections and 16 MiB of the input projection (under MTP
the wait is shorter and 48-64 MiB measured 0.4-1.4 % slower). This warming covers the window
the never-built lookahead prefetch targeted (§8.10).

### 8.7 Batch and verify windows

A round computes `batch` sequences of `width` consecutive columns in one Forward call. One decode
graph per batch size, one verification graph per (batch, width) — the first round of a shape runs
eagerly (the Ops' one-time setup), the second is captured, later rounds replay. The MTP drafter,
its catch-up and the DFlash2 proposal graphs are captured the same way (§11).

The dispatch unions the routed entries, so each distinct expert is fetched and computed once
for all its columns; the column limits of every route are MoE §6.2's. A DFlash2 batch takes one
shared draft length per round, chosen from the measured round times of its size (§11).

### 8.8 Round boundary

`decode()` stages the round's io with one `upload_pinned`, wakes the CPU service, runs
`before_round` (landing tables, §9), and replays the graph — a replayed round's n-gram rows are
read after the launch, behind the gate before `ple_embed` (§12). The graph ends with sampling; the
engine thread synchronizes, reads the sampled tokens, and checks the error word.

`commit()` runs the drafter catch-up, then `settle_round()`: the round's route log feeds `after_round`
— the cache accounting, promotions and landing adoption of §9. Host time between rounds was
re-measured without a profiler at 82-89 µs (settle 42-49, the decode prologue 26.6, the engine ~13),
about 1.1 % of a plain round at 132 tok/s (RTX 5090, 2026-10-10). An earlier 0.45 ms (~4 %,
2026-10-05) predates the LFRU victim-search fix (a budgeted step at 9,000 residents went from 22.8
to 10.9 µs with identical decisions) and the later launch-boundary fusions.

Pipelined decode — launching round r + 1 before round r is committed — was built in two parts and
measured (2026-10-05/06, RTX 5090): chaining the drafter behind the verification gained nothing, and
committing a round ahead into a spare state slot gained 1 % on the synthetic tg512 while losing
0.3 % on real code decode, for ~420 lines of state-slot indirection. Not kept; at the re-measured
host gap it is worth at most ~1.1 % and is not pursued.

### 8.9 Current state and candidates

Six concurrent 256-token requests through the Engine in process (DFlash2 auto) decode at a median
505.9 tok/s (range 499.2-506.5, RTX PRO 6000, 2026-10-22); a 6 x 1 round costs 11.3 ms, capping
plain 6 x 1 decode at 531 tok/s, and a 6 x 2 round 1.46 times that (more columns touch more distinct
experts). SGLang's 582 tok/s on the same card is a server-throughput measurement of DFlash on its ring
layout (its concurrency is not recorded; SGLang's one-request TPOTs are §3.3's); passing it needs
cheaper rounds as well as speculation.

The remaining bandwidth pools, by the 2026-10-31 atlas (§8.3):

- The dense FP8 projections. At T = 6 the 2560 x 6144 projection runs at 61 % of DRAM, latency- and
  issue-bound at 0.68 waves; every candidate of the current bits family measures 16.4 µs against an
  8.8 µs floor, and the 14.3 µs `sliced_t16w8s2` tile belongs to another bits family (§17).
- The hyper-connection pair, memory-latency-bound at about 17 % occupancy.
- Fusions: the GDN middle (five kernels between the two projections), the QSA small kernels, and
  the MoE tail into the next layer's mix.
- The routing chain is down to the router (about 6 µs own time per layer) and the in-place route
  plan (about 6 µs); the top-10 selection itself is under 1 µs. Removing the last boundary means
  routing in the router's last CTA.

### 8.10 Rejected

| Item | Why it lost (hardware, date) |
|---|---|
| Lookahead prefetch of next-layer experts by the router | Never built; the host cache policy (§9) and the CPU wait's L2 warming cover the same window |
| The 2026-10-03 kernel plan (K1-K7b, byte budgets at 1.5 TB/s) | Superseded by the built chain; `t1-kernels` measures the real round (2026-10-22) |
| Pipelined decode (chained drafting; a round committed ahead into a spare state slot) | tg512 MTP +1 %, code decode −0.3 %, ~420 lines (RTX 5090, 2026-10-05/06) |
| Weight self-prefetch at kernel start | Bulk prefetches compete with the kernel's own ordered stream: NVFP4 A16 T = 1 34.1 → 43.8 µs (+29 %, RTX 5090, 2026-10-12) |
| HC weights prefetched by the down partials | −0.3 to −0.7 % end to end; K1b already fetches during K1a (RTX 5090, 2026-10-17) |
| Splitting decode/verify graphs after 4 or 8 layers | IQ3_S MTP −0.5 to −2.2 %, Dense8 −1.2 % (RTX 5090, 2026-10-17) |
| Promotion schedule changes (interval 8 or doubling) | MTP +1.9 to +6.8 % but plain −0.5 to −3.3 %: fewer promotions help only text the cache already fits (RTX 5090, 2026-10-17) |
| 96-column gate/up tiles for GGUF experts | −10 to −21 % at 4,096 tokens, +13 % at 16,384, no end-to-end gain (RTX 5090, 2026-10-17) |
| Device-side PLE | No GPU wait to remove: 0 of 105 gated rounds waited on the reads (§12.5) |
| One ticketed launch per expert phase | Faster in the op bench (41.5 → 36.6 µs), end to end −0.5 %: the PDL chain already overlaps (RTX 5090, 2026-10-10) |
| Warp-reduction top-k in the route kernel | Noise; the kernel's cost is its load and launch (2026-10-10) |
| Gated RMSNorm inside the recurrent update (DSMEM cluster) | −1.1 % plain on both models; the update is DRAM-bound (2026-10-12) |
| NInfer's tuned 2560-wide linear routes | About +1 % plain engine speed for a change of output not shown harmless; the 96 x 2560 form alone diverged deterministically (RTX 5090, 2026-10-07) |
| Fused per-head RMSNorm + RoPE in QSA attention | Not bit-identical and within noise end to end (2026-10-07) |
| Shared expert on a branch forked at the mixer output | Competed with the routing chain: tg512 MTP −2.0 %; the kept form forks at the MoE input (IQ3_S +8 % plain, +3 % MTP, 2026-10-12) |
| Two rows of W_down per warp in `hc_down_partials_v4`; `__launch_bounds__(.., 2)` on the v4 pair; a 2-row-group down CTA | Each slower or a wash (RTX PRO 6000, 2026-10-22/31) |
| A register ring over FP8 SIMT weight phases | 16.4 µs at T = 6 became 18.4-22.5 µs at depth 2-8 (RTX PRO 6000, 2026-10-31) |
| GDN-middle megakernel with per-head dependency counters | Bit-exact and deadlock-safe on oversubscribed grids, but 5,165 µs against the 5,124 µs composed chain; the ordering protocol is reusable (2026-11-01) |

## 9. Expert residency: tiers, cache, placement and memory budget

The machinery of this section — the tier model, frame pools and device tables, the LFRU,
landing and safe frame reuse, frame leases, the RAM and SSD tiers, placement and the two
memory laws — is shared by every offloaded MoE family and is designed once in
[MoE offload design](moe-offload-design.md) §4 ("MoE §4" below). This section keeps what is
specific to Qwen3.8-Flash-Next: its pool structure, its constants and its measurements. The
subject is the 24,576 routed keys of §2 (48 layers × 512, top-10 per token). Code:
`src/models/offload/` (cache, residency, tiers, placement), driven by the Program in
`src/models/qwen4_exp/program/`.

### 9.1 Frame pools and device tables

This family's pools: an NVFP4 or block-FP8 artifact has one record size and therefore one
pool; a GGUF import's layers have up to seven record sizes, hence up to seven pools, each
with its own LFRU. The machinery — average frames, the VMM arena whose base never moves, the
per-layer I32 frame tables, the boundary ordering that keeps eviction updates ahead of
promotion copies — is MoE §4.2.

### 9.2 Replacement and admission: LFRU

The policy — the score law with global never-reset counts, the halving, the replay evidence
that decided it against nine other policies, the admission budget, pins, the warm start and
the prompt credit — is MoE §4.3. This family's constants:

- Halving period 8,192 ticks (`kLfruHalvingPeriod`, ~170 plain rounds; a pool's period is
  scaled by its share of layers). The mixed-trace halving evidence and the never-halving
  comparison are MoE §4.3's.
- The saved state caps counts at 16, so a stale expert yields to new uses.
- The seed is the shipped calibrated order (`calibrated_expert_order`, every artifact of
  this architecture's shape, 48 × 512; the program checks the shape) and fills half the
  frames (`kSeedShare`), leaving the rest free for the fill phase's landings. A static
  profile alone loses badly on a differing workload (+82 % misses on trace B), which is why
  the ranking is saved per session.

### 9.3 Round boundaries: promotions, landing, frame reuse

The boundary machinery — publish-after-copy with the ABA guard, landing frames and their
adoption, frame retirement, the failure path through the error word — is MoE §4.4. As this
family's Program composes it: the Program is synchronous (`after_round` runs on a quiescent
compute stream, `on_quiescent` frees every retired frame and drains the deferred-load queue,
`rounds_in_flight = 1`), so slack frames are 0. A failed round's requests fail through
`runtime::RecoverableExecutionError` and the prefix cache is cleared (§15).

### 9.4 Frame leases

The lease mechanism is MoE §4.5. Users today: the prefill expert stream's ring and staging
areas (§13) and the vision encode window (§16). Under a resident placement (§9.6) `lend()`
refuses and the wide prefill workspace falls back to a temporary allocation.

KV pages are not frames. The KV pool is a separate VMM arena (§14); when it must grow, the
sizing law's current allowance (`frame_target`, §9.7) is spent first, and only the rest is
taken from the pools by unmapping top chunks at a round boundary.

### 9.5 The RAM tier

The RAM tier's machinery — the decayed-LFU ranking, the slot lists and pins, demotions, the
unpinned records past the Windows page-lock limit — is MoE §4.6 and §4.11; its read agent,
tickets, read-aheads and the stream-slot lending for layer walks are MoE §4.7.

Measured on this family (RTX 5090, x8, Dense8 artifact, int8 KV, cold, 2026-10-05):
expert RAM at full / 32 GiB / 8 GiB: tg512 109.6 / 86.3 / 46.3 tok/s; code decode 79.6 / 39.1
/ 18.6, MTP 128.1 / 54.1 / 24.4; long-prompt prefill 977 / 542 / 407 tok/s; mean SSD read
0.6-0.8 ms in decode. Greedy ids identical in every configuration (6/6 checks). Walk-span
stream slots: with 640 about a quarter of an FP8 artifact's SSD-only experts were left to
demand reads in a 32K prefill; 1,280 streams them all (32K prefill 2,850 → 3,945 tok/s, and
the decode after it no slower, 2026-10-16). A host whose heartbeat word stops advancing for
10 s ends a wait (`kHeartbeatTimeoutNs`; raised from 1 s on 2026-10-16: under RAM pressure
1 s failed healthy FP8 requests in 8 of 34 SSD-tier runs at ~2 GiB available) and writes
`kErrorHostSilent` / `kErrorFetchSilent`; no trap.

### 9.6 Three-tier placement: mechanism and client policy

The placement forms, the runtime switch and its validation, the grow/shrink walks, the GET
report and the export are MoE §4.8; the mechanism/policy split itself is
[AGENTS.md](../../AGENTS.md)'s ("Mechanism and policy"). This family's grow walk refills 512
keys per maintenance tick, and its boot warm start seeds VRAM only from RAM (the pre-filled
copies are released and refilled with the next keys of the order). The client owns the tier
shares; client policy lives in `~/src/super-quant/scripts/infernix_placement.py` (`make
infernix-gaming`, `make infernix-full`); within the client's counts the per-round LFRU swaps
hot experts in and cold ones out. The runtime shrink/regrow law of the original design was
removed on 2026-10-30 for contradicting this split (it had been unreachable anyway, since
every default boot on the 96 GB card holds all experts).

This family resolves the startup forms in `models/qwen4_exp/instance.cpp`:

| Form | Effect |
|---|---|
| default (no flags) | Probe: the device plan with the KV pool fully mapped must hold every expert (plus a 512-frame margin). On the RTX PRO 6000 (96 GB) it always fits: every expert resident, nothing lent or evicted, RAM tier at its floor (1,536 slots of lists and fill staging, released after the load). A smaller card falls back to the maximal RAM tier. |
| `--experts vram=60%,ram=25%[,ssd=15%]` or counts | Shares or counts over all 24,576 keys, ranked by the saved state, else the calibrated order, else spread across the layers. Adaptive within the counts; the vram count is the frame ceiling. `vram=all`/`100%` is the resident placement and fails the start if it does not fit. |
| `--expert-placement FILE` | CSV of `layer,expert,tier` rows; `vram` rows are pinned exactly (never eviction victims), the pool is sized to hold them, `ram` rows fill the tier, the rest is SSD. |
| `--expert-ram-mib` / `--expert-pin-mib` | Byte-level control of the RAM tier (§9.8); cannot be combined with `--experts`/`--expert-placement`. |

`--expert-swap off` freezes the pools' content at what the seeds placed (misses still read
and compute).

### 9.7 Device memory budget

The sizing law, the spill guard and the monitor's role are MoE §4.9; one sizing function
(`size_expert_frames`, `models/qwen4_exp/memory_plan.h`) implements it at startup. This
family's inputs and measurements:

- `fixed` is `Program::plan_device`'s (the single authority for constructor sizes);
  `headroom` is `--vram-headroom-mib` (auto: 256 MiB headless, 512 MiB with a display and an
  elastic pool, 1 GiB with a display and a fixed pool); `reserve` is
  `max(256 MiB, 128 MiB + 4 MiB × min(graphs, 32))` (measured 2.0-2.6 MiB per executable,
  26 MiB at 13 graphs C=1 and 110 MiB at 43 graphs C=8, RTX 5090, 2026-10-04).
- The check runs before the 64 GiB of weights are read: a negative budget fails with the
  ledger and its largest contributors; fewer than 2,048 frames warns that decode will be
  slow. One `VRAM ledger` debug line and a one-line summary follow startup, naming the KV
  pool's growth if it is elastic.
- VMM reads equal `cudaMalloc` reads (streaming −0.01 %, gather +0.03 %, RTX 5090); resizes
  run on the engine thread at a boundary (unmap from another thread p99 7.6 ms slows the
  launch loop 9 %).

### 9.8 Host RAM budget

The ledger's law — one reserve, every pin re-checked against memory available at the moment
it is made, the experts capped by full mode and split into pinned and pageable bytes at the
pin room, the tier floor refused with a reason, an explicit `--expert-ram-mib` above the
ledger an error — is MoE §4.10. This family's terms (`plan_host_memory`,
`models/qwen4_exp/memory_plan.cpp`; reserve `--ram-headroom-mib`, default 2 GiB):

| Term | Default |
|---|---|
| Reserve for the OS and other programs | 2 GiB (`kDefaultRamHeadroomBytes`) |
| Expert banks (all layers) | from the plan |
| Other pinned weights (token embedding) | from the plan |
| Program pinned and mapped buffers | 256 MiB allowance |
| Prefix-cache Host tier (§15) | `--host-context-mib` |
| N-gram row cache (§12, pageable) | from the table |
| Load staging (transient) | 264 MiB |
| Process pageable growth | 1 GiB (also the hard minimum working set) |
| Lock overhead on every pinned byte | 0.25 % |
| Full-mode margin | 1 GiB |

Startup prints the ledger's summary and (full detail at debug level) every term.

### 9.9 The CPU-link split and its startup measurement

The divisor rule and the split's shape are MoE §5.6 and §5.7; the startup probes that feed
them (link copies with the SMs busy, CPU rates with the link busy, the busy-host retry) are
MoE §5.5, and this family's reference rates are §10.4's. The measurements behind this
family's defaults (nsys, RTX 5090, int8 KV, 8K context, 2026-10-06):

- A staged miss costs ~105-120 µs (one 2.76 MB record at the x8 link rate).
- Staged misses per layer call, plain decode: 0 in 47 % of calls, 1 in 30 %, 2 in 16 %, 3-4
  in 6 %. The CPU leg ends 30-50 µs after the PCIe leg in every bucket; the CPU computes
  about two thirds of the misses.
- Verification calls stage up to 30 records (3.5 ms); the first 4-6 rounds after a prefill
  are cold (plain 30-50 ms against ~19 ms steady-state, MTP up to 150 ms against ~20), partly
  because the prefill stream's lent frames come back empty.
- The CPU job cap (`ProgramOptions::cpu_expert_jobs`) defaults to 16: cap 16 gains in calls
  with many misses (serve C=4 plain fill +7.2 %, cold code MTP +2.8 %) and is neutral
  elsewhere; 32 gains nothing more.

### Rejected

- KV pages backed by frames and KV loans ahead of need: superseded by the separate elastic KV
  pool; engines that reserve KV without lending lost 19 % at C=8 (ninfer-ext).
- A separate MTP expert pool (128 slots, own LFRU): superseded by the drafting changes (§11).
- A 1 s heartbeat bound on device waits: failed healthy requests under RAM pressure (8 of 34
  SSD-tier runs); 10 s since 2026-10-16.
- CPU job cap 32: no further gain, tg128k −0.5 %.
- The rejections that decided the shared machinery — LRU and eight other policies, per-layer
  quotas, next-use prediction, host-shadow parameter dueling, the packed residency entry,
  slack frames, an OS mapped-file tier, inclusive residency, a fixed stream-slot pool,
  byte-exact pools, two-phase shrink, a frame floor for the display, the runtime
  shrink/regrow law — are MoE §4's.

## 10. CPU expert engine

The engine's design — role and limits, the ISA variants, the worker team's scheduling, the
miss-service protocol, the startup rate probes, the decode divisor and the prefill split —
is shared by every offloaded MoE family and is designed once in
[MoE offload design](moe-offload-design.md) §5 ("MoE §5" below). This section keeps this
family's kernels, constants and measurements. Code: `src/ops/offloaded_sparse_moe/cpu/`.

### 10.1 Role

The role and the in-place-versus-move trade are MoE §5.1; the placement-invariant route and
the exact-sum law that make a CPU output equal the GPU's bit for bit are MoE §2 (the
arithmetic itself is §17's). One team serves this family's three record kinds: NVFP4 W4A4
(`w4a4_expert.h`, the `nvfp4_expert_rg16_v1` records of §6.3 with their stored input
scales), NVFP4 W4A16 (`w4a16_expert.h`, §17), and the GGUF blocks of its GGUF imports
(`ggml_expert.h`: IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS gate/up, IQ4_NL and Q2_0 down; exact int128
sums rounded once).

### 10.2 Kernels

The four ISA variants and their CPUID selection, the column-count template, the prefetch law
and the compile flags are MoE §5.2, and the engine's measured capacity is MoE §5.2's table.
This family's decode paths:

AVX-512 VNNI: one 32-byte quad is broadcast into 512 bits (`vbroadcasti64x4`; lane r holds
row r's four codes), one masked `vpsrlw` and one `vpandd` split the nibbles, and `vpshufb`
maps each code to 2·e2m1 + 12 (unsigned LUT `kBiasedCodes`). `vpdpbusd` multiplies against
the activation's quad word; the product subtracts 12 · Σa per block, scales by `vpmulld`
(the activation `scale_scaled`) and two `vpmuldq` into int64 (the E4M3 weight scales). AVX2
and AVX-VNNI process two 8-row halves; AVX2 uses `vpmaddubsw` (a pair sums to at most 576,
so it cannot saturate) then `vpmaddwd`, AVX-VNNI `vdpbusd`. The ggml route uses 16-bit
multiply-adds (`vpmaddwd`, `vpdpwssd`) over two 16-bit planes (§17). Software prefetch runs
`prefetch_bytes` ahead (three 64-byte lines; default 2,048, `kDefaultPrefetchBytes`;
`host_probe` sweeps 0-8,192; the tuned distance is §18.1's calibration).

### 10.3 Worker team and miss service

The team's scheduling (worker 0 computing at once, epoch spin then futex park, work handed
out by atomic counters through the phases) and the service's protocol (the mapped request,
the sequence and heartbeat discipline, tiered jobs demanded before and computed after the
others, the idle wake) are MoE §5.3 and §5.4. This Program starts the service with `cpus`
empty, so production workers are not pinned; the worker counts are §10.5's. A stopped
service fails every later call (`kErrorHostSilent`), which the caller treats as fatal.

### 10.4 Startup rates

The probes and their method are MoE §5.5; a busy host halves the rate (~10,500 against
~19,500 experts/s at 6 workers, i9-13900K), which moved this family's decode divisor from 3
to 2 and cost a fifth of decode speed for a session — hence the retry. The measured link
rescales the costs fitted at the reference rate: the prefix restore rate and walk-span cost
(§15), the decode promotion interval (4 tokens at the reference; §9), and the split's width
limit (§10.6). Until records exist in RAM (the SSD tier before its fill) the reference rates
stand: 21,000 experts/s and 66,000 columns/s (i9-13900K, 6 workers);
`ProgramOptions::link_bytes_per_second` fixes the link rate in tests (reference 27.5 GB/s,
PCIe 5.0 x8).

### 10.5 The decode miss divisor

The rule `d = floor(1.65 + C·t)`, its step at `C·t = 1.35` and the fixed override are
MoE §5.6, with the step evidence. This family's settings:

- A GGUF import's one-column calls (plain decode, no speculation) take their divisor from
  the context instead: d = 1 below 1,536 tokens, d = 2 from there, written each round into
  a device io word the captured graphs read. Measured 2026-10-16 (Q2 IQ3_S artifact, same
  host, tok/s, d = 1 / d = 2): 512 + 256: 151.5 / 139-141; 4K + 512: 99.5 / 114.4. With
  MTP or n-gram drafts every call keeps the derived d: d = 1 in MTP's one-column calls
  measured 216.2 → 212.9 tok/s (2026-10-13, IQ3_S).
- Workers: 6 for NVFP4 banks, 16 for GGUF banks (compute-bound where the NVFP4 route is
  DRAM-bound; tg512, i9-13900K host, 2026-10-11: no CPU service 100.5 tok/s, 6 workers
  98.4, 16 workers 103.7, 20 workers 103.6). More than 6 NVFP4 workers lose end to end: CPU
  and PCIe share DRAM, and spinning threads only add contention (tg512, 2026-10: 6 / 12 /
  16 / 24 workers 62.25 / 62.01 / 61.51 / 58.89 ± 1.61 tok/s).
- Job cap 16 per call (C = 4 fill +7 %, cold MTP +3 %, 32 no faster).

### 10.6 The prefill CPU split

The split's shape — copies gated on each layer's routing, the thinnest non-resident experts
left to the CPU, the plan at a share of the measured rates, the link-scaled width cap, the
assist caps for wider calls — is MoE §5.7. This family's constants and measurements
(i9-13900K host, Dense8, INT8 KV):

- `kSplitRateShare` = 0.3 of the measured rates: the full fitted rates lose 4.5-6.2 %
  against it (6 workers), the probe's rates alone (16,600 / 47,300) gave only −3.3 / −0.3 /
  −0.1 %, 3 workers at the measured rates 11.1-18.9 % (10 repetitions per arm). Split off
  costs +15 / +7 / +2 % at pp256 / 512 / 1024.
- `cpu_split_columns` = 1,536 columns at the reference link, scaled inversely to the
  measured one (768 at x16, modelled, not measured); extending it to 3,072 / 4,096 cost
  pp2048 +2.4 % (2026-10-10).
- End to end (2026-10-06, Gold port, x8, warm expert cache, greedy ids identical off and
  on): pp512 2.463 → 1.567 s (1.57×), pp1024 2.272 → 1.752 s (1.30×), pp2048 and pp4096
  unchanged (the blind stream); cold first call, 512 tokens: 147 → 179 tok/s.
- Calls of at least `wide_from` columns (the decode width + 1, at most `kAssistMaxColumns` =
  255) take `cpu_assist_jobs` = 256 as their cap; wider chunks stream or stage on the GPU
  (§13).

### Rejected

- n-gram row prefetch into a host cache during a walk span: a 64K span's ~1.15 M rows would
  evict themselves from the 2^20-row cache; ~190 MB more RAM for at most ~3 % (2026-10-06).
- The rejections that decided the shared engine — static worker ownership, an AVX-VNNI-INT8
  variant, fractional and rounded divisor formulas, a divisor following the hit rate, a
  cost-based split, full or fitted rates for the split, CPU L2 warming of predicted misses —
  are MoE §5.8's.

## 11. Speculative decoding

The model speculates with one of two neural drafters — the MTP block bundled in the main artifact
(§6) or a DFlash2 companion artifact — plus n-gram copy proposals beside either. The target model
verifies every draft and stays the output authority: a drafter's precision and its draft lengths
change acceptance and speed, never the verified output (greedy output equals plain decoding on a
one-row round; the multi-row fast paths below are the exceptions, and the exact route disables
them). Chain verification only: `--draft-tree-nodes` is refused
(`src/models/qwen4_exp/instance.cpp`), so a round verifies one linear sequence per row.

Code: the round, the drafters and the policies are in
[src/models/qwen4_exp/program/program_impl.h](../../src/models/qwen4_exp/program/program_impl.h),
[draft_length.h](../../src/models/qwen4_exp/program/draft_length.h) and
[dflash2_program.cpp](../../src/models/qwen4_exp/program/dflash2_program.cpp); the DFlash2 Op
sequence is [execution/dflash2.cpp](../../src/models/qwen4_exp/execution/dflash2.cpp). DFlash2
mathematics, proposals and backend state are owned by [dflash.md](dflash.md); this section owns the
round, the policies and the measurements that decided them.

### 11.1 The round and its surface

A decode round of B rows drafts, then verifies W = 1 + the longest row's drafts columns. A row may
draft at most one token fewer than it may still emit, so every verified position stays in its
reservation. A longer n-gram copy proposal replaces a row's neural drafts (§11.7). Verification
runs one CUDA graph per (batch, width); MTP's draft chain one graph per batch size (plus a gated
variant per step count); DFlash2's proposal one graph per batch size.

Drafting runs on the compute stream at the round's start. The host stages the verification round's
n-gram rows while the drafter runs (`while_drafting`), the drafts and their confidences come back in
one copy, and the round synchronizes once — not per draft step. After the commit, a catch-up call
writes the drafter's context for the committed columns (a graph per batch size and width for
DFlash2; a K/V-only MTP call whose W = 1 form keeps the drafter current through plain rounds).

Neural rounds draft at most K = 7 (W ≤ 8, the widest column-invariant dense verification, §17);
n-gram copy proposals may draft up to 15. The option surface (flags in §18):

| Option | Meaning |
|---|---|
| `--spec mtp` | Draft with the artifact's MTP block. |
| `--spec dflash2 --draft-model FILE` | Draft with a DFlash2 companion artifact (§11.3). |
| `--draft-tokens N` (1..7) | Fixed length: draft and verify N every round. |
| `--draft-tokens auto` | The per-round policy of §11.4 chooses the length. |
| `--spec-min-p P` | A one-row round verifies drafts only up to the first below P. |
| `--lm-head-draft` | Draft heads read the proposal head instead of the text head. |
| `--ngram-draft-tokens N`, `--ngram-min-match M` | Copy proposals from the request's own history. |
| fast path `adaptive-speculation` | Multi-row rounds may draft (§11.5). |
| fast path `sampled-mtp-drafts` | Sampled rows draw their MTP drafts (§11.6). |

### 11.2 The MTP drafter

One QSA block of the text model's mathematics, stored in the main artifact: the input fusion of the
main model's final multi-stream residual with the draft token, one QSA block with a 512-expert
top-10 MoE and the shared expert, its own final mixer, and the head. Its dense projections are
`q8_g32_fp16` (router and shared-expert gate BF16, as NVIDIA stores them); its 512 routed experts
are `q4_g64_fp16` with MSE-chosen group scales — 1.34 GB, always resident in VRAM
([official_recipes.py](../../tools/convert/official_recipes.py), `_flash_next_drafter`). The IQ3_S
import instead stores them `q2_g64_fp16` (0.71 GB): MTP 4 + n-gram 7 decoded 201.4-203.1 against
196.1-196.7 tok/s with the Q4 experts, and the 0.63 GB difference went to expert-cache frames
(2026-10-16, RTX 5090). Routing is exact over all 512 experts; the old design's residency-bounded
draft routing and MTP expert pool were never built. The resident experts stream through
`resident_moe_experts` (shared-memory-swizzled row chunks); one draft step's expert call takes
~29 µs (Q4, T = 1, 10 experts, L2 cold, RTX 5090). A draft step costs ~0.24 ms end to end (measured
from the gated-chain A/B below, 2026-10-18, RTX 5090).

The drafter's K/V is an extra KV layer of the same paged groups, written by prefill calls and
forced tokens for every cell whose next token is known; the last committed residual is the lane's
pending cell, and each round runs K chained full steps from it. The draft head is the text head, or
with `--lm-head-draft` the converter's proposal head (131,072 most-frequent rows, `q4_g64_fp16`,
fused `linear_topk`); the head also writes each draft's log-probability (`ops::top_logprobs`).

**The gated draft chain.** A one-row argmax round with the `--spec-min-p` cut drafted all its
steps, read them back, and only then dropped the drafts past the first below the cut. The draft
graph is now a chain of IF nodes (`DecodeGraphDefinition::capture_gated_chain`,
`launch_graph_gate`): after step j a one-thread kernel applies the host's comparison (the float
log-probability widened to binary64 against `log(min_p)`) and sets the conditional that runs step
j + 1. Measured (RTX 5090, 2026-10-18): 34 % of mixed-topic MTP 4 draft steps lay past the cut, and
skipping them cut decode time 2.2 % (68.46 against 69.97 s, 32 segments); on the bench 6-21 % of
steps lay past it, IQ3_S tg512 +3.5 % pooled. Sampled rounds and multi-row rounds keep the ungated
graph. Core test `infernix_decode_graph_gated_test`: 70 gated-chain replays against the host's cut,
NaN and the float rounding of log 0.5 included, every one equal.

### 11.3 The DFlash2 drafter

A companion artifact holding only the drafter, loaded beside any Flash-Next artifact:
`tools.convert --components dflash2 --recipe qwen3_8_flash_next_dflash2` writes it from
[incoai/Qwen3.8-Flash-Next-DFlash2](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2). Its
projections are `q8_g32_fp16` (the dynamic-convolution kernel projections and the selector's hidden
projection stay BF16 with the norms, base kernels and codebooks); the context K/V roles share the
query block's K/V words. It embeds with the main artifact's token embedding and reads the text head
or, with `--lm-head-draft`, the proposal head.

The drafter has the 27B DFlash2 mathematics at Flash-Next geometry: hidden width 2560, 5 local
attention layers (sliding window 2048), intermediate width 7680, dynamic convolutions with two taps
and group width 16 around both attention and MLP, a top-16 conditional candidate selector of rank
256, conditioned on the target blocks [5, 15, 25, 35, 45]; RoPE theta 1e4; trained with a block of
8. The target feature of block l is the mean of the four hyper-connection streams of its completed
residual; the five taps give 12,800 values per position. Each lane keeps five context rings (BF16
K, FP16 V, 2048 slots — 40 MiB); prefill calls, forced tokens and layer walks write their last
window eagerly, a round's commit writes the committed columns. The rings ride prefix-cache
snapshots as the state image's drafter group; a stale ring costs acceptance, never output.

A round proposes once: the whole block of K + 1 columns in one masked forward, one graph per batch
size, whatever the round will verify — so a multi-row proposal costs no more than a one-row one.
The selector's conditional walk draws the drafts (a sampled row with its seed; a greedy row takes
the argmax path) and also writes each draft's temperature-1 probability at its selected rank, read
back with the drafts. The block width is a runtime choice (K = `--draft-tokens`, 1..7, the widest
column-invariant verification).

The proposal head's top-16 over its 131,072 rows runs in two launches after its producer kernel: a
selection kernel takes each column's 4,096-key slices of producer keys — a thread sorts its sixteen
keys in registers (bitonic network) and warp and CTA butterflies keep the top sixteen of two sorted
lists — and one warp per column then merges the slices' lists. No completion counters, so no clear
kernel and no atomics, and the workspace carries only the producer keys and the slices' lists.
Keys are unique (distinct ids), so the selected keys and their order are those of a full sort: the
selection is bit-equal to a full stable sort whatever the merge tree, ties included.

The drafter's context feature projection, Q8 [2560, 12800] (five target taps of 2560), has a tuned
fast route under the `q8-sliced-k` flag: sliced-K MMA tiles (double-buffered weight stages, 8 K warps
to 32 columns and 4 beyond, one stage past 64 where the kernel loops token blocks) through 128
columns and the r32/r64 MMA tiles beyond — on the RTX PRO 6000, cold-op minima against the
runtime-shape routes the problem ran before: 24.3 against 38.7 µs at one column, 26.4 against 127 at
16, 48.9 against 129 at 48, 63.5 against 124.9 at 64, 110.4 against 132.9 at 128, and 569.3 against
720.9 at 2,048. The exact route keeps the runtime-shape routes bit for bit (the table's exact
selector replicates them); the fast selector is width-keyed like the flag's other problems, its
FP64-oracle conformance at every seam is in `test_q8_a16`, and the flag reaches the projection only
where it is selected (Dense8 today; recipe C selects no `q8-sliced-k`, so production's recipe C
artifact keeps the exact routes until a qualified row adds it).

### 11.4 Draft-length policy

`--draft-tokens auto` chooses per round. K = 0 (a plain round) is always a candidate, so the
policy turns speculation off where it does not pay.

**One-row MTP rounds** pick K before drafting: the K in [0, 7] maximizing expected tokens per round
time, 1 + Σ products of the lane's per-position acceptance estimates (an EWMA, weight 0.1, prior
0.75) over the cost 1 + `kWidthCost` · K plain rounds. The cost is learned online — consecutive
one-row rounds' start-to-start times fit against their draft steps (gap-filtered, exponentially
weighted) — until the fit is determined, with the prior `kWidthCost` = 0.38: warm tg512 on the
RTX 5090 measured W = 3/4/5 rounds at 1.84/2.13/2.54 plain rounds (re-measured 2026-10-10 at
1.61-1.79 / 1.89-2.11 / 2.42-2.53 between 8 and 20 % expert misses). Every 8 rounds a row probes
one draft more than its choice, so the next position's estimate stays current. After drafting, the
round verifies only its confident drafts: those before the first whose head probability is below
`--spec-min-p` (default 0.5 with `auto`, 0 with a fixed length). Measured (RTX 5090, 2026-10-07, C =
1, AIME prompts sampled, MTP 4 with the proposal head): no cut 139.5 tok/s and 76.1 % accepted;
cut 0.5 **145.6** tok/s and 82.1 %; cut 0.7 142.6 tok/s and 86.3 %. A confirmation alternated 138.8
and 147.2 tok/s (+6.1 %); greedy output with the cut is byte-identical to without.

**One-row DFlash2 rounds** choose after the proposal, from the drafts' selector probabilities: the
K maximizing (1 + Σ products of the drafts' calibrated acceptances) / (1 + 0.1 + 0.31 K) plain
rounds. A draft's acceptance is read from its probability's decile, learned per request (an EWMA
over verified rounds; a decile's prior is its probability's face value), because the selector's
probability over its 16 candidates is not calibrated on its own. The priors stand from measurement
(RTX 5090, 2026-10-20, Dense8 import, greedy code, INT8 KV): fixed lengths 2-7 took 14.2-27.7 ms
rounds against 8.4 ms plain ones — about 2.6 ms a draft column and 0.6-1 ms of proposal — and
verifying all seven ran 18-24 % slower than MTP `auto`; a fixed cut at 0.6 was 3-15 % faster, the
calibrated length 8-11 % faster on code, docs and sampled docs (26 % on sampled code). Reusing
MTP's online round-cost fit here chose shorter lengths and was 6 % slower on code, so the priors
are fixed. One draft more is probed every 8 rounds.

A fixed `--draft-tokens N` drafts and verifies N on one-row rounds and min(N, 2) on a pair round;
the cut then applies only where `--spec-min-p` asks for it.

### 11.5 Multi-row rounds

Without the `adaptive-speculation` fast path, only one-row rounds draft: at B ≥ 2 the round decodes
plain and the W = 1 catch-up keeps the drafter current. The reason is measured, not assumed
(2026-10-04, RTX 5090): rows drafted as if alone and the round padded to 1 + max K, so fillers ran
the whole forward and their experts became recurring uncredited misses; calls of ≥ 7 columns do
not fork staging and compute; the CPU job cap binds; and two cold MTP rows ran 24.9 + 21.7 tok/s
against 72 aggregate for plain C = 2.

The fast path (a request-coupling path under the
[model-quality policy](model-quality.md#6-numerical-acceptance-policy-and-fast-paths); off on the
exact route, which decodes plain at two or more rows):

- **MTP pairs** draft one shared length K ∈ [0, 2] for both rows
  (`choose_pair_draft_length`): the K maximizing both rows' expected tokens over
  1 + `kPairWidthCost` · K plain two-row rounds, `kPairWidthCost` = 0.47 measured (K = 1 and 2
  rounds took 1.44 and 1.95 plain two-row rounds; K = 3 reached 8 columns, past the fork and
  CPU-job limits, took 3.1 and lost). Deciding table (RTX 5090, 2026-10-07, AIME prompts sampled,
  INT8 KV): C = 1 with its own policy 135-139 tok/s (74 % accepted); C = 2 plain 143-144; K = 1
  **163** (64 %); K = 2 160 (59 %); K = 3 120; C = 4 plain 197 against 155 with one draft each —
  so rounds of three or more rows stay plain. Against the gate on the same workload the pair
  policy ran C = 2 at 164.9 against 143.8 tok/s (+14.7 %); on the IQ3_S import two-row MTP went
  160 → 200 tok/s (2026-10-18). At most 2 × 3 = 6 columns, inside the column-invariant dense
  routes; a pair round's output still differed from the rows decoding alone in 1-4 of 96 segments
  (2026-10-14) — the coupling the fast path accepts.
- **DFlash2 rounds of any size** draft one shared length (`choose_batch_dflash_length` over
  `shared_draft_length`): every drafting row contributes its acceptance EWMA, the cost is
  1 + fixed + per_draft · K plain rounds of that size. The costs are measured online per round
  size (`batch_cost_`): an EMA of the plain rounds and a regression of the drafting rounds against
  their extra columns, fitted separately because a single line hides the proposal's fixed overhead
  in the intercept — on the RTX PRO 6000 at six rows it priced a draft at 0.47 plain rounds when
  wide rounds cost about twice that, and the policy drafted where it lost (2026-10-21). The
  warm-up prior drafts at two rows (where drafting paid on every card measured) and stays plain
  from three, where wide rounds paid only with much cheaper columns — the offloaded setups the
  priors came from. Drafting must beat plain by `kBatchDraftMargin` = 5 %: the pre-margin fit
  drafted at a score 5 % above plain and measured 8 % below it (2026-10-21, RTX PRO 6000). The
  probe alternates one draft more than the choice with a plain round (both sides of the fit stay
  measured), and thins to every 8th cycle once a size is fitted and chooses plain — the probe
  otherwise gave away about 4 % of the size's tokens at six rows. The fits describe the placement
  they were measured under and reset on a placement switch or a tier refill: a plain round timed
  with half the experts on
  the SSD (24 ms at six rows) priced drafts at a twentieth of the 11 ms all-VRAM round after a
  switch back, and C = 6 fell 444 → 351 tok/s before the reset was added (2026-10-22, RTX PRO
  6000). Measured with every expert resident, INT8 KV, six rows: the aggregate went from about
  360 to 590 tok/s with 60-80 % of drafts accepted (2026-10-21, RTX PRO 6000). At commit
  `168b0b3` (2026-10-22) the six-lane `t2-bench` median was 505.9 tok/s, most tokens served from
  plain rounds — a 6 × 2 round costs 1.46 times a 6 × 1 (more columns touch more distinct
  experts), and plain 6 × 1 decode is capped at 6 / 11.3 ms = 531 tok/s.

Diagnostics: a per-request verify-round report (rounds, columns past their row's drafts, per width)
and a `DFlash2 batch policy` line with each size's fitted plain time, fixed overhead and column
cost every 64 drafting rounds.

### 11.6 Verification and acceptance

Acceptance runs on the device, from BF16-rounded logits exactly as plain decode samples (§17).
Greedy rows drafted deterministically (an MTP argmax round, every greedy DFlash2 row) take
`speculative_accept_greedy_drafts`: the accepted prefix of the target-argmax matches, the target
correction at the first mismatch, the bonus token when all accept. A round with a sampled row
verifies by sparse rejection sampling with the drafts' laws (`speculative_accept_sparse_drafts`:
16 candidates and their probabilities per position — the MTP draft head's top ranks, the DFlash2
selector's retained distribution, an n-gram copy one-hot at its copied tokens), preserving the
target's processed distribution. The sparse Op must serve the token domain; the plan refuses a
DFlash2 configuration it cannot serve. Token counts for the next round's penalties are counted
after the sparse path.

The `sampled-mtp-drafts` fast path draws a sampled row's MTP drafts from the draft head's top
tokens under the row's own sampler (greedy rows unchanged, and sampled rounds skip the confidence
cut). Speed, sampled decode: geomean +2.7 % on Dense8 INT8 (MTP 4, temperature 1.0, top-p 0.95,
top-k 20, 5 of 6 cells, 2026-10-17); IQ3_S math +18.8 %, story +22.4 %, code +4.1 % (2026-10-18);
FP8 math +20.4 %, story +15.6 %, code −13.0 %.

A one-row verification call of at most 16 columns runs its GDN layers' causal convolution, gates
and recurrence record in one kernel (`gated_delta_net_replay_record_conv`): each CTA evaluates its
heads' q, k and v channels from the replay records' convolution inputs and the live convolution
history into shared memory — the arithmetic of `causal_conv1d_silu_from_states_split` — and its
gates from the control projection — `gdn_gating`'s (`gdn_gating_direct`'s for a GGUF import's A) —
so q, k, v, g and beta are never materialized and two kernels leave each layer's chain. The output
and all three records are bit-equal to the composed kernels (the op test covers every width 2..16,
both gate forms, eager and captured). On this GPU the fused kernel measured 15-25 % faster than the
composed chain at one row (every width 2..16, cold and warm, op-level), but its redundant
convolution reads scale with the row count and it measured slower from two rows up — so only
one-row verifications take it; multi-row rounds, mixed calls and wider verifications (n-gram, wide
trees) keep the composed kernels.

Each request reports `SpeculativeStats`: rounds, drafted and accepted tokens, acceptance per
position, n-gram rounds, fallback steps. The same observations drive the policies: the lane's
per-position acceptance EWMA, the n-gram bucket EWMA, and DFlash2's decile calibration.

### 11.7 N-gram copy proposals

The proposer ([ngram_proposer.h](../../src/models/qwen_common/program/ngram_proposer.h)) indexes
the request's own history in a bounded, position-indexed corpus (sized from `--max-context`;
Flash-Next takes no cross-request archive). A proposal copies up to `--ngram-draft-tokens` tokens
(1..15) after a match of at least `--ngram-min-match` (4..64, default 12). A proposal replaces the
row's neural drafts when it is longer; a one-row round takes it only when its expected tokens per
round time — its match-length bucket's learned acceptance per position — beat the neural drafts'.
Bucket priors (matches of 4-7, 8-11, 12-15, 16+ tokens): 0.75, 0.88, 0.93, 0.96, each an EWMA.
Measured (RTX 5090, IQ3_S import, 4K + 512 tokens): MTP 4 with 7 n-gram drafts decoded 194.2
against 190.2 tok/s (2026-10-15). The verification rows a copy round reads are staged while the
drafter runs and, in steady state, read behind the device-side gate of §12.

### Rejected

- MTP expert pool with residency-bounded draft routing: all-resident drafter experts
  with exact routing are simpler and faster; never measured for the pool.
- Draft trees over MTP top-2: not built; the model verifies chains and refuses
  `--draft-tree-nodes`.
- Exact verify-row kernels for width invariance: cost 2-11 % of verify time; the
  ≤ 8-column SIMT routes are already column-invariant. Decided 2026-10-04.
- ≤ 8-column SIMT splits of wider blocks: one extra pass over the dense weights per round, above
  the 3 % bar. Decided 2026-10-04.
- Masked fillers and the Dinkelbach joint draft-length search: model only;
  the measured pair policy and the batched DFlash2 policy replaced them.
- Verify-union prefetch: never built.
- Splitting the decode and verification graphs after 4 or 8 layers: −0.5 to −2.2 % on MTP
  (2026-10-17, RTX 5090).
- Skipping draft steps that a copy proposal replaces: IQ3_S +0.2 %, Dense8 −0.4 % (2026-10-17).
- One cost line through plain and drafting rounds (batched DFlash2): priced a draft at 0.47
  plain rounds where wide rounds cost about twice that; separate fits adopted (2026-10-21).
- Choosing multi-row drafting on a bare score without the 5 % margin: scored +5 %, measured −8 %
  (2026-10-21, RTX PRO 6000).

## 12. PLE n-gram table

### 12.1 Table and volume

Each token's PLE input is 16 table rows: `ngram_size` 3 with `heads_per_ngram` 8, so heads 0-7 hash
the bigram and heads 8-15 the trigram ending at the position. The engine worker hashes the ids on
the host (`NgramHash::row_ids`: exact integer arithmetic over per-head primes and per-layer
multipliers; an EOS in the window replaces every earlier context token, and the sequence start is
EOS-padded). The rows are consumed once, by `ple_embed` at the top of layer 1 — a plain launch, so
stream order can hold it behind an earlier copy.

The FP8 table holds 320M rows of 160 B — 51.2 GB of rows, 52.4 GB as the padded volume below —
with one BF16 per-tensor scale in the artifact, and cannot sit beside the weights (§9). The
converter writes it as its own volume beside the artifact
(`<artifact>.ngram`, `write_ngram_volume`): a 4 KiB header, then rows packed into 4 KiB blocks and
never straddling a block (FP8: 25 rows per block, 12,800,062 blocks, 2.4% padding). The volume may
live on another NVMe drive than the artifact (`--ngram-volume`). Rows are read with unbuffered I/O
only, so the table never fills the page cache.

### 12.2 Row formats

Three row formats share the layout; the format follows from the row size:

| Format | Row | Rows per block | Volume | Scale |
|---|---:|---:|---:|---|
| FP8 E4M3FN | 160 B | 25 | 52.4 GB | one BF16 per-tensor scale (the artifact's `ple/ngram_scale`) |
| IQ4_NL | 90 B | 45 | 29.1 GB | none (llama.cpp blocks: fp16 `d`, 16 nibble bytes) |
| BF16 | 320 B | 12 | 109.2 GB | none |

- FP8: `ple_embed` computes `bf16(e4m3(row) · scale)`.
- IQ4_NL (the GGUF import's table, byte for byte): `ple_embed_iq4_nl` decodes each value as
  `bf16_rn(fl32(d) · kvalue)`; the product is exact in FP32 and rounds once at the BF16 boundary,
  like the FP8 rows. Nibble order is llama.cpp's. `ngram_scale` does not apply.
- BF16: the forward uses the rows as the embedding with no decode.

`resolve_ngram_volume` accepts any volume of the model's table geometry in place of the artifact's
own, whatever its format or id; the Program then runs that format (`ForwardBatch::ngram_format`).
An FP8 volume that is not the artifact's own needs its scale, which is a property of the volume: the
BF16 word at header byte 60 (volumes written since 2026-10-18 record it; 0 = none), or the scale
known for a volume published before (`kKnownFp8Volumes`: NVIDIA's table, `0x3951`). Without either,
the volume is refused. The Program uploads the scale (`ForwardBatch::ngram_scale`).

The FP8 checkpoint's artifact binds the BF16 table it was quantized from (§6): `--ngram-reuse`
checks sampled rows against the checkpoint's FP8 codes (`NgramTable.bf16_rows_match`);
`verify_artifact --ngram-rows all` checks every row. Recipe B with its FP8 volume and with a BF16
volume of the same table's dequantized values produced the same 400-token greedy text and the same
mean NLL to every printed digit (1.5817995854760287 over 10,325 tokens; 2026-10-12, RTX 5090).

The prefix cache's disk-store signature names the format, the volume id and the FP8 scale (§15):
stored states embed the rows, so a cache saved with one table is not restored with another.

### 12.3 Host read path

`NgramVolume` opens the volume with `FileMapping::None`: the file is never mapped. On Windows,
every unbuffered read of a file with a mapped data section pays the cache manager's coherency work;
with the 52 GB volume mapped, random 4 KiB reads dropped from ~220K/s to ~56K/s (Optane P5800X
probe, 2026-10-05). Unmapping cut a 128K prompt's row reads from 21.6 s to 5.9 s and its prefill
from 3,255 to 5,331 tok/s (RTX 5090, 2026-10-05).

Reads go through a direct-mapped host cache of 2^20 rows (~170 MB pageable, alive with the
Program), then to one unbuffered 4 KiB read per distinct block: a call's misses are sorted by
(block, index), so a block serves every row it holds, however often each appears. Windows issues
overlapped reads, Linux kernel AIO; other POSIX systems read serially.

A call with few distinct blocks reads through one rolling ring of 64 blocks (256 KiB, fixed). A call
with 512 or more distinct blocks — a prefill chunk — splits them over 8 threads of a 16-block ring
each. One thread at depth 64 reaches ~220K reads/s and eight threads at depth 16 ~880K (probe,
Optane P5800X, 2026-10-05). As built, a 128K prompt reads its 1.27M blocks in 1.29 s and prefills
at 6,576 tok/s (RTX 5090, 2026-10-05).

The volume counts rows requested, cache hits, block reads and read time. Each finished request
reports them on one Debug line ("n-gram rows: N requested, H% host-cache hits, R NVMe reads, T ms of
reads"), with the gated rounds' reads and the GPU's gate waits appended (§12.4).

### 12.4 Overlap with the GPU: the gate

Rows read after a round's launch must not delay it by a host sync. The Core gate
`upload_pinned_when(dst, pinned_src, bytes, pinned_ready, expected, stats, row, stream)` copies the
rows inside the graph, before `ple_embed`, once the pinned ready word equals the round's device
word; `publish_pinned_word` (sfence, then store) releases it. At most 16 CTAs of 256 threads copy
32 KiB each; each CTA's thread 0 polls with `__nanosleep(200)`, then `__threadfence_system` and
`__ldcv` loads, so no CTA depends on another and no stale cache line of an earlier round is read. A
gate that waits past 120 s (`kPinnedGateTimeoutNs`) stores `layer << 16 | kErrorRowGateTimeout` in
the Program's device error word and skips its copy instead of trapping (a trap until this change;
on sm_120 a deliberate trap hung the driver until a power cycle, and a trap names no site): the
round drains and fails its requests while the context, the other lanes and the Engine survive.
Round
words are strictly increasing and skip zero, so a stale word never matches; the io layout places the
gate word before the rows, so a round uploads it with the io prefix.

- **Verification rounds.** `decode()` fixes each row's draft source and width before drafting. While
  the drafter runs (after a `device_.flush()`), the engine queues and reads the anchor column (the
  committed tokens) and every column of an n-gram copy row; after the launch, while the GPU embeds
  and runs layer 0, it reads the remaining columns and publishes. The worker reads and releases
  them before it queues anything else (route copies, sampler, acceptance): until the word is
  published the GPU waits on that thread, and a call that waits for the device deadlocks the round.
  A kernel's first launch is one, as the driver loads its code then and, with the process past its
  OS budget, waits for the device to do so (port of upstream c2afefb2; queued ahead of the release
  there, a first sampler or acceptance launch stalled the round until the gate trapped after
  `kPinnedGateTimeoutNs`, 120 s, in 5 of 7 such starts). All rows queued for a round are
  read in one volume call (`read_queued_ngram`): read lane by lane, six lanes' misses summed to a
  median 489 µs of GPU wait per round, and one call brought that to 11 µs (2026-10-22, RTX PRO 6000);
  six in-process requests of 256 tokens went from 464.1 to 493.1 tok/s.
- **Plain decode rounds** are gated the same way, but only when the graph is replayed, the round is
  not mixed (§15), and some row misses the host cache (`decode_rows_cached` probes the hashed ids).
  Gating every replayed round lost 1.2% on warm tg512 plain (110.26 → 108.98 tok/s, sd 0.06); the
  probe keeps warm rounds staged before the launch.
- A round that runs eagerly (first use, graph capture, mixed) reads its rows before the launch.
- **Prefill** reads each chunk's rows before the launch and uploads them with the io. The layer walk
  (§13) reads the next span's chunks' rows into the walk slots while the current span computes
  (`walk_prefetch_rows`, validated by an FNV-1a key over the token window); a prefetched chunk
  skips its reads.

Measured (2026-10-05, RTX 5090; cold CLI = a new process, 6 + 6 runs ABBA; greedy ids identical in
every run): cold story MTP 115.50 → 119.73 tok/s (+3.67%), cold code MTP 131.88 → 132.37 (+0.37%, at
the noise edge), warm tg512 MTP +0.06% (noise); with plain rounds gated, cold story plain
95.37 → 97.47 tok/s (+2.20%), cold code plain +0.18%, warm tg512 plain +0.04%. About two thirds of
the read time hides behind the launch; the rest waits at the gate.

Errors: a scope guard (`GateRelease`) zero-fills the unwritten rows and publishes the word if
anything throws between launch and publish, so the stream drains and the error keeps its class (§8).
A gated round's reads above 100 ms emit a Warning (`kSlowGatedReadNs`) — the line that also shows
NVMe power-state exits.

### 12.5 RAM residency

`--ngram-residency auto|ram|volume`, default `auto`. `volume` is §12.3-12.4. `ram` reads the whole
volume into a page-locked buffer at Program start (1 GiB unbuffered pieces, then one registration)
and serves every row from it; the row cache is not allocated, and the gate stays but never waits on
I/O. `auto` asks the host-memory ledger (§9) for the table beside the pinned experts and takes RAM
only when every routed expert still fits pinned — so an artifact with all experts resident pins the
table, and a tiered placement reads it from the volume.

Measured (2026-10-17, RTX PRO 6000): a cold 16K prompt ran 3,010 / 3,014 tok/s from the volume and
2,988 / 2,978 with the table in RAM; MTP + n-gram decode 211.1 from the volume against 215.2 in RAM,
one run each, within the base arm's 210.4-215.5 — no row-read time is left for the GPU to take
over. With the table on its volume, no gated round made the GPU wait (0 of 105 rounds; 129 reads,
2.3 ms behind the gate), and the after-draft GPU idle is 20.7 µs (IQ3_S) / 36.9 µs (Dense8) per
round.

Pitfall: with about 79 GiB of RAM free, `auto` flipped between runs (read from the volume, or
27.8 GiB pinned), costing one benchmark arm 25%; comparative runs pin `--ngram-residency volume`.

### Rejected

- Device-driven PLE (hash and row reads on the GPU over UVA, as vLLM pins the Engram table):
  after-draft idle 20.7-36.9 µs per round and no gated wait — nothing to take over (2026-10-17).
- Device-assembled verification (no host round trip between drafting and verification): the same
  idle stayed under its 0.15 ms go threshold; never built (2026-10-17).
- A dedicated NVMe agent thread: the engine worker already spins where the reads run.
- Speculative top-k row prefetch: k × 16 extra reads on an IOPS-bound drive, nothing left to hide
  them behind.
- A larger or associative decode cache: decode misses are compulsory (an infinite cache gains
  0-2.5 pp; simulated 2026-10-04).
- Filler-row reuse: filler token ids and rows would disagree, changing routing.
- A read deadline that fails the round, or post-launch errors made `runtime_error`: a slow or
  recoverable read would end the worker.
- IoRing (Windows): ~5% over overlapped `ReadFile` at depth 64 (probe, 2026-10-05).
- Eight read threads at depth 64: 3.16 → 4.09 s of reads on a 128K prompt, slower than one thread;
  the built split is 8 × 16 (2026-10-05).

Tests: `infernix_read_only_file_test` (both mappings, ring offsets),
`infernix_qwen4_exp_ngram_volume_test` (closed-form row oracle: one read per distinct block, ring
wrap, cache hits, slot collisions, bounds, wrong size, `resolve_ngram_volume` refusals),
`infernix_device_test`'s pinned-gate cases (poison then delayed data, a pre-published word, a stale
word over replays, a misaligned size), `test_flash_next_ngram.py` (the hash specification),
`infernix_qwen4_exp_forward_real_test` (rows through the Forward on the real artifact).

## 13. Prefill

A prompt of n tokens runs as prefill calls of the configured chunk (default 16,384, `kPrefillChunk`; `--prefill-chunk`). Calls of at least 256 columns (`kStreamMinColumns`) stream their experts by DMA, and a run of consecutive streamed calls of one prompt runs as one **layer walk** span (§13.2). The exact route is chunk-invariant: calls split on the 16-token GDN grid and every route is column-invariant, so a position's bits do not depend on how the prompt was split (36 perplexity windows of 14,336 tokens at chunks 1,024, 4,096 and 12,288 gave identical bits; Dense8, INT8 KV, RTX 5090, 2026-10-15). The prefix cache relies on this grid (§15).

### 13.1 Calls, chunks and the staged io

- **The chunk.** Flash-Next defaults to 16,384; Qwen3.5 keeps 1,024 (27B measured flat to slower beyond it). Wider chunks prefill faster with no decode cost (RTX 5090, MTP 4, INT8 KV, 2026-10-15):

  | Chunk | pp16384 | pp32768 | pp65536 | tg512 |
  |---|---:|---:|---:|---:|
  | 1,024 | 5,207 | 5,567 | 5,701 | 143.7-144.0 |
  | 4,096 | 6,648 | 9,639 | 9,887 | 143.7-144.0 |
  | 16,384 | 6,768 | 11,211 | 11,268 | 143.7-144.0 |

  Expert frames 8,561 → 8,535 (26 fewer). The tap ladder keeps the 4,096-token spacing (`tap_ladder_chunk`) whatever the chunk.
- **Workspaces.** Calls up to 512 columns (`kStaticWorkColumns`) run in the static arena. A wider call swaps in a wide arena, lent in this order: the stream lease's part, a frame lease of its own, or (no lendable frames) a temporary `DeviceBuffer` with a warning (`WideWork`). A chunk whose wide workspace the cache could never lend (a 65,536-token chunk on the SSD tier wanted 27 GiB) is halved repeatedly at startup until it fits, with a warning; the output is the same (`fit_wide_work_to_cache` / `fit_wide_work_to_free`).
- **Staged io.** The host hashes the chunk's PLE rows and reads them from the volume (§12), stages ids, positions and rows in one pinned buffer, and uploads the whole io layout per call (`run()`: one `cudaMemcpyAsync`). Each call carries the chunk's MTP drafter cells (`MtpChunk`, written in sub-chunks of 512 columns); in a walk the last layer writes them (§13.2). Media prompts carry their RoPE positions in the same layout (§16).
- **Attribution.** `advance_prefill` and the walk open permanent NVTX ranges: `prefill.chunk` (payload: width), `prefill.ple_rows`, `device.wait`, `prefill.residency`, `text.layer` (payload: the layer) and `ple.rows` (every host n-gram row read). They are chunk-granular and cost nothing without a tool attached.

While another lane decodes, a wide call is split to `kPrefillChunkBesideDecode` (1,024) and can ride in the decode round's Forward: mixed rounds are §15.

### 13.2 The layer walk

A chunk-major prompt of 4,096-token calls re-streams nearly every non-resident expert for every call: ~384 records per layer × 2.76 MB × 48 layers ≈ 51 GB, ~1.8 s at 27.6 GB/s (x8 link), against ~1.0 s of compute. A span of up to `kWalkMaxTokens` = 65,536 tokens streams them once and is compute-bound.

- **Order.** Decoder layer l runs over every chunk of the span, then layer l+1 (`prefill_walk.cpp`). The span keeps the prompt's call grid: chunk c at layer l reads exactly what call c reads at layer l (its own residual columns, and the layer's state after chunks 0..c−1), so every output bit equals the chunk-major prefill's. Only expert placement differs, and the MoE's arithmetic is placement-invariant (§17).
- **MoE groups.** `layer_mixer` runs per chunk (everything before the MoE); `layer_moe` runs once over a group of consecutive chunks, up to `kWalkMoeColumns` = 32,768 columns (16,384 before 2026-10-19) and what the arena holds. Every MoE step acts column by column, so grouping changes no bit; the expert calls see 2-8× the columns (IQ3_S expert tiles run 33-50 % more effective TOPS at 16,384 columns than at 4,096).
- **One-pass calls.** A walk MoE call whose layer has every non-resident expert in the ring and every record in RAM (`ExpertStream::complete`) stages nothing and runs as one pass over its jobs (`staging_slots` 0) instead of double-buffered passes of half the slots. An expert a decode round evicted meanwhile is read in place from its host record — the same bytes.
- **Steps.** Each `advance_prefill` after the first enqueues ⌈48 / chunks⌉ layers of every chunk (about one call's work), synchronizes, and returns zero processed tokens as a service unit. Each step opens a fresh SSD-tier round (`tier_step`): a fetch request outside an open round fails at once (`require_tier_round`). The last step returns the span's tokens and runs the call-end work once.
- **Memory.** The span's residual stream (BF16 [S·H, tokens]; 20 KB per token, 1.3 GB at 64K), every chunk's staged io image and an MoE group's input live in the walk area of the stream lease, lent with the ring (§9). Without enough lendable frames the prompt runs chunk-major.
- **Taps.** A prefix tap due at an inner chunk end is captured inside the span (`walk_plan_taps`: each layer's state is copied once the walk has enqueued that layer of the chunk) and committed after the span (`walk_commit_taps`). Ending the span at each tap instead streamed the non-resident experts once more per tap: ~2.4 s, 14 % of a cold 128K prompt.
- **Speed** (2026-10-05, RTX 5090, Windows, INT8 KV; OLD = pre-walk dev, ON = walk + QSA prompt kernel + vectorized split; `fn/rigs/layer/speed.bat`):

  | Case | OLD | ON | ON/OLD |
  |---|---:|---:|---:|
  | long pp131072 | 1,828 | 3,373 | 1.85× |
  | short pp16384 | 2,207 | 5,680 | 2.57× |
  | c1024 pp16384 | 521 | 2,420 | 4.64× |
  | short pp4096 (one chunk) | 1,551 | 1,637 | 1.06× |
  | tg512 | 110.0 | 110.1 | 1.00× |

- **Two prompts at once.** A walk starts beside another prefilling lane, which waits for its span; the spans of two long prompts alternate (ending both chunk-major re-streamed every expert per call: 31 of 244 busy minutes of a 2026-10-08 serve log were in that state). Two fresh ~90K prompts sent together prefill in 26.1 s instead of 72.5-73.1 s, each at 6.9-7.0K tok/s instead of 2.3-2.6K (RTX 5090, ABBA; greedy text equal).

An abort inside a span leaves no endpoint: the layers are at different positions. A Vision window and prefill interleaving are §16 and §15.

### 13.3 Expert supply: the streamed ring

- **The ring.** A chunk of at least 256 columns routes nearly every expert of every layer (P(unused) = (1 − 10/512)^T: 0.7 % at 256, ~10⁻⁹ at 1,024), so a layer's copies need not wait for its router. `ExpertStream` copies each layer's non-resident experts, in ascending id, on its own copy stream two layers ahead of compute, into half l % 2 of a ring of borrowed frames (§9). Events per layer: `landed[l]` (compute waits before the layer's experts) and `consumed[l]` (layer l+2's copies wait for it). Runs of consecutive experts coalesce into one copy; all of a layer's copies are enqueued as one `CopyBatch` (`cudaMemcpyBatchAsync`), so the host never waits on a full copy queue. The slot tables (I32 [layers][experts]) sit in the ring's first bytes.
- **Ring size.** The lease covers the largest non-resident count of any layer at lend time, widened by the experts the ring's own leases evict; when the placement keeps every expert resident (possible on this card's 96 GB), a layer has at most its lease-evicted experts to stream and prefill is compute-bound. Halves capped at 384 records were 9 % slower at pp4096 (RTX 5090, 2026-10-05).
- **SSD tier.** An expert without a host copy is streamed too when the tier has stream slots: its record is read from the artifact into a slot `kReadAhead` = 2 layers before its copy; a span thus reads each SSD-only expert once. Unpinned records go through four pinned bounce parts, still two layers ahead.
- **Results.** Streamed experts resolve like resident ones (`MoeExpertSource::prefetched`): never staged, never CPU-served — the same record bytes, so the same bits. A streamed build against staging (2026-10-05, RTX 5090, ABBA): pp16384 1,408.3 → 2,053.5 tok/s (+45.8 %), pp4096 at chunk 1024 +3.5 %, tg512 unchanged; greedy ids identical.

Chunks below 256 columns keep the MoE's staging, or the gated CPU split: a narrow streamed call copies each layer's experts only once its routing is known and leaves the thinnest non-resident ones to the CPU engine (§10). The split is planned from startup-measured rates at `kSplitRateShare` and takes calls up to `cpu_split_columns` (1,536 on the reference link, scaled inversely to the measured one; at most 4,096).

### 13.4 The wide expert route

- **Route.** An expert with more than `kMaxColumns` = 8 columns in a call takes the wide route when its gate and up input scales are equal; otherwise the narrow route, at any width (`moe::wide_route`). Both conditions are properties of the call's routing and the stored scales, never of placement, so outputs stay placement-invariant. Experts with unequal gate/up scales would need two A operands in one MMA tile; the narrow route computes them exactly. NVIDIA's export never produces such experts (24,576 of 24,576 share the scale in Qwen3.8-Flash-Next-NVFP4). The narrow kernels skip wide jobs; the fork route is limited to calls of at most 8 columns, where no expert can be wide.
- **Kernel.** A block-scaled tensor-core grouped GEMM (`mma…kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64` over the canonical A4 activations, `wide_expert.cuh`): a persistent warp-specialized TMA pipeline, 64 columns × 128 rows × 128-wide K tiles, the gate/up and down epilogues in registers. W4A16 experts take an exact wide route instead (`wide_expert_a16.cuh`; `WideA16::kA4Residual` for its three-term residual fast path, §17), which gives the narrow route's and the CPU engine's bits.
- **Not with the staging overlap.** Overlapped staging passes with wide experts gave run-to-run differences (prefix-cache exactness failures, varying greedy ids; 2026-10-05). The cause is not identified; the code keeps them apart (`moe_layer.cu`: `overlap` requires `!use_wide`).
- **Speed** (2026-10-05, RTX 5090, toggle ABBA): the expert op at T = 4096 79.2 → 1.86 ms per layer call; pp16384 at chunk 4096 646.8 → 1,081.9 tok/s (+67.3 %), pp4096 at chunk 1024 +12.8 %, tg512 unchanged (decode does not use it). Teacher-forced logits of the narrow and wide routes were bit-identical on this artifact (equal scales everywhere).

### 13.5 Dense compute of a wide call

With the link paid once per span, a span is bound by its kernels. The prefill-wide dense routes:

- **Router projection.** Calls of at least 64 columns use a tiled kernel (512-thread CTAs, K walked in 256-element slices): 370 µs against 755 µs per 4,096-token call, bit-identical by construction (each output's summation order is the narrow mapping's).
- **q8 MMA tiles.** The Flash-Next shapes select wider tiles above 96 columns (r64_t128 at 4,096 columns, −14 to −30 % per shape; every tile's output bit-identical to `r32_t128`): +4.8 % at pp131072, 0.0 % at pp16384 (RTX 5090, 2026-10-05).
- **`split_rows`.** Where every boundary, row count and pointer is 8-element aligned (all of Qwen4Exp's splits), a thread copies 16 bytes; the copy is exact.
- **GGUF imports (IQ3_S and the official FP8 checkpoint).** Calls of 32 or more columns decode each weight segment once into a plane (64 MiB of workspace) and run tensor-core tiles over it; with the `ggml-bf16-prefill` fast path (`LinearRoutes::GgmlBf16Wide`) wide calls run BF16 tiles over the same planes instead — each code converts exactly to BF16, each product is exact, only the sums round. The GGUF expert banks take the same tiles in calls of 14,336 or more columns (`WideA16::kGgmlBf16`, `kGgmlBf16MinColumns`; 32 experts' planes, 225 MiB). Measured (RTX 5090, IQ3_S, INT8 KV, 2026-10-19, three rotated rounds): dense tiles +27.4 % at pp32768 and +36.4 % at pp65536; with the expert tiles too, +40.8 % / +56.0 % (225.2 → 144.4 µs per token at 64K); below 14K columns the expert route is no faster (its planes are a fixed per-expert cost) and keeps the exact tiles. Time to first token on the server workload: 128K prompts 29.2-29.5 → 19.71-19.89 s, 250K 58.79-58.86 → 39.21-39.40 s.
- **Qualification.** The BF16 routes are fast paths under the numerical acceptance policy: the dense route against an FP64 oracle on every GGUF format, width and layout; the expert route against the exact one; the model-level prefill gate passes on INT8, FP8, NVFP4, K8V4, VQ2 and K4V2 KV and fails on BF16 KV, where the exact route is the reference ([model-quality.md](model-quality.md#6-numerical-acceptance-policy-and-fast-paths)). The exact items are bit-identical (`fast_path_gate identical`). The prefill's arithmetic can move greedy ties, so a fast-path build can differ from the exact route's text (§17).

### 13.6 PLE rows

The host reads a chunk's n-gram rows before its launch. A call with at least 512 distinct missing blocks splits them over 8 threads with a 16-block ring each (`NgramVolume`); the volume is opened unmapped (`FileMapping::None`), since a mapped data section makes every unbuffered read pay cache coherency. Both were measured on the 5090 machine (2026-10-05, 128K prompt, RTX 5090): unmapping 3,255 → 5,331 tok/s (+64 %, 21.6 → 5.9 s of reads), the thread split 5,331 → 6,576 (+23 %, reads 5.9 → 1.29 s). During a walk span's steps the host also reads the **next span's** rows into its chunk slots (checked by lane, call, position, width and a hash of the window): −2.5 % at pp128k (2026-10-06, RTX 5090). Row formats, the RAM-resident mode and the row cache are §12.

### 13.7 Residency: promotions and scores

- **No prefill call promotes.** The rule and its evidence are MoE §7.3. Calls of at most 8
  columns (`kServedCallColumns`, the CPU-served openers and tiny suffixes) promote at the
  decode rate; wider calls and walk spans promote nothing. The prompt's routes still count
  toward the cache's scores and seed the prompt-seeded expert counts (§9). Here a walk's
  promotions would churn ~12,000 records, ~1.2 s of link per 64K span, and 16 per layer and
  chunk measured slower than none in every regime (128K + 512 tokens: 23.16 → 21.10 s total,
  decode 66.3 → 89.5 tok/s; serve C = 4 fill −4.3 %, warm −5.6 %; 2026-10-06, RTX 5090).
- **Batched copies.** What promotions remain (decode) are enqueued through `CopyBatch`, in
  groups of 64 records with a completion event each: issuing 12,000 records as single
  `cudaMemcpyAsync` calls held the host ~1.2 s on a full copy queue, one batch returns in
  about a millisecond at the same link rate (`core/copy_batch.h`; the mechanism is MoE §7.3's).

### Rejected

- Chunk search in 1K steps from calibration: the chunk is a fixed default (16,384) measured directly; only the halving rule at startup remains.
- Strata's short first chunk and architectds/Strata's +21-59 % chunk-planning claim: never reproduced here; the walk removed the per-call link term the short chunk dodged.
- IoRing for volume reads: ~5 % over overlapped reads; not adopted (2026-10-05).
- 32,768-column prefill chunks (2026-10-16, RTX 5090): pp32768 +3.8 % but pp65536 −5.7 %; kept 16,384.
- A liveness-sized IQ3_S 16K-chunk workspace: 16K chunks −0.6 to −0.1 % against 8K; the BF16 tiles gain ~1 µs per token.
- Reading only the next chunk's rows behind the decode gate: superseded by the span-wide prefetch of §13.6.
- FP8 A8 prefill dense permission (inexact): superseded by the qualified BF16 GGUF tiles where the format allows them.
- A PCIe-bound prefill cap (40.8 GB per 16K chunk, ~21K tok/s on the 5090's x8 link at 60 % non-resident experts): no equivalent measurement exists on the RTX PRO 6000 (Gen5 x16, every expert resident), so no cap is stated here.

## 14. KV, attention and long context

QSA reads its cache in two separate ways. Selection scans the pooled index-key plane, one BF16 key
per 4-token block. Attention reads the K and V rows of the at most 2,051 selected tokens (budget
2,048 plus the ratio 4 minus 1) from the paged KV store. While the visible context is at most 2,051
tokens every token is selected, so QSA equals dense causal attention.

There are 13 KV layers: the 12 Main Text attention layers and the MTP block's, one `--kv-dtype`
profile and one 64-token page group for all of them
([paged-kv-cache.md](paged-kv-cache.md)). The profiles are `bf16` (the default), `int8`, `fp8`,
`nvfp4`, `k8v4`, `vq2` and `k4v2`.

### 14.1 Selection over the pooled index-key plane

- The pooled key of a block is the RMS-normed (unit-offset weight), RoPE'd BF16 mean of its four raw
  index keys, rotated at the block's first token. `qsa_pool_keys` and `qsa_prepare` write it when
  the block completes; the raw keys of a block's first three positions wait in the sequence's
  raw-key tail.
- The pooled keys live in their own BF16 plane per layer, `[Di/R, 64, 1, pages]` (32 rows here):
  64 B per token per layer, 832 B per token over 13 layers, 208 MiB per lane at a 256K context. The
  plane is a BF16 activation plane in every profile, so `--kv-dtype` never changes which tokens a
  query attends to, only how their keys and values are represented.
- Selection (`src/ops/qsa/qsa_select.cu`) is an exact top-512-blocks select: a score kernel over
  16-column row tiles, a register-tiled wide kernel for rows of at least 32 columns, and a radix
  select over 64-bit (score, lower-id-first) keys. The FP32 score arithmetic is fixed, so scores and
  ids are the same bits however the call is split, and every launch captures into CUDA Graphs.
- Cost (nsys, RTX 5090, `int8`, dense8m, 2026-10-06): 12 µs per call at an 8K context and 24 µs at
  128K, over 12 calls per plain forward (13 with MTP). At W = 256 and 128K the selection is 390 µs
  of the call's 580-700 µs (op bench, `infernix_qsa_attention_bench`): the scan grows with context,
  the attention does not.
- A truncated pooled plane (4-bit codes, partial BF16) was evaluated and rejected
  ([model-quality.md §3.5](model-quality.md#35-rejected-truncated-qsa-index-planes)).

### 14.2 Attention routes and kernels

`qsa_attention` picks a route by the call's width (`QsaAttentionRoute::ByWidth`, the default):

- Calls of at most 8 columns split each column's attended tokens into 64-token tiles, 33 splits at
  this geometry, the same for every width. A column's bits then do not depend on the other columns
  of its call, which preemption and prefix-cache resumes rely on; `test_qsa.cpp` checks it.
- Wider calls split to one wave of the decode kernel's CTAs, `clamp(SMs / (columns x kv_heads), 1,
  48)`, at least 4 once split; the SM count is queried at runtime. This is 42-73 % faster at 43-85
  columns than the earlier 340-CTA / 33-split rule (op bench, `int8`, 2026-10-06), and the calls it
  would run
  unsplit take the Tensor Core prompt route instead.
- The prompt route (`qsa_prompt.h`, one CTA per (column, KV head)) scores with FP32-accumulated
  m16n8k16 products and multiplies FP16 probabilities by FP16 values; it serves every storage at
  head_dim 256 and at most 16 query heads per KV head. Within its class a column's result does not
  depend on the call's width.
- The exact route (§17) takes the prompt route at every width (`QsaAttentionRoute::ColumnInvariant`),
  so a position's bits never depend on how its text was split into calls.

The decode kernel (`attention_kernel<Storage, Group, Exact, Stages>`) runs one 512-thread CTA per
(split, KV head, column). Each 64-token tile's codes and scales are gathered into dynamic shared
memory by `cp.async`, double-buffered when a split holds more than one tile; the first tile is
issued before the query heads load and rotate. Every cell decodes to its exact FP32 value, the
split partials merge in a fixed order (`merge_kernel`), and both kernels launch as PDL dependents.
Staging took the exposed time per call from 45-47 µs to 18.2 µs at T = 1 (8K and 128K alike, since
attention covers at most 2,051 tokens), and MTP at 8K from 46 to 34 µs (RTX 5090, dense8m, `int8`,
2026-10-06); decode moved +3.3 % at 8K plain and +2.3 % at 128K plain, neutral at 128K MTP. The
attention family as a whole is 2.4-4.8 % of a decode token's kernel time; its per-call time is
still above the ~3 µs kernel budget of §8 (open).

Beyond the dense window the same prompt route gathers each column's selected blocks (§13), with a
call's chains split into one-warp CTAs and merged by a second kernel when the exact route runs it
narrow.

### 14.3 The KV profiles

Per-token KV bytes (13 layers of 2 KV heads plus the pooled keys, head_dim 256):

| `--kv-dtype` | K+V per token and KV head | Per context token | 256K context |
|---|---:|---:|---:|
| `bf16` (BF16 K, FP16 V) | 1,024 B | 27.5 KB | 7.20 GB |
| `int8` (INT8-G64) | 528 B | 14.6 KB | 3.82 GB |
| `fp8` (E4M3-row256) | 516 B | 14.2 KB | 3.74 GB |
| `k8v4` (FP8 K, NVFP4 V) | 402 B | 11.3 KB | 2.96 GB |
| `nvfp4` (NVFP4-G16) | 288 B | 8.3 KB | 2.18 GB |
| `k4v2` (4-bit Lloyd-Max K, VQ2 V) | 196 B | 5.9 KB | 1.55 GB |
| `vq2` | 132 B | 4.3 KB | 1.12 GB |

`vq2` and `k4v2` add a fixed exact window of about 15 MB per lane (below). Stored codes are
`kv_cache_append_batch`'s, the same bytes the dense-attention models store. `qsa_prepare` appends
the five non-windowed storages itself; for `vq2`/`k4v2` `qsa_attention` appends the call's K/V.

- Decode reads: the tile loaders decode codes and scales to exact FP32 values (E2M1 times E4M3 is
  exact in FP16); rotated domains (`fp8`/`nvfp4`/`k8v4`/`vq2`/`k4v2` keys, and values for the last
  four) are rotated back on the unnormalized output. That map is linear, so split partials still
  merge by their weights.
- The `vq2`/`k4v2` exact window follows the position-only rule of
  [paged-kv-cache.md §9.3](paged-kv-cache.md#93-vq2k4v2-exact-recent-key-window): a key j of query
  p reads its exact INT8-G64 row when j < 64 or j >= p − 768 and its slot tag matches, and its codes
  otherwise, so the representation never depends on how the context was chunked. The window planes
  are five per KV layer and lane (`prefix::KvWindowGeometry`), about 1.1 MB each at 2 KV heads;
  `QsaWindow` state-image parts carry them through prefix snapshots, pause and replay (§15). Calls
  up to 256 columns write their window slots inline; wider single-sequence calls stage exact rows
  in the workspace and commit them after attention.
- Quality (teacher-forced, dense8m, chunk 256, against `bf16` KV, RTX 5090, 2026-10-06; 3,580
  positions over code/doc/chat in the dense regime, the last 1,024 of a 16K text in the sparse
  regime; paired dNLL ± 95 % interval):

| `--kv-dtype` | KL mean | dNLL, all | dNLL, long16k |
|---|---:|---:|---:|
| int8 | 0.061 | +0.007 ± 0.007 | +0.016 ± 0.014 |
| fp8 | 0.069 | +0.006 ± 0.008 | +0.024 ± 0.013 |
| nvfp4 | 0.070 | +0.004 ± 0.008 | +0.001 ± 0.013 |
| k8v4 | 0.069 | +0.001 ± 0.008 | +0.012 ± 0.015 |
| vq2 | 0.071 | +0.010 ± 0.008 | +0.019 ± 0.014 |
| k4v2 | 0.070 | +0.009 ± 0.007 | +0.014 ± 0.014 |

- Speed (single runs, same rig and date; the five upper rows under locked clocks about 13 % low in
  absolute terms, `vq2`/`k4v2` after the reset):

| `--kv-dtype` | pp32k tok/s | tg8k | tg128k |
|---|---:|---:|---:|
| bf16 | 5,384 | 99.2 | 57.5 |
| int8 | 5,552 | 99.4 | 61.4 |
| fp8 | 5,551 | 100.1 | 61.8 |
| nvfp4 | 5,501 | 99.5 | 60.9 |
| k8v4 | 5,549 | 99.7 | 60.5 |
| vq2 | 5,261 | 99.8 | 63.1 |
| k4v2 | 5,063 | 94.8 | 56.0 |

Decode at 8K is within ±1 % of `int8` for every profile: decode attention is latency-bound and
format-independent. `vq2`'s smaller KV leaves more expert frames at 128K; `k4v2` pays for its Q4 key
encode and decode (op bench, decode W = 1 including the Op's append: `int8` 15.8 µs, `vq2` 22.5 µs,
`k4v2` 24.6 µs).

- Production runs `--kv-dtype fp8` at `--max-context 262144` (recipe C; the `fp8-a8-pair` dense
  route is qualified with FP8 KV, §17 and §18).
- Qualification: `infernix_qsa_test` (`tests/ops/test_qsa.cpp`) is storage-generic — the FP64
  oracle over decoded K/V with the rotations, at decode, verification and prefill widths, across
  the dense/selected boundary and page boundaries, plus the window (matching and stale tags, a
  staged wide call) and page spaces. The real tests take `INFERNIX_QWEN4_KV` for every storage,
  plain and under MTP, prefix-cache and preemption scenarios (§17).

### 14.4 Long-context KV memory

Live KV is device-resident. The frame cost of a filled context is the table above (at 256K,
`int8` KV is 3.82 GB; the expert-frame budget is §9).

- **Elastic device pool.** The pool's backing is a `VmmRange`: the address space for the whole
  `--max-context` plan is reserved, and 2 MiB chunks are mapped individually. 512 pages (32,768
  tokens) are mapped at startup and counted as fixed; the rest is unmapped. Growth happens in
  64-page units when an admission is still short after prefix-cache eviction: memory the sizing law
  would give the expert frames anyway comes first, the rest is taken from the frames, as many as
  the growth needs. Growth is refused while frames are lent, except for leases a round boundary can
  return (a prefill's stream ring and wide arena; before that, a request queued behind a long
  prompt waited out its whole prefill, issue #24), and only when the pool's capacity leaves room
  for what the return would add — a return that cannot grow the pool only makes the prompt's next
  chunk lend again; a full-`--max-context` return once gave 4,221 frames back for nothing
  (2026-10-16). `tests/test_kv_cache.cpp` covers the backed limit and first-fit allocation.
- **Shrink.** A release lowers the backed limit to the free top and returns whole frames at once.
  After 60 s with every lane free, the maintenance pass first releases the prefix cache's idle
  host-backed device blocks (a later match restores them from the Host tier; nothing is lost), then
  shrinks; a monitor thread wakes an idle engine for it. The delay spares a follow-up turn a Host
  restore (about 90 ms for 128K `int8`) and an expert reload.
- **Resident placements.** When the expert placement holds every expert in VRAM (the default on the
  96 GB card when it fits; §9), the pool is fully mapped at startup and growth is inert — nothing is
  lent or evicted, so there is nothing to take frames from.
- Measured (RTX 5090, `int8`, 2026-10-05, one run each), elastic against a fixed pool at
  `--max-context 262144`:

| | Fixed | Elastic |
|---|---:|---:|
| Startup KV (MiB) | 3,360 | 458 |
| Expert frames | 8,197 | 9,299 (+2.8 GiB) |
| tg512 (tok/s) | 103.5 | 109.6 (+5.9 %) |
| 40K-prompt prefill (tok/s) | 2.83k | 2.91k |

  A ~40K prompt grew the pool to 626 MiB taking 88 frames; the release shrank it back and returned
  63 frames at once. Greedy ids are identical in every arm, including the same prompt at
  `--max-context 49152`.
- **Host residency.** The only host residency of KV today is the prefix cache's tiers
  ([paged-kv-cache.md §3.5](paged-kv-cache.md#35-host-capacity), §15): cached blocks and snapshot
  images in pinned Host slabs and the Disk tier, restored on a match. The QSA ops additionally
  support page spaces (`QsaPageSpaces`): pinned, GPU-mapped host planes read zero-copy by the
  attention and selection kernels, plus lent device planes, covered by `test_qsa.cpp`. The Program
  does not use them — `layer.spaces` stays at its default, every page a pool page. The planned live
  host tier with CLOCK over hot pages was not built; the elastic pool and the prefix
  tiers hold the frame budget instead.

### 14.5 Text YaRN beyond 262,144 positions

- Contract. `--rope-yarn-factor F` is startup-fixed, finite in [1, 4] (default 1). The reference
  model has one shared rotary embedding, so the factor scales every text rotation: attention Q/K,
  the QSA indexer's queries, the pooled index keys and the MTP layer. `max_position_embeddings`
  (262,144) is the original span; the context ceiling is 262,144 x F, at most 1,048,576, and a
  `--max-context` above the ceiling is refused at planning. The factor extends the ceiling only,
  not `--max-context` itself. Factor 1 keeps every existing rotation bit for bit.
- Implementation. Attention Q/K use `ops::rope` with the model's `PreparedRope`, built once per
  Forward by `prepare_rope(rotary_dim, theta, {F, 262144})` — the qualified Qwen3.5 YaRN path
  (correction range, linear ramp, attention scale 1 + 0.1 ln F). `QsaGeometry` carries
  `yarn_factor` and `original_positions`; the index-query and pool kernels take the same preparation
  and follow the attention kernel's coefficients: the FP64 angle reduced to one turn, FP32
  `sincosf`, cosine and sine both multiplied by the attention scale. At factor 1 the FP32 `powf`
  path is unchanged.
- Quality (teacher-forced NLL over the last 1,023 positions of one 327,680-token code and docs
  stream, RTX 5090, `int8` KV, dense8m, 2026-10-06; paired dNLL ± 95 % interval):

| Context | Factor | NLL (nats) | Against |
|---|---:|---:|---|
| 16K | 1 | 1.95308 | |
| 16K | 2 | 1.96378 | +0.0107 ± 0.0137 vs 16K f1 (KL 0.078) |
| 240K | 1 | 1.33160 | |
| 240K | 2 | 1.34469 | +0.0131 ± 0.0137 vs 240K f1 (KL 0.085) |
| 320K | 2 | 1.33085 | −0.0008 ± 0.0169 vs 240K f1 |
| 320K | 4 | 1.33653 | +0.0057 ± 0.0141 vs 320K f2 |

  Inside the native span, factor 2 costs about 0.011-0.013 nats at both lengths; neither change is
  significant on its own, but they agree, so YaRN is enabled only when the length is needed. Beyond
  the native span it works: at 320K with factor 2, the extra 80K tokens of context score the same
  tail as native 240K, and factor 4 at 320K is within noise of factor 2.
- Speed (one rep, same rig, a 311,296-token prompt at factor 2, chunk 4,096): prefill 7,955 tok/s,
  decode 66.4 tok/s plain and 76.0 with MTP. Against 245,760 tokens without YaRN at
  `--max-context 262144` (8.2k, 73.7, 87.9) these follow the length; the YaRN index rotation's own
  cost was not measured at equal length.
- Tests. `infernix_qsa_test` gains YaRN index rotations (factor 4, original span 4,096) against an
  independent YaRN formula in its FP64 oracle; `infernix_rope_test` covers the Flash-Next YaRN
  cases; `infernix_qwen4_exp_forward_real_test` takes `--rope-yarn-factor` for teacher-forced
  scoring.

### 14.6 Rejected

- QUILT shared-tile prompt kernel (one shared tile over the union of a query tile's selected
  blocks): an exactness-preserving CTA can keep at most two columns' accumulators live (width 4
  needs 130,560 of 65,536 registers); the union removes 31.8 % of per-query block loads at width 2
  at 16K and 27.5 % at 140K, below the 30 % trigger; neighbouring queries share fewer blocks than
  QUILT's models (63.6 % / 55.0 % against > 85.9 %); no prototype was timed (2026-10-13,
  kido5217/Infernix issues #2, #45, #48).
- Wave split rule for calls of at most 8 columns: a column's split count depended on its call's
  width (W = 2 gave 42 splits, W = 1 gave 48) and broke preemption and prefix-cache equality; kept
  the fixed 33 splits at about 2 µs (W = 5) to 10 µs (W = 8) per call (2026-10-06).
- INT8 Tensor Core prompt scores (Strata's integer prompt attention): FP16 query limbs as exact
  int8 limbs on m16n8k32.s8, combined exactly in int64; +12.5-13.1 % attention time with 2 limbs
  and +20-21 % with 3 at 4,096 columns (op bench, cold L2) — the FP16 path runs one MMA per score
  tile near the tensor-core bound (2026-10-14).
- Tensor-core QSA decode (NInfer): 23.1 µs against 14.3 µs at one query column, wins only from
  about 8 columns per row (1.35-1.5x); not ported (2026-10-14).
- Truncated pooled index-key planes: fails the selection-agreement bar at >= 128K context (0.9213
  mean against 0.95 required, worst layer 0.8951 against 0.90) —
  [model-quality.md §3.5](model-quality.md#35-rejected-truncated-qsa-index-planes) (2026-10-13).
- 8-CTA cluster per column in the selection (distributed shared memory): slower at every context
  (2026-10-06).
- Live KV host tier with CLOCK over hot pages: only the ops page spaces were built; the
  Program keeps live KV device-resident, and the elastic pool plus the prefix tiers hold the frame
  budget (2026-10-05).

## 15. Prefix cache, preemption and scheduling

Two documents are the authority; this section states only what is specific to the family:
[hybrid-prefix-cache-spec.md](hybrid-prefix-cache-spec.md) §17 (the binding's contract, and §17.1
the tier rules) and
[resource-scheduling-and-context-cache.md](resource-scheduling-and-context-cache.md) (permits,
queue, preemption policy, resume). Code: `src/models/qwen4_exp/program/prefix/`.

### 15.1 Index and binding

The family owns its prefix index (`prefix_index.{h,cpp}`, forked from `src/runtime/prefix_cache/`
on 2026-10-29): the runtime index serves Qwen3.5 alone and lacks the Host-born snapshots, the Disk
residency and the zero-token resume's head residual. The tree, matching, choice, supersession and
GDSF valuation are the spec's; the fork adds the tier rules of spec §17.1. The physical binding
(`prefix_cache.{h,cpp}`, `prefix_program.cpp`, `state_image.{h,cpp}`, `disk_tier.cpp`,
`disk_store.{h,cpp}`, `call_plan.{h,cpp}`) owns what the index's opaque ids name: `block id →
DeviceKVPageLease`, `node → mtp_next`, the pinned Host slab pool, the write-through queue,
restore batches with per-layer events, copy-outs and the Disk tier. The index decides; the
binding moves the bytes its decisions imply.

Mechanics the generic text leaves to the family:

- Pages have no refcounts: a path pin keeps a pinned node's Device copy; a lane holds private
  leases plus its path; `insert_block` moves a lease to a new node or an attaching Host-only node
  (spec §7.5: a recomputed block never holds two pages).
- Snapshots are Host-born, or written through to the Disk tier when `--host-context-mib 0`: a
  capture copies the lane's image and partial page one decoder-layer group at a time on the
  transfer stream; a restore is one batch per admission on the restore stream; the first call
  waits per layer (`ForwardBatch::layer_waits`), ticketed so a recycled event is never waited on.
- Copy engines run FIFO across streams (H2D measured on the RTX 5090): restores are submitted after
  their call's inputs, per lane the transfer stream runs endpoint tail, endpoint image, then
  write-through, and round inputs go through `upload_pinned` (§8). The planned D2H counterpart
  (`download_pinned`) was never built; the submission order sufficed.
- Lane-resident endpoints: when a finish captured a snapshot at the lane's frontier and the slot
  still holds that image, an admission from it skips the image restore, and the quote lowers that
  candidate's restore bytes by the image size. Duplicates (another lineage) get no shortcut.
- The frontend's keys and hints (`block_hashes`, `block_extras`, `tap_hints`) and `plan_taps` are
  the runtime's, unchanged; Vision's tap exclusions and encode-window re-planning are spec §17
  (§16).

### 15.2 MTP rules

Spec §17 sends the drafter's cache rules here. Cell c pairs the residual at c with the token at
c + 1, at KV slot c; pooling runs when a column completes a 4-block, reading earlier positions
from the tails.

| Rule | Content |
|---|---|
| MR1 | `Lane::mtp_cells` (cells `[0, mtp_cells)` final); settled invariant `saved = residual(state_tokens − 1)`, `mtp_cells ∈ {state_tokens − 1, state_tokens}`. A violation switches drafting off for the rest of the request and reports a Warning: verification rejects bad drafts, so output cannot change |
| MR2 | A cancelled plain round leaves `saved = residual(state_tokens − 2)`; `commit`'s cancelled branch copies the round's `mtp_residuals_` column into the saved column and sets `mtp_cells = state_tokens − 1`. Cancelled verify rounds commit 0 columns: nothing to repair |
| MR3 | Finish flush: when `mtp_live && mtp_cells == state_tokens − 1 && history.size() > state_tokens`, one kv-only cell (~0.2 ms), so every normal endpoint is `mtp_written` |
| MR4 | A snapshot at F records `mtp_written = (mtp_cells == F)` and `mtp_next = history[F]`; node b records `mtp_next = publisher.history[64(b+1)]` |
| MR5 | Device adoption only on a matching next token (`attach = !mtp \|\| lane.history[64(b+1)] == node.mtp_next`): a node's Device and Host copies are byte-identical; a mismatch stays private and is counted |
| MR6 | Resume at F, k = ⌊F/64⌋. F % 64 ≠ 0: share `[0, k)`, private tail, `mtp_cells = F` when `mtp_written && prompt[F] == mtp_next`, else F − 1. F % 64 == 0 with a matching next token: share `[0, k)`; mismatch: share `[0, k−1)` and copy-on-write the anchor. F == n (zero-token resume, §15.4): the next token is unknown at admission, so a block-aligned end always copies the anchor (and a tail otherwise); the resumed lane's first round writes the cell |
| MR7 | Prepends re-pool the 4-block holding F−1 from the tails; the tails kernels leave the latest block's `q % 4 < 3` keys (main layers never read them), so prepends are exact |
| MR8 | A lane continuing differently from block b's publisher reads one off-branch MTP cell per boundary: drafter-only, counted (`mtp_branch_mismatches`) |
| MR9 | Drafts, verification records and acceptance statistics are never cached |

### 15.3 Taps, the call planner and the cost model

`plan_taps` with Vision exclusions, then `plan_calls` (pure: plans depend only on their
arguments). Boundary exact taps (Explicit, Structural, coalescing, branch) always split; the
generation opener and Automatic taps split only when the split adds at most `split_budget` = 0.1 s
of fixed call cost and the tap's call is CPU-served (≤ 8 columns, ~35 ms), else become flexible at
the next inner call boundary. The layer walk's span cost is charged to every plan (§13); a cut
costs a span only where it leaves a narrower call between walk-eligible ones. Three family points:

- **Branch tap.** An exact boundary tap where the prompt leaves its cached path, planned when
  that point is at least `kBranchTapMinTokens` = 1,024 tokens past the resume frontier: a re-sent
  transcript with inserted content resumes at the branch instead of the last chunk boundary.
  Unlike Qwen3.5's removed branch-point taps, a Flash-Next prefill call has a large fixed cost (the
  expert stream), so re-prefilling after a boundary costs nearly a full call (measured 2026-10-17:
  40,128 against 32,768 tokens reused, 0.6-1.1 s against 2.1-2.2 s to first token).
- **Tail tap.** The finish endpoint covers continuations only, so a prompt that fits one call
  publishes nothing a repeat could resume from. An exact boundary tap at the resume-grid frontier
  at or before the prompt's last token splits a final call of at most `kResumeGrid` = 16 tokens
  off the plan: the snapshot is the prompt's own tail, which any sharer resumes from at the cost
  of one short tail call. It is cut for a fresh one-call span under free cache resume (the
  prompt-end capture of §15.4 covers repeats), and for every span without it (the opener is
  grid-floored and never realizes); a resumed span never takes it, because splitting a resumed
  span is the hazard class the preemption and replay equivalences of §15.7 are verified on.
- **Cost model.** `ContextMachineCostModel`
  (`src/runtime/engine/context_cache/context_cost.{h,cpp}`), fitted 2026-10-05 on the RTX 5090
  (PCIe 5.0 x8, INT8 KV, `infernix_bench` pp): 0.1645 s per further call, 22 µs per token, 1.06 ns
  per attention pair, 1.67 s per layer-walk span, ρ = 10/512, within 5 % from 4K to 128K tokens;
  `instance.cpp` records the fitted `prefix_cost` fields.
  The Program rescales the span and the restore rate to the link it measures at startup, so the
  constants travel to the RTX PRO 6000.

Without free cache resume (§15.6) every tap moves to the grid position at or before it, the opener
demotes, and every call boundary and resume is a multiple of 16 tokens.

### 15.4 Zero-token resume

The prefill-to-decode transition of a fresh span captures a `Boundary` snapshot at the prompt's own
frontier n whose image carries the pre-LM-head residual (`head_residual`; under MTP the drafter's
saved residual is that same column, so the image keeps it once). An exact repeat matches it —
`match` rejects a candidate at frontier == n without a residual — resumes with an empty prefill
plan (`computed_prefill_tokens` = 0), and regenerates its first token from the stored residual at
the same logit width the cold run's finish used. Under MTP the snapshot is the transition itself
(MR1): the drafter's pending cell n − 1 is unwritten, and the resumed lane's first round writes it
into a private copy (MR6 at F == n); the capture publishes its anchor block only once the
capturing lane's own first round (or the MR3 flush) has written the cell (`anchor_pending`). The
capture never raises the lane's `deepest`, so the finish endpoint still publishes over the turn's
generated tokens. The value gate skips it within one tap separation (64) of the path's deepest
published frontier — the image would cost more than the short tail call it saves: capturing at
every transition cost a concurrent six-request decode ~4 % tok/s and filled the Host tier to 99 %,
gated it sits within 2 % of the baseline (measured 2026-10-23 on the artifact, fp8 KV, DFlash2,
four boots alternating). Skips are counted: `hybrid_prompt_end_gated` (by design),
`hybrid_prompt_end_failed` (want of a staging slot, Host slabs or a free page). The measured
consequences are in spec §17.

### 15.5 Disk tier

`--prefix-cache-disk DIR` (`--prefix-cache-disk-gib N`, default 64 GiB; Linux only — cuFile has no
Windows build) adds Disk slots of one Host slab each to the index and replaces this model's
whole-file save, which rewrote up to the whole Host tier every save interval and could never
exceed RAM. The residency rules, the writer's claim discipline and `kDiskWriteWindow` = 64 MiB of
block writes in flight are spec §17.1. Family-specific facts:

- Device memory moves by cuFile batch I/O, host memory by `pread`/`pwrite` on the queue's four
  threads (cuFile 1.19, CUDA 13.4: its compatibility mode faults inside `cuFileBatchIOGetStatus`
  after batch reads into host memory, and a driver close-then-reopen freed its pools twice — the
  driver opens once per process). Batch reads reached 10.5 GiB/s from one thread where the
  stream-ordered `cuFileReadAsync` reached 2.9-3.6 GiB/s (T710, compatibility mode, 2026-10-29),
  so the restore's per-layer events stay on the copy streams and a Disk read happens before the
  admission, which is deferred meanwhile. No wait on the worker covers Disk writes: a `drain`
  that also waited blocked every admission under memory pressure for the whole write backlog
  (production steps of 15-35 s; a test: 2.6 s for 22,121 queued blocks).
- The spec §17.1 rules are the fix for the first spill policy: a spilled snapshot kept its Host
  block copies forever, the Host tier evicted snapshots only, and a full Host tier refused every
  new capture — after 37 prompts of 30k tokens, repeats reused 0 % with the Disk tier against
  99.9 % without it (test servers, 2026-10-29).
- Only the no-Host-tier mode stages in VRAM: two staging regions (a packed image plus its tail
  each) and 64 block records, 354 MiB (2026-10-29). With every expert resident the prefill chunk
  is sized from the VRAM left free, and the staging cut it from 4,096 to 2,048 tokens (a 30k
  prompt prefilled 12,600 against 13,500 tok/s, 2026-10-29). Without a Host tier an open tap and
  a tap being written can also hold both regions when the prompt end is captured; the capture
  waits in its fixed slot and taps found without one are skipped (on a three-prompt boot, two of
  three same-boot repeats resumed at a 14,336-token tap, 0.6 s to first token, 2026-10-29).

A sequence that recomputes a block replaces its Disk record and journals the records below it
again; `DiskStore` keeps the journal, the manifest and the lock (spec §17 Persistence).

### 15.6 Memory, flags and exactness

- The Host tier is one pinned slab pool, default 8 GiB (`kDefaultHybridHostCacheBytes`), pinned in
  chunks no slab crosses (4 GiB); a tier smaller than one image + 2 slabs fails at startup with
  the needed MiB. `--host-context-mib 0` needs the Disk tier: without either tier construction
  refuses to start.
- The cache spends no expert frames: cached blocks use idle KV-pool pages, and a shortage evicts
  them before the elastic pool grows (`hybrid_reclaim`, §14). Admission still checks the whole
  extent against the pool (`feasible`), so the oldest resident can always reach its extent.
- `--device-snapshot-slots` is refused (the binding has none); `--prefix-cache-file` and
  `--prefix-cache-save-mins` are Qwen3.5's; `--no-prefix-reuse` disables the cache and gives the
  deterministic cold decomposition; `--kv-capacity` defaults to `auto` (free VRAM becomes Device
  block cache). Tap budgets: 8 per request with a Host tier, else 2; the ladder chunk is capped at
  4,096 tokens (`tap_ladder_chunk`, §13.1), so the defaults are ladder max(4096, 2·min(chunk,
  4096)) and gap max(1024, min(chunk, 4096)) — 8,192 and 4,096 at the default 16,384 chunk.
- Exactness: storage is byte-exact and a resume computes what the capturing lineage computed. On a
  construction-exact exact route (recipe C, §17) every resume, coalescing wait and decode-state
  endpoint equals the cold run bit for bit; elsewhere equally precise orders may flip a near-tie
  (§17), and `--no-prefix-reuse` is the reproducibility switch. Preemption restores exactly;
  replay may re-round (§15.7).

### 15.7 Preemption

The `preemption` fast path changes the page contract: a binding reserves the prompt plus one
round — `round_positions() = max(max_width_, mtp ? mtp_k_ + 1 : 1) + 1` positions past the
frontier, the cells a round writes — and `reserve_units` grows a decode, control or prefill unit's
lane to frontier + one round before the round, evicting unpinned cached blocks, draining transfers
pinned only by in-flight Host writes, and growing the elastic pool. Without the fast path a request
holds its whole extent from admission and nothing pauses. A shortage the reclaim cannot cover goes
back to the Engine, which pauses the youngest resident (policy in
[resource-scheduling-and-context-cache.md](resource-scheduling-and-context-cache.md) §5).

`start_pause` runs at a committed boundary. It publishes the lane exactly as a consistent abort
does (`prefix_finish`: the MR3 flush, the remaining blocks, the state at the frontier, the
endpoint, write-through); inside a layer walk's span nothing past the span's start is published and
the staged tap captures are dropped. The ResumeState keeps the ledger, the frontier and the
request's measurements and decisions; the model state is not owned — the snapshot stays evictable,
and the cache holds, writes to Disk first and fetches it before the resume (spec §17.1 rule 7).

A request paused inside its prompt binds the prompt again and prefill continues from the snapshot
it finds. A request paused after its first token binds its ledger: lookup keys are the prompt's,
chained over the output blocks (`ledger_prompt`), and the index may return a snapshot at the
paused frontier itself — the deepest affordable one restores and prefill calls replay the ledger
from there without sampling; a restore at the frontier leaves nothing to replay and counts as a
snapshot restore. A restore is bit-exact (the image and blocks are the lane's own); a replay
recomputes with prefill arithmetic and the tokens after it may differ from an unpaused run, as
on Qwen3.5. A Vision item inside the replayed range is encoded again; penalty counts are rebuilt.

The MTP drafter clamps its cells to the lane's mapped positions (shared prefix pages plus private
ones); when the clamp counted only the private pages, a long reused prefix put every draft cell on
one position inside a shared page (a 44,783-token continuation: 98 rounds for 176 tokens before
the fix, 76 after, 22 % fewer, identical tokens; 2026-10-06).

Measured (RTX 5090, dense8m, INT8 KV, 2026-10-06, `infernix_qwen4_exp_preemption_real_test`: two
lanes, a 48-page pool, 1,024-token prompts, 1,200 outputs, plain and MTP; the older lane ran short
at frontier 1,596-1,599 and the younger was paused once in every scenario):

| Path | Result |
|---|---|
| Exact (snapshot at the frontier), plain and MTP | Both requests equal their solo runs, 1,200 of 1,200 tokens |
| Replay (restore ~1,600 tokens short of the frontier) | Plain equalled its solo run (1,200/1,200); MTP diverged after 572 tokens (replay arithmetic); both completed at full length |
| Speed against the pre-per-round build (ABBA) | tg512 110.73 → 110.70 tok/s, pp4096 2.470 → 2.459 s, MTP tg512 140.83 → 140.33 (−0.35 %) |

Faults those tests found, both fixed: a paused lane's pages stayed pinned by in-flight Host
writes until `reserve_units` and `hybrid_reclaim` learned to drain first; a resume whose cached
state is the paused frontier itself has no suffix to plan and decodes at once.

### 15.8 Mixed rounds: prefill beside decode

While another lane decodes, a prefilling lane's calls ride in the decode rounds' Forward
(`mixed_calls`; `--prefill-chunk-beside-decode`, default 1,024 tokens, a multiple of 128; 0 or
at least `--prefill-chunk` means no cap). Only where the route's contract holds: the
column-invariant exact route (the bits equal the separate calls') and request-coupled fast routes
(§17); `Program::can_mix` refuses the prompt's last call, a layer walk's span, a replay, an image
prompt, the MTP drafter's cells and scoring. DFlash2 windows ride (`run_mixed` stages the call's
draft-feature columns).

`mixable_width` caps the call at the smaller of the beside-decode chunk and the room left by the
decode rows (`chunk − C × max_width`), floors it to the 16-token grid, and inserts the split into
the lane's call plan so the plan continues on the grid. The mixed Forward is eager (the prompt
call's width needs the wide arena, which no CUDA graph may capture): the decode or verification
rows run as their round runs them, the prompt call's tokens, positions, n-gram rows and RoPE
blocks after theirs (`ForwardBatch::prefill_part`), then the call's DFlash2 window into its lane's
ring. The decode round's acceptance or draw is synchronized before the call advances, the call's
routes are credited once (with the decode rows', one residency round), and the round's time is no
decode cost sample. The decoding requests keep streaming between a long prompt's calls; the
prefilling request pays a longer time to first token. The Engine-side rule is
[resource-scheduling-and-context-cache.md](resource-scheduling-and-context-cache.md) §5.1; `make
t1`'s mixed shape (`6x2+1024`) checks the exact route's equality against the two calls, and the
T2 smoke test runs golden tokens beside a long prompt's mixed rounds.

### Rejected

- A new cache system: all sequence state is fixed-size per lane, KV is causal and page-local, and
  the QSA ratio (4) divides the page (64) — the Hybrid cache with a family binding covers it.
- Legacy (`--use-original-prefix-caching`, removed 2026-10-09) and moving Qwen3.5's
  `LogicalKVPageStore` to core: ~20 Qwen3.5-only hooks and a large refactor of the production 27B
  path for nothing path pins do not already guarantee.
- Device snapshot slots by default (`C + 1`): 84 frames at C = 1, ≈ 0.35 % decode (estimated), to
  save ≤ 4 ms per non-resident resume; refused outright by this binding.
- Aligning resumed calls to the absolute chunk grid: no exactness gain (equivalence is
  lineage-based), and it can add a wide call (~1.3 s).
- A multi-call narrow suffix route (≤ 8-column calls): each call re-stages its distinct experts
  and the CPU takes ≤ 8 misses per layer per call — 50 prompt-hot tokens as 7 narrow calls
  ≈ 0.6-0.7 s against ~0.3 s for one CPU-assisted call (§13). A single small call stays CPU-served.
- MTP cell c at KV slot c + 1: a QSA op change (pooled 4-blocks would straddle pages) plus
  requalification; the per-node next token (MR4-MR6) solves it in the binding.
- A zero-split GDN state tap (spec §7.2-7.3): automatic exact taps are admitted only at Δ ≤ 0.1 s,
  so the remaining cost is ~1.3 s once per new prefix, for a tap that splits a chunk — not worth
  a GDN kernel state tap no model implements.

## 16. Vision

Images and video run the Qwen3.5-family Vision tower, shared. The shared pieces live in the frontend
tier `src/models/qwen_common/` (after the 2026-10-30 model restructure): the stages and multi-item
pass (`execution/vision_tower.*`), the weight stream, the control plan, the overlay layout and the
config parser. Qwen4Exp's tower differs only in its output width: the merger's fc2 writes the text
hidden size, read from the bound `merger_fc2` (`load.cpp bind_vision`, Qwen3.5's parameter names).
The artifacts already carry the tower, so nothing was reconverted (§6); the deployed reports store
it under `model_type "qwen4_exp_vision"` with both preprocessor resources. Production serves with
`--vision --vision-offload auto`.

Facts of the deployed artifacts, verified against their conversion reports:

| Item | Value |
|---|---|
| Tower config | depth 27, hidden 1152, intermediate 4304, 16 heads (head dim 72), patch 16, temporal patch 2, merge 2, 2,304 position embeddings (48 x 48) |
| Text side | M-RoPE interleaved, sections [11,11,10], rotary 64 of head 256 (partial factor 0.25), theta 1e7; merger output = text hidden 2560 |
| Weights | 897,862,112 B BF16 (856 MiB), excluded from quantization: prelude 8,849,664; per layer 30,479,008 x 27; merger 66,079,232 |
| Weight staging | prelude + merger + two layer slots = 135,886,912 B, 256-B aligned |
| Caps | 32,768 merged tokens per prompt, 16,384 per item; video 2 fps, 4-768 frames |
| Load checks (`config.cpp`) | 72-wide heads; interleaved M-RoPE (pair i on axis i % 3); merger output = text hidden size |

### 16.1 Tower GEMM routes

The tower's eight BF16 problems run registered tensor-core routes, owned by the Qwen3.5 family in
`src/models/qwen3_5/shapes/bf16/`: patch [1152,1536], fused QKV [3456,1152], attention output
[1152,1152], fc1 [4304,1152], fc2 [1152,4304], merger fc1 [4608,4608], merger fc2 [2560,4608], and
[5120,4608] for 27B artifacts (Quasar) whose BF16 towers run the same code. The 4304 problems use the
tail-capable TMA form (64-element K boxes with hardware zero fill, stores skip rows at or beyond N);
large T rasterizes row-fast so the activation streams from DRAM about once. T <= 8 keeps the
fallback's skinny GEMV, unchanged. Measured (2026-10-05, RTX 5090, cold CUDA graphs): 12.4-18.3x the
scalar fallback at T >= 1,024 across all eight shapes (acceptance target was 5x); the merger
[5120,4608] reached 247 TFLOP/s at T = 65,536. The text BF16 problems are separate registrations
under `src/models/qwen4_exp/shapes/bf16/` (§8); selection and tuning in
[linear-tuning.md](linear-tuning.md#5-bf16-vision-tower-problems).

### 16.2 Frontend: media, caps, positions

The shared frontend acquires and preprocesses media (a worker pool, a retained decoded-media cache,
a live-payload cap; flags in 16.7) and packs each item's patches as pageable BF16 `[1536, P]`. The
prompt layout inserts the tower's markers (`<|vision_start|>`, `<|image_pad|>`, `<|video_pad|>`)
and `assign_positions` writes M-RoPE positions axis-major `[3, n]`: text tokens take equal axes, an
image run takes `(cur, cur+y, cur+x)` then advances by `max(gh, gw)`, and `rope_delta =
max + 1 - length`, so every later token takes `index + delta` on all three axes.
`--vision-max-merged` bounds an item's pixels through smart resize; an item above 16,384 merged
tokens is rejected. Video takes the same paths with per-frame spans and segments.

### 16.3 M-RoPE through every text call

Every Forward, verification and MTP call stages three-axis RoPE positions beside the KV index:

- io: `rope` I32 `[columns, 3]` axis-major and `block_rope` `[batch, 3]` precede `ngram`, so the
  decode and verification prefix upload reaches them; the vision and MTP arrays are prefill-only.
  Cost: 12 B per column.
- `LaneRope` keeps the prompt's `[3, prompt_tokens]` only for prompts with media; `rope_of` returns
  the stored axes or `index + delta`; `block_start_rope` is the position of token
  `R * floor(start / R)` (QSA ratio R = 4).
- `ops::rope` Text M-RoPE, `qsa_index_query` and the pooled keys all rotate by these positions
  (`ops/rope.h`, `ops/qsa.h`); a pooled key rotates by its block's first token. Equal axes rotate
  exactly as the 1-D index, so text output is unchanged, and every call has the same form, so no
  CUDA-graph topology depends on media.
- PLE and the n-gram table hash placeholders like any token (§12). QSA equals causal attention while
  `p + 1 <= B + R - 1` (invisible below 2,051 visible tokens), which is why the wiring oracle runs
  past that length (16.8).

### 16.4 The encode window

`plan_vision(prompt, reused)` encodes the items ending past the reused prefix; a resume frontier
strictly inside an item never exists (§15 drops those), and the plan refuses a prefix that ends inside
an item. One `VisionTowerPass` encodes every item layer-major: stage 0 embeds all items, stages 1-27
run one layer for all items, stage 28 merges. It is resumable (`advance(end_stage)`) and every Op runs
per item with its own T = P_i, so an item's embedding is bitwise the same alone or among others while
each layer's weights are read once for all. The outputs land in one token-ordered handoff BF16
`[2560, V_encoded]`; a call's visual tokens are a contiguous slice (`visual_slice`), so a prefill
chunk need not stop at an item boundary. The handoff lives until the lane's last prefill call.

Placement (pure planning in `vision_window.{h,cpp}`, host-tested): **W** runs the whole pass from the
idle prefill workspace in one step when the window fits `work_capacity` and the estimated time is
within `kVisionStepSeconds` = 0.25 s; only the handoff is lent. **L** lends handoff and window and runs
`layers_per_step = ceil(depth * 0.25 s / estimate)` layers per progress call, so other lanes decode
between steps. The rate constants (`kVisionGemmFlops` 1e14, `kVisionAttentionFlops` 6e13, 842 MFLOP
GEMM per patch, 27 * 4 * P^2 * 1152 attention FLOP) are estimates and say so in the header; the tower
was never profiled separately.

MTP cells take token `c + 1` — visual where that token is an image token, so a chunk's last cell can
read the next chunk's first image token. Cells run in sub-chunks of 512 (`kMtpChunkColumns`), each
with its own RoPE block and vision columns.

### 16.5 Offload, frames, fallback

With `--vision-offload on` (auto resolves on for Qwen4Exp, off for Qwen3.5) the tower's weights bind
`HostPinned` and a `VisionWeightStream` stages them per window — prelude, layers 0-26, then the
merger — through the 135,886,912 B staging inside the window. The first upload waits a compute fence
and the lease's wait events; `finish()` makes compute wait for every upload. Qwen4Exp uses no VMM pool
(`VisionOverlayAssets{pool = nullptr}`); Qwen3.5's overlay window keeps its `EvictableWeightPool`.

Device memory comes from the expert cache (§9): `ExpertResidency::lend` evicts the lending pool's
cheapest run and returns a lease; on completion only the handoff stays lent (`split` + `give_back`);
every release path (last call, replay, cancel, abort, fail-all) returns the frames. When the residency
lends no frame — a placement that holds every expert — or the window exceeds the lendable bytes, the
window is a device allocation for the encode's lifetime, and a failed allocation fails only that
request (`RecoverableExecutionError`, 2026-10-23; before this every media request failed on the
resident placement). Leases reload their evicted experts on return (§9): the decode-only hit rate
after 32K and 128K prompts was 94-95 % and 90-97 % with the reload, against 86-88 % and 81-85 %
without (2026-10-12).

Prefix reuse needs no vision mechanism: block keys hash the vision grid and placement, text after
an image carries the key, and a replayed range re-encodes any item it covers (planned from the
restore frontier). An image's LM prefill costs seconds, so reuse is the largest vision speed-up (§15).
Telemetry: `vision_seconds` (tower GPU event time) and the `vision_offload_*` window, evict, restore
and byte counters per request; the request log line prints them when set.

### 16.6 Costs and open items

Measured end to end (RTX 5090, one request, INT8 KV; user
[guide](../qwen3_8-flash-next.md)): a 1024 x 1024 image takes ~1.4 s to first token on recipe B
(`qwen3_8_flash_next_nvfp4_dense8`) and 1.9-2.0 s on the GSQ-RCO IQ3_S import. The split between
preprocess, tower and LM prefill was never attributed; the tower alone was estimated at 10-40 ms with
the tensor-core routes. A resident tower would instead hold ~325 frames permanently, about 1.3 % of
decode on every request (scaled from a VRAM-reserve A/B: 436 more frames measured +1.8 % tg512,
62.25 →
63.39 tok/s, RTX 5090, 2026-10-04), which is why offload is the default. Open: the pageable
patch upload can queue behind promotion DMA at C > 1
(never measured), and the window's rate constants and step length remain estimates.

### 16.7 Flags and protocol

`--vision` (off by default; media without it fails with HTTP 400 `vision_disabled` and the Program
refuses the prompt), `--vision-offload auto|on|off`, `--vision-max-merged N` (64-32,768, default
32,768), `--media-cache-mib N` (1024), `--media-live-mib N` (2048), `--media-preprocess-threads N`;
the serving schemas carry media as `image_url` / `input_image` (and the video forms).

### 16.8 Tests

Host: `test_vision_window.cpp` (placements W and L, slices, the MTP cell mapping, prefix rules),
`test_rope_positions.cpp` (against an independent port of the position rule), `test_expert_cache.cpp`
(lending and the load-serial ABA cases; §9). Real: `test_vision_oracle_real.cpp` with
`tools/flash_next/vision_oracle.py` runs a ~2,130-token media prompt — past the 2,051-token QSA
threshold, so selection runs over M-RoPE pooled keys — against the independent FP64 reference:
M-RoPE positions exact, token runs exactly covering the merged tokens, first greedy token exact.
`test_smoke_real.cpp` encodes a media request in the boot's resident placement and in a switched
(lending) placement. Qwen3.5 keeps `test_vision_workspace.cpp` and the vision Op tests; the shared
stages are one op sequence, so its embeddings are unchanged by construction.

### Rejected

- Quantized vision weights (Q8/Q4/Q5): below the upstream BF16 the artifacts carry.
- VMM eviction for the window (Qwen3.5's overlay mechanism): lent frames are clean caches, no restore needed.
- Elastic resident tower (keep the frames between close images): saves at most ~30 ms per small image, adds policy.
- Zero-copy weight reads inside the GEMMs, and a content-keyed embedding cache: the tiles re-read weights and the link cannot sustain it; repeats are covered by prefix reuse.
- 1-D RoPE for text lanes: replaced by uniform `[T,3]`; equal axes are bit-identical in every RoPE path.
- Per-lane device RoPE plane `[3, max_context]`: a gather kernel in every call; host `rope_of` covers every case for 12 B per column.
- Column-invariant vision Linear routes: the per-item pass gives batch invariance without constraining routes.
- Placeholder embedding at MTP image cells: the vLLM / Qwen3.5 convention (token c+1's embedding) taken instead.
- A lower item cap (e.g. 4,096 merged tokens): kept 16,384 and documented the cost instead.

## 17. Numerics and qualification

The family's numerical contract has three layers. The **exact route** (`--disable-fast-paths`) is
bit-stable from release to release, and request-invariant where the artifact's formats allow it.
The **canonical arithmetic** of §17.2 and §17.3 makes the routed experts exact: their output bits
depend on the stored words and the input only, never on placement, batch shape or route. The
**fast paths** give up bits for speed under the acceptance policy of
[model-quality.md](model-quality.md#6-numerical-acceptance-policy-and-fast-paths) section 6, whose
registry (6.7) this section does not repeat. What this section keeps is the family's own
arithmetic, its oracles, its invariance machinery and its tests.

### 17.1 Op oracles

Every Op is qualified against an independent oracle at the real shapes under the
[Op development](op-development.md#6-qualification) rules: a naive FP32/FP64 oracle for floating
point, an exact scalar for exact arithmetic, packed inputs decoded with their stored scales —
written from the upstream mathematics, never from the kernels. The PLE n-gram table's oracle is
§12's; per-op error is measured by `tools/flash_next/block_check.py` and `upstream_check.py`.

| Op | Oracle | Boundaries |
|---|---|---|
| `offloaded_sparse_moe` routing and dispatch | FP64 softmax, top-k and weighting formula over the FP32 `projection_fp32` logits; exact top-10, lower id wins exact ties; `moe_dispatch` has its own exact oracle | Near-tie fixtures; record sources device frames, pinned host bank and SSD providers (§9, §10) |
| `offloaded_sparse_moe` A4 quantizer | Exact scalar of the §17.2 rule | Zero blocks, saturating blocks (scale 448), subnormal scales, E2M1 and E4M3 ties; the GPU encoders swept exhaustively near every rounding boundary and strided over all FP32 words |
| Narrow expert route, GPU and CPU, W4A4 and W4A16 | Exact independent scalar of §17.2/§17.3, plus pinned golden hashes | n = 1-8; every residency mix (frame, staging, zero-copy, CPU-served); every CPU ISA; bit-exact equality across routes; `exp_c` and `swiglu_bf16` exhaustive over all 65,536 BF16 inputs |
| Wide expert route (W4A4) | FP64 with exactly decoded weights and canonical A4 activations | Per-accumulator bound against the exact block sum; end-to-end against the FP64 chain; experts with at most 8 columns or differing gate/up input scales keep the narrow route (bit-equal to the CPU engine); placement invariance over 3 and 64 staging slots |
| W4A16 wide route; GGUF and block-FP8 experts, every route | Exact rationals over the independently decoded stored values and the A16-encoded activations; the scalar reference is itself checked against a bignum sum | Bit-equal on every route: widths 1-300 tested, frames, staged, zero-copy, CPU-served; every CPU ISA (`infernix_ggml_linear_test`, `infernix_offloaded_moe_ggml_test`, `infernix_offloaded_moe_ggml_cpu_test`, `infernix_offloaded_moe_fp8_test`, `infernix_offloaded_moe_a16_cpu_test`) |
| MTP drafter (resident experts, proposal head) | FP64 with exactly decoded weights; the head is an FP32 `projection_fp32` over `q4_g64_fp16` weights | Drafter shapes; width-invariant by construction |
| `hyper_connection` | FP64, weights decoded exactly | Stream and rank shapes of the mix |
| `qsa_prep`, `qsa_select` | FP64 block scores | Near-tie allowance of the selector contracts; identical selection under every KV profile |
| QSA attention | FP64 attention over each profile's decoded K/V | Every `--kv-dtype` with that profile's decode criterion; VQ2/K4V2 exact-window positions |

### 17.2 The canonical W4A4 arithmetic

**Requirement.** For every routed expert computed on a **narrow route** (n ≤ 8 token columns in
this forward), the BF16 output is **bit-identical** whether the GPU computed it from a cache frame,
from a staging frame, or the CPU computed it from the host record. Experts with n > 8 columns use
the GPU **wide route** (§13); the CPU serves narrow jobs only (§10). The route depends on n, never
on placement.

**Why exactness is cheap.** Doubled E2M1 values are small integers, and E4M3FN values times 2⁹
are integers, so every term of a W4A4 dot product is an integer multiple of 2⁻²⁰ and a whole row
fits an int64 accumulator exactly. Integer addition is associative, so any split, order,
instruction set or processor gives the same sum; only two roundings remain, both at the end and
both fixed.

**Definitions** (`src/ops/common/canonical_math.h`, shared by host and device). For a code word c
and an E4M3FN scale word w (exponent e = bits 6:3, mantissa m = bits 2:0):

```text
c2(c) = 2 · e2m1(c)             ∈ {0, ±1, ±2, ±3, ±4, ±6, ±8, ±12}    (c = 0x0 and 0x8 give 0)
Ŝ(w)  = e4m3fn(w) · 2⁹          = (e == 0) ? m : (8 + m) << (e − 1)     ∈ [0, 229,376]
```

**A4 quantization** of a BF16 vector v with the matrix's FP32 `input_scale` g, per 16-element
block. This is ModelOpt's reference rule, evaluated in exact IEEE FP32 with round-to-nearest-even:

```text
amax = max_j |v_j|                                  exact
s    = e4m3fn_satfinite_rn( amax / fl(6 · g) )      0 ≤ s ≤ 448
if s == 0:  every code of the block is 0
else:       d      = fl(e4m3fn(s) · g)
            code_j = e2m1_satfinite_rn( v_j / d )   ties to even on the E2M1 grid, |value| ≤ 6
```

`tools/flash_next/a4_reference.py` ports this rule and ModelOpt's FP4 fake quantizer. They agree
except for one ModelOpt guard: a dequantized scale below 1e-5 is replaced by 1.0, zeroing any
block with amax below about 6·10⁻⁵. TensorRT-LLM's deployed quantizer has no guard, and neither
does this rule: the guard is a calibration-time convenience, not part of the NVFP4 format. The
ports were bit-equal on 327,680 blocks; with CUDA and `nvidia-modelopt` installed the tool also
runs ModelOpt's own Triton kernel.

**Row product.** For one output row r, one column, K inputs in K/16 blocks (K = 2,560 gate/up,
K = 640 down):

```text
P_b = Σ_{j<16} c2(w[r,16b+j]) · c2(a[16b+j])                      |P_b| ≤ 2,304, exact int32
S   = Σ_b P_b · Ŝ(ws[r,b]) · Ŝ(as[b])                               exact int64, any order
y   = bf16_rn( (fl32_rn(S) · 2⁻²⁰) ⊗ α ),   α = fl32(weight_scale_2 · input_scale)
```

Each term is below 2⁴⁷ and K/16 ≤ 160, so |S| < 2⁵⁵; the int64 → FP32 conversion is the sum's
only rounding, the scaling by 2⁻²⁰ is exact, and ⊗ is one FP32 multiply and one BF16 rounding. α
is computed once per matrix at load (§6), so both processors use the same word.

**SwiGLU** between the projections, on the BF16 outputs widened exactly:

```text
silu_c(g) = g ⊘ (1 ⊕ exp_c(−g))                  if g ≥ 0
           = (g ⊗ e) ⊘ (1 ⊕ e),  e = exp_c(g)     if g < 0
h_i = bf16_rn( silu_c(g_i) ⊗ u_i )
```

⊘, ⊕ and ⊗ are IEEE FP32 operations with the parenthesization exactly as written; the two branches
keep SiLU's sign and magnitude for large negative g, where the single formula overflows `exp_c`.
`exp_c` is the one shared implementation in `canonical_math.h`: Cody-Waite range reduction; a
fixed degree-7 Taylor polynomial by explicit `fmaf` (degree 6 measured 2.65 ulp against FP64
over every BF16 input, degree 7 measures 0.857 ulp); exponent reconstruction by two exact
power-of-two scalings, so subnormals form correctly; defined overflow, underflow and NaN (the
canonical 0x7FC0). Its inputs are BF16, so CPU-GPU equality of SiLU is verified exhaustively
over all 65,536 inputs. h is quantized to A4 with the down matrix's `input_scale`, and the down
product gives the expert output y_e in BF16.

**Semantic boundaries**, following NVIDIA's deployed NVFP4 MoE:

```text
x (BF16) → A4 → gate/up → BF16 → SwiGLU → BF16 h → A4 → down → BF16 y_e → combine (FP32) → BF16
```

Nothing else is frozen: tiling, split points, processing order, staging and the int64 reduction
order are free; the combine is an FP32 evaluation rounded once to BF16. **Compiler and
instruction rules:** CPU `-ffp-contract=off -fno-fast-math`, explicit FMA intrinsics only (in
`exp_c`); GPU `__fmaf_rn`, `__fmul_rn`, `__fadd_rn`, `__fdiv_rn` and integer intrinsics; the
`cvt.rn.satfinite` E4M3/E2M1 conversions are matched by the CPU's integer implementations,
verified by the exhaustive encoder sweeps of §17.1.

**What placement invariance buys.** Greedy output does not depend on cache contents, prefetch
timing or the CPU/GPU split, and no round ever waits for a promotion. An engine whose outputs move
with cache state pays for it: Strata's top-1 agreement between its cache states measured only
95-97.7 % (2026-10-04). Every residency route is tested by exact comparison (§17.6).

### 17.3 W4A16 and exact block arithmetic

**W4A16** (`a16_*` in `canonical_math.h`). Some NVFP4 exports are weight-only: their routed
experts carry codes, block scales and a global weight scale but no activation scale, because the
serving runtime keeps activations in BF16 (recipe C, §6). Such experts take 16-bit activations
with the same requirement: bit-identical output on every route, the wide one included, and on every
CPU ISA. A BF16 column v of K elements is encoded once:

```text
M_j  = 128 | mantissa_j (normal), mantissa_j (subnormal)      e_j = max(E_j, 1)
emax = max_j e_j          (1 for an all-zero column; Inf or NaN makes the column non-finite)
X_j  = sign_j · rne( M_j · 2^13 / 2^(emax − e_j) )            |X_j| < 2^21
```

v_j = X_j · 2^(emax − 147) exactly within 2^13 binades of the column's largest exponent; smaller
elements round at 2⁻²¹ of it, far below the outputs' BF16 rounding. 21 bits (not 23) let the wide
route form a block's products with two 32-deep integer MMAs; the golden expert outputs did not
change. The narrow routes store X as three bytes: lo and mid, the unsigned low bytes of its two's
complement, and hi = X >> 16 in [−32, 31]. With c2 and Ŝ as in §17.2:

```text
P_b = Σ_{j<16} c2(w[r,16b+j]) · X[16b+j]       |P_b| < 2^29, exact int32
S   = Σ_b P_b · Ŝ(ws[r,b])                     |S| < 2^55, exact int64, any order
y   = bf16_rn( (fl32_rn(S) · 2^(emax − 157)) ⊗ m ),   m = fl32(1 / weight global scale)
```

m is stored in the record's tail (§6); the power-of-two scaling takes two exact steps, so a
subnormal product rounds once; a non-finite column gives the canonical NaN (0x7FC0) in every row.
The semantic boundaries are x (BF16) → A16 → gate/up → BF16 → SwiGLU → BF16 h → A16 → down → BF16
y_e → combine. The byte limbs make every product a byte product: the GPU narrow kernels use dp4a
(signed codes against unsigned lo/mid bytes and signed hi bytes); the CPU uses `vpmaddubsw` on lo
and mid and biased codes (c2 + 12) against signed hi bytes, subtracting 12 · Σ hi per block — three
byte products per weight instead of one, so the CPU's integer work per miss triples (§10.2) while
the GPU narrow kernels stay bound by the record reads they share with A4.

**Wide route** (`wide_expert_a16.cuh`, integer tensor cores). X is split as 2^14·H1 + 2^11·H0 +
8·L1 + L0; per 16-element block, two `mma.m16n8k32` with B = [8·c2 | c2] give c2·X exactly in
int32, and S accumulates in int64. A k32 MMA costs a k16 one's issue time (13 cycles per SMSP on
RTX 5090; a BF16 m16n8k16 costs 26), so a block costs what a BF16 tensor-core GEMM's would, and
the route returns the narrow route's and the CPU engine's bits: a W4A16 expert's output depends
neither on placement nor on how many columns share its call. The first wide route was a BF16
tensor-core GEMM qualified against FP64 (mean relative L2 7.1e-5); it made the bits depend on the
column count, so a KV block one request computed in a 476-column call and another reused differed
from the cold 512-column computation, and the prefix-cache real test failed its restore and
restart-resume checks (2026-10-09); the exact route passes. Speed (RTX 5090, T = 4,096): two k32
MMAs per block 6.24 ms against 7.24 ms for three k16, 128-element stages 5.45 ms, the W4A4 route
1.65 ms; end to end (orca artifact, INT8 KV, ABBA) pp1024 +2.0 %, pp16384 −3.2 %. The first
exact version (three k16 MMAs) measured −13.0 % on pp16384.

**GGUF blocks and block FP8** (`src/ops/ggml/ggml_exact.h`, `src/ops/fp8_block/fp8_exact.h`;
formats in [tensor-formats.md](tensor-formats.md#36-gguf-ggml-block-formats) §3.6/§3.7). The GGUF
imports and the official FP8 checkpoint store integer codes times integer sub-scales times
floating block scales. Every such weight is computed against the A16 encoding with every sum exact
and one rounding at the end; experts and dense projections share the contract, so a column's
output depends on nothing but its own values — placement, column count, segment splits and
schedules return the same bits, and every route may split K, rows and columns however it likes.

- **GGUF blocks.** A weight is κ · (d · sub · code − dmin · min_sub) with d = mant · 2^(e − 25):

  ```text
  R = Σ over 8-element chunks of mant(d) · sub · (Σ code · X) · 2^(e(d) − 1)
      − mant(dmin) · min_sub · (Σ X) · 2^(e(dmin) − 1)                  exact, int128
  y = fl32(R) · 2^(log2 κ + emax − 171)                                 one rounding to binary32
  ```

  Chunk sums fit int32, chunk terms int64 (< 2⁴⁹), R int128 (< 2⁸⁹ for K ≤ 8,192).
- **Block FP8** (`fp8_e4m3fn_b128`). Every E4M3FN value is w · 2⁻⁹ with |w| < 2^18; each 128 × 128
  tile has one BF16 scale s = mant(s) · 2^(E(s) − 134). With E_min the smallest scale exponent of a
  row's tiles:

  ```text
  P_t = Σ over tile t's 128 elements of w · X                 exact int64 (< 2^46)
  R   = Σ_t P_t · mant(s_t) · 2^(E(s_t) − E_min)               exact int128
  y   = bf16_rn( fl32_rn( R · 2^(E_min + emax − 290) ) )
  ```

  The converter refuses NaN codes, subnormal or non-finite scales, and a row whose tile exponents
  span more than 62 binades (`kMaxExponentSpread`, the shift bound of `exact::add_shifted`); the
  real checkpoint's spread is at most 4.
- **The rounding** (`src/ops/common/exact_combine.h`): R rounds once to binary32 at the result's
  binade (`exact::to_f32_scaled_rn`); a BF16 output rounds once more, an FP32 output (the `lm_head`
  logits) does not.
- **Embedding and n-gram rows.** The IQ4_XS token embedding rows and the IQ4_NL n-gram rows are
  decoded exactly on the device from pinned host memory and rounded once to BF16 (§12).
- **Limb forms** are each route's choice: the GPU GGUF routes use dp4a (narrow) and int8
  `mma.sync` k32 (k16 for the half-sub formats Q6_K and IQ2_S); the CPU GGUF engine uses two int16
  limbs (X = hi · 2048 + lo) with 16-bit multiply-adds (`vpmaddwd`, `vpdpwssd` with VNNI), 1.5×
  faster than its first int32 form; the GPU FP8 route uses three base-64 digits against X's limbs;
  the CPU FP8 route decodes w = A + 2^8 · B by exponent from two `vpshufb` tables and multiplies by
  `vpmaddubsw`.

### 17.4 Column invariance and construction-exact configurations

Request invariance needs more than exact experts: a position's bits must not depend on how its
prompt and output were split into prefill calls, decode steps and verifications. A kernel whose
reduction order or route depends on the call's width breaks that. The family's answer, in
`src/models/qwen4_exp/program/fast_paths.cpp`:

- `kConstructionExact` lists the configurations whose exact route runs **column-invariant** kernels
  everywhere: recipe C (FP8-row dense, W4A16 experts) with BF16, INT8 or FP8 KV. There every
  projection, mixer, attention call and MTP drafter kernel of the exact route takes one arithmetic
  at every width (`LinearPolicy::A16ColumnInvariant` routes, `GdnChunkArithmetic::Recurrent` for
  the GDN prefill, the column-invariant QSA prompt kernel), so prefix-cache resumes at any
  position, coalescing and decode-state endpoints return the cold run's output without the
  `free-cache-resume` path (§15).
- `FastPathSelection::width_keyed()` names the fast paths that route by a call's width
  (`q8-sliced-k`, `expert-a4-residual`, `fp8-a8-pair`, `ggml-bf16-prefill`, `gdn-compensated`,
  `hc-sliced-k`, `bf16-index-mma`): they couple a request to its round. On a construction-exact
  configuration the schedule paths (`adaptive-speculation`, `free-cache-resume`, `preemption`)
  couple only timing — the column-invariant arithmetic keeps the bits through their resumes,
  replays and mixed rounds — so `request_invariant()` stays set with them; elsewhere their
  resumes and replays move the call grid and with it the bits.
- Width-routed kernels (`--width-routed-kernels`) require a construction-exact configuration; the
  selector refuses them elsewhere.

The gates that admitted it, and the record of the numerics change itself, are in
[model-quality.md](model-quality.md#68-exact-route-numerics-changes) 6.8; the FP64 Op oracle
favours the recurrent GDN there (output relative L2 at most 2.1e-3 against the chunked 4.4e-3).

### 17.5 Fast paths

The acceptance policy, the measurement method and every path's record are
[model-quality.md](model-quality.md#61-the-exact-route-and-the-fast-paths) 6.1-6.7; this section
states only the selection mechanism and the current qualified set.

`select_fast_paths` (`src/models/qwen4_exp/program/fast_paths.cpp`) derives the loaded artifact's
**configuration** from the GDN q/k/v/z projection's dense kernel family (Q8, BF16, row-FP8, NVFP4
or GGUF), whether the routed experts take 4-bit activations, and the KV format. A configuration
names kernels, never a checkpoint, so every artifact that runs the same kernels shares its
qualifications. A path runs when it applies to the configuration and appears in `kQualified`, the
registry mirror; explicit candidates and exclusions override, the startup log prints the selection
(`fast paths | ...`), and the paths not applicable to this family (the Qwen3.8-27B attention paths,
the 27B batched-decode and Q4W4A8 forms) are refused.

The qualified set today (evidence in model-quality.md 6.7):

| Fast path | Where it runs (configuration families) |
|---|---|
| `adaptive-speculation`, `free-cache-resume`, `preemption` | every Flash-Next configuration: Dense8 (Q8 dense, 4-bit experts), IQ3_S (GGUF dense), FP8 (BF16 dense), recipe A (BF16 dense, 4-bit experts), recipe C (FP8 dense, 16-bit experts) — every KV format each |
| `sampled-mtp-drafts` | every Flash-Next configuration, every KV format (changes no logit; distribution test) |
| `q8-sliced-k`, `gdn-conv-fused` | Dense8, every KV format |
| `qsa-prepare-fused` | Dense8 and recipe C, on the five KV formats the fused prepare serves |
| `gdn-f16-accumulate` | IQ3_S and FP8, every format but BF16 KV (no gain on Dense8) |
| `ggml-bf16-prefill` | IQ3_S, every format but BF16 KV |
| `expert-a4-residual` | recipe C, every format but BF16 KV |
| `fp8-a8-pair` | recipe C, with FP8, NVFP4, K8V4, VQ2 or K4V2 KV |
| `gdn-compensated` | recipe C with INT8 or FP8 KV |
| `hc-sliced-k`, `bf16-index-mma` | recipe C with FP8 KV |

With the width-keyed paths selected, MTP output no longer equals plain decode: the fused
convolution runs only in plain rounds and the Q8 table picks routes by width; the exact route
still does both. `tests/models/qwen4_exp/test_fast_paths.cpp` checks where each path's kernels run
and the configuration descriptions.

### 17.6 Engine-level and system tests

- **Bit stability and determinism.** `make t1` replays a round from one state twice or more and
  fails on differing bits, checking the logits and lane-state hashes against
  `bench/models/qwen4_exp/round_hashes.txt` (exact and fast routes). The smoke test compares the
  exact route's greedy tokens against `tests/models/qwen4_exp/smoke_golden.txt` — alone, six in
  flight, beside a long prompt's mixed rounds, resumed, and after an expert-placement switch.
- **Construction exactness.** `infernix_qwen4_exp_forward_real_test --split-equality` decomposes
  one prompt every way the Forward can (off-grid and 16-aligned splits, 16- and 48-wide middle
  calls, the tail call, prefill then decode) and requires byte-equal layers;
  `tests/ops/test_width_equality.cpp --invariant --strict` does the same per Op at every route
  threshold width; `tests/ops/test_column_invariant_projections.cpp` checks the
  `A16ColumnInvariant` routes' invariance itself.
- **Placement invariance end to end.** The prefix-cache real test resumes at taps, endpoints,
  off-grid points and through the Host and Disk tiers and compares against cold runs; the
  preemption real test pauses and resumes (snapshot restore and ledger replay) and compares
  against solo runs. Both need the exact route to be construction-exact on the cold-run
  comparisons (model-quality.md 6.8).
- **Recovery and policy.** `infernix_qwen4_exp_expert_fault_real_test` injects an expert-read
  fault: the round's requests fail, the Engine keeps serving, and a later request gives the
  pre-fault greedy ids. The transfer agent's LFRU victim and promotion decisions are replayed
  against `tools/expert_cache_replay`.
- **KV profiles.** Decode and prefill run under every `--kv-dtype` (bf16, int8, fp8, nvfp4, k8v4,
  vq2, k4v2); selection is identical across profiles on near-tie-free fixtures. The fast paths'
  acceptance runs are driven by `tools/quality/fast_path_gate.py` (model-quality.md 6.2-6.5).

### 17.7 Precision boundaries and measured quality

Every boundary was checked against the upstream code (Transformers `qwen4_exp` with its Qwen3.5
and Qwen3-Next parents, SGLang `qwen4_exp.py`) and measured on real text (2026-10-04, RTX 5090):
three frozen texts (511 positions of C++, 1,023 of a document, 1,023 of a chat transcript), scored
teacher-forced on identical token ids, differences paired per position as mean ΔNLL ± standard
error (`infernix_qwen4_exp_forward_real_test --dump-logits`,
`tools/flash_next/strata_compare.py`). A boundary is never less precise than upstream; where more
precision was possible it was kept only if measured not to hurt.

| Boundary | Upstream | The engine | Decision |
|---|---|---|---|
| Dense weights | BF16 | BF16, word-exact (§6) | The main quality advantage over the GGUF engines (model-quality.md §4) |
| Routed experts | NVFP4 W4A4 | NVFP4 W4A4, canonical arithmetic (§17.2) | No 8- or 16-bit activation path is needed |
| Residual stream (4 × 2,560) | BF16 | BF16 | FP32 tried: op chains stayed exact (median residual error 0.105 % → 0.03 %) but perplexity got worse (+0.030 ± 0.009 nats overall, +0.051 ± 0.015 chat); the model expects the BF16 rounding it was trained with. Rejected |
| Router logits | BF16 linear, FP32 softmax | FP32 (`projection_fp32`) | BF16 logits tie experts within one BF16 step and the lower id wins; BF16 − FP32 measured +0.008 ± 0.008 nats. No cost. Kept |
| LM-head logits | BF16 | FP32 (`projection_fp32`) | Perplexity unchanged (−0.0002 ± 0.0004 nats); BF16 logits flip greedy top-1 at 1.3 % of positions. The FP32 row is free next to the weight read. Kept |
| RMSNorm, hyper-connection norm | FP32 inside, one BF16 rounding | Same | Mixer error 0.13-0.23 % against Transformers' 0.27-0.35 % |
| GDN gated norm | Three roundings | One rounding | GDN error 0.35-0.50 % against 0.42-0.65 % |
| QSA v, gate, gated product | BF16 | BF16 | 0.43-1.8 % against 0.48-3.2 %; the excess is cancellation in deep `o_proj` sums |
| KV cache | BF16 | every profile of §14; BF16 the default, FP8 in production | INT8 costs +0.005 perplexity overall against BF16 KV (4.542 → 4.564) |

The engine-against-engine comparison (Infernix's NVFP4 artifact against the Unsloth UD-Q4_K_XL
GGUF that Strata runs) is recorded in [model-quality.md](model-quality.md#41-measured-quality)
4.1: Infernix ties on code and document and is better on chat (−0.10 nats), with BF16 dense
weights the largest single factor. These texts are short (≤ 1,024 tokens); long-context quality
is §14's.

**Rejected**

- FP32 residual stream: +0.030 ± 0.009 nats overall, +0.051 ± 0.015 on chat (2026-10-04, RTX 5090).
- BF16 router or LM-head logits: exact ties flip greedy top-1 (1.3 % of positions at the head);
  FP32 measured no worse at no cost (2026-10-04, RTX 5090).
- ModelOpt's 1e-5 scale guard in the A4 rule: TensorRT-LLM's deployed quantizer has none; it only
  zeroes blocks below ~6·10⁻⁵ amax.
- BF16 tensor-core wide route for W4A16 experts: column-count-dependent bits broke prefix-cache
  resumes (2026-10-09); replaced by the integer-MMA route at −3.2 % pp16384 (RTX 5090).
- llama.cpp's and Strata's Q8_1/Q8_K activations: less accurate than A16, and Strata's GPU and CPU
  forms disagree, so output depends on placement.
- FP8 tensor cores with FP32 accumulation for the FP8 experts: undocumented on sm_120, half rate,
  not reproducible on the CPU.
- A canonical fixed-order FP32 sum for the GGUF/FP8 formats: the tensor-core routes cannot follow
  one reduction tree.
- Expert refill after a prefill frame loan (NInfer `c7c36e88f`): decode unchanged, MTP decode
  −5.3 % (2026-10-15, RTX 5090).
- N-gram copy gate (giveen `4d39be1d4`): copies slower than MTP drafts; MTP −6.1 % at 16k context
  (2026-10-15, RTX 5090).
- Tensor-core QSA decode (NInfer `47cdc08ab`): 23.1 µs against 14.3 µs at one query column; wins
  only from about 8 columns per row.
- Fused PLE gate (NInfer `1b50c72d2`): about 5 µs per round for 26.6 % of normalized bits differing.
- 10,240-wide offset RMSNorm (NInfer `569861a3a`): 2-2.6 µs per draft step, not bit-exact past 16 rows.
- NInfer's NVFP4 expert kernels: about 10× the A4 contract's error, CPU and GPU sums differ,
  nondeterministic at T = 4,096 (2026-10-15, RTX 5090).
- `fp8-a8-pair` on INT8 KV (recipe C): fails the decode tail (p99.9 KL ratio 1.27 against 1.25,
  2026-10-17, RTX 5090); registered on the five formats it passed.

## 18. Calibration and configuration

The original design proposed a three-layer calibration system: a startup probe, an
`infernix-calibrate` product CLI with a saved profile, and online adaptation. Only parts of it
were built. There is no
calibration CLI, no profile, no fingerprint or drift check. What exists is narrower and cheaper:
development-time constants measured once on the development hosts, two short startup measurements
inside the Qwen4Exp Program, and per-subsystem online adaptation. This section states what adapts
to the host today, what became of the high-reward options of the original design, and where the configuration
surface is documented.

### 18.1 What adapts to the host

| Layer | What | Where |
|---|---|---|
| Compiled-in constants | Worker and job counts, cache intervals, cost-model coefficients, measured at development time on the development hosts (i9-13900K CPU; RTX 5090 and RTX PRO 6000) | `ProgramOptions` defaults (`program.h`) |
| Startup measurement | The host-to-device link rate and the CPU expert team's service rates, measured once per start (~60 ms for the team) | `measure_link()`, `measure_cpu_rate()` (`cpu_rates.cpp`) |
| Online adaptation | Draft length and its cost fits (§11); the expert swap, LFRU, prompt seeding and warm start (§9) | §11, §9 |

**Compiled-in constants.** The host-side defaults were measured once, not per machine:
`cpu_expert_workers = 6` (more measured slower: the CPU and the PCIe stage share host DRAM
bandwidth); `cpu_ggml_expert_workers = 16` for the compute-bound GGUF route (i9-13900K, tg512:
100.5 tok/s without CPU service, 98.4 at 6 workers, 103.7 at 16, 103.6 at 20; 2026-10-11);
`cpu_expert_jobs = 16` (C = 4 fill +7 %, cold MTP +3 %; 32 no faster). The prefix cost model's
coefficients were fitted 2026-10-05 (RTX 5090 at PCIe 5.0 x8, INT8 KV): 1.67 s per layer-walk
span, 0.1645 s per prefill call, 22 µs per token, 1.06 ns per attention pair, within 5 % from 4K
to 128K context; `instance.cpp` records them as the Program's `prefix_cost` fields.

**Startup measurements.** The Program measures two things per start and rescales what depends on
them:

- `measure_link()` times real expert-record copies into the staging slots (three repetitions) and
  rescales the link-bound costs to the result: the prefix restore rate, the layer-walk span cost
  (§15) and the decode promotion interval (§9). The reference is 27.5 GB/s (RTX 5090 at
  PCIe 5.0 x8); `ProgramOptions::link_bytes_per_second` fixes it instead of measuring. The
  promotion interval is 4 at the reference and scales with the measured rate (x8 4, x16 2 derived,
  not measured; clamped 1-8).
- `measure_cpu_rate()` (`cpu_rates.cpp`, ~60 ms) times the configured worker team on real records
  in one-column and 8-column jobs while the link copies, because decode misses and the prefill CPU
split (§13) compete with the staging's DRAM reads. The rates feed the decode miss divisor
(`cpu_pcie_divisor = 0` derives it) and the split's column limit; a busy host halves the team's
rate, so a loaded attempt is retried (§10.4). Reference rates stand when the team cannot be timed
(the SSD tier before its RAM copies fill).

Both measurements print one Debug line each at startup (`host-to-device link: ... measured; decode
promotes every N tokens; prefill CPU split up to K columns`).

**What is still per-host.** One constant states the old plan: `kDefaultPrefetchBytes = 2048`
(`w4a4_expert.h`) is a software-prefetch distance the comment calls a starting point, with
`infernix-calibrate` named as the tool that would measure it per target; no such tool exists.
A maintainer measuring a new host runs the probe suite by hand:
`tools/flash_next_probe/run_m0.sh` (host DRAM and expert-kernel service rates, GPU and PCIe,
NVMe, VRAM plan; outputs an inventory with system facts and advice such as the link's trained
width, DIMM speed and huge-page state). `tools/expert_cache_replay` replays cache policies on
route traces captured with `--route-trace` — development evidence, not a user-profile fit.

**Safety rule.** Everything above changes speed, never output. The exact int64 expert sums (§17)
make worker count, placement and measurement timing bit-invisible; the warm start and prompt
seeding touch cache counts only. The one performance knob that changes output,
`--expert-tail-skip`, is explicit, off by default, and the exact route refuses it
(`engine_options.cpp`).

### 18.2 High-reward options: outcomes

Built:

| Option | Built as | Where |
|---|---|---|
| Prompt-seeded cache | Prefill routes seed the LFRU counts at the prompt's end: counts only, credit 256 (`kPromptSeedCredit`), no promotions | §9 |
| Exclusive residency | The three-tier placement: every expert in exactly one of VRAM, RAM, SSD | §9 |
| Drafter at 8 bits | Every recipe quantizes the MTP drafter (Q8 projections, Q4 experts; Q2 optional) and the DFlash2 companion is Q8 | §6 |
| Warming (another form) | GPU warming CTAs prefetch the next layer's dense chain into L2 while the host computes misses: 40 MiB chain, 16 MiB input; tg512 plain 121.8 -> 123.3 tok/s | §8 |

Not built:

- Persistent decode megakernel. A 2026-11 experiment composing the GDN projection and middle
  chain into one dependency-counter launch was bit-exact but did not beat PDL at 1 x 1 (~5.7 µs of
  chain work behind a ~26 µs DRAM-bound projection). PDL plus CUDA graphs stayed the baseline.
- Draft trees over MTP top-2. The family verifies chains and refuses `--draft-tree-nodes` (§11).
- Intra-expert CPU + DMA split of one cold miss. The prefill CPU split (§13) splits at expert
  granularity instead; no row/block split of a single record exists.
- Two-layer-ahead prediction and the multi-token protection predictor. LFRU plus prompt seeding
  (§9) won on the replay evidence.
- Chained rounds. Rounds stay host-committed; the engine transaction is unchanged.
- BF16 GDN recurrent state. Snapshots keep exact state bytes (§15); no state-precision option
  exists for this model.

One rejected idea returned: Strata's expert tail-skip rule, first rejected as
"changes the mathematics", is offered since 2026-10-18 as an explicit approximate option,
`--expert-tail-skip R` (§18.3). The other rejected items (reduced-precision miss copies, W4A16,
W8A8, BAR-mapped mailboxes, entropy-coded codes, SM copy kernels, floating-point atomics) are
recorded with their sections (§17, §8, §3).

### 18.3 Configuration surface

The authorities, in order of exactness:

- `src/serve/serve_options.cpp` — every `infernix-serve` option, its arity and its `--help` text
  (one table drives both the parser and `--config`).
- `include/infernix/types.h` — `EngineOptions`, the in-process surface.
- [CLI](../cli.md) and [serving](../serving.md#server-options) — user documentation;
  [the model guide](../qwen3_8-flash-next.md#run) holds this model's recommended commands.
- §6 — the recipe, chosen at conversion (A, B `dense8`, C `orcarouter`, FP8, GSQ-RCO IQ3_S,
  DFlash2 drafter), recorded in the artifact.

Command-line options override the `--config` TOML file; file and command line share one option
table. Unspecified values take compiled-in defaults, adjusted only by the two startup
measurements (§18.1) — the old "unspecified values come from the calibration profile" is gone.

The flags specific to this family: the expert residency group (`--experts`, `--expert-placement`,
`--expert-swap`, `--expert-ram-mib`, `--expert-pin-mib`, `--expert-stripe`, `--expert-state`; runtime
switching through `POST /admin/expert-placement`), `--ngram-volume`, `--ngram-residency`,
`--ram-headroom-mib`, `--vram-headroom-mib`, `--vram-past-budget`, `--expert-tail-skip`, the prefill
group (`--prefill-chunk`, `--prefill-chunk-beside-decode`, `--prefill-split-workspace-mib`),
`--rope-yarn-factor`, `--draft-model`, `--spec-min-p` and
`--route-trace`; `--draft-tree-nodes`/`--draft-tree-paths` exist engine-wide, but this family
refuses them (§11). There is no flag for the cache policy (LFRU only), the expert frame count (the cache
fills the VRAM the plan leaves, §9) or the CPU worker count (`ProgramOptions::cpu_expert_workers`,
default 6).

Production (RTX PRO 6000, `~/src/super-quant/configs/Qwen3.8-Flash-Next/model.env`) serves recipe C
with `--kv-dtype fp8 --max-context 262144 --max-concurrency 6 --ngram-volume <BF16 volume>`,
`--spec dflash2 --draft-model <companion> --draft-tokens auto --lm-head-draft`, `--host-context-mib
8192` and `--vision`; every expert sits in VRAM. The user guide's example uses `--kv-dtype int8`
and a 16,384-token context. For reproducible benchmarking: `--expert-state off` (cold cache,
`infernix_bench` always starts that
way), `--no-prefix-reuse`, and `--seed`/`--greedy` for the sampler.

### 18.4 Ownership

| Component | Owner |
|---|---|
| Probe suite and inventory (`tools/flash_next_probe`) | Maintainer tools, outside the product |
| Startup measurements, rescaled costs, compiled-in constants | The Qwen4Exp Program (`src/models/qwen4_exp`) |
| Cache-policy and host-tier replay (`tools/expert_cache_replay`) | Maintainer tools |
| Serve options, `--help`, `--config` | `src/serve` |
| In-process options | `EngineOptions` (`include/infernix/types.h`) |

## 19. Sources

External works the sections of this document cite; claims taken from them were re-measured on this
model before they decided anything.

- **Strata** — https://github.com/Niko1221/Strata. The reference single-GPU implementation studied
  for mechanisms: expert offload and cache, the MTP transcription, the n-gram PLE reader.
- **ninfer-ext** — https://github.com/giveen/ninfer-ext. The fork this engine's Qwen4Exp work
  started from; its `docs/maintainer/qwen4-exp-model.md` is the model-mathematics reading aid.
- **FreeToken** — https://github.com/FlashML-org/FreeToken (offload / hybrid / disk-PLE
  strategies); community RTX 5090 measurements and the published routing traces used for cache
  replay: https://github.com/hz1ulqu01gmnZH4/qwen38-freetoken.
- **Model sources** — https://huggingface.co/Qwen/Qwen3.8-Flash-Next,
  https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4,
  https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored-NVFP4 (the production artifact's
  checkpoint), https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2 (the drafter).
- **Other implementations** — SGLang `python/sglang/srt/models/qwen4_exp*.py`; vLLM
  `vllm/models/qwen4_exp/`; llama.cpp `src/models/qwen4exp.cpp` and its per-layer expert LRU
  (https://github.com/ggml-org/llama.cpp/pull/27861); KTransformers
  https://github.com/kvcache-ai/ktransformers/pull/2209.
- **Quantizer semantics** — TensorRT Model Optimizer's NVFP4 quantizer and TensorRT-LLM's FP4
  quantization kernel; §17 follows their published rule.
- **sm_120 kernel measurements** — https://github.com/Theodore-Liu/sm120-fp4 (boundary costs and
  streaming rates of the micro-floor report); FlashInfer's SM12x kernels; b12x
  (https://github.com/yatesdr/b12x, MoE execution models and stage-2 baselines).
- **MoE offload and cache papers** — arXiv 2402.07033 (Fiddler), 2401.14361 (MoE-Infinity),
  2410.22134 (ProMoE), 2411.01433 (HOBBIT), 2408.10284 (AdapMoE), 2501.10375 (DAOP), 2502.05370
  (fMoE), 2504.05897 (HybriMoE), 2601.07372 (Engram), 2608.07911 (moe-cache-eval), 2509.02408
  (cache management for MoE LLMs), 2609.12978 (SeqMoE).
- **CUDA** — the CUDA programming guide on programmatic dependent launch (PDL; `src/core/pdl.cuh`).
- **Megakernels** (the rejected persistent-decode alternative, §8) — Hazy Research, "Look Ma, No Bubbles!"
  (2025), https://github.com/HazyResearch/Megakernels.
