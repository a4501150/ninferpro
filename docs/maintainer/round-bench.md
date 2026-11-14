# Decode profiling tiers: the per-family contract

The iteration tiers (AGENTS.md "Iteration tiers") run the same targets for every model family:
`make <target> FAMILY=<family>`. The Makefile and `tools/tiers/` are family-neutral; a family
provides two benches and one settings file, all copied from the closest existing family and then
specialized (the bench binds that family's Forward and state directly, like its Program does). This
page is the contract those pieces meet, the speed-of-light model the round bench reports, and the
order to use the tiers in when bringing up or tuning a family's decode.

## What a family provides

| Piece | Path | Used by |
|---|---|---|
| Settings | `bench/models/<family>/tiers.mk` | the Makefile: `ARTIFACT` and the family's other artifacts, `MODEL_ENV` (the environment its real-artifact tests and Engine bench read), `ROUND_ARGS` (the round bench's artifact arguments), `T1_FAST` and `T1_EXACT` (the golden shapes), `SWEEP_SHAPE`, `SWEEP_CONTEXTS`, `SWEEP_CHUNK` |
| Round bench | `infernix_<family>_round_bench`, `bench/models/<family>/round_bench.cpp` | `t1`, `t1-exact`, `t1-record`, `t1-kernels`, `t1-sweep` |
| Golden hashes | `bench/models/<family>/round_hashes.txt` | `t1`, `t1-exact`, `t1-record` |
| Smoke test | `infernix_<family>_smoke_real_test`, golden tokens in `tests/models/<family>/smoke_golden.txt` | `t2`, `t2-record` |
| Engine bench | `infernix_<family>_engine_bench` | `t2-bench`, `t2-step` |

`qwen4_exp` meets the whole contract. `qwen3_5`'s MTP and DFlash round benches predate it: they
drive the Program rather than the Forward and implement none of the options below, so the tier
targets do not run for `qwen3_5`.

## The round bench

One generation round of each requested shape through the production Forward and its CUDA Graph,
every routed expert resident, without the Engine. Options every family's round bench implements:

| Option | Meaning |
|---|---|
| `--shape RxW[+P][,...]` | R rows of W columns: W = 1 a plain decode round, W >= 2 a verification round and its commit; `+P` adds a P-token prefill chunk of the last lane as one mixed call |
| `--context N[xK][,N[xK]...]` | each lane's tokens before the round, in lane order, K lanes per value, the last value filling the rest; one value for every lane (default 1024) |
| `--prefill-chunk N` | the prefill calls that build each lane's context (part of the hash key when not the default) |
| `--replays N`, `--accept K`, `--kv FORMAT` | timed replays; committed columns per row; KV storage |
| `--exact` | the exact route (`--disable-fast-paths`) |
| `--check FILE`, `--record FILE` | compare with, or record, the golden hashes of the logits and the lane state |
| `--determinism N` | replay each shape N times from one state; differing bits fail the run |
| `--sections` | each section (embedding, every layer's mixer and MoE halves, the head) as its own graph |
| `--profile` | bracket the timed replays with `cudaProfilerStart`/`Stop` (`t1-kernels`) |
| `--report FILE` | append one JSON line per shape (below) |
| `--dram-gbs X` | use X GB/s instead of the measured device read rate |

Its output lines start with `device read rate`, `shape`, `  forward graph`, `  commit`, `  round`,
`  bytes`, `  speed of light` and `  hash`; `tools/tiers/kernels.py` passes these through.

### Report lines

Every shape appends one JSON object: `bench` (the family), `shape`, `rows`, `width`, `prefill`,
`contexts` (the round's rows), `kv`, `route` (`fast` or `exact`), `accept`, and then

- a plain shape: `forward_us`, `commit_us`, `round_us` (medians), `bytes` {`dense`, `experts`,
  `kv`, `state`, `total`}, `distinct_experts` (per layer, mean), `read_gbs`, `ideal_us`;
- a mixed shape: `round_us`, `chunk_us` (the two separate calls) and `mixed_us`.

`tools/tiers/sweep.py` reads only these fields.

### The speed of light

The bytes a round must move at least, over the device read rate measured at startup
(`measure_device_read_bytes_per_second`, `src/core/dram_probe.h`: a 1 GiB streaming read of
distinct words, the best of ten):

- dense: every dense weight the Forward reads, each stored parent once (projections, norms,
  convolutions, routers, the shared experts, the final mixer and the head; not the host-resident
  token embedding, not the drafters);
- experts: the record of every distinct routed expert, counted from one eager run's routes
  (`ForwardTap::routes`) from the state the timed replays start from;
- KV: per row, the K/V of the tokens its attention reads at least, and whatever per-token index
  its selection scans over the whole context (for QSA, the selected tokens and the pooled index keys);
- state: per row, the recurrent state read and written once.

The figure leaves out activations, logits and scratch. It is a lower bound on bytes, not a
prediction: streaming kernels top out near 90 % of the read rate, so a round at 1.1x its speed of
light is close to done, and the ratio says how much a change of kernels, fusion or overlap can
still save. A family's byte model is its bench's `round_bytes` and `dense_weight_bytes`; its design
document's hand budget should agree with it.

## The Engine bench

The public Engine in process with production's options: C concurrent requests of N greedy output
tokens, one warm-up set, then `--repeat` sets (`--concurrency`, `--tokens`, `--repeat`,
`--fast-path`). `--profile` brackets the measured sets with `cudaProfilerStart`/`Stop`.

`t2-step` runs it under Nsight Systems through `tools/tiers/step.py`, which needs only the ranges
every Program already emits through the runtime: the Engine's `decode` range per round and the
runtime's `program.submit`, `device.wait` and `program.post`. It splits each round's window (one
`decode` start to the next) into host phases, GPU idle time by what the Engine thread was doing,
and GPU busy time by launch site (the NVTX path open where each kernel, copy or graph was
launched), so the target graph and a drafter's calls separate by the ranges the family emits. A
family that wants a finer split names its phases with `nvtx::ScopedRange`.

## Order of work for a family's decode

1. `make t1`: the round time and its speed of light per golden shape. Compare the byte classes with
   the design's budget; a class larger than designed is a binding or layout question, not a kernel
   one.
2. `make t1-sweep`: the same against context length and skewed lanes (one long lane beside short
   ones). The `vs first` column is the cost of context; a class that grows with context and a
   ratio that grows with it point at the attention or the selection.
3. `make t1-kernels KSHAPE=<worst ratio> METRICS=1`: which kernels hold the time and where DRAM
   bandwidth goes unused; then Nsight Compute on one kernel (bench/README.md).
4. `make t2-step`: what T1 leaves out. Host phases and GPU idle per round bound what host work,
   scheduling and launch overlap can save; drafter sites against the target graph price the
   drafting.
5. `make t3`, then `make t3-frontier` (production-shaped agentic traffic from `bench/agentic_ab` at
   each concurrency level in one boot: total throughput against per-user decode speed, the best
   point at an SLA) for policy that depends on live timing; live counters for the same questions
   are on `/metrics` (docs/serving.md).
