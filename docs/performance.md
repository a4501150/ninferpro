# Serving performance

## Qwen3.8-Flash-Next on RTX PRO 6000

ninferpro's own measurements are the README's [Measured](../README.md#measured) table: one RTX PRO
6000 Blackwell, production's server flags (DFlash2, n-gram volume, FP8 KV, 6 lanes,
262,144-token context) and a placement matrix switched at runtime in one server boot. Reproduce it
with production stopped, since the run needs the whole GPU:

```bash
make t3 PLACEMENTS='vram=100%;vram=65%,ram=15%,ssd=20%;vram=50%,ram=50%;vram=50%,ram=0%,ssd=50%;vram=100%'
```

`make t3` runs `~/src/super-quant/scripts/bench_infernix.py` against `build/apps/infernix-serve` and
writes `<out>.md`, `<out>.json` and the server log under `~/src/super-quant/results/`. Every speed
is read from the server's own request log (`req#N done | ...`), not from client timing. For each
placement the script runs the workload once to warm up and once to measure:

| Column | Workload |
|---|---|
| Decode, 1 request | one short chat prompt, 256 output tokens; decode tok/s |
| Decode at 30K context | one prompt of about 29,000 tokens, 256 output tokens; decode tok/s |
| Prefill, 30K | a new prompt of about 29,000 tokens with a random prefix, so the prefix cache cannot absorb it; prefill tok/s |
| 6 requests, total | six concurrent short prompts, 256 output tokens each; output tokens over the wall time from the first start to the last finish |

The script marks a row as disturbed when the measured short decode is slower than the warm one at
equal acceptance; the GPU also drives the desktop, so such a row is not comparable. The closing
`vram=100%` row repeats the first one after every switch and shows whether switching left the engine
slower.

The tiers below `make t3` (Forward time, the in-process engine bench) and their scopes are in
[AGENTS.md](../AGENTS.md#iteration-tiers). Their numbers are not comparable with this table.

## Inherited Infernix measurements

The pages below come from Infernix and were measured there on one NVIDIA GeForce RTX 5090 through
its public HTTP serving route. ninferpro has not re-measured them; they describe Infernix at the
fork point, not this engine. Read the [measurement and publication rules](performance/methodology.md)
for workload definitions, metric formulas, statistics, comparison requirements, and the standard
result-page format. "Not published" describes measurement coverage, not product support. C is
configured request concurrency; K is the number of draft tokens.

| Model / weights | MTP0 context profile | Single-request speculative decode | Corpus makespan | MTP3 decode saturation |
|---|---|---|---|---|
| Qwen3.6-27B / `groupwise-int` | [8K–256K](performance/qwen3.6-27b.md#no-speculation-context-profile) | [MTP3](performance/qwen3.6-27b.md#single-request-speculative-decode) | Not published | [C=1, 2, 4, 8](performance/qwen3.6-27b.md#decode-saturation) |
| Qwen3.6-27B / `nvfp4` | [8K–256K](performance/qwen3.6-27b.md#no-speculation-context-profile) | [MTP3](performance/qwen3.6-27b.md#single-request-speculative-decode) | Not published | [C=1, 2, 4, 8](performance/qwen3.6-27b.md#decode-saturation) |
| Qwen3.6-35B-A3B / `groupwise-int` | [8K–256K](performance/qwen3.6-35b-a3b.md#no-speculation-context-profile) | [MTP3; DFlash K=7 stochastic/greedy](performance/qwen3.6-35b-a3b.md#single-request-speculative-decode) | [MTP3 C=1, 2, 4, 8; DFlash C=1](performance/qwen3.6-35b-a3b.md#corpus-makespan) | [C=1, 2, 4, 8](performance/qwen3.6-35b-a3b.md#decode-saturation) |
| Qwen3.8-27B / `groupwise-int` | [8K–256K](performance/qwen3.8-27b.md#no-speculation-context-profile) | [MTP3; DFlash2 K=7](performance/qwen3.8-27b.md#single-request-speculative-decode) | [MTP3 C=1, 2, 4, 8; DFlash2 C=1](performance/qwen3.8-27b.md#corpus-makespan) | [C=1, 2, 4, 8](performance/qwen3.8-27b.md#decode-saturation) |
| Qwen3.8-27B / `nvfp4` | [8K–256K](performance/qwen3.8-27b.md#no-speculation-context-profile) | [MTP3; DFlash2 K=7](performance/qwen3.8-27b.md#single-request-speculative-decode) | [MTP3 C=1, 2, 4, 8; DFlash2 C=1](performance/qwen3.8-27b.md#corpus-makespan) | [C=1, 2, 4, 8](performance/qwen3.8-27b.md#decode-saturation) |

The RTX 5090 figures in the [Flash-Next guide](qwen3_8-flash-next.md#performance) are also
Infernix measurements.

Qwen3.8 and Qwen3.6-35B-A3B C=1 corpus points also supply their single-request phase tables.
The Qwen3.6-27B NVFP4 MTP3 phase table comes from a corpus C=1 point whose full makespan is
not published here. Qwen3.8 measurements use FP8 E4M3 row-256 KV; the Qwen3.6 measurements
use INT8 group-64 KV. Each model page records its build and run conditions.

### Reading the inherited results

| Question | Metric to use |
|---|---|
| How fast is prompt processing or an individual decode phase? | Prefill phase, Server TTFT, Decode phase |
| How long does the full fixed request set take? | Corpus makespan, Corpus decode, Requests/s |
| What aggregate decode rate is sustained at a full batch? | Steady decode |

These rates use different time boundaries. Server TTFT is an internal phase sum; external
streaming TTFT has its [own benchmark contract](../tools/bench/ttft/README.md). Stochastic runs
can generate different token totals even with the same prompts and seeds. Output-limit and
repetition samples remain labeled in the measured corpus; throughput alone does not establish
successful task completion. See the [35B termination and anomalies](performance/qwen3.6-35b-a3b.md#termination-and-anomalies)
and [Qwen3.8 completion outcomes](performance/qwen3.8-27b.md#completion-outcomes).

## Related references

- [Serving benchmark runners](../tools/bench/README.md#serving-corpus-benchmark): usage and local report files.
- [Engine and Op benchmarks](../bench/README.md): their separate measurement scopes and commands.
- [Capability evaluation](../eval/README.md): evaluation workflow; published scores live in the
  [model cards](README.md#model-artifacts).
- [Perplexity](perplexity.md): offline causal-scoring measurement and comparison rules.

For the inherited models, the model pages are the detailed result authority. For Flash-Next, the
README table is.
