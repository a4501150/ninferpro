# ninferpro documentation

Start with the [project README](../README.md) to build ninferpro, download a published artifact, and
run the CLI or HTTP server.

## User guides

| Document | Purpose |
|---|---|
| [CLI](cli.md) | text, chat-history, image/video input, output streams, sampling, MTP, and common runtime options |
| [HTTP serving](serving.md) | OpenAI Responses/Chat Completions, Anthropic Messages, state, streaming, token counting, authentication, and tool calls |
| [Qwen3.8-Flash-Next](qwen3_8-flash-next.md) | requirements by card size, conversion (recipes A, B and C, the GSQ-RCO IQ3_S GGUF and FP8 imports), recommended flags, expert placement across VRAM, RAM and SSD, the n-gram table, measured speed and quality |
| [Ngram copy proposals](ngram.md) | optional copy acceleration alongside MTP, DFlash or DFlash2 at any concurrency |
| [Performance](performance.md) | how the README's RTX PRO 6000 Flash-Next table is measured; Infernix's inherited RTX 5090 results, methodology, and publication rules |
| [Weight conversion](weight-conversion.md) | official recipes, custom formats and sources, conversion methods, optional components and artifact output |
| [Perplexity](perplexity.md) | fixed-corpus and custom-text causal perplexity, comparison rules, progress, and reports |
| [CLI examples](../examples/cli/) | committed text, multimodal, thinking, long-decode, and long-context inputs |

The executable `--help` output is the exact source for command-line option spelling and defaults.

## Model artifacts

The [project README](../README.md#models) gives the conversion commands for the Qwen3.8-Flash-Next
artifacts used in production. [`model-cards/`](../model-cards/) holds the
versioned source of each repository's model card.

## Repository-local guides

- [Benchmarks](../bench/README.md)
- [Tests](../tests/README.md)
- [Tools](../tools/README.md)
- [Capability evaluation](../eval/README.md)

## Maintainer references

The active references under [`maintainer/`](maintainer/) record current architecture, model,
artifact, and maintenance contracts. These files are not additional user workflows or installed
API documentation.

[Engine architecture](maintainer/engine-architecture.md) is the single top-level reference. The
other references own narrower contracts:

| Document | Responsibility |
|---|---|
| [Engine architecture](maintainer/engine-architecture.md) | model/config/weight ownership, loading-to-execution flow, requests, scheduling, transactions and graphs |
| [Build system](maintainer/build-system.md) | CMake targets, explicit source ownership, CUDA compilation boundaries, presets and developer configuration |
| [Artifact container](maintainer/artifact-container.md) | v3 directory, objects, logical bindings, Uses, resources and file framing/sharding |
| [Numeric formats](maintainer/tensor-formats.md) | represented values, codes/scales, conversion arithmetic and numerical interpretation |
| [Storage layouts](maintainer/storage-layouts.md) | packing, plane offsets, padding, encoded sizes and view addressing |
| [Qwen3.5 model](maintainer/qwen3_5-model.md) | Dense/MoE mathematics, instance config, logical parameters, MTP, Vision and state semantics |
| [Offloaded MoE design](maintainer/moe-offload-design.md) | the shared offload tier every offloaded MoE family composes: expert banks and geometry, VRAM residency with LFRU, the RAM/SSD tiers, placement and the runtime switch, the CPU expert engine, decode and prefill miss service, the performance method, and the checklist for adding a family |
| [DFlash and DFlash2](maintainer/dflash.md) | conditioning, masked draft computation, proposal distributions and backend state |
| [Constrained decoding](maintainer/constrained-decoding.md) | GBNF/JSON/choice/regex, tool policies, masks, speculative transactions and observations |
| [DFlash2 tree verification](maintainer/tree-verification.md) | per-round lattice draft trees, tree acceptance and its exactness, automatic and per-batch-size tree widths, accepted-path compaction |
| [Resource scheduling and context cache](maintainer/resource-scheduling-and-context-cache.md) | core design for incremental resources, admission, preemption and Replay recovery |
| [Hybrid prefix cache](maintainer/hybrid-prefix-cache-spec.md) | the prefix cache: block tree, sparse state snapshots, tap placement, Host slab tier, eviction and automatic configuration |
| [Paged KV context store](maintainer/paged-kv-cache.md) | typed pools, pages, replicas, address spaces, reservations and consumer views |
| [ReplaySSM GDN](maintainer/replayssm-gdn.md) | raw transition records and faithful commitment of the verified state prefix |
| [Op development](maintainer/op-development.md) | semantic boundaries, source ownership, numerical qualification and performance evidence |
| [Model-level numerical quality](maintainer/model-quality.md) | rounding noise at model scale, the 16-bit-activation reference, Infernix-versus-NInfer prefill and decode results, Flash-Next NVFP4 against Unsloth UD-Q4_K_XL |
| [Basalt port assessment](maintainer/basalt-port-assessment.md) | which mechanisms of Basalt (a Strata fork for Flash-Next) are adopted, candidates, parked or rejected, and why |
| [NVFP4 quantization-method adoption](maintainer/nvfp4-quantization-methods.md) | evaluated converter-side 4-bit methods, what each demands of the stored representation, and the measured reason none is adopted |
| [Operational logging](maintainer/logging.md) | log ownership, presentation, severity and data policy |
| [Linear benchmark](maintainer/linear-benchmark.md) | pure Linear measurement, metrics and suites |
| [Linear tuning and reports](maintainer/linear-tuning.md) | tuning ranges, priority points, dispatch tradeoffs and final performance report format |
| [Qwen3.8-Flash-Next design](maintainer/qwen3_8-flash-next-design.md) | current design, as built and in production (status in §1.3): bit-exact import of NVIDIA's NVFP4 checkpoint with exact W4A4 expert arithmetic, routed experts placed in any mix of VRAM, RAM and SSD (the shared tier in [MoE offload design](maintainer/moe-offload-design.md)) with CPU miss service, NVMe n-gram table, QSA over every KV profile (262K, FP8 KV in production), performance model; the exact GGUF IQ3_S and official FP8 imports (§17.3 arithmetic, §12.2 n-gram rows and §12.5 RAM residency, §6.2 routes, §6.6 results and rejected alternatives) |
| [GLM-5.3-Flash design](maintainer/glm5_3-flash-design.md) | design for a planned family (not implemented): the EXL3 3.05 bpw GLM-5.3-Flash builds (MikeRoz's uncensored, turboderp's base) on the shared offload tier on the RTX PRO 6000 and RTX 5090, with exact E3A16 experts in VRAM, pinned RAM and expert volumes striped over up to four NVMe drives; sparse MLA, pooled indexer, KDA and mHC Ops; measured storage topology, performance model and milestones |
| [DeepSeek-V4.1-Flash design](maintainer/deepseek_v4_1-flash-design.md) | design for a planned family (not implemented): the MXFP4 build of DeepSeek-V4.1-Flash-1017 on the RTX PRO 6000 and RTX 5090, with its experts stored as the lossless original MXFP4 and computed with DeepSeek's exact 8-bit activations (MX8) from any mix of VRAM, RAM and NVMe (three striped volumes on the 5090 machine, the models NVMe or an expert stripe on the 6000) placed by the shared offload tier; CSA2 attention and the two-level indexer, Engram volumes in FP8 or MXFP4 rows, single-pass mHC, the bounded-replay and full-model prompt rules, DSpark; a streaming conversion that fits the free disk; DwarfStar and other engines, performance model and milestones |

Model cards contain official artifact facts and source provenance. The
[conversion guide](weight-conversion.md) is the entry point for making an artifact. Exact config
fields, parameter expansion and native supported domains are maintained by the code linked from
these references.
