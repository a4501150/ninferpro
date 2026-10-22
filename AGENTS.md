# AGENTS.md

These rules apply to the whole repository. `CLAUDE.md` imports this file; keep the rules here.

## This repository

ninferpro is the local inference engine for one machine and one owner. It starts as a single
commit from Infernix commit `7dbe99a` ([Wallawalla47/Infernix](https://github.com/Wallawalla47/Infernix));
`NOTICE` records the origin and the changes made here. Keep `LICENSE` and `NOTICE`, and update
`NOTICE` when a change is large enough to list.

- `main` is the only long-lived branch and the one production builds from. Feature work goes on
  short `feat/*` or `fix/*` branches (worktrees under `~/src/ninferpro-wt/` when parallel), lands
  on `main` by fast-forward, and is pushed to `origin`.
- There is no shared history with Infernix. To take an upstream change, diff it against `7dbe99a`
  (or the upstream commit last ported, recorded in the commit message) and port it as its own
  commit. Drop a local change when upstream ships an equivalent.
- What this repository adds to Infernix:
  - three-tier expert placement for Qwen3.8-Flash-Next: every routed expert lives in exactly one of
    VRAM, RAM or SSD (`--experts vram=N%,ram=M%[,ssd=K%]` or counts; `--expert-placement FILE`),
    switched at runtime through `POST /admin/expert-placement` with `GET` reporting counts, bytes,
    KV, free memory and the last switch's outcome; within the counts the engine swaps hot experts
    in and cold ones out; the frame limit covers every expert;
  - `--served-model-name` aliases;
  - the prefix-cache tail snapshot for single-chunk prompts on artifacts without free cache resume;
  - batched DFlash2 drafting at any round size with an online cost fit;
  - no spill detection without a WDDM budget (Linux allocations are VRAM-or-fail).

### Mechanism and policy

The engine provides mechanism; clients own policy. The engine places experts exactly where a
placement says, moves only the experts whose tier changes, reports what it did, and refuses or
reports a placement it cannot reach. It does not choose shares, react to other programs' memory
use, or keep headroom heuristics for the placement. Policies such as "leave 32 GB of VRAM and 24 GB
of RAM free" live in the client (`~/src/super-quant/scripts/infernix_placement.py`,
`make infernix-gaming`, `make infernix-full`). Which expert sits in which tier inside the client's
counts is mechanism and stays in the engine.

## Objective and scope

Complete the user's explicit deliverable within the applicable product and external contracts.
Choose a coherent solution with functional and numerical correctness, clear ownership, strong
architecture, and maximum performance at the requested scope. Do not sacrifice these goals to
reduce the diff or implementation effort. Evaluate complexity, maintenance cost, and verification
risk as engineering tradeoffs, not reasons to retain a known inferior design.

Before substantial work, identify the deliverable and its completion conditions. Work is relevant
when it completes that deliverable, preserves an applicable contract, resolves a material
uncertainty, or checks a realistic regression. A necessary redesign is in scope; unrelated cleanup,
hardening, compatibility, and benchmark campaigns are not. Address incidental findings when they
block the outcome or are inseparable from the selected implementation.

For implementation tasks, record the user's requirements, promised work, affected implementations
and routes, and the quality dimensions to be evaluated before judging results. For comparative
claims, also identify the baseline, workloads, metrics, aggregation method, and acceptance criteria.
Keep this accounting current as the task develops. Explain justified changes and preserve the
earlier findings; never narrow the scope or change the criteria after seeing unfavorable results
to make the task appear successful. This accounting may live in the working conversation or
existing task artifacts; it does not require a new permanent planning document.

For analysis or design, deliver the explanation or design. For diagnosis, establish the cause and
supporting evidence; implement a fix when requested. For implementation, complete the selected
design across its affected implementations, callers, tests, tools, and active documentation.

Requests for analysis or proposals for review do not authorize applying them;
implementation authorization must cover that same scope.

The current product and architecture govern ordinary work. An explicit task may change them;
update the affected contracts and implementation together instead of treating the current design
as an immutable prohibition. Skills provide task-specific methods, not additional deliverables or
approval requirements beyond the user's instructions and the actual execution environment.

## Product and architecture

The engine (Infernix's, built here as `infernix-*` binaries) is a from-scratch C++/CUDA inference
engine for maximum single-GPU performance. It implements
`Qwen3_5ForCausalLM` and `Qwen3_5MoeForCausalLM`; official Qwen3.6/3.8 artifacts and user recipes
use the same architecture, binding and execution path. It also implements `Qwen4ExpForCausalLM`
(Qwen3.8-Flash-Next, model family `qwen4_exp`), whose routed experts are offloaded to host memory or
the SSD. The implementation targets `sm_120a`. Infernix tunes on an RTX 5090 (32 GB, Windows); this
repository runs and tunes on one RTX PRO 6000 Blackwell (96 GB, PCIe Gen5 x16) under Linux, where every
expert fits in VRAM and placement decides how much to give back to other programs. Changes must keep
the smaller-card path working even though it is not measured here.

Generation uses one GPU, one resident model, one to eight resident execution lanes fixed at startup,
bounded FIFO ingress with finite bypass, resource-pressure preemption with Replay recovery,
one content-addressed prefix cache, and one compact decode batch per round.
Generation and offline CausalScoring use the same public `.infernix` Engine route. Delivered
capabilities and commands are documented in `README.md`, the product guides, and executable
`--help`. New mathematical architectures, execution platforms, large-scale continuous
batching, and priority/QoS require an explicit product change. Another training instance or mixture
of existing representations does not require a checkpoint-specific execution registration.

This is a local, single-owner project with trusted local models, generated artifacts, and
local workflow. Do not derive requirements from a different deployment or trust model.

Keep these ownership boundaries visible when selecting a design:

- The v3 container (`.infernix`, or `.ninfer` from NInfer) is the only C++ product artifact; CLI,
  serving, and inference benchmarks use the public Engine. Infernix has no Python model-inference
  route or installed/exported C++ SDK.
- Core owns physical primitives and raw transfers; artifact owns generic framing and
  materialization; Ops own closed mathematical and state-transition implementations.
- Models own fixed mathematics, config interpretation, logical parameter binding, frontend
  semantics and finite execution composition. Immutable Model data owns selected weights and
  resources; native Parameters supply the actual operands to planning and Program execution.
  Program owns mutable state, workspace, context stores and CUDA Graphs. Programs share no mutable
  state or device allocation.
- Converter recipes choose sources, formats, packing and per-input activation permissions. The
  loader validates, uploads and binds the stored representation. Native preparation, resource
  queries and execution enforce actual Op support; there is no whole-artifact capability registry.
- Runtime owns common execution contracts and Engine publication policy; product/serving own input
  acquisition and protocol translation. Model code does not acquire media or own transport.

Qwen3.8-Flash-Next is the model served today, and the structure must stay easy to extend to other
models at zero runtime cost. A new model family gets its own directory under `src/models/<family>/`,
started by copying the closest existing family and then specialized: its own mathematics, binding,
Program, kernels and tuning. Duplicated code between families is acceptable and preferred over
shared abstractions that add indirection, branching or virtual dispatch on the hot path. Share code
only where the shared piece is already a zero-cost primitive (core, ops, runtime contracts), and
never make one family's fast path slower to accommodate another.

A planned product change: load Hugging Face safetensors (BF16, FP8, ModelOpt NVFP4, AWQ/GPTQ) and
GGUF (Q4_K, Q8_0 and similar block formats) directly, next to `.infernix`. Until it lands, design
for it: kernels take the weight format as a compile-time trait (tile load, scale handling, MMA
variant) around shared grouping and scheduling, so a new stored format adds a trait, not a new
kernel. A future loader maps each source format either to an existing trait or, once at load time,
to a native layout; it never repacks per request.

Detailed model/runtime responsibilities and source ownership are defined in
[Engine architecture](docs/maintainer/engine-architecture.md). Read the relevant boundary before
changing it. Prefer explicit implementations for supported architectures. Do not introduce generic model
graphs, family base classes, plugin discovery, string-driven execution, hidden device allocation,
runtime weight repacking, or placeholders for hypothetical targets without a product requirement.

## Change consistency

Project-owned APIs, CLIs, Python tools, fixtures, reports, formats, and documentation do not preserve
backward compatibility. When replacing behavior, remove superseded aliases, fallbacks, transition
branches, and their tests within the affected contract. Leave unrelated paths alone.

Advertised OpenAI and Anthropic protocol behavior is an external contract. Changes update the
affected schema tests and serving documentation together.

Keep stable requirements in their existing active reference. Temporary plans are useful only for
active work; remove them when completed or abandoned. Maintain one current authority rather than
parallel `final`, `v2`, or `new-design` documents.

## Verification and completion

Select evidence to support the changed behavior and material claims. Tests should protect supported
observable behavior, mathematical or state semantics, and realistic regressions, including plausible
boundary failures that have not occurred yet. Avoid tests that merely mirror implementation,
freeze private file/class organization, or increase coverage numbers.

For numerical changes, identify represented public inputs, the independent mathematical oracle,
semantic cast/quantization/state boundaries, output criteria, and relevant real model shapes. Each
floating-point Op uses a naive FP32/FP64 oracle; exact transforms/codecs use an exact oracle. Packed
inputs are independently decoded with their stored scales. Qualify production routes directly
against that oracle, not another kernel or plausible model output. Private arithmetic need not
reproduce unfused materializations unless an intermediate is an observable semantic boundary.
[Op development](docs/maintainer/op-development.md) defines the full qualification contract.
The exact route (`--disable-fast-paths`) stays bit-exact from release to release and, where the
artifact's formats allow, request-invariant (other requests, concurrency, cache state and resource
pressure do not change a request's output); a change that alters it for speed ships as a fast
path, qualified under the
[numerical acceptance policy](docs/maintainer/model-quality.md#6-numerical-acceptance-policy-and-fast-paths)
for the kernels it changes (a configuration is the set of kernels an artifact runs, so artifacts
with the same kernels share a qualification) and registered only where it is measured faster.

Measure performance at the claimed scope. An Op microbenchmark establishes an Op result, not an
end-to-end improvement. Use whole-inference profiling when an in-scope end-to-end attribution is
unresolved; use kernel profiling when an identified kernel question can change the decision. Reuse
applicable evidence and stop collecting once the relevant alternatives can be distinguished.

Choose the affected checks, rather than running this table as a checklist:

| Change | Typical evidence |
|---|---|
| Documentation | affected links/references and `git diff --check` |
| C++ runtime/API | affected build targets and behavioral tests |
| Python tooling | Python 3.11 `py_compile` and affected tests |
| Artifact framing/binding/conversion | affected contract tests; real artifact when semantics require it |
| CUDA mathematics | independent oracle at relevant shapes and route boundaries |
| Memory or lifetime | affected execution; sanitizer for a concrete lifetime question |
| Performance | measurement at the claimed scope; profiling only for unresolved attribution |
| Serving | affected schema tests and observable request/stream behavior |

Record the target, relevant hardware/toolchain, workload or command, and summarized result needed
to interpret a material claim. Hashes, clean worktrees, full command transcripts, raw report
inventories, and exact probabilistic outputs are not default requirements. Use exact comparison for
exact outputs, and appropriate numerical or behavioral criteria otherwise. State checks that could
not run and their implications.

### Iteration tiers

Iterate at the lowest tier that can answer the question; a full server boot is for promotion only.
Each tier is one command from the repository root (`Makefile`); every target takes the build lock,
the memory cap and the GPU lock itself.

| Tier | Command | What it establishes | Time |
|---|---|---|---|
| T0 | `make t0 [R=regex]` | unit, Op and contract tests: CTest without the `real` label | minutes |
| T1 | `make t1 [SHAPE=6x1,6x2]`, `make t1-exact`, `make t1-kernels [KSHAPE=6x2]` | one generation round of each shape on the real artifact through the production Forward and its CUDA Graph (`bench/models/qwen4_exp/round_bench.cpp`): device time, commit time, `--sections` per layer half, per-kernel table, and the logits and lane-state hashes against `bench/models/qwen4_exp/round_hashes.txt`; every shape also replays 2 or more times from one state and fails on differing bits (`--determinism N`) | about 10 s |
| T2 | `make t2`, `make t2-record` | start-up and correctness smoke test (`tests/models/qwen4_exp/test_smoke_real.cpp`): the public Engine in process with production's options on the real artifacts (six concurrent requests, DFlash2 drafting, prefix-cache resume, an expert-placement switch and back), then the exact route's greedy tokens against `tests/models/qwen4_exp/smoke_golden.txt`, alone, six in flight, resumed and after the switch | about a minute |
| T3 | `make t3 ARGS=...` | promotion: `bench_infernix.py` and the quality gate against the server build; the only tier for policy behaviour that depends on live timing (the DFlash2 cost fit, placement swaps, scheduling) | many minutes |

- Kernel and Forward work: T1 for every change. A change meant to keep the bits must pass `make t1`
  and `make t1-exact` unchanged; an intended numerical change records new hashes with
  `make t1-record` in the same commit, and states why they changed. T1 leaves out the DFlash2
  drafter, acceptance, sampling and host work; claim only Forward and commit time from it.
- Op microbenchmarks (`build/bench/infernix_<op>_bench`) stay the tool for one Op's alternatives.
- T1 and T2 place every expert in VRAM, so they need the GPU production uses: stop production first.
  A T1 nondeterminism is a bug to fix, not noise; `--determinism N`, `--no-fork` and
  `--no-shared-stream` narrow which overlap it needs.
- T3 runs once a change is complete and its T0-T2 checks pass, before production is restarted on it.

## Reporting and completion

Selective reporting and evidence gaming are prohibited, even when every disclosed
statement is individually true. For every implementation task:

1. Cover the entire agreed deliverable, its completion status, and all affected or
   evaluated dimensions: behavior, numerical semantics, interfaces, architecture,
   performance, resources, and maintenance. Distinguish completed, incomplete, and
   unverified work; never describe an unmeasured aspect as unchanged.

2. Put favorable and unfavorable findings in the final reply itself, including
   regressions, costs, rejected approaches, failures subsequently fixed, unresolved
   issues, and verification gaps. Explain their disposition. Group repetition
   without hiding distinct problems or exceptions. Small or unexplained adverse
   results must remain visible; attachments cannot substitute for disclosure.

3. Make comparisons representative and comparable. State the baseline, workload,
   conditions, metrics, coverage, outcome distribution, worst changes, and exceptions.
   Distinguish new capability, fallback replacement, and improvement to an optimized
   implementation. Keep claims within the measured scope; neither a best case nor
   an average may stand in for the full results.

4. Apply the same evidence standard to gains and regressions. Label uncertainty;
   do not dismiss slowdowns as noise without evidence. Explain changes to scope,
   baselines, methods, or acceptance criteria and preserve earlier adverse findings.
   Never change these choices to manufacture a favorable conclusion.

5. Reuse sufficient evidence. Additional or repeated checks must satisfy required
   verification, replace invalidated evidence, or resolve a concrete question that
   could change implementation or acceptance. Once the deliverable and acceptance
   conditions are satisfied, stop and report. Report review checks existing work
   and findings; it must not become a new audit, sweep, or reporting-tool project.
   Disclose remaining uncertainty without silently making it a new requirement.
   Disclosure does not excuse unmet completion conditions.

## Reference navigation

Read the authority relevant to the current decision; this is not a mandatory reading list.

| Decision | Entry point |
|---|---|
| Product capabilities and exact commands | `README.md`, executable `--help`; `docs/cli.md`, `docs/serving.md`, `docs/perplexity.md` |
| Execution, model/runtime ownership, scheduling, transactions, graphs | `docs/maintainer/engine-architecture.md` |
| Scheduling, permits, preemption; prefix cache; physical KV | `docs/maintainer/resource-scheduling-and-context-cache.md`; `docs/maintainer/hybrid-prefix-cache-spec.md`; `docs/maintainer/paged-kv-cache.md` |
| Artifact, layout, codec, conversion, or model mathematics | model/artifact references and conversion guide linked from `docs/README.md` |
| Op contracts, implementation ownership, numerical/performance qualification | `docs/maintainer/op-development.md` |
| Test/benchmark commands and published performance | `tests/README.md`, `bench/README.md`, `docs/performance.md` |
| In-tree C++ interface | `include/infernix/engine.h`, `include/infernix/types.h` |

[Documentation map](docs/README.md) routes to narrower authorities when needed.

## Local operations

The machine: Ubuntu 26.04, kernel 7.0, one RTX PRO 6000 Blackwell (SM120, 96 GB), 89 GB of RAM,
CUDA 13.4 at `/usr/local/cuda-13.4`, a Gen5 NVMe (T710) holding the models.

- Builds: one `build/` directory, configured with the `dev` preset (the optimized release build
  plus tests and benchmarks), holds the production server and the tests. Export
  `CPATH=/usr/local/cuda-13.4/include` for the plain C++ test targets. Build only what the task
  needs (`--target infernix-serve`, or the affected tests). Wrap builds as
  `flock ~/.cache/super-quant/build.lock systemd-run --user --scope -p MemoryMax=24G cmake --build
  build -j12 ...` (the `Makefile` targets do, `JOBS=` and `MEM=` override): an unbounded nvcc build
  has run the host out of memory. A build peaks near 1.75 GiB per job; use `-j8` at 16G when
  production holds experts in host RAM.
  `build/` links with mold and compiles through ccache (`CMAKE_LINKER_TYPE=MOLD`,
  `CMAKE_{CXX,CUDA}_COMPILER_LAUNCHER=ccache`, set in its cache, not in the presets); a header
  that every CUDA file includes (`core/pdl.cuh`, `ops/common/*.cuh`) still recompiles about 180 of
  them (about 8 minutes at `-j8`). Its Python tests run with
  `.venv-convert/bin/python` (`Python3_EXECUTABLE`; it holds torch and jsonschema, the system
  interpreter has no torch).
- GPU: one model boot fits at a time. Hold `flock ~/.cache/super-quant/gpu.lock` for any boot, and
  the build lock as well while measuring. Production stops for a test boot and must be restarted
  afterwards.
- Production: `make -C ~/src/super-quant serve-infernix` / `stop-infernix` (detached with
  `setsid nohup`; pidfile `~/src/super-quant/.infernix.pid`, log `.infernix.log`). Its flags live in
  `~/src/super-quant/configs/Qwen3.8-Flash-Next/model.env`. Test servers use ports 8400-8499;
  port 8000 is production.
- Artifacts: `~/models/infernix/` (`README.local.md` records the conversion commands). Select them
  by explicit path, never glob order, modification time, or unqualified "latest". Source
  checkpoints live in `~/.cache/huggingface`.
- Processes: track PIDs. Never use `pkill -f` or `pgrep -f` (the pattern matches the calling shell),
  and every script that starts a server kills it before waiting on it.
- Python: the converter uses Python 3.11 from `.venv-convert/`; the system `python3` is 3.14.
  Install or upgrade dependencies only when the task needs it.

Benchmark with `~/src/super-quant/scripts/bench_infernix.py`: one boot per binary, placements
switched at runtime, concurrent requests from a thread pool, short polls instead of fixed sleeps,
unique prompt prefixes so the prefix cache does not hide prefill, and every speed read from the
server's `req#N done` log lines rather than client timing. Report the per-tier expert counts and the
process VRAM (experts / KV / other) with every speed table.

Commit and push to `origin` as work completes; do not wait to be asked. Use Conventional Commit
subjects with concise lowercase types such as `feat`, `fix`, `perf`, `bench`, `test`, `build`,
`refactor`, `docs`, or `chore`. Keep scratch notes (handoffs, plans) out of commits.
