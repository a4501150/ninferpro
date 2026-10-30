# ninferpro

A C++/CUDA inference engine for Qwen3.8-Flash-Next on one large Blackwell GPU, with control over
where each routed expert lives: VRAM, RAM or SSD. It is derived from
[Infernix](https://github.com/Wallawalla47/Infernix), which grew out of
[NInfer](https://github.com/Neroued/ninfer); see [Credits](#credits).

It runs on one RTX PRO 6000 Blackwell (96 GB, SM120, PCIe Gen5 x16) under Linux. On this card the
whole model fits in VRAM, and the engine can give part of that memory back to other programs (a game,
another model) at runtime, without a restart.

The binaries keep their Infernix names (`infernix-serve`, `infernix`, `infernix-perplexity`).

## What it adds to Infernix

- **Three-tier expert placement.** Each of Flash-Next's 24,576 routed experts lives in exactly one
  of VRAM, RAM or SSD. A placement is a set of shares that add up to 100 %
  (`--experts vram=65%,ram=15%,ssd=20%`), counts, or a per-expert file. At `vram=100%` every
  expert is in VRAM and the process holds no expert bytes in RAM.
- **Runtime switching.** `POST /admin/expert-placement` applies a new placement between decode
  rounds while requests keep running. Only the experts whose tier changes move, and freed VRAM and
  RAM go back to the system.
- **Expert swapping.** Within the counts a placement names, the engine moves hot experts into VRAM
  and cold ones out, following the workload.
- **Mechanism only.** The engine places what it is told and reports what it did. Policy, such as
  "leave 32 GB of VRAM free for a game", is client code (see [Expert placement](#expert-placement)).
- **A working prefix cache** for single-chunk prompts on artifacts without free cache resume
  (system prompts, repeats, forked agents, restarts).
- **Batched DFlash2 drafting** at any number of concurrent requests, with an online cost fit.
- **Prefill beside decode.** While another request streams, a long prompt prefills in calls of
  `--prefill-chunk-beside-decode` tokens (default 1,024), each but the last inside the decode
  round's Forward (a mixed round), so the stream keeps going; on the exact route the output is bit
  for bit the same.
- **Tool-call grammar fixes.** A constrained auto tool turn treats a call opener inside Markdown
  code as text, and may end inside its thinking (non-strict calls written there are recovered).
- `--served-model-name` aliases, and a Linux fix to VRAM spill detection.

## Measured

RTX PRO 6000 Blackwell, orcarouter Qwen3.8-Flash-Next-Uncensored-NVFP4 (recipe C conversion), BF16
n-gram volume, DFlash2 drafter, FP8 KV, 6 lanes, 262,144-token context, no Disk prefix cache. One
server boot, placements switched at runtime in the order shown; speeds are tok/s from the server's
own request log (commit `fc8826e`, 2026-10-30, `make t3`). [Performance](docs/performance.md)
defines each column's workload.

| Placement | VRAM / RAM / SSD experts | Process VRAM, GiB (experts / KV / other) | Process RAM | Decode, 1 request | Decode at 30K context | Prefill, 30K | 6 requests, total |
|---|---|---|---|---|---|---|---|
| `vram=100%` | 24,576 / 0 / 0 | 90.3 (63.3 / 19.3 / 7.7) | 12.8 GiB | 380 | 333 | 13,700 | 541 |
| `vram=65%,ram=15%,ssd=20%` | 15,974 / 3,658 / 4,944 | 68.2 (41.2 / 19.3 / 7.7) | 23.6 GiB | 232 | 220 | 9,750 | 380 |
| `vram=50%,ram=50%` | 12,288 / 12,288 / 0 | 58.7 (31.7 / 19.3 / 7.8) | 45.8 GiB | 263 | 208 | 8,160 | 388 |
| `vram=50%,ssd=50%` | 12,288 / 0 / 12,288 | 58.7 (31.7 / 19.3 / 7.8) | 14.2 GiB | 68 | 51 | 3,990 | 193 |

Switching back to `vram=100%` at the end of the same boot measured 379 / 287 / 13,600 / 525. The
30K-context decode moves with DFlash2 acceptance (69.5 % in the first row, 64.2 % in the last), and
a second boot's `vram=100%` row measured 289 there. The benchmark flags the
`vram=65%,ram=15%,ssd=20%` row as disturbed (its measured pass decoded slower than its warm pass)
in two boots; the second measured 242 / 196 / 9,750 / 400.

Switches take 1.7-8.6 s; growing back to `vram=100%` took 4.2 s. Process RAM at `vram=100%` is the
8 GiB prefix-cache host tier (`--host-context-mib`) plus about 4.8 GiB of engine heap. RAM is worth
far more than SSD for experts that leave VRAM: half the experts in RAM decode at 263 tok/s, half on
SSD at 68.

## Build

Ubuntu with the CUDA 13 toolkit (13.4 here), CMake 3.28+, a C++20 compiler, Ninja, `pkg-config`,
and the FFmpeg and curl development packages:

```bash
sudo apt-get install -y build-essential cmake ninja-build pkg-config \
  libavformat-dev libavcodec-dev libavutil-dev libswscale-dev libcurl4-openssl-dev
export CPATH=/usr/local/cuda-13.4/include   # the plain C++ test targets need the CUDA headers
cmake --preset dev
cmake --build build -j8 --target infernix-serve   # just the server
cmake --build build -j8                           # everything, tests and benchmarks included
```

One `build/` directory holds the server, the tests (`build/tests/`) and the benchmarks; the `dev`
preset is the optimized `release` build plus those targets. The server is
`build/apps/infernix-serve`. A parallel nvcc build can need a lot of host memory; cap it on smaller
machines.

## Models

Convert Hugging Face checkpoints with the Python 3.11 converter (see
[weight conversion](docs/weight-conversion.md) and the
[Flash-Next guide](docs/qwen3_8-flash-next.md)). The artifacts used above:

```bash
python -m tools.convert --model <orcarouter checkpoint> \
  --recipe qwen3_8_flash_next_nvfp4_orcarouter --components text,vision,mtp --proposal \
  --device cpu --name Qwen3.8-Flash-Next-Uncensored-NVFP4-Infernix \
  --resource chat_template.jinja=tools/chat_templates/qwen3_8.jinja \
  --out Qwen3.8-Flash-Next-Uncensored-NVFP4-Infernix.infernix \
  --ngram-out Qwen3.8-Flash-Next-Uncensored-NVFP4-Infernix.ngram
python -m tools.flash_next.ngram_bf16 --model <orcarouter checkpoint> \
  --out Qwen3.8-Flash-Next-BF16.ngram
python -m tools.convert --model <orcarouter checkpoint> \
  --source dflash2=<incoai/Qwen3.8-Flash-Next-DFlash2 checkpoint> \
  --components dflash2 --recipe qwen3_8_flash_next_dflash2 --device cpu \
  --name Qwen3.8-Flash-Next-DFlash2-Infernix --out Qwen3.8-Flash-Next-DFlash2-Infernix.infernix
```

## Serve

```bash
build/apps/infernix-serve Qwen3.8-Flash-Next-Uncensored-NVFP4-Infernix.infernix \
  --ngram-volume Qwen3.8-Flash-Next-BF16.ngram --kv-dtype fp8 \
  --max-context 262144 --max-concurrency 6 \
  --spec dflash2 --draft-model Qwen3.8-Flash-Next-DFlash2-Infernix.infernix \
  --draft-tokens auto --lm-head-draft --tolerant-tool-calls \
  --prefix-cache-disk prefix-cache --prefix-cache-disk-gib 64 \
  --served-model-name qwen3.8-flash-next --host 0.0.0.0 --port 8000
```

It serves the OpenAI Chat Completions and Responses APIs and the Anthropic Messages API
([serving](docs/serving.md), [CLI](docs/cli.md)). FP8 KV measured within 0.0013 mean KL of INT8
and enables an 8-bit activation fast path that decodes about 12 % faster; `--kv-dtype int8` is the
higher-accuracy choice. On this machine the server is started and stopped through
[super-quant](https://github.com/a4501150/super-quant) (`make serve-infernix`, `make stop-infernix`).

## Expert placement

At startup, `--experts` takes shares (`vram=60%,ram=25%,ssd=15%`, adding up to 100 %), counts
(`vram=14745,ram=6144`, the rest on SSD) or `vram=100%`; `--expert-placement FILE` assigns
individual experts. Without either, every expert goes to VRAM when it fits.

At runtime:

```bash
# State: per-tier counts and bytes, KV bytes, free VRAM and RAM, swap counters, last switch
curl -s localhost:8000/admin/expert-placement
# Predict a placement's memory use without applying it
curl -s -X POST 'localhost:8000/admin/expert-placement?dry_run=1' -d '{"vram":"50%","ram":"50%"}'
# Apply it; GET reports progress, and the outcome in last_switch
curl -s -X POST localhost:8000/admin/expert-placement -d '{"vram":"50%","ram":"50%"}'
```

A switch that cannot be reached, for example a grow while another program holds the VRAM, stops
where it can and reports that in `last_switch` (`ok: false`, with the reason); the server keeps
serving. Placement policy is client code: super-quant's `scripts/infernix_placement.py` computes
shares from reserves (`--reserve-vram-gb 32 --reserve-ram-gb 24`), checks them with a dry run and
applies them, behind `make infernix-gaming` and `make infernix-full`.

## Documentation

The design and reference documents come from Infernix and remain accurate for the parts this
repository did not change; start at the [documentation map](docs/README.md). The
[Flash-Next guide](docs/qwen3_8-flash-next.md) covers the placement flags. Repository rules for
people and agents are in [AGENTS.md](AGENTS.md).

## Credits

ninferpro starts from Infernix commit `7dbe99a` by [Wallawalla47](https://github.com/Wallawalla47),
which grew out of [NInfer](https://github.com/Neroued/ninfer) by
[Neroued](https://github.com/Neroued). The engine, its kernels, its artifact format and most of its
documentation are their work; [NOTICE](NOTICE) lists what changed here. Infernix in turn credits
[Strata](https://github.com/Niko1221/Strata) by Niko1221 and many other contributors.

Licensed under the Apache License, Version 2.0 ([LICENSE](LICENSE)). Third-party components keep
their own licenses ([third_party](third_party) and the
[third-party notices](packaging/windows/THIRD_PARTY_NOTICES.md)).
