# Qwen3.8-Flash-Next (`Qwen4ExpForCausalLM`)

ninferpro runs three checkpoints of Qwen3.8-Flash-Next on one `sm_120a` GPU: NVIDIA's NVFP4
checkpoint (recipes A and B below), ISTA-DASLab's GSQ-RCO IQ3_S GGUF bit-exactly
([GGUF import](#gguf-import-ista-daslab-gsq-rco-iq3_s)) and Qwen's official FP8 checkpoint
([official FP8 checkpoint](#official-fp8-checkpoint)). The 24,576 routed experts (63 GiB) live
across three tiers — VRAM, RAM and SSD — in any proportion you choose
([Expert placement](#expert-placement)). Production is an RTX PRO 6000 (96 GB) with every expert
in VRAM; an RTX 5090 (32 GB) runs part of the experts in VRAM and the rest in RAM and/or SSD.
The design, measurements and open work are in the
[design document](maintainer/qwen3_8-flash-next-design.md) (§1.3, §9).

Generation works through `infernix`, `infernix-serve` and `infernix_bench`, with Inco AI's
DFlash2 drafter (`--spec dflash2 --draft-model FILE`, [recommended](#run)) or the model's MTP
drafter (`--spec mtp`), n-gram copy proposals, images and video (`--vision`), and in
`infernix-serve` the hybrid prefix cache. Causal scoring (`infernix-perplexity`, the
decode-quality judge) runs teacher-forced through the generation path; `--a16-activations` scores
with 16-bit activations in every linear and every expert bank (the exact W4A16 expert
arithmetic): the [quality reference](maintainer/model-quality.md#6-numerical-acceptance-policy-and-fast-paths).
Pass the n-gram volume with `--ngram-volume` when it is not `<artifact>.ngram`, and an expert
stripe with `--expert-stripe` in the SSD tier. Where an expert is computed never changes a bit.

## Requirements

- **GPU.** One `sm_120a` card (RTX 5090, RTX PRO 6000 Blackwell). How much of the 63 GiB of
  routed experts fits depends on the card: a 96 GB card holds all 24,576 (production measured
  90.3 GiB of process VRAM at `vram=100%`, FP8 KV, 6 lanes, 262,144-token context); a 32 GB card
  holds about 9,000 expert frames beside the dense weights, the KV pool and the workspaces, and
  the placement's RAM and SSD tiers serve the rest. A **PCIe x16** link matters when experts
  live off the GPU: expert reads cross PCIe, and an x8 link halves their bandwidth
  (`nvidia-smi --query-gpu=pcie.link.width.current --format=csv`).
- **RAM.** What the placement asks for. `vram=100%` needs only the engine's own buffers
  (production's process RAM is the prefix cache's 8 GiB Host tier plus about 4.8 GiB of heap);
  every expert in RAM needs ~64.5 GiB pinned for recipes A and B, 46.9 GiB for the GGUF import
  and 112.6 GiB for the FP8 conversion. At startup a RAM ledger plans every allocation and keeps
  `--ram-headroom-mib` (default 2048) free for the system; `--expert-ram-mib` above what it
  leaves is an error, not a silent clamp. Windows page-locks RAM for the GPU only up to its
  *shared GPU memory*: see [Page-locking limit](#page-locking-limit).
- **Disk.** ~80 GB for the artifact, plus the n-gram volume (52 GB FP8, 29.1 GB IQ4_NL, ~109 GB
  BF16) on a fast NVMe drive: n-gram rows and SSD-tier experts are read in random 4 KiB blocks
  (about 1.1M block reads in a 128K-token prompt, a few per decode round). A SATA SSD or a hard
  disk is far slower.

## Convert

Two recipes convert NVIDIA's checkpoint. Both import the NVFP4 routed experts (codes, block
scales, per-matrix multipliers and input scales) and the FP8 n-gram table bit-exactly, and both
store the MTP drafter in `q8_g32_fp16` with `q4_g64_fp16` routed experts: the drafter only
proposes tokens that the full model verifies, so its precision changes speed, never output.

| | Recipe A, `qwen3_8_flash_next_nvfp4` | Recipe B, `qwen3_8_flash_next_nvfp4_dense8` |
|---|---|---|
| Published as | `Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix` | `Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix` |
| Main model and vision | **bit-exact**: every tensor as NVIDIA stores it (BF16 stays BF16) | dense projections and `lm_head` (629 tensors, 97 % of the dense bytes) in `q8_g32_fp16`; everything else bit-exact |
| Model files | 80.4 GB | 76.9 GB |
| Dense weights read per token | ~8.6 GB (BF16) | ~5.1 GB |

The two are level on quality (see [Performance](#performance)); recipe B leaves more VRAM
for the experts and reads fewer bytes per token, so it decodes faster. Use recipe A where
outputs must come from NVIDIA's weights unchanged. `--proposal` adds the smaller draft head
that `--lm-head-draft` uses, to either.

```text
python -m tools.convert --model <nvidia/Qwen3.8-Flash-Next-NVFP4 dir> --recipe qwen3_8_flash_next_nvfp4_dense8 \
  --components text,vision,mtp --proposal --device cpu --name Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix \
  --resource chat_template.jinja=tools/chat_templates/qwen3_8.jinja \
  --out <dir>/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix.infernix --ngram-out <nvme>/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram
python -m tools.artifact.rename <dir>/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix.infernix Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix.infernix --numbered
```

The second command names the three files as numbered parts
(`Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix-00001-of-00003.infernix` and so on); the engine
opens part 1. `--resource` stores the maintained Qwen3.8 template (see
[Chat template](#chat-template)); keep it in every recipe below.

- **Recipe A.** Use `--recipe qwen3_8_flash_next_nvfp4` and the name without `-Dense8`.
- **Recipe C, weight-only exports** (the production artifact). `--recipe
  qwen3_8_flash_next_nvfp4_orcarouter` converts
  [orcarouter/Qwen3.8-Flash-Next-Uncensored-NVFP4](https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored-NVFP4)
  (published as `Qwen3.8-Flash-Next-Uncensored-NVFP4-Infernix`): its NVFP4 experts and FP8 dense
  projections, stored without activation scales, are imported bit-exactly and run with BF16
  activations in an exact integer arithmetic (W4A16 experts, W8A16 dense); its BF16 classes
  follow recipe B. Its n-gram table (BF16 in the checkpoint) quantizes to NVIDIA's FP8 table byte
  for byte, so pass `--ngram-reuse` with the NVIDIA volume (§6.2, §17.3 of the
  [design notes](maintainer/qwen3_8-flash-next-design.md)).
- **Verification.** `python -m tools.flash_next.verify_artifact --model <checkpoint> --artifact
  <part 1> --ngram <volume>` compares every stored tensor, expert and n-gram row with the
  checkpoint. A second artifact of the same checkpoint can share an existing n-gram volume
  (`--ngram-reuse` instead of `--ngram-out`; the converter checks the geometry and 512 rows
  against the checkpoint, so a foreign volume is refused). A conversion takes ~11 minutes from
  a warm disk cache.

### GGUF import: ISTA-DASLab GSQ-RCO IQ3_S

Recipe `qwen3_8_flash_next_gsq_rco_iq3s` imports
[ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
`IQ3_S` bit-exactly into one artifact, `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix` (not published):

- Every quantized GGUF tensor is stored as its own GGUF blocks, byte for byte: the imatrix mix of
  `IQ2_S`-`IQ4_NL` gate/up experts, `IQ4_NL` and `Q2_0` down experts, and `Q4_K`-`IQ4_NL` dense
  projections (in tiles of 16 rows, `ggml_tiles16_v1`; bit-identical outputs). The one `Q8_0`
  tensor is stored as the identical `q8_g32_fp16` words, and BF16/F32 tensors keep their values.
  The converter undoes llama.cpp's layout transforms exactly and checks every one; the imatrix
  only shaped the quantization, so it needs nothing at run time.
- The PLE n-gram table is the GGUF's `IQ4_NL` table, a 29.1 GB volume (27.1 GiB).
- The GGUF has no MTP head, so `--model` names a Qwen checkpoint directory that supplies it (the
  `mtp.*` tensors, `config.json`, the tokenizer files; `tools/convert/fetch_tensors.py` reads
  just those from [Qwen/Qwen3.8-Flash-Next-FP8](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8)
  at a pinned revision). The drafter is re-quantized as in recipes A-C, with `q2_g64_fp16` experts.
- The vision tower comes from the repository's `mmproj` (BF16): `--components text,vision,mtp
  --mmproj <mmproj-BF16.gguf>`. Its 441 tensors equal the tower recipes A-C import from Qwen's
  checkpoint bit for bit.

Every weight is computed with BF16 activations in an exact integer arithmetic: each output is the
exact sum, rounded once, so the GPU's result has the same bits wherever an expert is computed and
at any batch width ([design §17.3](maintainer/qwen3_8-flash-next-design.md)). llama.cpp quantizes
activations to Q8_1 instead, so its logits differ slightly from these.

```text
python -m tools.convert.fetch_tensors --repo Qwen/Qwen3.8-Flash-Next-FP8 --revision 236dfdf285828023ca3bcd3f37366c58a3469b13 \
  --out <dir>/Qwen3.8-Flash-Next-FP8-partial --pattern "mtp.*"
python -m tools.convert --model <dir>/Qwen3.8-Flash-Next-FP8-partial --gguf <IQ3_S-00001-of-00002.gguf> \
  --gguf <IQ3_S-00002-of-00002.gguf> --mmproj <mmproj-BF16.gguf> --recipe qwen3_8_flash_next_gsq_rco_iq3s \
  --components text,vision,mtp --proposal --device cpu --name Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix \
  --out <dir>/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix.infernix \
  --ngram-out <nvme>/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix.ngram
python -m tools.artifact.rename <dir>/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix.infernix Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix.infernix --numbered
python -m tools.flash_next.verify_gguf_artifact --model <dir>/Qwen3.8-Flash-Next-FP8-partial --gguf <shard 1> \
  --gguf <shard 2> --mmproj <mmproj-BF16.gguf> --artifact <dir>/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix-00001-of-00002.infernix \
  --ngram <volume> --ngram-rows all
```

The rename names the two files as numbered parts, as for recipe B; open part 1. The conversion
took 11-14 minutes. The artifact is 56.8 GB, of which the routed experts are 46.9 GiB.

**Speed and quality** (RTX 5090 on PCIe 5.0 x8, i9-13900K, INT8 KV, one request, against recipe B
in the same session on 2026-10-16; the n-gram table in RAM, `auto`; one run of `-r 3`, or `-r 1`
at 128K):

| `infernix_bench` | GSQ-RCO IQ3_S | Recipe B |
|---|---:|---:|
| tg512, plain | 145.6 tok/s | 110.3 tok/s |
| tg512, `--spec mtp --draft-tokens 4 --lm-head-draft` (measured with a fixed 4; now `auto`, 0-7) | 259.9 tok/s (75 % of drafts accepted) | 143.1 tok/s |
| pp4096, `--prefill-chunk 4096` | 2,228 tok/s | 1,662 tok/s |
| pp131072, `--prefill-chunk 4096`, code and docs | 4,324 tok/s | 10,933 tok/s |

Decode is faster than recipe B's: the IQ3_S experts are smaller (the cache holds 13,008 of them,
and 98.6 % of routed experts hit it in tg512), and the CPU expert engine computes part of the
misses with the same bits as the GPU. Prefill of 4K tokens is faster; at 128K the tiles' integer
arithmetic is the limit against recipe B's FP4 tensor cores, so recipe B is 2.5× faster. Quality:
teacher-forced perplexity over the three frozen texts in [Performance](#performance)
(2,557 positions, INT8 KV) is 4.794
against 4.653 for recipe B (+0.030 ± 0.011 nats per token, mostly on chat), level with Strata's
UD-Q4_K_XL (4.864). The mean KL divergence from recipe B's distribution is 0.10.

### Official FP8 checkpoint

Recipe `qwen3_8_flash_next_fp8` imports
[Qwen/Qwen3.8-Flash-Next-FP8](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8) bit-exactly into
one artifact, `Qwen3.8-Flash-Next-FP8-Infernix` (not published). Each routed expert's E4M3FN codes
and BF16 128×128 tile scales are stored unchanged (`fp8_e4m3fn_b128`, a 4.69 MiB record)
— the same bits in every tier. The n-gram table is the [BF16 n-gram table](#bf16-n-gram-table):
the checkpoint's FP8 table is that table quantized, so `--ngram-reuse` with the BF16 volume checks
that sampled rows quantize to the checkpoint's FP8 codes. The MTP drafter is re-quantized as for
the IQ3_S import (projections `q8_g32_fp16`, routed experts `q2_g64_fp16`; the smaller drafter
leaves its 0.63 GB of VRAM to the experts):

```text
python -m tools.convert --model <dir>/Qwen3.8-Flash-Next-FP8 --recipe qwen3_8_flash_next_fp8 \
  --components text,vision,mtp --proposal --device cpu --name Qwen3.8-Flash-Next-FP8-Infernix \
  --out <dir>/Qwen3.8-Flash-Next-FP8-Infernix.infernix --ngram-reuse <fast drive>/Qwen3.8-Flash-Next-BF16.ngram
python -m tools.artifact.rename <dir>/Qwen3.8-Flash-Next-FP8-Infernix.infernix Qwen3.8-Flash-Next-FP8-Infernix.infernix --numbered
python -m tools.flash_next.verify_artifact --model <dir>/Qwen3.8-Flash-Next-FP8 \
  --artifact <dir>/Qwen3.8-Flash-Next-FP8-Infernix-00001-of-0000N.infernix \
  --ngram <fast drive>/Qwen3.8-Flash-Next-BF16.ngram --ngram-rows all
```

The artifact is 132.7 GB in five numbered parts, of which the routed experts are 112.6 GiB — more
than a 96 GB machine can hold in RAM, so it runs with its experts in the SSD tier (measured, INT8
KV, one request, the BF16 table, RTX 5090, 2026-10-16/17: tg512 plain
43.8-44.4 tok/s, MTP (a fixed 4 then, `auto` now) 43.5-49.3, with `--ngram-draft-tokens 7
--ngram-min-match 4` 53.8-54.2, pp4096 690 tok/s). On the 1M-token perplexity corpus (quick, 4K context, INT8 KV) it scores
3.487 against recipe B's 3.508, with a mean KL divergence of 0.050.

## Run

Two starting points. A card that holds every routed expert needs no placement flag: the default
probe keeps all of them in VRAM (an RTX PRO 6000 always fits; a smaller card falls back to the
maximal RAM tier). Production (RTX PRO 6000, recipe C artifact):

```text
infernix-serve <artifact>.infernix --ngram-volume Qwen3.8-Flash-Next-BF16.ngram --kv-dtype fp8 \
  --max-context 262144 --max-concurrency 6 \
  --spec dflash2 --draft-model Qwen3.8-Flash-Next-DFlash2-Infernix.infernix \
  --draft-tokens auto --lm-head-draft --vision
```

An RTX 5090 or another card that does not hold every expert: the default placement pins every
expert in RAM and keeps a VRAM cache of hot ones, or name the split explicitly with `--experts`
(counts or shares; the rest read from the artifact):

```text
infernix-serve <artifact>.infernix --ngram-volume <volume>.ngram --kv-dtype fp8 --max-context 16384 \
  --experts vram=8192,ram=16384
infernix <artifact>.infernix --ngram-volume <volume>.ngram --kv-dtype fp8 --max-context 16384 --prompt "..."
```

- **`--kv-dtype fp8` is the recommended KV format** (the production setting). On the recipe C
  artifact it also runs the `fp8-a8-pair` fast path, which is registered with FP8 KV but not INT8
  (+5-6 % prefill at 32K-64K); the qualified fast-path set was last gated on it 2026-11-03
  ([model-quality §6.7](maintainer/model-quality.md#67-fast-path-registry)). Its noise floor
  against the 16-bit reference is 7.5e-3 mean KL in prefill, against INT8's 6.1e-3 (recipe C);
  `int8` is also qualified. The other formats: `bf16` (the
  default), `nvfp4` (55 % of `int8`'s KV memory, which leaves more room for experts), `k8v4`,
  `vq2` and `k4v2` (each keeps a sequence's recent and first tokens exact and quantizes older
  ones harder; design §14.3). A context of 262,144 tokens takes 3.7 GB at `fp8` or `int8`.
- **Context.** The model's native context is 262,144 tokens. Long contexts work the same way,
  but their KV cache takes VRAM from the experts. A resident placement maps the whole pool at
  startup, so `--max-context` costs the pool's VRAM once, not frames per request.
- **Beyond 262,144 tokens: `--rope-yarn-factor F`** (1-4). It enables YaRN on every rotation of
  the model and raises the context ceiling to 262,144 × F (1,048,576 at 4). It does not raise
  `--max-context` itself: set both. Without a factor, a `--max-context` above 262,144 is refused.
  YaRN changes shorter contexts slightly (design §14.5), so use it only when you need the
  length: a catalog retrieval test was right at 470K tokens at factors 2 and 4, and lost the
  middle record at 970K and factor 4 (GitHub #22).
- The prefill chunk defaults to 16,384 tokens. The exact route's output measured identical at
  chunks of 1,024, 4,096 and 12,288, and wide chunks prefill long prompts faster at no decode
  cost (Dense8, INT8 KV, RTX 5090, 2026-10-16: 32K prompts 5,570 / 9,640 / 11,210 tok/s at chunks
  1,024 / 4,096 / 16,384; tg512 unchanged). While another request decodes, a prefilling request's
  calls ride in the decode round's Forward in pieces of `--prefill-chunk-beside-decode` tokens
  (default 1,024), so the decode keeps streaming.
- **`--spec dflash2 --draft-model FILE --draft-tokens auto --lm-head-draft` (recommended)**
  drafts with Inco AI's DFlash2 drafter, a separate 776 MB companion artifact
  ([Qwen3.8-Flash-Next-DFlash2-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-DFlash2-Infernix))
  that runs beside any Flash-Next conversion. Each round drafts a block of seven tokens in one
  pass; with `auto` it verifies the number of them that pays for its verification columns, from
  each draft's calibrated probability. It measured ahead of MTP on code and documents in every
  paired test (RTX 5090, INT8 KV, 2026-10-20; design §11.3). `--draft-tokens N` (1-7) verifies N
  every round. Greedy output is the same as without it when one request runs at a time;
  sampling keeps the sampler's distribution.
- `--spec mtp --draft-tokens auto --lm-head-draft` drafts with the model's MTP layer. `auto`:
  each round drafts the number of tokens, 0 to 7, the request's recent acceptance makes
  worthwhile, and verifies them up to the first the drafter gives below 0.5 probability
  (`--spec-min-p`). `--draft-tokens N` (1-7) drafts and verifies N every round. The drafter's 512
  experts (1.34 GB), its workspace and the proposal head (178 MB) stay in VRAM.
- `--ngram-draft-tokens 7` verifies copy proposals (with `--ngram-min-match`, default 12). It
  works alone or beside MTP; a longer copy proposal replaces a round's MTP drafts. On
  code-editing prompts it accepts most drafts; on prose it finds none and costs nothing. In the
  agentic replay (greedy, two lanes, 2026-10-18) it was level with none, so the recommended
  commands leave it off. See [ngram copy proposals](ngram.md).
- `--max-concurrency N` (1-8) runs up to N requests at once; the main gain is a shorter wait
  for requests that would otherwise queue. Speculation depends on how many requests decode
  together: one picks its own draft length; two draft the same length (at most 6 columns a
  round, the same greedy output as each alone, design §11.5); three or more draft nothing (one
  token per request per round). Each lane holds its KV extent (14.6 KB per token of
  `--max-context` at INT8/FP8 KV, unless `--kv-capacity` fixes one pool) plus its recurrent
  state, records and workspace; startup prints the cost per lane in expert frames.
- The engine starts six CPU worker threads for missed experts (up to 16 per layer call); more
  measured slower, as the CPU and the PCIe stage share the host's memory bandwidth. The GSQ-RCO
  IQ3_S artifact's CPU route is compute-bound and starts 16 workers instead.
- On Windows the engine opts its process out of power throttling, which otherwise slows a
  process it considers in the background and moves its threads to efficiency cores: a hidden
  run decoded 37.6 tok/s at 8K throttled and 98.7 exempt (design §8.1).
- `--vision` enables images and video. The vision tower stays in pinned RAM and borrows expert
  frames only while it encodes (`--vision-offload auto`, the default for this model); a resident
  placement lends no frame, and there the encode's window is a device allocation for the
  encode's lifetime instead.
- **Warm start.** The engine saves the expert cache's state (which experts it holds and how
  often each was used) to `<artifact>.expert-state` when it stops and every 10 minutes between
  requests. The next start loads the highest-ranked experts into half of the cache before the
  first request (under a second), so the first few hundred tokens decode faster when the work
  resembles the last session's (measured +24 % plain and +30 % with MTP on a repeated coding
  prompt, RTX 5090), and about as fast as a cold start when it does not. With no saved state
  yet, the engine seeds from a built-in calibrated expert order (ranked over 128 tasks in nine
  domains; +9 % on a coding prompt, +15 % on a math problem, +56 % with the SSD tier, against an
  empty cache). `--expert-state FILE` moves the file; `--expert-state off` starts with an empty
  cache and no seed, as cold benchmarks need (`infernix_bench` always starts that way). Output
  is the same either way.
- The `infernix-serve` statistics panel's `experts` column shows the expert cache's hit rate
  over the session and the last ten requests. After each request the engine logs two Debug
  lines (`--log-level debug`): the expert cache's hit rate, and the request's n-gram row
  traffic (`n-gram rows: N requested, H% host-cache hits, R NVMe reads, T ms of reads`).

## Expert placement

A placement names how many experts each tier holds; the engine places exactly that, moves only
the experts whose tier changes, and reports or refuses what it cannot reach. Within the counts,
the engine swaps hot experts into VRAM (and into RAM) and cold ones out as the workload changes
(`--expert-swap off` freezes the pools' content; misses still read and compute). Outputs are
the same bits in every placement.

**Startup forms**:

- **Default (no flags).** The engine probes whether every expert fits in VRAM beside the
  fully-mapped KV pool and the fixed allocations. On a 96 GB card it always fits: every expert
  resident, nothing lent or evicted, and the RAM tier at its floor (~1,536 slots of lists,
  released after the load). A smaller card falls back to the maximal RAM tier: every expert
  pinned in RAM, a VRAM cache of hot ones. The startup line names the outcome (`holding all
  24,576 resident`, or the ledger's summary).
- **`--experts SPEC`.** Shares that add up to 100 % or counts, by usage rank: the tiers a spec
  leaves out take the remainder (`ssd`) or 0 (`ram`); `vram` is required; `vram=all` (or `100%`)
  is the resident placement and fails the start if it does not fit; shares and counts cannot be
  mixed.

  ```text
  --experts vram=60%,ram=25%,ssd=15%
  --experts vram=14745,ram=6144
  --experts vram=all
  ```

  The rank is the saved session's counts (the same state the warm start reads), else the
  calibrated order, else spread across the layers. The vram count is the frame ceiling; a count
  the card cannot hold beside the fixed allocations fails the start with the numbers.
- **`--expert-placement FILE`.** A CSV assigning (layer, expert) pairs to tiers; `vram` rows are
  pinned exactly (never eviction victims), `ram` rows fill the RAM tier, and everything else
  (listed `ssd` or unlisted) reads from the artifact on use. It overrides `--experts`. Export the
  current placement with `GET /admin/expert-placement?export=PATH`: rows of
  `layer,expert,tier[,uses]`, best first, so loading the file reproduces the ranking too.

  ```text
  # infernix expert placement v1
  # tier is vram, ram or ssd; unlisted experts are ssd
  0,5,vram
  3,212,ram
  ```

- **`--expert-ram-mib N|auto` / `--expert-pin-mib N|auto`.** Byte-level control of the RAM tier
  (the SSD tier's slots) and of how much of it is page-locked. They cannot be combined with
  `--experts` or `--expert-placement`.

**Runtime switch.** `POST /admin/expert-placement` applies a new placement between rounds while
requests keep running; only the experts whose tier changes move, and freed VRAM and RAM go back
to the system:

```text
curl -s localhost:8000/admin/expert-placement
curl -s -X POST 'localhost:8000/admin/expert-placement?dry_run=1' -d '{"vram":"50%","ram":"50%"}'
curl -s -X POST localhost:8000/admin/expert-placement -d '{"vram":"50%","ram":"50%"}'
curl -s -X POST localhost:8000/admin/expert-placement -d '{"file":"/path/to/placement.csv"}'
```

The body takes `{"vram": N|"N%"|"all", "ram": ..., "ssd": ...}` or `{"file": PATH}`.
`dry_run` previews the placement without applying it: the per-tier counts, the predicted bytes,
VRAM left free, and whether it fits. A switch is validated before anything moves (the frame
limit, the pinned keys, the RAM target beside the tier's lists) and applied at the next round
boundary; `GET` reports the counts, bytes, KV, free VRAM and RAM, the fill's progress, the swap
counters and the last switch's outcome (`last_switch`), so a switch that cannot be reached (a
grow while another program holds the VRAM) says so and the server keeps serving. Switches
measured 1.7-8.6 s (RTX PRO 6000, 2026-10-30); a switch back to `vram=100%` left the engine whole.

**What each tier costs.** The measured table in [Performance](#performance) shows it on the
RTX PRO 6000 (2026-10-30): RAM is worth far more than SSD for experts that leave VRAM — half
the experts in RAM decode at 263 tok/s, half on SSD at 68.

**Policy is the client's.** The engine provides the mechanism; it does not choose shares, react
to other programs' memory use, or keep headroom heuristics for the placement. A policy such as
"leave 32 GB of VRAM and 24 GB of RAM free for other programs" belongs in client code:
[super-quant](https://github.com/a4501150/super-quant)'s `scripts/infernix_placement.py` computes
shares from reserves, previews them with `dry_run` and applies them, behind
`make infernix-gaming` and `make infernix-full` (`vram=all`).

## N-gram table in RAM

`--ngram-residency auto|ram|volume` chooses where the PLE n-gram table lives:

- `volume` reads rows from the NVMe volume on demand, through a host cache of 2^20 rows.
- `ram` reads the whole volume into pinned RAM at startup and never touches the drive again
  (27.1 GiB for the GGUF import's `IQ4_NL` table, 48.8 GiB for NVIDIA's FP8 table); the RAM
  ledger counts the table, and startup stops if it does not fit.
- `auto`, the default, takes RAM only when the ledger still keeps every routed expert in RAM
  beside the table, and the volume otherwise.

The startup line names the choice: `n-gram table: in RAM (N MiB pinned)`, or `n-gram table: read
from its volume`. The rows are the same bytes either way, so outputs do not change.

## N-gram table formats

The PLE n-gram table has three volume formats: NVIDIA's FP8 rows with one scale for the whole
table (recipes A-C bind it), the GGUF import's IQ4_NL rows (4.5 bits per value), and the BF16
rows of Qwen's checkpoint (the FP8 checkpoint's artifact binds it). Every artifact binds one of
them as its own, and **any Qwen3.8-Flash-Next artifact runs with a volume of any of the three
formats**: pass it with `--ngram-volume`; the engine reads the row format from the volume's
header and logs the swap. The three volumes hold the same table row for row: on 3,002 sampled
rows, the FP8 rows' relative L2 error against the BF16 rows was 2.7 % (median) and the IQ4_NL
rows' 7.6 %; with the NVIDIA NVFP4 Dense8 artifact (the 1M-token corpus's quick subset, INT8 KV;
2026-10-18) the perplexity was 3.5102 with its own FP8 table, 3.5101 with the BF16 table and
3.5140 with the IQ4_NL table.

FP8 rows need the table's scale. An FP8 artifact stores it (`ngram_scale`), so its own volume
always works. For any other artifact the engine reads it from the volume's header (volumes
record it since 2026-10-18), or recognises NVIDIA's published FP8 volume by the volume id in its
header, so the downloaded file works unchanged. An FP8 volume with neither serves only its own
artifact.

### BF16 n-gram table

A BF16 volume holds the table as Qwen's BF16 checkpoint stores it (the production volume):

```text
python -m tools.convert.fetch_tensors --repo Qwen/Qwen3.8-Flash-Next --revision <sha> --out <dir> \
  --pattern "model.language_model.layers.*.ple.ple_embedding.*"
python -m tools.flash_next.ngram_bf16 --model <dir> --out <fast drive>/Qwen3.8-Flash-Next-BF16.ngram
python -m tools.flash_next.ngram_bf16 --model <dir> --out <fast drive>/Qwen3.8-Flash-Next-BF16.ngram --verify
infernix-serve <artifact part 1> --ngram-volume <fast drive>/Qwen3.8-Flash-Next-BF16.ngram ...
```

Any checkpoint that carries Qwen's BF16 table unchanged serves as `--model`. The table is ~102
GB in the checkpoint and the volume ~109 GB (rows of 320 bytes, against 52 GB for FP8); a drive
with fast random reads serves it as fast, and it does not fit in RAM (`--ngram-residency auto`
keeps it on the volume). A persistent prefix cache saved with one table is not restored with
the other.

## Chat template

The conversion above stores [qwen3_8.jinja](../tools/chat_templates/qwen3_8.jinja), the template
Qwen3.8-27B uses; Flash-Next's own template is Qwen3.8-27B's. Among the template's changes, it
keeps a System or developer message after the first turn in place, as serving documents (Claude
Code's normal mode sends them). Artifacts converted without it, including the published
Flash-Next artifacts, store the checkpoint's template, which rejects such a message with
"System message must be at the beginning"; start them with
`--chat-template tools/chat_templates/qwen3_8.jinja` (also in the Windows release zip's
`chat_templates` folder).

## Prefix cache

`infernix-serve` reuses prompt prefixes with the hybrid prefix cache (its default;
[spec](maintainer/hybrid-prefix-cache-spec.md#17-qwen4exp-binding-qwen38-flash-next)). A
conversation's next turn resumes from a snapshot of the model state and prefills only the new
tokens; KV blocks are shared across requests. `infernix` (one request) runs without it.
Production runs the RAM tier only: an 8 GiB Host pool.

- **Host RAM.** Snapshots (about 110 MiB each) and KV blocks live in one pinned Host pool,
  `--host-context-mib` (default 8192; 0 only with `--prefix-cache-disk`), counted in the RAM
  ledger. `--no-prefix-reuse` turns the cache off and frees that RAM.
- **Snapshots.** Each turn leaves one at the generation opener (the assistant turn's start) and
  one at its end; shared prefixes (the system and tools block, client breakpoints) get one where
  they end. A snapshot inside a prefill chunk costs one extra chunk call (estimated ~1.3 s), once
  per distinct prefix.
- **Queue holds.** What each waiting request would resume from is evicted after every other cache
  entry, the request furthest back in the queue first, so parallel agents whose contexts together
  exceed the cache lose whole conversations at the back of the queue instead of every waiting
  request losing part of its own (`--no-queue-holds` turns this off).
- **Exactness.** A resumed request computes what the request that left the snapshot computed; it
  may differ from an uncached run where two tokens are near ties. `--no-prefix-reuse` gives
  uncached runs.
- **Disk tier (optional, Linux only).** `--prefix-cache-disk DIR` adds a tier on the SSD,
  `--prefix-cache-disk-gib N` large (default 64), that keeps blocks and snapshots across restarts
  and can be larger than RAM (transfers use cuFile). With the Host pool, snapshots it would evict
  next are written to the SSD once and the rest when the server stops; with `--host-context-mib 0`
  every snapshot is written when captured. A restart rebuilds the tier's entries at once;
  another artifact, n-gram table, KV format, drafter, size or build empties it. Measured restarts
  (T3, 2026-10-29) resumed 20.8k-token prompts from the SSD in 54-60 ms against 1.7-1.8 s cold.
  Not available for this model: `--device-snapshot-slots`, `--prefix-cache-file`.

## SSD expert tier

When a placement does not keep every routed expert in VRAM or RAM, the rest stay in the artifact:
startup pins RAM slots for the RAM tier's share and fills them before the first request, and a
round that routes an expert in no tier reads it from the artifact while the layer's other experts
compute — decode and verification hand it to the CPU expert engine, prefill copies it from the
read as soon as it lands. Read experts that rank above the coldest RAM expert replace it, and
records that follow each other in the artifact are read as one request, which the drive serves
faster than one per expert. The tier needs at least 1,024 pinned slots plus its lists; below that
startup refuses with the reason. Outputs are the same bits as with every expert in VRAM or RAM;
only speed changes, with the share of routed experts read from disk.

A read that fails fails the requests running in that round with an error and the server keeps
serving; queued requests wait as usual, and the prefix cache is emptied (the failed round may
have published state it never computed). If the tier's read thread or the CPU expert service
itself stops, every later round would fail too, so the server stops with that cause. A round
waits for either while it shows signs of life, and fails only when one has been silent for 10 s
("the CPU expert service stopped answering" or "the SSD expert tier's read agent stopped
answering").

**A second drive (`--expert-stripe FILE`).** One drive bounds the tier's reads, most of all in
prefill, which reads every expert missing from RAM. An expert stripe holds the tail of every
expert record on another drive; the tier then reads each record's head from the artifact and its
tail from the stripe at the same time. The artifact is unchanged, so the stripe costs a copy of
the tails only (by default half of every record). The stripe names its artifact, which refuses
any other. Write it once, on a drive other than the artifact's (and the n-gram volume's), and
check it against the artifact; the stripe tool's `--tail-fraction F` (default 0.5) moves more of
each record to the second drive:

```text
python -m tools.flash_next.expert_stripe <artifact part 1> D:\stripes\<model>.stripe
python -m tools.flash_next.expert_stripe <artifact part 1> D:\stripes\<model>.stripe --verify
infernix-serve <artifact part 1> --expert-stripe D:\stripes\<model>.stripe ...
```

With every expert in VRAM or RAM the stripe is not used. Outputs are the same bits with or
without it.

The tier and the n-gram volume read with unbuffered, 4 KiB-aligned I/O. On ZFS (NInfer #404) a
dataset's `recordsize` (128 KiB by default) is the unit the filesystem reads, so each small read
can cost a whole record; keeping the artifact and the `.ngram` volume on a dataset with
`recordsize=16K` or less avoids that (not measured here; Windows NTFS is unaffected).

### Page-locking limit

The GPU reads RAM experts only from page-locked (pinned) memory, and Windows page-locks at most
the adapter's shared system memory for it: on an RTX 5090 with 95.8 GiB of RAM page-locking
stopped at 76.0 GiB of a reported 76.6 GiB (many machines report half of RAM). The RAM ledger
reads that size and its current use at startup and keeps 1 GiB below it. Every other pin comes
first (the embedding, the prefix cache's Host tier, the Program's buffers); the experts pin what
is left, and their RAM beyond it holds **unpinned records**:

- The RAM the experts get does not change: a 96 GB machine with a half-of-RAM limit still holds
  every expert in RAM, a 64 GB machine the same experts as without the limit.
- Unpinned records are ordinary memory kept resident (a raised minimum working set and
  `VirtualLock`), filled at startup with the experts ranked after the slots' and never replaced;
  the slots keep adapting.
- Decode and verification hand an unpinned expert to the CPU expert engine, which reads it in
  place (as fast as from pinned memory). A prefill streams it two layers ahead like a pinned one,
  through pinned bounce slots that reach about a third of the RAM bandwidth — pinned experts
  stay faster for prefill, which is why the coldest experts are the unpinned ones. Outputs are
  the same bits.
- If Windows refuses a slot chunk anyway (another program pinned memory meanwhile), the tier
  gives back enough for the prefix cache and the Program's buffers, and the slots it lost become
  unpinned records; a warning says so.
- `--expert-pin-mib N` page-locks at most N MiB of the experts' RAM (the rest unpinned), for
  example to leave shared GPU memory to other programs. The startup info line names the split
  (`… page-locked for the GPU and … kept in RAM for the CPU`).

## VRAM

At startup the engine reads the GPU's free memory, the Windows video-memory budget of the
process and whether the GPU drives a display. The expert cache takes

`min(free - fixed - headroom, budget - fixed - 64 MiB) - reserve`

where `fixed` is the dense weights plus the KV cache, workspaces and staging, `reserve` is
max(256 MiB, 128 MiB + 4 MiB per CUDA graph, at most 32 counted while the KV pool is elastic),
and `headroom` is `--vram-headroom-mib`: by default 256 MiB without a display, 512 MiB with a
display and an elastic KV pool (a placement that does not hold every expert), and 1,024 MiB
with a display when every expert is resident. The console shows a one-line summary
(`VRAM: model …, KV cache …, expert cache …`); `--log-level debug` adds the full ledger:

```text
VRAM ledger: 32607 MiB card, no display, 547 in use before loading; dense weights 4198 MiB, KV 1024,
workspace 1236, expert staging 169, state and io 310, expert tables 7; reserve 256 (20 graphs);
headroom 256 (auto); expert frames 9300 (23.95 GiB); 380 free after startup
```

- **Before reading the weights**, the engine checks that the fixed allocations fit; if they do
  not, it stops with the largest terms and what to lower (`--max-context`, `--max-concurrency`,
  `--prefill-chunk`, `--kv-dtype int8`). Below 2,048 expert frames it warns that decode will be
  slow.
- **No silent spill.** On Windows the driver can place an allocation that does not fit in shared
  system memory without an error, which slows everything down. The engine checks that every
  startup allocation lowered free VRAM by its size: dense weights or fixed buffers that spilled
  stop startup; expert frames that spilled are given back once and the cache starts smaller.
  Setting the NVIDIA Control Panel's *CUDA - Sysmem Fallback Policy* to *Prefer No Sysmem
  Fallback* for the `infernix-*` executables makes such an allocation fail instead (optional). On
  Linux allocations are VRAM-or-fail, so no spill can hide.
- **At runtime** the cache keeps the size the placement names: it no longer shrinks when other
  programs take memory. A monitor thread reads free VRAM once a second and on OS budget changes
  and reports pressure below the headroom, naming the placement as the lever. To give VRAM to
  another program, switch the placement (above); to take it back, switch again.
- **The KV cache is elastic.** Startup backs only its first 32K tokens (the ledger shows
  `KV 482 (grows to 3360 from the expert cache)`). A request that needs more KV takes the memory
  the sizing law would give the experts first, then expert frames, and gives it back when it
  ends; cached prefix blocks that also have a host copy return theirs after 60 s with no request
  running, and a later match restores them from host RAM. A resident placement maps the whole
  pool at startup, so growth is inert: nothing is lent or evicted, and a long `--max-context`
  costs the pool's VRAM, never expert frames.
- **`--vram-past-budget`** (Windows) also sizes the expert cache into VRAM that the Windows
  per-process budget withholds: on an RTX 5090 about 640 MiB more, roughly 240 expert frames,
  measured at startup and named in the ledger (`… in use before loading (640 MiB past the OS
  budget)`). The cost: that memory may be moved to system memory if another program needs VRAM,
  which slows decode until the placement changes. Off by default.
- <a id="expert-tail-skip"></a>**`--expert-tail-skip R`** (approximate, off by default) trades
  quality for decode speed. Each token routes 10 experts, ranked by router weight (rank 0 the
  largest). In a decode or verification round, an expert missing from VRAM that every token of
  the round routes at rank `R` or lower (`1..9`) is neither copied nor computed: it adds zero,
  and the other experts keep the weights the router gave them (no renormalisation). It saves
  the PCIe copies and CPU work of those misses, so it helps most when many experts miss VRAM
  (the SSD tier, small cards); with every expert in VRAM it changes nothing. The output changes
  and is slightly worse, and it depends on which experts the cache holds at the time, so it is
  neither the exact route nor request-invariant (`--disable-fast-paths` refuses it). Prefill,
  forced and scored tokens are unchanged; experts routed by more than 8 tokens of a round are
  always computed. Measured (2026-10-18, RTX 5090 on PCIe 5.0 x8, INT8 KV, MTP): rank 7 took
  Dense8 from 155 to 207 tok/s (+33 %), FP8 from 47 to 61 (+28 %) and IQ3_S from 210 to 234
  (+12 %); it failed the fast-path decode gate (model-quality §6;
  [design §17.5](maintainer/qwen3_8-flash-next-design.md)), which is why it is an option.
  Strata turns the same rule on by default at rank 7. 9 skips least, 1 most.
- `--kv-capacity auto` means `--max-context` x `--max-concurrency`, the same as omitting it.

## Performance

**RTX PRO 6000 Blackwell** (orcarouter recipe C artifact, BF16 n-gram volume, DFlash2 drafter,
FP8 KV, 6 lanes, 262,144-token context, no Disk prefix cache; commit `e984e41`, 2026-10-30,
`make t3`; one server boot, placements switched at runtime; tok/s from the server's request
log; process VRAM 90.3 / 68.2 / 58.7 GiB and process RAM 12.8 / 23.6 / 45.8 / 14.2 GiB at
`vram=100%` / `65%,15%,20%` / the 50 % rows):

| Placement | Experts VRAM / RAM / SSD | Decode, 1 request | Decode, 30K context | Prefill, 30K | 6 requests, total |
|---|---|---:|---:|---:|---:|
| `vram=100%` | 24,576 / 0 / 0 | 380 | 333 | 13,700 | 541 |
| `vram=65%,ram=15%,ssd=20%` | 15,974 / 3,658 / 4,944 | 232 | 220 | 9,750 | 380 |
| `vram=50%,ram=50%` | 12,288 / 12,288 / 0 | 263 | 208 | 8,160 | 388 |
| `vram=50%,ssd=50%` | 12,288 / 0 / 12,288 | 68 | 51 | 3,990 | 193 |

Switching back to `vram=100%` at the end of the same boot measured 379 / 287 / 13,600 / 525.
Workload definitions, the `make t3` command and the per-tier process VRAM split are in
[performance.md](performance.md) and the README's [Measured](../README.md#measured) table.

**RTX 5090** (Infernix-era measurements, 2026-10-07 and earlier, on an artifact converted from
`RadixArk/Qwen3.8-Flash-Next-NVFP4`, PCIe 5.0 x8, i9-13900K, recipe B, INT8 KV, one request;
they predate the placement architecture — the 5090 path is kept working but not measured on
this machine):

| Workload | tok/s |
|---|---:|
| `infernix_bench` tg512, `--spec mtp --draft-tokens 4 --lm-head-draft`, warm cache (measured with a fixed 4; `auto` drafts 0-7) | ~134 |
| `infernix_bench` tg512, plain decode (warm cache) | ~87 |
| Single CLI request, cold cache, plain: code rewrite / prose story | ~66 / ~65 |
| Prompt, 4,096 tokens, `--prefill-chunk 4096` | ~670-690 |

Recipe A (bit-exact) decodes about 18 % slower than recipe B on the same rig and reaches the
first token 6-8 % later at 8K, level from 128K (the bit-exact
[model card's performance table](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix#performance)).
MTP accepts most drafts on code and other predictable text and fewer on free prose; the draft
length follows the measured acceptance, so prose mostly drafts one token and stays at least as
fast as plain decode (design §11.4).

**Quality.** Teacher-forced perplexity over three frozen texts (2,557 positions, INT8 KV):
recipe B 4.654 and recipe A 4.666 (2026-10-08 kernels). For comparison, Strata 0.1.40 with
UD-Q4_K_XL and INT8 KV gives 4.844 on the same token ids (−0.038 ± 0.011 nats for Infernix;
level on code and documents, better on chat).
