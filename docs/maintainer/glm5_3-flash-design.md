# GLM-5.3-Flash on the RTX PRO 6000 and RTX 5090: design

**Status: in progress.** M4 and M6 are built: the Engine, CLI, server and causal scoring on the
shared VRAM, RAM and SSD expert tiers, the INT8-G64 latent cache, and DFlash2 speculation for
one-request rounds (§16). Prefill streams its experts by DMA per call (§17); the layer walk is not
built, and a fresh prompt prefills at ≈ 1,600 tok/s on the production machine, far below §5.4's
model. The prefix cache and vision are not built yet; a request for one is refused at startup. This document designs
`Glm5NextForConditionalGeneration` (GLM-5.3-Flash, model type `glm5_next`) on the shared offload
tier, which is built and in production for Qwen3.8-Flash-Next. The shared machinery — expert banks,
residency, tiers, placement, the CPU expert engine, decode and prefill on an offloaded MoE — is
designed once for every offloaded MoE family in [MoE offload design](moe-offload-design.md), cited
below as **MoE §N**. This document keeps GLM-5.3-Flash's own facts, formats, kernels, constants,
memory plans and measurements, and cites [Qwen3.8-Flash-Next's
design](qwen3_8-flash-next-design.md) as **FN §x** only for that family's measurements.

Labels for every number: **measured** (date, hardware and family named), **upstream** (the model
card, reference code or artifact metadata, cited in [§26](#26-sources)), **model** (computed here
from the stated inputs), **estimate** (judgement with the stated range). A milestone in brackets
replaces each model or estimate with a measurement.

**Default artifact** (owner decision, 2026-11-23):
**[orcarouter/GLM-5.3-Flash-Uncensored-NVFP4](https://huggingface.co/orcarouter/GLM-5.3-Flash-Uncensored-NVFP4)**,
a compressed-tensors `nvfp4-pack-quantized` checkpoint, 191.0 GiB in 65 shards (upstream). Only the
routed experts of layers 3-44 are quantized (NVFP4, §7.1); every other tensor is BF16 dequantized
from the vendor's block-FP8 checkpoint, with FP32 kept where the vendor had FP32. The engine runs
the experts at BF16 activations — the exact W4A16 arithmetic and record layout
(`nvfp4_expert_rg16_v1`) Qwen3.8-Flash-Next already uses (FN §6.3), so no new expert kernel or codec
is needed. The ExLlamaV3 (EXL3) builds that were the default until 2026-11-23 moved to the
rejected/optional list (§7.4, §25.4): they would need a new codec and kernels (≈ 7k lines) where
NVFP4 reuses built kernels and the converter source `tools/convert/sources/compressed_tensors.py`,
which already reads `weight_packed`.

**Two target machines, both first class:**

- the **production machine**: one RTX PRO 6000 Blackwell (96 GB, `sm_120a`), Linux, PCIe Gen5 x16,
  192 GB of host RAM (182 GiB usable), one Gen5 NVMe (T710) holding the models (AGENTS.md, "Local
  operations") — every Flash-Next measurement in the repo comes from this card;
- the **RTX 5090 machine** (32 GB, Windows 11, PCIe Gen5 x8): an i9-13900K host with 96 GB of
  DDR5-5800 and four NVMe drives.

Neither machine holds the whole 159.4 GiB expert bank (§2.3) in VRAM, so both run a tiered
placement; the design states the placement and the expected speed for each machine (§5, §19). On
the production machine the default placement holds the whole bank in VRAM + RAM, so the SSD tier
is optional there (§19.1); on the 5090 machine roughly half the bank lives on the SSD.

---

## Contents

1. [Goals, baselines and acceptance](#1-goals-baselines-and-acceptance)
2. [Model facts](#2-model-facts)
3. [Machine facts](#3-machine-facts)
4. [What other GLM-5.3-Flash implementations measured](#4-what-other-glm-53-flash-implementations-measured)
5. [Performance model](#5-performance-model)
6. [Design overview](#6-design-overview)
7. [Artifacts, import and expert volumes](#7-artifacts-import-and-expert-volumes)
8. [Ownership in the engine](#8-ownership-in-the-engine)
9. [Decode pipeline](#9-decode-pipeline)
10. [Expert residency: tiers, cache and placement](#10-expert-residency-tiers-cache-and-placement)
11. [Multi-SSD expert streaming](#11-multi-ssd-expert-streaming)
12. [CPU expert engine](#12-cpu-expert-engine)
13. [Sparse MLA attention and the pooled indexer](#13-sparse-mla-attention-and-the-pooled-indexer)
14. [Kimi Delta Attention layers](#14-kimi-delta-attention-layers)
15. [Manifold-constrained hyper-connections](#15-manifold-constrained-hyper-connections)
16. [Speculative decoding](#16-speculative-decoding)
17. [Prefill](#17-prefill)
18. [Prefix cache](#18-prefix-cache)
19. [Memory plans](#19-memory-plans)
20. [Numerics and qualification](#20-numerics-and-qualification)
21. [Frontend and serving](#21-frontend-and-serving)
22. [Configuration surface](#22-configuration-surface)
23. [Implementation plan](#23-implementation-plan)
24. [Documentation and authority changes](#24-documentation-and-authority-changes)
25. [Risks, open questions and rejected alternatives](#25-risks-open-questions-and-rejected-alternatives)
26. [Sources](#26-sources)

---

## 1. Goals, baselines and acceptance

### 1.1 Deliverable

The product supports `Glm5NextForConditionalGeneration` through the public Engine, `infernix`,
`infernix-serve`, `infernix_bench` and offline CausalScoring, with:

- the Text model (45 layers: 34 Kimi Delta Attention, 11 sparse MLA, 4-stream mHC);
- **DFlash2 speculation** (`--spec dflash2`) with the
  [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) drafter
  (§16.2) as the only drafter (no n-gram copy proposals: §16.4);
- the KV profiles that apply to MLA latents (§13.5), `bf16` and `int8` first;
- the hybrid prefix cache in `infernix-serve`;
- the GLM chat template, reasoning effort, and GLM tool calls in the OpenAI and Anthropic protocol
  surfaces (§21);
- routed experts on the shared offload tier in **any ratio of VRAM : RAM : SSD** —
  `--experts vram=…,ram=…[,ssd=…]` or counts, a placement file, switched at runtime through
  `POST /admin/expert-placement` (MoE §4.8) — with this family's expert records in **expert
  volumes on one to four NVMe drives** (§7.3), optionally striped 12 ways per record for the
  multi-drive machines (§11);
- an **opt-in approximate expert mode** (§9.8), off by default and labelled wherever it is on;
- the default NVFP4 artifact, converted by one recipe (§7.2).

**Fidelity.** Every tensor is imported bit-exactly: the experts' packed FP4 words, their FP8 E4M3
per-16 scales and FP32 global scales, and the BF16/FP32 dense tensors as stored. The routed experts
are computed in one canonical **W4A16** arithmetic (§9.5) — FP4 weights dequantized by their stored
scales against BF16 activations, fixed-order FP32 accumulation — the arithmetic and record layout
Flash-Next already runs (FN §6.3), so a routed expert's output is the same bits on the GPU, on the
CPU and from any tier (the placement-invariance principle of MoE §2), and a dense layer's output is
the same bits on its decode and prompt routes. (The checkpoint declares FP4 input activations and
vLLM runs it W4A4; this engine chooses A16 to reuse the built kernels — strictly more precise on
the activation side; the execution gate of §20.3 checks the engine against the artifact's own
decoded weights.) No recipe changes a weight: the
artifact as its producer quantized it is the model.

**The quantization's quality is the card's published numbers.** The card measured perplexity
2.894 against 2.788 for the FP8 source (+3.8 %), mean KL 0.073 and top-1 agreement 91.7 %
(upstream, §7.4); the engine does not re-measure it (owner decision, 2026-11-23), and the gates
of §20.3 qualify the engine's arithmetic against the BF16 reference forward on the same
checkpoint.

**Vision is in scope** (owner, 2026-11-23). The checkpoint holds the 24-block tower in BF16
(347 tensors, 1.13 GB, unquantized; the FP8 source would only be less precise). The converter
stores it as the `vision` component, every tensor BF16 as the checkpoint keeps it, beside its
`processor_config.json` resource. The binder refuses `--vision` until milestone M9 (§23) loads
the tower, held in pinned host memory as Qwen3.8-Flash-Next's `--vision-offload` does, and adds
the image and video prompts after the text engine works.

### 1.2 Baselines

All comparisons run on the target machines with the same prompts, output caps and sampling.

| Baseline | What it shows |
|---|---|
| **Project Maya** (`mw00/project-maya`, Strata's GLM-5.3-Flash engine; §4.2) with its own quants, Maya-M (116 GB) and Maya-L (156 GB), where they run | **The main baseline**, compared as each engine runs in practice (each with its own quantization), as Flash-Next was compared with Strata. Maya-M's experts (≈ 8.6 MB) are smaller than this design's 14.16 MB; its published KL (0.093 vs BF16 on alesha-pro's table, §7.4) sits above the default artifact's card numbers (0.073 vs FP8, §7.4), so the comparison reports quality beside speed |
| **strata-glm** (§4.3) with NVFP4 experts | The only published engine measured on this model's NVFP4 format, on a machine like the 5090 machine (16-18 tok/s chat on an RTX 5090 with 128 GB and x16); a token-id CLI, so it is compared on decode and prefill only |
| **ExLlamaV3** (the EXL3 format's producer) and adrienbrault's offload fork, with the EXL3 builds where they fit | A rejected alternative format (§25.4) kept as a survey point: identical weights across engines, so speed compares without a quality footnote; MikeRoz's 2.51 bpw build (≈ 105 GB) is the one that fits the 5090 machine's VRAM + RAM |
| **KTransformers** (GLM-5.3-Flash tutorial; CPU/GPU hybrid MoE; AMX/AVX-512 kernels) | The best-known consumer offload engine for this model; its GLM-5.3 numbers are recorded in §4.1 with their hardware |
| **llama.cpp / ik_llama.cpp** with `--override-tensor` expert offload, GGUF of the same model | The common community path; streams from page cache with mmap when RAM is short |
| **FreeToken** (§4.1) | The offload engine whose Flash-Next numbers FN §3.3 used as a reference; its port list includes `RedHatAI/GLM-5.3-Flash-NVFP4`, the same layout as the default artifact |
| The engine's Flash-Next on the production machine | Not a quality baseline (another model); it is the reference for what each mechanism achieved on the same card |
| **vLLM `glm53-flash` image / SGLang**, all experts resident, on a large GPU | Generic-framework all-resident ceiling; runs the same W4A4 checkpoint; not runnable here |

### 1.3 Targets

The 2026-10-11 targets were set for the EXL3 builds and are **stale** with the 2026-11-23 artifact
decision: the NVFP4 artifact moves more bytes per token (14.16 MB per record instead of 9.47 MB,
and BF16 dense instead of 5-bit), and the model's central decode on the 5090 machine falls to
≈ 11 tok/s with striped volumes (§5.3), below the old Must. The owner re-sets both machines'
targets from M4's first
end-to-end numbers; until then the model column below is the working expectation, not a commitment.

Primary configuration (model, §5.3-§5.4): the default NVFP4 artifact, `--kv-dtype bf16` (MLA
latent cache), `--max-context 131072`, C = 1, the production machine's default placement
(vram ≈ 47 %, ram the rest, no SSD, §19.1).

| Metric | Production machine, model | 5090 machine, model |
|---|---:|---:|
| Plain decode, warm cache, C = 1, non-repeating session (§20.4) | ≈ 50 (35-65) tok/s | ≈ 11 (8-15) tok/s with striped volumes; ≈ 8 unstriped |
| Best speculative mode, same corpus | ≥ plain | ≥ plain |
| Prefill, 32K fresh prompt | ≈ 6,000-8,000 tok/s | ≈ 4,700-5,500 tok/s |
| Time to first token, 4K fresh prompt | ≈ 2.5 s | ≈ 6-6.5 s |
| Decode at 128K actual context, relative to 4K | ≥ 85 % | ≥ 85 % |
| Execution fidelity against the reference forward, §20.3 | within the gate | within the gate |

The acceptance report gives the median and range of at least three runs, the worst case, the
expert service composition (VRAM hits, RAM CPU-served, RAM DMA, SSD reads, SSD bytes per token),
and the machine each number came from. A missed target is reported as missed; targets are not
changed after measuring.

These expectations sit far below Flash-Next's (measured 380 tok/s one-request decode at
`vram=100%` on the production card, FN §1.3) because this model reads ~16.7 GB of BF16 dense
weights per token (§2.3) and 14.16 MB per miss, and misses far more
often at equal cache share (§5.2). That is a property of the model and the artifact on this
hardware, not a design choice; §5.5 lists the levers (an 8-bit dense recipe is now the largest,
§25.3).

### 1.4 Non-goals

- Multi-GPU, AMD GPUs, GPUs other than `sm_120a`.
- Any change to the routed experts' weights, dynamic top-k, expert skipping, or lower-precision
  copies of cold experts in the default configuration (they change the model; FN §3.2 "considered
  and rejected"). The one exception is the opt-in, labelled approximate mode of §9.8.
- Requantizing or converting an artifact's weights to another format at import.
- An SSD cache or spill file: the expert volumes are written once by the converter and only read
  afterwards (MoE §4.7's rejection).
- A generic model graph or plugin layer. The offload tier is shared through plain composition —
  no base class, no virtual model interface (MoE §2) — and this family adds to it only what §8.2
  lists.

---

## 2. Model facts

### 2.1 Configuration (upstream)

From `zai-org/GLM-5.3-Flash` `config.json` and Transformers `configuration_glm5_next.py`. M0 checks
the artifact's own `config.json`, tokenizer and template files (not gated) against these.

| Quantity | Value |
|---|---|
| Layers | 45 + 1 MTP layer (index 45). Layer i is KDA when i % 4 ≠ 3, sparse MLA when i % 4 = 3: 34 KDA, 11 MLA (3, 7, …, 43) |
| Hidden width | H = 4096; RMSNorm eps 1e-5, weight multiplies (no unit offset) |
| Residual | mHC, 4 streams (16,384 wide), two mixers per layer, 20 Sinkhorn iterations, eps 1e-6 |
| KDA | 64 heads, head dimension 128, short conv kernel 4 (per q/k/v, SiLU), gate lower bound −5, FP32 state 64 × 128 × 128 per layer. Separate `q_proj`, `k_proj`, `v_proj` [8192 × 4096] and separate `q_conv1d`, `k_conv1d`, `v_conv1d` weights [8192, 1, 4] (upstream) |
| Sparse MLA | 64 heads; q_lora_rank 1536; kv_lora_rank 512; qk_nope 256, **qk_rope 0 (no positional encoding anywhere in the text model)**; v_head 256; softmax scale 256^−½ |
| Indexer (every MLA layer `full`) | 32 heads × 128, LayerNorm keys, softmax-gated pools of 4 tokens with a learned positional bias, relu scores weighted per head, top 512 pools (2,048 tokens) plus the incomplete tail |
| MoE | Layers 0-2 dense SwiGLU (I = 12,288); layers 3-44: 288 routed experts, top-8, width 2,048, 1 ungated shared expert of width 2,048 |
| Router | FP32 logits, sigmoid scores; selection on scores + `e_score_correction_bias`; weights = unbiased scores of the 8, normalized, × 2.5 (`noaux_tc`, n_group = topk_group = 1) |
| SwiGLU clamp | gate ≤ 10, up ∈ [−10, 10] (`swiglu_limit` 10), on routed, shared and dense MLPs |
| Vocabulary | 154,880; untied `lm_head`; final RMSNorm after the unweighted mean of the 4 streams |
| Context | 1,048,576 positions (no RoPE, so no YaRN question) |
| MTP layer | enorm, hnorm, `eh_proj` [4096 × 8192], one sparse-MLA block with its own indexer and MoE (288 routed + shared), `shared_head.norm`, the main `lm_head`; no hyper-connections (no `hc_*` tensors on layer 45). Stored in BF16 in the default artifact, including its 288 routed experts (§7.1). `index_share_for_mtp_iteration`: chained draft steps reuse the first step's selection. Transformers ignores layer 45 on load (upstream); this engine binds the text component only and does not load it (§7.1) |

### 2.2 Mathematics (upstream, Transformers `modeling_glm5_next.py`)

Written once here so every Op and oracle uses the same statement. Shapes per token; R is the
residual, 4 streams of H.

**Embedding.** R_s = embed(t) for s = 0..3 (four identical copies).

**mHC mixer** (`Glm5NextTextHyperConnection`; `fn` BF16 [24, 16384] upcast to FP32, `base`/`scale`
FP32; the block input norm is an unweighted RMSNorm with eps 1e-5 over the flattened 16,384
values):

```text
f      = fn · (R_flat / rms(R_flat))            rms with eps 1e-5, no weight; f computed in FP32
pre    = sigmoid(f[0:4]  · scale0 + base[0:4]) + 1e-6
post   = 2 · sigmoid(f[4:8] · scale1 + base[4:8])
comb   = softmax_row(f[8:24] · scale2 + base[8:24] as 4×4) + 1e-6
comb   = comb / (colsum(comb) + 1e-6);  repeat 19×: comb /= rowsum + 1e-6; comb /= colsum + 1e-6
x      = bf16( Σ_s pre_s · R_s )                                     collapse in FP32, rounded once
... block (norm, attention or MLP) produces y ...
R'     = bf16(post) ⊙ y + bf16(comb)ᵀ @ R      the reference computes the update in BF16,
                                               matmul included; streams stay BF16
```

**Decoder layer.** `(post, comb, x) = attn_hc(R)`; `x = input_layernorm(x)`; `y = attn(x)`;
R = mix(post, comb, y, R); `(post, comb, x) = ffn_hc(R)`; `x = post_attention_layernorm(x)`;
`y = mlp(x)`; R = mix(...). Head: `logits = lm_head(norm(mean_s R_s))`.

**KDA layer** (`Glm5NextTextLinearAttention`):

```text
q, k, v = SiLU(conv4(W_q x)), SiLU(conv4(W_k x)), SiLU(conv4(W_v x))   each 64 × 128; decode runs
                                               causal_conv1d_update over the concatenated q|k|v
                                               channels (one conv state, 24,576 channels)
g_in    = W_fb (W_fa x) + dt_bias                                              64 × 128, FP32
log α   = −5 · sigmoid(exp(A_log) · g_in)                                      per channel
β       = sigmoid(W_b x)                                                       64
q, k    = l2norm(q), l2norm(k)  (FP32, eps 1e-6: x / sqrt(Σx² + eps));  q *= 128^−½
S_h     = S_h · diag(α);  S_h += β (v − S_h k) kᵀ;  o = S_h q                    FP32 state
o       = FP32 gated RMSNorm(o; w_o, eps 1e-5) · sigmoid(W_gb (W_ga x));  y = W_o o
```

This is exactly the existing `kimi_delta_attention` Op contract (lower bound in [−5, 0]), which is
built and qualified with no consumer yet (§14): the Op takes the raw gate input `g_in` with
`dt_bias` and applies the sigmoid itself, including β, so `kda_input_prep` need not pre-activate.

**Sparse MLA layer** (`Glm5NextTextAttention`, NoPE):

```text
c_q   = RMSNorm(W_qa x)                       1536        (also the indexer's query input)
q_h   = W_qb,h c_q                            64 × 256
c_kv  = RMSNorm(W_kva x)                      512         cached per token (the latent)
k_h   = W_uk,h c_kv;  v_h = W_uv,h c_kv       W_kvb = [W_uk; W_uv] per head, 64 × (256 + 256)
o_h   = softmax_j(q_h·k_h,j / 16) v_h,j  over the selected tokens j
y     = W_o concat_h(o_h)                     16384 → 4096
```

**Indexer** (`Glm5NextTextIndexer`), per MLA layer:

```text
k_t   = LayerNorm(W_k x_t; γ, β, eps 1e-6)                        128 (weight and bias)
z_t   = W_g x_t                                                   128 (index_kpool_compress_gate)
pool p covers tokens 4p..4p+3 (pools start at the sequence's first valid key);
key_p = Σ_{i<4} p_i ⊙ k_{4p+i}   p_i = softmax_i(z_{4p+i} + ape[i]) cast to the key dtype (BF16),
                                 the weighted sum in BF16 — the reference's pooled keys are BF16
q_h   = W_qb_idx c_q                                              32 × 128
w_h   = (W_w x)_h · 32^−½
score_p = Σ_h w_h · relu(q_h · key_p · 128^−½)                     a pool is selectable only when its
                                                                   last token is visible
selected = top-min(512, P) pools by score  ∪  the visible incomplete tail (up to 3 tokens)
```

While the visible context is at most 2,051 tokens every token is selected and the layer equals
dense causal attention. Ties are broken by lower pool id in the engine (Transformers' `topk` leaves
them unspecified; recorded in §20).

**MoE** (`Glm5NextTextMoE`): `s = sigmoid(W_r x)` (FP32 logits from the FP32 `gate.weight`);
`top8 = argtop8(s + bias)` (the group mask is a no-op: n_group = topk_group = 1);
`w_i = 2.5 · s_i / (Σ_{top8} s + 1e-20)`; `y = Σ_i w_i E_i(x) + E_shared(x)` with
`E(x) = W_d (SiLU(min(W_g x, 10)) ⊙ clamp(W_u x, −10, 10))` — the clamp applies to the dense,
shared and routed MLPs alike.

**NVFP4 expert** (upstream, compressed-tensors `nvfp4-pack-quantized`; §9.5 gives the arithmetic):
`W = dequant(W_packed, W_scale, W_global_scale)` with two FP4 E2M1 values per byte, one FP8 E4M3
scale per 16 inputs and one FP32 global scale per matrix; the engine computes `y = W x` at BF16
activations (W4A16, §9.5).

The precision details above (BF16 residual update, BF16 pooled keys, the FP32 norms and the clamp
scope) were read from Transformers `modeling_glm5_next.py` on 2026-11-23 (upstream); where the
reference's choices look unintentional they are still followed, and §20.3's gates measure the
engine against them.

### 2.3 Parameters and bytes (upstream shapes and dtypes; model elsewhere)

| Item | Parameters | Bytes stored (NVFP4 artifact) | Bytes read per decode token |
|---|---:|---:|---:|
| One routed expert: gate, up [4096 → 2048] and down [2048 → 4096], FP4 packed (2 per byte) + FP8 E4M3 scale per 16 inputs + FP32 global scale | 25,165,824 | **14,155,776 = 13.5 MiB = 3,456 × 4 KiB** (§7.2) | — |
| Routed bank, 42 × 288 | 304.4 B | **171.2 GB (159.4 GiB)** | 336 experts: **4.76 GB** |
| Dense text: KDA projections 4.68 B, MLA 1.37 B, shared experts 1.06 B, dense MLPs 0.45 B, routers FP32 0.05 B, mHC/conv/`A_log`/`dt_bias`/norms ≈ 0.05 B | 7.66 B | **≈ 15.4 GB** (model: BF16 at 2 B per parameter from §2.1's shapes, FP32 where the vendor kept it) | same |
| `lm_head` (BF16, untied) | 0.63 B | 1.27 GB | whole matrix |
| Token embedding (BF16) | 0.63 B | 1.27 GB | one row |
| MTP layer 45, BF16: 288 routed experts 7.25 B, MLA block + `eh_proj` + norms ≈ 0.19 B | 7.44 B | ≈ 14.9 GB (experts 14.5 GB, dense ≈ 0.4 GB) | — (stored in the artifact, not loaded, §7.1) |
| Vision tower (24 blocks, BF16) | 0.56 B | **≈ 1.13 GB** (BF16 as stored; the converter writes it as the `vision` component with its processor config, §7.1) | — (M9, §1.1) |
| **Whole artifact** (index `total_size`) | | **191.0 GiB in 65 shards** | |

The dense parameter counts are the §2.1 shapes summed (KDA: q/k/v/o [8192 × 4096]-class plus the
small projections; MLA: q_a, q_b, kv_a, kv_b, o, indexer); they match the 321 B parameter total the
EXL3 builds' index published for the same architecture, which checked that reading.

Consequences, before any mechanism is chosen:

- **Dense BF16 reads rival the experts.** ≈ 16.7 GB of dense weights (including `lm_head`) per
  token against 4.76 GB of all-hit routed weights: an all-hit token reads ≈ 21.8 GB and takes
  ≈ 14.5 ms at the 5090's sustained ≈ 1.5 TB/s (estimate; the budget Flash-Next's kernel plans
  used, FN §8.10) or ≈ 12.2 ms at the production card's measured 1.79 TB/s (FN §8.3) — an all-hit
  ceiling of ≈ 69 / 82 tok/s (model). The EXL3 builds' 5-bit dense classes read 5.45 GB, so the
  BF16 dense triples that term; an 8-bit dense recipe is now the largest single lever (§25.3).
- Neither machine holds the bank in VRAM: **≈ 7.5 % of the experts fit in the 5090's frame pool,
  ≈ 47 % in the production card's** (§19.1); the host holds the rest (§19.2), so the SSD tier is
  empty at the production machine's default placement (182 GiB of usable RAM takes the complement
  with ≈ 90 GiB spare) and carries roughly half the bank's capacity share on the 5090 (§5.2).
- **A miss moves 14.16 MB**, 5.1× Flash-Next's 2.76 MB record. At the 5090 machine's x8 link
  (27.6 GB/s measured, FN §4.2) that is 0.51 ms per miss; at the production machine's x16 (model:
  2× the measured x8 rate; strata-glm measured 50.8 GB/s on an x16 5090, §4.3), 0.28 ms; from one
  990 PRO at QD 1 ≈ 2.3-2.4 ms (measured — the §3.3 probe used exactly this record size).
- **The KV cache is small.** One MLA latent is 512 values per token and layer: 11 MLA layers × 1 KiB =
  11 KiB per token in BF16 plus 64 B of pooled index key per layer. 128K tokens of BF16 latents are
  1.5 GB, so a long context costs the expert cache far less than Flash-Next's GQA KV did.
- **The recurrent state is larger.** 34 × 64 × 128 × 128 × 4 B = 136 MiB of FP32 KDA state per
  sequence, read and written every token (285 MB of traffic per token at C = 1).
- **Expert decode is cheap** (§9.5): a table lookup and a scale multiply per weight against the
  EXL3 trellis's 4-5 integer instructions; the GPU route stays memory-bound, and the CPU lane is
  dequantize-then-dot rather than decode-bound (§12.2).
- **The MTP layer's ≈ 14.9 GB of BF16 weights stay in the artifact and are never loaded** (§7.1):
  no drafter runs on them.
- **No n-gram table.** The official `zai-org/GLM-5.3-Flash` config has no n-gram keys, its
  76,108-tensor index holds no table, and neither the card nor a web search mentions one (checked
  2026-11-23). There is nothing to place on the SSD like Flash-Next's n-gram volume (FN §6.4);
  GLM has no n-gram drafting at all (§16.4).

---

## 3. Machine facts

### 3.1 The two platforms

**Production machine** (the machine this repository builds and serves on; AGENTS.md): one RTX PRO
6000 Blackwell, `sm_120a`, 96 GB, 188 SMs, DRAM 1.79 TB/s (measured, Flash-Next, FN §8.3), PCIe
Gen5 x16; 192 GB of host RAM (182 GiB usable); an AMD Ryzen 9 9950X3D2 host (16 cores / 32 threads,
AVX-512 with VNNI and BF16, `lscpu` 2026-11-02), whose W4A16 expert cost M0 measures before any
constant depends on it; CUDA 13.4, Ubuntu 26.04; one Gen5 NVMe (T710) holding the models, measured
at 6.36 GB/s at queue depth 1 and 13.35 GB/s at depth 8 on Flash-Next's 2.76 MB records
(2026-11-02, MoE §8.2). Its rate at this family's 14.16 MB records is unmeasured; M0's probe (§23)
measures it. The shared tier's placement matrix measured on this card (Flash-Next, 2026-11-14) is
MoE §8.2's: half the experts in RAM decode at 257 tok/s, half on SSD at 63, against 380
all-VRAM, one request each. The owner further reports that on Flash-Next a 50 % VRAM / 20 % RAM /
30 % SSD placement measured faster than VRAM + RAM (owner-reported observation, not a measurement
of this design); M7 re-measures it on GLM before any placement default follows from it.

**RTX 5090 machine** (measured 2026-10-10 unless marked; tools named per row):

| Item | Value | Source |
|---|---|---|
| GPU link | **PCIe 5.0 x8** (x16 capable); copy-engine H2D 27.6 GB/s; a Mellanox NIC takes the other 8 CPU lanes | measured, Flash-Next (FN §4.2) |
| VRAM | 32,607 MiB; Windows budget 31,419 MiB with no display on the 5090 (the monitor is on the iGPU) | measured, Flash-Next 5090 sessions (the implementation log at its base commit, FN §1.2) |
| Host | i9-13900K, 8 P + 16 E cores, AVX-VNNI (`vpdpbusd`, `vpdpwssd`) on both core types, no AVX-512; 2 × 48 GB DDR5-5800, 82-85 GB/s measured read | `Win32_Processor`; measured, Flash-Next (MoE §5.2) |
| CPU expert team (Flash-Next, W4A4, 2.76 MB records) | 52-55 µs per cold expert with 8 workers; 6 workers end-to-end optimum; DRAM shared with the PCIe stage | measured, Flash-Next (MoE §5.2) |
| RAM visible / available | 95.8 GiB / 79 GiB with that session's other processes | `Win32_OperatingSystem`, 2026-10-10 |
| Windows pinning ceiling | shared system memory 76.61 GiB; pins fail at 76.07 GiB with 5-6 GiB still free | measured on this host (MoE §4.11) |
| Windows power throttling | Must be exempted from EcoQoS (+69-162 % decode on Flash-Next, i9-13900K) | measured, Flash-Next (FN §8.1) |
| Commit limit | 291 GiB: a fixed 200 GB page file on `G:` | `Win32_PageFileUsage`, 2026-10-10 |

### 3.2 Storage topology (RTX 5090 machine, measured 2026-10-10)

| Drive | Device | Attachment | Free |
|---|---|---|---:|
| `C:` (system) | Samsung 990 PRO 2 TB | CPU PCIe 4.0 x4 (root port 0:6.0) | 243 GB |
| `D:` | Samsung 990 PRO 2 TB | Chipset PCIe 4.0 x4 (RP25) | 144 GB |
| `E:` (repository) | Samsung 990 PRO 2 TB | Chipset PCIe 4.0 x4 (RP13) | 371 GB |
| `G:` | Intel Optane P5800X 400 GB | Chipset PCIe 4.0 x4 (RP21) | 128 GB |

The chipset reaches the CPU over DMI 4.0 x8 (≈ 15.75 GB/s raw). Every drive but `C:` shares it.
The production machine has one NVMe (the T710); striping across drives is therefore a 5090-machine
configuration there only if a second drive is added.

### 3.3 Expert-sized reads (measured, RTX 5090 machine, 2026-10-10)

Probe: unbuffered (`FILE_FLAG_NO_BUFFERING`) synchronous reads at random 4 KiB-aligned offsets of
existing large files, one read in flight per thread, 2-4 s per cell, machine otherwise idle. The
whole-record cells use exactly this design's NVFP4 record (14,155,776 B), so the single-drive and
multi-drive tables are this record's measured rates. The sub-record table covers the striped
volumes' 1.125 MiB units (§7.2) only by interpolation between its cells; M0's C++ probe
(`tools/glm5_next_probe/`, §23) measures the unit size and the T710 on the production machine.

Single drive, whole 14.16 MB records:

| Drive | QD 1 | QD 2 | QD 4 | p50 at QD 1 |
|---|---:|---:|---:|---:|
| `C:` 990 PRO (system) | 1.75 GB/s (first run; 5.6 at QD 2) | 5.64 | 5.59 | 7.0 ms |
| `D:` 990 PRO | 5.92 | 5.74 | 5.73 | 2.34 ms |
| `E:` 990 PRO | 5.47 | 6.03 | 5.74 | 2.35 ms (p99 7.8-41 ms: the repository drive is busy) |
| `G:` Optane P5800X | 5.97 | **7.13** | 7.13 | 2.24 ms |

Read size on one 990 PRO (`D:`), GB/s by queue depth:

| Read | QD 1 | QD 2 | QD 4 | QD 8 | p50 at QD 1 |
|---|---:|---:|---:|---:|---:|
| 6.75 MiB | 5.48 | 5.67 | 5.73 | 5.74 | 1.24 ms |
| 2 MiB | 4.27 | **6.46** | 5.82 | 5.74 | 0.47 ms |
| 1 MiB | 3.24 | 5.56 | **6.54** | 5.75 | 0.30 ms |
| 256 KiB | 1.46 | 2.85 | 4.59 | 5.83 | 0.17 ms |

Optane `G:` at 1 MiB: 4.61 (QD 1), 6.99 (QD 4), 7.03 (QD 8) GB/s.

Concurrent drives (QD 2 each, whole 14.16 MB records):

| Drives | Per drive, GB/s | Total |
|---|---|---:|
| `D:` + `E:` (both chipset) | 5.73 + 5.96 | **11.69** |
| `C:` + `E:` (CPU + chipset) | 5.74 + 6.01 | 11.75 |
| `C:` + `D:` | 5.76 + 5.73 | 11.49 |
| `D:` + `G:` (990 PRO + Optane, both chipset) | 5.71 + 7.02 | **12.73** |
| `D:` + `E:` + `G:` (three on the chipset) | 4.74 + 4.70 + 4.72 | 14.15 (DMI-bound) |
| `C:` + `D:` + `E:` | 5.78 + 5.74 + 6.06 | **17.57** |
| `C:` + `D:` + `E:` + `G:` | 5.59 + 4.57 + 4.54 + 4.56 | 19.27 |

What this decides:

- **Two drives add almost perfectly** (2 × 5.8 → 11.7 GB/s), including two chipset drives: the DMI
  link binds only from the third chipset drive (≈ 14.2 GB/s for all chipset drives together).
- **A third drive pays only if it bypasses the chipset** (`C:`): 17.6 GB/s for `C:` + `D:` + `E:`.
  `C:` is the system drive with page-file and OS traffic; it is supported but not the default.
- **A drive saturates with 1-2 MiB reads at QD 2-4, and with 256 KiB reads at QD 8.** The striped
  volume's 1.125 MiB units (§7.2) sit at the measured 1 MiB cell; six to eight in flight per drive
  should hold a 990 PRO at ≈ 5.6-5.8 GB/s and the Optane at ≈ 7 (model, interpolated between the
  1 MiB and 2 MiB cells). A 14.16 MB record striped over `D:` + `G:` then lands in ≈ 1.1-1.3 ms
  (model), so computation can start before the record is complete (§11.4).
- **Latency, not just bandwidth.** One whole record from one drive takes 2.24-2.35 ms at QD 1
  (measured, the p50 column above); striped over two, ≈ 1.2 ms (model). That is about seven layers
  of GPU pre-routing work (§9.3), so every SSD read that is not predicted ahead stalls decode
  (§11.5).

---

## 4. What other GLM-5.3-Flash implementations measured

The shared machinery's evidence — Flash-Next's traces, the policy replay study, the divisor and
split measurements — is MoE §2 and §4-§8; Flash-Next's own measurements are in its document. This
section keeps only the GLM-5.3-Flash-specific external evidence this family's choices lean on.
Numbers are the sources' own, none reproduced here; read 2026-10-10/12 as text from their
repositories, papers and READMEs ([M] = the source reports a run, [C] = a claim without one).

### 4.1 Other engines

| Engine | Mechanism | GLM-5.3-Flash result |
|---|---|---|
| **ExLlamaV3** (turboderp, MIT; the EXL3 format's producer) | QTIP-style trellis codes in 16 × 16 tiles, codebooks `3inst`, `mcg` and `mul1`, 128-point Hadamard rotations with sign-scale vectors; GEMV kernels up to 144 rows, reconstruct-then-HGEMM above; MoE kernels including a cooperative fused gate-up-activation-down kernel; FP16 activations | The rejected alternative's format (§25.4); EXL3 3.05 bpw (base) measured mean KL 0.047 against BF16 in alesha-pro's table (§7.4) |
| **ExLlamaV3 offload fork, adrienbrault** (2 × RTX 5090 at x8, Ryzen 7 9800X3D: 8 Zen 5 cores with AVX-512, 64 GB DDR5-6000) | EXL3 2.05 bpw; 96-102 experts per MoE layer computed on the CPU by 8 threads, hot experts exchanged onto the GPUs at runtime | [M] 64.8 tok/s at one stream (median of three boots), 74-98 tok/s total at 2-4 streams; MTP 1.741 tokens per step; prefill ≈ 2,000 tok/s at 32K-131K; **the CPU worker reads cold EXL3 experts at ≈ 61 GB/s of a 63 GB/s ceiling**: on 8 Zen 5 cores with AVX-512, EXL3 decode keeps up with DRAM |
| **FreeToken** (arXiv 2608.16157) | One global LRU expert cache; each layer's m misses split into q* ≈ m · B_pcie / B_host DMA fills and m − q* CPU-computed experts; NVMe tier an unmerged Linux-only PR | Listed as supported (`RedHatAI/GLM-5.3-Flash-NVFP4`); no published number |
| **FreeToken port, anoane** (RTX PRO 6000 96 GB, x16, 160 GiB host) | All experts pinned (≈ 95 GiB), 17 layers fully VRAM-resident plus a 576-slot LRU; FP8 non-expert weights; b12x W4A16 kernels | [M] decode 22.2-22.7 tok/s, prefill 703 → 2,853 tok/s (chunk 8192); MTP slower (≈ 12 tok/s, with a corrupt-output bug) |
| **FreeToken ox-boost patch set** (RTX PRO 6000, DDR5-3600) | Router-score "soft" cache fed by near-miss experts (ranks just below the top 8) beat LRU; decayed LFU **lost** ("drift-shaped locality"); hot resident layers; next-layer top-4 prefetch; FP8 non-expert weights | [M] decode 29.2 → **~38 tok/s** (36.8 at 128K); soft cache 38.1 → 33.7 misses per token; hot layers +29 % (miss rate 0.24 → 0.157); prefetch P = 4 +8 %, P = 6-8 slower; FP8 dense +28 % decode; MTP verify correct but −13 % |
| **strata-glm** (sergqwer; archived; §4.3) | NVFP4 experts repacked for ggml, int8 dense, static VRAM hot set + LRU, pinned RAM tier, mirrored `experts.bin` on two drives, all misses DMA'd, next-layer predictor | [M] the only published GLM-5.3-Flash run on a machine like the 5090 machine: decode 16-18 tok/s chat, 12.4-12.9 code; ≈ 13 % of experts in VRAM, 58 % in RAM, 22.2 disk reads per token (code) |
| **KTransformers** (SOSP'25) | Attention and shared experts on the GPU, routed experts on the CPU (AMX prefill, AVX-512 decode); "expert deferral" changes the mathematics | AMX and AVX-512 are absent on the i9-13900K; deferral is out of scope (§1.4) |
| **llama.cpp / ik_llama.cpp** | `--n-cpu-moe` / `-ot exps=CPU`; mmap of a larger-than-RAM model pages from SSD with 111× read amplification on interleaved GGUF (192 pages per expert after an expert-contiguous relayout) | No 5090 number found |
| **nvmoe** (llama.cpp patch) | VRAM / pinned 2Q-SLRU / NVMe tiers with io_uring; lossy pruning of low-probability experts | [M] 1.8-8.9 tok/s for a 2-bit GGUF on a 16 GB laptop GPU |
| **vLLM `glm53-flash` image, SGLang** | All experts resident; FlashMLA sparse kernels are SM90/SM100 only, so sm_120 runs community ports or dense attention | Reference logits only |

What this design does with it:

- **Quantized experts can be computed on a desktop CPU at DRAM speed** (adrienbrault's EXL3 on
  AVX-512). W4A16 dequantize-then-dot costs less per weight than EXL3 trellis decode, so this
  bounds the CPU lane from above; the 13900K has no AVX-512, so the lane here is modelled
  compute-bound (§12.2) and M0 measures it first — on both machines' hosts.
- **Router near-miss scores as a cache signal** (ox-boost's soft cache) is evidence specific to
  this model; it enters M7 as a candidate VRAM policy beside LFRU (§10.2).
- **Decayed LFU lost on this model's drift-shaped locality** (ox-boost). The shared RAM tier ranks
  by decayed LFU (MoE §4.6); M7 re-runs that choice on GLM traces before freezing (§10.4).
- **MTP over PCIe-transferred misses was slower; over CPU-computed misses it paid** (1.74 tokens
  per step, adrienbrault's fork). This engine does not run MTP; DFlash2 is its only drafter (§16),
  and it verifies only when the cost model predicts a gain (§16.3).
- **No sm_120 sparse-MLA or indexer kernels exist.** They are new engine Ops (§13).
- **Two chipset drives "do not add"** (an ASRock forum RAID 0 report) is contradicted by §3.3: with
  application-level striping and enough reads in flight they add; only a third chipset drive hits
  DMI.

### 4.2 Project Maya: Strata's GLM-5.3-Flash engine

[Project Maya](https://github.com/mw00/project-maya) (MIT, read 2026-10-11 at `main`) is Strata
rewritten for this one model: a C++/CUDA engine with VRAM, pinned-RAM and SSD expert tiers, a CPU
expert lane, MTP drafting, multi-GPU layer splits, serving and its own quants. It publishes its
measurements, including the ones that failed. Its numbers are its own, on its hardware (mostly
V100s on PCIe 3.0 with 30-64 GB of RAM, plus user reports); none were reproduced here.

**Quants.** Maya's experts are 2-4-bit GGUF made from the FP8 release with GPTQ-style error
feedback: Maya-S 96.5 GB (IQ2_XXS/IQ2_S, KL 0.428 vs FP8 on its protocol), Maya-M 116 GB (IQ2_S /
IQ3_XXS-IQ3_S; KL 0.29-0.33 vs FP8 on its protocol, **0.093 vs BF16** on alesha-pro's), Maya-L
156.3 GB (KL 0.188 vs FP8), with a "30+ tok/s decode, ~2,000 prefill on an RTX 5090" user report
for Maya-M (RAM and link not stated). A Maya-M expert is ≈ 8.6 MB against this design's 14.16 MB,
and on the one table that measured comparable formats (alesha-pro's, §7.4) Maya-M's KL 0.093
against BF16 sits above the default artifact's published 0.073 against FP8 (§7.4), at smaller
experts.

**Measured on GLM-5.3-Flash by Maya, and what each decides here:**

| Maya finding | Evidence | Disposition |
|---|---|---|
| VRAM hit rate by capacity (one V100, IQ1_S experts): 44-65 % at 10.8 % of experts, 61-84 % at 21 %, 74-82 % at 31 %; 98 % at 70 % (two cards) | `2026-10-05-glm-single-gpu` | Basis of §5.2's hit-rate estimates |
| Policy replay: LRU 67 %, aged LFU 77.5 %, decayed frequency 78 %, a learned model 77 %, Belady 87 % | `PROGRESS.md`, `2026-10-06-glm-spec` | LFU-family candidates in M7 beside LFRU (§10.2); ox-boost found the opposite, so GLM traces decide |
| Admission threshold (`PROMOTE_MIN` 6 aged routes) stopped one-off experts churning the cache (decode 1.9× with the RAM tier) | `SETTINGS.md` | M7 sweeps an admission threshold (§10.2) |
| Background promotion (8 per token) and free-slot refill after a prompt (24 per token) | `SETTINGS.md` | The promotion pacing swept in M7 (§10.2) |
| Disk lookahead in decode: only ~40-45 % of disk misses predictable 1-4 layers ahead; exact but slower on one NVMe | `PROGRESS.md` | Lookahead off by default; adopted only if M7 shows a gain on two volumes (§11.5) |
| MTP on one GPU does not pay: consecutive tokens share ~1 of 8 experts per layer and none of their RAM pulls | `PROGRESS.md` | Fact kept; not applicable here — this engine does not run MTP (§16) |
| Draft routing computes only the 2 best-scored missing experts (`MTP_KEEP` 2): better acceptance than skipping all, never slower | `SETTINGS.md` | Not adopted: it served Maya's MTP drafter, which this engine does not run |
| CPU lane: a persistent spinning pool (0.31-0.36 ms per ~7 MB expert); one thread per physical core, never hyperthreads (21.8 → 13.3 tok/s: they starve the disk readers); work cut into ~48 pieces claimed in turn (14.3 → 17.1 tok/s); the CPU/PCIe split measured per expert count and persisted | `PROGRESS.md`, `SETTINGS.md` | Adopted in §12.3 (piece claiming, per-count split table, no SMT siblings); the 0.31-0.36 ms figure is the closest published CPU cost, for a record half this design's size (§12.2 scales from it) |
| SSD reads in 8 pieces per expert (4 on a Windows RAID: +15 %); whole-record reader threads gain 25-50 % on DRAM-less drives and lose 15 % on others | `SETTINGS.md`, `CHANGELOG.md` | §11.3's 12 units per record; calibration (§11.7) chooses units versus pairs |
| A mapped model file sends every direct read down a slower path on Windows; unmapping fixed it | `CHANGELOG.md` | Rule in §11.3: volumes are never mapped |
| Windows commit charge: VRAM allocations and the pinned tier both count against RAM + page file; 8 GB page file 2.4-4.1 tok/s against 8.4-11 with 64 GB; page file ≥ total VRAM + 9 GB | `docs/WINDOWS.md` | Already the shared ledger's law (MoE §4.10); this family's startup warning names the page file (§10.4) |
| A tier-accounting bug left 12 % of experts in no tier; fixing it took real chat 13.2 → 17.0 tok/s | `PROGRESS.md` | §20.4's tier audit |
| Keeping a prompt's disk reads in RAM for the answer gave more disk reads, not fewer | `PROGRESS.md` | §10.4: the prompt landing reserve is off; routes seed counts only |
| Prefill: deep disk read-ahead ring (64-96 experts); the next layer's disk experts read ahead; experts copied while attention runs; the least-routed RAM experts on the CPU; expert outputs accumulated into the layer output as a window fills (~40 KB per token instead of ~185); the last layer's output projection and FFN skipped for non-final prompt tokens without an MTP block; indexer scores as a register-tiled FP32 GEMM (78.8 → 5.6 ms) | `SETTINGS.md`, `PROGRESS.md` | Adopted in §13.3 and §17, with the exactness rules stated there |
| INT8 latent cache: 544 B per token and layer, no measurable quality change | `SETTINGS.md` | §13.5's `int8` profile moves ahead of `fp8` |
| KDA layers as CUDA graphs: +2.35 % | `docs/GLM-KDA-GRAPHS.md` | Already the design (whole-round graphs, §9.3) |
| `TRUNK_KEEP` map: skip missing low-ranked experts in late layers, Maya-M +23 % at KL 0.018 to its own full model (layers 3-7 most sensitive) | `SETTINGS.md` | The opt-in approximate mode of §9.8, with an `ssd`-only scope and a reproducible static rule added |

**What does not transfer:** multi-GPU layer splits and two-GPU speculative pipelining (one GPU
here), GGUF-only kernels, and Maya's quality bar (its quants trade KL 0.19-0.43 against FP8 on its
protocol for speed; the default artifact publishes 0.073 against FP8 on its card, §7.4).

### 4.3 strata-glm: the NVFP4 build on a machine like the 5090 machine

[strata-glm](https://github.com/sergqwer/strata-glm) (archived; read 2026-10-12: `README.md` and
the development log `docs/glm-port-plan.md`, in Ukrainian) ports GLM-5.3-Flash NVFP4 to the
Strata engine on an RTX 5090 with 128 GB of DDR5, two NVMe drives (a PCIe 5.0 9100 PRO at
≈ 13.4 GiB/s unbuffered and a 990 PRO at ≈ 7 GB/s) and a PCIe link measured at 50.8 GB/s. It takes
token ids and prints token ids. It measured everything with a deterministic teacher-forced KL
harness against a BF16-dense reference, and kept an offline policy simulator that predicted its
engine's disk reads (22.6 predicted, 22.2 measured per token). None of its numbers were
reproduced here.

**As a calibration of §5.3.** Fed strata-glm's own setup (VRAM hit rate 55-58 %, 22.2 disk reads
per token, every miss DMA'd at 50.8 GB/s, record time 0.75-1.5 ms), §5.3's per-layer model
predicts 14.6-18.4 tok/s for its code chat, against 12.4-12.9 measured: **the model is 15-45 %
optimistic on the one comparable published run**. §5.3 states the derated range.

| strata-glm finding | Evidence | Disposition |
|---|---|---|
| Quantizing the dense classes costs real KL, more at long context: int8-per-32 dense 0.0105 mean KL against 0.0066 for BF16 dense after a 2,600-token prompt; NVFP4 per class 0.016 (head) to 0.028 (MLA) | log, KL tables | §20.3's frozen texts include positions beyond 2,051 tokens and a long-prompt block; option O9 (§25.3) is the 8-bit dense recipe, behind the reference-forward gate (§20.3) |
| A naive 3-bit expert test (sign + 2 bits, UE4M3 per 16, no rotation or trellis): long-context KL 0.19 | log (`GLM_FAKE3`) | A warning that 3-bit experts can fail at long context: §20.3 runs the long-context protocol on the default artifact |
| Static VRAM hot set + LRU remainder: 15.66 → 17.15-17.31 tok/s; disk reads 30.6 → 22.2 per token | README, log | M7 candidate (§10.2) |
| Exclusive RAM/VRAM residency cost 158 write-backs per token (rejected there) | log | This design is exclusive with shadows (MoE §4.1); demotions are capped at 32 per boundary and M7 reports the copies (§10.1) |
| Mirrored volumes with dynamic per-unit routing: each 1.4 MB read to the drive that frees first; best static balance 10:7 on unequal drives | README, log | Option O8 (§25.3), A/B'd against weighted striping |
| Next-layer predictor from the current layer's streams: top-1/2/3 accuracy 95/86/77 %; no isolated gain published | README, log | Corroborates the rank-limited prefetch forms in M7 (§11.5) |
| CPU share of RAM misses: slower on a 50.8 GB/s link | log | Consistent with the divisor rule (MoE §5.6): the CPU takes fewer misses as the link gets faster; here the link is x8 |
| Opt-in skipping by routing weight: +30 % at KL 0.022 (≈ 3 % threshold, SSD-only); RAM misses below 8 %: +20 % (KL 0.044) | README, log | §9.8 gains a weight-threshold rule beside Maya's rank map; the approximate-mode milestone compares them at equal KL |
| Codes do not compress: NVFP4 code entropy 3.97 of 4 bits; lzma 97.6 % | log | Confirms §25.4 (no compressed volumes) |
| INT8 latents: KL 0.0049 → 0.0069 over 2,600 tokens | log | `bf16` stays the default KV profile (§13.5), with this cost stated |
| Prefill: a pool streams all 288 experts of a layer while attention runs; indexer queries in 2,048-token sub-chunks so 262K fits; ≈ 1,000 tok/s | README | §13.3 bounds the prefill score tile by query sub-chunks |
| Belady bound 73 % VRAM hit rate against 55-58 % reached at ≈ 13 % of experts | README | Corroborates §5.2's capacity estimates and the headroom left for policy |

---

## 5. Performance model

### 5.1 Per-token work at C = 1, plain decode

Let h_v be the VRAM hit rate, h_r the share served from RAM and h_s = 1 − h_v − h_r the share read
from SSD, over the 336 routed experts of a token.

| Resource | Work per token | Rate | Time |
|---|---|---|---:|
| GPU HBM: dense BF16 (incl. `lm_head`) + KDA state + KV | 16.7 + 0.29 + ~0.05 GB | ≈ 1.5 TB/s (5090, estimate, FN §8.10) / 1.79 TB/s (production, measured, FN §8.3) | ≈ 11.4 ms (5090) / ≈ 9.5 ms (production, model) |
| GPU HBM: hit experts | 336 h_v × 14.16 MB | same | 3.2 h_v ms (5090) |
| CPU lane: RAM- and SSD-served experts it computes | ≈ 0.25-0.50 ms per expert (W4A16, §12.2, estimate) | 6-8 P-cores (+ E-cores) on the 5090 machine; unmeasured on the production machine's Ryzen 9 9950X3D2 | |
| PCIe H2D: DMA'd misses (≈ 1/2 of RAM misses, §9.6) and SSD landings sent to the GPU | 14.16 MB each | 27.6 GB/s (5090 machine, measured) / 50 GB/s (production, model) | 0.51 / 0.28 ms each |
| Host DRAM: CPU reads + DMA source + SSD landings | ≈ 336 (h_r + h_s) × 14.16 MB | ≈ 80 GB/s (5090 machine, measured; production model, dual-channel DDR5) | **59 (h_r + h_s) ms (model)** |
| SSD: 336 h_s records | 336 h_s × 14.16 MB | 12.7 GB/s (`D:` + `G:` striped, measured §3.3) on the 5090 machine; none at the production machine's default placement | 374 h_s ms (5090 machine) |

**The binding term is the host DRAM service of RAM misses.** Both ways of serving a RAM miss — the
CPU lane reading the record and a DMA reading its source — cross host DRAM, so a token's host time
is ≈ 59 (h_r + h_s) ms (model) regardless of the CPU/link split; the split (§9.6) balances the CPU
cores against the link underneath that DRAM floor. At the estimated W4A16 cost (§12.2) the CPU lane
is DRAM-bound, not compute-bound, unlike the EXL3 design's decode-bound estimate. The GPU term is
≈ 12-15 ms; the SSD term is second on the 5090 machine and zero at the production machine's
default placement.

### 5.2 Hit-rate model

No routing traces of this model are public, and M7 records them before the policy is frozen. The
evidence that exists says **GLM-5.3-Flash has far weaker expert locality than Flash-Next**:

| Source | Capacity in the cache | Miss rate |
|---|---:|---|
| Flash-Next, LFRU replay (MoE §4.3) | 16.3 % / 32.6 % / 48.8 % | 18.4 % / 5.2 % / 1.4 % |
| **GLM-5.3-Flash**, FreeToken ox-boost port, RTX PRO 6000 96 GB (§4.1): LRU with 8 resident layers → hot resident layers → router-score "soft" cache | ≈ 47 % (≈ 75 GiB of experts) | 24 % → 15.7 % → ≈ 10 % |
| DeepSeek-V4-Flash, FreeToken paper, global LRU | 11 % | 39 % |
| **GLM-5.3-Flash**, Project Maya, one V100 at 16 / 24 / 32 GB, aged LFU (§4.2) | 10.8 % / 21 % / 31 % | 35-56 % / 16-39 % / 18-26 % |
| Same, VRAM + a 23 GB RAM tier | 58 % | 7.4-9.8 % reach the SSD |

At the same capacity share GLM misses ~7× more than Flash-Next. Two causes fit: 8 active of 288
(2.8 %, against 10 of 512) and the auxiliary-free bias, which keeps experts near-uniformly loaded.

**RTX 5090 machine** (estimate from that evidence, at §19's capacities: ≈ 900 frames = 7.5 %,
≈ 5,600 RAM slots = 46 %, together 54 % of the 12,096 experts, exclusive):

| Scenario | h_v | h_r | h_s | RAM-served per token | SSD reads per token |
|---|---:|---:|---:|---:|---:|
| Optimistic (policy and seeding gains of §10 land) | 0.60 | 0.30 | 0.10 | 101 | 34 |
| Central | 0.55 | 0.30 | 0.15 | 101 | 50 |
| Pessimistic | 0.50 | 0.28 | 0.22 | 94 | 74 |

**Production machine (estimate):** the frame pool holds ≈ 47 % of the bank and the RAM tier the
rest (§19.1), so h_s = 0; h_v ≈ 0.70-0.85, central 0.80 (estimate, interpolated on Maya's capacity
curve between its 74-82 % at 31 % and 98 % at 70 %), h_r the rest.

The 5090 machine's SSD carries 0.5-1.0 GB per token: 40-80 ms of striped two-volume drive time
(model). Striped volumes are worth ≈ 2× there (§5.3); the production machine has no SSD term.

### 5.3 Decode projection

A layer's misses are known only after its router, and the next layer needs its combine, so per MoE
layer the critical path is

```text
t_layer = t_pre + max(t_hits, t_host)
t_host  = min over n_c of max((n_c + k) · c_cpu,  (n_r − n_c) · c_dma,  k · t_record + c_cpu,
                               (n_r + k) · 14.16 MB / B_dram)  + 30 µs
t_token = Σ_l t_layer + t_dense_layers + t_head
```

with the layer's n_r RAM misses and k SSD misses multinomial over its 8 experts (exact
expectation), n_c of the RAM misses computed by the CPU lane and the rest DMA'd, SSD records
computed by the CPU as they land, and B_dram the host DRAM read rate.

- t_pre: the layer's GPU work before routing (mHC, attention, mixer, router): ≈ 8.6 ms over the
  42 MoE layers on the 5090 machine at ≈ 1.5 TB/s (model — the BF16 dense weights tripled this
  term against the EXL3 design; §9.3); ≈ 7.2 ms on the production card (model, scaled by the
  measured 1.79 TB/s). Dense layers 0-2 and the head add ≈ 1.5 ms. Hit experts (≈ 9 µs each) hide
  under the host service except in layers with no miss.
- c_cpu: the CPU lane's time per expert, **estimated 0.25 / 0.35 / 0.50 ms** (optimistic,
  central, pessimistic: AVX-512, VNNI-only, scalar fallback; §12.2). M0 measures it first — on
  both hosts. At the central estimate an expert's DRAM read (14.16 MB at 80 GB/s ≈ 0.18 ms) is
  cheaper than its compute, so the lane sits near the DRAM/link boundary rather than deep in
  compute.
- c_dma = 14.16 MB / 27.6 GB/s = 0.51 ms per staged expert on the 5090 machine; 0.28 ms on the
  production machine's modelled x16 link.
- t_record: a whole record: 2.3 ms on one drive (measured, §3.3), 1.2 ms striped over two, 0.85 ms
  over three with `C:` (model from §3.3).
- Lookahead is off (the default, §11.5). With 34-74 SSD reads per token on the 5090 machine its
  ceiling is the gap to the no-SSD column below, of which Maya's 40-45 % predictability could
  recover part.

Once the family's round bench exists, its speed of light carries this model as measured bytes
(`round_bytes`, [Decode profiling tiers](round-bench.md)) and the two must agree; no separate model
script is kept.

Plain decode, tok/s (model):

| Scenario, RTX 5090 machine | One volume | Two striped volumes (`D:` + `G:`) | Three volumes (with `C:`) | No SSD misses (whole bank in memory) |
|---|---:|---:|---:|---:|
| Optimistic | 11 | **15** | 16 | 34 |
| Central | 8 | **11** | 12 | 27 |
| Pessimistic | 6 | **8** | 9 | 21 |

The 5090 machine's default placement leaves ≈ 46 % of the bank to the SSD, which is why its plain
decode sits near the strata-glm number (16-18 tok/s chat with 128 GB of RAM and 71 % memory
coverage, §4.3) only at the optimistic end. Striped volumes are the 5090 machine's designed form
(later option, §23).

**Production machine (model, no SSD term):** with h_v = 0.85 / 0.80 / 0.70 (optimistic /
central / pessimistic), c_cpu 0.25 / 0.35 / 0.50 ms, c_dma 0.28 ms and B_dram 90 / 80 / 70 GB/s
(estimate; M0 probes the host), the formula gives ≈ 65 / 48 / 36 tok/s — central ≈ 50. The
host DRAM floor binds before the CPU's compute; M0 measures both.

**Calibration against the one comparable published run.** Fed strata-glm's measured setup (§4.3:
55-58 % VRAM hits, 22.2 disk reads per token, every miss DMA'd at 50.8 GB/s, record 0.75-1.5 ms
on their drives), the model's term sum lands well above that engine's measured 12.4-12.9 tok/s:
the model is optimistic by a wide margin on the one comparable published run, as it was in the
EXL3-era reading (15-45 %). M4's first end-to-end numbers re-fit the model's overhead terms before
M7's sweeps; §25.1 carries the risk. For comparison, the best public GLM-5.3-Flash numbers are
22-38 tok/s (§4.1) on a 96 GB card with every other expert in pinned RAM over x16, 65 tok/s on two
5090s with a lossier 2.05 bpw EXL3 build, and Maya's "30+" on an RTX 5090 with 8.6 MB experts
(§4.2).

**Speculation helps little on this path.** RAM- and SSD-served expert bytes per accepted token
stay about the same (MoE §6.3), and every rejected draft adds misses. DFlash2's drafter is a small
self-contained BF16 model (2.2 GiB, §16.2) whose verify columns ride the target's round, so its
cost is the proposal and the extra columns, not extra expert misses on the target's host path; the
cost model of §16.3 prices it, and M6 measures it.

### 5.4 Prefill projection

Prefill runs layer-major over a **walk span** of up to 64K tokens (MoE §7.1, §17.1): every
non-VRAM expert crosses the link once per span, while the GPU computes the previous layer.

| Term, 32K-token span | Work | Rate | Time |
|---|---|---|---:|
| Expert stream over the link (5090 machine; ≈ 92 % of the bank non-resident) | 0.92 × 171.2 GB = 158 GB | 27.6 GB/s measured | 5.7 s |
| Same, production machine (≈ 53 % non-resident, §19.1) | 0.53 × 171.2 GB = 91 GB | 50 GB/s model | 1.8 s |
| Dense GEMMs and head, BF16 tensor cores | 2 × 8.3 B × 32,768 ≈ 540 T ops | ≈ 190-260 T ops/s on the 5090 (model: half the card's dense INT8 peak of ≈ 840 TOPS by vendor specification, unverified here, at 50-70 % efficiency); × 188/170 SMs on the production card | 2.1-2.8 s (5090); 1.9-2.6 s (production) |
| Routed experts, W4A16 wide route (BF16 tensor cores) | 554 T ops | same | 2.1-2.9 s (5090); 1.9-2.7 s (production) |
| MLA, indexer, KDA chunks | | | ~0.5 s |

Streaming and compute overlap, so a 32K prompt is compute-bound: the 5090 ≈ 6-7 s (**≈ 4,700-5,500
tok/s**, model) and the production card ≈ 4-5.5 s (**≈ 6,000-8,000 tok/s**, model). The BF16
tensor rate is the model's least measured input; M8 measures it before any prefill target is set.
A **4K prompt still touches every expert** (each sees ~114 tokens), so its time to first token is
a whole expert pass: ≈ 6-6.5 s on the 5090 machine and ≈ 2.4-2.6 s on the production machine
(model). Only the prefix cache (§18) and a faster link change that.

### 5.5 What moves the ceiling

| Lever | Effect (model, central) |
|---|---|
| **An 8-bit dense recipe** (§25.3: dense text and head quantized to 8 bits at conversion) | Dense bytes per token 16.7 → ≈ 9.6 GB: the all-hit ceiling rises ≈ 69 → 100 tok/s on the 5090 and the production decode term falls by ≈ 4 ms; the largest single lever, behind its own quality gate |
| **A 192 GB host on the 5090 machine** | The whole bank fits in RAM (159.4 GiB): h_s = 0; decode 11 → 27 tok/s (the no-SSD column) |
| **The 5090 machine at PCIe x16** (move the NIC to a chipset slot or remove it) | DMA'd misses at 2× the rate: decode +10-15 %; prefill streams halve |
| **A third expert volume on the 5090 machine** (`C:`) | Record time 1.2 → 0.85 ms (measured bandwidth, §3.3); decode +8 % |
| **Faster DDR5** | Now first-order: the host DRAM floor scales every RAM-served byte (§5.1) |
| **A CPU with AVX-512 or more cores** | Marginal while the lane is DRAM-bound; matters if M0 measures the W4A16 kernel compute-bound |

The design reports these as hardware advice in the user guide; nothing is tuned to x8.

---

## 6. Design overview

```text
┌────────────────────── one GPU: RTX PRO 6000 96 GB (or RTX 5090 32 GB) ──────────────────────┐
 │ dense BF16 ≈ 15.5 GiB · KDA state 136 MiB/lane · MLA latents + pooled index keys            │
 │ decode workspace · mailboxes · residency tables (MoE §4.1)                                  │
 │ ┌─ frame pool (MoE §4.2): ≈ 900 (5090) or ≈ 5,700 (96 GB) × 13.5 MiB, elastic ──────────┐   │
 │ │ cached experts · staging · landing frames (MoE §4.4) · leases (MoE §4.5)              │   │
 │ └───────────────────────────────────────────────────────────────────────────────────────┘   │
 └──────▲───────────────────▲───────────────────────────▲────────────────────────────────────┘
        │ copy engine       │ mapped mailboxes          │ copy engine (landed SSD records)
 ┌──────┴───────────────────┴───────────────────────────┴───────────┐   ┌──────────────────┐
 │ host RAM (182 GiB usable production / 96 GB on the 5090 machine)  │   │ expert volume 0  │
 │  RAM tier (MoE §4.6): pinned slots × 13.5 MiB + the landing ring  │◄──┤ (volume i of 1-4, │
 │  CPU expert team: W4A16 exact, from RAM or landing slots (§12)    │   │  one per drive)   │
 │  engine thread: policy at round boundaries, promotions           │   └──────────────────┘
 │  SSD read agent (MoE §4.7): one unbuffered queue per volume      │
 └───────────────────────────────────────────────────────────────────┘
```

One decode round at C = 1:

```text
embed(host row) → 4 streams
per layer l (45):
  mHC_attn (one kernel) → RMSNorm → KDA (q/k/v proj, conv, gates, recurrence, gated norm, o_proj)
                                     or MLA (q/kv latents, indexer, top-512, latent attention, o_proj)
  mHC post + mHC_ffn (one kernel) → RMSNorm → MLP:
     layers 0-2: dense SwiGLU
     layers 3-44: router(l) + predictors(l+1, l+2, l+3) → residency → GPU jobs | CPU jobs | SSD jobs
                  shared expert ‖ routed experts (narrow W4A16) ‖ CPU misses ‖ landed SSD records
                  combine (FP32, fixed order) → mHC post
head: mean of streams → norm → lm_head (FP32 logits) → sample → DFlash2 proposal (§16)
```

The ideas that carry the design:

- **Exact expert arithmetic makes placement free** (MoE §2): VRAM frame, staging frame, CPU from a
  RAM slot, CPU from a landing slot: the same bits. Policy can move work anywhere. This family
  needs no new arithmetic: the experts reuse Flash-Next's exact NVFP4 W4A16 route and record layout
  (§9.5), which the engine already qualifies (FN §6.3).
- **The tiered residency, placement and runtime switch are the shared tier's** (MoE §4): the
  engine thread applies the policies at round boundaries, the device only reads residency words,
  and any VRAM : RAM : SSD ratio is reachable at runtime.
- **The SSD tier is a read-only set of expert volumes**, one per drive, written once by the
  converter — this family's composition of the shared store (§7.3, §11). Records default to whole
  reads from one volume; 12-way striped records (every read uses every drive) are a later option
  for the multi-drive 5090 machine (§7.2).
- **SSD reads are made short, not guessed**: in the striped form a record's latency halves and
  partial records are consumed as their units land (§11.4). Lookahead reads of predicted experts
  are designed (§11.5) but off by default, since the one GLM measurement found them a net loss on
  one drive.
- **The host's service rate is the scarce resource**: everything is sized to minimize host-served
  expert work per accepted token: frames maximized, KV small, drafter experts bounded, promotions
  rationed (§10), and the host DRAM floor of §5.1 treated as the budget's first constraint.

---

## 7. Artifacts, import and expert volumes

### 7.1 The artifact

Read 2026-11-23 from [orcarouter/GLM-5.3-Flash-Uncensored-NVFP4](https://huggingface.co/orcarouter/GLM-5.3-Flash-Uncensored-NVFP4)
(`config.json`, `model.safetensors.index.json`, the card; metadata only, the weights were not
downloaded). Compressed-tensors `nvfp4-pack-quantized`, **191.0 GiB in 65 shards** (upstream), same
layout as `RedHatAI/GLM-5.3-Flash-NVFP4`. The abliterated edit is orcarouter's, baked into its
block-FP8 checkpoint `orcarouter/GLM-5.3-Flash-Uncensored-FP8` before quantization.

| Tensor class | Stored | Engine form |
|---|---|---|
| Routed experts, 42 × 288 (layers 3-44 only) | per expert and per matrix (`gate_proj`, `up_proj`, `down_proj`): `weight_packed` uint8 (two FP4 E2M1 per byte), `weight_scale` FP8 E4M3 (one per 16 inputs), `weight_global_scale` FP32 | Packed words and per-16 scales relaid out verbatim as `nvfp4_expert_rg16_v1` records at GLM geometry (§7.2) into the expert volumes (§7.3); the three global scales per expert ride the bank's binding |
| Everything else | BF16, dequantized from the vendor's block FP8; F32 kept where the vendor had F32: `hc_*_base`, `hc_*_scale`, `A_log`, `dt_bias`, router `gate.weight`, `e_score_correction_bias` | Direct |
| `hc_attn_fn`, `hc_ffn_fn` | BF16 [24, 16384] (the reference upcasts to FP32, §2.2) | Direct |
| KDA | separate `q_proj`, `k_proj`, `v_proj` [8192 × 4096] and `q_conv1d`, `k_conv1d`, `v_conv1d` [8192, 1, 4], BF16 | Direct |
| MTP layer 45 | BF16 throughout, including its 288 routed experts, plus `eh_proj`, `enorm`, `hnorm`, `shared_head.norm` | Stored in the artifact as an `mtp` component (§7.3); **not loaded** — the engine binds the text component only |
| Vision tower | 24 blocks, BF16 | Stored as the `vision` component (every tensor BF16 as stored, plus the checkpoint's `processor_config.json` as the component's one resource); loaded from M9 (§1.1) |

The config declares FP4 input activations (vLLM runs the checkpoint W4A4); this engine runs the
experts at **A16** — BF16 activations — the exact W4A16 arithmetic Qwen3.8-Flash-Next uses
(FN §6.3), reusing its built kernels and record layout instead of a new codec (§25.4).

The converter refuses any compressed-tensors NVFP4 artifact whose quantized set or scale layout
differs from the above (other quantized classes, a scale group other than 16 inputs, missing
global scales) with the property named, until a track adds it (§25.3).

### 7.2 Recipe and expert record layout

**One recipe**, `glm5_3_flash_nvfp4_orcarouter` (`tools/convert/official_recipes.py`), imports the
compressed-tensors NVFP4 artifact as stored. The adapter `tools/convert/glm5_next.py` binds the
parameters; the expert banks are written by `tools/convert/sources/nvfp4_banks.py` (the bank import
Qwen3.8-Flash-Next's weight-only recipe uses), and the dense BF16/FP32 tensors pass through
unchanged. The DFlash2 drafter is a companion artifact (`--components dflash2 --source
dflash2=PATH`, recipe `glm5_3_flash_dflash2`). No dense recipe variants exist at first; an 8-bit dense
recipe is option O9 (§25.3).

**Record layout `nvfp4_expert_rg16_v1` at GLM geometry** (H = 4,096, I = 2,048; the layout is
[Flash-Next's](storage-layouts.md) with the geometry as a compile-time parameter of the Op, §8.3).
A work unit is a 16-input block of a 16-row group: 16 rows × (8 packed bytes + 1 scale byte) =
144 B. The record:

| Section | Content | Groups × blocks × 144 B | Bytes |
|---|---|---|---:|
| gate/up | row groups interleaved gate/up (2 × I / 16 = 256 groups), K / 16 = 256 blocks each | 256 × 256 × 144 | 9,437,184 |
| down | H / 16 = 256 row groups, I / 16 = 128 blocks each | 256 × 128 × 144 | 4,718,592 |
| **Record** | | | **14,155,776 = 13.5 MiB = 3,456 × 4 KiB** |

12,096 main records = **171.2 GB (159.4 GiB)** (model). Scales live inside the record (there is no
expert vector plane and no frame side entry — that device existed for EXL3's sign-scale vectors,
§25.4); every unit is 4 KiB-aligned. The layout id carries no geometry: H and I are properties of
the bank, recorded in its binding and compiled as one of the Op's geometry instances (§8.3).

**Striped records are optional.** The default volume stores each record whole. The striped form
splits a record into **12 units of 1,179,648 B (1.125 MiB = 288 × 4 KiB)** — units 0-7 the gate/up
section (32 of the 256 gate/up row groups each), units 8-11 the down section (64 row groups each) —
spread over the volumes so every read uses every drive (§11). It exists for multi-drive machines; the production
machine has one NVMe, so its volumes are unstriped (§7.3).

### 7.3 Import and expert volumes

**No supplement.** Every tensor this family loads is in the artifact; the EXL3 plan's fetch of
missing indexer pooling tensors from `zai-org/GLM-5.3-Flash` is gone. (The official
`zai-org/GLM-5.3-Flash` checkpoint remains a source for reference forwards, read directly like any
safetensors source.)

The routed banks do not live in the `.infernix` file set. They live in **N expert volumes**
(`<name>.xv-<i>-of-<N>`, 1 ≤ N ≤ 4), each a plain file meant for its own drive, in the role the
n-gram volume has for Flash-Next (FN §6.4) — except that this artifact has no n-gram table to
place (§2.3), so the volumes are the only side files. The container's continuation volumes must
sit beside the entry file ([container spec](artifact-container.md) §2.1, §3.3), so they cannot be
spread over drives; the expert volumes are located by flag instead (§22).

**Placement, unstriped default.** A weight vector w (integers, Σ w_v = 12,096-weighted shares;
default equal) distributes whole records over the volumes: record r of a bank goes to volume
P[(r + b) mod N]-th in the interleaved sequence, records numbered bank by bank (the 42 main
banks), expert by expert. Within volume v, bank b starts at
`base_{v,b} = 4096 + Σ_{b' < b} R_{b'} · w_v · record_{b'}` and record r sits contiguously at
`base_{v,b} + (r · w_v + k) · record_b`. Every offset is computable; there is no per-record table.

**Striped placement (optional, §7.2).** With `--striped-volumes`, each record's 12 units are
spread by the same rule at unit granularity (unit j of record r to volume P[(j + r) mod 12], the
fixed interleaved sequence per weight vector w over 12), so a record read touches every drive. A
mixed configuration (some volumes striped, some not) is refused.

**Header** (first 4 KiB of each volume): magic `NINFXV\0\1`, the artifact set's 16-byte id, volume
index and count, weight vector, the striped flag, bank count, and per bank its record bytes, unit
bytes (equal to the record bytes when unstriped) and record count, plus a CRC32C of the header.
The main artifact records the same geometry and the expected size of each volume, so a volume from
another artifact, another weight vector or a truncated copy is refused at startup.

**Sizes (model from §2.3's record bytes).** 12,096 main records = 171.2 GB (159.4 GiB); two equal
volumes 85.6 GB each; with w = (5, 7) 71.3 and 99.8 GB. The T710 must hold the one production
volume (171.2 GB) beside the main artifact. The main artifact itself carries the `mtp` component
as stored (≈ 14.9 GB, §2.3); the engine binds the text component only and does not load it (§7.1).

**Writing.** The converter writes the main artifact and all volumes in one run
(`--expert-volumes PATH,PATH,…`, `--expert-volume-weights 6,6`, `--striped-volumes`), streaming
each source shard once and writing each volume sequentially with unbuffered 64 MiB writes.
`python -m tools.artifact.restripe` rewrites the volumes for another N, weight vector or striped
flag from existing volumes, without the source. These two are the only writers; runtime never
writes a volume (MoE §4.7's rule: the SSD tier is the artifact, read in place, never an SSD cache
or spill). The one-time write is ≈ 172 GB, ≈ 0.015 % of a 990 PRO 2 TB's rated 1,200 TBW.

**How the volumes meet the shared SSD tier.** The tier reads the artifact in place through
`ExpertStore`'s file segments (one per record, two when a record straddles the container's part
files — `src/models/offload/expert_store.h`, MoE §3 and §4.7). Unstriped volumes are exactly
today's description: one segment per record. The striped form is the optional extension (§8.2):
`ExpertStore`'s segment count per record rises from 3 to 12 (a static array; the units are
4 KiB-aligned and adjacent-unit coalescing is unchanged), and the read agent owns one
`DirectReadQueue` per volume with per-drive depth caps (§11.3) — one volume today, so
`qwen4_exp`'s behavior is byte-identical.

**Disk plan.** 5090 machine: the source (191.0 GiB) on `E:` (371 GB free — the source does not fit
beside everything and is deletable after verification); the main
artifact (dense 15.4 GB + embedding 1.27 + `lm_head` 1.27 + the stored, unloaded `mtp` component
≈ 14.9 GB ≈ 33 GB) on `E:`; volume 0 on `D:`
(144 GB free) and volume 1 on `G:` (128 GB free), w = (5, 7), striped (§11.6). Production
machine: one unstriped volume (171.2 GB) and the main artifact on the T710; a second drive
unlocks striping and the multi-volume forms (§3.2).

### 7.4 Other artifacts and the quant survey

A survey of GLM-5.3-Flash quants with a published KL (owner request, 2026-10-11) originally made
the EXL3 3.05 bpw builds the default; the 2026-11-23 decision moved the default to the NVFP4
artifact and the EXL3 builds to the rejected/optional list (§25.4). The survey is kept as the
alternatives map. Hugging Face lists ~440 GLM-5.3-Flash repositories; these are the ones with a
published KL (read 2026-10-11 from their cards; none reproduced here).

**KL is only comparable within one table.** Each card uses its own reference (BF16, FP8 or a Q8_0
conversion), texts, top-k truncation and engine. The clearest evidence is one table that measured
several formats against the same BF16 teacher logits (alesha-pro, `GLM-5.3-Flash-abliterated-Maya-M-GGUF`):
Maya-M scores 0.093 there, against 0.29-0.33 on Maya's own, harsher protocol.

| Quant (repository) | Size | Format | KL (reference, source) |
|---|---:|---|---|
| **orcarouter NVFP4 (the default, §7.1)** | 191.0 GiB | NVFP4 experts (W4A4 declared; this engine W4A16), BF16 dense | 0.073 vs FP8, perplexity 2.894 vs 2.788 (+3.8 %), top-1 91.7 % (card) |
| EXL3 3.05 bpw (`turboderp/GLM-5.3-Flash-exl3`; MikeRoz's uncensored build) | 122.3 GB | ExLlamaV3 trellis (QTIP-style) | 0.047 vs BF16, top-1 93.1 % (alesha-pro table); the uncensored build is unmeasured |
| Maya-M (`peasantsmith/GLM-5.3-Flash-Maya-GGUF`) | 116.0 GB | GGUF IQ2_S / IQ3_XXS experts | 0.093 vs BF16, top-1 90.7 % (same table); 0.29-0.33 vs FP8 (Maya's) |
| EXL3 2.4 bpw mix (gate/up 2, down 3 bits) | 98.1 GB | EXL3 | 0.096 vs BF16 (same table) |
| EXL3 2.05 bpw | 85.1 GB | EXL3 | 0.121 vs BF16 (same table) |
| GSQ-RCO 3.5-bit (`pfeifferj/GLM-5.3-Flash-GSQ-RCO-GGUF`, IST-DASLab methods, community reproduction) | 137.1 GB | GGUF K-quants: Q3_K mostly, Q2_K, Q4_K experts; Q8_0 dense | 0.071 vs Q8_0-of-FP8 (approximate); perplexity +4.5 %; MMLU-Pro −1.4 pp |
| GSQ-RCO 3.0-bit (same) | 117.5 GB | GGUF, mostly Q2_K experts | 0.142 vs Q8_0; perplexity +9.1 %; MMLU-Pro −2.0 pp |
| AesSedai IQ4_XS / Q4_K_M / IQ3_S (`AesSedai/GLM-5.3-Flash-GGUF`) | 159 / 202 / 125 GB | GGUF | 0.117 / 0.050 / 0.283 (own reference) |
| TheDrainFlorist VQ 3.6 / 3.1 / 2.7 bpw | 144 / 125 / 109 GB | MLX vector quantization (Apple) | 0.095 / 0.200 / 0.294 vs BF16, prose only |
| Sushi 2.4 bpw (`beamster`) | 97.8 GB | sushi (Apple only) | 0.074 vs BF16 in sushi's own forward |
| EXL3 3.0 bpw selective (`0xSero`) | 149.6 GB | EXL3 + source-precision dense | 0.153 vs BF16 |
| NestQuant 1.5-4 bit (`jarrelscy`) | 59.5 + 28.6 GB floating | research format with SSD-swapped residual planes | 0.069 (research harness, not reproduced by its author) |
| REAP pruned (`OpenMOSE` 250B and others) | 265 GB FP8 … | expert pruning | 0.45 and worse: pruning costs far more than quantization |
| GCSA-AiLab RCO-GSQ GGUF "Q4"/"Q6"/"Q8_0" | 127.66 / 218.76 / 310.32 GiB | GGUF k-quants, mixed expert types | none published for any of them; rejected, §25.4 |

**The abliteration is available at source precision.** orcarouter publishes
`orcarouter/GLM-5.3-Flash-Uncensored-FP8`, the same refusal edit baked into the vendor's block-FP8
checkpoint; the NVFP4 artifact quantized exactly that. GCSA-AiLab's uncensored RCO-GSQ builds use
another edit (a different LoRA-merged abliteration), and their GSQ training did not improve on
their starting point, so the stored weights are plain k-quants; no KL or perplexity is published
for any of them (§25.4).

**Other artifacts on the same code path:**

- **`RedHatAI/GLM-5.3-Flash-NVFP4`** and any compressed-tensors NVFP4 build with the same layout
  imports through the same recipe; the base (non-uncensored) model quantized the same way is the
  obvious A/B for the refusal edit's quality cost.
- **EXL3 builds** (MikeRoz's and turboderp's): rejected as the plan (a new codec and kernels
  ≈ 7k lines against reused NVFP4 kernels; §25.4), kept as option O7 behind an owner request.
- **K-quant experts, GSQ-RCO style**: a third expert format with new Q2_K/Q3_K kernels; not planned
  (§25.4 records the GCSA rejection).

---

## 8. Ownership in the engine

### 8.1 New family

This is an **explicit product change**: a new mathematical architecture
(`Glm5NextForConditionalGeneration`, model type `glm5_next`). A family gets its own directory under
`src/models/`, started by copying the closest existing family and specialized (AGENTS.md; MoE §9's
checklist). The layout follows the existing families:

| Component | Owner | Location |
|---|---|---|
| Config interpretation (exact key set, refusals), mathematics, binding, block order, DFlash2 alignment | Model | `src/models/glm5_next/` (`config.*`, `weights.h`, `load.*`, `model.h`, `instance.*`, `options.*`, `sampling_defaults.h`, `execution/` forward and parameters, `shapes/`, `frontend/`) |
| Program: lanes, KDA state pool, MLA latent KV and index keys, prefix-cache binding, DFlash2 drafter, graphs, offload-tier composition | Program | `src/models/glm5_next/program/` |
| Engine instance | Model | `src/models/glm5_next/instance.{h,cpp}`, instantiating `EngineBackendFor` (`src/runtime/engine/engine_backend_for.h`) over its own instance — the `EngineBackend` virtual interface (`src/runtime/engine/engine_backend.h`) is how every family reaches the Engine; there are no per-family arms in `engine.cpp`. The family enters `src/models/registry.{h,cpp}` only through its three startup entry points (`detect_artifact_architecture`, `family_option_rules`, `construct_engine_backend`), registers its linear shape tables in `src/models/shapes.cpp`, and is added to `tests/test_layering.py`'s `FAMILIES` so the layering test covers it |
| Expert record `nvfp4_expert_rg16_v1` at GLM geometry | Ops | `src/ops/offloaded_sparse_moe/expert_layout.h`: the geometry (H, I) becomes a compile-time parameter of the Op's CPU and GPU routes (§8.3); no new codec |
| Tokenizer pre-tokenizer, chat template, tool-call and reasoning parsing | Frontend | §21 |
| Recipe, compressed-tensors import, expert-volume writer, restripe, verification, reference forwards | Converter / tools | `tools/convert/glm5_next.py`, `tools/convert/sources/compressed_tensors.py` (exists; already reads `weight_packed`), `official_recipes.py`, `tools/artifact/expert_volume.py`, `tools/artifact/restripe.py`, `tools/glm5_next/` |

### 8.2 The shared offload tier, and what it gains here

The offload machinery is no longer inside a family: it lives in `src/models/offload/` and is composed
by each family's Program — `expert_cache.*` (LFRU, `FramePool`, `CacheController`),
`expert_residency.*`, `expert_placement.*`, `expert_store.*` (records as file segments, the expert
stripe), `host_tier.*` / `host_expert_tier.*` (the RAM tier, its read agent and tickets),
`expert_state.*`, `route_trace.*`, `vram_monitor.*`, `memory_plan.*` (the host RAM ledger and the
VRAM frame sizing), `expert_placement_startup.*` (the `--experts` / `--expert-placement` resolution an
instance runs before its Program), `expert_placement_runtime.*` (`ExpertPlacementRuntime`: the
residency, the RAM tier, the monitor, the warm start, the switches, the status and export, which a
Program composes through hooks), and `geometry.h` (`ExpertBankGeometry`: banks, experts per bank,
top-k). This family composes them as Qwen3.8-Flash-Next does; its own code is the Forward's wiring,
the round boundary and its error codes (`program/experts.cpp`). Placement keys count the MoE layers
only: key `b * experts + e` is expert e of model layer `dense_layers + b`, and a placement file's
layer rows are MoE layers. The composition contract — no base class, no
virtual model interface, the tier an execution resource of the MoE op, the Program driving
`before_round` / `after_round` / `tier_step` at its own round boundaries — is MoE §1 and §2;
Qwen3.8-Flash-Next runs on it in production (FN §7). The MoE op itself
(`include/infernix/ops/offloaded_sparse_moe.h`, `src/ops/offloaded_sparse_moe/`) carries the
residency, staging, CPU and fetch channels for every family.

What this family adds to that tier, each a zero-cost extension for `qwen4_exp`:

| Addition | Where | Why zero-cost for `qwen4_exp` |
|---|---|---|
| **Expert geometry as a compile-time parameter** | `src/ops/offloaded_sparse_moe/` (`expert_layout.h`, the CPU and GPU routes): today `kHidden = 2560`, `kIntermediate = 640` hard-coded; the plan makes (H, I) compile-time parameters with two instances, 2560 × 640 (`qwen4_exp`) and 4096 × 2048 (this family) | Qwen3.8-Flash-Next's compiled code and bits are unchanged — its instance instantiates exactly what it instantiates today; the GLM instance is a second expansion |
| **`noaux_tc` routing** (FP32 sigmoid, bias in selection only, normalized × 2.5) | A routing variant beside the softmax top-k of `moe_route` / `moe_route_dispatch` in `src/ops/offloaded_sparse_moe/` | A separate variant selected at plan time; the softmax route's bits and kernels are untouched |
| **One read queue per volume, per-drive depth caps** | The tier's read agent (`host_expert_tier.*`) over `core/direct_read_queue.h`; today one queue serves every file, and each read already names its file | One volume today ⇒ one queue, identical behavior; the per-drive caps only change which reads are in flight together |
| **12-segment striped records (optional, §7.2)**: a record's 12 volume units as up to 12 file segments, with landed-unit masks so consumers start before a record is whole | `ExpertStore` (`src/models/offload/expert_store.h`): `kMaxSegments` 3 → 12; the fetch channel and read agent publish a per-unit completion bit (§11.4) | Only the striped-volume form selects it; unstriped volumes produce one segment per record, exactly today's description, and `qwen4_exp` fetches land as they do now |

Frame leases, the VMM arena, the LFRU, the RAM tier's decayed-LFU ranking and pin lists, the host
RAM ledger, the Windows page-lock ceiling and unpinned records, the placement forms and the runtime
switch, the sizing laws and the CPU expert engine's team and protocol are taken unchanged (MoE
§4-§5); §10 and §12 record only this family's constants and candidates.

### 8.3 Op changes

| Op | Change |
|---|---|
| `offloaded_sparse_moe` | The expert geometry (H, I) as a **compile-time parameter** of the CPU and GPU routes, instances 2560 × 640 (`qwen4_exp`, compiled code and bits unchanged) and 4096 × 2048 (this family); the existing NVFP4 W4A16 bank and its kernels serve the records unchanged (§9.5). The **`noaux_tc` routing** variant (§2.2: FP32 sigmoid, `e_score_correction_bias` in selection only, weights the unbiased scores normalized, × 2.5; ties by lower id). The shared expert **ungated** (the gate row is absent, as GLM stores none). The SwiGLU clamp limit (10) as a semantic input on every MLP class. The striped form's 12 segments and landed-unit masks (§8.2, optional) |
| `kimi_delta_attention` | Unchanged mathematics — the existing Op (`src/ops/linear_attention/kimi_delta_attention/`, recurrent and chunked routes, qualified against an FP64 oracle, lower bound in [−5, 0]) gets its first consumer; it already takes the raw gate input with `dt_bias` and applies the sigmoid itself, including β (§2.2). New companions: `kda_input_prep` (conv over the concatenated q|k|v channels + SiLU with state update, L2 norm, gate and β inputs from the projection outputs), `kda_replay_record` / fold for speculative commit, mirroring the GDN replay records (`include/infernix/ops/gdn_replay_records.h`, `src/ops/linear_attention/gated_delta_net/replay_records.cpp`) |
| New `mhc` | §15: the Sinkhorn mixer (norm, 24 projections, gates, Sinkhorn, collapse) and the BF16 residual update, both fused with neighbours. Distinct from the existing `hyper_connection` Op, which is Qwen3.8-Flash-Next's different mixer; that Op's `hyper_connection_stream_mean` is reused unchanged for the DFlash2 target features (§16.2) |
| New `dsa_indexer` | §13.3: key preparation (LayerNorm), gated pooling with APE, scoring, top-512 selection with tail; shares the exact radix select with `qsa` (`src/ops/qsa/qsa_select.cu`), which moves to `src/ops/common/` to serve both Ops |
| New `mla_sparse_attention` | §13.2: latent append (paged), absorbed decode over selected tokens, prompt kernels |
| Core | Nothing new: `DirectReadQueue` (unbuffered reads, demand/prefetch classes, scatter) and `copy_batch` exist (`src/core/`); the per-volume composition is the tier's (§8.2) |

---

## 9. Decode pipeline

### 9.1 Execution agents

The agent structure is the shared tier's (MoE §4-§5): the engine thread runs the round and applies
the policies at its boundaries; the tiers and the CPU service own their threads. This family's
table:

| Agent | Thread / stream | Role |
|---|---|---|
| Compute stream | Program's CUDA stream | Captured graph per (batch B, width W); every GPU kernel of a round |
| Engine thread | existing | `before_round` / `after_round` (promotions, landing adoption, demotions; MoE §4.4), staging of landed SSD records through copy batches |
| SSD tier read agent | `HostExpertTier`'s agent thread, one read queue per volume (§8.2) | Demand reads, lookahead reads, RAM-tier refills; publishes landed-unit masks (§11.3) |
| CPU expert team | `CpuMissService`: service thread + workers (calibrated, §12.3) | W4A16 misses from RAM slots or landing slots (§12) |
| VRAM monitor | MoE §4.9 | Wakes an idle engine to return KV pages; reports pressure |

Host threads are not pinned, and the process is exempted from EcoQoS power throttling once at
Engine construction (measured +69-162 % decode on Flash-Next, i9-13900K; FN §8.1).

### 9.2 Host-device protocol

The mapped-memory protocol — payload, `__threadfence_system`, a never-zero sequence word, `.cv`
loads of host-rewritten memory, the heartbeat-bounded wait (10 s), error words that become
recoverable round failures — is the shared tier's (MoE §5.4; the channels `MoeCpuChannel`,
`MoeFetchChannel`, the route log and landing tables are `offloaded_sparse_moe`'s). GLM's additions:

- `lookahead_req[l]`: the predicted SSD-only experts of layers l+1..l+3 with depth and score
  (§9.4), consumed by the read agent only while lookahead is on (§11.5 — off by default).
- `landing[slot]`: per landing slot, a 12-bit landed-unit mask and a sequence word, written by the
  read agent after each unit completes (release store); the GPU stage kernel and the CPU workers
  read it with acquire loads (§8.2, §11.4). The mask exists only in the striped form (§7.2); a
  whole record publishes one bit.

### 9.3 Per-layer kernel sequence and budget (T = 1)

Budget = bytes at the card's sustained rate + 0.5 µs ramp under PDL — the planning budget this
family's kernels are designed against (Flash-Next's built chain is measured directly instead;
FN §8.3). "Pre" = before routing. All numbers model at the 5090 machine's ≈ 1.5 TB/s (estimate,
FN §8.10); the BF16 dense reads tripled this table's byte counts against the EXL3-era plan.

**KDA layer (34 of 45)**

| # | Kernel | Reads | Budget |
|---|---|---:|---:|
| K1 | `mhc` mixer for attention: reduces the previous layer's partial sums (sum of squares, 24 dot products) in fixed order, gates, Sinkhorn (20 iterations, FP32, one warp), collapse, `input_layernorm` | R 32 KB + fn 0.8 MB | 2 µs |
| K2 | KDA projections: `q_proj`, `k_proj`, `v_proj` [8192 × 4096] BF16 (three GEMVs, one launch) + f_a, g_a, b [320 × 4096] BF16 | 204 MB | 136 µs |
| K3 | f_b, g_b [8192 × 128] BF16; epilogue `kda_input_prep`: conv state update over the concatenated q|k|v channels, SiLU, L2 norm, gate, β | 4.2 MB | 3.5 µs |
| K4 | KDA recurrence (existing Op, recurrent route): 4 MiB FP32 state read and written | 8.4 MB | 6 µs |
| K5 | gated `o_norm` prologue + `o_proj` [4096 × 8192] BF16; epilogue: mHC post (the BF16 update, §15.1) and partial sums for K6 | 67 MB | 45 µs |
| K6 | `mhc` mixer for the MLP + `post_attention_layernorm` | 0.8 MB | 2 µs |
| | **Pre** | | **≈ 195 µs** |
| K7 | Router (§9.4): FP32 logits [288 × 4096] for l and, from L2-warm weights, l+1..l+3 (only for prediction), noaux_tc top-8, residency classification, jobs, mailboxes | 4.7 MB + 14.2 MB | 12 µs |
| K8 | Routed gate/up for GPU jobs (W4A16 narrow) ‖ shared expert gate/up + clamp + SwiGLU [4096 × 4096] BF16 | 8 h_v × 9.4 MB + 33.5 MB | 22 µs + hits at h_v = 0.55 |
| K9 | Routed down + combine (FP32, ascending expert id) + CPU and landed outputs + mHC post + partial sums for the next K1 | 8 h_v × 4.7 MB + 16.8 MB | 11 µs + waits |

**MLA layer (11 of 45).** K2-K5 become:

| # | Kernel | Reads | Budget |
|---|---|---:|---:|
| K2m | q_a [1,536 × 4,096], kv_a [512 × 4,096] BF16 + indexer wk, weights_proj, compress gate [288 × 4096] BF16; epilogue: RMSNorm of c_q and c_kv, LayerNorm of the index key | 19 MB | 13 µs |
| K3m | q_b [16,384 × 1,536] BF16 and indexer wq_b [4,096 × 1,536] BF16; epilogue: absorbed query q'_h = W_uk,hᵀ q_h (W_uk half of kv_b, BF16, §13.1) | 63 MB | 42 µs |
| K4m | latent append (paged, §13.5) + pool completion when a 4-token block closes; indexer scores over the pooled keys + exact top-512 radix select | 0.26 MB at 4K, 8.4 MB at 128K | 4-9 µs |
| K5m | sparse latent attention over ≤ 2,051 tokens, 64 heads × 512 (§13.2); W_uv,h per head (BF16); `o_proj` [4096 × 16384] BF16 with the mHC post epilogue | 2.1 MB + 151 MB | 102 µs |
| | **Pre** | | **≈ 162 µs at 4K** |

**Dense MLP layers 0-2:** gate/up [24,576 × 4,096] + down [4,096 × 12,288] BF16, clamp in the
SwiGLU epilogue: 302 MB, ≈ 201 µs each.

**Per token (GPU work only, h_v = 0.55, 5090 machine):** 31 KDA-MoE layers × 281 µs + 3 KDA-dense
layers × 396 µs + 11 MLA-MoE layers × 248 µs + head 850 µs (the 1.27 GB BF16 `lm_head`, FP32
logits) ≈ **13.5 ms**, ≈ 510 launches. Of that, ≈ 8.6 ms is before routing (t_pre of §5.3). On the
production card the same bytes at 1.79 TB/s (measured, FN §8.3) scale the budget by 0.84.

**Decode ALU.** NVFP4 dequantization costs a lookup and a scale multiply per weight (§9.5) — far
below the EXL3 trellis's 4-5 integer instructions — so the budget's memory-bound assumption is
safer here than it was there; the bytes are dominated by the BF16 dense streams. M7 measures the
built chain against this table.

**Rules**, kept from Flash-Next's built chain (FN §8.3): PDL so a kernel always runs (no
kernel-boundary idle), weights loaded before the PDL wait, evict-first weight streams and
evict-last activations, state and next-layer routers, deterministic fixed-order reductions, grids
in multiples of the SM count. One rule matters here: **stall-time warming**. A layer waits
≈ 0.3-0.6 ms on its host-served experts (§5.3). While K9 waits, idle CTAs warm the next layer's
K1-K3 weights into L2 in reading order — ≈ 210 MB for a KDA layer, more than the 96 MB L2, so the
form warms what fits (the small tensors and the head of the projection stream) — the shared op's
`MoeL2Warm` (§9.4); estimated saving 40-80 µs per MoE layer (M7 measures it).

### 9.4 Router and prediction (K7)

Phase 1 (all CTAs): warp-per-row FP32 GEMV of the 288 router rows of layer l (the logits must be
FP32: BF16 logits create ties, FN §17.7), and of layers l+1, l+2, l+3 for prediction (their
weights, 4.7 MB each in FP32, prefetched into L2 with evict-last during K5).

Phase 2 (last CTA by arrival counter), per column:

1. `s_e = sigmoid(logit_e)` in FP32; top-8 of `s_e + bias_e`, lower id winning exact ties;
   `w_i = 2.5 · s_i / (Σ s + 1e-20)` in FP32 (the `noaux_tc` rule of §2.2).
2. Union over the T columns with per-expert column masks and n_e.
3. Classification from the residency word and the tier tables — the plan classes of the shared
   op (MoE §6.1): `READY` frame → GPU job; RAM slot → CPU job or DMA job (divisor, §9.6);
   landing slot (SSD read issued) → CPU job that waits per unit, or a staged DMA after landing
   when the CPU cap is reached; absent everywhere → **demand read** published to the read agent,
   then the same as a landing slot.
4. Outputs: GPU job list, the CPU request only when a CPU job exists, the route log (including the
   scores of ranks 9-16, the "near-miss" signal of §10.2).
5. Prediction, only while lookahead is on (§11.5): per depth d = 1..3, the union of top-k′_d
   (16, 24, 24) over the columns. Candidates already in VRAM or RAM are dropped; SSD-only
   candidates go to `lookahead_req[l]` with depth and score.

Phase 2 is ≈ 3 µs at T = 1. The dispatch's job list, entry grouping and CPU plan follow the
shared `route_dispatch_plan` forms (MoE §6.1-§6.2); GLM's variant carries the sigmoid routing and
the 288-expert geometry.

### 9.5 Expert kernels and the W4A16 arithmetic

**The stored form** (upstream, compressed-tensors `nvfp4-pack-quantized`, §7.1): per matrix,
`weight_packed` uint8 (two FP4 E2M1 values per byte, K/2 bytes per row), `weight_scale` FP8 E4M3
(one per 16 inputs, K/16 per row) and `weight_global_scale` FP32 (one per matrix). The exact
dequantized weight is `w = e2m1(packed) · scale · global_scale` — a deterministic function of the
stored bytes, shared verbatim by every consumer.

**W4A16, the canonical arithmetic of every routed expert** (every route, both processors; the
arithmetic and record layout are Flash-Next's, already built and qualified — FN §6.3):

1. **Dequantization.** FP4 nibbles decode through the E2M1 table; each 16-weight group multiplies
   its FP8 E4M3 scale and the matrix's FP32 global scale, producing BF16 weights in registers or
   shared memory (the GPU) or in place per tile (the CPU). The operation per group is fixed-order
   and identical everywhere, so the dequantized bits do not depend on the route.
2. **Dot products.** BF16 activations × BF16 weights with FP32 accumulation in one fixed order per
   output (the record's row-group order, §7.2). The GPU narrow route (T ≤ 8) and wide route
   (prefill, large verify windows) run the existing NVFP4 A16 MMA kernels; the CPU lane runs the
   existing W4A16 GEMV path (§12).
3. **Clamp and SwiGLU.** gate and up outputs are clamped (gate ≤ 10, up ∈ [−10, 10]) as exact
   operations on BF16 values, SwiGLU runs in FP32 and h rounds to BF16; down takes h. The combine
   is `y = bf16(Σ_i w_i · y_i + y_shared)` in FP32, in ascending expert id (not rank: the prefill
   accumulator of §17.2 uses the same order); every input is the same bits wherever its expert
   ran, so placement cannot change it.

So **every route on every processor produces the same bits** — the placement-invariance principle
the whole offload tier is built on (MoE §2). The engine runs A16 activations where the checkpoint
declares FP4 (vLLM's W4A4): strictly more precise on the activation side, chosen to reuse the built
kernels; the execution gate of §20.3 checks the engine against the reference forward's decoded
weights.

**Work items** (record order, §7.2): a gate/up work item covers one 16-row group's 256 blocks
(36,864 B, contiguous); a down work item one 16-row group's 128 blocks (18,432 B). In the striped
form these coincide with landing units' interiors, so no work item straddles two drives' reads
(§11.4).

### 9.6 Miss arbitration

The arbitration rules — SSD-only misses first (only the CPU serves them without a fetch), then
host-memory misses fewest columns first in job order, up to the call's cap, the host share
`misses − misses / divisor` — are the shared op's (MoE §6.1), and the divisor rule and its startup
derivation are MoE §5.6. This family's inputs:

- **RAM misses:** with the CPU's ≈ 0.35 ms per expert (estimate, §12.2) against the link's
  0.51 ms on the 5090 machine and 0.28 ms on the production machine, d = 2 (half to each) on the
  5090 and d = 1-2 on the production machine at the central estimate, against d = 3 for
  Flash-Next's 2.76 MB records. Both sides also read host DRAM (§5.1's floor), so d comes from the
  measured per-expert pair through the per-count split table of §12.3, seeded from the formula.
  M7 re-sweeps it.
- **SSD misses:** CPU jobs first (computation starts as records land), so their reads overlap the
  layer's GPU work; staged DMA after landing only beyond the CPU cap (16 jobs per call).
- **Multi-column experts** (verify windows): the CPU W4A16 kernel dequantizes each weight group
  once for all columns, so its cost grows slowly with n until the products dominate (§12.2);
  beyond the calibrated n the expert goes to DMA.

### 9.7 Round boundary

The boundary machinery — publish-after-copy with the ABA guard, landing adoption, frame
retirement, the failure path — is MoE §4.4; this family's Program is synchronous (one round in
flight). Host time between rounds (≈ 0.3-0.5 ms, estimate) is ≈ 1-1.5 % of a ~32 ms token, so
chained rounds are not built (Flash-Next measured its own and kept the synchronous contract,
FN §8.8).

### 9.8 Opt-in approximate expert mode

**Off by default; it changes the model's outputs.** Requested by the owner (2026-10-11) after
Project Maya's `TRUNK_KEEP` map (§4.2). With the mode off, nothing in this section runs and every
exactness guarantee of this design holds; with it on, the engine computes a declared approximation
of the model, and every surface says so. The shared op already carries the mechanism's shape —
`MoeExpertSource::tail_skip_rank` drops an unavailable expert every column routes at or above a
rank, zeroing its output without renormalization (Flash-Next's `--expert-tail-skip`, FN §18.3);
this family adds the per-layer map, the scopes and the rules below.

**Rule.** At MoE layer l with keep count k_l ∈ {1..8}, the router ranks the 8 selected experts by
their selection score (s_e + bias_e, the order the top-8 is chosen in) and computes the weights
w_i over all 8 exactly as the model does. An expert at rank i ≥ k_l that is **not already
available** is left out: no fetch, no CPU job, no SSD read, and its term w_i · y_i is dropped from
the combine. The remaining weights are **not renormalized** (Maya's rule: the dropped terms carry
the least weight, and renormalizing would scale up the kept ones); renormalization is a variant
the approximate-mode milestone keeps only if it lowers KL. Ranks below k_l are always computed.
The shared expert is
never skipped.

**What "available" means: the scope**, chosen at startup:

| Scope | Left out at rank ≥ k_l | Model gain, 5090 machine (§5.3 model, Maya's map, two striped volumes) |
|---|---|---|
| `ssd` | only experts that would need an SSD read (not in VRAM, not in RAM, not landed) | central 11 → ≈ 30 tok/s (the SSD term of §5.3 disappears; model) |
| `vram` (Maya's rule) | every expert not in a VRAM frame or already staged; RAM and SSD misses alike | central 11 → ≈ 40-70 tok/s (model — the whole host and SSD service of the dropped ranks disappears, a far larger gain than Maya's +23 % because this machine's round is host-bound; bought with the mode's KL) |

`ssd` removes the most expensive stalls (≈ 1.2 ms each, striped) at the smallest quality cost, but
at the production machine's default placement there are no SSD reads to remove, and on the 5090
machine 34-74 per token (§5.2); `vram` is Maya's
form. The gains assume the skippable ranks miss 1.0-1.3× as often as the layer average (they are
the colder experts); the approximate-mode milestone measures both.

**Keep map.** One digit per layer, as Maya's `TRUNK_KEEP_MAP` (`8` = exact for that layer). The
starting point is Maya's measured map for this model: 8 on layers 3-22, 5 on 23-29, 4 on 30-44.
Maya measured each band's sensitivity on Maya-M (KL to its own full model with keep 5 in one band
at a time): layers 3-7 0.040, 23-29 0.012, 30-44 0.002-0.003; the whole map 0.018 (code 0.017),
against 0.032 for a uniform keep 7, for +23 % decode. Early layers are never reduced by a preset.
The engine measures its own band table on the default artifact (the approximate-mode milestone)
before any preset
ships; maps are fixed at startup (they are part of the round's captured graphs as per-layer
constants).

**Weight-threshold rule (strata-glm's form).** Instead of, or beside, the rank map, an unavailable
expert is left out when its weight w_i / 2.5 (its share of the normalized top-8) is below a
threshold θ (`--approx-min-weight θ`). strata-glm measured +30 % at KL 0.022 for SSD-only experts
below ≈ 3 %, and +20 % at KL 0.044 for RAM misses below 8 %, with renormalization (§4.3). Weight
is the quantity the combine actually scales by, so it may beat rank at equal KL; the
approximate-mode milestone compares the
two, each with and without renormalization, and a preset may use either.

**Static rule (reproducible variant).** With `--approx-experts-rule static`, ranks ≥ k_l are
dropped whether or not the expert is resident. Output then depends only on the map, not on cache
history, so runs reproduce and the result can be scored like any model. It saves less (resident
low-ranked experts are dropped for nothing) and costs more KL; it exists for evaluation and for
users who need reproducible outputs. The default rule is `residency`.

**What residency dependence means.** Under the `residency` rule an expert's inclusion depends on
the cache state, so the same prompt can give different tokens on different runs, and greedy outputs
are not reproducible. That is inherent to the mode, which is why it is opt-in and labelled.
Everything else stays exact: an expert that is computed is computed with the canonical arithmetic,
so placement still never changes the bits of a kept term. Like CausalScoring, the exact route
(`--disable-fast-paths`) refuses the mode at startup, either rule: its output must not depend on
the cache state or on a declared approximation.

**Interactions.**

- **Caches.** A skipped expert is still credited to the VRAM and RAM policies as routed (the
  exact model wanted it), so the tiers keep learning the true working set and promote it in the
  background within the promotion budget; otherwise skipped experts would stay cold for good. The
  approximate-mode milestone measures crediting at full and at half weight.
- **Prefill stays exact.** A prompt chunk streams almost every expert anyway (§5.4), so skipping
  saves little there; latents, KDA state and the prefix cache's blocks and snapshots are computed
  exactly. Only decode and verification rounds approximate. The prefix cache tags every
  decode-produced block and snapshot with the mode, map and rule, and exact requests never reuse
  a tagged one.
- **Speculation.** Verification runs in the same mode, so accepted tokens are the approximate
  model's.
- **CausalScoring** refuses the `residency` rule (its scores would depend on cache history) and
  accepts `static`, reporting the map in its output.
- **C > 1:** the map and scope are engine-wide; lanes in one round share them.
- **Kernels.** The router's Phase 2 (§9.4) applies the rule while classifying jobs; a skipped
  expert gets a null job and a zero term, so the combine's fixed order (ascending expert id) and
  every other kernel are unchanged.

**Labelling** (no change to the advertised OpenAI/Anthropic schemas): the startup banner and
`/v1/models`-adjacent server metadata name the mode, scope and map; every request's log line
reports the experts left out per token; the `infernix-serve` statistics panel shows it; the user
guide states that outputs differ from the model's. A per-request opt-in would extend the protocol
surface and needs its own schema tests; it is not part of this design.

**Gates** (§20.3): a preset ships only if, on the frozen texts with the `static` rule and with the
`residency` rule replayed at the 5090 machine's capacities (route traces through the tier model
decide which experts would have been missing), its mean KL to the exact model is ≤ 0.02 nats,
p99 ≤ 0.3, top-1 agreement ≥ 95 %, and the long-answer loop test passes. Each preset's measured
KL and speed are published side by side.

---

## 10. Expert residency: tiers, cache and placement

The tier model, frame pools and device tables, LFRU replacement and admission, landing and safe
frame reuse, frame leases, the RAM tier, the SSD tier, the placement forms and the runtime switch,
and the two memory laws are the shared tier's (MoE §4); this section keeps only this family's
capacities, constants, candidates and measurements. The subject is the 12,096 routed keys of the
default artifact (42 × 288). Code:
`src/models/offload/`,
driven by the Program in `src/models/glm5_next/program/`.

### 10.1 Tiers

| Tier | Capacity | Holds |
|---|---|---|
| VRAM frames (13.5 MiB) | 5090 machine: ≈ 900 (7.5 %), elastic with KV and the display headroom; production machine: ≈ 5,700 (47 %) — §19.1 | Hot experts; staging; landing frames; leases |
| Pinned RAM slots (13.5 MiB) | 5090 machine: ≈ 5,400-6,000 (44-50 %); production machine: the rest, ≈ 6,400 — §19.2 | Warm experts; the landing ring; the demotion list |
| Expert volumes | All 12,096 records, read-only | Everything |

Residency is **exclusive with shadows** (MoE §4.1): a VRAM resident keeps its RAM copy only as a
shadow, the first RAM victim. Its cost is demotion copies: strata-glm rejected its exclusive mode
at 158 write-backs per token (§4.3). Here a demotion whose shadow survives copies nothing,
demotions are capped at 32 per boundary, and M7 reports demotion copies per token beside SSD
reads, so the trade is measured.

### 10.2 VRAM policy

The policy is the shared LFRU (score `f / (now − last + 1)`, global never-reset counts, halving,
one pool per record size — one here; MoE §4.3). M7 replays these candidates on GLM traces at the
real capacities and adopts a replacement only if it has ≥ 5 % fewer misses per token than LFRU
overall and is not worse than LFRU by more than 2 % on any trace category:

1. LRU (reference).
2. **LFRU with near-miss credit**: each use credits the routed experts 1 and ranks 9-16 a fraction
   c ∈ {0.1, 0.25, 0.5} of 1 (ox-boost's soft cache, which beat LRU by 12 % on this model; §4.1).
3. LFRU with hot-layer reservations from the replay (ox-boost's resident layers; per-layer quotas
   lose in the shared replay, MoE §4.3).
4. **Aged LFU** (Maya's policy: 77.5 % against LRU's 67 % in its replay, Belady 87 %; §4.2).
5. **Soft per-layer shares**: each layer's share of the pool follows its measured routing share
   (Maya's `expert_counts.txt`, +8 % there), eviction inside the layer by the base policy —
   traffic-following, not the hard quotas that lost.
6. **Static hot set + LRU remainder** (strata-glm, §4.3): a fraction F ∈ {0.6, 0.7, 0.8} of the
   frames holds the hottest experts, never evicted between re-selections every N ∈ {8, 32} tokens,
   the rest LRU.

The GLM engines disagree (ox-boost: decayed LFU lost to a router-score cache; Maya: LFU beat LRU
by 10 points; strata-glm: a static hot set beat plain LRU), so the replay on this model's traces,
with Maya's route-log format (every route plus its 16 near misses), decides. strata-glm's simulator
predicted its engine's disk reads within 2 % (§4.3), the precedent for trusting the replay.

**Promotions** follow the shared admission budget (one per layer call per decode round, the
interval scaled to the measured link; MoE §4.3), land from CPU-served misses whose score beats the
victim's and from landed SSD records (copied straight from the landing slot). M7 sweeps the
interval for 13.5 MiB promotions ({1, 2, 4, 8} × the link factor) and two rules from Maya (§4.2): an
**admission threshold** (a fetched expert is kept only once its aged route count reaches
m ∈ {0, 2, 4, 6}), and a **fill rate for free frames** (after a prompt returns its borrowed frames,
up to 24 of the hottest RAM experts per token instead of the steady-state pace). Some misses
always stay on the link: with every miss on the CPU nothing is promoted, and Maya's hit rate fell
to 60 %. No promotions after a walked prefill (MoE §7.3).

### 10.3 Staging, landing and leases

Staging slots, landing frames and their adoption, frame retirement and leases are the shared
machinery (MoE §4.4-§4.5). Staging slots are frames with a staging role, at most 32 (0.42 GiB at
13.5 MiB),
since a staging slot costs as much VRAM as a cached expert. The prefill stream's ring and staging
areas (§17) and the vision encode window of a later track borrow frames through leases.

### 10.4 RAM tier

The tier's machinery — decayed-LFU ranking with O(1) lazy heaps, the slot lists and pins,
demotions, the unpinned records past the Windows page-lock ceiling — is MoE §4.6 and §4.11. This
family's terms and additions:

- **Sizing** by the host RAM ledger (MoE §4.10) with this family's reserve and buffers (§19.2),
  at 13.5 MiB per slot: on the 5090 machine at 88 GiB available ≈ 6,674 slots; at today's 79 GiB
  ≈ 5,983. On the production machine the complement of the frame pool (≈ 6,400 records, ≈ 84.3 GiB)
  fits with ≈ 90 GiB spare (§19.2).
- **Pinning ceiling.** The plan pins up to ~82 GiB on a 95.8 GiB host. The shared ledger already
  splits the tier at the pin room (measured on this host: shared system memory 76.61 GiB, pins
  fail at 76.07 GiB, MoE §4.11) and keeps the records past it as unpinned, CPU-only `ResidentHostBuffer`s
  promoted through bounce slots. M0 re-measures the ceiling for 1.32 GiB chunks; if it is lower
  than the plan, more of the tier lands in the unpinned sub-tier and the policy keeps the experts
  most often DMA'd in the pinned part.
- **Commit charge (Windows).** VRAM allocations under WDDM and the pinned tier are both charged to
  commit (RAM + page file). Maya measured an RTX 5090 + 128 GB at 2.4-4.1 tok/s with an 8 GB page
  file and 8.4-11 with 64 GB, because the commit limit, not free RAM, capped its tier (§4.2). The
  ledger plans against the commit limit (MoE §4.10), and startup warns, naming the size, when the
  page file is smaller than ≈ total VRAM + 9 GB plus the planned tier's shortfall (this host's
  fixed 200 GB page file on `G:` clears it).
- **Demotions in flight.** An expert whose copy down to RAM has not landed is served from its VRAM
  frame or waits for the copy; it is never read from the SSD again (Maya, +5 %; §4.2).
- **Policy.** The shared tier ranks RAM by decayed LFU (MoE §4.6). ox-boost reports decayed LFU
  losing on this model ("drift-shaped locality"), so M7 replays decayed LFU (half-lives 16-256),
  LRU, CLOCK and LFRU on the two-level simulation (`tools/expert_cache_replay/host_tier.py`) and
  adopts the policy with the fewest SSD reads per token on held-out traces; the half-life stays a
  family constant.
- **Admission** at the shared margins (a demand landing when it outranks the lowest resident, a
  prefetch at a margin above the victim; MoE §4.6); VRAM victims without a host copy are demoted
  when they outrank the RAM victim (≤ 32 per boundary in decode).

### 10.5 Warm start, prompt seeding and refill

- **Expert state file** (the shared `--expert-state`, `src/models/offload/expert_state.*`; MoE
  §4.3): startup fills VRAM from the top of the saved ranking and RAM with the next records,
  reading the volumes in record order with large coalesced reads at full bandwidth (on the 5090
  machine ≈ 86 GiB in ≈ 7 s at 12.7 GB/s, overlapped with pinning; the production machine reads
  its 159.4 GiB from the T710 at its measured 6.4-13.4 GB/s, ≈ 13-27 s — model, M0 measures the
  T710 at this record size): startup ≈ 15-30 s plus the dense load. A GLM calibrated order is
  shipped when M7's traces justify one, else the seed spreads across the layers (MoE §4.3).
- **Prompt seeding** is the shared prompt credit (counts only, no copies; MoE §4.3), with a
  weight M7 fits. The prompt landing reserve — admitting the prompt's SSD-read experts to RAM —
  is **off by default**: Maya measured that keeping a prompt's disk reads in RAM gave more disk
  reads in the answer, not fewer, since a long prompt touches every expert (§4.2). M7 re-measures
  it once.
- **Idle refill:** with no request active, the read agent re-ranks and refills RAM at full
  bandwidth (the shared boot warm start and tier fill; MoE §4.8).

---

## 11. Multi-SSD expert streaming

The read-only rule, the unbuffered reads, the demand-before-prefetch classes and the read queue
itself are the shared SSD tier's (MoE §4.7); this section is this family's composition of them
over its volumes (§7.3), plus the read queue per volume of §8.2 (landed-unit masks exist only in
the striped form).

### 11.1 Requirements

- One to four expert volumes on any drives; two is the designed case on the 5090 machine; the
  production machine runs one until a second drive exists.
- Read-only at runtime; no SSD cache, spill or swap (MoE §4.7).
- Unstriped default: a record read is one whole-record read from its volume. Striped (optional):
  every record read uses every volume; per-record latency ≈ record / Σ bandwidth.
- Demand reads never queue behind more than ≈ 0.3 ms of prefetch per drive.
- A failed read fails only the rounds that need it (the shared failure path, MoE §4.4), never the
  server.
- Windows (IOCP) and POSIX (the `pread` pool) behind the shared `DirectReadQueue`
  (`src/core/direct_read_queue.h`); io_uring is option O6 (§25.3).

### 11.2 Reading a record

Unstriped (the default): one whole-record read of 14.16 MB from the record's volume, landing in
one contiguous 13.5 MiB slot (pinned, 4 KiB-aligned): ≈ 2.3-2.4 ms at QD 1 on a 990 PRO (measured,
§3.3).

Striped (optional): the record read is 12 unit reads issued at once, w_v units to each volume v,
each landing at its byte offset in the same slot. With w = (5, 7) on `D:` + `G:`, the 990 PRO
serves five 1.125 MiB reads and the Optane seven; at ≈ 5.7 and ≈ 7.0 GB/s the record completes in
≈ 1.1-1.3 ms (model; the probe's 1 MiB cells are §3.3's, M0 measures the 1.125 MiB units). Unit
issue order is gate/up units first (0-7), round-robin over volumes, then down units (8-11), so
consumers can start the gate/up half while the down half is still in flight.

### 11.3 The read engine

- **One `DirectReadQueue` per volume** (§8.2), owned by the read agent thread: each queue maps
  (bank, record, unit) to (volume, offset) by §7.3, submits a record as its 12 unit reads (striped)
  or one whole-record read (unstriped) with one tag, and reports per-unit completions. Adjacent
  units in one volume coalesce into one read (the queue's scatter form).
- **Depth caps per volume**: demand 4, lookahead 2, background 1, total 7 (1.125 MiB units: the
  probe's knee lies at 1 MiB, QD 4, §3.3; §11.7 calibrates).
- **Priorities**: demand (current layer, by layer then score) > lookahead depth 1 > 2 > 3 >
  background refill. A queued, unissued lookahead unit is dropped when its round has passed its
  layer.
- **Landing ring**: 64 slots (0.84 GiB of the RAM tier) by default; a slot is reusable once its
  CPU job has finished or its staged copy has landed, and once the RAM tier has admitted or
  declined it (an admitted landing becomes a RAM slot by changing roles, without a copy).
- **Completion publication**: on each unit completion (striped) or record completion (unstriped)
  the agent sets the unit's bit in the slot's landed mask and bumps its sequence word (release).
  CPU workers spin on their units' bits; the GPU's stage kernel copies a unit to its staging frame
  as soon as its bit is set (one 1.125 MiB copy per unit, so a unit's gate/up row groups are
  usable as they land).
- **Never mapped.** Volumes are opened unbuffered and never memory-mapped, not even for metadata:
  Maya measured that on Windows a mapped file sends every direct read of it down a slower path
  (§4.2), and the shared tier opens deny-write handles for the same reason (MoE §4.7).
- **Readers.** One agent thread with the per-volume queues by default. Calibration (§11.7) also
  tries Maya's alternatives: pairs of units per read instead of single units, and a few
  whole-record reader threads, which gained 25-50 % on DRAM-less drives there and lost 15 % on
  drives with DRAM.

### 11.4 Consuming partial records

The striped unit boundaries sit inside the work items of both processors (§7.2), so no work item
ever straddles two drives' reads:

| Consumer | Granularity | Waits for |
|---|---|---|
| CPU gate/up phase (dequantize + SwiGLU) | 32 row groups per unit | its unit's landed bit |
| CPU down phase | 64 row groups per unit | all of h (gate/up barrier) and its unit |
| GPU narrow route via staging | row groups | the unit's staged-copy ticket |

Unstriped, every consumer waits for the whole record (one bit).

### 11.5 Lookahead

An SSD read that is not issued until its layer routes costs ≈ 1.2 ms of stall (striped): about
seven layers of pre-routing GPU time (§9.3). Reading the predicted SSD-only experts of layers
l+1..l+3 (§9.4) ahead is the obvious remedy, but the one measurement on this model is negative:
Project Maya found only ~40-45 % of disk misses predictable 1-4 layers ahead, and its lookahead
was exact but slower on one NVMe because wrong reads competed with demand (§4.2). Two volumes give
lookahead spare bandwidth that one does not, so it is designed here but **off by default**
(`--ssd-lookahead off`) until M7 shows a gain on the two-volume configuration. The production
machine's default placement has no SSD reads; on the 5090 machine the 34-74 reads per token (§5.2)
make its ceiling the gap to the no-SSD column of §5.3, of which Maya's 40-45 % predictability
could recover part:

- **Budget per layer:** lookahead reads may use the drives' idle time only. The agent tracks each
  volume's queued demand bytes; a lookahead read is issued only if it would complete before the
  predicted use (depth d ≈ d × 2 ms ahead at the central 5090 token rate) and the volume's demand
  queue is empty.
- **Over-fetch:** Fate (arXiv 2502.12224) reached 97 % coverage by fetching the confident set
  rather than the exact top-k. The candidate width k′_d is a calibrated parameter; ox-boost
  measured that over-fetch hurts when it competes with demand (P = 6-8 slower than 4), which the
  budget rule prevents for SSD reads.
- **Used landings** become CPU jobs at once; **unused** ones are offered to the RAM tier like any
  other landing, so a wrong guess costs drive time but not necessarily the bytes.
- **Measured before adoption** (M4 traces, M7 replay): recall of SSD-tier experts by depth and width, useful-byte
  ratio, exposed SSD stall per token with lookahead off, depth 1, and depth 1-3. Lookahead depth
  d is disabled automatically for a layer whose useful-byte ratio stays below 0.3.

### 11.6 Placement on the 5090 machine

| Configuration | Measured bandwidth (§3.3) | Free space for a volume | Advice |
|---|---:|---|---|
| **`D:` + `G:`**, w = (5, 7), striped | 12.7 GB/s (5.7 + 7.0) | 144 / 128 GB | **Default on this machine.** Both drives otherwise idle; the Optane has the lowest tail latency; volumes 71.3 and 99.8 GB. `G:` also holds the page file, which the pinned tier and the 2 GiB reserve keep quiet; §11.7's telemetry would show paging as a slow volume |
| `D:` + `E:`, w = (6, 6), striped | 11.7 GB/s | 144 / 371 GB | Works; `E:` is the repository and build drive and showed 15-41 ms p99 tails during the probe |
| `C:` + `D:` + `G:` or `C:` + `D:` + `E:`, w = (4, 4, 4), striped | ≈ 17.6 GB/s (`C:` bypasses the chipset) | `C:` 243 GB | Fastest; `C:` carries the OS and page file, so it is opt-in; +8 % decode (§5.3) |
| Three volumes all on the chipset | 14.2 GB/s (DMI-bound) | | Not recommended |
| One volume, unstriped | 5.7-7.0 GB/s | | Supported; ≈ 0.7× of two-volume striped decode (§5.3); the production machine's default |

Unequal weights matter little: with w = (6, 6) on `D:` + `G:` a striped record takes
max(1.18, 0.96) = 1.18 ms; with (5, 7), max(0.99, 1.13) = 1.13 ms (4 % faster).

### 11.7 Calibration and topology checks

At startup the read agent reads 64 units from each volume at QD 1 and QD 8 (≈ 25 ms total) and
records per-volume latency and bandwidth for the lookahead budget and the cost model. It also maps
each volume's file to its physical disk and PCIe root port (`IOCTL_STORAGE_GET_DEVICE_NUMBER`, the
PnP location paths used in §3.2) and warns when two volumes share a disk, when more than two
volumes sit behind the chipset, when a volume's drive is the system drive, or when a drive
measures below 60 % of the class rate (thermal throttling, power state, a filter driver).

### 11.8 Failures and integrity

- A unit read that fails is retried once (the queue's own retry and backoff, MoE §4.7); a second
  failure fails the rounds that need it through the error word (MoE §4.4) and marks the record
  unreadable for 60 s.
- A missing, short or mismatched volume refuses startup with the expected path, size and id.
- Startup checks every header and the CRC32C of 64 random units against the table the converter
  stores in the main artifact (≈ 72 MB read); `tools.glm5_next.verify_artifact` checks every unit
  offline. Per-read hashing stays rejected.

---

## 12. CPU expert engine

The engine's design — role and limits, the ISA variants, the worker team's scheduling, the
miss-service protocol, the startup rate probes, the decode divisor and the prefill split — is the
shared tier's (MoE §5); this section keeps this family's kernel, constants and candidates. Code:
`src/ops/offloaded_sparse_moe/cpu/` (this family runs the existing `w4a16` path, its geometry now
a compile-time parameter, §8.3).

### 12.1 Role

The CPU computes about half of the RAM misses (§9.6) and every SSD miss it can take within the job
cap, in place from RAM slots or landing slots, with W4A16 — the same bits as the GPU (MoE §2,
§5.1). The shared limits (512 jobs per layer call, 8 columns per job, 4,096 columns per served
call; `src/ops/offloaded_sparse_moe/cpu/miss_request.h`) bound this family's calls with room: a
decode round's 42 layers × 8 experts stay far inside them.

### 12.2 The W4A16 kernel and its cost

`cpu/` runs the existing W4A16 expert path: FP4 groups dequantize to BF16 in place (E2M1 table,
the group's FP8 scale, the matrix's global scale), BF16 dot products accumulate in FP32, and a
scalar form is the exact oracle. Per 16-weight group: 8 packed bytes decode through the table and
multiply two scales — a few instructions per group, where the EXL3 design cost about one vector
micro-operation per weight and Flash-Next's W4A4 lane dequantizes straight into the products
(MoE §5.2).

**Estimate** (model; M0 measures before anything depends on it): 25.2 M weights and 25.2 M MACs
per expert at GLM geometry. On the production machine's P-cores with AVX-512, ≈ 0.2-0.35 ms per
expert across 8-12 workers; on the 5090 machine's i9-13900K (AVX-VNNI, no AVX-512) ≈ 0.3-0.5 ms.
The record's DRAM read costs ≈ 0.18 ms at 80 GB/s, so the lane sits near the DRAM/compute boundary
— compute-heavier than Flash-Next's DRAM-bound lane, far lighter than the EXL3 design's
decode-bound estimate. §5.3 carries 0.25 / 0.35 / 0.50 ms. A second column reuses the dequantized
groups, so a verify column adds roughly a third of the first column's cost.

This is the largest difference from Flash-Next's CPU lane, which streams 2.76 MB records at DRAM
speed (MoE §5.2): here an expert takes about as long as its DMA on the production link (0.28 ms)
and clearly less than its DMA on the 5090 machine's (0.51 ms), so the CPU and the link split the
misses per §9.6 underneath the host DRAM floor of §5.1.

### 12.3 Execution

The team mechanics — the service thread as worker 0, epoch spin then futex park, work by atomic
counters, the phases with their barriers, exact sums making any split bit-exact — are MoE §5.3.
This family's settings:

- **Landing-aware items:** an item over a unit that has not landed is skipped and retried after
  the worker's other items, so workers never idle while another unit is available (§11.4).
- **Workers:** Flash-Next's optimum of six was set by DRAM sharing with the PCIe stage (MoE §5.2).
  The W4A16 lane sits near the DRAM boundary (§12.2): CPU reads plus DMA source already approach
  the ≈ 80 GB/s DRAM limit at the central miss counts, so M0 and M7 sweep 6-12 P-cores plus
  0-16 E-cores (E-cores add compute but also compete for DRAM). Workers sit on physical cores,
  never on SMT
  siblings: Maya measured hyperthreads at 21.8 → 13.3 tok/s, because they starve the disk readers
  and the service threads (§4.2). The read agent and the engine thread keep P-cores of their own.
- **Pieces, not shares:** a call's CPU work is cut into ~48 pieces claimed in turn, so one worker
  held up (by an SSD unit not yet landed, or by a slower E-core) does not hold up the layer (Maya:
  14.3 → 17.1 tok/s with 6 threads). The shared atomic item counters already have this shape; the
  piece count is sized for 13.5 MiB records and mixed core speeds.
- **Per-count split table:** the CPU/link split is a calibrated table indexed by the layer's
  number of RAM misses (Maya's `CPU_PLAN`), seeded from the divisor rule (MoE §5.6), measured at
  start over half a second (the shared startup probes, MoE §5.5), and kept across starts in the
  calibration profile with a drift check (Maya's `CPU_CAL`: a line more than 15 % off five times
  in a row is re-timed).
- Software prefetch distance and the worker set come from the shared probe and calibration
  (MoE §5.2, §5.5).

### 12.4 Prefill CPU assist

The split's shape — copies gated on each layer's routing, the thinnest non-resident experts left
to the CPU, the plan at a share of the measured rates, the link-scaled width cap, the assist caps
— is MoE §5.7. Flash-Next limits its CPU share to narrow calls because its wide W4A4 route is not
CPU-computable; here every W4A16 route is exact, so the CPU may take **any** column count without
changing a bit, and the limit is a cost decision. As in Maya's `PREFILL_CPU`, the least-routed
non-resident experts of each layer go to the CPU, the cut chosen per layer so the CPU and the
link finish together, from a learned cost per expert-row. The CPU decodes each state once per
expert regardless of columns, so this takes experts with up to a few dozen columns.

---

## 13. Sparse MLA attention and the pooled indexer

No sm_120 kernels exist for either (FlashMLA's sparse kernels and DeepGEMM's indexer kernels are
SM90/SM100; §4.1), so both are new engine Ops.

### 13.1 The absorbed form

With no RoPE part, the latent is the whole key, and the reference attention is exactly an MQA over
the latents:

```text
q'_h     = W_uk,hᵀ q_h                       512 per head (W_uk,h: 256 × 512)
score_hj = q'_h · c_j / 16                   c_j: the cached 512-value latent of token j
u_h      = Σ_j softmax_j(score_hj) c_j       512 per head
o_h      = W_uv,h u_h                        256 per head
```

The reference materializes k_h = W_uk,h c_j and v_h in BF16; those are private intermediates, not
cached or observable, so the absorbed form is a legal private arithmetic
([op development §6](op-development.md#6-qualification)), qualified against an FP64 oracle of
the reference formula.

The checkpoint's `kv_b` is one BF16 tensor [64 × 512 × 512]: per head [512 → 512],
`W_kvb = [W_uk; W_uv]` with 256 key rows and 256 value rows. The converter stores its two halves
as the Ops read them, both BF16: `value_up` [64 × 256 × 512] holds each head's W_uv as stored, so
o_h = W_uv,h u_h reads one contiguous row per output; `key_up` [64 × 512 × 256] holds each head's
W_uk transposed, so q'_h = W_uk,hᵀ q_h also reads one contiguous row (256 query weights) per
latent dimension. The transposition is done once in the converter, never at load time. Folding W_uk into W_qb or W_uv into W_o offline was rejected: it would
change the stored weights and read more bytes (50 M and 134 M parameters against 33 M and 75 M).

### 13.2 Decode kernel (`mla_sparse_attention`)

Per layer and sequence: Q' is 64 × 512, the selected latents ≤ 2,051 × 512.

- Two BF16 `mma.sync` GEMMs with FP32 accumulation: S = Q'Cᵀ, then U = P C, an online softmax
  between them. 268 MFLOP and 2.1 MB per layer and token: bandwidth-bound at ≈ 1.5 µs of reads.
- Split over token blocks: 32 CTAs × 64 tokens, partial (max, sum, U) merged in fixed block order.
- The selected set arrives as pool ids from K4m (§9.3); a CTA gathers whole 4-token pools through
  the block table plus the tail tokens.
- An INT8-G64 latent cache (§13.5) is staged as stored (its codes and FP16 group scales) and
  decoded in the score and value folds: a lane's 16 dimensions lie in one group of 64, so the group
  scale multiplies the lane's partial dot once and the probability once per token. FP8 and NVFP4
  latents would follow the same staging.

### 13.3 The indexer (`dsa_indexer`)

Every step follows the reference (§2.2) at its precision:

| Step | Precision | Stored |
|---|---|---|
| k_t = LayerNorm(W_k x_t), z_t = W_g x_t | FP32 LayerNorm, BF16 outputs as the reference's linears produce | A per-lane ring of the 64 most recent positions' k and z (32 KiB per lane and layer): a pool reads its earlier tokens there and the call's own from its inputs; rollbacks of up to 60 positions stay exact |
| Pool key = Σ_i p_i ⊙ k when a pool completes (p_i cast to BF16) | BF16, as the reference computes it | **BF16 pooled-key plane**: 128 BF16 per pool = 64 B per token and layer |
| q_h = W_qb_idx c_q (BF16), w_h = W_w x · 32^−½ | BF16 linear outputs, FP32 weights | — |
| score_p = Σ_h w_h relu(q_h · key_p / √128) | FP32 accumulation over BF16 keys | — |
| Top-min(512, P) pools + tail | Exact radix select, lower pool id on ties | Pool ids for K5m |

- **Decode** scans every complete pool: 2 bytes × 128 per pool, 8 MB per layer per token at 128K
  context (≈ 5.5 µs), 64 MB at 1M (≈ 43 µs; the long-context term in §13.6).
- **Prefill** scores are O(T × P): a register-tiled FP32 GEMM over (queries × pools) per head,
  followed by the weighted relu sum. Maya measured that form at 5.6 ms against 78.8 ms for a
  warp-per-pool kernel (RTX 3090, 4,096 queries at position 10K; §4.2), so exact FP32 is affordable
  and no reduced-precision split is planned. The score tile is bounded by processing queries in
  sub-chunks of 2,048 (strata-glm's choice, which let 262K contexts fit; §4.3): at 1M context a
  2,048-query sub-chunk scores 262K pools per head in slices, never materializing the whole
  queries × pools matrix. Selection is checked exactly on near-tie-free fixtures and by pool
  agreement on real activations.
- The radix select moves from `qsa` (`src/ops/qsa/qsa_select.cu`) into `src/ops/common/` and serves
  both Ops (QSA's selection rules match: 4-token blocks, budget 2,048, tail always attended, lower
  id on ties).
- **No KV-profile dependence:** pooled keys and selection never depend on `--kv-dtype` (Flash-Next's
  rule, FN §14.1), so changing the latent format never changes which tokens a query attends to.

### 13.4 Prefill attention

- While the visible context is ≤ 2,051 tokens every token is selected: dense causal attention in the
  **non-absorbed** form (q·k with d = 256, v with d = 256, k_h and v_h formed once per key by
  W_kvb in the chunk on the BF16 prompt route), the cheaper form when keys are shared by every
  query.
- Beyond that, each query has its own set, and the absorbed MQA form serves it (DeepSeek's choice
  for DSA): ≈ 269 MFLOP per query and layer, ≈ 1.9 s of BF16 tensor-core work for a 128K prompt
  (model). Query tiles of 64 share their tails and most recent pools; the kernel processes the tile
  against the union of its selected pools with a per-query mask, as Flash-Next's QSA prompt kernel
  does (FN §14.2).

### 13.5 KV formats

The latent is both key and value, so per-K/V split profiles do not apply.

| `--kv-dtype` | Latent per token and layer | With the 64 B pooled key | 11 MLA layers: 128K / 1M tokens |
|---|---:|---:|---:|
| `bf16` (primary; built) | 1,024 B | 1,088 B | 1.6 GB / 12.5 GB |
| `int8` (built: INT8-G64, one FP16 scale per 64, the persistent group codec of `kv_cache_append.h`) | 528 B | 592 B | 0.85 / 6.8 GB |
| `fp8` (E4M3, one FP32 scale per 128) | 528 B | 592 B | 0.85 / 6.8 GB |
| `nvfp4` (E2M1, E4M3 per 16) | 288 B | 352 B | 0.5 / 4.1 GB |

The indexer's own keys and gates live in its per-lane ring (§13.3), not per token. Measured
INT8-G64 cost (FP64 reference forward, `tools/glm5_next/reference.py --latent-codec`, 495 tokens): mean KL 8.0e-3 from the FP64 oracle
against 1.28e-2 for BF16 rounding alone, and BF16 with INT8 latents 1.30e-2; groups of 128, one
scale per row and a Hadamard rotation measured within 0.001 of it.

Orthogonal rotations commute with the absorbed product (q'·c = (Hq')·(Hc), Σ p c = Hᵀ Σ p (Hc)), so
Hadamard-domain latent profiles could apply later; they are refused for this family until a track
qualifies them. Order of work: `bf16`, then `int8` (built), then `fp8` and `nvfp4`.

### 13.6 Long context

Default `--max-context 131072`; up to 1,048,576 is supported. At 1M a `bf16` cache is ≈ 12.5 GB
with its pooled keys (§13.5; `int8` 6.8 GB): the expert cache would lose ~880 (`int8` ~480)
frames per sequence. Above 128K the design uses a live
KV host tier — planned
for Flash-Next and never built there (FN §14.4), so it is new work here: the pooled-key plane stays
on the device, latent pages live in pinned RAM with a device CLOCK cache of hot pools, and only the
selected ≤ 2,051 tokens per query are read. The host copy costs RAM-tier slots (9.0 GB of `fp8`
latents and pooled keys at 1M ≈ 640 slots), which the ledger shows. The indexer's per-token scan
grows linearly (≈ 0.5 ms per token across 11 layers at 1M); an FP8 copy of the pooled keys for the
scan, with FP32 rescoring of the top candidates, is option O4 (§25.3).

---

## 14. Kimi Delta Attention layers

- **Mathematics:** the existing `kimi_delta_attention` Op
  (`src/ops/linear_attention/kimi_delta_attention/`, `include/infernix/ops/kimi_delta_attention.h`)
  is exactly this model's recurrence (§2.2), including the −5 lower bound and FP32 state. It has
  recurrent and chunked routes, was qualified against an FP64 oracle, and has had no consumer.
- **Decode** (§9.3 K2-K5): one launch for `q_proj`, `k_proj`, `v_proj` (BF16) grouped with the small
  f_a, g_a, b projections (ox-boost: KDA gate fusion 5 → 2 GEMVs), `kda_input_prep` in the
  epilogue of f_b/g_b, the recurrent route, and the gated `o_norm` in `o_proj`'s prologue. State
  traffic is 285 MB per token: ≈ 0.2 ms, so a BF16 state is not worth its numerical change here.
- **Prefill:** the chunked route. FLA uses 64-token chunks and this Op 16; M8 measures 16, 32 and 64
  against the prefix cache's phase-aligned chunking (spec §7.2).
- **Speculative verification:** a verify call runs every row's columns from its lane's state
  without writing it (`kimi_delta_attention_batch_verify`, the one-token arithmetic, so column j
  equals j + 1 decode steps bit for bit; `causal_conv1d_silu_from_states_split` for the
  convolution). The Program records, per KDA block and verify column, the convolution input and
  the recurrence's k, v, g and β (≈ 98 KB), and the commit folds each row's accepted prefix into its
  lane (`kimi_delta_attention_fold`, `causal_conv1d_commit_window`).
- **Prefix snapshots:** 136 MiB of state plus 5 MB of conv state, 141 MB per snapshot (§18).

---

## 15. Manifold-constrained hyper-connections

### 15.1 Kernels (`mhc`)

The mixer reads all four streams (32 KB per token at T = 1) and 24 × 16,384 FP32 weights (1.6 MB).
Two kernel boundaries per sublayer are avoided by fusing with neighbours:

1. **Residual update in the producer's epilogue.** The kernel that produces y (`o_proj`, or the MoE
   combine K9) writes R'_s[d] = post_s · y[d] + Σ_j comb[j][s] · R_j[d] for its rows (local in d),
   and per tile the partial Σ R'² and the 24 partial dot products fn · R' over its rows.
2. **Mixer kernel** (K1/K6, 16 CTAs): every CTA sums the partials in fixed order, forms
   f = (fn · R') / rms(R') (the paper's "division after the matmul"), the gates, and comb by exactly
   one column normalization and 19 row-then-column rounds with eps 1e-6, all in FP32 (one warp; the
   4 × 4 work is trivial). Each CTA then collapses its 256 elements x = Σ_s pre_s R'_s and writes the
   partial sum of squares for the following RMSNorm, which the consumer's prologue applies.

The mixer is ≈ 2.5 µs per sublayer, 90 per token (≈ 0.23 ms). ox-boost found mHC fusions within
noise on this model; they are kept because they also remove the residual re-reads that cost more at
C > 1.

### 15.2 Numerics

- Mixing weights, Sinkhorn, pre, post and comb in FP32 (`fn` upcast from BF16, §2.2); the block
  input norm is the reference's unweighted RMSNorm over the flattened 16,384 values.
- Residual streams stay BF16. Flash-Next measured an FP32 residual as worse in perplexity
  (+0.030 nats; FN §17.7): the model expects the rounding it was trained with.
- The reference forms R' entirely in BF16 (`bf16(post) · y + bf16(comb)ᵀ @ R`, the matmul
  included, §2.2). The fused update computes it in FP32 and rounds once: more precise, kept only
  if the teacher-forced comparison of §20.3 shows it is not worse (Flash-Next's rule, FN §17.7);
  the default is the reference's BF16 form.
- Sinkhorn is reproduced iteration for iteration, never iterated to convergence.

---

## 16. Speculative decoding

### 16.1 Expected value

- **DFlash2 is the only drafter.** The artifact ships a matching drafter (§16.2), the same
  DFlash2 structure Qwen3.8-Flash-Next runs in production, and the drafter is a small
  self-contained BF16 model (2.2 GiB) that adds no expert misses to the target's host path. The
  checkpoint's MTP layer is stored in the artifact but not loaded (§7.1): both public
  GLM-5.3-Flash MTP experiments measured slower over PCIe-transferred misses (§4.1), and Project
  Maya traced why on one GPU — consecutive tokens share only ~1 of 8 experts per layer and none
  of their RAM pulls (§4.2).
- ericlewis measured DFlash2 acceptance ≈ 0.70 at 3 draft tokens on this drafter (upstream, third
  party); M6 measures it here, on this target.
- On the 5090 machine the round time is dominated by host-served expert work, which extra verify
  columns do not amortize (MoE §6.3); extra columns are cheap for CPU-computed experts (§12.2) and
  expensive for DMA'd and SSD-read ones. The draft-length policy (§16.3) therefore includes K = 0.

### 16.2 The DFlash2 drafter

[incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) (upstream,
read 2026-11-23; license Apache-2.0 per its card, accepted by the owner 2026-11-23):
`DFlash2DraftModel`, model_type `qwen3`, 2.2 GiB BF16; 5 sliding-window layers
(window 2048), hidden 4096, intermediate 12288, 32 heads, 8 KV heads, head dimension 128, rope
theta 10000; `dflash_config`: block_size 8, conv_kernel_size 2, conv_group_size 16, selector_rank
256, selector_top_k 16, mask_token_id 154856, target_layer_ids [5, 14, 24, 33, 42]; `fc.weight`
[4096, 20480] (five 4096-wide target features); a candidate selector with predecessor and
successor codebooks [154880, 256].

This is the same structure as Qwen3.8-Flash-Next's DFlash2 drafter
([dflash.md](dflash.md)), so the engine's DFlash2 path applies at hidden 4096 as a companion
artifact beside the main one (FN §11.3's binding and proposal graphs). The target feature of a
target layer is the mean of the four mHC streams of its completed residual
(`ops::hyper_connection_stream_mean`), the same convention as Qwen3.8-Flash-Next — **to be
confirmed by acceptance** in M6 (open question, §25.2). The drafter's own weights are all BF16 and
resident; it never touches the expert tiers.

### 16.3 Draft length

**Measured (M6, 2026-11-24, RTX PRO 6000, `vram=45%,ram=55%`: 5443 / 6621 / 32 experts in VRAM /
RAM / SSD, INT8 KV, 600-token essay continuations, two runs a level; aggregate output tok/s):**

| Decode | 1 request | 2 requests | 4 requests | Process VRAM |
|---|---|---|---|---|
| plain | 29.7, 33.5 | 43.2, 39.0 | 46.8, 45.2 | 90.1 GiB |
| DFlash2 K = 3, every round | 31.5, 37.2 | 34.4, 35.2 | 39.3, 40.7 | 91.7 GiB |
| DFlash2 K = 7, every round | 25.0, 26.8 | 24.6, 24.1 | 27.8, 29.7 | 91.8 GiB |
| DFlash2 K = 2, one-row rounds only | 37.2, 33.6 | 42.9, 40.9 | 47.3, 52.4 | 91.6 GiB |
| DFlash2 K = 3, one-row rounds only | 34.9, 33.6 | 47.7, 39.2 | 54.4, 53.0 | 91.6 GiB |

Acceptance on these prompts is 49-58 % at K = 3; the expert cache's hit rate falls from 84-86 % to
76-77 %, as every verify column routes its own experts and those beyond VRAM cost a RAM or SSD
service each. A greedy chat continuation (the real engine test) accepts 5.27 of 7 drafts a round; a sampled
one (temperature 0.8) 60 of 236 drafts at K = 7. With one-row rounds only, the concurrent levels
are no slower than plain (their tail rounds, once one request is left, draft) and one request gains
5-15 % at K = 2 or 3, within these runs' spread; K = 2 is the starting choice.
So a round of several rows decodes plainly, and only one-row rounds draft; the drafter adds 1.6 GiB
of VRAM (weights, context ring, verify records). The cost model below replaces this rule.

Flash-Next's policy shape (FN §11.4): maximize E[tokens(K)] / E[t_round(K)] with
t_round(K) = t_gpu(K+1) + t_host(U_{K+1}, current tier shares) + t_ssd + t_draft(K); every input an
online EWMA; K = 0 always a candidate; experts used only by rejected columns are not credited to
the caches (MoE §6.3); the cost fits are placement-scoped and reset on a placement switch
(MoE §6.3). Expected outcome: K ∈ {0, 1} on most text, larger on code edits.

### 16.4 Other proposals

- **No n-gram drafting.** The artifact carries no n-gram table (§2.3), and the copy proposals
  from a request's own history are left out: the frontend builds no n-gram sources, the Engine
  holds no archive, and `--ngram-draft-tokens`, `--ngram-archive-mib` and `--ngram-volume` are
  refused at startup.
- **Draft-informed prefetch** (SP-MoE / MoE-SpeQ): applying the main routers of early layers to the
  draft token's embedding to start SSD reads before the verify round. Option O3 (§25.3), gated on
  measured recall.

---

## 17. Prefill

The walk-span structure, the streamed expert ring, chunk sizing against the cache, and the
no-promotions rule are the shared prefill design (MoE §7); this section keeps this family's
constants and its own mechanisms.

### 17.1 Walk spans

Prompts are processed layer-major over spans of up to 64K tokens (MoE §7.1), with equal chunks and
the arena lent from frames. Every non-VRAM expert crosses the link once per span (§5.4); an expert
without a host copy is streamed too, through the tier's lent stream slots reading the volumes
(MoE §4.7, §7.1).

### 17.2 Expert supply within a layer: the streamed ring

**Built (M9, 2026-11-24):** a prefill chunk of at least 256 columns streams its non-resident
experts by copy-engine DMA two layers ahead of compute — the Flash-Next variant (FN §13.3, MoE
§7.1), not the expert-group pipelining this section first described. `ExpertStream`
(`src/models/glm5_next/execution/expert_stream.{h,cpp}`, specialized from Qwen3.8-Flash-Next's:
no gated CPU split, ggml banks or host record strides, which GLM does not use; the SSD stream
slots and the unpinned-record bounce stay, so a layer walk can be added later) copies each MoE
layer's non-resident experts with a pinned host record, in ascending id, into half l % 2 of a ring
of frames the expert cache lends for the prompt (`lend`/`give_back`, returned after the prompt's
last call once no lane is prefilling); runs of adjacent records of one bank coalesce into one
copy, and a layer's copies enqueue as one `CopyBatch`. The MoE reads a streamed record like a
resident one (`MoeExpertSource::prefetched`), so a call whose layers all fit stages nothing and
runs as one pass; an expert beyond the ring's half — or without a host copy, on the SSD tier —
stages or fetches as before, unchanged. An SSD-only expert is streamed too when `begin` plans with
the tier's stream slots (`SsdReads`, a later walk's span); a single call leaves it to the fetch
channel. `--prefill-expert-stream off` (the Engine option, default on) stages with the MoE's SM
kernel as decode does, the records the same bytes either way.

**Departure from the expert-group design below.** Expert groups stage the next group with the
engine thread and accumulate windowed outputs; that machinery was never built, and the staged
wide pass this implementation actually ran — `stage_kernel` on the compute stream, serially
before each pass — could not overlap: the MoE's staging-overlap route is refused for wide calls
(`moe_layer.cu`: `overlap` requires `!use_wide`; overlapped wide passes measured nondeterministic
in Flash-Next, cause unidentified, FN §13.4). The ring needs no new accumulation arithmetic (the
combine order stays the wide route's), reuses the frame pools through lending, and is proven
bit-exact in Flash-Next, so it replaces the group plan; a walk span (§17.1) remains the follow-up
that cuts the link bytes per prompt from per call to per span.

**Measured (M9, 2026-11-24, RTX PRO 6000, `--kv-dtype int8 --max-concurrency 4 --max-context
65536 --experts vram=45%,ram=55% --no-prefix-reuse`, prefill chunk 4096; 5,443 / 6,653 / 0
experts in VRAM / RAM / SSD, model 15.44 GiB + KV 0.40 GiB + expert cache 71.8 GiB ≈ 92.5 GiB
process VRAM):** one boot per build, `SIZES="8000 16000 32000"` (the words are not one token
each), streaming on and off on the same binary:

| Prompt tokens | Streamed off | Streamed on |
|---|---|---|
| 10,092 | 9.8 s TTFT, 1.03k tok/s | 7.1 s, 1.42k tok/s |
| 20,209 | 18.2 s, 1.11k tok/s | 12.4 s, 1.62k tok/s |
| 40,380 | 36.5 s, 1.11k tok/s | 24.9 s, 1.62k tok/s |

(+38 % at 10K, +46 % at 20K and 40K.) Greedy tokens, DFlash2 agreement lines and causal-scoring
log-probabilities are identical between the two builds (the engine test's 11,001-token prompt:
`stream: prefill expert stream: 4917 experts (64.82 GiB) in 4917 copies through 577 lent frames`);
decode after a long prompt is unchanged (a 256-token request after a 10K prompt: 37.5 tok/s with
the stream on and off, same binary; the ring returns before decode).

**Still design (from Maya, §4.2, each exact), not built here:**

- **Read-ahead of the next layer's SSD-only records** during layer l, in layer order, at full
  striped bandwidth, into a deep ring (Maya: 64-96 records; its reader had kept the NVMe at a
  quarter of its rate with 12). Prefill touches nearly every expert, so these reads are not
  guesses — the form the shared tier's fetch read-aheads already give a 64-record fetch (MoE §4.7).
- **Prestage during attention:** the link is idle while a layer's attention runs; the next layer's
  most-routed RAM experts are copied then (Maya `PRESTAGE`).
- **The last layer's tail for non-final tokens:** DFlash2's target features come from earlier
  layers' residuals (§16.2) and no other drafter exists, so layer 44's output projection and MoE
  are computed only for the prompt's last token, after its caches (latents, KDA state) are
  updated for every token (Maya `PREFILL_TAIL_SKIP`).
- **CPU share of the least-routed experts** (§12.4).

### 17.3 Compute

| Part | Route |
|---|---|
| Dense GEMMs and head | BF16 tensor cores (the existing dense kernels), bit-identical to the decode route; registered shapes at H = 4,096 (`src/models/shapes.cpp`) |
| Routed experts | W4A16 wide route, grouped by expert; ~910 tokens per expert per 32K span |
| MLA, indexer | §13.3-13.4 |
| KDA | chunked route |

### 17.4 Activation permissions

None in this design. The EXL3-era A8 permissions (8-bit rotated activations for the exact integer
prefill route) do not apply: the NVFP4 experts reuse Flash-Next's built W4A16 wide route (§9.5),
and the dense classes are BF16 tensor-core GEMMs already at the hardware's native precision. The
prefill compute lever is instead the **8-bit dense recipe** (option O9, §25.3), a conversion-time
choice behind the §20.3 gate; the prefix cache keys blocks by recipe, so a cached block is reused
only by the same arithmetic.

### 17.5 Short prompts and time to first token

A fresh prompt of ≥ ~1K tokens pays one whole expert pass, ≈ 6-6.5 s on the 5090 machine and
≈ 2.4-2.6 s on the production machine (§5.4). The levers, in order: the prefix cache (§18), the
CPU share for thin experts below ~1K tokens (§12.4), and the link. The user guide states the cost
plainly.

### 17.6 After prefill

No promotions after a walked prefill (MoE §7.3); the prompt's routes seed both policies, and the
prompt landing reserve stays off (§10.5). After a prompt returns its borrowed frames, the
free-frame fill rate of §10.2 refills them with the hottest RAM experts.

---

## 18. Prefix cache

The hybrid prefix cache binds as for Qwen4Exp ([spec §17](hybrid-prefix-cache-spec.md); the
family owns its prefix index, spec §17.1):

- **KV blocks:** 64-token content-addressed blocks of MLA latents and their BF16 pooled keys. Pools
  start at the sequence start, so a block holds exactly 16 complete pools and is self-contained;
  the incomplete pool's raw keys and gate scores are part of a snapshot.
- **State snapshots:** KDA state and conv state, 141 MB each, at the spec's tap points.
- **Value here is higher than on Flash-Next:** a hit avoids both the prompt compute and a whole
  expert pass (§5.4). Exactness holds because the expert arithmetic is placement- and
  route-invariant (§9.5).
- **Host tier default 2 GiB** (`--host-context-mib`). Each GiB costs ≈ 78 RAM expert slots
  (≈ 0.6 % of the bank); the ledger line names the trade, and agentic users may raise it.

---

## 19. Memory plans

### 19.1 VRAM

The sizing law, the spill guard and the monitor are the shared tier's (MoE §4.9); the tables below
apply it with this family's fixed terms (`plan_device` of the family's memory plan is the single
authority for constructor sizes). The token embedding stays host-resident (pinned, one row read
per token, as Flash-Next's) — the owner's "dense ≈ 19 GB resident" figure corresponds to also
keeping it in VRAM; this plan saves the 1.2 GiB.

**RTX 5090 machine (32,607 MiB card, no display, C = 1).** The frame budget follows the measured
Windows budget of 31,419 MiB (§3.1): 30.68 GiB.

| Item | Default, GiB |
|---|---:|
| In use before loading (context, libraries) | 0.55 |
| Dense Text, BF16 (`lm_head`, routers and mHC included; embedding host-resident) | **15.6** |
| KDA state, conv state, replay records | 0.20 |
| MLA latents + pooled keys, first 32K tokens (`bf16`; elastic) | 0.43 |
| Decode workspace, mailboxes, residency tables, io | 0.85 |
| Staging (32 frame-sized slots) | 0.42 |
| CUDA graph reserve; headroom | 0.25; 0.40 |
| **Frame pool** (of the 30.68 GiB budget) | **≈ 12.0** |
| Frames of 13.5 MiB | **≈ 900** (7.5 % of 12,096) |

At 128K actual context with `bf16` latents the KV takes ≈ 1.3 GiB more (−98 frames); with `fp8`,
−60. Each lane at C > 1 costs its KDA state (141 MB, ≈ 10 frames) and its KV extent.

**Production machine (RTX PRO 6000, 96 GB, Linux, C = 1; model).** The default startup form probes
a fully mapped KV pool (MoE §4.8); the bank does not fit, so the placement falls back to the
maximal RAM tier and the frame pool takes what is left:

| Item | Default, GiB |
|---|---:|
| In use before loading | 0.55 |
| Dense Text, BF16 | 15.6 |
| KDA state, conv state, replay records | 0.20 |
| MLA latents + pooled keys, `--max-context 131072` fully mapped (`bf16`) | 1.72 |
| Decode workspace, mailboxes, residency tables, io | 0.85 |
| Staging (32 slots) | 0.42 |
| CUDA graph reserve; headroom (the card also drives the display) | 0.25; 0.50 |
| **Frame pool** | **≈ 75.3** |
| Frames of 13.5 MiB | **≈ 5,700** (47 % of 12,096) |

At C > 1 each lane adds its KDA state and KV extent. `vram=100%` is refused: the bank (159.4 GiB
in frames) exceeds the card.

### 19.2 Host RAM

**RTX 5090 machine (95.8 GiB visible).** The ledger's law and terms are the shared tier's (MoE
§4.10); this family's constants:

| Item | GiB |
|---|---:|
| RAM expert tier, pinned (slots of 13.5 MiB), including the 64-slot landing ring | the rest |
| Token embedding, pinned | 1.18 |
| Prefix Host tier | 2.0 (default) |
| Program pinned buffers (mailboxes, io) | 0.25 |
| Process (tokenizer, server, frontend) | 1.5 |
| Reserve `--ram-headroom-mib`; growth allowance | 2.0; 1.0 |

| Available at startup | Expert tier | Slots | Share of the 12,096-expert bank (less 64 landing) | With VRAM frames |
|---:|---:|---:|---:|---:|
| 88 GiB (other programs closed) | 79.6 GiB | 6,040 | 49.4 % | 56.8 % |
| 79 GiB (this session's measurement) | 70.6 GiB | 5,360 | 43.8 % | 51.2 % |

Every GiB that other programs release becomes 78 more slots; the startup ledger prints the figure.
The pin room (MoE §4.11: shared system memory 76.61 GiB on this host) caps the pinned part at
≈ 76 GiB, so the top of the tier is unpinned records.

**Production machine (192 GB, 182 GiB usable).** Only ≈ 6,400 records (the VRAM pool's complement)
need RAM: the tier holds them all (≈ 86 GiB with the landing ring) and the remaining ≈ 90 GiB
stays with the OS, the page cache and other programs — the ledger reports it, and the SSD tier is
empty at this placement. Raising the placement's VRAM share (up to the frame pool's ceiling) and the
prefix Host tier are the only consumers.

### 19.3 Disk

**5090 machine:**

| Item | Size | Where (default) |
|---|---:|---|
| Source artifact (download) | 205 GB (191.0 GiB) | `E:` (deletable after verification) |
| Main artifact (includes the stored, unloaded `mtp` component) | ≈ 33 GB | `E:` |
| Expert volume 0, w = (5, 7), striped | 71.3 GB | `D:` |
| Expert volume 1 | 99.8 GB | `G:` |
| Expert state file | < 1 MB | beside the artifact |

**Production machine:** one unstriped volume (171.2 GB) beside the ≈ 33 GB main artifact and the
state file, all on the T710.

---

## 20. Numerics and qualification

The three-layer contract — the bit-stable exact route, the canonical arithmetic, fast paths under
the [acceptance policy](model-quality.md#6-numerical-acceptance-policy-and-fast-paths) — is the
engine's; this section keeps this family's arithmetic, oracles and gates and does not restate the
policy or the [Op development](op-development.md) qualification rules.

### 20.1 Op oracles

Each Op follows [op development](op-development.md#6-qualification): an independent naive FP32/FP64
oracle, exact oracles for exact transforms, packed inputs decoded independently with their stored
scales, real model shapes.

| Op | Oracle and criteria |
|---|---|
| `offloaded_sparse_moe` W4A16 at (4096, 2048) | **Bit-exact** across GPU frame, GPU staging, CPU on every ISA, every landing order of units, n = 1..8 and the wide route, against an independent FP64 oracle that decodes the packed words with their stored scales; the record relayout round-trips the source tensors bit-exactly. GLM routing (`noaux_tc`): exact against an FP64 sigmoid/bias/top-8 oracle on near-tie-free inputs, ties by lower id. Clamp boundaries (±10, NaN, Inf) |
| `mhc` | FP64 oracle of the reference formula including all 20 Sinkhorn rounds; gate and comb relative error ≤ 1e-6; the BF16 residual update reproduced as the reference forms it (§15.2) |
| `kda_input_prep`, `kda_replay_record`, fold | FP64 oracle; replay fold bit-exact against the committing form |
| `dsa_indexer` | FP64 pooling and scores; exact selection on near-tie-free fixtures; pool agreement on real activations |
| `mla_sparse_attention` | FP64 softmax attention over the latents of the selected set, absorbed and explicit forms, including the transposed `kv_b` route; every latent format |
| Expert volumes, per-volume read queues | Exact: every record of the first and last banks read through the queues equals the converter's bytes; permuted completion orders; fault seams |

### 20.2 Model-level checks

The whole model has no local reference runtime in the engine's sense. Flash-Next's method applies
(FN §17.1): per-block checks on the model's own activations against an FP64 reference built from
the decoded real weights (`tools/glm5_next/block_check.py`), and single-module agreement with
Transformers' modules (`upstream_check.py`), for every layer type, the MTP block and the head.

An offline **reference forward** (`tools/glm5_next/reference.py`, a qualification tool, not
a product route) runs the whole model teacher-forced, layer-major, streaming each layer's weights
from the disk once per pass (≈ 20-60 s of reads), so it fits either machine:

- **BF16 reference:** the artifact's own stored tensors (the BF16 dense classes and the NVFP4
  experts decoded exactly as §9.5 specifies), FP32 activations and accumulation. It isolates the
  engine's arithmetic from the quantization.

The model amplifies BF16-sized differences: on the 15-token forward-test prompt (2026-11-23) the
reference rounded at BF16 boundaries and the binary64 one differ by up to 4.6e-2 in a block's
residual and 0.22 in a position's logits, pick different routed sets at 26 % of the MoE positions,
and differ in argmax at two near-ties (MoE blocks 10 and 25 multiply an input difference by about
ten). The real forward test (`tests/models/glm5_next/test_forward_real.cpp`) therefore takes the
binary64 run as its oracle and the rounded run as the floor (within twice its deviation, the argmax
one of the two runs', routed sets equal about as often), and `block_check.py` qualifies each
block on the engine's own input, where every block measured stays below 4.4e-3.

### 20.3 Gates (fixed now, before any measurement)

Teacher-forced on frozen texts in the GLM chat template (code, document, chat, CJK, a tool-call
transcript; ≥ 2,500 positions), with `bf16` latents. Scored positions include a long-context block:
at least 512 positions after a prompt of ≥ 8K tokens, well past the 2,051 tokens below which the
indexer selects everything. strata-glm measured its KLs about twice as high there as on short texts
(§4.3), and short texts never exercise the selection.

| Gate | Against | Mean KL | p99 per-token KL | Top-1 agreement | Perplexity change |
|---|---|---:|---:|---:|---:|
| **Execution** (the engine, default settings) | BF16 reference forward | ≤ 0.002 nats | ≤ 0.05 | ≥ 99 % | ≤ 0.2 % |
| Fused mHC update (§15.2) | reference BF16-rounding form | not worse | | | |

**Quantization quality is not re-measured** (owner decision, 2026-11-23; there is no FP8-teacher
download and no quality report): the card's published numbers — perplexity 2.894 vs 2.788
(+3.8 %), mean KL 0.073, top-1 agreement 91.7 % (§7.4) — stand as upstream facts, and the gates
above stay defined against the BF16 reference forward on the same checkpoint.

### 20.4 Engine and system tests

- **Placement invariance end to end** (approximate expert mode off, §9.8): greedy ids identical
  with `--experts vram=0%,ram=100%`, a small and the full VRAM cache; RAM tier caps of 16, 48 GiB
  and full; one, two and three volumes; lookahead off and depth 1-3; every CPU ISA (AVX-VNNI,
  AVX2 without VNNI, scalar) and every core-type mix; randomized unit completion order (fault
  seam); CPU cap 0 and 16; a placement switch and back through `POST /admin/expert-placement`
  (MoE §4.8).
- **Faults:** unit read errors, a stalled volume, a missing or short volume, a silent CPU service,
  each reaching the recoverable path (the next request completes with correct ids; MoE §4.4).
- **Tier audit:** every expert is in exactly one tier (VRAM, RAM, or volumes only) plus at most one
  shadow, checked at every round boundary in debug builds and counted in release ("experts in no
  tier"). Maya's promotion-races-demotion bug left 12 % of its experts in no tier and cost 22 % of
  its real-chat speed without failing any output check (§4.2).
- **Policy conformance:** host decisions equal `tools/expert_cache_replay` step for step (the
  shared replay tool; MoE §4.3).
- **Other artifacts:** the same recipe on `RedHatAI/GLM-5.3-Flash-NVFP4` (same layout, §7.4) with a
  short generation.
- **Speed corpora:** Flash-Next's non-repeating session (NRS: distinct code, story, chat,
  multilingual prompts, ~5K generated tokens) for decode, reported by halves; a fixed
  teacher-forced routing corpus for hit-rate comparisons across builds, since greedy text drifts
  with rounding. The measurement method is the shared performance method (MoE §8.3).

---

## 21. Frontend and serving

- **Tokenizer.** Byte-level BPE from `tokenizer.json`. The Qwen-specific scanner refuses any other
  split pattern; GLM-4-family tokenizers group digits (`\p{N}{1,3}` instead of Qwen's `\p{N}`; M0
  confirms from the artifacts' files, which are not gated). The scanner gains that variant only;
  other patterns stay refused. Exact-oracle test against Hugging Face `tokenizers` over a
  multilingual corpus and the special tokens.
- **Template.** The GLM-5.3 Jinja template (`tools/chat_templates/glm5_3`) through the existing
  engine: the template's own role and section markers (M0 pins them verbatim from the artifacts'
  template file); tools as a `<tools>` block; `reasoning_effort` low/high/max (default max) mapped
  from OpenAI `reasoning_effort` and the Anthropic thinking budget; `clear_thinking`.
- **Reasoning.** The generation prompt ends with the thinking-opener tag, so every output session
  starts in the thinking channel; there is no thinking-off switch. Requests that ask for no
  thinking get `reasoning_effort: low` (documented), never an injected closing tag. Thinking
  budgets and `--reasoning-loop` work as for Qwen.
- **Tool calls.** A `glm` tool-call parser for the model's bracketed `NAME{K=V…}` call form
  (string values raw, others JSON, typed by the tool's schema; M0 pins the exact delimiters from
  the artifacts' template files), its tolerant recovery modes and its constrained-decoding grammar;
  results rendered in the template's tool-response form in call order. This is advertised OpenAI
  and Anthropic behaviour, so the schema tests and the serving documentation change together
  (AGENTS.md).
- **Serving.** `infernix-serve` serves the family; the expert-placement admin endpoint
  (`GET`/`POST /admin/expert-placement`, MoE §4.8) and the prefix cache apply unchanged.

---

## 22. Configuration surface

The engine-wide expert-residency group (`--experts vram=…,ram=…[,ssd=…]` or counts,
`--expert-placement FILE`, `--expert-swap`, `--expert-ram-mib`, `--expert-state`; MoE §4.8) is
shared and already served by `src/serve/serve_options.cpp`. This family adds:

| Option | Default | Meaning |
|---|---|---|
| `--expert-volume PATH` (repeat) | required | The artifact's expert volumes, any order; startup checks ids, sizes and headers |
| `--striped-volumes` | off | The striped record form (§7.2, §11); refused unless every volume was written striped |
| `--vram-headroom-mib N\|auto`, `--ram-headroom-mib N` | auto, 2048 | The sizing law's headroom terms (MoE §4.9-§4.10) |
| `--ssd-lookahead off\|1\|2\|3` | off | Lookahead depth (§11.5); the default changes only if M7 measures a gain on two volumes |
| `--cpu-experts auto\|off\|N`, `--cpu-expert-cores p\|pe` | auto (calibrated), auto | CPU team size; whether E-cores join it (§12.3) |
| `--spec dflash2 --draft-model FILE`, `--draft-tokens` | off | DFlash2 speculation (§16.2), the only drafter; the drafter is a companion artifact (FN §11.3's binding) |
| `--kv-dtype bf16\|fp8\|int8\|nvfp4` | `bf16` | **Changes output** |
| `--max-context N` | 131072 | Up to 1,048,576; the KV host tier above 128K (§13.6) |
| `--prefill-chunk N`, `--prefill-group N` | auto, 48 | Chunking; experts per prefill group (§17.2) |
| `--host-context-mib N` | 2048 | Prefix Host tier |
| `--reasoning-effort low\|high\|max` | max | Template default; requests may override |
| `--approx-experts off\|ssd\|vram`, `--approx-keep-map DIGITS\|PRESET`, `--approx-min-weight θ`, `--approx-experts-rule residency\|static` | off; Maya's map; none; residency | **Changes output.** The approximate expert mode (§9.8): scope, per-layer keep counts, weight threshold, rule |

Converter: `--model DIR --recipe glm5_3_flash_nvfp4_orcarouter` (§7.2); the DFlash2 companion with
`--components dflash2 --source dflash2=PATH --recipe glm5_3_flash_dflash2`. Planned (option, §7.3):
`--expert-volumes P0,P1,…`, `--expert-volume-weights w0,w1,…`, `--striped-volumes`. Tools:
`tools.artifact.restripe`, `tools.glm5_next.verify_artifact`, `tools.glm5_next.block_check`,
`tools.glm5_next.reference`; the SSD, pinning and CPU W4A16 probes join the maintainer
probe suite (the shape of `tools/flash_next_probe/`).

---

## 23. Implementation plan

| # | Milestone | Deliverables | Exit criteria |
|---|---|---|---|
| **M0** | Facts | Mostly done (2026-11-23): the artifact's config, checkpoint index and dtypes, the Transformers reference code (§2.2), the DFlash2 drafter's config (§16.2). Remaining: SGLang's GLM DFlash2 feature capture (PR sgl-project/sglang#36708) for the drafter's exact proposal mathematics; the tokenizer and chat template; C++ probes: pinning ceiling, 1.125 MiB reads with IOCP (`tools/glm5_next_probe/`), **CPU W4A16 at 13.5 MiB on P- and E-cores**, the T710 at this record size, host DRAM rate | §2, §3, §9.5 and §19 confirmed or revised; open questions of §25.2 closed; §5.3 re-run with the measured CPU cost |
| **M7** | Op geometry | `offloaded_sparse_moe`'s expert geometry (H, I) as a compile-time parameter of the CPU and GPU routes (§8.2); instances 2560 × 640 and 4096 × 2048 | `qwen4_exp`'s compiled code and bits unchanged: T1 hashes identical (`FAMILY=qwen4_exp`), the GLM instance passes the Op tests against its FP64 oracle |
| **M2** | New Ops with FP64 oracles | `mhc` (the Sinkhorn mixer + the BF16 residual update; distinct from the existing `hyper_connection` Op, §8.3); the `noaux_tc` routing variant; the SwiGLU clamp on dense, shared and routed MLPs; `kda_input_prep`; `dsa_indexer`; `mla_sparse_attention` (absorbed decode over selected tokens, prompt route) | Every §20.1 qualification passes |
| **M3** | Converter | `tools/convert/glm5_next.py` and the official recipe entry (§7.2) from the compressed-tensors checkpoint: experts to `nvfp4_expert_rg16_v1` records at GLM geometry, dense BF16 first; the DFlash2 drafter import (`--proposal`); volumes, restripe, `verify_artifact` | Every tensor round-trips bit-exactly; the volumes verify |
| **M4** | Functional family | `src/models/glm5_next/`: Engine, CLI, CausalScoring; plain greedy decode on the shared offload tier (VRAM + RAM); a Python reference forward (`tools/glm5_next/`) and golden; the registry value, backend and layering-test family | §20.2 checks; execution gate (§20.3); first end-to-end numbers; the owner sets the §1.3 targets from them |
| **M7** | Frontend and serving | Tokenizer, chat template, GLM tool-call and reasoning parsing, protocol schema tests | Tokenizer oracle exact; serving schema tests |
| **M6** | DFlash2 speculation | The drafter at hidden 4096 (companion artifact, target features by `ops::hyper_connection_stream_mean`), acceptance measured, the §16.3 cost model | Default chosen by measurement; never slower than plain beyond noise |
| **M7** | Prefix cache, lanes, decode performance | Prefix cache, multiple lanes, mixed rounds; `make t1`/`make t2` for `FAMILY=glm5_next` ([Decode profiling tiers](round-bench.md)); placement re-measurement including the owner's 50/20/30 observation (§3.1); route traces recorded on NRS and a broad corpus, then the policy replay of §10.2 and §10.4 against them | 128K decode ≥ 85 % of 4K; §9.3's budgets hold or get ncu; placement and policy defaults set by measurement |
| **M8** | Prefill and long context | Indexer prompt kernels and attention prompt route at speed, KV formats (`int8` next), the KV host tier beyond 128K | Prefill measured against §5.4; 128K prefill stable |
| **M9** | Vision | The `vision` component in the converter (BF16, as stored) and a re-conversion; the GLM tower (patch 14, temporal patch 2, merge 2, q/k norms, clamped SwiGLU, downsample and gated merger) with an FP64 reference; image and video prompts in the frontend | The tower against its reference; the frontend's image prompts against the Hugging Face processor |
| **Later options** | Behind their gates | The approximate expert mode (§9.8); 8-bit dense recipes (O9, §25.3); the 5090 machine and striped expert volumes (§11.6); EXL3 imports (O7, §25.4) | Each behind its own §25.3 gate |

Critical path: M0 → M1 → M2 → M3 → M4 → M6 → M7 → M8, with M5 parallel from the moment the
tokenizer files are read (they are available now) and M3 able to start beside M2 once M1's record
round-trips land. Rough size: ~20-26k new lines (attention family ~4k, mHC ~1k, KDA consumer ~1.5k,
expert-geometry parameterization and routing variant ~1.5k, tier extensions ~0.5k, DFlash2 binding
~1k, frontend ~2.5k, converter and tools including the reference forwards ~3.5k, tests ~7k), plus
the 205 GB download and GPU time for the campaigns. The offload tier itself is built; nothing is
extracted.

---

## 24. Documentation and authority changes

When the implementation lands: a user guide `docs/glm5_3-flash.md`; README models and benchmarks;
`docs/cli.md` and `docs/serving.md` (expert volumes, GLM tool calls, reasoning effort, the DFlash2
companion); a mathematics reference `docs/maintainer/glm5_next-model.md` from §2.2;
[storage layouts](storage-layouts.md) gains the GLM geometry instance of `nvfp4_expert_rg16_v1`
(bank geometry, global scales); [op development](op-development.md) records the W4A16
qualification at the second geometry instance; [artifact container](artifact-container.md)
(expert volumes); [engine architecture](engine-architecture.md) (the `Glm5Next` architecture value
in `src/models/registry.*`, the tier extensions of §8.2); [paged KV](paged-kv-cache.md) (latent
planes); [hybrid prefix cache](hybrid-prefix-cache-spec.md) (the GLM binding); the conversion guide
(the compressed-tensors NVFP4 import — its source reader exists — and the GLM entry). This file is
then removed.

---

## 25. Risks, open questions and rejected alternatives

### 25.1 Risks

| Risk | Effect | Mitigation |
|---|---|---|
| The BF16 dense reads dominate (16.7 GB per token) and no recipe lightens them | The all-hit ceiling caps at ≈ 69-82 tok/s (§2.3); decode sits well below Flash-Next's on the same card | The 8-bit dense recipe (O9) behind its quality gate; §5.5's levers in the guide |
| CPU W4A16 kernel slower than modelled | Host term grows beyond §5.1's DRAM floor | M0 probe first, on both hosts; E-cores; DMA share via the split table; the decode model re-run before M7 |
| Host DRAM slower than the ≈ 80 GB/s model | The §5.1 floor scales every RAM-served byte | M0 probes the production host; M7's placement re-measurement (including the owner's 50/20/30 observation, §3.1) |
| Compressed-tensors scale-layout details (scale group, global-scale placement, packing order) differ from the reading in §7.1 | Wrong weights | M0 from the checkpoint's first shards; the converter's round-trip oracle (§20.1) |
| §5.3's model is optimistic, as it was against strata-glm's measured run (§4.3) | Decode toward the strata-glm band on the 5090 machine | M4's first end-to-end numbers re-fit the model's overhead terms before M7's sweeps; M7 attributes one token with nsys |
| Hit rates below the central case (no traces yet) | Decode toward 8 tok/s on the 5090 machine | M7's traces first; policy candidates; the hardware levers of §5.5 in the guide |
| Windows pins less than the plan | Fewer pinned slots | The unpinned CPU-only sub-tier (MoE §4.11); measured in M0 |
| Drive thermal throttling under sustained reads (the probe ran 2-4 s) | SSD term grows | Per-volume telemetry and warnings (§11.7); heatsinks |
| `E:` contention if a volume is put there | Tail latency | Default placement avoids it |
| The DFlash2 target-feature convention differs from Qwen3.8-Flash-Next's | Acceptance collapses | M6 measures acceptance first; the convention is configurable until confirmed (§25.2) |
| BF16 tensor rate below the §5.4 estimate | Prefill below §5.4 | M8 measures it first; the 8-bit dense recipe (O9) behind its gate |
| Upstream ambiguities (DFlash2 feature capture, tokenizer pattern, top-k ties) | Wrong mathematics | M0 against the checkpoint, SGLang PR #36708 and vLLM; ids against Transformers on short prompts |
| The uncensored artifact's quality is published only on its card, and the quality report is dropped | Output quality | The card's published numbers stand (§7.4); the base NVFP4 build is the measured fallback; the execution gate (§20.3) is against the reference forward |

### 25.2 Open questions for M0

1. SGLang PR #36708's DFlash2 feature capture: the exact target features and positions (whether
   the stream-mean convention of §16.2 matches).
2. The tokenizer's split pattern and normalizer; EOS ids from the generation config.
3. The T710's rate at 14.16 MB records, and the production host's DRAM read rate (§5.3's B_dram).
4. The pinning ceiling and commit limit (page-file size) on the 5090 machine's Windows build and
   driver, re-measured for the 13.5 MiB slots (§10.4).
5. The CPU W4A16 cost per expert on P-cores and E-cores (§12.2), the model's least certain input.

### 25.3 Options (each behind its own measured gate)

| ID | Option | Gate |
|---|---|---|
| O1 | Striped expert volumes on the 5090 machine (§7.2, §11.6): the 12-unit form, landed-unit masks, per-drive depth caps | M7 A/B on the NRS corpus: striped vs unstriped two-volume decode |
| O2 | Hot-layer reservations or near-miss credit in the VRAM policy | The M7 replay rule (§10.2) |
| O3 | Draft-informed SSD prefetch | M7 recall ≥ 0.5 for SSD-tier experts, then a measured decode gain |
| O4 | FP8 pooled-key scan with FP32 rescoring at > 256K | Exact selection on fixtures; M8 speed |
| O5 | Hadamard-domain latent profiles | Qualification track |
| O6 | Linux io_uring path for the per-volume queues | Linux users; same tests |
| O7 | EXL3 imports (MikeRoz's and turboderp's builds, §7.4): the trellis codec, an `exl3` Linear, the E3A16 canonical arithmetic and a CPU kernel, ≈ 7k lines | An owner request; the full §20.1 qualification for the new codec |
| O8 | Mirrored volumes with per-record dynamic routing (strata-glm, §4.3): every drive holds every record, and each read goes to the drive that will deliver it first; adapts to a busy or throttling drive (the page file on `G:`), at 2× the disk (171 GB per drive, which `D:` and `G:` lack at 144/128 GB) | M7 A/B against weighted distribution on the NRS corpus, with background load on one drive |
| O9 | 8-bit dense recipe: dense text and `lm_head` quantized to 8 bits at conversion (dense bytes per token 16.7 → ≈ 9.6 GB, §5.5) | Teacher-forced KL of the 8-bit artifact against the BF16 artifact on the frozen texts, within the §20.3 execution gate; saves ≈ 6.6 GiB of VRAM |

### 25.4 Rejected alternatives

One line each; the rejections that decided the shared machinery are [MoE offload
design](moe-offload-design.md) §4.7, §5.8 and §6.3 and are not repeated here.

| Alternative | Reason |
|---|---|
| **The EXL3 builds as the plan** (MikeRoz's and turboderp's 3.05 bpw; the default until 2026-11-23) | A new codec and kernels ≈ 7k lines (trellis decode, an `exl3` Linear, the E3A16 canonical arithmetic, a CPU kernel) where the NVFP4 artifact reuses the built W4A16 kernels, the existing `nvfp4_expert_rg16_v1` layout and the converter source `tools/convert/sources/compressed_tensors.py`, which already reads `weight_packed`; kept as option O7 |
| **GCSA-AiLab/GLM-5.3-Flash-Uncensored-RCO-GSQ-GGUF** | Its "Q4" is 127.66 GiB with experts Q3_K ×105, Q2_K ×19, Q4_K ×2 (tensor counts), Q8_0 ×306 and BF16 ×203 other tensors; "Q6" is 218.76 GiB with no manifest; Q8_0 is 310.32 GiB; no KL or perplexity published for any of them. Its GSQ training did not improve on its starting point, so the stored weights are plain k-quants, and its abliteration is a different LoRA-merged edit. Serving it would need new Q2_K/Q3_K kernels (the engine's CPU ggml expert path supports IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS, IQ4_NL, Q2_0 only). The NVFP4 choice costs ≈ 30 % more expert bytes per token than that GGUF (model: 4.76 GB against ≈ 3.7 GB) — bought with published card numbers and no new kernels |
| Requantizing the artifact's weights to another format at import | Changes the weights for no speed gain; a lighter format is a recipe decision (O9), not an import transform |
| Storage Spaces / RAID 0 instead of application-level volume placement | Its 256 KiB default interleave does not align with the record reads; hides per-drive queues and health; needs admin setup; cannot weight unequal drives. A volume may still sit on a RAID set |
| DirectStorage | Stages through system memory, packed FP4 does not compress, needs D3D12 interop: no bandwidth gain over unbuffered reads into pinned memory |
| GPUDirect Storage | Linux and datacenter GPUs only |
| Memory-mapped experts / the page cache | Rejected by the shared tier (MoE §4.7); Flash-Next measured a mapped file's unbuffered reads at a quarter of the unmapped rate (FN §12.3), and llama.cpp measured 111× page amplification |
| Any SSD cache, spill or swap file | Violates the read-only rule; the volumes already hold the exact bytes (MoE §4.7) |
| Zero-copy SM reads of host experts | 10-14 GB/s against 24-27.6 staged (MoE §5.2's measurements on Flash-Next) |
| Expert deferral, dynamic top-k, low-bit miss copies, and expert skipping outside §9.8 | Change the model (§1.4); top-k 6 was rejected on quality by ox-boost. Skipping is offered only as §9.8's opt-in, labelled mode |
| BF16 KDA state | Saves ≈ 0.1 ms of a ~20 ms production token for a numerical change |
| Per-read hashing | ≈ 0.15 ms of CPU per record |
| Whole-layer double buffering in prefill | Expert groups free ≈ 500 frames with identical results (§17.2) |
| Folding W_uk into W_qb or W_uv into W_o | Changes the stored weights and reads more bytes per token (§13.1) |

---

## 26. Sources

- Model: [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (config, 76,108-tensor
  index, template); Transformers `src/transformers/models/glm5_next/modeling_glm5_next.py` and
  `configuration_glm5_next.py` (read 2026-11-23); GLM-5 report arXiv 2602.15763.
- Default artifact: [orcarouter/GLM-5.3-Flash-Uncensored-NVFP4](https://huggingface.co/orcarouter/GLM-5.3-Flash-Uncensored-NVFP4)
  (`config.json`, `model.safetensors.index.json`, quantization card, read 2026-11-23; files gated);
  [RedHatAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/RedHatAI/GLM-5.3-Flash-NVFP4) (same layout);
  [orcarouter/GLM-5.3-Flash-Uncensored-FP8](https://huggingface.co/orcarouter/GLM-5.3-Flash-Uncensored-FP8)
  (the quantization source).
- Rejected artifact: [GCSA-AiLab/GLM-5.3-Flash-Uncensored-RCO-GSQ-GGUF](https://huggingface.co/GCSA-AiLab/GLM-5.3-Flash-Uncensored-RCO-GSQ-GGUF)
  (tensor counts and sizes from its file list; no manifest for "Q6"; read 2026-11-23).
- EXL3 builds (survey and option O7): [MikeRoz/GLM-5.3-Flash-Uncensored-3.05bpw-h6-exl3](https://huggingface.co/MikeRoz/GLM-5.3-Flash-Uncensored-3.05bpw-h6-exl3)
  and [turboderp/GLM-5.3-Flash-exl3](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3) (branch
  `3.05bpw`): `quantization_config.json`, `model.safetensors.index.json`, `config.json`, read
  2026-10-11.
- Drafter: [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2)
  (`config.json`, model card, read 2026-11-23; Apache-2.0); ericlewis's third-party measurement
  (acceptance ≈ 0.70 at 3 draft tokens); SGLang PR sgl-project/sglang#36708 (the GLM DFlash2 feature
  capture).
- ExLlamaV3: `github.com/turboderp-org/exllamav3` (`exllamav3/modules/quant/exl3.py`,
  `exllamav3_ext/quant/codebook.cuh`, `exl3_dq.cuh`, `hadamard_inner.cuh`), read 2026-10-11 at
  `master`; QTIP (Tseng et al., arXiv 2406.11235); QuaRot arXiv 2404.00456.
- FreeToken: arXiv 2608.16157; `github.com/FlashML-org/FreeToken` (docs, PR #337); ports
  `github.com/anoane/02-glm53-flash-nvfp4-freetoken` and `github.com/Cerynitius/freetoken-ox-boost`.
- strata-glm: `github.com/sergqwer/strata-glm` (`README.md`, `docs/glm-port-plan.md`), read 2026-10-12.
- Other engines: `github.com/adrienbrault/glm5.3-flash-2x-rtx5090` (ExLlamaV3 offload fork),
  `github.com/Waltarb/nvmoe`, KTransformers (SOSP'25), llama.cpp discussion #27149,
  `github.com/deepseek-ai/FlashMLA`, `github.com/deepseek-ai/DeepGEMM`, `github.com/yatesdr/b12x`.
- Papers: Kimi Linear arXiv 2510.26692; mHC arXiv 2512.24880; Fate arXiv 2502.12224; Eliseev & Mazur
  arXiv 2312.17238; Fiddler arXiv 2402.07033; SP-MoE arXiv 2510.10302; MoE-SpeQ arXiv 2511.14102;
  "The Limits of Speculation" arXiv 2609.22156; Zhang, MoE expert caching arXiv 2608.07911 (via
  Flash-Next).
- Windows I/O: Microsoft "File Buffering"; IoRing API; DirectStorage developer guidance; NVIDIA forum
  thread on the pinned-memory limit.
- Project Maya: `github.com/mw00/project-maya` (`README.md`, `CHANGELOG.md`, `docs/GLM-FLASH.md`,
  `docs/SETTINGS.md`, `docs/WINDOWS.md`, `docs/GLM-KDA-GRAPHS.md`, `docs/QUANT-PLAN.md`,
  `bench/results/PROGRESS.md`, `bench/results/2026-10-05-glm-single-gpu/`, `bench/results/2026-10-06-glm-spec/`,
  `bench/results/MAYA-{S,S24,M,L}.md`), read 2026-10-11.
- ninferpro: the [MoE offload design](moe-offload-design.md) (the shared tier this family composes);
  [Flash-Next design](qwen3_8-flash-next-design.md) and [guide](../qwen3_8-flash-next.md) (its
  measurements, cited as FN §x); the storage measurements of §3.3 (2026-10-10, the 5090 machine).
