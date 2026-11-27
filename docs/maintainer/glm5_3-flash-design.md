# GLM-5.3-Flash: design

This document is the design authority for `Glm5NextForConditionalGeneration` (GLM-5.3-Flash),
model family `glm5_next` (the name is the Hugging Face model type). The family runs on the shared
offload tier: the expert banks, residency, tiers, placement, the CPU expert engine and decode and
prefill on an offloaded MoE are designed once in [MoE offload design](moe-offload-design.md),
cited below as **MoE §N**. This document keeps the family's own facts, formats, kernels,
constants, memory and measurements, and cites [Qwen3.8-Flash-Next's
design](qwen3_8-flash-next-design.md) as **FN §x** for that family's measurements. Commands and
flags are in [CLI](../cli.md) and [serving](../serving.md).

Built: the Engine, `infernix`, `infernix-serve`, `infernix_bench` and causal scoring; routed
experts in any VRAM : RAM : SSD ratio; BF16 and INT8-G64 latent caches; DFlash2 speculation for
one-row rounds; prefill by streamed experts and layer-major walk spans; the prefix cache; image
and video prompts. Qwen3.8-Flash-Next stays the production default; this family serves with
`MODEL_DIR=GLM-5.3-Flash make -C ~/src/super-quant serve-infernix`
(`~/src/super-quant/configs/GLM-5.3-Flash/model.env`). Every option designed but not built, and
every rejected option, is in §17 with its reason.

Every number is labelled **measured** (date, machine and placement named), **upstream** (the
model card, reference code or artifact metadata, §19) or **model** (computed from stated inputs).
Every measurement ran on the production machine — one RTX PRO 6000 Blackwell (96 GB, `sm_120a`,
188 SMs, 1.79 TB/s), PCIe Gen5 x16, a Ryzen 9 9950X3D2 with 182 GiB of usable RAM, one Gen5 NVMe
(T710) — on 2026-11-23 to 11-24, unless it says otherwise.

---

## Contents

1. [Status and measured performance](#1-status-and-measured-performance)
2. [Model facts](#2-model-facts)
3. [Artifact and conversion](#3-artifact-and-conversion)
4. [Ownership in the engine](#4-ownership-in-the-engine)
5. [Decode round](#5-decode-round)
6. [Expert residency, placement and miss service](#6-expert-residency-placement-and-miss-service)
7. [Sparse MLA attention and the pooled indexer](#7-sparse-mla-attention-and-the-pooled-indexer)
8. [Kimi Delta Attention](#8-kimi-delta-attention)
9. [Manifold-constrained hyper-connections](#9-manifold-constrained-hyper-connections)
10. [Speculative decoding](#10-speculative-decoding)
11. [Prefill](#11-prefill)
12. [Prefix cache](#12-prefix-cache)
13. [Memory](#13-memory)
14. [Numerics and qualification](#14-numerics-and-qualification)
15. [Frontend, vision and serving](#15-frontend-vision-and-serving)
16. [Configuration](#16-configuration)
17. [Rejected and not built](#17-rejected-and-not-built)
18. [External evidence](#18-external-evidence)
19. [Sources](#19-sources)

---

## 1. Status and measured performance

**Serving configuration** (`configs/GLM-5.3-Flash/model.env`): the BF16-dense artifact,
`--experts vram=40%,ram=60%`,
INT8 latents, `--max-context 262144`, four lanes, DFlash2 `--draft-tokens auto` on one-row rounds
(§10.2), the prefix cache with a 2 GiB Host tier, and `--vision` with
the tower streamed from pinned RAM. The placement puts
4,838 experts in VRAM, 7,258 in RAM and none on the SSD. The whole process stays under the 95 %
VRAM cap (92,993 MiB). KV is allocated for every lane at the full context at startup, and with the
prefix cache on, its KV holds every lane's full context (6.36 GiB INT8 at 4 × 262,144).

The serving build (2026-11-27: 3-term `expert-a4-residual` wide prefill, 4-column weight-reusing
narrow A16 rows with pre-wait down-weight preload, cold-placeholder pre-demotion, and 14/16/120
miss split, §6.3 and §14), `infernix_glm5_next_engine_bench --tokens 512 --repeat 4` at the serving
split (model 15.44 GiB, 16.70 with the drafter; KV 1.59 GiB at one lane, 6.36 at four; experts
63.8 GiB, 4,838 VRAM / 7,258 RAM / 0 SSD), median and range:

| Requests | Plain | DFlash2 `auto` (greedy) | DFlash2 `auto` (T = 1.0) |
|---|---|---|---|
| 1 | 42.0 (39.9-43.5) | 54.5 (49.3-56.1) | 55.4 (47.9-58.5) |
| 4 | 72.0 (65.2-76.0) | | |

Prior 2026-11-24 baseline at the same split (512 tokens, 4 sets): 1-request plain 36.9 (33.2-38.0),
K = 2 44.4 (40.8-46.4), K = 3 45.8 (34.6-45.8), 4-request plain 69.5 (60.2-70.8), 4-request with
drafter loaded 64.8 (60.0-65.5). Prefill (`infernix_bench`, INT8 KV, chunk 4,096, `max-ctx 65536`,
`work peak 1.16 GiB`, 2026-11-27): `pp4096` 2.40k (PCIe DMA-bound), `pp16384` 5.56k (+60.2 % over
exact 3.47k), `pp32768` 5.41k (+59.1 % over exact 3.40k); live serving 20.5K fresh prompt beside
three concurrent decodes measured 5.33k tok/s at peak process VRAM 92,670 MiB (cap 92,993).

The Q8-dense artifact (§17.2; not served: it misses the KL gate) against the BF16-dense one at the same split (`vram=40%,ram=60%`, 4,838 / 7,258 / 0
experts, plain decode, `infernix_glm5_next_engine_bench` 512 tokens, median and range of four sets;
fresh prompts on `infernix-serve`, 2026-11-24):

| Measurement | BF16 dense (model 15.44 GiB) | Q8 dense (model 9.17 GiB) |
|---|---|---|
| Decode, one request | 35.4 (30.9-36.5) tok/s | 38.6 (36.1-41.2) tok/s |
| Decode, four requests, aggregate | 64.6 (58.6-67.5) tok/s | 67.6 (64.1-69.9) tok/s |
| Prefill, 10K and 40K-token prompts | 2.51-2.58k tok/s | 2.42-2.46k tok/s (≈ 4 % slower: the MMA routes dequantize in the loop) |
| Serving process at `vram=44%` with the drafter, the 9.4K-prompt load below | — | peak 92,516 MiB (model 10.43 GiB with the drafter) |
| Teacher-forced KL against BF16 dense, quick corpus, context 8192, stride 4096, INT8 KV | — | mean 0.0058, p99 0.066, top-1 97.3 %, PPL 3.1515 against 3.1489 |

The tuned Q8 shapes (`src/models/glm5_next/shapes/q8/`) did not move one-request decode against
the generic routes (38.7 tok/s). The rows below are the BF16-dense artifact's.

| Measurement (serving split unless named) | Result |
|---|---|
| Decode, one request, DFlash2 K = 3 (`infernix_glm5_next_engine_bench`, 512 tokens, four sets after a warm-up) | median 41.1 tok/s, sets 37.1-42.7 |
| Decode, four requests of 256 tokens (rounds of several rows decode plainly) | median 59.9 tok/s aggregate, sets 56.5-62.0 |
| Same bench at `vram=42%,ram=58%` (5,080 / 7,016 / 0 experts), one request, by draft length | K = 0 35.0, K = 1 40.9, K = 2 44.4, K = 3 47.8, K = 4 43.2, K = 5 37.7 tok/s (§10.3) |
| Process VRAM while serving: four 1,024-token decodes beside a 9.4K prompt and its cached repeat | peak 92,412 MiB; at vram=42 % it sat at 95,672 MiB |
| Same with `--draft-tokens auto` (48 staging slots against 32) and `--vision`: three 1,024-token decodes beside a 20.5K fresh prompt (5.33k tok/s) and its cached repeat, 2026-11-27 | peak 92,670 MiB (cap 92,993; headroom 323 MiB; 2026-11-26: 92,680 MiB) |
| Same with `--vision`: two decodes, the 9.4K prompt and an image (551-token prompt), then a video (1,544 tokens) | peak 92,412 MiB; the image answer reads the chart's title |
| Prefill, fresh prompts of 10K-38K tokens, walk spans, `vram=42%,ram=58%` | 2.35-2.43k tok/s, compute-bound (§11.1) |
| One 255,771-token prompt, INT8 KV, `vram=42%,ram=58%` | prefill 2.09k tok/s (TTFT 2 min 2 s), then 256 tokens at 33.3 tok/s |
| Decode after a 130,164-token prompt against after a 3,964-token one (one lane, 256 tokens) | 36.8 against 38.1 tok/s (97 %) |

At the serving split the engine's VRAM ledger reads: model 16.70 GiB, KV 1.59 GiB at one lane and
6.36 GiB at four, expert cache 63.8 GiB (4,838 frames). These speeds sit far below Flash-Next's
(380 tok/s at `vram=100%`, FN §1.3), because this model reads ≈ 16.7 GB of BF16 dense weights per
token (§2.3), moves 14.16 MB per miss, and misses far more often at an equal cache share (§18.1).

## 2. Model facts

### 2.1 Configuration (upstream)

From `zai-org/GLM-5.3-Flash` `config.json` and Transformers `configuration_glm5_next.py`. The
loader checks the artifact's own `config.json` against these.

| Quantity | Value |
|---|---|
| Layers | 45 + 1 MTP layer (index 45). Layer i is KDA when i % 4 ≠ 3, sparse MLA when i % 4 = 3: 34 KDA, 11 MLA (3, 7, …, 43) |
| Hidden width | H = 4096; RMSNorm eps 1e-5, weight multiplies (no unit offset) |
| Residual | mHC, 4 streams (16,384 wide), two mixers per layer, 20 Sinkhorn iterations, eps 1e-6 |
| KDA | 64 heads, head dimension 128, short conv kernel 4 (per q/k/v, SiLU), gate lower bound −5, FP32 state 64 × 128 × 128 per layer. Separate `q_proj`, `k_proj`, `v_proj` [8192 × 4096] and separate `q_conv1d`, `k_conv1d`, `v_conv1d` weights [8192, 1, 4] |
| Sparse MLA | 64 heads; q_lora_rank 1536; kv_lora_rank 512; qk_nope 256, **qk_rope 0 (no positional encoding anywhere in the text model)**; v_head 256; softmax scale 256^−½ |
| Indexer (every MLA layer `full`) | 32 heads × 128, LayerNorm keys, softmax-gated pools of 4 tokens with a learned positional bias, relu scores weighted per head, top 512 pools (2,048 tokens) plus the incomplete tail |
| MoE | Layers 0-2 dense SwiGLU (I = 12,288); layers 3-44: 288 routed experts, top-8, width 2,048, 1 ungated shared expert of width 2,048 |
| Router | FP32 logits, sigmoid scores; selection on scores + `e_score_correction_bias`; weights = unbiased scores of the 8, normalized, × 2.5 (`noaux_tc`, n_group = topk_group = 1) |
| SwiGLU clamp | gate ≤ 10, up ∈ [−10, 10] (`swiglu_limit` 10), on routed, shared and dense MLPs |
| Vocabulary | 154,880; untied `lm_head`; final RMSNorm after the unweighted mean of the 4 streams |
| Context | 1,048,576 positions (no RoPE, so no YaRN question) |
| MTP layer | enorm, hnorm, `eh_proj` [4096 × 8192], one sparse-MLA block with its own indexer and MoE (288 routed + shared), `shared_head.norm`, the main `lm_head`; no hyper-connections. Stored in BF16 in the default artifact, including its 288 routed experts; the engine does not load it (§17.1) |

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

This is exactly the `kimi_delta_attention` Op contract (lower bound in [−5, 0]): the Op takes the
raw gate input `g_in` with `dt_bias` and applies the sigmoid itself, including β.

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
them unspecified).

**MoE** (`Glm5NextTextMoE`): `s = sigmoid(W_r x)` (FP32 logits from the FP32 `gate.weight`);
`top8 = argtop8(s + bias)` (the group mask is a no-op: n_group = topk_group = 1);
`w_i = 2.5 · s_i / (Σ_{top8} s + 1e-20)`; `y = Σ_i w_i E_i(x) + E_shared(x)` with
`E(x) = W_d (SiLU(min(W_g x, 10)) ⊙ clamp(W_u x, −10, 10))` — the clamp applies to the dense,
shared and routed MLPs alike.

**NVFP4 expert** (compressed-tensors `nvfp4-pack-quantized`): `W = dequant(W_packed, W_scale,
W_global_scale)` with two FP4 E2M1 values per byte, one FP8 E4M3 scale per 16 inputs and one FP32
global scale per matrix; the engine computes `y = W x` at BF16 activations (W4A16, §5.3).

Where the reference's precision choices (BF16 residual update, BF16 pooled keys, the FP32 norms
and the clamp scope) look unintentional they are still the oracle's; §9 and §14 state where the
engine computes more precisely and how that is bounded.

### 2.3 Parameters and bytes

| Item | Parameters | Bytes stored (NVFP4 artifact) | Bytes read per decode token |
|---|---:|---:|---:|
| One routed expert: gate, up [4096 → 2048] and down [2048 → 4096], FP4 packed + FP8 E4M3 scale per 16 inputs + FP32 global scale | 25,165,824 | **14,155,776 = 13.5 MiB = 3,456 × 4 KiB** (§3.2) | — |
| Routed bank, 42 × 288 | 304.4 B | **171.2 GB (159.4 GiB)** | 336 experts: **4.76 GB** |
| Dense text: KDA projections 4.68 B, MLA 1.37 B, shared experts 1.06 B, dense MLPs 0.45 B, routers FP32 0.05 B, mHC/conv/`A_log`/`dt_bias`/norms ≈ 0.05 B | 7.66 B | **≈ 15.4 GB** (BF16, FP32 where the vendor kept it) | same |
| `lm_head` (BF16, untied) | 0.63 B | 1.27 GB | whole matrix |
| Token embedding (BF16) | 0.63 B | 1.27 GB | one row (host-resident) |
| MTP layer 45, BF16 | 7.44 B | ≈ 14.9 GB | — (not loaded) |
| Vision tower (24 blocks, BF16) | 0.56 B | ≈ 1.13 GB | — |
| **Whole artifact** (index `total_size`) | | **191.0 GiB in 65 shards** | |

What follows from these bytes (model, at the card's 1.79 TB/s):

- **The dense BF16 reads rival the experts.** ≈ 16.7 GB of dense weights (including `lm_head`)
  per token against 4.76 GB of all-hit routed weights: an all-hit token reads ≈ 21.8 GB, ≈ 12.2 ms,
  a ceiling of ≈ 82 tok/s. The measured round profiles spend 8.1-8.4 ms in the dense GEMVs (§6.5,
  §10.3). An 8-bit dense recipe is the largest single lever and is not built (§17.2).
- **A miss moves 14.16 MB**, 5.1× Flash-Next's 2.76 MB record.
- **The KV cache is small.** One MLA latent is 512 values per token and layer: 11 MLA layers ×
  1 KiB = 11 KiB per token in BF16, plus 64 B of pooled index key per layer.
- **The recurrent state is large.** 34 × 64 × 128 × 128 × 4 B = 136 MiB of FP32 KDA state per
  sequence, read and written every token.
- **No n-gram table.** The official config has no n-gram keys and its 76,108-tensor index holds no
  table, so this family has no n-gram drafting (§17.3).

## 3. Artifact and conversion

### 3.1 The default artifact

[orcarouter/GLM-5.3-Flash-Uncensored-NVFP4](https://huggingface.co/orcarouter/GLM-5.3-Flash-Uncensored-NVFP4)
(the owner's choice): compressed-tensors `nvfp4-pack-quantized`, 191.0 GiB in 65 shards, the
layout of `RedHatAI/GLM-5.3-Flash-NVFP4`. The abliterated edit is orcarouter's, baked into its
block-FP8 checkpoint `orcarouter/GLM-5.3-Flash-Uncensored-FP8` before quantization. Only the routed
experts of layers 3-44 are quantized; every other tensor is BF16 dequantized from the vendor's
block FP8, with FP32 kept where the vendor had FP32.

| Tensor class | Stored | Engine form |
|---|---|---|
| Routed experts, 42 × 288 | per expert and matrix (`gate_proj`, `up_proj`, `down_proj`): `weight_packed` uint8, `weight_scale` FP8 E4M3 (one per 16 inputs), `weight_global_scale` FP32 | Packed words and per-16 scales relaid out verbatim as `nvfp4_expert_rg16_v1` records at GLM geometry (§3.2); the three global scales per expert ride the bank's binding |
| Everything else | BF16; F32 for `hc_*_base`, `hc_*_scale`, `A_log`, `dt_bias`, router `gate.weight`, `e_score_correction_bias` | Direct |
| `hc_attn_fn`, `hc_ffn_fn` | BF16 [24, 16384] (upcast to FP32 in the Op) | Direct |
| MTP layer 45 | BF16 throughout, including its 288 routed experts | Stored as the `mtp` component; not loaded |
| Vision tower | 24 blocks, BF16, plus `processor_config.json` | The `vision` component, every tensor BF16 as stored |

The checkpoint declares FP4 input activations (vLLM runs it W4A4). This engine runs the experts at
**A16**: the exact W4A16 arithmetic and record layout Qwen3.8-Flash-Next already uses (FN §6.3), so
no new expert kernel or codec was needed, and the activation side is strictly more precise.

**Quality is the card's published numbers.** Perplexity 2.894 against 2.788 for the FP8 source
(+3.8 %), mean KL 0.073 and top-1 agreement 91.7 % (upstream). The engine does not re-measure the
quantization (owner decision); its qualification is against references built from the same
checkpoint (§14).

### 3.2 Recipe and expert record

One recipe, `glm5_3_flash_nvfp4_orcarouter` (`tools/convert/official_recipes.py`), imports the
artifact as stored: `tools/convert/glm5_next.py` binds the parameters, the expert banks are written
by `tools/convert/sources/nvfp4_banks.py`, and the dense BF16/FP32 tensors pass through unchanged.
The converter refuses a compressed-tensors NVFP4 artifact whose quantized set or scale layout
differs (other quantized classes, a scale group other than 16 inputs, missing global scales),
naming the property. The DFlash2 drafter is a companion artifact
(`--components dflash2 --source dflash2=PATH`, recipe `glm5_3_flash_dflash2`).

**Record layout `nvfp4_expert_rg16_v1` at GLM geometry** (H = 4,096, I = 2,048; the
[storage layout](storage-layouts.md) with the geometry a compile-time parameter of the Op, §4.2).
A work unit is a 16-input block of a 16-row group: 16 rows × (8 packed bytes + 1 scale byte) =
144 B.

| Section | Content | Groups × blocks × 144 B | Bytes |
|---|---|---|---:|
| gate/up | row groups interleaved gate/up (2 × I / 16 = 256 groups), K / 16 = 256 blocks each | 256 × 256 × 144 | 9,437,184 |
| down | H / 16 = 256 row groups, I / 16 = 128 blocks each | 256 × 128 × 144 | 4,718,592 |
| **Record** | | | **14,155,776 = 13.5 MiB = 3,456 × 4 KiB** |

Scales live inside the record and every record is 4 KiB-aligned. The layout id carries no
geometry: H and I are properties of the bank, recorded in its binding. The records live in the
`.infernix` file set, and the shared SSD tier reads them in place with unbuffered reads (MoE §4.7);
that layout is what lets the converted artifact stream experts where a GGUF or safetensors file
read directly would not.

`kv_b` ([64 × 512 × 512] BF16, per head `[W_uk; W_uv]`) is stored as the Ops read it: `value_up`
[64 × 256 × 512] holds each head's W_uv as stored, and `key_up` [64 × 512 × 256] holds each head's
W_uk transposed, both BF16, so the absorbed products (§7.1) read one contiguous row per output. The
converter transposes it once.

### 3.3 The GGUF artifact

[GCSA-AiLab/GLM-5.3-Flash-Uncensored-RCO-GSQ-GGUF](https://huggingface.co/GCSA-AiLab/GLM-5.3-Flash-Uncensored-RCO-GSQ-GGUF)
`Q4/GLM-5.3-Flash-Uncensored-GSQ-RCO-Q4.gguf` (llama.cpp arch `glm5-next`, 1,383 tensors) imports
bit-exactly: `tools.convert --model <orcarouter snapshot> --gguf <file> --recipe
glm5_3_flash_gcsa_gguf` (`tools/convert/glm5_next_gguf.py`). Its abliteration is GCSA's, not
orcarouter's: the plain tensors equal the orcarouter checkpoint's, but the residual-writing
matrices (`attn_output`, `ffn_down`, the shared expert's down) and the embedding differ.

| Tensor class | GGUF | Artifact |
|---|---|---|
| Routed experts, layer 3 | Q4_K gate/up, Q3_K down | `ggml_expert_q3_k_v1` banks, records verbatim (13.0 MB) |
| Routed experts, layers 4-37 | Q3_K | `ggml_expert_q3_k_v1` (10.8 MB) |
| Routed experts, layer 38 | Q3_K gate/up, Q2_K down | `ggml_expert_q2_k_v1` (9.96 MB) |
| Routed experts, layers 39-44 | Q2_K | `ggml_expert_q2_k_v1` (8.26 MB) |
| KDA q/k/v/o, MLA query up and output, dense MLP, shared expert | Q8_0 | `q8_g32_fp16`, the same codes and scales |
| `kv_a_mqa`, `k_b`, `v_b`, KDA beta and gate/forget ranks | BF16 | BF16 (`k_b`/`v_b` are already the absorbed blocks of §3.2) |
| Norms, router, mHC, indexer, `dt_bias`, convolutions | F32 | BF16 where every value is a BF16 value, FP32 where the engine binds FP32; the three KDA kernels fused as §8 reads them |
| Token embedding | Q8_0 | BF16 rows: d × code rounded once |
| `lm_head`, MLA `q_a`, KDA `A_log` | Q8_0, Q8_0, A = -exp(A_log) | from the checkpoint (the head and `q_a` BF16, because the head's rows and `q_a`'s packing with the BF16 latent need BF16), each checked against the GGUF: every Q8_0 value within half a step plus its binary16 scale's rounding of the checkpoint's, and -exp(A_log) within one binary32 ulp of A |
| MTP layer, vision tower | absent | no MTP; the tower from the checkpoint |

The routed experts take 118.1 GiB against the NVFP4 artifact's 159.5 GiB. The experts run the exact
A16 ggml arithmetic (`src/ops/ggml/ggml_exact.h`; Q2_K's per-16 minimum term reads each column's
16-element X sums) on the GPU and on the CPU miss lane, so a miss's route never changes the bits.
Up to 8 columns (`moe::kMaxColumns`) run the exact narrow dp4a kernels (`ggml_dp4a.cuh`, block-level
4-lane cooperative `add_block_quad` with `cp.async` row/block staging); wider calls (prefill) on the
exact route run the same arithmetic on tensor cores (`ggml_tile.cuh` decode-in-loop tiles, which form
the X sums from the activation limbs, at this family's H4096 x I2048 geometry with the clamped
SwiGLU), bit-equal to the narrow route. By default the `ggml-bf16-prefill` fast path (§14) runs
BF16 tensor-core tiles (`ggml_bf16_tile.cuh` `compute_decode`, one CTA per SM over (tile, row block)
work items with templated 128/64/32/16-column warp layouts, conflict-free shared-memory staging and
`__byte_perm` unsigned BF16 decode): calls of 64 or more columns (`kGgmlBf16AllJobsMinColumns`) route
every active non-CPU job through the BF16 tiles directly from BF16 x (skipping limb encoding, the
exact-tile plan and the narrow dp4a pass), while calls of 9-63 columns keep narrow dp4a for jobs of
at most 8 entries. Each 128 x 128 tile copies its rows' raw 256-element units by cp.async one unit
ahead and decodes them in the K loop, x and h stay as stored, and the sums are FP32. Q3_K's and
Q2_K's code times sub-scale fits 8 bits, so it is stored exact in BF16 and the MMAs sum a whole block
before one multiply by d; Q4_K and Q8_0 scale per slice. A 4,096-token chunk gives an expert about
114 entries, one tile, so a decoded plane (the default geometry's form) would be used once. The
record strides differ by layer, so the expert cache keeps one frame pool per stride and gives every
layer the same frame count (a switch target rounds down to a multiple of 42). The RAM tier sizes
every slot at the largest stride (layer 3's 13.0 MB), so it holds about 20 % fewer records than its
bytes would at the mean stride.

Measured against the NVFP4 artifact (2026-11-27 optimized build; `infernix_glm5_next_engine_bench`,
four sets after warm-up; `infernix_bench` INT8 KV chunk 4,096; `infernix-perplexity` quick corpus,
context 8192, stride 4096, INT8 KV, 252,823 positions):

| Measurement | NVFP4, `vram=40%` (4,838 VRAM / 7,258 RAM / 0 SSD) | GGUF, `vram=59%` (7,098 VRAM / 4,998 RAM / 0 SSD) |
|---|---|---|
| Decode, one request, `auto`, greedy (512 tok, 4 sets) | 54.5 (49.3-56.1) tok/s (prior: 48.0, 43.9-52.5) | 87.8 (76.7-88.9) tok/s, sets 1-4 (prior: 85.3 sets 4-10) |
| Decode, one request, `auto`, temperature 1 (512 tok, 4 sets) | 55.4 (47.9-58.5) tok/s (prior: 47.0, 42.3-48.0) | 73.6 median, 54.2-77.2 across sets 1-4 (2026-11-27 prior; 2026-11-26 sets 4-10: 75.1, 72.7-80.7) |
| Decode, one request, plain (512 tok, 4 sets) | 42.0 (39.9-43.5) tok/s (prior: 36.9, 33.2-38.0) | 56.2 (47.0-60.6) tok/s (2026-11-27) |
| Decode, four requests, plain (4 sets; NVFP4 512 tok, GGUF 256 tok) | 72.0 (65.2-76.0) tok/s aggregate (prior: 69.5, 60.2-70.8) | 118.1 (103.4-124.2) tok/s aggregate (prior: 81.2, 65.8-88.7) |
| Same split for both, `vram=40%`, T1 `round_bench` (`1x1` / `4x1` / `1x4`, 2026-11-27) | 38.1 / 93.5 / 82.8 tok/s (prior: 27.8 / 70.0 / 47.9) | 45.7 / 117.0 / 100.0 tok/s (prior: 34.2 / 85.0 / 62.0) |
| Prefill (`infernix_bench` `pp4096` / `pp16384` / `pp32768`, INT8 KV, 2026-11-27) | 2.40k / 5.56k / 5.41k tok/s (exact: 2.38k / 3.47k / 3.40k; live 20.5K beside 3 decodes: 5.33k) | 4.21k / 4.05k / 3.92k tok/s (`pp256`: 305.0 tok/s; live serving 16.1K beside 3 decodes: 3.95k tok/s; 2026-11-26: 2.24k-2.33k tok/s, exact route 1.56k-1.59k) |
| Serving process, production flags, three 1K decodes beside a 16-20K prompt and its cached repeat | peak 92,670 MiB (cap 92,993, headroom 323 MiB) | peak 91,986 MiB (cap 92,993, headroom 1,007 MiB; 2026-11-26: 91,984 MiB) |
| Perplexity (different abliterations: not a measure of the quantization) | 3.1489 (exact) / 3.1479 (`expert-a4-residual`, PPL ratio 0.99967) | 3.2546 (2026-11-27; 2026-11-26: 3.2541) |

No published reference shares this artifact's abliteration, so its quantization loss is not
measured. GCSA's Q8_0 build of the same repository is not one: against the Q4 build, `attn_output`
of layers 15-44 (the 30 layers its adapter targets) differs by a rank-1 term beyond any rounding,
and layer 20's routed experts differ 4-6 times more along that direction than along a random
one, while layers 0-14 match byte for byte or as another rounding (2026-11-26, the converter's
`require_shared_tensors` refuses it for `--gguf-experts`).

A placement switch that rebuilds the RAM tier runs off the engine thread ([MoE offload
§4.8](moe-offload-design.md#48-placement-forms-and-the-runtime-switch)): switching `ram=41%` to
`25%` and back under a streaming decode (2026-11-26, `vram=59%`) left the longest gap between
tokens at 318 and 411 ms, against 17 s and more when the engine thread pinned, copied and freed
the tiers itself.

## 4. Ownership in the engine

### 4.1 The family

| Component | Owner | Location |
|---|---|---|
| Config interpretation (exact key set, refusals), mathematics, binding, block order, DFlash2 alignment | Model | `src/models/glm5_next/` (`config.*`, `weights.h`, `load.*`, `model.h`, `instance.*`, `options.*`, `sampling_defaults.h`, `execution/`, `shapes/`, `frontend/`) |
| Lanes, KDA state, MLA latent KV and index keys, the prefix cache binding, the DFlash2 drafter, graphs, the offload-tier composition, the prefill walk, vision encodes | Program | `src/models/glm5_next/program/` |
| Engine instance | Model | `instance.{h,cpp}`, instantiating `EngineBackendFor` over its own instance. The family enters `src/models/registry.*` only through `detect_artifact_architecture`, `family_option_rules` and `construct_engine_backend`, registers its linear shape tables in `src/models/shapes.cpp`, and is in `tests/test_layering.py`'s `FAMILIES` |
| Tokenizer pre-tokenizer, chat template, tool-call and reasoning parsing, image and video processing | Frontend | `src/models/glm5_next/frontend/` (§15) |
| Recipe, compressed-tensors import, block checks and reference forwards | Converter and tools | `tools/convert/glm5_next.py`, `tools/convert/sources/compressed_tensors.py`, `official_recipes.py`, `tools/glm5_next/` |

The vision weight stream and the pinned tower layout are shared with Qwen3.8-Flash-Next through
`src/models/common/vision/` (§15.2).

### 4.2 The shared offload tier and what this family added

The family composes `src/models/offload/` as Qwen3.8-Flash-Next does (MoE §1-§2); its own code is
the Forward's wiring, the round boundary and its error codes (`program/experts.cpp`). Placement
keys count the MoE layers only: key `b * experts + e` is expert e of model layer
`dense_layers + b`, and a placement file's layer rows are MoE layers.

Additions to the shared tier and Ops, each costing Qwen3.8-Flash-Next nothing:

- **Expert geometry as a compile-time parameter** (`src/ops/offloaded_sparse_moe/expert_layout.h`:
  `Geometry`, `Nvfp4Geometry<H, I>`), with instances 2560 × 640 (`qwen4_exp`, compiled code and
  bits unchanged) and 4096 × 2048 (this family) of the CPU and GPU routes.
- **`noaux_tc` routing** (`moe_route_dispatch_plan_noaux_tc`): FP32 sigmoid, bias in selection only,
  the unbiased scores normalized and × 2.5, the lower id on exact ties; a separate variant beside
  the softmax route, whose kernels are untouched.
- **The ungated shared expert and the SwiGLU clamp** (limit 10) as semantic inputs on every MLP
  class (`silu_mul_clamped`).

New Ops: `mhc` (§9), `dsa_indexer` (§7.3, sharing the exact radix select of
`src/ops/common/exact_topk_select.*` with `qsa`), `mla_sparse_attention` (§7.2) and the KDA calls
`kimi_delta_attention_batch_update`, `kimi_delta_attention_batch_verify` and
`kimi_delta_attention_fold` (§8). Core gained nothing.

## 5. Decode round

### 5.1 Round structure

```text
embed(host row) → 4 streams
per layer l (45):
  mhc_mix → RMSNorm → KDA (q/k/v projections, conv, gates, recurrence, gated norm, o_proj)
                      or MLA (q/kv latents, indexer, top-512 select, latent attention, o_proj)
  mhc_update_mix → RMSNorm → MLP:
     layers 0-2: dense SwiGLU
     layers 3-44: router (noaux_tc) → residency → GPU jobs | staged misses | CPU misses
                  shared expert ‖ routed experts (narrow W4A16) ‖ CPU misses → combine
  mhc_update
head: mean of streams → norm → lm_head (FP32 logits) → sample → DFlash2 proposal (§10)
```

One round is captured as a CUDA graph per (batch, width). The Program is synchronous: one round in
flight. Host time between rounds is small against a ~25 ms round, so chained rounds are not built
(Flash-Next measured its own and kept the synchronous contract, FN §8.8).

### 5.2 Router and miss classification

The router computes FP32 logits (BF16 logits create ties, FN §17.7), selects the top-8 of
`sigmoid(logit) + bias` with the lower id winning exact ties, and weights the unbiased scores as
§2.2 states. Classification and the job list are the shared dispatch plan (MoE §6.1-§6.2): a
`READY` frame is a GPU job; a RAM-resident miss is a CPU job or a staged copy; an SSD-only miss
goes to the fetch channel. The arbitration is MoE §6.1; this family's split is §6.3.

### 5.3 Expert arithmetic: W4A16

Every routed expert, on every route and processor, computes the same bits (MoE §2):

1. **Dequantization.** FP4 nibbles decode through the E2M1 table; each 16-weight group multiplies
   its FP8 E4M3 scale and the matrix's FP32 global scale, giving BF16 weights. The operation is
   fixed-order and identical everywhere.
2. **Dot products.** On the exact route, BF16 activations are encoded into 24-bit fixed-point
   limbs per column and accumulated in exact int64 row sums: narrow calls (`T <= 128`) run the
   dp4a row kernels (`gate_up_rows_kernel_a16`, `down_rows_kernel_a16`, streaming weight chunks
   reused across up to 4 columns with direct `__ldg` activations and pre-wait `w0` down-weight
   preloading at the 4096 × 2048 geometry), wider exact calls run the limb-tile tensor-core
   kernels (`grouped_kernel_a16`), and the CPU lane runs the bit-exact W4A16 AVX-512 path
   (`src/ops/offloaded_sparse_moe/cpu/w4a16_expert.cpp`). On the default INT8-KV route, wide NVFP4
   calls run the qualified `expert-a4-residual` 3-term residual FP4 Tensor Core tiles (§14; GGUF
   K-quant banks run `ggml_dp4a` / `ggml_tile` or `ggml-bf16-prefill` BF16 tiles, §3.3).
3. **Clamp and SwiGLU.** gate and up outputs are clamped as exact operations on BF16 values,
   SwiGLU runs in FP32 and h rounds to BF16. The combine is `y = bf16(Σ_i w_i · y_i + y_shared)` in
   FP32, in ascending expert id, so placement cannot change it.

## 6. Expert residency, placement and miss service

The tiers, frame pools, LFRU, landing, leases, the RAM tier, the SSD tier, the placement forms,
the runtime switch and the memory laws are the shared tier's (MoE §4). The subject is the 12,096
routed keys of the default artifact (42 × 288).

### 6.1 Placement

The serving split is `vram=40%,ram=60%` (§1); the Q8-dense artifact would fit `vram=44%`. Each is
the largest VRAM share that keeps the process under the 95 % cap with the prefix cache on at
4 × 262,144 tokens, and puts no expert on the SSD, which measured faster than any SSD share at
production's settings:

| Split | VRAM / RAM / SSD experts | One request, K = 3, tok/s | Four requests, aggregate tok/s |
|---|---|---|---|
| `vram=40%,ram=60%` | 4,838 / 7,258 / 0 | 41.1 (37.1-42.7) | 59.9 (56.5-62.0) |
| `vram=40%,ram=50%,ssd=10%` | 4,838 / 6,048 / 1,210 | 37.9 (34.0-38.5) | 54.7 (52.2-57.7) |
| `vram=40%,ram=40%,ssd=20%` | 4,839 / 4,838 / 2,419 | 34.2 (32.6-36.3) | 54.5 (50.9-57.2) |

(`infernix_glm5_next_engine_bench`, INT8 KV, the prefix cache on, DFlash2 K = 3: median and range
of four sets after a warm-up; model 16.70 GiB, KV 1.59 GiB at one lane and 6.36 GiB at four, expert
cache 63.8 GiB.)

An earlier measurement found a 20 % SSD share free at plain decode: one boot each,
`--no-prefix-reuse`, four lanes, 65,536 tokens of context, 256 greedy tokens without a drafter,
VRAM fixed at 42 % (5,080 experts; model 15.44 GiB, KV 0.40 GiB, experts 67.0 GiB):

| RAM / SSD | RAM slots (residents + 1,536 stream slots) | One request, tok/s | Four requests, tok/s each |
|---|---|---|---|
| 58 % / 0 % | 8,552 | 41.6, 41.9 | 19.4 |
| 38 % / 20 % | 6,132 | 42.4, 44.1 | 20.1 |
| 28 % / 30 % | 4,922 | 34.9, 38.2 | 16.9 |

The two measurements disagree at 20 % on the SSD. Their conditions differ: the first runs DFlash2
(every verify column routes its own experts, so a one-request round misses more) and the prefix
cache's larger KV, with four sets a point; the second ran plain decode with two runs a point. The
cause of the difference is not isolated. The serving split follows the first, which measured
production's settings.

### 6.2 VRAM policy and promotions

The policy is the shared LFRU (MoE §4.3). Promotions run at one per layer every 32 decode tokens
(`kDecodePromotionInterval`). Every expert lives in one tier, so a 13.5 MiB promotion also demotes
one; a promotion every token copied ≈ 15 records each way per token across the link the misses
need, and did not raise the hit rate (§6.3). No promotions follow a walked prefill (MoE §7.3).

### 6.3 Miss service

CPU-served misses default to 14 workers (`cpu_expert_workers`), up to 16 jobs per layer call
(`cpu_expert_jobs`), and per-format PCIe link shares: `cpu_link_share256` = 120/256 (~46.9 % on
the PCIe link, ~53.1 % on the CPU) for NVFP4 banks, and `cpu_ggml_link_share256` = 104/256
(~40.6 % on the PCIe link, ~59.4 % on the CPU) for GGUF K-quant banks (tuned 2026-11-27 alongside
AVX-512 VNNI block-half unrolling in `cpu/ggml_expert.cpp`, a 32-CTA × 32 KiB staging window in
`moe_layer.cu`, and cold-placeholder pre-demotion in `ExpertResidency` that asynchronously demotes
unranked `count == 0` startup seeds into free/shadow RAM slots on `d2h_stream_` so decode misses
land directly via `stage_kernel` without extra H2D traffic or SSD spill; MoE §5.6 states the
shared share forms). The CPU computes any column count without
changing a bit, since every W4A16 and exact ggml route is bit-exact. While the host computes a
call's misses the device runs the shared expert and warms the next layer's attention mix and first
input projections into L2 (40 MiB at one column, 16 MiB wider: FN §8.6's budgets); the bits are
unchanged and the one-row round median fell 21.6 → 20.3 ms (`vram=44%`, Q8-dense artifact). Earlier
NVFP4 sweep (`vram=42%,ram=58%`, INT8 KV, 256-token answers, two runs each):

| Configuration | One request, tok/s | Four requests, aggregate tok/s | Decode hits |
|---|---:|---:|---:|
| A promotion every token, no CPU lane | 30.7-31.7 | ≈ 45 | 86 % |
| No promotions, no CPU lane | 42.2-42.3 | | 88.9 % |
| A promotion every 8 / 32 tokens, no CPU lane | 37.9-39.2 / 41.2-41.3 | | 87 / 88.5 % |
| No promotions, CPU lane 8 workers, link share 1/2 | 47.1-47.5 | 75.4 | 88.9 / 80.1 % |
| Same, 12 workers | 47.8-47.9 | 78.7 | 88.9 / 80.9 % |
| Same, 8 workers, link share 1/3 | 46.5-48.0 | 75.0 | 88.9 / 80.8 % |

Twelve workers measured 1-4 % faster, inside the spread their higher hit rate explains; eight leave
the host cores to other work.

The one-row verify round spends more on staging than on the CPU wait (27.9 against 13.9 ms at
K = 3, §10.3), so calls of two or more columns were given their own link share (the shared
`wide_share` / `wide_share_from` mechanism). It did not pay: serving split, DFlash2 `auto`,
temperature 1.0, one request, 1,024-token essays, four sets, 2026-11-26:

| Calls of ≥ 2 columns | Median tok/s (range) |
|---|---:|
| link 1/2, 4 CPU jobs, 8 workers (served) | 49.2 (44.6-51.0) |
| link 1/3 | 48.4 (45.9-48.7) |
| link 1/4 | 47.2 (41.8-47.3) |
| link 2/3 | 46.3 (42.6-47.0) |
| link 1/3, 8 CPU jobs | 46.9 (43.6-48.2) |
| link 1/2, 12 workers | 47.5 (46.6-52.3) |

Moving misses either way made the round slower, so the single half share stays.

### 6.4 Staging slots

Staging slots are frames with a staging role: top_k × the widest decode or verify call's columns
(at most 8), and at least 32 (0.42 GiB), since a slot costs as much VRAM as a cached expert. A call
keeps the expert Op's fork route (staging beside the resident experts, staged misses landing in the
cache) only while its jobs fit the slots; past them it stages in serial passes and lands nothing.
With 32 slots a five-column call cost 35 ms of staging more per round than with slots sized to it
(§10.3). Serving's widest call (four lanes, or one row of K = 3) is 4 columns, 32 slots; one row of
K = 7 takes 64 (0.85 GiB). The prefill ring and walk span and the vision encode borrow frames
through leases (MoE §4.5).

### 6.5 Plain decode profile

At one request (`vram=42%,ram=58%`) a plain round spends 8.1 ms of 21.3 ms per token in the BF16
dense GEMVs (at the read rate), 4.3 ms in hit experts (≈ 54 % of the read rate), 4.2 ms staging
misses beside them and 2.7 ms waiting for the CPU lane; the GPU idles 0.5 ms.

### 6.6 RAM tier, warm start and multi-pool landing

The RAM tier, its ranking and the host RAM ledger are MoE §4.6 and §4.10. At the serving split
the RAM tier holds every expert outside VRAM (7,258 NVFP4 records, ≈ 96 GiB; 4,998 GGUF records at
`vram=59%,ram=41%`, 63.4 GiB, 0 on SSD). The shared expert state file (`--expert-state`) warm-starts
the tiers from the saved ranking; without one, auto placement seeds `kSeedShare` (90 %) of frames in
layer-spread order (`pp4096` rose from 1,856 to 4,206 tok/s and `pp256` from 117 to 305 tok/s on
2026-11-27), and explicit placements walk the tier fill in windows so every VRAM frame fills even
when VRAM frames exceed the RAM tier's slot count (eliminating unintended SSD spill on GGUF
`vram=59%,ram=41%`). Decode landing frames are reserved independently per frame pool
(`ExpertResidency::before_round`), so multi-pool GGUF banks land misses into frames freed by a
prefill lease in the Q3_K pool even when the other pools are full (raising GGUF `vram=40%` T1
`4x1` from 45.3 to 85.0 tok/s and `1x1` from 27.9 to 34.2 tok/s on 2026-11-27).

## 7. Sparse MLA attention and the pooled indexer

No sm_120 kernels existed for either (FlashMLA's sparse kernels and DeepGEMM's indexer kernels are
SM90/SM100), so both are engine Ops.

### 7.1 The absorbed form

With no RoPE part, the latent is the whole key, and the reference attention is exactly an MQA over
the latents:

```text
q'_h     = W_uk,hᵀ q_h                       512 per head (W_uk,h: 256 × 512)
score_hj = q'_h · c_j / 16                   c_j: the cached 512-value latent of token j
u_h      = Σ_j softmax_j(score_hj) c_j       512 per head
o_h      = W_uv,h u_h                        256 per head
```

The reference materializes k_h and v_h in BF16; those are private intermediates, so the absorbed
form is a legal private arithmetic ([op development §6](op-development.md#6-qualification)),
qualified against an FP64 oracle of the reference formula.

### 7.2 Attention (`mla_sparse_attention`)

- **Decode and short prompts (< 64 columns):** per layer and sequence, Q' is 64 × 512 and the
  selected latents ≤ 2,051 × 512. Eight-warp (256-thread) `mma.sync` tiles (`kSplitMmaTokens = 16`)
  over 16-head groups run S = Q'Cᵀ (BF16 MMA, FP32 acc) and U = P C with online softmax between
  them, split over 32-token blocks and merged in fixed block order. A CTA gathers whole 4-token
  pools through the block table plus the tail tokens.
- **Prompt calls of 64 or more columns** take the tensor-core envelope: one CTA per column with
  all 64 heads sharing its tokens, 16-token tiles of latent rows staged as BF16 (INT8 codes
  exactly), scores on BF16 MMAs with FP32 accumulation. One route serves both regimes (dense below
  2,051 visible tokens and sparse above), in the absorbed form throughout. Against the FP64 oracle
  both routes pass the `2.8e-3` relative-L2 / `5.0e-3` gross-relative criterion and strict bitwise
  column invariance. The prefill is bit-identical across chunks of 1,024, 4,096 and 8,192.
- **INT8-G64 latents** are staged as stored (codes and FP16 group scales) and decoded in the score
  and value folds: a lane's 16 dimensions lie in one group of 64, so the group scale multiplies the
  lane's partial dot once. For `kInt8 = true`, both `attention_split_kernel` and
  `attention_mma_kernel` run the Value phase with a single `mma_f16` (per-group `s_max`
  normalization, `w = p · scale / s_max`, rescaled into FP32 `acc`), halving Value MMAs and
  cutting `pp16384` attention time from ~720 ms to 456.6 ms (2026-11-27); BF16 latents keep the
  BF16 high/low probability pair.

### 7.3 The indexer (`dsa_indexer`)

Every step follows the reference (§2.2) at its precision:

| Step | Precision | Stored |
|---|---|---|
| k_t = LayerNorm(W_k x_t), z_t = W_g x_t | FP32 LayerNorm, BF16 outputs | A per-lane ring of the 64 most recent positions' k and z (32 KiB per lane and layer): a pool reads its earlier tokens there and the call's own from its inputs; rollbacks of up to 60 positions stay exact |
| Pool key = Σ_i p_i ⊙ k when a pool completes (p_i cast to BF16) | BF16, as the reference computes it | BF16 pooled-key plane: 128 BF16 per pool = 64 B per token and layer |
| q_h = W_qb_idx c_q (BF16), w_h = W_w x · 32^−½ | BF16 linear outputs, FP32 weights | — |
| score_p = Σ_h w_h relu(q_h · key_p / √128) | FP32 accumulation over BF16 keys | — |
| Top-min(512, P) pools + tail | Exact radix select, lower pool id on ties | Pool ids for the attention |

Decode scans every complete pool (8 MB per layer and token at 128K context). Prompt scores run a
column-invariant `mma_bf16` Tensor Core GEMM (FP32 accumulation) over queries × pools per head
followed by the weighted relu sum, in query sub-chunks so the score tile stays bounded at long
contexts (16.6× faster than the earlier register-tiled FP32 GEMM: 40.6 ms on `pp16384`,
2026-11-27). Pooled keys and selection never depend on `--kv-dtype`, so changing the latent format
never changes which tokens a query attends to.

### 7.4 KV formats

The latent is both key and value, so per-K/V split profiles do not apply. `--kv-dtype bf16` and
`int8` are built; the family refuses every other profile.

| `--kv-dtype` | Latent per token and layer | With the 64 B pooled key | 11 MLA layers: 128K / 262K tokens |
|---|---:|---:|---:|
| `bf16` | 1,024 B | 1,088 B | 1.6 / 3.2 GB |
| `int8` (INT8-G64, one FP16 scale per 64, the group codec of `kv_cache_append.h`) | 528 B | 592 B | 0.85 / 1.7 GB |

INT8-G64 cost (FP64 reference forward, `tools/glm5_next/reference.py --latent-codec`, 495 tokens):
mean KL 8.0e-3 from the FP64 oracle against 1.28e-2 for BF16 rounding alone, and BF16 with INT8
latents 1.30e-2; groups of 128, one scale per row and a Hadamard rotation measured within 0.001 of
it. Serving uses `int8`.

### 7.5 Long context

The device cache holds every served context: at 262,144 tokens the INT8 cache is 1.59 GiB per
lane. Up to 1,048,576 positions are accepted, also from the device cache. The KV host tier is not
built (§17.2).

## 8. Kimi Delta Attention

- **Mathematics:** the `kimi_delta_attention` Op (`src/ops/linear_attention/kimi_delta_attention/`,
  `include/infernix/ops/kimi_delta_attention.h`) is exactly this model's recurrence (§2.2), with
  the −5 lower bound and FP32 state, on recurrent and chunked routes qualified against an FP64
  oracle. The convolution is `causal_conv1d_silu_split` over the concatenated q|k|v channels.
- **Decode:** the recurrent route; a batch of lanes runs `kimi_delta_attention_batch_update`.
- **Prefill:** the chunked route.
- **Speculative verification:** a verify call runs every row's columns from its lane's state
  without writing it (`kimi_delta_attention_batch_verify`, the one-token arithmetic, so column j
  equals j + 1 decode steps bit for bit; `causal_conv1d_silu_from_states_split` for the
  convolution). The Program records, per KDA block and verify column, the convolution input and the
  recurrence's k, v, g and β (≈ 98 KB), and the commit folds each row's accepted prefix into its
  lane (`kimi_delta_attention_fold`, `causal_conv1d_commit_window`).
- **State per lane:** 136 MiB of recurrence plus 5 MB of convolution state; a BF16 state is
  rejected (§17.1).

## 9. Manifold-constrained hyper-connections

The `mhc` Op (`include/infernix/ops/mhc.h`) has three calls: `mhc_mix` (the mixer: norm, the 24
projections, gates, Sinkhorn, collapse), `mhc_update` (the residual update) and `mhc_update_mix`
(the update followed by the next block's mixer in one call, with the separate calls' bits). The
mixer reduces fixed-order partials of f and of the squared RMS per feature chunk, so a column's
bits depend only on that column.

Numerics:

- Mixing weights, Sinkhorn, pre, post and comb in FP32 (`fn` upcast from BF16); Sinkhorn is
  reproduced iteration for iteration, never iterated to convergence.
- Residual streams stay BF16. Flash-Next measured an FP32 residual as worse in perplexity
  (+0.030 nats; FN §17.7): the model expects the rounding it was trained with.
- The update computes `post_s · y + Σ_j comb[j][s] · R_j` in FP32 with post and comb in FP32 and
  rounds each output once. The reference forms it in BF16 (`bf16(post) · y + bf16(comb)ᵀ @ R`,
  §2.2); the engine's form is more precise, and the forward test bounds the whole model against the
  binary64 reference (§14.2).

## 10. Speculative decoding

### 10.1 The DFlash2 drafter

[incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) (Apache-2.0,
accepted by the owner): `DFlash2DraftModel`, model type `qwen3`, 2.2 GiB BF16; 5 sliding-window
layers (window 2048), hidden 4096, intermediate 12288, 32 heads, 8 KV heads, head dimension 128,
rope theta 10000; `dflash_config`: block_size 8, conv_kernel_size 2, conv_group_size 16,
selector_rank 256, selector_top_k 16, mask_token_id 154856, target_layer_ids [5, 14, 24, 33, 42];
`fc.weight` [4096, 20480] (five 4096-wide target features); a candidate selector with predecessor
and successor codebooks [154880, 256].

It has the structure of Qwen3.8-Flash-Next's drafter ([dflash.md](dflash.md)), so the engine's
DFlash2 path serves it at hidden 4096 as a companion artifact. A target layer's feature is the mean
of the four mHC streams of its completed residual (`ops::hyper_connection_stream_mean`), the
Flash-Next convention; the measured acceptance (§10.3) confirms it. The drafter's weights are all
BF16 and resident; it never touches the expert tiers.

### 10.2 Draft policy

Only one-row rounds draft; a round of several rows decodes plainly, since two rows of K = 3
already reach 8 columns, the fork route's limit (§6.4). No trees. A one-row round proposes its
block, then chooses how many of the drafts to verify:

- `--draft-tokens N` (1-7) verifies all N, or with `--spec-min-p P` those before the first whose
  selector probability (the candidate selector's temperature-1 softmax at the drafted rank) is
  below P.
- `--draft-tokens auto` proposes 5 (`kDraftTokensAuto`) and verifies the k in 0..5 that maximizes
  (1 + sum over j ≤ k of the product of the first j drafts' calibrated acceptances) / T(k), where a
  draft's acceptance is read from its probability's decile, learned per request (EWMA 0.1, one
  draft more every 8 rounds keeps the longer deciles measured), and T(k) is the measured one-row
  round time in plain rounds (`program/draft_length.h`: 1.07 for a plain round after a spent
  proposal, then 1.50 / 1.79 / 1.96 / 2.29 / 2.68 for k = 1-5, §10.3). k = 0 decodes the round
  plainly. Not 7: a verify column's routes miss on their own, so the length rarely pays past five,
  and each column more costs eight staging slots.

Greedy output equals plain decoding either way; a sampled row's verification is the same sparse
rejection sampling against the selector's laws on a prefix of its drafts.

Serving (`configs/GLM-5.3-Flash/model.env`) drafts `auto` (§10.3; `INFERNIX_SPEC=` serves plain
decode).

### 10.3 Measurements

Draft length at one request on the Q8-dense artifact (not served, §17.2), `vram=44%,ram=56%`
(5,322 / 6,774 / 0 experts; model 10.43 GiB with the drafter, KV 1.59 GiB, experts 70.2 GiB),
`infernix_glm5_next_engine_bench --concurrency 1 --tokens 512 --repeat 4`, 2026-11-24:

| K | Median tok/s | Range | Accepted, last two sets |
|---|---|---|---|
| 0 (plain) | 40.7 | 38.1-41.9 | – |
| 2 | **52.1** | 43.9-52.8 | 62 %, 64 % |
| 3 | 47.1 | 44.7-56.0 | 51 %, 60 % |
| 4 | 46.9 | 42.6-48.6 | 50 %, 43 % |

K = 2 has the best median; K = 3's acceptance varies more between sets (its best set, 56.0, is
the fastest single set). On these essays acceptance sits below DFlash2's published 80-85 % a
position for Qwen3.8-27B (no GLM numbers are published), while a greedy chat continuation accepts
5.27 of 7 (below); the drafter was trained against the official checkpoint, not the uncensored
edit served here. The BF16-dense measurements below set serving's K = 3.

Rounds of several rows, `vram=45%,ram=55%` (5,443 / 6,621 / 32 experts), INT8 KV, 600-token essay
continuations, two runs a level, aggregate output tok/s:

| Decode | 1 request | 2 requests | 4 requests | Process VRAM |
|---|---|---|---|---|
| plain | 29.7, 33.5 | 43.2, 39.0 | 46.8, 45.2 | 90.1 GiB |
| DFlash2 K = 3, every round | 31.5, 37.2 | 34.4, 35.2 | 39.3, 40.7 | 91.7 GiB |
| DFlash2 K = 7, every round | 25.0, 26.8 | 24.6, 24.1 | 27.8, 29.7 | 91.8 GiB |
| DFlash2 K = 2, one-row rounds only | 37.2, 33.6 | 42.9, 40.9 | 47.3, 52.4 | 91.6 GiB |
| DFlash2 K = 3, one-row rounds only | 34.9, 33.6 | 47.7, 39.2 | 54.4, 53.0 | 91.6 GiB |

Acceptance on these prompts is 49-58 % at K = 3; the expert cache's hit rate falls from 84-86 % to
76-77 %, since every verify column routes its own experts. A greedy chat continuation accepts 5.27
of 7 drafts a round; a sampled one (temperature 0.8) 60 of 236 at K = 7. Drafting every round loses
at two and four requests, and drafting one-row rounds only is no slower than plain there, which
decided the policy. The drafter adds 1.6 GiB of VRAM (weights, context ring, verify records).

Draft length at one request, `vram=42%,ram=58%` (5,080 / 7,016 / 0 experts; model 15.44 GiB,
experts 67.0 GiB), 512-token essays, four sets after a warm-up
(`infernix_glm5_next_engine_bench --draft-tokens K`; speeds climb over an Engine's first sets as
the cache warms, so the last two sets compare):

| K | Median tok/s | Last two sets | Accepted | ms a round |
|---|---|---|---|---|
| 0 (plain) | 35.0 | 35.0, 35.3 | – | 28 |
| 1 | 40.9 | 40.9, 42.1 | 74-82 % | 42 |
| 2 | 44.4 | 44.4, 44.5 | 56-65 % | 50 |
| 3 | **47.8** | 47.8, 48.1 | 50-57 % | 55 |
| 4 | 43.2 | 43.0, 46.5 | 41-52 % | 64 |
| 5 | 37.7 | 35.3, 41.6 | 35-43 % | 75 |

K = 3 measured fastest, which set serving's length. Under Nsight
(`make t2-bench FAMILY=glm5_next PROFILE=1`) a one-row round is GPU-bound (≥ 98 % busy) and the
drafter's own kernels are ≤ 2 ms of it; 80-84 % of the round is the target's expert service. At
K = 3, of 58.6 ms: miss staging 27.9, the CPU team's wait 13.9, GPU experts 11.7, dense BF16
weights 8.4 (the plain round: 27.7 ms, of which staging 5.6, CPU 9.2, dense 8.1). The second to
fifth verify columns cost 14, 8, 5 and 9 ms each, as every column routes its own experts.

The calibrated length against K = 3, serving split (`vram=40%,ram=60%`, 4,838 / 7,258 / 0 experts;
model 15.44 GiB, 16.70 with the drafter), one request, 512-token essays, four sets after a warm-up,
`infernix_glm5_next_engine_bench --concurrency 1 --tokens 512 --repeat 4`, 2026-11-26:

| Sampling | K = 3 | `auto` | `auto`: drafts verified a drafting round, accepted, plain rounds |
|---|---|---|---|
| greedy | 46.0 (39.1-48.5) | 45.9 (44.2-49.5) | 3.0, 58-63 %, 9-17 % |
| temperature 1.0, top-p 0.95 | 43.0 (41.9-43.2) | 48.2 (41.2-50.1) | 3.4, 63-69 %, 17-20 % |

On greedy essays the calibrated length commits 2.88 tokens a round against K = 3's 2.65, but its
rounds took ≈ 63 ms against ≈ 57.5, more than T(k) predicts for its mix of lengths; the two are
equal within the sets' spread. Sampled, where the selector's probabilities separate good drafts from
bad ones, it is 12 % faster at the median.

### 10.4 Fine-tuning the drafter on the served target

The incoai drafter was trained against the official checkpoint. `tools/drafter/` (README.md) fine-tunes
it on this engine's features of the served uncensored target: the target's own answers through the
public Engine (`infernix_glm5_next_drafter_generate`), each record's teacher-forced tap features and
final stream mean through the causal-scoring Program (`infernix_glm5_next_drafter_capture`,
`Program::enable_feature_capture`), then speculators' `DFlash2DraftModel` objective warm-started from
the incoai body. The capture is checked by the target head's agreement with the next token on the
target's own sampled answers (94-98 %), and an untrained round trip reproduces the incoai tensors
bit for bit.

The first run, 2026-11-26: 210 answers (298k tokens, temperature 1.0; generation ran at 39 tok/s
with eight requests, the corpus's limit) and 400 UltraChat conversations (458k tokens), weights 3 : 1,
lr 3e-5, 8,192-token rows of 512 anchors, four rows a step. The held-out sum of the selector's
teacher-forced accuracy at draft positions 1-5 rose 2.415 → 2.468 by step 400 and then fell (2.343
at step 3,993): the answers overfit. The step-400 drafter
(`GLM-5.3-Flash-DFlash2-FT-Infernix.infernix`, not served) against incoai, `auto`, one request,
512-token essays, four sets:

| Sampling | incoai | fine-tuned |
|---|---|---|
| temperature 1.0 | 47.7 (41.1-48.7) tok/s, 2.88 tokens a round | 48.2 (41.5-50.8), 2.87 |
| greedy | 45.9 (44.2-49.5), 2.88 (§10.3's run) | 49.3 (44.5-50.4), 2.85 |

Tokens a round did not change, so serving keeps the incoai drafter. A gain needs far more of the
target's own text than the generation rate gave overnight (the Apache-2.0 GLM-5.3-Flash DFlash2
trainer github.com/Enntity/glm-flash-drafter fine-tuned on ≈ 119M stored tokens of its own traffic
and reports 115-122 % of the incoai drafter's speed there, 87-95 % on public prompts).

## 11. Prefill

The walk-span structure, the streamed expert ring, chunk sizing against the cache and the
no-promotions rule are the shared prefill design (MoE §7).

### 11.1 Walk spans

Consecutive streamed calls (at least 256 columns each) of one prompt form a span of up to 32,768
tokens (`program/prefill_walk.cpp`); chunk c's layer l runs what the chunk-major call c runs there,
so the bits equal the chunk-major prefill's (the Program test compares a 1,100-token prompt's
greedy tokens both ways). The span's residual stream, mHC mix, post/comb weights, ids, positions
and per-chunk DFlash2 features live in a frame lease beside the ring; the Forward exposes
`begin_call`/`embed`/`step`/`finish`, which `run()` composes. Each `advance_prefill` enqueues about
one call's work of layers over every chunk (a beside-decode call's width while another lane
decodes) and synchronizes; other prefilling lanes wait, and other lanes' rounds land and promote
nothing while a span is active. Prefix taps due at an inner chunk end on the block grid are
captured in place (§12); only a due tap off the block grid ends the span.
`ProgramOptions::prefill_walk` turns walks off.

Measured (`vram=42%,ram=58%`, 5,080 / 7,016 / 0 experts; model 15.44 GiB, KV 0.40 GiB, experts
67.0 GiB; INT8 KV, `--no-prefix-reuse`, chunk 4,096):

| Prompt tokens | Chunk-major | Walk |
|---|---|---|
| 10,092 | 6.3 s TTFT (1.61k tok/s) | 4.3 s (2.35k tok/s) |
| 12,644 | 8.0 s (1.59k tok/s) | 5.2 s (2.43k tok/s) |
| 37,805 | 18.5 s (2.04k tok/s) | 15.7 s (2.41k tok/s) |

The flat rate says prefill is compute-bound. Under a small RAM tier the walk is no faster than
chunk-major: the stream's read slots are an eighth of the tier's, so a layer's SSD-only experts
beyond that share go to the fetch channel once per chunk (`--expert-ram-mib 24576`: about 74 s for
a 7,741-token prompt). The serving split leaves no SSD-only expert.

### 11.2 The streamed ring

A prefill chunk of at least 256 columns streams its non-resident experts by copy-engine DMA two
layers ahead of compute, the Flash-Next form (FN §13.3, MoE §7.1). `ExpertStream`
(`src/models/glm5_next/execution/expert_stream.{h,cpp}`, specialized from Qwen3.8-Flash-Next's
with per-layer record strides for GGUF banks and one-pass tier-mode streaming) copies each MoE
layer's non-resident experts with a pinned host record, in ascending id, into half l % 2 of a ring
of frames the expert cache lends for the prompt; runs of adjacent records of one bank coalesce into
one copy, and a layer's copies enqueue as one `CopyBatch`. The MoE reads a streamed record like a
resident one (`MoeExpertSource::prefetched`). An expert beyond the ring's half, or without a host
copy, stages or fetches as before; within a walk span the tier's stream slots read SSD-only
experts too. `--prefill-expert-stream off` stages with the MoE's SM kernel as decode does.

Measured (`--kv-dtype int8 --max-concurrency 4 --max-context 65536 --experts vram=45%,ram=55%
--no-prefix-reuse`, chunk 4,096; 5,443 / 6,653 / 0 experts; model 15.44 GiB + KV 0.40 GiB + expert
cache 71.8 GiB), one boot per build, streaming on and off on the same binary:

| Prompt tokens | Streamed off | Streamed on |
|---|---|---|
| 10,092 | 9.8 s TTFT, 1.03k tok/s | 7.1 s, 1.42k tok/s |
| 20,209 | 18.2 s, 1.11k tok/s | 12.4 s, 1.62k tok/s |
| 40,380 | 36.5 s, 1.11k tok/s | 24.9 s, 1.62k tok/s |

Greedy tokens, DFlash2 agreement lines and causal-scoring log-probabilities are identical between
the two, and decode after a long prompt is unchanged (37.5 tok/s after a 10K prompt either way; the
ring returns before decode).

### 11.3 Compute

| Part | Route |
|---|---|
| Dense GEMMs and head | BF16 tensor cores (the existing dense kernels), bit-identical to the decode route; shapes at H = 4,096 registered in `src/models/shapes.cpp` |
| Routed experts | `expert-a4-residual` 3-term residual FP4 Tensor Core tiles on INT8 KV (all active non-CPU jobs at 64+ columns with in-place `h_rows` reuse, > 8-entry jobs at 9-63 columns; exact W4A16 limb tiles under `--disable-fast-paths` or BF16 KV) for NVFP4; exact A16 tiles or `ggml-bf16-prefill` BF16 tiles (GGUF, §3.3) |
| MLA, indexer | §7.2-§7.3 |
| KDA | chunked route |

The dense classes are BF16 already (the GGUF recipe binds its Q8_0 dense classes `A16Only`); wide
routed-expert activation precision is controlled by `expert-a4-residual` (NVFP4) and
`ggml-bf16-prefill` (GGUF), both on by default with INT8 KV and disabled by `--disable-fast-paths`
(§14).

### 11.4 Prefill beside decode

While another lane decodes, a prompt's calls are split to `--prefill-chunk-beside-decode` (default
1,024) and run between the decode rounds. Mixed rounds (a prefill call inside the decode round's
Forward) are not built (§17.2).

## 12. Prefix cache

The hybrid prefix cache binds as for Qwen3.8-Flash-Next; the spec's
[§18](hybrid-prefix-cache-spec.md#18-glm-53-flash-binding) states the differences that a change to
the shared cache must respect, and this section gives the family's facts. The family owns its
prefix index (a fork of Qwen4Exp's, `program/prefix/prefix_index.*`) and its physical binding
(`prefix_cache.*`, `state_image.*`):

- **KV blocks:** 64-token content-addressed blocks of MLA latents (BF16, or INT8-G64 with their
  scales) and their BF16 pooled keys. Pools start at the sequence start, so a block holds exactly 16
  complete pools and is self-contained.
- **State snapshots:** the KDA recurrence and convolutions, the indexer's raw key/gate rings and
  the DFlash2 drafter's context ring, about 141 MiB each.
- **Taps on the chunk grid only.** The projection routes are chosen by call width, so a resume is
  exact only when its calls repeat the publishing run's call boundaries. Every tap lands on a
  multiple of the prefill chunk (which must be a multiple of 64): a tap splits nothing, and a resume
  from any tap replays the publishing run's call widths over the same restored state. The shared
  spec's finish endpoints are not published. The prompt-end capture holds the prefill transition's
  state with the target head's saved residual, so an exact repeat regenerates its first token with
  no prefill (exact under DFlash2: the drafter's ring is in the image, and its proposal reads only
  the ring).
- **Host tier only,** default 2 GiB (`--host-context-mib`; each GiB costs ≈ 78 RAM expert slots).
  The Disk tier, persistence, Device snapshot slots, packed Host state and tool-call holds are
  refused.
- Mixed rounds are refused for a lane that publishes, for the call-grid reason.

Measured (`vram=42%,ram=58%`, chunk 1,024, `tests/models/glm5_next/test_prefix_real.cpp`): a
7,741-token prompt is one 8-call span with its seven inner taps captured in place, 5.4-6.4 s; a
prompt sharing its first 7,168 tokens resumes from the last tap in 2.2 s (573 computed); an exact
repeat computes nothing. The tokens equal the cold runs', with and without DFlash2. A hit avoids
both the prompt compute and a whole expert pass, so the cache is worth more here than on
Flash-Next.

## 13. Memory

### 13.1 VRAM

The sizing law, the spill guard and the monitor are the shared tier's (MoE §4.9). The token
embedding stays host-resident (pinned, one row read per token). `vram=100%` is refused: the bank
(159.4 GiB in frames) exceeds the card.

The serving split's ledger (§1): model 16.70 GiB (the dense BF16 weights, the drafter, workspace),
KV 6.36 GiB at 4 × 262,144 INT8 tokens with the prefix cache on (1.59 GiB at one lane), and the
expert cache 63.8 GiB (4,838 frames of 13.5 MiB, staging included). The measured process peak while
serving is 92,670 MiB with `--draft-tokens auto` and `--vision` (92,412 MiB at fixed K = 3)
against the 92,993 MiB cap; 42 % VRAM sat at 95,672 MiB. KV is allocated for
every lane at the full context at startup, so the VRAM share must be re-measured whenever
`--max-context` or `--max-concurrency` changes. Each lane also holds its KDA state (141 MB).

### 13.2 Host RAM

The ledger's law and terms are the shared tier's (MoE §4.10). At the serving split the RAM tier
holds every expert outside VRAM (7,258 records, ≈ 96 GiB). Beside it sit the pinned token embedding
(1.18 GiB), the prefix Host tier (2 GiB), the streamed vision tower (1.13 GB pinned) and the
process; the rest of the 182 GiB stays with the OS, the page cache and other programs.

## 14. Numerics and qualification

The engine's three-layer contract — the canonical arithmetic, the exact route, fast paths under
the [acceptance policy](model-quality.md#6-numerical-acceptance-policy-and-fast-paths) — applies
in a reduced form here. The exact route (`--disable-fast-paths`; `--a16-activations` selects it too,
as the quality gate's reference) is canonical, and its prefill bits do not depend on the chunk size
(§7.2). The family has two qualified prefill fast paths, each on by default with INT8 KV (BF16 KV is
not gated and stays exact):

- **`expert-a4-residual` (NVFP4 artifact).** Wide expert calls encode BF16 activations into 3-term
  NVFP4 residuals and run native SM120 FP4 Tensor Core MMA tiles (`wide_expert.cuh` `ScheduleResidual`)
  at the 4096 × 2048 geometry with clamped SwiGLU (limit 10). Calls of 64 or more columns
  (`kA4ResidualAllJobsMinColumns`) route every active non-CPU job through it, skipping limb encoding
  and the narrow dp4a pass, and rewrite `h_rows` in place into `h_terms_plane` (`4096 B/row >=
  3456 B/row`, aliasing `x_token_terms` onto `h_rows`) for zero extra serving VRAM (`work peak
  1.16 GiB`); calls of 9-63 columns route jobs of more than 8 entries through it. Qualified
  2026-11-27 ([model quality §6.7](model-quality.md#67-fast-path-registry); mean KL excess
  -1.72e-4, top-1 excess -0.016 pts, PPL ratio 0.99967, p99.9 ratio 0.992), lifting `pp16384`
  3.47k → 5.56k tok/s (+60.2 %) and `pp32768` 3.40k → 5.41k tok/s (+59.1 %).
- **`ggml-bf16-prefill` (GGUF artifact, §3.3).** Calls of 64 or more columns run every active
  non-CPU expert job on BF16 tiles directly from BF16 x (skipping limb encoding and narrow dp4a),
  while calls of 9-63 columns route jobs of more than 8 entries to BF16 tiles. Qualified 2026-11-26
  ([model quality §6.7](model-quality.md#67-fast-path-registry)). One layer's expert call at 4,096
  tokens (288 experts, top-8, every record resident; `infernix_offloaded_moe_ggml_test --bench-glm
  4096 bf16`, 2026-11-27) takes 10.27-14.09 ms (117.1-160.5 effective TOPS) against the exact
  tiles' 27.99-36.81 ms.

Under either path a prompt's bits depend on its call widths (the chunk size, and the 1,024-token
calls beside another lane's decode), so neither artifact is request-invariant with fast paths on.
Decode and verification calls (≤ 8 columns, or narrow jobs up to 63 columns) keep the exact narrow
dp4a routes, so neither path changes decode arithmetic. Each artifact accepts only its own fast path
and refuses the other or unknown names.

### 14.1 Op oracles

Each Op follows [op development](op-development.md#6-qualification): an independent naive
FP32/FP64 oracle, exact oracles for exact transforms, packed inputs decoded independently with
their stored scales, real model shapes.

| Op | Oracle and criteria |
|---|---|
| `offloaded_sparse_moe` W4A16 at (4096, 2048) | Bit-exact across GPU frame, GPU staging and CPU, n = 1..8 and the wide route, against an FP64 oracle that decodes the packed words with their stored scales; the record relayout round-trips the source tensors bit-exactly. `noaux_tc` routing exact against an FP64 sigmoid/bias/top-8 oracle on near-tie-free inputs, ties by lower id. Clamp boundaries |
| `mhc` | FP64 oracle of the reference formula including all 20 Sinkhorn rounds |
| `kimi_delta_attention` and its batch, verify and fold calls | FP64 oracle; the fold bit-exact against the committing form |
| `dsa_indexer` | FP64 pooling and scores; exact selection on near-tie-free fixtures; pool agreement on real activations |
| `mla_sparse_attention` | FP64 softmax attention over the latents of the selected set, absorbed and explicit forms, BF16 and INT8 latents |

### 14.2 Model-level checks

Per-block checks run on the model's own activations against an FP64 reference built from the
decoded real weights (`tools/glm5_next/block_check.py`); every block measured stays below 4.4e-3.
The offline reference forward (`tools/glm5_next/reference.py`, a qualification tool, not a product
route) runs the whole model teacher-forced, layer-major, streaming each layer's weights from the
disk once per pass.

The model amplifies BF16-sized differences: on the 15-token forward-test prompt the reference
rounded at BF16 boundaries and the binary64 one differ by up to 4.6e-2 in a block's residual and
0.22 in a position's logits, pick different routed sets at 26 % of the MoE positions, and differ in
argmax at two near-ties (MoE blocks 10 and 25 multiply an input difference by about ten). The real
forward test (`tests/models/glm5_next/test_forward_real.cpp`) therefore takes the binary64 run as
its oracle and the rounded run as the floor: within twice its deviation, the argmax one of the two
runs', routed sets equal about as often.

### 14.3 Engine and system tests

- T1 (`make t1 FAMILY=glm5_next`, [round-bench.md](round-bench.md)): rounds of 1x1, 2x1, 4x1 and
  1x4 at 1,024 tokens of context on the real artifact at the serving split, with logits and
  lane-state hashes in `bench/models/glm5_next/round_hashes.txt` and five replays per shape.
- T2 (`make t2 FAMILY=glm5_next`, `tests/models/glm5_next/test_smoke_real.cpp`): the Engine with
  the serving options; golden tokens reproduce across boots, with four in flight, after a tap resume
  and after a placement switch.
- Real tests (`tests/models/glm5_next/`): load, forward, program, engine, prefix, frontend,
  tokenizer, the vision tower, and vision through the Engine.
- A RAM-tier switch releases the old tier first when both do not fit (`ensure_tier`), and the fill
  reads the records again.

## 15. Frontend, vision and serving

### 15.1 Text

- **Tokenizer.** Byte-level BPE from `tokenizer.json`, with the GLM-4-family digit grouping
  (`\p{N}{1,3}`); other split patterns stay refused. Exact-oracle tests against Hugging Face
  `tokenizers`.
- **Template.** The GLM-5.3 Jinja template through the engine's template runtime: tools as a
  `<tools>` block; `reasoning_effort` low/high/max (default max) mapped from OpenAI
  `reasoning_effort` and the Anthropic thinking budget; `clear_thinking`.
- **Reasoning.** The generation prompt ends with the thinking opener, so every output starts in the
  thinking channel; requests that ask for no thinking get `reasoning_effort: low`, never an injected
  closing tag.
- **Tool calls.** The `glm` parser for the model's call form (string values raw, others JSON, typed
  by the tool's schema), its recovery modes and its constrained-decoding grammar; results render in
  the template's tool-response form in call order.
- **Serving.** `infernix-serve` serves the family with the expert-placement admin endpoint
  (MoE §4.8) and the prefix cache.

### 15.2 Vision

Behind `--vision`.

- **Processor** (`frontend/processor.cpp`). Glm5NextImageProcessor and Glm5NextVideoProcessor: the
  smart resize to whole 28-pixel merge blocks within `min_image_tokens` and `max_image_tokens`
  merged tokens (an image: 16 to 8,000; a whole video: 16 to 240,000, capped by
  `--vision-max-merged-tokens` and the prompt's vision budget), the content resized into that
  canvas without upscaling once the source alone holds the minimum, then zero padding right and
  bottom; torchvision's antialiased uint8 bicubic (ATen's int16 weights and their precision); the
  fused CLIP normalization in FP32 rounded to BF16; 14-pixel patches of 1,176 features in
  merge-block order, an image's frame duplicated. Video frames are sampled by GLM's rule
  (`sample_frames`: 2 fps, at most 2,048 frames, an odd count's last frame repeated), and each
  frame pair's tokens follow `<|begin_of_image|>` and precede `<|end_of_image|>` and the pair's
  time (`"%.1f seconds"`). An image's `<|image|>` repeats once per merged token; video frames use
  the same token id inside `<|begin_of_video|>` … `<|end_of_video|>`; token types are 1 for an
  image's tokens and 2 for a video's. Positions stay the token indices: GLM-5.3 has no multimodal
  rope. Media decoding is the shared decoder's.
- **Tower** (`execution/vision.cpp`): patch embedding, 24 blocks (RMSNorm, separate Q/K/V GEMMs
  with biases, per-head q/k RMSNorm, 2-D rope over (row, column) at head dimension 64, packed
  attention within one item, clamped SwiGLU MLP in 4,096-column tiles), the post norm, the 2 × 2
  downsample (its weight permuted at load to the merged column order) and the gated merger into the
  text width. One pass runs every item of a prompt layer-major; each image, and each frame pair of a
  video, is its own item, so an item's embeddings are the same alone or beside others.
- **Offload** (`--vision-offload auto|on|off`, auto on). The tower binds `HostPinned`; the encode
  lends the weight staging with the handoff and the pass (`src/models/common/vision/`: the pinned
  layout in groups, and the weight stream, which uploads the patch embedding, then each layer one
  layer ahead of compute into two slots, then everything after the last layer). The pass reads a
  copy of its parameters rebased onto the staging (`vision_window_weights`); the blocks' joined
  gate/up biases (0.4 MB) stay in device memory.
- **Program** (`program/vision.cpp`). Binding plans the items whose tokens end past the reused
  prefix; the lane's first prefill call runs the whole pass into one token-ordered handoff
  (BF16 [hidden, tokens]) in lent expert frames, or an allocation when the residency lends none, and
  gives the pass's frames and the weight staging back; every call, chunk-major or in a walk span,
  scatters its columns of the handoff over the token embeddings before the stream expansion
  (`ForwardBatch::vision_embeddings`). The handoff returns after the prompt's last call. A resume
  inside an item encodes the item again and scatters only its remaining columns. Block keys carry
  each item's identity (content digest, grid, timing, placement), so the prefix cache shares a
  prefix only with the same media at the same place. Other lanes wait for an encode, as for one
  prefill call.
- **Qualification.** The processor equals the Hugging Face processor exactly on prompts, token
  types, grids and BF16 patches for PNG images (downscaled, upscaled, padded only, a 28 × 1,000
  strip, three images at once) and H.264 videos (8 to 30 fps, 1 to 7 s, an odd sampled count),
  through `tools/glm5_next/processor_reference.py` and the frontend real test. JPEG pixels differ by
  up to 3 levels in about 5 % of values, because FFmpeg's IDCT is not libjpeg's (the shared
  decoder's). The tower is qualified stage by stage against a binary64 reference
  (`tools/glm5_next/vision_reference.py`): it amplifies BF16 rounding chaotically in its middle
  blocks (two BF16 implementations land 7-25 % apart end to end), so each block and the merger,
  given the reference's own BF16 state, must stay within 1.25 × Transformers' BF16 error on the same
  input (measured 1.03-1.16 ×), and end to end within 2 × Transformers' error (0.10-0.23 against
  Transformers' 0.07-0.26). Through the Engine (`tests/models/glm5_next/test_vision_engine_real.cpp`):
  the examples' chart title, the image-and-video case `NIFER-9`, concurrent media requests equal
  alone, a resumed repeat and a resume inside an image equal their cold runs. The streamed pass
  gives the resident tower's bits, and an Engine with the tower in device memory gives the streamed
  Engine's tokens.
- **Cost.** Streamed (the default), the tower takes 1.13 GB of pinned RAM, and an encode lends
  370 MiB of staging (two layer slots and the 320 MiB merger) beside its handoff and pass. The pass
  over the four reference images took 411.0 ms streamed against 408.5 ms resident: the 1,075 MiB
  upload hides behind compute. Resident (`--vision-offload off`), the weights take 1.13 GB of VRAM,
  about 80 expert frames. An encode of 320-530 merged tokens took 96-149 ms on the GPU and 20 tokens
  22 ms (resident, after warm-up).

## 16. Configuration

The engine-wide expert-residency group (`--experts vram=…,ram=…[,ssd=…]` or counts,
`--expert-placement FILE`, `--expert-swap`, `--expert-ram-mib`, `--expert-state`; MoE §4.8) and the
sizing headrooms (`--vram-headroom-mib`, `--ram-headroom-mib`) are shared. This family's terms:

| Option | Default | Meaning |
|---|---|---|
| `--spec dflash2 --draft-model FILE --draft-tokens N|auto` | off | DFlash2, the only drafter; N in 1-7 or `auto` (calibrated 0-5, §10.2) |
| `--spec-min-p P` | 0 | Verify drafts only up to the first whose temperature-1 selector probability falls below P (§10.2) |
| `--kv-dtype bf16\|int8` | `bf16` | **Changes output**; every other profile is refused |
| `--max-context N` | the Engine's | Up to 1,048,576, from the device cache |
| `--prefill-chunk N`, `--prefill-chunk-beside-decode N` | 4,096, 1,024 | Chunking (§11) |
| `--prefill-expert-stream on\|off` | on | The streamed ring (§11.2) |
| `--host-context-mib N` | 2048 | The prefix Host tier; must be positive while the cache is on |
| `--vision`, `--vision-offload auto\|on\|off`, `--vision-max-merged-tokens N` | off, auto (on) | Image and video prompts (§15.2) |
| `--reasoning-effort low\|high\|max` | max | Template default; requests may override |

Refused at startup, each naming the reason: `--spec` other than `dflash2`, n-gram drafting
(`--ngram-draft-tokens`, `--ngram-archive-mib`, `--ngram-volume`), draft trees, fast-path
candidates and exclusions other than `expert-a4-residual` (NVFP4 artifact) and `ggml-bf16-prefill`
(GGUF artifact), `--expert-tail-skip`, `--route-trace`, and for the
prefix cache the Disk tier, persistence, Device snapshot slots, packed Host state and tool-call
holds.

Converter: `--model DIR --recipe glm5_3_flash_nvfp4_orcarouter` (default NVFP4),
`glm5_3_flash_nvfp4_orcarouter_q8dense` (opt-in Q8-dense, §17.2), `--gguf PATH --recipe
glm5_3_flash_gcsa_gguf` (GGUF K-quant, §3.3); the DFlash2 companion with `--components dflash2
--source dflash2=PATH --recipe glm5_3_flash_dflash2`.

## 17. Rejected and not built

Each entry names the option and the reason or measurement that decided it. The rejections that
decided the shared machinery are in
[MoE offload design §10](moe-offload-design.md#10-rejected-and-not-built).

### 17.1 Rejected

| Option | Reason |
|---|---|
| **The EXL3 3.05 bpw builds as the artifact** (MikeRoz's and turboderp's) | A new codec and kernels, ≈ 7k lines (trellis decode, an `exl3` Linear, an E3A16 canonical arithmetic, a CPU kernel), where the NVFP4 artifact reuses the built W4A16 kernels, the `nvfp4_expert_rg16_v1` layout and the converter's compressed-tensors source. EXL3 3.05 bpw measured mean KL 0.047 against BF16 in alesha-pro's table (§18.3); it stays an option on an owner request |
| Requantizing the artifact's weights at import | Changes the weights for no speed gain; a lighter format is a recipe decision, not an import transform |
| Any SSD cache, spill or swap file; memory-mapped experts | The artifact already holds the exact bytes (MoE §10.1); a mapped file's unbuffered reads measured a quarter of the unmapped rate on Flash-Next (FN §12.3) |
| Zero-copy SM reads of host experts | 10-14 GB/s against 24-27.6 staged (Flash-Next, MoE §5.2) |
| Expert deferral, dynamic top-k, low-bit miss copies, expert skipping in the default configuration | They change the model; top-k 6 was rejected on quality by ox-boost (§18.1). The family refuses `--expert-tail-skip`; an opt-in approximate mode is not built (§17.2) |
| A promotion every decode token | 30.7-31.7 against 41.2-41.3 tok/s at one per 32 tokens, with no hit-rate gain (§6.3) |
| Drafting every round, or a fixed K ≥ 4 | Every-round drafting is slower than plain at two and four requests; fixed K = 4-5 are slower than K = 3 at one request (§10.3); the calibrated length reaches 4-5 only where the drafts' probabilities are high (§10.2) |
| MTP drafting with the checkpoint's MTP layer | Both public GLM-5.3-Flash MTP experiments measured slower over PCIe-transferred misses, and Project Maya traced why on one GPU: consecutive tokens share ~1 of 8 experts per layer and none of their RAM pulls (§18.2). The layer stays in the artifact, not loaded |
| A BF16 KDA state | Saves ≈ 0.1 ms of a ~25 ms token for a numerical change |
| Folding W_uk into W_qb or W_uv into W_o offline | Changes the stored weights and reads more bytes per token (50 M and 134 M parameters against 33 M and 75 M) |
| The staged wide prefill pass, and an expert-group pipelining design | The staged pass could not overlap: the MoE's staging-overlap route is refused for wide calls, because overlapped wide passes measured nondeterministic in Flash-Next (FN §13.4). The streamed ring (§11.2) replaced both, reusing the frame pools and the wide route's combine order |
| An SSD share in the serving split | Slower at the serving settings (§6.1) |

### 17.2 Not built

| Option | State and reason |
|---|---|
| **8-bit dense recipe** | Built: the orcarouter recipe stores KDA q/k/v/o, MLA query down/latent/up/output, the dense MLP and the shared expert as `q8_g32_fp16` (grouped MSE scales) from the source's BF16; the embedding, `lm_head`, routers, KDA control projections, key_up/value_up and the indexer stay BF16. Model 15.44 → 9.17 GiB, +9 % one-request decode (§1). strata-glm measured int8-per-32 dense at 0.0105 mean KL against 0.0066 for BF16 dense after a 2,600-token prompt (§18.4). Not served: against the BF16-dense artifact (`infernix-perplexity` quick corpus, context 8192, stride 4096, INT8 KV, 252,823 positions) it measured mean KL 0.0058, p99 0.066, top-1 97.3 % (PPL 3.1515 against 3.1489), against the gate's 0.002 / 0.05 / 99 %; the BF16 artifact's own chunk-order floor is 0.00012 and 99.9 %. The error is spread over every context bucket (0.005-0.007). Untried: which projections carry it (a KDA-BF16 variant keeps most of the bytes) |
| **A teacher-forced KL gate on frozen texts** (≥ 2,500 positions in the GLM template, with a block past 8K tokens; against the BF16 reference forward: mean KL ≤ 0.002 nats, p99 ≤ 0.05, top-1 ≥ 99 %) | Defined but not run. The current qualification is §14 |
| **An opt-in approximate expert mode** (Maya's per-layer keep map, strata-glm's weight threshold, `ssd` or `vram` scope, a static reproducible rule) | Not built: it changes the model's outputs and would need its own labelling and KL gates. Maya measured +23 % decode at KL 0.018 to its own model; strata-glm +30 % at KL 0.022 for SSD-only experts below ≈ 3 % weight (§18) |
| **Expert volumes on several drives: 12-way striped records, per-volume read queues, landed-unit masks** | Designed for the RTX 5090 machine with four NVMe drives, where two drives measured additive on 14.16 MB records (2 × 5.8 → 11.7 GB/s; 12.7 GB/s for a 990 PRO beside an Optane P5800X, 2026-10-10) and a striped record would land in ≈ 1.1-1.3 ms against ≈ 2.3 ms whole (model). The production machine has one drive and serves with no SSD share |
| **SSD lookahead reads of predicted experts** | Project Maya found only ~40-45 % of disk misses predictable 1-4 layers ahead, and its lookahead was exact but slower on one NVMe (§18.2). No SSD share is served |
| **Draft-informed SSD prefetch** (SP-MoE, MoE-SpeQ) | Needs measured recall first; no SSD share is served |
| **An online round-time fit for the calibrated length** | T(k) is a constant table measured at one placement (§10.2); on greedy essays the chosen mix of lengths ran ≈ 10 % slower a round than the table predicts (§10.3). Flash-Next's MTP fit measured 6 % slower than its constant model on code (dflash.md) |
| **Mixed rounds** (a prefill call inside the decode round's Forward) | A 30,614-token prompt beside a decoding lane prefilled at 2.53k tok/s while that lane's 1,500 tokens took 55.6 s against 42.3 s alone: the lane gets one round per walk step either way. A mixed round would save one decode round (≈ 27 ms) per 1,024-token step of ≈ 430 ms (≈ 6 % of the prefill), give the decoding lane no more rounds, and could not run for a lane that publishes to the prefix cache, since a mixed call's bits differ from its own call's |
| **The KV host tier** (pooled keys on the device, latent pages in pinned RAM with a device cache of hot pools) | The device cache holds every served context on this card; the tier would trade expert frames for RAM-tier slots and a PCIe read per selected token without a workload that needs it. It stays the design for a smaller card or contexts past the device |
| **`fp8` and `nvfp4` latents; Hadamard-domain latent profiles** | Refused until qualified; INT8 measured within the BF16 rounding floor (§7.4). Orthogonal rotations commute with the absorbed product, so rotated profiles would apply |
| **An FP8 pooled-key scan with FP32 rescoring past 256K** | The indexer scan grows linearly (≈ 0.5 ms per token across 11 layers at 1M, model); no served context needs it |
| **VRAM and RAM policy candidates** (LFRU with near-miss credit from router ranks 9-16, hot-layer reservations, aged LFU, soft per-layer shares, a static hot set with an LRU remainder) | The GLM engines disagree (ox-boost: decayed LFU lost to a router-score cache; Maya: LFU beat LRU by 10 points; strata-glm: a static hot set beat LRU, §18), so a replay on this model's route traces would decide. The family records no route traces yet (`--route-trace` is refused) |
| **Prefill read-ahead and prestage** (the next layer's SSD-only records in a deep ring; the next layer's most-routed RAM experts during attention), **the last layer's tail for non-final tokens**, **a CPU share of the least-routed prefill experts** | Not built; the walk is compute-bound at the serving split (§11.1) |
| **Stepping a large vision encode between other lanes' rounds; fused vision Q/K/V GEMMs; a D64 2-D rope** | Not built |
| **Per-kernel decode budgets under ncu** | Not measured; the Nsight round profiles of §6.5 and §10.3 attribute the round |
| **The RTX 5090 machine** (32 GB, Windows, PCIe Gen5 x8) | The shared tier's smaller-card path supports it, but this family was never measured there |

### 17.3 Out of scope

- n-gram drafting: the artifact carries no n-gram table, and copy proposals from a request's own
  history are left out (the frontend builds no n-gram sources).
- Multi-GPU, AMD GPUs, GPUs other than `sm_120a`.
- Any change to the routed experts' weights in the default configuration.

## 18. External evidence

Measurements by other GLM-5.3-Flash implementations that this family's decisions lean on. The
numbers are the sources' own, read 2026-10-10 to 10-12 and not reproduced here ([M] = the source
reports a run). The shared machinery's evidence is in MoE §2 and §4-§8.

### 18.1 Other engines

| Engine | Mechanism | GLM-5.3-Flash result |
|---|---|---|
| **ExLlamaV3 offload fork, adrienbrault** (2 × RTX 5090 at x8, Ryzen 7 9800X3D, 64 GB DDR5-6000) | EXL3 2.05 bpw; 96-102 experts per MoE layer computed on the CPU by 8 threads | [M] 64.8 tok/s at one stream, 74-98 total at 2-4 streams; MTP 1.741 tokens per step; prefill ≈ 2,000 tok/s; the CPU worker reads cold EXL3 experts at ≈ 61 GB/s of a 63 GB/s ceiling |
| **FreeToken port, anoane** (RTX PRO 6000 96 GB, x16, 160 GiB host) | All experts pinned, 17 layers VRAM-resident plus a 576-slot LRU; FP8 non-expert weights | [M] decode 22.2-22.7 tok/s, prefill 703 → 2,853 tok/s (chunk 8192); MTP slower (≈ 12 tok/s) |
| **FreeToken ox-boost patch set** (RTX PRO 6000, DDR5-3600) | Router-score "soft" cache fed by near-miss experts beat LRU; decayed LFU **lost** ("drift-shaped locality"); hot resident layers; next-layer top-4 prefetch; FP8 non-expert weights | [M] decode 29.2 → ~38 tok/s (36.8 at 128K); soft cache 38.1 → 33.7 misses per token; hot layers +29 %; prefetch P = 4 +8 %, P = 6-8 slower; FP8 dense +28 %; MTP verify −13 % |
| **strata-glm** (archived) | NVFP4 experts repacked for ggml, int8 dense, static VRAM hot set + LRU, pinned RAM tier, mirrored `experts.bin` on two drives | [M] decode 16-18 tok/s chat, 12.4-12.9 code on an RTX 5090 with 128 GB; ≈ 13 % of experts in VRAM, 58 % in RAM, 22.2 disk reads per token |
| **KTransformers** (SOSP'25) | Routed experts on the CPU (AMX prefill, AVX-512 decode); "expert deferral" changes the mathematics | Deferral is out of scope |
| **llama.cpp / ik_llama.cpp** | `--n-cpu-moe` / `-ot exps=CPU`; mmap pages from SSD with 111× read amplification on interleaved GGUF | No 5090 number found |
| **vLLM `glm53-flash` image, SGLang** | All experts resident; FlashMLA sparse kernels are SM90/SM100 only | Reference logits only |

GLM-5.3-Flash misses far more often than Flash-Next at the same cache share: Flash-Next's LFRU
replay misses 18.4 % / 5.2 % / 1.4 % at 16.3 % / 32.6 % / 48.8 % of the bank (MoE §4.3), against
24 % → 10 % for ox-boost's policies at ≈ 47 %, and 35-56 % / 16-39 % / 18-26 % at 10.8 % / 21 % /
31 % on Maya's V100 runs. Two causes fit: 8 active of 288 (2.8 %, against 10 of 512) and the
auxiliary-free bias, which keeps experts near-uniformly loaded.

### 18.2 Project Maya

[Project Maya](https://github.com/mw00/project-maya) (MIT, read 2026-10-11) is Strata rewritten for
this one model: VRAM, pinned-RAM and SSD expert tiers, a CPU lane, MTP drafting, its own 2-4-bit
GGUF quants (Maya-M 116 GB: KL 0.093 against BF16 on alesha-pro's table). Its findings and what this
family did with each:

| Maya finding | This family |
|---|---|
| VRAM hit rate 44-65 % at 10.8 % of experts, 61-84 % at 21 %, 74-82 % at 31 %; 98 % at 70 % (two cards) | §18.1's locality comparison |
| Policy replay: LRU 67 %, aged LFU 77.5 %, decayed frequency 78 %, a learned model 77 %, Belady 87 % | A policy candidate, not built (§17.2) |
| Disk lookahead: only ~40-45 % of disk misses predictable 1-4 layers ahead; exact but slower on one NVMe | Not built (§17.2) |
| MTP on one GPU does not pay: consecutive tokens share ~1 of 8 experts per layer and none of their RAM pulls | No MTP drafting (§17.1) |
| CPU lane: one thread per physical core, never hyperthreads (21.8 → 13.3 tok/s); work in ~48 pieces claimed in turn (14.3 → 17.1 tok/s) | The shared CPU team's dynamic counters (MoE §5.3) |
| A mapped model file sends every direct read down a slower path on Windows | The shared tier never maps the artifact (MoE §4.7) |
| A tier-accounting bug left 12 % of experts in no tier; fixing it took real chat 13.2 → 17.0 tok/s | The shared tier's one-tier rule (MoE §4.1) |
| Keeping a prompt's disk reads in RAM for the answer gave more disk reads, not fewer | No prompt landing reserve |
| Prefill indexer scores as a register-tiled FP32 GEMM (78.8 → 5.6 ms) | §7.3's prompt scores |
| INT8 latent cache: no measurable quality change | §7.4: serving uses `int8` |
| KDA layers as CUDA graphs: +2.35 % | Whole-round graphs (§5.1) |
| `TRUNK_KEEP` map: skip missing low-ranked experts in late layers, +23 % at KL 0.018 to its own model | The approximate mode, not built (§17.2) |

### 18.3 The quant survey

KL is comparable only within one table, since each card uses its own reference, texts and engine.
alesha-pro's table measured several formats against the same BF16 teacher logits:

| Quant | Size | KL |
|---|---:|---|
| orcarouter NVFP4 (the default) | 191.0 GiB | 0.073 against FP8, perplexity +3.8 %, top-1 91.7 % (its card, not this table) |
| EXL3 3.05 bpw (turboderp; MikeRoz's uncensored build unmeasured) | 122.3 GB | 0.047 against BF16, top-1 93.1 % |
| Maya-M GGUF IQ2_S / IQ3_XXS | 116.0 GB | 0.093 against BF16, top-1 90.7 % |
| EXL3 2.4 bpw mix / 2.05 bpw | 98.1 / 85.1 GB | 0.096 / 0.121 against BF16 |

Other cards, each against its own reference: GSQ-RCO 3.5-bit 137.1 GB 0.071 against Q8_0; AesSedai
IQ4_XS / Q4_K_M / IQ3_S 0.117 / 0.050 / 0.283; REAP-pruned builds 0.45 and worse (pruning costs far
more than quantization); GCSA-AiLab RCO-GSQ none published (served as the optional GGUF K-quant
artifact, §3.3). orcarouter also publishes
`orcarouter/GLM-5.3-Flash-Uncensored-FP8`, the same refusal edit at source precision.
`RedHatAI/GLM-5.3-Flash-NVFP4` (the base model, same layout) imports through the same recipe.

### 18.4 strata-glm

[strata-glm](https://github.com/sergqwer/strata-glm) (archived, read 2026-10-12) ports
GLM-5.3-Flash NVFP4 to Strata on an RTX 5090 with 128 GB of DDR5 and two NVMe drives, measured with a
teacher-forced KL harness against a BF16-dense reference:

- Quantizing the dense classes costs real KL, more at long context: int8-per-32 dense 0.0105 mean
  KL against 0.0066 for BF16 dense after a 2,600-token prompt; NVFP4 per class 0.016 (head) to
  0.028 (MLA).
- INT8 latents: KL 0.0049 → 0.0069 over 2,600 tokens.
- A static VRAM hot set + LRU remainder: 15.66 → 17.15-17.31 tok/s; disk reads 30.6 → 22.2 per
  token.
- Opt-in skipping by routing weight: +30 % at KL 0.022 (≈ 3 % threshold, SSD-only).
- NVFP4 codes do not compress: code entropy 3.97 of 4 bits.

## 19. Sources

- Model: [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (config,
  76,108-tensor index, template); Transformers `src/transformers/models/glm5_next/modeling_glm5_next.py`
  and `configuration_glm5_next.py`; GLM-5 report arXiv 2602.15763.
- Default artifact: [orcarouter/GLM-5.3-Flash-Uncensored-NVFP4](https://huggingface.co/orcarouter/GLM-5.3-Flash-Uncensored-NVFP4);
  [RedHatAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/RedHatAI/GLM-5.3-Flash-NVFP4) (same layout);
  [orcarouter/GLM-5.3-Flash-Uncensored-FP8](https://huggingface.co/orcarouter/GLM-5.3-Flash-Uncensored-FP8)
  (the quantization source).
- GGUF K-quant artifact (§3.3): [GCSA-AiLab/GLM-5.3-Flash-Uncensored-RCO-GSQ-GGUF](https://huggingface.co/GCSA-AiLab/GLM-5.3-Flash-Uncensored-RCO-GSQ-GGUF).
- EXL3 builds: [MikeRoz/GLM-5.3-Flash-Uncensored-3.05bpw-h6-exl3](https://huggingface.co/MikeRoz/GLM-5.3-Flash-Uncensored-3.05bpw-h6-exl3)
  and [turboderp/GLM-5.3-Flash-exl3](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3).
- Drafter: [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2);
  SGLang PR sgl-project/sglang#36708 (the GLM DFlash2 feature capture).
- FreeToken: arXiv 2608.16157; `github.com/FlashML-org/FreeToken`; ports
  `github.com/anoane/02-glm53-flash-nvfp4-freetoken` and `github.com/Cerynitius/freetoken-ox-boost`.
- strata-glm: `github.com/sergqwer/strata-glm` (`README.md`, `docs/glm-port-plan.md`).
- Other engines: `github.com/adrienbrault/glm5.3-flash-2x-rtx5090`, KTransformers (SOSP'25),
  llama.cpp discussion #27149, `github.com/deepseek-ai/FlashMLA`, `github.com/deepseek-ai/DeepGEMM`.
- Papers: Kimi Linear arXiv 2510.26692; mHC arXiv 2512.24880; SP-MoE arXiv 2510.10302; MoE-SpeQ
  arXiv 2511.14102; Zhang, MoE expert caching arXiv 2608.07911.
- Project Maya: `github.com/mw00/project-maya` (`README.md`, `CHANGELOG.md`, `docs/SETTINGS.md`,
  `docs/GLM-KDA-GRAPHS.md`, `bench/results/PROGRESS.md`, `bench/results/2026-10-05-glm-single-gpu/`,
  `bench/results/2026-10-06-glm-spec/`).
- ninferpro: the [MoE offload design](moe-offload-design.md); the
  [Flash-Next design](qwen3_8-flash-next-design.md) and [guide](../qwen3_8-flash-next.md) (cited as
  FN §x).
