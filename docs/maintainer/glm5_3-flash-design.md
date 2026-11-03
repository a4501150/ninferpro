# GLM-5.3-Flash on the RTX PRO 6000 and RTX 5090: design

**Status: design for a planned family; not implemented.** This document designs
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

**Default artifacts** (owner decision, 2026-10-11): the two ExLlamaV3 (EXL3) 3.05 bpw builds, which
share one format variant, one per-module bit allocation and one set of tensor shapes (§7.1), so one
importer and one set of kernels serve both:

- **[MikeRoz/GLM-5.3-Flash-Uncensored-3.05bpw-h6-exl3](https://huggingface.co/MikeRoz/GLM-5.3-Flash-Uncensored-3.05bpw-h6-exl3)**
  (branch `main`), the **primary** artifact: orcarouter's abliterated GLM-5.3-Flash
  (`orcarouter/GLM-5.3-Flash-Uncensored-FP8`) quantized by ExLlamaV3 1.4.6, with the MTP layer;
- **[turboderp/GLM-5.3-Flash-exl3](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3)** (branch
  `3.05bpw`): the base model quantized by ExLlamaV3 1.4.4, without the MTP layer and without the
  indexer's pooling tensors, which the import takes from `zai-org/GLM-5.3-Flash` (§7.3).

Both store the routed experts (layers 3-44) as 3-bit EXL3 trellis codes (9 MiB per expert), the
attention projections and shared experts at 5 bits, the dense MLPs at 4 and the head at 6. The
NVFP4 build (`orcarouter/GLM-5.3-Flash-Uncensored-NVFP4`) is an optional later recipe (§7.4).

**Two target machines, both first class:**

- the **production machine**: one RTX PRO 6000 Blackwell (96 GB, `sm_120a`), Linux, PCIe Gen5 x16,
  89 GB of host RAM, one Gen5 NVMe (T710) holding the models — every Flash-Next measurement in the
  repo comes from this card;
- the **RTX 5090 machine** (32 GB, Windows 11, PCIe Gen5 x8): an i9-13900K host with 96 GB of
  DDR5-5800 and four NVMe drives.

Neither machine holds the whole 107 GiB expert bank (§2.3) in VRAM, so both run a tiered placement; the
design states the placement and the expected speed for each machine (§5, §19) and does not assume
SSD streaming is the only mode — on the production machine the default placement needs no SSD
(§19.1).

---

## Contents

1. [Goals, baselines and acceptance](#1-goals-baselines-and-acceptance)
2. [Model facts](#2-model-facts)
3. [Machine facts](#3-machine-facts)
4. [What other GLM-5.3-Flash implementations measured](#4-what-other-glm-53-flash-implementations-measured)
5. [Performance model](#5-performance-model)
6. [Design overview](#6-design-overview)
7. [Artifacts, import and the striped expert volumes](#7-artifacts-import-and-the-striped-expert-volumes)
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
- the MTP layer as a speculative drafter (`--spec mtp`, for artifacts that carry it) and n-gram copy
  proposals;
- the KV profiles that apply to MLA latents (§13.5), `bf16` and `int8` first;
- the hybrid prefix cache in `infernix-serve`;
- the GLM chat template, reasoning effort, and GLM tool calls in the OpenAI and Anthropic protocol
  surfaces (§21);
- routed experts on the shared offload tier in **any ratio of VRAM : RAM : SSD** —
  `--experts vram=…,ram=…[,ssd=…]` or counts, a placement file, switched at runtime through
  `POST /admin/expert-placement` (MoE §4.8) — with this family's expert records in **expert
  volumes on one to four NVMe drives read in parallel** (§11) as the SSD tier's source;
- an **opt-in approximate expert mode** (§9.8), off by default and labelled wherever it is on;
- both default EXL3 artifacts, converted by one recipe (§7.2).

**Fidelity.** Every quantized tensor is imported bit-exactly (trellis words, sign-scale vectors,
codebook multiplier) and computed in one canonical EXL3 arithmetic, **E3A16** (§9.5): BF16
activations, Hadamard rotations in a fixed FP32 order, and exact integer dot products of the
codebook's values against block-scaled rotated activations. A routed expert's output is therefore
the same bits on the GPU, on the CPU and from any tier (the placement-invariance principle of
MoE §2), and a dense layer's output is the same bits on its decode and prompt routes. E3A16 uses
the codebook's exact values where ExLlamaV3 rounds them to FP16, and rounds rotated activations
once to 14-bit blocks where ExLlamaV3 rounds them to FP16; both differences are at ExLlamaV3's own
rounding level, and §20.3 bounds them against a reference forward that decodes the weights exactly
as ExLlamaV3 does. No recipe changes a weight: the artifact as its producer quantized it is the
model. Two prefill speed permissions are designed and quality-gated (§17.4), never defaults: 8-bit
rotated activations for dense prefill GEMMs, and for wide-route routed experts.

**The quantization's own quality is reported, not claimed.** turboderp's base build measured mean KL
0.047 against BF16 in a third-party table (§7.4); MikeRoz's uncensored build has no published
measurement. M4 measures both against their FP8 sources (§20.3).

**Vision** (the 24-layer tower) is deferred: the artifact keeps it, the binder refuses `--vision`
for this family until a later track adds it (§23).

### 1.2 Baselines

All comparisons run on the target machines with the same prompts, output caps and sampling.

| Baseline | What it shows |
|---|---|
| **Project Maya** (`mw00/project-maya`, Strata's GLM-5.3-Flash engine; §4.2) with its own quants, Maya-M (116 GB) and Maya-L (156 GB), where they run | **The main baseline**, compared as each engine runs in practice (each with its own quantization), as Flash-Next was compared with Strata. Maya-M's experts (≈ 8.6 MB) are close to this design's 9.4 MB; its KL is about twice the default artifact's on the one table that measured both (§7.4), so the comparison reports quality beside speed |
| **ExLlamaV3** (the format's producer) and adrienbrault's offload fork, with the **same EXL3 artifacts** where they fit | Identical weights, so speed is compared without a quality footnote. The 3.05 bpw build needs ≈ 120 GB of VRAM + RAM with no SSD tier, more than the 5090 machine has; MikeRoz's 2.51 bpw build (≈ 105 GB) is the fallback for that row |
| **strata-glm** (§4.3) with the NVFP4 build | The only published engine measured on a machine like the 5090 machine (16-18 tok/s chat on an RTX 5090 with 128 GB and x16); a token-id CLI, so it is compared on decode and prefill only |
| **KTransformers** (GLM-5.3-Flash tutorial; CPU/GPU hybrid MoE; AMX/AVX-512 kernels) | The best-known consumer offload engine for this model; its GLM-5.3 numbers are recorded in §4.1 with their hardware |
| **llama.cpp / ik_llama.cpp** with `--override-tensor` expert offload, GGUF of the same model | The common community path; streams from page cache with mmap when RAM is short |
| **FreeToken** (§4.1) | The offload engine whose Flash-Next numbers FN §3.3 used as a reference |
| The engine's Flash-Next on the production machine | Not a quality baseline (another model); it is the reference for what each mechanism achieved on the same card |
| **vLLM `glm53-flash` image / SGLang**, all experts resident, on a large GPU | Generic-framework all-resident ceiling; not runnable here |

### 1.3 Targets

Set on 2026-10-11 for the 5090 machine when the owner made the EXL3 builds the default; no
measurement existed for either artifact then, and §7.4 keeps the NVFP4 projections the first targets
were set from. Primary configuration: MikeRoz's uncensored EXL3 3.05 bpw artifact, `--kv-dtype
bf16` (MLA latent cache), `--max-context 131072`, C = 1, two expert volumes on `D:` and `G:`
(§11.6), today's 79 GiB of available RAM, text with the template's default `max` reasoning
effort, 4K prompt, 512 output tokens.

| Metric | Must | Stretch | Model (§5.3-5.4), central (range) |
|---|---:|---:|---:|
| Plain decode, warm cache, C = 1, non-repeating session (§20.4) | ≥ 25 tok/s | ≥ 40 | 31 (24-42) |
| Best speculative mode, same corpus | ≥ plain, never slower beyond noise | ≥ 45 | ≈ plain (§16) |
| Decode with two expert volumes, relative to one | ≥ 1.15× | ≥ 1.3× | 1.17-1.32× |
| Prefill, 32K fresh prompt | ≥ 2,500 tok/s | ≥ 5,000 | 3,500 (3,000-4,200) |
| Time to first token, 4K fresh prompt | ≤ 6 s | ≤ 3.5 s | 3.7 (3.4-4.5) |
| Decode at 128K actual context, relative to 4K | ≥ 85 % | ≥ 95 % | ~94 % |
| Execution fidelity against the EXL3 reference forward, §20.3 | within the gate | — | — |

**Production machine (model, §5.3-5.4; no owner-set targets yet):** the default placement holds
the whole bank in VRAM + RAM (§19.1), so there is no SSD term; modelled decode ≈ 80 tok/s central
(45-115, the CPU E3A16 cost unmeasured), 32K prefill ≈ 3,400-4,900 tok/s, 4K time to first token
≈ 1.4-1.8 s. M4's first end-to-end numbers on that card replace these, and the owner sets the
targets then.

The acceptance report gives the median and range of at least three runs, the worst case, the
expert service composition (VRAM hits, RAM CPU-served, RAM DMA, SSD reads, SSD bytes per token),
and the machine each number came from. A missed target is reported as missed; targets are not
changed after measuring.

These targets are below Flash-Next's (measured 380 tok/s one-request decode at `vram=100%` on the
production card, FN §1.3) because this model moves ~2.4× Flash-Next's expert bytes per token,
misses far more often at equal cache share, and does not fit its bank in any available VRAM (§5).
That is a property of the model on this hardware, not a design choice; §5.5 lists the hardware
changes that move the ceiling.

### 1.4 Non-goals

- Multi-GPU, AMD GPUs, GPUs other than `sm_120a`.
- Any change to the routed experts' weights, dynamic top-k, expert skipping, or lower-precision
  copies of cold experts in the default configuration (they change the model; FN §3.2 "considered
  and rejected"). The one exception is the opt-in, labelled approximate mode of §9.8.
- Requantizing or converting an artifact's EXL3 weights to another format at import.
- An SSD cache or spill file: the expert volumes are written once by the converter and only read
  afterwards (MoE §4.7's rejection).
- The vision tower in this track.
- A generic model graph or plugin layer. The offload tier is shared through plain composition —
  no base class, no virtual interface (MoE §2) — and this family adds to it only what §8.2 lists.

---

## 2. Model facts

### 2.1 Configuration (upstream)

From `zai-org/GLM-5.3-Flash` `config.json` and Transformers `configuration_glm5_next.py`. M0 checks
the EXL3 repositories' own `config.json`, tokenizer and template files (not gated) against these.

| Quantity | Value |
|---|---|
| Layers | 45 + 1 MTP layer (index 45). Layer i is KDA when i % 4 ≠ 3, sparse MLA when i % 4 = 3: 34 KDA, 11 MLA (3, 7, …, 43) |
| Hidden width | H = 4096; RMSNorm eps 1e-5, weight multiplies (no unit offset) |
| Residual | mHC, 4 streams (16,384 wide), two mixers per layer, 20 Sinkhorn iterations, eps 1e-6 |
| KDA | 64 heads, head dimension 128, short conv kernel 4 (per q/k/v, SiLU), gate lower bound −5, FP32 state 64 × 128 × 128 per layer |
| Sparse MLA | 64 heads; q_lora_rank 1536; kv_lora_rank 512; qk_nope 256, **qk_rope 0 (no positional encoding anywhere in the text model)**; v_head 256; softmax scale 256^−½ |
| Indexer (every MLA layer `full`) | 32 heads × 128, LayerNorm keys, softmax-gated pools of 4 tokens with a learned positional bias, relu scores weighted per head, top 512 pools (2,048 tokens) plus the incomplete tail |
| MoE | Layers 0-2 dense SwiGLU (I = 12,288); layers 3-44: 288 routed experts, top-8, width 2,048, 1 ungated shared expert of width 2,048 |
| Router | FP32 logits, sigmoid scores; selection on scores + `e_score_correction_bias`; weights = unbiased scores of the 8, normalized, × 2.5 (`noaux_tc`, n_group = topk_group = 1) |
| SwiGLU clamp | gate ≤ 10, up ∈ [−10, 10] (`swiglu_limit` 10), on routed, shared and dense MLPs |
| Vocabulary | 154,880; untied `lm_head`; final RMSNorm after the unweighted mean of the 4 streams |
| Context | 1,048,576 positions (no RoPE, so no YaRN question) |
| MTP layer | enorm, hnorm, `eh_proj` [4096 × 8192], one sparse-MLA block with its own indexer and MoE (288 routed + shared), `shared_head.norm`, the main `lm_head`; no hyper-connections (no `hc_*` tensors on layer 45). `index_share_for_mtp_iteration`: chained draft steps reuse the first step's selection |

### 2.2 Mathematics (upstream, Transformers `modeling_glm5_next.py`)

Written once here so every Op and oracle uses the same statement. Shapes per token; R is the
residual, 4 streams of H.

**Embedding.** R_s = embed(t) for s = 0..3 (four identical copies).

**mHC mixer** (`Glm5NextTextHyperConnection`, FP32 for the mixing weights):

```text
f      = fn · (R_flat / rms(R_flat))            fn: FP32 [24, 16384]; rms with eps 1e-5, no weight
pre    = sigmoid(f[0:4]  · scale0 + base[0:4]) + 1e-6
post   = 2 · sigmoid(f[4:8] · scale1 + base[4:8])
comb   = softmax_row(f[8:24] · scale2 + base[8:24] as 4×4) + 1e-6
comb   = comb / (colsum(comb) + 1e-6);  repeat 19×: comb /= rowsum + 1e-6; comb /= colsum + 1e-6
x      = bf16( Σ_s pre_s · R_s )                                     block input
... block (norm, attention or MLP) produces y ...
R_s'   = post_s · y + Σ_j comb[j][s] · R_j                           (comb transposed), BF16 streams
```

**Decoder layer.** `(post, comb, x) = attn_hc(R)`; `x = input_layernorm(x)`; `y = attn(x)`;
R = mix(post, comb, y, R); `(post, comb, x) = ffn_hc(R)`; `x = post_attention_layernorm(x)`;
`y = mlp(x)`; R = mix(...). Head: `logits = lm_head(norm(mean_s R_s))`.

**KDA layer** (`Glm5NextTextLinearAttention`):

```text
q, k, v = SiLU(causal_conv4(W_q x)), SiLU(conv4(W_k x)), SiLU(conv4(W_v x))     each 64 × 128
g       = W_fb (W_fa x) + dt_bias                                               64 × 128, FP32
log α   = −5 · sigmoid(exp(A_log[h]) · g)                                       per channel
β       = sigmoid(W_b x)                                                        64
q, k    = l2norm(q), l2norm(k)  (eps 1e-6);  q *= 128^−½
S_h     = S_h · diag(α);  S_h += β (v − S_h k) kᵀ;  o = S_h q                    FP32 state
o       = RMSNorm_head(o; w_o, eps 1e-5) · sigmoid(W_gb (W_ga x));  y = W_o o
```

This is exactly the existing `kimi_delta_attention` Op contract (lower bound in [−5, 0]), which is
built and qualified with no consumer yet (§14).

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
k_t   = LayerNorm(W_k x_t; γ, β, eps 1e-6)                        128
z_t   = W_g x_t                                                   128 (index_kpool_compress_gate)
pool p covers tokens 4p..4p+3 (pools start at the sequence's first token);
key_p = Σ_{i<4} softmax_i(z_{4p+i} + ape[i]) ⊙ k_{4p+i}           per channel; complete pools only
q_h   = W_qb_idx c_q                                              32 × 128
w_h   = (W_w x)_h · 32^−½
score_p = Σ_h w_h · relu(q_h · key_p · 128^−½)                     complete pools whose last token is visible
selected = top-min(512, P) pools  ∪  tail tokens 4·⌊n/4⌋ .. n−1 (n = visible count)
```

While the visible context is at most 2,051 tokens every token is selected and the layer equals
dense causal attention. Ties are broken by lower pool id in the engine (Transformers' `topk` leaves
them unspecified; recorded in §20).

**MoE** (`Glm5NextTextMoE`): `s = sigmoid(W_r x)` (FP32 logits); `top8 = argtop8(s + bias)`;
`w_i = 2.5 · s_i / (Σ_{top8} s + 1e-20)`; `y = Σ_i w_i E_i(x) + E_shared(x)` with
`E(x) = W_d (SiLU(min(W_g x, 10)) ⊙ clamp(W_u x, −10, 10))`.

**EXL3 linear** (upstream, ExLlamaV3, the stored form of every quantized matrix; §9.5 gives the
arithmetic): `y = svh ⊙ H_N( H_K(x ⊙ suh) · W′ )`, with H a block-diagonal 128-point Hadamard
transform, `suh` and `svh` FP16 sign-scale vectors, and W′ decoded weight by weight from the
trellis codes.

### 2.3 Parameters and bytes (upstream sizes from the artifacts' `quantization_config.json`; model elsewhere)

| Item | Parameters | Bytes stored (EXL3 3.05 bpw builds) | Bytes read per decode token |
|---|---:|---:|---:|
| One routed expert: gate, up [4096 → 2048] and down [2048 → 4096], 3-bit trellis + FP16 suh/svh | 25,165,824 | **9,437,184 trellis (9 MiB = 2,304 × 4 KiB) + 36,864 vectors** | — |
| Routed bank, 42 × 288 | 304.4 B | **114.60 GB (106.7 GiB)** | 336 experts: **3.18 GB** |
| MTP routed experts, 288 (MikeRoz only; `mtp_bits` 4) | 7.25 B | 12,582,912 + 36,864 per expert; 3.63 GB | 8 per draft step |
| KDA layers: `qkv_proj` (fused q, k, v), `o_proj` at 5 bits; small projections FP16 | 4.68 B | 3.09 GB | same |
| MLA layers: q_a, q_b, kv_a, kv_b, o, indexer `wq_b` at 5 bits; indexer keys FP16 | 1.37 B | 0.75 GB | same |
| Shared experts 3-44 (5 bits), dense MLP layers 0-2 (4 bits) | 1.06 B, 0.45 B | 0.66 GB, 0.23 GB | same |
| `lm_head` (6 bits, `head_bits`), routers (FP16), mHC (FP32), norms | 0.63 B, 0.05 B, 0.04 B | 0.48 GB, 0.10 GB, 0.14 GB | same |
| **Text dense total (embedding excluded)** | **8.23 B** | **5.45 GB** (BF16 would be 16.5 GB) | same |
| Token embedding (BF16) | 0.63 B | 1.27 GB | one row |
| MTP dense (MikeRoz only) | 0.19 B | ≈ 0.2 GB | per draft step |
| Vision tower (6 bits) | | ≈ 0.97 GB | — |
| **Whole artifact** (index `total_size`) | 321 B | MikeRoz **126.11 GB**; turboderp **122.29 GB** (no MTP layer) | |

The 321 B matches the published count, which checks the shape reading. The 3.83 GB between the two
artifacts is MikeRoz's MTP layer.

Consequences, before any mechanism is chosen:

- **Experts dominate even when every expert hits.** 3.18 GB of routed weights per token against
  5.45 GB of dense weights: at the 5090's sustained ≈ 1.5 TB/s (estimate; the budget Flash-Next's
  kernel plans used, FN §8.10) a token reads ~8.9 GB and takes ≈ 6.2 ms with every expert in VRAM
  (§9.3); at the
  production card's measured 1.79 TB/s (FN §8.3), ≈ 5.3 ms (model). The all-hit ceiling is ~160
  tok/s on the 5090 and ~185 on the production card (model).
- Neither machine holds the bank in VRAM: **≈ 21 % of the experts fit in the 5090's frame pool,
  ≈ 80 % in the production card's** (§19.1); the host holds the rest (§19.2), so the SSD tier is
  empty at the production machine's default placement and carries ~3 % of misses on the 5090
  (§5.2).
- **A miss moves 9.4 MB**, 3.4× Flash-Next's 2.76 MB record. At the 5090 machine's x8 link
  (27.6 GB/s measured, FN §4.2) that is 0.34 ms per miss; at the production machine's x16 (model:
  2× the measured x8 rate; strata-glm measured 50.8 GB/s on an x16 5090, §4.3), 0.19 ms; on one
  990 PRO ≈ 1.6 ms (model from §3.3).
- **The KV cache is small.** One MLA latent is 512 values per token and layer: 12 layers × 1 KiB =
  12 KiB per token in BF16 plus 64 B of pooled index key per layer. 128K tokens of BF16 latents are
  1.6 GB, so a long context costs the expert cache far less than Flash-Next's GQA KV did.
- **The recurrent state is larger.** 34 × 64 × 128 × 128 × 4 B = 136 MiB of FP32 KDA state per
  sequence, read and written every token (285 MB of traffic per token at C = 1).
- **Each weight costs integer work to decode** (§9.5): about 4-5 integer instructions per weight on
  the GPU and about one vector micro-operation per weight on the CPU, where NVFP4 costs a table
  lookup. The GPU stays memory-bound; the CPU lane does not (§12.2).

---

## 3. Machine facts

### 3.1 The two platforms

**Production machine** (the machine this repository builds and serves on; AGENTS.md): one RTX PRO
6000 Blackwell, `sm_120a`, 96 GB, 188 SMs, DRAM 1.79 TB/s (measured, Flash-Next, FN §8.3), PCIe
Gen5 x16; 89 GB of host RAM; one Gen5 NVMe (T710) holding the models. The host CPU is not recorded
in the repository — unverified; M0 measures the E3A16 expert cost on it before any constant
depends on it. The T710's rate at this family's 9 MiB records is unmeasured; M0's probe (§23)
measures it. The shared tier's placement matrix measured on this card (Flash-Next, 2026-10-30) is
MoE §8.2's: half the experts in RAM decode at 263 tok/s, half on SSD at 68, against 380
all-VRAM, one request each.

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
probe was sized for the NVFP4 record (14,155,776 B); whole-record cells use that size. The 9 MiB
record and its 0.75 MiB units were not probed (another session held the machine's benchmark locks
that day): their figures below are **model** (interpolated), and M0's C++ probe
(`tools/glm5_next_probe/`, §23) measures them on both machines.

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
- **A drive saturates with 1-2 MiB reads at QD 2-4, and with 256 KiB reads at QD 8.** The EXL3
  record's 0.75 MiB units (§7.2) sit between those cells; six to eight in flight per drive should
  hold a 990 PRO at ≈ 5.6-5.8 GB/s and the Optane at ≈ 7 (model, interpolated). A 9 MiB record
  striped over `D:` + `G:` then lands in ≈ 0.75-0.85 ms, and its first gate/up units after
  ≈ 0.15-0.2 ms, so computation can start before the record is complete (§11.4).
- **Latency, not just bandwidth.** One record from one drive takes ≈ 1.6 ms; striped over two,
  ≈ 0.8 ms. That is about ten layers of GPU pre-routing work (§9.3), so every SSD read that is not
  predicted ahead stalls decode (§11.5).

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
| **ExLlamaV3** (turboderp, MIT; the EXL3 format's producer) | QTIP-style trellis codes in 16 × 16 tiles, codebooks `3inst`, `mcg` and `mul1`, 128-point Hadamard rotations with sign-scale vectors; GEMV kernels up to 144 rows, reconstruct-then-HGEMM above; MoE kernels including a cooperative fused gate-up-activation-down kernel; FP16 activations | The format definition this design imports; EXL3 3.05 bpw (base) measured mean KL 0.047 against BF16 in alesha-pro's table (§7.4) |
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

- **EXL3 experts can be computed on a desktop CPU at DRAM speed** (adrienbrault, AVX-512). The
  13900K has no AVX-512, so the CPU lane here is modelled compute-bound (§12.2) and M0 measures it
  first — on both machines' hosts.
- **Router near-miss scores as a cache signal** (ox-boost's soft cache) is evidence specific to
  this model; it enters M1 as a candidate VRAM policy beside LFRU (§10.2).
- **Decayed LFU lost on this model's drift-shaped locality** (ox-boost). The shared RAM tier ranks
  by decayed LFU (MoE §4.6); M1 re-runs that choice on GLM traces before freezing (§10.4).
- **MTP over PCIe-transferred misses was slower; over CPU-computed misses it paid** (1.74 tokens
  per step). The drafter here verifies only when the cost model predicts a gain (§16.3).
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
for Maya-M (RAM and link not stated). A Maya-M expert is ≈ 8.6 MB against this design's 9.4 MB,
and on the one table that measured both (alesha-pro's, §7.4) the default EXL3 build has half
Maya-M's KL for ≈ 10 % more bytes per expert.

**Measured on GLM-5.3-Flash by Maya, and what each decides here:**

| Maya finding | Evidence | Disposition |
|---|---|---|
| VRAM hit rate by capacity (one V100, IQ1_S experts): 44-65 % at 10.8 % of experts, 61-84 % at 21 %, 74-82 % at 31 %; 98 % at 70 % (two cards) | `2026-10-05-glm-single-gpu` | Basis of §5.2's hit-rate estimates |
| Policy replay: LRU 67 %, aged LFU 77.5 %, decayed frequency 78 %, a learned model 77 %, Belady 87 % | `PROGRESS.md`, `2026-10-06-glm-spec` | LFU-family candidates in M1 beside LFRU (§10.2); ox-boost found the opposite, so GLM traces decide |
| Admission threshold (`PROMOTE_MIN` 6 aged routes) stopped one-off experts churning the cache (decode 1.9× with the RAM tier) | `SETTINGS.md` | M5 sweeps an admission threshold (§10.2) |
| Background promotion (8 per token) and free-slot refill after a prompt (24 per token) | `SETTINGS.md` | The promotion pacing swept in M5 (§10.2) |
| Disk lookahead in decode: only ~40-45 % of disk misses predictable 1-4 layers ahead; exact but slower on one NVMe | `PROGRESS.md` | Lookahead off by default; adopted only if M5 shows a gain on two volumes (§11.5) |
| MTP on one GPU does not pay: consecutive tokens share ~1 of 8 experts per layer and none of their RAM pulls | `PROGRESS.md` | Confirms §16: `--spec mtp` off by default |
| Draft routing computes only the 2 best-scored missing experts (`MTP_KEEP` 2): better acceptance than skipping all, never slower | `SETTINGS.md` | Adopted in §10.5's residency-bounded draft routing |
| CPU lane: a persistent spinning pool (0.31-0.36 ms per ~7 MB expert); one thread per physical core, never hyperthreads (21.8 → 13.3 tok/s: they starve the disk readers); work cut into ~48 pieces claimed in turn (14.3 → 17.1 tok/s); the CPU/PCIe split measured per expert count and persisted | `PROGRESS.md`, `SETTINGS.md` | Adopted in §12.3 (piece claiming, per-count split table, no SMT siblings); the 0.31-0.36 ms figure is the closest published CPU cost to §12.2's E3A16 estimate |
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
protocol for speed; this design's default artifact measured 0.047 against BF16 on alesha-pro's).

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
| Quantizing the dense classes costs real KL, more at long context: int8-per-32 dense 0.0105 mean KL against 0.0066 for BF16 dense after a 2,600-token prompt; NVFP4 per class 0.016 (head) to 0.028 (MLA) | log, KL tables | §20.3's frozen texts include positions beyond 2,051 tokens and a long-prompt block; option O9 (§25.3) imports dense classes from the FP8 source if they carry most of the KL |
| A naive 3-bit expert test (sign + 2 bits, UE4M3 per 16, no rotation or trellis): long-context KL 0.19 | log (`GLM_FAKE3`) | A warning that 3-bit experts can fail at long context: §20.3 runs the long-context protocol on the default artifact |
| Static VRAM hot set + LRU remainder: 15.66 → 17.15-17.31 tok/s; disk reads 30.6 → 22.2 per token | README, log | M1 candidate (§10.2) |
| Exclusive RAM/VRAM residency cost 158 write-backs per token (rejected there) | log | This design is exclusive with shadows (MoE §4.1); demotions are capped at 32 per boundary and M5 reports the copies (§10.1) |
| Mirrored volumes with dynamic per-unit routing: each 1.4 MB read to the drive that frees first; best static balance 10:7 on unequal drives | README, log | Option O8 (§25.3), A/B'd against weighted striping |
| Next-layer predictor from the current layer's streams: top-1/2/3 accuracy 95/86/77 %; no isolated gain published | README, log | Corroborates the rank-limited prefetch forms in M5 (§11.5) |
| CPU share of RAM misses: slower on a 50.8 GB/s link | log | Consistent with the divisor rule (MoE §5.6): the CPU takes fewer misses as the link gets faster; here the link is x8 |
| Opt-in skipping by routing weight: +30 % at KL 0.022 (≈ 3 % threshold, SSD-only); RAM misses below 8 %: +20 % (KL 0.044) | README, log | §9.8 gains a weight-threshold rule beside Maya's rank map; M7a compares them at equal KL |
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
| GPU HBM: dense EXL3 + KDA state + KV + mHC | 5.45 + 0.29 + ~0.05 GB | ≈ 1.5 TB/s (5090, estimate, FN §8.10) / 1.79 TB/s (production, measured, FN §8.3) | ~3.7 ms (5090; §9.3's budget is 4.1 ms with launches) / ~3.2 ms (production, model) |
| GPU HBM: hit experts | 336 h_v × 9.47 MB | same | 2.1 h_v ms (5090) |
| CPU lane: RAM- and SSD-served experts it computes | 0.2-0.4 ms per expert (E3A16 decode-bound, §12.2) | 6-8 P-cores (+ E-cores) on the 5090 machine; unmeasured on the production machine | |
| PCIe H2D: DMA'd misses (≈ 1/2 of RAM misses, §9.6) and SSD landings sent to the GPU | 9.47 MB each | 27.6 GB/s (5090 machine, measured) / 50 GB/s (production, model) | 0.34 / 0.19 ms each |
| Host DRAM: CPU reads + DMA source + SSD landings | ≈ 336 (h_r + h_s) × 9.44 MB | ~80 GB/s on the 5090 machine (measured), shared | not binding at the central case |
| SSD: 336 h_s records | 336 h_s × 9.44 MB | 12.7 GB/s (`D:` + `G:`, measured) on the 5090 machine; none at the production machine's default placement | 250 h_s ms (5090 machine) |

**The binding term is the host service of RAM misses: the CPU lane's decode compute and the link,
in parallel.** The SSD term is second: on the 5090 machine at the central h_s = 3 % it is ≈ 10
reads per token; on the production machine it is zero at the default placement. The GPU term is
~4-6 ms on the 5090 and ~3.5-5 ms on the production card.

### 5.2 Hit-rate model

No routing traces of this model are public, and M1 records them before the policy is frozen. The
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

**RTX 5090 machine** (estimate from that evidence, at §19's capacities: ≈ 2,600 frames = 21 %,
≈ 8,000 RAM slots = 66 %, together 87 % of the 12,096 experts, exclusive):

| Scenario | h_v | h_r | h_s | RAM-served per token | SSD reads per token |
|---|---:|---:|---:|---:|---:|
| Optimistic (policy and seeding gains of §10 land) | 0.72 | 0.265 | 0.015 | 89 | 5.0 |
| Central | 0.66 | 0.31 | 0.03 | 104 | 10.1 |
| Pessimistic | 0.60 | 0.35 | 0.05 | 118 | 16.8 |

**Production machine (estimate):** the frame pool holds ≈ 80 % of the bank and the RAM tier the
rest (§19.1), so h_s = 0; h_v ≈ 0.80-0.88, central 0.85 (interpolated between Maya's 70 % → 98 %
with two cards and ox-boost's 47 % → 90 %), h_r the rest.

The 5090 machine's SSD carries 0.05-0.16 GB per token: 4-12 ms of drive time on two drives. Two
volumes are worth 1.2-1.3× (§5.3), less than they were worth for 13.5 MiB NVFP4 records, because
more of the bank fits in memory.

### 5.3 Decode projection

A layer's misses are known only after its router, and the next layer needs its combine, so per MoE
layer the critical path is

```text
t_layer = t_pre + max(t_hits, t_host)
t_host  = min over n_c of max((n_c + k) · c_cpu,  (n_r − n_c) · c_dma,  k · t_record + c_cpu) + 30 µs
t_token = Σ_l t_layer + t_dense_layers + t_head
```

with the layer's n_r RAM misses and k SSD misses multinomial over its 8 experts (exact
expectation), n_c of the RAM misses computed by the CPU lane and the rest DMA'd, and SSD records
computed by the CPU as they land.

- t_pre: the layer's GPU work before routing (mHC, attention, mixer, router): ≈ 3.4 ms over the
  42 MoE layers on the 5090 machine at ≈ 1.5 TB/s (estimate, §3.1; §9.3); ≈ 2.9 ms on the
  production card (model, scaled by the measured 1.79 TB/s). Dense layers 0-2 and the head add
  ≈ 0.7 / 0.6 ms. Hit experts (≈ 6 µs each) hide under the host service except in layers with
  no miss.
- c_cpu: the CPU lane's time per expert, **estimated 0.22 / 0.30 / 0.40 ms** (optimistic,
  central, pessimistic: with E-cores helping, 8 P-cores, 6 P-cores; §12.2, i9-13900K). This is the
  model's least certain input and M0 measures it first — on both hosts (the production machine's
  CPU is unrecorded in the repo).
- c_dma = 9.47 MB / 27.6 GB/s = 0.34 ms per staged expert on the 5090 machine; 0.19 ms on the
  production machine's modelled x16 link.
- t_record: a whole record striped over the volumes: ≈ 1.6 ms on one drive, 0.8 ms on two,
  0.55 ms on three with `C:` (model from §3.3).
- Lookahead is off (the default, §11.5). With ≈ 10 SSD reads per token its ceiling is the gap to
  the no-SSD column below, of which Maya's 40-45 % predictability could recover part.

M0 commits the model as `tools/glm5_next/decode_model.py` with its inputs.

Plain decode, tok/s (model):

| Scenario, RTX 5090 machine | One volume | **Two volumes** (`D:` + `G:`) | Three volumes (with `C:`) | No SSD misses (whole bank in memory) |
|---|---:|---:|---:|---:|
| Optimistic | 36.0 | **42.2** | 44.5 | 48.0 |
| Central | 24.8 | **31.1** | 33.6 | 37.2 |
| Pessimistic | 18.0 | **23.7** | 26.1 | 29.5 |

Central, two volumes, by CPU cost per expert: 0.15 ms → 39.5 tok/s, 0.20 → 35.7, 0.30 → 31.1,
0.40 → 28.1, 0.60 → 24.5. **The CPU E3A16 kernel is worth as much as the second drive.**

**Production machine (model, no SSD term):** with h_v = 0.85 / 0.80 / 0.75 (optimistic /
central / pessimistic), c_cpu 0.22 / 0.30 / 0.40 ms and c_dma 0.19 ms, the same formula gives
≈ 114 / 80 / 46 tok/s — central ≈ 80. The wide range is the unmeasured CPU cost; M0 narrows it.

**Calibration against the one comparable published run.** Fed strata-glm's measured setup (§4.3),
the model predicts 15-45 % more than that engine achieved. If this design lost as much to overheads
the model does not carry, two-volume decode on the 5090 machine would be ≈ 21-27 tok/s central
rather than 31, below the Must target at the low end. The targets stay as set; M4's first
end-to-end numbers decide whether the gap belongs to strata-glm's implementation or to the model,
and §25.1 carries it as a risk. For comparison, the best public GLM-5.3-Flash numbers are 22-38
tok/s (§4.1) on a 96 GB card with every other expert in pinned RAM over x16, 65 tok/s on two 5090s
with a lossier 2.05 bpw EXL3 build, and Maya's "30+" on an RTX 5090 with 8.6 MB experts (§4.2).

**Speculation helps little.** MTP amortizes the GPU's dense reads over accepted tokens, but
RAM- and SSD-served expert bytes per accepted token stay about the same (MoE §6.3), and every
rejected draft adds misses. Both public GLM-5.3-Flash MTP experiments over PCIe measured slower
(−13 %, §4.1); the positive result (1.74 tokens per step) computed its misses on the CPU, which
this design does for about half of them. The draft-length policy (§16.3) therefore includes K = 0
and is expected to choose it whenever acceptance is low.

### 5.4 Prefill projection

Prefill runs layer-major over a **walk span** of up to 64K tokens (MoE §7.1, §17.1): every
non-VRAM expert crosses the link once per span, while the GPU computes the previous layer.

| Term, 32K-token span | Work | Rate | Time |
|---|---|---|---:|
| Expert stream over the link (5090 machine; 79 % of the bank non-resident) | 0.79 × 114.6 GB = 90.5 GB | 27.6 GB/s measured | 3.3 s |
| Same, production machine (20 % non-resident, §19.1) | 0.20 × 114.6 GB = 23 GB | 50 GB/s model | 0.46 s |
| Dense GEMMs, E3A16 on int8 tensor cores (four products per MAC) | 2 × 8.23 B × 32,768 = 540 T ops | 105-150 TOPS on the 5090 (estimated: a quarter of the card's dense INT8 peak of ≈ 840 TOPS by vendor specification, unverified here, at 50-70 % efficiency); × 188/170 SMs on the production card (model) | 3.6-5.1 s (5090); 3.1-4.4 s (production) |
| Routed experts, E3A16 wide route | 554 T ops | same | 3.7-5.3 s (5090); 3.2-4.6 s (production) |
| MLA, indexer, KDA chunks | | | ~0.5 s |

Streaming and compute overlap, so a 32K prompt is compute-bound on both machines: the 5090 ≈
max(3.3, 7.3-10.4) s ≈ 8-11 s (**≈ 3,000-4,200 tok/s**, central 3,500); the production card ≈
max(0.5, 6.3-9.0) s ≈ 6.3-9.0 s (**≈ 3,400-4,900 tok/s**, model). The two gated permissions of
§17.4 (8-bit rotated activations for dense and wide-route experts: two products per MAC) halve
the compute: 32K ≈ 5,700-7,800 tok/s on the 5090 machine. A **4K prompt still touches every
expert** (each sees ~114 tokens), so its time to first token is a whole expert pass: ≈ 3.4-4.5 s,
central 3.7, on the 5090 machine; ≈ 1.4-1.8 s on the production machine (model). Only the prefix
cache (§18) and a faster link change that.

### 5.5 What moves the ceiling

| Lever | Effect (model, central) |
|---|---|
| **The 5090 machine at PCIe x16** (move the NIC to a chipset slot or remove it) | DMA'd misses at 2× the rate: decode 31 → 37 tok/s (+18 %); every prefill stream halves (TTFT 4K ≈ 2 s) |
| **A 192 GB host on either machine** (4 × 48 GB) | The whole bank fits in RAM: h_s = 0 on the 5090 machine, no SSD term; decode 31 → 37 tok/s; with x16 too, 50 |
| **A third expert volume on CPU lanes** (5090 machine, `C:`) | Record time 0.8 → 0.55 ms (measured bandwidth, §3.3); decode +8 % |
| **A CPU with AVX-512 or more cores** | The CPU lane's cost per expert falls (adrienbrault's Zen 5 reaches DRAM speed); 0.30 → 0.20 ms is +15 % on the 5090 machine |
| **Faster DDR5** | Little until the CPU lane is DRAM-bound |

The design reports these as hardware advice in the user guide; nothing is tuned to x8.

---

## 6. Design overview

```text
┌────────────────────── one GPU: RTX PRO 6000 96 GB (or RTX 5090 32 GB) ──────────────────────┐
 │ dense EXL3 5.1 GiB · KDA state 136 MiB/lane · MLA latents + pooled index keys              │
 │ decode workspace · mailboxes · residency tables (MoE §4.1)                                 │
 │ ┌─ frame pool (MoE §4.2): ≈ 2,600 (5090) or ≈ 9,500 (96 GB) × (9 MiB + 36 KiB), elastic ┐  │
 │ │ cached experts · staging · landing frames (MoE §4.4) · MTP pool · leases (MoE §4.5)   │ │
 │ └───────────────────────────────────────────────────────────────────────────────────────┘ │
 └──────▲───────────────────▲───────────────────────────▲────────────────────────────────────┘
        │ copy engine       │ mapped mailboxes          │ copy engine (landed SSD records)
 ┌──────┴───────────────────┴───────────────────────────┴───────────┐   ┌──────────────────┐
 │ host RAM (89 GB production / 96 GB on the 5090 machine)          │   │ expert volume 0  │
 │  RAM tier (MoE §4.6): pinned slots × 9 MiB + the landing ring    │◄──┤ (volume i of 1-4, │
 │  expert vector plane (suh/svh of every expert, 0.43 GiB)         │   │  one per drive)   │
 │  CPU expert team: E3A16 exact, from RAM or landing slots (§12)    │   └──────────────────┘
 │  engine thread: policy at round boundaries, promotions           │
 │  SSD read agent (MoE §4.7): one unbuffered queue per volume      │
 └───────────────────────────────────────────────────────────────────┘
```

One decode round at C = 1:

```text
embed(host row) → 4 streams
per layer l (45):
  mHC_attn (one kernel) → RMSNorm → KDA (fused proj, conv, gates, recurrence, gated norm, o_proj)
                                     or MLA (q/kv latents, indexer, top-512, latent attention, o_proj)
  mHC post + mHC_ffn (one kernel) → RMSNorm → MLP:
     layers 0-2: dense SwiGLU
     layers 3-44: router(l) + predictors(l+1, l+2, l+3) → residency → GPU jobs | CPU jobs | SSD jobs
                  shared expert ‖ routed experts (narrow E3A16) ‖ CPU misses ‖ landed SSD records
                  combine (FP32, fixed order) → mHC post
head: mean of streams → norm → lm_head (FP32 logits) → sample → MTP drafter (optional)
```

The ideas that carry the design:

- **Exact expert arithmetic makes placement free** (MoE §2): VRAM frame, staging frame, CPU from a
  RAM slot, CPU from a landing slot: the same bits. Policy can move work anywhere. For EXL3 this
  needs a new canonical arithmetic (E3A16, §9.5), since ExLlamaV3's FP16 kernels are not
  reproducible across processors or routes.
- **The tiered residency, placement and runtime switch are the shared tier's** (MoE §4): the
  engine thread applies the policies at round boundaries, the device only reads residency words,
  and any VRAM : RAM : SSD ratio is reachable at runtime.
- **The SSD tier is a striped, read-only set of expert volumes**, one per drive, written once by
  the converter — this family's composition of the shared store (§7.3, §11); a record's twelve
  0.75 MiB units are spread across the drives, so every read uses every drive.
- **SSD reads are made short, not guessed**: striping halves a record's latency, and partial
  records are consumed as their units land. Lookahead reads of predicted experts are designed
  (§11.5) but off by default, since the one GLM measurement found them a net loss on one drive.
- **The host's service rate is the scarce resource**: everything is sized to minimize host-served
  expert work per accepted token: frames maximized, KV small, MTP experts bounded, promotions
  rationed (§10), and the CPU lane's decode kernel treated as a first-class performance target
  (§12).

---

## 7. Artifacts, import and the striped expert volumes

### 7.1 The two artifacts

Read 2026-10-11 from both repositories' `quantization_config.json` and `model.safetensors.index.json`
(metadata only; the weights were not downloaded). Shared by both: `quant_method` `exl3`, `bits`
3.05, `head_bits` 6, codebook `mul1` with multiplier 2,212,286,765 (0x83DCD12D) on every matrix,
`out_scales: always`, calibration 250 rows × 2,048 columns, and identical tensor shapes and dtypes
for every module both contain.

| Tensor class | Stored (both builds) | Bytes | Engine form |
|---|---|---:|---|
| Routed experts, 42 × 288 | EXL3 3-bit: int16 trellis [K/16, N/16, 48] per matrix, FP16 `suh` [K] and `svh` [N], int32 `mul1` | 114.60 GB | Trellis tiles relaid out (bits verbatim) as `exl3_mul1_expert_v1` (§7.2) into the expert volumes (§7.3); `suh`/`svh` into the **expert vector plane** (a Model tensor, pinned host memory) |
| MTP routed experts, 288 (MikeRoz only) | EXL3 4-bit (`mtp_bits` 4) | ≈ 3.6 GB | Same layout at K = 4, as the MTP bank of the volumes |
| KDA `qkv_proj` (fused q, k, v), `o_proj` | EXL3 5-bit | 2.85 GB | `exl3` Linear (§9.5), resident |
| KDA f_a, f_b, g_a, g_b, b; conv weights; `A_log`, `dt_bias` | FP16; BF16; FP32 | 0.24 GB | Direct |
| MLA q_a, q_b, kv_a, kv_b, o | EXL3 5-bit | 0.69 GB | `exl3` Linear |
| Indexer `wq_b` / `wk`, `k_norm`, `weights_proj`, compress gate, APE | EXL3 5-bit / FP16 and BF16 | 0.06 GB | `exl3` Linear / direct. turboderp's build lacks the compress gate and APE (§7.3) |
| Shared experts | EXL3 5-bit | 0.66 GB | `exl3` Linear |
| Dense MLP layers 0-2 | EXL3 4-bit | 0.23 GB | `exl3` Linear |
| `lm_head` | EXL3 6-bit | 0.48 GB | `exl3` Linear, FP32 logits |
| Router weight / `e_score_correction_bias` | FP16 (ExLlamaV3 converted it from the source's BF16) / FP32 | 0.10 GB | Direct; FP32 logits. M0 counts the router values the conversion changed |
| mHC `fn`, `base`, `scale`; all norms | FP32; BF16 | 0.14 GB | Direct |
| Token embedding | BF16 | 1.27 GB | Direct, **host-resident** (§19.2) |
| MTP dense: MLA block, `eh_proj`, shared expert (MikeRoz only) | EXL3 4-bit (`mtp_bits`) + small tensors | ≈ 0.2 GB | `exl3` Linear |
| Vision tower | EXL3 6-bit (`vision_bits`) + BF16 | ≈ 0.97 GB | Stored, not loaded (§1.1) |

How the two builds differ, and what the import does about each:

| Difference | MikeRoz `…-Uncensored-3.05bpw-h6-exl3` | turboderp `GLM-5.3-Flash-exl3` @ `3.05bpw` | Import |
|---|---|---|---|
| Weights | orcarouter's refusal edit of the FP8 release | the base model | Both imported as stored |
| Quantizer | ExLlamaV3 1.4.6 | ExLlamaV3 1.4.4 | Same format variant; M0 checks the two versions' tile order and Hadamard scale are the same |
| MTP layer | present (4-bit experts) | absent | turboderp's artifact has no MTP bank; `--spec mtp` is refused for it |
| Indexer pooling tensors (`index_kpool_compress_gate`, `_ape`, 11 MLA layers) | present | **absent**, although its config enables pooling and current ExLlamaV3 loads them as required | Supplemented from `zai-org/GLM-5.3-Flash` (§7.3) |
| Whole artifact | 126.11 GB | 122.29 GB | |

The converter refuses any EXL3 artifact outside this variant (another codebook, a half-integer bit
rate on a routed matrix, unequal gate/up and down bits within a bank, mixed multipliers) with the
property named, until a track adds it (§25.3).

### 7.2 Recipe, expert record layout and the vector plane

**One recipe**, `glm5_3_flash_exl3`, imports an EXL3 artifact as stored. Two official entries pin
the defaults: `glm5_3_flash_uncensored_exl3_3_05` (MikeRoz, `main`) and `glm5_3_flash_exl3_3_05`
(turboderp, `3.05bpw`, with the supplement); M0 pins both revisions. No dense recipe variants
exist: the producers' bit allocation is imported unchanged.

**Record layout `exl3_mul1_expert_v1`** (to be defined byte-exactly in [storage
layouts](storage-layouts.md) when it lands). The record principles — one expert one contiguous
record read by every consumer, formats as kernel traits with the exact decode shared by CPU and
GPU, a bank's records a fixed stride apart — are the shared tier's (MoE §3); the layout is this
family's instance of them. EXL3's output Hadamard mixes 128 output channels, so a work item covers
128 outputs: eight 16-column tile strips. The converter regroups the stored tiles (each 16 × 16
tile's 3K × 16-bit words copied verbatim, so no bit changes) by output block:

| Section | Work unit | Content of one unit of work | Bytes per unit of work (K = 3) | Units of work | Bytes |
|---|---|---|---:|---:|---:|
| gate/up | **slice**: 128 intermediate channels | gate strips then up strips, 8 each, each strip's 256 K-tiles in ascending k, 96 B per tile | 393,216 | 16 | 6,291,456 |
| down | **block**: 128 hidden rows | 8 strips × 128 K-tiles × 96 B | 98,304 | 32 | 3,145,728 |
| **Record** | | | | | **9,437,184 = 9 MiB = 2,304 × 4 KiB** |

The MTP bank (K = 4: 128 B per tile) has the same structure at 12,582,912 B (12 MiB) per record. The
layout id carries no geometry: H, I and K are properties of the bank, recorded in its binding (§8.3).

**The expert vector plane.** Each expert's six sign-scale vectors (gate and up `suh` 4,096 and `svh`
2,048, down `suh` 2,048 and `svh` 4,096, all FP16: 36,864 B) are kept out of the record, so
records, RAM slots and stripe units are exact powers of two and every unit holds only trellis
words. The plane holds all 12,384 experts' vectors (456.5 MB) as one Model tensor in pinned host
memory: the CPU reads it in place, and a promotion copies a record's 9 MiB and its 36 KiB plane
entry to a frame in one batched copy. A VRAM frame is therefore 9 MiB plus a 36 KiB side entry —
the frame side entry the shared tier gains for this family (§8.2; the geometry stays three
integers, the side entry a pitch argument that is zero for `qwen4_exp`). The codebook multiplier is
one constant per bank, checked uniform at import.

### 7.3 Import and the striped expert volumes

**Supplementing turboderp's build.** The 22 missing pooling tensors are BF16 source tensors (≈ 11 MB
together). The converter fetches exactly those tensors from a pinned revision of
`zai-org/GLM-5.3-Flash` with HTTP range reads of the safetensors shards (each shard's header gives
the byte ranges; the same mechanism as Flash-Next's MTP fetch, FN §6.1), checks their shapes
against the config, and records the source and revision in the conversion report. No whole shard is
downloaded.

The routed banks do not live in the `.infernix` file set. They live in **N expert volumes**
(`<name>.xv-<i>-of-<N>`, 1 ≤ N ≤ 4), each a plain file meant for its own drive, in the role the
n-gram volume has for Flash-Next (FN §6.4). The container's continuation volumes must sit beside
the entry file ([container spec](artifact-container.md) §2.1, §3.3), so they cannot be spread over
drives; the expert volumes are located by flag instead (§22).

**Stripe unit u_b = record_b / 12** for bank b: 786,432 B (0.75 MiB = 192 × 4 KiB) for the main banks,
1,048,576 B for the MTP bank. A record is exactly 12 units. Units 0-7 are the gate/up section (each
holds exactly 2 slices) and units 8-11 the down section (each holds exactly 8 blocks). Twelve
divides evenly among 1, 2, 3, 4 and 6 volumes. Equal units need equal gate/up and down bits within a
bank, which both artifacts have; a mixed-bit build needs two unit sizes and a per-volume table of
12 offsets (§25.3).

**Placement.** A weight vector w (integers, Σ w_v = 12, chosen at conversion; default equal) gives
each volume w_v units of every record. Unit j of record r goes to volume P[(j + r) mod 12], where P
is the fixed sequence that repeats each volume index w_v times, interleaved (for w = (6, 6):
0, 1, 0, 1, …; for (5, 7): 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0). Rotating by r spreads the first and
last units over the drives. Within volume v, bank b starts at
`base_{v,b} = 4096 + Σ_{b' < b} R_{b'} · w_v · u_{b'}`, and record r of the bank stores its units
contiguously in increasing j, its k-th at `base_{v,b} + (r · w_v + k) · u_b`. Records are numbered bank
by bank (the 42 main banks, then the MTP bank), expert by expert. Every offset is computable; there is
no per-record table.

**Header** (first 4 KiB of each volume): magic `NINFXV\0\1`, the artifact set's 16-byte id, volume
index and count, weight vector, bank count, and per bank its record bytes, stripe unit and record
count, plus a CRC32C of the header. The main artifact records the same geometry and the expected size
of each volume, so a volume from another artifact, another weight vector or a truncated copy is
refused at startup.

**Sizes.** MikeRoz: 12,096 main records (114.15 GB) + 288 MTP records (3.62 GB) = 117.78 GB; with
w = (6, 6) each volume is 58.9 GB, with (5, 7) 49.1 and 68.7 GB. turboderp: 114.15 GB; 57.1 GB each, or
47.6 and 66.6 GB.

**Writing.** The converter writes the main artifact and all volumes in one run
(`--expert-volumes PATH,PATH,…`, `--expert-volume-weights 6,6`), streaming each source shard once and
writing each volume sequentially with unbuffered 64 MiB writes. `python -m tools.artifact.restripe`
rewrites the volumes for another N or weight vector from existing volumes, without the source. These
two are the only writers; runtime never writes a volume (MoE §4.7's rule: the SSD tier is the
artifact, read in place, never an SSD cache or spill). The one-time write is 118 GB, ≈ 0.01 %
of a 990 PRO 2 TB's rated 1,200 TBW.

**How the volumes meet the shared SSD tier.** The tier today reads the artifact in place through
`ExpertStore`'s file segments (one per record, two when a record straddles the container's part
files, plus an optional head/tail **expert stripe** that puts every record's tail on a second drive —
`src/models/offload/expert_store.h`, `tools/flash_next/expert_stripe.py`, MoE §3 and §4.7). The
volumes generalize the stripe: N drives instead of two, 12 equal units instead of a head/tail split.
What the tier gains for this family, and nothing else changes (§8.2): `ExpertStore`'s segment count
per record rises from 3 to 12 (a static array; the units are 4 KiB-aligned and adjacent-unit
coalescing is unchanged), and the read agent owns one `DirectReadQueue` per volume with per-drive
depth caps (§11.3) — one volume today, so `qwen4_exp`'s behavior is byte-identical.

**Disk plan, 5090 machine.** The source (126.1 GB, MikeRoz) on `E:` (371 GB free); the main artifact
(≈ 8.3 GB: dense 5.45, embedding 1.27, vector plane 0.46, MTP dense 0.2, vision 0.97) on `E:`;
volume 0 on `D:` (144 GB free) and volume 1 on `G:` (128 GB free) with w = (5, 7) (§11.6). The source
can be deleted after `verify_artifact` passes; the FP8 teacher used by M4's quality report (§20.3)
fits on `E:` only after that. The production machine has one NVMe (the T710): there the volumes
default to one file, and striping is a configuration only a second drive unlocks (§3.2).

### 7.4 Other artifacts and the quant survey

A survey of GLM-5.3-Flash quants with a published KL (owner request, 2026-10-11) made the EXL3
3.05 bpw builds the default. Hugging Face lists ~440 GLM-5.3-Flash repositories; these are the ones
with a published KL (read 2026-10-11 from their cards; none reproduced here).

**KL is only comparable within one table.** Each card uses its own reference (BF16, FP8 or a Q8_0
conversion), texts, top-k truncation and engine. The clearest evidence is one table that measured
several formats against the same BF16 teacher logits (alesha-pro, `GLM-5.3-Flash-abliterated-Maya-M-GGUF`):
Maya-M scores 0.093 there, against 0.29-0.33 on Maya's own, harsher protocol.

| Quant (repository) | Size | Format | KL (reference, source) |
|---|---:|---|---|
| **EXL3 3.05 bpw** (`turboderp/GLM-5.3-Flash-exl3`; **the default, with MikeRoz's uncensored build**) | 122.3 GB | ExLlamaV3 trellis (QTIP-style) | **0.047** vs BF16, top-1 93.1 % (alesha-pro table); the uncensored build is unmeasured |
| Maya-M (`peasantsmith/GLM-5.3-Flash-Maya-GGUF`) | 116.0 GB | GGUF IQ2_S / IQ3_XXS experts | 0.093 vs BF16, top-1 90.7 % (same table); 0.29-0.33 vs FP8 (Maya's) |
| EXL3 2.4 bpw mix (gate/up 2, down 3 bits) | 98.1 GB | EXL3 | 0.096 vs BF16 (same table) |
| EXL3 2.05 bpw | 85.1 GB | EXL3 | 0.121 vs BF16 (same table) |
| **GSQ-RCO 3.5-bit** (`pfeifferj/GLM-5.3-Flash-GSQ-RCO-GGUF`, IST-DASLab methods, community reproduction) | 137.1 GB | GGUF K-quants: Q3_K mostly, Q2_K, Q4_K experts; Q8_0 dense | 0.071 vs Q8_0-of-FP8 (approximate); perplexity +4.5 %; MMLU-Pro −1.4 pp |
| GSQ-RCO 3.0-bit (same) | 117.5 GB | GGUF, mostly Q2_K experts | 0.142 vs Q8_0; perplexity +9.1 %; MMLU-Pro −2.0 pp |
| AesSedai IQ4_XS / Q4_K_M / IQ3_S (`AesSedai/GLM-5.3-Flash-GGUF`) | 159 / 202 / 125 GB | GGUF | 0.117 / 0.050 / 0.283 (own reference) |
| TheDrainFlorist VQ 3.6 / 3.1 / 2.7 bpw | 144 / 125 / 109 GB | MLX vector quantization (Apple) | 0.095 / 0.200 / 0.294 vs BF16, prose only |
| Sushi 2.4 bpw (`beamster`) | 97.8 GB | sushi (Apple only) | 0.074 vs BF16 in sushi's own forward |
| EXL3 3.0 bpw selective (`0xSero`) | 149.6 GB | EXL3 + source-precision dense | 0.153 vs BF16 |
| NestQuant 1.5-4 bit (`jarrelscy`) | 59.5 + 28.6 GB floating | research format with SSD-swapped residual planes | 0.069 (research harness, not reproduced by its author) |
| REAP pruned (`OpenMOSE` 250B and others) | 265 GB FP8 … | expert pruning | 0.45 and worse: pruning costs far more than quantization |
| orcarouter NVFP4 (an optional later recipe, §25.3 O7) | ≈ 205 GB | NVFP4 experts, BF16 dense | 0.073 vs FP8 (card; WikiText) |

**The abliteration is available at source precision.** orcarouter publishes
`orcarouter/GLM-5.3-Flash-Uncensored-FP8`, the same refusal edit baked into the vendor's block-FP8
checkpoint, and MikeRoz quantized exactly that with EXL3:
`MikeRoz/GLM-5.3-Flash-Uncensored-{2.51,3.05,4.05}bpw-h6-exl3` (98 / 118 / 154 GiB; no KL published).
GCSA-AiLab's uncensored GSQ-RCO "Q4" uses another edit (a rank-1 LoRA on the expert `down_proj`)
requantized into the GSQ-RCO type allocation, likely without GSQ's learned grids; it publishes no KL.

**Expected decode, 5090 machine** (§5.3's model, two volumes, today's RAM; the production machine has
no SSD term at the default placement):

| Expert format | Bytes per expert | Plain decode, central (range) | Model |
|---|---:|---:|---|
| orcarouter NVFP4 | 14.2 MB | 17.2 (13.7-21.8) | §5.3's per-layer model |
| **EXL3 3.05 bpw (default)** | **9.47 MB** | **31.1 (23.7-42.2)** | same |
| GSQ-RCO 3.5-bit | ≈ 10.5 MB | ≈ 31 | continuous-model estimate, not re-run |
| Maya-M | ≈ 8.6 MB | ≈ 48 | same |

The GSQ-RCO and Maya-M figures are continuous-model estimates and are likely high by the same margin
the per-layer model found for EXL3 (§5.3's calibration).

**Other artifacts on the same code path:**

- **MikeRoz's 2.51 and 4.05 bpw builds** and turboderp's other branches use the same importer and
  kernels where their routed banks have equal gate/up and down bits (kernels are instantiated for
  K ∈ {2, 3, 4} in experts and {3, 4, 5, 6} in dense classes); M0 reads their allocations. A build
  with mixed expert bits (such as the 2.4 bpw mix) needs two unit sizes per record (§25.3).
- **orcarouter NVFP4** (option O7, a later recipe): reuses Flash-Next's NVFP4 expert layout and
  exact W4A16 arithmetic at GLM geometry (FN §6.3, §17.3), 13.5 MiB records in 1.125 MiB units, and
  BF16 or 8-bit dense classes from its own BF16 tensors. Everything else in this design is
  geometry-generic and serves it unchanged. It is ≈ 0.55× the default's decode on the 5090 machine.
- **K-quant experts, GSQ-RCO style** from the uncensored FP8: a third expert format (Q2_K, Q3_K, Q4_K
  with Q8_K-blocked activations as their exact canonical arithmetic), plus a GSQ/RCO quantization
  pipeline. Not planned; the default already has a lower published KL at a smaller size.

---

## 8. Ownership in the engine

### 8.1 New family

This is an **explicit product change**: a new mathematical architecture
(`Glm5NextForConditionalGeneration`, model type `glm5_next`) and a new weight format family (EXL3).
A family gets its own directory under `src/models/`, started by copying the closest existing family
and specialized (AGENTS.md; MoE §9's checklist). The layout follows the existing families:

| Component | Owner | Location |
|---|---|---|
| Config interpretation (exact key set, refusals), mathematics, binding, block order, MTP alignment | Model | `src/models/glm5_next/` (`config.*`, `weights.h`, `load.*`, `model.h`, `instance.*`, `memory_plan.*`, `options.*`, `execution/` forward and parameters, `shapes/`, `frontend/`) |
| Program: lanes, KDA state pool, MLA latent KV and index keys, prefix-cache binding, MTP drafter, graphs, offload-tier composition | Program | `src/models/glm5_next/program/` |
| Engine instance | Model | `src/models/glm5_next/instance.{h,cpp}`; a `Glm5Next` value beside the three in `src/models/registry.{h,cpp}`; a `Glm5NextCore` arm in `src/runtime/engine/engine.cpp` beside `ModelInstance`'s and `Qwen4ExpInstance`'s |
| EXL3 codec: tile decode, codebook, 128-point Hadamard, E3A16 block quantization | Ops (common) | the exact decode in `src/ops/common/canonical_math.h` (one header, CPU and GPU), shared by the Linear and MoE Ops and the CPU kernels; the scalar form is the exact oracle |
| Tokenizer pre-tokenizer, chat template, tool-call and reasoning parsing | Frontend | §21 |
| Recipe, EXL3 import, supplement fetch, expert-volume writer, restripe, verification, reference forwards | Converter / tools | `tools/convert/glm5_next.py`, `tools/convert/formats/exl3.py`, `official_recipes.py`, `tools/artifact/expert_volume.py`, `tools/artifact/restripe.py`, `tools/glm5_next/` |

ExLlamaV3 is MIT-licensed; any kernel code ported from it keeps its copyright notice.

### 8.2 The shared offload tier, and what it gains here

The offload machinery is no longer inside a family: it lives in `src/models/offload/` and is composed
by each family's Program — `expert_cache.*` (LFRU, `FramePool`, `CacheController`),
`expert_residency.*`, `expert_placement.*`, `expert_store.*` (records as file segments, the expert
stripe), `host_tier.*` / `host_expert_tier.*` (the RAM tier, its read agent and tickets),
`expert_state.*`, `calibrated_order.*`, `route_trace.*`, `vram_monitor.*`, and `geometry.h`
(`ExpertBankGeometry`: banks, experts per bank, top-k). The composition contract — no base class, no
virtual model interface, the tier an execution resource of the MoE op, the Program driving
`before_round` / `after_round` / `tier_step` at its own round boundaries — is MoE §1 and §2;
Qwen3.8-Flash-Next runs on it in production (FN §7). The MoE op itself
(`include/infernix/ops/offloaded_sparse_moe.h`, `src/ops/offloaded_sparse_moe/`) carries the
residency, staging, CPU and fetch channels for every family.

What this family adds to that tier, each a zero-cost extension for `qwen4_exp`:

| Addition | Where | Why zero-cost for `qwen4_exp` |
|---|---|---|
| **Frame side entries**: a frame holds a record plus a per-expert side entry (EXL3's 36 KiB vectors) | `ExpertResidency` / `FramePool` (`src/models/offload/`): a side-entry pitch beside the record stride in the constructors, 0 by default | A defaulted pitch changes no allocation, table or kernel for a bank without one; `ExpertBankGeometry` stays three integers (MoE §3) |
| **12-segment striped records**: a record's 12 volume units as up to 12 file segments | `ExpertStore` (`src/models/offload/expert_store.h`): `kMaxSegments` 3 → 12; units are 4 KiB-aligned, coalescing unchanged | A larger static array per record; a one-volume layout still produces one segment per record, exactly today's description |
| **One read queue per volume, per-drive depth caps** | The tier's read agent (`host_expert_tier.*`) over `core/direct_read_queue.h`; today one queue serves every file, and each read already names its file | One volume today ⇒ one queue, identical behavior; the per-drive caps only change which reads are in flight together |
| **`exl3_mul1` expert format trait** | `src/ops/offloaded_sparse_moe/` (`cuda/`, `cpu/`) beside `nvfp4_mul`, the ggml and FP8 banks; the exact trellis decode in `src/ops/common/canonical_math.h` | A new trait adds kernels beside the existing ones (MoE §3); no existing route changes |
| **`noaux_tc` routing** (FP32 sigmoid, bias in selection only, normalized × 2.5) | A routing variant beside the softmax top-k of `moe_route` / `moe_route_dispatch` in `src/ops/offloaded_sparse_moe/` | A separate variant selected at plan time; the softmax route's bits and kernels are untouched |
| **Landed-unit masks** (per-unit completion of a striped record) | The fetch channel and read agent: each unit's completion publishes a bit, so CPU jobs and stage copies consume a record before it is whole (§11.4) | Whole-record landing remains the one-unit case; `qwen4_exp` fetches land exactly as today |

Frame leases, the VMM arena, the LFRU, the RAM tier's decayed-LFU ranking and pin lists, the host
RAM ledger, the Windows page-lock ceiling and unpinned records, the placement forms and the runtime
switch, the sizing laws and the CPU expert engine's team and protocol are taken unchanged (MoE
§4-§5); §10 and §12 record only this family's constants and candidates.

### 8.3 Op changes

| Op | Change |
|---|---|
| New dense Linear format **`exl3`** | E3A16 (§9.5) for K ∈ {3, 4, 5, 6}: decode route (T ≤ 8, CUDA-core integer) and prompt route (int8 tensor cores), bit-identical to each other; transposed application for the absorbed MLA query (§13.1); variants in `linear_swiglu` (with the SwiGLU clamp) and `projection_fp32` (FP32 logits for the head); registered in the artifact format and layout registries (`src/artifact/formats.*`, `layouts.*`) |
| `offloaded_sparse_moe` | The **`exl3_mul1`** bank beside `nvfp4_mul` and the ggml/FP8 banks (E3A16 on the GPU narrow and wide routes and in the CPU engine; geometry from the bank's H, I, K). The **`noaux_tc` routing** variant (§2.2: FP32 sigmoid, `e_score_correction_bias` in selection only, weights the unbiased scores normalized, × 2.5; ties by lower id). The shared expert **ungated** (the gate row is absent, as GLM stores none). The SwiGLU clamp limit (10) as a semantic input on every MLP class. Frame side entries and landed-unit masks (§8.2) |
| `kimi_delta_attention` | Unchanged mathematics — the existing Op (`src/ops/linear_attention/kimi_delta_attention/`, recurrent and chunked routes, qualified against an FP64 oracle, lower bound in [−5, 0]) gets its first consumer. New companions: `kda_input_prep` (conv + SiLU on q, k, v with state update, L2 norm, gate and β from the projection outputs), `kda_replay_record` / fold for speculative commit, mirroring the GDN replay records (`src/core/gdn_replay_records.*`) |
| New `mhc` | §15: the mixer (norm, 24 projections, gates, Sinkhorn, collapse) and the residual update, both fused with neighbours |
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
| CPU expert team | `CpuMissService`: service thread + workers (calibrated, §12.3) | E3A16 misses from RAM slots or landing slots (§12) |
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
  read it with acquire loads (§8.2, §11.4).

### 9.3 Per-layer kernel sequence and budget (T = 1)

Budget = bytes at the card's sustained rate + 0.5 µs ramp under PDL — the planning budget this
family's kernels are designed against (Flash-Next's built chain is measured directly instead;
FN §8.3). "Pre" = before routing. EXL3 bytes include their vectors.

**KDA layer (34 of 45)**

| # | Kernel | Reads | Budget |
|---|---|---:|---:|
| K1 | `mhc` mixer for attention: reduces the previous layer's partial sums (sum of squares, 24 dot products) in fixed order, gates, Sinkhorn (20 iterations, FP32, one warp), collapse, `input_layernorm` | R 32 KB + fn 1.6 MB | 2.5 µs |
| K2 | KDA projection: `qkv_proj` [24,576 × 4,096] EXL3 5-bit + f_a, g_a, b [320 × 4,096] FP16 in one launch (two weight streams) | 65.5 MB | 44 µs |
| K3 | f_b, g_b [8,192 × 128] FP16; epilogue `kda_input_prep`: conv state update, SiLU, L2 norm, log α, β | 4.2 MB | 3.5 µs |
| K4 | KDA recurrence (existing Op, recurrent route): 4 MiB FP32 state read and written | 8.4 MB | 6 µs |
| K5 | gated `o_norm` prologue + `o_proj` [4,096 × 8,192] 5-bit; epilogue: mHC post (R' = post ⊗ y + combᵀ R) and partial sums for K6 | 21 MB | 14 µs |
| K6 | `mhc` mixer for the MLP + `post_attention_layernorm` | 1.6 MB | 2.5 µs |
| | **Pre** | | **≈ 73 µs** |
| K7 | Router (§9.4): FP32 logits [288 × 4,096] for l and, from L2-warm weights, l+1..l+3 (only for prediction), noaux_tc top-8, residency classification, jobs, mailboxes | 2.4 MB + 7.1 MB | 9 µs |
| K8 | Routed gate/up for GPU jobs (E3A16 narrow) ‖ shared expert gate/up + clamp + SwiGLU [4,096 × 4,096] 5-bit | 8 h_v × 6.3 MB + 10.5 MB | 30 µs at h_v = 0.66 |
| K9 | Routed down + combine (FP32, ascending expert id) + CPU and landed outputs + mHC post + partial sums for the next K1 | 8 h_v × 3.2 MB + 5.2 MB | 15 µs + waits |

**MLA layer (11 of 45).** K2-K5 become:

| # | Kernel | Reads | Budget |
|---|---|---:|---:|
| K2m | q_a [1,536 × 4,096] and kv_a [512 × 4,096] 5-bit + indexer wk, weights_proj, compress gate [288 × 4,096] FP16/BF16; epilogue: RMSNorm of c_q and c_kv, LayerNorm of the index key | 7.6 MB | 5.5 µs |
| K3m | q_b [16,384 × 1,536] 5-bit and indexer wq_b [4,096 × 1,536] 5-bit; epilogue: absorbed query q'_h = W_uk,hᵀ q_h (W_uk half of kv_b, 5-bit, transposed route, §13.1) | 24.8 MB | 17 µs |
| K4m | latent append (paged, §13.5) + pool completion when a 4-token block closes; indexer scores over the pooled keys + exact top-512 radix select | 0.26 MB at 4K, 8.4 MB at 128K | 4-9 µs |
| K5m | sparse latent attention over ≤ 2,051 tokens, 64 heads × 512 (§13.2); W_uv,h per head (5-bit); `o_proj` [4,096 × 16,384] 5-bit with the mHC post epilogue | 2.1 MB + 47 MB | 33 µs |
| | **Pre** | | **≈ 67 µs at 4K** |

**Dense MLP layers 0-2:** gate/up [24,576 × 4,096] + down [4,096 × 12,288], 4-bit, clamp in the
SwiGLU epilogue: 75.5 MB, ≈ 51 µs each.

**Per token (GPU work only, h_v = 0.66, 5090 machine):** 31 KDA-MoE layers × 127 µs + 3 KDA-dense
layers × 124 µs + 11 MLA-MoE layers × 121 µs + head 320 µs (476 MB 6-bit `lm_head`, FP32 logits)
≈ **6.0 ms**, ≈ 510 launches. Of that, ≈ 3.4 ms is before routing (t_pre of §5.3). On the
production card the same bytes at 1.79 TB/s (measured, FN §8.3) scale the budget by 0.84.

**Decode ALU.** EXL3 costs ≈ 4-5 integer instructions per weight (§9.5). At 5 bits per weight,
streaming 1.5 TB/s needs ≈ 11 T integer operations per second, about a fifth of the card's
CUDA-core integer rate, so the budget assumes memory-bound kernels. That assumption is the main
risk to this table; M6 measures it, and ExLlamaV3's own GEMV kernels are the comparison.

**Rules**, kept from Flash-Next's built chain (FN §8.3): PDL so a kernel always runs (no
kernel-boundary idle), weights loaded before the PDL wait, evict-first weight streams and
evict-last activations, state and next-layer routers, deterministic fixed-order reductions, grids
in multiples of the SM count. One rule matters far more here: **stall-time warming**. A layer
waits ≈ 0.3-0.6 ms on its host-served experts (§5.3), longer than the whole next layer's weight
stream. While K9 waits, idle CTAs warm the next layer's K1-K3 weights (≈ 70 MB, within the 96 MB
L2) into L2 in reading order, the form the shared op's `MoeL2Warm` provides (§9.4); estimated
saving 20-40 µs per MoE layer, ≈ 1-1.5 ms per token — a larger share of a ~32 ms token than of
Flash-Next's ~5-11 ms round (FN §4.3) (M6 measures it).

### 9.4 Router and prediction (K7)

Phase 1 (all CTAs): warp-per-row FP32 GEMV of the 288 router rows of layer l (the logits must be
FP32: BF16 logits create ties, FN §17.7), and of layers l+1, l+2, l+3 for prediction (their
weights, 2.4 MB each, prefetched into L2 with evict-last during K5).

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

### 9.5 Expert kernels and the canonical E3A16 arithmetic

**The stored form (upstream, ExLlamaV3 `codebook.cuh`, `exl3_dq.cuh`, `hadamard_inner.cuh`, read
2026-10-11 at `master`).** An EXL3 matrix [K → N] is `y = svh ⊙ H_N( H_K(x ⊙ suh) · W′ )`, H a
block-diagonal Walsh-Hadamard transform of size 128 with a scale the caller supplies. W′ is stored
in 16 × 16 tiles of 256 weights and 256 · K bits; weight i of a tile takes the 16-bit window
starting at bit K · i of the tile's stream, read cyclically, as its trellis state. The `mul1`
codebook decodes a state as `s = bytesum((state · 0x83DCD12D) mod 2³²)` ∈ [0, 1020] and
`w′ = (1024 + s) · κ + β` with the FP16 constants κ = 0x1eee and β = 0xc931, rounded by ExLlamaV3
to FP16 (its `__hfma`). ExLlamaV3 then runs FP16 activations. M0 pins, at the two producing
versions, the weight order within a tile, the bit order of the stream, the Hadamard scale and the
meaning of `out_scales: always`.

**E3A16, the canonical arithmetic of every EXL3 matrix in the engine** (routed experts, MTP experts
and the dense classes; every route, both processors):

1. **Input rotation.** u = H(x ⊙ suh), unnormalized, in FP32: x ⊙ suh is exact (BF16 × FP16), and
   the butterflies run in one fixed order (strides 1, 2, …, 64; sum, then difference) with no fused
   multiply-add.
2. **Block quantization.** Per 128-element block b: exponent e_b from max |u| in the block, floored
   at (the vector's largest e_b) − F with F = 21 − ⌈log₂(K / 128)⌉ (16 for K = 4,096, 14 for
   K = 16,384), so that step 4 cannot overflow; a_k = RNE(u_k · 2^(13 − e_b)), a 14-bit signed
   integer.
3. **Integer weights.** The exact codebook value is `w′ = (1774 s − 905,216) · 2⁻¹⁸` (κ and β are
   exact binary fractions), an integer below 2²⁰ times a power of two. E3A16 uses this exact value;
   ExLlamaV3's FP16 rounding of it is not reproduced.
4. **Exact dot products.** Per block, P_b = Σ a_k s_k and A_b = Σ a_k in int32 (|P_b| < 2³⁰), the
   block value V_b = 1774 P_b − 905,216 A_b in int64 (|V_b| < 2⁴²), and Y = Σ_b V_b · 2^(e_b − e_min)
   in int64 (shifts of at most F bits, so |Y| < 2⁶³). Every integer is exact, so the order of
   products and blocks is free.
5. **Output.** y′ = fl32(Y) · 2^(e_min − 31) · r_in · r_out (the two Hadamard scales; their product
   is the power of two 2⁻⁷ if each is 128^−½, as M0 confirms), then the output transform H in FP32
   in the same fixed order, × svh in FP32, and one rounding to BF16.

Integer steps are exact, FP32 steps are fixed-order IEEE operations compiled without contraction
(`-ffp-contract=off`; `__fadd_rn`/`__fmul_rn` on the device), and the block quantization is a
deterministic function of u. So **every route on every processor produces the same bits** — the
placement-invariance principle the whole offload tier is built on (MoE §2). Relative to
ExLlamaV3's own FP16 runtime, E3A16 differs by the codebook's FP16 rounding (≤ 2⁻¹¹ of each
weight) and by 14-bit block activations instead of FP16 elements (absolute error ≤ 2⁻¹⁴ of the
block maximum). Both are at the level of ExLlamaV3's own rounding and far below the 3-bit format's
error; §20.1 bounds them against an FP64 oracle of ExLlamaV3's decode.

**GLM experts on top of it.** gate and up are two E3A16 matrices with their own `suh`, so each has
its own input rotation; their BF16 outputs are clamped (gate ≤ 10, up ∈ [−10, 10]) as exact
operations on BF16 values, SwiGLU runs in FP32 and h rounds to BF16; down rotates h with its own
`suh`. The combine is `y = bf16(Σ_i w_i · y_i + y_shared)` in FP32, in ascending expert id (not
rank: the prefill accumulator of §17.2 uses the same order); every input is the same bits wherever
its expert ran, so placement cannot change it.

**Routes.**

- **Narrow** (T ≤ 8: decode and verification): CUDA-core integer arithmetic; per weight a funnel
  shift to extract the state, an `IMUL` by the multiplier, a `DP4A` byte sum and an `IMAD` into the
  block's P_b. The state decode is shared by the T columns. Work items: (job, gate/up slice), 16 per
  job; (job, down block), 32 per output, each reading one contiguous part of the record (§7.2). The
  rotations of a job's inputs (two 4,096-element and one 2,048-element transform per column) run in
  the job's prologue.
- **Wide** (n_e > 8: prefill and large verify windows): int8 tensor-core MMA (`mma.sync` `s8`/`u8`).
  With s = 128 s_hi + s_lo and a = 128 a_hi + a_lo, every limb fits int8 and P_b is four int8
  products per weight and activation, accumulated in int32 per block; decoded `s` limbs of a tile
  are reused across the CTA's token tile. Effective rate estimated at 105-150 TOPS of logical
  operations (§5.4); for scale, Flash-Next's exact int8 wide route measured 3.8× its A4 route's
  time (6.24 against 1.65 ms per call; FN §17.3).

### 9.6 Miss arbitration

The arbitration rules — SSD-only misses first (only the CPU serves them without a fetch), then
host-memory misses fewest columns first in job order, up to the call's cap, the host share
`misses − misses / divisor` — are the shared op's (MoE §6.1), and the divisor rule and its startup
derivation are MoE §5.6. This family's inputs:

- **RAM misses:** with the CPU's ≈ 0.3 ms per expert against the link's 0.34 ms, d = 2 (half to
  each) at the central estimate, against d = 3 for Flash-Next's 2.76 MB records. The CPU cost is
  compute-bound rather than a DRAM rate, so d comes from the measured per-expert pair through the
  per-count split table of §12.3, seeded from the formula. M5 re-sweeps it.
- **SSD misses:** CPU jobs first (computation starts on the first landed units), so their reads
  overlap the layer's GPU work; staged DMA after landing only beyond the CPU cap (16 jobs per call).
- **Multi-column experts** (verify windows): the CPU E3A16 kernel decodes each state once for all
  columns, so its cost grows slowly with n until the products dominate (§12.2); beyond the
  calibrated n the expert goes to DMA.

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
the least weight, and renormalizing would scale up the kept ones); renormalization is an M7a
variant, kept only if it lowers KL. Ranks below k_l are always computed. The shared expert is
never skipped.

**What "available" means: the scope**, chosen at startup:

| Scope | Left out at rank ≥ k_l | Model gain, 5090 machine (§5.3 model, Maya's map, two volumes) |
|---|---|---|
| `ssd` | only experts that would need an SSD read (not in VRAM, not in RAM, not landed) | +4-9 % decode |
| `vram` (Maya's rule) | every expert not in a VRAM frame or already staged; RAM and SSD misses alike | +18-29 % decode (central 31 → 37-40 tok/s) |

`ssd` removes the most expensive stalls (≈ 0.8 ms each) at the smallest quality cost, but with
the EXL3 bank mostly in memory there are only ≈ 10 SSD reads per token to remove; `vram` is Maya's
form. The gains assume the skippable ranks miss 1.0-1.3× as often as the layer average (they are
the colder experts); M7a measures both.

**Keep map.** One digit per layer, as Maya's `TRUNK_KEEP_MAP` (`8` = exact for that layer). The
starting point is Maya's measured map for this model: 8 on layers 3-22, 5 on 23-29, 4 on 30-44.
Maya measured each band's sensitivity on Maya-M (KL to its own full model with keep 5 in one band
at a time): layers 3-7 0.040, 23-29 0.012, 30-44 0.002-0.003; the whole map 0.018 (code 0.017),
against 0.032 for a uniform keep 7, for +23 % decode. Early layers are never reduced by a preset.
The engine measures its own band table on the default EXL3 artifact (M7a) before any preset
ships; maps are fixed at startup (they are part of the round's captured graphs as per-layer
constants).

**Weight-threshold rule (strata-glm's form).** Instead of, or beside, the rank map, an unavailable
expert is left out when its weight w_i / 2.5 (its share of the normalized top-8) is below a
threshold θ (`--approx-min-weight θ`). strata-glm measured +30 % at KL 0.022 for SSD-only experts
below ≈ 3 %, and +20 % at KL 0.044 for RAM misses below 8 %, with renormalization (§4.3). Weight
is the quantity the combine actually scales by, so it may beat rank at equal KL; M7a compares the
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
  background within the promotion budget; otherwise skipped experts would stay cold for good. M7a
  measures crediting at full and at half weight.
- **Prefill stays exact.** A prompt chunk streams almost every expert anyway (§5.4), so skipping
  saves little there; latents, KDA state and the prefix cache's blocks and snapshots are computed
  exactly. Only decode and verification rounds approximate. The prefix cache tags every
  decode-produced block and snapshot with the mode, map and rule, and exact requests never reuse
  a tagged one.
- **Speculation.** Verification runs in the same mode, so accepted tokens are the approximate
  model's; the MTP drafter's own bounded routing (§10.6) is unchanged.
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
capacities, constants, candidates and measurements. The subject is the 12,384 routed keys of
MikeRoz's artifact (42 × 288 main + 288 MTP; turboderp's has 12,096). Code: `src/models/offload/`,
driven by the Program in `src/models/glm5_next/program/`.

### 10.1 Tiers

| Tier | Capacity | Holds |
|---|---|---|
| VRAM frames (9 MiB + 36 KiB side entry) | 5090 machine: ≈ 2,600 (21 %), elastic with KV and the display headroom; production machine: ≈ 9,700 (80 %) — §19.1 | Hot experts; staging; landing frames; leases |
| Pinned RAM slots (9 MiB) | 5090 machine: ≈ 8,000-9,000 (66-74 %); production machine: the rest, ≈ 2,400 (20 %) — §19.2 | Warm experts; the landing ring; the demotion list |
| Expert volumes | All 12,384 records (12,096 for turboderp's), read-only | Everything |

Residency is **exclusive with shadows** (MoE §4.1): a VRAM resident keeps its RAM copy only as a
shadow, the first RAM victim. Its cost is demotion copies: strata-glm rejected its exclusive mode
at 158 write-backs per token (§4.3). Here a demotion whose shadow survives copies nothing,
demotions are capped at 32 per boundary, and M5 reports demotion copies per token beside SSD
reads, so the trade is measured. The vector plane holds every expert's vectors regardless of tier.

### 10.2 VRAM policy

The policy is the shared LFRU (score `f / (now − last + 1)`, global never-reset counts, halving,
one pool per record size — one here; MoE §4.3). M1 replays these candidates on GLM traces at the
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
victim's and from landed SSD records (copied straight from the landing slot). M5 sweeps the
interval for 9 MiB promotions ({1, 2, 4, 8} × the link factor) and two rules from Maya (§4.2): an
**admission threshold** (a fetched expert is kept only once its aged route count reaches
m ∈ {0, 2, 4, 6}), and a **fill rate for free frames** (after a prompt returns its borrowed frames,
up to 24 of the hottest RAM experts per token instead of the steady-state pace). Some misses
always stay on the link: with every miss on the CPU nothing is promoted, and Maya's hit rate fell
to 60 %. No promotions after a walked prefill (MoE §7.3).

### 10.3 Staging, landing and leases

Staging slots, landing frames and their adoption, frame retirement and leases are the shared
machinery (MoE §4.4-§4.5). Staging slots are frames with a staging role, at most 32 (0.28 GiB),
since a staging slot costs as much VRAM as a cached expert. The prefill stream's ring and staging
areas (§17) and the vision encode window of a later track borrow frames through leases.

### 10.4 RAM tier

The tier's machinery — decayed-LFU ranking with O(1) lazy heaps, the slot lists and pins,
demotions, the unpinned records past the Windows page-lock ceiling — is MoE §4.6 and §4.11. This
family's terms and additions:

- **Sizing** by the host RAM ledger (MoE §4.10) with this family's reserve and buffers (§19.2):
  at 88 GiB available ≈ 9,061 slots; at today's 79 GiB ≈ 8,037.
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
  losing on this model ("drift-shaped locality"), so M1 replays decayed LFU (half-lives 16-256),
  LRU, CLOCK and LFRU on the two-level simulation (`tools/expert_cache_replay/host_tier.py`) and
  adopts the policy with the fewest SSD reads per token on held-out traces; the half-life stays a
  family constant.
- **Admission** at the shared margins (a demand landing when it outranks the lowest resident, a
  prefetch at a margin above the victim; MoE §4.6); VRAM victims without a host copy are demoted
  when they outrank the RAM victim (≤ 32 per boundary in decode).

### 10.5 Warm start, prompt seeding and refill

- **Expert state file** (the shared `--expert-state`, `src/models/offload/expert_state.*`; MoE
  §4.3): startup fills VRAM from the top of the saved ranking and RAM with the next records,
  reading the volumes in record order with large coalesced reads at full striped bandwidth
  (≈ 93 GiB in ≈ 8 s at 12.7 GB/s, overlapped with pinning; model): startup ≈ 15-20 s plus the
  dense load. A GLM calibrated order is shipped when M1's traces justify one, else the seed
  spreads across the layers (MoE §4.3).
- **Prompt seeding** is the shared prompt credit (counts only, no copies; MoE §4.3), with a
  weight M1 fits. The prompt landing reserve — admitting the prompt's SSD-read experts to RAM —
  is **off by default**: Maya measured that keeping a prompt's disk reads in RAM gave more disk
  reads in the answer, not fewer, since a long prompt touches every expert (§4.2). M5 re-measures
  it once.
- **Idle refill:** with no request active, the read agent re-ranks and refills RAM at full
  bandwidth (the shared boot warm start and tier fill; MoE §4.8).

### 10.6 MTP experts (MikeRoz's artifact)

The 288 MTP experts are the volumes' MTP bank (4-bit, 12 MiB records), but the drafter never
waits for them:

- **Residency-bounded draft routing**: the drafter's router takes the top-8 among MTP experts that
  are `READY` in VRAM, renormalized over those; of the route's missing experts, the 2 best-scored
  are computed by the CPU lane from RAM (Maya's `MTP_KEEP` 2: better acceptance than skipping
  all, never slower; §4.2). This changes only acceptance, never verified output. (Flash-Next
  never needed the form — its drafter's experts are resident; FN §11.2 — but a GLM MTP bank is
  3.5 GB and mostly off-device.)
- **Bounded MTP pools of their own size**: at most P records in VRAM (default 48, 0.56 GiB, carved
  from the frame pool; M7 sweeps 0-96 against acceptance and the main cache's misses), ranked by
  LFRU weighted by acceptance; a RAM sub-pool of 12 MiB slots (default 128, 1.5 GiB, taken from
  the main tier only when `--spec mtp` is on), filled by background promotions and landed reads.
- `--spec mtp` is off by default for this model until M7 shows a gain (§16), and refused for an
  artifact without the MTP layer.

---

## 11. Multi-SSD expert streaming

The read-only rule, the unbuffered reads, the demand-before-prefetch classes and the read queue
itself are the shared SSD tier's (MoE §4.7); this section is this family's composition of them
over its striped volumes (§7.3), plus the two additions of §8.2: one read queue per volume, and
landed-unit masks.

### 11.1 Requirements

- One to four expert volumes on any drives; two is the designed case on the 5090 machine; the
  production machine runs one until a second drive exists.
- Read-only at runtime; no SSD cache, spill or swap (MoE §4.7).
- Every record read uses every volume; per-record latency ≈ record / Σ bandwidth.
- Demand reads never queue behind more than ≈ 0.3 ms of prefetch per drive.
- A failed read fails only the rounds that need it (the shared failure path, MoE §4.4), never the
  server.
- Windows (IOCP) and POSIX (the `pread` pool) behind the shared `DirectReadQueue`
  (`src/core/direct_read_queue.h`); io_uring is option O6 (§25.3).

### 11.2 Reading a record

A record read is 12 unit reads issued at once: w_v units to each volume v, each landing at its
byte offset in one contiguous 9 MiB landing slot (pinned, 4 KiB-aligned). With w = (5, 7) on
`D:` + `G:`, the 990 PRO serves five 0.75 MiB reads and the Optane seven; at ≈ 5.7 and ≈ 7.0 GB/s
the record completes in ≈ 0.7-0.8 ms and its first gate/up units after ≈ 0.15-0.2 ms (model; the
probe's cells are §3.3's, M0 measures the 0.75 MiB units).

Unit issue order is gate/up units first (0-7), round-robin over volumes, then down units (8-11), so
consumers can start the gate/up half while the down half is still in flight.

### 11.3 The read engine

- **One `DirectReadQueue` per volume** (§8.2), owned by the read agent thread: each queue maps
  (bank, record, unit) to (volume, offset) by §7.3, submits a record as its 12 unit reads with
  one tag, and reports per-unit completions. Adjacent units in one volume coalesce into one read
  (the queue's scatter form).
- **Depth caps per volume**: demand 8, lookahead 3, background 1, total 10 (0.75 MiB units: the
  probe's knee lies between 1 MiB at QD 4 and 256 KiB at QD 8; §11.7 calibrates).
- **Priorities**: demand (current layer, by layer then score) > lookahead depth 1 > 2 > 3 >
  background refill. A queued, unissued lookahead unit is dropped when its round has passed its
  layer.
- **Landing ring**: 64 slots (0.56 GiB of the RAM tier) by default; a slot is reusable once its
  CPU job has finished or its staged copy has landed, and once the RAM tier has admitted or
  declined it (an admitted landing becomes a RAM slot by changing roles, without a copy).
- **Completion publication**: on each unit completion the agent sets the unit's bit in the slot's
  landed mask and bumps its sequence word (release). CPU workers spin on their units' bits; the
  GPU's stage kernel copies a unit to its staging frame as soon as its bit is set (one 0.75 MiB
  copy per unit, so a pair of gate/up slices is usable as it lands). The record's vector-plane
  entry is copied with its first unit.
- **Never mapped.** Volumes are opened unbuffered and never memory-mapped, not even for metadata:
  Maya measured that on Windows a mapped file sends every direct read of it down a slower path
  (§4.2), and the shared tier opens deny-write handles for the same reason (MoE §4.7).
- **Readers.** One agent thread with the per-volume queues by default. Calibration (§11.7) also
  tries Maya's alternatives: pairs of units per read instead of single units, and a few
  whole-record reader threads, which gained 25-50 % on DRAM-less drives there and lost 15 % on
  drives with DRAM.

### 11.4 Consuming partial records

The unit boundaries coincide with the work units of both processors (§7.2), so no work item ever
straddles two drives' reads:

| Consumer | Granularity | Waits for |
|---|---|---|
| CPU Phase A (gate/up + SwiGLU) | 2 gate/up slices per unit | its unit's landed bit |
| CPU Phase B (down) | 8 down blocks per unit | all of h (Phase A barrier) and its unit |
| GPU narrow route via staging | slices / blocks | the unit's staged-copy ticket |

### 11.5 Lookahead

An SSD read that is not issued until its layer routes costs ≈ 0.8 ms of stall: about ten layers of
pre-routing GPU time. Reading the predicted SSD-only experts of layers l+1..l+3 (§9.4) ahead is the
obvious remedy, but the one measurement on this model is negative: Project Maya found only ~40-45 %
of disk misses predictable 1-4 layers ahead, and its lookahead was exact but slower on one NVMe
because wrong reads competed with demand (§4.2). Two volumes give lookahead spare bandwidth that
one does not, so it is designed here but **off by default** (`--ssd-lookahead off`) until M5 shows a
gain on the two-volume configuration. With the EXL3 bank mostly in memory there are only ≈ 10 SSD
reads per token, so its ceiling is ≈ +10-20 % (the §5.3 all-in-memory column):

- **Budget per layer:** lookahead reads may use the drives' idle time only. The agent tracks each
  volume's queued demand bytes; a lookahead read is issued only if it would complete before the
  predicted use (depth d ≈ d × 0.7 ms ahead at the central token rate) and the volume's demand
  queue is empty.
- **Over-fetch:** Fate (arXiv 2502.12224) reached 97 % coverage by fetching the confident set
  rather than the exact top-k. The candidate width k′_d is a calibrated parameter; ox-boost
  measured that over-fetch hurts when it competes with demand (P = 6-8 slower than 4), which the
  budget rule prevents for SSD reads.
- **Used landings** become CPU jobs at once; **unused** ones are offered to the RAM tier like any
  other landing, so a wrong guess costs drive time but not necessarily the bytes.
- **Measured before adoption** (M1, M5): recall of SSD-tier experts by depth and width, useful-byte
  ratio, exposed SSD stall per token with lookahead off, depth 1, and depth 1-3. Lookahead depth
  d is disabled automatically for a layer whose useful-byte ratio stays below 0.3.

### 11.6 Placement on the 5090 machine

| Configuration | Measured bandwidth (§3.3) | Free space for a volume | Advice |
|---|---:|---|---|
| **`D:` + `G:`**, w = (5, 7) | 12.7 GB/s (5.7 + 7.0) | 144 / 128 GB | **Default.** Both drives otherwise idle; the Optane has the lowest tail latency; volumes 49.1 and 68.7 GB. `G:` also holds the page file, which the pinned tier and the 2 GiB reserve keep quiet; §11.7's telemetry would show paging as a slow volume |
| `D:` + `E:`, w = (6, 6) | 11.7 GB/s | 144 / 371 GB | Works; `E:` is the repository and build drive and showed 15-41 ms p99 tails during the probe |
| `C:` + `D:` + `G:` or `C:` + `D:` + `E:`, w = (4, 4, 4) | ≈ 17.6 GB/s (`C:` bypasses the chipset) | `C:` 243 GB | Fastest; `C:` carries the OS and page file, so it is opt-in; +8 % decode (§5.3) |
| Three volumes all on the chipset | 14.2 GB/s (DMI-bound) | | Not recommended |
| One volume | 5.7-7.0 GB/s | | Supported; ≈ 0.76-0.85× of two-volume decode (§5.3); the production machine's default |

Unequal weights matter little: with w = (6, 6) on `D:` + `G:` a record takes max(0.83, 0.67) =
0.83 ms; with (5, 7), max(0.69, 0.79) = 0.79 ms (5 % faster).

### 11.7 Calibration and topology checks

At startup the read agent reads 64 units from each volume at QD 1 and QD 8 (≈ 40 ms total) and
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
  stores in the main artifact (≈ 50 MB read); `tools.glm5_next.verify_artifact` checks every unit
  offline. Per-read hashing stays rejected.

---

## 12. CPU expert engine

The engine's design — role and limits, the ISA variants, the worker team's scheduling, the
miss-service protocol, the startup rate probes, the decode divisor and the prefill split — is the
shared tier's (MoE §5); this section keeps this family's kernel, constants and candidates. Code:
`src/ops/offloaded_sparse_moe/cpu/` (the `exl3_mul1` trait joins `w4a4`/`w4a16`/`ggml`/`fp8`
there).

### 12.1 Role

The CPU computes about half of the RAM misses (§9.6) and every SSD miss it can take within the job
cap, in place from RAM slots or landing slots, with E3A16 — the same bits as the GPU (MoE §2,
§5.1). The shared limits (512 jobs per layer call, 8 columns per job, 4,096 columns per served
call; `src/ops/offloaded_sparse_moe/cpu/miss_request.h`) bound this family's calls with room: a
decode round's 42 layers × 8 experts stay far inside them.

### 12.2 The E3A16 kernel and its cost on the i9-13900K

`cpu/exl3_expert.cpp`: an AVX2 + AVX-VNNI kernel and a scalar form (the exact oracle, shared with
`ops/common/canonical_math.h`). Per 16 weights and column: extract the 16 states from the tile's
stream (variable shifts of gathered words), multiply by the codebook multiplier (`vpmulld`),
byte-sum (`vpmaddubsw` against ones, then `vpmaddwd` against ones), pack to 16 bits and accumulate
a · s with one `vpdpwssd`: about 17 instructions, ≈ 1.1 micro-operations per weight, most on the
two vector ALU ports that also execute the multiply. ExLlamaV3 has no CPU path; adrienbrault's fork
computes EXL3 experts on 8 Zen 5 cores with AVX-512 at DRAM speed (61 GB/s, §4.1).

**Estimate** (model; M0 measures before anything depends on it): ≈ 0.44 cycles per weight per
P-core, 25.2 M weights per expert, so ≈ 2 ms of one P-core at 5.5 GHz: **≈ 0.34 ms on 6 P-cores,
≈ 0.25 ms on 8**, compute-bound, against ≈ 0.12-0.19 ms to stream 9.4 MB from DRAM at 50-80 GB/s.
The 16 E-cores also have AVX-VNNI and 128-bit vector units; at roughly a third of a P-core's rate
each they could add ≈ 5 P-cores' worth, bringing an expert to ≈ 0.15-0.2 ms and the lane back to
DRAM-bound. §5.3 carries 0.22 / 0.30 / 0.40 ms. A second column costs only the products and the
rotations (the states are decoded once), so a verify column adds roughly a fifth of the first
column's cost.

This is the largest difference from Flash-Next's CPU lane, which streams 2.76 MB records at DRAM
speed (MoE §5.2): here each expert takes about as long as its DMA (0.34 ms), so the CPU and the
link split the misses evenly (§9.6), and the kernel's speed moves decode about as much as the
second drive does (§5.3).

### 12.3 Execution

The team mechanics — the service thread as worker 0, epoch spin then futex park, work by atomic
counters, the phases with their barriers, exact sums making any split bit-exact — are MoE §5.3.
This family's settings:

- **Landing-aware items:** an item over a unit that has not landed is skipped and retried after
  the worker's other items, so workers never idle while another unit is available (§11.4).
- **Workers:** Flash-Next's optimum of six was set by DRAM sharing with the PCIe stage (MoE §5.2).
  The E3A16 lane is compute-bound, with CPU plus DMA at ≈ 60 GB/s of the ≈ 80 GB/s DRAM limit,
  so M0 and M5 sweep 6-8 P-cores plus 0-16 E-cores. Workers sit on physical cores, never on SMT
  siblings: Maya measured hyperthreads at 21.8 → 13.3 tok/s, because they starve the disk readers
  and the service threads (§4.2). The read agent and the engine thread keep P-cores of their own.
- **Pieces, not shares:** a call's CPU work is cut into ~48 pieces claimed in turn, so one worker
  held up (by an SSD unit not yet landed, or by a slower E-core) does not hold up the layer (Maya:
  14.3 → 17.1 tok/s with 6 threads). The shared atomic item counters already have this shape; the
  piece count is sized for 9 MiB records and mixed core speeds.
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
CPU-computable; here every E3A16 route is exact, so the CPU may take **any** column count without
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

In both artifacts `kv_b` is one EXL3 matrix [512 → 32,768] whose output Hadamard blocks (128) fall
inside each head's 256 key rows and 256 value rows. o_h = W_uv,h u_h is its ordinary route restricted
to head h's value rows. q'_h = W_uk,hᵀ q_h is its **transposed** application: the roles of the two
rotations swap (q_h is rotated with head h's `svh` blocks, the trellis tiles are read along the
other axis, and the result is rotated back and scaled by `suh`), with the same E3A16 steps (§9.5).
Folding W_uk into W_qb or W_uv into W_o offline was rejected: it would requantize the weights and
read more bytes (50 M and 134 M parameters against 33 M and 75 M).

### 13.2 Decode kernel (`mla_sparse_attention`)

Per layer and sequence: Q' is 64 × 512, the selected latents ≤ 2,051 × 512.

- Two BF16 `mma.sync` GEMMs with FP32 accumulation: S = Q'Cᵀ, then U = P C, an online softmax
  between them. 268 MFLOP and 2.1 MB per layer and token: bandwidth-bound at ≈ 1.5 µs of reads.
- Split over token blocks: 32 CTAs × 64 tokens, partial (max, sum, U) merged in fixed block order.
- The selected set arrives as pool ids from K4m (§9.3); a CTA gathers whole 4-token pools through
  the block table plus the tail tokens.
- FP8 and INT8 latents (§13.5) are decoded to BF16 in the tile loader, as the existing
  `softmax_attention` consumers do.

### 13.3 The indexer (`dsa_indexer`)

Every step follows the reference (§2.2) at its precision:

| Step | Precision | Stored |
|---|---|---|
| k_t = LayerNorm(W_k x_t), z_t = W_g x_t | FP32 LayerNorm, BF16 outputs as the reference's linears produce | Tail: the incomplete pool's k and z per sequence (3 × 256 BF16) |
| Pool key = Σ_i softmax_i(z + APE) ⊙ k when a pool completes | FP32 (the reference's pooled keys are FP32) | **FP32 pooled-key plane**: 128 FP32 per pool = 128 B per token and layer |
| q_h = W_qb_idx c_q (EXL3 5-bit, E3A16), w_h = W_w x / √32 | BF16 linear outputs, FP32 weights | — |
| score_p = Σ_h w_h relu(q_h · key_p / √128) | FP32 | — |
| Top-min(512, P) pools + tail | Exact radix select, lower pool id on ties | Pool ids for K5m |

- **Decode** scans every complete pool: 4 bytes × 128 per pool, 16 MB per layer per token at 128K
  context (≈ 11 µs), 128 MB at 1M (≈ 85 µs; the long-context term in §13.6).
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
  W_kvb in the chunk on the E3A16 prompt route), the cheaper form when keys are shared by every
  query.
- Beyond that, each query has its own set, and the absorbed MQA form serves it (DeepSeek's choice
  for DSA): ≈ 269 MFLOP per query and layer, ≈ 1.9 s of BF16 tensor-core work for a 128K prompt
  (model). Query tiles of 64 share their tails and most recent pools; the kernel processes the tile
  against the union of its selected pools with a per-query mask, as Flash-Next's QSA prompt kernel
  does (FN §14.2).

### 13.5 KV formats

The latent is both key and value, so per-K/V split profiles do not apply.

| `--kv-dtype` | Latent per token and layer | With FP32 pooled key | 12 layers (incl. MTP): 128K / 1M tokens |
|---|---:|---:|---:|
| `bf16` (primary) | 1,024 B | 1,152 B | 1.8 GB / 14.5 GB |
| `int8` (INT8, one FP16 scale per 32, as Maya's) | 544 B | 672 B | 1.1 / 8.5 GB |
| `fp8` (E4M3, one FP32 scale per 128) | 528 B | 656 B | 1.0 / 8.3 GB |
| `nvfp4` (E2M1, E4M3 per 16) | 288 B | 416 B | 0.65 / 5.2 GB |

Orthogonal rotations commute with the absorbed product (q'·c = (Hq')·(Hc), Σ p c = Hᵀ Σ p (Hc)), so
the Hadamard-domain `vq2`/`k4v2` profiles could apply later; they are refused for this family until
a track qualifies them. Order of work: `bf16`, then `int8` (Maya measured it within FP16's own
run-to-run KL spread), then `fp8` and `nvfp4`. ExLlamaV3 stores 608 B per token and layer (FP8
latent, scales, pooled key), which matches the `fp8` row's scale.

### 13.6 Long context

Default `--max-context 131072`; up to 1,048,576 is supported. At 1M a `bf16` latent cache is 14.5 GB:
the expert cache would lose ~1,500 frames. Above 128K the design uses a live KV host tier — planned
for Flash-Next and never built there (FN §14.4), so it is new work here: the pooled-key plane stays
on the device, latent pages live in pinned RAM with a device CLOCK cache of hot pools, and only the
selected ≤ 2,051 tokens per query are read. The host copy costs RAM-tier slots (8.3 GB of `fp8`
latents at 1M = 880 slots), which the ledger shows. The indexer's per-token scan grows linearly
(≈ 1 ms per token across 11 layers at 1M); an FP8 copy of the pooled keys for the scan, with FP32
rescoring of the top candidates, is option O4 (§25.3).

---

## 14. Kimi Delta Attention layers

- **Mathematics:** the existing `kimi_delta_attention` Op
  (`src/ops/linear_attention/kimi_delta_attention/`, `include/infernix/ops/kimi_delta_attention.h`)
  is exactly this model's recurrence (§2.2), including the −5 lower bound and FP32 state. It has
  recurrent and chunked routes, was qualified against an FP64 oracle, and has had no consumer.
- **Decode** (§9.3 K2-K5): one projection launch for the fused `qkv_proj` (EXL3) and the small
  f_a, g_a, b projections (FP16; ox-boost: KDA gate fusion 5 → 2 GEMVs), `kda_input_prep` in the
  epilogue of f_b/g_b, the recurrent route, and the gated `o_norm` in `o_proj`'s prologue. State
  traffic is 285 MB per token: ≈ 0.2 ms, so a BF16 state is not worth its numerical change here.
- **Prefill:** the chunked route. FLA uses 64-token chunks and this Op 16; M8 measures 16, 32 and 64
  against the prefix cache's phase-aligned chunking (spec §7.2).
- **Speculative verification:** `kda_replay_record` writes, per verify column and layer, the inputs
  a fold needs (k, v, the 128-channel log α, β: ≈ 49 KB per layer and column), and the commit folds
  the accepted prefix from the pre-round state, as the GDN replay records do
  (`src/core/gdn_replay_records.*`). Bit-exact against the committing form.
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
C > 1. With EXL3 the producer of y applies its output Hadamard and `svh` before this epilogue.

### 15.2 Numerics

- Mixing weights, Sinkhorn, pre, post and comb in FP32, as the reference.
- Residual streams stay BF16. Flash-Next measured an FP32 residual as worse in perplexity
  (+0.030 nats; FN §17.7): the model expects the rounding it was trained with.
- The reference forms R' with two BF16 roundings (post · y, then the matmul, then the add). The
  fused update computes it in FP32 and rounds once: more precise, kept only if the
  teacher-forced comparison of §20.3 shows it is not worse (Flash-Next's rule, FN §17.7).
- Sinkhorn is reproduced iteration for iteration, never iterated to convergence.

---

## 16. Speculative decoding

### 16.1 Expected value

- Only MikeRoz's artifact carries the MTP layer; for turboderp's, n-gram proposals are the only
  speculation.
- The MTP layer's acceptance on this abliterated model is unknown; adrienbrault's ExLlamaV3 fork
  measured 1.74 tokens per step at depth 1 (acceptance 0.55-0.96) on the base model's 2.05 bpw
  EXL3 build (§4.1).
- On the 5090 machine the round time is dominated by host-served expert work, which speculation does
  not amortize (MoE §6.3). Extra verify columns are cheap for experts the CPU computes (each state
  is decoded once for all columns, §12.2) and expensive for DMA'd and SSD-read ones.
- Two GLM-5.3-Flash ports measured MTP slower (§4.1). Project Maya traced why on one GPU:
  consecutive tokens share only ~1 of 8 experts per layer and none of their RAM pulls, so a
  two-token verify costs nearly two tokens of PCIe time; its MTP pays only by pipelining across
  two GPUs (§4.2). `--spec mtp` is therefore **off by default** and becomes the default only if
  M7 shows it faster than plain on the NRS corpus. The case that could still pay here is a verify
  window whose extra columns land on CPU-computed experts, which the cost model of §16.3 prices.

### 16.2 Drafter

- Structure (upstream, M0 confirms against vLLM/SGLang): h₀ = `eh_proj`([enorm(embed(t)); hnorm(h)])
  where h is the main model's final hidden (M0 fixes whether before or after the final norm and
  stream mean), then one sparse-MLA block with its own latent KV layer and indexer, a 288-expert
  MoE with residency-bounded routing (§10.6), the shared expert, `shared_head.norm` and the head.
- **Index sharing:** with `index_share_for_mtp_iteration`, chained draft steps reuse the first
  step's pool selection; the drafter skips K4m on steps 2..K.
- **Draft head:** the full 6-bit `lm_head` (476 MB per step) or `--lm-head-draft` with a reduced head
  (frequent rows; M7 chooses its format), chosen by M7.
- **Bounded host waits**: draft experts are VRAM-resident ones plus at most the 2 best-scored
  missing ones, computed by the CPU lane (§10.6); it never waits for the SSD.

### 16.3 Draft length

Flash-Next's policy shape (FN §11.4): maximize E[tokens(K)] / E[t_round(K)] with
t_round(K) = t_gpu(K+1) + t_host(U_{K+1}, current tier shares) + t_ssd + t_draft(K); every input an
online EWMA; K = 0 always a candidate; experts used only by rejected columns are not credited to
the caches (MoE §6.3); the cost fits are placement-scoped and reset on a placement switch
(MoE §6.3). Expected outcome: K ∈ {0, 1} on most text, larger on code edits.

### 16.4 Other proposals

- **N-gram copy proposals** (`--ngram-draft-tokens`) through the same cost model; they need no
  drafter and no MTP frames, and work with both artifacts.
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

### 17.2 Expert supply within a layer: expert-group pipelining

Flash-Next double-buffers a whole layer's non-resident experts (MoE §7.1). Here that is
2 × ~2 GiB of frames, a sixth of the 5090 pool. Instead a layer is computed in **expert groups**
of G experts (default 48, 0.42 GiB):

1. Groups are the layer's experts in **ascending expert id**, G at a time, resident or not. The
   engine thread stages group g+1 (RAM records by batched copies, SSD records as their units land
   through the landing ring) while group g computes; resident experts compute in place in their group.
2. Each token's routed outputs are accumulated as their groups finish into an FP32 accumulator
   [T × 4,096]: acc_t += w_i · y_{t,i}, in ascending expert id, then the shared expert, then one
   BF16 rounding. That is Maya's windowed accumulation (~40 KB per token instead of ~185, so a chunk
   holds ~2× the tokens; §4.2), made exact: **the combine order is ascending expert id in decode
   too** (§9.5), so prefill and decode compute the same sum and no per-(token, rank) output buffer
   is needed.
3. 16 KB per token: 0.5 GB for a 32K span, against 2.1 GB for a BF16 buffer of every routed output.

The result is bit-identical to whole-layer processing and needs ≈ 0.85 GiB of staging plus the
accumulator instead of ≈ 4 GiB: ≈ 300 more experts stay resident through a long prefill.

Further prefill rules from Maya (§4.2), each exact:

- **Read-ahead of the next layer's SSD-only records** during layer l, in layer order, at full
  striped bandwidth, into a deep ring (Maya: 64-96 records; its reader had kept the NVMe at a
  quarter of its rate with 12). Prefill touches nearly every expert, so these reads are not
  guesses — the form the shared tier's fetch read-aheads already give a 64-record fetch (MoE §4.7).
- **Prestage during attention:** the link is idle while a layer's attention runs; the next layer's
  most-routed RAM experts are copied then (Maya `PRESTAGE`).
- **The last layer's tail for non-final tokens:** with no MTP drafter loaded, layer 44's output
  projection and MoE are computed only for the prompt's last token, after its caches (latents,
  KDA state) are updated for every token (Maya `PREFILL_TAIL_SKIP`). With the drafter loaded,
  every position's final hidden state is needed for its cells, so the full layer runs.
- **CPU share of the least-routed experts** (§12.4).

### 17.3 Compute

| Part | Route |
|---|---|
| Dense GEMMs | E3A16 prompt route on int8 tensor cores (four products per MAC), bit-identical to the decode route; registered shapes at H = 4,096 |
| Routed experts | E3A16 wide route, grouped by expert; ~910 tokens per expert per 32K span |
| MLA, indexer | §13.3-13.4 |
| KDA | chunked route |

### 17.4 Activation permissions (gated, off by default)

Two recipe-level permissions (the `AllowA8` classes of [engine
architecture](engine-architecture.md) §2.5), each a numerical change that passes §20.3 alone and
together before it can ship, and neither used in decode:

| Permission | Where | Basis | Expected (model) |
|---|---|---|---|
| **A8 dense prefill**: the rotated activations of step 2 (§9.5) quantized to 8 bits per 128-block instead of 14 | Dense prefill GEMMs only | Hadamard-rotated activations quantize well to 8 bits (QuaRot, arXiv 2404.00456); two int8 products per MAC instead of four | Dense prefill ≈ 2× |
| **A8 expert prefill**: the same for routed experts | Wide-route experts (n_e > 8) only | Same; Flash-Next's 8-bit permission measured 3.8× on its expert GEMM (6.24 vs 1.65 ms per call; FN §17.3) | Expert compute ≈ 2×; with both, 32K ≈ 5,700-7,800 tok/s (§5.4) |

A prompt computed with permissions differs from one computed without at rounding level; the
prefix cache keys blocks by recipe and permission set, so a cached block is reused only by the
same arithmetic.

### 17.5 Short prompts and time to first token

A fresh prompt of ≥ ~1K tokens pays one whole expert pass, ≈ 3.5 s on the 5090 machine and
≈ 1.4-1.8 s on the production machine (§5.4). The levers, in order: the prefix cache (§18), the
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

- **KV blocks:** 64-token content-addressed blocks of MLA latents and their FP32 pooled keys. Pools
  start at the sequence start, so a block holds exactly 16 complete pools and is self-contained;
  the incomplete pool's raw keys and gate scores are part of a snapshot.
- **State snapshots:** KDA state and conv state, 141 MB each, at the spec's tap points.
- **Value here is higher than on Flash-Next:** a hit avoids both the prompt compute and a whole
  expert pass (§5.4). Exactness holds because every EXL3 route is bit-identical (§9.5).
- **Host tier default 2 GiB** (`--host-context-mib`). Each GiB costs ≈ 114 RAM expert slots
  (≈ 0.9 % of the bank); the ledger line names the trade, and agentic users may raise it.

---

## 19. Memory plans

### 19.1 VRAM

The sizing law, the spill guard and the monitor are the shared tier's (MoE §4.9); the tables below
apply it with this family's fixed terms (`plan_device` of the family's memory plan is the single
authority for constructor sizes).

**RTX 5090 machine (32,607 MiB card, no display, C = 1).** The frame budget follows the measured
Windows budget of 31,419 MiB (§3.1): 30.68 GiB.

| Item | Default, GiB | With `--spec mtp` (MikeRoz), GiB |
|---|---:|---:|
| In use before loading (context, libraries) | 0.55 | 0.55 |
| Dense Text, EXL3 (`lm_head`, routers and mHC included) | **5.08** | 5.08 |
| MTP dense | — | 0.19 |
| KDA state, conv state, replay records | 0.20 | 0.20 |
| MLA latents + pooled keys, first 32K tokens (`bf16`; elastic) | 0.42 | 0.42 |
| Decode workspace, mailboxes, residency tables, io | 0.70 | 0.70 |
| Staging (32 frame-sized slots) | 0.28 | 0.28 |
| CUDA graph reserve; headroom | 0.25; 0.25 | 0.25; 0.25 |
| MTP expert pool (48 × 12.04 MiB, §10.6) | — | 0.56 |
| **Frame pool** (of the 30.68 GiB budget) | **22.95** | 22.20 |
| Frames of 9 MiB + 36 KiB | **2,601** (21.5 % of 12,096) | 2,516 |

At 128K actual context with `bf16` latents the KV takes 1.3 GiB more (−147 frames); with `fp8`, −63.
Each lane at C > 1 costs its KDA state (141 MB, ≈ 15 frames) and its KV extent.

**Production machine (RTX PRO 6000, 96 GB, Linux, C = 1; model).** The default startup form probes
a fully mapped KV pool (MoE §4.8); the bank does not fit, so the placement falls back to the
maximal RAM tier and the frame pool takes what is left:

| Item | Default, GiB |
|---|---:|
| In use before loading | 0.55 |
| Dense Text, EXL3 | 5.08 |
| KDA state, conv state, replay records | 0.20 |
| MLA latents + pooled keys, `--max-context 131072` fully mapped (`bf16`) | 1.80 |
| Decode workspace, mailboxes, residency tables, io | 0.70 |
| Staging (32 slots) | 0.28 |
| CUDA graph reserve; headroom (the card also drives the display) | 0.25; 0.50 |
| **Frame pool** | **≈ 85.6** |
| Frames of 9 MiB + 36 KiB | **≈ 9,700** (80 % of 12,096) |

With `--spec mtp` (MikeRoz): MTP dense 0.19 and the MTP pool 0.56 come out of the frames
(−84). At C > 1 each lane adds its KDA state and KV extent. `vram=100%` is refused: the bank
(106.7 GiB in frames) exceeds the card.

### 19.2 Host RAM

**RTX 5090 machine (95.8 GiB visible).** The ledger's law and terms are the shared tier's (MoE
§4.10); this family's constants:

| Item | GiB |
|---|---:|
| RAM expert tier, pinned (slots of 9 MiB), including the 64-slot landing ring | the rest |
| Expert vector plane, pinned | 0.43 |
| Token embedding, pinned | 1.18 |
| Prefix Host tier | 2.0 (default) |
| Program pinned buffers (mailboxes, io) | 0.25 |
| Process (tokenizer, server, frontend) | 1.5 |
| Reserve `--ram-headroom-mib`; growth allowance | 2.0; 1.0 |
| MTP RAM sub-pool (with `--spec mtp` only, §10.6) | 1.5 |

| Available at startup | Expert tier | Slots | Share of the 12,096-expert bank (less 64 landing) | With VRAM frames |
|---:|---:|---:|---:|---:|
| 88 GiB (other programs closed) | 79.6 GiB | 9,061 | 74.4 % | 95.9 % |
| 79 GiB (this session's measurement) | 70.6 GiB | 8,037 | 65.9 % | 87.4 % |

Every GiB that other programs release becomes 114 more slots; the startup ledger prints the figure.
The pin room (MoE §4.11: shared system memory 76.61 GiB on this host) caps the pinned part at
≈ 76 GiB, so the top of the tier is unpinned records.

**Production machine (89 GB).** Only ≈ 2,400 records (the VRAM pool's complement) need RAM: the
tier holds them all (≈ 21 GiB with the landing ring) and the remaining ≈ 58 GiB stays with the OS,
the page cache and other programs — the ledger reports it, and the SSD tier is empty at this
placement. Raising the placement's VRAM share (up to the frame pool's ceiling) or `--spec mtp`'s
sub-pool are the only consumers.

### 19.3 Disk

**5090 machine:**

| Item | Size | Where (default) |
|---|---:|---|
| Source artifact (download) | 126.1 GB (MikeRoz) / 122.3 GB (turboderp) | `E:` (deletable after verification) |
| Main artifact | ≈ 8.3 / 8.1 GB | `E:` |
| Expert volume 0, w = (5, 7) | 49.1 / 47.6 GB | `D:` |
| Expert volume 1 | 68.7 / 66.6 GB | `G:` |
| Expert state file | < 1 MB | beside the artifact |
| FP8 teacher source for the quality report (M4, optional, §20.3) | ≈ 330 GB | `E:`, after deleting the source artifact; deletable after |

**Production machine:** one volume beside the artifact on the T710 (114.15 GB for turboderp's
bank, 117.78 GB with MikeRoz's MTP bank), plus the same main artifact and state file.

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
| EXL3 codec (`canonical_math.h`) | Exact: an independent Python tile decoder (state windows, multiplier, byte sum) equals ExLlamaV3's own `reconstruct` output, as FP16 values, on every tile of sampled modules of both artifacts (a qualification-only dependency on ExLlamaV3 at the producing versions); the relaid-out records decode to the same weights as the source tensors |
| `exl3` Linear (K = 3-6), decode and prompt routes | **Bit-exact** between the routes and against the scalar E3A16 form, at T = 1..8, seams and real shapes; column invariance for T ≤ 8. FP64 oracle of the EXL3 definition with ExLlamaV3's FP16-rounded codebook: relative error within the bound E3A16's two deviations imply (§9.5), reported per class |
| `offloaded_sparse_moe` with `exl3_mul1` at (4096, 2048, K = 3) and the MTP bank (K = 4) | **Bit-exact** across GPU frame, GPU staging, CPU on every ISA, every landing order of units, n = 1..8 and the wide route; the same FP64 oracle. GLM routing (`noaux_tc`): exact against an FP64 sigmoid/bias/top-8 oracle on near-tie-free inputs, ties by lower id. Clamp boundaries (±10, NaN, Inf) |
| `mhc` | FP64 oracle of the reference formula including all 20 Sinkhorn rounds; gate and comb relative error ≤ 1e-6; streams within the BF16 rounding bound |
| `kda_input_prep`, `kda_replay_record`, fold | FP64 oracle; replay fold bit-exact against the committing form |
| `dsa_indexer` | FP64 pooling and scores; exact selection on near-tie-free fixtures; pool agreement on real activations |
| `mla_sparse_attention` | FP64 softmax attention over the decoded latents of the selected set, absorbed and explicit forms, including the transposed `kv_b` route; every latent format |
| Expert volumes, per-volume read queues | Exact: every unit of the first, last and MTP banks read through the queues equals the converter's bytes; permuted completion orders; fault seams |

### 20.2 Model-level checks

The whole model has no local reference runtime in the engine's sense. Flash-Next's method applies
(FN §17.1): per-block checks on the model's own activations against an FP64 reference built from
the decoded real weights (`tools/glm5_next/block_check.py`), and single-module agreement with
Transformers' modules (`upstream_check.py`), for every layer type, the MTP block and the head.

Two offline **reference forwards** (`tools/glm5_next/reference_forward.py`, qualification tools, not
a product route) run the whole model teacher-forced, layer-major, streaming each layer's weights
from the disk once per pass (≈ 10-60 s of reads), so they fit the 5090 machine:

- **EXL3 reference:** the artifact's weights decoded exactly as ExLlamaV3 decodes them (FP16 codebook
  values), FP32 activations and accumulation. It isolates the engine's arithmetic from the
  quantization.
- **FP8 teacher:** the artifact's FP8 source (`orcarouter/GLM-5.3-Flash-Uncensored-FP8` for
  MikeRoz's, `zai-org/GLM-5.3-Flash` for turboderp's), block scales applied, FP32 activations. It
  measures the quantization itself.

### 20.3 Gates (fixed now, before any measurement)

Teacher-forced on frozen texts in the GLM chat template (code, document, chat, CJK, a tool-call
transcript; ≥ 2,500 positions), with `bf16` latents. Scored positions include a long-context block:
at least 512 positions after a prompt of ≥ 8K tokens, well past the 2,051 tokens below which the
indexer selects everything. strata-glm measured its KLs about twice as high there as on short texts
(§4.3), and short texts never exercise the selection.

| Gate | Against | Mean KL | p99 per-token KL | Top-1 agreement | Perplexity change |
|---|---|---:|---:|---:|---:|
| **Execution** (the engine, default settings) | EXL3 reference forward | ≤ 0.002 nats | ≤ 0.05 | ≥ 99 % | ≤ 0.2 % |
| Each permission (A8 dense prefill, A8 expert prefill), alone and together | The engine without permissions | ≤ 0.005 | ≤ 0.08 | ≥ 98 % | ≤ 0.5 % |
| Fused mHC update (§15.2) | reference two-rounding form | not worse | | | |

For the permissions the prompt is prefill-computed with the permission and the scored positions are
decoded without it.

**Quantization quality, reported without a gate** (the artifact is the owner's choice): each
artifact's KL, top-1 and perplexity against its FP8 teacher on the same texts, beside alesha-pro's
0.047 for the base build, so the uncensored build gets its first measurement and the base build an
independent check of the published one. The report also attributes the KL by class: the FP8 teacher
run again with one class at a time (routed experts, attention projections, shared experts, dense
MLP, head) taken from the EXL3 artifact, as strata-glm attributed its dense formats (§4.3). That
says whether the 3-bit experts or the 5-bit dense classes carry the divergence, which option O9
depends on. Results are reported whether they pass or fail.

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
- **Both artifacts:** the import, the supplement and a short generation for turboderp's build; the
  MTP path only for MikeRoz's.
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
| `--vram-headroom-mib N\|auto`, `--ram-headroom-mib N` | auto, 2048 | The sizing law's headroom terms (MoE §4.9-§4.10) |
| `--ssd-lookahead off\|1\|2\|3` | off | Lookahead depth (§11.5); the default changes only if M5 measures a gain on two volumes |
| `--cpu-experts auto\|off\|N`, `--cpu-expert-cores p\|pe` | auto (calibrated), auto | CPU team size; whether E-cores join it (§12.3) |
| `--spec mtp`, `--mtp-expert-frames N\|auto`, `--draft-tokens`, `--lm-head-draft` | off, auto (48) | §16; `--spec mtp` refused without an MTP layer |
| `--ngram-draft-tokens N` | off | Copy proposals |
| `--kv-dtype bf16\|fp8\|int8\|nvfp4` | `bf16` | **Changes output** |
| `--max-context N` | 131072 | Up to 1,048,576; the KV host tier above 128K (§13.6) |
| `--prefill-chunk N`, `--prefill-group N` | auto, 48 | Chunking; experts per prefill group (§17.2) |
| `--host-context-mib N` | 2048 | Prefix Host tier |
| `--reasoning-effort low\|high\|max` | max | Template default; requests may override |
| `--approx-experts off\|ssd\|vram`, `--approx-keep-map DIGITS\|PRESET`, `--approx-min-weight θ`, `--approx-experts-rule residency\|static` | off; Maya's map; none; residency | **Changes output.** The approximate expert mode (§9.8): scope, per-layer keep counts, weight threshold, rule |

Converter: `--recipe glm5_3_flash_exl3` (or an official entry, §7.2) with `--source DIR`;
`--supplement-source zai-org/GLM-5.3-Flash@REV` (required for turboderp's build, refused when
nothing is missing); `--expert-volumes P0,P1,…`, `--expert-volume-weights w0,w1,…`;
`--prefill-permissions a8_dense,a8_experts` (records the gated permissions in the artifact);
`--proposal`. Tools: `tools.artifact.restripe`, `tools.glm5_next.verify_artifact`,
`tools.glm5_next.block_check`, `tools.glm5_next.reference_forward`; the SSD, pinning and CPU E3A16
probes join the maintainer probe suite (the shape of `tools/flash_next_probe/`).

---

## 23. Implementation plan

| # | Milestone | Deliverables | Exit criteria |
|---|---|---|---|
| **M0** | Facts | Both artifacts' config, tokenizer, template, generation config and safetensors headers (not gated; the weights are not needed yet); ExLlamaV3 at 1.4.4 and 1.4.6: tile weight order, stream bit order, Hadamard scale, `out_scales` semantics, MTP trellis shapes; vLLM/SGLang `glm5_next` MTP mathematics; C++ probes: pinning ceiling, striped 0.75 MiB reads with IOCP (`tools/glm5_next_probe/`), **CPU E3A16 at 9 MiB on P- and E-cores**, link, on both machines' hosts | §2, §3, §9.5 and §19 confirmed or revised; open questions of §25.2 closed; §5.3 re-run with the measured CPU cost |
| **M1** | Traces and policy | Route traces recorded by M4's engine on NRS and a broad corpus (code, chat, CJK, tool use, long context); two-level replay of the §10.2 and §10.4 candidates; lookahead recall by depth and width for SSD-tier experts | Policies chosen by the fixed rules; §5.2 replaced by measured hit rates |
| **M2** | Tier extensions | The §8.2 additions in `src/models/offload/` and `src/ops/offloaded_sparse_moe/`: frame side entries, `kMaxSegments` 12, per-volume read queues, landed-unit masks, the `noaux_tc` routing variant, the `exl3_mul1` trait's scaffolding | Qwen4Exp's unit and real tests pass unchanged (the tier is shared and in production; nothing moves); greedy ids identical; tg512 within noise |
| **M3** | Ops | EXL3 codec; `exl3` Linear (decode, prompt and transposed routes); `offloaded_sparse_moe` `exl3_mul1` on the GPU and the CPU (AVX-VNNI); GLM routing; `mhc`; `kda_input_prep` and replay; `dsa_indexer`; `mla_sparse_attention`; per-volume reading | Every §20.1 qualification passes |
| **M4** | Artifacts and functional model | The recipe and both official entries, the supplement fetch, volumes, restripe, `verify_artifact`; Engine, CLI, bench, CausalScoring with one volume and the existing tiers; the two reference forwards; the registry value and engine arm | Every tensor round-trips bit-exactly; §20.2 checks; execution gate (§20.3); the quantization-quality report for both artifacts; first end-to-end numbers, both machines |
| **M5** | Tiers at GLM scale | Striped reading on two volumes, landing ring, partial-record consumption, lookahead, M1's policies, promotion sweep, CPU worker and core-type sweep, warm start, prompt seeding, the unpinned sub-tier where the pin room binds | §20.4 invariance and fault tests; decode with two volumes ≥ 1.15× one; Must decode target |
| **M6** | Decode speed | §9.3 budget kernels and fusions, EXL3 decode ALU check against ExLlamaV3's GEMV, stall-time warming; nsys attribution of one token | Per-kernel times against the budget; anything > 30 % over gets ncu before M7 |
| **M7** | Speculation | MTP drafter (MikeRoz: bounded experts, index sharing, draft head), cost model, n-gram (both) | Default chosen by measurement; never slower than plain beyond noise |
| **M7a** | Approximate mode | §9.8: router rule, scopes, maps, weight threshold, static rule, cache crediting, prefix tags, labelling; band-sensitivity table on the default EXL3 artifact; replayed-residency KL evaluation | Mode off is bit-identical to a build without it; keep map all-8 equals exact; presets meet the §9.8 gates; KL and speed published per preset |
| **M8** | Prefill and long context | Walk with expert groups, E3A16 wide and prompt routes at speed, indexer and attention prompt kernels, KV formats, the KV host tier above 128K, permission gates | Prefill Must target; 128K decode ≥ 85 % of 4K |
| **M9** | Frontend and serving | Tokenizer, template, parser, protocol schema tests, prefix-cache binding | Tokenizer oracle exact; serving schema tests; prefix-cache real tests |
| **M10** | Report | Calibration, head-to-head with Maya, ExLlamaV3 and llama.cpp / KTransformers on the 5090 machine where they run, the §1.3 table filled, the production machine's numbers | Every result reported, favourable or not |

Critical path: M0 → M2 ∥ M3 → M4 → M1 → M5 → M6, then M7/M8; M9 can start once the tokenizer
files are read (they are available now). Rough size: ~27-33k new lines (attention family ~4k, mHC
~1k, KDA consumer ~1.5k, volumes and per-volume reading ~2k, **EXL3 codec, Linear and expert
routes, CPU kernel ~7k**, tier extensions ~1k, frontend ~2.5k, converter and tools including the
reference forwards ~3k, tests ~8k), plus the 126 GB download (≈ 330 GB more for the FP8 teacher)
and GPU time for the campaigns. The offload tier itself is built; nothing is extracted.

---

## 24. Documentation and authority changes

When the implementation lands: a user guide `docs/glm5_3-flash.md`; README models and benchmarks;
`docs/cli.md` and `docs/serving.md` (expert volumes, GLM tool calls, reasoning effort); a
mathematics reference `docs/maintainer/glm5_next-model.md` from §2.2; the EXL3 format and E3A16
arithmetic in [storage layouts](storage-layouts.md) (`exl3` Linear format, `exl3_mul1_expert_v1`,
the vector plane, bank geometry) and [op development](op-development.md) (the E3A16
qualification); [artifact container](artifact-container.md) (expert volumes); [engine
architecture](engine-architecture.md) (the fourth architecture, the tier extensions of §8.2);
[paged KV](paged-kv-cache.md) (latent planes); [hybrid prefix cache](hybrid-prefix-cache-spec.md)
(the GLM binding); the conversion guide (the EXL3 import and the supplement). This file is then
removed.

---

## 25. Risks, open questions and rejected alternatives

### 25.1 Risks

| Risk | Effect | Mitigation |
|---|---|---|
| CPU E3A16 kernel slower than modelled (no AVX-512) | Host term grows: 0.40 → 0.60 ms per expert is 28 → 25 tok/s | M0 probe first, on both hosts; E-cores; DMA share via the split table; the decode model re-run before M5 |
| EXL3 decode ALU-bound on the GPU at T = 1 | GPU term grows beyond §9.3 | M6 against ExLlamaV3's GEMV; state-decode sharing across the gate/up pair |
| ExLlamaV3 format details (tile order, Hadamard scale, `out_scales`) differ between 1.4.4 and 1.4.6 or from the reading in §9.5 | Wrong weights | M0 from source at both versions; the exact codec oracle against ExLlamaV3's own reconstruct (§20.1) |
| §5.3's model is optimistic, as it was by 15-45 % against strata-glm's measured run (§4.3) | Decode ≈ 21-27 tok/s central; the Must target missed at the low end | M4's first end-to-end numbers re-fit the model's overhead terms before M5's sweeps; M6 attributes one token with nsys |
| Hit rates below the central case (no traces yet) | Decode toward 24 tok/s on the 5090 machine | M1 first; policy candidates; the hardware levers of §5.5 in the guide |
| Windows pins less than the plan | Fewer pinned slots | The unpinned CPU-only sub-tier (MoE §4.11); measured in M0 |
| Drive thermal throttling under sustained reads (the probe ran 2-4 s) | SSD term grows | Per-volume telemetry and warnings (§11.7); heatsinks |
| `E:` contention if a volume is put there | Tail latency | Default placement avoids it |
| MTP slower than plain | No speculative gain | Off by default; cost model |
| E3A16 wide route slower than modelled | Prefill below target | M8 measures; the A8 permissions behind the gate |
| Upstream ambiguities (MTP hidden input, tokenizer pattern, top-k ties) | Wrong mathematics | M0 against the checkpoint and vLLM/SGLang; ids against Transformers on short prompts |
| The uncensored build's quality is unmeasured | Output quality | The FP8-teacher report of §20.3; turboderp's build as the measured fallback |
| The pooling supplement's source revision drifts | Wrong tensors for turboderp's build | Pinned revision, shape and value checks, recorded in the conversion report |

### 25.2 Open questions for M0

1. ExLlamaV3's tile weight order, stream bit order, Hadamard scale and `out_scales` semantics at
   1.4.4 and 1.4.6 (the codebook constants were read from `master` on 2026-10-11).
2. The exact MTP input hidden state; the MTP bank's trellis shapes in MikeRoz's files.
3. The tokenizer's split pattern and normalizer; EOS ids from the generation config.
4. How many router values ExLlamaV3's BF16 → FP16 conversion changed (§7.1).
5. The pinning ceiling and the commit limit (page-file size) on this Windows build and driver
   (§10.4).
6. The CPU E3A16 cost per expert on P-cores and E-cores (§12.2), the model's least certain input.

Questions for the owner, not for M0:

7. Whether M4 downloads the FP8 teacher (≈ 330 GB) for the quantization-quality report (§20.3), or
   the report stays limited to the published 0.047 for the base build.

### 25.3 Options (each behind its own measured gate)

| ID | Option | Gate |
|---|---|---|
| O1 | Mixed-bit EXL3 builds (gate/up and down at different bits, e.g. the 2.4 bpw mix): two unit sizes and a 12-offset table per volume | A build the owner chooses; §20.1 for its K |
| O2 | Hot-layer reservations or near-miss credit in the VRAM policy | M1 replay rule (§10.2) |
| O3 | Draft-informed SSD prefetch | M1 recall ≥ 0.5 for SSD-tier experts, M7 gain |
| O4 | FP8 pooled-key scan with FP32 rescoring at > 256K | Exact selection on fixtures; M8 speed |
| O5 | `vq2`/`k4v2` latent profiles | Qualification track |
| O6 | Linux io_uring path for the per-volume queues | Linux users; same tests |
| O7 | orcarouter NVFP4 recipe (Flash-Next's W4A16 at GLM geometry, §7.4) | An owner request; Flash-Next's qualification at GLM geometry |
| O8 | Mirrored volumes with per-unit dynamic routing (strata-glm, §4.3): every drive holds every unit, and each unit read goes to the drive that will deliver it first; adapts to a busy or throttling drive (the page file on `G:`), at 2× the disk (118 GB per drive, which `D:` and `G:` have) | M5 A/B against weighted striping on the NRS corpus, with background load on one drive |
| O9 | Dense classes from the FP8 source as `q8_g32_fp16` (strata-glm's int8-per-32 dense: 0.0105 long-context KL against 0.0066 for BF16) with the EXL3 experts | The §20.3 attribution shows the EXL3 dense classes carry most of the artifact's KL; then the §20.3 permission-style gate against the FP8 teacher; costs ≈ 3.7 GB of VRAM (≈ 400 frames) |

### 25.4 Rejected alternatives

One line each; the rejections that decided the shared machinery are [MoE offload
design](moe-offload-design.md) §4.7, §5.8 and §6.3 and are not repeated here.

| Alternative | Reason |
|---|---|
| ExLlamaV3's FP16 kernels as the arithmetic | Not reproducible across the CPU and GPU or across routes, which placement invariance and prefix-cache exactness need; the int8 tensor-core route has about the same peak as FP16 HGEMM |
| Requantizing the EXL3 weights to NVFP4 or K-quants at import | Changes the weights for no speed gain; the trellis is already smaller |
| Suh/svh inside each record (36 KiB per record) | Records and stripe units stop being powers of two; the vector plane costs 0.43 GiB of pinned RAM instead |
| Storage Spaces / RAID 0 instead of application striping | Its 256 KiB default interleave does not align with the kernels' work units; hides per-drive queues and health; needs admin setup; cannot weight unequal drives. A volume may still sit on a RAID set |
| DirectStorage | Stages through system memory, trellis codes do not compress, needs D3D12 interop: no bandwidth gain over unbuffered reads into pinned memory |
| GPUDirect Storage | Linux and datacenter GPUs only |
| Memory-mapped experts / the page cache | Rejected by the shared tier (MoE §4.7); Flash-Next measured a mapped file's unbuffered reads at a quarter of the unmapped rate (FN §12.3), and llama.cpp measured 111× page amplification |
| Any SSD cache, spill or swap file | Violates the read-only rule; the volumes already hold the exact bytes (MoE §4.7) |
| Zero-copy SM reads of host experts | 10-14 GB/s against 24-27.6 staged (MoE §5.2's measurements on Flash-Next) |
| Expert deferral, dynamic top-k, low-bit miss copies, and expert skipping outside §9.8 | Change the model (§1.4); top-k 6 was rejected on quality by ox-boost. Skipping is offered only as §9.8's opt-in, labelled mode |
| BF16 KDA state | Saves ≈ 0.1 ms of a ~32 ms token for a numerical change |
| Per-read hashing | ≈ 0.15 ms of CPU per record |
| Whole-layer double buffering in prefill | Expert groups free ≈ 300 frames with identical results (§17.2) |
| Folding W_uk into W_qb or W_uv into W_o | Requantizes the weights and reads more bytes per token (§13.1) |

---

## 26. Sources

- Model: [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (config, index, template);
  Transformers `models/glm5_next` (`modeling_glm5_next.py`, `configuration_glm5_next.py`); GLM-5 report
  arXiv 2602.15763.
- Artifacts: [MikeRoz/GLM-5.3-Flash-Uncensored-3.05bpw-h6-exl3](https://huggingface.co/MikeRoz/GLM-5.3-Flash-Uncensored-3.05bpw-h6-exl3)
  and [turboderp/GLM-5.3-Flash-exl3](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3) (branch
  `3.05bpw`): `quantization_config.json`, `model.safetensors.index.json`, `config.json`, read
  2026-10-11; `orcarouter/GLM-5.3-Flash-Uncensored-FP8`; `orcarouter/GLM-5.3-Flash-Uncensored-NVFP4`
  (model card; files gated).
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
