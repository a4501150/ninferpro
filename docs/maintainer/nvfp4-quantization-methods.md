# NVFP4 quantization-method adoption

This reference records which converter-side 4-bit quantization methods ninferpro has evaluated, what
each would demand of the stored NVFP4 representation, and the measured reason none is adopted. The
evaluation was carried out by Fedor Suchkov on the `kido5217/Infernix` fork
([issue #3](https://github.com/kido5217/Infernix/issues/3), map
[#25](https://github.com/kido5217/Infernix/issues/25)) against this repository's converter and
stored formats; the raw artifacts live on that fork's unmerged branches named in
[Provenance](#provenance).

Scope: weights and activations quantized to NVFP4 (E2M1 codes, E4M3FN group scales) by the
converter, and the runtime inputs such quantization would need. Whole-model speed, scheduling and
artifact framing are out of scope.

## What the representation allows

The stored `nvfp4` weight format is E2M1 4-bit codes, **one E4M3FN scale per 16-element K-group**,
and **one FP32 divisor for the whole matrix** (`docs/maintainer/tensor-formats.md` §3.3), laid out
and swizzled as `docs/maintainer/storage-layouts.md` §4 fixes. The tensor record rejects unknown
fields (`tools/artifact/schema.py`) and the numeric-format registry is closed
(`tools/artifact/formats.py`). The kernels issue
`mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64…ue4m3` and assert 16/64/128
geometry (`src/ops/common/mma.cuh`, `src/ops/linear/nvfp4/nvfp4_geometry.h`).

Two consequences decide every placement below:

- **A different scale *selection* rule is converter-only.** It rewrites the same words — the shape of
  the existing `nvfp4_mse` method (`tools/convert/quantization/nvfp4.py`,
  `tools/convert/methods.py`). No kernel, layout or format change.
- **A second scale per group, or a per-channel *activation* scale, is not representable.**
  The first needs a new format, layout and kernels; the second has no field at all: the activation
  quantizer takes one scalar divisor (`src/ops/linear/nvfp4/nvfp4_a4_quantize.cuh`), the artifact's
  `uses` record admits only that scalar (`tools/convert/recipe.py`, `include/infernix/ops/weight_input.h`),
  and the fused wide route requires the routed bank's contributing inputs to share one divisor
  (`src/ops/offloaded_sparse_moe/cuda/wide_expert.h`; the narrow route computes unequal gate/up
  scales exactly, so the requirement is the wide route's).

## Evaluated methods

| Method | Weight side | Activation side | Tier | Calibration |
|---|---|---|---|---|
| **OSFP4** (arXiv 2610.08231) | diagonal smoothing `W A⁻¹` folds into the stored weights; the calibrated E4M3 block-scale selection rewrites the scales | `A X` — a per-input-channel multiply, applied explicitly where it cannot fold into a preceding normalization | **runtime input** for the transform; **converter-only** for selection alone | corpus + forward passes (a second-moment estimate over a calibration set) |
| **DualQuant** (arXiv 2610.13767) | row scale `α` folded into the dequantization scale | column scale `β` as `X Diag(β)⁻¹`, applied inside the activation quantizer | **runtime input** | none — strictly calibration-free (no activation data, statistics or forward pass) |

Neither needs a second scale inside a 16-element group, so neither is a format change. Both need a
per-channel activation vector on the routed expert inputs: the experts' input is the
hyper-connection collapse of a nonlinearly gated norm output, and the down projection's input is the
product `silu(g)·u`, so a per-channel diagonal cannot ride the preceding normalization
(`tools/flash_next/reference.py`, `hc_mix`).

## Measurement

The protocol was pre-registered before any measurement
([#29](https://github.com/kido5217/Infernix/issues/29)): 25 real matrices from Unsloth's BF16 GGUF of
Qwen3.8-Flash-Next, an exact FP64 oracle, the repo's own quantizer and decoder, and a bar of
**≥ 22/25 wins, median relative GEMM-error reduction ≥ 10 %, no matrix worse than +2 %** against the
better of `nvfp4_absmax` and `nvfp4_mse` per matrix (relative Frobenius error of
`dequant(W) · dequant_A4(T(X))` against the unquantized FP64 `W · X`, weight and activation
quantization scored together; an offline CPU measurement on single matrices, not a whole-model one).

Nothing cleared the bar:

| Arm | Wins (of 25) | Median reduction | Verdict |
|---|---:|---:|---|
| OSFP4 smoothing (transform + the repo's MSE scale rule) | 25/25 | 3.94 % | not cleared |
| OSFP4 calibrated selection (no smoothing) | 25/25 | 2.57 % | not cleared |
| DualQuant | 0/25 | −3.75 % | not cleared |

`nvfp4_mse` is the better control on every matrix. The published gains do not transfer because both
papers report against round-to-nearest or GPTQ baselines while this converter's control is already an
exhaustive MSE scale search. Two results stand on their own: DualQuant's row scale `α` cannot be
absorbed by the per-16 E4M3FN scale (folding it rewrites 100 % of scale words at a median 2.46e-2
relative L2 cost), and DualQuant loses even with the fold disabled (5/25 wins), so the method fails
for two separable reasons.

## Disposition

- **Neither method is adopted**, and no converter or runtime change follows from this evaluation;
  nothing cleared the pre-registered bar.
- **Converter-only scale selection is untested at model scale.** OSFP4's selection rule without
  smoothing won 25/25 at 2.57 % and needs no runtime input, kernel or format change (a `quantize_*`
  function and one `METHODS` entry, like `nvfp4_mse`), but that number was produced under a bar set
  for a different question. Reopening it needs a whole-model KL and perplexity measurement per
  [model quality](model-quality.md), not the proxy number; the follow-up that would pre-register its
  own threshold ([#33](https://github.com/kido5217/Infernix/issues/33)) was closed without being run.
- **The runtime per-channel activation scale** that both methods require has no representation today
  (scalar `activation_input_divisor`, closed format registry, one shared divisor on the fused wide
  route); its design request
  ([#34](https://github.com/kido5217/Infernix/issues/34)) was closed without a design.

## Provenance

Papers: OSFP4, arXiv 2610.08231; DualQuant, arXiv 2610.13767. The evaluation, its pre-registration
and raw prototype output live on the `kido5217/Infernix` fork (issues
[#26](https://github.com/kido5217/Infernix/issues/26)–[#31](https://github.com/kido5217/Infernix/issues/31),
branch `proto/nvfp4-transform-prototype`), not merged there or here; the prototype's numbers were
independently recomputed from its raw output before publication.

## Related

- [Model-level numerical quality](model-quality.md) — how quality is judged at model scale (KL to a
  16-bit-activation reference, perplexity noise rules); the acceptance any future adoption must meet.
- [Numeric formats](tensor-formats.md) and [Storage layouts](storage-layouts.md) — the represented
  values and byte geometry this reference reasons about.
- [Weight conversion](../weight-conversion.md) — recipes, methods and the `METHODS` surface a
  converter-only selection change would extend.
