"""Qwen3.5 family GGUF name maps and export transforms (HF -> GGUF).

`tools.convert.sources.gguf` reads a GGUF file and `HFAliasSource` there exposes its tensors
under the HF Safetensors names a recipe's source tensors are keyed by, through an explicit
{hf_name: gguf_name} map plus optional per-tensor value transforms. This module holds the
Qwen3.5 family's maps: `standard_dense_name_map`, `qwen35_linear_attention_name_map`, and
`qwen35_mtp_name_map` (the vision tower's map, shared with the other Qwen family, lives in
tools/convert/qwen_common.py).

The maps were each checked against real checkpoint/GGUF pairs on this machine -- the
Swift-Qwen3.8-27B HF checkpoint (/mnt/storage/models/swift-qwen3.8-27b, byte-identical tensor
names/shapes to official Qwen/Qwen3.8-27B), a standalone MTP checkpoint and its matching
MTP-only GGUF, and the matching mmproj vision GGUF, against Q5_0- and Q8_0-quantized GGUF
exports of the same architecture -- comparing every mapped tensor's dequantized values back
to the HF original. Unquantized companions (F32 norms, A_log, dt_bias, conv1d) matched
exactly; quantized ones matched to within that quant level's own noise (a few percent
relative error, uniform across tensors, confirming the mapping and any reorder/shift are
structurally correct rather than merely close by chance). Together they cover: dense
attention/MLP/norms for full-attention layers, Qwen3.5's fused-QKV/Z/alpha/beta GDN layers,
and the MTP extra layer (a separate GGUF file in every case seen). None of them cover
mixture-of-experts routing/shared-expert tensors -- extend the maps for those using
GGUFSource.list_tensors() to see exactly what a given file calls things, and verify any
addition the same way (an unquantized companion tensor if you can find one, or a Q8_0/F16
GGUF for a tighter statistical check). DFlash2 needs no GGUF-specific code in the cases seen
so far -- see docs/weight-conversion.md.
"""

from __future__ import annotations

from typing import Callable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .sources.gguf import Transform


def _shift_norm(values: torch.Tensor, shape: tuple[int, ...]) -> tuple[torch.Tensor, tuple[int, ...]]:
    # Verified against /mnt/storage/models/swift-qwen3.8-27b +
    # .../qwen3.8/Qwen3.8-27B-Q5_0.gguf: every "*.norm.weight" tensor except
    # linear_attn.norm.weight is stored on export as (HF weight + 1), an
    # exact match confirmed on input_layernorm, post_attention_layernorm,
    # q_norm, k_norm and the final model.norm.
    return values - 1.0, shape


def standard_dense_name_map(
    num_hidden_layers: int,
    *,
    tie_word_embeddings: bool = False,
    text_prefix: str = "model.language_model.",
) -> tuple[dict[str, str], dict[str, Transform], dict]:
    """HF -> GGUF (name_map, transforms, shapes) for dense attention/MLP/norm
    tensors -- see sources.gguf.HFAliasSource for what each dict is for (`shapes` is
    always empty here: none of these tensors change rank).

    Matches the standard llama.cpp tensor roles used for Llama/Qwen2/Qwen3
    -family dense decoders, verified end-to-end (real values, not just
    names) against a real Qwen3.8-27B checkpoint pair on this machine -- see
    the module docstring. Covers only full-attention layers with a plain
    gate/up/down MLP; extend the returned maps for GDN (see
    qwen35_linear_attention_name_map), MoE routing/shared experts, or MTP
    tensors.

    `text_prefix` defaults to "model.language_model." because that is what
    Qwen3_5ForConditionalGeneration (vision-enabled) checkpoints use, verified
    against the real checkpoint above. A text-only Qwen3_5ForCausalLM
    checkpoint may use plain "model." instead -- check your checkpoint's
    model.safetensors.index.json weight_map (or GGUFSource.list_tensors())
    before trusting this default for that variant.
    """
    name_map = {
        text_prefix + "embed_tokens.weight": "token_embd.weight",
        text_prefix + "norm.weight": "output_norm.weight",
    }
    transforms: dict[str, Transform] = {text_prefix + "norm.weight": _shift_norm}
    if not tie_word_embeddings:
        name_map["lm_head.weight"] = "output.weight"
    for i in range(num_hidden_layers):
        hf = f"{text_prefix}layers.{i}."
        gg = f"blk.{i}."
        name_map.update(
            {
                hf + "input_layernorm.weight": gg + "attn_norm.weight",
                hf + "self_attn.q_proj.weight": gg + "attn_q.weight",
                hf + "self_attn.q_norm.weight": gg + "attn_q_norm.weight",
                hf + "self_attn.k_proj.weight": gg + "attn_k.weight",
                hf + "self_attn.k_norm.weight": gg + "attn_k_norm.weight",
                hf + "self_attn.v_proj.weight": gg + "attn_v.weight",
                hf + "self_attn.o_proj.weight": gg + "attn_output.weight",
                hf + "post_attention_layernorm.weight": gg + "post_attention_norm.weight",
                hf + "mlp.gate_proj.weight": gg + "ffn_gate.weight",
                hf + "mlp.up_proj.weight": gg + "ffn_up.weight",
                hf + "mlp.down_proj.weight": gg + "ffn_down.weight",
            }
        )
        for norm_name in (
            hf + "input_layernorm.weight",
            hf + "self_attn.q_norm.weight",
            hf + "self_attn.k_norm.weight",
            hf + "post_attention_layernorm.weight",
        ):
            transforms[norm_name] = _shift_norm
    return name_map, transforms, {}


def reorder_linear_attention_v_heads(
    values: torch.Tensor,
    shape: tuple[int, ...],
    dim: int,
    num_row_groups: int,
    group_size: int,
    head_dim: int,
) -> torch.Tensor:
    """Swap V-head grouping between HF's "grouped by K head" order and
    ggml's tiled broadcast order.

    Ports `_LinearAttentionVReorderBase._reorder_v_heads` from
    /mnt/storage/llama.cpp/conversion/qwen.py: reshape to
    [..., num_row_groups, group_size, head_dim, ...], swap the two head
    axes, reshape back. This is a plain transpose of a (num_row_groups,
    group_size) matrix of head-blocks, so it is only its own inverse when
    num_row_groups == group_size. In general (Qwen3.8-27B has 16 K-heads and
    3 V-heads per K-head, so it is not) the inverse call must swap the two
    count arguments: to undo `reorder(x, dim, K, V, d)`, call
    `reorder(y, dim, V, K, d)`. Verified exactly (down to float32 rounding)
    against real HF vs. GGUF tensor pairs for both the forward (export) and
    swapped-argument inverse (import) directions -- see the module docstring.
    """
    tensor = values.reshape(shape)
    if dim < 0:
        dim += len(shape)
    new_shape = (
        list(shape[:dim]) + [num_row_groups, group_size, head_dim] + list(shape[dim + 1 :])
    )
    tensor = tensor.reshape(*new_shape)
    perm = list(range(len(new_shape)))
    perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
    return tensor.permute(*perm).contiguous().reshape(*shape)


def qwen35_linear_attention_name_map(
    layer_index: int,
    *,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    text_prefix: str = "model.language_model.",
) -> tuple[dict[str, str], dict[str, Transform], dict[str, Callable[[tuple[int, ...]], tuple[int, ...]]]]:
    """HF -> GGUF (name_map, transforms, shapes) for one Qwen3.5 GDN (linear
    attention) layer -- see sources.gguf.HFAliasSource for what each dict is for.

    Ported from /mnt/storage/llama.cpp/conversion/qwen.py's
    _LinearAttentionVReorderBase/Qwen3NextModel and gguf-py/gguf/constants.py's
    QWEN35 TENSOR_NAMES, then verified end-to-end against a real Qwen3.8-27B
    checkpoint pair on this machine (see the module docstring): every name,
    the A_log `-exp()` transform, the norm.weight exclusion from the +1 shift
    applied elsewhere, and the V-head reorder (with swapped arguments for
    this import direction -- see reorder_linear_attention_v_heads) all
    reproduced the original HF tensor exactly for unquantized companions
    (A_log, dt_bias, conv1d are F32 in the GGUF checked) and within the
    source GGUF's own quantization noise for quantized ones (in_proj_qkv,
    in_proj_z, in_proj_a, in_proj_b, out_proj).
    """
    hf = f"{text_prefix}layers.{layer_index}.linear_attn."
    gg = f"blk.{layer_index}."
    name_map = {
        hf + "in_proj_qkv.weight": gg + "attn_qkv.weight",
        hf + "in_proj_z.weight": gg + "attn_gate.weight",
        hf + "in_proj_a.weight": gg + "ssm_alpha.weight",
        hf + "in_proj_b.weight": gg + "ssm_beta.weight",
        hf + "conv1d.weight": gg + "ssm_conv1d.weight",
        hf + "A_log": gg + "ssm_a",
        hf + "dt_bias": gg + "ssm_dt.bias",
        hf + "norm.weight": gg + "ssm_norm.weight",  # no +1 shift: excluded by the exporter
        hf + "out_proj.weight": gg + "ssm_out.weight",
    }

    # llama.cpp's exporter squeezes the singleton middle dim off Conv1d's
    # PyTorch (out_channels, 1, kernel_size) weight before writing it; insert
    # it back so describe() matches the shape the recipe captured from HF.
    shapes = {hf + "conv1d.weight": lambda native: (native[0], 1, native[1])}

    num_v_per_k = num_v_heads // num_k_heads
    q_dim = head_k_dim * num_k_heads
    k_dim = head_k_dim * num_k_heads

    def reorder_v(rows: torch.Tensor, head_dim: int, dim: int = 0) -> torch.Tensor:
        return reorder_linear_attention_v_heads(rows, rows.shape, dim, num_v_per_k, num_k_heads, head_dim)

    if num_v_per_k <= 1:
        return name_map, {hf + "A_log": _inverse_a_log}, shapes

    def qkv_transform(values, shape):
        n, k = shape
        matrix = values.reshape(n, k)
        q, kk, v = matrix[:q_dim], matrix[q_dim : q_dim + k_dim], matrix[q_dim + k_dim :]
        v = reorder_v(v, head_v_dim)
        result = torch.cat([q, kk, v], dim=0)
        return result.reshape(-1), shape

    def rows_transform(head_dim):
        def apply(values, shape):
            return reorder_v(values.reshape(shape), head_dim).reshape(-1), shape

        return apply

    def out_proj_transform(values, shape):
        return reorder_v(values.reshape(shape), head_v_dim, dim=1).reshape(-1), shape

    def conv1d_transform(values, shape):
        n, k = shape
        matrix = values.reshape(n, k)
        qk_channels = head_k_dim * num_k_heads * 2
        qk_part, v_part = matrix[:qk_channels], matrix[qk_channels:]
        v_part = reorder_v(v_part, head_v_dim)
        return torch.cat([qk_part, v_part], dim=0).reshape(-1), shape

    transforms = {
        hf + "in_proj_qkv.weight": qkv_transform,
        hf + "in_proj_z.weight": rows_transform(head_v_dim),
        hf + "in_proj_a.weight": rows_transform(1),
        hf + "in_proj_b.weight": rows_transform(1),
        hf + "out_proj.weight": out_proj_transform,
        hf + "conv1d.weight": conv1d_transform,
        hf + "dt_bias": rows_transform(1),
        hf + "A_log": _inverse_a_log_reordered(reorder_v),
    }
    return name_map, transforms, shapes


def _inverse_a_log(values: torch.Tensor, shape: tuple[int, ...]) -> tuple[torch.Tensor, tuple[int, ...]]:
    # GGUF stores -exp(A_log); undo it (no V-head reorder when num_v_per_k == 1).
    return torch.log(-values), shape


def _inverse_a_log_reordered(reorder_v):
    def apply(values: torch.Tensor, shape: tuple[int, ...]) -> tuple[torch.Tensor, tuple[int, ...]]:
        reordered = reorder_v(values.reshape(shape), 1)
        return torch.log(-reordered).reshape(-1), shape

    return apply


def qwen35_mtp_name_map(
    gguf_layer_index: int, *, hf_prefix: str = "mtp."
) -> tuple[dict[str, str], dict[str, Transform], dict]:
    """HF -> GGUF (name_map, transforms, shapes) for the Qwen3.5 MTP (next-token
    prediction) extra layer.

    llama.cpp exports the MTP block as one more transformer layer appended
    after the base model's real layers (here layer 64, since num_hidden_layers
    is 64), reusing the ordinary full-attention/MLP/norm tensor roles
    standard_dense_name_map already covers, plus four MTP-specific tensors
    (nextn.eh_proj/enorm/hnorm/shared_head_norm) renamed from HF's
    fc/pre_fc_norm_embedding/pre_fc_norm_hidden/norm.

    `hf_prefix` is "mtp." when the MTP tensors are fused into the main
    checkpoint's state dict (e.g. swift-qwen3.8-27b's "mtp.fc.weight", ...)
    and "" for a standalone MTP-only checkpoint whose tensors have no prefix
    at all (e.g. a repo containing only "fc.weight", "layers.0....").

    Verified end-to-end against a real standalone MTP checkpoint and its
    matching MTP-only GGUF export on this machine (see the module
    docstring's sibling text/GDN validation): unlike the base model's
    layers, NONE of the MTP layer's norm.weight tensors get the +1 shift
    standard_dense_name_map applies elsewhere -- input_layernorm, q_norm,
    k_norm, post_attention_layernorm, and the three nextn norm-ish tensors
    all matched their HF originals exactly with no transform at all. (The
    exporter appears to apply that shift only while iterating the base
    model's own layer loop, before the MTP mixin's tensors are renamed and
    appended -- but that is inference from the observed behavior, not from
    reading that exact code path; trust the validated "no shift" result over
    the guess at why.)
    """
    gg = f"blk.{gguf_layer_index}."
    hf = hf_prefix
    name_map = {
        hf + "fc.weight": gg + "nextn.eh_proj.weight",
        hf + "pre_fc_norm_embedding.weight": gg + "nextn.enorm.weight",
        hf + "pre_fc_norm_hidden.weight": gg + "nextn.hnorm.weight",
        hf + "norm.weight": gg + "nextn.shared_head_norm.weight",
        hf + "layers.0.input_layernorm.weight": gg + "attn_norm.weight",
        hf + "layers.0.self_attn.q_proj.weight": gg + "attn_q.weight",
        hf + "layers.0.self_attn.q_norm.weight": gg + "attn_q_norm.weight",
        hf + "layers.0.self_attn.k_proj.weight": gg + "attn_k.weight",
        hf + "layers.0.self_attn.k_norm.weight": gg + "attn_k_norm.weight",
        hf + "layers.0.self_attn.v_proj.weight": gg + "attn_v.weight",
        hf + "layers.0.self_attn.o_proj.weight": gg + "attn_output.weight",
        hf + "layers.0.post_attention_layernorm.weight": gg + "post_attention_norm.weight",
        hf + "layers.0.mlp.gate_proj.weight": gg + "ffn_gate.weight",
        hf + "layers.0.mlp.up_proj.weight": gg + "ffn_up.weight",
        hf + "layers.0.mlp.down_proj.weight": gg + "ffn_down.weight",
    }
    return name_map, {}, {}
