"""Glm5Next (GLM-5.3-Flash) architecture adapter: mathematical config, source axes and the routed
expert banks.

The mathematics is the upstream ``glm5_next`` text model (docs/maintainer/glm5_3-flash-design.md
§2): Kimi Delta Attention (KDA) and NoPE sparse MLA blocks inside four mHC residual streams, a DSA
indexer with k-pool compression on every MLA layer, dense SwiGLU layers then a 288-expert MoE with
``noaux_tc`` routing and one ungated shared expert, every MLP clamped at ``swiglu_limit``. Recipes
and quantization assignments live in official_recipes. The checkpoint's vision tower (a 24-block
DiT-style encoder with a SwiGLU patch merger) is stored as the ``vision`` component, BF16 as the
checkpoint keeps it, with its processor config; ``vision_config`` reads its record and
_Builder.vision binds its parameters.

This module owns its own builder and draft-config reading (the Qwen families' live in
qwen_common, which a GLM module does not import).
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch

from .config_values import _f32, _fixed, _positive
from .model import Model, Parameter
from .resources import load_resources
from .sources.compressed_tensors import matrix_source
from .sources.logical import LogicalSource, select_rows, transpose_source
from .sources.nvfp4_banks import ExpertBankSource
from .sources.safetensors import SafetensorsSource, tensor_source

ARCHITECTURES = ("Glm5NextForConditionalGeneration",)
_LAYER_KINDS = {"linear_attention": "linear_attention", "deepseek_sparse_attention": "sparse_attention"}


def is_glm5_next(config: Mapping) -> bool:
    architectures = config.get("architectures")
    return isinstance(architectures, list) and any(a in ARCHITECTURES for a in architectures)


def text_config(source: dict) -> dict:
    architectures = source.get("architectures")
    if not isinstance(architectures, list) or len(architectures) != 1 or architectures[0] not in ARCHITECTURES:
        raise ValueError(f"unsupported Glm5Next architecture {architectures!r}")
    raw = source.get("text_config", source)
    for key, value in (
        ("hidden_act", "silu"),
        ("attention_bias", False),
        ("tie_word_embeddings", False),
        ("norm_topk_prob", True),
        ("scoring_func", "sigmoid"),
        ("topk_method", "noaux_tc"),
        ("n_group", 1),
        ("topk_group", 1),
        ("n_shared_experts", 1),
        ("qk_rope_head_dim", 0),
        ("mhc", True),
        ("mla_use_nope", True),
        ("index_kpool_compress", True),
        ("index_kpool_always_select_tail", True),
        ("moe_router_dtype", "float32"),
    ):
        _fixed(raw, key, value, "text")
    result = {"architectures": ["Glm5NextForCausalLM"], "model_type": "glm5_next_text"}
    for key in (
        "hidden_size", "vocab_size", "num_hidden_layers", "max_position_embeddings", "intermediate_size",
        "moe_intermediate_size", "n_routed_experts", "num_experts_per_tok", "first_k_dense_replace",
        "num_attention_heads", "q_lora_rank", "kv_lora_rank", "qk_nope_head_dim", "v_head_dim",
        "index_n_heads", "index_head_dim", "index_topk", "index_kpool", "hc_mult", "hc_sinkhorn_iters",
    ):
        result[key] = _positive(raw.get(key), "text." + key)
    if result["num_experts_per_tok"] > result["n_routed_experts"]:
        raise ValueError("selected experts exceed expert count")
    if result["index_topk"] % result["index_kpool"]:
        raise ValueError("index_topk must be divisible by index_kpool")
    _fixed(raw, "qk_head_dim", result["qk_nope_head_dim"], "text")
    _fixed(raw, "num_key_value_heads", result["num_attention_heads"], "text")
    result["tie_word_embeddings"] = False
    for key in ("rms_norm_eps", "hc_eps", "routed_scaling_factor", "swiglu_limit"):
        result[key] = _f32(raw.get(key), "text." + key)
    layers = result["num_hidden_layers"]
    kinds = raw.get("layer_types")
    if not isinstance(kinds, list) or len(kinds) != layers or any(k not in _LAYER_KINDS for k in kinds):
        raise ValueError("text.layer_types must name a linear or sparse attention block for every layer")
    result["layer_types"] = [_LAYER_KINDS[k] for k in kinds]
    mlps = raw.get("mlp_layer_types")
    first = result["first_k_dense_replace"]
    if mlps != ["dense"] * first + ["sparse"] * (layers - first):
        raise ValueError("text.mlp_layer_types must be first_k_dense_replace dense layers, then sparse")
    if raw.get("indexer_types") != ["full"] * layers:
        raise ValueError("every sparse attention layer must run its own (full) indexer")
    linear = raw.get("linear_attn_config")
    if not isinstance(linear, dict):
        raise ValueError("text.linear_attn_config must be an object")
    result["linear_num_heads"] = _positive(linear.get("num_heads"), "text.linear_attn_config.num_heads")
    result["linear_head_dim"] = _positive(linear.get("head_dim"), "text.linear_attn_config.head_dim")
    result["linear_conv_kernel_dim"] = _positive(
        linear.get("short_conv_kernel_size"), "text.linear_attn_config.short_conv_kernel_size")
    bound = linear.get("gate_lower_bound")
    if type(bound) not in (float, int) or not -5.0 <= bound < 0.0:
        raise ValueError("text.linear_attn_config.gate_lower_bound must lie in [-5, 0)")
    result["linear_lower_bound"] = -_f32(-float(bound), "text.linear_attn_config.gate_lower_bound")
    kda = [i for i, kind in enumerate(result["layer_types"]) if kind == "linear_attention"]
    if linear.get("kda_layers") != kda:
        raise ValueError("linear_attn_config.kda_layers disagrees with layer_types")
    eos = raw.get("eos_token_id")
    eos = eos if isinstance(eos, list) else [eos]
    if not eos or any(type(v) is not int or not 0 <= v < result["vocab_size"] for v in eos):
        raise ValueError("text.eos_token_id must be token ids")
    result["eos_token_id"] = list(eos)
    mtp = raw.get("num_nextn_predict_layers", 0)
    if mtp not in (0, 1):
        raise ValueError("text.num_nextn_predict_layers must be 0 or 1")
    result["mtp_layers"] = mtp
    return result


def draft_config(raw: dict, target: dict) -> dict:
    """A DFlash2 drafter's config, checked against its GLM target."""

    backend = "dflash2"
    for key, value in (
        ("architectures", ["DFlash2DraftModel"]),
        ("model_type", "qwen3"),
        ("hidden_act", "silu"),
        ("attention_bias", False),
        ("is_causal", False),
        ("hidden_size", target["hidden_size"]),
        ("vocab_size", target["vocab_size"]),
        ("num_target_layers", target["num_hidden_layers"]),
    ):
        _fixed(raw, key, value, backend)
    result = {"architectures": ["DFlash2DraftModel"], "model_type": "qwen3"}
    for key in ("intermediate_size", "num_attention_heads", "num_key_value_heads", "head_dim", "num_hidden_layers",
                "max_position_embeddings"):
        result[key] = _positive(raw.get(key), backend + "." + key)
    if result["num_attention_heads"] % result["num_key_value_heads"] or result["head_dim"] % 2:
        raise ValueError("dflash2: invalid attention geometry")
    result["rms_norm_eps"] = _f32(raw.get("rms_norm_eps", 1e-6), backend + ".rms_norm_eps")
    rope = raw.get("rope_parameters") or {}
    _fixed(rope, "rope_type", "default", backend + ".rope_parameters")
    result["rope_parameters"] = {"rope_theta": _f32(rope.get("rope_theta"), backend + ".rope_theta")}
    layers = raw.get("layer_types")
    if (not isinstance(layers, list) or len(layers) != result["num_hidden_layers"]
            or any(k not in ("sliding_attention", "full_attention") for k in layers)):
        raise ValueError("dflash2: invalid layer_types")
    result["layer_types"] = list(layers)
    if "sliding_attention" in layers:
        result["sliding_window"] = _positive(raw.get("sliding_window"), backend + ".sliding_window")
    draft = raw.get("dflash_config", {})
    taps = draft.get("target_layer_ids")
    if (not isinstance(taps, list) or not taps or len(set(taps)) != len(taps)
            or any(type(i) is not int or not 0 <= i < target["num_hidden_layers"] for i in taps)):
        raise ValueError("dflash2: invalid target_layer_ids")
    mask = draft.get("mask_token_id")
    if type(mask) is not int or not 0 <= mask < target["vocab_size"]:
        raise ValueError("dflash2: mask token outside target embedding")
    output = {"target_layer_ids": list(taps), "mask_token_id": mask}
    for key in ("conv_kernel_size", "conv_group_size", "selector_rank", "selector_top_k"):
        output[key] = _positive(draft.get(key), backend + "." + key)
    if target["hidden_size"] % output["conv_group_size"]:
        raise ValueError("DFlash2 hidden width must be divisible by conv_group_size")
    for scope in (raw, draft):
        for key, expected in (("sample_from_anchor", False), ("input_embedding_scale", 1.0),
                              ("output_multiplier", 1.0)):
            _fixed(scope, key, expected, backend)
        if scope.get("final_logit_softcapping") not in (None, 0, 0.0):
            raise ValueError("DFlash2 final_logit_softcapping must be disabled")
    result["dflash_config"] = output
    return result


def vision_config(source: dict, text: dict) -> dict:
    """The vision tower's config record, checked against the checkpoint's ``vision_config``.

    Carries the media token ids (image/video start/end/placeholder) the engine needs beside the
    tower's geometry; the tower's rotary embedding is the family's axial 2D kind (no stored
    tables), so the record names it and keeps the checkpoint's theta when it declares one."""

    raw = source.get("vision_config")
    if not isinstance(raw, dict):
        raise ValueError("Glm5Next vision requires vision_config")
    for key, value in (
        ("model_type", "glm5_next_vision"),
        ("hidden_act", "silu"),
        ("attention_bias", True),
        ("attention_dropout", 0.0),
        ("in_channels", 3),
        ("temporal_patch_size", 2),
    ):
        _fixed(raw, key, value, "vision")
    result = {"model_type": "glm5_next_vision", "in_channels": 3, "temporal_patch_size": 2}
    for key in ("depth", "hidden_size", "num_heads", "image_size", "patch_size", "spatial_merge_size",
                "intermediate_size", "out_hidden_size", "projection_intermediate_size"):
        result[key] = _positive(raw.get(key), "vision." + key)
    if result["hidden_size"] % result["num_heads"]:
        raise ValueError("vision.hidden_size must be divisible by vision.num_heads")
    if result["image_size"] % result["patch_size"]:
        raise ValueError("vision.image_size must be a multiple of vision.patch_size")
    result["rms_norm_eps"] = _f32(raw.get("rms_norm_eps"), "vision.rms_norm_eps")
    result["swiglu_limit"] = _f32(raw.get("swiglu_limit"), "vision.swiglu_limit")
    rope = raw.get("rope_parameters")
    if rope is not None:
        if not isinstance(rope, dict):
            raise ValueError("vision.rope_parameters must be an object")
        _fixed(rope, "rope_type", "axial", "vision.rope_parameters")
        if "rope_theta" in rope:
            result["rope_theta"] = _f32(rope["rope_theta"], "vision.rope_parameters.rope_theta")
    result["rope_parameters"] = {"rope_type": "axial",
                                 **({"rope_theta": result["rope_theta"]} if "rope_theta" in result else {})}
    vocab = text["vocab_size"]
    for key in ("image_token_id", "image_start_token_id", "image_end_token_id",
                "video_token_id", "video_start_token_id", "video_end_token_id"):
        token = source.get(key)
        if type(token) is not int or not 0 <= token < vocab:
            raise ValueError(f"{key} must be a token id inside the vocabulary")
        result[key] = token
    return result


def _check_processor(value: dict, config: dict, path) -> None:
    """The checkpoint's processor_config.json resource must describe the tower the vision record
    binds: one image and one video processor, each agreeing with the geometry and holding a
    normalization pair and a bounded token budget."""

    for role in ("image_processor", "video_processor"):
        proc = value.get(role)
        if not isinstance(proc, dict):
            raise ValueError(f"{path}: missing {role}")
        for field, config_field in (("patch_size", "patch_size"),
                                    ("temporal_patch_size", "temporal_patch_size"),
                                    ("merge_size", "spatial_merge_size")):
            if proc.get(field) != config[config_field]:
                raise ValueError(f"{path}: {role}.{field} differs from the vision config")
        for field in ("image_mean", "image_std"):
            if not isinstance(proc.get(field), list) or len(proc[field]) != config["in_channels"]:
                raise ValueError(f"{path}: {role}.{field} must hold one value per input channel")
        for field in ("min_image_tokens", "max_image_tokens"):
            if type(proc.get(field)) is not int or proc[field] <= 0:
                raise ValueError(f"{path}: {role}.{field} must be a positive integer")
        if proc["min_image_tokens"] > proc["max_image_tokens"]:
            raise ValueError(f"{path}: {role}.min_image_tokens exceeds max_image_tokens")


class _Builder:
    """Parameters of one fixed checkpoint (no additional sources), with packing groups."""

    def __init__(self, model: Model):
        self.model = model

    def add(self, name, store, source_name, shape, *, inputs=(), direct="bf16", source_shape=None, rows=None,
            transpose=None, offset=0):
        shape = tuple(shape)
        original = shape if source_shape is None else tuple(source_shape)

        def factory(selected, format=None):
            if selected is not store:
                raise ValueError(f"{name}: Glm5Next converts from its single checkpoint")
            if transpose is not None:
                if format is not None:
                    raise ValueError(f"{name}: transpose source requires value access")
                return transpose_source(tensor_source(selected, source_name, original), transpose, shape)
            if len(original) == 2:
                ref = matrix_source(selected, source_name, original, format)
                return select_rows(ref, rows) if rows is not None else ref
            if format is not None:
                raise ValueError(f"{name}: provide an explicit encoded source mapping")
            return tensor_source(selected, source_name, shape, offset=offset, source_shape=original)

        self.model.add(Parameter(name, shape, factory(store), factory, tuple(inputs), direct,
                                 residency=name.split("/", 1)[0]))

    def kda_convolution(self, name, store, source_names, taps, width):
        """The q, k and v depthwise kernels as one (taps, 3 * width) tensor, channels in the fused
        projection's q | k | v order: one convolution state and one causal_conv1d_silu_split call
        serve all three (each stored kernel is (width, 1, taps), oldest tap first)."""

        stored = (taps, len(source_names) * width)

        def factory(selected, format=None):
            if selected is not store:
                raise ValueError(f"{name}: Glm5Next converts from its single checkpoint")
            if format is not None:
                raise ValueError(f"{name}: the convolution is stored BF16")
            parts = [tensor_source(selected, n, (width, 1, taps)) for n in source_names]

            def read(begin, end):
                kernels = [part.values().reshape(width, taps).T for part in parts]
                return torch.cat(kernels, dim=1).contiguous().reshape(-1)[begin:end]

            return LogicalSource(stored, f"concat({','.join(source_names)})", read)

        self.model.add(Parameter(name, stored, factory(store), factory, (), "bf16",
                                 residency=name.split("/", 1)[0]))

    def latent_up(self, name, store, source_name, heads, dn, dv, kv_rank, *, key):
        """One per-head block of kv_b_proj as the absorbed MLA Ops read it: the key blocks
        transposed to (heads, kv_rank, dn), the value blocks kept as (heads, dv, kv_rank)."""

        width, begin = (dn, 0) if key else (dv, dn)
        stored = (heads, kv_rank, dn) if key else (heads, dv, kv_rank)

        def factory(selected, format=None):
            if selected is not store:
                raise ValueError(f"{name}: Glm5Next converts from its single checkpoint")
            if format is not None:
                raise ValueError(f"{name}: the absorbed MLA operands are stored BF16")
            rows = tuple((i * (dn + dv) + begin, i * (dn + dv) + begin + width) for i in range(heads))
            blocks = select_rows(matrix_source(selected, source_name, (heads * (dn + dv), kv_rank)), rows)
            per_head = LogicalSource((heads, width, kv_rank), blocks.label, blocks.read_values)
            return transpose_source(per_head, (0, 2, 1), stored) if key else per_head

        self.model.add(Parameter(name, stored, factory(store), factory, (), "bf16",
                                 residency=name.split("/", 1)[0]))

    def group(self, *names):
        self.model.packing_groups.append(tuple(names))

    def hyper_connection(self, prefix, source_prefix, store, config):
        width = config["hc_mult"] * config["hidden_size"]
        rows = (2 + config["hc_mult"]) * config["hc_mult"]
        self.add(prefix + "fn", store, source_prefix + "_fn", (rows, width), inputs=(prefix + "input",))
        self.add(prefix + "base", store, source_prefix + "_base", (rows,), direct="fp32")
        self.add(prefix + "scale", store, source_prefix + "_scale", (3,), direct="fp32")

    def kda(self, prefix, source_prefix, store, config):
        h, heads, d = config["hidden_size"], config["linear_num_heads"], config["linear_head_dim"]
        width, taps = heads * d, config["linear_conv_kernel_dim"]
        sp, p, mixer_input = source_prefix + "self_attn.", prefix + "kda/", prefix + "mixer_input"
        for role, field in (("query", "q"), ("key", "k"), ("value", "v")):
            self.add(p + role, store, sp + field + "_proj.weight", (width, h), inputs=(mixer_input,))
        self.kda_convolution(p + "convolution", store, [sp + f + "_conv1d.weight" for f in "qkv"], taps, width)
        self.add(p + "forget_down", store, sp + "f_a_proj.weight", (d, h), inputs=(mixer_input,))
        self.add(p + "forget_up", store, sp + "f_b_proj.weight", (width, d), inputs=(p + "forget_rank",))
        self.add(p + "gate_down", store, sp + "g_a_proj.weight", (d, h), inputs=(mixer_input,))
        self.add(p + "gate_up", store, sp + "g_b_proj.weight", (width, d), inputs=(p + "gate_rank",))
        self.add(p + "beta", store, sp + "b_proj.weight", (heads, h), inputs=(mixer_input,))
        self.add(p + "a_log", store, sp + "A_log", (heads,), direct="fp32")
        self.add(p + "dt_bias", store, sp + "dt_bias", (width,), direct="fp32")
        self.add(p + "output_norm", store, sp + "o_norm.weight", (d,))
        self.add(p + "output", store, sp + "o_proj.weight", (h, width), inputs=(p + "gated_output",))
        self.group(p + "query", p + "key", p + "value")
        self.group(p + "forget_down", p + "gate_down", p + "beta")

    def sparse_attention(self, prefix, source_prefix, store, config):
        h, heads = config["hidden_size"], config["num_attention_heads"]
        dn, dv = config["qk_nope_head_dim"], config["v_head_dim"]
        q_rank, kv_rank = config["q_lora_rank"], config["kv_lora_rank"]
        ih, idim = config["index_n_heads"], config["index_head_dim"]
        sp, p, mixer_input = source_prefix + "self_attn.", prefix + "attention/", prefix + "mixer_input"
        self.add(p + "query_down", store, sp + "q_a_proj.weight", (q_rank, h), inputs=(mixer_input,))
        self.add(p + "query_norm", store, sp + "q_a_layernorm.weight", (q_rank,))
        self.add(p + "query_up", store, sp + "q_b_proj.weight", (heads * dn, q_rank), inputs=(p + "query_rank",))
        self.add(p + "latent", store, sp + "kv_a_proj_with_mqa.weight", (kv_rank, h), inputs=(mixer_input,))
        self.add(p + "latent_norm", store, sp + "kv_a_layernorm.weight", (kv_rank,))
        # kv_b_proj holds, per head, W_uk (dn rows) then W_uv (dv rows) over the latent. The absorbed
        # MLA Ops (mla_sparse_attention.h) read W_uk transposed per head, (heads, kv_rank, dn), and
        # W_uv as stored, (heads, dv, kv_rank); both stay BF16.
        self.latent_up(p + "key_up", store, sp + "kv_b_proj.weight", heads, dn, dv, kv_rank, key=True)
        self.latent_up(p + "value_up", store, sp + "kv_b_proj.weight", heads, dn, dv, kv_rank, key=False)
        self.add(p + "output", store, sp + "o_proj.weight", (h, heads * dv), inputs=(p + "attention_output",))
        ip = sp + "indexer."
        self.add(p + "index_query", store, ip + "wq_b.weight", (ih * idim, q_rank), inputs=(p + "query_rank",))
        self.add(p + "index_key", store, ip + "wk.weight", (idim, h), inputs=(mixer_input,))
        self.add(p + "index_key_norm", store, ip + "k_norm.weight", (idim,))
        self.add(p + "index_key_bias", store, ip + "k_norm.bias", (idim,))
        self.add(p + "index_gate", store, ip + "index_kpool_compress_gate", (idim, h), inputs=(mixer_input,))
        self.add(p + "index_position", store, ip + "index_kpool_compress_ape", (config["index_kpool"], idim),
                 direct="fp32")
        self.add(p + "index_weights", store, ip + "weights_proj.weight", (ih, h), inputs=(mixer_input,))
        self.group(p + "query_down", p + "latent")
        self.group(p + "index_key", p + "index_gate", p + "index_weights")

    def mlp(self, prefix, sp, store, h, intermediate, inputs):
        for role in ("gate", "up"):
            self.add(prefix + role, store, sp + role + "_proj.weight", (intermediate, h), inputs=(inputs,))
        self.add(prefix + "down", store, sp + "down_proj.weight", (h, intermediate), inputs=(prefix + "product",))
        self.group(prefix + "gate", prefix + "up")

    def moe(self, prefix, source_prefix, store, config):
        h, e, ir = config["hidden_size"], config["n_routed_experts"], config["moe_intermediate_size"]
        sp, p, ffn_input = source_prefix + "mlp.", prefix + "moe/", prefix + "ffn_input"
        self.add(p + "router", store, sp + "gate.weight", (e, h), inputs=(ffn_input,))
        self.add(p + "router_bias", store, sp + "gate.e_score_correction_bias", (e,), direct="fp32")
        self.mlp(p + "shared/", sp + "shared_experts.", store, h, ir, ffn_input)
        if not store.has(sp + "experts.0.gate_proj.weight_packed"):
            raise ValueError(f"{sp}experts: Glm5Next imports weight-only compressed-tensors NVFP4 experts")
        bank = ExpertBankSource((e, h, ir), f"{store.path}:{sp}experts", store, sp + "experts.", packed=True)
        self.model.add(Parameter(p + "experts", (e, h, ir), bank, None, (ffn_input,), "nvfp4_mul", residency="text"))

    def block(self, index, store, config):
        h = config["hidden_size"]
        prefix, sp = f"text/layers/{index}/", f"model.language_model.layers.{index}."
        self.hyper_connection(prefix + "attn_hc/", sp + "hc_attn", store, config)
        self.hyper_connection(prefix + "ffn_hc/", sp + "hc_ffn", store, config)
        self.add(prefix + "input_norm", store, sp + "input_layernorm.weight", (h,))
        self.add(prefix + "post_attention_norm", store, sp + "post_attention_layernorm.weight", (h,))
        if config["layer_types"][index] == "linear_attention":
            self.kda(prefix, sp, store, config)
        else:
            self.sparse_attention(prefix, sp, store, config)
        if index < config["first_k_dense_replace"]:
            self.mlp(prefix + "mlp/", sp + "mlp.", store, h, config["intermediate_size"], prefix + "ffn_input")
        else:
            self.moe(prefix, sp, store, config)

    def mtp(self, store, config):
        """The checkpoint's MTP layer (layer num_hidden_layers), stored as it is: its input fusion,
        one sparse-MLA block with its own indexer, a MoE whose routed experts are BF16 (288 x 3
        matrices) and its head norm. The engine does not load this component."""

        h, e, ir = config["hidden_size"], config["n_routed_experts"], config["moe_intermediate_size"]
        index = config["num_hidden_layers"]
        prefix, sp = "mtp/layers/0/", f"model.language_model.layers.{index}."
        self.add("mtp/embedding_norm", store, sp + "enorm.weight", (h,))
        self.add("mtp/hidden_norm", store, sp + "hnorm.weight", (h,))
        self.add("mtp/input_projection", store, sp + "eh_proj.weight", (h, 2 * h), inputs=("mtp/fused_input",))
        self.add("mtp/head_norm", store, sp + "shared_head.norm.weight", (h,))
        self.add(prefix + "input_norm", store, sp + "input_layernorm.weight", (h,))
        self.add(prefix + "post_attention_norm", store, sp + "post_attention_layernorm.weight", (h,))
        self.sparse_attention(prefix, sp, store, config)
        mp, ffn_input = sp + "mlp.", prefix + "ffn_input"
        self.add(prefix + "moe/router", store, mp + "gate.weight", (e, h), inputs=(ffn_input,))
        self.add(prefix + "moe/router_bias", store, mp + "gate.e_score_correction_bias", (e,), direct="fp32")
        self.mlp(prefix + "moe/shared/", mp + "shared_experts.", store, h, ir, ffn_input)
        for expert in range(e):
            self.mlp(prefix + f"moe/experts/{expert}/", mp + f"experts.{expert}.", store, h, ir, ffn_input)

    def vision(self, store, config):
        """The vision tower (component ``vision``, BF16 direct as the checkpoint stores it; no
        requantization). The engine's C++ binder follows this name table:

        ==============================  ==========================================  =====================
        logical name                    source tensor                               shape
        ==============================  ==========================================  =====================
        vision/patch_embedding          model.visual.patch_embed.proj.weight        (h, 3·t·p·p)
        vision/patch_embedding_bias     model.visual.patch_embed.proj.bias          (h,)
        vision/layers/N/norm1           model.visual.blocks.N.norm1.weight          (h,)
        vision/layers/N/norm2           model.visual.blocks.N.norm2.weight          (h,)
        vision/layers/N/attention/{qkv} model.visual.blocks.N.attn.qkv.weight       (h, h) each,
                                                                                     rows N·h..(N+1)·h
        vision/layers/N/attention/      model.visual.blocks.N.attn.qkv.bias         (h,) each,
          {q,k,v}_bias                                                              offset N·h
        vision/layers/N/attention/      model.visual.blocks.N.attn.q_norm.weight    (h/heads,)
          query_norm / key_norm         model.visual.blocks.N.attn.k_norm.weight
        vision/layers/N/attention/      model.visual.blocks.N.attn.proj.weight      (h, h)
          output
        vision/layers/N/attention/      model.visual.blocks.N.attn.proj.bias        (h,)
          output_bias
        vision/layers/N/mlp/gate        model.visual.blocks.N.mlp.gate_proj.weight  (i, h)
        vision/layers/N/mlp/gate_bias   model.visual.blocks.N.mlp.gate_proj.bias    (i,)
        vision/layers/N/mlp/up          model.visual.blocks.N.mlp.up_proj.weight    (i, h)
        vision/layers/N/mlp/up_bias     model.visual.blocks.N.mlp.up_proj.bias      (i,)
        vision/layers/N/mlp/down        model.visual.blocks.N.mlp.down_proj.weight  (h, i)
        vision/layers/N/mlp/down_bias   model.visual.blocks.N.mlp.down_proj.bias    (h,)
        vision/post_norm                model.visual.post_layernorm.weight          (h,)
        vision/downsample               model.visual.downsample.weight              (o, h, m, m)
        vision/downsample_bias          model.visual.downsample.bias                (o,)
        vision/merger/projection        model.visual.merger.proj.weight             (o, o)
        vision/merger/                  model.visual.merger.                        (o,) each
          post_projection_norm{,_bias}    post_projection_norm.{weight,bias}
        vision/merger/gate              model.visual.merger.gate_proj.weight        (pi, o)
        vision/merger/up                model.visual.merger.up_proj.weight          (pi, o)
        vision/merger/down              model.visual.merger.down_proj.weight        (o, pi)
        ==============================  ==========================================  =====================

        (h = hidden_size, i = intermediate_size, o = out_hidden_size, pi =
        projection_intermediate_size, m = spatial_merge_size, p = patch_size, t =
        temporal_patch_size.)

        ``query``/``key``/``value`` are three row blocks of the stored fused QKV matrix, with the
        biases cut at the matching element offsets; each block's three matrices (and the three
        biases) are one packing group. The block MLP and the merger keep the checkpoint's SwiGLU
        role names (gate/up/down), the same roles the text model binds; Qwen's tower has no
        counterpart to them (its merger is a two-layer GELU MLP bound as ``fc1``/``fc2``), so the
        merger's projection/post-projection-norm names follow the checkpoint instead.

        ``patch_embedding`` is the stored Conv3d kernel flattened to the matrix a patch GEMM
        consumes (the patch input is already a flat pixel vector in the kernel's C order, as for
        Qwen). ``downsample`` keeps its stored 4-D conv kernel (o, h, m, m): its input is a
        spatially arranged (m, m, h) block the engine permutes before consuming it, so it is bound
        as a convolution (or flattened at load time), not a plain patch GEMM.
        """

        h, i = config["hidden_size"], config["intermediate_size"]
        o, pi = config["out_hidden_size"], config["projection_intermediate_size"]
        heads = h // config["num_heads"]
        p_size, t_size = config["patch_size"], config["temporal_patch_size"]
        sp = "model.visual."
        self.add("vision/patch_embedding", store, sp + "patch_embed.proj.weight",
                 (h, 3 * t_size * p_size * p_size),
                 source_shape=(h, config["in_channels"], t_size, p_size, p_size),
                 inputs=("vision/patch_input",))
        self.add("vision/patch_embedding_bias", store, sp + "patch_embed.proj.bias", (h,))
        for index in range(config["depth"]):
            p, b = f"vision/layers/{index}/", f"{sp}blocks.{index}."
            self.add(p + "norm1", store, b + "norm1.weight", (h,))
            self.add(p + "norm2", store, b + "norm2.weight", (h,))
            for number, role in enumerate(("query", "key", "value")):
                self.add(p + "attention/" + role, store, b + "attn.qkv.weight", (h, h),
                         source_shape=(3 * h, h), rows=((number * h, (number + 1) * h),),
                         inputs=(p + "attention_input",))
                self.add(p + "attention/" + role + "_bias", store, b + "attn.qkv.bias", (h,),
                         source_shape=(3 * h,), offset=number * h)
            self.group(*(p + "attention/" + r for r in ("query", "key", "value")))
            self.group(*(p + "attention/" + r + "_bias" for r in ("query", "key", "value")))
            self.add(p + "attention/query_norm", store, b + "attn.q_norm.weight", (heads,))
            self.add(p + "attention/key_norm", store, b + "attn.k_norm.weight", (heads,))
            self.add(p + "attention/output", store, b + "attn.proj.weight", (h, h),
                     inputs=(p + "attention_output",))
            self.add(p + "attention/output_bias", store, b + "attn.proj.bias", (h,))
            for role in ("gate", "up"):
                self.add(p + "mlp/" + role, store, b + "mlp." + role + "_proj.weight", (i, h),
                         inputs=(p + "mlp_input",))
                self.add(p + "mlp/" + role + "_bias", store, b + "mlp." + role + "_proj.bias", (i,))
            self.group(p + "mlp/gate", p + "mlp/up")
            self.add(p + "mlp/down", store, b + "mlp.down_proj.weight", (h, i), inputs=(p + "mlp_product",))
            self.add(p + "mlp/down_bias", store, b + "mlp.down_proj.bias", (h,))
        merge = config["spatial_merge_size"]
        self.add("vision/post_norm", store, sp + "post_layernorm.weight", (h,))
        self.add("vision/downsample", store, sp + "downsample.weight", (o, h, merge, merge),
                 inputs=("vision/merged_input",))
        self.add("vision/downsample_bias", store, sp + "downsample.bias", (o,))
        self.add("vision/merger/projection", store, sp + "merger.proj.weight", (o, o),
                 inputs=("vision/merger/input",))
        self.add("vision/merger/post_projection_norm", store, sp + "merger.post_projection_norm.weight", (o,))
        self.add("vision/merger/post_projection_norm_bias", store, sp + "merger.post_projection_norm.bias", (o,))
        for role in ("gate", "up"):
            self.add("vision/merger/" + role, store, sp + "merger." + role + "_proj.weight", (pi, o),
                     inputs=("vision/merger/activation",))
        self.group("vision/merger/gate", "vision/merger/up")
        self.add("vision/merger/down", store, sp + "merger.down_proj.weight", (o, pi),
                 inputs=("vision/merger/product",))

    def draft(self, store, config, target):
        h, d = target["hidden_size"], config["head_dim"]
        q, k = config["num_attention_heads"] * d, config["num_key_value_heads"] * d
        draft = config["dflash_config"]
        backend = "dflash2"
        self.add(backend + "/feature_projection", store, "fc.weight", (h, len(draft["target_layer_ids"]) * h),
                 inputs=(backend + "/target_features",))
        self.add(backend + "/context_norm", store, "hidden_norm.weight", (h,))
        self.add(backend + "/final_norm", store, "norm.weight", (h,))
        taps, groups = draft["conv_kernel_size"], h // draft["conv_group_size"]
        for i in range(config["num_hidden_layers"]):
            p, sp = f"{backend}/layers/{i}/", f"layers.{i}."
            self.add(p + "input_norm", store, sp + "input_layernorm.weight", (h,))
            self.add(p + "post_attention_norm", store, sp + "post_attention_layernorm.weight", (h,))
            for role, field, rows in (("query", "q_proj", q), ("key", "k_proj", k), ("value", "v_proj", k)):
                self.add(p + "attention/" + role, store, sp + "self_attn." + field + ".weight", (rows, h),
                         inputs=(p + "query_projection_input",))
                if role in ("key", "value"):
                    self.add(p + "attention/context_" + role, store, sp + "self_attn." + field + ".weight", (rows, h),
                             inputs=(backend + "/context_input",))
            self.group(*(p + "attention/" + r for r in ("query", "key", "value")))
            for role, field in (("query_norm", "q_norm"), ("key_norm", "k_norm")):
                self.add(p + "attention/" + role, store, sp + "self_attn." + field + ".weight", (d,))
            self.add(p + "attention/output", store, sp + "self_attn.o_proj.weight", (h, q),
                     inputs=(p + "attention_output",))
            for role in ("gate", "up"):
                self.add(p + "mlp/" + role, store, sp + "mlp." + role + "_proj.weight",
                         (config["intermediate_size"], h), inputs=(p + "mlp_input",))
            self.add(p + "mlp/down", store, sp + "mlp.down_proj.weight", (h, config["intermediate_size"]),
                     inputs=(p + "mlp_product",))
            self.group(p + "mlp/gate", p + "mlp/up")
            for branch in ("attention_conv", "mlp_conv"):
                self.add(p + branch + "/base_kernel", store, sp + branch + ".base_kernel", (2, taps, h))
                self.add(p + branch + "/kernel_projection", store, sp + branch + ".kernel_projection.weight",
                         (2 * taps * groups, h), inputs=(p + branch + "_input",))
        rank = draft["selector_rank"]
        self.add(backend + "/candidate_selector/hidden_projection", store,
                 "candidate_selector.hidden_projection.weight", (rank, h), inputs=(backend + "/final_hidden",))
        for role in ("predecessor_codebook", "successor_codebook"):
            self.add(backend + "/candidate_selector/" + role, store, "candidate_selector." + role,
                     (target["vocab_size"], rank))


def build_model(base: SafetensorsSource, *, components: tuple[str, ...] = ("text",),
                resource_overrides: Mapping[str, str | Path] | None = None) -> Model:
    """The logical text model and, when the checkpoint has one, its MTP layer (component ``mtp``,
    stored so the artifact keeps the whole checkpoint; the engine loads only ``text``). With the
    ``vision`` component the artifact also stores the vision tower (BF16 as the checkpoint keeps
    it, ``_Builder.vision``'s name table) and its processor config; the converter CLI selects
    ``text,vision`` for a Glm5Next checkpoint with a tower by default.

    The vision component's one resource keeps the checkpoint's ``processor_config.json`` bytes
    verbatim (object ``resource/vision/processor_config.json``). The Qwen families instead store
    HF's two separate ``preprocessor_config.json`` / ``video_preprocessor_config.json`` files;
    GLM-5.3-Flash ships a single processor file holding both the image and the video processor,
    and keeping those bytes beats splitting and re-serializing them.
    """

    selected = set(components)
    if "text" not in selected or selected - {"text", "vision"}:
        raise ValueError("Glm5Next converts the text component and optional vision (plus the MTP layer its checkpoint stores)")
    want_vision = "vision" in selected
    if want_vision and not base.has("model.visual.patch_embed.proj.weight"):
        raise ValueError("vision requested, but the checkpoint has no model.visual tensors")
    config = text_config(base.config)
    vision = vision_config(base.config, config) if want_vision else None
    refs, resources, count, special = load_resources(
        base.root, vocab_size=config["vocab_size"], vision_config=vision,
        vision_resources=("processor_config.json",), vision_check=_check_processor,
        overrides=resource_overrides)
    records = {"text": {"config": config, "resources": refs["text"]}}
    if config["mtp_layers"]:
        records["mtp"] = {"config": {"architectures": ["Glm5NextMTP"]}, "target": "text"}
    if want_vision:
        records["vision"] = {"config": vision, "resources": refs["vision"], "target": "text"}
    model = Model(records, resources=resources, token_count=count, special_token_ids=special)
    builder = _Builder(model)
    h, v = config["hidden_size"], config["vocab_size"]
    builder.add("text/token_embedding", base, "model.language_model.embed_tokens.weight", (v, h))
    builder.add("text/output_head", base, "lm_head.weight", (v, h), inputs=("text/final_hidden",))
    builder.add("text/final_norm", base, "model.language_model.norm.weight", (h,))
    for index in range(config["num_hidden_layers"]):
        builder.block(index, base, config)
    if config["mtp_layers"]:
        builder.mtp(base, config)
    if want_vision:
        builder.vision(base, vision)
    return model


DFLASH2_TARGET_FIELDS = ("architectures", "hidden_size", "vocab_size", "num_hidden_layers")


def build_dflash2_companion(base: SafetensorsSource, draft: SafetensorsSource) -> Model:
    """A companion artifact holding only a DFlash2 drafter of a Glm5Next model, loaded beside any
    artifact of that model (``--draft-model``). ``base`` supplies the target's config and tokenizer;
    the drafter reuses the target's token embedding and output head (declared Uses)."""

    target = text_config(base.config)
    config = draft_config(draft.config, target)
    config["target_config"] = {key: target[key] for key in DFLASH2_TARGET_FIELDS}
    _, _, count, special = load_resources(base.root, vocab_size=target["vocab_size"])
    if config["dflash_config"]["selector_top_k"] > count:
        raise ValueError("DFlash2 selector_top_k exceeds public token domain")
    model = Model({"dflash2": {"config": config, "target": "text"}}, token_count=count, special_token_ids=special,
                  target=target, external_uses=(("text/output_head", "dflash2/final_hidden"),))
    _Builder(model).draft(draft, config, target)
    return model
