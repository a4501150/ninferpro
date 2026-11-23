"""Qwen-shared conversion pieces (mirrors src/models/qwen_common): the config builders both
families derive their records from (``vision_config``, ``draft_config``, ``_rope_source``), the
HF-checkpoint builder's shared source axes and validation (``_Builder``), the Qwen tokenizer
invariants the runtime frontend hard-requires, and the Qwen3-VL vision tower's GGUF name map
(``qwen35_vision_name_map``, shared by both families' vision components).

The family modules (tools/convert/qwen3_5*.py, tools/convert/qwen4_exp*.py) import these; this
module imports neither family.
"""

from __future__ import annotations

import json
from math import isqrt
from typing import Callable, TYPE_CHECKING

import torch

from .config_values import _f32, _fixed, _positive
from .model import Model, Parameter
from .sources.compressed_tensors import matrix_source
from .sources.logical import select_rows, transpose_source
from .sources.safetensors import SafetensorsSource, tensor_source

if TYPE_CHECKING:
    from .sources.gguf import Transform

# Must stay byte-identical to kQwenSplitPattern in
# src/models/qwen_common/frontend/tokenizer.cpp: the runtime tokenizer validates
# tokenizer.json's Split pattern for exact equality and rejects any artifact whose
# pattern differs.
STD_QWEN_SPLIT_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+"
    r"|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)

# Qwen3.5 tokenizer_config.json invariants the runtime frontend hard-requires
# (src/models/qwen_common/frontend/frontend.cpp validate_tokenizer_config): a missing
# field defaults to the failing value, so each must be present and set to the
# official Qwen3.5 value.
KQWEN35_PAD_TOKEN = "<|endoftext|>"


def _dump_json(value: dict) -> bytes:
    return (json.dumps(value, ensure_ascii=False) + "\n").encode("utf-8")


def _normalize_qwen_split(tokenizer: dict) -> bool:
    """Normalize a drifted pre-tokenizer split pattern to the standard Qwen one.

    The runtime tokenizer pre-tokenizes with its own hardcoded \\p{L}\\p{M} logic
    (not the JSON regex) but validates the stored Split pattern for exact equality
    against kQwenSplitPattern, so a drifted pattern (some third-party Qwen
    checkpoints drop \\p{M}) makes the artifact unloadable without changing
    tokenization. Rewriting it to the standard pattern is safe. Returns True if the
    pattern changed.
    """
    pre = tokenizer.get("pre_tokenizer")
    if not isinstance(pre, dict) or pre.get("type") != "Sequence":
        return False
    parts = pre.get("pretokenizers")
    if not isinstance(parts, list) or len(parts) != 2:
        return False
    split = parts[0]
    if not isinstance(split, dict) or split.get("type") != "Split":
        return False
    pattern = split.get("pattern")
    if not isinstance(pattern, dict):
        return False
    if pattern.get("Regex") == STD_QWEN_SPLIT_PATTERN:
        return False
    pattern["Regex"] = STD_QWEN_SPLIT_PATTERN
    split["behavior"] = "Isolated"
    split["invert"] = False
    return True


def _normalize_added_decoder(tokenizer: dict, config: dict) -> bool:
    """Ensure tokenizer_config.json carries an added_tokens_decoder object.

    The runtime tokenizer requires the field to exist as an object and
    cross-checks each entry against tokenizer.json's added_tokens for shared ids.
    Third-party Qwen checkpoints frequently omit the field; rebuild it from
    added_tokens (the canonical source) so the artifact loads and stays consistent.
    Returns True if the field was built.
    """
    if isinstance(config.get("added_tokens_decoder"), dict):
        return False
    decoder: dict[str, dict] = {}
    for token in tokenizer.get("added_tokens", []):
        raw_id = token.get("id")
        if type(raw_id) is not int:
            continue
        decoder[str(raw_id)] = {
            "content": token["content"],
            "single_word": token.get("single_word", False),
            "lstrip": token.get("lstrip", False),
            "rstrip": token.get("rstrip", False),
            "normalized": token.get("normalized", False),
            "special": token.get("special", False),
        }
    config["added_tokens_decoder"] = decoder
    return True


def _normalize_prefix_semantics(config: dict) -> bool:
    """Ensure the Qwen3.5 prefix-semantics invariants in tokenizer_config.json.

    The runtime frontend requires add_bos_token=false, add_prefix_space=false, and
    pad_token=<|endoftext|>; a missing field defaults to the failing value and a
    non-official pad token is rejected. Third-party Qwen checkpoints may omit or
    mis-set these. Returns True if any field was set.
    """
    changed = False
    if config.get("add_bos_token") is not False:
        config["add_bos_token"] = False
        changed = True
    if config.get("add_prefix_space") is not False:
        config["add_prefix_space"] = False
        changed = True
    if config.get("pad_token") != KQWEN35_PAD_TOKEN:
        config["pad_token"] = KQWEN35_PAD_TOKEN
        changed = True
    return changed


def normalize_tokenizer(parsed: dict[str, dict], payloads: dict[str, bytes]) -> None:
    """Apply the Qwen tokenizer invariants to resources.load_resources' parsed JSON resources,
    replacing the stored payloads whose bytes the normalization changed.

    Families pass this to resources.load_resources as its ``normalize`` hook; the shared loader
    itself knows no family."""
    tokenizer_json = parsed["tokenizer.json"]
    tokenizer_config = parsed["tokenizer_config.json"]
    if _normalize_qwen_split(tokenizer_json):
        payloads["resource/text/tokenizer.json"] = _dump_json(tokenizer_json)
        print(
            "normalized tokenizer.json split pattern to the standard Qwen pattern",
            flush=True,
        )
    config_changed = False
    if _normalize_added_decoder(tokenizer_json, tokenizer_config):
        print(
            "built tokenizer_config.json added_tokens_decoder "
            f"({len(tokenizer_config['added_tokens_decoder'])} tokens)",
            flush=True,
        )
        config_changed = True
    if _normalize_prefix_semantics(tokenizer_config):
        print(
            "normalized tokenizer_config.json prefix semantics "
            "(add_bos_token/add_prefix_space=false, pad_token=<|endoftext|>)",
            flush=True,
        )
        config_changed = True
    if config_changed:
        payloads["resource/text/tokenizer_config.json"] = _dump_json(tokenizer_config)


def _rope_source(raw: dict, label: str) -> dict:
    rope = {}
    for field in ("rope_scaling", "rope_parameters"):
        value = raw.get(field)
        if value is None:
            continue
        if not isinstance(value, dict):
            raise ValueError(f"{label}.{field}: expected an object")
        for alias in ("type", "rope_type"):
            _fixed(value, alias, "default", label + "." + field)
        if "factor" in value and _f32(value["factor"], label + ".factor") != 1.0:
            raise ValueError(f"{label}: scaled RoPE is not implemented")
        for key, item in value.items():
            if key in rope and rope[key] != item:
                raise ValueError(f"{label}: conflicting RoPE field {key}")
            rope[key] = item
    return rope


def vision_config(source: dict, text: dict) -> dict:
    raw = source.get("vision_config")
    if not isinstance(raw, dict):
        raise ValueError("Vision requires vision_config")
    _fixed(raw, "in_channels", 3, "vision")
    _fixed(raw, "hidden_act", "gelu_pytorch_tanh", "vision")
    _fixed(raw, "deepstack_visual_indexes", [], "vision")
    _fixed(raw, "out_hidden_size", text["hidden_size"], "vision")
    result = {
        "model_type": (
            "qwen3_5_moe_vision" if "num_experts" in text else "qwen3_5_vision"
        )
    }
    for key in (
        "depth",
        "hidden_size",
        "intermediate_size",
        "num_heads",
        "patch_size",
        "temporal_patch_size",
        "spatial_merge_size",
        "num_position_embeddings",
    ):
        result[key] = _positive(raw.get(key), "vision." + key)
    h, heads = result["hidden_size"], result["num_heads"]
    positions = result["num_position_embeddings"]
    if h % heads or (h // heads) % 4 or isqrt(positions) ** 2 != positions:
        raise ValueError("Vision head or square position geometry is invalid")
    return result


def draft_config(raw: dict, target: dict, backend: str) -> dict:
    architecture = "DFlash2DraftModel" if backend == "dflash2" else "DFlashDraftModel"
    _fixed(raw, "architectures", [architecture], backend)
    _fixed(raw, "model_type", "qwen3", backend)
    _fixed(raw, "hidden_act", "silu", backend)
    _fixed(raw, "attention_bias", False, backend)
    _fixed(raw, "is_causal", False, backend)
    for key, value in (
        ("hidden_size", target["hidden_size"]),
        ("vocab_size", target["vocab_size"]),
        ("num_target_layers", target["num_hidden_layers"]),
    ):
        _fixed(raw, key, value, backend)
    result = {"architectures": [architecture], "model_type": "qwen3"}
    for key in (
        "intermediate_size",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "num_hidden_layers",
        "max_position_embeddings",
    ):
        result[key] = _positive(raw.get(key), backend + "." + key)
    if (
        result["num_attention_heads"] % result["num_key_value_heads"]
        or result["head_dim"] % 2
    ):
        raise ValueError(f"{backend}: invalid attention geometry")
    result["rms_norm_eps"] = _f32(
        raw.get("rms_norm_eps", 1e-6), backend + ".rms_norm_eps"
    )
    rope = _rope_source(raw, backend)
    result["rope_parameters"] = {
        "rope_theta": _f32(
            rope.get("rope_theta", raw.get("rope_theta", 10_000_000)),
            backend + ".rope_theta",
        )
    }
    layers = raw.get("layer_types")
    if layers is None:
        enabled = raw.get("use_sliding_window", False)
        if type(enabled) is not bool:
            raise ValueError(f"{backend}.use_sliding_window must be boolean")
        # Qwen3Config applies sliding attention at and above this layer index.
        threshold = raw.get("max_window_layers", 28)
        if type(threshold) is not int or threshold < 0:
            raise ValueError(f"{backend}.max_window_layers must be nonnegative")
        layers = [
            "sliding_attention" if enabled and i >= threshold else "full_attention"
            for i in range(result["num_hidden_layers"])
        ]
    if (
        not isinstance(layers, list)
        or len(layers) != result["num_hidden_layers"]
        or any(k not in ("sliding_attention", "full_attention") for k in layers)
    ):
        raise ValueError(f"{backend}: invalid layer_types")
    result["layer_types"] = list(layers)
    if "sliding_attention" in layers:
        result["sliding_window"] = _positive(
            raw.get("sliding_window"), backend + ".sliding_window"
        )
    draft = raw.get("dflash_config", {})
    taps = draft.get("target_layer_ids")
    if (
        not isinstance(taps, list)
        or not taps
        or any(
            type(i) is not int or not 0 <= i < target["num_hidden_layers"] for i in taps
        )
        or len(set(taps)) != len(taps)
    ):
        raise ValueError(f"{backend}: invalid target_layer_ids")
    mask = draft.get("mask_token_id")
    if type(mask) is not int or not 0 <= mask < target["vocab_size"]:
        raise ValueError(f"{backend}: mask token outside target embedding")
    output = {"target_layer_ids": list(taps), "mask_token_id": mask}
    if backend == "dflash2":
        for key in (
            "conv_kernel_size",
            "conv_group_size",
            "selector_rank",
            "selector_top_k",
        ):
            output[key] = _positive(draft.get(key), backend + "." + key)
        if target["hidden_size"] % output["conv_group_size"]:
            raise ValueError(
                "DFlash2 hidden width must be divisible by conv_group_size"
            )
        for scope in (raw, draft):
            for key, expected in (
                ("sample_from_anchor", False),
                ("input_embedding_scale", 1.0),
                ("output_multiplier", 1.0),
            ):
                _fixed(scope, key, expected, backend)
            if scope.get("final_logit_softcapping") not in (None, 0, 0.0):
                raise ValueError("DFlash2 final_logit_softcapping must be disabled")
    result["dflash_config"] = output
    return result


def _has_model_config(source: SafetensorsSource) -> bool:
    return bool(
        source.config.keys()
        - {
            "quantization_config",
            "_name_or_path",
            "transformers_version",
            "torch_dtype",
            "dtype",
        }
    )


def _check_source_fields(name, actual, expected, fields):
    for field in sorted(fields):
        if actual.get(field) != expected.get(field):
            raise ValueError(
                f"{name}: additional source {field} differs; provide an explicit logical source"
            )


class _Builder:
    """The HF-checkpoint builder's shared source axes and validation: parameter adds with source
    remapping, packing groups, and the per-block adders (attention, GDN, dense, MoE, vision,
    draft) the families reuse. The family binds its text-config validator as ``text_config``
    (validate_source re-derives an additional source's config with it); a family whose sources
    are single and fixed overrides validate_source and never needs it."""

    text_config: Callable  # the family's text_config; validate_source requires it

    def __init__(self, model: Model):
        self.model = model

    def validate_source(self, selected, original, name):
        if selected is original or not _has_model_config(selected):
            return
        component = name.split("/", 1)[0]
        target = self.model.config
        if component in ("text", "mtp"):
            actual = self.text_config(selected.config, mtp=component == "mtp")
            expected = target
            fields = {"hidden_size"}
            if name.endswith(("token_embedding", "output_head")):
                fields.add("vocab_size")
            if name == "text/output_head":
                fields.add("tie_word_embeddings")
            if "/attention/" in name:
                fields.update(
                    ("num_attention_heads", "num_key_value_heads", "head_dim")
                )
            if "/gdn/" in name:
                fields.update(
                    (
                        "linear_num_key_heads",
                        "linear_key_head_dim",
                        "linear_num_value_heads",
                        "linear_value_head_dim",
                    )
                )
                if name.endswith("/convolution"):
                    fields.add("linear_conv_kernel_dim")
            if "/mlp/" in name:
                fields.add("intermediate_size")
            if "/moe/experts/" in name:
                fields.update(("num_experts", "moe_intermediate_size"))
            if "/moe/shared/" in name:
                fields.add("shared_expert_intermediate_size")
            if name.endswith("/moe/router"):
                fields.add("num_experts")
        elif component == "vision":
            actual = vision_config(selected.config, target)
            expected = self.model.components[component]["config"]
            fields = {"hidden_size"}
            if "/attention/" in name:
                fields.add("num_heads")
            if "/mlp/" in name:
                fields.add("intermediate_size")
            if name == "vision/patch_embedding":
                fields.update(("patch_size", "temporal_patch_size"))
            if name.startswith("vision/merger/fc"):
                fields.add("spatial_merge_size")
            if "position_embedding" in name:
                fields.add("num_position_embeddings")
        else:
            actual = draft_config(selected.config, target, component)
            expected = self.model.components[component]["config"]
            fields = set()
            if "/attention/" in name:
                fields.update(
                    ("num_attention_heads", "num_key_value_heads", "head_dim")
                )
            if "/mlp/" in name:
                fields.add("intermediate_size")
            draft_fields = set()
            if name.endswith("/feature_projection"):
                draft_fields.add("target_layer_ids")
            if "conv/" in name:
                draft_fields.add("conv_kernel_size")
                if name.endswith("/kernel_projection"):
                    draft_fields.add("conv_group_size")
            if "/candidate_selector/" in name:
                draft_fields.add("selector_rank")
            _check_source_fields(
                name, actual["dflash_config"], expected["dflash_config"], draft_fields
            )
        if "/layers/" in name:
            index = int(name.split("/layers/", 1)[1].split("/", 1)[0])
            if component == "vision":
                compatible = index < actual["depth"]
            elif component == "mtp":
                compatible = index == 0
            else:
                compatible = index < len(actual["layer_types"]) and (
                    actual["layer_types"][index] == expected["layer_types"][index]
                )
            if not compatible:
                raise ValueError(
                    f"{name}: additional source layer topology differs; provide an explicit logical source"
                )
        _check_source_fields(name, actual, expected, fields)

    def add(
        self,
        name,
        store,
        source_name,
        shape,
        *,
        inputs=(),
        direct="bf16",
        source_shape=None,
        offset=0,
        rows=None,
        transpose=None,
    ):
        shape = tuple(shape)
        original = shape if source_shape is None else tuple(source_shape)

        def factory(selected, format=None):
            self.validate_source(selected, store, name)
            selected_name = source_name
            if name.startswith("text/") and _has_model_config(selected):
                if "text_config" not in selected.config:
                    selected_name = selected_name.replace(
                        "model.language_model.", "model.", 1
                    )
                elif "text_config" not in store.config and selected_name.startswith(
                    "model."
                ):
                    selected_name = selected_name.replace(
                        "model.", "model.language_model.", 1
                    )
            if transpose is not None:
                if format is not None:
                    raise ValueError(f"{name}: transpose source requires value access")
                ref = tensor_source(selected, selected_name, original)
                return transpose_source(ref, transpose, shape)
            if len(original) == 2 and offset == 0:
                ref = matrix_source(selected, selected_name, original, format)
                return select_rows(ref, rows) if rows is not None else ref
            if format is not None:
                raise ValueError(f"{name}: provide an explicit encoded source mapping")
            return tensor_source(
                selected, selected_name, shape, offset=offset, source_shape=original
            )

        self.model.add(
            Parameter(
                name,
                shape,
                factory(store),
                factory,
                tuple(inputs),
                direct,
                residency=name.split("/", 1)[0],
            )
        )

    def group(self, *names):
        self.model.packing_groups.append(tuple(names))

    def attention(self, prefix, source_prefix, store, config):
        h, d = config["hidden_size"], config["head_dim"]
        q, k = config["num_attention_heads"] * d, config["num_key_value_heads"] * d
        for role, gate in (("query", False), ("gate", True)):
            ranges = tuple(
                (head * 2 * d + int(gate) * d, head * 2 * d + (int(gate) + 1) * d)
                for head in range(config["num_attention_heads"])
            )
            self.add(
                prefix + "attention/" + role,
                store,
                source_prefix + "self_attn.q_proj.weight",
                (q, h),
                source_shape=(2 * q, h),
                rows=ranges,
                inputs=(prefix + "mixer_input",),
            )
        for role, source in (("key", "k_proj"), ("value", "v_proj")):
            self.add(
                prefix + "attention/" + role,
                store,
                source_prefix + "self_attn." + source + ".weight",
                (k, h),
                inputs=(prefix + "mixer_input",),
            )
        for role, source in (("query_norm", "q_norm"), ("key_norm", "k_norm")):
            self.add(
                prefix + "attention/" + role,
                store,
                source_prefix + "self_attn." + source + ".weight",
                (d,),
            )
        self.add(
            prefix + "attention/output",
            store,
            source_prefix + "self_attn.o_proj.weight",
            (h, q),
            inputs=(prefix + "attention/gated_output",),
        )
        roles = [
            prefix + "attention/" + role for role in ("query", "key", "gate", "value")
        ]
        self.group(*roles)
        self.group(*roles[:2])
        self.group(*roles[2:])

    def gdn(self, prefix, source_prefix, store, config):
        h = config["hidden_size"]
        kg = config["linear_num_key_heads"] * config["linear_key_head_dim"]
        vg = config["linear_num_value_heads"] * config["linear_value_head_dim"]
        nv, dv, taps = (
            config["linear_num_value_heads"],
            config["linear_value_head_dim"],
            config["linear_conv_kernel_dim"],
        )
        channels = 2 * kg + vg
        source_prefix += "linear_attn."
        for role, field in (("a_log", "A_log"), ("dt_bias", "dt_bias")):
            self.add(
                prefix + "gdn/" + role,
                store,
                source_prefix + field,
                (nv,),
                direct="fp32",
            )
        self.add(
            prefix + "gdn/convolution",
            store,
            source_prefix + "conv1d.weight",
            (taps, channels),
            source_shape=(channels, 1, taps),
            transpose=(2, 0, 1),
        )
        for role, field in (
            ("a_projection", "in_proj_a"),
            ("b_projection", "in_proj_b"),
        ):
            self.add(
                prefix + "gdn/" + role,
                store,
                source_prefix + field + ".weight",
                (nv, h),
                inputs=(prefix + "mixer_input",),
            )
        for role, start, count in (
            ("query", 0, kg),
            ("key", kg, kg),
            ("value", 2 * kg, vg),
        ):
            self.add(
                prefix + "gdn/" + role,
                store,
                source_prefix + "in_proj_qkv.weight",
                (count, h),
                source_shape=(channels, h),
                rows=((start, start + count),),
                inputs=(prefix + "mixer_input",),
            )
        self.add(
            prefix + "gdn/z",
            store,
            source_prefix + "in_proj_z.weight",
            (vg, h),
            inputs=(prefix + "mixer_input",),
        )
        self.add(prefix + "gdn/norm", store, source_prefix + "norm.weight", (dv,))
        self.add(
            prefix + "gdn/output",
            store,
            source_prefix + "out_proj.weight",
            (h, vg),
            inputs=(prefix + "gdn/gated_output",),
        )
        roles = [prefix + "gdn/" + role for role in ("query", "key", "value", "z")]
        self.group(*roles)
        self.group(*roles[:2])
        self.group(*roles[2:])
        self.group(prefix + "gdn/a_projection", prefix + "gdn/b_projection")

    def dense(self, prefix, source_prefix, store, h, intermediate, *, draft=False):
        stem_input = prefix + ("mlp_input" if draft else "ffn_input")
        for role in ("gate", "up"):
            self.add(
                prefix + "mlp/" + role,
                store,
                source_prefix + "mlp." + role + "_proj.weight",
                (intermediate, h),
                inputs=(stem_input,),
            )
        self.add(
            prefix + "mlp/down",
            store,
            source_prefix + "mlp.down_proj.weight",
            (h, intermediate),
            inputs=(prefix + ("mlp_product" if draft else "mlp/product"),),
        )
        self.group(prefix + "mlp/gate", prefix + "mlp/up")

    def moe(self, prefix, source_prefix, store, config):
        h, e = config["hidden_size"], config["num_experts"]
        ir, shared = (
            config["moe_intermediate_size"],
            config["shared_expert_intermediate_size"],
        )
        sp, p = source_prefix + "mlp.", prefix + "moe/"
        self.add(
            p + "router",
            store,
            sp + "gate.weight",
            (e, h),
            inputs=(prefix + "ffn_input",),
        )
        self.add(
            p + "shared_score",
            store,
            sp + "shared_expert_gate.weight",
            (1, h),
            inputs=(prefix + "ffn_input",),
        )
        self.group(p + "router", p + "shared_score")
        gate_up, downs = [], []
        for expert in range(e):
            ep = p + f"experts/{expert}/"
            for role, half in (("gate", 0), ("up", 1)):
                self.add(
                    ep + role,
                    store,
                    sp + "experts.gate_up_proj",
                    (ir, h),
                    source_shape=(e, 2 * ir, h),
                    offset=(expert * 2 + half) * ir * h,
                    inputs=(prefix + "ffn_input",),
                )
                gate_up.append(ep + role)
            self.add(
                ep + "down",
                store,
                sp + "experts.down_proj",
                (h, ir),
                source_shape=(e, h, ir),
                offset=expert * h * ir,
                inputs=(ep + "product",),
            )
            downs.append(ep + "down")
        self.group(*gate_up)
        self.group(*downs)
        for role in ("gate", "up"):
            self.add(
                p + "shared/" + role,
                store,
                sp + "shared_expert." + role + "_proj.weight",
                (shared, h),
                inputs=(prefix + "ffn_input",),
            )
        self.add(
            p + "shared/down",
            store,
            sp + "shared_expert.down_proj.weight",
            (h, shared),
            inputs=(p + "shared/product",),
        )
        self.group(p + "shared/gate", p + "shared/up")

    def block(self, prefix, source_prefix, store, config, mixer):
        h = config["hidden_size"]
        self.add(
            prefix + "input_norm", store, source_prefix + "input_layernorm.weight", (h,)
        )
        if mixer == "full_attention":
            self.attention(prefix, source_prefix, store, config)
        else:
            self.gdn(prefix, source_prefix, store, config)
        self.add(
            prefix + "post_attention_norm",
            store,
            source_prefix + "post_attention_layernorm.weight",
            (h,),
        )
        if "num_experts" in config:
            self.moe(prefix, source_prefix, store, config)
        else:
            self.dense(prefix, source_prefix, store, h, config["intermediate_size"])

    def vision(self, store, config, output_width):
        h, intermediate = config["hidden_size"], config["intermediate_size"]
        ps, pt = config["patch_size"], config["temporal_patch_size"]
        self.add(
            "vision/patch_embedding",
            store,
            "model.visual.patch_embed.proj.weight",
            (h, 3 * pt * ps * ps),
            source_shape=(h, 3, pt, ps, ps),
            inputs=("vision/patch_input",),
        )
        self.add(
            "vision/patch_embedding_bias",
            store,
            "model.visual.patch_embed.proj.bias",
            (h,),
        )
        self.add(
            "vision/position_embedding",
            store,
            "model.visual.pos_embed.weight",
            (config["num_position_embeddings"], h),
        )
        for i in range(config["depth"]):
            p, sp = f"vision/layers/{i}/", f"model.visual.blocks.{i}."
            for role in ("norm1", "norm2"):
                for kind in ("weight", "bias"):
                    self.add(p + role + "_" + kind, store, sp + role + "." + kind, (h,))
            for number, role in enumerate(("query", "key", "value")):
                self.add(
                    p + "attention/" + role,
                    store,
                    sp + "attn.qkv.weight",
                    (h, h),
                    source_shape=(3 * h, h),
                    rows=((number * h, (number + 1) * h),),
                    inputs=(p + "attention_input",),
                )
                self.add(
                    p + "attention/" + role + "_bias",
                    store,
                    sp + "attn.qkv.bias",
                    (h,),
                    source_shape=(3 * h,),
                    offset=number * h,
                )
            self.group(*(p + "attention/" + r for r in ("query", "key", "value")))
            self.group(
                *(p + "attention/" + r + "_bias" for r in ("query", "key", "value"))
            )
            for role, field, dims, inputs in (
                (
                    "attention/output",
                    "attn.proj.weight",
                    (h, h),
                    (p + "attention_output",),
                ),
                ("attention/output_bias", "attn.proj.bias", (h,), ()),
                (
                    "mlp/fc1",
                    "mlp.linear_fc1.weight",
                    (intermediate, h),
                    (p + "mlp_input",),
                ),
                ("mlp/fc1_bias", "mlp.linear_fc1.bias", (intermediate,), ()),
                (
                    "mlp/fc2",
                    "mlp.linear_fc2.weight",
                    (h, intermediate),
                    (p + "mlp_activation",),
                ),
                ("mlp/fc2_bias", "mlp.linear_fc2.bias", (h,), ()),
            ):
                self.add(p + role, store, sp + field, dims, inputs=inputs)
        merger = config["spatial_merge_size"] ** 2 * h
        for role, field, dims, inputs in (
            ("norm_weight", "norm.weight", (h,), ()),
            ("norm_bias", "norm.bias", (h,), ()),
            ("fc1", "linear_fc1.weight", (merger, merger), ("vision/merger/input",)),
            ("fc1_bias", "linear_fc1.bias", (merger,), ()),
            (
                "fc2",
                "linear_fc2.weight",
                (output_width, merger),
                ("vision/merger/activation",),
            ),
            ("fc2_bias", "linear_fc2.bias", (output_width,), ()),
        ):
            self.add(
                "vision/merger/" + role,
                store,
                "model.visual.merger." + field,
                dims,
                inputs=inputs,
            )

    def draft(self, store, config, target, backend):
        h, d = target["hidden_size"], config["head_dim"]
        q, k = config["num_attention_heads"] * d, config["num_key_value_heads"] * d
        draft = config["dflash_config"]
        self.add(
            backend + "/feature_projection",
            store,
            "fc.weight",
            (h, len(draft["target_layer_ids"]) * h),
            inputs=(backend + "/target_features",),
        )
        self.add(backend + "/context_norm", store, "hidden_norm.weight", (h,))
        self.add(backend + "/final_norm", store, "norm.weight", (h,))
        for i in range(config["num_hidden_layers"]):
            p, sp = f"{backend}/layers/{i}/", f"layers.{i}."
            self.add(p + "input_norm", store, sp + "input_layernorm.weight", (h,))
            self.add(
                p + "post_attention_norm",
                store,
                sp + "post_attention_layernorm.weight",
                (h,),
            )
            for role, field, rows in (
                ("query", "q_proj", q),
                ("key", "k_proj", k),
                ("value", "v_proj", k),
            ):
                self.add(
                    p + "attention/" + role,
                    store,
                    sp + "self_attn." + field + ".weight",
                    (rows, h),
                    inputs=(p + "query_projection_input",),
                )
                if role in ("key", "value"):
                    self.add(
                        p + "attention/context_" + role,
                        store,
                        sp + "self_attn." + field + ".weight",
                        (rows, h),
                        inputs=(backend + "/context_input",),
                    )
            self.group(*(p + "attention/" + r for r in ("query", "key", "value")))
            for role, field in (("query_norm", "q_norm"), ("key_norm", "k_norm")):
                self.add(
                    p + "attention/" + role,
                    store,
                    sp + "self_attn." + field + ".weight",
                    (d,),
                )
            self.add(
                p + "attention/output",
                store,
                sp + "self_attn.o_proj.weight",
                (h, q),
                inputs=(p + "attention_output",),
            )
            self.dense(p, sp, store, h, config["intermediate_size"], draft=True)
            if backend == "dflash2":
                taps, groups = draft["conv_kernel_size"], h // draft["conv_group_size"]
                for branch in ("attention_conv", "mlp_conv"):
                    self.add(
                        p + branch + "/base_kernel",
                        store,
                        sp + branch + ".base_kernel",
                        (2, taps, h),
                    )
                    self.add(
                        p + branch + "/kernel_projection",
                        store,
                        sp + branch + ".kernel_projection.weight",
                        (2 * taps * groups, h),
                        inputs=(p + branch + "_input",),
                    )
        if backend == "dflash2":
            rank = draft["selector_rank"]
            self.add(
                backend + "/candidate_selector/hidden_projection",
                store,
                "candidate_selector.hidden_projection.weight",
                (rank, h),
                inputs=(backend + "/final_hidden",),
            )
            for role in ("predecessor_codebook", "successor_codebook"):
                self.add(
                    backend + "/candidate_selector/" + role,
                    store,
                    "candidate_selector." + role,
                    (target["vocab_size"], rank),
                )


def _patch_embed_composite(hf_name0: str, hf_name1: str, hf_shape: tuple[int, ...]):
    def build(source) -> tuple[torch.Tensor, tuple[int, ...]]:
        out_channels, in_channels, temporal, patch_h, patch_w = hf_shape
        slice0 = source.read_flat(hf_name0).reshape(out_channels, in_channels, patch_h, patch_w)
        slice1 = source.read_flat(hf_name1).reshape(out_channels, in_channels, patch_h, patch_w)
        stacked = torch.stack([slice0, slice1], dim=2)
        return stacked.reshape(-1), hf_shape

    return build


def qwen35_vision_name_map(
    num_blocks: int,
    *,
    hidden_size: int,
    in_channels: int = 3,
    temporal_patch_size: int = 2,
    patch_size: int,
    hf_prefix: str = "model.visual.",
) -> tuple[dict[str, str], dict[str, Transform], dict, dict]:
    """HF -> GGUF (name_map, transforms, shapes, composites) for the Qwen3.5
    vision tower + qwen3vl_merger projector, as exported by llama.cpp into a
    separate "mmproj-*.gguf" file (architecture "clip", not the text model's
    "qwen35"/"qwen35moe").

    Shared by both Qwen families' vision components (the towers are the same
    Qwen3-VL architecture). Verified end-to-end against a real checkpoint/mmproj pair on this
    machine: block norms (LayerNorm, not RMSNorm -- unlike the text model,
    none of these get a +1 shift), the merger's post-tower LayerNorm
    ("merger.norm" -> "v.post_ln", not obviously the same tensor by name
    alone), and the patch-embed Conv3d kernel's temporal-depth split (HF
    stores one (out, in, temporal_patch_size, patch, patch) kernel; the
    exporter splits it into "v.patch_embd.weight" and ".weight.1", one 2D
    kernel per temporal slice, reassembled here via a `composites` entry)
    all reproduced the original tensor exactly (F32/F16 companions) or
    within Q8_0's own quantization noise (matrix weights).

    Does not cover deepstack layers (this checkpoint's
    clip.vision.is_deepstack_layers were all false) -- extend for those if
    your model uses them.
    """
    hf = hf_prefix
    name_map = {
        hf + "patch_embed.proj.bias": "v.patch_embd.bias",
        hf + "pos_embed.weight": "v.position_embd.weight",
        hf + "merger.norm.weight": "v.post_ln.weight",
        hf + "merger.norm.bias": "v.post_ln.bias",
        hf + "merger.linear_fc1.weight": "mm.0.weight",
        hf + "merger.linear_fc1.bias": "mm.0.bias",
        hf + "merger.linear_fc2.weight": "mm.2.weight",
        hf + "merger.linear_fc2.bias": "mm.2.bias",
    }
    for i in range(num_blocks):
        b = f"{hf}blocks.{i}."
        v = f"v.blk.{i}."
        name_map.update(
            {
                b + "norm1.weight": v + "ln1.weight",
                b + "norm1.bias": v + "ln1.bias",
                b + "norm2.weight": v + "ln2.weight",
                b + "norm2.bias": v + "ln2.bias",
                b + "attn.qkv.weight": v + "attn_qkv.weight",
                b + "attn.qkv.bias": v + "attn_qkv.bias",
                b + "attn.proj.weight": v + "attn_out.weight",
                b + "attn.proj.bias": v + "attn_out.bias",
                b + "mlp.linear_fc1.weight": v + "ffn_up.weight",
                b + "mlp.linear_fc1.bias": v + "ffn_up.bias",
                b + "mlp.linear_fc2.weight": v + "ffn_down.weight",
                b + "mlp.linear_fc2.bias": v + "ffn_down.bias",
            }
        )
    # HF stores one (hidden_size, in_channels, temporal_patch_size, patch,
    # patch) Conv3d kernel; the exporter splits it into one 2D
    # (hidden_size, in_channels, patch, patch) kernel per temporal slice
    # ("v.patch_embd.weight" / ".weight.1"). Only temporal_patch_size == 2
    # is implemented, matching every Qwen3-VL checkpoint seen so far.
    if temporal_patch_size != 2:
        raise ValueError(
            f"qwen35_vision_name_map only supports temporal_patch_size=2, "
            f"got {temporal_patch_size}"
        )
    patch_shape = (hidden_size, in_channels, temporal_patch_size, patch_size, patch_size)
    composites = {
        hf
        + "patch_embed.proj.weight": _patch_embed_composite(
            "v.patch_embd.weight", "v.patch_embd.weight.1", patch_shape
        )
    }
    return name_map, {}, {}, composites
