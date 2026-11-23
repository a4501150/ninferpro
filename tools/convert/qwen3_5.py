"""Qwen3.5 architecture adapter: mathematical config, source axes, and packing groups.

Dense/MoE text, Vision, MTP, DFlash, and DFlash2 share this explicit mapping.
Official checkpoint names and quantization assignments live in official_recipes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from .config_values import _f32, _fixed, _positive
from .model import Model
from .qwen_common import (
    _Builder as _SharedBuilder,
    _rope_source,
    draft_config,
    normalize_tokenizer,
    vision_config,
)
from .resources import load_resources
from .sources.safetensors import SafetensorsSource

_TEXT_ARCHITECTURES = {
    "Qwen3_5ForCausalLM": False,
    "Qwen3_5ForConditionalGeneration": False,
    "Qwen3_5MoeForCausalLM": True,
    "Qwen3_5MoeForConditionalGeneration": True,
}


def text_config(source: dict, *, mtp: bool) -> dict:
    architectures = source.get("architectures")
    if (
        not isinstance(architectures, list)
        or len(architectures) != 1
        or architectures[0] not in _TEXT_ARCHITECTURES
    ):
        raise ValueError(f"unsupported Qwen text architecture {architectures!r}")
    moe = _TEXT_ARCHITECTURES[architectures[0]]
    raw = source.get("text_config", source)
    _fixed(raw, "hidden_act", "silu", "text")
    _fixed(raw, "attention_bias", False, "text")
    _fixed(raw, "attn_output_gate", True, "text")
    result = {
        "architectures": ["Qwen3_5MoeForCausalLM" if moe else "Qwen3_5ForCausalLM"],
        "model_type": "qwen3_5_moe_text" if moe else "qwen3_5_text",
    }
    for key in (
        "hidden_size",
        "vocab_size",
        "num_hidden_layers",
        "max_position_embeddings",
    ):
        result[key] = _positive(raw.get(key), "text." + key)
    tied = raw.get("tie_word_embeddings", source.get("tie_word_embeddings", False))
    if type(tied) is not bool:
        raise ValueError("text.tie_word_embeddings must be boolean")
    result["tie_word_embeddings"] = tied
    result["rms_norm_eps"] = _f32(raw.get("rms_norm_eps", 1e-6), "text.rms_norm_eps")
    layers = raw.get("layer_types")
    if layers is None:
        interval = _positive(
            raw.get("full_attention_interval", 4), "full_attention_interval"
        )
        layers = [
            "full_attention" if (i + 1) % interval == 0 else "linear_attention"
            for i in range(result["num_hidden_layers"])
        ]
    if (
        not isinstance(layers, list)
        or len(layers) != result["num_hidden_layers"]
        or any(k not in ("full_attention", "linear_attention") for k in layers)
    ):
        raise ValueError("text.layer_types must describe every block")
    result["layer_types"] = list(layers)
    if "full_attention" in layers or mtp:
        for key in ("num_attention_heads", "num_key_value_heads", "head_dim"):
            result[key] = _positive(raw.get(key), "text." + key)
        if result["num_attention_heads"] % result["num_key_value_heads"]:
            raise ValueError("text attention heads must be divisible by KV heads")
        rope = _rope_source(raw, "text")
        _fixed(rope, "mrope_interleaved", True, "text.rope_parameters")
        factor = _f32(
            rope.get("partial_rotary_factor", raw.get("partial_rotary_factor", 0.25)),
            "partial_rotary_factor",
        )
        theta = _f32(
            rope.get("rope_theta", raw.get("rope_theta", 10_000_000)), "rope_theta"
        )
        sections = rope.get("mrope_section")
        if (
            not isinstance(sections, list)
            or len(sections) != 3
            or any(type(v) is not int or v < 0 for v in sections)
        ):
            raise ValueError("mrope_section must contain three nonnegative integers")
        rotary = int(result["head_dim"] * factor)
        if factor > 1 or rotary <= 0 or rotary % 2 or sum(sections) != rotary // 2:
            raise ValueError("MRoPE sections and rotary width disagree")
        axes = [
            (
                1
                if r % 3 == 1 and r < 3 * sections[1]
                else 2 if r % 3 == 2 and r < 3 * sections[2] else 0
            )
            for r in range(rotary // 2)
        ]
        if [axes.count(i) for i in range(3)] != sections:
            raise ValueError("MRoPE sections exceed the interleaved axis ranges")
        result["rope_parameters"] = {
            "rope_theta": theta,
            "partial_rotary_factor": factor,
            "mrope_section": list(sections),
        }
    if "linear_attention" in layers:
        _fixed(raw, "mamba_ssm_dtype", "float32", "text")
        for key in (
            "linear_num_key_heads",
            "linear_key_head_dim",
            "linear_num_value_heads",
            "linear_value_head_dim",
            "linear_conv_kernel_dim",
        ):
            result[key] = _positive(raw.get(key), "text." + key)
        if result["linear_num_value_heads"] % result["linear_num_key_heads"]:
            raise ValueError("linear value heads must be divisible by key heads")
    fields = (
        (
            "num_experts",
            "num_experts_per_tok",
            "moe_intermediate_size",
            "shared_expert_intermediate_size",
        )
        if moe
        else ("intermediate_size",)
    )
    for key in fields:
        result[key] = _positive(raw.get(key), "text." + key)
    if moe and result["num_experts_per_tok"] > result["num_experts"]:
        raise ValueError("selected experts exceed expert count")
    if mtp:
        _fixed(raw, "mtp_num_hidden_layers", 1, "text")
        _fixed(raw, "mtp_use_dedicated_embeddings", False, "text")
    return result


class _Builder(_SharedBuilder):
    """The family's builder: validate_source re-derives an additional source's config with
    Qwen3.5's text config."""

    text_config = staticmethod(text_config)


def build_model(
    base: SafetensorsSource,
    *,
    components: tuple[str, ...] = ("text",),
    companions: Mapping[str, SafetensorsSource] | None = None,
    resource_overrides: Mapping[str, str | Path] | None = None,
) -> Model:
    selected = set(components)
    if "text" not in selected or selected - {
        "text",
        "vision",
        "mtp",
        "dflash",
        "dflash2",
    }:
        raise ValueError("select text and supported optional components")
    companions = {} if companions is None else companions
    config = text_config(base.config, mtp="mtp" in selected)
    records = {"text": {"config": config}}
    if "vision" in selected:
        records["vision"] = {
            "config": vision_config(base.config, config),
            "target": "text",
        }
    if "mtp" in selected:
        architecture = "Qwen3_5MoeMTP" if "num_experts" in config else "Qwen3_5MTP"
        records["mtp"] = {"config": {"architectures": [architecture]}, "target": "text"}
    for backend in ("dflash", "dflash2"):
        if backend in selected:
            if backend not in companions:
                raise ValueError(f"selected {backend} requires its source")
            records[backend] = {
                "config": draft_config(companions[backend].config, config, backend),
                "target": "text",
            }
    refs, resources, count, special = load_resources(
        base.root,
        vocab_size=config["vocab_size"],
        vision_config=records["vision"]["config"] if "vision" in selected else None,
        overrides=resource_overrides,
        normalize=normalize_tokenizer,
    )
    for component, resource_refs in refs.items():
        records[component]["resources"] = resource_refs
    model = Model(
        records, resources=resources, token_count=count, special_token_ids=special
    )
    builder = _Builder(model)
    h, r = config["hidden_size"], config["vocab_size"]
    text_prefix = "model.language_model." if "text_config" in base.config else "model."
    builder.add(
        "text/token_embedding", base, text_prefix + "embed_tokens.weight", (r, h)
    )
    head_source = (
        text_prefix + "embed_tokens.weight"
        if config["tie_word_embeddings"]
        else "lm_head.weight"
    )
    head_inputs = ("text/final_hidden",) + tuple(
        c + "/final_hidden" for c in ("mtp", "dflash", "dflash2") if c in selected
    )
    builder.add("text/output_head", base, head_source, (r, h), inputs=head_inputs)
    builder.add("text/final_norm", base, text_prefix + "norm.weight", (h,))
    for i, kind in enumerate(config["layer_types"]):
        builder.block(
            f"text/layers/{i}/", text_prefix + f"layers.{i}.", base, config, kind
        )
    if "mtp" in selected:
        builder.add(
            "mtp/input_projection",
            base,
            "mtp.fc.weight",
            (h, 2 * h),
            inputs=("mtp/stem_input",),
        )
        for role, field in (
            ("embedding_norm", "pre_fc_norm_embedding"),
            ("hidden_norm", "pre_fc_norm_hidden"),
            ("final_norm", "norm"),
        ):
            builder.add("mtp/" + role, base, "mtp." + field + ".weight", (h,))
        builder.block("mtp/layers/0/", "mtp.layers.0.", base, config, "full_attention")
    if "vision" in selected:
        builder.vision(base, records["vision"]["config"], h)
    for backend in ("dflash", "dflash2"):
        if backend in selected:
            if (
                backend == "dflash2"
                and records[backend]["config"]["dflash_config"]["selector_top_k"]
                > count
            ):
                raise ValueError("DFlash2 selector_top_k exceeds public token domain")
            builder.draft(
                companions[backend], records[backend]["config"], config, backend
            )
    return model
