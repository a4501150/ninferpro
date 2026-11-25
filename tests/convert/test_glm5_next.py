from __future__ import annotations

import copy
import json
import struct

import pytest
import torch

from tools.convert.glm5_next import build_model, text_config, vision_config
from tools.convert.official_recipes import RECIPES
from tools.convert.recipe import Recipe
from tools.convert.sources.nvfp4_banks import import_expert_bank, weight_only_bank
from tools.convert.sources.safetensors import SafetensorsSource

H, V, E, I_MOE, I_DENSE = 32, 8, 4, 16, 48
HEADS, D = 2, 16  # KDA heads and head width
MLA_HEADS, DN, DV, Q_RANK, KV_RANK = 2, 16, 16, 24, 16
IH, ID = 2, 16  # indexer heads and head width
VH, VHEADS, VI, VO, VPI = 8, 2, 12, 16, 20  # tiny vision tower
VP, VT, VM = 2, 2, 2  # its patch, temporal and merge sizes


def _config() -> dict:
    layers = ["linear_attention", "deepseek_sparse_attention"]
    return {
        "architectures": ["Glm5NextForConditionalGeneration"],
        "model_type": "glm5_next",
        "image_token_id": 2, "image_start_token_id": 3, "image_end_token_id": 4,
        "video_token_id": 5, "video_start_token_id": 6, "video_end_token_id": 7,
        "vision_config": {
            "attention_bias": True, "attention_dropout": 0.0, "depth": 1, "hidden_act": "silu",
            "hidden_size": VH, "image_size": 8, "in_channels": 3, "intermediate_size": VI,
            "model_type": "glm5_next_vision", "num_heads": VHEADS, "out_hidden_size": VO,
            "patch_size": VP, "projection_intermediate_size": VPI, "rms_norm_eps": 1e-5,
            "spatial_merge_size": VM, "swiglu_limit": 10.0, "temporal_patch_size": VT,
        },
        "text_config": {
            "attention_bias": False, "eos_token_id": [5, 6], "first_k_dense_replace": 1, "hc_eps": 1e-6,
            "hc_mult": 4, "hc_sinkhorn_iters": 20, "hidden_act": "silu", "hidden_size": H, "index_head_dim": ID,
            "index_kpool": 4, "index_kpool_always_select_tail": True, "index_kpool_compress": True,
            "index_n_heads": IH, "index_topk": 16, "indexer_types": ["full", "full"], "intermediate_size": I_DENSE,
            "kv_lora_rank": KV_RANK, "layer_types": layers,
            "linear_attn_config": {"num_heads": HEADS, "head_dim": D, "short_conv_kernel_size": 4,
                                   "gate_lower_bound": -5.0, "kda_layers": [0], "full_attn_layers": [1]},
            "max_position_embeddings": 1024, "mhc": True, "mla_use_nope": True, "mlp_layer_types": ["dense", "sparse"],
            "moe_intermediate_size": I_MOE, "moe_router_dtype": "float32", "n_group": 1, "n_routed_experts": E,
            "n_shared_experts": 1, "norm_topk_prob": True, "num_attention_heads": MLA_HEADS, "num_experts_per_tok": 2,
            "num_hidden_layers": 2, "num_key_value_heads": MLA_HEADS, "q_lora_rank": Q_RANK, "qk_head_dim": DN,
            "qk_nope_head_dim": DN, "qk_rope_head_dim": 0, "rms_norm_eps": 1e-5, "routed_scaling_factor": 2.5,
            "scoring_func": "sigmoid", "swiglu_limit": 10.0, "tie_word_embeddings": False, "topk_group": 1,
            "topk_method": "noaux_tc", "v_head_dim": DV, "vocab_size": V,
        },
    }


def _tensors(mtp: bool = False, vision: bool = False) -> dict[str, tuple[str, tuple[int, ...]]]:
    width, rows = 4 * H, 24
    p = "model.language_model."
    out = {p + "embed_tokens.weight": ("BF16", (V, H)), "lm_head.weight": ("BF16", (V, H)),
           p + "norm.weight": ("BF16", (H,))}
    if vision:
        _vision_tensors(out)
    for layer in range(2):
        lp = f"{p}layers.{layer}."
        for hc in ("hc_attn", "hc_ffn"):
            out[lp + hc + "_fn"] = ("BF16", (rows, width))
            out[lp + hc + "_base"] = ("F32", (rows,))
            out[lp + hc + "_scale"] = ("F32", (3,))
        out[lp + "input_layernorm.weight"] = ("BF16", (H,))
        out[lp + "post_attention_layernorm.weight"] = ("BF16", (H,))
    kda = p + "layers.0.self_attn."
    for field in ("q", "k", "v"):
        out[kda + field + "_proj.weight"] = ("BF16", (HEADS * D, H))
        out[kda + field + "_conv1d.weight"] = ("BF16", (HEADS * D, 1, 4))
    out.update({kda + "f_a_proj.weight": ("BF16", (D, H)), kda + "f_b_proj.weight": ("BF16", (HEADS * D, D)),
                kda + "g_a_proj.weight": ("BF16", (D, H)), kda + "g_b_proj.weight": ("BF16", (HEADS * D, D)),
                kda + "b_proj.weight": ("BF16", (HEADS, H)), kda + "A_log": ("F32", (HEADS,)),
                kda + "dt_bias": ("F32", (HEADS * D,)), kda + "o_norm.weight": ("BF16", (D,)),
                kda + "o_proj.weight": ("BF16", (H, HEADS * D))})
    for field, shape in (("gate", (I_DENSE, H)), ("up", (I_DENSE, H)), ("down", (H, I_DENSE))):
        out[f"{p}layers.0.mlp.{field}_proj.weight"] = ("BF16", shape)
    _sparse_attention(out, f"{p}layers.1.self_attn.")
    if mtp:
        _mtp_layer(out, p + "layers.2.")
    moe = p + "layers.1.mlp."
    out.update({moe + "gate.weight": ("BF16", (E, H)), moe + "gate.e_score_correction_bias": ("F32", (E,))})
    for field, shape in (("gate", (I_MOE, H)), ("up", (I_MOE, H)), ("down", (H, I_MOE))):
        out[f"{moe}shared_experts.{field}_proj.weight"] = ("BF16", shape)
        for e in range(E):
            leaf = f"{moe}experts.{e}.{field}_proj."
            out[leaf + "weight_packed"] = ("U8", (shape[0], shape[1] // 2))
            out[leaf + "weight_scale"] = ("F8_E4M3", (shape[0], shape[1] // 16))
            out[leaf + "weight_global_scale"] = ("F32", (1,))
    return out


def _mtp_layer(out: dict, mtp: str) -> None:
    """The checkpoint's MTP layer: BF16 throughout, its routed experts as plain matrices."""
    _sparse_attention(out, mtp + "self_attn.")
    for field in ("enorm", "hnorm", "shared_head.norm", "input_layernorm", "post_attention_layernorm"):
        out[mtp + field + ".weight"] = ("BF16", (H,))
    out[mtp + "eh_proj.weight"] = ("BF16", (H, 2 * H))
    out.update({mtp + "mlp.gate.weight": ("BF16", (E, H)), mtp + "mlp.gate.e_score_correction_bias": ("F32", (E,))})
    for field, shape in (("gate", (I_MOE, H)), ("up", (I_MOE, H)), ("down", (H, I_MOE))):
        out[f"{mtp}mlp.shared_experts.{field}_proj.weight"] = ("BF16", shape)
        for e in range(E):
            out[f"{mtp}mlp.experts.{e}.{field}_proj.weight"] = ("BF16", shape)


def _vision_tensors(out: dict) -> None:
    """The tiny vision tower: one block, the merger's SwiGLU projection, the downsample conv."""
    p = "model.visual."
    out[p + "patch_embed.proj.weight"] = ("BF16", (VH, 3, VT, VP, VP))
    out[p + "patch_embed.proj.bias"] = ("BF16", (VH,))
    b = p + "blocks.0."
    out.update({b + "norm1.weight": ("BF16", (VH,)), b + "norm2.weight": ("BF16", (VH,)),
                b + "attn.qkv.weight": ("BF16", (3 * VH, VH)), b + "attn.qkv.bias": ("BF16", (3 * VH,)),
                b + "attn.q_norm.weight": ("BF16", (VH // VHEADS,)),
                b + "attn.k_norm.weight": ("BF16", (VH // VHEADS,)),
                b + "attn.proj.weight": ("BF16", (VH, VH)), b + "attn.proj.bias": ("BF16", (VH,))})
    for field, shape in (("gate", (VI, VH)), ("up", (VI, VH)), ("down", (VH, VI))):
        out[f"{b}mlp.{field}_proj.weight"] = ("BF16", shape)
        out[f"{b}mlp.{field}_proj.bias"] = ("BF16", (shape[0],))
    out.update({p + "post_layernorm.weight": ("BF16", (VH,)),
                p + "downsample.weight": ("BF16", (VO, VH, VM, VM)),
                p + "downsample.bias": ("BF16", (VO,)),
                p + "merger.proj.weight": ("BF16", (VO, VO)),
                p + "merger.post_projection_norm.weight": ("BF16", (VO,)),
                p + "merger.post_projection_norm.bias": ("BF16", (VO,)),
                p + "merger.gate_proj.weight": ("BF16", (VPI, VO)),
                p + "merger.up_proj.weight": ("BF16", (VPI, VO)),
                p + "merger.down_proj.weight": ("BF16", (VO, VPI))})


def _sparse_attention(out: dict, mla: str) -> None:
    out.update({mla + "q_a_proj.weight": ("BF16", (Q_RANK, H)), mla + "q_a_layernorm.weight": ("BF16", (Q_RANK,)),
                mla + "q_b_proj.weight": ("BF16", (MLA_HEADS * DN, Q_RANK)),
                mla + "kv_a_proj_with_mqa.weight": ("BF16", (KV_RANK, H)),
                mla + "kv_a_layernorm.weight": ("BF16", (KV_RANK,)),
                mla + "kv_b_proj.weight": ("BF16", (MLA_HEADS * (DN + DV), KV_RANK)),
                mla + "o_proj.weight": ("BF16", (H, MLA_HEADS * DV))})
    ix = mla + "indexer."
    out.update({ix + "wq_b.weight": ("BF16", (IH * ID, Q_RANK)), ix + "wk.weight": ("BF16", (ID, H)),
                ix + "k_norm.weight": ("BF16", (ID,)), ix + "k_norm.bias": ("BF16", (ID,)),
                ix + "index_kpool_compress_gate": ("BF16", (ID, H)),
                ix + "index_kpool_compress_ape": ("BF16", (4, ID)), ix + "weights_proj.weight": ("BF16", (IH, H))})


_BYTES = {"BF16": 2, "F32": 4, "U8": 1, "F8_E4M3": 1}


def _processor_config() -> dict:
    def processor(kind: str, max_tokens: int) -> dict:
        return {"patch_size": VP, "temporal_patch_size": VT, "merge_size": VM,
                "image_mean": [0.5, 0.5, 0.5], "image_std": [0.25, 0.25, 0.25],
                "min_image_tokens": 4, "max_image_tokens": max_tokens, f"{kind}_processor_type": kind.title()}

    return {"image_processor": processor("image", 8000), "video_processor": processor("video", 240000),
            "processor_class": "Glm5NextProcessor"}


def _write_checkpoint(root, config: dict, *, mtp: bool = False, vision: bool = False) -> None:
    header, blobs, offset = {}, [], 0
    for name, (dtype, shape) in sorted(_tensors(mtp, vision).items()):
        count = 1
        for n in shape:
            count *= n
        if dtype == "F32":
            data = struct.pack(f"<{count}f", *([1.0] * count))
        elif dtype == "F8_E4M3":
            data = bytes([0x38]) * count  # E4M3 1.0
        elif dtype == "U8":
            data = bytes(i % 256 for i in range(count))
        else:
            data = ((torch.arange(count) % 251) * 0.0625).to(torch.bfloat16).view(torch.uint8).numpy().tobytes()
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + len(data)]}
        blobs.append(data)
        offset += len(data)
    raw = json.dumps(header).encode()
    (root / "model.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw + b"".join(blobs))
    (root / "config.json").write_text(json.dumps(config))
    if vision:
        (root / "processor_config.json").write_text(json.dumps(_processor_config()))
    vocab = {f"t{i}": i for i in range(V)}
    (root / "tokenizer.json").write_text(json.dumps({"model": {"type": "BPE", "vocab": vocab}}))
    (root / "tokenizer_config.json").write_text(json.dumps({"added_tokens_decoder": {}}))
    (root / "chat_template.jinja").write_text("{{ messages }}")
    (root / "generation_config.json").write_text("{}")


def test_text_config_reads_the_upstream_fields() -> None:
    config = text_config(_config())
    assert config["architectures"] == ["Glm5NextForCausalLM"]
    assert config["layer_types"] == ["linear_attention", "sparse_attention"]
    assert (config["linear_num_heads"], config["linear_head_dim"], config["linear_lower_bound"]) == (HEADS, D, -5.0)
    assert config["eos_token_id"] == [5, 6]


@pytest.mark.parametrize(
    "path, value",
    [
        (("qk_rope_head_dim",), 64),
        (("indexer_types",), ["full", "shared"]),
        (("mlp_layer_types",), ["sparse", "sparse"]),
        (("topk_method",), "greedy"),
        (("linear_attn_config", "kda_layers"), [0, 1]),
        (("linear_attn_config", "gate_lower_bound"), -6.0),
        (("num_key_value_heads",), 1),
    ],
)
def test_text_config_refuses_other_mathematics(path, value) -> None:
    config = copy.deepcopy(_config())
    target = config["text_config"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        text_config(config)


def test_build_model_binds_every_block_and_the_recipe_imports_the_banks(tmp_path) -> None:
    _write_checkpoint(tmp_path, _config())
    with SafetensorsSource(tmp_path) as store:
        model = build_model(store)
        names = set(model.parameters)
        for name in ("text/layers/0/kda/convolution", "text/layers/0/mlp/down",
                     "text/layers/1/attention/key_up", "text/layers/1/attention/index_position",
                     "text/layers/1/moe/router_bias", "text/layers/1/ffn_hc/fn", "text/final_norm"):
            assert name in names
        assert model.parameters["text/layers/0/kda/convolution"].shape == (4, 3 * HEADS * D)
        assert model.parameters["text/layers/1/attention/key_up"].shape == (MLA_HEADS, KV_RANK, DN)
        assert model.parameters["text/layers/1/attn_hc/base"].direct_format == "fp32"
        assert model.parameters["text/layers/1/attention/index_position"].direct_format == "fp32"
        bank = model.parameters["text/layers/1/moe/experts"]
        assert bank.shape == (E, H, I_MOE) and weight_only_bank(bank.source)
        recipe = Recipe(model)
        RECIPES["glm5_3_flash_nvfp4_orcarouter"](model, recipe, {"base": store})
        selection = recipe.selections["text/layers/1/moe/experts"][0]
        assert (selection.format, selection.method) == ("nvfp4_mul", import_expert_bank)
        assert "mtp" not in model.components
        bulk = ("0/kda/query", "0/kda/key", "0/kda/value", "0/kda/output", "0/mlp/gate", "0/mlp/up",
                "0/mlp/down", "1/attention/query_down", "1/attention/latent", "1/attention/query_up",
                "1/attention/output", "1/moe/shared/gate", "1/moe/shared/up", "1/moe/shared/down")
        for name in bulk:
            assert recipe.selections[f"text/layers/{name}"][0].format == "bf16", name
        # The Q8-dense recipe stores the bulk dense projections in Q8; the routers, KDA control
        # projections, absorbed MLA operands, indexer, embedding and head stay BF16.
        q8 = Recipe(model)
        RECIPES["glm5_3_flash_nvfp4_orcarouter_q8dense"](model, q8, {"base": store})
        assert q8.selections["text/layers/1/moe/experts"][0].format == "nvfp4_mul"
        for name in bulk:
            assert q8.selections[f"text/layers/{name}"][0].format == "q8_g32_fp16", name
        for name in ("layers/0/kda/forget_down", "layers/0/kda/gate_up", "layers/0/kda/beta",
                     "layers/1/attention/key_up", "layers/1/attention/index_query", "layers/1/attention/index_key",
                     "layers/1/moe/router", "token_embedding", "output_head"):
            assert q8.selections[f"text/{name}"][0].format == "bf16", name


def test_a_checkpoint_mtp_layer_is_stored_as_bf16(tmp_path) -> None:
    config = _config()
    config["text_config"]["num_nextn_predict_layers"] = 1
    _write_checkpoint(tmp_path, config, mtp=True)
    with SafetensorsSource(tmp_path) as store:
        model = build_model(store)
        assert "mtp" in model.components and model.parameters["mtp/input_projection"].shape == (H, 2 * H)
        assert model.parameters["mtp/layers/0/moe/experts/3/down"].shape == (H, I_MOE)
        recipe = Recipe(model)
        RECIPES["glm5_3_flash_nvfp4_orcarouter"](model, recipe, {"base": store})
        assert recipe.selections["mtp/layers/0/moe/experts/3/down"][0].format == "bf16"


def test_kda_convolution_holds_the_three_kernels_in_projection_order(tmp_path) -> None:
    _write_checkpoint(tmp_path, _config())
    with SafetensorsSource(tmp_path) as store:
        model = build_model(store)
        conv = model.parameters["text/layers/0/kda/convolution"].source.values().reshape(4, 3 * HEADS * D)
        for index, field in enumerate("qkv"):
            kernel = store.read_flat(f"model.language_model.layers.0.self_attn.{field}_conv1d.weight")
            expected = kernel.reshape(HEADS * D, 4).T
            assert torch.equal(conv[:, index * HEADS * D:(index + 1) * HEADS * D].float(), expected.float())


def test_key_and_value_up_store_the_absorbed_mla_operands(tmp_path) -> None:
    _write_checkpoint(tmp_path, _config())
    with SafetensorsSource(tmp_path) as store:
        model = build_model(store)
        key = model.parameters["text/layers/1/attention/key_up"]
        value = model.parameters["text/layers/1/attention/value_up"]
        assert (key.shape, value.shape) == ((MLA_HEADS, KV_RANK, DN), (MLA_HEADS, DV, KV_RANK))
        assert key.inputs == () and value.inputs == ()
        kv_b = store.read_flat("model.language_model.layers.1.self_attn.kv_b_proj.weight").reshape(
            MLA_HEADS, DN + DV, KV_RANK)
        # W_uk transposed per head (each row one latent dimension's query weights), W_uv as stored.
        assert torch.equal(key.source.values().reshape(MLA_HEADS, KV_RANK, DN).float(),
                           kv_b[:, :DN].transpose(1, 2).float())
        assert torch.equal(value.source.values().reshape(MLA_HEADS, DV, KV_RANK).float(),
                           kv_b[:, DN:].float())


def test_vision_config_reads_the_tower_fields_and_media_tokens() -> None:
    config = _config()
    record = vision_config(config, text_config(config))
    assert record["model_type"] == "glm5_next_vision"
    assert (record["depth"], record["hidden_size"], record["num_heads"], record["out_hidden_size"]) == \
        (1, VH, VHEADS, VO)
    assert record["image_token_id"] == 2 and record["video_end_token_id"] == 7
    assert record["rope_parameters"] == {"rope_type": "axial"}
    config["vision_config"]["rope_parameters"] = {"rope_type": "axial", "rope_theta": 10000.0}
    assert vision_config(config, text_config(config))["rope_theta"] == 10000.0


@pytest.mark.parametrize(
    "path, value",
    [
        (("hidden_act",), "gelu_pytorch_tanh"),
        (("attention_bias",), False),
        (("in_channels",), 4),
        (("model_type",), "glm4_vision"),
        (("num_heads",), 3),  # 8 % 3
        (("image_size",), 9),  # not a multiple of patch_size 2
    ],
)
def test_vision_config_refuses_other_towers(path, value) -> None:
    config = copy.deepcopy(_config())
    config["vision_config"][path[-1]] = value
    with pytest.raises(ValueError):
        vision_config(config, text_config(config))


def test_the_vision_component_binds_the_tower_and_its_processor(tmp_path) -> None:
    _write_checkpoint(tmp_path, _config(), vision=True)
    with SafetensorsSource(tmp_path) as store:
        model = build_model(store, components=("text", "vision"))
        assert set(model.components) == {"text", "vision"}
        assert model.components["vision"]["target"] == "text"
        for name, shape in (
            ("vision/patch_embedding", (VH, 3 * VT * VP * VP)),
            ("vision/patch_embedding_bias", (VH,)),
            ("vision/layers/0/norm1", (VH,)),
            ("vision/layers/0/attention/query", (VH, VH)),
            ("vision/layers/0/attention/key_bias", (VH,)),
            ("vision/layers/0/attention/query_norm", (VH // VHEADS,)),
            ("vision/layers/0/attention/output", (VH, VH)),
            ("vision/layers/0/mlp/gate", (VI, VH)),
            ("vision/layers/0/mlp/down_bias", (VH,)),
            ("vision/post_norm", (VH,)),
            ("vision/downsample", (VO, VH, VM, VM)),
            ("vision/downsample_bias", (VO,)),
            ("vision/merger/projection", (VO, VO)),
            ("vision/merger/post_projection_norm", (VO,)),
            ("vision/merger/gate", (VPI, VO)),
            ("vision/merger/down", (VO, VPI)),
        ):
            assert model.parameters[name].shape == shape, name
            assert model.parameters[name].direct_format == "bf16", name
        assert model.parameters["vision/patch_embedding"].inputs == ("vision/patch_input",)
        assert model.parameters["vision/merger/down"].inputs == ("vision/merger/product",)
        groups = model.packing_groups
        assert ("vision/layers/0/attention/query", "vision/layers/0/attention/key",
                "vision/layers/0/attention/value") in groups
        assert ("vision/layers/0/attention/query_bias", "vision/layers/0/attention/key_bias",
                "vision/layers/0/attention/value_bias") in groups
        assert ("vision/layers/0/mlp/gate", "vision/layers/0/mlp/up") in groups
        assert ("vision/merger/gate", "vision/merger/up") in groups
        # The processor resource keeps the checkpoint's bytes verbatim.
        raw = (tmp_path / "processor_config.json").read_bytes()
        assert model.resources == {"resource/vision/processor_config.json": raw,
                                   "resource/text/tokenizer.json": (tmp_path / "tokenizer.json").read_bytes(),
                                   "resource/text/tokenizer_config.json":
                                       (tmp_path / "tokenizer_config.json").read_bytes(),
                                   "resource/text/chat_template.jinja": (tmp_path / "chat_template.jinja").read_bytes(),
                                   "resource/text/generation_config.json":
                                       (tmp_path / "generation_config.json").read_bytes()}
        # The official recipe assigns nothing in the tower: every vision selection is BF16 direct.
        recipe = Recipe(model)
        RECIPES["glm5_3_flash_nvfp4_orcarouter"](model, recipe, {"base": store})
        for name, selection in recipe.selections.items():
            if name.startswith("vision/"):
                assert selection[0].format == "bf16", name


def test_the_vision_component_reads_the_stored_qkv_and_patch_values(tmp_path) -> None:
    _write_checkpoint(tmp_path, _config(), vision=True)
    with SafetensorsSource(tmp_path) as store:
        model = build_model(store, components=("text", "vision"))
        for number, role in enumerate(("query", "key", "value")):
            block = model.parameters[f"vision/layers/0/attention/{role}"]
            source = store.read_flat("model.visual.blocks.0.attn.qkv.weight").reshape(3 * VH, VH)
            assert torch.equal(block.source.values().reshape(VH, VH).float(),
                               source[number * VH:(number + 1) * VH].float())
            bias = model.parameters[f"vision/layers/0/attention/{role}_bias"]
            source_bias = store.read_flat("model.visual.blocks.0.attn.qkv.bias")
            assert torch.equal(bias.source.values().float(), source_bias[number * VH:(number + 1) * VH].float())
        q_norm = model.parameters["vision/layers/0/attention/query_norm"]
        assert torch.equal(q_norm.source.values().float(),
                           store.read_flat("model.visual.blocks.0.attn.q_norm.weight").float())
        patch = model.parameters["vision/patch_embedding"]
        kernel = store.read_flat("model.visual.patch_embed.proj.weight").reshape(VH, 3 * VT * VP * VP)
        assert torch.equal(patch.source.values().reshape(VH, 3 * VT * VP * VP).float(), kernel.float())


def test_a_checkpoint_without_a_tower_converts_text_only(tmp_path) -> None:
    _write_checkpoint(tmp_path, _config())
    with SafetensorsSource(tmp_path) as store:
        model = build_model(store, components=("text",))
        assert "vision" not in model.components
        assert not any(k.startswith("resource/vision/") for k in model.resources)
        with pytest.raises(ValueError):
            build_model(store, components=("text", "vision"))
