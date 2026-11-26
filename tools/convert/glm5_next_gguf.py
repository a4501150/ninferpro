"""Glm5Next (GLM-5.3-Flash) from a llama.cpp GGUF (arch ``glm5-next``), imported bit-exactly.

The logical model is the one ``glm5_next.build_model`` describes (the same parameter names and
shapes); the config, tokenizer, frontend resources and, with the ``vision`` component, the vision
tower come from the Hugging Face checkpoint given as ``--model``, whose metadata the GGUF must
agree with. Every text tensor comes from the GGUF, read exactly:

* the routed experts as ``ggml_expert_*_v1`` banks (the stored Q4_K/Q3_K/Q2_K blocks, verbatim);
* Q8_0 matrices as their stored rows (``q8_g32_fp16`` holds the same codes and binary16 scales);
* BF16 tensors as they are, and F32 ones as BF16 values (every value must be exact in BF16) or
  FP32 where the engine binds FP32;
* llama.cpp's export transforms undone: the KDA ``q``/``k``/``v`` depthwise kernels fused as
  ``glm5_next._Builder.kda_convolution`` fuses them; MLA's ``k_b``/``v_b`` are already the
  absorbed per-head blocks ``latent_up`` derives.

Three tensors come from the checkpoint instead, each checked against the GGUF over every element:

* ``text/output_head``: the engine's head is BF16 (the logits, the DFlash2 drafter and causal
  scoring read its rows), and the GGUF's Q8_0 head must be the Q8_0 rounding of the checkpoint's
  (every value within half a step of its block scale);
* ``attention/query_down``: it packs with ``latent`` (BF16 in the GGUF) into one parent, so it is
  stored BF16, under the same check against the GGUF's Q8_0 ``attn_q_a``;
* ``kda/a_log``: the GGUF stores A = -exp(A_log) in binary32; -exp of the checkpoint's A_log must
  be within one binary32 ulp of it (llama.cpp's exp rounds some values the other way).

The token embedding is a row gather the engine binds BF16: each row is the GGUF's Q8_0 value
d * code rounded once to BF16, the value the Q8 gather would produce. The GGUF holds no MTP
layer, so the artifact records none.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from . import glm5_next
from .model import Model, Parameter
from .resources import load_resources
from .sources.ggml_sources import GgmlExpertBank, ggml_or_plain, plain
from .sources.gguf_raw import GgufStore
from .sources.logical import LogicalSource
from .sources.safetensors import SafetensorsSource

ARCHITECTURE = "glm5-next"
_ROW_CHUNK = 4096


def _metadata_checks(store: GgufStore, config: dict) -> None:
    m = store.metadata
    if m.get("general.architecture") != ARCHITECTURE:
        raise ValueError(f"{store.path}: GGUF architecture {m.get('general.architecture')!r} is not {ARCHITECTURE}")
    a = ARCHITECTURE + "."
    layers = config["num_hidden_layers"]
    expected = {
        a + "block_count": layers,
        a + "embedding_length": config["hidden_size"],
        a + "vocab_size": config["vocab_size"],
        a + "feed_forward_length": config["intermediate_size"],
        a + "expert_feed_forward_length": config["moe_intermediate_size"],
        a + "expert_count": config["n_routed_experts"],
        a + "expert_used_count": config["num_experts_per_tok"],
        a + "expert_shared_count": 1,
        a + "leading_dense_block_count": config["first_k_dense_replace"],
        a + "attention.head_count": config["num_attention_heads"],
        a + "attention.q_lora_rank": config["q_lora_rank"],
        a + "attention.kv_lora_rank": config["kv_lora_rank"],
        a + "attention.key_length_mla": config["qk_nope_head_dim"],
        a + "attention.value_length_mla": config["v_head_dim"],
        a + "attention.indexer.head_count": config["index_n_heads"],
        a + "attention.indexer.key_length": config["index_head_dim"],
        a + "attention.indexer.top_k": config["index_topk"],
        a + "attention.indexer.kpool": config["index_kpool"],
        a + "hyper_connection.count": config["hc_mult"],
        a + "hyper_connection.sinkhorn_iterations": config["hc_sinkhorn_iters"],
        a + "kda.head_dim": config["linear_head_dim"],
        a + "ssm.conv_kernel": config["linear_conv_kernel_dim"],
        a + "rope.dimension_count": 0,
    }
    for key, value in expected.items():
        if m.get(key) != value:
            raise ValueError(f"{store.path}: {key} = {m.get(key)!r}, the checkpoint config says {value!r}")
    kinds = [1 if kind == "sparse_attention" else 0 for kind in config["layer_types"]]
    if m.get(a + "attention.head_count_kv") != kinds:
        raise ValueError(f"{store.path}: the MLA layers (head_count_kv) differ from the config's layer_types")
    for key in (a + "swiglu_clamp_exp", a + "swiglu_clamp_shexp"):
        if m.get(key) != [config["swiglu_limit"]] * layers:
            raise ValueError(f"{store.path}: {key} differs from the config's swiglu_limit")
    if not np.isclose(m.get(a + "expert_weights_scale", 0.0), config["routed_scaling_factor"], rtol=0, atol=1e-6):
        raise ValueError(f"{store.path}: expert_weights_scale differs from routed_scaling_factor")
    if any(name.startswith(f"blk.{layers}.") for name in store.tensors):
        raise ValueError(f"{store.path}: an MTP layer in the GGUF is not imported")


def _q8_blocks(store: GgufStore, name: str, begin: int, end: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Rows [begin, end) of a Q8_0 tensor: int8 codes [rows, k] and binary64 scales [rows, k / 32]."""

    from tools.artifact.codecs.ggml import decode_q8_0

    codes, scales = decode_q8_0(np.ascontiguousarray(store.raw(name, begin, end)), end - begin, k)
    return codes, scales.view(np.float16).astype(np.float64)


def require_q8_rounding(store: GgufStore, name: str, base: SafetensorsSource, source_name: str,
                        shape: tuple[int, int]) -> None:
    """The GGUF's Q8_0 ``name`` must be the Q8_0 rounding of the checkpoint's ``source_name``: every
    stored value d * code within half a step of the checkpoint's value, plus what the scale's binary16
    rounding moves it (llama.cpp rounds the codes against the unrounded scale: |code| * d * 2^-11,
    bounded here with a 2x margin)."""

    t = store.tensor(name)
    if t.shape != shape or t.ggml_type != 8:
        raise ValueError(f"{name}: expected a Q8_0 tensor of shape {shape}, found type {t.ggml_type} {t.shape}")
    rows, k = shape
    reference = base.describe(source_name)
    if tuple(reference.shape) != shape:
        raise ValueError(f"{source_name}: checkpoint shape {tuple(reference.shape)} differs from {shape}")
    for first in range(0, rows, _ROW_CHUNK):
        last = min(rows, first + _ROW_CHUNK)
        codes, scales = _q8_blocks(store, name, first, last, k)
        stored = codes.reshape(last - first, k // 32, 32) * scales[..., None]
        ref = base.read_flat(source_name, first * k, last * k).to(torch.float64).numpy().reshape(last - first, k // 32, 32)
        bound = scales[..., None] * (0.5 * (1 + 2.0**-10) + np.abs(codes.reshape(last - first, k // 32, 32)) * 2.0**-10)
        bad = np.abs(stored - ref) > bound
        if bad.any():
            row = first + int(np.flatnonzero(bad.any(axis=(1, 2)))[0])
            raise ValueError(f"{name}: row {row} is not the Q8_0 rounding of the checkpoint's {source_name}")


def require_a_log(store: GgufStore, name: str, base: SafetensorsSource, source_name: str, heads: int) -> None:
    """llama.cpp stores A = -exp(A_log) in binary32: -exp of the checkpoint's A_log (binary64) must be
    within one binary32 ulp (2^-23 relative, bounded at 2^-22) of it."""

    a = store.plain(name).reshape(-1)
    a_log = base.read_flat(source_name).to(torch.float64).numpy().reshape(-1)
    if a.size != heads or a_log.size != heads or (np.abs(-np.exp(a_log) - a) > np.abs(a) * 2.0**-22).any():
        raise ValueError(f"{name}: the GGUF's A is not -exp of the checkpoint's {source_name}")


def q8_bf16_rows(store: GgufStore, name: str, shape: tuple[int, int]) -> LogicalSource:
    """A Q8_0 matrix's values d * code, each rounded once to BF16 (the Q8 gather's output)."""

    t = store.tensor(name)
    if t.shape != shape or t.ggml_type != 8:
        raise ValueError(f"{name}: expected a Q8_0 tensor of shape {shape}")
    k = shape[1]

    def read(begin: int, end: int) -> torch.Tensor:
        first, last = begin // k, -(-end // k)
        codes, scales = _q8_blocks(store, name, first, last, k)
        v = torch.from_numpy((codes.reshape(last - first, k // 32, 32) * scales[..., None]).reshape(-1))
        start = begin - first * k
        return v[start : start + end - begin].to(torch.bfloat16).to(torch.float64)

    return LogicalSource(tuple(shape), f"{store.path}:{name} (BF16 rows)", read)


def kda_convolution(store: GgufStore, blk: str, taps: int, width: int) -> LogicalSource:
    """``ssm_conv1d_{q,k,v}`` (each [width, 1, taps], BF16 values in F32) as one (taps, 3 * width)
    kernel in the fused projection's q | k | v channel order (glm5_next._Builder.kda_convolution)."""

    parts = [plain(store, f"{blk}ssm_conv1d_{f}.weight", (width, 1, taps)) for f in "qkv"]

    def read(begin: int, end: int) -> torch.Tensor:
        kernels = [part.values().reshape(width, taps).T for part in parts]
        return torch.cat(kernels, dim=1).contiguous().reshape(-1)[begin:end]

    return LogicalSource((taps, 3 * width), f"{store.path}:{blk}ssm_conv1d_qkv", read)


class _Builder:
    def __init__(self, model: Model, store: GgufStore, base: SafetensorsSource, config: dict):
        self.model = model
        self.store = store
        self.base = base
        self.config = config
        self.checkpoint = glm5_next._Builder(model)

    def add(self, name, source, *, inputs=(), direct="bf16"):
        self.model.add(Parameter(name, tuple(source.shape), source, None, tuple(inputs), direct,
                                 residency=name.split("/", 1)[0]))

    def matrix(self, name, tensor, shape, *, inputs=()):
        self.add(name, ggml_or_plain(self.store, tensor, tuple(shape)), inputs=inputs)

    def group(self, *names):
        self.model.packing_groups.append(tuple(names))

    def hyper_connection(self, prefix, gguf_prefix):
        c = self.config
        width = c["hc_mult"] * c["hidden_size"]
        rows = (2 + c["hc_mult"]) * c["hc_mult"]
        self.add(prefix + "fn", plain(self.store, gguf_prefix + "fn.weight", (rows, width)), inputs=(prefix + "input",))
        self.add(prefix + "base", plain(self.store, gguf_prefix + "base.weight", (rows,), bf16_exact=False), direct="fp32")
        self.add(prefix + "scale", plain(self.store, gguf_prefix + "scale.weight", (3,), bf16_exact=False), direct="fp32")

    def kda(self, prefix, blk, layer):
        c = self.config
        h, heads, d = c["hidden_size"], c["linear_num_heads"], c["linear_head_dim"]
        width, taps = heads * d, c["linear_conv_kernel_dim"]
        p, mixer_input = prefix + "kda/", prefix + "mixer_input"
        for role, field in (("query", "q"), ("key", "k"), ("value", "v")):
            self.matrix(p + role, blk + f"attn_{field}.weight", (width, h), inputs=(mixer_input,))
        self.add(p + "convolution", kda_convolution(self.store, blk, taps, width))
        self.matrix(p + "forget_down", blk + "ssm_f_a.weight", (d, h), inputs=(mixer_input,))
        self.matrix(p + "forget_up", blk + "ssm_f_b.weight", (width, d), inputs=(p + "forget_rank",))
        self.matrix(p + "gate_down", blk + "ssm_g_a.weight", (d, h), inputs=(mixer_input,))
        self.matrix(p + "gate_up", blk + "ssm_g_b.weight", (width, d), inputs=(p + "gate_rank",))
        self.matrix(p + "beta", blk + "ssm_beta.weight", (heads, h), inputs=(mixer_input,))
        source = f"model.language_model.layers.{layer}.self_attn.A_log"
        require_a_log(self.store, blk + "ssm_a", self.base, source, heads)
        self.checkpoint.add(p + "a_log", self.base, source, (heads,), direct="fp32")
        self.add(p + "dt_bias", plain(self.store, blk + "ssm_dt.bias", (width,), bf16_exact=False), direct="fp32")
        self.add(p + "output_norm", plain(self.store, blk + "ssm_norm.weight", (d,)))
        self.matrix(p + "output", blk + "attn_output.weight", (h, width), inputs=(p + "gated_output",))
        self.group(p + "query", p + "key", p + "value")
        self.group(p + "forget_down", p + "gate_down", p + "beta")

    def sparse_attention(self, prefix, blk, layer):
        c = self.config
        h, heads = c["hidden_size"], c["num_attention_heads"]
        dn, dv = c["qk_nope_head_dim"], c["v_head_dim"]
        q_rank, kv_rank = c["q_lora_rank"], c["kv_lora_rank"]
        ih, idim = c["index_n_heads"], c["index_head_dim"]
        p, mixer_input = prefix + "attention/", prefix + "mixer_input"
        # BF16 from the checkpoint (it packs with the GGUF's BF16 latent), checked against the GGUF.
        source = f"model.language_model.layers.{layer}.self_attn.q_a_proj.weight"
        require_q8_rounding(self.store, blk + "attn_q_a.weight", self.base, source, (q_rank, h))
        self.checkpoint.add(p + "query_down", self.base, source, (q_rank, h), inputs=(mixer_input,))
        self.add(p + "query_norm", plain(self.store, blk + "attn_q_a_norm.weight", (q_rank,)))
        self.matrix(p + "query_up", blk + "attn_q_b.weight", (heads * dn, q_rank), inputs=(p + "query_rank",))
        self.matrix(p + "latent", blk + "attn_kv_a_mqa.weight", (kv_rank, h), inputs=(mixer_input,))
        self.add(p + "latent_norm", plain(self.store, blk + "attn_kv_a_norm.weight", (kv_rank,)))
        # llama.cpp's k_b and v_b are the absorbed blocks: (heads, kv_rank, dn) and (heads, dv, kv_rank).
        self.add(p + "key_up", plain(self.store, blk + "attn_k_b.weight", (heads, kv_rank, dn)))
        self.add(p + "value_up", plain(self.store, blk + "attn_v_b.weight", (heads, dv, kv_rank)))
        self.matrix(p + "output", blk + "attn_output.weight", (h, heads * dv), inputs=(p + "attention_output",))
        self.matrix(p + "index_query", blk + "indexer.attn_q_b.weight", (ih * idim, q_rank), inputs=(p + "query_rank",))
        self.matrix(p + "index_key", blk + "indexer.attn_k.weight", (idim, h), inputs=(mixer_input,))
        self.add(p + "index_key_norm", plain(self.store, blk + "indexer.k_norm.weight", (idim,)))
        self.add(p + "index_key_bias", plain(self.store, blk + "indexer.k_norm.bias", (idim,)))
        self.matrix(p + "index_gate", blk + "indexer_compressor_gate.weight", (idim, h), inputs=(mixer_input,))
        self.add(p + "index_position",
                 plain(self.store, blk + "indexer_compressor_ape.weight", (c["index_kpool"], idim), bf16_exact=False),
                 direct="fp32")
        self.matrix(p + "index_weights", blk + "indexer.proj.weight", (ih, h), inputs=(mixer_input,))
        self.group(p + "query_down", p + "latent")
        self.group(p + "index_key", p + "index_gate", p + "index_weights")

    def mlp(self, prefix, blk, suffix, intermediate, inputs):
        h = self.config["hidden_size"]
        for role in ("gate", "up"):
            self.matrix(prefix + role, blk + f"ffn_{role}{suffix}.weight", (intermediate, h), inputs=(inputs,))
        self.matrix(prefix + "down", blk + f"ffn_down{suffix}.weight", (h, intermediate), inputs=(prefix + "product",))
        self.group(prefix + "gate", prefix + "up")

    def moe(self, prefix, blk, layer):
        c = self.config
        h, e, ir = c["hidden_size"], c["n_routed_experts"], c["moe_intermediate_size"]
        p, ffn_input = prefix + "moe/", prefix + "ffn_input"
        self.matrix(p + "router", blk + "ffn_gate_inp.weight", (e, h), inputs=(ffn_input,))
        self.add(p + "router_bias", plain(self.store, blk + "exp_probs_b.bias", (e,), bf16_exact=False), direct="fp32")
        self.mlp(p + "shared/", blk, "_shexp", ir, ffn_input)
        bank = GgmlExpertBank(self.store, layer, (e, h, ir))
        self.model.add(Parameter(p + "experts", (e, h, ir), bank, None, (ffn_input,), bank.gate_up_format,
                                 residency="text"))

    def block(self, index):
        c = self.config
        h = c["hidden_size"]
        prefix, blk = f"text/layers/{index}/", f"blk.{index}."
        self.hyper_connection(prefix + "attn_hc/", blk + "hc_attn_")
        self.hyper_connection(prefix + "ffn_hc/", blk + "hc_ffn_")
        self.add(prefix + "input_norm", plain(self.store, blk + "attn_norm.weight", (h,)))
        self.add(prefix + "post_attention_norm", plain(self.store, blk + "ffn_norm.weight", (h,)))
        if c["layer_types"][index] == "linear_attention":
            self.kda(prefix, blk, index)
        else:
            self.sparse_attention(prefix, blk, index)
        if index < c["first_k_dense_replace"]:
            self.mlp(prefix + "mlp/", blk, "", c["intermediate_size"], prefix + "ffn_input")
        else:
            self.moe(prefix, blk, index)


def build_model(base: SafetensorsSource, store: GgufStore, *, components: tuple[str, ...] = ("text",),
                resource_overrides: Mapping[str, str | Path] | None = None) -> Model:
    """The Glm5Next logical model of a GGUF (``store``), configured by the checkpoint (``base``),
    whose vision tower the ``vision`` component stores (glm5_next._Builder.vision)."""

    selected = set(components)
    if "text" not in selected or selected - {"text", "vision"}:
        raise ValueError("a Glm5Next GGUF import converts the text component and optional vision")
    want_vision = "vision" in selected
    if want_vision and not base.has("model.visual.patch_embed.proj.weight"):
        raise ValueError("vision requested, but the checkpoint has no model.visual tensors")
    config = glm5_next.text_config(base.config)
    _metadata_checks(store, config)
    config["mtp_layers"] = 0  # the GGUF holds no MTP layer
    vision = glm5_next.vision_config(base.config, config) if want_vision else None
    refs, resources, count, special = load_resources(
        base.root, vocab_size=config["vocab_size"], vision_config=vision,
        vision_resources=("processor_config.json",), vision_check=glm5_next._check_processor,
        overrides=resource_overrides)
    records = {"text": {"config": config, "resources": refs["text"]}}
    if want_vision:
        records["vision"] = {"config": vision, "resources": refs["vision"], "target": "text"}
    model = Model(records, resources=resources, token_count=count, special_token_ids=special)
    b = _Builder(model, store, base, config)
    h, v = config["hidden_size"], config["vocab_size"]
    b.add("text/token_embedding", q8_bf16_rows(store, "token_embd.weight", (v, h)))
    require_q8_rounding(store, "output.weight", base, "lm_head.weight", (v, h))
    b.checkpoint.add("text/output_head", base, "lm_head.weight", (v, h), inputs=("text/final_hidden",))
    b.add("text/final_norm", plain(store, "output_norm.weight", (h,)))
    for index in range(config["num_hidden_layers"]):
        b.block(index)
    if want_vision:
        b.checkpoint.vision(base, vision)
    return model
