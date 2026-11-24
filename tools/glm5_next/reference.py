"""Independent FP64 reference forward of GLM-5.3-Flash (Glm5Next) on a safetensors checkpoint.

    python -m tools.glm5_next.reference --model PATH --tokens 154822,9707,11 --out DIR

The model-level oracle for the engine's ``glm5_next`` family (AGENTS.md "Verification"). It is
written from the upstream definitions (Transformers ``modeling_glm5_next.py``:
``Glm5NextTextHyperConnection``, ``Glm5NextTextLinearAttention``, ``Glm5NextTextAttention``,
``Glm5NextTextIndexer``, ``Glm5NextTextMoE``) and reads the checkpoint directly, never the engine's
artifact or converter:

- every BF16 and FP32 tensor is widened exactly and every product is formed in binary64;
- routed experts are compressed-tensors NVFP4 (``weight_packed`` E2M1 codes, E4M3 block scales,
  the FP32 ``weight_global_scale`` divisor), decoded exactly and applied to BF16 activations (the
  engine's W4A16 arithmetic); BF16 experts (another checkpoint) are read as stored;
- the router is the ``noaux_tc`` rule: FP32 logits, sigmoid scores, selection on scores plus
  ``e_score_correction_bias`` (exact ties to the lower expert id), weights the selected unbiased
  scores normalized and scaled by ``routed_scaling_factor``.

By default the reference rounds what deployed BF16 runtimes store in BF16 -- the residual streams,
every block input and output, projection outputs, the SwiGLU product, the pooled index keys;
``--no-bf16-boundaries`` keeps every value in binary64. The model amplifies BF16-sized
differences (some MoE blocks multiply a residual difference by ten), so the two runs differ by the
floor any BF16 implementation shares, and the engine's forward test takes the binary64 run as its
oracle and the rounded run as that floor. The indexer selection is evaluated exactly; for prompts
of at most ``index_topk + index_kpool - 1`` tokens every token is selected and the attention
equals causal attention.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from tools.convert.sources.safetensors import SafetensorsSource

F64 = torch.float64
DEVICE = torch.device("cpu")
_E2M1_VALUES = [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]


def bf16(x: torch.Tensor) -> torch.Tensor:
    """Round to BF16 (nearest even) and widen back to binary64."""
    return x.to(torch.float32).to(torch.bfloat16).to(F64)


def rmsnorm(x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    """GLM RMSNorm: x * rsqrt(mean(x^2) + eps) * w (no unit offset; unweighted when w is None)."""
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return y if weight is None else y * weight


def layer_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
    x = x - x.mean(-1, keepdim=True)
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight + bias


def int8_groups(x: torch.Tensor, group: int) -> torch.Tensor:
    """The engine's INT8 KV codec (ops/kv_cache_append.h) applied and decoded: per contiguous
    `group` values a = max|x|, s = FP16(a / 127), code = clamp(RNE(x * FP32(1 / s)), -127, 127),
    decode = code * s; an all-zero group decodes to zero."""
    shape = x.shape
    g = x.reshape(-1, group).to(torch.float32)
    s = (g.abs().amax(-1, keepdim=True) / 127).to(torch.float16).to(torch.float32)
    inv = torch.where(s == 0, torch.zeros_like(s), (1 / s))
    code = torch.clamp(torch.round(g * inv), -127, 127)
    return (code * s).to(F64).reshape(shape)


def hadamard(x: torch.Tensor) -> torch.Tensor:
    """The normalized Walsh-Hadamard transform of the last dimension (a power of two); its own inverse."""
    n = x.shape[-1]
    y = x.reshape(-1, n).clone()
    h = 1
    while h < n:
        y = y.reshape(-1, n // (2 * h), 2, h)
        a, b = y[:, :, 0, :], y[:, :, 1, :]
        y = torch.stack((a + b, a - b), dim=2).reshape(-1, n)
        h *= 2
    return (y / math.sqrt(n)).reshape(x.shape)


LATENT_CODECS = {
    "int8-g64": lambda x: int8_groups(x, 64),
    "int8-g128": lambda x: int8_groups(x, 128),
    "int8-row": lambda x: int8_groups(x, x.shape[-1]),
    "int8-g64-hadamard": lambda x: hadamard(int8_groups(hadamard(x), 64)),
    "int8-row-hadamard": lambda x: hadamard(int8_groups(hadamard(x), x.shape[-1])),
}


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """The FLA rule upstream uses: x / sqrt(sum x^2 + eps)."""
    return x / torch.sqrt((x * x).sum(-1, keepdim=True) + eps)


class Weights:
    def __init__(self, root: Path):
        self.store = SafetensorsSource(root)
        top = self.store.config
        self.config = top.get("text_config", top)
        self.prefix = "model.language_model."

    def get(self, name: str) -> torch.Tensor:
        info = self.store.describe(name)
        return self.store.read_flat(name).reshape(info.shape).to(DEVICE, F64)

    def text(self, name: str) -> torch.Tensor:
        return self.get(self.prefix + name)


class Experts:
    """Decodes each routed expert when it is used (one decoded expert is ~200 MB in binary64)."""

    def __init__(self, weights: Weights):
        self.w = weights

    def _matrix(self, leaf: str, shape: tuple[int, int]) -> torch.Tensor:
        store, (n, k) = self.w.store, shape
        if not store.has(leaf + ".weight_packed"):
            return self.w.get(leaf + ".weight")
        codes = store.read_flat(leaf + ".weight_packed").to(DEVICE).reshape(n, k // 2)
        codes = torch.stack((codes & 0xF, codes >> 4), dim=-1).reshape(n, k).long()
        scales = store.read_flat(leaf + ".weight_scale").to(DEVICE).view(torch.float8_e4m3fn).reshape(n, k // 16)
        divisor = float(store.read_flat(leaf + ".weight_global_scale").reshape(-1)[0])
        return torch.tensor(_E2M1_VALUES, dtype=F64)[codes] * scales.to(F64).repeat_interleave(16, dim=-1) / divisor

    def forward(self, layer: int, expert: int, x: torch.Tensor, limit: float, rnd) -> torch.Tensor:
        """One routed expert on BF16 rows x [n, H]: W4A16 with BF16 at the SwiGLU product and output."""
        h, i = self.w.config["hidden_size"], self.w.config["moe_intermediate_size"]
        leaf = f"{self.w.prefix}layers.{layer}.mlp.experts.{expert}."
        g = x @ self._matrix(leaf + "gate_proj", (i, h)).T
        u = x @ self._matrix(leaf + "up_proj", (i, h)).T
        a = rnd(silu(g.clamp(max=limit)) * u.clamp(-limit, limit))
        return rnd(a @ self._matrix(leaf + "down_proj", (h, i)).T)


class Reference:
    def __init__(self, root: Path, *, bf16_boundaries: bool = True, latent_codec: str | None = None):
        self.w = Weights(root)
        self.latent_codec = LATENT_CODECS[latent_codec] if latent_codec else None
        c = self.c = self.w.config
        self.round = bf16 if bf16_boundaries else (lambda x: x)
        self.H = int(c["hidden_size"])
        self.eps = float(c["rms_norm_eps"])
        self.hc = int(c["hc_mult"])
        self.hc_iters = int(c["hc_sinkhorn_iters"])
        self.hc_eps = float(c["hc_eps"])
        self.limit = float(c["swiglu_limit"])
        self.layer_types = list(c["layer_types"])
        self.mlp_types = list(c["mlp_layer_types"])
        if any(kind != "full" for kind in c["indexer_types"]):
            raise ValueError("shared indexer layers are not in this reference")
        if int(c["qk_rope_head_dim"]) != 0:
            raise ValueError("the reference implements the NoPE attention only")
        self.experts = Experts(self.w)

    # -- mHC
    def hc_mix(self, prefix: str, R: torch.Tensor):
        """R [T, S*H] -> (post [T, S], comb [T, S, S] indexed [in j][out s], x [T, H])."""
        S, T = self.hc, R.shape[0]
        fn = self.w.text(prefix + "_fn")
        base = self.w.text(prefix + "_base")
        scale = self.w.text(prefix + "_scale")
        f = rmsnorm(R, None, self.eps) @ fn.T
        pre = torch.sigmoid(f[:, :S] * scale[0] + base[:S]) + self.hc_eps
        post = 2.0 * torch.sigmoid(f[:, S : 2 * S] * scale[1] + base[S : 2 * S])
        comb = torch.softmax((f[:, 2 * S :] * scale[2] + base[2 * S :]).reshape(T, S, S), dim=-1)
        comb = comb + self.hc_eps
        comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
        for _ in range(self.hc_iters - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + self.hc_eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
        x = (pre[:, :, None] * R.reshape(T, S, self.H)).sum(1)
        return post, comb, self.round(x)

    def hc_update(self, R: torch.Tensor, y: torch.Tensor, post: torch.Tensor, comb: torch.Tensor):
        T, S = R.shape[0], self.hc
        mixed = torch.einsum("tjs,tjd->tsd", comb, R.reshape(T, S, self.H))
        return self.round(post[:, :, None] * y[:, None, :] + mixed).reshape(T, S * self.H)

    # -- KDA
    def kda(self, layer: int, x: torch.Tensor) -> torch.Tensor:
        c, w, r = self.c, self.w, self.round
        p = f"layers.{layer}.self_attn."
        linear = c["linear_attn_config"]
        nh, d = int(linear["num_heads"]), int(linear["head_dim"])
        T = x.shape[0]
        streams = []
        for name in ("q", "k", "v"):
            proj = r(x @ w.text(p + f"{name}_proj.weight").T)
            kernel = w.text(p + f"{name}_conv1d.weight")[:, 0, :]
            taps = kernel.shape[1]
            padded = torch.cat((torch.zeros(taps - 1, proj.shape[1], dtype=F64), proj), dim=0)
            conv = sum(kernel[:, j][None, :] * padded[j : j + T] for j in range(taps))
            streams.append(r(silu(conv)).reshape(T, nh, d))
        q, k, v = streams
        q = l2norm(q) / math.sqrt(d)
        k = l2norm(k)
        fa = r(x @ w.text(p + "f_a_proj.weight").T)
        g = r(fa @ w.text(p + "f_b_proj.weight").T) + w.text(p + "dt_bias")
        a = w.text(p + "A_log").exp()
        log_alpha = float(linear["gate_lower_bound"]) * torch.sigmoid(a[None, :, None] * g.reshape(T, nh, d))
        beta = torch.sigmoid(r(x @ w.text(p + "b_proj.weight").T))
        state = torch.zeros(nh, d, d, dtype=F64)  # [head, key, value]
        out = torch.zeros(T, nh, d, dtype=F64)
        for t in range(T):
            state = state * log_alpha[t].exp()[:, :, None]
            memory = (state * k[t][:, :, None]).sum(1)
            delta = (v[t] - memory) * beta[t][:, None]
            state = state + k[t][:, :, None] * delta[:, None, :]
            out[t] = (state * q[t][:, :, None]).sum(1)
        ga = r(x @ w.text(p + "g_a_proj.weight").T)
        gate = r(ga @ w.text(p + "g_b_proj.weight").T).reshape(T, nh, d)
        o = rmsnorm(out, w.text(p + "o_norm.weight"), self.eps) * torch.sigmoid(gate)
        return r(r(o.reshape(T, nh * d)) @ w.text(p + "o_proj.weight").T)

    # -- sparse MLA
    def index_select(self, layer: int, x: torch.Tensor, c_q: torch.Tensor) -> list[torch.Tensor]:
        """Upstream DSA indexer with k-pool compression: selected token positions per query."""
        c, w = self.c, self.w
        p = f"layers.{layer}.self_attn.indexer."
        nh, d, pool = int(c["index_n_heads"]), int(c["index_head_dim"]), int(c["index_kpool"])
        budget = int(c["index_topk"]) // pool
        T = x.shape[0]
        q = self.round(c_q @ w.text(p + "wq_b.weight").T).reshape(T, nh, d)
        k = self.round(layer_norm(self.round(x @ w.text(p + "wk.weight").T),
                                  w.text(p + "k_norm.weight"), w.text(p + "k_norm.bias"), 1e-6))
        z = self.round(x @ w.text(p + "index_kpool_compress_gate").T)
        ape = w.text(p + "index_kpool_compress_ape")
        weights = self.round(x @ w.text(p + "weights_proj.weight").T) / math.sqrt(nh)
        pools_total = T // pool
        keys = None
        if pools_total:
            zz = z[: pools_total * pool].reshape(pools_total, pool, d) + ape[None]
            prob = torch.softmax(zz, dim=1)
            keys = self.round((prob * k[: pools_total * pool].reshape(pools_total, pool, d)).sum(1))
        out = []
        for t in range(T):
            visible = t + 1
            pools = visible // pool
            chosen: list[int] = []
            if pools:
                scores = (weights[t][:, None] * torch.relu(q[t] @ keys[:pools].T / math.sqrt(d))).sum(0)
                count = min(budget, pools)
                order = sorted(range(pools), key=lambda b: (-float(scores[b]), b))[:count]
                for b in sorted(order):
                    chosen.extend(range(b * pool, b * pool + pool))
            chosen.extend(range(pools * pool, visible))
            out.append(torch.tensor(chosen, dtype=torch.long))
        return out

    def mla(self, layer: int, x: torch.Tensor) -> torch.Tensor:
        c, w, r = self.c, self.w, self.round
        p = f"layers.{layer}.self_attn."
        nh = int(c["num_attention_heads"])
        dn, dv, rank = int(c["qk_nope_head_dim"]), int(c["v_head_dim"]), int(c["kv_lora_rank"])
        T = x.shape[0]
        c_q = r(rmsnorm(r(x @ w.text(p + "q_a_proj.weight").T), w.text(p + "q_a_layernorm.weight"), self.eps))
        q = r(c_q @ w.text(p + "q_b_proj.weight").T).reshape(T, nh, dn)
        latent = r(x @ w.text(p + "kv_a_proj_with_mqa.weight").T)[:, :rank]
        latent = r(rmsnorm(latent, w.text(p + "kv_a_layernorm.weight"), self.eps))
        if self.latent_codec is not None:
            latent = self.latent_codec(latent)  # the cached latent serves as key and value
        kv_b = w.text(p + "kv_b_proj.weight").reshape(nh, dn + dv, rank)
        keys = r(torch.einsum("hdc,nc->nhd", kv_b[:, :dn], latent))
        values = r(torch.einsum("hdc,nc->nhd", kv_b[:, dn:], latent))
        selected = self.index_select(layer, x, c_q)
        out = torch.zeros(T, nh, dv, dtype=F64)
        for t in range(T):
            idx = selected[t]
            scores = torch.einsum("hd,nhd->hn", q[t], keys[idx]) / math.sqrt(dn)
            out[t] = torch.einsum("hn,nhd->hd", torch.softmax(scores, dim=-1), values[idx])
        return r(r(out.reshape(T, nh * dv)) @ w.text(p + "o_proj.weight").T)

    # -- MLPs
    def mlp(self, prefix: str, x: torch.Tensor) -> torch.Tensor:
        w, r = self.w, self.round
        g = r(x @ w.text(prefix + "gate_proj.weight").T)
        u = r(x @ w.text(prefix + "up_proj.weight").T)
        a = r(silu(g.clamp(max=self.limit)) * u.clamp(-self.limit, self.limit))
        return r(a @ w.text(prefix + "down_proj.weight").T)

    def moe(self, layer: int, x: torch.Tensor):
        c, w = self.c, self.w
        p = f"layers.{layer}.mlp."
        logits = x @ w.text(p + "gate.weight").T
        scores = torch.sigmoid(logits)
        choice = scores + w.text(p + "gate.e_score_correction_bias")
        # Exact ties go to the lower expert id (stable descending sort).
        idx = torch.sort(choice, dim=-1, descending=True, stable=True).indices[:, : c["num_experts_per_tok"]]
        top = torch.gather(scores, -1, idx)
        top = top / (top.sum(-1, keepdim=True) + 1e-20) * float(c["routed_scaling_factor"])
        T = x.shape[0]
        routed = torch.zeros(T, self.H, dtype=F64)
        for e in torch.unique(idx).tolist():
            rows, slots = torch.where(idx == e)
            y = self.experts.forward(layer, e, x[rows], self.limit, self.round)
            routed.index_add_(0, rows, y * top[rows, slots][:, None])
        shared = self.mlp(p + "shared_experts.", x)
        return self.round(routed + shared), idx

    # -- full forward
    def forward(self, ids: list[int], *, layers: int | None = None, log=print) -> dict:
        c, w, r = self.c, self.w, self.round
        T = len(ids)
        name = w.prefix + "embed_tokens.weight"
        x = torch.stack([w.store.read_flat(name, i * self.H, (i + 1) * self.H) for i in ids]).to(DEVICE, F64)
        R = x.repeat(1, self.hc)
        count = int(c["num_hidden_layers"]) if layers is None else layers
        record = {"layer_residual_rms": [], "layer_residuals": [], "routed": []}
        start = time.time()
        for layer in range(count):
            lp = f"layers.{layer}."
            post, comb, xa = self.hc_mix(lp + "hc_attn", R)
            xa = r(rmsnorm(xa, w.text(lp + "input_layernorm.weight"), self.eps))
            if self.layer_types[layer] == "linear_attention":
                y = self.kda(layer, xa)
            else:
                y = self.mla(layer, xa)
            R = self.hc_update(R, y, post, comb)
            post, comb, xm = self.hc_mix(lp + "hc_ffn", R)
            xm = r(rmsnorm(xm, w.text(lp + "post_attention_layernorm.weight"), self.eps))
            if self.mlp_types[layer] == "sparse":
                y, idx = self.moe(layer, xm)
                record["routed"].append(idx.cpu().numpy())
            else:
                y = self.mlp(lp + "mlp.", xm)
            R = self.hc_update(R, y, post, comb)
            record["layer_residual_rms"].append(float(R.pow(2).mean().sqrt()))
            record["layer_residuals"].append(R.to(torch.float32).cpu().numpy())
            log(f"layer {layer:2d} {self.layer_types[layer]:25s} rms {record['layer_residual_rms'][-1]:.5f} "
                f"({time.time() - start:.0f}s)")
        if layers is None or layers == int(c["num_hidden_layers"]):
            mean = r(R.reshape(T, self.hc, self.H).mean(1))
            xf = r(rmsnorm(mean, w.text("norm.weight"), self.eps))
            # Row chunks keep the binary64 widening of the [V, H] head to ~0.5 GB at a time.
            head, rows = "lm_head.weight", w.store.describe("lm_head.weight").shape[0]
            chunks = []
            for begin in range(0, rows, 16384):
                end = min(rows, begin + 16384)
                part = w.store.read_flat(head, begin * self.H, end * self.H).to(DEVICE).reshape(end - begin, self.H)
                chunks.append(xf @ part.to(F64).T)
            record["logits"] = torch.cat(chunks, dim=-1).to(torch.float32).cpu().numpy()
        return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokens", required=True, help="comma-separated token ids, or @FILE")
    parser.add_argument("--layers", type=int, help="stop after this many layers (no logits)")
    parser.add_argument("--no-bf16-boundaries", action="store_true")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--device", default="cpu", help="torch device of the evaluation (cpu, cuda)")
    parser.add_argument("--latent-codec", choices=sorted(LATENT_CODECS),
                        help="apply this KV codec to every cached MLA latent (a candidate format's effect)")
    parser.add_argument("--out", type=Path, required=True, help="directory for the raw outputs")
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.tokens.startswith("@"):
        ids = [int(v) for v in Path(args.tokens[1:]).read_text().replace(",", " ").split()]
    else:
        ids = [int(v) for v in args.tokens.split(",") if v]
    global DEVICE
    DEVICE = torch.device(args.device)
    torch.set_default_device(DEVICE)
    ref = Reference(args.model, bf16_boundaries=not args.no_bf16_boundaries, latent_codec=args.latent_codec)
    record = ref.forward(ids, layers=args.layers)
    # The raw files the engine's forward test reads (tests/models/glm5_next/test_forward_real.cpp).
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "tokens.txt").write_text(" ".join(str(i) for i in ids) + "\n")
    np.stack(record["layer_residuals"]).astype(np.float32).tofile(out / "residuals.f32")
    if record["routed"]:
        np.stack(record["routed"]).astype(np.int32).tofile(out / "routed.i32")
    if "logits" in record:
        record["logits"].astype(np.float32).tofile(out / "logits.f32")
        top = np.argsort(-record["logits"], axis=-1)[:, :5]
        print("top-5 next tokens per position:", top.tolist())
    (out / "rms.json").write_text(json.dumps({"layer_residual_rms": record["layer_residual_rms"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
