"""Independent FP64 reference of the GLM-5.3-Flash vision tower (Glm5NextVisionModel).

    python -m tools.glm5_next.vision_reference --model PATH --images IMG[,IMG...] --out DIR

The tower-level oracle for the engine's ``glm5_next`` vision encode (AGENTS.md "Verification").
It is written from the upstream definitions (Transformers ``modeling_glm5_next.py``:
``Glm5NextVisionModel``, ``Glm5NextVisionBlock``, ``Glm5NextVisionAttention``,
``Glm5NextVisionPatchMerger``, ``Glm5NextVisionPatchEmbed``) and reads the checkpoint directly,
never the engine's artifact or converter. Every BF16 tensor is widened exactly, every product is
formed in binary64, and the tower's pixels come from the Hugging Face processor
(``Glm5NextImageProcessor``), which is the pixel preprocessing oracle the engine's frontend is
tested against.

By default the reference rounds what a deployed BF16 runtime stores in BF16 -- the patch input,
every projection output (bias included), every norm's result, the attention output before its
residual add, the SwiGLU product, the rope output, the downsample and merger stages, and the
residual stream after every block. ``--no-bf16-boundaries`` keeps every value in binary64 (the
oracle; Transformers' own tower in float64 agrees with it to 1e-5). ``--hf-check`` also runs
Transformers' BF16 ``Glm5NextVisionModel`` on the same pixels and reports its relative L2 error
against this run: the error a BF16 implementation has. The tower amplifies rounding in its middle
blocks (a BF16 run's residual drifts about 60 % from the oracle's there and recovers to about
30 %), so end to end a BF16 run lands 7-25 % from the oracle; ``--one-step`` (with
``--no-bf16-boundaries``) writes every block's and the merger's oracle from the oracle's own
BF16-rounded states, which a BF16 implementation is compared with stage by stage (one_step).

Outputs (``--out DIR``): ``pixels/Image-K.f32`` + ``pixels/Image-K.json`` (the processor's
resized, normalized, patchified pixels and grid -- the golden inputs), ``embeddings.f32`` (the
merged tokens of every image in order, [sum V, out_hidden_size]), ``hidden.f32`` (the
post-downsample states before the merger), ``blocks/`` (``--one-step``) and ``vision.json``
(grids, counts, diagnostics).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from tools.convert.sources.safetensors import SafetensorsSource

F64 = torch.float64
F32 = torch.float32
BF16 = torch.bfloat16
DEVICE = torch.device("cpu")
PREFIX = "model.visual."


def bf16(x: torch.Tensor) -> torch.Tensor:
    """Round to BF16 (nearest even) and widen back to binary64."""
    return x.to(F32).to(BF16).to(F64)


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float, round=bf16) -> torch.Tensor:
    """Glm5NextRMSNorm: the variance over the last axis, the weight after the value is stored."""
    variance = (x * x).mean(-1, keepdim=True)
    normalized = round(x * torch.rsqrt(variance + eps))
    return round(normalized * weight)


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class Tower:
    """The 24-block tower in binary64 over widened BF16 tensors."""

    def __init__(self, root: Path, *, bf16_boundaries: bool = True):
        self.store = SafetensorsSource(root)
        self.round = bf16 if bf16_boundaries else (lambda x: x)
        top = self.store.config
        self.config = top["vision_config"]
        c = self.config
        self.eps = float(c["rms_norm_eps"])
        self.limit = float(c["swiglu_limit"])
        self.H = int(c["hidden_size"])
        self.heads = int(c["num_heads"])
        self.D = self.H // self.heads
        self.I = int(c["intermediate_size"])
        self.O = int(c["out_hidden_size"])
        self.PI = int(c["projection_intermediate_size"])
        self.merge = int(c["spatial_merge_size"])
        self.depth = int(c["depth"])
        rope = c.get("rope_parameters") or {}
        self.rope_theta = float(rope.get("rope_theta", 10_000.0))
        if rope.get("rope_type", "axial") != "axial":
            raise ValueError(f"unsupported vision rope type {rope.get('rope_type')}")
        if self.D % 4 != 0:
            raise ValueError("the axial rope needs a head dimension divisible by four")
        self.freqs = 1.0 / (
            self.rope_theta ** (torch.arange(0, self.D // 2, 2, dtype=F64) / (self.D // 2))
        )  # spatial_dim = D // 2, one frequency per two positions of a half rotation

    def get(self, name: str) -> torch.Tensor:
        full = PREFIX + name
        info = self.store.describe(full)
        return self.store.read_flat(full).reshape(info.shape).to(DEVICE, F64)

    # -- helpers
    def gemm(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        """x [T, K] against a stored [N, K] weight (nn.Linear convention); rounds the result."""
        y = x @ weight.T
        if bias is not None:
            y = y + bias
        return self.round(y)

    def patch_embed(self, pixels: torch.Tensor, grid: tuple[int, int, int]) -> torch.Tensor:
        """pixels [P, C*t*p*p] in the conv kernel's (c, t, py, px) order -> x [P, H]."""
        t, h, w = grid
        if t != 1:
            raise ValueError("the reference takes single-frame grids; flatten video frames upstream")
        if h * w != pixels.shape[0]:
            raise ValueError("the pixels do not match their grid")
        weight = self.get("patch_embed.proj.weight").reshape(self.H, -1)
        bias = self.get("patch_embed.proj.bias")
        return self.gemm(self.round(pixels), weight, bias)

    def rope(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Axial 2-D rope tables: position_ids [P, 2] (h, w) -> cos, sin [P, D]."""
        freqs = position_ids.unsqueeze(-1) * self.freqs  # [P, 2, D/4]
        cos = freqs.cos()
        sin = freqs.sin()

        def recompose(freq: torch.Tensor) -> torch.Tensor:
            freq_h, freq_w = freq[:, 0], freq[:, 1]
            freq_hw = torch.cat([freq_h, freq_w], dim=-1)
            return torch.cat([freq_hw, freq_hw], dim=-1)

        return recompose(cos), recompose(sin)

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

    def attention(self, index: int, x: torch.Tensor, position_ids: torch.Tensor,
                  cu_seqlens: list[int]) -> torch.Tensor:
        """One block's attention over the packed patches; x is the normed input [P, H]."""
        qkv_weight = self.get(f"blocks.{index}.attn.qkv.weight")
        qkv = self.gemm(x, qkv_weight, self.get(f"blocks.{index}.attn.qkv.bias"))
        q, k, v = qkv.split(self.H, dim=-1)
        q = q.view(-1, self.heads, self.D)
        k = k.view(-1, self.heads, self.D)
        v = v.view(-1, self.heads, self.D)
        q = rmsnorm(q, self.get(f"blocks.{index}.attn.q_norm.weight"), self.eps, self.round)
        k = rmsnorm(k, self.get(f"blocks.{index}.attn.k_norm.weight"), self.eps, self.round)
        cos, sin = self.rope(position_ids)
        # apply_rotary_pos_emb_vision runs in fp32 and stores back to the input dtype; both are
        # below the BF16 boundary the reference rounds at.
        q = self.round(q * cos.unsqueeze(1) + self.rotate_half(q) * sin.unsqueeze(1))
        k = self.round(k * cos.unsqueeze(1) + self.rotate_half(k) * sin.unsqueeze(1))
        attended = torch.empty_like(q)
        scaling = self.D ** -0.5
        for begin, end in zip(cu_seqlens[:-1], cu_seqlens[1:]):
            qs, ks, vs = q[begin:end], k[begin:end], v[begin:end]
            scores = torch.einsum("qhd,khd->hqk", qs, ks) * scaling
            weights = torch.softmax(scores, dim=-1)
            attended[begin:end] = torch.einsum("hqk,khd->qhd", weights, vs)
        projected = self.gemm(attended.reshape(-1, self.H),
                              self.get(f"blocks.{index}.attn.proj.weight"),
                              self.get(f"blocks.{index}.attn.proj.bias"))
        return projected

    def block(self, index: int, x: torch.Tensor, position_ids: torch.Tensor,
              cu_seqlens: list[int]) -> torch.Tensor:
        normed = rmsnorm(x, self.get(f"blocks.{index}.norm1.weight"), self.eps, self.round)
        attended = self.attention(index, normed, position_ids, cu_seqlens)
        x = self.round(x + attended)
        normed = rmsnorm(x, self.get(f"blocks.{index}.norm2.weight"), self.eps, self.round)
        gate = self.gemm(normed, self.get(f"blocks.{index}.mlp.gate_proj.weight"),
                         self.get(f"blocks.{index}.mlp.gate_proj.bias")).clamp(max=self.limit)
        up = self.gemm(normed, self.get(f"blocks.{index}.mlp.up_proj.weight"),
                       self.get(f"blocks.{index}.mlp.up_proj.bias")).clamp(-self.limit, self.limit)
        product = self.round(silu(gate) * up)
        down = self.gemm(product, self.get(f"blocks.{index}.mlp.down_proj.weight"),
                         self.get(f"blocks.{index}.mlp.down_proj.bias"))
        return self.round(x + down)

    def downsample(self, x: torch.Tensor) -> torch.Tensor:
        """The merge: view [V, m, m, H], permute to [V, H, m, m], conv2d with stride m."""
        x = x.view(-1, self.merge, self.merge, self.H).permute(0, 3, 1, 2).reshape(-1, self.H * self.merge * self.merge)
        weight = self.get("downsample.weight").reshape(self.O, -1)
        return self.gemm(x, weight, self.get("downsample.bias"))

    def merger(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.gemm(x, self.get("merger.proj.weight"))
        # nn.LayerNorm(dim): eps 1e-5, affine weight and bias.
        mean = projected.mean(-1, keepdim=True)
        variance = ((projected - mean) ** 2).mean(-1, keepdim=True)
        normalized = self.round((projected - mean) * torch.rsqrt(variance + 1e-5))
        affine = self.round(normalized * self.get("merger.post_projection_norm.weight")
                      + self.get("merger.post_projection_norm.bias"))
        activated = self.round(torch.nn.functional.gelu(affine))  # exact erf GELU
        gate = self.gemm(activated, self.get("merger.gate_proj.weight")).clamp(max=self.limit)
        up = self.gemm(activated, self.get("merger.up_proj.weight")).clamp(-self.limit, self.limit)
        product = self.round(silu(gate) * up)
        return self.gemm(product, self.get("merger.down_proj.weight"))

    def positions(self, grid: tuple[int, int, int]) -> torch.Tensor:
        """The (h, w) coordinates [P, 2] of a single-frame grid in merge-block order."""
        _, h, w = grid
        position_ids = torch.stack(torch.meshgrid(
            torch.arange(h, dtype=torch.long), torch.arange(w, dtype=torch.long), indexing="ij"), dim=-1)
        block_shape = (h // self.merge, self.merge, w // self.merge, self.merge)
        hpos = position_ids[..., 0].reshape(block_shape).transpose(1, 2).flatten()
        wpos = position_ids[..., 1].reshape(block_shape).transpose(1, 2).flatten()
        return torch.stack([hpos, wpos], dim=-1)

    def tail(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """The post norm, downsample and merger of a final residual [P, H]."""
        x = rmsnorm(x, self.get("post_layernorm.weight"), self.eps, self.round)
        hidden = self.downsample(x)
        return {"hidden": hidden, "merged": self.merger(hidden)}

    def forward(self, pixels: torch.Tensor, grid: tuple[int, int, int],
                states: list | None = None) -> dict[str, torch.Tensor]:
        """One item's tower forward: returns the pre-merger hidden [V, O] and merged [V, O].
        ``states`` collects the residual entering every block and the merger."""
        _, h, w = grid
        x = self.patch_embed(pixels, grid)
        position_ids = self.positions(grid)
        rms = [float(x.float().pow(2).mean().sqrt())]
        for index in range(self.depth):
            if states is not None:
                states.append(x)
            x = self.block(index, x, position_ids, [0, h * w])
            rms.append(float(x.float().pow(2).mean().sqrt()))
        if states is not None:
            states.append(x)
        return {**self.tail(x), "block_rms": rms}


def hf_reference(model, pixels: torch.Tensor, grid: tuple[int, int, int]) -> torch.Tensor:
    """Transformers' own BF16 vision model on the same pixels (the architecture cross-check)."""
    with torch.no_grad():
        out = model(hidden_states=pixels.to(BF16), grid_thw=torch.tensor([grid]))
    return out.pooler_output.to(F64)


def relative(got: torch.Tensor, want: torch.Tensor, base: torch.Tensor | None = None) -> float:
    """||got - want|| / ||want - base|| (base None: the plain relative L2 error)."""
    scale = want if base is None else want - base
    return float((got - want).pow(2).sum().sqrt() / scale.pow(2).sum().sqrt())


def one_step(tower: Tower, pixels: torch.Tensor, grid: tuple[int, int, int], out: Path, index: int,
             hf_model=None) -> dict:
    """Every block and the merger applied once, in binary64, to the BF16-rounded residual the
    oracle's own trajectory holds there: the per-stage oracle a BF16 implementation is compared
    with step by step, since the tower amplifies a difference made in one block through the next.
    Writes blocks/Image-K.in.bf16 (BF16 bits [depth + 1, P, H]: every block's input, then the
    merger's), .out.f32 ([depth, P, H]: each block's output) and .merged.f32 ([V, O]: the merger's).
    With ``hf_model`` (Transformers' BF16 tower) also returns its error on the same inputs: each
    block's relative to the block's update, the merger's relative to its output."""
    states: list[torch.Tensor] = []
    tower.forward(pixels, grid, states)
    inputs = [bf16(x) for x in states]
    positions = tower.positions(grid)
    _, h, w = grid
    outputs = [tower.block(i, inputs[i], positions, [0, h * w]) for i in range(tower.depth)]
    merged = tower.tail(inputs[-1])["merged"]
    (out / "blocks").mkdir(exist_ok=True)
    bits = (torch.stack(inputs).to(F32).contiguous().view(torch.int32) >> 16).to(torch.int16)
    (out / "blocks" / f"Image-{index}.in.bf16").write_bytes(bits.numpy().tobytes())
    (out / "blocks" / f"Image-{index}.out.f32").write_bytes(torch.stack(outputs).to(F32).numpy().tobytes())
    (out / "blocks" / f"Image-{index}.merged.f32").write_bytes(merged.to(F32).numpy().tobytes())
    if hf_model is None:
        return {}
    cos, sin = hf_model.rotary_pos_emb(torch.zeros(1), positions)
    cu = torch.tensor([0, h * w], dtype=torch.int32)
    blocks = []
    with torch.no_grad():
        for i in range(tower.depth):
            got = hf_model.blocks[i](inputs[i].to(BF16), cu_seqlens=cu,
                                     position_embeddings=(cos.to(BF16), sin.to(BF16))).to(F64)
            blocks.append(relative(got, outputs[i], inputs[i]))
        x = hf_model.post_layernorm(inputs[-1].to(BF16))
        x = x.view(-1, tower.merge, tower.merge, tower.H).permute(0, 3, 1, 2)
        got = hf_model.merger(hf_model.downsample(x).view(-1, tower.O)).to(F64)
    return {"hf_one_step": blocks, "hf_merger": relative(got, merged)}


def hf_model_of(tower: Tower):
    """Transformers' BF16 Glm5NextVisionModel with the checkpoint's tower weights."""
    from transformers import AutoConfig, Glm5NextVisionModel

    config = AutoConfig.from_pretrained(tower.store.root).vision_config
    model = Glm5NextVisionModel._from_config(config).to(BF16).eval()
    state = {name[len(PREFIX):]: tensor for name, tensor in _checkpoint_tensors(tower).items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    real_missing = [m for m in missing if "inv_freq" not in m and "original_inv_freq" not in m]
    if real_missing or unexpected:
        raise RuntimeError(f"state mismatch: missing {real_missing[:4]}, unexpected {unexpected[:4]}")
    return model


def _checkpoint_tensors(tower: Tower) -> dict[str, torch.Tensor]:
    """The checkpoint's ``model.visual.*`` tensors, widened from their stored dtype."""
    out = {}
    for name in tower.store.weight_map:
        if name.startswith(PREFIX):
            info = tower.store.describe(name)
            out[name] = tower.store.read_flat(name).reshape(info.shape)
    return out


def preprocess(tower: Tower, images: list[Path]) -> list[dict]:
    """The Hugging Face processor's pixels: the preprocessing oracle and the golden inputs."""
    from transformers import AutoImageProcessor

    processor = AutoImageProcessor.from_pretrained(tower.store.root)
    from PIL import Image

    loaded = [Image.open(path).convert("RGB") for path in images]
    batch = processor(images=loaded, return_tensors="pt")
    pixel_values = batch["pixel_values"]
    grids = batch["image_grid_thw"].tolist()
    items = []
    offset = 0
    for path, grid in zip(images, grids):
        patches = int(grid[1] * grid[2] * grid[0])
        items.append({
            "path": str(path),
            "grid": grid,
            "merged_tokens": grid[0] * grid[1] * grid[2] // (tower.merge * tower.merge),
            "patch_begin": offset,
            "patch_count": patches,
        })
        offset += patches
    if offset != pixel_values.shape[0]:
        raise RuntimeError("processor grids disagree with the pixel batch")
    return items, pixel_values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--images", required=True, help="comma-separated image paths, or @FILE")
    parser.add_argument("--out", type=Path, required=True, help="directory for the raw outputs")
    parser.add_argument("--no-bf16-boundaries", action="store_true")
    parser.add_argument("--hf-check", action="store_true",
                        help="also run Transformers' BF16 tower and report the difference")
    parser.add_argument("--one-step", action="store_true",
                        help="also write every block's and the merger's one-step oracle "
                             "(needs --no-bf16-boundaries)")
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args()
    if args.one_step and not args.no_bf16_boundaries:
        raise SystemExit("--one-step is the binary64 oracle: add --no-bf16-boundaries")
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.images.startswith("@"):
        images = [Path(p) for p in Path(args.images[1:]).read_text().split() if p]
    else:
        images = [Path(p) for p in args.images.split(",") if p]
    if not images:
        raise SystemExit("no images given")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "pixels").mkdir(exist_ok=True)

    tower = Tower(args.model, bf16_boundaries=not args.no_bf16_boundaries)
    items, pixel_values = preprocess(tower, images)

    embeddings, hiddens, report = [], [], []
    hf_model = hf_model_of(tower) if args.hf_check else None
    for index, item in enumerate(items):
        grid = tuple(item["grid"])
        count = item["patch_count"]
        pixels = pixel_values[item["patch_begin"]:item["patch_begin"] + count].to(F64)
        (args.out / "pixels" / f"Image-{index}.f32").write_bytes(
            pixels.to(F32).numpy().tobytes())
        (args.out / "pixels" / f"Image-{index}.json").write_text(json.dumps(
            {"path": item["path"], "grid": item["grid"], "patch_count": count}))
        record = tower.forward(pixels, grid)
        entry = {"image": index}
        if args.hf_check:
            hf = hf_reference(hf_model, pixel_values[item["patch_begin"]:item["patch_begin"] + count], grid)
            entry["hf_relative_l2"] = relative(hf, record["merged"])
        if args.one_step:
            entry.update(one_step(tower, pixels, grid, args.out, index, hf_model))
        if len(entry) > 1:
            report.append(entry)
        embeddings.append(record["merged"])
        hiddens.append(record["hidden"])
        item["block_rms"] = record["block_rms"]

    stacked = torch.cat(embeddings, dim=0)
    stacked.to(F32).numpy().tofile(args.out / "embeddings.f32")
    torch.cat(hiddens, dim=0).to(F32).numpy().tofile(args.out / "hidden.f32")
    summary = {
        "model": str(args.model),
        "bf16_boundaries": not args.no_bf16_boundaries,
        "images": items,
        "out_hidden_size": tower.O,
        "embedding_rows": int(stacked.shape[0]),
        "hf_check": report,
    }
    (args.out / "vision.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({"embedding_rows": int(stacked.shape[0]),
                      "hf_check": [{k: v for k, v in r.items() if k != "hf_one_step"} for r in report]},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
