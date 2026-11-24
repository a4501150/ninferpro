"""Per-block check of the engine's Glm5Next forward against the FP64 reference.

    python -m tools.glm5_next.block_check --model PATH --reference DIR --engine DIR [--blocks 1-44]

The whole forward amplifies BF16-sized differences (tools/glm5_next/reference.py), so its
comparison with the reference cannot separate a block's own error from what it inherits. This
check can: each block runs in the reference from the engine's own input (the previous block's
residual the engine wrote, ``INFERNIX_GLM_DUMP`` of tests/models/glm5_next/test_forward_real.cpp),
and the engine's output is compared with that. A correct block stays at BF16 resolution (a few
1e-3); a faulty one stands out at once. ``--reference`` is a reference run of the same prompt
(``reference.py --out DIR``); its residuals give the second column, the engine's whole-forward
error at that block.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from tools.glm5_next.reference import Reference, rmsnorm


def _blocks(text: str, count: int) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        first, _, last = part.partition("-")
        out.extend(range(int(first), int(last or first) + 1))
    if any(b < 1 or b >= count for b in out):
        raise ValueError(f"blocks are 1..{count - 1} (block 0 starts from the embedding)")
    return out


def _rel(a, b) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True, help="reference.py --out DIR of the prompt")
    parser.add_argument("--engine", type=Path, required=True, help="the forward test's INFERNIX_GLM_DUMP DIR")
    parser.add_argument("--blocks", default="", help="e.g. 1-44 or 9,10 (default: every block)")
    args = parser.parse_args()

    ref = Reference(args.model)
    c, w = ref.c, ref.w
    layers, width = int(c["num_hidden_layers"]), ref.hc * ref.H
    T = len((args.reference / "tokens.txt").read_text().split())
    dense = sum(kind != "sparse" for kind in ref.mlp_types)
    engine = np.fromfile(args.engine / "residuals.f32", np.float32).reshape(layers, T, width)
    expected = np.fromfile(args.reference / "residuals.f32", np.float32).reshape(layers, T, width)
    routes = np.fromfile(args.engine / "routed.i32", np.int32).reshape(layers - dense, T, -1)

    def block(layer: int, R: torch.Tensor):
        lp = f"layers.{layer}."
        post, comb, x = ref.hc_mix(lp + "hc_attn", R)
        x = ref.round(rmsnorm(x, w.text(lp + "input_layernorm.weight"), ref.eps))
        y = ref.kda(layer, x) if ref.layer_types[layer] == "linear_attention" else ref.mla(layer, x)
        R = ref.hc_update(R, y, post, comb)
        post, comb, x = ref.hc_mix(lp + "hc_ffn", R)
        x = ref.round(rmsnorm(x, w.text(lp + "post_attention_layernorm.weight"), ref.eps))
        idx = None
        if ref.mlp_types[layer] == "sparse":
            y, idx = ref.moe(layer, x)
        else:
            y = ref.mlp(lp + "mlp.", x)
        return ref.hc_update(R, y, post, comb), idx

    print("block  own error  whole-forward error  routed sets differing")
    worst = 0.0
    for layer in _blocks(args.blocks, layers) if args.blocks else range(1, layers):
        out, idx = block(layer, torch.from_numpy(engine[layer - 1].astype(np.float64)))
        own = _rel(engine[layer], out)
        worst = max(worst, own)
        differing = 0
        if idx is not None:
            mine = routes[layer - dense]
            differing = sum(set(mine[t].tolist()) != set(idx[t].tolist()) for t in range(T))
        print(f"{layer:5d}  {own:.3e}  {_rel(engine[layer], expected[layer]):.3e}  {differing}")
    print(f"worst block error on the engine's input: {worst:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
