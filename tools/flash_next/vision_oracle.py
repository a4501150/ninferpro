"""Compare the VT6c multimodal wiring dump with the FP64 reference (design §19.3.2).

    .venv-convert/bin/python tools/flash_next/vision_oracle.py --model SNAPSHOT --dump OUT_DIR

``infernix_qwen4_exp_vision_oracle_real_test OUT_DIR`` writes the prepared prompt's token ids,
M-RoPE positions, Vision patches and patch grid, and the public Engine's first greedy tokens, on a
~2,130-token media prompt (past the 2,051-token QSA threshold, so selection runs over M-RoPE pooled
keys). This script replays the same prompt through the independent FP64 reference
(``tools/flash_next/reference.py``, never Infernix's artifact or converter) and checks:

- the frontend's M-RoPE positions equal the upstream ``get_rope_index`` rule exactly, and its rope
  delta closes the position maximum (max + 1 - length);
- the media token runs in the prompt ids exactly cover the merged Vision tokens (the reference
  raises otherwise);
- the Engine's first greedy token equals the reference's argmax at the prompt's last position.

Run it after the engine test exits: the FP64 forward needs the host RAM the engine held.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

from tools.flash_next.reference import F64, Reference, get_rope_index


def read_dump(directory: Path) -> dict:
    ids = [int(v) for v in (directory / "ids.txt").read_text().split()]
    positions_lines = (directory / "positions.txt").read_text().splitlines()
    delta = int(positions_lines[0].split()[1])
    positions = np.array([[int(v) for v in line.split()] for line in positions_lines[1:]], dtype=np.int64)
    t, h, w, merged = [int(v) for v in (directory / "grid.txt").read_text().split()]
    patches = np.fromfile(directory / "patches.bin", dtype=np.uint16)
    features = 3 * 2 * 16 * 16
    if patches.size != t * h * w * features:
        raise ValueError(f"{patches.size} patch words do not cover the {t}x{h}x{w} grid")
    patches = patches.reshape(t * h * w, features)
    greedy = [int(v) for v in (directory / "greedy.txt").read_text().split()]
    return {"ids": ids, "positions": positions, "delta": delta, "grid": (t, h, w), "merged": merged,
            "patches": patches, "greedy": greedy}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True, help="HF checkpoint snapshot directory")
    parser.add_argument("--dump", type=Path, required=True, help="the engine test's output directory")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--out", type=Path, help="save the reference record to this npz")
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    dump = read_dump(args.dump)
    ids, (t, h, w) = dump["ids"], dump["grid"]
    print(f"prompt: {len(ids)} tokens, image grid {t}x{h}x{w} -> {dump['merged']} merged, "
          f"greedy {dump['greedy']}")

    # The BF16 patch words widened exactly as reference.py's --patches path does.
    bits = (dump["patches"].astype(np.uint32) << 16).view(np.float32)
    patches = torch.from_numpy(bits).to(F64)
    ref = Reference(args.model)
    record = ref.forward(ids, patches=patches, grids=[(t, h, w)])

    failures = 0
    def check(ok: bool, what: str) -> None:
        nonlocal failures
        print(f"{'  ok  ' if ok else '  FAIL'} {what}")
        if not ok:
            failures += 1

    expected = get_rope_index(ids, [(t, h // 2, w // 2)], ref.image_id, ref.video_id).numpy()
    got = dump["positions"]
    equal = got.shape == expected.shape and bool((got == expected).all())
    check(equal, f"the frontend's positions equal the upstream rule [{got.shape} vs {expected.shape}]")
    if not equal and got.shape == expected.shape:
        rows = np.where((got != expected).any(axis=1))[0]
        first = rows[0]
        print(f"    first difference at token {first}: {got[first]} vs {expected[first]} "
              f"({len(rows)} tokens differ)")
    check(int(dump["delta"]) == int(expected.max()) + 1 - len(ids),
          f"the rope delta closes the position maximum ({dump['delta']})")

    logits = record["logits"]
    top5 = np.argsort(-logits[-1])[:5]
    print(f"    reference top-5 next tokens: {top5.tolist()}")
    check(dump["greedy"][0] == int(logits[-1].argmax()),
          f"the Engine's first greedy token {dump['greedy'][0]} is the reference's argmax "
          f"{int(logits[-1].argmax())}")
    if args.out is not None:
        np.savez(args.out, **record)
    print("PASS" if failures == 0 else "FAIL")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
