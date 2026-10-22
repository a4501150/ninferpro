"""Per-kernel device time of one T1 round (AGENTS.md "Iteration tiers").

Runs infernix_qwen4_exp_round_bench --profile under Nsight Systems for one shape, then splits the
device time of the profiled replays between their kernels (time where several kernels overlap goes
to each in equal shares, so the column sums to the busy time) and prints the per-round mean.

    python3 tools/flash_next/t1_kernels.py [--shape 6x2] [--replays 20] [--top 25]
        [--bench build/bench/infernix_qwen4_exp_round_bench] [-- extra bench arguments]
"""

from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from collections import defaultdict


def exclusive_times(db: sqlite3.Connection) -> tuple[dict[str, float], float]:
    names = dict(db.execute("select id, value from StringIds"))
    rows = db.execute("select start, end, shortName from CUPTI_ACTIVITY_KIND_KERNEL").fetchall()
    events = []
    for start, end, name in rows:
        events.append((start, 1, names[name]))
        events.append((end, -1, names[name]))
    events.sort()
    active: dict[str, int] = defaultdict(int)
    shares: dict[str, float] = defaultdict(float)
    busy = 0.0
    last = None
    for t, delta, name in events:
        if last is not None and active:
            k = sum(active.values())
            busy += t - last
            for n, count in active.items():
                shares[n] += (t - last) * count / k
        last = t
        active[name] += delta
        if active[name] == 0:
            del active[name]
    return shares, busy


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shape", default="6x2")
    ap.add_argument("--replays", type=int, default=20)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--keep", default=None, help="also copy the profile's SQLite export to this path")
    ap.add_argument("--bench", default="build/bench/infernix_qwen4_exp_round_bench")
    ap.add_argument("extra", nargs="*", help="further round-bench arguments (after --)")
    args = ap.parse_args()
    if "," in args.shape:
        ap.error("one shape per profile")
    with tempfile.TemporaryDirectory() as tmp:
        report = os.path.join(tmp, "t1")
        command = ["nsys", "profile", "--trace=cuda,nvtx", "--cuda-graph-trace=node",
                   "--capture-range=cudaProfilerApi", "--capture-range-end=stop", "-o", report,
                   "--force-overwrite=true", args.bench, "--profile", "--shape", args.shape,
                   "--replays", str(args.replays), *args.extra]
        run = subprocess.run(command, capture_output=True, text=True)
        if run.returncode != 0:
            sys.stderr.write(run.stdout + run.stderr)
            return run.returncode
        sys.stdout.write("".join(line + "\n" for line in run.stdout.splitlines()
                                 if line.startswith(("shape", "  forward", "  commit", "  round", "  hash"))))
        subprocess.run(["nsys", "export", "--type", "sqlite", "-o", report + ".sqlite", "--force-overwrite=true",
                        report + ".nsys-rep"], check=True, capture_output=True)
        db = sqlite3.connect(report + ".sqlite")
        shares, busy = exclusive_times(db)
        db.close()
        if args.keep:
            shutil.copyfile(report + ".sqlite", args.keep)
    rounds = args.replays
    print(f"\nkernels per round ({rounds} profiled replays; overlapping time split between kernels)")
    for name, t in sorted(shares.items(), key=lambda item: -item[1])[: args.top]:
        print(f"  {t / rounds / 1e3:9.1f} us  {100 * t / busy:5.1f}%  {name}")
    rest = sum(sorted(shares.values(), reverse=True)[args.top:])
    if rest:
        print(f"  {rest / rounds / 1e3:9.1f} us  {100 * rest / busy:5.1f}%  ({len(shares) - args.top} others)")
    print(f"  {busy / rounds / 1e3:9.1f} us  busy per round")
    return 0


if __name__ == "__main__":
    sys.exit(main())
