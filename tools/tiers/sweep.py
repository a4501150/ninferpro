"""Decode round time against context length (AGENTS.md "Iteration tiers", T1).

Runs a model family's round bench (docs/maintainer/round-bench.md) once per --context spec with
--report, then prints for every shape and context: the round time, its speed of light (the bytes
the round must move at least over the device read rate) and the ratio between them, the speed per
row and in aggregate at the committed tokens per row, the bytes by class, and the round time
against the same shape at the first context (the slowdown long contexts cost).

    python3 tools/tiers/sweep.py --bench build/bench/infernix_<family>_round_bench \
        [--shape 1x1,6x1,6x3] [--context 1024 --context 32768 --context 131072,1024 ...]
        [--out FILE.jsonl] [-- extra bench arguments]

A context spec is the bench's --context value: one length for every lane, or lane lengths in lane
order (`131072,1024` puts one long lane beside short ones; the round's rows are the first lanes).
Every lane is prefilled, so the KV of all eight lanes must fit beside the experts: give the lanes a
round does not use a short context (`131072x6,1024`).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

DEFAULT_CONTEXTS = ["1024", "8192", "32768", "131072x6,1024"]


def run_point(bench: str, shape: str, context: str, extra: list[str]) -> tuple[list[dict], str]:
    with tempfile.TemporaryDirectory() as tmp:
        report = os.path.join(tmp, "report.jsonl")
        command = [bench, "--shape", shape, "--context", context, "--report", report, *extra]
        run = subprocess.run(command, capture_output=True, text=True)
        records = []
        if os.path.exists(report):
            with open(report) as f:
                records = [json.loads(line) for line in f if line.strip()]
        error = "" if run.returncode == 0 else (run.stderr.strip().splitlines() or ["exit %d" % run.returncode])[-1]
        for record in records:
            record["context_spec"] = context
        return records, error


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", required=True, help="the family's round bench binary")
    ap.add_argument("--shape", default="1x1,6x1,6x3", help="round shapes, as the bench's --shape (plain shapes only)")
    ap.add_argument("--context", action="append", help=f"a context spec; repeatable (default {' '.join(DEFAULT_CONTEXTS)})")
    ap.add_argument("--out", help="also write every report line, with its context spec, to this JSON-lines file")
    ap.add_argument("extra", nargs="*", help="further round-bench arguments (after --)")
    args = ap.parse_args()
    if "+" in args.shape:
        ap.error("mixed shapes have no speed of light; sweep plain shapes")
    contexts = args.context or DEFAULT_CONTEXTS

    records: list[dict] = []
    for context in contexts:
        print(f"context {context} ...", file=sys.stderr, flush=True)
        got, error = run_point(args.bench, args.shape, context, args.extra)
        records += got
        if error:
            print(f"  context {context}: the bench failed: {error}", file=sys.stderr)
    if args.out:
        with open(args.out, "w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
    if not records:
        print("no results", file=sys.stderr)
        return 1

    first: dict[str, float] = {}
    print(f"\nread rate {records[0]['read_gbs']:.0f} GB/s; speeds at the committed tokens per row (accept)")
    print(f"{'shape':>6} {'context':>18} {'round us':>9} {'ideal us':>9} {'x ideal':>7} {'tok/s/row':>9} {'tok/s':>8}"
          f" {'dense':>6} {'experts':>7} {'kv':>6} {'state':>6} {'GB':>6} {'vs first':>8}")
    for shape in dict.fromkeys(r["shape"] for r in records):
        for r in (r for r in records if r["shape"] == shape):
            b = r["bytes"]
            per_row = r["accept"] / (r["round_us"] * 1e-6)
            first.setdefault(shape, r["round_us"])
            print(f"{shape:>6} {r['context_spec']:>18} {r['round_us']:9.1f} {r['ideal_us']:9.1f} "
                  f"{r['round_us'] / r['ideal_us']:7.2f} {per_row:9.1f} {per_row * r['rows']:8.1f} "
                  f"{b['dense'] / 1e9:6.3f} {b['experts'] / 1e9:7.3f} {b['kv'] / 1e9:6.3f} {b['state'] / 1e9:6.3f} "
                  f"{b['total'] / 1e9:6.3f} {r['round_us'] / first[shape]:7.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
