"""Per-kernel device time of one T1 round (AGENTS.md "Iteration tiers").

Runs infernix_qwen4_exp_round_bench --profile under Nsight Systems for one shape, then splits the
device time of the profiled replays between their kernels (time where several kernels overlap goes
to each in equal shares, so the column sums to the busy time) and prints the per-round mean.

    python3 tools/flash_next/t1_kernels.py [--shape 6x2] [--replays 20] [--top 25] [--gpu-metrics]
        [--bench build/bench/infernix_qwen4_exp_round_bench] [-- extra bench arguments]

--gpu-metrics also samples the GPU's counters (Nsight Systems GPU metrics, GB20x set, every 5 us)
and reports where DRAM bandwidth goes unused: over the replays' busy time, each sample's period
times (1 - DRAM read+write throughput) is bandwidth left unused, shared between the kernels
running at the sample like the device time. The total bounds what fusion, overlap or faster
kernels can save, and the per-kernel column says where; streaming kernels top out near 90 % (the
routed gate/up kernel measured 89.9 % on the RTX PRO 6000), so about a tenth of the busy time is
not recoverable. Kernel traces cannot show this: under PDL a kernel is always running, and the cost of a
boundary (the wait, the ramp, the tail) is time inside kernels at low bandwidth.
"""

from __future__ import annotations

import argparse
import contextlib
import io
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


METRIC_PERIOD_NS = 5000  # --gpu-metrics-frequency 200000


def metrics_report(db: sqlite3.Connection, rounds: int, top: int) -> None:
    names = dict(db.execute("select id, value from StringIds"))
    metric = {name: mid for mid, name in db.execute("select metricId, metricName from TARGET_INFO_GPU_METRICS")}
    wanted = {"read": "DRAM Read Bandwidth [Throughput %]", "write": "DRAM Write Bandwidth [Throughput %]",
              "sms": "SMs Active [Throughput %]", "issue": "SM Issue [Throughput %]"}
    missing = [w for w in wanted.values() if w not in metric]
    if missing:
        print(f"\nGPU metrics: the export lacks {', '.join(missing)}")
        return
    samples: dict[int, dict[str, float]] = defaultdict(dict)
    for key, name in wanted.items():
        for ts, value in db.execute("select timestamp, value from GPU_METRICS where metricId = ?", (metric[name],)):
            samples[ts][key] = value
    events = []
    for start, end, name in db.execute("select start, end, shortName from CUPTI_ACTIVITY_KIND_KERNEL"):
        events.append((start, 1, names[name]))
        events.append((end, -1, names[name]))
    events.sort()
    # A sample covers the period before its timestamp; it goes to the kernels active at its midpoint.
    points = sorted((ts - METRIC_PERIOD_NS // 2, values) for ts, values in samples.items() if len(values) == len(wanted))
    active: dict[str, int] = defaultdict(int)
    unused: dict[str, float] = defaultdict(float)
    timed: dict[str, float] = defaultdict(float)
    dram_time: dict[str, float] = defaultdict(float)
    sums = defaultdict(float)
    bands = [0.0, 0.0, 0.0, 0.0]
    busy = 0.0
    e = 0
    for at, values in points:
        while e < len(events) and events[e][0] <= at:
            _, delta, name = events[e]
            active[name] += delta
            if active[name] == 0:
                del active[name]
            e += 1
        if not active:
            continue
        dram = min(100.0, values["read"] + values["write"])
        busy += METRIC_PERIOD_NS
        for key in ("sms", "issue"):
            sums[key] += values[key] * METRIC_PERIOD_NS
        sums["dram"] += dram * METRIC_PERIOD_NS
        bands[min(3, int(dram // 25))] += METRIC_PERIOD_NS
        k = sum(active.values())
        for name, count in active.items():
            share = METRIC_PERIOD_NS * count / k
            timed[name] += share
            dram_time[name] += share * dram
            unused[name] += share * (1.0 - dram / 100.0)
    if busy == 0:
        print("\nGPU metrics: no sample fell inside a kernel")
        return
    total_unused = sum(unused.values())
    print(f"\nGPU metrics ({busy / METRIC_PERIOD_NS:.0f} samples of {METRIC_PERIOD_NS / 1e3:.0f} us over the busy time)")
    print(f"  DRAM read+write {sums['dram'] / busy:5.1f} %   SMs active {sums['sms'] / busy:5.1f} %   "
          f"SM issue {sums['issue'] / busy:5.1f} %")
    print("  time by DRAM throughput: " + "  ".join(
        f"{label} {100 * b / busy:4.1f}%" for label, b in zip(("<25%", "25-50%", "50-75%", ">=75%"), bands)))
    print(f"  bandwidth left unused: {total_unused / rounds / 1e3:.1f} us per round "
          f"({100 * total_unused / busy:.1f}% of the busy time)")
    print("\nkernels by bandwidth left unused per round")
    print("     unused       time   DRAM %  kernel")
    for name, u in sorted(unused.items(), key=lambda item: -item[1])[:top]:
        print(f"  {u / rounds / 1e3:9.1f} {timed[name] / rounds / 1e3:9.1f} us  {dram_time[name] / timed[name]:5.1f}  {name}")
    rest = sorted(unused.values(), reverse=True)[top:]
    if rest:
        print(f"  {sum(rest) / rounds / 1e3:9.1f}                      ({len(rest)} others)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shape", default="6x2")
    ap.add_argument("--replays", type=int, default=20)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--keep", default=None, help="also copy the profile's SQLite export to this path")
    ap.add_argument("--gpu-metrics", action="store_true",
                    help="also sample GPU counters and report the DRAM bandwidth left unused, per kernel")
    ap.add_argument("--bench", default="build/bench/infernix_qwen4_exp_round_bench")
    ap.add_argument("extra", nargs="*", help="further round-bench arguments (after --)")
    args = ap.parse_args()
    if "," in args.shape:
        ap.error("one shape per profile")
    with tempfile.TemporaryDirectory() as tmp:
        report = os.path.join(tmp, "t1")
        metrics = ["--gpu-metrics-devices=0", "--gpu-metrics-set=gb20x",
                   f"--gpu-metrics-frequency={1_000_000_000 // METRIC_PERIOD_NS}"] if args.gpu_metrics else []
        command = ["nsys", "profile", "--trace=cuda,nvtx", "--cuda-graph-trace=node", *metrics,
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
        metrics_text = ""
        if args.gpu_metrics:
            with io.StringIO() as text, contextlib.redirect_stdout(text):
                metrics_report(db, args.replays, args.top)
                metrics_text = text.getvalue()
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
    sys.stdout.write(metrics_text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
