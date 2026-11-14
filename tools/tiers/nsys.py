"""Nsight Systems capture and timeline arithmetic shared by the tier tools."""

from __future__ import annotations

import os
import subprocess
from collections import defaultdict


def capture(command: list[str], directory: str, name: str, options: list[str] = ()) -> tuple[subprocess.CompletedProcess, str]:
    """Runs `command` under `nsys profile` (CUDA and NVTX, graph nodes traced, the capture bracketed
    by cudaProfilerStart/Stop) and exports the report to SQLite; returns the run and the export's
    path (empty when the run failed)."""
    report = os.path.join(directory, name)
    run = subprocess.run(["nsys", "profile", "--trace=cuda,nvtx", "--cuda-graph-trace=node", *options,
                          "--capture-range=cudaProfilerApi", "--capture-range-end=stop", "-o", report,
                          "--force-overwrite=true", *command], capture_output=True, text=True)
    if run.returncode != 0:
        return run, ""
    subprocess.run(["nsys", "export", "--type", "sqlite", "-o", report + ".sqlite", "--force-overwrite=true",
                    report + ".nsys-rep"], check=True, capture_output=True)
    return run, report + ".sqlite"


def shares(items: list[tuple[int, int, str]], lo: float = float("-inf"),
           hi: float = float("inf")) -> tuple[dict[str, float], float]:
    """Time of [lo, hi) covered by the intervals, per label (each instant split equally between the
    intervals open, so the labels sum to the busy time), and the busy time."""
    events = []
    for s, e, label in items:
        s, e = max(s, lo), min(e, hi)
        if s < e:
            events += [(s, 1, label), (e, -1, label)]
    events.sort()
    active: dict[str, int] = defaultdict(int)
    out: dict[str, float] = defaultdict(float)
    busy = 0.0
    last = None
    for t, delta, label in events:
        if last is not None and active:
            k = sum(active.values())
            busy += t - last
            for name, count in active.items():
                out[name] += (t - last) * count / k
        last = t
        active[label] += delta
        if active[label] == 0:
            del active[label]
    return out, busy
