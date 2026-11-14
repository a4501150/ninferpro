"""Where a decode step's time goes, end to end (AGENTS.md "Iteration tiers", T2).

Runs a command that drives the public Engine (a family's engine bench with --profile, which brackets
its measured sets with cudaProfilerStart/Stop) under Nsight Systems and splits every decode round
the Engine runs: the window from one `decode` NVTX range's start to the next one's.

  - Host: the Engine thread's time by the innermost range open below `decode` (the runtime's
    `program.submit`, `device.wait`, `program.post`, and whatever phases the family names: a
    drafter's proposal, the target's launch, its waits), `(decode)` where no finer range is open,
    and the Engine's loop between rounds.
  - GPU busy time by launch site: each kernel, copy or set goes to the NVTX path open on the thread
    that launched it (a graph's kernels to the path of its cudaGraphLaunch), so the target model's
    graph, a drafter's calls and the eager work around them separate by the ranges the family
    already emits. Overlapping work splits its time equally, so the column sums to the busy time.
  - GPU idle time, split by what the Engine thread was doing meanwhile: every microsecond the GPU
    waits on the host, by phase.

Rounds whose window also runs a prompt (a `prefill` range) are counted and left out of the table.

    python3 tools/tiers/step.py [--top 12] [--kernels 5] [--keep FILE.sqlite] -- COMMAND...
"""

from __future__ import annotations

import argparse
import bisect
import itertools
import shutil
import sqlite3
import statistics
import sys
import tempfile
from collections import defaultdict

import nsys

BETWEEN = "(between rounds)"


def columns(db: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in db.execute(f"pragma table_info({table})")}


def tables(db: sqlite3.Connection) -> set[str]:
    return {row[0] for row in db.execute("select name from sqlite_master where type = 'table'")}


def load_ranges(db: sqlite3.Connection) -> dict[int, list[tuple[int, int, str, int]]]:
    """Push/pop NVTX ranges per thread: (start, end, name, payload), sorted by start."""
    names = dict(db.execute("select id, value from StringIds"))
    have = columns(db, "NVTX_EVENTS")
    payload = "uint64Value" if "uint64Value" in have else "0"
    out: dict[int, list] = defaultdict(list)
    query = f"select start, end, globalTid, text, textId, {payload} from NVTX_EVENTS where eventType = 59 and end is not null"
    for start, end, tid, text, text_id, value in db.execute(query):
        name = text if text is not None else names.get(text_id, "?")
        out[tid].append((start, end, name, value or 0))
    for ranges in out.values():
        ranges.sort(key=lambda r: (r[0], -r[1]))
    return out


def stack_paths(ranges: list[tuple[int, int, str, int]], times: list[int]) -> list[str]:
    """The NVTX path open at each of `times` (sorted) on one thread."""
    out = []
    stack: list[tuple[int, str]] = []
    i = 0
    for t in times:
        while i < len(ranges) and ranges[i][0] <= t:
            while stack and stack[-1][0] <= ranges[i][0]:
                stack.pop()
            stack.append((ranges[i][1], ranges[i][2]))
            i += 1
        while stack and stack[-1][0] < t:
            stack.pop()
        out.append("/".join(name for _, name in stack) or "(no range)")
    return out


ACTIVITY_TABLES = (("CUPTI_ACTIVITY_KIND_KERNEL", None), ("CUPTI_ACTIVITY_KIND_MEMCPY", "[memcpy]"),
                   ("CUPTI_ACTIVITY_KIND_MEMSET", "[memset]"))


def gpu_activity(db: sqlite3.Connection) -> list[tuple[int, int, int, str]]:
    """Every kernel, copy and set: (start, end, correlationId, name)."""
    names = dict(db.execute("select id, value from StringIds"))
    have = tables(db)
    out = []
    for table, label in ACTIVITY_TABLES:
        if table not in have:
            continue
        if label is None:
            query = f"select start, end, correlationId, shortName from {table}"
            out += [(s, e, c, names.get(n, "?")) for s, e, c, n in db.execute(query)]
        else:
            out += [(s, e, c, label) for s, e, c in db.execute(f"select start, end, correlationId from {table}")]
    out.sort()
    return out


def launch_sites(db: sqlite3.Connection, ranges: dict[int, list]) -> dict[int, str]:
    """correlationId -> the NVTX path open on the launching thread at the launch call."""
    have = tables(db)
    launched = " union ".join(f"select correlationId from {t}" for t, _ in ACTIVITY_TABLES if t in have)
    calls: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for start, tid, corr in db.execute("select start, globalTid, correlationId from CUPTI_ACTIVITY_KIND_RUNTIME "
                                       f"where correlationId in ({launched})"):
        calls[tid].append((start, corr))
    out = {}
    for tid, items in calls.items():
        items.sort()
        paths = stack_paths(ranges.get(tid, []), [t for t, _ in items])
        for (_, corr), path in zip(items, paths):
            out[corr] = path
    return out


def union(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[list[int]] = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def overlap(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> int:
    """Total overlap of two sorted, disjoint interval lists."""
    i = j = total = 0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        total += max(0, hi - lo)
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def host_segments(nested: list[tuple[int, int, str, int]], start: int, end: int, hi: int,
                  depth: int) -> list[tuple[int, int, str]]:
    """Disjoint segments of [start, hi): inside the round each labelled by the path of the ranges
    open below `decode` (cut at `depth` names), then the loop up to the next round."""
    events = sorted([(s, 1, n) for s, e, n, _ in nested] + [(e, 0, n) for s, e, n, _ in nested])
    out, stack, last = [], [], start
    for t, opening, name in events:
        if t > last:
            out.append((last, t, "/".join(stack[:depth]) or "(decode)"))
            last = t
        if opening:
            stack.append(name)
        else:
            stack.pop()  # push/pop ranges nest, so the closing one is the innermost
    if end > last:
        out.append((last, end, "(decode)"))
    if hi > end:
        out.append((end, hi, BETWEEN))
    return out


def analyse(db: sqlite3.Connection, top: int, kernels: int, depth: int) -> int:
    ranges = load_ranges(db)
    engine_tid, rounds = None, []
    for tid, items in ranges.items():
        found = [r for r in items if r[2] == "decode"]
        if len(found) > len(rounds):
            engine_tid, rounds = tid, found
    if len(rounds) < 2:
        print("fewer than two decode rounds in the capture", file=sys.stderr)
        return 1
    engine = ranges[engine_tid]
    engine_starts = [r[0] for r in engine]
    prefills = union([(s, e) for s, e, n, _ in engine if n == "prefill" or n.startswith("prefill.")])
    activity = gpu_activity(db)
    sites = launch_sites(db, ranges)
    starts = [a[0] for a in activity]
    # The latest end so far: the first activity still running at t is at bisect_right(reach, t).
    reach = list(itertools.accumulate((a[1] for a in activity), max))

    host: dict[str, float] = defaultdict(float)
    idle_by: dict[str, float] = defaultdict(float)
    site_time = defaultdict(float)
    kernel_time: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    walls, busies, rows = [], [], defaultdict(int)
    skipped = 0
    for i, (start, end, _, payload) in enumerate(rounds[:-1]):
        lo, hi = start, rounds[i + 1][0]
        if overlap([(lo, hi)], prefills) > 0:
            skipped += 1
            continue
        rows[payload] += 1
        walls.append(hi - lo)
        # Host: the Engine thread's ranges inside the round, then its loop up to the next round.
        nested = [r for r in engine[bisect.bisect_left(engine_starts, start):bisect.bisect_left(engine_starts, end)]
                  if not (r[0] == start and r[2] == "decode")]
        segments = host_segments([(s, min(e, end), n, p) for s, e, n, p in nested], start, end, hi, depth)
        for s, e, label in segments:
            host[label] += e - s
        # GPU: the activity that starts or runs inside the window.
        window = [a for a in activity[bisect.bisect_right(reach, lo):bisect.bisect_left(starts, hi)] if a[1] > lo]
        busy = union([(max(s, lo), min(e, hi)) for s, e, _, _ in window])
        busies.append(sum(e - s for s, e in busy))
        for label, t in nsys.shares([(s, e, sites.get(c, "(unknown)")) for s, e, c, _ in window], lo, hi)[0].items():
            site_time[label] += t
        if kernels:
            for s, e, c, name in window:
                kernel_time[sites.get(c, "(unknown)")][name] += min(e, hi) - max(s, lo)
        # Idle: the gaps in `busy`, by the Engine thread's state meanwhile.
        gaps, cursor = [], lo
        for s, e in busy:
            if s > cursor:
                gaps.append((cursor, s))
            cursor = max(cursor, e)
        if cursor < hi:
            gaps.append((cursor, hi))
        by_label: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for s, e, label in segments:
            by_label[label].append((s, e))
        for label, spans in by_label.items():
            idle_by[label] += overlap(gaps, union(spans))

    n = len(walls)
    if n == 0:
        print(f"every one of {len(rounds) - 1} rounds ran beside a prompt", file=sys.stderr)
        return 1
    wall = sum(walls) / n
    us = lambda total: total / n / 1e3  # noqa: E731
    print(f"\n{n} decode rounds ({skipped} beside a prompt left out); rows per round: "
          + ", ".join(f"{r} x{c}" for r, c in sorted(rows.items())))
    print(f"round window: mean {wall / 1e3:.1f} us, median {statistics.median(walls) / 1e3:.1f} us")
    busy = sum(busies) / n
    print(f"\nGPU per round: busy {busy / 1e3:9.1f} us ({100 * busy / wall:4.1f}%), idle {(wall - busy) / 1e3:9.1f} us")
    def table(title: str, times: dict[str, float]) -> None:
        print(title)
        ordered = sorted(times.items(), key=lambda item: -item[1])
        for name, t in ordered[:top]:
            print(f"  {us(t):9.1f} us  {100 * t / n / wall:5.1f}%  {name}")
        rest = sum(t for _, t in ordered[top:])
        if rest:
            print(f"  {us(rest):9.1f} us  {100 * rest / n / wall:5.1f}%  ({len(ordered) - top} others)")

    table("\nhost per round (the Engine thread, by the innermost range open below `decode`)", host)
    table("\nGPU idle per round, by what the Engine thread was doing", idle_by)
    print("\nGPU busy per round, by launch site (NVTX path on the launching thread)")
    ordered = sorted(site_time.items(), key=lambda item: -item[1])
    for label, t in ordered[:top]:
        print(f"  {us(t):9.1f} us  {100 * t / n / busy:5.1f}%  {label}")
        for name, k in sorted(kernel_time[label].items(), key=lambda item: -item[1])[:kernels]:
            print(f"  {'':9}     {us(k):9.1f} us  {name}")
    rest = sum(t for _, t in ordered[top:])
    if rest:
        print(f"  {us(rest):9.1f} us  {100 * rest / n / busy:5.1f}%  ({len(ordered) - top} other sites)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--top", type=int, default=12, help="rows of each table to list")
    ap.add_argument("--kernels", type=int, default=0, help="also list each site's top kernels")
    ap.add_argument("--depth", type=int, default=2, help="names of a host range path below `decode` to keep")
    ap.add_argument("--keep", help="also copy the profile's SQLite export to this path")
    ap.add_argument("--sqlite", help="analyse this existing export instead of running a command")
    ap.add_argument("command", nargs="*", help="the Engine command (after --)")
    args = ap.parse_args()
    if args.sqlite:
        with sqlite3.connect(args.sqlite) as db:
            return analyse(db, args.top, args.kernels, args.depth)
    if not args.command:
        ap.error("give a command after --, or --sqlite")
    with tempfile.TemporaryDirectory() as tmp:
        run, export = nsys.capture(args.command, tmp, "step")
        sys.stdout.write("".join(line + "\n" for line in run.stdout.splitlines()
                                 if not line.startswith(("Collecting", "Generating", "Processing", "Capture", "  "))))
        if not export:
            sys.stderr.write(run.stderr)
            return run.returncode
        if args.keep:
            shutil.copyfile(export, args.keep)
        with sqlite3.connect(export) as db:
            return analyse(db, args.top, args.kernels, args.depth)


if __name__ == "__main__":
    sys.exit(main())
