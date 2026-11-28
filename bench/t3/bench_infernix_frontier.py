#!/usr/bin/env python3
"""Frontier benchmark: one infernix-serve boot, a sweep of client concurrency
levels over agentic mixed-length traffic, and the throughput / per-user-speed
frontier with an SLA pick.

Each level serves the traffic mix at one in-flight cap for a fixed number of
seconds (the level stops issuing new requests when its budget ends and waits
for the ones in flight), and gives one point:

- total output tok/s: all completion tokens over the level's wall, from the
  server's own request-log timestamps;
- per-user decode tok/s (median and p10 over the requests' own decode speeds),
  TTFT p50/p90, speculative acceptance and prefix-cache hit rate - all read
  from the server's --request-log-jsonl machine log, never client timing.

The frontier is the Pareto set over (median per-user tok/s, total tok/s); the
SLA pick is the level with the highest total throughput whose median per-user
decode speed still meets --sla-tok-s (default 30).

Traffic (reused, not reinvented):

- --traffic agentic (default): replays ninferpro's bench/agentic_ab workload -
  the deterministic plan shaped from ~3,300 production requests (long resumed
  sessions, tool loops, compaction, subagent fan-outs, retries and aborts),
  with each arm's own model output fed back as an agent client does. The A/B
  fixed send order is replaced by the level's in-flight cap; every level
  replays its own plan seed (seed + level), so request seeds stay unique per
  level and each level is one draw of the same shaped traffic.
- --traffic synthetic: one long-context session (growing conversation over a
  --long-prompt-tokens prompt) beside N-1 short conversations.

One server boot (production flags from a configs/*.toml profile, like
bench_infernix.py), client-side caps sweeping 1,2,4,6,8 by default while the
server boots --max-concurrency 8 (the engine's lanes). Levels run sequentially
on that one boot; the prefix cache carries state between levels, but each
level's prompts differ (per-level plan seeds / unique synthetic tags), so no
level re-reads another's cache.

The shared gpu lock is held for the whole life of the server - from before
the boot until the server has been killed by PID and has exited - so other
agents' GPU runs cannot collide with a live model outside the lock. Keep runs
short; a quiet-machine promotion run additionally holds the build lock.

Usage:
  python3 bench/t3/bench_infernix_frontier.py \
      --binary build/apps/infernix-serve --port 8411

  # quick smoke of both traffic modes:
  python3 bench/t3/bench_infernix_frontier.py --binary ... --port 8411 \
      --traffic agentic --concurrency 1,4 --seconds 60 --scale 0.3 \
      --max-tokens 512
  python3 bench/t3/bench_infernix_frontier.py --binary ... --port 8411 \
      --traffic synthetic --concurrency 2 --seconds 40

Output: <out>.json and <out>.md (default results/infernix_frontier_<ninferpro
commit>), plus <out>.server.log, <out>.requests.jsonl (the serve's machine
log) and <out>.levels/L<n>/client.jsonl (per-level request rows). Rows are
written incrementally after each level, so a partial run still leaves partial
results.
"""

import argparse
import fcntl
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bench_infernix as bi  # noqa: E402  (same bench/t3 directory)

NINFERPRO = os.environ.get("NINFERPRO_DIR", bi.REPO_ROOT)
DEFAULT_CONCURRENCY = (1, 2, 4, 6, 8)
CHAT_PROMPTS = bi.CHAT_PROMPTS

CURRENT = {"server": None, "lock": None}


def git_short(repo):
    try:
        out = subprocess.run(
            ["git", "-C", repo, "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def git_head(repo):
    try:
        out = subprocess.run(
            ["git", "-C", repo, "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def pct(vals, p):
    """Linear-interpolated percentile of `vals` at fraction p."""
    if not vals:
        return None
    s = sorted(vals)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def frontier(points):
    """Pareto set over (median per-user tok/s, total tok/s): the best total
    throughput at each per-user speed. A point is dominated when another is at
    least as good on both axes and strictly better on one."""
    front = []
    for p in points:
        x, y = p.get("per_user_med_tok_s"), p.get("total_output_tok_s")
        if x is None or y is None:
            continue
        dominated = False
        for q in points:
            if q is p:
                continue
            qx, qy = q.get("per_user_med_tok_s"), q.get("total_output_tok_s")
            if qx is None or qy is None:
                continue
            if qx >= x and qy >= y and (qx > x or qy > y):
                dominated = True
                break
        if not dominated:
            front.append(p)
    return sorted(front, key=lambda p: p["total_output_tok_s"], reverse=True)


def sla_pick(points, sla):
    """The level with the highest total throughput whose median per-user
    decode speed meets the SLA."""
    ok = [p for p in points
          if p.get("per_user_med_tok_s") is not None
          and p["per_user_med_tok_s"] >= sla
          and p.get("total_output_tok_s") is not None]
    if not ok:
        return None
    return max(ok, key=lambda p: p["total_output_tok_s"])


def load_request_records(path, seeds):
    """request_done / request_start events for the given request seeds, plus
    the count of error/reject events, from the serve's JSONL machine log."""
    done, starts, errors = {}, {}, 0
    with open(path, errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except ValueError:
                continue  # a torn line mid-write
            name = ev.get("event")
            req = ev.get("request") or {}
            try:
                seed = int((req.get("sampling") or {}).get("seed"))
            except (TypeError, ValueError):
                continue
            if seed not in seeds:
                continue
            if name == "request_done":
                done[seed] = ev
            elif name == "request_start":
                starts[seed] = ev
            elif name in ("request_error", "request_rejected"):
                errors += 1
    return done, starts, errors


def level_metrics(done, starts, errors):
    """One level's point, from the server's own request records."""
    recs = list(done.values())
    if not recs:
        return None
    ts = [int(e.get("timestamp_unix_ms") or 0) for e in recs] + \
         [int(e.get("timestamp_unix_ms") or 0) for e in starts.values()]
    span = (max(ts) - min(ts)) / 1000.0 if ts else None
    dec, ttft = [], []
    out_toks = prompt_toks = cache_toks = prefill_toks = 0
    acc_a = acc_b = 0
    for e in recs:
        r = e.get("result") or {}
        t = e.get("timings_seconds") or {}
        comp = int(r.get("completion_tokens") or 0)
        out_toks += comp
        prompt_toks += int(r.get("prompt_tokens") or 0)
        cache_toks += int(r.get("prefix_cache_hit_tokens") or 0)
        prefill_toks += int(r.get("computed_prefill_tokens") or 0)
        d = t.get("decode")
        if comp > 1 and d:
            dec.append((comp - 1) / float(d))
        if t.get("ttft") is not None:
            ttft.append(float(t["ttft"]))
        spec = e.get("speculative") or {}
        acc_a += int(spec.get("accepted_tokens") or 0)
        acc_b += int(spec.get("drafted_tokens") or 0)
    m = {
        "requests_done": len(recs),
        "request_errors": errors,
        "span_s": round(span, 1) if span else None,
        "output_tokens": out_toks,
        "prompt_tokens": prompt_toks,
        "computed_prefill_tokens": prefill_toks,
        "total_output_tok_s": round(out_toks / span, 1) if span else None,
        "per_user_med_tok_s": round(pct(dec, 0.5), 1) if dec else None,
        "per_user_p10_tok_s": round(pct(dec, 0.1), 1) if dec else None,
        "per_user_requests": len(dec),
        "ttft_p50_s": round(pct(ttft, 0.5), 3) if ttft else None,
        "ttft_p90_s": round(pct(ttft, 0.9), 3) if ttft else None,
        "accept_pct": round(100.0 * acc_a / acc_b, 1) if acc_b else None,
        "cache_hit_pct": round(100.0 * cache_toks / prompt_toks, 1)
        if prompt_toks else None,
    }
    return m


class GpuLock:
    """The shared gpu lock, held for the whole life of the server."""

    def __init__(self, path):
        self.path = path
        self.fd = None

    def acquire(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        fcntl.flock(self.fd, fcntl.LOCK_EX)

    def release(self):
        if self.fd is not None:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            os.close(self.fd)
            self.fd = None


def load_agentic_ab(args):
    """Import ninferpro's bench/agentic_ab runner + workload, with the corpus
    pinned to --corpus-commit (their pinned default e48a0d28 is an Infernix
    commit this repository has no shared history with, so the frontier resolves
    a ninferpro commit itself)."""
    d = args.agentic_ab_dir
    if not os.path.isdir(d):
        raise SystemExit("agentic_ab not found at " + d)
    os.environ["AB_CORPUS_REPO"] = NINFERPRO
    os.environ["AB_CORPUS_COMMIT"] = args.corpus_commit
    if d not in sys.path:
        sys.path.insert(0, d)
    import runner  # noqa: E402  (imports workload itself)
    runner.HOST = args.host
    runner.PORT = args.port
    return runner


def run_agentic_level(runner, args, level, model_id, levels_dir, max_context):
    """One level of the agentic replay: the agentic_ab plan with its closed
    loop (model output fed back), capped at `level` requests in flight and at
    `args.seconds` of issued requests."""
    workload = __import__("workload")
    plan = workload.build_plan(seed=args.seed + level, scale=args.scale)

    class LevelClient(runner.ArmClient):
        """ArmClient with the A/B fixed send order replaced by the level's
        in-flight cap: a slot is held from send to the last streamed token.
        Requests past the level's budget are skipped - the actor still walks
        its remaining steps, so no signal another actor waits on is stranded -
        and --max-tokens caps the per-request output budget for smoke runs
        (0 keeps the plan's own 64k agent turns)."""

        def __init__(self):
            runner.ArmClient.__init__(
                self, "c{0}".format(level), plan, model_id,
                os.path.join(levels_dir, "L{0}".format(level)), max_context)
            self.slots = threading.BoundedSemaphore(level)
            self.stop_at = time.time() + args.seconds
            self.seeds = set()
            self.skipped = 0

        def post(self, messages, tools, max_tokens, seed, abort_after=None,
                 order=None):
            if self.max_tokens_cap:
                max_tokens = min(max_tokens, self.max_tokens_cap)
            if time.time() >= self.stop_at:
                res = runner.Result()
                res.status = "skipped"
                res.t_send = res.t_done = time.time()
                self.skipped += 1
                return res
            self.seeds.add(seed)
            with self.slots:
                return runner.ArmClient.post(
                    self, messages, tools, max_tokens, seed, abort_after, None)

    client = LevelClient()
    client.max_tokens_cap = args.max_tokens or 0
    os.makedirs(os.path.join(levels_dir, "L{0}".format(level)), exist_ok=True)
    client.run()
    return client


def run_synthetic_level(args, level, base, model):
    """One level of the synthetic mix: one long-context session (a growing
    conversation over a --long-prompt-tokens prompt) beside level-1 short
    conversations, closed-loop, until the level's budget ends."""
    stop_at = time.time() + args.seconds
    seeds = set()
    failures = []
    lock = threading.Lock()

    def note_failure(msg):
        with lock:
            failures.append(msg)

    def chat(content, max_tokens, seed, timeout=600):
        messages = content if isinstance(content, list) else \
            [{"role": "user", "content": content}]
        body = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 1.0, "top_p": 0.95, "top_k": 20, "seed": seed,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        return bi.http("POST", base + "/v1/chat/completions", body, timeout)

    long_prompt = "[frontier-L{0}-{1}] ".format(level, uuid.uuid4().hex[:8]) \
        + bi.filler(args.long_prompt_tokens, 9000 + level) \
        + "\n\nWrite a detailed technical essay about the system described" \
          " above; continue it over the following turns."

    def long_session():
        history = [{"role": "user", "content": long_prompt}]
        turn = 0
        while time.time() < stop_at:
            seed = (level + 1) * 10 ** 9 + turn
            seeds.add(seed)
            st, resp = chat(history, args.long_max_tokens, seed)
            if st != 200 or not isinstance(resp, dict):
                note_failure("long turn {0}: HTTP {1}".format(turn, st))
                return
            msg = resp["choices"][0]["message"]
            # the whole conversation keeps growing, as an agent session does
            history = history + [
                {"role": "assistant", "content": msg.get("content") or ""},
                {"role": "user", "content": "Continue with the next section."},
            ]
            turn += 1

    def short_worker(w):
        turn = 0
        while time.time() < stop_at:
            seed = (level + 1) * 10 ** 9 + w * 10 ** 6 + turn
            seeds.add(seed)
            prompt = "[frontier-L{0}-w{1}-t{2}] {3}".format(
                level, w, turn, CHAT_PROMPTS[turn % len(CHAT_PROMPTS)])
            st, resp = chat(prompt, args.short_max_tokens, seed)
            if st != 200:
                note_failure("short w{0} t{1}: HTTP {2}".format(w, turn, st))
                return
            turn += 1

    workers = [threading.Thread(target=long_session, daemon=True)]
    workers += [threading.Thread(target=short_worker, args=(w,), daemon=True)
                for w in range(1, level)]
    for t in workers:
        t.start()
    for t in workers:
        t.join()
    return {"seeds": seeds, "failures": failures}


def render_md(data):
    lines = [
        "# Infernix frontier benchmark",
        "",
        "- binary: `{0}`".format(data["binary"]),
        "- engine commit: `{0}` (corpus `{1}`)".format(
            data["engine_commit"], data["corpus_commit"]),
        "- traffic: {0} (seed {1}, scale {2}, {3} s/level), SLA {4} tok/s per"
        " user".format(data["traffic"], data["seed"], data["scale"],
                       data["seconds"], data["sla_tok_s"]),
        "- boot: ready in {0:.1f} s, placement `{1}`, max-concurrency {2},"
        " drafter `{3}`".format(
            data["boot"]["ready_s"], data["boot"].get("placement") or "default",
            data["boot"]["max_concurrency"], data["boot"].get("drafter") or "n/a"),
        "",
        "| C | reqs | total out tok/s | per-user tok/s (med/p10) | TTFT p50/p90 s"
        " | accept % | cache hit % | proc VRAM GiB | frontier |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    front_ids = {id(p) for p in data["frontier"]}
    pick = data.get("sla_pick") or {}
    for r in data["levels"]:
        on_front = id(r) in front_ids
        lines.append(
            "| {0} | {1} | {2} | {3} / {4} | {5} / {6} | {7} | {8} | {9}"
            " | {10}{11} |".format(
                r["concurrency"], r.get("requests_done", 0),
                bi.fmt(r.get("total_output_tok_s"), 1),
                bi.fmt(r.get("per_user_med_tok_s"), 1),
                bi.fmt(r.get("per_user_p10_tok_s"), 1),
                bi.fmt(r.get("ttft_p50_s"), 2), bi.fmt(r.get("ttft_p90_s"), 2),
                bi.fmt(r.get("accept_pct")), bi.fmt(r.get("cache_hit_pct")),
                bi.fmt(r.get("proc_vram_gib"), 1),
                "yes" if on_front else "",
                " <- SLA" if r.get("concurrency") == pick.get("concurrency")
                and pick else ""))
    front_txt = ", ".join(
        "C{0} ({1} tok/s/user, {2} total)".format(
            p["concurrency"], bi.fmt(p["per_user_med_tok_s"], 1),
            bi.fmt(p["total_output_tok_s"], 1))
        for p in data["frontier"]) or "none"
    lines.append("")
    lines.append("Frontier: {0}.".format(front_txt))
    if pick:
        lines.append(
            "SLA pick (>= {0} tok/s per user, median): C{1} - {2} total"
            " tok/s at {3} tok/s per user (p10 {4}).".format(
                data["sla_tok_s"], pick["concurrency"],
                bi.fmt(pick.get("total_output_tok_s"), 1),
                bi.fmt(pick.get("per_user_med_tok_s"), 1),
                bi.fmt(pick.get("per_user_p10_tok_s"), 1)))
    else:
        lines.append(
            "SLA pick (>= {0} tok/s per user, median): no level meets it."
            .format(data["sla_tok_s"]))
    return "\n".join(lines) + "\n"


def write_outputs(out, data):
    with open(out + ".json", "w") as f:
        json.dump(data, f, indent=1)
    table = render_md(data)
    with open(out + ".md", "w") as f:
        f.write(table)
    print(table)


def cleanup(*_):
    if CURRENT["server"] is not None:
        CURRENT["server"].stop()
        CURRENT["server"] = None
    if CURRENT["lock"] is not None:
        CURRENT["lock"].release()
        CURRENT["lock"] = None
    sys.exit(130)


def parse_args():
    ap = argparse.ArgumentParser(
        description="one-boot concurrency-sweep frontier benchmark")
    ap.add_argument("--binary", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8411)
    ap.add_argument("--concurrency", default=",".join(str(c)
                                                       for c in DEFAULT_CONCURRENCY),
                    help="comma-separated in-flight caps to sweep, ascending"
                         " (default 1,2,4,6,8)")
    ap.add_argument("--traffic", choices=("agentic", "synthetic"),
                    default="agentic")
    ap.add_argument("--seconds", type=int, default=300,
                    help="budget of issued requests per level (in-flight"
                         " requests still finish)")
    ap.add_argument("--sla-tok-s", type=float, default=30.0,
                    help="per-user median decode tok/s the operating point"
                         " must meet (default 30)")
    ap.add_argument("--seed", type=int, default=1000,
                    help="agentic plan seed base (level i uses seed + i)")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="agentic_ab workload scale (0.3 is a quick smoke)")
    ap.add_argument("--max-tokens", type=int, default=0,
                    help="cap on per-request output tokens for smoke runs"
                         " (0 keeps the plan's own 64k agent turns)")
    ap.add_argument("--corpus-commit", default=None,
                    help="ninferpro commit the agentic corpus reads"
                         " (default: its HEAD at plan time)")
    ap.add_argument("--agentic-ab-dir",
                    default=os.path.join(NINFERPRO, "bench", "agentic_ab"))
    # synthetic-mix knobs
    ap.add_argument("--long-prompt-tokens", type=int, default=30000)
    ap.add_argument("--long-max-tokens", type=int, default=256)
    ap.add_argument("--short-max-tokens", type=int, default=256)
    # server boot (mirrors bench_infernix.py; profile TOML defaults apply)
    ap.add_argument("--config", default=None,
                    help="TOML serving profile (default: configs/GLM-5.3-Flash.toml)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--kv-dtype", default=None)
    ap.add_argument("--max-context", type=int, default=None)
    ap.add_argument("--max-concurrency", type=int, default=None,
                    help="server lanes (default: the largest sweep level)")
    ap.add_argument("--ram-headroom-mib", type=int, default=16384)
    ap.add_argument("--artifact", default=None)
    ap.add_argument("--ngram-volume", default=None)
    ap.add_argument("--spec", default=None,
                    help="speculative drafter for --spec (default: profile TOML's spec)")
    ap.add_argument("--draft-model", default=None)
    ap.add_argument("--experts", default=None,
                    help="boot placement spec (default: profile TOML's experts)")
    ap.add_argument("--server-args", default="",
                    help="extra server flags, split like a shell command line")
    ap.add_argument("--gpu-lock",
                    default=os.path.expanduser(
                        "~/.cache/ninferpro/gpu.lock"),
                    help="shared gpu lock, held from the boot until the server"
                         " is killed (empty string disables)")
    args = ap.parse_args()
    levels = []
    for part in args.concurrency.split(","):
        part = part.strip()
        if not part:
            continue
        c = int(part)
        if c < 1 or c > 8:
            raise SystemExit("--concurrency levels must be in [1, 8]")
        levels.append(c)
    if not levels or len(set(levels)) != len(levels):
        raise SystemExit("--concurrency must be a non-empty ascending list")
    args.concurrency_levels = levels
    if args.max_concurrency is None:
        args.max_concurrency = max(levels)
    if args.corpus_commit is None:
        args.corpus_commit = git_head(NINFERPRO) or "HEAD"
    return args


def main():
    args = parse_args()
    signal.signal(signal.SIGTERM, cleanup)
    signal.signal(signal.SIGINT, cleanup)
    cfg = bi.load_profile_config(args.config)
    os.makedirs(os.path.join(bi.REPO_ROOT, "results"), exist_ok=True)
    out = args.out or os.path.join(
        bi.REPO_ROOT, "results", "infernix_frontier_" + git_short(NINFERPRO))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    requests_path = out + ".requests.jsonl"
    logpath = out + ".server.log"
    levels_dir = out + ".levels"

    runner = None
    if args.traffic == "agentic":
        runner = load_agentic_ab(args)
        args.corpus_commit = sys.modules["workload"].Corpus().commit

    # Boot flags: bench_infernix's production set, plus the machine log and
    # the profile's host tier (so the prefix-cache metric reflects production).
    extra = ["--request-log-jsonl", requests_path]
    if cfg.get("host-context-mib") is not None:
        extra += ["--host-context-mib", str(cfg["host-context-mib"])]
    if args.server_args:
        extra += shlex.split(args.server_args)
    boot_args = argparse.Namespace(**vars(args))
    boot_args.server_args = shlex.join(extra) if extra else ""
    flags = bi.server_flags(boot_args, cfg, args.experts)

    data = {
        "mode": "frontier",
        "binary": os.path.abspath(args.binary),
        "engine_commit": git_short(NINFERPRO),
        "corpus_commit": args.corpus_commit if args.traffic == "agentic"
        else None,
        "traffic": args.traffic,
        "seed": args.seed,
        "scale": args.scale,
        "seconds": args.seconds,
        "sla_tok_s": args.sla_tok_s,
        "boot": {"ready_s": None, "placement": args.experts or cfg.get("experts", ""),
                 "max_concurrency": args.max_concurrency,
                 "drafter": args.spec if args.spec is not None else (cfg.get("spec") or "none")},
        "levels": [],
        "frontier": [],
        "sla_pick": None,
    }

    lock = GpuLock(args.gpu_lock) if args.gpu_lock else None
    try:
        if lock is not None:
            lock.acquire()
            CURRENT["lock"] = lock
        # The serve appends to --request-log-jsonl; a fresh run starts a fresh
        # log, so an earlier run's events cannot leak into this one's metrics.
        with open(requests_path, "w"):
            pass
        srv = bi.Server(args.binary, flags, args.host, args.port, logpath)
        CURRENT["server"] = srv
        try:
            ready = srv.wait_ready()
        except BaseException:
            srv.stop()
            raise
        data["boot"]["ready_s"] = round(ready, 1)
        print("boot ready in {0:.1f}s, model {1}, drafter {2}".format(
            ready, srv.model_id(), data["boot"]["drafter"]))

        model = srv.model_id()
        for level in args.concurrency_levels:
            print("level C{0} ({1} traffic, {2}s budget)".format(
                level, args.traffic, args.seconds))
            if args.traffic == "agentic":
                client = run_agentic_level(
                    runner, args, level, model, levels_dir,
                    int(args.max_context or cfg.get("max-context", 262144)))
                seeds, sent, skipped = client.seeds, len(client.seeds), \
                    client.skipped
                failures = [f for f in client.failures if f]
            else:
                res = run_synthetic_level(args, level, srv.base, model)
                seeds, sent, skipped = res["seeds"], len(res["seeds"]), 0
                failures = res["failures"]
            done, starts, errors = load_request_records(requests_path, seeds)
            m = level_metrics(done, starts, errors)
            if failures:
                print("warning: {0} client-side failure(s): {1}".format(
                    len(failures), failures[:3]), file=sys.stderr)
            if m is None:
                m = {"requests_done": 0}
            row = {"concurrency": level, "traffic": args.traffic,
                   "sent": sent, "skipped": skipped,
                   "client_failures": len(failures)}
            row.update(m)
            row["proc_vram_gib"] = (
                round(bi.vram_mib(srv.pid) / 1024.0, 1)
                if bi.vram_mib(srv.pid) else None)
            data["levels"].append(row)
            data["frontier"] = frontier(data["levels"])
            data["sla_pick"] = sla_pick(data["levels"], args.sla_tok_s)
            print(json.dumps(row)[:600])
            write_outputs(out, data)      # incremental
    finally:
        if CURRENT["server"] is not None:
            CURRENT["server"].stop()
            CURRENT["server"] = None
        if CURRENT["lock"] is not None:
            CURRENT["lock"].release()
            CURRENT["lock"] = None
    write_outputs(out, data)
    return 0


if __name__ == "__main__":
    sys.exit(main())
