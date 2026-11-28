#!/usr/bin/env python3
"""
Fast, reusable Infernix benchmark (stdlib only).

Designed to keep the GPU busy: one server boot per run, placements switched
at runtime through POST /admin/expert-placement, 0.5 s readiness/move polls
(no fixed sleeps), a thread pool for the concurrent phase, and every speed
parsed from the server's own request log ("req#N done | ... | prefill X
tok/s | decode Y tok/s | <drafter> accepted a/b"), never from client timing;
the drafter name is read from the log line, so any speculative backend parses.
The script always kills its server on exit (SIGTERM handler + finally).

Usage:
  # Short-boot proof: N back-to-back boots, each started immediately after
  # the previous one exits (the async driver-reclaim window that used to
  # stop the resident warm fill early). Pass = "resident warm fill loaded
  # 24575 of 24576" on every boot with zero spill warnings.
  python3 bench/t3/bench_infernix.py --startup-only 5 \
      --binary build/apps/infernix-serve --port 8410 \
      --out results/infernix_startup

  # Placement matrix in a single boot (switches walk the placements):
  python3 bench/t3/bench_infernix.py \
      --binary build/apps/infernix-serve --port 8410 \
      --config configs/Qwen3.8-Flash-Next.toml \
      --placements "vram=100%;vram=50%,ram=50%;vram=50%,ram=0%,ssd=50%;vram=65%,ram=15%,ssd=20%;vram=100%" \
      --out results/infernix_matrix

Model, n-gram volume, drafter mode and drafter paths default to the selected
TOML profile (default configs/GLM-5.3-Flash.toml; override with --config or
--artifact / --ngram-volume / --spec / --draft-model).
Bench boots pass no --prefix-cache-disk, so nothing is persisted or restored,
unless --server-args adds it (extra server flags, e.g. "--prefix-cache-disk DIR
--host-context-mib 0").
Prefill prompts carry a unique random prefix so the prefix cache never
absorbs the prefill being measured.

Output: <out>.json (one row per placement or boot, including the raw
GET /admin/expert-placement state) and <out>.md (markdown table), plus the
table on stdout. Rows are written incrementally, so a partial run still
leaves partial results.

Run under the shared locks so nothing disturbs the measurements:
  flock ~/.cache/ninferpro/gpu.lock flock ~/.cache/ninferpro/build.lock \
      python3 bench/t3/bench_infernix.py ...
"""

import argparse
import json
import os
import random
import re
import shlex
import signal
import subprocess
import sys
import time
import tomllib
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from urllib.error import HTTPError
from urllib.request import Request, urlopen

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_CONFIG = os.path.join(REPO_ROOT, "configs", "GLM-5.3-Flash.toml")

WORDS = (
    "system network kernel memory buffer thread process signal quantum index "
    "cache latency bandwidth token vector matrix gradient cluster replica "
    "shard journal schema cursor packet route bridge socket stream window "
    "frame page block sector stripe mirror parity entropy policy schedule "
    "budget ledger quota threshold envelope snapshot boundary checkpoint "
    "consensus quorum election follower leader candidate observer probe "
    "gauge relay beacon tower lattice grid mesh fabric weave pattern "
    "cipher digest signature certificate authority vault gate keeper "
    "filter funnel pipeline conveyor carriage elevator rotor stator valve "
    "sensor actuator beacon antenna relay switch router bridge firewall "
    "prism lens mirror filter aperture shutter iris focus zoom exposure "
    "harmony melody rhythm chord tempo cadence resonance silence echo "
).split()

SHORT_PROMPT = "Explain the difference between TCP and UDP, briefly."
C6_PROMPT = (
    "Write a short technical paragraph about distributed systems topic"
)


def load_profile_config(config_path=None):
    """Read an infernix-serve --config TOML profile (top-level + [serve] overrides)."""
    path = config_path or DEFAULT_CONFIG
    if not os.path.isabs(path) and not os.path.exists(path):
        candidate = os.path.join(REPO_ROOT, "configs", path if path.endswith(".toml") else path + ".toml")
        if os.path.exists(candidate):
            path = candidate
    with open(os.path.expanduser(path), "rb") as f:
        raw = tomllib.load(f)
    cfg = {k: v for k, v in raw.items() if not isinstance(v, dict)}
    serve = raw.get("serve")
    if isinstance(serve, dict):
        cfg.update(serve)
    for key in ("model", "ngram-volume", "draft-model", "chat-template"):
        val = cfg.get(key)
        if isinstance(val, str) and val:
            cfg[key] = os.path.expanduser(val)
    return cfg


def http(method, url, body=None, timeout=15):
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = Request(url, data=data, method=method, headers=headers)
    try:
        with urlopen(req, timeout=timeout) as r:
            raw = r.read()
            try:
                return r.status, json.loads(raw)
            except Exception:
                return r.status, raw.decode(errors="replace")
    except HTTPError as e:
        return e.code, e.read().decode(errors="replace")
    except Exception as e:  # connection refused, timeout, ...
        return None, str(e)


def num(s):
    s = str(s).strip()
    if s.endswith(("k", "K")):
        return float(s[:-1]) * 1000.0
    try:
        return float(s)
    except ValueError:
        return None


def vram_mib(pid):
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory",
         "--format=csv,noheader"],
        capture_output=True, text=True,
    ).stdout
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0] == str(pid):
            m = re.match(r"([\d.]+)", parts[1])
            if m:
                return float(m.group(1))
    return None


def breakdown(g, proc_vram_mib):
    """Per-tier counts and the process VRAM split from GET /admin/expert-placement."""
    m = g.get("mechanism", {})
    gib = float(1 << 30)
    experts = m.get("vram", {}).get("bytes")
    kv = m.get("kv", {}).get("bytes")
    out = {
        "tiers": "{0}/{1}/{2}".format(m.get("vram", {}).get("experts"),
                                      m.get("ram", {}).get("experts"),
                                      m.get("ssd", {}).get("experts")),
        "ram_shadows": m.get("ram", {}).get("shadows"),
        "proc_vram_gib": proc_vram_mib / 1024.0 if proc_vram_mib else None,
        "expert_vram_gib": experts / gib if experts is not None else None,
        "kv_vram_gib": kv / gib if kv is not None else None,
        "last_switch": g.get("last_switch"),
    }
    if out["proc_vram_gib"] is not None and experts is not None and kv is not None:
        out["other_vram_gib"] = out["proc_vram_gib"] - out["expert_vram_gib"] - out["kv_vram_gib"]
    return out


def rss_mib(pid):
    try:
        with open("/proc/{0}/status".format(pid)) as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return None


class Server:
    def __init__(self, binary, flags, host, port, logpath):
        self.logfile = open(logpath, "wb")
        self.proc = subprocess.Popen(
            [binary] + flags,
            stdout=self.logfile,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        self.pid = self.proc.pid
        self.host, self.port = host, port
        self.base = "http://{0}:{1}".format(host, port)
        self.logpath = logpath

    def alive(self):
        return self.proc.poll() is None

    def tail(self, n=30):
        try:
            with open(self.logpath, errors="replace") as f:
                return "".join(f.readlines()[-n:])
        except OSError:
            return ""

    def wait_ready(self, deadline=300):
        t0 = time.time()
        while time.time() - t0 < deadline:
            if not self.alive():
                raise RuntimeError(
                    "server exited during startup:\n" + self.tail()
                )
            st, _ = http("GET", self.base + "/v1/models", timeout=2)
            if st == 200:
                return time.time() - t0
            time.sleep(0.5)
        raise TimeoutError("server not ready in {0}s:\n{1}".format(
            deadline, self.tail()))

    def model_id(self):
        st, resp = http("GET", self.base + "/v1/models", timeout=10)
        if st == 200 and isinstance(resp, dict):
            return resp["data"][0]["id"]
        raise RuntimeError("cannot read model id: {0} {1}".format(st, resp))

    def stop(self):
        if self.alive():
            try:
                os.kill(self.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:
                    os.kill(self.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.proc.wait()
        try:
            self.logfile.close()
        except OSError:
            pass


class LogParser:
    """Incremental parser for the server's request log."""

    DONE = re.compile(r"req#(\d+) done")
    STARTED = re.compile(r"req#(\d+) started")
    TS = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+)")

    def __init__(self, path):
        self.path = path
        self.off = 0
        self.reqs = {}
        self.started = {}
        self.max_id = 0

    def poll(self):
        with open(self.path, errors="replace") as f:
            f.seek(self.off)
            data = f.read()
            self.off = f.tell()
        for line in data.splitlines():
            m = self.TS.match(line)
            ts = None
            if m:
                try:
                    ts = datetime.strptime(
                        m.group(1), "%Y-%m-%d %H:%M:%S.%f")
                except ValueError:
                    ts = None
            md = self.DONE.search(line)
            if md:
                rid = int(md.group(1))
                self.max_id = max(self.max_id, rid)
                d = {"ts": ts, "line": line}
                pats = [
                    ("prompt", r"prompt (\d+)"),
                    ("output", r"output (\d+)"),
                    ("prefill_tps", r"prefill ([\d.]+[kK]?) tok/s"),
                    ("decode_tps", r"decode ([\d.]+[kK]?) tok/s"),
                ]
                for key, pat in pats:
                    mm = re.search(pat, line)
                    if mm:
                        d[key] = num(mm.group(1))
                # Speculative acceptance names whichever drafter the server ran
                # (dflash2, mtp, mixed speculation, ...): that segment reads
                # "| <name> accepted a/b (p%)". The ngram ("ngram a/b accepted,
                # r rounds") and draft-archive ("accepted, gen ...") segments
                # never carry the parenthesised percentage, so they cannot match.
                acc = re.search(
                    r"\| ([^|]+?) accepted ([\d.,]+[kK]?)/([\d.,]+[kK]?) \(", line)
                if acc:
                    d["drafter"] = acc.group(1).strip()
                    d["acc_a"] = num(acc.group(2).replace(",", ""))
                    d["acc_b"] = num(acc.group(3).replace(",", ""))
                self.reqs[rid] = d
                continue
            ms = self.STARTED.search(line)
            if ms:
                rid = int(ms.group(1))
                self.max_id = max(self.max_id, rid)
                self.started.setdefault(rid, ts)

    def wait_done(self, ids, timeout=60):
        t0 = time.time()
        while time.time() - t0 < timeout:
            self.poll()
            if all(i in self.reqs for i in ids):
                return True
            time.sleep(0.2)
        return False


def filler(approx_tokens, seed):
    rnd = random.Random(seed)
    n = int(approx_tokens)
    return " ".join(rnd.choice(WORDS) for _ in range(n))


def chat(base, model, content, max_tokens, timeout=240):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    return http("POST", base + "/v1/chat/completions", body, timeout)


def run_workload(base, model, parser, tag, ctx30k_prompt):
    """Run the four phases once; returns req ids per phase."""
    ids = {}

    b = parser.max_id
    chat(base, model, SHORT_PROMPT, 256)
    parser.wait_done(list(range(b + 1, parser.max_id + 1)))
    ids["short_c1"] = list(range(b + 1, parser.max_id + 1))

    b = parser.max_id
    chat(base, model, ctx30k_prompt, 256)
    parser.wait_done(list(range(b + 1, parser.max_id + 1)))
    ids["ctx30k_c1"] = list(range(b + 1, parser.max_id + 1))

    b = parser.max_id
    prompt = "[bench-{0}-prefill-{1}] ".format(tag, uuid.uuid4().hex[:8]) \
        + filler(29000, 11)
    chat(base, model, prompt, 4)
    parser.wait_done(list(range(b + 1, parser.max_id + 1)))
    ids["prefill30k"] = list(range(b + 1, parser.max_id + 1))

    b = parser.max_id
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = [
            ex.submit(
                chat, base, model,
                "[bench-{0}-c6-{1}-{2}] {3} {1}.".format(
                    tag, i, uuid.uuid4().hex[:8], C6_PROMPT),
                256,
            )
            for i in range(6)
        ]
        for f in futs:
            f.result()
    parser.wait_done(list(range(b + 1, parser.max_id + 1)))
    ids["c6"] = list(range(b + 1, parser.max_id + 1))
    return ids


CHAT_PROMPTS = [
    "Explain how photosynthesis works to a curious teenager.",
    "What are good habits for sleeping better? Give practical advice.",
    "Summarize the causes of the French Revolution in plain language.",
    "Describe a relaxing weekend trip to the mountains.",
    "Why is the sky blue? Explain simply.",
    "Give advice for a first job interview.",
    "Tell a short story about a lighthouse keeper.",
    "What is the difference between weather and climate?",
]
CODE_PROMPTS = [
    "Write a Python function that parses an ISO 8601 date string without libraries.",
    "Implement a thread-safe LRU cache in C++ with a mutex and a std::list.",
    "Write a Rust function that merges two sorted vectors in place.",
    "Write a SQL query for the top 3 salaries per department with window functions.",
    "Implement binary search over a rotated sorted array in Go.",
    "Write a bash script that retries a command with exponential backoff.",
    "Implement a trie in TypeScript with insert, search and startsWith.",
    "Write a CUDA kernel that computes a prefix sum over 1024 floats per block.",
]


def swap_counters(base):
    st, g = http("GET", base + "/admin/expert-placement", timeout=10)
    sw = g.get("mechanism", {}).get("swaps", {}) if isinstance(g, dict) else {}
    return sw.get("routed", 0), sw.get("hits", 0), sw.get("promotions", 0)


def run_shift(base, model, spec, n):
    """Chat then code at one placement; VRAM hit rate per request from the swap counters."""
    st, resp, body = post_placement(base, spec)
    g, sw_s = wait_move_done(base)
    rows = []
    for phase, prompts in (("chat", CHAT_PROMPTS), ("code", CODE_PROMPTS)):
        for i in range(n):
            r0, h0, p0 = swap_counters(base)
            chat(base, model, prompts[i % len(prompts)], 256)
            r1, h1, p1 = swap_counters(base)
            routed = r1 - r0
            rows.append({"phase": phase, "i": i,
                         "hit_pct": 100.0 * (h1 - h0) / routed if routed else None,
                         "promotions": p1 - p0})
    return {"placement": spec, "switch_s": round(sw_s, 1), "rows": rows}


def print_shift(shift):
    print("swap shift at {0}:".format(shift["placement"]))
    print("| phase | request | VRAM hit % | promotions |")
    print("|---|---|---|---|")
    for r in shift["rows"]:
        print("| {0} | {1} | {2} | {3} |".format(r["phase"], r["i"], fmt(r["hit_pct"]), r["promotions"]))


def phase_metric(parser, ids, key):
    vals = [parser.reqs[i][key] for i in ids
            if i in parser.reqs and key in parser.reqs[i]
            and parser.reqs[i][key] is not None]
    return sum(vals) / len(vals) if vals else None


def c6_aggregate(parser, ids):
    dones = [parser.reqs[i] for i in ids if i in parser.reqs]
    starts = [parser.started[i] for i in ids if i in parser.started]
    if not dones or not starts:
        return None
    t_end = max(d["ts"] for d in dones if d["ts"])
    t_beg = min(t for t in starts if t)
    toks = sum(int(d.get("output") or 0) for d in dones)
    dt = (t_end - t_beg).total_seconds()
    return round(toks / dt, 1) if dt > 0 else None


def disturbed(parser, warm, meas):
    """Whether the measured pass ran on a disturbed GPU or host.

    Both passes decode the same short prompt with the same drafts (greedy, so the
    acceptance counts match), and the measured one runs 4-6 % faster than the warm
    one on an undisturbed machine. A measured short decode below the warm one with
    equal acceptance means something else slowed the boot during the pass (the GPU
    also drives the desktop); its numbers are not comparable."""
    w = [parser.reqs.get(i) for i in warm["short_c1"]]
    m = [parser.reqs.get(i) for i in meas["short_c1"]]
    if not w or not m or None in w or None in m:
        return False
    w, m = w[0], m[0]
    if (w.get("acc_a"), w.get("acc_b")) != (m.get("acc_a"), m.get("acc_b")):
        return False
    wt, mt = w.get("decode_tps"), m.get("decode_tps")
    return wt is not None and mt is not None and mt < wt


def acceptance_pct(parser, ids):
    a = sum(int(parser.reqs[i]["acc_a"]) for i in ids
           if i in parser.reqs and "acc_a" in parser.reqs[i])
    b = sum(int(parser.reqs[i]["acc_b"]) for i in ids
           if i in parser.reqs and "acc_b" in parser.reqs[i])
    return round(100.0 * a / b, 1) if b else None


def spec_to_body(spec):
    body = {}
    for part in spec.split(","):
        k, _, v = part.partition("=")
        body[k.strip()] = v.strip()
    return body


def post_placement(base, spec):
    body = spec_to_body(spec)
    attempts = [body]
    trimmed = {k: v for k, v in body.items() if v not in ("0%", "0")}
    if trimmed != body:
        attempts.append(trimmed)
    if body.get("vram") in ("100%", "all"):
        attempts.append({"vram": "all"})
    last = (None, None, None)
    for a in attempts:
        st, resp = http("POST", base + "/admin/expert-placement",
                        a, timeout=60)
        if st == 200:
            return st, resp, a
        last = (st, resp, a)
    return last


def wait_move_done(base, timeout=30):
    """Poll GET until switching is false and any background fill finished.

    Real switches clear in ~1-3 s. A placement whose switching flag never
    clears (observed for no-op re-applies and demotions from the resident
    default) is given up after `timeout` so the run keeps the GPU busy; the
    row still records the raw GET state.
    """
    t0 = time.time()
    g = None
    while time.time() - t0 < timeout:
        st, g = http("GET", base + "/admin/expert-placement", timeout=5)
        if isinstance(g, dict):
            switching = g.get("switching")
            if switching is False or switching is None:
                ft, fs = g.get("fill_total"), g.get("fill_seeded")
                if ft in (None, 0) or (
                        ft is not None and fs is not None and fs >= ft):
                    return g, time.time() - t0
        time.sleep(0.5)
    return g, time.time() - t0


def find_host_expert_bytes(g):
    """Best-effort: the host/RAM expert bytes from a GET body.

    Prefers the mechanism block (ram experts x record bytes); the generic
    walk is a fallback and skips per-record fields like record_bytes.
    """
    if not isinstance(g, dict):
        return None, None
    mech = g.get("mechanism")
    if isinstance(mech, dict) and isinstance(mech.get("ram"), dict):
        ram = mech["ram"]
        n, rec = ram.get("experts"), ram.get("record_bytes")
        if isinstance(n, (int, float)) and isinstance(rec, (int, float)):
            return "mechanism.ram.experts*record_bytes", int(n * rec)
    hits = []

    def walk(node, path):
        if isinstance(node, dict):
            for k, v in node.items():
                here = ".".join(path + [k])
                # (?<!v)ram: match ram/host on the path but not "vram"
                if re.search(r"(?<!v)ram|host", here) \
                        and re.search(r"byte", k) and k != "record_bytes" \
                        and isinstance(v, (int, float)):
                    hits.append((here, v))
                walk(v, path + [k])
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, path + [str(i)])

    walk(g, [])
    if hits:
        return hits[0]
    return None, None


def fmt_bytes(v):
    if v is None:
        return "n/a"
    if v > 2 ** 30:
        return "{0:.1f}GiB".format(v / 2 ** 30)
    if v > 2 ** 20:
        return "{0:.0f}MiB".format(v / 2 ** 20)
    if v > 2 ** 10:
        return "{0:.1f}KiB".format(v / 2 ** 10)
    return str(int(v))


def fmt(v, nd=1):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return "{0:.{1}f}".format(v, nd)
    return str(v)


def write_outputs(out, rows, mode):
    with open(out + ".json", "w") as f:
        json.dump({"mode": mode, "rows": rows}, f, indent=1)
    lines = []
    if mode == "startup":
        lines += ["| boot | ready_s | engine_s | warm_fill | spill_warn |"
                  " residents | vram_MiB |",
                  "|---|---|---|---|---|---|---|"]
        for r in rows:
            lines.append(
                "| {0} | {1} | {2} | {3}/{4} | {5} | {6} | {7} |".format(
                    r["boot"], fmt(r["ready_s"]), fmt(r.get("engine_s")),
                    r.get("warm_loaded"), r.get("warm_total"),
                    r.get("spill_warnings"), r.get("residents"),
                    fmt(r.get("vram_mib"), 0)))
    else:
        lines += ["| placement | VRAM/RAM/SSD experts | switch_s | proc VRAM GiB |"
                  " experts / KV / other GiB | RSS GiB | short_C1 | ctx30k_C1 |"
                  " prefill30k | C6_agg | accept_% |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            path, hb = r.get("host_expert_path"), r.get("host_expert_bytes")
            host = fmt_bytes(hb) if hb is not None else "n/a"
            if r.get("ram_experts") is not None:
                host = "{0} experts {1}".format(r["ram_experts"], host)
            split = "{0} / {1} / {2}".format(
                fmt(r.get("expert_vram_gib")), fmt(r.get("kv_vram_gib")),
                fmt(r.get("other_vram_gib")))
            rss = r.get("rss_mib")
            lines.append(
                "| {0} | {1} | {2} | {3} | {4} | {5} | {6} | {7} | {8}"
                " | {9} | {10} |".format(
                    r["placement"], r.get("tiers", "n/a"), fmt(r.get("switch_s")),
                    fmt(r.get("proc_vram_gib")), split,
                    fmt(rss / 1024.0 if rss else None),
                    fmt(r.get("short_c1_tps"), 0),
                    fmt(r.get("ctx30k_c1_tps"), 0),
                    fmt(r.get("prefill30k_tps"), 0),
                    fmt(r.get("c6_agg_tps"), 0), fmt(r.get("accept_pct"))))
            if r.get("disturbed"):
                lines[-1] += " DISTURBED"
    table = "\n".join(lines)
    with open(out + ".md", "w") as f:
        f.write(table + "\n")
    print(table)


def server_flags(args, cfg, experts_spec):
    artifact = os.path.expanduser(args.artifact) if args.artifact else cfg["model"]
    ngram = os.path.expanduser(args.ngram_volume) if args.ngram_volume else cfg.get("ngram-volume")
    spec = getattr(args, "spec", None)
    if spec is None:
        spec = cfg.get("spec")
    draft_model = os.path.expanduser(args.draft_model) if args.draft_model else cfg.get("draft-model")
    experts = experts_spec if experts_spec is not None else cfg.get("experts")
    flags = [
        artifact,
        "--kv-dtype", args.kv_dtype or cfg.get("kv-dtype", "fp8"),
        "--max-context", str(args.max_context or cfg.get("max-context", 262144)),
        "--max-concurrency", str(args.max_concurrency or cfg.get("max-concurrency", 6)),
        "--tolerant-tool-calls",
        "--ram-headroom-mib", str(args.ram_headroom_mib),
        "--host", args.host, "--port", str(args.port),
    ]
    if spec:
        flags += ["--spec", str(spec), "--draft-tokens", str(cfg.get("draft-tokens", "auto"))]
        if draft_model:
            flags += ["--draft-model", draft_model]
    if ngram:
        flags += ["--ngram-volume", ngram]
    if cfg.get("lm-head-draft"):
        flags += ["--lm-head-draft"]
    if experts:
        flags += ["--experts", str(experts)]
    if args.server_args:
        flags += shlex.split(args.server_args)
    return flags


CURRENT = {"server": None}


def cleanup(*_):
    if CURRENT["server"] is not None:
        CURRENT["server"].stop()
        CURRENT["server"] = None
    sys.exit(130)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--binary", required=True)
    ap.add_argument("--config", default=None,
                    help="TOML serving profile (default: configs/GLM-5.3-Flash.toml)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8410)
    ap.add_argument("--placements",
                    default="vram=100%;vram=50%,ram=50%;"
                            "vram=50%,ram=0%,ssd=50%;"
                            "vram=65%,ram=15%,ssd=20%;vram=100%")
    ap.add_argument("--startup-only", type=int, default=0)
    ap.add_argument("--shift", default=None,
                    help="after the matrix: switch to this placement, then run chat "
                         "prompts followed by code prompts and report the VRAM hit "
                         "rate per request (the swapper should raise it after the shift)")
    ap.add_argument("--shift-requests", type=int, default=8)
    ap.add_argument("--experts", default=None,
                    help="boot placement spec (default: first placement,"
                         " or vram=all for --startup-only)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--kv-dtype", default=None)
    ap.add_argument("--max-context", type=int, default=None)
    ap.add_argument("--max-concurrency", type=int, default=None)
    ap.add_argument("--ram-headroom-mib", type=int, default=16384)
    ap.add_argument("--artifact", default=None)
    ap.add_argument("--ngram-volume", default=None)
    ap.add_argument("--draft-model", default=None)
    ap.add_argument("--spec", default=None,
                    help="speculative drafter for --spec (default: profile TOML's spec)")
    ap.add_argument("--server-args", default="",
                    help="extra server flags, split like a shell command line")
    args = ap.parse_args()

    signal.signal(signal.SIGTERM, cleanup)
    signal.signal(signal.SIGINT, cleanup)
    cfg = load_profile_config(args.config)
    os.makedirs(os.path.join(REPO_ROOT, "results"), exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.startup_only:
        out = args.out or os.path.join(
            REPO_ROOT, "results", "infernix_startup_" + stamp)
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        return run_startup(args, cfg, out)
    out = args.out or os.path.join(REPO_ROOT, "results", "infernix_matrix_" + stamp)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    return run_matrix(args, cfg, out)


def run_startup(args, env, out):
    rows = []
    try:
        for i in range(args.startup_only):
            logpath = "{0}.boot{1}.log".format(out, i)
            srv = Server(args.binary, server_flags(args, env, "vram=all"),
                         args.host, args.port, logpath)
            CURRENT["server"] = srv
            ready = srv.wait_ready()
            st, g = http("GET", srv.base + "/admin/expert-placement",
                         timeout=10)
            with open(logpath, errors="replace") as f:
                text = f.read()
            m = re.search(r"resident warm fill loaded (\d+) of (\d+)", text)
            me = re.search(r"engine ready \|.*?total ([\d.]+)s", text)
            spill = len(re.findall(
                r"placed in system memory and given back", text))
            row = {
                "boot": i,
                "ready_s": round(ready, 1),
                "engine_s": float(me.group(1)) if me else None,
                "warm_loaded": int(m.group(1)) if m else None,
                "warm_total": int(m.group(2)) if m else None,
                "spill_warnings": spill,
                "residents": g.get("residents") if isinstance(g, dict)
                else None,
                "vram_mib": vram_mib(srv.pid),
                "get": g if isinstance(g, dict) else str(g),
            }
            rows.append(row)
            print(json.dumps(row))
            srv.stop()          # next boot starts in the reclaim window
            CURRENT["server"] = None
    finally:
        if CURRENT["server"] is not None:
            CURRENT["server"].stop()
            CURRENT["server"] = None
    # One expert of the 24576 is never routed (every healthy boot loads
    # 24575); the spill bug showed up as a small fraction of that. Pass =
    # every boot filled >= 95% with zero spill warnings.
    ok = rows and all(
        r["warm_loaded"] is not None
        and r["warm_total"]
        and r["warm_loaded"] >= 0.95 * r["warm_total"]
        and r["spill_warnings"] == 0
        for r in rows)
    print("STARTUP_PROOF:", "PASS" if ok else "FAIL")
    write_outputs(out, rows, "startup")
    return 0 if ok else 1


def run_matrix(args, env, out):
    placements = [p.strip() for p in args.placements.split(";") if p.strip()]
    boot_spec = args.experts or placements[0]
    if boot_spec == "vram=100%":
        boot_spec = "vram=all"
    logpath = out + ".server.log"
    rows = []
    try:
        srv = Server(args.binary, server_flags(args, env, boot_spec),
                     args.host, args.port, logpath)
        CURRENT["server"] = srv
        ready = srv.wait_ready()
        model = srv.model_id()
        parser = LogParser(logpath)
        print("boot ready in {0:.1f}s, model {1}".format(ready, model))
        st, g0 = http("GET", srv.base + "/admin/expert-placement",
                      timeout=10)
        print("GET /admin/expert-placement:",
              json.dumps(g0)[:2000] if isinstance(g0, dict) else g0)
        ctx30k_prompt = "[bench-ctx30k-{0}] ".format(uuid.uuid4().hex[:8]) \
            + filler(29000, 4242)

        for spec in placements:
            row = {"placement": spec,
                   "vram_before": vram_mib(srv.pid),
                   "rss_before": rss_mib(srv.pid)}
            st, resp, body = post_placement(srv.base, spec)
            row["post_status"] = st
            row["post_body"] = body
            g, sw_s = wait_move_done(srv.base)
            row["switch_s"] = round(sw_s, 1)
            row["get"] = g if isinstance(g, dict) else str(g)
            path, hb = find_host_expert_bytes(g)
            row["host_expert_path"] = path
            row["host_expert_bytes"] = hb
            mech = g.get("mechanism") if isinstance(g, dict) else None
            row["ram_experts"] = (
                mech.get("ram", {}).get("experts")
                if isinstance(mech, dict) else None)
            row["vram_mib"] = vram_mib(srv.pid)
            row["rss_mib"] = rss_mib(srv.pid)
            if st != 200:
                row["post_resp"] = str(resp)[:400]

            warm = run_workload(srv.base, model, parser, "warm", ctx30k_prompt)
            ids = run_workload(srv.base, model, parser, "meas",
                               ctx30k_prompt)
            row["disturbed"] = disturbed(parser, warm, ids)
            if row["disturbed"]:
                print("warning: the measured pass ran slower than the warm pass"
                      " (short decode {0} < {1} tok/s at equal acceptance);"
                      " the GPU or host was busy, so this row is not comparable".format(
                          parser.reqs[ids["short_c1"][0]].get("decode_tps"),
                          parser.reqs[warm["short_c1"][0]].get("decode_tps")),
                      file=sys.stderr)
            row["short_c1_tps"] = phase_metric(
                parser, ids["short_c1"], "decode_tps")
            row["ctx30k_c1_tps"] = phase_metric(
                parser, ids["ctx30k_c1"], "decode_tps")
            row["ctx30k_prompt_toks"] = next(
                (parser.reqs[i].get("prompt") for i in ids["ctx30k_c1"]
                 if i in parser.reqs), None)
            row["prefill30k_tps"] = phase_metric(
                parser, ids["prefill30k"], "prefill_tps")
            row["prefill30k_toks"] = next(
                (parser.reqs[i].get("prompt") for i in ids["prefill30k"]
                 if i in parser.reqs), None)
            row["c6_agg_tps"] = c6_aggregate(parser, ids["c6"])
            all_ids = [i for k in ids for i in ids[k]]
            row["accept_pct"] = acceptance_pct(parser, all_ids)
            # Counts and the VRAM breakdown after the fills settled.
            st2, g2 = http("GET", srv.base + "/admin/expert-placement", timeout=10)
            if isinstance(g2, dict):
                row["get_after"] = g2
                row.update(breakdown(g2, vram_mib(srv.pid)))
            for k in ("short_c1_tps", "ctx30k_c1_tps", "prefill30k_tps",
                      "c6_agg_tps"):
                if row[k] is not None:
                    row[k] = round(row[k], 1)
            rows.append(row)
            print(json.dumps(row)[:800])
            write_outputs(out, rows, "matrix")   # incremental
        if args.shift:
            shift = run_shift(srv.base, model, args.shift, args.shift_requests)
            with open(out + ".shift.json", "w") as f:
                json.dump(shift, f, indent=1)
            print_shift(shift)
    finally:
        if CURRENT["server"] is not None:
            CURRENT["server"].stop()
            CURRENT["server"] = None
    write_outputs(out, rows, "matrix")
    return 0


if __name__ == "__main__":
    sys.exit(main())
