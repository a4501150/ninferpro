#!/usr/bin/env python3
"""
Prefix-cache resume benchmark across the RAM and Disk tiers (stdlib only).

For each configuration: boot the server, send N long unique prompts cold, then
repeat each one exactly (a resume from VRAM, RAM or the Disk tier, depending on
the configuration), stop the server (the stop spills the RAM tier to the Disk
tier when both exist), boot it again on the same directory and repeat each
prompt once more (a resume from the Disk tier, or a cold run without one).
Every number comes from the server: TTFT, cached tokens, "from disk" and decode
speed from the "req#N done" log lines, Disk bytes from GET /metrics, the
expert tiers and VRAM from GET /admin/expert-placement and nvidia-smi.

  flock ~/.cache/ninferpro/gpu.lock flock ~/.cache/ninferpro/build.lock \\
      python3 bench/t3/bench_prefix_disk.py --binary build/apps/infernix-serve \\
      --config configs/Qwen3.8-Flash-Next.toml \\
      --configs "ram:;ram+disk:--prefix-cache-disk {DISK};disk:--prefix-cache-disk {DISK} --host-context-mib 0" \\
      --out results/prefix_disk

{DISK} is replaced by a directory under --disk-root (emptied first, removed after).
Output: <out>.json and <out>.md, plus the table on stdout.
"""

import argparse
import json
import os
import random
import re
import shlex
import shutil
import signal
import sys
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_infernix as bi  # noqa: E402

DONE = re.compile(r"req#(\d+) done")


def duration_seconds(text):
    m = re.match(r"([\d.]+)\s*(ms|s|min)", text)
    if not m:
        return None
    value = float(m.group(1))
    return value / 1000.0 if m.group(2) == "ms" else value * 60.0 if m.group(2) == "min" else value


def parse_done(line):
    out = {"line": line}
    m = re.search(r"prompt ([\d,]+)", line)
    out["prompt"] = int(m.group(1).replace(",", "")) if m else None
    m = re.search(r"\| cache ([\d,]+) \(([^)]*)\)", line)
    out["cached"] = int(m.group(1).replace(",", "")) if m else None
    out["from_disk"] = bool(m and "from disk" in m.group(2))
    m = re.search(r"TTFT ([\d.]+ ?(?:ms|s|min))", line)
    out["ttft_s"] = duration_seconds(m.group(1)) if m else None
    m = re.search(r"decode ([\d.]+[kK]?) tok/s", line)
    out["decode_tps"] = bi.num(m.group(1)) if m else None
    return out


def metrics(srv):
    st, text = bi.http("GET", srv.base + "/metrics", timeout=10)
    out = {}
    if st == 200 and isinstance(text, str):
        for line in text.splitlines():
            if line.startswith("infernix_prefix_cache_disk_"):
                name, value = line.split()
                out[name[len("infernix_prefix_cache_disk_"):]] = float(value)
    return out


def prompt_text(words, seed):
    rng = random.Random(seed)
    head = "Session {0}. Read the notes below, then answer in one sentence: what is the first word?\n".format(
        uuid.UUID(int=rng.getrandbits(128)))
    return head + " ".join(rng.choice(bi.WORDS) for _ in range(words))


class Run:
    def __init__(self, args, env, name, flags, logpath):
        self.srv = bi.Server(args.binary, bi.server_flags(args, env, None) + flags,
                             args.host, args.port, logpath)
        bi.CURRENT["server"] = self.srv
        self.ready_s = self.srv.wait_ready(deadline=600)
        self.model = self.srv.model_id()
        self.offset = 0

    def done_lines(self):
        with open(self.srv.logpath, errors="replace") as f:
            f.seek(self.offset)
            data = f.read()
            self.offset = f.tell()
        return [line for line in data.splitlines() if DONE.search(line)]

    def ask(self, text, max_tokens):
        body = {"model": self.model, "messages": [{"role": "user", "content": text}],
                "max_tokens": max_tokens, "temperature": 0}
        st, resp = bi.http("POST", self.srv.base + "/v1/chat/completions", body, timeout=600)
        if st != 200:
            raise RuntimeError("request failed: {0} {1}".format(st, resp))
        for _ in range(100):
            lines = self.done_lines()
            if lines:
                return parse_done(lines[-1])
            time.sleep(0.05)
        raise RuntimeError("no req done line for a completed request")

    def placement(self):
        st, g = bi.http("GET", self.srv.base + "/admin/expert-placement", timeout=10)
        return bi.breakdown(g if isinstance(g, dict) else {}, bi.vram_mib(self.srv.pid))

    def stop(self):
        t0 = time.time()
        self.srv.stop()
        bi.CURRENT["server"] = None
        return time.time() - t0


def startup_lines(logpath):
    with open(logpath, errors="replace") as f:
        return [line.strip() for line in f if "prefix cache" in line and "|" in line and "req#" not in line][:4]


def run_config(args, env, name, spec, rows):
    disk = os.path.join(args.disk_root, "t3-" + re.sub(r"[^a-z0-9]+", "-", name.lower()))
    if os.path.exists(disk):
        shutil.rmtree(disk)
    flags = shlex.split(spec.replace("{DISK}", disk))
    prompts = [prompt_text(args.words, hash((name, i)) & 0xffffffff) for i in range(args.sessions)]
    result = {"config": name, "flags": flags, "requests": []}
    try:
        boot1 = Run(args, env, name, flags, "{0}.{1}.boot1.log".format(args.out, name))
        result["boot1_ready_s"] = round(boot1.ready_s, 1)
        result["placement"] = boot1.placement()
        for i, text in enumerate(prompts):
            for phase in ("cold", "repeat"):
                r = boot1.ask(text, args.max_tokens)
                r.update(session=i, phase=phase)
                result["requests"].append(r)
        result["metrics_boot1"] = metrics(boot1.srv)
        result["stop_s"] = round(boot1.stop(), 1)
        result["startup_boot1"] = startup_lines(boot1.srv.logpath)
        boot2 = Run(args, env, name, flags, "{0}.{1}.boot2.log".format(args.out, name))
        result["boot2_ready_s"] = round(boot2.ready_s, 1)
        for i, text in enumerate(prompts):
            r = boot2.ask(text, args.max_tokens)
            r.update(session=i, phase="after restart")
            result["requests"].append(r)
        result["metrics_boot2"] = metrics(boot2.srv)
        boot2.stop()
        result["startup_boot2"] = startup_lines(boot2.srv.logpath)
    finally:
        if bi.CURRENT["server"] is not None:
            bi.CURRENT["server"].stop()
            bi.CURRENT["server"] = None
        if os.path.exists(disk):
            shutil.rmtree(disk)
    rows.append(result)
    return result


def table(rows):
    out = ["| config | phase | prompt | cached | from disk | TTFT s (each session) | decode tok/s |",
           "|---|---|---|---|---|---|---|"]
    for row in rows:
        for phase in ("cold", "repeat", "after restart"):
            reqs = [r for r in row["requests"] if r["phase"] == phase]
            if not reqs:
                continue
            out.append("| {0} | {1} | {2} | {3} | {4} | {5} | {6} |".format(
                row["config"], phase, reqs[0]["prompt"],
                "/".join(str(r["cached"]) for r in reqs),
                "/".join("y" if r["from_disk"] else "n" for r in reqs),
                "/".join("{0:.3f}".format(r["ttft_s"]) for r in reqs),
                "/".join("{0:.0f}".format(r["decode_tps"] or 0) for r in reqs)))
    out.append("")
    out.append("| config | Disk writes during boot 1 | Disk used at boot 2 | stop s | tiers v/r/s | VRAM GiB (experts / KV / other) |")
    out.append("|---|---|---|---|---|---|")
    for row in rows:
        m1, m2, p = row.get("metrics_boot1", {}), row.get("metrics_boot2", {}), row.get("placement", {})
        out.append("| {0} | {1:.1f} MiB ({2:.0f} images, {3:.0f} blocks) | {4:.1f} MiB | {5} | {6} | {7} / {8} / {9} |".format(
            row["config"], m1.get("write_bytes_total", 0) / 2**20, m1.get("image_writes_total", 0),
            m1.get("block_writes_total", 0), m2.get("used_bytes", 0) / 2**20, row.get("stop_s"),
            p.get("tiers"), fmt(p.get("expert_vram_gib")), fmt(p.get("kv_vram_gib")), fmt(p.get("other_vram_gib"))))
    return "\n".join(out)


def fmt(v):
    return "{0:.2f}".format(v) if isinstance(v, (int, float)) else "?"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--binary", required=True)
    ap.add_argument("--config", default=None,
                    help="TOML serving profile (default: configs/GLM-5.3-Flash.toml)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8411)
    ap.add_argument("--configs", required=True, help="name:flags;name:flags ({DISK}: the Disk directory)")
    ap.add_argument("--sessions", type=int, default=3)
    ap.add_argument("--words", type=int, default=20000)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--disk-root", default=os.path.expanduser("~/.cache/ninferpro/disk-test"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--kv-dtype", default=None)
    ap.add_argument("--max-context", type=int, default=None)
    ap.add_argument("--max-concurrency", type=int, default=None)
    ap.add_argument("--ram-headroom-mib", type=int, default=16384)
    ap.add_argument("--artifact", default=None)
    ap.add_argument("--ngram-volume", default=None)
    ap.add_argument("--draft-model", default=None)
    ap.add_argument("--spec", default=None)
    ap.add_argument("--server-args", default="")
    args = ap.parse_args()
    signal.signal(signal.SIGTERM, bi.cleanup)
    signal.signal(signal.SIGINT, bi.cleanup)
    cfg = bi.load_profile_config(args.config)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    rows = []
    for item in args.configs.split(";"):
        name, _, spec = item.partition(":")
        run_config(args, cfg, name.strip(), spec.strip(), rows)
        with open(args.out + ".json", "w") as f:
            json.dump(rows, f, indent=1, default=str)
    text = table(rows)
    with open(args.out + ".md", "w") as f:
        f.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
