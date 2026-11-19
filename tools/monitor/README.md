# infernix-monitor

A live status page for `infernix-serve`: GPU, CPU, RAM and disk readings, decode and prefill
speed, the expert placement, finished requests, and the server log. From the page you can also
switch the expert placement between VRAM, RAM and SSD, and, with `--control`, start and stop the
server. The page is adapted from [Basalt](https://github.com/jesdga95/basalt)'s status page (MIT,
see [LICENSE.basalt](LICENSE.basalt)).

The monitor is a separate process and a client of the engine's public endpoints; the engine has
no code for it. It can be started, stopped or restarted at any time without touching the server.
It reads:

| Source | Shown as |
|---|---|
| NVML (`libnvidia-ml.so.1`, loaded at run time) | GPU load, VRAM, power, temperature, PCIe link |
| `/proc/stat`, `/proc/meminfo`, `/proc/diskstats` | CPU load, RAM, disk read rate (whole disks only) |
| `GET /metrics` once a second | decode and prefill tok/s (counter deltas), request gauges, prefix-cache hit, draft acceptance, expert VRAM hit |
| `GET /admin/expert-placement` | VRAM/RAM/SSD expert counts, expert and KV bytes, the last placement switch |
| `--request-log FILE`, the engine's `--request-log-jsonl` | the Requests table and the session bests |
| `--log-file FILE`, the engine's console log or `--log-file` | the Logs pane |

Without `--request-log` or `--log-file` the page shows everything else. Both files are followed
like `tail -F`: a rotated request log, a log symlink pointed at a new file, or a truncated file
is read again from its start. On start the monitor reads the last 16 MiB of the request log and
the last 256 KiB of the console log.

## Build and run

`infernix-monitor` is built with the apps (Linux only; it needs no CUDA at run time):

```bash
cmake --build build --target infernix-monitor
build/apps/infernix-monitor --engine http://127.0.0.1:8000 \
  --request-log requests.jsonl --log-file server.log
```

Then open `http://127.0.0.1:8090`. The options:

| Option | Default | Meaning |
|---|---|---|
| `--engine URL` | `http://127.0.0.1:8000` | the server's base URL |
| `--api-key KEY` | none | the server's `--api-key`, sent as a bearer token |
| `--request-log FILE` | none | the server's `--request-log-jsonl` file |
| `--log-file FILE` | none | the server's console log or `--log-file` |
| `--control CMD` | none | the command behind Start and Stop ([Start and Stop](#start-and-stop)); without it the page has neither |
| `--host ADDR` | `127.0.0.1` | address the page is served on |
| `--port N` | `8090` | port the page is served on |

The page has no authentication of its own; it binds to loopback unless `--host` says otherwise.
Requests that change the server must come from the page itself: a JSON body, an `Origin` (when
sent) equal to the page's, and a `Host` that is an IP address, `localhost` or this machine's name,
so another web site open in the same browser cannot switch the placement or stop the server.

## Expert placement

`Change` in the Experts panel opens two sliders: the VRAM and RAM shares of the routed experts,
whole percents, the rest on SSD. They follow the server's placement until moved. While the server
runs, a move previews the shares with the engine's dry run (counts per tier, expert bytes, VRAM
left free, whether it fits) and Apply switches to them through `POST /admin/expert-placement`;
100 % VRAM is the engine's resident placement (`vram=all`). The engine places exactly what the
shares say; choosing them, such as leaving room for a game, stays with you.

## Start and Stop

`--control CMD` names an executable the monitor runs, one action at a time, as

- `CMD start [EXPERTS]`: start the server and exit 0 once it answers; `EXPERTS` is an `--experts`
  spec (`vram=80%,ram=20%,ssd=0%`), passed when the placement editor is open, and the server's
  configured placement applies without it;
- `CMD stop`: stop the server and exit 0 once it is gone.

Its stdout and stderr go to the Logs pane, and its last line is the outcome shown in the top bar.
It runs in a process group of its own with no other open file of the monitor's (the server it
starts must not hold the page's port); a start is killed after 420 s and a stop after 120 s.
Start is offered while the server does not answer, Stop while it does. The command owns how the
server is launched (binary, artifact, flags, log); super-quant's `scripts/infernix_control.sh`,
which `make monitor` passes, runs its `serve_infernix.sh` and `stop_infernix.sh`.

## Endpoints

- `GET /` with `app.js` and `app.css`: the page. The files in [web/](web/) are embedded in the
  binary at build time.
- `GET /api/state`: one JSON object with the latest readings and 60-sample histories.
- `GET /api/logs?after=N`: log lines after sequence number `N`, and the last sequence number.
- `POST /api/placement` with `{"vram": V, "ram": R, "dry_run": bool}` (whole percents): passed on
  to `POST /admin/expert-placement`; the engine's status and body come back.
- `POST /api/control` with `{"action": "start", "vram": V, "ram": R}` (without `vram`, the
  configured placement) or `{"action": "stop"}`: queues the action (202 with the control state;
  409 while another action runs).

## Differences from Basalt's page

Basalt's per-slot table needs live per-lane figures that `infernix-serve` does not publish; the
Engine tab shows the request gauges and the cache and acceptance rates instead. The Experts
panel shows the placement by tier rather than one request's lookups. The page uses system fonts.
