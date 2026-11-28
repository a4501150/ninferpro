# infernix-monitor

A first-class native server manager and live status page for `infernix-serve`: GPU, CPU, RAM and
disk telemetry, decode and prefill speed, expert placement, finished requests, server logs, and
direct model lifecycle management (Load, Switch, Unload to 0 MiB VRAM). The page is adapted from
[Basalt](https://github.com/jesdga95/basalt)'s status page (MIT, see
[LICENSE.basalt](LICENSE.basalt)).

The monitor is a separate, zero-CUDA process that runs on its own port (`127.0.0.1:8090` by
default) while `infernix-serve` binds the API port (`0.0.0.0:8000`) directly whenever a model is
loaded. Unloading exits `infernix-serve`, returning all VRAM, CUDA context, and host RAM to the
system while keeping the web UI alive across unloads, GPU faults, and server rebuilds.

| Source | Shown as |
|---|---|
| NVML (`libnvidia-ml.so.1`, loaded at run time) | GPU load, VRAM, power, temperature, PCIe link |
| `/proc/stat`, `/proc/meminfo`, `/proc/diskstats` | CPU load, RAM, disk read rate (whole disks only) |
| `configs/*.toml` (`--configs`) | Model selector dropdown, defaults, and profile details |
| `GET /metrics` once a second | decode and prefill tok/s, request gauges, prefix-cache hit, draft acceptance, expert VRAM hit |
| `GET /admin/expert-placement` | VRAM/RAM/SSD expert counts, expert and KV bytes, last placement switch |
| `<log-dir>/requests.jsonl` (enabled automatically) | Requests table and session bests |
| `<log-dir>/latest.log` (symlink to dated server log) | Logs pane |

## Build and run

`infernix-monitor` is built with the apps (Linux only; needs no CUDA at run time):

```bash
cmake --build build --target infernix-monitor infernix-serve
build/apps/infernix-monitor --daemon          # starts idle (0 VRAM) on http://127.0.0.1:8090
```

Running without arguments starts in the foreground; `--daemon` detaches into the background,
records its PID in `~/.cache/ninferpro/monitor.pid`, and writes its own log to
`logs/infernix/monitor.log`. By default it discovers `configs/*.toml` from the repository root,
selects `GLM-5.3-Flash` as the default profile, locates the sibling `infernix-serve` binary, and
starts **unloaded** (pass `--autoload` to boot the default profile immediately).

### Manager options

| Option | Default | Meaning |
|---|---|---|
| `--daemon`, `-d` | off | detach into the background and exit once listening |
| `--autoload` | off | load the default profile immediately on startup |
| `--configs DIR` | `<repo>/configs` | directory of standard `infernix-serve --config` TOML profiles |
| `--default NAME` | `GLM-5.3-Flash` | default profile name (file stem without `.toml`) |
| `--server-bin PATH` | sibling `infernix-serve` | `infernix-serve` executable spawned on Load |
| `--log-dir DIR` | `<repo>/logs/infernix` | directory for dated logs, `latest.log`, `requests.jsonl`, GPU core dumps, and `infernix-serve.pid` |
| `--gaming-vram-gb N` | `32` | VRAM GiB reserved by the Gaming placement preset |
| `--gaming-ram-gb N` | `24` | host RAM GiB reserved by the Gaming placement preset |
| `--no-request-log` | off | do not add `--request-log-jsonl` automatically |
| `--host ADDR` | `127.0.0.1` | address the web UI and control API bind to |
| `--port N` | `8090` | port the web UI and control API bind to |
| `--api-key KEY` | none | bearer token sent to `infernix-serve` |
| `--engine URL` | from profile | read-only monitor mode against an external server (disables lifecycle control unless `--configs` is also given) |
| `--log-file FILE` | `<log-dir>/latest.log` | override console log file to follow |
| `--request-log FILE` | `<log-dir>/requests.jsonl` | override request JSONL file to follow |

### CLI control subcommands

Every lifecycle and placement action works from the command line against a running manager
(`--port N` overrides `8090`):

```bash
build/apps/infernix-monitor status
build/apps/infernix-monitor load                              # loads default profile (GLM-5.3-Flash)
build/apps/infernix-monitor load Qwen3.8-Flash-Next
build/apps/infernix-monitor load GLM-5.3-Flash --experts vram=35%,ram=65%
build/apps/infernix-monitor placement --status
build/apps/infernix-monitor placement --gaming                # reserve 32 GiB VRAM / 24 GiB RAM
build/apps/infernix-monitor placement --reserve-vram-gb 40 --reserve-ram-gb 24
build/apps/infernix-monitor placement --shares vram=50%,ram=50%
build/apps/infernix-monitor placement --full                  # vram=all
build/apps/infernix-monitor unload                            # clean SIGTERM, 0 VRAM
build/apps/infernix-monitor stop [--with-server]              # stop --daemon manager
```

## Model lifecycle and crash isolation

Each `load` spawns `infernix-serve --config <profile.toml>` in its own session (`setsid`) with
closed inherited descriptors, `CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1`, a dated log file, an
atomically updated `latest.log` symlink, 30-day log/coredump pruning, and `<log-dir>/infernix-serve.pid`
so a restarted manager automatically re-adopts an already-running server. Switching from one profile
to another unloads the active server first and waits for it to exit cleanly before loading the next.

## Expert placement and Gaming preset

`Change` in the Experts panel opens VRAM and RAM sliders (the rest on SSD). While unloaded, the
sliders initialize to the selected profile's configured `experts` share and override it on Load if
modified. While loaded, slider changes preview with `?dry_run=1`, `All VRAM` requests `vram=all`,
and `Gaming` computes the largest VRAM/RAM expert counts that leave `--gaming-vram-gb` and
`--gaming-ram-gb` free before applying through `POST /admin/expert-placement`.

## Security and endpoints

The page has no authentication of its own and binds to loopback unless `--host` says otherwise.
State-changing POST requests must carry `Content-Type: application/json`, a `Host` that is an IP
literal, `localhost`, or this machine's hostname, and an `Origin` (when sent) equal to
`http://<Host>`.

- `GET /`, `/app.js`, `/app.css`: embedded web UI from [web/](web/).
- `GET /api/state`: latest telemetry, 60-sample histories, discovered profiles, active profile, and control state.
- `GET /api/logs?after=N`: log entries after sequence number `N`.
- `POST /api/placement`: accepts `{"vram": V, "ram": R, "dry_run": bool}`, `{"reserve_vram_gb": V, "reserve_ram_gb": R, "dry_run": bool}`, or `{"shares": SPEC, "dry_run": bool}`.
- `POST /api/control`: accepts `{"action": "load", "profile": NAME, "vram": V, "ram": R | "experts": SPEC}` or `{"action": "unload"}` (202 queued, 409 busy).
