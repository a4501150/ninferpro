# infernix-monitor

A live status page for a running `infernix-serve`: GPU, CPU, RAM and disk readings, decode and
prefill speed, the expert placement, finished requests, and the server log. The page is adapted
from [Basalt](https://github.com/jesdga95/basalt)'s status page (MIT, see
[LICENSE.basalt](LICENSE.basalt)).

The monitor is a separate process that only reads from the engine, so it can be started, stopped
or restarted at any time without touching the server. It reads:

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
| `--host ADDR` | `127.0.0.1` | address the page is served on |
| `--port N` | `8090` | port the page is served on |

The page has no authentication of its own; it binds to loopback unless `--host` says otherwise.

## Endpoints

- `GET /` with `app.js` and `app.css`: the page. The files in [web/](web/) are embedded in the
  binary at build time.
- `GET /api/state`: one JSON object with the latest readings and 60-sample histories.
- `GET /api/logs?after=N`: log lines after sequence number `N`, and the last sequence number.

## Differences from Basalt's page

Basalt's per-slot table needs live per-lane figures that `infernix-serve` does not publish; the
Engine tab shows the request gauges and the cache and acceptance rates instead. The Experts
panel shows the placement by tier rather than one request's lookups. The page uses system fonts.
