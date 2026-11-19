// Adapted from Basalt's status page (https://github.com/jesdga95/basalt, serve/web/app.js),
// Copyright (c) 2026 Jesús Gulfo, MIT License; see tools/monitor/LICENSE.basalt.
"use strict";
const $ = s => document.querySelector(s);
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const num = v => typeof v === "number" && Number.isFinite(v);
const fmt = (v, d = 0) => num(v) ? v.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d }) : "-";
const p2 = n => String(n).padStart(2, "0");
const clock = ms => { const d = new Date(ms); return `${p2(d.getHours())}:${p2(d.getMinutes())}:${p2(d.getSeconds())}`; };
const gib = n => num(n) ? (n / 1024 ** 3).toFixed(1) : "-";
const pct = (f, d = 0) => num(f) ? `${(f * 100).toFixed(d)}%` : "-";
const ESC = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ESC[c]);
const drawn = new WeakMap();
const html = (el, h) => { if (drawn.get(el) !== h) { el.innerHTML = h; drawn.set(el, h); } };   // skip unchanged redraws
const text = (el, v) => { if (el.textContent !== v) el.textContent = v; };
const HIST = 60;
const kt = n => n >= 100000 ? `${Math.round(n / 1000)}k` : n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(n);   // 41.2k, 262k
const tick = n => n >= 1000 ? `${+(n / 1000).toFixed(1)}k` : String(Math.round(n));
const shortGpu = s => String(s || "").replace(/^NVIDIA\s+/i, "").replace(/^GeForce\s+/i, "");
const shortCpu = s => String(s || "").replace(/\((R|TM)\)/gi, "").replace(/\s+\d+-Core Processor$/i, "")
  .replace(/\s+CPU\s+@.*$/i, "").replace(/\s+Processor$/i, "").replace(/\s+/g, " ").trim();
const dur = s => !num(s) ? "-" : s >= 86400 ? `${Math.floor(s / 86400)}d ${Math.floor(s % 86400 / 3600)}h`
  : s >= 3600 ? `${Math.floor(s / 3600)}h ${Math.floor(s % 3600 / 60)}m` : `${Math.floor(s / 60)}m ${p2(Math.floor(s % 60))}s`;

let M = null;          // the last /api/state reading

async function get(path) {
  const r = await fetch(path, { cache: "no-store" });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}
// a change to the server; the error carries the monitor's or the engine's message
async function post(path, body) {
  const r = await fetch(path, { method: "POST", cache: "no-store", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  let j = {};
  try { j = await r.json(); } catch (e) { /* no body */ }
  if (!r.ok) throw new Error(j.error || `HTTP ${r.status}`);
  return j;
}

/* ---- series and charts ---- */
// the last HIST samples, padded at the start with null (no line) so the newest sample sits at the right edge
const tail = a => { const v = (a || []).slice(-HIST).map(x => num(x) ? x : null); while (v.length < HIST) v.unshift(null); return v; };
const stats = a => {
  const v = a.filter(num);
  return v.length ? { min: Math.min(...v), max: Math.max(...v), avg: v.reduce((t, x) => t + x, 0) / v.length, last: v[v.length - 1] } : null;
};
const STEPS = [10, 20, 50, 100, 120, 150, 160, 200, 250, 300, 400, 500, 600, 800, 1000, 1200, 1500, 2000, 3000, 4000, 5000,
  8000, 10000, 16000, 20000, 32000, 50000];
const nice = v => STEPS.find(s => s >= v * 1.15) || Math.ceil(v * 1.15 / 10000) * 10000;
// polyline points in a 0..W x 0..100 box; a null sample breaks the line
function lines(a, y, W = 100) {
  const out = []; let cur = [];
  a.forEach((v, i) => {
    if (v == null) { if (cur.length) out.push(cur); cur = []; return; }
    cur.push(`${(i / (a.length - 1) * W).toFixed(2)},${y(v).toFixed(2)}`);
  });
  if (cur.length) out.push(cur);
  return out.map(c => c.length === 1 ? [c[0], c[0].replace(/^[\d.]+/, x => String(+x + .01))] : c);
}
const poly = (pts, attrs) => `<polyline points="${pts.join(" ")}" fill="none" vector-effect="non-scaling-stroke" stroke-linejoin="round" ${attrs}/>`;
const area = (pts, fill) => `<polygon points="${pts[0].split(",")[0]},100 ${pts.join(" ")} ${pts[pts.length - 1].split(",")[0]},100" fill="${fill}"/>`;
const baseLine = `<line x1="0" y1="100" x2="100" y2="100" stroke="var(--line)" stroke-width="1" vector-effect="non-scaling-stroke"/>`;
let gid = 0;
// a sidebar chart: filled line between lo and hi; `limit` draws a dashed line at the top
function sbChart(a, lo, hi, accent, limit) {
  const id = `sbg${++gid}`, tint = accent ? "255,122,61" : "154,164,173";
  const segs = lines(a, v => clamp(100 - (v - lo) / (hi - lo) * 100, 1, 99));
  return `<div class="sb-ch"><svg viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true"><defs><linearGradient id="${id}" x1="0" y1="0" x2="0" y2="1">`
    + `<stop offset="0" stop-color="rgba(${tint},.13)"/><stop offset="1" stop-color="rgba(${tint},.02)"/></linearGradient></defs>${baseLine}`
    + (limit ? `<line x1="0" y1="1" x2="100" y2="1" stroke="var(--basalt-500)" stroke-width="1" stroke-dasharray="2 3" vector-effect="non-scaling-stroke"/>` : "")
    + segs.map(p => area(p, `url(#${id})`) + poly(p, `stroke="${accent ? "var(--accent)" : "var(--fg-2)"}" stroke-width="1.5"`)).join("")
    + `</svg></div>`;
}
const gy = t => t ? 100 - t * 92 - 4 : 4;             // gridline height for a fraction of the axis
const grid = fr => fr.map(t => `<line x1="0" x2="100" y1="${gy(t)}" y2="${gy(t)}" stroke="var(--line-soft)" vector-effect="non-scaling-stroke"/>`).join("");
// small tick labels at the right edge, at the top and middle gridlines
const ticks = max => [[4, max], [50, max / 2]].map(t => `<span class="tk" style="top:${t[0]}%">${tick(t[1])}</span>`).join("");

/* ---- top bar ---- */
function setState(s, label) { $("#state").dataset.s = s; text($("#state"), label); }

/* ---- sidebar ---- */
const sbRow = (k, v, u, st) => `<div class="sb-row"><span class="k">${k}</span>${st ? `<span class="st">${st}</span>` : ""}<span class="num">${v}${u ? `<span class="u">${u}</span>` : ""}</span></div>`;
const bar = f => `<div class="bar4"><i style="width:${(clamp(f || 0, 0, 1) * 100).toFixed(1)}%"></i></div>`;
const r0 = v => num(v) ? String(Math.round(v)) : "-";
function sbLoad(h, now, label = "Load") {
  const s = stats(h);
  return `<div class="sb-met grow">${sbRow(label, num(now) ? `${Math.round(now)}%` : "-", "", s ? `Min ${Math.round(s.min)} · Avg ${Math.round(s.avg)}%` : "")}${sbChart(h, 0, 100, true)}</div>`;
}
function sbPower(h, now, lim) {
  const s = stats(h), top = num(lim) && lim > 0 ? lim : Math.max(1, s ? s.max : 1);
  return `<div class="sb-met grow">${sbRow("Power", `${r0(now)}${num(lim) ? ` / ${Math.round(lim)}` : ""}`, "W", s ? `Avg ${Math.round(s.avg)} W` : "")}${sbChart(h, 0, top, false, num(lim))}</div>`;
}
function sbTemp(h, now) {
  const s = stats(h), lo = Math.min(40, s ? Math.floor(s.min) - 5 : 40), hi = Math.max(85, s ? Math.ceil(s.max) + 5 : 85);
  return `<div class="sb-met grow">${sbRow("Temp", r0(now), "°C", s ? `Min ${Math.round(s.min)} · Max ${Math.round(s.max)} °C` : "")}${sbChart(h, lo, hi, false)}</div>`;
}
const link = g => num(g.pcie_gen) ? `Gen${g.pcie_gen}${num(g.pcie_width) ? ` x${g.pcie_width}` : ""}` : "";
const vramRow = g => `<div class="sb-met">${sbRow("VRAM", num(g.mem_used) ? `${gib(g.mem_used)} / ${gib(g.mem_total)}` : "-", "GiB")}${bar(num(g.mem_total) ? g.mem_used / g.mem_total : 0)}</div>`;
function sbGpu(g, h, name, alone) {
  const right = alone ? `<span class="num">${esc(link(g))}</span>`
    : `<span class="num">${num(g.temp) ? `<b>${Math.round(g.temp)}</b><span class="u">°C</span>` : ""}${num(g.temp) && link(g) ? " · " : ""}${esc(link(g))}</span>`;
  return `<div class="sb-gpu"><div class="sb-gt"><span>${esc(name)}</span>${right}</div>${vramRow(g)}`
    + sbLoad(tail(h.util), g.util) + sbPower(tail(h.power), g.power, g.power_limit) + (alone ? sbTemp(tail(h.temp), g.temp) : "") + `</div>`;
}
function drawSide(hw, st, hist, pl, eng) {
  gid = 0;
  const G = hw.gpus || [], H = Object.fromEntries((hist.gpus || []).map(x => [x.index, x]));
  const named = Object.fromEntries((st.gpus || []).map(g => [g.index, shortGpu(g.name)]));
  const title = g => `GPU ${g.index} · ${named[g.index] || "GPU"}`;
  html($("#gpus"), !G.length ? `<div class="sb-none">No GPU readings (NVML unavailable).</div>`
    : G.map(g => sbGpu(g, H[g.index] || {}, G.length === 1 ? named[g.index] || title(g) : title(g), G.length === 1)).join(""));

  const cpu = shortCpu(st.cpu_name);
  text($("#cpu-name"), cpu); $("#cpu-name").title = st.cpu_name || "";
  html($("#sys"), sbRow("CPU", num(hw.cpu) ? `${Math.round(hw.cpu)}%` : "-")
    + `<div class="sb-met">${sbRow("RAM", num(hw.ram_total) ? `${gib(hw.ram_used)} / ${gib(hw.ram_total)}` : "-", "GiB")}${bar(num(hw.ram_total) ? hw.ram_used / hw.ram_total : 0)}</div>`
    + sbRow("Disk read", num(hw.disk_read_mb) ? hw.disk_read_mb.toFixed(1) : "-", "MB/s")
    + (st.driver ? `<div class="sb-foot">Driver ${esc(st.driver)}${st.cuda ? ` · CUDA ${esc(st.cuda)}` : ""}</div>` : ""));

  // where the routed experts live, by count, and how often a routed lookup found its expert in VRAM
  const m = pl && pl.mechanism;
  if (!m || !m.total_experts) {
    html($("#experts"), `<div class="sb-none flat">${eng.reachable ? "This model has no expert placement." : "-"}</div>`);
    return;
  }
  const total = m.total_experts;
  const tiers = [["VRAM", "var(--fg)", m.vram.experts], ["RAM", "var(--fg-3)", m.ram.experts], ["SSD", "var(--basalt-500)", m.ssd.experts]];
  const win = (eng.window || {}).expert_hit, all = (eng.total || {}).expert_hit;
  html($("#experts"), `<div class="sb-stack" aria-hidden="true">${tiers.map(t => `<i style="width:${((t[2] || 0) / total * 100).toFixed(2)}%;background:${t[1]}"></i>`).join("")}</div>`
    + `<div class="sb-leg">${tiers.map(t => `<div style="--c:${t[1]}"><span>${t[0]}</span><b>${pct((t[2] || 0) / total, (t[2] || 0) / total < 1 ? 1 : 0)}</b></div>`).join("")}</div>`
    + `<div class="sb-body flat">${sbRow("VRAM hit", pct(num(win) ? win : all, 1), "", num(win) ? "Last 60 s" : num(all) ? "Since start" : "")}`
    + sbRow("Experts", `${fmt(m.vram.experts)} · ${fmt(m.ram.experts)} · ${fmt(m.ssd.experts)}`, "", `of ${fmt(total)}`)
    + sbRow("Expert VRAM", gib(m.vram.bytes), "GiB") + sbRow("KV cache", gib(m.kv.bytes), "GiB")
    + (pl.switching ? `<div class="sb-foot warn">Placement switch in progress</div>`
      : pl.last_switch && pl.last_switch.message ? `<div class="sb-foot${pl.last_switch.ok ? "" : " err"}" title="${esc(pl.last_switch.message)}">Last switch: ${esc(pl.last_switch.message)}</div>` : "")
    + `</div>`);
}

/* ---- Speed ---- */
function drawHero(live, hist, reqs, best, eng) {
  const generating = live.state === "generating", reading = live.state === "reading";
  // the live rate while generating, else the newest finished request's
  const rd = reqs.find(x => x.decode_s > 0 && x.output_tokens > 0), last = rd ? rd.output_tokens / rd.decode_s : null;
  const liveDec = generating && num(live.tok_s) && live.tok_s > 0, dec = liveDec ? live.tok_s : last;
  text($("#dec-k"), liveDec ? "Decode · All requests" : last != null ? "Decode · Last request" : "Decode");
  html($("#dec-v"), num(dec) ? `${dec.toFixed(1)}<small>tok/s</small>` : "-");
  $("#dec-v").classList.toggle("idle", !liveDec);
  const da = tail(hist.tok_s);
  const max = nice(Math.max(40, ...da.filter(num)));
  const y = v => 100 - 4 - Math.min(v, max) / max * 92;
  html($("#dec-chart"), `<svg viewBox="0 0 100 100" preserveAspectRatio="none" role="img" aria-label="Decode tokens per second, last 60 seconds">${grid([0, .25, .5, .75])}`
    + lines(da, y).map(p => poly(p, `stroke="var(--accent)" stroke-width="1.7"`)).join("") + `</svg>${ticks(max)}`);

  // the live read rate while reading, else the newest finished request's that read part of its prompt
  const rp = reqs.find(x => x.prefill_s > 0 && x.read_tokens > 0), lastPre = rp ? rp.read_tokens / rp.prefill_s : null;
  const livePre = reading && num(live.prefill_tok_s) && live.prefill_tok_s > 0;
  const pre = livePre ? live.prefill_tok_s : lastPre;
  text($("#pre-k"), livePre ? "Prefill" : lastPre != null ? "Prefill · Last request" : "Prefill");
  html($("#pre-v"), num(pre) ? `${fmt(pre)}<small>tok/s</small>` : "-");
  const pa = tail(hist.prefill_tok_s);
  const pmax = nice(Math.max(1000, ...pa.filter(num)));
  html($("#pre-chart"), `<svg viewBox="0 0 100 100" preserveAspectRatio="none" role="img" aria-label="Prefill tokens per second, last 60 seconds">${grid([0, .5])}`
    + lines(pa, v => 100 - 4 - Math.min(v, pmax) / pmax * 92).map(p => poly(p, `stroke="var(--fg)" stroke-width="1.6"`)).join("") + `</svg>${ticks(pmax)}`);
  const r = reqs[0];
  html($("#pre-last"), r && num(r.prompt_tokens) ? `<b>${fmt(r.prompt_tokens)}</b> tok${r.prefill_s > 0 ? ` in <b>${r.prefill_s.toFixed(2)}</b> s` : ""}` : "-");
  // the last request's own hit when there is one, else the engine's over the last minute
  const hit = r && r.prompt_tokens > 0 ? r.reused / r.prompt_tokens : (eng.window || {}).prefix_hit;
  text($("#pre-hit"), pct(hit));

  const bd = best.decode, bp = best.prefill, lanes = live.max_concurrency;
  html($("#foot-dec"), `${fmt(live.running)} of ${fmt(lanes)} lanes busy${live.waiting > 0 ? ` · ${fmt(live.waiting)} waiting` : ""} · Best <b>${bd ? fmt(bd.tok_s, 1) : "-"}</b> tok/s`);
  html($("#foot-pre"), `Best <b>${bp ? fmt(bp.tok_s) : "-"}</b> tok/s`);
  $("#foot-dec").title = bd ? `Fastest decode of a whole request on this server: req#${bd.id}, ${fmt(bd.tokens)} tokens, ${clock(bd.time * 1000)}` : "";
  $("#foot-pre").title = bp ? `Fastest prompt read on this server: req#${bp.id}, ${fmt(bp.tokens)} tokens read, ${clock(bp.time * 1000)}` : "";
}

/* ---- main card: Requests | Engine ---- */
let tab = sessionStorage.getItem("infernix-monitor.tab") || "req";
$("#tabs").addEventListener("click", e => {
  const b = e.target.closest("button[data-t]"); if (!b) return;
  tab = b.dataset.t; try { sessionStorage.setItem("infernix-monitor.tab", tab); } catch (err) { /* private mode */ }
  draw();
});
function requestsPane(reqs, sources) {
  if (!sources.request_log) return `<div class="empty">Start the monitor with --request-log FILE (the engine's --request-log-jsonl) to list finished requests.</div>`;
  if (!reqs.length) return `<div class="empty">No finished requests yet.</div>`;
  return `<table><thead><tr><th>Time</th><th>ID</th><th>API</th><th class="r">Prompt</th><th class="r">Hit</th><th class="r">Output</th>`
    + `<th class="r">TTFT</th><th class="r">Decode</th><th class="r">Accept</th><th class="r">Duration</th></tr></thead><tbody>`
    + reqs.map(r => {
      // a request that did not simply stop is marked on its id, with the reason
      const odd = r.finish && r.finish !== "stop" ? (r.finish === "error" ? "err" : "warn") : "";
      const rate = r.decode_s > 0 && r.output_tokens > 0 ? (r.output_tokens / r.decode_s).toFixed(1) : "-";
      return `<tr><td>${clock(r.time * 1000)}</td><td class="${odd}">req#${esc(r.id)}${odd ? ` · ${esc(String(r.finish).replace(/_/g, " "))}` : ""}</td>`
        + `<td class="dim">${esc(r.protocol || "-")}</td><td class="r">${fmt(r.prompt_tokens)}</td>`
        + `<td class="r dim">${r.prompt_tokens > 0 ? pct(r.reused / r.prompt_tokens) : "-"}</td>`
        + `<td class="r">${fmt(r.output_tokens)}</td><td class="r">${r.ttft_s > 0 ? `${r.ttft_s.toFixed(2)} s` : "-"}</td>`
        + `<td class="r">${rate}</td><td class="r dim">${r.drafted > 20 ? pct(r.accepted / r.drafted) : "-"}</td>`
        + `<td class="r">${num(r.duration_s) ? `${r.duration_s.toFixed(1)} s` : "-"}</td></tr>`;
    }).join("") + `</tbody></table>`;
}
function enginePane(live, eng) {
  const w = eng.window || {}, t = eng.total || {};
  const row = (k, v, note) => `<tr><td>${k}</td><td class="r">${v}</td><td class="dim">${note || ""}</td></tr>`;
  return `<table><thead><tr><th>Engine</th><th class="r">Now</th><th></th></tr></thead><tbody>`
    + row("Running", `${fmt(live.running)} / ${fmt(live.max_concurrency)}`, "resident requests / lanes")
    + row("Prefilling", fmt(live.prefilling)) + row("Decode-ready", fmt(live.decode_ready))
    + row("Waiting", fmt(live.waiting), "for initial admission") + row("Paused", fmt(live.paused), "waiting for recovery")
    + row("Max context", num(live.max_context) ? kt(live.max_context) : "-", "tokens per request")
    + `</tbody></table><table><thead><tr><th>Rates</th><th class="r">Last 60 s</th><th class="r">Since start</th></tr></thead><tbody>`
    + `<tr><td>Prefix-cache hit</td><td class="r">${pct(w.prefix_hit)}</td><td class="r">${pct(t.prefix_hit)}</td></tr>`
    + `<tr><td>Draft acceptance${eng.spec_backend ? ` (${esc(eng.spec_backend)})` : ""}</td><td class="r">${pct(w.accept)}</td><td class="r">${pct(t.accept)}</td></tr>`
    + `<tr><td>Expert VRAM hit</td><td class="r">${pct(w.expert_hit, 1)}</td><td class="r">${pct(t.expert_hit, 1)}</td></tr>`
    + `<tr><td>Tokens decoded</td><td class="r"></td><td class="r">${num(t.decode_tokens) ? kt(t.decode_tokens) : "-"}</td></tr>`
    + `<tr><td>Tokens prefilled</td><td class="r"></td><td class="r">${num(t.prefill_tokens) ? kt(t.prefill_tokens) : "-"}</td></tr>`
    + `</tbody></table><div class="qsum"><span>Up <b>${dur(eng.uptime_s)}</b></span><span>${esc(eng.url || "")}</span></div>`;
}
function drawMain(live, eng, reqs, sources) {
  const T = [["req", "Requests"], ["engine", `Engine${live.waiting > 0 ? ` (${live.waiting} waiting)` : ""}`]];
  const cur = T.some(t => t[0] === tab) ? tab : T[0][0];
  html($("#tabs"), T.map(t => `<button type="button" role="tab" data-t="${t[0]}" aria-selected="${t[0] === cur}">${t[1]}</button>`).join(""));
  html($("#pane"), cur === "req" ? requestsPane(reqs, sources) : enginePane(live, eng));
}

/* ---- placement: VRAM and RAM shares of the experts, the rest on SSD ---- */
// The editor opens from the Experts header. Its sliders follow the server's placement until moved.
// While the server runs, a move previews the shares (dry run) and Apply switches them; while it
// is stopped, Start starts with them (with the editor closed, with the configured placement).
const P = { open: false, dirty: false, applied: 0, seq: 0, timer: 0, err: false };
const plV = $("#pl-vram"), plR = $("#pl-ram");
const shares = () => ({ vram: +plV.value, ram: +plR.value });
function setShares(vram, ram) {
  plV.value = vram; plR.max = 100 - vram; plR.value = Math.min(ram, 100 - vram);
  const s = shares();
  text($("#pl-vram-v"), `${s.vram}%`); text($("#pl-ram-v"), `${s.ram}%`); text($("#pl-ssd-v"), `${100 - s.vram - s.ram}%`);
}
function note(msg, err) { P.err = !!err; text($("#pl-note"), msg); $("#pl-note").classList.toggle("err", P.err); }
const placementLive = () => M && M.engine && M.engine.reachable && M.placement && M.placement.supported;
function preview() {
  clearTimeout(P.timer);
  if (!placementLive()) { note(M && M.control && M.control.enabled ? "Start uses these shares." : ""); return; }
  const seq = ++P.seq;
  P.timer = setTimeout(async () => {
    try {
      const r = await post("api/placement", { ...shares(), dry_run: true });
      if (seq !== P.seq) return;
      const c = r.placement || {}, d = r.predicted || {};
      const problem = !d.fits ? "does not fit in VRAM" : !d.ram_fits ? "more RAM than is available" : "";
      note(`${fmt(c.vram)} · ${fmt(c.ram)} · ${fmt(c.ssd)} experts; VRAM ${gib(d.vram_expert_bytes)} GiB, ${gib(d.vram_free_after)} free after; RAM ${gib(d.ram_expert_bytes)} GiB${problem ? ` — ${problem}` : ""}`, !!problem);
    } catch (e) { if (seq === P.seq) note(e.message, true); }
  }, 250);
}
function moved() { P.dirty = true; setShares(+plV.value, +plR.value); preview(); drawPlace(); }
plV.addEventListener("input", moved);
plR.addEventListener("input", moved);
$("#pl-all").addEventListener("click", () => { setShares(100, 0); moved(); });
// back to following the server; a preview still in flight is dropped
function resetEditor() { P.dirty = false; P.seq++; note(""); }
$("#pl-reset").addEventListener("click", () => { resetEditor(); draw(); });
$("#pl-toggle").addEventListener("click", () => {
  P.open = !P.open;
  if (!P.open && !P.applied) resetEditor();
  draw();
});
$("#pl-apply").addEventListener("click", async () => {
  $("#pl-apply").disabled = true;
  try {
    await post("api/placement", shares());
    P.applied = Date.now(); note("Switching…");
  } catch (e) { note(e.message, true); }
  drawPlace();
});
function drawPlace() {
  const pl = M && M.placement, ctl = (M && M.control) || {}, live = placementLive();
  const offline = !(M && M.engine && M.engine.reachable), usable = live || (ctl.enabled && offline);
  $("#pl-toggle").hidden = !usable;
  text($("#pl-toggle"), P.open ? "Close" : "Change");
  $("#place").hidden = !(usable && P.open);
  // an applied switch keeps the sliders until the engine reports it done
  if (P.applied && live && !pl.switching && Date.now() - P.applied > 2500) {
    P.applied = 0; P.dirty = false;
    const ls = pl.last_switch || {};
    note(ls.ok === false ? ls.message || "The switch failed." : "", ls.ok === false);
  }
  if (!P.dirty && live) {
    const m = pl.mechanism, t = m.total_experts;
    setShares(Math.max(1, Math.round(m.vram.experts / t * 100)), Math.round(m.ram.experts / t * 100));
  }
  if (!P.dirty && !live && !P.err) note(ctl.enabled && offline ? "Start uses these shares." : "");
  $("#pl-apply").hidden = !live;
  $("#pl-apply").disabled = !P.dirty || !!P.applied || !!(pl && pl.switching);
  $("#pl-reset").hidden = !live;
  $("#pl-reset").disabled = !P.dirty;
}

/* ---- Start and Stop ---- */
function drawControl() {
  const ctl = (M && M.control) || {};
  $("#ctl").hidden = !ctl.enabled;
  if (!ctl.enabled) return;
  const up = !!(M.engine && M.engine.reachable), busy = !!ctl.busy || (M.live || {}).state === "starting";
  $("#ctl-start").disabled = busy || up;
  $("#ctl-stop").disabled = busy || !up;
  const last = ctl.last, msg = $("#ctl-msg");
  const show = last && (!last.ok || Date.now() / 1000 - last.time < 60);
  text(msg, busy ? "" : show ? `${last.action}: ${last.message || (last.ok ? "done" : "failed")}` : "");
  msg.title = show ? last.message || "" : "";
  msg.classList.toggle("err", !!(show && !last.ok));
}
async function control(body) {
  try { M.control = await post("api/control", body); }
  catch (e) { M.control = { ...M.control, last: { action: body.action, ok: false, message: e.message, time: Date.now() / 1000 } }; }
  draw();
}
$("#ctl-start").addEventListener("click", () => control(P.open ? { action: "start", ...shares() } : { action: "start" }));
$("#ctl-stop").addEventListener("click", () => {
  if (confirm("Stop the server? Requests in flight are cut off.")) control({ action: "stop" });
});

/* ---- the whole page ---- */
const STATE = { offline: "Offline", starting: "Starting", loading: "Loading", stopping: "Stopping", idle: "Idle",
  reading: "Reading", generating: "Generating" };
function draw() {
  if (!M) return;
  const live = M.live || {}, eng = M.engine || {}, hist = M.history || {}, reqs = M.requests || [];
  const s = live.state || "offline";
  setState(s, `${STATE[s] || s}${live.waiting > 0 ? ` · ${live.waiting} waiting` : ""}`);
  text($("#model"), eng.model || "");
  drawSide(M.hardware || {}, M.hardware_static || {}, hist, M.placement, eng);
  drawHero(live, hist, reqs, M.best || {}, eng);
  drawMain(live, eng, reqs, M.sources || {});
  drawPlace();
  drawControl();
}

/* ---- logs ---- */
const logEl = $("#log"), MAXLOG = 400;
let logCount = 0, follow = true;
function lineEl(e) {
  const d = document.createElement("div");
  if (e.lvl !== "info") d.className = e.lvl;
  const t = document.createElement("i"); t.textContent = clock(e.t * 1000);
  d.append(t, `  ${e.msg}`);
  return d;
}
function append(lines) {
  logEl.querySelector(".empty")?.remove();
  logEl.append(...lines.map(lineEl));
  while (logEl.childElementCount > MAXLOG) logEl.firstChild.remove();
  logCount += lines.length; text($("#log-count"), `${fmt(logCount)} lines`);
  if (follow) logEl.scrollTop = logEl.scrollHeight;
}
logEl.addEventListener("scroll", () => { follow = logEl.scrollHeight - logEl.scrollTop - logEl.clientHeight < 12; });
function clearLog(note) {
  logCount = 0; text($("#log-count"), "");
  const d = document.createElement("div"); d.className = "empty"; d.textContent = note; logEl.replaceChildren(d);
}

/* ---- polling ---- */
let lastLog = 0;
async function pollState() {
  try {
    M = await get("api/state"); draw();
  } catch (e) {
    M = null; setState("offline", "Monitor offline"); $("#ctl").hidden = true;
  }
  setTimeout(pollState, 1000);
}
async function pollLog() {
  try {
    const r = await get(`api/logs?after=${lastLog}`);
    if (r.last < lastLog) { lastLog = 0; clearLog("The monitor restarted."); }
    else if (r.lines.length) { lastLog = r.last; append(r.lines); }
    else if (!logCount && !logEl.firstChild) clearLog(M && M.sources && !M.sources.log_file
      ? "Start the monitor with --log-file FILE to follow the engine's log." : "No log lines yet.");
  } catch (e) { /* the state poll reports the connection */ }
  setTimeout(pollLog, 1000);
}

$("#host").textContent = location.host;
setShares(100, 0);
pollState(); pollLog();
