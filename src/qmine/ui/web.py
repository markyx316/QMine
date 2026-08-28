"""A browsable dashboard for a run, written as one self-contained HTML file.

**Why a file and not a terminal panel.** The Rich panel keeps `agents[-8:]`,
`activity[-6:]` and `metrics[-8:]` because that is what fits on a screen. live40
emitted **696 agent calls**, 24 gates and ~1000 events over four hours, so the
panel showed about 1% of the run and discarded the rest as it went. What an
operator actually wants to ask — *what did that researcher return?*, *which
artifacts exist so far?*, *what is queued behind this?* — needs scrolling,
folding and search, and a terminal has none of them.

**Why a file and not a server.** No port, no dependency, no lifecycle to get
wrong, and it keeps working after the run ends: the page IS the record. It is
rewritten in place while the run proceeds and carries a meta-refresh until the
run finishes, so a browser left open follows along. Everything is inlined —
opening it over `file://` from a copied directory works identically.

**One model, two views.** This renders `LiveDashboard`'s state; it does not parse
anything itself. The terminal view and this view therefore cannot disagree, and
`.claude/rules/dashboard.md`'s rule still applies — build against a recorded
`run.log`, then re-render against a live one.
"""
from __future__ import annotations

import html
import json
import time
from pathlib import Path
from typing import Any

#: A single agent return can be tens of thousands of characters and live40 made
#: 696 calls. Embedding every full return produced a **5 MB page that hung the
#: browser's renderer** — measured, not guessed — which is a worse dashboard than
#: one that shows less. So full text is kept for the most recent `DETAIL_LIMIT`
#: calls, which are the ones an operator is actually watching, and everything
#: older keeps its one-line result. Nothing is lost: `agent_transcript.json`
#: carries every return in full, and the page says so.
MAX_OUTPUT_CHARS = 4000
DETAIL_LIMIT = 40
MAX_TRANSCRIPT_ITEMS = 1200
REFRESH_SECONDS = 5

_CSS = """
:root{--bg:#fbfbfd;--fg:#1a1a1f;--dim:#6b6b76;--line:#e3e3ea;--card:#fff;
--ok:#1a7f4b;--warn:#a8690b;--bad:#c0392b;--run:#1f6feb;--pend:#a9a9b4;--accent:#5b3df5}
@media(prefers-color-scheme:dark){:root{--bg:#0f1115;--fg:#e6e6ee;--dim:#8b8b98;
--line:#242833;--card:#161922;--ok:#3fb950;--warn:#d29922;--bad:#f85149;
--run:#58a6ff;--pend:#484f58;--accent:#a78bfa}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.55 -apple-system,
BlinkMacSystemFont,"Segoe UI","PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif}
.wrap{max-width:1400px;margin:0 auto;padding:20px}
h1{font-size:19px;margin:0 0 2px}
h2{font-size:14px;text-transform:uppercase;letter-spacing:.07em;color:var(--dim);
margin:26px 0 10px;font-weight:600}
.sub{color:var(--dim);font-size:13px}
.card{background:var(--card);border:1px solid var(--line);border-radius:9px;padding:14px}
.grid{display:grid;gap:12px}
.g4{grid-template-columns:repeat(auto-fit,minmax(150px,1fr))}
.g2{grid-template-columns:2fr 1fr}
@media(max-width:900px){.g2{grid-template-columns:1fr}}
.kpi .v{font-size:22px;font-weight:600;font-variant-numeric:tabular-nums}
.kpi .k{color:var(--dim);font-size:12px}
table{width:100%;border-collapse:collapse}
th{text-align:left;font-size:12px;color:var(--dim);font-weight:600;
padding:6px 8px;border-bottom:1px solid var(--line);position:sticky;top:0;background:var(--card)}
td{padding:6px 8px;border-bottom:1px solid var(--line);vertical-align:top}
tr:last-child td{border-bottom:0}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px}
.num{text-align:right;font-variant-numeric:tabular-nums}
.pill{display:inline-block;padding:1px 8px;border-radius:999px;font-size:11px;font-weight:600}
.s-done{background:color-mix(in srgb,var(--ok) 15%,transparent);color:var(--ok)}
.s-running{background:color-mix(in srgb,var(--run) 15%,transparent);color:var(--run)}
.s-pending{background:color-mix(in srgb,var(--pend) 20%,transparent);color:var(--dim)}
.s-failed,.s-FAILED{background:color-mix(in srgb,var(--bad) 15%,transparent);color:var(--bad)}
.s-PASSED{background:color-mix(in srgb,var(--ok) 15%,transparent);color:var(--ok)}
.s-WARNED{background:color-mix(in srgb,var(--warn) 18%,transparent);color:var(--warn)}
.bar{height:5px;border-radius:3px;background:var(--line);overflow:hidden;margin-top:8px}
.bar>i{display:block;height:100%;background:var(--accent)}
.branch{border-left:3px solid var(--line);padding-left:10px;margin:8px 0}
.b-topdown{border-color:#e0803c}.b-bottomup{border-color:#3c9ae0}
details{border-bottom:1px solid var(--line)}
details:last-child{border-bottom:0}
summary{cursor:pointer;padding:7px 4px;list-style:none;display:flex;gap:10px;align-items:baseline}
summary::-webkit-details-marker{display:none}
summary:hover{background:color-mix(in srgb,var(--accent) 6%,transparent)}
pre{white-space:pre-wrap;word-break:break-word;background:var(--bg);border:1px solid var(--line);
border-radius:6px;padding:10px;margin:6px 0 12px;font-size:12px;max-height:460px;overflow:auto}
.feed{max-height:320px;overflow:auto;font-size:12px}
.feed div{padding:2px 0;border-bottom:1px dotted var(--line)}
.t{color:var(--dim);font-size:11px}
input[type=search]{width:100%;padding:7px 10px;border:1px solid var(--line);
border-radius:7px;background:var(--card);color:var(--fg);font-size:13px;margin-bottom:8px}
.right{float:right;color:var(--dim);font-weight:400}
"""

_JS = """
function qmFilter(inputId, tableId){
  const q=(document.getElementById(inputId).value||'').toLowerCase();
  document.querySelectorAll('#'+tableId+' [data-row]').forEach(r=>{
    r.style.display = !q || r.getAttribute('data-row').toLowerCase().includes(q) ? '' : 'none';
  });
}
"""


def _e(x: Any) -> str:
    return html.escape(str(x if x is not None else ""), quote=True)


def _dur(sec: float | None) -> str:
    if sec is None:
        return ""
    sec = int(sec)
    if sec < 60:
        return f"{sec}s"
    if sec < 3600:
        return f"{sec // 60}m{sec % 60:02d}s"
    return f"{sec // 3600}h{(sec % 3600) // 60:02d}m"


#: Anything below this is a REPLAY clock (seconds since midnight), not an epoch.
#: Two days of seconds; a real epoch is ~1.7e9.
_EPOCH_FLOOR = 2 * 86400


def _clock(ts: float | None) -> str:
    """Wall-clock string for either an epoch or a replay clock.

    `parse_log_clock` yields seconds-since-midnight so a replay can time itself,
    and `time.localtime` on that treats it as an epoch — 22:03:48 became 79,428,
    which rendered as "06:03:48" once the timezone was added. Durations were
    unaffected (differences cancel) but every absolute time on the page was wrong
    by the UTC offset, and a run watched after midnight looked like it happened
    the previous morning.
    """
    if not ts:
        return ""
    if ts < _EPOCH_FLOOR:
        ts = int(ts) % 86400
        return f"{ts // 3600:02d}:{(ts % 3600) // 60:02d}:{ts % 60:02d}"
    return time.strftime("%H:%M:%S", time.localtime(ts))


def _card(title: str, body: str, sub: str = "") -> str:
    head = f"<h2>{_e(title)}{f'<span class=right>{_e(sub)}</span>' if sub else ''}</h2>"
    return head + f"<div class=card>{body}</div>"


def _pipeline(dash: Any, phases: list[Any]) -> str:
    """Every phase: done, running, or queued — and the two branches as branches.

    A flat list cannot show that p2a and p3 run at the same time, which is the
    single most confusing thing about watching this pipeline: two rows advance
    at once and a reader assumes one of them is stale.
    """
    rows: list[str] = []
    seen_branch: set[str] = set()
    for sp in phases:
        st = dash.status.get(sp.key, "pending")
        secs = dash.timings.get(sp.key)
        if st == "running" and sp.key in getattr(dash, "_phase_start", {}):
            secs = time.time() - dash._phase_start[sp.key]
        label = sp.label(getattr(dash, "language", "zh"))
        why = sp.why(getattr(dash, "language", "zh"))
        br = getattr(sp, "branch", "")
        open_br = close_br = ""
        if br and br not in seen_branch:
            seen_branch.add(br)
            open_br = f"<div class='branch b-{_e(br)}'>"
        row = (f"<tr data-row='{_e(sp.key + ' ' + label)}'>"
               f"<td class=mono>{_e(sp.key)}</td>"
               f"<td><span class='pill s-{_e(st)}'>{_e(st)}</span></td>"
               f"<td>{_e(label)}<div class=t>{_e(why)}</div></td>"
               f"<td class='num mono'>{_e(_dur(secs))}</td></tr>")
        rows.append(open_br + row + close_br)
    body = ("<table><tr><th>phase</th><th>state</th><th>what it does</th>"
            "<th class=num>elapsed</th></tr>" + "".join(rows) + "</table>")
    done = sum(1 for sp in phases if dash.status.get(sp.key) == "done")
    pct = int(100 * done / max(len(phases), 1))
    body += f"<div class=bar><i style='width:{pct}%'></i></div>"
    return body


def _agents(dash: Any, transcript: list[dict[str, Any]] | None,
            *, detail_limit: int = DETAIL_LIMIT) -> str:
    """Every agent call, with what it actually returned.

    The log carries a one-line summary per call; the full return lives only in
    `registry.raw_log` and, at the very end, `agent_transcript.json`. Pairing them
    here is the difference between "researcher_legacy_audit ok 225s" and being
    able to read what it concluded while the run is still going.
    """
    by_role: dict[str, list[dict[str, Any]]] = {}
    for rec in (transcript or [])[-MAX_TRANSCRIPT_ITEMS:]:
        by_role.setdefault(str(rec.get("role", "")), []).append(rec)

    recent = list(reversed(dash.all_agents[-MAX_TRANSCRIPT_ITEMS:]))
    items: list[str] = []
    for i, a in enumerate(recent):
        role = a.get("role", "")
        mark = "✓" if a.get("ok") else "✗"
        colour = "var(--ok)" if a.get("ok") else "var(--bad)"
        head = (f"<summary><span style='color:{colour}'>{mark}</span>"
                f"<b class=mono>{_e(role)}</b>"
                f"<span class=t>{_e(a.get('model',''))} · {_e(a.get('seconds',''))}s · "
                f"out {_e(a.get('out_tokens',''))} · {_e(_clock(a.get('at')))}</span>"
                f"<span style='flex:1'></span></summary>"
                f"<div class=sub style='padding:0 4px'>{_e(a.get('returned',''))}</div>")
        row = _e(role + " " + str(a.get("model", "")) + " " + str(a.get("returned", "")))
        if i >= detail_limit:
            # Older calls keep their result line; the full return stays in the
            # transcript artifact rather than in a page that has to stay openable.
            items.append(f"<details data-row='{row}'>{head}"
                         "<pre class=t>full return in agent_transcript.json — this page keeps "
                         f"the newest {detail_limit} in full so it stays responsive</pre></details>")
            continue
        pool = by_role.get(role) or by_role.get(role.split("_")[0], [])
        full = ""
        if pool:
            rec = pool[-1] if len(pool) == 1 else pool[min(i, len(pool) - 1)]
            out = str(rec.get("output", ""))
            if len(out) > MAX_OUTPUT_CHARS:
                out = out[:MAX_OUTPUT_CHARS] + f"\n… [{len(out) - MAX_OUTPUT_CHARS:,} more chars]"
            full = out
        detail = (f"<pre>{_e(full)}</pre>" if full else
                  "<pre class=t>full return not captured for this call</pre>")
        items.append(f"<details data-row='{row}'>{head}{detail}</details>")
    if not items:
        return "<div class=sub>no agent calls yet</div>"
    return ("<input type=search id=agq placeholder='filter by role, model or result…' "
            "oninput=\"qmFilter('agq','agents')\">"
            f"<div id=agents>{''.join(items)}</div>")


def _artifacts(arts: list[dict[str, Any]]) -> str:
    if not arts:
        return "<div class=sub>nothing written yet</div>"
    def _row(a: dict[str, Any]) -> str:
        size = a.get("bytes") or a.get("size_bytes")
        size_s = f"{int(size):,}" if isinstance(size, (int, float)) else ""
        key, summary = a.get("key", ""), a.get("summary", "")
        return (f"<tr data-row='{_e(key)} {_e(summary)}'>"
                f"<td class=mono>{_e(key)}</td>"
                f"<td class=mono style='color:var(--dim)'>{_e(a.get('producer',''))}</td>"
                f"<td>{_e(summary)}</td>"
                f"<td class='num mono'>{_e(size_s)}</td></tr>")

    rows = "".join(_row(a) for a in reversed(arts))
    return ("<input type=search id=arq placeholder='filter artifacts…' "
            "oninput=\"qmFilter('arq','arts')\">"
            "<table id=arts><tr><th>artifact</th><th>from</th><th>summary</th>"
            f"<th class=num>bytes</th></tr>{rows}</table>")


def _gates(gates: list[dict[str, Any]]) -> str:
    if not gates:
        return "<div class=sub>no gates evaluated yet</div>"
    rows = "".join(
        f"<tr data-row='{_e(g.get('gate',''))} {_e(g.get('status',''))}'>"
        f"<td class=mono>{_e(g.get('gate',''))}</td>"
        f"<td><span class='pill s-{_e(g.get('status',''))}'>{_e(g.get('status',''))}</span></td>"
        f"<td>{_e(g.get('note',''))}</td></tr>" for g in reversed(gates))
    return f"<table><tr><th>gate</th><th></th><th>what it measured</th></tr>{rows}</table>"


def render(dash: Any, phases: list[Any], *, finished: bool = False) -> str:
    """Render the whole page from the dashboard's own state."""
    elapsed = time.time() - dash.started
    usage: dict[str, Any] = {}
    if getattr(dash, "usage_fn", None):
        try:
            usage = dash.usage_fn() or {}
        except Exception:                                        # noqa: BLE001
            usage = {}
    transcript = None
    if getattr(dash, "transcript_fn", None):
        try:
            transcript = dash.transcript_fn()
        except Exception:                                        # noqa: BLE001
            transcript = None
    arts = list(getattr(dash, "artifacts", []) or [])

    running = sorted(getattr(dash, "running", set()))
    done = sum(1 for sp in phases if dash.status.get(sp.key) == "done")
    queued = [sp.key for sp in phases if dash.status.get(sp.key, "pending") == "pending"]

    state = ("halted" if getattr(dash, "halted", False)
             else "finished" if finished else "running")
    state_colour = {"halted": "var(--bad)", "finished": "var(--ok)",
                    "running": "var(--run)"}[state]

    kpis = [("elapsed", _dur(elapsed)), ("phases", f"{done}/{len(phases)}"),
            ("agent calls", f"{usage.get('calls', 0):,}"),
            ("output tokens", f"{usage.get('output_tokens', 0):,}"),
            ("artifacts", f"{len(arts):,}")]
    if usage.get("estimated_cost_usd"):
        kpis.append(("est. spend", f"${usage['estimated_cost_usd']:.2f}"))
    if usage.get("errors"):
        kpis.append(("errors", f"{usage['errors']:,}"))
    kpi_html = "".join(f"<div class='card kpi'><div class=v>{_e(v)}</div>"
                       f"<div class=k>{_e(k)}</div></div>" for k, v in kpis)

    if running:
        blocks = []
        for key in running:
            sp = next((x for x in phases if x.key == key), None)
            lbl = sp.label(getattr(dash, "language", "zh")) if sp else key
            since = dash._phase_start.get(key)
            why = sp.why(getattr(dash, "language", "zh")) if sp else ""
            blocks.append(f"<div style='margin-bottom:10px'><b class=mono>{_e(key)}</b> "
                          f"{_e(lbl)} <span class=t>{_e(_dur(time.time() - since) if since else '')}"
                          f"</span><div class=t>{_e(why)}</div></div>")
        now_body = "".join(blocks)
    else:
        now_body = "<div class=sub>nothing running</div>"

    # The event feed is its own card and is ALWAYS shown. It was nested inside
    # "running now", so the moment the run finished — the moment you most want to
    # read back what happened — it vanished along with the running phases.
    feed_rows = "".join(
        f"<div data-row='{_e(m)}'><span class=t>{_e(_clock(t))}</span> {_e(m)}</div>"
        for t, m in dash.all_activity[-400:][::-1])
    feed_body = (f"<input type=search id=evq placeholder='filter events…' "
                 f"oninput=\"qmFilter('evq','events')\">"
                 f"<div class=feed id=events>{feed_rows}</div>"
                 if feed_rows else "<div class=sub>no events yet</div>")

    queue_body = ("<div class=sub>nothing queued — this is the last phase</div>" if not queued
                  else "".join(
                      f"<div><span class=mono>{_e(k)}</span> <span class=t>"
                      f"{_e(next((sp.label(getattr(dash,'language','zh')) for sp in phases if sp.key == k), ''))}"
                      "</span></div>" for k in queued))

    refresh = ("" if finished or state == "halted"
               else f"<meta http-equiv=refresh content={REFRESH_SECONDS}>")
    halt = (f"<div class=card style='border-color:var(--bad);margin-bottom:12px'>"
            f"<b style='color:var(--bad)'>run halted</b><div class=sub>"
            f"{_e(getattr(dash, 'halt_reason', ''))}</div></div>"
            if getattr(dash, "halted", False) else "")

    return f"""<!doctype html><html lang=zh><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">{refresh}
<title>QMine · {_e(dash.run_id)}</title><style>{_CSS}</style></head><body><div class=wrap>
<h1>QMine · <span class=mono>{_e(dash.run_id)}</span>
<span class=pill style="background:color-mix(in srgb,{state_colour} 15%,transparent);
color:{state_colour}">{_e(state)}</span></h1>
<div class=sub>{_e(dash.domain)} · provider <b>{_e(dash.provider or '?')}</b>
· started {_e(_clock(dash.started))} · refreshed {_e(_clock(time.time()))}</div>
{halt}
<div class="grid g4" style="margin-top:14px">{kpi_html}</div>
<div class="grid g2" style="margin-top:8px">
  <div>{_card('pipeline', _pipeline(dash, phases))}</div>
  <div>{_card('running now', now_body, f'{len(running)} concurrent')}
       {_card('queued', queue_body, f'{len(queued)} left')}</div>
</div>
{_card('event log', feed_body, f'{len(dash.all_activity):,} events')}
{_card('agents', _agents(dash, transcript), f'{len(dash.all_agents):,} calls')}
{_card('artifacts produced', _artifacts(arts), f'{len(arts):,}')}
{_card('quality gates', _gates(dash.all_gates), f'{len(dash.all_gates)} evaluated')}
<script>{_JS}</script></div></body></html>"""


class HtmlWriter:
    """Throttled writer so a 4-hour run does not rewrite the page on every line.

    **And a heartbeat, because events are bursty.** The page is rewritten from
    `LiveDashboard.handle`, so a phase that runs for forty minutes without
    emitting — five researchers fanning out, 240 annotation batches — leaves it
    frozen: the meta-refresh reloads, and reloads the same stale elapsed time.
    Observed within the first minute of live41. A daemon thread re-renders on a
    timer so "running for 38m" keeps counting whether or not anything is logged.
    """

    def __init__(self, path: Path, phases: list[Any], *, every: float = 3.0,
                 heartbeat: float = 10.0) -> None:
        self.path, self.phases, self.every = Path(path), phases, every
        self._last = 0.0
        self._beat = heartbeat
        self._thread: Any = None
        self._stop: Any = None

    def start_heartbeat(self, dash: Any) -> None:
        """Re-render on a timer. Daemon, so it can never hold the process open."""
        import threading

        if self._thread is not None or not self._beat:
            return
        self._stop = threading.Event()
        def _tick() -> None:
            while not self._stop.wait(self._beat):
                try:
                    self.maybe_write(dash, force=True)
                except Exception:                                # noqa: BLE001
                    pass
        self._thread = threading.Thread(target=_tick, name="qmine-dashboard",
                                        daemon=True)
        self._thread.start()

    def stop_heartbeat(self) -> None:
        if self._stop is not None:
            self._stop.set()
        self._thread = None

    def maybe_write(self, dash: Any, *, force: bool = False, finished: bool = False) -> None:
        now = time.time()
        if not force and not finished and now - self._last < self.every:
            return
        self._last = now
        try:
            tmp = self.path.with_suffix(".html.tmp")
            tmp.write_text(render(dash, self.phases, finished=finished), encoding="utf-8")
            tmp.replace(self.path)       # atomic: a reader never sees a half-page
        except Exception:                                        # noqa: BLE001
            pass


def artifacts_from_index(index_path: Path) -> list[dict[str, Any]]:
    """Read `index.jsonl`, which the store appends to as artifacts are written."""
    out: list[dict[str, Any]] = []
    try:
        for line in Path(index_path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line))
    except Exception:                                            # noqa: BLE001
        pass
    return out
