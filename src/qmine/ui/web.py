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
/* The branch marker lives on the ROW. A wrapper div between </tr> and <tr> is
   foster-parented out of the table by the HTML5 parser and renders as nothing. */
tr.br-topdown td:first-child{box-shadow:inset 3px 0 #e0803c}
tr.br-bottomup td:first-child{box-shadow:inset 3px 0 #3c9ae0}
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
/* An agent return is prose, a list, or a few scalars — not a Python repr. */
.ret{white-space:pre-wrap;word-break:break-word;background:var(--bg);
border:1px solid var(--line);border-radius:6px;padding:8px 10px;margin:4px 0;
font-size:13px;line-height:1.7;max-height:26em;overflow:auto}
ul.ret{padding:8px 10px 8px 26px;white-space:normal}
ul.ret li{margin:2px 0}
.fld{margin:8px 0}
.fldk{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11px;
color:var(--dim);text-transform:none;letter-spacing:.02em;margin-bottom:2px}
.kv{display:flex;flex-wrap:wrap;gap:4px 14px;margin:4px 0 2px;font-size:12px}
.kv b{font-weight:600;color:var(--dim);font-weight:500}
details.ask{margin:2px 0 6px}
details.ask summary{cursor:pointer}
input[type=search]{width:100%;padding:7px 10px;border:1px solid var(--line);
border-radius:7px;background:var(--card);color:var(--fg);font-size:13px;margin-bottom:8px}
.right{float:right;color:var(--dim);font-weight:400}
/* The event feed. A four-hour run emits ~1,000 lines; a flat list of them is
   not something a person reads, it is something they scroll past. */
.chips{display:flex;flex-wrap:wrap;gap:6px;margin:0 0 8px}
.chip{border:1px solid var(--line);background:var(--card);color:var(--dim);
border-radius:999px;padding:3px 11px;font-size:12px;cursor:pointer;
font-family:inherit;line-height:1.5}
.chip:hover{border-color:var(--accent)}
.chip[aria-pressed=true]{background:var(--accent);border-color:var(--accent);
color:#fff;font-weight:500}
.chip .n{opacity:.72;margin-left:5px;font-variant-numeric:tabular-nums}
.ev{display:grid;grid-template-columns:58px 74px 1fr;gap:8px;align-items:baseline;
padding:3px 0;border-bottom:1px dotted var(--line)}
.ev .ph{font-size:10px;color:var(--dim);font-family:ui-monospace,Menlo,monospace;
overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
/* Severity is carried by a WORD, not by colour alone: WCAG 1.4.1 — colour must
   never be the only means of conveying information. Colour is the fast scan;
   the label is what makes it readable to anyone the colour fails for. */
.ev .lv{font-size:10px;font-weight:600;letter-spacing:.04em}
.ev.warn .lv{color:var(--warn)} .ev.error .lv{color:var(--bad)}
.ev.edit .lv{color:var(--accent)} .ev.ok .lv{color:var(--ok)}
.ev.info .lv{color:var(--pend)}
.ev.warn{background:color-mix(in srgb,var(--warn) 7%,transparent)}
.ev.error{background:color-mix(in srgb,var(--bad) 8%,transparent)}
.howto{font-size:12px;color:var(--dim);background:var(--bg);border:1px solid var(--line);
border-left:3px solid var(--accent);border-radius:0 6px 6px 0;padding:7px 10px;margin:0 0 9px}
"""

_JS = """
function qmFilter(inputId, tableId){
  const q=(document.getElementById(inputId).value||'').toLowerCase();
  document.querySelectorAll('#'+tableId+' [data-row]').forEach(r=>{
    r.style.display = !q || r.getAttribute('data-row').toLowerCase().includes(q) ? '' : 'none';
  });
}
var qmEv={lv:'all',ph:'all'};
function qmChip(kind,val,el){
  qmEv[kind]=val;
  document.querySelectorAll('.chip[data-kind='+kind+']').forEach(c=>
    c.setAttribute('aria-pressed', c===el ? 'true':'false'));
  qmEvApply();
}
function qmEvApply(){
  var q=(document.getElementById('evq').value||'').toLowerCase(), n=0;
  document.querySelectorAll('#events .ev').forEach(function(r){
    var ok = (qmEv.lv==='all' || r.dataset.lv===qmEv.lv)
          && (qmEv.ph==='all' || r.dataset.ph===qmEv.ph)
          && (!q || r.getAttribute('data-row').toLowerCase().includes(q));
    r.style.display = ok ? '' : 'none'; if(ok) n++;
  });
  var c=document.getElementById('evcount'); if(c) c.textContent=n+' 条可见';
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


def _return_html(payload: Any, budget: int = MAX_OUTPUT_CHARS) -> str:
    """Render what an agent RETURNED so a person can read it.

    The detail pane used to be `<pre>{str(payload)}</pre>`, and an agent return is
    a dict, so a reader got a Python repr: `{'markdown': '...', 'covered': [...]}`
    — one unwrapped line with escaped newlines, quotes and brackets in the way of
    the prose it was meant to show. The longest and most valuable returns (the
    report sections) were the least readable.

    An agent return is a schema object, so it has FIELDS, and the fields are what
    a reader wants: long text as text with its line breaks, lists as lists,
    scalars as a compact header. Chinese narrative prose is the common case here,
    so long text is rendered wrapped rather than in a horizontally scrolling
    `<pre>`.
    """
    def _clip(t: str) -> str:
        return (t if len(t) <= budget
                else t[:budget] + f"\n… [{len(t) - budget:,} more chars]")

    if payload is None or payload == "" or payload == {}:
        return "<pre class=t>this call recorded no return value</pre>"
    if isinstance(payload, str):
        return f"<div class=ret>{_e(_clip(payload))}</div>"
    if isinstance(payload, list):
        return ("<ul class=ret>"
                + "".join(f"<li>{_e(_clip(str(v)))}</li>" for v in payload[:60])
                + ("<li class=t>…</li>" if len(payload) > 60 else "") + "</ul>")
    if not isinstance(payload, dict):
        return f"<div class=ret>{_e(_clip(str(payload)))}</div>"

    scalars, blocks = [], []
    for k, v in payload.items():
        if isinstance(v, str) and ("\n" in v or len(v) > 120):
            blocks.append(f"<div class=fld><div class=fldk>{_e(k)}</div>"
                          f"<div class=ret>{_e(_clip(v))}</div></div>")
        elif isinstance(v, (list, tuple)):
            if not v:
                scalars.append((k, "[]"))
            else:
                inner = "".join(f"<li>{_e(str(x)[:300])}</li>" for x in list(v)[:40])
                more = "<li class=t>…</li>" if len(v) > 40 else ""
                blocks.append(f"<div class=fld><div class=fldk>{_e(k)} "
                              f"<span class=t>({len(v)})</span></div>"
                              f"<ul class=ret>{inner}{more}</ul></div>")
        elif isinstance(v, dict):
            blocks.append(f"<div class=fld><div class=fldk>{_e(k)}</div>"
                          f"<pre class=ret>{_e(_clip(json.dumps(v, ensure_ascii=False, indent=1)))}</pre></div>")
        else:
            scalars.append((k, v))
    head = ""
    if scalars:
        head = ("<div class=kv>"
                + "".join(f"<span><b>{_e(k)}</b> {_e(v)}</span>" for k, v in scalars)
                + "</div>")
    return head + "".join(blocks)


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
        # A BRANCH IS A PROPERTY OF THE ROW, NOT A WRAPPER AROUND IT.
        #
        # This opened `<div class='branch …'>` before a row and closed it with
        # `close_br`, which was initialised to "" and never once reassigned — two
        # unclosed divs, emitted between `</tr>` and `<tr>`. The HTML5 parser
        # foster-parents non-table content out of the table, so the grouping
        # rendered as NOTHING and the two concurrent branches read as sequential:
        # p2a (50m30s), p2b (48m24s) and p3 (14m03s) stack in a column that sums
        # past the 4h30m elapsed KPI with nothing on the page explaining why.
        #
        # A wrapper could not express this shape in any case — `PHASES`
        # interleaves branch phases with spine phases, so the branch members are
        # not contiguous. A per-row class and a column state it directly, and
        # survive the parser.
        br = getattr(sp, "branch", "")
        seen_branch.add(br) if br else None
        row = (f"<tr class='{('br-' + _e(br)) if br else ''}' "
               f"data-row='{_e(sp.key + ' ' + label + ' ' + br)}'>"
               f"<td class=mono>{_e(sp.key)}</td>"
               f"<td><span class='pill s-{_e(st)}'>{_e(st)}</span></td>"
               f"<td class=t>{_e(br) if br else '—'}</td>"
               f"<td>{_e(label)}<div class=t>{_e(why)}</div></td>"
               f"<td class='num mono'>{_e(_dur(secs))}</td></tr>")
        rows.append(row)
    body = ("<table><tr><th>phase</th><th>state</th><th>branch</th>"
            "<th>what it does</th><th class=num>elapsed</th></tr>"
            + "".join(rows) + "</table>")
    if seen_branch:
        body += ("<div class=t>本流水线在 " + _e("、".join(sorted(seen_branch)))
                 + " 两条分支上并行, 因此各阶段耗时之和大于总时长 —— "
                 "同一段时间被两条分支同时占用。</div>")
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
    # JOIN ON THE CALL KEY, NEVER ON POSITION.
    #
    # These are two streams describing the same calls: `dash.all_agents` (the
    # one-line summaries) and the transcript (the full returns). They used to be
    # paired by index — `pool[min(i, len(pool)-1)]`, where `i` counts down a
    # REVERSED list across ALL roles and `pool` is one role's calls in
    # chronological order. Those two orderings have nothing to do with each
    # other, so an expanded row showed some other call's output: on live42 the
    # row headed `reporter … 04:42:11`, the first attempt at `audit_and_limits`,
    # opened onto the top-down taxonomy section. A reader cannot tell a
    # mispaired answer from a correct one, which makes it worse than a blank.
    by_key: dict[str, dict[str, Any]] = {}
    for rec in (transcript or [])[-MAX_TRANSCRIPT_ITEMS:]:
        k = str(rec.get("cache_key") or "")
        if k:
            by_key[k] = rec

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
                         + ("<pre class=t>完整返回在 agent_transcript.json —— "
                            f"本页只保留最新 {detail_limit} 条全文以保持可打开</pre></details>"
                            if transcript else
                            "<pre class=t>完整返回未随本次运行保存 "
                            "(agent_transcript.json 缺失或未加载)</pre></details>"))
            continue
        rec = by_key.get(str(a.get("key") or ""))
        if rec is not None:
            detail = _return_html(rec.get("output"))
            ask = str(rec.get("user_head") or "")
            if ask:
                detail = (f"<details class=ask><summary class=t>它被问了什么 (开头)"
                          f"</summary><div class=ret>{_e(ask)}</div></details>") + detail
        elif transcript is None:
            # `qmine watch` replays a FINISHED run, where there is no registry to
            # call back into — so every row opened onto "not captured" and the
            # feature looked broken rather than unavailable. Say which it is.
            detail = ("<pre class=t>本页在回放已结束的运行, 完整返回来自 "
                      "agent_transcript.json —— 该文件缺失或未被加载</pre>")
        else:
            detail = "<pre class=t>full return not captured for this call</pre>"
        items.append(f"<details data-row='{row}'>{head}{detail}</details>")
    if not items:
        return "<div class=sub>no agent calls yet</div>"
    return ("<input type=search id=agq placeholder='filter by role, model or result…' "
            "oninput=\"qmFilter('agq','agents')\">"
            f"<div id=agents>{''.join(items)}</div>")


def _feed(dash: Any, phases: list[Any]) -> str:
    """The event log, as something a person can actually read.

    It was a flat reverse-chronological list of ~1,000 terse bilingual strings
    with a timestamp — no severity, no phase, no way in. Over a four-hour run the
    two lines that decide whether the delivery is trustworthy ("3/9 节通过校验",
    "unfixable finding dropped") sit between hundreds of routine ones and read
    exactly the same.

    Three things fix that, and each is doing one job:

    * **Severity, as a word and a tint.** Colour alone would not do it — WCAG 2.2
      SC 1.4.1 (Use of Color) requires that colour never be the sole carrier of
      information — so every row states its level in text as well.
    * **Faceting**, the pattern every log explorer converges on: pre-computed
      chips over the dimensions that exist (level, phase), each showing its own
      count, so a reader can see there ARE 14 warnings without hunting for them.
    * **A sentence saying how to read it.** The audience did not build this
      pipeline. One line of orientation next to the thing it orients costs
      nothing and is the difference between a log and a wall.
    """
    rows = dash.all_activity[-400:]
    if not rows:
        return "<div class=sub>no events yet</div>"

    label_of = {sp.key: sp.label(getattr(dash, "language", "zh")) for sp in phases}
    lv_names = {"error": "失败", "warn": "警告", "edit": "改稿", "ok": "通过", "info": "进展"}
    lv_counts: dict[str, int] = {}
    ph_counts: dict[str, int] = {}
    html_rows: list[str] = []
    for t, m, lv, ph in rows[::-1]:
        lv_counts[lv] = lv_counts.get(lv, 0) + 1
        if ph:
            ph_counts[ph] = ph_counts.get(ph, 0) + 1
        html_rows.append(
            f"<div class='ev {lv}' data-lv='{_e(lv)}' data-ph='{_e(ph)}' "
            f"data-row='{_e(m)} {_e(ph)} {_e(lv_names.get(lv, lv))}'>"
            f"<span class=t>{_e(_clock(t))}</span>"
            f"<span class=lv>{_e(lv_names.get(lv, lv))}</span>"
            f"<span><span class=ph>{_e(label_of.get(ph, ph))}</span> {_e(m)}</span></div>")

    def chips(kind: str, items: list[tuple[str, str, int]]) -> str:
        out = [f"<button class=chip data-kind={kind} aria-pressed=true "
               f"onclick=\"qmChip('{kind}','all',this)\">全部</button>"]
        for val, text, n in items:
            out.append(f"<button class=chip data-kind={kind} aria-pressed=false "
                       f"onclick=\"qmChip('{kind}','{_e(val)}',this)\">{_e(text)}"
                       f"<span class=n>{n}</span></button>")
        return f"<div class=chips>{''.join(out)}</div>"

    lv_chips = chips("lv", [(k, lv_names[k], lv_counts[k])
                           for k in ("error", "warn", "edit", "ok", "info")
                           if lv_counts.get(k)])
    ph_chips = chips("ph", [(k, label_of.get(k, k), n)
                            for k, n in sorted(ph_counts.items(), key=lambda kv: -kv[1])[:8]])
    n_flag = lv_counts.get("warn", 0) + lv_counts.get("error", 0)
    howto = (
        "<div class=howto><b>怎么读这一栏。</b> 每一行是流水线自己发出的一条事件, 最新的在最上面。"
        "左起依次是<b>时间</b>、<b>级别</b>、<b>发出它的阶段</b>, 然后是内容。"
        f"本次运行有 <b>{n_flag}</b> 条<b>警告或失败</b> —— 先点下面的「警告」只看这些, "
        "它们是唯一可能改变交付结论的事件; 「改稿」是交付前审核对报告正文做的逐条修订。"
        "</div>")
    return (howto + lv_chips + ph_chips
            + "<input type=search id=evq placeholder='在事件里搜索…' oninput='qmEvApply()'>"
            + f"<div class=t id=evcount>{len(rows)} 条可见</div>"
            + f"<div class=feed id=events>{''.join(html_rows)}</div>")


def _artifacts(arts: list[dict[str, Any]]) -> str:
    if not arts:
        return "<div class=sub>nothing written yet</div>"
    def _row(a: dict[str, Any]) -> str:
        size = a.get("bytes") or a.get("size_bytes")
        size_s = f"{int(size):,}" if isinstance(size, (int, float)) else ""
        # `index.jsonl` writes `name`; this read `key` and found nothing, so the
        # artifact column was blank for every row of every run. `key` is kept
        # first in case a caller supplies the store's own key.
        key, summary = a.get("key") or a.get("name", ""), a.get("summary", "")
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
    # THE RUN'S OWN CLOCK, NOT THE READER'S.
    #
    # `dash.started` is set from the first event's timestamp, and on a REPLAY
    # that timestamp is seconds-since-midnight (`parse_log_clock`), not an epoch.
    # Subtracting it from `time.time()` gave "496,632h" at the top of every page
    # `qmine watch` produced for a finished run — the single most prominent
    # number on the page, wrong by fifty-six years. Same `_EPOCH_FLOOR` rule the
    # clock formatter already uses: when the run's clock is a replay clock,
    # measure against the LAST event rather than against now.
    if dash.started and dash.started < _EPOCH_FLOOR:
        last = max((e[0] for e in dash.all_activity), default=dash.started)
        elapsed = max(0.0, last - dash.started)
    else:
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
    feed_body = _feed(dash, phases)

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
