"""A live dashboard for a run that takes tens of minutes.

The problem with a log stream is that it answers "what just happened" and not
"where am I". On a twelve-phase pipeline where one phase can take eight minutes,
a scrolling log leaves the operator unable to tell progress from a hang without
reading carefully — and it buries the two things they actually want continuously
visible: which quality gates have fired, and how much money has been spent.

So the dashboard keeps four things on screen at once: the phase list with
elapsed times, what the agents are doing right now, the metrics as they land,
and a running gate/cost tally. Everything else still goes to the log file.

Degrades deliberately: with `--plain`, `--quiet`, a pipe, or a terminal too
narrow to split, it falls back — to stacked panels, and past that to plain lines.
A dashboard that breaks CI output is worse than no dashboard. Whatever it does,
`<run>/run.log` still receives the full stream, so choosing the pretty view never
costs you the record of the run.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from rich.cells import cell_len
from rich.markup import escape


@dataclass
class PhaseSpec:
    """One row of the phase list, and what that row is actually doing."""

    key: str
    label_zh: str
    label_en: str
    #: What this phase does and why, for the operator watching it run. A phase
    #: name tells you where you are; it does not tell you what a four-minute
    #: pause is buying, which is the question a long run actually raises.
    why_zh: str = ""
    why_en: str = ""

    def label(self, lang: str) -> str:
        return self.label_zh if lang == "zh" else self.label_en

    def why(self, lang: str) -> str:
        return self.why_zh if lang == "zh" else self.why_en


#: The pipeline, in execution order, with the names an operator recognises.
PHASES: list[PhaseSpec] = [
    PhaseSpec("p0", "工程底座", "foundation",
              why_zh="固定随机种子、解析配置、建立产物目录与断点",
              why_en="seeds, config, artifact store, checkpoints"),
    PhaseSpec("p1", "数据审计与模板挖掘", "audit & template mining",
              why_zh="统计语料形态、挖掘模板句式、筛查语言与风险",
              why_en="corpus shape, phrasing families, language & risk screen"),
    PhaseSpec("p2a", "意图体系设计 (研究扇出)", "taxonomy design",
              why_zh="5 名研究员并行提案 → 架构师定类目 → 规则员写裁决规则 → 评审 → 试标校准",
              why_en="5 researchers, architect, rule writer, critic, pilot calibration"),
    PhaseSpec("p2b", "金标构建 (双盲+κ)", "gold standard",
              why_zh="双盲标注金标集, 用标注者自一致上限解读 κ",
              why_en="double-blind gold set; kappa read against the self-consistency ceiling"),
    PhaseSpec("p3", "表征构建 (bake-off + α)", "representation",
              why_zh="编码器 bake-off、字符 TF-IDF 稀疏块、α 扫描定混合表征",
              why_en="encoder bake-off, sparse block, alpha sweep"),
    PhaseSpec("p2c", "规则+ML 分类器", "classifier",
              why_zh="规则优先、ML 兜底的分类器, 含交叉验证与概率校准",
              why_en="rules-first classifier with ML fallback, CV and calibration"),
    PhaseSpec("p2d", "对抗验证", "adversarial validation",
              why_zh="用对抗样本攻击分类器, 检验其真实鲁棒性",
              why_en="adversarial probes against the classifier"),
    PhaseSpec("p2e", "L2 子意图层", "sub-intents",
              why_zh="在每个 L1 之下切分 L2 子意图",
              why_en="L2 sub-intents beneath each L1"),
    PhaseSpec("p4", "算法选型 battery", "algorithm battery",
              why_zh="多算法对照 — 检验 KMeans 的球形簇假设是否被推翻",
              why_en="battery — can KMeans's assumption be falsified"),
    PhaseSpec("p5", "粒度选择 (K 三角验证)", "granularity",
              why_zh="K 扫描: 意图对齐定位 K, 稳定性只做否决",
              why_en="K sweep: intent alignment locates K, stability only rejects"),
    PhaseSpec("p6", "两层层级构建", "hierarchy",
              why_zh="构建家族/叶子两层结构, 并做留出复现检验",
              why_en="two-level tree plus held-out reproduction"),
    PhaseSpec("p7", "盲评命名与树审计", "blind naming & audit",
              why_zh="命名者看不到任何旧标签, 命名后审计整棵树",
              why_en="naming behind the blindness firewall, then tree audit"),
    PhaseSpec("p8", "治理合并 (执行)", "governance",
              why_zh="执行治理处方: 合并、拆分、重命名",
              why_en="execute governance prescriptions: merge, split, rename"),
    PhaseSpec("p9", "统一度量面板", "metrics panel",
              why_zh="在同一度量口径下重测每个候选方案",
              why_en="re-measure every candidate under one harness"),
    PhaseSpec("p10", "部署与全量打标", "deployment",
              why_zh="全量打标、质心模型、两条路线的对照表",
              why_en="full labelling, centroid model, route crosswalk"),
    PhaseSpec("p11", "报告与 notebook", "reporting",
              why_zh="生成中文报告、可执行 notebook 与全部图表",
              why_en="Chinese report, executed notebook, figures"),
    PhaseSpec("p12", "维护循环", "maintenance",
              why_zh="基线快照、新意图哨兵、重跑契约",
              why_en="baseline, novelty sentinel, rerun contract"),
]

_PHASE_RE = re.compile(r"^\s*(P\d+[a-e]?)\b", re.I)

#: One agent turn, as the runner writes it:
#:   "  ~ annotator_a ok 12.3s out 1,010 · qwen3-next-80b · labels=25"
_AGENT_RE = re.compile(
    r"^\s*~ (\S+) (ok|!!) ([\d.]+)s out ([\d,]+) · ([^·]+?) · ?(.*)$"
)


#: Wide enough for the longest phase label at its true rendered width. CJK
#: glyphs occupy two cells, so measuring these labels with ``len`` under-counts
#: by about a third and the column clips exactly the Chinese names an operator
#: reads. Derived, not guessed, so adding a phase cannot silently truncate it.
_PHASE_PANE_WIDTH = max(
    cell_len(spec.label_zh) for spec in PHASES
) + 2 + 7 + 6 + 4  # icon column, elapsed column, three column gutters, panel


_GATE_RE = re.compile(r"gate\s+(\S+):\s*(PASSED|WARNED|FAILED|REJECTED)\s*—?\s*(.*)", re.I)

#: A metric is a label immediately followed by a decimal. The label must begin at
#: a real boundary — start of line, or after punctuation — or the scrape slices
#: words in half: "held-out structure reproduction 0.926" was being shown as
#: "out structure reproduction", which reads like a different quantity.
_METRIC_RE = re.compile(
    r"(?:^\s*|[—:,·()\[\]]\s*)"
    r"([A-Za-z\u4e00-\u9fff][A-Za-z\u4e00-\u9fff_ -]{2,30}?)[\s:：]+([0-9]*\.[0-9]+)"
)

#: The keys the phase list can actually draw.
_PHASE_KEYS = {spec.key for spec in PHASES}


def compact(n: float) -> str:
    """1,884,783 -> 1.9M. A panel column is narrower than a token count."""
    for cut, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if abs(n) >= cut:
            return f"{n / cut:.1f}{suffix}".replace(".0", "")
    return f"{n:,.0f}"


def declared_phase(key: str) -> str | None:
    """Map an emitted phase key onto the row that represents it.

    The pipeline announces sub-steps — ``P3a`` encoder bake-off, ``P3b`` sparse
    block, ``P3c`` alpha sweep — while the list declares one ``p3`` row. Keying
    the status dict on the emitted string meant that row was never touched: the
    representation phase sat at ○ for the entire run and finished a completed run
    still looking un-started, with its timings filed under keys nothing renders.
    """
    key = key.lower()
    while key and key not in _PHASE_KEYS:
        if not key[-1].isalpha():
            return None
        key = key[:-1]
    return key or None

_ICON = {"pending": "○", "running": "◐", "done": "●", "failed": "✗"}
_GATE_ICON = {"PASSED": "[green]✓[/green]", "WARNED": "[yellow]![/yellow]",
              "FAILED": "[red]✗[/red]", "REJECTED": "[red]⊘[/red]"}


#: ``run.log`` lines are "HH:MM:SS LEVEL   logger.name: <the emitted message>".
_LOG_LINE_RE = re.compile(r"^\d\d:\d\d:\d\d \s*\w+\s+[\w.]+: (.*)$")


def parse_log_line(line: str) -> str | None:
    """Recover the emitted message from a ``run.log`` line, or ``None``.

    The dashboard consumes the same strings whether they arrive from a live
    ``deps.emit`` or are read back off disk. Having one parser means a follower
    attached to a finished run shows exactly what the operator saw live, and it
    is why the panel can be developed against a recorded run instead of against
    a two-hour paid one.
    """
    m = _LOG_LINE_RE.match(line.rstrip("\n"))
    if not m:
        return None
    msg = m.group(1)
    return msg if msg.strip() else None


@dataclass
class LiveDashboard:
    """Consumes ``deps.emit`` strings and renders them as a live panel."""

    run_id: str = ""
    domain: str = ""
    provider: str = ""
    language: str = "zh"
    enabled: bool = True

    started: float = field(default_factory=time.time)
    status: dict[str, str] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    _phase_start: dict[str, float] = field(default_factory=dict)
    current: str = ""
    activity: list[str] = field(default_factory=list)
    metrics: list[tuple[str, str]] = field(default_factory=list)
    gates: list[tuple[str, str, str]] = field(default_factory=list)
    #: (role, ok, seconds, out_tokens, model, what it returned)
    agents: list[tuple[str, bool, str, str, str, str]] = field(default_factory=list)
    usage_fn: Any = None

    _live: Any = None
    _console: Any = None

    # -- lifecycle ----------------------------------------------------------
    def __enter__(self) -> "LiveDashboard":
        if not self.enabled:
            return self
        try:
            from rich.console import Console
            from rich.live import Live

            self._console = Console()
            if not self._console.is_terminal:
                self.enabled = False
                return self
            self._live = Live(self._render(), console=self._console,
                              refresh_per_second=4, transient=False)
            self._live.__enter__()
        except Exception:
            self.enabled = False
        return self

    def __exit__(self, *exc: Any) -> None:
        self.finish(ok=exc[0] is None)
        if self._live is not None:
            try:
                self._live.update(self._render())
                self._live.__exit__(*exc)
            except Exception:
                pass

    def finish(self, *, ok: bool = True) -> None:
        """Close the phase still marked running.

        A blocking gate halts the run by returning normally, not by raising, so
        `!!` never fires and the phase list left its last row spinning at ◐ —
        the dashboard showing "working" at the exact moment the run had stopped,
        which is the one thing it exists to prevent.
        """
        if self.current and self.status.get(self.current) == "running":
            self.status[self.current] = "done" if ok else "failed"
            self.timings.setdefault(
                self.current, time.time() - self._phase_start.get(self.current, time.time())
            )

    # -- event intake -------------------------------------------------------
    def handle(self, msg: str) -> None:
        """Parse one emitted line and update the model behind the view."""
        self._ingest(msg)
        if self._live is not None:
            try:
                self._live.update(self._render())
            except Exception:
                pass
        elif self.enabled is False and msg.strip():
            print(f"  {msg}")

    def _ingest(self, msg: str) -> None:
        now = time.time()

        m = _AGENT_RE.match(msg)
        if m:
            role, ok, secs, out, model, ret = m.groups()
            self.agents.append((role, ok == "ok", secs, out, model.strip(), ret.strip()))
            self.agents = self.agents[-8:]
            return

        m = _GATE_RE.search(msg)
        if m:
            # Keep the whole note. It used to be cut at 70 characters and then cut
            # again by the panel, which reliably removed the actionable half:
            # "…97.4%, threshold 98%" became "…Use a monolin".
            self.gates.append((m.group(1), m.group(2).upper(), m.group(3).strip()))
            return

        m = _PHASE_RE.match(msg)
        if m:
            key = declared_phase(m.group(1))
            if key is None:
                return
            if self.current and self.current != key and self.status.get(self.current) == "running":
                self.status[self.current] = "done"
                self.timings[self.current] = now - self._phase_start.get(self.current, now)
            self.current = key
            self.status[key] = "running"
            self._phase_start.setdefault(key, now)
            self.activity.clear()
            self.activity.append(msg.strip())
            return

        if msg.startswith("!!"):
            if self.current:
                self.status[self.current] = "failed"
            self.activity.append(f"[red]{escape(msg.strip()[:120])}[/red]")
            return

        for label, val in _METRIC_RE.findall(msg)[:2]:
            # No second truncation here: the regex already bounds the label, and
            # slicing it again reintroduced the mid-word cut in the metrics box
            # ("held-out structure reproduct"). The panel folds long labels.
            self.metrics.append((label.strip(), val))
        self.metrics = self.metrics[-8:]

        if msg.strip():
            # Escape before this reaches `Text.from_markup`. Agent lines carry
            # bracketed specialties — "researcher[log_reading] → 1 candidates" —
            # and Rich read those as style tags and swallowed them, so the feed
            # showed three identical "researcher" lines for three different agents.
            line = escape(msg.strip()[:140])
            # Collapse repeats. Eight concurrent batches fail identically in the
            # same second, each wrapping to two display lines, so one benign
            # already-handled error filled all six slots and pushed out the
            # progress it was competing with — the panel showing noise at exactly
            # the moment it was meant to show work.
            prev = self.activity[-1] if self.activity else ""
            base = prev.split("  ×")[0]
            if line == base:
                n = int(prev.rsplit("×", 1)[1]) + 1 if "  ×" in prev else 2
                self.activity[-1] = f"{base}  ×{n}"
            else:
                self.activity.append(line)
            self.activity = self.activity[-6:]

    # -- rendering ----------------------------------------------------------
    def _render(self) -> Any:
        """Two panes: the phase list on the left, everything volatile on the right.

        This was `Columns([left, right], expand=True)`, which sizes columns by
        their content's natural width. The right pane held three nested panels
        whose combined natural width exceeded any terminal, so Rich gave up on
        fitting them side by side and stacked everything vertically: the phase
        list rendered alone across 175 columns of empty space, and the three
        inner panels wrapped into a ragged two-then-one arrangement. A grid with
        declared column widths is what actually splits a screen.
        """
        from rich.console import Group
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text

        zh = self.language == "zh"
        elapsed = int(time.time() - self.started)
        width = getattr(self._console, "width", 175) if self._console else 175

        def T(zh_s: str, en_s: str) -> str:
            return zh_s if zh else en_s

        # -- left: the phase list -------------------------------------------
        phases = Table.grid(padding=(0, 1))
        phases.add_column(width=2)
        phases.add_column(ratio=1)
        phases.add_column(justify="right", width=7)
        for spec in PHASES:
            st = self.status.get(spec.key, "pending")
            colour = {"pending": "dim", "running": "bold cyan",
                      "done": "green", "failed": "red"}[st]
            secs = self.timings.get(spec.key)
            # `if secs` hid a phase that genuinely took under a second, which on
            # a cached re-run is most of them — the list then looked unfinished.
            t = f"{secs:.0f}s" if secs is not None else ("…" if st == "running" else "")
            phases.add_row(f"[{colour}]{_ICON[st]}[/{colour}]",
                           f"[{colour}]{spec.label(self.language)}[/{colour}]",
                           f"[dim]{t}[/dim]")
        done = sum(1 for spec in PHASES if self.status.get(spec.key) == "done")
        phases.add_row("", f"[dim]{done}/{len(PHASES)}[/dim]",
                       f"[dim]{elapsed // 60}m{elapsed % 60:02d}s[/dim]")

        # What the phase on screen is actually doing. A name says where you are;
        # it does not say what a four-minute pause is buying.
        running = next((sp for sp in PHASES if sp.key == self.current), None)
        left_body: Any = phases
        if running is not None and running.why(self.language):
            why = Table.grid(padding=(0, 0))
            why.add_column(overflow="fold")
            why.add_row(Text.from_markup(f"[cyan]{escape(running.label(self.language))}[/cyan]"))
            why.add_row(Text.from_markup(f"[dim]{escape(running.why(self.language))}[/dim]"))
            left_body = Group(phases, Text(""), why)

        # -- right: activity, metrics, gates ---------------------------------
        act = Table.grid(padding=(0, 0))
        act.add_column(overflow="fold")
        for line in self.activity[-6:]:
            act.add_row(Text.from_markup(f"[dim]{line}[/dim]"))
        if not self.activity:
            act.add_row(Text("…", style="dim"))

        met = Table.grid(padding=(0, 2))
        met.add_column(ratio=1, overflow="fold")
        met.add_column(justify="right", width=9)
        # Live vitals first. The scrape only fires on lines carrying a decimal,
        # and the longest phase of a run — 240 annotation batches — emits none, so
        # this panel sat empty for an hour of a live run while plenty was
        # happening. Errors are here because 21 of them went unremarked once.
        vitals: list[tuple[str, str]] = []
        if self.usage_fn:
            try:
                u = self.usage_fn() or {}
                out, calls = u.get("output_tokens", 0), u.get("calls", 0)
                vitals = [(T("调用", "calls"), f"{calls:,}"),
                          (T("输出 token", "out tokens"), compact(out))]
                # A rate over the first few seconds is noise, and dividing by a
                # near-zero elapsed prints something absurd with confidence.
                if elapsed >= 60:
                    vitals.append((T("token/分", "tokens/min"),
                                   compact(out / (elapsed / 60))))
                if u.get("cache_hits"):
                    vitals.append((T("缓存命中", "cached"), f"{u['cache_hits']:,}"))
                if u.get("errors"):
                    vitals.append((T("错误", "errors"), f"[red]{u['errors']:,}[/red]"))
                if u.get("failovers"):
                    vitals.append((T("已切换供应商", "failovers"),
                                   f"[yellow]{len(u['failovers'])}[/yellow]"))
            except Exception:
                vitals = []
        for label, val in vitals:
            met.add_row(f"[dim]{label}[/dim]", val if "[" in val else f"[bold]{val}[/bold]")
        if vitals and self.metrics:
            met.add_row("", "")
        for label, val in self.metrics[-6:]:
            met.add_row(f"[dim]{escape(label)}[/dim]", f"[bold]{val}[/bold]")
        if not vitals and not self.metrics:
            met.add_row("[dim]—[/dim]", "")

        gate_tbl = Table.grid(padding=(0, 1))
        gate_tbl.add_column(width=2)
        # The note is the whole point of a gate line — "held-out reproduction
        # 97.4%, threshold 98%" is what a watcher acts on, while the gate's name
        # alone only says which check ran. It wraps now instead of being cut.
        gate_tbl.add_column(ratio=1, overflow="fold")
        for name, st, note in self.gates[-7:]:
            gate_tbl.add_row(_GATE_ICON.get(st, "?"),
                             f"[dim]{escape(name)}[/dim]"
                             + (f"\n[dim]  {escape(note)}[/dim]" if note else ""))
        if not self.gates:
            gate_tbl.add_row(" ", "[dim]—[/dim]")

        agents = Table.grid(padding=(0, 1))
        agents.add_column(width=2)
        agents.add_column(ratio=1, overflow="fold")
        agents.add_column(justify="right", width=7)
        agents.add_column(justify="right", width=9)
        for role, ok, secs, out, model, ret in self.agents[-8:]:
            mark = "[green]✓[/green]" if ok else "[red]✗[/red]"
            body = f"[bold]{escape(role)}[/bold]  [dim]{escape(model)}[/dim]"
            if ret:
                body += f"\n   [dim]↳ {escape(ret)}[/dim]"
            agents.add_row(mark, body, f"[dim]{secs}s[/dim]", f"[dim]{out}[/dim]")
        if not self.agents:
            agents.add_row(" ", "[dim]—[/dim]", "", "")

        cost = ""
        if self.usage_fn:
            try:
                u = self.usage_fn()
                cost = (f"{u.get('calls', 0)} calls · "
                        f"{u.get('cache_hits', 0)} cached · "
                        f"${u.get('estimated_cost_usd', 0):.3f}")
            except Exception:
                pass

        head = (f"[bold]QMine[/bold]  {escape(self.run_id)}  ·  {escape(self.domain)}  ·  "
                f"{escape(self.provider)}" + (f"  ·  {cost}" if cost else ""))

        act_p = Panel(act, title=T("当前活动", "activity"), border_style="dim")
        agents_p = Panel(agents, title=T("智能体 (角色 · 模型 · 用时 · 输出 token)",
                                         "agents (role · model · elapsed · out)"),
                         border_style="dim")
        met_p = Panel(met, title=T("指标", "metrics"), border_style="dim")
        gate_p = Panel(gate_tbl, title=T("质量门", "gates"), border_style="dim")

        # Below roughly 120 columns there is no honest way to show two panes plus
        # a nested split, so the layout collapses by steps rather than wrapping.
        if width >= 150:
            lower = Table.grid(expand=True)
            lower.add_column(ratio=1)
            lower.add_column(ratio=2)
            lower.add_row(met_p, gate_p)
            right: Any = Group(act_p, agents_p, lower)
        else:
            right = Group(act_p, agents_p, met_p, gate_p)

        left_p = Panel(left_body, title=T("阶段", "phases"), border_style="blue")
        right_p = Panel(right, title=head, border_style="cyan")

        if width < 120:
            return Group(left_p, right_p)

        split = Table.grid(expand=True)
        split.add_column(width=_PHASE_PANE_WIDTH)
        split.add_column(ratio=1)
        split.add_row(left_p, right_p)
        return split
