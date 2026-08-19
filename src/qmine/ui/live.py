"""A live dashboard for a run that takes tens of minutes.

The problem with a log stream is that it answers "what just happened" and not
"where am I". On a twelve-phase pipeline where one phase can take eight minutes,
a scrolling log leaves the operator unable to tell progress from a hang without
reading carefully — and it buries the two things they actually want continuously
visible: which quality gates have fired, and how much money has been spent.

So the dashboard keeps four things on screen at once: the phase list with
elapsed times, what the agents are doing right now, the metrics as they land,
and a running gate/cost tally. Everything else still goes to the log file.

Degrades deliberately: with `--quiet`, no TTY, or Rich unavailable, it falls back
to plain lines. A dashboard that breaks CI output is worse than no dashboard.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PhaseSpec:
    """One row of the phase list."""

    key: str
    label_zh: str
    label_en: str

    def label(self, lang: str) -> str:
        return self.label_zh if lang == "zh" else self.label_en


#: The pipeline, in execution order, with the names an operator recognises.
PHASES: list[PhaseSpec] = [
    PhaseSpec("p0", "工程底座", "foundation"),
    PhaseSpec("p1", "数据审计与模板挖掘", "audit & template mining"),
    PhaseSpec("p2a", "意图体系设计 (研究扇出)", "taxonomy design"),
    PhaseSpec("p2b", "金标构建 (双盲+κ)", "gold standard"),
    PhaseSpec("p3", "表征构建 (bake-off + α)", "representation"),
    PhaseSpec("p2c", "规则+ML 分类器", "classifier"),
    PhaseSpec("p2d", "对抗验证", "adversarial validation"),
    PhaseSpec("p2e", "L2 子意图层", "sub-intents"),
    PhaseSpec("p4", "算法选型 battery", "algorithm battery"),
    PhaseSpec("p5", "粒度选择 (K 三角验证)", "granularity"),
    PhaseSpec("p6", "两层层级构建", "hierarchy"),
    PhaseSpec("p7", "盲评命名与树审计", "blind naming & audit"),
    PhaseSpec("p8", "治理合并 (执行)", "governance"),
    PhaseSpec("p9", "统一度量面板", "metrics panel"),
    PhaseSpec("p10", "部署与全量打标", "deployment"),
    PhaseSpec("p11", "报告与 notebook", "reporting"),
    PhaseSpec("p12", "维护循环", "maintenance"),
]

_PHASE_RE = re.compile(r"^\s*(P\d+[a-e]?)\b", re.I)
_GATE_RE = re.compile(r"gate\s+(\S+):\s*(PASSED|WARNED|FAILED|REJECTED)\s*—?\s*(.*)", re.I)
_AGENT_RE = re.compile(r"^\s{2,}(researcher|namer|annotator|referee|adversary|auditor)\[?([^\]\s]*)\]?")

_ICON = {"pending": "○", "running": "◐", "done": "●", "failed": "✗"}
_GATE_ICON = {"PASSED": "[green]✓[/green]", "WARNED": "[yellow]![/yellow]",
              "FAILED": "[red]✗[/red]", "REJECTED": "[red]⊘[/red]"}


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
        if self._live is not None:
            try:
                self._live.update(self._render())
                self._live.__exit__(*exc)
            except Exception:
                pass

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

        m = _GATE_RE.search(msg)
        if m:
            self.gates.append((m.group(1), m.group(2).upper(), m.group(3)[:70]))
            return

        m = _PHASE_RE.match(msg)
        if m:
            key = m.group(1).lower()
            if self.current and self.current != key and self.status.get(self.current) == "running":
                self.status[self.current] = "done"
                self.timings[self.current] = now - self._phase_start.get(self.current, now)
            self.current = key
            self.status[key] = "running"
            self._phase_start[key] = now
            self.activity.clear()
            self.activity.append(msg.strip())
            return

        if msg.startswith("!!"):
            if self.current:
                self.status[self.current] = "failed"
            self.activity.append(f"[red]{msg.strip()[:90]}[/red]")
            return

        # a metric-looking line: "  something 0.873"
        nums = re.findall(r"([A-Za-z一-鿿_ ]{3,28})\s+([0-9]*\.[0-9]+)", msg)
        for label, val in nums[:2]:
            self.metrics.append((label.strip()[:26], val))
        self.metrics = self.metrics[-8:]

        if msg.strip():
            self.activity.append(msg.strip()[:100])
            self.activity = self.activity[-6:]

    # -- rendering ----------------------------------------------------------
    def _render(self) -> Any:
        from rich.columns import Columns
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text

        zh = self.language == "zh"
        elapsed = int(time.time() - self.started)

        phases = Table.grid(padding=(0, 1))
        phases.add_column(width=2)
        phases.add_column(ratio=1)
        phases.add_column(justify="right", width=7)
        for spec in PHASES:
            st = self.status.get(spec.key, "pending")
            icon = _ICON[st]
            colour = {"pending": "dim", "running": "bold cyan",
                      "done": "green", "failed": "red"}[st]
            secs = self.timings.get(spec.key)
            t = f"{secs:.0f}s" if secs else ("…" if st == "running" else "")
            phases.add_row(f"[{colour}]{icon}[/{colour}]",
                           f"[{colour}]{spec.label(self.language)}[/{colour}]",
                           f"[dim]{t}[/dim]")

        act = Table.grid(padding=(0, 0))
        for line in self.activity[-6:]:
            act.add_row(Text.from_markup(f"[dim]{line}[/dim]"))
        if not self.activity:
            act.add_row(Text("…", style="dim"))

        met = Table.grid(padding=(0, 2))
        met.add_column(ratio=1)
        met.add_column(justify="right")
        for label, val in self.metrics[-8:]:
            met.add_row(f"[dim]{label}[/dim]", f"[bold]{val}[/bold]")
        if not self.metrics:
            met.add_row("[dim]—[/dim]", "")

        gate_tbl = Table.grid(padding=(0, 1))
        gate_tbl.add_column(width=2)
        gate_tbl.add_column(ratio=1)
        for name, st, note in self.gates[-7:]:
            # The note is the whole point of a gate line — "held-out reproduction
            # 97.4%, threshold 98%" is what a watcher acts on, while the gate's
            # name alone only says which check ran. It was being captured and
            # dropped.
            gate_tbl.add_row(_GATE_ICON.get(st, "?"),
                             f"[dim]{name}[/dim]" + (f"\n   [dim]{note}[/dim]" if note else ""))
        if not self.gates:
            gate_tbl.add_row(" ", "[dim]—[/dim]")

        cost = ""
        if self.usage_fn:
            try:
                u = self.usage_fn()
                cost = (f"{u.get('calls', 0)} calls · "
                        f"{u.get('cache_hits', 0)} cached · "
                        f"${u.get('estimated_cost_usd', 0):.3f}")
            except Exception:
                pass

        head = (f"[bold]QMine[/bold]  {self.run_id}  ·  {self.domain}  ·  "
                f"{self.provider}  ·  {elapsed // 60}m{elapsed % 60:02d}s"
                + (f"  ·  {cost}" if cost else ""))

        left = Panel(phases, title="阶段" if zh else "phases", border_style="blue")
        right = Panel(
            Columns([
                Panel(act, title="当前活动" if zh else "activity", border_style="dim"),
                Panel(met, title="指标" if zh else "metrics", border_style="dim"),
                Panel(gate_tbl, title="质量门" if zh else "gates", border_style="dim"),
            ], expand=True),
            title=head, border_style="cyan",
        )
        return Columns([left, right], expand=True)
