"""The conversation: route, show, confirm, execute, remember.

THE CONFIRMATION IS NOT A FORMALITY AND IT IS NOT ASKED FOR EVERYTHING. Asking
before listing a directory teaches people to type `y` without reading, and the
one prompt that matters — the one before a run that costs money and takes hours
— then gets the same reflex. So free, reversible actions just happen, and the
ones that spend or write stop with the exact command, the cost if there is one,
and what will be on disk afterwards.

Every executed step prints its command line. The point is not decoration: a user
who has seen `qmine compare med-pool8 --axis stratum` five times can run it
without this interface, and an interface you can stop needing is the only honest
kind.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .actions import CATALOGUE, render_help
from .intent import ChatStep, route


@dataclass
class ChatState:
    """What the conversation has learned. Small on purpose."""

    run_root: str = "runs"
    run_id: str = ""
    inputs: list[str] = field(default_factory=list)
    prepared: str = ""
    axis: str = ""
    domain: str = ""
    config: str = ""
    known_runs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v}


class Session:
    """One conversation. `ask` is injected so the tests can drive it."""

    def __init__(self, *, console: Any, state: ChatState | None = None,
                 ctx: Any = None, ask: Callable[[str], str] | None = None,
                 auto_confirm: bool = False) -> None:
        self.console = console
        self.state = state or ChatState()
        self.ctx = ctx
        self.ask = ask or (lambda q: input(q))
        self.auto_confirm = auto_confirm
        self.history: list[str] = []
        self.state.known_runs = _list_runs(self.state.run_root)

    # ------------------------------------------------------------------ turn
    def handle(self, message: str) -> list[dict[str, Any]]:
        plan, source = route(message, self.state.to_dict(), self.ctx, self.history[-6:])
        self.history.append(f"用户：{message}")
        if plan.reply:
            self.console.print(plan.reply)
        if plan.clarify:
            self.console.print(f"[yellow]{plan.clarify}[/yellow]")
        self.console.print(f"[dim]（路由：{source}）[/dim]")
        results = []
        for step in plan.steps:
            r = self._step(step)
            results.append(r)
            self.history.append(f"助手：{step.action} → {r.get('status')}")
            if r.get("status") in ("declined", "error"):
                break
        return results

    # ------------------------------------------------------------------ step
    def _step(self, step: ChatStep) -> dict[str, Any]:
        action = CATALOGUE.get(step.action)
        if action is None:
            return {"action": step.action, "status": "error",
                    "detail": "not an action this system has"}
        params = dict(step.params)
        self._fill(step.action, params)
        cmd = _command(action, params)
        if step.why:
            self.console.print(f"[dim]· {step.why}[/dim]")
        if cmd:
            self.console.print(f"[dim]等价命令：[/dim] [bold]{cmd}[/bold]")

        if action.needs_confirmation and not self.auto_confirm:
            note = []
            if action.spends:
                note.append("[red]会调用付费模型[/red]")
            if action.writes:
                note.append("会写文件")
            answer = self.ask(f"{action.summary}（{'，'.join(note)}）。执行吗？[y/N] ")
            if str(answer).strip().lower() not in ("y", "yes", "是", "好", "确认"):
                self.console.print("[dim]没执行。[/dim]")
                return {"action": step.action, "status": "declined", "command": cmd}

        try:
            detail = getattr(self, f"_do_{step.action}")(params)
        except Exception as exc:  # noqa: BLE001
            self.console.print(f"[red]{type(exc).__name__}[/red]: {exc}")
            return {"action": step.action, "status": "error", "command": cmd,
                    "detail": f"{type(exc).__name__}: {exc}"}
        return {"action": step.action, "status": "ok", "command": cmd, "detail": detail}

    def _fill(self, action: str, params: dict[str, str]) -> None:
        """Fill from what the conversation already knows, never over what was said."""
        if "inputs" in CATALOGUE[action].params and not params.get("inputs") \
                and self.state.inputs:
            params["inputs"] = ",".join(self.state.inputs)
        if "run_id" in CATALOGUE[action].params and not params.get("run_id") \
                and self.state.run_id:
            params["run_id"] = self.state.run_id
        if "axis" in CATALOGUE[action].params and not params.get("axis") and self.state.axis:
            params["axis"] = self.state.axis
        if "domain" in CATALOGUE[action].params and not params.get("domain") \
                and self.state.domain:
            params["domain"] = self.state.domain

    # --------------------------------------------------------------- actions
    def _do_help(self, _params: dict[str, str]) -> str:
        self.console.print(render_help())
        return "help"

    def _do_list_runs(self, _params: dict[str, str]) -> str:
        runs = _list_runs(self.state.run_root)
        self.state.known_runs = runs
        self.console.print("已有运行：" + ("、".join(runs) if runs else "（还没有）"))
        return f"{len(runs)} runs"

    def _do_explain(self, params: dict[str, str]) -> str:
        self.console.print(
            "这个程序把一份查询日志跑成两套标签：自上而下的意图体系（由 agent 写、"
            "金标准训练出分类器）和自下而上的聚类树（嵌入 + 聚类 + 治理）。"
            "给它多份导出，它会把它们**合成一次运行**——一次运行只有一套体系，"
            "所以快照之间才能比；分开跑的两次运行标签互不相通（实测共享 0 个类目编码）。")
        q = params.get("question", "")
        if q:
            self.console.print(f"[dim]（你问的是：{q}。更细的问题请看 CLAUDE.md "
                               "与 Universal_Query_Mining_Playbook.md。）[/dim]")
        return "explained"

    def _do_inspect(self, params: dict[str, str]) -> str:
        from ..prepare import profile_inputs

        paths = _paths(params)
        if not paths:
            self.console.print("[yellow]没有可测量的文件。[/yellow]")
            return "no inputs"
        profiles = profile_inputs(paths)
        self.state.inputs = paths
        for pr in profiles:
            self.console.print(
                f"[cyan]{Path(pr.path).name}[/cyan] {pr.n_rows:,} 行 · "
                f"文本列 {pr.text_candidates[:1] or '?'} · "
                f"权重列 {pr.weight_candidates[:1] or '无'} · "
                f"快照名 {pr.suggested_tag} · 中位长度 {pr.len_median:g}")
            for n in pr.notes:
                self.console.print(f"    · {n}")
            if pr.repeated_prefixes:
                self.console.print(
                    "    · 高覆盖前缀（多半是产品自己的包装）："
                    + "、".join(f"「{p}」{100 * f:.1f}%" for p, f in pr.repeated_prefixes[:3]))
        return f"{len(profiles)} profiled"

    def _do_plan(self, params: dict[str, str]) -> str:
        from ..prepare import heuristic_plan, profile_inputs
        from ..prepare.agent import plan_corpus

        paths = _paths(params) or self.state.inputs
        if not paths:
            self.console.print("[yellow]先告诉我要合并哪些文件。[/yellow]")
            return "no inputs"
        profiles = profile_inputs(paths)
        axis = params.get("axis") or self.state.axis or None
        plan, meta = (plan_corpus(profiles, self.ctx, axis=axis) if self.ctx
                      else (heuristic_plan(profiles, axis=axis), {"source": "heuristic"}))
        self.state.inputs = paths
        self.state.axis = plan.comparison_axis
        self._print_plan(plan, meta)
        return f"plan from {meta.get('source')}"

    def _print_plan(self, plan: Any, meta: dict[str, Any]) -> None:
        self.console.print(f"对比轴 [bold]{plan.comparison_axis}[/bold]"
                           f" · 信心 {plan.confidence} · 方案来源 {meta.get('source')}")
        self.console.print(f"[dim]{plan.rationale}[/dim]")
        for spec in plan.inputs:
            self.console.print(f"[cyan]{Path(spec.path).name}[/cyan] → 快照 "
                               f"[bold]{spec.snapshot}[/bold]"
                               + (f" · 组 {spec.group}" if spec.group else ""))
            for op in spec.ops:
                self.console.print(f"   · {op.op}"
                                   + (f" `{op.pattern}`" if op.pattern else "")
                                   + (f" — {op.why}" if op.why else ""))
        for c in plan.concerns:
            self.console.print(f"[yellow]⚠[/yellow] {c}")

    def _do_prepare(self, params: dict[str, str]) -> str:
        from ..prepare import execute, heuristic_plan, profile_inputs
        from ..prepare.agent import plan_corpus

        paths = _paths(params) or self.state.inputs
        if not paths:
            self.console.print("[yellow]先告诉我要合并哪些文件。[/yellow]")
            return "no inputs"
        out = params.get("out") or "prepared"
        profiles = profile_inputs(paths)
        axis = params.get("axis") or self.state.axis or None
        plan, meta = (plan_corpus(profiles, self.ctx, axis=axis) if self.ctx
                      else (heuristic_plan(profiles, axis=axis), {"source": "heuristic"}))
        rep = execute(plan, out)
        self.state.prepared = rep["corpus"]
        self.state.axis = plan.comparison_axis
        for i in rep["inputs"]:
            self.console.print(f"  {Path(i['path']).name}: {i['原始行数']:,} → "
                               f"{i['进入挖掘']:,}（剔除 {i['剔除占比%']}%）")
            for r in i["拒绝执行"]:
                self.console.print(f"    [yellow]⛔ 拒绝执行[/yellow] {r['op']}: {r['reason']}")
        self.console.print(f"[green]→ {rep['corpus']}[/green] {rep['n_mined']:,} 行")
        return rep["corpus"]

    def _do_estimate(self, params: dict[str, str]) -> str:
        return self._shell(["qmine", "models"]
                           + (["--config", params["config"]] if params.get("config") else []))

    def _do_run(self, params: dict[str, str]) -> str:
        args = ["qmine", "run"]
        inp = params.get("inputs") or self.state.prepared or ",".join(self.state.inputs)
        if not inp:
            self.console.print("[yellow]没有输入。[/yellow]")
            return "no inputs"
        args += ["--input", inp]
        rid = params.get("run_id") or self.state.run_id
        if rid:
            args += ["--run-id", rid]
            self.state.run_id = rid
        if params.get("domain") or self.state.domain:
            args += ["--domain", params.get("domain") or self.state.domain]
        if params.get("config") or self.state.config:
            args += ["--config", params.get("config") or self.state.config]
        if str(params.get("fast", "")).lower() in ("1", "true", "yes", "是"):
            args.append("--fast")
        if str(params.get("prepare", "")).lower() in ("1", "true", "yes", "是"):
            args.append("--prepare")
        args += ["--run-root", self.state.run_root]
        return self._shell(args)

    def _do_status(self, params: dict[str, str]) -> str:
        rid = params.get("run_id") or self.state.run_id
        if not rid:
            self.console.print("[yellow]是哪一次运行？[/yellow]")
            return "no run id"
        path = Path(self.state.run_root) / rid
        gens = sorted(p.name for p in path.glob("gen*")) if path.is_dir() else []
        if not gens:
            self.console.print(f"[yellow]{path} 下还没有代次。[/yellow]")
            return "no generations"
        summ = path / gens[-1] / "run_summary.json"
        if not summ.exists():
            self.console.print(f"{rid}/{gens[-1]}：还在跑（没有 run_summary.json）。")
            return "running"
        d = json.loads(summ.read_text(encoding="utf-8"))
        gates = d.get("gates") or {}
        # `warned` is not `failed`. A warn-only gate records something a reader
        # should look at; counting it as a failure makes every healthy run look
        # broken and trains people to ignore the count.
        def _status(v):
            return str(getattr(v, "status", v.get("status") if isinstance(v, dict) else ""))

        bad = [k for k, v in gates.items() if _status(v) == "failed"]
        warned = [k for k, v in gates.items() if _status(v) == "warned"]
        self.console.print(
            f"{rid}/{gens[-1]}：{'已停在闸门' if d.get('halted') else '跑完'}，"
            f"阶段 {len(d.get('completed_phases') or [])} 个，闸门 {len(gates)} 个"
            + (f"，[red]未通过 {len(bad)} 个[/red]：{'、'.join(bad[:5])}" if bad else "")
            + (f"，告警 {len(warned)} 个：{'、'.join(warned[:5])}" if warned else "")
            + ("" if (bad or warned) else "，全部通过或跳过"))
        self.state.run_id = rid
        return "ok"

    def _do_compare(self, params: dict[str, str]) -> str:
        rid = params.get("run_id") or self.state.run_id
        if not rid:
            self.console.print("[yellow]是哪一次运行？[/yellow]")
            return "no run id"
        args = ["qmine", "compare", rid, "--run-root", self.state.run_root]
        axis = params.get("axis") or self.state.axis
        if axis:
            args += ["--axis", axis]
        self.state.run_id = rid
        return self._shell(args)

    def _do_render(self, params: dict[str, str]) -> str:
        rid = params.get("run_id") or self.state.run_id
        if not rid:
            return "no run id"
        args = ["qmine", "render", rid, "--run-root", self.state.run_root]
        if str(params.get("agents", "")).lower() in ("1", "true", "yes", "是"):
            args.append("--agents")
        return self._shell(args)

    def _do_verify(self, params: dict[str, str]) -> str:
        rid = params.get("run_id") or self.state.run_id
        control = params.get("control", "")
        if not rid:
            return "no run id"
        if not control:
            # A harness that passes on one run proves nothing about the harness.
            self.console.print("[yellow]需要一个对照运行[/yellow]：只在一次运行上通过的检查，"
                               "证明不了这套检查本身有效。给我一个已知有问题的旧运行做对照。")
            return "no control"
        root = Path(self.state.run_root)
        return self._shell([sys.executable, "tools/verify_run.py",
                            str(root / rid / "gen01"), str(root / control / "gen01")])

    # ---------------------------------------------------------------- shell
    def _shell(self, args: list[str]) -> str:
        self.console.print(f"[dim]$ {' '.join(args)}[/dim]")
        proc = subprocess.run(args, capture_output=True, text=True)
        out = (proc.stdout or "") + (proc.stderr or "")
        for line in out.splitlines()[-40:]:
            self.console.print(line)
        self.state.known_runs = _list_runs(self.state.run_root)
        return f"exit {proc.returncode}"


def _paths(params: dict[str, str]) -> list[str]:
    raw = params.get("inputs", "")
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def _command(action: Any, params: dict[str, str]) -> str:
    if not action.command:
        return ""
    try:
        return action.command.format(**{k: params.get(k, f"<{k}>") for k in action.params})
    except (KeyError, IndexError):
        return action.command


def _list_runs(run_root: str) -> list[str]:
    p = Path(run_root)
    if not p.is_dir():
        return []
    return sorted(d.name for d in p.iterdir()
                  if d.is_dir() and any(d.glob("gen*")))
