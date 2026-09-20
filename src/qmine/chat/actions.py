"""Everything the assistant is allowed to do, and what each one costs.

THE MODEL CHOOSES FROM THIS LIST AND FILLS IN PARAMETERS. IT DOES NOT ACT.
The separation matters for one specific reason: a run costs real money and takes
hours, and a corpus prepared under the wrong assumption produces a study that
looks finished and is about the wrong rows. So the conversation resolves to a
named action with typed arguments, the program decides whether that action needs
a human's agreement, and the human sees the exact command before anything runs.

Actions are marked `spends` when they call a paid model and `writes` when they
put something on disk. Everything marked either way stops for confirmation; the
rest runs straight away, because asking permission to list a directory trains
people to say yes without reading.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

#: The action vocabulary. A name outside this set cannot be produced by the
#: router's schema, so an unrecognised intent becomes `help`, never an
#: improvised command.
ACTION_NAMES = [
    "help", "inspect", "plan", "prepare", "estimate", "run", "status",
    "compare", "render", "verify", "list_runs", "explain",
]


@dataclass
class Action:
    name: str
    summary: str
    #: What a user might say to mean this. Used by the deterministic router and
    #: shown in `help`, so the two can never describe different things.
    triggers: list[str]
    params: dict[str, str] = field(default_factory=dict)
    spends: bool = False
    writes: bool = False
    #: The equivalent command line, as a format string over the parameters.
    #: Always shown, so the conversation teaches the CLI instead of hiding it.
    command: str = ""

    @property
    def needs_confirmation(self) -> bool:
        return self.spends or self.writes


CATALOGUE: dict[str, Action] = {
    "help": Action(
        "help", "说明我能做什么，以及每一步会不会花钱",
        ["help", "帮助", "怎么用", "能做什么", "what can you do", "usage", "?"]),
    "inspect": Action(
        "inspect", "测量几份导出文件：列、长度、重复率、包装前缀、权重分布、互相重合度",
        ["inspect", "看看", "检查", "测量", "profile", "这几个文件", "有什么数据"],
        params={"inputs": "逗号分隔的文件路径"},
        command="qmine prepare {inputs} --dry-run"),
    "plan": Action(
        "plan", "给出合并与清洗方案（不执行）",
        ["plan", "方案", "计划", "打算怎么做", "怎么清洗", "dry run", "dry-run"],
        params={"inputs": "文件路径", "axis": "time | stratum"},
        command="qmine prepare {inputs} --dry-run"),
    "prepare": Action(
        "prepare", "执行方案，把几份导出合并成一份可挖掘的语料",
        ["prepare", "准备", "合并", "清洗", "pool", "merge", "build corpus"],
        params={"inputs": "文件路径", "out": "输出目录", "axis": "time | stratum"},
        writes=True,
        command="qmine prepare {inputs} --out {out} --axis {axis}"),
    "estimate": Action(
        "estimate", "估算一次运行要调用多少次模型、大概多少钱（不花钱）",
        ["estimate", "多少钱", "成本", "cost", "how much", "预算", "贵不贵"],
        params={"config": "配置文件"},
        command="qmine models"),
    "run": Action(
        "run", "跑完整的十二阶段挖掘流程",
        ["run", "跑", "开始", "挖掘", "start", "launch", "mine"],
        params={"inputs": "文件路径", "run_id": "运行编号", "domain": "领域档案",
                "config": "配置文件", "fast": "是否 fast 模式", "prepare": "是否先准备语料"},
        spends=True, writes=True,
        command="qmine run --input {inputs} --run-id {run_id} --domain {domain}"),
    "status": Action(
        "status", "看一次运行到哪了、有没有被闸门拦住",
        ["status", "状态", "跑到哪了", "进度", "watch", "怎么样了", "好了吗"],
        params={"run_id": "运行编号"},
        command="qmine watch {run_id}"),
    "compare": Action(
        "compare", "把一次已完成运行的各个快照做逐类对比，出表、出图、出报告（不花钱）",
        ["compare", "对比", "比较", "跨快照", "差异", "drift", "快照"],
        params={"run_id": "运行编号", "axis": "time | stratum"},
        writes=True,
        command="qmine compare {run_id} --axis {axis}"),
    "render": Action(
        "render", "用已有产物重新生成交付文档，开一个新代次",
        ["render", "重新生成", "重出报告", "rebuild", "重渲染"],
        params={"run_id": "运行编号", "agents": "是否重跑写作 agent"},
        writes=True,
        command="qmine render {run_id}"),
    "verify": Action(
        "verify", "对一次已完成的运行跑机械检查",
        ["verify", "检查运行", "核对", "check run", "验证"],
        params={"run_id": "运行编号", "control": "对照运行（必须给）"},
        command="python tools/verify_run.py runs/{run_id}/gen01 runs/{control}/gen01"),
    "list_runs": Action(
        "list_runs", "列出已有的运行",
        ["list", "列出", "有哪些运行", "runs", "哪些跑过"],
        command="ls runs/"),
    "explain": Action(
        "explain", "解释这个程序的某个概念或某条规则",
        ["explain", "解释", "什么是", "为什么", "what is", "why"],
        params={"question": "你的问题"}),
}


def catalogue_for_prompt() -> list[dict[str, Any]]:
    """The catalogue as the router sees it — names, meanings, costs."""
    return [{"action": a.name, "summary": a.summary,
             "params": a.params, "spends_money": a.spends,
             "writes_files": a.writes, "triggers": a.triggers[:6]}
            for a in CATALOGUE.values()]


def render_help() -> str:
    lines = ["我能做这些事。**花钱的和会写文件的都会先问你**，其它的直接做。", ""]
    lines.append("| 做什么 | 说法 | 花钱 | 写文件 |")
    lines.append("|---|---|---|---|")
    for a in CATALOGUE.values():
        lines.append(f"| {a.summary} | {'、'.join(a.triggers[:3])} | "
                     f"{'是' if a.spends else '否'} | {'是' if a.writes else '否'} |")
    lines += ["", "你不用记命令。每做一步我都会把等价的命令行打出来，"
              "想自己跑的时候照抄就行。"]
    return "\n".join(lines)


#: Filled by `session`, so `actions` stays free of imports it does not need.
Runner = Callable[..., Any]
