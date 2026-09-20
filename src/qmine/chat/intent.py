"""Turning what someone said into a sequence of named actions.

Two routers, and the deterministic one is not a fallback nobody exercises: it
runs whenever there is no model configured, it is what the tests use, and its
answer is handed to the model as a baseline so a disagreement has to be earned.

THE ROUTER CANNOT INVENT AN ACTION. `ChatPlan.steps[].action` is a Literal over
the catalogue, so a model that wants to do something outside it fails validation
and the request falls back to `help` — which is the correct behaviour for
"I did not understand you", and is not the same thing as improvising a command.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..agents.base import Agent
from .actions import ACTION_NAMES, CATALOGUE, catalogue_for_prompt

ActionName = Literal[
    "help", "inspect", "plan", "prepare", "estimate", "run", "status",
    "compare", "render", "verify", "list_runs", "explain"]


class ChatStep(BaseModel):
    action: ActionName
    #: Parameters for this action, as strings. The executor coerces and
    #: validates; anything it does not recognise is ignored rather than passed on.
    params: dict[str, str] = Field(default_factory=dict)
    #: One short sentence, in the user's language, saying why this step.
    why: str = ""


class ChatPlan(BaseModel):
    steps: list[ChatStep] = Field(default_factory=list)
    #: What to say before doing anything. Plain language, the user's own.
    reply: str = ""
    #: Ask instead of acting when the request genuinely does not determine what
    #: to do. An assistant that guesses about a paid run is worse than one that
    #: asks.
    clarify: str = ""


class ChatRouterAgent(Agent):
    role = "chat_router"
    prompt_name = "chat_router"
    schema = ChatPlan

    def build_user(self, *, message: str = "", history: list[str] | None = None,
                   state: dict[str, Any] | None = None,
                   baseline: dict[str, Any] | None = None, **_: Any) -> str:
        import json

        return (
            "## 可以做的事（只能从这里选）\n"
            + json.dumps(catalogue_for_prompt(), ensure_ascii=False, indent=1)
            + "\n\n## 这次会话已经知道的事\n"
            + json.dumps(state or {}, ensure_ascii=False, indent=1)
            + "\n\n## 最近几轮\n" + "\n".join(history or [])
            + "\n\n## 一份只看关键词的机械判断，供对照\n"
            + json.dumps(baseline or {}, ensure_ascii=False, indent=1)
            + f"\n\n## 用户刚说的话\n{message}\n"
        )


#: A path, bounded by any separator a person might type around one. The CJK
#: punctuation matters: with only ASCII separators in the class, "我有三份导出：
#: a.xlsx、b.xlsx" matched as ONE path, prefix and all, and the whole sentence
#: was handed to `open()`.
_PATH = re.compile(
    r"[^\s,，;；:：、。'\"()（）「」【】]+\.(?:xlsx|xls|csv|parquet|jsonl|json)", re.I)
_RUNID = re.compile(r"\b((?:[a-z][a-z0-9]*-)?[a-z][a-z0-9]{1,}\d{0,3})\b", re.I)


def extract_paths(text: str) -> list[str]:
    """File paths someone pasted or typed, in order, de-duplicated."""
    return list(dict.fromkeys(_PATH.findall(text)))


def keyword_route(message: str, state: dict[str, Any]) -> ChatPlan:
    """A router with no model in it.

    Deliberately conservative: when the message names files it offers to LOOK at
    them, not to mine them, because "here are my files" is not consent to spend
    an afternoon's compute. Escalation to a paid action needs the user to say so.
    """
    m = message.strip()
    low = m.lower()
    paths = extract_paths(m)
    hits: list[tuple[int, str]] = []
    for name, action in CATALOGUE.items():
        score = sum(1 for t in action.triggers if t.lower() in low)
        if score:
            hits.append((score, name))
    hits.sort(reverse=True)
    best = hits[0][1] if hits else ""

    if paths and best in ("", "inspect", "plan", "help"):
        return ChatPlan(
            steps=[ChatStep(action="inspect", params={"inputs": ",".join(paths)},
                            why="先把每份文件量一遍，再决定怎么合并")],
            reply=f"看到 {len(paths)} 份文件。我先测量它们——不花钱，也不写任何东西。")
    if paths and best in ("prepare", "run", "compare"):
        return ChatPlan(
            steps=[ChatStep(action="inspect", params={"inputs": ",".join(paths)},
                            why="动手之前先看清楚"),
                   ChatStep(action="plan", params={"inputs": ",".join(paths)},
                            why="给出合并与清洗方案，等你确认")],
            reply="先测量再出方案。方案你看过之后我才动手。")
    if best in ("status", "compare", "render", "verify") and not _run_id(m, state):
        return ChatPlan(clarify="是哪一次运行？给我 run id，或者说「列出运行」。",
                        steps=[ChatStep(action="list_runs", why="先看看有哪些")])
    if best:
        params: dict[str, str] = {}
        rid = _run_id(m, state)
        if rid and "run_id" in CATALOGUE[best].params:
            params["run_id"] = rid
        if paths and "inputs" in CATALOGUE[best].params:
            params["inputs"] = ",".join(paths)
        if best == "explain":
            params["question"] = m
        return ChatPlan(steps=[ChatStep(action=best, params=params,
                                        why=CATALOGUE[best].summary)], reply="")
    return ChatPlan(steps=[ChatStep(action="help", why="没听懂，先说说我能做什么")],
                    reply="我没把握你要做什么。")


def _run_id(message: str, state: dict[str, Any]) -> str:
    known = set(state.get("known_runs") or [])
    for tok in re.split(r"[\s,，。；;]+", message):
        if tok in known:
            return tok
    return str(state.get("run_id") or "")


def route(message: str, state: dict[str, Any], ctx: Any = None,
          history: list[str] | None = None) -> tuple[ChatPlan, str]:
    """The plan, and where it came from."""
    base = keyword_route(message, state)
    if ctx is None:
        return base, "keyword"
    try:
        import json

        out = ChatRouterAgent(ctx).run(
            message=message, history=history or [], state=state,
            baseline=json.loads(base.model_dump_json()))
    except Exception:  # noqa: BLE001 — an unavailable router must not end the session
        return base, "keyword (router unavailable)"
    bad = [s.action for s in out.steps if s.action not in ACTION_NAMES]
    if bad:
        return base, f"keyword (router proposed unknown action {bad})"
    return out, "agent"


def summarise_paths(paths: list[str]) -> str:
    return "、".join(Path(p).name for p in paths)
