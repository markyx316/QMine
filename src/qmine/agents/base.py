"""The agent contract.

Every agent in this team is the same shape: a role name, a versioned prompt
loaded from disk, a Pydantic output schema, and a single call through the
registry.  Uniformity buys three things that matter more than flexibility here.

*Auditability.*  Prompts live in ``agents/prompts/*.md`` as files, so their
hashes go into the run manifest and "which prompt produced this taxonomy" has an
answer six months later.

*Isolation.*  An agent receives exactly the context its ``build_user`` assembles.
Nothing leaks in from a shared conversation, because there is no shared
conversation — the parallel naming shards in Phase 7 depend on this being
structurally true rather than carefully arranged.

*Testability.*  Swap the registry for the offline model and every agent still
returns a schema-valid object, so the graph can be tested end to end without a
network.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from ..artifacts import ArtifactStore
from ..config import QMineConfig
from ..llm.registry import ModelRegistry
from ..memory.context import BlindnessFirewall
from ..memory.store import QMineMemory

log = logging.getLogger("qmine.agents")

T = TypeVar("T", bound=BaseModel)

PROMPT_DIR = Path(__file__).parent / "prompts"


@dataclass
class AgentContext:
    """Everything an agent is allowed to reach.

    Passing this explicitly, rather than letting agents import globals, is what
    makes "which agent could have seen the legacy labels?" a question with a
    checkable answer.
    """

    cfg: QMineConfig
    registry: ModelRegistry
    store: ArtifactStore
    memory: QMineMemory
    firewall: BlindnessFirewall = field(default_factory=BlindnessFirewall)
    run_id: str = ""

    def recall(self, situation: str) -> str:
        try:
            return self.memory.context_block(situation)
        except Exception as exc:  # noqa: BLE001
            log.debug("memory recall failed: %s", exc)
            return ""


_PROMPT_CACHE: dict[str, tuple[str, str]] = {}


def load_prompt(name: str) -> tuple[str, str]:
    """Return ``(text, sha)`` for a prompt file.  The sha goes into the manifest."""
    if name in _PROMPT_CACHE:
        return _PROMPT_CACHE[name]
    path = PROMPT_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"prompt {name!r} not found at {path}")
    text = path.read_text(encoding="utf-8")
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    _PROMPT_CACHE[name] = (text, sha)
    return text, sha


def prompt_manifest() -> dict[str, str]:
    """Hashes of every prompt shipped, for the run manifest."""
    return {
        p.stem: hashlib.sha256(p.read_text(encoding="utf-8").encode()).hexdigest()[:12]
        for p in sorted(PROMPT_DIR.glob("*.md"))
    }


class Agent:
    """One role, one prompt, one output schema.

    Subclasses override :meth:`build_user` to assemble evidence.  They do not
    override the calling convention, which is what keeps caching, budgeting,
    repair, and usage accounting uniform across the team.
    """

    role: str = "generic"
    prompt_name: str = ""
    schema: type[BaseModel] | None = None
    #: Prepended to the system prompt for every agent, without exception.
    charter: str = (
        "You are one member of a data-mining agent team working a query-intent "
        "mining pipeline. Two standing rules override any instinct to be helpful:\n"
        "1. EVIDENCE OVER FLUENCY. Every claim you make must be traceable to "
        "something in the material you were given. If the evidence does not "
        "support a confident answer, say so and say what would settle it. A "
        "hedged accurate answer is worth more than a clean invented one.\n"
        "2. STAY IN YOUR LANE. You were given exactly the context your role "
        "needs. If you find yourself wanting information you were not given, "
        "that is usually deliberate — report the gap, do not guess around it."
    )

    def __init__(self, ctx: AgentContext, *, suffix: str = "") -> None:
        self.ctx = ctx
        self.suffix = suffix
        self.prompt_text, self.prompt_sha = load_prompt(self.prompt_name) if self.prompt_name else ("", "")

    # -- to override --------------------------------------------------------
    def build_user(self, **kwargs: Any) -> str:
        raise NotImplementedError

    def build_system(self, **kwargs: Any) -> str:
        return f"{self.charter}\n{self._language_directive()}\n\n---\n\n{self.prompt_text}"

    def _language_directive(self) -> str:
        """Pin the output language to the one the deliverables are written in.

        Without this, a model reading Chinese queries answers in English roughly
        as often as not — and every category name, definition sentence and
        rationale then has to be translated before it can be checked against the
        corpus by the people who own it. Observed live: a researcher returned
        "Character Pronunciation Lookup" for a corpus of 拼音 queries.

        Field *names* and `code` values stay English; only human-readable prose
        follows the corpus.
        """
        lang = getattr(self.ctx.cfg, "report_language", "zh")
        if lang != "zh":
            return ("\n3. WRITE IN ENGLISH. All names, definitions and rationales are read "
                    "by the team that owns this corpus.")
        return (
            "\n3. 用中文书写。所有类目名称、定义句 (user_need)、理由、备注一律使用中文 — "
            "这些文字要交给拥有该语料的团队直接使用, 换一种语言就无法与数据对照核验。"
            "例外: 字段名与 `code` 字段保持英文 snake_case。"
        )

    # -- the call -----------------------------------------------------------
    def run(self, **kwargs: Any) -> Any:
        system = self.build_system(**kwargs)
        user = self.build_user(**kwargs)
        role = f"{self.role}{self.suffix}" if self.suffix else self.role
        out = self.ctx.registry.complete(role, system, user, schema=self.schema)
        return self.postprocess(out, **kwargs)

    def postprocess(self, out: Any, **kwargs: Any) -> Any:
        return out

    # -- helpers ------------------------------------------------------------
    @property
    def provenance(self) -> dict[str, str]:
        return {
            "role": self.role,
            "prompt": self.prompt_name,
            "prompt_sha": self.prompt_sha,
            "provider": self.ctx.registry.provider,
        }



def _is_response_format_rejection(exc: Exception) -> bool:
    """Is this the "tools yes, structured output no" refusal, not a real outage?

    Matched on the phrases providers actually return. Kept narrow on purpose: a
    genuine outage must NOT be retried as if it were a capability quirk.
    """
    msg = str(exc).lower()
    if "400" not in msg and "invalid_request" not in msg:
        return False
    return any(k in msg for k in
               ("tool_choice", "response_format", "structured output",
                "json_schema", "does not support"))

def _tool_loop_usage(result: Any) -> dict[str, int]:
    """Total the tokens every model turn in a tool loop actually spent."""
    totals = {"input_tokens": 0, "output_tokens": 0}
    for msg in (result or {}).get("messages", []) or []:
        usage = getattr(msg, "usage_metadata", None) or {}
        for key in totals:
            totals[key] += int(usage.get(key, 0) or 0)
    return totals


class ToolAgent(Agent):
    """An agent that can call tools before answering.

    Phase 2a's literature angle instructs a researcher to ground its proposals in
    published work. Without tools that instruction produces confident citations
    the agent cannot have checked — worse than no literature angle, because the
    design record then carries unverifiable evidence.

    So this subclass runs a real tool loop via ``create_agent`` and still returns
    a validated structured object. It degrades to the single-shot path whenever
    tools are unavailable — offline mode, no provider, or a model without tool
    support — and records which path it took, so a report can say whether its
    citations were fetched or recalled.
    """

    tools: list[Any] = []
    max_tool_iterations: int = 12

    def run(self, **kwargs: Any) -> Any:
        system = self.build_system(**kwargs)
        user = self.build_user(**kwargs)
        role = f"{self.role}{self.suffix}" if self.suffix else self.role

        if not self.tools or self.ctx.registry.is_offline:
            self.used_tools = False
            out = self.ctx.registry.complete(role, system, user, schema=self.schema)
            return self.postprocess(out, **kwargs)

        # Replay first. Without this the tool path re-fetched live pages on every
        # run, which changed this agent's answer, which changed the architect's
        # prompt, which missed ITS cache — and so on through every annotation call
        # downstream. It is why "resume after a failure" replayed almost nothing
        # twice today, despite the entries being on disk the whole time.
        cached = self.ctx.registry.replay_external_turn(
            role, "deep", system, user, self.schema)
        if cached is not None:
            self.used_tools = True
            return self.postprocess(cached, **kwargs)

        t0 = time.time()
        try:
            from langchain.agents import create_agent

            model = self.ctx.registry.get(role)
            agent = create_agent(
                model, tools=self.tools, system_prompt=system,
                response_format=self.schema,
            )
            # STREAM, DO NOT INVOKE. `recursion_limit` is 2 x tool calls (measured:
            # 24 -> exactly 12 searches), and `invoke()` raises GraphRecursionError
            # carrying NO state — so a researcher that spent all twelve searches
            # had every finding discarded and then answered from parametric
            # knowledge alone. That is not "cut off early", it is a total loss of
            # the work, and it hit 2 of 5 researchers on live38 and 1 of 5 on
            # live39. Streaming keeps the last state, so the findings survive.
            # Verified equivalent to `invoke()` when the loop completes normally.
            result, cutoff = self._stream_tool_loop(agent, user)
            parsed = (result or {}).get("structured_response")
            if parsed is None:
                parsed = self._salvage_partial(role, system, user, result, cutoff)
            if parsed is None:
                raise cutoff or ValueError("tool agent returned no structured response")
            self.used_tools = True
            # A tool loop is several model calls, and this recorded a hardcoded
            # zero for all of them: every web-researching agent's spend was
            # invisible in `run_summary.json`, and — the part that matters — the
            # ledger's output-token ceiling could not see the one code path that
            # can iterate. A runaway search loop was the single thing the budget
            # guard was blind to. The messages carry their own usage; sum it.
            usage = _tool_loop_usage(result)
            self.ctx.registry.ledger.record(role, **usage)
            # The tool path bypasses `registry.complete`, so it must announce
            # itself or the web-researching agents stay invisible to a watcher —
            # the same gap that hid their token spend.
            # `record_external_turn` files the cache entry and the transcript row
            # through `_store`, which announces the turn itself. Calling
            # `report_call` here as well printed every web-researching agent twice
            # in the log and the agents panel.
            self.ctx.registry.record_external_turn(
                role, "deep", system, user, parsed, time.time() - t0)
            return self.postprocess(parsed, **kwargs)
        except Exception as exc:  # noqa: BLE001
            # SOME MODELS SUPPORT TOOLS BUT NOT `response_format` ALONGSIDE THEM.
            # LangChain implements a structured response by forcing a
            # `tool_choice`, and a thinking-mode model can reject exactly that:
            # deepseek-v4-pro answers 400 "Thinking mode does not support this
            # tool_choice" while `bind_tools` and a plain agent loop both work.
            #
            # Falling straight through to a single call throws the tools away and
            # the agent still returns plausible candidates from parametric
            # knowledge — so a WEB researcher that never searched looks identical
            # in the phase result. live38 hit its tool cap at step 24 (searching
            # hard) and live39 hit this 400 at step 0 (never searched); both
            # logged "tool loop failed" and both produced candidates.
            #
            # So: retry the loop WITHOUT the structured response, then parse the
            # final message into the schema separately. Tools survive.
            if _is_response_format_rejection(exc):
                try:
                    salvaged = self._tool_loop_without_schema(role, system, user, t0)
                    if salvaged is not None:
                        log.warning("tool loop for %s rejected response_format (%s); "
                                    "re-ran WITHOUT it so the tools still run",
                                    role, str(exc)[:90])
                        return self.postprocess(salvaged, **kwargs)
                except Exception as exc2:  # noqa: BLE001
                    log.warning("schema-free tool retry for %s also failed (%s)",
                                role, str(exc2)[:110])
            log.warning("tool loop failed for %s (%s); falling back to a single call "
                        "WITHOUT TOOLS — this agent did not search",
                        role, str(exc)[:140])
            self.used_tools = False
            out = self.ctx.registry.complete(role, system, user, schema=self.schema)
            return self.postprocess(out, **kwargs)

    def _tool_loop_without_schema(self, role: str, system: str, user: str, t0: float) -> Any:
        """Run the tool loop with no bound response format, then parse the answer.

        The tools are the point of this path; the structured wrapper is a
        convenience. When a model can have one but not both, keep the tools.
        """
        from langchain.agents import create_agent

        model = self.ctx.registry.get(role)
        agent = create_agent(model, tools=self.tools, system_prompt=system)
        result = agent.invoke(
            {"messages": [{"role": "user", "content": user}]},
            {"recursion_limit": self.max_tool_iterations * 2},
        )
        msgs = result.get("messages") or []
        text = ""
        for m in reversed(msgs):
            text = getattr(m, "content", "") or ""
            if isinstance(text, str) and text.strip():
                break
        if not text:
            return None
        # Reuse the registry's own repair path, which already handles fenced JSON
        # and the providers that will not emit a native schema.
        parsed = self.ctx.registry.complete(
            role, "Return ONLY the structured object described by the schema.",
            f"Convert this research write-up into the schema verbatim; invent nothing.\n\n{text}",
            schema=self.schema)
        self.used_tools = True
        usage = _tool_loop_usage(result)
        self.ctx.registry.ledger.record(role, **usage)
        self.ctx.registry.record_external_turn(
            role, "deep", system, user, parsed, time.time() - t0)
        return parsed


    def _stream_tool_loop(self, agent: Any, user: str) -> tuple[Any, Exception | None]:
        """Run the loop keeping the last state, so a cutoff is not a total loss."""
        last, err = None, None
        try:
            for chunk in agent.stream(
                {"messages": [{"role": "user", "content": user}]},
                {"recursion_limit": self.max_tool_iterations * 2},
                stream_mode="values",
            ):
                last = chunk
        except Exception as exc:  # noqa: BLE001
            err = exc
        return last, err

    def _salvage_partial(self, role: str, system: str, user: str,
                         state: Any, cutoff: Exception | None) -> Any:
        """Turn whatever the tools found into the schema, or return None.

        Only worth doing when tools ACTUALLY RAN. A cutoff with no tool output is
        just a failure, and pretending otherwise would report an un-researched
        answer as researched.
        """
        msgs = (state or {}).get("messages") or []
        n_tools = sum(1 for m in msgs if getattr(m, "type", "") == "tool")
        if not n_tools:
            return None
        found = "\n\n".join(
            str(getattr(m, "content", ""))[:4000]
            for m in msgs if getattr(m, "type", "") == "tool"
        )
        why = ("hit the tool-call limit of "
               f"{self.max_tool_iterations}" if cutoff is not None else
               "finished without emitting the structured object")
        log.warning("%s %s after %d tool call(s) — keeping what it found rather "
                    "than discarding the research", role, why, n_tools)
        try:
            return self.ctx.registry.complete(
                role,
                "Return ONLY the structured object described by the schema.",
                "Research notes gathered from the tools are below. Convert them into "
                "the schema. Use ONLY what is here; invent nothing, and do not fill "
                f"gaps from memory.\n\n{found[:40000]}",
                schema=self.schema)
        except Exception as exc:  # noqa: BLE001
            log.warning("could not convert %s's partial research (%s)", role, type(exc).__name__)
            return None

    used_tools: bool = False
