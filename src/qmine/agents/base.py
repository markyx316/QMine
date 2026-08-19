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

        try:
            from langchain.agents import create_agent

            model = self.ctx.registry.get(role)
            agent = create_agent(
                model, tools=self.tools, system_prompt=system,
                response_format=self.schema,
            )
            result = agent.invoke(
                {"messages": [{"role": "user", "content": user}]},
                {"recursion_limit": self.max_tool_iterations * 2},
            )
            parsed = result.get("structured_response")
            if parsed is None:
                raise ValueError("tool agent returned no structured response")
            self.used_tools = True
            self.ctx.registry.ledger.record(role, output_tokens=0)
            return self.postprocess(parsed, **kwargs)
        except Exception as exc:  # noqa: BLE001
            log.warning("tool loop failed for %s (%s); falling back to a single call",
                        role, str(exc)[:140])
            self.used_tools = False
            out = self.ctx.registry.complete(role, system, user, schema=self.schema)
            return self.postprocess(out, **kwargs)

    used_tools: bool = False
