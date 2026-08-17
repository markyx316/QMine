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
        return f"{self.charter}\n\n---\n\n{self.prompt_text}"

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
