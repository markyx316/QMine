"""Long-term memory: what the team knows across runs, not just within one.

Three tiers, each with a different lifetime and a different failure mode if you
skip it:

============  ==================================  ==========================
tier          mechanism                           what breaks without it
============  ==================================  ==========================
working       LangGraph checkpointer (per thread) a crash costs the whole run
artifact      the filesystem + ArtifactRef        state balloons; context dies
long-term     this module (BaseStore, SQLite)     the team re-learns lessons
============  ==================================  ==========================

The long-term tier is namespaced by *kind of knowledge*, following the
semantic / episodic / procedural split:

* ``decisions`` — semantic.  What we chose, why, and what we rejected.  Read by
  the report writer for its mandatory failure-history section, and by later
  agents so a settled question is not silently reopened.
* ``lessons`` — episodic.  Situation → action → outcome → lesson, written after
  a gate fails or a reviewer vetoes, retrieved by similarity into later prompts.
  This is the reflection loop TradingAgents uses on losing trades, pointed at
  methodology instead of P&L.
* ``rules`` — procedural.  Adjudication rules that *grow* when a referee finds a
  case the guide does not cover.  The taxonomy's behaviour changes as this
  namespace fills, which is what makes it procedural rather than merely stored.
* ``glossary`` — the current taxonomy and leaf definitions, for annotators.
* ``domain_priors`` — what we learned about a vertical, reusable next quarter.

Semantic search over these namespaces is optional: with an embedding function
configured, ``search(query=...)`` ranks by similarity; without one it degrades
to filtered listing, which keeps the whole system runnable offline.
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Any, Iterator, Sequence

from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

log = logging.getLogger("qmine.memory")

NS_DECISIONS = "decisions"
NS_LESSONS = "lessons"
NS_RULES = "rules"
NS_GLOSSARY = "glossary"
NS_DOMAIN = "domain_priors"
NS_REJECTIONS = "rejections"

ALL_NAMESPACES = (NS_DECISIONS, NS_LESSONS, NS_RULES, NS_GLOSSARY, NS_DOMAIN, NS_REJECTIONS)


class QMineMemory:
    """A thin, typed facade over a LangGraph ``BaseStore``.

    The facade exists so that call sites read as memory operations
    (``remember_decision``, ``recall_lessons``) rather than as store plumbing,
    and so that swapping SQLite for Postgres is a one-line change here rather
    than a search-and-replace across twelve phase modules.
    """

    def __init__(self, store: BaseStore, *, project: str = "qmine", domain: str = "generic") -> None:
        self.store = store
        self.project = project
        self.domain = domain

    def ns(self, kind: str, *extra: str) -> tuple[str, ...]:
        return (self.project, self.domain, kind, *extra)

    # -- write --------------------------------------------------------------
    def put(self, kind: str, key: str, value: dict[str, Any], *, index: list[str] | None = None) -> None:
        try:
            self.store.put(self.ns(kind), key, value, index=index)
        except TypeError:
            # stores without index support
            self.store.put(self.ns(kind), key, value)

    def remember_decision(self, record: Any) -> None:
        d = record.model_dump() if hasattr(record, "model_dump") else dict(record)
        self.put(NS_DECISIONS, d["id"], d, index=["question", "choice", "rationale"])

    def remember_lesson(self, record: Any) -> None:
        d = record.model_dump() if hasattr(record, "model_dump") else dict(record)
        self.put(NS_LESSONS, d["id"], d, index=["situation", "lesson", "outcome"])

    def remember_rule(self, rule: Any) -> None:
        d = rule.model_dump() if hasattr(rule, "model_dump") else dict(rule)
        self.put(NS_RULES, d["id"], d, index=["when", "then"])

    def remember_rejection(self, key: str, payload: dict[str, Any]) -> None:
        """A human veto.  Kept forever: it is the strongest signal in the system."""
        self.put(NS_REJECTIONS, key, payload, index=["reason"])

    def put_glossary(self, key: str, value: dict[str, Any]) -> None:
        self.put(NS_GLOSSARY, key, value, index=["name", "definition", "user_need"])

    # -- read ---------------------------------------------------------------
    def search(self, kind: str, *, query: str | None = None, limit: int = 5, filter: dict | None = None) -> list[dict[str, Any]]:
        try:
            items = self.store.search(self.ns(kind), query=query, limit=limit, filter=filter)
        except Exception as exc:  # noqa: BLE001 — degrade rather than fail a run
            log.debug("store.search(%s) fell back to listing: %s", kind, exc)
            try:
                items = self.store.search(self.ns(kind), limit=limit)
            except Exception:
                return []
        return [dict(getattr(i, "value", i)) for i in items]

    def recall_lessons(self, situation: str, *, limit: int = 5) -> list[dict[str, Any]]:
        """The episodic-memory read.  Feeds "what went wrong last time" into a prompt."""
        return self.search(NS_LESSONS, query=situation, limit=limit)

    def recall_decisions(self, question: str, *, limit: int = 5) -> list[dict[str, Any]]:
        return self.search(NS_DECISIONS, query=question, limit=limit)

    def all_rules(self, limit: int = 200) -> list[dict[str, Any]]:
        return self.search(NS_RULES, limit=limit)

    def all_rejections(self, limit: int = 50) -> list[dict[str, Any]]:
        return self.search(NS_REJECTIONS, limit=limit)

    # -- prompt assembly -----------------------------------------------------
    def context_block(self, situation: str, *, max_lessons: int = 4, max_decisions: int = 3) -> str:
        """Render remembered context as a prompt section.

        Deliberately short.  Memory that floods a prompt is not memory, it is
        distraction — the agent stops attending to the task and starts
        summarising its own history.
        """
        lessons = self.recall_lessons(situation, limit=max_lessons)
        decisions = self.recall_decisions(situation, limit=max_decisions)
        rejections = self.all_rejections(limit=3)
        if not (lessons or decisions or rejections):
            return ""
        parts = ["## What this team already learned (do not repeat these mistakes)"]
        for l in lessons:
            parts.append(f"- LESSON [{l.get('severity', 'info')}] {l.get('lesson', '')}")
        for d in decisions:
            parts.append(f"- SETTLED: {d.get('question', '')} → {d.get('choice', '')} ({d.get('rationale', '')[:120]})")
        for r in rejections:
            parts.append(f"- REVIEWER REJECTED: {r.get('what', '')} — {r.get('reason', '')[:150]}")
        return "\n".join(parts)


@contextlib.contextmanager
def open_memory(
    path: str | Path | None = None,
    *,
    project: str = "qmine",
    domain: str = "generic",
    embed: Any = None,
    dims: int = 256,
) -> Iterator[QMineMemory]:
    """Open persistent memory at ``path``, or in-process memory if ``path`` is None.

    SQLite is the default because it gives durable, queryable, single-file
    memory with no server — which is what a data-science team on a laptop
    actually has.
    """
    index = None
    if embed is not None:
        index = {"dims": dims, "embed": embed, "fields": ["$"]}

    if path is None:
        store = InMemoryStore(index=index) if index else InMemoryStore()
        yield QMineMemory(store, project=project, domain=domain)
        return

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    try:
        from langgraph.store.sqlite import SqliteStore

        with SqliteStore.from_conn_string(str(path)) as store:
            with contextlib.suppress(Exception):
                store.setup()
            yield QMineMemory(store, project=project, domain=domain)
    except Exception as exc:  # noqa: BLE001
        log.warning("SQLite store unavailable (%s); using in-process memory", exc)
        store = InMemoryStore(index=index) if index else InMemoryStore()
        yield QMineMemory(store, project=project, domain=domain)


def make_embedding_fn(encoder: Any) -> Any:
    """Adapt a sentence encoder into the callable ``BaseStore`` expects."""

    def embed(texts: Sequence[str]) -> list[list[float]]:
        import numpy as np

        from ..ops.represent import encode_corpus

        return np.asarray(encode_corpus(encoder, list(texts))).tolist()

    return embed
