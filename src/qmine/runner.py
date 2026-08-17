"""Wiring a run: stores, memory, checkpointer, graph, execution.

This is the seam between "a pipeline" and "a pipeline you can operate".  It owns
the things that must be opened and closed in the right order — the SQLite
checkpointer, the SQLite memory store, the artifact generation — and it owns the
recursion limit, the resume path, and the new-generation path taken after a
reviewer's veto.
"""

from __future__ import annotations

import contextlib
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Iterator

from .artifacts import ArtifactStore
from .config import QMineConfig
from .graph.build import build_graph
from .graph.deps import Deps
from .llm.registry import ModelRegistry
from .memory.context import BlindnessFirewall
from .memory.store import open_memory
from .state import PipelineState, new_state, state_summary

log = logging.getLogger("qmine")


def make_run_id(prefix: str = "run") -> str:
    return f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"


@contextlib.contextmanager
def open_run(
    cfg: QMineConfig,
    *,
    run_id: str | None = None,
    generation: int = 1,
    resume: bool = False,
    on_event: Any = None,
) -> Iterator[tuple[Deps, Any, dict[str, Any]]]:
    """Open every resource a run needs and yield ``(deps, graph, config_dict)``."""
    run_id = run_id or make_run_id(cfg.domain.key)
    root = Path(cfg.run_root) / run_id
    root.mkdir(parents=True, exist_ok=True)

    store = ArtifactStore(root, generation=generation)
    registry = ModelRegistry(cfg.llm, cache_dir=root / "llm_cache")

    with open_memory(root.parent / "memory.sqlite", project="qmine", domain=cfg.domain.key) as memory:
        deps = Deps(
            cfg=cfg, store=store, registry=registry, memory=memory,
            firewall=BlindnessFirewall(), run_id=run_id, on_event=on_event,
        )
        with _checkpointer(root / "checkpoints.sqlite") as saver:
            graph = build_graph(cfg, deps, checkpointer=saver, human_review=cfg.gates.human_review_points and False)
            yield deps, graph, {
                "run_id": run_id,
                "root": str(root),
                "thread": {"configurable": {"thread_id": f"{run_id}-gen{generation}"},
                           "recursion_limit": cfg.llm.recursion_limit},
            }


#: Our own record types, declared to the checkpoint serialiser.
#:
#: State carries Pydantic models — ArtifactRef, DecisionRecord, GateResult,
#: Prescription, MetricSet, LeafNaming. LangGraph currently deserialises unknown
#: types with a warning and has announced it will refuse them. Declaring the
#: allowlist now silences the warning and, more importantly, means a future
#: strict default does not turn every existing checkpoint into an unresumable
#: file.
_ALLOWED_MSGPACK_MODULES = (
    ("qmine.artifacts", "ArtifactRef"),
    ("qmine.records", "DecisionRecord"),
    ("qmine.records", "GateResult"),
    ("qmine.records", "Prescription"),
    ("qmine.records", "MetricRecord"),
    ("qmine.records", "MetricSet"),
    ("qmine.records", "LeafNaming"),
    ("qmine.records", "FamilyNaming"),
    ("qmine.records", "TreeAudit"),
    ("qmine.records", "TemplateGroup"),
    ("qmine.records", "Taxonomy"),
    ("qmine.records", "TaxonomyNode"),
    ("qmine.records", "AdjudicationRule"),
    ("qmine.records", "GoldRow"),
    ("qmine.records", "NamingCard"),
    ("qmine.records", "LessonRecord"),
)


def _serializer() -> Any:
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    try:
        return JsonPlusSerializer(allowed_msgpack_modules=_ALLOWED_MSGPACK_MODULES)
    except TypeError:  # older langgraph without the parameter
        return JsonPlusSerializer()


@contextlib.contextmanager
def _checkpointer(path: Path) -> Iterator[Any]:
    """SQLite checkpointing, degrading to in-memory if unavailable.

    Durability is what makes a multi-hour pipeline operable: without it, an
    exception in Phase 9 costs the encoding, the sweep, and the tree. Verified
    in practice on this project — a run killed during Phase 4 resumed at Phase 4
    rather than re-encoding 50,000 rows.
    """
    try:
        import sqlite3

        from langgraph.checkpoint.sqlite import SqliteSaver

        conn = sqlite3.connect(str(path), check_same_thread=False)
        try:
            saver = SqliteSaver(conn, serde=_serializer())
            saver.setup()
            yield saver
            return
        finally:
            conn.close()
    except Exception as exc:  # noqa: BLE001
        log.warning("SQLite checkpointer unavailable (%s); using in-memory", exc)
    from langgraph.checkpoint.memory import InMemorySaver

    yield InMemorySaver()


def run_pipeline(
    cfg: QMineConfig,
    *,
    run_id: str | None = None,
    generation: int = 1,
    human_review: bool = False,
    on_event: Any = None,
    stream: bool = True,
) -> dict[str, Any]:
    """Execute the whole pipeline and return the final state plus a summary."""
    run_id = run_id or make_run_id(cfg.domain.key)
    root = Path(cfg.run_root) / run_id
    root.mkdir(parents=True, exist_ok=True)

    store = ArtifactStore(root, generation=generation)
    registry = ModelRegistry(cfg.llm, cache_dir=root / "llm_cache")
    events: list[str] = []

    def _emit(msg: str) -> None:
        events.append(msg)
        if on_event:
            on_event(msg)

    with open_memory(Path(cfg.run_root) / "memory.sqlite", project="qmine", domain=cfg.domain.key) as memory:
        deps = Deps(cfg=cfg, store=store, registry=registry, memory=memory,
                    firewall=BlindnessFirewall(), run_id=run_id, on_event=_emit)
        with _checkpointer(root / "checkpoints.sqlite") as saver:
            graph = build_graph(cfg, deps, checkpointer=saver, human_review=human_review)
            init = new_state(run_id, cfg.config_hash, cfg.domain.key, generation)
            config = {
                "configurable": {"thread_id": f"{run_id}-gen{generation}"},
                "recursion_limit": cfg.llm.recursion_limit,
            }
            t0 = time.time()
            final: PipelineState = init
            if stream:
                for chunk in graph.stream(init, config=config, stream_mode="values"):
                    final = chunk
            else:
                final = graph.invoke(init, config=config)

            summary = write_summary(final, store, registry, run_id=run_id,
                                    generation=generation, elapsed=time.time() - t0)
            return {"state": final, "summary": summary, "events": events, "deps": deps}


def write_summary(
    final: PipelineState, store: ArtifactStore, registry: ModelRegistry, *,
    run_id: str, generation: int, elapsed: float, resumed: bool = False,
) -> dict[str, Any]:
    """Write ``run_summary.json``.

    Shared by the fresh-run and resume paths on purpose: a resumed run used to
    finish with every artifact on disk and no summary, so `qmine inspect` — which
    reads exactly this file — reported nothing about the run that actually
    produced the deliverables.
    """
    if registry.raw_log:
        store.put_json("agent_transcript", registry.raw_log, producer="run",
                       summary=f"{len(registry.raw_log)} agent calls")
    summary = {
        "run_id": run_id,
        "generation": generation,
        "resumed": resumed,
        "elapsed_s": round(elapsed, 1),
        "state_summary": state_summary(final),
        "completed_phases": final.get("completed_phases", []),
        "halted": final.get("halted", False),
        "halt_reason": final.get("halt_reason", ""),
        "gates": {k: {"status": g.status, "blocking": g.blocking, "message": g.message}
                  for k, g in final.get("gates", {}).items()},
        "artifacts": sorted(final.get("artifacts", {})),
        "llm_usage": registry.usage(),
        "artifact_root": str(store.gen_dir),
        "n_prescriptions": len(final.get("prescriptions", [])),
        "n_decisions": len(final.get("decisions", [])),
    }
    store.put_json("run_summary", summary, producer="run", summary="end-of-run summary")
    return summary


def resume_run(cfg: QMineConfig, run_id: str, *, generation: int = 1, resume_value: Any = None) -> dict[str, Any]:
    """Resume a checkpointed run, optionally answering a pending human review."""
    from langgraph.types import Command

    root = Path(cfg.run_root) / run_id
    store = ArtifactStore(root, generation=generation)
    registry = ModelRegistry(cfg.llm, cache_dir=root / "llm_cache")
    with open_memory(Path(cfg.run_root) / "memory.sqlite", project="qmine", domain=cfg.domain.key) as memory:
        deps = Deps(cfg=cfg, store=store, registry=registry, memory=memory,
                    firewall=BlindnessFirewall(), run_id=run_id)
        with _checkpointer(root / "checkpoints.sqlite") as saver:
            graph = build_graph(cfg, deps, checkpointer=saver, human_review=True)
            config = {"configurable": {"thread_id": f"{run_id}-gen{generation}"},
                      "recursion_limit": cfg.llm.recursion_limit}
            payload = Command(resume=resume_value) if resume_value is not None else None
            t0 = time.time()
            final = graph.invoke(payload, config=config)
            summary = write_summary(final, store, registry, run_id=run_id, generation=generation,
                                    elapsed=time.time() - t0, resumed=True)
            return {"state": final, "summary": summary, "state_summary": state_summary(final)}


def new_generation(cfg: QMineConfig, run_id: str, *, from_generation: int, reason: str) -> int:
    """Open the next generation after a veto.  The old one is never touched.

    This is Principle 8 in operation: a rejected tree is not deleted, because a
    rejected artifact is still evidence — the source project's discarded 107-leaf
    tree later became its phrasing-pattern library.
    """
    root = Path(cfg.run_root) / run_id
    store = ArtifactStore(root, generation=from_generation)
    nxt = store.new_generation(reason)
    return nxt.generation
