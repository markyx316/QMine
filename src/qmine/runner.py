"""Wiring a run: stores, memory, checkpointer, graph, execution.

This is the seam between "a pipeline" and "a pipeline you can operate".  It owns
the things that must be opened and closed in the right order — the SQLite
checkpointer, the SQLite memory store, the artifact generation — and it owns the
recursion limit, the resume path, and the new-generation path taken after a
reviewer's veto.
"""

from __future__ import annotations

import contextlib
import json
import logging
import time
import uuid
from pathlib import Path
from collections.abc import Sequence
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
def _run_log(root: Path) -> Iterator[None]:
    """Tee every emitted line to ``<run>/run.log``, whatever the console is doing.

    The live dashboard owns the screen, so the CLI drops the console logger to
    WARNING while it runs — which used to mean the INFO stream simply ceased to
    exist. Nothing was written to disk anywhere, so watching a run *prettily* and
    being able to debug it afterwards were mutually exclusive, and a two-hour run
    that halted left only whatever six lines the dashboard happened to be showing.

    The level lives on the *handler*, never on the logger. A logger's level gates
    records before any handler sees them, so quieting the console by lowering the
    logger — which is what the CLI used to do — silences the file too. Keeping the
    logger permissive and letting each handler choose is what allows one INFO
    stream to reach disk while the console shows only warnings.
    """
    path = root / "run.log"
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                                           datefmt="%H:%M:%S"))
    previous = log.level
    log.setLevel(logging.INFO)
    log.addHandler(handler)
    try:
        yield
    finally:
        log.setLevel(previous)
        log.removeHandler(handler)
        handler.close()


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
    registry = ModelRegistry(cfg.llm, cache_dir=root / "llm_cache", run_cfg=cfg)
    events: list[str] = []

    def _emit(msg: str) -> None:
        events.append(msg)
        # A follower attached with `qmine watch` reads the log, which carries no
        # spend. Snapshotting usage beside it keeps the two things the dashboard
        # promises to show continuously — gates and money — available to a viewer
        # that is not this process.
        try:
            (root / "usage.json").write_text(json.dumps(registry.usage(), default=str))
        except Exception:  # noqa: BLE001
            pass
        if on_event:
            on_event(msg)

    def _agent(rec: dict[str, Any]) -> None:
        # One line, readable in `--plain` and parseable by the dashboard. The
        # marker is what lets a follower rebuild the agent panel from run.log
        # alone, so it must survive into the file verbatim.
        mark = "ok" if rec["ok"] else "!!"
        line = (f"  ~ {rec['role']} {mark} {rec['latency_s']}s "
                f"out {rec['output_tokens']:,} · {rec['model']} · {rec['returned']}")[:400]
        # `log.info` FIRST. `_emit` only feeds the in-process dashboard; `deps.emit`
        # is what reaches `run.log`, and a follower attached with `qmine watch`
        # reads the file. Emitting only to `_emit` left the agents panel empty for
        # an entire live run while the mechanism underneath it worked fine.
        log.info(line)
        _emit(line)

    registry.on_call = _agent

    if on_event is not None and hasattr(on_event, "__self__"):
        # A LiveDashboard: give it a handle on usage so it can show spend live.
        dash = on_event.__self__
        if hasattr(dash, "usage_fn"):
            dash.usage_fn = registry.usage
        if hasattr(dash, "provider"):
            dash.provider = registry.provider

    with _run_log(root), open_memory(Path(cfg.run_root) / "memory.sqlite", project="qmine", domain=cfg.domain.key) as memory:
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

            summary = write_summary(final, store, registry,
                                    declared_gates=cfg.gates.blocking, run_id=run_id,
                                    generation=generation, elapsed=time.time() - t0)
            return {"state": final, "summary": summary, "events": events, "deps": deps}


def write_summary(
    final: PipelineState, store: ArtifactStore, registry: ModelRegistry, *,
    run_id: str, generation: int, elapsed: float, resumed: bool = False,
    declared_gates: Sequence[str] = (),
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
        # `remediation` is the only field that tells an operator what to DO about a
        # failure; dropping it here left the halt reason visible and the fix
        # invisible in every downstream view.
        "gates": {k: {"status": g.status, "blocking": g.blocking, "message": g.message,
                      "remediation": g.remediation}
                  for k, g in final.get("gates", {}).items()},
        # A blocking gate that never fires is indistinguishable from one that
        # passed. Declaring it and never emitting it is a silent hole in the
        # quality bar, so the run states plainly which declared gates never ran.
        "declared_gates_never_evaluated": sorted(
            set(declared_gates) - set(final.get("gates", {}))
        ),
        "artifacts": sorted(final.get("artifacts", {})),
        "llm_usage": registry.usage(),
        "artifact_root": str(store.gen_dir),
        "n_prescriptions": len(final.get("prescriptions", [])),
        "n_decisions": len(final.get("decisions", [])),
    }
    store.put_json("run_summary", summary, producer="run", summary="end-of-run summary")
    return summary


def _first_missing_phase(state: dict[str, Any]) -> str:
    """The earliest phase node whose phase has not been recorded complete."""
    from .graph.build import PHASE_NODES

    done = set(state.get("completed_phases") or [])
    for node, _ in PHASE_NODES:
        # Node names are "<phase>_<slug>"; the phase id is the leading token.
        phase = node.split("_", 1)[0]
        if phase not in done:
            return node
    return ""


def _node_before(node: str) -> str:
    """The phase node that precedes `node`, or "" if it is first/unknown."""
    from .graph.build import PHASE_NODES

    order = [n for n, _ in PHASE_NODES]
    if node not in order:
        return ""
    i = order.index(node)
    return order[i - 1] if i > 0 else ""


def _halt_kind(state: dict[str, Any]) -> str:
    """Why the run halted: ``crash``, ``gate``, ``review``, or ``""``.

    Reads the recorded field when present. Checkpoints written before that field
    existed are classified from the evidence a crash leaves behind — the failing
    node records ``phase_status[name] = "error"`` — rather than by pattern-matching
    the reason string, which contains a user-supplied exception message.
    """
    kind = str(state.get("halt_kind") or "")
    if kind:
        return kind
    reason = str(state.get("halt_reason") or "")
    errored = [n for n, st in (state.get("phase_status") or {}).items() if st == "error"]
    if any(reason.startswith(f"{n}:") for n in errored):
        return "crash"
    return "gate" if reason else ""


def resume_run(cfg: QMineConfig, run_id: str, *, generation: int = 1, resume_value: Any = None) -> dict[str, Any]:
    """Resume a checkpointed run, optionally answering a pending human review."""
    from langgraph.types import Command

    root = Path(cfg.run_root) / run_id
    store = ArtifactStore(root, generation=generation)
    registry = ModelRegistry(cfg.llm, cache_dir=root / "llm_cache", run_cfg=cfg)
    with _run_log(root), open_memory(Path(cfg.run_root) / "memory.sqlite", project="qmine", domain=cfg.domain.key) as memory:
        deps = Deps(cfg=cfg, store=store, registry=registry, memory=memory,
                    firewall=BlindnessFirewall(), run_id=run_id)
        with _checkpointer(root / "checkpoints.sqlite") as saver:
            graph = build_graph(cfg, deps, checkpointer=saver, human_review=True)
            config = {"configurable": {"thread_id": f"{run_id}-gen{generation}"},
                      "recursion_limit": cfg.llm.recursion_limit}
            # A run that halted on a CRASH is halted because of a defect. Once
            # that defect is fixed, resuming should retry the phase that raised —
            # otherwise the checkpoint replays the halt forever and the only way
            # forward is to re-run phases that already succeeded (37 minutes of
            # web research, in the case that motivated this). A gate halt or a
            # reviewer veto is the opposite: a deliberate judgement that resume
            # must not quietly overturn.
            snap = graph.get_state(config)
            prev = dict(getattr(snap, "values", {}) or {})

            # Where to restart is decided from what actually completed, not from
            # the checkpoint's position pointer. Those two can disagree: a run
            # whose halt flag is cleared without rewinding sits at END while most
            # of the pipeline never ran, and a plain resume then reports success
            # for a pipeline that stopped at phase three.
            missing = _first_missing_phase(prev)
            if missing and not prev.get("halted"):
                back_to = _node_before(missing)
                if back_to:
                    graph.update_state(config, {"halted": False, "halt_kind": "",
                                                "halt_reason": ""}, as_node=back_to)
                    log.info("resume: %s never ran though the run is not halted — "
                             "rewound to %s to continue", missing, back_to)

            if prev.get("halted") and _halt_kind(prev) == "crash":
                # Clearing the flag is not enough: the checkpoint also records
                # *where* the graph is, and after a halt that position is past the
                # failing phase — so a plain resume runs straight to the end and
                # reports success having skipped the phase that crashed. Writing
                # the state back `as_node=<predecessor>` puts the graph on that
                # node's outgoing edge, which leads into the phase to retry.
                crashed = str(prev.get("halt_reason", "")).split(":", 1)[0].strip()
                back_to = _node_before(crashed)
                cleared = {"halted": False, "halt_kind": "", "halt_reason": "",
                           "phase_status": {crashed: "pending"}}
                if back_to:
                    graph.update_state(config, cleared, as_node=back_to)
                    log.info("resuming past a crash in %s — rewound to %s to retry it "
                             "(%s)", crashed, back_to, prev.get("halt_reason", "")[:100])
                else:
                    graph.update_state(config, cleared)
                    log.warning("crash halt in %r, but its position in the phase order is "
                                "unknown — resuming without rewinding, which may skip it",
                                crashed)
            elif prev.get("halted"):
                log.warning(
                    "run halted by %s and stays halted: %s. Resume does not overturn a "
                    "deliberate refusal — fix the underlying issue and start a new "
                    "generation, or answer the pending review.",
                    _halt_kind(prev) or "a blocking gate", prev.get("halt_reason", "")[:160],
                )

            payload = Command(resume=resume_value) if resume_value is not None else None
            t0 = time.time()
            final = graph.invoke(payload, config=config)
            summary = write_summary(final, store, registry, declared_gates=cfg.gates.blocking,
                                    run_id=run_id, generation=generation,
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
