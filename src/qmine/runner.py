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

from .artifacts import ArtifactStore, latest_generation
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
    # BUILD THE SAVER INSIDE try/except; YIELD IT OUTSIDE.
    #
    # `@contextmanager` throws an exception raised in the WITH-BODY back in at the
    # yield point, so a single try/except wrapped around the yield caught the
    # entire pipeline's failures — not just this function's setup. live42 ended
    # its 17th phase, LangGraph's loop teardown raised, and this relabelled it
    # "SQLite checkpointer unavailable (Type is not msgpack serializable:
    # DecisionRecord)" — a message about a serializer that encodes
    # `DecisionRecord` perfectly well — then fell through to a SECOND `yield`,
    # which a generator may not do: "generator didn't stop after throw()".
    #
    # The damage was not the noise. The RuntimeError replaced the real exception
    # and killed the run BEFORE `write_summary`, so `run_summary.json` was never
    # written — and five of `verify_run.py`'s six checks read that file, so a
    # completed 17/17 run scored as though observers had never run and no phase
    # had completed. A masked exception cost the run its entire verifiability.
    saver: Any = None
    conn = None
    try:
        import sqlite3

        from langgraph.checkpoint.sqlite import SqliteSaver

        conn = sqlite3.connect(str(path), check_same_thread=False)
        saver = SqliteSaver(conn, serde=_serializer())
        saver.setup()
    except Exception as exc:  # noqa: BLE001 — setup only; the body is not in scope here
        log.warning("SQLite checkpointer unavailable (%s); using in-memory", exc)
        if conn is not None:
            with contextlib.suppress(Exception):
                conn.close()
        conn, saver = None, None

    if saver is None:
        from langgraph.checkpoint.memory import InMemorySaver

        saver = InMemorySaver()

    try:
        # Exactly one yield, and no `except` around it: a pipeline exception now
        # propagates unchanged to the caller instead of being renamed.
        yield saver
    finally:
        if conn is not None:
            with contextlib.suppress(Exception):
                conn.close()


def _wire_events(
    root: Path, registry: ModelRegistry, on_event: Any = None,
    usage_path: Path | None = None,
) -> tuple[list[str], Any]:
    """Wire agent lines and usage snapshots, for either entry point.

    `resume_run` wired NEITHER: no `registry.on_call`, no event sink. So a
    resumed run wrote not one `~ role ok ...` line to `run.log` — the agents
    panel stayed empty for its whole duration and `qmine watch` had nothing to
    replay — while the same mechanism worked perfectly on a fresh run. Sharing
    it is the point: this had already been fixed once, on the fresh path only.
    """
    events: list[str] = []

    def _emit(msg: str) -> None:
        events.append(msg)
        # A follower attached with `qmine watch` reads the log, which carries no
        # spend. Snapshotting usage beside it keeps the two things the dashboard
        # promises to show continuously — gates and money — available to a viewer
        # that is not this process.
        # `usage_path` EXISTS BECAUSE A RE-RENDER IS NOT THE RUN.
        #
        # This was hardcoded to `root/usage.json`, which is run-level. A render
        # calls `_wire_events` too, so re-rendering live42 overwrote that run's
        # own spend record — 702 calls and $29.69 replaced by the render's 11
        # calls and $0.78, unrecoverably, because live42's teardown bug meant no
        # `run_summary.json` held a second copy. A render writes its own spend
        # into its own generation instead.
        try:
            (usage_path or root / "usage.json").write_text(
                json.dumps(registry.usage(), default=str))
        except Exception:  # noqa: BLE001
            pass
        if on_event:
            on_event(msg)

    def _agent(rec: dict[str, Any]) -> None:
        # One line, readable in `--plain` and parseable by the dashboard. The
        # marker is what lets a follower rebuild the agent panel from run.log
        # alone, so it must survive into the file verbatim.
        mark = "ok" if rec["ok"] else "!!"
        # THIS call's output, not the role's running total — see `report_call`.
        # The total read as per-call and sent a live diagnosis down the wrong path.
        out = rec.get("call_output_tokens", rec["output_tokens"])
        line = (f"  ~ {rec['role']} {mark} {rec['latency_s']}s "
                f"out {out:,} · {rec['model']} · {rec['returned']}")[:400]
        # `log.info` FIRST. `_emit` only feeds the in-process dashboard; the log
        # file is what a follower reads. Emitting only to `_emit` left the agents
        # panel empty for an entire live run.
        log.info(line)
        _emit(line)

    registry.on_call = _agent
    return events, _emit


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
    events, _emit = _wire_events(root, registry, on_event)

    if on_event is not None and hasattr(on_event, "__self__"):
        # A LiveDashboard: give it a handle on usage so it can show spend live.
        dash = on_event.__self__
        if hasattr(dash, "usage_fn"):
            dash.usage_fn = registry.usage
        if hasattr(dash, "provider"):
            dash.provider = registry.provider
        # A BROWSABLE VIEW BESIDE THE TERMINAL ONE.
        #
        # The panel keeps the last 8 agent calls because that is what fits;
        # live40 made 696. `registry.raw_log` holds every full return, and
        # `index.jsonl` lists artifacts as they are written, so the HTML view can
        # answer "what did that researcher actually say" and "what exists so far"
        # while the run is still going. Same model, so the two cannot disagree.
        if hasattr(dash, "transcript_fn"):
            dash.transcript_fn = lambda: registry.raw_log

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
            # WRITE THE SUMMARY EVEN IF THE GRAPH RAISES.
            #
            # It used to sit after the stream, so any exception — including one
            # thrown during LangGraph's own loop teardown, after every phase had
            # finished — skipped it. live42 completed 17/17 phases and shipped
            # every deliverable with no `run_summary.json`, and FIVE of
            # `verify_run.py`'s six checks read that file: the run scored as
            # though no phase had completed and no observer had run.
            #
            # A summary of the last state reached is worth strictly more than
            # none, and on a crash it is the only record of how far the run got.
            # The exception still propagates; it is no longer paid for twice.
            try:
                if stream:
                    for chunk in graph.stream(init, config=config, stream_mode="values"):
                        final = chunk
                else:
                    final = graph.invoke(init, config=config)
            finally:
                try:
                    summary = write_summary(
                        final, store, registry, declared_gates=cfg.gates.blocking,
                        run_id=run_id, generation=generation, elapsed=time.time() - t0)
                except Exception as exc:  # noqa: BLE001
                    log.warning("run_summary could not be written (%s)", exc)
                    summary = {}
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


def resume_run(cfg: QMineConfig, run_id: str, *, generation: int = 1,
               resume_value: Any = None, on_event: Any = None) -> dict[str, Any]:
    """Resume a checkpointed run, optionally answering a pending human review."""
    from langgraph.types import Command

    root = Path(cfg.run_root) / run_id
    store = ArtifactStore(root, generation=generation)
    registry = ModelRegistry(cfg.llm, cache_dir=root / "llm_cache", run_cfg=cfg)
    events, _emit = _wire_events(root, registry, on_event)
    with _run_log(root), open_memory(Path(cfg.run_root) / "memory.sqlite", project="qmine", domain=cfg.domain.key) as memory:
        deps = Deps(cfg=cfg, store=store, registry=registry, memory=memory,
                    firewall=BlindnessFirewall(), run_id=run_id, on_event=_emit)
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

            if resume_value is not None:
                payload: Any = Command(resume=resume_value)
            elif not prev:
                # A NEW GENERATION IS A NEW THREAD. `new-generation` is the move
                # the gate-halt message tells you to make, and the thread id is
                # `{run_id}-gen{generation}` — so gen02 has no checkpoint, and
                # invoking with None (meaning "continue from the checkpoint")
                # raised "Received no input for __start__", which `open_memory`
                # then masked as "generator didn't stop after throw()". Seed it
                # exactly as a fresh run does; the artifacts and the llm_cache
                # carry over, so the replay is cheap.
                payload = new_state(run_id, cfg.config_hash, cfg.domain.key, generation)
                log.info("resume: generation %d has no checkpoint of its own — running "
                         "its graph from the start; artifacts and llm_cache carry over",
                         generation)
            else:
                payload = None
            t0 = time.time()
            final = graph.invoke(payload, config=config)
            summary = write_summary(final, store, registry, declared_gates=cfg.gates.blocking,
                                    run_id=run_id, generation=generation,
                                    elapsed=time.time() - t0, resumed=True)
            return {"state": final, "summary": summary, "events": events,
                    "state_summary": state_summary(final)}


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


#: State channels a report needs and cannot rebuild from artifacts alone.
#: `run_summary.json` records `gates` and `completed_phases` but only the COUNT
#: of decisions and prescriptions, and it records observations not at all.
_STATE_FROM_CHECKPOINT = ("decisions", "observations", "prescriptions",
                          "gates", "completed_phases", "metrics")


def recover_state(root: Path, generation: int, cfg: QMineConfig,
                  deps: Any) -> tuple[dict[str, Any], list[str]]:
    """Rebuild a finished run's pipeline state, and say what could not be found.

    Returns ``(state, missing)``. The second half is the point: a narrative
    written against a starved state does not come out cautious, it comes out
    invented — the same failure the fact-sheet work established — so a caller
    that is about to spend money on agents has to be able to see the hole first.

    Three sources, most complete first:

    1. **The LangGraph checkpoint.** The whole state, including the channels
       nothing else preserves. `resume_run` reads it the same way.
    2. **`run_summary.json`** — carries `gates` (24 of them on live40) and
       `completed_phases`, which covers the gate ledger every report prints.
    3. **`findings.json`** — run-level, beside `llm_cache`, and therefore intact
       even when a generation is set aside.
    """
    state: dict[str, Any] = {}
    missing: list[str] = []

    try:
        from .graph.build import build_graph

        with _checkpointer(root / "checkpoints.sqlite") as saver:
            graph = build_graph(cfg, deps, checkpointer=saver, human_review=False)
            snap = graph.get_state(
                {"configurable": {"thread_id": f"{deps.run_id}-gen{generation}"}})
            state = dict(getattr(snap, "values", {}) or {})
    except Exception as exc:  # noqa: BLE001 — a missing checkpoint is recoverable
        log.warning("checkpoint unreadable (%s); rebuilding state from artifacts", exc)

    summary = root / f"gen{generation:02d}" / "run_summary.json"
    if summary.exists():
        try:
            s = json.loads(summary.read_text(encoding="utf-8"))
            from .records import GateResult

            if not state.get("gates") and s.get("gates"):
                # `name` and `phase` are required on GateResult and are NOT in
                # the serialised value — the name is the dict key, and the phase
                # is dropped. Validating the value alone raises, which would
                # lose the whole gate ledger to a `except` two frames up.
                state["gates"] = {
                    k: GateResult.model_validate(
                        {"name": k, "phase": str(v.get("phase") or ""), **v})
                    for k, v in s["gates"].items() if isinstance(v, dict)}
            state.setdefault("completed_phases", s.get("completed_phases") or [])
            state.setdefault("declared_gates_never_evaluated",
                             s.get("declared_gates_never_evaluated") or [])
        except Exception as exc:  # noqa: BLE001
            log.warning("run_summary.json unreadable (%s)", exc)

    state.setdefault("run_id", deps.run_id)
    state.setdefault("events", [])
    for ch in _STATE_FROM_CHECKPOINT:
        if not state.get(ch):
            missing.append(ch)
    return state, missing


def render_run(cfg: QMineConfig, run_id: str, *, generation: int | None = None,
               agents: bool = False, on_event: Any = None) -> dict[str, Any]:
    """Rebuild a finished run's deliverables into a NEW generation.

    Every deliverable is a projection of artifacts that are already on disk, so
    re-deriving them costs nothing but CPU — and until now there was no way to do
    it. Every fix to a report generator was therefore unverifiable against a real
    run without paying for the whole 4-hour pipeline again, which is why a
    section that had NEVER rendered (`§2.1 L2 子意图`) survived to live42, and why
    the reference shelf existed in code for a day without a single run producing
    it.

    Writes into a new generation, never over the source. A delivered document is
    evidence of what a run said; overwriting it in place would destroy the only
    record of the defect a re-render is meant to fix.

    With ``agents=False`` this makes NO model calls: the agent-authored steps are
    switched off at their config gates rather than pointed at a stand-in, so the
    output contains no invented prose. With ``agents=True`` the narrative writer,
    the interpreter and the pre-delivery auditor run again — replaying from
    `llm_cache/` wherever the prompt is byte-identical, which it will be for any
    section whose evidence did not change.
    """
    from .graph.nodes.delivery import p11_report

    root = Path(cfg.run_root) / run_id
    if not root.exists():
        raise FileNotFoundError(f"no run at {root}")
    src_gen = generation or latest_generation(root)

    registry = ModelRegistry(cfg.llm, cache_dir=root / "llm_cache", run_cfg=cfg)
    # Into the generation this render is about to write, never over the run's own
    # record of what it spent. See `_wire_events`.
    # A NEW GENERATION IS THE ONE AFTER THE LATEST, NOT AFTER THE SOURCE.
    #
    # `ArtifactStore.new_generation` increments from ITSELF, so a store opened at
    # the source generation returns `src + 1` — and rendering gen01 twice wrote
    # both renders into gen02, interleaving a no-agents pass with an agent pass
    # in one directory. The store resolves artifacts across generations
    # (`ref.generation <= self.generation`), so reading from gen01 while writing
    # to the newest is exactly what is wanted.
    target_gen = latest_generation(root) + 1
    events, _emit = _wire_events(
        root, registry, on_event,
        usage_path=root / f"gen{target_gen:02d}" / "usage.json")

    if not agents:
        # Off at the gate, not redirected to the offline stand-in: a stand-in
        # produces complete-looking prose that no model wrote, which is worse in
        # a deliverable than an honest absence.
        cfg.final_report = False
        cfg.interpret_results = False
        cfg.observe_phases = False
        cfg.delivery_audit = False

    with _run_log(root), open_memory(Path(cfg.run_root) / "memory.sqlite",
                                    project="qmine", domain=cfg.domain.key) as memory:
        src_store = ArtifactStore(root, generation=src_gen)
        deps_src = Deps(cfg=cfg, store=src_store, registry=registry, memory=memory,
                        firewall=BlindnessFirewall(), run_id=run_id, on_event=_emit)
        state, missing = recover_state(root, src_gen, cfg, deps_src)
        if missing:
            _emit(f"  ⚠ 这些状态通道没能恢复, 相关章节会相应变短: {', '.join(missing)}")

        store = ArtifactStore(root, generation=target_gen - 1).new_generation(
            note=f"re-rendered deliverables from gen{src_gen:02d}"
                 f" ({'with' if agents else 'without'} agents)")
        deps = Deps(cfg=cfg, store=store, registry=registry, memory=memory,
                    firewall=BlindnessFirewall(), run_id=run_id, on_event=_emit)
        _emit(f"重新生成交付物: {run_id} gen{src_gen:02d} → gen{store.generation:02d}"
              f" ({'含 agent 撰写' if agents else '纯脚本, 不调用模型'})")
        out = p11_report(state, deps)

        # A RENDERED GENERATION MUST BE AS RE-RENDERABLE AS THE ONE IT CAME FROM.
        #
        # `write_summary` runs at the end of a RUN, so a generation produced by
        # rendering had no `run_summary.json` — and `recover_state` falls back to
        # that file for the gate ledger. Rendering gen02 from gen01 then
        # rendering again from gen02 lost `gates` and `completed_phases`
        # entirely, and the second render's reports quietly shrank. Write the
        # summary here too, from the state actually used.
        try:
            merged = {**state, **{k: v for k, v in (out or {}).items()
                                  if k in ("gates", "completed_phases")}}
            write_summary(merged, store, registry, run_id=run_id,
                          generation=store.generation, elapsed=0.0, resumed=True)
        except Exception as exc:  # noqa: BLE001 — the documents are the deliverable
            _emit(f"  run_summary not written for the rendered generation ({exc})")

    written = sorted(p.name for p in Path(store.gen_dir).glob("*.md"))
    return {
        "run_id": run_id, "source_generation": src_gen,
        "generation": store.generation, "agents": agents,
        "state_channels_missing": missing,
        "documents": written,
        "llm": registry.usage() if agents else {"provider": "not used"},
        "events": len(out.get("events") or []),
    }
