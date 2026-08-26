"""The graph state — deliberately small, because checkpoints are cheap only if it is.

A 50k x 768 embedding matrix is 150 MB.  Put it in state and every checkpoint
write serialises 150 MB; put an :class:`~qmine.artifacts.ArtifactRef` in state
and every checkpoint write serialises about 400 bytes.  That single decision is
what lets this pipeline checkpoint after *every* node, resume from any failure,
and time-travel to a prior decision without a storage problem.

The reducers below are all commutative-ish merges, which matters because Phase 7
fans five naming agents out in parallel and they all write to the same channels —
and, since the graph forks, because the top-down and bottom-up branches run
concurrently and write to the same channels for the whole of p2a/p2b.

**Every field a forked branch writes needs a reducer.** langgraph rejects a plain
field that receives two values in one superstep, and it does so at RUNTIME, on
the first step where both branches happen to write it. `phase` was the one that
had none.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from .artifacts import ArtifactRef, merge_artifacts
from .records import (
    DecisionRecord,
    GateResult,
    LeafNaming,
    MetricSet,
    Prescription,
)


# --------------------------------------------------------------------------
# Reducers
# --------------------------------------------------------------------------

#: Canonical phase order, used only to decide which of two concurrent `phase`
#: values is the furthest along. Kept here rather than imported from
#: `graph.build` because state must not depend on the graph.
_PHASE_ORDER = (
    "p0", "p1", "p2a", "p2b", "p3", "p4", "p5", "p6",
    "p2c", "p2d", "p2e", "p7", "p8", "p9", "p10", "p11", "p12",
)


def furthest_phase(left: str | None, right: str | None) -> str:
    """Which of two concurrently-written phases is further along.

    `phase` is DISPLAY ONLY — the graph's edges decide routing, and the only
    reader is the dashboard's progress line. It still needs a reducer, because
    the top-down and bottom-up branches both write it in the same superstep and
    langgraph refuses a plain field that receives two values in one step
    (`INVALID_CONCURRENT_GRAPH_UPDATE`).

    "Furthest along" rather than last-writer-wins so the answer does not depend
    on which branch happened to finish first — a progress line that jumps
    backwards between refreshes reads as a stall, which is the one thing the
    dashboard exists to rule out. Unknown values lose to known ones, and two
    unknowns keep the right-hand side, so the reducer stays total.
    """
    if not left:
        return right or ""
    if not right:
        return left
    try:
        li = _PHASE_ORDER.index(left)
    except ValueError:
        return right
    try:
        ri = _PHASE_ORDER.index(right)
    except ValueError:
        return left
    return left if li >= ri else right


def merge_dict(left: dict | None, right: dict | None) -> dict:
    """Shallow last-writer-wins merge.  Safe under parallel fan-in on disjoint keys."""
    out = dict(left or {})
    out.update(right or {})
    return out


def merge_metrics(
    left: dict[str, MetricSet] | None, right: dict[str, MetricSet] | None
) -> dict[str, MetricSet]:
    """Merge metric sets subject-by-subject rather than clobbering whole subjects.

    Two nodes may legitimately measure the same candidate along different axes
    (a stability node and a fragmentation node), and neither should erase the
    other's numbers.
    """
    out: dict[str, MetricSet] = dict(left or {})
    for subject, ms in (right or {}).items():
        if subject in out:
            merged = out[subject].model_copy(deep=True)
            merged.metrics.update(ms.metrics)
            merged.panel_id = ms.panel_id or merged.panel_id
            out[subject] = merged
        else:
            out[subject] = ms
    return out


def merge_prescriptions(
    left: list[Prescription] | None, right: list[Prescription] | None
) -> list[Prescription]:
    """Append, but let a later status update supersede an earlier one by id.

    This is what makes the Phase 8 ledger honest under parallelism: an
    ``executed`` record overwrites its own ``proposed`` predecessor instead of
    sitting next to it and double-counting.
    """
    by_id: dict[str, Prescription] = {}
    for p in list(left or []) + list(right or []):
        by_id[p.id] = p
    return sorted(by_id.values(), key=lambda p: p.id)


def merge_namings(
    left: list[LeafNaming] | None, right: list[LeafNaming] | None
) -> list[LeafNaming]:
    """Collect blind namings from parallel shards, one verdict per leaf."""
    by_leaf: dict[int, LeafNaming] = {}
    for n in list(left or []) + list(right or []):
        by_leaf[n.leaf_id] = n
    return [by_leaf[k] for k in sorted(by_leaf)]


def merge_usage(left: dict | None, right: dict | None) -> dict:
    """Sum token/call counters across every agent in the run."""
    out = dict(left or {})
    for k, v in (right or {}).items():
        if isinstance(v, (int, float)) and isinstance(out.get(k), (int, float)):
            out[k] = out[k] + v
        else:
            out[k] = v
    return out


def last_wins(left: Any, right: Any) -> Any:
    return right if right is not None else left


# --------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------

class PipelineState(TypedDict, total=False):
    """State for the twelve-phase pipeline graph.

    Everything here is either a scalar, a small record, or a *pointer* to
    something large.  If you find yourself wanting to add an ``np.ndarray``
    field, add an artifact instead.
    """

    # --- identity -----------------------------------------------------
    run_id: str
    config_hash: str
    generation: int
    domain: str
    started_at: float

    # --- what phase we are in ----------------------------------------
    phase: Annotated[str, furthest_phase]
    phase_status: Annotated[dict[str, str], merge_dict]
    completed_phases: Annotated[list[str], operator.add]

    # --- the data plane (pointers only) -------------------------------
    artifacts: Annotated[dict[str, ArtifactRef], merge_artifacts]

    # --- the evidence plane -------------------------------------------
    metrics: Annotated[dict[str, MetricSet], merge_metrics]
    decisions: Annotated[list[DecisionRecord], operator.add]
    gates: Annotated[dict[str, GateResult], merge_dict]
    prescriptions: Annotated[list[Prescription], merge_prescriptions]

    # --- phase-specific carriers --------------------------------------
    #: Chosen by Phase 3c's alpha sweep; consumed by Phase 4 onward.
    chosen_alpha: float
    chosen_encoder: str
    chosen_algorithm: str
    family_k: int
    leaf_count: int
    namings: Annotated[list[LeafNaming], merge_namings]

    # --- bookkeeping ---------------------------------------------------
    events: Annotated[list[str], operator.add]
    warnings: Annotated[list[str], operator.add]
    errors: Annotated[list[str], operator.add]
    llm_usage: Annotated[dict[str, Any], merge_usage]

    #: Set when a blocking gate fails or a human vetoes.  The router reads it.
    halted: bool
    halt_reason: str
    #: ``"crash"`` (a node raised — a defect, retryable once fixed), ``"gate"``
    #: (a blocking gate refused — a deliberate judgement), or ``"review"`` (a
    #: human vetoed). Resume treats these differently, and parsing the reason
    #: string to tell them apart is exactly the kind of guesswork that breaks
    #: the first time an exception message contains a colon.
    halt_kind: str
    #: Set by a human veto: re-run these phases in a fresh generation.
    replan: dict[str, Any]


def new_state(run_id: str, config_hash: str, domain: str, generation: int = 1) -> PipelineState:
    import time

    return PipelineState(
        run_id=run_id,
        config_hash=config_hash,
        generation=generation,
        domain=domain,
        started_at=time.time(),
        phase="p0",
        phase_status={},
        completed_phases=[],
        artifacts={},
        metrics={},
        decisions=[],
        gates={},
        prescriptions=[],
        namings=[],
        events=[],
        warnings=[],
        errors=[],
        llm_usage={"calls": 0, "input_tokens": 0, "output_tokens": 0},
        halted=False,
        halt_reason="",
        halt_kind="",
        replan={},
    )


def state_summary(state: PipelineState) -> str:
    """A compact, human-readable status line.  Also what the CLI prints."""
    done = state.get("completed_phases", [])
    gates = state.get("gates", {})
    failed = [g for g in gates.values() if not g.ok]
    pending = [p for p in state.get("prescriptions", []) if not p.settled]
    return (
        f"run={state.get('run_id')} gen={state.get('generation')} "
        f"phase={state.get('phase')} done={len(done)} "
        f"artifacts={len(state.get('artifacts', {}))} "
        f"gates_failed={len(failed)} prescriptions_open={len(pending)}"
        + (f" HALTED: {state.get('halt_reason')}" if state.get("halted") else "")
    )
