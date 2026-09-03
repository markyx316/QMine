"""Assembling the twelve-phase graph.

The topology is mostly a chain, and that is the correct answer here.  A
supervisor that re-decides the phase order on every step would be spending model
calls to rediscover a sequence that is fixed by the methodology — you cannot
sweep alpha before you have an encoder, or name clusters before you have any.
LangGraph earns its place through *durability, isolation, and interruption*
rather than through dynamic routing:

* **durability** — a checkpoint after every node, so a run that dies in Phase 9
  resumes at Phase 9 rather than re-encoding 50,000 rows;
* **isolation** — Phase 7's ``Send`` fan-out gives each naming agent a state
  containing only its own payload, which is what makes blind review structural;
* **interruption** — ``interrupt()`` at the human review points, so a reviewer's
  veto is a first-class control-flow event rather than a note in a document.

The one piece of genuine routing is the gate router: a failed blocking gate
halts the run instead of letting later phases build on a foundation that already
failed its own test.
"""

from __future__ import annotations

import functools
import time
import logging
from typing import Any, Callable, Literal

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from ..config import QMineConfig
from ..state import PipelineState
from .deps import Deps
from .nodes import bottomup, delivery, foundation, naming, topdown

log = logging.getLogger("qmine.graph")

#: Phase order, and the node that implements each.
#:
#: THE TWO ROUTES RUN CONCURRENTLY, AND THE BRANCH SHAPES ARE LOad-BEARING.
#:
#: The playbook draws top-down and bottom-up as parallel branches; the graph used
#: to run them as one chain, so the whole bottom-up route waited for the gold set.
#: Measured on live39: p2a 38 min + p2b 69 min = 107 min of provider latency in
#: front of 39 min of bottom-up CPU work that depends on none of it.
#:
#: The real dependency, derived from what the nodes actually read: `p3_represent`
#: consumes only `template_groups`, from p1. `p2c_classifier` is the genuine join
#: — its feature recipe concatenates the dense embedding p3 selects, and it needs
#: the gold set from p2b.
#:
#: `BRANCHES` must stay the same LENGTH. langgraph 1.2.11 advances parallel
#: branches in supersteps and a fan-in node fires once per incoming edge unless
#: they all arrive in the same step; equal lengths make `p2c` receive both at
#: once. `_wrap` also guards on `phase_status`, so an imbalance costs wall clock
#: rather than a double-trained classifier — but the balance is what makes the
#: schedule good, and `test_the_two_branches_stay_the_same_length` pins it.
TOPDOWN_BRANCH: list[tuple[str, Callable]] = [
    ("p2a_taxonomy", topdown.p2a_taxonomy),
    ("p2b_gold", topdown.p2b_gold),
]
BOTTOMUP_BRANCH: list[tuple[str, Callable]] = [
    ("p3_represent", bottomup.p3_represent),
    # p4 + p5 + p6 in one node: see `bottomup.p456_tree` for why this grouping
    # and not another. Two-against-two is the only one that hides the whole
    # bottom-up branch inside p2b.
    ("p456_tree", bottomup.p456_tree),
]
#: Where the branches join, and everything after it.
JOIN_NODE = "p2c_classifier"

PHASE_NODES: list[tuple[str, Callable]] = [
    ("p0_foundation", foundation.p0_foundation),
    ("p1_audit", foundation.p1_audit),
    *TOPDOWN_BRANCH,
    *BOTTOMUP_BRANCH,
    ("p2c_classifier", topdown.p2c_classifier),
    ("p2d_validate", topdown.p2d_validate),
    ("p2e_subintents", topdown.p2e_subintents),
    ("p7_prepare", naming.p7_prepare),
    # p7_name_shard is reached only by Send
    ("p7_audit", naming.p7_audit),
    ("p8_governance", naming.p8_governance),
    ("p9_panel", delivery.p9_panel),
    ("p10_deploy", delivery.p10_deploy),
    # Fires ONLY when the corpus carries more than one snapshot; a no-op
    # otherwise, which is every single-input run. Placed here because it reads
    # the DELIVERED labels (p10) and its artifact becomes a p11 document.
    ("p10b_drift", delivery.p10b_drift),
    ("p11_report", delivery.p11_report),
    ("p12_maintain", delivery.p12_maintain),
]

#: The strictly sequential tail: everything from the join onwards.
SEQUENTIAL_TAIL: list[str] = [
    n for n, _ in PHASE_NODES
    if n not in {"p0_foundation", "p1_audit"}
    and n not in {x for x, _ in TOPDOWN_BRANCH}
    and n not in {x for x, _ in BOTTOMUP_BRANCH}
]

#: Where a human is asked to look before the run continues (Principle 2).
HUMAN_REVIEW_AFTER = {
    "p2a_taxonomy": "p2a_taxonomy",
    "p7_audit": "p7_tree",
    "p9_panel": "p9_panel",
}


def _wrap(fn: Callable, deps: Deps, name: str) -> Callable:
    """Bind deps, record phase status, and convert exceptions into halts.

    A node that raises would otherwise lose the entire run; catching here means
    the checkpoint still records what completed, and the operator can inspect
    state, fix the cause, and resume.
    """

    @functools.wraps(fn)
    def node(state: PipelineState) -> dict[str, Any]:
        if state.get("halted"):
            return {}
        # A FAN-IN NODE CAN BE INVOKED ONCE PER INCOMING EDGE.
        #
        # Measured on langgraph 1.2.11: a node with two incoming edges runs once
        # if both edges arrive in the same superstep, and once PER EDGE if they
        # do not. `p2c_classifier` is the join of the top-down and bottom-up
        # branches, so the moment those branches differ in length — which any
        # future phase addition would do — it would train the classifier twice
        # and register every artifact twice.
        #
        # Guarding on the phase's own recorded status makes that structural
        # rather than a property of how the branches happen to be balanced. It is
        # safe on resume: `thread_id` is per GENERATION, so a new generation
        # starts with an empty `phase_status` and re-runs everything it should.
        if (state.get("phase_status") or {}).get(name) == "ok":
            return {}
        # EMIT COMPLETION RATHER THAN LEAVING THE UI TO INFER IT.
        #
        # The dashboard inferred "phase done" from "next phase started", which is
        # unobservable for the fork: under the p1 branch split the next phase to
        # start is usually on the OTHER branch, and a branch that finishes early
        # then WAITS at the superstep boundary. On live40 that rendered p3 as 72
        # minutes when its work took 12 — the rest was waiting. One line here
        # makes the duration measured instead of guessed, and no inference rule
        # can substitute for it.
        _t0 = time.time()
        try:
            out = fn(state, deps) or {}
            out.setdefault("phase_status", {})[name] = "ok"
            deps.emit(f"✔ {name} completed in {time.time() - _t0:.1f}s")
            return out
        except Exception as exc:  # noqa: BLE001
            log.exception("node %s failed", name)
            deps.emit(f"!! {name} failed: {type(exc).__name__}: {exc}")
            deps.lesson(
                situation=f"node {name} raised {type(exc).__name__}",
                action="ran the phase",
                outcome=str(exc)[:300],
                lesson=f"{name} is not robust to this input; check the artifact it consumes",
                phase=name,
                severity="critical",
            )
            return {
                "halted": True,
                "halt_kind": "crash",
                "halt_reason": f"{name}: {type(exc).__name__}: {exc}",
                "errors": [f"{name}: {exc}"],
                "phase_status": {name: "error"},
            }

    return node


def _shard_node(deps: Deps) -> Callable:
    """The Send target.  Its state is *only* the payload — that is the point."""

    def node(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return naming.p7_name_shard(payload, deps)
        except Exception as exc:  # noqa: BLE001
            log.exception("naming shard failed")
            return {"warnings": [f"naming shard {payload.get('shard_id')} failed: {exc}"]}

    return node


def _halt_node(state: PipelineState) -> dict[str, Any]:
    """Terminal node.  Its whole job is to make the reason legible.

    Without this the run stops with an empty ``halt_reason`` and the operator has
    to reconstruct which gate fired from the log — which is exactly the moment
    they are least inclined to go reading logs.
    """
    if state.get("halted"):
        return {"events": [f"HALTED: {state.get('halt_reason')}"]}
    failed = [g for g in state.get("gates", {}).values() if g.halts_run]
    reason = (
        "; ".join(f"{g.name} {g.status} ({g.message})" for g in failed)
        if failed else "blocking gate failed"
    )
    remediation = " | ".join(g.remediation for g in failed if g.remediation)
    return {
        "halted": True,
        "halt_kind": "gate",
        "halt_reason": reason,
        "events": [f"HALTED by blocking gate: {reason}"]
                  + ([f"remediation: {remediation}"] if remediation else []),
    }


def _gate_router(state: PipelineState) -> Literal["continue", "halt"]:
    """Stop the run if a blocking gate failed or a reviewer vetoed."""
    if state.get("halted"):
        return "halt"
    for g in state.get("gates", {}).values():
        if g.halts_run:
            return "halt"
    return "continue"


def _make_review_node(deps: Deps, gate_name: str, node_name: str) -> Callable:
    """A human checkpoint.

    ``interrupt()`` suspends the graph and persists the pending question in the
    checkpoint, so review can happen minutes or days later, in a different
    process. A rejection is not an argument to win — it is recorded as a veto,
    written to long-term memory, and routed to a fresh generation rather than an
    in-place patch, because the thing the reviewer could not read was produced by
    the configuration we would otherwise be editing around.
    """

    def node(state: PipelineState) -> dict[str, Any]:
        if state.get("halted"):
            return {}
        payload = _review_payload(state, deps, gate_name)
        try:
            answer = interrupt(payload)
        except Exception:  # no checkpointer configured — auto-approve and say so
            return {
                "events": [f"review[{gate_name}]: auto-approved (no checkpointer, so no way to ask)"],
            }
        if isinstance(answer, str):
            answer = {"decision": answer}
        decision = (answer or {}).get("decision", "approve")
        reason = (answer or {}).get("reason", "")
        if decision in ("approve", "accept", "yes", True):
            g = deps.gate(gate_name, node_name, passed=True,
                          observed={"reviewer": "human"}, threshold={"decision": "approve"},
                          message=f"reviewer approved: {reason or 'no comment'}")
            return {"gates": {g.name: g}, "events": [f"review[{gate_name}]: approved"]}

        deps.memory.remember_rejection(
            f"{state.get('run_id')}:{gate_name}",
            {"what": gate_name, "reason": reason, "generation": state.get("generation")},
        )
        deps.lesson(
            situation=f"reviewer looked at {gate_name}",
            action="presented the current output for sign-off",
            outcome=f"rejected: {reason}",
            lesson=(
                "A reviewer saying 'I cannot read this' is a measurement, not an objection to "
                "answer. Re-derive rather than defend."
            ),
            phase=node_name,
            severity="critical",
        )
        g = deps.gate(gate_name, node_name, passed=False,
                      observed={"reviewer": "human", "reason": reason},
                      threshold={"decision": "approve"},
                      message=f"reviewer rejected: {reason}",
                      remediation="Open a new generation and re-derive; do not patch in place.")
        g.status = "rejected"
        g.reviewer = "human"
        return {
            "gates": {g.name: g},
            "halted": True,
            "halt_kind": "review",
            "halt_reason": f"reviewer rejected {gate_name}: {reason}",
            "replan": {"rejected_gate": gate_name, "reason": reason, "action": "new_generation"},
            "events": [f"review[{gate_name}]: REJECTED — {reason}"],
        }

    return node


def _review_payload(state: PipelineState, deps: Deps, gate_name: str) -> dict[str, Any]:
    """What the reviewer is shown.  Enough to judge, short enough to read."""
    base = {
        "gate": gate_name,
        "run_id": state.get("run_id"),
        "generation": state.get("generation"),
        "question": "Approve and continue, or reject with a reason?",
        "respond_with": {"decision": "approve | reject", "reason": "why"},
    }
    if gate_name == "p2a_taxonomy":
        tax = deps.load("taxonomy") if deps.has("taxonomy") else {}
        nodes = tax.get("taxonomy", {}).get("nodes", [])
        base["show"] = {
            "n_l1": len([n for n in nodes if n.get("level", 1) == 1]),
            "n_rules": len(tax.get("taxonomy", {}).get("rules", [])),
            "classes": [{"code": n.get("code"), "name": n.get("name"),
                         "user_need": n.get("user_need", "")[:90]} for n in nodes[:30]],
            "critic_verdict": tax.get("critique", {}).get("verdict"),
        }
    elif gate_name == "p7_tree":
        nm = deps.load("tree_naming") if deps.has("tree_naming") else {}
        base["show"] = {
            "mean_coherence": nm.get("mean_coherence"),
            "families": [f.get("name_zh") for f in nm.get("audit", {}).get("families", [])],
            "n_prescriptions": len(state.get("prescriptions", [])),
            "risk_found_independently": nm.get("independent_risk_discovery", {}).get("found_without_being_told"),
            "leaves": [{"id": n["leaf_id"], "name": n["name_zh"], "coherence": n["coherence"]}
                       for n in nm.get("namings", [])[:40]],
        }
        base["prompt"] = ("Does the family layer read as a coherent set of user intents? "
                          "If any family is unreadable, reject — that judgment is data, not taste.")
    elif gate_name == "p9_panel":
        panel = deps.load("metrics_panel") if deps.has("metrics_panel") else {}
        base["show"] = panel.get("table", {}).get("rows", [])
    return base


def _wire_tail(g: Any, deps: Deps, _route: Callable) -> None:
    """Everything from the join onwards — a strict chain, both topologies."""
    for idx, name in enumerate(SEQUENTIAL_TAIL):
        nxt = SEQUENTIAL_TAIL[idx + 1] if idx + 1 < len(SEQUENTIAL_TAIL) else END
        # Phase 7 fans out to the naming shards before the audit.
        if name == "p7_prepare":
            g.add_conditional_edges(
                "p7_prepare",
                lambda s: naming.fan_out_namers(s, deps) if not s.get("halted") else "p7_audit",
                ["p7_name_shard", "p7_audit"],
            )
            g.add_edge("p7_name_shard", "p7_audit")
            continue
        _route(name, nxt)


def build_graph(
    cfg: QMineConfig,
    deps: Deps,
    *,
    checkpointer: Any = None,
    store: Any = None,
    human_review: bool = False,
) -> Any:
    """Compile the pipeline.

    ``human_review`` is off by default so that automated and CI runs complete
    unattended; turning it on inserts ``interrupt()`` nodes at the points where
    the playbook says a person must look.
    """
    g = StateGraph(PipelineState)

    for name, fn in PHASE_NODES:
        g.add_node(name, _wrap(fn, deps, name))
    g.add_node("p7_name_shard", _shard_node(deps))
    g.add_node("halt", _halt_node)

    review_points = set(cfg.gates.human_review_points) if human_review else set()
    for node_name, gate_name in HUMAN_REVIEW_AFTER.items():
        if gate_name in review_points:
            g.add_node(f"review_{gate_name}", _make_review_node(deps, gate_name, node_name))

    g.add_edge(START, "p0_foundation")

    def _route(name: str, nxt: str) -> None:
        """Wire one node onward through the gate router, via review if configured."""
        gate = HUMAN_REVIEW_AFTER.get(name)
        target = f"review_{gate}" if gate and gate in review_points else nxt
        g.add_conditional_edges(name, _gate_router, {"continue": target, "halt": "halt"})
        if gate and gate in review_points:
            g.add_conditional_edges(f"review_{gate}", _gate_router,
                                    {"continue": nxt, "halt": "halt"})

    # p0 -> p1, then p1 FORKS.
    _route("p0_foundation", "p1_audit")

    td = [n for n, _ in TOPDOWN_BRANCH]
    bu = [n for n, _ in BOTTOMUP_BRANCH]
    if not getattr(cfg, "concurrent_branches", True) or not bu:
        # The strict chain, as it was before the fork: the bottom-up branch runs
        # after the top-down one instead of beside it. Same phases, same order
        # within each branch, no overlap — so a comparison between the two
        # schedules changes only the scheduling.
        chain = td + bu
        for i, name in enumerate(chain):
            _route(name, chain[i + 1] if i + 1 < len(chain) else JOIN_NODE)
        _route("p1_audit", chain[0])
        _wire_tail(g, deps, _route)
        g.add_edge("halt", END)
        return g.compile(checkpointer=checkpointer, store=store, name="qmine-pipeline")
    # Both heads are reached from p1. To start several branches from one node the
    # ROUTER returns a list of node names — a path map whose *value* is a list is
    # rejected at compile time ("unhashable type: 'list'"). The router still gets
    # to halt the run before either branch begins.
    g.add_conditional_edges(
        "p1_audit",
        lambda st: "halt" if _gate_router(st) == "halt" else [td[0], bu[0]],
        [td[0], bu[0], "halt"],
    )
    # Inside each branch, node i -> node i+1; the last one joins.
    for branch in (td, bu):
        for i, name in enumerate(branch):
            _route(name, branch[i + 1] if i + 1 < len(branch) else JOIN_NODE)

    _wire_tail(g, deps, _route)
    g.add_edge("halt", END)
    return g.compile(checkpointer=checkpointer, store=store, name="qmine-pipeline")
