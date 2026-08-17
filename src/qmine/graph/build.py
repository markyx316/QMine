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
#: Note the placement of ``p3_represent`` *between* the gold set and the
#: classifier. The playbook draws the two routes as parallel branches, and they
#: are conceptually independent — but Phase 2c's feature recipe concatenates the
#: dense embedding chosen by Phase 3a, so the top-down classifier genuinely
#: cannot be built before the bottom-up route has picked an encoder. Taxonomy
#: design and gold annotation (2a, 2b) need no embedding and stay first.
PHASE_NODES: list[tuple[str, Callable]] = [
    ("p0_foundation", foundation.p0_foundation),
    ("p1_audit", foundation.p1_audit),
    ("p2a_taxonomy", topdown.p2a_taxonomy),
    ("p2b_gold", topdown.p2b_gold),
    ("p3_represent", bottomup.p3_represent),
    ("p2c_classifier", topdown.p2c_classifier),
    ("p2d_validate", topdown.p2d_validate),
    ("p2e_subintents", topdown.p2e_subintents),
    ("p4_battery", bottomup.p4_battery),
    ("p5_granularity", bottomup.p5_granularity),
    ("p6_hierarchy", bottomup.p6_hierarchy),
    ("p7_prepare", naming.p7_prepare),
    # p7_name_shard is reached only by Send
    ("p7_audit", naming.p7_audit),
    ("p8_governance", naming.p8_governance),
    ("p9_panel", delivery.p9_panel),
    ("p10_deploy", delivery.p10_deploy),
    ("p11_report", delivery.p11_report),
    ("p12_maintain", delivery.p12_maintain),
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
        try:
            out = fn(state, deps) or {}
            out.setdefault("phase_status", {})[name] = "ok"
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

    def _next_after(current: str) -> str:
        order = [n for n, _ in PHASE_NODES]
        i = order.index(current)
        return order[i + 1] if i + 1 < len(order) else END

    g.add_edge(START, "p0_foundation")
    order = [n for n, _ in PHASE_NODES]
    for idx, name in enumerate(order):
        nxt = order[idx + 1] if idx + 1 < len(order) else END

        # Phase 7 fans out to the naming shards before the audit.
        if name == "p7_prepare":
            g.add_conditional_edges(
                "p7_prepare",
                lambda s: naming.fan_out_namers(s, deps) if not s.get("halted") else "p7_audit",
                ["p7_name_shard", "p7_audit"],
            )
            g.add_edge("p7_name_shard", "p7_audit")
            continue

        gate = HUMAN_REVIEW_AFTER.get(name)
        target = f"review_{gate}" if gate and gate in review_points else nxt
        g.add_conditional_edges(
            name,
            _gate_router,
            {"continue": target, "halt": "halt"},
        )
        if gate and gate in review_points:
            g.add_conditional_edges(
                f"review_{gate}",
                _gate_router,
                {"continue": nxt, "halt": "halt"},
            )

    g.add_edge("halt", END)
    return g.compile(checkpointer=checkpointer, store=store, name="qmine-pipeline")
