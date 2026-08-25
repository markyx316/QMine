"""Live phase observation, with the two guardrails that make it worth running.

**Guardrail 1 — every observation must cite an artifact key that exists.** The
agent writes the path it read (`granularity.triangulation.locator`); this module
resolves it against the artifacts actually handed over. An observation whose
citation does not resolve is dropped before anyone sees it. This is the same
discipline `agents.verify` applies to numbers, for the same reason: a claim that
cannot be traced to an artifact is not evidence, and a review that mixes traceable
and untraceable claims costs more to read than it saves.

**Guardrail 2 — the output must reach something that acts.** This codebase has
already run the other experiment. A critic agent identified the kappa defect
*before* the run that shipped it, its finding was written to an artifact, and
nothing consumed it — so the defect shipped anyway. Observations here are turned
into a gate on the phase that produced them, and a `blocking` observation makes
that gate fail. An observation that only lands in a JSON file is a note to nobody.

**The observer WARNS by default; it does not halt.** `pN_observer` is not in
`cfg.gates.blocking`, so a `blocking` observation records the gate as *warned* and
the run continues to completion. That is a deliberate asymmetry: a false positive
from an LLM should not be able to kill a 25-minute paid run, while a true positive
still reaches the operator through the gate ledger the report prints. Add the gate
name to `cfg.gates.blocking` on a corpus where you have calibrated the observer
and want it to stop the line.

**What this deliberately cannot do.** It cannot change a parameter, re-run a
phase, or overrule a metric. The pipeline's decision authority stays exactly where
it is: measured quantities select, agents describe. An observer that could retune
the thing it is judging would be marking its own work.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

_SEVERITIES = ("blocking", "warn", "note")


@dataclass
class ObserverResult:
    kept: list[Any] = field(default_factory=list)
    dropped: list[tuple[Any, str]] = field(default_factory=list)
    checked: list[str] = field(default_factory=list)
    #: The gate this observation produced. `deps.gate()` returns a GateResult and
    #: registers nothing, so the CALLING NODE has to place it in state — the first
    #: version dropped it on the floor and the observer's verdict reached the log
    #: and nowhere else, which is the exact failure this module was written
    #: against. `as_state_gates()` is what a node merges into its return.
    gate: Any = None

    def as_state_gates(self) -> dict[str, Any]:
        return {self.gate.name: self.gate} if self.gate is not None else {}

    @property
    def blocking(self) -> list[Any]:
        return [o for o in self.kept if getattr(o, "severity", "") == "blocking"]

    @property
    def warnings(self) -> list[Any]:
        return [o for o in self.kept if getattr(o, "severity", "") == "warn"]


def resolve_key(key: str, artifacts: dict[str, Any]) -> tuple[bool, Any]:
    """Resolve a dotted/indexed artifact path, e.g. `panel.sets.leaves.metrics`.

    Returns (found, value). Tolerant of the shapes an agent actually writes —
    bracket indices, a leading artifact name, trailing punctuation — because the
    guardrail should reject uncited claims, not merely badly-punctuated ones.
    """
    if not key or not isinstance(key, str):
        return False, None
    cur: Any = artifacts
    parts = [p for p in re.split(r"[.\[\]]+", key.strip().strip("`.,;: ")) if p]
    for p in parts:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        elif isinstance(cur, (list, tuple)) and p.isdigit() and int(p) < len(cur):
            cur = cur[int(p)]
        else:
            return False, None
    return True, cur


def verified_observations(raw: Any, artifacts: dict[str, Any]) -> ObserverResult:
    """Keep only observations whose citation resolves and whose severity is valid."""
    res = ObserverResult(checked=list(getattr(raw, "checked", []) or []))
    for o in (getattr(raw, "observations", None) or []):
        sev = str(getattr(o, "severity", "") or "").lower().strip()
        if sev not in _SEVERITIES:
            res.dropped.append((o, f"unknown severity {sev!r}"))
            continue
        if not str(getattr(o, "claim", "") or "").strip():
            res.dropped.append((o, "empty claim"))
            continue
        found, _ = resolve_key(getattr(o, "artifact_key", ""), artifacts)
        if not found:
            res.dropped.append((o, f"cites {getattr(o, 'artifact_key', '')!r}, "
                                   "which is not in this phase's artifacts"))
            continue
        o.severity = sev
        res.kept.append(o)
    return res


def observe_phase(
    deps: Any,
    phase: str,
    artifacts: dict[str, Any],
    *,
    decisions: Any = None,
    gates: Any = None,
    blocking_enabled: bool = True,
) -> ObserverResult:
    """Run the observer over one phase and record a gate on what it found.

    The gate is the point. `blocking` observations fail it, so the run stops at
    the phase that produced the problem instead of surfacing it in a report weeks
    later — and `warn` observations reach the operator in the gate ledger the
    deliverable already prints.
    """
    from .roles import ObserverAgent

    def _j(x: Any, limit: int = 60000) -> str:
        try:
            return json.dumps(x, ensure_ascii=False, default=str)[:limit]
        except Exception:  # noqa: BLE001
            return str(x)[:limit]

    try:
        raw = ObserverAgent(deps.agent_ctx(), suffix=f"_{phase}").run(
            phase=phase, artifacts=_j(artifacts),
            decisions=_j(decisions or [], 12000), gates=_j(gates or {}, 8000))
    except Exception as exc:  # noqa: BLE001
        # An observer that cannot run must not stop a run that is otherwise fine.
        # It is a second opinion, not a dependency.
        deps.emit(f"  observer for {phase} unavailable ({type(exc).__name__}) — "
                  "continuing without a second opinion on this phase")
        return ObserverResult()

    res = verified_observations(raw, artifacts)
    for o, why in res.dropped:
        deps.emit(f"  ⚠ observation dropped — {why}: {str(getattr(o, 'claim', ''))[:90]}")
    if res.kept:
        deps.emit(f"  observer[{phase}]: {len(res.blocking)} blocking, "
                  f"{len(res.warnings)} warn, {len(res.kept)} kept "
                  f"({len(res.dropped)} dropped as uncited)")
        for o in res.kept:
            deps.emit(f"    [{o.severity}] {o.claim[:120]}  ← {o.artifact_key}")

    res.gate = deps.gate(
        f"{phase}_observer", phase,
        passed=not (blocking_enabled and res.blocking),
        observed={"n_blocking": len(res.blocking), "n_warn": len(res.warnings),
                  "n_dropped_uncited": len(res.dropped),
                  "claims": [o.claim[:140] for o in res.kept[:6]]},
        threshold={"blocking_allowed": 0},
        message=(f"observer found nothing blocking in {phase}"
                 if not res.blocking else
                 f"observer flagged {len(res.blocking)} blocking issue(s) in {phase}: "
                 + "; ".join(o.claim[:110] for o in res.blocking[:3])),
        remediation=("Each blocking observation cites an artifact key. Read that key, "
                     "decide whether the phase's conclusion actually follows from it, "
                     "and either fix the phase or record why the observation is wrong. "
                     "Do not silence the observer to make the gate pass."),
    )
    return res
