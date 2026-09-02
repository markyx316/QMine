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

**Guardrail 3 — a claim that can be settled mechanically IS settled, before it
counts for anything.** An observation may carry a `check`: the assertion that
should hold if the artifacts are sound. `ops.checks` evaluates it against those
same artifacts, and the three outcomes are not symmetric.

- **confirmed** (the assertion fails) — the finding is now a *measurement*. Only
  these are allowed to fail the gate, and only these are ever blocking.
- **refuted** (the assertion holds) — the observer's own test contradicts it, so
  it is dropped exactly like an uncited claim. This is a free false-positive
  filter, paid for by the agent rather than by the reader.
- **unverifiable** (no check, or it does not evaluate) — advisory, as before.

This is what the observer was actually missing on live39. It reported
`n_leaves = 29` against a `leaves_per_family` summing to 32; the claim was true
and mechanically checkable, and nothing checked it, so a human verified it by hand
a day later. The observer never needed permission to act — it needed a way to be
proven right.

**The observer still WARNS by default; it does not halt.** `pN_observer` is not in
`cfg.gates.blocking`, so even a confirmed blocking observation records the gate as
*warned* and the run finishes. A false positive from an LLM should not kill a
25-minute paid run. What changed is that the finding no longer evaporates: it is
written to the run-level ledger in `ops/findings.py`, inherited by every later
generation, and closes only when its own check passes again. Add the gate name to
`cfg.gates.blocking` to stop the line — which is now defensible, because a
confirmed finding is an assertion that failed rather than an opinion.

**What this deliberately cannot do.** It cannot change a parameter, re-run a
phase, edit an artifact, or overrule a metric — and giving it any of those would
be the mistake, not the fix. The authority a confirmed check earns is exactly the
authority a failing assertion has: name what is wrong and refuse to be forgotten.
Choosing the remedy stays with the operator. The pipeline's rule is unchanged:
measured quantities decide, agents describe. What is new is that an agent can now
hand over a measurement instead of a description.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_SEVERITIES = ("blocking", "warn", "note")


@dataclass
class ObserverResult:
    kept: list[Any] = field(default_factory=list)
    dropped: list[tuple[Any, str]] = field(default_factory=list)
    checked: list[str] = field(default_factory=list)
    #: One CheckResult per kept observation, index-aligned with `kept`.
    check_results: list[Any] = field(default_factory=list)
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
        """Blocking observations that a check CONFIRMED.

        A blocking observation whose check could not be evaluated is a warning,
        not a halt. The severity is the observer's estimate of consequence; the
        verdict is whether the claim is true at all, and only the second one is
        measured. Letting an unverified `blocking` fail the gate would hand an
        LLM's confidence the authority this module exists to withhold.
        """
        return [o for o in self.kept
                if getattr(o, "severity", "") == "blocking"
                and getattr(o, "_verdict", "") == "confirmed"]

    @property
    def unverified_blocking(self) -> list[Any]:
        return [o for o in self.kept
                if getattr(o, "severity", "") == "blocking"
                and getattr(o, "_verdict", "") != "confirmed"]

    @property
    def confirmed(self) -> list[Any]:
        return [o for o in self.kept if getattr(o, "_verdict", "") == "confirmed"]

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


def citable_namespace(artifacts: dict[str, Any], **shown: Any) -> dict[str, Any]:
    """Everything the agent was SHOWN — which is exactly what it may cite.

    `observe()` hands the observer `artifacts` AND `decisions` AND `gates`; the
    delivery auditor is also handed `findings`. Resolution used `artifacts`
    alone, so a claim about a decision record or an open finding could never
    resolve no matter how true it was. Measured across three runs, **8 of 20
    dropped observations (40%)** cited exactly these side channels.

    It is not only the citation. `verified_observations` passes the same mapping
    to `ops.checks.evaluate`, so widening one without the other would admit the
    claim with a check that cannot run — silently demoting a measurable claim to
    advisory. One namespace feeds both.

    On live44 this deleted the pre-delivery audit's finding that `00_索引.md`
    claims 21 L1 classes where the taxonomy has 20. It was correct, and it was
    the last check before delivery.

    A side channel never shadows a real artifact: an artifact named `gates`
    keeps its meaning, and the side channel is dropped rather than overwriting
    it. (Verified on live44: none of these names collides.)
    """
    ns = dict(artifacts)
    for name, value in shown.items():
        if value is None or name in ns:
            continue
        ns[name] = value

    # DECISIONS ARE ALSO ADDRESSABLE BY THEIR ID, because that is how an agent
    # naturally cites one. `decisions` is a LIST, so only `decisions.2.evidence`
    # resolved — and on med01 two real observations were dropped citing
    # `D003.evidence.critic_verdict`, which is the form the decision record
    # itself prints. Position is an implementation detail; the id is the name.
    for d in (shown.get("decisions") or []):
        did = str((d.get("id") if isinstance(d, dict) else getattr(d, "id", "")) or "")
        if did and did not in ns:
            ns[did] = d
    return ns


def verified_observations(raw: Any, artifacts: dict[str, Any]) -> ObserverResult:
    """Keep observations that cite a real artifact and survive their own check.

    Order matters. Citation first, because an uncited claim cannot be checked
    either; then the check, because a claim the observer's own assertion refutes
    should never reach a reader, whatever it cites.
    """
    from ..ops.checks import evaluate

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
        chk = evaluate(str(getattr(o, "check", "") or ""), artifacts)
        if chk.verdict == "refuted":
            # The observer supplied the test and the test says it is wrong. This
            # is the cheapest true finding in the system: an agent's own claim
            # falsified before it costs a reader any attention.
            res.dropped.append((o, f"its own check REFUTES it — `{chk.expression}` "
                                   "evaluates true, so the artifacts are consistent"))
            continue
        o.severity = sev
        # `_verdict` rather than a schema field: this is the pipeline's finding
        # about the observation, not the observation's own claim about itself,
        # and the two must not be confusable in the record.
        try:
            o._verdict = chk.verdict
        except (AttributeError, ValueError):  # a frozen model still gets counted
            pass
        res.kept.append(o)
        res.check_results.append(chk)
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
        """Serialise for the prompt, dropping WHOLE entries rather than bytes.

        This was `json.dumps(x)[:limit]` — a raw string slice that cuts JSON
        mid-token and logs nothing. Measured on med02: the taxonomy artifact
        alone serialises to **59,959 characters against a 60,000 limit**, and the
        artifacts payload carries more than the taxonomy, so the observer was
        handed unparseable JSON on every phase. Its p2a observer reported exactly
        that — the key cut at `self_consistency_ka…` — and the observation was
        then DROPPED for citing a key the truncation had mangled, which reads as
        the agent's fault.

        So: drop whole top-level entries, say which, and never hand an agent
        JSON that does not parse.
        """
        try:
            whole = json.dumps(x, ensure_ascii=False, default=str)
        except Exception:  # noqa: BLE001
            return str(x)[:limit]
        if len(whole) <= limit:
            return whole
        if not isinstance(x, dict):
            # A list or scalar has no entries to drop; say it was cut rather
            # than pretending the fragment is the whole.
            return whole[:limit] + f'\n… [TRUNCATED at {limit} chars — not valid JSON]'

        kept: dict[str, Any] = {}
        withheld: list[str] = []
        used = 2
        for key, val in sorted(x.items(), key=lambda kv: len(str(kv[1]))):
            piece = json.dumps({key: val}, ensure_ascii=False, default=str)[1:-1]
            if used + len(piece) + 1 > limit - 220:
                withheld.append(str(key))
                continue
            kept[key] = val
            used += len(piece) + 1
        out = json.dumps(kept, ensure_ascii=False, default=str)
        if withheld:
            out = out[:-1] + ', "__withheld__": ' + json.dumps(
                {"note": "these artifacts did not fit and are NOT shown; "
                         "do not cite or conclude anything about them",
                 "keys": sorted(withheld)}, ensure_ascii=False) + "}"
        return out

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

    res = verified_observations(
        raw, citable_namespace(artifacts, decisions=decisions, gates=gates))
    for o, why in res.dropped:
        deps.emit(f"  ⚠ observation dropped — {why}: {str(getattr(o, 'claim', ''))[:90]}")
    if res.kept:
        deps.emit(f"  observer[{phase}]: {len(res.confirmed)} CONFIRMED by check, "
                  f"{len(res.kept) - len(res.confirmed)} unverified, "
                  f"{len(res.kept)} kept ({len(res.dropped)} dropped)")
        for o, chk in zip(res.kept, res.check_results):
            mark = "✔MEASURED" if chk.verdict == "confirmed" else "unverified"
            deps.emit(f"    [{o.severity}/{mark}] {o.claim[:110]}  ← {o.artifact_key}")
            if chk.verdict == "confirmed":
                deps.emit(f"        check FAILED: {chk.expression}")

    _record_findings(deps, phase, res)

    n_unver = len(res.unverified_blocking)
    if res.blocking:
        msg = (f"observer flagged {len(res.blocking)} blocking issue(s) in {phase}, "
               f"CONFIRMED by their own checks: "
               + "; ".join(o.claim[:110] for o in res.blocking[:3]))
    elif n_unver:
        # Said plainly, because "nothing blocking" would be the wrong summary of
        # a run where the observer was worried and could not prove it.
        msg = (f"observer raised {n_unver} blocking concern(s) in {phase} that no check "
               "could settle — advisory, not measured: "
               + "; ".join(o.claim[:100] for o in res.unverified_blocking[:2]))
    else:
        msg = f"observer found nothing blocking in {phase}"

    res.gate = deps.gate(
        f"{phase}_observer", phase,
        passed=not (blocking_enabled and res.blocking),
        observed={"n_blocking_confirmed": len(res.blocking),
                  "n_blocking_unverified": n_unver,
                  "n_warn": len(res.warnings), "n_dropped": len(res.dropped),
                  "n_checks_run": sum(1 for c in res.check_results
                                      if c.verdict != "unverifiable"),
                  "claims": [o.claim[:140] for o in res.kept[:6]],
                  "confirmed_checks": [c.expression for c in res.check_results
                                       if c.verdict == "confirmed"][:6]},
        threshold={"blocking_allowed": 0,
                   "rule": "only an observation whose own check FAILED can fail this gate"},
        message=msg,
        remediation=("A CONFIRMED finding is an assertion that failed against the "
                     "artifacts, not an opinion — read the check, fix the phase, and the "
                     "finding closes itself when the assertion holds again. An "
                     "unverified one cites an artifact key: read it and decide. Either "
                     "way it is now in the run's findings ledger and survives into the "
                     "next generation. Do not silence the observer to make this pass."),
    )
    return res


def _record_findings(deps: Any, phase: str, res: ObserverResult) -> None:
    """Put every kept observation into the run-level ledger.

    Wrapped in its own try/except for the same reason the agent call is: a
    bookkeeping failure must not take down a phase that produced good artifacts.
    A finding that cannot be filed is still emitted above, so nothing is lost
    silently.
    """
    try:
        from ..ops.findings import FINDINGS_FILE, FindingLedger

        led = FindingLedger(Path(deps.store.root) / FINDINGS_FILE)
        seen_at = f"{getattr(deps, 'run_id', '')}/{Path(deps.store.gen_dir).name}"
        for o, chk in zip(res.kept, res.check_results):
            led.record(phase=phase, severity=o.severity, claim=o.claim,
                       artifact_key=o.artifact_key, evidence=str(getattr(o, "evidence", ""))[:400],
                       check=chk.expression, verdict=chk.verdict, seen_at=seen_at)
        led.save(run_id=getattr(deps, "run_id", ""), generation=seen_at)
    except Exception as exc:  # noqa: BLE001
        deps.emit(f"  ⚠ could not update the findings ledger ({type(exc).__name__})")
