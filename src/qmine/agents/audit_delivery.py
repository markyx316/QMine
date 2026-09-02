"""The pre-delivery audit: read everything the run warned about, then fix the text.

Every warning this pipeline raises is currently read by a human, later, if at all.
live39 finished with fifteen gates, one of them carrying a real defect, and the
deliverables shipped describing a tree the artifact contradicted. Nothing between
the last phase and the reader ever held the warnings and the documents side by
side.

This does. It is given the gate ledger, the findings ledger, the artifacts and the
finished deliverables in one context, and it is asked the one question nothing
else asks: **did any of these warnings leave a defect in what we are about to
hand over?**

**It may act, and that is deliberate.** Every other agent here describes, because
every other agent could be wrong in a way nothing would catch. This one edits —
and the reason that is safe is not trust, it is the shape of the operation. An
edit is an anchored replacement whose anchor must be proven unique, whose numbers
must come from the artifact it cites, and whose language is checked. `ops/edits.py`
holds each rule and the failure it was written against. The agent chooses *where*
and *what*; the mechanism decides *whether*.

**The report is not optional output, it is the point.** An audit that silently
improved some documents would be worse than no audit: the reader would have no way
to know what changed or why. So every applied edit is written down with its reason
and its citation, every REFUSED edit is written down with the rule that refused it,
and every warning the auditor considered and dismissed is written down with why.
A refusal is a finding about the auditor, and hiding it would turn the report into
a list of successes.

**The bounds, stated plainly.** It edits `.md` deliverables only — never an
artifact, never code, never a parameter, never the executable notebook. A report
is a *description* of a measurement, and correcting it changes no measurement;
editing an artifact would change the measurement itself, and no agent gets that.
Defects it cannot fix with an anchored edit go to the findings ledger, where they
survive into the next generation instead of being lost.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..ops.edits import MAX_EDITS, ProposedEdit, apply_edits


def _j(x: Any, limit: int) -> str:
    try:
        return json.dumps(x, ensure_ascii=False, default=str)[:limit]
    except Exception:  # noqa: BLE001
        return str(x)[:limit]


def _gate_bundle(state: Any) -> list[dict[str, Any]]:
    """Every gate, with the ones that did NOT simply pass first.

    Passing gates are included on purpose: a gate that passed on a threshold that
    was never applied is exactly the kind of defect worth auditing, and it can
    only be seen by reading the passing ones too.
    """
    out = []
    for name, g in (state.get("gates") or {}).items():
        rec = {
            "gate": name,
            "status": getattr(g, "status", None) or (g.get("status") if isinstance(g, dict) else None),
            "passed": getattr(g, "passed", None) if not isinstance(g, dict) else g.get("passed"),
            "message": (getattr(g, "message", "") or "") if not isinstance(g, dict) else g.get("message", ""),
            "observed": (getattr(g, "observed", None) if not isinstance(g, dict) else g.get("observed")),
        }
        out.append(rec)
    out.sort(key=lambda r: (r.get("status") == "passed", r.get("passed") is True))
    return out


def audit_deliverables(state: Any, deps: Any) -> dict[str, Any]:
    """Run the audit and apply what passes. Returns the record for the report."""
    from ..ops.findings import FINDINGS_FILE, FindingLedger, all_json_artifacts
    from .roles import DeliveryAuditorAgent

    gen = Path(deps.store.gen_dir)
    idle = {"ran": False, "n_applied": 0, "n_refused": 0}
    if not getattr(deps.cfg, "audit_delivery", False) or deps.cfg.smoke_mode:
        return {**idle, "skipped": "disabled"}

    docs = {p.name: p.read_text(encoding="utf-8") for p in sorted(gen.glob("*.md"))
            if not p.name.endswith(".pre_audit.md")}
    if not docs:
        return {**idle, "skipped": "no deliverables to audit"}

    artifacts = all_json_artifacts(gen)
    led = FindingLedger(Path(deps.store.root) / FINDINGS_FILE)

    # THE AUDITOR IS GIVEN GATES AND FINDINGS, SO IT MUST BE ABLE TO CITE THEM.
    #
    # `apply_edits` resolves a citation against this dict and requires every
    # number in the replacement to come from the cited subtree. Gates live in
    # `run_summary.json` under `gates`, not as top-level artifact keys — so an
    # auditor doing exactly what it was asked (read the warnings, correct what
    # they left wrong, cite the source) had its edits refused as "unsourced".
    #
    # Measured on live40: 3 of 4 correct corrections were rejected this way,
    # including a report claiming adversarial accuracy was higher than
    # cross-validation when it was lower. Adding them keeps the pool small — a
    # single gate's `observed` is exactly the right scope for a claim about that
    # gate — while removing a refusal that punished the intended behaviour.
    gate_ns = {name: {"status": getattr(g, "status", None) or (
                          g.get("status") if isinstance(g, dict) else None),
                      "passed": getattr(g, "passed", None) if not isinstance(g, dict)
                                else g.get("passed"),
                      "message": (getattr(g, "message", "") if not isinstance(g, dict)
                                  else g.get("message", "")),
                      "observed": (getattr(g, "observed", None) if not isinstance(g, dict)
                                   else g.get("observed"))}
                for name, g in (state.get("gates") or {}).items()}
    artifacts = {**artifacts, "gates": gate_ns,
                 "findings": {f.id: f.as_record() for f in led.open_findings}}
    lang = getattr(deps.cfg, "report_language", "zh")

    # A LIST of whole documents. Joined into one blob, the budget cut the middle
    # and the auditor could not tell it had been shown a third of them.
    deliverables = [f"=== FILE: {name} ===\n{text}" for name, text in docs.items()]

    try:
        out = DeliveryAuditorAgent(deps.agent_ctx()).run(
            deliverables=deliverables,
            gates=_j(_gate_bundle(state), 24000),
            findings=_j([f.as_record() for f in led.open_findings], 12000),
            artifacts=_j(artifacts, 60000),
            language=lang,
        )
    except Exception as exc:  # noqa: BLE001
        # An audit that cannot run must not stop a delivery that is otherwise
        # complete. It is a last check, not a dependency — and saying so beats a
        # report that silently claims to have been audited.
        deps.emit(f"  pre-delivery audit unavailable ({type(exc).__name__}) — "
                  "deliverables ship unaudited, and the report says so")
        return {**idle, "skipped": type(exc).__name__}

    proposed = [
        ProposedEdit(file=e.file, anchor=e.anchor, replacement=e.replacement,
                     reason=e.reason, artifact_key=e.artifact_key,
                     severity=str(getattr(e, "severity", "warn")), check=str(getattr(e, "check", "")))
        for e in (getattr(out, "edits", None) or [])
    ]
    res = apply_edits(gen, proposed, artifacts, language=lang, max_edits=MAX_EDITS)

    deps.emit(f"  pre-delivery audit: {len(res.applied)} edit(s) applied, "
              f"{len(res.refused)} refused, {len(getattr(out, 'unfixable', []) or [])} "
              "defect(s) recorded as unfixable by edit")
    for a in res.applied:
        deps.emit(f"    ✎ {a['file']}: {a['reason'][:100]}")
    for r in res.refused:
        deps.emit(f"    ✗ REFUSED {r['file']}: {r['why'][:110]}")

    # A defect the auditor could not fix is not lost. It goes where every other
    # unresolved finding goes, and survives into the next generation — but it
    # goes through the SAME verification every observation does. Recording them
    # raw put three empty-claim rows into the ledger on the first end-to-end run:
    # a ledger that accumulates blanks is a ledger nobody reads, which is the
    # failure this whole mechanism exists to end.
    from .observe import citable_namespace, verified_observations

    seen_at = f"{getattr(deps, 'run_id', '')}/{gen.name}"
    # The auditor is SHOWN gates and the open findings ledger alongside the
    # artifacts, so a claim citing either is a claim about something it was
    # handed. Resolving against `artifacts` alone deleted its two real
    # cross-document contradictions on live44 — including that `00_索引.md`
    # claims 21 L1 classes where the taxonomy has 20.
    ver = verified_observations(
        type("_Raw", (), {"observations": list(getattr(out, "unfixable", None) or []),
                          "checked": []})(),
        citable_namespace(
            artifacts,
            gates=_gate_bundle(state),
            findings={f.id: f.as_record() for f in led.open_findings},
        ))
    for o, why in ver.dropped:
        deps.emit(f"    ⚠ unfixable finding dropped — {why}: {str(getattr(o, 'claim', ''))[:70]}")
    for o, chk in zip(ver.kept, ver.check_results):
        led.record(phase="p11_audit", severity=str(getattr(o, "severity", "warn")),
                   claim=str(getattr(o, "claim", "")),
                   artifact_key=str(getattr(o, "artifact_key", "")),
                   evidence=str(getattr(o, "evidence", ""))[:400],
                   check=chk.expression, verdict=chk.verdict, seen_at=seen_at)
    led.save(run_id=getattr(deps, "run_id", ""), generation=seen_at)

    return {
        "ran": True,
        "n_applied": len(res.applied),
        "n_refused": len(res.refused),
        "summary": str(getattr(out, "summary", ""))[:2000],
        "dismissed": [str(x)[:300] for x in (getattr(out, "dismissed", None) or [])][:20],
        "unfixable": [
            {"severity": str(getattr(o, "severity", "")), "claim": str(getattr(o, "claim", "")),
             "artifact_key": str(getattr(o, "artifact_key", "")),
             "verdict": getattr(o, "_verdict", "unverifiable")}
            for o in ver.kept
        ][:20],
        "n_unfixable_dropped": len(ver.dropped),
        **res.as_record(),
    }
