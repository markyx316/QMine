"""Ask an agent to widen a search grid, and make its additions safe to use.

The pipeline's grids are K12 artefacts applied to every corpus. This is the path
by which a grid can instead come from the corpus in front of it — without giving
an agent any authority over the outcome.

The sequence is fixed, and every step is a guardrail:

1. Build a corpus profile from artifacts that exist **before anything is fitted**.
2. `assert_blind` the rendered payload. A score-shaped token anywhere in it aborts
   the call. Pre-registration is the property that makes an addition trustworthy,
   and it is worth losing a proposal to protect.
3. `validate_additions` keeps only what is legal, novel and inside the cap.
4. The widened grid is swept by the same code as before, and the winner is chosen
   by the same rule — with `challenger_beats_incumbent` requiring a proposed value
   to clear the incumbent by more than measured noise.
5. `grade_proposal` records whether any addition actually won, so the mechanism
   can be evaluated over runs instead of believed.

If any step fails, the incumbent grid is used unchanged. Nothing here can make the
pipeline worse than not running it — the failure mode is "no additions", never
"bad additions".
"""
from __future__ import annotations

import json
from typing import Any

from ..ops.propose import GridSpec, NotBlind, ProposalOutcome, assert_blind, validate_additions


def corpus_profile(deps: Any) -> dict[str, Any]:
    """What this corpus is like, with nothing about how well anything scored.

    Deliberately assembled by hand rather than by dumping artifacts: `battery` and
    `granularity` are full of scores, and `representation` carries the alpha sweep.
    A profile built by exclusion would leak the first time someone added a field.
    """
    prof: dict[str, Any] = {}
    try:
        audit = deps.load("data_audit") if deps.has("data_audit") else {}
        prof["n_rows"] = audit.get("n_rows") or audit.get("rows")
        prof["n_unique"] = audit.get("n_unique") or audit.get("unique")
        for k in ("median_chars", "p90_chars", "duplicate_share", "empty_share"):
            if k in audit:
                prof[k] = audit[k]
    except Exception:  # noqa: BLE001
        pass
    try:
        lang = deps.load("language_profile") if deps.has("language_profile") else {}
        shares = lang.get("shares") or {}
        prof["language_shares"] = {k: round(float(v), 4) for k, v in list(shares.items())[:6]}
    except Exception:  # noqa: BLE001
        pass
    try:
        tmpl = deps.load("template_groups") if deps.has("template_groups") else {}
        cov = tmpl.get("coverage") or {}
        prof["n_phrasing_groups"] = cov.get("n_groups")
        prof["phrasing_group_coverage"] = cov.get("union_coverage")
        prof["phrasing_group_names"] = [g.get("name") for g in (tmpl.get("groups") or [])][:12]
    except Exception:  # noqa: BLE001
        pass
    try:
        prof["domain"] = deps.cfg.domain.key
        # `expected_family_range` is deliberately NOT shown. It is a hardcoded
        # per-domain prior — [15, 25] because someone expected that of K12 — and
        # it is precisely the inherited constant this mechanism exists to stop
        # relying on. Showing it to the proposer would anchor it to the answer we
        # are trying to derive from the corpus instead.
    except Exception:  # noqa: BLE001
        pass
    return {k: v for k, v in prof.items() if v is not None}


def propose_grid(
    deps: Any,
    parameter: str,
    incumbent: list[Any],
    spec: GridSpec,
) -> tuple[list[Any], dict[str, Any]]:
    """Return (grid_to_sweep, record). The incumbent grid on any failure."""
    idle = ProposalOutcome(incumbent=list(incumbent))
    if not getattr(deps.cfg, "propose_grids", False) or deps.cfg.fast_mode:
        return list(incumbent), {**idle.as_record(), "skipped": "disabled"}

    from .roles import ProposerAgent

    prof = corpus_profile(deps)
    corpus_txt = json.dumps(prof, ensure_ascii=False, indent=2)
    limits = (f"legal range: {spec.lo} .. {spec.hi}; "
              f"at most {spec.max_additions} additions; "
              f"values within {spec.resolution} of an existing point count as duplicates")
    payload = f"{parameter}\n{incumbent}\n{limits}\n{corpus_txt}"
    try:
        # `k` and `alpha` are the parameter names themselves; `peak`/`chosen` etc.
        # must still be refused, so nothing is allow-listed here.
        assert_blind(payload)
    except NotBlind as exc:
        deps.emit(f"  ⚠ grid proposal for {parameter} ABORTED — {exc}")
        return list(incumbent), {**idle.as_record(), "skipped": f"payload not blind: {exc}"}

    try:
        out = ProposerAgent(deps.agent_ctx(), suffix=f"_{parameter}").run(
            parameter=parameter, corpus=corpus_txt,
            incumbent=json.dumps(list(incumbent)), limits=limits)
    except Exception as exc:  # noqa: BLE001
        deps.emit(f"  grid proposer unavailable for {parameter} "
                  f"({type(exc).__name__}) — sweeping the configured grid")
        return list(incumbent), {**idle.as_record(), "skipped": type(exc).__name__}

    res = validate_additions(getattr(out, "add", []) or [], spec, incumbent)
    rec = res.as_record()
    rec.update({
        "rationale": getattr(out, "rationale", "")[:600],
        "corpus_signals": list(getattr(out, "corpus_signals", []) or [])[:8],
        # Recorded and deliberately not acted on: dropping a grid point can drop
        # the true optimum, and the compute saved is not worth a silent ceiling.
        "drop_advisory_ignored": list(getattr(out, "drop", []) or [])[:8],
    })
    if res.kept:
        deps.emit(f"  grid proposer[{parameter}]: +{res.kept} "
                  f"({len(res.rejected)} rejected) — {rec['rationale'][:100]}")
    else:
        deps.emit(f"  grid proposer[{parameter}]: nothing usable proposed "
                  f"({len(res.rejected)} rejected); sweeping the configured grid")
    return res.widened, rec
