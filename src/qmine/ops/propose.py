"""Widening a search grid safely: the mechanical half of the candidate proposer.

The grids this pipeline sweeps are K12 artefacts. `alpha_grid` is
`[0.0, 0.1, 0.2, 0.3, 0.5]` under a comment reading "re-run per domain; never
inherit the K12 answer" — while the grid *itself* is the K12 answer. `k_sweep`
runs 8…120. `expected_family_range` is `[15, 25]` because that is what someone
expected of K12. A corpus of support tickets, or of SQL fragments, or of product
searches, has no reason to be well served by any of them, and nothing in the
pipeline notices.

An agent reading *what this corpus is actually like* can propose a grid that fits
it. Everything dangerous about that idea is handled here rather than in the prompt.

**Blindness is the load-bearing guarantee.** The proposer never sees a score. It
proposes from corpus characteristics before anything is fitted, so its additions
are *pre-registered* — it cannot look at where the current optimum sits and crowd
the grid around it, which is the mechanism by which an adaptive search inflates
its own winner. `assert_blind` enforces this on the actual payload, not by
convention.

**Additions are capped, and the cap is a statistical parameter, not a budget
knob.** Every extra candidate is one more comparison, and taking the maximum over
more comparisons inflates the winner most in exactly the near-tie, low
signal-to-noise regime this pipeline occupies. The cap bounds that inflation; the
`challenger_beats_incumbent` toll in `ops/select.py` collects the rest.

**Nothing is ever dropped.** The proposer may say a value looks pointless, and
that opinion is recorded and ignored. Removing grid points to save compute can
remove the true optimum, and the saving is not worth a silent ceiling.

**The proposer is graded.** Every run records whether any proposed value survived
to be chosen. If proposals never win, the proposer is dead weight and the record
says so — an agent nobody can evaluate is an agent nobody should keep.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

#: Substrings that betray a score. If any appears in what the proposer is about
#: to be shown, the payload is not blind and the call must not happen. Kept broad
#: on purpose: a false alarm here costs one refused proposal, while a miss costs
#: the pre-registration guarantee that makes the whole mechanism sound.
SCORE_TOKENS = (
    "silhouette", "stability", "purity", "kappa", "accuracy", "fragmentation",
    "heldout", "reproduction", "coherence", "score", "best_", "chosen",
    "winner", "ranking", "optimal",
)

#: Metric abbreviations short enough to appear inside ordinary words. Matched as
#: WHOLE WORDS only. Substring matching flagged `domain_expected_family_range`,
#: because "ami" is inside "f-ami-ly" — a checker that cries wolf on real corpus
#: fields gets widened until it stops checking anything, which is the failure mode
#: a guardrail can least afford.
SCORE_WORDS = ("ari", "ami", "nmi", "f1", "auc", "peak", "rank")


class NotBlind(RuntimeError):
    """Raised when a proposer payload carries scoring information."""


def assert_blind(payload: str, *, allow: Sequence[str] = ()) -> None:
    """Refuse to show the proposer anything score-shaped.

    Checked against the rendered payload rather than the dict that built it,
    because the payload is what the model actually reads — a nested key three
    levels down still arrives as text.
    """
    low = payload.lower()
    for token in SCORE_TOKENS:
        if token in allow:
            continue
        m = re.search(rf"\b\w*{re.escape(token)}\w*\b", low)
        if m:
            raise NotBlind(
                f"payload contains {m.group(0)!r} — the proposer must not see scores, "
                "or its additions stop being pre-registered"
            )
    for word in SCORE_WORDS:
        if word in allow:
            continue
        # Whole word, allowing an underscore-delimited component: `stability_ari`
        # must be caught, `family` must not.
        if re.search(rf"(?:^|[^a-z0-9])({re.escape(word)})(?:[^a-z0-9]|$)", low):
            raise NotBlind(
                f"payload contains the metric {word!r} — the proposer must not see "
                "scores, or its additions stop being pre-registered"
            )


@dataclass
class GridSpec:
    """What a legal addition to one parameter's grid looks like."""

    name: str
    kind: type = float
    lo: float | None = None
    hi: float | None = None
    max_additions: int = 4
    #: Two proposals closer than this to an existing point are the same point.
    resolution: float = 0.0
    allowed: Sequence[Any] | None = None       # for categorical grids
    coerce: Callable[[Any], Any] | None = None

    def normalise(self, v: Any) -> Any:
        if self.coerce:
            return self.coerce(v)
        return self.kind(v)


@dataclass
class ProposalOutcome:
    kept: list[Any] = field(default_factory=list)
    rejected: list[tuple[Any, str]] = field(default_factory=list)
    incumbent: list[Any] = field(default_factory=list)

    @property
    def widened(self) -> list[Any]:
        """The incumbent grid plus what survived, in a stable order."""
        out = list(self.incumbent) + [v for v in self.kept if v not in self.incumbent]
        try:
            return sorted(out)
        except TypeError:
            return out

    def as_record(self) -> dict[str, Any]:
        return {
            "incumbent": list(self.incumbent),
            "proposed_kept": list(self.kept),
            "proposed_rejected": [{"value": str(v), "why": w} for v, w in self.rejected],
            "n_extra_comparisons": len(self.kept),
            "widened": self.widened,
        }


def validate_additions(
    proposed: Sequence[Any],
    spec: GridSpec,
    incumbent: Sequence[Any],
) -> ProposalOutcome:
    """Keep only additions that are legal, novel, and within the cap.

    Order matters: legality first, then novelty, then the cap — so the cap trims
    genuinely usable candidates rather than being spent on duplicates, and the
    rejection reasons stay honest about which rule bit.
    """
    out = ProposalOutcome(incumbent=list(incumbent))
    seen = list(incumbent)
    for raw in proposed or []:
        if len(out.kept) >= spec.max_additions:
            out.rejected.append((raw, f"cap of {spec.max_additions} additions already reached"))
            continue
        try:
            v = spec.normalise(raw)
        except (TypeError, ValueError):
            out.rejected.append((raw, f"not a {spec.kind.__name__}"))
            continue
        if spec.allowed is not None and v not in spec.allowed:
            out.rejected.append((raw, "not in the allowed set for this parameter"))
            continue
        if spec.lo is not None and v < spec.lo:
            out.rejected.append((raw, f"below the legal minimum {spec.lo}"))
            continue
        if spec.hi is not None and v > spec.hi:
            out.rejected.append((raw, f"above the legal maximum {spec.hi}"))
            continue
        if _is_duplicate(v, seen, spec.resolution):
            out.rejected.append((raw, "already in the grid"))
            continue
        out.kept.append(v)
        seen.append(v)
    return out


def _is_duplicate(v: Any, seen: Sequence[Any], resolution: float) -> bool:
    if v in seen:
        return True
    if not resolution:
        return False
    try:
        return any(abs(float(v) - float(s)) < resolution for s in seen)
    except (TypeError, ValueError):
        return False


def grade_proposal(record: dict[str, Any], chosen: Any) -> dict[str, Any]:
    """Did any proposed value actually win?

    Recorded every run so the proposer can be evaluated rather than believed. A
    proposer whose additions never survive selection is costing tokens and
    comparisons for nothing, and this is the number that says so.
    """
    kept = list(record.get("proposed_kept", []))
    won = chosen in kept
    return {
        **record,
        "chosen": chosen,
        "a_proposed_value_won": bool(won),
        "verdict": ("a proposed candidate was selected" if won else
                    "the incumbent grid still held the winner — "
                    "these additions cost comparisons and returned nothing"),
    }


#: The two grids worth widening, with the limits that keep widening safe.
#: `max_additions` is deliberately small: it is the multiple-comparisons budget.
ALPHA_SPEC = GridSpec(name="alpha", kind=float, lo=0.0, hi=1.0,
                      max_additions=3, resolution=0.02)


def k_spec(n_rows: int, min_leaf_size: int, max_additions: int = 4) -> GridSpec:
    """K's legal range comes from the corpus, not from a constant.

    The upper bound is the point past which the average cluster is smaller than a
    leaf worth naming — a limit the data sets, so it travels to any corpus.
    """
    hi = max(4, int(n_rows / max(1, min_leaf_size)))
    return GridSpec(name="family_k", kind=int, lo=2, hi=hi,
                    max_additions=max_additions, resolution=0.5)
