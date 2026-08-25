"""Selecting among candidates when no candidate wins on every metric.

Three separate problems live here, and keeping them separate is the point.

**1. Domination is a measurement; preference is not.** When several metrics
disagree, the options that are worse on every metric can be removed mechanically
— that is the Pareto frontier, and it needs no judgement. What remains cannot be
ordered without saying which metric matters more, and no amount of data supplies
that. It is a decision, and the honest thing is to make it explicit rather than to
bury it in a hardcoded priority list that reads like a measurement.

**2. Most "trade-offs" on this corpus are not real.** Replay stability's seed-to-
seed sd is about 0.10 while the differences it was being asked to resolve between
adjacent K are about 0.05: every K sat inside every other K's error bar. Before
agonising over a trade-off, test whether the difference exceeds the measurement's
own noise. Very often the correct finding is *these options are tied, the choice
is arbitrary, and here is what we took and why* — which is a far more useful
sentence than a false ranking.

**3. Adding candidates makes selection worse, not better.** Choosing the maximum
over a larger set inflates the winner: selection bias is largest exactly in the
low signal-to-noise, near-tie regime this corpus lives in, and it shrinks only as
the sample grows or the true gap widens. So a proposer that widens the grid must
pay for it — a challenger has to beat the incumbent by more than noise, not merely
score higher. `challenger_beats_incumbent` is that toll.

Nothing in this module is an agent, and that is deliberate: every function here is
reproducible, auditable, and identical on every re-run.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np


def noise_floor(values: Sequence[float]) -> float:
    """Estimate a metric's own noise from the roughness of its curve.

    A sweep over K produces a curve whose low-frequency part is signal and whose
    point-to-point jitter is noise. The median absolute deviation of successive
    differences isolates the jitter, is robust to the trend, and costs nothing
    extra — no refitting, no extra seeds.

    Validated against a figure measured a different way: on live38's 14-point K
    sweep this returns **0.0105** for `intent_alignment_ami`, where the sweep's
    own documentation records a seed-to-seed sd of "~0.01".

    Returns NaN when there are too few points to estimate anything.
    """
    v = np.asarray([x for x in values if x is not None], dtype=float)
    v = v[~np.isnan(v)]
    if v.size < 4:
        return float("nan")
    d = np.diff(v)
    mad = float(np.median(np.abs(d - np.median(d))))
    # 1.4826 converts MAD to a sd equivalent; /sqrt(2) because each difference
    # carries the noise of two points.
    return 1.4826 * mad / np.sqrt(2.0)


def tie_set(
    rows: Sequence[dict[str, Any]],
    metric: str,
    *,
    higher_is_better: bool = True,
    z: float = 2.0,
    se: float | None = None,
    fallback_band: float | None = None,
) -> tuple[list[dict[str, Any]], float, str]:
    """Every candidate statistically indistinguishable from the leader.

    Returns (tied_rows, band, how). `se` is measured from the candidates
    themselves unless supplied; `fallback_band` is used only when the sweep is
    too short to estimate noise, and the caller is told which happened — a
    hardcoded band that silently stands in for a measurement is the defect this
    exists to remove.
    """
    live = [r for r in rows if r.get(metric) is not None and not _isnan(r[metric])]
    if not live:
        return [], float("nan"), "no candidate carries this metric"
    best = (max if higher_is_better else min)(live, key=lambda r: r[metric])
    if se is None:
        se = noise_floor([r[metric] for r in live])
    if _isnan(se):
        if fallback_band is None:
            return [best], 0.0, "noise not estimable and no fallback — leader only"
        band, how = fallback_band, f"configured band {fallback_band} (too few points to measure noise)"
    else:
        band, how = z * se, f"{z:g}x the metric's own measured noise (se={se:.4f})"
    tied = [r for r in live
            if abs(best[metric] - r[metric]) <= band]
    return tied, band, how


def pareto_front(
    rows: Sequence[dict[str, Any]],
    objectives: dict[str, bool],
) -> list[dict[str, Any]]:
    """Candidates not beaten on every objective at once.

    `objectives` maps metric name → higher_is_better. A row missing an objective
    cannot be shown to dominate or be dominated on it, so it is kept: dropping a
    candidate for having an unmeasured metric would silently prefer whatever
    happened to be measured.
    """
    live = list(rows)

    def dominates(a: dict[str, Any], b: dict[str, Any]) -> bool:
        strictly_better = False
        for m, hib in objectives.items():
            av, bv = a.get(m), b.get(m)
            if av is None or bv is None or _isnan(av) or _isnan(bv):
                return False
            if av == bv:
                continue
            better = (av > bv) if hib else (av < bv)
            if not better:
                return False
            strictly_better = True
        return strictly_better

    return [a for a in live if not any(dominates(b, a) for b in live if b is not a)]


def challenger_beats_incumbent(
    incumbent: dict[str, Any],
    challenger: dict[str, Any],
    metric: str,
    *,
    higher_is_better: bool = True,
    se: float,
    z: float = 2.0,
) -> tuple[bool, str]:
    """Does a newly-proposed candidate actually displace the current choice?

    Scoring higher is not enough. Widening a search space and taking the new
    maximum is precisely the operation that manufactures winners out of noise,
    and the effect is worst in the near-tie, low signal-to-noise regime this
    pipeline occupies. So the incumbent keeps the seat unless the challenger
    clears it by more than the measurement's own error.

    This is what lets a proposer be useful without being dangerous: it can widen
    the grid all it likes, and a candidate that is merely lucky loses.
    """
    a, b = incumbent.get(metric), challenger.get(metric)
    if a is None or b is None or _isnan(a) or _isnan(b):
        return False, "one side does not carry the metric — incumbent stands"
    gap = (b - a) if higher_is_better else (a - b)
    if _isnan(se) or se <= 0:
        return False, "the metric's noise is not estimable — incumbent stands"
    need = z * se
    if gap > need:
        return True, f"challenger clears the incumbent by {gap:.4f} > {need:.4f} ({z:g} se)"
    return False, (f"challenger's margin {gap:+.4f} does not clear {need:.4f} "
                   f"({z:g} se) — incumbent stands")


@dataclass
class Selection:
    """An auditable record of how one choice was actually made."""

    chosen: dict[str, Any] | None = None
    tied: list[dict[str, Any]] = field(default_factory=list)
    frontier: list[dict[str, Any]] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    band: float = 0.0
    how: str = ""
    tie_broken_by: str = ""
    real_tradeoff: bool = False

    def as_record(self, key: str = "k") -> dict[str, Any]:
        return {
            "chosen": self.chosen.get(key) if self.chosen else None,
            "tied": [r.get(key) for r in self.tied],
            "frontier": [r.get(key) for r in self.frontier],
            "n_rejected": len(self.rejected),
            "band": round(float(self.band), 6),
            "band_source": self.how,
            "tie_broken_by": self.tie_broken_by,
            "a_real_tradeoff_exists": self.real_tradeoff,
        }


def select(
    rows: Sequence[dict[str, Any]],
    *,
    locator: str,
    objectives: dict[str, bool],
    reject: Any = None,
    higher_is_better: bool = True,
    z: float = 2.0,
    fallback_band: float | None = None,
    prefer: Any = None,
) -> Selection:
    """Reject → frontier → tie-test → break the tie by a STATED preference.

    `real_tradeoff` is the field worth reading. It is True only when more than
    one candidate survives on the frontier AND their differences exceed the
    measured noise — i.e. when a genuine judgement is being made. When it is
    False the report should say the choice was arbitrary among equals rather than
    implying the winner was better.
    """
    sel = Selection()
    live = list(rows)
    if reject is not None:
        kept = [r for r in live if reject(r)]
        sel.rejected = [r for r in live if not reject(r)]
        live = kept or live          # never reject everything
    # NOISE IS ESTIMATED ON THE FULL SWEEP, NOT ON THE FRONTIER. The frontier is
    # a filtered distribution — measuring a mechanism's noise on the set it has
    # already selected is the failure this project has hit before. Concretely: on
    # live38 the full 14-point sweep gives se=0.0105 (matching the documented
    # seed sd) while the 4-point frontier gives 0.0220, and the inflated band
    # pulls k=15 into a tie it does not belong in.
    se = noise_floor([r[locator] for r in rows
                      if r.get(locator) is not None and not _isnan(r[locator])])
    sel.frontier = pareto_front(live, objectives) if objectives else live
    sel.tied, sel.band, sel.how = tie_set(
        sel.frontier or live, locator, higher_is_better=higher_is_better,
        z=z, se=None if _isnan(se) else se, fallback_band=fallback_band)
    pool = sel.tied or sel.frontier or live
    if prefer is not None and pool:
        sel.chosen = min(pool, key=prefer)
        sel.tie_broken_by = getattr(prefer, "__doc__", None) or "stated preference"
    elif pool:
        sel.chosen = (max if higher_is_better else min)(pool, key=lambda r: r[locator])
        sel.tie_broken_by = f"highest {locator}"
    sel.real_tradeoff = len(sel.frontier) > 1 and len(sel.tied) < len(sel.frontier)
    return sel


def _isnan(x: Any) -> bool:
    try:
        return bool(np.isnan(float(x)))
    except (TypeError, ValueError):
        return False
