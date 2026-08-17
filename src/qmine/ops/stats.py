"""Statistical honesty for threshold gates.

A gate that compares a point estimate against a threshold treats
"0.978 measured on 1,000 rows" and "0.978 measured on 100,000 rows" as the same
evidence.  They are not.  The first is indistinguishable from 0.98; the second
is a real miss.

Left alone, this failure mode is self-correcting in the worst way: the gate
fires spuriously, someone lowers the threshold to make the run go green, and the
gate stops protecting anything.  So the gates in this pipeline compare a
*confidence interval* against the threshold and report one of three verdicts —
met, missed, or **underpowered**, which is a statement about the test set rather
than about the pipeline.
"""

from __future__ import annotations

import math
from typing import Any, Literal

Verdict = Literal["met", "missed", "underpowered"]


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Preferred over the normal approximation because these proportions live near
    1.0, where the normal interval overshoots past 1 and understates uncertainty
    exactly when we care most.
    """
    if n <= 0:
        return (0.0, 1.0)
    p = successes / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def proportion_gate(
    observed: float,
    n: int,
    threshold: float,
    *,
    z: float = 1.96,
) -> dict[str, Any]:
    """Judge ``observed ≥ threshold`` with the sample size taken seriously.

    * ``met`` — the lower bound clears the threshold, so the requirement is
      satisfied with confidence.
    * ``missed`` — the upper bound is below the threshold, so it genuinely is
      not satisfied. This is the only verdict that should stop a run.
    * ``underpowered`` — the interval straddles the threshold. The pipeline has
      not failed; the *test* is too small to tell, and the report says so along
      with how many rows would be needed to decide.
    """
    successes = int(round(observed * n))
    lo, hi = wilson_interval(successes, n, z)
    if lo >= threshold:
        verdict: Verdict = "met"
    elif hi < threshold:
        verdict = "missed"
    else:
        verdict = "underpowered"
    return {
        "observed": round(observed, 4),
        "n": int(n),
        "threshold": threshold,
        "ci95": [round(lo, 4), round(hi, 4)],
        "verdict": verdict,
        "passed": verdict in ("met", "underpowered"),
        "blocking_failure": verdict == "missed",
        "n_needed": required_n(observed, threshold, z) if verdict == "underpowered" else None,
        "note": {
            "met": "lower confidence bound clears the threshold",
            "missed": "upper confidence bound is below the threshold — a real miss",
            "underpowered": (
                "the confidence interval straddles the threshold: this test set is too "
                "small to decide. Not counted as a failure, and not counted as a pass."
            ),
        }[verdict],
    }


def required_n(observed: float, threshold: float, z: float = 1.96) -> int | None:
    """Rows needed for an interval around ``observed`` to clear ``threshold``.

    Returned so an underpowered gate produces an action ("run it on 4,300 rows")
    rather than a shrug.
    """
    gap = abs(observed - threshold)
    if gap <= 1e-9:
        return None
    var = max(observed * (1 - observed), 1e-6)
    return int(math.ceil(z * z * var / (gap * gap)))
