"""The statistics the cross-snapshot comparison rests on, and the seeding rule.

Every one of these was written for the hand-built `analysis/pooled5/` study and has
been read against real corpora many times; they are lifted here unchanged in
behaviour so that the integrated path and the delivered studies compute the same
numbers.  What changed is only that nothing here knows a corpus name any more.

**EVERY RESAMPLE DERIVES ITS OWN SEED.** The study originally drew from one
module-level generator, so adding a bootstrap anywhere shifted every interval
computed after it — point estimates identical, interval endpoints moved. Someone
had already quoted an endpoint in prose by then. `rng_for` derives the seed from
*what is being measured* (purpose, level, the pair of snapshots), so a given cell
gets the same interval no matter what else ran.
"""

from __future__ import annotations

import math
import zlib
from typing import Sequence

import numpy as np
import pandas as pd

#: Bootstrap replicates for a TVD interval, and the same-source null draws.
#: 400/300 are the study's values; they are parameters here so a smoke run can
#: shrink them, and `pooled.report` prints whichever was used.
B_BOOT = 400
B_NULL = 300

_BASE_SEED = 20260913


def rng_for(*tag: object) -> np.random.Generator:
    """A generator keyed to what is being measured, not to call order."""
    return np.random.default_rng(
        [_BASE_SEED, zlib.crc32("|".join(map(str, tag)).encode("utf-8"))])


def wilson(k: int, n: int) -> tuple[float, float]:
    """95% Wilson score interval for a proportion. n=0 returns (0, 0)."""
    if n == 0:
        return (0.0, 0.0)
    z, p = 1.959964, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def newcombe(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float, float]:
    """Difference in proportions (2 minus 1) with a Newcombe hybrid-score 95% interval."""
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    d = k2 / n2 - k1 / n1 if n1 and n2 else 0.0
    return (d, d - math.sqrt((k2 / n2 - l2) ** 2 + (u1 - k1 / n1) ** 2),
            d + math.sqrt((u2 - k2 / n2) ** 2 + (k1 / n1 - l1) ** 2))


def one_sided_upper(n: int) -> float:
    """The largest share still consistent with observing ZERO in n draws (97.5%).

    This is the number that makes an absence readable. 0 of 950 leaves room for
    0.39%; the same 0 of 10,000 leaves 0.037%. Reporting "absent" without it
    states a sample size as if it were a finding.
    """
    return 1 - 0.025 ** (1 / n) if n else 1.0


def cramers_v(counts: np.ndarray | Sequence[Sequence[float]]) -> float:
    counts = np.asarray(counts, dtype=float)
    n = counts.sum()
    if n == 0 or min(counts.shape) < 2:
        return 0.0
    exp = counts.sum(1, keepdims=True) @ counts.sum(0, keepdims=True) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum(np.where(exp > 0, (counts - exp) ** 2 / exp, 0.0))
    return float(math.sqrt((chi2 / n) / (min(counts.shape) - 1)))


def tvd(a: np.ndarray, b: np.ndarray, k: int) -> float:
    """Total variation distance between two label distributions over k classes."""
    if not len(a) or not len(b):
        return float("nan")
    pa = np.bincount(a, minlength=k) / len(a)
    pb = np.bincount(b, minlength=k) / len(b)
    return float(0.5 * np.abs(pa - pb).sum())


def tvd_with_bounds(a: np.ndarray, b: np.ndarray, k: int, *, tag: Sequence[object],
                    n_boot: int = B_BOOT, n_null: int = B_NULL) -> dict[str, float]:
    """Point TVD, its bootstrap interval, and the SAME-SOURCE noise ceiling.

    The ceiling is what makes the point estimate mean anything: a TVD of 0.08 is
    a finding only if two samples drawn from ONE distribution do not routinely
    give 0.08 by themselves.

    **THE NULL IS DRAWN AT THE OBSERVED SAMPLE SIZES.** It used to split each
    side in half and compare n/2 against n/2, which is a different question:
    TVD noise scales as 1/sqrt(n), so halving n inflates the ceiling. Measured
    against the correct null — pool both sides, re-split at n_a and n_b — the
    half-split ceiling is **1.41x too high at 1,000 vs 1,000, 1.44x at 10,000 vs
    10,316, and 1.80x at 400 vs 4,000**, and the distortion grows exactly where
    the snapshots are most unbalanced. The error is in the conservative
    direction (it hides real differences, never invents one), which is why it
    survived; `conditional_mix` in this same package already pooled correctly,
    so the two were answering different questions under one column name.
    """
    point = tvd(a, b, k)
    rng = rng_for("tvd", *tag)
    boot = np.array([tvd(rng.choice(a, len(a)), rng.choice(b, len(b)), k)
                     for _ in range(n_boot)])
    pool = np.concatenate([a, b])
    na = len(a)
    null: list[float] = []
    for _ in range(max(1, n_null)):
        idx = rng.permutation(len(pool))
        null.append(tvd(pool[idx[:na]], pool[idx[na:]], k))
    lo = float(np.percentile(boot, 2.5))
    ceiling = float(np.percentile(null, 95)) if null else float("nan")
    return {"tvd": point, "lo": lo, "hi": float(np.percentile(boot, 97.5)),
            "noise_ceiling": ceiling,
            "exceeds_noise": bool(null and lo > ceiling)}


def codes(s: pd.Series, order: Sequence[object]) -> np.ndarray:
    m = {k: i for i, k in enumerate(order)}
    return s.map(m).to_numpy()


def absence_verdict(expected: float, p_zero: float) -> str:
    """Whether a zero count is evidence of absence, weak evidence, or nothing.

    The thresholds are fixed HERE rather than in the prose, because a reader who
    meets a raw 0 will otherwise interpret it themselves, and the interpretation
    that comes naturally ("this class does not occur here") is the wrong one at
    every sample size this study works with.
    """
    if expected >= 5 and p_zero < 0.01:
        return "真缺席"
    if expected < 3:
        return "不可判定（样本量不足）"
    return "偏少但证据弱"
