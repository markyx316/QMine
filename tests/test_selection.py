"""Choosing among candidates when nothing wins on every metric.

Three separate concerns, deliberately not blended: what can be decided by
measurement (domination), what is only noise (ties), and what widening the search
space costs (selection bias).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from qmine.ops.select import (
    challenger_beats_incumbent, noise_floor, pareto_front, select, tie_set,
)

#: live38's real 14-point K sweep. Embedded rather than invented so the
#: noise floor these tests exercise is the one the pipeline actually sees.
SWEEP = [
    {"k": 8,    "ami": 0.6868, "frag": 1.6192, "stab": 0.9191},
    {"k": 10,   "ami": 0.7534, "frag": 1.6220, "stab": 0.6950},
    {"k": 12,   "ami": 0.7507, "frag": 1.7170, "stab": 0.8284},
    {"k": 15,   "ami": 0.7270, "frag": 2.0633, "stab": 0.9086},
    {"k": 18,   "ami": 0.7156, "frag": 2.3436, "stab": 0.7639},
    {"k": 20,   "ami": 0.7065, "frag": 2.1342, "stab": 0.8158},
    {"k": 25,   "ami": 0.6537, "frag": 2.6316, "stab": 0.8145},
    {"k": 30,   "ami": 0.6274, "frag": 3.0440, "stab": 0.7725},
    {"k": 40,   "ami": 0.6299, "frag": 3.6856, "stab": 0.7410},
    {"k": 50,   "ami": 0.6226, "frag": 3.7485, "stab": 0.6591},
    {"k": 65,   "ami": 0.5806, "frag": 4.6665, "stab": 0.6491},
    {"k": 80,   "ami": 0.5748, "frag": 5.0130, "stab": 0.6306},
    {"k": 100,  "ami": 0.5572, "frag": 6.0970, "stab": 0.6368},
    {"k": 120,  "ami": 0.5358, "frag": 6.8887, "stab": 0.6439},
]
OBJ = {"ami": True, "frag": False, "stab": True}


def test_domination_removes_candidates_without_any_judgement():
    """A candidate worse on EVERY metric needs no preference to discard."""
    front = pareto_front(SWEEP, OBJ)
    ks = sorted(r["k"] for r in front)
    assert 20 not in ks and 30 not in ks, f"dominated candidates survived: {ks}"
    assert {8, 10, 12} <= set(ks), ks
    # k=8 wins stability and fragmentation; k=10 wins AMI. Neither dominates.
    assert 8 in ks and 10 in ks, "a genuine trade-off pair was collapsed"


def test_a_row_missing_a_metric_is_not_silently_discarded():
    """Dropping it would prefer whatever happened to be measured."""
    rows = SWEEP + [{"k": 99, "ami": 0.10, "frag": 9.0}]      # no `stab`
    assert any(r["k"] == 99 for r in pareto_front(rows, OBJ))


def test_the_noise_floor_recovers_a_dispersion_measured_another_way():
    """`intent_alignment_ami`'s seed sd is documented in `cluster.py` as ~0.01.

    `noise_floor` reads it off the curve's roughness with no extra compute, and
    on live38's real 14-point sweep returns 0.0105. Two independent routes to the
    same number is the only reason to trust a zero-cost estimator.
    """
    gen = Path("runs/live38/gen06/granularity.json")
    if not gen.exists():
        pytest.skip("live38 artifacts not present")
    sweep = json.loads(gen.read_text())["k_sweep"]
    se = noise_floor([r["intent_alignment_ami"] for r in sweep])
    assert 0.005 <= se <= 0.020, f"se={se:.4f} is nowhere near the documented ~0.01"


def test_noise_is_measured_on_the_full_sweep_not_the_frontier():
    """Never measure a mechanism's noise on the set it has already filtered.

    Estimating on the 4-point frontier instead of the 14-point sweep inflated
    live38's se from 0.0105 to 0.0220 and pulled k=15 into a tie it does not
    belong in — the selector's own answer changed because of where it looked.
    """
    sel = select(SWEEP, locator="ami", objectives=OBJ, z=2.0, prefer=lambda r: r["k"])
    full_se = noise_floor([r["ami"] for r in SWEEP])
    front_se = noise_floor([r["ami"] for r in pareto_front(SWEEP, OBJ)])
    assert not np.isnan(full_se)
    assert abs(sel.band - 2.0 * full_se) < 1e-9, (
        f"band {sel.band:.4f} was not built from the full-sweep se {full_se:.4f} "
        f"(frontier se would give {2 * front_se:.4f})"
    )


def test_a_tie_is_reported_as_a_tie_rather_than_a_ranking():
    """When the leader's margin is inside noise, saying it "won" is a false claim."""
    tied, band, how = tie_set(SWEEP, "ami", z=2.0)
    ks = sorted(r["k"] for r in tied)
    assert ks == [10, 12], ks
    assert "measured noise" in how, how
    assert band > 0


def test_the_band_says_when_it_had_to_fall_back():
    """A configured constant standing in for a measurement must be visible."""
    short = SWEEP[:2]                                  # too few points to estimate
    tied, band, how = tie_set(short, "ami", z=2.0, fallback_band=0.02)
    assert band == 0.02
    assert "configured" in how and "too few points" in how, how


def test_a_lucky_challenger_does_not_displace_the_incumbent():
    """Widening a search space and taking the new maximum manufactures winners.

    Selection bias is largest exactly here — near-ties, low signal-to-noise — so
    a proposer may widen the grid freely only because a candidate that merely
    scores higher does not win.
    """
    se = noise_floor([r["ami"] for r in SWEEP])
    inc = {"k": 10, "ami": 0.7534}
    lucky = {"k": 11, "ami": 0.7534 + 0.5 * se}
    real = {"k": 11, "ami": 0.7534 + 4.0 * se}

    ok, why = challenger_beats_incumbent(inc, lucky, "ami", se=se, z=2.0)
    assert not ok and "does not clear" in why, why

    ok, why = challenger_beats_incumbent(inc, real, "ami", se=se, z=2.0)
    assert ok, why


def test_an_unmeasurable_noise_floor_protects_the_incumbent():
    """Failing closed: if we cannot size the noise, we cannot justify a swap."""
    ok, why = challenger_beats_incumbent(
        {"k": 10, "ami": 0.5}, {"k": 11, "ami": 0.99}, "ami", se=float("nan"))
    assert not ok and "not estimable" in why


def test_the_record_says_whether_a_real_tradeoff_was_made():
    """The reader must be able to tell 'we judged' from 'they were tied'."""
    sel = select(SWEEP, locator="ami", objectives=OBJ, z=2.0, prefer=lambda r: r["k"])
    rec = sel.as_record()
    assert rec["chosen"] == 10
    assert rec["tied"] == [10, 12]
    assert 8 in rec["frontier"], "the stability/AMI trade-off partner is missing"
    assert rec["a_real_tradeoff_exists"] is True
    assert "measured noise" in rec["band_source"]
