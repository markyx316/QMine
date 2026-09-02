"""The candidate proposer, and the four things that keep widening a grid safe.

The grids are K12 artefacts — `alpha_grid = [0.0, 0.1, 0.2, 0.3, 0.5]` sits under
a comment saying "never inherit the K12 answer" while being the K12 answer. An
agent can propose a grid that fits the corpus in front of it. What must never
happen is a proposal winning because it was proposed rather than because it is
better.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from qmine.agents.propose_grid import corpus_profile, propose_grid
from qmine.ops.propose import (
    ALPHA_SPEC, NotBlind, assert_blind, grade_proposal, k_spec, validate_additions,
)
from qmine.ops.select import challenger_beats_incumbent, noise_floor

ALPHA_INCUMBENT = [0.0, 0.1, 0.2, 0.3, 0.5]


# ---------------------------------------------------------------- blindness

def test_the_proposer_is_never_shown_a_score():
    """Pre-registration is what makes an addition trustworthy.

    A proposer that can see where the current optimum sits can crowd the grid
    around it, and the extra candidates then win by proximity to a peak rather
    than on merit. Enforced on the rendered payload, since that is what the model
    reads — a nested key three levels down still arrives as text.
    """
    for leak in ("stability_ari=0.919", "silhouette 0.07", "the ami peak is at k=10",
                 "chosen_family_k = 10", "template_fragmentation", "macro f1 0.77",
                 "ranking of candidates", "best_alpha"):
        with pytest.raises(NotBlind):
            assert_blind(leak)


def test_the_blindness_check_does_not_cry_wolf_on_ordinary_corpus_words():
    """A guardrail that fires on real fields gets widened until it checks nothing.

    Substring matching flagged `domain_expected_family_range`, because "ami" is
    inside "f-ami-ly". Short metric names are matched as whole words for that
    reason.
    """
    for ok in ("family_range", "dynamic corpus", "n_rows=49999", "median_chars=9",
               "phrasing_group_coverage=0.36597", "language_shares han 0.9765"):
        assert_blind(ok)


def test_the_real_corpus_profile_is_blind(tmp_path):
    """The profile is built by inclusion, not by filtering artifacts.

    `battery`, `granularity` and `representation` are full of scores; a profile
    assembled by exclusion would leak the first time a field was added.
    """
    art = {
        "data_audit": {"n_rows": 49999, "n_unique": 49999, "median_chars": 9},
        "language_profile": {"shares": {"han": 0.9765, "latin": 0.0021}},
        "template_groups": {"coverage": {"n_groups": 12, "union_coverage": 0.36597},
                            "groups": [{"name": "pronunciation"}, {"name": "meaning"}]},
    }
    deps = SimpleNamespace(
        cfg=SimpleNamespace(domain=SimpleNamespace(key="k12_zh")),
        has=lambda k: k in art, load=lambda k: art[k])

    prof = corpus_profile(deps)
    assert_blind(json.dumps(prof, ensure_ascii=False))
    assert prof["n_rows"] == 49999 and prof["phrasing_group_coverage"] == 0.36597
    assert not any("expected" in k for k in prof), (
        "the hardcoded per-domain prior is the constant this exists to replace; "
        "showing it anchors the proposer to the answer"
    )


# ---------------------------------------------------------------- validation

def test_only_legal_novel_capped_additions_survive():
    out = validate_additions([0.05, 0.7, 0.1, 1.5, -0.2, "abc", 0.15, 0.9],
                             ALPHA_SPEC, ALPHA_INCUMBENT)
    assert out.kept == [0.05, 0.7, 0.15]
    why = {str(v): w for v, w in out.rejected}
    assert "already in the grid" in why["0.1"]
    assert "above the legal maximum" in why["1.5"]
    assert "below the legal minimum" in why["-0.2"]
    assert "not a float" in why["abc"]
    assert "cap of 3" in why["0.9"]
    assert out.widened == [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7]


def test_k_upper_bound_comes_from_the_corpus_not_a_constant():
    """Past n_rows/min_leaf_size the average cluster is too small to name — a
    limit the data sets, so it travels to any corpus."""
    assert k_spec(49999, 150).hi == 333
    assert k_spec(2000, 150).hi == 13
    out = validate_additions([400, 1, 11], k_spec(49999, 150), [8, 10])
    assert out.kept == [11]


def test_a_near_duplicate_is_not_a_new_candidate():
    """0.101 is 0.1. Admitting it spends a comparison to re-measure a point."""
    out = validate_additions([0.101, 0.3001], ALPHA_SPEC, ALPHA_INCUMBENT)
    assert out.kept == []
    assert all("already in the grid" in w for _, w in out.rejected)


# ---------------------------------------------------------------- the toll

def test_a_proposed_candidate_cannot_win_by_luck():
    """The load-bearing safety property.

    Widening a grid and taking the new maximum manufactures winners out of noise,
    worst in the near-tie regime this pipeline occupies. So a proposal may be
    swept freely only because merely scoring higher does not win it the seat.
    """
    sweep = [0.6868, 0.7534, 0.7507, 0.7270, 0.7065, 0.6537, 0.6274, 0.6299]
    se = noise_floor(sweep)
    incumbent = {"k": 10, "ami": 0.7534}

    lucky = {"k": 11, "ami": 0.7534 + 0.4 * se}
    assert not challenger_beats_incumbent(incumbent, lucky, "ami", se=se, z=2.0)[0]

    earned = {"k": 11, "ami": 0.7534 + 3.0 * se}
    assert challenger_beats_incumbent(incumbent, earned, "ami", se=se, z=2.0)[0]


# ---------------------------------------------------------------- fail-safe

def _deps(*, enabled=True, agent=None, art=None):
    events = []
    art = art or {"data_audit": {"n_rows": 5000}}
    return SimpleNamespace(
        cfg=SimpleNamespace(propose_grids=enabled, smoke_mode=False,
                            domain=SimpleNamespace(key="test")),
        emit=events.append, agent_ctx=lambda: None,
        has=lambda k: k in art, load=lambda k: art[k]), events


def test_a_dead_proposer_leaves_the_configured_grid_untouched(monkeypatch):
    """The failure mode must be "no additions", never "bad additions"."""
    import qmine.agents.roles as roles

    class Dead:
        def __init__(self, ctx, suffix=""): pass
        def run(self, **kw): raise RuntimeError("provider down")

    monkeypatch.setattr(roles, "ProposerAgent", Dead)
    deps, events = _deps()

    grid, rec = propose_grid(deps, "alpha", ALPHA_INCUMBENT, ALPHA_SPEC)

    assert grid == ALPHA_INCUMBENT
    assert rec["proposed_kept"] == [] and rec["skipped"] == "RuntimeError"
    assert any("unavailable" in e for e in events)


def test_the_drop_list_is_recorded_and_ignored(monkeypatch):
    """Removing a grid point can remove the true optimum; the compute saved is
    not worth a silent ceiling."""
    import qmine.agents.roles as roles

    class Dropper:
        def __init__(self, ctx, suffix=""): pass
        def run(self, **kw):
            return SimpleNamespace(add=[0.05], drop=[0.0, 0.5],
                                   rationale="heavy templating", corpus_signals=["coverage"])

    monkeypatch.setattr(roles, "ProposerAgent", Dropper)
    deps, _ = _deps()

    grid, rec = propose_grid(deps, "alpha", ALPHA_INCUMBENT, ALPHA_SPEC)

    assert 0.0 in grid and 0.5 in grid, "an advisory drop was acted on"
    assert rec["drop_advisory_ignored"] == [0.0, 0.5]


def test_disabled_by_default_costs_nothing(monkeypatch):
    deps, _ = _deps(enabled=False)
    grid, rec = propose_grid(deps, "alpha", ALPHA_INCUMBENT, ALPHA_SPEC)
    assert grid == ALPHA_INCUMBENT and rec["skipped"] == "disabled"


# ---------------------------------------------------------------- grading

def test_the_proposer_is_graded_every_run():
    """An agent nobody can evaluate is an agent nobody should keep."""
    rec = validate_additions([0.05, 0.15], ALPHA_SPEC, ALPHA_INCUMBENT).as_record()

    won = grade_proposal(rec, 0.05)
    assert won["a_proposed_value_won"] is True

    lost = grade_proposal(rec, 0.1)
    assert lost["a_proposed_value_won"] is False
    assert "returned nothing" in lost["verdict"]
    assert lost["n_extra_comparisons"] == 2, "the cost of the additions must be recorded"
