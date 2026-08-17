"""Tests for the quantitative core: does the science actually work?"""

from __future__ import annotations

import numpy as np
import pytest

from qmine.ops.cluster import (
    build_hierarchy,
    deep_aligned_estimate,
    heldout_reproduction,
    kmeans_labels,
    margins,
    refine,
    replay_stability,
    triangulate_k,
)
from qmine.ops.represent import HashingEncoder, build_sparse, hybrid, surface_vote_share
from qmine.ops.templates import build_groups, mine_affixes, template_fragmentation


# -- the alpha algebra ------------------------------------------------------

def test_surface_vote_share_is_alpha_squared_not_alpha():
    """The single most load-bearing arithmetic fact in the playbook.

    alpha=0.5 *reads* like 'half the weight' and is actually 20%. Getting this
    wrong is how a representation ends up phrasing-dominated while its author
    believes it is balanced.
    """
    assert surface_vote_share(0.0) == 0.0
    assert surface_vote_share(0.1) == pytest.approx(0.0099, abs=1e-4)
    assert surface_vote_share(0.5) == pytest.approx(0.20, abs=1e-3)
    assert surface_vote_share(1.0) == pytest.approx(0.5)


def test_hybrid_cosine_matches_the_derived_formula():
    """cos(H,H') = (cos_sem + a^2 cos_surf) / (1 + a^2), verified numerically."""
    from sklearn.preprocessing import normalize

    rng = np.random.RandomState(0)
    e = normalize(rng.randn(2, 16)).astype(np.float32)
    s = normalize(rng.randn(2, 8)).astype(np.float32)
    for a in (0.1, 0.3, 0.5, 1.0):
        H = hybrid(e, s, a)
        got = float(H[0] @ H[1])
        expected = (float(e[0] @ e[1]) + a**2 * float(s[0] @ s[1])) / (1 + a**2)
        assert got == pytest.approx(expected, abs=1e-5), f"alpha={a}"


def test_hybrid_output_is_unit_norm():
    from sklearn.preprocessing import normalize

    rng = np.random.RandomState(1)
    H = hybrid(normalize(rng.randn(20, 16)), normalize(rng.randn(20, 8)), 0.3)
    assert np.allclose(np.linalg.norm(H, axis=1), 1.0, atol=1e-5)


# -- template fragmentation -------------------------------------------------

def test_fragmentation_is_one_when_a_family_stays_together():
    labels = np.zeros(100, dtype=int)
    mask = np.ones(100, dtype=bool)
    assert template_fragmentation(labels, {"g": mask})["mean_fragmentation"] == pytest.approx(1.0)


def test_fragmentation_counts_an_even_three_way_split_as_three():
    labels = np.array([0] * 30 + [1] * 30 + [2] * 30)
    mask = np.ones(90, dtype=bool)
    assert template_fragmentation(labels, {"g": mask})["mean_fragmentation"] == pytest.approx(3.0, abs=0.01)


def test_fragmentation_barely_moves_for_a_few_stragglers():
    """Perplexity rather than a raw cluster count: 97/3 should stay near 1."""
    labels = np.array([0] * 97 + [1] * 3)
    mask = np.ones(100, dtype=bool)
    v = template_fragmentation(labels, {"g": mask})["mean_fragmentation"]
    assert 1.0 < v < 1.3, v


# -- clustering primitives --------------------------------------------------

def test_replay_stability_is_high_on_genuinely_separated_blobs(toy_embedding):
    assert replay_stability(toy_embedding, 3, seeds=(0, 1)) > 0.9


def test_replay_stability_collapses_on_structureless_data():
    from sklearn.preprocessing import normalize

    X = normalize(np.random.RandomState(0).randn(600, 32)).astype(np.float32)
    assert replay_stability(X, 12, seeds=(0, 1)) < 0.5


def test_deep_aligned_recovers_the_right_order_of_magnitude(toy_embedding):
    est = deep_aligned_estimate(toy_embedding, 3, multiplier=3)
    assert 1 <= est["k_estimate"] <= 9


def test_heldout_reproduction_is_near_perfect_on_clean_structure(toy_embedding):
    labels = kmeans_labels(toy_embedding, 3, seed=0)
    assert heldout_reproduction(toy_embedding, labels)["agreement"] > 0.95


def test_triangulation_takes_the_stability_peak_not_the_silhouette_peak():
    sweep = [
        {"k": 10, "stability_ari": 0.90, "silhouette": 0.10},
        {"k": 20, "stability_ari": 0.95, "silhouette": 0.05},
        {"k": 40, "stability_ari": 0.60, "silhouette": 0.30},
    ]
    tri = triangulate_k(sweep, {"k_estimate": 60}, (15, 25))
    assert tri["chosen_family_k"] == 20
    assert tri["silhouette_disagrees"] is True


def test_hierarchy_is_two_levels_and_respects_the_minimum_leaf_size(toy_embedding):
    tree = build_hierarchy(toy_embedding, 3, min_leaf_size=50, min_leaf_fraction=0.0,
                           family_min_size_for_split=100)
    sizes = np.bincount(tree["leaf_labels"])
    assert tree["n_leaves"] >= tree["n_families"]
    assert sizes.min() >= 40, sizes
    assert len(tree["leaf_family"]) == tree["n_leaves"]


def test_refinement_converges_and_keeps_the_family_map_aligned(toy_embedding):
    tree = build_hierarchy(toy_embedding, 4, min_leaf_size=40, min_leaf_fraction=0.0)
    out = refine(toy_embedding, tree["leaf_labels"], tree["leaf_family"], rounds=4, min_leaf_size=40)
    assert out["history"]
    assert len(out["leaf_family"]) == out["n_leaves"]
    assert out["leaf_labels"].max() + 1 == out["n_leaves"]


def test_margins_are_non_negative(toy_embedding):
    from sklearn.preprocessing import normalize

    labels = kmeans_labels(toy_embedding, 3, seed=0)
    cents = normalize(np.vstack([toy_embedding[labels == c].mean(0) for c in range(3)]))
    assert (margins(toy_embedding, cents) >= -1e-6).all()


# -- representation ---------------------------------------------------------

def test_hashing_encoder_is_deterministic_and_normalised():
    enc = HashingEncoder(dim=64)
    a = enc.encode(["中文查询", "another query"])
    b = enc.encode(["中文查询", "another query"])
    assert np.allclose(a, b)
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-5)


def test_sparse_block_is_normalised_and_compressed(k12_queries):
    out = build_sparse(k12_queries[:500], svd_dims=32, min_df=1)
    assert out["svd_block"].shape == (500, 32)
    assert np.allclose(np.linalg.norm(out["svd_block"], axis=1), 1.0, atol=1e-4)


# -- template mining --------------------------------------------------------

def test_affix_miner_finds_real_markers(frame):
    aff = mine_affixes(frame["query"].tolist(), min_count=5)
    suffixes = {a["affix"] for a in aff["suffixes"]}
    assert any("拼音" in s or "意思" in s or "怎么读" in s for s in suffixes), sorted(suffixes)[:20]


def test_longer_markers_outrank_shorter_fragments_of_themselves(frame):
    aff = mine_affixes(frame["query"].tolist(), min_count=5)
    order = [a["affix"] for a in aff["suffixes"]]
    if "什么意思" in order and "么意思" in order:
        assert order.index("什么意思") < order.index("么意思")


def test_groups_reject_patterns_that_match_almost_nothing(frame):
    groups = build_groups(frame, seeds=[{"name": "impossible", "pattern": "ZZZZQQQQ", "intent_hint": ""}])
    assert not any(g.name == "impossible" for g in groups)
