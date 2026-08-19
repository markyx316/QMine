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


def test_k_is_located_by_intent_alignment_and_only_filtered_by_stability():
    """Stability rejects; alignment with the phrasing groups locates.

    Stability cannot rank K here — its seed-to-seed sd (~0.10) exceeds the gaps
    between adjacent K (~0.05), and its curve still climbs below the grid, so
    ranking by it reads noise and trends toward a degenerate two-way split.
    """
    sweep = [
        # k=10 is the most stable, but alignment says it merges distinct intents.
        {"k": 10, "stability_ari": 0.95, "silhouette": 0.10, "intent_alignment_ami": 0.50},
        {"k": 20, "stability_ari": 0.70, "silhouette": 0.05, "intent_alignment_ami": 0.72},
        {"k": 40, "stability_ari": 0.65, "silhouette": 0.30, "intent_alignment_ami": 0.60},
    ]
    tri = triangulate_k(sweep, {"k_estimate": 60}, (15, 25))
    assert tri["chosen_family_k"] == 20, "the most STABLE k won — stability is ranking again"
    assert tri["locator"] == "intent_alignment_ami"
    assert tri["silhouette_disagrees"] is True

    # Stability's only job is rejection: an irreproducible k is dropped even when
    # its alignment is the best on the board.
    unstable = [
        {"k": 10, "stability_ari": 0.20, "silhouette": 0.10, "intent_alignment_ami": 0.95},
        {"k": 20, "stability_ari": 0.80, "silhouette": 0.05, "intent_alignment_ami": 0.60},
    ]
    tri2 = triangulate_k(unstable, {"k_estimate": 60}, (15, 25))
    assert tri2["chosen_family_k"] == 20, "an irreproducible k survived the floor"
    assert tri2["n_rejected_as_unstable"] == 1


def test_triangulation_falls_back_to_stability_only_without_phrasing_groups():
    """No phrasing groups mined means nothing to align against. The old rule then
    applies, and the artifact must say the evidence is weak."""
    sweep = [
        {"k": 10, "stability_ari": 0.90, "silhouette": 0.10},
        {"k": 20, "stability_ari": 0.95, "silhouette": 0.05},
        {"k": 40, "stability_ari": 0.60, "silhouette": 0.30},
    ]
    tri = triangulate_k(sweep, {"k_estimate": 60}, (15, 25))
    assert tri["chosen_family_k"] == 20
    assert "stability_ari" in tri["locator"] and "weak" in tri["locator"]


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


# ==========================================================================
# Defects found by the first full live run
# ==========================================================================

def test_active_learning_accepts_the_sparse_matrix_its_caller_passes():
    """`len()` raises on scipy sparse, and the only caller passes a char-TFIDF
    matrix. The TypeError was swallowed upstream, so the playbook's round-2
    active learning had never once run."""
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    from qmine.ops.classify import select_active_learning_batch

    texts = [f"查询样本{i % 37}的内容" for i in range(200)]
    X = TfidfVectorizer(analyzer="char", ngram_range=(2, 3), min_df=2).fit_transform(texts)
    assert not isinstance(X, np.ndarray), "fixture must exercise the sparse path"
    y = np.array(["A", "B"] * 100)
    model = LogisticRegression(max_iter=200).fit(X[:100], y[:100])

    for fraction in (0.0, 0.3):   # the diversity pass must survive sparse too
        out = select_active_learning_batch(
            model, X, already_labelled=list(range(100)), batch=10,
            diversity_fraction=fraction,
        )
        assert out["selected"], f"nothing selected at diversity_fraction={fraction}"
        assert not set(out["selected"]) & set(range(100)), "re-selected a labelled row"

    dense = select_active_learning_batch(
        model, X.toarray(), already_labelled=list(range(100)), batch=10)
    sparse = select_active_learning_batch(
        model, X, already_labelled=list(range(100)), batch=10)
    assert dense["selected"] == sparse["selected"], "sparse and dense must agree"


def test_a_missing_annotation_is_not_scored_as_a_disagreement():
    """When an annotator's batch fails the caller fills UNLABELED. Counting that
    as disagreement charges an infrastructure failure to the methodology and
    depresses a blocking metric — it cost 0.008 kappa on the first live run."""
    from qmine.ops.classify import UNLABELED, agreement

    # Two labels, not one: with a single label kappa is undefined by design and
    # this module deliberately scores that 0.0 (see `agreement`'s docstring).
    b = (["X", "Y"] * 5)
    a = list(b)
    clean = agreement(a, b)
    assert clean["kappa"] == 1.0 and clean["n_unscored_unlabelled"] == 0

    a_missing = [UNLABELED, UNLABELED] + a[2:]
    out = agreement(a_missing, b)
    assert out["n"] == 8, "unlabelled rows must be excluded from the scored set"
    assert out["n_submitted"] == 10
    assert out["n_unscored_unlabelled"] == 2
    assert out["n_disagreements"] == 0, "an omission is missing data, not disagreement"
    assert out["kappa"] == 1.0

    allmissing = agreement([UNLABELED] * 5, ["X", "Y", "X", "Y", "X"])
    assert allmissing["n"] == 0, "must not claim agreement when nothing was labelled"


def test_boundaries_are_decided_from_agreed_rows_not_from_a_second_opinion():
    """The repair loop settles boundaries the referee resolved inconsistently.
    It does so from the rows both annotators agreed on — the only labels in the
    set with no arbitration in them — so the resulting rule is checkable rather
    than another model's opinion."""
    from qmine.ops.classify import (
        boundary_default, contested_boundaries, discriminating_markers,
    )

    def row(q, a, b, final):
        return type("R", (), {"query": q, "label_a": a, "label_b": b,
                              "final": final, "agreed": a == b})()

    rows = (
        # Agreed evidence: the marker 意思 always means MEANING…
        [row(f"词{i}的意思", "MEANING", "MEANING", "MEANING") for i in range(12)]
        # …and marker-less bare strings mostly mean IDIOM. These must share no
        # substring, or they would themselves become a marker and be excluded
        # from the default — which is the correct behaviour, just not what this
        # part of the test is about.
        + [row(q, "IDIOM", "IDIOM", "IDIOM") for q in
           ("飞檐走壁", "鳞次栉比", "南柯一梦", "似水流年", "喧宾夺主", "把持不住",
            "梭天摸地", "画龙点睛", "守株待兔", "刻舟求剑", "亡羊补牢", "опять算了",
            "杯弓蛇影", "掩耳盗铃", "叶公好龙", "买椟还珠", "滥竽充数", "囫囵吞枣",
            "邯郸学步", "东施效颦", "望梅止渴", "指鹿为马")]
        + [row(q, "MEANING", "MEANING", "MEANING") for q in
           ("寝", "翘楚", "鬻", "饕餮")]
        # The referee then sent this same pair both ways — an open boundary.
        + [row("甲的意思", "IDIOM", "MEANING", "MEANING"),
           row("乙", "IDIOM", "MEANING", "IDIOM")]
    )

    open_pairs = contested_boundaries(rows)
    assert [p["pair"] for p in open_pairs] == [["IDIOM", "MEANING"]]
    assert open_pairs[0]["resolved_as"] == {"MEANING": 1, "IDIOM": 1}

    markers = discriminating_markers(rows, ["IDIOM", "MEANING"], min_support=4)
    assert any(m["marker"] == "意思" and m["then"] == "MEANING" for m in markers), markers
    assert all(m["precision"] >= 0.90 for m in markers)

    default = boundary_default(rows, ["IDIOM", "MEANING"],
                               [m["marker"] for m in markers])
    assert default and default["then"] == "IDIOM", "marker-less rows lean IDIOM"

    # A genuine coin flip must stay open rather than be closed on noise.
    coin = ([row(f"x{i}", "A", "A", "A") for i in range(11)]
            + [row(f"y{i}", "B", "B", "B") for i in range(11)])
    assert boundary_default(coin, ["A", "B"]) is None


def test_stratified_sample_returns_positions_even_for_a_sliced_frame():
    """Two of the three branches returned positions and one returned index
    labels. They agree on a default RangeIndex — which every caller happened to
    pass — so a slice like `df.iloc[unseen]` silently produced out-of-range
    indices, and the guide-repair round crashed with IndexError on a live run."""
    import numpy as np
    import pandas as pd

    from qmine.ops.audit import stratified_sample

    df = pd.DataFrame({"q": [f"q{i}" for i in range(400)],
                       "stratum": [f"s{i % 7}" for i in range(400)]})
    sliced = df.iloc[200:]          # labels 200..399, positions 0..199
    assert sliced.index[0] != 0, "fixture must not have a RangeIndex starting at 0"

    for cols in ([], ["stratum"]):
        out = stratified_sample(sliced, 50, strata_cols=cols, seed=7)
        assert len(out) == 50
        assert out.min() >= 0 and out.max() < len(sliced), (
            f"strata_cols={cols}: returned an index outside the frame it was given"
        )
        # Positional, so .iloc must resolve every one of them.
        assert len(sliced.iloc[out]) == 50

    # Stratification must still be doing its job, not merely staying in range.
    strat = stratified_sample(sliced, 50, strata_cols=["stratum"], seed=7)
    assert sliced.iloc[strat]["stratum"].nunique() == 7, "a stratum was dropped"
