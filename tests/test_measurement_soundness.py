

def test_the_falsification_probe_compares_against_a_structurally_different_algorithm():
    """A "best alternative" that IS the reference makes the gap 0.0 by
    construction, and the probe's conclusion then rests on no comparison at all.

    `best_other` was `ranked[0]` — the best OVERALL, which is the reference
    whenever KMeans wins, which is most of the time. live38 and live40 both
    shipped `reference = best_alternative = kmeans_k15, gap = 0.0` under the
    note "no structurally different algorithm is materially more reproducible".

    The probe asks whether a different cluster-shape assumption finds the same
    structure, so the candidate has to be non-KMeans. Replaying live40's real
    ranking: `agglo_average_k15` at 0.7529 against 0.8305 gives -0.0776 — the
    same reassuring answer, but measured.
    """
    from qmine.ops.cluster import _battery_verdict

    rows = [
        {"algorithm": "kmeans_k15", "stability_ari": 0.8305, "k": 15, "silhouette": 0.03},
        {"algorithm": "agglo_average_k15", "stability_ari": 0.7529, "k": 15, "silhouette": 0.02},
        {"algorithm": "agglo_ward_k15", "stability_ari": 0.7000, "k": 15, "silhouette": 0.01},
    ]
    v = _battery_verdict(rows)

    assert v["reference_algorithm"] == "kmeans_k15"
    assert v["best_alternative"] == "agglo_average_k15", (
        "the alternative must be structurally different from the reference")
    assert v["best_alternative"] != v["reference_algorithm"]
    assert v["alternative_beats_reference_by"] == -0.0776
    assert v["kmeans_assumption_contradicted"] is False


def test_the_probe_can_still_fire_when_an_alternative_really_is_better():
    """The gate must be able to fail, or it is decoration."""
    from qmine.ops.cluster import _battery_verdict

    rows = [
        {"algorithm": "agglo_average_k15", "stability_ari": 0.95, "k": 15, "silhouette": 0.05},
        {"algorithm": "kmeans_k15", "stability_ari": 0.60, "k": 15, "silhouette": 0.01},
    ]
    v = _battery_verdict(rows)
    assert v["best_alternative"] == "agglo_average_k15"
    assert v["alternative_beats_reference_by"] == 0.35
    assert v["kmeans_assumption_contradicted"] is True


def test_a_battery_with_no_structural_alternative_reports_none_not_itself():
    """Better to say "nothing to compare against" than to compare with itself."""
    from qmine.ops.cluster import _battery_verdict

    v = _battery_verdict([{"algorithm": "kmeans_k15", "stability_ari": 0.83, "k": 15, "silhouette": 0.03},
                          {"algorithm": "kmeans_k20", "stability_ari": 0.80, "k": 20, "silhouette": 0.02}])
    assert v["best_alternative"] is None
    assert v["alternative_beats_reference_by"] is None
    assert v["kmeans_assumption_contradicted"] is False


def test_the_falsification_probe_cannot_compare_kmeans_with_a_kmeans_variant():
    """"Structurally different" is about the cluster-shape assumption, not the name.

    The probe asks whether the delivered structure is an artefact of KMeans's
    isotropic-cluster assumption. Answering it with MiniBatchKMeans (KMeans with
    stochastic updates) or BisectingKMeans (recursive 2-means) is the same
    tautology the first fix was written against, one name away: the original bug
    compared the reference with itself, and `not startswith("kmeans")` still let
    both variants through. live40 escaped only because `agglo_average` happened to
    outrank `minibatch`.
    """
    from qmine.ops.cluster import _best_structural_alternative, _is_kmeans_family

    for name in ("kmeans_k15", "minibatch_k20", "bisecting_k30"):
        assert _is_kmeans_family(name), f"{name} shares KMeans's assumption"
    for name in ("agglo_average_k15", "gmm_diag_k20", "hdbscan_mcs50"):
        assert not _is_kmeans_family(name), f"{name} does not"

    ranked = [
        {"algorithm": "kmeans_k15", "stability_ari": 0.83, "n_clusters": 15},
        {"algorithm": "minibatch_k15", "stability_ari": 0.81, "n_clusters": 15},
        {"algorithm": "bisecting_k15", "stability_ari": 0.79, "n_clusters": 15},
        {"algorithm": "agglo_average_k15", "stability_ari": 0.75, "n_clusters": 15},
    ]
    alt = _best_structural_alternative(ranked, ranked[0])
    assert alt["algorithm"] == "agglo_average_k15", (
        f"picked {alt['algorithm']} — a KMeans variant cannot falsify KMeans")


def test_the_probe_margin_is_paired_within_k():
    """An unpaired comparison of two maxima confounds shape with granularity.

    On live40 the granularity effect is the larger one — KMeans replay ARI runs
    0.8305 at k=15 and 0.6649 at k=20 — so an alternative at one k "beating" a
    reference at another measures k, not the assumption. Paired, live40's own
    numbers even change sign: -0.0776 at k=15 but **+0.0603 at k=20**, where
    gmm_diag is genuinely ahead. The unpaired form reported a single -0.0776 and
    that sign flip reached no reader.
    """
    from qmine.ops.cluster import _paired_margins

    ranked = [
        {"algorithm": "kmeans_k15", "stability_ari": 0.8305, "n_clusters": 15},
        {"algorithm": "agglo_average_k15", "stability_ari": 0.7529, "n_clusters": 15},
        {"algorithm": "kmeans_k20", "stability_ari": 0.6649, "n_clusters": 20},
        {"algorithm": "gmm_diag_k20", "stability_ari": 0.7252, "n_clusters": 20},
    ]
    per_k = _paired_margins(ranked)
    assert per_k["15"]["margin"] == -0.0776
    assert per_k["20"]["margin"] == 0.0603, "the within-k sign flip was lost"
    # Both sides of every pair come from the same k.
    for k, m in per_k.items():
        assert m["reference"].endswith(f"k{k}") and m["alternative"].endswith(f"k{k}")


def test_the_k_locator_never_scores_against_the_top_down_labels():
    """Locating K against the taxonomy would turn the headline result into a fit.

    The project's central claim is that two INDEPENDENT routes found the same
    structure — measured on live40 at the leaf layer as AMI 0.6175, 19/25 leaves
    majority-one-intent. That is evidence only because the bottom-up tree was
    built without seeing the taxonomy. Locate K by maximising agreement with
    `td_l1` and the same number stops being a measurement and becomes the
    objective: on live40's full corpus it moves 0.5748 -> 0.6308, and the +0.056
    is the fit, not agreement.

    Two further measured consequences, so this is not a purity argument:
      * the top-down reference locates K=18, not 7, and the delivered LEAF layer
        is 25 — the two-layer design collapses when the layers coincide;
      * `BlindnessFirewall.add_taxonomy` already forbids it, so a run doing this
        would contradict its own firewall.

    `ref_legacy_l1` is the right anchor for the same question: 9 classes, complete,
    external to BOTH routes, available at p1, and it locates K=18 too — so the
    signal is obtainable without the circularity or the serialisation cost.
    """
    import inspect

    from qmine.graph.nodes import bottomup

    src = inspect.getsource(bottomup.p5_granularity)
    for forbidden in ("td_l1", "td_l2", "taxonomy()", "topdown_metrics"):
        assert forbidden not in src, (
            f"p5 references `{forbidden}` — locating K against the top-down route "
            "makes the route-concordance result circular")

    # The references it DOES use must come from declared input columns, which p3
    # already reads for `nmi_reference`, so no new visibility is introduced.
    assert "reference_label_columns" in src


def test_k_is_reported_under_every_available_reference():
    """The located K tracks the reference, so one number alone is misleading.

    live40, full corpus: trusted phrasing groups (6 classes, our own seed regexes)
    locate K=7; `ref_legacy_l1` (9 classes, external to both routes) locates K=18;
    `td_l1` (25 classes) locates K=18. The reference that DECIDES is the outlier,
    and no artifact carried that fact.
    """
    from qmine.ops.cluster import reference_sensitivity

    sweep = [
        {"k": 7, "intent_alignment_ami": 0.7500, "ami_vs_ref_legacy_l1": 0.2978},
        {"k": 12, "intent_alignment_ami": 0.7432, "ami_vs_ref_legacy_l1": 0.3036},
        {"k": 18, "intent_alignment_ami": 0.7071, "ami_vs_ref_legacy_l1": 0.3300},
    ]
    out = reference_sensitivity(sweep, chosen_k=7)
    assert out["located_k_values"] == {"phrasing_groups": 7, "ref_legacy_l1": 18}
    assert out["references_agree"] is False
    assert out["by_reference"]["phrasing_groups"]["decides"] is True
    assert out["by_reference"]["ref_legacy_l1"]["decides"] is False
    assert "note" in out and "18" in out["note"]

    # And a run with a single anchor must say so rather than imply triangulation.
    solo = reference_sensitivity(
        [{"k": 7, "intent_alignment_ami": 0.75}, {"k": 12, "intent_alignment_ami": 0.74}], 7)
    assert solo["references_agree"] is True
    assert "note" not in solo


def test_reach_not_row_coverage_is_what_qualifies_a_reference_to_locate_k():
    """A reference can cover many rows and still see almost none of the partition.

    K is located by maximising agreement with a reference, so a reference that
    occupies only part of the partition cannot express a preference about the
    rest — however good its rows are. That is the entire explanation for live40's
    K disagreement, established by control rather than argument: scoring the SAME
    reference on the rows the templates match gives K=7, on a random sample of
    identical size gives K=18, and on the rows they miss gives K=18. Row coverage
    was identical in all three; what differed was which clusters they reached.

    So the qualifying statistic is reach, and row coverage alone can be actively
    misleading — as constructed here, where a reference holding a third of the
    rows sees only one cluster of four.
    """
    import numpy as np

    from qmine.ops.cluster import locator_reach

    labels = np.repeat([0, 1, 2, 3], 250)          # four equal clusters

    # Concentrated: a third of the rows, all inside cluster 0.
    concentrated = np.full(1000, -1, dtype=np.int64)
    concentrated[:250] = 0
    concentrated[250:333] = 1                       # a few strays elsewhere
    c = locator_reach(labels, concentrated)

    # Spread: the SAME number of rows, taken evenly from every cluster.
    spread = np.full(1000, -1, dtype=np.int64)
    for i in range(4):
        spread[i * 250: i * 250 + 83] = i
    sp = locator_reach(labels, spread)

    assert abs(c["row_coverage"] - sp["row_coverage"]) < 0.02, (
        f"the control requires equal row coverage: {c['row_coverage']} vs {sp['row_coverage']}")
    assert sp["reach"] > c["reach"], (
        f"reach failed to separate them — concentrated {c['reach']}, spread {sp['reach']}")
    assert sp["reach"] == 1.0, sp
    assert c["reach"] <= 0.5, (
        f"a reference living in one cluster of four must not read as corpus-wide: {c}")


def test_the_reference_that_reaches_more_of_the_partition_locates_k():
    """`k_locator: auto` is a rule, not a preference for a particular column.

    live40: trusted phrasing groups reach 38.9% of clusters at k=18 and locate
    K=7; `ref_legacy_l1` reaches 100% and locates K=18. Choosing by reach picks
    the second without naming it, so the rule carries to a corpus whose references
    are different ones.
    """
    from qmine.ops.cluster import triangulate_k

    sweep = [
        {"k": 7, "stability_ari": 0.99, "silhouette": 0.07,
         "intent_alignment_ami": 0.7500, "ami_vs_ref_legacy_l1": 0.2978},
        {"k": 12, "stability_ari": 0.83, "silhouette": 0.065,
         "intent_alignment_ami": 0.7432, "ami_vs_ref_legacy_l1": 0.3036},
        {"k": 18, "stability_ari": 0.76, "silhouette": 0.059,
         "intent_alignment_ami": 0.7071, "ami_vs_ref_legacy_l1": 0.3300},
        {"k": 25, "stability_ari": 0.81, "silhouette": 0.056,
         "intent_alignment_ami": 0.6714, "ami_vs_ref_legacy_l1": 0.3284},
    ]
    da = {"k_estimate": 28}
    by_phrasing = triangulate_k(sweep, da, (15, 25), locator_key="intent_alignment_ami")
    by_legacy = triangulate_k(sweep, da, (15, 25), locator_key="ami_vs_ref_legacy_l1")

    assert by_phrasing["chosen_family_k"] == 7, by_phrasing["chosen_family_k"]
    assert by_legacy["chosen_family_k"] > 7, (
        f"the higher-reach reference produced the same K as the low-reach one: "
        f"{by_legacy['chosen_family_k']}")
    assert by_legacy["locator"] == "ami_vs_ref_legacy_l1"
    # And the run must record which reference actually decided.
    assert by_phrasing["locator"] != by_legacy["locator"]


def test_the_locator_is_chosen_by_reach_not_by_row_coverage():
    """The selection RULE, not just its inputs.

    Swapping `reach` for `row_coverage` here changes the delivered K and was
    invisible to every test while this logic lived inline in p5. The case that
    separates them is the one live40 actually presents: a reference covering every
    row but reaching few clusters must not outrank one that reaches all of them.
    """
    from qmine.ops.cluster import choose_locator

    reach = {
        "phrasing_groups": {"reach": 0.389, "row_coverage": 0.334,
                            "column": "intent_alignment_ami"},
        "ref_legacy_l1": {"reach": 1.0, "row_coverage": 1.0,
                          "column": "ami_vs_ref_legacy_l1"},
    }
    assert choose_locator(reach) == ("ami_vs_ref_legacy_l1", "ref_legacy_l1")

    # High row coverage, poor reach — must NOT win.
    tricky = {
        "phrasing_groups": {"reach": 0.9, "row_coverage": 0.30,
                            "column": "intent_alignment_ami"},
        "wide_but_blind": {"reach": 0.2, "row_coverage": 1.0,
                           "column": "ami_vs_wide_but_blind"},
    }
    assert choose_locator(tricky)[1] == "phrasing_groups", (
        "a reference that sees 20% of the partition outranked one that sees 90%")

    # The escape hatches still work, and an unknown name falls back safely.
    assert choose_locator(reach, "phrasing")[1] == "phrasing_groups"
    assert choose_locator(reach, "ref_legacy_l1")[1] == "ref_legacy_l1"
    assert choose_locator(reach, "nonexistent")[1] == "phrasing_groups"
    assert choose_locator({})[0] == "intent_alignment_ami"
