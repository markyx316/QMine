

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
    # `True` here IMPLIED triangulation, which is what the line above says must
    # not happen — the assertion contradicted its own stated intent. med01 shipped
    # exactly this: one reference, `references_agree: true`, a delivered summary
    # reading "full", and that lone reference had located K=18 against a chosen 12.
    assert solo["references_agree"] is None, \
        "agreement over a singleton is vacuous; None means 'not established'"
    assert solo["n_references"] == 1, "and the basis must ship with the verdict"
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


def test_a_reference_that_cannot_tell_k_apart_does_not_get_to_locate_k():
    """Reach is necessary and NOT sufficient — found by running the reach-only rule.

    A locator must do two separate things: SEE the whole partition, and TELL
    DIFFERENT K APART. The first rule shipped with only the first requirement, and
    live41 measured what that costs on its own sweep:

    | reference | reach | range | range/se | tie set |
    |---|---|---|---|---|
    | phrasing groups | 40% | 0.2190 | 17.5 | 4 |
    | `legacy_l1` | 100% | 0.0373 | **5.9** | **8** |
    | `legacy_l2` | 100% | 0.1072 | 17.5 | 3 |

    Reach alone picked `legacy_l1` — nine coarse classes, two holding ~79% of the
    corpus, so its entire curve spans 0.037 against a 0.0126 tie band. Eight of
    seventeen k values tied and `min(tie_set)` returned **12** while the raw argmax
    was **30**. The honest reading of that output is "this reference cannot
    distinguish 12 from 65", and it was reported as a located K.

    `legacy_l2` dominates it outright: identical reach, 3x the signal-to-noise,
    a 3-way tie, and it locates 18 — the value every full-coverage reference gave
    on live40.
    """
    from qmine.ops.cluster import choose_locator, discrimination

    # live41's ACTUAL 17-point sweep. Not abridged: `noise_floor` reads the
    # roughness of successive differences, so a thinned curve has different
    # spacing and its ratios are not the ones the run computed.
    sweep = [
        {'k': 5, 'intent_alignment_ami': 0.575, 'ami_vs_legacy_l1': 0.2867, 'ami_vs_legacy_l2': 0.4287},
        {'k': 6, 'intent_alignment_ami': 0.7548, 'ami_vs_legacy_l1': 0.2917, 'ami_vs_legacy_l2': 0.4857},
        {'k': 7, 'intent_alignment_ami': 0.7495, 'ami_vs_legacy_l1': 0.2974, 'ami_vs_legacy_l2': 0.4843},
        {'k': 8, 'intent_alignment_ami': 0.6868, 'ami_vs_legacy_l1': 0.2897, 'ami_vs_legacy_l2': 0.4646},
        {'k': 10, 'intent_alignment_ami': 0.7534, 'ami_vs_legacy_l1': 0.3009, 'ami_vs_legacy_l2': 0.499},
        {'k': 12, 'intent_alignment_ami': 0.7507, 'ami_vs_legacy_l1': 0.3142, 'ami_vs_legacy_l2': 0.5167},
        {'k': 15, 'intent_alignment_ami': 0.727, 'ami_vs_legacy_l1': 0.3035, 'ami_vs_legacy_l2': 0.5128},
        {'k': 18, 'intent_alignment_ami': 0.7156, 'ami_vs_legacy_l1': 0.3216, 'ami_vs_legacy_l2': 0.5359},
        {'k': 20, 'intent_alignment_ami': 0.7065, 'ami_vs_legacy_l1': 0.3227, 'ami_vs_legacy_l2': 0.5167},
        {'k': 25, 'intent_alignment_ami': 0.6539, 'ami_vs_legacy_l1': 0.3229, 'ami_vs_legacy_l2': 0.5267},
        {'k': 30, 'intent_alignment_ami': 0.6274, 'ami_vs_legacy_l1': 0.324, 'ami_vs_legacy_l2': 0.5227},
        {'k': 40, 'intent_alignment_ami': 0.6299, 'ami_vs_legacy_l1': 0.3195, 'ami_vs_legacy_l2': 0.5254},
        {'k': 50, 'intent_alignment_ami': 0.6226, 'ami_vs_legacy_l1': 0.3146, 'ami_vs_legacy_l2': 0.5198},
        {'k': 65, 'intent_alignment_ami': 0.5806, 'ami_vs_legacy_l1': 0.3123, 'ami_vs_legacy_l2': 0.513},
        {'k': 80, 'intent_alignment_ami': 0.5748, 'ami_vs_legacy_l1': 0.3043, 'ami_vs_legacy_l2': 0.504},
        {'k': 100, 'intent_alignment_ami': 0.5572, 'ami_vs_legacy_l1': 0.3005, 'ami_vs_legacy_l2': 0.4988},
        {'k': 120, 'intent_alignment_ami': 0.5358, 'ami_vs_legacy_l1': 0.2935, 'ami_vs_legacy_l2': 0.4907},
    ]
    reach = {
        "phrasing_groups": {"reach": 0.40, "row_coverage": 0.33, "column": "intent_alignment_ami"},
        "legacy_l1": {"reach": 1.0, "row_coverage": 1.0, "column": "ami_vs_legacy_l1"},
        "legacy_l2": {"reach": 1.0, "row_coverage": 1.0, "column": "ami_vs_legacy_l2"},
    }
    d1 = discrimination(sweep, "ami_vs_legacy_l1")
    d2 = discrimination(sweep, "ami_vs_legacy_l2")
    assert round(d1, 1) == 5.9 and round(d2, 1) == 17.5, (
        f"the run measured 5.93 and 17.48, got {d1} and {d2}")

    col, name = choose_locator(reach, sweep=sweep)
    assert name == "legacy_l2", (
        f"picked {name}: a reference that sees everything and distinguishes "
        "nothing has not located anything")
    assert col == "ami_vs_legacy_l2"

    # Reach still gates: the sharpest curve of all belongs to a reference that
    # sees 40% of the partition, and it must not win on sharpness alone.
    assert discrimination(sweep, "intent_alignment_ami") >= d2
    assert choose_locator(reach, sweep=sweep)[1] != "phrasing_groups"

    # With no sweep it degrades to the reach rule rather than breaking.
    assert choose_locator(reach)[1] in {"legacy_l1", "legacy_l2"}


def test_the_stated_tie_band_arithmetic_actually_checks_out():
    """A reader who multiplies the numbers must get the number printed beside them.

    `chosen_by` rendered `se` at 4dp while the band came from the unrounded value,
    so live41 shipped "ties within 0.0125 — 2x measured noise (se=0.0062)". Two
    times 0.0062 is 0.0124. Its own p5 observer did the multiplication and found
    it did not hold.

    The band was correct and the tie set was unaffected; what failed was the
    sentence a reader uses to check it. That is worth fixing precisely because it
    is small: a stated conversion that does not hold makes every other number on
    the page suspect, and this pipeline asks readers to check its numbers.
    """
    import re

    from qmine.ops.cluster import triangulate_k

    # live41's ACTUAL legacy_l2 curve — the one that produced se=0.00625 and
    # exposed the rounding. A synthetic curve with even spacing gives a MAD of
    # zero and se~1e-16, which tests nothing.
    sweep = [
        {'k': 5, 'stability_ari': 0.9944, 'silhouette': 0.0769, 'intent_alignment_ami': 0.4287},
        {'k': 6, 'stability_ari': 0.9959, 'silhouette': 0.0661, 'intent_alignment_ami': 0.4857},
        {'k': 7, 'stability_ari': 0.9966, 'silhouette': 0.0714, 'intent_alignment_ami': 0.4843},
        {'k': 8, 'stability_ari': 0.9191, 'silhouette': 0.0652, 'intent_alignment_ami': 0.4646},
        {'k': 10, 'stability_ari': 0.695, 'silhouette': 0.0662, 'intent_alignment_ami': 0.499},
        {'k': 12, 'stability_ari': 0.8284, 'silhouette': 0.0651, 'intent_alignment_ami': 0.5167},
        {'k': 15, 'stability_ari': 0.9086, 'silhouette': 0.073, 'intent_alignment_ami': 0.5128},
        {'k': 18, 'stability_ari': 0.7639, 'silhouette': 0.059, 'intent_alignment_ami': 0.5359},
        {'k': 20, 'stability_ari': 0.8158, 'silhouette': 0.056, 'intent_alignment_ami': 0.5167},
        {'k': 25, 'stability_ari': 0.8136, 'silhouette': 0.0557, 'intent_alignment_ami': 0.5267},
        {'k': 30, 'stability_ari': 0.7725, 'silhouette': 0.0543, 'intent_alignment_ami': 0.5225},
        {'k': 40, 'stability_ari': 0.7407, 'silhouette': 0.0557, 'intent_alignment_ami': 0.5254},
        {'k': 50, 'stability_ari': 0.6609, 'silhouette': 0.0547, 'intent_alignment_ami': 0.5198},
        {'k': 65, 'stability_ari': 0.6491, 'silhouette': 0.0586, 'intent_alignment_ami': 0.513},
        {'k': 80, 'stability_ari': 0.6306, 'silhouette': 0.0552, 'intent_alignment_ami': 0.504},
        {'k': 100, 'stability_ari': 0.6368, 'silhouette': 0.049, 'intent_alignment_ami': 0.4988},
        {'k': 120, 'stability_ari': 0.6439, 'silhouette': 0.0509, 'intent_alignment_ami': 0.4907},
    ]
    out = triangulate_k(sweep, {"k_estimate": 28}, (15, 25))
    said = out["chosen_by"]

    m = re.search(r"ties within ([\d.]+) — ([\d.]+)x measured noise \(se=([\d.]+)\)", said)
    assert m, f"the tie band is no longer stated in a checkable form: {said}"
    band, z, se = (float(g) for g in m.groups())
    # Both are printed at 5 significant figures, so a reader's multiplication must
    # land within the last printed digit — not to machine precision, which no
    # rounded display can offer.
    assert abs(z * se - band) <= 5e-7, (
        f"the stated arithmetic does not hold: {z} x {se} = {z * se}, printed {band}")


def test_the_recorded_locator_profile_cannot_contradict_the_deciding_reference():
    """One artifact must not give two answers to "what located K?".

    `reference_profile` was written when the phrasing groups were the only
    locator, and kept describing them after `choose_locator` was given other
    candidates. live42 shipped, in the same `granularity.json`:

        triangulation.deciding_reference : legacy_l2
        triangulation.locator            : ami_vs_legacy_l2
        locator_reference.reference      : "phrasing (template) groups"
        locator_reference.caveat         : "K is located by maximising AMI
                                            against THIS partition"

    Its own p5 observer caught the contradiction. This is the same defect as the
    decision record's hardcoded `decisive_metrics`, one file over — a field that
    was correct when there was one option and silently wrong once there were
    three.

    The caveat also asserted the located K "tracks its cardinality", which a
    coverage-controlled experiment refuted: holding the row set fixed and varying
    only class count, 6, 9 and 25 classes all locate the same K. It tracks REACH.
    """
    from qmine.ops.cluster import reference_profile

    masks = {f"g{i}": None for i in range(3)}
    prof = reference_profile(None, 100)
    assert prof["reference"] == "none"

    import numpy as np

    real = {f"g{i}": np.zeros(100, bool) for i in range(3)}
    for i, m in enumerate(real.values()):
        m[i * 10:(i + 1) * 10] = True
    prof = reference_profile(real, 100)

    # It must not claim to BE the locator...
    assert "THIS partition" not in prof["caveat"]
    # ...and must point at where the real answer lives.
    assert "deciding_reference" in prof["caveat"]
    # ...and must not repeat the refuted cardinality claim.
    assert "cardinality" not in prof["caveat"].lower()

    # p5 stamps which reference actually decided.
    import inspect

    from qmine.graph.nodes.bottomup import p5_granularity

    src = inspect.getsource(p5_granularity)
    assert 'profile["deciding_reference"]' in src, (
        "the profile ships without saying which reference actually decided")
    assert "deciding_reference_profile" in src, (
        "when another reference decides, its own reach/discrimination must be "
        "recorded — otherwise the artifact describes the wrong partition")


def test_a_flagged_row_shows_its_NEAREST_neighbours_and_the_whole_set():
    """Two fields that measured different populations under one name.

    `disagreement` was computed over all k neighbours while `neighbour_labels`
    stored `neigh[:6]` — and because `np.argpartition` does not order the top-k,
    those six were an ARBITRARY subset, not the six most similar. The docstring
    calls these flags "candidates for human review", so a reviewer was being
    shown six neighbours that need not include the closest ones.

    It also manufactured false confirmations. live44's observer asserted
    `disagreement == 1.0 or label in neighbour_labels`; the check FAILED and was
    NOT a defect — at k=10 and disagreement 0.9 exactly one neighbour agrees, and
    P(it is absent from an arbitrary 6 of 10) = 40%.
    """
    import numpy as np

    from qmine.ops.classify import knn_label_scan

    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 8))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    labels = ["A"] * 30 + ["B"] * 30

    out = knn_label_scan(X, list(range(60)), labels)
    assert out["flags"], "the fixture must actually flag something"

    for f in out["flags"]:
        # The published neighbourhood IS the one disagreement was computed over.
        assert len(f["neighbour_labels"]) == f["k"], \
            "a 6-of-k sample is not commensurable with a k-wide statistic"
        recomputed = sum(1 for v in f["neighbour_labels"] if v != f["label"]) / f["k"]
        assert abs(recomputed - f["disagreement"]) < 1e-9, \
            "disagreement must be recomputable from the labels published beside it"
        # And the majority must be the majority OF THAT SET.
        from collections import Counter
        assert f["neighbour_majority"] == Counter(f["neighbour_labels"]).most_common(1)[0][0]


def test_the_neighbour_sample_is_ordered_by_similarity():
    """`argpartition` selects the top-k without ordering them."""
    import inspect

    from qmine.ops import classify

    src = inspect.getsource(classify.knn_label_scan)
    assert "argsort" in src, \
        "argpartition does not order its output; the reviewer needs nearest-first"


def test_an_agent_boolean_never_defaults_to_the_reassuring_answer():
    """`coherent=True` and `risk=False` made a failed generation look clean.

    A partially-failed tree_auditor produced families asserted coherent and
    not-risky rather than families carrying no verdict — the same permissive-
    default mode as `SectionDraft.markdown=""` and `AnnotationBatch.labels=[]`,
    which silently lost 1,500 gold rows, except here it is on the SAFETY path.
    live44's F10 and F21 read `risk=false` while containing members the audit
    had itself flagged for isolation.

    `None` (not asserted) must be distinguishable from `False` (asserted safe).
    """
    from qmine.records import FamilyNaming, TreeAudit

    fam = FamilyNaming(family_id=1, name_zh="x", code="x", definition="d")
    assert fam.risk is None, "an unstated risk verdict must not read as 'no risk'"
    assert fam.coherent is None, "an unstated coherence verdict must not read as 'coherent'"

    audit = TreeAudit()
    assert audit.risk_isolated is None, \
        "an unstated isolation claim must not read as 'not isolated' either — " \
        "it is a claim the pipeline never verifies"

    # An explicit verdict still round-trips.
    assert FamilyNaming(family_id=1, name_zh="x", code="x", definition="d",
                        risk=True, coherent=False).risk is True


def test_a_phase_that_lost_its_agent_says_so():
    """live44: `✔ p12_maintain completed` after 44 minutes and zero tokens.

    The maintainer timed out on every attempt. The mechanical half of the phase
    — baseline, novelty sentinel, drift comparison — had genuinely succeeded, so
    completing is correct. But the failure was written to `maintenance.json` and
    NOWHERE else: not the log, not a deliverable. An operator watching the run
    saw a green phase.
    """
    import inspect

    from qmine.graph.nodes import delivery

    src = inspect.getsource(delivery)
    i = src.index("analyst_error")
    window = src[i:i + 600]
    assert "deps.emit" in window, \
        "an agent that failed entirely must reach the operator, not just the artifact"


def test_calibration_is_measured_against_a_null_not_a_constant():
    """ECE moved 0.023 → 0.065 between runs and no gate read either number.

    `p2c_trainable` only asks whether there was enough gold to fit anything, so
    accuracy, macro-F1 and ECE all reached the log and the artifact and nothing
    gated any of them.

    A bare ECE also cannot be read as good or bad: finite samples push it off
    zero, so the reference must be computed on THIS run. Measured, a PERFECTLY
    calibrated model at n=5,812 still shows ECE ≈ 0.0074 ± 0.0028 — which means
    live42's 0.023 was already ~5 sd out and was never the clean baseline it
    looked like. Comparing one run's ECE to another's imports that run's sample
    size and confidence profile, which is the mistake this replaces.
    """
    import numpy as np

    from qmine.ops.classify import ece_noise_floor

    rng = np.random.default_rng(1)
    conf = np.clip(rng.normal(0.85, 0.1, 5000), 0.3, 0.999)
    mean, sd = ece_noise_floor(conf)

    assert mean > 0, "a perfectly calibrated model still shows ECE at finite n"
    assert sd > 0, "and the gate needs its spread, not just a point estimate"
    # The null must SHRINK with n — that is what makes it a sample-size correction.
    small, _ = ece_noise_floor(conf[:400])
    assert small > mean, f"null must be larger at smaller n, got {small} vs {mean}"


def test_a_classifier_quality_gate_exists_and_imports_no_threshold():
    """Gates must not carry a constant that only fits one corpus."""
    import inspect

    from qmine.graph.nodes import topdown

    src = inspect.getsource(topdown.p2c_classifier)
    assert '"p2c_calibration"' in src, "calibration must be gated, not merely logged"
    assert "ece_null_mean" in src, "and gated against the run's own null"
    assert "z_max" in src, "a z-threshold travels across corpora; an ECE level does not"


def test_a_duplicate_leaf_pair_can_actually_be_acted_on():
    """live44 listed 14 duplicate pairs and prescribed nothing on any of them.

    The vocabulary had `merge_families` but no `merge_leaves`, so the tree could
    SPLIT a leaf and MERGE families and never merge two leaves. The auditor's
    `duplicate_leaf_pairs` was therefore a write-only measurement — it named
    leaves 12/14 as "汉字读音查询重复，任务无法区分" and the delivered tree shipped
    both, with byte-identical names, in the same family.

    The asymmetry ran in the damaging direction: every run could only fragment.
    """
    import numpy as np

    from qmine.ops.governance import execute_prescriptions
    from qmine.records import Prescription

    leaf = np.array([0, 0, 1, 1, 2, 2])
    fam = np.array([0, 0, 0, 0, 1, 1])
    pres = [Prescription(id="P1", kind="merge_leaves", targets=[1, 0],
                         rationale="no user could tell these apart")]

    *_, res = execute_prescriptions(pres, fam, leaf_labels=leaf)

    assert pres[0].status == "executed", "a merge must not silently decline"
    assert res["leaf_merges"]["n_rows_moved"] == 2
    assert len(set(res["leaf_labels"])) == 2, "the duplicate leaf must be gone"
    # Folded into the SMALLEST id, so re-runs are stable regardless of order.
    assert res["leaf_merges"]["map"] == {"1": 0}


def test_the_auditor_is_told_every_duplicate_needs_a_disposition():
    """Finding a duplicate and prescribing nothing is the write-only failure."""
    from pathlib import Path

    import qmine

    prompt = (Path(qmine.__file__).parent / "agents/prompts/auditor.md").read_text()
    assert "merge_leaves" in prompt, "the auditor cannot prescribe a kind it is never told exists"
    assert "disposition" in prompt, "every listed pair must resolve to merge or a documented keep"


def test_a_delivered_family_is_named_after_the_leaves_it_actually_holds():
    """`混合·主要成分「词语含义查询」45%` was being used AS a family's name.

    The tree auditor names the PHASE 7 tree; governance then merged 18 families
    into 14 and isolated them back out to 23. Different id spaces, and a
    delivered family routinely spans several audit families — so the only thing
    `family_names` could emit was a composition label, in headings, table cells,
    a Mermaid node and a CSV column. A reader cannot tell whether that is a name
    or a warning, and it reports a percentage whose denominator is not the
    family.

    p8 now names the delivered partition directly, the same way it already
    re-names the leaves governance changed.
    """
    from qmine.report._shape import family_names

    naming = {
        "families_final": [
            {"family_id": 7, "name_zh": "汉字读音查询", "leaf_ids": [12, 14]},
            {"family_id": 8, "name_zh": "汉字字形查询", "leaf_ids": [27, 29]},
        ],
        # The audit's stale view must NOT win.
        "audit": {"families": [{"family_id": 7, "name_zh": "字词释义查询", "leaf_ids": [3, 4]}]},
    }
    out = family_names(naming, [7, 7, 8, 8], [1, 1, 1, 1])
    assert out[7] == "汉字读音查询", "the delivered name must win over the audit's"
    assert "混合" not in out[7] and "%" not in out[7], \
        "a delivered family has a NAME, not a composition diagnostic"


def test_two_delivered_leaves_in_one_family_never_share_a_name():
    """live44 shipped two `汉字读音查询` (12/14) and two `汉字笔画数查询` (30/50).

    Deterministic on purpose. Every other duplicate check here is a cosine with a
    threshold, and live44's auditor returned `cosine: null` for 4 of the 14 pairs
    it reported — the geometric test could not score a third of its own findings.
    An exact name collision needs no threshold, no embedding and no corpus
    constant, and it is the condition that actually matters to a reader: two
    identically-named leaves cannot be chosen between, however far apart their
    centroids are.
    """
    import numpy as np

    from qmine.ops.governance import indistinguishable_leaves

    lab = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    fam = np.array([7, 7, 8, 8])
    namings = [
        {"leaf_id": 0, "name_zh": "汉字读音查询"},
        {"leaf_id": 1, "name_zh": "汉字读音查询"},   # collision, same family
        {"leaf_id": 2, "name_zh": "笔画数查询"},
        {"leaf_id": 3, "name_zh": "笔顺查询"},
    ]
    out = indistinguishable_leaves(lab, fam, namings)
    assert len(out) == 1 and out[0]["leaf_ids"] == [0, 1]
    assert out[0]["family_id"] == 7

    # The same name in DIFFERENT families is not a collision: a reader picks the
    # family first, so the pair is still distinguishable.
    fam2 = np.array([7, 9, 8, 8])
    assert indistinguishable_leaves(lab, fam2, namings) == []

    # An unnamed leaf is a different defect (`p7_all_leaves_named` owns it) and
    # must not be reported as a name collision with another unnamed leaf.
    blank = [{"leaf_id": 0, "name_zh": ""}, {"leaf_id": 1, "name_zh": ""},
             {"leaf_id": 2, "name_zh": "x"}, {"leaf_id": 3, "name_zh": "y"}]
    assert indistinguishable_leaves(lab, fam, blank) == []


def test_the_collision_gate_reads_a_re_measurement_not_the_attempt():
    """A disambiguation pass that ran is not a disambiguation that worked.

    The resolver must return what is STILL colliding after it edits the namings,
    and it must refuse a "fix" that returns the same string twice — otherwise the
    gate passes on a promise.
    """
    import inspect

    from qmine.graph.nodes import naming

    src = inspect.getsource(naming._resolve_indistinguishable_leaves)
    assert src.rstrip().endswith('return indistinguishable_leaves(leaf_labels, leaf_family, naming["namings"])'), \
        "the resolver must end by RE-MEASURING, not by returning its own edits"
    assert "len(set(fresh.values())) == len(g[\"leaf_ids\"])" in src, \
        "names that are not actually distinct must be rejected"

    gate_src = inspect.getsource(naming)
    assert '"p8_leaves_are_distinguishable"' in gate_src
    assert "passed=not still_colliding" in gate_src, "the gate must read the re-measurement"


def test_a_non_default_text_column_survives_canonicalisation():
    """`--text-column original_query` halted p1 with KeyError: 'original_query'.

    `build_frame` is "the canonical dataframe every later phase reads" and always
    names the text column `query`. `cfg.data.text_column` is the SOURCE FILE's
    name and stops being true the moment p1 builds that frame — while 44
    downstream sites read it, the first being `audit_corpus` twenty lines later.

    Invisible on K12, where the column is already called `query`. The first real
    corpus with a different column name could not run at all.
    """
    import inspect

    from qmine.graph.nodes import foundation

    src = inspect.getsource(foundation.p1_audit)
    build = src.index("build_frame(")
    audit = src.index("audit_corpus(")
    assert build < audit, "fixture assumption: the frame is built before it is audited"
    between = src[build:audit]
    assert 'cfg.data.text_column = "query"' in between, \
        "the config must be canonicalised between building the frame and reading it"


def test_agreement_needs_two_references_to_be_agreement():
    """med01 published `references_agree: true` with exactly ONE reference.

    `len(set(located.values())) <= 1` is vacuously true for a singleton. A corpus
    with no legacy labels — where mined phrasing groups are the only reference —
    therefore reported agreement, and the delivered summary read "full", while
    that single reference had located K=18 against a chosen K=12.

    `None` (not established) must be distinguishable from `True` (compared and
    consistent) and from `False` (compared and disagreed).
    """
    import inspect

    from qmine.ops import cluster

    src = _code_only(inspect.getsource(cluster))
    assert "if len(located) >= 2 else None" in src, \
        "agreement over a singleton is vacuous and must report None, not True"
    assert '"n_references"' in src, \
        "the basis must ship with the verdict so a reader can see it was a singleton"


def test_the_kmeans_verdict_is_never_read_from_a_cross_k_gap():
    """The verdict was always paired; the REPORT printed the unpaired number.

    `contradicted` comes from `largest_paired_margin`, which is correct. But the
    artifact also carried `alternative_beats_reference_by` — the best-alternative
    minus the reference, two rows that can sit at DIFFERENT k — and both the
    report and the decision evidence printed THAT. On med01 it read 0.1025
    (agglo@k15 minus kmeans@k20) where the within-k margin at the reference's own
    k was 0.0531, and the p4 observer reasonably took it for the decisive value.
    """
    import inspect

    from qmine.graph.nodes import bottomup
    from qmine.ops import cluster
    from qmine.report import zh_bottomup

    verdict_src = _code_only(inspect.getsource(cluster))
    assert 'per_k[_ref_k]["margin"]' in verdict_src, \
        "the gap must be the paired margin at the REFERENCE'S OWN k"
    assert '"decided_by": "largest_paired_margin"' in verdict_src, \
        "the artifact must say which number decided"
    assert '"alternative_beats_reference_at_k"' in verdict_src, \
        "and the k it was measured at, so a reader can check it is not cross-k"

    # The report must print reference, alternative and margin from ONE k. Naming
    # a reference at k=20 beside an alternative at k=15 is what made the p4
    # observer read a gap no single comparison produced.
    rep = _code_only(inspect.getsource(zh_bottomup))
    assert "paired_margins_within_k" in rep, \
        "the report must source its triple from the paired margins"
    assert "_ref_at_k" in rep, \
        "the reference shown must be the one at the k being reported"


def test_a_metric_row_carries_the_basis_it_was_measured_on():
    """Two rows that must be identical differed, and nothing could explain why.

    `alpha_algebra` reduces to `cos_semantic` at a=0, so the a=0 sweep row IS the
    base encoder. med01 read template_fragmentation 2.4868 there against the
    bake-off's 2.4511 for that same encoder. The cause is benign — the bake-off
    scores a SUBSAMPLE and the sweep the full corpus — but neither row recorded
    `n`, so a correct observation became unresolvable.
    """
    import inspect

    from qmine.ops import represent

    src = inspect.getsource(represent)
    for fn in ("alpha_sweep", "encoder_bakeoff"):
        body = _code_only(inspect.getsource(getattr(represent, fn)))
        for field in ('"k"', '"n_rows"', '"seed"'):
            assert field in body, f"{fn} rows must record {field} to be comparable"


def _code_only(src: str) -> str:
    """Source with comment lines removed.

    A prose comment explaining why a field was removed necessarily NAMES that
    field, so a static assertion over raw source matches its own explanation.
    That trap cost real time more than once; strip comments before asserting.
    """
    out = []
    for line in src.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        out.append(line.split("  # ")[0] if "  # " in line else line)
    return "\n".join(out)



def test_delivered_family_names_are_actually_persisted():
    """The K12 demo logged "named 32/32 delivered families" and shipped none.

    `_name_delivered_families` mutates the dict it is handed. The call site
    passed a throwaway `deps.load("tree_naming")` and never wrote it back, so
    `families_final` was `[]` in the artifact and the reports still carried
    `混合·主要成分「X」N%` — the exact label the naming exists to replace.

    A naming that is not written back is a naming that did not happen, and the
    emit line said otherwise, which is worse than silence.
    """
    import inspect

    from qmine.graph.nodes import naming

    src = _code_only(inspect.getsource(naming))
    i = src.index("_name_delivered_families(deps,")
    window = src[i:i + 500]
    assert "put_json" in window, \
        "the mutated tree_naming must be persisted or the naming is discarded"
    assert "cache_put" in window, \
        "and cached, so later phases in the same run see the names"


def test_an_offline_family_name_never_suppresses_the_audit_disclosure():
    """A stand-in name is not a name, and it must not hide a missing one.

    `family_names` prefers `families_final` — the names p8 gives the DELIVERED
    partition. Offline, `FamilyNamerAgent` returns "[offline-heuristic] 未命名分组"
    for every family, and persisting that put a placeholder in the title position
    AND suppressed the disclosure the audit-derived path makes: a family the tree
    audit never covered is supposed to read "树审计未覆盖 (治理新建)".

    Same rule as `render --no-agents`: complete-looking prose no model wrote is
    worse than a marked hole.
    """
    import inspect

    from qmine.graph.nodes import naming

    src = _code_only(inspect.getsource(naming._name_delivered_families))
    assert 'provider != "offline"' in src, \
        "an offline stand-in must not be recorded as a delivered family name"
    assert 'naming["families_final"] = out' in src, \
        "and a real provider's names must still be recorded"


def test_a_class_code_matches_regardless_of_case():
    """med02 discarded 29% of its adjudication rules to a casing difference.

    The architect chooses the casing of `TaxonomyNode.code`; the rule writer is a
    SEPARATE call that independently chooses the casing it emits in `then`.
    Nothing makes them agree. med02's nodes came back lowercase
    (`substance_condition_matching`) while every rule target was uppercase, and
    **14 of 42 rules were dropped as "naming no class"** — including every rule
    addressing that corpus's own measured top confusions.

    A case difference is not ambiguity: `code` is the identity, so `X` and `x`
    cannot be two different classes. Resolve it. Genuine ambiguity — two DISTINCT
    codes named — must still refuse.
    """
    from qmine.ops.rule_conflict import normalise_then

    codes = ["substance_condition_matching", "lab_result_interpretation",
             "symptom_cause_analysis"]

    # Case-mismatched, embedded in a sentence -> resolves to the CANONICAL code.
    r = normalise_then("SUBSTANCE_CONDITION_MATCHING；仅药名+功效模板归", codes)
    assert r.is_key and r.code == "substance_condition_matching", \
        "a rule must not die because two agents disagreed about capitalisation"

    # Bare, case-mismatched.
    assert normalise_then("SYMPTOM_CAUSE_ANALYSIS", codes).code == "symptom_cause_analysis"

    # Still refuses what it should: no class at all.
    assert not normalise_then("如果有用法/剂量词，归", codes).is_key
    # And two distinct classes remain ambiguous, whatever the case.
    two = normalise_then("归 SYMPTOM_CAUSE_ANALYSIS，不归 LAB_RESULT_INTERPRETATION", codes)
    assert not two.is_key and len(two.found) == 2, \
        "naming two classes is ambiguous; picking a side would rewrite the rule"


def test_an_alpha_chosen_at_the_grid_edge_says_the_optimum_was_not_bracketed():
    """med03 chose alpha=1.0 — the largest value searched — with fragmentation
    still falling at that edge (2.487 -> 1.771 -> 1.481).

    So the sweep never bracketed the optimum, and alpha > 1.0 may be better. At
    1.0 the phrasing block controls 50% of the cosine, which is a large
    representational commitment to make at an unexplored boundary. The winner was
    also the LEAST stable point in the sweep (0.642 vs 0.841 at alpha=0) —
    legitimate under the rule that fragmentation locates and stability only
    vetoes, but three facts a reader should get together and currently gets none
    of.

    Disclosed, never auto-extended: the grid proposer is blind to scores so its
    additions stay pre-registered, and widening BECAUSE the winner sits at the
    edge would use the scores it must not see.
    """
    import inspect

    from qmine.ops import represent

    src = _code_only(inspect.getsource(represent.alpha_sweep))
    assert '"optimum_bracketed"' in src, "the artifact must say whether the search bracketed"
    assert '"chosen_alpha_at_grid_edge"' in src, "and whether the winner sits at an edge"

    # The rule itself: edge + still-improving => not bracketed; interior => bracketed.
    def bracketed(rows, winner):
        a = sorted(r["alpha"] for r in rows)
        by = {r["alpha"]: r["template_fragmentation"] for r in rows}
        edge = winner in (a[0], a[-1])
        still = (by[a[-1]] < by[a[-2]]) if winner == a[-1] else (by[a[0]] < by[a[1]])
        return not (edge and still)

    med03 = [{"alpha": 0.0, "template_fragmentation": 2.4868},
             {"alpha": 0.7, "template_fragmentation": 1.771},
             {"alpha": 1.0, "template_fragmentation": 1.4808}]
    assert bracketed(med03, 1.0) is False, "still improving at the largest alpha searched"

    interior = [{"alpha": 0.0, "template_fragmentation": 2.5},
                {"alpha": 0.5, "template_fragmentation": 1.4},
                {"alpha": 1.0, "template_fragmentation": 2.0}]
    assert bracketed(interior, 0.5) is True, "an interior optimum IS bracketed"


def test_a_leaf_merge_never_voids_a_pending_risk_isolation():
    """med04 shipped two risk clusters unisolated, and I caused it.

    `merge_leaves` rewrites `new_labels`; `isolate_leaves` indexes `new_family`
    by LEAF ID. Merging a leaf away empties it, so the isolation that follows
    moves an EMPTY leaf into a fresh family while the rows it carried sit in the
    survivor, unisolated. med04 did this to leaves 14 and 24 — both in the risk
    report — and left ghost families 42 and 33 holding no rows, so the artifact
    reported 36 families where only 34 had content.

    My original comment claimed "the survivor carries the merged rows into
    whatever family isolation then puts it in". It does not: isolation names a
    LEAF, not a row set.

    Isolation is a SAFETY action, merging a QUALITY one, so the merge yields.
    Redirecting the isolation to the survivor is worse — it would isolate the
    survivor's OTHER rows too. A surviving duplicate is cosmetic; unisolated
    risk content is not, and the duplicate still routes to
    `p8_leaves_are_distinguishable`.
    """
    import numpy as np

    from qmine.ops.governance import execute_prescriptions, isolate_leaves
    from qmine.records import Prescription

    leaf = np.array([12, 12, 14, 14, 4, 4])
    fam = np.zeros(25, dtype=int)
    pres = [
        Prescription(id="M1", kind="merge_leaves", targets=[14, 12], rationale="duplicates"),
        Prescription(id="I1", kind="isolate_leaf", targets=[14], rationale="risk content"),
    ]
    *_, res = execute_prescriptions(pres, fam, leaf_labels=leaf)

    assert pres[0].status == "declined", "the merge must yield to the isolation"
    assert "isolation" in (pres[0].decline_reason or ""), "and say why"
    assert pres[1].status == "executed", "the safety action must still run"
    assert 14 in {int(x) for x in np.unique(res["leaf_labels"])}, \
        "leaf 14 must still carry rows, or the isolation moves nothing"
    assert 14 in (res["isolations"]["isolated"]), "and it must actually be isolated"
    assert res["leaf_merges"]["refused_because_isolated"] == [14]

    # Defence in depth: isolating an already-empty leaf creates a GHOST family.
    out, det = isolate_leaves(fam, [14], leaf_labels=np.array([12, 12, 4, 4]))
    assert 14 in det.get("skipped", {}), "an empty leaf must not be isolated"
    assert not det["isolated"], "and no family may be created for it"


def test_declining_a_merge_still_counts_as_settled():
    """A declined prescription is settled; `p8_governance_executed` must not halt
    on a refusal that names its reason. The gate exists for UNAPPLIED findings,
    not for ones the pipeline deliberately refused and explained."""
    import numpy as np

    from qmine.ops.governance import execute_prescriptions
    from qmine.records import Prescription

    pres = [
        Prescription(id="M1", kind="merge_leaves", targets=[1, 0], rationale="dup"),
        Prescription(id="I1", kind="isolate_leaf", targets=[1], rationale="risk"),
    ]
    execute_prescriptions(pres, np.zeros(5, dtype=int),
                          leaf_labels=np.array([0, 0, 1, 1]))
    assert all(p.status in {"executed", "declined"} for p in pres), \
        "every prescription must end settled, never left 'proposed'"


def test_the_fragmentation_cell_survives_a_corpus_with_no_trusted_groups():
    """med04's notebook ran 3 of 18 cells and died: KeyError '有效家族数 exp(H)'.

    All 12 mined template groups came back `trusted=False`, so the cell's
    `groups` dict was empty, `rows` stayed `[]`, and `pd.DataFrame([])` has no
    columns — sorting by name raises. A generated cell must survive its own
    corpus, and a broken notebook is a broken DELIVERABLE, not a cosmetic issue.
    """
    import re

    import numpy as np
    import pandas as pd

    from qmine.report import zh_notebook

    src = zh_notebook.__file__ and open(zh_notebook.__file__).read()
    i = src.index("# %% [3] 现场演算")
    cell = src[i:src.index('"""))', i)].replace("\\\\n", "\\n")

    df = pd.DataFrame({"query": ["血压正常值"] * 40 + ["黄精的功效"] * 40})
    labels = np.array([0] * 40 + [1] * 40)
    fam = np.array([0, 1])
    # THE FAILING CASE: every group untrusted, so none is usable.
    tmpl = {"groups": [{"name": f"suffix:g{i}", "pattern": "正常值", "trusted": False}
                       for i in range(12)]}
    ns = {"pd": pd, "np": np, "re": re, "df": df, "labels": labels, "fam": fam,
          "tmpl": tmpl, "display": lambda x: None}
    exec(cell, ns)  # must not raise

    assert ns["frag"] is not None and len(ns["frag"]) == 0, \
        "an empty result must be an empty TABLE, not a crash"

    # And the normal path still works.
    tmpl2 = {"groups": [{"name": "suffix:正常值", "pattern": "正常值", "trusted": True}]}
    ns2 = dict(ns); ns2["tmpl"] = tmpl2
    exec(cell, ns2)  # noqa: S102
    assert len(ns2["frag"]) == 1, "a trusted group with enough hits must still be measured"
