

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
