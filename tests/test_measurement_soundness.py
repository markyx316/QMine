

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
