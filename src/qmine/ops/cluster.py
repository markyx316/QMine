"""Phases 4-6 — algorithm selection, granularity, hierarchy, refinement.

Three decisions live here, and each has a designated evidence source:

* **Which algorithm** — a bake-off (Phase 4), not a default.  KMeans usually
  wins on L2-normalised embeddings for a structural reason: the points lie on a
  unit sphere, cosine neighbourhoods there are close to isotropic, and that is
  precisely KMeans' assumption.  We still run the battery, because "usually"
  is not evidence and the losers have jobs afterwards — HDBSCAN's density view
  becomes the novelty sentinel in Phase 12.

* **How many clusters** — replay stability, triangulated (Phase 5).  There is no
  ground truth to appeal to, but there is reproducibility: if two different
  seeds over the same data produce partitions that disagree, the structure was
  never there.  Three independent estimators must land on the same scale before
  we commit.

* **What shape the tree is** — coarse and stable at the top, locally chosen
  below (Phase 6).  A single global K fine enough for leaves is not stable
  enough to deliver; the fix is to stop asking one number to do two jobs.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Sequence

import numpy as np
from sklearn.cluster import (
    BisectingKMeans,
    HDBSCAN,
    KMeans,
    MiniBatchKMeans,
)
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize

from ..determinism import SEED_METRIC, deterministic_subsample, rng

log = logging.getLogger("qmine.cluster")


# ==========================================================================
# Primitives
# ==========================================================================

def kmeans_labels(X: np.ndarray, k: int, *, seed: int = SEED_METRIC, minibatch: bool | None = None) -> np.ndarray:
    """Fit KMeans and return labels.  Switches to MiniBatch above 200k rows."""
    if minibatch is None:
        minibatch = len(X) > 200_000
    if minibatch:
        km = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=5, batch_size=2048)
    else:
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    return km.fit_predict(X)


def kmeans_fit(
    X: np.ndarray, k: int, *, seed: int = SEED_METRIC, minibatch: bool | None = None, fast: bool = False
):
    """Fit KMeans. ``fast`` lowers restart count for sweeps; see :func:`replay_stability`."""
    if minibatch is None:
        minibatch = len(X) > 200_000 or (fast and len(X) > 20_000)
    if minibatch:
        est = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=3 if fast else 5, batch_size=2048)
    else:
        est = KMeans(n_clusters=k, random_state=seed, n_init=3 if fast else 10)
    return est.fit(X)


def cosine_silhouette(
    X: np.ndarray, labels: np.ndarray, *, sample: int = 8000, seed: int = SEED_METRIC
) -> float:
    """Cosine silhouette on a fixed sub-sample.  Advisory only, never decisive."""
    if len(np.unique(labels)) < 2:
        return float("nan")
    return round(
        float(
            silhouette_score(
                X, labels, metric="cosine", sample_size=min(sample, len(X)), random_state=seed
            )
        ),
        4,
    )


def replay_stability(
    X: np.ndarray,
    k: int,
    *,
    seeds: tuple[int, int] = (0, 1),
    sample: int = 8000,
    algorithm: str = "kmeans",
    fit_sample: int = 25000,
    fast: bool = False,
) -> float:
    """**The primary evidence that a partition is real.**

    Fit the same algorithm twice on the same data under two different seeds,
    then compare the two partitions on a common held-out sample with the
    Adjusted Rand Index.  High agreement means the structure survives the
    arbitrary choices inside the optimiser; low agreement means we are reading
    tea leaves, no matter how good the silhouette looks.

    This is the metric that killed the one-shot fine-grained partition in the
    source project: at K≥65 agreement fell to roughly 0.5, which is not a
    deliverable, and the two-level design in :func:`build_hierarchy` exists
    because of it.
    """
    idx = deterministic_subsample(len(X), min(sample, len(X)), SEED_METRIC)
    Xs = X[idx]
    if algorithm == "agglomerative":
        # Ward has no seed to vary, so stability is measured the honest way:
        # fit on two disjoint halves and compare their predictions on the
        # common sample via nearest-centroid extrapolation.
        half = len(X) // 2
        perm = rng(SEED_METRIC).permutation(len(X))
        a, b = perm[:half], perm[half : 2 * half]
        la = _agglo_predict(X[a], Xs, k)
        lb = _agglo_predict(X[b], Xs, k)
        return round(float(adjusted_rand_score(la, lb)), 4)

    # Fit on a capped sub-sample. Stability asks whether the *structure* survives
    # a change of seed, and that question is answered as well by 25k rows as by
    # 500k — while an uncapped fit turns a five-point alpha sweep into fifteen
    # full-corpus KMeans runs. The comparison itself still happens on the fixed
    # common sample, so the number stays panel-comparable.
    # `fast` additionally lowers the restart count. A K-sweep is comparison work:
    # every K is measured by the same cheaper estimator, so the shape of the
    # curve — the only thing read off it — is preserved. The K that curve
    # selects is rebuilt at full effort in Phase 6, where the fit is a
    # commitment rather than a comparison.
    Xf = X if len(X) <= fit_sample else X[deterministic_subsample(len(X), fit_sample, SEED_METRIC)]
    m1 = kmeans_fit(Xf, k, seed=seeds[0], fast=fast)
    m2 = kmeans_fit(Xf, k, seed=seeds[1], fast=fast)
    return round(float(adjusted_rand_score(m1.predict(Xs), m2.predict(Xs))), 4)


#: One entry per (data fingerprint, subsample size). A linkage tree is expensive
#: and K-independent, so building it once and cutting it many times is both
#: faster and more faithful to what a hierarchical method actually produces.
_LINKAGE_CACHE: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}


def _linkage_for(X: np.ndarray, subsample: int) -> tuple[np.ndarray, np.ndarray]:
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import pdist

    from ..determinism import hash_array

    key = (hash_array(X[:: max(len(X) // 500, 1)]), subsample)
    if key not in _LINKAGE_CACHE:
        idx = deterministic_subsample(len(X), min(subsample, len(X)), SEED_METRIC)
        Z = linkage(pdist(X[idx], metric="cosine"), method="average")
        _LINKAGE_CACHE[key] = (idx, Z)
    return _LINKAGE_CACHE[key]


def _agglo_cut(X: np.ndarray, k: int, subsample: int) -> np.ndarray:
    """Cut the cached linkage at ``k`` and extend to the full corpus by centroid.

    Extension by nearest centroid is how a sub-sampled hierarchical partition
    becomes comparable with the partitional methods, which all see every row.
    """
    from scipy.cluster.hierarchy import fcluster

    idx, Z = _linkage_for(X, subsample)
    lab = fcluster(Z, t=k, criterion="maxclust") - 1
    present = [c for c in range(lab.max() + 1) if (lab == c).any()]
    cents = normalize(np.vstack([X[idx][lab == c].mean(0) for c in present]))
    return np.asarray((X @ cents.T).argmax(1))


def _agglo_predict(X_fit: np.ndarray, X_query: np.ndarray, k: int) -> np.ndarray:
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist

    sub = X_fit[deterministic_subsample(len(X_fit), min(4000, len(X_fit)), SEED_METRIC)]
    Z = linkage(pdist(sub, metric="cosine"), method="average")
    lab = fcluster(Z, t=k, criterion="maxclust") - 1
    present = [c for c in range(lab.max() + 1) if (lab == c).any()]
    cents = normalize(np.vstack([sub[lab == c].mean(0) for c in present]))
    return np.asarray((X_query @ cents.T).argmax(1))


# ==========================================================================
# Phase 4 — the battery
# ==========================================================================

def algorithm_battery(
    X: np.ndarray,
    *,
    ks: Sequence[int] = (15, 20, 30),
    seeds: tuple[int, int] = (0, 1),
    silhouette_sample: int = 8000,
    include_hdbscan: bool = True,
    hdbscan_min_sizes: Sequence[int] = (50, 100, 200),
    agglo_subsample: int = 8000,
    battery_subsample: int = 20000,
    hdbscan_subsample: int = 15000,
    hdbscan_dims: int = 50,
) -> dict[str, Any]:
    """Run every candidate through one identical measurement harness.

    Uniformity is the whole value: the same sub-sample, the same seeds, the same
    silhouette call.  Numbers quoted from each algorithm's own paper, or from
    whenever we happened to run it last, are not comparable and quietly decide
    things they should not.

    The battery runs on a capped sub-sample because it is a *ranking* exercise —
    we are asking which algorithm suits this geometry, not producing the final
    partition. Agglomerative clustering materialises an n x n distance matrix, so
    an uncapped battery on a large corpus is an out-of-memory error rather than
    an answer.
    """
    if len(X) > battery_subsample:
        X = X[deterministic_subsample(len(X), battery_subsample, SEED_METRIC)]
    rows: list[dict[str, Any]] = []

    def _row(name: str, labels: np.ndarray, stab: float, **extra: Any) -> dict[str, Any]:
        uniq = np.unique(labels)
        n_noise = int((labels == -1).sum())
        return {
            "algorithm": name,
            "n_clusters": int(len([u for u in uniq if u >= 0])),
            "noise_rate": round(n_noise / len(labels), 4),
            "silhouette": cosine_silhouette(X, labels, sample=silhouette_sample),
            "stability_ari": stab,
            **extra,
        }

    for k in ks:
        rows.append(_row(f"kmeans_k{k}", kmeans_labels(X, k, seed=seeds[0]),
                         replay_stability(X, k, seeds=seeds, sample=silhouette_sample)))

        mb1 = MiniBatchKMeans(k, random_state=seeds[0], n_init=5, batch_size=2048).fit(X)
        mb2 = MiniBatchKMeans(k, random_state=seeds[1], n_init=5, batch_size=2048).fit(X)
        common = deterministic_subsample(len(X), min(silhouette_sample, len(X)), SEED_METRIC)
        rows.append(_row(f"minibatch_k{k}", mb1.labels_,
                         round(float(adjusted_rand_score(mb1.predict(X[common]), mb2.predict(X[common]))), 4)))

        b1 = BisectingKMeans(k, random_state=seeds[0]).fit(X)
        b2 = BisectingKMeans(k, random_state=seeds[1]).fit(X)
        rows.append(_row(f"bisecting_k{k}", b1.labels_,
                         round(float(adjusted_rand_score(b1.predict(X[common]), b2.predict(X[common]))), 4)))

        # One linkage tree, cut at each K. Refitting per K would repeat an O(n²)
        # computation that a hierarchical method only needs to do once — and it
        # is the single most expensive thing in this battery.
        agglo_labels = _agglo_cut(X, k, agglo_subsample)
        rows.append(_row(f"agglo_average_k{k}", agglo_labels,
                         replay_stability(X, k, seeds=seeds, algorithm="agglomerative",
                                          sample=silhouette_sample)))

        # GMM on a PCA projection: full-covariance Gaussians in 768-d are both
        # intractable and a poor fit for points constrained to a sphere.
        try:
            from sklearn.decomposition import PCA

            P = PCA(n_components=min(128, X.shape[1] - 1), random_state=SEED_METRIC).fit_transform(X)
            g1 = GaussianMixture(k, covariance_type="diag", random_state=seeds[0], max_iter=60).fit(P)
            g2 = GaussianMixture(k, covariance_type="diag", random_state=seeds[1], max_iter=60).fit(P)
            rows.append(_row(f"gmm_diag_k{k}", g1.predict(P),
                             round(float(adjusted_rand_score(g1.predict(P[common]), g2.predict(P[common]))), 4)))
        except Exception as exc:  # noqa: BLE001
            log.warning("GMM failed at k=%d: %s", k, exc)

    if include_hdbscan:
        # Density estimation degrades in high dimensions — distances concentrate,
        # every point looks equidistant, and the mutual-reachability graph costs
        # far more to build for a worse answer. Projecting first is both the
        # standard remedy and what the Phase 12 novelty sentinel will do, so the
        # screen here matches the deployed use.
        from sklearn.decomposition import PCA

        Xd = X if len(X) <= hdbscan_subsample else X[
            deterministic_subsample(len(X), hdbscan_subsample, SEED_METRIC)
        ]
        Xd = PCA(n_components=min(hdbscan_dims, Xd.shape[1] - 1),
                 random_state=SEED_METRIC).fit_transform(Xd)
        for mcs in hdbscan_min_sizes:
            try:
                h = HDBSCAN(min_cluster_size=int(mcs), metric="euclidean", copy=True).fit(Xd)
                uniq = [u for u in np.unique(h.labels_) if u >= 0]
                rows.append({
                    "algorithm": f"hdbscan_mcs{mcs}",
                    "n_clusters": len(uniq),
                    "noise_rate": round(float((h.labels_ == -1).mean()), 4),
                    "silhouette": cosine_silhouette(Xd, h.labels_, sample=silhouette_sample),
                    "stability_ari": float("nan"),
                    "min_cluster_size": int(mcs),
                    "space": f"PCA-{Xd.shape[1]} of {len(Xd)} rows",
                })
            except Exception as exc:  # noqa: BLE001
                log.warning("HDBSCAN failed at mcs=%s: %s", mcs, exc)

    return {"rows": rows, "verdict": _battery_verdict(rows)}


def _battery_verdict(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick a winner on stability, with density methods judged on their own terms.

    HDBSCAN is *not* ranked against the centroid methods on stability — it does
    not answer the same question, and its own paper's parameter search
    optimises a different objective.  It is screened separately by noise rate
    and cluster count, sorted for a human to read, because an automatic
    composite score once selected a configuration that discarded 46% of the
    corpus as noise and reported it as a win.
    """
    partitional = [r for r in rows if not r["algorithm"].startswith("hdbscan")]
    density = [r for r in rows if r["algorithm"].startswith("hdbscan")]
    ranked = sorted(
        [r for r in partitional if not np.isnan(r["stability_ari"])],
        key=lambda r: (-r["stability_ari"], -r["silhouette"]),
    )
    density_sorted = sorted(density, key=lambda r: (r["noise_rate"], -r["n_clusters"]))
    # The tree is built with KMeans - `build_hierarchy` hardcodes it - so an
    # "election" here was fiction: in half the runs it crowned an algorithm the
    # pipeline then ignored, and the report announced that winner as though it had
    # produced the delivered tree. Reframed as what it can honestly be: a
    # falsification probe. Does a structurally different algorithm, on the same
    # data, find the same structure? If yes, the partition is a property of the
    # corpus rather than of KMeans's spherical-cluster assumption. If no, that is a
    # warning worth printing, not a reason to silently switch algorithms mid-flight.
    km = [r for r in partitional if r["algorithm"].startswith("kmeans")]
    reference = max(km, key=lambda r: r["stability_ari"]) if km else (ranked[0] if ranked else None)
    best_other = ranked[0] if ranked else None
    margin = (round(best_other["stability_ari"] - reference["stability_ari"], 4)
              if reference and best_other else None)
    contradicted = bool(
        best_other and reference
        and not best_other["algorithm"].startswith("kmeans")
        and (margin or 0) > 0.10
    )
    return {
        "role": "falsification probe - the delivered tree is always KMeans (build_hierarchy)",
        "reference_algorithm": reference["algorithm"] if reference else None,
        "best_alternative": best_other["algorithm"] if best_other else None,
        "alternative_beats_reference_by": margin,
        "kmeans_assumption_contradicted": contradicted,
        "probe_note": (
            "A structurally different algorithm is more than 0.10 ARI more reproducible "
            "than KMeans here - the spherical-cluster assumption is doing visible work, "
            "and the family layer should be read as provisional."
            if contradicted else
            "No structurally different algorithm is materially more reproducible than "
            "KMeans, so the partition is not an artefact of its cluster-shape assumption."
        ),
        "chosen": reference["algorithm"] if reference else None,
        "chosen_family": "kmeans",
        "chosen_by": "not a selection - the tree is built with KMeans regardless",
        "ranking": [{"algorithm": r["algorithm"], "stability_ari": r["stability_ari"]} for r in ranked[:8]],
        "density_candidates_for_manual_review": [
            {"algorithm": r["algorithm"], "noise_rate": r["noise_rate"], "n_clusters": r["n_clusters"]}
            for r in density_sorted
        ],
        "density_note": (
            "HDBSCAN is screened by (noise_rate asc, n_clusters desc) for human review, "
            "never auto-selected by a composite score. Its role downstream is the "
            "Phase 12 novelty sentinel, not the main partition."
        ),
    }


# ==========================================================================
# Phase 5 — granularity
# ==========================================================================

def _intent_alignment(labels: np.ndarray, masks: dict[str, np.ndarray] | None) -> float:
    """Adjusted mutual information between a partition and the phrasing groups.

    The phrasing groups are the pipeline's only cheap source of *known* same-intent
    rows, and template fragmentation already uses them — but one-sidedly, penalising
    only over-splitting, which is why it falls monotonically as K rises and cannot
    say where K should be. AMI penalises both directions and is adjusted for chance,
    so it is comparable across K.

    Returns NaN when there are no groups, which keeps it out of any ranking.
    """
    if not masks:
        return float("nan")
    from sklearn.metrics import adjusted_mutual_info_score

    y = np.full(len(labels), -1, dtype=np.int64)
    for i, m in enumerate(masks.values()):
        m = np.asarray(m)
        if m.shape[0] != len(labels):
            return float("nan")
        y[m & (y == -1)] = i
    known = y >= 0
    if known.sum() < 50 or len(np.unique(y[known])) < 2:
        return float("nan")
    return round(float(adjusted_mutual_info_score(y[known], np.asarray(labels)[known])), 4)


def k_sweep(
    X: np.ndarray,
    ks: Sequence[int],
    *,
    seeds: tuple[int, int] = (0, 1),
    silhouette_sample: int = 8000,
    template_masks: dict[str, np.ndarray] | None = None,
    fast: bool = False,
    fit_sample: int = 25000,
) -> list[dict[str, Any]]:
    """Stability and silhouette across K.  The two curves rarely peak together.

    ``fast`` defaults to **off**, and that default was measured rather than
    assumed.  A cheap sweep (MiniBatch, three restarts) runs 13x faster on the
    50k K12 corpus, and it is tempting because this sweep is the slowest step in
    the pipeline.  But against a full-effort sweep over the same grid it scored a
    Spearman rank correlation of only **0.43** — the ordering of K values is
    substantially scrambled, even though on that particular run the argmax
    happened to agree.

    One lucky agreement is not evidence, and K is the most consequential number
    the bottom-up route produces: it sets the family layer that naming, audit,
    governance and the delivered table all inherit.  Six minutes of full-effort
    fitting is cheap insurance against re-running everything downstream.

    ``fast=True`` remains available for smoke tests and ``fast_mode`` runs, where
    the goal is to exercise the wiring rather than to choose anything.
    """
    from .templates import template_fragmentation

    # Fit the sweep's labelling model on the same capped sub-sample that
    # `replay_stability` uses, then assign the full corpus by nearest centroid.
    # Fitting all 50k rows only to measure silhouette on an 8k sub-sample is work
    # nobody reads, and at K=120 it is most of the phase's runtime. The chosen K
    # is rebuilt on the full corpus in Phase 6, where the fit is a commitment.
    Xf = X if len(X) <= fit_sample else X[deterministic_subsample(len(X), fit_sample, SEED_METRIC)]

    out = []
    for k in ks:
        model = kmeans_fit(Xf, k, seed=seeds[0], fast=fast)
        labels = model.predict(X) if len(Xf) < len(X) else model.labels_
        # Alignment against the phrasing groups — queries already known to share an
        # intent. This is the only metric in the sweep with a **two-sided** penalty
        # and therefore the only one that can locate K rather than merely bound it:
        # too few clusters and the groups get merged into unrelated traffic, too
        # many and they get split, so it has a genuine interior optimum. It is also
        # roughly ten times more precise than replay stability (seed sd ~0.01 vs
        # ~0.10 on these corpora), which matters because the differences being
        # resolved between adjacent K are about 0.05.
        row = {
            "k": int(k),
            "intent_alignment_ami": _intent_alignment(labels, template_masks),
            "stability_ari": replay_stability(X, k, seeds=seeds, sample=silhouette_sample, fast=fast),
            "silhouette": cosine_silhouette(X, labels, sample=silhouette_sample),
        }
        if template_masks:
            row["template_fragmentation"] = template_fragmentation(labels, template_masks)["mean_fragmentation"]
        out.append(row)
    return out


def deep_aligned_estimate(X: np.ndarray, k_expected: int, *, multiplier: int = 3, seed: int = SEED_METRIC) -> dict[str, Any]:
    """Over-cluster deliberately, then count the survivors.

    Fit ``multiplier × k_expected`` clusters and keep those that ended up with at
    least their fair share of rows.  Real categories can feed a cluster of
    average size; artefacts of over-splitting starve.  The count of survivors is
    an estimate of the *leaf* scale, arrived at without reusing the stability
    curve — which is what makes it usable as independent corroboration.
    """
    k_over = max(int(multiplier * k_expected), 4)
    km = kmeans_fit(X, k_over, seed=seed)
    sizes = np.bincount(km.labels_, minlength=k_over)
    threshold = len(X) / k_over
    survivors = int((sizes >= threshold).sum())
    return {
        "k_overcluster": k_over,
        "survival_threshold": round(float(threshold), 1),
        "k_estimate": survivors,
        "size_distribution": {
            "max": int(sizes.max()), "median": int(np.median(sizes)), "min": int(sizes.min())
        },
    }


def triangulate_k(
    sweep: list[dict[str, Any]],
    deep_aligned: dict[str, Any],
    expert_range: tuple[int, int],
    *,
    leaf_ratio: int = 3,
    #: Below this, a partition is rejected as irreproducible rather than ranked.
    stability_floor: float = 0.55,
    #: AMI differences at or under this are treated as ties. Measured, not assumed:
    #: the seed-to-seed sd of AMI on these corpora is 0.005-0.023.
    ami_tie_band: float = 0.02,   # fallback only; the band is measured when it can be
    ami_tie_z: float = 2.0,
) -> dict[str, Any]:
    """Locate the family scale, and name every K the measurement cannot rule out.

    Stability rejects; alignment with the phrasing groups locates.  The DeepAligned survivor count
    estimates the *leaf* scale, so it enters the comparison divided by the
    expected leaves-per-family.  The expert range is a prior from someone who
    knows the vertical.  Agreement is strong evidence; disagreement is not
    resolved by averaging — we take the K the LOCATOR points at (intent alignment;
    stability only rejects) and record the dissent,
    because a number nobody can defend is worse than a number with a caveat.
    """
    valid = [r for r in sweep if not np.isnan(r["stability_ari"])]

    # --- how K is actually located ------------------------------------------
    # Replay stability used to rank K directly. It cannot. Two measurements say so:
    #
    #   * It is degenerate. On every completed run its curve is still climbing
    #     below the grid — K=2 scores ARI 1.0000 on both the 8k and the 50k corpus,
    #     beating every K the sweep considers. Only the grid's lower bound stands
    #     between the pipeline and a two-way split, which makes an undocumented
    #     config constant the real author of the granularity decision.
    #   * It is too noisy to rank with. Four seed pairs at one K on this corpus
    #     gave 0.63, 0.60, 0.38, 0.69 — sd 0.14, against inter-K differences of
    #     about 0.05. Every K in the grid is inside every other K's error bar.
    #
    # That is precisely the role the stability literature gives it: a filter that
    # can reject a partition as irreproducible, not a ranker that can order
    # partitions. So it rejects here, and something else locates.
    #
    # The locator is alignment with the phrasing groups (AMI). It is two-sided —
    # penalised for merging known-same-intent rows AND for splitting them — so it
    # has an interior optimum, and it is ~10x more precise (sd ~0.01).
    stable = [r for r in valid if r["stability_ari"] >= stability_floor] or valid
    located = [r for r in stable if not np.isnan(r.get("intent_alignment_ami", float("nan")))]

    if located:
        # THE TIE BAND IS MEASURED, NOT ASSUMED. It used to be the constant 0.02,
        # with a comment beside it recording that AMI's seed sd is "~0.01" — i.e.
        # a 2-sd band, correct for THIS corpus and imported everywhere else. This
        # codebase's worst recurring defect is exactly that. `noise_floor` reads
        # the sweep's own roughness and returns 0.0105 here, which is the same
        # answer arrived at independently, and adapts on a corpus where it is not.
        #
        # Estimated on the FULL sweep, never on a filtered subset.
        from .select import noise_floor

        best = max(located, key=lambda r: r["intent_alignment_ami"])
        se = noise_floor([r["intent_alignment_ami"] for r in sweep])
        if np.isnan(se):
            band, band_source = ami_tie_band, f"configured {ami_tie_band} (sweep too short to measure)"
        else:
            band, band_source = ami_tie_z * se, f"{ami_tie_z:g}x measured noise (se={se:.4f})"
        tie_set = [r for r in located
                   if best["intent_alignment_ami"] - r["intent_alignment_ami"] <= band]
        peak = min(tie_set, key=lambda r: r["k"])   # inside a tie, prefer the simpler tree
        locator = "intent_alignment_ami"
    else:
        # No phrasing groups mined, so there is nothing to align against. Fall back
        # to the old rule and say so — this is the one case where stability ranks,
        # and it should be read as weak.
        peak = max(valid, key=lambda r: r["stability_ari"])
        tie_set = [peak]
        locator = "stability_ari (no phrasing groups available — weak evidence)"
        band, band_source = 0.0, "not applicable — no locator metric available"
    sil_peak = max(valid, key=lambda r: r["silhouette"])
    da_family = deep_aligned["k_estimate"] / max(leaf_ratio, 1)
    lo, hi = expert_range
    estimates = {
        # NAMED AFTER WHAT ACTUALLY LOCATED IT. This key used to be
        # `stability_peak_k`, and both the report label ("稳定性峰 K (主证据)") and
        # fig1's title ("定案 K 取自稳定性峰") inherited that name — while
        # `locator` three lines above says `intent_alignment_ami` and
        # `test_k_is_located_by_intent_alignment_and_only_filtered_by_stability`
        # enforces it. On live38 the true stability peak was k=8; the delivered
        # k=10 ranks 9th of 14 on stability. The code was right and the label
        # contradicted it in three shipped places.
        "located_k": peak["k"],
        "located_by": locator,
        "deep_aligned_leaf_k": deep_aligned["k_estimate"],
        "deep_aligned_implied_family_k": round(da_family, 1),
        "expert_range": [lo, hi],
        "silhouette_peak_k": sil_peak["k"],
    }

    # The two MEASURED estimators are judged separately from the borrowed prior.
    # Lumping them together loses the distinction that matters most: two
    # independent measurements of this corpus agreeing, while a number carried
    # over from a different corpus disagrees, is a strong result — not a
    # three-way muddle. The prior is a hypothesis about the domain; the
    # estimators are observations of the data, and observations win.
    tolerance = max(0.5 * peak["k"], 5)
    measured_agree = abs(da_family - peak["k"]) <= tolerance
    prior_agrees = lo <= peak["k"] <= hi

    if measured_agree and prior_agrees:
        agreement = "full"
        note = ""
    elif measured_agree:
        agreement = "measured_only"
        note = (
            f"两个实测估计都落在 K={peak['k']} 附近 (定案 {peak['k']}, 过聚类存活推出 "
            f"{da_family:.1f}), 但领域先验期望 {lo}-{hi}。**对本语料的两个独立测量一致, "
            "胜过从别处带来的先验** — 该修的是先验。值得复核: 这里的意图轴是否真的比预期更粗, "
            "因为承载细分的是叶层而非家族层。"
        )
    else:
        agreement = "none"
        note = (
            "两个实测估计彼此不一致。**记录分歧而不取平均** — 平均出来的 K 谁也说服不了。"
            f"定案由「{locator}」给出 (稳定性只负责剔除不可复现的 K, 不负责排序); "
            "家族层按暂定读取, 细分靠叶层。"
        )

    return {
        "estimates": estimates,
        "chosen_family_k": peak["k"],
        "locator": locator,
        "stability_floor": stability_floor,
        "n_rejected_as_unstable": len(valid) - len([r for r in valid if r["stability_ari"] >= stability_floor]),
        # Every K the measurement cannot separate from the winner. When this has
        # more than one entry the honest deliverable is the set, not the winner.
        "tie_set": [{"k": r["k"],
                     "intent_alignment_ami": r.get("intent_alignment_ami"),
                     "stability_ari": r["stability_ari"],
                     "template_fragmentation": r.get("template_fragmentation")}
                    for r in sorted(tie_set, key=lambda r: r["k"])],
        "chosen_by": (
            f"stability >= {stability_floor} rejects irreproducible K; among survivors the "
            f"highest {locator}; ties within {band:.4f} — {band_source} — broken toward the simpler tree"
        ),
        "converged": agreement == "full",
        "agreement": agreement,
        "measured_estimators_agree": bool(measured_agree),
        "prior_agrees": bool(prior_agrees),
        "divergence_note": note,
        "silhouette_disagrees": sil_peak["k"] != peak["k"],
    }


# ==========================================================================
# Phase 6 — hierarchy
# ==========================================================================

def build_hierarchy(
    X: np.ndarray,
    family_k: int,
    *,
    seed: int = SEED_METRIC,
    min_leaf_size: int = 150,
    min_leaf_fraction: float = 0.003,
    max_leaves: int = 8,
    family_min_size_for_split: int = 300,
    silhouette_sample: int = 6000,
) -> dict[str, Any]:
    """Two levels: one stable global partition, then a locally chosen split per family.

    Each family picks its own leaf count by silhouette *within itself*.  This is
    the TaxoGen "local granularity" idea, and it is the right shape for the
    problem: a family of near-duplicate lookups genuinely has one leaf, while a
    sprawling family of reading-comprehension queries genuinely has six, and no
    single global K can express both.

    Silhouette gets to decide here, unlike everywhere else in the pipeline —
    the choice is *within* one already-coherent family, where "tighter" and
    "more interpretable" no longer pull apart the way they do across families.
    """
    fam_model = kmeans_fit(X, family_k, seed=seed)
    fam = fam_model.labels_
    min_leaf = max(min_leaf_size, int(min_leaf_fraction * len(X)))

    leaf_labels = np.full(len(X), -1, dtype=np.int64)
    leaf_family: list[int] = []
    leaf_local_k: dict[int, int] = {}
    # Why each family split the way it did — silhouette's choice, the stability
    # it would have cost, and the no-split test. Previously invisible.
    local_k_detail: dict[int, dict[str, Any]] = {}
    next_leaf = 0

    for f in range(family_k):
        mask = fam == f
        Xf = X[mask]
        n = int(mask.sum())
        if n < max(family_min_size_for_split, 2 * min_leaf):
            leaf_labels[mask] = next_leaf
            leaf_family.append(f)
            leaf_local_k[f] = 1
            local_k_detail[f] = {"k": 1, "rejected_because":
                                 f"family of {n} rows is below the split floor"}
            next_leaf += 1
            continue

        verdict = choose_local_k(Xf, max_k=max_leaves, min_size=min_leaf, seed=seed,
                                 silhouette_sample=silhouette_sample)
        best_k = verdict["k"]
        local_k_detail[f] = verdict

        if best_k == 1:
            leaf_labels[mask] = next_leaf
            leaf_family.append(f)
            leaf_local_k[f] = 1
            next_leaf += 1
        else:
            lab = kmeans_labels(Xf, best_k, seed=seed)
            for j in range(best_k):
                leaf_labels[np.where(mask)[0][lab == j]] = next_leaf
                leaf_family.append(f)
                next_leaf += 1
            leaf_local_k[f] = best_k

    centroids = _centroids(X, leaf_labels, next_leaf)
    n_sil_overruled = sum(1 for v in local_k_detail.values() if v.get("silhouette_disagrees"))
    n_no_split = sum(1 for v in local_k_detail.values() if v.get("k") == 1)
    return {
        "local_k": {
            "detail": {int(k): v for k, v in local_k_detail.items()},
            "n_families_not_split": n_no_split,
            "n_silhouette_overruled": n_sil_overruled,
            "note": ("leaf count inside each family is chosen by silhouette, but only "
                     "after the split beats a structureless reference, and silhouette is "
                     "overruled when a negligible lead costs real reproducibility"),
        },
        "family_labels": fam,
        "family_centroids": normalize(fam_model.cluster_centers_),
        "leaf_labels": leaf_labels,
        "leaf_family": np.array(leaf_family, dtype=np.int64),
        "leaf_centroids": centroids,
        "n_families": int(family_k),
        "n_leaves": int(next_leaf),
        "leaves_per_family": leaf_local_k,
        "min_leaf_size_applied": int(min_leaf),
    }


def _centroids(X: np.ndarray, labels: np.ndarray, n: int) -> np.ndarray:
    cents = np.zeros((n, X.shape[1]), dtype=np.float32)
    for c in range(n):
        m = labels == c
        if m.any():
            cents[c] = X[m].mean(0)
    return normalize(cents)


def refine(
    X: np.ndarray,
    leaf_labels: np.ndarray,
    leaf_family: np.ndarray,
    *,
    rounds: int = 5,
    merge_cos: float = 0.92,
    move_tolerance: float = 0.005,
    negative_silhouette_split: float = 0.30,
    min_leaf_size: int = 150,
    seed: int = SEED_METRIC,
) -> dict[str, Any]:
    """Merge near-duplicate leaves, split incoherent ones, reassign, repeat.

    Convergence is defined by movement, not by a fixed iteration count: once a
    round moves fewer than ``move_tolerance`` of rows, the partition has stopped
    changing in any way a reader would notice, and further rounds only burn time.
    """
    labels = leaf_labels.copy()
    fam = leaf_family.copy()
    history: list[dict[str, Any]] = []

    for r in range(rounds):
        n_leaves = int(labels.max()) + 1
        cents = _centroids(X, labels, n_leaves)

        # --- merge -------------------------------------------------------
        sim = cents @ cents.T
        np.fill_diagonal(sim, -1)
        merge_map: dict[int, int] = {}
        for i in range(n_leaves):
            for j in range(i + 1, n_leaves):
                if sim[i, j] > merge_cos and j not in merge_map and i not in merge_map:
                    merge_map[j] = i
        if merge_map:
            labels = np.array([merge_map.get(int(l), int(l)) for l in labels])
            labels, fam = _compact(labels, fam)

        # --- split -------------------------------------------------------
        # A leaf created by this round's merge is exempt from splitting. Without
        # the exemption, merge and split fight: the merge joins two leaves, the
        # negative-silhouette probe immediately sees the seam and splits them
        # back, and the loop oscillates forever at one merge and one split per
        # round while movement asymptotes above the tolerance.
        n_leaves = int(labels.max()) + 1
        just_merged = set(merge_map.values())
        splits = 0
        for c in range(n_leaves):
            if c in just_merged:
                continue
            m = labels == c
            if m.sum() < 2 * min_leaf_size:
                continue
            sub = X[m]
            neg = _negative_silhouette_fraction(sub, seed=seed)
            if neg > negative_silhouette_split:
                lab2 = kmeans_labels(sub, 2, seed=seed)
                new_id = int(labels.max()) + 1
                idx = np.where(m)[0]
                labels[idx[lab2 == 1]] = new_id
                fam = np.append(fam, fam[c])
                splits += 1

        # --- reassign ----------------------------------------------------
        n_leaves = int(labels.max()) + 1
        cents = _centroids(X, labels, n_leaves)
        new_labels = np.asarray((X @ cents.T).argmax(1))
        moved = float((new_labels != labels).mean())
        labels = new_labels
        labels, fam = _compact(labels, fam)

        history.append({
            "round": r + 1,
            "merges": len(merge_map),
            "splits": splits,
            "moved_fraction": round(moved, 5),
            "n_leaves": int(labels.max()) + 1,
            "silhouette": cosine_silhouette(X, labels),
        })
        if moved < move_tolerance:
            break

    n_leaves = int(labels.max()) + 1
    return {
        "leaf_labels": labels,
        "leaf_family": fam[:n_leaves],
        "leaf_centroids": _centroids(X, labels, n_leaves),
        "n_leaves": n_leaves,
        "history": history,
        "converged": bool(history and history[-1]["moved_fraction"] < move_tolerance),
    }


def _compact(labels: np.ndarray, fam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Renumber leaves to 0..n-1 after merges left holes, keeping family links aligned."""
    uniq = np.unique(labels)
    remap = {int(u): i for i, u in enumerate(uniq)}
    new_labels = np.array([remap[int(l)] for l in labels], dtype=np.int64)
    new_fam = np.array([fam[int(u)] if int(u) < len(fam) else 0 for u in uniq], dtype=np.int64)
    return new_labels, new_fam


def _negative_silhouette_fraction(X: np.ndarray, *, sample: int = 2000, seed: int = SEED_METRIC) -> float:
    """Share of members closer to another cluster's centre than their own.

    A cheap two-way probe: if splitting a leaf in two leaves a large minority
    sitting on the wrong side, the leaf was carrying two things.
    """
    if len(X) < 20:
        return 0.0
    idx = deterministic_subsample(len(X), min(sample, len(X)), seed)
    Xs = X[idx]
    lab = kmeans_labels(Xs, 2, seed=seed)
    c = normalize(np.vstack([Xs[lab == 0].mean(0), Xs[lab == 1].mean(0)]))
    sims = Xs @ c.T
    own = sims[np.arange(len(Xs)), lab]
    other = sims[np.arange(len(Xs)), 1 - lab]
    return float((own - other < 0.02).mean())


def heldout_reproduction(
    X: np.ndarray, labels: np.ndarray, *, fraction: float = 0.2, seed: int = SEED_METRIC
) -> dict[str, Any]:
    """Rebuild centroids from 80% of the data and see if the other 20% lands the same way.

    This is the final structural check before naming.  If a partition only exists
    when it can see every row, it is a description of this sample rather than of
    the phenomenon, and naming it would be naming noise.
    """
    n = len(X)
    idx = rng(seed).permutation(n)
    cut = int(n * (1 - fraction))
    train, test = idx[:cut], idx[cut:]
    k = int(labels.max()) + 1
    cents = _centroids(X[train], labels[train], k)
    pred = np.asarray((X[test] @ cents.T).argmax(1))
    agree = float((pred == labels[test]).mean())
    return {
        "agreement": round(agree, 4),
        "n_test": int(len(test)),
        "train_fraction": round(1 - fraction, 2),
    }


def margins(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """top1 − top2 centroid cosine, per row.  Drives the Phase 10 routing split."""
    sims = X @ centroids.T
    part = np.partition(sims, -2, axis=1)
    return part[:, -1] - part[:, -2]


# ==========================================================================
# Local granularity — one policy, used by every layer that subdivides a group
# ==========================================================================

def _shuffled_reference(X: np.ndarray, rs: Any) -> np.ndarray:
    """`X` with each column independently permuted: same marginals, no structure."""
    Z = np.array(X, dtype=np.float32, copy=True)
    for j in range(Z.shape[1]):
        rs.shuffle(Z[:, j])
    return normalize(Z)


def choose_local_k(
    X: np.ndarray,
    *,
    max_k: int,
    min_size: int,
    seed: int = SEED_METRIC,
    silhouette_sample: int = 6000,
    stability_seeds: Sequence[int] = (0, 1),
    null_margin: float = 0.02,
    stability_floor: float = 0.55,
    #: A silhouette difference at or under this is treated as noise, not signal.
    sil_noise: float = 0.02,
    #: ...and is overruled only by a stability gain of at least this much.
    stability_gain: float = 0.15,
    seed_metric: int = SEED_METRIC,
) -> dict[str, Any]:
    """How many sub-clusters a single group should be split into, if any.

    Two layers ask this question — the bottom-up leaf layer inside a family, and
    the top-down sub-intent layer inside an L1 class — and until now each had its
    own copy of the same loop: ``argmax over k of cosine silhouette``, nothing
    else. That loop has two defects that only show up at this scale.

    **It cannot say "do not split."** ``best_score`` starts at -1.0, which is
    below every attainable silhouette, so the first admissible k wins by default
    and the ``k == 1`` branch is unreachable. Every group large enough to divide
    therefore gets divided, whether or not it contains any structure. Silhouette
    is undefined at k=1, so no amount of it can supply the missing test — the
    absence of a no-split test is structural, not an oversight. Here the split
    must first beat a **random relabelling of the same group at the same k**,
    which is what silhouette looks like when there is nothing to find.

    **It lets a rounding-error gain overrule a collapse in reproducibility.** On
    the reference corpus one sub-intent took k=6 (silhouette 0.0749, replay ARI
    0.533) over k=2 (silhouette 0.0696, ARI 0.973) — buying 0.005 of silhouette
    with 0.44 of stability. So candidates must clear a stability floor, and among
    those that do, a silhouette lead inside the noise band does not outrank a
    materially more reproducible split.

    Silhouette keeps a real vote here, and that is deliberate: within one fixed
    representation every candidate encodes phrasing identically, so its bias is a
    constant offset and its *variation* carries information. What it does not get
    is the casting vote when the alternatives are indistinguishable to it.
    """
    n = len(X)
    out: dict[str, Any] = {"k": 1, "candidates": [], "rejected_because": ""}
    if n < 2 * min_size or max_k < 2:
        out["rejected_because"] = f"group of {n} cannot yield two sub-groups of {min_size}"
        return out

    rng_null = rng(seed_metric)
    cands: list[dict[str, Any]] = []
    for k in range(2, max_k + 1):
        if n / k < min_size:
            break
        lab = kmeans_labels(X, k, seed=seed)
        sil = cosine_silhouette(X, lab, sample=min(silhouette_sample, n), seed=seed_metric)
        if np.isnan(sil):
            continue
        # What silhouette looks like at this k when there is nothing to find.
        # The null must be *the same clustering algorithm on structureless data*,
        # not random labels on the real data: k-means optimises compactness and so
        # beats a random relabelling even on isotropic noise, which made an earlier
        # version of this test pass everything. Shuffling each column independently
        # destroys the joint structure while preserving every marginal — the
        # reference-distribution idea behind the gap statistic.
        null = float(np.mean([
            cosine_silhouette(Xn, kmeans_labels(Xn, k, seed=seed),
                              sample=min(silhouette_sample, n), seed=seed_metric)
            for Xn in (_shuffled_reference(X, rng_null) for _ in range(2))
        ]))
        stab = replay_stability(X, k, seeds=tuple(stability_seeds),
                                sample=min(silhouette_sample, n))
        cands.append({"k": k, "silhouette": round(float(sil), 4),
                      "silhouette_null": round(null, 4),
                      "lift_over_null": round(float(sil) - null, 4),
                      "stability_ari": round(float(stab), 4)})
    out["candidates"] = cands
    if not cands:
        out["rejected_because"] = "no admissible k produced a finite silhouette"
        return out

    # 1. The split must beat chance on this group. Silhouette cannot test this.
    real = [c for c in cands if c["lift_over_null"] > null_margin]
    if not real:
        best = max(cands, key=lambda c: c["lift_over_null"])
        out["rejected_because"] = (
            f"no k beats a structureless reference by more than {null_margin}: best lift "
            f"{best['lift_over_null']} at k={best['k']} — this group has no internal structure"
        )
        return out

    # 2. Reproducible splits only; fall back if the floor excludes everything.
    stable = [c for c in real if c["stability_ari"] >= stability_floor] or real

    # 3. Silhouette ranks. It is overruled only on an explicitly bad trade: a
    #    negligible silhouette lead bought with a large collapse in
    #    reproducibility. The thresholds name the case this exists to prevent —
    #    k=6 at silhouette 0.0749 / ARI 0.533 beating k=2 at 0.0696 / 0.973.
    top = max(stable, key=lambda c: c["silhouette"])
    pick = top
    for c in stable:
        d_sil = top["silhouette"] - c["silhouette"]
        d_stab = c["stability_ari"] - pick["stability_ari"]
        if d_sil <= sil_noise and d_stab >= stability_gain:
            pick = c

    out["k"] = pick["k"]
    out["chosen"] = pick
    out["silhouette_would_have_chosen"] = top["k"]
    out["silhouette_disagrees"] = top["k"] != pick["k"]
    out["chosen_by"] = (
        f"beats a structureless reference by > {null_margin}; stability >= {stability_floor}; "
        f"then highest silhouette, overruled only when a lead <= {sil_noise} costs "
        f">= {stability_gain} of replay stability"
    )
    return out


# ==========================================================================
# Selection under measurement noise
# ==========================================================================

def stability_with_error(
    X: np.ndarray,
    k: int,
    *,
    n_pairs: int = 4,
    sample: int = 8000,
    algorithm: str = "kmeans",
    fast: bool = False,
) -> dict[str, float]:
    """Replay stability as an estimate **with a standard error**, not a point.

    The pipeline used to read a single seed pair as if it were the truth. On this
    corpus the seed-to-seed spread of that number is about 0.08 ARI, while the
    differences it was being asked to resolve between adjacent K are about 0.05 —
    so more than half the "decisions" were reading noise. A rule that cannot see
    its own error bar will always produce a confident answer, and confidently
    reordering candidates that are statistically tied is worse than admitting the
    tie, because it hides the tie from the reader.

    Repeating the measurement is also the cheapest possible improvement: it costs
    linear compute and it is the axis with real signal, whereas widening the
    search grid costs multiplicatively and buys candidates that cannot be told
    apart anyway.
    """
    vals: list[float] = []
    for i in range(max(1, n_pairs)):
        vals.append(replay_stability(X, k, seeds=(2 * i, 2 * i + 1), sample=sample,
                                     algorithm=algorithm, fast=fast))
    arr = np.asarray(vals, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return {"mean": float("nan"), "sd": float("nan"), "n_pairs": 0}
    sd = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return {
        "mean": round(float(arr.mean()), 4),
        "sd": round(sd, 4),
        "sem": round(sd / np.sqrt(arr.size), 4) if arr.size > 1 else 0.0,
        "n_pairs": int(arr.size),
        "values": [round(float(v), 4) for v in arr],
    }


def tie_aware_best(
    rows: Sequence[dict[str, Any]],
    *,
    key: str,
    error_key: str | None = None,
    higher_is_better: bool = True,
    absolute_floor: float = 0.0,
    tiebreak: Callable[[dict[str, Any]], Any] | None = None,
) -> dict[str, Any]:
    """Pick a winner, and name everyone who is statistically indistinguishable from it.

    Returns the winner plus the whole **tie set**. That set is not a diagnostic
    afterthought — it is the honest answer whenever the leader's margin is inside
    the measurement error, and it is what makes "here are the two or three
    clusterings that are equally defensible" a deliverable rather than an excuse.

    The tie band is the leader's own standard error when the caller supplies one,
    so the threshold is measured rather than assumed.
    """
    usable = [r for r in rows if r.get(key) is not None and not _isnan(r.get(key))]
    if not usable:
        return {"winner": None, "tied": [], "band": 0.0, "note": "no candidate had a finite score"}

    lead = max(usable, key=lambda r: r[key]) if higher_is_better else min(usable, key=lambda r: r[key])
    band = float(lead.get(error_key) or 0.0) if error_key else 0.0
    band = max(band, absolute_floor)

    if higher_is_better:
        tied = [r for r in usable if lead[key] - r[key] <= band]
    else:
        tied = [r for r in usable if r[key] - lead[key] <= band]

    winner = min(tied, key=tiebreak) if tiebreak else lead
    return {
        "winner": winner,
        "tied": tied,
        "band": round(band, 4),
        "n_tied": len(tied),
        "decided_by_tiebreak": len(tied) > 1,
        "note": (
            f"{len(tied)} candidates lie within {band:.4f} of the leader on {key!r} — "
            "the measurement cannot separate them"
            if len(tied) > 1 else f"the leader on {key!r} is clear of the measurement error"
        ),
    }


def _isnan(v: Any) -> bool:
    try:
        return bool(np.isnan(v))
    except (TypeError, ValueError):
        return False


def partition_stability(
    X: np.ndarray,
    labels: np.ndarray,
    *,
    n_splits: int = 3,
    sample: int = 8000,
    seed: int = SEED_METRIC,
) -> dict[str, float]:
    """Does **this particular partition** reproduce from half the data?

    `replay_stability` answers a different question — "if I re-run KMeans at this
    k, do two seeds agree?" — and answering it requires only ``X`` and ``k``. That
    made it the wrong instrument for the uniform panel, which attaches a stability
    number to *delivered* partitions: leaves after refinement, families after
    governance merges. Those are not KMeans output. Re-running KMeans and
    reporting the result as their stability describes a partition nobody shipped.

    The delivered partitions are defined by their centroids — the codebase's own
    rule is that belonging to a group means being nearest to its centroid — so the
    honest test is whether that geometry survives seeing only half the data:
    derive each group's centroid from one half, from the other half independently,
    then assign a common held-out sample under both and compare. It needs no
    re-clustering, so it costs a few centroid computations rather than a second
    Phase 6, and unlike the old number it is a function of the partition.

    Repeated over several disjoint splits so it arrives with an error bar, because
    a stability figure without one invites exactly the noise-reading this pipeline
    has already been caught doing.
    """
    labels = np.asarray(labels)
    ok = labels >= 0
    if ok.sum() < 50 or len(np.unique(labels[ok])) < 2:
        return {"mean": float("nan"), "sd": float("nan"), "n_splits": 0}

    idx_all = np.flatnonzero(ok)
    common = idx_all if len(idx_all) <= sample else deterministic_subsample(
        len(idx_all), sample, seed)
    common_idx = idx_all[common] if len(idx_all) > sample else idx_all
    Xc = X[common_idx]
    groups = np.unique(labels[ok])

    def centroids_from(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cents, keep = [], []
        for g in groups:
            m = rows[labels[rows] == g]
            if m.size:
                cents.append(X[m].mean(axis=0))
                keep.append(g)
        return (normalize(np.vstack(cents)) if cents else np.empty((0, X.shape[1]))), np.asarray(keep)

    vals: list[float] = []
    for s in range(max(1, n_splits)):
        perm = rng(seed + s).permutation(idx_all)
        half = len(perm) // 2
        ca, ga = centroids_from(perm[:half])
        cb, gb = centroids_from(perm[half : 2 * half])
        if ca.size == 0 or cb.size == 0:
            continue
        la = ga[np.argmax(Xc @ ca.T, axis=1)]
        lb = gb[np.argmax(Xc @ cb.T, axis=1)]
        vals.append(float(adjusted_rand_score(la, lb)))

    if not vals:
        return {"mean": float("nan"), "sd": float("nan"), "n_splits": 0}
    arr = np.asarray(vals)
    return {
        "mean": round(float(arr.mean()), 4),
        "sd": round(float(arr.std(ddof=1)), 4) if arr.size > 1 else 0.0,
        "n_splits": int(arr.size),
        "values": [round(float(v), 4) for v in arr],
    }


def leaves_per_family(labels: np.ndarray, family_lut: np.ndarray) -> dict[str, int]:
    """Count leaves per family FROM THE LABELS, never from the lookup table.

    `family_lut` keeps a row for every leaf id the tree ever had, including the
    ones refinement merged away. Grouping over it counts leaves that no longer
    exist: live39 shipped `n_leaves = 29` beside a breakdown summing to 32, with
    families 2, 3 and 8 each credited a leaf that had already been merged. The
    two numbers sat in the same artifact with nothing marking that one was taken
    before refinement and the other after.

    Counting the distinct labels that actually appear makes the breakdown and the
    total the same measurement, so they cannot drift apart again.
    """
    present = sorted({int(v) for v in np.unique(labels)})
    lut = np.asarray(family_lut)
    out: dict[str, int] = {}
    for lid in present:
        f = int(lut[lid]) if lid < len(lut) else -1
        out[str(f)] = out.get(str(f), 0) + 1
    return out
