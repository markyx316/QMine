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
    return {
        "chosen": ranked[0]["algorithm"] if ranked else None,
        "chosen_family": ranked[0]["algorithm"].split("_k")[0] if ranked else None,
        "chosen_by": "stability_ari desc (silhouette breaks ties only)",
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
        row = {
            "k": int(k),
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
) -> dict[str, Any]:
    """Reconcile three independent estimates of the family scale.

    The stability peak is the primary signal.  The DeepAligned survivor count
    estimates the *leaf* scale, so it enters the comparison divided by the
    expected leaves-per-family.  The expert range is a prior from someone who
    knows the vertical.  Agreement is strong evidence; disagreement is not
    resolved by averaging — we take the stability peak and record the dissent,
    because a number nobody can defend is worse than a number with a caveat.
    """
    valid = [r for r in sweep if not np.isnan(r["stability_ari"])]
    peak = max(valid, key=lambda r: r["stability_ari"])
    sil_peak = max(valid, key=lambda r: r["silhouette"])
    da_family = deep_aligned["k_estimate"] / max(leaf_ratio, 1)
    lo, hi = expert_range
    estimates = {
        "stability_peak_k": peak["k"],
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
            f"Both measured estimators land near K={peak['k']} (stability peak {peak['k']}, "
            f"over-clustering survival implies {da_family:.1f}), but the domain prior expected "
            f"{lo}-{hi}. Two independent measurements of THIS corpus agreeing outweighs a prior "
            "carried in from elsewhere — the prior is the thing to revise. Worth checking that "
            "the intent axis really is coarser here than expected, since the leaf layer, not the "
            "family layer, carries fine distinctions."
        )
    else:
        agreement = "none"
        note = (
            "The measured estimators disagree with each other. Taking the stability peak and "
            "recording the disagreement rather than averaging: an averaged K is defensible to "
            "nobody. Treat the family layer as provisional and lean on the leaf layer."
        )

    return {
        "estimates": estimates,
        "chosen_family_k": peak["k"],
        "chosen_by": "stability peak (primary); other estimators used as corroboration only",
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
    next_leaf = 0

    for f in range(family_k):
        mask = fam == f
        Xf = X[mask]
        n = int(mask.sum())
        if n < max(family_min_size_for_split, 2 * min_leaf):
            leaf_labels[mask] = next_leaf
            leaf_family.append(f)
            leaf_local_k[f] = 1
            next_leaf += 1
            continue

        best_k, best_score = 1, -1.0
        for k in range(2, max_leaves + 1):
            if n / k < min_leaf:
                break
            lab = kmeans_labels(Xf, k, seed=seed)
            score = cosine_silhouette(Xf, lab, sample=min(silhouette_sample, n), seed=seed)
            if not np.isnan(score) and score > best_score:
                best_k, best_score = k, score

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
    return {
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
