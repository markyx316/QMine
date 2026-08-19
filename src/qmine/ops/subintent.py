"""Phase 2e — the L2 sub-intent layer, and the geometric audit that precedes it.

The audit is the interesting half.  Before subdividing an intent it is worth
asking whether the representation can *see* that intent at all, and the answer
varies enormously between classes of the same taxonomy.  In the source project
one class was 0.96 separable by nearest-neighbour agreement and another was
0.36 — and the second one was never going to be learned by an embedding model no
matter how much gold you threw at it.  That class needs the rule layer, and
knowing so before training saves a cycle of blaming the data.

So this module reports, per L1 class:

* **cohesion** — mean cosine of members to their own class centroid;
* **kNN agreement** — how often a member's nearest neighbours share its label,
  which is the practical question "is this class a region, or is it scattered?";
* **nearest rival** — the class whose centroid sits closest, i.e. what it will
  most often be confused with;
* **verdict** — learnable from geometry, or rule-dependent.

Only then does it subdivide, and only classes big and coherent enough to have
internal structure worth naming.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.preprocessing import normalize

from ..determinism import SEED_METRIC, deterministic_subsample


def geometric_audit(
    X: np.ndarray,
    labels: Sequence[str],
    *,
    k: int = 5,
    sample: int = 8000,
    seed: int = SEED_METRIC,
    rule_dependent_threshold: float = 0.5,
) -> dict[str, Any]:
    """Per-class geometry: can the representation see this intent?

    ``rule_dependent_threshold`` is the kNN-agreement level below which we stop
    expecting the embedding to carry the class.  It is a reporting threshold,
    not a filter — the class stays in the taxonomy either way, because the
    taxonomy answers to users rather than to geometry.  What changes is where
    the class gets its accuracy from.
    """
    y = np.asarray(list(labels))
    idx = deterministic_subsample(len(X), min(sample, len(X)), seed)
    Xs, ys = X[idx], y[idx]

    classes = sorted(set(ys.tolist()))
    if len(classes) < 2:
        return {"classes": [], "note": "fewer than two classes present; audit skipped"}

    cents = normalize(np.vstack([Xs[ys == c].mean(0) for c in classes]))
    sims = Xs @ cents.T
    # kNN agreement on the sub-sample; cosine on unit vectors is a dot product
    nn = (Xs @ Xs.T)
    np.fill_diagonal(nn, -np.inf)
    top = np.argpartition(-nn, kth=min(k, len(Xs) - 1), axis=1)[:, :k]

    rows: list[dict[str, Any]] = []
    for ci, c in enumerate(classes):
        m = ys == c
        n = int(m.sum())
        if n < 3:
            continue
        cohesion = float(sims[m, ci].mean())
        agree = float(np.mean([(ys[top[i]] == ys[i]).mean() for i in np.where(m)[0]]))
        rival_sims = cents[ci] @ cents.T
        rival_sims[ci] = -np.inf
        rival = classes[int(rival_sims.argmax())]
        rows.append({
            "class": c,
            "n": n,
            "share": round(n / len(ys), 4),
            "cohesion": round(cohesion, 4),
            "knn_agreement": round(agree, 4),
            "nearest_rival": rival,
            "rival_centroid_cosine": round(float(rival_sims.max()), 4),
            "verdict": (
                "geometry-visible" if agree >= rule_dependent_threshold else "rule-dependent"
            ),
        })

    rows.sort(key=lambda r: r["knn_agreement"])
    rule_dep = [r["class"] for r in rows if r["verdict"] == "rule-dependent"]
    return {
        "classes": rows,
        "k": k,
        "n_evaluated": int(len(idx)),
        "rule_dependent_classes": rule_dep,
        "interpretation": (
            "kNN agreement is the practical test of whether a class occupies a region of "
            "the embedding space. Classes below the threshold are not a data problem and "
            "not a model problem — they are intents whose meaning lives in pragmatics "
            "rather than wording, and they must draw their accuracy from the rule layer. "
            "Adding gold labels for them raises cost without raising accuracy."
        ),
        "rule_dependent_note": (
            f"{len(rule_dep)} class(es) are rule-dependent: {rule_dep}" if rule_dep
            else "every class is visible to the representation"
        ),
    }


def subdivide(
    X: np.ndarray,
    labels: Sequence[str],
    *,
    min_class_size: int = 300,
    min_sub_size: int = 100,
    max_sub: int = 6,
    seed: int = SEED_METRIC,
    silhouette_sample: int = 4000,
) -> dict[str, Any]:
    """Split each L1 into sub-intents, choosing k locally per class.

    Same local-granularity idea as the bottom-up leaf layer: a narrow class has
    one sub-intent and a sprawling one has five, and no single global number
    expresses both. Classes too small or too uniform to subdivide keep a single
    child rather than being forced apart.
    """
    from .cluster import choose_local_k, kmeans_labels

    y = np.asarray(list(labels))
    sub = np.array(["" for _ in range(len(y))], dtype=object)
    detail: dict[str, Any] = {}

    for c in sorted(set(y.tolist())):
        m = y == c
        n = int(m.sum())
        if n < max(min_class_size, 2 * min_sub_size):
            sub[m] = f"{c}__1"
            detail[c] = {"n": n, "k": 1, "reason": "too small to subdivide"}
            continue

        Xc = X[m]
        # The same policy the bottom-up leaf layer uses. It used to be a second
        # copy of an argmax-silhouette loop, which meant the two routes could
        # disagree about what silhouette is permitted to decide — and this copy
        # carried no disclosure at all, so its choices reached `subintents.json`
        # with no record of what they cost.
        verdict = choose_local_k(Xc, max_k=max_sub, min_size=min_sub_size, seed=seed,
                                 silhouette_sample=silhouette_sample)
        best_k = verdict["k"]

        if best_k == 1:
            sub[m] = f"{c}__1"
            detail[c] = {"n": n, "k": 1,
                         "reason": verdict.get("rejected_because") or "no split improved cohesion",
                         "candidates": verdict.get("candidates", [])}
        else:
            lab = kmeans_labels(Xc, best_k, seed=seed)
            for j in range(best_k):
                sub[np.where(m)[0][lab == j]] = f"{c}__{j + 1}"
            detail[c] = {
                "n": n, "k": best_k,
                "silhouette": verdict["chosen"]["silhouette"],
                "stability_ari": verdict["chosen"]["stability_ari"],
                "lift_over_null": verdict["chosen"]["lift_over_null"],
                "silhouette_would_have_chosen": verdict.get("silhouette_would_have_chosen"),
                "silhouette_disagrees": verdict.get("silhouette_disagrees", False),
                "chosen_by": verdict.get("chosen_by", ""),
            }

    return {
        "sub_labels": sub,
        "per_class": detail,
        "n_sub_intents": int(len(set(sub.tolist()))),
    }


def sub_samples(
    queries: Sequence[str], sub_labels: Sequence[str], *, n: int = 12, seed: int = SEED_METRIC
) -> dict[str, list[str]]:
    """Member samples per sub-intent, for the interpreting agent."""
    from ..determinism import rng

    q = np.asarray(list(queries))
    s = np.asarray(list(sub_labels))
    r = rng(seed)
    out: dict[str, list[str]] = {}
    for lab in sorted(set(s.tolist())):
        idx = np.where(s == lab)[0]
        if idx.size == 0:
            continue
        pick = r.choice(idx, size=min(n, idx.size), replace=False)
        out[lab] = [str(x) for x in q[np.sort(pick)]]
    return out
