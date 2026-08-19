"""Phase 9 — the uniform metrics panel.

The rule this module exists to enforce: **two numbers may be compared only if the
same code produced them, on the same sub-sample, under the same seed.**  Quoting
a silhouette from the alpha sweep next to one from last week's battery run is
comparing two different measurements that happen to share a name.

:class:`UniformPanel` makes that structural.  It fixes the sub-sample and seed at
construction, stamps every metric it produces with a ``panel_id`` derived from
that configuration, and the comparison renderer refuses to place two different
``panel_id`` values in one table.  You cannot accidentally mix panels; you have
to go out of your way.

The panel also carries the fairness footnote that the source project learned to
write: template fragmentation is negatively correlated with cluster count — a
partition with fewer families finds it *arithmetically harder* to fragment
anything — so every fragmentation comparison must report cluster counts beside
it, and conclusions must be phrased as the two-condition statement they are.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ..determinism import SEED_METRIC, deterministic_subsample, hash_params
from ..records import METRIC_AUTHORITY, MetricRecord, MetricSet


class UniformPanel:
    """A fixed measurement harness.  Everything measured through it is comparable."""

    def __init__(
        self,
        n_rows: int,
        *,
        subsample: int = 8000,
        seed: int = SEED_METRIC,
        replay_seeds: tuple[int, int] = (0, 1),
        code_version: str = "panel-v1",
    ) -> None:
        self.n_rows = n_rows
        self.subsample = min(subsample, n_rows)
        self.seed = seed
        self.replay_seeds = replay_seeds
        self.code_version = code_version
        self.indices = deterministic_subsample(n_rows, self.subsample, seed)
        self.panel_id = hash_params(
            {
                "code": code_version,
                "n_rows": n_rows,
                "subsample": self.subsample,
                "seed": seed,
                "replay_seeds": list(replay_seeds),
            },
            length=12,
        )
        self._sets: dict[str, MetricSet] = {}

    # -- measurement --------------------------------------------------------
    def measure(
        self,
        subject: str,
        X: np.ndarray,
        labels: np.ndarray,
        *,
        template_masks: dict[str, np.ndarray] | None = None,
        reference_labels: np.ndarray | None = None,
        compute_stability: bool = True,
        heldout: bool = True,
        distill: bool = False,
        centroids: np.ndarray | None = None,
        margin_threshold: float = 0.02,
    ) -> MetricSet:
        """Measure one candidate.  Every metric lands in the same MetricSet."""
        from sklearn.metrics import normalized_mutual_info_score

        from .cluster import (
            cosine_silhouette, heldout_reproduction, margins, partition_stability,
            replay_stability,
        )
        from .templates import template_fragmentation

        ms = MetricSet(subject=subject, panel_id=self.panel_id)
        k = int(len(np.unique(labels[labels >= 0])))
        ms.add(MetricRecord.make("n_clusters", k, n=self.n_rows, seed=self.seed))
        ms.add(
            MetricRecord.make(
                "silhouette",
                cosine_silhouette(X, labels, sample=self.subsample, seed=self.seed),
                n=self.subsample,
                seed=self.seed,
                note="advisory only — maximised by phrasing-tight clusters (Principle 3)",
            )
        )
        if compute_stability:
            # Measure THIS partition, not a fresh KMeans run at the same k. The old
            # call took only (X, k), so the "decisive" number attached to the
            # delivered leaves described a partition nobody shipped — and it was
            # pessimistic by about 0.25 ARI, because refinement and governance
            # improve the tree and re-running KMeans throws that away.
            ps = partition_stability(X, labels, sample=self.subsample, seed=self.seed)
            ms.add(
                MetricRecord.make(
                    "stability_ari", ps["mean"], n=self.subsample,
                    detail={"method": "half-sample centroid replay on this partition",
                            "sd": ps["sd"], "n_splits": ps["n_splits"],
                            "values": ps.get("values", [])},
                )
            )
            # The old quantity, kept under a name that says what it is: a property
            # of the corpus and k, useful for reading the K sweep, useless as a
            # description of a delivered partition.
            ms.add(
                MetricRecord.make(
                    "kmeans_refit_stability", 
                    replay_stability(X, k, seeds=self.replay_seeds, sample=self.subsample),
                    n=self.subsample,
                    detail={"seeds": list(self.replay_seeds),
                            "note": "re-runs KMeans at this k; does not depend on the candidate"},
                )
            )
        if template_masks:
            frag = template_fragmentation(labels, template_masks)
            ms.add(
                MetricRecord.make(
                    "template_fragmentation",
                    frag["mean_fragmentation"],
                    n=self.n_rows,
                    detail={
                        "per_group": frag["per_group"],
                        "n_groups_scored": frag["n_groups_scored"],
                        "n_clusters": k,
                    },
                    note=(
                        f"compare only against partitions of similar size "
                        f"(this one has {k} clusters; fewer clusters fragment less by construction)"
                    ),
                )
            )
        if reference_labels is not None:
            ms.add(
                MetricRecord.make(
                    "nmi_reference",
                    float(normalized_mutual_info_score(reference_labels, labels)),
                    n=self.n_rows,
                    note="alignment with a reference taxonomy — not a correctness score",
                )
            )
            ms.add(MetricRecord.make("purity_reference", _purity(reference_labels, labels), n=self.n_rows))
        if heldout:
            hr = heldout_reproduction(X, labels, seed=self.seed)
            ms.add(MetricRecord.make("heldout_reproduction", hr["agreement"], n=hr["n_test"], detail=hr))
        if centroids is not None:
            m = margins(X, centroids)
            ms.add(
                MetricRecord.make(
                    "ambiguous_rate",
                    float((m < margin_threshold).mean()),
                    n=self.n_rows,
                    detail={"threshold": margin_threshold, "median_margin": round(float(np.median(m)), 4)},
                    note="rows whose top-2 centroids are near-tied; these route to fallback in Phase 10",
                )
            )
        if distill:
            ms.add(
                MetricRecord.make(
                    "distill_accuracy",
                    _distill_accuracy(X, labels, seed=self.seed),
                    n=self.n_rows,
                    note=(
                        "measures how LEARNABLE these clusters are, NOT agreement with "
                        "human judgment (Principle 12)"
                    ),
                )
            )
        self._sets[subject] = ms
        return ms

    def add_external(self, subject: str, name: str, value: float, **kw: Any) -> None:
        """Record a metric computed elsewhere but belonging to this panel."""
        ms = self._sets.setdefault(subject, MetricSet(subject=subject, panel_id=self.panel_id))
        ms.add(MetricRecord.make(name, value, panel_id=self.panel_id, subject=subject, **kw))

    # -- reporting ----------------------------------------------------------
    def sets(self) -> dict[str, MetricSet]:
        return dict(self._sets)

    def comparison_table(self, metrics: Sequence[str] | None = None) -> dict[str, Any]:
        """Render the cross-candidate table, refusing to mix panels."""
        panels = {ms.panel_id for ms in self._sets.values()}
        if len(panels) > 1:
            raise ValueError(
                f"refusing to build a comparison table across {len(panels)} different panels "
                f"({sorted(panels)}). Two numbers are comparable only if the same code measured "
                "them on the same sub-sample under the same seed (Principle 7)."
            )
        names = list(metrics) if metrics else sorted(
            {n for ms in self._sets.values() for n in ms.metrics}
        )
        rows = []
        for subject, ms in self._sets.items():
            row: dict[str, Any] = {"subject": subject}
            for n in names:
                row[n] = ms.get(n)
            rows.append(row)
        return {
            "panel_id": self.panel_id,
            "panel_config": {
                "code_version": self.code_version,
                "subsample": self.subsample,
                "seed": self.seed,
                "replay_seeds": list(self.replay_seeds),
                "n_rows": self.n_rows,
            },
            "metrics": [
                {"name": n, "authority": METRIC_AUTHORITY.get(n, "diagnostic"),
                 "higher_is_better": n not in _LOWER}
                for n in names
            ],
            "rows": rows,
            "footnotes": self.footnotes(),
        }

    def footnotes(self) -> list[str]:
        return [
            "Every number in this table was produced by the same code, on the same "
            f"{self.subsample}-row sub-sample, under seed {self.seed} (panel {self.panel_id}).",
            "Silhouette is ADVISORY. It is maximised by clusters that are tight in "
            "surface form, which is precisely the failure mode — one intent split "
            "into several phrasing-shaped families — that the decisive metrics exist "
            "to detect. It is reported for completeness and given no vote.",
            "Template fragmentation is negatively correlated with cluster count: a "
            "partition with fewer clusters has fewer places to fragment into. Read it "
            "beside n_clusters and phrase conclusions as two-condition statements "
            "(\"finer AND less fragmented\"), never as a bare comparison.",
            "Distillation accuracy measures learnability of the cluster labels, not "
            "agreement with human judgment. Human agreement requires a gold set "
            "(Phase 2b) and adversarial validation (Phase 2d).",
        ]

    def decisive_ranking(self, metric: str) -> list[dict[str, Any]]:
        """Rank candidates by a decisive metric.  Raises on an advisory one."""
        auth = METRIC_AUTHORITY.get(metric, "diagnostic")
        if auth != "decisive":
            raise ValueError(
                f"{metric!r} has authority {auth!r} and cannot decide anything. "
                f"Decisive metrics are: {sorted(k for k, v in METRIC_AUTHORITY.items() if v == 'decisive')}"
            )
        higher = metric not in _LOWER
        rows = [
            {"subject": s, "value": ms.get(metric)}
            for s, ms in self._sets.items()
            if ms.get(metric) is not None and not np.isnan(ms.get(metric))  # type: ignore[arg-type]
        ]
        rows.sort(key=lambda r: r["value"], reverse=higher)
        return rows


_LOWER = {"template_fragmentation", "inertia", "davies_bouldin", "noise_rate", "ambiguous_rate", "ece"}


def _purity(reference: np.ndarray, labels: np.ndarray) -> float:
    """Fraction of rows whose cluster's majority reference label is their own."""
    total = 0
    for c in np.unique(labels):
        m = labels == c
        if not m.any():
            continue
        vals, counts = np.unique(reference[m], return_counts=True)
        total += int(counts.max())
    return round(total / len(labels), 4)


def _distill_accuracy(X: np.ndarray, labels: np.ndarray, *, seed: int = SEED_METRIC, folds: int = 3) -> float:
    """Cross-validated accuracy of a linear model predicting the cluster label.

    Deliberately a *linear* head. Tree ensembles fed raw embedding coordinates
    collapse on this task — they must reconstruct directional similarity from
    axis-aligned splits, and with a large label set the per-class boosting budget
    is spread far too thin. A linear head reads the geometry directly.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    keep = np.isin(labels, [c for c in np.unique(labels) if (labels == c).sum() >= folds])
    if keep.sum() < folds * 2 or len(np.unique(labels[keep])) < 2:
        return float("nan")
    idx = deterministic_subsample(int(keep.sum()), min(12000, int(keep.sum())), seed)
    Xs, ys = X[keep][idx], labels[keep][idx]
    keep2 = np.isin(ys, [c for c in np.unique(ys) if (ys == c).sum() >= folds])
    Xs, ys = Xs[keep2], ys[keep2]
    if len(np.unique(ys)) < 2:
        return float("nan")
    cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
    return round(
        float(cross_val_score(LogisticRegression(max_iter=1500, C=4), Xs, ys, cv=cv).mean()), 4
    )
