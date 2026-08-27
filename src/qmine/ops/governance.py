"""Phase 8 — executing the audit's prescriptions.

The failure this module prevents is specific and embarrassing: an audit finds
four twin families and two risk leaves, the report includes a tidy
"recommendations" section, the delivered CSV contains none of it, and the first
question from the reviewer is "so were the issues actually fixed?"

So a prescription here is a state machine, not a sentence.  It is ``proposed``
until something executes it, ``executed`` only with an evidence pointer naming
the artifact and column that changed, or ``declined`` with a stated reason.
:func:`assert_all_settled` fails the run on anything still ``proposed``, which
means the report physically cannot ship ahead of the data.

Merges are applied as a **lookup-table remap**: the leaf assignments and
centroids do not move, only the leaf→family mapping changes.  No re-clustering,
no re-encoding, and — importantly — the pre-merge column stays in the delivered
table so every merge is reversible and auditable.
"""

from __future__ import annotations

import time
from typing import Any, Sequence

import numpy as np

from ..records import Prescription


class GovernanceError(RuntimeError):
    """Raised when a prescription would ship unexecuted."""


def apply_merges(
    leaf_family: np.ndarray, merge_map: dict[int, int]
) -> tuple[np.ndarray, dict[str, Any]]:
    """Remap families through a lookup table, following chains to a root.

    Chain-following matters: an auditor may prescribe 10→6 and 6→0 in the same
    pass, and applying those naively in dict order leaves some rows at 6.
    """
    def root(f: int, _seen: set[int] | None = None) -> int:
        seen = _seen or set()
        while f in merge_map and f not in seen:
            seen.add(f)
            f = merge_map[f]
        return f

    resolved = {int(k): root(int(k)) for k in merge_map}
    merged = np.array([resolved.get(int(f), int(f)) for f in leaf_family], dtype=np.int64)

    # renumber to a dense 0..n-1 so downstream code never sees gaps
    uniq = np.unique(merged)
    dense = {int(u): i for i, u in enumerate(uniq)}
    final = np.array([dense[int(f)] for f in merged], dtype=np.int64)
    return final, {
        "merge_map_raw": {str(k): int(v) for k, v in merge_map.items()},
        "merge_map_resolved": {str(k): int(v) for k, v in resolved.items()},
        "renumbering": {str(k): int(v) for k, v in dense.items()},
        "n_families_before": int(len(np.unique(leaf_family))),
        "n_families_after": int(len(uniq)),
    }


def isolate_leaves(
    leaf_family: np.ndarray, leaf_ids: Sequence[int]
) -> tuple[np.ndarray, dict[str, Any]]:
    """Pull leaves out into families of their own (Principle 10).

    Used for risk content, which must never sit inside a topically similar
    normal family regardless of how close its centroid is.
    """
    out = leaf_family.copy()
    next_family = int(out.max()) + 1 if len(out) else 0
    moved = {}
    for lid in leaf_ids:
        if 0 <= int(lid) < len(out):
            moved[int(lid)] = {"from": int(out[int(lid)]), "to": next_family}
            out[int(lid)] = next_family
            next_family += 1
    return out, {"isolated": moved, "n_families_after": int(len(np.unique(out)))}


def split_leaves(
    X: np.ndarray,
    leaf_labels: np.ndarray,
    leaf_family: np.ndarray,
    leaf_ids: Sequence[int],
    *,
    min_size: int = 60,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Split leaves the audit judged to be carrying two things.

    Unlike merges and isolations, this genuinely re-partitions: a new leaf is
    appended and member assignments change. It is recorded as such rather than
    described as a lookup remap, because claiming otherwise in a report would be
    false in a way that matters — the centroid matrix shipped to production is
    different afterwards.

    **Every split is measured, and the measurement does not veto.** On live40 this
    door made 9 of the 25 delivered leaves — 36% of the layer a reader sees —
    while `choose_local_k` applied a null test and a stability floor to every other
    leaf in the tree and this applied a `min_size` guard and nothing else. That
    asymmetry was invisible: the two facts lived in different artifacts.

    Measuring but not vetoing is deliberate, and the direction matters. When the 9
    live40 splits were replayed through `choose_local_k`'s own tests, **all 9
    passed** — the audit was correctly compensating for an under-split caused by
    ranking local k on raw silhouette (see `_rank_local_candidates`). Had the
    measurement been a veto built on the same biased geometry, it would have
    rejected the corrections to its own bias. A split can also be semantically
    right and geometrically unsupported — two intents that share phrasing — which
    is exactly the case the audit exists to catch and the geometry cannot see.

    So the number is recorded beside the split and a reader can weigh it. A leaf
    whose split fails the null is not blocked; it is *disclosed*.
    """
    from sklearn.cluster import KMeans

    labels = leaf_labels.copy()
    fam = list(leaf_family)
    done: dict[str, Any] = {}
    for lid in sorted({int(l) for l in leaf_ids}):
        idx = np.where(labels == lid)[0]
        if idx.size < 2 * min_size:
            done[str(lid)] = {"split": False, "why": f"only {idx.size} members; needs {2 * min_size}"}
            continue
        sub = KMeans(n_clusters=2, random_state=seed, n_init=10).fit_predict(X[idx])
        new_id = int(labels.max()) + 1
        labels[idx[sub == 1]] = new_id
        fam.append(int(leaf_family[lid]) if lid < len(leaf_family) else 0)
        done[str(lid)] = {
            "split": True, "new_leaf": new_id,
            "sizes": [int((sub == 0).sum()), int((sub == 1).sum())],
            **_measure_split(X[idx], sub, seed=seed),
        }
    return labels, np.array(fam, dtype=np.int64), done


def _measure_split(Xs: np.ndarray, sub: np.ndarray, *, seed: int = 0) -> dict[str, Any]:
    """The same two tests `choose_local_k` applies, run on an agent's split.

    Advisory by construction — see `split_leaves`. Failure to measure is recorded
    as `null` rather than as a pass, because a missing measurement that reads as
    approval is the failure mode this whole record exists to prevent.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score

    from .cluster import cosine_silhouette

    out: dict[str, Any] = {"measured_by": "choose_local_k's null + stability tests (advisory)"}
    try:
        real = float(cosine_silhouette(Xs, sub, sample=min(8000, len(Xs))))
        rng = np.random.default_rng(seed)
        Z = Xs.copy()
        for j in range(Z.shape[1]):
            Z[:, j] = Z[rng.permutation(len(Z)), j]
        null_lab = KMeans(n_clusters=2, random_state=seed, n_init=4).fit_predict(Z)
        null = float(cosine_silhouette(Z, null_lab, sample=min(8000, len(Z))))
        a = KMeans(n_clusters=2, random_state=seed, n_init=4).fit_predict(Xs)
        b = KMeans(n_clusters=2, random_state=seed + 1, n_init=4).fit_predict(Xs)
        out.update({
            "silhouette": round(real, 4),
            "silhouette_null": round(null, 4),
            "lift_over_null": round(real - null, 4),
            "stability_ari": round(float(adjusted_rand_score(a, b)), 4),
        })
        out["geometry_supports_the_split"] = bool(
            out["lift_over_null"] > 0.02 and out["stability_ari"] >= 0.55)
    except Exception as exc:                                     # noqa: BLE001
        out["measurement_failed"] = f"{type(exc).__name__}: {exc}"
        out["geometry_supports_the_split"] = None
    return out


def execute_prescriptions(
    prescriptions: Sequence[Prescription],
    leaf_family: np.ndarray,
    *,
    metrics_before: dict[str, float] | None = None,
    recompute: Any = None,
    X: np.ndarray | None = None,
    leaf_labels: np.ndarray | None = None,
) -> tuple[np.ndarray, list[Prescription], dict[str, Any]]:
    """Run every prescription against the data and stamp each with its evidence.

    ``recompute`` is an optional callable taking the new ``leaf_family`` and
    returning a metrics dict.  When supplied, each executed prescription records
    the metric delta it caused — which is the difference between "we merged some
    families" and "merging these four families moved mean fragmentation from
    1.76 to 1.52 while family count fell from 20 to 17".
    """
    merge_map: dict[int, int] = {}
    isolate: list[int] = []
    to_split: list[int] = []
    relabel: dict[int, str] = {}
    handled: list[Prescription] = []
    can_split = X is not None and leaf_labels is not None

    #: The family ids that actually exist. A `merge_families` target outside this
    #: set is not a merge that happens to do nothing — it is a target in the
    #: WRONG NAMESPACE, and executing it silently is how a prescription gets
    #: recorded as done while changing nothing.
    existing_families = {int(f) for f in np.asarray(leaf_family).tolist()}
    unresolvable: dict[str, list[int]] = {}

    for p in prescriptions:
        if p.kind == "merge_families" and len(p.targets) >= 2:
            # A PRESCRIPTION THAT CHANGED NOTHING MUST NOT READ AS EXECUTED.
            #
            # live40's P011 targeted [10, 11, 15] with LEAF names ("拼音查询",
            # "汉字读音查询", "词语拼音查询") while `merge_families` runs in the
            # FAMILY namespace, where only 0..6 existed. Families 10, 11 and 15
            # were never there, so those map entries were no-ops — and the
            # prescription was still stamped `executed`, the report's §6 table
            # said so, and the three duplicate `pinyin_query` leaves it was meant
            # to collapse are all still in the delivered partition.
            #
            # Same leaf-id-vs-family-id confusion that once made every family
            # heading in the report wrong. Principle 6 wants a matching executed
            # change or an explicit declined reason; an unresolvable target has
            # neither.
            resolvable = [int(t) for t in p.targets if int(t) in existing_families]
            missing = [int(t) for t in p.targets if int(t) not in existing_families]
            if len(resolvable) < 2:
                p.status = "declined"
                p.decline_reason = (
                    f"targets {sorted(int(t) for t in p.targets)} do not name two existing "
                    f"families (present: {sorted(existing_families)}). These look like LEAF "
                    "ids — `merge_families` operates on the family namespace, and merging a "
                    "family that does not exist changes nothing while reading as executed."
                )
                handled.append(p)
                continue
            if missing:
                unresolvable[p.id] = missing
            keep = int(min(resolvable))
            for t in resolvable:
                if int(t) != keep:
                    merge_map[int(t)] = keep
            handled.append(p)
        elif p.kind in ("isolate_leaf", "flag_risk") and p.targets:
            isolate.extend(int(t) for t in p.targets)
            handled.append(p)
        elif p.kind == "split_leaf" and p.targets:
            if can_split:
                to_split.extend(int(t) for t in p.targets)
                handled.append(p)
            else:
                # Declined with a reason, never silently dropped — an audit
                # finding that vanishes is the exact failure Principle 6 names.
                p.status = "declined"
                p.decline_reason = (
                    "split requires the embedding and leaf assignments, which were not "
                    "supplied to this execution; re-run Phase 8 with them to apply it"
                )
                handled.append(p)
        elif p.kind == "relabel" and p.targets:
            if p.target_names and len(p.target_names) == len(p.targets):
                relabel.update({int(t): str(n) for t, n in zip(p.targets, p.target_names)})
                handled.append(p)
            else:
                p.status = "declined"
                p.decline_reason = (
                    "relabel carried no replacement name for each target; a rename with "
                    "nothing to rename to cannot be applied"
                )
                handled.append(p)
        elif p.kind == "keep_as_is":
            p.status = "declined"
            p.decline_reason = p.rationale or "deliberate: audit judged the split to have a real basis"
            handled.append(p)
        else:
            # Exhaustive by construction. An unrecognised kind — or a recognised
            # one with no usable targets — is DECLINED with a reason, never left
            # `proposed`. Left proposed it halts the run at the Phase 8 gate,
            # which is correct behaviour for a real unapplied finding and pure
            # noise for a malformed one; and dropping it silently would be the
            # exact failure Principle 6 names.
            p.status = "declined"
            p.decline_reason = (
                f"prescription kind {p.kind!r} with targets {p.targets} could not be applied: "
                "no executor is defined for this combination"
            )
            handled.append(p)

    new_family = leaf_family.copy()
    new_labels = leaf_labels.copy() if leaf_labels is not None else None
    detail: dict[str, Any] = {}
    if to_split and can_split:
        new_labels, new_family, detail["splits"] = split_leaves(
            X, new_labels, new_family, to_split
        )
    if merge_map:
        new_family, detail["merges"] = apply_merges(new_family, merge_map)
        # Named per prescription, so a PARTIALLY applied merge is visible rather
        # than averaging into a delta that reads as complete.
        detail["merges"]["targets_that_named_no_family"] = {
            k: v for k, v in unresolvable.items()}
    if isolate:
        new_family, detail["isolations"] = isolate_leaves(new_family, isolate)

    metrics_after = recompute(new_family) if recompute else {}
    deltas = {
        k: round(float(metrics_after[k]) - float((metrics_before or {}).get(k, 0.0)), 4)
        for k in metrics_after
        if isinstance(metrics_after.get(k), (int, float))
    }

    now = time.time()
    for p in handled:
        if p.status == "declined":
            continue
        p.status = "executed"
        p.executed_at = now
        p.evidence = {
            "artifact": "labels_full",
            "column": {"split_leaf": "bu_leaf", "relabel": "bu_leaf_name"}.get(
                p.kind, "family_final"),
            "mechanism": (
                "leaf re-partitioned by local KMeans; assignments and centroids change"
                if p.kind == "split_leaf" else
                "leaf→family lookup-table remap; leaf assignments and centroids unchanged"
            ),
            "before": detail.get("merges", {}).get("n_families_before"),
            "after": int(len(np.unique(new_family))),
            "metric_deltas": deltas,
        }

    if relabel:
        detail["relabelled"] = {str(k): v for k, v in relabel.items()}
    if new_labels is not None:
        detail["leaf_labels"] = new_labels
    return new_family, list(prescriptions), {
        **detail,
        "metrics_before": metrics_before or {},
        "metrics_after": metrics_after,
        "metric_deltas": deltas,
        "n_executed": sum(1 for p in prescriptions if p.status == "executed"),
        "n_declined": sum(1 for p in prescriptions if p.status == "declined"),
    }


def assert_all_settled(prescriptions: Sequence[Prescription]) -> dict[str, Any]:
    """**The Principle 6 gate.**  Every prescription is executed or explicitly declined.

    An unsettled prescription is not a minor bookkeeping lapse — it is a report
    claiming a fix that the data does not contain.  This raises rather than warns.
    """
    open_ones = [p for p in prescriptions if not p.settled]
    if open_ones:
        raise GovernanceError(
            f"{len(open_ones)} prescription(s) never reached the data: "
            + "; ".join(f"{p.id}({p.kind}→{p.targets})" for p in open_ones[:6])
            + ". Principle 6: every 'we recommend X' in a report must have a corresponding "
            "executed change in a delivered column, or an explicit declined reason."
        )
    return {
        "n_total": len(prescriptions),
        "n_executed": sum(1 for p in prescriptions if p.status == "executed"),
        "n_declined": sum(1 for p in prescriptions if p.status == "declined"),
        "declined_reasons": [
            {"id": p.id, "reason": p.decline_reason} for p in prescriptions if p.status == "declined"
        ],
        "verdict": "all prescriptions settled",
    }


def governance_ledger(prescriptions: Sequence[Prescription]) -> list[dict[str, Any]]:
    """The table that goes into the report, one row per prescription."""
    return [
        {
            "id": p.id,
            "kind": p.kind,
            "targets": p.targets,
            "target_names": p.target_names,
            "status": p.status,
            "rationale": p.rationale,
            "executed_column": p.evidence.get("column", ""),
            "metric_deltas": p.evidence.get("metric_deltas", {}),
            "decline_reason": p.decline_reason,
        }
        for p in prescriptions
    ]
