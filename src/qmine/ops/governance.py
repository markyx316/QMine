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
        }
    return labels, np.array(fam, dtype=np.int64), done


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

    for p in prescriptions:
        if p.kind == "merge_families" and len(p.targets) >= 2:
            keep = int(min(p.targets))
            for t in p.targets:
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
