"""Assemble the one frame everything else reads, and name every class in it.

THE LABEL JOIN IS POSITIONAL AND IT IS CHECKED. `labels_full.csv` is written in
corpus order, so position is the only correct key. Joining on query TEXT would
be wrong and silently so: the same string occurs in several snapshots — that
overlap is part of what the study measures — and a text join binds a row to the
wrong snapshot. Length and per-row text are both asserted, and the analysis
refuses to run rather than produce a plausible wrong table.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .manifest import SnapshotManifest, build_manifest
from .source import CANONICAL_SNAPSHOT_COL, RunSource

#: Columns the analysis reads off the labels table. Anything missing is simply
#: absent from the tables that use it; nothing here is required except the two
#: route keys.
LABEL_COLS = ["td_l1", "td_l2", "bu_leaf", "bu_family_final", "td_confidence",
              "td_margin", "td_ambiguous", "bu_margin", "bu_ambiguous",
              "td_l1_name", "bu_leaf_name", "td_decided_by"]


class PooledJoinError(RuntimeError):
    """The corpus and the delivered labels do not line up."""


def load_frame(src: RunSource, *, weight_column: str | None = None,
               axis: str = "time", labels: dict[str, str] | None = None,
               surfaces: dict[str, str] | None = None,
               contrasts: Any = None) -> tuple[pd.DataFrame, SnapshotManifest]:
    """The corpus joined to its delivered labels, plus the manifest."""
    corpus = src.table("corpus")
    lab = src.table("labels_full")
    if corpus is None:
        raise PooledJoinError("this generation has no corpus.parquet — nothing to compare")
    if lab is None:
        raise PooledJoinError(
            "this generation has no labels_full.csv — the run has not reached p10, "
            "so there are no delivered labels to compare")
    if len(corpus) != len(lab):
        raise PooledJoinError(
            f"corpus has {len(corpus):,} rows and labels_full.csv has {len(lab):,}. "
            "The join is positional and cannot be verified; refusing to guess.")
    qc = corpus["query"].astype(str).to_numpy()
    ql = lab["query"].astype(str).to_numpy()
    bad = int((qc != ql).sum())
    if bad:
        raise PooledJoinError(
            f"{bad:,} of {len(qc):,} rows differ in query text between the corpus and "
            "labels_full.csv. A positional join is no longer valid here.")

    d = corpus.reset_index(drop=True).copy()
    for c in LABEL_COLS:
        if c in lab.columns:
            d[c] = lab[c].reset_index(drop=True).to_numpy()
    # p10 copies the snapshot column onto the delivered labels; either source is
    # acceptable, and disagreeing sources are a bug we would rather see.
    if CANONICAL_SNAPSHOT_COL not in d.columns and CANONICAL_SNAPSHOT_COL in lab.columns:
        d[CANONICAL_SNAPSHOT_COL] = lab[CANONICAL_SNAPSHOT_COL].to_numpy()
    if CANONICAL_SNAPSHOT_COL not in d.columns:
        raise PooledJoinError(
            "no `snapshot` column — this run pooled a single input, so there is "
            "nothing to compare across snapshots")
    d[CANONICAL_SNAPSHOT_COL] = d[CANONICAL_SNAPSHOT_COL].astype(str)

    weight = weight_column or _guess_weight(d)
    man, pv = build_manifest(d, snapshot_col=CANONICAL_SNAPSHOT_COL, weight_col=weight,
                             axis=axis, labels=labels, surfaces=surfaces,
                             contrasts=contrasts)
    d["pv_norm"] = pv
    if weight and weight in d.columns:
        d["pv_raw"] = pd.to_numeric(d[weight], errors="coerce")
    else:
        d["pv_raw"] = np.nan
    man.weight_column = weight or ""
    for c in ("td_ambiguous", "bu_ambiguous"):
        if c in d.columns:
            d[c] = _as_bool(d[c])
    for c in ("td_confidence", "td_margin", "bu_margin"):
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d, man


def _as_bool(s: pd.Series) -> pd.Series:
    """CSV round-trips booleans as the strings 'True'/'False'."""
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin({"true", "1", "yes", "是"})


def _guess_weight(d: pd.DataFrame) -> str | None:
    """Which column is traffic. Named candidates first, then the config's own."""
    for c in ("pv_norm", "pv_raw", "pv", "weight", "search_num", "wise_pv", "count"):
        if c in d.columns and pd.to_numeric(d[c], errors="coerce").notna().any():
            return c
    return None


def class_names(src: RunSource) -> dict[str, dict[str, dict[str, Any]]]:
    """Display name and definition per class, per level, FROM THE DELIVERED TREE.

    `p8` rewrites the hierarchy, so `hierarchy_meta` / `leaf_labels` / p7's
    namings describe a tree that no longer exists — one run delivered 16 leaves
    where those artifacts said 25. Families therefore come from
    `families_final`, never `audit.families`.
    """
    tn = src.json("tree_naming") or {}
    out: dict[str, dict[str, dict[str, Any]]] = {
        "bu_leaf": {}, "bu_family_final": {}, "td_l1": {}, "td_l2": {}}
    for n in (tn.get("namings") or []):
        if n.get("leaf_id") is None:
            continue
        out["bu_leaf"][str(int(n["leaf_id"]))] = {
            "name": n.get("name_zh") or n.get("name") or "",
            "definition": n.get("user_need") or "",
            "user_need": n.get("user_need") or "",
            "coherence": n.get("coherence"),
            "risk_flag": n.get("risk_flag") or ""}
    for f in (tn.get("families_final") or []):
        if f.get("family_id") is None:
            continue
        out["bu_family_final"][str(int(f["family_id"]))] = {
            "name": f.get("name_zh") or f.get("name") or "",
            "definition": f.get("definition") or "",
            "coherent": "是" if f.get("coherent") else "否",
            "audit_notes": f.get("audit_notes") or ""}
    tx = src.json("taxonomy_v2") or src.json("taxonomy") or {}
    tx = tx.get("taxonomy", tx)
    for n in (tx.get("nodes") or []):
        if n.get("level") != 1 or not n.get("code"):
            continue
        out["td_l1"][str(n["code"])] = {
            "name": n.get("name") or n.get("name_zh") or str(n["code"]),
            "definition": n.get("definition") or "",
            "user_need": n.get("user_need") or "",
            "risk": "是" if n.get("risk") else "否",
            "expected_share": n.get("expected_share")}
    return out
