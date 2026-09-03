#!/usr/bin/env python
"""Restore rows p1 dropped, so the delivered table accounts for EVERY input row.

p1 drops rows whose text cell is empty — an empty query is not a query, and
labelling one would put a cluster and an intent on a blank. But the delivered
`labels_full.csv` then has fewer rows than the input, and a table that silently
holds 19,999 rows against a 20,000-row export is an unexplained discrepancy in
whatever report it lands in. The reader cannot tell a dropped blank from a lost
row.

This puts them back AT THEIR ORIGINAL POSITION, with every label column empty and
an explicit reason. The row count reconciles, the ordering is unchanged, and the
blank is visible as a blank rather than as an absence.

It writes a NEW file and never touches `labels_full.csv`: the delivered artifact
is evidence of what the run produced, and editing it in place would destroy that.

    python tools/reconcile_labels.py runs/edu-pool/gen01 data/raw/教育query-pooled.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("gen_dir")
    ap.add_argument("source", help="the exact file the run was given")
    ap.add_argument("--text-column", default="original_query")
    ap.add_argument("-o", "--out", default=None)
    a = ap.parse_args()

    gen = Path(a.gen_dir)
    lab_p = gen / "labels_full.csv"
    if not lab_p.exists():
        sys.exit(f"no labels_full.csv in {gen}")
    lab = pd.read_csv(lab_p)
    src = (pd.read_parquet(a.source) if a.source.endswith(".parquet")
           else pd.read_csv(a.source) if a.source.endswith(".csv")
           else pd.read_excel(a.source))
    if a.text_column not in src.columns:
        sys.exit(f"--text-column {a.text_column!r} not in {list(src.columns)}")

    kept = src[a.text_column].notna()
    n_dropped = int((~kept).sum())
    if len(lab) != int(kept.sum()):
        sys.exit(f"cannot reconcile: {len(lab):,} labelled rows but {int(kept.sum()):,} "
                 f"source rows survive the empty-text filter. These are not the same "
                 f"corpus, or a --sample was used.")
    if n_dropped == 0:
        print("nothing was dropped — the delivered table already accounts for every row")
        return 0

    # Rebuild at full length: label columns land on the KEPT positions, and the
    # dropped positions keep their source fields with empty labels.
    out = src.copy()
    label_cols = [c for c in lab.columns if c != "query"]
    for c in label_cols:
        out[c] = pd.Series(pd.NA, index=out.index, dtype="object")
        out.loc[kept, c] = lab[c].values
    out["_unlabelled_reason"] = pd.Series(pd.NA, index=out.index, dtype="object")
    out.loc[~kept, "_unlabelled_reason"] = (
        f"empty {a.text_column} in the source export — dropped by p1 before analysis; "
        "there is no query text to cluster or classify")

    dest = Path(a.out) if a.out else gen / "labels_full_reconciled.csv"
    # utf-8-sig, MATCHING `artifacts.py`. Excel does not sniff UTF-8 in a
    # CSV: without the BOM it falls back to the system codepage and every
    # Chinese label renders as mojibake. The pipeline's own writer uses
    # utf-8-sig for exactly this reason; a sidecar tool that omits it
    # produces a file that looks corrupt next to one that does not.
    out.to_csv(dest, index=False, encoding="utf-8-sig")
    print(f"restored {n_dropped:,} unlabelled row(s); {len(out):,} rows total -> {dest}")
    for i in src.index[~kept]:
        row = src.loc[i]
        print(f"  position {i:,} (row {i + 1:,} of {len(src):,}): "
              + ", ".join(f"{c}={row[c]!r}" for c in src.columns
                          if c not in ("_row_in_snapshot",))[:150])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
