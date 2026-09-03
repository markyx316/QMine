#!/usr/bin/env python
"""Stack time-separated snapshots of one vertical into a single corpus.

WHY POOL RATHER THAN RUN EACH SNAPSHOT AND DIFF THE RESULTS.

Two runs of this pipeline do not produce comparable labels. Measured on the two
finance runs: `fin02` and `fin03` ran the SAME 10,000 rows under the same config
and shared **0 of 35 class codes** — the architect re-invents its naming
convention every run, and the bottom-up tree is re-fitted from scratch. Diffing
two runs by label therefore reports that nothing reproduced when most of it did,
and no amount of care in the analysis can recover a correspondence the runs never
established.

Pooling removes the problem at the source: one run, one taxonomy, one cluster
tree, and every row from every snapshot labelled in that single frame. The
snapshot each row came from is preserved so the analysis can split on it
afterwards.

`_snapshot` is NOT passed to the pipeline as a reference label column. Reference
columns are the frame the K locator scores against, so declaring the snapshot
would ask the clustering to find a K that separates 2025 from 2026 — which is
both meaningless and the opposite of what a shared frame is for. The column rides
along in this file and is re-joined POSITIONALLY afterwards (see
`tools/drift_report.py`); `labels_full.csv` preserves input row order exactly,
verified 10,000/10,000 on `fin03`.

    python tools/pool_snapshots.py data/raw/医疗query-250701.xlsx \
        data/raw/医疗query-260701.xlsx -o data/raw/医疗query-pooled.csv
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd


def _read(p: Path) -> pd.DataFrame:
    if p.suffix in (".xlsx", ".xls"):
        return pd.read_excel(p)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p)


def _label(p: Path, df: pd.DataFrame, col: str | None) -> str:
    """Name the snapshot: the date column if it is constant, else the filename."""
    if col and col in df.columns:
        vals = df[col].dropna().unique()
        if len(vals) == 1:
            return str(vals[0])
    m = re.search(r"(\d{6,8})", p.stem)
    return m.group(1) if m else p.stem


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--date-column", default="event_day",
                    help="used to NAME each snapshot when it holds one value")
    a = ap.parse_args()

    frames, seen = [], []
    for raw in a.inputs:
        p = Path(raw)
        if not p.exists():
            sys.exit(f"no such file: {p}")
        df = _read(p)
        tag = _label(p, df, a.date_column)
        if tag in seen:
            sys.exit(f"two inputs resolve to the same snapshot label {tag!r} — "
                     "the analysis could not tell them apart")
        seen.append(tag)
        df = df.copy()
        df["_snapshot"] = tag
        df["_row_in_snapshot"] = range(len(df))
        # NO `_source_file` COLUMN. It was provenance nobody reads, and p1's
        # `_label_like_columns` heuristic flagged it as an undeclared label
        # column — producing a WARNED `p1_reference_columns_declared` gate whose
        # message ("the gold set and pilot are UNSTRATIFIED by legacy label") is
        # misleading here, because there is no legacy label. A tool should not
        # manufacture a warning about itself. `_snapshot` carries everything the
        # drift analysis needs, and it is not flagged.
        frames.append(df)
        print(f"  {p.name:30s} {len(df):>7,} rows  -> _snapshot={tag}")

    cols = [set(f.columns) for f in frames]
    common = set.intersection(*cols)
    if any(c != cols[0] for c in cols):
        # Concatenating mismatched schemas silently fills NaN, and a column that
        # exists in one snapshot and not another becomes a spurious drift signal.
        extra = sorted(set.union(*cols) - common)
        print(f"  ⚠ columns differ across inputs; keeping the {len(common)} shared "
              f"and DROPPING {extra}")
        frames = [f[sorted(common)] for f in frames]

    out = pd.concat(frames, ignore_index=True)
    dest = Path(a.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.suffix == ".parquet":
        out.to_parquet(dest, index=False)
    else:
        # utf-8-sig, MATCHING `artifacts.py`. Excel does not sniff UTF-8 in a
        # CSV: without the BOM it falls back to the system codepage and every
        # Chinese label renders as mojibake. The pipeline's own writer uses
        # utf-8-sig for exactly this reason; a sidecar tool that omits it
        # produces a file that looks corrupt next to one that does not.
        out.to_csv(dest, index=False, encoding="utf-8-sig")
    print(f"\n  pooled -> {dest}  {len(out):,} rows x {len(out.columns)} cols")
    print(f"  snapshots: {out._snapshot.value_counts().to_dict()}")
    print("\n  Run it, then join the labels back positionally:")
    print(f"    qmine run --input {dest} --domain <d> --config configs/corpus_wise_export.yaml --fast --run-id <id>")
    print(f"    python tools/drift_report.py runs/<id>/gen01 {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
