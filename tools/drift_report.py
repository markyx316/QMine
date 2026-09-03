#!/usr/bin/env python
"""SUPERSEDED by phase p10b — kept as an independent check on it.

`ops/drift.py` + `report/zh_drift.py` now do this inside the run and ship
`快照对比_漂移分析.md`. This script computes the same quantities from outside the
pipeline, which is how the p10b numbers were first validated (it reproduced
`LOOKUP_MARKET_QUOTE +12.802pp`, purity 0/54, jaccard 0.3743 exactly). Useful for
the five pooled runs that predate p10b and have no `drift_analysis.json`.

Compare labelled snapshots inside ONE pooled run — what moved between periods.

Joins a run's `labels_full.csv` back to the pooled source POSITIONALLY, not on
query text. `labels_full.csv` preserves input row order exactly (verified
10,000/10,000 on `fin03`), and a text join fans out on repeated queries — the
finance snapshot contains one, and a bigger log contains many.

TWO NUMBERS, AND THEY ANSWER DIFFERENT QUESTIONS.

* **Row share** — what fraction of DISTINCT queries fell in this class. Answers
  "did the variety of things people ask shift?"
* **PV share** — what fraction of TRAFFIC. Answers "did what people actually
  search shift?"

Both are WITHIN-SNAPSHOT shares, never raw counts, because the snapshots are not
the same size in traffic: the two 医疗 files carry 9,736,913 and 5,208,934 PV, a
47% fall. Comparing raw PV would report every class as declining and say nothing
about composition.

SIGNIFICANCE. Row shares get a two-proportion z-test. At n=10,000 per snapshot a
0.5pp move clears p<0.05 on its own, so the test is a floor and not a finding:
with ~20 classes tested at once, roughly one in twenty crosses by chance. The
report prints the test count so the reader can discount accordingly, and ranks by
EFFECT SIZE rather than by p. PV share gets no test — traffic is not a sample of
anything, it is the population, and a p-value over it would be theatre.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd


def _z_two_prop(x1: int, n1: int, x2: int, n2: int) -> float:
    if not n1 or not n2:
        return 0.0
    p1, p2 = x1 / n1, x2 / n2
    p = (x1 + x2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    return 0.0 if se == 0 else (p1 - p2) / se


def _table(df: pd.DataFrame, col: str, snaps: list[str], weight: str, top: int):
    n = {s: (df._snapshot == s).sum() for s in snaps}
    pv = {s: df.loc[df._snapshot == s, weight].sum() or 1 for s in snaps}
    a, b = snaps[0], snaps[1]
    rows = []
    for val, g in df.groupby(col, dropna=False):
        xa = int((g._snapshot == a).sum())
        xb = int((g._snapshot == b).sum())
        ra, rb = xa / n[a], xb / n[b]
        pa = g.loc[g._snapshot == a, weight].sum() / pv[a]
        pb = g.loc[g._snapshot == b, weight].sum() / pv[b]
        rows.append({"label": str(val)[:34], "rows_a": xa, "rows_b": xb,
                     "row_share_a": ra, "row_share_b": rb, "d_row_pp": (rb - ra) * 100,
                     "pv_share_a": pa, "pv_share_b": pb, "d_pv_pp": (pb - pa) * 100,
                     "z": _z_two_prop(xb, n[b], xa, n[a])})
    t = pd.DataFrame(rows)
    if t.empty:
        return t, n, pv
    return t.reindex(t.d_pv_pp.abs().sort_values(ascending=False).index).head(top), n, pv


def _print(t: pd.DataFrame, title: str, n: dict, snaps: list[str]) -> None:
    a, b = snaps
    print(f"\n{title}   ({a} n={n[a]:,}  ->  {b} n={n[b]:,})")
    if t.empty:
        print("   (no rows)")
        return
    print(f"   {'label':34s} {'rows':>11s} {'row share':>16s} {'PV share':>16s} {'ΔPV':>7s} {'z':>7s}")
    for r in t.itertuples():
        star = "*" if abs(r.z) >= 1.96 else " "
        print(f"   {r.label:34s} {r.rows_a:5,}->{r.rows_b:5,} "
              f"{r.row_share_a:6.2%}->{r.row_share_b:6.2%} "
              f"{r.pv_share_a:6.2%}->{r.pv_share_b:6.2%} "
              f"{r.d_pv_pp:+6.2f} {r.z:+6.1f}{star}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("gen_dir", help="runs/<id>/genNN")
    ap.add_argument("pooled", help="the pooled source built by pool_snapshots.py")
    ap.add_argument("--weight", default="wise_pv")
    ap.add_argument("--top", type=int, default=14)
    a = ap.parse_args()

    lab_p = Path(a.gen_dir) / "labels_full.csv"
    if not lab_p.exists():
        sys.exit(f"no labels_full.csv in {a.gen_dir}")
    lab = pd.read_csv(lab_p)
    src = pd.read_csv(a.pooled) if not a.pooled.endswith(".parquet") else pd.read_parquet(a.pooled)

    text_col = "original_query" if "original_query" in src.columns else src.columns[0]

    # MIRROR p1's OWN ROW FILTER, or the two disagree about which rows exist.
    #
    # p1 drops rows whose text cell is empty (an empty query is not a query), so
    # a pooled file with one blank yields 19,999 labelled rows against 20,000
    # source rows and the positional join is off by one from that point on.
    # `edu-pool` did exactly this. Reproducing the filter here is what keeps row
    # i of the labels equal to row i of the source; it is NOT a fudge to make the
    # counts match, and the verification below still has to pass afterwards.
    _n0 = len(src)
    src = src[src[text_col].notna()].reset_index(drop=True)
    if len(src) < _n0:
        print(f"dropped {_n0 - len(src):,} source row(s) with an empty {text_col!r}, "
              f"matching the filter p1 applied")

    if len(lab) != len(src):
        sys.exit(f"POSITIONAL JOIN IMPOSSIBLE: {len(lab):,} labelled rows vs "
                 f"{len(src):,} source rows (after the empty-text filter). Either a "
                 f"--sample was used, or these are not the same corpus. Re-run "
                 f"without --sample rather than joining rows that do not correspond.")
    mism = (src[text_col].astype(str).values != lab["query"].astype(str).values).sum()
    if mism:
        sys.exit(f"POSITIONAL JOIN REFUSED: {mism:,} of {len(lab):,} rows disagree on "
                 "the query text, so row i of the labels is not row i of the source.")
    print(f"positional join verified: {len(lab):,}/{len(lab):,} rows agree on query text")

    df = pd.concat([src.reset_index(drop=True), lab.reset_index(drop=True)
                    .drop(columns=[c for c in ("query",) if c in lab.columns])], axis=1)
    snaps = sorted(df._snapshot.astype(str).unique())
    df["_snapshot"] = df._snapshot.astype(str)
    if len(snaps) != 2:
        sys.exit(f"expected 2 snapshots, found {snaps}")

    w = a.weight if a.weight in df.columns else None
    if w is None:
        df["_w"] = 1.0
        w = "_w"
        print("⚠ no weight column — PV shares are row shares")

    n_tests = 0
    for col, title in (("td_l1", "TOP-DOWN INTENT (td_l1)"),
                       ("bu_family_final", "BOTTOM-UP FAMILY (bu_family_final)"),
                       ("bu_leaf_name", "BOTTOM-UP LEAF (bu_leaf_name)")):
        if col not in df.columns:
            continue
        t, n, _pv = _table(df, col, snaps, w, a.top)
        n_tests += len(t)
        _print(t, title, n, snaps)

    print(f"\n* = |z| >= 1.96 on ROW share. {n_tests} comparisons shown; at ~1 in 20 by "
          f"chance, expect ~{n_tests / 20:.0f} spurious star(s). Rank by ΔPV, not by z.")
    print("PV share is a population, not a sample — it carries no test by design.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
