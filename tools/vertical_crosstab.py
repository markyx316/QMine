#!/usr/bin/env python
"""Cross a finished run's DELIVERED classes against the source categories.

WHY THIS EXISTS. Pooling 33 verticals into one run buys exactly one thing: every
category is labelled by the SAME taxonomy, so "what does 金融 ask for that 医疗
does not" becomes answerable. Nothing in the run's deliverables actually showed
that — `route_crosswalk.csv` crosses the bottom-up families against the top-down
classes, which is a different question — so the payoff for pooling was computed
and never delivered. That is the `produced but not delivered` failure this project
has hit before.

Mechanical: reads `labels_full.csv`, no model call, so it runs on any finished run
in either mode and can be re-run without spending anything.

READS THE DELIVERED PARTITION. p8 rewrites the tree, so `bu_family_final` and
`bu_leaf_name` are the delivered columns and `*_pre_governance` describes a tree
that no longer exists. This refuses the pre-governance columns rather than
offering them.

SHARES ARE WITHIN-CATEGORY, and that is not a stylistic choice. The export
allocates exactly 1,000 rows per category per stratum, so a raw count across
categories measures the export's allocation, not the corpus. Traffic shares are
within-category for the same reason a drift report uses within-snapshot shares.

    python tools/vertical_crosstab.py runs/ai01/gen01 --category-column ref_l1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

#: Columns p8 may have rewritten. Naming one of these would describe a tree the
#: run did not deliver.
_PRE_GOVERNANCE = {"bu_leaf_pre_governance", "bu_family_pre_governance"}


def _family_names_from_labels(run: Path) -> dict[str, str]:
    """`31` -> `#31 图片编辑操作`, derived from the rows themselves.

    The fallback for a run that shipped no family table. A family's display name
    is the leaf name the largest share of its rows carries, with that share
    stated when it is not dominant — `#31 图片编辑操作 +2` means two other leaf
    names are present, so the reader knows the label is a plurality rather than
    a description of the whole family.
    """
    f = run / "labels_full.csv"
    if not f.exists():
        return {}
    try:
        d = pd.read_csv(f, usecols=["bu_family_final", "bu_leaf_name"]).dropna()
        out: dict[str, str] = {}
        for fam, g in d.groupby("bu_family_final"):
            counts = g["bu_leaf_name"].value_counts()
            extra = len(counts) - 1
            name = str(counts.index[0])[:40]
            key = str(int(fam)) if str(fam).replace(".", "").isdigit() else str(fam)
            out[key] = f"#{key} {name}" + (f" +{extra}" if extra > 0 else "")
        return out
    except Exception:  # noqa: BLE001
        return {}


def _family_names(run: Path) -> dict[str, str]:
    """`31` -> `#31 图片编辑操作`, from the run's own delivered family table.

    A BARE ID IS NOT A FINDING. p10b learned this the hard way — its first real
    render shipped `11` and `15` as the two biggest movers and nobody could act on
    it. `家族与叶.csv` is the delivered join, so no artifact reconstruction and no
    integer-keyed join across files is needed.

    Never raises: a table with bare ids beats no table.
    """
    f = run / "家族与叶.csv"
    if not f.exists():
        # FAST MODE SHIPS THREE REFERENCE DOCUMENTS AND NOT THIS CSV, so the
        # delivered family table simply is not there — and falling back to bare
        # ids would then be the NORMAL outcome in the mode this project now runs
        # by default, not a rare degradation. `labels_full.csv` ships in both
        # modes and carries both columns, so a family can be named by the leaf
        # name most of its rows already carry.
        return _family_names_from_labels(run)
    try:
        d = pd.read_csv(f)
        if "family_id" not in d.columns:
            return {}
        label = ("family_label" if "family_label" in d.columns
                 else "family_name" if "family_name" in d.columns else None)
        if label is None:
            return {}
        pairs = d[["family_id", label]].dropna().drop_duplicates("family_id")
        return {str(k): f"#{k} {str(v)[:40]}" for k, v in pairs.itertuples(index=False)}
    except Exception:  # noqa: BLE001
        return {}


def crosstab(df: pd.DataFrame, cat: str, label: str, weight: str | None) -> pd.DataFrame:
    """Rows and traffic per (category, class), plus each share WITHIN its category."""
    work = df[[cat, label] + ([weight] if weight else [])].copy()
    work = work[work[cat].notna() & work[label].notna()]
    agg = {"rows": (label, "size")}
    if weight:
        agg["traffic"] = (weight, "sum")
    out = work.groupby([cat, label], as_index=False).agg(**agg)
    out["row_share_in_category"] = out.rows / out.groupby(cat).rows.transform("sum")
    if weight:
        out["traffic_share_in_category"] = (
            out.traffic / out.groupby(cat).traffic.transform("sum"))
    return out.sort_values([cat, "rows"], ascending=[True, False])


def concentration(ct: pd.DataFrame, cat: str, label: str) -> pd.DataFrame:
    """Is a class one category's speciality, or is it everywhere?

    Names the measurement: `n_categories` and the share held by the largest
    category. A class in 32 of 33 categories is not a property of any vertical —
    that is how the acknowledgement family shows up — and a class at 90% in one
    category is that category's speciality. Which of those matters is the reader's
    call, so both numbers ship and no verdict does.
    """
    g = ct.groupby(label)
    out = pd.DataFrame({
        "rows": g.rows.sum(),
        "n_categories": g[cat].nunique(),
        "top_category": g.apply(lambda d: d.loc[d.rows.idxmax(), cat], include_groups=False),
        "top_category_share": g.rows.max() / g.rows.sum(),
    })
    return out.sort_values("rows", ascending=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir", help="a run generation directory, e.g. runs/ai01/gen01")
    ap.add_argument("--category-column", default="ref_l1")
    ap.add_argument("--label-column", default="bu_family_final",
                    help="delivered class column (bu_family_final / bu_leaf_name / td_l1)")
    ap.add_argument("--weight-column", default=None,
                    help="a traffic column in the corpus parquet, joined by row order")
    ap.add_argument("--top", type=int, default=6, help="classes shown per category")
    ap.add_argument("-o", "--out", default=None,
                    help="write a Chinese markdown table here (default: <run_dir>/垂类交叉表.md)")
    a = ap.parse_args()

    run = Path(a.run_dir)
    labels = run / "labels_full.csv"
    if not labels.exists():
        sys.exit(f"{labels} not found — is that a run generation directory?")
    if a.label_column in _PRE_GOVERNANCE:
        sys.exit(f"{a.label_column} describes the tree BEFORE p8 rewrote it. "
                 f"Use bu_family_final, bu_leaf_name or td_l1.")
    df = pd.read_csv(labels)
    for col in (a.category_column, a.label_column):
        if col not in df.columns:
            sys.exit(f"{col!r} is not in labels_full.csv, which has: {list(df.columns)}")

    weight = None
    if a.weight_column:
        corpus = run / "corpus.parquet"
        if not corpus.exists():
            sys.exit(f"--weight-column needs {corpus}, which is missing")
        cdf = pd.read_parquet(corpus)
        if a.weight_column not in cdf.columns:
            sys.exit(f"{a.weight_column!r} not in corpus.parquet: {list(cdf.columns)}")
        if len(cdf) != len(df):
            sys.exit(f"corpus has {len(cdf):,} rows and labels {len(df):,} — refusing to "
                     f"join by position when the lengths disagree")
        df = df.assign(_w=cdf[a.weight_column].to_numpy())
        weight = "_w"

    # Names before aggregation, so both tables and the CSV carry them.
    if a.label_column == "bu_family_final":
        names = _family_names(run)
        if names:
            df[a.label_column] = df[a.label_column].map(
                lambda i: names.get(str(int(i)) if pd.notna(i) else "", f"#{i}"))
        else:
            print("  ⚠ 家族与叶.csv unreadable — families will show as bare ids")

    ct = crosstab(df, a.category_column, a.label_column, weight)
    conc = concentration(ct, a.category_column, a.label_column)

    ct.to_csv(run / "垂类交叉表.csv", index=False)
    out = Path(a.out) if a.out else run / "垂类交叉表.md"

    L = ["# 垂类 × 交付类目 交叉表",
         "",
         f"**运行**: `{run}` · **类目列**: `{a.label_column}`（交付分区）· "
         f"**垂类列**: `{a.category_column}`",
         "",
         "> **占比一律是「垂类内占比」。** 本导出对每个垂类固定抽取相同行数，"
         "所以跨垂类比原始行数量到的是抽样设计，不是语料。",
         "",
         f"## 1. 每个垂类里最大的 {a.top} 个交付类目",
         ""]
    for cat, g in ct.groupby(a.category_column):
        L += [f"### {cat}", "",
              "| 交付类目 | 行数 | 垂类内行占比 |" + (" 垂类内流量占比 |" if weight else ""),
              "|---|---:|---:|" + ("---:|" if weight else "")]
        for _, r in g.head(a.top).iterrows():
            row = f"| `{r[a.label_column]}` | {int(r.rows):,} | {r.row_share_in_category:.1%} |"
            if weight:
                row += f" {r.traffic_share_in_category:.1%} |"
            L.append(row)
        L.append("")

    L += ["## 2. 哪些类目是某个垂类特有的，哪些哪里都有", "",
          "> `n_categories` 与 `top_category_share` 是**测量**，不是结论。"
          "出现在 30+ 个垂类里的类目不是任何一个垂类的属性（无内容确认就长这样）；"
          "单一垂类占比 90% 以上的，是那个垂类的专属需求。哪一种重要由读者判断。",
          "",
          "| 交付类目 | 总行数 | 覆盖垂类数 | 最大垂类 | 最大垂类占比 |",
          "|---|---:|---:|---|---:|"]
    for lab, r in conc.head(40).iterrows():
        L.append(f"| `{lab}` | {int(r.rows):,} | {int(r.n_categories)} | "
                 f"{r.top_category} | {r.top_category_share:.1%} |")
    out.write_text("\n".join(L) + "\n", encoding="utf-8")

    print(f"wrote {out}")
    print(f"wrote {run / '垂类交叉表.csv'}")
    print(f"\n  {len(ct):,} (垂类, 类目) cells over {ct[a.category_column].nunique()} "
          f"categories and {ct[a.label_column].nunique()} delivered classes")
    print(f"  classes present in >=80% of categories: "
          f"{(conc.n_categories >= 0.8 * ct[a.category_column].nunique()).sum()}")
    print(f"  classes >=90% inside one category:      {(conc.top_category_share >= 0.9).sum()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
