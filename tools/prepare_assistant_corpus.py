#!/usr/bin/env python
"""Pool the AI-assistant head and tail exports into one corpus the pipeline can mine.

WHY THIS TOOL EXISTS RATHER THAN `--input a.xlsx,b.xlsx`.

`--input` with several paths means SNAPSHOTS OF ONE VERTICAL AT DIFFERENT TIMES
(`.claude/rules/multi-snapshot.md`). These two files are not that. They are two
SAMPLING STRATA of the same period: `Top1000` is each category's top 1,000 rows
by `search_num`, `随机1000` is a random 1,000 from the same category. Passing them
as `--input` tags them with `_snapshot`, and p10b then ships
「快照对比 · 漂移分析」 — a document whose every interpretive sentence ("不是趋
势", "同月同日不等于季节可比", "时段性事件") is false about a head-vs-tail split.
So the pooling is right and the LABEL is not: this tool writes the stratum into a
column named `stratum`, and `data.comparison_axis: stratum` makes the report say
what was actually compared.

WHAT THIS TOOL DECIDES, AND WHY EACH DECISION IS THE WAY IT IS.

1. POOLED, NOT TWO RUNS. Same argument as snapshots, same evidence: two runs over
   the same rows shared 0 of 35 class codes. One taxonomy has to label both
   strata or the comparison is between two vocabularies, not two populations.

2. DUPLICATES COLLAPSE ONLY WITHIN (stratum, l1, l2). The same string appears
   under up to 32 of the 33 first-level categories — `👌 好的，继续吧` does — and
   the category is real data about where the event happened. Collapsing across
   categories would move 2.13M PV into whichever category happened to be modal
   and wreck that category's normalisation. Exact duplicate rows are summed;
   cross-category occurrences stay as separate rows and carry `n_l1_categories`.

3. `pv_norm` IS THE DEFAULT WEIGHT, NOT `search_num`. Each category's top-1,000
   has its OWN PV floor — measured, 3 (招商加盟) to 419 (书籍文档) — so the file is
   a union of 33 censuses at 33 different floors and raw PV is not comparable
   across them. Under raw PV two categories hold 55% of pooled weight and one
   string (`变清晰`, 5.23M) holds 17%, so every weighted metric in the run would
   mostly be about that string. `pv_norm` normalises within (stratum, category)
   to 1,000, which matches the export's own equal-allocation design and makes a
   traffic share readable as "share of a typical category's stratum traffic".
   `search_num` is written too — switch `weight_column` if you want raw.

   NOT RECOVERABLE, and do not try: reweighting to the true population needs each
   category's total traffic, and the two files cannot supply it. The
   capture-recapture bridge (what fraction of the random sample sits at or above
   the head floor) collapses — 17 of 33 categories have ZERO random rows above
   their head floor and 28 have fewer than 5, so the estimator returns 1,000,000
   distinct queries off a single row. Cross-category traffic comparison is out of
   reach with these two files; say so rather than estimating it.

4. ACKNOWLEDGEMENT ROWS ARE FLAGGED, NEVER DROPPED. Seven strings carry 15.97% of
   pooled head PV and appear in 29-32 of the 33 categories; the full closed-list
   family is 1,065 rows and 17.02% of head PV. That they dominate the head is the
   single largest finding in the corpus, and dropping them would delete it. They
   are marked `is_ack` by the explicit pattern below — a stated rule the reader
   can audit — and `ai_assistant_zh.yaml` seeds them as their own phrasing family
   so the miner isolates rather than blends them.

Reads the two exports, writes one parquet plus an audit to stdout.

    python tools/prepare_assistant_corpus.py \
        --head ~/Downloads/ai助手_Top1000query.xlsx \
        --tail ~/Downloads/ai助手_随机1000query.xlsx \
        -o data/raw/ai_assistant_pooled.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

#: A bare acknowledgement or continuation token, optionally wrapped in the
#: emoji the assistant's own suggested-reply chips carry. Deliberately anchored
#: at BOTH ends: `好的` is an ack, `好的，帮我写一份年终总结` is a request, and a
#: substring match cannot tell them apart. Printed by `--show-ack` so the list
#: can be argued with.
ACK_PATTERN = (
    r"^[\s👌🆗👍😊🙏]*"
    r"(嗯+|好的?|行|是的?|对|要|需要|可以|继续|OK|ok|Ok|yes|谢谢|没有|不用|知道了?|明白)"
    r"[\s，,。！!、~～]*(继续吧|继续|吧|的|了|啊|呀)?[\s。！!~～]*$"
)

REQUIRED = ["normalized_query", "query_1st_category_new", "query_2nd_category_new", "search_num"]

#: NAME THE MEASUREMENT, NOT THE CONCLUSION — and these tags name the SAMPLING.
#:
#: The first draft called them `head` and `tail`. `tail` is a conclusion about
#: where those rows sit in the traffic distribution; the file is a RANDOM sample,
#: which is a statement about how it was drawn. The distinction is not pedantry:
#: only 0.35% of the random rows actually fall inside the top-1000 band, so the
#: two ARE nearly disjoint — but that is a measured property of this export, not
#: something the sampling guarantees, and a reader given the name `tail` cannot
#: tell which of the two they were told.
#:
#: Downstream the tag appears in the delivered table, the stratum comparison and
#: every share in it, so it is read far more often than it is written.
HEAD_TAG = "top1k"
RANDOM_TAG = "random1k"


def _read(path: str, stratum: str) -> pd.DataFrame:
    p = Path(path).expanduser()
    df = pd.read_excel(p) if p.suffix in (".xlsx", ".xls") else pd.read_csv(p)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        sys.exit(f"{p.name}: missing column(s) {missing}; it has {list(df.columns)}")
    df = df[REQUIRED].copy()
    df["stratum"] = stratum
    return df


def build(head: str, tail: str, weight_scale: float = 1000.0) -> tuple[pd.DataFrame, dict]:
    raw = pd.concat([_read(head, HEAD_TAG), _read(tail, RANDOM_TAG)], ignore_index=True)
    audit: dict = {"rows_in": len(raw)}

    raw["query"] = raw["normalized_query"].astype(str).str.strip()
    # An empty query is not a query. p1 would drop these anyway and say so; doing
    # it here keeps the row counts in this audit and in the run agreeing.
    blank = raw["query"].eq("") | raw["normalized_query"].isna()
    audit["dropped_blank"] = int(blank.sum())
    raw = raw[~blank]

    raw = raw.rename(columns={"query_1st_category_new": "l1", "query_2nd_category_new": "l2"})

    # How many first-level categories does each string touch, per stratum? This is
    # the measurement behind "content-free": a string in 32 of 33 topical
    # categories is not topical, whatever it says.
    n_l1 = raw.groupby(["stratum", "query"])["l1"].transform("nunique")
    raw["n_l1_categories"] = n_l1

    # Exact duplicate rows only — see decision 2 in the module docstring.
    grouped = (raw.groupby(["stratum", "query", "l1", "l2"], as_index=False)
                  .agg(search_num=("search_num", "sum"),
                       n_l1_categories=("n_l1_categories", "max")))
    audit["rows_after_exact_dedup"] = len(grouped)
    audit["exact_duplicate_rows_collapsed"] = len(raw) - len(grouped)

    # Within (stratum, category) to `weight_scale`. Each category then carries the
    # same traffic mass in each stratum, which is what the equal-allocation export
    # design already assumes and what makes the 33 incomparable PV floors harmless.
    cell = grouped.groupby(["stratum", "l1"])["search_num"].transform("sum")
    grouped["pv_norm"] = grouped["search_num"] / cell * weight_scale

    grouped["is_ack"] = grouped["query"].str.match(ACK_PATTERN, na=False)
    grouped["query_len"] = grouped["query"].str.len()

    out = grouped[["query", "l1", "l2", "stratum", "search_num", "pv_norm",
                   "n_l1_categories", "is_ack", "query_len"]].reset_index(drop=True)

    for s, g in out.groupby("stratum"):
        audit[f"{s}_rows"] = len(g)
        audit[f"{s}_pv_raw"] = int(g.search_num.sum())
        audit[f"{s}_ack_rows"] = int(g.is_ack.sum())
        audit[f"{s}_ack_pv_share_raw"] = round(
            float(g.loc[g.is_ack, "search_num"].sum() / g.search_num.sum()), 4)
        audit[f"{s}_ack_pv_share_norm"] = round(
            float(g.loc[g.is_ack, "pv_norm"].sum() / g.pv_norm.sum()), 4)
    audit["l1_categories"] = int(out.l1.nunique())
    audit["l2_categories"] = int(out.l2.nunique())
    return out, audit


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--head", required=True, help="the Top-N-by-search_num export")
    ap.add_argument("--tail", required=True, help="the random-sample export")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--weight-scale", type=float, default=1000.0)
    ap.add_argument("--show-ack", action="store_true",
                    help="print the acknowledgement family and the pattern that defines it")
    a = ap.parse_args()

    df, audit = build(a.head, a.tail, a.weight_scale)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False) if out.suffix == ".parquet" else df.to_csv(out, index=False)

    print(f"wrote {out}  ({len(df):,} rows)")
    print()
    for k, v in audit.items():
        print(f"  {k:<32} {v:,}" if isinstance(v, int) else f"  {k:<32} {v}")
    print()
    print("  PV floor per category in the HEAD stratum (why raw search_num is not")
    print("  comparable across categories):")
    fl = df[df.stratum == HEAD_TAG].groupby("l1").search_num.min().sort_values()
    print(f"    min {fl.min():,} ({fl.idxmin()})   max {fl.max():,} ({fl.idxmax()})"
          f"   median {fl.median():,.0f}")
    print()
    print("  Top 8 by raw PV share vs normalised PV share, head stratum:")
    h = df[df.stratum == HEAD_TAG]
    cmp = pd.DataFrame({"raw%": h.groupby("l1").search_num.sum() / h.search_num.sum() * 100,
                        "norm%": h.groupby("l1").pv_norm.sum() / h.pv_norm.sum() * 100})
    print(cmp.nlargest(8, "raw%").round(2).to_string().replace("\n", "\n    ").rjust(4))

    if a.show_ack:
        print()
        print(f"  ACK_PATTERN = {ACK_PATTERN}")
        acks = df[df.is_ack & (df.stratum == HEAD_TAG)].nlargest(20, "search_num")
        print(acks[["query", "l1", "search_num", "n_l1_categories"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
