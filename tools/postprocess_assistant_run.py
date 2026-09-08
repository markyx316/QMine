#!/usr/bin/env python
"""Re-sort and re-label a finished run's delivered tables, without re-running it.

WHY THIS IS A POST-RUN TOOL AND NOT A PIPELINE CHANGE.

Everything it does is either presentational or specific to how ONE export was
sampled, and neither belongs in code that has to be right on every corpus:

* **Sort order.** The delivered table comes out in corpus order, which for the
  pooled AI-assistant corpus is stratum-then-query — so it reads as alphabetical
  and a reader cannot see one category's rows together. Sorting by
  (category, sub-category, stratum, traffic) is what a reader of THIS export
  wants; it is not a claim about any other corpus.
* **Stratum names.** The prepared corpus tagged its two groups `head` and `tail`.
  `tail` is a CONCLUSION about where those rows sit in the traffic distribution;
  the file is a RANDOM sample, which is a statement about how it was drawn. The
  project's own rule is to name the measurement, so `top1k` / `random1k`.

What is NOT here, because it was general rather than corpus-specific: the
missing `snapshot` column in `labels_full.csv`. The drift/stratum document ends
「原始数据: labels_full.csv（逐行标签，含分层列）」 and that was false on all six
pooled runs on disk as well as `ai04` — so it was fixed in `p10` instead, where
every future pooled run gets it.

Non-destructive: writes into `<generation>/postprocessed/`, leaving the run's own
deliverables untouched as evidence.

    python tools/postprocess_assistant_run.py runs/ai04/gen01 \
        --rename head=top1k,tail=random1k
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

#: Columns joined back from the corpus frame. `snapshot` is the stratum tag,
#: `weight` the normalised traffic, `l1`/`l2` the source taxonomy.
_FROM_CORPUS = ["snapshot", "l1", "l2", "weight"]


def _load(run: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    lab_p, cor_p = run / "labels_full.csv", run / "corpus.parquet"
    for p in (lab_p, cor_p):
        if not p.exists():
            sys.exit(f"{p} not found — is that a run generation directory?")
    return pd.read_csv(lab_p), pd.read_parquet(cor_p)


def join_verified(labels: pd.DataFrame, corpus: pd.DataFrame) -> pd.DataFrame:
    """Attach the corpus columns BY POSITION, having checked that position means something.

    `p10` writes `labels_full` in corpus order, so the join is positional — and a
    positional join that is silently wrong produces a table where every row's
    category belongs to a different query. `backfill_drift` refuses a misaligned
    positional join for the same reason; this checks the same way, on the one
    column both frames carry.
    """
    if len(labels) != len(corpus):
        sys.exit(f"labels_full has {len(labels):,} rows against {len(corpus):,} in the "
                 f"corpus — refusing to join by position")
    mismatch = (labels["query"].astype(str).to_numpy()
                != corpus["query"].astype(str).to_numpy())
    if mismatch.any():
        sys.exit(f"{int(mismatch.sum()):,} rows disagree on `query` between "
                 f"labels_full and the corpus — the two are not in the same order, "
                 f"so a positional join would mislabel every row after the first "
                 f"divergence. Refusing.")
    out = labels.copy()
    for c in _FROM_CORPUS:
        if c in corpus.columns:
            out[c] = corpus[c].to_numpy()
    return out


def stratum_addendum(df: pd.DataFrame, col: str, head: str, rand: str,
                     raw: pd.DataFrame | None = None) -> str:
    """What the two strata ARE, and which column of the comparison to trust.

    MEASURED FROM THE RUN'S OWN CORPUS, not written down once. The shipped
    stratum vocabulary in `report/zh_drift.py` is deliberately frame-agnostic —
    it cannot know that one of these groups is a top-N census and the other a
    uniform sample — so the corpus-specific reading belongs here, beside the
    numbers that justify it.

    The finding this exists to state: the document's own advice is 「请按 Δ流量的
    大小读」, which is right when BOTH sides are censuses. Here one side is a
    1,000-row uniform sample, and a traffic share taken off it is dominated by
    whichever few high-traffic rows happened to be drawn.
    """
    import math

    h, r = df[df[col] == head], df[df[col] == rand]
    if h.empty or r.empty or "weight" not in df.columns:
        return ""

    # RAW TRAFFIC, NOT `weight`. `weight` is normalised WITHIN (stratum,
    # category) to a constant total, so a floor taken from one stratum and
    # compared against the other's values is comparing two different scales —
    # it returns "100% of the random sample sits above the head floor", which is
    # an artefact of the normalisation and the exact incomparability this whole
    # corpus preparation exists to handle. Every claim below that needs an
    # ACROSS-STRATUM traffic comparison is therefore computed from the source
    # export's raw counts, or omitted entirely.
    src = None
    if raw is not None and {"stratum", "l1", "search_num"} <= set(raw.columns):
        src = raw
    top1 = r.groupby("l1").apply(
        lambda g: g["weight"].max() / g["weight"].sum(), include_groups=False)
    ci = {p: 1.96 * math.sqrt(p * (1 - p) / max(len(r) / max(r["l1"].nunique(), 1), 1))
          for p in (0.05, 0.20, 0.50)}
    overlap_block: list[str] = []
    floor_txt = ("各垂类门槛并不相同，所以**跨垂类比较原始流量没有意义**。")
    rand_txt = ""
    if src is not None:
        sh = src[src.stratum == "head"]
        sr = src[src.stratum == "tail"]
        floors = sh.groupby("l1").search_num.min()
        j = sr.join(floors.rename("floor"), on="l1")
        n_in = int((j.search_num >= j.floor).sum())
        zero_cats = sum(1 for _c, g in j.groupby("l1")
                        if (g.search_num >= g.floor).sum() == 0)
        floor_txt = (f"各垂类门槛并不相同 —— 实测最低 **{int(floors.min())}**，"
                     f"最高 **{int(floors.max())}**，中位 {floors.median():.0f}（原始 PV）"
                     f"—— 所以**跨垂类比较原始流量没有意义**。")
        rand_txt = (f"实测该层原始 PV 的中位数 {sr.search_num.median():.0f}、"
                    f"均值 {sr.search_num.mean():.2f}；若按事件（PV）加权抽取，"
                    f"均值会远高于中位数。")
        overlap_block = [
            f"**两层的近乎不重叠是实测的，不是设计保证的。** 随机层中落在各自垂类"
            f"门槛之上的只有 **{n_in:,} / {len(sr):,} = {n_in / len(sr):.2%}**，"
            f"{sr.l1.nunique()} 个垂类里有 **{zero_cats}** 个一条都没有。所以它在事实上"
            f"等价于「头部之外」——但这是一个测量结果，换一份导出未必成立。", ""]
    else:
        overlap_block = [
            "> ⚠ 两层的重叠程度**本次未计算**：它需要跨分层比较**原始** PV，而运行产物里"
            "只有按 (分层, 垂类) 归一化后的 `weight`，两者不同尺度。传 "
            "`--source-corpus data/raw/....parquet` 可以补上这个数字。", ""]

    return "\n".join([
        "", "---", "",
        "## 附录：这两个分层到底是什么，以及哪一列可信",
        "",
        "> 本附录由 `tools/postprocess_assistant_run.py` 从本次运行的语料直接算出。"
        "正文的分层措辞是**与口径无关**的通用表述 —— 它无法知道这两组具体是怎么抽的 ——"
        "所以口径相关的读法写在这里，紧挨着支撑它的数字。",
        "",
        f"**`{head}`** —— 每个垂类按流量取前 1,000 条。这是一次**普查**而非抽样："
        f"在该垂类自己的流量门槛之上，它是完整的。{floor_txt}",
        "",
        f"**`{rand}`** —— 每个垂类从**去重 query** 中均匀随机抽 1,000 条。{rand_txt}"
        f"它估计的是该垂类**「问过哪些东西」的构成**，不是**「流量落在哪里」**的构成。",
        "",
        *overlap_block,
        "### 哪一列可信 —— 这一点与正文的读法建议相反",
        "",
        "正文写着「请按 Δ流量的大小读，不要按 z 读」。那条建议是为**两边都是普查**的"
        "时间对比写的。在这里两边的性质不同：",
        "",
        f"| 列 | `{head}` 侧 | `{rand}` 侧 |",
        "|---|---|---|",
        "| 行占比 | 可信（普查） | 可信，均匀抽样的无偏估计 —— "
        + "；".join(f"{int(p*100)}% 的类 ±{c*100:.1f}pp" for p, c in ci.items()) + " |",
        f"| 流量占比 | 可信（普查） | **高方差** —— 单条 query 占该垂类样本流量的"
        f"中位数 {top1.median():.1%}，最高 {top1.max():.1%}"
        f"（{top1.idxmax()}），{int((top1 > 0.10).sum())}/{len(top1)} 个垂类里"
        f"有一条 query 超过 10% |",
        "",
        "**所以在这份报告里请按「行占比」读，Δ流量 只作参考。**",
        "",
        "### 正文的「不能推回总体」需要收窄",
        "",
        "那条告诫过强，准确的说法分三种情况：",
        "",
        f"- **可以**：在**一个垂类内部**，`{rand}` 的行占比是该垂类 query 词表构成的"
        "无偏估计，精度见上表。",
        "- **不可以**：跨垂类合并成全量估计 —— 每个垂类抽取的行数相同而真实规模未知，"
        "且捕获-再捕获桥接在这份数据上失败。",
        f"- **不可以**：把 `{rand}` 的流量占比当作总体流量构成 —— 见上表的高方差。",
        "",
    ])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir")
    ap.add_argument("--rename", default="head=top1k,tail=random1k",
                    help="comma-separated old=new stratum labels")
    ap.add_argument("--sort-by", default="l1,l2,snapshot",
                    help="columns to sort by, before descending traffic")
    ap.add_argument("--stratum-column", default="snapshot")
    ap.add_argument("--source-corpus", default=None,
                    help="the prepared parquet, for the RAW traffic counts the "
                         "run artifacts do not keep (they hold only the "
                         "within-stratum-normalised weight)")
    ap.add_argument("--stratum-order", default="top1k,random1k",
                    help="stratum values in the order they should appear; "
                         "alphabetical would put random1k first, which is not "
                         "the order anyone reads these two in")
    a = ap.parse_args()

    run = Path(a.run_dir)
    labels, corpus = _load(run)
    df = join_verified(labels, corpus)

    col = a.stratum_column
    mapping = dict(kv.split("=", 1) for kv in a.rename.split(",") if "=" in kv)
    if col in df.columns and mapping:
        before = sorted(df[col].dropna().unique())
        unknown = [v for v in before if v not in mapping]
        if unknown:
            print(f"  ⚠ stratum value(s) {unknown} have no rename and are kept as-is")
        df[col] = df[col].map(lambda v: mapping.get(str(v), v))
        print(f"  stratum {col!r}: {before} -> {sorted(df[col].dropna().unique())}")

    # THE STRATUM ORDER IS STATED, NOT INHERITED FROM THE ALPHABET. Sorting
    # these two by name puts `random1k` before `top1k`, which is backwards for
    # every reader: the head band is what you look at first and the random sample
    # is what you check it against.
    order = [v for v in a.stratum_order.split(",") if v]
    if col in df.columns and order:
        seen = list(df[col].dropna().unique())
        df[col] = pd.Categorical(df[col], categories=order + [v for v in seen
                                                             if v not in order],
                                 ordered=True)
    sort_cols = [c for c in a.sort_by.split(",") if c in df.columns]
    if "weight" in df.columns:
        df = df.sort_values(sort_cols + ["weight"],
                            ascending=[True] * len(sort_cols) + [False])
    else:
        df = df.sort_values(sort_cols)
    if col in df.columns:
        df[col] = df[col].astype(str)          # categorical -> plain text for CSV/xlsx
    df = df.reset_index(drop=True)
    print(f"  sorted by {sort_cols} then descending traffic; "
          f"stratum order {order}")

    outdir = run / "postprocessed"
    outdir.mkdir(exist_ok=True)
    df.to_csv(outdir / "labels_full_按垂类排序.csv", index=False, encoding="utf-8-sig")

    # Rewrite the workbook: the labelled sheet replaced, every other sheet copied
    # through untouched so the deliverable stays whole.
    src = next((p for p in run.glob("*_query_挖掘结果.xlsx")), None)
    if src is not None:
        book = pd.ExcelFile(src)
        dst = outdir / src.name
        with pd.ExcelWriter(dst, engine="openpyxl") as w:
            for sheet in book.sheet_names:
                (df if sheet == "全量标注" else book.parse(sheet)).to_excel(
                    w, sheet_name=sheet, index=False)
        print(f"  wrote {dst}  ({len(book.sheet_names)} sheets, 全量标注 re-sorted)")
    else:
        print("  ⚠ no *_query_挖掘结果.xlsx found; wrote the CSV only")

    print(f"  wrote {outdir / 'labels_full_按垂类排序.csv'}")

    # The stratum document, regenerated with the corrected names and the
    # corpus-specific reading appended. Rebuilt from `drift_analysis.json` — no
    # model call, so this is a rendering step and not a second opinion.
    dj = run / "drift_analysis.json"
    src_md = next((q for q in run.glob("分层对比*.md")), None)
    if dj.exists() and src_md is not None:
        md = src_md.read_text(encoding="utf-8")
        for old_v, new_v in mapping.items():
            md = md.replace(f"`{old_v}`", f"`{new_v}`").replace(f" {old_v} ", f" {new_v} ")
        # the false pointer: labels_full.csv only carries the stratum column for
        # runs made after the p10 fix, and this run predates it
        md = md.replace("`labels_full.csv`（逐行标签，含分层列）",
                        "`postprocessed/labels_full_按垂类排序.csv`"
                        "（逐行标签，已补回分层列并按垂类排序）")
        head_v = mapping.get("head", "head")
        rand_v = mapping.get("tail", "tail")
        _raw = (pd.read_parquet(a.source_corpus) if a.source_corpus else None)
        md += stratum_addendum(df, col, head_v, rand_v, raw=_raw)
        (outdir / src_md.name).write_text(md, encoding="utf-8")
        print(f"  wrote {outdir / src_md.name}  (renamed + addendum)")
    if col in df.columns:
        print()
        print(df.groupby([c for c in ("l1", col) if c in df.columns])
                .size().head(6).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
