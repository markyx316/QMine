#!/usr/bin/env python
"""Build the five per-domain POOLED-5 corpora: 2025 search + 2026 search + assistant head +
assistant tail + (finance/medical) assistant voice, in ONE file per domain so one taxonomy
labels every source.

WHY POOLED AND NOT FIVE RUNS. Measured on this project: two runs over the SAME 10,000 finance
rows shared 0 of 35 class codes. Labels from separate runs are not joinable, so a
source-by-source comparison has to come from one run.

WHAT EACH DECISION IS AND WHY.

1. `source` (5 values) is the analysis key and rides along for the POSITIONAL re-join after the
   run — `labels_full.csv` preserves input order exactly (verified 20,000/20,000 on fin-pool).
   `surface` (2 values: 搜索 / AI助手) is what `snapshot_column` points at, because
   `ops/drift.label_drift` compares EXACTLY two groups and skips with a reason on more.
2. `comparison_axis: stratum`, not time. The two groups differ by SURFACE, not by date; the
   time branch's prose ("同月同日不等于季节可比") would be false here, and one of its caveats
   inverts (see .claude/rules/multi-snapshot.md).
3. WEIGHTS ARE NORMALISED WITHIN SOURCE. Search PV (wise_pv, 10k head rows per year) and
   assistant PV (search_num, 33 categories each with its own floor) are different instruments,
   and the voice export carries no traffic at all. Raw PV pooled across them would make every
   weighted metric a statement about search. `pv_norm` sends each source to 10,000, so a
   traffic share reads "share of that source's own traffic"; voice is uniform, which makes its
   traffic share equal to its row share, and that is stated rather than hidden.
4. CLEANING FLAGS, IT DOES NOT DELETE. Assistant tiers are reused verbatim from the audited v3
   cleaning (`analysis/sva2026/work/clean_v3`), so this corpus and the published report agree
   row for row. Search rows get the same C1/S5 rules the search side got there. Voice rows get
   the content-free rules only — no PV means no PV-gated rule can fire. Only `tier == user`
   rows go into the mining input; everything else is written to the `_all_rows` table with its
   tier, so any figure can be recomputed with the system text put back.
5. Duplicate strings are NOT collapsed. The same string in two sources is the overlap this
   study measures; the voice export's 115 repeats are its only frequency signal.
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import clean_assistant_functional as C  # noqa: E402

OUT_MINE = ROOT / "data/raw/pooled5"
OUT_WORK = Path(__file__).resolve().parent / "work"
DOMAINS = {  # search-file domain -> assistant l1 in ai_assistant_pooled.parquet
    "金融": "金融", "医疗": "医疗", "教育": "教育培训", "影视": "影视动漫", "人物": "人物",
    "书籍文档": "书籍文档", "软件": "软件",
}
#: 2026-09-13 新增的两个垂类，搜索导出的文件名不是 `<domain>query-<yy>.xlsx` 那一套。
#: 助手侧**不需要**另开一条路：这两个 l1 本来就在 `ai_assistant_pooled.parquet` 与 v3 清洗的
#: `assistant_tiered.parquet` 里（逐条比对新导出的 xlsx：query 集合完全相同，search_num 与 l2
#: 一致率 0.987–1.000，差异只来自重复串取首行），所以走同一条已审计的路，结果与已交付的五域可比。
SEARCH_FILE = {
    "书籍文档": {"250701": "25 书籍文档wise.xlsx", "260701": "26 书籍文档wise.xlsx"},
    "软件": {"250701": "25 软件wise.xlsx", "260701": "26 软件wise.xlsx"},
}
VOICE = {"金融": "financial voice.xlsx", "医疗": "medical voice.xlsx"}
SURF = {"2025search": "搜索", "2026search": "搜索",
        "assistant_top": "AI助手", "assistant_random": "AI助手", "assistant_voice": "AI助手"}


EMPTY_DROPPED: list[dict] = []


def _drop_empty(d: pd.DataFrame, col: str, where: str) -> pd.DataFrame:
    """An empty text cell is not a query. p1 drops it too (`edu-pool` halted on one at row
    17,717 before that was added); dropping it here keeps our row ids and p1's in step."""
    bad = d[col].isna() | (d[col].astype("string").fillna("").str.strip() == "")
    if int(bad.sum()):
        EMPTY_DROPPED.append({"where": where, "n": int(bad.sum())})
    return d[~bad].reset_index(drop=True)


def _search(domain: str, yy: str, source: str) -> pd.DataFrame:
    fn = SEARCH_FILE.get(domain, {}).get(yy, f"{domain}query-{yy}.xlsx")
    d = _drop_empty(pd.read_excel(ROOT / f"data/raw/{fn}"),
                    "original_query", f"{domain}/{source}")
    q = d["original_query"].astype(str)
    out = pd.DataFrame({"query": q, "source": source, "domain": domain,
                        "l2": pd.NA, "pv_raw": d["wise_pv"].astype(float)})
    c1 = C.content_free_flags(q, surface="search")
    s5 = C.headline_flags(q, out["pv_raw"])
    out["tier"] = "user"
    out.loc[s5.values, "tier"] = "S5_headline"
    out.loc[c1.values, "tier"] = "C1_content_free"      # C1 last: it is the narrower rule here
    return out


def _assistant(domain: str, l1: str, tiers: pd.DataFrame) -> pd.DataFrame:
    p = pd.read_parquet(ROOT / "data/raw/ai_assistant_pooled.parquet")
    p = p[p.l1 == l1]
    rows = []
    for stratum, source in (("head", "assistant_top"), ("tail", "assistant_random")):
        s = p[p.stratum == stratum].copy()
        t = tiers[(tiers.l1 == l1) & (tiers.snapshot == ("top1k" if stratum == "head" else "random1k"))]
        t = t[["query", "l2", "tier"]].drop_duplicates(subset=["query", "l2"])
        m = s.merge(t, on=["query", "l2"], how="left", validate="m:1")
        assert m.tier.notna().all(), f"{l1}/{stratum}: {int(m.tier.isna().sum())} rows without a tier"
        rows.append(pd.DataFrame({"query": m["query"].astype(str), "source": source, "domain": domain,
                                  "l2": m["l2"], "pv_raw": m["search_num"].astype(float), "tier": m["tier"]}))
    return pd.concat(rows, ignore_index=True)


def _voice(domain: str, f: str) -> pd.DataFrame:
    d = _drop_empty(pd.read_excel(ROOT / f"data/raw/{f}"), "original_query", f"{domain}/voice")
    q = d["original_query"].astype(str)
    out = pd.DataFrame({"query": q, "source": "assistant_voice", "domain": domain,
                        "l2": pd.NA, "pv_raw": float("nan")})
    out["tier"] = "user"
    out.loc[C.content_free_flags(q, surface="assistant").values, "tier"] = "C1_content_free"
    out.loc[q.isin(C.FEATURE_ENTRIES).values, "tier"] = "S2_system_template"
    return out


def main() -> int:
    C._check_engine_semantics()
    OUT_MINE.mkdir(parents=True, exist_ok=True)
    OUT_WORK.mkdir(parents=True, exist_ok=True)
    tiers = pd.read_parquet(ROOT / "analysis/sva2026/work/clean_v3/assistant_tiered.parquet")
    report = []
    for domain, l1 in DOMAINS.items():
        parts = [_search(domain, "250701", "2025search"), _search(domain, "260701", "2026search"),
                 _assistant(domain, l1, tiers)]
        if domain in VOICE:
            parts.append(_voice(domain, VOICE[domain]))
        allrows = pd.concat(parts, ignore_index=True)
        allrows["surface"] = allrows["source"].map(SURF)
        allrows["kept"] = allrows["tier"] == "user"
        allrows["row_id_all"] = range(len(allrows))
        mine = allrows[allrows.kept].reset_index(drop=True).copy()
        # within-source normalisation to 10,000; voice (no PV) becomes uniform
        pv = mine["pv_raw"].copy()
        pv[mine.source == "assistant_voice"] = 1.0
        tot = pv.groupby(mine["source"]).transform("sum")
        mine["pv_norm"] = (pv / tot * 10_000).astype(float)
        mine["row_id"] = range(len(mine))
        cols = ["query", "source", "surface", "domain", "l2", "pv_raw", "pv_norm", "tier", "row_id", "row_id_all"]
        mine[cols].to_parquet(OUT_MINE / f"{domain}_pooled5.parquet", index=False)
        allrows.to_parquet(OUT_WORK / f"{domain}_all_rows.parquet", index=False)
        for src, g in allrows.groupby("source", sort=False):
            report.append({"domain": domain, "source": src, "rows": len(g), "kept": int(g.kept.sum()),
                           "dropped": int((~g.kept).sum()),
                           "drop_%": round(100 * (~g.kept).mean(), 2),
                           "pv_dropped_%": round(100 * g.loc[~g.kept, "pv_raw"].sum()
                                                 / max(g["pv_raw"].sum(), 1), 2) if g["pv_raw"].notna().any() else 0.0})
        print(f"{domain}: mining rows {len(mine):,} of {len(allrows):,}")
    if EMPTY_DROPPED:
        print("dropped empty text cells:", EMPTY_DROPPED)
    r = pd.DataFrame(report)
    r.to_csv(OUT_WORK / "build_audit.csv", index=False)
    print(r.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
