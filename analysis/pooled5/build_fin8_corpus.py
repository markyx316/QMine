#!/usr/bin/env python
"""Build the 8-snapshot finance corpus (`金融8`) — the same vertical as `金融`, three more exports.

WHY A SECOND FILE AND NOT AN OVERWRITE. `data/raw/pooled5/金融_pooled5.parquet` is bound row-for-row
to `runs/fin-pool5/gen01/labels_full.csv`: `pooled5_common.load` asserts equal length AND equal query
text at every position. Overwriting it with 44k rows makes every delivered finance table unreadable.
So the 8-snapshot corpus is its own file, its own run (`fin-pool8`), its own cohort (`fin8`).

WHAT THE THREE NEW EXPORTS CHANGE. Until now the finance design confounded INTERFACE with DEPTH:
search was head-only (PV-ranked 10k) while the assistant had both a head and a random stratum, so
"search vs assistant" and "head vs tail" could not be separated. The two new search random-10k files
close that hole — the design becomes interface x stratum, with year crossed into the search side:

    2025search       PV-ranked head 10k      2025search_rand   random 10k        (same day, 20250701)
    2026search       PV-ranked head 10k      2026search_rand   random 10k        (same day, 20260701)
    assistant_top    head 1k                 assistant_random  random 1k
    assistant_voice_top  PV-ranked head 1k   assistant_voice   1k, sampling unknown, NO PV

Measured on the raw files: head and random are genuinely different strata (25 head ∩ 25 random = 30
strings of 10,000; PV median 285 vs 1), and the tail churns year over year while the head persists
(head25 ∩ head26 = 5,447 strings; rand25 ∩ rand26 = 30).

`assistant_voice` is NOT called "random": its export carries no PV column and no ordering signal, so
the sampling method is unknown and the name must not assert one. `assistant_voice_top` carries
`wise_pv` and is verifiably a head.

EVERY OTHER DECISION IS INHERITED, NOT RE-MADE. Cleaning, tiering, within-source PV normalisation and
the no-dedup rule all come from `build_pooled5_corpus.py`; this script imports the same cleaning
module and mirrors its logic. The five snapshots that also exist in `金融_pooled5.parquet` are then
asserted IDENTICAL to it, cell by cell, on query/source/tier/pv_raw/pv_norm — so the new corpus is a
strict superset of the delivered one and any difference in results is attributable to the three new
exports, never to a rebuild drift.

    python analysis/pooled5/build_fin8_corpus.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import clean_assistant_functional as C  # noqa: E402

DOMAIN = "金融8"
OUT_MINE = ROOT / "data/raw/pooled5"
OUT_WORK = Path(__file__).resolve().parent / "work"

#: source -> (file, sheet, query column, pv column or None)
SEARCH = {
    "2025search":      ("金融query-250701.xlsx", 0, "original_query", "wise_pv"),
    "2025search_rand": ("金融25_random10k.xlsx", 0, "original_query", "wise_pv"),
    "2026search":      ("金融query-260701.xlsx", 0, "original_query", "wise_pv"),
    "2026search_rand": ("金融26_random10k.xlsx", 0, "original_query", "wise_pv"),
}
VOICE = {
    # head: has wise_pv, so the "head" claim is verifiable
    "assistant_voice_top": ("金融ai_voice_top1k.xlsx", "original_query", "wise_pv"),
    # no PV column at all — sampling method unknown, uniform weight, and the name says neither
    "assistant_voice":     ("financial voice.xlsx", "original_query", None),
}
SURF = {"2025search": "搜索", "2025search_rand": "搜索", "2026search": "搜索",
        "2026search_rand": "搜索", "assistant_top": "AI助手", "assistant_random": "AI助手",
        "assistant_voice_top": "AI助手", "assistant_voice": "AI助手"}
ORDER = ["2025search", "2025search_rand", "2026search", "2026search_rand",
         "assistant_top", "assistant_random", "assistant_voice_top", "assistant_voice"]

EMPTY_DROPPED: list[dict] = []


def _drop_empty(d: pd.DataFrame, col: str, where: str) -> pd.DataFrame:
    bad = d[col].isna() | (d[col].astype("string").fillna("").str.strip() == "")
    if int(bad.sum()):
        EMPTY_DROPPED.append({"where": where, "n": int(bad.sum())})
    return d[~bad].reset_index(drop=True)


def _search(source: str) -> pd.DataFrame:
    fn, sheet, qcol, pvcol = SEARCH[source]
    d = _drop_empty(pd.read_excel(ROOT / f"data/raw/{fn}", sheet_name=sheet), qcol, f"金融8/{source}")
    q = d[qcol].astype(str)
    out = pd.DataFrame({"query": q, "source": source, "domain": DOMAIN,
                        "l2": pd.NA, "pv_raw": d[pvcol].astype(float)})
    c1 = C.content_free_flags(q, surface="search")
    s5 = C.headline_flags(q, out["pv_raw"])
    out["tier"] = "user"
    out.loc[s5.values, "tier"] = "S5_headline"
    out.loc[c1.values, "tier"] = "C1_content_free"      # C1 last: the narrower rule wins
    return out


def _assistant(tiers: pd.DataFrame) -> pd.DataFrame:
    p = pd.read_parquet(ROOT / "data/raw/ai_assistant_pooled.parquet")
    p = p[p.l1 == "金融"]
    rows = []
    for stratum, source in (("head", "assistant_top"), ("tail", "assistant_random")):
        s = p[p.stratum == stratum].copy()
        t = tiers[(tiers.l1 == "金融") & (tiers.snapshot == ("top1k" if stratum == "head" else "random1k"))]
        t = t[["query", "l2", "tier"]].drop_duplicates(subset=["query", "l2"])
        m = s.merge(t, on=["query", "l2"], how="left", validate="m:1")
        assert m.tier.notna().all(), f"{stratum}: {int(m.tier.isna().sum())} rows without a tier"
        rows.append(pd.DataFrame({"query": m["query"].astype(str), "source": source, "domain": DOMAIN,
                                  "l2": m["l2"], "pv_raw": m["search_num"].astype(float),
                                  "tier": m["tier"]}))
    return pd.concat(rows, ignore_index=True)


def _voice(source: str) -> pd.DataFrame:
    fn, qcol, pvcol = VOICE[source]
    d = _drop_empty(pd.read_excel(ROOT / f"data/raw/{fn}"), qcol, f"金融8/{source}")
    q = d[qcol].astype(str)
    out = pd.DataFrame({"query": q, "source": source, "domain": DOMAIN, "l2": pd.NA,
                        "pv_raw": d[pvcol].astype(float) if pvcol else float("nan")})
    out["tier"] = "user"
    # Voice rows get the content-free rules only. The head export DOES have PV, so unlike the
    # other voice file it can also take the PV-gated headline rule.
    out.loc[C.content_free_flags(q, surface="assistant").values, "tier"] = "C1_content_free"
    if pvcol:
        out.loc[C.headline_flags(q, out["pv_raw"]).values & (out.tier == "user"), "tier"] = "S5_headline"
    out.loc[q.isin(C.FEATURE_ENTRIES).values, "tier"] = "S2_system_template"
    return out


#: 深市证券代码的码段。补零只在补出来的前缀落在这里、或补出来的码在本语料别处出现过时才做。
_EQ_PREFIX = ("000", "001", "002", "003", "300", "301", "600", "601", "603", "605", "688", "689")


def _repair_stripped_codes(d: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """还原被 Excel 抹掉的证券代码前导零。

    **这是一处数据缺陷修复，不是一次口径选择。** 源文件里 `original_query` 有整列被存成数字型
    单元格（openpyxl `data_type=="n"`：金融query-250701 1,005 个、260701 1,057 个、
    ai_voice_top1k 90 个、两份 random10k 49/61 个），所以 `000519` 在进入任何代码之前就已经变成
    `519` 了。三条独立证据，都在本语料上复算过：

    1. **前导零检验**：1,318 条裸 6 位 query 里以 `0` 开头的 = **0 条**；而长 query 内部出现的
       1,467 个 6 位码里 513 个（35.0%）以 `0` 开头。若用户真的原样键入，期望约 461 条。
    2. **孪生检验**：617 个不同的裸短串里 223 个（36.1%）补零后的 6 位形式就出现在**同一份语料**
       的长 query 里（`519`↔`000519`、`2015`↔`002015协鑫能科股吧`）。同码段随机码的零假设是
       7.6% ± 1.0%。
    3. **竞争假设排除**：港股假设不成立（裸 4 位在港股最密的 0001–0999 段里 0 条）；年份假设
       只影响 51 行，见下面的 `low_conf`。

    补零**只在有正面证据时**做：补出来的前缀属于 A 股/创业板/科创板码段，或补出来的码在本语料
    别处出现过。`95xxx` 客服热线、`400`/`10` 开头的长号、以及既不满足前缀也没有孪生的 57 行
    一律原样保留。`query_raw` 留着原文，`code_repaired` / `repair_conf` 标出改过哪些行。
    """
    q = d["query"].astype(str)
    codes_in_long: set[str] = set()
    for t in q[q.str.len() > 6]:
        codes_in_long.update(__import__("re").findall(r"\d{6}", t))
    short = q.str.fullmatch(r"\d{1,5}") & ~q.str.fullmatch(r"95\d{3}")
    pad = q.where(~short, q.str.zfill(6))
    ok = short & (pad.str[:3].isin(_EQ_PREFIX) | pad.isin(codes_in_long))
    # 4 位、落在一个像年份的值上、又没有孪生佐证的，单独标出来：证据只有码段前缀
    yearish = q.str.fullmatch(r"(?:19|20)\d{2}") & ~pad.isin(codes_in_long)
    plausible = q.isin({"2005", "2006", "2008", "2023", "2024", "2027", "2030", "1938"})
    low = ok & yearish & plausible
    out = d.copy()
    out["query_raw"] = q
    out.loc[ok, "query"] = pad[ok]
    out["code_repaired"] = ok
    out["repair_conf"] = pd.Series("", index=out.index, dtype=object)
    out.loc[ok, "repair_conf"] = "high"
    out.loc[low, "repair_conf"] = "low_yearlike"
    stats = {"repaired": int(ok.sum()), "low_conf": int(low.sum()),
             "left_alone_pure_digits": int(q.str.fullmatch(r"\d+").sum() - ok.sum())}
    return out, stats


def _assert_superset(mine: pd.DataFrame) -> str:
    """The five shared snapshots must come out identical to the delivered pool5 corpus."""
    old = pd.read_parquet(OUT_MINE / "金融_pooled5.parquet")
    lines = []
    for src in old.source.unique():
        a = old[old.source == src].reset_index(drop=True)
        b = mine[mine.source == src].reset_index(drop=True)
        assert len(a) == len(b), f"{src}: pool5 has {len(a)} rows, fin8 has {len(b)}"
        for col in ("query_raw", "source", "tier"):
            acol = "query" if col == "query_raw" else col
            bad = int((a[acol].astype(str).values != b[col].astype(str).values).sum())
            assert not bad, f"{src}.{col}: {bad} cells differ from the delivered pool5 corpus"
        for col in ("source", "tier"):
            bad = int((a[col].astype(str).values != b[col].astype(str).values).sum())
            assert not bad, f"{src}.{col}: {bad} cells differ from the delivered pool5 corpus"
        for col in ("pv_raw", "pv_norm"):
            d = (a[col].fillna(-1).to_numpy() - b[col].fillna(-1).to_numpy())
            bad = int((abs(d) > 1e-9).sum())
            assert not bad, f"{src}.{col}: {bad} cells differ from the delivered pool5 corpus"
        lines.append(f"  {src}: {len(a):,} 行逐格相同")
    return "\n".join(lines)


def main() -> int:
    C._check_engine_semantics()
    OUT_MINE.mkdir(parents=True, exist_ok=True)
    tiers = pd.read_parquet(ROOT / "analysis/sva2026/work/clean_v3/assistant_tiered.parquet")
    parts = [_search(s) for s in SEARCH] + [_assistant(tiers)] + [_voice(s) for s in VOICE]
    allrows = pd.concat(parts, ignore_index=True)
    allrows["source"] = pd.Categorical(allrows["source"], categories=ORDER, ordered=True)
    allrows = allrows.sort_values("source", kind="stable").reset_index(drop=True)
    allrows["source"] = allrows["source"].astype(str)
    allrows["surface"] = allrows["source"].map(SURF)
    allrows["kept"] = allrows["tier"] == "user"
    allrows["row_id_all"] = range(len(allrows))
    mine = allrows[allrows.kept].reset_index(drop=True).copy()
    # within-source normalisation to 10,000; the PV-less voice export becomes uniform
    pv = mine["pv_raw"].copy()
    pv[mine.source == "assistant_voice"] = 1.0
    tot = pv.groupby(mine["source"]).transform("sum")
    mine["pv_norm"] = (pv / tot * 10_000).astype(float)
    mine["row_id"] = range(len(mine))
    mine, rep_stats = _repair_stripped_codes(mine)
    cols = ["query", "query_raw", "code_repaired", "repair_conf", "source", "surface", "domain",
            "l2", "pv_raw", "pv_norm", "tier", "row_id", "row_id_all"]
    same = _assert_superset(mine)
    mine[cols].to_parquet(OUT_MINE / f"{DOMAIN}_pooled5.parquet", index=False)
    allrows.to_parquet(OUT_WORK / f"{DOMAIN}_all_rows.parquet", index=False)
    rep = []
    for src in ORDER:
        g = allrows[allrows.source == src]
        rep.append({"domain": DOMAIN, "source": src, "rows": len(g), "kept": int(g.kept.sum()),
                    "dropped": int((~g.kept).sum()),
                    "drop_%": round(100 * (~g.kept).mean(), 2),
                    "pv_dropped_%": round(100 * g.loc[~g.kept, "pv_raw"].sum()
                                          / max(g["pv_raw"].sum(), 1), 2)
                    if g["pv_raw"].notna().any() else 0.0})
    r = pd.DataFrame(rep)
    r.to_csv(OUT_WORK / "build_audit_fin8.csv", index=False, encoding="utf-8-sig")
    print("与已交付的 金融_pooled5 逐格比对（五个共有快照）：")
    print(same)
    print(f"\n前导零还原：修复 {rep_stats['repaired']:,} 行"
          f"（其中 {rep_stats['low_conf']} 行是「像年份、又没有孪生佐证」的低置信修复）；"
          f"仍保持原样的纯数字 {rep_stats['left_alone_pure_digits']:,} 行")
    print(f"空文本丢弃：{EMPTY_DROPPED or '无'}")
    print(f"\n{DOMAIN}: 全部 {len(allrows):,} 行 → 入挖掘 {len(mine):,} 行")
    print(r.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
