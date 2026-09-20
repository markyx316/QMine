#!/usr/bin/env python
"""Build the 8-snapshot medical corpus (`医疗8`) — the same vertical as `医疗`, three more exports.

WHY A SECOND FILE AND NOT AN OVERWRITE. `data/raw/pooled5/医疗_pooled5.parquet` is bound row-for-row
to `runs/med-pool5/gen01/labels_full.csv`: `pooled5_common.load` asserts equal length AND equal query
text at every position. So the 8-snapshot corpus is its own file, its own run (`med-pool8`), its own
cohort (`med8`) — exactly as `金融8` was built beside `金融` (`build_fin8_corpus.py`).

WHAT THE THREE NEW EXPORTS CHANGE. Search was head-only in `医疗`, so "search vs assistant" and
"head vs tail" were confounded. The two search random-10k files cross interface with stratum, and the
voice head gives the voice side a PV-ranked head beside the export of unknown sampling:

    2025search       PV-ranked head 10k      2025search_rand   random 10k        (same day, 20250701)
    2026search       PV-ranked head 10k      2026search_rand   random 10k        (same day, 20260701)
    assistant_top    head 1k                 assistant_random  random 1k
    assistant_voice_top  PV-ranked voice head    assistant_voice   1k, sampling unknown, NO PV

Measured on the raw files (2026-09-15): head and random are genuinely different strata
(head25 ∩ rand25 = 26 strings, head26 ∩ rand26 = 4; PV median 604 / 303 vs 1); the head persists
year over year (head25 ∩ head26 = 6,329) while the tail churns (rand25 ∩ rand26 = 2). Every query cell in
all six medical exports is text — no leading-zero damage, so `build_fin8_corpus.py`'s code repair is
not needed and not run.

THE VOICE HEAD IS CUT AT THE FIRST TIE-FREE BOUNDARY, NOT AT ROW 1,000. `医疗ai_voice_top1k.xlsx`
holds 10,000 rows sorted by `wise_pv` descending (the finance export of the same name held 1,000).
Row 1,000 has PV 11, and 152 rows share PV 11 across rows 916-1,067, so "the first 1,000 rows" would
admit 85 of those tied rows and reject 67 on export order alone. The stratum is therefore every row
with PV >= the PV of row 1,000 — the smallest head containing the top 1,000 that needs no tie-break
(1,067 rows). Rows below it are outside the design and are not mined; the builder asserts the file is
PV-sorted and that every excluded row has a strictly lower PV.

EVERY OTHER DECISION IS INHERITED, NOT RE-MADE. Cleaning, tiering, within-source PV normalisation and
the no-dedup rule come from `build_pooled5_corpus.py` (same cleaning module, same order of rules), and
the voice head takes the PV-gated headline rule exactly as `金融8`'s did. The five snapshots that also
exist in `医疗_pooled5.parquet` are asserted IDENTICAL to it, cell by cell, on
query/source/surface/l2/tier/pv_raw/pv_norm — so the new corpus is a strict superset of the delivered
one, and any difference in results is attributable to the three new exports, never to a rebuild.

    python analysis/pooled5/build_med8_corpus.py
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import clean_assistant_functional as C  # noqa: E402

DOMAIN = "医疗8"
POOL5 = "医疗"
ASSISTANT_L1 = "医疗"
OUT_MINE = ROOT / "data/raw/pooled5"
OUT_WORK = Path(__file__).resolve().parent / "work"

#: source -> (file, query column, pv column)
SEARCH = {
    "2025search":      ("医疗query-250701.xlsx", "original_query", "wise_pv"),
    "2025search_rand": ("25医疗搜索随机1w.xlsx", "original_query", "wise_pv"),
    "2026search":      ("医疗query-260701.xlsx", "original_query", "wise_pv"),
    "2026search_rand": ("26医疗搜索随机1w.xlsx", "original_query", "wise_pv"),
}
VOICE = {
    # head: has wise_pv and is PV-sorted, so the "head" claim is verifiable
    "assistant_voice_top": ("医疗ai_voice_top1k.xlsx", "original_query", "wise_pv"),
    # no PV column at all — sampling method unknown, uniform weight, and the name says neither
    "assistant_voice":     ("medical voice.xlsx", "original_query", None),
}
VOICE_HEAD_TOP = 1000
SURF = {"2025search": "搜索", "2025search_rand": "搜索", "2026search": "搜索",
        "2026search_rand": "搜索", "assistant_top": "AI助手", "assistant_random": "AI助手",
        "assistant_voice_top": "AI助手", "assistant_voice": "AI助手"}
ORDER = ["2025search", "2025search_rand", "2026search", "2026search_rand",
         "assistant_top", "assistant_random", "assistant_voice_top", "assistant_voice"]

EMPTY_DROPPED: list[dict] = []
FACTS: dict = {}


def _drop_empty(d: pd.DataFrame, col: str, where: str) -> pd.DataFrame:
    bad = d[col].isna() | (d[col].astype("string").fillna("").str.strip() == "")
    if int(bad.sum()):
        EMPTY_DROPPED.append({"where": where, "n": int(bad.sum())})
    return d[~bad].reset_index(drop=True)


def _search(source: str) -> pd.DataFrame:
    fn, qcol, pvcol = SEARCH[source]
    d = _drop_empty(pd.read_excel(ROOT / f"data/raw/{fn}"), qcol, f"{DOMAIN}/{source}")
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
    p = p[p.l1 == ASSISTANT_L1]
    rows = []
    for stratum, source in (("head", "assistant_top"), ("tail", "assistant_random")):
        s = p[p.stratum == stratum].copy()
        t = tiers[(tiers.l1 == ASSISTANT_L1) & (tiers.snapshot == ("top1k" if stratum == "head" else "random1k"))]
        t = t[["query", "l2", "tier"]].drop_duplicates(subset=["query", "l2"])
        m = s.merge(t, on=["query", "l2"], how="left", validate="m:1")
        assert m.tier.notna().all(), f"{stratum}: {int(m.tier.isna().sum())} rows without a tier"
        rows.append(pd.DataFrame({"query": m["query"].astype(str), "source": source, "domain": DOMAIN,
                                  "l2": m["l2"], "pv_raw": m["search_num"].astype(float),
                                  "tier": m["tier"]}))
    return pd.concat(rows, ignore_index=True)


def _voice_head(d: pd.DataFrame, pvcol: str) -> pd.DataFrame:
    """Every row with PV >= the PV of row VOICE_HEAD_TOP: the smallest tie-free head containing the top N."""
    pv = d[pvcol].astype(float)
    assert (pv.diff().dropna() <= 0).all(), "voice head export is not sorted by PV descending"
    assert len(d) >= VOICE_HEAD_TOP, f"voice head export has only {len(d)} rows"
    cut = float(pv.iloc[VOICE_HEAD_TOP - 1])
    head = d[pv >= cut].reset_index(drop=True)
    rest = pv[pv < cut]
    assert len(head) >= VOICE_HEAD_TOP and (rest < cut).all()
    FACTS["voice_head"] = {
        "file_rows": int(len(d)), "top_n": VOICE_HEAD_TOP, "cutoff_pv": cut,
        "rows_at_cutoff_pv": int((pv == cut).sum()),
        "first_row_at_cutoff": int((pv == cut).idxmax()) + 1,
        "last_row_at_cutoff": int(pv[pv == cut].index.max()) + 1,
        "head_rows": int(len(head)), "rows_not_mined": int(len(d) - len(head)),
        "head_pv_share_of_file_%": round(float(pv[pv >= cut].sum() / pv.sum() * 100), 2),
    }
    return head


def _voice(source: str) -> pd.DataFrame:
    fn, qcol, pvcol = VOICE[source]
    d = _drop_empty(pd.read_excel(ROOT / f"data/raw/{fn}"), qcol, f"{DOMAIN}/{source}")
    if pvcol:
        d = _voice_head(d, pvcol)
    q = d[qcol].astype(str)
    out = pd.DataFrame({"query": q, "source": source, "domain": DOMAIN, "l2": pd.NA,
                        "pv_raw": d[pvcol].astype(float) if pvcol else float("nan")})
    out["tier"] = "user"
    # Voice rows get the content-free rules. The head export DOES have PV, so unlike the other
    # voice file it can also take the PV-gated headline rule — the same order as 金融8.
    out.loc[C.content_free_flags(q, surface="assistant").values, "tier"] = "C1_content_free"
    if pvcol:
        out.loc[C.headline_flags(q, out["pv_raw"]).values & (out.tier == "user"), "tier"] = "S5_headline"
    out.loc[q.isin(C.FEATURE_ENTRIES).values, "tier"] = "S2_system_template"
    return out


def _assert_superset(mine: pd.DataFrame) -> str:
    """The five shared snapshots must come out identical to the delivered pool5 corpus."""
    old = pd.read_parquet(OUT_MINE / f"{POOL5}_pooled5.parquet")
    lines = []
    for src in old.source.unique():
        a = old[old.source == src].reset_index(drop=True)
        b = mine[mine.source == src].reset_index(drop=True)
        assert len(a) == len(b), f"{src}: pool5 has {len(a)} rows, med8 has {len(b)}"
        # MISSING IS COMPARED AS MISSING, NOT AS TEXT. The delivered parquet stores an absent `l2`
        # as the string-dtype NaN (repr 'nan') and this build holds pd.NA (repr '<NA>'), so a
        # plain astype(str) comparison reported all 9,998 search rows as different when not one
        # cell was: measured, 0 one-side-missing and 0 differing values in every column. A cell
        # missing on BOTH sides is equal; missing on ONE side, or a different value, still fails.
        for col in ("query", "source", "surface", "l2", "tier", "pv_raw", "pv_norm"):
            na_a, na_b = a[col].isna().to_numpy(), b[col].isna().to_numpy()
            one_side = int((na_a ^ na_b).sum())
            assert not one_side, f"{src}.{col}: {one_side} cells missing on one side only"
            both = ~na_a & ~na_b
            if col in ("pv_raw", "pv_norm"):
                bad = int((both & (abs(a[col].fillna(0).to_numpy() - b[col].fillna(0).to_numpy()) > 1e-9)).sum())
            else:
                bad = int((both & (a[col].astype(str).to_numpy() != b[col].astype(str).to_numpy())).sum())
            assert not bad, f"{src}.{col}: {bad} cells differ from the delivered pool5 corpus"
        lines.append(f"  {src}: {len(a):,} 行逐格相同")
    assert set(old.source.unique()) == set(ORDER) - {"2025search_rand", "2026search_rand", "assistant_voice_top"}
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
    assert allrows["surface"].notna().all()
    allrows["kept"] = allrows["tier"] == "user"
    allrows["row_id_all"] = range(len(allrows))
    mine = allrows[allrows.kept].reset_index(drop=True).copy()
    # within-source normalisation to 10,000; the PV-less voice export becomes uniform
    pv = mine["pv_raw"].copy()
    pv[mine.source == "assistant_voice"] = 1.0
    assert pv.notna().all(), "a source other than assistant_voice has a missing PV"
    tot = pv.groupby(mine["source"]).transform("sum")
    mine["pv_norm"] = (pv / tot * 10_000).astype(float)
    mine["row_id"] = range(len(mine))
    cols = ["query", "source", "surface", "domain", "l2", "pv_raw", "pv_norm", "tier", "row_id", "row_id_all"]
    same = _assert_superset(mine)
    assert list(mine.source.drop_duplicates()) == ORDER, list(mine.source.drop_duplicates())
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
    r.to_csv(OUT_WORK / "build_audit_med8.csv", index=False, encoding="utf-8-sig")

    # measured facts for the audit note and the report (counts only; no query text)
    kept_sets = {s: set(mine.loc[mine.source == s, "query"].astype(str)) for s in ORDER}
    FACTS["rows_all"] = int(len(allrows))
    FACTS["rows_mined"] = int(len(mine))
    FACTS["tiers_by_source"] = {s: allrows[allrows.source == s].tier.value_counts().to_dict() for s in ORDER}
    FACTS["distinct_mined_by_source"] = {s: len(v) for s, v in kept_sets.items()}
    FACTS["pv_median_by_source"] = {s: (None if s == "assistant_voice" else
                                        float(mine.loc[mine.source == s, "pv_raw"].median())) for s in ORDER}
    FACTS["overlap_distinct_mined"] = {f"{a}∩{b}": len(kept_sets[a] & kept_sets[b])
                                       for a, b in combinations(ORDER, 2)}
    FACTS["empty_dropped"] = EMPTY_DROPPED
    (OUT_WORK / "build_facts_med8.json").write_text(json.dumps(FACTS, ensure_ascii=False, indent=1),
                                                     encoding="utf-8")

    print("与已交付的 医疗_pooled5 逐格比对（五个共有快照）：")
    print(same)
    print(f"\n语音头部：{json.dumps(FACTS['voice_head'], ensure_ascii=False)}")
    print(f"空文本丢弃：{EMPTY_DROPPED or '无'}")
    print(f"\n{DOMAIN}: 全部 {len(allrows):,} 行 → 入挖掘 {len(mine):,} 行")
    print(r.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
