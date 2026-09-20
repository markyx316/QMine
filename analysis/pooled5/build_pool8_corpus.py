#!/usr/bin/env python
"""Build the 8-snapshot 人物 / 影视 corpora (`人物8`, `影视8`) — same verticals as `人物` / `影视`, four more exports each.

WHY NEW FILES AND NOT AN OVERWRITE. `data/raw/pooled5/{人物,影视}_pooled5.parquet` are bound row-for-row to
`runs/{ppl-pool5b,film-pool5}/gen01/labels_full.csv`: `pooled5_common.load` asserts equal length AND equal query
text at every position. So each 8-snapshot corpus is its own file, its own run, its own cohort — exactly as
`金融8` and `医疗8` were built beside `金融` / `医疗`.

WHAT THE FOUR NEW EXPORTS CHANGE. Both pool5 corpora had FOUR snapshots and search was head-only, so
"search vs assistant" and "head vs tail" were confounded, and neither domain had any voice at all. The two
search random-10k files cross interface with stratum, and the two voice exports add the input-mode axis with
a head and a random stratum of its own:

    2025search           PV-ranked head 10k    2025search_rand         random 10k      (same day, 20250701)
    2026search           PV-ranked head 10k    2026search_rand         random 10k      (same day, 20260701)
    assistant_top        head 1k               assistant_random        random 1k
    assistant_voice_top  PV-ranked voice head  assistant_voice_random  random voice 1k

Measured on the raw files (2026-09-16), head and random are genuinely different strata — 人物 head25 ∩ rand25 =
14 strings of 10,000, 影视 19; PV median 1 in every random file against a head whose top rows run to six
figures. The tail churns completely (rand25 ∩ rand26 = 0 in both domains) while the voice head overlaps the
search heads heavily (人物 643 / 692, 影视 346 / 700) — the same string can be typed and spoken.

THE VOICE HEAD IS THE WHOLE EXPORT, AND ITS TIE BOUNDARY IS UNKNOWABLE. Unlike 医疗8 — whose voice export held
10,000 PV-sorted rows, letting the builder cut at the first tie-free boundary at or below row 1,000 — these
exports arrive already truncated at exactly 1,000 PV-sorted rows. So the head is the file, and whether rows
just outside it share the PV of the last row cannot be checked from the data. The builder asserts what IS
checkable (PV present, sorted descending, exactly 1,000 rows) and records the cut PV and how many rows sit on
it, so the report can say plainly that the boundary is the export's, not a measured one.

BOTH VOICE EXPORTS CARRY PV, unlike 医疗8's second voice file (no PV, sampling unknown, which is why that one
could never be called random). Here the random voice export has PV with median 1 across both domains, so it
takes within-source normalisation like every other source and may be called what it is.

EVERY OTHER DECISION IS INHERITED, NOT RE-MADE. Cleaning, tiering, within-source PV normalisation and the
no-dedup rule come from `build_pooled5_corpus.py` (same cleaning module, same order of rules). The four
snapshots that also exist in the pool5 parquet are asserted IDENTICAL to it, cell by cell, on
query/source/surface/l2/tier/pv_raw/pv_norm — so each new corpus is a strict superset of the delivered one,
and any difference in results is attributable to the four new exports, never to a rebuild.

    python analysis/pooled5/build_pool8_corpus.py            # both domains
    python analysis/pooled5/build_pool8_corpus.py 人物8       # one
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

OUT_MINE = ROOT / "data/raw/pooled5"
OUT_WORK = Path(__file__).resolve().parent / "work"
VOICE_HEAD_ROWS = 1000

#: domain -> everything that differs between the two builds. `assistant_l1` is the l1 value inside
#: `ai_assistant_pooled.parquet` (影视 is stored as 影视动漫 there, as in `build_pooled5_corpus.py`).
SPEC = {
    "人物8": {"tag": "ppl8", "pool5": "人物", "assistant_l1": "人物", "head": "人物query-{yy}.xlsx",
             "rand": {"2025search_rand": "25人物搜索随机1w.xlsx", "2026search_rand": "26人物搜索随机1w.xlsx"},
             "voice": {"assistant_voice_top": "人物ai_voice_top1k.xlsx",
                       "assistant_voice_random": "人物ai_voice_random1k.xlsx"}},
    "影视8": {"tag": "film8", "pool5": "影视", "assistant_l1": "影视动漫", "head": "影视query-{yy}.xlsx",
             "rand": {"2025search_rand": "25影视搜索随机1w.xlsx", "2026search_rand": "26影视搜索随机1w.xlsx"},
             "voice": {"assistant_voice_top": "影视ai_voice_top1k.xlsx",
                       "assistant_voice_random": "影视ai_voice_random1kxlsx.xlsx"}},
}
ORDER = ["2025search", "2025search_rand", "2026search", "2026search_rand",
         "assistant_top", "assistant_random", "assistant_voice_top", "assistant_voice_random"]
SURF = {s: ("搜索" if "search" in s else "AI助手") for s in ORDER}
QCOL, PVCOL = "original_query", "wise_pv"


def _drop_empty(d: pd.DataFrame, col: str, where: str, dropped: list) -> pd.DataFrame:
    bad = d[col].isna() | (d[col].astype("string").fillna("").str.strip() == "")
    if int(bad.sum()):
        dropped.append({"where": where, "n": int(bad.sum())})
    return d[~bad].reset_index(drop=True)


def _search(domain: str, source: str, dropped: list) -> pd.DataFrame:
    spec = SPEC[domain]
    fn = spec["rand"][source] if source in spec["rand"] else spec["head"].format(yy="250701" if source.startswith("2025") else "260701")
    d = _drop_empty(pd.read_excel(ROOT / f"data/raw/{fn}"), QCOL, f"{domain}/{source}", dropped)
    q = d[QCOL].astype(str)
    out = pd.DataFrame({"query": q, "source": source, "domain": domain, "l2": pd.NA,
                        "pv_raw": d[PVCOL].astype(float)})
    out["tier"] = "user"
    out.loc[C.headline_flags(q, out["pv_raw"]).values, "tier"] = "S5_headline"
    out.loc[C.content_free_flags(q, surface="search").values, "tier"] = "C1_content_free"  # narrower rule wins
    return out


def _assistant(domain: str, tiers: pd.DataFrame) -> pd.DataFrame:
    p = pd.read_parquet(ROOT / "data/raw/ai_assistant_pooled.parquet")
    p = p[p.l1 == SPEC[domain]["assistant_l1"]]
    rows = []
    for stratum, source in (("head", "assistant_top"), ("tail", "assistant_random")):
        s = p[p.stratum == stratum].copy()
        t = tiers[(tiers.l1 == SPEC[domain]["assistant_l1"]) & (tiers.snapshot == ("top1k" if stratum == "head" else "random1k"))]
        t = t[["query", "l2", "tier"]].drop_duplicates(subset=["query", "l2"])
        m = s.merge(t, on=["query", "l2"], how="left", validate="m:1")
        assert m.tier.notna().all(), f"{domain}/{stratum}: {int(m.tier.isna().sum())} rows without a tier"
        rows.append(pd.DataFrame({"query": m["query"].astype(str), "source": source, "domain": domain,
                                  "l2": m["l2"], "pv_raw": m["search_num"].astype(float), "tier": m["tier"]}))
    return pd.concat(rows, ignore_index=True)


def _voice(domain: str, source: str, facts: dict, dropped: list) -> pd.DataFrame:
    fn = SPEC[domain]["voice"][source]
    d = _drop_empty(pd.read_excel(ROOT / f"data/raw/{fn}"), QCOL, f"{domain}/{source}", dropped)
    pv = d[PVCOL].astype(float)
    assert pv.notna().all(), f"{domain}/{source}: rows without PV"
    if source == "assistant_voice_top":
        # The export IS the head: 1,000 PV-sorted rows. Whether rows below row 1,000 tie with the last one
        # cannot be checked — the file stops there — so this boundary is the export's, not a measured one.
        assert (pv.diff().dropna() <= 0).all(), f"{domain}: voice head export is not sorted by PV descending"
        assert len(d) == VOICE_HEAD_ROWS, f"{domain}: voice head export has {len(d)} rows, expected {VOICE_HEAD_ROWS}"
        facts["voice_head"] = {"file_rows": int(len(d)), "cut_pv": float(pv.iloc[-1]),
                               "rows_at_cut_pv": int((pv == pv.iloc[-1]).sum()),
                               "tie_boundary_checkable": False,
                               "note": "export arrives truncated at 1,000 PV-sorted rows; ties below the cut are unknowable"}
    q = d[QCOL].astype(str)
    out = pd.DataFrame({"query": q, "source": source, "domain": domain, "l2": pd.NA, "pv_raw": pv})
    out["tier"] = "user"
    out.loc[C.content_free_flags(q, surface="assistant").values, "tier"] = "C1_content_free"
    out.loc[C.headline_flags(q, out["pv_raw"]).values & (out.tier == "user"), "tier"] = "S5_headline"
    out.loc[q.isin(C.FEATURE_ENTRIES).values, "tier"] = "S2_system_template"
    return out


def _assert_superset(domain: str, mine: pd.DataFrame) -> str:
    """The four shared snapshots must come out identical to the delivered pool5 corpus."""
    old = pd.read_parquet(OUT_MINE / f"{SPEC[domain]['pool5']}_pooled5.parquet")
    shared = set(ORDER) - set(SPEC[domain]["rand"]) - set(SPEC[domain]["voice"])
    assert set(old.source.unique()) == shared, f"{domain}: pool5 has sources {sorted(old.source.unique())}"
    lines = []
    for src in old.source.unique():
        a = old[old.source == src].reset_index(drop=True)
        b = mine[mine.source == src].reset_index(drop=True)
        assert len(a) == len(b), f"{src}: pool5 has {len(a)} rows, {domain} has {len(b)}"
        # Missing is compared AS MISSING (the delivered parquet stores an absent l2 as a string-dtype NaN
        # while this build holds pd.NA — see build_med8_corpus.py, where astype(str) reported every search
        # row as different when not one cell was).
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
    return "\n".join(lines)


def _md_table(df: pd.DataFrame) -> str:
    """Markdown table without pandas.to_markdown — `tabulate` is not a dependency of this venv."""
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    rule = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, rule, *rows])


def _audit_md(domain: str, facts: dict, r: pd.DataFrame, same: str) -> str:
    spec = SPEC[domain]
    ov = facts["overlap_distinct_mined"]
    keep = [(k, v) for k, v in ov.items() if v]
    lines = [f"# {domain} 语料构建实测（`build_pool8_corpus.py`，{facts['built']}）", "",
             f"全部读入 {facts['rows_all']:,} 行 → 入挖掘 {facts['rows_mined']:,} 行。四个与 `{spec['pool5']}_pooled5.parquet` 共有的快照逐格相同：", "",
             same, "",
             "## 每个快照", "", _md_table(r), "",
             "## 语音头部的边界", "",
             f"- 导出就是头部：{facts['voice_head']['file_rows']:,} 行，按 PV 降序；最后一行 PV = {facts['voice_head']['cut_pv']:.0f}，"
             f"与它同 PV 的有 {facts['voice_head']['rows_at_cut_pv']} 行。",
             "- **这个边界是导出方切的，不是本仓库测出来的**：文件在 1,000 行处截断，行外是否还有同 PV 的行无从判断"
             "（医疗8 的语音导出有 1 万行，所以那次可以切在无并列的边界上）。", "",
             "## 快照之间的重合（去重后的入挖掘串）", "",
             "| 快照对 | 重合串数 |", "|---|---:|"]
    lines += [f"| {k} | {v:,} |" for k, v in sorted(keep, key=lambda x: -x[1])]
    lines += ["", "## 空文本丢弃", "", f"{facts['empty_dropped'] or '无'}", ""]
    return "\n".join(lines)


def build(domain: str) -> int:
    spec = SPEC[domain]
    facts: dict = {"built": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}
    dropped: list = []
    tiers = pd.read_parquet(ROOT / "analysis/sva2026/work/clean_v3/assistant_tiered.parquet")
    parts = [_search(domain, s, dropped) for s in ("2025search", "2025search_rand", "2026search", "2026search_rand")]
    parts.append(_assistant(domain, tiers))
    parts += [_voice(domain, s, facts, dropped) for s in spec["voice"]]
    allrows = pd.concat(parts, ignore_index=True)
    allrows["source"] = pd.Categorical(allrows["source"], categories=ORDER, ordered=True)
    allrows = allrows.sort_values("source", kind="stable").reset_index(drop=True)
    allrows["source"] = allrows["source"].astype(str)
    allrows["surface"] = allrows["source"].map(SURF)
    assert allrows["surface"].notna().all()
    allrows["kept"] = allrows["tier"] == "user"
    allrows["row_id_all"] = range(len(allrows))
    mine = allrows[allrows.kept].reset_index(drop=True).copy()
    pv = mine["pv_raw"]
    assert pv.notna().all(), "every source in this build has PV; a missing one means a changed export"
    tot = pv.groupby(mine["source"]).transform("sum")
    mine["pv_norm"] = (pv / tot * 10_000).astype(float)
    mine["row_id"] = range(len(mine))
    cols = ["query", "source", "surface", "domain", "l2", "pv_raw", "pv_norm", "tier", "row_id", "row_id_all"]
    same = _assert_superset(domain, mine)
    assert list(mine.source.drop_duplicates()) == ORDER, list(mine.source.drop_duplicates())
    mine[cols].to_parquet(OUT_MINE / f"{domain}_pooled5.parquet", index=False)
    allrows.to_parquet(OUT_WORK / f"{domain}_all_rows.parquet", index=False)

    rep = []
    for src in ORDER:
        g = allrows[allrows.source == src]
        rep.append({"domain": domain, "source": src, "rows": len(g), "kept": int(g.kept.sum()),
                    "dropped": int((~g.kept).sum()), "drop_%": round(100 * (~g.kept).mean(), 2),
                    "pv_dropped_%": round(100 * g.loc[~g.kept, "pv_raw"].sum() / max(g["pv_raw"].sum(), 1), 2)})
    r = pd.DataFrame(rep)
    r.to_csv(OUT_WORK / f"build_audit_{spec['tag']}.csv", index=False, encoding="utf-8-sig")
    kept_sets = {s: set(mine.loc[mine.source == s, "query"].astype(str)) for s in ORDER}
    facts["rows_all"], facts["rows_mined"] = int(len(allrows)), int(len(mine))
    facts["tiers_by_source"] = {s: allrows[allrows.source == s].tier.value_counts().to_dict() for s in ORDER}
    facts["distinct_mined_by_source"] = {s: len(v) for s, v in kept_sets.items()}
    facts["pv_median_by_source"] = {s: float(mine.loc[mine.source == s, "pv_raw"].median()) for s in ORDER}
    facts["overlap_distinct_mined"] = {f"{a}∩{b}": len(kept_sets[a] & kept_sets[b]) for a, b in combinations(ORDER, 2)}
    facts["empty_dropped"] = dropped
    (OUT_WORK / f"build_facts_{spec['tag']}.json").write_text(json.dumps(facts, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT_WORK / f"{spec['tag']}_build_audit.md").write_text(_audit_md(domain, facts, r, same), encoding="utf-8")

    print(f"===== {domain}")
    print(f"与已交付的 {spec['pool5']}_pooled5 逐格比对（四个共有快照）：\n{same}")
    print(f"语音头部：{json.dumps(facts['voice_head'], ensure_ascii=False)}")
    print(f"空文本丢弃：{dropped or '无'}")
    print(f"{domain}: 全部 {len(allrows):,} 行 → 入挖掘 {len(mine):,} 行")
    print(r.to_string(index=False))
    return 0


def main() -> int:
    C._check_engine_semantics()
    OUT_MINE.mkdir(parents=True, exist_ok=True)
    doms = sys.argv[1:] or list(SPEC)
    for d in doms:
        assert d in SPEC, f"unknown domain {d}; expected one of {list(SPEC)}"
        build(d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
