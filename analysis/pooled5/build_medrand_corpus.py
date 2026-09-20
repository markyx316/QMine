#!/usr/bin/env python
"""Build the `医疗随机` corpus: one day (2026-09-14) of 医疗-category TRADITIONAL SEARCH and of the
健康管家 assistant, both as RANDOM 10k exports, pooled into one file so one taxonomy labels both.

WHY THIS PAIR EXISTS BESIDE `健康`. `health-pool2` mined the same two products a week earlier, but from
"top 10,000 (query x day)" exports: after aggregation and cleaning the assistant side was left with 225
usable rows — too few to say anything. These exports are **random samples of DISTINCT queries** (10,000
and 10,622 rows, every string unique), so the long tail is present and the product layer, which repeats
heavily and therefore dominated a top-by-day export, collapses to one row per string. Measured with
health-pool2's own rules on this export: 11 intake answers, 4 doctor cards, 0 feature buttons, 0
content-free. `健康_pooled5.parquet` is untouched (it is row-bound to health-pool2's labels_full.csv).

THE WRAPPER IS STRIPPED, NOT DELETED — the decision that makes this corpus usable.
75.4% of the assistant rows (8,014 of 10,622, 77.6% of PV) begin with 我想咨询, the product's input
template. health-pool2 tiered that form as H4_template_wrapper and dropped it; there it matched 94 of
1,887 strings. Here the text after the prefix is the user's own question — measured: median length 12,
66.6% carry a question marker, and stripping produces 10,621 distinct strings (one collision). Dropping
them would repeat exactly the failure this corpus exists to fix. So the prefix is removed from `query`,
the original is kept in `query_raw`, and `wrapper_stripped` marks the rows.

WHAT IS REMOVED, AND HOW THE DECISION IS MADE. Only the product's own artefacts: intake-answer chips,
feature or card-facet buttons, doctor/institution cards, content-free acknowledgements. Two instruments:
- RULES, reused from `build_health_corpus.py` (feature list, facet grammar, doctor-card pattern,
  content-free), which measured precision 0.991 there;
- a BLIND AUDIT of the 1,192 ambiguous strings (non-wrapper, no question marker) — three independent
  readings under one criteria-first codebook whose own tie-break is "if unsure between chip and user,
  choose user" (`work/医疗随机/ai_audit_labels.csv`).
**A row is removed only when all three readings agree it is product layer.** Any split reading stays a
user row. That is deliberately less stringent than health-pool2, which let the rules decide split cases.
Nothing is deleted from the record: every string is written to `work/医疗随机_all_rows.parquet` with its
tier, its rule flags and its three votes, so any figure can be recomputed with the removed rows included.

SEARCH SIDE. No aggregation is needed (already one row per distinct query, one day). The content-free and
headline rules match 0 rows. health-pool2's S6 doctor-card rule is NOT applied: it required a seven-day
uniform-traffic signature (16 doctors, cross-doctor CV 0.026) that a one-day random export cannot show —
here the 13 strings matching the card shape carry ordinary tail PV, so they are kept as user rows and
flagged (`flag_doctor_card_shape`) for the quote guard rather than removed.

LEGACY LABELS. Both exports carry the platform's own three-level category on every row (100% coverage,
a shared 6-value level-2 vocabulary, and a level-3 that is 科室-类型 crossed, split here into
`legacy_dept` / `legacy_type`). Whether they are declared as reference columns is decided by the same
PRE-REGISTERED condition health-pool2 used, applied to the final corpus and printed by this script:
Cramér's V(column, surface) <= 0.55 AND no class holding more than 1% of rows appears on one surface only.

    python analysis/pooled5/build_medrand_corpus.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import clean_assistant_functional as C  # noqa: E402
import build_health_corpus as H  # noqa: E402

DOMAIN = "医疗随机"
TAG = "medr"
OUT_MINE = ROOT / "data/raw/pooled5"
WORK = Path(__file__).resolve().parent / "work"
AUDIT = WORK / DOMAIN / "ai_audit_labels.csv"
WRAPPER = "我想咨询"

SPEC = {
    "msearch_2609r": {"file": "传统搜-医疗-随机10000-20260915.xlsx", "q": "original_query", "pv": "cumulative_pv",
                      "l1": "query_1st_category", "l2": "query_2nd_category", "l3": "query_3rd_category",
                      "surface": "搜索", "zh": "传统搜索随机1w", "product": "传统搜索"},
    "mai_2609r": {"file": "健康管家-医疗-随机10000-20260915..xlsx", "q": "query", "pv": "total_pv",
                  "l1": "query_level1_type", "l2": "query_level2_type", "l3": "query_level3_type",
                  "surface": "AI助手", "zh": "健康管家随机1w", "product": "健康管家"},
}
#: audit code -> tier. Only these five are product layer; U is the user's own content.
AUDIT_TIER = {"A": "H1_intake_answer", "F": "H2_feature_or_facet", "P": "H3_pushed_suggestion",
              "D": "H5_doctor_card", "C": "C1_content_free", "U": "user"}
FACTS: dict = {}


def _read(src: str) -> pd.DataFrame:
    sp = SPEC[src]
    d = pd.read_excel(ROOT / f"data/raw/{sp['file']}", dtype=str)
    q = d[sp["q"]].astype("string")
    assert q.notna().all() and (q.str.strip() != "").all(), f"{src}: empty query cells"
    assert (q == q.str.strip()).all(), f"{src}: leading/trailing whitespace would split strings"
    assert q.is_unique, f"{src}: the export is supposed to be de-duplicated, found {int(q.duplicated().sum())} repeats"
    assert d[sp["l1"]].nunique() == 1, f"{src}: level-1 category is not constant"
    pv = pd.to_numeric(d[sp["pv"]], errors="coerce")
    assert pv.notna().all(), f"{src}: {int(pv.isna().sum())} rows without PV"
    dept, typ = zip(*d[sp["l3"]].astype(str).map(H._split_l3))
    return pd.DataFrame({"query": q.astype(str), "query_raw": q.astype(str), "source": src, "source_zh": sp["zh"],
                         "product": sp["product"], "source_file": sp["file"], "surface": sp["surface"],
                         "domain": DOMAIN, "legacy_l1": d[sp["l1"]].astype(str), "legacy_l2": d[sp["l2"]].astype(str),
                         "legacy_l3": d[sp["l3"]].astype(str), "legacy_dept": list(dept), "legacy_type": list(typ),
                         "pv_raw": pv.astype(float), "event_day": d["event_day"].astype(str)})


def tier_search(t: pd.DataFrame) -> pd.DataFrame:
    q = t["query"]
    t["flag_C1"] = C.content_free_flags(q, surface="search").values
    t["flag_S5"] = C.headline_flags(q, t["pv_raw"]).values
    # kept, not removed: a one-day random export cannot show the uniform-traffic signature that made
    # health-pool2's S6 rule a product-card detector (see the module docstring).
    t["flag_doctor_card_shape"] = q.str.match(H.DOCTOR_CARD_SEARCH).values
    t["tier"] = "user"
    t.loc[t.flag_S5, "tier"] = "S5_headline"
    t.loc[t.flag_C1, "tier"] = "C1_content_free"
    t["tier_source"] = "rule"
    return t


def tier_ai(t: pd.DataFrame) -> pd.DataFrame:
    raw = t["query_raw"]
    t["wrapper_stripped"] = raw.str.startswith(WRAPPER).values
    t["query"] = raw.str.replace(rf"^{WRAPPER}", "", regex=True).str.strip()
    empty = t["query"].str.len() == 0
    t.loc[empty, "query"] = raw[empty]          # a bare "我想咨询" keeps its text and falls to C1/F below
    q = t["query"]
    t["flag_C1"] = (C.content_free_flags(q, surface="assistant") | q.str.match(H.ACK_HEALTH)).values
    t["flag_H5"] = q.str.contains(H.DOCTOR_CARD_AI).values
    t["flag_H2"] = (q.isin(H.FEATURE_HEALTH | H.FACET_HEALTH) | q.str.contains(H.TOOL) | q.str.match(H.FACET)).values
    t["flag_H1"] = q.map(H.intake_answer).values
    t["rule_tier"] = "user"
    for flag, tier in [("flag_H1", "H1_intake_answer"), ("flag_H2", "H2_feature_or_facet"),
                       ("flag_H5", "H5_doctor_card"), ("flag_C1", "C1_content_free")]:
        t.loc[t[flag], "rule_tier"] = tier
    # OUTSIDE the audited subset, only the NARROW rules may remove a row. health-pool2's intake regex has an
    # arm of the shape (有|无|没有|轻微|明显…) + 1-6 CJK characters, which on this export removed three rows that
    # are plainly user questions (fig efficacy, a thyroid panel value, and a sexual-health question). Inside the
    # audited subset the three readings decide, so the regex is not needed there either.
    NARROW = {"H2_feature_or_facet", "H5_doctor_card", "C1_content_free"}
    # blind audit over the ambiguous subset
    assert AUDIT.exists(), f"missing {AUDIT} — run the blind-audit workflow first"
    a = pd.read_csv(AUDIT, encoding="utf-8-sig").set_index("query")
    t["audit_votes"] = t["query"].map(a["votes"]).fillna("").values
    t["audit_majority"] = t["query"].map(a["majority"]).fillna("").values
    unanimous_product = t["audit_votes"].map(lambda v: len(v) == 3 and len(set(v)) == 1 and v[0] != "U")
    audited = t["audit_votes"].str.len() == 3
    t["tier"] = np.where(audited | t["rule_tier"].isin(NARROW), t["rule_tier"], "user")
    t.loc[unanimous_product, "tier"] = t.loc[unanimous_product, "audit_majority"].map(AUDIT_TIER)
    # LESS STRINGENT THAN health-pool2, on purpose: a split reading keeps the row, and a row the audit
    # called user is kept even when a rule fired on it.
    audit_user = t["audit_votes"].map(lambda v: len(v) == 3 and len(set(v)) == 1 and v[0] == "U")
    t.loc[audit_user, "tier"] = "user"
    split = t["audit_votes"].map(lambda v: len(v) == 3 and len(set(v)) > 1)
    t.loc[split, "tier"] = "user"
    t["tier_source"] = np.where(unanimous_product | audit_user, "audit_unanimous",
                                np.where(split, "audit_split_kept", "rule"))
    FACTS["ai_audit"] = {"audited": int((t["audit_votes"].str.len() == 3).sum()),
                         "unanimous_product": int(unanimous_product.sum()), "unanimous_user": int(audit_user.sum()),
                         "split_kept_as_user": int(split.sum()),
                         "rule_only_rows": int((~audited).sum()),
                         "rule_removed_outside_audit": int(((~audited) & (t["tier"] != "user")).sum()),
                         "kept_by_narrowing_rules_outside_audit": int(((~audited) & t["rule_tier"].ne("user")
                                                                       & ~t["rule_tier"].isin(NARROW)).sum())}
    return t


def cramers_v(a: pd.Series, b: pd.Series) -> float:
    ct = pd.crosstab(a, b).to_numpy(float)
    n = ct.sum()
    exp = ct.sum(1, keepdims=True) @ ct.sum(0, keepdims=True) / n
    chi2 = ((ct - exp) ** 2 / exp).sum()
    return float(np.sqrt((chi2 / n) / (min(ct.shape) - 1)))


def main() -> int:
    C._check_engine_semantics()
    (WORK / DOMAIN).mkdir(parents=True, exist_ok=True)
    s = tier_search(_read("msearch_2609r"))
    a = tier_ai(_read("mai_2609r"))
    allrows = pd.concat([s, a], ignore_index=True)
    allrows["kept"] = allrows["tier"] == "user"
    allrows["row_id_all"] = range(len(allrows))
    mine = allrows[allrows.kept].reset_index(drop=True).copy()
    # a stripped string can collide with another row's text: sum their PV, keep one row
    dupe = int(mine.duplicated(["source", "query"]).sum())
    if dupe:
        pv = mine.groupby(["source", "query"])["pv_raw"].transform("sum")
        mine["pv_raw"] = pv
        mine = mine.drop_duplicates(["source", "query"]).reset_index(drop=True)
    FACTS["stripped_collisions_merged"] = dupe
    tot = mine.groupby("source")["pv_raw"].transform("sum")
    mine["pv_norm"] = (mine["pv_raw"] / tot * 10_000).astype(float)
    mine["row_id"] = range(len(mine))
    cols = ["query", "query_raw", "source", "source_zh", "product", "source_file", "surface", "domain",
            "legacy_l1", "legacy_l2", "legacy_l3", "legacy_dept", "legacy_type",
            "pv_raw", "pv_norm", "tier", "tier_source", "wrapper_stripped", "row_id", "row_id_all"]
    for c in cols:
        if c not in mine.columns:
            mine[c] = pd.NA
    mine[cols].to_parquet(OUT_MINE / f"{DOMAIN}_pooled5.parquet", index=False)
    allrows.to_parquet(WORK / f"{DOMAIN}_all_rows.parquet", index=False)

    rep = []
    for src in SPEC:
        g = allrows[allrows.source == src]
        rep.append({"domain": DOMAIN, "source": src, "rows": len(g), "kept": int(g.kept.sum()),
                    "dropped": int((~g.kept).sum()), "drop_%": round(100 * (~g.kept).mean(), 2),
                    "pv_dropped_%": round(100 * g.loc[~g.kept, "pv_raw"].sum() / max(g["pv_raw"].sum(), 1), 2)})
    r = pd.DataFrame(rep)
    r.to_csv(WORK / f"build_audit_{TAG}.csv", index=False, encoding="utf-8-sig")
    FACTS["rows_all"], FACTS["rows_mined"] = int(len(allrows)), int(len(mine))
    FACTS["tiers_by_source"] = {s_: allrows[allrows.source == s_].tier.value_counts().to_dict() for s_ in SPEC}
    FACTS["wrapper_rows"] = int(allrows["wrapper_stripped"].fillna(False).sum())
    FACTS["overlap_mined"] = int(len(set(mine.loc[mine.source == "msearch_2609r", "query"])
                                     & set(mine.loc[mine.source == "mai_2609r", "query"])))
    # pre-registered reference-column condition, applied to the FINAL corpus
    ref = {}
    for col in ("legacy_l2", "legacy_dept", "legacy_type"):
        v = cramers_v(mine[col], mine["surface"])
        by = pd.crosstab(mine[col], mine["surface"])
        share = by.sum(1) / len(mine)
        one_sided = [(str(k), round(100 * share[k], 2)) for k in by.index if (by.loc[k] == 0).any() and share[k] > 0.01]
        ref[col] = {"cramers_v": round(v, 3), "one_sided_classes_over_1%": one_sided,
                    "declare": bool(v <= 0.55 and not one_sided)}
    FACTS["reference_columns"] = ref
    (WORK / f"build_facts_{TAG}.json").write_text(json.dumps(FACTS, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"{DOMAIN}: 全部 {len(allrows):,} 行 → 入挖掘 {len(mine):,} 行（合并了 {dupe} 条去前缀后重复）")
    print(r.to_string(index=False))
    print(f"\n模板前缀「{WRAPPER}」：{FACTS['wrapper_rows']:,} 行被剥离前缀后保留")
    print(f"盲标：{json.dumps(FACTS['ai_audit'], ensure_ascii=False)}")
    print(f"两侧共有串：{FACTS['overlap_mined']}")
    print("\n参考列（预先写死的条件：V<=0.55 且没有占比>1% 的单侧类）：")
    for col, v in ref.items():
        print(f"  {col}: V={v['cramers_v']} one-sided>1%={v['one_sided_classes_over_1%']} → {'declare' if v['declare'] else 'WITHDRAW'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
