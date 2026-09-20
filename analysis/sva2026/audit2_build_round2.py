# -*- coding: utf-8 -*-
"""Audit 2, round 2: a U05 census and holdout sets for rules FROZEN from round-1 misses.

Rules were written from round-1 labelled strings only, checked for false positives on those
strings, then frozen here. Holdout rows exclude every round-1 and first-audit string. Everything
is pooled with decoys into one shuffled blind list (id, surface, category, string).
"""
import importlib.util
import re
import sys

import numpy as np
import pandas as pd

QM = __import__("os").path.abspath(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", ".."))
SP = __import__("os").environ.get("SVA_WORKDIR", __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "work"))
sys.path.insert(0, SP)
import audit2_labels as LB  # noqa: E402

spec = importlib.util.spec_from_file_location("caf", f"{QM}/tools/clean_assistant_functional.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)
SEED = 20260911
CAT = {"金融": "金融", "医疗": "医疗", "教育": "教育培训", "影视": "影视动漫", "人物": "人物"}

# ---- frozen candidate rules (assistant) ----
HX = re.compile(r"(逝世|去世|离世|疑似|被偶遇|偶遇|同框|拟邀|辞去|涨粉|自曝|发声|现身|互动|引热议|热议|官宣|离婚|出轨|夺冠|当选"
                r"|即将|首次|回应|表态|被查|落马|获刑|身亡|否认|曝光|爆料)")
KW = re.compile(r"(最新消息|最新进展|最新通知|最新政策|调整政策|后续|进展|路径图|事故|前后对比)$")
# ---- frozen candidate exemptions (search S5) ----
SX_QUOTED = re.compile(r"^\s*[“\"][^”\"]+[”\"]\s*$")
SX_NAME_DEATH = re.compile(r"^[一-鿿]{2,4}(去世|逝世)$")
SX_ASK = re.compile(r"(有何|何时|为何|是谁|咋)")


def rules(v3):
    news_head = set(v3.loc[(v3.l1 == "新闻") & (v3.snapshot == "top1k"), "qs"])
    q, head = v3.qs, v3.snapshot == "top1k"
    return {
        "R1_headline_verbs": head & (v3.l1 != "新闻") & q.str.contains(HX) & ~q.str.contains(M.ASKS)
                             & (v3.search_num >= 100) & (q.str.len() >= 6),
        "R1b_crosslisted_news": head & (v3.l1 != "新闻") & q.isin(news_head) & ~q.str.contains(M.NEWS_ASKS),
        "R2a_more_chip": q.str.contains(r"^有没有更多"),
        "R2b_llm_followup": q.str.contains(r"有哪些感人瞬间$|是否公开过|有官方报道吗[？?]?$"),
        "R3_reply": q.str.match(r"^\s*(什么|啥|随便|都行|无所谓|你妈的?)[\s。！!~～？?]*$"),
        "R4_hashtag": q.str.match(r"^\s*#[^#]{2,}#\s*$"),
        "R5_card": q.str.contains(r"^[一-鿿]{2,4}[，,]现任|‌"),
        "R6_essay_edit_template": q.str.contains(r"^我想对作文《"),
    }


if __name__ == "__main__":
    v3 = pd.read_parquet(f"{SP}/clean_v3/assistant_tiered.parquet")
    v3["qs"] = v3["query"].astype(str)
    F = pd.read_parquet(f"{SP}/sva_final_rows.parquet")
    v3["u_ds"] = None
    for d, c in CAT.items():
        for snap, surf in (("top1k", "assistant_top1k"), ("random1k", "assistant_random1k")):
            idx = v3.index[(v3.l1 == c) & (v3.snapshot == snap)]
            f = F[(F.domain == d) & (F.surface == surf)]
            assert (v3.loc[idx, "qs"].values == f["query"].astype(str).values).all()
            v3.loc[idx, "u_ds"] = f.u_ds.values
    L1 = pd.read_parquet(f"{SP}/audit_clean_samples_labeled.parquet")
    LS = pd.read_parquet(f"{SP}/audit_clean_search_samples.parquet")
    FIRST = set(L1.qs.astype(str)) | set(LS["query"].astype(str))
    USED_A = FIRST | set(LB.LAB_ASSISTANT)
    USED_S = FIRST | set(LB.LAB_SEARCH)
    rng = np.random.RandomState(SEED)
    parts = []
    user = v3.tier == "user"
    five = v3.l1.isin(CAT.values())

    census = v3[five & (v3.snapshot == "top1k") & user & (v3.u_ds == "U05") & ~v3.qs.isin(USED_A)]
    parts.append(census.assign(set="U05_census"))
    R = rules(v3)
    for name, m in R.items():
        pool = v3[m & user & ~v3.qs.isin(USED_A)].drop_duplicates("qs")
        cap = {"R1b_crosslisted_news": 40, "R4_hashtag": 30, "R5_card": 40}.get(name)
        if cap and len(pool) > cap:
            pool = pool.sample(cap, random_state=rng)
        parts.append(pool.assign(set=name))
    s5 = v3[(v3.tier == "S5_headline") & v3.qs.str.contains(KW) & ~v3.qs.isin(USED_A)].drop_duplicates("qs")
    parts.append(s5.assign(set="R7_news_keyword_exemption"))
    taken = set(pd.concat(parts).qs)
    dec = v3[(v3.snapshot == "top1k") & user & ~v3.qs.isin(USED_A | taken)].drop_duplicates("qs").sample(40, random_state=rng)
    parts.append(dec.assign(set="decoy_assistant"))
    A = pd.concat(parts, ignore_index=True)
    A = A.assign(surface="assistant", category=A.l1, pv=A.search_num, rank=np.nan)[
        ["set", "surface", "qs", "category", "snapshot", "pv", "rank", "tier", "u_ds", "l1"]]

    srows = []
    for dom in CAT:
        raw = pd.read_csv(f"{QM}/data/raw/{dom}query-pooled.csv")
        raw = raw[raw.original_query.notna() & (raw._snapshot.astype(str) == "20250701")].copy()
        raw = raw.sort_values("wise_pv", ascending=False).reset_index(drop=True)
        raw["rank"] = raw.index + 1
        raw["qs"] = raw.original_query.astype(str)
        raw["domain"] = dom
        srows.append(raw[["domain", "qs", "wise_pv", "rank"]])
    s25 = pd.concat(srows, ignore_index=True)
    flag = M.headline_flags(s25.qs, s25.wise_pv)
    hs = s25[flag & ~s25.qs.isin(USED_S)].drop_duplicates(["domain", "qs"])
    ds = s25[~flag & (s25["rank"] <= 1000) & ~s25.qs.isin(USED_S)].drop_duplicates(["domain", "qs"]).sample(15, random_state=rng)
    S = pd.concat([hs.assign(set="S5_search_2025"), ds.assign(set="decoy_search_2025")], ignore_index=True)
    S = S.assign(surface="search", category=S.domain, snapshot="search2025", pv=S.wise_pv, tier=np.where(S.set == "S5_search_2025", "S5_headline(2025 replay)", "user(2025 replay)"),
                 u_ds=None, l1=None)[["set", "surface", "qs", "category", "snapshot", "pv", "rank", "tier", "u_ds", "l1"]]
    allr = pd.concat([A, S], ignore_index=True)
    allr.to_parquet(f"{SP}/audit2_round2_samples.parquet", index=False)

    u = allr.groupby(["surface", "qs"]).agg(cats=("category", lambda c: "/".join(sorted(set(map(str, c)))[:3]))).reset_index()
    u = u.sample(frac=1.0, random_state=SEED + 1).reset_index(drop=True)
    u["id"] = np.arange(2001, 2001 + len(u))
    u.to_parquet(f"{SP}/audit2_round2_list.parquet", index=False)
    with open(f"{SP}/audit2_round2_list.txt", "w", encoding="utf-8") as fh:
        for r in u.itertuples():
            s = "".join(f"\\u{ord(ch):04x}" if (ch in "\t\n\r" or ch in "​‌‍﻿") else ch for ch in r.qs)
            fh.write(f"{r.id}\t{'A' if r.surface == 'assistant' else 'S'}\t{r.cats}\t{s}\n")
    print("rows per set:", allr["set"].value_counts().to_dict())
    print("distinct per set:", allr.groupby("set").qs.nunique().to_dict())
    print("pool sizes before caps:", {k: int((m & user & ~v3.qs.isin(USED_A)).sum()) for k, m in R.items()})
    print("search 2025 headline-flagged fresh rows:", len(hs), "of flagged", int(flag.sum()))
    print("blind list:", len(u), u.surface.value_counts().to_dict())
