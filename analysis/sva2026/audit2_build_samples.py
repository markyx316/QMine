# -*- coding: utf-8 -*-
"""Audit 2: draw FRESH samples (no first-audit string) and write a blind, shuffled label list.

The list shows only: id, surface (assistant/search), category, string. No tier, PV, snapshot
or sample membership, so a label cannot be anchored on what the rules decided.
"""
import unicodedata
import numpy as np
import pandas as pd

SP = __import__("os").environ.get("SVA_WORKDIR", __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "work"))
SEED = 20260910
CAT = {"金融": "金融", "医疗": "医疗", "教育": "教育培训", "影视": "影视动漫", "人物": "人物"}
DOM_OF = {v: k for k, v in CAT.items()}

v3 = pd.read_parquet(f"{SP}/clean_v3/assistant_tiered.parquet")
v2 = pd.read_parquet(f"{SP}/clean_v2/assistant_tiered.parquet")
s3 = pd.read_parquet(f"{SP}/clean_v3/search2026_tiered.parquet")
s2 = pd.read_parquet(f"{SP}/clean_v2/search2026_tiered.parquet")
assert (v2["query"].astype(str).values == v3["query"].astype(str).values).all()
assert (s2["query"].astype(str).values == s3["query"].astype(str).values).all()
L = pd.read_parquet(f"{SP}/audit_clean_samples_labeled.parquet")
LS = pd.read_parquet(f"{SP}/audit_clean_search_samples.parquet")
FIRST = set(L["qs"].astype(str)) | set(LS["query"].astype(str))

a = v3.assign(qs=v3["query"].astype(str), v2_tier=v2["tier"].values, rid=np.arange(len(v3)))
s = s3.assign(qs=s3["query"].astype(str), v2_tier=s2["tier"].values, rid=np.arange(len(s3)))
a["fresh"] = ~a.qs.isin(FIRST)
s["fresh"] = ~s.qs.isin(FIRST)
five = a.l1.isin(CAT.values())

parts, excl, sizes = [], {}, {}


def take(df, sample, surface):
    sizes[sample] = sizes.get(sample, 0) + len(df)
    df = df.copy()
    df["sample"], df["surface"] = sample, surface
    parts.append(df)


rng = np.random.RandomState(SEED)
# 1 / 2: kept user rows, head and tail, stratified by domain
for sample, snap, k in (("1_head_kept", "top1k", 60), ("2_tail_kept", "random1k", 40)):
    for cat in CAT.values():
        pool = a[(a.l1 == cat) & (a.snapshot == snap) & (a.tier == "user") & a.fresh]
        take(pool.sample(k, random_state=rng), sample, "assistant")
# 3a: every changed row in the 5 domains
ch = a[five & (a.tier != a.v2_tier)]
excl["3a_changed"] = int((~ch.fresh).sum())
take(ch[ch.fresh], "3a_changed", "assistant")
# 3b / 3c: 100 random S4 and S5 rows from all 33 categories
for sample, t in (("3b_S4", "S4_suggested_chip"), ("3c_S5", "S5_headline")):
    pool = a[(a.tier == t)]
    excl[sample] = int((~pool.fresh).sum())
    take(pool[pool.fresh].sample(100, random_state=rng), sample, "assistant")
# 4a: 30 kept user rows per domain from each domain's top 1,000 by wise_pv
for d in CAT:
    pool = s[(s.domain == d) & (s["rank"] <= 1000) & (s.tier == "user") & s.fresh]
    take(pool.sample(30, random_state=rng), "4a_search_kept", "search")
# 4b: every non-user search row ; 3d: every search row whose tier changed v2 -> v3 and is now user
nu = s[s.tier != "user"]
excl["4b_search_nonuser"] = int((~nu.fresh).sum())
take(nu[nu.fresh], "4b_search_nonuser", "search")
chs = s[(s.tier != s.v2_tier) & (s.tier == "user")]
excl["3d_search_restored"] = int((~chs.fresh).sum())
take(chs[chs.fresh], "3d_search_restored", "search")

print("sample sizes after exclusion:", sizes)
A = pd.concat([p for p in parts if len(p) and p.surface.iloc[0] == "assistant"], ignore_index=True)
S = pd.concat([p for p in parts if len(p) and p.surface.iloc[0] == "search"], ignore_index=True)
samp = pd.concat([
    A.assign(domain=A.l1.map(DOM_OF), pv=A.search_num, category=A.l1, rank=np.nan)[
        ["sample", "surface", "rid", "qs", "domain", "category", "snapshot", "pv", "rank", "tier", "v2_tier", "td_l1"]],
    S.assign(category=S.domain, snapshot="search2026", pv=S.wise_pv)[
        ["sample", "surface", "rid", "qs", "domain", "category", "snapshot", "pv", "rank", "tier", "v2_tier", "td_l1"]],
], ignore_index=True)
assert not samp.qs.isin(FIRST).any()
samp.to_parquet(f"{SP}/audit2_samples.parquet", index=False)


def show(x: str) -> str:
    out = []
    for ch in x:
        cat = unicodedata.category(ch)
        if ch in "\t\n\r" or cat in ("Cf", "Cc", "Zl", "Zp") or (cat == "Zs" and ch != " "):
            out.append(f"\\u{ord(ch):04x}")
        else:
            out.append(ch)
    return "".join(out)


u = (samp.groupby(["surface", "qs"]).agg(cats=("category", lambda c: "/".join(sorted(set(c))[:3]))).reset_index())
u = u.sample(frac=1.0, random_state=SEED + 1).reset_index(drop=True)
u["id"] = np.arange(1, len(u) + 1)
u.to_parquet(f"{SP}/audit2_label_list.parquet", index=False)
with open(f"{SP}/audit2_label_list.txt", "w", encoding="utf-8") as fh:
    for r in u.itertuples():
        fh.write(f"{r.id}\t{'A' if r.surface == 'assistant' else 'S'}\t{r.cats}\t{show(r.qs)}\n")

print("rows per sample:", samp["sample"].value_counts().sort_index().to_dict())
print("distinct strings per sample:", samp.groupby("sample").qs.nunique().to_dict())
print("excluded first-audit rows:", excl)
print("label list distinct (surface,string):", len(u), u.surface.value_counts().to_dict())
