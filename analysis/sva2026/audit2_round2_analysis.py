# -*- coding: utf-8 -*-
"""Audit 2 round-2 metrics: U05 census, holdout precision of frozen rules, search-S5 2025 replay, decoys."""
import pickle
import sys

import numpy as np
import pandas as pd

SP = __import__("os").environ.get("SVA_WORKDIR", __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "work"))
sys.path.insert(0, SP)
import audit2_build_round2 as B2  # noqa: E402  (frozen rules)
import audit2_labels as LB  # noqa: E402
import audit2_labels2 as LB2  # noqa: E402
from audit2_analysis import CAT, wilson  # noqa: E402


def lab_assistant(q):
    return LB2.LAB2_ASSISTANT.get(q, LB.LAB_ASSISTANT.get(q))


def compute():
    R2 = pd.read_parquet(f"{SP}/audit2_round2_samples.parquet")
    LB2.check(R2)
    isA = R2.surface == "assistant"
    R2["lab"] = np.where(isA, R2.qs.map(LB2.LAB2_ASSISTANT), R2.qs.map(LB2.LAB2_SEARCH))
    R2["why"] = np.where(isA, R2.qs.map(LB2.REASON2_ASSISTANT), R2.qs.map(LB2.REASON2_SEARCH))
    assert R2.lab.isin(["A", "B", "C"]).all()
    out = {"R2": R2}

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
    five = v3.l1.isin(CAT.values())
    user = v3.tier == "user"

    # ---- U05 census: every kept 5-domain head U05 row, labels from round 2 or (22 rows) round 1 ----
    cen = v3[five & (v3.snapshot == "top1k") & user & (v3.u_ds == "U05")].copy()
    cen["lab"] = cen.qs.map(lab_assistant)
    assert cen.lab.notna().all(), f"{cen.lab.isna().sum()} U05 rows unlabelled"
    cen["domain"] = cen.l1.map({v: k for k, v in CAT.items()})
    out["u05_census"] = cen
    N = int((five & (v3.snapshot == "top1k") & user).sum())
    a, c = int((cen.lab == "A").sum()), int((cen.lab == "C").sum())
    S = pd.read_parquet(f"{SP}/audit2_samples.parquet")
    s1 = S[S["sample"] == "1_head_kept"]
    s1lab = s1.qs.map(LB.LAB_ASSISTANT)
    k_all = int((s1lab == "A").sum())
    r, rL, rU = wilson(k_all, len(s1))
    p_u05 = len(cen) / N
    # U05 share after removing its census-measured A rows, with total kept-head A rows taken from sample 1
    after = lambda rr: (len(cen) - a) / (N - rr * N)
    out["u05_summary"] = {"kept_head_rows": N, "u05_rows": len(cen), "u05_share": p_u05, "A": a, "B": int((cen.lab == "B").sum()),
                          "C": c, "A_pv": int(cen.loc[cen.lab == "A", "search_num"].sum()), "pv": int(cen.search_num.sum()),
                          "A_rate": wilson(a, len(cen)), "share_after_point": after(r), "share_after_range": (after(rL), after(rU)),
                          "share_after_also_minus_C": (len(cen) - a - c) / (N - r * N),
                          "by_domain": cen.groupby("domain").lab.value_counts().unstack(fill_value=0)}

    # ---- rule holdouts ----
    R = B2.rules(v3)
    train = set(LB.LAB_ASSISTANT)
    s12 = S[S["sample"].isin(["1_head_kept", "2_tail_kept"])].copy()
    s12["lab"] = s12.qs.map(LB.LAB_ASSISTANT)
    misses = s12[s12.lab == "A"]
    rows = []
    for name, m in R.items():
        h = R2[R2["set"] == name]
        k = int((h.lab == "A").sum())
        p, lo, hi = wilson(k, len(h))
        pop = m & user
        caught = int(m.loc[misses.rid].sum())
        census_hit = cen[m.loc[cen.index] & ~cen.qs.isin(train)]
        rows.append({"rule": name, "holdout_n": len(h), "A": k, "B": int((h.lab == "B").sum()), "C": int((h.lab == "C").sum()),
                     "precision": p, "lo": lo, "hi": hi,
                     "kept_rows_33cat": int(pop.sum()), "kept_rows_5dom_head": int((pop & five & (v3.snapshot == "top1k")).sum()),
                     "kept_rows_5dom_tail": int((pop & five & (v3.snapshot == "random1k")).sum()),
                     "sample12_misses_caught": f"{caught}/{len(misses)}",
                     "u05_census_holdout_hits": len(census_hit), "u05_census_hits_A": int((census_hit.lab == "A").sum())})
    h = R2[R2["set"] == "R7_news_keyword_exemption"]
    k = int((h.lab == "B").sum())
    p, lo, hi = wilson(k, len(h))
    s5 = (v3.tier == "S5_headline") & v3.qs.str.contains(B2.KW)
    rows.append({"rule": "R7_news_keyword_exemption (precision = B/n)", "holdout_n": len(h), "A": int((h.lab == "A").sum()),
                 "B": k, "C": int((h.lab == "C").sum()), "precision": p, "lo": lo, "hi": hi,
                 "kept_rows_33cat": int(s5.sum()), "kept_rows_5dom_head": int((s5 & five).sum()), "kept_rows_5dom_tail": 0,
                 "sample12_misses_caught": "n/a", "u05_census_holdout_hits": 0, "u05_census_hits_A": 0})
    out["rules"] = pd.DataFrame(rows)
    # union of the two headline rules on the census (holdout part only)
    un = (R["R1_headline_verbs"] | R["R1b_crosslisted_news"])
    ch = cen[~cen.qs.isin(train)]
    hit = un.loc[ch.index]
    out["u05_union"] = {"holdout_census_rows": len(ch), "A_in_holdout": int((ch.lab == "A").sum()),
                        "union_hits": int(hit.sum()), "union_hits_A": int((ch.lab[hit] == "A").sum()),
                        "recall_on_A": wilson(int((ch.lab[hit] == "A").sum()), int((ch.lab == "A").sum())),
                        "precision": wilson(int((ch.lab[hit] == "A").sum()), int(hit.sum()))}
    out["rule_false_positives"] = R2[R2["set"].str.startswith("R") & (R2.lab != "A") & (R2["set"] != "R7_news_keyword_exemption")]
    out["r7_wrong"] = R2[(R2["set"] == "R7_news_keyword_exemption") & (R2.lab != "B")]

    # ---- search S5 replay on the 2025 snapshot, with the frozen exemptions ----
    s = R2[R2["set"] == "S5_search_2025"].copy()
    ex = s.qs.str.match(B2.SX_QUOTED) | s.qs.str.match(B2.SX_NAME_DEATH) | s.qs.str.contains(B2.SX_ASK)
    s["exempt"] = ex
    out["s25"] = s
    out["s25_summary"] = {"n": len(s), "S5_precision": wilson(int((s.lab == "A").sum()), len(s)),
                          "exempted": int(ex.sum()), "exempt_precision_B": wilson(int((s.lab[ex] == "B").sum()), int(ex.sum())),
                          "S5_after_exemption_precision": wilson(int((s.lab[~ex] == "A").sum()), int((~ex).sum()))}
    # the same exemptions on the 2026 round-1 search S5 rows (training: they were written from these)
    S4b = S[S["sample"] == "4b_search_nonuser"].copy()
    S4b["lab"] = S4b.qs.map(LB.LAB_SEARCH)
    ex26 = S4b.qs.str.match(B2.SX_QUOTED) | S4b.qs.str.match(B2.SX_NAME_DEATH) | S4b.qs.str.contains(B2.SX_ASK)
    out["s26_train"] = {"exempted": int(ex26.sum()), "exempted_B": int((S4b.lab[ex26] == "B").sum()),
                        "S5_after": wilson(int((S4b.lab[~ex26] == "A").sum()), int((~ex26).sum()))}
    for nm in ("decoy_assistant", "decoy_search_2025"):
        d = R2[R2["set"] == nm]
        out[nm] = {"n": len(d), "A": int((d.lab == "A").sum()), "C": int((d.lab == "C").sum()),
                   "A_rate": wilson(int((d.lab == "A").sum()), len(d))}
    return out


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 40)
    pd.set_option("display.max_rows", 300)
    o = compute()
    u = o["u05_summary"]
    print("U05 census:", {k: v for k, v in u.items() if k != "by_domain"})
    print(u["by_domain"])
    print(o["u05_census"][o["u05_census"].lab == "A"].groupby("domain").search_num.agg(["size", "sum"]))
    t = o["rules"].copy()
    for cc in ("precision", "lo", "hi"):
        t[cc] = (t[cc] * 100).round(1)
    print(t.to_string(index=False))
    print("union R1|R1b on U05 census holdout:", o["u05_union"])
    print("\nrule false positives:\n", o["rule_false_positives"][["set", "qs", "category", "snapshot", "pv", "lab", "why"]].to_string(index=False))
    print("\nR7 wrong:\n", o["r7_wrong"][["qs", "category", "pv", "lab", "why"]].to_string(index=False))
    print("\nsearch 2025 S5 replay:", o["s25_summary"], "\n2026 training:", o["s26_train"])
    print(o["s25"][["qs", "category", "pv", "rank", "lab", "exempt"]].to_string(index=False))
    print("decoys:", o["decoy_assistant"], o["decoy_search_2025"])
    print(o["R2"][o["R2"]["set"].str.startswith("decoy") & (o["R2"].lab != "B")][["set", "qs", "category", "pv", "lab", "why"]].to_string(index=False))
    with open(f"{SP}/audit2_round2_results.pkl", "wb") as fh:
        pickle.dump(o, fh)
