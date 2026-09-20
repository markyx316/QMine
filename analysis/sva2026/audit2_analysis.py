# -*- coding: utf-8 -*-
"""Audit 2 metrics: miss rates, precision, lists, and the u_ds bias check. Import compute() or run."""
import math
import pickle
import sys

import numpy as np
import pandas as pd

SP = __import__("os").environ.get("SVA_WORKDIR", __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "work"))
sys.path.insert(0, SP)
import audit2_labels as LB  # noqa: E402

CAT = {"金融": "金融", "医疗": "医疗", "教育": "教育培训", "影视": "影视动漫", "人物": "人物"}
SURF = {"top1k": "assistant_top1k", "random1k": "assistant_random1k"}
SEED = 20260910


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, max(0.0, (c - h) / d), min(1.0, (c + h) / d)


def load():
    v3 = pd.read_parquet(f"{SP}/clean_v3/assistant_tiered.parquet")
    s3 = pd.read_parquet(f"{SP}/clean_v3/search2026_tiered.parquet")
    F = pd.read_parquet(f"{SP}/sva_final_rows.parquet")
    v3["u_ds"] = None
    for d, c in CAT.items():
        for snap, surf in SURF.items():
            idx = v3.index[(v3.l1 == c) & (v3.snapshot == snap)]
            f = F[(F.domain == d) & (F.surface == surf)]
            assert len(idx) == len(f)
            assert (v3.loc[idx, "query"].astype(str).values == f["query"].astype(str).values).all()
            assert (v3.loc[idx, "tier"].values == f["tier"].values).all()
            v3.loc[idx, "u_ds"] = f["u_ds"].values
    fs = F[F.surface == "search_top1000"][["domain", "rank", "query", "u_ds"]]
    s3 = s3.merge(fs, on=["domain", "rank"], how="left", suffixes=("", "_f"))
    ok = s3.query_f.notna()
    assert (s3.loc[ok, "query"].astype(str) == s3.loc[ok, "query_f"].astype(str)).all()
    S = pd.read_parquet(f"{SP}/audit2_samples.parquet")
    LB.check(S)
    isA = S.surface == "assistant"
    S["lab"] = np.where(isA, S.qs.map(LB.LAB_ASSISTANT), S.qs.map(LB.LAB_SEARCH))
    S["why"] = np.where(isA, S.qs.map(LB.REASON_ASSISTANT), S.qs.map(LB.REASON_SEARCH))
    assert S.lab.isin(["A", "B", "C"]).all()
    S["u_ds"] = np.where(isA, S.rid.map(v3["u_ds"]), S.rid.map(s3["u_ds"]))
    return v3, s3, S


def counts(df):
    return {"n": len(df), "distinct": df.qs.nunique(), "A": int((df.lab == "A").sum()),
            "B": int((df.lab == "B").sum()), "C": int((df.lab == "C").sum())}


def row(name, df, target, what):
    c = counts(df)
    k = c[target]
    p, lo, hi = wilson(k, c["n"])
    pe, loe, hie = wilson(k, c["A"] + c["B"])
    pv = df.pv.sum()
    pvw = df.loc[df.lab == target, "pv"].sum() / pv if pv else float("nan")
    return {"sample": name, **c, "metric": what, "k": k, "rate": p, "lo": lo, "hi": hi,
            "rate_exclC": pe, "lo_exclC": loe, "hi_exclC": hie, "pv_weighted": pvw}


def boot_pv_share(df, B=5000):
    rng = np.random.default_rng(SEED)
    groups = [g for _, g in df.groupby("domain")]
    out = []
    for _ in range(B):
        parts = [g.iloc[rng.integers(0, len(g), len(g))] for g in groups]
        x = pd.concat(parts)
        out.append(x.loc[x.lab == "A", "pv"].sum() / x.pv.sum())
    return np.percentile(out, [2.5, 97.5])


def distortion(pop_codes: pd.Series, samp: pd.DataFrame, lab: str):
    """Per class: share among kept rows p, and how far removing the sample-estimated `lab` rows moves it.
    q_k = share of kept rows that are `lab` AND class k; r = sum q_k. p' = (p - q)/(1 - r).
    max inflation uses Wilson upper q_k with no other removals; max deflation uses Wilson lower q_k
    with the Wilson-upper total r."""
    n = len(samp)
    p = pop_codes.value_counts(normalize=True)
    kk = samp.loc[samp.lab == lab, "u_ds"].value_counts()
    rt = int((samp.lab == lab).sum())
    r, _, rU = wilson(rt, n)
    rows = []
    for code in sorted(set(p.index) | set(kk.index)):
        pk = float(p.get(code, 0.0))
        ak = int(kk.get(code, 0))
        q, qL, qU = wilson(ak, n)
        qc, qUc, qLc = min(q, pk), min(qU, pk), min(qL, pk)     # a class cannot lose more rows than it has
        p_after = (pk - qc) / (1 - r) if r < 1 else float("nan")
        infl = qUc * (1 - pk) / (1 - qUc)
        defl = pk * (rU - qLc) / (1 - rU)
        ns = int((samp.u_ds == code).sum())
        ks = int(((samp.u_ds == code) & (samp.lab == lab)).sum())
        w, wL, wU = wilson(ks, ns)
        rows.append({"u_ds": code, "share_kept": pk, "n_sample_in_class": ns, f"n_{lab}": ak,
                     f"within_class_{lab}": w, "within_lo": wL, "within_hi": wU, "q": q,
                     "share_after": p_after, "shift_pp": (pk - p_after) * 100,
                     "max_inflation_pp": infl * 100, "max_deflation_pp": defl * 100})
    return pd.DataFrame(rows), r, rU


def compute():
    v3, s3, S = load()
    five = v3.l1.isin(CAT.values())
    res = {"S": S}
    rows = []
    for name, snap in (("1_head_kept", "top1k"), ("2_tail_kept", "random1k")):
        df = S[S["sample"] == name]
        rows.append(row(name, df, "A", "miss rate A/n"))
        for d in CAT:
            rows.append(row(f"{name}:{d}", df[df.domain == d], "A", "miss rate A/n"))
        # population-weighted estimate over domain pools of kept user rows
        N = v3[five & (v3.snapshot == snap) & (v3.tier == "user")].l1.value_counts()
        w = {d: N[c] / N.sum() for d, c in CAT.items()}
        res[f"{name}_weighted_A"] = sum(w[d] * (df[df.domain == d].lab == "A").mean() for d in CAT)
        res[f"{name}_weighted_C"] = sum(w[d] * (df[df.domain == d].lab == "C").mean() for d in CAT)
        res[f"{name}_pv_boot"] = boot_pv_share(df)
    ch = S[S["sample"] == "3a_changed"]
    rem = ch[ch.v2_tier == "user"]
    rst = ch[ch.v2_tier != "user"]
    assert len(rem) + len(rst) == len(ch) and (rst.tier == "user").all() and (rem.tier != "user").all()
    rows.append(row("3a_removed (v2 user -> v3 non-user)", rem, "A", "precision A/n"))
    for (t2, t3), g in rem.groupby(["v2_tier", "tier"]):
        for snap, gg in g.groupby("snapshot"):
            rows.append(row(f"3a_removed:{t3}:{snap}", gg, "A", "precision A/n"))
    rows.append(row("3a_restored (v2 non-user -> v3 user)", rst, "B", "precision B/n"))
    for name, t in (("3b_S4", "S4_suggested_chip"), ("3c_S5", "S5_headline")):
        df = S[S["sample"] == name]
        rows.append(row(name, df, "A", "precision A/n"))
        for snap, g in df.groupby("snapshot"):
            rows.append(row(f"{name}:{snap}", g, "A", "precision A/n"))
    df = S[S["sample"] == "4a_search_kept"]
    rows.append(row("4a_search_kept", df, "A", "miss rate A/n"))
    df = S[S["sample"] == "4b_search_nonuser"]
    rows.append(row("4b_search_nonuser", df, "A", "precision A/n"))
    for t, g in df.groupby("tier"):
        rows.append(row(f"4b_search_nonuser:{t}", g, "A", "precision A/n"))
    res["table"] = pd.DataFrame(rows)

    cols = ["qs", "domain", "category", "snapshot", "pv", "tier", "v2_tier", "why", "u_ds", "sample"]
    kept = S["sample"].isin(["1_head_kept", "2_tail_kept", "4a_search_kept"])
    res["misses"] = S[kept & (S.lab == "A")][cols].sort_values(["sample", "why", "pv"], ascending=[True, True, False])
    res["kept_C"] = S[kept & (S.lab == "C")][cols].sort_values(["sample", "why", "pv"], ascending=[True, True, False])
    res["wrong_removed"] = rem[rem.lab != "A"][cols + ["lab"]].sort_values(["lab", "pv"], ascending=[True, False])
    res["wrong_restored"] = rst[rst.lab != "B"][cols + ["lab"]].sort_values(["lab", "pv"], ascending=[True, False])
    for name in ("3b_S4", "3c_S5", "4b_search_nonuser"):
        df = S[S["sample"] == name]
        res[f"wrong_{name}"] = df[df.lab != "A"][cols + ["lab"]].sort_values(["lab", "pv"], ascending=[True, False])

    # bias check: assistant head / tail kept rows; search top-1000 kept rows
    for name, snap in (("1_head_kept", "top1k"), ("2_tail_kept", "random1k")):
        pop = v3[five & (v3.snapshot == snap) & (v3.tier == "user")].u_ds
        samp = S[S["sample"] == name]
        for lab in ("A", "C"):
            t, r, rU = distortion(pop, samp, lab)
            res[f"dist_{name}_{lab}"] = (t, r, rU)
    pops = s3[(s3["rank"] <= 1000) & (s3.tier == "user")].u_ds
    for lab in ("A", "C"):
        res[f"dist_4a_search_kept_{lab}"] = distortion(pops, S[S["sample"] == "4a_search_kept"], lab)
    res["pop_pv_head"] = v3[five & (v3.snapshot == "top1k") & (v3.tier == "user")].search_num.sum()
    return res


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 40)
    pd.set_option("display.max_rows", 500)
    res = compute()
    t = res["table"].copy()
    for c in ("rate", "lo", "hi", "rate_exclC", "lo_exclC", "hi_exclC", "pv_weighted"):
        t[c] = (t[c] * 100).round(1)
    print(t.to_string(index=False))
    for k in ("1_head_kept", "2_tail_kept"):
        print(k, "domain-weighted A %.3f C %.3f ; PV-share bootstrap 95%%:" % (res[f"{k}_weighted_A"], res[f"{k}_weighted_C"]),
              np.round(res[f"{k}_pv_boot"] * 100, 1))
    for k in ("misses", "kept_C", "wrong_removed", "wrong_restored", "wrong_3b_S4", "wrong_3c_S5", "wrong_4b_search_nonuser"):
        print(f"\n== {k} (n={len(res[k])}) ==")
        print(res[k].to_string(index=False))
    for k in [k for k in res if k.startswith("dist_")]:
        tt, r, rU = res[k]
        print(f"\n== {k}: total q = {r*100:.1f}% (Wilson upper {rU*100:.1f}%) ==")
        print(tt.round(4).to_string(index=False))
    with open(f"{SP}/audit2_results.pkl", "wb") as fh:
        pickle.dump(res, fh)
