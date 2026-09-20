# -*- coding: utf-8 -*-
"""人物 sections 1 and 3: tiers (rows/PV) and unified intent (u_ds) with Wilson / Newcombe CIs; own taxonomies."""
import sys, math, numpy as np
SP=__import__("os").environ.get("SVA_WORKDIR", __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "work"))
sys.path.insert(0, SP)
from sva_common import *
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
fm = frame(); CODES = list(fm.FRAME); NAME = {k: v[0] for k, v in fm.FRAME.items()}
def wilson(k, n, z=1.96):
    if n == 0: return float("nan"), float("nan"), float("nan")
    p = k/n; den = 1+z*z/n; c = (p+z*z/(2*n))/den; h = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den; return p, max(0.0, c-h), min(1.0, c+h)
def newcombe(k1, n1, k2, n2):
    p1, l1, u1 = wilson(k1, n1); p2, l2, u2 = wilson(k2, n2); d = p2-p1
    return d, d-math.sqrt((p2-l2)**2+(u1-p1)**2), d+math.sqrt((u2-p2)**2+(p1-l1)**2)
D = "人物"
a, s = load()
# ================= 1. tiers =================
sd = s[s.domain == D].sort_values("wise_pv", ascending=False).reset_index(drop=True)
pos = np.where((sd.tier == "user").values)[0]; span = sd.iloc[:pos[999]+1]
ad = a[a.l1 == CAT[D]]
TC = {"search_top1000": (span, "wise_pv"), "search_top10k": (sd, "wise_pv"),
      "assistant_top1k": (ad[ad.snapshot == "top1k"], "search_num"), "assistant_random1k": (ad[ad.snapshot == "random1k"], "search_num")}
rows = []
for k, (df, pv) in TC.items():
    N = len(df); PV = df[pv].sum()
    for t, g in df.groupby("tier"):
        rows.append(dict(surface=k, tier=t, rows=len(g), row_share=len(g)/N, pv=int(g[pv].sum()), pv_share=g[pv].sum()/PV, N=N, PV=int(PV)))
T = pd.DataFrame(rows); T.to_csv(f"{SP}/sva_dom_ppl_tiers.csv", index=False, encoding="utf-8-sig")
print("#### 1. TIERS (filter: domain==人物; search span = PV-sorted rows down to the 1000th user row; assistant l1==人物)")
print(T.to_string(index=False, formatters={"row_share": "{:.1%}".format, "pv_share": "{:.1%}".format}))
print("\n#### 1b. removed examples (top PV per tier)")
for k, (df, pv) in TC.items():
    for t, g in df[df.tier != "user"].groupby("tier"):
        print(f"  [{k}] {t} n={len(g)} PV={int(g[pv].sum()):,}: " + " | ".join(f"{q[:60]}({int(v)})" for q, v in g.sort_values(pv, ascending=False)[["query", pv]].head(14).values))
# cleaning impact on key shape figures (assistant top1k, all rows vs user rows)
Q = r"(怎么|如何|为什么|为何|什么|哪个|哪些|哪里|多少|吗|呢|是不是|是否|谁|\?|？)"
for k in ("assistant_top1k", "assistant_random1k"):
    df, pv = TC[k]; u = df[df.tier == "user"]
    for nm, x in (("all", df), ("user", u)):
        qq = x["query"].astype(str); w = x[pv].astype(float)
        print(f"  [{k}] {nm}: n={len(x)} len_med={qq.str.len().median():.0f} >40字={(qq.str.len()>40).mean()*100:.1f}% (PV {(w*(qq.str.len()>40)).sum()/w.sum()*100:.1f}%) 疑问={(qq.str.contains(Q)).mean()*100:.1f}% (PV {(w*qq.str.contains(Q)).sum()/w.sum()*100:.1f}%)")
# ================= 3. unified intent =================
F = pd.read_parquet(f"{SP}/sva_final_rows.parquet"); F["query"] = F["query"].astype(str)
FD = F[F.domain == D]
print("\n#### 3a. coverage (user rows): ", {sf: (int(((FD.surface==sf)&(FD.tier=="user")).sum()), int(((FD.surface==sf)&(FD.tier=="user")&FD.u_ds.notna()).sum()), int(((FD.surface==sf)&(FD.tier=="user")&FD.u_qw.notna()).sum())) for sf in FD.surface.unique()})
U = FD[FD.tier == "user"].copy()
SURFS = ["search_top1000", "assistant_top1k", "assistant_random1k"]; HEADS = {"search_top1000", "assistant_top1k"}
sh = []
for view in ("all", "interpretable"):
    for sf in SURFS:
        x = U[U.surface == sf].dropna(subset=["u_ds"])
        if view == "interpretable": x = x[x.u_ds != "U13"]
        n = len(x); w = x.pv.astype(float)
        for c in CODES:
            if view == "interpretable" and c == "U13": continue
            mk = x.u_ds == c; k = int(mk.sum()); p, lo, hi = wilson(k, n)
            sh.append(dict(view=view, surface=sf, code=c, name=NAME[c], n=n, k=k, share=p, lo=lo, hi=hi, pv_share=(float((w*mk).sum()/w.sum()) if sf in HEADS else float("nan"))))
SH = pd.DataFrame(sh); SH.to_csv(f"{SP}/sva_dom_ppl_intent_shares.csv", index=False, encoding="utf-8-sig")
for view in ("all", "interpretable"):
    print(f"\n#### 3b. u_ds shares view={view} (filter: final_rows domain==人物 & tier==user" + (" & u_ds!=U13" if view == "interpretable" else "") + ")")
    x = SH[SH.view == view]
    print("  n: " + " / ".join(f"{sf} {int(x[x.surface==sf].n.iloc[0])}" for sf in SURFS))
    for c in x.code.unique():
        print(f"   {c} {NAME[c]:<8} " + " | ".join(f"{sf[:12]} {r.share*100:5.1f} [{r.lo*100:4.1f}–{r.hi*100:4.1f}] k={r.k}" + (f" PV{r.pv_share*100:5.1f}" if sf in HEADS else "") for sf in SURFS for r in [x[(x.surface==sf)&(x.code==c)].iloc[0]]))
dr = []
for view in ("all", "interpretable"):
    x = SH[SH.view == view]
    for s1, s2 in (("search_top1000", "assistant_top1k"), ("assistant_top1k", "assistant_random1k")):
        for c in x.code.unique():
            r1 = x[(x.surface==s1)&(x.code==c)].iloc[0]; r2 = x[(x.surface==s2)&(x.code==c)].iloc[0]
            d, lo, hi = newcombe(r1.k, r1.n, r2.k, r2.n)
            dr.append(dict(view=view, pair=f"{s1}→{s2}", code=c, name=NAME[c], p1=r1.share, p2=r2.share, diff=d, lo=lo, hi=hi, sig=(lo > 0 or hi < 0)))
DR = pd.DataFrame(dr); DR.to_csv(f"{SP}/sva_dom_ppl_intent_diffs.csv", index=False, encoding="utf-8-sig")
print("\n#### 3c. diffs (pp, Newcombe 95%)")
for (view, pair), g in DR.groupby(["view", "pair"], sort=False):
    g = g.reindex(g["diff"].abs().sort_values(ascending=False).index).head(8)
    print(f"  [{view}] {pair}: " + " ; ".join(f"{r.code}{r['name']} {r.p1*100:.1f}→{r.p2*100:.1f} ({r['diff']*100:+.1f} [{r.lo*100:+.1f},{r.hi*100:+.1f}]{'*' if r.sig else ''})" for _, r in g.iterrows()))
print("\n#### 3d. examples per code × surface (heads: top PV; tail: random)")
for c in CODES:
    for sf in SURFS:
        x = U[(U.surface == sf) & (U.u_ds == c)]
        if not len(x): continue
        ex = x.sort_values("pv", ascending=False).head(14) if sf in HEADS else x.sample(min(14, len(x)), random_state=11)
        print(f"  {c}{NAME[c]} [{sf}] n={len(x)}: " + " | ".join(f"{q[:40]}({int(v)})" for q, v in ex[["query", "pv"]].values))
print("\n#### 3e. personal-situation flag p_ds==1")
for sf in SURFS:
    x = U[U.surface == sf]; k = int((x.p_ds == 1).sum()); p, lo, hi = wilson(k, len(x)); print(f"  {sf}: {k}/{len(x)} = {p*100:.1f}% [{lo*100:.1f}–{hi*100:.1f}]")
# ================= own taxonomies =================
print("\n#### 3f. own taxonomy (own_intent) shares within surface, user rows")
ot = []
c10 = cells(a, s, D)
OWN = {sf: U[U.surface == sf][["own_intent", "own_leaf", "pv", "query"]] for sf in SURFS}
st = c10["搜索top10k"]; OWN["search_top10k(cells)"] = pd.DataFrame({"own_intent": st.td_l1_name, "own_leaf": st.bu_leaf_name, "pv": st.wise_pv, "query": st["query"].astype(str)})
for sf, x in OWN.items():
    n = len(x); vc = x.own_intent.value_counts()
    print(f"  [{sf}] n={n} classes={len(vc)}")
    for nm, k in vc.items():
        p, lo, hi = wilson(int(k), n); w = x.pv.astype(float); pvs = float(w[x.own_intent == nm].sum()/w.sum())
        ot.append(dict(surface=sf, own_intent=nm, n=n, k=int(k), share=p, lo=lo, hi=hi, pv_share=pvs))
        exs = x[x.own_intent == nm].sort_values("pv", ascending=False)["query"].head(6).str[:30].tolist()
        print(f"     {nm:<16} {p*100:5.1f}% [{lo*100:4.1f}–{hi*100:4.1f}] k={k} PV{pvs*100:5.1f}%  例: " + " | ".join(exs))
    lv = x.own_leaf.value_counts().head(10)
    print("     leaves top10: " + " ; ".join(f"{nm} {k/n*100:.1f}%" for nm, k in lv.items()))
pd.DataFrame(ot).to_csv(f"{SP}/sva_dom_ppl_own_tax.csv", index=False, encoding="utf-8-sig")
# cross-tab: own_intent vs u_ds on each surface (how the slicing differs)
print("\n#### 3g. own_intent × u_ds (row counts)")
for sf in SURFS:
    x = U[U.surface == sf]; ct = pd.crosstab(x.own_intent, x.u_ds)
    print(f"  [{sf}]"); print(ct.to_string())
