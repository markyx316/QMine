# -*- coding: utf-8 -*-
"""教育 §1: n per surface per tier, PV share by tier, removed examples (clean_v3)."""
import sys; SP=__import__("os").environ.get("SVA_WORKDIR", __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "work")); sys.path.insert(0,SP)
from sva_common import *
a,s=load()
f=pd.read_parquet(f"{SP}/sva_final_rows.parquet"); e=f[f.domain=="教育"]
rows=[]
for surf,g in e.groupby("surface"):
    tot_n=len(g); tot_pv=g.pv.sum()
    for t,gg in g.groupby("tier"):
        rows.append(dict(surface=surf,tier=t,n=len(gg),row_pct=len(gg)/tot_n*100,pv=gg.pv.sum(),pv_pct=gg.pv.sum()/tot_pv*100))
    rows.append(dict(surface=surf,tier="ALL",n=tot_n,row_pct=100,pv=tot_pv,pv_pct=100))
t=pd.DataFrame(rows); print(t.to_string(index=False,float_format=lambda x:f"{x:,.1f}"))
t.to_csv(f"{SP}/sva_dom_edu_tiers.csv",index=False)
# search top10k tiers
ss=s[s.domain=="教育"]; print("\nsearch top10k tiers:", ss.groupby("tier").agg(n=("query","size"),pv=("wise_pv","sum")).to_string())
print("search top10k PV rank cutoffs: 1000th user", ss[ss.tier=="user"].sort_values("wise_pv",ascending=False).wise_pv.iloc[999], "last", ss[ss.tier=="user"].wise_pv.min(), "n user", (ss.tier=="user").sum())
aa=a[(a.l1=="教育培训")]
for snap in ["top1k","random1k"]:
    x=aa[aa.snapshot==snap]; print(f"\nassistant {snap}: n={len(x)} pv={x.search_num.sum():,} min pv={x.search_num.min()} median pv={x.search_num.median()} user min pv={x[x.tier=='user'].search_num.min()}")
    print(x.groupby("l2").size().sort_values(ascending=False).head(15).to_string())
# examples of removed rows, top PV per tier
for t_,g in e[(e.tier!="user")].groupby(["surface","tier"]):
    g=g.sort_values("pv",ascending=False)
    print(f"\n[{t_[0]} | {t_[1]}] n={len(g)} PV={g.pv.sum():,.0f}: "+" | ".join(f"{q}({p:,.0f})" for q,p in zip(g["query"].head(14),g.pv.head(14))))
