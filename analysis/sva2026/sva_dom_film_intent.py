# -*- coding: utf-8 -*-
"""影视: u_ds intent shares (all user rows / interpretable rows), Wilson CI, PV-weighted head shares,
Newcombe differences; own_intent top classes per surface."""
import sys, math, pandas as pd
SP=__import__("os").environ.get("SVA_WORKDIR", __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "work"))
sys.path.insert(0,SP); from sva_common import *
fm=frame(); NAME={k:v[0] for k,v in fm.FRAME.items()}; CODES=list(fm.FRAME)
F=pd.read_parquet(f"{SP}/sva_final_rows.parquet"); U=F[(F.domain=="影视")&(F.tier=="user")]
SURFS=["search_top1000","assistant_top1k","assistant_random1k"]
def wilson(k,n,z=1.96):
    p=k/n; den=1+z*z/n; c=(p+z*z/(2*n))/den; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den; return p,max(0,c-h),min(1,c+h)
def newcombe(k1,n1,k2,n2):
    p1,l1,u1=wilson(k1,n1); p2,l2,u2=wilson(k2,n2); d=p2-p1
    return d, d-math.sqrt((p2-l2)**2+(u1-p1)**2), d+math.sqrt((u2-p2)**2+(p1-l1)**2)
rows=[]
for view in ("all","interp"):
    for sf in SURFS:
        x=U[U.surface==sf]; x=x if view=="all" else x[x.u_ds!="U13"]; n=len(x); w=x.pv
        for c in CODES:
            m=x.u_ds==c; k=int(m.sum()); p,lo,hi=wilson(k,n)
            rows.append(dict(view=view,surface=sf,code=c,name=NAME[c],n=n,k=k,share=p,lo=lo,hi=hi,pv_share=(w*m).sum()/w.sum() if sf!="assistant_random1k" else float("nan")))
SH=pd.DataFrame(rows); SH.to_csv(f"{SP}/sva_dom_film_intent_shares.csv",index=False,encoding="utf-8-sig")
for view in ("all","interp"):
    print(f"\n### view={view}  n: "+" / ".join(f"{sf} {SH[(SH.view==view)&(SH.surface==sf)].n.iloc[0]}" for sf in SURFS))
    for c in CODES:
        t=[]
        for sf in SURFS:
            r=SH[(SH.view==view)&(SH.surface==sf)&(SH.code==c)].iloc[0]
            t.append(f"{r.share*100:5.1f}[{r.lo*100:4.1f}-{r.hi*100:4.1f}] k={r.k:<3}"+(f" PV{r.pv_share*100:5.1f}" if sf!="assistant_random1k" else ""))
        print(f" {c}{NAME[c]:<8} "+" | ".join(t))
dr=[]
for view in ("all","interp"):
    for s1,s2 in (("search_top1000","assistant_top1k"),("assistant_top1k","assistant_random1k")):
        for c in CODES:
            r1=SH[(SH.view==view)&(SH.surface==s1)&(SH.code==c)].iloc[0]; r2=SH[(SH.view==view)&(SH.surface==s2)&(SH.code==c)].iloc[0]
            d,lo,hi=newcombe(r1.k,r1.n,r2.k,r2.n); dr.append(dict(view=view,frm=s1,to=s2,code=c,name=NAME[c],p1=r1.share,p2=r2.share,diff=d,lo=lo,hi=hi))
DR=pd.DataFrame(dr); DR.to_csv(f"{SP}/sva_dom_film_intent_diffs.csv",index=False,encoding="utf-8-sig")
for (view,s1,s2),g in DR.groupby(["view","frm","to"],sort=False):
    g=g.reindex(g["diff"].abs().sort_values(ascending=False).index).head(8)
    print(f"\n### diffs {view} {s1}->{s2}"); [print(f"  {r.code}{r.name} {r.p1*100:.1f}->{r.p2*100:.1f} {r.diff*100:+.1f}[{r.lo*100:+.1f},{r.hi*100:+.1f}]") for r in g.itertuples()]
print("\n### p_ds (personal) share")
for sf in SURFS:
    x=U[U.surface==sf]; p,lo,hi=wilson(int(x.p_ds.sum()),len(x)); print(f"  {sf} {p*100:.1f}[{lo*100:.1f}-{hi*100:.1f}] k={int(x.p_ds.sum())}")
print("\n### own_intent per surface (row share, PV share)")
own=[]
for sf in SURFS:
    x=U[U.surface==sf]; n=len(x)
    g=x.groupby("own_intent").agg(k=("query","size"),pv=("pv","sum")).sort_values("k",ascending=False)
    g["share"]=g.k/n; g["pvs"]=g.pv/x.pv.sum()
    print(f"  == {sf} n={n} classes={len(g)}")
    for nm,r in g.iterrows():
        p,lo,hi=wilson(int(r.k),n); own.append(dict(surface=sf,own_intent=nm,n=n,k=int(r.k),share=p,lo=lo,hi=hi,pv_share=r.pvs))
        ex=x[x.own_intent==nm].sort_values("pv",ascending=False)["query"].head(4).str[:22].tolist()
        print(f"   {nm:<24} k={int(r.k):>4} {p*100:5.1f}[{lo*100:.1f}-{hi*100:.1f}] PV{r.pvs*100:5.1f}  例: "+" | ".join(ex))
pd.DataFrame(own).to_csv(f"{SP}/sva_dom_film_own_intent.csv",index=False,encoding="utf-8-sig")
