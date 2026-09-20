# -*- coding: utf-8 -*-
"""影视 search_top10k lens from sva_final_rows2.parquet: coverage gate (>=95% u_ds on user rows), consistency with sva_final_rows,
then u_ds shares (all / interpretable) with Wilson CI + PV share, Newcombe diffs vs search_top1000 and assistant_top1k."""
import sys, math, pandas as pd, numpy as np
SP=__import__("os").environ.get("SVA_WORKDIR", __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "work"))
sys.path.insert(0,SP); from sva_common import *; import sva_dom_film_rx as R
fm=frame(); NAME={k:v[0] for k,v in fm.FRAME.items()}; CODES=list(fm.FRAME)
def wilson(k,n,z=1.96):
    p=k/n; den=1+z*z/n; c=(p+z*z/(2*n))/den; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den; return p,max(0,c-h),min(1,c+h)
def newcombe(k1,n1,k2,n2):
    p1,l1,u1=wilson(k1,n1); p2,l2,u2=wilson(k2,n2); d=p2-p1
    return d, d-math.sqrt((p2-l2)**2+(u1-p1)**2), d+math.sqrt((u2-p2)**2+(p1-l1)**2)
F2=pd.read_parquet(f"{SP}/sva_final_rows2.parquet"); F1=pd.read_parquet(f"{SP}/sva_final_rows.parquet")
print(F2.groupby(["domain","surface"]).size().unstack(fill_value=0))
D2=F2[F2.domain=="影视"]; t=D2[(D2.surface=="search_top10k")&(D2.tier=="user")]
cov=t.u_ds.notna().mean(); print(f"影视 search_top10k user rows n={len(t)} u_ds coverage {cov*100:.2f}% ; u_qw coverage {t.u_qw.notna().mean()*100:.1f}%")
# consistency with final_rows on the surfaces both have
k=["surface","query","tier"]
for sf in ("search_top1000","assistant_top1k","assistant_random1k"):
    a1=F1[(F1.domain=="影视")&(F1.surface==sf)&(F1.tier=="user")][["query","u_ds"]].astype(str); a2=D2[(D2.surface==sf)&(D2.tier=="user")][["query","u_ds"]].astype(str)
    m=a1.drop_duplicates("query").merge(a2.drop_duplicates("query"),on="query",suffixes=("_1","_2"))
    print(f"  {sf}: rows F1 {len(a1)} F2 {len(a2)}; matched {len(m)}; same u_ds {(m.u_ds_1==m.u_ds_2).mean()*100:.1f}%")
# top10k user rows vs top1000 within F2: u_ds on first 1000 user rows equal to search_top1000?
if cov>=0.95:
    U=D2[D2.tier=="user"]
    S={"search_top1000":F1[(F1.domain=="影视")&(F1.surface=="search_top1000")&(F1.tier=="user")],"search_top10k":t.dropna(subset=["u_ds"]),
       "assistant_top1k":F1[(F1.domain=="影视")&(F1.surface=="assistant_top1k")&(F1.tier=="user")]}
    rows=[]
    for view in ("all","interp"):
        for sf,x in S.items():
            x=x if view=="all" else x[x.u_ds!="U13"]; n=len(x); w=x.pv.astype(float)
            for cd in CODES:
                mm=x.u_ds==cd; kk=int(mm.sum()); p,lo,hi=wilson(kk,n); rows.append(dict(view=view,surface=sf,code=cd,name=NAME[cd],n=n,k=kk,share=p,lo=lo,hi=hi,pv_share=float((w*mm).sum()/w.sum())))
    SH=pd.DataFrame(rows); SH.to_csv(f"{SP}/sva_dom_film_intent_top10k.csv",index=False,encoding="utf-8-sig")
    for view in ("all","interp"):
        print(f"\n### {view}: n "+" / ".join(f"{sf} {SH[(SH.view==view)&(SH.surface==sf)].n.iloc[0]}" for sf in S))
        for cd in CODES:
            r=[SH[(SH.view==view)&(SH.surface==sf)&(SH.code==cd)].iloc[0] for sf in S]
            d1=newcombe(r[0].k,r[0].n,r[1].k,r[1].n); d2=newcombe(r[1].k,r[1].n,r[2].k,r[2].n)
            print(f"  {cd}{NAME[cd]:<8} 1k {r[0].share*100:5.1f} | 10k {r[1].share*100:5.1f}[{r[1].lo*100:.1f}-{r[1].hi*100:.1f}] PV{r[1].pv_share*100:5.1f} | head {r[2].share*100:5.1f} || 1k→10k {d1[0]*100:+.1f}[{d1[1]*100:+.1f},{d1[2]*100:+.1f}] | 10k→head {d2[0]*100:+.1f}[{d2[1]*100:+.1f},{d2[2]*100:+.1f}]")
    x=S["search_top10k"]; u=x[x.u_ds=="U01"]; live=u["query"].astype(str).map(lambda q: R.hit("电视频道/直播",q))
    k10=int(len(u)-live.sum()); ah=S["assistant_top1k"]; kh=int(((ah.u_ds=="U01")&~ah["query"].astype(str).map(lambda q: R.hit("电视频道/直播",q))).sum())
    d=newcombe(k10,len(x),kh,len(ah)); print(f"\n  U01 excl live TV: 10k {k10}/{len(x)}={k10/len(x)*100:.1f}% → head {kh}/{len(ah)}={kh/len(ah)*100:.1f}% diff {d[0]*100:+.1f}[{d[1]*100:+.1f},{d[2]*100:+.1f}]")
    print("  U12 10k examples count only:",int((x.u_ds=="U12").sum()), " U10 10k:",int((x.u_ds=="U10").sum()))
    for cd in ("U04","U08","U13","U03","U05"):
        y=x[x.u_ds==cd].sort_values("pv",ascending=False); y=y[[not R.adult_like(q,uu,None) for q,uu in zip(y["query"].astype(str),y.u_ds)]]
        print(f"  10k {cd} 头例: "+" | ".join(y["query"].astype(str).head(6)))
else:
    print("coverage below 95% — do not report unified top10k shares")
