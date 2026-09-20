# -*- coding: utf-8 -*-
from sva_dom_edu_lib import *
F=frame().FRAME
e=edu(); U=users(e)
rows=[]
for k,g in U.items():
    print(f"{k}: n={len(g)} u_ds coverage={g.u_ds.notna().mean()*100:.1f}% p_ds=1 {fmt(int((g.p_ds==1).sum()),len(g))}")
codes=[f"U{i:02d}" for i in range(1,14)]
print("\n#### shares (all user rows | excl. U13 | PV-weighted all)")
for c in codes:
    line=f"{c} {F[c][0]:<8}"
    for k,g in U.items():
        n=len(g); kk=int((g.u_ds==c).sum()); gi=g[g.u_ds!="U13"]; ni=len(gi); ki=int((gi.u_ds==c).sum()) if c!="U13" else 0
        lo,hi=wilson(kk,n); pvw=g.loc[g.u_ds==c,"pv"].sum()/g.pv.sum()*100
        loi,hii=wilson(ki,ni) if c!="U13" else (float('nan'),float('nan'))
        rows.append(dict(code=c,name=F[c][0],surface=k,n=n,k=kk,pct=kk/n*100,lo=lo,hi=hi,n_interp=ni,k_interp=ki,pct_interp=(ki/ni*100 if c!="U13" else np.nan),lo_i=loi,hi_i=hii,pv_pct=pvw))
        line+=f" | {SN[k]} {kk/n*100:5.1f} [{lo:4.1f}–{hi:4.1f}]" + (f" ex13 {ki/ni*100:5.1f} [{loi:4.1f}–{hii:4.1f}]" if c!="U13" else " ex13   —        ") + (f" PV {pvw:5.1f}" if k!="assistant_random1k" else "")
    print(line)
T=pd.DataFrame(rows); T.to_csv(f"{SP}/sva_dom_edu_intent_shares.csv",index=False)
print("\n#### diffs (Newcombe), all rows and interpretable rows")
for a_,b_ in [("search_top1000","assistant_top1k"),("assistant_top1k","assistant_random1k")]:
    out=[];outi=[]
    for c in codes:
        A=U[a_];B=U[b_]
        d,lo,hi=newcombe(int((A.u_ds==c).sum()),len(A),int((B.u_ds==c).sum()),len(B)); out.append((abs(d),c,d,lo,hi))
        if c!="U13":
            Ai=A[A.u_ds!="U13"];Bi=B[B.u_ds!="U13"]; d,lo,hi=newcombe(int((Ai.u_ds==c).sum()),len(Ai),int((Bi.u_ds==c).sum()),len(Bi)); outi.append((abs(d),c,d,lo,hi))
    print(f"  {SN[a_]}→{SN[b_]} ALL: "+" ; ".join(f"{c}{F[c][0]} {d:+.1f} [{lo:+.1f},{hi:+.1f}]" for _,c,d,lo,hi in sorted(out,reverse=True)[:8]))
    print(f"  {SN[a_]}→{SN[b_]} EX13: "+" ; ".join(f"{c}{F[c][0]} {d:+.1f} [{lo:+.1f},{hi:+.1f}]" for _,c,d,lo,hi in sorted(outi,reverse=True)[:8]))
print("\n#### examples per class (head: top PV; tail: random)")
for c in codes:
    for k,g in U.items():
        x=g[g.u_ds==c]
        if len(x)==0: continue
        ex=x.sort_values("pv",ascending=False).head(12) if k!="assistant_random1k" else x.sample(min(12,len(x)),random_state=3)
        print(f"  {c} {SN[k]} (n={len(x)}): "+" | ".join(f"{q[:34]}({p:,.0f})" for q,p in zip(ex["query"],ex.pv)))
