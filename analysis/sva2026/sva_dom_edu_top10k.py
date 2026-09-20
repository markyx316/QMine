# -*- coding: utf-8 -*-
"""教育: search_top10k intent shares from sva_final_rows2.parquet, only if it exists and user-row u_ds coverage >= 95%."""
import os
from sva_dom_edu_lib import *
p=f"{SP}/sva_final_rows2.parquet"
if not os.path.exists(p):
    print("sva_final_rows2.parquet MISSING -> use cells() fallback"); raise SystemExit
f=pd.read_parquet(p); f["query"]=f["query"].astype(str); e=f[f.domain=="教育"]
t=e[(e.surface=="search_top10k")&(e.tier=="user")]; cov=t.u_ds.notna().mean()
print(f"search_top10k user n={len(t)} u_ds coverage {cov*100:.1f}%")
if cov<0.95: print("coverage < 95% -> use cells() fallback"); raise SystemExit
F=frame().FRAME; H=e[(e.surface=="assistant_top1k")&(e.tier=="user")]; S1=e[(e.surface=="search_top1000")&(e.tier=="user")]
tt=t[t.u_ds.notna()]; rows=[]
for c in [f"U{i:02d}" for i in range(1,14)]:
    k=int((tt.u_ds==c).sum()); lo,hi=wilson(k,len(tt)); d,dl,dh=newcombe(k,len(tt),int((H.u_ds==c).sum()),len(H))
    rows.append(dict(code=c,name=F[c][0],n=len(tt),k=k,pct=k/len(tt)*100,lo=lo,hi=hi,top1000_pct=(S1.u_ds==c).mean()*100,head_pct=(H.u_ds==c).mean()*100,diff_to_head=d,d_lo=dl,d_hi=dh))
    print(f"  {c} {F[c][0]:<8} top10k {k/len(tt)*100:5.1f} [{lo:4.1f}–{hi:4.1f}] | top1000 {(S1.u_ds==c).mean()*100:5.1f} | 助手头 {(H.u_ds==c).mean()*100:5.1f} | top10k→助手头 {d:+.1f} [{dl:+.1f},{dh:+.1f}]")
pd.DataFrame(rows).to_csv(f"{SP}/sva_dom_edu_intent_top10k.csv",index=False)
