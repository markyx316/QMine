# -*- coding: utf-8 -*-
"""What v3 (second-audit rules) changed relative to v2: transitions, PV, examples; head traffic composition."""
import sys; sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from sva_common import *
v2a=pd.read_parquet(f"{SP}/clean_v2/assistant_tiered.parquet"); v3a=pd.read_parquet(f"{SP}/clean_v3/assistant_tiered.parquet")
v2s=pd.read_parquet(f"{SP}/clean_v2/search2026_tiered.parquet"); v3s=pd.read_parquet(f"{SP}/clean_v3/search2026_tiered.parquet")
assert (v2a["query"].astype(str).values==v3a["query"].astype(str).values).all() and (v2s["query"].astype(str).values==v3s["query"].astype(str).values).all()
five=v3a.l1.isin(list(CAT.values()))
for scope,mask in (("5域",five),("全33",pd.Series(True,index=v3a.index))):
    for snap in ("top1k","random1k"):
        m=mask&(v3a.snapshot==snap); x=pd.DataFrame({"v2":v2a.tier[m],"v3":v3a.tier[m],"pv":v3a.search_num[m]})
        ch=x[x.v2!=x.v3]
        print(f"\n#### {scope} {snap}: rows {m.sum():,}; changed {len(ch):,} ({len(ch)/m.sum()*100:.1f}%), PV {ch.pv.sum():,.0f} ({ch.pv.sum()/x.pv.sum()*100:.1f}%)")
        print("   user rows v2→v3: %d→%d ; user PV share %.1f%%→%.1f%%" % ((x.v2=="user").sum(),(x.v3=="user").sum(),x.pv[x.v2=="user"].sum()/x.pv.sum()*100,x.pv[x.v3=="user"].sum()/x.pv.sum()*100))
        g=ch.groupby(["v2","v3"]).agg(rows=("pv","size"),pv=("pv","sum")).sort_values("rows",ascending=False)
        print(g.to_string())
print("\n#### 5域 by domain × snapshot: user rows and user PV share v2→v3; tier composition v3 (rows%/PV%)")
for d,cat in CAT.items():
    for snap in ("top1k","random1k"):
        m=(v3a.l1==cat)&(v3a.snapshot==snap); pv=v3a.search_num[m]
        comp=" | ".join(f"{t} {((v3a.tier[m]==t).mean()*100):.1f}%/{pv[v3a.tier[m]==t].sum()/pv.sum()*100:.1f}%" for t in sorted(v3a.tier[m].unique()))
        print(f"  {d} {snap}: user {int((v2a.tier[m]=='user').sum())}→{int((v3a.tier[m]=='user').sum())} ; userPV {pv[v2a.tier[m]=='user'].sum()/pv.sum()*100:.1f}%→{pv[v3a.tier[m]=='user'].sum()/pv.sum()*100:.1f}%  ||  {comp}")
print("\n#### EXAMPLES (5 domains) — newly removed (v2 user → v3 non-user) and restored (v2 non-user → v3 user)")
for d,cat in CAT.items():
    m=(v3a.l1==cat); x=pd.DataFrame({"q":v3a["query"].astype(str)[m],"snap":v3a.snapshot[m],"v2":v2a.tier[m],"v3":v3a.tier[m],"pv":v3a.search_num[m]})
    print(f"\n== {d} ==")
    for (a2,a3),g in x[x.v2!=x.v3].groupby(["v2","v3"]):
        g=g.sort_values("pv",ascending=False)
        print(f"  {a2}→{a3} n={len(g)} PV={int(g.pv.sum()):,}: "+" | ".join(f"{q[:30]}({s[:3]},{int(p)})" for q,s,p in zip(g.q.head(10),g.snap.head(10),g.pv.head(10))))
print("\n#### SEARCH v2→v3 by domain")
for d in CAT:
    m=v3s.domain==d; x=pd.DataFrame({"q":v3s["query"].astype(str)[m],"v2":v2s.tier[m],"v3":v3s.tier[m],"pv":v3s.wise_pv[m],"rank":v3s["rank"][m]})
    ch=x[x.v2!=x.v3]; print(f"  {d}: changed {len(ch)}; " + " ; ".join(f"{a2}→{a3} {len(g)}: "+" | ".join(f"{q[:20]}(#{r})" for q,r in zip(g.sort_values('pv',ascending=False).q.head(8),g.sort_values('pv',ascending=False)['rank'].head(8))) for (a2,a3),g in ch.groupby(["v2","v3"])))
