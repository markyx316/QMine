# -*- coding: utf-8 -*-
"""教育: recompute derived numbers quoted in the report."""
from sva_dom_edu_lib import *
e=edu(); U=users(e)
al=e[e.surface=="assistant_top1k"]; tot=al.pv.sum()
sysm=al.tier.isin(["S1_system_repeated","S2_system_template","S4_suggested_chip","S5_headline"])
print(f"head PV removed {al.loc[al.tier!='user','pv'].sum()/tot*100:.1f}% ; system tiers S1+S2+S4+S5 {al.loc[sysm,'pv'].sum()/tot*100:.1f}% ; C1 {al.loc[al.tier=='C1_content_free','pv'].sum()/tot*100:.1f}%")
S=U["search_top1000"]; Si=S[S.u_ds!="U13"]
d=max(abs((Si.u_ds==c).mean()-(S.u_ds==c).mean())*100 for c in [f"U{i:02d}" for i in range(1,13)]); print(f"search head ex-U13 n={len(Si)} max |diff| vs all = {d:.2f}pp")
wh=pd.read_csv(f"{SP}/sva_dom_edu_wraps_head.csv"); print("head wrap cores:",wh.core.value_counts().head(8).to_dict(), "n",len(wh))
wt=pd.read_csv(f"{SP}/sva_dom_edu_wraps_tail.csv"); u2=wt[wt.u_core=="U02"]
school=has(u2["core"].astype(str),r"(大学|学院|学校|中学|小学|工大|理工|师范)$")
print(f"tail wraps n={len(wt)}; U02-core rows {len(u2)}: school-name cores {int(school.sum())}, other cores {u2.loc[~school,'core'].value_counts().to_dict()}")
print("tail U02-core transitions (school cores):",(u2[school].u_core+"→"+u2[school].u_ds).value_counts().to_dict())
print("tail U02-core transitions (non-school cores):",(u2[~school].u_core+"→"+u2[~school].u_ds).value_counts().to_dict())
