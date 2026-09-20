# -*- coding: utf-8 -*-
from sva_dom_edu_lib import *
e=edu(); U=users(e); H=U["assistant_top1k"]
h13=H[H.u_ds=="U13"].sort_values("pv",ascending=False)
print("ALL head U13 (query|pv|l2|own_leaf):")
print(" ; ".join(f"{q}|{int(p)}|{l}|{o}" for q,p,l,o in zip(h13['query'],h13.pv,h13.l2,h13.own_leaf)))
al=e[e.surface=="assistant_top1k"]; tot=al.pv.sum()
s1=al[al.tier=="S1_system_repeated"].groupby("query").pv.sum().sort_values(ascending=False)
print(f"\nhead ALL-tier PV={tot:,.0f}; S1 PV share={s1.sum()/tot*100:.1f}%")
for q in ["嗯","需要","要","好","可以","继续"]:
    print(f"  S1 '{q}': PV {s1.get(q,0):,.0f} = {s1.get(q,0)/tot*100:.1f}% of head PV")
cont=s1[s1.index.str.contains("继续吧")].sum(); print(f"  S1 '…继续吧' chips: PV {cont:,.0f} = {cont/tot*100:.1f}%")
print(f"  嗯+需要+要: {(s1.get('嗯',0)+s1.get('需要',0)+s1.get('要',0))/tot*100:.1f}%")
# Latin-script user rows in head by l2 and u_ds
lat=H[has(H["query"],r"^[A-Za-z][A-Za-z\s'’,.!?]*$")]
print(f"\nLatin-only head user rows n={len(lat)}; l2:",lat.l2.value_counts().to_dict(),"u_ds:",lat.u_ds.value_counts().to_dict())
print(" ; ".join(f"{q}|{int(p)}|{l}|{u}" for q,p,l,u in zip(lat.sort_values('pv',ascending=False)['query'],lat.sort_values('pv',ascending=False).pv,lat.sort_values('pv',ascending=False).l2,lat.sort_values('pv',ascending=False).u_ds)))
