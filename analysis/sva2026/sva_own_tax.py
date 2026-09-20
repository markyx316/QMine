# -*- coding: utf-8 -*-
"""Each surface's OWN taxonomy within each shared domain (user tier): top-down class shares + examples, bottom-up leaves."""
import sys
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from sva_common import *
a,s=load()
for d in CAT:
    c=cells(a,s,d); print(f"\n==================== {d} ====================")
    for k in ("搜索top1000","助手top1k","助手random1k"):
        x=c[k].copy(); x["query"]=x["query"].astype(str); pvc="wise_pv" if k.startswith("搜索") else "search_num"
        vc=x["td_l1_name"].fillna("(无)").value_counts(); n=len(x)
        print(f"\n  [{k}] n={n:,} — 自身top-down类 {x['td_l1_name'].nunique()} 个出现; bottom-up叶 {x['bu_leaf_name'].nunique()} 个出现")
        for cls,cnt in vc.head(12).items():
            ex=x[x.td_l1_name.fillna("(无)")==cls].sort_values(pvc,ascending=False)["query"]
            smp=list(ex.head(3))+list(ex.iloc[3:].sample(min(2,max(len(ex)-3,0)),random_state=3)) if len(ex)>3 else list(ex)
            print(f"    {cls:<22} {cnt/n*100:5.1f}% ({cnt:>4})  例: "+" | ".join(q[:26] for q in smp))
        lv=x["bu_leaf_name"].fillna("(无)").value_counts().head(8)
        print("    bottom-up top8: "+" ; ".join(f"{l[:20]} {v/n*100:.1f}%" for l,v in lv.items()))
