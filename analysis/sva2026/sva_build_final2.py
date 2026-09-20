# -*- coding: utf-8 -*-
"""Join final tiers, each surface's own labels and the single-instrument labels into one analysis table.
search_top1000 = rows by PV down to (and including) the 1,000th USER row, so non-user rows inside that span are kept."""
import os, sys, numpy as np
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from sva_common import *
LABELS=os.environ.get("LABELS", f"{SP}/label_full/unique_labels.parquet"); OUT=os.environ.get("FINAL_OUT", f"{SP}/sva_final_rows2.parquet")
a,s=load(); lab=pd.read_parquet(LABELS).reindex(columns=["query","u_ds","p_ds","u_qw","p_qw"]); lab["query"]=lab["query"].astype(str); lab=lab.drop_duplicates("query")
for extra in os.environ.get("EXTRA_LABEL_FILES","").split(","):   # search-side strings labelled later (same instrument)
    if extra and os.path.exists(extra):
        e=pd.read_parquet(extra); e=e[e.get("is_anchor",0)==0] if "is_anchor" in e.columns else e
        e=e.reindex(columns=["query","u_ds","p_ds","u_qw","p_qw"]); e["query"]=e["query"].astype(str)
        lab=pd.concat([lab,e[~e["query"].isin(set(lab["query"]))]]).drop_duplicates("query")
parts=[]
for d,cat in CAT.items():
    sd=s[s.domain==d].sort_values("wise_pv",ascending=False).head(1100).reset_index(drop=True)
    ur=(sd.tier=="user").values; pos=np.where(ur)[0]; assert len(pos)>=1000, (d,len(pos))
    sd=sd.iloc[:pos[999]+1].copy(); ur=ur[:pos[999]+1]
    parts.append(pd.DataFrame({"domain":d,"surface":"search_top1000","query":sd["query"].astype(str),"pv":sd.wise_pv.astype(float),
        "rank":sd["rank"],"user_rank":np.where(ur,np.cumsum(ur),0),"tier":sd.tier,"own_run":RUN[d],"own_code":sd.td_l1,
        "own_intent":sd.td_l1_name,"own_leaf":sd.bu_leaf_name,"l2":None}))
    sa=s[s.domain==d].sort_values("wise_pv",ascending=False).reset_index(drop=True); ua=(sa.tier=="user").values
    parts.append(pd.DataFrame({"domain":d,"surface":"search_top10k","query":sa["query"].astype(str),"pv":sa.wise_pv.astype(float),
        "rank":sa["rank"],"user_rank":np.where(ua,np.cumsum(ua),0),"tier":sa.tier,"own_run":RUN[d],"own_code":sa.td_l1,
        "own_intent":sa.td_l1_name,"own_leaf":sa.bu_leaf_name,"l2":None}))
    for snap,surf in (("top1k","assistant_top1k"),("random1k","assistant_random1k")):
        ad=a[(a.l1==cat)&(a.snapshot==snap)]
        parts.append(pd.DataFrame({"domain":d,"surface":surf,"query":ad["query"].astype(str),"pv":ad.search_num.astype(float),"rank":None,
            "user_rank":None,"tier":ad.tier,"own_run":ad.run_id,"own_code":ad.td_l1,"own_intent":ad.td_l1_name,"own_leaf":ad.bu_leaf_name,"l2":ad.l2}))
f=pd.concat(parts,ignore_index=True).merge(lab,on="query",how="left")
f.to_parquet(OUT,index=False)
g=f.groupby(["domain","surface"]).agg(rows=("query","size"),user=("tier",lambda x:(x=="user").sum()),ds_all=("u_ds",lambda x:x.notna().mean()),qw_all=("u_qw",lambda x:x.notna().mean()))
print(g.to_string()); print("rows written:",len(f),"->",OUT)
