# -*- coding: utf-8 -*-
"""Rebuild the deepseek unique-string labels from raw_ds.jsonl, reproducing build_cells' id assignment exactly
(cells.parquet → drop_duplicates → sample(frac=1, seed 20260910)). Merges the extra-neighbour labels too."""
import json, re, pandas as pd
SP=__import__("os").environ.get("SVA_WORKDIR", __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "work"))
def parse(raw):
    got={}
    for line in open(raw,encoding="utf-8"):
        r=json.loads(line); t=r.get("text")
        if not t: continue
        want=set(r["ids"])
        try: arr=json.loads(t[t.index("["):t.rindex("]")+1])
        except Exception: continue
        for o in arr:
            try: i,u,p=int(o.get("id")),str(o.get("u","")).upper(),int(o.get("p",0))
            except Exception: continue
            if i in want and re.fullmatch(r"U(0[1-9]|1[0-3])",u): got[i]=(u,1 if p else 0)
    return got
cells=pd.read_parquet(f"{SP}/label_full/cells.parquet")
uniq=pd.DataFrame({"query":cells["query"].drop_duplicates()}).sample(frac=1.0,random_state=20260910).reset_index(drop=True); uniq["id"]=uniq.index+1
g=parse(f"{SP}/label_full/raw_ds.jsonl")
uniq["u_ds"]=uniq.id.map(lambda i:g.get(i,(None,None))[0]); uniq["p_ds"]=uniq.id.map(lambda i:g.get(i,(None,None))[1])
print(f"main: unique {len(uniq):,}; labelled {uniq.u_ds.notna().sum():,}")
uniq.to_parquet(f"{SP}/label_full/unique_labels_ds_reconstructed.parquet",index=False)
ex=pd.read_parquet(f"{SP}/label_extra/labels.parquet"); print(f"extra: {len(ex):,} labelled {ex.u_ds.notna().sum():,}; overlap with main strings {ex['query'].isin(set(uniq['query'])).sum()}")
print(uniq.u_ds.value_counts().to_string())
