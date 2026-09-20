# -*- coding: utf-8 -*-
"""Extend the single-instrument intent labels to EVERY string of the five search top-10k exports.
Same prompt/model as tools/label_unified_intent.py. 1,000 already-labelled strings are mixed in as anchors
(test-retest under different batch neighbours). Qwen labels 10% of the new strings once the main Qwen pass is done."""
import sys, time, importlib.util
from pathlib import Path
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from sva_common import *
spec=importlib.util.spec_from_file_location("lui",f"{QM}/tools/label_unified_intent.py"); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pd.read_parquet(f"{SP}/clean_v2/search2026_tiered.parquet")
allq=pd.Series(s["query"].astype(str).unique()); have=set(pd.read_parquet(f"{SP}/label_full/cells.parquet")["query"].astype(str))
new=allq[~allq.isin(have)]; anch=pd.Series(sorted(have)).sample(1000,random_state=5)
df=pd.DataFrame({"query":list(new)+list(anch),"is_anchor":[0]*len(new)+[1]*len(anch)}).sample(frac=1,random_state=20260911).reset_index(drop=True)
df["id"]=df.index+1; out=Path(f"{SP}/label_search10k"); out.mkdir(exist_ok=True); df.to_parquet(out/"todo.parquet",index=False)
print(f"new strings {len(new):,} + anchors {len(anch):,} = {len(df):,}",flush=True)
m._env(); t=time.time()
lab=m.run("ds",df[["id","query"]],out,25,20); df=df.merge(lab[["id","u_ds","p_ds"]],on="id",how="left"); df.to_parquet(out/"labels_ds.parquet",index=False)
print(f"deepseek labelled {df.u_ds.notna().sum():,}/{len(df):,} in {time.time()-t:.0f}s",flush=True)
for _ in range(180):
    if Path(f"{SP}/label_full/unique_labels.parquet").exists(): break
    time.sleep(60)
sub=df[df.is_anchor==0].sample(frac=0.10,random_state=7); t=time.time()
lab2=m.run("qw",sub[["id","query"]],out,25,8); df=df.merge(lab2[["id","u_qw","p_qw"]],on="id",how="left"); df.to_parquet(out/"labels.parquet",index=False)
print(f"qwen labelled {df.u_qw.notna().sum():,}/{len(sub):,} in {time.time()-t:.0f}s",flush=True)
