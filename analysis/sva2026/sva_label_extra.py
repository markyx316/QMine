# -*- coding: utf-8 -*-
"""Label the nearest search counterparts (sim 0.70–0.999) that fall outside the labelled search head, with the SAME
instrument (tools/label_unified_intent.py, deepseek). Dry run by default; --run spends."""
import sys, importlib.util
from pathlib import Path
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from sva_common import *
spec=importlib.util.spec_from_file_location("lui",f"{QM}/tools/label_unified_intent.py"); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
NN=pd.read_parquet(f"{SP}/sva_semantic_nn.parquet")
have=set(pd.read_parquet(f"{SP}/label_full/cells.parquet")["query"].astype(str))
cand=NN[(NN.sim>=0.70)&(NN.sim<0.999)]["nn_query"].astype(str)
todo=pd.DataFrame({"query":sorted(set(cand)-have)})
print(f"assistant rows with 0.70≤sim<0.999: {len(cand):,}; unique neighbours {cand.nunique():,}; already labelled {len(set(cand)&have):,}; to label {len(todo):,}")
if "--run" in sys.argv:
    m._env(); out=Path(f"{SP}/label_extra"); out.mkdir(exist_ok=True)
    todo=todo.sample(frac=1.0,random_state=11).reset_index(drop=True); todo["id"]=todo.index+1
    lab=m.run("ds",todo,out,25,16); lab.to_parquet(out/"labels.parquet",index=False)
    print(f"labelled {lab.u_ds.notna().sum():,}/{len(lab):,}")
