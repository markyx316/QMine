# -*- coding: utf-8 -*-
"""Length-controlled semantic coverage + second-encoder robustness (bge-large-zh-v1.5)."""
import sys, numpy as np, torch
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from sva_common import *
from sentence_transformers import SentenceTransformer
dev="mps" if torch.backends.mps.is_available() else "cpu"
a,s=load(); nnp=pd.read_parquet(f"{SP}/sva_semantic_nn.parquet")
BK=[(0,6,"≤6字"),(7,10,"7–10"),(11,15,"11–15"),(16,25,"16–25"),(26,10**6,">25")]
def nn(E,R):
    best=np.full(len(E),-1.0,np.float32)
    for i in range(0,len(E),512): best[i:i+512]=(E[i:i+512]@R.T).max(1)
    return best
def fmt(sim,ln):
    out=[]
    for lo,hi,lab in BK:
        m=(ln>=lo)&(ln<=hi)
        out.append(f"{lab}: n={m.sum():>4} med={np.median(sim[m]):.2f} <0.70={((sim[m]<0.70).mean()*100 if m.sum() else float('nan')):4.0f}%" if m.sum()>=20 else f"{lab}: n={m.sum():>4}   —")
    return " | ".join(out)
for model in ("BAAI/bge-base-zh-v1.5","BAAI/bge-large-zh-v1.5"):
    m=SentenceTransformer(model,device=dev); print(f"\n################ {model}",flush=True)
    for d in CAT:
        c=cells(a,s,d); qs=c["搜索top10k"]["query"].astype(str).tolist(); ln=np.array([len(q) for q in qs])
        ES=m.encode(qs,batch_size=256,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)
        print(f"\n== {d} ==",flush=True)
        simA=nn(ES[:1000],ES[1000:]); print("  对照A 搜索头部1000→其余  "+fmt(simA,ln[:1000]))
        t=len(ES)-1000; simB=nn(ES[t:],ES[:t]); print("  对照B 搜索最深1000→其余  "+fmt(simB,ln[t:]))
        for k in ("助手top1k","助手random1k"):
            if model.endswith("base-zh-v1.5"):
                x=nnp[(nnp.domain==d)&(nnp.surface==k)]; sim=x["sim"].values; al=x["query"].str.len().values
            else:
                aq=c[k]["query"].astype(str).tolist(); EA=m.encode(aq,batch_size=128,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False); sim=nn(EA,ES); al=np.array([len(q) for q in aq])
            print(f"  {k:<10}→搜索top10k  "+fmt(sim,al)+f"   | 全体 med={np.median(sim):.3f} <0.70={(sim<0.70).mean()*100:.1f}%",flush=True)
