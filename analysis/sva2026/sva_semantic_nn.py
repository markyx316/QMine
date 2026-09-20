# -*- coding: utf-8 -*-
"""Semantic coverage: for each assistant query (user tier), its nearest search top-10k query in the same
domain (bge-base-zh-v1.5 cosine). Controls: search head -> rest of search; search deepest 1k -> rest."""
import sys, os, numpy as np, torch
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from sva_common import *
from sentence_transformers import SentenceTransformer
dev="mps" if torch.backends.mps.is_available() else "cpu"
m=SentenceTransformer("BAAI/bge-base-zh-v1.5",device=dev)
a,s=load(); rng=np.random.default_rng(7)
BANDS=[(0.999,1.01,"完全相同"),(0.90,0.999,"≥0.90"),(0.80,0.90,"0.80–0.90"),(0.70,0.80,"0.70–0.80"),(-1,0.70,"<0.70")]
rows=[]
def enc(q): return m.encode(list(q),batch_size=256,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False).astype(np.float32)
def nn(E,R):
    best=np.full(len(E),-1.0,np.float32); arg=np.zeros(len(E),int)
    for i in range(0,len(E),512):
        sim=E[i:i+512]@R.T; j=sim.argmax(1); best[i:i+512]=sim[np.arange(len(j)),j]; arg[i:i+512]=j
    return best,arg
def bands(x): return " | ".join(f"{lab} {((x>=lo)&(x<hi)).mean()*100:5.1f}%" for lo,hi,lab in BANDS)
for d in CAT:
    c=cells(a,s,d); S=c["搜索top10k"].reset_index(drop=True); qs=S["query"].astype(str).tolist(); ES=enc(qs)
    print(f"\n==================== {d} (search top10k n={len(S):,}, device={dev}) ====================")
    # controls
    h=1000; b,_=nn(ES[:h],ES[h:]); print(f"  对照A 搜索top1000 -> 搜索1001+      median {np.median(b):.3f} | "+bands(b))
    t=len(ES)-1000; b,_=nn(ES[t:],ES[:t]); print(f"  对照B 搜索最深1000 -> 搜索其余     median {np.median(b):.3f} | "+bands(b))
    rp=(ES[rng.integers(0,len(ES),20000)]*ES[rng.integers(0,len(ES),20000)]).sum(1); print(f"  随机搜索对 cos 分位 p50 {np.percentile(rp,50):.3f} p90 {np.percentile(rp,90):.3f} p99 {np.percentile(rp,99):.3f}")
    for k in ("助手top1k","助手random1k"):
        A=c[k].reset_index(drop=True); aq=A["query"].astype(str).tolist(); EA=enc(aq); b,j=nn(EA,ES)
        print(f"  {k:<12} -> 搜索top10k     median {np.median(b):.3f} | "+bands(b)+f"   (n={len(A):,})")
        for q,sv,jj,pv in zip(aq,b,j,A["search_num"]): rows.append(dict(domain=d,surface=k,query=q,pv=int(pv),nn_query=qs[jj],nn_search_rank=int(jj)+1,sim=float(sv)))
        for lo,hi,lab in BANDS[1:]:
            idx=np.where((b>=lo)&(b<hi))[0]
            if len(idx): print(f"     [{lab}] "+" ‖ ".join(f"{aq[i][:26]} → {qs[j[i]][:18]}(#{j[i]+1},{b[i]:.2f})" for i in rng.choice(idx,min(5,len(idx)),replace=False)))
pd.DataFrame(rows).to_parquet(f"{SP}/sva_semantic_nn.parquet",index=False)
print("\nwrote sva_semantic_nn.parquet", len(rows))
