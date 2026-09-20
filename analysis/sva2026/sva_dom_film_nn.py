# -*- coding: utf-8 -*-
"""影视: nearest search-top10k neighbour (bge-base-zh-v1.5 cosine) for every v3 USER assistant row; within-search controls;
length buckets and a length-standardised expectation from control B (deepest 1000 search rows -> rest)."""
import sys, numpy as np, pandas as pd, torch
SP=__import__("os").environ.get("SVA_WORKDIR", __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "work"))
sys.path.insert(0,SP); from sva_common import *
from sentence_transformers import SentenceTransformer
dev="mps" if torch.backends.mps.is_available() else "cpu"; m=SentenceTransformer("BAAI/bge-base-zh-v1.5",device=dev)
a,s=load(); c=cells(a,s,"影视"); S=c["搜索top10k"].reset_index(drop=True); qs=S["query"].astype(str).tolist(); spv=S.wise_pv.tolist()
F=pd.read_parquet(f"{SP}/sva_final_rows.parquet"); U=F[(F.domain=="影视")&(F.tier=="user")&F.surface.str.startswith("assistant")].reset_index(drop=True)
enc=lambda q: m.encode(list(q),batch_size=256,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False).astype(np.float32)
def nn(E,R):
    best=np.full(len(E),-1.0,np.float32); arg=np.zeros(len(E),int)
    for i in range(0,len(E),512):
        sm=E[i:i+512]@R.T; j=sm.argmax(1); best[i:i+512]=sm[np.arange(len(j)),j]; arg[i:i+512]=j
    return best,arg
ES=enc(qs); EA=enc(U["query"].astype(str))
BK=[(0,6,"≤6"),(7,10,"7–10"),(11,15,"11–15"),(16,25,"16–25"),(26,10**6,">25")]
def bands(x): return f"同≥0.999 {(x>=0.999).mean()*100:5.1f}% | 近0.80–0.999 {((x>=0.8)&(x<0.999)).mean()*100:5.1f}% | 中0.70–0.80 {((x>=0.7)&(x<0.8)).mean()*100:5.1f}% | 远<0.70 {(x<0.7).mean()*100:5.1f}%"
def lb(sim,ln):
    out=[]
    for lo,hi,lab in BK:
        k=(ln>=lo)&(ln<=hi); out.append(f"{lab}: n={k.sum()} 远{(sim[k]<0.7).mean()*100:.0f}%" if k.sum()>=20 else f"{lab}: n={k.sum()} —")
    return " | ".join(out)
slen=np.array([len(q) for q in qs])
bA,_=nn(ES[:1000],ES[1000:]); t=len(ES)-1000; bB,_=nn(ES[t:],ES[:t])
print(f"search top10k user n={len(qs)}  device={dev}")
print("对照A 搜索top1000→1001+ n=1000 median %.3f | %s"%(np.median(bA),bands(bA))); print("   "+lb(bA,slen[:1000]))
print("对照B 搜索最深1000→其余 n=1000 median %.3f | %s"%(np.median(bB),bands(bB))); print("   "+lb(bB,slen[t:]))
b,j=nn(EA,ES); U["sim"]=b; U["nn_query"]=[qs[x] for x in j]; U["nn_rank"]=j+1; U["nn_pv"]=[spv[x] for x in j]; U["len"]=U["query"].astype(str).str.len()
U.to_parquet(f"{SP}/sva_dom_film_nn.parquet",index=False)
# control B per-bucket far rate for length standardisation
farB={lab:(bB[(slen[t:]>=lo)&(slen[t:]<=hi)]<0.7).mean() for lo,hi,lab in BK if ((slen[t:]>=lo)&(slen[t:]<=hi)).sum()>=20}
farA={lab:(bA[(slen[:1000]>=lo)&(slen[:1000]<=hi)]<0.7).mean() for lo,hi,lab in BK if ((slen[:1000]>=lo)&(slen[:1000]<=hi)).sum()>=20}
for sf in ("assistant_top1k","assistant_random1k"):
    x=U[U.surface==sf]; sim=x.sim.values; ln=x.len.values
    print(f"\n{sf} n={len(x)} median {np.median(sim):.3f} | {bands(sim)}")
    if sf=="assistant_top1k":
        w=x.pv.values; print("   PV加权: 同%.1f%% 近%.1f%% 中%.1f%% 远%.1f%%"%tuple(100*np.array([w[sim>=0.999].sum(),w[(sim>=0.8)&(sim<0.999)].sum(),w[(sim>=0.7)&(sim<0.8)].sum(),w[sim<0.7].sum()])/w.sum()))
    print("   "+lb(sim,ln))
    # expected far share if each assistant row had control-B's far rate at its length (buckets with no control use the >25 → 16–25 rate)
    def rate(Lr,tab):
        for lo,hi,lab in BK:
            if lo<=Lr<=hi: return tab.get(lab, tab["16–25"])
    expB=np.mean([rate(L_,farB) for L_ in ln]); expA=np.mean([rate(L_,farA) for L_ in ln])
    print(f"   观测 远<0.70 {(sim<0.7).mean()*100:.1f}% ; 按长度标准化的期望: 若像搜索最深1000 {expB*100:.1f}% , 若像搜索头部 {expA*100:.1f}%")
old=pd.read_parquet(f"{SP}/sva_semantic_nn.parquet"); old=old[old.domain=="影视"]; old["surface"]=old.surface.map({"助手top1k":"assistant_top1k","助手random1k":"assistant_random1k"})
mm=U.merge(old[["surface","query","sim"]].drop_duplicates(["surface","query"]),on=["surface","query"],how="left",suffixes=("","_old"))
print(f"\n与旧 sva_semantic_nn 对比: 匹配 {mm.sim_old.notna().sum()}/{len(mm)}; |Δsim| 中位 {np.nanmedian(np.abs(mm.sim-mm.sim_old)):.4f}, 最大 {np.nanmax(np.abs(mm.sim-mm.sim_old)):.4f}")
