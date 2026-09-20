# -*- coding: utf-8 -*-
"""教育: fresh v3 semantic NN (bge-base-zh-v1.5) + controls, exact overlap, wraps, near-paraphrase transitions."""
import os; os.environ["HF_HUB_OFFLINE"]="1"
from sva_dom_edu_lib import *
import torch
from sentence_transformers import SentenceTransformer
a,s=load(); e=edu(); U=users(e)
C=cells(a,s,"教育"); S10=C["搜索top10k"].reset_index(drop=True); S10["query"]=S10["query"].astype(str)
sq=S10["query"].tolist(); spv=S10.wise_pv.values
dev="mps" if torch.backends.mps.is_available() else "cpu"
m=SentenceTransformer("BAAI/bge-base-zh-v1.5",device=dev)
enc=lambda q: m.encode(list(q),batch_size=256,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False).astype(np.float32)
def nn(E,R):
    best=np.full(len(E),-1.0,np.float32); arg=np.zeros(len(E),int)
    for i in range(0,len(E),512):
        sim=E[i:i+512]@R.T; j=sim.argmax(1); best[i:i+512]=sim[np.arange(len(j)),j]; arg[i:i+512]=j
    return best,arg
ES=enc(sq)
BANDS=[(0.999,1.01,"完全相同"),(0.90,0.999,"≥0.90"),(0.80,0.90,"0.80–0.90"),(0.70,0.80,"0.70–0.80"),(-1,0.70,"<0.70")]
def bands(x): n=len(x); return " | ".join(f"{lab} {int(((x>=lo)&(x<hi)).sum())/n*100:.1f}% [{wilson(int(((x>=lo)&(x<hi)).sum()),n)[0]:.1f}–{wilson(int(((x>=lo)&(x<hi)).sum()),n)[1]:.1f}]" for lo,hi,lab in BANDS)
b,_=nn(ES[:1000],ES[1000:]); print(f"search top10k user n={len(sq)} device={dev}\n对照A 搜索top1000→搜索1001+ n=1000 median {np.median(b):.3f} | "+bands(b))
t=len(ES)-1000; b,_=nn(ES[t:],ES[:t]); print(f"对照B 搜索最深1000→其余 n=1000 median {np.median(b):.3f} | "+bands(b))
# labels for search neighbours
lab={}
ex=pd.read_parquet(f"{SP}/label_extra/labels.parquet"); print("label_extra cols",list(ex.columns)[:10], len(ex))
if "u_ds" in ex.columns: lab.update(dict(zip(ex["query"].astype(str),ex["u_ds"])))
lab.update(dict(zip(U["search_top1000"]["query"],U["search_top1000"]["u_ds"])))
out=[]
for k in ["assistant_top1k","assistant_random1k"]:
    g=U[k].reset_index(drop=True); EA=enc(g["query"]); b,j=nn(EA,ES)
    g["sim"]=b; g["nn_query"]=[sq[i] for i in j]; g["nn_rank"]=j+1; g["nn_pv"]=spv[j]; g["u_nn"]=g["nn_query"].map(lab)
    out.append(g)
    print(f"\n{SN[k]} n={len(g)} median {np.median(b):.3f} | "+bands(b))
    print(f"  neighbour labelled: {g.u_nn.notna().mean()*100:.1f}%")
    print("  <0.70 share by u_ds: "+" ; ".join(f"{c} n={int((g.u_ds==c).sum())} {((g.u_ds==c)&(g.sim<0.70)).sum()/max(1,(g.u_ds==c).sum())*100:.0f}%" for c in sorted(g.u_ds.unique())))
R=pd.concat(out); R.drop(columns=["u_qw","p_qw"]).to_csv(f"{SP}/sva_dom_edu_nn_rows.csv",index=False)
H=R[R.surface=="assistant_top1k"]; T=R[R.surface=="assistant_random1k"]
# exact overlap
s10=set(sq); s1k=set(U["search_top1000"]["query"])
for lab_,g in [("助手头",H),("助手尾",T)]:
    x=g["query"].isin(s10); y=g["query"].isin(s1k)
    print(f"\nEXACT {lab_}: in search top10k {fmt(int(x.sum()),len(g))} PV {g.loc[x,'pv'].sum()/g.pv.sum()*100:.1f}% ; in search top1000 {fmt(int(y.sum()),len(g))}")
    xi=x&(g.u_ds!="U13"); print(f"   excluding U13 rows: in top10k {fmt(int(xi.sum()),int((g.u_ds!='U13').sum()))}")
    print("   exact matches by u_ds: "+str(g[x].u_ds.value_counts().to_dict()))
    xx=g[x].sort_values("pv",ascending=False); print("   例 "+" | ".join(f"{q}(助手{p:,.0f}; 搜索#{r})" for q,p,r in zip(xx['query'].head(20),xx.pv.head(20),xx.nn_rank.head(20))))
sa=set(e[(e.surface=="assistant_top1k")&(e.tier=="user")]["query"]); sall=set(e[e.surface=="assistant_top1k"]["query"])
rv=U["search_top1000"]["query"].isin(sa); print(f"REVERSE: search top1000 user queries found verbatim among assistant head user rows: {fmt(int(rv.sum()),1000)}; among any-tier head rows {int(U['search_top1000']['query'].isin(sall).sum())}")
print("   "+" | ".join(U["search_top1000"][rv].sort_values("pv",ascending=False)["query"].head(25)))
# wraps: search top1000 user cores (len>=2) inside longer assistant user queries
cores=U["search_top1000"][["query","u_ds","pv"]].copy(); cores=cores[cores["query"].str.len()>=2].sort_values("query",key=lambda x:x.str.len(),ascending=False)
cl=list(zip(cores["query"],cores["u_ds"]))
def wrap(q):
    for c,u in cl:
        if len(q)>len(c) and c in q: return c,u
    return None,None
for lab_,g in [("助手头",H),("助手尾",T)]:
    w=g["query"].map(wrap); g=g.assign(core=[x[0] for x in w],u_core=[x[1] for x in w]); ww=g[g.core.notna()]
    print(f"\nWRAP (search top1000 core, len>=2) {lab_}: {fmt(len(ww),len(g))}; intent unchanged {fmt(int((ww.u_core==ww.u_ds).sum()),max(1,len(ww)))}")
    tr=(ww.u_core+"→"+ww.u_ds).value_counts().head(10); print("   transitions: "+" ; ".join(f"{i} {v}" for i,v in tr.items()))
    print("   例 "+" | ".join(f"[{c}]→{q[:30]}({u})" for c,q,u in zip(ww['core'].head(40),ww['query'].head(40),ww['u_ds'].head(40))))
    ww.to_csv(f"{SP}/sva_dom_edu_wraps_{'head' if lab_=='助手头' else 'tail'}.csv",index=False)
# wraps with top10k cores len>=4
c10=sorted([q for q in sq if len(q)>=4],key=len,reverse=True)
def wrap10(q):
    for c in c10:
        if len(q)>len(c) and c in q: return c
    return None
for lab_,g in [("助手头",H),("助手尾",T)]:
    w=g["query"].map(wrap10); k=int(w.notna().sum()); print(f"WRAP (search top10k core len>=4) {lab_}: {fmt(k,len(g))}  例 "+" | ".join(f"[{c}]→{q[:30]}" for c,q in zip(w[w.notna()].head(25),g["query"][w.notna()].head(25))))
# near-paraphrase transitions
for lab_,g in [("助手头",H),("助手尾",T)]:
    npb=g[(g.sim>=0.80)&(g.sim<0.999)&g.u_nn.notna()]
    print(f"\nNEAR-PARAPHRASE 0.80≤sim<0.999 {lab_}: n={len(npb)} (of {int(((g.sim>=0.80)&(g.sim<0.999)).sum())}); intent unchanged {fmt(int((npb.u_nn==npb.u_ds).sum()),max(1,len(npb)))}")
    ch=npb[npb.u_nn!=npb.u_ds]; tr=(ch.u_nn+"→"+ch.u_ds).value_counts().head(8)
    print("   changed transitions: "+" ; ".join(f"{i} {v}" for i,v in tr.items()))
    print("   例(changed) "+" | ".join(f"[{nq}]→{q[:28]}({un}→{u},{sm:.2f})" for nq,q,un,u,sm in zip(ch['nn_query'].head(30),ch['query'].head(30),ch['u_nn'].head(30),ch['u_ds'].head(30),ch['sim'].head(30))))
    same=npb[npb.u_nn==npb.u_ds]; print("   例(same) "+" | ".join(f"[{nq}]→{q[:28]}({u},{sm:.2f})" for nq,q,u,sm in zip(same['nn_query'].head(20),same['query'].head(20),same['u_ds'].head(20),same['sim'].head(20))))
# low sim
for lab_,g in [("助手头",H),("助手尾",T)]:
    lo=g[g.sim<0.70]; gi=g[g.u_ds!="U13"]; loi=gi[gi.sim<0.70]
    print(f"\nLOW-SIM <0.70 {lab_}: {fmt(len(lo),len(g))} ; among interpretable (non-U13) {fmt(len(loi),len(gi))}; PV {lo.pv.sum()/g.pv.sum()*100:.1f}%")
    print("   u_ds of low-sim interpretable: "+str((loi.u_ds.value_counts(normalize=True)*100).round(1).to_dict()))
    exm=loi.sort_values("pv",ascending=False).head(25) if lab_=="助手头" else loi.sample(min(40,len(loi)),random_state=9)
    print("   例 "+" | ".join(f"{q[:32]}→[{nq[:14]}]({sm:.2f},{u})" for q,nq,sm,u in zip(exm['query'],exm['nn_query'],exm['sim'],exm['u_ds'])))
# candidate pairs by band for §4 (interpretable, 0.60-0.95)
cand=R[(R.u_ds!="U13")&(R.sim>=0.62)&(R.sim<0.97)]
print("\nCANDIDATE PAIRS (interpretable, 0.62≤sim<0.97), head by PV then tail random:")
hh=cand[cand.surface=="assistant_top1k"].sort_values("pv",ascending=False).head(60)
print(" ; ".join(f"[{nq}#{r}]→{q[:36]}({u},{sm:.2f})" for nq,r,q,u,sm in zip(hh['nn_query'],hh['nn_rank'],hh['query'],hh['u_ds'],hh['sim'])))
tt=cand[cand.surface=="assistant_random1k"].sample(min(80,int((cand.surface=='assistant_random1k').sum())),random_state=5)
print(" ; ".join(f"[{nq}#{r}]→{q[:40]}({u},{sm:.2f})" for nq,r,q,u,sm in zip(tt['nn_query'],tt['nn_rank'],tt['query'],tt['u_ds'],tt['sim'])))
