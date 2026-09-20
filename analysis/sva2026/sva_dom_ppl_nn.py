# -*- coding: utf-8 -*-
"""人物: semantic nearest neighbours recomputed on clean_v3 user rows (bge-base-zh-v1.5).
assistant top1k / random1k -> search top10k (user); controls: search top1000 -> search rest; search deepest 1000 -> rest.
Reverse: search top1000 -> assistant (top1k ∪ random1k)."""
import sys, numpy as np, torch
SP=__import__("os").environ.get("SVA_WORKDIR", __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "work"))
sys.path.insert(0, SP)
from sva_common import *
from sentence_transformers import SentenceTransformer
dev = "mps" if torch.backends.mps.is_available() else "cpu"
m = SentenceTransformer("BAAI/bge-base-zh-v1.5", device=dev)
a, s = load(); c = cells(a, s, "人物")
S = c["搜索top10k"].reset_index(drop=True); qs = S["query"].astype(str).tolist(); srank = S["rank"].tolist()
def enc(q): return m.encode(list(q), batch_size=256, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
def nn(E, R):
    best = np.full(len(E), -1.0, np.float32); arg = np.zeros(len(E), int)
    for i in range(0, len(E), 512):
        sim = E[i:i+512] @ R.T; j = sim.argmax(1); best[i:i+512] = sim[np.arange(len(j)), j]; arg[i:i+512] = j
    return best, arg
ES = enc(qs)
BANDS = [(0.999, 1.01, "完全相同"), (0.90, 0.999, "≥0.90"), (0.80, 0.90, "0.80–0.90"), (0.70, 0.80, "0.70–0.80"), (-1, 0.70, "<0.70")]
def bands(x): return " | ".join(f"{lab} {((x>=lo)&(x<hi)).mean()*100:5.1f}%" for lo, hi, lab in BANDS)
print(f"search top10k user n={len(S):,} device={dev}")
h = 1000; b, _ = nn(ES[:h], ES[h:]); print(f"对照A 搜索top1000→搜索1001+ n={h} median {np.median(b):.3f} <0.70 {(b<0.70).mean()*100:.1f}% | " + bands(b))
t = len(ES) - 1000; b2, _ = nn(ES[t:], ES[:t]); print(f"对照B 搜索最深1000→其余 n=1000 median {np.median(b2):.3f} <0.70 {(b2<0.70).mean()*100:.1f}% | " + bands(b2))
ctrl = pd.DataFrame({"which": ["A"]*h + ["B"]*1000, "query": qs[:h] + qs[t:], "sim": np.r_[b, b2]})
ctrl["len"] = ctrl["query"].str.len()
rows = []
EA_all = {}
for k in ("助手top1k", "助手random1k"):
    A = c[k].reset_index(drop=True); aq = A["query"].astype(str).tolist(); EA = enc(aq); EA_all[k] = (aq, EA)
    bb, jj = nn(EA, ES)
    print(f"{k} → 搜索top10k n={len(A):,} median {np.median(bb):.3f} <0.70 {(bb<0.70).mean()*100:.1f}% | " + bands(bb))
    for q, sv, j, pv in zip(aq, bb, jj, A["search_num"]):
        rows.append(dict(surface=k, query=q, pv=int(pv), sim=float(sv), nn_query=qs[j], nn_rank=int(srank[j]), nn_pv=int(S.wise_pv.iloc[j])))
R = pd.DataFrame(rows); R["len"] = R["query"].str.len()
R.to_parquet(f"{SP}/sva_dom_ppl_nn.parquet", index=False); ctrl.to_parquet(f"{SP}/sva_dom_ppl_nn_ctrl.parquet", index=False)
# length-banded
BK = [(0, 4, "≤4字"), (5, 6, "5–6"), (7, 10, "7–10"), (11, 15, "11–15"), (16, 10**6, ">15")]
def lb(sim, ln):
    return " | ".join((f"{lab}: n={((ln>=lo)&(ln<=hi)).sum()} med={np.median(sim[(ln>=lo)&(ln<=hi)]):.2f} <0.70={(sim[(ln>=lo)&(ln<=hi)]<0.70).mean()*100:.0f}%" if ((ln>=lo)&(ln<=hi)).sum()>=20 else f"{lab}: n={((ln>=lo)&(ln<=hi)).sum()} —") for lo, hi, lab in BK)
print("\n长度分层")
for w, x in ctrl.groupby("which"): print(f"  对照{w}: " + lb(x.sim.values, x.len.values))
for k, x in R.groupby("surface"): print(f"  {k}: " + lb(x.sim.values, x.len.values))
# reverse coverage: search top1000 -> assistant union
aq = EA_all["助手top1k"][0] + EA_all["助手random1k"][0]; EA = np.vstack([EA_all["助手top1k"][1], EA_all["助手random1k"][1]])
src = ["top1k"]*len(EA_all["助手top1k"][0]) + ["random1k"]*len(EA_all["助手random1k"][0])
bb, jj = nn(ES[:1000], EA)
REV = pd.DataFrame({"query": qs[:1000], "pv": S.wise_pv.iloc[:1000].values, "sim": bb, "nn_query": [aq[j] for j in jj], "nn_src": [src[j] for j in jj]})
REV.to_parquet(f"{SP}/sva_dom_ppl_nn_rev.parquet", index=False)
print(f"\n反向 搜索top1000→助手(头+尾) n=1000 median {np.median(bb):.3f} <0.70 {(bb<0.70).mean()*100:.1f}% | " + bands(bb))
