# -*- coding: utf-8 -*-
"""医疗 only: nearest search-top10k neighbours (bge-base-zh-v1.5 cosine, top-3) for v3 assistant user rows,
plus within-search controls (search top1000 -> rest; search deepest 1000 -> rest), by length band."""
import sys, numpy as np, pandas as pd, torch
SP = __import__("os").environ.get("SVA_WORKDIR", __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "work"))
sys.path.insert(0, SP)
import sva_common as C
from sentence_transformers import SentenceTransformer
dev = "mps" if torch.backends.mps.is_available() else "cpu"
m = SentenceTransformer("BAAI/bge-base-zh-v1.5", device=dev)
a, s = C.load(); c = C.cells(a, s, "医疗")
S = c["搜索top10k"].reset_index(drop=True); sq = S["query"].astype(str).tolist(); spv = S.wise_pv.values
def enc(q): return m.encode(list(q), batch_size=256, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
def topk(E, R, k=3):
    I = np.zeros((len(E), k), int); V = np.zeros((len(E), k), np.float32)
    for i in range(0, len(E), 512):
        sim = E[i:i + 512] @ R.T; idx = np.argpartition(-sim, k, axis=1)[:, :k]; vals = np.take_along_axis(sim, idx, 1); o = np.argsort(-vals, 1)
        I[i:i + 512] = np.take_along_axis(idx, o, 1); V[i:i + 512] = np.take_along_axis(vals, o, 1)
    return I, V
ES = enc(sq)
out = open(f"{SP}/sva_dom_med_nn.txt", "w", encoding="utf-8")
def P(*x): print(*x, file=out, flush=True)
BANDS = [(0.999, 1.01, "完全相同"), (0.90, 0.999, "≥0.90"), (0.80, 0.90, "0.80–0.90"), (0.70, 0.80, "0.70–0.80"), (-1, 0.70, "<0.70")]
LB = [(0, 6, "≤6字"), (7, 10, "7–10"), (11, 15, "11–15"), (16, 25, "16–25"), (26, 99999, ">25")]
def summ(name, v, lens):
    v = np.asarray(v); lens = np.asarray(lens)
    P(f"  {name:<22} n={len(v):,} median {np.median(v):.3f} | " + " | ".join(f"{lab} {((v>=lo)&(v<hi)).mean()*100:5.1f}%" for lo, hi, lab in BANDS))
    P("      by length: " + " | ".join(f"{lab}: n={int(((lens>=lo)&(lens<=hi)).sum())} med={np.median(v[(lens>=lo)&(lens<=hi)]) if ((lens>=lo)&(lens<=hi)).sum()>=20 else float('nan'):.2f} <0.70={(v[(lens>=lo)&(lens<=hi)]<0.70).mean()*100 if ((lens>=lo)&(lens<=hi)).sum()>=20 else float('nan'):.0f}%" for lo, hi, lab in LB))
lens = np.array([len(q) for q in sq])
_, VA = topk(ES[:1000], ES[1000:], 1); summ("对照A 搜索1k→搜索其余", VA[:, 0], lens[:1000])
t = len(ES) - 1000; _, VB = topk(ES[t:], ES[:t], 1); summ("对照B 搜索最深1k→其余", VB[:, 0], lens[t:])
rows = []
for k, lab in (("助手top1k", "assistant_top1k"), ("助手random1k", "assistant_random1k")):
    A = c[k].reset_index(drop=True); aq = A["query"].astype(str).tolist(); EA = enc(aq); I, V = topk(EA, ES, 3)
    summ(lab + "→搜索10k", V[:, 0], [len(q) for q in aq])
    for i, q in enumerate(aq):
        r = dict(surface=lab, query=q, pv=float(A.search_num.iloc[i]), l2=A.l2.iloc[i])
        for j in range(3):
            r[f"nn{j+1}"] = sq[I[i, j]]; r[f"nn{j+1}_rank"] = int(I[i, j]) + 1; r[f"nn{j+1}_pv"] = float(spv[I[i, j]]); r[f"sim{j+1}"] = float(V[i, j])
        rows.append(r)
pd.DataFrame(rows).to_parquet(f"{SP}/sva_dom_med_nn.parquet", index=False)
P(f"wrote sva_dom_med_nn.parquet rows={len(rows)} device={dev}")
out.close()
