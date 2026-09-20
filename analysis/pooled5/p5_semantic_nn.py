# -*- coding: utf-8 -*-
"""Is an assistant query even EXPRESSIBLE as a search query? For every row of every source, its
nearest neighbour among the 2026 search rows of the same domain (bge-base-zh-v1.5 cosine).

THE CONTROL IS THE POINT. 2025 search → 2026 search is the same instrument one year apart, so
it measures how much novelty TIME alone produces. Anything the assistant sources show beyond
that is what the interface adds. A second control splits 2026 search itself (top 1,000 → the
rest) to show what the metric reads inside one export.

Bands follow the earlier study so the two are comparable: 完全相同 ≥0.999, ≥0.90, 0.80–0.90,
0.70–0.80, <0.70. The thresholds are conventions, not a definition of "same need" — the random
-pair percentiles printed per domain are what calibrates them.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pooled5_common import WORK, cohort, work_file
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[2]
BANDS = [(0.999, 1.01, "完全相同"), (0.90, 0.999, "≥0.90"), (0.80, 0.90, "0.80–0.90"),
         (0.70, 0.80, "0.70–0.80"), (-1, 0.70, "<0.70")]
dev = "mps" if torch.backends.mps.is_available() else "cpu"
m = SentenceTransformer("BAAI/bge-base-zh-v1.5", device=dev)
rng = np.random.default_rng(7)


def enc(q):
    return m.encode(list(q), batch_size=256, normalize_embeddings=True,
                    convert_to_numpy=True, show_progress_bar=False).astype(np.float32)


def nn(E, R):
    best = np.full(len(E), -1.0, np.float32); arg = np.zeros(len(E), int)
    for i in range(0, len(E), 512):
        sim = E[i:i + 512] @ R.T
        j = sim.argmax(1)
        best[i:i + 512] = sim[np.arange(len(j)), j]
        arg[i:i + 512] = j
    return best, arg


def bands(x):
    return " | ".join(f"{lab} {((x >= lo) & (x < hi)).mean() * 100:5.1f}%" for lo, hi, lab in BANDS)


rows, ctrl = [], []
for domain in cohort():
    d = pd.read_parquet(ROOT / f"data/raw/pooled5/{domain}_pooled5.parquet")
    R = d[d.source == "2026search"].reset_index(drop=True)
    rq = R["query"].astype(str).tolist()
    ER = enc(rq)
    print(f"\n===== {domain}  (2026搜索参照池 n={len(R):,}, device={dev}) =====", flush=True)
    rp = (ER[rng.integers(0, len(ER), 20000)] * ER[rng.integers(0, len(ER), 20000)]).sum(1)
    print(f"  随机同域搜索对 cos: p50 {np.percentile(rp, 50):.3f}  p90 {np.percentile(rp, 90):.3f}  p99 {np.percentile(rp, 99):.3f}")
    b, _ = nn(ER[:1000], ER[1000:])
    ctrl += [dict(domain=domain, kind="对照·2026搜索前1000→其余", query=q, sim=float(v)) for q, v in zip(rq[:1000], b)]
    print(f"  对照 2026搜索前1000 → 其余      median {np.median(b):.3f} | " + bands(b))
    for src in ("2025search", "assistant_top", "assistant_random", "assistant_voice"):
        A = d[d.source == src].reset_index(drop=True)
        if not len(A):
            continue
        aq = A["query"].astype(str).tolist()
        b, j = nn(enc(aq), ER)
        tag = "对照·时间" if src == "2025search" else "探针"
        print(f"  {src:<17} → 2026搜索      median {np.median(b):.3f} | " + bands(b) + f"   (n={len(A):,}, {tag})")
        for q, sv, jj, pv in zip(aq, b, j, A["pv_raw"]):
            rows.append(dict(domain=domain, source=src, query=q, pv_raw=float(pv) if pd.notna(pv) else float("nan"),
                             nn_query=rq[jj], nn_rank=int(jj) + 1, sim=float(sv)))
        for lo, hi, lab in BANDS[1:]:
            idx = np.where((b >= lo) & (b < hi))[0]
            if len(idx):
                pick = rng.choice(idx, min(4, len(idx)), replace=False)
                print(f"     [{lab}] " + " ‖ ".join(f"{aq[i][:24]} → {rq[j[i]][:18]}({b[i]:.2f})" for i in pick))

WORK.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_parquet(work_file("semantic_nn.parquet"), index=False)
pd.DataFrame(ctrl).to_parquet(work_file("semantic_nn_controls.parquet"), index=False)
print("\nwrote semantic_nn.parquet", len(rows))
