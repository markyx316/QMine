# -*- coding: utf-8 -*-
"""A bootstrap interval for every TVD, because the contrasts differ in sample size by 10x.

The film verifier's objection is correct: `2025search → 2026search` compares two 10,000-row
sources while `assistant_top → assistant_random` compares two ~1,000-row sources, so a bare
"depth is almost twice interface" is not readable without knowing how noisy each number is.
TVD is also biased UPWARD at small n (two samples from one population still differ), so the
null band is reported too: the TVD between two halves of the SAME source.

    python analysis/pooled5/p5_tvd_ci.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pooled5_common import CONTRASTS, SRC_ZH, available, cross_dir, load

LBL, B = "td_l1_name", 400
rng = np.random.default_rng(11)


def _tvd(a: np.ndarray, b: np.ndarray, keys: list[str]) -> float:
    pa = np.array([(a == k).mean() for k in keys])
    pb = np.array([(b == k).mean() for k in keys])
    return float(0.5 * np.abs(pa - pb).sum())


rows = []
for domain in available():
    d = load(domain)
    for a_s, b_s, why in CONTRASTS:
        ga = d[d.source == a_s][LBL].to_numpy()
        gb = d[d.source == b_s][LBL].to_numpy()
        if not len(ga) or not len(gb):
            continue
        keys = sorted(set(ga) | set(gb))
        point = _tvd(ga, gb, keys)
        boot = np.array([_tvd(rng.choice(ga, len(ga)), rng.choice(gb, len(gb)), keys) for _ in range(B)])
        # null band: same source split in two, so any TVD below this is what noise alone gives
        null = []
        for g in (ga, gb):
            for _ in range(B // 2):
                idx = rng.permutation(len(g))
                h = len(g) // 2
                null.append(_tvd(g[idx[:h]], g[idx[h:2 * h]], keys))
        rows.append({"domain": domain, "a": a_s, "b": b_s, "why": why,
                     "n_a": len(ga), "n_b": len(gb), "tvd": round(point, 4),
                     "lo": round(float(np.percentile(boot, 2.5)), 4),
                     "hi": round(float(np.percentile(boot, 97.5)), 4),
                     "null_p95": round(float(np.percentile(null, 95)), 4)})
        print(f"{domain} {SRC_ZH[a_s]}→{SRC_ZH[b_s]}: TVD {point:.3f} [{rows[-1]['lo']:.3f}, {rows[-1]['hi']:.3f}] "
              f"· 同源噪声上界 {rows[-1]['null_p95']:.3f}", flush=True)
pd.DataFrame(rows).to_csv(cross_dir() / "tvd_ci.csv", index=False)
print("\nwrote tvd_ci.csv")
