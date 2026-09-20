# -*- coding: utf-8 -*-
"""How concentrated is each source — does a handful of strings carry it?

Read WITHIN a source only. Search PV and assistant PV are different instruments, and the voice
export has no PV at all (uniform weight, so its "traffic" concentration is just its duplicate
rate). What is comparable is the SHAPE inside each source: what share of a source's own traffic
its ten biggest strings hold, and how much of the source is unique strings.

    python analysis/pooled5/p5_concentration.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pooled5_common import COHORT, SOURCES, SRC_ZH, available, cohort, cross_dir, load

if not available():
    raise SystemExit(f"cohort {COHORT!r}: no finished run yet — nothing to sweep. "
                     f"(expected labels_full.csv under runs/<id>/gen01 for {cohort()})")

rows = []
for domain in available():
    d = load(domain)
    for src in SOURCES:
        g = d[d.source == src]
        if not len(g):
            continue
        w = g["pv_norm"].to_numpy(dtype=float)
        w = np.sort(w)[::-1]
        tot = w.sum()
        n = len(g)
        srt = np.sort(w)
        gini = float((2 * np.arange(1, n + 1) - n - 1).dot(srt) / (n * srt.sum())) if srt.sum() else float("nan")
        rows.append({"domain": domain, "source": src, "n": n,
                     "distinct": int(g["query"].nunique()),
                     "dup_rate_%": round(100 * (1 - g["query"].nunique() / n), 2),
                     "top10_pv_%": round(100 * w[:10].sum() / tot, 2) if tot else float("nan"),
                     "top1%_rows_pv_%": round(100 * w[:max(1, n // 100)].sum() / tot, 2) if tot else float("nan"),
                     "gini_pv": round(gini, 3)})
t = pd.DataFrame(rows)
t["o"] = t["source"].map({s: i for i, s in enumerate(SOURCES)})
t = t.sort_values(["domain", "o"]).drop(columns="o")
t.to_csv(cross_dir() / "concentration_by_source.csv", index=False)
x = t.copy()
x["source"] = x["source"].map(SRC_ZH)
print(x.to_string(index=False))
