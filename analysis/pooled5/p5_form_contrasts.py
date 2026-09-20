# -*- coding: utf-8 -*-
"""What replicates across domains, measured WITHOUT a class-name mapping.

Comparing class shares across domains needs a judgement call: each domain grew its own taxonomy,
so "does 医疗's 可行性裁决 mean the same as 金融's 个股买卖建议" is an opinion. The surface-form
markers do not have that problem — they are one regex applied identically to every domain and
every source, so the same marker moving the same way in five domains IS a cross-domain regularity,
and a marker moving in two domains and not the others is a domain effect.

Each cell is the change in row share between two sources with a Newcombe 95% interval.
Regex markers have false positives and misses; only RELATIVE differences under one rule are read.

    python analysis/pooled5/p5_form_contrasts.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pooled5_common import COHORT, CONTRASTS, FORM, SRC_ZH, available, cohort, cross_dir, load, newcombe

if not available():
    raise SystemExit(f"cohort {COHORT!r}: no finished run yet — nothing to sweep. "
                     f"(expected labels_full.csv under runs/<id>/gen01 for {cohort()})")

rows = []
for domain in available():
    d = load(domain)
    q = d["query"].astype(str)
    hits = {name: q.str.contains(pat, regex=True) for name, pat in FORM.items()}
    for a_s, b_s, why in CONTRASTS:
        ma, mb = (d.source == a_s).to_numpy(), (d.source == b_s).to_numpy()
        na, nb = int(ma.sum()), int(mb.sum())
        if not na or not nb:
            continue
        for name, h in hits.items():
            hv = h.to_numpy()
            ka, kb = int(hv[ma].sum()), int(hv[mb].sum())
            diff, lo, hi = newcombe(ka, na, kb, nb)
            rows.append({"domain": domain, "a": a_s, "b": b_s, "why": why, "marker": name,
                         "share_a": round(ka / na, 4), "share_b": round(kb / nb, 4),
                         "diff_pp": round(100 * diff, 2), "lo_pp": round(100 * lo, 2),
                         "hi_pp": round(100 * hi, 2), "sig": bool(lo > 0 or hi < 0)})
t = pd.DataFrame(rows)
t.to_csv(cross_dir() / "form_contrasts.csv", index=False)
for a_s, b_s, why in CONTRASTS:
    sub = t[(t.a == a_s) & (t.b == b_s)]
    if not len(sub):
        continue
    piv = sub.pivot_table(index="marker", columns="domain", values="diff_pp")
    sig = sub.pivot_table(index="marker", columns="domain", values="sig", aggfunc="first")
    n_sig_same = ((piv > 0) & sig).sum(axis=1) - 0
    n_sig_neg = ((piv < 0) & sig).sum(axis=1)
    keep = piv[(n_sig_same >= 2) | (n_sig_neg >= 2)]
    print(f"\n===== {SRC_ZH[a_s]} → {SRC_ZH[b_s]}（{why}）  单位：百分点，只列至少两个领域显著的标记")
    if len(keep):
        out = keep.round(1).copy()
        out["显著域数"] = (n_sig_same + n_sig_neg).reindex(keep.index)
        print(out.to_string())
print("\nwrote form_contrasts.csv")
