# -*- coding: utf-8 -*-
"""Do the two independent routes agree on this source?

The run produces two labelings of every row: a top-down intent taxonomy written by an architect,
and a bottom-up cluster tree fitted on embeddings. `route_concordance.json` reports their
agreement over the whole corpus. Per SOURCE it is more useful: if the two routes agree less on
the assistant rows than on the search rows, that is independent evidence that the frame fits
those rows worse — evidence that does not depend on the classifier's own confidence score.

    python analysis/pooled5/p5_route_agreement.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score
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
        if len(g) < 50:
            continue
        rows.append({"domain": domain, "source": src, "n": len(g),
                     "ami_leaf_vs_intent": round(adjusted_mutual_info_score(
                         g["bu_leaf_name"].astype(str), g["td_l1_name"].astype(str)), 4),
                     "ami_family_vs_intent": round(adjusted_mutual_info_score(
                         g["bu_family_final"].astype(str), g["td_l1_name"].astype(str)), 4)})
t = pd.DataFrame(rows)
t.to_csv(cross_dir() / "route_agreement_by_source.csv", index=False)
p = t.pivot_table(index="domain", columns="source", values="ami_leaf_vs_intent")
p = p.reindex(columns=[c for c in SOURCES if c in p.columns]).rename(columns=SRC_ZH)
print("两条独立路线（聚类叶 vs 自上而下意图）在各来源上的一致度 AMI\n")
print(p.to_string())
