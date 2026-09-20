# -*- coding: utf-8 -*-
"""Is the interface gap just an export-depth artefact?

The obvious objection to `2026search → assistant_top`: search contributes its top 10,000 rows
and the assistant only its top 1,000, so search reaches far deeper into its own tail. This
control cuts search to its own top 1,000 by PV — the same nominal depth as the assistant head —
and recomputes the contrast. If the gap shrinks, it was depth; if it holds or grows, it is not.

    python analysis/pooled5/p5_depth_control.py [domain ...]
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pooled5_common import available, cross_dir, load, newcombe

LBL = "td_l1_name"


def tvd(a: pd.Series, b: pd.Series) -> float:
    pa, pb = a.value_counts(normalize=True), b.value_counts(normalize=True)
    return 0.5 * sum(abs(float(pa.get(k, 0)) - float(pb.get(k, 0))) for k in set(pa.index) | set(pb.index))


rows, detail = [], []
for domain in (sys.argv[1:] or available()):
    d = load(domain)
    s = d[d.source == "2026search"].sort_values("pv_raw", ascending=False)
    head1k = s.head(1000)
    asst = d[d.source == "assistant_top"]
    full_tvd, cut_tvd = tvd(s[LBL], asst[LBL]), tvd(head1k[LBL], asst[LBL])
    rows.append({"domain": domain, "n_search_full": len(s), "n_search_top1000": len(head1k),
                 "n_assistant_top": len(asst), "tvd_vs_full10k": round(full_tvd, 4),
                 "tvd_vs_top1000": round(cut_tvd, 4), "change": round(cut_tvd - full_tvd, 4),
                 "pv_floor_search_top1000": float(head1k.pv_raw.min()),
                 "pv_floor_search_10k": float(s.pv_raw.min()),
                 "pv_floor_assistant_top": float(asst.pv_raw.min())})
    for lab in sorted(set(asst[LBL]) | set(head1k[LBL])):
        ka, na = int((head1k[LBL] == lab).sum()), len(head1k)
        kb, nb = int((asst[LBL] == lab).sum()), len(asst)
        diff, lo, hi = newcombe(ka, na, kb, nb)
        kf, nf = int((s[LBL] == lab).sum()), len(s)
        detail.append({"domain": domain, "label": lab,
                       "share_search10k": round(kf / nf, 4), "share_search_top1000": round(ka / na, 4),
                       "share_assistant_top": round(kb / nb, 4),
                       "diff_vs_top1000_pp": round(100 * diff, 2),
                       "lo_pp": round(100 * lo, 2), "hi_pp": round(100 * hi, 2),
                       "sig": bool(lo > 0 or hi < 0)})
t = pd.DataFrame(rows)
cross_dir().mkdir(parents=True, exist_ok=True)
t.to_csv(cross_dir() / "depth_control.csv", index=False)
pd.DataFrame(detail).to_csv(cross_dir() / "depth_control_classes.csv", index=False)
print(t.to_string(index=False))
