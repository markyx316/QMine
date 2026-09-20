# -*- coding: utf-8 -*-
"""Containment: a search query written verbatim INSIDE a longer assistant query.

Exact overlap says the same string was typed on both surfaces. Containment says the search
string is the CORE and the assistant added something to it — the thing to look at is what got
added. Core rule (from the earlier study, kept identical so the two are comparable): a 2026
search string of >=3 characters, not all digits, contained in an assistant query at least 2
characters longer; longest core wins.

    python analysis/pooled5/p5_wrap.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pooled5_common import WORK, available, cohort, load

DOMS = sys.argv[1:] or available()
rows = []
for domain in DOMS:
    d = load(domain)
    cores = sorted({q for q in d[d.source == "2026search"]["query"].astype(str)
                    if len(q) >= 3 and not q.isdigit()}, key=len, reverse=True)
    by_first = {}
    for c in cores:
        by_first.setdefault(c[0], []).append(c)
    for src in ("assistant_top", "assistant_random", "assistant_voice"):
        g = d[d.source == src]
        for _, r in g.iterrows():
            q = str(r["query"])
            hit = ""
            for ch in set(q):
                for c in by_first.get(ch, ()):
                    if len(c) + 2 <= len(q) and c in q and len(c) > len(hit):
                        hit = c
            if hit:
                rows.append({"domain": domain, "source": src, "core": hit, "query": q,
                             "added_chars": len(q) - len(hit),
                             "core_td": d[(d.source == "2026search") & (d["query"] == hit)]["td_l1_name"].iloc[0],
                             "wrap_td": r["td_l1_name"], "same_intent": bool(
                                 d[(d.source == "2026search") & (d["query"] == hit)]["td_l1_name"].iloc[0] == r["td_l1_name"])})
    print(f"{domain}: {sum(1 for x in rows if x['domain'] == domain):,} wrap pairs", flush=True)
w = pd.DataFrame(rows)
suffix = "" if set(DOMS) == set(cohort()) else "_" + "_".join(DOMS)
w.to_csv(WORK / f"wrap_pairs{suffix}.csv", index=False)
if len(w):
    s = (w.groupby(["domain", "source"])
          .agg(pairs=("query", "size"), same_intent=("same_intent", "mean"),
               median_added=("added_chars", "median")).reset_index())
    s.to_csv(WORK / f"wrap_summary{suffix}.csv", index=False)
    print(s.to_string(index=False))
