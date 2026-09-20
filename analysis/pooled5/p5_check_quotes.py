# -*- coding: utf-8 -*-
"""Every quoted query string in a document must be a real row, with the source it claims.

SCOPE: the MAIN report only. It holds a convention this check depends on — 「」 wraps a query
string and nothing else. The five domain deep-dives were written by different analysts and use
backticks for identifiers, file names, class names AND queries interchangeably, so there is no
rule that separates a citation from a label in them; running this over the companion reports
~230 "misses" that are almost all concept phrases. The companion's query citations were checked
another way: each domain's independent verifier looked every quoted string up by (query, source)
in that domain's labelled parquet, and reported the count.

Quotes are matched against the union of all five labelled corpora. A quote ending in … is
matched as a prefix. Read the miss list; do not assume it is empty.

    python analysis/pooled5/p5_check_quotes.py docs/POOLED5_2026_五域同体系对比.zh.md
"""
from __future__ import annotations
import re, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pooled5_common import available, load

rows = []
for dom in available():
    d = load(dom)
    rows.append(d[["query", "source", "domain"]])
allq = pd.concat(rows, ignore_index=True)
allq["query"] = allq["query"].astype(str)
Q = set(allq["query"])
print(f"reference corpus: {len(allq):,} rows, {len(Q):,} distinct strings, domains {sorted(set(allq.domain))}")
for path in sys.argv[1:]:
    txt = Path(path).read_text(encoding="utf-8")
    quotes = list(dict.fromkeys(re.findall(r"[「“]([^」”\n]{2,80})[」”]", txt)))
    hit, miss = 0, []
    for q in quotes:
        core = q.rstrip("…").rstrip(".")
        if q in Q or (q != core and allq["query"].str.startswith(core).any()):
            hit += 1
        else:
            miss.append(q)
    print(f"\n{Path(path).name}: {len(quotes)} quoted strings, {hit} are real rows, {len(miss)} are not")
    for m in miss:
        n = int(allq["query"].str.contains(re.escape(m.replace('…', '')), regex=True).sum()) if len(m) > 1 else -1
        print(f"   [{n:>4} rows contain it] {m}")
