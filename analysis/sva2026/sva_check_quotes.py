# -*- coding: utf-8 -*-
"""Every “…” quote in a report draft must be a real row (exact, or a prefix when the quote ends with …). Prints misses."""
import sys, re
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from sva_common import *
a,s=load()
rows=pd.concat([a.assign(surf=a.snapshot.map({"top1k":"助手头部","random1k":"助手随机"}),pv=a.search_num)[["query","surf","tier","l1","pv"]],
                s.assign(surf="搜索",l1=s.domain,pv=s.wise_pv)[["query","surf","tier","l1","pv"]]])
rows["query"]=rows["query"].astype(str)
Q=set(rows["query"])
for path in sys.argv[1:]:
    txt=open(path,encoding="utf-8").read()
    quotes=[q for q in re.findall(r"“([^”]{2,80})”",txt)]
    miss=[]; ok=0
    for q in dict.fromkeys(quotes):
        qq=q.rstrip("…").rstrip(".")
        if q in Q or (q.endswith("…") and rows["query"].str.startswith(qq).any()): ok+=1
        else: miss.append(q)
    print(f"{path.split('/')[-1]}: quotes {len(set(quotes))}, found {ok}, NOT FOUND {len(miss)}")
    for m in miss: print("   ✗", m)
