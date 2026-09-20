# -*- coding: utf-8 -*-
import sys, re, collections; sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from sva_common import *
a,s=load()
FIRST=re.compile(r"(我|我的|我家|我们|本人)"); Q=re.compile(r"(怎么|如何|为什么|什么|哪个|哪些|多少|吗|呢|是不是|能不能|\?|？)")
tot=collections.Counter()
for d,cat in CAT.items():
    c=cells(a,s,d); su=c["搜索top10k"]; rank={q:i+1 for i,q in enumerate(su["query"].astype(str))}
    S=set(rank); top1000=su.head(1000)["query"].astype(str).tolist()
    print(f"\n==================== {d} ====================")
    for k in ("助手top1k","助手random1k"):
        x=c[k]["query"].astype(str); ex=x[x.isin(S)]
        pv=c[k].loc[x.isin(S),"search_num"].sum()/c[k].search_num.sum()
        print(f"  EXACT {k}: {len(ex)}/{len(x)} = {len(ex)/max(len(x),1):.1%} of rows, {pv:.1%} of that cell's PV")
        if len(ex): print("     e.g. "+" | ".join(f"{q}(搜索第{rank[q]}名)" for q in c[k][x.isin(S)].sort_values('search_num',ascending=False)['query'].astype(str).head(8)))
        tot[(k,"n")]+=len(x); tot[(k,"exact")]+=len(ex)
    au=set(pd.concat([c["助手top1k"]["query"],c["助手random1k"]["query"]]).astype(str))
    print(f"  REVERSE: search top1000 queries appearing verbatim in the assistant: {sum(q in au for q in top1000)}/1000")
    tot[("rev","n")]+=1000; tot[("rev","hit")]+=sum(q in au for q in top1000)
    cores=sorted([q for q in top1000 if len(q)>=3 and not q.isdigit()], key=len, reverse=True)
    for k in ("助手top1k","助手random1k"):
        wrapped=[]
        for q in c[k]["query"].astype(str):
            for core in cores:
                if core in q and len(q)>=len(core)+2: wrapped.append((core,q)); break
        n=len(c[k]); w=len(wrapped); tot[(k,"wrap")]+=w
        print(f"  WRAP {k}: {w}/{n} = {w/max(n,1):.1%}")
        if wrapped:
            wq=pd.Series([q for _,q in wrapped]); wc=pd.Series([cc for cc,_ in wrapped])
            print(f"     wrapped rows: question {wq.str.contains(Q).mean():.1%} (cores {wc.str.contains(Q).mean():.1%}); first-person {wq.str.contains(FIRST).mean():.1%}; median length core {wc.str.len().median():.0f} -> {wq.str.len().median():.0f}")
            for cc,q in pd.DataFrame(wrapped,columns=["c","q"]).sample(min(8,w),random_state=3).values: print(f"       [{cc}] -> {q[:64]}")
print("\n==================== POOLED ====================")
for k in ("助手top1k","助手random1k"):
    print(f"  {k}: exact {tot[(k,'exact')]}/{tot[(k,'n')]} = {tot[(k,'exact')]/tot[(k,'n')]:.1%}; wrap {tot[(k,'wrap')]}/{tot[(k,'n')]} = {tot[(k,'wrap')]/tot[(k,'n')]:.1%}")
print(f"  reverse: {tot[('rev','hit')]}/5000 = {tot[('rev','hit')]/5000:.1%}")
