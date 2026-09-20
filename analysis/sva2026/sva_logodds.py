# -*- coding: utf-8 -*-
"""Distinctive vocabulary per domain: weighted log-odds with an informative Dirichlet prior
(Monroe et al. 2008) on query document-frequency of jieba tokens, plus vocabulary coverage."""
import sys, re, math, collections
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from sva_common import *
import jieba; jieba.setLogLevel(60)
a,s=load()
TOK=re.compile(r"[一-鿿A-Za-z0-9]")
def docs(q): return [{t.strip().lower() for t in jieba.lcut(x) if TOK.search(t)} for x in q]
def dfc(dl):
    c=collections.Counter()
    for t in dl: c.update(t)
    return c
def logodds(ci,cj,a0=500):
    allc=ci+cj; N=sum(allc.values()); ni=sum(ci.values()); nj=sum(cj.values()); out=[]
    for w,t in allc.items():
        aw=a0*t/N; yi,yj=ci[w],cj[w]
        d=math.log((yi+aw)/(ni+a0-yi-aw))-math.log((yj+aw)/(nj+a0-yj-aw))
        out.append((d/math.sqrt(1/(yi+aw)+1/(yj+aw)),w,yi,yj))
    return sorted(out)
PAIRS=[("搜索top1000","助手top1k","同深度头部"),("搜索top10k","助手random1k","搜索更深层 vs 助手长尾(深度不匹配,仅作参考)")]
D={d:{k:docs(v["query"].astype(str)) for k,v in cells(a,s,d).items()} for d in CAT}
for d in list(CAT)+["合计"]:
    dd=D[d] if d!="合计" else {k:sum((D[x][k] for x in CAT),[]) for k in SURF}
    print(f"\n==================== {d} ====================")
    for si,ai,title in PAIRS:
        cs,ca=dfc(dd[si]),dfc(dd[ai]); lo=logodds(cs,ca)
        top_s=[f"{w}({ys}/{ya},z={z:.1f})" for z,w,ys,ya in reversed(lo) if max(ys,ya)>=5][:25]
        top_a=[f"{w}({ys}/{ya},z={z:.1f})" for z,w,ys,ya in lo if max(ys,ya)>=5][:25]
        vocab_s=set(cs)
        occ=sum(len(t) for t in dd[ai]); oov=sum(1 for t in dd[ai] for w in t if w not in vocab_s)
        allin=sum(1 for t in dd[ai] if t and t<=vocab_s)/max(len(dd[ai]),1)
        print(f"\n  [{title}] {si} n={len(dd[si]):,} vs {ai} n={len(dd[ai]):,}  (括号内: 搜索文档频次/助手文档频次)")
        print(f"   助手token中未出现在{si}词表的比例: {oov/occ*100:.1f}% ; 所有token都在搜索词表内的助手query: {allin*100:.1f}%")
        print("   搜索特有: "+"  ".join(top_s))
        print("   助手特有: "+"  ".join(top_a))
