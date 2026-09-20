# -*- coding: utf-8 -*-
"""Helpers for the 教育 deep-dive (Python re, not RE2)."""
import sys, re, math
SP=__import__("os").environ.get("SVA_WORKDIR", __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "work"))
sys.path.insert(0,SP)
from sva_common import *
import pandas as pd, numpy as np
def wilson(k,n,z=1.96):
    if n==0: return (float('nan'),float('nan'))
    p=k/n; den=1+z*z/n; c=(p+z*z/(2*n))/den; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return (max(0,c-h)*100,min(1,c+h)*100)
def fmt(k,n):
    lo,hi=wilson(k,n); return f"{k/n*100:.1f}% [{lo:.1f}–{hi:.1f}] (k={k}/n={n})"
def newcombe(k1,n1,k2,n2):
    p1,p2=k1/n1,k2/n2; l1,u1=[x/100 for x in wilson(k1,n1)]; l2,u2=[x/100 for x in wilson(k2,n2)]
    d=p2-p1; lo=d-math.sqrt((p2-l2)**2+(u1-p1)**2); hi=d+math.sqrt((u2-p2)**2+(p1-l1)**2)
    return d*100,lo*100,hi*100
def has(series,pat):
    r=re.compile(pat); return series.astype(str).map(lambda x: bool(r.search(x)))
def edu():
    f=pd.read_parquet(f"{SP}/sva_final_rows.parquet"); e=f[f.domain=="教育"].copy(); e["query"]=e["query"].astype(str); return e
def users(e):
    return {k:e[(e.surface==k)&(e.tier=="user")].copy() for k in ["search_top1000","assistant_top1k","assistant_random1k"]}
SN={"search_top1000":"搜索头","assistant_top1k":"助手头","assistant_random1k":"助手尾"}
