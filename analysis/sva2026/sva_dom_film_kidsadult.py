# -*- coding: utf-8 -*-
"""Counts only (no text): children's IP rows that any adult signal touches — U12, own-taxonomy adult class, explicit keywords."""
import sys, re, pandas as pd
SP=__import__("os").environ.get("SVA_WORKDIR", __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "work"))
sys.path.insert(0,SP); from sva_common import *; import sva_dom_film_rx as R
EXPL=re.compile(r"(性感|妩媚|泳装|泳衣|比基尼|内衣|丝袜|脱衣|脱掉|脱下|脱光|往下拉|舌吻|大胸|胸部|露胸|湿身|裸|色情|黄片|成人|诱惑|撩人)")
a,s=load(); c=cells(a,s,"影视"); T=c["搜索top10k"]
F=pd.read_parquet(f"{SP}/sva_final_rows.parquet"); U=F[(F.domain=="影视")&(F.tier=="user")].copy(); U["query"]=U["query"].astype(str)
for nm,x,own in (("search_top1000",U[U.surface=="search_top1000"],"own_intent"),("search_top10k",T.assign(query=T["query"].astype(str),u_ds=None),"td_l1_name"),("assistant_top1k",U[U.surface=="assistant_top1k"],"own_intent"),("assistant_random1k",U[U.surface=="assistant_random1k"],"own_intent")):
    kid=x["query"].map(lambda q: R.hit("儿童向IP/动画片",q)); oad=x[own].astype(str).str.contains("成人|情色"); u12=(x.u_ds=="U12"); ex=x["query"].map(lambda q: bool(EXPL.search(q)))
    print(f"{nm}: kids {int(kid.sum())}; kids∩own-adult {int((kid&oad).sum())}; kids∩U12 {int((kid&u12).sum())}; kids∩explicit {int((kid&ex).sum())}; own-adult total {int(oad.sum())}; own-adult∩U12 {int((oad&u12).sum())}; own-adult∩explicit {int((oad&ex).sum())}")
    if nm=="assistant_top1k":
        g=x["query"].map(lambda q: R.hit("视频/图像生成请求",q)); print(f"   generation∩own-adult {int((g&oad).sum())} of {int(g.sum())}; own_leaf of kids∩own-adult rows: {x[kid&oad].own_leaf.value_counts().to_dict()}")
    if nm=="assistant_random1k":
        print(f"   own_leaf of kids∩own-adult rows: {x[kid&oad].own_leaf.value_counts().to_dict()}")
