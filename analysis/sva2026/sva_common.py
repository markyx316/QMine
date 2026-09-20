# -*- coding: utf-8 -*-
"""Shared definitions for the search-vs-assistant analyses. Import with CLEAN=<dir>."""
import os, re, importlib.util
import pandas as pd
QM=__import__("os").path.abspath(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", ".."))
SP=__import__("os").environ.get("SVA_WORKDIR", __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "work"))
CLEAN=os.environ.get("CLEAN", f"{SP}/clean_v3")
CAT={"金融":"金融","医疗":"医疗","教育":"教育培训","影视":"影视动漫","人物":"人物"}
RUN={"金融":"fin-pool","医疗":"med-pool","教育":"edu-pool","影视":"film-pool","人物":"ppl-pool"}
SURF=["搜索top1000","搜索top10k","助手top1k","助手random1k"]
def frame():
    spec=importlib.util.spec_from_file_location("uif",f"{QM}/tools/unified_intent_frame.py"); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
def load():
    return pd.read_parquet(f"{CLEAN}/assistant_tiered.parquet"), pd.read_parquet(f"{CLEAN}/search2026_tiered.parquet")
def cells(a,s,dom,user_only=True):
    ss=s[(s.domain==dom)&((s.tier=="user") if user_only else True)].sort_values("wise_pv",ascending=False)
    aa=a[(a.l1==CAT[dom])&((a.tier=="user") if user_only else True)]
    return {"搜索top1000":ss.head(1000),"搜索top10k":ss,"助手top1k":aa[aa.snapshot=="top1k"],"助手random1k":aa[aa.snapshot=="random1k"]}
