# -*- coding: utf-8 -*-
"""影视 §1: tier composition (rows, PV) per surface, v2 vs v3; head PV removed by each tier; video-generation entry strings."""
import sys, re, pandas as pd
SP=__import__("os").environ.get("SVA_WORKDIR", __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "work"))
sys.path.insert(0,SP)
import sva_common as sc
a3,s3=sc.load(); a2=pd.read_parquet(f"{SP}/clean_v2/assistant_tiered.parquet"); s2=pd.read_parquet(f"{SP}/clean_v2/search2026_tiered.parquet")
out=[]
for ver,a,s in (("v2",a2,s2),("v3",a3,s3)):
    aa=a[a.l1=="影视动漫"]
    for snap in ("top1k","random1k"):
        x=aa[aa.snapshot==snap]; g=x.groupby("tier").agg(rows=("query","size"),pv=("search_num","sum"))
        for t,r in g.iterrows(): out.append(dict(ver=ver,surface="assistant_"+snap,tier=t,rows=int(r.rows),row_share=r.rows/len(x),pv=int(r.pv),pv_share=r.pv/x.search_num.sum()))
    ss=s[s.domain=="影视"].sort_values("wise_pv",ascending=False)
    for nm,x in (("search_top1000_span",ss.head(1000)),("search_top10k",ss)):
        g=x.groupby("tier").agg(rows=("query","size"),pv=("wise_pv","sum"))
        for t,r in g.iterrows(): out.append(dict(ver=ver,surface=nm,tier=t,rows=int(r.rows),row_share=r.rows/len(x),pv=int(r.pv),pv_share=r.pv/x.wise_pv.sum()))
O=pd.DataFrame(out); O.to_csv(f"{SP}/sva_dom_film_tiers.csv",index=False,encoding="utf-8-sig")
pd.set_option("display.width",200); print(O.assign(row_share=(O.row_share*100).round(2),pv_share=(O.pv_share*100).round(2)).to_string(index=False))
h=a3[(a3.l1=="影视动漫")&(a3.snapshot=="top1k")]; tot=h.search_num.sum()
print("\nhead total PV (all tiers):",int(tot), " rows:",len(h))
GENV=re.compile(r"(生成.{0,30}视频|变视频|做成视频|变成视频|AI视频)")
for q in ["生成视频","帮我生成一个视频","变视频","根据对话内容生成视频","请帮我生成一个视频，比例为原比例。","滤镜"]:
    y=h[h["query"]==q]; print(f"  {q}: rows {len(y)} PV {int(y.search_num.sum())} = {y.search_num.sum()/tot*100:.1f}% of head PV; tiers {y.tier.value_counts().to_dict()}")
m=h["query"].astype(str).map(lambda q: bool(GENV.search(q)))
print(f"  all rows matching video-gen regex: rows {int(m.sum())}, PV {int(h.search_num[m].sum())} = {h.search_num[m].sum()/tot*100:.1f}% ; of which non-user PV {h.search_num[m&(h.tier!='user')].sum()/tot*100:.1f}% ; user rows {int((m&(h.tier=='user')).sum())} PV {h.search_num[m&(h.tier=='user')].sum()/tot*100:.2f}% of head PV")
# the '继续吧' chips
m2=h["query"].astype(str).str.contains("继续吧",regex=False); print(f"  '继续吧' chips: rows {int(m2.sum())} PV {h.search_num[m2].sum()/tot*100:.1f}%")
print("  S1 short replies (需要/好/要/可以/还有吗): rows",int(h[(h.tier=='S1_system_repeated')&h['query'].isin(['需要','好','好的','要','可以','还有吗'])].shape[0]),"PV%",round(h[(h.tier=='S1_system_repeated')&h['query'].isin(['需要','好','好的','要','可以','还有吗'])].search_num.sum()/tot*100,2))
# v2->v3 changed rows for 影视
k=["query","snapshot"]; m3=a3[a3.l1=="影视动漫"][k+["tier","search_num"]]; m2_=a2[a2.l1=="影视动漫"][k+["tier"]]
j=m3.merge(m2_,on=k,suffixes=("_v3","_v2")); ch=j[j.tier_v3!=j.tier_v2]
print("\nv2→v3 changed (影视):"); print(ch.groupby(["snapshot","tier_v2","tier_v3"]).agg(rows=("query","size"),pv=("search_num","sum")))
