# -*- coding: utf-8 -*-
"""(1) PV-weighted head metrics + concentration; (2) metric shift caused by cleaning; (3) top removed rows per tier."""
import sys, numpy as np
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from sva_common import *
src=open(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "sva_metrics.py"),encoding="utf-8").read(); i,j=src.index("M={"),src.index("C={d:"); exec(src[i:j]); assert "疑问句" in M
M["委托/动作开头"]=r"^(请你?|帮我|帮忙|给我|麻烦|你给我|你帮我|能不能帮我|能否帮我|能帮我|可以帮我|写|画|生成|制作|分析|预测|推荐|总结|概括|解读|翻译|计算|对比|比较|列出|整理)"
M["是非核实"]=r"(是不是|是否|真的|吗[？?。，,]?$|对吗|对不对)"; M["数量追问"]=r"(多少|几(个|次|天|岁|年|周|月|粒|片|号|点)|多久|多长)"
a,s=load(); print("assistant cols:",list(a.columns)); print("search cols:",list(s.columns))
def wq(x,w,q):
    o=np.argsort(x); x,w=np.asarray(x)[o],np.asarray(w,float)[o]; c=np.cumsum(w)/w.sum(); return x[np.searchsorted(c,q)]
out=open(f"{SP}/sva_pv_weighted_v3.txt","w")
def P(*x): print(*x,file=out)
P("#### HEAD SURFACES — row share vs PV-weighted share (user tier). 搜索top1000 weights=wise_pv, 助手top1k weights=search_num")
heads={}
for d in CAT:
    c=cells(a,s,d); heads[d]={"搜索top1000":(c["搜索top1000"]["query"].astype(str),c["搜索top1000"]["wise_pv"].astype(float)),
                               "助手top1k":(c["助手top1k"]["query"].astype(str),c["助手top1k"]["search_num"].astype(float))}
for name,pat in M.items():
    P(f"\n## {name}   (row% → PV%)")
    for d in list(CAT)+["合计(行等权,PV按各域PV求和)"]:
        cells_txt=[]
        for k in ("搜索top1000","助手top1k"):
            if d in CAT: q,w=heads[d][k]
            else: q=pd.concat([heads[x][k][0] for x in CAT]); w=pd.concat([heads[x][k][1] for x in CAT])
            m=q.str.contains(pat,regex=True).values
            cells_txt.append(f"{k} {m.mean()*100:5.1f}% → {(w.values*m).sum()/w.sum()*100:5.1f}%")
        P(f"  {d:<8} "+" | ".join(cells_txt))
P("\n## 长度: 行中位数 → PV加权中位数")
for d in CAT:
    P(f"  {d:<4} "+" | ".join(f"{k} {heads[d][k][0].str.len().median():.0f} → {wq(heads[d][k][0].str.len().values,heads[d][k][1].values,.5):.0f}" for k in heads[d]))
P("\n## 流量集中度: top10 query PV占比 / top100 PV占比 / 有效query数(1/Σp²)")
for d in CAT:
    t=[]
    for k in heads[d]:
        w=np.sort(heads[d][k][1].values)[::-1]; p=w/w.sum(); t.append(f"{k} {p[:10].sum()*100:4.1f}% / {p[:100].sum()*100:4.1f}% / {1/(p**2).sum():6.1f}")
    P(f"  {d:<4} "+" | ".join(t))
out.close()
# (2) cleaning impact
out=open(f"{SP}/sva_clean_impact_v3.txt","w")
KEYS=["疑问句","祈使句(严格)","第一人称","输出约束","资源词","导航词"]
P("#### CLEANING IMPACT — metric on ALL rows vs USER rows (row share); top1k also PV-weighted")
for d in CAT:
    P(f"\n== {d} ==")
    ca,cu=cells(a,s,d,user_only=False),cells(a,s,d,user_only=True)
    for k in ("搜索top1000","助手top1k","助手random1k"):
        qa,qu=ca[k]["query"].astype(str),cu[k]["query"].astype(str)
        wcol="wise_pv" if k.startswith("搜索") else "search_num"; wa,wu=ca[k][wcol].astype(float).values,cu[k][wcol].astype(float).values
        seg=[f"n {len(qa):,}→{len(qu):,}", f"长度中位 {qa.str.len().median():.0f}→{qu.str.len().median():.0f}", f">60字 {(qa.str.len()>60).mean()*100:.2f}%→{(qu.str.len()>60).mean()*100:.2f}%"]
        for key in KEYS:
            ma,mu=qa.str.contains(M[key],regex=True).values,qu.str.contains(M[key],regex=True).values
            t=f"{key} {ma.mean()*100:.1f}→{mu.mean()*100:.1f}"
            if k!="助手random1k": t+=f" (PV {(wa*ma).sum()/wa.sum()*100:.1f}→{(wu*mu).sum()/wu.sum()*100:.1f})"
            seg.append(t)
        P(f"  {k:<12} "+" ; ".join(seg))
out.close()
# (3) removed examples
out=open(f"{SP}/sva_tier_examples_v3.txt","w")
P("#### TOP REMOVED ROWS PER TIER (by PV) — assistant top1k / random1k and search top-10k, 5 domains")
for d,cat in CAT.items():
    P(f"\n== {d} ==")
    ad=a[a.l1==cat]; sd=s[s.domain==d]
    for t in ["S1_system_repeated","S2_system_template","S3_card_passage","S4_suggested_chip","S5_headline","C1_content_free","C2_feed_control"]:
        for snap in ("top1k","random1k"):
            x=ad[(ad.snapshot==snap)&(ad.tier==t)].sort_values("search_num",ascending=False)
            if len(x): P(f"  助手{snap} {t} n={len(x)} PV={int(x.search_num.sum()):,}: "+" | ".join(f"{str(q)[:34]}({int(v):,})" for q,v in zip(x["query"].head(6),x.search_num.head(6))))
        y=sd[sd.tier==t].sort_values("wise_pv",ascending=False)
        if len(y): P(f"  搜索top10k {t} n={len(y)} PV={int(y.wise_pv.sum()):,}: "+" | ".join(f"{str(q)[:34]}({int(v):,})" for q,v in zip(y["query"].head(6),y.wise_pv.head(6))))
out.close(); print("done")
