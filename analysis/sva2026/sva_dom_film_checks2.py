# -*- coding: utf-8 -*-
"""影视 checks 2: templated 《X》 question rows (possible chips) and hot-title sensitivity; ranks for pair examples;
description→title search rows; parents choosing content; platform/account rows among p_ds=1; v2→v3 restored rows."""
import sys, re, math, pandas as pd
SP=__import__("os").environ.get("SVA_WORKDIR", __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "work"))
sys.path.insert(0,SP); from sva_common import *; import sva_dom_film_rx as R
def wilson(k,n,z=1.96):
    p=k/n; den=1+z*z/n; c=(p+z*z/(2*n))/den; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den; return p,max(0,c-h),min(1,c+h)
W=lambda k,n: "%.1f%%[%.1f–%.1f](k=%d,n=%d)"%(wilson(k,n)[0]*100,wilson(k,n)[1]*100,wilson(k,n)[2]*100,k,n)
a,s=load(); c=cells(a,s,"影视"); T=c["搜索top10k"].reset_index(drop=True); T["query"]=T["query"].astype(str); rank={q:i+1 for i,q in enumerate(T["query"])}
F=pd.read_parquet(f"{SP}/sva_final_rows.parquet"); U=F[(F.domain=="影视")&(F.tier=="user")].copy(); U["query"]=U["query"].astype(str)
S1=U[U.surface=="search_top1000"]; AH=U[U.surface=="assistant_top1k"]; AT=U[U.surface=="assistant_random1k"]
print("## 1. head rows containing 《》 (adult-masked)")
y=AH[AH["query"].str.contains("《",regex=False)].sort_values("pv",ascending=False)
y=y.loc[pd.Series([not R.adult_like(q,u,o) for q,u,o in zip(y["query"],y.u_ds,y.own_intent)],index=y.index,dtype=bool)]
print("  n=",len(y)); print("  "+" | ".join(f"{q}({int(p)})" for q,p in zip(y["query"],y.pv)))
TPL=re.compile(r"^(给我(提供)?)?《([^》]{1,15})》(的)?(完整演员表|结局是什么|大结局是什么|剧情简介是什么|还有哪些演员|还有哪些配角演员|主要演员有哪些|演员表完整版|演员表完整名单|剧情结局是怎样的|剧情介绍|分集剧情介绍|第.{1,3}集有哪些重要细节|里有哪些值得关注的配角)[？?]?$")
print("\n## 2. templated 《X》question rows")
for nm,x in (("assistant_top1k",AH),("assistant_random1k",AT),("search_top10k",T.assign(pv=T.wise_pv))):
    m=x["query"].map(lambda q: bool(TPL.match(q))); z=x[m]
    titles=z["query"].map(lambda q: TPL.match(q).group(3)); frames=z["query"].map(lambda q: re.sub(r"《[^》]+》","《X》",q))
    print(f"  {nm}: rows {W(int(m.sum()),len(x))} PV {z.pv.sum()/x.pv.sum()*100:.1f}% ; distinct titles {titles.nunique()} ; frames:")
    for fr,g in z.groupby(frames): print(f"     {fr}: rows {len(g)} titles {g['query'].map(lambda q: TPL.match(q).group(3)).nunique()} PV {int(g.pv.sum())} → "+", ".join(g['query'].map(lambda q: TPL.match(q).group(3)).unique()[:8]))
HA=["早春晴朗","醒来","重器","藏锋","花开锦绣"]
x=AH[AH["query"].map(lambda q: any(t in q for t in HA))]; xt=x[~x["query"].map(lambda q: bool(TPL.match(q)))]
for nm,z in (("with templates",x),("templates removed",xt)):
    n=len(z); k1=int(z.u_ds.isin(["U09","U02","U01"]).sum()); k2=int((z.u_ds=="U08").sum()); k3=int((z.u_ds=="U04").sum())
    print(f"  hot-title head rows {nm}: n={n} PV share of head {z.pv.sum()/AH.pv.sum()*100:.1f}% ; watch/find {W(k1,n)} ; U08 {W(k2,n)} ; U04 {W(k3,n)} ; U08+U04 {W(k2+k3,n)}")
print("\n## 3. ranks (search top10k user rows) for pair candidates")
for q in ["舞蹈视频","美女视频","动画片儿童3-6岁免费大全","奥特曼动画片3-6岁","汪汪队立大功全集免费","唐朝诡事录之蜀道","蜡笔小新电影","鬼灭之刃","鬼灭之刃第一季","熊出没","西游记","斗罗大陆","悬案","短剧","热门短剧推荐","醒来电视剧","重器电视剧免费观看全集于和伟","cctv5今日赛程表","动画片","cctv5在线直播","小猪佩奇","早春晴朗电视剧","森林里的熊先生冬眠中类似动漫","主角演员表","关晓彤演的全部剧","铁道游击队","新闻联播今天19:00回放","抓特务","抓特务演员表全部名单","票房","电视剧推荐2026热播的剧","哔哩哔哩","凡人修仙传动漫全集在线观看"]:
    print(f"  {q}: #{rank.get(q)}  u_ds(top1000)={S1.set_index('query').u_ds.get(q,'-') if q in set(S1['query']) else '-'}")
print("\n## 4. search '由描述或别名定位标准片名' rows (masked, top 25 by PV)")
z=T[T.td_l1_name=="由描述或别名定位标准片名"]; z=z.loc[pd.Series([not R.adult_like(q) for q in z["query"]],index=z.index,dtype=bool)]
print("  n_safe=",len(z)," of ",int((T.td_l1_name=="由描述或别名定位标准片名").sum())); print("  "+" | ".join(f"{q}(#{rank[q]})" for q in z["query"].head(25)))
print("\n## 5. parents / audience-fit rows")
PAR=re.compile(r"(适合.{0,8}(孩子|儿童|小朋友|小学生|岁|宝宝|幼儿)|(孩子|儿童|小朋友|小学生|宝宝|幼儿).{0,6}(看的|适合|能看)|\d{1,2}\s*[-~到]\s*\d{1,2}岁|\d{1,2}岁(孩子|儿童|宝宝))")
for nm,x in (("search_top1000",S1),("search_top10k",T),("assistant_top1k",AH),("assistant_random1k",AT)):
    m=x["query"].map(lambda q: bool(PAR.search(q))); print(f"  {nm}: {W(int(m.sum()),len(x))}: "+" | ".join(x[m]['query'].head(10).str[:30]))
print("\n## 6. p_ds=1 tail rows about platform/account")
PL=re.compile(r"(抖音|快手|[Bb]站|哔哩哔哩|视频号|账号|播放量|收益|作品|屏蔽|登录|注册|红包|发视频|评论)")
y=AT[AT.p_ds==1]; m=y["query"].map(lambda q: bool(PL.search(q))); print(f"  {W(int(m.sum()),len(y))}")
m2=AT["query"].map(lambda q: bool(PL.search(q))&bool(re.search(r"(抖音|快手|[Bb]站|哔哩哔哩|视频号)",q))); print(f"  all tail rows naming a platform with account/creator words: {W(int(m2.sum()),len(AT))}; l2 of them: {AT[m2].l2.value_counts().to_dict()}")
print("\n## 7. v2→v3 restored rows (影视)")
a2=pd.read_parquet(f"{SP}/clean_v2/assistant_tiered.parquet"); k=["query","snapshot"]
j=a[a.l1=="影视动漫"][k+["tier","search_num"]].merge(a2[a2.l1=="影视动漫"][k+["tier"]],on=k,suffixes=("_v3","_v2"))
print(j[(j.tier_v2!="user")&(j.tier_v3=="user")].to_string())
