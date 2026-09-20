# -*- coding: utf-8 -*-
"""影视 §6 evidence: hot-title intent mix per surface; title-key PV concentration; what head U13 is; children's-IP row form;
adult-leaning counts (refined screen, never printed as text in the report); voice-like proxy."""
import sys, re, math, numpy as np, pandas as pd
SP=__import__("os").environ.get("SVA_WORKDIR", __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "work"))
sys.path.insert(0,SP); from sva_common import *; import sva_dom_film_rx as R
fm=frame(); NAME={k:v[0] for k,v in fm.FRAME.items()}
def wilson(k,n,z=1.96):
    if n==0: return float('nan'),float('nan'),float('nan')
    p=k/n; den=1+z*z/n; c=(p+z*z/(2*n))/den; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den; return p,max(0,c-h),min(1,c+h)
W=lambda k,n: "%.1f%%[%.1f–%.1f](k=%d,n=%d)"%(wilson(k,n)[0]*100,wilson(k,n)[1]*100,wilson(k,n)[2]*100,k,n)
a,s=load(); c=cells(a,s,"影视"); T=c["搜索top10k"]
F=pd.read_parquet(f"{SP}/sva_final_rows.parquet"); U=F[(F.domain=="影视")&(F.tier=="user")].copy(); U["query"]=U["query"].astype(str)
S1=U[U.surface=="search_top1000"]; AH=U[U.surface=="assistant_top1k"]; AT=U[U.surface=="assistant_random1k"]
# 1. title-key concentration
GEN=r"(《|》|\s|电视剧|电影|动漫|动画片|短剧|韩剧|日剧|泰剧|美剧|国产剧|免费观看|在线观看|在线播放|在线看|免费播放|免费看|全集|完整版|高清|观看|下载|百度网盘|百度云|网盘|演员表|分集剧情介绍|剧情介绍|剧情简介|大结局|结局|是什么|给我|提供|完整|所有|全部|介绍|第.{1,3}集|\d+-\d+集|\d+集|第.{1,3}季|国语|中文|字幕|版|免费|在线|视频|百度|大全|还有哪些|哪些|配角|演员|的|吗|？|\?|是怎样|怎样|剧情|简介|百科|网|播放|看)"
def key(q): k=re.sub(GEN,"",q); return k if len(k)>=2 else None
print("## 1. 片名键 PV 集中度 (近似: 去掉通用词后的剩余串)")
for nm,x,pvc in (("搜索top1000",S1,"pv"),("搜索top10k",T.assign(query=T["query"].astype(str)),"wise_pv"),("助手top1k",AH,"pv")):
    k=x["query"].map(key); g=x.assign(k=k).dropna(subset=["k"]).groupby("k")[pvc].agg(["sum","size"]).sort_values("sum",ascending=False)
    tot=x[pvc].sum(); print(f"  {nm}: top5键 PV占 {g['sum'].head(5).sum()/tot*100:.1f}% ; top10 {g['sum'].head(10).sum()/tot*100:.1f}% ; "+" | ".join(f"{i}({r['size']}行,{r['sum']/tot*100:.1f}%)" for i,r in g.head(10).iterrows()))
# 2. hot-title rows: intent mix
HS=["爱情有烟火","千香","莫离","昨夜将至","问心","炽夏","南部档案","检察官的提案"]; HA=["早春晴朗","醒来","重器","藏锋","花开锦绣"]
print("\n## 2. 热播片名行的意图构成 (各面自己的头部热剧)")
for nm,x,L in (("搜索top1000×搜索热剧",S1,HS),("助手top1k×助手热剧",AH,HA),("助手random1k×助手热剧",AT,HA)):
    m=x["query"].map(lambda q: any(t in q for t in L)); y=x[m]; n=len(y)
    print(f"  {nm}: n={n}, PV占该面 {(y.pv.sum()/x.pv.sum()*100):.1f}%")
    for grp,codes in (("看/找(U09+U02+U01)",["U09","U02","U01"]),("演员表/清单(U08)",["U08"]),("剧情·结局·介绍(U04)",["U04"]),("事实(U03)",["U03"]),("其他",None)):
        k=int(y.u_ds.isin(codes).sum()) if codes else int((~y.u_ds.isin(["U09","U02","U01","U08","U04","U03"])).sum())
        print(f"     {grp}: {W(k,n)}")
    for mk in ("播放页/资源词","演员/角色","剧情/结局"):
        k=sum(R.hit(mk,q) for q in y["query"]); print(f"     marker {mk}: {W(k,n)}")
# 3. head U13 decomposition
print("\n## 3. 助手头部 U13 是什么")
y=AH[AH.u_ds=="U13"]; n=len(y)
mid=y["query"].map(lambda q: R.hit("识别作品/人物",q) or R.hit("指代画面/上文",q))
mvid=y["query"].map(lambda q: bool(re.search(r"(视频|有没有|要)",q)))
print(f"  n={n}, PV占头部user {y.pv.sum()/AH.pv.sum()*100:.1f}% ; 识别/指代画面 {W(int(mid.sum()),n)} PV占U13 {y.pv[mid].sum()/y.pv.sum()*100:.1f}% ; '视频/有没有/要'类 {W(int((mvid&~mid).sum()),n)} ; 余下 {n-int(mid.sum())-int((mvid&~mid).sum())}")
print("  识别类例: "+" | ".join(y[mid].sort_values('pv',ascending=False)["query"].head(10))); print("  其余例: "+" | ".join(y[~mid&~mvid].sort_values('pv',ascending=False)["query"].head(15).str[:20]))
# 4. children's IP rows
print("\n## 4. 儿童向IP行")
PUN=re.compile(r"[，。？！,.?!、：:；;“”\"'（）()《》…~～]")
NOST=re.compile(r"(小时候|童年|长大|当年|以前看|那时候|回忆)")
AGE=re.compile(r"(\d{1,2}岁|幼儿园|小学|孩子|小朋友|宝宝|儿子|女儿|我家娃)")
for nm,x in (("搜索top1000",S1),("搜索top10k",T.assign(query=T["query"].astype(str),u_ds=None,pv=T.wise_pv)),("助手top1k",AH),("助手random1k",AT)):
    m=x["query"].map(lambda q: R.hit("儿童向IP/动画片",q)); y=x[m]; z=x[~m]; n=len(y)
    if n==0: continue
    def rate(df,f): return W(int(df["query"].map(f).sum()),len(df))
    print(f"  {nm}: kids rows {W(n,len(x))}")
    for lab,f in (("疑问句",lambda q:R.hit("疑问句",q)),("因果追问",lambda q:R.hit("因果追问(为什么)",q)),("假设/对战",lambda q:R.hit("假设/对战",q)),("资源词",lambda q:R.hit("播放页/资源词",q)),(">30字",lambda q:len(q)>30),("≥20字且无标点",lambda q: len(q)>=20 and not PUN.search(q)),("怀旧词(小时候/童年…)",lambda q: bool(NOST.search(q))),("年龄/孩子词",lambda q: bool(AGE.search(q)))):
        print(f"     {lab}: kids {rate(y,f)} | non-kids {rate(z,f)}")
    if nm.startswith("助手"): print("     u_ds: "+" ; ".join(f"{k}{NAME[k]} {v}" for k,v in y.u_ds.value_counts().items()))
    if nm=="助手random1k":
        print("     怀旧例: "+" | ".join(y[y['query'].map(lambda q: bool(NOST.search(q)))]["query"].str[:40]))
        print("     假设/对战例: "+" | ".join(y[y['query'].map(lambda q: R.hit('假设/对战',q))]["query"].str[:40]))
        print("     无标点长例: "+" | ".join(y[y['query'].map(lambda q: len(q)>=20 and not PUN.search(q))]["query"].str[:40]))
# voice-like proxy on all surfaces
print("\n## 5. ≥20字且无标点 (口述转写的弱代理)")
for nm,x in (("搜索top1000",S1),("助手top1k",AH),("助手random1k",AT)):
    k=int(x["query"].map(lambda q: len(q)>=20 and not PUN.search(q)).sum()); k20=int((x["query"].str.len()>=20).sum()); print(f"  {nm}: 全体 {W(k,len(x))} ; 在≥20字行中 {W(k,k20)}")
# 6. adult-leaning (refined screen) — counts only
EXPL=re.compile(r"(性感|妩媚|泳装|泳衣|比基尼|内衣|丝袜|脱衣|脱掉|脱下|脱光|往下拉|舌吻|大胸|胸部|露胸|湿身|裸|色情|黄片|成人|诱惑|撩人)")
KISS=re.compile(r"(接吻|亲吻|亲亲|嘴对嘴|吻)")
print("\n## 6. 成人/灰色 (计数)")
for nm,x in (("搜索top1000",S1),("助手top1k",AH),("助手random1k",AT)):
    n=len(x); u12=int((x.u_ds=="U12").sum()); ownad=int(x.own_intent.astype(str).str.contains("成人|情色").sum())
    ex=x["query"].map(lambda q: bool(EXPL.search(q)))
    print(f"  {nm}: U12 {W(u12,n)} PV{(x.pv[x.u_ds=='U12'].sum()/x.pv.sum()*100):.1f}% ; own成人类 {W(ownad,n)} ; 露骨关键词 {W(int(ex.sum()),n)} ; U12∪露骨词 {W(int(((x.u_ds=='U12')|ex).sum()),n)}")
    g=x["query"].map(lambda q: R.hit("视频/图像生成请求",q)); gn=int(g.sum())
    if gn:
        ga=g&((x.u_ds=="U12")|ex); gk=g&~((x.u_ds=="U12")|ex)&x["query"].map(lambda q: bool(KISS.search(q)))
        print(f"     生成请求 n={gn}: 含性暗示(U12∪露骨词) {W(int(ga.sum()),gn)} ; 仅亲吻类 {W(int(gk.sum()),gn)} ; 其余 {gn-int(ga.sum())-int(gk.sum())}")
        print("     生成请求 u_ds: "+" ; ".join(f"{k} {v}" for k,v in x[g].u_ds.value_counts().items()))
        print("     非暗示生成请求例: "+" | ".join(x[g&~((x.u_ds=='U12')|ex)&~x['query'].map(lambda q: bool(KISS.search(q)))]["query"].str[:36].head(14)))
    kid=x["query"].map(lambda q: R.hit("儿童向IP/动画片",q)); print(f"     儿童IP ∩ (U12∪露骨词∪亲吻词): {int((kid&((x.u_ds=='U12')|ex|x['query'].map(lambda q: bool(KISS.search(q))))).sum())}")
tt=T.assign(query=T["query"].astype(str)); ex10=tt["query"].map(lambda q: bool(EXPL.search(q)))
print(f"  搜索top10k: own成人类 {W(int(tt.td_l1_name.str.contains('成人|情色').sum()),len(tt))} PV{tt.wise_pv[tt.td_l1_name.str.contains('成人|情色')].sum()/tt.wise_pv.sum()*100:.1f}% ; 露骨关键词 {W(int(ex10.sum()),len(tt))}")
t1=tt.head(1000); print(f"  搜索top1000(cells) own成人类 {W(int(t1.td_l1_name.str.contains('成人|情色').sum()),1000)} ; 露骨关键词 {W(int(t1['query'].map(lambda q: bool(EXPL.search(q))).sum()),1000)}")
# 7. personal flag & tail length
print("\n## 7. p_ds=1 例 (tail, 仅看类型)"); y=AT[AT.p_ds==1]; print("  n=",len(y)," u_ds:",y.u_ds.value_counts().to_dict())
