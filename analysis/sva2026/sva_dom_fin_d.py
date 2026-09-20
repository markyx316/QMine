# -*- coding: utf-8 -*-
"""金融 deep dive: grouped intent diffs, full own-class counts, ai04 advice-class composition, PV concentration."""
import sys, math
SP=__import__("os").environ.get("SVA_WORKDIR", __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "work"))
sys.path.insert(0, SP)
from sva_common import *
import numpy as np
OUT=open(f"{SP}/sva_dom_fin_d.txt","w",encoding="utf-8")
def P(*x): print(*x,file=OUT,flush=True)
def wilson(k,n,z=1.96):
    if n==0: return float("nan"),float("nan"),float("nan")
    p=k/n; den=1+z*z/n; c=(p+z*z/(2*n))/den; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den; return p,max(0.0,c-h),min(1.0,c+h)
def newcombe(k1,n1,k2,n2):
    p1,l1,u1=wilson(k1,n1); p2,l2,u2=wilson(k2,n2); d=p2-p1
    return d, d-math.sqrt((p2-l2)**2+(u1-p1)**2), d+math.sqrt((u2-p2)**2+(p1-l1)**2)
D="金融"; a,s=load(); C=cells(a,s,D)
F=pd.read_parquet(f"{SP}/sva_final_rows.parquet"); F=F[(F.domain==D)&(F.tier=="user")].copy(); F["query"]=F["query"].astype(str)
SL={"search_top1000":"搜索1k","assistant_top1k":"助手头","assistant_random1k":"助手尾"}
G={"查值与裸实体 U02+U03":["U02","U03"],"解释/核实/办事/判断/清单 U04–U08":["U04","U05","U06","U07","U08"],"导航+获取内容 U01+U09":["U01","U09"],"无法判定 U13":["U13"]}
P("#### D1. grouped intent shares and diffs  [filter: sva_final_rows domain==金融 tier==user]")
rows=[]
for view in ("all","interp"):
    for g,codes in G.items():
        if view=="interp" and g.startswith("无法"): continue
        r={}
        for sf in SL:
            x=F[F.surface==sf]; x=x if view=="all" else x[x.u_ds!="U13"]; m=x.u_ds.isin(codes); k=int(m.sum()); n=len(x)
            r[sf]=(k,n); p,lo,hi=wilson(k,n); pvs=x[m].pv.sum()/x.pv.sum()
            rows.append(dict(view=view,group=g,surface=sf,n=n,k=k,share=p,lo=lo,hi=hi,pv_share=pvs if sf!="assistant_random1k" else np.nan))
        d1=newcombe(*r["search_top1000"],*r["assistant_top1k"]); d2=newcombe(*r["assistant_top1k"],*r["assistant_random1k"])
        P(f"  [{view}] {g}: "+" | ".join(f"{SL[sf]} {wilson(*r[sf])[0]*100:.1f}[{wilson(*r[sf])[1]*100:.1f}–{wilson(*r[sf])[2]*100:.1f}]" for sf in SL)+f" ; 搜索1k→助手头 {d1[0]*100:+.1f}[{d1[1]*100:+.1f},{d1[2]*100:+.1f}] ; 助手头→助手尾 {d2[0]*100:+.1f}[{d2[1]*100:+.1f},{d2[2]*100:+.1f}]")
pd.DataFrame(rows).to_csv(f"{SP}/sva_dom_fin_intent_groups.csv",index=False,encoding="utf-8-sig")

P("\n#### D2. full own-class counts (td_l1 code / name)  [filter: tier==user]")
for k in ("搜索top1000","搜索top10k","助手top1k","助手random1k"):
    x=C[k]; vc=x.groupby(["td_l1","td_l1_name"]).size().sort_values(ascending=False); n=len(x)
    P(f"  [{k}] n={n}: "+" ; ".join(f"{c}/{nm} {v} ({v/n*100:.1f}%)" for (c,nm),v in vc.items()))

P("\n#### D3. ai04 '投资理财与创业加盟建议' composition by u_ds")
for sf in ("assistant_top1k","assistant_random1k"):
    x=F[(F.surface==sf)&(F.own_intent=="投资理财与创业加盟建议")]; vc=x.u_ds.value_counts()
    P(f"  {SL[sf]} n={len(x)}: "+" ; ".join(f"{c} {v} ({v/len(x)*100:.1f}%)" for c,v in vc.items()))
    P("    U02 examples: "+" | ".join(x[x.u_ds=="U02"].sort_values("pv",ascending=False)["query"].head(8)))

P("\n#### D4. PV concentration and anchor rows")
S1k=C["搜索top1000"]; tot=S1k.wise_pv.sum()
P(f"  search top1000 user PV {tot:,}; top10: "+" | ".join(f"{q} {pv:,} ({pv/tot*100:.1f}%)" for q,pv in S1k[["query","wise_pv"]].head(10).values))
AH=C["助手top1k"]; tah=AH.search_num.sum()
P(f"  assistant top1k user PV {tah:,}; top10: "+" | ".join(f"{q} {pv:,} ({pv/tah*100:.1f}%)" for q,pv in AH.sort_values('search_num',ascending=False)[["query","search_num"]].head(10).values))
S10=C["搜索top10k"]
for rng,(lo,hi) in {"rank1-1000":(0,1000),"1001-3000":(1000,3000),"3001-10000":(3000,10000)}.items():
    x=S10.iloc[lo:hi]; q=x["query"].astype(str)
    P(f"  search {rng}: n={len(x)} 纯数字代码 {q.str.contains(r'^\s*\d{{3,6}}\s*$').mean()*100:.1f}% ; 股吧 {q.str.contains('股吧').mean()*100:.1f}% ; …股票 {q.str.contains(r'(股票|股份|股价)\s*$').mean()*100:.1f}% ; 疑问 {q.str.contains(r'(怎么|如何|为什么|什么|哪个|哪些|多少|吗|呢|是不是|能不能|\?|？)').mean()*100:.1f}%")

P("\n#### D5. p_ds among head marker rows")
BANK=r"(扣费|扣款|扣钱|扣了|扣\d|代扣|批扣|余额|冻结|只收不付|止付|信使费|管理费|年费|9920|4318|1072|9078)"
FC=r"(未来|上涨空间|目标价|涨停|走势如何|走势怎么样|预测|能涨|会涨|会跌|还能涨|值得买|值得购买|值得入手|值得持有|是否值得|能买吗|可以买|还可以买|前景|首日能|首日大概|能赚多少|抄底|加仓|减仓|买入|卖出)"
for sf in ("assistant_top1k","assistant_random1k"):
    x=F[F.surface==sf]
    for nm,pat in (("银行账户异常",BANK),("预测/买卖",FC)):
        y=x[x["query"].str.contains(pat,regex=True)]
        P(f"  {SL[sf]} {nm}: n={len(y)} p_ds=1 {int(y.p_ds.sum())} ; u_ds {dict(y.u_ds.value_counts())} ; PV {y.pv.sum():,.0f} ({y.pv.sum()/x.pv.sum()*100:.1f}%)")
P("\n#### D6. U12 rows in tail (describe only)")
x=F[(F.surface=="assistant_random1k")&(F.u_ds=="U12")]; P("  n="+str(len(x)))
P("\n#### D7. random1k l2 × u_ds (U07 share by l2)")
x=F[F.surface=="assistant_random1k"]
for l2,y in x.groupby("l2"):
    k=int((y.u_ds=="U07").sum()); p,lo,hi=wilson(k,len(y)); P(f"  {l2}: n={len(y)} U07 {p*100:.1f}[{lo*100:.1f}–{hi*100:.1f}] ; U03 {(y.u_ds=='U03').mean()*100:.1f} ; U06 {(y.u_ds=='U06').mean()*100:.1f}")
OUT.close(); print("ok")
