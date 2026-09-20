# -*- coding: utf-8 -*-
"""金融 deep dive, overlap/wrap/NN/templates/dates. v3 tiers, user rows."""
import sys, math, re, collections
SP=__import__("os").environ.get("SVA_WORKDIR", __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "work"))
sys.path.insert(0, SP)
from sva_common import *
import numpy as np
D="金融"; OUT=open(f"{SP}/sva_dom_fin_b.txt","w",encoding="utf-8")
def P(*x): print(*x,file=OUT,flush=True)
def wilson(k,n,z=1.96):
    if n==0: return float("nan"),float("nan"),float("nan")
    p=k/n; den=1+z*z/n; c=(p+z*z/(2*n))/den; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den; return p,max(0.0,c-h),min(1.0,c+h)
def W(k,n): p,lo,hi=wilson(k,n); return f"{p*100:.1f}%[{lo*100:.1f}–{hi*100:.1f}] ({k}/{n})"
fm=frame(); NAME={k:v[0] for k,v in fm.FRAME.items()}
a,s=load(); C=cells(a,s,D)
F=pd.read_parquet(f"{SP}/sva_final_rows.parquet"); F=F[(F.domain==D)&(F.tier=="user")].copy(); F["query"]=F["query"].astype(str)
S10=C["搜索top10k"].copy(); S10["query"]=S10["query"].astype(str); S1k=S10.head(1000)
rank={q:i+1 for i,q in enumerate(S10["query"])}; spv=dict(zip(S10["query"],S10.wise_pv))
AH=F[F.surface=="assistant_top1k"].copy(); AT=F[F.surface=="assistant_random1k"].copy(); SH=F[F.surface=="search_top1000"].copy()

P("#### B1. duplicates across l2 in assistant cells (same string under several l2)")
for nm,x in (("助手头",C["助手top1k"]),("助手尾",C["助手random1k"])):
    q=x["query"].astype(str); P(f"  {nm}: rows {len(q)} distinct {q.nunique()} ; strings appearing >1 {int((q.value_counts()>1).sum())}  例: "+" | ".join(q.value_counts()[q.value_counts()>1].index[:8]))

P("\n#### B2. EXACT overlap  [filter: assistant user rows; search user rows top10k/top1000 by wise_pv]")
for nm,x in (("助手头",AH),("助手尾",AT)):
    in10=x["query"].isin(set(S10["query"])); in1=x["query"].isin(set(S1k["query"]))
    P(f"  {nm}: ∈搜索top10k {W(int(in10.sum()),len(x))} PV% {x[in10].pv.sum()/x.pv.sum()*100:.1f} ; ∈搜索top1000 {W(int(in1.sum()),len(x))} PV% {x[in1].pv.sum()/x.pv.sum()*100:.1f}")
    if nm=="助手头":
        e=x[in10].copy(); e["spv"]=e["query"].map(spv); e["srank"]=e["query"].map(rank)
        from scipy.stats import spearmanr
        r=spearmanr(e.spv,e.pv); P(f"   exact rows: Spearman(search PV, assistant PV) rho={r.correlation:.3f} n={len(e)} ; median search rank {e.srank.median():.0f}; share with search rank ≤1000 {(e.srank<=1000).mean()*100:.1f}%")
        e["ratio"]=e.pv/e.spv; P(f"   assistant/search PV ratio quantiles p10 {e.ratio.quantile(.1):.4f} p50 {e.ratio.quantile(.5):.4f} p90 {e.ratio.quantile(.9):.4f}")
        P("   intent of exact vs non-exact head rows: "+" ; ".join(f"{c}{NAME[c]} {(e.u_ds==c).mean()*100:.1f}/{(x[~in10].u_ds==c).mean()*100:.1f}" for c in ["U01","U02","U03","U04","U05","U06","U07","U08","U09","U13"]))
        e[["query","pv","l2","u_ds","spv","srank"]].sort_values("pv",ascending=False).to_csv(f"{SP}/sva_dom_fin_exact_head.csv",index=False,encoding="utf-8-sig")
        P("   top exact by assistant PV: "+" | ".join(f"{q}(助手{int(p)},搜索#{int(r_)})" for q,p,r_ in e.sort_values("pv",ascending=False)[["query","pv","srank"]].head(12).values))
        P("   top1000-search strings missing from assistant (top PV): "+" | ".join(S1k[~S1k["query"].isin(set(C['助手top1k']['query'].astype(str))|set(C['助手random1k']['query'].astype(str)))]["query"].head(20)))
rev=S1k["query"].isin(set(AH["query"])|set(AT["query"])); P(f"  REVERSE: search top1000 strings found verbatim in assistant (head or tail) {W(int(rev.sum()),1000)} ; search PV share of those {S1k[rev].wise_pv.sum()/S1k.wise_pv.sum()*100:.1f}%")

P("\n#### B3. WRAP (search top1000 core, len≥3, not digits, contained in longer assistant query len≥core+2) and intent change")
lab_core=SH.drop_duplicates("query").set_index("query").u_ds.to_dict()
cores=sorted([q for q in SH["query"].unique() if len(q)>=3 and not q.isdigit()],key=len,reverse=True)
pairs=[]
for nm,x in (("assistant_top1k",AH),("assistant_random1k",AT)):
    for q,u,pv,p in x[["query","u_ds","pv","p_ds"]].itertuples(index=False):
        for core in cores:
            if core in q and len(q)>=len(core)+2: pairs.append(dict(surface=nm,core=core,query=q,u_core=lab_core.get(core),u_wrap=u,pv=pv,p=p)); break
PW=pd.DataFrame(pairs); PW.to_csv(f"{SP}/sva_dom_fin_wrap_pairs.csv",index=False,encoding="utf-8-sig")
for nm,x in (("assistant_top1k",AH),("assistant_random1k",AT)):
    y=PW[PW.surface==nm]; P(f"  {nm}: wrapped {W(len(y),len(x))} ; intent unchanged {(y.u_core==y.u_wrap).mean()*100:.1f}% ; top cores: "+", ".join(f"{k}({v})" for k,v in y.core.value_counts().head(10).items()))
    for (c1,c2),v in y[y.u_core!=y.u_wrap].groupby(["u_core","u_wrap"]).size().sort_values(ascending=False).head(8).items():
        e=y[(y.u_core==c1)&(y.u_wrap==c2)]
        P(f"    {c1}{NAME[c1]}→{c2}{NAME[c2]} {v} ({v/len(y)*100:.1f}%): "+" | ".join(f"[{a_}]→{b_[:30]}" for a_,b_ in zip(e.core.head(6),e["query"].head(6))))

P("\n#### B4. NEAREST SEARCH NEIGHBOUR (bge-base; sva_semantic_nn.parquet) on v3 user rows")
NN=pd.read_parquet(f"{SP}/sva_semantic_nn.parquet"); NN=NN[NN.domain==D].copy(); NN["surface"]=NN.surface.map({"助手top1k":"assistant_top1k","助手random1k":"assistant_random1k"}); NN=NN.drop_duplicates(["surface","query"])
A_=pd.concat([AH,AT]).merge(NN[["surface","query","sim","nn_query","nn_search_rank"]],on=["surface","query"],how="left")
BAND=lambda v: None if pd.isna(v) else ("同" if v>=0.999 else ("近" if v>=0.80 else ("中" if v>=0.70 else "远")))
A_["band"]=A_.sim.map(BAND)
lab_all=pd.read_parquet(f"{SP}/sva_final_rows.parquet"); lab_all=lab_all[lab_all.domain==D].dropna(subset=["u_ds"]).drop_duplicates("query").set_index("query").u_ds.to_dict()
A_["u_nn"]=A_.nn_query.map(lab_all)
A_.to_parquet(f"{SP}/sva_dom_fin_nn.parquet",index=False)
for nm in ("assistant_top1k","assistant_random1k"):
    x=A_[A_.surface==nm]; n=int(x.sim.notna().sum())
    P(f"  {nm}: matched {n}/{len(x)} ; "+" | ".join(f"{b} {W(int((x.band==b).sum()),n)}" for b in ["同","近","中","远"])+f" ; median sim {x.sim.median():.3f}")
    for b in ["近","中"]:
        y=x[(x.band==b)].dropna(subset=["u_nn"]); P(f"   band {b}: neighbour labelled {len(y)}/{int((x.band==b).sum())}; intent unchanged {(y.u_nn==y.u_ds).mean()*100:.1f}%")
        for (c1,c2),v in y[y.u_nn!=y.u_ds].groupby(["u_nn","u_ds"]).size().sort_values(ascending=False).head(6).items():
            e=y[(y.u_nn==c1)&(y.u_ds==c2)]
            P(f"    {c1}{NAME[c1]}→{c2}{NAME[c2]} {v}: "+" | ".join(f"[{a_[:16]}]→{b_[:28]}({s_:.2f})" for a_,b_,s_ in zip(e.nn_query.head(5),e["query"].head(5),e.sim.head(5))))
    P("   far-band (<0.70) intent mix: "+" ; ".join(f"{c}{NAME[c]} {k}" for c,k in x[x.band=="远"].u_ds.value_counts().items()))

P("\n#### B5. TEMPLATE CONCENTRATION of forecast-type questions")
FC=r"(未来|上涨空间|目标价|涨停|走势如何|走势怎么样|预测|能涨|会涨|会跌|还能涨|值得买|值得购买|值得入手|值得持有|是否值得|能买吗|可以买|还可以买|前景|首日能|首日大概|能赚多少|抄底|加仓|减仓|买入|卖出)"
SUF=["未来有上涨空间吗","上市首日能涨多少","未来走势如何","目标价是多少","能否继续涨停","值得买入吗","上市首日能赚多少","走势如何","还能涨吗","能涨停吗","目标价","后市怎么看"]
S10q=S10["query"]
for nm,x in (("搜索top1000",SH),("搜索top10k",S10.assign(pv=S10.wise_pv)),("助手头",AH),("助手尾",AT)):
    m=x["query"].str.contains(FC,regex=True); y=x[m]
    P(f"  {nm}: forecast-marker rows {W(int(m.sum()),len(x))}")
    t=[]
    for sfx in SUF:
        k=int(y["query"].str.endswith(sfx).sum())
        if k: t.append(f"…{sfx} {k}")
    P("    exact suffix counts: "+" ; ".join(t))
    suf=collections.Counter()
    for q in y["query"]:
        for L in range(5,10):
            if len(q)>L+1: suf[q[-L:]]+=1
    top=[(k,v) for k,v in suf.most_common(40) if v>=4][:12]
    P("    most common 5–9 char endings: "+" ; ".join(f"…{k}:{v}" for k,v in top))
    # how many forecast rows share an ending (>=6 chars) with >=3 other forecast rows
    ends=collections.Counter(q[-7:] for q in y["query"] if len(q)>=9)
    shared=sum(1 for q in y["query"] if len(q)>=9 and ends[q[-7:]]>=4)
    P(f"    rows whose last-7-char ending is shared by ≥4 forecast rows: {W(shared,len(y))}")

P("\n#### B6. DATE WINDOW — explicit X月X日 mentions (assistant all 33 categories user rows; 金融 search top10k user)")
DT=re.compile(r"(\d{1,2})月(\d{1,2})(日|号)")
for nm,x in (("助手random1k 全33",a[(a.snapshot=="random1k")&(a.tier=="user")]),("助手top1k 全33",a[(a.snapshot=="top1k")&(a.tier=="user")]),("搜索top10k 5域",s[s.tier=="user"]),("助手 金融 两层",pd.concat([C["助手top1k"],C["助手random1k"]]))):
    ms=collections.Counter()
    for q in x["query"].astype(str):
        for mo,dd,_ in DT.findall(q):
            if 1<=int(mo)<=12 and 1<=int(dd)<=31: ms[f"{int(mo)}月"]+=1
    P(f"  {nm}: n rows {len(x):,}; month mentions "+", ".join(f"{k}:{v}" for k,v in sorted(ms.items(),key=lambda kv:-kv[1])[:8]))
md=collections.Counter()
for q in pd.concat([C["助手top1k"],C["助手random1k"]])["query"].astype(str):
    for mo,dd,_ in DT.findall(q): md[f"{int(mo)}月{int(dd)}日"]+=1
P("  金融 assistant exact dates: "+", ".join(f"{k}:{v}" for k,v in md.most_common(15)))
md=collections.Counter()
for q in a[(a.tier=="user")]["query"].astype(str):
    for mo,dd,_ in DT.findall(q): md[f"{int(mo)}月{int(dd)}日"]+=1
P("  全33 assistant exact dates top: "+", ".join(f"{k}:{v}" for k,v in md.most_common(20)))

P("\n#### B7. CLUSTERS: bank-code decode, 蚂蚁 daily quiz, 9920")
for nm,x in (("搜索top1000",SH),("搜索top10k",S10.assign(pv=S10.wise_pv)),("助手头",AH),("助手尾",AT)):
    q=x["query"]
    for lab,pat in (("9920",r"9920"),("4位银行代码(银行|工行|农行|建行|卡).*\d{4}",r"(银行|工行|农行|建行|卡).*\d{4}"),("蚂蚁庄园/新村",r"(蚂蚁庄园|蚂蚁新村)"),("余额/可用余额",r"余额"),("扣费/扣款/扣钱/代扣",r"(扣费|扣款|扣钱|扣了|代扣|批扣|自动扣|扣\d)")):
        m=q.str.contains(pat,regex=True)
        P(f"  {nm} {lab}: {W(int(m.sum()),len(x))} PV% {x[m].pv.sum()/x.pv.sum()*100:.2f}  例: "+" | ".join(x[m].sort_values('pv',ascending=False)["query"].head(5).str[:26]))
OUT.close(); print("ok")
