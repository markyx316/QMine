# -*- coding: utf-8 -*-
"""金融 deep dive, sections 1-3: tiers, finance markers, unified intent, own taxonomy. v3 tiers."""
import sys, math, re
SP=__import__("os").environ.get("SVA_WORKDIR", __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "work"))
sys.path.insert(0, SP)
from sva_common import *
import numpy as np
pd.set_option("display.width",250); pd.set_option("display.max_colwidth",50)
D="金融"; OUT=open(f"{SP}/sva_dom_fin_a.txt","w",encoding="utf-8")
def P(*x): print(*x,file=OUT,flush=True)
def wilson(k,n,z=1.96):
    if n==0: return float("nan"),float("nan"),float("nan")
    p=k/n; den=1+z*z/n; c=(p+z*z/(2*n))/den; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den; return p,max(0.0,c-h),min(1.0,c+h)
def newcombe(k1,n1,k2,n2):
    p1,l1,u1=wilson(k1,n1); p2,l2,u2=wilson(k2,n2); d=p2-p1
    return d, d-math.sqrt((p2-l2)**2+(u1-p1)**2), d+math.sqrt((u2-p2)**2+(p1-l1)**2)
fm=frame(); CODES=list(fm.FRAME); NAME={k:v[0] for k,v in fm.FRAME.items()}
F=pd.read_parquet(f"{SP}/sva_final_rows.parquet"); F=F[F.domain==D].copy(); F["query"]=F["query"].astype(str)
a,s=load(); C=cells(a,s,D,user_only=True); CALL=cells(a,s,D,user_only=False)
SL={"search_top1000":"搜索1k","assistant_top1k":"助手头","assistant_random1k":"助手尾"}

# ---------- 1. tiers ----------
P("#### 1. TIERS (v3) — rows / PV / PV% per surface  [filter: sva_final_rows.parquet domain==金融; search top10k from clean_v3 search2026_tiered domain==金融]")
for sf in SL:
    x=F[F.surface==sf]; g=x.groupby("tier").agg(rows=("query","size"),pv=("pv","sum")); g["rows%"]=g.rows/g.rows.sum()*100; g["pv%"]=g.pv/g.pv.sum()*100
    P(f"  [{sf}] total rows {len(x):,} PV {x.pv.sum():,.0f}"); P(g.round(2).to_string())
x=CALL["搜索top10k"]; g=x.groupby("tier").agg(rows=("query","size"),pv=("wise_pv","sum")); g["pv%"]=g.pv/g.pv.sum()*100
P(f"  [search_top10k] total rows {len(x):,} PV {x.wise_pv.sum():,}"); P(g.round(3).to_string())
ah=F[F.surface=="assistant_top1k"]; tot=ah.pv.sum()
chip=ah["query"].isin(["👌 好的，继续吧","🆗 行，继续吧"]); ack=ah["query"].isin(["需要","好的","好","要","可以"])
P(f"  head PV: two emoji chips {ah[chip].pv.sum():,.0f} = {ah[chip].pv.sum()/tot*100:.1f}% ({chip.sum()} rows) ; short acks 需要/好的/好/要/可以 {ah[ack].pv.sum():,.0f} = {ah[ack].pv.sum()/tot*100:.1f}% ({ack.sum()} rows)")
P(f"  search user PV cutoffs: 1000th={C['搜索top1000'].wise_pv.min():,}  last(top10k)={C['搜索top10k'].wise_pv.min():,}; assistant top1k user min PV={C['助手top1k'].search_num.min():,}, median={C['助手top1k'].search_num.median():.0f}; random1k user median PV={C['助手random1k'].search_num.median():.0f}")
P(f"  head PV concentration: search top1000 user top10 rows share {C['搜索top1000'].wise_pv.head(10).sum()/C['搜索top1000'].wise_pv.sum()*100:.1f}% ; assistant top1k user top10 {C['助手top1k'].search_num.sort_values(ascending=False).head(10).sum()/C['助手top1k'].search_num.sum()*100:.1f}%")
P("\n  assistant l2 mix (user rows): rows% / PV%")
for snap,k in (("top1k","助手top1k"),("random1k","助手random1k")):
    x=C[k]; g=x.groupby("l2").agg(rows=("query","size"),pv=("search_num","sum")); g["rows%"]=g.rows/len(x)*100; g["pv%"]=g.pv/g.pv.sum()*100
    P(f"  [{k}] n={len(x)}"); P(g.sort_values("rows",ascending=False).round(1).to_string())

# ---------- 2. markers ----------
src=open(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "sva_metrics.py"),encoding="utf-8").read(); exec(src[src.index("M={"):src.index("C={d:")])
FM={
 "疑问句(通用)": M["疑问句"],
 "第一人称": M["第一人称"],
 "纯数字代码": r"^\s*\d{3,6}\s*$",
 "以股票/股份/股价结尾": r"(股票|股份|股价)\s*$",
 "股吧": r"股吧",
 "时效词(今日/实时/最新/现在)": r"(今日|今天|实时|最新|现在|当前|目前)",
 "行情名词(金价/汇率/指数/行情/走势图)": r"(金价|黄金|白银|汇率|兑换|兑|行情|指数|大盘|走势图|[kK]线)",
 "客服电话/号码": r"(电话|客服|热线|人工|955\d\d)",
 "每日答题(蚂蚁庄园/新村)": r"(蚂蚁庄园|蚂蚁新村|庄园.*答案|今日答案)",
 "银行账户异常解读": r"(扣费|扣款|扣钱|扣了|扣\d|代扣|批扣|余额|冻结|只收不付|止付|信使费|管理费|年费|9920|4318|1072|9078)",
 "预测/前景/买卖判断": r"(未来|上涨空间|目标价|涨停|走势如何|走势怎么样|预测|能涨|会涨|会跌|还能涨|值得买|值得购买|值得入手|值得持有|是否值得|能买吗|可以买|还可以买|前景|首日能|首日大概|能赚多少|抄底|加仓|减仓|买入|卖出)",
 "具体日期(X月X日)": r"\d{1,2}月\d{1,2}(日|号)",
 "计算(含数字+利息/还款/手续费/收益/等于多少)": r"(?=.*\d)(?=.*(利息|还款|月供|分期|手续费|收益|利率|年化|等于多少|是多少人民币|多少人民币|多少钱))",
 "平台/机构可信度": r"(正规|靠谱|可靠|合法|是真的吗|真的吗|骗|诈骗|安全吗|跑路|套路|黑平台|合规)",
 "借贷/征信/信用卡": r"(贷款|借款|网贷|征信|逾期|信用卡|花呗|借呗|分期|额度|下款|催收|还款)",
 "保险": r"(保险|车险|寿险|医疗险|重疾|理赔|保单|投保|保费)",
 "家庭成员": M["家庭成员"],
 "导航/下载词": M["导航词"],
}
SURF4={"搜索top1000":C["搜索top1000"].assign(pv=C["搜索top1000"].wise_pv),"搜索top10k":C["搜索top10k"].assign(pv=C["搜索top10k"].wise_pv),
       "助手top1k":C["助手top1k"].assign(pv=C["助手top1k"].search_num),"助手random1k":C["助手random1k"].assign(pv=C["助手random1k"].search_num)}
rows=[]
P("\n#### 2. MARKERS — row% [Wilson95] ; PV% for heads  [filter: tier==user; search from clean_v3 top1000/top10k by wise_pv; assistant l1==金融 snapshot top1k/random1k]")
for k,x in SURF4.items():
    q=x["query"].astype(str); L=q.str.len()
    P(f"  {k}: n={len(x):,} 长度中位 {L.median():.0f} p90 {L.quantile(.9):.0f} >20字 {(L>20).mean()*100:.1f}%")
    rows.append(dict(surface=k,marker="长度中位",n=len(x),k=np.nan,share=L.median(),lo=np.nan,hi=np.nan,pv_share=np.nan))
for name,pat in FM.items():
    t=[]
    for k,x in SURF4.items():
        q=x["query"].astype(str); m=q.str.contains(pat,regex=True); kk=int(m.sum()); p,lo,hi=wilson(kk,len(x))
        pvs=float((x.pv*m).sum()/x.pv.sum()) if k!="助手random1k" else np.nan
        rows.append(dict(surface=k,marker=name,n=len(x),k=kk,share=p,lo=lo,hi=hi,pv_share=pvs))
        t.append(f"{k} {p*100:5.1f}[{lo*100:4.1f}–{hi*100:4.1f}]"+(f" PV{pvs*100:4.1f}" if k!="助手random1k" else "")+f" k={kk}")
    P(f"  {name}: "+" | ".join(t))
pd.DataFrame(rows).to_csv(f"{SP}/sva_dom_fin_markers.csv",index=False,encoding="utf-8-sig")
P("\n  marker examples (top-PV 4 + random 4 per surface)")
for name in ["纯数字代码","股吧","客服电话/号码","每日答题(蚂蚁庄园/新村)","银行账户异常解读","预测/前景/买卖判断","具体日期(X月X日)","计算(含数字+利息/还款/手续费/收益/等于多少)","平台/机构可信度","借贷/征信/信用卡","保险"]:
    P(f"  -- {name}")
    for k,x in SURF4.items():
        y=x[x["query"].astype(str).str.contains(FM[name],regex=True)]
        if len(y): P(f"     [{k}] "+" | ".join(list(y.sort_values("pv",ascending=False)["query"].astype(str).head(4).str[:30])+["~"]+list(y["query"].astype(str).sample(min(4,len(y)),random_state=7).str[:34])))

# ---------- 3. intent ----------
U=F[F.tier=="user"].dropna(subset=["u_ds"])
P("\n#### 3. UNIFIED INTENT u_ds  [filter: sva_final_rows domain==金融 tier==user; coverage below]")
for sf in SL: P(f"  coverage {sf}: {F[(F.surface==sf)&(F.tier=='user')].u_ds.notna().sum()}/{(F[(F.surface==sf)&(F.tier=='user')]).shape[0]}")
sh=[]
for view in ("all","interp"):
    for sf in SL:
        x=U[U.surface==sf]; x=x if view=="all" else x[x.u_ds!="U13"]; n=len(x); w=x.pv
        for c in CODES:
            if view=="interp" and c=="U13": continue
            m=x.u_ds==c; kk=int(m.sum()); p,lo,hi=wilson(kk,n)
            sh.append(dict(view=view,surface=sf,code=c,name=NAME[c],n=n,k=kk,share=p,lo=lo,hi=hi,pv_share=float((w*m).sum()/w.sum()) if sf!="assistant_random1k" else np.nan))
SH=pd.DataFrame(sh); SH.to_csv(f"{SP}/sva_dom_fin_intent_shares.csv",index=False,encoding="utf-8-sig")
for view in ("all","interp"):
    P(f"\n  view={view}: n "+" / ".join(f"{SL[sf]} {int(SH[(SH.view==view)&(SH.surface==sf)].n.iloc[0])}" for sf in SL))
    for c in CODES:
        r=SH[(SH.view==view)&(SH.code==c)]
        if not len(r): continue
        P(f"   {c} {NAME[c]:<8} "+" | ".join(f"{SL[z.surface]} {z.share*100:5.1f}[{z.lo*100:4.1f}–{z.hi*100:4.1f}] k={z.k}"+(f" PV{z.pv_share*100:4.1f}" if z.surface!='assistant_random1k' else '') for z in r.itertuples()))
dr=[]
for view in ("all","interp"):
    for s1,s2 in (("search_top1000","assistant_top1k"),("assistant_top1k","assistant_random1k")):
        for c in CODES:
            r1=SH[(SH.view==view)&(SH.surface==s1)&(SH.code==c)]; r2=SH[(SH.view==view)&(SH.surface==s2)&(SH.code==c)]
            if not len(r1): continue
            r1,r2=r1.iloc[0],r2.iloc[0]; d_,lo,hi=newcombe(r1.k,r1.n,r2.k,r2.n)
            dr.append(dict(view=view,frm=s1,to=s2,code=c,name=NAME[c],p_from=r1.share,p_to=r2.share,diff=d_,lo=lo,hi=hi,sig=bool(lo>0 or hi<0)))
DR=pd.DataFrame(dr); DR.to_csv(f"{SP}/sva_dom_fin_intent_diffs.csv",index=False,encoding="utf-8-sig")
for view in ("all","interp"):
    for s1,s2 in (("search_top1000","assistant_top1k"),("assistant_top1k","assistant_random1k")):
        x=DR[(DR.view==view)&(DR.frm==s1)&(DR.to==s2)]; x=x.reindex(x["diff"].abs().sort_values(ascending=False).index).head(8)
        P(f"  diff {view} {SL[s1]}→{SL[s2]}: "+" ; ".join(f"{r.code}{r.name} {r.p_from*100:.1f}→{r.p_to*100:.1f} ({r.diff*100:+.1f}[{r.lo*100:+.1f},{r.hi*100:+.1f}]{'*' if r.sig else ''})" for r in x.itertuples()))
P("\n  intent examples: top-PV 5 + random 6 per (code, surface)")
for c in CODES:
    for sf in SL:
        x=U[(U.surface==sf)&(U.u_ds==c)]
        if len(x): P(f"   {c}{NAME[c]} [{SL[sf]}] n={len(x)}: "+" | ".join(list(x.sort_values("pv",ascending=False)["query"].head(5).str[:30])+["~"]+list(x["query"].sample(min(6,len(x)),random_state=11).str[:34])))
P("\n  personal flag p_ds=1")
for sf in SL:
    x=F[(F.surface==sf)&(F.tier=="user")]; kk=int(x.p_ds.sum()); p,lo,hi=wilson(kk,len(x)); P(f"   {SL[sf]} {p*100:.1f}[{lo*100:.1f}–{hi*100:.1f}] k={kk}/{len(x)}  例: "+" | ".join(x[x.p_ds==1]["query"].sample(min(6,kk),random_state=3).str[:36]) if kk else f"   {SL[sf]} 0/{len(x)}")

# ---------- 3b. own taxonomy ----------
P("\n#### 3b. OWN TAXONOMY (own_intent) shares within surface  [filter: tier==user; search top10k from clean_v3 td_l1_name]")
OWN={"搜索top1000":C["搜索top1000"],"搜索top10k":C["搜索top10k"],"助手top1k":C["助手top1k"],"助手random1k":C["助手random1k"]}
orows=[]
for k,x in OWN.items():
    vc=x.td_l1_name.fillna("(无)").value_counts(); n=len(x)
    P(f"  [{k}] n={n} classes={len(vc)}")
    for nm,v in vc.items():
        p,lo,hi=wilson(int(v),n); y=x[x.td_l1_name==nm]
        orows.append(dict(surface=k,own_intent=nm,n=n,k=int(v),share=p,lo=lo,hi=hi))
        if v/n>=0.01: P(f"    {nm:<22} {p*100:5.1f}[{lo*100:4.1f}–{hi*100:4.1f}] ({v})  例: "+" | ".join(list(y.sort_values(y.columns[y.columns.isin(['wise_pv','search_num'])][0],ascending=False)["query"].astype(str).head(4).str[:22])+list(y["query"].astype(str).sample(min(3,len(y)),random_state=5).str[:26])))
pd.DataFrame(orows).to_csv(f"{SP}/sva_dom_fin_own_taxonomy.csv",index=False,encoding="utf-8-sig")
# cross-tab own class x u_ds, heads
P("\n  own_intent × u_ds (row counts; top classes)")
for sf in ("search_top1000","assistant_top1k","assistant_random1k"):
    x=U[U.surface==sf]; ct=pd.crosstab(x.own_intent,x.u_ds); ct=ct.loc[ct.sum(1).sort_values(ascending=False).index[:10]]
    P(f"  [{sf}]"); P(ct.to_string())
OUT.close(); print("ok")
