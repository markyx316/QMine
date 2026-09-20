# -*- coding: utf-8 -*-
"""金融 deep dive: assistant-only themes, pair evidence, label-boundary and template checks, persona markers."""
import sys, math, re, collections
SP=__import__("os").environ.get("SVA_WORKDIR", __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "work"))
sys.path.insert(0, SP)
from sva_common import *
import numpy as np
OUT=open(f"{SP}/sva_dom_fin_c.txt","w",encoding="utf-8")
def P(*x): print(*x,file=OUT,flush=True)
def wilson(k,n,z=1.96):
    if n==0: return float("nan"),float("nan"),float("nan")
    p=k/n; den=1+z*z/n; c=(p+z*z/(2*n))/den; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den; return p,max(0.0,c-h),min(1.0,c+h)
def W(k,n): p,lo,hi=wilson(k,n); return f"{p*100:.1f}%[{lo*100:.1f}–{hi*100:.1f}] ({k}/{n})"
fm=frame(); NAME={k:v[0] for k,v in fm.FRAME.items()}
D="金融"; a,s=load(); C=cells(a,s,D)
F=pd.read_parquet(f"{SP}/sva_final_rows.parquet"); F=F[(F.domain==D)&(F.tier=="user")].copy(); F["query"]=F["query"].astype(str)
A_=pd.read_parquet(f"{SP}/sva_dom_fin_nn.parquet")
S10=C["搜索top10k"].copy(); S10["query"]=S10["query"].astype(str); srank={q:i+1 for i,q in enumerate(S10["query"])}; spv=dict(zip(S10["query"],S10.wise_pv))
SH=F[F.surface=="search_top1000"]; AH=F[F.surface=="assistant_top1k"]; AT=F[F.surface=="assistant_random1k"]

P("#### C1. U02/U03 boundary on search '…股票' rows and combined U02+U03")
for nm,x in (("搜索1k",SH),("助手头",AH),("助手尾",AT)):
    y=x[x["query"].str.contains(r"股票\s*$")]; P(f"  {nm} rows ending 股票 n={len(y)}: "+" ; ".join(f"{k} {v}" for k,v in y.u_ds.value_counts().items()))
    kk=int(x.u_ds.isin(["U02","U03"]).sum()); P(f"   U02+U03 all rows {W(kk,len(x))} PV% {x[x.u_ds.isin(['U02','U03'])].pv.sum()/x.pv.sum()*100:.1f} ; interp {W(kk,int((x.u_ds!='U13').sum()))}")
    kk=int(x.u_ds.isin(["U04","U05","U06","U07","U08"]).sum()); P(f"   U04–U08 (解释/核实/办事/判断/清单) {W(kk,len(x))}")

P("\n#### C2. forecast template '未来有上涨空间吗' and pasted prompt '现在是2026年8月…' across ALL assistant categories and tiers")
for pat in (r"未来有上涨空间吗$", r"^现在是20\d\d年\d+月\d+日", r"上市首日能涨多少$", r"目标价是多少$"):
    x=a[a["query"].astype(str).str.contains(pat,regex=True)]
    P(f"  /{pat}/ rows {len(x)} ; by l1 {dict(x.l1.value_counts())} ; by snapshot {dict(x.snapshot.value_counts())} ; tiers {dict(x.tier.value_counts())}")
    for q,pv,snap,l2 in x.sort_values("search_num",ascending=False)[["query","search_num","snapshot","l2"]].head(14).values: P(f"     {pv:>5} {snap:<8} {l2} | {str(q)[:90]}")
    y=S10[S10["query"].str.contains(pat,regex=True)]; P(f"   in search top10k 金融: {len(y)}  "+" | ".join(y["query"].head(5)))

P("\n#### C3. 9920 / bank codes: all assistant rows (any category, any tier) and search 5 domains")
x=a[a["query"].astype(str).str.contains("9920")]
P(f"  assistant rows with 9920: {len(x)} ; by l1 {dict(x.l1.value_counts())}; tiers {dict(x.tier.value_counts())}; PV {x.search_num.sum()}")
for q,pv,snap,t in x.sort_values("search_num",ascending=False)[["query","search_num","snapshot","tier"]].values: P(f"     {pv:>5} {snap} {t} | {q}")
P(f"  search 5域 rows with 9920: {int(s['query'].astype(str).str.contains('9920').sum())}")

P("\n#### C4. persona / usage markers (row% [CI]; PV% heads)  [filter: tier==user]")
SURF4={"搜索top1000":C["搜索top1000"].assign(pv=C["搜索top1000"].wise_pv),"搜索top10k":C["搜索top10k"].assign(pv=C["搜索top10k"].wise_pv),"助手top1k":C["助手top1k"].assign(pv=C["助手top1k"].search_num),"助手random1k":C["助手random1k"].assign(pv=C["助手random1k"].search_num)}
PM={"交易术语(开盘/涨停/主力资金/仓位/短线…)": r"(开盘|高开|低开|收盘|涨停|跌停|主力资金|资金流入|资金流出|仓位|补仓|加仓|减仓|止损|做[Tt]|短线|长线|日内|均线|MACD|macd|量比|换手|夜盘|分时|回调|缩量|放量|筹码)",
    "新股申购/上市首日/中签": r"(申购|中签|上市首日|新股|打新|上市时间)",
    "公司事件/财报/重组/政策影响": r"(财报|业绩|中报|半年报|年报|季报|预告|重组|利好|利空|影响|分红|增持|减持|回购|退市|停牌|复牌|解禁)",
    "养老/退休/老人": r"(养老|退休|老人|老年)",
    "长句无标点(>20字)": r"^[^，,。？?！!、；;：:\s]{21,}$",
    "回指/依赖上文(这/那/他/你说/刚才/上面)": r"(^这|^那|这只|这张|这种|这些|他们?的|你说|刚才|上面|上述|以上)",
    "明确年份(20xx年)": r"20\d\d年",
}
rows=[]
for name,pat in PM.items():
    t=[]
    for k,x in SURF4.items():
        m=x["query"].astype(str).str.contains(pat,regex=True); kk=int(m.sum())
        p,lo,hi=wilson(kk,len(x)); pvs=float((x.pv*m).sum()/x.pv.sum())
        rows.append(dict(surface=k,marker=name,n=len(x),k=kk,share=p,lo=lo,hi=hi,pv_share=pvs if k!="助手random1k" else np.nan))
        t.append(f"{k} {p*100:.1f}[{lo*100:.1f}–{hi*100:.1f}]"+(f" PV{pvs*100:.1f}" if k!="助手random1k" else "")+f" k={kk}")
    P(f"  {name}: "+" | ".join(t))
    for k in ("助手top1k","助手random1k"):
        x=SURF4[k]; y=x[x["query"].astype(str).str.contains(pat,regex=True)]
        if len(y): P(f"     例[{k}] "+" | ".join(y["query"].astype(str).sample(min(6,len(y)),random_state=5).str[:34]))
pd.DataFrame(rows).to_csv(f"{SP}/sva_dom_fin_persona_markers.csv",index=False,encoding="utf-8-sig")

P("\n#### C5. exact vs non-exact head rows: question rate, length, p_ds")
Q=r"(怎么|如何|为什么|什么|哪个|哪家|哪些|多少|吗|呢|是不是|能不能|可不可以|\?|？)"
ex=AH["query"].isin(set(S10["query"]))
for nm,x in (("exact",AH[ex]),("non-exact",AH[~ex])):
    kk=int(x["query"].str.contains(Q).sum()); P(f"  {nm}: n={len(x)} question {W(kk,len(x))} ; len median {x['query'].str.len().median():.0f} ; p_ds=1 {W(int(x.p_ds.sum()),len(x))} ; PV share of head {x.pv.sum()/AH.pv.sum()*100:.1f}%")

P("\n#### C6. head U13 rows: how many are bank-notice fragments")
u13=AH[AH.u_ds=="U13"]; bank=u13["query"].str.contains(r"(银行|工行|农行|卡).*(余额|不够|不满|不足|冻结|扣)|余额|扣",regex=True)
P(f"  head U13 n={len(u13)} ; bank-notice-like {W(int(bank.sum()),len(u13))}: "+" | ".join(u13[bank]["query"]))
P("  other head U13: "+" | ".join(u13[~bank]["query"]))

P("\n#### C7. ASSISTANT-ONLY THEMES among far-band (sim<0.70) rows; priority order; also whole-cell share")
TH=[("银行卡/账户异常与扣费解读", r"(扣费|扣款|扣钱|扣了|扣\d|代扣|批扣|余额|冻结|只收不付|止付|解控|信使费|管理费|年费|到账|转账|提示\d|显示\s?\d|\d{4}代码|短信)"),
    ("个股/大盘/金价走势预测与买卖时机", r"(未来|上涨空间|目标价|涨停|跌停|走势|预测|能涨|会涨|会跌|还能涨|值得|能买|可以买|买入|卖出|抄底|加仓|减仓|补仓|仓位|前景|首日|潜力|入手|持有|高开|低开|开盘|做多|做空|回调|见顶|见底)"),
    ("公司事件、财报与政策影响解读", r"(财报|业绩|中报|半年报|年报|季报|预告|重组|利好|利空|影响|分红|增持|减持|回购|退市|停牌|复牌|解禁|政策|新规|融资|上市)"),
    ("借贷、征信、存款与利息计算", r"(贷款|借款|借钱|网贷|征信|逾期|信用卡|花呗|借呗|分期|额度|下款|催收|还款|月供|利息|利率|存款|定期|存单|理财|收益|余额宝|零钱通)"),
    ("保险条款、理赔与机构可信度", r"(保险|车险|寿险|理赔|保单|投保|保费|正规|靠谱|可靠|合法|骗|安全吗|跑路)"),
    ("依赖上文的追问与碎片", r"(^这|^那|这只|这张|这种|这些|他们?的|你说|刚才|上面|上述|以上|^[^一-鿿]*$)"),
]
trow=[]
for nm,x in (("assistant_top1k",A_[A_.surface=="assistant_top1k"]),("assistant_random1k",A_[A_.surface=="assistant_random1k"])):
    far=x[x.band=="远"].copy(); n=len(far); far["theme"]="其他"
    for th,pat in TH[::-1]:
        far.loc[far["query"].str.contains(pat,regex=True),"theme"]=th
    P(f"  [{nm}] cell n={len(x)} ; far n={n} ({W(n,len(x))}) ; far PV share {far.pv.sum()/x.pv.sum()*100:.1f}%")
    for th in [t for t,_ in TH]+["其他"]:
        y=far[far.theme==th]; kk=len(y)
        trow.append(dict(surface=nm,theme=th,far_n=n,k=kk,share_of_far=kk/max(n,1),share_of_cell=kk/len(x)))
        P(f"   {th}: {W(kk,n)} of far ; {kk/len(x)*100:.1f}% of cell ; u_ds {dict(y.u_ds.value_counts().head(4))} ; p=1 {int(y.p_ds.sum())}")
        if kk: P("      例: "+" | ".join(f"{q[:40]}(→{nq[:10]},{sm:.2f})" for q,nq,sm in y.sort_values("pv",ascending=False)[["query","nn_query","sim"]].head(3).values)+" ‖ "+" | ".join(f"{q[:44]}" for q in y["query"].sample(min(6,kk),random_state=9)))
pd.DataFrame(trow).to_csv(f"{SP}/sva_dom_fin_assistant_only_themes.csv",index=False,encoding="utf-8-sig")

P("\n#### C8. PAIR EVIDENCE — assistant rows with their nearest search neighbour (rank, PV, sim) and labels")
def show(q_list):
    for q in q_list:
        r=A_[A_["query"]==q]
        if not len(r): P(f"   !! not found: {q}"); continue
        r=r.iloc[0]; nq=r.nn_query
        if pd.isna(r.sim): P(f"   [{r.surface} PV{int(r.pv)} {r.u_ds} p{r.p_ds}] {q}  ⇐  (no neighbour)"); continue
        P(f"   [{r.surface.replace('assistant_','助手')} PV{int(r.pv)} {r.u_ds} p{r.p_ds}] {q}  ⇐  [搜索#{int(r.nn_search_rank)} PV{spv.get(nq,'?')} {r.u_nn}] {nq}  sim={r.sim:.2f}")
groups={
 "问法变成预测/买卖判断": ["黄金价格未来走势如何","上证指数未来走势如何","光迅科技股票值得买入吗","潍柴动力长线持有是否值得","包钢股份8月26日开盘后走势如何","现在买黄金合适吗","明日金价还会涨吗","农业银行股票最佳买入点"],
 "加入自己的账户情境/代码": ["工商银行9920只收不付","农业银行账户余额200可用余额0是什么","工商银行手机app显示4318","中国工商银行不够三百扣钱","工商银行转账显示成功却没有到账","农业银行卡自动扣80是扣什么"],
 "加入具体金额/期限→计算或配置": ["余额宝适合放多少零钱","现在换汇划算还是存美元","贷款45万，15年还清，每个月还多少钱？","花呗1万块钱分期12期,手续费利息多少钱","年利率11.95%的借款，一年利息是多少","100万存银行一年，哪个银行最划算"],
 "从行情到原因/核实": ["阳光电源股票为什么暴跌","星宇股份跌停原因","度小满是不是正规网贷","抖音放心借为什么审核不通过","金价还会继续涨吗"],
 "从名称到清单/组合": ["长江电力适合搭配哪些股票构建组合","A500最厉害三个ETF","美国股市三大股指","长鑫存储概念龙头股票"],
 "同一关键词的追问变体": ["蚂蚁庄园今日答案有哪些","蚂蚁新村今日答案还有哪些","蚂蚁庄园今日答案还有哪些","今日金价走势如何"],
}
for g,ql in groups.items(): P(f"  -- {g}"); show(ql)
P("\n  search rows for calculators/dates:")
for pat in (r"(计算器)", r"(利息|月供|还款)", r"(为什么.*跌|大跌|暴跌)", r"(正规|靠谱)"):
    y=S10[S10["query"].str.contains(pat)]; P(f"   /{pat}/ top10k n={len(y)} top1000 n={int((y.index.isin(S10.head(1000).index)).sum())}: "+" | ".join(f"{q}(#{srank[q]},{spv[q]})" for q in y["query"].head(8)))
OUT.close(); print("ok")
