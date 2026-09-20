# -*- coding: utf-8 -*-
"""Every number table in SEARCH_VS_ASSISTANT_2026.zh.md, generated from data (v3 tiers). Python-level regex throughout.
Label-independent tables T1–T11 here; intent tables are appended by sva_report_tables_intent.py."""
import sys, re, collections, numpy as np
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from sva_common import *
a,s=load(); DOM5=list(CAT)
OUT=open(f"{SP}/report_tables_v3.md","w",encoding="utf-8")
def W(x=""): print(x,file=OUT)
def pc(x,d=1): return "—" if x!=x else f"{x*100:.{d}f}%"
def has(series,pat): p=re.compile(pat); return series.astype(str).map(lambda x: bool(p.search(x))).values
def grab(path,start,end):
    t=open(path,encoding="utf-8").read(); ns={}; exec(t[t.index(start):t.index(end)],{},ns); return ns
M=grab(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "sva_metrics.py"),"M={","C={d:")["M"]
SUB=grab(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "sva_question_specificity.py"),"SUB={","C={d:"); SPEC=SUB["SPEC"]; SUB=SUB["SUB"]
CONV=grab(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "sva_conv_markers.py"),"CONV={","VERBS={")["CONV"]
MK=grab(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "sva_behaviour_v3.py"),"MK={","VOICE={")["MK"]
DELEG=r"^(请你?|帮我|帮忙|给我|麻烦|你给我|你帮我|能不能帮我|能否帮我|能帮我|可以帮我|写|画|生成|制作|分析|预测|推荐|总结|概括|解读|翻译|计算|对比|比较|列出|整理)"
SN={"搜索top1000":"搜索前1000","搜索top10k":"搜索前1万","助手top1k":"助手头部1k","助手random1k":"助手随机1k"}
CEL={d:cells(a,s,d) for d in DOM5}
def pooled(k): return pd.concat([CEL[d][k] for d in DOM5])
def group(g,k): return pooled(k) if g=="5域合计" else CEL[g][k]
PVC=lambda k: "wise_pv" if k.startswith("搜索") else "search_num"
GROUPS=DOM5+["5域合计"]

W("### T1 数据规模与流量截断点（全部行，含非 user 层）")
W("| 领域 | 搜索第1名 PV | 搜索第1000名 PV（user） | 搜索第1万名 PV（导出下限） | 搜索前1000 占前1万 PV | 搜索前1万 总PV | 助手头部1k 第1名 PV | 助手头部1k 第1000名 PV | 助手头部1k 中位 PV | 助手头部1k 总PV | 助手随机1k 中位 PV |")
W("|---|---|---|---|---|---|---|---|---|---|---|")
for d,cat in CAT.items():
    sd=s[s.domain==d].sort_values("wise_pv",ascending=False).reset_index(drop=True); su=sd[sd.tier=="user"].reset_index(drop=True)
    at=a[(a.l1==cat)&(a.snapshot=="top1k")].sort_values("search_num",ascending=False).reset_index(drop=True); ar=a[(a.l1==cat)&(a.snapshot=="random1k")]
    W(f"| {d} | {int(sd.wise_pv[0]):,} | {int(su.wise_pv[999]):,} | {int(sd.wise_pv.iloc[-1]):,} | {pc(su.wise_pv[:1000].sum()/sd.wise_pv.sum())} | {int(sd.wise_pv.sum()):,} | {int(at.search_num[0]):,} | {int(at.search_num.iloc[-1]):,} | {int(at.search_num.median()):,} | {int(at.search_num.sum()):,} | {ar.search_num.median():.0f} |")

TIERS=["user","S1_system_repeated","S2_system_template","S3_card_passage","S4_suggested_chip","S5_headline","C1_content_free","C2_feed_control"]
TN={"user":"用户输入(user)","S1_system_repeated":"S1 跨类目重复系统串","S2_system_template":"S2 模板/功能入口","S3_card_passage":"S3 百科卡片段落",
    "S4_suggested_chip":"S4 推荐追问","S5_headline":"S5 热点标题","C1_content_free":"C1 无内容","C2_feed_control":"C2 信息流反馈"}
five=a.l1.isin(list(CAT.values()))
SC=[("助手头部1k·5域",a[five&(a.snapshot=="top1k")],"search_num"),("助手随机1k·5域",a[five&(a.snapshot=="random1k")],"search_num"),
    ("助手头部1k·全33类",a[a.snapshot=="top1k"],"search_num"),("助手随机1k·全33类",a[a.snapshot=="random1k"],"search_num"),("搜索前1万·5域",s,"wise_pv")]
W("\n### T2 清洗分层（v3）：行数与流量")
W("| 层级 | "+" | ".join(f"{n} 行" for n,_,_ in SC)+" | "+" | ".join(f"{n} PV占比" for n,_,_ in SC)+" |")
W("|---|"+"---|"*(2*len(SC)))
for t in TIERS:
    W(f"| {TN[t]} | "+" | ".join(f"{int((df.tier==t).sum()):,} ({pc((df.tier==t).mean())})" for _,df,_ in SC)+" | "+" | ".join(pc(df[pv][df.tier==t].sum()/df[pv].sum()) for _,df,pv in SC)+" |")

W("\n### T3 各领域助手头部1k 的流量构成（PV 占比；括号内为行占比）")
W("| 领域 | "+" | ".join(TN[t] for t in TIERS)+" |"); W("|---|"+"---|"*len(TIERS))
for d,cat in list(CAT.items())+[("全33类",None)]:
    df=a[(a.snapshot=="top1k")&((a.l1==cat) if cat else True)]
    W(f"| {d} | "+" | ".join(f"{pc(df.search_num[df.tier==t].sum()/df.search_num.sum())} ({pc((df.tier==t).mean())})" for t in TIERS)+" |")

W("\n### T4 清洗对助手头部1k 指标的影响（全部行 → 仅 user 行；行占比 / PV 占比）")
KEYS=[("疑问句",M["疑问句"]),("祈使句(严格)",M["祈使句(严格)"]),("第一人称",M["第一人称"]),("资源词",M["资源词"])]
W("| 领域 | 行数 | 长度中位 | >60字 行占比 | "+" | ".join(f"{k} 行% | {k} PV%" for k,_ in KEYS)+" |"); W("|---|"+"---|"*(3+2*len(KEYS)))
for d,cat in CAT.items():
    al=a[(a.l1==cat)&(a.snapshot=="top1k")]; us=al[al.tier=="user"]; seg=[]
    for k,p in KEYS:
        ma,mu=has(al["query"],p),has(us["query"],p); wa,wu=al.search_num.values,us.search_num.values
        seg.append(f"{pc(ma.mean())}→{pc(mu.mean())} | {pc((wa*ma).sum()/wa.sum())}→{pc((wu*mu).sum()/wu.sum())}")
    W(f"| {d} | {len(al):,}→{len(us):,} | {al['query'].astype(str).str.len().median():.0f}→{us['query'].astype(str).str.len().median():.0f} | {pc((al['query'].astype(str).str.len()>60).mean())}→{pc((us['query'].astype(str).str.len()>60).mean())} | "+" | ".join(seg)+" |")

FORM=[("疑问句",M["疑问句"]),("是非核实",SUB["是非核实(是不是/是否/真的/吗)"]),("委托/动作开头",DELEG),("第一人称",M["第一人称"]),("称呼对方(你/您)",M["称呼对方(你/您)"]),
      ("回指上文",CONV["回指上文(上面/刚才/你说的/这道题…)"]),("句首承接",CONV["句首承接(那么/所以/还有/再…)"]),("输出约束",M["输出约束"]),
      ("带数量+单位",SPEC["带数量+单位"]),("带具体日期",SPEC["带具体日期"]),("具名二选一比较",SPEC["具名二选一比较"]),("资源词",M["资源词"]),("导航词",M["导航词"])]
W("\n### T5 查询形态（user 行；行占比）")
W("| 领域·切片 | n | 长度中位/p90 | "+" | ".join(k for k,_ in FORM)+" |"); W("|---|---|---|"+"---|"*len(FORM))
for g in GROUPS:
    for k in SURF:
        q=group(g,k)["query"].astype(str); ln=q.str.len()
        W(f"| {g}·{SN[k]} | {len(q):,} | {ln.median():.0f}/{ln.quantile(.9):.0f} | "+" | ".join(pc(has(q,p).mean()) for _,p in FORM)+" |")
W("\n### T5b 头部切片的 PV 加权占比（user 行）")
W("| 领域·切片 | "+" | ".join(k for k,_ in FORM[:3]+[FORM[3],FORM[11],FORM[12]])+" |"); W("|---|"+"---|"*6)
for g in GROUPS:
    for k in ("搜索top1000","搜索top10k","助手top1k"):
        df=group(g,k); q=df["query"].astype(str); w=df[PVC(k)].astype(float).values
        W(f"| {g}·{SN[k]} | "+" | ".join(pc((w*has(q,p)).sum()/w.sum()) for _,p in FORM[:3]+[FORM[3],FORM[11],FORM[12]])+" |")

W("\n### T6 问句内部的类型构成（5域合计，user 行；“行%”=占全部行，“问句内%”=占含疑问标记的行）")
W("| 问句类型 | "+" | ".join(f"{SN[k]} 行% | {SN[k]} 问句内%" for k in SURF)+" |"); W("|---|"+"---|"*(2*len(SURF)))
Qall={k:pooled(k)["query"].astype(str) for k in SURF}; Qm={k:has(Qall[k],M["疑问句"]) for k in SURF}
W("| 含疑问标记（合计） | "+" | ".join(f"{pc(Qm[k].mean())} | 100%" for k in SURF)+" |")
for name,p in SUB.items():
    W(f"| {name} | "+" | ".join(f"{pc(has(Qall[k],p).mean())} | {pc((has(Qall[k],p)&Qm[k]).sum()/max(Qm[k].sum(),1))}" for k in SURF)+" |")

W("\n### T7 搜索内部的深度对照（清洗后 user 行，按 PV 排名分段：长度中位 / 疑问句% / 第一人称% / 委托%）")
W("| 领域 | 第1–1000名 | 第1001–3000名 | 第3001–6000名 | 第6001名以后 | 助手头部1k | 助手随机1k |"); W("|---|---|---|---|---|---|---|")
def fp(q): return f"{q.str.len().median():.0f} / {pc(has(q,M['疑问句']).mean())} / {pc(has(q,M['第一人称']).mean())} / {pc(has(q,DELEG).mean())}"
for d in DOM5:
    q=CEL[d]["搜索top10k"]["query"].astype(str).reset_index(drop=True)
    W(f"| {d} | "+" | ".join(fp(q.iloc[lo:hi]) for lo,hi in [(0,1000),(1000,3000),(3000,6000),(6000,len(q))])+f" | {fp(CEL[d]['助手top1k']['query'].astype(str))} | {fp(CEL[d]['助手random1k']['query'].astype(str))} |")

W("\n### T8 原样重合与“包装”（user 行）")
W("| 领域 | 助手头部1k 原样出现在搜索前1万：行% / PV% | 助手随机1k 原样重合 行% | 搜索前1000 原样出现在助手（任一切片） | 助手头部1k 包装了搜索头部查询 行% | 助手随机1k 包装 行% |"); W("|---|---|---|---|---|---|")
tot=collections.Counter()
for d in DOM5:
    c=CEL[d]; su=c["搜索top10k"]; S=set(su["query"].astype(str)); top1000=su.head(1000)["query"].astype(str).tolist()
    r={}
    for k in ("助手top1k","助手random1k"):
        x=c[k]["query"].astype(str); m=x.isin(S).values; pv=c[k].search_num.values
        r[k]=(m.mean(),pv[m].sum()/pv.sum()); tot[k+"n"]+=len(x); tot[k+"e"]+=m.sum()
    au=set(pd.concat([c["助手top1k"]["query"],c["助手random1k"]["query"]]).astype(str)); rev=sum(q in au for q in top1000); tot["rev"]+=rev
    cores=sorted([q for q in top1000 if len(q)>=3 and not q.isdigit()],key=len,reverse=True); wr={}
    for k in ("助手top1k","助手random1k"):
        n=0
        for q in c[k]["query"].astype(str):
            if any(core in q and len(q)>=len(core)+2 for core in cores): n+=1
        wr[k]=n/len(c[k]); tot[k+"w"]+=n
    W(f"| {d} | {pc(r['助手top1k'][0])} / {pc(r['助手top1k'][1])} | {pc(r['助手random1k'][0])} | {rev}/1000 | {pc(wr['助手top1k'])} | {pc(wr['助手random1k'])} |")
W(f"| 5域合计 | {pc(tot['助手top1ke']/tot['助手top1kn'])} / — | {pc(tot['助手random1ke']/tot['助手random1kn'])} | {tot['rev']}/5000 | {pc(tot['助手top1kw']/tot['助手top1kn'])} | {pc(tot['助手random1kw']/tot['助手random1kn'])} |")

nn=pd.read_parquet(f"{SP}/sva_semantic_nn_v3.parquet"); ct=pd.read_parquet(f"{SP}/sva_semantic_nn_controls_v3.parquet")
W("\n### T9 语义近邻：助手查询在搜索前1万里的最近邻（bge-base-zh-v1.5 余弦；中位数 / 找不到近邻(<0.70)的比例）")
W("| 领域 | 对照A 搜索前1000→其余搜索 | 对照B 搜索最深1000→其余搜索 | 助手头部1k | 助手随机1k | 仅≤6字：对照B vs 助手随机1k | 仅7–10字：对照B vs 助手随机1k |"); W("|---|---|---|---|---|---|---|")
def ms(x): return f"{np.median(x):.2f} / {pc((x<0.70).mean())}"
def lenb(df,lo,hi): L=df["query"].astype(str).str.len(); return df.sim.values[((L>=lo)&(L<=hi)).values]
for d in DOM5:
    cA,cB=ct[(ct.domain==d)&(ct.kind=="对照A")],ct[(ct.domain==d)&(ct.kind=="对照B")]; h,t_=nn[(nn.domain==d)&(nn.surface=="助手top1k")],nn[(nn.domain==d)&(nn.surface=="助手random1k")]
    b6,t6=lenb(cB,0,6),lenb(t_,0,6); b10,t10=lenb(cB,7,10),lenb(t_,7,10)
    W(f"| {d} | {ms(cA.sim.values)} | {ms(cB.sim.values)} | {ms(h.sim.values)} | {ms(t_.sim.values)} | {pc((b6<0.7).mean())} (n={len(b6)}) vs {pc((t6<0.7).mean())} (n={len(t6)}) | {pc((b10<0.7).mean())} (n={len(b10)}) vs {pc((t10<0.7).mean())} (n={len(t10)}) |")

W("\n### T10 行为标记（user 行；行占比）")
W("| 标记 | "+" | ".join(f"5域·{SN[k]}" for k in SURF)+" | 助手头部1k·全33类 | 助手随机1k·全33类 | 最集中的领域（助手随机1k） |"); W("|---|"+"---|"*(len(SURF)+3))
U33={snap:a[(a.tier=="user")&(a.snapshot==snap)]["query"].astype(str) for snap in ("top1k","random1k")}
for name,p in MK.items():
    best=max(DOM5,key=lambda d: has(CEL[d]["助手random1k"]["query"],p).mean())
    W(f"| {name} | "+" | ".join(pc(has(pooled(k)['query'],p).mean(),2) for k in SURF)+f" | {pc(has(U33['top1k'],p).mean(),2)} | {pc(has(U33['random1k'],p).mean(),2)} | {best} {pc(has(CEL[best]['助手random1k']['query'],p).mean(),2)} |")
long_np=lambda q: q.astype(str).map(lambda x: len(x)>20 and not re.search(r"[，,。？?！!、；;：:\s]",x)).values
W(f"| 长句无标点(>20字，语音输入迹象) | "+" | ".join(pc(long_np(pooled(k)['query']).mean(),2) for k in SURF)+f" | {pc(long_np(U33['top1k']).mean(),2)} | {pc(long_np(U33['random1k']).mean(),2)} | — |")
W(f"| 粘贴材料(>60字) | "+" | ".join(pc((pooled(k)['query'].astype(str).str.len()>60).mean(),2) for k in SURF)+f" | {pc((U33['top1k'].str.len()>60).mean(),2)} | {pc((U33['random1k'].str.len()>60).mean(),2)} | — |")

W("\n### T11 流量集中度（user 行）：前10条占PV / 前100条占PV / 有效查询数 1/Σp²")
W("| 领域 | 搜索前1000 | 搜索前1万 | 助手头部1k |"); W("|---|---|---|---|")
for d in DOM5:
    t=[]
    for k in ("搜索top1000","搜索top10k","助手top1k"):
        w=np.sort(CEL[d][k][PVC(k)].astype(float).values)[::-1]; p=w/w.sum(); t.append(f"{pc(p[:10].sum())} / {pc(p[:100].sum())} / {1/(p**2).sum():,.0f}")
    W(f"| {d} | "+" | ".join(t)+" |")
CUES=grab(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "sva_persona_cues.py"),"CUES={","C={d:")["CUES"]
W("\n### T21 “谁在问”的线索词（user 行；行占比；最后两列分别是助手头部1k、助手随机1k 中占比最高的领域）")
W("| 线索 | "+" | ".join(f"5域·{SN[k]}" for k in SURF)+" | 最集中的领域（助手头部1k） | 最集中的领域（助手随机1k） |"); W("|---|"+"---|"*(len(SURF)+2))
for name,p in CUES.items():
    bh=max(DOM5,key=lambda d: has(CEL[d]["助手top1k"]["query"],p).mean()); bt=max(DOM5,key=lambda d: has(CEL[d]["助手random1k"]["query"],p).mean())
    W(f"| {name} | "+" | ".join(pc(has(pooled(k)['query'],p).mean(),2) for k in SURF)+f" | {bh} {pc(has(CEL[bh]['助手top1k']['query'],p).mean(),2)} | {bt} {pc(has(CEL[bt]['助手random1k']['query'],p).mean(),2)} |")
OUT.close(); print("ok")
