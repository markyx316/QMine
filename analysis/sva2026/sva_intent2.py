# -*- coding: utf-8 -*-
"""Intent comparison v2: four surfaces (search top1000, search top10k, assistant top1k, assistant random1k);
shares on all user rows AND on interpretable rows (U13 excluded); differences for both search lenses;
reliability; personal flag; crosswalk; phrasing within intent; semantic distance; wrap and near-paraphrase
transitions; and the search→assistant ladder."""
import os, sys, math, numpy as np
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from sva_common import *
F=pd.read_parquet(os.environ.get("FINAL_OUT",f"{SP}/sva_final_rows2.parquet")); F["query"]=F["query"].astype(str)
PFX=os.environ.get("INTENT_PREFIX",f"{SP}/sva_intent2")
fm=frame(); CODES=list(fm.FRAME); NAME={k:v[0] for k,v in fm.FRAME.items()}; CW=fm.CROSSWALK
SURFS=[x for x in ["search_top1000","search_top10k","assistant_top1k","assistant_random1k"] if x in set(F.surface)]
SL={"search_top1000":"搜索1k","search_top10k":"搜索10k","assistant_top1k":"助手头","assistant_random1k":"助手尾"}
HEADS={"search_top1000","search_top10k","assistant_top1k"}
src=open(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "sva_metrics.py"),encoding="utf-8").read(); exec(src[src.index("M={"):src.index("C={d:")])
src2=open(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "sva_conv_markers.py"),encoding="utf-8").read(); exec(src2[src2.index("CONV={"):src2.index("VERBS={")])
DELEG=r"^(请你?|帮我|帮忙|给我|麻烦|你给我|你帮我|能不能帮我|能否帮我|能帮我|可以帮我|写|画|生成|制作|分析|预测|推荐|总结|概括|解读|翻译|计算|对比|比较|列出|整理)"
U=F[F.tier=="user"].copy(); DOMS=list(CAT)+["合计"]
out=open(PFX+".txt","w",encoding="utf-8")
def P(*x): print(*x,file=out,flush=True)
def wilson(k,n,z=1.96):
    if n==0: return float("nan"),float("nan"),float("nan")
    p=k/n; den=1+z*z/n; c=(p+z*z/(2*n))/den; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den; return p,max(0.0,c-h),min(1.0,c+h)
def newcombe(k1,n1,k2,n2):
    p1,l1,u1=wilson(k1,n1); p2,l2,u2=wilson(k2,n2); d=p2-p1
    return d, d-math.sqrt((p2-l2)**2+(u1-p1)**2), d+math.sqrt((u2-p2)**2+(p1-l1)**2)
def kappa(a,b):
    a=np.asarray(a); b=np.asarray(b); n=len(a)
    if n==0: return float("nan"),float("nan"),0
    po=(a==b).mean(); pe=sum((a==c).mean()*(b==c).mean() for c in set(a)|set(b))
    return ((po-pe)/(1-pe) if pe<1 else float("nan")), po, n
def cell(d,sf,df=None):
    df=U if df is None else df; x=df[df.surface==sf]; return x if d=="合计" else x[x.domain==d]
P("#### A. 覆盖率 (user 行; ds=deepseek 标签, qw=qwen 标签)")
for d in DOMS: P(f"  {d:<4} "+" | ".join(f"{SL[sf]} n={len(cell(d,sf)):,} ds={cell(d,sf).u_ds.notna().sum():,} qw={cell(d,sf).u_qw.notna().sum():,}" for sf in SURFS))
P("\n#### B. 标注可靠性 deepseek vs qwen")
R=U.dropna(subset=["u_ds","u_qw"]).drop_duplicates(["domain","surface","query"])
for nm,x in [("全体(去重 query)",R.drop_duplicates("query"))]+[(SL[sf],R[R.surface==sf]) for sf in SURFS]+[(d,R[R.domain==d].drop_duplicates("query")) for d in CAT]:
    k,po,n=kappa(x.u_ds,x.u_qw); P(f"  {nm}: n={n:,} 一致率 {po:.3f} κ {k:.3f}")
Rq=R.drop_duplicates("query")
P("  按类 (去重 query): n_ds / n_qw / 同判 / P(qw同|ds) / P(ds同|qw)")
for c in CODES:
    nd=(Rq.u_ds==c).sum(); nq=(Rq.u_qw==c).sum(); ag=((Rq.u_ds==c)&(Rq.u_qw==c)).sum()
    P(f"   {c} {NAME[c]:<9} {nd:>5} {nq:>5} {ag:>5}  {ag/max(nd,1):.2f}  {ag/max(nq,1):.2f}")
P("  最常见分歧 (ds→qw):")
for (x1,x2),v in Rq[Rq.u_ds!=Rq.u_qw].groupby(["u_ds","u_qw"]).size().sort_values(ascending=False).head(12).items():
    P(f"   {x1}{NAME[x1]}→{x2}{NAME[x2]} {v}  例: "+" | ".join(Rq[(Rq.u_ds==x1)&(Rq.u_qw==x2)]["query"].head(3).str[:24]))
Rp=U.dropna(subset=["p_ds","p_qw"]).drop_duplicates("query")
k,po,n=kappa(Rp.p_ds.astype(int),Rp.p_qw.astype(int)); P(f"  p 字段: n={n:,} 一致率 {po:.3f} κ {k:.3f}")
def shares(df_filter,tag):
    rows=[]
    for d in DOMS:
        for sf in SURFS:
            x=df_filter(cell(d,sf)).dropna(subset=["u_ds"]); n=len(x); w=x.pv.astype(float)
            for c in CODES:
                m=(x.u_ds==c); kk=int(m.sum()); p,lo,hi=wilson(kk,n)
                rows.append(dict(view=tag,domain=d,surface=sf,code=c,name=NAME[c],n=n,k=kk,share=p,lo=lo,hi=hi,
                    pv_share=(float((w*m).sum()/w.sum()) if sf in HEADS and w.sum()>0 else float("nan"))))
    return pd.DataFrame(rows)
SH=pd.concat([shares(lambda x:x,"all"),shares(lambda x:x[x.u_ds!="U13"],"interpretable")]); SH.to_csv(PFX+"_shares.csv",index=False,encoding="utf-8-sig")
def sh(view,d,sf,c): return SH[(SH.view==view)&(SH.domain==d)&(SH.surface==sf)&(SH.code==c)].iloc[0]
for view,title in (("all","C. 意图占比 — 全部 user 行"),("interpretable","C2. 意图占比 — 可判定行 (去掉 U13)")):
    P(f"\n#### {title} (行%, Wilson 95%CI; 头部另附 PV 加权%)")
    for d in DOMS:
        P(f"\n  == {d} ==  n: "+" / ".join(f"{SL[sf]} {int(sh(view,d,sf,'U01').n):,}" for sf in SURFS))
        for c in CODES:
            if view=="interpretable" and c=="U13": continue
            t=[]
            for sf in SURFS:
                r=sh(view,d,sf,c); x_=f"{r.share*100:5.1f}[{r.lo*100:4.1f}–{r.hi*100:4.1f}]"
                if sf in HEADS: x_+=f"PV{r.pv_share*100:4.1f}"
                t.append(f"{SL[sf]} {x_}")
            P(f"   {c} {NAME[c]:<9} "+" | ".join(t))
P("\n#### D. 差异 (百分点, Newcombe 95%CI; * = CI 不含 0) — 视图: all 与 interpretable")
PAIRS=[p for p in (("search_top1000","assistant_top1k"),("search_top10k","assistant_top1k"),("assistant_top1k","assistant_random1k"),("search_top10k","assistant_random1k")) if p[0] in SURFS and p[1] in SURFS]
dr=[]
for view in ("all","interpretable"):
    for d in DOMS:
        for s1,s2 in PAIRS:
            for c in CODES:
                r1,r2=sh(view,d,s1,c),sh(view,d,s2,c); dd,lo,hi=newcombe(r1.k,r1.n,r2.k,r2.n)
                dr.append(dict(view=view,domain=d,frm=s1,to=s2,code=c,name=NAME[c],p_from=r1.share,p_to=r2.share,diff=dd,lo=lo,hi=hi,sig=bool(lo>0 or hi<0)))
DF=pd.DataFrame(dr); DF.to_csv(PFX+"_diffs.csv",index=False,encoding="utf-8-sig")
for view in ("all","interpretable"):
    P(f"\n  ---- view={view} ----")
    for d in DOMS:
        P(f"  == {d} ==")
        for s1,s2 in PAIRS:
            x=DF[(DF.view==view)&(DF.domain==d)&(DF.frm==s1)&(DF.to==s2)&~((DF.view=="interpretable")&(DF.code=="U13"))].copy()
            x=x.reindex(x["diff"].abs().sort_values(ascending=False).index).head(7)
            P(f"   {SL[s1]}→{SL[s2]}: "+" ; ".join(f"{r.code}{r.name} {r.p_from*100:.1f}→{r.p_to*100:.1f} ({r.diff*100:+.1f}[{r.lo*100:+.1f},{r.hi*100:+.1f}]{'*' if r.sig else ''})" for r in x.itertuples()))
P("\n#### E. 稳健性: qwen 覆盖子集上, 两模型各自计算差异的方向 (|ds 差异|≥3pp)")
Q_=U.dropna(subset=["u_ds","u_qw"])
for s1,s2 in PAIRS[:2]:
    tot=agr=0; lines=[]
    for d in CAT:
        a1,a2=cell(d,s1,Q_),cell(d,s2,Q_)
        if len(a1)<30 or len(a2)<30: continue
        for c in CODES:
            vd=(a2.u_ds==c).mean()-(a1.u_ds==c).mean(); vq=(a2.u_qw==c).mean()-(a1.u_qw==c).mean()
            if abs(vd)>=0.03:
                tot+=1; same=bool(np.sign(vd)==np.sign(vq)); agr+=same; lines.append(f"   {d} {c}{NAME[c]} (n {len(a1)}/{len(a2)}) ds {vd*100:+.1f} qw {vq*100:+.1f} {'✓' if same else '✗'}")
    P(f"  {SL[s1]}→{SL[s2]}: {tot} 个; qwen 同向 {agr} ({agr/max(tot,1)*100:.0f}%)"); [P(l) for l in lines]
P("\n#### F. 个人处境 p=1 占比 (deepseek; Wilson CI)")
for d in DOMS:
    t=[]
    for sf in SURFS:
        x=cell(d,sf).dropna(subset=["p_ds"]); kk=int(x.p_ds.sum()); p,lo,hi=wilson(kk,len(x)); t.append(f"{SL[sf]} {p*100:.1f}%[{lo*100:.1f}–{hi*100:.1f}](k={kk})")
    P(f"  {d:<4} "+" | ".join(t))
P("\n#### G. 各自 taxonomy 经 crosswalk 映射 vs 统一标注 (同一行)")
U["cw"]=[fm.frame_code(r,c) if (r,c) in CW else None for r,c in zip(U.own_run,U.own_code)]
for sf in SURFS:
    x=U[U.surface==sf].dropna(subset=["u_ds","cw"])
    P(f"  {SL[sf]}: n={len(x):,} 一致 {(x.cw==x.u_ds).mean()*100:.1f}% | 5域合计占比 crosswalk/统一: "+" ; ".join(f"{c} {(x.cw==c).mean()*100:.1f}/{(x.u_ds==c).mean()*100:.1f}" for c in CODES))
P("\n#### H. 同一意图, 不同问法 (5域合计, user 行): n / 长度中位 / 疑问% / 第一人称% / 委托% / p=1%")
for c in CODES:
    t=[]; ok=0
    for sf in SURFS:
        x=U[(U.surface==sf)&(U.u_ds==c)]; ok+=len(x)>=20
        if len(x): t.append(f"{SL[sf]} n={len(x)} 长{x['query'].str.len().median():.0f} 问{x['query'].str.contains(M['疑问句']).mean()*100:.0f}% 我{x['query'].str.contains(M['第一人称']).mean()*100:.0f}% 托{x['query'].str.contains(DELEG).mean()*100:.0f}% p{x.p_ds.mean()*100:.0f}%")
    if ok>=2:
        P(f"  {c} {NAME[c]}: "+" | ".join(t))
        for sf in SURFS:
            x=U[(U.surface==sf)&(U.u_ds==c)]
            if len(x): P(f"     例[{SL[sf]}] "+" | ".join(list(x.sort_values("pv",ascending=False)["query"].head(3).str[:22])+list(x["query"].sample(min(3,len(x)),random_state=1).str[:22])))
with open(PFX+"_examples.txt","w",encoding="utf-8") as ex:
    for d in CAT:
        print(f"\n==== {d} ====",file=ex)
        for c in CODES:
            for sf in SURFS:
                x=cell(d,sf).dropna(subset=["u_ds"]); x=x[x.u_ds==c]
                if len(x): print(f"  {c}{NAME[c]} [{SL[sf]}] n={len(x)}: "+" | ".join(list(x.sort_values("pv",ascending=False)["query"].head(4).str[:30])+list(x["query"].sample(min(4,len(x)),random_state=2).str[:30])),file=ex)
NN=pd.read_parquet(os.environ.get("NN_PATH",f"{SP}/sva_semantic_nn_v3.parquet")); NN["surface"]=NN.surface.map({"助手top1k":"assistant_top1k","助手random1k":"assistant_random1k"})
NN=NN.drop_duplicates(["domain","surface","query"])
A_=U[U.surface.isin(["assistant_top1k","assistant_random1k"])].merge(NN[["domain","surface","query","sim","nn_query","nn_search_rank"]],on=["domain","surface","query"],how="left")
BAND=lambda v: None if pd.isna(v) else ("完全相同" if v>=0.999 else ("近似" if v>=0.80 else ("中等" if v>=0.70 else "远")))
A_["band"]=A_.sim.map(BAND)
P(f"\n#### J. 意图 × 与搜索的语义距离 (助手 user 行; 最近搜索 top10k 邻居; 匹配率 {A_.sim.notna().mean()*100:.1f}%) — 按 远% 排序")
for sf in ("assistant_top1k","assistant_random1k"):
    x=A_[A_.surface==sf].dropna(subset=["u_ds","band"]); g=[]
    for c in CODES:
        y=x[x.u_ds==c]
        if len(y)>=15: g.append(((y.band=="远").mean(),c,len(y),(y.band=="完全相同").mean(),(y.band=="近似").mean(),(y.band=="中等").mean()))
    P(f"  {SL[sf]} (n={len(x):,}): 全体 同{(x.band=='完全相同').mean()*100:.1f}% 近{(x.band=='近似').mean()*100:.1f}% 中{(x.band=='中等').mean()*100:.1f}% 远{(x.band=='远').mean()*100:.1f}%")
    for far,c,n,e_,near,mid in sorted(g): P(f"   {c} {NAME[c]:<9} n={n:>4} 同{e_*100:5.1f}% 近{near*100:5.1f}% 中{mid*100:5.1f}% 远{far*100:5.1f}%")
P("\n#### K. 包装转移: 搜索头部 (top1000) 查询被原样包含在更长的助手查询里 → 意图如何变")
lab_core=U[U.surface=="search_top1000"].dropna(subset=["u_ds"]).drop_duplicates(["domain","query"]).set_index(["domain","query"]).u_ds.to_dict()
pairs=[]
for d in CAT:
    cores=sorted([q for q in U[(U.surface=="search_top1000")&(U.domain==d)]["query"].unique() if len(q)>=3 and not q.isdigit()],key=len,reverse=True)
    for sf in ("assistant_top1k","assistant_random1k"):
        for q,u in U[(U.surface==sf)&(U.domain==d)][["query","u_ds"]].itertuples(index=False):
            if pd.isna(u): continue
            for core in cores:
                if core in q and len(q)>=len(core)+2:
                    if (d,core) in lab_core: pairs.append(dict(domain=d,surface=sf,core=core,query=q,u_core=lab_core[(d,core)],u_wrap=u))
                    break
PW=pd.DataFrame(pairs,columns=["domain","surface","core","query","u_core","u_wrap"]); PW.to_csv(PFX+"_wrap_pairs.csv",index=False,encoding="utf-8-sig")
for sf in ("assistant_top1k","assistant_random1k"):
    x=PW[PW.surface==sf]
    if not len(x): continue
    P(f"  {SL[sf]}: n={len(x)} 意图不变 {(x.u_core==x.u_wrap).mean()*100:.1f}%")
    for (c1,c2),v in x[x.u_core!=x.u_wrap].groupby(["u_core","u_wrap"]).size().sort_values(ascending=False).head(10).items():
        e=x[(x.u_core==c1)&(x.u_wrap==c2)].sample(min(3,v),random_state=4)
        P(f"   {c1}{NAME[c1]}→{c2}{NAME[c2]} {v} ({v/len(x)*100:.1f}%)  例: "+" | ".join(f"[{a_}]→{b_[:26]}" for a_,b_ in zip(e.core,e["query"])))
lm=F.dropna(subset=["u_ds"]).drop_duplicates("query").set_index("query").u_ds.to_dict()
A_["u_nn"]=A_.nn_query.map(lm)
for lo_,hi_,title in ((0.80,0.999,"近似改写 0.80≤sim<0.999"),(0.70,0.80,"中等相似 0.70≤sim<0.80")):
    x=A_[(A_.sim>=lo_)&(A_.sim<hi_)].dropna(subset=["u_ds"])
    P(f"\n#### L. {title}: 助手查询 vs 其最近搜索查询的意图转移 (搜索邻居有标签 {x.u_nn.notna().mean()*100:.1f}%)")
    x=x.dropna(subset=["u_nn"])
    for sf in ("assistant_top1k","assistant_random1k"):
        y=x[x.surface==sf]
        if not len(y): continue
        P(f"  {SL[sf]}: n={len(y)} 意图不变 {(y.u_nn==y.u_ds).mean()*100:.1f}%")
        for (c1,c2),v in y[y.u_nn!=y.u_ds].groupby(["u_nn","u_ds"]).size().sort_values(ascending=False).head(10).items():
            e=y[(y.u_nn==c1)&(y.u_ds==c2)].sample(min(3,v),random_state=4)
            P(f"   {c1}{NAME[c1]}→{c2}{NAME[c2]} {v} ({v/len(y)*100:.1f}%)  例: "+" | ".join(f"[{a_[:16]}]→{b_[:24]}" for a_,b_ in zip(e.nn_query,e["query"])))
# M. ladder
A_["conv"]=False
for v in CONV.values(): A_["conv"]=A_["conv"] | A_["query"].str.contains(v,regex=True)
LV=["F 对话/委托","E 个人处境/个案判断","A 原样搜索词","B 近似改写(≥0.80)","C 同题延伸(0.70–0.80)","D 搜索无近邻(<0.70)"]
def level(r):
    if r.u_ds in ("U10","U11") or r.conv: return LV[0]
    if r.p_ds==1 or r.u_ds=="U07": return LV[1]
    if pd.isna(r.sim): return None
    return LV[2] if r.sim>=0.999 else LV[3] if r.sim>=0.80 else LV[4] if r.sim>=0.70 else LV[5]
A_["level"]=[level(r) for r in A_.itertuples()]
A_.to_parquet(PFX+"_assistant_nn.parquet",index=False)
P("\n#### M. 从搜索到助手的阶梯 (助手 user 行; 优先级 F>E>A>B>C>D; 行% [CI]; 头部附 PV%)")
for d in DOMS:
    P(f"  == {d} ==")
    for sf in ("assistant_top1k","assistant_random1k"):
        x=A_[(A_.surface==sf)&((A_.domain==d) if d!="合计" else True)].dropna(subset=["level"]); n=len(x); w=x.pv.astype(float)
        P(f"   {SL[sf]} n={n}: "+" | ".join(f"{lv} {wilson(int((x.level==lv).sum()),n)[0]*100:.1f}%[{wilson(int((x.level==lv).sum()),n)[1]*100:.1f}–{wilson(int((x.level==lv).sum()),n)[2]*100:.1f}]"+(f" PV{(w*(x.level==lv)).sum()/w.sum()*100:.1f}" if sf=="assistant_top1k" else "") for lv in LV))
    for sf in ("search_top1000","search_top10k"):
        if sf not in SURFS: continue
        x=cell(d,sf).dropna(subset=["u_ds"]).copy(); x["conv"]=False
        for v in CONV.values(): x["conv"]=x["conv"] | x["query"].str.contains(v,regex=True)
        f_=(x.u_ds.isin(["U10","U11"])|x.conv).mean(); e_=((~(x.u_ds.isin(["U10","U11"])|x.conv))&((x.p_ds==1)|(x.u_ds=="U07"))).mean()
        P(f"   {SL[sf]} n={len(x)} (对照): {LV[0]} {f_*100:.1f}% | {LV[1]} {e_*100:.1f}%")
with open(PFX+"_ladder_examples.txt","w",encoding="utf-8") as ex:
    for d in CAT:
        print(f"\n==== {d} ====",file=ex)
        for sf in ("assistant_top1k","assistant_random1k"):
            for lv in LV:
                x=A_[(A_.surface==sf)&(A_.domain==d)&(A_.level==lv)]
                if len(x): print(f"  [{SL[sf]}] {lv} n={len(x)}: "+" | ".join(f"{q[:28]}→({nq[:14]},{s_:.2f})" for q,nq,s_ in zip(x.sample(min(6,len(x)),random_state=8)["query"],x.sample(min(6,len(x)),random_state=8).nn_query.fillna(""),x.sample(min(6,len(x)),random_state=8).sim.fillna(0))),file=ex)
out.close(); print("ok",PFX)
