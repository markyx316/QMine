# -*- coding: utf-8 -*-
"""Own taxonomies per surface + search top10k lens (cells, markers)."""
from sva_dom_edu_lib import *
a,s=load()
e=edu(); U=users(e)
for k,g in U.items():
    vc=g.own_intent.value_counts(); n=len(g)
    print(f"\n[{SN[k]}] own_intent n={n}, classes={g.own_intent.nunique()}, leaves={g.own_leaf.nunique()}")
    for cls,kk in vc.head(12).items():
        ex=g[g.own_intent==cls].sort_values("pv",ascending=False)["query"].head(6).str[:20].tolist() if k!="assistant_random1k" else g[g.own_intent==cls].sample(min(6,kk),random_state=2)["query"].str[:24].tolist()
        print(f"   {cls:<22} {fmt(int(kk),n)} 例: "+" | ".join(ex))
    print("   leaves top10: "+" ; ".join(f"{l} {v/n*100:.1f}%" for l,v in g.own_leaf.value_counts().head(10).items()))
    # u_ds x own_intent crosstab top
    ct=pd.crosstab(g.own_intent,g.u_ds)
    print("   own_intent → main u_ds: "+" ; ".join(f"{i}: "+",".join(f"{c}{ct.loc[i,c]}" for c in ct.loc[i].sort_values(ascending=False).head(3).index if ct.loc[i,c]>0) for i in vc.head(8).index))
pd.concat([g.own_intent.value_counts().rename("k").to_frame().assign(surface=k,n=len(g)) for k,g in U.items()]).to_csv(f"{SP}/sva_dom_edu_own_intent.csv")
# search top10k lens
C=cells(a,s,"教育"); S10=C["搜索top10k"].copy(); S10["query"]=S10["query"].astype(str); S10=S10.reset_index(drop=True); S10["r"]=np.arange(1,len(S10)+1)
print(f"\n#### SEARCH top10k user n={len(S10)}; PV at rank 1000={S10.wise_pv.iloc[999]}, 10000th/last={S10.wise_pv.iloc[-1]}")
bands=[(1,1000),(1001,3000),(3001,6000),(6001,len(S10))]
vc=S10.td_l1_name.value_counts()
print("own td_l1_name shares by rank band:")
print(f"  {'class':<22}"+"".join(f"{f'{lo}-{hi}':>14}" for lo,hi in bands)+f"{'top10k':>14}")
for cls in vc.index:
    line=f"  {cls:<22}"
    for lo,hi in bands:
        b=S10[(S10.r>=lo)&(S10.r<=hi)]; line+=f"{(b.td_l1_name==cls).mean()*100:13.1f}%"
    line+=f"{(S10.td_l1_name==cls).mean()*100:13.1f}%"; print(line)
M={"裸校名(以大学/学院/学校/中学/小学结尾,≤16字,无问词)":r"^[^\s怎么什么哪多少吗？?]{2,16}(大学|学院|学校|中学|小学|职业技术学院|分校|校区)$",
   "读音/拼音":r"(怎么读|读音|拼音|念什么)","笔顺/笔画/部首":r"(笔顺|笔画|部首|偏旁)","组词/造句/近反义":r"(组词|造句|近义词|反义词)",
   "分数线/录取/位次":r"(分数线|录取|位次|一分一段|投档)","成绩查询/查分":r"(成绩|查分|查询入口)","官网/入口/登录":r"(官网|入口|登录|平台)",
   "生肖谜语":r"(生肖|打一肖|什么肖|何肖)","翻译/英文":r"(翻译|英文|英语怎么说)","古诗/原文":r"(古诗|原文|全文|诗句)","意思/含义":r"(意思|含义|是什么)",
   "作文/读后感":r"(作文|读后感|观后感|日记)","排名/985/211":r"(排名|985|211|双一流)","志愿/专业/就业":r"(志愿|专业|就业)","考试(高考/中考/考研/四六级/普通话/教资)":r"(高考|中考|考研|四六级|普通话|教资|教师资格)",
   "学费/招生":r"(学费|招生|简章)","怎么/如何(疑问)":r"(怎么|如何|怎样)"}
print("\nmarkers (row share, Wilson) — search top1000 vs top10k vs 1001-10000:")
out=[]
for name,pat in M.items():
    m=has(S10["query"],pat); top=m[S10.r<=1000]; deep=m[S10.r>1000]
    ex=S10[m&(S10.r>1000)].sample(min(6,int((m&(S10.r>1000)).sum())),random_state=1)["query"].tolist() if (m&(S10.r>1000)).sum() else []
    print(f"  {name}: top1000 {fmt(int(top.sum()),len(top))} | top10k {fmt(int(m.sum()),len(m))} | 1001+ {fmt(int(deep.sum()),len(deep))} 例(1001+): "+" | ".join(ex))
    out.append(dict(marker=name,top1000_k=int(top.sum()),top1000_n=len(top),top10k_k=int(m.sum()),top10k_n=len(m),deep_k=int(deep.sum()),deep_n=len(deep)))
pd.DataFrame(out).to_csv(f"{SP}/sva_dom_edu_search10k_markers.csv",index=False)
