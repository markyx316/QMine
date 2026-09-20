# -*- coding: utf-8 -*-
"""What are the U13 rows in the 教育 assistant head?"""
from sva_dom_edu_lib import *
a,s=load()
e=edu(); U=users(e); H=U["assistant_top1k"]; T=U["assistant_random1k"]
h13=H[H.u_ds=="U13"].copy(); t13=T[T.u_ds=="U13"].copy()
print(f"head U13 {fmt(len(h13),len(H))}; PV share {h13.pv.sum()/H.pv.sum()*100:.1f}% ; tail U13 {fmt(len(t13),len(T))}")
print("head U13 length: median",h13['query'].str.len().median(),"share len<=2:",fmt(int((h13['query'].str.len()<=2).sum()),len(h13)))
print("head non-U13 length median",H[H.u_ds!='U13']['query'].str.len().median())
# shapes
SH=[("单个汉字",r"^[一-鿿]$"),("两个汉字",r"^[一-鿿]{2}$"),("纯拉丁字母",r"^[A-Za-z]+$"),("纯数字/数学式",r"^[\d\s\.\+\-×÷\*/=（）()%]+$"),
    ("以问号结尾或含疑问词",r"(\?|？|吗|呢|什么|怎么|哪|多少|几)"),("指代(这/那/它/他/她)",r"(这|那|它|他|她)"),("题目/图片/拍照词",r"(题|图|拍|照片|识别|清晰|模糊|扫)"),
    ("语言工具词(拼音/读音/翻译/组词/造句)",r"(拼音|读音|翻译|组词|造句|笔顺|部首|偏旁)"),("字数/格式约束",r"(\d+字|字数|格式|简短|分点)")]
for name,pat in SH:
    m=has(h13["query"],pat); mt=has(t13["query"],pat)
    ex=h13[m].sort_values("pv",ascending=False).head(12)
    print(f"  {name}: head {fmt(int(m.sum()),len(h13))} PV {h13[m].pv.sum()/h13.pv.sum()*100:.1f}% | tail {fmt(int(mt.sum()),len(t13))} | 例 "+" | ".join(f"{q}({p:,.0f})" for q,p in zip(ex['query'],ex.pv)))
print("\nTOP 60 head U13 by PV:")
print(" | ".join(f"{q}({p:,.0f})" for q,p in zip(h13.sort_values('pv',ascending=False)['query'].head(60),h13.sort_values('pv',ascending=False).pv.head(60))))
# cross-category recurrence of the same string in assistant top1k (all tiers) and in all 33 categories user tier
at=a[a.snapshot=="top1k"].copy(); at["query"]=at["query"].astype(str)
g=at.groupby("query").agg(ncat=("l1","nunique"),pv_all=("search_num","sum"))
edu_pv=at[at.l1=="教育培训"].groupby("query").search_num.sum()
h13=h13.join(g,on="query"); h13["edu_pv_share"]=h13["query"].map(edu_pv)/h13.pv_all
Hn=H[H.u_ds!="U13"].join(g,on="query")
print("\nCROSS-CATEGORY: same string in N assistant top1k categories")
for lab,x in [("head U13",h13),("head non-U13",Hn)]:
    print(f"  {lab}: n={len(x)} ncat==1 {fmt(int((x.ncat==1).sum()),len(x))} ; ncat>=3 {fmt(int((x.ncat>=3).sum()),len(x))} ; ncat median {x.ncat.median()}")
multi=h13[h13.ncat>=3].sort_values("pv",ascending=False)
print("  U13 strings in >=3 categories (top by PV): "+" | ".join(f"{q}(教育PV{p:,.0f};{int(n)}类;全部PV{pa:,.0f})" for q,p,n,pa in zip(multi['query'].head(25),multi.pv.head(25),multi.ncat.head(25),multi.pv_all.head(25))))
# in search 教育 top10k?
ss=s[(s.domain=="教育")].sort_values("wise_pv",ascending=False).reset_index(drop=True); ss["r"]=np.arange(1,len(ss)+1); ss["query"]=ss["query"].astype(str)
sr=dict(zip(ss["query"],ss["r"])); spv=dict(zip(ss["query"],ss["wise_pv"]))
h13["in_search10k"]=h13["query"].map(lambda q: q in sr)
print(f"\nhead U13 strings that exist verbatim in 教育 search top10k: {fmt(int(h13.in_search10k.sum()),len(h13))}; non-U13 head: {fmt(int(Hn['query'].map(lambda q:q in sr).sum()),len(Hn))}")
x=h13[h13.in_search10k].sort_values("pv",ascending=False)
print("  e.g. "+" | ".join(f"{q}(助手PV{p:,.0f}; 搜索#{sr[q]} PV{spv[q]:,})" for q,p in zip(x['query'].head(25),x.pv.head(25))))
# own labels for U13 rows
print("\nown_intent of head U13:"); print((h13.own_intent.value_counts(normalize=True)*100).round(1).head(10).to_string())
print("\nown_leaf of head U13:"); print((h13.own_leaf.value_counts(normalize=True)*100).round(1).head(15).to_string())
print("\nl2 of head U13 vs head non-U13:"); print(pd.concat([h13.l2.value_counts(normalize=True).rename("U13"),Hn.l2.value_counts(normalize=True).rename("nonU13")],axis=1).mul(100).round(1).to_string())
# tail U13 examples
print("\ntail U13 random 40: "+" | ".join(t13.sample(min(40,len(t13)),random_state=11)['query'].str[:40]))
# answer-to-question shapes: grade/subject/number/option words alone
ANS=[("年级/学段单独",r"^(一|二|三|四|五|六|初一|初二|初三|高一|高二|高三|大一|大二|小学|初中|高中|大学)(年级)?[。.！!？?]?$"),
     ("学科单独",r"^(语文|数学|英语|物理|化学|生物|历史|地理|政治)$"),("选项字母",r"^[A-Da-d][。.]?$"),
     ("数字单独",r"^\d{1,4}$"),("字数单独(如50字)",r"^\d+字(左右)?$"),("价格/费用/多少钱单独",r"^(价格|价钱|费用|学费|多少钱|价位多少|学费多少)$")]
print("\nANSWER-LIKE shapes among ALL head user rows (any intent) and their u_ds:")
for name,pat in ANS:
    m=has(H["query"],pat); x=H[m]
    print(f"  {name}: {fmt(int(m.sum()),len(H))} ; U13 among them {int((x.u_ds=='U13').sum())} ; 例 "+" | ".join(f"{q}({p:,.0f},{u})" for q,p,u in zip(x.sort_values('pv',ascending=False)['query'].head(10),x.sort_values('pv',ascending=False).pv.head(10),x.sort_values('pv',ascending=False).u_ds.head(10))))
h13.drop(columns=[c for c in ["u_qw","p_qw"] if c in h13.columns]).to_csv(f"{SP}/sva_dom_edu_u13_head.csv",index=False)
