# -*- coding: utf-8 -*-
"""教育 §5/§6: persona markers by surface; assistant-tail themes."""
from sva_dom_edu_lib import *
a,s=load(); e=edu(); U=users(e)
C=cells(a,s,"教育"); S10=C["搜索top10k"].copy(); S10["query"]=S10["query"].astype(str); S10["pv"]=S10.wise_pv.astype(float)
SURF={"搜索头":U["search_top1000"],"搜索top10k":S10,"助手头":U["assistant_top1k"],"助手尾":U["assistant_random1k"]}
P={
 "小学段(小学/一至六年级/幼儿园/小升初)":r"(小学|[一二三四五六]年级|幼儿园|小升初|幼小衔接)",
 "初中段(初中/初一至初三/七八九年级/中考)":r"(初中|初[一二三]|中考|中招|[七八九]年级|[七八九][上下]册?)",
 "高中段(高中/高一至高三/高考/选科)":r"(高中|高一|高二|高三|高考|选科|复读)",
 "大学与研究生(本科/专科/考研/四六级/专升本/保研)":r"(大学生|大一|大二|大三|大四|本科|专科|考研|研究生|硕士|博士|四六级|四级|六级|专升本|保研|推免|论文|绩点|辅导员)",
 "成人与职业(考公/考编/教资/考证/自考/成考/在职/学历提升)":r"(考公|国考|省考|公务员|考编|事业编|特岗|教资|教师资格|考证|证书|成人|自考|成考|电大|开放大学|在职|职业资格|会计|电工证|建造师|社工证|学历提升|培训班|报班)",
 "家长视角(家长/孩子/儿子/女儿/宝宝)":r"(家长|孩子|儿子|女儿|宝宝|娃)",
 "教师视角(教案/课件/评语/班主任/学生们/说课)":r"(教案|课件|评语|班主任|学生们|说课|听课|评课|教学设计|家长会|作为老师|我是老师|课时)",
 "字数要求(\\d+字)":r"\d+字",
}
rows=[]
for name,pat in P.items():
    line=f"{name}:"
    for s_,g in SURF.items():
        mm=has(g["query"],pat); k=int(mm.sum()); n=len(g); lo,hi=wilson(k,n); pvp=g.loc[mm,"pv"].sum()/g.pv.sum()*100
        rows.append(dict(marker=name,surface=s_,n=n,k=k,pct=k/n*100,lo=lo,hi=hi,pv_pct=pvp)); line+=f" | {s_} {k/n*100:.1f}% [{lo:.1f}–{hi:.1f}] k={k}"+(f" PV{pvp:.1f}%" if s_ in ("搜索头","助手头") else "")
    print(line)
    for s_ in ["搜索头","搜索top10k","助手头","助手尾"]:
        g=SURF[s_]; x=g[has(g["query"],pat)]
        if len(x): 
            exm=x.sort_values("pv",ascending=False).head(10) if s_!="助手尾" else x.sample(min(12,len(x)),random_state=8)
            print(f"   例[{s_}] "+" | ".join(q[:40] for q in exm['query']))
pd.DataFrame(rows).to_csv(f"{SP}/sva_dom_edu_persona.csv",index=False)
# word-count values distribution
for s_ in ["助手头","助手尾"]:
    g=SURF[s_]; wc=g["query"].str.extract(r"(\d+)字")[0].dropna().astype(int)
    print(f"\n{s_} 字数要求 values: n={len(wc)} median {wc.median() if len(wc) else float('nan')} ; ≤300字 {(wc<=300).mean()*100 if len(wc) else 0:.0f}% ; values {sorted(wc.tolist())}")
# l2 (platform subcategory) share
for s_ in ["助手头","助手尾"]:
    g=SURF[s_]; print(f"\n{s_} l2 (platform subcategory): "+" ; ".join(f"{l} {fmt(int(v),len(g))}" for l,v in g.l2.value_counts().items()))
# tail themes
T=U["assistant_random1k"]; H=U["assistant_top1k"]
TH={
 "生成与编辑(u_ds=U10)":None,
 "具体学校的事实(xx中学/小学/学院+分数线/招生/分班/电话/放假…)":r"(中学|小学|学校|学院|大学|一中|二中|三中|附中|实验|幼儿园|职校|技校|校区).{0,14}(分数线|录取|招生|简章|学费|收费|分班|名单|电话|地址|放假|开学|宿舍|食堂|升学率|师资|派位|摇号|排名|占地|校服|作息|军训|转学|插班|宿舍)",
 "升学/专业/院校选择(志愿/专业/就业前景/哪个好)":r"(志愿|专业|就业|前景|报考|选科|哪个好|哪个更|更适合|值得|复读|能上什么|能报什么)",
 "评价老师或学校(老师…怎么样/教学/任教；学校…怎么样/口碑)":r"(老师|教师|班主任|校长).{0,14}(怎么样|评价|口碑|教学|风格|能力|任教|荣誉|获奖|提分|代表作|资历)|(学校|中学|小学|校区|机构|培训|教育).{0,8}(怎么样|好不好|口碑|评价|靠谱|优势)",
 "成人与职业学习(考公/考编/教资/考证/自考/在职)":P["成人与职业(考公/考编/教资/考证/自考/成考/在职/学历提升)"],
 "题目求解(这道题/答案/第N题/选择题/计算)":r"(这道题|这题|题目|答案|解题|第.题|第.问|选择题|填空|阅读题|竖式|计算|方程|解析|[A-D][\.．、 ]|\d+[\+\-×÷\*/]\d+)",
 "字词与语言知识(拼音/读音/组词/意思/翻译/语法)":r"(拼音|读音|怎么读|组词|造句|意思|近义词|反义词|词性|成语|翻译|英文|英语|单词|语法|从句|笔顺|偏旁)",
 "粘贴材料(>60字)":r"^.{61,}$",
}
print("\n#### 助手尾 themes (overlapping), n=",len(T))
trows=[]
for name,pat in TH.items():
    mm=(T.u_ds=="U10") if pat is None else has(T["query"],pat); k=int(mm.sum())
    mh=(H.u_ds=="U10") if pat is None else has(H["query"],pat); kh=int(mh.sum())
    trows.append(dict(theme=name,tail_k=k,tail_n=len(T),head_k=kh,head_n=len(H)))
    x=T[mm]; exm=x.sample(min(14,len(x)),random_state=21)
    print(f"  {name}: 尾 {fmt(k,len(T))} | 头 {fmt(kh,len(H))}\n     例[尾] "+" | ".join(q[:44] for q in exm['query']))
    if kh: print("     例[头] "+" | ".join(q[:30] for q in H[mh].sort_values('pv',ascending=False)['query'].head(8)))
pd.DataFrame(trows).to_csv(f"{SP}/sva_dom_edu_tail_themes.csv",index=False)
# p_ds=1 rows (personal situation) in tail — describe
x=T[T.p_ds==1]; print(f"\n助手尾 p_ds=1 {fmt(len(x),len(T))} u_ds: {x.u_ds.value_counts().to_dict()}\n   例 "+" | ".join(q[:44] for q in x.sample(min(20,len(x)),random_state=2)['query']))
