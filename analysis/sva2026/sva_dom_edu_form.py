# -*- coding: utf-8 -*-
"""教育 §2 form measurements at matched depth (+ search top10k lens, assistant tail)."""
from sva_dom_edu_lib import *
a,s=load(); e=edu(); U=users(e)
C=cells(a,s,"教育"); S10=C["搜索top10k"].copy(); S10["query"]=S10["query"].astype(str); S10["pv"]=S10.wise_pv.astype(float)
SURF={"搜索头":U["search_top1000"],"搜索top10k":S10,"助手头":U["assistant_top1k"],"助手尾":U["assistant_random1k"]}
M={
 "疑问句":r"(怎么|如何|为什么|什么|哪个|哪家|哪些|多少|吗|呢|是不是|能不能|可不可以|\?|？)",
 "委托/祈使(严格)":r"(帮我|帮忙|请你|请帮|给我(写|画|生成|做|翻译|总结|出|来)|(写|画|生成|做|翻译|总结|改写|续写|润色)(一|个|篇|份|张|下|成|出)|^(写|画|生成|翻译|总结|改写|续写|润色).{2,})",
 "第一人称":r"(我|我的|我家|我们|本人|俺)",
 "裸校名":r"^[^\s怎么什么哪多少吗？?]{2,16}(大学|学院|学校|中学|小学|职业技术学院|分校|校区)$",
 "官方入口/考试院/学信网":r"(官网|入口|登录|考试院|学信网|招生考试|服务平台|教务|认证平台|办事大厅)",
 "作业与写作任务词":r"(作文|读后感|观后感|日记|周记|手抄报|组词|造句|这道题|这题|竖式|脱式|解方程|看图写话|仿写|续写|扩写|缩写|赏析|批注|思维导图|好词好句)",
 "依赖上文的跟进/改写":r"(^(再|换|简化|简洁|缩短|精简|重写|重新|翻译成|继续|详细一点|扩写|不用|直接)|这篇|这首诗|这段|这道题|这题|上面|刚才|上一|第.问|第.题|更简单|更短|字数)",
 "工具名单独输入(翻译/拼音/组词…)":r"^(翻译|在线翻译|百度翻译|英语翻译|翻译器|拼音|组词|注音|造句|笔顺|涂色|上色|画图|翻译一下|翻译中文|翻译成英文|翻译成中文)[。.！!]?$",
 "字音字形字义(读音/拼音/笔顺/组词/意思)":r"(怎么读|读音|拼音|笔顺|笔画|组词|部首|偏旁|多音字|什么意思|的意思)",
 "输出约束(字数/格式)":r"(\d+字|字数|字左右|格式|表格|简短|一句话|分点|分条|不超过|以内)",
 "生肖谜语":r"(生肖|打一肖|何肖|什么肖)",
 "分数线/录取/招生/学费":r"(分数线|录取|招生|简章|学费|收费|位次|一分一段|价目表)",
 "≤2字":r"^.{1,2}$",
}
rows=[]
print("filter: sva_final_rows.parquet domain==教育 tier==user (search_top1000/assistant_top1k/assistant_random1k); 搜索top10k = sva_common.cells() user rows\n")
for s_,g in SURF.items():
    L=g["query"].str.len(); print(f"{s_}: n={len(g)} 长度中位 {L.median():.0f} p90 {L.quantile(.9):.0f}; PV加权长度中位 {np.median(np.repeat(L.values,np.maximum(1,(g.pv.values/g.pv.min()).astype(int))) ) if s_!='助手尾' else float('nan'):.0f}")
print()
for name,pat in M.items():
    line=f"{name}:"
    for s_,g in SURF.items():
        m=has(g["query"],pat); k=int(m.sum()); n=len(g); lo,hi=wilson(k,n); pvp=g.loc[m,"pv"].sum()/g.pv.sum()*100
        rows.append(dict(marker=name,surface=s_,n=n,k=k,pct=k/n*100,lo=lo,hi=hi,pv_pct=pvp))
        line+=f" | {s_} {k/n*100:.1f}% [{lo:.1f}–{hi:.1f}] k={k}" + (f" PV{pvp:.1f}%" if s_ in ("搜索头","助手头") else "")
    print(line)
    for s_ in ["搜索头","助手头","助手尾"]:
        g=SURF[s_]; x=g[has(g["query"],pat)]
        if len(x)==0: continue
        ex=x.sort_values("pv",ascending=False).head(8) if s_!="助手尾" else x.sample(min(8,len(x)),random_state=4)
        print(f"    例[{s_}] "+" | ".join(f"{q[:30]}({p:,.0f})" for q,p in zip(ex['query'],ex.pv)))
pd.DataFrame(rows).to_csv(f"{SP}/sva_dom_edu_form.csv",index=False)
# exact string '翻译' on both surfaces
for q in ["翻译","拼音","组词","我","一","作文","读后感","高考","志愿填报","高考志愿填报"]:
    print(f"'{q}': 搜索top10k PV {S10.loc[S10['query']==q,'pv'].sum():,.0f} rank {list(np.where(S10['query'].values==q)[0]+1)} | 助手头 PV {U['assistant_top1k'].loc[U['assistant_top1k']['query']==q,'pv'].sum():,.0f}")
