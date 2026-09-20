# -*- coding: utf-8 -*-
"""教育: season/date confound — search snapshot is 2026-07-01; when is the assistant log from? + feature-entry PV shares."""
from sva_dom_edu_lib import *
from collections import Counter
a,s=load(); e=edu(); U=users(e)
C=cells(a,s,"教育"); S10=C["搜索top10k"].copy(); S10["query"]=S10["query"].astype(str); S10["pv"]=S10.wise_pv.astype(float)
SURF={"搜索头":U["search_top1000"],"搜索top10k":S10,"助手头":U["assistant_top1k"],"助手尾":U["assistant_random1k"]}
SEA={"招录季(志愿/出成绩/成绩查询/录取/分数线/一分一段)":r"(志愿|出成绩|成绩查询|查成绩|成绩怎么查|录取|分数线|一分一段|位次|投档)",
     "开学季(开学/暑假/军训/教师节/新学期/寒假)":r"(开学|暑假|军训|教师节|新学期|寒假|秋季学期)"}
for nm,pat in SEA.items():
    print(nm+": "+" | ".join(f"{k} {fmt(int(has(g['query'],pat).sum()),len(g))}"+(f" PV{g.loc[has(g['query'],pat),'pv'].sum()/g.pv.sum()*100:.1f}%" if k in ('搜索头','助手头') else "") for k,g in SURF.items()))
    for k in ["搜索头","助手头","助手尾"]:
        g=SURF[k]; x=g[has(g['query'],pat)]
        if len(x): print(f"   例[{k}] "+" | ".join(x.sort_values('pv',ascending=False)['query'].head(8).str[:30]))
# assistant date mentions, all 33 categories, all tiers
aq=a["query"].astype(str)
pat=re.compile(r"(?:(20\d\d)年)?(\d{1,2})月(\d{1,2})[日号]")
cnt=Counter(); ex={}
for q,snap in zip(aq,a.snapshot):
    if re.search(r"(今天|今日|现在是|目前是|当前)",q):
        for y,mo,d in pat.findall(q):
            key=f"{y or '----'}-{int(mo):02d}-{int(d):02d}"; cnt[key]+=1; ex.setdefault(key,q[:50])
print("\nassistant (all 33 cats) rows with 今天/今日/现在是 + a date: top dates:")
for k,v in cnt.most_common(12): print(f"   {k}: {v}  例 {ex[k]}")
yrmo=Counter()
for q in aq:
    for y,mo,d in pat.findall(q):
        if y=="2026": yrmo[int(mo)]+=1
print("   all rows mentioning '2026年M月D日': by month",dict(sorted(yrmo.items())))
# feature entries & S1 shares of head PV (all tiers)
al=e[e.surface=="assistant_top1k"]; tot=al.pv.sum()
for q in ["帮我全网文本搜题","翻译文字","领取AI志愿报告"]:
    x=al[al["query"]==q]; print(f"feature '{q}': tier {x.tier.unique()} PV {x.pv.sum():,.0f} = {x.pv.sum()/tot*100:.2f}% of head PV")
t2=al[al.tier=="S2_system_template"]; tmpl=t2[has(t2["query"],r"^帮我画[：:]")]
print(f"S2 '帮我画：…手抄报' templates: n={len(tmpl)} PV {tmpl.pv.sum():,.0f} = {tmpl.pv.sum()/tot*100:.2f}%")
