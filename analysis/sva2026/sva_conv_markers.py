# -*- coding: utf-8 -*-
"""Conversational / delegation markers at matched depth, cleaned (user tier)."""
import sys; sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from sva_common import *
a,s=load()
C={d:{k:v["query"].astype(str) for k,v in cells(a,s,d).items()} for d in CAT}
P={k:pd.concat([C[d][k] for d in CAT]) for k in SURF}
def table(name,pat,nex=8):
    print(f"\n#### {name}")
    print(f"  {'':<6}"+"".join(f"{k:>14}" for k in SURF))
    for d in list(CAT)+["合计"]:
        c=P if d=="合计" else C[d]
        print(f"  {d:<6}"+"".join(f"{c[k].str.contains(pat,regex=True).mean()*100:13.2f}%" for k in SURF))
    for k in ("搜索top10k","助手top1k","助手random1k"):
        m=P[k][P[k].str.contains(pat,regex=True)]
        if len(m): print(f"   例[{k}] ({len(m):,}): "+" | ".join(m.sample(min(nex,len(m)),random_state=5).str[:36]))
CONV={
 "句首承接(那么/所以/还有/再…)": r"^(那么|那(?!英|些|年|一|座|个|里|时|天|夜|山|片|是)|所以|还有|另外|然后|再(?!见|婚|生|别)|继续|接着|也就是说|不是|不对|对了|好的|嗯)",
 "回指上文(上面/刚才/你说的/这道题…)": r"(上面|上述|以上|刚才|刚刚|你说的|你刚|前面说|之前说|你给的|你写的|你画的|你生成的|这篇|这首诗|这段话|这张图|这道题|这个题|这句话|这份报告)",
 "对助手说话(你说/你能/谢谢/错了/重新)": r"(你(说|写|画|给|生成|回答|搞|弄|是谁|能|会|知道|觉得)|谢谢|错了|不是这个|重新|重来)",
 "委托句式(请/帮我/给我…开头)": r"^(请你?|帮我|帮忙|给我|麻烦|你给我|你帮我|能不能帮我|能否帮我|能帮我|可以帮我)",
 "动作动词开头(写/画/分析/预测…)": r"^(写|画|生成|制作|分析|预测|推荐|总结|概括|解读|翻译|计算|对比|比较|列出|整理|查询|查一下|介绍一下|讲讲|说说|解释|评价)",
 "多问或带输出约束": r"([？?].{2,}[？?]|要求|并且|同时要|包括|字数|不少于|不超过|\d+字|格式|分点|表格|语气|风格)",
}
VERBS={"分析/研判":r"分析|研判|判断一下","预测/走势":r"预测|目标价|走势|会涨|会跌|能涨|还会涨|还会跌|后市","推荐":r"推荐",
 "总结/整理/梳理":r"总结|概括|归纳|整理|梳理","解读/解释/讲解":r"解读|解释|讲解|讲讲|说说",
 "写作(作文/读后感/文案…)":r"写一|写个|写篇|写段|帮我写|作文|读后感|观后感|文案|周记|日记|对联|下联",
 "画/生成图像视频":r"画一|帮我画|画个|画张|生成.{0,8}(图|视频|照片|海报|头像)","翻译":r"翻译",
 "计算/解题":r"计算|算一下|算算|竖式|解方程|这道题|怎么解","对比/评价":r"对比|比较|哪个好|哪个更|评价|怎么样|靠谱吗|好不好"}
print("######## CONVERSATIONAL MARKERS (row share %, user tier)")
for k,v in CONV.items(): table(k,v)
print("\n\n######## TASK VERBS (row share %, user tier)")
for k,v in VERBS.items(): table(k,v,nex=6)
