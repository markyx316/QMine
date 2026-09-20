# -*- coding: utf-8 -*-
"""Persona cue words (who is asking) per domain × surface, user rows, v3 tiers. Python-level regex."""
import sys, re
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from sva_common import *
a,s=load(); DOM5=list(CAT)
def has(series,pat): p=re.compile(pat); return series.astype(str).map(lambda x: bool(p.search(x))).values
CUES={
 "年级/学段词": r"(小学|初中|高中|大学生|[一二三四五六]年级|初[一二三]|高[一二三]|中考|高考|研究生|大[一二三四])",
 "家长视角(孩子/儿子/女儿/宝宝)": r"(孩子|儿子|女儿|宝宝|小孩)",
 "成人应试/考证": r"(考研|考公|考编|公务员|教资|教师资格|考证|职称|自考|成人本科|专升本)",
 "称呼助手为“老师”": r"^老师[，,、\s]",
 "老人/父母/退休": r"(老人|老年|爸妈|父母|退休|养老)",
 "孕产": r"(怀孕|孕妇|备孕|孕期|产后|宝妈|哺乳)",
 "持仓操作词(股民)": r"(持仓|加仓|减仓|割肉|套牢|成本价|补仓|止损|抄底|满仓|清仓|仓位)",
 "病程词(确诊/复查/术后/住院)": r"(确诊|复查|术后|化疗|住院|出院|吃了\d+天|已经\d+天|\d+天了)",
 "儿童动画 IP": r"(奥特曼|喜羊羊|熊出没|小猪佩奇|汪汪队|迪迦|赛罗|巴啦啦|猪猪侠|超级飞侠|小马宝莉)",
 "二创/同人/角色扮演": r"(同人|续写|角色扮演|人设|番外|假如|如果.{0,15}会)",
 "具体地方机构(XX中学/医院/支行…)": r"[一-鿿]{2,8}(中学|小学|医院|分行|支行|派出所|卫生院|人民医院)",
}
C={d:cells(a,s,d) for d in DOM5}
print("#### PERSONA CUES (row share %, user rows)")
for name,pat in CUES.items():
    print(f"\n#### {name}\n  {'':<6}"+"".join(f"{k:>14}" for k in SURF))
    for d in DOM5+["5域"]:
        vals=[]
        for k in SURF:
            q=pd.concat([C[x][k]["query"] for x in DOM5]) if d=="5域" else C[d][k]["query"]
            vals.append(f"{has(q,pat).mean()*100:13.2f}%")
        print(f"  {d:<6}"+"".join(vals))
    for k in ("搜索top10k","助手top1k","助手random1k"):
        q=pd.concat([C[x][k]["query"] for x in DOM5]).astype(str); m=q[has(q,pat)]
        if len(m): print(f"   例[{k}] ({len(m):,}): "+" | ".join(m.sample(min(8,len(m)),random_state=13).str[:34]))
