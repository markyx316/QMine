# -*- coding: utf-8 -*-
import sys; sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from sva_common import *
a,s=load()
M={"疑问句":r"(怎么|如何|为什么|什么|哪个|哪家|哪些|多少|吗|呢|是不是|能不能|可不可以|\?|？)",
 "祈使句(严格)":r"(帮我|帮忙|请你|请帮|给我(写|画|生成|做|翻译|总结|出|来)|(写|画|生成|做|翻译|总结|改写|续写|润色)(一|个|篇|份|张|下|成|出)|^(写|画|生成|翻译|总结|改写|续写|润色).{2,})",
 "第一人称":r"(我|我的|我家|我们|本人|俺)","称呼对方(你/您)":r"(你|您)",
 "家庭成员":r"(孩子|儿子|女儿|宝宝|宝贝|老公|老婆|男朋友|女朋友|男友|女友|我妈|我爸|妈妈|爸爸|父母|老人|家人|婆婆|公公|闺女|对象)",
 "身份/人生阶段":r"(学生|同学|老师|家长|宝妈|孕妇|怀孕|备孕|退休|高考|中考|考研|考公|上班|打工|公司|老板|员工|单位|领导|同事|客户)",
 "年龄自述":r"(\d{1,2}岁|\d{2}后|\d{1,2}个月大)","情绪词":r"(焦虑|害怕|担心|难受|崩溃|烦死|伤心|委屈|生气|郁闷|痛苦|着急|心累|绝望|开心|高兴)",
 "求助(怎么办)":r"(怎么办|咋办|救命)","礼貌用语":r"(请问|请你|请帮|谢谢|麻烦|您)",
 "纠错/抱怨助手":r"(你说错|说错了|不对|错了|听不懂|废话|你怎么|不是这个|答非所问|重新回答)",
 "输出约束":r"(\d+字|字数|字左右|格式|表格|简短|一句话|分点|分条|比例为|不超过|以内)",
 "时效词":r"(今天|今日|明天|现在|最新|刚刚|实时|最近)","比较选择":r"(还是|哪个好|哪个更|区别|对比|比较|vs|VS)",
 "导航词":r"(官网|下载|入口|app|APP|登录|直播|客户端|网址)","资源词":r"(在线观看|免费观看|全集|原文|图片|网盘|百度云|完整版|高清)"}
C={d:{k:v["query"].astype(str) for k,v in cells(a,s,d).items()} for d in CAT}
P={k:pd.concat([C[d][k] for d in CAT]) for k in SURF}
print("#### n"); [print(f"  {d}: "+"  ".join(f"{k} {len(v):,}" for k,v in C[d].items())) for d in CAT]; print("  合计: "+"  ".join(f"{k} {len(v):,}" for k,v in P.items()))
print("\n#### LENGTH (median / p90 / >30字% / >60字%)")
for d in list(CAT)+["合计"]:
    c=P if d=="合计" else C[d]
    print(f"  {d:<4} "+" | ".join(f"{k}: {v.str.len().median():.0f}/{v.str.len().quantile(.9):.0f}/{(v.str.len()>30).mean()*100:.1f}%/{(v.str.len()>60).mean()*100:.2f}%" for k,v in c.items()))
for name,pat in M.items():
    print(f"\n#### {name} (row share %)")
    print(f"  {'':<6}"+"".join(f"{k:>14}" for k in SURF))
    for d in list(CAT)+["合计"]:
        c=P if d=="合计" else C[d]
        print(f"  {d:<6}"+"".join(f"{c[k].str.contains(pat,regex=True).mean()*100:>13.2f}%" for k in SURF))
