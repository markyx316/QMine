# -*- coding: utf-8 -*-
"""Behaviour, voice and brand markers on v3 tiers (user rows). Python-level regex (not pandas/RE2), so \\s, backreferences
and Unicode classes behave as written. 5 domains × 4 surfaces, plus the assistant's 33-category totals."""
import sys, re
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from sva_common import *
a,s=load()
def has(series,pat): p=re.compile(pat); return series.map(lambda x: bool(p.search(x)))
MK={
 "图像/画面指代(多模态)": r"(图中|图片中|图里|图上|这张(图|照片|截图)|这个图|照片(中|里)|截图|如图|上传的(图|照片)|识别(一下)?(这|图))",
 "核查身边的人(非名人:老师/医生/校长)": r"[一-鿿]{2,3}(语文|数学|英语|物理|化学|班主任)?(老师|医生|大夫|教授|校长)(是否|有没有|有哪些|怎么样|的评价|评价|教学|获得|获过|还在|在哪|现在|的代表)",
 "事件/八卦词": r"(去世|逝世|离世|离婚|结婚|恋情|出轨|分手|被查|落马|曝光|回应|官宣|绯闻|传闻|丑闻|塌房|免职)",
 "作业/学习任务": r"(作文|读后感|观后感|手抄报|这道题|题目|竖式|组词|造句|练习题|作业|解题|周记)",
 "个人健康处境(我/家人+症状/检查)": r"(我|我家|我妈|我爸|孩子|宝宝|老公|老婆|老人)[^，。？?]{0,8}(发烧|咳嗽|疼|痛|怀孕|血压|血糖|过敏|检查|吃了|手术|化验|报告|症状)",
 "个人财务处境(我/家人+贷款/股票…)": r"(我|我的|我家|老婆|老公)[^，。？?]{0,10}(贷款|征信|信用卡|工资|股票|基金|亏|赚|保险|存款|理财|借|还款)",
 "假设/脑洞(如果/假如)": r"(如果|假如|假设|要是)",
 "预测/判断走势(金融)": r"(预测|目标价|会涨|会跌|能涨|还会涨|还会跌|上涨空间|值得买|值得投资|能买吗|还能买|走势如何|涨停)",
 "敏感健康话题(人流/私密/性功能/腋臭)": r"(人流|打胎|流产|私密|私处|阴|性功能|阳痿|早泄|腋臭|狐臭|包皮|前列腺|避孕|性病|梅毒|hpv|HPV)",
}
VOICE={"唤醒词(小度小度/你好小度)": r"(小度小度|你好小度|小度你好)","喂开头": r"^喂","口语填充(那个那个/就是就是)": r"(那个那个|就是就是|然后然后)",
 "重复词(同一2字词连说)": r"([一-鿿]{2})\1"}
BRAND={"百度":r"百度","文心/文心一言":r"文心","小度":r"小度","豆包":r"豆包","DeepSeek":r"(?i)deepseek","元宝":r"元宝","千问/通义":r"(千问|通义)","Kimi":r"(?i)kimi","ChatGPT/GPT":r"(?i)(chatgpt|gpt)","AI/人工智能(泛称)":r"(AI|ai|人工智能)"}
C={d:{k:v["query"].astype(str) for k,v in cells(a,s,d).items()} for d in CAT}
P={k:pd.concat([C[d][k] for d in CAT]) for k in SURF}
U33={snap:a[(a.tier=="user")&(a.snapshot==snap)]["query"].astype(str) for snap in ("top1k","random1k")}
print("#### n (user rows, v3)"); [print(f"  {d}: "+"  ".join(f"{k} {len(v):,}" for k,v in C[d].items())) for d in CAT]
print("  5域: "+"  ".join(f"{k} {len(v):,}" for k,v in P.items())+f" | 全33 助手top1k {len(U33['top1k']):,} random1k {len(U33['random1k']):,}")
for title,D in (("BEHAVIOUR",MK),("VOICE",VOICE)):
    print(f"\n######## {title} (row share %)")
    for name,pat in D.items():
        print(f"\n#### {name}\n  {'':<6}"+"".join(f"{k:>14}" for k in SURF))
        for d in list(CAT)+["5域"]:
            c=P if d=="5域" else C[d]
            print(f"  {d:<6}"+"".join(f"{has(c[k],pat).mean()*100:13.2f}%" for k in SURF))
        print(f"  全33类: 助手top1k {has(U33['top1k'],pat).mean()*100:.2f}% | 助手random1k {has(U33['random1k'],pat).mean()*100:.2f}%")
        for k in ("搜索top10k","助手top1k","助手random1k"):
            m=P[k][has(P[k],pat)]
            if len(m): print(f"   例[{k}] ({len(m):,}): "+" | ".join(m.sample(min(8,len(m)),random_state=11).str[:36]))
long_np=lambda q: q.map(lambda x: len(x)>20 and not re.search(r"[，,。？?！!、；;：:\s]",x))
print("\n#### 长句无标点(>20字)  "+" | ".join(f"{k}(5域) {long_np(P[k]).mean()*100:.2f}%" for k in SURF)+f" | 助手random1k(全33) {long_np(U33['random1k']).mean()*100:.2f}%")
print("\n######## BRAND MENTIONS (rows)")
for name,pat in BRAND.items():
    ex=U33["random1k"][has(U33["random1k"],pat)]
    print(f"  {name}: "+" | ".join(f"{k}(5域) {int(has(P[k],pat).sum())}" for k in SURF)+f" | 助手top1k(全33) {int(has(U33['top1k'],pat).sum())} | 助手random1k(全33) {len(ex)}  例: "+" / ".join(ex.head(5).str[:30]))
