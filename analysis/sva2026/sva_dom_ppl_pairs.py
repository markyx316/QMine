# -*- coding: utf-8 -*-
"""人物 section 4: the selected matched pairs (public figures only). Asserts every string exists on its surface
(clean_v3, user tier), looks up search rank/PV, and computes the pair's own bge-base-zh-v1.5 cosine."""
import sys, numpy as np, torch
SP=__import__("os").environ.get("SVA_WORKDIR", __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "work"))
sys.path.insert(0, SP)
from sva_common import *
from sentence_transformers import SentenceTransformer
a, s = load(); C = cells(a, s, "人物")
S10 = C["搜索top10k"].copy(); S10["query"] = S10["query"].astype(str)
NN = pd.read_parquet(f"{SP}/sva_dom_ppl_nn.parquet")
PAIRS = [
 ("A 加入第二个人/关系", "孙宇晨", "助手top1k", "孙宇晨和景甜的关系如何"),
 ("A 加入第二个人/关系", "杨绛先生的简介", "助手random1k", "杨绛的丈夫是谁"),
 ("B 问原因/来龙去脉", "陈武", "助手top1k", "陈武同志的逝世原因是什么"),
 ("B 问原因/来龙去脉", "孙殿英", "助手random1k", "孙殿英晚年结局如何"),
 ("C 从资料卡转向私生活/财富", "景甜个人简介", "助手top1k", "景甜的个人生活如何"),
 ("C 从资料卡转向私生活/财富", "周杰伦个人资料简介", "助手random1k", "周杰伦的资产总额有多少"),
 ("D 要判断/比较/排名", "十大元帅", "助手top1k", "十大元帅中谁最厉害"),
 ("D 要判断/比较/排名", "上海市委现任书记是谁", "助手random1k", "上海历任市长和市委书记中谁最著名"),
 ("E 加时间: 近况与现任", "耿彦波", "助手top1k", "耿彦波现在什么职位"),
 ("E 加时间: 近况与现任", "任振鹤", "助手random1k", "任振鹤省长最新任职"),
 ("F 从资料/打榜转向清单与量化", "田栩宁百科个人资料大全", "助手top1k", "田栩宁的代表作品有哪些"),
 ("F 从资料/打榜转向清单与量化", "梓渝的影响力", "助手top1k", "梓渝的粉丝群体规模有多大"),
]
m = SentenceTransformer("BAAI/bge-base-zh-v1.5", device="mps" if torch.backends.mps.is_available() else "cpu")
rows = []
for g, sq, surf, aq in PAIRS:
    sr = S10[S10["query"] == sq]; assert len(sr) == 1, (sq, len(sr))
    ar = C[surf][C[surf]["query"].astype(str) == aq]; assert len(ar) >= 1, (aq, surf)
    e = m.encode([sq, aq], normalize_embeddings=True); cos = float(e[0] @ e[1])
    nn = NN[(NN.surface == surf) & (NN["query"] == aq)].iloc[0]
    link = "近邻" if nn.nn_query == sq else ("包装(含该裸名)" if (sq in aq) else "同实体字符串匹配")
    rows.append(dict(group=g, search_query=sq, search_rank=int(sr["rank"].iloc[0]), search_pv=int(sr.wise_pv.iloc[0]), assistant_surface=surf,
                     assistant_query=aq, assistant_pv=int(ar.search_num.iloc[0]), pair_cos=round(cos, 3), assistant_nn=nn.nn_query, assistant_nn_sim=round(float(nn.sim), 3), link=link))
R = pd.DataFrame(rows); R.to_csv(f"{SP}/sva_dom_ppl_pairs.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
print(R.to_string(index=False))
