# -*- coding: utf-8 -*-
"""人物: raw samples for regex/theme design (not for quoting numbers)."""
import sys
SP=__import__("os").environ.get("SVA_WORKDIR", __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "work"))
sys.path.insert(0, SP)
from sva_common import *
a, s = load(); c = cells(a, s, "人物")
def show(title, q, per=12):
    q = [x[:34] for x in q]
    print(f"\n==== {title} (shown {len(q)})")
    for i in range(0, len(q), per): print("  " + " | ".join(q[i:i+per]))
st = c["搜索top10k"]
show("搜索 top1000 按PV 1-240", st.head(240)["query"].astype(str).tolist())
show("搜索 1001-10000 随机160", st.iloc[1000:].sample(160, random_state=1)["query"].astype(str).tolist())
show("助手 top1k user 按PV 1-260", c["助手top1k"].sort_values("search_num", ascending=False).head(260)["query"].astype(str).tolist())
show("助手 top1k user 按PV 261-889 随机120", c["助手top1k"].sort_values("search_num", ascending=False).iloc[260:].sample(120, random_state=2)["query"].astype(str).tolist())
show("助手 random1k user 随机320", c["助手random1k"].sample(320, random_state=3)["query"].astype(str).tolist(), per=8)
