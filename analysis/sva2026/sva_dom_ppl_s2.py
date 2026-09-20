# -*- coding: utf-8 -*-
"""人物 section 2: query-shape measurements at matched depth (user tier, clean_v3).
Surfaces via sva_common.cells(): 搜索top1000 / 搜索top10k / 助手top1k / 助手random1k. Python `re` only (not pandas/RE2)."""
import sys, re, math, numpy as np
SP=__import__("os").environ.get("SVA_WORKDIR", __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "work"))
sys.path.insert(0, SP)
from sva_common import *
import jieba, jieba.posseg as pseg
jieba.setLogLevel(60)
def wilson(k, n, z=1.96):
    if n == 0: return float("nan"), float("nan"), float("nan")
    p = k/n; den = 1+z*z/n; c = (p+z*z/(2*n))/den; h = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den; return p, max(0.0, c-h), min(1.0, c+h)
def newcombe(k1, n1, k2, n2):
    p1, l1, u1 = wilson(k1, n1); p2, l2, u2 = wilson(k2, n2); d = p2-p1
    return d, d-math.sqrt((p2-l2)**2+(u1-p1)**2), d+math.sqrt((u2-p2)**2+(p1-l1)**2)
a, s = load(); C = cells(a, s, "人物")
SURF = ["搜索top1000", "搜索top10k", "助手top1k", "助手random1k"]
Q = {k: v["query"].astype(str).tolist() for k, v in C.items()}
PV = {k: (v["wise_pv"] if "wise_pv" in v.columns else v["search_num"]).astype(float).to_numpy() for k, v in C.items()}
CJK = "[一-鿿]"
M = {
 "M01 纯2–4个汉字(裸名形态)": rf"^{CJK}{{2,4}}$",
 "M02 百科资料后缀": r"(简介|个人资料|资料|简历|履历|百科|生平|个人信息|介绍)",
 "M03 粉丝打榜(送花/影响力)": r"(送花|影响力|打榜|应援|超话|明星势力)",
 "M04 问句形式": r"(怎么|如何|为什么|为何|什么|哪个|哪些|哪里|哪位|哪年|多少|几岁|吗|呢|是不是|是否|谁|\?|？)",
 "M05 指代/识图(不给名字)": r"(这是谁|这个人|这人|这谁|这个.{0,3}是谁|这.{0,2}(女|男)的是谁|图中|图片中|照片中|(他|她)叫什么|像哪个明星|这是哪(个|位))",
 "M06 关系与私生活": r"(关系|婚姻|丈夫|妻子|老公|老婆|男友|女友|恋情|恋爱|离婚|结婚|分手|出轨|前夫|前妻|子女|儿子|女儿|家族|家庭|家世|父亲|母亲|感情|个人生活|同框|[cC][pP]|闺蜜|求婚)",
 "M07 生死健康": r"(去世|逝世|离世|死因|死了|病逝|身亡|遗体|火化|症状|身体不适|病情|自杀|还活着|在世|过世)",
 "M08 职务任免词": r"(任职|职务|调任|任免|免职|辞职|辞去|现任|历任|升任|当选|担任|接任|上任|在任|领导班子|班子成员|调离|职级|级别)",
 "M09 查处与司法": r"(被查|落马|违纪|违法|双开|审查|调查|判刑|判了|判处|无期|死刑|被抓|入狱|获刑|处分|查封)",
 "M10 清单/比较(有哪些/谁最)": r"(有哪些|哪几|名单|排名|排行|谁最|哪个更|谁更|十大|四大|都有谁|都是谁|获得者)",
 "M11 原因/评价/影响": r"(为什么|为何|原因|怎么回事|如何评价|怎么评价|评价|怎么样|如何$|影响|有何)",
 "M12 财富/资产/私密": r"(身价|资产|财富|房产|名下|收入|薪酬|工资|存款|多少钱|有钱|电话|号码|住址|家住)",
 "M13 地方/机构层级词": r"(县|镇|乡|村|街道|派出所|区委|区政府|市委|市政府|中学|小学|一中|二中|三中|学院|大学|医院|公司|集团|分行|支行|局)",
 "M14 教师/医生/学者角色词": r"(老师|教师|班主任|校长|教授|博士|医生|大夫|研究员|团队|课题组)",
 "M15 历史/古代词": r"(朝代|皇帝|帝王|皇后|太后|太子|古代|古人|元帅|大将|上将|将军|长征|抗战|民国|清朝|明朝|汉朝|唐朝|宋朝|秦朝|三国|春秋|战国|历史)",
 "M16 时效/近况": r"(现在|最新|近况|目前|最近|近日|如今|还在|今年|2026|动态)",
 "M17 图片/影音资源": r"(图片|照片|写真|视频|原文|在线|高清|壁纸)",
}
RX = {k: re.compile(v) for k, v in M.items()}
rows = []
out = open(f"{SP}/sva_dom_ppl_s2.txt", "w", encoding="utf-8")
def P(*x): print(*x, file=out, flush=True)
P("#### n (user tier): " + " | ".join(f"{k} {len(Q[k]):,}" for k in SURF))
P("#### 长度 (median / p90 / >10字% / >20字%)")
for k in SURF:
    L = np.array([len(x) for x in Q[k]]); P(f"  {k}: {np.median(L):.0f} / {np.percentile(L,90):.0f} / {(L>10).mean()*100:.1f}% / {(L>20).mean()*100:.1f}%")
rng = np.random.default_rng(5)
for name, rx in RX.items():
    P(f"\n#### {name}   pattern: {M[name]}")
    hit = {}
    for k in SURF:
        h = np.array([bool(rx.search(x)) for x in Q[k]]); hit[k] = h; n = len(h); kk = int(h.sum()); p, lo, hi = wilson(kk, n)
        pvs = float((PV[k]*h).sum()/PV[k].sum()) if k != "助手random1k" else float("nan")
        rows.append(dict(metric=name, surface=k, n=n, k=kk, share=p, lo=lo, hi=hi, pv_share=pvs))
        idx = np.where(h)[0]
        if k in ("搜索top1000", "助手top1k"): ex = [Q[k][i] for i in sorted(idx, key=lambda i: -PV[k][i])[:10]]
        else: ex = [Q[k][i] for i in rng.choice(idx, min(10, len(idx)), replace=False)] if len(idx) else []
        P(f"  {k:<10} {p*100:5.1f}% [{lo*100:4.1f}–{hi*100:4.1f}] k={kk:>4}/{n}" + (f"  PV{pvs*100:5.1f}%" if k != "助手random1k" else "") + "  例: " + " | ".join(x[:30] for x in ex))
    d, lo, hi = newcombe(int(hit["搜索top1000"].sum()), len(hit["搜索top1000"]), int(hit["助手top1k"].sum()), len(hit["助手top1k"]))
    P(f"  Δ 搜索top1000→助手top1k {d*100:+.1f}pp [{lo*100:+.1f},{hi*100:+.1f}]")
pd.DataFrame(rows).to_csv(f"{SP}/sva_dom_ppl_shape.csv", index=False, encoding="utf-8-sig")
# ---- person names (jieba nr) and whether they exist anywhere in search top10k
P("\n#### 人名覆盖 (jieba.posseg nr, len>=2): 含人名的行中, 所有人名都不作为子串出现在搜索top10k任何query里的比例")
S10 = "\n".join(Q["搜索top10k"])
def names(q): return sorted({w.word for w in pseg.cut(q) if w.flag == "nr" and len(w.word) >= 2})
def cover(qs, ref):
    has = 0; absent = 0; multi = 0; ex_abs = []; ex_multi = []
    for q in qs:
        nm = names(q)
        if not nm: continue
        has += 1
        if len(nm) >= 2: multi += 1; ex_multi.append(q)
        if all(x not in ref for x in nm): absent += 1; ex_abs.append(q)
    return has, absent, multi, ex_abs, ex_multi
# control: deepest 982 search rows vs search rows 1..9000
deep = Q["搜索top10k"][9000:]; ref_ctrl = "\n".join(Q["搜索top10k"][:9000])
nr_rows = []
for tag, qs, ref in (("对照 搜索9001+→搜索1–9000", deep, ref_ctrl), ("助手top1k→搜索top10k", Q["助手top1k"], S10), ("助手random1k→搜索top10k", Q["助手random1k"], S10), ("搜索top1000→搜索1001+", Q["搜索top10k"][:1000], "\n".join(Q["搜索top10k"][1000:]))):
    has, ab, mu, exa, exm = cover(qs, ref); n = len(qs)
    p1, l1, h1 = wilson(has, n); p2, l2, h2 = wilson(ab, has); p3, l3, h3 = wilson(mu, n)
    nr_rows.append(dict(view=tag, n=n, rows_with_name=has, share_with_name=p1, all_names_absent=ab, absent_share=p2, absent_lo=l2, absent_hi=h2, multi_name=mu, multi_share=p3, multi_lo=l3, multi_hi=h3))
    P(f"  {tag}: n={n} 含人名 {has} ({p1*100:.1f}%) | 人名全不在参照集 {ab}/{has} = {p2*100:.1f}% [{l2*100:.1f}–{h2*100:.1f}] | ≥2个人名 {mu}/{n} = {p3*100:.1f}% [{l3*100:.1f}–{h3*100:.1f}]")
    P("     不在参照集 例: " + " | ".join(x[:28] for x in exa[:14]))
    P("     ≥2人名 例: " + " | ".join(x[:28] for x in exm[:10]))
pd.DataFrame(nr_rows).to_csv(f"{SP}/sva_dom_ppl_names.csv", index=False, encoding="utf-8-sig")
out.close()
