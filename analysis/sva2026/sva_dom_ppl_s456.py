# -*- coding: utf-8 -*-
"""人物 sections 4–6 on clean_v3 user rows: exact overlap and reverse coverage; wrapping of search bare names;
change types on near-paraphrases; low-similarity themes; non-public-person / private-info counts;
card-follow-up phrasing check; persona evidence. Python `re` only."""
import sys, re, math, collections, numpy as np
SP=__import__("os").environ.get("SVA_WORKDIR", __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "work"))
sys.path.insert(0, SP)
from sva_common import *
import jieba, jieba.posseg as pseg
jieba.setLogLevel(60)
def wilson(k, n, z=1.96):
    if n == 0: return float("nan"), float("nan"), float("nan")
    p = k/n; den = 1+z*z/n; c = (p+z*z/(2*n))/den; h = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den; return p, max(0.0, c-h), min(1.0, c+h)
def fmt(k, n):
    p, lo, hi = wilson(k, n); return f"{k}/{n} = {p*100:.1f}% [{lo*100:.1f}–{hi*100:.1f}]"
a, s = load(); C = cells(a, s, "人物")
S10 = C["搜索top10k"].reset_index(drop=True).copy(); S10["query"] = S10["query"].astype(str)
S1k = S10.head(1000).copy()
A1 = C["助手top1k"].reset_index(drop=True).copy(); A1["query"] = A1["query"].astype(str)
AR = C["助手random1k"].reset_index(drop=True).copy(); AR["query"] = AR["query"].astype(str)
AS = {"助手top1k": A1, "助手random1k": AR}
F = pd.read_parquet(f"{SP}/sva_final_rows.parquet"); F = F[(F.domain == "人物") & (F.tier == "user")].copy(); F["query"] = F["query"].astype(str)
UDS = {(sf, q): u for sf, q, u in zip(F.surface, F["query"], F.u_ds)}
SFK = {"助手top1k": "assistant_top1k", "助手random1k": "assistant_random1k"}
NN = pd.read_parquet(f"{SP}/sva_dom_ppl_nn.parquet"); REV = pd.read_parquet(f"{SP}/sva_dom_ppl_nn_rev.parquet")
out = open(f"{SP}/sva_dom_ppl_s456.txt", "w", encoding="utf-8")
def P(*x): print(*x, file=out, flush=True)
S10TXT = "\n".join(S10["query"])
_nc = {}
def names(q):
    if q not in _nc: _nc[q] = sorted({w.word for w in pseg.cut(q) if w.flag == "nr" and len(w.word) >= 2})
    return _nc[q]
LOCAL = re.compile(r"(县|镇|乡|村|街道|派出所|区委|区政府|中学|小学|一中|二中|三中|五中|学院|大学|医院|公司|集团|分行|支行|局|科委|团队|课题组|实验室|学校|幼儿园)")
ROLE = re.compile(r"(老师|教师|班主任|校长|教授|博士|医生|大夫|研究员|院长|科长|所长|股长|村干部|支书|队长|经理|主管|法人|法定代表人|职工|员工|同学|园长)")
PRIVQ = re.compile(r"(电话|号码|住址|家住|房产|名下|私人|身份证|机主|微信|手机号)")
TEACH = re.compile(r"(老师|教师|班主任|校长|教授|博士|医生|大夫|研究员|院长|园长)")
ADULT = re.compile(r"(性感|胸|臀|比基尼|萝莉|色情|裸|内裤|屁股|大腿|翘|深V|痴女|吉泽|波多野|苍井|三上)")
def nonpublic(q):
    nm = names(q)
    return bool(nm) and bool(ROLE.search(q) or LOCAL.search(q)) and all(x not in S10TXT for x in nm)
def tag(q): return "[P]" if (nonpublic(q) or PRIVQ.search(q)) else ""
# ======================= 4a exact overlap =======================
P("#### 4a. 精确重合 (filter: cells() user 行; 字符串完全相同)")
set10 = set(S10["query"]); set1k = set(S1k["query"])
for k, x in AS.items():
    m10 = x["query"].isin(set10); m1k = x["query"].isin(set1k)
    P(f"  {k}: 在搜索top10k {fmt(int(m10.sum()), len(x))} (该格PV {x.search_num[m10].sum()/x.search_num.sum()*100:.1f}%) | 在搜索top1000 {fmt(int(m1k.sum()), len(x))} (PV {x.search_num[m1k].sum()/x.search_num.sum()*100:.1f}%)")
    P("     例(按助手PV): " + " | ".join(f"{q}(搜索#{int(S10.loc[S10['query']==q,'rank'].iloc[0])})" for q in x[m10].sort_values("search_num", ascending=False)["query"].drop_duplicates().head(12)))
aset1 = set(A1["query"]); asetall = aset1 | set(AR["query"])
for nm, st in (("助手top1k", aset1), ("助手top1k∪random1k", asetall)):
    m = S1k["query"].isin(st); P(f"  反向: 搜索top1000 原样出现在{nm}: {fmt(int(m.sum()), 1000)} (占搜索top1000 PV {S1k.wise_pv[m].sum()/S1k.wise_pv.sum()*100:.1f}%)")
R = S1k.reset_index(drop=True)[["query", "wise_pv", "td_l1_name"]].copy()
assert (REV["query"].values == R["query"].values).all()
R["sim"] = REV["sim"].values; R["nn_query"] = REV["nn_query"].values; R["exact"] = R["query"].isin(asetall)
P("  搜索top1000 按自身类: n / 原样出现在助手(头+尾) / 最近助手邻居 sim 中位 / sim<0.70")
rr = []
for c, g in R.groupby("td_l1_name"):
    rr.append(dict(own_intent=c, n=len(g), exact_share=g.exact.mean(), sim_median=g.sim.median(), low_share=(g.sim < 0.70).mean()))
    P(f"   {c:<14} n={len(g):>3} 原样 {g.exact.mean()*100:5.1f}% | sim中位 {g.sim.median():.2f} | <0.70 {(g.sim<0.70).mean()*100:5.1f}%  例: " + " | ".join(f"{q}→{nq}({sv:.2f})" for q, nq, sv in g.sort_values('wise_pv', ascending=False)[['query','nn_query','sim']].head(4).values))
pd.DataFrame(rr).to_csv(f"{SP}/sva_dom_ppl_reverse_by_class.csv", index=False, encoding="utf-8-sig")
FAN = re.compile(r"(送花|影响力|打榜|应援|超话)")
for k, x in {"搜索top1000": S1k, "搜索top10k": S10, **AS}.items():
    m = x["query"].map(lambda q: bool(FAN.search(q))); pvc = "wise_pv" if "wise_pv" in x.columns else "search_num"
    P(f"  含 送花/影响力/打榜/应援/超话 {k}: {fmt(int(m.sum()), len(x))} PV {x[pvc][m].sum()/x[pvc].sum()*100:.1f}%  例: " + " | ".join(x["query"][m].head(8)))
# ======================= 4b wrapping =======================
P("\n#### 4b. 包装: 搜索top10k 裸名(自身类=裸名身份查询, 全为2–5个汉字) 被原样包含在更长(≥+2字)的助手查询里; 2字核需是 jieba 分词 token")
GENERIC = re.compile(r"^(女|男|美女|年轻的?|漂亮的?|小|老|大)?(老师|教师|教授|医生|演员|歌手|明星|美女|帅哥|人物|女人|男人|女生|男生|女孩|男孩|校长|书记|主席|总理|总统|国王|将军|元帅|皇帝|作者|主持人|模特|网红|博主|主播|大哥|大姐|哥|姐|小姐|同学|老人|富豪|老板|领导|干部|院士|专家|名人|英雄|烈士|空姐|秘书|护士|警察|律师|画家|作家|诗人|演员们)$")
cores = sorted({q for q, t in zip(S10["query"], S10.td_l1_name) if t == "裸名身份查询" and re.fullmatch(r"[一-鿿·]{2,5}", q) and not GENERIC.match(q)}, key=len, reverse=True)
P(f"  cores n={len(cores)}")
def find_core(q):
    toks = None
    for c in cores:
        if c in q and len(q) >= len(c) + 2:
            if len(c) == 2:
                if toks is None: toks = set(jieba.lcut(q))
                if c not in toks: continue
            return c
    return None
S10s = S10.sort_values("wise_pv", ascending=False)
def search_side(c, k=3):
    m = S10s[S10s["query"].map(lambda q: c in q)]
    return " / ".join(f"{q}(#{int(r)})" for q, r in m[["query", "rank"]].head(k).values)
W = []
for k, x in AS.items():
    for q, pv in zip(x["query"], x.search_num):
        c = find_core(q)
        if c: W.append(dict(surface=k, query=q, pv=int(pv), core=c, u_ds=UDS.get((SFK[k], q)), np=bool(tag(q))))
W = pd.DataFrame(W)
CHG = {
 "关系/加第二人": r"(关系|同框|分手|恋|婚|丈夫|妻子|老公|老婆|男友|女友|前夫|前妻|子女|儿子|女儿|家族|家庭|父亲|母亲|感情|[cC][pP]|闺蜜|和.{1,4}谁)",
 "原因/评价/影响": r"(为什么|为何|原因|怎么回事|如何评价|怎么评价|评价|怎么样|如何$|影响|有何|怎么死|为啥)",
 "近况/时效/结局": r"(现在|最新|近况|目前|最近|近日|如今|还在|今年|2026|动态|后来|晚年|结局)",
 "判断/比较/核实": r"(谁更|哪个更|谁最|是否|会不会|可能|能不能|算不算|是不是|真的|吗)",
 "具体属性/数值": r"(多少|几岁|多大|哪年|哪里人|籍贯|任期|身高|生肖|属什么|出生)",
 "职务任免": r"(任职|职务|调任|任免|免职|辞职|辞去|现任|历任|升任|当选|担任|接任|上任|在任|调离|级别)",
 "财富/私生活": r"(身价|资产|财富|房产|名下|收入|个人生活|私生活|有钱|花费)",
 "清单/作品/奖项": r"(有哪些|哪几|名单|代表作|作品|奖项|排名|演过)",
 "生死健康": r"(去世|逝世|离世|死因|病|身体|症状|火化|遗体)",
 "查处司法": r"(被查|落马|违纪|违法|双开|审查|调查|判|被抓|入狱|查封)",
 "资料/简介(与搜索同类)": r"(简介|个人资料|资料|简历|履历|百科|生平|介绍|背景)",
}
CHGR = {k: re.compile(v) for k, v in CHG.items()}
wc = []
for k, x in AS.items():
    w = W[W.surface == k]
    P(f"\n  {k}: 包装行 {fmt(len(w), len(x))}; 其中 [P]可能涉及非公众个人/私密 {int(w.np.sum())}; u_ds: " + " ; ".join(f"{u} {v}" for u, v in w.u_ds.value_counts().items()))
    for t, rx in CHGR.items():
        m = w["query"].map(lambda q: bool(rx.search(q)))
        wc.append(dict(surface=k, change=t, k=int(m.sum()), n_wrapped=len(w), share=int(m.sum())/max(len(w), 1)))
        ex = w[m].sort_values("pv", ascending=False) if k == "助手top1k" else w[m].sample(min(len(w[m]), 40), random_state=4)
        ex = ex.drop_duplicates("query").head(10)
        P(f"   {t:<14} {fmt(int(m.sum()), len(w))}  例: " + " ‖ ".join(f"{tag(q)}{q[:30]} ⇐ {search_side(c, 2)}" for q, c in ex[["query", "core"]].values))
    m2 = w["query"].map(lambda q: len(names(q)) >= 2); P(f"   ≥2个人名(jieba nr) {fmt(int(m2.sum()), len(w))}")
pd.DataFrame(wc).to_csv(f"{SP}/sva_dom_ppl_wrap_changes.csv", index=False, encoding="utf-8-sig")
# ======================= 4c near paraphrase =======================
P("\n#### 4c. 近似改写 (NN 0.70≤sim<0.999): 助手查询含该类词而其最近搜索邻居不含 → '新增了什么'")
for k in AS:
    x = NN[(NN.surface == k) & (NN.sim >= 0.70) & (NN.sim < 0.999)]
    P(f"  {k}: 近似行 {fmt(len(x), int((NN.surface == k).sum()))}")
    for t, rx in CHGR.items():
        m = x["query"].map(lambda q: bool(rx.search(q))) & ~x["nn_query"].map(lambda q: bool(rx.search(q)))
        ex = x[m].sort_values("pv", ascending=False).drop_duplicates("query").head(10)
        P(f"   +{t:<14} {fmt(int(m.sum()), len(x))}  例: " + " ‖ ".join(f"{tag(q)}{q[:28]} ⇐ {nq}(#{r},{sv:.2f})" for q, nq, r, sv in ex[["query", "nn_query", "nn_rank", "sim"]].values))
# ======================= 5 low-sim themes =======================
THEMES = [
 ("T1 指代/识图(依赖图片或上文)", r"(这是谁|这个人|这人|这谁|这个.{0,4}是谁|这.{0,2}(女|男)的?是谁|图中|图片中|照片|他叫|她叫|叫什么名字|像哪个|这是哪|^他|^她|^那|^这|第.个人)"),
 ("T2 虚构/动漫角色与创作", r"(奥特曼|同人|角色|设定|自拟|帮我画|生成|二次元|动漫|扮演|续写|魔|神明|精灵|美少女|初音|喜羊羊|猜.{0,4}名字)"),
 ("T3 历史人物与历史讨论", r"(朝代|皇帝|帝王|皇后|太后|太子|古代|古人|元帅|大将|上将|将军|长征|抗战|民国|清朝|明朝|汉朝|唐朝|宋朝|秦朝|三国|春秋|战国|历史|毛主席|毛泽东|蒋介石|汪精卫|周恩来|朱德|彭德怀|邓小平|嫪毐|曾国藩|袁术|吕后|刘邦|刘恒|孙殿英|刘湘|国民党|红军|鲁班|李白|苏轼|十大元帅|上将)"),
 ("T4 教师/医生/学者等职业个人", r"(老师|教师|班主任|校长|教授|博士|医生|大夫|研究员|院长|团队|课题组|学者|园长)"),
 ("T5 查处与司法", r"(被查|落马|违纪|违法|双开|审查|调查|判刑|判了|判处|无期|死刑|被抓|入狱|获刑|处分|查封|涉案|免职)"),
 ("T6 干部/机构人事与任职", r"(任职|职务|调任|任免|辞职|现任|历任|升任|当选|担任|接任|上任|在任|领导|班子|成员|调离|职级|级别|书记|县长|市长|镇长|局长|科长|所长|主任|常委|部长|厅长|副.{0,4}长|高管|董事长|总经理|法人|法定代表人|工作|单位)"),
 ("T7 名人动态/关系/财富/生死", r"(去世|逝世|离世|死因|病|关系|婚姻|丈夫|妻子|老公|老婆|恋|离婚|结婚|分手|出轨|前夫|前妻|子女|儿子|女儿|家族|家庭|父亲|母亲|感情|个人生活|身价|资产|财富|房产|名下|收入|电话|号码|绯闻|传闻|退圈|综艺|演唱会|新剧|进组|作品|演过|明星|演员|歌手)"),
 ("T8 比较/清单/评判", r"(谁更|哪个更|谁最|有哪些|名单|排名|推荐|十大|都有谁|哪几)"),
]
THR = [(t, re.compile(p)) for t, p in THEMES]
QW = re.compile(r"(怎么|如何|为什么|为何|什么|哪个|哪些|哪里|哪位|哪年|多少|几岁|吗|呢|是不是|是否|谁|\?|？)")
T9 = ["T9a 裸名/短词(≤5字,无问词)", "T9b 陈述式标题/话题串(≥6字,无问词)", "T9c 其他问句"]
def theme(q):
    for t, rx in THR:
        if rx.search(q): return t
    if QW.search(q): return T9[2]
    return T9[0] if len(q) <= 5 else T9[1]
P("\n#### 5a. 低相似 (NN sim<0.70) 规模")
for k in AS:
    x = NN[NN.surface == k]; P(f"  {k}: sim<0.70 {fmt(int((x.sim<0.70).sum()), len(x))}")
th = []
P("\n#### 5b. 主题 (first match, 按上列顺序) — 在各界面全部 user 行上 (搜索用于对照)")
ALLS = {"搜索top1000": S1k["query"].tolist(), "搜索top10k": S10["query"].tolist(), "助手top1k": A1["query"].tolist(), "助手random1k": AR["query"].tolist()}
TH_ALL = {k: [theme(q) for q in v] for k, v in ALLS.items()}
names_t = [t for t, _ in THEMES] + T9
for t in names_t:
    P(f"   {t:<20} " + " | ".join(f"{k} {fmt(sum(1 for z in TH_ALL[k] if z == t), len(TH_ALL[k]))}" for k in ALLS))
P("\n#### 5c. 主题 — 仅低相似行 (sim<0.70)")
for k in AS:
    x = NN[(NN.surface == k) & (NN.sim < 0.70)].copy(); x["theme"] = x["query"].map(theme); x["np"] = x["query"].map(lambda q: bool(tag(q)))
    P(f"\n  {k}: 低相似 n={len(x)}")
    for t in names_t:
        g = x[x.theme == t]
        th.append(dict(surface=k, theme=t, k=len(g), n_low=len(x), share=len(g)/max(len(x), 1), nonpublic_or_private=int(g.np.sum())))
        exq = g[~g.np].sample(min(len(g[~g.np]), 12), random_state=8) if k == "助手random1k" else g[~g.np].sort_values("pv", ascending=False).head(12)
        exp = g[g.np].sample(min(len(g[g.np]), 5), random_state=8)
        P(f"   {t:<20} {fmt(len(g), len(x))}  [P]={int(g.np.sum())}  例: " + " | ".join(f"{q[:32]}(→{nq[:10]},{sv:.2f})" for q, nq, sv in exq[["query", "nn_query", "sim"]].values) + "  ‖ [P]例(仅供改写): " + " | ".join(q[:32] for q in exp["query"]))
pd.DataFrame(th).to_csv(f"{SP}/sva_dom_ppl_lowsim_themes.csv", index=False, encoding="utf-8-sig")
# ======================= 6 privacy / vetting =======================
P("\n#### 6a. 非公众个人 / 私密信息 (filter: user 行)")
pv_rows = []
for k, qs in ALLS.items():
    n = len(qs)
    teach_name = sum(1 for q in qs if TEACH.search(q) and names(q))
    priv = sum(1 for q in qs if PRIVQ.search(q))
    line = f"  {k}: 职业角色词(老师/校长/教授/医生…)+人名 {fmt(teach_name, n)} | 私密信息词(电话/房产/名下/住址…) {fmt(priv, n)}"
    if k.startswith("助手"):
        npf = [q for q in qs if nonpublic(q)]
        tn = [q for q in qs if TEACH.search(q) and names(q) and all(x not in S10TXT for x in names(q))]
        line += f" | 非公众个人flag(人名不在搜索top10k且带机构/角色词) {fmt(len(npf), n)} | 其中职业角色(老师/医生/学者) {fmt(len(tn), n)}"
        pv_rows.append(dict(surface=k, n=n, nonpublic=len(npf), teacher_doctor=len(tn), private_words=priv))
    P(line)
pd.DataFrame(pv_rows).to_csv(f"{SP}/sva_dom_ppl_privacy.csv", index=False, encoding="utf-8-sig")
P("\n  [审计用, 不引用] 助手random1k 非公众个人flag 全部行:")
npr = [q for q in ALLS["助手random1k"] if nonpublic(q)]
for i in range(0, len(npr), 6): P("   " + " | ".join(f"{i+j}:{q[:34]}" for j, q in enumerate(npr[i:i+6])))
# single-reader manual audit: rows judged NOT to name a specific non-public individual
FP_PREFIX = ["有没有乐平市金鹅山中小学的校长名单", "洼里王镇2026年有哪些新任职人员", "贵州财经大学现任领导中谁的籍贯最接近四川",
             "郭干县金川镇瓦桥村委会五小组现在有多少人", "90年代的中后期", "景德镇颜色釉品牌", "李必原型李泌的真实结局是什么", "沈阳铁路局许仲仁简历"]
fp = []
for pfx in FP_PREFIX:
    hits = [q for q in npr if q.startswith(pfx)]
    assert len(hits) == 1, (pfx, hits)
    fp.append(hits[0])
TP = [q for q in npr if q not in fp]
P(f"  人工审计(单人判读; 误报=未指向某个具体非公众个人 {len(fp)} 行) → flag 精度 {fmt(len(TP), len(npr))}; 按精度折算约占 random1k {len(TP)/len(AR)*100:.1f}%")
SUB = {"教育(师生/学校)": r"(老师|教师|班主任|校长|园长|教授|博士|学院|中学|小学|一中|二中|五中|学校|幼儿园|大学|学生|同学|课题组|团队)",
       "政务(基层干部/机关)": r"(县|镇|乡|村|局|派出所|政府|书记|科长|所长|人大|政协|调研员|检察|纪检|区委|组织部)",
       "企业(公司员工/老板)": r"(公司|集团|老板|法人|经理|企业|品牌|煤矿)"}
for nm, pat in SUB.items():
    rx = re.compile(pat); P(f"   子类(可重叠) {nm}: {fmt(sum(bool(rx.search(q)) for q in TP), len(TP))}")
STU = re.compile(r"(学生|同学|初[一二三]|高[一二三]|[一二三四五六七八九]\d?班|报考|考入|毕业生|上学)")
P(f"   涉及学生(可能未成年, 一律不引用): {fmt(sum(bool(STU.search(q)) for q in TP), len(TP))}")
PRIVASK = re.compile(r"(电话|号码|住址|家住|房产|名下|私人|身份证|机主|微信|手机号|婚|子女|恋爱|感情|户口)")
P(f"   其中索要私人信息(电话/房产/婚恋/户口…): {fmt(sum(bool(PRIVASK.search(q)) for q in TP), len(TP))}")
P("\n  U12 行分解 (adult / private-info or nonpublic / gossip / other)")
GOS = re.compile(r"(传闻|黑料|爆料|曝|出轨|私生活|个人生活|花费|前女友|扛事)")
for sf in ("search_top1000", "assistant_top1k", "assistant_random1k"):
    u = F[(F.surface == sf) & (F.u_ds == "U12")]["query"].tolist()
    cats = collections.Counter("adult" if ADULT.search(q) else ("private" if (PRIVQ.search(q) or nonpublic(q)) else ("gossip" if GOS.search(q) else "other")) for q in u)
    P(f"   {sf}: n={len(u)} " + str(dict(cats)) + "  other例: " + " | ".join(q[:28] for q in u if not ADULT.search(q) and not PRIVQ.search(q) and not nonpublic(q) and not GOS.search(q))[:10])
P("\n  U13 / U05 构成")
RECOG = re.compile(r"(这是谁|这个人|这人|这谁|这个.{0,4}是谁|这.{0,2}(女|男)的?是谁|图中|图片中|照片|他叫|她叫|叫什么名字|像哪个|这是哪|作者是谁|演员是谁|有照片吗|哪里人|年龄)")
u13 = F[(F.surface == "assistant_top1k") & (F.u_ds == "U13")]; mr = u13["query"].map(lambda q: bool(RECOG.search(q)))
P(f"   助手top1k U13 行中 指代/识图/缺主语 {fmt(int(mr.sum()), len(u13))} (占U13 PV {u13.pv[mr].sum()/u13.pv.sum()*100:.1f}%)  非此类例: " + " | ".join(u13[~mr].sort_values('pv', ascending=False)['query'].head(10)))
s13u = F[(F.surface == "search_top1000") & (F.u_ds == "U13")]
P(f"   搜索top1000 U13 行中 送花/影响力 {fmt(int(s13u['query'].map(lambda q: bool(FAN.search(q))).sum()), len(s13u))}")
fanu = F[(F.surface == "search_top1000") & F["query"].map(lambda q: bool(FAN.search(q)))]
P(f"   搜索top1000 送花/影响力 行 n={len(fanu)} 的 u_ds 分布: " + " ; ".join(f"{u} {v}" for u, v in fanu.u_ds.value_counts().items()))
u05 = F[(F.surface == "assistant_top1k") & (F.u_ds == "U05")]
stm = u05["query"].map(lambda q: len(q) >= 6 and not QW.search(q))
P(f"   助手top1k U05 行中 陈述式(≥6字且无问词; 可能是 S5 未捕获的话题/标题点击) {fmt(int(stm.sum()), len(u05))} (占U05 PV {u05.pv[stm].sum()/u05.pv.sum()*100:.1f}%)  例: " + " | ".join(u05[stm].sort_values('pv', ascending=False)['query'].head(10)))
# card follow-up phrasing check
P("\n#### 6b. 卡片追问句式检查: S3 卡片段落末尾附带的问句 → 取末5字作为'模板结尾'; 看各界面 user 行以这些结尾收尾的比例")
S3q = a[a.tier == "S3_card_passage"]["query"].astype(str)
tails = []
for q in S3q:
    parts = [p.strip() for p in re.split(r"[。]", q) if p.strip()]
    if len(parts) >= 2:
        t = parts[-1]
        if 4 <= len(t) <= 40 and re.search(r"(吗|什么|哪些|如何|多久|多少|是否|怎么|谁|怎样)", t): tails.append(t)
ends = collections.Counter(t[-5:] for t in tails)
ENDS = {e for e, c in ends.items() if c >= 2}
P(f"  S3 行 n={len(S3q)}; 附带问句 {len(tails)}; 出现≥2次的末5字结尾 {len(ENDS)}: " + " | ".join(f"{e}({c})" for e, c in ends.most_common(25)))
for k, qs in ALLS.items():
    m = [q for q in qs if q[-5:] in ENDS]
    P(f"  {k}: 以卡片模板结尾收尾 {fmt(len(m), len(qs))}  例: " + " | ".join(f"{tag(q)}{q[:26]}" for q in m[:8]))
npq = [q for q in ALLS["助手random1k"] if nonpublic(q)]; oth = [q for q in ALLS["助手random1k"] if not nonpublic(q)]
P(f"  助手random1k 非公众flag行: 模板结尾 {fmt(sum(q[-5:] in ENDS for q in npq), len(npq))} | 其他行 {fmt(sum(q[-5:] in ENDS for q in oth), len(oth))}")
COLLOQ = re.compile(r"(啊|呢|吧|嗯|呀|哦|我|你|咋|啥|那个|这个|？|\?|，|！)")
P(f"  口语/标点标记(啊呢吧嗯我你咋啥？，！): 非公众flag行 {fmt(sum(bool(COLLOQ.search(q)) for q in npq), len(npq))} | random1k其他行 {fmt(sum(bool(COLLOQ.search(q)) for q in oth), len(oth))} | 助手top1k {fmt(sum(bool(COLLOQ.search(q)) for q in ALLS['助手top1k']), len(ALLS['助手top1k']))}")
# ======================= 6c persona evidence =======================
P("\n#### 6c. 粉丝: 在搜索top10k 与 送花/影响力 共现的明星名")
fan_rows = S10[S10["query"].map(lambda q: bool(FAN.search(q)))]
def strip_fan(q): return re.sub(r"(百度百科|百度|明星|百科|送花|影响力|入口|的|个人|\s|介绍|打榜|排名|应援|超话|怎么|如何|在哪)", "", q)
fan_names = sorted({x for x in (strip_fan(q) for q in fan_rows["query"]) if re.fullmatch(r"[一-鿿]{2,4}", x)})
fan_names = [x for x in fan_names if not any(y != x and x.startswith(y) for y in fan_names)]
P(f"  送花/影响力 行 {len(fan_rows)}; 明星名 {len(fan_names)}: " + "、".join(fan_names[:40]))
for k, x in (("搜索top1000", S1k), ("搜索top10k", S10), ("助手top1k", A1), ("助手random1k", AR)):
    pvc = "wise_pv" if "wise_pv" in x.columns else "search_num"
    m = x["query"].map(lambda q: any(nm in q for nm in fan_names)); fw = x["query"].map(lambda q: bool(FAN.search(q)))
    P(f"  {k}: 提及这些明星 {fmt(int(m.sum()), len(x))} (PV {x[pvc][m].sum()/x[pvc].sum()*100:.1f}%) | 其中带送花/影响力词 {fmt(int((m & fw).sum()), int(m.sum()))} | 带百科资料后缀 {fmt(int((m & x['query'].map(lambda q: bool(re.search(r'(简介|资料|简历|百科|介绍|信息)', q)))).sum()), int(m.sum()))}")
    if k.startswith("助手"):
        P("     不带送花词的例: " + " | ".join(x[m & ~fw].sort_values(pvc, ascending=False)["query"].drop_duplicates().head(14).str[:24]))
P("\n#### 6d. 新闻周期集中度: 人名 (jieba nr) 频次 top12, 以及 top5 人名覆盖的行/PV")
for k, x in (("搜索top1000", S1k), ("助手top1k", A1), ("助手random1k", AR)):
    pvc = "wise_pv" if "wise_pv" in x.columns else "search_num"
    cnt = collections.Counter(); pvn = collections.Counter()
    for q, v in zip(x["query"], x[pvc]):
        for nm in names(q): cnt[nm] += 1; pvn[nm] += v
    top = [nm for nm, _ in cnt.most_common(5)]
    m = x["query"].map(lambda q: any(nm in q for nm in top))
    P(f"  {k}: top12 " + " ".join(f"{nm}({c})" for nm, c in cnt.most_common(12)) + f" | top5 覆盖 {fmt(int(m.sum()), len(x))} PV {x[pvc][m].sum()/x[pvc].sum()*100:.1f}%")
CL = {"景甜/孙宇晨": r"(景甜|孙宇晨)", "李维康/陈武(逝世)": r"(李维康|陈武)", "叶建春/覃伟中/朱忠明(任免与查处)": r"(叶建春|覃伟中|朱忠明)", "包贝尔/包文婧": r"(包贝尔|包文婧)", "许家印": r"许家印"}
for cl, pat in CL.items():
    rx = re.compile(pat)
    P(f"  簇 {cl}: " + " | ".join(f"{k} {fmt(sum(bool(rx.search(q)) for q in qs), len(qs))}" for k, qs in ALLS.items()) + "  助手头部例: " + " | ".join(A1[A1['query'].map(lambda q: bool(rx.search(q)))].sort_values('search_num', ascending=False)['query'].drop_duplicates().head(10).str[:22]))
out.close()
