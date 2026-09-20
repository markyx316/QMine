# -*- coding: utf-8 -*-
"""医疗 deep-dive, sections 1-3: tiers & cleaning impact, matched-depth shape measures, unified intent + own taxonomies.
All regexes are applied with Python `re` over lists (never pandas .str.contains: pyarrow RE2 has ASCII \\w and no lookbehind)."""
import sys, re, math, os
import numpy as np, pandas as pd
SP = __import__("os").environ.get("SVA_WORKDIR", __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "work"))
sys.path.insert(0, SP)
import sva_common as C

OUT = open(f"{SP}/sva_dom_med_core.txt", "w", encoding="utf-8")
def P(*x): print(*x, file=OUT, flush=True)
DOM = "医疗"
fm = C.frame(); CODES = list(fm.FRAME); NAME = {k: v[0] for k, v in fm.FRAME.items()}

def wilson(k, n, z=1.96):
    if n == 0: return float("nan"), float("nan"), float("nan")
    p = k / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, c - h), min(1.0, c + h)
def newcombe(k1, n1, k2, n2):
    p1, l1, u1 = wilson(k1, n1); p2, l2, u2 = wilson(k2, n2); d = p2 - p1
    return d, d - math.sqrt((p2 - l2) ** 2 + (u1 - p1) ** 2), d + math.sqrt((u2 - p2) ** 2 + (p1 - l1) ** 2)
def ci(k, n): p, lo, hi = wilson(k, n); return f"{p*100:.1f}% [{lo*100:.1f}–{hi*100:.1f}]"
def match(qs, pat):
    R = re.compile(pat); return np.fromiter((bool(R.search(q)) for q in qs), bool, len(qs))

# ---- privacy filter for quoting (conservative: anything sexual/genital/STD/abortion, phone-like digits, named staff, self-intro)
UNSAFE = re.compile(r"(阳痿|早泄|勃起|性功能|性生活|同房|房事|做爱|射精|精子|壮阳|延时|时间太短|达泊西汀|西地那非|他达拉非|司美那非|伟哥|男科|生殖|私处|私密|下体|下身|阴|睾丸|包皮|龟头|宫颈|白带|肛|痔|屁|胸|乳|奶|内裤|裸|色|性感|巨乳|梅毒|淋病|湿疣|hpv|HPV|艾滋|性病|人流|流产|打胎|药流|避孕|腿|臀|布料|前襟|身体|姿势|\d{7,}|医生|大夫|主任|教授|老师|团队|专家|我是|叫)")
def safe(q): return not UNSAFE.search(q)
def tag(q, L=40): return (q[:L] if safe(q) else "⚠" + q[:L])

# ---------------- data
F = pd.read_parquet(f"{SP}/sva_final_rows.parquet"); F["query"] = F["query"].astype(str)
M = F[F.domain == DOM].copy()
a, s = C.load()
cl = C.cells(a, s, DOM)
s10 = cl["搜索top10k"].reset_index(drop=True)
S10 = pd.DataFrame({"surface": "search_top10k", "query": s10["query"].astype(str).values, "pv": s10.wise_pv.astype(float).values,
                    "u_ds": np.nan, "p_ds": np.nan, "own_intent": s10.td_l1_name.values, "own_leaf": s10.bu_leaf_name.values,
                    "l2": None, "rank": np.arange(1, len(s10) + 1)})
use10k = False
F2p = f"{SP}/sva_final_rows2.parquet"
if os.path.exists(F2p):
    F2 = pd.read_parquet(F2p); F2["query"] = F2["query"].astype(str)
    x = F2[(F2.domain == DOM) & (F2.surface == "search_top10k") & (F2.tier == "user")]
    cov = x.u_ds.notna().mean() if len(x) else 0
    P(f"[rows2] exists; search_top10k user n={len(x):,} u_ds coverage {cov*100:.1f}%")
    if cov >= 0.95:
        use10k = True
        S10 = S10.drop(columns=["u_ds", "p_ds"]).merge(x.drop_duplicates("query")[["query", "u_ds", "p_ds"]], on="query", how="left")
else:
    P("[rows2] sva_final_rows2.parquet NOT present -> search_top10k described by markers + own taxonomy only")

cols = ["surface", "query", "pv", "u_ds", "p_ds", "own_intent", "own_leaf", "l2", "rank"]
U = pd.concat([M[M.tier == "user"][cols], S10[cols]], ignore_index=True)
SURFS = ["search_top1000", "search_top10k", "assistant_top1k", "assistant_random1k"]
SL = {"search_top1000": "搜索1k", "search_top10k": "搜索10k", "assistant_top1k": "助手头", "assistant_random1k": "助手尾"}
HEADS = {"search_top1000", "search_top10k", "assistant_top1k"}
def cell(sf): return U[U.surface == sf]
P("\n#### 0. n (user rows) & label coverage")
for sf in SURFS:
    x = cell(sf); P(f"  {SL[sf]}: n={len(x):,} u_ds coverage {x.u_ds.notna().mean()*100:.1f}% PV sum {x.pv.sum():,.0f} PV min {x.pv.min():,.0f} median {x.pv.median():,.0f}")

# ============ SECTION 1
P("\n######## SECTION 1: tiers & cleaning impact")
am = a[a.l1 == "医疗"].copy(); am["query"] = am["query"].astype(str)
trows = []
for snap in ["top1k", "random1k"]:
    x = am[am.snapshot == snap]; tot = x.search_num.sum()
    g = x.groupby("tier").agg(rows=("query", "size"), pv=("search_num", "sum"))
    P(f"\n  assistant {snap}: all rows {len(x):,}, PV {tot:,}")
    for t, r in g.iterrows():
        P(f"    {t:<20} rows {r.rows:>4} ({r.rows/len(x)*100:5.1f}%)  PV {r.pv:>7,} ({r.pv/tot*100:5.1f}%)")
        trows.append(dict(surface=f"assistant_{snap}", tier=t, rows=int(r.rows), row_share=r.rows / len(x), pv=int(r.pv), pv_share=r.pv / tot))
    for t in g.index:
        if t == "user": continue
        e = x[x.tier == t].sort_values("search_num", ascending=False)
        P(f"      e.g. {t}: " + " | ".join(f"{tag(q,50)}({pv:,})" for q, pv in zip(e["query"].head(6), e.search_num.head(6))))
sm = s[s.domain == DOM].sort_values("wise_pv", ascending=False).reset_index(drop=True)
span = M[M.surface == "search_top1000"]
P(f"\n  search top1000 span: rows {len(span):,}; tiers " + str(span.tier.value_counts().to_dict()))
g = sm.groupby("tier").agg(rows=("query", "size"), pv=("wise_pv", "sum")); tot = sm.wise_pv.sum()
P(f"  search full export (10k): rows {len(sm):,}, PV {tot:,}")
for t, r in g.iterrows():
    P(f"    {t:<20} rows {r.rows:>5} ({r.rows/len(sm)*100:5.2f}%)  PV {r.pv:>9,} ({r.pv/tot*100:5.2f}%)")
    trows.append(dict(surface="search_top10k", tier=t, rows=int(r.rows), row_share=r.rows / len(sm), pv=int(r.pv), pv_share=r.pv / tot))
e = sm[sm.tier != "user"]; P("      e.g. " + " | ".join(f"{q}(#{rk},{pv})" for q, rk, pv in zip(e["query"], e["rank"], e.wise_pv)))
pd.DataFrame(trows).to_csv(f"{SP}/sva_dom_med_tiers.csv", index=False, encoding="utf-8-sig")

Q = r"(怎么|如何|为什么|什么|哪个|哪家|哪些|多少|吗|呢|是不是|能不能|可不可以|\?|？)"
x_all = am[am.snapshot == "top1k"]; x_u = x_all[x_all.tier == "user"]
P("\n  cleaning sensitivity, assistant top1k (all tiers -> user tier): row% / PV%")
for nm, pat in [("疑问句", Q), ("功效/作用/主治", r"(功效|作用|主治)"), ("症状/原因", r"(症状|原因|怎么回事)")]:
    def rp(x):
        mk = match(x["query"].tolist(), pat); w = x.search_num.values.astype(float)
        return mk.mean() * 100, (w * mk).sum() / w.sum() * 100
    r1, p1 = rp(x_all); r2, p2 = rp(x_u)
    P(f"    {nm}: rows {r1:.1f}->{r2:.1f} ; PV {p1:.1f}->{p2:.1f}")
P(f"    median length: {x_all['query'].str.len().median():.0f} -> {x_u['query'].str.len().median():.0f}")

# ============ SECTION 2
P("\n######## SECTION 2: matched-depth shape (row% [Wilson]; heads also PV-weighted)")
FAMILY = r"(孩子|儿子|女儿|宝宝|宝贝|婴儿|新生儿|小孩|男童|女童|男孩|女孩|老公|老婆|丈夫|妻子|男朋友|女朋友|男友|女友|我妈|我爸|妈妈|爸爸|母亲|父亲|父母|老人|家人|婆婆|公公|爷爷|奶奶|外婆|姥姥|对象|闺女)"
LAB = r"(\d+(\.\d+)?\s*(mmol|mg|ml|g/l|u/l|iu|μ|微克|毫克|毫升|克|mm|cm|毫米|厘米|公分|度|℃|斤|公斤|kg|%|次/分))|((血糖|血压|尿酸|血脂|胆固醇|甘油三酯|肌酐|转氨酶|白细胞|血小板|心率|hcg|HCG|孕酮|TSH|tsh|C反应蛋白|crp|CRP|血红蛋白|体温|低压|高压|指标|结节)[^，。,]{0,6}\d)"
DUR = r"(\d+\s*个?(天|周|星期|月|年|小时))|([两三四五六七八九十半好几]个?(天|周|星期|月|年)(了|多|左右|以来|之后|后|前|内))"
AGE = r"(\d{1,3}\s*(岁|周岁)|\d{1,2}\s*个月大|[一二三四五六七八九十]{1,3}岁)"
COMBO = r"(一起吃|一起服用|一起用|同时吃|同时服用|同时用|同服|同吃|合用|联用|并用|搭配(吃|服用|使用)|错开|间隔|隔多久|和.{1,15}(能|可以|能否|可否)(一起|同时)|(能|可以)和.{1,15}(一起|同时))"
TIMING = r"(饭前|饭后|空腹|睡前|早上吃|晚上吃|什么时候吃|什么时间吃|几点吃|吃多久|吃多长时间|连续(吃|服用|用)|长期(吃|服用)|停药|漏服|用量|用法|剂量|一次吃|一天吃|每次吃|每天吃|吃几(片|粒|颗|袋|次)|几片|几粒|多少毫克|吃多少|服用方法|怎么服用)"
HOSP = r"(医院|诊所|门诊|专科|卫生院|卫生服务中心|体检中心|科室|挂.{0,3}科|看什么科|哪个科|医生|大夫|专家|名医|中医馆|药店|药房)"
COST = r"(多少钱|费用|价格|收费|价目|报销|医保|贵不贵)"
SENS = r"(阳痿|早泄|勃起|性功能|性生活|同房|房事|时间太短|壮阳|延时|达泊西汀|西地那非|他达拉非|司美那非|伟哥|男科|生殖|私处|私密|下体|下身|阴囊|睾丸|包皮|龟头|阴茎|外阴|阴道|宫颈|白带|肛门|肛周|肛裂|肛瘘|痔疮|阴虱|阴毛|梅毒|淋病|尖锐湿疣|湿疣|hpv|HPV|艾滋|hiv|HIV|性病|衣原体|支原体|人流|流产|打胎|药流|避孕|狐臭|腋臭|腋下异味|腋下净味|口臭|体臭|体味|脚臭)"
PREG = r"(怀孕|怀上|备孕|孕妇|孕期|孕|试管|排卵|生孩子|生育|不孕|不育|产后)"
MENTAL = r"(抑郁|焦虑|精神病|精神分裂|双相|双向情感|躁郁|强迫症|自闭|多动症)"
IMG_GIVE = r"(图中|图片中|图里|图上|图片上|这张图|这张照片|照片中|照片里|看图|帮我看看|这是什么(病|药|药材|中药|菌|菌子|咬的|症状|皮肤病|虫|东西|草|植物)|解读.{0,6}(报告|单|说明书)|报告单|化验单)"
IMG_WANT = r"(图片|图解|示意图|位置图|结构图|样子图|图大全|图全身|图$|视频演示)"
CTX = (r"^(这|那个|它|图中|图片)|^(吃|用)什么(药膏?|药)[？?]?$|^(怎么|如何)(治疗?|调理|服用|食用|预约|取下来|处理|办)[？?]?$|"
       r"^(用法用量|服用方法|功效与作用|功效作用与主治|营养价值|药用价值|副作用)[？?]?$|^(一次|一天|每次|每天)吃几(粒|片|次)[？?]?$|^饭前(吃)?还是饭后吃[？?]?$|"
       r"^有(什么)?(功效|作用|副作用|好处|药用价值)[？?]?$|^什么(功效|病|药)[？?]?$|^(严重|正常|有用|有效|要紧)(吗|不)[？?]?$|^需要(手术|吃药|吃消炎药|治疗|去医院)吗[？?]?$|"
       r"^会传染吗[？?]?$|^孕妇可以用吗[？?]?$|^(能|可以)减肥吗[？?]?$|^(热量多少|多少大卡|能长多大|是激素药吗|是医疗器械吗|有副作用吗|多少钱一支)[？?]?$")
BODY = (r"((双腿|两腿|腿|身体|屁股|臀|胸口|胸部|胸|奶|内裤|布料|肚子|肌肉|双手|手|前襟|腰)(部)?(再|要|变得?|更加|超级|最)?.{0,3}"
        r"(张开|叉开|分开|弯曲|抬高|抬起|并拢|弯下来|变大|变小|变细|变窄|窄小|放大|减少|打开|坚挺|圆润|背后|下垂|平躺|摆正|转正|平行|大一点|巨大|变胖))|"
        r"变成孕妇肚子|变胖\d+斤|胸波|巨乳|^胸大$|身材微胖|正面平躺|眼睛睁开")
EXAM = r"(下列|以下(哪|说法|选项)|正确的是|错误的是|不属于|的是[？?]?$|选项|单选|多选|判断题)"
JUDGE = r"(严重吗|要紧吗|正常吗|正常不|正常嘛|有事吗|有问题吗|有影响吗|需要.{0,6}吗|要不要|该不该|能不能|可不可以|可以.{0,10}吗|能.{0,10}吗|会不会|是不是|有必要|有用吗|有效吗)"
PAT = {
    "疑问句(sva_metrics)": Q,
    "模板:功效/作用/主治": r"(功效|作用|主治|营养价值|药用价值)",
    "模板:症状/表现/前兆": r"(症状|表现|前兆|征兆)",
    "模板:原因/怎么回事": r"(原因|引起|造成|导致|怎么回事|咋回事|为什么|为啥)",
    "模板:定义(什么意思/是什么)": r"(什么意思|是什么东西|是什么病|是指什么|是什么$)",
    "速效根治词": r"(最快|最有效|根治|断根|除根|特效|好得快|立马|快速)",
    "判断请求(严重吗/能…吗/需要…吗)": JUDGE,
    "第一人称": r"(我|本人|俺|咱)",
    "家人代问": FAMILY,
    "年龄自述": AGE,
    "数值+单位/指标带数值": LAB,
    "病程时长": DUR,
    "联合用药": COMBO,
    "服药时间/剂量": TIMING,
    "就医机构/科室/医生": HOSP,
    "费用/价格": COST,
    "隐私难言(性功能/私处/性病/人流避孕/体味)": SENS,
    "孕育(怀孕/备孕/试管/不孕)": PREG,
    "精神心理": MENTAL,
    "给出图片/指代图中": IMG_GIVE,
    "索要图片": IMG_WANT,
    "无主语或指代追问(依赖上文/图片)": CTX,
    "身体/姿态编辑指令(非医疗)": BODY,
    "考试题式": EXAM,
}
shape = []
P("\n  length: median / p90 / ≤4字% / ≥20字%")
for sf in SURFS:
    L = cell(sf)["query"].map(len); n = len(L)
    P(f"    {SL[sf]} n={n:,}: {L.median():.0f} / {L.quantile(.9):.0f} / ≤4 {ci(int((L<=4).sum()),n)} / ≥20 {ci(int((L>=20).sum()),n)}")
    shape.append(dict(metric="len_median", surface=sf, n=n, share=L.median()))
    shape.append(dict(metric="len_le4", surface=sf, n=n, k=int((L <= 4).sum()), share=(L <= 4).mean()))
for name, pat in PAT.items():
    line = []
    for sf in SURFS:
        x = cell(sf); mk = match(x["query"].tolist(), pat); n = len(x); k = int(mk.sum()); p, lo, hi = wilson(k, n)
        w = x.pv.values.astype(float); pvs = (w * mk).sum() / w.sum() if sf in HEADS else np.nan
        shape.append(dict(metric=name, surface=sf, n=n, k=k, share=p, lo=lo, hi=hi, pv_share=pvs))
        line.append(f"{SL[sf]} {p*100:5.1f}[{lo*100:.1f}–{hi*100:.1f}]k={k}" + (f" PV{pvs*100:.1f}" if sf in HEADS else ""))
    P(f"\n  {name}:\n    " + " | ".join(line))
    for sf in ["search_top1000", "search_top10k", "assistant_top1k", "assistant_random1k"]:
        x = cell(sf); mk = match(x["query"].tolist(), pat); y = x[mk]
        if len(y):
            top = y.sort_values("pv", ascending=False)["query"].head(5).tolist(); rnd = y.sample(min(5, len(y)), random_state=3)["query"].tolist()
            P(f"     例[{SL[sf]}] top: " + " | ".join(tag(q) for q in top) + " || rnd: " + " | ".join(tag(q) for q in rnd))
pd.DataFrame(shape).to_csv(f"{SP}/sva_dom_med_shape.csv", index=False, encoding="utf-8-sig")

P("\n  ---- question-rate decomposition")
bands = [(0, 4, "≤4"), (5, 8, "5–8"), (9, 12, "9–12"), (13, 9999, "≥13")]
BW = {}
for sf in SURFS:
    x = cell(sf); qs = x["query"].tolist(); L = np.array([len(q) for q in qs]); qm = match(qs, Q); n = len(qs)
    parts = []; BW[sf] = {}
    for lo, hi, lab in bands:
        b = (L >= lo) & (L <= hi); nb = int(b.sum()); BW[sf][lab] = (nb / n, (qm[b].mean() if nb else float("nan")))
        parts.append(f"{lab}: 行占{nb/n*100:.1f}% 问{(qm[b].mean()*100 if nb else float('nan')):.1f}% (n={nb})")
    P(f"    {SL[sf]} overall 问 {ci(int(qm.sum()),n)} | " + " ; ".join(parts))
for tgt in ["assistant_top1k", "assistant_random1k"]:
    std = sum(BW["search_top1000"][lab][0] * (BW[tgt][lab][1] if not math.isnan(BW[tgt][lab][1]) else 0) for _, _, lab in bands)
    P(f"    {SL[tgt]} question rate standardised to 搜索1k length mix: {std*100:.1f}%")
for sf in ["search_top1000", "assistant_top1k", "assistant_random1k"]:
    x = cell(sf); qs = x["query"].tolist(); qm = match(qs, Q)
    for nm, sel in [("U02+U13 excluded", ~x.u_ds.isin(["U02", "U13"]).values), ("U02 excluded", (x.u_ds != "U02").values),
                    ("body-edit excluded", ~match(qs, BODY)), ("≥5字", np.array([len(q) >= 5 for q in qs]))]:
        P(f"    {SL[sf]} 问 on {nm}: {ci(int((qm & sel).sum()), int(sel.sum()))} (n={int(sel.sum())})")
# where do search's questions come from?
for sf in ["search_top1000", "assistant_top1k"]:
    x = cell(sf); qs = x["query"].tolist(); qm = match(qs, Q); nq = int(qm.sum())
    tpl = match(qs, r"(症状|表现|前兆|原因|引起|怎么回事|什么意思|是什么|功效|作用|主治|正常值|正常范围)")
    ctx = match(qs, CTX); body = match(qs, BODY)
    P(f"    {SL[sf]}: question rows {nq}; of them template-suffix {ci(int((qm&tpl).sum()),nq)}; ctx-fragment {ci(int((qm&ctx).sum()),nq)}")
    P(f"      template-suffix rows {int(tpl.sum())} ({tpl.mean()*100:.1f}% of rows), question rate inside {ci(int((qm&tpl).sum()),int(tpl.sum()))}; non-template question rate {ci(int((qm&~tpl).sum()),int((~tpl).sum()))}")
    P(f"      功效/作用/主治 rows {int(match(qs,r'(功效|作用|主治)').sum())}, of which question-marked {int((qm & match(qs,r'(功效|作用|主治)')).sum())}")
    for tok in ["什么", "哪些", "怎么", "多少", "吗", "？|\\?"]:
        P(f"      token {tok}: {int(match(qs,tok).sum())} rows")
P("    search rank-band control (search top10k user rows): question rate")
x = cell("search_top10k"); qm = match(x["query"].tolist(), Q); rk = x["rank"].values
for lo, hi in [(1, 1000), (1001, 3000), (3001, 6000), (6001, 10**6)]:
    b = (rk >= lo) & (rk <= hi); P(f"      rank {lo}-{hi}: {ci(int(qm[b].sum()), int(b.sum()))} n={int(b.sum())}")

P("\n  ---- composition of assistant U13 / U02 (rows, PV)")
for sf in ["assistant_top1k", "assistant_random1k"]:
    for code in ["U13", "U02"]:
        x = cell(sf); x = x[x.u_ds == code]; qs = x["query"].tolist(); n = len(qs); w = x.pv.values.astype(float)
        body = match(qs, BODY); ctx = match(qs, CTX) & ~body; img = match(qs, IMG_GIVE) & ~body & ~ctx
        hosp = match(qs, HOSP) & ~body & ~ctx; sens = match(qs, SENS) & ~body & ~ctx & ~hosp
        other = ~(body | ctx | img | hosp | sens)
        P(f"    {SL[sf]} {code} n={n} PV {w.sum():,.0f}: " + " ; ".join(f"{nm} {int(mk.sum())} ({mk.mean()*100:.1f}% rows, {(w*mk).sum()/max(w.sum(),1)*100:.1f}% PV)"
              for nm, mk in [("身体编辑", body), ("无主语追问", ctx), ("给图", img), ("机构名", hosp), ("隐私话题", sens), ("其他", other)]))
        P("      其他 e.g. " + " | ".join(tag(q, 20) for q in x[other].sort_values("pv", ascending=False)["query"].head(12)))
xh = cell("assistant_top1k"); qs = xh["query"].tolist(); w = xh.pv.values.astype(float)
for nm, pat in [("身体编辑", BODY), ("无主语追问", CTX)]:
    mk = match(qs, pat); P(f"    助手头 all rows {nm}: {ci(int(mk.sum()),len(qs))} PV {(w*mk).sum()/w.sum()*100:.1f}% ; u_ds of matches: " + str(xh[mk].u_ds.value_counts().to_dict()))
    P("      all matches: " + " | ".join(tag(q, 16) for q in xh[mk].sort_values("pv", ascending=False)["query"]))
P("    助手头 body-edit matches by l2: " + str(xh[match(qs, BODY)].l2.value_counts().to_dict()))
P("    助手头 l2 distribution: " + str(xh.l2.value_counts().to_dict()) + " ; 助手尾: " + str(cell("assistant_random1k").l2.value_counts().to_dict()))

P("\n  ---- sensitive topics × hospital/cost; by l2")
for sf in SURFS:
    x = cell(sf); qs = x["query"].tolist(); n = len(qs); w = x.pv.values.astype(float)
    sens = match(qs, SENS); hosp = match(qs, HOSP); cost = match(qs, COST); choice = match(qs, r"(哪家|哪个医院|哪里|正规|医院好|成功率|排名|口碑|靠谱|专科)")
    care = hosp | cost | choice
    P(f"    {SL[sf]} n={n}: 隐私话题 {ci(int(sens.sum()),n)}" + (f" PV{(w*sens).sum()/w.sum()*100:.1f}" if sf in HEADS else "")
      + f" | 就医/费用/选择 {ci(int(care.sum()),n)} | 隐私∩就医/费用/选择 {ci(int((sens&care).sum()),n)} | 就医/费用/选择 rows 中隐私占 {ci(int((sens&care).sum()),int(care.sum()))}"
      + f" | 隐私话题中带就医/费用/选择 {ci(int((sens&care).sum()),int(sens.sum()))}")
    for sub, sp in [("性功能", r"(阳痿|早泄|勃起|性功能|性生活|同房|房事|时间太短|壮阳|延时|达泊西汀|西地那非|他达拉非|司美那非|伟哥|男科)"),
                    ("私处/肛肠", r"(生殖|私处|私密|下体|下身|阴囊|睾丸|包皮|龟头|阴茎|外阴|阴道|宫颈|白带|肛门|肛周|肛裂|肛瘘|痔疮|阴虱|阴毛)"),
                    ("性传播感染", r"(梅毒|淋病|尖锐湿疣|湿疣|hpv|HPV|艾滋|hiv|HIV|性病|衣原体|支原体)"),
                    ("人流避孕", r"(人流|流产|打胎|药流|避孕)"), ("体味", r"(狐臭|腋臭|腋下异味|腋下净味|口臭|体臭|体味|脚臭)")]:
        mk = match(qs, sp); P(f"       {sub}: {ci(int(mk.sum()),n)}" + (f" PV{(w*mk).sum()/w.sum()*100:.1f}" if sf in HEADS else "") + "  e.g. " + " | ".join(tag(q, 22) for q in x[mk].sort_values("pv", ascending=False)["query"].head(4)))
x = cell("search_top10k"); qs = x["query"].tolist(); sens = match(qs, SENS); rk = x["rank"].values
P("    search top10k sensitive share by rank band: " + " ; ".join(f"{lo}-{hi}: {ci(int(sens[(rk>=lo)&(rk<=hi)].sum()), int(((rk>=lo)&(rk<=hi)).sum()))}" for lo, hi in [(1, 1000), (1001, 3000), (3001, 6000), (6001, 10**6)]))
x = cell("search_top10k"); qs = x["query"].tolist(); sens = match(qs, SENS); hosp = match(qs, HOSP)
P("    search top10k: sensitive ∩ hospital e.g. " + " | ".join(tag(q, 24) for q in x[sens & hosp].sort_values("pv", ascending=False)["query"].head(10)) + f" (n={int((sens&hosp).sum())})")
P("    search top10k: hospital rows e.g. " + " | ".join(tag(q, 24) for q in x[hosp].sort_values("pv", ascending=False)["query"].head(15)) + f" (n={int(hosp.sum())})")
x = cell("assistant_top1k"); qs = x["query"].tolist()
P("    助手头 sensitive by l2: " + str(x[match(qs, SENS)].l2.value_counts().to_dict()))
P("    助手头 sensitive u_ds: " + str(x[match(qs, SENS)].u_ds.value_counts().to_dict()) + " ; search1k sensitive u_ds: " + str(cell("search_top1000")[match(cell("search_top1000")["query"].tolist(), SENS)].u_ds.value_counts().to_dict()))
P("    助手头 hospital rows (all): " + " | ".join(f"{tag(q,22)}({pv:.0f})" for q, pv in x[match(qs, HOSP)].sort_values("pv", ascending=False)[["query", "pv"]].itertuples(index=False)))

# ============ SECTION 3
P("\n######## SECTION 3: unified intent (u_ds)")
SI = [sf for sf in SURFS if cell(sf).u_ds.notna().mean() >= 0.95]
P("  surfaces with ≥95% u_ds coverage: " + ", ".join(SL[x] for x in SI))
SH = []
for view in ["all", "interpretable"]:
    P(f"\n  == view={view} ==")
    for sf in SI:
        x = cell(sf).dropna(subset=["u_ds"])
        if view == "interpretable": x = x[x.u_ds != "U13"]
        n = len(x); w = x.pv.values.astype(float)
        for c in CODES:
            mk = (x.u_ds == c).values; k = int(mk.sum()); p, lo, hi = wilson(k, n)
            SH.append(dict(view=view, surface=sf, code=c, name=NAME[c], n=n, k=k, share=p, lo=lo, hi=hi, pv_share=((w * mk).sum() / w.sum() if sf in HEADS else np.nan)))
    for c in CODES:
        if view == "interpretable" and c == "U13": continue
        P(f"   {c} {NAME[c]:<8} " + " | ".join(
            f"{SL[r['surface']]} {r['share']*100:5.1f}[{r['lo']*100:.1f}–{r['hi']*100:.1f}]" + (f" PV{r['pv_share']*100:.1f}" if r['surface'] in HEADS else "")
            for r in SH if r["view"] == view and r["code"] == c))
    P("   n: " + " | ".join(f"{SL[r['surface']]} {r['n']}" for r in SH if r["view"] == view and r["code"] == "U01"))
SHd = pd.DataFrame(SH); SHd.to_csv(f"{SP}/sva_dom_med_intent_shares.csv", index=False, encoding="utf-8-sig")
PAIRS = [("search_top1000", "assistant_top1k"), ("assistant_top1k", "assistant_random1k")] + ([("search_top10k", "assistant_top1k")] if "search_top10k" in SI else [])
DR = []
for view in ["all", "interpretable"]:
    for s1, s2 in PAIRS:
        for c in CODES:
            if view == "interpretable" and c == "U13": continue
            r1 = SHd[(SHd.view == view) & (SHd.surface == s1) & (SHd.code == c)].iloc[0]; r2 = SHd[(SHd.view == view) & (SHd.surface == s2) & (SHd.code == c)].iloc[0]
            d, lo, hi = newcombe(int(r1.k), int(r1.n), int(r2.k), int(r2.n))
            DR.append(dict(view=view, frm=s1, to=s2, code=c, name=NAME[c], p_from=r1.share, p_to=r2.share, diff=d, lo=lo, hi=hi, sig=bool(lo > 0 or hi < 0)))
DRd = pd.DataFrame(DR); DRd.to_csv(f"{SP}/sva_dom_med_intent_diffs.csv", index=False, encoding="utf-8-sig")
for view in ["all", "interpretable"]:
    for s1, s2 in PAIRS:
        x = DRd[(DRd.view == view) & (DRd.frm == s1) & (DRd.to == s2)].copy(); x = x.reindex(x["diff"].abs().sort_values(ascending=False).index).head(7)
        P(f"  [{view}] {SL[s1]}→{SL[s2]}: " + " ; ".join(f"{r.code}{r.name} {r.p_from*100:.1f}→{r.p_to*100:.1f} ({r.diff*100:+.1f} [{r.lo*100:+.1f},{r.hi*100:+.1f}]{'*' if r.sig else ''})" for r in x.itertuples()))
P("\n  personal flag p_ds=1:")
for sf in SI:
    x = cell(sf).dropna(subset=["p_ds"]); k = int(x.p_ds.sum()); P(f"    {SL[sf]}: {ci(k,len(x))} (k={k}, n={len(x)})")
P("\n  examples per code × surface (top-PV 5 + random 5; ⚠ = do not quote)")
for c in CODES:
    for sf in ["search_top1000", "assistant_top1k", "assistant_random1k"]:
        x = cell(sf); x = x[x.u_ds == c]
        if len(x):
            P(f"   {c}{NAME[c]} [{SL[sf]}] n={len(x)} top: " + " | ".join(tag(q, 30) for q in x.sort_values("pv", ascending=False)["query"].head(5))
              + " || rnd: " + " | ".join(tag(q, 30) for q in x.sample(min(5, len(x)), random_state=7)["query"]))

P("\n######## SECTION 3b: own taxonomies")
OT = []
for sf in SURFS:
    x = cell(sf); n = len(x); vc = x.own_intent.value_counts()
    P(f"\n  [{SL[sf]}] n={n}, classes present {len(vc)}")
    for nm, k in vc.head(14).items():
        OT.append(dict(surface=sf, own_intent=nm, k=int(k), n=n, share=k / n))
        sub = x[x.own_intent == nm]
        ud = (sub.u_ds.value_counts(normalize=True).head(3) * 100).round(0).to_dict() if sub.u_ds.notna().any() else {}
        P(f"    {nm:<22} {ci(int(k),n)} k={k}  u_ds top3 {ud}  e.g. " + " | ".join(tag(q, 18) for q in sub.sort_values('pv', ascending=False)['query'].head(4)))
pd.DataFrame(OT).to_csv(f"{SP}/sva_dom_med_own_tax.csv", index=False, encoding="utf-8-sig")
x = cell("assistant_top1k"); qs = x["query"].tolist()
for nm in ["索取或生成成人、色情或暴力内容", "残缺或不可独立解释输入"]:
    sub = x[x.own_intent == nm]; sq = sub["query"].tolist()
    P(f"  助手头 own '{nm}' n={len(sub)}: body-edit {int(match(sq,BODY).sum())}, ctx {int(match(sq,CTX).sum())}, SENS {int(match(sq,SENS).sum())}, contains 'bmi' {int(match(sq,'(?i)bmi').sum())}")
P("\ndone")
OUT.close()
