# -*- coding: utf-8 -*-
"""医疗 sections 4-6: exact overlap & wrapping, matched pairs by change type (NN + entity matching),
assistant-only needs (low-sim themes), lead-generation strings, and safety markers. Regexes applied with Python `re`."""
import sys, re, math
import numpy as np, pandas as pd
SP = __import__("os").environ.get("SVA_WORKDIR", __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "work"))
sys.path.insert(0, SP)
import sva_common as C
OUT = open(f"{SP}/sva_dom_med_pairs.txt", "w", encoding="utf-8")
def P(*x): print(*x, file=OUT, flush=True)

def wilson(k, n, z=1.96):
    if n == 0: return float("nan"), float("nan"), float("nan")
    p = k / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, c - h), min(1.0, c + h)
def ci(k, n): p, lo, hi = wilson(int(k), int(n)); return f"{p*100:.1f}% [{lo*100:.1f}–{hi*100:.1f}] (k={int(k)}/n={int(n)})"
def match(qs, pat):
    R = re.compile(pat); return np.fromiter((bool(R.search(q)) for q in qs), bool, len(qs))
def m1(q, pat): return bool(re.search(pat, q))
UNSAFE = re.compile(r"(阳痿|早泄|勃起|性功能|性生活|同房|房事|做爱|射精|精子|壮阳|延时|时间太短|达泊西汀|西地那非|他达拉非|司美那非|伟哥|男科|生殖|私处|私密|下体|下身|阴|睾丸|包皮|龟头|宫颈|白带|肛|痔|屁|胸|乳|奶|内裤|裸|色|性感|巨乳|梅毒|淋病|湿疣|hpv|HPV|艾滋|性病|人流|流产|打胎|药流|避孕|腿|臀|布料|前襟|身体|姿势|\d{7,}|医生|大夫|主任|教授|老师|团队|专家|我是|叫)")
def tag(q, L=40): return (q[:L] if not UNSAFE.search(q) else "⚠" + q[:L])

# ---- regex constants (identical to sva_dom_med_core.py)
Q = r"(怎么|如何|为什么|什么|哪个|哪家|哪些|多少|吗|呢|是不是|能不能|可不可以|\?|？)"
FAMILY = r"(孩子|儿子|女儿|宝宝|宝贝|婴儿|新生儿|小孩|男童|女童|男孩|女孩|老公|老婆|丈夫|妻子|男朋友|女朋友|男友|女友|我妈|我爸|妈妈|爸爸|母亲|父亲|父母|老人|家人|婆婆|公公|爷爷|奶奶|外婆|姥姥|对象|闺女)"
LAB = r"(\d+(\.\d+)?\s*(mmol|mg|ml|g/l|u/l|iu|μ|微克|毫克|毫升|克|mm|cm|毫米|厘米|公分|度|℃|斤|公斤|kg|%|次/分))|((血糖|血压|尿酸|血脂|胆固醇|甘油三酯|肌酐|转氨酶|白细胞|血小板|心率|hcg|HCG|孕酮|TSH|tsh|C反应蛋白|crp|CRP|血红蛋白|体温|低压|高压|指标|结节)[^，。,]{0,6}\d)"
DUR = r"(\d+\s*个?(天|周|星期|月|年|小时))|([两三四五六七八九十半好几]个?(天|周|星期|月|年)(了|多|左右|以来|之后|后|前|内))"
AGE = r"(\d{1,3}\s*(岁|周岁)|\d{1,2}\s*个月大|[一二三四五六七八九十]{1,3}岁)"
COMBO = r"(一起吃|一起服用|一起用|同时吃|同时服用|同时用|同服|同吃|合用|联用|并用|搭配(吃|服用|使用)|错开|间隔|隔多久|和.{1,15}(能|可以|能否|可否)(一起|同时)|(能|可以)和.{1,15}(一起|同时))"
TIMING = r"(饭前|饭后|空腹|睡前|早上吃|晚上吃|什么时候吃|什么时间吃|几点吃|吃多久|吃多长时间|连续(吃|服用|用)|长期(吃|服用)|停药|漏服|用量|用法|剂量|一次吃|一天吃|每次吃|每天吃|吃几(片|粒|颗|袋|次)|几片|几粒|多少毫克|吃多少|服用方法|怎么服用)"
HOSP = r"(医院|诊所|门诊|专科|卫生院|卫生服务中心|体检中心|科室|挂.{0,3}科|看什么科|哪个科|医生|大夫|专家|名医|中医馆|药店|药房)"
COST = r"(多少钱|费用|价格|收费|价目|报销|医保|贵不贵)"
SENS = r"(阳痿|早泄|勃起|性功能|性生活|同房|房事|时间太短|壮阳|延时|达泊西汀|西地那非|他达拉非|司美那非|伟哥|男科|生殖|私处|私密|下体|下身|阴囊|睾丸|包皮|龟头|阴茎|外阴|阴道|宫颈|白带|肛门|肛周|肛裂|肛瘘|痔疮|阴虱|阴毛|梅毒|淋病|尖锐湿疣|湿疣|hpv|HPV|艾滋|hiv|HIV|性病|衣原体|支原体|人流|流产|打胎|药流|避孕|狐臭|腋臭|腋下异味|腋下净味|口臭|体臭|体味|脚臭)"
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

# ---- data
F = pd.read_parquet(f"{SP}/sva_final_rows.parquet"); F["query"] = F["query"].astype(str)
M = F[(F.domain == "医疗") & (F.tier == "user")]
a, s = C.load(); cl = C.cells(a, s, "医疗")
S10 = cl["搜索top10k"].reset_index(drop=True); S10["query"] = S10["query"].astype(str); S10["urank"] = np.arange(1, len(S10) + 1)
lab1k = M[M.surface == "search_top1000"].drop_duplicates("query").set_index("query").u_ds.to_dict()
NN = pd.read_parquet(f"{SP}/sva_dom_med_nn.parquet")
A = NN.merge(M[M.surface.isin(["assistant_top1k", "assistant_random1k"])].drop_duplicates(["surface", "query"])[["surface", "query", "u_ds", "p_ds"]],
             on=["surface", "query"], how="left")
A["u_nn"] = A.nn1.map(lab1k)
SL = {"assistant_top1k": "助手头", "assistant_random1k": "助手尾"}
P(f"NN rows {len(NN)}; u_ds match {A.u_ds.notna().mean()*100:.1f}%; neighbour inside search top1000 (labelled) {A.u_nn.notna().mean()*100:.1f}%")

# ============ A. exact overlap
P("\n######## A. exact overlap (string equality)")
sset = set(S10["query"]); s1k = set(S10["query"].head(1000)); ranks = S10.set_index("query").urank
OV = []
for sf in SL:
    x = A[A.surface == sf]; ex = x["query"].isin(sset).values; w = x.pv.values
    rr = x[ex]["query"].map(ranks)
    P(f"  {SL[sf]}: exact in search top10k {ci(ex.sum(), len(x))} PV {(w*ex).sum()/w.sum()*100:.1f}% ; in search top1000 {ci(x['query'].isin(s1k).sum(), len(x))}")
    if ex.sum():
        P(f"    matched search ranks: median {rr.median():.0f}; ≤1000 {(rr<=1000).mean()*100:.1f}% ; u_ds of matched rows {x[ex].u_ds.value_counts().to_dict()}")
        P("    e.g. " + " | ".join(f"{tag(q,18)}(#{int(ranks[q])})" for q in x[ex].sort_values('pv', ascending=False)['query'].head(14)))
    OV.append(dict(surface=sf, n=len(x), exact10k=int(ex.sum()), exact1k=int(x['query'].isin(s1k).sum()), pv_share_exact=(w * ex).sum() / w.sum()))
hq = set(A[A.surface == "assistant_top1k"]["query"]); tq = set(A[A.surface == "assistant_random1k"]["query"])
top1k = S10.head(1000)
P(f"  reverse: search top1000 strings found verbatim in 助手头 {ci(top1k['query'].isin(hq).sum(), 1000)} ; in 助手尾 {ci(top1k['query'].isin(tq).sum(), 1000)}")
xr = top1k[top1k["query"].isin(hq)]
P("    reverse by search own class: " + str(xr.td_l1_name.value_counts().head(8).to_dict()))
P("    search top1000 own-class base rates for comparison: " + str((top1k.td_l1_name.value_counts(normalize=True).head(8) * 100).round(1).to_dict()))
P("    reverse rate within class: " + " ; ".join(f"{c}: {ci((xr.td_l1_name==c).sum(), (top1k.td_l1_name==c).sum())}" for c in top1k.td_l1_name.value_counts().head(6).index))
pd.DataFrame(OV).to_csv(f"{SP}/sva_dom_med_overlap.csv", index=False, encoding="utf-8-sig")

# ============ B. wrapping
P("\n######## B. wrapping: a search top1000 query (≥3 chars) contained in a longer (+≥2 chars) assistant query")
cores = sorted([q for q in top1k["query"] if len(q) >= 3 and not q.isdigit()], key=len, reverse=True)
WR = []
for sf in SL:
    for q, u in A[A.surface == sf][["query", "u_ds"]].itertuples(index=False):
        for core in cores:
            if core in q and len(q) >= len(core) + 2:
                WR.append(dict(surface=sf, core=core, query=q, u_core=lab1k.get(core), u_wrap=u)); break
WRd = pd.DataFrame(WR, columns=["surface", "core", "query", "u_core", "u_wrap"])
for sf in SL:
    x = WRd[WRd.surface == sf]; n = (A.surface == sf).sum()
    P(f"  {SL[sf]}: wrapped {ci(len(x), n)} ; intent unchanged {ci((x.u_core==x.u_wrap).sum(), len(x)) if len(x) else 'n/a'}")
    P("    transitions: " + str(x.groupby(["u_core", "u_wrap"]).size().sort_values(ascending=False).head(8).to_dict()))
    P("    all: " + " | ".join(f"[{tag(c,10)}]→{tag(q,30)}" for c, q in zip(x.core, x["query"])))

# ============ C. near-paraphrase intent transitions (neighbour labelled only if inside search top1000)
P("\n######## C. near paraphrase 0.80≤sim<0.999 with a labelled neighbour (search top1000): intent transitions")
for sf in SL:
    x = A[(A.surface == sf) & (A.sim1 >= 0.80) & (A.sim1 < 0.999)]; y = x.dropna(subset=["u_nn", "u_ds"])
    P(f"  {SL[sf]}: near-paraphrase rows {len(x)}, with labelled neighbour {len(y)}; unchanged {ci((y.u_nn==y.u_ds).sum(), len(y)) if len(y) else 'n/a'}")
    for (c1, c2), v in y[y.u_nn != y.u_ds].groupby(["u_nn", "u_ds"]).size().sort_values(ascending=False).head(8).items():
        e = y[(y.u_nn == c1) & (y.u_ds == c2)].sort_values("sim1", ascending=False).head(4)
        P(f"    {c1}→{c2} {v}: " + " | ".join(f"[{tag(nq,16)}]→{tag(q,22)}({sv:.2f})" for nq, q, sv in zip(e.nn1, e["query"], e.sim1)))

# ============ D. matched pairs by change type
P("\n######## D. matched pairs by change type (assistant side has the marker; search side does not)")
FLAGS = {
    "加入具体的人(家人/年龄/我)": f"({FAMILY})|({AGE})|(我|本人)",
    "加入数值/病程": f"({LAB})|({DUR})",
    "要判断(严重吗/能…吗/需要…吗)": JUDGE,
    "要去处/费用": f"({HOSP})|({COST})",
    "用药组合/时间/剂量": f"({COMBO})|({TIMING})",
    "省略主语/给图": f"({CTX})|({IMG_GIVE})",
}
# entity dictionary from search top10k
ENT = re.compile(r"^(.{2,10}?)(的)?(功效|作用|主治|症状|表现|前兆|是什么原因|什么原因|怎么回事|是什么意思|是什么病|是什么|吃什么药|用什么药|怎么治疗|怎么治|怎么办|图片|副作用|用法用量|正常值|能吃吗|可以吃吗|严重吗|会传染吗|多少钱|挂什么科)")
STOP = {"吃什么", "怎么", "什么", "孩子", "女性", "男性", "身体", "女人", "男人", "宝宝", "老人", "经常", "总是", "突然", "为什么", "晚上", "早上", "小孩", "中药", "药物", "食物", "小便", "大便", "肚子", "头疼", "喝什么", "吃了", "可以", "这个"}
ent2q = {}
for q, pv in zip(S10["query"], S10.wise_pv):
    m_ = ENT.match(q)
    if m_:
        e = m_.group(1).strip("，, ")
        if len(e) >= 2 and e not in STOP and not re.search(r"(吃|喝|怎么|什么|为什么|哪)", e):
            if e not in ent2q or pv > ent2q[e][1]: ent2q[e] = (q, pv)
ents = sorted(ent2q, key=len, reverse=True)
P(f"  entity dictionary from search top10k templates: {len(ents)} entities")
cand = []
for r in A.itertuples(index=False):
    q = r.query
    if r.sim1 >= 0.75: cand.append(dict(surface=r.surface, query=q, pv=r.pv, u_ds=r.u_ds, p_ds=r.p_ds, s_query=r.nn1, s_rank=r.nn1_rank, sim=r.sim1, via="nn"))
    for e in ents:
        if (len(e) >= 3 and e in q) or (len(e) == 2 and q.startswith(e)):
            sq_, spv = ent2q[e]
            if sq_ != q: cand.append(dict(surface=r.surface, query=q, pv=r.pv, u_ds=r.u_ds, p_ds=r.p_ds, s_query=sq_, s_rank=int(ranks[sq_]), sim=np.nan, via="entity:" + e))
            break
CD = pd.DataFrame(cand)
for nm, pat in FLAGS.items():
    CD[nm] = [m1(q, pat) and not m1(sq_, pat) for q, sq_ in zip(CD["query"], CD.s_query)]
CD["变短为裸词"] = [len(q) <= 4 and len(sq_) >= 6 for q, sq_ in zip(CD["query"], CD.s_query)]
CD["方向反转:搜索要图→助手给图"] = [m1(q, IMG_GIVE) and m1(sq_, IMG_WANT) for q, sq_ in zip(CD["query"], CD.s_query)]
CD.to_csv(f"{SP}/sva_dom_med_pairs_candidates.csv", index=False, encoding="utf-8-sig")
for nm in list(FLAGS) + ["变短为裸词", "方向反转:搜索要图→助手给图"]:
    for sf in SL:
        x = CD[(CD.surface == sf) & CD[nm]].drop_duplicates("query")
        x = x.assign(o=x.sim.fillna(0.0)).sort_values(["o", "pv"], ascending=False)
        P(f"\n  [{nm}] {SL[sf]} candidates {len(x)}:")
        for rr_ in x.head(18).itertuples(index=False):
            P(f"     {tag(rr_.s_query,26)} (#{rr_.s_rank}, {lab1k.get(rr_.s_query,'-')}) ⇄ {tag(rr_.query,46)} ({rr_.u_ds}, p={rr_.p_ds}) sim={rr_.sim if not pd.isna(rr_.sim) else float('nan'):.2f} via={rr_.via}")

# ============ E. assistant-only needs: low similarity themes
P("\n######## E. assistant-only needs: sim1<0.70 (control: search deepest 1000 -> rest = 13.4% <0.70, see sva_dom_med_nn.txt)")
THEMES = [
    ("身体/姿态编辑(非医疗)", BODY),
    ("依赖上文/图片的追问", f"({CTX})|({IMG_GIVE})|^(那|那么|还有|所以|对了|好的|嗯|其他|除了)"),
    ("用药组合/时间/剂量", f"({COMBO})|({TIMING})"),
    ("带数值/病程/年龄的个案", f"({LAB})|({DUR})|({AGE})"),
    ("替家人问", FAMILY),
    ("机构/医生/购药渠道/真伪", f"({HOSP})|(哪里(可以)?买|购买|正规吗|靠谱吗|电话|地址|品牌|口碑)"),
    ("第一人称或长叙述", r"(我|本人)|^.{25,}$"),
    ("食物/营养/减肥宜忌", r"(能吃|可以吃|能喝|可以喝|吃了|食物|营养|热量|卡路里|减肥|蛋白质|维生素|水果|蔬菜|泡水|煮水|茶)"),
    ("中药方剂/药材", r"(方子|药方|处方|汤|丸|配伍|中药|药材|医案|中医)"),
    ("医学知识/考试", f"({EXAM})|(细胞|病毒|基因|机制|原理|代表什么|是指|克隆|抗原|抗体)"),
]
TH = []
for sf in SL:
    x = A[A.surface == sf].copy(); n = len(x); low = (x.sim1 < 0.70).values; w = x.pv.values
    P(f"\n  {SL[sf]}: sim<0.70 {ci(low.sum(), n)} PV {(w*low).sum()/w.sum()*100:.1f}% ; p=1 in low {ci((x.p_ds[low]==1).sum(), low.sum())} vs high {ci((x.p_ds[~low]==1).sum(), (~low).sum())} ; U07 in low {ci((x.u_ds[low]=='U07').sum(), low.sum())} vs high {ci((x.u_ds[~low]=='U07').sum(), (~low).sum())}")
    xl = x[low].copy(); qs = xl["query"].tolist(); assigned = np.array([""] * len(qs), dtype=object)
    for nm, pat in THEMES:
        mk = match(qs, pat) & (assigned == ""); assigned[mk] = nm
    assigned[assigned == ""] = "其他"; xl["theme"] = assigned
    for nm in [t[0] for t in THEMES] + ["其他"]:
        y = xl[xl.theme == nm]; k = len(y)
        if k == 0: continue
        wy = y.pv.values
        TH.append(dict(surface=sf, theme=nm, k=k, n_low=len(xl), n_all=n, share_low=k / len(xl), share_all=k / n, pv_share_all=wy.sum() / w.sum(),
                       u07=(y.u_ds == "U07").mean(), p1=(y.p_ds == 1).mean()))
        ex = (y.sort_values("pv", ascending=False) if sf == "assistant_top1k" else y.sample(min(len(y), 40), random_state=13))["query"].head(14)
        P(f"   {nm}: {k} = {ci(k,len(xl))} of low ; {ci(k,n)} of all ; PV {wy.sum()/w.sum()*100:.1f}% ; U07 {(y.u_ds=='U07').mean()*100:.0f}% p=1 {(y.p_ds==1).mean()*100:.0f}% ; u_ds {y.u_ds.value_counts().head(4).to_dict()}")
        P("      e.g. " + " | ".join(tag(q, 34) for q in ex))
pd.DataFrame(TH).to_csv(f"{SP}/sva_dom_med_lowsim_themes.csv", index=False, encoding="utf-8-sig")

# ============ F. lead-generation / advertising-entry strings; G. safety markers
P("\n######## F. lead-generation strings & G. safety markers (all user rows)")
U = pd.concat([pd.DataFrame({"surface": "search_top1000", "query": top1k["query"], "pv": top1k.wise_pv}),
               pd.DataFrame({"surface": "search_top10k", "query": S10["query"], "pv": S10.wise_pv}),
               A[["surface", "query", "pv", "u_ds", "p_ds"]]], ignore_index=True)
LEAD = r"(留电话|留手机号|手机号|咨询电话|电话咨询|24小时|在线咨询|免费咨询|咨询|预约|入口|挂号)"
REDFLAG = r"(胸痛|胸口痛|呼吸困难|喘不过气|昏迷|抽搐|大出血|便血|吐血|咯血|意识不清|晕倒|中毒|误服|误吃|吃多了|过量|一次性(喝|吃)了|高烧不退|持续高烧)"
SELFDX = r"(是不是(得了|患了|有)|是否(患有|得了|是)|会不会是|是.{1,8}(引起|导致)的吗|可能是.{1,10}(吗|？|\?)|是.{1,6}还是.{1,6}(引起|导致|吗))"
ADJUST = r"(停药|减量|加量|自行|换药|不吃了|可以不吃|能不能不吃|漏服|多吃了|少吃了|吃错)"
DRUGWORD = r"(药|片|胶囊|颗粒|丸|口服液|注射|针|软膏|乳膏|滴眼液|喷雾|他汀|沙坦|地平|二甲双胍|布洛芬|阿莫西林|头孢)"
MARK = {"导流/咨询入口词": LEAD, "隐私∩导流词": None, "危险信号(红旗症状/过量)": REDFLAG, "自我诊断式(是不是得了/会不会是)": SELFDX,
        "自行调整用药(停药/减量/漏服)": ADJUST, "具名药物∩剂量/时间/联用": None, "儿童∩剂量/时间": None}
for sf in ["search_top1000", "search_top10k", "assistant_top1k", "assistant_random1k"]:
    x = U[U.surface == sf]; qs = x["query"].tolist(); n = len(qs); w = x.pv.values.astype(float)
    lead = match(qs, LEAD); sens = match(qs, SENS); rf = match(qs, REDFLAG); sd = match(qs, SELFDX); adj = match(qs, ADJUST)
    dose = match(qs, f"({TIMING})|({COMBO})"); drug = match(qs, DRUGWORD); fam = match(qs, FAMILY) | match(qs, AGE)
    res = {"导流/咨询入口词": lead, "隐私∩导流词": lead & sens, "危险信号(红旗症状/过量)": rf, "自我诊断式(是不是得了/会不会是)": sd,
           "自行调整用药(停药/减量/漏服)": adj, "具名药物∩剂量/时间/联用": drug & dose, "儿童/年龄∩剂量/时间/联用": fam & dose}
    P(f"\n  {sf} n={n}")
    for nm, mk in res.items():
        P(f"    {nm}: {ci(mk.sum(), n)}" + (f" PV{(w*mk).sum()/w.sum()*100:.1f}%" if sf != "assistant_random1k" else "")
          + "  e.g. " + " | ".join(tag(q, 30) for q in x[mk].sort_values("pv", ascending=False)["query"].head(6)))
    if sf == "assistant_top1k":
        P(f"    隐私话题 rows {int(sens.sum())}: of which lead words {ci((sens&lead).sum(), sens.sum())} ; hospital {ci((sens&match(qs,HOSP)).sum(), sens.sum())}")
        P("    隐私话题 without lead/hospital/cost words: " + " | ".join(tag(q, 20) for q in x[sens & ~lead & ~match(qs, HOSP) & ~match(qs, COST)].sort_values("pv", ascending=False)["query"].head(40)))
        P(f"    隐私话题 without lead/hospital/cost: {ci((sens & ~lead & ~match(qs,HOSP) & ~match(qs,COST)).sum(), n)}")
        P(f"    lead words overall: {ci(lead.sum(), n)} ; lead ∩ hospital {ci((lead&match(qs,HOSP)).sum(), n)}")
x = A[A.surface == "assistant_random1k"]
P(f"\n  助手尾 U07∩p=1 {ci(((x.u_ds=='U07')&(x.p_ds==1)).sum(), len(x))} ; p=1 among U07 {ci(((x.u_ds=='U07')&(x.p_ds==1)).sum(), (x.u_ds=='U07').sum())}")
qs = x["query"].tolist()
pers = match(qs, f"({FAMILY})|({AGE})|({LAB})|({DUR})|(我|本人)")
P(f"  助手尾 any personal marker (家人/年龄/数值/病程/我) {ci(pers.sum(), len(x))}; U07 inside {ci((x.u_ds[pers]=='U07').sum(), pers.sum())} vs outside {ci((x.u_ds[~pers]=='U07').sum(), (~pers).sum())}")
P("\ndone"); OUT.close()
