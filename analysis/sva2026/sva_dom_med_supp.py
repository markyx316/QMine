# -*- coding: utf-8 -*-
"""医疗 supplementary: Newcombe differences for key shape markers (search1k -> assistant head), privacy-seeking decomposition,
refined low-sim themes (symptom descriptions), dictation / addressing-the-assistant signals."""
import sys, re, math
import numpy as np, pandas as pd
SP = __import__("os").environ.get("SVA_WORKDIR", __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "work"))
sys.path.insert(0, SP)
import sva_common as C
OUT = open(f"{SP}/sva_dom_med_supp.txt", "w", encoding="utf-8")
def P(*x): print(*x, file=OUT, flush=True)
def wilson(k, n, z=1.96):
    if n == 0: return float("nan"), float("nan"), float("nan")
    p = k / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, c - h), min(1.0, c + h)
def newcombe(k1, n1, k2, n2):
    p1, l1, u1 = wilson(k1, n1); p2, l2, u2 = wilson(k2, n2); d = p2 - p1
    return d, d - math.sqrt((p2 - l2) ** 2 + (u1 - p1) ** 2), d + math.sqrt((u2 - p2) ** 2 + (p1 - l1) ** 2)
def ci(k, n): p, lo, hi = wilson(int(k), int(n)); return f"{p*100:.1f}% [{lo*100:.1f}–{hi*100:.1f}] (k={int(k)}/n={int(n)})"
def match(qs, pat):
    R = re.compile(pat); return np.fromiter((bool(R.search(q)) for q in qs), bool, len(qs))
UNSAFE = re.compile(r"(阳痿|早泄|勃起|性功能|性生活|同房|房事|做爱|射精|精子|壮阳|延时|时间太短|达泊西汀|西地那非|他达拉非|司美那非|伟哥|男科|生殖|私处|私密|下体|下身|阴|睾丸|包皮|龟头|宫颈|白带|肛|痔|屁|胸|乳|奶|内裤|裸|色|性感|巨乳|梅毒|淋病|湿疣|hpv|HPV|艾滋|性病|人流|流产|打胎|药流|避孕|腿|臀|布料|前襟|身体|姿势|\d{7,}|医生|大夫|主任|教授|老师|团队|专家|我是|叫)")
def tag(q, L=36): return (q[:L] if not UNSAFE.search(q) else "⚠" + q[:L])

Q = r"(怎么|如何|为什么|什么|哪个|哪家|哪些|多少|吗|呢|是不是|能不能|可不可以|\?|？)"
FAMILY = r"(孩子|儿子|女儿|宝宝|宝贝|婴儿|新生儿|小孩|男童|女童|男孩|女孩|老公|老婆|丈夫|妻子|男朋友|女朋友|男友|女友|我妈|我爸|妈妈|爸爸|母亲|父亲|父母|老人|家人|婆婆|公公|爷爷|奶奶|外婆|姥姥|对象|闺女)"
LAB = r"(\d+(\.\d+)?\s*(mmol|mg|ml|g/l|u/l|iu|μ|微克|毫克|毫升|克|mm|cm|毫米|厘米|公分|度|℃|斤|公斤|kg|%|次/分))|((血糖|血压|尿酸|血脂|胆固醇|甘油三酯|肌酐|转氨酶|白细胞|血小板|心率|hcg|HCG|孕酮|TSH|tsh|C反应蛋白|crp|CRP|血红蛋白|体温|低压|高压|指标|结节)[^，。,]{0,6}\d)"
DUR = r"(\d+\s*个?(天|周|星期|月|年|小时))|([两三四五六七八九十半好几]个?(天|周|星期|月|年)(了|多|左右|以来|之后|后|前|内))"
AGE = r"(\d{1,3}\s*(岁|周岁)|\d{1,2}\s*个月大|[一二三四五六七八九十]{1,3}岁)"
COMBO = r"(一起吃|一起服用|一起用|同时吃|同时服用|同时用|同服|同吃|合用|联用|并用|搭配(吃|服用|使用)|错开|间隔|隔多久|和.{1,15}(能|可以|能否|可否)(一起|同时)|(能|可以)和.{1,15}(一起|同时))"
TIMING = r"(饭前|饭后|空腹|睡前|早上吃|晚上吃|什么时候吃|什么时间吃|几点吃|吃多久|吃多长时间|连续(吃|服用|用)|长期(吃|服用)|停药|漏服|用量|用法|剂量|一次吃|一天吃|每次吃|每天吃|吃几(片|粒|颗|袋|次)|几片|几粒|多少毫克|吃多少|服用方法|怎么服用)"
HOSP = r"(医院|诊所|门诊|专科|卫生院|卫生服务中心|体检中心|科室|挂.{0,3}科|看什么科|哪个科|医生|大夫|专家|名医|中医馆|药店|药房)"
COST = r"(多少钱|费用|价格|收费|价目|报销|医保|贵不贵)"
CHOICE = r"(哪家|哪个医院|哪里|正规|医院好|成功率|排名|口碑|靠谱|专科)"
LEAD = r"(留电话|留手机号|手机号|咨询电话|电话咨询|24小时|在线咨询|免费咨询|咨询|预约|入口|挂号)"
SENS = r"(阳痿|早泄|勃起|性功能|性生活|同房|房事|时间太短|壮阳|延时|达泊西汀|西地那非|他达拉非|司美那非|伟哥|男科|生殖|私处|私密|下体|下身|阴囊|睾丸|包皮|龟头|阴茎|外阴|阴道|宫颈|白带|肛门|肛周|肛裂|肛瘘|痔疮|阴虱|阴毛|梅毒|淋病|尖锐湿疣|湿疣|hpv|HPV|艾滋|hiv|HIV|性病|衣原体|支原体|人流|流产|打胎|药流|避孕|狐臭|腋臭|腋下异味|腋下净味|口臭|体臭|体味|脚臭)"
IMG_GIVE = r"(图中|图片中|图里|图上|图片上|这张图|这张照片|照片中|照片里|看图|帮我看看|这是什么(病|药|药材|中药|菌|菌子|咬的|症状|皮肤病|虫|东西|草|植物)|解读.{0,6}(报告|单|说明书)|报告单|化验单)"
CTX = (r"^(这|那个|它|图中|图片)|^(吃|用)什么(药膏?|药)[？?]?$|^(怎么|如何)(治疗?|调理|服用|食用|预约|取下来|处理|办)[？?]?$|"
       r"^(用法用量|服用方法|功效与作用|功效作用与主治|营养价值|药用价值|副作用)[？?]?$|^(一次|一天|每次|每天)吃几(粒|片|次)[？?]?$|^饭前(吃)?还是饭后吃[？?]?$|"
       r"^有(什么)?(功效|作用|副作用|好处|药用价值)[？?]?$|^什么(功效|病|药)[？?]?$|^(严重|正常|有用|有效|要紧)(吗|不)[？?]?$|^需要(手术|吃药|吃消炎药|治疗|去医院)吗[？?]?$|"
       r"^会传染吗[？?]?$|^孕妇可以用吗[？?]?$|^(能|可以)减肥吗[？?]?$|^(热量多少|多少大卡|能长多大|是激素药吗|是医疗器械吗|有副作用吗|多少钱一支)[？?]?$")
BODY = (r"((双腿|两腿|腿|身体|屁股|臀|胸口|胸部|胸|奶|内裤|布料|肚子|肌肉|双手|手|前襟|腰)(部)?(再|要|变得?|更加|超级|最)?.{0,3}"
        r"(张开|叉开|分开|弯曲|抬高|抬起|并拢|弯下来|变大|变小|变细|变窄|窄小|放大|减少|打开|坚挺|圆润|背后|下垂|平躺|摆正|转正|平行|大一点|巨大|变胖))|"
        r"变成孕妇肚子|变胖\d+斤|胸波|巨乳|^胸大$|身材微胖|正面平躺|眼睛睁开")
EXAM = r"(下列|以下(哪|说法|选项)|正确的是|错误的是|不属于|的是[？?]?$|选项|单选|多选|判断题)"
JUDGE = r"(严重吗|要紧吗|正常吗|正常不|正常嘛|有事吗|有问题吗|有影响吗|需要.{0,6}吗|要不要|该不该|能不能|可不可以|可以.{0,10}吗|能.{0,10}吗|会不会|是不是|有必要|有用吗|有效吗)"
SYMPTOM = r"(疼|痛|痒|麻|肿|胀|酸|晕|出血|发烧|发热|咳|拉肚子|腹泻|便秘|吐|起.{0,3}(疙瘩|包|痘|疹|泡)|长了|红斑|发黄|发黑|乏力|失眠|睡不着|心慌|胸闷|抽筋|耳鸣|流鼻血|反酸|烧心)"
RUNON = r"^[^，,。？?！!；;、：:\s]{20,}$"
ADDRESS = r"(^(老师|医生|大夫|专家)[，,、 ]|你说|你刚才|您说|你是说|你搞错|你给|请你|你能|你觉得|你帮)"

F = pd.read_parquet(f"{SP}/sva_final_rows.parquet"); F["query"] = F["query"].astype(str)
M = F[(F.domain == "医疗") & (F.tier == "user")]
a, s = C.load(); cl = C.cells(a, s, "医疗")
S10 = cl["搜索top10k"].reset_index(drop=True)
CELL = {"search_top1000": M[M.surface == "search_top1000"], "search_top10k": pd.DataFrame({"query": S10["query"].astype(str), "pv": S10.wise_pv.astype(float)}),
        "assistant_top1k": M[M.surface == "assistant_top1k"], "assistant_random1k": M[M.surface == "assistant_random1k"]}
SL = {"search_top1000": "搜索1k", "search_top10k": "搜索10k", "assistant_top1k": "助手头", "assistant_random1k": "助手尾"}

P("#### 1. search1k -> assistant head, Newcombe 95% CI (pp). '!' = |diff| < 4pp (below cleaning-noise threshold)")
x1, x2 = CELL["search_top1000"], CELL["assistant_top1k"]
def mk(x, name):
    qs = x["query"].tolist()
    if name == "≤4字": return np.array([len(q) <= 4 for q in qs]), np.ones(len(qs), bool)
    if name == "疑问句(去掉U02+U13)": return match(qs, Q), ~x.u_ds.isin(["U02", "U13"]).values
    if name == "隐私难言(去掉就医/费用/选择/导流)":
        care = match(qs, HOSP) | match(qs, COST) | match(qs, CHOICE) | match(qs, LEAD); return match(qs, SENS) & ~care, np.ones(len(qs), bool)
    if name == "p_ds=1": return (x.p_ds == 1).values, np.ones(len(qs), bool)
    pats = {"疑问句": Q, "模板功效/作用/主治": r"(功效|作用|主治|营养价值|药用价值)", "模板症状": r"(症状|表现|前兆|征兆)", "模板原因": r"(原因|引起|造成|导致|怎么回事|咋回事|为什么|为啥)",
            "隐私难言": SENS, "就医机构/科室/医生": HOSP, "费用/价格": COST, "无主语/指代追问": CTX, "给图/指代图中": IMG_GIVE, "身体/姿态编辑": BODY,
            "家人代问": FAMILY, "判断请求": JUDGE, "第一人称": r"(我|本人|俺|咱)", "服药时间/剂量": TIMING, "导流/咨询入口词": LEAD, "速效根治词": r"(最快|最有效|根治|断根|除根|特效|好得快|立马|快速)"}
    return match(qs, pats[name]), np.ones(len(qs), bool)
rows = []
for name in ["疑问句", "疑问句(去掉U02+U13)", "≤4字", "模板功效/作用/主治", "模板症状", "模板原因", "速效根治词", "隐私难言", "隐私难言(去掉就医/费用/选择/导流)",
             "就医机构/科室/医生", "费用/价格", "导流/咨询入口词", "无主语/指代追问", "给图/指代图中", "身体/姿态编辑", "家人代问", "判断请求", "第一人称", "服药时间/剂量", "p_ds=1"]:
    m1_, s1_ = mk(x1, name); m2_, s2_ = mk(x2, name)
    k1, n1, k2, n2 = int((m1_ & s1_).sum()), int(s1_.sum()), int((m2_ & s2_).sum()), int(s2_.sum())
    d, lo, hi = newcombe(k1, n1, k2, n2)
    rows.append(dict(metric=name, k_search=k1, n_search=n1, p_search=k1 / n1, k_head=k2, n_head=n2, p_head=k2 / n2, diff=d, lo=lo, hi=hi))
    P(f"  {name:<26} 搜索1k {k1/n1*100:5.1f}% (n={n1}) → 助手头 {k2/n2*100:5.1f}% (n={n2}) : {d*100:+.1f} [{lo*100:+.1f},{hi*100:+.1f}]{'*' if (lo>0 or hi<0) else ''}{' !' if abs(d)<0.04 else ''}")
pd.DataFrame(rows).to_csv(f"{SP}/sva_dom_med_shape_diffs.csv", index=False, encoding="utf-8-sig")

P("\n#### 2. privacy-seeking decomposition (PV)")
for sf in ["search_top1000", "search_top10k", "assistant_top1k"]:
    x = CELL[sf]; qs = x["query"].tolist(); w = x.pv.values.astype(float); sens = match(qs, SENS)
    care = match(qs, HOSP) | match(qs, COST) | match(qs, CHOICE) | match(qs, LEAD)
    top = np.argmax(np.where(sens, w, -1))
    no_top = np.ones(len(qs), bool); no_top[top] = False
    P(f"  {SL[sf]}: 隐私 PV {(w*sens).sum()/w.sum()*100:.1f}% ; 去掉最高PV一行 ({tag(qs[top],20)} {w[top]:.0f}) {(w*(sens&no_top)).sum()/(w*no_top).sum()*100:.1f}% ; "
      f"去掉就医/费用/选择/导流 {(w*(sens&~care)).sum()/w.sum()*100:.1f}% (rows {ci((sens&~care).sum(), len(qs))}) ; 隐私∩就医等 rows {ci((sens&care).sum(), len(qs))}")
x = CELL["assistant_top1k"]; qs = x["query"].tolist(); sens = match(qs, SENS); care = match(qs, HOSP) | match(qs, COST) | match(qs, CHOICE) | match(qs, LEAD)
P("  助手头 隐私∩就医等 l2: " + str(x[sens & care].l2.value_counts().to_dict()) + " ; 隐私∩就医等 u_ds: " + str(x[sens & care].u_ds.value_counts().to_dict()))
P("  助手头 隐私 by subgroup ∩ care: " + " ; ".join(f"{nm}: {int((match(qs,p)&care).sum())}/{int(match(qs,p).sum())}" for nm, p in [
    ("性功能", r"(阳痿|早泄|勃起|性功能|时间太短|壮阳|达泊西汀|西地那非|他达拉非|司美那非|男科)"), ("私处/肛肠", r"(生殖|私处|私密|阴囊|睾丸|包皮|外阴|阴道|宫颈|白带|肛门|痔疮|阴虱)"),
    ("性传播感染", r"(梅毒|淋病|湿疣|hpv|HPV|艾滋|性病)"), ("人流避孕", r"(人流|流产|打胎|药流|避孕)"), ("体味", r"(狐臭|腋臭|腋下异味|腋下净味|口臭|体臭)")]))
for sf in ["assistant_top1k", "search_top10k"]:
    x = CELL[sf]; qs = x["query"].tolist(); h = match(qs, HOSP); ch = match(qs, CHOICE) | match(qs, COST) | match(qs, LEAD)
    P(f"  {SL[sf]} 就医机构 rows with choice/cost/lead words {ci((h&ch).sum(), h.sum())}; bare hospital names (institution word, ≤10 chars, no question) {ci((h & ~ch & ~match(qs,Q) & np.array([len(q)<=10 for q in qs])).sum(), h.sum())}")

P("\n#### 3. refined assistant-only themes (sim1<0.70), symptom descriptions separated")
NN = pd.read_parquet(f"{SP}/sva_dom_med_nn.parquet")
A = NN.merge(M[M.surface.isin(["assistant_top1k", "assistant_random1k"])].drop_duplicates(["surface", "query"])[["surface", "query", "u_ds", "p_ds"]], on=["surface", "query"], how="left")
THEMES = [
    ("身体/姿态编辑(非医疗)", BODY),
    ("依赖上文/图片的追问", f"({CTX})|({IMG_GIVE})|^(那|那么|还有|所以|对了|好的|嗯|其他|除了)"),
    ("用药组合/时间/剂量", f"({COMBO})|({TIMING})"),
    ("带数值/病程/年龄的个案", f"({LAB})|({DUR})|({AGE})"),
    ("替家人问", FAMILY),
    ("机构/医生/购药渠道/费用", f"({HOSP})|({COST})|(哪里(可以)?买|购买|正规吗|靠谱吗|电话|地址|品牌|口碑)"),
    ("第一人称或长叙述", r"(我|本人)|^.{25,}$"),
    ("具体症状描述(部位+感受)", SYMPTOM),
    ("食物/营养/减肥宜忌", r"(能吃|可以吃|能喝|可以喝|吃了|食物|营养|热量|卡路里|减肥|蛋白质|维生素|水果|蔬菜|泡水|煮水|茶)"),
    ("中药方剂/药材", r"(方子|药方|处方|汤|丸|配伍|中药|药材|医案|中医)"),
    ("医学知识/考试", f"({EXAM})|(细胞|病毒|基因|机制|原理|代表什么|是指|克隆|抗原|抗体)"),
]
TH = []
for sf in ["assistant_top1k", "assistant_random1k"]:
    x = A[A.surface == sf]; n = len(x); xl = x[x.sim1 < 0.70].copy(); qs = xl["query"].tolist(); assigned = np.array([""] * len(qs), dtype=object)
    for nm, pat in THEMES:
        m_ = match(qs, pat) & (assigned == ""); assigned[m_] = nm
    assigned[assigned == ""] = "其他"; xl["theme"] = assigned
    P(f"\n  {SL[sf]} n={n}, sim<0.70 {len(xl)}")
    for nm in [t[0] for t in THEMES] + ["其他"]:
        y = xl[xl.theme == nm]; k = len(y)
        if not k: continue
        TH.append(dict(surface=sf, theme=nm, k=k, n_low=len(xl), n_all=n, share_low=k / len(xl), share_all=k / n, u07=(y.u_ds == "U07").mean(), p1=(y.p_ds == 1).mean()))
        ex = (y.sort_values("pv", ascending=False) if sf == "assistant_top1k" else y.sample(min(len(y), 60), random_state=21))
        ex = [q for q in ex["query"] if not UNSAFE.search(q)][:12]
        P(f"   {nm}: {ci(k,len(xl))} of low ; {ci(k,n)} of all ; U07 {(y.u_ds=='U07').mean()*100:.0f}% p=1 {(y.p_ds==1).mean()*100:.0f}%")
        P("      safe e.g. " + " | ".join(q[:34] for q in ex))
pd.DataFrame(TH).to_csv(f"{SP}/sva_dom_med_lowsim_themes.csv", index=False, encoding="utf-8-sig")

P("\n#### 4. dictation / addressing signals (row%)")
for sf in ["search_top1000", "search_top10k", "assistant_top1k", "assistant_random1k"]:
    qs = CELL[sf]["query"].tolist(); n = len(qs)
    P(f"  {SL[sf]}: 无标点长句(≥20字) {ci(match(qs,RUNON).sum(), n)} ; 对助手说话/称老师医生 {ci(match(qs,ADDRESS).sum(), n)} ; ≥20字 {ci(sum(len(q)>=20 for q in qs), n)} ; 含逗号或句号的多句 {ci(match(qs, r'[，,。].{3,}').sum(), n)}")
    for nm, pat in [("无标点长句", RUNON), ("对助手说话", ADDRESS)]:
        e = [q for q in pd.Series(qs)[match(qs, pat)].sample(min(40, int(match(qs, pat).sum())), random_state=4) if not UNSAFE.search(q)][:6] if match(qs, pat).sum() else []
        P(f"     {nm} safe e.g. " + " | ".join(q[:40] for q in e))
P("\ndone"); OUT.close()
