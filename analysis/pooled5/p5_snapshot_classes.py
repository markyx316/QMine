# -*- coding: utf-8 -*-
"""每个意图 / 每个聚类叶在五个快照上的完整对照表（确定性计算，后置分析，不碰 src/）。

这份表回答的问题和主报告不同。主报告问「两个界面差多远」，用的是一个汇总距离；这里问
**每一个类目在每一个快照里各占多少、哪些类只在某个快照出现**，所以输出是逐类 × 逐快照的矩阵。

三条口径写在最前面，因为它们决定了这些数字能读出什么：

1. **一切占比都是快照内占比。** 搜索每年 ~10,000 行、助手每层 ~1,000 行，跨快照比原始条数
   只是在比导出文件的大小。
2. **「只在某快照出现」必须配可检出性。** 一个类在 950 行的助手头部里 0 条，95% 单侧上界仍有
   ~0.39%；同一个 0 条落在 10,000 行的搜索里，上界只有 ~0.037%。所以 `absence_*.csv` 对每个
   缺席都给出参照占比下的期望条数与 P(0)，并据此判 `真缺席 / 证据弱 / 不可判定`。
3. **原始条数的「独占」会被样本量带偏。** 除了 `独占快照`（该类全部行来自一个快照），还给
   `均衡归属%` = 该快照内占比 ÷ 各快照内占比之和 —— 把每个快照当成同等大小来看归属。

    python analysis/pooled5/p5_snapshot_classes.py [domain ...]
"""
from __future__ import annotations
import json, re, sys, zlib
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pooled5_common import (DOMAINS, SOURCES, SRC_ZH, WORK, available, cramers_v, load,
                            newcombe, run_dir, wilson)

ROOT = Path(__file__).resolve().parents[2]
#: **每一处重抽都用自己的种子。** 原来是一条模块级流，于是任何一处新增的抽样都会把后面每个领域
#: 的自助法区间挪动一点——点估计不变、区间变。有人据此写进正文的区间端点，下一次重跑就对不上了。
#: 现在按 (用途, 领域, 层级, 快照对) 派生种子：同一格永远得到同一个区间，与别处算了什么无关。
_BASE_SEED = 20260913


def rng_for(*tag: str) -> np.random.Generator:
    return np.random.default_rng([_BASE_SEED, zlib.crc32("|".join(map(str, tag)).encode("utf-8"))])
B = 400                      # bootstrap replicates, same as p5_tvd_ci.py
NULL_B = 300                 # 内层 TVD 的同源零分布重抽次数
#: **不要再按 source 名单分界面。** 这两个名单原来是写死的，于是 2026-09-14 的金融8 语料里
#: `2025search_rand` / `2026search_rand`（两万行随机搜索）和 `assistant_voice_top`（一千行语音头部）
#: 被表 *-E 与「仅搜索/仅助手」**静默排除**了——搜索n 读成 19,997、助手n 读成 2,937。
#: 界面现在从每行自己的 `surface` 列取（语料构建时写入，搜索 / AI助手），新 source 不需要再登记。
#: 这两个名单只留作旧域的对照，任何函数都不再读它们。
SEARCH = ["2025search", "2026search"]
ASSIST = ["assistant_top", "assistant_random", "assistant_voice"]


def surface_groups(d: pd.DataFrame, srcs: list[str]) -> tuple[list[str], list[str]]:
    """按 `surface` 把快照分成（搜索侧, 助手侧），次序跟 srcs 一致。任何一个快照没有、或有两个
    surface 值，都直接报错——静默归错一边正是上一次的缺陷。"""
    if "surface" not in d.columns:
        raise SystemExit("语料没有 surface 列：无法判定每个快照属于哪个界面，拒绝按名单猜")
    se, asr = [], []
    for s in srcs:
        v = set(d.loc[d.source == s, "surface"].astype(str))
        if len(v) != 1:
            raise SystemExit(f"快照 {s} 的 surface 取值是 {v}，必须恰好一个")
        (se if v == {"搜索"} else asr if v == {"AI助手"} else []).append(s)
        if v not in ({"搜索"}, {"AI助手"}):
            raise SystemExit(f"快照 {s} 的 surface={v} 不是 搜索 / AI助手")
    return se, asr

#: 引用护栏，三层，全部只影响**引用**，不影响任何统计。
#:
#: 第一层：风控图层命中的行（`risk_screen.json` 的 `flag_mask_indices`）一律不引原文。
#: 第二层：下面这组与领域无关的硬规则。第一版只拦了露骨词与号码串，**漏掉了「十岁苗条小女孩」
#:        这种「低龄指向 + 外貌描述」的写法**——它出现在人物领域一个被命名员和风控哨兵同时标记的
#:        叶里，而机械取例把它选成了该叶的代表。所以这里把「年龄/学段词 + 外貌或性相关词」单列一条。
#: 第三层：运行自己的风控机器点名过的具体字符串（`tree_naming.json` 的 findings.evidence、
#:        risk_reason 与 rationale 里加引号的串，以及 `risk_screen.json` 各类的 exemplar/samples）
#:        一律不引。这比按叶屏蔽窄得多——按叶屏蔽会掉 60%（金融）到 90%（影视）的可引行，
#:        因为那些叶的风险是「不该那样作答」，不是「这句话不能被引用」。
_AGE = r"(?:[0-9]{1,2}|[一二三四五六七八九十两]{1,3})\s*岁|未成年|小学生|初中生|学生妹|中学生|女童|男童"
_LOOK = r"苗条|身材|胸|腿|足|丝袜|泳装|比基尼|性感|裸|色情|写真|私房|艳|诱惑|白虎|萝莉|玉足"
_PERSON = r"女孩|男孩|小女孩|小男孩|少女|少男|女生|男生|女童|男童|学生"
#: 共现（不要求相邻）：一条里同时出现「年轻女性称谓」和「身体/情色描述」就不引原文。
#: 相邻规则漏掉了人设生成类的长 prompt——「少女…身材清瘦…身高151…」里两类词隔着几十个字。
QUOTE_COOC = (re.compile(r"少女|女孩|小女孩|女童|学生妹|萝莉|女生|JK"),
              re.compile(r"身材|三围|胸围|裸|丝袜|泳装|比基尼|性感|情色|色气|诱惑|白丝|黑丝"))
QUOTE_BLOCK = re.compile(
    r"(?:做爱|性交|自慰|裸体|裸照|色情|黄片|毛片|情色|av[0-9]|无码|三级片|强奸|迷奸|约炮|援交|卖淫|嫖|"
    r"萝莉|幼女|童女|未成年.{0,4}(?:性|裸|色)|小学生.{0,4}(?:性|裸)|"
    # 低龄指向 + 外貌/性相关词（两种词序都算）
    rf"(?:{_AGE}).{{0,8}}(?:{_PERSON}).{{0,8}}(?:{_LOOK})|(?:{_AGE}).{{0,6}}(?:{_LOOK})|"
    rf"(?:{_LOOK}).{{0,8}}(?:{_AGE})|(?:少女|女生|女童|小女孩).{{0,6}}(?:{_LOOK})|"
    rf"(?:{_LOOK}).{{0,6}}(?:少女|女童|小女孩|女生|学生妹)|"
    # 软色情图片检索
    r"(?:大?胸|丝袜|泳装|比基尼|性感|玉足|艳星|艳女|私房照|艳照|裸照|写真)(?:美女|图|照|视频)|"
    r"美女(?:图|照|视频|秀)|"
    # 身体部位 + 图像名词。**这一条是补上去的**：为了修掉「摇粒绒卫衣」那个误报，早先把 胸/大腿
    # 从单独触发词里删了，于是 `女生的胸` / `胸的图片` / `胸部照片` 又变成可引的了——实测 50 行
    # 「成人内容进图片场景」里护栏只拦住 23 行。部位 + 图像名词这个组合本身没有歧义。
    r"(?:胸|胸部|乳房|乳沟|臀|臀部|屁股|奶子|三点)(?:的)?(?:图片|照片|图$|写真|大图)|"
    r"(?:女生|美女|少女|女人|女性|妹子)(?:的)?(?:胸|胸部|乳房|臀|臀部|屁股)|"
    r"束胸|束腰|充气娃娃|哺乳照|乳房图|胸部图|翘臀|美臀|蜜臀|"
    # 故意**不**拦的近义词：胸像（半身雕塑/证件胸像）、胸透（胸部透视）、妆前乳（化妆品）——
    # 它们与性化无关，拦掉会让这几类的卡片无例可引。
    # 可识别到个人的联系方式与证件号
    r"(?:私人)?(?:联系方式|电话号码|手机号|家庭住址|身份证号)|私人电话|"
    r"1[3-9][0-9]{9}|[0-9]{15,18}|"
    # 名誉与非自愿私密影像
    r"包养|艳照门|不雅照|不雅视频|乱伦|"
    # 改写式的脱衣/去码指令。**这一类是软件跑的风控哨兵找出来的，档案里没有**，而且它绕过
    # 的方式不是换字符（像 ⅴpn），是换说法：P掉外衣 / 内衣服P掉 / 衣服变成透视装 /
    # 泳装透明度替换为 100% / 把打码去掉。实测 15 行，11 行在助手头部——那是一个生成面，
    # 请求一旦被应答就产出图像本身，所以这几行一律只给条数。
    r"p掉.{0,8}(?:衣|服)|(?:衣服|外衣|内衣|上衣|衣物).{0,8}(?:p掉|去掉|删掉|变透明|透视)|"
    r"透视装|透明度.{0,10}(?:100|全透)|肚脐以下|"
    r"(?:去|删|消)(?:除)?.{0,4}(?:打码|马赛克)|马赛克.{0,6}(?:去除|删除|消除|工具)|去码|"
    # 反向词序也要拦：「…白色打码部分去掉」把动词放在了后面。
    r"打码.{0,8}(?:去掉|删掉|去除|删除|消除)|"
    # 成人向题材标记（催眠、调教等词单独出现是正常的，只在成人向组合里拦）
    r"里番|本子下载|工口|绅士向|肉番|18禁|R18|催眠性|调教(?:少女|美少女|人妻|女仆)|凌辱|痴汉)", re.I)

#: 第四层护栏：**按运行自己的分类判**，不按正则判。
#:
#: 起因是软件的 `生成露骨性图像编辑`：51 行，正则只拦住 12 行。漏的是「p掉短裙和短裤」
#: 「三角裤消除」「删除裙子」「保持人物主体不变服装改为半透明比基尼」以及一条露骨的动作描述——
#: 换一个衣物名词、换一个动词，任何关键词表都会漏。运行已经把这 51 行判成了一个带风险标注的类，
#: 那个判断比我的正则可靠，所以直接用它。
#:
#: **故意只列性/自伤这两类，不是所有带风险标注的类。** 全量拦截代价是软件 11.4%、书籍文档 23.0%
#: 的行不可引——而引用「漫蛙」「请假条模板」「vpn」并不构成伤害，把它们也拦掉只会让卡片无例可引。
#: 这里列的是：引用原文本身就等于复述露骨内容，或等于复述某个人的危机披露。
NEVER_QUOTE_CLASSES = {
    "软件": {
        "NON_CONSENSUAL_EXPLICIT_IMAGE_EDIT",   # 51 行，全在助手侧；引用即复述露骨编辑指令
        "GET_ADULT_CONTENT_OR_APP",             # 217 行；站点名与数字暗语本身就是入口
    },
    "书籍文档": {
        "REQUEST_EXPLICIT_OR_SEXUAL_CONTENT",   # 240 行
        "REQUEST_RESTRICTED_OR_HARMFUL_CONTENT",# 111 行，含自残影像与可冒充凭证
        "VENT_SEEK_EMPATHY",                    # 274 行，含个人危机披露；按条数与来源分布描述
    },
    "医疗8": {
        # 危机与急救内容：行级风控（self_harm_and_crisis 13 行、accidental_ingestion_and_overdose 36 行）之外再加一层类级，
        # 词形没被风控正则抓到的危机行也不引。生殖健康等正当就医类不列在这里，敏感行由第五层与风控图层逐行拦。
        "SELF_HARM_CRISIS",                     # 定义了，0 行（med-pool8 top-down 标签）
        "ACCIDENTAL_INGESTION_FIRST_AID",       # 6 行
    },
}

#: 第五层护栏：**按领域追加**的不可引正则。只对列在这里的域生效，其它八个域的可引行与报告逐字节不变。
#:
#: 健康（2026-09-14）的搜索里有三类原文不能进报告：未成年人 + 私密/性（风控图层也列了，但只有少数
#: 几条命中，靠这一层兜底）；露骨的性行为描述；以及具名的个人医生（「科室 + 姓名 + 医生」与
#: 「医院 + 职称 + 姓名医生」，那是可识别到个人的信息）。
#: 故意**不**把 白带异常 / 阴道出血 / 乳腺癌 这类正常的妇产科与男科问题拦下——拦了卡片就无例可引，
#: 而引用它们并不构成伤害。年龄写成 `(?:^|[^0-9])`，因为 `47岁` 里的 `7岁` 实测会被误当成未成年。
EXTRA_QUOTE_BLOCK = {
    "健康": re.compile(
        r"(?:^|[^0-9])(?:1[0-7]|[1-9])岁.{0,10}(?:私处|私密|处女|破处|射精|发育|胸|怀孕|内裤|下面)|"
        r"(?:初中|高中|中学生|小学生|青春期|未成年|少女|少男|小女孩|小男孩|女生|男生).{0,10}"
        r"(?:私处|私密|处女|破处|射精|性|发育|怀孕|内裤|下面)|"
        r"做爱|性交|内射|内精|无套|口交|舔.{0,4}(?:私处|下面)|破处|处破|一边亲一边摸|夫妻生活|起不来|姿势最舒服|喷水|"
        r"^[\u4e00-\u9fff]{1,8}科[\u4e00-\u9fff]{2,3}医生$|"
        r"(?:医院|保健院|卫生院|中心|诊所).{0,10}(?:主任医师|副主任医师|主治医师|住院医师|医师|教授|专家).{0,8}医生$"),
}


#: 医疗8（2026-09-15）：同一个医疗垂类的 8 快照语料，长尾里有健康那一层没见过的写法，所以在健康那一条的基础上
#: 按实测补四块，全部只影响**引用**：
#: ① 露骨性行为词（手淫/自慰/射精/肛交…）——在 44,019 行上 91 行命中，其中 57 行三层护栏加健康那一条都没拦住；
#: ② 具名医生——「科/院 + 2–3 字 + 医生/大夫」或整串「2–3 字 + 医生/大夫」，排除 儿科医生、家庭医生 这类泛称，
#:    26 行，21 行原来可引，几乎全在随机层；
#: ③ 伴侣 + 性行为描述的共现——构建前抽查里穿过所有护栏的那一条就是这种写法，不含任何露骨词；
#: ④ 健康那一条原样保留（未成年人 + 私密/性、健康那批露骨词、医院 + 职称 + 医生）。
_MED8_GENERIC_DOCTOR = (r"(?:中医|西医|牙科|儿科|眼科|皮肤科|妇科|男科|外科|内科|骨科|口腔|家庭|社区|私人|心理|值班|主治|实习|"
                        r"乡村|全科|专科|兽医|宠物|中西医|男医|女医|好的|优秀|执业|助理|住院|门诊|急诊|在线|网上|医院|首席|主任)")
EXTRA_QUOTE_BLOCK["医疗8"] = re.compile(
    EXTRA_QUOTE_BLOCK["健康"].pattern + "|"
    r"手淫|自慰|打飞机|撸管|射精|肛交|爱爱|鸡巴|奶子|命根|硬不起|高潮|房事|性爱|"
    # med-pool8 跑完后补：治理理由里点名的成人标题与自伤求法、以及「黄色大片」（常指色情片）。在 44,019 行上实测只新增 3 行。
    r"黄色大片|榨精|让自己.{0,4}(?:患|得|染|生)上?.{0,6}病|(?:弄|搞)(?:伤|断|骨折)自己|自己的?.{0,4}(?:弄|搞)(?:伤|断|骨折)|"
    r"(?:科|院)[一-鿿]{2,3}(?:医生|大夫)$|"
    rf"^(?!{_MED8_GENERIC_DOCTOR}(?:医生|大夫)$)[一-鿿]{{2,3}}(?:医生|大夫)$|"
    r"(?:女朋友|男朋友|老婆|老公|对象|女友|男友).{0,12}(?:进去|软了|硬了|水太多|水多|射了|插进|同房时|爱爱)")

#: 医疗随机（2026-09-16）和 医疗8 是同一个医疗垂类、同一条产品线的另外两张快照（传统搜索随机 1w +
#: 健康管家随机 1w），写法分布同源，所以直接沿用 医疗8 那条按语料实测出来的正则，不另起一套——
#: 新域缺这一层等于把 医疗8 上验证过的「未成年人 + 私密/性、露骨性行为、具名个人医生」重新放行。
#: 指向同一个已编译对象即可，`extra_quote_mask` 只读不改。
EXTRA_QUOTE_BLOCK["医疗随机"] = EXTRA_QUOTE_BLOCK["医疗8"]
#: 医疗3 就是 医疗随机 的两个快照加一个同产品的头部快照，同一个垂类、同一批写法。
EXTRA_QUOTE_BLOCK["医疗3"] = EXTRA_QUOTE_BLOCK["医疗8"]


def extra_quote_mask(domain: str, q: pd.Series) -> pd.Series:
    """**用 Python 的 re 求值，不走 `Series.str.contains`。** 语料从 parquet 读进来是 pyarrow 字符串，
    pandas 会把正则交给 RE2，而 RE2 不认 `\\u4e00-\\u9fff` 这种转义——实测直接抛
    `ArrowInvalid: invalid escape sequence`。只对 健康 生效所以其它域的回归没抓到，是验证时撞上的。"""
    pat = EXTRA_QUOTE_BLOCK.get(domain)
    if pat is None:
        return pd.Series(False, index=q.index)
    return q.astype(str).map(lambda s: bool(pat.search(s))).astype(bool)



#: 第六层护栏：**逐串读过**以后标出的不可引串（按领域，名单是数据文件，不是正则）。只对列在这里的域生效。
#: 为什么需要：med-pool8 叙述复核员在报告表格引文里查出约 45 行违规（未成年人与性、露骨描述、具名医生、民营医院名、
#: 试管选性别），全部穿过了前五层——词表只能拦写得出来的写法。名单由两名筛查员（DeepSeek 读全部不同串、Claude 读高风险子集）
#: 按 `work/医疗8/privacy_screen/criteria.md` 标出后取并集；文件不存在时是空集合，行为与没有这一层完全相同。
SCREENED_QUOTE_BLOCK = {d: WORK / d / "privacy_screen" / "quote_block.json"
                        for d in ("医疗8", "医疗随机", "医疗3", "人物8", "影视8")}
# 人物 / 影视两域按用户要求印真实 query，广谱拦截会把例子掏空，所以那两域的名单只收
# `p5_quote_hardrule_scan.py` 按三条硬规则（未成年人与性、露骨性内容、普通个人可识别信息）
# 捞出来、再由人逐条确认过的串；文件不存在时是空集合，与没有这一层完全相同。


def screened_quote_block(domain: str) -> set[str]:
    path = SCREENED_QUOTE_BLOCK.get(domain)
    if path is None or not path.exists():
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(x) for x in (data["strings"] if isinstance(data, dict) else data)}

LEVELS = [("td_l1", "自上而下 L1 意图"), ("td_l2", "自上而下 L2 子意图"),
          ("bu_family_final", "自下而上 家族"), ("bu_leaf", "自下而上 叶")]
SHEET = {"td_l1": "L1意图", "td_l2": "L2子意图", "bu_family_final": "家族", "bu_leaf": "叶"}


# ---------------------------------------------------------------- 名称与定义

def _names(domain: str, d: pd.DataFrame) -> dict[str, dict]:
    """每个层级的 key -> 显示名 / 定义。交付分区优先：家族取 families_final（p8 之后的那棵树），
    不是 audit.families（p7 之前的 31 个家族，已经不存在了）。"""
    gen = run_dir(domain)
    tn = json.loads((gen / "tree_naming.json").read_text(encoding="utf-8"))
    leaf = {int(n["leaf_id"]): (n.get("name_zh") or "", n.get("user_need") or "",
                                float(n.get("coherence") or 0.0), n.get("risk_flag") or "")
            for n in tn.get("namings", [])}
    fam = {int(f["family_id"]): (f.get("name_zh") or "", f.get("definition") or "",
                                 bool(f.get("coherent")), f.get("audit_notes") or "")
           for f in tn.get("families_final", [])}
    p = gen / "taxonomy_v2.json"
    if not p.exists():
        p = gen / "taxonomy.json"
    tx = json.loads(p.read_text(encoding="utf-8"))
    tax = tx.get("taxonomy", tx)
    l1 = {n.get("code"): (n.get("name") or "", n.get("definition") or "", n.get("user_need") or "",
                          bool(n.get("risk")), n.get("expected_share"))
          for n in tax.get("nodes", []) if n.get("level") == 1}
    return {"leaf": leaf, "fam": fam, "l1": l1}


def _class_frame(d: pd.DataFrame, level: str, nm: dict) -> pd.DataFrame:
    """把一个层级压成 key / 显示名 / 定义。key 一律是能唯一识别该类的东西（叶用 id，因为
    金融有两对同名叶）。"""
    out = d.copy()
    if level == "td_l1":
        out["_key"] = out["td_l1"].astype(str)
        disp = {k: (nm["l1"].get(k, ("", "", "", False, None))[0] or k) for k in out["_key"].unique()}
        defi = {k: nm["l1"].get(k, ("", "", "", False, None))[1] for k in out["_key"].unique()}
    elif level == "td_l2":
        out["_key"] = out["td_l2"].astype(str)
        disp = {k: k for k in out["_key"].unique()}
        defi = {k: "" for k in out["_key"].unique()}          # 子意图是几何切分，没有名字
    elif level == "bu_family_final":
        out["_key"] = out["bu_family_final"].astype(int).astype(str)
        disp = {k: (nm["fam"].get(int(k), ("", "", False, ""))[0] or f"家族{k}") for k in out["_key"].unique()}
        defi = {k: nm["fam"].get(int(k), ("", "", False, ""))[1] for k in out["_key"].unique()}
    else:
        out["_key"] = out["bu_leaf"].astype(int).astype(str)
        disp = {k: (nm["leaf"].get(int(k), ("", "", 0.0, ""))[0] or f"叶{k}") for k in out["_key"].unique()}
        defi = {k: nm["leaf"].get(int(k), ("", "", 0.0, ""))[1] for k in out["_key"].unique()}
    # 同名不同 id 的类必须能分开：金融有两对同名叶，同名相加会把两个不同的簇读成一个
    seen: dict[str, list[str]] = {}
    for k, v in disp.items():
        seen.setdefault(v, []).append(k)
    disp = {k: (f"{v}#{k}" if len(seen[v]) > 1 else v) for k, v in disp.items()}
    out["_disp"] = out["_key"].map(disp)
    out["_def"] = out["_key"].map(defi)
    return out


# ---------------------------------------------------------------- 逐类 × 逐快照矩阵

def matrix(d: pd.DataFrame, srcs: list[str]) -> pd.DataFrame:
    n_s = {s: int((d.source == s).sum()) for s in srcs}
    pv_s = {s: float(d.loc[d.source == s, "pv_norm"].sum()) for s in srcs}
    base = d["_key"].value_counts(normalize=True)              # 全语料占比（= class_profile.csv 的分母）
    rows = []
    for key, g in d.groupby("_key", sort=False):
        r = {"key": key, "类目": g["_disp"].iloc[0], "定义": g["_def"].iloc[0],
             "全语料条数": len(g), "全语料占比%": round(100 * len(g) / len(d), 3)}
        sh = {}
        for s in srcs:
            k = int((g.source == s).sum())
            lo, hi = wilson(k, n_s[s])
            sh[s] = k / n_s[s] if n_s[s] else 0.0
            r[f"{SRC_ZH[s]}_条数"] = k
            r[f"{SRC_ZH[s]}_占比%"] = round(100 * sh[s], 3)
            r[f"{SRC_ZH[s]}_95CI低%"] = round(100 * lo, 3)
            r[f"{SRC_ZH[s]}_95CI高%"] = round(100 * hi, 3)
            r[f"{SRC_ZH[s]}_流量占比%"] = round(
                100 * g.loc[g.source == s, "pv_norm"].sum() / pv_s[s], 3) if pv_s[s] else float("nan")
        tot = sum(sh.values())
        for s in srcs:
            r[f"{SRC_ZH[s]}_均衡归属%"] = round(100 * sh[s] / tot, 2) if tot else float("nan")
            # 集中度指数：快照内占比 ÷ 全语料占比，与 class_profile.csv 同义（分母含 87% 搜索行）
            r[f"{SRC_ZH[s]}_指数"] = round(sh[s] / base[key], 4) if base[key] else float("nan")
            # 均衡指数：快照内占比 ÷ 各快照占比的未加权均值（把每个快照当同等大小）
            r[f"{SRC_ZH[s]}_均衡指数"] = round(sh[s] / (tot / len(srcs)), 4) if tot else float("nan")
        present = [s for s in srcs if r[f"{SRC_ZH[s]}_条数"] > 0]
        absent = [s for s in srcs if r[f"{SRC_ZH[s]}_条数"] == 0]
        hi_s = max(srcs, key=lambda s: sh[s])
        lo_s = min(srcs, key=lambda s: sh[s])
        r["出现快照数"] = len(present)
        r["缺席快照"] = " / ".join(SRC_ZH[s] for s in absent)
        r["独占快照"] = SRC_ZH[present[0]] if len(present) == 1 else ""
        r["最高快照"] = SRC_ZH[hi_s]
        r["最低快照"] = SRC_ZH[lo_s]
        r["极差pp"] = round(100 * (sh[hi_s] - sh[lo_s]), 3)
        r["最高比最低倍数"] = round(sh[hi_s] / sh[lo_s], 2) if sh[lo_s] > 0 else float("inf")
        rows.append(r)
    t = pd.DataFrame(rows)
    for s in srcs:                                             # 快照内排名（1 = 该快照里最大的类）
        t[f"{SRC_ZH[s]}_排名"] = t[f"{SRC_ZH[s]}_占比%"].rank(ascending=False, method="min").astype(int)
    return t.sort_values("全语料条数", ascending=False).reset_index(drop=True)


def enrich(t: pd.DataFrame, d: pd.DataFrame, level: str, nm: dict, quotable: pd.Series) -> pd.DataFrame:
    """把另一条路线的对应关系接上去。两条路线是独立产生的（意图由架构师写、叶由嵌入聚类得到），
    所以「这个叶主要落在哪个意图」本身就是一个测量，不是定义。"""
    other = "td_l1_name" if level.startswith("bu_") else "bu_leaf_name"
    lab = "主导意图" if level.startswith("bu_") else "主导叶"
    dom, share = {}, {}
    for key, g in d.groupby("_key", sort=False):
        vc = g[other].astype(str).value_counts(normalize=True)
        dom[key], share[key] = vc.index[0], round(100 * float(vc.iloc[0]), 1)
    t = t.copy()
    t[lab] = t["key"].map(dom)
    t[f"{lab}占比%"] = t["key"].map(share)
    if level == "bu_leaf":
        t["盲评一致性"] = t["key"].map(lambda k: nm["leaf"].get(int(k), ("", "", 0.0, ""))[2])
        t["风险标注"] = t["key"].map(lambda k: nm["leaf"].get(int(k), ("", "", 0.0, ""))[3])
        t["user_need"] = t["key"].map(lambda k: nm["leaf"].get(int(k), ("", "", 0.0, ""))[1])
    if level == "bu_family_final":
        t["家族内部一致"] = t["key"].map(lambda k: "是" if nm["fam"].get(int(k), ("", "", False, ""))[2] else "否")
        t["家族审计意见"] = t["key"].map(lambda k: nm["fam"].get(int(k), ("", "", False, ""))[3])
    if level == "td_l1":
        t["体系内风险标注"] = t["key"].map(lambda k: "是" if nm["l1"].get(k, ("", "", "", False, None))[3] else "否")
        t["架构师预估占比"] = t["key"].map(lambda k: nm["l1"].get(k, ("", "", "", False, None))[4])
        t["user_need"] = t["key"].map(lambda k: nm["l1"].get(k, ("", "", "", False, None))[2])
    if level == "td_l2":
        # 子意图没有名字（几何细分），所以给代表串：该子意图里流量最高的可引串
        rep = {}
        for key, g in d.groupby("_key", sort=False):
            pool = g[quotable.reindex(g.index, fill_value=False)]
            top = pool.sort_values("pv_norm", ascending=False)["query"].head(3).tolist()
            rep[key] = " / ".join(str(x)[:24] for x in top) if top else "（该子意图无可引行）"
        t["代表串(流量最高，仅供辨识)"] = t["key"].map(rep)
    return t


# ---------------------------------------------------------------- 缺席的可检出性

def absence(d: pd.DataFrame, srcs: list[str]) -> pd.DataFrame:
    """每一个「该类在该快照 0 条」，配上它在别处的占比、这里的期望条数与 P(0)。

    0 条本身从不证明「不存在」：950 行里 0 条，单侧 97.5% 上界 ~0.39%。判定阈值写死在这里，
    报告只引用 verdict，不再自己解释 0。"""
    n_s = {s: int((d.source == s).sum()) for s in srcs}
    rows = []
    for key, g in d.groupby("_key", sort=False):
        cnt = {s: int((g.source == s).sum()) for s in srcs}
        for s in srcs:
            if cnt[s]:
                continue
            k_o = len(g) - cnt[s]
            n_o = sum(n_s[t] for t in srcs if t != s)
            p_ref = k_o / n_o if n_o else 0.0
            exp = p_ref * n_s[s]
            p0 = (1 - p_ref) ** n_s[s]
            ub = 1 - 0.025 ** (1 / n_s[s]) if n_s[s] else 1.0   # 单侧 97.5% 上界
            best = max((t for t in srcs if t != s), key=lambda t: cnt[t] / n_s[t]) if n_o else None
            rows.append({"key": key, "类目": g["_disp"].iloc[0], "缺席快照": SRC_ZH[s],
                         "该快照n": n_s[s], "其余快照条数": k_o, "其余快照n": n_o,
                         "参照占比%": round(100 * p_ref, 3), "期望条数": round(exp, 2),
                         "P(0)": round(p0, 4), "该快照上界%": round(100 * ub, 3),
                         "最高占比快照": SRC_ZH[best] if best else "",
                         "最高占比%": round(100 * cnt[best] / n_s[best], 3) if best else float("nan"),
                         "判定": ("真缺席" if exp >= 5 and p0 < 0.01 else
                                  "不可判定（样本量不足）" if exp < 3 else "偏少但证据弱")})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 逐对差异（全部快照对）

def newcombe_all(d: pd.DataFrame, srcs: list[str]) -> pd.DataFrame:
    rows = []
    for i, a in enumerate(srcs):
        for b in srcs[i + 1:]:
            ga, gb = d[d.source == a], d[d.source == b]
            na, nb = len(ga), len(gb)
            ca, cb = ga["_key"].value_counts(), gb["_key"].value_counts()
            disp = d.drop_duplicates("_key").set_index("_key")["_disp"]
            for key in sorted(set(ca.index) | set(cb.index)):
                ka, kb = int(ca.get(key, 0)), int(cb.get(key, 0))
                diff, lo, hi = newcombe(ka, na, kb, nb)
                rows.append({"key": key, "类目": disp.get(key, key), "a": SRC_ZH[a], "b": SRC_ZH[b],
                             "k_a": ka, "k_b": kb, "n_a": na, "n_b": nb,
                             "占比_a%": round(100 * ka / na, 3), "占比_b%": round(100 * kb / nb, 3),
                             "差_pp": round(100 * diff, 3), "低_pp": round(100 * lo, 3),
                             "高_pp": round(100 * hi, 3), "显著": bool(lo > 0 or hi < 0),
                             "类型": ("b独有" if ka == 0 and kb else "a独有" if kb == 0 and ka else "共有")})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 快照两两距离

def _codes(s: pd.Series, order: list[str]) -> np.ndarray:
    m = {k: i for i, k in enumerate(order)}
    return s.map(m).to_numpy()


def _tvd(a: np.ndarray, b: np.ndarray, K: int) -> float:
    pa = np.bincount(a, minlength=K) / len(a)
    pb = np.bincount(b, minlength=K) / len(b)
    return float(0.5 * np.abs(pa - pb).sum())


def pairwise_tvd(d: pd.DataFrame, srcs: list[str], tag: str = "") -> pd.DataFrame:
    order = sorted(d["_key"].unique())
    K = len(order)
    arr = {s: _codes(d.loc[d.source == s, "_key"], order) for s in srcs}
    rows = []
    for i, a in enumerate(srcs):
        for b in srcs[i + 1:]:
            ga, gb = arr[a], arr[b]
            point = _tvd(ga, gb, K)
            rng = rng_for("pairwise_tvd", tag, a, b)
            boot = np.array([_tvd(rng.choice(ga, len(ga)), rng.choice(gb, len(gb)), K) for _ in range(B)])
            null = []
            for g in (ga, gb):
                for _ in range(B // 2):
                    idx = rng.permutation(len(g))
                    h = len(g) // 2
                    null.append(_tvd(g[idx[:h]], g[idx[h:2 * h]], K))
            ct = pd.crosstab(pd.concat([d.loc[d.source == a, "_key"], d.loc[d.source == b, "_key"]]),
                             pd.concat([d.loc[d.source == a, "source"], d.loc[d.source == b, "source"]]))
            # 秩相关：两个快照把这些类排成的顺序有多像。TVD 看的是量差，这个看的是次序——
            # 两个快照可以量差很大而次序几乎不变（份额整体缩放），也可以量差不大而次序全乱。
            ca = d.loc[d.source == a, "_key"].value_counts().reindex(order).fillna(0)
            cb = d.loc[d.source == b, "_key"].value_counts().reindex(order).fillna(0)
            rho = float(ca.rank().corr(cb.rank(), method="spearman"))
            top_a = list(ca.sort_values(ascending=False).head(5).index)
            top_b = list(cb.sort_values(ascending=False).head(5).index)
            rows.append({"a": SRC_ZH[a], "b": SRC_ZH[b], "n_a": len(ga), "n_b": len(gb),
                         "秩相关rho": round(rho, 4),
                         "前5重合个数": len(set(top_a) & set(top_b)),
                         "首位是否相同": bool(top_a[0] == top_b[0]),
                         "TVD": round(point, 4),
                         "TVD低": round(float(np.percentile(boot, 2.5)), 4),
                         "TVD高": round(float(np.percentile(boot, 97.5)), 4),
                         "同源噪声上界": round(float(np.percentile(null, 95)), 4),
                         "超出噪声": bool(float(np.percentile(boot, 2.5)) > float(np.percentile(null, 95))),
                         "CramersV": round(cramers_v(ct.values), 4)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 类目边界的可靠度

def confidence(d: pd.DataFrame, srcs: list[str]) -> pd.DataFrame:
    """这个类在这个快照里，标得有多稳。

    占比差本身不说明边界靠不靠得住：一个类可以在某个快照里份额稳定，而那一层的行恰好全是
    分类器勉强判下来的。`td_confidence` / `td_ambiguous` / `bu_ambiguous` 是运行自己留下的
    逐行读数，按 (类, 快照) 汇总后就能看出「这一格的数字该打几折」。"""
    rows = []
    for key, g in d.groupby("_key", sort=False):
        r = {"key": key, "类目": g["_disp"].iloc[0], "全语料条数": len(g)}
        for s in srcs:
            gg = g[g.source == s]
            r[f"{SRC_ZH[s]}_n"] = len(gg)
            r[f"{SRC_ZH[s]}_置信度均值"] = round(float(gg["td_confidence"].mean()), 4) if len(gg) else float("nan")
            r[f"{SRC_ZH[s]}_意图歧义%"] = round(100 * float(gg["td_ambiguous"].mean()), 2) if len(gg) else float("nan")
            r[f"{SRC_ZH[s]}_聚类歧义%"] = round(100 * float(gg["bu_ambiguous"].mean()), 2) if len(gg) else float("nan")
        vals = [r[f"{SRC_ZH[s]}_置信度均值"] for s in srcs if r[f"{SRC_ZH[s]}_n"] >= 20]
        r["置信度最低快照"] = (min((s for s in srcs if r[f"{SRC_ZH[s]}_n"] >= 20),
                                   key=lambda s: r[f"{SRC_ZH[s]}_置信度均值"], default=None) or "")
        r["置信度最低快照"] = SRC_ZH.get(r["置信度最低快照"], r["置信度最低快照"])
        r["置信度极差"] = round(max(vals) - min(vals), 4) if len(vals) >= 2 else float("nan")
        r["全语料置信度均值"] = round(float(g["td_confidence"].mean()), 4)
        rows.append(r)
    return pd.DataFrame(rows).sort_values("全语料置信度均值")


# ---------------------------------------------------------------- 快照的特征类

def signature(d: pd.DataFrame, srcs: list[str]) -> pd.DataFrame:
    """某个快照的「特征类」：它在这个快照里的占比，**显著高于其余每一个快照**。

    「独占某快照」几乎永远是 0（一个类只要在别处出现一条就不算独占），所以那个口径对
    「这个快照有什么特别的」几乎没有信息量。这里换一个能真正承重的口径：对每一对
    （该快照 vs 另一个快照）都做 Newcombe 检验，全部方向一致且区间都不含 0 才算。
    要求 n>=2 个快照；快照越多这个条件越严。"""
    n_s = {s: int((d.source == s).sum()) for s in srcs}
    rows = []
    for key, g in d.groupby("_key", sort=False):
        cnt = {s: int((g.source == s).sum()) for s in srcs}
        for s in srcs:
            others = [t for t in srcs if t != s]
            hi = lo = True
            worst_hi, worst_lo = 1.0, -1.0
            for t in others:
                diff, l, h = newcombe(cnt[t], n_s[t], cnt[s], n_s[s])   # s 相对 t
                if not (l > 0):
                    hi = False
                if not (h < 0):
                    lo = False
                worst_hi = min(worst_hi, l)
                worst_lo = max(worst_lo, h)
            if hi or lo:
                rows.append({"key": key, "类目": g["_disp"].iloc[0], "快照": SRC_ZH[s],
                             "方向": "显著高于其余全部" if hi else "显著低于其余全部",
                             "该快照条数": cnt[s], "该快照n": n_s[s],
                             "该快照占比%": round(100 * cnt[s] / n_s[s], 3),
                             "其余快照合并占比%": round(
                                 100 * (len(g) - cnt[s]) / (len(d) - n_s[s]), 3),
                             "最弱一对的区间端点pp": round(100 * (worst_hi if hi else worst_lo), 3),
                             "对比了几个快照": len(others)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 界面层（搜索 vs 助手）

def interface_split(d: pd.DataFrame, srcs: list[str]) -> pd.DataFrame:
    se, asr = surface_groups(d, srcs)
    ds, da = d[d.source.isin(se)], d[d.source.isin(asr)]
    ns, na = len(ds), len(da)
    cs, ca = ds["_key"].value_counts(), da["_key"].value_counts()
    disp = d.drop_duplicates("_key").set_index("_key")["_disp"]
    rows = []
    for key in sorted(set(cs.index) | set(ca.index)):
        ks, ka = int(cs.get(key, 0)), int(ca.get(key, 0))
        diff, lo, hi = newcombe(ks, ns, ka, na)
        # 「只在一边出现」必须配可检出性：另一边的 n 若只有 2,000 行，0 条几乎证明不了什么。
        exp = p0 = float("nan")
        verdict = ""
        if ka == 0:
            exp, p0 = (ks / ns) * na, (1 - ks / ns) ** na
        elif ks == 0:
            exp, p0 = (ka / na) * ns, (1 - ka / na) ** ns
        if ka == 0 or ks == 0:
            verdict = ("真缺席" if exp >= 5 and p0 < 0.01 else
                       "不可判定（样本量不足）" if exp < 3 else "偏少但证据弱")
        rows.append({"key": key, "类目": disp.get(key, key),
                     "搜索条数": ks, "搜索n": ns, "搜索占比%": round(100 * ks / ns, 3),
                     "助手条数": ka, "助手n": na, "助手占比%": round(100 * ka / na, 3),
                     "差_pp": round(100 * diff, 3), "低_pp": round(100 * lo, 3), "高_pp": round(100 * hi, 3),
                     "显著": bool(lo > 0 or hi < 0),
                     "归属": ("仅搜索" if ka == 0 else "仅助手" if ks == 0 else "两界面都有"),
                     "另一边期望条数": round(exp, 2) if exp == exp else float("nan"),
                     "另一边P(0)": round(p0, 4) if p0 == p0 else float("nan"),
                     "缺席判定": verdict,
                     "助手行占该类比例%": round(100 * ka / (ks + ka), 2)})
    return pd.DataFrame(rows).sort_values("差_pp")


# ------------------------------------------------- 「没有助手独有的类」这句话有多大把握

def assistant_only_power(d: pd.DataFrame, srcs: list[str], level_name: str) -> dict:
    """如果真有一个类只在助手侧出现，这份数据看得见吗？

    「仅助手 = 0」只有配上检出力才是结论。取助手侧最小的那个类的发生率，问：同样的发生率放到
    搜索侧那两万行上，期望几条、一条都不出现的概率多大。期望条数够大、P(0) 够小，这句话才立得住。
    反方向（「仅搜索」）的检出力弱得多，因为助手侧只有两三千行——这个不对称是结构性的。"""
    _se, _asr = surface_groups(d, [s for s in SOURCES if (d.source == s).any()])
    se = d[d.source.isin(_se)]
    asr = d[d.source.isin(_asr)]
    if not len(se) or not len(asr):
        return {}
    ka = asr["_key"].value_counts()
    kmin = int(ka.min())
    p = kmin / len(asr)
    return {"层级": level_name, "助手n": len(asr), "搜索n": len(se),
            "助手侧最小类的条数": kmin, "该发生率%": round(100 * p, 4),
            "同发生率下搜索侧期望条数": round(p * len(se), 1),
            "搜索侧一条都不出现的概率": round((1 - p) ** len(se), 4),
            "实际仅助手类数": int(((se["_key"].value_counts().reindex(ka.index).fillna(0)) == 0).sum())}


# ---------------------------------------------------------------- 快照画像

def coverage(d: pd.DataFrame, srcs: list[str], level: str) -> pd.DataFrame:
    rows = []
    for s in srcs:
        g = d[d.source == s]
        p = g["_key"].value_counts(normalize=True).to_numpy()
        ent = float(-(p * np.log(p)).sum())
        rows.append({"快照": SRC_ZH[s], "层级": level, "n": len(g), "类目数": int(g["_key"].nunique()),
                     "有效类目数": round(float(np.exp(ent)), 2), "熵": round(ent, 4),
                     "HHI": round(float((p ** 2).sum()), 4),
                     "首位类目": g["_disp"].value_counts().index[0],
                     "首位占比%": round(100 * p.max(), 2),
                     "前3占比%": round(100 * np.sort(p)[::-1][:3].sum(), 2),
                     "前5占比%": round(100 * np.sort(p)[::-1][:5].sum(), 2)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 前 N 名合起来占多少

#: 每个快照自己的前 N 名，合起来覆盖了这个快照的多少行、多少流量。
#:
#: **每个快照各排各的。** 第 10 名在两个快照里几乎从来不是同一个类，所以这张表只能竖着读：
#: 它回答「这个快照的头部有多集中」，**不**回答「同样这十个类在别的快照里占多少」——后者是
#: 表 *-A 的事。
#:
#: **两个单位都给。** `行占比` 是条数占比（= 表 2.2 每格显示的那个数）；`流量占比` 是**同样
#: 这十个类**（仍按行占比选出）按 `pv_norm` 加权后的占比，**不是**「按流量重排以后的前十」。
#: pv_norm 在每个来源内部各自归一到 10,000，所以流量占比同样只能在一个快照内部读；语音导出
#: 没有 PV，按均匀权重处理，它的流量占比恒等于行占比。
#:
#: **合计对并列免疫。** 七个领域里有 8 个格子的第 10 名与第 11 名条数完全相同（例如金融的叶 ×
#: 助手头部1k，两者都是 2.316%），表 2.2 显示的是其中一个；但「最大的 10 个占比之和」与挑中
#: 哪一个无关，所以合计是确定的，比它上面那一行更稳。`第10与第11并列` 把这件事标出来。
TOPN_CUTS = (1, 3, 5, 10)


def topn(d: pd.DataFrame, srcs: list[str], level_zh: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """返回 (逐快照合计, 前十成员明细)。成员明细是**报告渲染第1..第10 的唯一依据**。"""
    rows, members = [], []
    for s in srcs:
        g = d[d.source == s]
        n = len(g)
        cnt = g["_key"].value_counts()
        pv = g.groupby("_key")["pv_norm"].sum()
        pv_tot = float(pv.sum())
        # **确定性排序，并且是唯一的一条排序路径。** `value_counts()` 在并列时的次序是任意的，
        # 而报告原来另有一条 `sort_values(占比%)` 路径——两条路径在并列格上会选到不同的类，于是
        # 「同样这十个类」的流量合计算的其实是另外十个类：教育 助手随机1k 叶实测 73.98% vs 79.23%，
        # 差 5.25pp。现在成员由这里定下来，报告只负责显示。
        order = sorted(cnt.index, key=lambda k: (-int(cnt[k]), str(k)))
        r = {"层级": level_zh, "快照": SRC_ZH[s], "n": n, "类目数": len(order)}
        for k in TOPN_CUTS:
            sel = order[:k]
            c = int(cnt.loc[sel].sum())
            r[f"前{k}条数"] = c
            # **不在这里四舍五入。** 表 1 与表 2.2 都从同一个未舍入的值格式化，双重舍入
            # 曾让同一个量在两张表里印成 98.24 与 98.23。
            r[f"前{k}行占比%"] = 100 * c / n if n else float("nan")
            r[f"前{k}流量占比%"] = (100 * float(pv.loc[sel].sum()) / pv_tot
                                   if pv_tot else float("nan"))
        for i, k in enumerate(order[:10], 1):
            members.append({"层级": level_zh, "快照": SRC_ZH[s], "名次": i, "key": k,
                            "条数": int(cnt[k]), "行占比%": 100 * int(cnt[k]) / n,
                            "流量占比%": 100 * float(pv[k]) / pv_tot if pv_tot else float("nan")})
        # **流量口径的有效样本量。** 行占比的 n 是行数；流量占比的 n 不是——一行的权重可以占
        # 整个快照的 7%。n_eff = 1/Σw²：金融 2026搜索 有 9,999 行，流量口径只有 111.5。
        # 不给这个数，读者会拿行占比的精度去读流量占比。
        w = g["pv_norm"].to_numpy(dtype=float)
        w = w / w.sum() if w.sum() else w
        r["流量有效n"] = float(1.0 / np.square(w).sum()) if w.sum() else float("nan")
        r["单行最大流量%"] = float(100 * w.max()) if len(w) else float("nan")
        r["前10之外类数"] = max(len(order) - 10, 0)
        r["前10之外条数"] = n - r["前10条数"]
        r["前10之外行占比%"] = 100 - r["前10行占比%"]
        r["前10之外流量占比%"] = 100 - r["前10流量占比%"]
        # 并列摆动：行占比的合计对「挑中哪一个」免疫（并列的类条数相同），**流量合计不免疫**，
        # 因为并列的类 pv_norm 不同。这里给出挑法能造成的最小/最大值，让读者看到这一格有多松。
        if len(order) > 10:
            thr = int(cnt[order[9]])
            strict = [k for k in order if int(cnt[k]) > thr]
            tied = [k for k in order if int(cnt[k]) == thr]
            need = 10 - len(strict)
            base = float(pv.loc[strict].sum()) if strict else 0.0
            cand = sorted(float(pv[k]) for k in tied)
            lo, hi = base + sum(cand[:need]), base + sum(cand[len(cand) - need:])
            r["第10名条数"] = thr
            r["并列类数"] = len(tied)
            r["并列需选"] = need
            r["边界并列"] = bool(len(tied) > need)
            r["前10流量占比%低"] = 100 * lo / pv_tot if pv_tot else float("nan")
            r["前10流量占比%高"] = 100 * hi / pv_tot if pv_tot else float("nan")
        else:
            r["第10名条数"] = int(cnt[order[-1]]) if order else 0
            r["并列类数"], r["并列需选"], r["边界并列"] = 0, 0, False
            r["前10流量占比%低"] = r["前10流量占比%高"] = r["前10流量占比%"]
        r["前10流量摆动pp"] = r["前10流量占比%高"] - r["前10流量占比%低"]
        rows.append(r)
    return pd.DataFrame(rows), pd.DataFrame(members)


# ---------------------------------------------------------------- 健康：被清洗掉的产品层、盲标一致度、裸词敏感性
#
# 只对带这些列/文件的语料生效（目前只有 健康），其余域不产出这些表、报告与工作簿逐字节不变。

#: 盲标文件按领域登记。
AUDIT_LABELS = {"健康": "health_ai_audit_labels.csv", "医疗随机": "医疗随机/ai_audit_labels.csv",
                "医疗3": "医疗3/ai_audit_labels.csv"}
#: 这些层的串是具名个人（医生卡），产品层表里只给条数。
COUNT_ONLY_TIERS = {"H5_doctor_card", "S6_doctor_card_uniform"}


def _fleiss_from_votes(votes: pd.Series, cats: str = "UAFPWDCX") -> float:
    mat = np.array([[str(v).count(c) for c in cats] for v in votes], dtype=float)
    n = mat.sum(1)
    assert (n == n[0]).all(), "every string must have the same number of votes"
    n0 = n[0]
    p_i = (np.sum(mat * (mat - 1), 1)) / (n0 * (n0 - 1))
    p_j = mat.sum(0) / mat.sum()
    return float((p_i.mean() - np.sum(p_j ** 2)) / (1 - np.sum(p_j ** 2)))


def product_layer(domain: str) -> dict[str, pd.DataFrame]:
    p = WORK / f"{domain}_all_rows.parquet"
    if not p.exists():
        return {}
    a = pd.read_parquet(p)
    if "tier_source" not in a.columns:
        return {}
    a = a.reset_index(drop=True)
    # 健康 是一周 7 天的导出，PV 是周合计；医疗随机 是单日导出（`n_days` 这类列根本不存在），
    # 把它的列也叫「周PV」会让读者以为这是一周的量。列名跟着语料的时间口径走，健康那一份逐字节不变。
    week = "n_days" in a.columns
    PV = "周PV" if week else "当日PV"
    tiers = a.groupby(["source", "tier"]).agg(**{"串数": ("query", "size"), PV: ("pv_raw", "sum")}).reset_index()
    tiers["串数占该来源%"] = (100 * tiers["串数"] / tiers.groupby("source")["串数"].transform("sum")).round(3)
    tiers[f"{PV}占该来源%"] = (100 * tiers[PV] / tiers.groupby("source")[PV].transform("sum")).round(3)
    tiers.insert(0, "来源", tiers["source"].map(SRC_ZH))
    tiers[PV] = tiers[PV].astype(int)
    q = a["query"].astype(str)
    blocked = (q.str.contains(QUOTE_BLOCK) | (q.str.contains(QUOTE_COOC[0]) & q.str.contains(QUOTE_COOC[1]))
               | extra_quote_mask(domain, q))
    rows = []
    for (src, tier), g in a[a["tier"] != "user"].groupby(["source", "tier"]):
        tot = float(a.loc[a["source"] == src, "pv_raw"].sum())
        for rank, (i, r) in enumerate(g.sort_values("pv_raw", ascending=False).head(12).iterrows(), 1):
            show = tier not in COUNT_ONLY_TIERS and not bool(blocked.iloc[i])
            rows.append({"来源": SRC_ZH[src], "tier": tier, "名次": rank,
                         "串": r["query"] if show else "（不引原文）",
                         PV: int(r["pv_raw"]), f"{PV}占该来源%": round(100 * float(r["pv_raw"]) / tot, 3),
                         **({"在榜天数": int(r["n_days"])} if "n_days" in a.columns else {}),
                         "决定方式": (r["tier_source"] if isinstance(r.get("tier_source"), str) else "rule")})
    agg = []
    for src, g in a.groupby("source", sort=False):
        if not week:
            agg.append({"来源": SRC_ZH[src], "source": src, "合并后串数": len(g),
                        "平台分类跨行冲突的串": int(g.get("legacy_label_conflict", pd.Series(False, index=g.index)).fillna(False).astype(bool).sum()),
                        "PV合计": int(g["pv_raw"].sum())})
            continue
        agg.append({"来源": SRC_ZH[src], "source": src, "原始导出行": int(g["n_rows_raw"].sum()), "合并后串数": len(g),
                    "7天都在榜的串": int((g["n_days"] == 7).sum()),
                    "周PV为精确值的串占比%": round(100 * float((g["n_days"] == 7).mean()), 2),
                    "周PV为精确值的PV占比%": round(100 * float(g.loc[g["n_days"] == 7, "pv_raw"].sum() / g["pv_raw"].sum()), 2),
                    "只在1天上榜的串": int((g["n_days"] == 1).sum()),
                    "同一天被拆成两行的串": int((g.get("same_day_split_rows", pd.Series(0, index=g.index)) > 0).sum()),
                    "平台分类跨行冲突的串": int(g.get("legacy_label_conflict", pd.Series(False, index=g.index)).fillna(False).astype(bool).sum()),
                    "周PV合计": int(g["pv_raw"].sum()),
                    "周PV上界合计": int(g["pv_week_upper"].sum()) if "pv_week_upper" in g.columns else None})
    out = {"tiers": tiers, "top": pd.DataFrame(rows), "aggregation": pd.DataFrame(agg)}
    lab = AUDIT_LABELS.get(domain)
    if lab and (WORK / lab).exists():
        L = pd.read_csv(WORK / lab)
        ai = a[a["tier_source"].isin(["audit_unanimous", "rule"])].copy()
        non = list("AFPWDC")
        dec = ai["audit_majority"].isin(["U"] + non)
        rnon = ai["rule_tier"].ne("user")
        anon = ai["audit_majority"].isin(non)
        w = ai["pv_raw"].astype(float)
        tp, fp, fn = (rnon & anon & dec), (rnon & ~anon & dec), (~rnon & anon & dec)
        maj = L.groupby("majority").agg(**{"串数": ("query", "size"), PV: ("pv_raw", "sum")}).reset_index()
        maj[f"{PV}占比%"] = (100 * maj[PV] / maj[PV].sum()).round(3)
        maj[PV] = maj[PV].round().astype(int)
        maj.insert(1, "含义", maj["majority"].map({
            "U": "用户键入", "A": "作答选项", "F": "功能或卡片按钮", "P": "推送问题", "W": "包装模板",
            "D": "医生卡", "C": "无内容", "X": "无法判定", "NOMAJ": "三票各异"}))
        maj = maj.sort_values(PV, ascending=False).reset_index(drop=True)
        metrics = [
            ("盲标串数", len(L)), ("每串票数", int(L["votes"].astype(str).str.len().iloc[0])),
            ("Fleiss_kappa_8类", round(_fleiss_from_votes(L["votes"]), 3)),
            ("三票全同占比%", round(100 * float((L["votes"].astype(str).map(lambda v: len(set(v)) == 1)).mean()), 1)),
            ("无多数串数", int((L["majority"] == "NOMAJ").sum())),
            ("由盲标一致决定的串数", int((ai["tier_source"] == "audit_unanimous").sum())),
            ("分票由规则决定的串数", int(((ai["tier_source"] == "rule") & dec).sum())),
            ("规则单独_精确率", round(float(tp.sum() / (tp.sum() + fp.sum())), 3)),
            ("规则单独_召回", round(float(tp.sum() / (tp.sum() + fn.sum())), 3)),
            ("规则单独_PV精确率", round(float(w[tp].sum() / (w[tp].sum() + w[fp].sum())), 3)),
            ("规则单独_PV召回", round(float(w[tp].sum() / (w[tp].sum() + w[fn].sum())), 3)),
            ("入挖掘AI串数", int((ai["tier"] == "user").sum())),
            ("其中裸词来源不明", int(ai.get("flag_bare_term_origin_unknown", pd.Series(False, index=ai.index)).fillna(False).astype(bool).sum())),
        ]
        out["audit"] = pd.DataFrame(metrics, columns=["指标", "值"])
        out["audit_majority"] = maj
    return out


#: 参考列的预注册条件（写在最终语料出来之前，见 configs/pool2_health.yaml）：每列与界面的
#: Cramér's V <= 0.55，且没有占比 >1% 的单边类。这里重新实测，并**断言它与本次运行实际声明的列一致**——
#: 条件不满足却声明了、或满足却没声明，都说明配置与预注册脱节，直接拒绝继续。
REF_CANDIDATES = {"健康": ["legacy_l2", "legacy_type", "legacy_dept"],
                  "医疗随机": ["legacy_l2", "legacy_dept", "legacy_type"],
                  "医疗3": ["legacy_l2", "legacy_dept", "legacy_type"]}
REF_V_MAX, REF_ONESIDED_MAX_SHARE = 0.55, 0.01


def reference_columns_check(domain: str, d: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in REF_CANDIDATES.get(domain, []) if c in d.columns]
    if not cols:
        return pd.DataFrame()
    cfg = run_dir(domain) / "config.resolved.yaml"
    declared = []
    if cfg.exists():
        import yaml
        declared = list((yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}).get("data", {}).get("reference_label_columns", []) or [])
    rows = []
    for c in cols:
        ct = pd.crosstab(d[c].astype(str), d["surface"])
        v = cramers_v(ct.values)
        one = ct[(ct == 0).any(axis=1)]
        big = int((one.sum(axis=1) / len(d) > REF_ONESIDED_MAX_SHARE).sum())
        ok = bool(v <= REF_V_MAX and big == 0)
        rows.append({"参考列候选": c, "类数": int(ct.shape[0]), "CramersV_与界面": round(v, 3),
                     "单边类数": int(len(one)), "占比超1%的单边类数": big,
                     "预注册条件满足": ok, "本次运行声明": c in declared})
        if cfg.exists():
            assert ok == (c in declared), (f"{domain}: 参考列 {c} 预注册条件{'满足' if ok else '不满足'}，"
                                           f"但运行{'没有' if ok else '却'}声明了它——配置与预注册脱节")
    return pd.DataFrame(rows)


def bare_term_sensitivity(dd: pd.DataFrame, srcs: list[str]) -> pd.DataFrame:
    """助手侧保留下来的裸词来源不明（键入的，还是从症状列表点出来的）。把它们拿掉以后，
    界面差异的方向与显著性还站得住吗？——每个类给两套数，并标出结论翻转的类。"""
    if "flag_bare_term_origin_unknown" not in dd.columns:
        return pd.DataFrame()
    se, asr = surface_groups(dd, srcs)
    flag = dd["flag_bare_term_origin_unknown"].fillna(False).astype(bool)
    S, A = dd[dd.source.isin(se)], dd[dd.source.isin(asr)]
    A2 = A[~flag.loc[A.index].to_numpy()]
    ns, na, na2 = len(S), len(A), len(A2)
    cs, ca, ca2 = S["_key"].value_counts(), A["_key"].value_counts(), A2["_key"].value_counts()
    disp = dd.drop_duplicates("_key").set_index("_key")["_disp"]
    rows = []
    for key in sorted(set(cs.index) | set(ca.index)):
        ks, ka, kb = int(cs.get(key, 0)), int(ca.get(key, 0)), int(ca2.get(key, 0))
        d1, lo1, hi1 = newcombe(ks, ns, ka, na)
        d2, lo2, hi2 = newcombe(ks, ns, kb, na2) if na2 else (float("nan"),) * 3
        sig1, sig2 = bool(lo1 > 0 or hi1 < 0), bool(na2 and (lo2 > 0 or hi2 < 0))
        rows.append({"key": key, "类目": disp.get(key, key), "搜索占比%": round(100 * ks / ns, 3),
                     "助手条数_全部": ka, "助手占比%_全部": round(100 * ka / na, 3),
                     "助手条数_去掉裸词": kb, "助手占比%_去掉裸词": round(100 * kb / na2, 3) if na2 else float("nan"),
                     "该类助手行里裸词占比%": round(100 * (ka - kb) / ka, 1) if ka else float("nan"),
                     "差_pp_全部": round(100 * d1, 3), "低_pp_全部": round(100 * lo1, 3), "高_pp_全部": round(100 * hi1, 3),
                     "差_pp_去掉裸词": round(100 * d2, 3), "低_pp_去掉裸词": round(100 * lo2, 3), "高_pp_去掉裸词": round(100 * hi2, 3),
                     "显著_全部": sig1, "显著_去掉裸词": sig2,
                     "结论翻转": bool(sig1 != sig2 or (sig1 and sig2 and np.sign(d1) != np.sign(d2)))})
    return pd.DataFrame(rows).sort_values("差_pp_全部").reset_index(drop=True)


# ---------------------------------------------------------------- 意图 ↔ 叶 的条件分布

def conditional_mix(d: pd.DataFrame, outer: str, inner: str, srcs: list[str],
                    min_n: int = 30, tag: str = "") -> pd.DataFrame:
    """同一个意图，在不同快照里主要落在哪一个叶（反之亦然）。

    这是「占比一样但实现方式不同」的检验：某个意图在搜索和助手里份额接近，内部的叶分布却可能
    完全不同。只在两边都 >= min_n 行时计算，否则 TVD 由噪声主导。

    **内层 TVD 必须配同源噪声上界。** 第一版没有这一列，两位独立复核员各自补算了一遍，并据此
    推翻了叙述里的若干条「一年没动 / 换个界面就动了」——在 n 只有几十到几百的格子里，两份来自
    同一分布的样本本来就能给出 0.2–0.3 的 TVD。上界 = 把两侧的行合起来随机对半切 B 次，取 TVD
    的 95 分位；观测值不超过它，这一格就读作「测不出差别」。"""
    rows = []
    for key, g in d.groupby(outer, sort=False):
        cnt = {s: int((g.source == s).sum()) for s in srcs}
        inner_order = sorted(g[inner].astype(str).unique())
        for i, a in enumerate(srcs):
            for b in srcs[i + 1:]:
                if cnt[a] < min_n or cnt[b] < min_n:
                    continue
                ga = g.loc[g.source == a, inner].astype(str)
                gb = g.loc[g.source == b, inner].astype(str)
                pa = ga.value_counts(normalize=True).reindex(inner_order).fillna(0)
                pb = gb.value_counts(normalize=True).reindex(inner_order).fillna(0)
                obs = float(0.5 * (pa - pb).abs().sum())
                # 同源噪声上界：把两侧合起来随机对半切（切成 n_a / n_b 两块，保持原来的大小比），
                # 看纯抽样能给出多大的 TVD。
                pool = np.concatenate([ga.to_numpy(), gb.to_numpy()])
                codes = pd.Index(inner_order).get_indexer(pool)
                K = len(inner_order)
                rng = rng_for("conditional_mix", tag, str(key), a, b)
                null = np.empty(NULL_B)
                for t in range(NULL_B):
                    idx = rng.permutation(len(codes))
                    x, y = codes[idx[:cnt[a]]], codes[idx[cnt[a]:]]
                    null[t] = 0.5 * np.abs(np.bincount(x, minlength=K) / len(x)
                                           - np.bincount(y, minlength=K) / len(y)).sum()
                hi = float(np.percentile(null, 95))
                rows.append({"外层": g["_disp"].iloc[0] if outer == "_key" else str(key),
                             "a": SRC_ZH[a], "b": SRC_ZH[b], "n_a": cnt[a], "n_b": cnt[b],
                             "内层TVD": round(obs, 4),
                             "同源噪声上界": round(hi, 4),
                             "超出噪声": bool(obs > hi),
                             "a主导内层": str(ga.value_counts().index[0]),
                             "a主导占比%": round(100 * float(pa.max()), 2),
                             "b主导内层": str(gb.value_counts().index[0]),
                             "b主导占比%": round(100 * float(pb.max()), 2),
                             "主导是否相同": bool(ga.value_counts().index[0] == gb.value_counts().index[0])})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 例子

def examples(d: pd.DataFrame, srcs: list[str], ok: pd.Series, per: int = 3) -> pd.DataFrame:
    rows = []
    for (key, s), g in d.groupby(["_key", "source"], sort=False):
        if s not in srcs:
            continue
        pool = g[ok.reindex(g.index, fill_value=False)]
        base = {"key": key, "类目": g["_disp"].iloc[0], "快照": SRC_ZH[s], "该类该快照条数": len(g),
                "可引条数": len(pool)}
        if not len(pool):
            rows.append({**base, "取法": "不引原文（该类该快照的行全部命中风控图层或引用护栏）",
                         "query": "", "pv_raw": float("nan")})
            continue
        for _, r in pool.sort_values("pv_norm", ascending=False).head(per).iterrows():
            rows.append({**base, "取法": "流量最高", "query": r["query"], "pv_raw": r["pv_raw"]})
        rest = pool.iloc[per:] if len(pool) > per else pool.iloc[0:0]
        if len(rest):
            for _, r in rest.sample(min(per, len(rest)), random_state=7).iterrows():
                rows.append({**base, "取法": "随机", "query": r["query"], "pv_raw": r["pv_raw"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 驱动

def _risk_indices(domain: str, n: int) -> set[int]:
    p = run_dir(domain) / "risk_screen.json"
    if not p.exists():
        return set()
    rs = json.loads(p.read_text(encoding="utf-8"))
    return {i for i in rs.get("flag_mask_indices", []) if 0 <= i < n}


_QUOTED = re.compile(r"[“”\"「」]([^“”\"「」]{2,40})[“”\"「」]")


def risk_strings(domain: str) -> set[str]:
    """运行自己的风控机器点名过的具体字符串。引用护栏的第三层——它们被点名过一次，就不再引原文。"""
    out: set[str] = set()
    gen = run_dir(domain)
    p = gen / "tree_naming.json"
    if p.exists():
        tn = json.loads(p.read_text(encoding="utf-8"))
        for f in (tn.get("risk_report") or {}).get("findings", []) or []:
            out.update(str(x).strip() for x in (f.get("evidence") or []))
            out.update(m.strip() for m in _QUOTED.findall(str(f.get("rationale") or "")))
        for nmg in tn.get("namings", []) or []:
            out.update(m.strip() for m in _QUOTED.findall(str(nmg.get("risk_reason") or "")))
        for fam in tn.get("families_final", []) or []:
            out.update(m.strip() for m in _QUOTED.findall(str(fam.get("risk") or "")))
    p = gen / "risk_screen.json"
    if p.exists():
        rs = json.loads(p.read_text(encoding="utf-8"))
        for c in rs.get("categories", []) or []:
            if c.get("exemplar"):
                out.add(str(c["exemplar"]).strip())
            out.update(str(x).strip() for x in (c.get("samples") or []))
    return {x for x in out if x}


def build(domain: str) -> dict:
    d = load(domain).reset_index(drop=True)
    srcs = [s for s in SOURCES if (d.source == s).any()]
    nm = _names(domain, d)
    risk_idx = _risk_indices(domain, len(d))
    named = risk_strings(domain)
    q = d["query"].astype(str)
    cooc = q.str.contains(QUOTE_COOC[0]) & q.str.contains(QUOTE_COOC[1])
    never = d["td_l1"].astype(str).isin(NEVER_QUOTE_CLASSES.get(domain, set()))
    extra = extra_quote_mask(domain, q)
    screened = q.isin(screened_quote_block(domain))
    quotable = (~d.index.isin(risk_idx) & ~q.str.contains(QUOTE_BLOCK) & ~cooc
                & ~q.isin(named) & ~never & ~extra & ~screened)
    out = WORK / domain / "snapshot_classes"
    out.mkdir(parents=True, exist_ok=True)
    summary = {"domain": domain, "run": run_dir(domain).parent.name, "n_rows": len(d),
               "snapshots": {SRC_ZH[s]: int((d.source == s).sum()) for s in srcs},
               "risk_flagged_rows": len(risk_idx), "quotable_rows": int(quotable.sum()),
               "risk_named_strings": len(named),
               "never_quote_class_rows": int(never.sum()),
               **({"extra_quote_block_rows": int(extra.sum())} if domain in EXTRA_QUOTE_BLOCK else {}),
               **({"screened_quote_block_rows": int(screened.sum())} if domain in SCREENED_QUOTE_BLOCK else {}),
               "l1_defined": len(nm["l1"]), "l1_delivered": int(d["td_l1"].nunique()),
               "l1_defined_but_unused": sorted(set(nm["l1"]) - set(d["td_l1"].astype(str))),
               "leaves_named": len(nm["leaf"]), "leaves_delivered": int(d["bu_leaf"].nunique()),
               "levels": {}}
    cov, tvd, power, tns, tms = [], [], [], [], []
    frames: dict[str, dict[str, pd.DataFrame]] = {}
    for level, zh in LEVELS:
        dd = _class_frame(d, level, nm)
        m = enrich(matrix(dd, srcs), dd, level, nm, quotable)
        ab = absence(dd, srcs)
        nc = newcombe_all(dd, srcs)
        isp = interface_split(dd, srcs)
        pt = pairwise_tvd(dd, srcs, tag=f"{domain}|{level}")
        pt.insert(0, "层级", zh)
        tvd.append(pt)
        tn, tm = topn(dd, srcs, zh)
        tns.append(tn)
        tms.append(tm)
        # 表 1 的前3/前5 阶梯延长到前10，并配一个流量口径。**从 topn() 取，不另算一遍**——
        # 同一个数出现在表 1 与表 2.2 两处，只有同源才不会漂。
        cov.append(coverage(dd, srcs, zh)
                   .merge(tn[["快照", "层级", "前10行占比%", "前10流量占比%", "前10流量摆动pp",
                              "前10流量占比%低", "前10流量占比%高", "流量有效n"]],
                          on=["快照", "层级"], how="left")
                   .rename(columns={"前10行占比%": "前10占比%"})
                   .round({"前10占比%": 2, "前10流量占比%": 2, "前10流量摆动pp": 2,
                           "前10流量占比%低": 2, "前10流量占比%高": 2, "流量有效n": 1}))
        ex = examples(dd, srcs, quotable) if level in ("td_l1", "bu_leaf") else pd.DataFrame()
        sg = signature(dd, srcs)
        cf = confidence(dd, srcs)
        pw = assistant_only_power(dd, srcs, zh)
        if pw:
            power.append(pw)
        for nmx, t in [("matrix", m), ("absence", ab), ("newcombe", nc), ("interface", isp),
                       ("signature", sg), ("confidence", cf), ("examples", ex)]:
            if len(t):
                t.to_csv(out / f"{nmx}_{level}.csv", index=False, encoding="utf-8-sig")
        frames[level] = {"matrix": m, "absence": ab, "newcombe": nc, "interface": isp,
                         "signature": sg, "confidence": cf, "examples": ex}
        summary["levels"][zh] = {
            "n_classes": int(dd["_key"].nunique()),
            "独占某快照的类": int((m["独占快照"] != "").sum()),
            "全快照都出现的类": int((m["出现快照数"] == len(srcs)).sum()),
            "真缺席条目": int((ab["判定"] == "真缺席").sum()) if len(ab) else 0,
            "仅搜索": int((isp["归属"] == "仅搜索").sum()),
            "仅助手": int((isp["归属"] == "仅助手").sum()),
            "界面差异显著的类": int(isp["显著"].sum()),
            "特征类条目(显著高于其余全部快照)": int((sg["方向"] == "显著高于其余全部").sum()) if len(sg) else 0,
            "反特征类条目(显著低于其余全部快照)": int((sg["方向"] == "显著低于其余全部").sum()) if len(sg) else 0,
            "CramersV_快照×类": round(cramers_v(pd.crosstab(dd["_key"], dd["source"]).values), 4)}
    pd.concat(cov, ignore_index=True).to_csv(out / "coverage.csv", index=False, encoding="utf-8-sig")
    pd.concat(tns, ignore_index=True).to_csv(out / "topn_coverage.csv", index=False, encoding="utf-8-sig")
    pl = product_layer(domain)
    for k, t in pl.items():
        t.to_csv(out / f"product_layer_{k}.csv", index=False, encoding="utf-8-sig")
    refc = reference_columns_check(domain, d)
    if len(refc):
        refc.to_csv(out / "reference_columns_check.csv", index=False, encoding="utf-8-sig")
    sens = {}
    for level in ("td_l1", "bu_leaf"):
        sb = bare_term_sensitivity(_class_frame(d, level, nm), srcs)
        if len(sb):
            sb.to_csv(out / f"sensitivity_bare_{level}.csv", index=False, encoding="utf-8-sig")
            sens[level] = sb
    if sens:
        summary["bare_term_sensitivity"] = {
            lv: {"类数": int(len(t)), "显著_全部": int(t["显著_全部"].sum()),
                 "显著_去掉裸词": int(t["显著_去掉裸词"].sum()), "结论翻转": int(t["结论翻转"].sum())}
            for lv, t in sens.items()}
    pd.concat(tms, ignore_index=True).to_csv(out / "topn_members.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(power).to_csv(out / "assistant_only_power.csv", index=False, encoding="utf-8-sig")
    pd.concat(tvd, ignore_index=True).to_csv(out / "pairwise_tvd.csv", index=False, encoding="utf-8-sig")
    dl1 = _class_frame(d, "td_l1", nm)
    cm = conditional_mix(dl1, "_key", "bu_leaf_name", srcs, tag=f"{domain}|intent_leafmix")
    cm.to_csv(out / "intent_leafmix.csv", index=False, encoding="utf-8-sig")
    dlf = _class_frame(d, "bu_leaf", nm)
    lm = conditional_mix(dlf, "_key", "td_l1_name", srcs, tag=f"{domain}|leaf_intentmix")
    lm.to_csv(out / "leaf_intentmix.csv", index=False, encoding="utf-8-sig")
    frames["_cov"] = {"power": pd.DataFrame(power),
                      "coverage": pd.concat(cov, ignore_index=True),
                      "topn": pd.concat(tns, ignore_index=True),
                      "topn_members": pd.concat(tms, ignore_index=True),
                      "product_layer": pl, "sensitivity": sens,
                      "tvd": pd.concat(tvd, ignore_index=True),
                      "intent_leafmix": cm, "leaf_intentmix": lm}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    _workbook(domain, srcs, frames, summary)
    print(f"{domain}: {len(d):,} 行 · " + " · ".join(f"{SRC_ZH[s]} {int((d.source==s).sum()):,}" for s in srcs)
          + " · " + " · ".join(f"{zh} {v['n_classes']}类(仅搜索{v['仅搜索']}/仅助手{v['仅助手']})"
                               for zh, v in summary["levels"].items()))
    return summary


ILLEGAL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _clean(t: pd.DataFrame) -> pd.DataFrame:
    t = t.copy()
    for c in t.columns:
        if t[c].dtype == object:
            t[c] = t[c].map(lambda x: ILLEGAL.sub("", x) if isinstance(x, str) else x)
    return t


def _workbook(domain: str, srcs: list[str], frames: dict, summary: dict) -> Path:
    gen = run_dir(domain)
    books = sorted(gen.glob("*_query_挖掘结果.xlsx"))
    stem = books[0].stem.replace("_query_挖掘结果", "") if books else DOMAINS[domain]
    outdir = gen / "postprocessed"
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{stem}_意图与聚类叶_跨快照对比.xlsx"
    note = pd.DataFrame([
        ["这份表是什么", f"{domain}（运行 {summary['run']}/{gen.name}）里每一个自上而下意图、每一个自下而上聚类叶，"
                        f"在 {len(srcs)} 个快照上的逐类占比、差异与独有性。"],
        ["快照", "；".join(f"{SRC_ZH[s]} n={summary['snapshots'][SRC_ZH[s]]:,}" for s in srcs)],
        ["占比的口径", "一律是快照内占比（该类条数 ÷ 该快照总行数）。各快照行数相差 10 倍，跨快照比原始条数没有意义。"],
        ["95CI", "Wilson 区间。两快照之差用 Newcombe 混合评分区间，区间不含 0 记为显著。"],
        ["均衡归属%", "该快照内占比 ÷ 各快照内占比之和。把每个快照当成同等大小来看这个类归谁，"
                     "不受搜索行数是助手 10 倍的影响。"],
        ["指数 / 均衡指数", "指数 = 快照内占比 ÷ 全语料占比（与 class_profile.csv 同义，分母里 87% 是搜索行）；"
                          "均衡指数 = 快照内占比 ÷ 各快照占比的未加权均值。"],
        ["0 条不等于不存在", "absence 页对每个「该快照 0 条」给出参照占比下的期望条数与 P(0)：期望≥5 且 P(0)<1% 记"
                            "「真缺席」，期望<3 记「不可判定」。950 行里 0 条的单侧 97.5% 上界仍有约 0.39%。"],
        ["TVD", "总变差距离：把一个快照的类目分布变成另一个，需要改动多少比例的行。附自助法区间与"
                "同一快照对半切的噪声上界——低于噪声上界的距离不可读。"],
        ["L2 子意图没有名字", "td_l2 是在 L1 内部按表示做的几何细分（subintents.json），运行没有给它们命名，"
                             "所以这里用 `L1__序号` 原样呈现，含义看例子页的代表串。"],
        ["家族与叶用的是交付分区", "家族取 tree_naming.json 的 families_final（p8 治理之后的那棵树），"
                                  "不是 audit.families；叶取 namings 里交付分区实际出现的那些 id。"],
        ["引用护栏", f"风控图层命中的 {summary['risk_flagged_rows']:,} 行一律不引原文，另加一层露骨内容与号码串的硬规则；"
                    "整类都不可引时，例子页只给条数。"],
        ["前N名覆盖页 / 前十成员页", "每个快照**各排各的**前 1/3/5/10 名，合起来覆盖这个快照多少行、多少流量。"
                        "只能竖着读：同一名次在两列里通常不是同一个类。「前10流量占比%」是**同样这十个类**"
                        "（仍按行占比选出）按 pv_norm 加权后的占比，不是「按流量重排以后的前十」。"
                        "成员由「前十成员」页定下来，报告表 2.2 的第1..第10 就是这一页，两者不可能对不上。"],
        ["流量口径能承多重", "流量占比的有效样本量是 1/Σw²，**不是行数**：金融 2026搜索 9,999 行、流量口径只有 111.5，"
                            "单行最高占该快照流量的 7.39%。而且 `助手头部1k` 是按 PV 取的前 1,000 条**再过清洗层**，"
                            "被清洗掉的行带走了该导出大部分原始 PV（软件 98.47%、书籍文档 88.45%、影视 78.33%，"
                            "见 `work/build_audit.csv` 的 `pv_dropped_%`），所以助手头部那一格的流量主要是清洗规则的函数。"
                            "行占比不受这些影响。"],
        ["边界并列怎么读", "「边界并列」为真时，第 10 名的位置上有多个条数相同的类，只能选其中一部分。"
                          "**行占比的合计对挑法免疫**（并列的类条数相同），前3/前5/前10合计与前10之外都是定值；"
                          "**流量合计不免疫**——并列的类 pv_norm 不同，所以另给「前10流量占比%低/高」与"
                          "「前10流量摆动pp」，那是挑法能造成的全部范围。教育 助手随机1k 的叶层摆动 5.25pp。"],
        ["这次是 fast 模式", "单标注员，kappa 不存在（不是 1.0）。下面每个占比都建立在没有第二意见校验的标注层上。"],
        ["同串同标签", "同一个字符串在一次运行里必然拿到同一个标签（五域零例外），所以「两个快照共有的串标签一致」"
                      "是构造必然，不是测量结果。"],
        ["快照特征类页", "「独占某快照」几乎恒为 0（一个类只要在别处出现一条就不算独占），对「这个快照有什么特别的」"
                        "几乎没有信息量。特征类换了一个能承重的口径：该类在这个快照里的占比，与其余**每一个**快照"
                        "逐一做 Newcombe 检验都显著更高。快照越多，条件越严。"],
        ["意图内的叶分布 / 叶内的意图分布页", "这两页的内层 TVD 必须先跟同一行的「同源噪声上界」比。"
                                          "格子的 n 只有几十到几百，两份来自同一个分布的样本本来就能给出 0.2–0.3 的 TVD。"
                                          "上界 = 把两侧的行合起来、按原来的大小比随机对半切 300 次取 95 分位。"
                                          "`超出噪声=否` 的格子既不能读成「内部换了」，**也不能读成「内部没换」**，"
                                          "只能写「本表测不出」。"],
        ["自助法区间是可复现的", "每一处重抽都按 (用途, 领域, 层级, 快照对) 各自播种，所以同一格永远给同一个区间，"
                              "与同一次运行里别处算了什么无关；单独重跑一个领域与在五域批量里跑，结果逐位相同。"],
        ["标注可靠度页", "占比稳不代表边界稳。td_confidence 是自上而下分类器对该行的把握（0–1），"
                        "td_ambiguous / bu_ambiguous 是两条路线各自的歧义标记。按（类, 快照）汇总以后，"
                        "就能看出某一格的占比该打几折。「置信度最低快照」只在该快照 ≥20 行时参与比较。"],
    ], columns=["项", "说明"])
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        _clean(note).to_excel(w, sheet_name="口径说明", index=False)
        _clean(pd.DataFrame([{"层级": k, **v} for k, v in summary["levels"].items()])).to_excel(
            w, sheet_name="总览", index=False)
        _clean(frames["_cov"]["coverage"]).to_excel(w, sheet_name="快照画像", index=False)
        # 2 位，和报告、和「快照画像」页逐位一致。取 3 位会让这一页印 98.235 而「快照画像」印
        # 98.24——同一个量在同一个工作簿里两个写法。**未舍入的值在 topn_coverage.csv 里。**
        _clean(frames["_cov"]["topn"].round(2)).to_excel(w, sheet_name="前N名覆盖", index=False)
        _clean(frames["_cov"]["topn_members"].round(2)).to_excel(w, sheet_name="前十成员", index=False)
        _PL = {"tiers": "产品层_分层", "top": "产品层_头部串", "audit": "盲标_一致度", "audit_majority": "盲标_多数分布",
               "aggregation": "合并前后"}
        for k, t in frames["_cov"].get("product_layer", {}).items():
            _clean(t).to_excel(w, sheet_name=_PL[k], index=False)
        for lv, t in frames["_cov"].get("sensitivity", {}).items():
            _clean(t).to_excel(w, sheet_name=f"裸词敏感性_{SHEET[lv]}", index=False)
        if len(frames["_cov"]["power"]):
            _clean(frames["_cov"]["power"]).to_excel(w, sheet_name="仅助手类的检出力", index=False)
        _clean(frames["_cov"]["tvd"]).to_excel(w, sheet_name="快照两两距离", index=False)
        for level, zh in LEVELS:
            sh = SHEET[level]
            f = frames[level]
            _clean(f["matrix"]).to_excel(w, sheet_name=f"{sh}_逐快照占比", index=False)
            _clean(f["interface"]).to_excel(w, sheet_name=f"{sh}_搜索vs助手", index=False)
            if len(f["absence"]):
                _clean(f["absence"]).to_excel(w, sheet_name=f"{sh}_缺席可检出性", index=False)
            if len(f["signature"]):
                _clean(f["signature"]).to_excel(w, sheet_name=f"{sh}_快照特征类", index=False)
            _clean(f["confidence"]).to_excel(w, sheet_name=f"{sh}_标注可靠度", index=False)
            _clean(f["newcombe"]).to_excel(w, sheet_name=f"{sh}_逐对差异", index=False)
            if len(f["examples"]):
                _clean(f["examples"]).to_excel(w, sheet_name=f"{sh}_例子", index=False)
        _clean(frames["_cov"]["intent_leafmix"]).to_excel(w, sheet_name="意图内的叶分布", index=False)
        _clean(frames["_cov"]["leaf_intentmix"]).to_excel(w, sheet_name="叶内的意图分布", index=False)
    print(f"  → {path.relative_to(ROOT)}")
    return path


if __name__ == "__main__":
    doms = sys.argv[1:] or available()
    allsum = [build(x) for x in doms]
    (WORK / "snapshot_classes_all.json").write_text(
        json.dumps(allsum, ensure_ascii=False, indent=1), encoding="utf-8")
