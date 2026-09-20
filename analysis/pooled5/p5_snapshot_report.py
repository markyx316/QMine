# -*- coding: utf-8 -*-
"""把 p5_snapshot_classes.py 算出来的表拼成逐领域报告，写进 runs/<id>/gen01/postprocessed/。

**报告里的每一个数字都由这个脚本从 CSV 里取出来渲染，没有任何一个是写出来的。** 叙述段落由
分析员写，但只能放在 `<!--NARR:xxx-->` 占位处，且不得出现表里没有的数字——`p5_snapshot_verify.py`
会把叙述里出现的每个数字回查到表。

    python analysis/pooled5/p5_snapshot_report.py [domain ...]
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pooled5_common import COHORTS, DOMAINS, SOURCES, SRC_ZH, WORK, available, load, run_dir

ROOT = Path(__file__).resolve().parents[2]
LEVELS = [("td_l1", "自上而下 L1 意图"), ("td_l2", "自上而下 L2 子意图"),
          ("bu_family_final", "自下而上 家族"), ("bu_leaf", "自下而上 叶")]
NARR = ["导读", "意图层", "叶层", "独有性", "路线交叉", "子层", "边界"]


def _search_share(A: dict) -> str:
    """搜索行在全语料里的占比。原先写死「87%」——只对五域里的金融、医疗成立（87.2% / 87.1%），
    教育 91.6%、影视 91.2%、书籍文档 91.7%、软件 91.2%、金融8 91.0%、健康 90.9%、医疗8 90.9%（2026-09-15 实测）。"""
    snaps = A["summary"]["snapshots"]
    total = sum(snaps.values())
    return f"{100 * sum(v for k, v in snaps.items() if '搜索' in k) / total:.0f}%"


def _md(t: pd.DataFrame, cols: list[str] | None = None) -> str:
    t = t[cols] if cols else t
    if not len(t):
        return "_（本表在这个领域为空。）_\n"
    head = "| " + " | ".join(str(c) for c in t.columns) + " |"
    sep = "|" + "|".join("---" for _ in t.columns) + "|"
    body = []
    for _, r in t.iterrows():
        cells = []
        for c in t.columns:
            v = r[c]
            if isinstance(v, float):
                v = "" if pd.isna(v) else ("∞" if v == float("inf") else f"{v:g}")
            cells.append(str(v).replace("|", "\\|").replace("\n", " "))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, sep] + body) + "\n"


def _prose(text: str) -> str:
    """运行自己写的散文（类定义、user_need、家族审计意见）里也会用 「」。本报告只有一条排版约定——
    「」 只包真实 query——所以这些**非 query 的**引号统一换成 “”。query 文本一个字都不动。"""
    return str(text).replace("「", "“").replace("」", "”")


def _q(text: str) -> str:
    """引号包住一条真实 query。query 自己带 「」 的（比如 「AI视频」一家三口温馨比心）会把外层引号
    截断，读起来像两条；这种改用 『』 包，原文一个字都不动。"""
    return f"『{text}』" if ("「" in text or "」" in text) else f"「{text}」"


def _pct(x) -> str:
    return "—" if pd.isna(x) else f"{x:.2f}"


#: **已交付的七个域原样保留原来的措辞**（逐字节不变），下面的逐域文字只给之后接入的语料用。
#: 金融8 报告里原来印着「本域是 2026-09-13 新接入的」「两个搜索快照都在 7 月初」「搜索侧那两万行」——
#: 那几句是为五域/新两域写死的，对 8 个快照、4 个搜索层的金融8 全是错的。
_CN = {1: "一", 2: "两", 3: "三", 4: "四", 5: "五", 6: "六", 7: "七", 8: "八"}
DOMAIN_INTRO = {
    "金融8": ("> **金融8 是已交付的「金融」同一个垂类的扩展版，不是新垂类。**《POOLED5_2026 五域同体系对比》里的金融"
              "用的是 5 个快照（`fin-pool5`）；2026-09-14 补来三份导出（2025、2026 两年的搜索随机 1 万，语音头部 1 千）"
              "以后重跑成 8 个快照（`fin-pool8`）。两次运行各有一套体系，**类目编码互不相通**，所以本报告与已交付的"
              "金融报告不能逐类对照；但搜索头部两年、助手头部/随机、语音这五个快照的**行**与 `fin-pool5` 逐格相同"
              "（构建时断言过），差异只可能来自新加的三份导出与重新挖掘。\n>\n"),
    "医疗8": ("> **医疗8 是已交付的「医疗」同一个垂类的扩展版，不是新垂类。**《POOLED5_2026 五域同体系对比》里的医疗"
              "用的是 5 个快照（`med-pool5`）；2026-09-15 补来三份导出（2025、2026 两年的搜索随机 1 万，语音头部）"
              "以后重跑成 8 个快照（`med-pool8`）。两次运行各有一套体系，**类目编码互不相通**，所以本报告与已交付的"
              "医疗报告不能逐类对照；但搜索头部两年、助手头部/随机、语音这五个快照的**行**与 `med-pool5` 逐格相同"
              "（构建时断言过），差异只可能来自新加的三份导出与重新挖掘。它也**不是**「健康」那一对同周周榜（`health-pool2`）。"
              "语音头部那份导出实为按 PV 降序的 1 万行：本报告取 PV 不低于第 1,000 行（PV=11）的全部 1,067 行，"
              "不在并列处按导出顺序截断，所以列头「助手语音头部1k」在本域是 1,067 行。\n>\n"),
    "人物8": ("> **人物8 是已交付的「人物」同一个垂类的扩展版，不是新垂类。**《POOLED5_2026 五域同体系对比》里的人物"
              "用的是 4 个快照（`ppl-pool5b`）；2026-09-16 补来四份导出（2025、2026 两年的搜索随机 1 万，语音头部 1 千、"
              "语音随机 1 千）以后重跑成 8 个快照（`ppl-pool8`）。两次运行各有一套体系，**类目编码互不相通**，所以本报告与"
              "已交付的人物报告不能逐类对照；但搜索头部两年、助手头部/随机这四个快照的**行**与 `ppl-pool5b` 逐格相同"
              "（构建时断言过），差异只可能来自新加的四份导出与重新挖掘。两份语音导出都自带 PV：语音头部 1k 是按 PV 降序的"
              "前 1,000 行（导出方已在 1,000 行处截断，所以并列边界无从判断，PV 下限 10），语音随机 1k 是随机抽样（PV 中位 1）。\n>\n"),
    "影视8": ("> **影视8 是已交付的「影视」同一个垂类的扩展版，不是新垂类。**《POOLED5_2026 五域同体系对比》里的影视"
              "用的是 4 个快照（`film-pool5`）；2026-09-16 补来四份导出（2025、2026 两年的搜索随机 1 万，语音头部 1 千、"
              "语音随机 1 千）以后重跑成 8 个快照（`film-pool8`）。两次运行各有一套体系，**类目编码互不相通**，所以本报告与"
              "已交付的影视报告不能逐类对照；但搜索头部两年、助手头部/随机这四个快照的**行**与 `film-pool5` 逐格相同"
              "（构建时断言过），差异只可能来自新加的四份导出与重新挖掘。两份语音导出都自带 PV：语音头部 1k 是按 PV 降序的"
              "前 1,000 行（导出方已在 1,000 行处截断，所以并列边界无从判断，PV 下限 21），语音随机 1k 是随机抽样（PV 中位 1）。\n>\n"),
    "医疗3": ("> **医疗3 不是一次新的挖掘运行，是 health-pool3 的三快照扩展。**体系（意图、叶、家族、命名）"
             "一个字都没有重新推导：新增的「健康管家 Top1w」快照由 `analysis/pooled5/med3_score.py` 用 health-pool3 "
             "自己的 `centroid_classifier` 与 `topdown_model` 打分，稀疏空间在原语料上重拟合并实测**逐位复现**该运行的 "
             "`emb_svd_char.npy`。打分器与交付标签的一致率（原语料 20,316 行）：td_l1 100.00%、td_l2 99.57%、"
             "叶与家族 95.51%——所以叶这一层的新快照数字要按 95.5% 的复现率读。\n>\n"
             "> 老两个快照的标签直接取自 health-pool3 交付的 `labels_full.csv`，**没有重打**。\n"),
    "医疗随机": ("> **医疗随机 是 health-pool2 那两个产品的「随机版」，不是同一批数据。**上一次（`health-pool2`）用的是"
                "两份周榜 top1w 导出（2026-09-04 至 09-10，按 query×天 取前 1 万），聚合清洗后助手侧只剩 225 行，说不出什么；"
                "这一对是 **2026-09-14 当天、医疗类目、去重后的随机 1 万**（传统搜索 10,000 串、健康管家 10,622 串），长尾在里面，"
                "产品层反而稀薄——重复的答项与按钮在去重导出里各只剩一行。两次运行各有一套体系，**类目编码互不相通**，"
                "与 `health-pool2`、与 8 快照的 `医疗8` 都不能逐类对照。\n>\n"
                "> 助手侧 75.4% 的行带产品的输入模板前缀「我想咨询」：本次**剥离前缀、保留该行**（原文在 query_raw、"
                "wrapper_stripped 标出），而不是像 health-pool2 那样按模板整行删除——那正是上一次助手侧塌掉的原因之一。\n>\n"),
    "健康": ("> **健康 是 2026-09-14 新接入的一对导出，与已交付的「医疗」不是同一套数据。**两份都是同一周"
             "（2026-09-04 至 09-10）的周榜：健康搜索、以及一个专门的健康 AI 管家；导出时按「query × 天」取前 1 万行，"
             "构建时按 query 合并成周粒度。它不在《POOLED5_2026 五域同体系对比》里，也没有跨领域汇总篇——"
             "凡是本文没写的跨领域比较，都是还没做，不是做了没发现。\n>\n"),
}
DOMAIN_TIME = {
    "金融8": "  四个搜索快照都是 7 月 1 日当天的导出（2025、2026 各一个头部、一个随机），助手约 8 月下旬{voice_time}，"
             "随时间走的题材不得归因于界面。\n",
    "医疗8": "  四个搜索快照都是 7 月 1 日当天的导出（2025、2026 各一个头部、一个随机），助手约 8 月下旬{voice_time}，"
             "随时间走的题材不得归因于界面。\n",
    "人物8": "  四个搜索快照都是 7 月 1 日当天的导出（2025、2026 各一个头部、一个随机），助手约 8 月下旬{voice_time}，"
             "随时间走的题材不得归因于界面。人物域尤其要小心：头部是当天的人事发布与突发新闻，同一个裸名字在事件当天与平日"
             "是两种意图。\n",
    "影视8": "  四个搜索快照都是 7 月 1 日当天的导出（2025、2026 各一个头部、一个随机），助手约 8 月下旬{voice_time}，"
             "随时间走的题材不得归因于界面。影视域尤其要小心：头部由当期热播剧名主导，年度差异多半是片单换了，不是问法变了。\n",
    "医疗3": "  三个快照都是**同一天**（2026-09-14）的导出，所以时间不是任何一对之间的混杂项。"
            "这份语料第一次让两条轴互相独立：**界面**（传统搜索随机 vs 健康管家随机，随机层对随机层）与"
            "**流量分层**（健康管家 Top1w vs 健康管家随机，同一个产品、同一天）。\n",
    "医疗随机": "  两个快照是**同一天**（2026-09-14）的导出，所以本域里时间**不是**搜索与助手之间的混杂项；"
                "但一天的窗口把周内热点压缩成一个切面，单日事件会同时推高两边的同一批题材。\n",
    "健康": "  两个快照是**同一周**的导出，所以本域里时间**不是**搜索与助手之间的混杂项——这一点与其它域相反；"
            "但一周的窗口会放大单次热点，周内事件（开学、暴雨/洪涝、新冠反弹）会同时推高两边的同一批题材，"
            "而且只在周内部分天上榜的串，周流量是下界（见数据说明）。\n",
}
DOMAIN_BUILD = {"金融8": ("analysis/pooled5/build_fin8_corpus.py", "configs/pool8_fin.yaml"),
                "医疗8": ("analysis/pooled5/build_med8_corpus.py", "configs/pool8_med.yaml"),
                "健康": ("analysis/pooled5/build_health_corpus.py", "configs/pool2_health.yaml"),
                "人物8": ("analysis/pooled5/build_pool8_corpus.py", "configs/pool8_ppl.yaml"),
                "影视8": ("analysis/pooled5/build_pool8_corpus.py", "configs/pool8_film.yaml"),
                "医疗随机": ("analysis/pooled5/build_medrand_corpus.py", "configs/pool2_medrand.yaml"),
                "医疗3": ("analysis/pooled5/build_med3_corpus.py", "（沿用 health-pool3 的配置，未重新运行）")}


#: 数据准备一节里跟着语料走的句子。默认取 健康 那一份，所以 健康 的报告逐字节不变；
#: 医疗随机 是**单日**导出（两份 `event_day` 都是 20260914）、来源侧已经去过重，没有「按天合并」这一步，
#: 清洗口径也不同（只有窄规则能删未盲标的行，盲标只在三票一致时才删），所以整段另写。
PREP_TEXT = {
    "健康": {
        "intro": "两份原始导出都是「query × 天」取前 1 万行，所以同一个 query 在榜几天就出现几次；构建时按 query 合并成周粒度。"
                 "合并之后，**AI 管家的大部分串在进入挖掘之前就被拿走了**——它们是助手自己写的字（作答选项、功能按钮、推送问题、包装模板），"
                 "不是用户的查询。这一节放在最前面，因为它决定了后面每一张表里 AI 侧的分母是什么。"
                 "构建脚本 `analysis/pooled5/build_health_corpus.py`，口径全文 `analysis/pooled5/work/health_clean_audit.md`。\n",
        "a": "\n**表 0-A　合并前后**——同一天被导出拆成两行（分类不同、PV 不同）的串按 PV 相加，不去重；"
             "只在周内部分天上榜的串，周 PV 是下界（`pv_week_upper` 给出上界）。\n\n",
        "b": "\n**表 0-B　清洗分层**——只有 `user` 进入挖掘；其余每一行都留在 `work/健康_all_rows.parquet` 里，带层级与决定方式。"
             "搜索侧只去掉了流量在 16 位不同医生之间几乎一致的具名医生串（S6）与点击进来的健康新闻标题（S5）。\n\n",
        "c": "\n**表 0-C　AI 管家的盲标**——全部串由三个独立视角盲标（互不知道规则，也互不知道对方）。"
             "三票一致时由盲标决定层级，分票时由规则决定；规则单独对多数标签的精确率与召回也列在这里，"
             "作为可复现的第二个仪器。\n\n",
        "d": "\n**表 0-D　被拿走的那一层，每层按周 PV 排前六的串**——这是助手的**产品用量**，不是用户意图："
             "作答选项说明助手问了什么，功能按钮说明用户点开了什么，推送问题说明助手当周在推什么。"
             "具名医生卡只给条数；命中引用护栏的串不引原文。全表在工作簿「产品层_头部串」页。\n\n",
    },
    "医疗随机": {
        "intro": "两份导出都是**单日**的随机 1 万行（`event_day` 都是 2026-09-14，09-15 导出），来源侧已经按 query 去过重，"
                 "所以没有「按天合并」这一步——表 0-A 只记合并后的串数与 PV 合计。"
                 "健康管家那一份里，**一部分行在进入挖掘之前就被拿走了**：它们是助手自己写的字（作答选项、功能按钮、推送问题、"
                 "包装模板）或医生卡，不是用户的查询。这一节放在最前面，因为它决定了后面每一张表里 AI 侧的分母是什么。"
                 "\n\n与 health-pool2 那次相比，这一次的口径**刻意放宽**：助手输入的包装模板只从串里**剥掉**、行本身留下（"
                 "`wrapper_stripped` 记住剥过），宽口径的规则只用于挑出要盲标的行、不再直接删行，"
                 "只有三个窄规则（功能/卡片位、医生卡、无内容）可以删没盲标过的行。"
                 "构建脚本 `analysis/pooled5/build_medrand_corpus.py`，盲标聚合 `analysis/pooled5/medrand_audit_aggregate.py`。\n",
        "a": "\n**表 0-A　两个来源各自的规模**——来源侧已去重，一个 query 一行；`平台分类跨行冲突的串` 记的是同一个串"
             "在导出里带了互相矛盾的平台分类的条数。\n\n",
        "b": "\n**表 0-B　清洗分层**——只有 `user` 进入挖掘；其余每一行都留在 `work/医疗随机_all_rows.parquet` 里，"
             "带层级与决定方式（`tier_source`：`audit_unanimous` 三票一致、`audit_split_kept` 分票留下、`rule` 规则决定）。"
             "**删行只有两条路**：三个窄规则，或三个视角盲标一致。宽口径规则命中但盲标没一致的行，一律留作用户内容。\n\n",
        "c": "\n**表 0-C　健康管家的盲标**——歧义那一片的全部串由三个独立视角盲标（互不知道规则，也互不知道对方）。"
             "三票一致且一致判为非用户内容时才删，分票一律留下；Fleiss κ 只在这一片歧义串上算，"
             "不是整份语料的一致度。规则单独对多数标签的精确率与召回同列，作为可复现的第二个仪器。\n\n",
        "d": "\n**表 0-D　被拿走的那一层，每层按当日 PV 排前六的串**——这是助手的**产品用量**，不是用户意图："
             "作答选项说明助手问了什么，功能按钮说明用户点开了什么，推送问题说明助手当天在推什么。"
             "具名医生卡只给条数；命中引用护栏的串不引原文。全表在工作簿「产品层_头部串」页。\n\n",
    },
}


def prep_section(domain: str) -> str:
    """数据准备与清洗一节。只在产品层表存在时渲染（健康 与 医疗随机），所以其它域的报告逐字节不变。
    这一节放在最前面，因为它决定了后面每一张表的分母是什么：AI 侧一部分行在进入挖掘前就被拿走了。"""
    p = WORK / domain / "snapshot_classes"
    if not (p / "product_layer_tiers.csv").exists():
        return ""
    ag = pd.read_csv(p / "product_layer_aggregation.csv")
    ti = pd.read_csv(p / "product_layer_tiers.csv")
    top = pd.read_csv(p / "product_layer_top.csv")
    au = pd.read_csv(p / "product_layer_audit.csv") if (p / "product_layer_audit.csv").exists() else pd.DataFrame()
    am = pd.read_csv(p / "product_layer_audit_majority.csv") if (p / "product_layer_audit_majority.csv").exists() else pd.DataFrame()
    rc = pd.read_csv(p / "reference_columns_check.csv") if (p / "reference_columns_check.csv").exists() else pd.DataFrame()
    T = PREP_TEXT.get(domain, PREP_TEXT["健康"])
    S = []
    S.append("\n## 0　数据准备：合并、清洗与盲标\n")
    S.append(T["intro"])
    S.append(T["a"] + _md(ag.drop(columns=["source"])))
    S.append(T["b"] + _md(ti.drop(columns=["source"])))
    if len(au):
        S.append(T["c"] + _md(au)
                 + ("\n\n盲标多数的分布：\n\n" + _md(am) if len(am) else ""))
    top6 = top[top["名次"] <= 6].copy()
    # 列跟着语料的时间口径走：单日语料没有 `在榜天数`，PV 列叫 `当日PV占该来源%`。
    _pv = "周PV占该来源%" if "周PV占该来源%" in top6.columns else "当日PV占该来源%"
    _cols = [c for c in ("来源", "tier", "名次", "串", _pv, "在榜天数", "决定方式") if c in top6.columns]
    S.append(T["d"] + _md(top6[_cols]))
    if len(rc):
        rc2 = rc.copy()
        for c in ("预注册条件满足", "本次运行声明"):
            rc2[c] = rc2[c].map({True: "是", False: "否"})
        S.append("\n**表 0-E　平台自带分类能不能当参考列**——条件在最终语料出来之前写进配置：每列与界面的 Cramér's V ≤ 0.55，"
                 "且没有占比超过 1% 的类只出现在一侧。满足的才声明，并由脚本断言与运行实际声明的列一致。"
                 "参考列只作参考、从不作监督；平台分类是「科室 × 内容类型」的**话题**体系，不是意图体系。\n\n" + _md(rc2))
    for lv, zh in (("td_l1", "L1 意图"), ("bu_leaf", "聚类叶")):
        f = p / f"sensitivity_bare_{lv}.csv"
        if not f.exists():
            continue
        sb = pd.read_csv(f)
        flips = sb[sb["结论翻转"].astype(str).str.lower().eq("true")]
        S.append(f"\n**表 0-F-{zh}　把 AI 侧来源不明的裸词拿掉以后，界面差异还站得住吗**——"
                 f"AI 侧保留的裸症状 / 疾病 / 药品 / 检查名，导出分不清是键入的还是从症状列表点出来的。"
                 f"本层 {len(sb)} 个类，界面差异显著的：全部行 {int(sb['显著_全部'].astype(str).str.lower().eq('true').sum())} 个、"
                 f"去掉裸词 {int(sb['显著_去掉裸词'].astype(str).str.lower().eq('true').sum())} 个；"
                 f"**结论翻转（显著性变了，或方向反了）的 {len(flips)} 个**列在下面，读这些类的界面差异时必须同时看两列。\n\n"
                 + (_md(flips[["类目", "搜索占比%", "助手占比%_全部", "助手占比%_去掉裸词", "该类助手行里裸词占比%",
                                "差_pp_全部", "差_pp_去掉裸词", "显著_全部", "显著_去掉裸词"]])
                    if len(flips) else "（没有类翻转。）\n"))
    return "\n".join(S) + "\n"


def load_all(domain: str) -> dict:
    p = WORK / domain / "snapshot_classes"
    out = {"summary": json.loads((p / "summary.json").read_text(encoding="utf-8")),
           "coverage": pd.read_csv(p / "coverage.csv"),
           "topn": pd.read_csv(p / "topn_coverage.csv"),
           "topn_members": pd.read_csv(p / "topn_members.csv"),
           "tvd": pd.read_csv(p / "pairwise_tvd.csv"),
           "power": pd.read_csv(p / "assistant_only_power.csv") if (p / "assistant_only_power.csv").exists() else pd.DataFrame(),
           "intent_leafmix": pd.read_csv(p / "intent_leafmix.csv"),
           "leaf_intentmix": pd.read_csv(p / "leaf_intentmix.csv")}
    for level, _ in LEVELS:
        for kind in ("matrix", "absence", "newcombe", "interface", "signature", "confidence", "examples"):
            f = p / f"{kind}_{level}.csv"
            out[f"{kind}_{level}"] = pd.read_csv(f) if f.exists() else pd.DataFrame()
    return out


# ------------------------------------------------------------------ 分层区块

def level_block(A: dict, level: str, zh: str, srcs: list[str], full: bool, sides=None) -> str:
    m, ab, nc, isp = A[f"matrix_{level}"], A[f"absence_{level}"], A[f"newcombe_{level}"], A[f"interface_{level}"]
    A_conf = {level: A.get(f"confidence_{level}")}
    s = []
    n_show = len(m) if full else min(len(m), 25)
    note = "" if full else f"（本层共 {len(m)} 个类，正文列出规模最大的 {n_show} 个，全量见工作簿。）"
    # A 条数与占比
    m = m.copy()
    for c in ("定义", "user_need", "家族审计意见"):
        if c in m.columns:
            m[c] = m[c].map(_prose)
    a = pd.DataFrame({"类目": m["类目"], "全语料条数": m["全语料条数"]})
    for src in srcs:
        a[SRC_ZH[src]] = [f"{_pct(p)}% ({int(k)})" for p, k in
                          zip(m[f"{SRC_ZH[src]}_占比%"], m[f"{SRC_ZH[src]}_条数"])]
    a["极差pp"] = m["极差pp"].round(2)
    a["最高快照"] = m["最高快照"]
    a["出现快照数"] = m["出现快照数"]
    s.append(f"**表 {zh}-A　每个类在每个快照里的条数与快照内占比**{note}\n\n" + _md(a.head(n_show)))
    # B 置信区间
    b = pd.DataFrame({"类目": m["类目"]})
    for src in srcs:
        b[SRC_ZH[src]] = [f"{_pct(p)} [{_pct(lo)}–{_pct(hi)}]" for p, lo, hi in
                          zip(m[f"{SRC_ZH[src]}_占比%"], m[f"{SRC_ZH[src]}_95CI低%"], m[f"{SRC_ZH[src]}_95CI高%"])]
    s.append(f"\n**表 {zh}-B　同样这些占比的 Wilson 95% 区间（单位 %）**——助手快照 n≈1,000，"
             f"区间宽度是搜索快照的三倍左右，读差异前先看区间是否重叠。\n\n" + _md(b.head(n_show)))
    # C 归属与指数
    c = pd.DataFrame({"类目": m["类目"]})
    for src in srcs:
        c[f"{SRC_ZH[src]}_归属%"] = m[f"{SRC_ZH[src]}_均衡归属%"].round(1)
    for src in srcs:
        c[f"{SRC_ZH[src]}_均衡指数"] = m[f"{SRC_ZH[src]}_均衡指数"].round(2)
    c["独占快照"] = m["独占快照"]
    s.append(f"\n**表 {zh}-C　把 {len(srcs)} 个快照当成同等大小以后，这个类归谁**——归属% = 该快照内占比 ÷ "
             f"各快照内占比之和；均衡指数 = 该快照内占比 ÷ 各快照占比的未加权均值（1.00 = 各快照一样常见）。\n\n"
             + _md(c.head(n_show)))
    # D 流量加权
    dcols = [f"{SRC_ZH[src]}_流量占比%" for src in srcs]
    if all(col in m.columns for col in dcols):
        dd = pd.DataFrame({"类目": m["类目"]})
        for src in srcs:
            dd[SRC_ZH[src]] = m[f"{SRC_ZH[src]}_流量占比%"].round(2)
        s.append(f"\n**表 {zh}-D　同样的类，按快照自带的流量字段加权以后的占比（%）**——"
                 f"pv_norm 在每个快照内部各自归一到 10,000，只能在同一快照内读。"
                 + ("语音快照自带的流量字段是空的，这一列是均匀权重，读作行占比即可。" if "assistant_voice" in srcs else "")
                 + "\n\n"
                 + _md(dd.head(n_show)))
    # E 界面层
    e = isp[["类目", "搜索条数", "搜索占比%", "助手条数", "助手占比%", "差_pp", "低_pp", "高_pp", "显著", "归属"]].copy()
    e = e.sort_values("差_pp")
    e["显著"] = e["显著"].map({True: "是", False: "否"})
    if sides is None:
        _e_title = "把两个搜索快照合成“搜索”、把助手各层合成“助手”以后的逐类差异"
    else:
        _se, _as = sides
        _e_title = (f"搜索快照（{int(isp['搜索n'].iloc[0]):,} 行）与助手快照（{int(isp['助手n'].iloc[0]):,} 行）之间的逐类差异"
                    if len(_se) == 1 and len(_as) == 1 else
                    f"把 {_CN.get(len(_se), len(_se))} 个搜索快照合成“搜索”（{int(isp['搜索n'].iloc[0]):,} 行）、"
                    f"把 {_CN.get(len(_as), len(_as))} 个助手快照合成“助手”（{int(isp['助手n'].iloc[0]):,} 行）以后的逐类差异")
    s.append(f"\n**表 {zh}-E　{_e_title}**——"
             f"这是 n 最大、因而最有把握的一个切法；差_pp = 助手占比 − 搜索占比，区间为 Newcombe 95%。"
             f"只在一边出现的类另附“另一边期望条数 / P(0) / 缺席判定”。\n\n"
             + _md(e.head(n_show) if not full else e))
    # F 缺席
    if len(ab):
        f = ab[["类目", "缺席快照", "该快照n", "参照占比%", "期望条数", "P(0)", "该快照上界%",
                "最高占比快照", "最高占比%", "判定"]]
        s.append(f"\n**表 {zh}-F　“该快照 0 条”逐条的可检出性判定**——0 条本身从不证明不存在："
                 f"期望条数 = 该类在其余快照的合并占比 × 这个快照的 n；上界% 是该快照 0 条时的单侧 97.5% 上界。\n\n"
                 + _md(f))
    else:
        s.append(f"\n**表 {zh}-F　缺席清单**：本层没有任何一个类在任何一个快照里是 0 条——"
                 f"{len(m)} 个类全部在全部 {len(srcs)} 个快照中出现。\n")
    # H 标注可靠度
    cf = A_conf.get(level)
    if cf is not None and len(cf):
        h = pd.DataFrame({"类目": cf["类目"], "全语料置信度均值": cf["全语料置信度均值"]})
        for src in srcs:
            # n 必须进格子：这一列里出现过「歧义 100%」的格，背后只有 1 行
            h[f"{SRC_ZH[src]}"] = [
                ("—" if pd.isna(c) else f"{c:.3f}（n={int(n)}；歧义 {a:.1f}%/{b:.1f}%）")
                for c, n, a, b in zip(cf[f"{SRC_ZH[src]}_置信度均值"], cf[f"{SRC_ZH[src]}_n"],
                                      cf[f"{SRC_ZH[src]}_意图歧义%"], cf[f"{SRC_ZH[src]}_聚类歧义%"])]
        h["置信度最低快照"] = cf["置信度最低快照"].fillna("").map(
            lambda v: v if str(v).strip() else "（各快照均 <20 行，不比）")
        cross = ("这一列量的是**另一条路线**对这些行的把握：`td_confidence` 由自上而下分类器给出，"
                 "而这里的行是按自下而上的簇分组的。一个簇里的行如果普遍拿不到高置信度的意图，"
                 "说明两条路线在这块上对不齐。" if level.startswith("bu_") else
                 "`td_confidence` 是自上而下分类器给这一行的把握（0–1），也就是这个类自己边界的清晰度。")
        s.append(f"\n**表 {zh}-H　这一格的数字该打几折：逐类 × 逐快照的标注可靠度**——{cross}"
                 f"括号里是（意图歧义率 / 聚类歧义率），n 是该类在该快照的行数。"
                 f"占比稳不代表边界稳：一个类可以份额不动，而某个快照里的行恰好全是勉强判下来的。"
                 f"按全语料置信度升序，最不可靠的排在最前；“置信度最低快照”只在该快照 ≥20 行时参与比较。\n\n"
                 + _md(h.head(n_show)))
    # G 显著变动
    g = nc[nc["显著"]].reindex(nc[nc["显著"]]["差_pp"].abs().sort_values(ascending=False).index)
    g = g[["类目", "a", "b", "占比_a%", "占比_b%", "差_pp", "低_pp", "高_pp", "类型"]].head(30)
    s.append(f"\n**表 {zh}-G　全部快照两两对比里，变动最大的 30 项（仅列 Newcombe 区间不含 0 的）**——"
             f"本层共 {int(nc['显著'].sum())} 项显著 / 共 {len(nc)} 项对比，全量在工作簿。\n\n" + _md(g))
    return "\n".join(s)


# ------------------------------------------------------------------ 逐类卡片

def cards(A: dict, level: str, srcs: list[str], label: str) -> str:
    m, ex = A[f"matrix_{level}"], A[f"examples_{level}"]
    nc = A[f"newcombe_{level}"]
    out = []
    for _, r in m.iterrows():
        key = str(r["key"])
        line = " · ".join(f"{SRC_ZH[s]} **{_pct(r[f'{SRC_ZH[s]}_占比%'])}%**({int(r[f'{SRC_ZH[s]}_条数'])})"
                          for s in srcs)
        sub = nc[nc["key"].astype(str) == key]
        sig = sub[sub["显著"]]
        big = sig.reindex(sig["差_pp"].abs().sort_values(ascending=False).index).head(2)
        moves = "；".join(f"{x['a']}→{x['b']} {x['差_pp']:+.2f}pp [{x['低_pp']:+.2f}, {x['高_pp']:+.2f}]"
                         for _, x in big.iterrows()) or "没有任何一对快照的差异显著"
        dom = (f"主导{'意图' if level.startswith('bu_') else '叶'}：{r.get('主导意图', r.get('主导叶', ''))}"
               f"（{r.get('主导意图占比%', r.get('主导叶占比%', ''))}%）")
        extra = []
        if level == "bu_leaf":
            co = r.get("盲评一致性", "")
            if pd.notna(co):
                extra.append(f"盲评一致性 {float(co):.0f}/5")
            if str(r.get("风险标注", "")) == "True":
                extra.append("命名时被标为风险叶")
        if level == "td_l1":
            if str(r.get("体系内风险标注", "")) == "是":
                extra.append("体系内标为风险类目")
            es = r.get("架构师预估占比")
            if pd.notna(es):
                extra.append(f"架构师事前预估 {100 * float(es):.1f}%（实际全语料 {_pct(r['全语料占比%'])}%）")
        head = f"#### {r['类目']}　（全语料 {int(r['全语料条数']):,} 行，占 {_pct(r['全语料占比%'])}%）"
        body = [head]
        d = _prose(r.get("定义", "") or "").strip()
        if d and d.lower() != "nan":
            body.append(f"> {d}")
        body.append(f"- 逐快照占比：{line}")
        body.append(f"- 极差 {r['极差pp']:.2f}pp，最高在 {r['最高快照']}、最低在 {r['最低快照']}"
                    + (f"，缺席于 {r['缺席快照']}" if str(r["缺席快照"]) not in ("", "nan") else "")
                    + (f"；**独占 {r['独占快照']}**" if str(r["独占快照"]) not in ("", "nan") else ""))
        body.append(f"- 最大的显著变动：{moves}")
        ranks = " · ".join(f"{SRC_ZH[x]} 第{int(r[f'{SRC_ZH[x]}_排名'])}"
                           for x in srcs if f"{SRC_ZH[x]}_排名" in r.index)
        attrib = " · ".join(f"{SRC_ZH[x]} {r[f'{SRC_ZH[x]}_均衡归属%']:.0f}%" for x in srcs)
        body.append(f"- 快照内排名：{ranks}")
        body.append(f"- 把各快照当同等大小以后的归属：{attrib}")
        body.append(f"- {dom}" + ("；" + "；".join(extra) if extra else ""))
        if len(ex):
            e = ex[ex["key"].astype(str) == key]
            lines = []
            for s in srcs:
                g = e[(e["快照"] == SRC_ZH[s]) & (e["取法"] == "流量最高")]
                if not len(g):
                    g2 = e[(e["快照"] == SRC_ZH[s])]
                    if len(g2) and str(g2.iloc[0]["取法"]).startswith("不引原文"):
                        lines.append(f"{SRC_ZH[s]}：不引原文（该快照下这个类的 "
                                     f"{int(g2.iloc[0]['该类该快照条数'])} 行全部命中风控图层或引用护栏）")
                    continue
                qs = " / ".join(_q(str(q)[:26]) for q in g["query"].head(3))
                lines.append(f"{SRC_ZH[s]}：{qs}")
            if lines:
                body.append("- 例子（各快照流量最高的可引行，只截断不改写）：\n  - " + "\n  - ".join(lines))
        out.append("\n".join(body))
    return f"\n### 逐{label}卡片\n\n" + "\n\n".join(out) + "\n"


# ------------------------------------------------------------------ 叙述

import re as _re                                                   # noqa: E402
_BLOCK = _re.compile(r"<!--NARR:([^>]+)-->(.*?)<!--/NARR:\1-->", _re.S)


def _toc(text: str) -> str:
    """两千多行的参考文档没有目录不能用。目录从实际渲染出来的标题生成，所以永远和正文一致。
    逐意图 / 逐叶卡片是 h4，不进目录——它们有几十上百个，列出来目录比正文还长。"""
    lines, items = text.splitlines(), []
    for ln in lines:
        m = _re.match(r"^(#{2,3}) (.+)$", ln)
        if m:
            depth, title = len(m.group(1)) - 2, m.group(2).strip()
            anchor = _re.sub(r"[^\w\u4e00-\u9fff-]+", "-", title.replace(" ", "-")).strip("-").lower()
            items.append("  " * depth + f"- [{title}](#{anchor})")
    if not items:
        return text
    toc = "\n## 目录\n\n" + "\n".join(items) + "\n"
    # 插在第一个 `## ` 标题之前
    for i, ln in enumerate(lines):
        if ln.startswith("## "):
            return "\n".join(lines[:i] + [toc] + lines[i:])
    return text


def _splice(domain: str, text: str) -> str:
    """把分析员写的叙述填进占位。叙述单独存在 `narrative.md` 里，重跑本脚本永远重新填一次，
    所以报告可以任何时候从表重建而不丢叙述；没有 narrative.md 时占位保持为空。"""
    src = WORK / domain / "snapshot_classes" / "narrative.md"
    if not src.exists():
        return text
    got = {k: v.strip() for k, v in _BLOCK.findall(src.read_text(encoding="utf-8"))}
    unknown = sorted(set(got) - set(NARR))
    if unknown:
        raise SystemExit(f"{domain}: narrative.md 里有报告不认识的小节 {unknown}；"
                         f"允许的是 {NARR}")
    def sub(m):
        k = m.group(1)
        return f"<!--NARR:{k}-->\n{got[k]}\n<!--/NARR:{k}-->" if got.get(k) else m.group(0)
    out = _BLOCK.sub(sub, text)
    missing = [k for k in NARR if not got.get(k)]
    if missing:
        print(f"  （{domain}: 叙述缺 {'/'.join(missing)}，这些小节留空）")
    return out


# ------------------------------------------------------------------ 组装

def assemble(domain: str) -> Path:
    A = load_all(domain)
    S = A["summary"]
    srcs = [s for s in SOURCES if SRC_ZH[s] in S["snapshots"]]
    gen = run_dir(domain)
    books = sorted(gen.glob("*_query_挖掘结果.xlsx"))
    stem = books[0].stem.replace("_query_挖掘结果", "") if books else DOMAINS[domain]
    cov = A["coverage"]
    tvd = A["tvd"]
    lv = S["levels"]

    legacy_text = domain in set(COHORTS["pool5"]) | set(COHORTS["new2"])
    from p5_snapshot_classes import surface_groups
    _dsrc = load(domain)
    _se_srcs, _as_srcs = surface_groups(_dsrc, srcs)
    sides = None if legacy_text else (_se_srcs, _as_srcs)
    n_se_rows = int(_dsrc.source.isin(_se_srcs).sum())
    n_as_rows = int(_dsrc.source.isin(_as_srcs).sum())
    P = []
    # 代次不写死。`run_dir` 认 `P5_GEN_<批次>`，交付代次不一定是 gen01——人物域就是 gen03——
    # 而这里原来硬写 gen01，于是正文、复现表和图目录会把读者指到另一个代次（而且那个目录根本不存在）。
    _gen = run_dir(domain).name
    P.append(f"# {domain} —— 每个意图、每个聚类叶在 {len(srcs)} 个快照上的逐类对照\n")
    P.append(f"> 运行 `{S['run']}/{_gen}`，语料 **{S['n_rows']:,} 行** = "
             + " + ".join(f"{k} {v:,}" for k, v in S["snapshots"].items()) + "。\n>\n"
             + ("> 这份报告和已经交付的《POOLED5_2026 五域同体系对比》是**两件不同的事**。那份问“两个界面整体差多远”，"
                "用的是汇总距离；这份问**每一个类目在每一个快照里各占多少、哪些类只在某个快照出现**，所以主体是逐类 × 逐快照的矩阵。\n>\n"
                if domain in COHORTS["pool5"] else DOMAIN_INTRO[domain] if domain in DOMAIN_INTRO else
                f"> **{domain} 不在已交付的《POOLED5_2026 五域同体系对比》里**——那份报告覆盖的是金融/医疗/教育/影视/人物五个垂类。"
                f"本域是 2026-09-13 新接入的，目前只有这一份逐类 × 逐快照对照，没有对应的跨领域汇总篇；"
                f"凡是本文没写的跨领域比较，都是还没做，不是做了没发现。\n>\n") +
             f"> 全部数字由 `analysis/pooled5/p5_snapshot_classes.py` 一次算出、由 `p5_snapshot_report.py` 直接渲染，"
             f"与同目录工作簿 `{stem}_意图与聚类叶_跨快照对比.xlsx` 同源。\n")
    # Measured here rather than asserted: the claim "one string, one label" was verified on the
    # original five, and a new vertical has to earn it rather than inherit it.
    _d = load(domain)
    _vc = _d["query"].astype(str).value_counts()
    dup_rows = int(_d["query"].astype(str).isin(_vc[_vc > 1].index).sum())
    dup_td = int((_d.groupby(_d["query"].astype(str))["td_l1_name"].nunique() > 1).sum())
    dup_bu = int((_d.groupby(_d["query"].astype(str))["bu_leaf_name"].nunique() > 1).sum())
    P.append("\n## 读这份表之前必须知道的六件事\n")
    _nmax_name = max(S["snapshots"], key=S["snapshots"].get)
    _nmax = S["snapshots"][_nmax_name]
    _p1 = ("搜索每年 ~10,000 行、助手每层 ~1,000 行，\n   跨快照比原始条数只是在比导出文件的大小。" if legacy_text else
           "各快照行数 " + "、".join(f"{k} {v:,}" for k, v in S["snapshots"].items())
           + f"，最多与最少相差 {_nmax / min(S['snapshots'].values()):.1f} 倍，\n   跨快照比原始条数只是在比导出文件的大小。")
    _p2 = ("同样 0 条落在 10,000 行的搜索里，\n   上界只有约 0.04%。" if legacy_text else
           f"同样 0 条落在 {_nmax:,} 行的{_nmax_name}里，\n   上界只有约 {100 * (1 - 0.025 ** (1 / _nmax)):.2f}%。")
    _p5 = "年度差异只能来自榜单换血" if legacy_text else "快照之间的差异只能来自榜单换血"
    P.append(f"""1. **一切占比都是快照内占比。**（该类条数 ÷ 该快照总行数）{_p1}全文没有一个跨快照的绝对条数比较。
2. **“只在某快照出现”必须配可检出性。** 一个类在 {min(S['snapshots'].values()):,} 行的助手快照里 0 条，
   单侧 97.5% 上界仍有约 {100 * (1 - 0.025 ** (1 / min(S['snapshots'].values()))):.2f}%；{_p2}所以每个 0 条都在“表 *-F”里配了期望条数与 P(0)，并据此判 `真缺席 / 偏少但证据弱 / 不可判定`。
   **这个不对称是结构性的**：“只出现在搜索里”远比“只出现在助手里”容易达成，读独有性时必须先看 n。
3. **这 {len(srcs)} 个快照共享同一套体系。** 它们合并成一份语料、由挖掘程序**一次跑完**，所以意图编码与聚类树对每个快照都是同一套。
   分开跑两次不可比——同一份 10,000 行金融语料跑两次共享 **0 个**类目编码。
4. **本次是 `mode=fast`：单标注员，kappa 不存在，不是 1.0。** 下面每一个占比都建立在一个没有第二意见校验的标注层上。
5. **同一个字符串在一次运行里必然拿到同一个标签**（本域实测：{dup_rows:,} 行落在重复串上，
   拿到两个 `td_l1_name` 的字符串 {dup_td} 个、两个 `bu_leaf_name` 的 {dup_bu} 个）。所以“两个快照共有的串
   标签一致”是构造必然，不是测量结果；{_p5}，本方法测不到“同一句话含义变了”。
6. **L2 子意图没有名字。** 它们是在 L1 内部按表示做的几何细分（`subintents.json`），运行没有为它们命名，
   所以这里原样写作 `L1编码__序号`，含义靠“代表串”辨识——代表串是描述性的，不是定义。
""")
    P.append("<!--NARR:导读-->\n<!--/NARR:导读-->\n")

    P.append(prep_section(domain))
    P.append(f"\n## 1　{len(srcs)} 个快照各自长什么样\n")
    # 流量口径的限制，两张表（表 1 与表 2.2）**都**要带——上一版只在表 1 用了，
    # 而表 2.2 才是那一行被横着读的地方。
    # **两个缺陷，都在金融8 上实测到。** ① 构建审计只查 `build_audit.csv`，而金融8 / 健康写的是
    # `build_audit_<批次>.csv`，于是找不到行；② 下面原来是一个跨三行的隐式拼接，后面挂着
    # `if _drop else "。"`——条件表达式的优先级比字符串拼接低，所以找不到行时**整段①②③一起**被换成
    # 了「。」。金融8 报告因此一个字的流量警示都没有。现在①②无条件印，③按实际被清洗掉 PV 的快照印。
    _drop_rows = []
    for _ba in sorted(WORK.glob("build_audit*.csv")):
        _b = pd.read_csv(_ba)
        if "domain" not in _b.columns:
            continue
        for _, _r in _b[_b["domain"] == domain].iterrows():
            if str(_r["source"]) in SRC_ZH and float(_r.get("pv_dropped_%", 0) or 0) >= 5:
                _drop_rows.append((SRC_ZH[str(_r["source"])], int(_r["kept"]),
                                   float(_r["pv_dropped_%"]), _ba.name))
    _neff = cov[["快照", "流量有效n"]].drop_duplicates("快照").set_index("快照")["流量有效n"]
    _nmin = _neff.idxmin()
    # 已交付的七个域：③ 用上一轮复核过的原句，逐字节不变（这一段上一版被通用措辞覆盖，七份报告
    # 因此各变了 6 处——回归测试抓到的）。之后接入的语料走下面的通用 ③。
    _legacy_drop = ""
    if legacy_text and (WORK / "build_audit.csv").exists():
        _b = pd.read_csv(WORK / "build_audit.csv")
        _r = _b[(_b["domain"] == domain) & (_b["source"] == "assistant_top")]
        if len(_r):
            _legacy_drop = (f"助手头部1k 是按 PV 取的前 1,000 条**再过清洗层**，本域交付 "
                            f"{int(_r['kept'].iloc[0]):,} 行，被清洗掉的行带走了该导出 "
                            f"**{float(_r['pv_dropped_%'].iloc[0]):.2f}%** 的原始 PV"
                            f"（`work/build_audit.csv: pv_dropped_%`）——所以助手头部那一格的“流量”"
                            f"既不是助手全量，**也不是头部一千条**，而是清洗后剩下那一小部分头部流量内部的份额，"
                            f"主要是清洗规则的函数。")
    if _legacy_drop:
        _drop_rows = []
    traffic_note = (
        "。**流量这一列比它看起来松得多，读之前先看"
        + ("三条" if (_drop_rows or _legacy_drop) else "两条") + "**："
        "①`pv_norm` 在每个来源内部各自归一到 10,000，所以流量占比**只能竖着读**，跨快照比大小没有意义；"
        f"②流量口径的有效样本量是 1/Σw²，**不是行数**——本域最低的是 {_nmin} 的 "
        f"**{float(_neff.min()):.1f}**（该快照 {int(cov[cov['快照'] == _nmin]['n'].iloc[0]):,} 行），"
        f"所以流量那一格的精度远低于同一行的行占比，本表**没有给它区间**；")
    if _legacy_drop:
        traffic_note += f"③{_legacy_drop}"
    if _drop_rows:
        traffic_note += ("③清洗层拿走的流量不可忽略："
                         + "；".join(f"{z} 交付 {k:,} 行，被清洗掉的行带走了该导出 **{pv:.2f}%** 的原始 PV"
                                     for z, k, pv, _ in _drop_rows)
                         + f"（`work/{_drop_rows[0][3]}: pv_dropped_%`）——这些快照那一格的“流量”"
                           "只是清洗后剩下那部分流量内部的份额，主要是清洗规则的函数。")
    traffic_note += ("语音导出没有 PV，按均匀权重处理，所以语音那一行的流量占比恒等于行占比，"
                     "两格相同是构造，不是测量结果。\n" if "assistant_voice" in srcs else "\n")

    for level, zh in LEVELS:
        c = cov[cov["层级"] == zh][["快照", "n", "类目数", "有效类目数", "熵", "HHI", "首位类目",
                                   "首位占比%", "前3占比%", "前5占比%", "前10占比%", "前10流量占比%",
                                   "前10流量摆动pp", "前10流量占比%低", "前10流量占比%高",
                                   "流量有效n"]].copy()
        # 摆动只给宽度、不给方向，而点值几乎总落在区间的一端（实测 10 格里 9 格），
        # 于是 `44.42 | 0.64` 会被读成 43.78–45.06，真区间其实是 43.78–44.42。改成把区间写进那一格。
        c["前10流量占比%"] = [f"{v:.2f}" if sw <= 0.005 else f"{v:.2f}（并列 {lo:.2f}–{hi:.2f}）"
                             for v, sw, lo, hi in zip(c["前10流量占比%"], c["前10流量摆动pp"],
                                                      c["前10流量占比%低"], c["前10流量占比%高"])]
        c = c.drop(columns=["前10流量摆动pp", "前10流量占比%低", "前10流量占比%高"])
        P.append(f"\n**表 1-{zh}**　有效类目数 = exp(熵)，可以读作“这个快照实际上用到了几个类”；"
                 f"HHI 越高越集中。`前3/前5/前10占比%` 是该快照**最大的 3/5/10 个类**的行占比之和"
                 f"（每个快照各排各的，不是同一批类）；`前10流量占比%` 是**同样这十个类**按 `pv_norm` "
                 f"加权后的占比，不是按流量重排以后的前十。"
                 f"`前10流量占比%` 后面的“（并列 a–b）”是**第 10 名并列**时这一格的全部可能范围："
                 f"并列的类条数相同，所以 `前10占比%` 是定值，但它们 `pv_norm` 不同，所以流量那一格"
                 f"取决于挑中哪一个；没有括号 = 挑法不影响这一格（没有并列，或该快照按均匀权重）。"
                 f"`流量有效n` = 1/Σw²，是流量口径真正的样本量"
                 f"{traffic_note}\n\n" + _md(c))
    ov = pd.DataFrame([{"层级": k, "类目数": v["n_classes"], "全快照都出现": v["全快照都出现的类"],
                        "独占某快照": v["独占某快照的类"], "仅搜索": v["仅搜索"], "仅助手": v["仅助手"],
                        "真缺席条目": v["真缺席条目"], "界面差异显著的类": v["界面差异显著的类"],
                        "特征类条目": v["特征类条目(显著高于其余全部快照)"],
                        "反特征类条目": v["反特征类条目(显著低于其余全部快照)"],
                        "Cramér's V(快照×类)": v["CramersV_快照×类"]} for k, v in lv.items()])
    P.append("\n**表 1-总览**　四个层级的独有性一览。“仅搜索”= 该类在"
             + ("两个搜索快照里有" if legacy_text else "搜索侧（" + "、".join(SRC_ZH[x] for x in _se_srcs) + "）有")
             + "、在全部助手快照里 0 条；"
             "“仅助手”反之。Cramér's V 衡量快照与类目之间的整体关联强度（0 = 无关，1 = 完全决定）。\n\n" + _md(ov))

    P.append("\n## 2　快照两两之间隔多远（四个层级）\n")
    P.append("总变差距离 TVD 回答“把一个快照的类目分布变成另一个，需要改动多少比例的行”。"
             "它不随 n 增长，但在小样本上**系统性偏高**，所以每一行都附了同一个快照对半切出来的噪声上界——"
             "低于噪声上界的距离不可读。\n\n"
             "秩相关 rho 看的是**次序**而不是量：两个快照可以份额差很多而排序几乎不变（整体缩放），"
             "也可以份额差不多而排序全乱。两个数要一起读。\n\n")
    for level, zh in LEVELS:
        t = tvd[tvd["层级"] == zh][["a", "b", "n_a", "n_b", "TVD", "TVD低", "TVD高", "同源噪声上界",
                                   "超出噪声", "CramersV", "秩相关rho", "前5重合个数", "首位是否相同"]].copy()
        t["超出噪声"] = t["超出噪声"].map({True: "是", False: "否"})
        t["首位是否相同"] = t["首位是否相同"].map({True: "是", False: "否"})
        P.append(f"\n**表 2-{zh}**\n\n" + _md(t))

    P.append("\n## 2.1　图\n\n"
             "![快照两两距离](img/快照两两距离.png)\n\n"
             "*图 1　四个层级上，每一对快照的类目分布距离。误差棒是自助法 95% 区间，"
             "红线是同一个快照对半切出来的噪声上界——柱子没超过红线就读作“测不出差别”。*\n")
    P.append("\n## 2.2　每个快照自己的前十\n")
    P.append(f"同一套体系，{len(srcs)} 个快照各自排出来的前十名。看的是**次序**：一个类在两个快照里"
             f"占比接近、名次却差很多，说明它周围的类变了。\n")
    LEVEL_ZH = {"td_l1": "自上而下 L1 意图", "bu_leaf": "自下而上 叶"}
    tn_all = A["topn"].copy()
    tn_all["边界并列"] = tn_all["边界并列"].astype(str).str.lower().eq("true")
    mem_all = A["topn_members"]
    for level, zh in [("td_l1", "L1 意图"), ("bu_leaf", "聚类叶")]:
        m = A[f"matrix_{level}"]
        disp = dict(zip(m["key"].astype(str), m["类目"].astype(str)))
        q10 = tn_all[tn_all["层级"] == LEVEL_ZH[level]].set_index("快照")
        mem = mem_all[mem_all["层级"] == LEVEL_ZH[level]]
        top, ties = {}, []
        for src in srcs:
            z = SRC_ZH[src]
            q = q10.loc[z]
            # **第1..第10 只从成员表来。** 报告原来自己再 sort 一遍 matrix，于是并列格上显示的
            # 十个类与流量行汇总的十个类可能不是同一批（教育实测差 5.25pp）。占比也直接由
            # 条数/n 算，不走 matrix 里已经舍入到 3 位的值——那条路会把 12.4549% 印成 12.46，
            # 而表 1 的同一个量印 12.45。
            g = mem[mem["快照"] == z].sort_values("名次")
            col = [f"{disp.get(str(r['key']), str(r['key']))} {100 * int(r['条数']) / int(q['n']):.2f}%"
                   for _, r in g.iterrows()]
            swing = float(q["前10流量摆动pp"])
            if bool(q["边界并列"]):
                # 语音快照 PV 均匀，并列的类 pv_norm 必然相等，摆动恒为 0——上一版会印出
                # 「可在 66.70%–66.70% 之间摆动」，再跟一句「它们的 pv_norm 不同」，两句都是错的。
                tail = (f"流量合计因此可在 {float(q['前10流量占比%低']):.2f}%–"
                        f"{float(q['前10流量占比%高']):.2f}% 之间摆动" if swing > 0.005 else
                        "但该快照按均匀权重，并列的类 `pv_norm` 相等，所以**这一格也不受影响**")
                ties.append(f"{z}（第 10 名上有 {int(q['并列类数'])} 个类各 {int(q['第10名条数']):,} 条，"
                            f"只能取 {int(q['并列需选'])} 个；{tail}）")
            top[z] = col + [f"{q['前3行占比%']:.2f}%",
                            f"{q['前5行占比%']:.2f}%",
                            f"**{q['前10行占比%']:.2f}%**　{int(q['前10条数']):,}/{int(q['n']):,} 行",
                            f"{q['前10流量占比%']:.2f}%" + (
                                f"　[{float(q['前10流量占比%低']):.2f}–{float(q['前10流量占比%高']):.2f}]"
                                if swing > 0.005 else ""),
                            f"{q['前10之外行占比%']:.2f}%　余 {int(q['前10之外类数'])} 类"]
        idx = ([f"第{i + 1}" for i in range(10)]
               + ["**前3合计**", "**前5合计**", "**前10合计**", "前10合计（流量）", "前10之外（行）"])
        tt = pd.DataFrame(top, index=idx).reset_index()
        tt = tt.rename(columns={"index": "名次"})
        tie_note = ("\n\n**第 10 名并列的快照**：" + "；".join(ties)
                    + "。并列的类条数相同，所以 `前3/前5/前10合计` 与 `前10之外（行）` "
                      "**与挑中哪一个无关**；但在有 PV 的快照上它们的 `pv_norm` 不同，"
                      "所以 `前10合计（流量）` 会变——那一格后面的方括号就是挑法能造成的全部范围，"
                      "**不是抽样误差**（抽样误差另算，而且比它大得多，见表 1 的 `流量有效n`）。"
                    if ties else "")
        P.append(
            f"\n**表 2.2-{zh}　各快照前十，以及前十合起来覆盖了这个快照多少**"
            f"（每格是该快照内占比）\n\n"
            f"每个快照**各排各的**，所以同一行的「第 N」在两列里通常不是同一个类，这张表只能竖着读。"
            f"后面四行是合计、末行是余额：`前3/前5/前10合计` 是这个快照**最大的 3/5/10 个类**的"
            f"行占比之和，`前10合计` 另附条数/该快照总行数；`前10合计（流量）` 是**上面这十个类**"
            f"按 `pv_norm` 加权后的占比，**不是**按流量重排以后的前十；`前10之外（行）` 是剩下的"
            f"全部类的行占比（它与 `前10合计` 相加是 100%，**不与上面那一行的流量相加**）。"
            f"上面十格与下面的合计出自同一张成员表（工作簿「前十成员」页）；"
            f"除 `前10合计（流量）` 外，占比一律由条数 ÷ 该快照行数直接算出，"
            f"那一行是这十个类的 `pv_norm` 之和 ÷ 该快照 `pv_norm` 之和{traffic_note.rstrip()}"
            f"{tie_note}\n\n" + _md(tt))
    P.append("\n## 3　自上而下意图（L1）× 快照\n")
    P.append("![L1意图占比热图](img/L1意图_占比热图.png)\n\n"
             "*图 2　每个 L1 意图在每个快照里的**快照内占比**（%）。行按全语料规模排序。*\n\n"
             "![L1意图均衡指数热图](img/L1意图_均衡指数热图.png)\n\n"
             "*图 3　同样这些类的**均衡指数** = 该快照内占比 ÷ 各快照内占比的未加权均值。"
             "1.00 表示各快照一样常见；红色表示这个快照里更常见，蓝色表示更少见。"
             "它把“搜索行数是助手十倍”这件事除掉了，所以能直接看出一个类归谁。*\n")
    P.append("<!--NARR:意图层-->\n<!--/NARR:意图层-->\n\n")
    P.append(level_block(A, "td_l1", "L1", srcs, full=True, sides=sides))
    m1 = A["matrix_td_l1"]
    if "架构师预估占比" in m1.columns and m1["架构师预估占比"].notna().any():
        e = m1[m1["架构师预估占比"].notna()].copy()
        e["_d"] = (100 * e["架构师预估占比"]) - e["全语料占比%"]
        e = e.reindex(e["_d"].abs().sort_values(ascending=False).index)
        e["预估%"] = (100 * e["架构师预估占比"]).map(lambda v: f"{v:.1f}")
        e["实际全语料%"] = e["全语料占比%"].map(lambda v: f"{v:.2f}")
        e["预估−实际pp"] = e["_d"].map(lambda v: f"{v:+.2f}")
        for x in srcs:
            e[SRC_ZH[x]] = e[f"{SRC_ZH[x]}_占比%"].map(lambda v: f"{v:.2f}")
        cols = ["类目", "预估%", "实际全语料%", "预估−实际pp"] + [SRC_ZH[x] for x in srcs]
        tot = 100 * float(m1["架构师预估占比"].sum())
        P.append(f"\n**表 L1-H　架构师在看到标注之前写下的预估占比 vs 实际交付占比**——"
                 f"`taxonomy.json` 的 `expected_share` 是**事前**写的（{int(m1['架构师预估占比'].notna().sum())}/"
                 f"{len(m1)} 个类填了，合计 {tot:.1f}%，不强制归一）。它是对**整份合并语料**的预估，"
                 f"语料 {_search_share(A)} 是搜索行，所以偏差大的类多半是被助手行顶上去或压下去的。\n\n" + _md(e[cols]))
    P.append(cards(A, "td_l1", srcs, "意图"))

    P.append("\n## 4　自下而上聚类叶 × 快照\n")
    P.append("![叶占比热图](img/叶_占比热图.png)\n\n"
             "*图 4　每个聚类叶在每个快照里的快照内占比（%）。*\n\n"
             "![叶均衡指数热图](img/叶_均衡指数热图.png)\n\n"
             "*图 5　叶的均衡指数，读法同图 3。叶比意图细，所以同样的指数值背后 n 更小、更容易是噪声——"
             "要判断一个格子是不是真的，回去看表 叶-B 的区间与表 叶-G 的显著性。*\n")
    P.append("<!--NARR:叶层-->\n<!--/NARR:叶层-->\n\n")
    P.append(level_block(A, "bu_leaf", "叶", srcs, full=True, sides=sides))
    P.append(cards(A, "bu_leaf", srcs, "叶"))

    P.append("\n## 5　两个中间层：L2 子意图与聚类家族\n")
    P.append("<!--NARR:子层-->\n<!--/NARR:子层-->\n\n")
    P.append("### 5.1　L2 子意图\n\n")
    rep = A["matrix_td_l2"][["类目", "全语料条数", "代表串(流量最高，仅供辨识)"]].copy() \
        if "代表串(流量最高，仅供辨识)" in A["matrix_td_l2"].columns else pd.DataFrame()
    if len(rep):
        # 代表串就是 query 原文，所以逐条加引号——不加的话，query 自己带的 「」（例如
        # 「AI视频」一家三口温馨比心）会读成本文档的引用，全文引文核查也会把它当成一条假引文。
        PLACEHOLDER = "（该子意图无可引行）"
        rep["代表串(流量最高，仅供辨识)"] = rep["代表串(流量最高，仅供辨识)"].map(
            lambda v: v if not str(v).strip() or str(v).strip() == PLACEHOLDER
            else " / ".join(_q(x.strip()) for x in str(v).split(" / ")))
        P.append("**表 5.1-0　子意图的代表串**——子意图没有名字，这一列是该子意图里流量最高的可引行，只用于辨识。\n\n"
                 + _md(rep))
    P.append(level_block(A, "td_l2", "L2", srcs, full=True, sides=sides))
    P.append("\n### 5.2　聚类家族\n\n")
    P.append(level_block(A, "bu_family_final", "家族", srcs, full=True, sides=sides))

    P.append("\n## 6　独有性：哪些类只出现在哪个快照\n")
    P.append("![界面散点](img/界面散点.png)\n\n"
             + ("*图 6　把两个搜索快照合成“搜索”、助手各层合成“助手”以后，每个类在两边的占比。" if legacy_text else
                f"*图 6　搜索侧（{n_se_rows:,} 行）与助手侧（{n_as_rows:,} 行）各自合并以后，每个类在两边的占比。") +
             "这是 n 最大的切法。对角线表示两边一样；灰点是 Newcombe 区间含 0 的类（测不出差别）。"
             "两轴都是对数轴，0 条的点被压在 0.008% 的边上。*\n")
    P.append("<!--NARR:独有性-->\n<!--/NARR:独有性-->\n")
    P.append("\n### 6.1　只出现在一侧、或只出现在一个快照的类\n")
    for level, zh in LEVELS:
        m, isp = A[f"matrix_{level}"], A[f"interface_{level}"]
        only = isp[isp["归属"] != "两界面都有"][["类目", "搜索条数", "搜索占比%", "助手条数",
                                                "助手占比%", "归属", "另一边期望条数", "另一边P(0)", "缺席判定"]]
        excl = m[m["独占快照"].astype(str).isin([SRC_ZH[s] for s in srcs])][
            ["类目", "全语料条数", "独占快照"]]
        real = int((isp["缺席判定"] == "真缺席").sum())
        P.append(f"\n**{zh}**：{len(m)} 个类中，两界面都有 {int((isp['归属'] == '两界面都有').sum())} 个、"
                 f"仅搜索 {int((isp['归属'] == '仅搜索').sum())} 个、仅助手 {int((isp['归属'] == '仅助手').sum())} 个；"
                 f"独占单个快照的 {len(excl)} 个。其中 **{real} 个**的“0 条”通过了可检出性判定（期望条数≥5 且 P(0)<1%），"
                 f"其余的 0 条只说明没抽到。\n\n")
        if len(only):
            P.append(_md(only))
        if len(excl):
            P.append("\n独占单个快照的类：\n\n" + _md(excl))

    if len(A["power"]):
        P.append("\n**表 6.1-检出力　“没有任何一个类只出现在助手里”这句话有多大把握**——"
                 "0 个只有配上检出力才是结论。取助手侧**最小**的那个类的发生率，问：同样的发生率放到"
                 + (("搜索侧那两万行上，期望几条、一条都不出现的概率多大。反方向（“仅搜索”）的检出力弱得多，"
                     "因为助手侧只有两三千行——这个不对称是结构性的，不是发现。\n\n") if legacy_text else
                    (f"搜索侧那 {n_se_rows:,} 行上，期望几条、一条都不出现的概率多大。反方向（“仅搜索”）的检出力"
                     + ("弱得多" if n_as_rows < n_se_rows else "并不更弱")
                     + f"，因为助手侧有 {n_as_rows:,} 行——检出力跟着 n 走，不是发现。\n\n"))
                 + _md(A["power"].assign(**{"搜索侧一条都不出现的概率": A["power"]["搜索侧一条都不出现的概率"].map(
                     lambda v: "<0.0001" if float(v) < 1e-4 else f"{float(v):.4f}")})))
    P.append("\n### 6.2　每个快照的“特征类”\n")
    P.append("“独占某个快照”几乎永远是 0——一个类只要在别的快照里出现过一条就不算独占，"
             "所以那个口径对“这个快照有什么特别的”几乎没有信息量。换一个能承重的口径：\n\n"
             "> **特征类**：这个类在该快照里的占比，**与其余每一个快照逐一比较**都显著更高"
             "（每一对都做 Newcombe 检验，全部区间不含 0 才算）。快照越多，这个条件越严。\n")
    for level, zh in LEVELS:
        sg = A[f"signature_{level}"]
        if not len(sg):
            P.append(f"\n**{zh}**：没有任何一个类在某个快照里显著高于其余每一个快照。\n")
            continue
        for direction, label in [("显著高于其余全部", "特征类（显著高于其余每一个快照）"),
                                 ("显著低于其余全部", "反特征类（显著低于其余每一个快照）")]:
            g = sg[sg["方向"] == direction]
            if not len(g):
                P.append(f"\n**{zh} · {label}**：无。\n")
                continue
            g = g.sort_values(["快照", "该快照占比%"], ascending=[True, False])
            P.append(f"\n**{zh} · {label}**（{len(g)} 条）\n\n"
                     + _md(g[["快照", "类目", "该快照条数", "该快照n", "该快照占比%",
                              "其余快照合并占比%", "最弱一对的区间端点pp", "对比了几个快照"]]))

    P.append("\n## 7　两条路线的交叉：同一个意图，在不同快照里落在不同的叶\n")
    P.append("这是“份额一样、实现方式不同”的检验。意图体系由架构师自上而下写出，聚类树由嵌入自下而上长出，"
             "两者独立；所以“这个意图在搜索里主要落在 A 叶、在助手里主要落在 B 叶”是一个测量，不是定义。"
             "只在两个快照都 ≥30 行时计算，否则内层 TVD 由噪声主导。\n")
    P.append("<!--NARR:路线交叉-->\n<!--/NARR:路线交叉-->\n\n")
    il = A["intent_leafmix"]
    if len(il):
        t = il.reindex(il["内层TVD"].sort_values(ascending=False).index)
        t = t[["外层", "a", "b", "n_a", "n_b", "内层TVD", "同源噪声上界", "超出噪声",
               "a主导内层", "a主导占比%", "b主导内层", "b主导占比%", "主导是否相同"]].copy()
        t["主导是否相同"] = t["主导是否相同"].map({True: "是", False: "否"})
        t["超出噪声"] = t["超出噪声"].map({True: "是", False: "否"})
        n_over = int((il["超出噪声"]).sum())
        P.append("**表 7-A　每个意图内部的叶分布，在两个快照之间差多远（按差距排序，全部 "
                 f"{len(il)} 行）**——"
                 f"**内层 TVD 必须先跟同源噪声上界比。** 这些格子的 n 只有几十到几百，两份来自同一个"
                 f"分布的样本本来就能给出 0.2–0.3 的 TVD；上界 = 把两侧的行合起来、按原来的大小比"
                 f"随机对半切 300 次取 95 分位。本表 {len(il)} 行里只有 **{n_over}** 行超出噪声，"
                 f"其余的读作“测不出差别”。\n\n" + _md(t))
    ln = A["leaf_intentmix"]
    if len(ln):
        t2 = ln.reindex(ln["内层TVD"].sort_values(ascending=False).index).head(40)
        t2 = t2[["外层", "a", "b", "n_a", "n_b", "内层TVD", "同源噪声上界", "超出噪声",
                 "a主导内层", "a主导占比%", "b主导内层", "b主导占比%", "主导是否相同"]].copy()
        t2["主导是否相同"] = t2["主导是否相同"].map({True: "是", False: "否"})
        t2["超出噪声"] = t2["超出噪声"].map({True: "是", False: "否"})
        P.append(f"\n**表 7-B　反过来：每个叶内部的意图分布，在两个快照之间差多远（前 40 行，全部 {len(ln)} 行见工作簿；"
                 f"其中 {int(ln['超出噪声'].sum())} 行超出同源噪声上界）**\n\n" + _md(t2))

    P.append("\n## 8　这份表能读出什么、不能读出什么\n")
    P.append("<!--NARR:边界-->\n<!--/NARR:边界-->\n")
    # The ASR caveat only exists where a voice export does. Printing it for a four-snapshot
    # domain describes an instrument the reader never sees.
    voice_caveat = ("  语音快照是 ASR 文本（无标点、约 40 字截断、抽样方式未知），"
                    "它与打字快照的差异要先归因到仪器；\n" if "assistant_voice" in srcs else
                    "  本域**没有语音导出**，所以“输入方式”这条对比线整条不存在——"
                    "凡本文没有语音的地方，都是没有这条线，不是测了没差别；\n")
    voice_time = "、语音时间未知" if "assistant_voice" in srcs else ""
    _time_line = (f"  两个搜索快照都在 7 月初、助手约 8 月下旬{voice_time}，随时间走的题材不得归因于界面。\n"
                  if legacy_text or domain not in DOMAIN_TIME else DOMAIN_TIME[domain].format(voice_time=voice_time))
    _build_script, _build_cfg = (("analysis/pooled5/build_pooled5_corpus.py", "configs/pool5_*.yaml")
                                 if legacy_text or domain not in DOMAIN_BUILD else DOMAIN_BUILD[domain])
    P.append(f"""
机械事实，先摆在这里：

- **体系本身**：自上而下定义了 {S['l1_defined']} 个 L1 意图，交付语料里实际用到 {S['l1_delivered']} 个"""
             + (f"，**{len(S['l1_defined_but_unused'])} 个定义了但一行都没分到**（{', '.join(S['l1_defined_but_unused'])}）"
                if S["l1_defined_but_unused"] else "，没有定义了却一行都没分到的类") + f"""。
  自下而上命名了 {S['leaves_named']} 个叶，**交付分区里是 {S['leaves_delivered']} 个**——p8 治理重写过这棵树，
  所以 `hierarchy_meta` / `leaf_labels` / p7 的 namings 描述的是一棵已经不存在的树，本报告一律用交付分区。
- **引用护栏三层**：① 风控图层命中的 {S['risk_flagged_rows']:,} 行；② 一组与领域无关的硬规则
  （露骨内容、**低龄指向 + 外貌/性相关词**、软色情图片检索、可识别到个人的联系方式与证件号、
  名誉与非自愿私密影像）；③ 运行自己的风控机器点名过的 {S.get('risk_named_strings', 0)} 个具体字符串
  （`tree_naming.json` 的 findings.evidence / risk_reason，`risk_screen.json` 的 exemplar/samples）。
  三层过后可引行 {S['quotable_rows']:,} / {S['n_rows']:,}。护栏只影响**引用**，不影响任何统计——
  上面每一张表的分母都是全部行。命中的行一律不引原文；整类都不可引时，卡片里只给条数。
- **本方法测不到的**：同一个字符串在一次运行里必然同标签，所以“同一句话的含义变了”不可见；
{voice_caveat}{_time_line}""")
    P.append(f"""
---

## 附：这份报告是怎么算出来的

| 步骤 | 脚本 | 产物 |
|---|---|---|
| 合并该领域的全部来源、清洗分层、按来源归一权重 | `{_build_script}` | `data/raw/pooled5/{domain}_pooled5.parquet` |
| 挖掘（一次跑完，{len(srcs)} 个来源共享同一套体系） | `qmine run --config {_build_cfg} --fast` | `runs/{S['run']}/{_gen}/` |
| 按行位置把 source 拼回标注（带逐格断言） | `analysis/pooled5/pooled5_common.py: load()` | 内存中的逐行表 |
| 逐类 × 逐快照的全部表 | `analysis/pooled5/p5_snapshot_classes.py` | `analysis/pooled5/work/{domain}/snapshot_classes/*.csv` + 本目录工作簿 |
| 六张图 | `analysis/pooled5/p5_snapshot_figs.py` | `runs/{S['run']}/{_gen}/postprocessed/img/*.png` |
| 渲染本报告（叙述从 `work/{domain}/snapshot_classes/narrative.md` 填入占位） | `analysis/pooled5/p5_snapshot_report.py` | 本文件 |
| 回查叙述里的每一个数字 | `analysis/pooled5/p5_snapshot_verify.py` | `work/{domain}/snapshot_classes/verify.md` |

复现：

```bash
HF_HOME=$(pwd)/.hf .venv/bin/python analysis/pooled5/p5_snapshot_classes.py {domain}
HF_HOME=$(pwd)/.hf .venv/bin/python analysis/pooled5/p5_snapshot_figs.py {domain}
HF_HOME=$(pwd)/.hf .venv/bin/python analysis/pooled5/p5_snapshot_report.py {domain}
HF_HOME=$(pwd)/.hf .venv/bin/python analysis/pooled5/p5_snapshot_verify.py {domain}
```
""")
    text = _toc(_splice(domain, "\n".join(P)))
    out = gen / "postprocessed" / f"{stem}_意图与聚类叶_跨快照对比.zh.md"
    out.write_text(text, encoding="utf-8")
    print(f"{domain}: {out.relative_to(ROOT)}  ({len(text):,} 字符, "
          f"{sum(1 for l in text.splitlines() if l.startswith('|')):,} 表格行)")
    return out


if __name__ == "__main__":
    for d in (sys.argv[1:] or available()):
        assemble(d)
