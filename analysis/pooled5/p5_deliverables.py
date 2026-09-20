# -*- coding: utf-8 -*-
"""Per-domain deliverable: every row carries the label the run gave it AND the source it came
from (2025搜索 / 2026搜索 / 助手头部1k / 助手随机1k / 助手语音1k), plus the rows the cleaning
tiers removed before mining, so nothing is silently missing.

    python analysis/pooled5/p5_deliverables.py [domain ...]
"""
from __future__ import annotations
import re, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pooled5_common import SRC_ZH, all_rows, available, load, run_dir

OUT = Path(__file__).resolve().parent / "deliverables"
ILLEGAL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
NOTE = [
    ["本文件是什么", "同一领域的五个来源合并成一份语料、由挖掘程序一次跑完后的逐行标注结果。"
     "一次运行=一套意图体系=一套聚类树，所以不同来源之间可以直接比较；分开跑的两次运行不能（实测同样 10,000 行的两次金融运行共享 0/35 个类目编码）。"],
    ["来源列 source", "2025search=2025-07-01 搜索前1万；2026search=2026-07-01 搜索前1万；"
     "assistant_top=AI 助手该类目 Top1000；assistant_random=AI 助手该类目随机 1000；assistant_voice=AI 助手语音 query 1000（仅金融、医疗）。"],
    ["清洗", "助手行沿用已审计的 v3 分层（新移除行精度 96.7%）；搜索行套用同一套 C1/S5 规则；"
     "语音行只跑无内容规则（没有 PV，按 PV 触发的规则无法生效）。tier=user 的行才进入挖掘，其余在「被清洗剔除」表里，带原因。"],
    ["权重 pv_norm", "按来源各自归一到 10,000。搜索 PV、助手 PV、以及完全没有 PV 的语音导出是三种不同的计量，"
     "混在一起的原始 PV 会让所有加权指标变成只讲搜索。语音行为均匀权重，所以它的流量占比等于行占比。"],
    ["怎么读占比", "一律看「来源内占比」。各来源行数不同（搜索每年 1 万行，助手每层约 1 千行），跨来源的原始条数没有意义。"],
    ["标签列", "td_* 为自上而下意图体系（L1/L2、置信度、是否模糊、由谁判定）；bu_* 为自下而上聚类（叶、家族、边界margin）。"
     "两套都由本次运行产生，定义见「意图体系定义」「聚类叶定义」两张表。"],
]


#: **逐域的说明与列**：上面的说明是为五域写的（五个来源、v3 清洗、搜索每年 1 万行、l2 列）。
#: 金融8 与健康各有自己的来源、清洗与溯源列；健康的「被清洗剔除」页正是用户要求看得见的产品层
#: （作答 chip、功能/卡片按钮、推送问题、包装模板、医生卡），每行带决定它的是盲标还是规则。
#: 未列在这里的域逐字不变。
DOMAIN_ORDER_EXTRA = {
    "医疗随机": ["query_raw", "wrapper_stripped", "tier_source", "product", "source_file",
                "legacy_l1", "legacy_l2", "legacy_l3", "legacy_dept", "legacy_type"],
    # 医疗3 = 医疗随机 的两个快照 + 健康管家同一天的头部快照，列结构完全一样。
    "医疗3": ["query_raw", "wrapper_stripped", "tier_source", "product", "source_file",
             "legacy_l1", "legacy_l2", "legacy_l3", "legacy_dept", "legacy_type"],
    "金融8": ["query_raw", "code_repaired", "repair_conf"],
    "健康": ["product", "source_file", "legacy_l1", "legacy_l2", "legacy_l3", "legacy_dept", "legacy_type",
             "legacy_label_conflict", "n_days", "days_present", "pv_week_upper", "in_other_source",
             "tier_source", "audit_majority", "flag_bare_term_origin_unknown"],
}
DOMAIN_REMOVED_COLS = {
    # 医疗随机 has no `l2` (the platform's own three-level labels take its place), and every removed row
    # carries the instrument that removed it: the three blind votes, their majority, and the rule tier.
    "医疗随机": ["query", "query_raw", "source", "surface", "domain", "legacy_l2", "legacy_dept", "legacy_type",
                "pv_raw", "tier", "tier_source", "rule_tier", "audit_majority", "audit_votes"],
    "医疗3": ["query", "query_raw", "source", "surface", "domain", "legacy_l2", "legacy_dept", "legacy_type",
             "pv_raw", "tier", "tier_source", "rule_tier", "audit_majority", "audit_votes"],
    # 金融8 不在这里：`金融8_all_rows.parquet` 写于前导零修复**之前**，被剔除的行本来就没被修，没有 query_raw。
    "健康": ["query", "source", "product", "source_file", "surface", "domain", "legacy_l2", "legacy_l3",
             "n_days", "days_present", "pv_raw", "tier", "tier_source", "audit_majority", "audit_votes", "rule_tier"],
}


def domain_note(domain: str, d: pd.DataFrame, removed: pd.DataFrame) -> list[list[str]]:
    if domain == "金融8":
        return [
            ["本文件是什么", "金融垂类 8 个快照合并成一份语料、由挖掘程序一次跑完后的逐行标注结果。"
             "一次运行=一套意图体系=一套聚类树，所以不同来源之间可以直接比较；与已交付的 5 快照金融（fin-pool5）是两次运行，"
             "类目编码互不相通，不能逐类对照。"],
            ["来源列 source", "2025search / 2026search = 该年 7 月 1 日搜索按 PV 排序的前 1 万；2025search_rand / 2026search_rand = "
             "同一天的随机 1 万；assistant_top = AI 助手该类目 Top1000；assistant_random = 随机 1000；"
             "assistant_voice_top = 语音头部 1000（自带 wise_pv）；assistant_voice = 语音导出 1000（没有 PV，抽样方式未知）。"],
            ["清洗", "助手行沿用已审计的 v3 分层；四个搜索层套用同一套 C1/S5 规则；语音头部自带 PV，可以触发 S5；无 PV 的语音导出"
             "只跑无内容规则。tier=user 的行才进入挖掘，其余在「被清洗剔除」表里，带原因。另：源 xlsx 把纯数字 query 存成了数字，"
             "构建时补回证券代码前导零 839 行（query_raw 为原文，code_repaired / repair_conf 标出改过的行）。"],
            NOTE[3],
            ["怎么读占比", "一律看「来源内占比」。搜索每层 1 万行、助手每层约 1 千行，跨来源的原始条数没有意义。"],
            NOTE[5],
        ]
    if domain == "医疗8":
        return [
            ["本文件是什么", "医疗垂类 8 个快照合并成一份语料、由挖掘程序一次跑完后的逐行标注结果。"
             "一次运行=一套意图体系=一套聚类树，所以不同来源之间可以直接比较；与已交付的 5 快照医疗（med-pool5）是两次运行，"
             "类目编码互不相通，不能逐类对照；与「健康」周榜（health-pool2）也不是同一套数据。"],
            ["来源列 source", "2025search / 2026search = 该年 7 月 1 日医疗搜索按 PV 排序的前 1 万；2025search_rand / 2026search_rand = "
             "同一天的随机 1 万；assistant_top = AI 助手医疗类目 Top1000；assistant_random = 随机 1000；"
             "assistant_voice_top = 语音头部（1 万行按 PV 降序的导出里 PV 不低于第 1,000 行的 1,067 行）；"
             "assistant_voice = 语音导出 1000（没有 PV，抽样方式未知）。"],
            ["清洗", "助手行沿用已审计的 v3 分层；四个搜索层套用同一套 C1/S5 规则；语音头部自带 PV，可以触发 S5；无 PV 的语音导出"
             "只跑无内容规则。tier=user 的行才进入挖掘，其余在「被清洗剔除」表里，带原因。"],
            NOTE[3],
            ["怎么读占比", "一律看「来源内占比」。搜索每层 1 万行、助手每层约 1 千行，跨来源的原始条数没有意义。"],
            NOTE[5],
        ]
    if domain == "人物8":
        return [
            ["本文件是什么", "人物垂类 8 个快照合并成一份语料、由挖掘程序一次跑完后的逐行标注结果。"
             "一次运行=一套意图体系=一套聚类树，所以不同来源之间可以直接比较；与已交付的 4 快照人物（ppl-pool5b）是两次运行，"
             "类目编码互不相通，不能逐类对照。"],
            ["来源列 source", "2025search / 2026search = 该年 7 月 1 日人物搜索按 PV 排序的前 1 万；2025search_rand / 2026search_rand = "
             "同一天的随机 1 万；assistant_top = AI 助手人物类目 Top1000；assistant_random = 随机 1000；"
             "assistant_voice_top = 语音头部 1000（自带 wise_pv、按 PV 降序，导出方已在 1,000 行处截断，PV 下限 10，所以并列边界无从判断）；"
             "assistant_voice_random = 语音随机 1000（自带 wise_pv，PV 中位 1）。"],
            ["清洗", "助手行沿用已审计的 v3 分层；四个搜索层套用同一套 C1/S5 规则；两份语音导出都自带 PV，所以都跑无内容规则与 PV 门控的标题规则。"
             "tier=user 的行才进入挖掘，其余在「被清洗剔除」表里，带原因。助手头部被剔除的 111 行带走了该导出 33.43% 的原始 PV，"
             "所以本域一律以行占比为主、流量只作补充。"],
            NOTE[3],
            ["怎么读占比", "一律看「来源内占比」。搜索每层 1 万行、助手每层约 1 千行，跨来源的原始条数没有意义。"],
            NOTE[5],
        ]
    if domain == "影视8":
        return [
            ["本文件是什么", "影视垂类 8 个快照合并成一份语料、由挖掘程序一次跑完后的逐行标注结果。"
             "一次运行=一套意图体系=一套聚类树，所以不同来源之间可以直接比较；与已交付的 4 快照影视（film-pool5）是两次运行，"
             "类目编码互不相通，不能逐类对照。"],
            ["来源列 source", "2025search / 2026search = 该年 7 月 1 日影视搜索按 PV 排序的前 1 万；2025search_rand / 2026search_rand = "
             "同一天的随机 1 万；assistant_top = AI 助手影视动漫类目 Top1000；assistant_random = 随机 1000；"
             "assistant_voice_top = 语音头部 1000（自带 wise_pv、按 PV 降序，导出方已在 1,000 行处截断，PV 下限 21，所以并列边界无从判断）；"
             "assistant_voice_random = 语音随机 1000（自带 wise_pv，PV 中位 1）。"],
            ["清洗", "助手行沿用已审计的 v3 分层；四个搜索层套用同一套 C1/S5 规则；两份语音导出都自带 PV，所以都跑无内容规则与 PV 门控的标题规则。"
             "tier=user 的行才进入挖掘，其余在「被清洗剔除」表里，带原因。助手头部被剔除的 45 行带走了该导出 78.33% 的原始 PV，"
             "所以本域一律以行占比为主、流量只作补充。"],
            NOTE[3],
            ["怎么读占比", "一律看「来源内占比」。搜索每层 1 万行、助手每层约 1 千行，跨来源的原始条数没有意义。"],
            NOTE[5],
        ]
    if domain == "医疗随机":
        return [
            ["本文件是什么", "2026-09-14 当天、医疗类目、去重后的随机 1 万：传统搜索与健康管家两份导出并成一份语料、"
             "由挖掘程序一次跑完后的逐行标注结果。与 health-pool2（同两个产品的周榜 top1w）是两次运行，类目编码互不相通。"],
            ["来源列 source", "msearch_2609r = 传统搜索随机 1 万；mai_2609r = 健康管家随机 1 万（10,622 串）。"],
            ["query 与 query_raw", "助手侧 75.4% 的行带产品输入模板前缀「我想咨询」，本次剥离前缀保留该行；query 是剥离后的文本，"
             "query_raw 是导出原文。"],
            ["清洗", "只剔除产品自己的产物（问诊答项、功能入口、医生卡片、无内容），且必须三名盲标视角一致才剔除，分票保留为 user；"
             "被剔除的行在「被清洗剔除」表里，带三票与规则标记。"],
            NOTE[3],
            ["怎么读占比", "一律看「来源内占比」。两侧各约 1 万行，但 PV 是两种计量，跨来源的原始条数没有意义。"],
            NOTE[5],
        ]
    if domain == "健康":
        n_by = d.groupby("source").size()
        r_by = removed.groupby("source").size()
        z = {k: SRC_ZH[k] for k in n_by.index}
        return [
            ["本文件是什么", "同一周（2026-09-04 至 09-10）的健康搜索周榜与专科健康 AI 管家周榜，按 query 合并后并成一份语料、"
             "由挖掘程序一次跑完后的逐行标注结果。一次运行=一套意图体系=一套聚类树，所以两个来源之间可以直接比较。"],
            ["来源列 source", "hsearch_2609w = 健康搜索周榜；hai_2609w = 健康 AI 管家周榜。两份原始导出都是按「query × 天」取前 1 万行，"
             "构建时按 query 合并成周粒度（pv_raw = 周 PV 之和；n_days / days_present = 在榜天数与日期）。product / source_file "
             "列给出产品名与原始文件名。7 天都在榜时周 PV 精确，否则是下界，pv_week_upper 给出上界。"],
            ["清洗", "搜索：无内容规则、新闻标题规则，外加 16 个「科室+姓名+医生」且 16 位医生周流量几乎一致的串。"
             "AI 管家：全部串由三个独立视角盲标（Fleiss κ 0.945），三票一致时由盲标决定、分票时由规则决定（tier_source 列）。"
             "作答 chip、功能/卡片按钮、推送问题、包装模板、医生卡进「被清洗剔除」表，带 tier、tier_source、audit_majority、"
             "rule_tier。口径与全部数字见 analysis/pooled5/work/health_clean_audit.md。"
             + "；".join(f"{z[k]} 入挖掘 {int(n_by[k]):,} 行、剔除 {int(r_by.get(k, 0)):,} 行" for k in n_by.index) + "。"],
            ["权重 pv_norm", "周 PV 按来源各自归一到 10,000，只能在同一来源内部读作占比。搜索与 AI 管家的 PV 是两种计量。"],
            ["怎么读占比", "一律看「来源内占比」。两个来源入挖掘的行数相差约十倍，跨来源的原始条数没有意义。"],
            ["平台自带分类 legacy_*", "legacy_l2（6 类）、legacy_l3（科室-类型，拆成 legacy_dept / legacy_type）。本次运行把 legacy_l2 与 "
             "legacy_type 声明为参考列（仅参考，从不作监督）；legacy_dept 按运行前写下的条件撤下。"],
            ["flag_bare_term_origin_unknown", "AI 侧保留的裸症状/疾病/药品/检查名：可能是键入的，也可能是从助手症状列表里点出来的，"
             "导出无法区分，任何 AI 侧占比都可以去掉这些行重算。"],
            NOTE[5],
        ]
    return NOTE


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == object or str(df[c].dtype).startswith("str"):
            df[c] = df[c].astype("string").map(lambda x: ILLEGAL.sub("", x) if isinstance(x, str) else x)
    return df


def build(domain: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    d = load(domain)
    d = d.rename(columns={"source_zh": "来源"})
    order = ["query", "来源", "source", "surface", "domain", "l2", "pv_raw", "pv_norm", "tier",
             "td_l1", "td_l1_name", "td_l2", "td_user_need", "td_confidence", "td_margin",
             "td_ambiguous", "td_decided_by", "bu_leaf", "bu_leaf_name", "bu_user_need",
             "bu_family_final", "bu_margin", "bu_ambiguous", "row_id", "row_id_all"]
    extra = DOMAIN_ORDER_EXTRA.get(domain, [])
    if extra:
        i = order.index("tier") + 1
        order = order[:i] + extra + order[i:]
    full = d[[c for c in order if c in d.columns]].sort_values(["source", "pv_norm"], ascending=[True, False])
    full.to_parquet(OUT / f"{domain}_pooled5_labeled.parquet", index=False)
    a = all_rows(domain)
    rcols = DOMAIN_REMOVED_COLS.get(domain, ["query", "source", "surface", "domain", "l2", "pv_raw", "tier"])
    miss = [c for c in rcols if c not in a.columns]
    assert not miss, f"{domain}: all_rows lacks {miss}"
    removed = a[~a.kept][rcols].copy()
    removed["来源"] = removed["source"].map(SRC_ZH)
    rows = (d.groupby(["td_l1_name", "source"]).size().unstack(1).fillna(0).astype(int))
    rowshare = (rows / rows.sum()).round(4)
    pvs = d.groupby(["td_l1_name", "source"])["pv_norm"].sum().unstack(1).fillna(0)
    pvshare = (pvs / pvs.sum()).round(4)
    fams = (d.groupby(["bu_family_final", "source"]).size().unstack(1).fillna(0).astype(int))
    src_x = run_dir(domain) / next((p.name for p in run_dir(domain).glob("*_query_挖掘结果.xlsx")), "")
    defs = {}
    # `.is_file()`, not `.exists()`: when the glob finds nothing `next(..., "")` makes `src_x` the
    # run DIRECTORY, and a directory exists — pd.ExcelFile then gets handed a folder. Domains that
    # have the workbook are unaffected.
    if src_x.is_file():
        xl = pd.ExcelFile(src_x)
        for sh in ("意图体系定义", "裁决规则", "聚类叶定义", "模板群定义", "风控图层分布", "风险内容清单"):
            if sh in xl.sheet_names:
                defs[sh] = xl.parse(sh)
    out = OUT / f"{domain}_POOLED5_逐行标注.xlsx"
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        pd.DataFrame(domain_note(domain, d, removed), columns=["项", "说明"]).to_excel(w, sheet_name="说明", index=False)
        _clean(full).to_excel(w, sheet_name="全量标注(含来源)", index=False)
        _clean(removed).to_excel(w, sheet_name="被清洗剔除", index=False)
        rows.to_excel(w, sheet_name="条数_意图×来源")
        rowshare.to_excel(w, sheet_name="行占比_意图×来源")
        pvshare.to_excel(w, sheet_name="流量占比_意图×来源")
        fams.to_excel(w, sheet_name="条数_家族×来源")
        for sh, t in defs.items():
            _clean(t).to_excel(w, sheet_name=sh, index=False)
    print(f"{domain}: {out.name} — {len(full):,} labelled rows, {len(removed):,} removed rows, "
          f"{rows.shape[0]} L1 classes")
    return out


if __name__ == "__main__":
    for x in (sys.argv[1:] or available()):
        build(x)
