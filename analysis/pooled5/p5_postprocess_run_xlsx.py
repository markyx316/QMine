# -*- coding: utf-8 -*-
"""Add the SOURCE of every row to the run's own delivered workbook, without touching it.

The pipeline's `<domain>_query_挖掘结果.xlsx` labels every row but cannot say where the row came
from: `source` never entered the run (declaring it would have asked the K locator to find a
partition that separates the surfaces, which is the comparison, not the frame). This tool joins
it back and writes a copy into `<generation>/postprocessed/`, leaving every original artifact in
place as evidence.

THE JOIN IS POSITIONAL AND IT IS ASSERTED. `全量标注` comes out of `labels_full.csv`, which
preserves input row order exactly — verified here per domain by comparing all ~22,000 query
strings cell by cell, and the tool refuses to write if a single one differs. A text join would
be WRONG: the same string occurs in several sources (that overlap is what the study measures),
so joining on query text would bind a row to the wrong surface.

`风险内容清单` has no row ids, so its source is resolved by string lookup; a string that exists
in more than one source is labelled with all of them and flagged `来源是否唯一 = 否`.

    python analysis/pooled5/p5_postprocess_run_xlsx.py [domain ...]
"""
from __future__ import annotations
import re, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pooled5_common import SOURCES, SRC_ZH, available, run_dir

ROOT = Path(__file__).resolve().parents[2]
ILLEGAL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
SRC_COLS = ["来源", "source", "surface", "l2", "pv_raw", "pv_norm", "tier"]
NOTE = [
    ["来源列是怎么加上去的",
     "挖掘运行本身不知道每行来自哪个来源（source 从未进入运行：把它声明为参照列，等于要求聚类去把不同来源分开，"
     "而那正是本研究要比较的东西）。本文件把来源按行位置拼回去：运行交付的「全量标注」页出自 labels_full.csv，"
     "而它逐行保持输入顺序——本工具对全部行的 query 字符串逐格比对确认一致后才写出，有一格不同就拒绝生成。"],
    ["为什么不能按文本join",
     "同一个字符串常常同时出现在多个来源（这种重合正是本研究要测量的），按 query 文本 join 会把一行绑到错误的来源上。"],
    ["source 的五个取值",
     "2025search = 2025-07-01 搜索该垂类前 1 万；2026search = 2026-07-01 同上；assistant_top = AI 助手该类目 Top1000；"
     "assistant_random = AI 助手该类目随机 1000；assistant_voice = AI 助手语音 query 1000（仅金融、医疗）。"],
    ["tier 列", "v3 清洗分层的结果。本次挖掘只喂进 tier=user 的行，所以这张表里全部是 user；"
                "被剔除的系统串、按钮串、卡片段落见 analysis/pooled5/deliverables/<领域>_POOLED5_逐行标注.xlsx 的「被清洗剔除」页。"],
    ["pv_raw / pv_norm",
     "pv_raw 是各来源自带的流量字段（搜索 wise_pv、助手 search_num、语音没有）。pv_norm 按来源各自归一到 10,000，"
     "所以只能在同一来源内部读作占比，不能跨来源比较绝对值。"],
    ["风险内容清单页", "该页没有行号，来源按字符串反查；字符串若同时存在于多个来源，会列出全部并把「来源是否唯一」标为否。"],
]
#: **逐域的列与说明。** 上面那一套是为五域/新两域写的（五个 source、v3 清洗、l2 列），印在金融8
#: 与健康的交付物上就是错的：金融8 有 8 个 source、还有前导零修复列；健康的语料没有 `l2` 列（直接
#: KeyError），却有来源文件、周粒度、平台分类、盲标决定层级等用户要求「看得出每行从哪来」的列。
#: 列在这里的域用自己的列与说明，其余域逐字不变。
DOMAIN_SRC_COLS = {
    "金融8": ["来源", "source", "surface", "l2", "pv_raw", "pv_norm", "tier", "query_raw", "code_repaired", "repair_conf"],
    "健康": ["来源", "source", "product", "source_file", "surface", "legacy_l1", "legacy_l2", "legacy_l3", "legacy_dept",
             "legacy_type", "legacy_label_conflict", "n_days", "days_present", "first_day", "last_day", "n_rows_raw",
             "pv_raw", "pv_week_upper", "pv_norm", "daily_cv", "in_other_source", "tier", "tier_source",
             "audit_majority", "flag_bare_term_origin_unknown"],
    # 医疗随机 没有 `l2`（那是 pool5 系语料的旧类目列），也没有周榜那批列（n_days/pv_week_upper/daily_cv
    # 都是「一周内出现几天」的产物，这份语料是单日导出）。它自己的来龙去脉在 legacy_* 与 tier_source 里：
    # `wrapper_stripped` 说明这一行的助手输入包装被剥掉过，`query_raw` 保留剥之前的原文，两列一起才能复核清洗。
    "医疗3": ["来源", "source", "product", "source_file", "surface", "legacy_l1", "legacy_l2", "legacy_l3",
             "legacy_dept", "legacy_type", "pv_raw", "pv_norm", "tier", "tier_source",
             "wrapper_stripped", "query_raw"],
    "医疗随机": ["来源", "source", "product", "source_file", "surface", "legacy_l1", "legacy_l2", "legacy_l3",
                 "legacy_dept", "legacy_type", "pv_raw", "pv_norm", "tier", "tier_source",
                 "wrapper_stripped", "query_raw"],
}
DOMAIN_NOTE = {
    "医疗随机": NOTE[:2] + [
        ["source 的两个取值",
         "msearch_2609r = 2026-09-14 传统搜索医疗类目的随机 1 万（已去重，每串一行）；"
         "mai_2609r = 同一天健康管家医疗类目的随机 1 万（10,622 串）。两份都带平台自己的三级类目。"],
        ["query 与 query_raw",
         "助手侧 75.4% 的行以产品输入模板「我想咨询」开头：query 是剥离前缀后的文本（用户自己的问题），"
         "query_raw 是导出原文，wrapper_stripped 标出改过的行。"],
        ["tier 列", "只有产品自己的产物被剔除（问诊答项、功能入口、医生卡片、无内容），而且**三名盲标视角一致**才剔除；"
         "分票的一律保留为 user。被剔除的行及其三票见 analysis/pooled5/work/医疗随机_all_rows.parquet。"],
    ],
    "人物8": NOTE[:2] + [
        ["source 的八个取值",
         "2025search / 2026search = 该年 7 月 1 日搜索按 PV 排序的前 1 万；2025search_rand / 2026search_rand = 同一天的随机 1 万；"
         "assistant_top = AI 助手该类目 Top1000；assistant_random = AI 助手该类目随机 1000；"
         "assistant_voice_top = 语音头部 1000（自带 wise_pv，按 PV 降序，导出在 1,000 行截断，PV 下限 10）；"
         "assistant_voice_random = 语音随机 1000（自带 wise_pv，PV 中位 1）。"],
        ["tier 列", "本次挖掘只喂进 tier=user 的行，所以这张表里全部是 user；被清洗剔除的行及其分层见 "
         "analysis/pooled5/work/<域>_all_rows.parquet 与 build_audit_*.csv。"],
    ],
    "影视8": NOTE[:2] + [
        ["source 的八个取值",
         "2025search / 2026search = 该年 7 月 1 日搜索按 PV 排序的前 1 万；2025search_rand / 2026search_rand = 同一天的随机 1 万；"
         "assistant_top = AI 助手该类目 Top1000；assistant_random = AI 助手该类目随机 1000；"
         "assistant_voice_top = 语音头部 1000（自带 wise_pv，按 PV 降序，导出在 1,000 行截断，PV 下限 21）；"
         "assistant_voice_random = 语音随机 1000（自带 wise_pv，PV 中位 1）。"],
        ["tier 列", "本次挖掘只喂进 tier=user 的行，所以这张表里全部是 user；被清洗剔除的行及其分层见 "
         "analysis/pooled5/work/<域>_all_rows.parquet 与 build_audit_*.csv。"],
    ],
    "金融8": NOTE[:2] + [
        ["source 的八个取值",
         "2025search / 2026search = 该年 7 月 1 日搜索按 PV 排序的前 1 万；2025search_rand / 2026search_rand = 同一天的随机 1 万；"
         "assistant_top = AI 助手该类目 Top1000；assistant_random = AI 助手该类目随机 1000；"
         "assistant_voice_top = 语音头部 1000（自带 wise_pv）；assistant_voice = 语音导出 1000（没有 PV 字段，抽样方式未知）。"],
        ["query_raw / code_repaired / repair_conf",
         "源 xlsx 把纯数字 query 存成了数字，证券代码的前导零在进任何代码之前就没了（000519 → 519）。"
         "构建时按码段与语料内孪生证据补回 839 行：query 是修复后的文本，query_raw 是原文，code_repaired 标出改过的行，"
         "repair_conf=low_yearlike 的 10 行是「像年份、又没有孪生佐证」的低置信修复。证据见 build_fin8_corpus.py 的说明。"],
        ["tier 列", "本次挖掘只喂进 tier=user 的行，所以这张表里全部是 user；被清洗剔除的行及其分层见 "
                    "analysis/pooled5/work/金融8_all_rows.parquet。"],
        NOTE[4], NOTE[5]],
    "医疗8": NOTE[:2] + [
        ["source 的八个取值",
         "2025search / 2026search = 该年 7 月 1 日医疗搜索按 PV 排序的前 1 万；2025search_rand / 2026search_rand = 同一天的随机 1 万；"
         "assistant_top = AI 助手医疗类目 Top1000；assistant_random = AI 助手医疗类目随机 1000；"
         "assistant_voice_top = 语音头部（导出为按 wise_pv 降序的 1 万行，取 PV 不低于第 1,000 行 PV=11 的 1,067 行，不在并列处截断）；"
         "assistant_voice = 语音导出 1000（没有 PV 字段，抽样方式未知）。"],
        ["tier 列", "本次挖掘只喂进 tier=user 的行，所以这张表里全部是 user；被清洗剔除的行及其分层见 "
                    "analysis/pooled5/work/医疗8_all_rows.parquet。"],
        NOTE[4], NOTE[5]],
    "健康": NOTE[:2] + [
        ["source 的两个取值",
         "hsearch_2609w = 健康搜索周榜；hai_2609w = 健康 AI 管家周榜。两份都是同一周（2026-09-04 至 09-10）、"
         "按「query × 天」取前 1 万行的导出，构建时按 query 合并成一行。product 与 source_file 列给出产品名和原始文件名。"],
        ["每一行是一个 query 的一周合计",
         "pv_raw = 该 query 当周所有原始行的 PV 之和；n_days / days_present / first_day / last_day = 当周在榜的天数与日期；"
         "n_rows_raw = 合并前的原始行数（同一天被导出拆到两个分类上的也各算一行，PV 相加）。7 天都在榜时 pv_raw 是精确值，"
         "否则是下界；pv_week_upper 按每个缺席日自己的截断线补出的上界。"],
        ["legacy_* 列",
         "平台自带的分类：legacy_l1（恒为 医疗）、legacy_l2（6 类）、legacy_l3（科室-类型），并把 l3 拆成 legacy_dept 与 legacy_type。"
         "一个 query 跨天分类不一致时取 PV 占优者并置 legacy_label_conflict。本次运行把 legacy_l2 与 legacy_type 声明为参考列"
         "（仅参考，从不作监督）；legacy_dept 按运行前写下的条件撤下（两个占比 >1% 的类只出现在一侧）。"],
        ["tier / tier_source / audit_majority",
         "本表只含入挖掘的 user 行。AI 侧每行的 tier_source 说明它由三个独立视角的盲标一致决定（audit_unanimous），"
         "还是在分票时由规则决定（rule）；audit_majority 是盲标多数。被清洗掉的产品层——作答 chip、功能/卡片按钮、推送问题、"
         "包装模板、医生卡——连同分层见 analysis/pooled5/work/健康_all_rows.parquet，口径与全部数字见 work/health_clean_audit.md。"],
        ["flag_bare_term_origin_unknown",
         "AI 侧保留下来的裸症状/疾病/药品/检查名（184 行）：它可能是用户键入的，也可能是从助手的症状列表里点出来的，"
         "导出无法区分。任何 AI 侧占比都可以去掉这些行重算。"],
        ["in_other_source", "同一个字符串是否也出现在另一个来源里（24 串两边都有）。两边各保留一行，不合并。"],
        NOTE[4], NOTE[5]],
}
#: 运行自己交付的工作簿里，以 `=` 开头的 query 被 openpyxl 写成了公式，读回来是空。
#: 本工具从语料 parquet 修回来，并在「来源说明」页记一行。七个域实测只有 1 格（软件）。
REPAIR_NOTE = ["工作簿里被修回来的格子",
               "运行自己交付的 `全量标注` 页里，**以 = 开头的 query 被 Excel 当成公式**，读回来是空白。"
               "本文件从 `data/raw/pooled5/<领域>_pooled5.parquet`（权威原文）把这些格子修了回来，"
               "并在此记明。七个领域逐格比对的结果：只有 {n} 格。除这一种情况外，"
               "任何一格文本对不上都会让本工具**拒绝生成**，因为那意味着位置对齐不安全。"]


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == object or str(df[c].dtype).startswith("str"):
            df[c] = df[c].astype("string").map(lambda x: ILLEGAL.sub("", x) if isinstance(x, str) else x)
    return df


def postprocess(domain: str) -> Path:
    gen = run_dir(domain)
    books = sorted(gen.glob("*_query_挖掘结果.xlsx"))
    if not books:
        raise SystemExit(f"{domain}: no delivered workbook under {gen}")
    book = books[0]
    corpus = pd.read_parquet(ROOT / f"data/raw/pooled5/{domain}_pooled5.parquet")
    xl = pd.ExcelFile(book)
    full = xl.parse("全量标注")
    if len(full) != len(corpus):
        raise SystemExit(f"{domain}: workbook has {len(full):,} rows, corpus has {len(corpus):,} — "
                         f"positional join refused")
    # ONE tolerated difference, and only this one: openpyxl writes a string starting with `=`
    # as a FORMULA, so `===发货啦~发货啦===…` (soft-pool5 row 4725 — the spam/phishing payload the
    # risk sentinel flagged) comes back as NaN. Measured across all seven delivered workbooks:
    # exactly 1 cell, in 软件. The corpus parquet is authoritative, so the cell is repaired from it
    # and the repair is recorded in the 来源说明 sheet. ANY other mismatch still refuses — a
    # genuine text difference means the positional join is unsafe.
    wq, cq = full["query"], corpus["query"].astype(str)
    mismatch = wq.astype(str).values != cq.values
    formula_loss = mismatch & wq.isna().to_numpy() & cq.str.startswith("=").to_numpy()
    repaired = int(formula_loss.sum())
    if repaired:
        full = full.copy()
        full.loc[formula_loss, "query"] = cq[formula_loss].values
    bad = int((mismatch & ~formula_loss).sum())
    if bad:
        raise SystemExit(f"{domain}: {bad} rows where the workbook text differs from the corpus text — "
                         f"positional join refused")
    out_dir = gen / "postprocessed"
    out_dir.mkdir(parents=True, exist_ok=True)
    tagged = full.copy()
    cols = DOMAIN_SRC_COLS.get(domain, SRC_COLS)
    missing = [c for c in cols if c != "来源" and c not in corpus.columns]
    assert not missing, f"{domain}: corpus lacks provenance column(s) {missing}"
    for i, c in enumerate(cols):
        tagged.insert(1 + i, c, corpus["source"].map(SRC_ZH) if c == "来源" else corpus[c].values)
    # 风险内容清单: no row ids, so resolve by string and say so when it is ambiguous
    risk = xl.parse("风险内容清单") if "风险内容清单" in xl.sheet_names else pd.DataFrame()
    if len(risk):
        by_q = corpus.groupby("query")["source"].apply(lambda s: sorted(set(s)))
        srcs = risk["query"].astype(str).map(lambda q: by_q.get(q, []))
        risk = risk.copy()
        risk.insert(1, "来源", srcs.map(lambda xs: " / ".join(SRC_ZH.get(x, x) for x in xs) or "（未找到）"))
        risk.insert(2, "来源是否唯一", srcs.map(lambda xs: "是" if len(xs) == 1 else "否"))
    counts = (corpus.groupby("source").size().reindex([s for s in SOURCES if (corpus.source == s).any()])
              .rename("行数").reset_index())
    counts["来源"] = counts["source"].map(SRC_ZH)
    counts["占比"] = (counts["行数"] / counts["行数"].sum()).round(4)
    out = out_dir / f"{book.stem}_含来源.xlsx"
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        note = list(DOMAIN_NOTE.get(domain, NOTE))
        if repaired:
            note.append([REPAIR_NOTE[0], REPAIR_NOTE[1].format(n=repaired)])
        pd.DataFrame(note, columns=["项", "说明"]).to_excel(w, sheet_name="来源说明", index=False)
        _clean(tagged).to_excel(w, sheet_name="全量标注(含来源)", index=False)
        counts[["来源", "source", "行数", "占比"]].to_excel(w, sheet_name="来源分布", index=False)
        for sh in xl.sheet_names:
            if sh in ("全量标注",):
                continue
            t = risk if (sh == "风险内容清单" and len(risk)) else xl.parse(sh)
            _clean(t).to_excel(w, sheet_name=sh, index=False)
    tagged.to_csv(out_dir / f"{book.stem}_含来源.csv", index=False, encoding="utf-8-sig")
    if repaired:
        print(f"{domain}: repaired {repaired} query cell(s) that Excel had swallowed as a formula")
    print(f"{domain}: {out.relative_to(ROOT)}  ({len(tagged):,} rows, "
          + " · ".join(f"{SRC_ZH[r['source']]} {int(r['行数']):,}" for _, r in counts.iterrows()) + ")")
    return out


if __name__ == "__main__":
    for d in (sys.argv[1:] or available()):
        postprocess(d)
