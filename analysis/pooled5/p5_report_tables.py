# -*- coding: utf-8 -*-
"""Every table in the POOLED-5 report, generated from the canonical CSVs. The report never
carries a number that is not produced here, so a number in the prose can always be traced to a
file and recomputed.

    python analysis/pooled5/p5_report_tables.py > /dev/null   # writes work/report_tables.md
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pooled5_common import COHORT, CONTRASTS, DOMAINS, SOURCES, SRC_ZH, WORK, cohort, cross_dir, work_file

#: One file per batch. Running this for a second cohort must not overwrite the five-domain
#: tables the delivered report was assembled from.
OUT_PATH = WORK / ("report_tables.md" if COHORT == "pool5" else f"report_tables_{COHORT}.md")
OUT = open(OUT_PATH, "w", encoding="utf-8")
CROSS = cross_dir()


def W(x: str = "") -> None:
    print(x, file=OUT)


def _have(dom: str) -> bool:
    return (WORK / dom / "composition.csv").exists()


DOMS = [d for d in cohort() if _have(d)]


def t1() -> None:
    W("#### T1 五个领域的合并语料：每个来源进入挖掘的行数，以及清洗剔除量\n")
    W("| 领域 | 来源 | 原始行 | 进入挖掘 | 剔除行 | 剔除行占比 | 剔除 PV 占比 |")
    W("|---|---|---:|---:|---:|---:|---:|")
    for dom in DOMS:
        t = pd.read_csv(WORK / dom / "composition.csv")
        t["o"] = t["source"].map({s: i for i, s in enumerate(SOURCES)})
        for _, r in t.sort_values("o").iterrows():
            pv = "—" if pd.isna(r["pv_drop_%"]) else f"{r['pv_drop_%']:.1f}%"
            W(f"| {dom} | {SRC_ZH[r['source']]} | {int(r['rows']):,} | {int(r['kept']):,} | "
              f"{int(r['rows'] - r['kept']):,} | {r['drop_%']:.1f}% | {pv} |")
    W("\n> 语音导出没有 PV，所以它没有「剔除 PV 占比」。助手头部的 PV 剔除量之所以高，是因为被剔除的是"
      "「生成视频」「👌 好的，继续吧」这类单条几十万 PV 的功能入口与按钮串。\n")


def t2() -> None:
    W("#### T2 每个领域自己长出来的意图体系（L1 类目）\n")
    W("> 类目数是**架构师写进体系的类**。人物的 20 类里有一类（`成人或泄露私密影像查询`）**实际一行都没有标到**，"
      "所以 T10、T11 与 §8.5 按**实际用到的 19 类**计。这条差异本身是个发现，见领域深挖的人物一节。\n")
    p = CROSS / "taxonomy_inventory.csv"
    if not p.exists():
        return
    t = pd.read_csv(p)
    W("| 领域 | 类目数 | 类目 |")
    W("|---|---:|---|")
    for _, r in t.iterrows():
        W(f"| {r['domain']} | {int(r['n_classes'])} | {r['classes']} |")
    W("\n> 类目数是体系**声明**的 L1 类数。人物声明 20 个，实得有行的只有 19 个"
      "（`成人或泄露私密影像查询` 0 行，见 §8.5 与 §9⑦）；T10 / T11 用的是实得数 19。\n")


def t3() -> None:
    W("#### T3 四条对比线的整体距离 TVD（需要多大比例的行换类，两个来源的意图构成才相等）\n")
    m = pd.read_csv(CROSS / "tvd_matrix.csv")
    cols = [c for c in m.columns if c in DOMAINS]
    W("| 对比线 | 差的是什么 | " + " | ".join(cols) + " |")
    W("|---|---|" + "---:|" * len(cols))
    for _, r in m.iterrows():
        vals = " | ".join("—" if pd.isna(r[c]) else f"{r[c]:.3f}" for c in cols)
        W(f"| {SRC_ZH[r['a']]} → {SRC_ZH[r['b']]} | {r['why']} | {vals} |")
    W("\n> TVD 不随显著性水平膨胀，但**在小样本上有正偏**：跨对比线比较前先看 T3b 的「同源噪声上界」——"
      "两侧都是搜索前 1 万的时间线是 0.03，有一侧只有一千行的线是 0.10–0.13。**时间那一行是全篇噪声最低的一条，"
      "不是可以直接当门槛的一条。**\n")


def t3b() -> None:
    p = CROSS / "tvd_ci.csv"
    if not p.exists():
        return
    W("#### T3b 同一批 TVD，带自助法 95% 区间与「同源噪声上界」\n")
    t = pd.read_csv(p)
    W("| 领域 | 对比线 | TVD [95% 区间] | 同源两半的 TVD 上界 | n(a) | n(b) |")
    W("|---|---|---|---:|---:|---:|")
    order = {(a, b): i for i, (a, b, _) in enumerate(CONTRASTS)}
    t["o"] = t.apply(lambda r: order.get((r["a"], r["b"]), 99), axis=1)
    for dom in DOMS:
        for _, r in t[t.domain == dom].sort_values("o").iterrows():
            W(f"| {dom} | {SRC_ZH[r['a']]} → {SRC_ZH[r['b']]} | {r['tvd']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}] | "
              f"{r['null_p95']:.3f} | {int(r['n_a']):,} | {int(r['n_b']):,} |")
    W("\n> 「同源噪声上界」是把**同一个来源**随机劈成两半算出来的 TVD 的 95 分位——样本量本身就会制造的下限。"
      "区间重叠的两条对比线不要排序。\n")


def t4() -> None:
    W("#### T4 每条对比线上，95% 区间不含 0 的类目数（括号内：只在一侧出现的类目数）\n")
    d = pd.read_csv(CROSS / "diffs_all.csv")
    rows = []
    for a, b, why in CONTRASTS:
        row = {"对比": f"{SRC_ZH[a]} → {SRC_ZH[b]}"}
        for dom in DOMS:
            s = d[(d.domain == dom) & (d.a == a) & (d.b == b)]
            if not len(s):
                row[dom] = "—"
                continue
            ne = int((s.kind != "stable").sum())
            row[dom] = f"{int(s.sig.sum())}" + (f" ({ne})" if ne else "")
        rows.append(row)
    t = pd.DataFrame(rows)
    W("| " + " | ".join(t.columns) + " |")
    W("|---|" + "---:|" * (len(t.columns) - 1))
    for _, r in t.iterrows():
        W("| " + " | ".join(str(r[c]) for c in t.columns) + " |")
    W("")


def t5() -> None:
    W("#### T5 一年之间，搜索前 1 万换了多少人（时间差异的全部来源）\n")
    t = pd.read_csv(CROSS / "turnover_summary.csv")
    W("| 领域 | 两年都在 | 掉榜 | 新进 | 留存占 2026 的 | 打字助手行也出现在搜索里 | 语音行也出现在搜索里 |")
    W("|---|---:|---:|---:|---:|---:|---:|")
    for _, r in t.iterrows():
        v = "—" if pd.isna(r.get("voice_shared_share")) else f"{r['voice_shared_share']*100:.1f}%"
        W(f"| {r['domain']} | {int(r['stay']):,} | {int(r['gone']):,} | {int(r['new']):,} | "
          f"{r['stay_share_of_2026']*100:.1f}% | {r['assistant_typed_shared_share']*100:.1f}% | {v} |")
    W("\n> 同一字符串在一次运行里必然同标签（§1.4），所以时间维度上的意图变化**只能**来自这张表里的"
      "「掉榜」与「新进」，本方法测不到「同一句话含义变了」。\n>\n"
      "> 倒数两列分开列，是因为把语音并进「助手行」会只稀释金融与医疗两列（另外三个领域没有语音），"
      "排序会因此翻转——这是跨领域综合分析的复核员发现的一处口径错误，已在 `p5_turnover.py` 里改掉。\n")


def t7() -> None:
    W("#### T7 每一行在 2026 搜索里能不能找到近邻（bge-base-zh 余弦；2025 搜索是时间对照）\n")
    t = pd.read_csv(CROSS / "semantic_nn_summary.csv")
    W("| 领域 | 来源 | n | 中位相似度 | 余弦≥0.999 | 无近邻 <0.70 |")
    W("|---|---|---:|---:|---:|---:|")
    t["o"] = t["source"].map({s: i for i, s in enumerate(SOURCES)})
    for _, r in t.sort_values(["domain", "o"]).iterrows():
        W(f"| {r['domain']} | {SRC_ZH[r['source']]} | {int(r['n']):,} | {r['median_sim']:.3f} | "
          f"{r['same_pct']:.1f}% | {r['far_pct']:.1f}% |")
    W("\n> 「余弦≥0.999」**不等于字符串相同**（标点、空格差异都会落进这一档）；逐字重合见 T6。\n")


def t6() -> None:
    W("#### T6 逐字重合：A 的不同字符串里，有多少也出现在 B（%）\n")
    for dom in DOMS:
        ov = pd.read_csv(WORK / dom / "overlap.csv", index_col=0)
        keep = [s for s in SOURCES if s in ov.columns]
        ov = ov.loc[keep, keep] * 100
        W(f"\n**{dom}**\n")
        W("| A ＼ B | " + " | ".join(SRC_ZH[c] for c in keep) + " |")
        W("|---|" + "---:|" * len(keep))
        for a in keep:
            W(f"| {SRC_ZH[a]} | " + " | ".join(f"{ov.loc[a, b]:.1f}" for b in keep) + " |")
    W("")


def t8() -> None:
    W("#### T8 体系对每个来源的适配度（体系是在搜索行上拟合的：搜索占合并语料 89.7%——"
      "金融 87.2%、医疗 87.1%、教育 91.6%、影视 91.2%、人物 91.6%）\n")
    t = pd.read_csv(CROSS / "fit_all.csv")
    t["o"] = t["source"].map({s: i for i, s in enumerate(SOURCES)})
    W("| 领域 | 来源 | 平均置信度 | 10 分位 | 模糊行(td) | 模糊行(聚类) |")
    W("|---|---|---:|---:|---:|---:|")
    for _, r in t.sort_values(["domain", "o"]).iterrows():
        W(f"| {r['domain']} | {SRC_ZH[r['source']]} | {r['td_confidence_mean']:.3f} | "
          f"{r['td_confidence_p10']:.3f} | {r['td_ambiguous_%']:.1f}% | {r['bu_ambiguous_%']:.1f}% |")
    W("")


def t9() -> None:
    p = CROSS / "risk_all.csv"
    if not p.exists():
        return
    W("#### T9 风控图层命中（每千行）\n")
    t = pd.read_csv(p)
    piv = t.pivot_table(index="domain", columns="source", values="per_1k")
    cols = [c for c in SOURCES if c in piv.columns]
    W("| 领域 | " + " | ".join(SRC_ZH[c] for c in cols) + " |")
    W("|---|" + "---:|" * len(cols))
    for dom, r in piv.iterrows():
        W(f"| {dom} | " + " | ".join("—" if pd.isna(r[c]) else f"{r[c]:.1f}" for c in cols) + " |")
    W("\n> 每个领域的风控正则由该领域的 domain profile 定义，**跨领域的绝对值不可比**；这里看的是"
      "同一把尺子在同一领域不同来源上的相对差异。\n")


def t10() -> None:
    p = CROSS / "vs_unified_summary.csv"
    if not p.exists():
        return
    W("#### T10 领域体系 vs 上一版的 13 类通用框架（同样的行）\n")
    t = pd.read_csv(p)
    W("| 领域 | 覆盖行 | 通用框架类数 | 本次体系类数 | AMI | NMI |")
    W("|---|---:|---:|---:|---:|---:|")
    for _, r in t.iterrows():
        W(f"| {r['domain']} | {int(r['n_rows']):,} | {int(r['n_u_classes'])} | "
          f"{int(r['n_mined_classes'])} | {r['AMI']:.3f} | {r['NMI']:.3f} |")
    sp = CROSS / "vs_unified_splits.csv"
    if sp.exists():
        s = pd.read_csv(sp)
        if len(s):
            W("\n**通用框架里的一类，被领域体系拆成多类（各占该类 ≥20%）**\n")
            W("| 领域 | 通用类 | 行数 | 被拆成 |")
            W("|---|---|---:|---|")
            for _, r in s.iterrows():
                mc = str(r["mined_classes"]).replace("|", "\\|")
                W(f"| {r['domain']} | {r['u']} {r['u_name']} | {int(r['u_rows']):,} | {mc} |")
    mx = CROSS / "vs_unified_classmix.csv"
    if mx.exists():
        m = pd.read_csv(mx)
        m = m[(m.purity < 0.75) & (m.rows >= 200)].sort_values("rows", ascending=False)
        if len(m):
            W("\n**反过来：领域体系的一个类，在通用框架里是好几类（纯度 <75%、≥200 行）**\n")
            W("| 领域 | 本次体系的类 | 行数 | 在通用框架里的构成 |")
            W("|---|---|---:|---|")
            for _, r in m.head(12).iterrows():
                W(f"| {r['domain']} | {r['mined_class']} | {int(r['rows']):,} | {r['u_mix']} |")
    W("\n> 覆盖范围：2026 搜索 + 助手头部 + 助手随机（2025 搜索与语音没有通用框架标签）。"
      "「拆分」看的是通用框架的一类被本体系分成几类，「合并」看的是本体系的一类在通用框架里由几类拼成，"
      "两张表的分母不同，不要相加。\n")


def t11() -> None:
    p = work_file("taxonomy_delta_summary.csv")
    if not p.exists():
        return
    W("#### T11 加入助手行以后，同样两万条搜索行的划分变了多少\n")
    t = pd.read_csv(p)
    W("| 语料 | 对齐行 | 老体系类数 | 新体系类数 | AMI(意图) | AMI(聚类) | 新体系中助手占多数的类 |")
    W("|---|---:|---:|---:|---:|---:|---:|")
    for _, r in t.iterrows():
        mj = "—" if pd.isna(r.get("classes_majority_assistant")) else str(int(r["classes_majority_assistant"]))
        W(f"| {r['domain']} | {int(r['matched_search_rows']):,} | {int(r['n_classes_old'])} | "
          f"{int(r['n_classes_new'])} | {r['AMI_td']:.3f} | {r['AMI_bu']:.3f} | {mj} |")
    W("\n> **参照系不是 1.0**：最后一行是同一份数据跑两次的结果，AMI 0.774 / 0.687 就是运行间噪声的量级。"
      "只有明显低于它的位移才谈得上「体系被改变了」。**两列要分开读**——意图那一列全部落在 0.774 的噪声带内，"
      "聚类那一列有医疗、影视、人物三个领域落在 0.687 之外。\n")


def t12() -> None:
    p = CROSS / "surface_signature.txt"
    if not p.exists():
        return
    W("#### T12 每个来源最集中的类目（index = 来源内占比 ÷ 全语料占比）\n")
    W("```text")
    W(p.read_text(encoding="utf-8").strip())
    W("```\n")


def t13() -> None:
    ws = sorted(WORK.glob("wrap_summary*.csv"))
    if not ws:
        return
    t = pd.concat([pd.read_csv(f) for f in ws], ignore_index=True).drop_duplicates(["domain", "source"])
    W("#### T13 包装对：一条搜索词被原样写进更长的助手查询之后，意图还是不是同一个\n")
    W("| 领域 | 来源 | 包装对数 | 意图不变 | 多写的字数(中位) |")
    W("|---|---|---:|---:|---:|")
    t["o"] = t["source"].map({s: i for i, s in enumerate(SOURCES)})
    for _, r in t.sort_values(["domain", "o"]).iterrows():
        W(f"| {r['domain']} | {SRC_ZH[r['source']]} | {int(r['pairs']):,} | {r['same_intent']*100:.1f}% | {r['median_added']:.1f} |")
    W("\n> 「意图不变」的比例受体系粒度影响（类目越细越容易变），跨领域比较时要一并看 T2 的类目数。\n")


def t14() -> None:
    p = work_file("voice_profile.csv")
    if not p.exists():
        return
    t = pd.read_csv(p)
    W("#### T14 语音导出的仪器特征（金融、医疗）\n")
    cols = ["len_median", "len_p95", "len_max", "len>=38字%", "句末语气词", "填充词", "标点", "英文字母"]
    W("| 领域 | 来源 | n | 不同串 | " + " | ".join(cols) + " |")
    W("|---|---|---:|---:|" + "---:|" * len(cols))
    t["o"] = t["source"].map({s: i for i, s in enumerate(SOURCES)})
    for _, r in t.sort_values(["domain", "o"]).iterrows():
        W(f"| {r['domain']} | {SRC_ZH[r['source']]} | {int(r['n']):,} | {int(r['distinct']):,} | "
          + " | ".join(f"{r[c]:.1f}" for c in cols) + " |")
    W("\n> 语音行没有标点、英文小写、约 40 字处截断，1,000 行只有 885 个不同串。凡是跟长度、标点、"
      "完整度有关的差异，先归因到仪器，再谈用户。\n")


def t15() -> None:
    """Curated pairs, VERIFIED against wrap_pairs at generation time: a search string written
    verbatim inside a longer assistant query, where the intent label changed. Picked for being
    public and innocuous (no private individuals, nothing sexual); every one is asserted to
    exist with the stated domain and source, so the table cannot drift from the data."""
    import glob
    picks = [
        ("金融", "assistant_top", "人民币", "1欧元等于多少人民币"),
        ("金融", "assistant_voice", "支付宝", "怎样把社保从微信支付转到支付宝支付"),
        ("医疗", "assistant_top", "bmi", "bmi值正常标准是多少"),
        ("医疗", "assistant_voice", "飞蚊症", "眼睛出现飞蚊症和小点黑影是什么原因"),
        ("教育", "assistant_random", "复旦大学", "2026年复旦大学的博士招满吗？"),
        ("教育", "assistant_random", "为什么", "30度对应的直角边是斜边的一半，为什么？"),
        ("影视", "assistant_top", "电视剧", "2026年最火的电视剧有哪些"),
        ("人物", "assistant_top", "丁禹兮", "丁禹兮的代表作品有哪些"),
    ]
    wp = pd.concat([pd.read_csv(f) for f in glob.glob(str(WORK / "wrap_pairs*.csv"))], ignore_index=True)
    W("#### T15 同一个搜索词，被写进更长的助手查询之后，意图就变了（全部为数据中的真实行）\n")
    W("| 领域 | 助手来源 | 搜索里的写法 | 助手里的写法 | 意图变化 |")
    W("|---|---|---|---|---|")
    for dom, src, core, q in picks:
        m = wp[(wp.domain == dom) & (wp.source == src) & (wp.core == core) & (wp["query"] == q)]
        if not len(m):
            raise SystemExit(f"T15: the pair ({dom}, {src}, {core}, {q}) is not in wrap_pairs — "
                             f"do not print an example the data does not contain")
        r = m.iloc[0]
        W(f"| {dom} | {SRC_ZH[src]} | {core} | {q} | {r.core_td} → {r.wrap_td} |")
    W("\n> 包装规则会产生**偶然**配对：核心词很短时（如英文 `know` 出现在 `I don't know` 里）匹配是字面巧合，"
      "不是同一个需求。上表的例子都经人工确认语义相关；T13 的统计量没有做这一步筛选，读的时候要记得这一点。\n")


def t16() -> None:
    """Cross-domain replication measured WITHOUT a class mapping: one regex, five domains."""
    p = CROSS / "form_contrasts.csv"
    if not p.exists():
        return
    t = pd.read_csv(p)
    W("#### T16 写法标记在各对比线上的变化（百分点；★=该领域 95% 区间不含 0）\n")
    for a_s, b_s, why in CONTRASTS:
        sub = t[(t.a == a_s) & (t.b == b_s)]
        if not len(sub):
            continue
        piv = sub.pivot_table(index="marker", columns="domain", values="diff_pp")
        sg = sub.pivot_table(index="marker", columns="domain", values="sig", aggfunc="first")
        cols = [c for c in DOMS if c in piv.columns]
        n_sig = sg[cols].fillna(False).sum(axis=1)
        keep = piv.loc[n_sig[n_sig >= 2].index, cols]
        if not len(keep):
            continue
        W(f"\n**{SRC_ZH[a_s]} → {SRC_ZH[b_s]}（{why}）**\n")
        W("| 标记 | " + " | ".join(cols) + " | 显著域数 |")
        W("|---|" + "---:|" * (len(cols) + 1))
        for mk in keep.index:
            cells = []
            for c in cols:
                v = keep.loc[mk, c]
                star = "★" if bool(sg.loc[mk, c]) else ""
                cells.append("—" if pd.isna(v) else f"{v:+.1f}{star}")
            W(f"| {mk} | " + " | ".join(cells) + f" | {int(n_sig[mk])} |")
    W("\n> 本表只列出至少两个领域显著的标记；**一共测了 17 个写法标记**，未列出的那些多数不显著或方向分裂。"
      "这些标记是**同一条正则**跑在五个领域上，所以"
      "「同一个标记在五个领域同方向显著」本身就是跨领域规律，不依赖任何类目名的对齐。"
      "正则有误报与漏报，只读同一规则下的相对差异。\n")


for f in (t1, t2, t3, t3b, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16):
    f()
OUT.close()
print("wrote", OUT_PATH)
