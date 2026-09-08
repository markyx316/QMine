"""The drift document: what changed between snapshots, and what it does not say.

Generated entirely from `drift_analysis.json`. Every number here is a lookup, not
a calculation and not a judgement, so the document says the same thing the
artifact does and a reader can check any line against it.

It is written to stand alone WITHOUT an agent narrative, because `mode="fast"`
disables every agent-written deliverable and the drift analysis has to survive
that. What an agent would add is interpretation; what this cannot do without one
is explain WHY a class moved — so it does not try, and says so.
"""

from __future__ import annotations

from typing import Any

from .i18n import prose


def _pct(x: float | None, nd: int = 2) -> str:
    return "—" if x is None else f"{x * 100:.{nd}f}%"


def _pp(x: float | None) -> str:
    return "—" if x is None else f"{x:+.2f}pp"


def _movers_table(rows: list[dict[str, Any]], limit: int) -> list[str]:
    out = ["| 类目 | 行数 | 行占比 | 流量占比 | Δ流量 | z(行占比) | 变化集中度 |",
           "|---|---:|---|---|---:|---:|---|"]
    for r in rows[:limit]:
        star = " \\*" if abs(r.get("z_row_share") or 0) >= 1.96 else ""
        out.append(
            f"| `{r['label']}` | {r['rows_a']:,} → {r['rows_b']:,} "
            f"| {_pct(r['row_share_a'])} → {_pct(r['row_share_b'])} "
            f"| {_pct(r['weight_share_a'])} → {_pct(r['weight_share_b'])} "
            f"| **{_pp(r['weight_share_delta_pp'])}** | {r['z_row_share']:+.1f}{star} "
            f"| {_conc(r.get('delta_concentration'))} |")
    return out


def _conc(c: dict[str, Any] | None) -> str:
    """One query, or many? The single most actionable column in the table.

    A class can move because one entity blew up, or because the behaviour
    broadened — and the product response is opposite. HHI over the per-query
    delta separates them: measured on 影视, the -13.6pp streaming decline spread
    over 6,201 distinct queries (HHI 0.004) while the +9.7pp live-TV rise had one
    query carrying 23% of it.
    """
    if not c or not c.get("n_distinct_queries"):
        return "—"
    hhi, top1, nq = c["hhi_of_delta"], c["top1_share_of_delta"], c["n_distinct_queries"]
    # NAME THE MEASUREMENT, NOT THE CONCLUSION. An earlier draft labelled the
    # concentrated bucket 「疑似单一事件」 and the first real case refuted it: the
    # +9.72pp 央视直播 class had top1=23%, but its top five deltas were all cctv5
    # phrasings — one ENTITY spread over many surface forms, not one event. HHI
    # measures concentration; whether that is an event, an entity or a campaign is
    # read off the query list beside it.
    tag = "**变化集中在少数 query**" if top1 >= 0.20 or hhi >= 0.10 else \
          ("略集中" if top1 >= 0.08 or hhi >= 0.03 else "变化分散在大量 query")
    return f"{tag}<br>HHI {hhi}｜top1 {_pct(top1)}｜{nq:,} 个不同 query"


def _delta_queries(rows: list[dict[str, Any]], limit: int) -> list[str]:
    """The queries that actually carry each move — the evidence behind the column.

    Without these the concentration number is unactionable: 「集中」 tells you to
    look, and this is the looking. On 影视 it turns "-13.6pp streaming" into five
    named 2025 dramas that simply aged out, and "+9.7pp live TV" into cctv5.
    """
    out: list[str] = []
    for r in rows[:limit]:
        qs = (r.get("delta_concentration") or {}).get("top_queries") or []
        if not qs:
            continue
        out.append(f"- `{r['label']}` （{_pp(r['weight_share_delta_pp'])}）："
                   + "；".join(f"{q['query']} ({q['delta']:+,.0f})" for q in qs[:5]))
    return out


#: THE VOCABULARY THAT DEPENDS ON WHAT THE TWO GROUPS DIFFER BY.
#:
#: `ops/drift.py` compares the composition of two groups and knows nothing about
#: time. This module did: it said 「不是趋势」, 「同月同日不等于季节可比」,
#: 「时段性事件」. Those sentences are true of two dates and FALSE of two sampling
#: strata of one period, and a reader who believes them concludes that behaviour
#: changed over time when nothing changed at all.
#:
#: `time` reproduces the document exactly as it shipped before this table existed
#: — pinned by `test_the_time_axis_document_is_unchanged_by_the_axis_parameter`.
_AXIS: dict[str, dict[str, str]] = {
    "time": {
        "title": "快照对比 · 漂移分析",
        "subtitle": "同一套标签体系下，两个时间点的差异",
        "unit": "快照",
        "both": "两期",
        "inventory_h": "两个快照本身",
        "shared_h": "有多少「还是同一批 query」",
        "stable_h": "两期都存在的类目",
        "emergent_h": "新出现的类目",
        "receded_h": "消失的类目",
        "thin_h": "两期都太小、不做比较",
        "purity_h": "这两期真的被放在同一个框架里比较了吗？",
        "why_pool": (
            "> **为什么必须放在同一次运行里做。** 本流水线跑两次会给出两套互不相干的标签："
            "金融语料的 2025-07 与 2026-07 分两次跑（`fin01` / `fin02`），得到 **20 类**与 **19 类**，"
            "两边**共享的类目代码为 0 个** —— `LOOKUP_FX_RATE` 与 `FX_RATE_LOOKUP` 说的是同一件事，"
            "却无法对齐。架构师"
            "每次都会重新发明命名，聚类树也是重新拟合的。所以「跑两次再对比」比较的是噪声。"
            "本报告的两个时间段由**同一套类目体系、同一棵树**打标，这正是排除「标签变了」"
            "这一解释、只留下「查询变了」的原因。"),
        "why_share": (
            "> **占比一律是「快照内占比」，不是原始计数。** 两个快照的总流量并不相同，"
            "直接比原始值会把每一类都显示成下降。"),
        "shared_note": (
            "> 这个数字决定了下面所有占比变化**最多能解释到什么程度**。共有部分承载的流量越高，"
            "漂移越是「既有需求之间的此消彼长」；越低，则越多是「问的东西整个换了一批」，"
            "此时逐条 query 的对比没有意义，只有意图层面的对比有。"),
        "emergent_why": (
            "在前一个快照里几乎不存在。**两种读法数据都支持**：要么是真的出现了新需求，"
            "要么是这类需求一直都在、只是之前太稀疏、直到合并语料后才够量被单独命名。"
            "本报告无法区分这两者。"),
        "receded_why": "在后一个快照里几乎不存在。同样有「需求消失」和「稀疏到不再成类」两种读法。",
        "purity_note": ("> ⚠ 这些组**没有被比较，而是被分开了**。少数几个通常是真实的时段性事件；"
                        "如果很多，说明「同一框架」这个前提不成立，上面所有占比都要重读。逐组如下："),
    },
    "stratum": {
        # DELIBERATELY FRAME-AGNOSTIC. An earlier draft said 「头部按流量取 TopN，
        # 尾部是随机抽样」 — the AI-assistant corpus's own design, written into the
        # shipped vocabulary. Any other stratum pair (two sampling frames, two
        # collection methods, two devices) would then be described wrongly by a
        # document that sounds authoritative. What is true of EVERY stratum
        # comparison is that the groups come from one period and differ by how
        # they were drawn; what each frame IS belongs to the run, not to this
        # table.
        "title": "分层对比 · 两种抽样口径的结构差异",
        "subtitle": "同一套标签体系下，两种抽样口径的差异",
        "unit": "分层",
        "both": "两层",
        "inventory_h": "两个分层本身",
        "shared_h": "两层有多少「还是同一批 query」",
        "stable_h": "两层都存在的类目",
        "emergent_h": "只出现在后一层的类目",
        "receded_h": "只出现在前一层的类目",
        "thin_h": "两层都太小、不做比较",
        "purity_h": "这两层真的被放在同一个框架里比较了吗？",
        "why_pool": (
            "> **为什么必须放在同一次运行里做。** 本流水线跑两次会给出两套互不相干的标签："
            "同一份金融语料分两次跑（`fin01` / `fin02`）得到 **20 类**与 **19 类**，两边"
            "**共享的类目代码为 0 个** —— `LOOKUP_FX_RATE` 与 `FX_RATE_LOOKUP` 说的是同一件事，"
            "却无法对齐。架构师每次都会重新发明命名，聚类树也是重新拟合的。所以"
            "「两份数据各跑一次再对比」比较的是噪声。本报告的两个分层由**同一套类目体系、"
            "同一棵树**打标，这正是排除「标签变了」这一解释、只留下「两种抽样口径看到的"
            "东西不同」的原因。"),
        "why_share": (
            "> **占比一律是「层内占比」，不是原始计数。** 两个分层的规模与总流量并不相同"
            "（这正是抽样口径不同的直接后果），直接比原始值会把差异读成量级差异。"),
        "shared_note": (
            "> 这个数字量的是**两种抽样口径的重叠程度**，不是时间上的延续性。重叠越低，"
            "说明两层看到的根本是不同的 query 集合，此时逐条 query 的对比没有意义，"
            "只有意图层面的对比有。"),
        "emergent_why": (
            "在前一层里几乎不存在。**这不是「新需求」**：两层来自同一时间段，差别只在抽样"
            "口径。正确的读法是「这种意图只在这一层的量级上才够量成类」。"),
        "receded_why": (
            "在后一层里几乎不存在。同样**不是「需求消失」**，而是这种意图在另一层的量级上"
            "不成类。"),
        "purity_note": ("> ⚠ 这些组**没有被比较，而是被分开了**。在分层对比里这经常发生："
                        "只在某一层达到成类量级的意图本来就会整组落在一边。数量多说明两层的"
                        "可比范围窄，上面所有占比都要在这个前提下读。逐组如下："),
    },
}


def _axis(deps: Any) -> dict[str, str]:
    """Which vocabulary this run's comparison needs.

    Degrades to `time` rather than raising: an unreadable config must not kill a
    document whose every NUMBER is still correct.
    """
    try:
        key = str(getattr(deps.cfg.data, "comparison_axis", "time"))
    except Exception:  # noqa: BLE001
        key = "time"
    return _AXIS.get(key, _AXIS["time"])


def _caveats(ax: dict[str, str]) -> list[str]:
    """What the comparison cannot support — and this DEPENDS ON THE AXIS.

    Two of the time-axis bullets are not merely inapplicable to a stratum
    comparison, they are INVERTED. 「两期的抽样方式必须一致」 warns that a
    difference in sampling would masquerade as a real change; between a top-N-by-PV
    head and a random tail the sampling difference IS the independent variable, and
    a reader who applies the time-axis caveat concludes the whole document is
    confounded when it is measuring exactly what it set out to. 「不是趋势」 and
    「同月同日不等于季节可比」 are about calendars that are not involved at all.
    """
    shared = [
        "- **不能替代分段核对。** 整体占比上升，完全可能每个子段都在下降（辛普森悖论）。"
        "要下结论前，请按你关心的维度再拆一次。",
    ]
    if ax is _AXIS["stratum"]:
        return [
            "- **不能告诉你原因。** 它能说某一类在哪一层占比更高，不能说为什么。任何因果"
            "解释都需要本语料之外的信息（产品入口、推荐位、默认按钮），而那些信息不在这里。",
        ] + shared + [
            "- **不是时间变化。** 两层来自同一时间段，差别只在抽样口径（一层按流量取 TopN，"
            "一层随机抽）。这里任何一个差异都**不能**读成「用户行为变了」。",
            "- **抽样口径不同正是本报告的自变量。** 时间对比里「两边抽样方式必须一致」是前提，"
            "在这里恰恰相反：差异来自口径本身。要问的不是「是不是口径造成的」（是的），"
            "而是「口径揭示了什么结构」——头部与长尾是不是同一批需求。",
            "- **头部的量级不代表需求的重要性。** TopN 是按流量截断的，一次点击、一个"
            "推荐位、一个默认按钮都会把某条 query 顶进头部，而它未必对应一次真实的表达。",
            "- **不能推回总体。** 每个类目的 TopN 有各自的流量门槛，两层也不构成对总体的"
            "概率抽样，所以这里的占比**不能**换算成「全量里有多少」。",
            "- **不能发现「同一句 query 的含义变了」。** 两层由同一位架构师、同一套规则标注，"
            "这排除了「标签变了」这一解释，代价是标签体系内看不见的差别也同样看不到。",
        ]
    return [
        "- **不能告诉你原因。** 它能说某一类涨了或跌了，不能说为什么。任何因果解释都需要"
        "本语料之外的信息（活动、改版、事件），而那些信息不在这里。",
    ] + shared + [
        "- **不是趋势。** 这是两个时间点，不是一条曲线。两点之间无法区分「持续变化」与"
        "「其中一天恰好特殊」。",
        "- **不能发现「同一句 query 的含义变了」。** 本报告测的是**固定标签体系下各类占比的"
        "变化**；两期是由同一位架构师、同一套规则标注的，所以「taxonomy 变了」被排除了，"
        "代价是「query 没变、但它现在指的是另一件事」也同样看不到。这是为了可比性而付的价。",
        "- **同月同日不等于季节可比。** 两期相隔一年、日期相同，这是对季节性最朴素的控制，"
        "但农历、考试与放榜、赛事、节日档期都会在年与年之间移动。若某一类的变化正好落在"
        "这类日程上，请先核对日程再下结论。",
        "- **两期的抽样方式必须一致。** 如果两份数据的抽取口径不同，这里的每一个差异都"
        "可能只是口径差异。这一点数据本身无法验证。",
    ]


def build(state: Any, deps: Any) -> str:
    d = deps.load("drift_analysis")
    ax = _axis(deps)
    tags = d.get("snapshots") or []
    a, b = (tags + ["?", "?"])[:2]
    L: list[str] = [
        f"# {ax['title']}",
        f"## {a} → {b}：{ax['subtitle']}",
        "",
        f"**运行**: `{state.get('run_id')}` · **领域**: `{deps.cfg.domain.key}`",
        "",
        ax["why_pool"],
        "",
        ax["why_share"],
        "",
        f"## 1. {ax['inventory_h']}",
        "",
        f"| {ax['unit']} | 行数 | 去重 query | 总流量 |",
        "|---|---:|---:|---:|",
    ]
    for r in d.get("inventory", []):
        L.append(f"| `{r['snapshot']}` | {r['rows']:,} | {r['distinct_queries']:,} "
                 f"| {r['weight_total']:,.0f} |")

    ch = d.get("query_churn") or {}
    if ch.get("comparable"):
        L += ["", f"### 1.1 {ax['shared_h']}",
              "",
              f"- 两个{ax['unit']}共有 **{ch['shared']:,}** 条相同 query，占并集的 **{ch['jaccard']:.1%}**",
              f"- 这批共有 query 承载了 {a} 的 **{_pct(ch.get('shared_weight_share_a'), 1)}** 流量、"
              f"{b} 的 **{_pct(ch.get('shared_weight_share_b'), 1)}**",
              "",
              ax["shared_note"]]

    for axis, title in (("td_l1", "自上而下意图 (td_l1)"),
                        ("bu_family_final", "自下而上家族"),
                        ("bu_leaf_name", "自下而上叶")):
        dd = (d.get("by_label") or {}).get(axis)
        if not dd or not dd.get("comparable"):
            continue
        L += ["", f"## 2. {title}", "",
              f"- 共 **{dd['n_classes']}** 类；整体差异 Cramér's V = **{dd['cramers_v']}** "
              f"(0={ax['both']}分布相同，1=完全不同)",
              f"- **总变差 = {_pct(dd.get('total_variation_weight', 0))}（按流量）／"
              f"{_pct(dd.get('total_variation_rows', 0))}（按行）** —— 读作：要把 {b} 的分布"
              f"变回 {a}，需要重新分配这么多比例的流量。它是这一对{ax['unit']}的**幅度锚点**，"
              "不能拿去和别的语料比大小。",
              "- 下表按**流量占比变化**排序，不按 z 排序。",
              ""]
        if dd.get("stable"):
            L += [f"### 2.1 {ax['stable_h']}", ""] + _movers_table(dd["stable"], 12)
            ev = _delta_queries(dd["stable"], 6)
            if ev:
                L += ["", "**变化由哪些 query 承担**（各类中 |Δ流量| 最大的 5 条，"
                          "带符号；这是「集中度」一列的证据，也是判断「事件 / 某个实体 / "
                          "面上变化」的依据——报告本身不做这个判断）：", ""] + ev
            L += ["",
                  f"> `*` = 行占比的 |z| ≥ 1.96。共 **{dd['n_comparisons']}** 个比较，"
                  f"按 5% 水平约有 **{dd['n_comparisons'] / 20:.0f}** 个会偶然显著，"
                  f"所以**请按 Δ流量的大小读，不要按 z 读**。流量占比不做检验：流量是总体，"
                  "不是抽样，对它做显著性检验没有意义。"]
        for key, hd, why in (
            ("emergent", f"### 2.2 {ax['emergent_h']}", ax["emergent_why"]),
            ("receded", f"### 2.3 {ax['receded_h']}", ax["receded_why"]),
        ):
            rows = dd.get(key) or []
            if rows:
                L += ["", hd, "", f"> {why}", ""] + _movers_table(rows, 8)
        thin = dd.get("too_thin_to_compare") or []
        if thin:
            L += ["", f"### 2.4 {ax['thin_h']}",
                  "", f"共 {len(thin)} 类，{ax['both']}行数都不足 30。占比在这种量级上会因几行而剧烈摆动，"
                      "列出但不参与排序：",
                  "", "、".join(f"`{r['label']}`" for r in thin[:20])]

        pur = (d.get("purity") or {}).get(axis) or {}
        if pur.get("checked"):
            n = pur.get("n_single_snapshot", 0)
            L += ["", f"### 2.5 {ax['purity_h']}（{title}）", "",
                  f"- {pur['n_groups']} 个分组中，**{n}** 个几乎完全落在某一个{ax['unit']}里",
                  f"- 各组的 `{a}` 占比：最小 {pur['share_min']}，中位 {pur['share_median']}，"
                  f"最大 {pur['share_max']}"]
            if n:
                L += ["",
                      ax["purity_note"],
                      ""]
                L += [f"- `{g['label']}` — {a} 占比 {g.get('share_of_' + str(a))}，{g['rows']:,} 行"
                      for g in pur.get("single_snapshot", [])[:10]]
            else:
                L.append(f"> 每一组都同时包含两个{ax['unit']}的行，共享框架成立。")

    L += ["", "---", "", "## 3. 这份报告不能告诉你什么", ""] + _caveats(ax) + [
          "",
          "**原始数据**: `drift_analysis.json`（本报告每个数字都可在其中查到）、"
          f"`labels_full.csv`（逐行标签，含{ax['unit']}列）。"]
    return prose("\n".join(L))
