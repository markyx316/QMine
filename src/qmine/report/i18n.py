"""Report vocabulary, in the language the deliverable is read in.

Defaults to Chinese. This is not a localisation nicety: a `user_need` sentence
is simultaneously the annotation guideline, the acceptance criterion and the
downstream product requirement, and it can only do those jobs in the language
the team that owns the corpus actually works in.

Metric *names* stay in Latin script (silhouette, ARI, NMI) because that is how
they appear in the literature and in the code, and translating them would make
the report harder to check rather than easier.
"""

from __future__ import annotations

ZH: dict[str, str] = {
    # headings
    "exec_summary": "执行摘要",
    "research_path": "研究路径",
    "representation": "表征构建",
    "encoder_bakeoff": "底座 embedding 选型 (bake-off)",
    "sparse_block": "稀疏表征与 SVD 压缩",
    "alpha_sweep": "Hybrid 拼接与 α-sweep",
    "tuning": "调优过程",
    "algorithm_battery": "无监督算法选型 (battery)",
    "granularity": "粒度选择 (K 的三角验证)",
    "hierarchy": "两层层级构建与迭代精化",
    "naming": "盲评命名与树审计",
    "governance": "治理合并 (执行, 不是记录)",
    "panel": "统一度量面板",
    "deployment": "部署与全量应用",
    "leaf_catalogue": "完整命名档案: 逐叶定义",
    "failure_history": "被否决的尝试与失败历史",
    "decisions": "关键决策记录",
    "gates": "质量门",
    "limits": "这些数字不代表什么",
    "tree_listing": "完整家族→叶子清单",
    "query_comparison": "真实 Query 归类对照",
    "maintenance": "维护循环",
    # table headers
    "metric": "指标", "value": "数值", "candidate": "方案", "authority": "裁决权",
    "family": "家族", "leaf": "叶子", "size": "规模", "share": "占比",
    "name": "名称", "definition": "定义", "user_need": "user_need (一句话定义)",
    "coherence": "纯度", "rationale": "理由", "status": "状态",
    "role": "角色", "model": "模型", "cost": "成本",
    # metric labels
    "silhouette": "silhouette (轮廓系数)",
    "stability_ari": "重播稳定性 ARI",
    "template_fragmentation": "模板碎裂度",
    "heldout_reproduction": "held-out 结构复现",
    "nmi_reference": "NMI vs 参考体系",
    "ambiguous_rate": "margin 模糊行占比",
    "distill_accuracy": "蒸馏可学性",
    "n_clusters": "簇数",
    "kappa": "Cohen's κ",
    # authority
    "decisive": "主裁判", "advisory": "仅参考 (无投票权)", "diagnostic": "描述性",
    # recurring sentences
    "fairness_note": (
        "**公平性提示**: 碎裂度与家族数负相关 (家族越少越难碎) — 跨方案对比时必须同时报家族数, "
        "结论应使用「细而不碎」式的双条件表述。"
    ),
    "silhouette_no_vote": (
        "silhouette 衡量「簇内紧、簇间远」, 但**措辞相同的 query 天然最紧** — 用它选型会系统性偏向"
        "「模板孪生簇」(同一意图按句式劈成多个家族), 恰好与「家族可解释」的目标相反。"
        "因此本报告中 silhouette 全程**只报告、不投票**。"
    ),
    "distill_caveat": (
        "蒸馏分类器精度度量的是**可蒸馏性** — 即这套聚类标签能否被表征线性学出 — "
        "**不是**与人类判断的一致性。人类一致性需由金标 (Cohen's κ) 与对抗验证单独测量。"
    ),
    "blind_protocol": (
        "命名 agent **看不到任何既有标签**: 不给旧分类、不给自上而下的意图名、不给彼此的答案, "
        "只看成员样本卡片。锚定效应是真实的 — 见过既有体系的命名者会把簇「认领」到旧类目下, "
        "掩盖数据的真实形状。"
    ),
    "governance_executed": (
        "**审计处方已全部执行**, 而非仅写入建议章节。家族合并 = 改写「叶→家族」查找表 "
        "(叶分配与质心完全不动), 原家族列保留可追溯。"
    ),
    "no_data": "_(本次运行未产出该项)_",
}

EN: dict[str, str] = {
    "exec_summary": "Executive summary", "research_path": "Research path",
    "representation": "Representation", "encoder_bakeoff": "Encoder bake-off",
    "sparse_block": "Sparse block and SVD", "alpha_sweep": "Hybrid and the alpha sweep",
    "tuning": "Tuning",
    "algorithm_battery": "Algorithm battery", "granularity": "Granularity",
    "hierarchy": "Two-level hierarchy", "naming": "Blind naming and tree audit",
    "governance": "Governance (executed)", "panel": "Uniform metrics panel",
    "deployment": "Deployment", "leaf_catalogue": "Leaf catalogue",
    "failure_history": "What we rejected", "decisions": "Decisions", "gates": "Quality gates",
    "limits": "What these numbers do not mean", "tree_listing": "Full family/leaf tree",
    "query_comparison": "Real query comparison", "maintenance": "Maintenance loop",
    "metric": "metric", "value": "value", "candidate": "candidate", "authority": "authority",
    "family": "family", "leaf": "leaf", "size": "size", "share": "share",
    "name": "name", "definition": "definition", "user_need": "user_need",
    "coherence": "coherence", "rationale": "rationale", "status": "status",
    "role": "role", "model": "model", "cost": "cost",
    "silhouette": "silhouette", "stability_ari": "replay stability (ARI)",
    "template_fragmentation": "template fragmentation",
    "heldout_reproduction": "held-out reproduction", "nmi_reference": "NMI vs reference",
    "ambiguous_rate": "ambiguous rate", "distill_accuracy": "distillability",
    "n_clusters": "clusters", "kappa": "Cohen's kappa",
    "decisive": "decisive", "advisory": "advisory (no vote)", "diagnostic": "diagnostic",
    "fairness_note": (
        "**Fairness note**: fragmentation is negatively correlated with cluster count — "
        "always report cluster counts alongside it and phrase conclusions as two-condition "
        "statements (\"finer AND less fragmented\")."
    ),
    "silhouette_no_vote": (
        "Silhouette is maximised by phrasing-tight clusters, which is the failure this "
        "pipeline exists to detect. It is reported throughout and given no vote."
    ),
    "distill_caveat": (
        "Distillation accuracy measures how LEARNABLE the clusters are, not their agreement "
        "with human judgment."
    ),
    "blind_protocol": (
        "Naming agents saw no existing labels of any kind — not the taxonomy, not legacy "
        "categories, not each other's answers."
    ),
    "governance_executed": (
        "Every audit prescription was EXECUTED against the delivered data, as a leaf→family "
        "lookup remap with the pre-merge column retained."
    ),
    "no_data": "_(not produced in this run)_",
}


def vocab(language: str = "zh") -> dict[str, str]:
    return ZH if language == "zh" else EN


def stars(score: float | int | None, out_of: int = 5) -> str:
    """Coherence as ★ marks, matching the reference deliverables' convention."""
    if score is None:
        return ""
    n = max(0, min(int(round(float(score))), out_of))
    return "★" * n


def pct(x: float | None, digits: int = 1) -> str:
    return "—" if x is None else f"{x * 100:.{digits}f}%"


def num(x: float | None, digits: int = 4) -> str:
    if x is None:
        return "—"
    try:
        import math

        if math.isnan(float(x)):
            return "—"
    except (TypeError, ValueError):
        return "—"
    return f"{float(x):.{digits}g}"


#: The decision ledger is recorded in English so the artifacts stay stable across
#: report languages; the reader sees it in the deliverable's language. Only the
#: seven questions the pipeline actually asks need an entry — an unmapped question
#: falls through unchanged rather than being dropped, so adding a decision without
#: a translation degrades to English instead of vanishing.
DECISION_QUESTIONS_ZH = {
    "Which base encoder?": "选哪个底座 encoder?",
    "How much weight should phrasing get?": "措辞该占多大话语权 (α)?",
    "Which clustering algorithm?": "用哪个聚类算法?",
    "How many families?": "家族层切多少个 (K)?",
    "What is the intent taxonomy?": "意图体系应该长什么样?",
    "Which model family for the top-down classifier?": "自上而下分类器用哪类模型?",
    "Which L1 classes can the representation actually carry?": "表征实际撑得起哪些 L1 类目?",
}


def decision_question(text: str, language: str = "zh") -> str:
    """The decision's question in the report's language, or unchanged if unmapped."""
    return DECISION_QUESTIONS_ZH.get(text, text) if language == "zh" else text


#: Prose the pipeline authors in English — decision rationales and gate
#: remediations — rendered in the deliverable's language. This is the reasoning
#: content, so leaving it untranslated in a Chinese report defeats the point of
#: including it. Matching is on a distinctive PREFIX rather than the full string,
#: so small edits to the English tail do not silently drop a translation; a test
#: asserts that everything reaching a real report is covered.
PROSE_ZH: dict[str, str] = {
    "Open a new generation":
        "开一个新 generation 重新推导; **不要就地打补丁** — 被否决的产物本身也是证据。",
    "A partition that only exists when it can see every row":
        "只有在看得见全部数据时才存在的划分, 是对**这份样本**的描述, 而不是对现象的描述。"
        "请降低粒度, 或补充更多数据后重跑。",
    "A naming shard failed":
        "有命名分片执行失败。请在运行日志中查异常 — 未命名的叶子会一路带到交付表, "
        "看起来像「没有意图」而不是「没跑成」。",
    "Low coherence means clusters are carrying more than one intent":
        "内聚度偏低意味着簇里装了不止一个意图 — 这是**粒度问题, 不是命名问题**。"
        "应回到层级构建重切, 而不是换个名字了事。",
    "If only the seeded pre-screen finds risk content":
        "如果只有预置种子筛出了风险内容, 那么这个发现来自**你给的清单**, 而不是来自数据。"
        "需要一次不看清单的独立复核才能算数。",
    "Every 'we recommend X' in a report must have a matching executed change":
        "报告里每一句「建议合并 X」都必须对应一次**已执行**的改动。只提议不执行, "
        "等于把工作留给了下一个人, 而交付物看起来却像已经做完了。",
    "A minority language between 0.5% and 5%":
        "占比在 0.5%–5% 之间的少数语言是**最危险的区间**: 小到撑不起自己的簇, "
        "又大到足以污染别人的簇。需要分层抽样或单独的子意图处理。",
    "Coverage below the window means the fragmentation metric":
        "模板群覆盖率低于窗口, 意味着碎裂度指标建立在过少的行上 — 请继续挖掘模板群, "
        "或明确接受这个指标此次证据偏弱。",
    "The guide is ambiguous before a single gold row":
        "在还没有为任何一行金标付费之前, 指南就已经有歧义了。请修正上面这些易混类目对的"
        "定义与裁决规则, 然后重跑 2a — 手册明确要求此刻回炉, 而不是直接开标。",
    "Too few classes and a catch-all swells":
        "类目太少, 兜底类会膨胀到失去意义; 太多则标注一致性崩塌, 金标随之报废。"
        "裁决规则太少则裁判无据可引, 仲裁沦为口味之争。",
    "Increase gold_sample_size":
        "请增大 gold_sample_size, 或检查标注环节是否真的产出了标签。",
    "Low kappa means the guide is ambiguous":
        "κ 偏低说明**指南有歧义, 而不是标注员不用心**。请把裁判起草的规则并入指南后重新标注, "
        "再在此基础上训练任何模型。",
    "Annotator coverage collapsed":
        "标注覆盖率崩塌 — 在解读这个数字之前, 请先查运行日志里的服务商错误 (鉴权、限流、超时)。"
        "服务恢复后重跑本阶段。",
    "Replay stability only REJECTS here":
        "重播稳定性在此**只负责否决**: 它在本语料上的种子间标准差 (~0.10 ARI) 大于相邻 K "
        "之间的差距 (~0.05), 且曲线在网格下界之外仍在上升, 用它排序等于读噪声, 并会滑向"
        "退化的二分。K 由**与模板群的对齐度 (AMI)** 定位 —— 这是此处唯一具备双向惩罚、"
        "因而存在内部最优的指标。若多个 K 无法区分, 则报告整个并列集合并取其中最简单的一个。",
    "This phase does not choose the algorithm":
        "本阶段**不选择算法**: 交付的树始终由 KMeans 构建。这里跑的是一次**证伪检验** —— "
        "把结构上完全不同的算法送进同一套度量 harness, 问的是「这套结构是语料的性质, "
        "还是 KMeans『簇近似球形』这一假设的产物」。若某个替代算法明显更可复现, "
        "那是一个「家族层应按暂定读取」的警告, 而不是中途换算法的理由。",
    "Highest replay stability under an identical measurement harness":
        "在完全相同的度量 harness 下重播稳定性最高。L2 归一化后点落在单位球面上, "
        "余弦邻域与欧氏邻域一致, 簇近似各向同性 — 正中 KMeans 的假设。",
    "Highest replay stability on this corpus's own clustering task":
        "在**本语料自己的聚类任务**上重播稳定性最高 —— 而不是在公开检索榜单上。",
    "Synthesised from":
        "由多路独立研究综合而成; 评审员返回了若干条修订意见。",
    "Tree ensembles must reconstruct directional similarity":
        "树模型必须用轴对齐的切分去重建方向相似度, 并把有限的提升预算摊到很多类目上; "
        "线性头在这种几何下更契合。",
}


def prose(text: str, language: str = "zh") -> str:
    """Authored rationale/remediation prose in the report's language.

    Falls through to the original when unmapped, so a new string degrades to
    English rather than disappearing.
    """
    if language != "zh" or not text:
        return text
    for prefix, zh in PROSE_ZH.items():
        if text.startswith(prefix):
            return zh
    return text
