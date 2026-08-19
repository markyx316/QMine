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
