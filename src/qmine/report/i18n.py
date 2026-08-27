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

import re

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
    # WAS "**审计处方已全部执行**" — false, and false in the direction that
    # flatters: live40 executed 8 of 17 and declined 9, with the table listing
    # them one line below. The guaranteed property is not that everything ran, it
    # is that nothing was left hanging — `assert_all_settled` fails the run on
    # anything still `proposed`. State THAT, because it is what the gate checks
    # and it stays true whatever the split turns out to be.
    "governance_executed": (
        "**每一条审计处方都已了结 —— 要么执行, 要么写明拒绝理由**, 而不是仅写入"
        "建议章节。逐条的处置与理由见下表。家族合并 = 改写「叶→家族」查找表 "
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
    # --- p2b_rules_match_their_evidence, authored in `graph/nodes/topdown.py` --
    # The prefix must stay free of interpolated numbers: `prose()` matches a
    # LITERAL PREFIX, so a count spliced into the front means the mapping never
    # fires and the sentence ships in English.
    "every boundary's stated discriminator actually divides its adjudicated rows":
        "每条边界上, 规则写明的判别词都确实把该边界的裁决行分开了",
    "the discriminator these rules name does not divide this boundary at all":
        "有边界上规则写明的判别词一行都没分开 —— 判据落在全部行的同一侧, 等于没有判据",
    "The words these rules name as the test do not appear in the rows the":
        "这些规则写明的判别词, 在裁判实际裁决过的行里一次都没出现 —— "
        "所以无论是什么决定了这条边界, 都不是规则声称的那个判据, "
        "标注者照字面执行只会得到「全部归同一类」。请给这条边界一个可观察的判据, "
        "或如实记下它靠人工判断。**不要删规则** —— 删掉指引会让边界更没人管, "
        "曾有一次按措辞相似度过滤, 41 条规则被砍掉 32 条。"
        "另注: 本门刻意**不**统计规则朝向。裁判只在它认为指南失效处起草规则, "
        "该信号被起草率污染, 在没有缺陷的指南上同样会报警。",
    # --- algorithm battery verdicts, authored in `ops/battery.py` ----------
    # Interpolated straight into a Chinese figure title, so they shipped as
    # English inside an otherwise-Chinese chart.
    "lowest template_fragmentation within a tie-band, broken on":
        "在容差带以内取碎裂度最低者视为并列, 再以重播稳定性最高者胜出",
    "configured relative band, NOT measured":
        "沿用配置的 5% 相对容差带, **不是实测值** —— 本次 alpha 扫描并非单调, "
        "其起伏本身带有真实信号, 因此无法用曲线粗糙度估计噪声",
    "Alpha is not chosen by taking the lowest fragmentation outright":
        "**alpha 不是直接取碎裂度最低的那个。** 碎裂度差异落在容差带以内时视为并列, "
        "再用**重播稳定性**破并列 —— 稳定性是两者中更扎实的测量。因此当选的 alpha 是"
        "「碎裂度实质并列者之中最可复现的那个」, 通常**两头都不占**: 既不是碎裂度最低的, "
        "也不是稳定性最高的。具体数值见下方证据行。措辞块以 **alpha 的平方**进入余弦, "
        "所以较小的 alpha 是破并列用的权重, 而不是与语义平起平坐的信号。",
    "not a selection - the tree is built with KMeans regardless":
        "这不是一次「选择」—— 交付的树始终由 KMeans 构建。本图跑其他算法, "
        "是为了回答另一个问题: **换一种算法, 这个结构还在不在?**",
    # --- uniform-panel footnotes -------------------------------------------
    # Authored in `ops/panel.py`, which has no language. They carry the three
    # fairness caveats a reader needs in order to not over-read the table, and
    # on a `report_language: zh` run they shipped in English.
    "Every number in this table was produced by the same code":
        "本表中每一个数字都由**同一份代码**、以**同一个随机种子**产生。"
        "需要子样的指标用同一个子样本, 其余指标跑全量 —— **每个指标各自带着自己的 n**, "
        "比较两个数之前必须先看 n。从不同阶段各引一个 silhouette 放在一起, "
        "比较的是两次碰巧同名的不同测量。",
    "Silhouette is ADVISORY":
        "**silhouette 只报告, 不投票。** 它在「簇内紧凑」时取到最大值, 而**措辞相同的 query 天然最紧凑** —— "
        "这恰恰就是决定性指标要去发现的失败模式: 同一个意图被按句式劈成多个「模板孪生」家族。"
        "因此它在本表中列出仅为完整起见, **不具备任何裁决权**。",
    "Template fragmentation is POSITIVELY correlated with cluster count":
        "**模板碎裂度与簇数正相关**: 簇越多, 可供碎裂的去处也越多 "
        "(本次面板实测 Pearson +0.90)。必须与 `n_clusters` 并排读, "
        "并且结论要写成**双条件**句式 (「更细**并且**更不碎」), 绝不可只比碎裂度本身。",
    "Distillation accuracy measures learnability":
        "**蒸馏准确率量的是簇标签的可学习性, 不是与人类判断的一致性。** "
        "后者需要金标准 (第 2b 阶段) 与对抗验证 (第 2d 阶段) 才能回答 —— "
        "一个完全自洽但与人类理解无关的划分, 同样可以拿到很高的蒸馏准确率。",
    "Open a new generation":
        "开一个新 generation 重新推导; **不要就地打补丁** — 被否决的产物本身也是证据。",
    "A partition that only exists when it can see every row":
        "只有在看得见全部数据时才存在的划分, 是对**这份样本**的描述, 而不是对现象的描述。"
        "请降低粒度, 或补充更多数据后重跑。",
    "A naming shard failed":
        "有命名分片执行失败。请在运行日志中查异常 — 未命名的叶子会一路带到交付表, "
        "看起来像「没有意图」而不是「没跑成」。",
    "Inter-annotator agreement has reached this annotator's own self-consistency":
        "标注者之间的一致性已经达到该标注者**自己与自己**的一致性上限, 再修指南也提不上去 —— "
        "剩余的分歧是标注者噪声, 不是指南歧义。要再高只能换更强的模型或人工标注; "
        "否则请把自一致性 kappa 当作本语料的诚实天花板, 并据此解读下游每一个数字。",
    "Low coherence means those clusters are carrying more than one intent":
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
    "Highest replay stability on this corpus's own clustering task":
        "在**本语料自己的聚类任务**上重播稳定性最高 —— 而不是在公开检索榜单上。",
    "Synthesised from":
        "由多路独立研究综合而成; 评审员返回了若干条修订意见。",
    "Tree ensembles must reconstruct directional similarity":
        "树模型必须用轴对齐的切分去重建方向相似度, 并把有限的提升预算摊到很多类目上; "
        "线性头在这种几何下更契合。",
    # These two reached live40's Chinese reports untranslated. The coverage test
    # that exists to prevent exactly that runs on the OFFLINE fixture, which never
    # takes the p2e audit branch or the HDBSCAN screen — so it passed while both
    # shipped. The test now reads the mapping itself; see its docstring.
    "Measured by k-nearest-neighbour label agreement per class":
        "按类目逐个测量 **k 近邻标签一致度**: 取该类每条查询在表征空间中的最近邻, "
        "看邻居是否也属于同一类。若一个类目的邻域并不共享它的标签, 那它就不是表征空间里的"
        "一块**区域**, 而是一条只能靠规则判定的界线 —— 再多的标注数据也不会让它变成区域。"
        "这类类目被判为「规则依赖」, 交给规则层而不是几何层去承担。",
    # Found by turning the English detector up: these three reached live40's
    # Chinese deliverables 13 times between them. All were already routed through
    # prose() — they simply had no mapping, which is why the fix is a mapping.
    "A CONFIRMED finding is an assertion that failed against the":
        "**CONFIRMED 表示一条断言在产物上核验失败了, 不是某个 agent 的意见** —— "
        "去读它的 check, 修对应的阶段; 断言重新成立时, 这条发现会自行关闭。"
        "未核验的那类会给出一个 artifact key: 自己去读, 然后判断。"
        "无论哪种, 它都已经进入本次运行的发现台账, 并会延续到下一个 generation。"
        "**不要为了让这道门通过而去关掉 observer。**",
    "agents were instructed to PROVE each label wrong":
        "让 agent 逐条去**证伪**每一个标签; 估计值是「攻击之后仍然站得住」的标签所占比例。"
        "这不是准确率的无偏估计 —— 它衡量的是标签在针对性质疑下的存活率。",
    "Do NOT auto-apply these":
        "**不要自动套用这些标记。** 在模板化语料里, 大多数标记是表征造成的假象 —— "
        "措辞相近的查询会挨在一起, 与它们的含义无关。请人工复核后再作处置。",
    "HDBSCAN is screened by":
        "HDBSCAN 只按 (噪声率升序, 簇数降序) **筛选出来供人工查看**, 绝不由某个合成分数"
        "自动选中。它**仅作诊断** —— 下游没有任何环节使用它。第 12 阶段的新颖内容哨兵用的"
        "是「到最近叶质心的余弦距离取最低 1%」, 与 HDBSCAN 无关。",
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


def looks_like_english_prose(line: str, *, min_words: int = 8) -> bool:
    """True for a line of running English inside a Chinese document.

    The canonical detector, shared by the runtime narrator and the test that
    guards the scripted reports, so the two cannot drift apart.

    An earlier version asked for three consecutive >=6-letter lowercase words.
    Real English almost never has that — function words ("the", "of", "is",
    "not") break every run — so it matched NEITHER of the two English paragraphs
    that shipped in live40's Chinese deliverables, and would not have matched
    them on a live report either. What actually identifies English prose in a
    Chinese document is the absence of CJK, so that is what this asks. Code
    spans, links, identifiers and table rows are excluded first, because a
    Chinese report legitimately carries all of them.
    """
    raw = line.strip()
    if not raw or raw.startswith(("|", "#", "![", "```", "---")):
        return False
    body = re.sub(r"`[^`]*`", " ", raw)
    body = re.sub(r"!?\[[^\]]*\]\([^)]*\)", " ", body)
    body = re.sub(r"^[>*\-\d.\s]+", " ", body)
    if re.search(r"[一-鿿]", body):
        return False
    return len(re.findall(r"[A-Za-z][A-Za-z'-]+", body)) >= min_words
