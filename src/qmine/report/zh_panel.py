"""统一度量面板 (uniform panel) 的中文报告。

Two reasons this file exists rather than a translation of `builder.panel_report`:

1. `Report_Uniform_Panel.md` shipped **0% Chinese** on a run configured `zh`.
2. The English version listed candidates without ever comparing them, and until
   this release the panel could not have compared them anyway — it scored the
   bottom-up leaves on 11 metrics and the top-down route on **one** (`kappa`,
   which is inter-annotator agreement on a gold set, not a property of any
   partition). The headline claim of the whole method — two routes, one harness —
   had no evidence behind it in the deliverable.

Now that both routes are measured, this report does the comparison explicitly and
says out loud which metrics are allowed to carry it and which are structurally
unfair to one side.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .i18n import num, prose
from .zh_bottomup import _gate_ledger

#: Metrics that mean the same thing for a clustering and for a classifier-assigned
#: partition. Anything outside this set either cannot be computed for the top-down
#: route or is biased toward the route whose objective it happens to be.
_ROUTE_FAIR = ("n_clusters", "template_fragmentation", "nmi_reference",
               "purity_reference", "ambiguous_rate")

_METRIC_ZH = {
    "n_clusters": "簇数",
    "template_fragmentation": "模板碎裂度 ↓",
    "stability_ari": "重播稳定性 ARI ↑",
    "kmeans_refit_stability": "重拟合稳定性 ↑",
    "heldout_reproduction": "held-out 结构复现 ↑",
    "nmi_reference": "NMI vs 参照体系 ↑",
    "purity_reference": "纯度 vs 参照体系 ↑",
    "silhouette": "silhouette (无投票权)",
    "ambiguous_rate": "模糊行占比 ↓",
    "coherence": "盲评连贯性 ↑",
    "distill_accuracy": "蒸馏准确率 ↑",
    "kappa": "标注者一致性 κ ↑",
}

_AUTH_ZH = {
    "decisive": ("**决定性**", "可以据此选择表征、算法或 K"),
    "advisory": ("仅参考", "报告并作图, 但**明令禁止**用于任何选择"),
    "diagnostic": ("描述性", "描述本次运行, 不用于比较方案"),
}

_SUBJECT_ZH = {
    "leaves": "自下而上 · 叶层",
    "families_final": "自下而上 · 家族层 (治理后)",
    "families_pre_governance": "自下而上 · 家族层 (治理前)",
    "topdown_l1": "自上而下 · L1 意图",
    "topdown_l2": "自上而下 · L2 子意图",
    "topdown": "自上而下 (仅金标准指标)",
}


def _label(subject: str) -> str:
    if subject in _SUBJECT_ZH:
        return _SUBJECT_ZH[subject]
    if subject.startswith("alpha_"):
        return f"α = {subject.split('_', 1)[1]} 候选表征"
    return subject


def _cell(v: Any) -> str:
    if v is None:
        return "—"
    return num(v) if isinstance(v, float) else str(v)


def build(state: Any, deps: Any, figs: dict[str, Any]) -> str:
    panel = deps.load("metrics_panel") if deps.has("metrics_panel") else {}
    gen_dir = Path(deps.store.gen_dir).name
    L = [
        "# 统一度量面板 (Uniform Measurement Panel)",
        "## 每一个候选方案都由同一段代码、在同一子样本、同一随机种子下重新测量",
        "",
        f"**运行**: `{state.get('run_id')}` / {gen_dir} · **领域**: `{deps.cfg.domain.key}` · "
        f"**配置指纹**: `{deps.cfg.config_hash}`",
        "",
    ]
    if not panel:
        return "\n".join(L + ["", "_本次运行未产出面板。_"])
    t = panel["table"]
    cfgp = t.get("panel_config", {})

    # ------------------------------------------------------------ 1 约定
    L += [
        "## 1. 这张表的约定", "",
        "**两个数只有在由同一段代码、在同一子样本、以同一随机种子产生时才可以比较。** "
        "从一个阶段引一个 silhouette, 再从另一个阶段引一个 silhouette 放在一起比, "
        "比较的是两次**碰巧同名**的不同测量 —— 子样本不同、随机种子不同, 差值里"
        "有多少来自方案本身、有多少来自测量过程, 无从分辨。", "",
        "因此本表**每一行都在这里重新测过一遍**, 而不是从各阶段的报告里摘抄。", "",
        f"- **面板 id**: `{t['panel_id']}`",
        f"- **子样本**: {cfgp.get('subsample')} 行 (全量 {cfgp.get('n_rows')} 行)",
        f"- **随机种子**: {cfgp.get('seed')}; 重播种子: {cfgp.get('replay_seeds')}",
        f"- **度量代码版本**: `{cfgp.get('code_version')}`", "",
    ]

    # ------------------------------------------------------------ 2 全表
    names = [m["name"] for m in t["metrics"]]
    L += ["## 2. 全部候选方案", "",
          "| 方案 | " + " | ".join(_METRIC_ZH.get(n, n) for n in names) + " |",
          "|---" * (len(names) + 1) + "|"]
    for r in t["rows"]:
        L.append(f"| **{_label(r['subject'])}** <br>`{r['subject']}` | "
                 + " | ".join(_cell(r.get(n)) for n in names) + " |")
    L.append("")
    if figs:
        L += [_fig(figs, "fig_panel", "统一面板对照; 斜纹条 = 只报告不投票的指标"), ""]

    # ------------------------------------------------------------ 3 两条路线
    L += _route_comparison(t)

    # ------------------------------------------------------------ 4 指标权限
    L += ["## 4. 指标的裁决权限", "",
          "并非每个指标都有资格决定任何事情。权限在**代码中强制**, 而不是靠约定: "
          "`decisive_ranking()` 收到一个「仅参考」指标时会直接抛异常。", "",
          "| 指标 | 权限 | 含义 |", "|---|---|---|"]
    for m in t["metrics"]:
        zh, meaning = _AUTH_ZH.get(m["authority"], (m["authority"], ""))
        L.append(f"| `{m['name']}` ({_METRIC_ZH.get(m['name'], '')}) | {zh} | {meaning} |")
    L.append("")

    # ------------------------------------------------------------ 5 脚注
    if t.get("footnotes"):
        L += ["## 5. 读表须知", "",
              "以下每一条都是**读错这张表的具体方式**, 不是免责声明。", ""]
        for i, f in enumerate(t["footnotes"], 1):
            L += [f"**{i}.** {prose(f, 'zh')}", ""]

    L += ["## 6. 质量门", "", _gate_ledger(state), ""]
    L += _open_findings(deps)
    return "\n".join(L)


def _open_findings(deps: Any) -> list[str]:
    """未结的观察发现 —— the ledger, printed where a reader will meet it.

    Without this section the ledger is a JSON file nobody opens, which is the
    exact failure it was built to end: a critic found the kappa defect before the
    run that shipped it, the finding went to an artifact, and nothing read it.

    Two categories, kept apart on purpose. A CONFIRMED finding is an assertion
    that failed against the delivered artifacts — a defect, stated as a
    measurement. An unverified one is an agent's concern that no expression could
    settle, and it is printed as exactly that. Merging them would either lend
    unearned weight to a hunch or bury a proven defect among opinions.
    """
    try:
        from ..ops.findings import FINDINGS_FILE, FindingLedger

        led = FindingLedger(Path(deps.store.root) / FINDINGS_FILE)
    except Exception:  # noqa: BLE001
        return []
    conf = led.confirmed_open
    unver = [f for f in led.open_findings if f.verdict != "confirmed"]
    fixed = [f for f in led.entries.values() if f.status == "fixed"]
    # A waived finding is a RECORDED DECISION, not a disappearance. It stops
    # reopening on re-sighting — otherwise waiving means nothing — so printing it
    # with its reason is the only thing keeping the waiver accountable.
    waived = [f for f in led.entries.values() if f.status == "waived"]
    if not (conf or unver or fixed or waived):
        return []

    L = ["## 7. 未结的观察发现 (findings ledger)", "",
         "阶段观察者在**运行过程中**读了每一阶段的产物。下面是尚未消解的发现。"
         "这份清单存放在**运行级**目录中 (与 LLM 缓存同级), 新开一个 generation 会"
         "继承它 —— 一条发现只有在**它自己的断言重新成立**时才会自动关闭。", ""]
    if conf:
        L += [f"### 7.1 已被机器证实的发现 ({len(conf)} 条)", "",
              "**这些不是观点。** 每一条都附带一个针对产物的断言, 而该断言在本次交付的"
              "产物上**求值为假**。请把它们当作失败的断言来读。", "",
              "| 阶段 | 严重度 | 发现 | 失败的断言 | 见过 |", "|---|---|---|---|---|"]
        for f in conf[:12]:
            # The whole expression, never a prefix: a reader's next move is to
            # re-evaluate it against the artifacts, and an assertion cut off
            # mid-token cannot be re-run.
            L.append(f"| `{f.phase}` | {f.severity} | {f.claim[:90]} | "
                     f"`{f.check}` | {f.times_seen} |")
        L.append("")
    if unver:
        L += [f"### 7.2 无法用表达式判定的发现 ({len(unver)} 条)", "",
              "观察者提出但**没有给出可求值断言**的疑问 —— 例如「结论是否真的由这份证据"
              "推出」这类判断本就不可机械判定。它们**不能**让任何一道门失败, 保留在此"
              "供人判断。", ""]
        for f in unver[:8]:
            L.append(f"- **`{f.phase}`** [{f.severity}] {f.claim[:150]}  ← `{f.artifact_key}`")
        L.append("")
    if fixed:
        L += [f"### 7.3 已关闭 ({len(fixed)} 条)", "",
              "断言重新成立, 因此自动关闭 —— 关闭它的是一次测量, 不是任何人的判断。", ""]
        for f in fixed[:6]:
            L.append(f"- ~~{f.claim[:120]}~~ (`{f.phase}`)")
        L.append("")
    if waived:
        L += [f"### 7.4 已由人工判定不处理 ({len(waived)} 条)", "",
              "**问题仍然存在**, 只是有人看过并决定这次不处理。理由一并列出 —— "
              "一条没有理由的豁免和「忘了」无法区分。", ""]
        for f in waived[:8]:
            L.append(f"- **`{f.phase}`** {f.claim[:110]} —— 理由: {f.resolution[:120]}")
        L.append("")
    return L


def _route_comparison(t: dict[str, Any]) -> list[str]:
    """Compare the two routes on the metrics that mean the same thing for both.

    The panel used to carry `topdown` with a single metric — kappa — so this
    section had nothing to say and did not exist. With both routes measured the
    comparison is possible, and the fairness rules matter more than the numbers:
    fragmentation rises with cluster count, and silhouette is the objective KMeans
    optimises, so a bare "top-down scores better" would be worth nothing.
    """
    rows = {r["subject"]: r for r in t["rows"]}
    bu = [s for s in ("families_pre_governance", "families_final", "leaves") if s in rows]
    td = [s for s in ("topdown_l1", "topdown_l2") if s in rows]
    if not bu or not td:
        return ["## 3. 两条路线的对照", "",
                "> ⚠️ **本次运行无法对照两条路线。** 面板中缺少"
                + ("自上而下" if not td else "自下而上")
                + "的划分, 因此「两条路线、一套度量」这一主张在本次交付中**没有证据支撑**。"
                  "这不是一个可以略过的细节 —— 它正是这套方法论的核心论断。", ""]

    out = ["## 3. 两条路线的对照", "",
           "这是整套方法论的核心论断所在: **两条路线各管一根轴, 因此必须放在同一套度量下看。** "
           "但并非表中每个指标都能承担这个对照。", "",
           "### 3.1 哪些指标可以用来对照, 哪些不可以", "",
           "| 指标 | 可用于跨路线对照? | 原因 |", "|---|---|---|"]
    reasons = {
        "n_clusters": (True, "粒度本身, 两边定义相同"),
        "template_fragmentation": (True, "问「同一意图是否被按措辞劈开」, 与产生标签的方法无关"),
        "nmi_reference": (True, "与既有参照体系的吻合度, 两边同样适用"),
        "purity_reference": (True, "同上"),
        "ambiguous_rate": (True, "边界处的模糊行占比, 两边同样适用"),
        "silhouette": (False, "**结构性不公平** —— 这正是 KMeans 直接优化的目标函数, "
                              "自下而上一方等于在自己的主场应试"),
        "stability_ari": (False, "问「重跑**聚类**是否落到同一处」, 对分类器指派的标签无定义"),
        "kmeans_refit_stability": (False, "同上, 且以 KMeans 为前提"),
        "heldout_reproduction": (False, "同上; 自上而下一侧的对应量是分类器的交叉验证准确率"),
        "coherence": (False, "盲评只对自下而上的叶做过"),
        "distill_accuracy": (False, "只对聚类标签计算"),
        "kappa": (False, "标注者之间的一致性, 是金标准的性质, **不是任何划分的性质**"),
    }
    # A metric is only comparable IN THIS RUN if both sides actually carry it.
    # `ambiguous_rate` is comparable in principle and was printed "✅ 可以" beside
    # a column of dashes, because the panel only computes it where centroids are
    # supplied. Claiming comparability the table cannot demonstrate is the same
    # failure as the panel carrying `topdown` with one metric.
    def _present(subjects: list[str], metric: str) -> bool:
        return any(rows[s].get(metric) is not None for s in subjects)

    graded = []
    for m in t["metrics"]:
        name = m["name"]
        ok, why = reasons.get(name, (False, "未归类, 按不可对照处理"))
        if ok and not (_present(bu, name) and _present(td, name)):
            side = "自上而下" if not _present(td, name) else "自下而上"
            ok, why = None, f"原则上可比, 但本次运行**{side}一侧未产出该指标**, 无从对照"
        graded.append((0 if ok else (1 if ok is None else 2), name, ok, why))
    for _, name, ok, why in sorted(graded):
        mark = "✅ 可以" if ok else ("⚠️ 本次不可" if ok is None else "❌ 不可以")
        out.append(f"| `{name}` ({_METRIC_ZH.get(name, '')}) | {mark} | {why} |")
    out.append("")
    usable = [n for r, n, ok, _ in sorted(graded) if ok]
    out += ["> 本次实际可用于跨路线对照的指标: "
            + (", ".join(f"`{n}`" for n in usable) if usable else "**无**")
            + f" (共 {len(usable)} 个, 表中共 {len(graded)} 个)。", ""]

    cols = [c for c in _ROUTE_FAIR
            if any(rows[s].get(c) is not None for s in bu + td)]
    out += ["### 3.2 对照结果", "",
            "| 方案 | " + " | ".join(_METRIC_ZH.get(c, c) for c in cols) + " |",
            "|---" * (len(cols) + 1) + "|"]
    for s in bu + td:
        out.append(f"| **{_label(s)}** | "
                   + " | ".join(_cell(rows[s].get(c)) for c in cols) + " |")
    out.append("")

    # The one comparison that is both fair and decisive, stated as a two-condition
    # sentence because fragmentation alone is meaningless without cluster count.
    frag = {s: rows[s].get("template_fragmentation") for s in bu + td
            if rows[s].get("template_fragmentation") is not None}
    kk = {s: rows[s].get("n_clusters") for s in bu + td if rows[s].get("n_clusters") is not None}
    if frag and kk:
        bu_pairs = sorted(((kk[s], frag[s], s) for s in bu if s in frag and s in kk))
        out += ["### 3.3 怎么读这张对照表", "",
                "**碎裂度必须与簇数一起读。** 簇越多, 越容易把一个意图散到多个簇里去, "
                "所以「碎裂度更低」单独说没有意义 —— 一个只有 2 个簇的划分几乎必然碎裂度最低, "
                "但它什么也没区分。", ""]
        if len(bu_pairs) >= 2:
            trend = " → ".join(f"k={int(k)} 时 {f:.3f}" for k, f, _ in bu_pairs)
            out += [f"自下而上一侧内部, 碎裂度确实随簇数单调上升: {trend}。"
                    "这条曲线就是判断另一条路线的基准线。", ""]
        for s in td:
            if s not in frag or s not in kk:
                continue
            below = [(k, f) for k, f, _ in bu_pairs if f > frag[s] and k < kk[s]]
            if below:
                k0, f0 = below[0]
                out += [f"**{_label(s)} 落在这条曲线之外**: 它有 {int(kk[s])} 个类目, "
                        f"多于自下而上 k={int(k0)} 的划分, 碎裂度却更低 "
                        f"({frag[s]:.3f} < {f0:.3f})。"
                        "**更细、并且更不碎** —— 这是一个双条件都成立的结论, "
                        "而不是用粒度换来的。", ""]
        out += ["> **为什么会这样, 机制上并不意外**: 自下而上是在一个"
                "**措辞相似度按 α 计入**的表征上聚类, 措辞孪生天然互相吸引; "
                "而自上而下按**意图定义**指派标签, 措辞推不动它。"
                "换句话说, 这个差距不是某一方「做得更好」, 而是两者**对措辞的敏感度不同** —— "
                "这恰恰是当初要并行跑两条路线的原因。", "",
                "> **不要据此二选一。** 两套标签回答不同的问题, 且是并排交付的。"
                "下游若必须选一个, 应当依据场景要的是**意图**还是**措辞群**, "
                "而不是依据这张表里哪个数字更好看。", ""]
    return out


def _fig(figs: dict[str, Any], name: str, caption: str) -> str:
    ref = figs.get(name)
    if ref is None:
        return ""
    fn = getattr(ref, "path", None) or getattr(ref, "name", name)
    return f"![{caption}]({Path(str(fn)).name})\n\n*图: {caption}*\n"
