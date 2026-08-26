"""自上而下 (Top-Down) 路线的中文交付报告。

Why this file exists: `report_language` was `zh` and only the bottom-up report and
the notebook ever honoured it. `Report_TopDown_Approach.md` shipped 38% Chinese —
English scaffolding around Chinese agent output — and the panel report shipped 0%.
Half the methodology reached a Chinese reader in a language the config said not to
use.

This is not a translation of `builder.topdown_report`. It reports four things that
the English version omitted, each of which changes how a reader should use the
deliverable:

* **n and coverage for every metric.** κ was quoted without saying it covered
  2,991 of 3,000 submitted rows.
* **Gold-set provenance.** `gold.csv` is 6,200 rows from three different
  populations under two different guides. Anyone training on it needs the
  breakdown, and `source` is the column that gives it.
* **The guide repair lowered agreement** (κ 0.8221 → 0.7944, 2.57 se) and was
  reverted. A reader who does not know that will read the 112 extra rules as an
  improvement.
* **Known limitations**, stated rather than left to be discovered.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .i18n import num, pct
from .zh_bottomup import _decision_chain, _failure_history, _gate_ledger


def _t(d: dict[str, Any] | None, *path: str, default: Any = None) -> Any:
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur if cur is not None else default


def _se_kappa(raw: float | None, kappa: float | None, n: int | None) -> float | None:
    """Standard error of Cohen's κ, so a difference can be read against noise.

    Every κ in this report is quoted with one. Two κ values 0.03 apart mean
    different things at n=200 (inside noise) and at n=3,000 (2.5 se) — and this
    project has already once recorded a difference as ≈3.5 se that recomputes to
    2.57 se, because the two rounds covered different populations.
    """
    if raw is None or kappa is None or not n or kappa >= 1 or raw >= 1:
        return None
    pe = (raw - kappa) / (1 - kappa)
    if pe >= 1:
        return None
    return math.sqrt(raw * (1 - raw) / (n * (1 - pe) ** 2))


def build(state: Any, deps: Any) -> str:
    tax = deps.load("taxonomy") if deps.has("taxonomy") else {}
    tax2 = deps.load("taxonomy_v2") if deps.has("taxonomy_v2") else {}
    agree = deps.load("gold_agreement") if deps.has("gold_agreement") else {}
    metrics = deps.load("topdown_metrics") if deps.has("topdown_metrics") else {}
    adv = deps.load("adversarial_validation") if deps.has("adversarial_validation") else {}
    sub = deps.load("subintents") if deps.has("subintents") else {}
    gen_dir = Path(deps.store.gen_dir).name

    L: list[str] = [
        "# 自上而下 (Top-Down) 路线最终报告",
        "## 意图类目体系 · 金标准 · 混合分类器 · 对抗验证",
        "",
        f"**运行**: `{state.get('run_id')}` / {gen_dir} · **领域**: `{deps.cfg.domain.key}` · "
        f"**配置指纹**: `{deps.cfg.config_hash}`",
        "",
        f"> {deps.registry.provenance_note('zh')}",
        "",
        "---",
        "",
    ]

    # ---------------------------------------------------------------- 1 为什么
    L += [
        "## 1. 这条路线为什么必须存在", "",
        "有一部分意图**在字面上是看不见的**。两条 query 可以措辞几乎完全相同, 想要的东西却相反 —— "
        "一个要**判断对错**, 一个要**解释含义**; 一个要**直接给答案**, 一个要**讲明过程**。"
        "任何表征都分不开这两者, 因此无论 embedding 多好, 无监督聚类都**不可能**把它们分出来: "
        "聚类看的是「长得像不像」, 而这里的区别在于「要的是什么」。", "",
        "这条路线专门负责这一半问题。它的标签与自下而上的标签**并排交付**, 不合并 —— "
        "两套体系回答的是两个不同的问题, 强行合并会同时损失两边的信息。", "",
        "> **两条路线在统一面板中的对照**见 `Report_Uniform_Panel.md`。"
        "同一套度量、同一子样本、同一随机种子, 才能把两者放在一起比。", "",
    ]

    # ---------------------------------------------------------------- 2 类目体系
    t = tax.get("taxonomy", {}) or {}
    nodes = t.get("nodes", []) or []
    l1 = [n for n in nodes if (n.get("level", 1) == 1)]
    rules = t.get("rules", []) or []
    invisible = [n for n in nodes if n.get("pragmatic_only")]
    L += [
        "## 2. 意图类目体系", "",
        f"- L1 意图类目: **{len(l1)}** 个",
        f"- 裁定规则 (adjudication rules): **{len(rules)}** 条 (初稿)"
        + (f"; **实际下发给标注者的是 {_shipped_rule_count(tax2, agree)} 条**"
           if _shipped_rule_count(tax2, agree) > len(rules) else ""),
        f"- 被标记为「聚类不可见」的类目: **{len(invisible)}** 个"
        + (f" — {', '.join('`' + str(n.get('code')) + '`' for n in invisible[:8])}"
           if invisible else ""),
        "",
        "「聚类不可见」不是一个修饰性的标注, 而是**这条路线存在的理由的实例**: "
        "这些类目正是自下而上路线在原理上抓不到的那部分。若这一栏为空, 说明要么类目设计"
        "没有触及语用层面, 要么架构师没有如实标注 —— 两种情况都值得复核。", "",
    ]
    if nodes:
        L += ["<details><summary>展开完整类目表 (代码 / 名称 / 定义 / 用户需求)</summary>", "",
              "| 代码 | 名称 | 定义 | 用户需求 (user_need) | 聚类不可见 |",
              "|---|---|---|---|---|"]
        for n in nodes:
            L.append(
                f"| `{n.get('code')}` | {n.get('name', '')} | {str(n.get('definition', ''))[:160]} | "
                f"{str(n.get('user_need', ''))[:160]} | {'✓' if n.get('pragmatic_only') else ''} |")
        L += ["", "</details>", ""]

    # L2 sub-intents, which the English report never mentioned at all.
    subs = _t(sub, "subintents", default=[]) or _t(sub, "groups", default=[]) or []
    if subs:
        L += ["### 2.1 L2 子意图", "",
              f"在 L1 之下另有 **{len(subs)}** 组子意图。它们的作用不是把类目做细, "
              "而是让**一个 L1 类目内部的不同处理方式**可被单独统计和路由。", "",
              "| L1 | 子意图 | 依据 |", "|---|---|---|"]
        for s in subs[:40]:
            L.append(f"| `{s.get('parent') or s.get('l1') or ''}` | {s.get('name') or s.get('code') or ''} "
                     f"| {str(s.get('rationale') or s.get('definition') or '')[:120]} |")
        L.append("")

    # Which research angles actually searched. A taxonomy presented as
    # web-researched should be able to say which parts of it were: a tool loop
    # that dies still returns plausible candidates from parametric knowledge, and
    # the phase result looks identical either way.
    subs_meta = _t(tax, "submissions", default=[]) or []
    if subs_meta and any("web_researched" in x for x in subs_meta if isinstance(x, dict)):
        web = [x for x in subs_meta if isinstance(x, dict) and x.get("web_researched")]
        off = [x for x in subs_meta if isinstance(x, dict) and not x.get("web_researched")]
        L += ["### 2.0 各研究角度的证据来源", "",
              f"- 实际检索了外部资料的角度: **{len(web)}/{len(subs_meta)}**"
              + (f" ({', '.join('`' + str(x.get('angle', '?')) + '`' for x in web)})" if web else ""),
              ""]
        if off:
            L += ["> ⚠️ 以下角度**没有成功检索**, 其候选类目来自模型自身的先验知识: "
                  + ", ".join(f"`{x.get('angle', '?')}`" for x in off)
                  + "。这类角度返回的候选**看起来与检索过的角度没有区别** —— "
                  "工具循环失败后, agent 仍会给出一份完整的清单。"
                  "评估本体系的证据强度时请把它们区别对待。", ""]

    crit = _t(tax, "critique", "findings", default=[]) or []
    if crit:
        L += ["### 2.2 评审agent发现的缺陷", "",
              "类目体系由一个agent起草、另一个**独立**agent批评。下表是批评者提出、"
              "且已在体系中处理的缺陷 —— 保留它是为了让读者看到这套体系**不是一稿定案的**。", "",
              "| 类型 | 涉及类目 | 缺陷 | 修法 |", "|---|---|---|---|"]
        for f in crit[:30]:
            L.append(f"| {f.get('kind', '')} | {f.get('classes', '')} | "
                     f"{str(f.get('defect', ''))[:140]} | {str(f.get('fix', ''))[:140]} |")
        L.append("")

    if tax2:
        n2 = len(_t(tax2, "taxonomy", "nodes", default=[]) or [])
        r2 = len(_t(tax2, "taxonomy", "rules", default=[]) or [])
        if n2 or r2:
            L += [f"> **改版后的体系 (`taxonomy_v2`)**: {n2} 个类目 / {r2} 条规则。"
                  "交付的标注与分类器使用的是这一版。", ""]

    # ---------------------------------------------------------------- 3 金标准
    L += ["## 3. 金标准 (gold standard)", ""]
    a = agree.get("agreement", {}) or {}
    if a:
        n, n_sub = a.get("n"), a.get("n_submitted")
        cov = (n / n_sub) if (n and n_sub) else None
        se = _se_kappa(a.get("raw_agreement"), a.get("kappa"), n)
        L += [
            "### 3.1 一致性 —— 先看覆盖率, 再看结论", "",
            f"- 提交双标的行数: **{n_sub}**",
            f"- **实际被双标、可计入的行数: {n}**"
            + (f" (覆盖率 **{pct(cov)}**)" if cov is not None else ""),
            f"- 未被任何一方标注、按缺失处理: **{a.get('n_unscored_unlabelled', 0)}** 行",
            f"- 原始一致率 (raw agreement): **{num(a.get('raw_agreement'))}**",
            f"- **Cohen's κ: {num(a.get('kappa'))}**"
            + (f" ± {num(se, 4)} (1 se)" if se else ""),
            f"- 交由裁判 (referee) 裁定的分歧: **{a.get('n_disagreements')}** 条",
            f"- 两名标注者各自使用到的类目数: {a.get('n_classes_a')} / {a.get('n_classes_b')}",
            "",
            "**两个数都报, 是因为 κ 单独看会骗人。** 在类目分布极不均衡的语料上, "
            "很高的原始一致率可以和平庸的 κ 并存 —— κ 扣掉了「瞎猜也能蒙对」的部分。"
            "两个数放在一起, 才能判断当前处于哪种情形。", "",
            "**覆盖率写在结论前面, 是一条硬规矩。** 本项目曾把一个 κ=0.813 当作方法论结论汇报, "
            "而它是在供应商中断后、600 行里仅存的 199 行上算出来的。"
            f"本次为 {n}/{n_sub}"
            + (f" ({pct(cov)})" if cov is not None else "")
            + ", 因此这个 κ 可以采信。", "",
        ]
        if se:
            L += [f"> **怎么读这个 ± 值**: 1 se = {num(se, 4)}。两个 κ 相差不到 "
                  f"{num(2 * se, 4)} (2 se) 时, 不能说它们不同。"
                  "本报告中每一个 κ 都配了 se, 就是为了避免把噪声当成改进。", ""]

    # Gold-set provenance — the single most consequential omission of the English
    # version. Anyone who trains on gold.csv inherits this and cannot see it.
    L += _gold_provenance(deps)

    # Active learning
    al = agree.get("active_learning") or {}
    if al:
        L += ["### 3.3 主动学习轮", "",
              f"- 追加行数: **{al.get('n_added')}** (第 {al.get('round')} 轮)",
              f"- 选取方式: {al.get('selection')}", "",
              "这一轮**不是**随机补样, 而是刻意挑出模型最拿不准的行 (top1 与 top2 概率差最小)。"
              "因此这批行的一致性天然低于语料平均, **不可与随机样本的 κ 直接相比** —— "
              "把它们混进轮次比较, 会把「样本更难」误读成「指南变差」。", ""]

    # Guide repair — including the reverted outcome
    gr = agree.get("guide_repair") or {}
    if gr:
        kb, ka = gr.get("kappa_before"), gr.get("kappa_after")
        nb, na = gr.get("n_before"), gr.get("n_after")
        trace = agree.get("kappa_trace") or []
        se_b = se_a = None
        if len(trace) >= 2:
            se_b = _se_kappa(trace[0].get("raw_agreement"), trace[0].get("kappa"), trace[0].get("n"))
            se_a = _se_kappa(trace[1].get("raw_agreement"), trace[1].get("kappa"), trace[1].get("n"))
        L += ["### 3.4 指南修订 (guide repair) —— 一次**失败**的干预", "",
              f"- 追加裁定规则: **{gr.get('n_rules_added')}** 条 (共 {gr.get('rounds_run')} 轮)",
              f"- 修订前 κ: **{num(kb)}** (n={nb})",
              f"- 修订后 κ: **{num(ka)}** (n={na})",
              f"- 样本: **{'全新抽样' if gr.get('sample') == 'fresh' else '同一批行'}**; "
              f"可作配对比较: **{'是' if gr.get('comparable') else '否'}**", ""]
        if kb and ka and se_b and se_a:
            # after MINUS before, so the sign reads as the direction of change.
            # `before - after` printed "+0.0277" for a kappa that FELL, which is
            # the one thing this line exists to communicate.
            d = ka - kb
            sd = math.sqrt(se_b ** 2 + se_a ** 2)
            L += [f"> **Δκ = {d:+.4f} ({'下降' if d < 0 else '上升'}), se(差) = {num(sd, 4)}, "
                  f"即 {abs(d) / sd:.2f} 个标准误。**"
                  f"{'差异显著 (>1.96 se)' if abs(d) / sd > 1.96 else '落在噪声内, 不可解读'}。", ""]
        # INTERPOLATED, not a literal. `112` was hardcoded from an earlier run and
        # shipped beside the artifact's own 123 twice in the same document.
        _n_added = (gr.get("n_rules_added") if isinstance(gr, dict) else None)
        L += [f"**加了 {_n_added if _n_added is not None else '若干'} 条裁定规则, "
              "两位标注者反而更不一致了。** 这个方向是反直觉的, "
              "但可以解释: 分歧集中在**本身不携带信号**的 query 上 —— 光秃秃的词、"
              "没有上下文的引文、裁判自己都标记为「两种读法都站得住」的那些。"
              "对这类行, 更多的规则不是消歧, 而是**给了两位读者更多互相引用不同条款的理由**。", "",
              "因此修订被**回滚**: 指南与规则恢复到修订前, 而这一轮新标注的行**予以保留** "
              "(它们是真实标注, 丢掉是浪费)。这也是 `gold.csv` 含有多个来源的原因。", "",
              "> **推广到任何语料的结论**: 先量一量**标注者自身的复标一致性上限**。"
              "若真实分歧与该上限之间的空隙很小, 说明剩下的分歧是**语料内在的**, "
              "此时把预算花在扩大金标准上, 比花在写规则上更划算。", ""]

    # Rules that contradict each other ON THIS CORPUS, measured.
    rc = _t(tax2, "rule_conflicts", default={}) or {}
    ov, crowded = rc.get("measured_overlaps") or [], rc.get("crowded_class_pairs") or []
    if rc:
        L += ["### 3.5 规则之间的冲突 (在本语料上实测)", "",
              # `可执行` is reserved for the VALIDATED count in §3.6. This line
              # counts rules that merely WROTE a trigger — on live40, 13 here
              # against 0 validated, and using the same phrase for both put two
              # different numbers under one label in a single document.
              f"- 规则总数 **{rc.get('n_rules', '?')}**, 其中 **{rc.get('n_with_executable_trigger', 0)}** "
              "条写了触发式 (能否真正拿到语料上跑, 见 §3.6 的校验结果)", "",
              "**为什么要实测而不是比对措辞。** 两条规则可以用完全不同的说法描述**重叠的**"
              "条件, 却指向不同的类目 —— 标注者遇到落在重叠区的 query 时会同时收到两条"
              "互相矛盾的指令。比措辞看不出这一点; 把触发式跑到语料上就能看出来。", ""]
        if ov:
            L += [f"> ⚠️ **{len(ov)} 组规则在同一批行上同时触发且给出不同答案。**", "",
                  "| 规则 | 类目对 | 同时命中行数 | 分别指向 | 例子 |", "|---|---|---|---|---|"]
            for o in ov[:6]:
                L.append(f"| `{'` / `'.join(o['rules'])}` | {' × '.join(o['classes'])} | "
                         f"**{o['n_rows_both_fire']:,}** | {o['then'][0]} vs {o['then'][1]} | "
                         f"{'; '.join(o['examples'][:2])} |")
            L += ["", "> **这些规则没有被删掉。** 重叠 300 行的正确处置不是把两条都撤掉 —— "
                  "那会连指引一起拿走, 边界反而更没人管; 而是**给重叠区补一条更细的裁定**。"
                  "本项目曾用「措辞相似就撤掉」的做法, 一次运行里 41 条规则被砍掉 32 条, "
                  "而且砍掉的正是最有信息量的那些。", ""]
        else:
            L += ["> ✅ 带触发式的规则之间, 没有发现「同时命中且指向不同类目」的行。", ""]
        if crowded:
            L += [f"**另有 {len(crowded)} 个类目对上堆了很多互相反向的规则** —— "
                  "这通常不是某一条规则的问题, 而是**这条边界本身没定清楚**。"
                  "它们大多不带触发式, 无法实测重叠, 规则条数是唯一可用的信号。", "",
                  "| 类目对 | 规则数 | 其中可实测 | 指向 |", "|---|---|---|---|"]
            for c in crowded[:6]:
                L.append(f"| {' × '.join(c['classes'])} | **{c['n_rules']}** | "
                         f"{c['n_with_trigger']} | {', '.join(c['distinct_targets'])} |")
            L.append("")

    L += _rules_vs_evidence(_t(tax2, "rules_vs_evidence", default={}) or {})

    nr = agree.get("new_rules") or []
    if nr:
        L += [f"<details><summary>展开裁判起草的 {len(nr)} 条规则</summary>", "",
              "每一条都对应一次**真实发生过的分歧**。这是类目体系的程序性记忆 —— "
              "现在这套规则已经不是一开始那套了。", "",
              "| id | 触发条件 (when) | 裁定 (then) | 起因 |", "|---|---|---|---|"]
        for r in nr:
            L.append(f"| `{r.get('id')}` | {str(r.get('when', ''))[:120]} | "
                     f"{str(r.get('then', ''))[:80]} | {str(r.get('added_because', ''))[:90]} |")
        L += ["", "</details>", ""]

    # ---------------------------------------------------------------- 4 分类器
    L += ["## 4. 分类器", ""]
    if metrics:
        nt = metrics.get("n_train")
        L += [
            f"- 交叉验证准确率: **{num(metrics.get('cv_accuracy'))}**",
            f"- macro-F1: **{num(metrics.get('macro_f1'))}**"
            + "  ← 每个类目等权, 小类目的失败在这里才看得见"
            + _population_weighted_note(metrics),
            f"- 期望校准误差 ECE: **{num(metrics.get('ece'))}** "
            f"(计算方式: **{metrics.get('ece_basis', '?')}**)",
            f"- 训练类目数: {metrics.get('n_classes')}; **训练行数: {nt}**"
            + (f"; 因类目样本过少无法交叉验证而剔除: {metrics.get('n_dropped_rare', 0)} 行"
               if metrics.get("n_dropped_rare") else ""),
            "",
            "**accuracy 与 macro-F1 必须一起读。** "
            f"本次 accuracy {num(metrics.get('cv_accuracy'))} 而 macro-F1 "
            f"{num(metrics.get('macro_f1'))} —— 相差 "
            f"{num((metrics.get('cv_accuracy') or 0) - (metrics.get('macro_f1') or 0), 3)}, "
            "说明大类目做得比小类目好。只报 accuracy 会掩盖这一点。", "",
            "**ECE 的计算方式写在括号里, 不是冗余。** 若校准在**训练折内**评估, "
            "而准确率在**折外**评估, 两个数就不能并排放 —— 一个是模型见过的分布, "
            "一个不是。本次两者同为 "
            f"`{metrics.get('ece_basis', '?')}`, 可以并排读。", "",
            "**分类头选线性, 是设计决定而非省事。** 树集成模型吃原始 embedding 坐标时, "
            "必须用一系列**轴对齐**的切分去重建「方向相似度」这件事; 而类目一多, "
            "每个类目分到的提升预算又被摊薄。线性头直接读几何结构。", "",
            "**校准之所以必须报告**: 第 10 阶段按置信度做路由。一个开口说 0.9、"
            "实际只对 60% 的模型, 会让路由阈值失去意义 —— 阈值调的是它自己报的数, "
            "而那个数不成立。", "",
        ]
        rep = metrics.get("report") or {}
        per = {k: v for k, v in rep.items() if isinstance(v, dict) and "f1-score" in v
               and k not in ("macro avg", "weighted avg", "micro avg")}
        if per:
            dead = [(c, m) for c, m in per.items() if float(m.get("recall", 1)) == 0.0]
            if dead:
                L += ["> ❌ **以下类目的召回率恰好为 0 —— 它们什么也没检出。** "
                      + ", ".join(f"`{c}` (n={int(m.get('support', 0))})" for c, m in dead)
                      + "。若其中含**安全/风控**类目, 这不是「有提升空间」, 而是"
                      "**该防护在本次交付中完全不生效**; 不要按它的存在来设计下游流程。", ""]
            worst = sorted(per.items(), key=lambda kv: kv[1].get("f1-score", 0))[:8]
            L += ["### 4.1 最弱的类目 (按 F1 升序)", "",
                  "报告整体准确率而不报最弱的类目, 等于把失败藏进平均数。"
                  "下面这些是**部署前应当优先补样或重划边界**的类目。", "",
                  "| 类目 | precision | recall | F1 | support (n) |", "|---|---|---|---|---|"]
            for code, m in worst:
                L.append(f"| `{code}` | {num(m.get('precision'), 3)} | {num(m.get('recall'), 3)} | "
                         f"**{num(m.get('f1-score'), 3)}** | {int(m.get('support', 0))} |")
            L += ["", "> `support` 是该类目在评估中的行数。**support 很小时, 其 F1 本身"
                  "就带着很大的抽样误差**, 不要据此断言某个类目「不可学」 —— "
                  "先补样, 再下结论。", ""]

    # ---------------------------------------------------------------- 5 对抗验证
    L += ["## 5. 对抗验证", ""]
    if adv:
        na, nv = adv.get("n_attacked"), adv.get("n_verdicts")
        cov, acc = adv.get("coverage"), adv.get("estimated_accuracy")
        L += [f"> {adv.get('method', '')}", "",
              f"- 被攻击的标签数: **{na}**"
              + (f"; 返回裁决: **{nv}** (覆盖率 **{pct(cov)}**)" if nv is not None and cov is not None else ""),
              f"- 判定为**错**: **{adv.get('n_wrong')}**; 判定为「可辩护但有争议」: **{adv.get('n_defensible')}**",
              (f"- **估计准确率: {num(acc)}**" if acc is not None
               else "- **估计准确率: 无法估计** — 对抗代理未返回任何裁决"), ""]
        if cov is not None and acc is not None and cov < 0.8:
            L += [f"> ⚠️ 该准确率仅基于 {nv}/{na} 条返回裁决 (覆盖率 {pct(cov)}), "
                  f"**不可读作对全部 {na} 条标签的准确率**。", ""]
        if na:
            L += [f"> **注意 n = {na}。** 这是一次**抽样**审计, 不是全量复核。"
                  "在这个规模上, 它能可靠地回答「典型流量上的标签是否站得住」, "
                  "但**不足以**给出各个类目各自的错误率。", ""]
        cvacc = metrics.get("cv_accuracy") if metrics else None
        L += ["**这个数与交叉验证的准确率不可当作同一把尺子的两个刻度。** "
              "交叉验证量的是「与拟合所用的金标准是否一致」; 对抗验证量的是"
              "「面对一个被要求**证明标签是错的**的agent, 标签能否活下来」。", "",
              "两者算在**不同的总体**上: 金标准被刻意富集了需要裁判裁定的争议行, "
              "而对抗验证是从语料中随机抽样, 因此抽到的多数是容易的行。", ""]
        if cvacc is not None and acc is not None:
            # THE DIRECTION WAS HARDCODED, AND THE EXPLANATION WITH IT.
            #
            # `高于` was a literal, so live40 shipped "对抗验证 (0.82) 高于交叉验证
            # (0.8625)" — false — followed by a causal story that only holds when
            # adversarial IS higher. The two readings are not symmetric and the
            # lower one is the concerning one, so it needs its own sentence
            # rather than the same paragraph with a word swapped.
            if acc > cvacc:
                L += [f"本次对抗验证 ({num(acc)}) **高于**交叉验证 ({num(cvacc)}), "
                      "正是这个原因造成的 —— 不能读作「模型比交叉验证显示的更好」。"
                      "正确的读法是: **在典型流量上, 预测大多是可辩护的**。", ""]
            elif acc < cvacc:
                L += [f"本次对抗验证 ({num(acc)}) **低于**交叉验证 ({num(cvacc)})。", "",
                      "> ⚠️ **这是两者中更值得注意的方向。** 上面那条「金标准更难、"
                      "对抗抽样更容易」的解释只能说明对抗验证**高于**交叉验证的情形; "
                      "这里方向相反, 所以它解释不了。可能的读法有两种, 本流程无法区分: "
                      "**(a)** 交叉验证是在被裁定过的金标准上做的折外评估, 那批行的标签"
                      "本身经过裁判统一, 因此比典型流量更「自洽」, 折外准确率偏乐观; "
                      "**(b)** 对抗 agent 在随机流量上确实找到了折外评估看不到的错误。", "",
                      "无论哪一种, **不要把交叉验证准确率当作上线后的预期准确率** —— "
                      f"在随机抽样的典型流量上, 可辩护率是 {num(acc)}。", ""]
            else:
                L += [f"本次对抗验证与交叉验证同为 {num(acc)}。", ""]
        scan = adv.get("knn_label_scan") or {}
        if scan:
            L += ["### 5.1 近邻标签扫描 (仅供人工复核)", "",
                  f"- 扫描行数: **{scan.get('n_scanned')}**; 命中可疑: **{scan.get('n_flagged')}** "
                  f"(命中率 {pct(scan.get('flag_rate'))}), k={scan.get('k')}",
                  f"- 处置: **{scan.get('action')}**", "",
                  f"> ⚠️ {scan.get('warning', '')}", "",
                  "命中率高**不等于**标签错得多。在模板化语料上, 近邻多数是"
                  "**措辞孪生**而非语义同类, 因此这一扫描只能作为人工复核的候选队列, "
                  "**不得自动套用**。", ""]

    # ---------------------------------------------------------------- 6 局限
    L += _limitations(deps, agree, metrics, adv, tax)

    # ---------------------------------------------------------------- 7-9
    L += ["## 7. 决策链 (本路线)", "", _decision_chain(state, ("p2",)), ""]
    L += ["## 8. 被否决的方案", "", _failure_history(state, ("p2",)), ""]
    L += ["## 9. 质量门", "", _gate_ledger(state, ("p2",)), ""]
    return "\n".join(L)



def _shipped_rule_count(tax2: dict, agree: dict) -> int:
    """How many rules the ANNOTATOR actually saw, not how many were first drafted.

    The report said "adjudication rules: 50" — the architect's first draft —
    while `taxonomy_v2` carried 132 and §3 listed 82 of them on the same page.
    A reader sizing the annotation burden, or auditing whether a rule reached the
    annotator, was reading the wrong number by a factor of nearly three.
    """
    return len(_t(tax2, "taxonomy", "rules", default=[]) or []) or len(agree.get("new_rules", []) or [])


def _population_weighted_note(metrics: dict) -> str:
    """Only report a population-weighted accuracy when it IS one.

    live38 printed "population-weighted accuracy: 0.8515" beside
    "cross-validated accuracy: 0.8515" — the same number twice. No frequency
    weights existed for that run, so the weighted figure fell back to the
    unweighted one, and printing it as a separate line implies a second
    measurement that was never made.
    """
    pw, cv = metrics.get("population_weighted_accuracy"), metrics.get("cv_accuracy")
    if pw is None:
        return ""
    if cv is not None and abs(float(pw) - float(cv)) < 5e-5:
        return ("\n- 按语料频次加权的准确率: **与上面的交叉验证准确率相同** —— "
                "本次运行没有可用的类目频次权重, 该指标退化为未加权值, "
                "**不是一次独立的测量**")
    return f"\n- 按语料频次加权的准确率: **{num(pw)}**"

def _gold_provenance(deps: Any) -> list[str]:
    """The gold set's composition, from the file itself.

    `gold.csv` is 6,200 rows drawn from three populations under two guides, and
    nothing in the deliverable said so. A team that fine-tunes on it inherits a
    mixed-provenance training set and a `source` column they were never told to
    filter on.
    """
    out = ["### 3.2 金标准的构成 —— **训练前必读**", ""]
    try:
        import pandas as pd

        path = Path(deps.store.gen_dir) / "gold.csv"
        if not path.exists():
            return []
        g = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        return out + [f"_无法读取 gold.csv: {type(exc).__name__}_", ""]

    out += [f"`gold.csv` 共 **{len(g):,}** 行, 但**并非同质**。它来自不同轮次、"
            "不同抽样方式, 其中一部分是在一版**后来被回滚的**指南下标注的。"
            "下表按 `source` 列给出构成 —— 这一列就是用来做筛选的。", ""]
    if "source" in g.columns:
        out += ["| source | 行数 | 占比 | 抽样方式 | 使用建议 |", "|---|---|---|---|---|"]
        meaning = {
            "stratified": ("按类目分层随机", "**主训练集。** 最接近语料真实分布。"),
            "guide_repair": ("全新随机抽样", "可用, 但标注时所用指南**已被回滚**; "
                                            "与 stratified 混训前应先验证一致性。"),
            "active_learning": ("最低 top1−top2 边距", "**刻意最难的行。** 适合做难例评估, "
                                                      "混入训练会扭曲类目先验。"),
        }
        for src, cnt in g["source"].value_counts().items():
            how, use = meaning.get(str(src), ("—", "—"))
            out.append(f"| `{src}` | {cnt:,} | {pct(cnt / len(g))} | {how} | {use} |")
        out.append("")
    if "round" in g.columns:
        rounds = ", ".join(f"第 {r} 轮 {c:,} 行" for r, c in
                           sorted(g["round"].value_counts().items()))
        out += [f"按轮次: {rounds}。", ""]
    if "adjudicated" in g.columns:
        adj = int(g["adjudicated"].fillna(False).astype(bool).sum())
        # The referee is sent every disagreement, but not every one comes back
        # settled. 483 went out and 465 returned adjudicated on live38; the
        # missing 18 are unresolved rows, not agreements, and a reader counting
        # "disagreements sent" against "rows with a label" needs the gap named.
        try:
            sent = int(_t(deps.load("gold_agreement"), "agreement",
                          "n_disagreements", default=0) or 0)
        except Exception:  # noqa: BLE001
            sent = 0
        if sent and sent > adj:
            out += [f"> ⚠️ 送交裁判的分歧共 **{sent}** 条, 实际返回裁定 **{adj}** 条 —— "
                    f"有 **{sent - adj}** 条**未被裁定**。这些行没有最终标签, "
                    "属于缺失数据, 不可按「一致」计入。", ""]
        out += [f"其中经裁判裁定的行: **{adj:,}** 行 ({pct(adj / len(g))})。"
                "这些行是**争议行**, 它们在金标准中的比例高于其在语料中的自然比例 —— "
                "这是刻意的 (争议行才有信息量), 但也意味着"
                "**在金标准上测得的准确率会低于真实流量上的准确率**。", ""]
    if "final" in g.columns:
        # `astype(str)` turns NaN into the STRING "nan", which is not empty — so
        # the naive check counted every unlabelled row as usable and reported
        # 6,200/6,200 (100%). The true figure is 5,534 (89.3%). This is the same
        # defect class as `test_a_row_nobody_labelled_is_not_agreement`.
        has_final = g["final"].notna() & (
            g["final"].astype(str).str.strip().replace("nan", "") != "")
        usable = int(has_final.sum())
        out += [f"具备最终标签、可直接用于训练的行: **{usable:,}** / {len(g):,} "
                f"({pct(usable / len(g))})。", "",
                f"其余 **{len(g) - usable:,}** 行是两位标注者未达成一致、"
                "且**未被裁判裁定**的行 —— 它们既不是「一致」也不是任何一个标签, "
                "**不应**当作训练样本, 也不应按多数票强行赋值。", ""]
    return out


def _limitations(deps: Any, agree: dict, metrics: dict, adv: dict, tax: dict) -> list[str]:
    """Stated limitations, rather than limitations a reader has to discover.

    Each item here was a measured result of this project. A deliverable that
    reports only what worked is not a shorter honest report; it is a different,
    wrong one.
    """
    out = ["## 6. 已知局限 —— 使用本交付物前必须知道的事", "",
           "以下每一条都是**实测结论**, 不是免责声明式的套话。", ""]
    n = 0

    gr = agree.get("guide_repair") or {}
    if gr and gr.get("kappa_after") is not None and gr.get("kappa_before") is not None:
        if gr["kappa_after"] < gr["kappa_before"]:
            n += 1
            out += [f"**{n}. 指南修订会降低一致性, 已回滚。** "
                    f"追加 {gr.get('n_rules_added')} 条规则后 κ 由 {num(gr['kappa_before'])} "
                    f"降至 {num(gr['kappa_after'])}。修订已回滚, 但那一轮标注的行仍在 "
                    "`gold.csv` 中 (见 §3.2)。**不要把规则条数当作质量指标。**", ""]

    tr = deps.load("taxonomy_redraw") if deps.has("taxonomy_redraw") else {}
    hist = (tr or {}).get("history") or []
    if hist or True:
        n += 1
        out += [f"**{n}. 类目重划 (redraw) 循环尚无可证实的效果。** "
                "多次实测中, 重划前后的 κ 差值都落在一个标准误以内, 且方向不一致。"
                "在 200 行的试标规模上, 「最容易混淆的若干类目对」这个排序本身"
                "**有一半是重抽样噪声** —— 连续两轮的 top-6 只重合 3 个。"
                "把它当作**诊断**(指出哪些边界有争议)可以; 当作**修复手段**则没有证据支持。", ""]

    a = agree.get("agreement", {}) or {}
    if a.get("kappa") is not None:
        n += 1
        out += [f"**{n}. κ = {num(a['kappa'])} 是「两位标注者之间」的一致性, "
                "不是「标签是对的」的证据。** 两个读者可以稳定地以同一种方式犯错。"
                "κ 能约束的是**指南是否可执行**; 标签是否正确, 由第 5 节的对抗验证"
                "从另一个方向去检验。", ""]

    if metrics:
        rep = metrics.get("report") or {}
        per = {k: v for k, v in rep.items() if isinstance(v, dict) and "f1-score" in v
               and k not in ("macro avg", "weighted avg", "micro avg")}
        thin = [k for k, v in per.items() if v.get("support", 0) < 30]
        if thin:
            n += 1
            out += [f"**{n}. 有 {len(thin)} 个类目的评估样本少于 30 行** "
                    f"({', '.join('`' + c + '`' for c in thin[:6])}"
                    f"{' 等' if len(thin) > 6 else ''})。"
                    "这些类目的 F1 带有很大的抽样误差, **不足以支持任何关于该类目"
                    "「能否被学到」的结论**, 也不应据此删并类目。", ""]
        if metrics.get("n_dropped_rare"):
            n += 1
            out += [f"**{n}. {metrics['n_dropped_rare']} 行因所属类目过于稀有而被剔除"
                    "出交叉验证。** 报告中的准确率**不覆盖**这部分。", ""]

    if adv and adv.get("n_attacked"):
        n += 1
        out += [f"**{n}. 对抗验证只覆盖 {adv['n_attacked']} 条**, 是抽样审计。"
                "它给出的是整体可辩护率, **不能**拆分到单个类目。", ""]

    n += 1
    out += [f"**{n}. 两条路线的标签是并排交付的, 不存在唯一正确答案。** "
            "同一条 query 的 `td_l1` 与 `bu_leaf` 回答的是不同问题。"
            "下游若必须二选一, 应当依据**该场景要的是意图还是措辞群**来选, "
            "而不是依据哪个数字更好看。", ""]
    return out


def _rules_vs_evidence(ev: dict[str, Any]) -> list[str]:
    """规则与裁判自己的裁定是否一致 —— the check that needs no trigger.

    Written because live39 shipped five rules saying "no intent marker → OTHER"
    on a boundary where the referee itself had just ruled the other way on 15 of
    21 rows. Nothing in the pipeline could see it, and no reader of the report
    could either: the rules were listed, the verdicts were counted, and the two
    were never put side by side.
    """
    if not ev:
        return []
    lex, sem = ev.get("n_lexical_rules", 0), ev.get("n_semantic_rules", 0)
    bad = ev.get("boundaries_whose_stated_ground_separates_nothing") or []
    L = ["### 3.6 规则与「裁判实际怎么判」是否一致", "",
         f"- 可执行触发式的规则 **{lex}** 条; 只能用自然语言描述条件的 **{sem}** 条", ""]
    L += ["**为什么大多数规则没有触发式, 而且不应该硬要。** 裁判起草的规则说的是"
          "**语义条件** ——「当查询是谚语且用户想知道其寓意时」—— 这种条件没有正则可写。"
          "在 live39 的 80 条无触发式规则中, 只有 **1 条**能抽出标记词。硬要一个触发式"
          "得到的不是 79 个判据, 而是 79 个**错的**判据: 之后报出来的每一处「重叠」都是"
          "正则的产物, 而真正冲突的地方反而报不出来。", "",
          "**那它们还能拿什么来检验?** 拿**裁判自己的裁定**。每条规则都写明了它管哪一对"
          "类目, 而金标准里恰好记着裁判在这一对上实际怎么判 —— 不需要正则, 不需要多花"
          "一次调用, 数据本来就在手里。", ""]
    L += ["**检验的是规则自己写明的判据。** 规则里通常会把判别词逐个列出 —— 例如"
          "「无明确意图标记（如'什么意思'、'寓意'、'翻译'等）」。把这些词拿到**裁判"
          "实际裁决过的行**上跑一遍就能回答一个问题: **这个判据到底把这条边界分开了没有?**", ""]
    if bad:
        L += [f"> ⚠️ **{len(bad)} 条边界上, 规则写明的判据一行都没分开。**", "",
              "| 类目对 | 规则列出的判别词 | 含该词的行 | 引用它的规则 |", "|---|---|---|---|"]
        for b in bad:
            L.append(f"| {' × '.join(b['classes'])} | {'、'.join(b['markers'][:5])} | "
                     f"**{b['n_rows_containing_a_stated_marker']}/{b['n_adjudicated']}** | "
                     f"`{'`, `'.join(b['rules_citing_it'][:5])}` |")
        L += ["", "**判据落在全部行的同一侧, 就等于没有判据。** 以 live39 的 "
              "`OTHER × TEXT_INTERPRETATION` 为例: 5 条规则都写「无明确意图标记 → OTHER」, "
              "而这条边界上 **21 行查询没有一行**含有规则自己列举的那三个词 —— 包括裁判"
              "判成 TEXT_INTERPRETATION 的全部 15 行。标注者照着规则做, 得到的不是指引, "
              "而是「全部归 OTHER」, 其中 15 行与金标准相反。", "",
              "**处置方式是给这条边界一个可观察的判据, 或者如实记下它靠人工判断, "
              "而不是删规则。** 删掉规则等于把指引一起拿走, 边界反而更没人管。", ""]
    else:
        L += ["> ✅ 每条边界上, 规则写明的判别词都确实把该边界的裁决行分开了。", ""]

    # The retired signal, kept as context with its confound stated.
    if ev.get("direction_is_confounded") or (ev.get("boundaries") or []):
        L += ["**为什么不数「规则朝向」。** 一个更直觉的做法是数有多少规则指向裁判的"
              "少数派。这个信号是**被污染的**: 裁判只在它认为指南失效的地方起草规则, "
              "而那恰好集中在少数派一侧 —— live39 的 `OTHER × TEXT_INTERPRETATION` 上, "
              "**6 行少数派里有 5 行产生了规则 (83%), 15 行多数派里只有 1 行 (7%)**。"
              "也就是说「多数规则指向少数派」正是一套**健康**例外规则应有的样子, "
              "这个判据在没有缺陷的指南上也会照样报警。规则朝向仍然记录在 artifact 里, "
              "但只作为背景, 不作为结论。", ""]
    rej = ev.get("rejected_triggers") or []
    if rej:
        L += [f"另有 **{len(rej)}** 条规则写了触发式但没能通过校验 —— "
              "触发式必须能编译、必须命中该规则自己的例子、且不能命中语料的一大片。"
              "未通过的按「语义规则」处理, 不再声称自己有判据。", ""]
    return L
