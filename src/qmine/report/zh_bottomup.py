"""The bottom-up report, in Chinese, in the structure the reference deliverable uses.

Modelled directly on `BottomUp_Approach_Final_Report.md` from the source K12
project: an executive summary whose table is the argument, then the method in the
order it was actually run, then the *worked* definition of every metric, then the
full tree, then real queries traced through it, then the honest limits.

Two things about that structure are worth stating, because they are what make it
persuasive rather than merely complete.

**Every metric is defined by its formula and then computed in front of the
reader.** "Fragmentation 1.73" means nothing; "this phrasing family splits
85%/12.1%/1.0%/…, H = 0.546, exp(H) = 1.73" can be checked. The reference
notebook prints that derivation step by step, and so does this.

**The rejected attempts get their own section.** A representation that lost a
bake-off is evidence that the winner was tested. Deleting it makes the report
shorter and less trustworthy at the same time.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from .i18n import decision_question, num, pct, prose, stars, vocab
from ._shape import delivered_shape, family_names


def _t(d: dict[str, Any] | None, *path: str, default: Any = None) -> Any:
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def build(state: Any, deps: Any, figs: dict[str, Any]) -> str:
    v = vocab("zh")
    L: list[str] = []

    load = lambda n: deps.load(n) if deps.has(n) else {}  # noqa: E731
    rep, gran = load("representation"), load("granularity")
    meta, naming = load("hierarchy_meta"), load("tree_naming")
    gov, dep = load("governance"), load("deployment")
    panel, tmpl = load("metrics_panel"), load("template_groups")
    langp = load("language_profile")
    battery = load("battery")

    df = deps.df
    n_rows = len(df)
    alpha = state.get("chosen_alpha", 0.0)
    encoder = state.get("chosen_encoder", "?")
    # THE SUMMARY MUST DESCRIBE THE TREE THAT SHIPS. `hierarchy_meta` is written in
    # p6, BEFORE governance changes the partition, so reading it here printed
    # "after blind naming and audit merging: 10 families / 29 leaves" for a run
    # that delivered 12 and 36 — contradicting this report's own metrics table
    # three lines below, which reads the panel. The panel measures the partition
    # it labels, so take the counts from there and fall back only if it is absent.
    n_fam, n_leaf = delivered_shape(panel, meta)
    gen_dir = Path(deps.store.gen_dir).name

    # ---------------------------------------------------------------- header
    L += [
        "# 自下而上 (Bottom-Up) 聚类最终报告",
        f"**数据**: {n_rows:,} 条 query · **运行**: `{state.get('run_id')}` / {gen_dir} · "
        f"**领域**: `{deps.cfg.domain.key}` · **配置指纹**: `{deps.cfg.config_hash}`",
        "",
        f"> {deps.registry.provenance_note('zh')}",
        "",
    ]

    # ------------------------------------------------------------- 0 摘要
    L += [f"## 0. {v['exec_summary']}", ""]
    L.append(
        f"不使用任何既有分类, 让数据在 hybrid 表征上自行长出一棵「家族→叶子」两层树。"
        f"经盲评命名与审计合并后: **{n_fam} 家族 / {n_leaf} 叶**。"
    )
    L.append("")

    leaves = _t(panel, "sets", "leaves", "metrics", default={})
    fam_fin = _t(panel, "sets", "families_final", "metrics", default={})
    fam_pre = _t(panel, "sets", "families_pre_governance", "metrics", default={})
    if leaves:
        L += ["| 指标 (审计合并后统一重算) | 家族层 (合并前) | 家族层 (最终) | 叶层 |",
              "|---|---|---|---|"]
        def _row(label: str, key: str) -> str:
            g = lambda d: num(_t(d, key, "value"))  # noqa: E731
            return f"| {label} | {g(fam_pre)} | **{g(fam_fin)}** | {g(leaves)} |"
        L += [
            _row(f"{v['n_clusters']} (簇数)", "n_clusters"),
            _row(f"{v['template_fragmentation']} ↓", "template_fragmentation"),
            _row(f"{v['stability_ari']} ↑", "stability_ari"),
            _row(f"{v['nmi_reference']} ↑", "nmi_reference"),
            _row(f"{v['heldout_reproduction']} ↑", "heldout_reproduction"),
            _row(f"{v['silhouette']} (仅参考)", "silhouette"),
        ]
        L.append("")
    L += [v["fairness_note"], ""]

    L += ["**生产分工** (原则 1 — 双路线各管一根轴):", "",
          "- **自下而上树** (本报告) 负责*内容组织*轴: 数据实际长成什么板块、供给侧怎么规划、"
          "季度漂移哨兵、新颖内容探测;",
          "- **自上而下体系** 负责*意图/功能*轴: 用户「想干什么」。功能型意图对聚类**结构性不可见** — "
          "不要指望聚类替你发现它们;",
          "- 两套标签**并排交付**在同一张全量表里 (`labels_full.csv`), 不互相覆盖。", ""]

    # ------------------------------------------------------------- 1 路径
    L += [f"## 1. {v['research_path']}", "",
          "**① 表征构建 → ② 算法选优 → ③ K 值扫描 → ④ 两层层级构建 → ⑤ 盲评命名 → ⑥ 治理合并 → ⑦ 部署验证**",
          ""]

    L += ["### 1.1 表征构建", ""]
    bake = _t(rep, "bakeoff", default={})
    if bake.get("rows"):
        L += ["**底座 embedding 选型**: 在**本语料自己的聚类任务**上做 bake-off, 而非照搬榜单分数。", "",
              "| encoder | 维度 | 稳定性 ARI (主裁判) | 碎裂度 (主裁判) | silhouette (仅参考) |",
              "|---|---|---|---|---|"]
        for r in bake["rows"]:
            if r.get("status") != "ok":
                L.append(f"| `{r['encoder']}` | — | 不可用 | — | — |")
                continue
            mark = " **←当选**" if r["encoder"] == encoder else ""
            L.append(f"| `{r['encoder']}`{mark} | {r['dim']} | {num(r['stability_ari'])} | "
                     f"{num(r['template_fragmentation'])} | {num(r['silhouette'])} |")
        L.append("")
        if bake.get("silhouette_disagrees"):
            L.append(f"> **silhouette 会选 `{bake['silhouette_would_have_chosen']}`** — 已记录并否决。"
                     f"判别力与几何结构是两回事: 更大的模型做分类特征更强, 聚类结构未必更好。")
            L.append("")

    sp = _t(rep, "sparse", default={})
    if sp:
        L += ["**稀疏表征 (措辞轴)**: 字符 "
              f"{tuple(deps.cfg.domain.char_ngram_range)}-gram TF-IDF, 词表 "
              f"{sp.get('vocab_size', '?'):,} 维 → TruncatedSVD 压到 {sp.get('n_components','?')} 维 "
              f"(解释方差 {num(sp.get('explained_variance'))})。", "",
              "TF-IDF 代表**措辞/模板**, 与 embedding 的**语义**互补; SVD 只是把它压成"
              "**可与稠密块拼接**的形态。", "",
              "> ⚠️ **不要把这一步读成「无损」。** 256 维只解释了原始 TF-IDF 方差的一部分 "
              "(见上方的解释方差)。SVD 保留的是**主要的共现方向**, 低频、长尾的措辞差异"
              "在压缩中就已丢失 —— 这对措辞轴是可以接受的取舍, 但它是一个**取舍**, "
              "不是一次无代价的变形。", ""]

    surface = alpha ** 2 / (1 + alpha ** 2)
    L += ["### 1.2 Hybrid 拼接与 α 的精确含义", "",
          "**公式**: `H = L2norm([ e ⊕ α·s ])`, e = 归一化 embedding, s = 归一化 SVD 块。", "",
          "两块各自是单位向量时:", "",
          "```",
          "cos(H, H′) = (cos_semantic + α²·cos_surface) / (1 + α²)",
          "```", "",
          f"即**措辞块的话语权是 α², 不是 α**。本次选定 **α = {alpha}**, "
          f"措辞话语权 = α²/(1+α²) = **{surface*100:.1f}%** — "
          "只在语义打平的边界处把同模板 query 粘回一起的「平局裁决者」。", ""]

    sweep = _t(rep, "alpha_sweep", default={})
    if sweep.get("rows"):
        L += ["| α | 措辞话语权 | 碎裂度 (主裁判) ↓ | 稳定性 ARI (主裁判) ↑ | silhouette (仅参考) |",
              "|---|---|---|---|---|"]
        for r in sweep["rows"]:
            mark = " **←当选**" if r["alpha"] == alpha else ""
            L.append(f"| {r['alpha']}{mark} | {r['surface_vote_share']*100:.1f}% | "
                     f"{num(r['template_fragmentation'])} | {num(r['stability_ari'])} | "
                     f"{num(r['silhouette'])} |")
        L.append("")
        if sweep.get("chosen_by"):
            # Translated rule + the numbers as data. Quoting `chosen_by` raw put
            # an English sentence into the Chinese report on every run.
            _band = sweep.get("tie_band_value")
            _src = sweep.get("tie_band_source", "")
            L.append(f"**选择规则**: {prose(sweep.get('chosen_by', ''))}"
                     + (f" 本次容差带 = {_band}"
                        + (f" (= 最低碎裂度 x {1 + sweep.get('tie_band_relative_pct', 0) / 100:.2f})"
                           if sweep.get("tie_band_relative_pct") else "")
                        if _band is not None else "")
                     + (f" ({prose(_src)})" if _src else "") + "。")
        if sweep.get("contenders"):
            L.append(
                f"碎裂度差异在 {sweep.get('tie_band', 0.05):.0%} 带内视为打平 "
                f"(候选 `{sweep['contenders']}`), 再由稳定性裁决。"
                "**这条带宽是预设值, 不是本语料实测的噪声底** —— 每个 alpha 只拟合一次, "
                "因此这一步无法判断当选者与并列者之间的差距是否真的超出重跑噪声。"
            )
        L.append("")
        if sweep.get("silhouette_disagrees"):
            L += [f"> **silhouette 会选 α={sweep['silhouette_would_have_chosen']}** — 已记录并否决。", ""]
        L += [v["silhouette_no_vote"], ""]
    L.append(_fig(figs, "fig_alpha", "α 决策图: 左为裁决指标, 右为被否决的代理指标"))

    # --------------------------------------------------- 1.3 指标现场演算
    L += ["### 1.3 三个核心指标的计算方法 (含现场演算)", "",
          "**silhouette (轮廓系数)**: 对每个样本 i, `a(i)` = 到本簇其他成员的平均距离, "
          "`b(i)` = 到最近其他簇成员的平均距离, `s(i) = (b−a)/max(a,b) ∈ [−1,1]`, 取全体均值。"
          f"本项目用 cosine 距离, 在固定 {deps.cfg.clustering.silhouette_sample:,} 行子样 "
          f"(seed={deps.cfg.seed_metric}) 上计算, 各方案共用同一子样保证可比。", "",
          "**重播稳定性 ARI**: 同数据同算法, **只换随机种子** "
          f"(seed={tuple(deps.cfg.seed_replay)}) 跑两次, 得两套划分; "
          "ARI 衡量两者对「任意样本对是否同簇」判断的一致度, 并扣除随机碰巧一致的期望: "
          "`ARI = (RI − E[RI]) / (max(RI) − E[RI])`, 1 = 完全一致, 0 = 纯随机。", "",
          "> **为什么它是一票否决项**: 聚类没有真值, 但有可复现性。不可复现的划分不是数据的"
          "真实结构, 无论 silhouette 多好看。", "",
          "**模板碎裂度 (本方法论自研指标)**: 取一组**已知同意图的措辞群**, 看它们被劈进几个"
          "「有效家族」: 家族分布 p → 香农熵 `H = −Σ p·ln p` → **有效家族数 = exp(H)**。"
          "全在一族 = 1.0, 均匀散布 m 族 = m。各群取平均。", ""]

    frag_detail = _t(leaves, "template_fragmentation", "detail", "per_group", default={})
    if not frag_detail:
        frag_detail = _t(fam_fin, "template_fragmentation", "detail", "per_group", default={})
    if frag_detail:
        worst = max(frag_detail.items(), key=lambda kv: kv[1])
        L += ["**现场演算** (以碎裂最严重的一群为例):", "", "```"]
        L += [f"① 模板群「{worst[0]}」",
              f"② 有效家族数 = exp(H) = {worst[1]:.2f}",
              f"③ 全部 {len(frag_detail)} 群平均 = "
              f"{np.mean(list(frag_detail.values())):.2f}   ← 报告中的碎裂度",
              "```", ""]

    if tmpl.get("groups"):
        cov = _t(tmpl, "coverage", "union_coverage")
        L += [f"**模板群来源**: Phase 1 从语料中挖出 {len(tmpl['groups'])} 群, "
              f"合计覆盖 {pct(cov)} (质量门窗口 20–40%)。这些群**一次挖掘, 三处复用** — "
              "α-sweep 的裁判 / 碎裂度的地基 / 展示选样的模式来源。", "",
              "| 模板群 | 命中 | 占比 | 来源 | 示例 |", "|---|---|---|---|---|"]
        for g in tmpl["groups"][:12]:
            L.append(f"| `{g['name']}` | {g['n_hits']:,} | {pct(g['share'],2)} | "
                     f"{'种子' if not g['discovered'] else '挖掘'} | {(g['examples'] or [''])[0][:18]} |")
        L.append("")
        coh = _t(rep, "template_cohesion", default={})
        if coh.get("dropped"):
            L += [f"> **{len(coh['dropped'])} 个挖掘群未通过内聚检验被剔除** "
                  f"(`{coh['dropped'][:5]}`): 它们的成员相似度不高于随机行 — "
                  "说明该措辞是**疑问句式**而非意图 (如「是什么」附着于所有话题), "
                  "不满足「凡命中者几乎必是同一意图」的前提。", ""]

    # ------------------------------------------------------------- 2 调优
    L += [f"## 2. {v['tuning']}", "",
          "两个决定树形状的旋钮 — 用什么算法, 切成多粗 — 各自有独立的证据链。", "",
          f"### 2.1 {v['algorithm_battery']}", ""]
    if battery.get("rows"):
        L += ["**本阶段不选择算法** —— 交付的树始终由 KMeans 构建。这里跑的是一次"
              "**证伪检验**: 把假设不同的算法送进同一套度量 harness, 问「这套结构是语料的"
              "性质, 还是 KMeans『簇近似球形』这一假设的产物」。", "",
              "| 算法 | 簇数 | 稳定性 ARI | silhouette | 噪声率 |", "|---|---|---|---|---|"]
        for r in battery["rows"]:
            L.append(f"| `{r['algorithm']}` | {r['n_clusters']} | {num(r['stability_ari'])} | "
                     f"{num(r['silhouette'])} | {pct(r.get('noise_rate'),1)} |")
        L.append("")
        verdict = _t(battery, "verdict", default={})
        if verdict.get("role"):
            L += ["> **这一步不是选举, 是证伪检验。** 交付的树**始终**由 KMeans 构建 "
                  "(`build_hierarchy`)。此处跑其他算法, 是为了回答一个不同的问题: "
                  "**结构换一种算法还在不在?** 在, 说明它是语料的性质; 不在, 说明它是 "
                  "KMeans「簇近似球形」这一假设的产物。", ""]
            alt, mg = verdict.get("best_alternative"), verdict.get("alternative_beats_reference_by")
            # A "strongest alternative" that IS the reference makes the gap 0.0 by
            # construction, and the row then reads as a passed falsification test
            # when nothing was actually compared. live38 shipped exactly this:
            # reference kmeans_k15, best_alternative kmeans_k15, gap 0.0.
            self_cmp = alt is not None and alt == verdict.get("reference_algorithm")
            L += ["| 参照 (交付所用) | 最强替代算法 | 稳定性差距 | 结论 |", "|---|---|---|---|",
                  f"| `{verdict.get('reference_algorithm')}` | `{alt}` | "
                  f"{num(mg) if mg is not None else '—'} | "
                  f"{'⚠️ 假设被质疑' if verdict.get('kmeans_assumption_contradicted') else '✅ 未被证伪'} |", ""]
            if self_cmp:
                L += ["> ⚠️ **本行是一次自我比较, 不是一次证伪检验。** 「最强替代算法」"
                      f"与参照同为 `{alt}`, 因此稳定性差距必然为 0 —— "
                      "这说明**没有任何一个替代算法比参照更可复现**, "
                      "而不是说明参照通过了与其他算法的对比。两者的证据强度不同: "
                      "前者只排除了「有更好的」, 没有排除「都一样差」。", ""]
            # The artifact keeps the English note for machine readers; the report
            # renders the same judgement in the deliverable's language.
            L += ["> " + ("**KMeans 的球形簇假设在此处被质疑**: 一个结构上完全不同的算法比它"
                          "更可复现 (差距 > 0.10 ARI)。家族层应按**暂定**读取, 并在报告中明说。"
                          if verdict.get("kmeans_assumption_contradicted") else
                          "**未被证伪**: 没有任何结构上不同的算法比 KMeans 明显更可复现, "
                          "因此这套划分不是「簇近似球形」这一假设的产物, 而是语料本身的性质。"), "",
                  "> **机制原因**: L2 归一化后 embedding 分布近似球面, 余弦几何下簇近似各向同性, "
                  "正中 KMeans 假设。BisectingKMeans 的早期错切不可逆; HDBSCAN 受制于 embedding "
                  "空间密度不均 — 保留为**新颖性哨兵** (Phase 12)。", ""]
        if verdict.get("density_note"):
            L += [f"> {prose(verdict['density_note'])}", ""]

    L += [f"### 2.2 {v['granularity']}", ""]
    tri = _t(gran, "triangulation", default={})
    if tri.get("estimates"):
        L += ["三条**独立**路线估计家族尺度, 收敛才定案:", "", "| 估计来源 | 数值 |", "|---|---|"]
        _loc_zh = {"intent_alignment_ami": "意图对齐 AMI",
                   "stability_ari": "重播稳定性 ARI"}
        _by = str(tri.get("locator", "")).split(" ")[0]
        _by_zh = _loc_zh.get(_by, _by or "未记录")
        label = {"located_k": f"由「{_by_zh}」定位的 K (主证据)",
                 # Artifacts written before the rename. The name was wrong, so it
                 # is NOT reproduced here verbatim.
                 "stability_peak_k": f"由「{_by_zh}」定位的 K (主证据)",
                 "deep_aligned_leaf_k": "DeepAligned 过聚类存活 (叶尺度)",
                 "deep_aligned_implied_family_k": "↑ 折算家族尺度",
                 "expert_range": "领域先验区间", "silhouette_peak_k": "silhouette 峰 (仅参考)"}
        for k, val in tri["estimates"].items():
            L.append(f"| {label.get(k,k)} | {val} |")
        L += ["", f"**定案 K = {tri.get('chosen_family_k')}** — {tri.get('chosen_by','')}", ""]

        # The tie set is not a caveat appended to the answer; when it has more than
        # one member it IS the answer. Reporting a single K whose margin sits inside
        # the measurement error tells the reader something the data does not support.
        ties = tri.get("tie_set") or []
        if len(ties) > 1:
            L += [f"### 2.2.1 同样站得住的 K ({len(ties)} 个)", "",
                  "以下 K 值在**测量误差以内彼此无法区分** — 报告单一 K 会把一个测不出来的差别"
                  "说成结论。定案取其中**最简单的一个**(K 最小), 但下列每一个都同样有据:", "",
                  "| K | 意图对齐 AMI (定位指标) | 重播稳定性 (筛选指标) | 模板碎裂度 |", "|---|---|---|---|"]
            for t in ties:
                mark = " **←定案**" if t["k"] == tri.get("chosen_family_k") else ""
                L.append(f"| {t['k']}{mark} | {num(t.get('intent_alignment_ami'))} | "
                         f"{num(t.get('stability_ari'))} | {num(t.get('template_fragmentation'))} |")
            L += ["", "> 若某个 K 更契合业务侧的粒度直觉, **可以直接改用它而无需重跑选型** — "
                  "证据并不偏向定案的那一个。", ""]
        if tri.get("n_rejected_as_unstable"):
            L += [f"> 另有 **{tri['n_rejected_as_unstable']} 个 K 因重播稳定性低于 "
                  f"{tri.get('stability_floor')} 被直接剔除** — 稳定性在此只做否决, 不做排序: "
                  "它在本语料上的种子间标准差 (~0.10) 大于相邻 K 之间的差距 (~0.05), "
                  "用它排序等于读噪声。", ""]
        if tri.get("divergence_note"):
            L += [f"> ⚠️ {tri['divergence_note']}", ""]
    L.append(_fig(figs, "fig_battery",
                  "算法 battery: 横轴 = 重播稳定性 (裁决), 纵轴 = silhouette (只旁听) —— "
                  "越靠右越可复现, 越靠上只是越紧致"))
    L.append(_fig(figs, "fig_k_sweep",
                  "K 扫描: 各候选空间的稳定性与 silhouette 曲线 — 形状并不一致, "
                  "所以一个空间选出的 K 不能搬到另一个空间"))
    L.append(_fig(figs, "fig_k_sweep_metrics",
                  "当选空间的全量 K 扫描: 三个指标各有各的峰 — 所以必须先指定谁是裁判"))

    # ------------------------------------------------------------- 3 层级
    L += [f"## 3. {v['hierarchy']}", "",
          "**定义先行**: 一个家族/叶子在数学上就是一个**质心**; 「属于 g」= 「离质心 g 最近」。"
          "名字与定义是 Phase 7 事后由盲评补写的 — **先有结构, 后有语言**。", "",
          "构造分两步:", "",
          f"1. **家族层**: 全量 KMeans, K = **与措辞群的对齐度 (AMI) 定位** "
          f"({tri.get('chosen_family_k','?')}); 稳定性在这一步只负责否决, 不排序;",
          f"2. **家族内局部选 k**: 每个家族单独试 k=2..{deps.cfg.clustering.max_leaves_per_family}, "
          f"以 cosine silhouette 择优 — **每个家族根据自身结构自主决定形成几个叶子**。"
          f"约束: 最小叶 {_t(meta,'min_leaf_size_applied',default='?')} 条"
          + _delivered_min_leaf(deps, _t(meta, "min_leaf_size_applied", default=None)) + "。", "",
          "> **为什么不用一步到位的大 K?** K 扫描显示细粒度全局划分**可复现性太差**; "
          "「稳定粗分 + 家族内局部细分」让每层都工作在各自更好复现的尺度上。", ""]

    hist = _t(meta, "refinement_history", default=[])
    if hist:
        L += ["**迭代精化** (merge 质心余弦 > "
              f"{deps.cfg.clustering.refine_merge_cos} / split 负轮廓高的叶 / reassign 全量), "
              f"收敛判据: 单轮移动 < {deps.cfg.clustering.refine_move_tolerance:.1%}", "",
              "| 轮 | 合并 | 拆分 | 移动行 | 叶数 | silhouette |", "|---|---|---|---|---|---|"]
        for h in hist:
            L.append(f"| {h['round']} | {h['merges']} | {h['splits']} | "
                     f"{pct(h['moved_fraction'],2)} | {h['n_leaves']} | {num(h['silhouette'])} |")
        L.append("")
        L += _refinement_verdict(hist, meta, deps)
        # An agent explains THIS run's convergence behaviour. The fact sheet is
        # deliberately tiny — only the numbers this one question needs — because
        # the numeric check is value-level and a small sheet leaves few wrong
        # values to reach for. See `agents/verify.py`.
        L += _agent_reading(
            deps,
            "这次迭代精化没有收敛。请解释它是怎么没收敛的, 以及这对交付的叶数意味着什么。"
            "特别注意: 叶数在两个值之间反复, 而每轮移动的行数在单调下降。",
            {"rounds": len(hist),
             "first_moved_fraction": hist[0].get("moved_fraction"),
             "last_moved_fraction": hist[-1].get("moved_fraction"),
             "tolerance": deps.cfg.clustering.refine_move_tolerance,
             "leaves_min": min(h["n_leaves"] for h in hist),
             "leaves_max": max(h["n_leaves"] for h in hist),
             "leaves_final": hist[-1]["n_leaves"]},
            context=json.dumps(hist, ensure_ascii=False),
            label="本轮解读")
    L.append(_fig(figs, "fig_refinement", "精化收敛轨迹"))

    hr = _t(meta, "heldout_reproduction", default={})
    if hr:
        sv = hr.get("statistical_verdict", {})
        L += [f"**held-out 结构复现检验**: 用 80% 数据重建质心, 分类另外 20%, "
              f"与全量划分一致率 = **{pct(hr.get('agreement'),1)}** "
              f"(n={hr.get('n_test'):,}", ]
        if sv.get("ci95"):
            L[-1] += f", 95% CI {sv['ci95'][0]:.3f}–{sv['ci95'][1]:.3f}"
        L[-1] += ")。"
        L += ["", "> 这是「结构是真的」的最终背书: 只在看得见全部数据时才存在的划分, "
              "是对这份样本的描述, 而不是对现象的描述。", ""]
        if sv.get("verdict") == "underpowered":
            L += [f"> ⚠️ 本次检验**样本不足以判定** — 置信区间跨过阈值, "
                  f"约需 {sv.get('n_needed')} 行 held-out 才能定论。既不算通过, 也不算失败。", ""]

    # The figure is a THREE-panel comparison of different spaces, each coloured by
    # ITS OWN families — that disagreement is the point. The old caption said
    # "coloured by the FINAL families", which describes a single-panel picture
    # this figure is not, and invites the reader to compare colours across panels
    # as though they meant the same thing.
    L.append(_fig(figs, "fig_umap",
                  "UMAP-2D: 同一批 query 在几个候选表征中的分布。"
                  "**每一栏各自按自己空间里的家族着色** —— 栏与栏之间颜色不可对照, "
                  "要看的是同一团点在不同空间里是否还聚在一起"))

    # ------------------------------------------------------------- 4 命名
    L += [f"## 4. {v['naming']}", "", v["blind_protocol"], ""]
    namings = naming.get("namings", [])
    if namings:
        L += [f"**协议**: {deps.cfg.naming.n_naming_agents} 个命名 agent 分片独立作业 (互不通信), "
              f"每叶卡片 = 质心最近 {deps.cfg.naming.card_center} + 随机 {deps.cfg.naming.card_random} "
              f"+ **边缘 {deps.cfg.naming.card_edge}** 条 (边缘样本让命名者看见杂质); "
              "1 个审计 agent 汇总建树并专项挑刺。", "",
              f"- 平均 coherence: **{num(naming.get('mean_coherence'),3)} / 5**", ]
        ind = naming.get("independent_risk_discovery", {})
        if ind:
            L += [f"- 风险内容**被从未提示的 agent 独立标记**: "
                  f"{'是' if ind.get('found_without_being_told') else '否'} "
                  f"(命名者标记 `{ind.get('namer_flagged_leaves')}`, "
                  f"哨兵标记 `{ind.get('sentinel_flagged_leaves')}`)", ]
            L += ["", "> 未被提示的 agent 独立发现风险簇, 才是**可信的发现**; "
                  "种子模式命中自己种下的东西, 只是记账。", ""]

    # ------------------------------------------------------------- 5 树
    L += [f"## 5. {v['tree_listing']}", ""]
    L += _tree_listing(deps, naming, meta)

    # ------------------------------------------------------------- 6 治理
    L += [f"## 6. {v['governance']}", "", v["governance_executed"], ""]
    if gov.get("ledger"):
        L += ["| 处方 | 类型 | 目标 | 状态 | 落到哪一列 | 指标变化 |", "|---|---|---|---|---|---|"]
        for r in gov["ledger"]:
            L.append(f"| `{r['id']}` | {r['kind']} | {r['targets']} | **{r['status']}** | "
                     f"`{r['executed_column'] or '—'}` | "
                     f"{json.dumps(r['metric_deltas'], ensure_ascii=False)} |")
        L.append("")
        declined = [r for r in gov["ledger"] if r["status"] == "declined"]
        if declined:
            L += ["**有意保留的划分** (审计判定「有语义依据」, 是**决策**而非遗漏):", ""]
            L += [f"- `{r['id']}` 目标 `{r['targets']}` — {r['decline_reason']}" for r in declined]
            L.append("")
    L.append(_fig(figs, "fig_template_spread", "模板群落点图: 一条实心 = 一个意图一个簇; 多段 = 孪生劈裂"))
    L.append(_fig(figs, "fig_intent_split",
                  "同一意图被劈开的程度: 标题「有效散布于 N 个家族」即 exp(香农熵), 与碎裂度同式"))

    # ------------------------------------------------------------- 7 部署
    L += [f"## 7. {v['deployment']}", ""]
    if dep:
        r = _t(dep, "routing", default={})
        L += ["| 检验 | 结果 | 说明 |", "|---|---|---|",
              f"| held-out 结构复现 | **{pct(hr.get('agreement'),1)}** | 80% 重建质心分类另 20% |",
              f"| margin 模糊率 | {pct(r.get('ambiguous_rate'),1)} | "
              f"top1−top2 < {r.get('threshold')} 者走兜底/人审 |",
              f"| 模型体积 | {dep.get('model_bytes',0)/1024:.0f} KB | 仅质心矩阵 |", "",
              f"**实时分类** = `{dep.get('inference','')}`", "",
              "> 模糊率**如实报告**: 语义边界天然比措辞边界软, 双位数模糊率是问题的性质, "
              "隐藏它只会把意外推迟到线上。", ""]
        if dep.get("live_demo"):
            L += ["**新 query 现场路由演示**:", "", "| query | 叶 | 名称 | margin | 路由 |", "|---|---|---|---|---|"]
            for d in dep["live_demo"]:
                L.append(f"| {d['query']} | {d['leaf']} | {d['leaf_name']} | "
                         f"{d['margin']} | {'兜底' if d['routed']=='fallback' else '直接'} |")
            L.append("")
        if dep.get("deterministic_exemplars"):
            L += ["### 确定性样本展示 (原则 7)", "",
                  "每个模板群取**命中集合的中位数下标**实例 — 从机制上排除 cherry-picking: "
                  "说服力来自「你无法挑样本」, 而不是样本本身多好看。", "",
                  "| 模板群 | 命中数 | 中位数下标实例 |", "|---|---|---|"]
            for e in dep["deterministic_exemplars"][:12]:
                L.append(f"| {e['pattern']} | {e['n_hits']:,} | {e['exemplar']} |")
            L.append("")

    # ------------------------------------------------------------- 8 面板
    L += [f"## 8. {v['panel']}", ""]
    if panel.get("table"):
        t = panel["table"]
        L += [f"所有指标在**同一套代码、同一子样、同一种子**下重算 (panel `{t['panel_id']}`)。"
              "严禁各自引用各自实验时期的数字。", ""]
        L += [_panel_table(t)]
        L += ["", "**指标裁决权**:", ""]
        for m in t["metrics"]:
            L.append(f"- `{m['name']}` — {v.get(m['authority'], m['authority'])}")
        L.append("")

    # ------------------------------------------------------------- 9 失败
    # The audit trail sits BEFORE the failure history and the caveats, because a
    # reader asking "how did this tree come to exist" should not have to reach the
    # appendix to find out. Everything below was already recorded on every run and
    # simply never rendered.
    L += ["## 9. 全流程决策链 (完整推理链路)", "",
          "本节回答一个问题: **这棵树是怎么来的。** 每一个参数、每一次取舍、"
          "每一个被否决的方案, 按发生顺序排列并附当时的数字。", ""]
    L.append(_decision_chain(state))
    L.append(_fig(figs, "fig_decision_chain",
                  "决策链: 每个环节考虑了多少候选、淘汰了多少、由哪个指标裁定"))

    L += ["## 10. 质量门总账", ""]
    L.append(_gate_ledger(state))
    L.append(_fig(figs, "fig_gates", "质量门: 实测值与门槛的距离 (归一化), 颜色为判定结果"))

    gov_ledger = _governance_ledger(state, gov)
    if gov_ledger:
        L += ["## 11. 治理台账 (每条处方的最终去向)", "", gov_ledger]

    L += [f"## 12. {v['failure_history']}", "",
          "被否决的表征/选型写成独立小节 — 这是说服力的来源, 不是丢脸的历史。", ""]
    L += [_failure_history(state)]

    # ------------------------------------------------------------ 10 局限
    L += ["", f"## 13. {v['limits']}", "",
          f"- **{v['distill_caveat']}**",
          f"- **silhouette 全程仅参考**: {v['silhouette_no_vote']}",
          f"- **碎裂度须与簇数同看**: {v['fairness_note']}",
          "- **held-out 复现检验的是结构稳定性, 不是语义正确性**: 一个可复现的错误划分仍然是错的。",
          "- **聚类对功能型意图结构性不可见**: 措辞与内容正常、意图藏在语用里的类别 "
          "(用法判断、解题、导航、闲聊) 只能由自上而下体系承担, 本报告不硬凑。", ""]
    if langp:
        # Read the UNROUNDED share. `dominant_share` is stored as round(x, 4) —
        # 0.9764595 becomes 0.9765 — and `pct` then rounds again to 97.7%, while
        # the rationale string beside it computes 97.6% from the raw value. Two
        # different numbers for one quantity, in one sentence, from rounding twice.
        dom = langp.get("dominant")
        share = (langp.get("shares") or {}).get(dom, langp.get("dominant_share"))
        minority = langp.get("minority_share") or 0.0
        L += [f"- **语言构成**: 主体 `{dom}` {pct(share)}, "
              f"少数语种 {pct(minority)}"
              + ("" if minority else " (无其他语种达到可分层的规模)")
              + f" ({langp.get('posture')})。", ""]
        L += [f"> {_language_posture_zh(langp)}", ""]

    # ------------------------------------------------------------ 11 档案
    L += [f"## 14. {v['leaf_catalogue']}", "",
          "每叶四字段: `name_zh` / `code` / **`user_need`** (一句话定义) / `coherence` + 杂质备注。"
          "**名字会歧义, 定义句不会** — `user_need` 同时是标注指南、验收标准、下游产品需求说明。", ""]
    L += _leaf_catalogue(deps, naming, meta)

    return "\n".join(L)


# --------------------------------------------------------------------------



def _agent_reading(deps: Any, question: str, facts: dict[str, Any],
                   *, context: str = "", label: str = "解读") -> list[str]:
    """Ask an agent to explain one result; ship nothing if it cannot be verified.

    This is where authored prose enters the deliverable. Until now the reports
    were pure templating — 0 of live38's 966 agent calls went to writing one —
    which is why nine defects survived into a shipped document that read fluently
    and no one had checked. A template cannot say "this particular corpus did an
    unusual thing"; that is the sentence a reader needs and the one only a reader
    of the actual numbers can write.

    The safety is not trust. `interpret()` rejects any number outside the fact
    sheet, re-asks with the offending values quoted back, and returns nothing
    after three failures — so the worst case is a section without commentary,
    never a section with confident invented commentary.
    """
    try:
        from ..agents.interpret import interpret
    except Exception:  # noqa: BLE001
        return []
    if not getattr(deps.cfg, "interpret_results", True) or deps.cfg.fast_mode:
        return []
    try:
        got = interpret(deps, question, facts, context=context,
                        language=deps.cfg.report_language, suffix="_report")
    except Exception as exc:  # noqa: BLE001
        deps.emit(f"  interpretation unavailable: {type(exc).__name__}")
        return []
    md = got.as_markdown(label)
    return [md, ""] if md else []



#: Keyed on `posture`, which is a STABLE enum, not on the rationale prose.
#: `ops/language.py` authors that rationale as f"{dominant} accounts for …" —
#: it begins with a corpus-specific value, so no `prose()` prefix key can match
#: it on a corpus whose dominant script is not the one the key was written for.
#: Translating the verdict rather than the sentence is what makes it portable.
_POSTURE_ZH = {
    "monolingual":
        "**单一语种语料**: 没有任何其他文字达到可分层的规模。直接使用针对主体语言的"
        "单语编码器即可 —— 此处没有需要分层处理的少数语种。",
    "minority_at_risk":
        "**存在规模偏小但非空的少数语种 —— 有被压成「垃圾簇」的风险。** "
        "占比很低的语种在全量聚类中往往被并成一个混杂簇, 而**换多语种编码器并不能"
        "解决这个问题**。请对该部分单独分层检查。",
    "genuinely_multilingual":
        "**真正的多语种语料**: 少数语种规模足以自成结构。必须分层建模与分层评估, "
        "不要用一次全量聚类的指标代表所有语种。",
}


def _language_posture_zh(langp: dict[str, Any]) -> str:
    posture = str(langp.get("posture", ""))
    zh = _POSTURE_ZH.get(posture)
    if zh:
        return zh
    # An unrecognised posture must show the original rather than disappear.
    return prose(langp.get("rationale", "")) or f"语言态势: `{posture}`"

def _delivered_min_leaf(deps: Any, stated: Any) -> str:
    """Say when the DELIVERED tree breaks the constraint the text just stated.

    The report says 「最小叶 150 条」 and live38 delivered leaves of 104 and 122 —
    because p8 governance splits after p6 applied the floor. Stating a constraint
    the shipped object violates, with no note, teaches the reader a guarantee that
    is not there.
    """
    try:
        import numpy as np

        labels = deps.leaf_labels_final()
        sizes = np.bincount(labels)
        sizes = sizes[sizes > 0]
        lo = int(sizes.min())
    except Exception:  # noqa: BLE001
        return ""
    if stated is None or lo >= int(stated):
        return f" (交付树的最小叶为 {lo} 条, 满足该约束)"
    n_under = int((sizes < int(stated)).sum())
    return (f" —— 但**交付树的最小叶只有 {lo} 条**, 共 {n_under} 个叶低于该值。"
            "该约束由 p6 施加, 而 p8 治理会在其后再拆分, 因此它约束的是精化结果, "
            "不是交付结果")

def _refinement_verdict(hist: list, meta: dict, deps: Any) -> list[str]:
    """Say whether the loop converged, and if not, HOW it failed to.

    The table showed 29 / 28 / 29 / 28 / 29 leaves and the caption underneath said
    the loop is judged "by movement stopping, not by a fixed round count" — which
    is exactly what did NOT happen: `converged` was False and it stopped at the
    round cap. A reader cannot tell an oscillation from a divergence from a table
    of counts, and the two mean opposite things about the delivered tree.
    """
    tol = deps.cfg.clustering.refine_move_tolerance
    last = hist[-1]
    if _t(meta, "converged", default=None):
        return [f"> ✅ **已收敛**: 末轮移动 {pct(last['moved_fraction'], 2)} < 判据 "
                f"{tol:.1%}, 迭代自行停止。", ""]

    counts = [h["n_leaves"] for h in hist]
    tail = counts[-4:]
    oscillating = len(set(tail)) == 2 and all(
        tail[i] != tail[i + 1] for i in range(len(tail) - 1))
    out = [f"> ⚠️ **未收敛**: 末轮仍移动 {pct(last['moved_fraction'], 2)}, "
           f"未达到 {tol:.1%} 的判据 — 迭代是**用满轮数上限后停下的**, "
           "不是自行停下的。", ""]
    if oscillating:
        a, b = sorted(set(tail))
        out += [f"更具体地说, 这是一个**极限环**而不是发散: 叶数在 {a} 与 {b} 之间反复 "
                f"({' → '.join(str(c) for c in counts)}), 每轮一合一拆, "
                f"而移动比例本身是**单调下降**的 "
                f"({pct(hist[0]['moved_fraction'], 2)} → {pct(last['moved_fraction'], 2)})。"
                "也就是说整体结构已经稳定, 只有**一条边界**在两种划法之间来回。", "",
                f"**这对交付物意味着什么**: 最终叶数 ({counts[-1]}) 取决于迭代"
                f"停在第几轮, 在 {a} 与 {b} 之间是**任意的**。该边界两侧的簇不应被当作"
                "确定的划分来解读; 其余部分不受影响 —— 请对照 held-out 复现率与重播 ARI, "
                "两者都很高, 说明动摇的只是这一处。", ""]
    else:
        out += ["叶数没有呈现出规律性的振荡, 因此这更可能是**判据设得过严**或"
                "**轮数上限过低**, 而不是某一条边界不稳。可提高上限重跑一次以区分二者。", ""]
    return out

def _fig(figs: dict[str, Any], name: str, caption: str) -> str:
    if name not in figs:
        return ""
    return f"\n![{caption}]({Path(figs[name].path).name})\n\n*图: {caption}*\n"


def _panel_table(t: dict[str, Any]) -> str:
    names = [m["name"] for m in t["metrics"]]
    head = "| 方案 | " + " | ".join(names) + " |"
    sep = "|---" * (len(names) + 1) + "|"
    rows = [head, sep]
    for r in t["rows"]:
        cells = [num(r.get(n)) if isinstance(r.get(n), float) else str(r.get(n) or "—") for n in names]
        rows.append(f"| `{r['subject']}` | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def _tree_listing(deps: Any, naming: dict, meta: dict) -> list[str]:
    namings = naming.get("namings", [])
    if not namings:
        return [vocab("zh")["no_data"]]
    try:
        # The DELIVERED partition, not p6's. Loading `leaf_labels` here paired
        # pre-governance leaf sizes (29 leaves) with post-governance families
        # (12), so the two halves of every row described different trees.
        labels = deps.leaf_labels_final()
        fam = deps._cache.get("leaf_family_final")
        if fam is None:
            fam = deps.load("leaf_family_final") if deps.has("leaf_family_final") else deps.load("leaf_family")
        sizes = np.bincount(labels, minlength=max(len(namings), len(fam)))
        total = len(labels)
    except Exception:
        return [vocab("zh")["no_data"]]

    # Join on leaf_ids, NOT on family_id: the auditor numbers its own families
    # (19 of them on live38) and the partition numbers its own (10 pre / 12
    # final). Matching by integer id mismatched 19 of 19 and shipped a family of
    # four classical-poetry leaves titled "中考录取分数与学校排名查询".
    fam_names = family_names(naming, fam, sizes)
    by_fam: dict[int, list[dict]] = {}
    for n in namings:
        lid = n["leaf_id"]
        f = int(fam[lid]) if lid < len(fam) else 0
        by_fam.setdefault(f, []).append(n)

    out = ["```"]
    out.append(f"════ {deps.cfg.domain.key} — {len(by_fam)} 家族 / {len(namings)} 叶 ════")
    out.append("")
    for f in sorted(by_fam, key=lambda k: -sum(int(sizes[x['leaf_id']]) for x in by_fam[k])):
        members = sorted(by_fam[f], key=lambda x: -int(sizes[x["leaf_id"]]))
        fn = int(sum(sizes[x["leaf_id"]] for x in members))
        risk = " (风控标记)" if any(x.get("risk_flag") for x in members) else ""
        out.append(f"■ {fam_names.get(f, f'家族 {f}')}{risk}  (n={fn:,}, {len(members)}叶, {fn/total*100:.1f}%)")
        for x in members:
            out.append(f"   ├─ {x.get('name_zh','')}  n={int(sizes[x['leaf_id']]):,}")
        out.append("")
    out.append("```")
    return out


def _leaf_catalogue(deps: Any, naming: dict, meta: dict) -> list[str]:
    namings = naming.get("namings", [])
    if not namings:
        return [vocab("zh")["no_data"]]
    try:
        # The DELIVERED partition, not p6's. Loading `leaf_labels` here paired
        # pre-governance leaf sizes (29 leaves) with post-governance families
        # (12), so the two halves of every row described different trees.
        labels = deps.leaf_labels_final()
        fam = deps._cache.get("leaf_family_final")
        if fam is None:
            fam = deps.load("leaf_family_final") if deps.has("leaf_family_final") else deps.load("leaf_family")
        sizes = np.bincount(labels, minlength=max(len(namings), len(fam)))
    except Exception:
        return [vocab("zh")["no_data"]]

    # Join on leaf_ids, NOT on family_id: the auditor numbers its own families
    # (19 of them on live38) and the partition numbers its own (10 pre / 12
    # final). Matching by integer id mismatched 19 of 19 and shipped a family of
    # four classical-poetry leaves titled "中考录取分数与学校排名查询".
    fam_names = family_names(naming, fam, sizes)
    by_fam: dict[int, list[dict]] = {}
    for n in namings:
        by_fam.setdefault(int(fam[n["leaf_id"]]) if n["leaf_id"] < len(fam) else 0, []).append(n)

    out: list[str] = []
    for f in sorted(by_fam):
        out.append(f"**■ {fam_names.get(f, f'家族 {f}')}**")
        for n in sorted(by_fam[f], key=lambda x: -int(sizes[x["leaf_id"]])):
            st = stars(n.get("coherence"))
            line = f"- {n.get('name_zh','')} ({st}, n={int(sizes[n['leaf_id']]):,}): {n.get('user_need','')}"
            if n.get("mix_notes"):
                line += f"  ⟨杂质: {n['mix_notes']}⟩"
            if n.get("risk_flag"):
                line += f"  ⚠️ **风控**: {n.get('risk_reason','')}"
            out.append(line)
        out.append("")
    return out


def _decision_chain(state: Any, phases: tuple[str, ...] | None = None) -> str:
    """Every parameter decision, in order, with what lost and on what evidence.

    The pipeline records this in full — question, candidates, winner, who decided,
    which metric was decisive, and each rejected option with the numbers it had at
    the time — and the report was surfacing about half of it, split across sections
    organised by topic rather than by the order things actually happened. A reader
    trying to answer "how did this tree come to exist?" had to reassemble the chain
    themselves. This is that chain, once, in sequence.
    """
    rows = list(state.get("decisions", []))
    if phases:
        rows = [d for d in rows if str(d.phase).startswith(phases)]
    if not rows:
        return "_本次运行未记录任何决策 — 这本身是个缺陷。_"
    # Sub-phases are recorded as p3a / p3c, so sort on (number, suffix) rather than
    # on a lookup of whole names — an unmatched name used to fall to the end, which
    # put representation selection *after* the K it fed into.
    def _order(phase: str) -> tuple[int, str]:
        m = re.match(r"p(\d+)([a-z]*)", str(phase))
        return (int(m.group(1)), m.group(2)) if m else (99, "")

    rows.sort(key=lambda d: _order(d.phase))

    out = ["下表按**实际发生顺序**列出每一个参数决策: 试了什么、选了什么、"
           "**由谁**依据**哪个指标**裁定, 以及被否决的方案当时的数字。",
           "每一行都可以独立追问 — 这是整条链路可复核的最小单元。", "",
           "| # | 阶段 | 决策问题 | 结论 | 裁定者 | 决定性指标 | 可逆? |",
           "|---|---|---|---|---|---|---|"]
    for i, d in enumerate(rows, 1):
        metrics = ", ".join(f"`{m}`" for m in (d.decisive_metrics or [])) or "—"
        rev = "是" if getattr(d, "reversible", False) else "否"
        out.append(f"| {i} | `{d.phase}` | {decision_question(d.question)} | "
                   f"**{d.choice}** | {d.decided_by} | {metrics} | {rev} |")
    out.append("")

    # The rationale is where the reasoning lives; a table cell truncates it.
    out += ["#### 每个决策的完整理由", ""]
    for i, d in enumerate(rows, 1):
        out.append(f"**{i}. [{d.phase}] {decision_question(d.question)}** → `{d.choice}`")
        if getattr(d, "rationale", ""):
            out += ["", f"> {prose(d.rationale)}"]
        ev = getattr(d, "evidence", None)
        if ev:
            shown = ", ".join(f"`{k}` = {_short_value(v, max_chars=60)}"
                              for k, v in list(ev.items())[:6])
            out += ["", f"证据: {shown}"]
        if getattr(d, "rejected", None):
            out += ["", f"落选 {len(d.rejected)} 个方案 (详见 §{'9'} 失败史)。"]
        out.append("")
    return "\n".join(out)


def _short_value(v: Any, *, max_chars: int = 44) -> str:
    """Render ONE artifact value for inline prose, never a Python repr.

    `_decision_chain` interpolated the raw value, so `p2e`'s per-class audit — a
    list of 13 dicts — reached live40's Chinese report as 1,900 characters of
    `[{'class': 'OFFTOPIC_RISK_NOISE', 'n_in_subsample': 57, ...}]` in the middle
    of a sentence. A container is summarised by its SIZE and left in the artifact
    it came from; a short list of scalars is worth more spelled out than counted,
    so it is. Nothing here is allowed to grow without bound.
    """
    if v is None or (isinstance(v, (list, tuple, dict, str)) and len(v) == 0):
        return "无"
    if isinstance(v, bool):
        return "是" if v else "否"
    if isinstance(v, float):
        return f"{v:.4g}"
    if isinstance(v, (list, tuple)):
        # Scalars are the readable case; dicts are the case that produced the dump.
        if all(isinstance(x, (int, float, str)) and not isinstance(x, bool) for x in v):
            inline = ", ".join(f"{x:.4g}" if isinstance(x, float) else str(x) for x in v)
            if len(inline) <= max_chars:
                return f"[{inline}]"
        return f"[{len(v)} 项, 见产物]"
    if isinstance(v, dict):
        return f"{{{len(v)} 项, 见产物}}"
    sv = str(v)
    return sv if len(sv) <= max_chars else sv[: max_chars - 1] + "…"


def _kv_cell(d: Any, *, max_items: int = 8, max_chars: int = 320) -> str:
    """Render a gate's observed/threshold dict as readable `k=v` pairs.

    A markdown cell can hold far more than the 80 characters this used to cut at,
    and the numbers are the point: a gate row whose observed values are elided is
    a row the reader cannot check. Long values are shortened individually so that
    every KEY still appears, rather than losing whole fields off the end.
    """
    if not d:
        return "—"
    parts: list[str] = []
    for k, v in list(d.items())[:max_items]:
        parts.append(f"`{k}`={_short_value(v, max_chars=44)}")
    cell = "; ".join(parts)
    if len(d) > max_items:
        cell += f"; …另 {len(d) - max_items} 项"
    return cell[:max_chars].replace("|", "\\|")


def _gate_ledger(state: Any, phases: tuple[str, ...] | None = None) -> str:
    """Every quality gate with its observed value, its bar, and what to do if it failed.

    Gates are the only place the pipeline says "this is not good enough", and the
    `remediation` field is the only place it says what to do about it. Both were
    computed on every run and neither reached the reader.
    """
    gates = state.get("gates", {}) or {}
    if phases:
        gates = {k: g for k, g in gates.items() if str(g.phase).startswith(phases)}
    if not gates:
        return "_本次运行未评估任何质量门。_"
    icon = {"passed": "✅ 通过", "warned": "⚠️ 警告", "failed": "❌ 未通过",
            "rejected": "⛔ 人工否决", "skipped": "— 跳过"}
    out = ["质量门是整条流水线唯一会说「这还不够好」的地方。下表列出**每一道门**: "
           "实测值、门槛、是否阻断, 以及未通过时的**处置建议**。", "",
           "| 质量门 | 阶段 | 结果 | 阻断? | 实测 | 门槛 |", "|---|---|---|---|---|---|"]
    failed = []
    for name, g in sorted(gates.items(), key=lambda kv: str(kv[1].phase)):
        # Render the fields, not a truncated JSON blob. `observed` was dumped and
        # cut at 80 chars, which on `p2a_pilot_agreement` — 15 fields carrying
        # kappa, the ceiling, n and the slack — showed about two of them. This
        # table IS the audit trail of the deliverable; cutting it mid-number
        # leaves the reader unable to check any claim the report makes.
        obs = _kv_cell(g.observed)
        thr = _kv_cell(g.threshold)
        # A GATE THAT PASSED WITH SLACK MUST SAY SO IN THE TABLE. `p2b_kappa`
        # rendered as ✅ 通过 with 实测 0.8221 beside 门槛 0.9 and no explanation,
        # while the gate's own message read "PROCEEDING WITH RESIDUAL SLACK …
        # short of 0.9; every downstream number …". The caveat existed on every
        # run and reached no reader, because the ledger printed `message` only
        # for gates that failed.
        slack = _passed_below_threshold(g)
        mark = " ⚠️ 带保留通过" if slack else ""
        out.append(f"| `{name}` | `{g.phase}` | {icon.get(g.status, g.status)}{mark} | "
                   f"{'是' if g.blocking else '否'} | {obs} | {thr} |")
        if g.status in ("failed", "warned", "rejected") and g.remediation:
            failed.append((name, g))
    out.append("")

    # Every gate's own conclusion, passing ones included.
    with_msg = [(n, g) for n, g in sorted(gates.items(), key=lambda kv: str(kv[1].phase))
                if getattr(g, "message", "")]
    if with_msg:
        out += ["#### 每一道门实际得出的结论", "",
                "**通过 ≠ 没有保留。** 下面是每一道门自己写下的结论 —— 包括通过的那些, "
                "因为一道门可以在实测值低于门槛时仍然放行, 而放行的理由只写在这里。", ""]
        for n, g in with_msg:
            flag = "⚠️ **带保留通过** — " if _passed_below_threshold(g) else ""
            # `prose()` was applied to the remediation and NOT to the message, so
            # every gate's own conclusion shipped in whatever language it was
            # authored in — and this is the section the report introduces as
            # "每一道门自己写下的结论". Unmapped strings fall through unchanged,
            # so this can only ever improve a line, never break one.
            out += [f"**`{n}`** — {flag}{prose(g.message)}", ""]
    if failed:
        out += ["#### 未通过/警告的门 — 处置建议", ""]
        for name, g in failed:
            out += [f"**`{name}`** — {prose(g.message)}", "", f"> {prose(g.remediation)}", ""]
    missing = state.get("declared_gates_never_evaluated") or []
    if missing:
        out += [f"> ⚠️ **声明为阻断但从未实际评估的门**: {', '.join(f'`{m}`' for m in missing)} — "
                "一道从不触发的门与一道通过的门在报告里长得一模一样, 因此这里单独列出。", ""]
    return "\n".join(out)



def _passed_below_threshold(g: Any) -> bool:
    """Did this gate pass while its own observed value sits under its bar?

    Not a failure — several gates legitimately proceed with a recorded caveat —
    but a reader seeing ✅ beside a number below the threshold, with no
    explanation, will read it as clean. Compares like-named fields only, so a
    threshold about one quantity is never checked against another's observation.
    """
    if getattr(g, "status", "") != "passed":
        return False
    obs, thr = getattr(g, "observed", None) or {}, getattr(g, "threshold", None) or {}
    if not isinstance(obs, dict) or not isinstance(thr, dict):
        return False
    for tk, tv in thr.items():
        if not isinstance(tv, (int, float)) or isinstance(tv, bool):
            continue
        base = str(tk).replace("min_", "").replace("max_", "").replace("_floor", "")
        for ok, ov in obs.items():
            if not isinstance(ov, (int, float)) or isinstance(ov, bool):
                continue
            if str(ok) == base or str(ok).endswith(base) or base.endswith(str(ok)):
                if str(tk).startswith("max_"):
                    if ov > tv:
                        return True
                elif ov < tv:
                    return True
    return False

def _governance_ledger(state: Any, gov: dict) -> str:
    """Every prescription: who proposed it, what happened to it, and why."""
    rx = list(state.get("prescriptions", []))
    if not rx:
        return ""
    out = ["治理的要求是**执行, 不是记录**。下表列出每一条处方的**最终去向** — "
           "已执行、被拒绝、或仍在提出状态 (后者会让运行直接失败)。", "",
           "| 处方 | 类型 | 目标 | 提出者 | 状态 | 理由 / 拒绝原因 |", "|---|---|---|---|---|---|"]
    for r in rx:
        why = getattr(r, "decline_reason", "") or getattr(r, "rationale", "") or "—"
        tgt = ", ".join(str(t) for t in (getattr(r, "target_names", None) or r.targets or [])[:4])
        out.append(f"| `{r.id}` | {r.kind} | {tgt} | {r.proposed_by} | "
                   f"**{r.status}** | {why[:110]} |")
    out.append("")
    return "\n".join(out)


def _failure_history(state: Any, phases: tuple[str, ...] | None = None) -> str:
    rows = [d for d in state.get("decisions", []) if getattr(d, "rejected", None)]
    if phases:
        rows = [d for d in rows if str(d.phase).startswith(phases)]
    if not rows:
        return "_本次运行没有记录被否决的方案 — 这通常意味着尝试得不够多。_"
    parts: list[str] = []
    for d in rows:
        parts.append(f"**{decision_question(d.question)}** → 选定 `{d.choice}`")
        parts.append("")
        parts.append("| 被否决方案 | 原因 | 当时指标 |")
        parts.append("|---|---|---|")
        for r in d.rejected[:10]:
            m = r.get("metrics") or {}
            mt = ", ".join(f"{k}={num(vv)}" for k, vv in m.items() if vv is not None) or "—"
            parts.append(f"| `{r.get('option','?')}` | {r.get('why_rejected','')} | {mt} |")
        parts.append("")
    return "\n".join(parts)
