#!/usr/bin/env python
"""渲染「意图内部结构与代表性样例」深挖报告（中文）+ 工作簿 + 图。只读 `work/<域>/intent_structure/` 下的表。

    P5_COHORT=fin8 python analysis/pooled5/p5_intent_structure_report.py 金融8

叙述段落写在 `work/<域>/intent_structure/narrative.md` 里，用 <!--NARR:键-->…<!--/NARR:键--> 包起来，
渲染时原样填进对应位置；没有就留一行「（叙述待写）」。所有数字表都由本脚本从 CSV 直接生成，不从正文读。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pooled5_common import SOURCES, SRC_ZH, WORK, run_dir  # noqa: E402
from p5_snapshot_report import _md, _q  # noqa: E402
from p5_intent_structure import PAIRS, newcombe  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Hiragino Sans GB", "Heiti SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
CASES = [("NEWS_EVENT", "案例一"), ("VERDICT_QUESTION", "案例二"), ("INVEST_ADVICE", "案例三")]
SNAP_ORDER = [SRC_ZH[s] for s in SOURCES]
FACET_SHOW = ["咨询/求建议", "事件与进度", "问原因", "时效", "是非核实", "问句", "第一人称"]
# 读法分组：本文的解释层，把盲标码并成回答「核实还是决策」这类问题的几组；不是盲标的码，码本一变断言就拦下。
FACET_GROUPS = {
    "NEWS_EVENT": {"查找型（消息/进度/数据）": ["LATEST_NEWS", "STATUS_TIMING", "FACT_DETAIL"],
                   "分析型（预测/评估/问原因）": ["FORECAST", "ASSESS_JUDGE", "EXPLAIN_CAUSE"],
                   "规则与业务求助": ["RULE_CONCEPT", "SERVICE_HELP"]},
    "VERDICT_QUESTION": {"核实事实/身份": ["CATEGORY_CHECK", "WORLD_STATE"],
                         "核实规则/资格/后果": ["SERVICE_RULE", "ELIGIBILITY_COVER", "IMPACT_EXPOSURE", "REMEDY_UNDO"],
                         "辅助决策（评估或建议）": ["MERIT_SAFETY", "DECISION_ADVICE"]},
    "INVEST_ADVICE": {"要一个决定": ["DECIDE"], "要判断/预测": ["FORECAST", "APPRAISE", "RISK"],
                      "要一张名单": ["SELECT", "SCREEN"], "要信息/方法": ["INFO", "RULE"]},
}
UNGROUPED = "无法判定/无多数"
REFS = [
    ("Monroe, Colaresi & Quinn (2008). Fightin' Words: Lexical Feature Selection and Evaluation for Identifying the Content of Political Conflict. Political Analysis 16(4): 372–403.",
     "https://ideas.repec.org/a/cup/polals/v16y2008i04p372-403_00.html"),
    ("Kim, Khanna & Koyejo (2016). Examples are not Enough, Learn to Criticize! Criticism for Interpretability (MMD-critic). NeurIPS 2016.",
     "https://proceedings.neurips.cc/paper_files/paper/2016/file/5680522b8e2bb01943234bce7bf84534-Paper.pdf"),
    ("Lopez-Paz & Oquab (2017). Revisiting Classifier Two-Sample Tests. ICLR 2017.", "https://arxiv.org/abs/1610.06545"),
    ("Kaufman & Rousseeuw (1990). Partitioning Around Medoids (Program PAM). In Finding Groups in Data, Wiley, ch. 2.",
     "https://onlinelibrary.wiley.com/doi/10.1002/9780470316801.ch2"),
]


def _snap_sort(df: pd.DataFrame, col: str = "快照") -> pd.DataFrame:
    order = {s: i for i, s in enumerate(SNAP_ORDER)}
    return df.assign(_o=df[col].map(order)).sort_values("_o", kind="stable").drop(columns="_o")


def _narr(narr: dict, key: str) -> str:
    return f"<!--NARR:{key}-->\n{narr[key].strip()}\n<!--/NARR:{key}-->\n" if key in narr else f"<!--NARR:{key}-->\n（叙述待写）\n<!--/NARR:{key}-->\n"


def _load_narr(p: Path) -> dict:
    if not p.exists():
        return {}
    s = p.read_text(encoding="utf-8")
    return {m.group(1): m.group(2) for m in re.finditer(r"<!--NARR:([^>]+)-->\n(.*?)\n<!--/NARR:\1-->", s, re.S)}


def fig_methods(cmp_: pd.DataFrame, out: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    labels = [("旧", "旧：流量最高 3 条"), ("H3", "Hamilton k=3"), ("显式", "构成显式（≤8 条，覆盖 80%）")]
    for ax, (metric, zh) in zip(axes, [("被代表质量%", "被代表质量（%）"), ("典型性分位", "典型性分位（0.5=普通）"), ("自助法Jaccard", "自助法稳定性（Jaccard）")]):
        data = [cmp_[f"{m}_{metric}"].dropna().to_numpy() for m, _ in labels]
        ax.boxplot(data, widths=.55, showmeans=True)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels([z for _, z in labels], fontsize=9)
        ax.set_title(zh, fontsize=11)
        ax.grid(axis="y", alpha=.25)
    fig.suptitle("三种取例方法，同一组指标（每个意图 × 快照格子一个点，n>=10 的 120 格）", fontsize=12)
    fig.tight_layout()
    f = out / "意图结构_取例方法对比.png"
    fig.savefig(f, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return f


def fig_c2st(a: pd.DataFrame, out: Path) -> Path:
    pairs = list(dict.fromkeys(a["对比"]))
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for i, p in enumerate(pairs):
        g = a[a["对比"] == p]
        jit = (np.arange(len(g)) - len(g) / 2) * 0.04
        ax.scatter(i + jit, g["AUC"], s=26, c=np.where(g["两侧可分(p<0.05)"], "#d9480f", "#adb5bd"), zorder=3)
        ax.scatter(i + jit, g["零分布95分位"], s=10, marker="_", c="#343a40", zorder=2)
    ax.axhline(0.5, color="#adb5bd", lw=1, ls="--")
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels(pairs, fontsize=9, rotation=20)
    ax.set_ylabel("意图内部 embedding 分类 AUC（5 折交叉验证）")
    ax.set_title("同一个意图，两侧能不能被分开：橙点 = 置换检验 p<0.05；短横 = 该格零分布 95 分位", fontsize=11)
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    f = out / "意图结构_可分性.png"
    fig.savefig(f, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return f


def fig_stack(comp: pd.DataFrame, code: str, title: str, out: Path, top: int = 7) -> Path:
    g = comp[comp["外层"] == code]
    order = g.groupby("内层名")["条数"].sum().sort_values(ascending=False).index[:top].tolist()
    snaps = [s for s in SNAP_ORDER if s in set(g["快照"])]
    M = np.zeros((len(snaps), len(order) + 1))
    ns = []
    for i, s in enumerate(snaps):
        gs = g[g["快照"] == s]
        ns.append(int(gs["n"].iloc[0]))
        for j, leaf in enumerate(order):
            M[i, j] = gs.loc[gs["内层名"] == leaf, "占比%"].sum()
        M[i, -1] = 100 - M[i, :-1].sum()
    fig, ax = plt.subplots(figsize=(11, 0.55 * len(snaps) + 1.8))
    left = np.zeros(len(snaps))
    cmap = plt.get_cmap("tab20")
    for j, leaf in enumerate(order + ["其它叶"]):
        ax.barh(range(len(snaps)), M[:, j], left=left, color="#dee2e6" if j == len(order) else cmap(j), label=leaf[:16])
        left += M[:, j]
    ax.set_yticks(range(len(snaps)))
    ax.set_yticklabels([f"{s} (n={n})" for s, n in zip(snaps, ns)], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("该意图在该快照内部的叶构成（%）")
    ax.set_title(title, fontsize=11)
    ax.legend(ncol=4, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    f = out / f"意图结构_{code}_叶构成.png"
    fig.savefig(f, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return f


def fig_facet(comp: pd.DataFrame, code: str, title: str, out: Path) -> Path:
    snaps = [s for s in SNAP_ORDER if s in set(comp["快照"])]
    # 「无法判定 / 无多数」一律画灰色放最后；其余码用 tab10（10 色），9 个码不再撞色（Set2 只有 8 色，第 9 个码曾与第 1 个同色）
    unsure = {n for n in comp["子功能名"].unique() if ("无法判定" in n or "不可判定" in n or n == "无多数")}
    names = [n for n in comp.groupby("子功能名")["条数"].sum().sort_values(ascending=False).index.tolist() if n not in unsure] + sorted(unsure)
    fig, ax = plt.subplots(figsize=(11, 0.55 * len(snaps) + 1.8))
    left = np.zeros(len(snaps))
    cmap = plt.get_cmap("tab10")
    assert len(names) - len(unsure) <= 10, f"{code}: {len(names)} sub-functions exceed the 10-colour palette"
    k = 0
    for nm in names:
        vals = np.array([comp.loc[(comp["快照"] == s) & (comp["子功能名"] == nm), "占比%"].sum() for s in snaps])
        color = "#ced4da" if nm in unsure else cmap(k)
        k += 0 if nm in unsure else 1
        ax.barh(range(len(snaps)), vals, left=left, color=color, label=nm[:22])
        left += vals
    ns = [int(comp.loc[comp["快照"] == s, "样本n"].iloc[0]) for s in snaps]
    ax.set_yticks(range(len(snaps)))
    ax.set_yticklabels([f"{s} (样本n={n})" for s, n in zip(snaps, ns)], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("盲标多数子功能占比（%，快照内样本）")
    ax.set_title(title, fontsize=11)
    ax.legend(ncol=3, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    f = out / f"意图结构_{code}_语用子功能.png"
    fig.savefig(f, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return f


def facet_groups(fdir: Path, code: str) -> tuple[str, dict]:
    groups = FACET_GROUPS.get(code)
    if not groups:
        return "", {}
    cb = json.loads((fdir / f"{code}_codebook.json").read_text(encoding="utf-8"))
    real = sorted(c["code"] for c in cb["codes"] if c["code"] != cb.get("unsure_code"))
    covered = sorted(x for xs in groups.values() for x in xs)
    assert covered == real, f"{code}: facet groups {covered} != codebook codes {real}"
    name = {c["code"]: c["name_zh"] for c in cb["codes"]}
    lab = pd.read_csv(fdir / f"{code}_labels.csv")
    g2 = {x: g for g, xs in groups.items() for x in xs}
    lab["读法分组"] = lab["多数"].map(g2).fillna(UNGROUPED)
    cols = list(groups) + [UNGROUPED]
    snaps = [s for s in SNAP_ORDER if s in set(lab["快照"])]
    ct = pd.crosstab(lab["快照"], lab["读法分组"]).reindex(index=snaps, columns=cols, fill_value=0)
    by_snap = (ct.div(ct.sum(1), axis=0) * 100).round(1)
    by_snap.insert(0, "样本n", ct.sum(1))
    by_snap = by_snap.reset_index().rename(columns={"index": "快照"})
    w = lab.groupby("读法分组")["抽样权重"].sum().reindex(cols, fill_value=0)
    pooled = pd.DataFrame({"读法分组": cols, "包含的码": ["、".join(name[x] for x in groups[g]) if g in groups else "" for g in cols],
                           "加权占比%": (100 * w / w.sum()).round(1).to_numpy()})
    prow = []
    for a, b, label in PAIRS:
        ga, gb = lab[lab.source == a], lab[lab.source == b]
        if len(ga) < 30 or len(gb) < 30:
            continue
        r = {"对比": label, "n_a": len(ga), "n_b": len(gb)}
        for g in cols:
            ka, kb = int((ga["读法分组"] == g).sum()), int((gb["读法分组"] == g).sum())
            dd, lo, hi = newcombe(ka, len(ga), kb, len(gb))
            r[g] = f"{100 * ka / len(ga):.1f}→{100 * kb / len(gb):.1f}（{100 * dd:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]{'*' if lo > 0 or hi < 0 else ''}）"
        prow.append(r)
    pairs = pd.DataFrame(prow)
    lf = lab.pivot_table(index="叶名", columns="读法分组", values="抽样权重", aggfunc="sum", fill_value=0).reindex(columns=cols, fill_value=0)
    lf = lf.loc[lf.sum(1).sort_values(ascending=False).index]
    leaf = (lf.div(lf.sum(1), axis=0) * 100).round(1)
    leaf.insert(0, "加权行数", lf.sum(1).round(0).astype(int))
    leaf.insert(1, "样本行数", lab.groupby("叶名").size().reindex(leaf.index).fillna(0).astype(int))
    leaf = leaf.reset_index()
    md = ("\n**读法分组**（本文的解释层：把上面的盲标码并成几组，直接回答「是核实还是决策」这类问题；组的定义见下表「包含的码」，不是盲标本身）\n\n"
          + _md(pooled) + "\n**各快照（样本内占比%）**\n\n" + _md(by_snap)
          + ("\n**对齐快照对（a→b，差 pp [95% Newcombe]，* = 区间不含 0）**\n\n" + _md(pairs) if len(pairs) else "")
          + "\n**读法分组 × 叶（前 12 个叶，行内占比%，按抽样权重）**\n\n" + _md(leaf.head(12)))
    return md, {"by_snapshot": by_snap, "pooled": pooled, "pairs": pairs, "by_leaf": leaf}


def build(domain: str) -> Path:
    base = WORK / domain / "intent_structure"
    fdir = base / "facets"
    gen = run_dir(domain)
    stem = json.loads(json.dumps(__import__("yaml").safe_load((gen / "config.resolved.yaml").read_text(encoding="utf-8"))["domain"]["key"]))
    post = gen / "postprocessed"
    img = post / "img"
    img.mkdir(parents=True, exist_ok=True)
    narr = _load_narr(base / "narrative.md")
    summ = json.loads((base / "summary.json").read_text(encoding="utf-8"))
    cmp_ = pd.read_csv(base / "example_method_comparison.csv")
    ex = pd.read_csv(base / "representative_examples.csv")
    struct = pd.read_csv(base / "l1_structure_summary.csv")
    comp = pd.read_csv(base / "l1_leaf_composition.csv")
    leafmix = pd.read_csv(base / "l1_leafmix_contrasts.csv")
    facets = pd.read_csv(base / "facets_by_cell.csv")
    fpairs = pd.read_csv(base / "facets_pair_contrasts.csv")
    words = pd.read_csv(base / "distinctive_words.csv")
    c2 = pd.read_csv(base / "c2st_auc.csv")
    nf_path = base / "l1_leaf_namefit.csv"
    namefit = pd.read_csv(nf_path, keep_default_na=False, na_values=[""]) if nf_path.exists() else pd.DataFrame()
    fsum = json.loads((fdir / "summary.json").read_text(encoding="utf-8")) if (fdir / "summary.json").exists() else {}
    run = gen.parent.name
    group_tables: dict = {}
    P: list[str] = []
    P.append(f"# {domain}　意图内部结构与代表性样例（`{run}` 深挖）\n")
    P.append(f"> 本文是《{stem}_意图与聚类叶_跨快照对比》的**补充深挖**，不改那份报告的任何数字。它回答三个问题："
             "①报告里每个意图、每个快照印出来的真实 query 例子是怎么选的、能不能代表该格；②同一个意图在不同快照之间，"
             "真实 query 到底哪里不同、差别有多真；③怎样用自下而上的叶把一个宽泛的 L1 意图拆开来看。\n>\n"
             f"> 语料 {summ['rows']:,} 行，可引 {summ['quotable_rows']:,} 行（引用护栏与主报告逐项相同，脚本断言过）。"
             "所有表都在 `analysis/pooled5/work/" + domain + "/intent_structure/`，同名工作簿在本目录。\n")
    P.append(_narr(narr, "总览"))

    # 1 现有例子
    oldm = {k: cmp_[f"旧_{k}"] for k in ("被代表质量%", "典型性分位", "长度分位", "自助法Jaccard")}
    P.append("\n## 1　现有例子是怎么选的，能不能代表该格\n")
    P.append("主报告的每张意图卡片印的是：**该意图、该快照里可引行中 `pv_norm` 最高的 3 条**（脚本 `p5_snapshot_classes.examples`，"
             "另有 3 条随机行写进工作簿但不印）。流量最高的串回答的是「最多人搜的是哪几句」，不是「这一格里装的是什么」。"
             f"在 n≥10 的 {len(cmp_)} 个格子上逐格实测：\n")
    arb = cmp_[cmp_["旧_顺序实为任意"].fillna(False).astype(bool)]
    P.append(f"- **长尾里流量不是排序**：{len(arb)} 格（{', '.join(f'{k} {v}' for k, v in arb['快照'].value_counts().items())}）"
             "第 3 条例子与 3 行以上的流量并列——随机层与语音导出的 PV 几乎全是 1 或没有，所谓「流量最高」在这些格子里实际是文件顺序。\n")
    P.append(f"- **只覆盖一小块**：例子所在的叶合计只占该格行数的中位 {oldm['被代表质量%'].median():.1f}%。\n")
    P.append(f"- **偏短、不稳**：长度分位均值 {oldm['长度分位'].mean():.2f}（0.5 为普通长度）；把该格的行重抽 {20} 次，"
             f"「流量最高 3 条」与原来的 Jaccard 均值只有 {oldm['自助法Jaccard'].mean():.2f}；另有 {int((cmp_['旧_重复'].fillna(0) > 0).sum())} 格同一串印了两遍。\n")

    # 2 代表性
    P.append("\n## 2　什么叫「代表性」、怎么做到、能保证到什么程度\n")
    P.append("代表性拆成两件可以测的事：**构成**——例子覆盖了该格内部多大份额、各子群分到的名额是否与真实构成一致；"
             "**典型性**——每条例子是否是它所在子群的中心，而不是边缘或异类。子群用另一条路线给：意图格子按自下而上的叶分，"
             "叶格子按 L1 意图分，L2 格子按叶分。子群中心取 embedding 平均余弦最大的那条可引串（行向量归一化以后，平均余弦 = x·均值向量，"
             "所以这是精确的 medoid，不是近似），去掉重复串和余弦 > 0.97 的近重复。\n")
    P.append("**3–5 条例子本来就代表不了一个意图格子。** n≥30 的格子里，覆盖 80% 行所需的叶数中位 "
             f"{int(struct[struct.n >= 30]['覆盖80%所需内层数'].median())}、p75 {int(struct[struct.n >= 30]['覆盖80%所需内层数'].quantile(.75))}，"
             f"有效叶数中位 {struct[struct.n >= 30]['有效内层数(exp熵)'].median():.1f}。所以代表性只能靠**把构成写出来**：每条例子标明它代表该格多少行。\n")
    rows = []
    for m, zh in (("旧", "旧：流量最高 3 条"), ("H3", "Hamilton k=3（同样 3 条，按构成分配 + 子群中心）"), ("显式", "构成显式（每个子群 1 条、标份额，覆盖到 80% 或 8 条）")):
        rows.append({"方法": zh, "被代表质量%（中位）": round(cmp_[f"{m}_被代表质量%"].median(), 1),
                     "典型性分位（均值）": round(cmp_[f"{m}_典型性分位"].mean(), 2), "长度分位（均值）": round(cmp_[f"{m}_长度分位"].mean(), 2),
                     "自助法 Jaccard（均值）": round(cmp_[f"{m}_自助法Jaccard"].mean(), 2),
                     "每格例子数（均值）": round(cmp_[f"{m}_例子数"].mean(), 1)})
    P.append("**表 2-A　三种取例方法，同一组指标**（Hamilton = 最大余数法按各子群全部行的份额分配名额；典型性分位是例子在该格内部离格子中心的分位，"
             "构成显式的例子是**子群**的中心，所以对整格而言它低于 Hamilton k=3，这是设计如此）\n\n" + _md(pd.DataFrame(rows)))
    bysnap = cmp_.groupby("快照")[["旧_被代表质量%", "H3_被代表质量%", "显式_被代表质量%", "旧_自助法Jaccard", "H3_自助法Jaccard", "显式_自助法Jaccard"]].mean().round(2).reset_index()
    P.append("\n**表 2-B　按快照**\n\n" + _md(_snap_sort(bysnap)))
    f1 = fig_methods(cmp_, img)
    P.append(f"\n![取例方法对比](img/{f1.name})\n")
    P.append("**能保证什么、不能保证什么。** 构成显式能按构造保证：①每条例子都是它所在叶的中心；②印出来的例子合计代表了多少行、"
             "没被代表的份额是多少，都明写；③同一格不会出现重复或近重复；④可复算（确定性，不依赖随机数）。它**不能**保证：叶是话题簇，"
             "叶的中心在话题上典型，却不一定在「怎么问」上典型——同一个叶里「最新消息」和「可能性有多大」可以并存，这一层要靠第 5–7 节的"
             "说法特征与盲标子功能来看；而且 n 很小的格子（助手头部、语音头部里有的只有个位数）任何例子都只是个例。\n")
    P.append(_narr(narr, "例子"))

    # 3 结构总览
    P.append("\n## 3　意图内部结构总览：每个意图在每个快照里横跨多少个叶\n")
    piv = struct.pivot_table(index="外层名", columns="快照", values="有效内层数(exp熵)")
    piv = piv[[s for s in SNAP_ORDER if s in piv.columns]].round(1)
    piv = piv.loc[piv.mean(1).sort_values(ascending=False).index].reset_index().rename(columns={"外层名": "意图"})
    P.append("**表 3　有效叶数（exp 熵）**——1 表示全在一个叶里；数越大，这个意图在该快照里越是「跨话题的问法」而不是一个话题。n<10 的格子数字不稳。\n\n" + _md(piv))
    gpath = base / "leaf_family_granularity.json"
    if gpath.exists():
        gr = json.loads(gpath.read_text(encoding="utf-8"))
        lv, fm = gr["叶"], gr["家族"]
        def _fam_str(names: list) -> str:
            cnt = pd.Series(names).value_counts()
            return "、".join(f"{n}×{c}" if c > 1 else n for n, c in cnt.sort_index().items())
        odd = [_fam_str(x) for x in gr["multi_leaf_families"]]
        flagged = []
        for fam_names in gr["multi_leaf_families"]:
            for stray, kin in (("查询电话号码归属", "代码"), ("蚂蚁庄园今日答案查询", "行情")):
                if stray in fam_names:
                    mates = sorted({x for x in fam_names if kin in x})
                    if mates:
                        flagged.append(f"「{stray}」与「{'」「'.join(mates)}」同族")
        flag_txt = ("例如" + "，".join(flagged) + "；") if flagged else ""
        P.append("\n**为什么用叶、不用家族。** 同一组指标在两层上实测（交付分区 `bu_family_final`）：\n\n"
                 + _md(pd.DataFrame([{"层": k, "组数": g["groups"], "每组行数中位": g["rows_per_group_median"], "每组行数p10": g["rows_per_group_p10"],
                                      "组内到中心平均余弦（行加权，越高越紧）": g["within_group_mean_cosine_to_centre_row_weighted"],
                                      "意图×快照格有效组数中位（n≥30）": g["effective_groups_median"], "覆盖80%所需组数中位": g["groups_for_80%_median"],
                                      "首位组占比中位%": g["top_group_share_median_%"]} for k, g in (("叶", lv), ("家族", fm))]))
                 + f"\n{gr['families']} 个家族里只有 {gr['families_with_more_than_one_leaf']} 个含多个叶，但它们装着 {gr['rows_in_multi_leaf_families_%']}% 的行。"
                 f"换成家族，意图格子的有效组数只从 {lv['effective_groups_median']} 降到 {fm['effective_groups_median']}——**分散不是叶切得太细造成的，"
                 "而是一个 L1 意图本来就是横跨多个话题的问法**，换哪一层话题粒度都一样分散；家族反而更松（组内余弦更低），而且有的家族把不相干的叶并在一起，"
                 f"{flag_txt}同名的叶记作「×2」（两个不同的叶被起了同一个名字）。所以本文以叶为单位，家族不单列。\n\n"
                 "多叶家族一览：\n\n" + "\n".join(f"- {x}" for x in odd) + "\n")
    P.append(_narr(narr, "结构"))

    # 4 检验框架
    P.append("\n## 4　同一意图在不同快照之间差在哪里：检验框架\n")
    P.append("一次只变一个因素的对齐快照对：界面（头部层：2026搜索↔助手头部1k；随机层：2026搜索随机1w↔助手随机1k）、深度（搜索内、助手内）、"
             "时间（头部、随机）、输入方式（助手头部↔语音头部）。每一对看四种证据：\n\n"
             "1. **构成**：意图内部的叶构成，TVD 配同源噪声上界（两侧合起来随机对半切 200 次的 95 分位）；\n"
             "2. **说法**：几个可数的说法特征（咨询/求建议、事件与进度、问原因、时效、是非核实、问句、第一人称），差值配 Newcombe 区间——"
             "这些是词表代理，只用来定位差别在哪，不单独下结论；\n"
             "3. **用词**：带信息先验的加权对数几率（Monroe 等 2008），只列词不列句；\n"
             "4. **可分性**：只在该意图内部、只用两侧行的 embedding，平衡抽样后做 5 折交叉验证的逻辑回归，AUC 与打乱标签 100 次的零分布比"
             "（分类器双样本检验，Lopez-Paz & Oquab 2017）。AUC≈0.5 表示两侧在语义空间里分不开；显著只说明「分得开」，不说明差在哪。\n")
    c2sum = c2.groupby("对比").agg(格子数=("AUC", "size"), 可分格子数=("两侧可分(p<0.05)", "sum"), AUC中位=("AUC", "median")).reset_index()
    P.append("**表 4　可分性总表（每个意图 × 对齐快照对，两侧各 ≥40 行才测）**\n\n" + _md(c2sum.round(3)))
    f2 = fig_c2st(c2, img)
    P.append(f"\n![可分性](img/{f2.name})\n")

    # 5-7 案例
    l1name = dict(zip(struct["外层"], struct["外层名"]))
    for code, tag in CASES:
        if code not in l1name:
            continue
        nm = l1name[code]
        P.append(f"\n## {'5' if tag == '案例一' else '6' if tag == '案例二' else '7'}　{tag}：{nm}\n")
        P.append(_narr(narr, f"案例_{code}"))
        e = ex[(ex["层级"] == "L1意图") & (ex["类"] == code)]
        lines = []
        for snap, g in _snap_sort(e).groupby("快照", sort=False):
            parts = [f"{_q(r['query'])}〔{r['子群']} {100 * r['stratum_share']:.0f}%〕" for _, r in g.iterrows()]
            cov = 100 * g["stratum_share"].sum()
            lines.append(f"- **{snap}**（n={int(g['n'].iloc[0])}，这几条合计代表 {cov:.0f}%）：" + " / ".join(parts))
        P.append("**代表性样例**（构成显式：每条是所在叶的中心，〔〕里是它代表该格的份额）\n\n" + "\n".join(lines) + "\n")
        l2 = ex[(ex["层级"] == "L2子意图") & (ex["类"].astype(str).str.startswith(code + "__"))]
        if len(l2):
            lines = []
            for l2c, g in l2.groupby("类"):
                g = g.sort_values("n", ascending=False).head(6)
                lines.append(f"- `{l2c}`：" + " / ".join(f"{_q(r['query'])}〔{r['快照']}·{r['子群']}〕" for _, r in g.iterrows()))
            P.append("\n**L2 子意图的读法**（L2 没有名字；这里每个 L2 取行数最多的几个快照格子里各子群的中心例子）\n\n" + "\n".join(lines) + "\n")
        g = comp[comp["外层"] == code]
        rows = []
        for snap, gs in _snap_sort(g).groupby("快照", sort=False):
            gs = gs.sort_values("条数", ascending=False)
            rows.append({"快照": snap, "n": int(gs["n"].iloc[0]),
                         "前 6 个叶（占比% [95%CI]）": "；".join(f"{r['内层名']} {r['占比%']} [{r['95CI低%']}, {r['95CI高%']}]" for _, r in gs.head(6).iterrows())})
        P.append("\n**叶构成（每个快照前 6 个叶）**\n\n" + _md(pd.DataFrame(rows)))
        f3 = fig_stack(comp, code, f"{nm}：各快照内部的叶构成", img)
        P.append(f"\n![叶构成](img/{f3.name})\n")
        nf = namefit[namefit["意图"] == code].head(10) if len(namefit) else namefit
        if len(nf):
            allc = namefit["cos差"].dropna()
            p10, p50 = allc.quantile(.10), allc.median()
            show_nf = nf.assign(意图内中心例子=nf["意图内中心例子"].fillna("").map(lambda x: _q(x) if x else "（无可引行）"),
                                叶其余中心例子=nf["叶其余中心例子"].fillna("").map(lambda x: _q(x) if x else "（无可引行）"),
                                读法=np.where(nf["cos差"] < p10, "叶名描述的是另一回事", np.where(nf["cos差"] < p50, "偏离叶的多数", "与叶的多数相近")))
            P.append("\n**叶名在这个意图里还适用吗（该意图行数最多的 10 个叶，全快照合并）**——叶是全语料上的话题簇，叶名描述的是整叶的多数；"
                     "「该意图占叶%」是这个叶的行里属于该意图的份额，「cos差」是该意图在叶里的行与叶里其余行的中心余弦、减去同样大小随机切分的基线。"
                     f"意图本身就是按语义分的，所以 cos差 几乎总是负的（全部 {len(allc)} 个意图×叶格子中位 {p50:.3f}），读相对位置："
                     f"低于全体 p10（{p10:.3f}）记「叶名描述的是另一回事」——另一个话题，或同一话题下的另一种功能，这时读左边的中心例子，不读叶名。\n\n"
                     + _md(show_nf[["叶名", "意图内行数", "占该意图%", "该意图占叶%", "叶主导意图", "cos差", "读法", "意图内中心例子", "叶其余中心例子"]]))
        fc = _snap_sort(facets[facets["L1"] == code])
        show = ["快照", "n"] + [c for f in FACET_SHOW for c in (f"{f}%", f"{f}_CI%")]
        P.append("\n**说法特征（快照内占比%，[95% Wilson]）**\n\n" + _md(fc[[c for c in show if c in fc.columns]]))
        fp = fpairs[fpairs["L1"] == code]
        if len(fp):
            cols = ["对比", "n_a", "n_b"] + [f"{f}_差pp" for f in FACET_SHOW]
            P.append("\n**对齐快照对的说法差（b−a，pp [95% Newcombe]，* = 区间不含 0）**\n\n" + _md(fp[cols]))
        lm = leafmix[leafmix["外层"] == code]
        if len(lm):
            P.append("\n**叶构成的对比（TVD 与同源噪声上界）**\n\n" + _md(lm[["对比", "n_a", "n_b", "内层TVD", "同源噪声上界", "超出噪声", "差最大的三个内层(b-a, *=区间不含0)"]]))
        cc = c2[c2["L1"] == code]
        if len(cc):
            P.append("\n**可分性（分类器双样本检验）**\n\n" + _md(cc[["对比", "a", "b", "每侧抽样n", "AUC", "零分布95分位", "p", "两侧可分(p<0.05)"]]))
        w = words[words["L1"] == code]
        if len(w):
            lines = []
            for pair, g in w.groupby("对比", sort=False):
                a_, b_ = g["a"].iloc[0], g["b"].iloc[0]
                wa = "、".join(f"{r['词']}({r['z']})" for _, r in g[g["方向"] == "偏a"].head(10).iterrows()) or "无显著词"
                wb = "、".join(f"{r['词']}({r['z']})" for _, r in g[g["方向"] == "偏b"].head(10).iterrows()) or "无显著词"
                lines.append(f"- **{pair}**：偏 {a_}：{wa}；偏 {b_}：{wb}")
            P.append("\n**区分用词（z 值，|z|≥1.96）**\n\n" + "\n".join(lines) + "\n")
        if code in fsum:
            fs = fsum[code]
            cb = json.loads((fdir / f"{code}_codebook.json").read_text(encoding="utf-8"))
            # 交付是中文：码本代理有的用英文写定义，译文放在 facets/codebook_zh.json；拉丁字母占比高的定义没有译文就拒绝渲染
            zh_path = fdir / "codebook_zh.json"
            zh = json.loads(zh_path.read_text(encoding="utf-8")).get(code, {}) if zh_path.exists() else {}
            defs = {}
            for c in cb["codes"]:
                d0 = c["definition"]
                latin = len(re.findall(r"[A-Za-z]", d0)) / max(len(d0), 1)
                assert latin < 0.3 or c["code"] in zh, f"{code}: definition of {c['code']} is not Chinese and has no translation in codebook_zh.json"
                defs[c["code"]] = zh.get(c["code"], d0)
            extra = set(zh) - set(defs)
            assert not extra, f"{code}: codebook_zh.json translates codes the codebook does not have: {sorted(extra)}"
            cbt = pd.DataFrame([{"码": c["code"], "子功能": c["name_zh"], "定义": defs[c["code"]]} for c in cb["codes"]])
            fcomp = pd.read_csv(fdir / f"{code}_composition_by_snapshot.csv")
            pooled = pd.read_csv(fdir / f"{code}_composition_pooled_weighted.csv")
            fpc = pd.read_csv(fdir / f"{code}_pair_contrasts.csv") if (fdir / f"{code}_pair_contrasts.csv").exists() else pd.DataFrame()
            byleaf = pd.read_csv(fdir / f"{code}_by_leaf.csv")
            xm_path = fdir / "crossmodel" / "summary.json"
            xm = json.loads(xm_path.read_text(encoding="utf-8")).get(code) if xm_path.exists() else None
            if xm:
                dis = "；".join(f"{k} {v} 行" for k, v in xm["top_disagreements"].items()) or "无"
                xm_txt = (f"**信度看跨模型**：另一家模型（DeepSeek `{xm['model']}`）拿同一份码本、同样不知道快照，独立标了一个分层子样本 "
                          f"n={xm['n_sub']}（每个快照最多 25 行），与三视角多数标签的 Cohen κ = {xm['kappa_claude_majority_vs_deepseek']}，"
                          f"原始一致 {xm['raw_agreement_%']}%；分歧（多数→DeepSeek）：{dis}。")
            else:
                xm_txt = "**跨模型信度尚未测**（`p5_facet_crossmodel.py`），下面的构成只能当作同一模型的读法。"
            P.append(f"\n**语用子功能（盲标）**：按快照分层抽样 {fs['n_sample']} 行，三个视角各标一遍（不知道快照、顺序打乱），"
                     f"码本先由另一名代理依据全部样本写出并试标修订。三视角 Fleiss κ = {fs['fleiss_kappa']}（三票全同 {fs['unanimous_%']}%，"
                     f"无多数 {fs['no_majority_rows']} 行，单列不并进任何码）——三个视角是同一个模型、只差一行读法提示，"
                     "码本又是按字面规则写的，所以这个 κ 量的是**同一模型的自洽**，不是信度（查过：各视角逐行手工给码，没有互相读文件）。"
                     f"{xm_txt}两家模型高度一致，说明码本的纳入/排除规则写得足够明确、换一个认真读码本的标注者会得到同一划分；"
                     "它不证明这套子功能是唯一正确的切法——码本是从这批样本里归纳出来的。\n\n" + _md(cbt))
            pv = fcomp.pivot_table(index="子功能名", columns="快照", values="占比%", fill_value=0)
            pv = pv[[s for s in SNAP_ORDER if s in pv.columns]]
            ns = fcomp.groupby("快照")["样本n"].first()
            pv.columns = [f"{c}(n={int(ns[c])})" for c in pv.columns]
            P.append("\n**子功能构成（快照内样本占比%）**\n\n" + _md(pv.reset_index()))
            P.append("\n**全意图合并（按抽样权重还原到总体）**\n\n" + _md(pooled.sort_values("加权占比%", ascending=False)[["子功能名", "加权占比%"]]))
            if len(fpc):
                sig = fpc[fpc["显著"]]
                P.append("\n**对齐快照对里显著的子功能差（b−a）**\n\n" + (_md(sig[["对比", "a", "b", "n_a", "n_b", "子功能名", "a%", "b%", "差pp(b-a)", "95CI低pp", "95CI高pp"]]) if len(sig) else "_（没有显著差。）_\n"))
            P.append("\n**子功能 × 叶（前 10 个叶，行内占比%，按抽样权重）**\n\n" + _md(byleaf.head(10)))
            gmd, gt = facet_groups(fdir, code)
            if gmd:
                P.append(gmd)
                group_tables[code] = gt
            f4 = fig_facet(fcomp, code, f"{nm}：盲标语用子功能在各快照的构成", img)
            P.append(f"\n![语用子功能](img/{f4.name})\n")

    # 8 速查
    P.append("\n## 8　其余意图速查（随机层的界面对比：2026搜索随机1w ↔ 助手随机1k）\n")
    rows = []
    for code, nm in l1name.items():
        r = {"意图": nm}
        s_ = struct[(struct["外层"] == code) & (struct["快照"] == SRC_ZH["2026search_rand"])]
        r["有效叶数(2026随机)"] = float(s_["有效内层数(exp熵)"].iloc[0]) if len(s_) else float("nan")
        lm = leafmix[(leafmix["外层"] == code) & (leafmix["对比"] == "界面·随机层")]
        r["叶构成TVD/噪声上界"] = f"{lm['内层TVD'].iloc[0]:.2f}/{lm['同源噪声上界'].iloc[0]:.2f}{'*' if bool(lm['超出噪声'].iloc[0]) else ''}" if len(lm) else "—"
        cc = c2[(c2["L1"] == code) & (c2["对比"] == "界面·随机层")]
        r["AUC(p)"] = f"{cc['AUC'].iloc[0]:.2f} ({cc['p'].iloc[0]:.2f})" if len(cc) else "—"
        fp = fpairs[(fpairs["L1"] == code) & (fpairs["对比"] == "界面·随机层")]
        r["咨询/求建议差"] = fp["咨询/求建议_差pp"].iloc[0] if len(fp) else "—"
        r["问句差"] = fp["问句_差pp"].iloc[0] if len(fp) else "—"
        rows.append(r)
    P.append("两侧各 ≥40 行才测，否则记 —。* = 叶构成差超出同源噪声上界。\n\n" + _md(pd.DataFrame(rows)))

    # 9 限制与出处
    P.append("\n## 9　读法、限制与方法出处\n")
    P.append(_narr(narr, "限制"))
    P.append("- 标签来自同一次 fast 模式运行（单标注员，kappa 不存在），所以「某条 query 属于这个意图」本身有误差；本文比较的是**同一套标签下**的快照差异。\n"
             "- 说法特征是词表代理，只用来定位差别；盲标子功能只做了三个宽意图，其余意图的「怎么问」差别只能读说法特征与样例。\n"
             "- 可分性检验显著只说明两侧在语义空间里分得开，不说明分开它们的是意图、话题还是说法；要连同构成与说法一起读。\n"
             "- 头部与语音头部的很多格子只有几十行甚至个位数，任何占比都配了区间，读的时候先看 n。\n\n")
    P.append("方法出处（均已核对原始出处）：\n\n" + "\n".join(f"- {t} <{u}>" for t, u in REFS) + "\n")
    P.append("\n## 附　复现\n\n```bash\n"
             f"P5_COHORT=fin8 HF_HOME=$(pwd)/.hf .venv/bin/python analysis/pooled5/p5_intent_structure.py {domain}\n"
             f"HF_HOME=$(pwd)/.hf .venv/bin/python analysis/pooled5/p5_leaf_namefit.py {domain}\n"
             f"HF_HOME=$(pwd)/.hf .venv/bin/python analysis/pooled5/p5_facet_aggregate.py {domain} <工作流结果.json>\n"
             f"HF_HOME=$(pwd)/.hf .venv/bin/python analysis/pooled5/p5_facet_crossmodel.py {domain} NEWS_EVENT VERDICT_QUESTION INVEST_ADVICE\n"
             f"P5_COHORT=fin8 HF_HOME=$(pwd)/.hf .venv/bin/python analysis/pooled5/p5_intent_structure_report.py {domain}\n```\n")
    out_md = post / f"{stem}_意图内部结构与代表性样例.zh.md"
    out_md.write_text("\n".join(P), encoding="utf-8")
    with pd.ExcelWriter(post / f"{stem}_意图内部结构与代表性样例.xlsx", engine="openpyxl") as xl:
        for name, df in [("取例方法对比", cmp_), ("代表性样例", ex), ("意图结构", struct), ("叶构成", comp), ("叶构成对比", leafmix),
                         ("说法特征", facets), ("说法特征对比", fpairs), ("区分用词", words), ("可分性", c2), ("叶名适用度", namefit)]:
            df.to_excel(xl, sheet_name=name, index=False)
        for code in fsum:
            for part in ("composition_by_snapshot", "pair_contrasts", "by_leaf", "by_l2"):
                p = fdir / f"{code}_{part}.csv"
                if p.exists():
                    pd.read_csv(p).to_excel(xl, sheet_name=f"{code[:10]}_{part[:18]}"[:31], index=False)
        for code, gt in group_tables.items():
            for part, df in gt.items():
                df.to_excel(xl, sheet_name=f"{code[:10]}_读法分组_{part}"[:31], index=False)
    print(f"{domain}: {out_md}  ({len(out_md.read_text(encoding='utf-8')):,} 字符)")
    return out_md


if __name__ == "__main__":
    for dom in sys.argv[1:]:
        build(dom)
