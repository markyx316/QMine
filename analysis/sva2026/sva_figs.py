# -*- coding: utf-8 -*-
"""Figures for SEARCH_VS_ASSISTANT_2026.zh.md. Every number is computed here from the tiered data,
the semantic-neighbour parquet and the intent table — never read from a text report.
Usage: python sva_figs.py 1 2 3   (4 5 6 need the unified-intent tables)"""
import os, sys, re
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from sva_common import *
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Hiragino Sans GB", "Heiti SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
OUT = f"{QM}/docs/img/sva2026"
os.makedirs(OUT, exist_ok=True)
DOM5 = list(CAT)
SC = {"搜索top1000": "#1f5fa8", "搜索top10k": "#7fb3e6", "助手top1k": "#d9480f", "助手random1k": "#f59f00"}
SNAME = {"搜索top1000": "搜索 前1000", "搜索top10k": "搜索 前1万", "助手top1k": "助手 头部1k", "助手random1k": "助手 随机1k(长尾)"}


def save(fig, name):
    p = f"{OUT}/{name}.png"
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


def fig1():
    a, _ = load()
    tiers = ["user", "S1_system_repeated", "S2_system_template", "S3_card_passage", "S4_suggested_chip", "S5_headline", "C1_content_free", "C2_feed_control"]
    lab = {"user": "用户输入", "S1_system_repeated": "S1 跨类目重复的系统串", "S2_system_template": "S2 模板/功能入口",
           "S3_card_passage": "S3 百科卡片段落", "S4_suggested_chip": "S4 推荐追问", "S5_headline": "S5 热点标题",
           "C1_content_free": "C1 无内容", "C2_feed_control": "C2 信息流反馈"}
    col = {"user": "#2b8a3e", "S1_system_repeated": "#495057", "S2_system_template": "#868e96", "S3_card_passage": "#c08457",
           "S4_suggested_chip": "#f08c00", "S5_headline": "#7048e8", "C1_content_free": "#ced4da", "C2_feed_control": "#e64980"}
    groups = [(d, a[(a.l1 == CAT[d]) & (a.snapshot == "top1k")]) for d in DOM5]
    groups.append(("全部33类", a[a.snapshot == "top1k"]))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    for ax, (metric, title) in zip(axes, (("pv", "按流量(PV)"), ("rows", "按行数"))):
        left = np.zeros(len(groups))
        for t in tiers:
            vals = []
            for _, g in groups:
                tot = g.search_num.sum() if metric == "pv" else len(g)
                v = g.search_num[g.tier == t].sum() if metric == "pv" else (g.tier == t).sum()
                vals.append(v / tot * 100)
            vals = np.array(vals)
            ax.barh(range(len(groups)), vals, left=left, color=col[t], label=lab[t], edgecolor="white", linewidth=0.5)
            for i, v in enumerate(vals):
                if t == "user" and v > 0:
                    ax.text(left[i] + v / 2, i, f"{v:.1f}%", ha="center", va="center", color="white", fontsize=9, fontweight="bold")
            left += vals
        ax.set_xlim(0, 100)
        ax.set_yticks(range(len(groups)), [g for g, _ in groups])
        ax.invert_yaxis()
        ax.set_title(f"助手头部(top1k)的构成 — {title}")
        ax.set_xlabel("占比 %")
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9, frameon=False)
    fig.suptitle("图1  清洗后：助手头部流量里真正由用户输入的部分", y=1.03, fontsize=13)
    save(fig, "fig1_助手头部构成_清洗分层")


def fig2():
    a, s = load()
    src = open(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "sva_metrics.py"), encoding="utf-8").read()
    ns = {}
    exec(src[src.index("M={"):src.index("C={d:")], {}, ns)
    M = ns["M"]
    marks = [("疑问句", M["疑问句"]), ("是非核实(吗/是否/真的)", r"(是不是|是否|真的|吗[？?。，,]?$|对吗|对不对)"),
             ("委托/动作开头(帮我/写/画/分析)", r"^(请你?|帮我|帮忙|给我|麻烦|你给我|你帮我|能不能帮我|能否帮我|能帮我|可以帮我|写|画|生成|制作|分析|预测|推荐|总结|概括|解读|翻译|计算|对比|比较|列出|整理)"),
             ("第一人称(我/我的)", M["第一人称"]), ("带数量+单位", r"\d+(\.\d+)?\s?(岁|周|天|个月|年|mg|毫克|克|斤|公斤|kg|元|块|万|千|cm|厘米|米|分|次|粒|片|度|℃|mmol|%)"),
             ("输出约束(字数/格式)", M["输出约束"]), ("资源词(在线观看/全集…)", M["资源词"]), ("导航词(官网/下载/入口…)", M["导航词"])]
    C = {d: {k: v["query"].astype(str) for k, v in cells(a, s, d).items()} for d in DOM5}
    C["5域合计"] = {k: pd.concat([C[d][k] for d in DOM5]) for k in SURF}
    groups = DOM5 + ["5域合计"]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.5))
    for ax, (name, pat) in zip(axes.flat, marks):
        for j, k in enumerate(SURF):
            ys = [C[g][k].str.contains(pat, regex=True).mean() * 100 for g in groups]
            ax.scatter(np.arange(len(groups)) + (j - 1.5) * 0.16, ys, color=SC[k], s=28, label=SNAME[k], zorder=3)
        ax.set_xticks(range(len(groups)), groups, fontsize=9)
        ax.set_title(name, fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylabel("行占比 %", fontsize=9)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("图2  同一话题，四个切片的查询形态（清洗后，user 行；行占比）", fontsize=13)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=10)
    save(fig, "fig2_查询形态_四个切片")


def fig3():
    nn = pd.read_parquet(f"{SP}/sva_semantic_nn_v3.parquet")
    ctrl = pd.read_parquet(f"{SP}/sva_semantic_nn_controls_v3.parquet")
    bk = [(0, 6, "≤6字"), (7, 10, "7–10"), (11, 15, "11–15"), (16, 25, "16–25"), (26, 10**6, ">25")]
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.8), sharey=True)
    series = [("对照B", ctrl[ctrl.kind == "对照B"], "#1f5fa8", "搜索 最深1000条→其余搜索(对照)"),
              ("助手top1k", nn[nn.surface == "助手top1k"], "#d9480f", "助手 头部1k→搜索前1万"),
              ("助手random1k", nn[nn.surface == "助手random1k"], "#f59f00", "助手 随机1k→搜索前1万")]
    for ax, d in zip(axes, DOM5):
        for key, df, c, lab in series:
            x = df[df.domain == d]
            ln = x["query"].astype(str).str.len().values
            ys, xs = [], []
            for i, (lo, hi, _) in enumerate(bk):
                m = (ln >= lo) & (ln <= hi)
                if m.sum() >= 20:
                    ys.append((x.sim.values[m] < 0.70).mean() * 100)
                    xs.append(i)
            ax.plot(xs, ys, marker="o", color=c, label=lab)
        ax.set_xticks(range(len(bk)), [b[2] for b in bk], fontsize=9)
        ax.set_title(d)
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("在搜索前1万里找不到近邻的比例 %\n(最近邻余弦 < 0.70)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.12))
    fig.suptitle("图3  控制长度后，助手查询在搜索里有没有“近亲”（bge-base-zh-v1.5；每组 n≥20 才画点）", y=1.04, fontsize=13)
    save(fig, "fig3_语义近邻_按长度控制")


def fig4():
    sh = pd.read_csv(f"{SP}/sva_intent2_shares.csv")
    fm = frame()
    codes = [c for c in fm.FRAME if c != "U13"]
    surfs = [x for x in ["search_top1000", "search_top10k", "assistant_top1k", "assistant_random1k"] if x in set(sh.surface)]
    sl = {"search_top1000": "搜索\n前1000", "search_top10k": "搜索\n前1万", "assistant_top1k": "助手\n头部1k", "assistant_random1k": "助手\n随机1k"}
    fig, axes = plt.subplots(1, 6, figsize=(20, 6.5), sharey=True)
    for ax, d in zip(axes, DOM5 + ["合计"]):
        mat = np.array([[sh[(sh.view == "interpretable") & (sh.domain == d) & (sh.surface == sf) & (sh.code == c)].share.iloc[0] * 100 for sf in surfs] for c in codes])
        ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=60, aspect="auto")
        for i in range(len(codes)):
            for j in range(len(surfs)):
                ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center", fontsize=8, color="black" if mat[i, j] < 40 else "white")
        ax.set_xticks(range(len(surfs)), [sl[x] for x in surfs], fontsize=8)
        ax.set_title("5域合计" if d == "合计" else d)
    axes[0].set_yticks(range(len(codes)), [f"{c} {fm.FRAME[c][0]}" for c in codes], fontsize=9)
    fig.suptitle("图4  统一意图框架下的占比（可判定行，去掉 U13；同一模型盲标；行占比 %；U05 未扣除审计发现的漏网热点标题）", fontsize=13)
    fig.tight_layout()
    save(fig, "fig4_统一意图占比")


def fig5():
    A = pd.read_parquet(f"{SP}/sva_intent2_assistant_nn.parquet")
    LV = ["A 原样搜索词", "B 近似改写(≥0.80)", "C 同题延伸(0.70–0.80)", "D 搜索无近邻(<0.70)", "E 个人处境/个案判断", "F 对话/委托"]
    col = ["#1f5fa8", "#4dabf7", "#a5d8ff", "#adb5bd", "#f08c00", "#c92a2a"]
    rows = []
    for d in DOM5 + ["合计"]:
        for sf, nm in (("assistant_top1k", "头部1k"), ("assistant_random1k", "随机1k")):
            x = A[(A.surface == sf) & ((A.domain == d) if d != "合计" else True)].dropna(subset=["level"])
            rows.append((f"{'5域合计' if d == '合计' else d}·{nm}", [(x.level == lv).mean() * 100 for lv in LV]))
    fig, ax = plt.subplots(figsize=(12, 6))
    left = np.zeros(len(rows))
    for k, lv in enumerate(LV):
        v = np.array([r[1][k] for r in rows])
        ax.barh(range(len(rows)), v, left=left, color=col[k], label=lv, edgecolor="white", linewidth=0.5)
        for i, vv in enumerate(v):
            if vv >= 6:
                ax.text(left[i] + vv / 2, i, f"{vv:.0f}", ha="center", va="center", fontsize=8, color="white" if k in (0, 3, 5) else "black")
        left += v
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("行占比 %")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    ax.set_title("图5  助手查询离“搜索式表达”有多远：六级阶梯（优先级 F>E>A>B>C>D）")
    save(fig, "fig5_搜索到助手的阶梯")


def fig6():
    pw = pd.read_csv(f"{SP}/sva_intent2_wrap_pairs.csv")
    A = pd.read_parquet(f"{SP}/sva_intent2_assistant_nn.parquet")
    fm = frame()
    codes = list(fm.FRAME)
    near = A[(A.sim >= 0.70) & (A.sim < 0.999)].dropna(subset=["u_ds", "u_nn"])
    pairs = pd.concat([pw.rename(columns={"u_core": "src", "u_wrap": "dst"})[["surface", "src", "dst"]],
                       near.rename(columns={"u_nn": "src", "u_ds": "dst"})[["surface", "src", "dst"]]], ignore_index=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2))
    for ax, (sf, nm) in zip(axes, (("assistant_top1k", "助手头部1k"), ("assistant_random1k", "助手随机1k"))):
        x = pairs[pairs.surface == sf]
        mat = pd.crosstab(x.src, x.dst).reindex(index=codes, columns=codes, fill_value=0).values.astype(float)
        rowsum = mat.sum(1, keepdims=True)
        pct = np.divide(mat, rowsum, out=np.zeros_like(mat), where=rowsum > 0) * 100
        ax.imshow(pct, cmap="Blues", vmin=0, vmax=80)
        for i in range(len(codes)):
            for j in range(len(codes)):
                if mat[i, j] >= 5:
                    ax.text(j, i, f"{pct[i, j]:.0f}", ha="center", va="center", fontsize=7, color="white" if pct[i, j] > 45 else "black")
        ax.set_xticks(range(len(codes)), codes, fontsize=8, rotation=90)
        ax.set_yticks(range(len(codes)), [f"{c} {fm.FRAME[c][0]} (n={int(rowsum[i, 0])})" for i, c in enumerate(codes)], fontsize=8)
        ax.set_xlabel("助手查询的意图")
        ax.set_title(f"{nm}：n={len(x):,} 对")
    axes[0].set_ylabel("对应搜索查询的意图（行内 %，仅标注 ≥5 对的格）")
    fig.suptitle("图6  同一需求从搜索式写法到助手式写法：意图怎么变（包装对 + 语义近邻对 0.70≤sim<0.999）", fontsize=13)
    fig.tight_layout()
    save(fig, "fig6_意图转移矩阵")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        globals()[f"fig{arg}"]()
