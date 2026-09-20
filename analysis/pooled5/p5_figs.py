# -*- coding: utf-8 -*-
"""Figures for the POOLED-5 report. Every number is read from the canonical tables in work/,
never from prose. Usage: python analysis/pooled5/p5_figs.py [1 2 3 4 5]"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pooled5_common import DOMAINS, SOURCES, SRC_ZH, WORK, all_rows, cohort, work_file
plt.rcParams["font.sans-serif"] = ["Hiragino Sans GB", "Heiti SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/img/pooled5"
OUT.mkdir(parents=True, exist_ok=True)
SC = {"2025search": "#1f5fa8", "2026search": "#7fb3e6", "assistant_top": "#d9480f",
      "assistant_random": "#f59f00", "assistant_voice": "#2b8a3e"}


def fig1():
    fig, axes = plt.subplots(1, 5, figsize=(18, 4.2), sharey=False)
    for ax, dom in zip(axes, DOMAINS):
        a = all_rows(dom)
        g = a.groupby("source", sort=False).agg(kept=("kept", "sum"), n=("kept", "size"))
        g = g.reindex([s for s in SOURCES if s in g.index])
        ax.bar(range(len(g)), g.kept, color=[SC[s] for s in g.index], label="进入挖掘")
        ax.bar(range(len(g)), g.n - g.kept, bottom=g.kept, color="#adb5bd", label="清洗剔除")
        pv = a[~a.kept].groupby("source")["pv_raw"].sum() / a.groupby("source")["pv_raw"].sum()
        for i, (src, k, n) in enumerate(zip(g.index, g.kept, g.n)):
            if n - k:
                lab = f"-{n-k}行"
                if pd.notna(pv.get(src)) and pv.get(src, 0) > 0.02:
                    lab += f"\n(-{pv[src]*100:.0f}% PV)"
                ax.text(i, n + 150, lab, ha="center", fontsize=8, color="#495057")
        ax.set_xticks(range(len(g)))
        ax.set_xticklabels([SRC_ZH[s] for s in g.index], rotation=35, ha="right", fontsize=8)
        ax.set_title(dom, fontsize=11)
        ax.set_ylim(0, 12200)
    axes[0].set_ylabel("行数")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("图1  五个领域的合并语料构成（每个来源的行数与清洗剔除量）", fontsize=13)
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    fig.savefig(OUT / "fig1_语料构成.png", dpi=160, bbox_inches="tight")
    print("wrote fig1")


def fig2():
    for dom in cohort():
        f = WORK / dom / "shares_td_l1_name.csv"
        if not f.exists():
            continue
        s = pd.read_csv(f)
        piv = s.pivot_table(index="label", columns="source", values="row_share", fill_value=0) * 100
        piv = piv[[c for c in SOURCES if c in piv.columns]]
        piv = piv.loc[piv.max(axis=1).sort_values(ascending=False).index]
        fig, ax = plt.subplots(figsize=(1.6 + 1.5 * piv.shape[1], 0.42 * len(piv) + 1.6))
        im = ax.imshow(piv.values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=min(60, piv.values.max()))
        ax.set_xticks(range(piv.shape[1]))
        ax.set_xticklabels([SRC_ZH[c] for c in piv.columns], fontsize=9)
        ax.set_yticks(range(len(piv)))
        ax.set_yticklabels([x[:22] for x in piv.index], fontsize=9)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8,
                        color="white" if v > 35 else "#212529")
        ax.set_title(f"{dom}：各来源的意图占比（行占比 %，同一套体系）", fontsize=12)
        fig.colorbar(im, ax=ax, shrink=0.7, label="%")
        fig.tight_layout()
        fig.savefig(OUT / f"fig2_意图占比_{dom}.png", dpi=160, bbox_inches="tight")
    print("wrote fig2 (per domain)")


def fig3():
    rows = []
    for dom in cohort():
        f = WORK / dom / "contrast_summary_td_l1_name.csv"
        if f.exists():
            t = pd.read_csv(f)
            t["domain"] = dom
            rows.append(t)
    if not rows:
        return
    t = pd.concat(rows)
    order = t.groupby(["a", "b", "why"], sort=False).size().index.tolist()
    fig, ax = plt.subplots(figsize=(13, 5))
    w = 0.16
    for i, dom in enumerate(DOMAINS):
        sub = t[t.domain == dom].set_index(["a", "b", "why"])
        vals = [sub.tvd.get(k, np.nan) for k in order]
        ax.bar(np.arange(len(order)) + i * w, vals, w, label=dom)
    ax.set_xticks(np.arange(len(order)) + 2 * w)
    ax.set_xticklabels([f"{SRC_ZH[a]}→{SRC_ZH[b]}\n{why}" for a, b, why in order], fontsize=8)
    ax.set_ylabel("意图构成的总变差距离 TVD\n(需要换类的行占比)")
    ax.set_title("图3  哪一种差异最大：时间、界面、深度、输入方式", fontsize=13)
    ax.legend(frameon=False, ncol=5, fontsize=9)
    ax.grid(axis="y", alpha=.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_对比距离.png", dpi=160, bbox_inches="tight")
    print("wrote fig3")


def fig4():
    p = work_file("semantic_nn.parquet")
    if not p.exists():
        return
    nn = pd.read_parquet(p)
    bands = [(0.999, 1.01, "完全相同"), (0.90, 0.999, "≥0.90"), (0.80, 0.90, "0.80–0.90"),
             (0.70, 0.80, "0.70–0.80"), (-1, 0.70, "<0.70 无近邻")]
    cols = ["#1864ab", "#4c93d1", "#a5c8e8", "#ffd8a8", "#e8590c"]
    fig, axes = plt.subplots(1, 5, figsize=(18, 4.4), sharey=True)
    for ax, dom in zip(axes, DOMAINS):
        d = nn[nn.domain == dom]
        srcs = [s for s in SOURCES if (d.source == s).any()]
        bottom = np.zeros(len(srcs))
        for (lo, hi, lab), c in zip(bands, cols):
            v = np.array([((d[d.source == s].sim >= lo) & (d[d.source == s].sim < hi)).mean() * 100 for s in srcs])
            ax.bar(range(len(srcs)), v, bottom=bottom, color=c, label=lab)
            bottom += v
        ax.set_xticks(range(len(srcs)))
        ax.set_xticklabels([SRC_ZH[s] for s in srcs], rotation=35, ha="right", fontsize=8)
        ax.set_title(dom, fontsize=11)
    axes[0].set_ylabel("占比 %")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("图4  每一行在 2026 搜索里能不能找到近邻（bge-base-zh 余弦；2025搜索是时间对照）", fontsize=13)
    fig.tight_layout(rect=[0, 0.06, 1, 0.93])
    fig.savefig(OUT / "fig4_语义近邻.png", dpi=160, bbox_inches="tight")
    print("wrote fig4")


def fig5(markers=("疑问句", "第一人称", "祈使/委托", "是非核实", "回指上文", "输出约束")):
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=False)
    for ax, mk in zip(axes.ravel(), markers):
        for i, dom in enumerate(DOMAINS):
            f = WORK / dom / "form.csv"
            if not f.exists():
                continue
            t = pd.read_csv(f).set_index("source")
            srcs = [s for s in SOURCES if s in t.index]
            ax.plot(range(len(srcs)), [t.loc[s, mk] * 100 for s in srcs], marker="o", label=dom)
            ax.set_xticks(range(len(srcs)))
            ax.set_xticklabels([SRC_ZH[s] for s in srcs], rotation=25, ha="right", fontsize=8)
        ax.set_title(mk, fontsize=11)
        ax.grid(alpha=.3)
        ax.set_ylabel("行占比 %")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("图5  写法标记在各来源上的占比（正则规则，只作相对比较）", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "fig5_形态标记.png", dpi=160, bbox_inches="tight")
    print("wrote fig5")


if __name__ == "__main__":
    want = sys.argv[1:] or ["1", "2", "3", "4", "5"]
    for k in want:
        globals()[f"fig{k}"]()
