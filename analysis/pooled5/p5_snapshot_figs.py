# -*- coding: utf-8 -*-
"""逐领域的四张图，全部从 p5_snapshot_classes.py 写出的表里读数，不从正文读。

    python analysis/pooled5/p5_snapshot_figs.py [domain ...]
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pooled5_common import SOURCES, SRC_ZH, WORK, available, run_dir

plt.rcParams["font.sans-serif"] = ["Hiragino Sans GB", "Heiti SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
ROOT = Path(__file__).resolve().parents[2]
SC = {"2025search": "#1f5fa8", "2026search": "#7fb3e6", "assistant_top": "#d9480f",
      "assistant_random": "#f59f00", "assistant_voice": "#2b8a3e"}
LEVELS = [("td_l1", "自上而下 L1 意图"), ("td_l2", "自上而下 L2 子意图"),
          ("bu_family_final", "自下而上 家族"), ("bu_leaf", "自下而上 叶")]


def _dirs(domain: str) -> tuple[Path, Path]:
    p = WORK / domain / "snapshot_classes"
    out = run_dir(domain) / "postprocessed" / "img"
    out.mkdir(parents=True, exist_ok=True)
    return p, out


def _srcs(m: pd.DataFrame) -> list[str]:
    return [s for s in SOURCES if f"{SRC_ZH[s]}_占比%" in m.columns]


def _heat(ax, M, rows, cols, fmt, cmap, norm, small, text=None):
    """text 与 M 分开：均衡指数图的颜色走 log2 并且被夹到 1/64，直接按夹过的值印字会把
    「这个快照 0 条」印成 0.02。印字一律用未夹的原值。"""
    T = M if text is None else text
    ax.imshow(M, aspect="auto", cmap=cmap, norm=norm)
    # 字色按格子的实际亮度选，不按数值区间选。之前那版写死「norm<0.12 也用白字」——那条是为发散
    # 色标（低值=深蓝）写的，套到顺序色标（低值=浅黄）上就成了浅黄底白字，整片读不出来。
    cm = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap

    def ink(v):
        r, g, b, _ = cm(norm(v))
        return "white" if (0.299 * r + 0.587 * g + 0.114 * b) < 0.5 else "#212529"
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, fontsize=9)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[:26] for r in rows], fontsize=small)
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = M[i, j]
            ax.text(j, i, fmt(T[i, j]), ha="center", va="center", fontsize=small - 0.5,
                    color=ink(v))
    ax.set_xticks(np.arange(-.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", length=0)


def fig_shares(domain: str, level: str, zh: str, tag: str) -> Path:
    p, out = _dirs(domain)
    m = pd.read_csv(p / f"matrix_{level}.csv").sort_values("全语料条数", ascending=False)
    srcs = _srcs(m)
    M = m[[f"{SRC_ZH[s]}_占比%" for s in srcs]].to_numpy(dtype=float)
    h = max(3.0, 0.34 * len(m) + 1.8)
    fig, ax = plt.subplots(figsize=(1.65 * len(srcs) + 4.6, h))
    _heat(ax, M, list(m["类目"]), [SRC_ZH[s] for s in srcs],
          lambda v: f"{v:.1f}" if v >= 0.05 else ("0" if v == 0 else "·"),
          "YlGnBu", plt.Normalize(0, min(60.0, float(M.max()) or 1.0)), 8.5)
    ax.set_title(f"{domain}　{zh}在每个快照里的占比（%，快照内占比；按全语料规模排序）", fontsize=12, pad=12)
    fig.tight_layout()
    f = out / f"{tag}_占比热图.png"
    fig.savefig(f, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return f


def fig_index(domain: str, level: str, zh: str, tag: str) -> Path:
    """均衡指数：快照内占比 ÷ 各快照占比的未加权均值。1 = 各快照一样常见。"""
    p, out = _dirs(domain)
    m = pd.read_csv(p / f"matrix_{level}.csv").sort_values("全语料条数", ascending=False)
    srcs = _srcs(m)
    M = m[[f"{SRC_ZH[s]}_均衡指数" for s in srcs]].to_numpy(dtype=float)
    L = np.log2(np.clip(M, 1 / 64, 64))
    h = max(3.0, 0.34 * len(m) + 1.8)
    fig, ax = plt.subplots(figsize=(1.65 * len(srcs) + 4.6, h))
    _heat(ax, L, list(m["类目"]), [SRC_ZH[s] for s in srcs],
          lambda v: "0" if v == 0 else (f"{v:.2f}" if v < 10 else f"{v:.0f}"),
          "RdBu_r", TwoSlopeNorm(vmin=-3, vcenter=0, vmax=3), 8.5, text=M)
    ax.set_title(f"{domain}　{zh}的均衡指数（该快照占比 ÷ 各快照占比的未加权均值）\n"
                 f"1.00 = 各快照一样常见；红 = 这个快照里更常见；蓝 = 更少见（色标按 log2 展开）",
                 fontsize=11, pad=12)
    fig.tight_layout()
    f = out / f"{tag}_均衡指数热图.png"
    fig.savefig(f, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return f


#: 五快照的领域有 10 对，横轴放不下全名——之前那版直接糊成一团。缩写只用在这张图的刻度上。
SHORT = {"2025搜索": "25搜", "2026搜索": "26搜", "助手头部1k": "头部",
         "助手随机1k": "随机", "助手语音1k": "语音",
         # 8 快照（金融8）新增的三个
         "2025搜索随机1w": "25随", "2026搜索随机1w": "26随", "助手语音头部1k": "语头"}

#: 超过这个对数就不画柱子了。8 个快照是 C(8,2)=28 对，28 根柱子 x 4 个层级，横轴刻度必然糊成
#: 一团——上一次 10 对已经要靠缩写加 45 度才救回来。改画下三角矩阵：同样的数，读起来是
#: 「哪一对远」而不是「第 17 根柱子是谁」。5 快照及以下仍走柱状图，所以已交付的七个域逐字节不变。
TVD_MATRIX_MIN_PAIRS = 15


def _fig_tvd_matrix(domain: str, t: pd.DataFrame, out: Path) -> Path:
    """下三角热力矩阵：格子是 TVD，超出同源噪声上界的格子描红边，格内第二行是上界。"""
    srcs = list(dict.fromkeys(list(t["a"]) + list(t["b"])))
    n = len(srcs)
    idx = {s: i for i, s in enumerate(srcs)}
    fig, axes = plt.subplots(2, 2, figsize=(7.2 * 2, 6.4 * 2))
    vmax = max(0.1, float(t["TVD"].max()))
    for ax, (_, zh) in zip(axes.ravel(), LEVELS):
        g = t[t["层级"] == zh]
        M = np.full((n, n), np.nan)
        for _, r in g.iterrows():
            i, j = idx[r["a"]], idx[r["b"]]
            M[max(i, j), min(i, j)] = r["TVD"]
        im = ax.imshow(M, cmap="magma_r", vmin=0, vmax=vmax)
        for _, r in g.iterrows():
            i, j = max(idx[r["a"]], idx[r["b"]]), min(idx[r["a"]], idx[r["b"]])
            v, nz = float(r["TVD"]), float(r["同源噪声上界"])
            rgba = im.cmap(im.norm(v))
            ink = "white" if (0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]) < 0.5 else "#212529"
            ax.text(j, i - 0.13, f"{v:.2f}", ha="center", va="center", fontsize=8.5, color=ink)
            ax.text(j, i + 0.21, f"噪声{nz:.2f}", ha="center", va="center", fontsize=6.2, color=ink)
            if bool(r["超出噪声"]):
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                           edgecolor="#2f9e44", lw=2.2))
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(srcs, fontsize=7.5, rotation=45, ha="right")
        ax.set_yticklabels(srcs, fontsize=7.5)
        ax.set_title(zh, fontsize=11)
        ax.set_xticks(np.arange(-.5, n, 1), minor=True)
        ax.set_yticks(np.arange(-.5, n, 1), minor=True)
        ax.grid(which="minor", color="white", lw=1.2)
        ax.tick_params(which="minor", length=0)
    fig.suptitle(f"{domain}　每一对快照的类目分布距离 TVD（绿框 = 超出同源噪声上界，可读）",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    f = out / "快照两两距离.png"
    fig.savefig(f, dpi=170, bbox_inches="tight"); plt.close(fig)
    return f


def fig_tvd(domain: str) -> Path:
    p, out = _dirs(domain)
    t = pd.read_csv(p / "pairwise_tvd.csv")
    n_pairs = int(t[t["层级"] == t["层级"].iloc[0]].shape[0])
    if n_pairs >= TVD_MATRIX_MIN_PAIRS:
        return _fig_tvd_matrix(domain, t, out)
    fig, axes = plt.subplots(1, 4, figsize=(20 if n_pairs <= 6 else 24, 5.0), sharey=True)
    for ax, (_, zh) in zip(axes, LEVELS):
        g = t[t["层级"] == zh].reset_index(drop=True)
        x = np.arange(len(g))
        ax.bar(x, g["TVD"], color="#4c6ef5", width=.62,
               yerr=[g["TVD"] - g["TVD低"], g["TVD高"] - g["TVD"]],
               error_kw=dict(ecolor="#343a40", capsize=3, lw=1))
        for i, v in enumerate(g["同源噪声上界"]):
            ax.plot([i - .36, i + .36], [v, v], color="#e03131", lw=2,
                    label="同源噪声上界（对半切）" if i == 0 else None)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{SHORT.get(a, a)}→{SHORT.get(b, b)}" for a, b in zip(g["a"], g["b"])],
                           fontsize=8.5, rotation=45, ha="right")
        ax.set_title(zh, fontsize=11)
        ax.set_ylim(0, max(0.8, float(t["TVD高"].max()) + .06))
        ax.grid(axis="y", alpha=.25)
    axes[0].set_ylabel("总变差距离 TVD")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    # 缩写说明只在刻度真的用了缩写时才印：健康这类来源名不在 SHORT 里的域，刻度是全名，
    # 「25搜/26搜 = 两个搜索快照」对它是错话。用了缩写的域（五域、新两域）标题逐字不变。
    used_short = any(x in SHORT for x in list(t["a"]) + list(t["b"]))
    fig.suptitle(f"{domain}　快照两两之间的类目分布距离（误差棒 = 自助法 95% 区间；"
                 f"红线以下的距离不可读）"
                 + ("　　刻度缩写：25搜/26搜 = 两个搜索快照，头部/随机/语音 = 助手三层" if used_short else ""),
                 fontsize=12.5)
    fig.tight_layout()
    f = out / "快照两两距离.png"
    fig.savefig(f, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return f


# 界面散点的轴名。默认值是五域/七域/金融8 一直在用的原句（逐字不变）；健康是同一周的两份单一导出，
# 「2025+2026 合并」「各层合并」对它都是错的，所以按域覆盖。
IFACE_AXES_DEFAULT = ("搜索（2025+2026 合并）内占比 %", "助手（各层合并）内占比 %")
IFACE_AXES = {"健康": ("健康搜索周榜（2026-09-04 至 09-10）内占比 %", "健康AI管家周榜（同一周）内占比 %"),
              "医疗随机": ("传统搜索随机1w（2026-09-14）内占比 %", "健康管家随机1w（同一天）内占比 %"),
              "医疗3": ("传统搜索随机1w（2026-09-14）内占比 %", "健康管家两个快照合并内占比 %")}


def fig_iface(domain: str) -> Path:
    """搜索占比 vs 助手占比的散点：对角线 = 两界面一样。n 最大的切法，所以最有把握。"""
    p, out = _dirs(domain)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.2))
    for ax, (level, zh) in zip(axes, [("td_l1", "自上而下 L1 意图"), ("bu_leaf", "自下而上 叶")]):
        t = pd.read_csv(p / f"interface_{level}.csv")
        sig, ns = t[t["显著"]], t[~t["显著"]]
        lim = max(1.2 * max(t["搜索占比%"].max(), t["助手占比%"].max()), 1.0)
        ax.plot([0.008, lim], [0.008, lim], color="#adb5bd", lw=1, ls="--", zorder=1)
        ax.scatter(ns["搜索占比%"].clip(lower=0.008), ns["助手占比%"].clip(lower=0.008),
                   s=18, c="#ced4da", edgecolor="#868e96", lw=.5, zorder=2,
                   label=f"Newcombe 区间含 0（{len(ns)}）")
        ax.scatter(sig["搜索占比%"].clip(lower=0.008), sig["助手占比%"].clip(lower=0.008),
                   s=34, c=["#e03131" if d > 0 else "#1971c2" for d in sig["差_pp"]],
                   edgecolor="white", lw=.6, zorder=3,
                   label=f"区间不含 0（{len(sig)}；红=助手更高）")
        big = t.reindex(t["差_pp"].abs().sort_values(ascending=False).index).head(6)
        # 标注错开垂直偏移：同一片区域里两个点的标签会重叠，之前那版就糊成一团
        for i, (_, r) in enumerate(big.iterrows()):
            x, y = max(r["搜索占比%"], 0.009), max(r["助手占比%"], 0.009)
            right = x < lim ** 0.5                       # 靠右的点把标签放到左边，免得出画面
            dy = (9, -13, 19, -23, 30, -33)[i % 6]
            ax.annotate(str(r["类目"])[:13], (x, y), fontsize=7.5, color="#343a40",
                        xytext=(5 if right else -5, dy), textcoords="offset points",
                        ha="left" if right else "right", va="center", clip_on=False,
                        arrowprops=dict(arrowstyle="-", lw=.6, color="#adb5bd",
                                        shrinkA=0, shrinkB=3))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.008, lim)
        ax.set_ylim(0.008, lim)
        ax.set_aspect("equal")
        # 默认的 LogFormatter 用 mathtext 的 U+2212 负号，中文字体没有这个字形（会渲染成方块），
        # 所以刻度标签在这里显式写成十进制。
        ticks = [t for t in (0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100) if 0.008 <= t <= lim]
        for setter, labeller in ((ax.set_xticks, ax.set_xticklabels), (ax.set_yticks, ax.set_yticklabels)):
            setter(ticks)
            labeller([("%g" % t) for t in ticks], fontsize=8)
        ax.minorticks_off()
        xl, yl = IFACE_AXES.get(domain, IFACE_AXES_DEFAULT)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(zh, fontsize=11)
        # scatter 传入颜色列表时图例只会取第一个颜色，所以三个图例项各用一个代理句柄
        handles = [Line2D([], [], marker="o", ls="", ms=5, mfc="#ced4da", mec="#868e96",
                          label=f"Newcombe 区间含 0（{len(ns)}）"),
                   Line2D([], [], marker="o", ls="", ms=6, mfc="#e03131", mec="white",
                          label=f"区间不含 0，助手更高（{int((sig['差_pp'] > 0).sum())}）"),
                   Line2D([], [], marker="o", ls="", ms=6, mfc="#1971c2", mec="white",
                          label=f"区间不含 0，搜索更高（{int((sig['差_pp'] < 0).sum())}）")]
        ax.legend(handles=handles, frameon=False, fontsize=8, loc="upper left")
        ax.grid(alpha=.2, which="major")
    fig.suptitle(f"{domain}　每个类在两个界面上的占比（对数轴；对角线 = 两界面一样；"
                 f"0 条的点压在 0.008% 的边上）", fontsize=13)
    fig.tight_layout()
    f = out / "界面散点.png"
    fig.savefig(f, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return f


def build(domain: str) -> list[Path]:
    fs = [fig_shares(domain, "td_l1", "自上而下 L1 意图", "L1意图"),
          fig_index(domain, "td_l1", "自上而下 L1 意图", "L1意图"),
          fig_shares(domain, "bu_leaf", "自下而上 聚类叶", "叶"),
          fig_index(domain, "bu_leaf", "自下而上 聚类叶", "叶"),
          fig_tvd(domain), fig_iface(domain)]
    print(f"{domain}: " + " · ".join(f.name for f in fs))
    return fs


if __name__ == "__main__":
    for d in (sys.argv[1:] or available()):
        build(d)
