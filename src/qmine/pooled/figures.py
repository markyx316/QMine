"""Figures, drawn from the tables and never from the prose.

Two traps are encoded here rather than left to whoever edits next:

* **Colour and text are separate matrices.** The balance-index map takes log2 of
  a value clipped to [1/64, 64] so the diverging scale works; printing the
  clipped number turns "this snapshot has zero rows" into "0.02". The text
  matrix is the unclipped one.
* **Ink colour is chosen from the rendered cell, not from the value.** A rule
  written for a diverging scale (low = dark blue, so white ink) produced
  white-on-pale-yellow across a whole sequential map.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .manifest import SnapshotManifest

_FONTS = ["Hiragino Sans GB", "Heiti SC", "Microsoft YaHei", "Noto Sans CJK SC",
          "Arial Unicode MS", "DejaVu Sans"]


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = _FONTS
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def _ink(rgba) -> str:
    r, g, b = rgba[0], rgba[1], rgba[2]
    return "white" if (0.299 * r + 0.587 * g + 0.114 * b) < 0.5 else "#212529"


def _heat(ax, colour: np.ndarray, text: np.ndarray, rows: list[str], cols: list[str],
          fmt, cmap, norm, small: bool) -> None:
    im = ax.imshow(colour, aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[:26] for r in rows], fontsize=8 if small else 9)
    for i in range(colour.shape[0]):
        for j in range(colour.shape[1]):
            ax.text(j, i, fmt(text[i, j]), ha="center", va="center",
                    fontsize=6.5 if small else 7.5,
                    color=_ink(im.cmap(im.norm(colour[i, j]))))
    ax.set_xticks(np.arange(-.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.7)
    ax.tick_params(which="minor", length=0)


def share_heatmap(matrix: pd.DataFrame, man: SnapshotManifest, path: Path,
                  title: str) -> Path | None:
    """Within-snapshot share of every class. The plainest reading of the corpus."""
    plt = _plt()
    from matplotlib.colors import Normalize
    cols = [man.display(s) for s in man.snapshots
            if f"{man.display(s)}_占比%" in matrix.columns]
    if not cols or not len(matrix):
        return None
    m = matrix.head(60)
    M = m[[f"{c}_占比%" for c in cols]].to_numpy(dtype=float)
    small = len(m) > 28
    fig, ax = plt.subplots(figsize=(1.65 * len(cols) + 4.6, max(3.0, 0.34 * len(m) + 1.8)))
    _heat(ax, M, M, m["类目"].astype(str).tolist(), cols,
          lambda v: f"{v:.1f}" if v >= 0.05 else ("0" if v == 0 else "·"),
          "YlGnBu", Normalize(0, min(60.0, float(np.nanmax(M)) or 1.0)), small)
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def balance_heatmap(matrix: pd.DataFrame, man: SnapshotManifest, path: Path,
                    title: str) -> Path | None:
    """Share divided by the unweighted mean across snapshots — size-blind."""
    plt = _plt()
    from matplotlib.colors import TwoSlopeNorm
    cols = [man.display(s) for s in man.snapshots
            if f"{man.display(s)}_均衡指数" in matrix.columns]
    if not cols or not len(matrix):
        return None
    m = matrix.head(60)
    raw = m[[f"{c}_均衡指数" for c in cols]].to_numpy(dtype=float)
    colour = np.log2(np.clip(np.nan_to_num(raw, nan=1.0), 1 / 64, 64))
    small = len(m) > 28
    fig, ax = plt.subplots(figsize=(1.65 * len(cols) + 4.6, max(3.0, 0.34 * len(m) + 1.8)))
    _heat(ax, colour, raw, m["类目"].astype(str).tolist(), cols,
          lambda v: ("—" if not np.isfinite(v) else "0" if v == 0
                     else f"{v:.2f}" if v < 10 else f"{v:.0f}"),
          "RdBu_r", TwoSlopeNorm(vmin=-3, vcenter=0, vmax=3), small)
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def distance_figure(tvd: pd.DataFrame, path: Path) -> Path | None:
    """TVD per level, with its interval AND the same-source noise ceiling.

    The red segment is the ceiling. A bar that does not clear it is a pair whose
    difference this data cannot distinguish from resampling one snapshot.
    """
    plt = _plt()
    if not len(tvd):
        return None
    levels = list(dict.fromkeys(tvd["层级"]))
    n_pairs = int((tvd["层级"] == levels[0]).sum())
    width = max(2.8 * len(levels), 0.95 * n_pairs * len(levels) + 2.4)
    fig, axes = plt.subplots(1, len(levels), figsize=(min(26.0, width), 5.0),
                             sharey=True, squeeze=False)
    top = max(float(tvd["TVD高"].max()), float(tvd["同源噪声上界"].max())) * 1.25
    for ax, lv in zip(axes[0], levels):
        t = tvd[tvd["层级"] == lv].reset_index(drop=True)
        x = np.arange(len(t))
        err = np.vstack([(t["TVD"] - t["TVD低"]).clip(lower=0),
                         (t["TVD高"] - t["TVD"]).clip(lower=0)])
        ax.bar(x, t["TVD"], color="#4c6ef5", width=min(.62, 1.6 / max(len(t), 1)),
               yerr=err, ecolor="#343a40", capsize=3)
        for i, v in enumerate(t["同源噪声上界"]):
            if np.isfinite(v):
                ax.plot([i - .36, i + .36], [v, v], color="#e03131", lw=2)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{a}→{b}" for a, b in zip(t["a"], t["b"])],
                           rotation=45, ha="right", fontsize=8)
        ax.set_title(lv, fontsize=10)
        ax.set_ylim(0, max(top, 0.05))
        ax.set_xlim(-0.7, len(t) - 0.3)
        ax.grid(axis="y", alpha=.3)
    axes[0][0].set_ylabel("TVD（越大越不同）")
    fig.suptitle("快照两两距离：蓝条为 TVD 及其自助法区间，红线为同源噪声上界", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def surface_scatter(surf: pd.DataFrame, path: Path, title: str) -> Path | None:
    """One point per class, one axis per side. Off-diagonal is the finding."""
    plt = _plt()
    if not len(surf):
        return None
    cols = [c for c in surf.columns if c.endswith("_占比%")]
    if len(cols) != 2:
        return None
    xa, ya = cols
    x = surf[xa].astype(float).clip(lower=0.008)
    y = surf[ya].astype(float).clip(lower=0.008)
    lim = max(1.2 * float(max(x.max(), y.max())), 1.0)
    sig = surf["显著"].astype(str).isin({"True", "是", "true"}) if "显著" in surf.columns \
        else pd.Series(False, index=surf.index)
    fig, ax = plt.subplots(figsize=(7.4, 6.6))
    ax.plot([0.008, lim], [0.008, lim], "--", color="#adb5bd", lw=1)
    ax.scatter(x[~sig], y[~sig], s=18, color="#ced4da", edgecolor="#868e96", lw=.4)
    up = sig & (surf.get("差_pp", pd.Series(0, index=surf.index)).astype(float) > 0)
    ax.scatter(x[sig & up], y[sig & up], s=34, color="#e03131", edgecolor="white", lw=.6)
    ax.scatter(x[sig & ~up], y[sig & ~up], s=34, color="#1971c2", edgecolor="white", lw=.6)
    if "差_pp" in surf.columns:
        for i in surf["差_pp"].astype(float).abs().sort_values(ascending=False).head(6).index:
            ax.annotate(str(surf.at[i, "类目"])[:13], (x[i], y[i]), fontsize=7.5,
                        textcoords="offset points", xytext=(6, 6))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.008, lim)
    ax.set_ylim(0.008, lim)
    ticks = [t for t in (0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100) if t <= lim]
    for setter, labeller in ((ax.set_xticks, ax.set_xticklabels),
                             (ax.set_yticks, ax.set_yticklabels)):
        setter(ticks)
        # matplotlib's LogFormatter uses U+2212, which the CJK fonts render as
        # tofu; the labels are written explicitly instead.
        labeller([f"{t:g}" for t in ticks], fontsize=8)
    ax.set_xlabel(xa.replace("_占比%", "") + " 内占比 %")
    ax.set_ylabel(ya.replace("_占比%", "") + " 内占比 %")
    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def build_figures(frames: dict[str, pd.DataFrame], man: SnapshotManifest,
                  img_dir: Path) -> list[Path]:
    img_dir.mkdir(parents=True, exist_ok=True)
    made: list[Path] = []
    for level, label in (("td_l1", "L1意图"), ("bu_leaf", "叶")):
        m = frames.get(f"matrix_{level}")
        if m is None or not len(m):
            continue
        for fn, stem, title in (
                (share_heatmap, f"{label}_占比热图", f"{label}：每个类在每个快照里的快照内占比 %"),
                (balance_heatmap, f"{label}_均衡指数热图",
                 f"{label}：均衡指数（快照内占比 ÷ 各快照占比的未加权均值）")):
            p = fn(m, man, img_dir / f"{stem}.png", title)
            if p:
                made.append(p)
    p = distance_figure(frames.get("pairwise_tvd", pd.DataFrame()), img_dir / "快照两两距离.png")
    if p:
        made.append(p)
    for level, label in (("td_l1", "L1意图"), ("bu_leaf", "叶")):
        s = frames.get(f"surface_{level}")
        if s is None or not len(s):
            continue
        p = surface_scatter(s, img_dir / f"{label}_界面散点.png", f"{label}：两侧各自的类内占比")
        if p:
            made.append(p)
    return made
