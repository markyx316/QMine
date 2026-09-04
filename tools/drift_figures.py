#!/usr/bin/env python
"""Figures for the cross-domain drift analysis.

Three things a table cannot show as well:

1. `fig_drift_regimes` — every class's share change against its ABSOLUTE traffic
   change. This is the figure that stops the report being wrong. Within-snapshot
   shares are the right way to compare composition, but when a vertical's head
   lost 47% of its traffic, a class can gain share while losing a third of its
   audience. The plot separates "grew" from "shrank more slowly than everything
   else", and the five verticals fall into two visibly different regimes.
2. `fig_drift_sampling` — the PV floor against the top-10k total. They move in
   lockstep, which is what makes the fixed-N cut the central caveat of the whole
   analysis rather than a footnote.
3. `fig_drift_controls` — how much of each vertical's measured drift survives
   removing its dated calendar event.

Light and dark variants, sized for a ~890px column, same as tools/readme_figures.py.

    HF_HOME=$(pwd)/.hf .venv/bin/python tools/drift_figures.py
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# CJK labels need a font that has the glyphs, or every character renders as a box.
# Same stack the notebook generator uses (`report/zh_notebook.py`), and
# `axes.unicode_minus` must be off or the minus sign renders as a box too.
matplotlib.rcParams.update({
    "font.family": ["Arial Unicode MS", "Heiti TC", "PingFang SC", "Songti SC", "sans-serif"],
    "axes.unicode_minus": False,
})

#: Every label that differs between the English and Chinese renderings. Keyed by
#: the English string so the figure code reads in one language and the lookup is
#: a single call.
ZH = {
    "A class can gain share and still lose most of its audience":
        "一个类目可以占比上升，却失去大部分受众",
    "change in the class's share of its vertical's traffic  (pp)":
        "该类目在本垂类流量中的占比变化（百分点）",
    "change in the class's ABSOLUTE traffic  (%, clipped at +300)":
        "该类目【绝对流量】的变化（%，上限截断于 +300）",
    "dotted line = that vertical's own total": "虚线 = 该垂类自身的总量变化",
    'The "traffic collapse" is a fixed-N cut moving, not a measured demand fall':
        "所谓「流量暴跌」是固定条数截断在移动，而非测到的需求下滑",
    "change in the PV of the 10,000th query  (the cut's depth, %)":
        "第 10,000 名 query 的 PV 变化（截断深度，%）",
    "change in the top-10k total PV  (%)": "前 1 万条 query 总 PV 的变化（%）",
    "floor and total\nmove together": "门槛与总量\n同步移动",
    "The same corpus does not give the same tree twice": "同一份语料两次运行不会给出同一棵树",
    "Governance sets the family count, not the K locator": "家族数由治理决定，而非 K 定位",
    "What survives the control": "控制变量之后还剩下什么",
    "total variation of the traffic-share distribution": "流量占比分布的总变差",
    "as measured": "原始测量值", "after the control": "施加控制之后",
    "2026 World Cup on CCTV-5": "2026 世界杯，CCTV-5 直播",
    "two 2025 political-news figures": "2025 年两位时政人物",
    "(no single event found)": "（未找到单一事件）",
    "(depth control, not an event)": "（深度对齐，非事件）",
    "FOOT_regimes":
        "位于本垂类虚线之上者，跌幅小于该垂类整体；位于【零线】之上者才是真正增长。\n"
        "医疗与教育几乎没有任何类目在零线之上 —— 所谓「上升」不过是跌得比别人慢。",
    "FOOT_sampling":
        "每个快照都是当日 PV 前 1 万条 query。当第 10,000 名变小，截断就会伸向更长的尾部，\n"
        "其总量随之下降 —— 无论真实需求是否下降。只有全量日志能区分二者。金融在两个轴上都是例外。",
    "FOOT_controls":
        "影视与人物：剔除当日事件后漂移【下降】—— 其中三分之一与八分之一原本来自日历。\n"
        "医疗与教育：把两次截断对齐到同一深度后漂移【上升】—— 原始数值低估了 40-50%。",
    "finance": "金融", "film/TV": "影视", "medical": "医疗",
    "education": "教育", "people": "人物",
}
LANG = "en"


def L(text):
    """Label lookup. Falls back to the English string, so an untranslated label is
    visible in the figure rather than silently blank."""
    return ZH.get(text, text) if LANG == "zh" else text

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from qmine.ops import drift  # noqa: E402

THEMES = {
    "light": dict(fg="#1f2328", muted="#656d76", grid="#d0d7de", bg="#ffffff", zero="#8c959f"),
    "dark": dict(fg="#e6edf3", muted="#8b949e", grid="#30363d", bg="#0d1117", zero="#6e7681"),
}
DOMAIN = {"finance": ("金融", "fin-pool", "#0969da"), "film/TV": ("影视", "film-pool", "#bf3989"),
          "medical": ("医疗", "med-pool", "#1a7f37"), "education": ("教育", "edu-pool", "#9a6700"),
          "people": ("人物", "ppl-pool", "#8250df")}
DOMAIN_DARK = {"finance": "#58a6ff", "film/TV": "#f778ba", "medical": "#3fb950",
               "education": "#d29922", "people": "#a371f7"}


def _style(ax, t):
    ax.set_facecolor(t["bg"])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=t["muted"], labelsize=9, length=0)
    ax.xaxis.label.set_color(t["muted"])
    ax.yaxis.label.set_color(t["muted"])
    ax.title.set_color(t["fg"])
    ax.grid(color=t["grid"], lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def _frame(en):
    cn, run, _ = DOMAIN[en]
    src = pd.read_csv(f"data/raw/{cn}query-pooled.csv")
    lp = f"runs/{run}/gen01/labels_full_reconciled.csv"
    lab = pd.read_csv(lp if os.path.exists(lp) else f"runs/{run}/gen01/labels_full.csv",
                      encoding="utf-8-sig")
    return pd.DataFrame({"snapshot": src["_snapshot"].astype(str),
                         "query": src["original_query"].astype(str),
                         "weight": src["wise_pv"].astype(float),
                         "leaf": lab["bu_leaf_name"].values}).dropna(subset=["leaf"])


def fig_regimes(t, path, dark):
    fig, ax = plt.subplots(figsize=(8.2, 6.0), dpi=110)
    fig.patch.set_facecolor(t["bg"])
    for en in DOMAIN:
        d = _frame(en)
        pv = d.pivot_table(index="leaf", columns="snapshot", values="weight", aggfunc="sum").fillna(0)
        base = pv["20260701"].sum() / pv["20250701"].sum() - 1
        dr = drift.label_drift(d, "leaf", "snapshot", "weight")
        col = DOMAIN_DARK[en] if dark else DOMAIN[en][2]
        xs, ys = [], []
        for r in dr["stable"]:
            lb = r["label"]
            if lb not in pv.index or pv.loc[lb, "20250701"] <= 0:
                continue
            ab = pv.loc[lb, "20260701"] / pv.loc[lb, "20250701"] - 1
            if abs(r["weight_share_delta_pp"]) < 0.25:
                continue                    # tiny movers are noise on this plot
            xs.append(r["weight_share_delta_pp"])
            ys.append(min(ab, 3.0) * 100)   # clip the +1238% outlier so the rest is readable
        ax.scatter(xs, ys, s=34, color=col, alpha=0.75, label=L(en), zorder=3)
        ax.axhline(base * 100, color=col, lw=1.0, ls=":", alpha=0.75, zorder=1)
    ax.axhline(0, color=t["zero"], lw=1.2, zorder=2)
    ax.axvline(0, color=t["zero"], lw=1.2, zorder=2)
    ax.set_xlabel(L("change in the class's share of its vertical's traffic  (pp)"))
    ax.set_ylabel(L("change in the class's ABSOLUTE traffic  (%, clipped at +300)"))
    ax.set_title(L("A class can gain share and still lose most of its audience"),
                 fontsize=11.5, pad=14, loc="left")
    _style(ax, t)
    leg = ax.legend(loc="upper left", frameon=False, fontsize=9, title=L("dotted line = that vertical's own total"))
    leg.get_title().set_color(t["muted"]); leg.get_title().set_fontsize(8)
    for x in leg.get_texts():
        x.set_color(t["muted"])
    fig.text(0.01, 0.015,
             L("FOOT_regimes"),
             color=t["muted"], fontsize=8)
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    fig.savefig(path, facecolor=t["bg"])
    plt.close(fig)


def fig_sampling(t, path, dark):
    # dx/dy are hand-placed label offsets: medical/education and film/people sit
    # close enough that the default placement overlapped each pair.
    pts = {"finance": (3.3, 34.7, 12, -4), "film/TV": (-19.8, -24.0, -12, -12),
           "medical": (-51.0, -46.5, 12, 6), "education": (-47.6, -48.1, 12, -9),
           "people": (-13.5, -21.8, 12, 6)}
    fig, ax = plt.subplots(figsize=(7.6, 5.8), dpi=110)
    fig.patch.set_facecolor(t["bg"])
    lo, hi = -58, 40
    ax.plot([lo, hi], [lo, hi], color=t["grid"], lw=1.4, ls="--", zorder=1)
    ax.text(hi - 2, hi - 9, L("floor and total\nmove together"), color=t["muted"], fontsize=8.5, ha="right")
    for en, (fl, tt, dx, dy) in pts.items():
        col = DOMAIN_DARK[en] if dark else DOMAIN[en][2]
        ax.scatter(fl, tt, s=150, color=col, zorder=3)
        ax.annotate(L(en), (fl, tt), textcoords="offset points", xytext=(dx, dy),
                    fontsize=9.5, color=t["fg"], ha="right" if dx < 0 else "left")
    ax.axhline(0, color=t["zero"], lw=1.1)
    ax.axvline(0, color=t["zero"], lw=1.1)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel(L("change in the PV of the 10,000th query  (the cut's depth, %)"))
    ax.set_ylabel(L("change in the top-10k total PV  (%)"))
    ax.set_title(L('The "traffic collapse" is a fixed-N cut moving, not a measured demand fall'),
                 fontsize=11, pad=14, loc="left")
    _style(ax, t)
    fig.text(0.01, 0.015,
             L("FOOT_sampling"),
             color=t["muted"], fontsize=8)
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    fig.savefig(path, facecolor=t["bg"])
    plt.close(fig)


def fig_controls(t, path, dark):
    """How much drift survives removing each vertical's dated event."""
    rows = [("film/TV", 0.203, 0.131, "2026 World Cup on CCTV-5"),
            ("people", 0.288, 0.251, "two 2025 political-news figures"),
            ("finance", 0.216, 0.215, "(no single event found)"),
            ("medical", 0.111, 0.152, "(depth control, not an event)"),
            ("education", 0.103, 0.154, "(depth control, not an event)")]
    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=110)
    fig.patch.set_facecolor(t["bg"])
    y = range(len(rows))
    for i, (en, raw, ctl, _) in enumerate(rows):
        col = DOMAIN_DARK[en] if dark else DOMAIN[en][2]
        ax.plot([raw, ctl], [i, i], color=t["grid"], lw=2.4, zorder=1)
        ax.scatter([raw], [i], s=64, color=t["muted"], zorder=3)
        ax.scatter([ctl], [i], s=64, color=col, zorder=3)
    ax.set_yticks(list(y))
    ax.set_yticklabels([f"{L(en)}  ·  {L(note)}" for en, _, _, note in rows], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(L("total variation of the traffic-share distribution"))
    ax.set_title(L("What survives the control"), fontsize=11.5, pad=14, loc="left")
    _style(ax, t)
    ax.scatter([], [], s=64, color=t["muted"], label=L("as measured"))
    ax.scatter([], [], s=64, color=DOMAIN_DARK["finance"] if dark else DOMAIN["finance"][2],
               label=L("after the control"))
    leg = ax.legend(loc="lower right", frameon=False, fontsize=9)
    for x in leg.get_texts():
        x.set_color(t["muted"])
    fig.text(0.01, 0.02,
             L("FOOT_controls"),
             color=t["muted"], fontsize=8)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, facecolor=t["bg"])
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="docs/img")
    ap.add_argument("--lang", choices=("en", "zh"), default="en",
                    help="label language; zh writes *_zh.png alongside the English ones")
    a = ap.parse_args()
    global LANG
    LANG = a.lang
    suffix_lang = "_zh" if a.lang == "zh" else ""
    os.makedirs(a.out, exist_ok=True)
    made = []
    for name, fn in (("fig_drift_regimes", fig_regimes),
                     ("fig_drift_sampling", fig_sampling),
                     ("fig_drift_controls", fig_controls)):
        for theme, t in THEMES.items():
            p = os.path.join(
                a.out, f"{name}{suffix_lang}{'' if theme == 'light' else '_dark'}.png")
            fn(t, p, theme == "dark")
            made.append(p)
    print(f"{len(made)} files written")
    for m in made:
        print("  ", m)


if __name__ == "__main__":
    main()
