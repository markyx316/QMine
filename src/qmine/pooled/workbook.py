"""The two workbooks: the comparison tables, and every labelled row with its snapshot.

The run's own deliverable labels each row but cannot say which snapshot it came
from — the snapshot never entered the run as a feature, and declaring it would
have asked the clustering to separate the very thing the study compares. These
files put it back.

A NOTE SHEET COMES FIRST IN BOTH. A workbook of 30 numeric sheets handed over
without one is read by guessing, and the two things most often guessed wrong
here — that shares are within-snapshot, and that two separate runs' labels are
comparable — are both wrong in ways that invalidate whatever is concluded.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from .manifest import NORM_TOTAL, SnapshotManifest

ILLEGAL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

#: Excel sheet names: 31 characters, and none of : \ / ? * [ ]
_BAD_SHEET = re.compile(r"[:\\/?*\[\]]")


def _sheet_name(name: str, used: set[str]) -> str:
    s = _BAD_SHEET.sub("_", name)[:31] or "sheet"
    base, i = s, 2
    while s in used:
        s = f"{base[:28]}_{i}"
        i += 1
    used.add(s)
    return s


def _scrub(t: pd.DataFrame) -> pd.DataFrame:
    out = t.copy()
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].map(lambda v: ILLEGAL.sub(" ", v) if isinstance(v, str) else v)
    return out


def _notes(man: SnapshotManifest, summary: dict[str, Any], run_id: str,
           generation: str) -> pd.DataFrame:
    rows = [
        ["本文件是什么",
         f"运行 `{run_id}/{generation}` 把 {len(man.snapshots)} 份数据合并成一份语料、"
         "一次跑完后的逐类对照。一次运行 = 一套意图体系 = 一套聚类树，所以快照之间可以直接比较。"],
        ["为什么不能比两次运行",
         "分开跑的两次运行不能比：实测同样 10,000 行的两次金融运行，20 个类与 19 个类"
         "共享 0 个类目编码。要比较，必须把它们放进同一次运行。"],
        ["对比轴", ("分层：快照是同一时期的不同抽样方式，之间没有先后，不要读成「变化」。"
                    if man.axis == "stratum" else "时间：快照按先后排列，可以读成变化。")],
        ["怎么读占比",
         "一律看快照内占比。各快照行数不同（"
         + "、".join(f"{man.display(s)} {man.sizes[s]:,}" for s in man.snapshots)
         + "），跨快照的原始条数没有意义。"],
        ["权重 pv_norm",
         f"每个快照各自归一到 {int(NORM_TOTAL):,}。"
         "不同快照的权重可能来自不同的计量口径，混在一起的原始权重会让所有加权指标只讲最大的那个快照。"
         + (f" 以下快照没有可用权重，按均匀权重处理，它们的流量占比恒等于行占比："
            f"{'、'.join(man.display(s) for s in man.uniform_weight)}。"
            if man.uniform_weight else "")],
        ["0 条怎么读",
         "0 条从不证明不存在。每个缺席都带期望条数、P(0) 与该快照的单侧 97.5% 上界，"
         "结论看「判定」列。"],
        ["引用护栏",
         "例子是真实的行。七层护栏决定一行能不能被引用；护栏只影响引用，不影响任何统计。"
         f"本次拦下 {summary.get('quote_guard', {}).get('任一层拦下', 0):,} 行，"
         f"可引 {summary.get('quote_guard', {}).get('可引行数', 0):,} 行。"],
        ["标签列",
         "td_* 为自上而下意图体系（L1/L2、置信度、是否模糊）；bu_* 为自下而上聚类"
         "（叶、家族、边界 margin）。两套都由本次运行产生，且是互相独立的两条路线。"],
    ]
    return pd.DataFrame(rows, columns=["项", "说明"])


def write_tables_workbook(frames: dict[str, pd.DataFrame], summary: dict[str, Any],
                          man: SnapshotManifest, path: Path, *, run_id: str,
                          generation: str) -> Path:
    """Every comparison table, one per sheet, note sheet first."""
    order = ["coverage", "pairwise_tvd", "topn_coverage", "topn_members",
             "exclusive_power", "intent_leafmix", "leaf_intentmix"]
    label = {"coverage": "快照画像", "pairwise_tvd": "两两距离", "topn_coverage": "前十合计",
             "topn_members": "前十成员", "exclusive_power": "独有性检出力",
             "intent_leafmix": "意图内的叶构成", "leaf_intentmix": "叶内的意图构成"}
    lvl = {"td_l1": "L1意图", "td_l2": "L2子意图", "bu_family_final": "家族", "bu_leaf": "叶"}
    kind = {"matrix": "矩阵", "absence": "缺席", "newcombe": "两两差异",
            "signature": "特征类", "confidence": "可靠度", "surface": "分组差异",
            "examples": "例子"}
    used: set[str] = set()
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        _notes(man, summary, run_id, generation).to_excel(
            xw, sheet_name=_sheet_name("说明", used), index=False)
        pd.DataFrame([{"快照": man.display(s), "内部标识": s, "行数": man.sizes[s],
                       "分组": man.surface_of(s) or "（未声明）"}
                      for s in man.snapshots]).to_excel(
            xw, sheet_name=_sheet_name("快照总表", used), index=False)
        pd.DataFrame([{"层级": zh, **{k: v for k, v in d.items() if k != "level"}}
                      for zh, d in summary["levels"].items()]).to_excel(
            xw, sheet_name=_sheet_name("层级总览", used), index=False)
        for name in order:
            t = frames.get(name)
            if t is not None and len(t):
                _scrub(t).to_excel(xw, sheet_name=_sheet_name(label[name], used), index=False)
        for k in ("matrix", "absence", "surface", "signature", "newcombe",
                  "confidence", "examples"):
            for level, zhl in lvl.items():
                t = frames.get(f"{k}_{level}")
                if t is not None and len(t):
                    _scrub(t).to_excel(
                        xw, sheet_name=_sheet_name(f"{zhl}_{kind[k]}", used), index=False)
        pd.DataFrame([{"项": k, "值": v}
                      for k, v in summary.get("quote_guard", {}).items()]).to_excel(
            xw, sheet_name=_sheet_name("引用护栏", used), index=False)
    return path


#: Columns of the row-level file, in reading order. Missing ones are skipped.
ROW_ORDER = ["query", "快照", "snapshot", "分组", "pv_raw", "pv_norm",
             "td_l1", "td_l1_name", "td_l2", "td_confidence", "td_margin",
             "td_ambiguous", "td_decided_by", "bu_leaf", "bu_leaf_name",
             "bu_family_final", "bu_margin", "bu_ambiguous"]


def write_rows_workbook(d: pd.DataFrame, man: SnapshotManifest, summary: dict[str, Any],
                        path: Path, *, run_id: str, generation: str,
                        removed: pd.DataFrame | None = None) -> Path:
    """Every mined row with its snapshot and its labels; plus what cleaning removed."""
    full = d.copy()
    full["快照"] = full["snapshot"].map(man.display)
    full["分组"] = full["snapshot"].map(lambda s: man.surface_of(s) or "")
    cols = [c for c in ROW_ORDER if c in full.columns]
    full = full[cols].sort_values(["快照", "pv_norm"], ascending=[True, False])
    path.parent.mkdir(parents=True, exist_ok=True)
    used: set[str] = set()
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        _notes(man, summary, run_id, generation).to_excel(
            xw, sheet_name=_sheet_name("说明", used), index=False)
        _scrub(full).to_excel(xw, sheet_name=_sheet_name("全量标注(含快照)", used), index=False)
        pd.DataFrame([{"快照": man.display(s), "内部标识": s, "行数": man.sizes[s],
                       "占比%": round(100 * man.sizes[s] / max(len(d), 1), 2)}
                      for s in man.snapshots]).to_excel(
            xw, sheet_name=_sheet_name("快照分布", used), index=False)
        if removed is not None and len(removed):
            # Removed rows SHIP. A cleaning step nobody can see is a cleaning
            # step nobody can check, and the product layer these corpora carry
            # is exactly what a reader wants to inspect.
            _scrub(removed).to_excel(
                xw, sheet_name=_sheet_name("被清洗剔除", used), index=False)
    return path
