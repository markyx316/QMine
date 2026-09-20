"""Drive the per-class analysis over every level and write the tables.

The report renders from these files and from nothing else. That is the property
that makes the prose checkable: `verify` rebuilds the set of numbers that appear
anywhere in these tables, and any figure in an authored paragraph that is not in
that set is listed as unmatched. A renderer that computed its own numbers would
break the check silently, so it does not.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from . import classes as C
from .guards import QuoteGuard
from .manifest import SnapshotManifest
from .stats import cramers_v

#: Levels that get worked examples. The sub-intent level is geometric and gets
#: representative strings inside its own matrix instead; the family level is
#: covered by its leaves.
EXAMPLE_LEVELS = {"td_l1", "bu_leaf"}

SHEET = {"td_l1": "L1意图", "td_l2": "L2子意图",
         "bu_family_final": "家族", "bu_leaf": "叶"}


def _clean(t: pd.DataFrame) -> pd.DataFrame:
    """Excel and CSV both choke on control characters that queries really contain."""
    out = t.copy()
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].map(
                lambda v: v.translate({i: " " for i in list(range(9)) + [11, 12]
                                       + list(range(14, 32))}) if isinstance(v, str) else v)
    return out


def build_tables(d: pd.DataFrame, man: SnapshotManifest, names: dict[str, dict],
                 guard: QuoteGuard, out_dir: Path, *,
                 n_boot: int | None = None, n_null: int | None = None,
                 min_conditional_n: int = 30,
                 tag: str = "") -> dict[str, Any]:
    """Write every table for every level; return the summary that describes them."""
    out_dir.mkdir(parents=True, exist_ok=True)
    quotable = guard.quotable(d)
    frames: dict[str, pd.DataFrame] = {}
    summary: dict[str, Any] = {
        "run_id": tag,
        "n_rows": int(len(d)),
        "snapshots": {man.display(s): man.sizes[s] for s in man.snapshots},
        "manifest": man.to_dict(),
        "quote_guard": guard.counts(d),
        "levels": {},
    }

    present = [(lv, zh) for lv, zh in C.LEVELS if lv in d.columns]
    if not present:
        raise ValueError("the delivered labels carry none of the four label levels")

    cov_parts, tvd_parts, topn_parts, mem_parts, power_rows = [], [], [], [], []
    for level, zh in present:
        dd = C.class_frame(d, level, names)
        m = C.enrich(C.matrix(dd, man), dd, level, names, quotable)
        frames[f"matrix_{level}"] = m
        ab = C.absence(dd, man)
        frames[f"absence_{level}"] = ab
        nc = C.pairwise_change(dd, man)
        frames[f"newcombe_{level}"] = nc
        sig = C.signature(dd, man)
        frames[f"signature_{level}"] = sig
        conf = C.confidence(dd, man)
        if len(conf):
            frames[f"confidence_{level}"] = conf
        surf = C.surface_split(dd, man)
        if len(surf):
            frames[f"surface_{level}"] = surf
        if level in EXAMPLE_LEVELS:
            frames[f"examples_{level}"] = C.examples(dd, man, quotable)

        cov = C.coverage(dd, man, zh)
        tp, mem = C.topn(dd, man, zh)
        merged = cov.merge(
            tp[["快照", "前10行占比%", "前10流量占比%", "前10流量摆动pp",
                "前10流量占比%低", "前10流量占比%高", "流量有效n"]],
            on="快照", how="left").rename(columns={"前10行占比%": "前10占比%"})
        # ROUNDED FOR DISPLAY HERE, NOT UPSTREAM. `topn_coverage.csv` keeps full
        # precision on purpose — the report formats the top-ten rows from it —
        # but this table is read next to columns already at two decimals, and a
        # `92.6918` beside a `58.52` reads as a different kind of measurement.
        for _c in ("前10占比%", "前10流量占比%", "前10流量摆动pp",
                   "前10流量占比%低", "前10流量占比%高"):
            merged[_c] = merged[_c].round(2)
        merged["流量有效n"] = merged["流量有效n"].round(1)
        cov_parts.append(merged)
        topn_parts.append(tp)
        mem_parts.append(mem)
        t = C.pairwise_distance(dd, man, tag=f"{tag}|{level}",
                                n_boot=n_boot, n_null=n_null)
        t.insert(0, "层级", zh)
        tvd_parts.append(t)
        pw = C.exclusive_power(dd, man, zh)
        if pw:
            power_rows.append(pw)

        ct = pd.crosstab(dd["_key"], dd[C.SNAP])
        summary["levels"][zh] = {
            "level": level,
            "n_classes": int(dd["_key"].nunique()),
            "全快照都出现": int((m["出现快照数"] == len(man.snapshots)).sum()),
            "独占某快照": int((m["独占快照"].astype(str) != "").sum()),
            "真缺席条目": int((ab["判定"] == "真缺席").sum()) if len(ab) else 0,
            "缺席条目": int(len(ab)),
            "特征类条目": int((sig["方向"] == "显著高于其余全部").sum()) if len(sig) else 0,
            "反特征类条目": int((sig["方向"] == "显著低于其余全部").sum()) if len(sig) else 0,
            "显著变动对": int(nc["显著"].sum()) if len(nc) else 0,
            "比较对数": int(len(nc)),
            "CramersV_快照×类": round(cramers_v(ct.values), 4),
        }
        if len(surf):
            groups = list(man.surface_groups())
            summary["levels"][zh]["界面差异显著的类"] = int(surf["显著"].sum())
            for g in groups:
                summary["levels"][zh][f"仅{g}"] = int((surf["归属"] == f"仅{g}").sum())

    frames["coverage"] = pd.concat(cov_parts, ignore_index=True)
    frames["topn_coverage"] = pd.concat(topn_parts, ignore_index=True)
    frames["topn_members"] = pd.concat(mem_parts, ignore_index=True)
    frames["pairwise_tvd"] = pd.concat(tvd_parts, ignore_index=True)
    if power_rows:
        frames["exclusive_power"] = pd.DataFrame(power_rows)

    # Cross-route conditional mixes: "same share, different content".
    levels = {lv for lv, _ in present}
    if {"td_l1", "bu_leaf"} <= levels:
        dl1 = C.class_frame(d, "td_l1", names)
        inner = "bu_leaf_name" if "bu_leaf_name" in d.columns else "bu_leaf"
        frames["intent_leafmix"] = C.conditional_mix(
            dl1, "_key", inner, man, min_n=min_conditional_n,
            tag=f"{tag}|intent", n_null=n_null or 300)
        dlf = C.class_frame(d, "bu_leaf", names)
        inner2 = "td_l1_name" if "td_l1_name" in d.columns else "td_l1"
        frames["leaf_intentmix"] = C.conditional_mix(
            dlf, "_key", inner2, man, min_n=min_conditional_n,
            tag=f"{tag}|leaf", n_null=n_null or 300)

    for name, t in frames.items():
        if t is None or not len(t):
            continue
        _clean(t).to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8-sig")

    summary["tables_written"] = sorted(n for n, t in frames.items() if t is not None and len(t))
    summary["tables_empty"] = sorted(n for n, t in frames.items() if t is None or not len(t))
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return {"summary": summary, "frames": frames}
