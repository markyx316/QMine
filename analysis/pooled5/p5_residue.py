# -*- coding: utf-8 -*-
"""What is in the catch-all class, and is it the user's fault or the frame's?

Every mined taxonomy has one residual class. It matters most on the assistant sources, because
the frame was fitted on a corpus that is ~87% search. For each domain and source this splits the
residue by what the earlier study's generic 13-class frame called the same rows:

  * U13 无法判定 on both sides  → the row really is uninterpretable out of context;
  * U11 会话与系统指令 / U02 裸实体 → the generic frame HAD a class for it and this taxonomy
    does not — that is a frame gap, not a user failure;
  * anything else               → the domain frame could not place a row the generic one could.

    python analysis/pooled5/p5_residue.py
"""
from __future__ import annotations
import re, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pooled5_common import COHORT, SOURCES, SRC_ZH, WORK, available, cohort, cross_dir, load

if not available():
    raise SystemExit(f"cohort {COHORT!r}: no finished run yet — nothing to sweep. "
                     f"(expected labels_full.csv under runs/<id>/gen01 for {cohort()})")

UNAME = {"U01": "导航直达", "U02": "裸实体", "U03": "事实与数值", "U04": "解释与介绍", "U05": "核实与动态",
         "U06": "办事与操作", "U07": "个案判断与建议", "U08": "清单与推荐", "U09": "获取现成内容",
         "U10": "生成与编辑", "U11": "会话与系统指令", "U12": "违规或灰色内容", "U13": "无法判定"}
CATCHALL = re.compile(r"(UNCLASS|OTHER|NOISE|OUT_OF_DOMAIN|RESIDUAL|UNDETERMIN)", re.I)

# THIS ANALYSIS NEEDS THE GENERIC 13-CLASS FRAME LABELLED ON THE SAME STRINGS. It is not a
# property of the run; it comes from the earlier study (`analysis/sva2026/.../label_*`). The two
# 2026-09-13 verticals were never labelled with it — measured coverage of 2026search+assistant:
# 书籍文档 0.5% (63/11,798), 软件 0.1% (14/11,923), against 100% for all five original domains.
# Splitting a residue class against a frame that covers 0.1% of the rows produces a table whose
# every cell is noise, so this refuses rather than emitting one.
def _unified_coverage(doms: list[str]) -> dict[str, float]:
    sva = Path(__file__).resolve().parents[2] / "analysis/sva2026/work"
    have: set[str] = set()
    for f in ("label_full/unique_labels.parquet", "label_search10k/labels.parquet"):
        if (sva / f).exists():
            have |= set(pd.read_parquet(sva / f)["query"].astype(str))
    out = {}
    for d in doms:
        x = load(d)
        x = x[x.source.isin(["2026search", "assistant_top", "assistant_random"])]
        out[d] = float(x["query"].astype(str).isin(have).mean()) if len(x) else 0.0
    return out


_cov = _unified_coverage(available())
if min(_cov.values(), default=0.0) < 0.5:
    raise SystemExit(
        "p5_residue needs the generic 13-class frame on the same strings, and this batch does not "
        "have it: " + " · ".join(f"{d} {100 * c:.1f}%" for d, c in _cov.items())
        + ". Label the corpus with tools/label_unified_intent.py first, or skip this analysis and "
          "say in the report that it was SKIPPED for lack of coverage — never run it on ~0%.")

rows = []
for domain in available():
    d = load(domain)
    defs = pd.read_csv(WORK / domain / "class_definitions.csv")
    codes = [c for c in defs["code"].astype(str) if CATCHALL.search(c)]
    names = defs[defs["code"].isin(codes)]["name"].tolist()
    if not names:
        print(f"{domain}: no catch-all class matched — check class_definitions.csv")
        continue
    ct = pd.read_csv(WORK / domain / "vs_unified_crosstab_by_source.csv", index_col=[0, 1])
    for src in SOURCES:
        g = d[(d.source == src) & (d.td_l1_name.isin(names))]
        if not len(d[d.source == src]):
            continue
        row = {"domain": domain, "source": src, "catch_all": " / ".join(names),
               "rows": len(g), "share_of_source": round(len(g) / len(d[d.source == src]), 4)}
        if (src, names[0]) in ct.index:
            u = ct.loc[(src, names[0])]
            u = u[u > 0]
            tot = u.sum()
            row["u_coverage_rows"] = int(tot)
            shown = ("U13", "U11", "U02")
            for code in shown:
                row[f"{code} {UNAME[code]}"] = round(float(u.get(code, 0) / tot), 3) if tot else None
            # EVERYTHING ELSE, so the row sums to 1. The first version listed U09 and U10 as
            # columns and then subtracted them from the remainder as well, so each row summed to
            # 0.93–0.99 and the "domain frame could not place it" block read too small.
            row["其他实质类目"] = round(float(1 - sum(u.get(c, 0) for c in shown) / tot), 3) if tot else None
        rows.append(row)
t = pd.DataFrame(rows)
t.to_csv(cross_dir() / "residue_breakdown.csv", index=False)
x = t.copy()
x["source"] = x["source"].map(SRC_ZH)
print(x.drop(columns=["catch_all"]).to_string(index=False))
