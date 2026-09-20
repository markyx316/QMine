# -*- coding: utf-8 -*-
"""Domain-tuned taxonomy vs the generic 13-class frame, on the SAME rows.

The earlier study labelled every 2026-search and assistant string with one 13-class frame
(U01–U13, `tools/unified_intent_frame.py`). This run mined a taxonomy per domain from the data.
Both label the same strings, so the two frames can be compared directly:

  * AMI / NMI — how much of one partition the other already determines.
  * For each mined class, the U class it draws from and how pure it is.
  * SPLITS: a U class that the mined taxonomy divides into two or more classes, each holding a
    real share of it. That is the concrete form of "domain-tuned": the resolution the generic
    frame could not express.
  * MERGES: mined classes that pool several U classes — the opposite, and worth seeing too.

Coverage note: 2025 search and the voice export were never labelled with the U frame, so this
comparison covers 2026search + assistant_top + assistant_random only.

    python analysis/pooled5/p5_vs_unified.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pooled5_common import DOMAINS, WORK, cohort, cross_dir, load

ROOT = Path(__file__).resolve().parents[2]
SVA = ROOT / "analysis/sva2026/work"
UNAME = {"U01": "导航直达", "U02": "裸实体", "U03": "事实与数值", "U04": "解释与介绍", "U05": "核实与动态",
         "U06": "办事与操作", "U07": "个案判断与建议", "U08": "清单与推荐", "U09": "获取现成内容",
         "U10": "生成与编辑", "U11": "会话与系统指令", "U12": "违规或灰色内容", "U13": "无法判定"}
COVERED = ["2026search", "assistant_top", "assistant_random"]


def umap() -> pd.Series:
    a = pd.read_parquet(SVA / "label_full/unique_labels.parquet")[["query", "u_ds", "p_ds"]]
    b = pd.read_parquet(SVA / "label_search10k/labels.parquet")[["query", "u_ds", "p_ds"]]
    both = pd.concat([a, b], ignore_index=True).dropna(subset=["u_ds"])
    both = both.drop_duplicates(subset=["query"], keep="first")
    return both.set_index(both["query"].astype(str))


def main(doms=None) -> None:
    U = umap()
    out = cross_dir()
    out.mkdir(parents=True, exist_ok=True)
    rows, splits, merges = [], [], []
    for domain in (doms or list(cohort())):
        d = load(domain)
        d = d[d.source.isin(COVERED)].copy()
        d["u"] = d["query"].astype(str).map(U["u_ds"])
        d["p"] = d["query"].astype(str).map(U["p_ds"])
        cov = float(d.u.notna().mean())
        m = d.dropna(subset=["u"])
        ct = pd.crosstab(m.td_l1_name, m.u)
        ct.to_csv(WORK / domain / "vs_unified_crosstab.csv")
        # Per source as well: "what did the generic frame call the rows this taxonomy cannot
        # place?" is a different question on the assistant head than on search.
        per = pd.crosstab([m.source, m.td_l1_name], m.u)
        per.to_csv(WORK / domain / "vs_unified_crosstab_by_source.csv")
        rows.append({"domain": domain, "n_rows": len(d), "u_coverage": round(cov, 4),
                     "n_mined_classes": int(m.td_l1_name.nunique()), "n_u_classes": int(m.u.nunique()),
                     "AMI": round(adjusted_mutual_info_score(m.u, m.td_l1_name), 4),
                     "NMI": round(normalized_mutual_info_score(m.u, m.td_l1_name), 4)})
        for u, col in ct.items():
            tot = col.sum()
            if tot < 50:
                continue
            big = col[col / tot >= 0.20].sort_values(ascending=False)
            if len(big) >= 2:
                splits.append({"domain": domain, "u": u, "u_name": UNAME.get(u, u), "u_rows": int(tot),
                               "mined_classes": " | ".join(f"{k} {v/tot*100:.0f}%" for k, v in big.items())})
        for cls, row in ct.iterrows():
            tot = row.sum()
            if tot < 50:
                continue
            big = row[row / tot >= 0.20].sort_values(ascending=False)
            rec = {"domain": domain, "mined_class": cls, "rows": int(tot),
                   "dominant_u": f"{row.idxmax()} {UNAME.get(row.idxmax(), '')}",
                   "purity": round(float(row.max() / tot), 3),
                   "u_mix": " | ".join(f"{UNAME.get(k, k)} {v/tot*100:.0f}%" for k, v in big.items())}
            merges.append(rec)
        # do the two frames tell the same story about the surfaces?
        sh = []
        for src, g in m.groupby("source"):
            for frame, col in (("统一13类", "u"), ("本次挖掘", "td_l1_name")):
                vc = g[col].value_counts(normalize=True)
                for lab, v in vc.items():
                    sh.append({"domain": domain, "source": src, "frame": frame,
                               "label": UNAME.get(lab, lab) if frame == "统一13类" else lab,
                               "share": round(float(v), 4)})
        pd.DataFrame(sh).to_csv(WORK / domain / "vs_unified_shares.csv", index=False)
        print(f"{domain}: U coverage {cov*100:.1f}% · AMI {rows[-1]['AMI']} · "
              f"mined {rows[-1]['n_mined_classes']} vs U {rows[-1]['n_u_classes']}")
    if doms and len(doms) < len(DOMAINS):
        # A PARTIAL RUN MUST NOT OVERWRITE THE CROSS-DOMAIN TABLES. Running this with one domain
        # to inspect it wiped four domains out of vs_unified_summary/splits/classmix, and the
        # report shipped a T10 containing only that domain while the prose discussed the others.
        print(f"partial run ({', '.join(doms)}): per-domain files written, cross-domain tables left alone")
        return
    pd.DataFrame(rows).to_csv(out / "vs_unified_summary.csv", index=False)
    pd.DataFrame(splits).to_csv(out / "vs_unified_splits.csv", index=False)
    pd.DataFrame(merges).to_csv(out / "vs_unified_classmix.csv", index=False)
    print("\n拆分（统一框架里的一类，被本次体系分成多类）")
    print(pd.DataFrame(splits).to_string(index=False) if splits else "  （无）")


if __name__ == "__main__":
    main(sys.argv[1:] or None)
