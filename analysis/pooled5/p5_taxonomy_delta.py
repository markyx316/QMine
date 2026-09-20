# -*- coding: utf-8 -*-
"""What did adding the assistant rows do to the TAXONOMY ITSELF?

The five `*-pool` runs mined the SAME 20,000 search rows (2025 + 2026) without any assistant
data. The five `*-pool5` runs mined those rows again with ~2,000–3,000 assistant rows mixed in.
So the search rows can be compared label-for-label across the two frames — the one thing two
ordinary runs can never give you, because they share no codes (fin02/fin03: 0 of 35).

What this measures:
  * AMI / NMI between the old and new partition OF THE SAME SEARCH ROWS. High AMI means the
    assistant rows did not disturb how search is carved up; low means the frame moved.
  * Which NEW classes absorbed search rows, and what share of each new class is assistant rows —
    a class that is >50% assistant is a class the assistant data brought into existence.
  * The old class each new class drew from (top mappings, row counts).

    python analysis/pooled5/p5_taxonomy_delta.py [domain ...]
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, normalized_mutual_info_score
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pooled5_common import DOMAINS, WORK, available, load, work_file

ROOT = Path(__file__).resolve().parents[2]
SNAP2SRC = {"20250701": "2025search", "20260701": "2026search"}


def old_labels(domain: str) -> pd.DataFrame:
    key = DOMAINS[domain]
    lab = pd.read_csv(ROOT / f"runs/{key}-pool/gen01/labels_full.csv", encoding="utf-8-sig")
    src = pd.read_csv(ROOT / f"data/raw/{domain}query-pooled.csv")
    # p1 drops empty text cells (教育 has one), so the old run is a row shorter than its input.
    src = src[src["original_query"].notna() &
              (src["original_query"].astype("string").fillna("").str.strip() != "")].reset_index(drop=True)
    if len(lab) != len(src):
        raise SystemExit(f"{domain}: old run {len(lab)} rows vs pooled csv {len(src)}")
    bad = (lab["query"].astype(str).values != src["original_query"].astype(str).values).sum()
    if bad:
        raise SystemExit(f"{domain}: {bad} old rows misaligned")
    out = pd.DataFrame({"query": src["original_query"].astype(str),
                        "source": src["_snapshot"].astype(str).map(SNAP2SRC),
                        "old_td": lab["td_l1_name"], "old_bu": lab["bu_family_final"].astype(str)})
    return out


def reproducibility_baseline() -> dict:
    """WITHOUT THIS NUMBER THE COMPARISON MEANS NOTHING. `fin02` and `fin03` ran the SAME 10,000
    finance rows under the same config, so whatever AMI they reach is what two runs give when
    the data did not change at all. Measured: AMI(td) 0.774, AMI(bu) 0.687. Any old-vs-new AMI
    at that level is run-to-run variation, not an effect of adding the assistant rows.
    (Caveat: it is one pair, in one domain, at half the row count.)"""
    import glob
    def lab(r):
        g = sorted(glob.glob(str(ROOT / f"runs/{r}/gen*/labels_full.csv")))[-1]
        return pd.read_csv(g, encoding="utf-8-sig")
    a, b = lab("fin02"), lab("fin03")
    m = (a[["query", "td_l1_name", "bu_family_final"]]
         .merge(b[["query", "td_l1_name", "bu_family_final"]], on="query", suffixes=("_1", "_2"))
         .drop_duplicates("query"))
    return {"domain": "对照·同数据两次运行(fin02/fin03)", "matched_search_rows": len(m),
            "AMI_td": round(adjusted_mutual_info_score(m.td_l1_name_1, m.td_l1_name_2), 4),
            "AMI_bu": round(adjusted_mutual_info_score(m.bu_family_final_1.astype(str),
                                                       m.bu_family_final_2.astype(str)), 4),
            "n_classes_old": int(m.td_l1_name_1.nunique()), "n_classes_new": int(m.td_l1_name_2.nunique())}


def delta(domain: str) -> dict:
    new = load(domain)
    new_s = new[new.source.isin(["2025search", "2026search"])][
        ["query", "source", "td_l1_name", "bu_family_final"]].rename(
        columns={"td_l1_name": "new_td", "bu_family_final": "new_bu"})
    old = old_labels(domain)
    new_s["query"] = new_s["query"].astype(str)
    # A handful of strings repeat inside one year's export (金融 2, 影视 5 of 20,000). Keep the
    # first occurrence on both sides so the merge stays 1:1 rather than forming a cross product.
    dup = (int((new_s.groupby(["query", "source"]).size() > 1).sum()),
           int((old.groupby(["query", "source"]).size() > 1).sum()))
    new_s = new_s.drop_duplicates(subset=["query", "source"], keep="first")
    old = old.drop_duplicates(subset=["query", "source"], keep="first")
    m = new_s.merge(old, on=["query", "source"], how="inner", validate="1:1")
    out = WORK / domain
    out.mkdir(parents=True, exist_ok=True)
    res = {"domain": domain, "matched_search_rows": len(m), "dup_keys_new_old": dup,
           "AMI_td": round(adjusted_mutual_info_score(m.old_td, m.new_td), 4),
           "NMI_td": round(normalized_mutual_info_score(m.old_td, m.new_td), 4),
           "ARI_td": round(adjusted_rand_score(m.old_td.astype("category").cat.codes,
                                               m.new_td.astype("category").cat.codes), 4),
           "AMI_bu": round(adjusted_mutual_info_score(m.old_bu, m.new_bu), 4),
           "n_classes_old": int(m.old_td.nunique()), "n_classes_new": int(m.new_td.nunique())}
    ct = pd.crosstab(m.new_td, m.old_td)
    ct.to_csv(out / "taxonomy_delta_crosstab.csv")
    # what each new class is made of: its biggest old-class source, and its assistant share
    asst = new[~new.source.isin(["2025search", "2026search"])]
    mix = []
    for cls, g in new.groupby("td_l1_name", sort=False):
        row = ct.loc[cls] if cls in ct.index else None
        mix.append({"new_class": cls, "n_total": len(g),
                    "n_assistant": int((~g.source.isin(["2025search", "2026search"])).sum()),
                    "assistant_share": round(float((~g.source.isin(["2025search", "2026search"])).mean()), 4),
                    "main_old_class": (row.idxmax() if row is not None and row.sum() else ""),
                    "main_old_share": (round(float(row.max() / row.sum()), 4) if row is not None and row.sum() else float("nan")),
                    "n_old_classes_feeding": (int((row > 0).sum()) if row is not None else 0)})
    mixdf = pd.DataFrame(mix).sort_values("assistant_share", ascending=False)
    mixdf.to_csv(out / "taxonomy_delta_newclass_mix.csv", index=False)
    res["classes_majority_assistant"] = int((mixdf.assistant_share > 0.5).sum())
    res["assistant_rows"] = int(len(asst))
    print(f"{domain}: matched {len(m):,} search rows | old {res['n_classes_old']} → new {res['n_classes_new']} classes "
          f"| AMI(td) {res['AMI_td']} | AMI(bu) {res['AMI_bu']} | new classes >50% assistant: {res['classes_majority_assistant']}")
    return res


if __name__ == "__main__":
    rs = [delta(x) for x in (sys.argv[1:] or available())]
    base = reproducibility_baseline()
    print(f"对照（同一份数据跑两次）: AMI(td) {base['AMI_td']} · AMI(bu) {base['AMI_bu']} "
          f"— 上面的数字必须跟这个比，不能跟 1 比")
    rs.append(base)
    pd.DataFrame(rs).to_csv(work_file("taxonomy_delta_summary.csv"), index=False)
