# -*- coding: utf-8 -*-
"""Canonical per-domain tables for the POOLED-5 comparison. Deterministic: every number an
agent later interprets is computed here, so the interpretation can be re-derived.

    python analysis/pooled5/p5_tables.py [domain ...]
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pooled5_common import (CONTRASTS, COHORT, SOURCES, SRC_ZH, WORK, all_rows,
                            available, cramers_v,
                            diff_table, form_table, load, overlap_matrix, run_dir, shares)

LABELS = [("td_l1_name", "自上而下意图(L1)"), ("bu_family_final", "自下而上家族"), ("bu_leaf_name", "自下而上叶子")]


def _defs(domain: str) -> pd.DataFrame:
    """Class definitions from the run, so an interpretation can cite what the class MEANS."""
    p = run_dir(domain) / "taxonomy_v2.json"
    if not p.exists():
        p = run_dir(domain) / "taxonomy.json"
    t = json.loads(p.read_text(encoding="utf-8"))
    tax = t.get("taxonomy", t)
    nodes = [n for n in tax.get("nodes", []) if n.get("level") == 1]
    rows = [{"code": n.get("code"), "name": n.get("name"), "definition": n.get("definition", ""),
             "user_need": n.get("user_need", ""), "risk": n.get("risk", False),
             "expected_share": n.get("expected_share"),
             "positive_examples": " / ".join(n.get("positive_examples", [])[:4]),
             "negative_examples": " / ".join(n.get("negative_examples", [])[:3]),
             "n_rules": len(n.get("adjudication_rules", []))} for n in nodes]
    return pd.DataFrame(rows)


def _examples(df: pd.DataFrame, label_col: str, per: int = 6) -> pd.DataFrame:
    out = []
    for (src, lab), g in df.groupby(["source", label_col], sort=False):
        g = g.sort_values("pv_norm", ascending=False)
        for _, r in g.head(per).iterrows():
            out.append({"source": src, "label": lab, "rank": "top_pv", "query": r["query"],
                        "pv_raw": r["pv_raw"], "tier": r["tier"]})
        if len(g) > per:
            for _, r in g.iloc[per:].sample(min(per, len(g) - per), random_state=0).iterrows():
                out.append({"source": src, "label": lab, "rank": "sample", "query": r["query"],
                            "pv_raw": r["pv_raw"], "tier": r["tier"]})
    return pd.DataFrame(out)


def _concentration(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    out = []
    for (src, lab), g in df.groupby(["source", label_col], sort=False):
        vc = g["query"].value_counts(normalize=True)
        out.append({"source": src, "label": lab, "n": len(g),
                    "distinct": int(g["query"].nunique()),
                    "hhi_strings": float((vc ** 2).sum()),
                    "top1_pv_share": float(g["pv_norm"].max() / g["pv_norm"].sum()) if g["pv_norm"].sum() else 0.0,
                    "top5_pv_share": float(g["pv_norm"].nlargest(5).sum() / g["pv_norm"].sum()) if g["pv_norm"].sum() else 0.0})
    return pd.DataFrame(out)


def _fit(df: pd.DataFrame) -> pd.DataFrame:
    """How well the mined frame FITS each source — the frame was fitted on a corpus that is 87%
    search, so a source sitting in low-confidence or ambiguous rows is a finding, not noise."""
    rows = []
    for src, g in df.groupby("source", sort=False):
        rows.append({"source": src, "n": len(g),
                     "td_confidence_mean": float(g["td_confidence"].mean()),
                     "td_confidence_p10": float(g["td_confidence"].quantile(.1)),
                     "td_ambiguous_%": float(100 * g["td_ambiguous"].mean()),
                     "bu_ambiguous_%": float(100 * g["bu_ambiguous"].mean()),
                     "td_margin_mean": float(g["td_margin"].mean()),
                     "decided_by_model_%": float(100 * (g["td_decided_by"] == "model").mean()),
                     "distinct_td_l1": int(g["td_l1_name"].nunique()),
                     "distinct_bu_leaf": int(g["bu_leaf_name"].nunique())})
    return pd.DataFrame(rows)


def _contrast_summary(d: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """ONE comparable number per contrast: how far apart are the two intent profiles.

    Total variation distance (half the L1 distance over the union of classes) answers "what
    share of one population would have to change class to look like the other" — unlike a
    p-value it does not grow with n, and every contrast here has n in the thousands.
    """
    out = []
    for a_s, b_s, why in CONTRASTS:
        ga, gb = d[d.source == a_s], d[d.source == b_s]
        if not len(ga) or not len(gb):
            continue
        pa = ga[label_col].value_counts(normalize=True)
        pb = gb[label_col].value_counts(normalize=True)
        keys = set(pa.index) | set(pb.index)
        tvd = 0.5 * sum(abs(float(pa.get(k, 0.0)) - float(pb.get(k, 0.0))) for k in keys)
        ct = pd.crosstab(pd.concat([ga[label_col], gb[label_col]]),
                         pd.concat([ga["source"], gb["source"]]))
        t = diff_table(d, label_col, a_s, b_s)
        movers = t.reindex(t.diff_pp.abs().sort_values(ascending=False).index).head(5)
        out.append({"a": a_s, "b": b_s, "why": why, "n_a": len(ga), "n_b": len(gb),
                    "tvd": round(tvd, 4), "cramers_v": round(cramers_v(ct.values), 4),
                    "n_sig_classes": int(t.sig.sum()) if len(t) else 0,
                    "emergent": int((t.kind == "emergent").sum()) if len(t) else 0,
                    "receded": int((t.kind == "receded").sum()) if len(t) else 0,
                    "top_movers": " | ".join(f"{r.label} {r.diff_pp:+.1f}pp" for r in movers.itertuples())})
    return pd.DataFrame(out)


def domain_tables(domain: str) -> dict:
    d = load(domain)
    out = WORK / domain
    out.mkdir(parents=True, exist_ok=True)
    a = all_rows(domain)
    comp = (a.groupby("source", sort=False)
             .agg(rows=("query", "size"), kept=("kept", "sum"),
                  pv_raw=("pv_raw", "sum"), pv_dropped=("pv_raw", lambda s: s[~a.loc[s.index, "kept"]].sum()))
             .reset_index())
    comp["drop_%"] = 100 * (1 - comp.kept / comp.rows)
    comp["pv_drop_%"] = 100 * comp.pv_dropped / comp.pv_raw.replace(0, float("nan"))
    comp.to_csv(out / "composition.csv", index=False)
    (a[~a.kept].groupby(["source", "tier"]).size().rename("rows").reset_index()
     .to_csv(out / "cleaning_removed.csv", index=False))
    _defs(domain).to_csv(out / "class_definitions.csv", index=False)
    form_table(d).to_csv(out / "form.csv", index=False)
    overlap_matrix(d).to_csv(out / "overlap.csv")
    _fit(d).to_csv(out / "fit.csv", index=False)
    summary = {"domain": domain, "n_rows": len(d),
               "sources": {s: int((d.source == s).sum()) for s in SOURCES if (d.source == s).any()}}
    for col, zh in LABELS:
        shares(d, col).to_csv(out / f"shares_{col}.csv", index=False)
        _concentration(d, col).to_csv(out / f"concentration_{col}.csv", index=False)
        if col != "bu_leaf_name":
            _examples(d, col).to_csv(out / f"examples_{col}.csv", index=False)
        parts = []
        for a_s, b_s, why in CONTRASTS:
            t = diff_table(d, col, a_s, b_s)
            if len(t):
                t["why"] = why
                parts.append(t)
        if parts:
            pd.concat(parts, ignore_index=True).to_csv(out / f"diffs_{col}.csv", index=False)
        _contrast_summary(d, col).to_csv(out / f"contrast_summary_{col}.csv", index=False)
        ct = pd.crosstab(d[col], d["source"])
        summary[f"cramers_v_{col}"] = round(cramers_v(ct.values), 4)
        summary[f"n_classes_{col}"] = int(d[col].nunique())
    # WHERE IS THIS CLASS CONCENTRATED. The index is (class share inside the source) / (class
    # share over the whole corpus): 1.0 = the class is as common here as everywhere, 3.0 = three
    # times as concentrated. It is a ratio of within-source shares, so the sources' very
    # different row counts cannot drive it.
    base = d["td_l1_name"].value_counts(normalize=True)
    prof = (d.groupby(["td_l1_name", "source"]).size().unstack(1).fillna(0))
    prof = prof / prof.sum()
    # 4 decimals, not 2: the report formats this to one decimal, and rounding twice turned
    # 4.2549 into "4.2" instead of "4.3" (caught by the report's number verifier).
    idx = prof.div(base, axis=0).round(4)
    idx.columns = [f"index_{c}" for c in idx.columns]
    pd.concat([(prof * 100).round(2).add_prefix("share%_"), idx], axis=1).to_csv(out / "class_profile.csv")
    # PV-weighted class shares for the head slices, kept separate: the assistant head's PV is
    # dominated by cleaning decisions, so this is a supplement, never the headline.
    (d.groupby(["source", "td_l1_name"])["pv_norm"].sum().unstack(0).fillna(0)
     .to_csv(out / "pv_shares_td_l1_name.csv"))
    # RISK LAYER BY SOURCE. `flag_mask_indices` are positions in the corpus frame, which is the
    # input file's order — the same order `load()` asserts on — so they map straight to sources.
    rp = run_dir(domain) / "risk_screen.json"
    if rp.exists():
        rs = json.loads(rp.read_text(encoding="utf-8"))
        idx = [i for i in rs.get("flag_mask_indices", []) if 0 <= i < len(d)]
        hit = d.iloc[idx][["query", "source", "td_l1_name", "pv_raw"]].copy() if idx else pd.DataFrame(
            columns=["query", "source", "td_l1_name", "pv_raw"])
        hit.to_csv(out / "risk_rows.csv", index=False)
        by = (hit.groupby("source").size().rename("hits").reindex(
            [s for s in SOURCES if (d.source == s).any()]).fillna(0).astype(int).reset_index())
        by["n"] = [int((d.source == s).sum()) for s in by["source"]]
        by["per_1k"] = (1000 * by.hits / by.n).round(2)
        by.to_csv(out / "risk_by_source.csv", index=False)
        summary["risk_total"] = int(rs.get("total_flagged", 0))
    for extra in ("drift_analysis.json", "risk_screen.json", "metrics_panel.json", "granularity.json"):
        p = run_dir(domain) / extra
        if p.exists():
            (out / extra).write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{domain}: {len(d):,} rows · " + " · ".join(f"{SRC_ZH[k]} {v:,}" for k, v in summary["sources"].items())
          + f" · L1 classes {summary['n_classes_td_l1_name']} · V(source×L1) {summary['cramers_v_td_l1_name']}")
    return summary


if __name__ == "__main__":
    # No argument means "this batch's finished domains", not "every domain on disk": since the
    # two 2026-09-13 verticals joined DOMAINS, a bare sweep would try to load runs that may not
    # exist yet, and would write a seven-domain summary under the five-domain report's filename.
    doms = sys.argv[1:] or available()
    allsum = [domain_tables(x) for x in doms]
    name = "summary_all.json" if COHORT == "pool5" else f"summary_all_{COHORT}.json"
    (WORK / name).write_text(json.dumps(allsum, ensure_ascii=False, indent=1), encoding="utf-8")
