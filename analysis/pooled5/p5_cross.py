# -*- coding: utf-8 -*-
"""Cross-domain view: the same four contrasts side by side in all five domains, so a pattern
can be called general only when it actually repeats.

    python analysis/pooled5/p5_cross.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pooled5_common import CONTRASTS, SOURCES, SRC_ZH, WORK, cohort, cross_dir, work_file

def _cat(name: str, **read) -> pd.DataFrame:
    parts = []
    for dom in cohort():
        p = WORK / dom / name
        if p.exists():
            t = pd.read_csv(p, **read)
            t.insert(0, "domain", dom)
            parts.append(t)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def main() -> None:
    out = cross_dir()
    out.mkdir(parents=True, exist_ok=True)
    cs = _cat("contrast_summary_td_l1_name.csv")
    cs.to_csv(out / "contrast_summary_all.csv", index=False)
    if len(cs):
        piv = cs.pivot_table(index=["a", "b", "why"], columns="domain", values="tvd")
        piv = piv.reindex([(a, b, w) for a, b, w in CONTRASTS if (a, b, w) in piv.index])
        piv.round(3).to_csv(out / "tvd_matrix.csv")
        print("TVD（需要换类的行占比）\n" + piv.round(3).to_string())
    form = _cat("form.csv")
    form.to_csv(out / "form_all.csv", index=False)
    fit = _cat("fit.csv")
    fit.to_csv(out / "fit_all.csv", index=False)
    comp = _cat("composition.csv")
    comp.to_csv(out / "composition_all.csv", index=False)
    risk = _cat("risk_by_source.csv")
    if len(risk):
        risk.to_csv(out / "risk_all.csv", index=False)
        print("\n风控命中（每千行）\n" + risk.pivot_table(index="domain", columns="source", values="per_1k").to_string())
    diffs = _cat("diffs_td_l1_name.csv")
    diffs.to_csv(out / "diffs_all.csv", index=False)
    if len(diffs):
        sig = diffs[diffs.sig]
        (sig.groupby(["a", "b", "domain"]).size().unstack("domain").fillna(0).astype(int)
         .to_csv(out / "sig_class_counts.csv"))
    nn = work_file("semantic_nn.parquet")
    if nn.exists():
        x = pd.read_parquet(nn)
        far = (x.assign(far=x.sim < 0.70, same=x.sim >= 0.999)
                .groupby(["domain", "source"]).agg(n=("sim", "size"), median_sim=("sim", "median"),
                                                   same_pct=("same", "mean"), far_pct=("far", "mean")).reset_index())
        far["same_pct"] = (100 * far.same_pct).round(1)
        far["far_pct"] = (100 * far.far_pct).round(1)
        far.to_csv(out / "semantic_nn_summary.csv", index=False)
        print("\n无近邻（余弦<0.70）占比 %\n" + far.pivot_table(index="domain", columns="source", values="far_pct")
              .reindex(columns=[s for s in SOURCES if s in set(far.source)]).to_string())
    td = work_file("taxonomy_delta_summary.csv")
    if td.exists():
        print("\n体系位移（同样的 2 万条搜索行，加入助手行前后）\n" + pd.read_csv(td).to_string(index=False))
    # WHICH SURFACE PULLS WHICH KIND OF CLASS, across domains. The index is a ratio of
    # within-source shares (see p5_tables), so the sources' different row counts cannot drive it.
    prof = _cat("class_profile.csv")
    if len(prof):
        # `_cat` inserts `domain` at position 0, so the class name is whatever the CSV's own
        # first column is called (`td_l1_name`). Renaming `columns[0]` would rename `domain`.
        first = [c for c in prof.columns if c not in ("domain",)][0]
        prof = prof.rename(columns={first: "label"})
        prof.to_csv(out / "class_profile_all.csv", index=False)
        lines = []
        for src in ["2026search", "assistant_top", "assistant_random", "assistant_voice"]:
            col = f"index_{src}"
            if col not in prof.columns:
                continue
            lines.append(f"\n【{SRC_ZH[src]} 最集中的类目（index = 来源内占比 / 全语料占比）】")
            for dom in cohort():
                sub = prof[(prof.domain == dom) & prof[col].notna()].nlargest(3, col)
                if len(sub):
                    lines.append(f"  {dom}: " + " | ".join(f"{r.label[:20]} {getattr(r, col):.1f}×" for r in sub.itertuples()))
        (out / "surface_signature.txt").write_text("\n".join(lines), encoding="utf-8")
        print("\n".join(lines))
    inv = []
    for dom in cohort():
        f = WORK / dom / "class_definitions.csv"
        if f.exists():
            t = pd.read_csv(f)
            inv.append({"domain": dom, "n_classes": len(t), "classes": " / ".join(t["name"].astype(str))})
    if inv:
        pd.DataFrame(inv).to_csv(out / "taxonomy_inventory.csv", index=False)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
