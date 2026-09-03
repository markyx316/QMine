#!/usr/bin/env python
"""Aggregate every COMPLETE live run into one comparable table.

Written for the README: a claim backed by one run is an anecdote, and this repo
now has thirteen. It exists as a tool rather than a scratch script because the
table goes stale the moment another run lands, and a figure regenerated from a
stale hand-copied number is worse than no figure.

THREE TRAPS IT ENCODES, EACH FROM A REAL MISTAKE.

* **Delivered, not p7's.** `hierarchy_meta` and `leaf_labels` describe the tree
  BEFORE p8 governance rewrote it. Leaves come from `leaf_labels_final` and are
  cross-checked against the shipped document (fin-pool 54/34, live44 53/23).
* **Families are indexed BY LEAF.** Counting `set(leaf_family_final)` includes
  entries for leaves governance merged away — it reported 26 families for
  ppl-pool's 25 leaves, which is impossible. Only delivered leaves are counted.
* **Absent is not zero.** A fast run has ONE annotator, so its kappa is `None`.
  Never render it as 0, and never average it in with the full runs.

Usage:  HF_HOME=$(pwd)/.hf .venv/bin/python tools/run_evidence.py [--json out.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

#: A run that halted early is not evidence of anything. 17 phases is a complete
#: pipeline; 15 allows for a run that legitimately skipped a late optional phase.
MIN_PHASES = 15


def _json(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:  # noqa: BLE001
        return {}


def _npy(*paths: str):
    for p in paths:
        try:
            return np.load(p, allow_pickle=True)
        except Exception:  # noqa: BLE001
            continue
    return None


def collect(root: str = "runs") -> list[dict]:
    out = []
    for summary in sorted(glob.glob(f"{root}/*/gen*/run_summary.json")):
        gen = os.path.dirname(summary)
        s = _json(summary)
        if (s.get("llm_usage") or {}).get("provider") != "routed":
            continue                       # offline output looks complete and is a stand-in
        if len(s.get("completed_phases") or []) < MIN_PHASES:
            continue
        audit = _json(f"{gen}/data_audit.json")
        agree = (_json(f"{gen}/gold_agreement.json") or {}).get("agreement", {})
        conc = (_json(f"{gen}/route_concordance.json") or {}).get("by_level", {})
        # RECORD WHICH PARTITION THE SHAPE CAME FROM. Falling back to the
        # pre-governance array is correct only when governance changed nothing
        # (film-pool: 12 leaves in, 12 out, verified against its shipped
        # document). Silently falling back on a run where it DID change would
        # publish p7's tree as the delivered one — so the source ships with the
        # number and a caller can refuse the fallback.
        final = os.path.exists(f"{gen}/leaf_labels_final.npy")
        lab = _npy(f"{gen}/leaf_labels_final.npy", f"{gen}/leaf_labels.npy")
        fam = _npy(f"{gen}/leaf_family_final.npy", f"{gen}/leaf_family.npy")
        leaves = sorted({int(x) for x in lab.tolist()}) if lab is not None else []
        families = ({int(fam[i]) for i in leaves if i < len(fam)}
                    if fam is not None else set())
        # `taxonomy` is {version, axes, nodes, rules, ...} — the L1 classes are
        # `nodes`, and `rules` is the adjudication set the annotator may cite.
        tax = (_json(f"{gen}/taxonomy_v2.json") or {}).get("taxonomy", {})
        classes = tax.get("nodes") if isinstance(tax, dict) else tax
        rules = tax.get("rules") if isinstance(tax, dict) else None
        usage = s.get("llm_usage") or {}
        gates = s.get("gates") or {}
        out.append({
            "run": summary.split("/")[1],
            "generation": summary.split("/")[2],
            "mode": s.get("mode") or "full",
            "rows": audit.get("n_rows"),
            "n_classes": len(classes) if isinstance(classes, list) else None,
            "n_rules": len(rules) if isinstance(rules, list) else None,
            "shape_source": "delivered" if final else "pre_governance_fallback",
            "leaves": len(leaves) or None,
            "families": len(families) or None,
            # None, never 0.0 — a fast run did not measure this.
            "kappa": agree.get("kappa"),
            "kappa_n": agree.get("n"),
            "raw_agreement": agree.get("raw_agreement"),
            "ami_leaf": (conc.get("leaf") or {}).get("ami"),
            "ami_family": (conc.get("family") or {}).get("ami"),
            "calls": usage.get("calls") or usage.get("n_calls"),
            "cost_usd": round(usage.get("estimated_cost_usd") or 0, 2) or None,
            "hours": round((s.get("elapsed_s") or 0) / 3600, 2) or None,
            "phases": len(s.get("completed_phases") or []),
            "gates_recorded": len(gates),
            "gates_not_passed": sum(1 for g in gates.values()
                                    if g.get("status") in ("warned", "failed")),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", help="write the table here as JSON")
    ap.add_argument("--root", default="runs")
    a = ap.parse_args()
    rows = collect(a.root)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, ensure_ascii=False, indent=1)
    hdr = (f"{'run':12s}{'mode':6s}{'rows':>7s}{'cls':>5s}{'leaf':>5s}{'fam':>5s}"
           f"{'kappa':>8s}{'n':>6s}{'AMI':>7s}{'calls':>6s}{'cost':>8s}{'hrs':>6s}{'gates':>7s}")
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(rows, key=lambda x: (-(x["rows"] or 0), x["run"])):
        k = f"{r['kappa']:.4f}" if isinstance(r["kappa"], (int, float)) else "—"
        ami = f"{r['ami_leaf']:.3f}" if isinstance(r["ami_leaf"], (int, float)) else "—"
        print(f"{r['run']:12s}{r['mode']:6s}{str(r['rows'] or ''):>7s}"
              f"{str(r['n_classes'] or ''):>5s}{str(r['leaves'] or ''):>5s}"
              f"{str(r['families'] or ''):>5s}{k:>8s}{str(r['kappa_n'] or ''):>6s}{ami:>7s}"
              f"{str(r['calls'] or ''):>6s}"
              f"{('$%.2f' % r['cost_usd']) if r['cost_usd'] else '—':>8s}"
              f"{r['hours'] if r['hours'] else 0:>6.2f}"
              f"{str(r['gates_not_passed']) + '/' + str(r['gates_recorded']):>7s}")
    print(f"\n{len(rows)} complete live runs")


if __name__ == "__main__":
    main()
