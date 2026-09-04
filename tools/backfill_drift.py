#!/usr/bin/env python
"""Compute `p10b` for a pooled run that finished BEFORE the phase existed.

Five pooled runs (fin/film/med/edu/ppl) were produced before `p10b_drift` was
built, so they carry no `drift_analysis.json` and no 快照对比_漂移分析.md. Nothing
about them needs re-running to fix that: **p10b makes no model calls.** It is a
deterministic function of three things already on disk —

  * the delivered labels          `runs/<id>/<gen>/labels_full.csv`
  * the snapshot tag per row      the pooled source CSV's `_snapshot` column
  * the family naming             `tree_naming.json` + the leaf arrays

— so re-running the pipeline would spend money to recompute labels that already
exist, and would produce DIFFERENT ones (the same 20,000 film rows run twice
delivered 12 leaves and then 34). Backfilling is not the cheap approximation
here; it is the only method that preserves the labels the reports describe.

FAITHFULNESS. This calls `ops.drift` and `report.zh_drift` with the same
arguments `p10b_drift` and `_drift_document` pass, so the artifact and the
document are byte-comparable with a live run's. Verified against `filmdrift`,
which DID run p10b live: backfilling it reproduces its shipped numbers exactly.

WHAT IT REFUSES. A positional join is only sound when the two frames are the
same length and the same order. `labels_full.csv` is written in corpus order, but
p1 DROPS rows with empty text (one row on 教育), which silently shifts every label
after it. So the join is refused on a length mismatch unless a reconciled labels
file exists, and `--check` re-derives the alignment before anything is written.

Output goes to a NEW GENERATION. The old one is evidence and is never edited.

    HF_HOME=$(pwd)/.hf .venv/bin/python tools/backfill_drift.py fin-pool
    HF_HOME=$(pwd)/.hf .venv/bin/python tools/backfill_drift.py --all --check
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from qmine.graph.nodes.foundation import _read_one  # noqa: E402
from qmine.ops import drift  # noqa: E402
from qmine.report import zh_drift  # noqa: E402
from qmine.report._shape import family_names  # noqa: E402

#: The label axes p10b compares, in the order it writes them.
AXES = ("td_l1", "bu_leaf_name", "bu_family_final")


def _newest_gen(run: Path) -> Path:
    gens = sorted(p for p in run.glob("gen*") if p.is_dir())
    if not gens:
        raise SystemExit(f"{run}: no generations")
    return gens[-1]


def _labels(gen: Path, n_expected: int) -> pd.DataFrame:
    """Prefer a reconciled file when p1 dropped a row; refuse a silent misalign."""
    for name in ("labels_full_reconciled.csv", "labels_full.csv"):
        p = gen / name
        if not p.exists():
            continue
        lab = pd.read_csv(p, encoding="utf-8-sig")
        if len(lab) == n_expected:
            return lab
        print(f"    {name}: {len(lab):,} rows against {n_expected:,} in the source — skipping")
    raise SystemExit(
        f"{gen}: no labels file aligns with the pooled source. A positional join on a "
        f"shifted frame mislabels every row after the drop; fix the alignment first.")


def _family_display(gen: Path, col: pd.Series) -> pd.Series:
    """`11` -> `#11 观看直播`, via LEAF MEMBERSHIP — never an integer id join."""
    try:
        naming = json.loads((gen / "tree_naming.json").read_text(encoding="utf-8"))
        fam = np.load(gen / ("leaf_family_final.npy" if (gen / "leaf_family_final.npy").exists()
                             else "leaf_family.npy"), allow_pickle=True)
        leaves = np.load(gen / ("leaf_labels_final.npy" if (gen / "leaf_labels_final.npy").exists()
                                else "leaf_labels.npy"), allow_pickle=True)
        names = family_names(naming, fam, np.bincount(np.asarray(leaves), minlength=len(fam)))
    except Exception as exc:  # noqa: BLE001
        print(f"    ⚠ family names unavailable ({type(exc).__name__}) — ids only")
        names = {}
    def _one(i):
        # A row can be genuinely unlabelled: the reconciled 教育 table reinserts
        # the row p1 dropped for empty text, with no labels. It must stay NaN so
        # the axis excludes it rather than inventing a family for it.
        if pd.isna(i):
            return np.nan
        return f"#{int(i)} {names[int(i)]}" if int(i) in names else f"#{int(i)}"

    return col.map(_one)


def compute(run_id: str, runs: Path, check_only: bool = False) -> dict | None:
    run = runs / run_id
    gen = _newest_gen(run)
    cfg = yaml.safe_load((gen / "config.resolved.yaml").read_text(encoding="utf-8"))
    data = cfg.get("data", cfg.get("domain", {}))
    src_path = Path(data.get("input_path") or cfg["domain"]["input_path"])
    text_col = data.get("text_column") or cfg["domain"]["text_column"]
    w_col = data.get("weight_column") or cfg["domain"].get("weight_column")
    snap_col = data.get("snapshot_column", "_snapshot")

    # A run may have been given ONE pre-pooled CSV (the five legacy runs) or
    # SEVERAL paths that the pipeline stacked itself (`filmdrift`). Reproduce the
    # second case the way `foundation._load_input` does, so both are backfillable
    # — which is what lets `filmdrift`, the one run that executed p10b live, act
    # as the control for this whole tool.
    paths = [Path(x) for x in (data.get("input_paths") or [])] or [src_path]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"{run_id}: source(s) gone, cannot re-derive tags: {missing}")

    if len(paths) > 1:
        from qmine.graph.nodes.foundation import _snapshot_tag
        frames = []
        for p_ in paths:
            one = _read_one(str(p_))
            one[snap_col] = _snapshot_tag(str(p_), one)
            frames.append(one)
        src = pd.concat(frames, ignore_index=True)
    else:
        src = _read_one(str(src_path)) if src_path.suffix != ".csv" else pd.read_csv(src_path)
    if snap_col not in src.columns:
        raise SystemExit(f"{run_id}: {src_path} has no {snap_col} column; not a pooled corpus")
    tags = sorted(src[snap_col].astype(str).unique())
    if len(tags) < 2:
        print(f"  {run_id}: one snapshot — nothing to compare")
        return None

    lab = _labels(gen, len(src))
    work = pd.DataFrame({
        "snapshot": src[snap_col].astype(str).values,
        "query": src[text_col].astype(str).values,
        "weight": src[w_col].astype(float).values if w_col and w_col in src.columns else 1.0,
    })
    present = [c for c in AXES if c in lab.columns]
    for c in present:
        work[c] = lab[c].values
    if "bu_family_final" in work.columns:
        work["bu_family_final"] = _family_display(gen, work["bu_family_final"])

    print(f"  {run_id}: {len(tags)} snapshots {tags}, {len(work):,} rows, axes {present}")
    if check_only:
        return None

    out = {
        "snapshots": tags,
        "inventory": drift.snapshot_inventory(work, "snapshot", "query", "weight"),
        "query_churn": drift.query_churn(work, "snapshot", "query", "weight"),
        "by_label": {}, "purity": {},
        "note": ("Shares are WITHIN-SNAPSHOT. The snapshots differ in total weight, "
                 "so raw counts would report every class as declining. Both periods "
                 "were labelled by ONE taxonomy and ONE tree in this run, which is "
                 "what excludes 'the taxonomy changed' as an explanation."),
        "backfilled": ("computed by tools/backfill_drift.py from this run's delivered "
                       "labels and the pooled source; the run predates phase p10b. "
                       "No model call was involved, then or now."),
    }
    for c in present:
        # AN UNLABELLED ROW IS NOT A ROW OF SOME CLASS. Left in, it inflates the
        # snapshot denominator that every share is computed against. One row in
        # 20,000 is 0.005% and would change nothing here, but the rule has to
        # hold for the case where it does.
        sub = work[work[c].notna()]
        dropped = len(work) - len(sub)
        if dropped:
            out.setdefault("excluded_unlabelled", {})[c] = dropped
            print(f"    {c}: {dropped} unlabelled row(s) excluded from this axis")
        out["by_label"][c] = drift.label_drift(sub, c, "snapshot", "weight", text_col="query")
        out["purity"][c] = drift.snapshot_purity(sub, c, "snapshot")
    return {"run_id": run_id, "gen": gen, "cfg": cfg, "out": out}


def emit(res: dict, out_dir: Path) -> None:
    out = res["out"]
    (out_dir / "drift_analysis.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    class _Deps:
        cfg = type("C", (), {"domain": type("D", (), {
            "key": res["cfg"].get("domain", {}).get("key", res["run_id"])})()})()

        @staticmethod
        def load(_):
            return out

    md = zh_drift.build({"run_id": res["run_id"]}, _Deps())
    (out_dir / "快照对比_漂移分析.md").write_text(md, encoding="utf-8")
    n_pure = sum(v.get("n_single_snapshot", 0) for v in out["purity"].values())
    print(f"    -> {out_dir}/  ({len(md):,} chars, {n_pure} single-snapshot group(s))")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", help="run ids; omit with --all")
    ap.add_argument("--all", action="store_true", help="every pooled run found")
    ap.add_argument("--check", action="store_true", help="verify alignment, write nothing")
    ap.add_argument("--out", help="write here instead of a new generation (for verification)")
    ap.add_argument("--root", default="runs")
    a = ap.parse_args()

    root = Path(a.root)
    ids = a.runs or ([p.name for p in sorted(root.iterdir())
                      if p.is_dir() and p.name.endswith("-pool")] if a.all else [])
    if not ids:
        raise SystemExit("name at least one run, or pass --all")

    for run_id in ids:
        res = compute(run_id, root, check_only=a.check)
        if res is None:
            continue
        if a.out:
            d = Path(a.out) / run_id
            d.mkdir(parents=True, exist_ok=True)
        else:
            os.system(f'.venv/bin/qmine new-generation {run_id} '
                      f'--reason "backfill p10b drift analysis; run predates the phase" >/dev/null')
            d = _newest_gen(root / run_id)
        emit(res, d)


if __name__ == "__main__":
    main()
