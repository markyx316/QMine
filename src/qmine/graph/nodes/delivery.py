"""Phases 9-12 — uniform panel, deployment, reporting, maintenance."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from ...determinism import deterministic_subsample
from ...ops.cards import deterministic_exemplars
from ...ops.classify import CentroidClassifier
from ...ops.panel import UniformPanel
from ...state import PipelineState
from ..deps import Deps


def p9_panel(state: PipelineState, deps: Deps) -> dict[str, Any]:
    """Recompute every comparison metric through one harness.

    Numbers quoted from the phase that happened to produce them are not
    comparable — different sub-samples, different seeds, sometimes different
    code paths.  This phase throws all of that away and re-measures every
    candidate the run considered, so the report's comparison table is a table
    rather than a collage.
    """
    cfg = deps.cfg
    H = deps.embedding("emb_hybrid")
    labels_pre = deps.load("leaf_labels")          # the partition before governance
    leaf_family = deps.load("leaf_family")
    labels = deps.leaf_labels_final()              # what actually ships
    final_family = deps.leaf_family_final()
    centroids = deps.leaf_centroids_final()
    masks = deps.template_masks()
    df = deps.df
    deps.emit("P9 uniform panel — re-measuring every candidate under one harness")

    ref_labels = None
    for c in cfg.data.reference_label_columns:
        if c in df.columns:
            ref_labels = df[c].astype(str).to_numpy()
            break

    panel = UniformPanel(len(H), subsample=cfg.clustering.silhouette_sample, seed=cfg.seed_metric,
                         replay_seeds=tuple(cfg.seed_replay))

    panel.measure("leaves", H, labels, template_masks=masks, reference_labels=ref_labels,
                  centroids=centroids, margin_threshold=cfg.deployment.margin_threshold,
                  distill=not cfg.fast_mode)
    panel.measure("families_pre_governance", H, leaf_family[labels_pre], template_masks=masks,
                  reference_labels=ref_labels, heldout=False)
    panel.measure("families_final", H, final_family[labels], template_masks=masks,
                  reference_labels=ref_labels, heldout=False)

    # Every alpha the sweep considered, re-measured here rather than quoted.
    dense = deps.load("emb_base") if deps.has("emb_base") else None
    svd = deps.load("emb_svd_char") if deps.has("emb_svd_char") else None
    if dense is not None and svd is not None and not cfg.fast_mode:
        from ...ops.cluster import kmeans_labels
        from ...ops.represent import hybrid

        k = state.get("family_k") or cfg.representation.alpha_sweep_k
        for a in cfg.representation.alpha_grid:
            Ha = hybrid(dense, svd, a)
            panel.measure(f"alpha_{a}", Ha, kmeans_labels(Ha, k, seed=cfg.seed_metric),
                          template_masks=masks, reference_labels=ref_labels, heldout=False)

    naming = deps.load("tree_naming") if deps.has("tree_naming") else {}
    if naming.get("mean_coherence"):
        panel.add_external("leaves", "coherence", naming["mean_coherence"],
                           note="mean blind-review coherence, 1-5")
    agreement = deps.load("gold_agreement") if deps.has("gold_agreement") else {}
    if agreement.get("agreement", {}).get("kappa") is not None:
        panel.add_external("topdown", "kappa", agreement["agreement"]["kappa"],
                           note="inter-annotator agreement on the gold set")

    table = panel.comparison_table()
    ref = deps.store.put_json("metrics_panel", {"table": table,
                                                "sets": {k: v.model_dump() for k, v in panel.sets().items()}},
                              producer="p9", summary=f"panel {panel.panel_id}, {len(table['rows'])} candidates")
    deps.cache_put("panel", panel)

    leaves = panel.sets().get("leaves")
    events = [f"P9: panel {panel.panel_id} over {len(table['rows'])} candidates"]
    if leaves:
        events.append(
            f"P9: leaves — stability {leaves.get('stability_ari')}, "
            f"fragmentation {leaves.get('template_fragmentation')}, "
            f"held-out {leaves.get('heldout_reproduction')}, "
            f"silhouette {leaves.get('silhouette')} (advisory)"
        )
    return {
        "phase": "p10",
        "artifacts": {"metrics_panel": ref},
        "metrics": panel.sets(),
        "completed_phases": ["p9"],
        "events": events,
    }


def p10_deploy(state: PipelineState, deps: Deps) -> dict[str, Any]:
    """Assemble the delivered table and the deployable classifier.

    The delivered table carries **both** label systems side by side (Principle 1)
    plus the margin, so a consumer can route on confidence without re-deriving
    anything, and can see where the two routes agree and where they do not.
    """
    cfg = deps.cfg
    df = deps.df
    H = deps.embedding("emb_hybrid")
    labels_pre = deps.load("leaf_labels")
    labels = deps.leaf_labels_final()
    centroids = deps.leaf_centroids_final()
    leaf_family = deps.load("leaf_family")
    final_family = deps.leaf_family_final()
    deps.emit("P10 deployment — building the delivered table and centroid model")

    names: dict[int, str] = {}
    needs: dict[int, str] = {}
    naming = deps.load("tree_naming") if deps.has("tree_naming") else {}
    for n in naming.get("namings", []):
        names[int(n["leaf_id"])] = n.get("name_zh", "")
        needs[int(n["leaf_id"])] = n.get("user_need", "")
    # Governance renames land in the delivered column, not just the ledger.
    names.update(deps.recover("leaf_relabels", "leaf_relabels", default={}) or {})

    clf = CentroidClassifier(centroids, final_family, names=names,
                             margin_threshold=cfg.deployment.margin_threshold,
                             alpha=state.get("chosen_alpha", 0.0))
    routing = clf.route(H)
    p = routing["predictions"]

    out = pd.DataFrame({
        cfg.data.text_column: df[cfg.data.text_column],
        "bu_leaf": labels,
        "bu_leaf_name": [names.get(int(l), "") for l in labels],
        "bu_user_need": [needs.get(int(l), "") for l in labels],
        "bu_leaf_pre_governance": labels_pre,
        "bu_family_pre_governance": leaf_family[labels_pre],
        "bu_family_final": final_family[labels],
        "bu_margin": np.round(p["margin"], 5),
        "bu_ambiguous": p["ambiguous"],
    })
    # Recovered from the artifact on a resumed run. Reading process memory here
    # once dropped the entire top-down label column from the delivered table
    # without any error — a silent violation of the two-routes-side-by-side rule.
    td = deps.recover(
        "topdown_preds", "topdown_labels",
        rebuild=lambda df: df["l1_pred"].to_numpy(),
    )
    if td is not None:
        out["td_l1"] = td
    sub = deps.recover(
        "topdown_sub", "topdown_l2_labels", rebuild=lambda d: d["td_l2"].to_numpy()
    )
    if sub is not None:
        out["td_l2"] = sub
    for c in cfg.data.reference_label_columns:
        if c in df.columns:
            out[f"ref_{c}"] = df[c]
    out["run_id"] = deps.run_id
    out["generation"] = deps.store.generation

    labels_ref = deps.store.put_table("labels_full", out, fmt="csv", producer="p10",
                                      summary=f"{len(out)} rows, both label systems side by side")
    model_ref = deps.store.put_model("centroid_classifier", clf, producer="p10",
                                     summary=f"{clf.size_bytes() / 1024:.0f} KB centroid matrix")

    # Live routing demo on rows the tree never saw during construction.
    holdout = deterministic_subsample(len(df), min(cfg.deployment.live_demo_n * 20, len(df)), cfg.seed_viz)
    demo_idx = holdout[:: max(len(holdout) // cfg.deployment.live_demo_n, 1)][: cfg.deployment.live_demo_n]
    demo = [
        {
            "query": str(df[cfg.data.text_column].iloc[i]),
            "leaf": int(p["leaf"][i]),
            "leaf_name": names.get(int(p["leaf"][i]), ""),
            "family": int(p["family"][i]),
            "margin": round(float(p["margin"][i]), 4),
            "routed": "fallback" if p["ambiguous"][i] else "direct",
        }
        for i in demo_idx
    ]

    masks = deps.template_masks(trusted=False)
    deploy_ref = deps.store.put_json(
        "deployment",
        {"routing": {k: v for k, v in routing.items() if k != "predictions"},
         "model_bytes": clf.size_bytes(),
         "inference": "encode(query) → hybrid transform → argmax(x @ centroids.T)",
         "alpha": state.get("chosen_alpha", 0.0),
         "live_demo": demo,
         "deterministic_exemplars": deterministic_exemplars(df, masks, text_col=cfg.data.text_column)},
        producer="p10", summary=f"ambiguous {routing['ambiguous_rate'] * 100:.1f}%",
    )
    return {
        "phase": "p11",
        "artifacts": {"labels_full": labels_ref, "centroid_classifier": model_ref, "deployment": deploy_ref},
        "completed_phases": ["p10"],
        "events": [
            f"P10: delivered {len(out)} rows; model {clf.size_bytes() / 1024:.0f} KB; "
            f"{routing['ambiguous_rate'] * 100:.1f}% route to fallback"
        ],
    }


def p11_report(state: PipelineState, deps: Deps) -> dict[str, Any]:
    """Write the reports and the executed notebook."""
    from ...report.builder import build_all_reports

    deps.emit("P11 reporting — assembling deliverables")
    refs = build_all_reports(state, deps)
    return {
        "phase": "p12",
        "artifacts": refs,
        "completed_phases": ["p11"],
        "events": [f"P11: {len(refs)} deliverables written to {deps.store.gen_dir}"],
    }


def p12_maintain(state: PipelineState, deps: Deps) -> dict[str, Any]:
    """Write the maintenance kit: drift baseline, novelty sentinel, rerun contract.

    The sentinel is the density method that lost the Phase 4 bake-off.  It was
    never the right tool for the main partition and is exactly the right tool for
    "this query resembles nothing we have seen", so it gets a job rather than a
    footnote.
    """
    cfg = deps.cfg
    H = deps.embedding("emb_hybrid")
    centroids = deps.leaf_centroids_final()
    df = deps.df
    deps.emit("P12 maintenance — baseline, novelty sentinel, rerun contract")

    sims = (H @ centroids.T).max(1)
    cut = float(np.quantile(sims, 0.01))
    novel_idx = np.where(sims <= cut)[0]
    novel = [str(df[cfg.data.text_column].iloc[i]) for i in novel_idx[:60]]

    labels = deps.leaf_labels_final()
    final_family = deps.leaf_family_final()
    naming = deps.load("tree_naming") if deps.has("tree_naming") else {}
    names = {int(n["leaf_id"]): n.get("name_zh", "") for n in naming.get("namings", [])}

    baseline = {
        "run_id": deps.run_id,
        "domain": cfg.domain.key,
        "generation": deps.store.generation,
        "config_hash": cfg.config_hash,
        "encoder": state.get("chosen_encoder"),
        "alpha": state.get("chosen_alpha"),
        "family_k": state.get("family_k"),
        "n_leaves": int(labels.max()) + 1,
        "n_families": int(len(np.unique(final_family))),
        "leaf_sizes": np.bincount(labels).tolist(),
        "leaf_names": names,
        "family_sizes": np.bincount(final_family[labels]).tolist(),
    }
    # If an earlier run of this project exists, diff against it and let the
    # maintenance analyst interpret the change. Without this the phase produces a
    # baseline nobody ever compares anything to, which is a filing cabinet rather
    # than a maintenance loop.
    drift: dict[str, Any] = {}
    prev = _previous_baseline(deps)
    if prev:
        from ...agents.roles import MaintainerAgent
        from ...ops.handoff import diff_runs

        comparison = diff_runs(prev, {"baseline": baseline})
        deps.emit(f"  diffed against run {prev.get('baseline', {}).get('run_id')} — "
                  f"{comparison['verdict'][:60]}")
        try:
            report = MaintainerAgent(deps.agent_ctx()).run(
                previous=json.dumps(prev.get("baseline", {}), ensure_ascii=False)[:12000],
                current=json.dumps(baseline, ensure_ascii=False)[:12000],
                novel=novel[:40],
            )
            drift = {"comparison": comparison, "analyst": report.model_dump()}
        except Exception as exc:  # noqa: BLE001
            drift = {"comparison": comparison, "analyst_error": str(exc)[:200]}
    else:
        drift = {"note": "no earlier run of this project found; this baseline is the first"}

    ref = deps.store.put_json(
        "maintenance",
        {
            "baseline": baseline,
            "novelty_sentinel": {
                "rule": "rows in the bottom 1% of max-centroid cosine",
                "threshold": round(cut, 4),
                "n_flagged": int(len(novel_idx)),
                "samples": novel,
                "why_this_method": (
                    "the density method that lost the Phase 4 bake-off is kept for exactly "
                    "this job: it is poor at partitioning a corpus and good at noticing "
                    "something that belongs to no partition"
                ),
            },
            "drift": drift,
            "rerun_contract": {
                "cadence": "quarterly by default; monthly or biweekly for fast-drifting verticals",
                "steps": [
                    "re-run P1 template mining — phrasing ecology drifts and alpha was tuned to it",
                    "re-run P3c alpha sweep; do NOT inherit the previous alpha",
                    "rebuild the tree, then diff family sizes and names against this baseline",
                    "route the novelty sentinel's flags to a human before the next naming pass",
                    "promote a new model's labels only after a head-to-head referee pass on "
                    "disagreements, keeping the old labels in a _v1 column",
                ],
                "diff_interpretation": (
                    "compare config_hash FIRST. If it differs, the two trees are not "
                    "comparable and any 'drift' you see may be method change."
                ),
            },
        },
        producer="p12", summary=f"baseline + {len(novel_idx)} novel-query flags",
    )
    return {
        "phase": "done",
        "artifacts": {"maintenance": ref},
        "completed_phases": ["p12"],
        "events": [f"P12: baseline stored, novelty sentinel flagged {len(novel_idx)} queries"],
    }


def _previous_baseline(deps: Any) -> dict[str, Any] | None:
    """The most recent earlier baseline **for this same domain**, if one exists.

    Filtering on domain is not fussiness. Diffing an English e-commerce tree
    against a Chinese K12 tree produces a page of confident nonsense — every
    family "appeared", every family "vanished" — which is worse than no diff at
    all because it looks like a finding.

    The exception handling here is deliberately narrow. An earlier version caught
    bare ``Exception`` around the JSON read, which silently swallowed a
    ``NameError`` from a missing import and made this function return ``None``
    forever: the maintenance loop reported "no earlier run found" on every run
    and nobody noticed, because that is also the correct answer the first time.
    A malformed file should be skipped; a bug in this module should not be.
    """
    from pathlib import Path

    root = Path(deps.cfg.run_root)
    if not root.exists():
        return None

    domain = deps.cfg.domain.key
    candidates: list[tuple[float, dict[str, Any]]] = []
    for gen in sorted(root.glob("*/gen*/maintenance.json")):
        if deps.run_id in str(gen):
            continue
        try:
            blob = json.loads(gen.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        b = blob.get("baseline", {})
        if not (b.get("run_id") and b.get("config_hash")):
            continue
        # Strict: a baseline that does not record its domain is skipped rather
        # than assumed to match. Predating the field is not evidence of being
        # the same vertical, and a wrong cross-domain diff reads as a finding.
        if b.get("domain") != domain:
            continue
        candidates.append((gen.stat().st_mtime, blob))

    if not candidates:
        return None
    candidates.sort(key=lambda t: -t[0])
    return candidates[0][1]
