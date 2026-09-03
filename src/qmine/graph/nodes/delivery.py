"""Phases 9-12 — uniform panel, deployment, reporting, maintenance."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import observe as _observe
from ...determinism import deterministic_subsample
from ...ops.cards import deterministic_exemplars
from ...ops.classify import CentroidClassifier
from ...ops.panel import UniformPanel
from ...state import PipelineState
from ..deps import Deps

#: Filename stem of the agent-written report. Named so it sorts first in the
#: delivery directory: it is the document a reader is meant to open first.
FINAL_REPORT_STEM = "00_最终报告"


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
                  distill=not cfg.smoke_mode)
    panel.measure("families_pre_governance", H, leaf_family[labels_pre], template_masks=masks,
                  reference_labels=ref_labels, heldout=False)
    panel.measure("families_final", H, final_family[labels], template_masks=masks,
                  reference_labels=ref_labels, heldout=False)

    # Every alpha the sweep considered, re-measured here rather than quoted.
    dense = deps.load("emb_base") if deps.has("emb_base") else None
    svd = deps.load("emb_svd_char") if deps.has("emb_svd_char") else None
    if dense is not None and svd is not None and not cfg.smoke_mode:
        from ...ops.cluster import kmeans_labels
        from ...ops.represent import hybrid

        from ...config import alpha_sweep_k_for

        k = state.get("family_k") or alpha_sweep_k_for(cfg)
        for a in cfg.representation.alpha_grid:
            Ha = hybrid(dense, svd, a)
            panel.measure(f"alpha_{a}", Ha, kmeans_labels(Ha, k, seed=cfg.seed_metric),
                          template_masks=masks, reference_labels=ref_labels, heldout=False)

    # THE PANEL EXISTS TO COMPARE THE TWO ROUTES, AND IT WAS NOT DOING SO.
    # live38 measured 11 metrics for the bottom-up leaves and exactly ONE for the
    # top-down route — kappa, which is inter-annotator agreement on the gold set,
    # not a property of a partition at all. So the headline claim of the whole
    # method ("two routes, one uniform harness") was never actually delivered:
    # nothing in the panel put the two schemes on a common axis.
    #
    # The top-down labels are a partition of the same 49,999 rows in the same
    # representation, so the representation-neutral metrics apply directly.
    # `compute_stability`/`heldout` are deliberately OFF: replay-ARI and held-out
    # reproduction ask "does re-running the CLUSTERING land in the same place",
    # which is undefined for labels a classifier assigned from a taxonomy. Its
    # analogue is the classifier's own cross-validated accuracy, reported in the
    # top-down report. Printing a blank is honest; printing a zero is not.
    for subject, key, artifact, col in (
        ("topdown_l1", "topdown_preds", "topdown_labels", "l1_pred"),
        ("topdown_l2", "topdown_sub", "topdown_l2_labels", "td_l2"),
    ):
        vals = deps.recover(key, artifact, rebuild=lambda d, c=col: d[c].to_numpy())
        if vals is None or len(vals) != len(H):
            deps.emit(f"  {subject} not measurable in the panel — the two routes "
                      f"will not be comparable in this run")
            continue
        codes = pd.factorize(pd.Series(vals).astype(str))[0]
        panel.measure(subject, H, codes, template_masks=masks,
                      reference_labels=ref_labels,
                      compute_stability=False, heldout=False)

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
        "gates": _observe(deps, "p9", {"metrics_panel": deps.load("metrics_panel")}),
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
    # Keys come back as strings from JSON; the column is keyed by int leaf id.
    names.update({int(k): v for k, v in
                  (deps.recover("leaf_relabels", "leaf_relabels", default={}) or {}).items()})

    # EVERY DELIVERED LEAF MUST HAVE A NAME. `p7_all_leaves_named` is blocking
    # but runs in p7, and p8 governance then changes the partition — 6 splits and
    # 2 isolates took live38 from 29 leaves to 36, and the 7 new ones shipped
    # nameless in 4,931 rows (9.9% of the table) while that gate read PASSED.
    # p8 now names what it creates; this checks the partition actually delivered,
    # so any other cause is caught too.
    delivered = sorted({int(v) for v in np.unique(deps.leaf_labels_final())})
    unnamed = [i for i in delivered if not str(names.get(i, "")).strip()]
    named_gate = deps.gate(
        "p10_delivered_leaves_named", "p10",
        # AN EMPTY PARTITION IS NOT A NAMED PARTITION. `not unnamed` is true both
        # when every delivered leaf has a name and when there are no delivered
        # leaves at all — and the second is a total clustering failure that would
        # read here as "all 0 delivered leaves carry a name". Requiring at least
        # one leaf costs nothing and removes the false green.
        passed=bool(delivered) and not unnamed,
        observed={"n_leaves_delivered": len(delivered), "n_unnamed": len(unnamed),
                  "unnamed_leaf_ids": unnamed[:20]},
        threshold={"unnamed_allowed": 0},
        message=("交付分区里一个叶都没有 —— 这不是「全部已命名」, 而是没有可命名的东西"
                 if not delivered else
                 f"all {len(delivered)} delivered leaves carry a name" if not unnamed else
                 f"{len(unnamed)} of {len(delivered)} DELIVERED leaves have no name "
                 f"({unnamed[:8]}) — those rows ship with an empty name column"),
        remediation=("A leaf reaches the delivered table without a name when a phase "
                     "changes the partition after p7 named it. Name it where it is "
                     "created; re-asserting the gate later only reports the loss."),
    )

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
        # Name and definition, so the two routes read alike: the bottom-up leaf
        # ships `bu_leaf_name` / `bu_user_need` and a bare `td_l1` code cannot be
        # compared against them by anyone who does not have the taxonomy open.
        try:
            nodes = {n.code: n for n in deps.taxonomy().nodes}
            hit = sum(1 for c in td if str(c) in nodes)
            if hit >= 0.5 * len(td):
                out["td_l1_name"] = [getattr(nodes.get(str(c)), "name", "") for c in td]
                out["td_user_need"] = [getattr(nodes.get(str(c)), "definition", "") for c in td]
            else:
                # Emitting the columns anyway would ship blanks that read as
                # missing data rather than as a taxonomy that does not describe
                # the labels the classifier is producing. Say which it is.
                deps.emit(f"  ⚠ 仅 {hit}/{len(td)} 个 td_l1 取值能在类目表中找到 — "
                          "省略 td_l1_name/td_user_need 两列 (分类器与类目表不同源)")
        except Exception as exc:  # noqa: BLE001
            deps.emit(f"  td name columns unavailable: {type(exc).__name__}")
    conf = deps.recover("topdown_confidence", "topdown_labels",
                        rebuild=lambda d: d["confidence"].to_numpy())
    if conf is not None:
        out["td_confidence"] = np.round(conf, 5)
    tdm = deps.recover("topdown_margin", "topdown_labels",
                       rebuild=lambda d: d["margin"].to_numpy())
    if tdm is not None:
        out["td_margin"] = np.round(tdm, 5)
        # Same threshold as the bottom-up flag, so "ambiguous" means one thing.
        out["td_ambiguous"] = tdm < cfg.deployment.margin_threshold
    dby = deps.recover("topdown_decided_by", "topdown_labels",
                       rebuild=lambda d: d["decided_by"].to_numpy())
    if dby is not None:
        out["td_decided_by"] = [str(x) for x in dby]
    sub = deps.recover(
        "topdown_sub", "topdown_l2_labels", rebuild=lambda d: d["td_l2"].to_numpy()
    )
    if sub is not None:
        out["td_l2"] = sub
    msi = deps.recover("minority_sub_intent", "minority_sub_intent")
    if msi is not None:
        # Intents inside a minority-language family, resolved in a script-appropriate
        # space. A column rather than a leaf, because it is not a centroid region of
        # the space the classifier deploys — see ops/language.minority_sub_intents.
        out["minority_sub_intent"] = [str(x) for x in msi]
    for c in cfg.data.reference_label_columns:
        if c in df.columns:
            out[f"ref_{c}"] = df[c]
    out["run_id"] = deps.run_id
    out["generation"] = deps.store.generation

    # Neither route is the answer key, so the useful artifact is not a score for
    # each but a map of where they concur. Every bottom-up family gets its
    # top-down composition and a concentration figure: a family that maps almost
    # entirely onto one L1 is a place the two methodologies independently found
    # the same intent, and a family that splays across five is a real
    # disagreement worth a human's attention — in either direction, since the
    # top-down class may be the one that is too coarse.
    # AND IT MUST BE RUN AT MATCHED GRANULARITY, OR THE VERDICT IS ARITHMETIC.
    #
    # This compared 7 families against 25 top-down classes and reported "routes
    # disagree" on every row — which a 7-vs-25 comparison forces regardless of
    # what either route found. Measured on live40 at the LEAF layer instead, where
    # the cardinalities match (25 vs 25): AMI rises 0.5395 -> 0.6175, median
    # single-intent concentration 39.5% -> 80.3%, and 19 of 25 leaves are majority
    # one top-down intent against 1 of 7 families. The two routes agree, at the
    # layer that carries the intents.
    #
    # This also resolves the run's loudest open question. live40 concluded its
    # domain prior of 15-25 families was wrong because K came out 7. The prior was
    # right about how many intent classes the corpus has and wrong about which
    # LAYER carries them: the delivered leaf layer has 25, inside [15,25], and the
    # top-down taxonomy independently produced 25 L1 intents.
    from sklearn.metrics import adjusted_mutual_info_score

    if "td_l1" in out.columns:
        _levels = [("bu_family_final", "family"), ("bu_leaf", "leaf")]
        _by_level = {}
        for col, level in _levels:
            if col not in out.columns:
                continue
            sub = out.dropna(subset=["td_l1", col])
            if sub.empty:
                continue
            td_codes = pd.factorize(sub["td_l1"].astype(str))[0]
            conc = sub.groupby(col)["td_l1"].apply(
                lambda g: float(g.astype(str).value_counts().iloc[0] / len(g)))
            _by_level[level] = {
                "n_clusters": int(sub[col].nunique()),
                "n_td_classes": int(sub["td_l1"].nunique()),
                "ami": round(float(adjusted_mutual_info_score(td_codes, sub[col].to_numpy())), 4),
                "median_dominant_share": round(float(conc.median()), 4),
                "n_majority_one_intent": int((conc >= 0.5).sum()),
            }
        # AND WHILE BOTH LABEL SETS ARE IN HAND, VALIDATE THE K LOCATOR'S REFERENCE.
        #
        # K is located by AMI against the trusted phrasing groups, and the only
        # test those groups ever face is `validate_group_cohesion` — mean pairwise
        # cosine vs random, computed in the same embedding they later judge. That
        # gate measures TOPICAL tightness, and an intent group spans topics by
        # construction, so it ranks groups almost backwards from what matters:
        # on live40, Spearman(lift, purity) = -0.60 across the six seeded groups.
        #
        # The top-down taxonomy is an independent yardstick built by a different
        # methodology on the same corpus. If a phrasing group is one intent, its
        # members concentrate on one L1 class. That is the validation, and it can
        # only be run here, after p2 has produced labels.
        try:
            from ...ops.templates import group_masks
            from ...records import TemplateGroup

            _tg = deps.load("template_groups") if deps.has("template_groups") else {}
            _groups = [TemplateGroup.model_validate(g) for g in (_tg.get("groups") or [])]
            if _groups and "td_l1" in out.columns:
                _trusted = set(group_masks(_groups, deps.df,
                                           text_col=cfg.data.text_column,
                                           trusted_only=True))
                _all = group_masks(_groups, deps.df, text_col=cfg.data.text_column)
                _td = out["td_l1"].astype(str).to_numpy()
                _chance = float(pd.Series(_td).value_counts().iloc[0] / len(_td))
                _rows = []
                for _name, _m in _all.items():
                    _m = np.asarray(_m)
                    if _m.sum() < 10 or _m.shape[0] != len(_td):
                        continue
                    _vc = pd.Series(_td[_m]).value_counts()
                    _share = float(_vc.iloc[0] / _m.sum())
                    _rows.append({
                        "group": _name, "n": int(_m.sum()),
                        "votes_in_k_locator": _name in _trusted,
                        "dominant_intent": str(_vc.index[0]),
                        "single_intent_share": round(_share, 4),
                        "lift_over_chance": round(_share / _chance, 2) if _chance else None,
                        "n_intents_to_cover_90pct": int(
                            ((_vc / _m.sum()).cumsum() < 0.9).sum() + 1),
                    })
                if _rows:
                    _v = [r["single_intent_share"] for r in _rows if r["votes_in_k_locator"]]
                    deps.store.put_json("locator_reference_validation", {
                        "chance_baseline": round(_chance, 4),
                        "median_purity_of_voting_groups": round(float(np.median(_v)), 4) if _v else None,
                        "groups": sorted(_rows, key=lambda r: -r["single_intent_share"]),
                        "what_this_tests": (
                            "whether the phrasing groups the K locator scores AMI "
                            "against are really same-intent groups, judged against a "
                            "taxonomy built by a different methodology. This is the "
                            "check `validate_group_cohesion` cannot make: it measures "
                            "topical tightness, and an intent group spans topics."),
                    }, producer="p10", summary="are the K locator's reference groups one intent?")
                    if _v:
                        deps.emit(f"  K 定位参照校验: 参与投票的措辞群单一意图占比中位数 "
                                  f"{np.median(_v):.1%} (随机基线 {_chance:.1%})")
        except Exception as exc:                                  # noqa: BLE001
            deps.emit(f"  locator reference validation skipped ({type(exc).__name__}: {exc})")

        if _by_level:
            deps.store.put_json("route_concordance", {
                "by_level": _by_level,
                "note": ("compare at MATCHED cardinality. A 7-family vs 25-class "
                         "comparison forces disagreement arithmetically; the leaf "
                         "layer is where the two routes are commensurable."),
            }, producer="p10", summary="both routes, compared at each bottom-up level")
            for level, m in _by_level.items():
                deps.emit(f"  路线一致性 ({level}, {m['n_clusters']} 簇 vs "
                          f"{m['n_td_classes']} 类): AMI={m['ami']}, "
                          f"{m['n_majority_one_intent']}/{m['n_clusters']} 单一意图占多数")

        rows = []
        for fam, grp in out.groupby("bu_family_final"):
            comp = grp["td_l1"].astype(str).value_counts()
            share = float(comp.iloc[0] / len(grp))
            p_ = (comp / len(grp)).to_numpy()
            rows.append({
                "bu_family_final": int(fam),
                "n": int(len(grp)),
                "td_dominant": str(comp.index[0]),
                "td_dominant_share": round(share, 4),
                "td_classes_touched": int((comp > 0).sum()),
                # exp(H): the same effective-count formula the fragmentation
                # metric uses, so the two numbers are read the same way.
                "td_effective_classes": round(float(np.exp(-(p_ * np.log(p_)).sum())), 3),
                "verdict": ("routes agree" if share >= 0.80 else
                            "partial overlap" if share >= 0.50 else "routes disagree"),
            })
        crosswalk = pd.DataFrame(rows).sort_values("td_dominant_share")
        deps.store.put_table("route_crosswalk", crosswalk, fmt="csv", producer="p10",
                             summary="bottom-up family × top-down L1 concordance")
        agree = int((crosswalk["verdict"] == "routes agree").sum())
        deps.emit(f"  路线对照: {agree}/{len(crosswalk)} 个家族与自上而下 L1 高度一致 "
                  f"(≥80% 落在同一类目)")

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
        # `deps.gate()` RETURNS a GateResult and registers nothing — the node has
        # to put it in state. Discarding it left `p10_delivered_leaves_named`,
        # which is BLOCKING, reaching the log and nothing else: a blocking gate
        # that could never block, absent from run_summary and from every report.
        "gates": {named_gate.name: named_gate},
        "completed_phases": ["p10"],
        "events": [
            f"P10: delivered {len(out)} rows; model {clf.size_bytes() / 1024:.0f} KB; "
            f"{routing['ambiguous_rate'] * 100:.1f}% route to fallback"
        ],
    }


def _p11_fast(state: PipelineState, deps: Deps) -> dict[str, Any]:
    """Write the three fast-mode deliverables.

    Deliberately does NOT run the narrative writer, the interpreter or the
    pre-delivery auditor — `mode="fast"` already turned all three off in config,
    and reaching them from here would silently re-enable what `fast_skipped`
    tells the reader was skipped. The figures ARE still built: they are generated
    by Python from artifacts, cost no model call, and both documents reference
    them.
    """
    from ...report import fast_deliver as fast
    from ...report.builder import build_figures_only

    refs: dict[str, Any] = {}
    try:
        refs.update(build_figures_only(state, deps))
    except Exception as exc:  # noqa: BLE001
        deps.emit(f"  figures skipped ({type(exc).__name__}: {exc})")
    n_figures = len(refs)

    deps.emit("  fast 模式交付: 核心参考文档 (完整中间产物保留)")
    refs["report_fast_topdown"] = deps.store.put_markdown(
        f"{deps.cfg.domain.key}_自上而下_意图体系完整定义",
        fast.build_topdown(state, deps), producer="p11",
        summary="fast 模式交付物: 类目定义、裁定规则、金标准与分类器")
    refs["report_fast_bottomup"] = deps.store.put_markdown(
        f"{deps.cfg.domain.key}_自下而上_聚类树完整定义",
        fast.build_bottomup(state, deps), producer="p11",
        summary="fast 模式交付物: 已交付的家族与叶, 逐叶定义")
    try:
        wb = fast.build_workbook(state, deps)
        refs["workbook"] = deps.store.register_file(
            wb.stem, wb, "table", producer="p11",
            summary="fast 模式交付物: 全量逐行标注 + 定义与分布 sheet")
    except Exception as exc:  # noqa: BLE001
        deps.emit(f"  ⚠ 工作簿生成失败 ({type(exc).__name__}: {exc}) — "
                  "labels_full.csv 仍在运行目录中")

    _drift_document(state, deps, refs)

    # COUNT WHAT SHIPPED, INCLUDING THE DRIFT DOCUMENT. The set is 3 on a
    # single-snapshot run and 4 when there are snapshots to compare, so the
    # expected total is derived rather than written as a literal — a hardcoded 3
    # was already wrong the moment p10b started shipping a fourth.
    _expected = ["report_fast_topdown", "report_fast_bottomup", "workbook"]
    if deps.has("drift_analysis"):
        _expected.append("report_drift")
    n_delivered = sum(1 for k in _expected if k in refs)
    return {
        "phase": "p12",
        "artifacts": refs,
        "completed_phases": ["p11"],
        # COUNT WHAT SHIPPED, not what was intended. This said "3 deliverables"
        # unconditionally and derived the figure count as `len(refs) - 3`, so a
        # failed workbook was still reported as three, and a run that lost both
        # the workbook and the figures printed "-1 figures".
        "events": [f"P11 (fast): {n_delivered}/{len(_expected)} deliverables + {n_figures} figures "
                   f"written to {deps.store.gen_dir}; "
                   f"skipped {', '.join(deps.cfg.fast_skipped)}"
                   + ("" if n_delivered == len(_expected) else
                      "  ⚠ a deliverable did not build — see the warnings above")],
    }


def _drift_document(state: PipelineState, deps: Deps, refs: dict[str, Any]) -> None:
    """Add the snapshot-comparison document, when there is one to add.

    Shipped in BOTH modes. It is generated entirely from `drift_analysis.json`, so
    it costs no model call and survives `mode="fast"`, which disables every
    agent-written deliverable. A multi-snapshot run whose whole point is the
    comparison must not lose the comparison to the cheap mode.
    """
    if not deps.has("drift_analysis"):
        return
    try:
        from ...report.zh_drift import build as build_drift

        refs["report_drift"] = deps.store.put_markdown(
            "快照对比_漂移分析", build_drift(state, deps), producer="p11",
            summary="两个快照在同一套标签下的差异")
    except Exception as exc:  # noqa: BLE001
        deps.emit(f"  ⚠ drift document not written ({type(exc).__name__}: {exc}) — "
                  "drift_analysis.json still holds every number")


def _family_display(deps: Deps, col: "pd.Series") -> "pd.Series":
    """`11` → `#11 观看直播`, falling back to `#11` when naming is unavailable.

    Never raises: a drift table with bare ids is worse than one with names, but a
    p10b that dies on a missing naming artifact is worse than both.
    """
    try:
        import numpy as np

        from qmine.report._shape import family_names

        naming = deps.load("tree_naming") if deps.has("tree_naming") else {}
        fam = (deps.load("leaf_family_final") if deps.has("leaf_family_final")
               else deps.load("leaf_family"))
        leaves = (deps.load("leaf_labels_final") if deps.has("leaf_labels_final")
                  else deps.load("leaf_labels"))
        sizes = np.bincount(np.asarray(leaves), minlength=len(fam))
        names = family_names(naming, fam, sizes)
    except Exception as exc:  # noqa: BLE001
        deps.emit(f"  ⚠ family names unavailable ({type(exc).__name__}) — "
                  "the family drift table will show ids only")
        names = {}
    return col.map(lambda i: f"#{i} {names[int(i)]}" if int(i) in names else f"#{i}")


def p10b_drift(state: PipelineState, deps: Deps) -> dict[str, Any]:
    """Compare the snapshots — ONLY when there is more than one.

    A no-op on a single-snapshot corpus, which is every run this pipeline made
    before multi-input existed. It reads the DELIVERED labels, so it must run
    after p10, and it writes an artifact p11 turns into a document.

    Why this is a phase and not a script: the comparison is only valid inside one
    run. Two runs of this pipeline on the same 10,000 rows shared 0 of 35 class
    codes, so anything that compares two runs' labels is comparing noise. Putting
    the measurement where the shared taxonomy lives is what makes it mean
    anything.
    """
    from ...ops import drift

    cfg = deps.cfg
    snap_col = "snapshot"
    df = deps.df
    if snap_col not in getattr(df, "columns", []):
        return {"phase": "p11", "completed_phases": ["p10b"],
                "events": ["P10b: single snapshot — no drift analysis"]}
    tags = sorted(df[snap_col].astype(str).unique())
    if len(tags) < 2:
        return {"phase": "p11", "completed_phases": ["p10b"],
                "events": [f"P10b: one snapshot ({tags}) — nothing to compare"]}

    deps.emit(f"P10b drift — comparing {len(tags)} snapshots: {', '.join(tags)}")
    work = pd.DataFrame({snap_col: df[snap_col].astype(str),
                         "query": df[cfg.data.text_column].astype(str),
                         "weight": df["weight"] if "weight" in df.columns else 1.0})

    # THE DELIVERED LABELS, JOINED BY POSITION. `labels_full` is written in corpus
    # order (verified 20,000/20,000 on a real run), and a join on query TEXT fans
    # out on repeated queries — these corpora contain them.
    labels: dict[str, Any] = {}
    try:
        lab = pd.read_csv(Path(deps.store.gen_dir) / "labels_full.csv")
        if len(lab) == len(work):
            for col in ("td_l1", "bu_leaf_name", "bu_family_final"):
                if col in lab.columns:
                    work[col] = lab[col].values
                    labels[col] = col
            # FAMILIES ARE KEYED BY ID, AND AN ID IS NOT A FINDING. The first
            # real render showed `11` and `15` as the two biggest movers, which
            # no reader can act on. Names come from `_shape.family_names`, which
            # joins through LEAF MEMBERSHIP — never by integer id, which
            # mismatched 19 of 19 on live38. The id is kept as the primary key.
            if "bu_family_final" in work.columns:
                work["bu_family_final"] = _family_display(deps, work["bu_family_final"])
        else:
            deps.emit(f"  ⚠ labels_full has {len(lab):,} rows against {len(work):,} in the "
                      f"frame — not joining by position; drift is corpus-level only")
    except Exception as exc:  # noqa: BLE001
        deps.emit(f"  ⚠ delivered labels unreadable ({type(exc).__name__}) — "
                  "drift is corpus-level only")

    out: dict[str, Any] = {
        "snapshots": tags,
        "inventory": drift.snapshot_inventory(work, snap_col, "query", "weight"),
        "query_churn": drift.query_churn(work, snap_col, "query", "weight"),
        "by_label": {}, "purity": {},
        "note": ("Shares are WITHIN-SNAPSHOT. The snapshots differ in total weight, "
                 "so raw counts would report every class as declining. Both periods "
                 "were labelled by ONE taxonomy and ONE tree in this run, which is "
                 "what excludes 'the taxonomy changed' as an explanation."),
    }
    for col in labels:
        out["by_label"][col] = drift.label_drift(work, col, snap_col, "weight",
                                                 text_col="query")
        out["purity"][col] = drift.snapshot_purity(work, col, snap_col)

    ref = deps.store.put_json("drift_analysis", out, producer="p10b",
                              summary=f"{len(tags)} snapshots compared over "
                                      f"{len(labels)} label axes")
    n_pure = sum(v.get("n_single_snapshot", 0) for v in out["purity"].values())
    gate = deps.gate(
        "p10b_snapshots_share_one_frame", "p10b",
        # A group sitting almost entirely in one snapshot was separated, not
        # compared. A few are ordinary (a real event only one period saw); many
        # mean the pooled frame is a fiction and no share here means what it looks
        # like. Advisory, because only reading them can tell which.
        passed=n_pure == 0,
        observed={"n_single_snapshot_groups": n_pure,
                  **{f"{k}_purity": v for k, v in out["purity"].items()}},
        threshold={"n_single_snapshot_groups": 0},
        message=("every labelled group spans both snapshots" if n_pure == 0 else
                 f"{n_pure} group(s) sit almost entirely in ONE snapshot — read them "
                 f"before trusting any share below; they were separated, not compared"),
        remediation="A single-snapshot group is either a genuine period-specific event "
                    "or a sign the two extracts are not the same population. Open the "
                    "queries in it and decide which; the numbers cannot.",
        warn_only=True,
    )
    return {"phase": "p11", "artifacts": {"drift_analysis": ref},
            "gates": {gate.name: gate}, "completed_phases": ["p10b"],
            "events": [f"P10b: {len(tags)} snapshots compared; "
                       f"{n_pure} single-snapshot group(s)"]}


def p11_report(state: PipelineState, deps: Deps) -> dict[str, Any]:
    """Write the reports and the executed notebook."""
    from ...report.builder import build_all_reports

    from ...ops.findings import recheck_run

    deps.emit("P11 reporting — assembling deliverables")
    # BEFORE the reports are written, so they print the current ledger. Every
    # open finding's assertion is re-evaluated against the artifacts that are
    # actually about to ship: one that has been fixed closes itself here, and
    # one that has not is carried forward rather than quietly aging out.
    try:
        recheck_run(deps)
    except Exception as exc:  # noqa: BLE001
        deps.emit(f"  findings re-check skipped ({type(exc).__name__}: {exc})")
    # INSTALL THE TRANSLATOR BEFORE ANY REPORT RENDERS.
    #
    # `prose()` is called from report modules with no access to the registry, so
    # the translator is configured here rather than threaded through 20 call
    # sites. Curated `PROSE_ZH` wording still wins; this only reaches strings the
    # mapping cannot cover — a newly authored rationale nobody has noticed yet,
    # and the 22 `deps.gate()` f-strings that no fixed prefix can ever match.
    # Every result is checked for altered numbers and identifiers before use, and
    # anything suspect keeps the English.
    if getattr(deps.cfg, "translate_prose", True) and not deps.cfg.offline:
        from ...report.i18n import set_translator
        from ...report.translate import registry_translator

        try:
            set_translator(registry_translator(deps.registry))
        except Exception as exc:  # noqa: BLE001
            deps.emit(f"  translator unavailable ({type(exc).__name__}); prose stays English")

    if getattr(deps.cfg, "mode", "full") == "fast":
        # THREE DOCUMENTS INSTEAD OF THIRTEEN — and every artifact still written.
        #
        # The documents a full run ships exist to ARGUE: each shows the
        # measurement behind a decision and why the alternative lost. A fast run
        # removed the layer that produces that argument, so shipping the same
        # the same set would be shipping those arguments with the evidence cut out
        # of them. These three REFER instead — the classes, the tree, and every
        # labelled row — and each carries the generated disclosure banner naming
        # what was skipped. The figures and the store are untouched.
        return _p11_fast(state, deps)

    refs = build_all_reports(state, deps)
    _drift_document(state, deps, refs)

    # THE DOCUMENT A READER OPENS FIRST, AND THE ONLY ONE NOT ASSEMBLED BY PYTHON.
    #
    # Everything above is generated section by section, which is why it is correct
    # and why it reads like a changelog: the order is the order the code ran, and
    # every defect ever fixed added its caveat where it happened rather than where
    # a reader needs it. A template cannot write a through-line because it does not
    # know what the run turned out to be about. So this one is written by an agent
    # — outline and prose both — and checked mechanically instead: numbers against
    # a per-section fact sheet, omissions against a must-cover list derived from
    # the run's own warnings. It runs HERE, after the figures exist and before the
    # auditor, so the auditor reads it like any other deliverable.
    narrative = {"ran": False, "skipped": "disabled"}
    if getattr(deps.cfg, "final_report", True):
        from ...agents.narrate import narrate

        deps.emit("  最终报告 — agent 撰写, 机器校验")
        try:
            narrative = narrate(state, deps)
        except Exception as exc:  # noqa: BLE001
            deps.emit(f"  final report skipped ({type(exc).__name__}: {exc})")
            narrative = {"ran": False, "skipped": f"{type(exc).__name__}: {exc}"}
        if narrative.get("ran") and narrative.get("markdown"):
            refs["report_final"] = deps.store.put_markdown(
                FINAL_REPORT_STEM, narrative["markdown"], producer="p11",
                summary=(f"agent-written: {narrative.get('n_sections_ok')}/"
                         f"{narrative.get('n_sections')} sections verified, "
                         f"{narrative.get('n_musts') - narrative.get('n_musts_missing')}"
                         f"/{narrative.get('n_musts')} required points covered"))
        refs["final_report_meta"] = deps.store.put_json(
            "final_report_meta", {k: v for k, v in narrative.items()
                                  if k != "markdown"},
            producer="p11", summary="how the agent-written report was verified")

    # THE LAST THING BETWEEN THE RUN AND A READER.
    #
    # Every warning this pipeline raises is otherwise read by a human, later, if
    # at all. This is the only step that holds the gate ledger, the findings
    # ledger, the artifacts and the finished documents in one place and asks
    # whether any of those warnings left a defect in what is about to be handed
    # over — and it is the only agent allowed to fix one. Its authority is
    # bounded to an anchored replacement whose numbers must come from the
    # artifact it cites; `ops/edits.py` holds each rule and the failure behind it.
    audit: dict[str, Any] = {"ran": False, "skipped": "disabled"}
    if getattr(deps.cfg, "delivery_audit", True):
        from ...agents.audit_delivery import audit_deliverables

        try:
            audit = audit_deliverables(state, deps)
        except Exception as exc:  # noqa: BLE001
            deps.emit(f"  pre-delivery audit skipped ({type(exc).__name__}: {exc})")
            audit = {"ran": False, "skipped": f"{type(exc).__name__}: {exc}"}

    # An edited file no longer matches the hash the index recorded when it was
    # written. Re-registering keeps provenance honest: the manifest must describe
    # the bytes that actually ship, not the draft the auditor corrected.
    for name in (audit.get("files_changed") or []):
        stem = Path(name).stem
        refs[f"report_{stem}"] = deps.store.put_markdown(
            stem, (Path(deps.store.gen_dir) / name).read_text(encoding="utf-8"),
            producer="p11", summary="re-registered after the pre-delivery audit")

    # Stored as an artifact rather than pushed into state: `PipelineState` is a
    # TypedDict and an undeclared key is an update error, and the audit record is
    # something a later generation and `verify_run.py` both want on disk anyway.
    refs["delivery_audit"] = deps.store.put_json(
        "delivery_audit", audit, producer="p11",
        summary=(f"{audit.get('n_applied', 0)} edits applied, "
                 f"{audit.get('n_refused', 0)} refused" if audit.get("ran")
                 else f"not run: {audit.get('skipped', 'disabled')}"))

    try:
        from ...report.zh_audit import build as build_audit

        refs["report_audit"] = deps.store.put_markdown(
            "交付前审核报告", build_audit(audit, state, deps),
            producer="p11", summary="pre-delivery audit: every edit, refusal and dismissal")
    except Exception as exc:  # noqa: BLE001
        deps.emit(f"  audit report not written ({type(exc).__name__}: {exc})")

    return {
        "phase": "p12",
        "artifacts": refs,
        "completed_phases": ["p11"],
        "events": [f"P11: {len(refs)} deliverables written to {deps.store.gen_dir}"
                   + (f"; audit applied {audit.get('n_applied', 0)} edit(s), "
                      f"refused {audit.get('n_refused', 0)}" if audit.get("ran") else
                      "; deliverables NOT audited")],
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
            # SAY SO. This was recorded in `maintenance.json` and nowhere else,
            # so on live44 the maintainer failed every attempt — 44 minutes,
            # zero output tokens — and the only thing an operator saw was
            # "✔ p12_maintain completed". The mechanical half of the phase had
            # genuinely succeeded, which is why the phase completes; the missing
            # half must still be visible without opening an artifact.
            drift = {"comparison": comparison, "analyst_error": str(exc)[:200]}
            deps.emit(f"  ⚠ maintenance analyst unavailable ({type(exc).__name__}) — "
                      "the baseline, novelty sentinel and drift comparison are "
                      "recorded, but nothing interprets them this run")
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
                    "distance to the nearest delivered leaf centroid, bottom 1% by cosine. "
                    "NOT a density method: HDBSCAN is instantiated only inside the Phase 4 "
                    "probe and nothing downstream consumes it. This said 'the density method "
                    "that lost the Phase 4 bake-off is kept for exactly this job', which named "
                    "a bake-off the spec has retracted, for a method that is not the one running"
                ),
                "method": "max-centroid cosine, 1st percentile",
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
