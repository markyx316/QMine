"""Phases 3-6 — representation, algorithm, granularity, hierarchy."""

from __future__ import annotations

from typing import Any


from ...determinism import hash_texts
from ...ops.stats import proportion_gate
from ...ops.cluster import (
    algorithm_battery,
    build_hierarchy,
    deep_aligned_estimate,
    heldout_reproduction,
    k_sweep,
    refine,
    triangulate_k,
)
from ...ops.represent import (
    alpha_sweep,
    build_sparse,
    encode_corpus,
    encoder_bakeoff,
    hybrid,
    load_encoder,
    surface_vote_share,
)
from ...state import PipelineState
from ..deps import Deps


def p3_represent(state: PipelineState, deps: Deps) -> dict[str, Any]:
    """Encoder bake-off, sparse block, and the alpha sweep.

    Three choices, all made on decisive metrics.  Silhouette is computed at every
    step and given a vote at none of them — its disagreement with the winner is
    recorded explicitly, because "the metric everyone reports would have chosen
    differently" is exactly the kind of thing a reader deserves to see.
    """
    cfg = deps.cfg
    df = deps.df
    texts = df[cfg.data.text_column].astype(str).tolist()
    masks = deps.template_masks()
    ref_labels = None
    for c in cfg.data.reference_label_columns:
        if c in df.columns:
            ref_labels = df[c].astype(str).to_numpy()
            break

    # --- 3a: which encoder? ------------------------------------------------
    deps.emit(f"P3a encoder bake-off — {len(cfg.domain.embedding_candidates)} candidates")
    candidates = list(cfg.domain.embedding_candidates)
    if cfg.offline:
        candidates = ["hashing-256"]
    bake = encoder_bakeoff(
        texts, candidates, k=cfg.representation.alpha_sweep_k, template_masks=masks,
        subsample=cfg.representation.bakeoff_subsample, seeds=tuple(cfg.seed_replay),
        reference_labels=ref_labels, offline=cfg.offline,
        cache_folder=str(deps.store.root.parent / ".hf") if not cfg.offline else None,
        instruction=cfg.domain.instruction_prefix,
    )
    chosen_encoder = bake["chosen_encoder"]
    deps.emit(f"  chose {chosen_encoder} on stability; silhouette would have chosen "
              f"{bake['silhouette_would_have_chosen']}")

    encoder = load_encoder(chosen_encoder, offline=cfg.offline,
                           cache_folder=str(deps.store.root.parent / ".hf") if not cfg.offline else None)
    input_hash = hash_texts(texts)
    dense, cached = deps.store.memoize(
        "encode", {"encoder": chosen_encoder, "instruction": cfg.domain.instruction_prefix},
        input_hash, lambda: encode_corpus(encoder, texts, instruction=cfg.domain.instruction_prefix),
    )
    deps.emit(f"  encoded {dense.shape} ({'cache hit' if cached else 'computed'})")
    emb_ref = deps.store.put_matrix("emb_base", dense, producer="p3a",
                                    summary=f"{chosen_encoder}, L2-normalised")
    deps.cache_put("emb_base", dense)

    # --- 3b: sparse block --------------------------------------------------
    deps.emit("P3b sparse block — char TF-IDF → SVD")
    sp = build_sparse(
        texts, analyzer="char", ngram_range=tuple(cfg.domain.char_ngram_range),
        min_df=cfg.representation.tfidf_min_df, svd_dims=cfg.representation.svd_dims,
        seed=cfg.seed_metric, tokenizer=cfg.domain.tokenizer,
    )
    svd_ref = deps.store.put_matrix("emb_svd_char", sp["svd_block"], producer="p3b",
                                    summary=f"char {cfg.domain.char_ngram_range} TF-IDF → "
                                            f"{sp['n_components']}d SVD, evr {sp['explained_variance']:.3f}")
    deps.cache_put("emb_svd_char", sp["svd_block"])

    # --- 3c: alpha sweep ---------------------------------------------------
    deps.emit(f"P3c alpha sweep — {cfg.representation.alpha_grid}")
    sweep = alpha_sweep(
        dense, sp["svd_block"], alphas=cfg.representation.alpha_grid,
        k=cfg.representation.alpha_sweep_k, template_masks=masks,
        seeds=tuple(cfg.seed_replay), silhouette_sample=cfg.clustering.silhouette_sample,
        reference_labels=ref_labels,
    )
    alpha = sweep["chosen_alpha"]
    H = hybrid(dense, sp["svd_block"], alpha)
    hyb_ref = deps.store.put_matrix("emb_hybrid", H, producer="p3c",
                                    summary=f"alpha={alpha} (surface vote {surface_vote_share(alpha) * 100:.1f}%)")
    deps.cache_put("emb_hybrid", H)

    rep_ref = deps.store.put_json(
        "representation",
        {"bakeoff": bake, "sparse": {k: v for k, v in sp.items() if k in ("vocab_size", "explained_variance", "n_components")},
         "alpha_sweep": sweep,
         "alpha_algebra": {
             "formula": "cos(H,H') = (cos_semantic + a^2 * cos_surface) / (1 + a^2)",
             "chosen_alpha": alpha,
             "surface_vote_share": round(surface_vote_share(alpha), 4),
             "note": "the phrasing block votes with weight alpha SQUARED, not alpha",
         }},
        producer="p3", summary=f"encoder {chosen_encoder}, alpha {alpha}",
    )

    decisions = [
        deps.decision(
            "p3a", "Which base encoder?", chosen_encoder,
            f"Highest replay stability on this corpus's own clustering task. "
            f"Silhouette would have chosen {bake['silhouette_would_have_chosen']} — recorded and overruled.",
            evidence={"rows": bake["rows"]}, decisive_metrics=["stability_ari", "template_fragmentation"],
            rejected=[{"option": r["encoder"], "why_rejected": r.get("error", "lost on stability"),
                       "metrics": {k: r.get(k) for k in ("stability_ari", "template_fragmentation", "silhouette")}}
                      for r in bake["rows"] if r.get("encoder") != chosen_encoder],
        ),
        deps.decision(
            "p3c", "How much weight should phrasing get?", f"alpha = {alpha}",
            f"Lowest template fragmentation with highest stability. At alpha={alpha} the phrasing "
            f"block controls {surface_vote_share(alpha) * 100:.1f}% of the cosine — a tie-breaker, "
            "not a co-equal signal.",
            evidence={"sweep": sweep["rows"]}, decisive_metrics=["template_fragmentation", "stability_ari"],
            rejected=[{"option": f"alpha={r['alpha']}", "why_rejected": "higher fragmentation or lower stability",
                       "metrics": {k: r[k] for k in ("template_fragmentation", "stability_ari", "silhouette")}}
                      for r in sweep["rows"] if r["alpha"] != alpha],
        ),
    ]
    events = [
        f"P3a: encoder {chosen_encoder} ({dense.shape[1]}d)",
        f"P3c: alpha {alpha} → phrasing controls {surface_vote_share(alpha) * 100:.1f}% of the cosine",
    ]
    if sweep["silhouette_disagrees"]:
        events.append(
            f"P3c: silhouette would have chosen alpha={sweep['silhouette_would_have_chosen']} "
            "— overruled by design (Principle 3)"
        )
    return {
        "phase": "p4",
        "artifacts": {"emb_base": emb_ref, "emb_svd_char": svd_ref, "emb_hybrid": hyb_ref,
                      "representation": rep_ref},
        "decisions": decisions,
        "chosen_alpha": alpha,
        "chosen_encoder": chosen_encoder,
        "completed_phases": ["p3"],
        "events": events,
    }


def p4_battery(state: PipelineState, deps: Deps) -> dict[str, Any]:
    """Run every clustering algorithm through one identical harness."""
    cfg = deps.cfg
    H = deps.embedding("emb_hybrid")
    deps.emit(f"P4 algorithm battery — {len(cfg.clustering.battery_k)} K values × 6 algorithms")
    result = algorithm_battery(
        H, ks=cfg.clustering.battery_k, seeds=tuple(cfg.seed_replay),
        silhouette_sample=cfg.clustering.silhouette_sample,
        include_hdbscan=not cfg.fast_mode,
    )
    result["note"] = (
        "the battery ranks algorithms on a capped sub-sample; the winner is then "
        "fitted on the full corpus in Phase 6"
    )
    verdict = result["verdict"]
    family = verdict["chosen_family"] or "kmeans"
    deps.emit(f"  winner: {verdict['chosen']} (stability {verdict['ranking'][0]['stability_ari'] if verdict['ranking'] else 'n/a'})")

    ref = deps.store.put_json("battery", result, producer="p4",
                              summary=f"{len(result['rows'])} configurations, winner {verdict['chosen']}")
    decision = deps.decision(
        "p4", "Which clustering algorithm?", family,
        "Highest replay stability under an identical measurement harness. On L2-normalised "
        "embeddings the points lie on a unit sphere where cosine neighbourhoods are close to "
        "isotropic — which is exactly KMeans' assumption, and why it tends to win here.",
        evidence={"ranking": verdict["ranking"]},
        decisive_metrics=["stability_ari"],
        rejected=[{"option": r["algorithm"], "why_rejected": "lower replay stability",
                   "metrics": {"stability_ari": r["stability_ari"], "silhouette": r["silhouette"]}}
                  for r in result["rows"] if r["algorithm"] != verdict["chosen"]][:12],
    )
    events = [f"P4: {verdict['chosen']} wins on stability"]
    if verdict["density_candidates_for_manual_review"]:
        d = verdict["density_candidates_for_manual_review"][0]
        events.append(
            f"P4: density methods held back for the novelty sentinel "
            f"(best HDBSCAN: {d['noise_rate'] * 100:.0f}% noise, {d['n_clusters']} clusters)"
        )
    return {
        "phase": "p5",
        "artifacts": {"battery": ref},
        "decisions": [decision],
        "chosen_algorithm": family,
        "completed_phases": ["p4"],
        "events": events,
    }


def p5_granularity(state: PipelineState, deps: Deps) -> dict[str, Any]:
    """Triangulate the family scale from three independent estimators."""
    cfg = deps.cfg
    H = deps.embedding("emb_hybrid")
    masks = deps.template_masks()
    deps.emit(f"P5 granularity — K sweep over {len(cfg.clustering.k_sweep)} values")

    # Full-effort unless this is an explicit smoke run: the cheap estimator was
    # measured at only 0.43 rank correlation with the full sweep on this corpus,
    # and K is inherited by every phase after this one.
    sweep = k_sweep(H, cfg.clustering.k_sweep, seeds=tuple(cfg.seed_replay),
                    silhouette_sample=cfg.clustering.silhouette_sample, template_masks=masks,
                    fast=cfg.fast_mode)
    expected_mid = int(sum(cfg.domain.expected_family_range) / 2)
    da = deep_aligned_estimate(H, expected_mid, multiplier=cfg.clustering.deep_aligned_multiplier,
                               seed=cfg.seed_metric)
    tri = triangulate_k(sweep, da, tuple(cfg.domain.expected_family_range))
    k = tri["chosen_family_k"]
    deps.emit(f"  stability peak K={k}; DeepAligned leaf estimate {da['k_estimate']}; "
              f"converged={tri['converged']}")

    ref = deps.store.put_json(
        "granularity", {"k_sweep": sweep, "deep_aligned": da, "triangulation": tri},
        producer="p5", summary=f"family K={k}",
    )
    decision = deps.decision(
        "p5", "How many families?", f"K = {k}",
        "Replay-stability peak, corroborated by an over-clustering survival estimate and a "
        "domain prior. Where the three disagree we take the stability peak and record the "
        "disagreement — an averaged K is defensible to nobody.",
        evidence=tri["estimates"], decisive_metrics=["stability_ari"],
        rejected=[{"option": f"K={r['k']}", "why_rejected": "lower replay stability",
                   "metrics": {"stability_ari": r["stability_ari"], "silhouette": r["silhouette"]}}
                  for r in sweep if r["k"] != k],
    )
    events = [f"P5: family K={k} (stability {max(r['stability_ari'] for r in sweep):.3f})"]
    if not tri["converged"]:
        events.append(f"P5: estimators did NOT converge — {tri['divergence_note']}")
    if tri["silhouette_disagrees"]:
        events.append(f"P5: silhouette peaks at K={tri['estimates']['silhouette_peak_k']} — advisory only")
    return {
        "phase": "p6",
        "artifacts": {"granularity": ref},
        "decisions": [decision],
        "family_k": k,
        "completed_phases": ["p5"],
        "events": events,
    }


def p6_hierarchy(state: PipelineState, deps: Deps) -> dict[str, Any]:
    """Build the two-level tree, refine it to convergence, then prove it reproduces.

    One stable global partition on top, a locally chosen split per family below.
    Asking a single global K to be both stable and fine-grained is the request
    that produced an undeliverable partition in the source project; splitting the
    question in two is the fix.
    """
    cfg = deps.cfg
    H = deps.embedding("emb_hybrid")
    k = state.get("family_k") or int(sum(cfg.domain.expected_family_range) / 2)
    deps.emit(f"P6 hierarchy — {k} families, then local leaf selection")

    tree = build_hierarchy(
        H, k, seed=cfg.seed_metric, min_leaf_size=cfg.clustering.min_leaf_size,
        min_leaf_fraction=cfg.clustering.min_leaf_fraction,
        max_leaves=cfg.clustering.max_leaves_per_family,
        family_min_size_for_split=cfg.clustering.family_min_size_for_split,
    )
    deps.emit(f"  built {tree['n_families']} families / {tree['n_leaves']} leaves")

    ref_out = refine(
        H, tree["leaf_labels"], tree["leaf_family"], rounds=cfg.clustering.refine_rounds,
        merge_cos=cfg.clustering.refine_merge_cos, move_tolerance=cfg.clustering.refine_move_tolerance,
        min_leaf_size=cfg.clustering.min_leaf_size, seed=cfg.seed_metric,
    )
    deps.emit(f"  refined to {ref_out['n_leaves']} leaves in {len(ref_out['history'])} rounds "
              f"(converged={ref_out['converged']})")

    hr = heldout_reproduction(H, ref_out["leaf_labels"], fraction=cfg.clustering.heldout_fraction,
                              seed=cfg.seed_metric)
    deps.emit(f"  held-out structure reproduction {hr['agreement']:.3f}")

    artifacts = {
        "leaf_labels": deps.store.put_matrix("leaf_labels", ref_out["leaf_labels"], producer="p6",
                                             summary=f"{ref_out['n_leaves']} leaves"),
        "leaf_family": deps.store.put_matrix("leaf_family", ref_out["leaf_family"], producer="p6",
                                             summary="leaf → family lookup table"),
        "leaf_centroids": deps.store.put_matrix("leaf_centroids", ref_out["leaf_centroids"], producer="p6",
                                                summary="the deployed model"),
        "family_labels": deps.store.put_matrix("family_labels", tree["family_labels"], producer="p6",
                                               summary="pre-refinement family assignment"),
        "hierarchy_meta": deps.store.put_json(
            "hierarchy_meta",
            {"n_families": tree["n_families"], "n_leaves": ref_out["n_leaves"],
             "leaves_per_family": tree["leaves_per_family"],
             "min_leaf_size_applied": tree["min_leaf_size_applied"],
             "refinement_history": ref_out["history"], "converged": ref_out["converged"],
             "heldout_reproduction": hr},
            producer="p6", summary=f"{tree['n_families']}→{ref_out['n_leaves']} leaves",
        ),
    }
    for name in ("leaf_labels", "leaf_family", "leaf_centroids"):
        deps.cache_put(name, ref_out[name if name != "leaf_centroids" else "leaf_centroids"])
    deps.cache_put("leaf_labels", ref_out["leaf_labels"])
    deps.cache_put("leaf_family", ref_out["leaf_family"])
    deps.cache_put("leaf_centroids", ref_out["leaf_centroids"])

    # Judged against a confidence interval, not a point estimate: on a small
    # held-out set the difference between 0.978 and 0.980 is sampling noise, and
    # a gate that fires on noise only teaches people to lower thresholds.
    verdict = proportion_gate(hr["agreement"], hr["n_test"], cfg.gates.heldout_reproduction)
    hr["statistical_verdict"] = verdict
    msg = (
        f"held-out rows land in the same cluster {hr['agreement'] * 100:.1f}% of the time "
        f"(95% CI {verdict['ci95'][0]:.3f}-{verdict['ci95'][1]:.3f}, n={hr['n_test']}) — {verdict['verdict']}"
    )
    if verdict["verdict"] == "underpowered":
        msg += f"; ~{verdict['n_needed']} held-out rows would be needed to decide"
    gate = deps.gate(
        "p6_heldout_reproduction", "p6",
        passed=verdict["passed"],
        observed=hr, threshold={"agreement": cfg.gates.heldout_reproduction, "rule": "95% CI vs threshold"},
        message=msg,
        remediation=(
            "A partition that only exists when it can see every row is a description of "
            "this sample, not of the phenomenon. Reduce K or revisit the representation "
            "before naming anything."
        ),
        warn_only=verdict["verdict"] == "underpowered",
    )
    return {
        "phase": "p7",
        "artifacts": artifacts,
        "gates": {gate.name: gate},
        "leaf_count": ref_out["n_leaves"],
        "completed_phases": ["p6"],
        "events": [
            f"P6: {tree['n_families']} families / {ref_out['n_leaves']} leaves, "
            f"held-out reproduction {hr['agreement']:.3f}"
        ],
    }
