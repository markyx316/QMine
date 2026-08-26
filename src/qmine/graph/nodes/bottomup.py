"""Phases 3-6 — representation, algorithm, granularity, hierarchy."""

from __future__ import annotations

from typing import Any

import numpy as np


from ...config import alpha_sweep_k_for
from ...determinism import hash_texts
from ...ops.stats import proportion_gate
from ...agents.observe import observe_phase
from ...agents.propose_grid import propose_grid
from ...ops.propose import grade_proposal
from ...ops.cluster import (
    partition_stability,
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
    _obs_gates: dict[str, Any] = {}
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
        texts, candidates, k=alpha_sweep_k_for(cfg), template_masks=masks,
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

    # --- vet the phrasing families before they judge anything --------------
    # These groups are about to decide alpha and to define the fragmentation
    # metric, so they must actually be single-intent. A seeded group carries a
    # human's assertion; a mined one carries nothing until it earns trust by
    # being measurably tighter than random. Markers like "是什么" attach to every
    # topic and score at chance — they are question forms, not intents.
    from ...ops.templates import validate_group_cohesion

    cohesion: dict[str, Any] = {}
    if masks:
        cohesion = validate_group_cohesion(masks, dense, seed=cfg.seed_metric)
        seeded = {g["name"] for g in deps.load("template_groups")["groups"] if not g["discovered"]}
        keep = set(cohesion["trusted"]) | seeded          # seeds keep their human vouch
        dropped = [n for n in masks if n not in keep]
        if dropped:
            deps.emit(f"  dropped {len(dropped)} phrasing group(s) that were no tighter than "
                      f"chance: {dropped[:4]}")
        masks = {k: v for k, v in masks.items() if k in keep}
        deps.cache_put("template_masks", masks)
        cohesion["dropped"] = dropped
        cohesion["kept_because_seeded"] = sorted(seeded - set(cohesion["trusted"]))

    # --- 3c: alpha sweep ---------------------------------------------------
    # The grid is a K12 artefact under a comment saying not to inherit the K12
    # answer. A proposer may widen it from CORPUS characteristics only (blindness
    # enforced on the payload), and anything it adds still has to win on merit.
    from ...ops.propose import ALPHA_SPEC

    alpha_grid, alpha_proposal = propose_grid(
        deps, "alpha", list(cfg.representation.alpha_grid), ALPHA_SPEC)
    deps.emit(f"P3c alpha sweep — {alpha_grid}")
    sweep = alpha_sweep(
        dense, sp["svd_block"], alphas=alpha_grid,
        k=alpha_sweep_k_for(cfg), template_masks=masks,
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
        {"bakeoff": bake, "template_cohesion": cohesion, "sparse": {k: v for k, v in sp.items() if k in ("vocab_size", "explained_variance", "n_components")},
         "alpha_sweep": sweep,
         # Graded, not just recorded: whether a PROPOSED alpha actually won is the
         # only way to tell over runs whether the proposer earns its comparisons.
         "grid_proposal": grade_proposal(alpha_proposal, alpha),
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
            # STATE THE RULE THAT ACTUALLY RAN — as a STABLE, TRANSLATABLE
            # sentence, with this run's numbers in `evidence` beside it.
            #
            # It used to read "Lowest template fragmentation with highest
            # stability", and on live39 the chosen alpha=0.1 had NEITHER
            # (fragmentation 2.0193 vs 1.9799 at alpha=0.0; alpha=0.5 had higher
            # stability). The phase observer caught it. The artifact's
            # `chosen_by` was right all along; only the sentence a reader sees
            # was false.
            #
            # The numbers live in `evidence`, not in this string: `prose()`
            # matches a prefix and returns a FIXED translation, so interpolating
            # run-specific values here would either defeat the translation or
            # ship English into a Chinese report — which is exactly what the
            # first attempt at this fix did.
            "Alpha is not chosen by taking the lowest fragmentation outright. "
            "Fragmentation differences inside a tie-band are treated as ties and "
            "broken on replay stability, which is the sturdier measurement — so "
            "the winner is the most reproducible option among those effectively "
            "tied on fragmentation, and is normally neither extreme. The phrasing "
            "block enters the cosine with weight alpha SQUARED, so a small alpha "
            "is a tie-breaker rather than a co-equal signal.",
            evidence={
                "chosen_alpha": alpha,
                "its_fragmentation": _alpha_row(sweep, alpha).get("template_fragmentation"),
                "its_stability": _alpha_row(sweep, alpha).get("stability_ari"),
                "lowest_fragmentation_in_sweep": min(r["template_fragmentation"] for r in sweep["rows"]),
                "highest_stability_in_sweep": max(r["stability_ari"] for r in sweep["rows"]),
                "surface_vote_share_pct": round(surface_vote_share(alpha) * 100, 1),
                "tie_band": sweep.get("chosen_by", ""),
            },
            decisive_metrics=["template_fragmentation", "stability_ari"],
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
    if deps.cfg.observe_phases and not deps.cfg.fast_mode:
        _obs_gates.update(observe_phase(deps, "p3", {"representation": deps.load("representation")}, decisions=decisions).as_state_gates())
    return {
        "phase": "p4",
        "gates": _obs_gates,
        "artifacts": {"emb_base": emb_ref, "emb_svd_char": svd_ref, "emb_hybrid": hyb_ref,
                      "representation": rep_ref},
        "decisions": decisions,
        "chosen_alpha": alpha,
        "chosen_encoder": chosen_encoder,
        "completed_phases": ["p3"],
        "events": events,
    }



def _alpha_row(sweep: dict[str, Any], alpha: float) -> dict[str, Any]:
    """The sweep row for one alpha, so a rationale can quote its own numbers."""
    for r in sweep.get("rows", []):
        if abs(float(r.get("alpha", -1)) - float(alpha)) < 1e-9:
            return r
    return {}

def p4_battery(state: PipelineState, deps: Deps) -> dict[str, Any]:
    """Run every clustering algorithm through one identical harness."""
    _obs_gates: dict[str, Any] = {}
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
        "p4", "Which clustering algorithm?", "kmeans (fixed — this phase does not select)",
        "This phase does not choose the algorithm: the tree is always built with KMeans. "
        "It is a falsification probe. Running structurally different algorithms through the "
        "same harness asks whether the structure is a property of the corpus or of KMeans' "
        "spherical-cluster assumption; a materially more reproducible alternative would be a "
        "warning to read the family layer as provisional, not a reason to switch mid-flight.",
        evidence={"reference": verdict.get("reference_algorithm"),
                  "best_alternative": verdict.get("best_alternative"),
                  "alternative_beats_reference_by": verdict.get("alternative_beats_reference_by"),
                  "kmeans_assumption_contradicted": verdict.get("kmeans_assumption_contradicted")},
        decisive_metrics=[],
        rejected=[{"option": r["algorithm"],
                   "why_rejected": "probe arm — never a candidate for the delivered tree",
                   "metrics": {"stability_ari": r["stability_ari"], "silhouette": r["silhouette"]}}
                  for r in result["rows"] if not r["algorithm"].startswith("kmeans")][:12],
    )
    events = [f"P4: {verdict['chosen']} wins on stability"]
    if verdict["density_candidates_for_manual_review"]:
        d = verdict["density_candidates_for_manual_review"][0]
        events.append(
            f"P4: density methods held back for the novelty sentinel "
            f"(best HDBSCAN: {d['noise_rate'] * 100:.0f}% noise, {d['n_clusters']} clusters)"
        )
    if deps.cfg.observe_phases and not deps.cfg.fast_mode:
        _obs_gates.update(observe_phase(deps, "p4", {"battery": deps.load("battery")}, decisions=[decision]).as_state_gates())
    return {
        "phase": "p5",
        "gates": _obs_gates,
        "artifacts": {"battery": ref},
        "decisions": [decision],
        "chosen_algorithm": family,
        "completed_phases": ["p4"],
        "events": events,
    }


def p5_granularity(state: PipelineState, deps: Deps) -> dict[str, Any]:
    """Triangulate the family scale from three independent estimators."""
    _obs_gates: dict[str, Any] = {}
    cfg = deps.cfg
    H = deps.embedding("emb_hybrid")
    masks = deps.template_masks()
    from ...ops.propose import k_spec

    ks, k_proposal = propose_grid(
        deps, "family_k", list(cfg.clustering.k_sweep),
        k_spec(len(deps.df), cfg.clustering.min_leaf_size))
    deps.emit(f"P5 granularity — K sweep over {len(ks)} values")

    # Full-effort unless this is an explicit smoke run: the cheap estimator was
    # measured at only 0.43 rank correlation with the full sweep on this corpus,
    # and K is inherited by every phase after this one.
    sweep = k_sweep(H, ks, seeds=tuple(cfg.seed_replay),
                    silhouette_sample=cfg.clustering.silhouette_sample, template_masks=masks,
                    fast=cfg.fast_mode)
    expected_mid = int(sum(cfg.domain.expected_family_range) / 2)
    da = deep_aligned_estimate(H, expected_mid, multiplier=cfg.clustering.deep_aligned_multiplier,
                               seed=cfg.seed_metric)
    tri = triangulate_k(sweep, da, tuple(cfg.domain.expected_family_range))
    k = tri["chosen_family_k"]
    deps.emit(f"  located K={k} (by {tri.get('locator', '?')}); "
              f"DeepAligned leaf estimate {da['k_estimate']}; "
              f"converged={tri['converged']}")

    ref = deps.store.put_json(
        "granularity", {"k_sweep": sweep, "deep_aligned": da, "triangulation": tri,
                        "grid_proposal": grade_proposal(k_proposal, tri.get("chosen_family_k"))},
        producer="p5", summary=f"family K={k}",
    )
    tie_ks = {t["k"] for t in tri.get("tie_set", [])}
    decision = deps.decision(
        "p5", "How many families?", f"K = {k}",
        "Replay stability only REJECTS here — its seed-to-seed spread on this corpus "
        "(~0.10 ARI) is larger than the differences between adjacent K (~0.05), and its "
        "curve is still climbing below the grid, so ranking by it reads noise and trends "
        "toward a degenerate two-way split. K is LOCATED by alignment with the phrasing "
        "groups (AMI), the one metric here with a two-sided penalty and therefore a real "
        "interior optimum. Where several K are indistinguishable the tie set is reported "
        "and the simplest is taken — an averaged K is defensible to nobody.",
        evidence={**tri["estimates"], "locator": tri.get("locator"),
                  "tie_set": [t["k"] for t in tri.get("tie_set", [])],
                  "n_rejected_as_unstable": tri.get("n_rejected_as_unstable")},
        decisive_metrics=["intent_alignment_ami", "stability_ari (rejection only)"],
        rejected=[{"option": f"K={r['k']}",
                   "why_rejected": ("rejected: replay stability below the floor"
                                    if r["stability_ari"] < tri.get("stability_floor", 0.55)
                                    else "lower alignment with the phrasing groups"),
                   "metrics": {"intent_alignment_ami": r.get("intent_alignment_ami"),
                               "stability_ari": r["stability_ari"],
                               "silhouette": r["silhouette"]}}
                  for r in sweep if r["k"] != k and r["k"] not in tie_ks],
    )
    events = [f"P5: family K={k} via {tri.get('locator')}; "
              f"{len(tri.get('tie_set', []))} K statistically tied; "
              f"{tri.get('n_rejected_as_unstable', 0)} rejected as irreproducible"]
    if not tri["converged"]:
        events.append(f"P5: estimators did NOT converge — {tri['divergence_note']}")
    if tri["silhouette_disagrees"]:
        events.append(f"P5: silhouette peaks at K={tri['estimates']['silhouette_peak_k']} — advisory only")
    if deps.cfg.observe_phases and not deps.cfg.fast_mode:
        _obs_gates.update(observe_phase(deps, "p5", {"granularity": deps.load("granularity")}, decisions=[decision]).as_state_gates())
    return {
        "phase": "p6",
        "gates": _obs_gates,
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
    _obs_gates: dict[str, Any] = {}
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
    lk = tree.get("local_k", {})
    deps.emit(f"  built {tree['n_families']} families / {tree['n_leaves']} leaves"
              + (f" — {lk['n_families_not_split']} 个家族无内部结构未再分, "
                 f"{lk['n_silhouette_overruled']} 个 silhouette 的选择被稳定性否决"
                 if lk else ""))

    ref_out = refine(
        H, tree["leaf_labels"], tree["leaf_family"], rounds=cfg.clustering.refine_rounds,
        merge_cos=cfg.clustering.refine_merge_cos, move_tolerance=cfg.clustering.refine_move_tolerance,
        min_leaf_size=cfg.clustering.min_leaf_size, seed=cfg.seed_metric,
    )
    deps.emit(f"  refined to {ref_out['n_leaves']} leaves in {len(ref_out['history'])} rounds "
              f"(converged={ref_out['converged']})")

    # --- minority-language families get a script-appropriate subdivision ---
    lang_meta: dict[str, Any] = {}
    langprof = deps.recover("language_profile", "language_profile")
    row_lang = None
    if deps.has("row_language"):
        row_lang = [str(x) for x in deps.load("row_language")]
    elif langprof and langprof.get("row_labels"):
        row_lang = langprof["row_labels"]
    if langprof and row_lang and langprof.get("posture") in ("minority_at_risk", "genuinely_multilingual"):
        from ...ops.language import minority_sub_intents

        res = minority_sub_intents(
            deps.df[cfg.data.text_column].astype(str).tolist(), row_lang,
            ref_out["leaf_labels"], ref_out["leaf_family"],
            dominant=langprof["dominant"], seed=cfg.seed_metric,
        )
        if res["families_treated"]:
            deps.cache_put("minority_sub_intent", res["sub_intent_facet"])
            deps.store.put_matrix(
                "minority_sub_intent",
                np.asarray(res["sub_intent_facet"], dtype=object).astype("U32"),
                producer="p6", summary=f"{res['n_sub_intents']} minority-language sub-intents",
            )
            lang_meta = {"families_treated": res["families_treated"],
                         "n_sub_intents": res["n_sub_intents"],
                         "contract": res["contract"]}
            deps.emit(
                f"  minority-language facet: {len(res['families_treated'])} family(ies) resolved "
                f"into {res['n_sub_intents']} sub-intents (a column, not leaves — the hybrid "
                "space cannot express intent within a minority language)"
            )

    hr = heldout_reproduction(H, ref_out["leaf_labels"], fraction=cfg.clustering.heldout_fraction,
                              seed=cfg.seed_metric)
    deps.emit(f"  held-out structure reproduction {hr['agreement']:.3f}")

    # The bar is relative, not absolute. 0.98 is a K12 observation: across three
    # corpora the achievable value runs 0.973 (k12 12k) to 0.991 (e-commerce), and
    # it falls as K rises, so an absolute constant fails perfectly good runs on
    # harder corpora and passes weak ones on easy corpora.
    #
    # What the gate is really asking is whether the partition survives NOT having
    # seen every row. So bound the demand by how well it reproduces when the data
    # is split at all — `partition_stability`, the disjoint-half centroid replay.
    #
    # The two are deliberately not the same test, and the direction matters: the
    # half-sample replay compares two INDEPENDENT halves, while held-out fits on
    # 80% and scores the remaining 20% against the full-data assignment. The former
    # is strictly harder, so it sits BELOW held-out on the same partition (0.893 vs
    # 0.973 on the reference run). That makes it a valid floor and a conservative
    # one: it is used only to stop the gate demanding out-of-sample agreement that
    # exceeds what the structure manages when it is split at all, which is what the
    # absolute 0.98 was doing. It is not used to raise the bar.
    ceiling = partition_stability(H, ref_out["leaf_labels"],
                                  sample=cfg.clustering.silhouette_sample,
                                  seed=cfg.seed_metric)
    floor = cfg.gates.heldout_reproduction
    if ceiling.get("mean") == ceiling.get("mean") and ceiling["mean"] > 0:
        # Never demand more than the structure can achieve on its own data, and
        # never accept a large shortfall against it.
        # `min`: the configured value stays the ceiling on the demand. This can only
        # ever RELAX the bar, never tighten it beyond what was configured.
        floor = min(floor, round(ceiling["mean"] * cfg.gates.heldout_share_of_ceiling, 4))
    hr["in_sample_ceiling"] = ceiling
    hr["effective_threshold"] = floor
    verdict = proportion_gate(hr["agreement"], hr["n_test"], floor)
    hr["statistical_verdict"] = verdict


    # Counted from the labels that SHIP, not from the pre-refinement tree. A leaf
    # that refinement merged away is absent from `leaf_labels` and must be absent
    # here; `leaf_family` still carries a row for it, so grouping over the lookup
    # table instead of over the labels would reproduce the defect exactly.
    from ...ops.cluster import leaves_per_family

    _lpf = leaves_per_family(ref_out["leaf_labels"], ref_out["leaf_family"])
    # Every count in this artifact now comes from ONE source — the labels that
    # ship — so they cannot disagree. That is the fix; an assert would only have
    # detected a disagreement this construction makes impossible, and would have
    # crashed a paid run at p6 if `_compact` ever left a gap in the label space.
    # `n_leaves` is the number of leaves a reader can actually find in the table.
    _n_leaves = sum(_lpf.values())
    if _n_leaves != int(ref_out["n_leaves"]):
        deps.emit(f"  ⚠ refinement reports {ref_out['n_leaves']} leaves but "
                  f"{_n_leaves} appear in the labels — reporting the labels")

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
            {"n_families": len(_lpf), "n_leaves": _n_leaves,
             "n_leaves_reported_by_refinement": int(ref_out["n_leaves"]),
             "leaves_per_family": _lpf,
             # THE SAME ARTIFACT MUST NOT MIX PRE- AND POST-REFINEMENT COUNTS.
             # `n_leaves` came from `ref_out` and `leaves_per_family` from `tree`,
             # with nothing marking the difference, so live39 shipped
             # `n_leaves = 29` beside a breakdown summing to 32 — families 2, 3
             # and 8 were each shown leaves that refinement had already merged
             # away. Found live by `p6_observer`, which was right and could not
             # prove it; `p6_leaf_counts_agree` below is now the proof.
             "leaves_per_family_before_refinement": tree["leaves_per_family"],
             "n_families_before_refinement": tree["n_families"],
             "min_leaf_size_applied": tree["min_leaf_size_applied"],
             "refinement_history": ref_out["history"], "converged": ref_out["converged"],
             # How each family decided its own leaf count, what silhouette alone
             # would have done, and which families declined to split at all. This
             # layer used to choose k by silhouette argmax with no record kept.
             "local_k": tree.get("local_k", {}),
             "heldout_reproduction": hr, "minority_language_rescue": lang_meta},
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
    msg = (
        f"held-out rows land in the same cluster {hr['agreement'] * 100:.1f}% of the time "
        f"(95% CI {verdict['ci95'][0]:.3f}-{verdict['ci95'][1]:.3f}, n={hr['n_test']}) — {verdict['verdict']}"
    )
    if verdict["verdict"] == "underpowered":
        msg += f"; ~{verdict['n_needed']} held-out rows would be needed to decide"
    gate = deps.gate(
        "p6_heldout_reproduction", "p6",
        passed=verdict["passed"],
        observed=hr,
        threshold={"agreement": floor, "absolute_floor_configured": cfg.gates.heldout_reproduction,
                   "in_sample_ceiling": ceiling.get("mean"),
                   "rule": "95% CI vs min(configured floor, share of the in-sample ceiling)"},
        message=msg,
        remediation=(
            "A partition that only exists when it can see every row is a description of "
            "this sample, not of the phenomenon. Reduce K or revisit the representation "
            "before naming anything."
        ),
        warn_only=verdict["verdict"] == "underpowered",
    )
    if deps.cfg.observe_phases and not deps.cfg.fast_mode:
        _obs_gates.update(observe_phase(deps, "p6", {"hierarchy_meta": deps.load("hierarchy_meta")}, decisions=[]).as_state_gates())
    return {
        "phase": "p7",
        "artifacts": artifacts,
        "gates": {gate.name: gate, **_obs_gates},
        "leaf_count": ref_out["n_leaves"],
        "completed_phases": ["p6"],
        "events": [
            f"P6: {tree['n_families']} families / {ref_out['n_leaves']} leaves, "
            f"held-out reproduction {hr['agreement']:.3f}"
        ],
    }


def _centroids_for(X, labels):
    """Recompute unit centroids after leaves were added outside the refiner."""
    from sklearn.preprocessing import normalize

    n = int(labels.max()) + 1
    cents = np.zeros((n, X.shape[1]), dtype=np.float32)
    for c in range(n):
        m = labels == c
        if m.any():
            cents[c] = X[m].mean(0)
    return normalize(cents)


def p456_tree(state: PipelineState, deps: Deps) -> dict[str, Any]:
    """p4 → p5 → p6 as ONE graph node, so the two branches are the same length.

    Nothing moves into here: the three phases are called unchanged and their
    returns are merged. The reason this exists is scheduling, and it is worth
    stating precisely because it looks like a cosmetic wrapper.

    LangGraph 1.2.11 advances parallel branches in SUPERSTEPS — measured, not
    assumed: a branch waits for the slowest node of each step before starting its
    next one, and a fan-in node fires once per incoming edge unless all of them
    arrive in the same step. So the cost of a fork is decided by how the phases
    are grouped, not by their total work.

    Against live39's real timings — p2a 38 min, p2b 69 min, p3 10, p4 7, p5 14,
    p6 8 — the groupings compare like this::

        p3 | p4 | p5 | p6   vs  p2a | p2b | noop | noop   ->  38+69+14+8 = 129
        p3 | p4+p5 | p6     vs  p2a | p2b | noop          ->  38+69+8    = 115
        p3 | p4+p5+p6       vs  p2a | p2b                 ->  38+69      = 107

    Two nodes against two is the only grouping where the ENTIRE bottom-up branch
    disappears into the shadow of p2b, and it is also the only one where the join
    at `p2c` receives both edges in the same superstep. `_wrap`'s idempotence
    guard covers the join in any case; this makes the fast path the correct one.

    The three phases keep their own gates, decisions, observers and artifacts —
    a node may return as many as it likes, and dropping any of them is the
    documented way an observer's verdict reaches the log and no operator.
    """
    merged: dict[str, Any] = {}
    for fn in (p4_battery, p5_granularity, p6_hierarchy):
        if state.get("halted"):
            break
        out = fn({**state, **merged}, deps) or {}
        for key, value in out.items():
            if key in ("artifacts", "gates", "metrics", "phase_status"):
                merged.setdefault(key, {}).update(value)
            elif key in ("events", "decisions", "prescriptions", "completed_phases",
                         "errors", "warnings"):
                merged.setdefault(key, []).extend(value)
            else:
                merged[key] = value
    # `phase` is whatever the last one asked for; the graph's edges decide the
    # route, and leaving three conflicting values in the merge would be noise.
    merged["phase"] = "p2c"
    return merged
