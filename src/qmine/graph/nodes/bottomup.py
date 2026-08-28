"""Phases 3-6 — representation, algorithm, granularity, hierarchy."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


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

        # WHAT THE COHESION GATE CAN AND CANNOT SEE.
        #
        # It measures TOPICAL tightness in a semantic embedding, and an intent
        # group spans topics by construction — "X的意思是什么" for thousands of
        # different X. So it systematically penalises exactly the broad intent
        # groups that make the best references. Measured on live40 against the
        # top-down taxonomy as an independent yardstick, cohesion lift ranks the
        # six seeded groups almost backwards from their actual single-intent
        # purity (Spearman -0.60, n=6): it PASSED `word_formation` (lift 1.670,
        # purity 81.2%, the worst of the six) and REJECTED `meaning` (lift 1.269,
        # purity 88.6%, third best). Keeping seeds despite a failed lift is
        # therefore not a courtesy to the human — on this corpus it is the more
        # accurate call, and the gate is the thing that was wrong.
        cohesion["gate_scope"] = (
            "lift_over_random measures topical tightness, NOT intent purity. A "
            "group can be one intent and topically vast. Do not read a low lift "
            "as evidence the group is a bad reference; p10 measures purity "
            "against the top-down taxonomy, which is the yardstick that matters."
        )

    # NO SEEDED GROUP AT ALL IS A DIFFERENT SITUATION, AND IT WAS SILENT.
    # `deps.template_masks` falls back to unvalidated mined groups, and K is then
    # located by AMI against them. `generic.yaml` ships 0 seeds, so this is the
    # default path for a corpus with no domain profile.
    fb = getattr(deps, "_trusted_fallback", None)
    if fb and fb.get("fell_back"):
        _obs_gates["p3_locator_reference_validated"] = deps.gate(
            "p3_locator_reference_validated", phase="p3",
            passed=False,
            observed={"n_seeded_groups": 0, "n_mined_groups_used": fb.get("n_groups_used")},
            threshold={"rule": "at least one seeded phrasing group must survive"},
            message=("K 的定位参照没有任何**种子**措辞群 — 改用未经验证的挖掘群。"
                     "定下来的 K 会随这个参照的粒度走, 请人工复核参照本身。"),
            remediation=("为这个语料写一份 domain profile 的 template_seeds, "
                         "或者接受 K 只是相对于挖掘群的粒度锚点并在报告中如实说明。"),
        )

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
    # THE NOTE STILL DESCRIBED THE RETRACTED "ELECTION".
    #
    # "the winner is then fitted on the full corpus" is the framing the spec
    # withdrew: `build_hierarchy` hardcodes KMeans, so there is no winner and
    # nothing is promoted. live41's own p4 observer caught the contradiction
    # between this string and `verdict.role` ("falsification probe") sitting in
    # the same artifact. Same class as the retracted 淘汰赛 sentence removed from
    # the reports — this copy was in the artifact itself.
    result["note"] = (
        "this phase selects nothing: the delivered tree is always KMeans. The "
        "battery runs structurally different algorithms through one harness on a "
        "capped sub-sample to ask whether the structure is a property of the "
        "corpus or of KMeans' spherical-cluster assumption"
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
    # SCORE EVERY REFERENCE THE RUN ALREADY HAS, NOT ONLY OUR OWN SEED REGEXES.
    #
    # The located K tracks whichever partition AMI is scored against, and the one
    # that currently decides is the one WE wrote. Measured on live40's full corpus:
    # trusted phrasing groups (6 classes, our seeds) locate K=7, while
    # `ref_legacy_l1` — the corpus's own pre-existing labelling, 9 classes, complete
    # on all 49,999 rows, independent of BOTH routes — locates K=18. The deciding
    # reference is the outlier and no artifact said so.
    #
    # These columns are already read by p3 for `nmi_reference`, so scoring them here
    # adds no visibility the bottom-up path did not already have. The top-down
    # labels are deliberately NOT used: they do not exist yet under the p1 fork, and
    # locating K against them would make the bottom-up tree a function of the
    # top-down taxonomy — turning the route-concordance result from a measurement
    # into the objective that was fitted. `BlindnessFirewall.add_taxonomy` forbids
    # it architecturally for exactly that reason.
    reference_partitions: dict[str, np.ndarray] = {}
    for _c in cfg.data.reference_label_columns:
        if _c in deps.df.columns:
            reference_partitions[_c] = pd.factorize(deps.df[_c].astype(str))[0]

    sweep = k_sweep(H, ks, seeds=tuple(cfg.seed_replay),
                    silhouette_sample=cfg.clustering.silhouette_sample, template_masks=masks,
                    reference_partitions=reference_partitions or None,
                    fast=cfg.fast_mode)
    # WHICH REFERENCE MAY LOCATE K IS DECIDED BY REACH, NOT BY PREFERENCE.
    #
    # A reference that occupies only part of the partition cannot express a
    # preference about the rest, however good its rows are. Measured at a
    # representative k so the choice does not depend on the K it is choosing.
    from ...ops.cluster import choose_locator, discrimination, kmeans_fit, locator_reach

    _probe_k = sorted(ks)[len(ks) // 2]
    _probe_labels = kmeans_fit(H, _probe_k, seed=cfg.seed_metric,
                               fast=cfg.fast_mode).labels_
    _cands: dict[str, Any] = {}
    _tr_y = np.full(len(deps.df), -1, dtype=np.int64)
    for _i, _m in enumerate(masks.values()):
        _tr_y[np.asarray(_m) & (_tr_y == -1)] = _i
    if masks:
        _cands["intent_alignment_ami"] = ("phrasing_groups", _tr_y)
    for _name, _part in reference_partitions.items():
        _cands[f"ami_vs_{_name}"] = (_name, _part)

    reach = {}
    for _key, (_label, _part) in _cands.items():
        reach[_label] = {**locator_reach(_probe_labels, _part), "column": _key}
    for _label, _r in sorted(reach.items()):
        deps.emit(f"  参照系 `{_label}`: 触及 {_r['reach']:.0%} 的簇 "
                  f"(行覆盖 {_r['row_coverage']:.0%}) @ k={_probe_k}")

    locator_key, _deciding = choose_locator(
        reach, getattr(cfg.clustering, "k_locator", "auto"), sweep=sweep)
    for _label, _r in sorted(reach.items()):
        _r["discrimination"] = discrimination(sweep, _r["column"])
    # SAY WHICH CRITERION ACTUALLY DECIDED.
    #
    # This always said "highest reach", but reach is only the ADMISSION test —
    # on live41 both legacy columns reach 1.0 and discrimination broke the tie
    # (legacy_l2 17.19 against legacy_l1's 5.88). A reader told "highest reach"
    # would look at two identical numbers and be unable to see why one won.
    _r = reach.get(_deciding, {})
    _tied_on_reach = sum(1 for r in reach.values()
                         if (r.get("reach") or 0) >= (_r.get("reach") or 0) - 1e-9)
    _why = ("触及率最高" if _tied_on_reach == 1
            else f"触及率并列 ({_tied_on_reach} 个), 由分辨力 {_r.get('discrimination')} 胜出")
    deps.emit(f"  K 由 `{_deciding}` 定位 ({_why})")

    expected_mid = int(sum(cfg.domain.expected_family_range) / 2)
    da = deep_aligned_estimate(H, expected_mid, multiplier=cfg.clustering.deep_aligned_multiplier,
                               seed=cfg.seed_metric)
    tri = triangulate_k(sweep, da, tuple(cfg.domain.expected_family_range),
                        stability_floor=cfg.clustering.stability_floor,
                        locator_key=locator_key)
    tri["locator_reach"] = reach
    tri["deciding_reference"] = _deciding
    k = tri["chosen_family_k"]
    deps.emit(f"  located K={k} (by {tri.get('locator', '?')}); "
              f"DeepAligned leaf estimate {da['k_estimate']}; "
              f"converged={tri['converged']}")

    # WHAT THE LOCATOR WAS SCORED AGAINST IS ITSELF A DECISION, AND IT WAS NEVER
    # RECORDED. The located K tracks the reference's cardinality — measured on
    # live40 by holding everything else fixed and swapping only the reference: 6
    # trusted groups -> k=12, all 12 groups -> k=12, the 25-class top-down L1 ->
    # k=25. live40 then concluded its 15-25 domain prior was wrong, on a number
    # that would have agreed with that prior under a reference it already had.
    from ...ops.cluster import reference_profile

    # PROFILE THE REFERENCE THAT ACTUALLY DECIDED, not the one that used to.
    #
    # `reference_profile` was added when the phrasing groups were the only
    # locator, and it kept describing them after `choose_locator` was given other
    # candidates — so live42 shipped `locator_reference.reference = "phrasing
    # (template) groups"` beside `deciding_reference = legacy_l2`, with a caveat
    # asserting K was located against "THIS partition". Two contradictory answers
    # in one artifact; its own p5 observer caught it. Same defect as the decision
    # record's hardcoded `decisive_metrics`, one file over.
    profile = reference_profile(masks, len(deps.df))
    profile["is_the_deciding_reference"] = (locator_key == "intent_alignment_ami")
    profile["deciding_reference"] = _deciding
    if not profile["is_the_deciding_reference"]:
        _dr = reach.get(_deciding, {})
        profile["deciding_reference_profile"] = {
            "reference": _deciding,
            "reach": _dr.get("reach"),
            "row_coverage": _dr.get("row_coverage"),
            "discrimination": _dr.get("discrimination"),
            "n_clusters_reached": _dr.get("n_clusters_reached"),
        }
    deps.emit(f"  locator scored against {profile.get('n_classes')} phrasing classes "
              f"covering {profile.get('coverage')} of rows")

    # A DISAGREEMENT BETWEEN REFERENCES IS A FINDING, NOT A FOOTNOTE.
    # It says the delivered K is a property of the anchor we picked rather than of
    # the corpus, which is the single most important caveat on the whole tree.
    # THE CASE THIS EXISTS FOR: no reference reaches enough of the partition.
    #
    # That is the portable failure, not an edge case — a corpus with no external
    # labelling has only the phrasing groups, and on live40 those reach 38.9% of
    # clusters at k=18. K is then located for the third of the corpus they occupy
    # and applied to all of it. There is no label-free repair: background-as-a-class
    # and downsampled-background both still returned K=7, because the templates
    # carry no information about the rows they never match. So the honest move is
    # to say so, loudly, rather than to pretend the number is corpus-wide.
    _best_reach = max((r.get("reach") or 0.0) for r in reach.values()) if reach else 0.0
    if _best_reach < 0.80:
        _obs_gates["p5_locator_reaches_the_corpus"] = deps.gate(
            "p5_locator_reaches_the_corpus", phase="p5",
            passed=False,
            observed={"best_reach": round(_best_reach, 4),
                      "deciding_reference": _deciding,
                      "reach_by_reference": {n: r.get("reach") for n, r in reach.items()},
                      "probe_k": _probe_k},
            threshold={"min_reach": 0.80,
                       "rule": "the deciding reference must hold a real share of "
                               "at least 80% of the clusters it is scoring"},
            message=(f"定位 K 用的参照系只触及 {_best_reach:.0%} 的簇 —— "
                     f"**这个 K 是为它覆盖到的那部分语料定的**, 然后被用到了全量上。"),
            remediation=("提供一列覆盖全量的既有标注 (data.reference_label_columns), "
                         "或补充模板种子扩大覆盖。注意: 给未匹配行加一个「背景类」"
                         "并不能修复这件事 —— 实测仍然定位到同一个 K, 因为模板对"
                         "它没匹配到的行不携带任何信息。"),
        )

    sens = tri.get("reference_sensitivity") or {}
    located = sens.get("located_k_values") or {}
    if len(set(located.values())) > 1:
        deps.emit("  ⚠ 参照系之间对 K 不一致: "
                  + ", ".join(f"{n}→K={v}" for n, v in sorted(located.items())))
        _obs_gates["p5_k_references_agree"] = deps.gate(
            "p5_k_references_agree", phase="p5",
            passed=False,
            observed={"located_k_by_reference": located, "chosen_k": k,
                      "deciding_reference": "phrasing_groups"},
            threshold={"rule": "every available reference partition should locate the same K"},
            message=("不同参照系定位到不同的 K —— 交付的 K 是**相对于当选参照系**的"
                     "粒度锚点, 不是语料常数。请连同参照系一起阅读这个 K。"),
            remediation=("在报告中同时给出各参照系各自定位到的 K, 并说明当选参照系"
                         "的类目数与覆盖率; 若外部参照 (如既有标注列) 与自建模板"
                         "分歧较大, 优先复核**模板**而不是直接改 K。"),
        )

    ref = deps.store.put_json(
        "granularity", {"k_sweep": sweep, "deep_aligned": da, "triangulation": tri,
                        "locator_reference": profile,
                        "grid_proposal": grade_proposal(k_proposal, tri.get("chosen_family_k"))},
        producer="p5", summary=f"family K={k}",
    )
    tie_ks = {t["k"] for t in tri.get("tie_set", [])}
    decision = deps.decision(
        "p5", "How many families?", f"K = {k}",
        "Replay stability only REJECTS here — its seed-to-seed spread on this corpus "
        "(~0.10 ARI) is larger than the differences between adjacent K (~0.05), and its "
        "curve is still climbing below the grid, so ranking by it reads noise and trends "
        "toward a degenerate two-way split. K is LOCATED by alignment (AMI) with the "
        "reference named in `deciding_reference` — chosen for REACH across the partition "
        "and for its ability to tell different K apart, not fixed in advance. AMI is the "
        "one metric here with a two-sided penalty and therefore a real interior optimum. "
        "Where several K are indistinguishable the tie set is reported and the simplest "
        "is taken — an averaged K is defensible to nobody.",
        # NAME THE REFERENCE THAT ACTUALLY DECIDED.
        #
        # These two fields were hardcoded to `intent_alignment_ami` from when the
        # phrasing groups were the only locator. live41 located K by
        # `ami_vs_legacy_l1` and the decision record still said phrasing groups —
        # its own observer caught the contradiction between the rationale and
        # `reference_sensitivity` in the same artifact.
        evidence={**tri["estimates"], "locator": tri.get("locator"),
                  "deciding_reference": tri.get("deciding_reference"),
                  "locator_reach": tri.get("locator_reach"),
                  "tie_set": [t["k"] for t in tri.get("tie_set", [])],
                  "n_rejected_as_unstable": tri.get("n_rejected_as_unstable")},
        decisive_metrics=[tri.get("locator") or "intent_alignment_ami",
                          "stability_ari (rejection only)"],
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
        stability_floor=cfg.clustering.stability_floor,
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
