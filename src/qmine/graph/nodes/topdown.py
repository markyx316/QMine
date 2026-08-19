"""Phase 2 — the top-down route.

This is the half of the deliverable that clustering cannot produce.  Some
intents are invisible in the wording: "苹果的拼音" and "苹果怎么写" are lexically
near-identical and want different things, while "X和Y哪个对" wants a verdict and
looks exactly like a definition request.  No amount of geometry recovers those
distinctions, so a human-designed taxonomy owns them — and the two label systems
are delivered **side by side** rather than one overwriting the other.

The sub-phases are 2a design (research fan-out → architect → critic), 2b gold
(two blind annotators → kappa → referee), 2c the hybrid classifier, and 2d
adversarial validation.
"""

from __future__ import annotations

import json
import re
from typing import Any

import time

import numpy as np
import pandas as pd

from ...agents.roles import (
    RESEARCH_ANGLES,
    AdversaryAgent,
    AnnotatorAgent,
    ArchitectAgent,
    CriticAgent,
    RefereeAgent,
    ResearcherAgent,
)
from ...config import gold_size_for
from ...determinism import deterministic_subsample, rng
from ...ops.audit import stratified_sample
from ...ops.classify import UNLABELED, RuleEngine, agreement, build_features, train_classifier
from ...records import AdjudicationRule, GoldRow, Taxonomy, TaxonomyNode
from ...state import PipelineState
from ..deps import Deps


# ==========================================================================
# 2a — taxonomy design
# ==========================================================================

def p2a_taxonomy(state: PipelineState, deps: Deps) -> dict[str, Any]:
    """Five researchers on disjoint angles → one architect → one critic.

    The angles are disjoint by design.  Give five agents the same brief and you
    get five rediscoveries of the obvious categories and nobody covering the
    awkward ones; give them distinct assignments and the union actually spans
    the problem.  The critic runs afterwards rather than in parallel, because a
    critique of a draft is worth more than a fifth opinion on the raw log.
    """
    cfg = deps.cfg
    df = deps.df
    audit = deps.load("data_audit")
    templates = deps.load("template_groups")
    risk = deps.load("risk_screen")
    ctx = deps.agent_ctx()
    deps.emit(f"P2a taxonomy — {cfg.taxonomy.n_researchers} researchers fanning out")

    angles = RESEARCH_ANGLES[: max(cfg.taxonomy.n_researchers, 1)]

    def _one(angle: dict[str, Any]) -> Any:
        evidence = _evidence_for_angle(
            angle["key"], df, audit, templates, risk, cfg, seed=angles.index(angle)
        )
        agent = ResearcherAgent(ctx, suffix=f"_{angle['key']}")
        if angle.get("web"):
            from ...tools.web import RESEARCH_TOOLS

            agent.tools = RESEARCH_TOOLS
        sub = agent.run(
            assignment=angle["assignment"], evidence=evidence,
            domain_notes=cfg.domain.domain_notes,
        )
        sub.angle = angle["key"]
        deps.emit(f"  researcher[{angle['key']}] → {len(sub.candidates)} candidates"
                  + (" (web-researched)" if getattr(agent, "used_tools", False) else ""))
        return sub

    # The angles are independent by construction — that is the whole point of the
    # fan-out — so they run concurrently. Sequentially, five agents that each take
    # a minute (longer for the web-researching ones) turn Phase 2a into the
    # slowest step in the pipeline for no methodological reason.
    submissions: list[Any] = []
    if len(angles) > 1 and not deps.registry.is_offline:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=min(len(angles), cfg.llm.max_concurrency)) as pool:
            for sub in pool.map(_one, angles):
                submissions.append(sub)
    else:
        submissions = [_one(a) for a in angles]

    architect = ArchitectAgent(ctx)
    draft = architect.run(
        submissions=submissions,
        domain_notes=cfg.domain.domain_notes,
        pragmatic_hints=cfg.domain.pragmatic_intents_hint,
        memory_block=deps.recall_block("designing an intent taxonomy"),
        l1_range=cfg.taxonomy.l1_target_range,
        min_rules=cfg.taxonomy.min_adjudication_rules,
    )
    taxonomy = Taxonomy(
        nodes=draft.nodes or _fallback_nodes(submissions),
        rules=draft.rules,
        labeling_guide=draft.labeling_guide,
        notes=draft.design_notes,
    )

    sample = df[cfg.data.text_column].astype(str).iloc[
        deterministic_subsample(len(df), 120, cfg.seed_metric)
    ].tolist()
    critique = CriticAgent(ctx).run(taxonomy=taxonomy, sample_queries=sample)
    deps.emit(f"  critic → {len(critique.findings)} findings, verdict {critique.verdict}")

    rule_health = _validate_rules(taxonomy, deps)

    # Playbook 2a quality gate: 50 queries, two independent annotators, and a
    # hard stop below 85% agreement — "一致率 <85% 则回炉改指南/裁决规则, 而非直接开标".
    # The point is economic as much as methodological. Annotating the full gold
    # set costs hundreds of calls; this costs four, and it asks the same question.
    # Discovering an ambiguous guide here rather than after 3,000 annotations is
    # the difference between a cheap redraft and an expensive one.
    pilot = _pilot_agreement(deps, ctx, df, taxonomy)
    # Capture it. `deps.gate` builds and returns a GateResult but does not write to
    # state, so a gate that is not put in this node's returned `gates` dict is
    # invisible to the router and can never halt anything — which is what happened
    # to this one, in five complete runs, while `declared_gates_never_evaluated`
    # named it every time.
    # The pilot must be judged against the bar it exists to predict. The playbook
    # sets it at 85% RAW agreement while the downstream gate wants kappa >= 0.90 —
    # and those are not the same demand. Backing chance agreement out of six real
    # runs, kappa 0.90 needs raw agreement of 0.91-0.93, so a pilot passing at 0.85
    # waves through a guide that then fails kappa by a wide margin, which is exactly
    # the expensive discovery the pilot exists to prevent. Judge it on kappa.
    #
    # At n=50 kappa is noisy, so the test is one-sided and generous: fail only when
    # even the upper confidence bound falls short. "Even reading this sample
    # charitably, this guide cannot reach the bar."
    # Two ways to clear this gate, because there are two things it can discover.
    #
    #   (1) The guide already reaches the kappa target. Proceed.
    #   (2) It does not, but it is at the ANNOTATOR'S OWN CEILING — this annotator
    #       does not agree with itself any better either. Redrafting cannot close
    #       that gap; only a stronger annotator can. Halting the run would demand a
    #       repair the operator has no way to perform, so it warns loudly, records
    #       the ceiling as the honest achievable bar, and lets the run continue with
    #       every downstream number carrying that caveat.
    #
    # What still halts is the case the gate exists for: agreement well below what
    # the annotator demonstrably can do, i.e. a guide with real slack in it.
    # The ceiling only means something if a real annotator produced it. The offline
    # stand-in is a deterministic function of its batch, so re-asking it in a
    # different composition measures the stand-in's batching, not an annotator's
    # reliability — and it would report "at ceiling" for any guide at all.
    at_ceiling = (
        not deps.registry.is_offline
        and (pilot.get("share_of_ceiling_reached") or 0) >= 0.90
    )
    reaches_target = pilot["kappa_upper"] >= cfg.gates.kappa
    pilot_gate = deps.gate(
        "p2a_pilot_agreement", "p2a",
        passed=reaches_target or at_ceiling,
        observed={"kappa": pilot["kappa"], "kappa_upper_95": pilot["kappa_upper"],
                  "raw_agreement": pilot["raw_agreement"], "n": pilot["n"],
                  "raw_agreement_implied_by_target": pilot["raw_needed_for_target"],
                  "annotator_self_consistency_kappa": pilot.get("self_consistency_kappa"),
                  "share_of_ceiling_reached": pilot.get("share_of_ceiling_reached"),
                  "at_annotator_ceiling": at_ceiling},
        threshold={"kappa": cfg.gates.kappa,
                   "playbook_raw_agreement_floor": cfg.taxonomy.pilot_agreement_threshold},
        message=(f"pilot: kappa {pilot['kappa']:.3f} (95% upper {pilot['kappa_upper']:.3f}) "
                 f"on {pilot['n']} queries; raw agreement {pilot['raw_agreement']:.1%}, "
                 f"but kappa {cfg.gates.kappa} needs about "
                 f"{pilot['raw_needed_for_target']:.1%}"
                 + (f"; annotator self-consistency kappa {pilot['self_consistency_kappa']} "
                    f"({pilot.get('share_of_ceiling_reached')} of ceiling reached)"
                    if pilot.get("self_consistency_kappa") is not None else "")
                 + (" — AT THE ANNOTATOR CEILING: this is the achievable bar on this "
                    "corpus with this annotator, not a fixable guide defect"
                    if at_ceiling and not reaches_target else "")
                 + (f" — top confusions {pilot['top_confusions']}" if pilot["top_confusions"] else "")),
        remediation=(
            pilot.get("ceiling_verdict", "")
            + ". The guide is ambiguous before a single gold row has been paid for. Fix the "
            "definitions and adjudication rules for the confused pairs above and re-run "
            "2a — the playbook is explicit that this is the moment to redraft, not to "
            "start annotating (回炉改指南/裁决规则, 而非直接开标)."
            if not at_ceiling else
            "Inter-annotator agreement has reached this annotator's own self-consistency, "
            "so no amount of guide repair will raise it — the remaining disagreement is "
            "annotator noise, not guide ambiguity. To go higher, use a stronger model or "
            "human annotators. Otherwise treat the self-consistency kappa as the honest "
            "ceiling for this corpus and read every downstream number against it."
        ),
        warn_only=deps.registry.is_offline,
    )

    n_l1 = len(taxonomy.l1()) or len(taxonomy.nodes)
    lo, hi = cfg.taxonomy.l1_target_range
    gate = deps.gate(
        "p2a_taxonomy_shape",
        "p2a",
        passed=(lo <= n_l1 <= hi) and len(taxonomy.rules) >= cfg.taxonomy.min_adjudication_rules,
        observed={"n_l1": n_l1, "n_rules": len(taxonomy.rules), "critic_verdict": critique.verdict,
                  "rules_repaired": rule_health["n_repaired"], "rules_dropped": rule_health["n_dropped"]},
        threshold={"l1_range": [lo, hi], "min_rules": cfg.taxonomy.min_adjudication_rules},
        message=f"{n_l1} L1 intents, {len(taxonomy.rules)} adjudication rules",
        remediation=(
            "Too few classes and a catch-all swells until it means nothing; too many and "
            "annotators stop agreeing, which destroys the gold set. Too few rules means "
            "the referee has nothing to cite and adjudication becomes taste."
        ),
        warn_only=True,
    )

    for node in taxonomy.nodes:
        deps.memory.put_glossary(node.code, node.model_dump())
    for rule in taxonomy.rules:
        deps.memory.remember_rule(rule)

    tax_ref = deps.store.put_json(
        "taxonomy",
        {
            "taxonomy": taxonomy.model_dump(),
            "submissions": [s.model_dump() for s in submissions],
            "critique": critique.model_dump(),
            "dropped_candidates": draft.dropped_candidates,
        },
        producer="p2a",
        summary=f"{n_l1} L1 intents, {len(taxonomy.rules)} rules",
    )
    deps.cache_put("taxonomy_obj", taxonomy)

    decision = deps.decision(
        "p2a",
        "What is the intent taxonomy?",
        f"{n_l1} L1 intents across {len({n.axis for n in taxonomy.nodes})} axes",
        f"Synthesised from {len(submissions)} independent research angles; "
        f"critic returned {len(critique.findings)} findings ({critique.verdict}).",
        decided_by="agent",
        evidence={"angles": [a["key"] for a in angles], "critic_verdict": critique.verdict},
        rejected=draft.dropped_candidates,
    )

    return {
        "phase": "p2b",
        "artifacts": {"taxonomy": tax_ref},
        "gates": {gate.name: gate, pilot_gate.name: pilot_gate},
        "decisions": [decision],
        "completed_phases": ["p2a"],
        "events": [f"P2a: taxonomy with {n_l1} L1 intents and {len(taxonomy.rules)} rules"],
    }


def _evidence_for_angle(
    key: str, df: pd.DataFrame, audit: dict, templates: dict, risk: dict, cfg: Any, seed: int
) -> str:
    """Give each researcher only what its angle needs.

    Context isolation is not politeness here — a researcher handed the full
    evidence bundle writes the same generic taxonomy every other researcher
    would, and the fan-out stops buying anything.
    """
    col = cfg.data.text_column
    if key == "log_reading":
        idx = stratified_sample(df, 400, strata_cols=[c for c in cfg.data.reference_label_columns if c in df.columns][:1], seed=seed)
        rows = df[col].astype(str).iloc[idx].tolist()
        return "## Raw queries (stratified sample, read every one)\n" + "\n".join(f"- {q}" for q in rows)
    if key == "literature":
        return (
            f"## Corpus profile\n{json.dumps({k: v for k, v in audit.items() if k in ('n_rows', 'length', 'script_mix')}, ensure_ascii=False, indent=1)}\n\n"
            f"## Phrasing families found in the data\n"
            + "\n".join(f"- {g['name']}: {g['n_hits']} hits ({g['share'] * 100:.1f}%), e.g. {g['examples'][:3]}"
                        for g in templates["groups"])
        )
    if key == "legacy_audit":
        ref = audit.get("reference_taxonomy", {})
        if not ref:
            return "## No legacy taxonomy exists in this corpus.\nReport that, and propose nothing."
        parts = []
        for c, info in ref.items():
            parts.append(f"### Legacy column {c} — {info['n_classes']} classes")
            parts += [f"- {d['label']}: {d['n']} ({d['share'] * 100:.2f}%)" for d in info["distribution"][:30]]
            parts.append(f"Flagged as shape-defined or catch-all: {json.dumps(info['form_defined_suspects'], ensure_ascii=False)}")
            for d in info["distribution"][:8]:
                mask = df[c].astype(str) == d["label"]
                ex = df[col].astype(str)[mask].head(10).tolist()
                parts.append(f"  samples of {d['label']}: {ex}")
        return "\n".join(parts)
    if key == "pragmatic_intents":
        r = rng(seed)
        pool = df[df["len"] >= 6] if "len" in df.columns else df
        take = pool.iloc[r.choice(len(pool), size=min(300, len(pool)), replace=False)]
        return (
            "## Longer queries, where pragmatic intent is most visible\n"
            + "\n".join(f"- {q}" for q in take[col].astype(str).tolist())
            + "\n\n## Phrasing families already identified (these are the SURFACE-visible ones — "
            "your job is what they miss)\n"
            + "\n".join(f"- {g['name']}: {g['intent_hint']}" for g in templates["groups"])
        )
    if key == "risk_compliance":
        parts = ["## Pre-screen results (patterns supplied by the domain profile)"]
        for c in risk["categories"]:
            parts.append(f"- {c['name']}: {c['n_hits']} hits ({c['share'] * 100:.2f}%), samples: {c['samples'][:6]}")
        idx = deterministic_subsample(len(df), 200, seed)
        parts.append("\n## Random queries — look for risk the pre-screen patterns MISSED")
        parts += [f"- {q}" for q in df[col].astype(str).iloc[idx].tolist()]
        return "\n".join(parts)
    return ""


def _fallback_nodes(submissions: Any) -> list[TaxonomyNode]:
    """If the architect returns nothing usable, promote the researchers' candidates.

    Offline runs land here: the heuristic stand-in cannot synthesise a taxonomy,
    so rather than proceed with an empty one we carry the candidates forward and
    let the shape gate report what actually happened.
    """
    nodes: list[TaxonomyNode] = []
    seen: set[str] = set()
    for s in submissions:
        for c in getattr(s, "candidates", []):
            code = (c.code or re.sub(r"\W+", "_", c.name)[:24]).upper()
            if code in seen:
                continue
            seen.add(code)
            nodes.append(
                TaxonomyNode(
                    code=code, name=c.name, definition=c.definition, user_need=c.user_need,
                    positive_examples=c.evidence[:5], risk=c.risk, pragmatic_only=c.pragmatic,
                    axis=c.axis, source_evidence=[f"researcher:{getattr(s, 'angle', '')}"],
                )
            )
    return nodes


# ==========================================================================
# 2b — gold standard
# ==========================================================================

def p2b_gold(state: PipelineState, deps: Deps) -> dict[str, Any]:
    """Two blind annotators, kappa, then a referee who writes rules as it goes.

    The annotators are separate agent invocations with no shared context, so
    "independent" is structural.  Their disagreements are the deliverable almost
    as much as their agreements: each one localises a hole in the guide, and the
    referee's job is to fill it rather than merely to arbitrate it.
    """
    cfg = deps.cfg
    df = deps.df
    taxonomy: Taxonomy = deps.taxonomy()
    ctx = deps.agent_ctx()

    strata = [c for c in cfg.data.reference_label_columns if c in df.columns][:1]
    n_gold = cfg.taxonomy.gold_sample_size or gold_size_for(len(df), cfg.taxonomy)
    idx = stratified_sample(df, n_gold, strata_cols=strata, seed=cfg.seed_metric)
    queries = df[cfg.data.text_column].astype(str).iloc[idx].tolist()
    deps.emit(f"P2b gold — {len(queries)} queries ({len(queries)/len(df):.1%} of corpus, "
              + ("derived from corpus size" if not cfg.taxonomy.gold_sample_size else "pinned by config")
              + "), double-blind annotation")

    classes_txt = _render_classes(taxonomy)
    rules_txt = _render_rules(taxonomy)
    kappa_trace: list[dict[str, Any]] = []

    labels_a = _annotate(ctx, "a", queries, classes_txt, rules_txt, taxonomy.labeling_guide, deps)
    labels_b = _annotate(ctx, "b", queries, classes_txt, rules_txt, taxonomy.labeling_guide, deps)

    agree = agreement([l["label"] for l in labels_a], [l["label"] for l in labels_b])
    deps.emit(f"  raw agreement {agree['raw_agreement']:.3f}, kappa {agree['kappa']:.3f}, "
              f"{agree['n_disagreements']} disagreements")

    rows: list[GoldRow] = []
    for i, q in enumerate(queries):
        la, lb = labels_a[i]["label"], labels_b[i]["label"]
        rows.append(GoldRow(
            query=q, idx=int(idx[i]), label_a=la, label_b=lb,
            final=la if la == lb else "", agreed=la == lb,
            rationale_a=labels_a[i].get("rationale", ""), rationale_b=labels_b[i].get("rationale", ""),
        ))

    new_rules: list[AdjudicationRule] = []
    disagreements = [
        {"query": r.query, "label_a": r.label_a, "label_b": r.label_b,
         "rationale_a": r.rationale_a, "rationale_b": r.rationale_b}
        for r in rows if not r.agreed
    ]
    if disagreements:
        # Sequential on purpose. Adjudication is not a per-row task: the referee is
        # settling *boundaries*, and a boundary settled in batch 1 must bind batch 5.
        # Run in parallel, each batch decides the same boundary independently and
        # they disagree — which is how a rule set acquires two rules that fire on
        # the same trigger and give opposite answers. The cost is wall-clock on one
        # phase; the benefit is a rule set that is internally consistent, which is
        # the entire point of having a referee.
        verdicts: list[Any] = []
        decided: dict[frozenset[str], dict[str, str]] = {}
        ref_batches = list(_chunks(disagreements, 25))
        n_failed = 0

        for bn, chunk in enumerate(ref_batches, 1):
            try:
                vs = RefereeAgent(ctx).run(
                    disagreements=chunk, classes=classes_txt, rules=rules_txt,
                    decided=list(decided.values()),
                ).verdicts
            except Exception as exc:  # noqa: BLE001
                n_failed += 1
                deps.emit(f"  ⚠ referee batch {bn}/{len(ref_batches)} FAILED: {type(exc).__name__}")
                continue
            verdicts.extend(vs)
            src = {d["query"]: d for d in chunk}
            for v in vs:
                d = src.get(v.query)
                if not d:
                    continue
                key = frozenset((d["label_a"], d["label_b"]))
                decided.setdefault(key, {
                    "pair": " × ".join(sorted(key)),
                    "final": v.final_label,
                    "example": d["query"][:40],
                })
            if len(ref_batches) > 1:
                deps.emit(f"  referee batch {bn}/{len(ref_batches)} — "
                          f"{len(decided)} boundaries settled so far")
        by_q = {v.query: v for v in verdicts}
        for r in rows:
            if r.agreed:
                continue
            v = by_q.get(r.query)
            if v is None:
                # No verdict — the batch failed or the referee skipped the row.
                # Taking annotator A's label and stamping `adjudicated=True`
                # fabricates provenance: the row then reads as refereed, teaches
                # the classifier a label nobody adjudicated, and silently favours
                # A on every contested row. Leave it unresolved and visible.
                r.final, r.adjudicated = "", False
                r.referee_rationale = "no referee verdict — excluded from the gold set"
                continue
            r.final, r.rule_cited, r.referee_rationale, r.adjudicated = (
                v.final_label, v.rule_cited, v.rationale, True
            )
            if v.rule_gap and v.proposed_rule:
                rid = f"R{len(taxonomy.rules) + len(new_rules) + 1:03d}"
                new_rules.append(AdjudicationRule(
                    id=rid, when=v.proposed_rule, then=v.final_label,
                    rationale="drafted by the referee to close a gap this disagreement exposed",
                    classes=[r.label_a, r.label_b], added_in_round=1,
                    added_because=f"disagreement on {r.query!r}",
                ))
        n_unresolved = sum(1 for r in rows if not r.agreed and not r.adjudicated)
        new_rules = _dedupe_rules(new_rules, taxonomy, deps)
        deps.emit(f"  referee adjudicated {len(disagreements) - n_unresolved}/{len(disagreements)}, "
                  f"drafted {len(new_rules)} rules"
                  + (f", {n_unresolved} left unresolved ({n_failed} batch failures)"
                     if n_unresolved else ""))

    # Procedural memory: the rule set is now different from the one we started with.
    for rule in new_rules:
        deps.memory.remember_rule(rule)
    taxonomy.rules.extend(new_rules)

    kappa_trace.append({"round": 1, "kappa": agree["kappa"], "n": agree["n"],
                        "raw_agreement": agree["raw_agreement"], "sample": "initial"})

    # --- guide repair: the playbook's 达不到先修指南再重标 ---------------------
    # Measuring kappa once, before the referee's rules exist, can only ever score
    # the un-repaired guide. When it misses, decide the boundaries the referee
    # could not settle, fold everything into the guide, and re-annotate.
    repair_meta: dict[str, Any] = {}
    rounds = cfg.taxonomy.kappa_repair_rounds
    if rounds > 0 and agree["kappa"] < cfg.gates.kappa and not deps.registry.is_offline:
        repair_meta = _repair_guide_and_reannotate(
            deps, ctx, cfg, df, taxonomy, rows, idx, strata, kappa_trace, rounds
        )
        if repair_meta.get("agreement"):
            agree = repair_meta["agreement"]
            rows = repair_meta["rows"]
            idx = repair_meta["idx"]
            rules_txt = _render_rules(taxonomy)

    # --- round 2: active learning on the boundary -------------------------
    # The playbook asks for a second batch aimed at where the first round was
    # least certain. We fit a cheap character TF-IDF model on round 1 rather than
    # waiting for the embedding — the point is only to rank rows by how close
    # they sit to a decision boundary, and a sparse model ranks that fine.
    al_meta: dict[str, Any] = {}
    if cfg.taxonomy.active_learning_rounds > 0:
        try:
            # Re-render AFTER `taxonomy.rules.extend(new_rules)` above; the string
            # bound before adjudication does not contain the referee's rules, which
            # are precisely what round 2's annotators need.
            al_rows = _active_learning_round(
                deps, ctx, rows, df, classes_txt, _render_rules(taxonomy), taxonomy, idx
            )
            if al_rows:
                rows.extend(al_rows)
                al_meta = {"n_added": len(al_rows), "round": 2,
                           "selection": "lowest top1-top2 margin under a round-1 model"}
                deps.emit(f"  active learning added {len(al_rows)} boundary rows")
        except Exception as exc:  # noqa: BLE001
            # The playbook requires this round. Continuing is acceptable, but the
            # artifact must not read like it ran: a downstream reader seeing
            # `active_learning: {}` cannot tell "disabled" from "crashed".
            al_meta = {"n_added": 0, "round": 2, "status": "failed",
                       "error": f"{type(exc).__name__}: {exc}"}
            deps.emit(f"  ⚠ active-learning round FAILED (playbook-required): {exc}")

    gold_df = pd.DataFrame([r.model_dump() for r in rows])
    gold_ref = deps.store.put_table("gold", gold_df, fmt="csv", producer="p2b",
                                    summary=f"{len(rows)} gold rows, kappa {agree['kappa']:.3f}")
    agree_ref = deps.store.put_json(
        "gold_agreement",
        {"agreement": agree, "new_rules": [r.model_dump() for r in new_rules],
         "n_adjudicated": len(disagreements), "active_learning": al_meta,
         "kappa_trace": kappa_trace, "guide_repair": repair_meta.get("summary", {})},
        producer="p2b", summary=f"kappa {agree['kappa']:.3f}",
    )
    deps.store.put_json("taxonomy_v2", {"taxonomy": taxonomy.model_dump()}, producer="p2b",
                        summary="taxonomy with referee-drafted rules folded in")
    deps.cache_put("taxonomy_obj", taxonomy)
    deps.cache_put("gold_df", gold_df)

    # Coverage first: kappa is only a statement about annotator agreement if the
    # annotators actually answered. When a provider outage left one of them with
    # 199 of 600 rows, the gate reported "kappa 0.813" as though that were a
    # verdict on the labelling guide — it was a verdict on whichever rows
    # happened to survive. Too little coverage is neither a pass nor a fail; it
    # is an unusable measurement, and saying so is the honest outcome.
    n_sub = agree.get("n_submitted", agree["n"]) or 1
    coverage = agree["n"] / n_sub
    unsound = coverage < cfg.gates.min_annotation_coverage

    gate = deps.gate(
        "p2b_kappa", "p2b",
        passed=(not unsound) and agree["kappa"] >= cfg.gates.kappa,
        observed={"kappa": agree["kappa"], "raw_agreement": agree["raw_agreement"], "n": agree["n"],
                  "kappa_trace": [round(k["kappa"], 4) for k in kappa_trace],
                  "n_unscored_unlabelled": agree.get("n_unscored_unlabelled", 0),
                  "annotation_coverage": round(coverage, 4)},
        threshold={"kappa": cfg.gates.kappa},
        message=(
            (f"MEASUREMENT UNSOUND — only {agree['n']}/{n_sub} rows ({coverage:.0%}) were "
             f"labelled by both annotators; kappa {agree['kappa']:.3f} describes the "
             "rows that survived, not the guide. "
             if unsound else "")
            + f"kappa {agree['kappa']:.3f} on {agree['n']} double-annotated queries"
            + (" — NOTE: offline stand-ins are deterministic functions, so this number "
               "measures the stand-in, not annotator agreement" if deps.registry.is_offline else "")
        ),
        remediation=(
            ("Annotator coverage collapsed — check the run log for provider errors "
             "(auth, rate limits, timeouts) before reading anything into this number. "
             "Re-run the phase once the provider is healthy. "
             if unsound else "")
            + "Low kappa means the guide is ambiguous, not that the annotators are careless. "
            "Fold the referee's drafted rules into the guide and re-annotate before "
            "training anything on this gold set."
        ),
        warn_only=deps.registry.is_offline,
    )

    return {
        "phase": "p2c",
        "artifacts": {"gold": gold_ref, "gold_agreement": agree_ref},
        "gates": {gate.name: gate},
        "completed_phases": ["p2b"],
        "events": [f"P2b: kappa {agree['kappa']:.3f}, {len(new_rules)} rules drafted from disagreements"],
    }


def _pilot_agreement(deps: Deps, ctx: Any, df: Any, taxonomy: Taxonomy) -> dict[str, Any]:
    """A dry run of the annotation task, plus the ceiling it could possibly hit.

    Two annotators disagreeing tells you something is wrong; it does not tell you
    *what*. An ambiguous guide and an unreliable annotator produce the same number,
    and they have opposite remedies — redraft the guide, or change the annotator.

    So this also measures **self-consistency**: the same annotator, the same
    queries, re-asked in a different batch composition. That is the ceiling any two
    independent annotators could reach, because two annotators cannot agree with
    each other more reliably than one agrees with itself. Comparing the two numbers
    separates the cases:

      inter << intra   the guide has slack — redraft it, the annotator can do better
      inter ~= intra   the guide is as good as this annotator supports; a higher
                       bar needs a stronger model or human annotation, not more rules

    This is the piece that makes the phase portable. A fixed kappa bar imported
    from one project silently assumes the new corpus and the new annotator resemble
    the old ones — and on this corpus collapsing the taxonomy from 21 classes to 4
    moved kappa only 0.808 to 0.832, so the ceiling was never the taxonomy.
    """
    from collections import Counter

    cfg = deps.cfg
    n = cfg.taxonomy.pilot_sample_size
    idx = deterministic_subsample(len(df), n, cfg.seed_metric)
    queries = df[cfg.data.text_column].astype(str).iloc[idx].tolist()
    classes_txt, rules_txt = _render_classes(taxonomy), _render_rules(taxonomy)
    la = _annotate(ctx, "a", queries, classes_txt, rules_txt, taxonomy.labeling_guide, deps)
    lb = _annotate(ctx, "b", queries, classes_txt, rules_txt, taxonomy.labeling_guide, deps)
    agree = agreement([x["label"] for x in la], [x["label"] for x in lb])

    # The ceiling: annotator A again, same queries, shuffled so the batches differ.
    # The LLM cache is keyed on the rendered prompt, so a different batch
    # composition is a genuine second draw rather than a replay.
    order = rng(deps.cfg.seed_metric + 7).permutation(len(queries))
    la2_raw = _annotate(ctx, "a", [queries[i] for i in order],
                        classes_txt, rules_txt, taxonomy.labeling_guide, deps)
    back = [None] * len(queries)
    for pos, i in enumerate(order):
        back[i] = la2_raw[pos]["label"]
    self_agree = agreement([x["label"] for x in la], [str(v) for v in back])

    # Which pairs drove the disagreement — this is what a redraft needs to act on.
    conf = Counter(
        " × ".join(sorted((x["label"], y["label"])))
        for x, y in zip(la, lb) if x["label"] != y["label"]
    )
    # Chance agreement backed out of the pair, so the pilot can state the raw
    # agreement its own kappa target actually implies rather than a constant.
    po, kp, n = agree["raw_agreement"], agree["kappa"], max(1, agree["n"])
    pe = (po - kp) / (1 - kp) if kp < 1 else 0.0
    target = deps.cfg.gates.kappa
    raw_needed = target * (1 - pe) + pe
    # One-sided upper bound on kappa at this n; se via the normal approximation.
    se = ((po * (1 - po) / n) ** 0.5) / max(1e-6, 1 - pe)
    ceiling = self_agree.get("kappa")
    # How much of the gap to the ceiling the guide is responsible for. Near 1.0 the
    # guide is already extracting everything this annotator can give.
    headroom = None
    if ceiling and ceiling > 0:
        headroom = round(min(1.0, kp / ceiling), 4)
    return {
        "n": agree["n"],
        "self_consistency_kappa": ceiling,
        "self_consistency_raw": self_agree.get("raw_agreement"),
        "share_of_ceiling_reached": headroom,
        "ceiling_measured_on_real_annotator": not deps.registry.is_offline,
        "ceiling_verdict": (
            "guide has slack — inter-annotator agreement is well below what this "
            "annotator achieves against itself, so redrafting should help"
            if headroom is not None and headroom < 0.90 else
            "at the annotator ceiling — a higher bar needs a stronger model or human "
            "annotation, not more adjudication rules"
            if headroom is not None else
            "self-consistency unavailable"
        ),
        "raw_agreement": po,
        "kappa": kp,
        "kappa_se": round(se, 4),
        "kappa_upper": round(min(1.0, kp + 1.645 * se), 4),
        "chance_agreement": round(pe, 4),
        "raw_needed_for_target": round(raw_needed, 4),
        "top_confusions": [f"{k} ({v})" for k, v in conf.most_common(3)],
    }


def _repair_guide_and_reannotate(
    deps: Deps, ctx: Any, cfg: Any, df: Any, taxonomy: Taxonomy,
    rows: list[GoldRow], idx: Any, strata: list[str],
    kappa_trace: list[dict[str, Any]], rounds: int,
) -> dict[str, Any]:
    """Decide the open boundaries, rewrite the guide, annotate a fresh sample.

    The playbook's remedy for a missed kappa is to fix the guide and re-label,
    and the reason it works is worth stating: low kappa is a property of the
    *instructions*, not of the annotators. Two observations from the run that
    motivated this, both measured:

    * Writing rules only from boundaries the referee settled consistently tops
      out at kappa 0.898 — short of the 0.90 gate. The remaining disagreement
      lives in the dozen boundaries the referee itself resolved both ways, and
      no rule can be written for a boundary nobody has decided.
    * Deciding those boundaries explicitly reaches 0.955.

    So this does two things a plain "fold the rules in" pass does not: it finds
    the boundaries that are still open, and it settles each one from the rows
    both annotators already agreed on — evidence with no arbitration in it.

    The re-measurement uses a **fresh** sample by default. Re-scoring the rows
    the rules were derived from measures how well those rules fit those rows,
    which is not the question; the question is whether the guide got clearer.
    """
    from ...ops.classify import (
        boundary_default, contested_boundaries, discriminating_markers,
    )

    out: dict[str, Any] = {"summary": {}}
    for attempt in range(1, rounds + 1):
        open_pairs = contested_boundaries(rows)
        if not open_pairs:
            deps.emit("  guide repair: no boundary left open — nothing to decide")
            break

        decisions: list[dict[str, Any]] = []
        added: list[AdjudicationRule] = []
        for b in open_pairs:
            markers = discriminating_markers(rows, b["pair"])
            # Markers cannot reach queries that carry none — which on a Chinese
            # query log is most of the hard cases, a bare idiom naming its object
            # and not what the user wants to know about it. Where the agreed rows
            # lean decisively, add a tie-breaker for exactly those rows.
            fallback = boundary_default(rows, b["pair"], [m["marker"] for m in markers])
            if not markers and not fallback:
                decisions.append({"pair": b["pair"], "decided": False,
                                  "why": "no marker and no decisive default in the agreed rows"})
                continue
            if fallback:
                rid = f"R{len(taxonomy.rules) + len(added) + 1:03d}"
                added.append(AdjudicationRule(
                    id=rid,
                    when=(f"查询不含任何意图标记词, 且候选类目为 "
                          f"{b['pair'][0]} 或 {b['pair'][1]} (如四字成语单独出现)"),
                    then=fallback["then"],
                    rationale=(f"双标一致且无标记词的 {fallback['support']} 行中, "
                               f"{fallback['precision']:.0%} 落在 {fallback['then']} — "
                               "无标记行是分歧的主要来源, 必须有默认裁定"),
                    classes=list(b["pair"]), added_in_round=2, trigger="<no-marker>",
                    added_because="marker-less rows on an open boundary",
                ))
            for m in markers:
                rid = f"R{len(taxonomy.rules) + len(added) + 1:03d}"
                added.append(AdjudicationRule(
                    id=rid,
                    when=f"查询包含「{m['marker']}」且候选类目为 {b['pair'][0]} 或 {b['pair'][1]}",
                    then=m["then"],
                    rationale=(f"在双标一致的行中, 「{m['marker']}」判别该边界的精确率为 "
                               f"{m['precision']:.0%} (支持 {m['support']} 行) — "
                               "由已达成一致的证据裁定, 非再次征询模型意见"),
                    classes=list(b["pair"]), added_in_round=2, trigger=m["marker"],
                    added_because=f"referee resolved this boundary inconsistently: {b['resolved_as']}",
                ))
            decisions.append({"pair": b["pair"], "decided": True,
                              "markers": [m["marker"] for m in markers],
                              "default": fallback["then"] if fallback else None})

        settled = [d for d in decisions if d["decided"]]
        deps.emit(f"  guide repair {attempt}/{rounds}: {len(open_pairs)} open boundaries, "
                  f"{len(settled)} decided from agreed rows, {len(added)} rules added")
        if not added:
            out["summary"] = {"rounds_run": attempt, "decisions": decisions,
                              "outcome": "no boundary could be decided from the evidence"}
            break

        added = _dedupe_rules(added, taxonomy, deps)
        taxonomy.rules.extend(added)
        for rule in added:
            deps.memory.remember_rule(rule)
        taxonomy.labeling_guide = _guide_with_decisions(taxonomy.labeling_guide, decisions, added)

        # Fresh, disjoint sample — never the rows the rules were written from.
        # `stratified_sample` speaks positional indices, so everything here does
        # too; mixing in label-based `.loc` silently breaks on any frame whose
        # index is not a bare RangeIndex.
        if cfg.taxonomy.repair_on_fresh_sample:
            unseen = np.ones(len(df), dtype=bool)
            unseen[np.asarray(idx, dtype=np.int64)] = False
            pool_pos = np.flatnonzero(unseen)
            n_repair = cfg.taxonomy.gold_sample_size or gold_size_for(len(df), cfg.taxonomy)
            if len(pool_pos) < n_repair:
                deps.emit("  ⚠ too few unseen rows for a disjoint sample; "
                          "re-scoring overlaps round 1 and reads optimistic")
                pool_pos = np.arange(len(df))
            rel = stratified_sample(df.iloc[pool_pos], n_repair,
                                    strata_cols=strata, seed=cfg.seed_metric + attempt)
            new_idx = pool_pos[np.asarray(rel, dtype=np.int64)]
        else:
            new_idx = np.asarray(idx, dtype=np.int64)

        queries = df[cfg.data.text_column].astype(str).iloc[new_idx].tolist()
        classes_txt, rules_txt = _render_classes(taxonomy), _render_rules(taxonomy)
        deps.emit(f"  re-annotating {len(queries)} "
                  f"{'fresh' if cfg.taxonomy.repair_on_fresh_sample else 'original'} queries "
                  "under the repaired guide")
        la = _annotate(ctx, "a", queries, classes_txt, rules_txt, taxonomy.labeling_guide, deps)
        lb = _annotate(ctx, "b", queries, classes_txt, rules_txt, taxonomy.labeling_guide, deps)
        agree2 = agreement([x["label"] for x in la], [x["label"] for x in lb])
        # Read the previous kappa BEFORE appending, rather than indexing [-2]
        # afterwards: the caller owns the trace and may hand over an empty one.
        prev = kappa_trace[-1]["kappa"] if kappa_trace else agree2["kappa"]
        prev_n = kappa_trace[-1]["n"] if kappa_trace else agree2["n"]
        kappa_trace.append({
            "round": len(kappa_trace) + 1, "kappa": agree2["kappa"], "n": agree2["n"],
            "raw_agreement": agree2["raw_agreement"],
            "sample": "fresh" if cfg.taxonomy.repair_on_fresh_sample else "original",
        })
        deps.emit(f"  kappa after repair: {agree2['kappa']:.3f} on n={agree2['n']} "
                  f"({agree2['kappa'] - prev:+.3f} vs {prev:.3f} on n={prev_n}) — "
                  "两轮样本不同, 差值不可直接解读为指南改进")

        rows2 = [
            GoldRow(query=q, idx=int(new_idx[i]), label_a=la[i]["label"], label_b=lb[i]["label"],
                    final=la[i]["label"] if la[i]["label"] == lb[i]["label"] else "",
                    agreed=la[i]["label"] == lb[i]["label"],
                    rationale_a=la[i].get("rationale", ""), rationale_b=lb[i].get("rationale", ""),
                    round=attempt + 1, source="guide_repair")
            for i, q in enumerate(queries)
        ]
        out.update({"agreement": agree2, "rows": rows2, "idx": new_idx})
        out["summary"] = {
            "rounds_run": attempt, "decisions": decisions,
            "n_rules_added": len(added), "kappa_before": prev, "kappa_after": agree2["kappa"],
            "n_before": prev_n, "n_after": agree2["n"],
            "comparable": prev_n == agree2["n"],
            "sample": "fresh" if cfg.taxonomy.repair_on_fresh_sample else "original",
        }
        rows = rows2
        cov2 = agree2["n"] / (agree2.get("n_submitted") or agree2["n"] or 1)
        if cov2 < cfg.gates.min_annotation_coverage:
            deps.emit(f"  ⚠ 本轮仅 {agree2['n']}/{agree2.get('n_submitted')} 行被双标 "
                      f"({cov2:.0%}) — 该 kappa 不能与上一轮比较, 也不足以判定修订是否有效")
        elif agree2["kappa"] >= cfg.gates.kappa:
            deps.emit("  guide repair cleared the gate")
            break
    return out


def _guide_with_decisions(guide: str, decisions: list[dict[str, Any]],
                          added: list[AdjudicationRule]) -> str:
    """Append the boundary rulings to the guide, in the annotators' language."""
    lines = ["", "", "## 边界裁定 (第二轮修订)", "",
             "以下边界在第一轮中裁判自身给出了不一致的结果。现依据**双标一致行**的证据",
             "作出**具有约束力**的裁定; 遇到这些类目对时必须照此执行。", ""]
    for d in decisions:
        pair = " × ".join(d["pair"])
        if not d.get("decided"):
            lines.append(f"- {pair}: 证据不足, 仍为开放边界 — 如遇到请在 rationale 中说明理由。")
            continue
        rules = [r for r in added if set(r.classes) == set(d["pair"])]
        for r in rules:
            lines.append(f"- {pair}: {r.when} → **{r.then}**  ({r.rationale})")
    return (guide or "") + "\n".join(lines)


def _annotate(ctx: Any, which: str, queries: list[str], classes: str, rules: str, guide: str, deps: Deps) -> list[dict]:
    """Label every query, in independent batches.

    The batches are independent by construction — each carries its own copy of
    the guide and shares no state — so they run concurrently. Sequentially, a
    600-row gold set is 24 calls per annotator and dominates the wall clock of
    the entire pipeline for no methodological reason.
    """
    agent = AnnotatorAgent(ctx, suffix=f"_{which}")
    batches = list(_chunks(queries, 25))

    def _one(chunk: list[str]) -> dict[str, dict]:
        # Retry before giving up. A failed batch does not merely lose 25 rows —
        # they come back as UNLABELED and, before the scorer learned to exclude
        # them, were counted as disagreements. On one live run a provider auth
        # fault took 16 of 24 batches and left the phase measuring kappa on a
        # third of its sample. The provider SDK retries transport errors; this
        # retries the ones it gives up on, which is where outages land.
        last = ""
        for attempt in range(3):
            try:
                batch = agent.run(queries=chunk, classes=classes, rules=rules, guide=guide)
                if attempt:
                    deps.emit(f"  annotator[{which}] batch recovered on retry {attempt}")
                return {l.query: l.model_dump() for l in batch.labels}
            except Exception as exc:  # noqa: BLE001
                last = f"{type(exc).__name__}: {exc}"
                if attempt < 2:
                    time.sleep(2 ** attempt)
        deps.emit(f"  ⚠ annotator[{which}] batch lost {len(chunk)} rows after 3 attempts: {last[:110]}")
        return {}

    if len(batches) > 1 and not ctx.registry.is_offline:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=ctx.cfg.llm.max_concurrency) as pool:
            maps = list(pool.map(_one, batches))
    else:
        maps = [_one(b) for b in batches]

    got: dict[str, dict] = {}
    for m in maps:
        got.update(m)
    done = sum(1 for q in queries if q in got)
    deps.emit(f"  annotator[{which}] labelled {done}/{len(queries)}"
              + ("" if done == len(queries)
                 else f"  ⚠ {len(queries) - done} rows unlabelled — kappa will be "
                      "computed on the remainder and the shortfall reported"))
    return [
        got.get(q) or {"query": q, "label": UNLABELED, "rationale": "annotator omitted"}
        for q in queries
    ]


def _chunks(seq: list, n: int):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def _dedupe_rules(new_rules: list[AdjudicationRule], taxonomy: Taxonomy,
                  deps: Deps) -> list[AdjudicationRule]:
    """Drop duplicate rules and refuse genuinely contradictory ones.

    Every rule is rendered into both annotators' prompts, so redundancy costs
    prompt budget and a contradiction is actively harmful — one live run shipped
    a rule sending bare single characters to CHITCHAT_NOISE and another sending
    them to CHAR_PRONUNCIATION, manufacturing disagreement on a stratum that had
    been 78% agreed.

    Machine-generated rules are compared on their **structured key** — the class
    pair plus the exact trigger. An earlier version compared the rendered `when`
    sentence, which was a disaster: two markers for one boundary render as the
    same template differing by two characters out of forty-five (0.957 similar),
    so every legitimate discriminating pair looked like a contradiction and both
    halves were withheld. That shredded 32 of 41 rules on a live run — and
    precisely the informative ones, since a marker pair pointing at opposite
    classes is exactly what settling a boundary looks like.

    Referee-drafted rules carry no trigger, so they still fall back to text
    similarity; there is nothing structured to compare them on.
    """
    from difflib import SequenceMatcher

    def norm(t: str) -> str:
        return "".join(ch for ch in str(t).lower() if ch.isalnum() or ch > "\u4e00")

    def key(r: AdjudicationRule) -> tuple[frozenset[str], str] | None:
        return (frozenset(r.classes), r.trigger) if r.trigger and r.classes else None

    kept: list[AdjudicationRule] = []
    dropped_dup = contradictions = 0
    existing = list(taxonomy.rules)

    for rule in new_rules:
        k, t = key(rule), str(rule.then)
        clash = None
        for other in kept + existing:
            ok = key(other)
            if k and ok:
                # Both structured: they collide only on an identical trigger for
                # the identical boundary. Different markers on one boundary are
                # complementary, not conflicting.
                same = k == ok
            elif k or ok:
                continue          # one structured, one prose — not comparable
            else:
                same = SequenceMatcher(None, norm(rule.when), norm(other.when)).ratio() >= 0.85
            if same:
                clash = other
                break
        if clash is None:
            kept.append(rule)
        elif str(clash.then) == t:
            dropped_dup += 1
        else:
            contradictions += 1
            deps.emit(f"  ⚠ rule {rule.id} and {clash.id} fire on the same trigger and "
                      f"disagree ({t} vs {clash.then}) — both withheld, boundary left open")
            kept = [x for x in kept if x is not clash]

    if dropped_dup:
        deps.emit(f"  {dropped_dup} duplicate rule(s) dropped before reaching the guide")
    if contradictions:
        deps.emit(f"  ⚠ {contradictions} contradictory rule pair(s) withheld")
    deps.emit(f"  {len(kept)}/{len(new_rules)} new rules reach the guide")
    return kept



def _validate_rules(taxonomy: Taxonomy, deps: Deps) -> dict[str, Any]:
    """Every rule must resolve to a class that exists.

    Rules are rendered verbatim into both annotators' prompts, so a rule whose
    target is not a declared node instructs them to pick a label that cannot be
    chosen — and they then diverge on whatever they pick instead. A live run
    shipped `R12 → 选 EXOD_INFO` against a taxonomy declaring `EXAM_INFO`, and
    `EXAM_INFO × POLICY_REGULATION` became a top-five disagreement pair.

    A target one small edit from exactly one real class is repaired; anything
    else is dropped, because an unfollowable rule is worse than no rule. Both
    outcomes are recorded rather than fixed silently.
    """
    from difflib import SequenceMatcher

    codes = [n.code for n in taxonomy.nodes]
    repaired: list[dict[str, str]] = []
    dropped: list[dict[str, str]] = []
    kept: list[Any] = []
    for rule in taxonomy.rules:
        then = str(rule.then)
        if any(c in then for c in codes):
            kept.append(rule)
            continue
        # Compare on the bare token: `then` reads like "选 EXAM_INFO".
        token = max(then.replace("选", " ").split(), key=len, default="")
        ranked = sorted(((SequenceMatcher(None, token, c).ratio(), c) for c in codes),
                        reverse=True)
        # Repair only when one candidate *dominates*, which is a stronger test
        # than clearing a fixed cutoff: a typo has one obvious intended target
        # and everything else far behind (EXOD_INFO → EXAM_INFO scores .78 with
        # the runner-up at .48). Two plausible targets means we cannot know.
        best, second = ranked[0], (ranked[1] if len(ranked) > 1 else (0.0, ""))
        if best[0] >= 0.70 and best[0] - second[0] >= 0.20:
            rule.then = then.replace(token, best[1])
            repaired.append({"id": rule.id, "was": token, "now": best[1],
                             "similarity": round(best[0], 3)})
            kept.append(rule)
        else:
            dropped.append({"id": rule.id, "target": token,
                            "why": "names no declared class; no unambiguous match"})
    taxonomy.rules = kept
    for r in repaired:
        deps.emit(f"  ⚠ rule {r['id']}: target `{r['was']}` is not a class — repaired to `{r['now']}`")
    for r in dropped:
        deps.emit(f"  ⚠ rule {r['id']}: target `{r['target']}` names no class — rule dropped")
    return {"n_repaired": len(repaired), "n_dropped": len(dropped),
            "repaired": repaired, "dropped": dropped}


def _render_classes(t: Taxonomy) -> str:
    return "\n".join(
        f"- **{n.code}** ({n.name}): {n.definition}\n"
        f"    need: {n.user_need}\n"
        f"    yes: {n.positive_examples[:4]}\n"
        f"    no:  {n.negative_examples[:3]}"
        for n in t.nodes
    )


def _render_rules(t: Taxonomy) -> str:
    return "\n".join(f"- [{r.id}] when {r.when} → {r.then} ({r.rationale})" for r in t.rules)


# ==========================================================================
# 2c / 2d — classifier and adversarial validation
# ==========================================================================

def p2c_classifier(state: PipelineState, deps: Deps) -> dict[str, Any]:
    """Rules first, then a linear head over dense ⊕ sparse ⊕ rule-flag features."""
    cfg = deps.cfg
    df = deps.df
    gold: pd.DataFrame = deps._cache.get("gold_df")
    if gold is None:
        gold = deps.load("gold")
    labelled = gold[gold["final"].astype(str).str.len() > 0]
    deps.emit(f"P2c classifier — {len(labelled)} labelled rows")

    if len(labelled) < 30 or labelled["final"].nunique() < 2:
        gate = deps.gate("p2c_trainable", "p2c", passed=False,
                         observed={"n_labelled": len(labelled), "n_classes": int(labelled["final"].nunique())},
                         threshold={"min_rows": 30, "min_classes": 2},
                         message="not enough usable gold to train a top-down classifier",
                         remediation="Increase gold_sample_size, or check that annotation produced real labels.",
                         warn_only=True)
        return {"phase": "p3", "gates": {gate.name: gate}, "completed_phases": ["p2c"],
                "events": ["P2c: skipped — insufficient gold"]}

    dense = deps.load("emb_base") if deps.has("emb_base") else deps.embedding("emb_base")
    rows = labelled["idx"].to_numpy()

    rules = [
        {"pattern": g["pattern"], "label": "", "name": g["name"]}
        for g in deps.load("template_groups")["groups"]
    ]
    engine = RuleEngine(rules, precision_floor=cfg.taxonomy.rule_precision_floor)
    rule_feats = engine.feature_matrix(df[cfg.data.text_column].astype(str).tolist())

    X, scaler = build_features(df, dense, rule_features=rule_feats)
    result = train_classifier(
        X[rows], labelled["final"].tolist(), seed=cfg.seed_metric,
        weights=df["weight"].to_numpy()[rows] if "weight" in df.columns else None,
    )
    deps.emit(f"  CV accuracy {result['cv_accuracy']:.3f}, macro-F1 {result['macro_f1']:.3f}, "
              f"ECE {result['ece']:.3f}")

    model_ref = deps.store.put_model(
        "topdown_model", {"model": result["model"], "scaler": scaler, "engine": engine},
        producer="p2c", summary=f"linear head, CV acc {result['cv_accuracy']:.3f}",
    )
    preds = result["model"].predict(X)
    # The bottom-up route delivers a margin and an ambiguity flag per row; the
    # top-down route delivered a bare code, so a reader comparing the two at row
    # level could only interrogate one of them. The probabilities exist already —
    # the calibration error above is computed from them — they were simply not
    # carried out of this node.
    proba = result["model"].predict_proba(X)
    top2 = np.sort(proba, axis=1)[:, -2:]
    td_conf = top2[:, -1]
    td_margin = top2[:, -1] - top2[:, -2]
    # Rules-first is the designed order, so which rows a rule decided is
    # provenance a reader needs: a rule-fired row is auditable against a cited
    # rule, a model-predicted row is not.
    rule_labels, rule_fired = engine.apply(df[cfg.data.text_column].astype(str).tolist())

    pred_ref = deps.store.put_table(
        "topdown_labels",
        pd.DataFrame({
            cfg.data.text_column: df[cfg.data.text_column], "l1_pred": preds,
            "confidence": np.round(td_conf, 4), "margin": np.round(td_margin, 4),
            "decided_by": np.where(rule_fired, "rule", "model"),
            "rule_label": rule_labels,
        }),
        producer="p2c", summary="full-corpus top-down labels, with confidence and provenance",
    )
    deps.cache_put("topdown_confidence", td_conf)
    deps.cache_put("topdown_margin", td_margin)
    deps.cache_put("topdown_decided_by", np.where(rule_fired, "rule", "model"))
    metrics_ref = deps.store.put_json(
        "topdown_metrics", {k: v for k, v in result.items() if k != "model"},
        producer="p2c", summary="classifier quality",
    )
    deps.cache_put("topdown_preds", preds)
    deps.cache_put("topdown_features", X)

    decision = deps.decision(
        "p2c", "Which model family for the top-down classifier?",
        "linear head on dense ⊕ surface ⊕ rule-flag features",
        "Tree ensembles must reconstruct directional similarity from axis-aligned splits "
        "and spread a small per-class boosting budget across many classes; a linear head "
        "reads the embedding geometry directly.",
        decided_by="metric",
        evidence={"cv_accuracy": result["cv_accuracy"], "macro_f1": result["macro_f1"], "ece": result["ece"]},
        rejected=[{"option": "gradient-boosted trees on raw embedding coordinates",
                   "why_rejected": "collapses on this feature geometry"}],
    )
    return {
        "phase": "p2d",
        "artifacts": {"topdown_model": model_ref, "topdown_labels": pred_ref, "topdown_metrics": metrics_ref},
        "decisions": [decision],
        "completed_phases": ["p2c"],
        "events": [f"P2c: CV accuracy {result['cv_accuracy']:.3f}, ECE {result['ece']:.3f}"],
    }


def p2d_validate(state: PipelineState, deps: Deps) -> dict[str, Any]:
    """Adversarial validation: attack the labels, and count how often the attack fails.

    An agent asked "is this right?" agrees with whatever it is shown.  An agent
    asked to prove the label wrong finds the real errors, and the rate at which
    it *fails* is an honest accuracy estimate rather than a rubber stamp.
    """
    cfg = deps.cfg
    df = deps.df
    preds = deps.recover(
        "topdown_preds", "topdown_labels",
        rebuild=lambda df: df["l1_pred"].to_numpy(),
    )
    if preds is None:
        deps.emit("P2d — skipped, no top-down predictions")
        return {"phase": "p3", "completed_phases": ["p2d"], "events": ["P2d: skipped"]}

    taxonomy: Taxonomy = deps.taxonomy()
    idx = deterministic_subsample(len(df), min(150, len(df)), cfg.seed_metric)
    rows = [{"query": str(df[cfg.data.text_column].iloc[i]), "label": str(preds[i])} for i in idx]
    deps.emit(f"P2d adversarial validation — attacking {len(rows)} predicted labels")

    # Step 1 of the playbook's 2d: scan the gold set for rows whose embedding
    # neighbourhood disagrees with their label. Flags go to review, never to an
    # automatic relabel — see knn_label_scan's docstring for why that matters.
    scan: dict[str, Any] = {}
    try:
        from ...ops.classify import knn_label_scan

        X_base = deps.load("emb_base") if deps.has("emb_base") else None
        gold = deps._cache.get("gold_df")
        if gold is None and deps.has("gold"):
            gold = deps.load("gold")
        if X_base is not None and gold is not None:
            g = gold[gold["final"].astype(str).str.len() > 0]
            if len(g):
                scan = knn_label_scan(X_base, g["idx"].tolist(), g["final"].tolist())
                deps.emit(
                    f"  kNN label scan: {scan['n_flagged']}/{scan['n_scanned']} rows flagged "
                    f"for REVIEW (never auto-relabelled)"
                )
    except Exception as exc:  # noqa: BLE001
        deps.emit(f"  kNN label scan skipped: {exc}")

    ctx = deps.agent_ctx()
    results: list[Any] = []
    for chunk in _chunks(rows, 30):
        out = AdversaryAgent(ctx).run(rows=chunk, classes=_render_classes(taxonomy) if taxonomy else "")
        results.extend(out.results)

    verdicts = [r.verdict for r in results]
    n = max(len(verdicts), 1)
    wrong = sum(v == "wrong" for v in verdicts)
    defensible = sum(v == "defensible" for v in verdicts)
    est = 1 - wrong / n
    deps.emit(f"  survived attack: {est:.3f} ({wrong} wrong, {defensible} defensible of {n})")

    ref = deps.store.put_json(
        "adversarial_validation",
        {
            "n_attacked": n, "n_wrong": wrong, "n_defensible": defensible,
            "estimated_accuracy": round(est, 4),
            "knn_label_scan": scan,
            "results": [r.model_dump() for r in results[:200]],
            "method": (
                "agents were instructed to PROVE each label wrong; the estimate is the "
                "share of labels whose attack failed, not a self-reported confidence"
            ),
        },
        producer="p2d", summary=f"adversarial accuracy estimate {est:.3f}",
    )
    return {
        "phase": "p3",
        "artifacts": {"adversarial_validation": ref},
        "completed_phases": ["p2d"],
        "events": [f"P2d: adversarial accuracy estimate {est:.3f} over {n} attacked labels"],
    }


# ==========================================================================
# 2e — sub-intent layer
# ==========================================================================

def p2e_subintents(state: PipelineState, deps: Deps) -> dict[str, Any]:
    """Audit each L1's geometry, subdivide it, and test whether L2 is learnable.

    The audit runs first and is the part worth reading. It answers, per class,
    "can the representation see this intent at all?" — and the answer is
    routinely no for pragmatic intents, whose meaning lives in what the user
    wants rather than in how they phrased it. Those classes are not defective;
    they simply have to draw their accuracy from the rule layer, and knowing
    that before training saves a cycle of blaming the gold set.
    """
    from ...ops.subintent import geometric_audit, subdivide

    cfg = deps.cfg
    df = deps.df
    preds = deps.recover(
        "topdown_preds", "topdown_labels", rebuild=lambda d: d["l1_pred"].to_numpy()
    )
    if preds is None:
        deps.emit("P2e — skipped, no top-down labels")
        return {"phase": "p4", "completed_phases": ["p2e"], "events": ["P2e: skipped"]}

    X = deps.load("emb_base") if deps.has("emb_base") else None
    if X is None:
        deps.emit("P2e — skipped, no base embedding")
        return {"phase": "p4", "completed_phases": ["p2e"], "events": ["P2e: skipped"]}

    deps.emit(f"P2e sub-intents — auditing {len(set(preds.tolist()))} L1 classes")
    audit = geometric_audit(X, preds, seed=cfg.seed_metric)
    n_rule_dep = len(audit.get("rule_dependent_classes", []))
    deps.emit(f"  {n_rule_dep} class(es) are rule-dependent (kNN agreement below threshold)")

    div = subdivide(X, preds, seed=cfg.seed_metric)
    sub = div["sub_labels"]
    deps.emit(f"  subdivided into {div['n_sub_intents']} sub-intents")

    # Is the L2 layer learnable? Reported as distillability, never as truth.
    learnability = _l2_learnability(X, sub, seed=cfg.seed_metric)

    out = pd.DataFrame({
        cfg.data.text_column: df[cfg.data.text_column],
        "td_l1": preds,
        "td_l2": sub,
    })
    labels_ref = deps.store.put_table("topdown_l2_labels", out, producer="p2e",
                                      summary=f"{div['n_sub_intents']} sub-intents")
    meta_ref = deps.store.put_json(
        "subintents",
        {"geometric_audit": audit, "subdivision": div["per_class"],
         "n_sub_intents": div["n_sub_intents"], "learnability": learnability},
        producer="p2e", summary=f"{div['n_sub_intents']} sub-intents, "
                                f"{n_rule_dep} rule-dependent L1 classes",
    )
    deps.cache_put("topdown_sub", sub)

    events = [f"P2e: {div['n_sub_intents']} sub-intents across {len(set(preds.tolist()))} L1 classes"]
    if n_rule_dep:
        events.append(
            f"P2e: {n_rule_dep} L1 class(es) invisible to the embedding — "
            f"{audit['rule_dependent_classes'][:4]} must rely on the rule layer"
        )
    decision = deps.decision(
        "p2e", "Which L1 classes can the representation actually carry?",
        f"{len(audit['classes']) - n_rule_dep} geometry-visible, {n_rule_dep} rule-dependent",
        "Measured by k-nearest-neighbour label agreement per class. A class whose "
        "neighbourhood does not share its label is not a region of the embedding space, "
        "and more gold labels will not make it one.",
        decided_by="metric", evidence={"audit": audit["classes"][:12]},
        decisive_metrics=["knn_agreement"],
    )
    return {
        "phase": "p4",
        "artifacts": {"topdown_l2_labels": labels_ref, "subintents": meta_ref},
        "decisions": [decision],
        "completed_phases": ["p2e"],
        "events": events,
    }


def _l2_learnability(X: np.ndarray, sub: np.ndarray, *, seed: int, folds: int = 3) -> dict[str, Any]:
    """Cross-validated accuracy of predicting the sub-intent from the embedding."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    y = np.asarray(sub)
    counts = pd.Series(y).value_counts()
    keep = np.isin(y, counts[counts >= folds].index)
    if keep.sum() < folds * 2 or len(set(y[keep].tolist())) < 2:
        return {"accuracy": None, "note": "too few sub-intents with support to cross-validate"}
    idx = deterministic_subsample(int(keep.sum()), min(12000, int(keep.sum())), seed)
    Xs, ys = X[keep][idx], y[keep][idx]
    counts2 = pd.Series(ys).value_counts()
    keep2 = np.isin(ys, counts2[counts2 >= folds].index)
    Xs, ys = Xs[keep2], ys[keep2]
    if len(set(ys.tolist())) < 2:
        return {"accuracy": None, "note": "insufficient class support after sub-sampling"}
    cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
    acc = float(cross_val_score(LogisticRegression(max_iter=1500, C=4), Xs, ys, cv=cv).mean())
    return {
        "accuracy": round(acc, 4),
        "n_classes": int(len(set(ys.tolist()))),
        "n": int(len(ys)),
        "caveat": (
            "This measures how LEARNABLE the sub-intents are from the representation — "
            "their distillability — NOT their agreement with human judgment. A high number "
            "here and a low kNN agreement in the audit above are not a contradiction: a "
            "linear head can separate what a neighbourhood vote cannot."
        ),
    }


def _active_learning_round(
    deps: Deps, ctx: Any, rows: list[GoldRow], df: pd.DataFrame,
    classes_txt: str, rules_txt: str, taxonomy: Taxonomy, already: np.ndarray,
) -> list[GoldRow]:
    """Annotate a second batch drawn from the decision boundary.

    Rows near a boundary are where an extra label buys the most: the model has
    no opinion there, so each one resolves a genuine ambiguity rather than
    confirming something already settled. Rows the model is confident about
    teach it almost nothing, which is why random second batches disappoint.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    from ...ops.classify import select_active_learning_batch

    cfg = deps.cfg
    labelled = [r for r in rows if r.final]
    if len(labelled) < 40 or len({r.final for r in labelled}) < 2:
        return []

    texts = df[cfg.data.text_column].astype(str).tolist()
    vec = TfidfVectorizer(analyzer="char", ngram_range=tuple(cfg.domain.char_ngram_range), min_df=2)
    Xall = vec.fit_transform(texts)
    tr = np.array([r.idx for r in labelled], dtype=np.int64)
    yt = np.array([r.final for r in labelled])
    counts = pd.Series(yt).value_counts()
    keep = np.isin(yt, counts[counts >= 2].index)
    if keep.sum() < 20 or len(set(yt[keep].tolist())) < 2:
        return []

    model = LogisticRegression(max_iter=1500, C=2).fit(Xall[tr[keep]], yt[keep])
    # The cosine diversity pass works on sparse input (normalize and `@` both do);
    # it was disabled to route around the `len(sparse)` crash one frame down, which
    # aborted the call before this branch could ever run. With that fixed the
    # documented behaviour is restored: uncertainty sampling alone collapses onto
    # one confusing region and re-annotates fifty variants of the same edge case.
    sel = select_active_learning_batch(
        model, Xall, already_labelled=already.tolist(),
        batch=cfg.taxonomy.active_learning_batch, seed=cfg.seed_metric,
    )
    picks = sel["selected"]
    if not picks:
        return []

    queries = [str(df[cfg.data.text_column].iloc[i]) for i in picks]
    la = _annotate(ctx, "a", queries, classes_txt, rules_txt, taxonomy.labeling_guide, deps)
    lb = _annotate(ctx, "b", queries, classes_txt, rules_txt, taxonomy.labeling_guide, deps)
    out: list[GoldRow] = []
    for j, q in enumerate(queries):
        a2, b2 = la[j]["label"], lb[j]["label"]
        out.append(GoldRow(
            query=q, idx=int(picks[j]), label_a=a2, label_b=b2,
            final=a2 if a2 == b2 else "", agreed=a2 == b2,
            rationale_a=la[j].get("rationale", ""), rationale_b=lb[j].get("rationale", ""),
            round=2, source="active_learning",
        ))
    return out
