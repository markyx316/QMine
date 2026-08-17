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
from ...determinism import deterministic_subsample, rng
from ...ops.audit import stratified_sample
from ...ops.classify import RuleEngine, agreement, build_features, train_classifier
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
    submissions = []
    for i, angle in enumerate(angles):
        evidence = _evidence_for_angle(angle["key"], df, audit, templates, risk, cfg, seed=i)
        agent = ResearcherAgent(ctx, suffix=f"_{angle['key']}")
        sub = agent.run(
            assignment=angle["assignment"], evidence=evidence, domain_notes=cfg.domain.domain_notes
        )
        sub.angle = angle["key"]
        submissions.append(sub)
        deps.emit(f"  researcher[{angle['key']}] → {len(sub.candidates)} candidates")

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

    n_l1 = len(taxonomy.l1()) or len(taxonomy.nodes)
    lo, hi = cfg.taxonomy.l1_target_range
    gate = deps.gate(
        "p2a_taxonomy_shape",
        "p2a",
        passed=(lo <= n_l1 <= hi) and len(taxonomy.rules) >= cfg.taxonomy.min_adjudication_rules,
        observed={"n_l1": n_l1, "n_rules": len(taxonomy.rules), "critic_verdict": critique.verdict},
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
        "gates": {gate.name: gate},
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
    idx = stratified_sample(df, cfg.taxonomy.gold_sample_size, strata_cols=strata, seed=cfg.seed_metric)
    queries = df[cfg.data.text_column].astype(str).iloc[idx].tolist()
    deps.emit(f"P2b gold — {len(queries)} queries, double-blind annotation")

    classes_txt = _render_classes(taxonomy)
    rules_txt = _render_rules(taxonomy)

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
        verdicts: list[Any] = []
        for chunk in _chunks(disagreements, 25):
            out = RefereeAgent(ctx).run(disagreements=chunk, classes=classes_txt, rules=rules_txt)
            verdicts.extend(out.verdicts)
        by_q = {v.query: v for v in verdicts}
        for r in rows:
            if r.agreed:
                continue
            v = by_q.get(r.query)
            if v is None:
                r.final, r.adjudicated = r.label_a, True
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
        deps.emit(f"  referee adjudicated {len(disagreements)}, drafted {len(new_rules)} new rules")

    # Procedural memory: the rule set is now different from the one we started with.
    for rule in new_rules:
        deps.memory.remember_rule(rule)
    taxonomy.rules.extend(new_rules)

    # --- round 2: active learning on the boundary -------------------------
    # The playbook asks for a second batch aimed at where the first round was
    # least certain. We fit a cheap character TF-IDF model on round 1 rather than
    # waiting for the embedding — the point is only to rank rows by how close
    # they sit to a decision boundary, and a sparse model ranks that fine.
    al_meta: dict[str, Any] = {}
    if cfg.taxonomy.active_learning_rounds > 0:
        try:
            al_rows = _active_learning_round(
                deps, ctx, rows, df, classes_txt, rules_txt, taxonomy, idx
            )
            if al_rows:
                rows.extend(al_rows)
                al_meta = {"n_added": len(al_rows), "round": 2,
                           "selection": "lowest top1-top2 margin under a round-1 model"}
                deps.emit(f"  active learning added {len(al_rows)} boundary rows")
        except Exception as exc:  # noqa: BLE001
            deps.emit(f"  active-learning round skipped: {exc}")

    gold_df = pd.DataFrame([r.model_dump() for r in rows])
    gold_ref = deps.store.put_table("gold", gold_df, fmt="csv", producer="p2b",
                                    summary=f"{len(rows)} gold rows, kappa {agree['kappa']:.3f}")
    agree_ref = deps.store.put_json(
        "gold_agreement",
        {"agreement": agree, "new_rules": [r.model_dump() for r in new_rules],
         "n_adjudicated": len(disagreements), "active_learning": al_meta},
        producer="p2b", summary=f"kappa {agree['kappa']:.3f}",
    )
    deps.store.put_json("taxonomy_v2", {"taxonomy": taxonomy.model_dump()}, producer="p2b",
                        summary="taxonomy with referee-drafted rules folded in")
    deps.cache_put("taxonomy_obj", taxonomy)
    deps.cache_put("gold_df", gold_df)

    gate = deps.gate(
        "p2b_kappa", "p2b",
        passed=agree["kappa"] >= cfg.gates.kappa,
        observed={"kappa": agree["kappa"], "raw_agreement": agree["raw_agreement"], "n": agree["n"]},
        threshold={"kappa": cfg.gates.kappa},
        message=(
            f"kappa {agree['kappa']:.3f} on {agree['n']} double-annotated queries"
            + (" — NOTE: offline stand-ins are deterministic functions, so this number "
               "measures the stand-in, not annotator agreement" if deps.registry.is_offline else "")
        ),
        remediation=(
            "Low kappa means the guide is ambiguous, not that the annotators are careless. "
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


def _annotate(ctx: Any, which: str, queries: list[str], classes: str, rules: str, guide: str, deps: Deps) -> list[dict]:
    agent = AnnotatorAgent(ctx, suffix=f"_{which}")
    out: list[dict] = []
    for chunk in _chunks(queries, 25):
        batch = agent.run(queries=chunk, classes=classes, rules=rules, guide=guide)
        got = {l.query: l.model_dump() for l in batch.labels}
        for q in chunk:
            out.append(got.get(q) or {"query": q, "label": "UNLABELED", "rationale": "annotator omitted"})
    return out


def _chunks(seq: list, n: int):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


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
    pred_ref = deps.store.put_table(
        "topdown_labels",
        pd.DataFrame({cfg.data.text_column: df[cfg.data.text_column], "l1_pred": preds}),
        producer="p2c", summary="full-corpus top-down labels",
    )
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
    from ...ops.subintent import geometric_audit, sub_samples, subdivide

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
    dense_like = Xall  # sparse is fine: predict_proba only needs the matrix
    sel = select_active_learning_batch(
        model, dense_like, already_labelled=already.tolist(),
        batch=cfg.taxonomy.active_learning_batch, seed=cfg.seed_metric,
        diversity_fraction=0.0,   # sparse rows: skip the cosine diversity pass
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
