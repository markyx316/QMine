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

import threading

import json
import re
from typing import Any

import time

import numpy as np
import pandas as pd

from ...agents.roles import (
    AdversaryAgent,
    AnnotatorAgent,
    ArchitectAgent,
    CriticAgent,
    RESEARCH_ANGLES,
    RefereeAgent,
    ResearcherAgent,
    RuleWriterAgent,
    TaxonomyRedrawAgent,
)
from ...config import gold_size_for
from ...determinism import deterministic_subsample, rng
from ...ops.audit import stratified_sample
from ...ops.classify import UNLABELED, RuleEngine, agreement, build_features, train_classifier
from . import observe as _observe
from ...records import AdjudicationRule, GoldRow, Taxonomy, TaxonomyNode
from ...records import Prescription
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

    # REUSE A TAXONOMY INSTEAD OF RE-DERIVING ONE.
    #
    # The web-using researchers are not deterministic: asked the same question
    # twice they return different candidate sets, which changes the architect's
    # prompt, which changes every class and rule below it. Every annotator prompt
    # then differs too, so a resumed run misses the cache on all 3,000 gold rows
    # and re-pays for annotation it already bought. That cascade has broken every
    # resume this project has attempted.
    #
    # Pointing at a finished `taxonomy.json` cuts it at the source: identical
    # classes and rules mean byte-identical annotator prompts, so the gold set
    # replays for free and only the phases after it actually run.
    reuse = getattr(cfg.taxonomy, "reuse_taxonomy_from", None)
    if reuse:
        reused = _load_taxonomy_for_reuse(deps, str(reuse))
        n_l1 = sum(1 for n in reused.nodes if n.level == 1)
        deps.emit(f"P2a taxonomy — REUSED from {reuse}: {n_l1} L1 intents, "
                  f"{len(reused.rules)} rules; researchers, architect and critic skipped")
        tax_ref = deps.store.put_json(
            "taxonomy", {"taxonomy": reused.model_dump(), "submissions": [],
                         "critique": {"verdict": "reused", "findings": []},
                         "dropped_candidates": [], "redraw_history": [],
                         "reused_from": str(reuse)},
            producer="p2a", summary=f"reused: {n_l1} L1 intents, {len(reused.rules)} rules")
        deps.cache_put("taxonomy_obj", reused)
        return {
            "phase": "p2b",
            "artifacts": {"taxonomy": tax_ref},
            "completed_phases": ["p2a"],
            "events": [f"P2a: taxonomy reused from {reuse} "
                       f"({n_l1} L1 intents, {len(reused.rules)} rules)"],
        }

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
        # WHETHER THIS ANGLE ACTUALLY SEARCHED IS PROVENANCE, NOT A LOG DETAIL.
        # `used_tools` reached the log line below and nothing else, so
        # `taxonomy.json` could not distinguish an angle that ran twelve web
        # searches from one whose tool loop died and answered from parametric
        # knowledge. On live38 that was 2 of 5 angles; on live39, 1 of 5. A
        # taxonomy described as web-researched should be able to say which parts
        # of it were.
        try:
            sub.web_researched = bool(getattr(agent, "used_tools", False))
        except (AttributeError, ValueError):
            pass
        # An angle that returns nothing is a paid-for perspective missing from the
        # design record, and it is silent: the phase still announces five
        # researchers fanning out. One live run had `literature` return zero while
        # the other four returned 6-12, and nothing anywhere noticed.
        if not sub.candidates:
            deps.emit(f"  ⚠ researcher[{angle['key']}] returned NO candidates — "
                      "this angle contributed nothing to the taxonomy")
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
    nodes = draft.nodes or _fallback_nodes(submissions)

    # Rules are written in a second call, shown the finalised class list. Asking one
    # call for both blew a 42,000-token ceiling, and hardening the rule requirement
    # inside that single call made it return two classes instead of nineteen. The
    # split removes both failures structurally: each call fits, and a writer that
    # can see the class list cannot invent a tie-break between classes that do not
    # exist. The architect's own rules are kept and merged, so this only ever adds.
    rules = list(draft.rules)
    try:
        extra = RuleWriterAgent(ctx).run(
            nodes=nodes, domain_notes=cfg.domain.domain_notes,
            min_rules=cfg.taxonomy.min_adjudication_rules,
        )
        have = {(str(r.when), str(r.then)) for r in rules}
        rules += [r for r in extra.rules if (str(r.when), str(r.then)) not in have]
        deps.emit(f"  rule writer → {len(extra.rules)} rules over "
                  f"{extra.pairs_considered or len(nodes)} class pairs "
                  f"({len(rules)} total after merge)")
    except Exception as exc:  # noqa: BLE001
        # The architect's own rules stand. The shape gate below decides whether
        # what survived is enough to annotate against.
        deps.emit(f"  ⚠ rule writer failed ({type(exc).__name__}) — "
                  f"proceeding with the architect's {len(rules)} rule(s)")

    taxonomy = Taxonomy(
        nodes=nodes,
        rules=rules,
        labeling_guide=draft.labeling_guide,
        notes=draft.design_notes,
    )

    sample = df[cfg.data.text_column].astype(str).iloc[
        deterministic_subsample(len(df), 120, cfg.seed_metric)
    ].tolist()
    critique = CriticAgent(ctx).run(taxonomy=taxonomy, sample_queries=sample)
    deps.emit(f"  critic → {len(critique.findings)} findings, verdict {critique.verdict}")

    rule_health = _validate_rules(taxonomy, deps)
    # Count what an annotator can actually cite, which is what the gate is for.
    # `len(taxonomy.rules)` counts only the top-level list; per-class rules now
    # render too, and one live taxonomy carried 55 of them behind a reported "1".
    n_rules = len(_render_rules(taxonomy).splitlines())

    # Playbook 2a quality gate: 50 queries, two independent annotators, and a
    # hard stop below 85% agreement — "一致率 <85% 则回炉改指南/裁决规则, 而非直接开标".
    # The point is economic as much as methodological. Annotating the full gold
    # set costs hundreds of calls; this costs four, and it asks the same question.
    # Discovering an ambiguous guide here rather than after 3,000 annotations is
    # the difference between a cheap redraft and an expensive one.
    # Pilot, and if it finds boundaries that are not in the data, redraw those
    # boundaries and pilot again. The pilot has always known which pairs were
    # structural; three runs printed that and halted, and a human redrew by hand.
    # A round costs one redraw plus one pilot against a 3,000-row gold set, so
    # spending two of them to avoid annotating against a broken taxonomy is
    # straightforwardly cheaper than not.
    pilot = _pilot_agreement(deps, ctx, df, taxonomy)
    taxonomy, pilot, redraw_history = _redraw_until_stable(
        deps, ctx, df, taxonomy, pilot)

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
        and pilot.get("self_consistency_kappa") is not None
        and not pilot.get("slack_is_significant", False)
    )
    reaches_target = pilot["kappa_upper"] >= cfg.gates.kappa

    # The slack test exists to TRIGGER a remedy, and the pipeline has one: redraw
    # the boundaries the annotator cannot reproduce, then re-pilot. Once that has
    # run and failed to improve agreement, the slack is no longer something this
    # pipeline can act on — and halting then asks the operator to do by hand what
    # it just tried and could not. Measured on `live35`: kappa 0.844 against a
    # 0.9243 ceiling, the redraw rewrote the six worst boundaries and kappa FELL
    # to 0.827, so the redraw was reverted and the slack stood.
    #
    # At that point the question is only whether the labels can support
    # conclusions, and kappa 0.844 is above the reliability floor by any
    # convention. So: proceed, loudly, with the residual slack recorded — rather
    # than stop with nothing delivered and no remaining move.
    redraw_attempted = bool(redraw_history)
    redraw_helped = any(r.get("kept") for r in redraw_history)
    remedy_exhausted = _remedy_is_exhausted(
        redraw_history, cfg.taxonomy.max_taxonomy_redraws)
    usable_despite_slack = (
        remedy_exhausted and pilot["kappa"] >= cfg.gates.annotator_fitness_kappa
    )

    # Is the ANNOTATOR fit to apply this taxonomy at all? That is the question an
    # absolute floor can answer, and self-consistency is the quantity it describes.
    # The playbook's 0.90 was measured on a project whose annotators reached 0.966;
    # ours self-agree at 0.883, so requiring 0.90 *between* two of them demanded
    # more reliability than one of them has. Enforcing it made the gate unpassable:
    # a perfect guide would still have halted, every time, forever.
    ceiling = pilot.get("self_consistency_kappa")
    annotator_fit = (
        deps.registry.is_offline               # the stand-in is not an annotator
        or ceiling is None
        or ceiling >= cfg.gates.annotator_fitness_kappa
    )
    pilot_gate = deps.gate(
        "p2a_pilot_agreement", "p2a",
        # Two independent conditions, each answering a different question.
        # Fitness: can this annotator apply this taxonomy reproducibly? If not, no
        # guide work helps and the run must stop. Slack: is the guide extracting
        # what the annotator can give? If there is significant slack, redraft —
        # that is cheap here and ruinous after 3,000 gold rows are paid for.
        passed=annotator_fit and (reaches_target or at_ceiling or usable_despite_slack),
        observed={"kappa": pilot["kappa"], "kappa_upper_95": pilot["kappa_upper"],
                  "raw_agreement": pilot["raw_agreement"], "n": pilot["n"],
                  "raw_agreement_implied_by_target": pilot["raw_needed_for_target"],
                  "annotator_self_consistency_kappa": pilot.get("self_consistency_kappa"),
                  "share_of_ceiling_reached": pilot.get("share_of_ceiling_reached"),
                  "recoverable_slack": pilot.get("recoverable_slack"),
                  "recoverable_slack_se": pilot.get("recoverable_slack_se"),
                  "target_above_ceiling": pilot.get("target_above_ceiling"),
                  "annotator_fit": annotator_fit,
                  "ceiling_verdict": pilot.get("ceiling_verdict", ""),
                  "redraw_attempted": redraw_attempted,
                  "redraw_helped": redraw_helped,
                  "proceeded_with_residual_slack": usable_despite_slack,
                  "at_annotator_ceiling": at_ceiling},
        threshold={"annotator_fitness_kappa": cfg.gates.annotator_fitness_kappa,
                   "playbook_aspiration_kappa": cfg.gates.kappa,
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
                 + (f" — NOTE the target {cfg.gates.kappa} is ABOVE this annotator's own "
                    f"ceiling of {pilot['self_consistency_kappa']}, so it is unreachable "
                    f"by any guide; the recoverable slack is "
                    f"{pilot.get('recoverable_slack')} ± {pilot.get('recoverable_slack_se')}"
                    if pilot.get("target_above_ceiling") else "")
                 + (f" — PROCEEDING WITH RESIDUAL SLACK of "
                    f"{pilot.get('recoverable_slack')}: the redraw ran "
                    + ("and improved agreement but could not close the slack"
                       if redraw_helped else "and did not improve agreement")
                    + f", so this pipeline has no remaining move; "
                    f"kappa {pilot['kappa']:.3f} is above the "
                    f"{cfg.gates.annotator_fitness_kappa} reliability floor, and every "
                    f"downstream number must be read against this gap"
                    if usable_despite_slack else "")
                 + (f" — top confusions {pilot['top_confusions']}" if pilot["top_confusions"] else "")),
        remediation=(
            # STATIC per branch. This used to be `ceiling_verdict + ". The guide
            # is ambiguous…"`, and `prose()` matches with `startswith`, so a
            # dynamic prefix meant the key could never match and the single most
            # important remediation in the pipeline reached a Chinese reader in
            # English. The verdict itself is DATA — it travels in `observed`.
            "The guide is ambiguous before a single gold row has been paid for. Fix the "
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
        # The class-count range warns; the rule floor BLOCKS. They are not the same
        # kind of finding. A count slightly outside the expected range is a judgement
        # call about granularity, but a taxonomy without tie-breaks cannot be
        # annotated consistently by anyone — measured directly: 19 classes with one
        # rule produced inter-annotator kappa 0.761 against an annotator that agreed
        # with itself at 0.900. Every point of that gap was missing rules, and the
        # pilot cost real money to discover what this gate can see for free.
        passed=(lo <= n_l1 <= hi) and n_rules >= cfg.taxonomy.min_adjudication_rules,
        observed={"n_l1": n_l1, "n_rules": n_rules, "n_rules_top_level": len(taxonomy.rules), "critic_verdict": critique.verdict,
                  "rules_repaired": rule_health["n_repaired"], "rules_dropped": rule_health["n_dropped"]},
        threshold={"l1_range": [lo, hi], "min_rules": cfg.taxonomy.min_adjudication_rules},
        message=f"{n_l1} L1 intents, {n_rules} adjudication rules the annotator can cite",
        remediation=(
            "Too few classes and a catch-all swells until it means nothing; too many and "
            "annotators stop agreeing, which destroys the gold set. Too few rules means "
            "the referee has nothing to cite and adjudication becomes taste."
        ),
        # Three outcomes, not two. Missing rules block, because a taxonomy without
        # tie-breaks cannot be annotated by anyone. A class count *near* the range
        # only warns, because granularity is a judgement call the data may
        # legitimately settle differently. But a count wildly outside it is not a
        # judgement call — it is broken output: one draft returned two classes for
        # a 15-25 target, and its two dozen rules named a dozen classes that did
        # not exist, so all of them were discarded as unfollowable.
        #
        # `n_rules` is read after validation, so rules dropped for naming a class
        # that does not exist count as missing. That is exactly right here.
        warn_only=(n_rules >= cfg.taxonomy.min_adjudication_rules
                   and lo / 2 <= n_l1 <= hi * 1.5),
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
            # What the pilot made us change, and whether it helped. Without this
            # the delivered taxonomy silently differs from the one the architect
            # designed, and the report cannot say why.
            "redraw_history": redraw_history,
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

    # A halted run used to name what failed and prescribe nothing — three halts,
    # `n_prescriptions: 0` each time, an operator told the taxonomy is wrong and
    # not which part. The pilot already knows: it measured which pairs one
    # annotator cannot reproduce against itself (redraw the boundary — a tie-break
    # cannot rescue a distinction the query does not contain) and which pairs two
    # annotators split on while each stayed self-consistent (write the tie-break).
    # A prescription's FORCE depends on what the gate decided. Blocked, these are
    # the work that must happen before the run may continue. Passed with residual
    # slack, the same measurements are a recorded LIMITATION — regions where the
    # intent is not determinable from the query — and telling an operator to
    # "merge them" prescribes work the pipeline already attempted, measured, and
    # deliberately declined to act on. Three live runs printed the imperative on
    # a gate that had already passed.
    proceeding = _gate_let_the_run_proceed(pilot_gate)
    structural_advice = (
        "The run PROCEEDED with this slack recorded, so this is a limitation to "
        "report, not an action outstanding: read every downstream number for "
        "these classes against it."
        if proceeding else
        "Merge them, or re-cut them on one basis of division; an adjudication "
        "rule cannot fix a distinction the query lacks."
    )
    guide_advice = (
        "The run PROCEEDED, so the referee's own rules are where this gets "
        "settled; no separate action is outstanding at 2a."
        if proceeding else
        "This is what an adjudication rule is for."
    )
    prescriptions: list[Prescription] = []
    for pair, count in pilot.get("structural_confusions", []):
        classes = [c.strip() for c in pair.split("×")]
        prescriptions.append(Prescription(
            id=deps.next_prescription_id(), kind="merge_families",
            target_names=classes, proposed_by="p2a_pilot",
            rationale=(f"{count} of {pilot['n']} pilot queries land differently when the "
                       f"SAME annotator is re-asked, so the boundary between {pair} is not "
                       f"in the data. {structural_advice}"),
        ))
    for pair, count in pilot.get("guide_confusions", []):
        classes = [c.strip() for c in pair.split("×")]
        prescriptions.append(Prescription(
            id=deps.next_prescription_id(), kind="flag_risk",
            target_names=classes, proposed_by="p2a_pilot",
            rationale=(f"{count} pilot disagreements on {pair}, but each annotator was "
                       f"self-consistent — the boundary exists and is merely unstated. "
                       f"{guide_advice}"),
        ))
    if prescriptions:
        kind = "recorded limitations" if proceeding else "outstanding actions"
        deps.emit(f"  prescriptions ({kind}): "
                  f"{sum(1 for p in prescriptions if p.kind == 'merge_families')} "
                  f"structural (redraw), "
                  f"{sum(1 for p in prescriptions if p.kind == 'flag_risk')} guide (tie-break)")

    # Read the taxonomy from the artifact that was just written rather than from
    # local variables: the observer must see what p2a DELIVERED, and a payload
    # assembled by hand drifts from it the moment either side changes.
    _obs = _observe(deps, "p2a",
                    {"taxonomy": deps.load("taxonomy") if deps.has("taxonomy") else {},
                     "pilot": pilot, "prescriptions": [x.model_dump() for x in prescriptions]},
                    decisions=[decision])
    return {
        "phase": "p2b",
        "artifacts": {"taxonomy": tax_ref},
        "gates": {gate.name: gate, pilot_gate.name: pilot_gate, **_obs},
        "decisions": [decision],
        "prescriptions": prescriptions,
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

    labels_a, labels_b = _annotate_both(
        ctx, queries, classes_txt, rules_txt, taxonomy.labeling_guide, deps)

    agree = agreement([l["label"] for l in labels_a], [l["label"] for l in labels_b])
    deps.emit(f"  raw agreement {agree['raw_agreement']:.3f}, kappa {agree['kappa']:.3f}, "
              f"{agree['n_disagreements']} disagreements")

    rows: list[GoldRow] = []
    n_missing = 0
    for i, q in enumerate(queries):
        la, lb = labels_a[i]["label"], labels_b[i]["label"]
        # A row NOBODY LABELLED is missing data, not agreement. `agreement()`
        # already excludes UNLABELED from kappa, but this construction did not:
        # when both annotators omitted the same row, `la == lb == UNLABELED`
        # satisfied the equality, so the row was recorded agreed with
        # `final="UNLABELED"`, never reached the referee, and passed p2c's
        # non-empty filter into the classifier AS A CLASS. The metric was
        # protected and the gold set was not.
        missing = UNLABELED in (la, lb)
        if missing:
            n_missing += 1
        rows.append(GoldRow(
            query=q, idx=int(idx[i]), label_a=la, label_b=lb,
            final="" if missing else (la if la == lb else ""),
            agreed=(not missing) and la == lb,
            rationale_a=labels_a[i].get("rationale", ""), rationale_b=labels_b[i].get("rationale", ""),
        ))
    if n_missing:
        deps.emit(f"  ⚠ {n_missing}/{len(queries)} rows missing a label from at least "
                  f"one annotator — excluded from the gold set, not counted as agreement")

    new_rules: list[AdjudicationRule] = []
    # A row one annotator never labelled is not a disagreement either — there is
    # no second opinion to adjudicate, and sending it to the referee spends a
    # frontier call asking it to choose between a class and "UNLABELED".
    disagreements = [
        {"query": r.query, "label_a": r.label_a, "label_b": r.label_b,
         "rationale_a": r.rationale_a, "rationale_b": r.rationale_b}
        for r in rows if not r.agreed and UNLABELED not in (r.label_a, r.label_b)
    ]
    if disagreements:
        # BATCHED BY CLASS PAIR, which is what makes them independent.
        #
        # This was chunked by row position and run strictly sequentially, for a
        # real reason: adjudication is not a per-row task — the referee settles
        # *boundaries*, and if the same boundary turns up in batch 1 and batch 5
        # they decide it independently and the rule set acquires two rules that
        # fire on the same trigger with opposite answers.
        #
        # But that argument only requires ordering WITHIN a boundary, never
        # across boundaries. Rows on `A × B` must be settled together; rows on
        # `C × D` are independent of them. Packing each pair ENTIRELY into one
        # batch removes the hazard at the source — a pair can no longer span two
        # batches, so no two batches can contradict each other — and the batches
        # become concurrent without giving anything up.
        #
        # It also shrinks each response, which is the other half of the problem:
        # live38's first 25-row batch emitted 34,099 tokens and failed to parse
        # with 144,000 tokens of room available. Size was the defect, not the cap.
        # Sequentially it cost ~10-11 min per batch, ~5-6h for a 3,000-row gold
        # set, making the referee the wall-clock bottleneck of the pipeline.
        verdicts: list[Any] = []
        decided: dict[frozenset[str], dict[str, str]] = {}
        ref_groups = _batch_by_class_pair(disagreements, target=15)
        n_batches = sum(len(g) for g in ref_groups)
        n_failed = 0
        n_pairs = len({frozenset((d["label_a"], d["label_b"])) for d in disagreements})
        deps.emit(f"  referee: {len(disagreements)} disagreements over {n_pairs} class "
                  f"pairs → {len(ref_groups)} independent groups, {n_batches} calls")

        def _adjudicate(part: list[dict[str, Any]], prior: Any = ()) -> Any:
            """One referee call. Returns verdicts, or None if the call failed.

            `decided` is deliberately NOT passed any more. It existed so a later
            batch would honour a boundary an earlier one had settled — a real
            need when a pair could span batches, and unnecessary now that it
            cannot. Passing the running dict here would also make each prompt
            depend on which sibling batches happened to finish first, so the same
            run would send different prompts on a replay and miss its own cache:
            the resume cascade, re-created inside one phase.
            """
            try:
                return RefereeAgent(ctx).run(
                    disagreements=part, classes=classes_txt, rules=rules_txt,
                    decided=prior,
                ).verdicts
            except Exception as exc:  # noqa: BLE001
                deps.emit(f"    referee call on {len(part)} rows failed: {type(exc).__name__}")
                return None

        _fold_lock = threading.Lock()

        def _fold(vs: Any, part: list[dict[str, Any]]) -> None:
            """Record the verdicts and the boundaries they settle.

            Folded per call, not per batch, so a bisected half still binds the
            halves and batches that follow it.
            """
            src = {d["query"]: d for d in part}
            with _fold_lock:
                verdicts.extend(vs)
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

        def _one_group(gn_group: tuple[int, list[list[dict[str, Any]]]]) -> int:
            """Run one group's chunks IN ORDER; groups themselves are independent."""
            gn, group = gn_group
            failed = 0
            for ci, chunk in enumerate(group):
                # A multi-chunk group is ONE oversized pair, split because it will
                # not fit in a single call. Its later chunks must honour the ruling
                # its earlier ones made, or the split re-creates exactly the
                # contradiction this batching exists to prevent. Deterministic
                # here, unlike across groups, because these run in order.
                prior: Any = ()
                if ci and len(group) > 1:
                    key = frozenset((chunk[0]["label_a"], chunk[0]["label_b"]))
                    with _fold_lock:
                        got = decided.get(key)
                    prior = (got,) if got else ()
                failed_here, recovered = _run_batch_with_bisect(
                    lambda part, _p=prior: _adjudicate(part, _p), _fold, chunk)
                failed += failed_here
                if failed_here or recovered < len(chunk):
                    deps.emit(f"  ⚠ referee group {gn}/{len(ref_groups)} lost rows; "
                              f"split recovered {recovered}/{len(chunk)}")
            deps.emit(f"  referee group {gn}/{len(ref_groups)} done — "
                      f"{len(decided)} boundaries settled so far")
            return failed

        # Concurrent ACROSS groups, sequential within one. Each group owns its
        # boundaries outright, so no two groups can rule against each other. This
        # phase was the pipeline's wall-clock bottleneck at ~10-11 min per
        # sequential call — measured on live38's own 524 disagreements, 87 pairs
        # become ~31 calls, which is ~5.7h serially and well under an hour here.
        if len(ref_groups) > 1 and not deps.registry.is_offline:
            from concurrent.futures import ThreadPoolExecutor

            workers = max(1, min(cfg.llm.max_concurrency, len(ref_groups)))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                n_failed += sum(pool.map(_one_group, enumerate(ref_groups, 1)))
        else:
            n_failed += sum(_one_group(g) for g in enumerate(ref_groups, 1))
        by_q = {v.query: v for v in verdicts}
        # The class list the referee was given. Anything outside it is a
        # malformed verdict, not a new class.
        valid_codes = {n.code for n in taxonomy.nodes if getattr(n, "code", None)}
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
            snapped, note = _snap_label_to_taxonomy(v.final_label, valid_codes)
            if not snapped:
                # Not a class in this taxonomy. Writing it through creates a
                # one-row phantom class that later reads as "too rare to
                # cross-validate" — a wrong explanation for a malformed verdict.
                deps.emit(f"  ⚠ referee returned {v.final_label!r}, {note} — row left unresolved")
                r.final, r.adjudicated = "", False
                r.referee_rationale = f"referee verdict rejected: {note}"
                continue
            if note:
                deps.emit(f"  referee label repaired: {note} → {snapped}")
            r.final, r.rule_cited, r.referee_rationale, r.adjudicated = (
                snapped, v.rule_cited, v.rationale, True
            )
            if v.rule_gap and v.proposed_rule:
                rid = f"R{len(taxonomy.rules) + len(new_rules) + 1:03d}"
                new_rules.append(AdjudicationRule(
                    id=rid, when=v.proposed_rule, then=snapped,
                    rationale="drafted by the referee to close a gap this disagreement exposed",
                    classes=[r.label_a, r.label_b], added_in_round=1,
                    added_because=f"disagreement on {r.query!r}",
                ))
        # Count ONLY rows the referee was actually given. `not r.agreed` also
        # matches rows excluded for missing a label, which never entered
        # `disagreements` — subtracting them from 483 under-reported coverage on
        # both sides of the fraction (450/483 when 459 rows were adjudicated).
        n_unresolved = sum(1 for r in rows if not r.agreed and not r.adjudicated
                           and UNLABELED not in (r.label_a, r.label_b))
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
    # Snapshot what the repair is allowed to overwrite, so a repair that makes
    # agreement worse can be undone rather than merely regretted.
    guide_before_repair = taxonomy.labeling_guide
    rules_before_repair = list(taxonomy.rules)
    if rounds > 0 and agree["kappa"] < cfg.gates.kappa and not deps.registry.is_offline:
        repair_meta = _repair_guide_and_reannotate(
            deps, ctx, cfg, df, taxonomy, rows, idx, strata, kappa_trace, rounds
        )
        if repair_meta.get("agreement"):
            # A REPAIR THAT LOWERS AGREEMENT IS A REPAIR TO DISCARD — the same
            # rule the redraw has had since it was built, and for the same
            # reason: keeping a change because it is newer is how a loop walks a
            # guide downhill. Measured on live38 gen05, the first run in which
            # the repair's own rules actually reached the annotator (before that
            # they were truncated out of the prompt): kappa 0.822 -> 0.794, about
            # 3.5 standard errors at n≈2978, and BELOW the 0.80 reliability
            # floor — so the repair turned a gate that would have passed into a
            # halt. Adding 112 tie-breaks cannot resolve a query that carries no
            # marker, but it can give two readers more ways to justify differing.
            #
            # The fresh ROWS are kept either way: they are real annotations and
            # more gold data is still gold data. It is the GUIDE that is unproven,
            # so that is what reverts.
            before_k = agree["kappa"]
            after_k = repair_meta["agreement"]["kappa"]
            if after_k < before_k:
                deps.emit(f"  ⚠ guide repair lowered kappa {before_k:.3f} → {after_k:.3f} "
                          f"— reverting the guide and rules; the fresh rows are kept")
                taxonomy.labeling_guide = guide_before_repair
                taxonomy.rules = list(rules_before_repair)
                repair_meta["reverted"] = True
                repair_meta["kappa_reverted_from"] = round(after_k, 4)
            else:
                agree = repair_meta["agreement"]
            # MERGE, do not replace. With `repair_on_fresh_sample` (the default)
            # round 2 annotates DIFFERENT queries, so swapping the lists threw
            # away all ~3,000 round-1 rows INCLUDING every referee verdict — the
            # entire adjudication phase, discarded silently — and substituted a
            # set whose only labelled rows are the ones both annotators already
            # agreed on, because the referee runs before repair and never sees
            # round 2. The gold set became agreement-only, i.e. systematically
            # the easy rows, and every classifier number computed from it reads
            # high for that reason alone.
            fresh_rows = repair_meta["rows"]
            if cfg.taxonomy.repair_on_fresh_sample:
                seen_idx = {r.idx for r in rows}
                added_rows = [r for r in fresh_rows if r.idx not in seen_idx]
                kept = sum(1 for r in rows if r.adjudicated)
                rows = rows + added_rows
                idx = np.concatenate([np.asarray(idx, dtype=np.int64),
                                      np.asarray([r.idx for r in added_rows], dtype=np.int64)])
                deps.emit(f"  guide repair added {len(added_rows)} fresh rows "
                          f"(gold set now {len(rows)}); {kept} refereed rows kept")
            else:
                # Same queries re-labelled: the newer labels supersede, but say so
                # — the referee's verdicts on those rows go with them.
                dropped = sum(1 for r in rows if r.adjudicated)
                rows = fresh_rows
                idx = repair_meta["idx"]
                if dropped:
                    deps.emit(f"  ⚠ guide repair re-labelled the SAME sample — "
                              f"{dropped} referee verdicts superseded")
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

    # THE SAME QUESTION p2a ALREADY LEARNED TO ASK. This gate was
    # `kappa >= cfg.gates.kappa`, a flat bar, while p2a — on the same corpus and
    # the same annotators — reasons about the annotator's own ceiling. On
    # `live38` that produced an incoherent pair: p2a PASSED at kappa 0.824
    # (89.3% of a 0.9228 ceiling) and p2b HALTED at 0.832 (90.2%). The HIGHER
    # number blocked. A 0.90 bar there demands 97.5% of what one annotator
    # manages against ITSELF, which is the condition p2a's own comment describes
    # as making a gate unpassable by a perfect guide.
    #
    # So ask p2a's three questions here too: did we reach the target, is the
    # target above the ceiling, or is the remedy spent with agreement still above
    # the reliability floor? And keep failing when the annotator itself is unfit,
    # which no amount of guide work can repair.
    pilot_gate = (state.get("gates") or {}).get("p2a_pilot_agreement")
    pilot_obs = getattr(pilot_gate, "observed", {}) or {}
    pilot_ceiling = pilot_obs.get("annotator_self_consistency_kappa")

    share_of_ceiling = (round(agree["kappa"] / pilot_ceiling, 4)
                        if pilot_ceiling else None)
    target_above_ceiling = bool(pilot_ceiling and cfg.gates.kappa > pilot_ceiling)
    # The remedy here is the guide repair, not the redraw. It is spent when every
    # configured round has run; `kappa_trace` holds the initial round plus one
    # entry per repair.
    repair_exhausted = (max(0, len(kappa_trace) - 1) >= cfg.taxonomy.kappa_repair_rounds
                        and not deps.registry.is_offline)
    usable_despite_slack = (repair_exhausted
                            and agree["kappa"] >= cfg.gates.annotator_fitness_kappa)
    annotator_fit = (deps.registry.is_offline or pilot_ceiling is None
                     or pilot_ceiling >= cfg.gates.annotator_fitness_kappa)
    p2b_pass = annotator_fit and (
        agree["kappa"] >= cfg.gates.kappa or target_above_ceiling or usable_despite_slack)


    gate = deps.gate(
        "p2b_kappa", "p2b",
        passed=(not unsound) and p2b_pass,
        observed={"kappa": agree["kappa"], "raw_agreement": agree["raw_agreement"], "n": agree["n"],
                  "kappa_trace": [round(k["kappa"], 4) for k in kappa_trace],
                  "n_unscored_unlabelled": agree.get("n_unscored_unlabelled", 0),
                  "annotation_coverage": round(coverage, 4),
                  "annotator_self_consistency_kappa": pilot_ceiling,
                  "share_of_ceiling_reached": share_of_ceiling,
                  "target_above_ceiling": target_above_ceiling,
                  "repair_rounds_run": max(0, len(kappa_trace) - 1),
                  "repair_exhausted": repair_exhausted,
                  "proceeded_with_residual_slack": usable_despite_slack},
        threshold={"kappa": cfg.gates.kappa},
        message=(
            (f"MEASUREMENT UNSOUND — only {agree['n']}/{n_sub} rows ({coverage:.0%}) were "
             f"labelled by both annotators; kappa {agree['kappa']:.3f} describes the "
             "rows that survived, not the guide. "
             if unsound else "")
            + f"kappa {agree['kappa']:.3f} on {agree['n']} double-annotated queries"
            + (f"; annotator self-consistency {pilot_ceiling} "
               f"({share_of_ceiling} of ceiling reached)" if pilot_ceiling else "")
            + (f" — NOTE the target {cfg.gates.kappa} is ABOVE this annotator's own "
               f"ceiling of {pilot_ceiling}, so no guide can reach it"
               if target_above_ceiling else "")
            + (f" — PROCEEDING WITH RESIDUAL SLACK: the guide repair ran its "
               f"{cfg.taxonomy.kappa_repair_rounds} configured round(s) and agreement "
               f"is {agree['kappa']:.3f}, above the "
               f"{cfg.gates.annotator_fitness_kappa} reliability floor but short of "
               f"{cfg.gates.kappa}; every downstream number must be read against "
               f"that gap" if usable_despite_slack and agree["kappa"] < cfg.gates.kappa
               else "")
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

    # PERSIST THE TAXONOMY THE REFEREE AND THE REPAIR ACTUALLY PRODUCED.
    # `taxonomy.rules.extend(new_rules)` and the repaired `labeling_guide` were
    # applied to the in-memory object only, and p2b returned no taxonomy
    # artifact — so `runs/live38/gen02/taxonomy.json` still showed 50 rules and a
    # guide with no 边界裁定 section while the run had 133 rules and a rewritten
    # guide in hand. On a resumed run `deps.taxonomy()` recovered the PRE-referee
    # version, and the deliverable reported the wrong rule count. `taxonomy()`
    # already prefers `taxonomy_v2`; it simply was never written.
    # MEASURE which rules actually contradict each other on THIS corpus.
    # `_dedupe_rules` can only compare rules structurally: identical trigger on an
    # identical pair, or 85% text similarity for prose. It cannot see two rules
    # that are worded differently, fire on overlapping queries, and disagree —
    # which is the case that reaches an annotator as two instructions for one row.
    # Measured on live39: R018/R019 co-fire on 301 rows and point opposite ways,
    # while R006/R007 co-fire on 4 and are a legitimate discriminating pair. The
    # difference is only visible by running the triggers over the corpus.
    from ...ops.rule_conflict import find_conflicts

    _codes = [n.code for n in taxonomy.nodes if getattr(n, "code", None)]
    conflicts = find_conflicts(
        taxonomy.rules, deps.df[cfg.data.text_column].astype(str).tolist(), codes=_codes)
    if conflicts.overlaps:
        deps.emit(f"  ⚠ {len(conflicts.overlaps)} rule pair(s) fire on the SAME rows and "
                  "disagree — the boundary needs a tie-break for that region:")
        for o in conflicts.overlaps[:3]:
            deps.emit(f"      {o.rule_a}/{o.rule_b} on {' x '.join(o.classes)}: "
                      f"{o.n_both:,} rows, e.g. {o.examples[:2]}")
    if conflicts.crowded_pairs:
        deps.emit(f"  {len(conflicts.crowded_pairs)} class pair(s) carry many two-way "
                  "rules — the boundary itself is contested:")
        for c in conflicts.crowded_pairs[:3]:
            deps.emit(f"      {' x '.join(c['classes'])}: {c['n_rules']} rules "
                      f"({c['n_with_trigger']} measurable)")

    # HOLD THE RULES TO THE VERDICTS THAT PRODUCED THEM.
    #
    # 77% of the shipped rules carry no executable trigger, and that is not an
    # omission to prompt away: the referee's rules are semantic ("when the query
    # is a proverb and the user wants its moral"), and a regex demanded from one
    # would be a fabricated predicate that makes every overlap it reports
    # meaningless. What a rule with no predicate CAN still be measured against is
    # the gold set — every rule names a class pair, and the referee's own
    # adjudications on that pair are already in hand.
    #
    # live39 is why this exists. On `OTHER x TEXT_INTERPRETATION` the referee
    # ruled TEXT_INTERPRETATION on 15 of 21 rows, and drafted FIVE rules saying a
    # query with no intent marker goes to OTHER — the same principle restated at
    # five different disagreements, each one the opposite of what it had just
    # done on the rows in front of it. The referee contradicted its own verdicts,
    # all five rules shipped in the guide the annotators then used, and nothing
    # could see it: `_dedupe_rules` compares wording, `find_conflicts` needs
    # triggers, and `crowded_pairs` can only say "6 rules, two directions".
    from ...ops.rule_conflict import rules_against_evidence

    ev_report = rules_against_evidence(
        taxonomy.rules, rows, deps.df[cfg.data.text_column].astype(str).tolist(),
        codes=_codes)
    deps.emit(f"  rule triggers: {ev_report.n_lexical} executable, "
              f"{ev_report.n_semantic} semantic"
              + (f" ({ev_report.n_rejected_triggers} rejected as unusable)"
                 if ev_report.n_rejected_triggers else ""))
    # ARE THE TWO ANNOTATORS COMPARABLE? Kappa cannot answer that, and it is the
    # question kappa's meaning depends on: a capability gap between the two shows
    # up as disagreement no guide fix can close, and p2a's pilot then reads it as
    # a structural confusion and redraws boundaries that were never the problem.
    #
    # Discovered by hand two runs late, by grepping gold sets. Measured now.
    from ...ops.annotator_balance import annotator_balance

    # `route_for`, not a guessed accessor. The referee's model id is the one
    # field that makes a BALANCED result interpretable — a win-rate near 50%
    # means the annotators are matched OR the adjudicator is deciding at chance,
    # and only the referee separates those. Leaving it blank on an exception
    # would quietly remove the distinction.
    _ref_model = ""
    try:
        _routed = deps.registry.route_for("referee")
        _ref_model = f"{_routed[0]}:{_routed[1]}" if _routed else "offline-stand-in"
    except Exception as _exc:  # noqa: BLE001
        deps.emit(f"  ⚠ could not record the referee's model ({type(_exc).__name__}) — "
                  "a balanced annotator result cannot be told from a chance adjudicator")
    bal = annotator_balance(rows, _ref_model)
    if bal.undecidable:
        deps.emit(f"  annotator balance: UNDECIDABLE — only {bal.n_decided} contested "
                  f"row(s) adjudicated, {bal.MIN_DECIDED} needed to say anything")
    else:
        deps.emit(f"  annotator balance: a won {bal.a_won}/{bal.n_decided} "
                  f"({bal.a_share:.1%}, z={bal.z:+.1f}) of contested rows"
                  + (" — LOPSIDED" if bal.lopsided else ""))
    bal_gate = deps.gate(
        "p2b_annotator_symmetry", "p2b",
        passed=not bal.lopsided,
        observed=bal.as_record(),
        threshold={"rule": "|z| <= 3 for annotator_a's win-rate against an even split",
                   "why_z_not_a_share": "the same 60/40 split is noise on 40 rows and "
                                        "decisive on 400"},
        message=(f"UNDECIDABLE — only {bal.n_decided} contested row(s) were adjudicated, "
                 f"{bal.MIN_DECIDED} needed. This is NOT a finding that the annotators "
                 "are comparable; it is an absence of evidence either way"
                 if bal.undecidable else
                 f"the two annotators are comparable — a won {bal.a_share:.1%} of "
                 f"{bal.n_decided} contested rows (z={bal.z:+.1f})"
                 if not bal.lopsided else
                 f"the annotators are NOT comparable — a won {bal.a_share:.1%} of "
                 f"{bal.n_decided} contested rows (z={bal.z:+.1f}), so kappa is measuring "
                 "the gap between them rather than the clarity of the guide"),
        remediation=(
            "DIAGNOSTIC ONLY — do not choose a model from this number. It cannot "
            "separate a genuine capability gap from a referee deciding at chance, "
            "which is why the referee's own model id is recorded beside it. Model "
            "choice needs an independent capability evaluation, which this pipeline "
            "cannot produce; `capable_models` in the config is where that human "
            "judgement is recorded.\n\n"
            "Match the two annotators. A gap between them is measured as guide "
            "ambiguity by construction, and the p2a pilot will spend redraws on "
            "boundaries that were never the problem. Also check the REFEREE before "
            "trusting a balanced result: a win-rate near 50% means the annotators are "
            "matched OR that the adjudicator is deciding at chance, and only the "
            "referee's own capability separates those. Measured on this project: the "
            "same annotator pair read 78.3% under glm-5.2 and 55.1% under glm-4.5-airx."),
        warn_only=True,
    )

    vac = ev_report.vacuous_grounds
    for gr in vac:
        deps.emit(f"  ⚠ {' x '.join(gr.classes)}: the rules name "
                  f"{', '.join(gr.markers[:4])} as the discriminator, and "
                  f"{gr.n_matching}/{gr.n_rows} adjudicated rows contain any of them — "
                  f"the stated test does not separate this boundary "
                  f"(rules: {', '.join(gr.rules_citing[:5])})")
    ev_gate = deps.gate(
        "p2b_rules_match_their_evidence", "p2b",
        passed=not vac,
        observed=ev_report.as_record(),
        threshold={"rule": ("a boundary fails when the discriminator its own rules "
                            "enumerate falls on ONE side for every adjudicated row — a "
                            "test that divides nothing")},
        message=(
                 # ONE contiguous literal: `prose()` matches a literal prefix, and
                 # `test_no_translation_key_is_dead` scans the source for it — a key
                 # split across two f-string lines exists in neither place.
                 "every boundary's stated discriminator actually divides its adjudicated rows"
                 f" ({len(ev_report.stated_grounds)} tested; {ev_report.n_lexical} "
                 f"lexical / {ev_report.n_semantic} semantic rules)"
                 if not vac else
                 "the discriminator these rules name does not divide this boundary at all"
                 f" — {len(vac)} boundary(ies): "
                 + "; ".join(f"{' x '.join(g.classes)} — rules name "
                             f"{'/'.join(g.markers[:3])}, {g.n_matching}/{g.n_rows} rows "
                             "carry any of them" for g in vac[:3])),
        remediation=(
            "The words these rules name as the test do not appear in the rows the "
            "referee adjudicated, so whatever decided this boundary, it was not the "
            "stated criterion — and an annotator applying the rule literally gets no "
            "guidance. Give the boundary an observable test, or record that it is "
            "decided by judgement. DO NOT delete the rules: removing guidance leaves the "
            "boundary unaddressed, which is how a text-similarity filter once shredded "
            "32 of 41 rules on a live run.\n\n"
            "NOTE: this gate deliberately does NOT count which way a boundary's rules "
            "point. That signal was tried and retired — a referee drafts a rule only "
            "where it judges the guide to have failed, so on live39 5 of 6 minority rows "
            "produced a rule against 1 of 15 majority rows, and 'most rules point away "
            "from the majority' is the expected shape of a healthy exception set."),
        warn_only=True,
    )

    tax_v2_ref = deps.store.put_json(
        "taxonomy_v2",
        {"taxonomy": taxonomy.model_dump(),
         "referee_rules_added": len(new_rules),
         "rule_conflicts": conflicts.as_record(),
         "rules_vs_evidence": ev_report.as_record(),
         "annotator_balance": bal.as_record(),
         "guide_repaired": bool(repair_meta),
         "provenance": "p2b: taxonomy after the referee's rules and any guide repair"},
        producer="p2b",
        summary=(f"{sum(1 for n in taxonomy.nodes if n.level == 1)} L1 intents, "
                 f"{len(taxonomy.rules)} rules ({len(new_rules)} from the referee)"))
    deps.cache_put("taxonomy_obj", taxonomy)
    deps.emit(f"  taxonomy_v2 persisted: {len(taxonomy.rules)} rules "
              f"({len(new_rules)} drafted by the referee), guide "
              f"{'repaired' if repair_meta else 'unchanged'}")

    return {
        "phase": "p2c",
        "artifacts": {"gold": gold_ref, "gold_agreement": agree_ref,
                      "taxonomy_v2": tax_v2_ref},
        # `deps.gate()` BUILDS a GateResult and registers nothing — a gate this
        # node creates and does not return here reaches the log and no operator.
        "gates": {gate.name: gate, ev_gate.name: ev_gate, bal_gate.name: bal_gate,
                  **_observe(deps, "p2b", {
                      "gold_agreement": agree,
                      "kappa_trace": kappa_trace,
                      "rules_vs_evidence": ev_report.as_record(),
         "annotator_balance": bal.as_record(),
                      "rule_conflicts": conflicts.as_record(),
                  })},
        "completed_phases": ["p2b"],
        "events": [f"P2b: kappa {agree['kappa']:.3f}, {len(new_rules)} rules drafted from disagreements"],
    }


def _gate_let_the_run_proceed(gate: Any) -> bool:
    """Did this gate allow the run to continue?

    `GateResult` carries `status`, never `passed`. Asking for the latter with a
    default — `getattr(gate, "passed", False)` — turns a wrong attribute name
    into a silent wrong ANSWER rather than an AttributeError: on `live38` the
    p2a gate PASSED and the prescriptions still printed "outstanding actions",
    telling an operator to redo work the pipeline had already declined to do.

    `warned` counts as proceeding. A warned gate is non-blocking by construction,
    so the run continues and its findings are limitations to report, not actions
    outstanding.
    """
    return getattr(gate, "status", "") in ("passed", "warned")


def _load_taxonomy_for_reuse(deps: Deps, spec: str) -> Taxonomy:
    """Load a finished taxonomy from `RUN_ID`, `RUN_ID/genNN`, or a path.

    Raises rather than falling back to re-deriving one. Asking to reuse a
    taxonomy and silently getting a fresh one is the worst outcome: the run looks
    like it obeyed, costs a full architect pass, and misses the cache it was
    pointed at — which is exactly the cascade this exists to prevent.
    """
    from pathlib import Path

    cand: list[Path] = []
    direct = Path(spec)
    if direct.suffix == ".json":
        cand.append(direct)
    else:
        root = Path(deps.cfg.run_root) / spec if "/" not in spec else Path(deps.cfg.run_root) / spec
        cand.append(root / "taxonomy.json")
        if root.is_dir():
            cand.extend(sorted(root.glob("gen*/taxonomy.json"), reverse=True))

    for c in cand:
        if not c.exists():
            continue
        blob = json.loads(c.read_text(encoding="utf-8"))
        payload = blob.get("taxonomy", blob)
        deps.emit(f"  reusing taxonomy from {c}")
        return Taxonomy.model_validate(payload)

    raise FileNotFoundError(
        f"--reuse-taxonomy {spec!r}: no taxonomy.json found (looked in "
        + ", ".join(str(c) for c in cand) + ")")


def _remedy_is_exhausted(
    redraw_history: list[dict[str, Any]], max_redraws: int,
) -> bool:
    """Has the redraw loop run out of moves?

    "Exhausted" must mean THE LOOP HAS NO MOVE LEFT, which happens two ways: its
    last attempt was reverted, or it used every attempt it is allowed.

    Asking instead whether *any* attempt helped is the wrong aggregation, and it
    inverted the gate on `live36`: redraw 1 raised kappa 0.781 → 0.806 and redraw
    2 was reverted at 0.795, so the loop was finished — but `any(kept)` was True,
    `remedy_exhausted` came out False, and the gate FAILED a run whose agreement
    had gone UP. A redraw that helps must never make this gate harder to pass
    than a redraw that fails.
    """
    if not redraw_history:
        return False
    return (not redraw_history[-1].get("kept")
            or len(redraw_history) >= max_redraws)



def _snap_label_to_taxonomy(raw: str, codes: set[str]) -> tuple[str, str]:
    """Map a referee verdict onto a real class code, or refuse it.

    The referee's `final_label` used to be written into the gold set unchecked.
    On live38 five rows carried codes that do not exist —
    `LOOKUP_WORD_MEANNING`, `UNDERSPECIFIED_OR_NOICE`, `UNDDERSPECIFIED_OR_NOISE`,
    `LOOKUP_WORD词语释义`, and one literal `LOOKUP_WORD_MEADAR = LOOKUP_WORD_MEANING`.
    Every one is a typo of a real class. They became five one-row "classes",
    were dropped from cross-validation as too rare, and the report explained the
    loss to the reader as rarity rather than as a malformed verdict.

    Returns (code, note). An empty code means the verdict could not be trusted;
    the caller must leave the row unresolved rather than guess.
    """
    import difflib
    import re as _re

    if not raw:
        return "", "empty verdict"
    txt = str(raw).strip()
    if txt in codes:
        return txt, ""
    # "A = B" — the referee spelling out a correction. Prefer a side that exists.
    for part in _re.split(r"[=:\u2192>]+", txt):
        part = part.strip()
        if part in codes:
            return part, f"verdict {txt!r} carried an explicit mapping"
    norm = _re.sub(r"[^A-Z0-9_]", "", txt.upper())
    by_norm = {_re.sub(r"[^A-Z0-9_]", "", c.upper()): c for c in codes}
    if norm in by_norm:
        return by_norm[norm], f"verdict {txt!r} normalised"
    close = difflib.get_close_matches(norm, list(by_norm), n=1, cutoff=0.90)
    if close:
        return by_norm[close[0]], f"verdict {txt!r} snapped (typo)"
    return "", f"verdict {txt!r} is not a class in this taxonomy"


def _batch_by_class_pair(
    disagreements: list[dict[str, Any]], *, target: int = 15,
) -> list[list[list[dict[str, Any]]]]:
    """Group disagreements into GROUPS of sequential chunks.

    Returns ``[[chunk, chunk, ...], [chunk], ...]``. Chunks inside a group must
    run IN ORDER; groups are independent of each other and run concurrently.

    The referee settles boundaries, not rows. Two calls that both see `A × B`
    decide that boundary independently and can rule opposite ways — the hazard
    that forced this entire phase to run sequentially. A group holds all the rows
    for a set of class pairs, so no two GROUPS can contradict each other and they
    are safe to run at once.

    A pair bigger than `target` cannot fit in one call — measured on live38, one
    pair had 52 rows while 25-row batches were already emitting 34,099 tokens and
    failing to parse. Splitting it is unavoidable, so its chunks stay in one
    group and run in order, threading the earlier ruling forward. That is the one
    place `decided` is still needed, and it is deterministic there because the
    chunks are sequential.
    """
    from collections import defaultdict

    by_pair: dict[frozenset[str], list[dict[str, Any]]] = defaultdict(list)
    for d in disagreements:
        by_pair[frozenset((d["label_a"], d["label_b"]))].append(d)

    groups: list[list[list[dict[str, Any]]]] = []
    cur: list[dict[str, Any]] = []
    for _pair, rows_ in sorted(by_pair.items(), key=lambda kv: (-len(kv[1]), sorted(kv[0]))):
        if len(rows_) > target:
            # One boundary, too big for a single call: sequential chunks, own group.
            groups.append(list(_chunks(rows_, target)))
        elif len(cur) + len(rows_) <= target:
            cur.extend(rows_)
        else:
            if cur:
                groups.append([cur])
            cur = list(rows_)
    if cur:
        groups.append([cur])
    return groups


def _run_batch_with_bisect(
    run_batch: Any, fold: Any, chunk: list[dict[str, Any]],
) -> tuple[int, int]:
    """Adjudicate one batch; if the whole call fails, split once and retry halves.

    Returns ``(failed_sub_batches, rows_covered)``.

    A referee batch dies for two reasons and halving addresses both: more to write
    than the model's cap allows, or a transient provider failure that a second
    call clears. Dropping it outright cost `live36` four of its first seven
    batches — 100 adjudications — and those are BY CONSTRUCTION the hardest rows
    in the gold set, so losing them strips the difficult cases and every
    downstream number reads optimistically.

    `fold` is applied per successful call, NOT once at the end, so a recovered
    half still binds the halves and batches that follow it. Adjudication is not a
    per-row task: a boundary settled early must bind later, or the rule set
    acquires two rules that fire on the same trigger with opposite answers.

    Split ONCE, not recursively: the failure budget stays bounded at two extra
    calls per bad batch, which is what keeps a systematically-failing referee from
    turning one bad batch into a retry storm.
    """
    vs = run_batch(chunk)
    if vs is not None:
        fold(vs, chunk)
        return 0, len(chunk)

    mid = len(chunk) // 2
    failed = covered = 0
    for half in (chunk[:mid], chunk[mid:]):
        if not half:
            continue
        hv = run_batch(half)
        if hv is None:
            failed += 1
        else:
            fold(hv, half)
            covered += len(half)
    return failed, covered


def _redraw_until_stable(
    deps: Deps, ctx: Any, df: Any, taxonomy: Taxonomy, pilot: dict[str, Any],
) -> tuple[Taxonomy, dict[str, Any], list[dict[str, Any]]]:
    """Redraw the boundaries a pilot proved are not in the data, then re-pilot.

    Extracted from the node so it can be tested. It decides how money is spent and
    which taxonomy is delivered, and inside `p2a_taxonomy` it could only be
    exercised by a paid run — the offline stand-in is skipped, because re-asking a
    deterministic function in a different batch order measures its batching rather
    than an annotator's reliability, so every pair would look structural.
    """
    cfg = deps.cfg
    redraw_history: list[dict[str, Any]] = []
    for attempt in range(cfg.taxonomy.max_taxonomy_redraws):
        structural = pilot.get("structural_confusions") or []
        if deps.registry.is_offline or not structural or not pilot.get("slack_is_significant"):
            break
        n_b = len(structural)
        deps.emit(f"  redraw {attempt + 1}/{cfg.taxonomy.max_taxonomy_redraws} — "
                  f"{n_b} boundar{'y' if n_b == 1 else 'ies'} the annotator "
                  f"could not reproduce")
        try:
            redrawn = TaxonomyRedrawAgent(ctx).run(
                nodes=taxonomy.nodes, pairs=structural[:6],
                domain_notes=cfg.domain.domain_notes, n_pilot=pilot["n"],
                l1_range=cfg.taxonomy.l1_target_range)
        except Exception as exc:  # noqa: BLE001
            deps.emit(f"  ⚠ redraw failed ({type(exc).__name__}) — keeping the current taxonomy")
            break
        if not redrawn.nodes:
            deps.emit("  ⚠ redraw returned no classes — keeping the current taxonomy")
            break

        before_nodes = list(taxonomy.nodes)
        before_codes = {n.code for n in before_nodes}
        taxonomy = taxonomy.model_copy(update={"nodes": list(redrawn.nodes)})
        after_codes = {n.code for n in taxonomy.nodes}
        # A MERGE ORPHANS THE RULES THAT ROUTED TO THE CLASS IT REMOVED. The
        # redraw replaces nodes only, so the rule set survives intact — and a
        # rule reading "when X → INTERPRET_LITERARY_MEANING" now sends the
        # annotator to a label that is not in its class list. Measured on
        # `live36` gen02: dropping one class left 4 of 45 rules dangling, 2 of
        # them routing straight to the deleted code, and BOTH governed a pair the
        # redraw had targeted — so the damage landed exactly on the rows the
        # before/after kappa is decided by. That comparison (0.825 -> 0.809,
        # reverted) was therefore measuring this bug as much as the redraw.
        gone = before_codes - after_codes
        if gone:
            keep = [r for r in taxonomy.rules if r.then not in gone]
            n_orphaned = len(taxonomy.rules) - len(keep)
            if n_orphaned:
                taxonomy = taxonomy.model_copy(update={"rules": keep})
                deps.emit(f"  dropped {n_orphaned} rule(s) routing to "
                          f"{', '.join(sorted(gone))} — merged away by this redraw")
        candidate = _pilot_agreement(deps, ctx, df, taxonomy)
        deps.emit(f"  redraw {attempt + 1}: {len(before_codes)} → {len(after_codes)} classes, "
                  f"kappa {pilot['kappa']:.3f} → {candidate['kappa']:.3f}, "
                  f"ceiling {pilot.get('self_consistency_kappa')} → "
                  f"{candidate.get('self_consistency_kappa')}")
        redraw_history.append({
            "attempt": attempt + 1,
            "pairs_targeted": [p for p, _ in structural[:6]],
            "classes_before": len(before_codes), "classes_after": len(after_codes),
            "dropped": sorted(before_codes - after_codes),
            "added": sorted(after_codes - before_codes),
            "kappa_before": pilot["kappa"], "kappa_after": candidate["kappa"],
            "ceiling_before": pilot.get("self_consistency_kappa"),
            "ceiling_after": candidate.get("self_consistency_kappa"),
        })
        # A redraw that makes agreement worse is a redraw to discard. Keeping it
        # because it is newer is how a loop like this walks a taxonomy downhill.
        if candidate["kappa"] < pilot["kappa"]:
            deps.emit("  ⚠ redraw lowered kappa — reverting to the previous taxonomy")
            redraw_history[-1]["kept"] = False
            # Restore the ORIGINAL node objects. Filtering the redrawn list by the
            # old codes would keep the redrawn *definitions* under the old names —
            # a revert that reverts nothing, and silently.
            taxonomy = taxonomy.model_copy(update={"nodes": before_nodes})
            break
        redraw_history[-1]["kept"] = True
        pilot = candidate

    return taxonomy, pilot, redraw_history


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
    la, lb = _annotate_both(
        ctx, queries, classes_txt, rules_txt, taxonomy.labeling_guide, deps)
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

    # The ceiling can say WHICH pairs are broken, not just how many, and that
    # distinction decides the remedy. A pair one annotator cannot reproduce
    # against itself is a boundary that is not in the data — merging or re-cutting
    # is the only fix, and writing a tie-break for it is wasted effort. A pair two
    # annotators split on while each stays self-consistent is exactly what a
    # tie-break is for. Replaying one run's pilot, these were about half its
    # disagreements and the split was invisible in the aggregate kappa.
    structural = Counter(
        " × ".join(sorted((x["label"], str(v))))
        for x, v in zip(la, back) if x["label"] != str(v)
    )
    structural_pairs = {k for k, _ in structural.most_common()}
    guide_only = Counter({k: v for k, v in conf.items() if k not in structural_pairs})
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

    # Is the remaining slack big enough to be worth a redraft, or is it noise?
    #
    # This used to be `share_of_ceiling >= 0.90` — another constant carried in from
    # one corpus, and one that decided whether a run halted. The question it stands
    # for is answerable from the pilot itself: the slack a redraft could recover is
    # `ceiling - kappa`, and it is only worth acting on if it is larger than the
    # error on that difference. Treating the two kappas as independent overstates
    # that error, because both involve annotator A's first pass — which is the
    # conservative direction: it declares slack less often, so it halts less often.
    #
    # Note the test tightens with the pilot size: at the configured 200 the error on
    # the difference is about 0.05, so nothing under roughly 0.08 of slack can fire
    # it, which is already a meaningful effect. Raise `pilot_sample_size` far above
    # that and statistical significance stops implying practical significance — a
    # 0.02 slack would halt a run for a redraft not worth doing. If the pilot ever
    # grows, pair this with a floor on the effect size.
    slack = slack_se = None
    slack_is_real = False
    if ceiling is not None:
        po2, k2 = self_agree["raw_agreement"], self_agree["kappa"]
        pe2 = (po2 - k2) / (1 - k2) if k2 < 1 else 0.0
        se2 = ((po2 * (1 - po2) / n) ** 0.5) / max(1e-6, 1 - pe2)
        slack = round(ceiling - kp, 4)
        slack_se = round((se ** 2 + se2 ** 2) ** 0.5, 4)
        slack_is_real = slack > 1.645 * slack_se

    # A bar above the ceiling cannot be met by any amount of guide work: two
    # annotators cannot agree with each other more reliably than one agrees with
    # itself. Saying so is the difference between "your guide needs work" and
    # "this target was never reachable with this annotator".
    target_above_ceiling = ceiling is not None and target > ceiling

    # A COUNT FLOOR, not just a rank. Measured by replaying all three live36
    # pilots against an UNCHANGED class list: 36 disagreements spread over 25
    # distinct pairs, 19 of them seen exactly once, so the expected count per
    # pair is about 1.4 — and "the top 6" did not survive re-measurement, with
    # only 3 of 6 pairs recurring between consecutive pilots. Ranking noise sent
    # the architect a target list that was half chance, at ~350s of frontier
    # model plus a full three-pass re-pilot per round. A pair seen once is not
    # evidence of a boundary; it is the tail of a flat distribution.
    top_structural = [(pr, c) for pr, c in structural.most_common(6) if c >= 2]
    top_guide = [(pr, c) for pr, c in guide_only.most_common(6) if c >= 2]
    thin = len(structural.most_common(6)) - len(top_structural)
    if thin:
        # Never cap silently: a dropped target is a boundary nobody will look at.
        deps.emit(f"  {thin} confused pair(s) seen only once — below the noise "
                  f"floor at n={agree['n']}, not sent to the redraw")
    return {
        "n": agree["n"],
        "self_consistency_kappa": ceiling,
        "self_consistency_raw": self_agree.get("raw_agreement"),
        "share_of_ceiling_reached": headroom,
        "structural_confusions": top_structural,
        "guide_confusions": top_guide,
        "confusions_below_noise_floor": thin,
        "recoverable_slack": slack,
        "recoverable_slack_se": slack_se,
        "slack_is_significant": slack_is_real,
        "target_above_ceiling": target_above_ceiling,
        "ceiling_measured_on_real_annotator": not deps.registry.is_offline,
        "ceiling_verdict": (
            "guide has slack — inter-annotator agreement is well below what this "
            "annotator achieves against itself, so redrafting should help"
            if slack_is_real else
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
        la, lb = _annotate_both(
            ctx, queries, classes_txt, rules_txt, taxonomy.labeling_guide, deps)
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

        def _row2(i: int, q: str) -> GoldRow:
            a, b = la[i]["label"], lb[i]["label"]
            missing = UNLABELED in (a, b)   # nobody labelled it: not agreement
            return GoldRow(
                query=q, idx=int(new_idx[i]), label_a=a, label_b=b,
                final="" if missing else (a if a == b else ""),
                agreed=(not missing) and a == b,
                rationale_a=la[i].get("rationale", ""), rationale_b=lb[i].get("rationale", ""),
                round=attempt + 1, source="guide_repair")

        rows2 = [_row2(i, q) for i, q in enumerate(queries)]
        out.update({"agreement": agree2, "rows": rows2, "idx": new_idx})
        out["summary"] = {
            "rounds_run": attempt, "decisions": decisions,
            "n_rules_added": len(added), "kappa_before": prev, "kappa_after": agree2["kappa"],
            "n_before": prev_n, "n_after": agree2["n"],
            # EQUAL n IS NOT THE SAME ROWS. This flag used to read
            # `prev_n == agree2["n"]`, which reports "comparable" for two
            # DISJOINT samples that happen to be the same size — precisely the
            # case a reader needs warned about. On a fresh sample the rows differ
            # by construction, so nothing is paired and the delta is a
            # between-sample z-test, never a before/after on the same rows.
            "comparable": (not cfg.taxonomy.repair_on_fresh_sample) and prev_n == agree2["n"],
            "paired": not cfg.taxonomy.repair_on_fresh_sample,
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
        # CITE the rule; do not repeat it. These same rules are already sent in
        # full in the `## Adjudication rules` block, so rendering their text here
        # duplicated ~24,000 characters of an annotation prompt — and both copies
        # were then truncated for being too big, which is how the binding
        # rulings ended up 75% cut on live38. The guide's job is different from
        # the rule list's: it states which boundaries are BINDING. That needs the
        # pair, the winner, and an id to look up, not the rule text again.
        rules = [r for r in added if set(r.classes) == set(d["pair"])]
        for r in rules:
            lines.append(f"- {pair} → **{r.then}**  [{r.id}]")
    return (guide or "") + "\n".join(lines)



def _annotate_both(ctx: Any, queries: list[str], classes: str, rules: str,
                   guide: str, deps: Deps) -> tuple[list[dict], list[dict]]:
    """Run the two annotators AT THE SAME TIME.

    They are independent by design — that independence is the whole
    methodological point of having two — so there was never an ordering
    dependency between them, only the fact that the code called one and then the
    other. Measured on live38, per call:

        annotator_a  deepseek-v4-flash   median 161.3s   ~20,600 output tokens
        annotator_b  qwen3-next-80b       median  23.5s   ~1,792 output tokens

    At the configured 8-way batch concurrency that is roughly 142 min and 27 min.
    Sequentially the phase costs their SUM; concurrently it costs their MAX, so
    this returns ~16% of the run's wall clock. p2b is 75% of the pipeline, and
    the two annotators are 94% of its calls.

    **This is a `max()`, not a halving.** The two are wildly unbalanced, so the
    saving is `min(a, b)` — the whole of the faster annotator disappears into the
    shadow of the slower one, and nothing more.

    **Peak concurrency doubles while this runs**: each annotator keeps its own
    `llm.max_concurrency` batch pool, so p2b issues up to `2 x max_concurrency`
    requests. Splitting one budget between them instead would be WORSE than
    sequential — halving annotator_a's pool roughly doubles its 142 min, which
    more than eats the 27 min saved. If a provider rate-limits, lower
    `llm.max_concurrency` or set `llm.annotators_concurrent: false`.

    Results are unaffected: `_annotate` rebuilds its return as
    `[got.get(q) for q in queries]`, i.e. in QUERY order rather than completion
    order, so the positional pairing the kappa computation relies on holds
    however the batches interleave.
    """
    args = (queries, classes, rules, guide, deps)
    if ctx.registry.is_offline or not getattr(ctx.cfg.llm, "annotators_concurrent", True):
        # Offline is deterministic and near-instant; there is no latency to hide,
        # and keeping it sequential keeps the stand-in's logs readable.
        return _annotate(ctx, "a", *args), _annotate(ctx, "b", *args)

    from concurrent.futures import ThreadPoolExecutor

    peak = ctx.cfg.llm.max_concurrency * 2
    deps.emit(f"  annotators a and b running concurrently "
              f"(peak {peak} in-flight requests)")
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="annot") as pool:
        fut = {w: pool.submit(_annotate, ctx, w, *args) for w in ("a", "b")}
        out: dict[str, Any] = {}
        errs: dict[str, BaseException] = {}
        for w, f in fut.items():
            try:
                out[w] = f.result()
            except BaseException as exc:  # noqa: BLE001
                # Collect BOTH outcomes before raising. Letting the first
                # exception escape the `with` would still block on the other
                # thread, then discard its completed work and report only one of
                # two possible causes.
                errs[w] = exc
    if errs:
        who = ", ".join(sorted(errs))
        raise RuntimeError(
            f"annotator {who} failed: "
            + "; ".join(f"{w}={type(e).__name__}: {e}" for w, e in sorted(errs.items()))
        ) from next(iter(errs.values()))
    return out["a"], out["b"]

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
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Emit as batches land. A 3,000-row gold set is 120 batches per
        # annotator at eight concurrent, and on success `_one` said nothing at
        # all — so the phase went dark for roughly forty minutes per annotator
        # and a watcher could not distinguish that from a hang, which is the one
        # thing the dashboard exists to rule out. Order is irrelevant here: the
        # results are merged into a dict keyed by query.
        maps = []
        with ThreadPoolExecutor(max_workers=ctx.cfg.llm.max_concurrency) as pool:
            futures = [pool.submit(_one, b) for b in batches]
            # About ten updates whatever the size. `len // 10` alone emits once
            # per batch below ten, so the 200-query pilot printed all eight.
            step = max(1, round(len(futures) / 10)) if len(futures) >= 10 else len(futures)
            for i, fut in enumerate(as_completed(futures), 1):
                maps.append(fut.result())
                if i % step == 0 and i < len(futures):
                    deps.emit(f"  annotator[{which}] {i}/{len(futures)} batches"
                              f" · {sum(len(m) for m in maps)} rows")
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

    **`then` must BE a class, not merely mention one.** This asked
    `any(c in then for c in codes)`, which a whole sentence satisfies:
    `'归 JUDGE_LANGUAGE_USAGE，不归 LOOKUP_CHAR_PRONUNCIATION。'` names a real
    class and is still unusable as a key — and live38 shipped 18 rules like it.
    Everything downstream keys on `then`: `_dedupe_rules` compares it with `==`
    (two phrasings of one answer read as a CONTRADICTION and both valid rules are
    withheld), and `rules_against_evidence` asks whether it equals the referee's
    majority verdict, which a sentence can never do — so every prose rule would
    count as pointing AGAINST the evidence and a sound boundary would be reported
    contradicted. `normalise_then` resolves the single-class case to the bare code
    and keeps the sentence in `rationale`, which the annotator prompt renders; a
    rule naming TWO classes is left alone and reported, because `归 A，不归 B` and
    `有裁决框架的归 A；单纯问 X 的归 B` are indistinguishable mechanically and
    picking a side would silently rewrite the rule.
    """
    from difflib import SequenceMatcher

    from ...ops.rule_conflict import normalise_then

    codes = [n.code for n in taxonomy.nodes]
    repaired: list[dict[str, str]] = []
    dropped: list[dict[str, str]] = []
    rewritten: list[dict[str, str]] = []
    ambiguous: list[dict[str, str]] = []
    kept: list[Any] = []
    for rule in taxonomy.rules:
        then = str(rule.then)
        res = normalise_then(then, codes)
        if res.is_key:
            if res.code != then:
                # Preserve the instruction where the annotator still reads it —
                # `_render_rules` prints the rationale beside every rule.
                rule.rationale = (f"{rule.rationale} " if rule.rationale else "") + then
                rule.then = res.code
                rewritten.append({"id": rule.id, "was": then[:90], "now": res.code})
            kept.append(rule)
            continue
        if res.found:
            # Names several classes. Still readable guidance, so it ships — but
            # it is not a key, and the mechanisms that key on `then` skip it.
            ambiguous.append({"id": rule.id, "then": then[:90], "why": res.note})
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
            # The BARE code, not the repaired sentence. Writing back
            # `then.replace(token, best[1])` produced `选 EXAM_INFO` — exactly the
            # shape the normalisation above exists to remove, reintroduced by the
            # repair path. The original wording goes where the annotator still
            # reads it.
            if then != best[1]:
                rule.rationale = (f"{rule.rationale} " if rule.rationale else "") + then
            rule.then = best[1]
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
    if rewritten:
        deps.emit(f"  {len(rewritten)} rule(s) held a sentence in `then` — reduced to the "
                  "class they name, sentence kept in the rationale")
    if ambiguous:
        deps.emit(f"  ⚠ {len(ambiguous)} rule(s) name TWO classes in `then` and cannot be "
                  "used as a key — they still reach the annotator, but rule-conflict and "
                  "rule-vs-evidence measurement skips them:")
        for r in ambiguous[:4]:
            deps.emit(f"      {r['id']}: {r['why'][:100]}")
    return {"n_repaired": len(repaired), "n_dropped": len(dropped),
            "n_then_reduced_to_code": len(rewritten), "n_then_ambiguous": len(ambiguous),
            "repaired": repaired, "dropped": dropped,
            "then_reduced": rewritten, "then_ambiguous": ambiguous}


def _render_classes(t: Taxonomy) -> str:
    return "\n".join(
        f"- **{n.code}** ({n.name}): {n.definition}\n"
        f"    need: {n.user_need}\n"
        f"    yes: {n.positive_examples[:4]}\n"
        f"    no:  {n.negative_examples[:3]}"
        for n in t.nodes
    )


def _render_rules(t: Taxonomy) -> str:
    """Every adjudication rule the annotator is allowed to cite.

    `TaxonomyNode.adjudication_rules` used to be write-only. It is declared to
    hold rule *ids* cross-referencing the top-level list; the models fill it with
    whole rules instead, and nothing in the codebase ever read it — so the
    architect spent tokens writing per-class tie-breaks that reached no annotator,
    no referee and no classifier. That was 55 rules discarded in one live run and
    24 in the next. Whatever the field holds, it is adjudication content the model
    produced on purpose, so it belongs in front of the annotator.

    Structured rules come first: they are deduplicated and carry ids the referee
    can cite, and the section is budgeted downstream, so anything truncated should
    be the free text rather than the ids.
    """
    # NEWEST FIRST, so truncation eats the oldest rules rather than the freshest.
    # The referee's rules are APPENDED to this list and are the most
    # evidence-driven in it — each was written in response to a disagreement
    # actually observed on this corpus. Rendered in insertion order they sit at
    # the end, which is exactly what a head-limited budget discards: on `live38`
    # 83 referee rules pushed the block to 18,496 chars against a 9,000 budget
    # and every one of them was cut from the guide-repair round that existed to
    # apply them. Ordering by round descending makes the survivors the rules
    # worth keeping, whatever the budget turns out to be.
    ordered = sorted(t.rules, key=lambda r: -(r.added_in_round or 0))
    lines = [f"- [{r.id}] when {r.when} → {r.then} ({r.rationale})" for r in ordered]
    seen = {line.strip() for line in lines}
    for node in t.nodes:
        for raw in node.adjudication_rules:
            text = str(raw).strip()
            # A bare id is a cross-reference to a rule already listed above.
            if not text or text in seen or any(text == r.id for r in t.rules):
                continue
            seen.add(text)
            lines.append(f"- [{node.code}] {text}")
    return "\n".join(lines)


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
        "gates": _observe(deps, "p2c", {"topdown_metrics": result}, decisions=[decision]),
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
    n_failed_chunks = 0
    for chunk in _chunks(rows, 30):
        try:
            out = AdversaryAgent(ctx).run(
                rows=chunk, classes=_render_classes(taxonomy) if taxonomy else "")
            results.extend(out.results)
        except Exception as exc:  # noqa: BLE001
            # An unguarded call here crashed the whole phase on any provider
            # blip. Losing 30 attacks is recoverable; losing the run is not — so
            # record the loss in the coverage figure below and carry on.
            n_failed_chunks += 1
            deps.emit(f"  ⚠ adversary chunk failed ({type(exc).__name__}) — "
                      f"{len(chunk)} labels not attacked")

    verdicts = [r.verdict for r in results]
    n_attacked = len(rows)
    n_verdicts = len(verdicts)
    wrong = sum(v == "wrong" for v in verdicts)
    defensible = sum(v == "defensible" for v in verdicts)
    # THE DENOMINATOR IS THE ROWS THAT GOT A VERDICT, AND THE COVERAGE IS
    # REPORTED BESIDE IT. This was `n = max(len(verdicts), 1)`, so a short
    # response silently shrank the denominator and a response with NO verdicts
    # gave `1 - 0/1` = a perfect 1.000. An accuracy no row was actually judged
    # for must be undefined, never flattering.
    est = None if not n_verdicts else 1 - wrong / n_verdicts
    coverage = n_verdicts / max(n_attacked, 1)
    shown = "undefined" if est is None else f"{est:.3f}"
    deps.emit(f"  survived attack: {shown} ({wrong} wrong, {defensible} defensible "
              f"of {n_verdicts} verdicts on {n_attacked} attacked; "
              f"coverage {coverage:.0%})")
    if est is None:
        deps.emit("  ⚠ the adversary returned no verdicts — no accuracy is estimable")
    elif coverage < 0.8:
        deps.emit(f"  ⚠ only {coverage:.0%} of attacked labels came back — read the "
                  f"estimate against that, not as an accuracy over {n_attacked}")

    ref = deps.store.put_json(
        "adversarial_validation",
        {
            "n_attacked": n_attacked, "n_verdicts": n_verdicts,
            "coverage": round(coverage, 4), "n_failed_chunks": n_failed_chunks,
            "n_wrong": wrong, "n_defensible": defensible,
            "estimated_accuracy": None if est is None else round(est, 4),
            "knn_label_scan": scan,
            "results": [r.model_dump() for r in results[:200]],
            "method": (
                "agents were instructed to PROVE each label wrong; the estimate is the "
                "share of labels whose attack failed, not a self-reported confidence"
            ),
        },
        producer="p2d",
        summary=f"adversarial accuracy estimate {shown} on {n_verdicts}/{n_attacked}",
    )
    return {
        "phase": "p3",
        "artifacts": {"adversarial_validation": ref},
        "gates": _observe(deps, "p2d", {"adversarial_validation": deps.load("adversarial_validation")}),
        "completed_phases": ["p2d"],
        "events": [f"P2d: adversarial accuracy {shown} on {n_verdicts} verdicts "
                   f"over {n_attacked} attacked labels ({coverage:.0%} coverage)"],
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
    la, lb = _annotate_both(
        ctx, queries, classes_txt, rules_txt, taxonomy.labeling_guide, deps)
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
