"""Phase 7 — blind naming and tree audit; Phase 8 — executing the audit.

The naming fan-out is the one place in this pipeline where the *graph topology*
carries a methodological guarantee rather than just scheduling work.

Five naming agents run in parallel via ``Send``.  A ``Send`` worker's state
contains exactly the keys in its payload — the parent state is structurally
unreachable, not merely undisclosed.  So a shard cannot see the taxonomy, the
legacy labels, or the other shards' answers even if a future edit to the prompt
tries to include them.  On top of that structural guarantee sits the blindness
firewall, which scans each card for label vocabulary before it can be rendered.

Belt and braces, because anchoring is not a hypothetical: a namer who has seen
the existing category list will file clusters under it, and the resulting tree is
a picture of the old taxonomy rather than of the data.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from langgraph.types import Send

from ...agents.roles import AuditorAgent, NamerAgent, RiskSentinelAgent
from ...ops.cards import (
    build_naming_cards,
    centroid_similarity_pairs,
    cluster_samples,
    template_spread,
)
from ...ops.governance import assert_all_settled, execute_prescriptions, governance_ledger
from ...records import LeafNaming, NamingCard, Prescription
from ...state import PipelineState
from ..deps import Deps


def _recover_cards(deps: Deps) -> dict[int, NamingCard]:
    """Naming cards from memory, or rebuilt from the stored artifact.

    A resumed run that reached the naming shards with an empty cache used to
    name zero clusters and report success.
    """
    return deps.recover(
        "naming_cards", "naming_cards",
        rebuild=lambda blob: {
            c["leaf_id"]: NamingCard.model_validate(c) for c in blob.get("cards", [])
        },
        default={},
    ) or {}


def p7_prepare(state: PipelineState, deps: Deps) -> dict[str, Any]:
    """Build one card per leaf and arm the blindness firewall.

    The firewall is loaded here — with the taxonomy's names and codes, every
    legacy label value, and nothing else — so that any leak in a downstream
    prompt raises instead of quietly anchoring a namer.
    """
    cfg = deps.cfg
    df = deps.df
    H = deps.embedding("emb_hybrid")
    labels = deps.load("leaf_labels")
    centroids = deps.load("leaf_centroids")
    deps.emit(f"P7 prepare — building {int(labels.max()) + 1} naming cards")

    deps.firewall.add_taxonomy(deps.taxonomy())
    for col in cfg.data.reference_label_columns:
        if col in df.columns:
            deps.firewall.add_reference_labels(df[col].astype(str).unique().tolist())
    deps.emit(f"  firewall armed with {len(deps.firewall.forbidden)} forbidden label terms")

    cards = build_naming_cards(
        df, labels, H, centroids, text_col=cfg.data.text_column,
        n_center=cfg.naming.card_center, n_random=cfg.naming.card_random,
        n_edge=cfg.naming.card_edge, n_ngrams=cfg.naming.card_top_ngrams, seed=cfg.seed_metric,
    )
    deps.cache_put("naming_cards", {c.leaf_id: c for c in cards})

    ref = deps.store.put_json(
        "naming_cards",
        {"cards": [c.model_dump() for c in cards],
         "firewall": deps.firewall.summary(),
         "contract": (
             "cards contain member queries and n-grams only — no taxonomy names, no legacy "
             "labels, no other agent's output. Enforced by BlindnessFirewall.assert_blind()."
         )},
        producer="p7", summary=f"{len(cards)} blind naming cards",
    )
    return {
        "artifacts": {"naming_cards": ref},
        "events": [f"P7: {len(cards)} cards prepared, firewall armed "
                   f"({len(deps.firewall.forbidden)} forbidden terms)"],
    }


def fan_out_namers(state: PipelineState, deps: Deps) -> list[Send]:
    """Shard the leaves across naming agents.

    Each ``Send`` payload carries a shard index and a list of leaf ids — nothing
    else.  That is the whole isolation mechanism: the worker's state *is* this
    payload.
    """
    cards = _recover_cards(deps)
    n_agents = max(deps.cfg.naming.n_naming_agents, 1)
    leaf_ids = sorted(cards)
    shards: list[list[int]] = [[] for _ in range(n_agents)]
    for i, lid in enumerate(leaf_ids):
        shards[i % n_agents].append(lid)
    return [
        Send("p7_name_shard", {"shard_id": i, "leaf_ids": ids})
        for i, ids in enumerate(shards) if ids
    ]


def p7_name_shard(payload: dict[str, Any], deps: Deps) -> dict[str, Any]:
    """One naming agent, one shard.  Sees its cards and nothing else."""
    shard_id = payload["shard_id"]
    leaf_ids = payload["leaf_ids"]
    cards: dict[int, NamingCard] = _recover_cards(deps)
    ctx = deps.agent_ctx()
    agent = NamerAgent(ctx, suffix=f"_{shard_id}")

    out: list[LeafNaming] = []
    for lid in leaf_ids:
        card = cards.get(lid)
        if card is None:
            continue
        out.append(agent.run(card=card))
    deps.emit(f"  namer[{shard_id}] named {len(out)} clusters")
    return {"namings": out}


def p7_audit(state: PipelineState, deps: Deps) -> dict[str, Any]:
    """Assemble the tree, audit it, and run an independent risk sweep.

    The risk sentinel runs on cluster samples with no knowledge of what the
    namers flagged.  When both independently land on the same cluster, that is
    evidence; when only the pre-screen finds it, that is bookkeeping.
    """
    cfg = deps.cfg
    namings: list[LeafNaming] = state.get("namings", [])
    labels = deps.load("leaf_labels")
    leaf_family = deps.load("leaf_family")
    centroids = deps.load("leaf_centroids")
    masks = deps.template_masks()
    deps.emit(f"P7 audit — {len(namings)} namings in")

    n_leaves = int(labels.max()) + 1
    sizes = np.bincount(labels, minlength=n_leaves)
    audit = AuditorAgent(deps.agent_ctx()).run(
        namings=namings, leaf_family=leaf_family.tolist(), leaf_sizes=sizes.tolist(),
        template_spread=template_spread(masks, labels) if masks else None,
        centroid_similarity=centroid_similarity_pairs(centroids),
    )

    samples = cluster_samples(deps.df, labels, text_col=cfg.data.text_column, seed=cfg.seed_metric)
    risk = RiskSentinelAgent(deps.agent_ctx()).run(cluster_samples=samples)
    deps.emit(f"  auditor: {len(audit.prescriptions)} prescriptions; "
              f"risk sentinel: {len(risk.findings)} findings")

    # Did anyone find the risk content without being told about it?
    namer_flagged = {n.leaf_id for n in namings if n.risk_flag}
    sentinel_flagged = {c for f in risk.findings for c in f.cluster_ids}
    independently_found = bool(namer_flagged or sentinel_flagged)

    # Ids are assigned here, never by the agent. The id namespace is the
    # governance ledger's primary key — an agent that invents one can collide
    # with an existing prescription and silently overwrite it in the reducer.
    # An agent may PROPOSE; only the pipeline may EXECUTE. Trusting an agent's
    # own `status` field let a prescription arrive already marked `executed`
    # with an empty evidence pointer — which is precisely the "recommended but
    # never applied" failure that Principle 6's gate exists to catch, arriving
    # through the gate's own front door.
    prescriptions = []
    for p in audit.prescriptions:
        p.id = deps.next_prescription_id()
        p.proposed_by = p.proposed_by or "tree_auditor"
        p.status = "proposed"
        p.executed_at = None
        p.evidence = {}
        p.decline_reason = ""
        p.targets = sorted({int(t) for t in p.targets if 0 <= int(t) < n_leaves})
        if p.targets or p.kind == "keep_as_is":
            prescriptions.append(p)
    for f in risk.findings:
        targets = sorted({int(c) for c in f.cluster_ids if 0 <= int(c) < n_leaves})
        if targets:
            prescriptions.append(Prescription(
                id=deps.next_prescription_id(), kind="flag_risk", targets=targets,
                rationale=f"{f.category}: {f.rationale}", proposed_by="risk_sentinel",
            ))
    for lid in sorted(namer_flagged - sentinel_flagged):
        prescriptions.append(Prescription(
            id=deps.next_prescription_id(), kind="flag_risk", targets=[int(lid)],
            rationale="flagged independently by the blind namer that saw this cluster",
            proposed_by="namer",
        ))

    coherences = [n.coherence for n in namings if n.coherence]
    mean_coh = float(np.mean(coherences)) if coherences else float("nan")

    # Every leaf must come back named. A shard that fails silently leaves
    # clusters in the delivered table with no name, no user_need, and no
    # coherence score — a hole that the mean coherence above would happily
    # average over without mentioning.
    named = {n.leaf_id for n in namings}
    unnamed = sorted(set(range(n_leaves)) - named)

    naming_ref = deps.store.put_json(
        "tree_naming",
        {"namings": [n.model_dump() for n in namings],
         "audit": audit.model_dump(),
         "risk_report": risk.model_dump(),
         "independent_risk_discovery": {
             "namer_flagged_leaves": sorted(namer_flagged),
             "sentinel_flagged_leaves": sorted(sentinel_flagged),
             "both_agree_on": sorted(namer_flagged & sentinel_flagged),
             "found_without_being_told": independently_found,
         },
         "mean_coherence": round(mean_coh, 3) if coherences else None},
        producer="p7", summary=f"{len(namings)} named, {len(prescriptions)} prescriptions",
    )

    gates = {}
    g0 = deps.gate(
        "p7_all_leaves_named", "p7",
        passed=not unnamed,
        observed={"n_leaves": n_leaves, "n_named": len(named), "unnamed": unnamed[:20]},
        threshold={"unnamed": 0},
        message=(f"all {n_leaves} leaves named" if not unnamed
                 else f"{len(unnamed)} leaves came back unnamed: {unnamed[:10]}"),
        remediation="A naming shard failed. Check the run log for the exception — an unnamed "
                    "cluster ships with no name, no user_need, and no coherence score.",
    )
    gates[g0.name] = g0
    g1 = deps.gate(
        "p7_coherence", "p7",
        passed=(not coherences) or mean_coh >= cfg.gates.coherence,
        observed={"mean_coherence": round(mean_coh, 3) if coherences else None, "n": len(coherences)},
        threshold={"mean_coherence": cfg.gates.coherence},
        message=f"mean blind coherence {mean_coh:.2f}/5" if coherences else "no coherence scores",
        remediation="Low coherence means clusters are carrying more than one intent — "
                    "split them in refinement or reduce K before delivering.",
        warn_only=True,
    )
    gates[g1.name] = g1
    if cfg.gates.require_risk_independently_found:
        g2 = deps.gate(
            "p7_risk_independently_found", "p7",
            passed=independently_found,
            observed={"namer_flagged": sorted(namer_flagged), "sentinel_flagged": sorted(sentinel_flagged)},
            threshold={"at_least_one_unprompted_flag": True},
            message="risk content was flagged by an agent that was never told to look for it"
                    if independently_found else "no agent independently flagged risk content",
            remediation="If only the seeded pre-screen finds risk content, the finding is "
                        "bookkeeping rather than evidence. Check whether risk clusters are "
                        "being absorbed into topically similar normal families.",
            warn_only=True,
        )
        gates[g2.name] = g2

    return {
        "phase": "p8",
        "artifacts": {"tree_naming": naming_ref},
        "gates": gates,
        "prescriptions": prescriptions,
        "completed_phases": ["p7"],
        "events": [
            f"P7: mean coherence {mean_coh:.2f}/5, {len(prescriptions)} prescriptions raised",
            f"P7: risk found independently: {independently_found}",
        ],
    }


def p8_governance(state: PipelineState, deps: Deps) -> dict[str, Any]:
    """Execute every prescription against the data, then prove none were left open."""
    prescriptions: list[Prescription] = list(state.get("prescriptions", []))
    leaf_family = deps.load("leaf_family")
    labels = deps.load("leaf_labels")
    masks = deps.template_masks()
    deps.emit(f"P8 governance — executing {len(prescriptions)} prescriptions")

    from ...ops.templates import template_fragmentation

    def family_metrics(lf: np.ndarray) -> dict[str, float]:
        fam_of_row = lf[labels]
        out: dict[str, float] = {"n_families": float(len(np.unique(lf)))}
        if masks:
            out["template_fragmentation"] = template_fragmentation(fam_of_row, masks)["mean_fragmentation"]
        return out

    before = family_metrics(leaf_family)
    new_family, prescriptions, detail = execute_prescriptions(
        prescriptions, leaf_family, metrics_before=before, recompute=family_metrics,
        X=deps.embedding("emb_hybrid"), leaf_labels=labels,
    )
    new_labels = detail.pop("leaf_labels", None)
    if detail.get("relabelled"):
        deps.cache_put("leaf_relabels", {int(k): v for k, v in detail["relabelled"].items()})

    try:
        settled = assert_all_settled(prescriptions)
        passed = True
        msg = f"{settled['n_executed']} executed, {settled['n_declined']} explicitly declined"
    except Exception as exc:  # noqa: BLE001
        settled = {"error": str(exc)}
        passed = False
        msg = str(exc)[:200]

    artifacts_out = {}
    if new_labels is not None and not np.array_equal(new_labels, labels):
        from ...ops.cluster import _centroids

        n_new = int(new_labels.max()) + 1
        cents = _centroids(deps.embedding("emb_hybrid"), new_labels, n_new)
        # Written under NEW names. Overwriting `leaf_labels` would destroy the
        # pre-governance partition that Phase 9 compares against and that the
        # delivered table's `bu_family_pre_governance` column makes auditable —
        # the same append-only rule that keeps rejected generations on disk.
        artifacts_out["leaf_labels_final"] = deps.store.put_matrix(
            "leaf_labels_final", new_labels, producer="p8",
            summary=f"post-governance leaves ({n_new}); pre-governance kept in leaf_labels")
        artifacts_out["leaf_centroids_final"] = deps.store.put_matrix(
            "leaf_centroids_final", cents, producer="p8", summary="post-split centroids")
        deps.cache_put("leaf_labels_final", new_labels)
        deps.cache_put("leaf_centroids_final", cents)
    fam_ref = deps.store.put_matrix("leaf_family_final", new_family, producer="p8",
                                    summary=f"post-governance families ({len(np.unique(new_family))})")
    ledger_ref = deps.store.put_json(
        "governance",
        {"ledger": governance_ledger(prescriptions), "execution": detail, "settled": settled,
         "mechanism": (
             "family merges and risk isolations are leaf→family lookup-table remaps: leaf "
             "assignments and centroids are untouched, the pre-governance column is retained, "
             "and every change is reversible. Leaf splits genuinely re-partition — a new leaf "
             "is appended and the shipped centroid matrix changes — and are written under "
             "leaf_labels_final so the pre-governance partition survives."
         )},
        producer="p8", summary=f"{detail['n_executed']} executed, {detail['n_declined']} declined",
    )
    deps.cache_put("leaf_family_final", new_family)

    gate = deps.gate(
        "p8_governance_executed", "p8",
        passed=passed, observed={"n_executed": detail["n_executed"], "n_declined": detail["n_declined"]},
        threshold={"all_prescriptions_settled": True}, message=msg,
        remediation="Every 'we recommend X' in a report must have a matching executed change "
                    "in a delivered column, or an explicit declined reason (Principle 6).",
    )
    deltas = detail.get("metric_deltas", {})
    return {
        "phase": "p9",
        "artifacts": {**artifacts_out, "leaf_family_final": fam_ref, "governance": ledger_ref},
        "gates": {gate.name: gate},
        "prescriptions": prescriptions,
        "completed_phases": ["p8"],
        "events": [
            f"P8: families {before.get('n_families', 0):.0f} → {len(np.unique(new_family))}; "
            f"fragmentation delta {deltas.get('template_fragmentation', 0):+.3f}"
        ],
    }
