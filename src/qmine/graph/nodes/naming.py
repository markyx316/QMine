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

import time

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
from . import observe as _observe
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

    def _name(lid: int) -> LeafNaming | None:
        card = cards.get(lid)
        if card is None:
            return None
        # RETRY, the way `_annotate` does. A namer failure is not a lost row: a
        # single unnamed leaf fails `p7_all_leaves_named`, which is a BLOCKING
        # gate, and `resume` refuses to overturn a gate — so one transient blip
        # among ~60 calls ends a paid run at phase 7 and forces a new generation.
        # Returning None on the first exception made that a coin-flip on provider
        # weather, in the one phase where nothing repairs a missing result.
        last = ""
        for attempt in range(3):
            try:
                return agent.run(card=card)
            except Exception as exc:  # noqa: BLE001
                last = type(exc).__name__
                if attempt < 2:
                    time.sleep(2 ** attempt)
        deps.emit(f"  namer[{shard_id}] leaf {lid} failed after 3 attempts: {last}")
        return None

    # Within a shard the clusters are independent too — the shard exists to keep
    # agents from seeing each other's answers, not to serialise their work.
    if len(leaf_ids) > 1 and not ctx.registry.is_offline:
        from concurrent.futures import ThreadPoolExecutor

        # Respect the configured ceiling. This was a hard-coded 4, and the shards
        # themselves run concurrently — 5 shards x 4 threads put 20 calls on one
        # provider while `llm.max_concurrency` said 8 and every other fan-out
        # obeyed it. Worse, it made that knob INERT in exactly the phase whose
        # blocking gate has no repair path, so turning it down when a provider
        # throttles had no effect here. Divide the budget across the shards that
        # share the key rather than multiplying by them.
        n_shards = max(1, int(getattr(deps.cfg.naming, "n_naming_agents", 1) or 1))
        per_shard = max(1, deps.cfg.llm.max_concurrency // n_shards)
        with ThreadPoolExecutor(max_workers=min(per_shard, len(leaf_ids))) as pool:
            out = [n for n in pool.map(_name, leaf_ids) if n is not None]
    else:
        out = [n for n in (_name(l) for l in leaf_ids) if n is not None]
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
    # A MEAN against an absolute bar is the wrong test twice over.
    #
    # First, raters are not calibrated across models or domains: the same tree
    # scored 3.93 and 4.27 on two runs of the same corpus, so 4.0 decides on rater
    # mood. Second, and worse, a mean hides the thing the gate is for. Twenty
    # excellent leaves and five incoherent ones average out to a pass, and the five
    # incoherent ones are precisely what must not ship — they are the clusters
    # carrying more than one intent.
    #
    # So gate on the TAIL: what share of leaves the raters placed at the bottom of
    # their own scale. That is comparable across raters who are systematically
    # generous or harsh, because it asks where each rater put this leaf relative to
    # the rest of their own judgements rather than against an imported number.
    weak_floor = cfg.gates.coherence_weak_below
    weak = [c for c in coherences if c < weak_floor]
    weak_share = (len(weak) / len(coherences)) if coherences else 0.0
    g1 = deps.gate(
        "p7_coherence", "p7",
        passed=(not coherences) or weak_share <= cfg.gates.coherence_max_weak_share,
        observed={"mean_coherence": round(mean_coh, 3) if coherences else None,
                  "n": len(coherences), "n_weak": len(weak),
                  "weak_share": round(weak_share, 4),
                  "weak_below": weak_floor,
                  "min": round(min(coherences), 2) if coherences else None},
        threshold={"max_weak_share": cfg.gates.coherence_max_weak_share,
                   "weak_below": weak_floor,
                   "note": "mean is reported, not gated — it hides the incoherent tail"},
        message=(f"blind coherence: mean {mean_coh:.2f}/5, "
                 f"{len(weak)}/{len(coherences)} leaves ({weak_share:.0%}) below {weak_floor}"
                 if coherences else "no coherence scores"),
        remediation="Low coherence means those clusters are carrying more than one intent — "
                    "split them in refinement or reduce K before delivering. Note the gate "
                    "reads the weak tail, not the mean: a good average with a bad tail is "
                    "exactly the case that must not ship.",
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
        # Read back the artifact that was just written, so the observer sees what
        # p7 DELIVERED rather than a payload assembled by hand beside it.
        "gates": {**gates, **_observe(
            deps, "p7",
            {"tree_naming": deps.load("tree_naming") if deps.has("tree_naming") else {}},
            gates={k: getattr(v, "message", "") for k, v in gates.items()})},
        "prescriptions": prescriptions,
        "completed_phases": ["p7"],
        "events": [
            f"P7: mean coherence {mean_coh:.2f}/5, {len(prescriptions)} prescriptions raised",
            f"P7: risk found independently: {independently_found}",
        ],
    }


def _name_leaves_governance_created(
    deps: Deps, new_labels: np.ndarray, cents: np.ndarray, n_new: int,
    old_labels: np.ndarray | None = None, churn: float = 0.20,
) -> None:
    """Name the leaves p8 just created, because p7's gate can no longer see them.

    `p7_all_leaves_named` is BLOCKING and it ran in p7 — before this phase
    exists. Governance does not only merge: on `live38` it executed 6
    `split_leaf` and 2 `isolate_leaf` prescriptions, taking the partition from 29
    leaves to 36. The seven new leaves were never named, and `p10_deliver` builds
    its name column from p7's namings plus governance RENAMES only — so **4,931
    rows, 9.9% of the delivered table, shipped with an empty `bu_leaf_name`**
    while the gate guaranteeing "all leaves named" had passed.

    A gate placed before the operation that invalidates it is not a guarantee.
    This closes the hole at its source rather than re-asserting the gate later:
    whatever governance creates gets a name in the same phase.
    """
    naming = deps.load("tree_naming") if deps.has("tree_naming") else {}
    namings = list(naming.get("namings", []))
    have = {int(n["leaf_id"]) for n in namings}
    live = [i for i in range(n_new) if int((new_labels == i).sum())]
    missing = [i for i in live if i not in have]

    # A SPLIT LEAVES TWO LEAVES NEEDING A NAME, NOT ONE. The new id is obviously
    # unnamed, but the REMNANT keeps a name the namer assigned to a cluster that
    # no longer exists — and on live38 the name went to the wrong half three
    # times. Leaf 8 kept 「汉字拼音查询」 while every one of its 122 remaining rows
    # is a 怎么写 (how-to-write) query and the actual pinyin rows moved to the new
    # leaf; leaf 23 kept 「汉字读音笔顺查询」 while its pronunciation rows likewise
    # left. A name that actively misdescribes its rows is worse than no name,
    # because nothing downstream can tell it is wrong.
    stale: list[int] = []
    if old_labels is not None and len(old_labels) == len(new_labels):
        for i in live:
            if i not in have:
                continue
            now = new_labels == i
            before = old_labels == i
            n_now = int(now.sum())
            if not n_now:
                continue
            kept = int((now & before).sum())
            # Fraction of the NAMED cluster that is still here, and fraction of
            # what is here that the namer actually saw. Either drifting far means
            # the name describes a different set of rows.
            held = kept / max(1, int(before.sum()))
            pure = kept / n_now
            if held < (1 - churn) or pure < (1 - churn):
                stale.append(i)
    if stale:
        deps.emit(f"  {len(stale)} leaf/leaves changed membership past {churn:.0%} — "
                  f"their p7 name no longer describes them: {stale}")
        namings = [n for n in namings if int(n["leaf_id"]) not in set(stale)]
    missing = sorted(set(missing) | set(stale))
    if not missing:
        return

    deps.emit(f"  naming {len(missing)} leaf/leaves after governance "
              f"({len(missing) - len(stale)} new, {len(stale)} re-named)")
    from ...ops.cards import build_naming_cards

    cfg = deps.cfg
    cards = build_naming_cards(
        deps.df, new_labels, deps.embedding("emb_hybrid"), cents,
        text_col=cfg.data.text_column, n_center=cfg.naming.card_center,
        n_random=cfg.naming.card_random, n_edge=cfg.naming.card_edge,
        n_ngrams=cfg.naming.card_top_ngrams, seed=cfg.seed_metric,
    )
    by_id = {c.leaf_id: c for c in cards}
    agent = NamerAgent(deps.agent_ctx(), suffix="_postgov")
    named = 0
    for lid in missing:
        card = by_id.get(lid)
        if card is None:
            continue
        last = ""
        for attempt in range(3):      # same retry the shard namer uses
            try:
                out = agent.run(card=card)
                d = out.model_dump()
                d["leaf_id"] = lid
                d["named_by"] = "namer_postgov@routed"
                namings.append(d)
                named += 1
                break
            except Exception as exc:  # noqa: BLE001
                last = type(exc).__name__
                if attempt < 2:
                    time.sleep(2 ** attempt)
        else:
            deps.emit(f"  ⚠ leaf {lid} could not be named after 3 attempts: {last}")

    naming["namings"] = sorted(namings, key=lambda n: int(n["leaf_id"]))
    deps.store.put_json("tree_naming", naming, producer="p8",
                        summary=f"{len(namings)} namings ({named} written after governance)")
    deps.cache_put("tree_naming", naming)
    deps.emit(f"  named {named}/{len(missing)} leaves affected by governance")


def _resolve_indistinguishable_leaves(deps: Any, naming: dict, leaf_labels: Any,
                                      leaf_family: Any, cents: Any) -> list[dict]:
    """Re-name delivered leaves that share a name, or record a merge.

    Runs on the DELIVERED partition, which is the only place it can run: p7
    audits the tree and p8 then splits it, so leaf 50 did not exist when the
    duplicate audit looked. live44 shipped two `汉字读音查询` and two
    `汉字笔画数查询`, and the split that produced the second pair was
    geometrically sound — lift 0.1565 over null, ARI 0.9887 — with no nameable
    difference.

    So this asks rather than assumes. One agent sees every colliding cluster's
    queries at once (never a peer's NAME — the firewall bans peer namings so the
    Phase 7 fan-out stays independent) and either names the difference or says
    there is none. `same_need` produces a `merge_leaves` prescription, which
    Phase 8 can now execute.

    Returns whatever is STILL colliding, so the caller gates on a re-measurement
    rather than on the attempt.
    """
    from ...ops.governance import indistinguishable_leaves

    groups = indistinguishable_leaves(leaf_labels, leaf_family, naming.get("namings", []))
    if not groups:
        return []

    deps.emit(f"  {len(groups)} name collision(s) among delivered leaves: "
              + "; ".join(f"{g['name']} = leaves {g['leaf_ids']}" for g in groups[:4]))

    from ...agents.roles import DisambiguatorAgent
    from ...ops.cards import build_naming_cards

    cfg = deps.cfg
    cards = {c.leaf_id: c for c in build_naming_cards(
        deps.df, leaf_labels, deps.embedding("emb_hybrid"), cents,
        text_col=cfg.data.text_column, n_center=cfg.naming.card_center,
        n_random=cfg.naming.card_random, n_edge=cfg.naming.card_edge,
        n_ngrams=cfg.naming.card_top_ngrams, seed=cfg.seed_metric,
    )}
    agent = DisambiguatorAgent(deps.agent_ctx(), suffix="_disambig")
    by_id = {int(n["leaf_id"]): n for n in naming.get("namings", [])}
    merges: list[dict] = []

    for g in groups:
        picked = [cards[i] for i in g["leaf_ids"] if i in cards]
        if len(picked) < 2:
            continue
        try:
            out = agent.run(cards=picked, collided_on=g["name"])
        except Exception as exc:  # noqa: BLE001
            deps.emit(f"  ⚠ could not disambiguate {g['name']!r} "
                      f"({type(exc).__name__}) — the collision stands")
            continue

        if getattr(out, "same_need", False):
            # An honest merge beats a laboured name nobody would use.
            merges.append({"leaf_ids": g["leaf_ids"], "name": g["name"],
                           "rationale": str(getattr(out, "rationale", ""))[:300]})
            deps.emit(f"  {g['name']!r}: one user need — merge prescribed for {g['leaf_ids']}")
            continue

        fresh = {int(n.leaf_id): str(n.name_zh or "").strip()
                 for n in (getattr(out, "namings", None) or []) if n.name_zh}
        # Only accept names that are actually DISTINCT — an agent that returns
        # the same string twice has not disambiguated anything, and accepting it
        # would let the gate below pass on a promise.
        if len(set(fresh.values())) == len(g["leaf_ids"]) and len(fresh) == len(g["leaf_ids"]):
            for lid, nm in fresh.items():
                if lid in by_id:
                    by_id[lid]["name_zh"] = nm
                    by_id[lid]["named_by"] = "namer_disambig@routed"
            deps.emit(f"  {g['name']!r} → " + " / ".join(fresh[i] for i in g["leaf_ids"]))
        else:
            deps.emit(f"  ⚠ {g['name']!r}: disambiguation returned "
                      f"{len(set(fresh.values()))} distinct name(s) for "
                      f"{len(g['leaf_ids'])} leaves — the collision stands")

    if merges:
        naming["indistinguishable_merges"] = merges
    naming["namings"] = sorted(by_id.values(), key=lambda n: int(n["leaf_id"]))
    # RE-MEASURE. The gate must not read the attempt.
    return indistinguishable_leaves(leaf_labels, leaf_family, naming["namings"])


def _name_delivered_families(deps: Any, naming: dict, leaf_labels: Any,
                             leaf_family: Any) -> None:
    """Give every DELIVERED family its own name, from the leaves it now holds.

    The tree auditor names families, but it names the Phase 7 tree — before
    governance merged 18 into 14 and isolated them back out to 23. Those id
    spaces differ and a delivered family routinely spans several audit families,
    so `report/_shape.family_names` could only emit a composition label:
    `混合·主要成分「词语含义查询」45%`. That is a diagnostic wearing a name's
    clothes, and it is what a reader sees as the family's title in headings,
    tables and a CSV column.

    Named here rather than in Phase 7 for the same reason the leaves are: this
    is the first point at which the delivered partition exists.
    """
    import numpy as np

    if leaf_labels is None or leaf_family is None:
        return
    lab = np.asarray(leaf_labels)
    fam = np.asarray(leaf_family)
    by_leaf = {int(n["leaf_id"]): n for n in naming.get("namings", [])}
    sizes = dict(zip(*[x.tolist() for x in np.unique(lab, return_counts=True)]))

    members: dict[int, list[dict]] = {}
    for leaf_id in sorted(sizes):
        if int(leaf_id) >= len(fam):
            continue
        f = int(fam[int(leaf_id)])
        n = by_leaf.get(int(leaf_id), {})
        members.setdefault(f, []).append({
            "leaf_id": int(leaf_id),
            "name_zh": n.get("name_zh", ""),
            "user_need": n.get("user_need", ""),
            "n_rows": int(sizes[leaf_id]),
        })
    if not members:
        return

    from ...agents.roles import FamilyNamerAgent

    agent = FamilyNamerAgent(deps.agent_ctx(), suffix="_family")
    out, failed = [], []
    for fid, leaves in sorted(members.items()):
        sibs = [ (by_leaf.get(l[0]["leaf_id"], {}) or {}).get("name_zh", "")
                 for f2, l in sorted(members.items()) if f2 != fid and l ]
        try:
            rec = agent.run(family_id=fid, leaves=leaves, siblings=sibs[:12]).model_dump()
        except Exception:  # noqa: BLE001
            failed.append(fid)
            continue
        rec["family_id"] = int(fid)
        rec["leaf_ids"] = [l["leaf_id"] for l in leaves]
        rec["named_by"] = "namer_family@routed"
        out.append(rec)

    # A STAND-IN NAME IS NOT A NAME. The offline heuristic returns
    # "[offline-heuristic] 未命名分组" for every family; persisting that would put
    # a placeholder in the title position and, worse, SUPPRESS the disclosure the
    # audit-derived path makes — a family the tree audit never covered is
    # supposed to read "树审计未覆盖 (治理新建)", and a stand-in name hides that.
    # Same rule as `render --no-agents`: complete-looking prose no model wrote is
    # worse than a marked hole.
    provider = str(getattr(getattr(agent, "ctx", None), "registry", None)
                   and getattr(agent.ctx.registry, "provider", "") or "")
    if out and provider and provider != "offline":
        naming["families_final"] = out
    elif out:
        deps.emit(f"  delivered-family names NOT recorded — provider is "
                  f"{provider or 'unknown'}; families keep their audit-derived label")
    # A family that could not be named must be visible, not silently absent:
    # `_shape.family_names` falls back to the composition label for it, and a
    # reader deserves to know which of the two they are looking at.
    deps.emit(f"  named {len(out)}/{len(members)} delivered families"
              + (f" — {len(failed)} failed: {failed}" if failed else ""))


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
        relabels = {int(k): v for k, v in detail["relabelled"].items()}
        deps.cache_put("leaf_relabels", relabels)
        # PERSIST IT TOO. `p10_deliver` reads this through `deps.recover(...)`,
        # whose whole purpose is to fall back to the artifact store on a resumed
        # run — but nothing ever WROTE the artifact, so the fallback was dead and
        # returned the `{}` default. Governance renames then vanished from the
        # delivered column on any resume, silently, exactly the failure
        # `recover`'s own docstring describes.
        deps.store.put_json(
            "leaf_relabels", {str(k): v for k, v in relabels.items()},
            producer="p7", summary=f"{len(relabels)} governance renames")

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
        _name_leaves_governance_created(deps, new_labels, cents, n_new, old_labels=labels)
    # DELIVERED-PARTITION CHECKS. Both of these can only run here: p7 audits the
    # tree and p8 then splits it, so a leaf governance created was never audited.
    still_colliding: list[dict] = []
    try:
        _nm = deps.load("tree_naming")
        still_colliding = _resolve_indistinguishable_leaves(
            deps, _nm, new_labels, new_family, cents)
        deps.store.put_json("tree_naming", _nm, producer="p8",
                            summary="namings after disambiguation")
        deps.cache_put("tree_naming", _nm)
    except Exception as exc:  # noqa: BLE001
        deps.emit(f"  ⚠ leaf disambiguation unavailable ({type(exc).__name__}) — "
                  "name collisions, if any, are unresolved and the gate below says so")

    deps.gate(
        "p8_leaves_are_distinguishable", "p8",
        passed=not still_colliding,
        observed={"n_collisions": len(still_colliding),
                  "collisions": [{"family_id": g["family_id"], "name": g["name"],
                                  "leaf_ids": g["leaf_ids"]} for g in still_colliding]},
        threshold={"n_collisions": 0},
        message=(f"{len(still_colliding)} delivered leaf name(s) are shared by two or "
                 f"more leaves in the same family: "
                 + "; ".join(f"{g['name']} = {g['leaf_ids']}" for g in still_colliding)
                 if still_colliding else
                 "every delivered leaf is distinguishable from its siblings by name"),
        remediation="A reader choosing between two identically-named leaves cannot "
                    "choose. Either the namer must name the difference, or the leaves "
                    "must merge (`merge_leaves`).",
        warn_only=True,
    )

    # The delivered partition exists only now, so this is the first point at
    # which a family can be named after the leaves it actually contains.
    if getattr(deps.cfg.naming, 'name_delivered_families', True):
        try:
            # PERSIST IT. `_name_delivered_families` mutates the dict it is
            # given, and this used to hand it a throwaway `deps.load(...)` — so
            # the K12 demo logged "named 32/32 delivered families" and shipped
            # `families_final: []`, with the reports still carrying
            # `混合·主要成分「X」N%`. A naming that is not written back is a
            # naming that did not happen.
            _fam_nm = deps.load("tree_naming")
            _name_delivered_families(deps, _fam_nm, new_labels, new_family)
            deps.store.put_json("tree_naming", _fam_nm, producer="p8",
                                summary=f"{len(_fam_nm.get('families_final') or [])} "
                                        "delivered families named")
            deps.cache_put("tree_naming", _fam_nm)
        except Exception as exc:  # noqa: BLE001
            deps.emit(f'  ⚠ delivered-family naming unavailable ({type(exc).__name__}) — families fall back to the composition label')
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
        # p8 REWRITES the partition, and everything written before it describes a
        # tree that no longer exists — the single richest defect family in this
        # project's history. It is the phase most worth a second pair of eyes.
        "gates": {gate.name: gate,
                  **_observe(deps, "p8", {"governance": deps.load("governance")
                                          if deps.has("governance") else detail})},
        "prescriptions": prescriptions,
        "completed_phases": ["p8"],
        "events": [
            f"P8: families {before.get('n_families', 0):.0f} → {len(np.unique(new_family))}; "
            f"fragmentation delta {deltas.get('template_fragmentation', 0):+.3f}"
        ],
    }
