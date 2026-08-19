"""Phase 11 — assembling the deliverables.

Reports here are *projections of recorded state*, not prose written from memory
at the end of a run.  The failure-history section is a rendering of the rejected
options inside each :class:`DecisionRecord`; the governance section is a
rendering of the prescription ledger; the metrics tables come from the uniform
panel with its footnotes attached.  That is deliberate: a report assembled by
recalling what happened will quietly omit the parts nobody enjoyed, and those
are the parts that make the rest credible.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..artifacts import ArtifactRef
from ..ops import viz
from .i18n import decision_question
from ..state import PipelineState


def build_all_reports(state: PipelineState, deps: Any) -> dict[str, ArtifactRef]:
    """Write every figure, report, and the executed notebook."""
    refs: dict[str, ArtifactRef] = {}
    zh = getattr(deps.cfg, "report_language", "zh") == "zh"

    # The notebook runs first because executing it *is* how the figure suite is
    # produced. The report then embeds the very images a reader can regenerate by
    # re-running a cell — the same figure, not a look-alike drawn by other code.
    # `_build_figures` fills only the slots the notebook did not (or could not).
    nb_ref, nb_figs = None, {}
    try:
        if zh:
            from .zh_notebook import build as build_zh_nb

            nb_ref = build_zh_nb(state, deps)
            nb_figs = _register_notebook_figures(deps)
        else:
            from .notebook import build_walkthrough

            nb_ref = build_walkthrough(state, deps)
    except Exception as exc:  # noqa: BLE001
        deps.emit(f"  notebook build skipped: {exc}")
    if nb_ref is not None:
        refs["notebook"] = nb_ref

    figs = _build_figures(state, deps, have=set(nb_figs))
    figs.update(nb_figs)          # notebook output wins any contested slot
    refs.update(figs)
    if zh:
        from .zh_bottomup import build as zh_bottomup

        refs["report_bottomup"] = deps.store.put_markdown(
            "自下而上聚类最终报告", zh_bottomup(state, deps, figs),
            producer="p11", summary="自下而上路线: 表征 → 树 → 治理 → 部署")
    else:
        refs["report_bottomup"] = deps.store.put_markdown(
            "Report_BottomUp_Approach", bottomup_report(state, deps, figs),
            producer="p11", summary="bottom-up route: representation → tree → governance → deployment")
    refs["report_topdown"] = deps.store.put_markdown(
        "Report_TopDown_Approach", topdown_report(state, deps),
        producer="p11", summary="top-down route: taxonomy → gold → classifier → validation")
    refs["report_panel"] = deps.store.put_markdown(
        "Report_Uniform_Panel", panel_report(state, deps, figs),
        producer="p11", summary="cross-candidate comparison under one measurement harness")
    refs["leaf_catalogue"] = deps.store.put_markdown(
        "Leaf_Catalogue", leaf_catalogue(state, deps),
        producer="p11", summary="every leaf with its user_need definition")
    return refs


# ==========================================================================
# Figures
# ==========================================================================

#: Figures the executed notebook writes, and the report slot each one fills.
#: The first four displace a `viz.*` drawing of the same quantity; the last two
#: have no `viz` equivalent and are additions.
NOTEBOOK_FIGURES = {
    "fig1_ksweep":       "fig_k_sweep",
    "fig1b_ksweep_metrics": "fig_k_sweep_metrics",
    "fig2_alpha":        "fig_alpha",
    "fig3_battery":      "fig_battery",
    "fig4_spaces":       "fig_umap",
    "fig5_intent_split": "fig_intent_split",
    "fig6_panel":        "fig_panel",
}


def _register_notebook_figures(deps: Any) -> dict[str, ArtifactRef]:
    """Register the PNGs the notebook wrote while executing."""
    out: dict[str, ArtifactRef] = {}
    for filename, slot in NOTEBOOK_FIGURES.items():
        path = deps.store.gen_dir / f"{filename}.png"
        if path.exists():
            out[slot] = deps.store.register_file(slot, path, "figure", producer="p11-notebook")
    if out:
        deps.emit(f"  图表: notebook 现场生成 {len(out)} 张")
    return out


def _build_figures(state: PipelineState, deps: Any, have: set[str] | None = None) -> dict[str, ArtifactRef]:
    """Draw the figures the notebook did not produce.

    `have` names slots the executed notebook already filled; drawing them again
    would put two different pictures of the same number in one deliverable.
    """
    have = have or set()
    out: dict[str, ArtifactRef] = {}
    lang = deps.cfg.domain.language

    def _reg(name: str, path: Path | None) -> None:
        if path and Path(path).exists():
            out[name] = deps.store.register_file(name, path, "figure", producer="p11")


    def _try(slot: str | None, label: str, fn) -> None:
        """Draw one figure, unless the notebook already filled its slot."""
        if slot and slot in have:
            return
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            deps.emit(f"  fig {label} skipped: {exc}")

    _try("fig_k_sweep", "k_sweep", lambda: _reg(
        "fig_k_sweep", viz.plot_k_sweep(
            deps.load("granularity")["k_sweep"], deps.store.put_figure_path("fig_k_sweep"),
            chosen_k=state.get("family_k"), language=lang)))

    _try("fig_alpha", "alpha", lambda: _reg(
        "fig_alpha", viz.plot_alpha_decision(
            deps.load("representation")["alpha_sweep"]["rows"],
            deps.store.put_figure_path("fig_alpha"),
            chosen=state.get("chosen_alpha"), language=lang)))

    _try("fig_panel", "panel", lambda: _reg(
        "fig_panel", viz.plot_panel(
            deps.load("metrics_panel")["table"],
            deps.store.put_figure_path("fig_panel"), language=lang)))

    # No notebook equivalent: the refinement trace and the per-template spread
    # answer questions none of the notebook figures cover, so they always draw.
    _try(None, "refinement", lambda: _reg(
        "fig_refinement", viz.plot_refinement(
            deps.load("hierarchy_meta")["refinement_history"],
            deps.store.put_figure_path("fig_refinement"), language=lang)))

    def _spread() -> None:
        from ..ops.cards import template_spread

        masks = deps.template_masks(trusted=False)
        fam, labels = deps.leaf_family_final(), deps.leaf_labels_final()
        _reg("fig_template_spread", viz.plot_template_spread(
            template_spread(masks, fam[labels]),
            deps.store.put_figure_path("fig_template_spread"), language=lang))

    _try(None, "template_spread", _spread)

    # The audit-trail figures. Their inputs are the run's own decision and gate
    # records, so they need no artifacts and cannot fail on a partial run.
    _try(None, "decision_chain", lambda: _reg(
        "fig_decision_chain", viz.plot_decision_chain(
            [d.model_dump() if hasattr(d, "model_dump") else dict(d)
             for d in state.get("decisions", [])],
            deps.store.put_figure_path("fig_decision_chain"), language=lang,
            localise=(lambda x: decision_question(x, lang)))))

    _try(None, "gates", lambda: _reg(
        "fig_gates", viz.plot_gates(
            {k: (g.model_dump() if hasattr(g, "model_dump") else dict(g))
             for k, g in (state.get("gates", {}) or {}).items()},
            deps.store.put_figure_path("fig_gates"), language=lang)))

    def _umap() -> None:
        fam, labels = deps.leaf_family_final(), deps.leaf_labels_final()
        _reg("fig_umap", viz.plot_umap(
            deps.embedding("emb_hybrid"), fam[labels], deps.store.put_figure_path("fig_umap"),
            seed=deps.cfg.seed_viz, language=lang, title="Corpus layout, coloured by family"))

    if not deps.cfg.fast_mode:
        _try("fig_umap", "umap", _umap)

    return out


# ==========================================================================
# Shared blocks
# ==========================================================================

def _header(state: PipelineState, deps: Any, title: str, subtitle: str) -> str:
    return "\n".join([
        f"# {title}",
        f"## {subtitle}",
        "",
        f"**Run** `{state.get('run_id')}` · **generation** {state.get('generation')} · "
        f"**domain** `{deps.cfg.domain.key}` · **config hash** `{deps.cfg.config_hash}`",
        "",
        f"> {deps.registry.provenance_note()}",
        "",
    ])


def _decision_table(state: PipelineState, phases: tuple[str, ...]) -> str:
    rows = [d for d in state.get("decisions", []) if d.phase.startswith(phases)]
    if not rows:
        return "_No decisions recorded for these phases._"
    out = ["| phase | question | choice | decided by | decisive metric |", "|---|---|---|---|---|"]
    for d in rows:
        out.append(
            f"| `{d.phase}` | {d.question} | **{d.choice}** | {d.decided_by} | "
            f"{', '.join(d.decisive_metrics) or '—'} |"
        )
    return "\n".join(out)


def _failure_history(state: PipelineState, phases: tuple[str, ...]) -> str:
    """What lost, and why.  A mandatory section, not an appendix."""
    rows = [d for d in state.get("decisions", []) if d.phase.startswith(phases) and d.rejected]
    if not rows:
        return "_Nothing was rejected in these phases — which usually means not enough was tried._"
    parts = []
    for d in rows:
        parts.append(f"**{d.question}** — chose `{d.choice}`.\n")
        parts.append("| rejected option | why | metrics |")
        parts.append("|---|---|---|")
        for r in d.rejected[:10]:
            m = r.get("metrics") or {}
            m_txt = ", ".join(f"{k}={v}" for k, v in m.items() if v is not None) or "—"
            parts.append(f"| `{r.get('option', r.get('name', '?'))}` | {r.get('why_rejected', r.get('why', ''))} | {m_txt} |")
        parts.append("")
    return "\n".join(parts)


def _gate_table(state: PipelineState) -> str:
    gates = state.get("gates", {})
    if not gates:
        return "_No gates evaluated._"
    icon = {"passed": "PASS", "warned": "WARN", "failed": "FAIL", "rejected": "VETO", "skipped": "skip"}
    out = ["| gate | phase | status | observed | threshold |", "|---|---|---|---|---|"]
    for name, g in sorted(gates.items()):
        out.append(
            f"| `{name}` | {g.phase} | **{icon.get(g.status, g.status)}**"
            f"{' (blocking)' if g.blocking else ''} | "
            f"{json.dumps(g.observed, ensure_ascii=False)} | {json.dumps(g.threshold, ensure_ascii=False)} |"
        )
    return "\n".join(out)


def _fig(figs: dict[str, ArtifactRef], name: str, caption: str) -> str:
    if name not in figs:
        return ""
    return f"\n![{caption}]({Path(figs[name].path).name})\n\n*{caption}*\n"


def _panel_table_md(table: dict[str, Any], metrics: list[str] | None = None) -> str:
    auth = {m["name"]: m["authority"] for m in table["metrics"]}
    names = metrics or [m["name"] for m in table["metrics"]]
    head = "| candidate | " + " | ".join(f"{n}<br><sub>{auth.get(n, '')}</sub>" for n in names) + " |"
    sep = "|---" * (len(names) + 1) + "|"
    lines = [head, sep]
    for r in table["rows"]:
        cells = []
        for n in names:
            v = r.get(n)
            cells.append("—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.4g}" if isinstance(v, float) else str(v))
        lines.append(f"| `{r['subject']}` | " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ==========================================================================
# Reports
# ==========================================================================

def bottomup_report(state: PipelineState, deps: Any, figs: dict[str, ArtifactRef]) -> str:
    rep = deps.load("representation") if deps.has("representation") else {}
    gran = deps.load("granularity") if deps.has("granularity") else {}
    meta = deps.load("hierarchy_meta") if deps.has("hierarchy_meta") else {}
    naming = deps.load("tree_naming") if deps.has("tree_naming") else {}
    gov = deps.load("governance") if deps.has("governance") else {}
    dep = deps.load("deployment") if deps.has("deployment") else {}
    panel = deps.load("metrics_panel") if deps.has("metrics_panel") else {}

    alpha = state.get("chosen_alpha", 0.0)
    surface = alpha ** 2 / (1 + alpha ** 2)
    p: list[str] = [_header(state, deps, "Bottom-Up Route",
                    "Unsupervised structure discovery, blind naming, and executed governance")]

    p += ["## 1. Executive summary", ""]
    if panel:
        p += [_panel_table_md(panel["table"],
                              ["n_clusters", "stability_ari", "template_fragmentation",
                               "heldout_reproduction", "silhouette", "coherence"]), ""]
    p += [
        f"- **Encoder**: `{state.get('chosen_encoder')}` — chosen on replay stability over this "
        "corpus's own clustering task, not on a leaderboard score.",
        f"- **Alpha**: `{alpha}`, which gives the phrasing block **{surface * 100:.1f}%** of the "
        "cosine. The algebra is `cos(H,H') = (cos_semantic + α²·cos_surface)/(1+α²)`, so phrasing "
        "votes with weight **α², not α** — a tie-breaker, not a co-equal signal.",
        f"- **Algorithm**: `{state.get('chosen_algorithm')}` — won a six-algorithm battery run "
        "through one identical measurement harness.",
        f"- **Shape**: {meta.get('n_families', '?')} families → **{meta.get('n_leaves', '?')} leaves**, "
        f"held-out structure reproduction **{meta.get('heldout_reproduction', {}).get('agreement', '?')}**.",
        "",
    ]

    p += ["## 2. Representation", "",
          "### 2.1 Encoder bake-off", ""]
    if rep.get("bakeoff"):
        b = rep["bakeoff"]
        p += ["| encoder | dim | stability (decisive) | fragmentation (decisive) | silhouette (advisory) |",
              "|---|---|---|---|---|"]
        for r in b["rows"]:
            if r.get("status") != "ok":
                p.append(f"| `{r['encoder']}` | — | unavailable | — | — |")
                continue
            p.append(f"| `{r['encoder']}` | {r['dim']} | {r['stability_ari']} | "
                     f"{r['template_fragmentation']} | {r['silhouette']} |")
        p += ["", f"**Selection rule:** {b['chosen_by']}."]
        if b.get("silhouette_disagrees"):
            p.append(f"**Silhouette would have chosen `{b['silhouette_would_have_chosen']}`** — "
                     "recorded and overruled, for the reason in §2.3.")
        p.append("")

    p += ["### 2.2 The sparse block", "",
          f"Character {tuple(deps.cfg.domain.char_ngram_range)}-gram TF-IDF over "
          f"{rep.get('sparse', {}).get('vocab_size', '?')} features, compressed by SVD to "
          f"{rep.get('sparse', {}).get('n_components', '?')} dimensions "
          f"(explained variance {rep.get('sparse', {}).get('explained_variance', 0):.3f}). "
          "The SVD is compression, not modelling — it exists so a sparse phrasing matrix can "
          "share one cosine space with a dense semantic one.", ""]

    p += ["### 2.3 The alpha sweep, and why silhouette has no vote", ""]
    if rep.get("alpha_sweep"):
        s = rep["alpha_sweep"]
        p += ["| alpha | phrasing's share of cosine | fragmentation (decisive) | stability (decisive) | silhouette (advisory) |",
              "|---|---|---|---|---|"]
        for r in s["rows"]:
            mark = " **←chosen**" if r["alpha"] == alpha else ""
            p.append(f"| {r['alpha']}{mark} | {r['surface_vote_share'] * 100:.1f}% | "
                     f"{r['template_fragmentation']} | {r['stability_ari']} | {r['silhouette']} |")
        # Explain the selection rule explicitly. Without this a reader sees a
        # candidate with lower fragmentation losing to the chosen one and has no
        # way to tell a considered tie-break from a bug — which is corrosive in
        # exactly the section that exists to justify the choice.
        p += ["", f"**Selection rule:** {s.get('chosen_by', 'lowest fragmentation')}."]
        if s.get("contenders"):
            band = s.get("tie_band")
            p += [
                "",
                f"Fragmentation differences inside a {band:.0%} band are treated as a tie, because "
                "a two-percent gap in this metric is within run-to-run noise and is not worth "
                f"overriding a real stability difference. The contenders were "
                f"`{s['contenders']}`; among them **alpha={alpha}** had the highest replay "
                "stability, which is what broke the tie.",
            ]
        p += ["",
              "Silhouette measures whether clusters are tight and well separated. Queries that share "
              "a phrasing template are the tightest possible cluster — so optimising silhouette "
              "systematically selects representations that split one intent into several "
              "phrasing-shaped families, which is the exact failure this phase exists to prevent. "
              "It is computed, reported, and given no vote.", ""]
        if s.get("silhouette_disagrees"):
            p += [f"> Here it would have chosen **alpha={s['silhouette_would_have_chosen']}** "
                  f"instead of **{alpha}**.", ""]
    p += [_fig(figs, "fig_alpha", "The alpha decision. Left: the metrics that decided it. Right: the metric that did not.")]

    p += ["## 3. Granularity", ""]
    if gran.get("triangulation"):
        t = gran["triangulation"]
        p += ["Three independent estimators of the family scale:", "",
              "| estimator | value |", "|---|---|"]
        for k, v in t["estimates"].items():
            p.append(f"| {k} | {v} |")
        p += ["", f"**Chosen K = {t['chosen_family_k']}** by *{t['chosen_by']}*.", ""]
        if not t["converged"]:
            p += [f"> ⚠️ {t['divergence_note']}", ""]
    p += [_fig(figs, "fig_k_sweep", "Stability and silhouette across K.")]

    p += ["## 4. The tree", "",
          "A single stable global partition on top, a locally chosen split per family below. "
          "Asking one global K to be both stable and fine-grained does not work — at leaf-level K "
          "the replay stability of a one-shot partition collapses. Splitting the question in two "
          "is the fix.", "",
          f"- families: **{meta.get('n_families', '?')}**",
          f"- leaves after refinement: **{meta.get('n_leaves', '?')}**",
          f"- minimum leaf size enforced: {meta.get('min_leaf_size_applied', '?')} rows",
          f"- refinement converged: **{meta.get('converged')}** in {len(meta.get('refinement_history', []))} rounds",
          f"- held-out structure reproduction: **{meta.get('heldout_reproduction', {}).get('agreement')}**", ""]
    if meta.get("refinement_history"):
        p += ["| round | merges | splits | rows moved | leaves | silhouette |", "|---|---|---|---|---|---|"]
        for h in meta["refinement_history"]:
            p.append(f"| {h['round']} | {h['merges']} | {h['splits']} | {h['moved_fraction'] * 100:.2f}% | "
                     f"{h['n_leaves']} | {h['silhouette']} |")
        p.append("")
    p += [_fig(figs, "fig_refinement", "Refinement converges on movement, not on a round count.")]

    p += ["## 5. Blind naming", "",
          f"{deps.cfg.naming.n_naming_agents} naming agents worked disjoint shards in parallel. Each saw "
          f"{deps.cfg.naming.card_center} centre members, {deps.cfg.naming.card_random} random members, and "
          f"{deps.cfg.naming.card_edge} **edge** members — the ones that barely belong, included so that "
          "impurity is visible and coherence scores mean something.", "",
          "Nothing else was in the card: no taxonomy, no legacy labels, no other agent's answer. "
          "That is enforced two ways. The fan-out uses `Send`, so a worker's state contains only its "
          "own payload and the parent state is structurally unreachable. And every card passes a "
          "blindness firewall that enforces the card contract — a fixed field whitelist, so a "
          "smuggled `legacy_label` or `taxonomy_hint` cannot pass — plus a lexical scan of every "
          "field that is not corpus-derived.", "",
          "Member queries themselves are exempt from the lexical scan, deliberately. A member query "
          "cannot anchor a namer: it is the thing being judged. And because good category names come "
          "from their domain's own vocabulary, label strings and ordinary query words overlap "
          "heavily — scanning corpus text lexically flags real queries as leaks and silently drops "
          "clusters from the naming pass.", ""]
    if naming:
        p += [f"- mean coherence: **{naming.get('mean_coherence')}/5**", ""]
        ind = naming.get("independent_risk_discovery", {})
        if ind:
            p += ["### Risk discovery", "",
                  f"- flagged by a blind namer: `{ind.get('namer_flagged_leaves')}`",
                  f"- flagged by the independent risk sentinel: `{ind.get('sentinel_flagged_leaves')}`",
                  f"- both agree on: `{ind.get('both_agree_on')}`", "",
                  "An agent that was never told to look for risk finding it anyway is evidence. "
                  "The seeded pre-screen finding what it was seeded with is bookkeeping.", ""]

    p += ["## 6. Governance — executed, not recommended", ""]
    if gov:
        p += [f"> {gov.get('mechanism', '')}", "",
              "Two mechanisms, and the distinction is reported rather than blurred. Family "
              "merges and risk isolations are **lookup-table remaps**: leaf assignments and "
              "centroids are untouched, the pre-governance column is retained, and the change "
              "is reversible. A leaf **split** genuinely re-partitions — a new leaf is appended "
              "and the shipped centroid matrix differs — so it is recorded as such. The "
              "pre-governance partition is kept under its own artifact either way.", ""]
        ledger = gov.get("ledger", [])
        if ledger:
            p += ["| id | kind | targets | status | column changed | metric deltas |", "|---|---|---|---|---|---|"]
            for r in ledger:
                p.append(f"| `{r['id']}` | {r['kind']} | {r['targets']} | **{r['status']}** | "
                         f"`{r['executed_column'] or '—'}` | {json.dumps(r['metric_deltas'], ensure_ascii=False)} |")
            p.append("")
        ex = gov.get("execution", {})
        if ex.get("metric_deltas"):
            p += [f"Effect on the panel: `{json.dumps(ex['metric_deltas'], ensure_ascii=False)}`", ""]
        declined = [r for r in ledger if r["status"] == "declined"]
        if declined:
            p += ["### Deliberate non-merges", "",
                  "Splits the audit judged to have a real basis. Recorded so they read as decisions "
                  "rather than oversights:", ""]
            p += [f"- `{r['id']}` targets `{r['targets']}` — {r['decline_reason']}" for r in declined]
            p.append("")
    p += [_fig(figs, "fig_template_spread", "Where each phrasing family landed after governance.")]

    p += ["## 7. Deployment", ""]
    if dep:
        r = dep.get("routing", {})
        p += [f"- inference: `{dep.get('inference')}`",
              f"- model size: **{dep.get('model_bytes', 0) / 1024:.0f} KB** (a centroid matrix)",
              f"- rows routed directly: **{r.get('n_direct')}**; to fallback: **{r.get('n_fallback')}** "
              f"(**{r.get('ambiguous_rate', 0) * 100:.1f}%** below margin {r.get('threshold')})", "",
              "The ambiguous share is reported rather than hidden. Semantic boundaries are genuinely "
              "softer than phrasing boundaries, so a double-digit rate is a property of the problem; "
              "concealing it would only move the surprise to production.", ""]
        if dep.get("live_demo"):
            p += ["### Live routing demo", "",
                  "| query | leaf | name | margin | routed |", "|---|---|---|---|---|"]
            for d in dep["live_demo"]:
                p.append(f"| {d['query']} | {d['leaf']} | {d['leaf_name']} | {d['margin']} | {d['routed']} |")
            p.append("")
        if dep.get("deterministic_exemplars"):
            p += ["### Deterministic exemplars", "",
                  "One example per phrasing family, taken at the **median index of the hit set**. "
                  "Nobody — human or agent — chose these, which is the entire point.", "",
                  "| family | hits | exemplar (median index) |", "|---|---|---|"]
            for e in dep["deterministic_exemplars"][:12]:
                p.append(f"| {e['pattern']} | {e['n_hits']} | {e['exemplar']} |")
            p.append("")

    p += ["## 8. What we rejected", "", _failure_history(state, ("p3", "p4", "p5", "p6")), ""]
    p += ["## 9. Decisions", "", _decision_table(state, ("p3", "p4", "p5", "p6", "p7", "p8")), ""]
    p += ["## 10. Quality gates", "", _gate_table(state), ""]
    p += ["## 11. What these numbers do not mean", "",
          "- **Distillation accuracy measures learnability, not correctness.** A high score means "
          "the clusters are a learnable function of the representation. It says nothing about "
          "whether a human would draw the same boundaries; that requires the gold set and "
          "adversarial validation of the top-down route.",
          "- **Silhouette is advisory throughout.** Where it appears, it is context, not evidence.",
          "- **Template fragmentation is negatively correlated with cluster count.** Read it beside "
          "`n_clusters`; a partition with fewer clusters fragments less by construction.",
          "- **Held-out reproduction tests structural stability, not semantic validity.** A "
          "reproducible partition of the wrong thing is still the wrong thing.",
          "- **Clustering is structurally blind to pragmatic intents.** Where two queries are "
          "phrased alike and want different things, no representation separates them. Those "
          "categories belong to the top-down route and are delivered alongside, not merged in.", ""]
    return "\n".join(p)


def topdown_report(state: PipelineState, deps: Any) -> str:
    tax = deps.load("taxonomy") if deps.has("taxonomy") else {}
    agree = deps.load("gold_agreement") if deps.has("gold_agreement") else {}
    metrics = deps.load("topdown_metrics") if deps.has("topdown_metrics") else {}
    adv = deps.load("adversarial_validation") if deps.has("adversarial_validation") else {}

    p = [_header(state, deps, "Top-Down Route",
                 "Intent taxonomy, gold standard, hybrid classifier, adversarial validation")]
    p += ["## 1. Why this route exists at all", "",
          "Some intents are invisible in the wording. Two queries can be phrased near-identically "
          "and want opposite things — a verdict versus a definition, solve-this versus explain-this. "
          "No representation separates those, so unsupervised clustering will never surface them "
          "regardless of how good the embedding is. This route owns that half of the problem, and "
          "its labels are delivered **beside** the bottom-up labels rather than merged into them.", ""]

    t = tax.get("taxonomy", {})
    nodes = t.get("nodes", [])
    p += ["## 2. The taxonomy", "",
          f"- L1 intents: **{len([n for n in nodes if n.get('level', 1) == 1])}**",
          f"- adjudication rules: **{len(t.get('rules', []))}**",
          f"- classes marked structurally invisible to clustering: "
          f"**{len([n for n in nodes if n.get('pragmatic_only')])}**", ""]
    if nodes:
        p += ["| code | name | definition | user_need | invisible to clustering |", "|---|---|---|---|---|"]
        for n in nodes[:60]:
            p.append(f"| `{n.get('code')}` | {n.get('name')} | {n.get('definition', '')[:110]} | "
                     f"{n.get('user_need', '')[:110]} | {'yes' if n.get('pragmatic_only') else ''} |")
        p.append("")
    if tax.get("critique", {}).get("findings"):
        p += ["### Critic findings", "", "| kind | classes | defect | fix |", "|---|---|---|---|"]
        for f in tax["critique"]["findings"][:20]:
            p.append(f"| {f.get('kind')} | {f.get('classes')} | {f.get('defect', '')[:110]} | {f.get('fix', '')[:110]} |")
        p.append("")

    p += ["## 3. Gold standard", ""]
    if agree.get("agreement"):
        a = agree["agreement"]
        p += [f"- double-annotated rows: **{a['n']}**",
              f"- raw agreement: **{a['raw_agreement']}**",
              f"- Cohen's κ: **{a['kappa']}**",
              f"- disagreements sent to the referee: **{a['n_disagreements']}**", "",
              "Both numbers are reported because κ alone misleads under class skew: on a corpus "
              "where one class dominates, high raw agreement can coexist with mediocre κ, and the "
              "pair tells you which situation you are in.", ""]
        if agree.get("new_rules"):
            p += ["### Rules the referee drafted", "",
                  "Each of these closes a gap that a real disagreement exposed. This is the "
                  "taxonomy's procedural memory — the rule set is not the one we started with.", "",
                  "| id | when | then | drafted because |", "|---|---|---|---|"]
            for r in agree["new_rules"]:
                p.append(f"| `{r['id']}` | {r['when'][:100]} | {r['then']} | {r.get('added_because', '')[:70]} |")
            p.append("")

    p += ["## 4. Classifier", ""]
    if metrics:
        p += [f"- cross-validated accuracy: **{metrics.get('cv_accuracy')}**",
              f"- macro-F1: **{metrics.get('macro_f1')}**",
              f"- expected calibration error: **{metrics.get('ece')}**",
              f"- classes trained: {metrics.get('n_classes')}, rows: {metrics.get('n_train')}"
              f" (dropped {metrics.get('n_dropped_rare', 0)} rows in classes too rare to cross-validate)", ""]
        if metrics.get("population_weighted_accuracy") is not None:
            p += [f"- population-weighted accuracy: **{metrics['population_weighted_accuracy']}**", ""]
        p += ["The head is linear by choice. Tree ensembles fed raw embedding coordinates must "
              "reconstruct directional similarity from axis-aligned splits, and a large label set "
              "spreads the per-class boosting budget too thin; a linear head reads the geometry "
              "directly. Calibration is reported because Phase 10 routes on confidence — a model "
              "that says 0.9 and is right 0.6 of the time makes the routing threshold meaningless.", ""]

    p += ["## 5. Adversarial validation", ""]
    if adv:
        p += [f"> {adv.get('method', '')}", "",
              f"- labels attacked: **{adv.get('n_attacked')}**",
              f"- judged wrong: **{adv.get('n_wrong')}**; defensible-but-arguable: **{adv.get('n_defensible')}**",
              f"- **estimated accuracy: {adv.get('estimated_accuracy')}**", "",
              "This number is lower and more trustworthy than cross-validated accuracy. CV measures "
              "agreement with the gold set the model was fitted to; this measures survival against "
              "an agent whose instruction was to prove the label wrong.", ""]

    p += ["## 6. Decisions", "", _decision_table(state, ("p2",)), ""]
    p += ["## 7. What we rejected", "", _failure_history(state, ("p2",)), ""]
    p += ["## 8. Quality gates", "", _gate_table(state), ""]
    return "\n".join(p)


def panel_report(state: PipelineState, deps: Any, figs: dict[str, ArtifactRef]) -> str:
    panel = deps.load("metrics_panel") if deps.has("metrics_panel") else {}
    p = [_header(state, deps, "Uniform Measurement Panel",
                 "Every candidate re-measured by one code path, on one sub-sample, under one seed")]
    if not panel:
        return "\n".join(p + ["_No panel produced._"])
    table = panel["table"]
    p += ["## 1. The contract", "",
          "Two numbers may be compared only if the same code produced them, on the same sub-sample, "
          "under the same seed. Quoting a silhouette from one phase beside a silhouette from another "
          "compares two different measurements that happen to share a name. Every row below was "
          "re-measured here for that reason.", "",
          f"- **panel id**: `{table['panel_id']}`",
          f"- **configuration**: `{json.dumps(table['panel_config'], ensure_ascii=False)}`", "",
          "## 2. Comparison", "", _panel_table_md(table), "",
          _fig(figs, "fig_panel", "Panel comparison; bar colour encodes metric authority."),
          "## 3. Metric authority", "",
          "| metric | authority | meaning |", "|---|---|---|"]
    meaning = {
        "decisive": "may select a representation, an algorithm, or a K",
        "advisory": "reported and plotted; explicitly barred from selecting anything",
        "diagnostic": "describes this run; does not compare options",
    }
    for m in table["metrics"]:
        p.append(f"| `{m['name']}` | **{m['authority']}** | {meaning.get(m['authority'], '')} |")
    p += ["", "The distinction is enforced in code, not by convention: `decisive_ranking()` raises "
          "if handed an advisory metric.", "",
          "## 4. Footnotes", ""]
    p += [f"{i + 1}. {f}" for i, f in enumerate(table["footnotes"])]
    p += ["", "## 5. Gates", "", _gate_table(state), ""]
    return "\n".join(p)


def leaf_catalogue(state: PipelineState, deps: Any) -> str:
    naming = deps.load("tree_naming") if deps.has("tree_naming") else {}
    labels = deps.leaf_labels_final() if deps.has("leaf_labels") else None
    fam = deps.leaf_family_final() if deps.has("leaf_family") else None
    p = [_header(state, deps, "Leaf Catalogue", "Every leaf, its definition, and what satisfies the user")]
    p += ["Each entry's `user_need` is a single sentence stating what the user must receive to be "
          "satisfied. It is simultaneously the annotation guideline, the acceptance criterion, and "
          "the downstream product requirement — which is why a name alone is not enough. A name is "
          "ambiguous; a definition sentence is checkable.", ""]
    namings = naming.get("namings", [])
    if not namings:
        return "\n".join(p + ["_No namings available._"])
    sizes = np.bincount(labels) if labels is not None else None
    total = len(labels) if labels is not None else 1
    by_family: dict[int, list[dict]] = {}
    for n in namings:
        f = int(fam[n["leaf_id"]]) if fam is not None and n["leaf_id"] < len(fam) else 0
        by_family.setdefault(f, []).append(n)
    for f in sorted(by_family):
        p.append(f"## Family {f}")
        p.append("")
        for n in sorted(by_family[f], key=lambda x: -(sizes[x["leaf_id"]] if sizes is not None else 0)):
            lid = n["leaf_id"]
            sz = int(sizes[lid]) if sizes is not None else 0
            p += [f"### Leaf {lid} — {n.get('name_zh', '')} (`{n.get('code', '')}`)",
                  f"- size: {sz} rows ({sz / total * 100:.2f}%)",
                  f"- **user_need**: {n.get('user_need', '')}",
                  f"- blind coherence: {n.get('coherence')}/5"
                  + (f" — {n.get('mix_notes')}" if n.get("mix_notes") else "")]
            if n.get("risk_flag"):
                p.append(f"- ⚠️ **risk flagged by the blind namer**: {n.get('risk_reason', '')}")
            p.append("")
    return "\n".join(p)
