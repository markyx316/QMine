"""Evidence assembly for the agent-written final report — no prose lives here.

This module answers one question: *what is the narrator allowed to know, and
what is it not allowed to leave out?* It builds no sentences. Everything it
returns is either a number lifted from an artifact, the name of a figure that
exists on disk, or a requirement.

Three properties, each chosen against a specific failure mode.

**Bundles are scoped AND sufficient.** The data-to-text literature is blunt about
this: when a writer needs a number its source does not contain, it invents one.
Wiseman et al.'s RotoWire corpus grounds only ~60% of its summary content in the
box score, and that deficiency is what teaches a conditioned model to emit
unconditioned facts. So a bundle is not merely *small* — it is complete for the
topic it names. Withholding a number the section must state is not caution; it is
how fabrication gets induced. The existing `interpret()` rule "keep sheets small"
is a rule about *scope*, not about starving the author.

**Coverage is a requirement, not a hope.** `check_numbers` is a PRECISION check:
every number in the prose must be in the sheet. It is silent about omission, so a
triumphant narrative that mentions no warned gate, no open finding and no
limitation passes it perfectly. That is exactly the "selective reporting of
favourable, easy-to-express facts" that coverage-aware evaluation of grounded
generation exists to catch. `must_cover()` therefore derives — mechanically, from
the run's own ledgers — the things this run is not permitted to ship without
mentioning, each with an ANCHOR string that cannot be written without addressing
it. The assembled document is checked against them.

**The catalogue is closed.** The narrator picks which bundles a section needs from
a fixed menu; it cannot name a bundle that does not exist, so it cannot request —
or imagine — evidence the run never produced.

What this module deliberately does NOT do: decide section order, write headings,
or phrase anything. Those are the narrator's, and that is the whole point of it.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------- primitives


def _read(gen: Path, name: str) -> Any:
    """Load an artifact, or return None. A missing artifact is normal."""
    try:
        return json.loads((gen / name).read_text())
    except Exception:                                        # noqa: BLE001
        return None


def _dig(obj: Any, *path: str, default: Any = None) -> Any:
    """Walk a nested mapping without raising on a missing level."""
    cur = obj
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def _pick(obj: Any, keys: tuple[str, ...]) -> dict[str, Any]:
    """Copy just the named keys that are present and are worth stating."""
    if not isinstance(obj, dict):
        return {}
    return {k: obj[k] for k in keys if k in obj and obj[k] is not None}


@dataclass
class Bundle:
    """One topic's evidence: what it covers, its facts, its figures, its sources."""

    id: str
    title: str
    remit: str
    facts: dict[str, Any] = field(default_factory=dict)
    figures: list[tuple[str, str]] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.facts and not self.figures


@dataclass
class MustCover:
    """Something this run may not ship without saying.

    `anchors` are strings the assembled document must contain. They are chosen so
    that a document addressing the point cannot avoid them and one that skips the
    point cannot accidentally contain them — a specific number, or a gate's own
    identifier. A vague requirement ("mention the limitations") is unenforceable
    and would degrade into a checkbox the narrator learns to tick.
    """

    id: str
    what: str
    anchors: list[str]
    severity: str = "must"
    #: A sentence the pipeline SUPPLIES and the section must reproduce verbatim.
    #:
    #: `check_numbers` guarantees numbers and nothing else, so an attribution — a
    #: NOUN — is invisible to it. live42's §4 wrote 「交付的 K=18 是参照
    #: phrasing_groups 的粒度锚点」 when `legacy_l2` located K and
    #: `phrasing_groups` located 10, two lines after listing all three correctly.
    #: Every number in that sentence was sourced; the subject was wrong.
    #:
    #: A general "is this noun right?" check is not achievable, and pattern-
    #: matching Chinese prose for wrong attributions would be fragile in a way
    #: that fails silently. So for the few claims where attribution is
    #: load-bearing, the pipeline writes the sentence and the section must contain
    #: it exactly. The model still writes everything around it — this owns one
    #: sentence, not the argument.
    verbatim: str = ""


def citable_numbers(facts: dict[str, Any]) -> dict[str, float]:
    """Every number the agent can SEE in `sheet(facts)`, keyed for `check_numbers`.

    The contract a writer is given is "use only numbers from this sheet", so the
    checker's pool must be what the sheet SHOWS. `verify._flatten` pools only
    values, and `sheet` renders dotted paths — so a dict keyed by id puts numbers
    in front of the author that the checker will not accept:

        execution.splits.32.new_leaf = 49     <- the agent reads "32"

    live42's `governance_and_risk` section was rejected three times for citing
    `32, 40, 42, 43, 44, 45` — the leaf ids it had just been shown — and shipped
    as a hole. The numbers were in the sheet; only the pool disagreed.

    This widens the pool to exactly the sheet's own content and no further: a
    number absent from the rendered text is still refused.
    """
    return numbers_in(sheet(facts))


def numbers_in(text: str) -> dict[str, float]:
    """Every number appearing in a block of text, keyed for `check_numbers`.

    The pool a writer is checked against must be what the writer was SHOWN.
    Anything shown through another channel is a trap: visible, quotable, and
    guaranteed to be refused. The must-cover items are the sharp case — they
    arrive under "必须原样包含这句话", so a number inside one is an ORDER to
    write a number the checker would then reject, which no retry can satisfy.
    """
    pool: dict[str, float] = {}
    for i, tok in enumerate(re.findall(r"-?\d+(?:\.\d+)?", text)):
        try:
            pool[f"_shown_{i}"] = float(tok)
        except ValueError:                                       # noqa: PERF203
            continue
    return pool


def sheet(facts: dict[str, Any]) -> str:
    """Render a bundle for a prompt — numbers AND names.

    `verify.fact_sheet` renders only what `_flatten` keeps, which is numbers; a
    narrator handed that sheet can state that stability was 0.9768 but cannot name
    the encoder that produced it. Strings are safe to add because the numeric
    guarantee is enforced separately by `check_numbers`, which pools numbers only.
    """
    lines: list[str] = []

    def walk(v: Any, prefix: str) -> None:
        if isinstance(v, dict):
            for k, sub in v.items():
                walk(sub, f"{prefix}.{k}" if prefix else str(k))
        elif isinstance(v, (list, tuple)):
            if all(isinstance(x, (str, int, float, bool)) for x in v) and len(v) <= 30:
                lines.append(f"{prefix} = {list(v)}")
            else:
                for i, sub in enumerate(v[:12]):
                    walk(sub, f"{prefix}[{i}]")
                if len(v) > 12:
                    lines.append(f"{prefix}.length = {len(v)}")
        elif isinstance(v, bool):
            lines.append(f"{prefix} = {'true' if v else 'false'}")
        elif isinstance(v, float):
            lines.append(f"{prefix} = {v:g}")
        elif v is not None:
            lines.append(f"{prefix} = {v}")

    walk(facts, "")
    return "\n".join(sorted(lines))


# ---------------------------------------------------------------- catalogue


def build_catalogue(state: Any, deps: Any) -> dict[str, Bundle]:
    """Every bundle of evidence the narrator may draw on, keyed by id.

    Bundles that came out empty — a phase that did not run, an artifact a shorter
    configuration never wrote — are dropped, so the menu the narrator sees
    describes THIS run rather than the pipeline in general.
    """
    gen = Path(deps.store.gen_dir)
    A = {name: _read(gen, f"{name}.json") for name in (
        "data_audit", "language_profile", "template_groups", "taxonomy",
        "taxonomy_v2",
        "gold_agreement", "adversarial_validation", "representation",
        "granularity", "hierarchy_meta", "governance", "metrics_panel",
        "deployment", "run_summary", "risk_screen", "naming_cards",
        "subintents", "delivery_audit", "maintenance",
    )}
    figs = {p.name for p in gen.glob("*.png")}

    def fig(name: str, caption: str) -> list[tuple[str, str]]:
        return [(name, caption)] if name in figs else []

    B: list[Bundle] = []

    # ---- the corpus itself
    da, lp, tg = A["data_audit"], A["language_profile"], A["template_groups"]
    B.append(Bundle(
        "corpus", "语料本身", "这批查询是什么: 规模、长度、语种构成、有多少是模板化措辞",
        facts={
            "n_rows": _dig(da, "n_rows"), "n_unique": _dig(da, "n_unique"),
            "duplicate_rate": _dig(da, "duplicate_rate"),
            "length": _dig(da, "length"), "script_mix": _dig(da, "script_mix"),
            "language": _pick(lp, ("dominant", "dominant_share", "minority_share",
                                   "nonlinguistic_share", "posture")),
            "template_groups": _pick(_dig(tg, "coverage", default={}) or {},
                                     ("n_groups", "union_coverage")),
            "n_trusted_template_groups": len(_dig(tg, "trusted_groups", default=[]) or []),
        },
        figures=fig("fig_template_spread.png", "模板化措辞在语料中的分布"),
        sources=["data_audit", "language_profile", "template_groups"]))

    # ---- top-down: the taxonomy, and how much of it is real
    tax, sub = A["taxonomy"], A["subintents"]
    # The taxonomy sits under `taxonomy.taxonomy`; reading the outer level found
    # two facts and no class names at all. A section asked to explain the scheme
    # without being able to name one class is a section that will invent one —
    # this is the "scoped AND sufficient" rule failing on the sufficient side.
    inner = _dig(tax, "taxonomy", default={}) or {}
    nodes = _dig(inner, "nodes", default=[]) or []
    B.append(Bundle(
        "topdown_taxonomy", "自上而下: 意图体系怎么长出来的",
        "多路研究 → 架构师起草 → 评审 → 修订。体系有多少类、每类是什么、依据什么轴切分",
        facts={
            "n_l1": len(nodes), "version": _dig(inner, "version"),
            "axes": _dig(inner, "axes"), "notes": _dig(inner, "notes"),
            "labeling_guide": _dig(inner, "labeling_guide"),
            "n_rules": len(_dig(inner, "rules", default=[]) or []),
            "classes": [{k: n.get(k) for k in ("code", "name_zh", "name",
                                               "user_need", "definition")
                         if n.get(k)} for n in nodes if isinstance(n, dict)],
            "n_dropped_candidates": len(_dig(tax, "dropped_candidates",
                                             default=[]) or []),
            "n_redraws": len(_dig(tax, "redraw_history", default=[]) or []),
            "critique": _pick(_dig(tax, "critique", default={}) or {},
                              ("verdict", "n_findings")),
            "n_subintents": len(_dig(sub, "subintents", default=[]) or []),
        },
        sources=["taxonomy", "subintents"]))

    ga = A["gold_agreement"]
    B.append(Bundle(
        "topdown_gold", "自上而下: 标注一致性与指南修复",
        "两个独立标注员 + 裁判。kappa 是多少、天花板在哪、修指南有没有用",
        facts={
            "agreement": _pick(_dig(ga, "agreement", default={}) or {},
                               ("n", "n_submitted", "raw_agreement", "kappa",
                                "n_disagreements", "n_unscored_unlabelled")),
            "n_adjudicated": _dig(ga, "n_adjudicated"),
            "n_new_rules": len(_dig(ga, "new_rules", default=[]) or []),
            "guide_repair": _pick(_dig(ga, "guide_repair", default={}) or {},
                                  ("rounds_run", "n_rules_added", "kappa_before",
                                   "kappa_after", "n_before", "n_after",
                                   "comparable", "paired")),
            # A SHEET THE NARRATOR CAN ONLY PARTLY SEE INVITES ARITHMETIC.
            #
            # `annotator_balance` is measured and lives in `taxonomy_v2.json`,
            # which this catalogue never read. live42's narrator was shown
            # `n_contested = 274` and `annotator_a_won = 92` through other
            # bundles and DERIVED the rest -- 178 = 270 - 92, 0.3407 = 92/270 --
            # every one of which the checker then refused, correctly, as a
            # number not in the sheet. Two sections burned all three attempts on
            # it and shipped as holes.
            #
            # It also belongs here on the merits: `lopsided` is a headline
            # quality warning about the gold labels this bundle exists to
            # describe. Starving a sheet does not produce caution, it produces
            # invention -- the same rule as `test_the_taxonomy_bundle_can_
            # actually_name_a_class`.
            "annotator_balance": _pick(
                _dig(A["taxonomy_v2"], "annotator_balance", default={}) or {},
                ("n_contested", "n_decided", "annotator_a_won", "annotator_b_won",
                 "referee_chose_neither", "annotator_a_share", "z_vs_even",
                 "lopsided", "undecidable")),
        },
        sources=["gold_agreement", "taxonomy_v2"]))

    av = A["adversarial_validation"]
    B.append(Bundle(
        "topdown_adversarial", "自上而下: 对抗验证",
        "让一个 agent 专门去证伪每一条标签, 剩下站得住的比例就是标签准确率的估计",
        facts=_pick(av or {}, ("n_attacked", "n_verdicts", "coverage", "n_wrong",
                               "n_defensible", "estimated_accuracy", "method")),
        sources=["adversarial_validation"]))

    # ---- representation
    rep = A["representation"]
    B.append(Bundle(
        "representation", "表征: 用什么向量来度量「像不像」",
        "encoder 选型 (在本语料自己的任务上比, 不看公开榜单) 与 alpha 的取值含义",
        facts={
            "bakeoff": _pick(_dig(rep, "bakeoff", default={}) or {},
                             ("chosen_encoder", "chosen_by", "subsample_size",
                              "silhouette_would_have_chosen", "silhouette_disagrees")),
            "bakeoff_rows": _dig(rep, "bakeoff", "rows"),
            "alpha": _pick(_dig(rep, "alpha_sweep", default={}) or {},
                           ("chosen_alpha", "chosen_by", "tie_band_value",
                            "tie_band_relative_pct", "silhouette_would_have_chosen",
                            "silhouette_disagrees")),
            "alpha_rows": _dig(rep, "alpha_sweep", "rows"),
            "alpha_algebra": _pick(_dig(rep, "alpha_algebra", default={}) or {},
                                   ("formula", "surface_vote_share")),
            "sparse": _pick(_dig(rep, "sparse", default={}) or {},
                            ("vocab_size", "explained_variance", "n_components")),
        },
        figures=fig("fig2_alpha.png", "alpha 扫描: 碎裂度与稳定性的取舍")
        + fig("fig4_spaces.png", "三种表征空间的对照"),
        sources=["representation"]))

    # ---- granularity
    gr = A["granularity"]
    B.append(Bundle(
        "granularity", "粒度: 家族层到底切几个",
        "K 由与模板群的对齐度定位, 稳定性只负责否决; 并列的 K 全部报出来",
        facts={
            "triangulation": _pick(_dig(gr, "triangulation", default={}) or {},
                                   ("chosen_family_k", "locator", "chosen_by",
                                    "tie_set", "stability_floor", "converged",
                                    "n_rejected_as_unstable", "estimates",
                                    "measured_estimators_agree", "prior_agrees",
                                    "silhouette_disagrees")),
            "deep_aligned": _pick(_dig(gr, "deep_aligned", default={}) or {},
                                  ("k_overcluster", "k_estimate", "survival_threshold")),
            "n_k_tried": len(_dig(gr, "k_sweep", default=[]) or []),
            "grid_proposal": _pick(_dig(gr, "grid_proposal", default={}) or {},
                                   ("incumbent", "chosen", "a_proposed_value_won",
                                    "n_extra_comparisons", "verdict")),
        },
        figures=fig("fig1_ksweep.png", "K 扫描")
        + fig("fig1b_ksweep_metrics.png", "K 扫描: 各指标随 K 的走向"),
        sources=["granularity"]))

    # ---- the tree
    hm = A["hierarchy_meta"]
    B.append(Bundle(
        "hierarchy", "两层树: 家族 → 叶子, 以及迭代精化",
        "树是怎么建起来的, 精化改了什么, held-out 复现能不能站住",
        facts={
            **_pick(hm or {}, ("n_families", "n_leaves", "min_leaf_size_applied",
                               "converged", "n_families_before_refinement",
                               "n_leaves_reported_by_refinement")),
            "leaves_per_family": _dig(hm, "leaves_per_family"),
            "n_refinement_steps": len(_dig(hm, "refinement_history", default=[]) or []),
            "heldout": _pick(_dig(hm, "heldout_reproduction", default={}) or {},
                             ("agreement", "n_test", "train_fraction",
                              "in_sample_ceiling", "effective_threshold",
                              "statistical_verdict")),
            "local_k": _pick(_dig(hm, "local_k", default={}) or {},
                             ("n_families_not_split", "n_silhouette_overruled")),
        },
        figures=fig("fig_refinement.png", "迭代精化: 每一步改了什么"),
        sources=["hierarchy_meta"]))

    nc = A["naming_cards"]
    B.append(Bundle(
        "naming", "盲评命名与树审计",
        "命名者只看查询样本、不看任何既有标签; 审计员再检查这棵树本身",
        facts={"n_named": len(nc if isinstance(nc, list) else
                              _dig(nc, "cards", default=[]) or [])},
        sources=["naming_cards"]))

    # ---- governance
    gov = A["governance"]
    B.append(Bundle(
        "governance", "治理: 处方被执行了, 不只是被记录",
        "审计与风险处方的最终去向 — 合并、拆分、隔离各多少, 执行前后指标怎么变",
        facts={
            "execution": _pick(_dig(gov, "execution", default={}) or {},
                               ("splits", "merges", "isolations", "n_executed",
                                "n_declined", "metric_deltas")),
            "settled": _pick(_dig(gov, "settled", default={}) or {},
                             ("n_total", "n_executed", "n_declined", "verdict")),
            "n_ledger_entries": len(_dig(gov, "ledger", default=[]) or []),
        },
        sources=["governance"]))

    rs = A["risk_screen"]
    B.append(Bundle(
        "risk", "风险内容筛查",
        "哨兵在没有被告知要找什么的情况下, 独立标出了哪些内容",
        facts=_pick(rs or {}, ("n_flagged", "n_scanned", "flag_rate", "categories",
                               "independently_found", "action")),
        sources=["risk_screen"]))

    # ---- the panel: the ONE place both routes are on a common axis
    mp = A["metrics_panel"]
    B.append(Bundle(
        "panel", "统一度量面板: 两条路线放在同一把尺子上",
        "这是全流程唯一把自上而下与自下而上放在同一套指标下对照的地方",
        facts={"comparison": _panel_grid(mp),
               "panel_config": _dig(mp, "table", "panel_config"),
               "footnotes": _dig(mp, "table", "footnotes")},
        figures=fig("fig6_panel.png", "统一度量面板")
        + fig("fig5_intent_split.png", "意图轴的可见性: 哪些类目结构上看得见"),
        sources=["metrics_panel"]))

    # ---- deployment and the crosswalk
    dep = A["deployment"]
    B.append(Bundle(
        "deployment", "部署: 全量打标与路由",
        "两套标签并排写进同一张全量表; 新查询怎么被路由到叶子",
        facts={
            "routing": _pick(_dig(dep, "routing", default={}) or {},
                             ("n_direct", "n_fallback", "ambiguous_rate",
                              "threshold", "policy")),
            "model_bytes": _dig(dep, "model_bytes"),
            "alpha": _dig(dep, "alpha"),
            "inference": _dig(dep, "inference"),
            "n_live_demo": len(_dig(dep, "live_demo", default=[]) or []),
        },
        sources=["deployment"]))

    # ---- gates and findings: the run's own warnings
    B.append(Bundle(
        "gates", "质量门总账", "流水线唯一会说「这还不够好」的地方, 逐条列出",
        facts=_gate_facts(state),
        figures=fig("fig_gates.png", "质量门总览"),
        sources=["run_summary"]))

    B.append(Bundle(
        "findings", "未关闭的发现与交付前审核",
        "本次运行提出过、但到交付时仍未消解的问题, 以及审核 agent 做了什么",
        facts=_finding_facts(state, A["delivery_audit"]),
        sources=["delivery_audit"]))

    # ---- the decision chain, in the order it happened
    B.append(Bundle(
        "decisions", "决策链", "每一个参数是谁、依据哪个指标定下来的, 淘汰了什么",
        facts=_decision_facts(state),
        figures=fig("fig_decision_chain.png", "决策链: 每个环节考虑与淘汰了多少候选"),
        sources=["run_summary"]))

    # ---- worked examples
    B.append(Bundle(
        "samples", "样本实例", "被完整打上两套标签的真实查询, 逐条可讲",
        facts={"exemplars": _exemplars(state, deps, A)},
        sources=["deployment", "labels_full.csv"]))

    # ---- how the run itself was executed
    rsum = A["run_summary"]
    B.append(Bundle(
        "run_meta", "这次运行本身",
        "跑了多久、用了哪些模型、花了多少、哪些阶段完成了",
        facts={
            "run_id": _dig(rsum, "run_id"), "generation": _dig(rsum, "generation"),
            "elapsed_s": _dig(rsum, "elapsed_s"),
            "n_completed_phases": len(_dig(rsum, "completed_phases", default=[]) or []),
            "halted": _dig(rsum, "halted"),
            "llm": _pick(_dig(rsum, "llm_usage", default={}) or {},
                         ("calls", "input_tokens", "output_tokens", "cache_hits",
                          "errors", "provider", "estimated_cost_usd",
                          "models_used", "deep_model", "fast_model")),
        },
        sources=["run_summary"]))

    return {b.id: b for b in B if not b.is_empty()}


def _gate_facts(state: Any) -> dict[str, Any]:
    """Every gate, flattened to status counts plus the non-passing ones in full."""
    gates = state.get("gates", {}) or {}
    if not gates:
        return {}
    by_status: dict[str, int] = {}
    notable: dict[str, Any] = {}
    for name, g in gates.items():
        st = getattr(g, "status", "unknown")
        by_status[st] = by_status.get(st, 0) + 1
        if st != "passed" or getattr(g, "blocking", False):
            notable[name] = {
                "phase": getattr(g, "phase", ""), "status": st,
                "blocking": getattr(g, "blocking", False),
                "observed": getattr(g, "observed", {}),
                "threshold": getattr(g, "threshold", {}),
                "remediation": getattr(g, "remediation", ""),
            }
    return {"n_gates": len(gates), "by_status": by_status, "notable": notable}


def _finding_facts(state: Any, audit: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    findings = state.get("findings") or []
    if findings:
        out["n_findings"] = len(findings)
        out["open"] = [
            {"id": f.get("id"), "what": f.get("what"), "status": f.get("status")}
            for f in findings if isinstance(f, dict) and f.get("status") == "open"][:12]
        out["n_open"] = sum(1 for f in findings
                            if isinstance(f, dict) and f.get("status") == "open")
    if isinstance(audit, dict):
        out["audit"] = _pick(audit, ("ran", "n_applied", "n_refused",
                                     "n_dismissed", "skipped"))
    obs = state.get("observations") or []
    if obs:
        out["n_observations"] = len(obs)
    return out


def _decision_facts(state: Any) -> dict[str, Any]:
    decisions = state.get("decisions") or []
    if not decisions:
        return {}
    rows = []
    for d in decisions:
        rows.append({
            "phase": getattr(d, "phase", ""),
            "question": getattr(d, "question", ""),
            "choice": getattr(d, "choice", ""),
            "decided_by": getattr(d, "decided_by", ""),
            "decisive_metrics": getattr(d, "decisive_metrics", []) or [],
            "n_rejected": len(getattr(d, "rejected", []) or []),
        })
    return {"n_decisions": len(rows), "chain": rows}


def _panel_grid(mp: Any) -> dict[str, dict[str, float]]:
    """The panel as `subject → metric → value`, and nothing else.

    Read whole, `metrics_panel.json` contributes 783 facts to one section — every
    metric wrapped in its panel id, seed, authority, detail and note. That is not
    a scoped sheet, it is the artifact with a different name, and it buries the
    eleven numbers the comparison actually turns on.
    """
    out: dict[str, dict[str, float]] = {}
    for subject, rec in (_dig(mp, "sets", default={}) or {}).items():
        vals = {}
        for name, m in (_dig(rec, "metrics", default={}) or {}).items():
            v = m.get("value") if isinstance(m, dict) else m
            if isinstance(v, (int, float)):
                vals[name] = v
        if vals:
            out[str(subject)] = vals
    return out


def _exemplars(state: Any, deps: Any, A: dict[str, Any]) -> list[dict[str, Any]]:
    """Fully-labelled real queries, chosen by a DETERMINISTIC rule.

    The narrator comments on these; it does not pick them. Letting an author
    choose its own illustrations is how a report ends up showing only the cases
    that flatter it — the same selective-reporting failure `must_cover` blocks one
    level up. So the rule is fixed and it deliberately includes the hard cases:
    one median-index row per bottom-up family, plus the rows the router itself
    marked ambiguous. A sample set of clean hits would misrepresent the system.

    `deployment.deterministic_exemplars` is NOT what is wanted here — those are
    phrasing-pattern exemplars carrying no labels at all. The delivered table is
    the only place a query appears with both routes' labels beside it.
    """
    import pandas as pd

    path = Path(deps.store.gen_dir) / "labels_full.csv"
    if not path.exists():
        return []
    try:
        df = pd.read_csv(path)
    except Exception:                                        # noqa: BLE001
        return []
    cols = [c for c in ("query", "bu_family_final", "bu_leaf", "bu_leaf_name",
                        "bu_user_need", "bu_margin", "bu_ambiguous",
                        "td_l1", "td_l1_name", "td_user_need", "td_confidence",
                        "td_ambiguous", "td_decided_by", "ref_legacy_l1")
            if c in df.columns]
    if "query" not in cols:
        return []

    picked: list[int] = []
    if "bu_family_final" in df.columns:
        for fam in sorted(df["bu_family_final"].dropna().unique()):
            block = df[df["bu_family_final"] == fam].sort_values("query")
            if len(block):
                picked.append(int(block.index[len(block) // 2]))
    # The cases the system itself was least sure about, by the same fixed rule.
    if "bu_margin" in df.columns:
        amb = df.sort_values(["bu_margin", "query"]).head(3)
        picked.extend(int(i) for i in amb.index)

    out: list[dict[str, Any]] = []
    seen: set[int] = set()
    for i in picked:
        if i in seen:
            continue
        seen.add(i)
        row = df.loc[i, cols]
        rec = {c: (None if pd.isna(row[c]) else
                   (float(row[c]) if isinstance(row[c], float) else row[c]))
               for c in cols}
        out.append({k: v for k, v in rec.items() if v is not None})
    return out[:12]


# ---------------------------------------------------------------- coverage


def must_cover(state: Any, deps: Any, catalogue: dict[str, Bundle]) -> list[MustCover]:
    """What this run may not ship without saying, derived from its own ledgers.

    Every item here exists because a *specific* thing went less than perfectly and
    a reader who is not told about it would draw a wrong conclusion. None of it is
    hand-written per corpus: a run with no warned gates and no open findings gets
    a shorter list, which is correct.
    """
    out: list[MustCover] = []

    # 1. Both routes must be explained. This is the headline claim of the method,
    #    and a report that describes one of them has described half the work.
    out.append(MustCover(
        "both_routes",
        "自上而下与自下而上两条路线都必须讲到, 并说明各自负责哪根轴",
        ["自上而下", "自下而上"]))

    # 2. Every gate that did not simply pass.
    for name, g in (state.get("gates", {}) or {}).items():
        st = getattr(g, "status", "")
        if st in ("warned", "failed", "rejected"):
            out.append(MustCover(
                f"gate:{name}",
                f"质量门 `{name}` 的结果是 {st}, 必须在报告中出现并说明影响",
                [name]))

    # 3. A gate that passed while sitting below its own bar is the most
    #    dangerous kind: it reads as a pass in every table.
    for name, g in (state.get("gates", {}) or {}).items():
        msg = str(getattr(g, "message", "") or "")
        if getattr(g, "status", "") == "passed" and "SLACK" in msg.upper():
            out.append(MustCover(
                f"slack:{name}", f"`{name}` 是带保留通过的, 不能当作干净通过来写",
                [name]))

    # 4. Open findings.
    for f in (state.get("findings") or []):
        if isinstance(f, dict) and f.get("status") == "open" and f.get("id"):
            out.append(MustCover(
                f"finding:{f['id']}",
                f"未关闭的发现 {f['id']}: {str(f.get('what', ''))[:80]}",
                [str(f["id"])]))

    # 5. A tie set means the chosen K is one of several that stand up equally.
    #    Reporting the winner alone converts a tie into a result.
    # THE ATTRIBUTION THE MODEL GOT WRONG, SUPPLIED RATHER THAN CHECKED.
    gran = catalogue.get("granularity", Bundle("", "", "")).facts
    deciding = _dig(gran, "triangulation", "deciding_reference")
    located = _dig(gran, "triangulation", "reference_sensitivity",
                   "located_k_values", default={}) or {}
    chosen_k = _dig(gran, "triangulation", "chosen_family_k")
    if deciding and chosen_k is not None:
        per_ref = ", ".join(f"{n}→K={v}" for n, v in sorted(located.items()))
        out.append(MustCover(
            "k_deciding_reference",
            f"必须写明定位 K 的参照系是 `{deciding}`, 且不得张冠李戴",
            [str(deciding)],
            verbatim=(f"交付的家族层 K={chosen_k} 由参照系 `{deciding}` 定位; "
                      f"各参照系各自定位到的 K 为 {per_ref}。"
                      if per_ref else
                      f"交付的家族层 K={chosen_k} 由参照系 `{deciding}` 定位。"),
        ))

    tie = _dig(catalogue.get("granularity", Bundle("", "", "")).facts,
               "triangulation", "tie_set", default=[]) or []
    if len(tie) > 1:
        out.append(MustCover(
            "k_tie_set",
            f"K 的并列集合有 {len(tie)} 个取值, 必须报出整个集合而不是只报当选的那个",
            [str(t) for t in tie]))

    # 6. Classes the geometry cannot carry. Without this the reader assumes
    #    clustering would eventually find them, and it will not.
    inv = _dig(catalogue.get("panel", Bundle("", "", "")).facts,
               "metrics", default=None)
    if isinstance(inv, dict) and inv.get("rule_dependent_classes"):
        out.append(MustCover(
            "structurally_invisible",
            "有一部分类目在表征里结构性不可见, 必须说明聚类不会发现它们",
            ["规则"]))

    return out


def digest(catalogue: dict[str, Bundle], musts: list[MustCover]) -> str:
    """A compact map of the whole run, for the pass that plans the story.

    Deliberately not the full evidence: the planner decides SHAPE, and handing it
    every number would both blow the context and invite it to start writing.
    """
    L = ["## 可用证据 (按 id 取用)"]
    for b in catalogue.values():
        n = len(sheet(b.facts).splitlines())
        figs = ", ".join(f for f, _ in b.figures) or "无"
        L.append(f"- `{b.id}` — **{b.title}**: {b.remit} (事实 {n} 条; 图: {figs})")
    L += ["", "## 本次运行不得省略的内容"]
    for m in musts:
        L.append(f"- `{m.id}` — {m.what}")
    return "\n".join(L)
