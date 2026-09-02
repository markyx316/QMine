"""Configuration: the knobs the playbook says you must re-tune per domain.

Part IV of the playbook splits every setting into two piles — *invariant*
(copy it straight across domains) and *must be re-derived* (alpha, K, the
template regexes, the risk lexicon, the tokenizer).  That split is the spine of
this module.  :class:`QMineConfig` holds the invariants with defaults that came
out of the K12 run; :class:`DomainProfile` holds the pile you are obliged to
re-derive, and every field in it is either seeded from a YAML profile or
discovered at runtime and written back.

A config is hashable (``config_hash``) and the hash goes into the run manifest,
so "which settings produced this number" is never a guess.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from .determinism import SEED_METRIC, SEED_REPLAY, SEED_VIZ, hash_params


# ==========================================================================
# Domain profile — the "must be re-derived per domain" pile
# ==========================================================================

class TemplateSeed(BaseModel):
    """A hand-seeded phrasing family, refined by the Phase 1 miner.

    The contract is strict: *everything matching this regex is almost certainly
    the same intent*.  These groups become the judge of the alpha sweep, the
    denominator of template fragmentation, and the source of display exemplars.
    """

    name: str
    pattern: str
    intent_hint: str = ""
    expected_min_share: float = 0.0


class RiskCategory(BaseModel):
    """A category that must never be blended into a normal family (Principle 10)."""

    name: str
    patterns: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    rationale: str = ""
    policy: str = Field(default="isolate", description="isolate | isolate_and_flag | drop")


class DomainProfile(BaseModel):
    """Everything that changes when you move from K12 to finance to sports."""

    # THE CLASS DEFAULTS MUST NOT BE CHINESE WHILE CALLING THEMSELVES GENERIC.
    # Constructing `QMineConfig()` with no `--domain` produced key="generic" with
    # `language=zh`, `tokenizer=jieba`, Chinese-only bake-off candidates, ZERO
    # risk categories and ZERO pragmatic hints — strictly worse than
    # `--domain generic`, and silently so on an English corpus. These defaults now
    # match `configs/domains/generic.yaml`, which the CLI also loads explicitly.
    key: str = "generic"
    display_name: str = "Unknown / mixed-vertical query log"
    language: Literal["zh", "en", "multi"] = "multi"

    #: Tokenisation and n-gram ranges (Part IV section 4.5).
    #: ``auto`` is resolved in Phase 1 from the corpus's actual script mix. Use it
    #: whenever the language is not known in advance — assuming a tokeniser is
    #: one of the cheapest ways to quietly degrade every downstream phase.
    tokenizer: Literal["jieba", "whitespace", "none", "auto"] = "auto"
    char_ngram_range: tuple[int, int] = (1, 3)
    word_ngram_range: tuple[int, int] = (1, 2)

    #: Base encoder candidates for the bake-off (Phase 3a).  Order is a hint,
    #: not a decision — the bake-off decides.
    embedding_candidates: list[str] = Field(
        default_factory=lambda: ["intfloat/multilingual-e5-small", "BAAI/bge-m3"]
    )
    instruction_prefix: str | None = None

    #: Seeds for Phase 1 template mining.  The miner adds discovered families.
    template_seeds: list[TemplateSeed] = Field(default_factory=list)

    #: Compliance-critical categories (Principle 10).
    risk_categories: list[RiskCategory] = Field(default_factory=list)

    #: Prior expectations used only to triangulate, never to override the data.
    expected_l1_range: tuple[int, int] = (15, 25)
    expected_family_range: tuple[int, int] = (10, 30)

    #: Intents the playbook predicts clustering will be structurally blind to
    #: (Principle 1).  Written into the top-down brief so the taxonomy owns them.
    pragmatic_intents_hint: list[str] = Field(default_factory=list)

    #: Domain notes injected into research and naming prompts.
    domain_notes: str = ""

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "DomainProfile":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(data)


# ==========================================================================
# Invariant sections — safe to copy across domains
# ==========================================================================

class DataConfig(BaseModel):
    input_path: str = ""
    text_column: str = "query"
    weight_column: str | None = Field(
        default=None, description="Full-log frequency, for population-weighted metrics."
    )
    reference_label_columns: list[str] = Field(
        default_factory=list, description="Legacy labels, if any. Reference only — never supervision."
    )
    sample_size: int | None = Field(default=None, description="None = use all rows.")
    min_query_len: int = 1
    max_query_len: int = 200
    drop_duplicates: bool = False


class RepresentationConfig(BaseModel):
    svd_dims: int = 256
    tfidf_min_df: int = 2
    tfidf_sublinear: bool = True
    #: Phase 3c.  Re-run per domain; never inherit the K12 answer.
    alpha_grid: list[float] = Field(default_factory=lambda: [0.0, 0.1, 0.2, 0.3, 0.5])
    #: The K at which the encoder bake-off and the alpha sweep are judged. ``None``
    #: derives it from the domain's expected family range, which is where a prior
    #: about granularity belongs; a bare 20 was a K12 artefact standing in for that
    #: prior on every corpus. Measured on the reference run the choice was benign —
    #: the alpha picked at K=20 was still optimal at the final K=25 — but "benign on
    #: the corpus it came from" is exactly the property that does not travel.
    alpha_sweep_k: int | None = None
    bakeoff_subsample: int = 15000


class ClusteringConfig(BaseModel):
    k_sweep: list[int] = Field(
        default_factory=lambda: [8, 10, 12, 15, 18, 20, 25, 30, 40, 50, 65, 80, 100, 120]
    )
    battery_k: list[int] = Field(default_factory=lambda: [15, 20, 30])
    silhouette_sample: int = 8000
    stability_common_sample: int = 8000
    #: Phase 6: leaves smaller than this are not worth naming.
    min_leaf_size: int = 150
    min_leaf_fraction: float = 0.003
    max_leaves_per_family: int = 8
    family_min_size_for_split: int = 300
    refine_rounds: int = 5
    refine_merge_cos: float = 0.92
    refine_move_tolerance: float = 0.005
    heldout_fraction: float = 0.2
    deep_aligned_multiplier: int = 3
    #: Replay-ARI below this rejects a partition as irreproducible rather than
    #: ranking it. It was a bare default inside `ops/cluster.py` with no config
    #: entry and no test, i.e. an imported constant of exactly the kind
    #: `test_gates_do_not_import_thresholds_that_only_fit_one_corpus` exists to
    #: catch — it simply was not reachable to be caught. Read it with its own
    #: instrument's spread: `replay_stability`'s seed-to-seed sd is ~0.14 on these
    #: corpora, so a hard cut at 0.55 against observed minima near 0.63 is inside
    #: one sd, and on live40 it filtered exactly one candidate all run. Raise it
    #: only with a measured spread to justify the new value.
    stability_floor: float = 0.55
    #: Which reference partition locates the family K.
    #:
    #: ``"auto"`` picks the one that REACHES the most of the partition (see
    #: `ops.cluster.locator_reach`), which is the property that decides whether a
    #: reference can speak for the corpus at all. Measured on live40 at k=18: the
    #: six trusted phrasing groups hold a real share of 38.9% of clusters and
    #: locate K=7; `ref_legacy_l1` holds a share of 100% and locates K=18. Scoring
    #: the same reference on the rows the templates match gives 7, on a random
    #: sample of the same size gives 18, and on the rows they miss gives 18 — the
    #: templates select a structurally atypical third and locate K for it.
    #:
    #: ``"phrasing"`` forces the old behaviour; any other string names a column.
    #: There is no label-free repair for a low-reach reference: background-class
    #: and downsampled-background variants both still returned K=7, because the
    #: templates carry no information about the rows they do not match.
    k_locator: str = "auto"


class TaxonomyConfig(BaseModel):
    #: How many INDEPENDENT annotators label the gold set. Two is the method:
    #: kappa, the pilot ceiling and the referee all exist because two labelled
    #: the same row. One is `mode="fast"` — and then kappa is not 1.0, it is
    #: ABSENT, because there is no second reading to agree with.
    annotators: Literal[1, 2] = 2
    #: Which one survives at `annotators=1`. `a` is the primary role and `b` is
    #: DEFINED as the independent second opinion (`llm/requirements.py`), so `a`
    #: is the default — NOT because it is measurably better. It is not reliably
    #: better: on live38 annotator_a took 78.3% of the referee's contested
    #: decisions (z=+14.7); on med04, with a different corpus and different
    #: models, it took 40.3% (z=-3.35). Which one wins is a property of the
    #: pairing, and fast mode never measures it, so this is a recorded choice
    #: rather than a finding.
    primary_annotator: Literal["a", "b"] = "a"
    n_researchers: int = 5
    l1_target_range: tuple[int, int] = (15, 25)
    min_adjudication_rules: int = 20
    #: ``None`` derives it from the corpus (see :func:`gold_size_for`), which is
    #: what the playbook asks for. An explicit int pins it and skips the scaling.
    gold_sample_size: int | None = None
    #: The playbook's stratified gold range, 分层抽 3,000-5,000 条.
    gold_size_range: tuple[int, int] = (3000, 5000)
    #: Floor on the share of a small corpus that becomes gold. The playbook is
    #: explicit that a small corpus needs a HIGHER proportion, not the same
    #: absolute count — 「<1 万条 → 金标比例提高」 — because the tail shapes a
    #: classifier must learn do not shrink proportionally with the corpus.
    gold_min_fraction_small_corpus: float = 0.30
    #: Never annotate more than this share of the corpus, whatever the rules say.
    gold_max_fraction: float = 0.60
    #: The playbook says 50. Measured against its own downstream bar, 50 rows have
    #: no power: at kappa 0.83 the 95% upper bound is 0.924, so a guide that will
    #: fail the 0.90 gate sails through the pilot. n=100 is the break-even and 200
    #: gives margin — 16 LLM calls against the ~240 the gold set costs, so the
    #: cheap-early-check economics are untouched.
    pilot_sample_size: int = 200
    pilot_agreement_threshold: float = 0.85
    kappa_threshold: float = 0.90
    #: Playbook 2b: "达不到先修指南再重标" — if kappa misses, repair the guide and
    #: re-annotate rather than shipping an ambiguous gold set. Each round costs a
    #: second full annotation pass, so the default is one.
    kappa_repair_rounds: int = 1
    #: How many times P2a may redraw the taxonomy and re-pilot before giving up.
    #: Each round costs one redraw call plus one pilot — cheap against a 3,000-row
    #: gold set, and the alternative was halting with a printed diagnosis nobody
    #: acted on. 0 disables the loop and restores the report-and-stop behaviour.
    max_taxonomy_redraws: int = 2
    #: Reuse a previous run's taxonomy instead of designing a new one. The design
    #: work is expensive (5 researchers + architect + rule writer + critic, ~25
    #: minutes) and TWO of the researchers read the live web, so re-running it
    #: never reproduces the same taxonomy. That is what made recovery from a
    #: mid-annotation provider outage cost ninety minutes rather than five: the
    #: routing was pinned, the cache keys still moved, because the class list and
    #: rules are embedded in every annotator prompt. Point this at a run id.
    reuse_taxonomy_from: str | None = None
    #: The repair round scores a FRESH sample. Re-scoring the rows the repair was
    #: derived from measures how well the rules fit those rows, not whether the
    #: guide got clearer.
    repair_on_fresh_sample: bool = True
    active_learning_rounds: int = 1
    active_learning_batch: int = 200
    rule_precision_floor: float = 0.98


class NamingConfig(BaseModel):
    n_naming_agents: int = 5
    #: Name each DELIVERED family after governance, from the leaves it actually
    #: holds. Off, the reports fall back to `混合·主要成分「X」N%` — a composition
    #: diagnostic that was being used as the family's title in headings, table
    #: cells, a Mermaid node and a CSV column.
    name_delivered_families: bool = True
    card_center: int = 15
    card_random: int = 10
    card_edge: int = 5
    card_top_ngrams: int = 12
    coherence_threshold: float = 4.0


class GateConfig(BaseModel):
    """Quality gates.  ``blocking`` gates stop the run; the rest warn."""

    #: Retained for reporting and the report's narrative; NOT gated on. A coverage
    #: share is a property of how templated a corpus is — this K12 window flagged
    #: the e-commerce corpus at 0.534 for no defect at all.
    template_coverage_range: tuple[float, float] = (0.20, 0.40)
    #: What the gate actually requires: enough grouped rows for the fragmentation
    #: and intent-alignment metrics to be computed on signal rather than noise.
    min_template_rows: int = 400
    min_template_row_fraction: float = 0.05
    pilot_agreement: float = 0.85
    #: The playbook's aspiration (line 205), reported for reference. It is NOT a
    #: pass condition: it came with "K12 达 0.966", i.e. it was set as a floor
    #: beneath what *that* project's annotators achieved. Ours self-agree at
    #: 0.883, so 0.90 sits above this annotator's ceiling and two annotators can
    #: never reach it — a bar that cannot be cleared is a wall, not a gate.
    kappa: float = 0.90
    #: What the gate actually enforces, on the quantity it actually describes.
    #: Two annotators cannot agree with each other more reliably than one agrees
    #: with itself, so the fitness question is about the ANNOTATOR: can it apply
    #: this taxonomy reproducibly at all? 0.80 is the conventional reliability
    #: threshold in content analysis (Landis & Koch "almost perfect" begins at
    #: 0.81; Krippendorff wants alpha >= 0.800 before drawing conclusions) — a
    #: disciplinary constant about measurement, not one imported from another
    #: project's result. Below it the labels cannot support conclusions and no
    #: guide work will change that.
    annotator_fitness_kappa: float = 0.80
    #: Fraction of the intended gold sample that both annotators must actually
    #: label before kappa is treated as a measurement at all.
    min_annotation_coverage: float = 0.90
    #: Absolute ceiling on the demand. The effective bar is the LOWER of this and
    #: `heldout_share_of_ceiling` x the partition's own in-sample reproducibility,
    #: because a partition cannot generalise better than it self-replicates and the
    #: achievable value is corpus- and K-dependent (0.973-0.991 across three corpora).
    heldout_reproduction: float = 0.98
    #: How much of its own in-sample ceiling a partition must retain out of sample.
    heldout_share_of_ceiling: float = 0.97
    #: Reported, not gated. Rater calibration varies by model and domain — the same
    #: tree scored 3.93 and 4.27 on two runs of one corpus.
    coherence: float = 4.0
    #: A leaf at or below this on the raters' 1-5 scale is "weak".
    coherence_weak_below: float = 3.0
    #: How much of the tree may be weak before the tree is not deliverable. A mean
    #: cannot express this: 20 good leaves and 5 incoherent ones pass on average.
    coherence_max_weak_share: float = 0.15
    require_risk_independently_found: bool = True
    require_governance_executed: bool = True
    blocking: list[str] = Field(
        default_factory=lambda: [
            "p2a_pilot_agreement",
            "p2b_kappa",
            "p6_heldout_reproduction",
            "p7_all_leaves_named",
            "p8_governance_executed",
            # p7_all_leaves_named runs BEFORE governance changes the partition, so
            # it cannot see the leaves p8 creates. live38 passed it with 29 leaves
            # and delivered 36, 4,931 rows nameless. This one reads the table the
            # reader receives.
            "p10_delivered_leaves_named",
        ]
    )
    #: Human sign-off points (Principle 2 — the reviewer holds a veto).
    human_review_points: list[str] = Field(
        default_factory=lambda: ["p2a_taxonomy", "p7_tree", "p9_panel"]
    )


class LLMConfig(BaseModel):
    """Two-tier model routing, borrowed from TradingAgents' deep/quick split.

    Three details here are not stylistic and will produce hard API errors if
    changed carelessly:

    * **Model IDs carry no date suffix.** ``claude-opus-5``, not
      ``claude-opus-5-20260101``.
    * **``temperature`` is rejected outright by the current frontier models.**
      Opus 5 and Fable 5 removed the parameter; sending any value is a 400. It
      is kept in this config only for providers that still accept it, and the
      registry strips it for those that do not.
    * **The fast tier is Sonnet, not Haiku.** Haiku 4.5's minimum cacheable
      prefix is 4096 tokens, and the taxonomy prefix we resend on every
      annotation batch sits below that — so on Haiku the prompt cache silently
      never engages and bulk labelling costs full price on every call.
    """

    #: ``router`` resolves a model per role from a live catalogue and whatever API
    #: keys are present. ``auto`` keeps the simple two-tier behaviour. A named
    #: provider pins everything to that one.
    provider: Literal["anthropic", "openai", "mock", "auto", "router"] = "auto"
    deep_model: str = "claude-opus-5"
    fast_model: str = "claude-sonnet-5"
    temperature: float | None = Field(
        default=None, description="None = omit. Required for Opus 5 / Fable 5, which 400 on any value."
    )
    #: Generous because on Opus 5 thinking is on by default and max_tokens caps
    #: thinking plus response together — a limit sized for a non-thinking model
    #: truncates mid-answer.
    max_tokens: int = 16000
    max_concurrency: int = 8
    #: Run the two gold-set annotators at the same time. They are independent by
    #: design, so there is no ordering dependency — only that the code used to
    #: call one and then the other. Measured on live38 this returns ~16% of the
    #: run's wall clock (p2b is 75% of the pipeline; the annotators are 94% of
    #: its calls, and they are unbalanced enough that the saving is the whole of
    #: the faster one). NOTE it doubles peak in-flight requests to
    #: `2 x max_concurrency` while p2b runs, because splitting one budget between
    #: them is worse than sequential. Turn off if a provider rate-limits.
    annotators_concurrent: bool = True
    #: SDK-level retries, and the answer is NONE — this layer is redundant with
    #: ours and multiplies every timeout.
    #:
    #: The warning that used to stand here ("three layers at 3 attempts each is
    #: 27 calls") was describing something that was actually happening. `_call`
    #: retries `max_repair + 1 = 3` times, and each of those was 3 HTTP requests
    #: at this setting: **9 requests per logical call**. On live44 the maintainer
    #: timed out at its 292s deadline nine times in a row — 9 x 292 = 2,628s
    #: against the 2,638s the log recorded — then `p12_maintain` reported
    #: ✔ completed having produced nothing, because the mechanical half of the
    #: phase had succeeded. 44 minutes, zero tokens.
    #:
    #: A timeout is the case that makes SDK retries actively harmful: re-issuing
    #: an identical request with an identical deadline fails again by
    #: construction, and it does so silently, INSIDE what our logs count as one
    #: attempt. Our own layer already catches transport blips ("transient
    #: transport error — retrying the same call"), logs them per role, and can
    #: escalate the budget between tries, which the SDK cannot.
    max_retries: int = 0
    #: Per-request timeout. Without one, a slow or wedged provider blocks a
    #: twelve-phase run indefinitely at 0% CPU, which looks exactly like a hang.
    request_timeout: float = 180.0
    #: Hard ceilings.  A run that would exceed them aborts rather than surprising you.
    max_total_calls: int = 4000
    #: Runaway guard, derived as ~2x the whole-run output estimate. The RULE was
    #: right and its INPUT was stale: 6,000,000 was 2.0x an estimate built from
    #: `output_tokens_per_call` figures that were 4.4x too low, which made it
    #: **0.50x** what an honest run needs — it aborted every legitimate full run,
    #: not just runaways. Re-derived once the annotator, referee and architect
    #: budgets were measured: honest estimate 12,096,234, so 2x is ~24,000,000.
    #: Re-derive this whenever a role's budget changes; a guard sized from a
    #: stale projection is a guard that fires on correct behaviour.
    max_total_output_tokens: int = 24_000_000
    cache_llm_calls: bool = True
    record_raw_outputs: bool = True
    #: LangGraph's verified default is 10007 super-steps. An unbounded refinement
    #: loop would spend five figures before erroring, so we set it explicitly.
    recursion_limit: int = 200

    # --- multi-provider routing -----------------------------------------
    #: Refresh the model catalogue at most this often. Sources publish a
    #: 5-minute freshness window; six hours is well inside what a batch pipeline
    #: needs and avoids hammering them on every run.
    catalog_ttl_hours: float = 6.0
    #: Never reach the network for the catalogue. Falls back to cache, then a
    #: pinned snapshot, then the static deep/fast models.
    catalog_offline: bool = False
    #: Path to a pinned catalogue snapshot, for reproducing an old run's routing.
    catalog_pinned: str | None = None
    #: Labs whose models must never be candidates, by originating lab rather than
    #: by gateway — excluding "openai" also excludes `openrouter:openai/gpt-5.1`,
    #: which is the only version of this that works once an aggregator is in the
    #: pool. Empty by default: a general tool should not have opinions about whose
    #: models you may use.
    excluded_labs: list[str] = Field(default_factory=list)
    #: Explicit role -> model overrides. These win outright over the router.
    model_overrides: dict[str, str] = Field(default_factory=dict)
    #: Bare model ids a human has judged capable — the capability signal the
    #: router otherwise lacks. See `llm/router.route`'s `capable_models`: tier is
    #: a PRICE percentile, so without this the cheapest model clearing the bar
    #: wins, and across Chinese labs price does not rank capability at all.
    #: Gates the candidate set for run/phase blast-radius roles only.
    capable_models: list[str] = Field(default_factory=list)
    #: NOTE: there is deliberately no "prefer direct provider" flag. The router
    #: already promotes a lab's own endpoint over a gateway reselling the SAME
    #: bare model, unconditionally and with measured evidence (a 6.3x median-to-
    #: worst latency spread on the gateway path against 1.4x direct). A config
    #: knob would imply the preference is optional; it is not.
    #: Abort before starting if the estimated run cost exceeds this.
    budget_usd: float | None = None
    #: Nudge multilingual-critical roles toward Chinese-native labs when the
    #: corpus is Chinese and such a provider is configured.
    prefer_chinese_native: bool = False


class DeploymentConfig(BaseModel):
    margin_threshold: float = 0.02
    distill_cv_folds: int = 3
    live_demo_n: int = 8


class QMineConfig(BaseModel):
    """The whole configuration for one run."""

    project_name: str = "qmine"
    #: Language for reports and notebooks. Defaults to Chinese: the deliverables
    #: are read by the team that owns the corpus, and a definition sentence in a
    #: language they do not work in cannot be checked against the data.
    report_language: Literal["zh", "en"] = "zh"
    run_root: str = "runs"
    domain: DomainProfile = Field(default_factory=DomainProfile)
    data: DataConfig = Field(default_factory=DataConfig)
    representation: RepresentationConfig = Field(default_factory=RepresentationConfig)
    clustering: ClusteringConfig = Field(default_factory=ClusteringConfig)
    taxonomy: TaxonomyConfig = Field(default_factory=TaxonomyConfig)
    naming: NamingConfig = Field(default_factory=NamingConfig)
    gates: GateConfig = Field(default_factory=GateConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    deployment: DeploymentConfig = Field(default_factory=DeploymentConfig)

    seed_metric: int = SEED_METRIC
    seed_viz: int = SEED_VIZ
    seed_replay: tuple[int, int] = SEED_REPLAY

    #: Skip the expensive optional phases when smoke-testing the wiring.
    smoke_mode: bool = False
    #: Run a second-opinion agent over each bottom-up phase's artifacts while the
    #: run is still going. Measured on live38, the bottom-up path used 36 of 966
    #: agent calls (3.7%) and none before P7 — every representation, algorithm, K
    #: and hierarchy decision was made with no agent looking at it. The observer
    #: decides nothing; it cites artifact keys and can fail a gate. Off in
    #: `smoke_mode` so the demo stays cheap.
    #: Run the top-down and bottom-up routes as CONCURRENT graph branches.
    #: On by default: measured on live39, 38 min of taxonomy design plus 69 min
    #: of gold annotation sit in front of 39 min of bottom-up CPU work that
    #: depends on none of it, and the fork hides the whole of it.
    #:
    #: Turning it off restores the strict chain — the escape hatch if a provider,
    #: a filesystem or a future phase turns out not to tolerate two branches. It
    #: is also how the two schedules are compared on identical inputs.
    concurrent_branches: bool = True
    observe_phases: bool = True
    #: The pre-delivery audit — the only agent allowed to CHANGE a deliverable.
    #: On by default: its edits are mechanically bounded (anchored, sourced,
    #: language-checked) so it cannot ship a wrong number, and everything it does
    #: or is refused is printed in 交付前审核报告.md. Turn it off for a run whose
    #: deliverables must be byte-identical to what the phases produced.
    audit_delivery: bool = True
    #: The agent-written final report — the one document not assembled by Python.
    #: The scripted reports are correct and exhaustive; they are also generated
    #: section by section, so they read in the order the code ran rather than the
    #: order the argument makes sense in. This writes the narrative over the same
    #: artifacts: the agent chooses the structure and every sentence, while the
    #: numbers are checked value-by-value against a per-section fact sheet and the
    #: run's own warnings become a must-cover list checked over the whole text.
    #: Off means the run delivers the scripted documents alone.
    final_report: bool = True
    #: The pre-delivery auditor — the only agent allowed to EDIT a deliverable.
    #: Gated because a re-render that wants no model calls must be able to say so
    #: and mean it: with the offline stand-in this role produced three refused
    #: edits against a file called "[offline-heuristic] file" and three findings
    #: whose claim was the empty string, all written into 交付前审核报告.md. A
    #: stand-in's output is complete-looking prose that no model wrote, which in
    #: a deliverable is worse than an honest absence.
    delivery_audit: bool = True
    #: Machine-translate authored prose the curated `PROSE_ZH` mapping does not
    #: cover. Guarded: a translation that alters a number or an identifier is
    #: refused and the English kept, and results are cached by content hash so a
    #: string renders identically on every future run. Off in `offline`.
    translate_prose: bool = True
    #: Let an agent write the "what this means for THIS corpus" sentences in the
    #: deliverables. Every number it writes is checked against a fact sheet built
    #: from artifacts, and an unverifiable one is rejected and re-asked; after
    #: three failures the section ships with no commentary rather than unchecked
    #: commentary. Off in `smoke_mode`.
    interpret_results: bool = True
    #: Let an agent propose additional grid values from CORPUS CHARACTERISTICS —
    #: never from scores, which `ops.propose.assert_blind` enforces on the payload.
    #: The grids are K12 artefacts (`alpha_grid`, `k_sweep`, `expected_family_range`)
    #: applied unchanged to every corpus, and this is how a grid can come from the
    #: corpus instead. ON by default: additions are capped (the cap IS the
    #: multiple-comparisons budget), the proposer never sees a score, and every
    #: run grades whether a proposal actually won.
    #: **The "challenger must clear the incumbent by >2x the metric's measured
    #: noise" clause that stood here is NOT in force** — `challenger_beats_incumbent`
    #: has no production call site, and `propose_grid` returns a flat widened list
    #: so selection cannot tell a proposed value from a configured one. live40's
    #: K=7 was a proposed value that won; it is also the best value in its tie set
    #: on both reported metrics, so nothing was harmed — but it paid no toll.
    #: Off in `smoke_mode` regardless.
    propose_grids: bool = True
    #: Attack the classifier's own predictions with a paraphrase/perturbation
    #: agent and re-score. A pure second opinion: it changes no parameter and
    #: decides nothing, it only reports whether the accuracy survives contact.
    #: Off in `mode="fast"`.
    validate_adversarial: bool = True
    #: `full` runs the method as designed. `fast` runs THE SAME ANALYSIS with the
    #: second-opinion layer removed — one annotator instead of two, no phase
    #: observers, no adversarial attack, no agent-written narrative, no
    #: pre-delivery audit — and ships three reference documents instead of
    #: the full set of argued ones.
    #:
    #: What it does NOT do is shrink anything: the grids, the corpus, the gold
    #: size, the researcher panel and every intermediate artifact are identical
    #: to `full`. That is the whole distinction from `smoke_mode`, which shrinks
    #: the analysis and keeps the checks. A fast run's numbers are the numbers a
    #: full run would have produced; what is missing is the evidence that they
    #: were checked, and `fast_skipped` names every piece of it.
    mode: Literal["full", "fast"] = "full"
    #: WRITTEN BY THE VALIDATOR, NOT BY A USER — the machine-readable record of
    #: what `mode="fast"` turned off. Every fast deliverable's disclosure banner
    #: is rendered from this list, so a component cannot be skipped without the
    #: banner naming it: the two cannot drift, because there is only one source.
    fast_skipped: list[str] = Field(default_factory=list)
    offline: bool = Field(
        default=False, description="No network: hashing encoder + mock LLM."
    )

    @model_validator(mode="after")
    def _fast_mode_shrinks_grids(self) -> "QMineConfig":
        if self.smoke_mode:
            self.representation.alpha_grid = [0.0, 0.1, 0.5]
            self.clustering.k_sweep = [10, 15, 20, 30, 50]
            self.clustering.battery_k = [20]
            self.clustering.refine_rounds = 2
            self.taxonomy.gold_sample_size = min(self.taxonomy.gold_sample_size or 120, 120)
            self.taxonomy.n_researchers = 3
        return self

    @model_validator(mode="after")
    def _fast_mode_drops_the_second_opinion(self) -> "QMineConfig":
        """Turn off the checking layer — and RECORD it, in one place.

        Every component here is a second opinion: it reads what another part of
        the run produced and says whether to believe it. None of them chooses a
        parameter, a K, an alpha or a label, so removing them cannot move a
        result — which is exactly why they are the safe things to remove, and
        exactly why removing them leaves a run whose results are unverified.

        Deliberately NOT touched:

        * `alpha_grid`, `k_sweep`, `battery_k`, `refine_rounds`, `gold_sample_size`,
          `n_researchers` — the analysis. `smoke_mode` shrinks these; fast mode
          must not, or its answer is a different answer rather than the same
          answer unchecked.
        * `propose_grids` — a grid WIDENER, not a check. Dropping it would
          narrow the search and change the result.
        * `translate_prose` — the deliverables are still Chinese.
        * the findings ledger, the gate ledger, and every `store.put_*` call —
          the user's requirement is fewer documents, not less evidence.

        `fast_skipped` is written here and read by the deliverable banner, so a
        component added to this list appears in the banner with no second edit,
        and one removed from it disappears from the banner. A skip the reader is
        not told about is the failure this guards.
        """
        if self.mode != "fast":
            return self
        skipped: list[str] = []
        if self.taxonomy.annotators != 1:
            self.taxonomy.annotators = 1
            skipped.append("dual_annotation")
        # A single reading has nothing to agree with, so every measurement built
        # on agreement is not "skipped for time" — it is undefined. Zeroing the
        # repair rounds here is what stops the pipeline from trying to repair a
        # kappa that was never computed.
        self.taxonomy.kappa_repair_rounds = 0
        self.taxonomy.max_taxonomy_redraws = 0
        skipped += ["kappa_agreement", "pilot_ceiling", "kappa_repair", "taxonomy_redraw"]
        for field, name in (("observe_phases", "phase_observers"),
                            ("validate_adversarial", "adversarial_validation"),
                            ("final_report", "narrative_report"),
                            ("delivery_audit", "delivery_audit"),
                            ("interpret_results", "result_interpretation")):
            if getattr(self, field):
                setattr(self, field, False)
                skipped.append(name)
        self.fast_skipped = skipped
        return self

    # -- io -----------------------------------------------------------------
    @classmethod
    def load(cls, path: str | os.PathLike[str], **overrides: Any) -> "QMineConfig":
        """Read a config file. `extends:` pulls in another file underneath it.

        A config file REPLACES the default `configs/live.yaml` rather than
        merging with it — so a small file saying nothing but "this corpus\'s text
        column is `original_query`" silently discarded the entire provider
        policy: the pins, the lab-independence requirement that double-blind
        annotation depends on, and the capability list. `_load_config`\'s own
        docstring calls that "the one launch mistake nothing catches", and
        without `extends:` the only way to avoid it is to copy the whole policy
        into every corpus config, where it then drifts.

        `extends:` is resolved relative to THIS file, recursively, and the
        extending file wins key by key (`_deep_merge`), so a corpus config states
        only what is true of its corpus.
        """
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        base = raw.pop("extends", None)
        if base:
            base_path = Path(base)
            if not base_path.is_absolute():
                base_path = Path(path).parent / base
            if not base_path.exists():
                raise FileNotFoundError(
                    f"{path} extends {base!r}, which does not exist at {base_path}")
            parent = yaml.safe_load(base_path.read_text(encoding="utf-8")) or {}
            # A parent may itself extend; resolve depth-first by recursing on the
            # parent's own `extends` before merging this file over it.
            while parent.get("extends"):
                gp = Path(parent.pop("extends"))
                if not gp.is_absolute():
                    gp = base_path.parent / gp
                parent = _deep_merge(yaml.safe_load(gp.read_text(encoding="utf-8")) or {},
                                     parent)
            raw = _deep_merge(parent, raw)
        dom = raw.pop("domain_profile", None)
        if dom:
            dom_path = Path(dom)
            if not dom_path.is_absolute():
                dom_path = Path(path).parent / dom
            raw["domain"] = yaml.safe_load(dom_path.read_text(encoding="utf-8"))
        raw = _deep_merge(raw, overrides)
        return cls.model_validate(raw)

    def dump(self, path: str | os.PathLike[str]) -> None:
        Path(path).write_text(
            yaml.safe_dump(self.model_dump(mode="json"), allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

    @property
    def config_hash(self) -> str:
        return hash_params(self.model_dump(mode="json"))

    def run_dir(self, run_id: str) -> Path:
        return Path(self.run_root) / run_id


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def gold_size_for(n_rows: int, cfg: "TaxonomyConfig") -> int:
    """How many rows Phase 2b should annotate for a corpus of `n_rows`.

    The playbook asks for a stratified 3,000-5,000 (line 191) and, separately,
    for a *higher proportion* on small corpora (line 119) — the two rules point
    in different directions and the second is the one that is easy to miss. A
    fixed 600, which is what this shipped with, satisfies neither: it is five
    times under spec on a large corpus and, on a 2,000-row one, annotates 30% of
    the data by accident rather than by intent.

    The shape is: take the low end of the range; on a corpus small enough that
    this would be a thin slice, raise it to a fraction instead; never exceed a
    cap, because a gold set that is most of the corpus has stopped being a sample.
    """
    lo, hi = cfg.gold_size_range
    target = lo
    if n_rows < 10_000:
        target = max(target, int(n_rows * cfg.gold_min_fraction_small_corpus))
    target = min(target, hi, int(n_rows * cfg.gold_max_fraction))
    return max(50, target)


def alpha_sweep_k_for(cfg: "QMineConfig") -> int:
    """The K at which representation candidates are compared.

    Any fixed K is a stand-in for a prior about how coarse the intent axis is, so
    take it from the place that prior is actually declared — the domain's expected
    family range — instead of from a constant carried over from one corpus. The
    midpoint is used because the comparison only needs a scale at which the
    candidates are distinguishable, not the eventual answer: the real K is chosen
    later, in Phase 5, against the representation this step selects.
    """
    explicit = cfg.representation.alpha_sweep_k
    if explicit:
        return int(explicit)
    lo, hi = cfg.domain.expected_family_range
    return max(2, int(round((lo + hi) / 2)))
