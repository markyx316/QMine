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

    key: str = "generic"
    display_name: str = "Generic query log"
    language: Literal["zh", "en", "multi"] = "zh"

    #: Tokenisation and n-gram ranges (Part IV section 4.5).
    #: ``auto`` is resolved in Phase 1 from the corpus's actual script mix. Use it
    #: whenever the language is not known in advance — assuming a tokeniser is
    #: one of the cheapest ways to quietly degrade every downstream phase.
    tokenizer: Literal["jieba", "whitespace", "none", "auto"] = "jieba"
    char_ngram_range: tuple[int, int] = (1, 3)
    word_ngram_range: tuple[int, int] = (1, 2)

    #: Base encoder candidates for the bake-off (Phase 3a).  Order is a hint,
    #: not a decision — the bake-off decides.
    embedding_candidates: list[str] = Field(
        default_factory=lambda: ["BAAI/bge-small-zh-v1.5", "BAAI/bge-base-zh-v1.5"]
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


class TaxonomyConfig(BaseModel):
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
    #: The repair round scores a FRESH sample. Re-scoring the rows the repair was
    #: derived from measures how well the rules fit those rows, not whether the
    #: guide got clearer.
    repair_on_fresh_sample: bool = True
    active_learning_rounds: int = 1
    active_learning_batch: int = 200
    rule_precision_floor: float = 0.98


class NamingConfig(BaseModel):
    n_naming_agents: int = 5
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
    kappa: float = 0.90
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
    #: Kept low on purpose: the provider SDK already retries, and LangGraph node
    #: policies retry on top of that. Three layers at 3 attempts each is 27 calls
    #: for one logical request.
    max_retries: int = 2
    #: Per-request timeout. Without one, a slow or wedged provider blocks a
    #: twelve-phase run indefinitely at 0% CPU, which looks exactly like a hang.
    request_timeout: float = 180.0
    #: Hard ceilings.  A run that would exceed them aborts rather than surprising you.
    max_total_calls: int = 4000
    max_total_output_tokens: int = 6_000_000
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
    #: Explicit role -> model overrides. These win outright over the router.
    model_overrides: dict[str, str] = Field(default_factory=dict)
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
    fast_mode: bool = False
    offline: bool = Field(
        default=False, description="No network: hashing encoder + mock LLM."
    )

    @model_validator(mode="after")
    def _fast_mode_shrinks_grids(self) -> "QMineConfig":
        if self.fast_mode:
            self.representation.alpha_grid = [0.0, 0.1, 0.5]
            self.clustering.k_sweep = [10, 15, 20, 30, 50]
            self.clustering.battery_k = [20]
            self.clustering.refine_rounds = 2
            self.taxonomy.gold_sample_size = min(self.taxonomy.gold_sample_size or 120, 120)
            self.taxonomy.n_researchers = 3
        return self

    # -- io -----------------------------------------------------------------
    @classmethod
    def load(cls, path: str | os.PathLike[str], **overrides: Any) -> "QMineConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
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
