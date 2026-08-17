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
    tokenizer: Literal["jieba", "whitespace", "none"] = "jieba"
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
    alpha_sweep_k: int = 20
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
    gold_sample_size: int = 600
    pilot_sample_size: int = 50
    pilot_agreement_threshold: float = 0.85
    kappa_threshold: float = 0.90
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

    template_coverage_range: tuple[float, float] = (0.20, 0.40)
    pilot_agreement: float = 0.85
    kappa: float = 0.90
    heldout_reproduction: float = 0.98
    coherence: float = 4.0
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

    provider: Literal["anthropic", "openai", "mock", "auto"] = "auto"
    deep_model: str = "claude-opus-5"
    fast_model: str = "claude-sonnet-5"
    temperature: float | None = Field(
        default=None, description="None = omit. Required for Opus 5 / Fable 5, which 400 on any value."
    )
    #: Generous because on Opus 5 thinking is on by default and max_tokens caps
    #: thinking plus response together — a limit sized for a non-thinking model
    #: truncates mid-answer.
    max_tokens: int = 16000
    max_concurrency: int = 4
    #: Kept low on purpose: the provider SDK already retries, and LangGraph node
    #: policies retry on top of that. Three layers at 3 attempts each is 27 calls
    #: for one logical request.
    max_retries: int = 2
    #: Hard ceilings.  A run that would exceed them aborts rather than surprising you.
    max_total_calls: int = 4000
    max_total_output_tokens: int = 6_000_000
    cache_llm_calls: bool = True
    record_raw_outputs: bool = True
    #: LangGraph's verified default is 10007 super-steps. An unbounded refinement
    #: loop would spend five figures before erroring, so we set it explicitly.
    recursion_limit: int = 200


class DeploymentConfig(BaseModel):
    margin_threshold: float = 0.02
    distill_cv_folds: int = 3
    live_demo_n: int = 8


class QMineConfig(BaseModel):
    """The whole configuration for one run."""

    project_name: str = "qmine"
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
            self.taxonomy.gold_sample_size = min(self.taxonomy.gold_sample_size, 120)
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
