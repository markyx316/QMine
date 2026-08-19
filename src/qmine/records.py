"""The typed records that flow through the pipeline.

These are deliberately more opinionated than plain dicts.  Three of the
playbook's principles are enforceable only if they are represented as *data*
rather than prose, and this module is where that representation lives:

* **Principle 3 — metrics must not betray the objective.**  Every
  :class:`MetricRecord` carries an ``authority``.  Selection functions accept
  only ``decisive`` metrics; silhouette is permanently ``advisory`` and the
  type system, not the analyst's memory, keeps it out of the vote.
* **Principle 6 — governance is executed, not recorded.**  A
  :class:`Prescription` starts life ``proposed`` and is only allowed to reach a
  report once it is ``executed`` with an ``evidence`` pointer.  The Phase 8 gate
  fails the run on any prescription that never made it into a data column.
* **Principle 2 — the reviewer holds a veto.**  A :class:`GateResult` from a
  human can be ``rejected``, and rejection carries a reason that is written into
  the decision ledger and re-read by later agents.
"""

from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel, Field

# ==========================================================================
# Metrics
# ==========================================================================

MetricAuthority = Literal["decisive", "advisory", "diagnostic"]

#: Which metrics are allowed to decide anything.  Phase 3c and Phase 5 read
#: this table rather than trusting a caller to remember Principle 3.
METRIC_AUTHORITY: dict[str, MetricAuthority] = {
    # decisive — these choose representations, K, and algorithms
    "stability_ari": "decisive",
    "template_fragmentation": "decisive",
    "heldout_reproduction": "decisive",
    "coherence": "decisive",
    "kappa": "decisive",
    # advisory — reported, plotted, argued about, but never given a vote
    "silhouette": "advisory",
    "inertia": "advisory",
    "davies_bouldin": "advisory",
    "calinski_harabasz": "advisory",
    # diagnostic — describe the run, do not compare options
    "n_clusters": "diagnostic",
    "noise_rate": "diagnostic",
    "ambiguous_rate": "diagnostic",
    "nmi_reference": "diagnostic",
    "purity_reference": "diagnostic",
    "distill_accuracy": "diagnostic",
    "ece": "diagnostic",
    "deep_aligned_k": "diagnostic",
    "population_weighted_accuracy": "diagnostic",
}


class MetricRecord(BaseModel):
    """One number, plus everything needed to compare it with another number.

    The uniform-panel rule (Principle 7) says two numbers may only be compared
    if they were produced by the same code on the same sub-sample under the same
    seed.  ``panel_id`` is that contract made checkable: the panel builder
    stamps every metric it computes, and the report renderer refuses to put two
    different ``panel_id`` values in one comparison table.
    """

    name: str
    value: float
    authority: MetricAuthority = "diagnostic"
    higher_is_better: bool = True
    panel_id: str = Field(default="", description="Uniform-panel identity: code+sample+seed.")
    subject: str = Field(default="", description="What was measured, e.g. 'hybrid_a0.1'.")
    n: int | None = None
    seed: int | None = None
    detail: dict[str, Any] = Field(default_factory=dict)
    note: str = ""

    @classmethod
    def make(cls, name: str, value: float, **kw: Any) -> "MetricRecord":
        kw.setdefault("authority", METRIC_AUTHORITY.get(name, "diagnostic"))
        kw.setdefault("higher_is_better", name not in _LOWER_IS_BETTER)
        return cls(name=name, value=float(value), **kw)

    @property
    def decisive(self) -> bool:
        return self.authority == "decisive"


_LOWER_IS_BETTER = {
    "template_fragmentation",
    "inertia",
    "davies_bouldin",
    "noise_rate",
    "ambiguous_rate",
    "ece",
}


class MetricSet(BaseModel):
    """All metrics for one candidate (a representation, an algorithm, a K)."""

    subject: str
    panel_id: str = ""
    metrics: dict[str, MetricRecord] = Field(default_factory=dict)

    def add(self, m: MetricRecord) -> "MetricSet":
        m.subject = m.subject or self.subject
        m.panel_id = m.panel_id or self.panel_id
        self.metrics[m.name] = m
        return self

    def get(self, name: str) -> float | None:
        m = self.metrics.get(name)
        return None if m is None else m.value

    def decisive_only(self) -> dict[str, MetricRecord]:
        return {k: v for k, v in self.metrics.items() if v.decisive}


# ==========================================================================
# Decisions and gates
# ==========================================================================

class DecisionRecord(BaseModel):
    """Why we chose what we chose — including what we rejected and why.

    Phase 11 requires a "failure history" section in every report.  It is not
    reconstructed at write-time from memory; it is a projection of these
    records, which are appended the moment a choice is made.
    """

    id: str
    phase: str
    question: str
    choice: str
    rationale: str
    decided_by: str = Field(default="agent", description="agent | human | metric")
    evidence: dict[str, Any] = Field(default_factory=dict)
    rejected: list[dict[str, Any]] = Field(
        default_factory=list, description="[{option, why_rejected, metrics}] — kept, never deleted."
    )
    decisive_metrics: list[str] = Field(default_factory=list)
    reversible: bool = True
    created_at: float = Field(default_factory=time.time)
    supersedes: str | None = None


GateStatus = Literal["passed", "warned", "failed", "rejected", "skipped", "pending"]


class GateResult(BaseModel):
    """The outcome of a quality gate.

    ``rejected`` is reserved for a human veto (Principle 2): the numbers may be
    fine and the reviewer may still say the family layer is incoherent.  That is
    a data point, not an argument to win, so it is stored with the same weight
    as a failed threshold — and it routes to a *new generation*, not a patch.
    """

    name: str
    phase: str
    status: GateStatus = "pending"
    blocking: bool = False
    observed: dict[str, Any] = Field(default_factory=dict)
    threshold: dict[str, Any] = Field(default_factory=dict)
    message: str = ""
    reviewer: str = ""
    remediation: str = ""
    created_at: float = Field(default_factory=time.time)

    @property
    def ok(self) -> bool:
        return self.status in ("passed", "warned", "skipped")

    @property
    def halts_run(self) -> bool:
        return self.blocking and self.status in ("failed", "rejected")


# ==========================================================================
# Governance (Phase 8)
# ==========================================================================

PrescriptionKind = Literal[
    "merge_families", "isolate_leaf", "split_leaf", "relabel", "flag_risk", "keep_as_is"
]
PrescriptionStatus = Literal["proposed", "accepted", "executed", "declined"]


class Prescription(BaseModel):
    """An audit finding, tracked until it becomes a column in the delivered data.

    ``status='executed'`` requires ``evidence``: the artifact and column that
    now carries the change, plus the metric deltas it caused.  Nothing else
    counts.  This is the mechanism behind the question that once broke a
    delivery — "were the issues actually fixed?"
    """

    id: str
    kind: PrescriptionKind
    targets: list[int] = Field(default_factory=list, description="Leaf or family ids.")
    target_names: list[str] = Field(default_factory=list)
    rationale: str = ""
    proposed_by: str = ""
    status: PrescriptionStatus = "proposed"
    executed_at: float | None = None
    evidence: dict[str, Any] = Field(
        default_factory=dict,
        description="{artifact, column, before, after, metric_deltas} once executed.",
    )
    decline_reason: str = Field(
        default="", description="If declined: the deliberate reason (Phase 8 step 5)."
    )

    @property
    def settled(self) -> bool:
        """True once the prescription can no longer embarrass us in a report."""
        return self.status in ("executed", "declined")


# ==========================================================================
# Taxonomy (Phase 2)
# ==========================================================================

class TaxonomyNode(BaseModel):
    """One class in the top-down intent taxonomy."""

    code: str
    name: str
    level: int = 1
    parent: str | None = None
    definition: str = Field(description="One sentence: what the user wants the system to do.")
    user_need: str = Field(default="", description="What, once received, satisfies them (Principle 11).")
    positive_examples: list[str] = Field(default_factory=list)
    negative_examples: list[str] = Field(default_factory=list)
    adjudication_rules: list[str] = Field(default_factory=list, description="Rule ids that touch this class.")
    axis: Literal["intent", "domain", "facet"] = "intent"
    risk: bool = False
    expected_share: float | None = None
    source_evidence: list[str] = Field(default_factory=list)
    pragmatic_only: bool = Field(
        default=False,
        description="True if clustering is expected to be structurally blind to it (Principle 1).",
    )


class AdjudicationRule(BaseModel):
    """A tie-break between two confusable classes.  Cited by id in every referee call."""

    id: str
    when: str = Field(description="The confusion this rule resolves.")
    then: str = Field(description="The class that wins.")
    rationale: str = ""
    examples: list[str] = Field(default_factory=list)
    classes: list[str] = Field(default_factory=list)
    added_in_round: int = 0
    added_because: str = Field(default="", description="Procedural memory: which disagreement created it.")
    #: For machine-generated rules, the exact condition that fires them — a marker
    #: substring, or ``"<no-marker>"`` for the fallback. Two rules conflict only
    #: when the SAME trigger on the SAME class pair yields different labels;
    #: comparing the rendered `when` sentence instead cannot tell "two markers for
    #: one boundary" (legitimate, and the whole point) from "one trigger, two
    #: answers" (a real contradiction), because both differ by a couple of
    #: characters in an otherwise identical template.
    trigger: str = ""


class Taxonomy(BaseModel):
    version: str = "v1"
    axes: dict[str, str] = Field(default_factory=dict)
    nodes: list[TaxonomyNode] = Field(default_factory=list)
    rules: list[AdjudicationRule] = Field(default_factory=list)
    labeling_guide: str = ""
    notes: str = ""

    def l1(self) -> list[TaxonomyNode]:
        return [n for n in self.nodes if n.level == 1]

    def l2(self) -> list[TaxonomyNode]:
        return [n for n in self.nodes if n.level == 2]

    def by_code(self, code: str) -> TaxonomyNode | None:
        return next((n for n in self.nodes if n.code == code), None)

    def label_vocabulary(self) -> set[str]:
        """Every string an agent must not see during blind naming (Principle 5)."""
        vocab: set[str] = set()
        for n in self.nodes:
            vocab.add(n.name)
            vocab.add(n.code)
        return {v for v in vocab if v and len(v) >= 2}


class GoldRow(BaseModel):
    """One gold-standard row: two independent labels, an adjudicated verdict, a trail."""

    query: str
    idx: int
    label_a: str = ""
    label_b: str = ""
    final: str = ""
    agreed: bool = False
    adjudicated: bool = False
    rule_cited: str = ""
    rationale_a: str = ""
    rationale_b: str = ""
    referee_rationale: str = ""
    round: int = 1
    #: `guide_repair` rows were labelled after the guide was rewritten, on a
    #: sample disjoint from the one the repair was derived from — they are the
    #: only rows whose kappa reflects the repaired guide rather than the original.
    source: Literal["stratified", "active_learning", "guide_repair"] = "stratified"


# ==========================================================================
# Bottom-up naming (Phase 7)
# ==========================================================================

class NamingCard(BaseModel):
    """What a blind naming agent is allowed to see.  Nothing else.

    No existing label, no other agent's answer, no taxonomy name, no legacy
    category.  The firewall in ``qmine.memory.context`` asserts this before the
    card is ever rendered into a prompt.
    """

    leaf_id: int
    size: int
    share: float
    center_samples: list[str] = Field(default_factory=list)
    random_samples: list[str] = Field(default_factory=list)
    edge_samples: list[str] = Field(
        default_factory=list, description="Deliberately included so impurity is visible."
    )
    top_ngrams: list[str] = Field(default_factory=list)
    length_stats: dict[str, float] = Field(default_factory=dict)


class LeafNaming(BaseModel):
    """A blind agent's verdict on one leaf."""

    leaf_id: int
    name_zh: str = Field(description="Action-object short name.")
    code: str = Field(description="English snake_case.")
    user_need: str = Field(description="One sentence: what satisfies the user (Principle 11).")
    coherence: int = Field(ge=1, le=5)
    mix_notes: str = ""
    risk_flag: bool = False
    risk_reason: str = ""
    named_by: str = ""


class FamilyNaming(BaseModel):
    family_id: int
    name_zh: str
    code: str
    definition: str
    leaf_ids: list[int] = Field(default_factory=list)
    coherent: bool = True
    audit_notes: str = ""
    risk: bool = False


class TreeAudit(BaseModel):
    """The auditor's report: the tree, plus everything wrong with it."""

    families: list[FamilyNaming] = Field(default_factory=list)
    cross_family_twins: list[dict[str, Any]] = Field(default_factory=list)
    duplicate_leaf_pairs: list[dict[str, Any]] = Field(default_factory=list)
    risk_isolated: bool = False
    risk_findings: list[dict[str, Any]] = Field(default_factory=list)
    incoherent_families: list[int] = Field(default_factory=list)
    prescriptions: list[Prescription] = Field(default_factory=list)
    summary: str = ""


# ==========================================================================
# Memory
# ==========================================================================

class LessonRecord(BaseModel):
    """Episodic memory, in the TradingAgents sense: what happened, what to do next time.

    Written after a gate fails, a human vetoes, or a metric contradicts an
    expectation.  Retrieved by similarity into later prompts so the team does not
    repeat a mistake it has already paid for.
    """

    id: str
    situation: str = Field(description="The state that preceded the outcome.")
    action: str = Field(description="What the team did.")
    outcome: str = Field(description="What happened.")
    lesson: str = Field(description="The transferable instruction.")
    phase: str = ""
    domain: str = ""
    severity: Literal["info", "warning", "critical"] = "info"
    created_at: float = Field(default_factory=time.time)


class TemplateGroup(BaseModel):
    """A phrasing family — mined once in Phase 1, used in three later phases."""

    name: str
    pattern: str
    intent_hint: str = ""
    n_hits: int = 0
    share: float = 0.0
    examples: list[str] = Field(default_factory=list)
    discovered: bool = Field(default=False, description="False = seeded, True = mined.")
    trusted: bool = Field(
        default=True,
        description=(
            "Whether this family may JUDGE a representation. Only groups whose "
            "contract actually holds — everything matching is the same intent — "
            "qualify. Seeded families are vouched for by a human; mined affixes "
            "are candidates until proven, because a marker like '是什么' attaches "
            "to every intent in the corpus and would poison the metric."
        ),
    )
    median_exemplar_idx: int | None = Field(
        default=None, description="Deterministic display exemplar (Principle 7)."
    )
