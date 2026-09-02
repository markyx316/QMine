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

import numpy as np
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

# ==========================================================================
# Metrics
# ==========================================================================

MetricAuthority = Literal["decisive", "advisory", "diagnostic"]

#: Which metrics are allowed to decide anything.  Phase 3c and Phase 5 read
#: this table rather than trusting a caller to remember Principle 3.
METRIC_AUTHORITY: dict[str, MetricAuthority] = {
    # decisive — these choose representations, K, and algorithms
    "stability_ari": "decisive",
    #: The corpus-and-k quantity the panel used to mislabel as `stability_ari`.
    #: Advisory: it cannot distinguish two candidates that share a k.
    "kmeans_refit_stability": "advisory",
    #: Two-sided alignment with the known-same-intent phrasing groups. The only
    #: metric here with an interior optimum in k, and ~10x more precise than replay.
    "intent_alignment_ami": "decisive",
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


#: Free-form record fields are `dict[str, Any]`, and "Any" in this pipeline
#: routinely means a numpy scalar: nearly every number in it is computed with
#: numpy. ormsgpack cannot encode one, so a single `numpy.float64` anywhere in a
#: record's evidence breaks LangGraph's checkpoint write — and the error names
#: the pydantic wrapper it was nested in ("Type is not msgpack serializable:
#: DecisionRecord"), not the scalar. That misdirection cost three investigations,
#: all of them aimed at the serializer's allowlist, which gates DECODING and was
#: never involved.
#:
#: Measured cost of the leak: `make demo` wrote 5 checkpoints for 17 phases, so
#: the run silently lost the ability to resume; live42 wrote none at all.
#:
#: Coercing at construction rather than at each call site is deliberate. The call
#: sites are every place that computes a metric, and one missed conversion
#: reintroduces the whole failure with a message that points somewhere else.
def _to_builtin(value: Any) -> Any:
    """Recursively convert numpy scalars and arrays to Python natives."""
    if isinstance(value, dict):
        return {k: _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    if isinstance(value, np.generic):          # np.float64, np.int64, np.bool_
        return value.item()
    if isinstance(value, np.ndarray):
        return [_to_builtin(v) for v in value.tolist()]
    return value


class _PlainValues(BaseModel):
    """A record whose free-form fields hold only checkpointable types."""

    @model_validator(mode="before")
    @classmethod
    def _coerce_numpy(cls, data: Any) -> Any:
        return _to_builtin(data) if isinstance(data, dict) else data


class MetricRecord(_PlainValues):
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

class DecisionRecord(_PlainValues):
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


class GateResult(_PlainValues):
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

#: What governance can be asked to do. `merge_leaves` is here because the tree
#: could SPLIT a leaf and MERGE families but never merge two leaves — so the
#: auditor's `duplicate_leaf_pairs` was a write-only measurement. live44 found
#: **14 duplicate pairs**, named them precisely ("汉字读音查询重复，任务无法区分"
#: for leaves 12/14), prescribed nothing on any of them, and shipped a delivered
#: tree containing two leaves with byte-identical names in the same family.
#: The asymmetry ran in the damaging direction: every run could only fragment.
PrescriptionKind = Literal[
    "merge_families", "merge_leaves", "isolate_leaf", "split_leaf",
    "relabel", "flag_risk", "keep_as_is"
]
PrescriptionStatus = Literal["proposed", "accepted", "executed", "declined"]


class Prescription(_PlainValues):
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
    #: The axes this taxonomy classifies along. DERIVED from the nodes when the
    #: agent leaves it empty, which it usually does: on live44 every one of the
    #: 20 nodes carried `axis="intent"` while this registry was `{}`, so the
    #: decision record read "across 1 axes" (counted from the nodes) against a
    #: registry naming none. An observer confirmed the contradiction and it was
    #: still open at delivery. A registry that cannot disagree with the nodes it
    #: describes is better than one nothing populates — this was write-only for
    #: the whole project, exactly like `TaxonomyNode.adjudication_rules` was.
    axes: dict[str, str] = Field(default_factory=dict)
    nodes: list[TaxonomyNode] = Field(default_factory=list)
    rules: list[AdjudicationRule] = Field(default_factory=list)
    labeling_guide: str = ""
    notes: str = ""

    @model_validator(mode="after")
    def _register_the_axes_the_nodes_actually_use(self) -> "Taxonomy":
        # An agent-supplied description wins; a missing one still gets an entry,
        # so `len(axes)` and `len({n.axis for n in nodes})` can never disagree.
        for node in self.nodes:
            axis = str(getattr(node, "axis", "") or "").strip()
            if axis:
                self.axes.setdefault(axis, "")
        return self

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
    #: "no contest pending on this row" — which is what all three consumers use
    #: it for (skip the referee, count the unresolved, size the agreement). It
    #: is NOT readable as "two annotators agreed" on its own: at
    #: `n_annotators == 1` every row is uncontested because there is only one
    #: reading, and nothing agreed with anything. Read it WITH `n_annotators`.
    agreed: bool = False
    #: How many independent annotators produced this row. 2 is the method; 1 is
    #: `mode="fast"`, and then `label_b` is empty, `agreed` carries no
    #: agreement, and kappa over these rows is undefined rather than perfect.
    n_annotators: int = 2
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
    """One family as the tree auditor describes it.

    **The booleans are TRI-STATE on purpose.** They defaulted to `True`
    (coherent) and `False` (risk) — the reassuring answer in both cases — so a
    generation that failed halfway produced families asserted coherent and
    not-risky rather than families with no verdict. That is the same permissive
    -default failure mode as `SectionDraft.markdown=""` and
    `AnnotationBatch.labels=[]`, the one that silently lost 1,500 gold rows, and
    here it sits on the SAFETY path: live44's F10 and F21 read `risk=false`
    while containing members the audit had itself flagged for isolation.

    `None` means the auditor did not say. A reader — or a check — can tell that
    apart from an assertion; `False` cannot.
    """

    family_id: int
    name_zh: str
    code: str
    definition: str
    leaf_ids: list[int] = Field(default_factory=list)
    coherent: bool | None = None
    audit_notes: str = ""
    risk: bool | None = None


class TreeAudit(_PlainValues):
    """The auditor's report: the tree, plus everything wrong with it."""

    families: list[FamilyNaming] = Field(default_factory=list)
    cross_family_twins: list[dict[str, Any]] = Field(default_factory=list)
    duplicate_leaf_pairs: list[dict[str, Any]] = Field(default_factory=list)
    #: The auditor's CLAIM that risk leaves are isolated — not a measurement.
    #: Nothing in `src/` computes or checks it, so it entered as a schema field
    #: rather than through one of the four agent doors and carries no guardrail.
    #: On live44 it read `true` while leaf 47's gambling member still shared
    #: family F10 with benign leaves 8 and 9, and its recorded action was only
    #: "建议拆分/隔离该成员" — governance had not run yet. Three separate
    #: observers reached confirmed-but-wrong conclusions from it.
    #: `None` = the auditor did not assert isolation.
    risk_isolated: bool | None = None
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


def paired_gate_metric(gate: Any) -> tuple[str, float, float, bool] | None:
    """The one observed value a gate's threshold is actually ABOUT.

    Returns ``(field, observed, threshold, higher_is_better)`` or ``None`` when
    the gate has no numeric pair — a boolean assertion, a gate that skipped, a
    threshold recorded as prose.

    **Both callers used to guess, differently.** `plot_gates` took the first
    numeric value out of `observed` and the first out of `threshold` with nothing
    tying them together, so a gate observing `{"n": 600, "kappa": 0.8928}`
    against `{"min_kappa": 0.70}` was drawn as 600 versus 0.70 — headroom of
    +856, on a chart whose whole premise is distance from the bar. And because
    `bool` is a subclass of `int` in Python, a gate asserting `{"lopsided":
    True}` contributed a value of 1 and was plotted as if it had cleared its bar.

    Matching is by like NAME, with the `min_` / `max_` / `_floor` prefixes that
    thresholds conventionally carry stripped off, which is the rule
    `report.zh_bottomup._passed_below_threshold` already applied. One definition
    so a figure and a table can never disagree about which number a gate is
    about.
    """
    obs = getattr(gate, "observed", None)
    thr = getattr(gate, "threshold", None)
    if obs is None and isinstance(gate, dict):
        obs, thr = gate.get("observed"), gate.get("threshold")
    if not isinstance(obs, dict) or not isinstance(thr, dict):
        return None

    def numeric(v: Any) -> bool:
        # `isinstance(True, int)` is True. A boolean assertion is not a
        # measurement against a bar and must never be plotted as one.
        return isinstance(v, (int, float)) and not isinstance(v, bool)

    pairs = gate_metric_pairs(gate)
    return pairs[0] if pairs else None


def gate_metric_pairs(gate: Any) -> list[tuple[str, float, float, bool]]:
    """EVERY observed/threshold pair a gate carries, not just the first.

    A gate can be judged on several numbers at once. `paired_gate_metric`
    returns one because a bar chart plots one bar; a question like "is this gate
    passing while any of its own numbers sits under its bar" has to look at all
    of them. Collapsing the two cost a real regression: `_passed_below_threshold`
    stopped flagging `p2b_kappa`, whose SECOND threshold is the one below bar,
    and the Chinese "带保留通过" prefix that flag adds was the only CJK on a line
    whose message is authored in English — so an untranslated gate conclusion
    started reaching a Chinese report, caught by
    `test_every_authored_rationale_reaches_the_reader_in_the_report_language`.
    """
    obs = getattr(gate, "observed", None)
    thr = getattr(gate, "threshold", None)
    if obs is None and isinstance(gate, dict):
        obs, thr = gate.get("observed"), gate.get("threshold")
    if not isinstance(obs, dict) or not isinstance(thr, dict):
        return []

    def numeric(v: Any) -> bool:
        return isinstance(v, (int, float)) and not isinstance(v, bool)

    out: list[tuple[str, float, float, bool]] = []
    for tk, tv in thr.items():
        if not numeric(tv):
            continue
        base = str(tk).replace("min_", "").replace("max_", "").replace("_floor", "")
        higher_is_better = not str(tk).startswith("max_")
        for ok, ov in obs.items():
            if not numeric(ov):
                continue
            k = str(ok)
            if k == base or k.endswith(base) or base.endswith(k):
                # NO `break`. One threshold can name several observed values —
                # `min_kappa` matches `kappa` AND `self_consistency_kappa` — and
                # stopping at the first silently drops the rest. That regressed
                # `_passed_below_threshold`, which then stopped flagging a gate
                # whose LATER number is the one under its bar; the Chinese
                # "带保留通过" prefix that flag adds was the only CJK on a line
                # whose message is authored in English, so an untranslated gate
                # conclusion reached a Chinese report. Caught by
                # `test_every_authored_rationale_reaches_the_reader_in_the_report_language`.
                out.append((k, float(ov), float(tv), higher_is_better))
    return out
