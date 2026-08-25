"""The roster.

Eleven roles, mapped one-to-one onto the playbook's division of labour.  Each is
a thin subclass of :class:`~qmine.agents.base.Agent` that knows how to assemble
its evidence and what shape its answer must take.

The interesting design work is not in any single role; it is in what each role
is *denied*.  Researchers see one slice and no other researcher's findings.
Annotators see the guide and no other annotator's labels.  Namers see member
queries and nothing else at all.  The adversary is told to attack rather than
verify.  Those restrictions are what make the aggregate trustworthy, and they
are enforced by what ``build_user`` puts in the prompt — not by asking the model
to forget.
"""

from __future__ import annotations

import json
from typing import Any, Literal, Sequence

from pydantic import BaseModel, Field

from ..memory.context import budget_text, render_card
from ..records import (
    AdjudicationRule,
    LeafNaming,
    NamingCard,
    Taxonomy,
    TaxonomyNode,
    TreeAudit,
)
from .base import Agent, ToolAgent


# ==========================================================================
# Phase 2a — research fan-out
# ==========================================================================

class CandidateCategory(BaseModel):
    name: str
    code: str = ""
    definition: str
    user_need: str = ""
    evidence: list[str] = Field(default_factory=list)
    estimated_share: float | None = None
    pragmatic: bool = False
    risk: bool = False
    axis: Literal["intent", "domain", "facet"] = "intent"


class ResearchSubmission(BaseModel):
    angle: str = ""
    #: Did this angle's tool loop actually run? Set by the phase, not the model —
    #: an agent cannot be asked whether it searched. Without it `taxonomy.json`
    #: cannot distinguish an angle that ran a dozen web searches from one whose
    #: tool loop died and answered from parametric knowledge, and the taxonomy is
    #: described to the reader as web-researched either way.
    web_researched: bool = False
    candidates: list[CandidateCategory] = Field(default_factory=list)
    observations: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(
        default_factory=list, description="Where the evidence contradicted an expectation."
    )
    open_questions: list[str] = Field(default_factory=list)


#: The five standing research angles.  Deliberately non-overlapping: overlapping
#: assignments produce three agents rediscovering the same obvious categories
#: and nobody covering the awkward ones.
#: `web` marks angles whose evidence lives outside the corpus. The log-reading
#: and pragmatic-intent angles are deliberately excluded: their value is direct
#: observation of the rows, and a search box invites recall to replace it.
RESEARCH_ANGLES: list[dict[str, Any]] = [
    {
        "key": "log_reading",
        "web": False,
        "assignment": (
            "Read your assigned slice of raw queries line by line and propose intent "
            "categories grounded ONLY in what you actually read. You are the closest "
            "thing this team has to direct observation — no other researcher is "
            "reading raw rows. Quote real queries for every candidate."
        ),
    },
    {
        "key": "literature",
        "web": True,
        "assignment": (
            "Draw on published query-intent taxonomies for this kind of corpus "
            "(e.g. informational/navigational/transactional; e-commerce five-way "
            "splits; question-intent taxonomies) and propose which of their "
            "distinctions transfer here and which do not. Say explicitly where a "
            "published distinction would NOT survive contact with this corpus — a "
            "borrowed taxonomy that does not fit is worse than none."
        ),
    },
    {
        "key": "legacy_audit",
        "web": False,
        "assignment": (
            "Audit the existing/legacy taxonomy in the evidence. For each legacy "
            "class decide: inheritable structure, or unsorted traffic wearing a "
            "label? Pay attention to classes defined by query SHAPE (length, script, "
            "punctuation) rather than intent — those hide real categories inside "
            "them, and recovering what is buried there is your main deliverable."
        ),
    },
    {
        "key": "pragmatic_intents",
        "web": False,
        "assignment": (
            "Hunt specifically for intents that are INVISIBLE IN THE WORDING — where "
            "two queries can be phrased near-identically and want opposite things "
            "(verification vs definition; solve-this vs explain-this; navigate vs "
            "learn). Unsupervised clustering will never recover these, so if you do "
            "not find them, nobody will. This is the highest-leverage angle on the team."
        ),
    },
    {
        "key": "risk_compliance",
        "web": True,
        "assignment": (
            "Identify categories with a safety, legal, or compliance dimension: "
            "gambling probes, requests for individualised professional advice, fraud, "
            "age-inappropriate content, rumour amplification. Volume is irrelevant to "
            "whether a category belongs on this list."
        ),
    },
]


class ResearcherAgent(ToolAgent):
    """A taxonomy researcher, with web access on the angles that need it.

    Only the literature and competitor angles get tools. The log-reading angle
    deliberately does not: its whole value is that one agent forms its view from
    the raw rows and nothing else, and giving it a search box invites it to
    replace observation with recall.
    """

    role = "researcher"
    prompt_name = "researcher"
    schema = ResearchSubmission
    tools: list[Any] = []

    def build_system(self, *, assignment: str = "", **kw: Any) -> str:
        return super().build_system().replace("{{ASSIGNMENT}}", assignment)

    def build_user(self, *, evidence: str = "", domain_notes: str = "", **kw: Any) -> str:
        return (
            f"## Domain\n{domain_notes}\n\n"
            f"## Evidence for your angle\n{budget_text(evidence, 24000, tail=2000)}\n\n"
            "Return your submission."
        )


# ==========================================================================
# Phase 2a — synthesis and critique
# ==========================================================================

class TaxonomyDraft(BaseModel):
    nodes: list[TaxonomyNode] = Field(default_factory=list)
    rules: list[AdjudicationRule] = Field(default_factory=list)
    labeling_guide: str = ""
    design_notes: str = ""
    dropped_candidates: list[dict[str, str]] = Field(
        default_factory=list, description="[{name, why_dropped}] — kept for the failure-history section."
    )


class RuleSet(BaseModel):
    rules: list[AdjudicationRule] = Field(default_factory=list)
    pairs_considered: int = Field(
        0, description="How many class pairs were examined, whether or not a rule was written."
    )


class RuleWriterAgent(Agent):
    """Writes the adjudication rules, given a class list that already exists.

    Split out of the architect for two measured reasons. Asking one call for the
    classes *and* the rules exceeded a 42,000-token output ceiling — most of it
    reasoning — and when the rule requirement was hardened to stop it writing only
    one rule, it satisfied that by returning two classes instead of nineteen.

    Separating them removes both failures structurally rather than by instruction.
    Each call is small enough to finish, and this one is *shown* the final class
    list, so it cannot invent a tie-break between classes that do not exist — which
    is what made two dozen otherwise-good rules unusable.
    """

    role = "taxonomy_architect"
    prompt_name = "rule_writer"
    schema = RuleSet

    def build_system(self, *, min_rules: int = 20, **kw: Any) -> str:
        return super().build_system().replace("{{MIN_RULES}}", str(min_rules))

    def build_user(self, *, nodes: Sequence[Any] = (), domain_notes: str = "", **kw: Any) -> str:
        listing = "\n".join(
            f"- `{n.code}` — {n.name}: {n.definition}" for n in nodes
        )
        return (
            f"## Domain\n{domain_notes}\n\n"
            f"## The final class list ({len(nodes)} classes)\n{listing}\n\n"
            "Every rule's `then` MUST be one of the codes above, exactly as written. "
            "Work through the pairs a rater would have to think about and write a "
            "tie-break for each."
        )


class TaxonomyRedrawAgent(Agent):
    """Redraws only the boundaries a pilot proved are not in the data.

    The pilot already knows which pairs are broken *and* which remedy applies: a
    pair one annotator cannot reproduce against itself is a boundary the query
    does not carry, and no tie-break rule can rescue it. That finding used to be
    printed in a halt message and thrown away — three runs halted, prescribed
    nothing, and waited for a human to redraw by hand.

    Reusing the architect for this does not work: it is shown the researchers'
    evidence and rebuilds from scratch, which discards the classes that were
    fine and re-rolls the ones that were not. This one is shown the *current*
    taxonomy and the specific failing pairs, and is told to change nothing else.
    """

    role = "taxonomy_architect"
    prompt_name = "taxonomy_redraw"
    schema = TaxonomyDraft

    def build_system(self, *, l1_range: tuple[int, int] = (15, 25), **kw: Any) -> str:
        lo, hi = l1_range
        return (super().build_system()
                .replace("{{L1_MIN}}", str(lo))
                .replace("{{L1_MAX}}", str(hi)))

    def build_user(self, *, nodes: Sequence[Any] = (), pairs: Sequence[Any] = (),
                   domain_notes: str = "", n_pilot: int = 0, **kw: Any) -> str:
        listing = "\n".join(
            f"- `{n.code}` — {n.name}: {n.definition}\n"
            f"    yes: {list(n.positive_examples)[:4]}\n"
            f"    no:  {list(n.negative_examples)[:2]}"
            for n in nodes
        )
        broken = "\n".join(
            f"- **{pair}** — the same annotator resolved {count} of {n_pilot} pilot "
            f"queries differently on a second pass"
            for pair, count in pairs
        )
        return (
            f"## Domain\n{domain_notes}\n\n"
            f"## The current taxonomy ({len(nodes)} classes)\n{listing}\n\n"
            f"## Boundaries that failed the reproducibility test\n{broken}\n\n"
            "Return the complete class list with these boundaries redrawn and "
            "everything else byte-identical."
        )


class ArchitectAgent(Agent):
    role = "taxonomy_architect"
    prompt_name = "architect"
    schema = TaxonomyDraft

    def build_system(self, *, l1_range: tuple[int, int] = (15, 25), min_rules: int = 20, **kw: Any) -> str:
        return (
            super()
            .build_system()
            .replace("{{L1_MIN}}", str(l1_range[0]))
            .replace("{{L1_MAX}}", str(l1_range[1]))
            .replace("{{MIN_RULES}}", str(min_rules))
        )

    def build_user(
        self, *, submissions: Sequence[ResearchSubmission] = (), domain_notes: str = "",
        pragmatic_hints: Sequence[str] = (), memory_block: str = "", **kw: Any
    ) -> str:
        parts = [f"## Domain\n{domain_notes}"]
        if pragmatic_hints:
            parts.append(
                "## Intents predicted to be invisible to clustering\n"
                "These must be carried by this taxonomy — nothing downstream will find them:\n"
                + "\n".join(f"- {h}" for h in pragmatic_hints)
            )
        if memory_block:
            parts.append(memory_block)
        for i, s in enumerate(submissions):
            parts.append(
                f"## Researcher {i + 1} — angle: {s.angle}\n"
                + json.dumps(s.model_dump(), ensure_ascii=False, indent=1)[:9000]
            )
        parts.append("Synthesise one taxonomy. Return the draft.")
        return "\n\n".join(parts)


class CritiqueFinding(BaseModel):
    kind: Literal["overlap", "gap", "catchall", "form_defined", "untestable", "missing_risk"]
    classes: list[str] = Field(default_factory=list)
    evidence_query: str = ""
    defect: str
    fix: str


class CritiqueReport(BaseModel):
    findings: list[CritiqueFinding] = Field(default_factory=list)
    estimated_catchall_share: float | None = None
    verdict: Literal["ship", "revise", "reject"] = "revise"
    summary: str = ""


class CriticAgent(Agent):
    role = "taxonomy_critic"
    prompt_name = "critic"
    schema = CritiqueReport

    def build_user(self, *, taxonomy: Taxonomy | TaxonomyDraft = None, sample_queries: Sequence[str] = (), **kw: Any) -> str:
        nodes = taxonomy.nodes if taxonomy else []
        rules = taxonomy.rules if taxonomy else []
        return (
            "## Taxonomy under review\n"
            + json.dumps([n.model_dump() for n in nodes], ensure_ascii=False, indent=1)[:20000]
            + "\n\n## Adjudication rules\n"
            + json.dumps([r.model_dump() for r in rules], ensure_ascii=False, indent=1)[:8000]
            + "\n\n## Sample queries to test it against\n"
            + "\n".join(f"- {q}" for q in sample_queries[:120])
            + "\n\nBreak it."
        )


# ==========================================================================
# Phase 2b — annotation and adjudication
# ==========================================================================

class QueryLabel(BaseModel):
    query: str
    label: str
    confidence: Literal["high", "medium", "low"] = "medium"
    rule_cited: str = ""
    rationale: str = ""


class AnnotationBatch(BaseModel):
    labels: list[QueryLabel] = Field(default_factory=list)


class AnnotatorAgent(Agent):
    role = "annotator"
    prompt_name = "annotator"
    schema = AnnotationBatch

    def build_user(
        self, *, queries: Sequence[str] = (), guide: str = "", classes: str = "", rules: str = "", **kw: Any
    ) -> str:
        return (
            # KEEP THE TAIL. Both of these blocks are APPENDED to, and a
            # head-only trim removes exactly the newest content: the referee's
            # rules are added by `taxonomy.rules.extend(new_rules)`, and the
            # guide-repair round appends its 边界裁定 section to the guide. Those
            # are the most authoritative lines in each block — the ones written
            # in response to observed disagreement — and they sit last.
            # Measured: 72 rules render to 8,128 chars, 90% of the old 9,000
            # budget, and the referee adds more on top of that.
            f"## Classes\n{budget_text(classes, 18000, tail=2000, label='classes')}\n\n"
            # Sized to FIT, not to trim: a 22-class taxonomy plus the referee's
            # drafted rules rendered to 18,496 chars on live38, and every char
            # over the budget was adjudication guidance that never reached the
            # annotator. `_render_rules` now orders newest-first so that if this
            # is ever exceeded again, the rules lost are the oldest.
            # Sized from what the pipeline MEASURABLY produces, not from a guess.
            # On live38 the referee's 82 rules rendered the block to 46,814 chars
            # (~478 each — its `when` is the model's proposed rule text, 3x the
            # architect's) and the guide reached 24,155 because the repair
            # appended those same rules verbatim. The guide now cites rule ids
            # instead of repeating them (44,070 -> 8,654), so the BINDING
            # rulings always fit; the rule block stays reference detail and is
            # ordered newest-first, so anything cut is the oldest.
            # Wide enough that nothing the pipeline currently produces is cut:
            # measured 46,814 chars of rules and 8,654 of binding rulings on
            # live38, against models with 262,144-1,000,000 token windows. These
            # are a RUNAWAY GUARD, not a working limit — a corpus with ten times
            # the classes should be truncated loudly rather than silently blow a
            # context window, and `_render_rules` orders newest-first so what
            # goes is the oldest.
            f"## Adjudication rules\n{budget_text(rules, 60000, tail=10000, label='adjudication rules')}\n\n"
            f"## Labelling guide\n{budget_text(guide, 20000, tail=4000, label='labelling guide')}\n\n"
            f"## Queries to label ({len(queries)})\n"
            + "\n".join(f"{i + 1}. {q}" for i, q in enumerate(queries))
            + "\n\nLabel every one."
        )


class RefereeVerdict(BaseModel):
    query: str
    final_label: str
    rule_cited: str = ""
    rule_gap: bool = False
    proposed_rule: str = ""
    rationale: str = ""
    both_defensible: bool = False


class RefereeBatch(BaseModel):
    verdicts: list[RefereeVerdict] = Field(default_factory=list)


class RefereeAgent(Agent):
    role = "referee"
    prompt_name = "referee"
    schema = RefereeBatch

    def build_user(
        self, *, disagreements: Sequence[dict[str, Any]] = (), classes: str = "", rules: str = "",
        decided: Sequence[dict[str, str]] = (), **kw: Any
    ) -> str:
        rows = "\n".join(
            f"{i + 1}. QUERY: {d['query']}\n   A said {d['label_a']} ({d.get('rationale_a', '')})"
            f"\n   B said {d['label_b']} ({d.get('rationale_b', '')})"
            for i, d in enumerate(disagreements)
        )
        # Rulings this referee already made on earlier batches. Without them each
        # batch re-decides the same boundary from scratch and they contradict each
        # other: one live run produced a rule sending bare single characters to
        # CHITCHAT_NOISE and another sending them to CHAR_PRONUNCIATION.
        prior = ""
        if decided:
            lines = "\n".join(
                f"- {d['pair']} → **{d['final']}** (e.g. {d['example']})" for d in decided
            )
            prior = (
                f"\n\n## Boundaries you have ALREADY decided in this session ({len(decided)})\n"
                f"{budget_text(lines, 6000)}\n\n"
                "These are binding. Rule the same way on the same boundary, and do not "
                "propose a rule that contradicts one of them. If you believe an earlier "
                "ruling was wrong, follow it anyway and say so in your rationale — "
                "consistency across the gold set matters more than any single row."
            )
        return (
            f"## Classes\n{budget_text(classes, 14000)}\n\n"
            f"## Adjudication rules\n{budget_text(rules, 9000)}{prior}\n\n"
            f"## Disagreements ({len(disagreements)})\n{rows}\n\nAdjudicate every one."
        )


# ==========================================================================
# Phase 2d — adversarial validation
# ==========================================================================

class AttackResult(BaseModel):
    query: str
    assigned_label: str
    attack: str
    better_label: str = ""
    verdict: Literal["wrong", "defensible", "correct"] = "correct"


class AdversarialBatch(BaseModel):
    results: list[AttackResult] = Field(default_factory=list)


class AdversaryAgent(Agent):
    role = "adversary"
    prompt_name = "adversary"
    schema = AdversarialBatch

    def build_user(self, *, rows: Sequence[dict[str, str]] = (), classes: str = "", **kw: Any) -> str:
        listing = "\n".join(f"{i + 1}. {r['query']}  →  labelled {r['label']}" for i, r in enumerate(rows))
        return (
            f"## Available classes\n{budget_text(classes, 12000)}\n\n"
            f"## Labelled queries to attack ({len(rows)})\n{listing}\n\nAttack every one."
        )


# ==========================================================================
# Phase 7 — blind naming and audit
# ==========================================================================

class NamerAgent(Agent):
    """The blind namer.

    ``build_user`` renders through the firewall, so a card carrying any label
    vocabulary raises before it can reach the model.  The check is here rather
    than at the call site because a call site can be forgotten.
    """

    role = "namer"
    prompt_name = "namer"
    schema = LeafNaming

    def build_user(self, *, card: NamingCard = None, **kw: Any) -> str:
        return render_card(card, firewall=self.ctx.firewall) + "\n\nName this group."

    def postprocess(self, out: LeafNaming, *, card: NamingCard = None, **kw: Any) -> LeafNaming:
        out.leaf_id = card.leaf_id
        out.named_by = f"{self.role}{self.suffix}@{self.ctx.registry.provider}"
        return out


class AuditorAgent(Agent):
    role = "tree_auditor"
    prompt_name = "auditor"
    schema = TreeAudit

    def build_user(
        self,
        *,
        namings: Sequence[LeafNaming] = (),
        leaf_family: Sequence[int] = (),
        leaf_sizes: Sequence[int] = (),
        template_spread: dict[str, Any] | None = None,
        centroid_similarity: list[dict[str, Any]] | None = None,
        **kw: Any,
    ) -> str:
        rows = [
            {
                "leaf_id": n.leaf_id,
                "name": n.name_zh,
                "code": n.code,
                "user_need": n.user_need,
                "coherence": n.coherence,
                "mix_notes": n.mix_notes,
                "risk_flag": n.risk_flag,
                "risk_reason": n.risk_reason,
                "family_hint": int(leaf_family[n.leaf_id]) if n.leaf_id < len(leaf_family) else None,
                "size": int(leaf_sizes[n.leaf_id]) if n.leaf_id < len(leaf_sizes) else None,
            }
            for n in namings
        ]
        parts = [
            "## Every cluster, as named independently by blind reviewers\n"
            + json.dumps(rows, ensure_ascii=False, indent=1)[:40000]
        ]
        if template_spread:
            parts.append(
                "## Phrasing families and where their members landed\n"
                "A family spread across several clusters is direct evidence of a "
                "cross-family twin — same intent, split by wording.\n"
                + json.dumps(template_spread, ensure_ascii=False, indent=1)[:8000]
            )
        if centroid_similarity:
            parts.append(
                "## Most similar cluster pairs by centroid cosine\n"
                + json.dumps(centroid_similarity[:40], ensure_ascii=False, indent=1)[:6000]
            )
        parts.append("Build the family layer, then audit it. Write prescriptions.")
        return "\n\n".join(parts)


# ==========================================================================
# Risk, reporting, maintenance
# ==========================================================================

class RiskFinding(BaseModel):
    category: str
    cluster_ids: list[int] = Field(default_factory=list)
    severity: Literal["low", "medium", "high"] = "medium"
    rationale: str = ""
    evidence: list[str] = Field(default_factory=list)
    recommended_policy: Literal["isolate", "isolate_and_flag", "drop", "monitor"] = "isolate_and_flag"


class RiskReport(BaseModel):
    findings: list[RiskFinding] = Field(default_factory=list)
    clean: bool = True
    summary: str = ""


class RiskSentinelAgent(Agent):
    role = "risk_sentinel"
    prompt_name = "risk_sentinel"
    schema = RiskReport

    def build_user(self, *, cluster_samples: dict[int, list[str]] = None, **kw: Any) -> str:
        blocks = [
            f"### Cluster {cid}\n" + "\n".join(f"- {s}" for s in samples[:12])
            for cid, samples in (cluster_samples or {}).items()
        ]
        return "## Clusters to review\n\n" + "\n\n".join(blocks)[:40000] + "\n\nReport findings."


class ReportSection(BaseModel):
    heading: str
    body_markdown: str


class ReportDraft(BaseModel):
    title: str = ""
    executive_summary: str = ""
    sections: list[ReportSection] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class GridProposal(BaseModel):
    parameter: str = ""
    add: list[float] = Field(default_factory=list)
    drop: list[float] = Field(default_factory=list)
    rationale: str = ""
    corpus_signals: list[str] = Field(default_factory=list)


class ProposerAgent(Agent):
    """Proposes search-grid values from corpus characteristics, never from scores.

    Blind by construction: `ops.propose.assert_blind` refuses the payload if any
    score-shaped token reaches it, which is what makes the additions
    pre-registered rather than an adaptive search around a peak it was shown.
    """

    role = "proposer"
    prompt_name = "proposer"
    schema = GridProposal

    def build_user(self, *, parameter: str = "", corpus: str = "",
                   incumbent: str = "", limits: str = "", **kw: Any) -> str:
        return (
            f"## Parameter to propose for\n{parameter}\n\n"
            f"## The grid currently swept\n{incumbent}\n\n"
            f"## Legal range and limits\n{limits}\n\n"
            f"## What this corpus is like (no scores, by design)\n"
            f"{budget_text(corpus, 24000, tail=4000, label='corpus profile')}\n"
        )


class Observation(BaseModel):
    severity: str = "note"                 # blocking | warn | note
    claim: str = ""
    artifact_key: str = ""
    evidence: str = ""
    would_change: str = ""


class ObservationList(BaseModel):
    observations: list[Observation] = Field(default_factory=list)
    checked: list[str] = Field(default_factory=list)


class ObserverAgent(Agent):
    """Reads a phase's artifacts WHILE THE RUN IS STILL GOING.

    It decides nothing. Every observation must cite an artifact key that really
    exists, which `agents.observe.verified_observations` checks mechanically —
    the same discipline as the numeric check on authored prose, for the same
    reason: a claim that cannot be traced to an artifact is not evidence.
    """

    role = "observer"
    prompt_name = "observer"
    schema = ObservationList

    def build_user(self, *, phase: str = "", artifacts: str = "", decisions: str = "",
                   gates: str = "", **kw: Any) -> str:
        parts = [f"## Phase just finished\n{phase}\n",
                 "## Artifacts it produced\n"
                 f"{budget_text(artifacts, 60000, tail=8000, label='artifacts')}\n"]
        if decisions:
            parts.append(f"## Decisions it recorded\n{budget_text(decisions, 12000, tail=2000)}\n")
        if gates:
            parts.append(f"## Gates it evaluated\n{budget_text(gates, 8000, tail=1500)}\n")
        return "\n".join(parts)


class Interpretation(BaseModel):
    reading: str = ""
    caveats: list[str] = Field(default_factory=list)
    unavailable: list[str] = Field(default_factory=list)


class InterpreterAgent(Agent):
    """Explains ONE result. Every number it writes is checked against the sheet.

    Deliberately narrow. A whole-report author has to be trusted across thousands
    of words; an interpreter answers one question with a bounded fact sheet, and
    `agents.verify.check_numbers` can decide mechanically whether it stayed inside
    it. See `agents/interpret.py` for the rejection-and-retry loop.
    """

    role = "interpreter"
    prompt_name = "interpreter"
    schema = Interpretation

    def build_user(self, *, question: str = "", facts: str = "", context: str = "",
                   language: str = "zh", rejected: str = "", **kw: Any) -> str:
        parts = [
            f"## Report language\nWrite `reading` and `caveats` in: {language}\n",
            f"## The question\n{question}\n",
            "## Fact sheet — THE ONLY NUMBERS YOU MAY USE\n"
            f"{budget_text(facts, 24000, tail=4000, label='fact sheet')}\n",
        ]
        if context:
            parts.append("## Context from the run's artifacts\n"
                         f"{budget_text(context, 30000, tail=5000, label='context')}\n")
        if rejected:
            # External feedback naming the exact failure. Re-asking without it
            # is intrinsic self-correction, which degrades rather than improves.
            parts.append(
                "## YOUR PREVIOUS ANSWER WAS REJECTED\n"
                "These numbers are not in the fact sheet. Remove them, or replace "
                "them with a value that is in the sheet, or say the number is not "
                f"available:\n{rejected}\n")
        return "\n".join(parts)


class ReporterAgent(Agent):
    role = "reporter"
    prompt_name = "reporter"
    schema = ReportDraft

    def build_user(self, *, brief: str = "", evidence: str = "", **kw: Any) -> str:
        return (
            f"## What to write\n{brief}\n\n"
            f"## Evidence available (every number must come from here)\n"
            f"{budget_text(evidence, 60000, tail=6000)}"
        )


class DriftReport(BaseModel):
    new_families: list[dict[str, Any]] = Field(default_factory=list)
    disappeared_families: list[dict[str, Any]] = Field(default_factory=list)
    grown: list[dict[str, Any]] = Field(default_factory=list)
    shrunk: list[dict[str, Any]] = Field(default_factory=list)
    novel_queries: list[str] = Field(default_factory=list)
    alpha_recheck_needed: bool = False
    config_changed: bool = False
    recommended_actions: list[str] = Field(default_factory=list)
    summary: str = ""


class MaintainerAgent(Agent):
    role = "maintainer"
    prompt_name = "maintainer"
    schema = DriftReport

    def build_user(self, *, previous: str = "", current: str = "", novel: Sequence[str] = (), **kw: Any) -> str:
        return (
            f"## Previous run\n{budget_text(previous, 20000)}\n\n"
            f"## Current run\n{budget_text(current, 20000)}\n\n"
            f"## Queries far from every centroid\n" + "\n".join(f"- {q}" for q in novel[:60])
        )


ALL_ROLES = {
    "researcher": ResearcherAgent,
    "taxonomy_architect": ArchitectAgent,
    "taxonomy_critic": CriticAgent,
    "annotator": AnnotatorAgent,
    "referee": RefereeAgent,
    "adversary": AdversaryAgent,
    "namer": NamerAgent,
    "tree_auditor": AuditorAgent,
    "risk_sentinel": RiskSentinelAgent,
    "proposer": ProposerAgent,
    "observer": ObserverAgent,
    "interpreter": InterpreterAgent,
    "reporter": ReporterAgent,
    "maintainer": MaintainerAgent,
}


# ==========================================================================
# Phase 1 — domain scouting for an unknown vertical
# ==========================================================================

class CandidateSeed(BaseModel):
    name: str
    pattern: str = Field(description="Regex. Almost everything matching must share one intent.")
    intent_hint: str = ""


class CandidateRisk(BaseModel):
    name: str
    patterns: list[str] = Field(default_factory=list)
    rationale: str = ""


class DomainScoutReport(BaseModel):
    vertical: str = ""
    confidence: Literal["high", "medium", "low"] = "low"
    spans_multiple_verticals: bool = False
    verticals_present: list[str] = Field(default_factory=list)
    candidate_template_seeds: list[CandidateSeed] = Field(default_factory=list)
    candidate_risk_categories: list[CandidateRisk] = Field(default_factory=list)
    pragmatic_intent_hints: list[str] = Field(default_factory=list)
    notes: str = ""


class DomainScoutAgent(Agent):
    """Infers the vertical from a sample when no profile was supplied.

    Runs before the taxonomy researchers and produces hypotheses rather than
    conclusions — its seeds are statistically validated afterwards and dropped
    if they do not hold, and its vertical is only ever a steer.
    """

    role = "domain_scout"
    prompt_name = "domain_scout"
    schema = DomainScoutReport

    def build_user(
        self, *, sample: Sequence[str] = (), profile: dict[str, Any] | None = None, **kw: Any
    ) -> str:
        parts = []
        if profile:
            parts.append(
                "## What we already measured about this corpus\n"
                + json.dumps(profile, ensure_ascii=False, indent=1)[:3000]
            )
        parts.append(
            f"## Query sample ({len(sample)} rows, stratified)\n"
            + "\n".join(f"- {q}" for q in sample)
        )
        parts.append("Work out what this corpus is.")
        return "\n\n".join(parts)


ALL_ROLES["domain_scout"] = DomainScoutAgent
