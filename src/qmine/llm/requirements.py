"""What each agent role actually needs from a model.

Routing well requires separating two things that a single "deep/fast" switch
conflates: how *hard* a role's judgment is, and how *much* of it there is. Those
vary independently across this team.

The annotator makes an easy judgment thousands of times; the referee makes a
hard one a few hundred times; the tree auditor makes one very hard judgment with
the entire tree in context. A tier system that offers two models cannot express
that, so it overpays for the annotator and underpowers the auditor
simultaneously.

Each role therefore declares *requirements* — minimum reasoning tier, context
floor, structured-output need, expected call volume, and how much a mistake
costs — and the router matches those against whatever models the user's API keys
actually reach. Requirements are properties of the task and stay stable as the
model landscape churns; only the matching changes.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

#: How demanding a role's judgment is, independent of how often it runs.
ReasoningTier = Literal["light", "standard", "strong", "frontier"]

#: What a wrong answer costs downstream. Drives how much we will pay per call.
BlastRadius = Literal["contained", "phase", "run"]


class RoleRequirement(BaseModel):
    """The contract a model must satisfy to serve one role."""

    role: str
    reasoning: ReasoningTier
    blast_radius: BlastRadius = Field(
        description="contained = one row/cluster; phase = one phase's output; "
                    "run = every downstream artifact inherits the error"
    )
    min_context_tokens: int = Field(
        default=32_000, description="Floor, from the largest payload this role assembles."
    )
    needs_structured_output: bool = True
    #: Rough call count on a 50k-row corpus. Drives the cost weighting.
    typical_calls: int = 1
    #: Approximate output tokens per call, for pre-run cost estimation.
    output_tokens_per_call: int = 1200
    multilingual_critical: bool = Field(
        default=False,
        description="Reads raw corpus text rather than our own English scaffolding.",
    )
    rationale: str = ""

    @property
    def max_output_tokens(self) -> int:
        """Generation cap for this role.

        Three times the declared per-call budget, so a role that legitimately
        runs long is not truncated, while a model that has fallen into a
        repetition loop is cut off instead of burning a full global cap. Roles
        that emit one short object get a floor of 2000 tokens so a verbose
        preamble cannot eat the whole allowance.
        """
        return max(2000, self.output_tokens_per_call * 3)

    @property
    def timeout_seconds(self) -> float:
        """Request timeout, scaled to how much this role has to write.

        One global timeout cannot serve both a 1,500-token annotation batch and a
        researcher composing 4,000 tokens of structured findings over a long
        evidence bundle. Measured on a live run: a researcher on a mid-tier model
        exceeded 180s and was retried three times, each retry as slow as the
        first — the request was not stuck, merely long.
        """
        return 180.0 if self.output_tokens_per_call <= 2000 else 420.0

    @property
    def cost_sensitivity(self) -> float:
        """How much price should weigh against capability, 0..1.

        High-volume roles dominate the bill, so a cheaper model that clears the
        capability bar is genuinely better for them. Low-volume, high-blast-radius
        roles are the opposite: their entire cost is a rounding error next to the
        cost of redoing the run they poisoned.
        """
        if self.blast_radius == "run":
            return 0.05
        if self.typical_calls >= 200:
            return 0.85
        if self.typical_calls >= 40:
            return 0.55
        return 0.25


#: The roster. Volumes assume a 50k-row corpus with ~60 leaves and a 3k gold set.
ROLE_REQUIREMENTS: dict[str, RoleRequirement] = {
    "researcher": RoleRequirement(
        role="researcher", reasoning="strong", blast_radius="phase",
        min_context_tokens=128_000, typical_calls=5, output_tokens_per_call=4000,
        multilingual_critical=True,
        rationale="Reads thousands of raw queries and must notice what is NOT there. "
                  "Long context is the binding constraint; volume is trivial.",
    ),
    "taxonomy_architect": RoleRequirement(
        role="taxonomy_architect", reasoning="frontier", blast_radius="run",
        min_context_tokens=200_000, typical_calls=1, output_tokens_per_call=8000,
        multilingual_critical=True,
        rationale="Synthesises every researcher's evidence into the taxonomy that gold "
                  "labels, the classifier, and every report inherit. One call, unbounded stakes.",
    ),
    "taxonomy_critic": RoleRequirement(
        role="taxonomy_critic", reasoning="frontier", blast_radius="run",
        min_context_tokens=128_000, typical_calls=1, output_tokens_per_call=4000,
        multilingual_critical=True,
        rationale="Adversarial review only pays if the critic is at least as capable as "
                  "the architect. A weaker critic rubber-stamps.",
    ),
    "annotator_a": RoleRequirement(
        role="annotator_a", reasoning="standard", blast_radius="contained",
        min_context_tokens=32_000, typical_calls=240, output_tokens_per_call=1500,
        multilingual_critical=True,
        rationale="Thousands of rows, batched. Judgment per row is narrow and the guide "
                  "does the hard part. Highest-volume role — dominates the bill.",
    ),
    "annotator_b": RoleRequirement(
        role="annotator_b", reasoning="standard", blast_radius="contained",
        min_context_tokens=32_000, typical_calls=240, output_tokens_per_call=1500,
        multilingual_critical=True,
        rationale="Must be INDEPENDENT of annotator_a. Where two providers are available, "
                  "routing the two annotators to different model families makes their "
                  "agreement a stronger signal than shared-architecture agreement.",
    ),
    "referee": RoleRequirement(
        role="referee", reasoning="strong", blast_radius="phase",
        min_context_tokens=64_000, typical_calls=30, output_tokens_per_call=3000,
        multilingual_critical=True,
        rationale="Adjudicates exactly the cases two annotators disagreed on — by "
                  "construction the hardest rows in the set — and drafts the rules that "
                  "prevent the next disagreement.",
    ),
    "adversary": RoleRequirement(
        role="adversary", reasoning="strong", blast_radius="contained",
        min_context_tokens=32_000, typical_calls=10, output_tokens_per_call=2500,
        multilingual_critical=True,
        rationale="Its job is to find errors a weaker model would agree with. Capability "
                  "is the entire point; a cheap adversary produces a flattering number.",
    ),
    "namer": RoleRequirement(
        role="namer", reasoning="standard", blast_radius="phase",
        min_context_tokens=32_000, typical_calls=60, output_tokens_per_call=1200,
        multilingual_critical=True,
        rationale="One call per cluster, reading 30 member queries. Moderate volume; the "
                  "names reach the catalogue and the report, so not the cheapest tier.",
    ),
    "tree_auditor": RoleRequirement(
        role="tree_auditor", reasoning="frontier", blast_radius="run",
        min_context_tokens=200_000, typical_calls=1, output_tokens_per_call=8000,
        multilingual_critical=True,
        rationale="Holds every cluster naming at once and must spot cross-family twins by "
                  "comparing definition sentences. Long context AND hard reasoning, once.",
    ),
    "risk_sentinel": RoleRequirement(
        role="risk_sentinel", reasoning="strong", blast_radius="run",
        min_context_tokens=128_000, typical_calls=1, output_tokens_per_call=4000,
        multilingual_critical=True,
        rationale="A missed risk category is a production incident, not a metric "
                  "regression. Never route this to the cheap tier to save a few cents.",
    ),
    "reporter": RoleRequirement(
        role="reporter", reasoning="strong", blast_radius="phase",
        min_context_tokens=200_000, typical_calls=3, output_tokens_per_call=8000,
        rationale="Assembles the whole evidence bundle. Long context; must not invent "
                  "numbers, which is a reasoning property as much as a prompting one.",
    ),
    "maintainer": RoleRequirement(
        role="maintainer", reasoning="strong", blast_radius="phase",
        min_context_tokens=64_000, typical_calls=1, output_tokens_per_call=3000,
        rationale="Distinguishes content drift from method change across two runs — a "
                  "subtle judgment that a weak model gets confidently wrong.",
    ),
    "domain_scout": RoleRequirement(
        role="domain_scout", reasoning="strong", blast_radius="run",
        min_context_tokens=128_000, typical_calls=1, output_tokens_per_call=4000,
        multilingual_critical=True,
        rationale="Runs before anything else knows what the corpus is. A confidently wrong "
                  "vertical propagates into every later phase without being re-checked, so "
                  "this is cheap to run and expensive to get wrong.",
    ),
    "l2_interpreter": RoleRequirement(
        role="l2_interpreter", reasoning="standard", blast_radius="contained",
        min_context_tokens=32_000, typical_calls=20, output_tokens_per_call=1200,
        multilingual_critical=True,
        rationale="Reads sub-cluster samples and proposes a sub-intent name. Same shape "
                  "as naming, lower stakes.",
    ),
}

#: Ordering used when comparing tiers.
TIER_ORDER: dict[str, int] = {"light": 0, "standard": 1, "strong": 2, "frontier": 3}


def requirement_for(role: str) -> RoleRequirement:
    """Requirements for a role, falling back to a safe middle for unknown roles.

    The fallback is ``standard`` rather than ``light``: an unrecognised role is
    more likely to be a new agent someone forgot to register than a trivial one,
    and under-powering it fails silently while over-powering it merely costs.
    """
    base = role.split("_")[0] if role not in ROLE_REQUIREMENTS else role
    if role in ROLE_REQUIREMENTS:
        return ROLE_REQUIREMENTS[role]
    for known in ROLE_REQUIREMENTS:
        if role.startswith(known) or known.startswith(base):
            return ROLE_REQUIREMENTS[known].model_copy(update={"role": role})
    return RoleRequirement(
        role=role, reasoning="standard", blast_radius="phase",
        # Same reasoning as the tier: a truncated answer from an unregistered
        # role surfaces as a parse error three frames away, while an unused
        # allowance costs nothing.
        output_tokens_per_call=4000,
        rationale="unregistered role — defaulting to standard, which fails visibly "
                  "rather than silently",
    )


def estimate_job(requirements: dict[str, RoleRequirement] | None = None) -> dict[str, int]:
    """Total calls and output tokens for a full run, before it starts.

    Used to price a run against the user's budget *in advance*, which is the
    only point at which the answer is still actionable.
    """
    reqs = requirements or ROLE_REQUIREMENTS
    calls = sum(r.typical_calls for r in reqs.values())
    out = sum(r.typical_calls * r.output_tokens_per_call for r in reqs.values())
    return {"total_calls": calls, "total_output_tokens": out}
