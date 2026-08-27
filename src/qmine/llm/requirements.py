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

from typing import Any, ClassVar, Literal

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
    #: Approximate output tokens per call, for pre-run COST ESTIMATION.
    output_tokens_per_call: int = 1200
    #: Worst-case output for the GENERATION CAP, when it differs from the
    #: expected. One field cannot serve both: the cap must cover the noisiest
    #: model that could be routed here or the call truncates, while the estimate
    #: must reflect what is actually emitted or it over-charges the role — and
    #: the router weighs cost, so an inflated estimate silently changes which
    #: model gets picked. Setting both annotators to their peak did exactly that.
    peak_output_tokens_per_call: int | None = None
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
        return max(2000, (self.peak_output_tokens_per_call
                          or self.output_tokens_per_call) * 3)

    #: Output tokens per second, measured on this project's providers: the taxonomy
    #: architect emitted 20,441 tokens inside a 420s window on a live run.
    #: Deliberately conservative — the cost of underestimating it is a timeout and
    #: a full retry, the cost of overestimating is waiting longer for a hung call.
    THROUGHPUT_TOK_PER_SEC: ClassVar[float] = 40.0

    @property
    def timeout_seconds(self) -> float:
        """Request timeout, scaled to how much this role has to write.

        A two-step function (180s / 420s) served while every role's budget sat near
        its lower rung. It stopped working the moment the architect's budget was
        raised to cover what it actually emits: at ~49 tokens/sec measured, a
        42,000-token ceiling needs about 860 seconds, and 420 cut it off mid-answer
        so the run paid for a full retry down the plain-JSON repair path.

        Derived from the role's own generation cap instead, so raising a budget can
        never again leave a timeout that cannot accommodate it.
        """
        needed = self.max_output_tokens / self.THROUGHPUT_TOK_PER_SEC
        return float(min(1800.0, max(180.0, round(needed * 1.3))))

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
        min_context_tokens=128_000, typical_calls=5,
        # Measured on glm-5.2: this role emits up to 13,377 tokens, which
        # overran the previous 12,000 cap on every single call of a live run.
        output_tokens_per_call=6000,
        multilingual_critical=True,
        rationale="Reads thousands of raw queries and must notice what is NOT there. "
                  "Long context is the binding constraint; volume is trivial.",
    ),
    "taxonomy_architect": RoleRequirement(
        role="taxonomy_architect", reasoning="frontier", blast_radius="run",
        # Measured at 23,759 output tokens when this call also wrote the rules;
        # asking it for both blew a 42,000-token ceiling outright. Rules now live
        # in a separate call, so the budget covers the class list plus the
        # reasoning these models emit alongside it — which is most of the total.
        # Re-measured TWICE, rising each time on the same prompt: 23,759 on the
        # model this was first taken on, 38,073 on live36, then 39,647 on live38
        # — 94% of a 42,000 cap, i.e. it cleared by ~2,350 tokens. A budget set
        # to just clear the last observation is a budget that truncates on the
        # next one, so this is sized against the trend, not the maximum.
        min_context_tokens=200_000, typical_calls=2, output_tokens_per_call=18000,
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
        # The budget must cover the NOISIEST model that could be routed here, not
        # the quietest — reasoning tokens are billed and capped as output. That
        # principle was right and the number was four years behind it: measured
        # 1,439, then 11,910, and on live38 `deepseek-v4-flash` emitted **21,975**
        # per 25-row batch. At the declared 5,000 the whole-run projection came
        # out 4.4x under, which is what sized `max_total_output_tokens` below its
        # own honest requirement.
        # EXPECTED is the pair average; PEAK is what the noisy one emits. Exactly
        # one annotator draws a reasoning model and which one is a routing
        # decision made later, so pricing both at the peak inflated the pair by
        # ~2x and moved annotator_b to a different lab on a number that was not
        # real. Measured on live38: 21,975 and 1,792, averaging ~11,900.
        # min_context is the ROUTER'S ELIGIBILITY FILTER, so under-declaring it
        # selects models that cannot hold the prompt. 32,000 was measured before
        # the referee wrote rules: live38 assembles 7,028 chars of classes,
        # 46,814 of rules and 8,654 of binding rulings — ~63,000 chars, i.e.
        # 44,000-63,000 tokens of Chinese. Declared at 2x that so a role whose
        # rule set keeps growing does not silently outgrow its own model.
        min_context_tokens=128_000, typical_calls=240, output_tokens_per_call=12000,
        peak_output_tokens_per_call=22000,
        multilingual_critical=True,
        rationale="Thousands of rows, batched. Judgment per row is narrow and the guide "
                  "does the hard part. Highest-volume role — dominates the bill.",
    ),
    "annotator_b": RoleRequirement(
        role="annotator_b", reasoning="standard", blast_radius="contained",
        # Held EQUAL to annotator_a on purpose: the two do identical work and
        # which one draws the noisy model is a routing decision made later, so a
        # budget that fits only the quiet one truncates whichever gets the other.
        # (On live38 the routing inverted — a emitted 21,975, b emitted 1,792.)
        min_context_tokens=128_000, typical_calls=240, output_tokens_per_call=12000,
        peak_output_tokens_per_call=22000,
        multilingual_critical=True,
        rationale="Must be INDEPENDENT of annotator_a. Where two providers are available, "
                  "routing the two annotators to different model families makes their "
                  "agreement a stronger signal than shared-architecture agreement.",
    ),
    "referee": RoleRequirement(
        # `run`, not `phase`. The referee's verdicts BECOME the gold set: they
        # train the classifier in p2c, define kappa and every metric quoted from
        # it, and the rules it drafts ship in the guide the annotators read. A
        # phase-scoped blast radius set `cost_sensitivity` to 0.25, and on a
        # candidate set where price cannot rank capability that is enough to pick
        # the cheapest model clearing the bar.
        #
        # MEASURED, and this is the whole argument: on the SAME annotators, a
        # referee on `glm-5.2` chose annotator_a's label on 78.3% of contested
        # rows (z=+12.1) while one on `glm-4.5-airx` chose it on 55.1% (z=+2.2).
        # Near-chance adjudication is the signature of an adjudicator that cannot
        # discriminate — and live39's entire gold set rests on the second one.
        role="referee", reasoning="strong", blast_radius="run",
        # Measured at 8,179 adjudicating a 25-row batch with cited rules — on a
        # model long since replaced. On `live36`/`glm-5.2` the SUCCESSFUL calls
        # emitted 19,279-19,597 for the same 25-row batch, and 10 of 15 calls
        # died at exactly 24,001 tokens: the 12,000 cap, bumped once to 2x, and
        # unable to grow further. Truncated JSON does not parse, so each of those
        # silently discarded 25 adjudications. Set from what it actually writes.
        min_context_tokens=64_000, typical_calls=30, output_tokens_per_call=12000,
        multilingual_critical=True,
        rationale="Adjudicates exactly the cases two annotators disagreed on — by "
                  "construction the hardest rows in the set — and drafts the rules that "
                  "prevent the next disagreement.",
    ),
    "adversary": RoleRequirement(
        # `run`, not `contained` — the field contradicted the rationale below.
        # The adversary's output is the accuracy estimate the deliverable quotes
        # for the whole taxonomy, so an adversary that MISSES ships false
        # assurance about every label, not about one row. `contained` also kept
        # it outside the `capable_models` gate, which is the one place capability
        # is now stated rather than inferred from price.
        role="adversary", reasoning="strong", blast_radius="run",
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
    # TIER IS DECIDED BY HOW THE ROLE FAILS, NOT BY HOW IMPORTANT IT SOUNDS.
    # This file's own doctrine: "under-powering it fails silently while
    # over-powering it merely costs." The two new roles sit on opposite sides of
    # that line, so they get different tiers rather than both being maxed out.
    "proposer": RoleRequirement(
        role="proposer", reasoning="strong", blast_radius="contained",
        min_context_tokens=32_000, typical_calls=2, output_tokens_per_call=1200,
        multilingual_critical=True,
        rationale="Proposes grid values from corpus characteristics. NOT frontier: a "
                  "bad proposal simply loses in scoring, so it fails visibly and "
                  "cheaply. NOTE: the 'cannot win by luck' guarantee this budget once "
                  "cited is NOT currently in force — ops/select.py's challenger margin "
                  "has no production call site. The remaining protection is blindness "
                  "to scores (pre-registration), which is real and enforced.",
    ),
    "observer": RoleRequirement(
        role="observer", reasoning="frontier", blast_radius="run",
        # 11, not 4: observers were extended from the four bottom-up phases to
        # every phase that decides, including the whole top-down route. This
        # number drives the router's COST WEIGHTING, so leaving it stale makes
        # the role look cheaper than it is and can win it a pricier model.
        min_context_tokens=200_000, typical_calls=11, output_tokens_per_call=4000,
        multilingual_critical=True,
        rationale="Reads a whole phase's artifacts looking for a conclusion its own "
                  "numbers do not support — the hardest reasoning task here, and the "
                  "one that found nine shipped defects when a human did it by hand. "
                  "It cannot change anything, so its blast radius is not its own "
                  "output: it is the run-wide defect that ships when it MISSES. A "
                  "weak observer fails silently and, worse, supplies false assurance.",
    ),
    "delivery_auditor": RoleRequirement(
        role="delivery_auditor", reasoning="frontier", blast_radius="run",
        min_context_tokens=200_000, typical_calls=1, output_tokens_per_call=6000,
        multilingual_critical=True,
        rationale="The only role that may CHANGE a deliverable, and the last thing "
                  "between the run and a reader. It must hold every warning the run "
                  "raised against every document at once and decide which ones left a "
                  "defect — the same reasoning as the observer, over a larger surface. "
                  "Its edits are mechanically bounded (anchored, sourced, language- "
                  "checked), so it cannot ship a wrong number; what a weak model costs "
                  "here is the DEFECT IT FAILS TO SEE, which then ships with an audit "
                  "report attached saying the deliverables were checked.",
    ),
    "interpreter": RoleRequirement(
        role="interpreter", reasoning="strong", blast_radius="contained",
        min_context_tokens=64_000, typical_calls=6, output_tokens_per_call=1500,
        multilingual_critical=True,
        rationale="Writes one section's prose against a small fact sheet. Deliberately "
                  "NOT frontier: every number is checked mechanically, so a weaker "
                  "model fails VISIBLY — rejected, retried, or the section ships with "
                  "no commentary. Paying frontier rates here buys nicer sentences, not "
                  "safety, and it is called several times per report.",
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


#: Roles whose call volume is set by configuration rather than being a constant of
#: the method. Everything else really does run a fixed number of times.
def scaled_requirements(cfg: Any) -> dict[str, "RoleRequirement"]:
    """`ROLE_REQUIREMENTS` with the config-dependent volumes filled in.

    `typical_calls` is a constant per role, which makes the pre-run cost estimate
    a constant too — it read $2.99 whether the gold set was 600 rows or 3,000. That
    number exists to be consulted *before* spending, so being wrong by 5x defeats
    the only purpose it has.

    Annotation dominates the bill and scales with three knobs: the gold sample, the
    repair round that re-annotates it, and the pilot. Those are computed here; the
    rest are left alone.
    """
    from ..config import gold_size_for

    reqs = dict(ROLE_REQUIREMENTS)
    tax = cfg.taxonomy
    batch = 25

    gold = tax.gold_sample_size or gold_size_for(getattr(cfg.data, "sample_size", None)
                                                 or 50_000, tax)
    per_annotator = -(-gold // batch)
    # A repair round re-annotates a fresh sample of the same size.
    per_annotator *= (1 + max(0, tax.kappa_repair_rounds))
    # Round-2 active learning adds a fixed batch.
    per_annotator += -(-tax.active_learning_batch // batch)
    pilot = -(-tax.pilot_sample_size // batch)

    # Annotator A also runs the self-consistency pass on the pilot sample.
    reqs["annotator_a"] = reqs["annotator_a"].model_copy(
        update={"typical_calls": per_annotator + 2 * pilot})
    reqs["annotator_b"] = reqs["annotator_b"].model_copy(
        update={"typical_calls": per_annotator + pilot})
    # The referee sees the disagreements, historically ~16% of the gold set.
    reqs["referee"] = reqs["referee"].model_copy(
        update={"typical_calls": max(1, -(-int(gold * 0.16) // batch))})
    return reqs


def estimate_job(requirements: dict[str, RoleRequirement] | None = None) -> dict[str, int]:
    """Total calls and output tokens for a full run, before it starts.

    Used to price a run against the user's budget *in advance*, which is the
    only point at which the answer is still actionable.
    """
    reqs = requirements or ROLE_REQUIREMENTS
    calls = sum(r.typical_calls for r in reqs.values())
    out = sum(r.typical_calls * r.output_tokens_per_call for r in reqs.values())
    return {"total_calls": calls, "total_output_tokens": out}
