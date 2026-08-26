"""Choosing a model per role from whatever the user's keys can reach.

The routing problem here is narrow enough to solve well: thirteen roles with
stable, declared requirements, and a catalogue of a few thousand models with
prices and capability flags. What makes it tractable is that the *requirements*
do not churn even though the models do.

**How capability is estimated, and why that estimate is honest about itself.**
There is no free, current, per-model benchmark covering every provider. What
exists is price — and within a generation, price tracks capability closely,
because labs price against each other on quality. So models are assigned a tier
by their price percentile *within the set the user can actually reach*, which is
robust to the whole market getting cheaper over time.

That proxy has two known failure modes and both are handled rather than ignored:
a genuinely cheap-but-strong new model looks weak (mitigated by
``prefer`` overrides and by never routing a ``run``-blast-radius role below the
top tier), and an old expensive model looks strong (mitigated by excluding
anything with a `deprecation_date` in the past). Where a user knows better, an
explicit preference wins outright — the router is a default, not an authority.

**Fallbacks span providers.** A chain of three models from one provider is not a
fallback chain; it is one outage. Alternatives are chosen from different
providers where the catalogue allows.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

from .catalog import Catalog, ModelCard
from .providers import CHINESE_NATIVE
from .requirements import TIER_ORDER, RoleRequirement, requirement_for

log = logging.getLogger("qmine.router")

#: Roles that must not share a model with another role, and who they must differ
#: from. Two annotators labelling the same rows are a *measurement*: Cohen's
#: kappa between them is only evidence about the labelling guide if their errors
#: are independent. Give both the same model and shared architecture makes them
#: agree on the same mistakes, inflating kappa and hiding exactly the guide
#: ambiguity the gold set exists to find.
#: A role must not share a lab with any of these. The referee decides every row
#: the annotators split on, so sharing a lab with either would make it side
#: systematically with that one — corrupting the gold set the whole pipeline
#: rests on, silently and in a direction nobody would look for.
MUST_DIFFER_FROM: dict[str, tuple[str, ...]] = {
    "annotator_b": ("annotator_a",),
    "referee": ("annotator_a", "annotator_b"),
}

#: Model-id fragments that identify the LAB a model comes from, regardless of
#: which gateway serves it. An aggregator makes `provider` useless as a proxy for
#: independence: `zhipu/zai/glm-5.1` and `openrouter/z-ai/glm-5.3` are different
#: providers and the same lab, so two annotators there would agree on the same
#: mistakes and inflate kappa exactly where the gold set is meant to expose the
#: guide. Order matters — longest, most specific fragment first.
_LAB_MARKERS: tuple[tuple[str, str], ...] = (
    ("deepseek", "deepseek"), ("z-ai/", "zhipu"), ("zai/", "zhipu"), ("glm", "zhipu"),
    ("qwen", "qwen"), ("dashscope/", "qwen"), ("moonshot", "moonshot"),
    ("openai/", "openai"), ("anthropic/", "anthropic"), ("google/", "google"),
    ("kimi", "moonshot"), ("claude", "anthropic"), ("gpt-", "openai"),
    ("o1", "openai"), ("o3", "openai"), ("gemini", "google"), ("llama", "meta"),
    ("mistral", "mistral"), ("minimax", "minimax"), ("baichuan", "baichuan"),
    ("yi-", "01ai"), ("ernie", "baidu"), ("hunyuan", "tencent"), ("doubao", "bytedance"),
)


#: API gateways, which are never a lab.
_GATEWAYS = frozenset({"openrouter", "dashscope", "volcengine", "siliconflow"})


def bare_model(card_or_id: Any) -> str:
    """The model's own name, with the gateway's namespacing removed.

    `deepseek/deepseek-v4-flash` served by OpenRouter and `deepseek-v4-flash`
    served by DeepSeek are the same model reached two ways, and the router had no
    way to notice that.
    """
    mid = str(getattr(card_or_id, "api_id", "") or getattr(card_or_id, "id", card_or_id))
    return mid.split("/")[-1].lower()


def lab_of(card_or_id: Any) -> str:
    """Which lab trained this model, not which gateway sells it.

    Independence between two annotators is a property of the models, not of the
    billing relationship. Falls back to the provider when nothing matches, which
    is the old behaviour and safe for direct providers.
    """
    mid = getattr(card_or_id, "id", card_or_id)
    low = str(mid).lower()
    for fragment, lab in _LAB_MARKERS:
        if fragment in low:
            return lab
    # No marker. Aggregators namespace ids as `vendor/model`, so the vendor
    # segment is the lab. Falling back to `provider` here returned "openrouter"
    # for every unrecognised model — which made them indistinguishable from each
    # other AND unexcludable by name, since the caller has no way to say "not
    # that vendor" when every vendor reports as the gateway.
    parts = [seg for seg in low.split("/") if seg and seg not in _GATEWAYS]
    if len(parts) > 1:
        return parts[0]
    return str(getattr(card_or_id, "provider", "") or mid)


#: Price percentile boundaries used to assign capability tiers, computed over the
#: reachable set rather than over absolute dollars.
TIER_PERCENTILES: list[tuple[str, float]] = [
    ("light", 0.35), ("standard", 0.65), ("strong", 0.88), ("frontier", 1.01),
]


@dataclass
class Assignment:
    """The routing decision for one role, and the reasoning behind it."""

    role: str
    model: str
    provider: str
    tier: str
    #: The id the provider's own API accepts, which differs from the catalogue key.
    api_model: str = ""
    #: The model's own generation ceiling, so the caller never asks for more than
    #: it will emit. ``None`` when the catalogue does not publish one.
    max_output_tokens: int | None = None
    #: What this model charges, carried through so spend is measured at the same
    #: prices that chose it. The ledger used to bill every token at a hardcoded
    #: frontier rate and overstated a live run by 11x.
    input_per_mtok: float | None = None
    output_per_mtok: float | None = None
    fallbacks: list[str] = field(default_factory=list)
    estimated_cost_usd: float = 0.0
    estimated_calls: int = 0
    why: str = ""
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": self.role, "model": self.model, "api_model": self.api_model,
            "input_per_mtok": self.input_per_mtok, "output_per_mtok": self.output_per_mtok,
            "provider": self.provider,
            "tier": self.tier, "fallbacks": self.fallbacks,
            "estimated_cost_usd": round(self.estimated_cost_usd, 4),
            "estimated_calls": self.estimated_calls,
            "why": self.why, "warnings": self.warnings,
        }


@dataclass
class RoutingPlan:
    assignments: dict[str, Assignment] = field(default_factory=dict)
    total_cost_usd: float = 0.0
    catalog_provenance: dict[str, Any] = field(default_factory=dict)
    providers_used: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "assignments": {k: v.as_dict() for k, v in self.assignments.items()},
            "total_estimated_cost_usd": round(self.total_cost_usd, 2),
            "providers_used": self.providers_used,
            "catalog": self.catalog_provenance,
            "notes": self.notes,
        }


def _generation(model_id: str) -> float:
    """The largest version number in a model id, as a recency proxy.

    `glm-5.1` -> 5.1, `glm-4.5-airx` -> 4.5, `qwen3-next-80b` -> 80 would be
    wrong, so parameter-count suffixes are stripped first. Crude, and only ever
    consulted to break an exact tie — never to override a real score difference.
    """
    import re as _re

    low = str(model_id).lower()
    # Date stamps first. `qwen-flash-2025-07-28` read as generation 28 and became
    # the top-ranked model of its lab, then won every role in the pipeline.
    low = _re.sub(r"\d{4}-\d{2}-\d{2}", " ", low)
    low = _re.sub(r"\b\d{6,8}\b", " ", low)
    low = _re.sub(r"\b20\d\d\b", " ", low)
    # Then parameter counts — `qwen3-next-80b-a3b` must read as 3, not 80.
    low = _re.sub(r"\d+(?:\.\d+)?\s*[bkm]\b", " ", low)
    nums = _re.findall(r"\d+(?:\.\d+)?", low)
    # A version number is small. Anything larger is a date, a size or an id.
    return max((float(n) for n in nums if float(n) < 30), default=0.0)


def _assign_tiers(cards: Sequence[ModelCard]) -> dict[str, str]:
    """Bucket models into capability tiers by price percentile.

    Price is a poor capability signal and we know exactly how it fails: a newer,
    roomier model priced BELOW an older one reads as less capable — that is how
    the referee kept `zai/glm-4.5-airx` (128k context) over `z-ai/glm-5.2` (1M
    context) for being $1.46/M dearer. The fix for that is the tie-break in
    `route`, which now rounds the score so a 1e-15 cost difference cannot outvote
    a whole model generation.

    Replacing the percentile with a price-free capability score was tried and
    reverted, three times, because price was quietly doing two other jobs as well:
    keeping $0.00 and unpriced endpoints — free tiers, previews, `openrouter/auto`
    — out of the roles that matter. Strip price out of the ranking and those win
    everything, because a cost-weighted score cannot resist a free model. Those
    exclusions are now explicit in `_eligible`, but the ordering stays priced
    until there is a capability signal worth trusting; the catalogue publishes
    none, and inventing one produced worse routing than the flawed proxy.
    """
    priced = sorted([c for c in cards if c.priced],
                    key=lambda c: (c.input_per_mtok or 0) + 3 * (c.output_per_mtok or 0))
    n = len(priced)
    tiers: dict[str, str] = {}
    for i, c in enumerate(priced):
        pct = (i + 1) / max(n, 1)
        for name, bound in TIER_PERCENTILES:
            if pct <= bound:
                tiers[c.id] = name
                break
    return tiers


def _eligible(card: ModelCard, req: RoleRequirement, tier: str) -> tuple[bool, str]:
    """Hard constraints. A model that fails any of these is not a candidate."""
    if req.needs_structured_output and card.supports_structured_output is False:
        return False, "no structured output"
    if card.context_tokens and card.context_tokens < req.min_context_tokens:
        return False, f"context {card.context_tokens} < required {req.min_context_tokens}"
    if card.deprecated_on:
        import datetime as _dt

        try:
            if _dt.date.fromisoformat(card.deprecated_on) <= _dt.date.today():
                return False, f"deprecated {card.deprecated_on}"
        except ValueError:
            pass
    if card.emits_non_text:
        return False, "emits images; tuned for a different job than text reasoning"
    # Asynchronous variants. A `:batch` endpoint trades latency for price and is
    # answered on the provider's schedule, not the caller's — this pipeline waits
    # on every call, so one would hang a phase rather than serve it. They are
    # cheap, which is exactly why the cost-weighted score reaches for them: the
    # router picked `openai/gpt-5.1:batch` for the referee and five other roles.
    if ":batch" in card.id or card.id.endswith(":async"):
        return False, "asynchronous batch endpoint; this pipeline calls synchronously"
    # Free and preview tiers. A run makes ~500 calls over two hours; free tiers are
    # aggressively rate-limited and previews are withdrawn without notice. They
    # also cost nothing, so a cost-weighted score reaches for them first — with
    # price removed from the capability estimate, `:free` variants won every role
    # in the pipeline including the architect.
    if ":free" in card.id or "preview" in card.id or "-exp" in card.id:
        return False, "free/preview tier: rate-limited and impermanent, unfit for a long run"
    # A model with no price, or a price of zero, cannot be reasoned about and is
    # almost always a free tier, a preview, or a meta-endpoint like
    # `openrouter/auto` that resolves to something else at call time. A
    # cost-weighted score ranks all of them first.
    if not card.priced or (card.output_per_mtok or 0) <= 0:
        return False, "no published price: free tier, preview, or a meta-endpoint"
    # The price-derived tier is a HARD gate only where an error is unrecoverable.
    # Everywhere else it is a scoring term, because the proxy's known failure is
    # precisely a strong-and-cheap model: DeepSeek at $0.28/$1M with 131k context
    # and structured output lands in a low price band and would be excluded from
    # the very high-volume roles it suits best. For a `run` blast radius we keep
    # the gate — being wrong there costs the whole run, so we pay for certainty.
    if req.blast_radius == "run" and TIER_ORDER.get(tier, 0) < TIER_ORDER.get(req.reasoning, 1):
        return False, f"tier {tier} below required {req.reasoning} for a run-critical role"
    return True, ""


def _pin_warnings(role: str, chosen_id: str, card: Any, req: Any,
                  fallbacks: list[str]) -> list[str]:
    """What an operator must know about an explicitly pinned model.

    A pin is an escape hatch that bypasses `_eligible` entirely, so the checks
    that protect a routed choice do not run. Two of them matter enough to say out
    loud, and both were found on real pins:

    * **No published price.** `_eligible` refuses an unpriced card outright, as a
      likely free tier, preview or meta-endpoint. A pin overrides that, and the
      cost then reads $0.00 for however many calls the role makes while the spend
      ledger — which prices from the same card — under-reports. `qwen3.7-plus` is
      a genuine 991k-context direct model the catalogue simply carries no rate
      for, so the override is reasonable; it still must not read as free.
    * **No fallback.** A pin says which model to prefer, not that the run should
      end when that provider has an outage. `observer` and `delivery_auditor`
      need 200k of context and nothing reachable in another lab provides it, so
      both would vanish on a single gateway outage — quietly, because each wraps
      its own agent call and degrades rather than failing.
    """
    out: list[str] = []
    if not card or card.input_per_mtok is None:
        out.append(f"{chosen_id} publishes NO PRICE in the catalogue — the pin overrides "
                   "`_eligible`'s exclusion of unpriced models. Its estimate reads $0.00 "
                   "and the spend ledger will under-report: the cost is UNKNOWN, not free.")
    if not fallbacks:
        out.append(f"pinned with NO FALLBACK — nothing reachable meets this role's "
                   f"{req.min_context_tokens:,}-token context need in another lab, so a "
                   "single provider outage silently removes this role from the run")
    return out


def _pin_fallbacks(
    role: str, chosen_id: str, cards: Any, tiers: Any, req: Any,
    by_id: Any, assignments: Any,
) -> list[str]:
    """Lab-diverse alternates for an explicitly pinned model.

    The pin says which model to use; it does not say the run should end when
    that provider has an outage. Diversifies by LAB rather than gateway, and
    skips the labs already used by partner roles, because failing over into a
    partner's lab would quietly destroy the independence kappa depends on.
    """
    out: list[str] = []
    seen = {lab_of(by_id.get(chosen_id) or chosen_id)}
    # BOTH DIRECTIONS. `MUST_DIFFER_FROM` is declared one way — the referee must
    # differ from the annotators — but the property is symmetric: if annotator_b
    # fails over INTO the referee's lab, the referee is no longer independent of
    # annotator_b, and the kappa that independence underwrites becomes
    # shared-architecture agreement. Observed: a pinned annotator_b was handed a
    # zhipu fallback while the referee ran on zhipu.
    related = set(MUST_DIFFER_FROM.get(role, ()))
    related |= {other for other, must in MUST_DIFFER_FROM.items() if role in must}
    for pname in related:
        a = assignments.get(pname)
        if a is not None and a.model:
            seen.add(lab_of(by_id.get(a.model) or a.model))
    for c in cards:
        ok, _ = _eligible(c, req, tiers.get(c.id, "light"))
        if not ok or c.id == chosen_id or lab_of(c) in seen:
            continue
        out.append(f"{c.provider}:{c.id}")
        seen.add(lab_of(c))
        if len(out) >= 2:
            break
    return out


def route(
    catalog: Catalog,
    available_providers: Sequence[str],
    *,
    roles: Sequence[str] | None = None,
    prefer: dict[str, str] | None = None,
    budget_usd: float | None = None,
    prefer_chinese_native: bool = False,
    #: Accepted and ignored: a "same model, fewer hops" promotion already runs
    #: after ranking, scoped to the SAME BARE MODEL. A scoring bonus on top of it
    #: is not the same rule — it biases toward models whose lab happens to be
    #: their own provider, which silently swapped `l2_interpreter` from a
    #: deepseek model to a qwen one. Kept as a parameter so callers that pass it
    #: do not break.
    prefer_direct: bool = True,
    excluded_labs: Sequence[str] = (),
    #: Bare model ids a human has judged capable. THE MISSING SIGNAL.
    #:
    #: `TIER_PERCENTILES` derives capability from PRICE PERCENTILE, and `_score`
    #: caps capability credit at what the role requires — so among candidates at
    #: or above the bar the cheapest always wins. That is sound only while price
    #: tracks capability, and across Chinese labs it does not: after excluding
    #: the Western labs not one Chinese model rates `frontier`, and
    #: `deepseek-v4-pro`, `glm-5.2`, `qwen3.8-max` and `kimi-k3` all land in one
    #: `strong` band. The referee was handed `glm-4.5-airx` over `glm-5.2` on
    #: $0.30/M of input, and adjudicated at near chance as a result.
    #:
    #: Removing price from the formula was tried and made things worse — price is
    #: the only thing keeping a 260-call role off an expensive model. So price
    #: stays, and capability is supplied EXPLICITLY instead of inferred. This
    #: list gates the candidate set for roles whose errors are expensive; price
    #: then only breaks ties INSIDE it.
    #:
    #: Cheaper to maintain than pinning every role, and it degrades: if none of
    #: these is reachable for a role, the old behaviour returns with a warning
    #: rather than leaving the role unserved.
    capable_models: Sequence[str] = (),
    avg_input_tokens: int = 6000,
    #: Config-scaled call volumes. Without it the estimate is a constant and reads
    #: the same whether the gold set is 600 rows or 3,000 — a number consulted
    #: before spending has to move when the spending does.
    requirements: dict[str, Any] | None = None,
) -> RoutingPlan:
    """Pick a model per role, with fallbacks and a cost estimate.

    ``prefer`` is an escape hatch that wins outright: a user who knows a specific
    model is right for a role should not have to argue with a heuristic.
    """
    from .requirements import ROLE_REQUIREMENTS

    reqs = requirements or ROLE_REQUIREMENTS
    role_list = list(roles or reqs.keys())
    # Partner-constrained roles are routed after the role they must differ from.
    role_list.sort(key=lambda r: (len(MUST_DIFFER_FROM.get(r, ())), r))
    cards = catalog.for_providers(available_providers)
    if excluded_labs:
        banned = {lab.lower() for lab in excluded_labs}
        before = len(cards)
        cards = [c for c in cards if lab_of(c) not in banned]
        plan_note = (f"excluded {before - len(cards)} models from "
                     f"{sorted(banned)} — by LAB, so an aggregator cannot smuggle "
                     "them back in under a different provider name")
    else:
        plan_note = ""
    plan = RoutingPlan(catalog_provenance=catalog.provenance(),
                       providers_used=sorted(set(available_providers)))
    if plan_note:
        plan.notes.append(plan_note)

    if not cards:
        plan.notes.append(
            "No models in the catalogue for the configured providers. Routing falls back to "
            "the statically configured deep/fast models — which is a working default, not a "
            "failure, but nothing here is optimised."
        )
        return plan
    if catalog.degraded:
        plan.notes.append(f"catalogue degraded: {catalog.degraded}")

    # Tiers are computed over the WHOLE catalogue, not the reachable subset.
    # Percentiles over a handful of models are meaningless — with one provider
    # configured, its best model lands mid-table and every frontier role goes
    # unserved. The market-wide distribution is the stable reference.
    tiers = _assign_tiers(list(catalog.models.values()))
    by_id = {c.id: c for c in cards}

    for role in role_list:
        req = reqs.get(role) or requirement_for(role)

        if prefer and role in prefer:
            chosen_id = prefer[role]
            card = by_id.get(chosen_id)
            _pin_fb = _pin_fallbacks(role, chosen_id, cards, tiers, req,
                                     by_id, plan.assignments)
            plan.assignments[role] = Assignment(
                role=role, model=chosen_id,
                # Carry the API id. Without it an explicitly-preferred model is
                # called by its CATALOGUE id — `zai/glm-4.5-airx` where the
                # endpoint wants `glm-4.5-airx` — which 404s, and which also
                # changes the LLM cache key, so pinning a run to its own previous
                # models would silently replay nothing.
                api_model=card.api_id if card else None,
                max_output_tokens=card.max_output_tokens if card else None,
                input_per_mtok=card.input_per_mtok if card else None,
                output_per_mtok=card.output_per_mtok if card else None,
                provider=card.provider if card else "explicit",
                tier=tiers.get(chosen_id, "explicit"),
                estimated_calls=req.typical_calls,
                estimated_cost_usd=(card.blended_cost(avg_input_tokens, req.output_tokens_per_call) or 0)
                                   * req.typical_calls if card else 0.0,
                # A PIN STILL NEEDS FAILOVER. This branch returned with an empty
                # `fallbacks`, so pinning a model meant "use this one, and abandon
                # the run if its provider is unavailable" — on a 256-call role,
                # against a project that has already lost twelve gold batches to a
                # single 402. Pinning expresses which model to prefer, not a
                # willingness to have no alternative. Same lab-diversity rule as
                # the ranked path: a chain within one lab is one outage.
                fallbacks=_pin_fb,
                why="explicitly preferred by the user; the router did not second-guess it",
                # A MODEL WITH NO PUBLISHED PRICE ESTIMATES AS FREE.
                #
                # `blended_cost` returns None when the card carries no rate, and
                # `or 0` turns that into a confident zero — on `annotator_b`, the
                # highest-volume role in the pipeline, that is ~256 calls reading
                # as $0.00. The spend ledger prices from the same card, so the
                # run under-reports too. Same family as the pinned-roles bug
                # above: the largest line item is the one that goes missing.
                # A PIN BYPASSES `_eligible`, INCLUDING ITS PRICE EXCLUSION.
                #
                # `_eligible` refuses an unpriced card outright — "free tier,
                # preview, or a meta-endpoint" — so the ranked path cannot pick
                # one. Pinning overrides that deliberate exclusion, and the cost
                # then reads $0.00 for however many calls the role makes while
                # the spend ledger, which prices from the same card, silently
                # under-reports. Real case: `qwen3.7-plus` is a genuine direct
                # model with 991k context that the catalogue simply carries no
                # rate for — a catalogue gap, not a meta-endpoint, so the
                # override is reasonable. It still must not read as free.
                warnings=_pin_warnings(role, chosen_id, card, req, _pin_fb),
            )
            # A PINNED ROLE STILL COSTS MONEY.
            #
            # This branch `continue`d straight past the accumulator below, so
            # every pinned role was excluded from the estimated total — and the
            # pins are on the highest-VOLUME roles precisely because those are
            # the ones worth choosing deliberately. With both annotators pinned
            # the header read "$0.79 per full run" against a table showing $36 of
            # annotators on the same screen. An estimate that omits the largest
            # line items is worse than no estimate: it is used to decide whether
            # a model upgrade is affordable.
            plan.total_cost_usd += plan.assignments[role].estimated_cost_usd
            continue

        # THE CAPABILITY GATE. Applied only where a wrong choice is expensive —
        # a 260-call `contained` role is exactly where cost weighting belongs.
        pool, capability_gated = cards, False
        if capable_models and req.blast_radius in ("run", "phase"):
            allow = {str(m).strip().lower() for m in capable_models if str(m).strip()}
            keep = [c for c in cards
                    if c.id.lower() in allow or c.id.lower().split("/")[-1] in allow]
            # HARD constraints only — context, structured output, deprecation.
            # Checking the derived TIER here would defeat the purpose: a role
            # asking for `frontier` finds no Chinese model rated that (they are
            # all `strong` on price), the gate would find nothing eligible, and
            # the role would fall back to exactly the price-decided choice this
            # list exists to replace. The capability list OVERRIDES the price
            # tier; it must not be gated by it. Tier relaxation is still
            # reported below, now against a capable candidate set.
            if any(_eligible(c, req, "frontier")[0] for c in keep):
                pool, capability_gated = keep, True

        scored: list[tuple[float, ModelCard, str]] = []
        relaxed: list[tuple[float, ModelCard, str]] = []
        for c in pool:
            tier = tiers.get(c.id, "light")
            ok, _why = _eligible(c, req, tier)
            cost = c.blended_cost(avg_input_tokens, req.output_tokens_per_call) or 0.0
            if ok:
                scored.append((cost, c, tier))
            elif _eligible(c, req, "frontier")[0]:
                # Meets every HARD requirement and only missed the price-derived
                # tier. Kept as a relaxed candidate so a run-critical role is
                # never left unserved — which halts the pipeline entirely.
                relaxed.append((cost, c, tier))

        tier_relaxed = False
        if not scored and relaxed:
            scored, tier_relaxed = relaxed, True

        if not scored:
            plan.assignments[role] = Assignment(
                role=role, model="", provider="", tier="",
                estimated_calls=req.typical_calls,
                why="no reachable model met this role's requirements",
                warnings=[
                    f"needs {req.reasoning} reasoning, >={req.min_context_tokens} context"
                    + (", structured output" if req.needs_structured_output else "")
                    + ". Add a provider key, or lower the requirement knowingly."
                ],
            )
            continue

        max_cost = max(c for c, _, _ in scored) or 1.0
        w_cost = req.cost_sensitivity

        def _score(item: tuple[float, ModelCard, str]) -> float:
            cost, card, tier = item
            # Capability credit is capped at what the role actually needs. Beyond
            # that, extra tier is money spent on headroom nobody uses — which is
            # how a frontier model ends up doing five thousand annotation calls.
            need = TIER_ORDER.get(req.reasoning, 1)
            cap = min(TIER_ORDER.get(tier, 0), need) / max(need, 1)
            cheap = 1.0 - (cost / max_cost)
            bonus = 0.0
            if prefer_chinese_native and card.provider in CHINESE_NATIVE and req.multilingual_critical:
                bonus += 0.08
            return (1 - w_cost) * cap + w_cost * cheap + bonus

        def _tiebreak(item: tuple[float, ModelCard, str]) -> tuple[float, float, float]:
            """Score first; then break ties toward newer and roomier.

            Measured: `zai/glm-5.1` (1.40/4.40, 200k context) and
            `zai/glm-4.5-airx` (1.10/4.50, 128k) score **identically** on the cost
            proxy — 14.60 each. The referee got the older, smaller-context model
            because it came first in the catalogue. Price is the only capability
            signal the router has, so when price cannot separate two candidates
            something else must, and "newer generation, more context" is the
            least-bad available proxy.
            """
            _cost, card, _tier = item
            # ROUND the score. Two candidates differing by 1e-16 of floating-point
            # noise are tied in every sense that matters, and comparing raw floats
            # meant the tie-break below was unreachable: the referee kept an older
            # 128k model over a newer 1M-context one on a 1.78e-15 cost difference.
            return (round(_score(item), 6), _generation(card.id),
                    float(card.context_tokens or 0))

        scored.sort(key=_tiebreak, reverse=True)

        # Within ONE lab, a newer model holding at least as much context is not
        # less capable than an older one, whatever it costs. The price-derived
        # tier says otherwise: `z-ai/glm-5.2` ($3.04/M, 1M context, structured
        # output) is bucketed a tier BELOW `zai/glm-4.5-airx` ($4.50/M, 128k) for
        # being cheaper, and the capability term — weighted 75% — then keeps the
        # older model. Cross-lab this comparison is meaningless, so it is confined
        # to a single lab, where a version number really does order the lineup.
        if scored:
            top = scored[0][1]
            top_lab, top_gen = lab_of(top), _generation(top.id)
            upgrade = next(
                (it for it in scored
                 if lab_of(it[1]) == top_lab
                 and _generation(it[1].id) > top_gen
                 and (it[1].context_tokens or 0) >= (top.context_tokens or 0)
                 and (it[1].supports_structured_output is not False
                      or top.supports_structured_output is not True)),
                None,
            )
            if upgrade is not None:
                scored = [upgrade] + [it for it in scored if it is not upgrade]

        # Independence constraint: prefer a different provider from the partner
        # role, dropping down the ranking to get it. Only a preference — if the
        # user has one provider, one model is the honest outcome, and it is
        # recorded as a warning rather than silently accepted.
        partners = [p for p in MUST_DIFFER_FROM.get(role, ()) if p in plan.assignments]
        partner = ", ".join(partners)
        independence_note = ""
        if partners:
            twin_labs = {lab_of(by_id.get(plan.assignments[p].model)
                                or plan.assignments[p].model) for p in partners}
            twin_lab = ", ".join(sorted(twin_labs))
            alt = next((it for it in scored if lab_of(it[1]) not in twin_labs), None)
            if alt is not None:
                scored = [alt] + [it for it in scored if it is not alt]
                independence_note = (
                    f" Routed to a different LAB from {partner} ({twin_lab}) so that their "
                    "agreement measures the guide rather than shared architecture."
                )
            else:
                independence_note = (
                    f" WARNING: shares a lab ({twin_lab}) with {partner}; their kappa will "
                    "partly reflect shared architecture rather than guide clarity."
                )

        # Same model, fewer hops. An aggregator adds a routing layer with its own
        # queueing, its own outages and its own rate limits; the lab that trained
        # the model is the more direct route to it. Measured on one pilot: the
        # OpenRouter-served path showed a 6.3x spread between its median and worst
        # call (67.8s to 429.9s) where a direct provider held 1.4x — a difference
        # model verbosity does not explain. Preference, not a rule: an aggregator
        # is sometimes cheaper and is sometimes the only route to a model.
        from .providers import BY_KEY

        def _is_direct(prov: str) -> bool:
            spec = BY_KEY.get(prov)
            return bool(spec and spec.kind == "direct")

        if scored and not _is_direct(scored[0][1].provider):
            same = next((it for it in scored
                         if _is_direct(it[1].provider)
                         and bare_model(it[1]) == bare_model(scored[0][1])), None)
            if same is not None:
                scored = [same] + [it for it in scored if it is not same]

        cost, best, tier = scored[0]

        relax_note = ""
        if tier_relaxed:
            relax_note = (
                f" NOTE: no reachable model reached the {req.reasoning} price tier, so the "
                "best available was used instead. Price is only a proxy for capability and "
                "under-prices strong cheap models — verify this choice if the role is critical."
            )

        # Fallbacks from *other* providers — a chain within one provider is one outage.
        # Fallbacks diversify by LAB, not gateway. A chain within one lab is one
        # outage; for a partner-constrained role it is also one architecture, so
        # failing over would quietly destroy the independence kappa depends on.
        fallbacks: list[str] = []
        seen = {lab_of(best)}
        for pname in partners:
            seen.add(lab_of(by_id.get(plan.assignments[pname].model)
                            or plan.assignments[pname].model))
        for _c, cand, _t in scored[1:]:
            if lab_of(cand) in seen:
                continue
            fallbacks.append(f"{cand.provider}:{cand.id}")
            seen.add(lab_of(cand))
            if len(fallbacks) >= 2:
                break

        # WHEN PRICE CANNOT RANK THE CANDIDATES, SAY SO.
        #
        # `cap` is capped at what the role requires, so every candidate at or
        # above the required tier scores identically on capability and `cheap`
        # decides. That is sound when price tracks capability. Across Chinese
        # labs it does not: `TIER_PERCENTILES` infers tier from PRICE PERCENTILE
        # over the reachable set, and after excluding the Western labs not one
        # Chinese model rates `frontier` — `deepseek-v4-pro`, `glm-5.2`,
        # `qwen3.8-max` and `kimi-k3` all land in the same `strong` band.
        #
        # So the router picked `glm-4.5-airx` for the referee over `glm-5.2` on a
        # $0.30/M input difference (airx is DEARER on output). Measured on the
        # same annotators: glm-5.2 chose annotator_a on 78.3% of contested rows,
        # glm-4.5-airx on 55.1% — near chance, which is what an adjudicator that
        # cannot discriminate looks like.
        #
        # The router has no capability signal to fix this with, so it must not
        # pretend otherwise: it names the rivals it could not separate and says
        # to pin deliberately. Advisory only — nothing is re-ranked here.
        rivals = [cand for _c, cand, t in scored[1:]
                  if TIER_ORDER.get(t, 0) >= TIER_ORDER.get(req.reasoning, 1)
                  and lab_of(cand) != lab_of(best)]
        close = [c for c in rivals
                 if (c.input_per_mtok or 0) + (c.output_per_mtok or 0)
                 <= 2.0 * ((best.input_per_mtok or 0) + (best.output_per_mtok or 0))]
        price_note: list[str] = []
        # No unpriced branch here: `_eligible` already refuses a card with no
        # published price, so the ranked path can never select one. A guard for
        # it would be unreachable — and it was, which is how mutation testing
        # found it. Only a PIN can bring an unpriced model in, because a pin
        # bypasses eligibility; that path warns.
        if capability_gated:
            price_note = [
                "chosen from the configured `capable_models` list; price only broke "
                "ties inside it, never against it"]
        elif capable_models and req.blast_radius in ("run", "phase"):
            price_note = [
                "NO configured `capable_model` was eligible for this role — fell back "
                "to price-derived tiers. Add one that meets the role's context and "
                "structured-output needs, or accept that price is choosing."]
        elif close and req.blast_radius in ("run", "phase"):
            price_note = [
                f"price chose among {len(close) + 1} candidates it cannot rank — "
                f"{best.id} won on cost over {', '.join(c.id for c in close[:3])} at the "
                f"same tier. Tier here is a PRICE PERCENTILE, not a capability score. "
                f"Pin this role in the config if the choice matters."]

        total = cost * req.typical_calls
        plan.assignments[role] = Assignment(
            role=role, model=best.id, api_model=best.api_id or best.id,
            input_per_mtok=best.input_per_mtok, output_per_mtok=best.output_per_mtok,
            max_output_tokens=best.max_output_tokens,
            provider=best.provider, tier=tier,
            fallbacks=fallbacks, estimated_cost_usd=total, estimated_calls=req.typical_calls,
            why=(f"{tier} tier meets the role's {req.reasoning} requirement; "
                 f"cost weighted {w_cost:.0%} because {req.typical_calls} call(s) at "
                 f"{req.blast_radius} blast radius" + independence_note + relax_note),
            warnings=(["shares a LAB with its partner annotator"]
                      if "WARNING" in independence_note else [])
                     + ([f"price tier relaxed: nothing reachable rated {req.reasoning}"]
                        if tier_relaxed else [])
                     + price_note,
        )
        plan.total_cost_usd += total

    if budget_usd is not None and plan.total_cost_usd > budget_usd:
        plan.notes.append(
            f"estimated ${plan.total_cost_usd:.2f} exceeds the ${budget_usd:.2f} budget. "
            "The high-volume roles (annotators, namers) dominate — lowering their tier or "
            "shrinking the gold set moves the total far more than changing the frontier roles, "
            "which are a rounding error by design."
        )
    # INDEPENDENCE IS CHECKED AFTER EVERY ROLE IS ASSIGNED, NOT DURING.
    # Both this loop and the pinned branch could only consult partners that had
    # ALREADY been assigned (`if p in plan.assignments`), so a role processed
    # early was checked against an empty set. Observed: a pinned `annotator_b`
    # was given a zhipu fallback because the referee — which runs on zhipu — had
    # not been assigned yet. Failing over there would turn independent
    # annotation into shared-architecture agreement without anything saying so.
    for role, a in plan.assignments.items():
        related = set(MUST_DIFFER_FROM.get(role, ()))
        related |= {o for o, must in MUST_DIFFER_FROM.items() if role in must}
        if not related or not a.fallbacks:
            continue
        forbidden = set()
        for other in related:
            b = plan.assignments.get(other)
            if b is None or not b.model:
                continue
            forbidden.add(lab_of(by_id.get(b.model) or b.model))
            for f in b.fallbacks:
                forbidden.add(lab_of(f.split(":", 1)[-1]))
        kept = [f for f in a.fallbacks if lab_of(f.split(":", 1)[-1]) not in forbidden]
        if len(kept) != len(a.fallbacks):
            dropped = [f for f in a.fallbacks if f not in kept]
            plan.notes.append(
                f"{role}: dropped {len(dropped)} fallback(s) sharing a lab with "
                f"{sorted(related)} — failing over there would destroy the "
                f"independence the gold set depends on ({', '.join(dropped)})")
            a.fallbacks = kept

    return plan
