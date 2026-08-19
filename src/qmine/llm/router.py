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
MUST_DIFFER_FROM: dict[str, str] = {"annotator_b": "annotator_a"}

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
    fallbacks: list[str] = field(default_factory=list)
    estimated_cost_usd: float = 0.0
    estimated_calls: int = 0
    why: str = ""
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": self.role, "model": self.model, "api_model": self.api_model,
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


def _assign_tiers(cards: Sequence[ModelCard]) -> dict[str, str]:
    """Bucket models into capability tiers by price percentile.

    Relative rather than absolute, so the tiers stay meaningful as the whole
    market gets cheaper — which it does, continuously. Computed across the full
    catalogue so that the reference distribution does not shift with how many
    provider keys happen to be configured.
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
    # The price-derived tier is a HARD gate only where an error is unrecoverable.
    # Everywhere else it is a scoring term, because the proxy's known failure is
    # precisely a strong-and-cheap model: DeepSeek at $0.28/$1M with 131k context
    # and structured output lands in a low price band and would be excluded from
    # the very high-volume roles it suits best. For a `run` blast radius we keep
    # the gate — being wrong there costs the whole run, so we pay for certainty.
    if req.blast_radius == "run" and TIER_ORDER.get(tier, 0) < TIER_ORDER.get(req.reasoning, 1):
        return False, f"tier {tier} below required {req.reasoning} for a run-critical role"
    return True, ""


def route(
    catalog: Catalog,
    available_providers: Sequence[str],
    *,
    roles: Sequence[str] | None = None,
    prefer: dict[str, str] | None = None,
    budget_usd: float | None = None,
    prefer_chinese_native: bool = False,
    avg_input_tokens: int = 6000,
) -> RoutingPlan:
    """Pick a model per role, with fallbacks and a cost estimate.

    ``prefer`` is an escape hatch that wins outright: a user who knows a specific
    model is right for a role should not have to argue with a heuristic.
    """
    from .requirements import ROLE_REQUIREMENTS

    role_list = list(roles or ROLE_REQUIREMENTS.keys())
    # Partner-constrained roles are routed after the role they must differ from.
    role_list.sort(key=lambda r: (r in MUST_DIFFER_FROM, r))
    cards = catalog.for_providers(available_providers)
    plan = RoutingPlan(catalog_provenance=catalog.provenance(),
                       providers_used=sorted(set(available_providers)))

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
        req = requirement_for(role)

        if prefer and role in prefer:
            chosen_id = prefer[role]
            card = by_id.get(chosen_id)
            plan.assignments[role] = Assignment(
                role=role, model=chosen_id,
                max_output_tokens=card.max_output_tokens if card else None,
                provider=card.provider if card else "explicit",
                tier=tiers.get(chosen_id, "explicit"),
                estimated_calls=req.typical_calls,
                estimated_cost_usd=(card.blended_cost(avg_input_tokens, req.output_tokens_per_call) or 0)
                                   * req.typical_calls if card else 0.0,
                why="explicitly preferred by the user; the router did not second-guess it",
            )
            continue

        scored: list[tuple[float, ModelCard, str]] = []
        relaxed: list[tuple[float, ModelCard, str]] = []
        for c in cards:
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
                bonus = 0.08
            return (1 - w_cost) * cap + w_cost * cheap + bonus

        scored.sort(key=_score, reverse=True)

        # Independence constraint: prefer a different provider from the partner
        # role, dropping down the ranking to get it. Only a preference — if the
        # user has one provider, one model is the honest outcome, and it is
        # recorded as a warning rather than silently accepted.
        partner = MUST_DIFFER_FROM.get(role)
        independence_note = ""
        if partner and partner in plan.assignments:
            twin = plan.assignments[partner]
            alt = next((it for it in scored if it[1].provider != twin.provider), None)
            if alt is not None:
                scored = [alt] + [it for it in scored if it is not alt]
                independence_note = (
                    f" Routed to a different provider from {partner} so that their agreement "
                    "measures the guide rather than shared architecture."
                )
            else:
                independence_note = (
                    f" WARNING: shares a provider with {partner}; their kappa will partly "
                    "reflect shared architecture rather than guide clarity."
                )

        cost, best, tier = scored[0]

        relax_note = ""
        if tier_relaxed:
            relax_note = (
                f" NOTE: no reachable model reached the {req.reasoning} price tier, so the "
                "best available was used instead. Price is only a proxy for capability and "
                "under-prices strong cheap models — verify this choice if the role is critical."
            )

        # Fallbacks from *other* providers — a chain within one provider is one outage.
        fallbacks: list[str] = []
        seen = {best.provider}
        for _c, cand, _t in scored[1:]:
            if cand.provider in seen:
                continue
            fallbacks.append(f"{cand.provider}:{cand.id}")
            seen.add(cand.provider)
            if len(fallbacks) >= 2:
                break

        total = cost * req.typical_calls
        plan.assignments[role] = Assignment(
            role=role, model=best.id, api_model=best.api_id or best.id,
            max_output_tokens=best.max_output_tokens,
            provider=best.provider, tier=tier,
            fallbacks=fallbacks, estimated_cost_usd=total, estimated_calls=req.typical_calls,
            why=(f"{tier} tier meets the role's {req.reasoning} requirement; "
                 f"cost weighted {w_cost:.0%} because {req.typical_calls} call(s) at "
                 f"{req.blast_radius} blast radius" + independence_note + relax_note),
            warnings=(["shares a provider with its partner annotator"]
                      if "WARNING" in independence_note else [])
                     + ([f"price tier relaxed: nothing reachable rated {req.reasoning}"]
                        if tier_relaxed else []),
        )
        plan.total_cost_usd += total

    if budget_usd is not None and plan.total_cost_usd > budget_usd:
        plan.notes.append(
            f"estimated ${plan.total_cost_usd:.2f} exceeds the ${budget_usd:.2f} budget. "
            "The high-volume roles (annotators, namers) dominate — lowering their tier or "
            "shrinking the gold set moves the total far more than changing the frontier roles, "
            "which are a rounding error by design."
        )
    return plan
