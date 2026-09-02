"""Tests for multi-provider routing.

These lock down the judgment calls, not the model list — the catalogue changes
weekly and a test asserting "opus is frontier" would be a maintenance burden
that fails for the wrong reason. What must stay true is the *reasoning*: never
route a run-critical role to an unvetted cheap model, never pay frontier prices
for a role that does not need them, never let a fallback chain sit inside one
provider, and never let two annotators quietly share an architecture.
"""

from __future__ import annotations

import os
import sys
from unittest import mock

import pytest

from qmine.llm.catalog import Catalog, ModelCard
from qmine.llm.providers import detect
from qmine.llm.requirements import ROLE_REQUIREMENTS, estimate_job, requirement_for
from qmine.llm.router import route


@pytest.fixture
def catalog() -> Catalog:
    """A small synthetic catalogue spanning price bands and providers."""
    def m(mid, prov, i, o, ctx=200_000, struct=True, img=False, dep=None):
        return ModelCard(id=mid, provider=prov, input_per_mtok=i, output_per_mtok=o,
                         context_tokens=ctx, supports_structured_output=struct,
                         emits_non_text=img, deprecated_on=dep)
    cards = [
        m("frontier-a", "anthropic", 15.0, 75.0),
        m("mid-a", "anthropic", 1.0, 5.0),
        m("frontier-b", "openai", 12.0, 60.0),
        m("mid-b", "openai", 0.8, 4.0),
        m("cheap-c", "deepseek", 0.28, 0.42, ctx=131_072),
        m("cheap-d", "openrouter", 0.20, 0.40, ctx=131_072),
        m("nostruct", "openai", 0.1, 0.2, struct=False),
        m("tiny-ctx", "openai", 0.1, 0.2, ctx=8_000),
        m("imagey", "openai", 14.0, 70.0, img=True),
        m("stale", "openai", 14.0, 70.0, dep="2020-01-01"),
    ]
    return Catalog(models={f"{c.provider}/{c.id}": c for c in cards},
                   fetched_at=1.0, sources=["test"])


def test_run_critical_roles_never_get_a_cheap_unvetted_model(catalog):
    """A wrong taxonomy poisons every downstream artifact; a wrong annotation costs one row."""
    plan = route(catalog, ["anthropic", "openai", "deepseek"])
    for role in ("taxonomy_architect", "tree_auditor", "taxonomy_critic"):
        a = plan.assignments[role]
        assert a.tier == "frontier", f"{role} landed on {a.tier}"


def test_high_volume_roles_are_not_given_frontier_prices(catalog):
    plan = route(catalog, ["anthropic", "openai", "deepseek"])
    annot = plan.assignments["annotator_a"]
    arch = plan.assignments["taxonomy_architect"]
    per_call_annot = annot.estimated_cost_usd / max(annot.estimated_calls, 1)
    per_call_arch = arch.estimated_cost_usd / max(arch.estimated_calls, 1)
    assert per_call_annot < per_call_arch, "the 240-call role costs more per call than the 1-call role"


def test_models_without_structured_output_are_excluded(catalog):
    plan = route(catalog, ["openai"])
    assert all(a.model != "nostruct" for a in plan.assignments.values())


def test_models_below_the_context_floor_are_excluded(catalog):
    plan = route(catalog, ["openai"])
    assert all(a.model != "tiny-ctx" for a in plan.assignments.values())


def test_image_emitting_models_are_excluded(catalog):
    """An image model prices like a frontier text model and is tuned for another job.

    This is not hypothetical — before the modality filter existed, the router
    chose a `gpt-5.4-image` variant for the tree auditor.
    """
    plan = route(catalog, ["openai"])
    assert all(a.model != "imagey" for a in plan.assignments.values())


def test_deprecated_models_are_excluded(catalog):
    plan = route(catalog, ["openai"])
    assert all(a.model != "stale" for a in plan.assignments.values())


def test_fallbacks_come_from_other_providers(catalog):
    """Three models from one provider is one outage, not a fallback chain."""
    plan = route(catalog, ["anthropic", "openai", "deepseek", "openrouter"])
    a = plan.assignments["annotator_a"]
    assert a.fallbacks
    assert all(not f.startswith(f"{a.provider}:") for f in a.fallbacks)


def test_the_two_annotators_are_routed_to_different_providers(catalog):
    """Their kappa is only evidence about the guide if their errors are independent."""
    plan = route(catalog, ["anthropic", "openai", "deepseek"])
    assert plan.assignments["annotator_a"].provider != plan.assignments["annotator_b"].provider


def test_single_provider_shares_annotators_but_says_so(catalog):
    plan = route(catalog, ["anthropic"])
    b = plan.assignments["annotator_b"]
    assert plan.assignments["annotator_a"].provider == b.provider
    assert b.warnings, "sharing an architecture between annotators must be surfaced"


def test_explicit_overrides_beat_the_router(catalog):
    plan = route(catalog, ["anthropic"], prefer={"namer": "mid-a"})
    assert plan.assignments["namer"].model == "mid-a"


def test_a_pin_too_new_for_the_catalogue_can_still_name_its_provider(catalog):
    """The catalogue is FETCHED, so a just-released model cannot have a card.

    `glm-5.3-flash` and `qwen3.8-flash` both answer on their providers' direct
    endpoints while appearing nowhere in a 1,930-model catalogue. Pinning one has
    to remain possible, or the price feed's release cadence decides which models
    this project may use.

    The provider must survive to the assignment: it is what resolves a base URL.
    Note the COLON — `qwen/qwen3.8-flash` is a catalogue id meaning the
    OpenRouter gateway, which is the hop these pins exist to avoid.
    """
    plan = route(catalog, ["anthropic"], prefer={"namer": "zhipu:glm-5.3-flash"})
    a = plan.assignments["namer"]
    assert a.provider == "zhipu", "a qualified pin must route to the named provider"
    assert a.api_model == "glm-5.3-flash", "the endpoint wants the bare id, not the pin"
    assert any("NO PRICE" in w for w in a.warnings), (
        "a synthesised card has no rate; it must read as UNKNOWN, never as free"
    )


def test_a_pin_that_names_no_reachable_provider_fails_before_the_run_starts(catalog):
    """A pin with no card used to reach `provider="explicit"`, a sentinel
    nothing handles: no base URL was resolved and `init_chat_model` got a bare
    id it could not attribute. The run died on that role's FIRST REAL CALL —
    after `qmine models` printed a clean plan, and after every phase above it
    had been paid for. Same rule as the startup schema probe: find the broken
    model before a real call pays for it.
    """
    from qmine.llm.router import UnroutablePin

    with pytest.raises(UnroutablePin) as e:
        route(catalog, ["anthropic"], prefer={"namer": "glm-9.9-imaginary"})
    # The message must teach the fix, not just report the failure.
    assert "glm-9.9-imaginary" in str(e.value)
    assert "zhipu:glm-9.9-imaginary" in str(e.value)

    with pytest.raises(UnroutablePin):
        route(catalog, ["anthropic"], prefer={"namer": "notaprovider:some-model"})


def test_an_unroutable_pin_does_not_degrade_to_the_static_tiers(catalog, tmp_path):
    """`_build_routing_plan` degrades on ANY exception so that a missing
    catalogue or a dead network cannot make the pipeline unrunnable offline.
    That handler also swallowed a bad pin — and degrading there honours NO pin
    at all, running the whole pipeline on models the user did not choose behind
    one `warning` line. Environment failures degrade; config errors fail closed.
    """
    from qmine.config import QMineConfig
    from qmine.llm.registry import ModelRegistry
    from qmine.llm.router import UnroutablePin

    cfg = QMineConfig()
    cfg.llm.provider = "router"
    cfg.llm.model_overrides = {"namer": "glm-9.9-imaginary"}
    with mock.patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-test"}, clear=False), \
         mock.patch("qmine.llm.catalog.fetch", return_value=catalog):
        with pytest.raises(UnroutablePin):
            ModelRegistry(cfg.llm, cache_dir=tmp_path, run_cfg=cfg)


def test_no_reachable_model_is_reported_not_silently_dropped(catalog):
    empty = Catalog(models={}, fetched_at=1.0, sources=["test"])
    plan = route(empty, ["anthropic"])
    assert plan.notes, "an empty catalogue must explain itself"


def test_budget_overrun_names_the_roles_that_actually_matter(catalog):
    plan = route(catalog, ["anthropic", "openai"], budget_usd=0.0001)
    assert any("high-volume" in n for n in plan.notes)


def test_cost_sensitivity_tracks_volume_and_stakes():
    assert requirement_for("annotator_a").cost_sensitivity > requirement_for("referee").cost_sensitivity
    assert requirement_for("taxonomy_architect").cost_sensitivity < 0.1


def test_every_registered_role_has_a_requirement():
    from qmine.agents.roles import ALL_ROLES

    for role in ALL_ROLES:
        assert requirement_for(role).reasoning in ("light", "standard", "strong", "frontier")


def test_job_estimate_is_dominated_by_the_annotators():
    est = estimate_job()
    annot = sum(ROLE_REQUIREMENTS[r].typical_calls for r in ("annotator_a", "annotator_b"))
    assert annot / est["total_calls"] > 0.5


def test_provider_detection_reads_the_environment():
    av = detect({"ANTHROPIC_API_KEY": "x", "DEEPSEEK_API_KEY": "y"})
    assert set(av.configured) == {"anthropic", "deepseek"}
    assert av.summary()["chinese_native_available"] == ["deepseek"]
    assert av.summary()["evidence"] == "environment variables only"


def test_catalog_degrades_without_network_or_cache(tmp_path):
    from qmine.llm.catalog import fetch

    cat = fetch(cache_dir=tmp_path, allow_network=False)
    assert cat.degraded and "floor" in cat.sources
    plan = route(cat, ["anthropic"])
    assert plan.notes, "flying blind must be stated, not hidden"


# -- structured-output robustness on OpenAI-compatible endpoints ------------

def test_fenced_json_is_salvaged_not_rerequested():
    """The commonest structured-output failure is correct JSON in a markdown fence.

    Observed on a live run against a Chinese-provider endpoint: three consecutive
    ```json blocks rejected by the strict parser and re-requested at ~60s each.
    The answer was right every time; only the wrapper was wrong.
    """
    from qmine.llm.registry import _salvage
    from qmine.records import LeafNaming

    class Raw:
        content = ('```json\n{"leaf_id": 3, "name_zh": "读音查询", "code": "pron", '
                   '"user_need": "拿到读音即满足", "coherence": 5}\n```')

    got = _salvage({"raw": Raw(), "parsed": None}, LeafNaming)
    assert got is not None and got.name_zh == "读音查询"


def test_prose_wrapped_json_is_salvaged():
    from qmine.llm.registry import _salvage
    from qmine.records import LeafNaming

    class Raw:
        content = ('好的，这是结果：\n{"leaf_id": 1, "name_zh": "汉字组词", "code": "wf", '
                   '"user_need": "拿到词语列表即满足", "coherence": 4}\n希望有帮助')

    got = _salvage({"raw": Raw(), "parsed": None}, LeafNaming)
    assert got is not None and got.name_zh == "汉字组词"


def test_salvage_declines_when_there_is_nothing_to_recover():
    from qmine.llm.registry import _salvage
    from qmine.records import LeafNaming

    class Raw:
        content = "I'm not going to answer that."

    assert _salvage({"raw": Raw(), "parsed": None}, LeafNaming) is None


# ==========================================================================
# Structured-output degradation
# ==========================================================================

def test_permanent_schema_failures_are_told_from_transient_ones():
    """The registry learns once that a model's native structured-output mode is
    unusable and stops paying for the discovery on every later call. Classifying
    a timeout as permanent would wrongly disable the fast path for the whole run;
    classifying a `response_format` rejection as transient wastes one attempt on
    each of ~240 annotation calls, which is what happened on the first live run.
    """
    from qmine.llm.registry import _native_schema_is_broken

    permanent = [
        # Returned a body, but not one the schema parser accepts (fenced JSON).
        "ValidationError: Invalid JSON: expected value at line 1 column 1",
        # Endpoint refuses the parameter outright.
        "BadRequestError: Error code: 400 - {'error': {'message': "
        "'This response_format type is unavailable now', 'type': 'invalid_request_error'}}",
        "BadRequestError: json_schema is not supported by this model",
        "BadRequestError: this model does not support tools",
    ]
    transient = [
        "APITimeoutError: Request timed out.",
        "RateLimitError: Error code: 429 - rate limit exceeded",
        "InternalServerError: Error code: 500",
        "APIConnectionError: Connection error.",
    ]
    for err in permanent:
        assert _native_schema_is_broken(err), f"should disable the native path: {err[:60]}"
    for err in transient:
        assert not _native_schema_is_broken(err), f"should stay retryable: {err[:60]}"


def test_a_region_probe_never_caches_a_host_it_has_evidence_against():
    """The probe tries each configured region until one accepts the key. If none
    answers it used to cache the default anyway — pinning every later call in the
    process to a host that had just returned 401. One transient failure cost an
    annotator 24 batches mid-run, and the resulting gold set was a third of its
    intended size."""
    import qmine.llm.providers as P

    spec = P.BY_KEY.get("qwen")
    assert spec and spec.alt_base_urls, "fixture needs a provider with alternates"

    saved_cache = dict(P._RESOLVED_BASE)
    saved_env = {v: os.environ.get(v) for v in spec.env_vars}
    try:
        P._RESOLVED_BASE.clear()
        os.environ[spec.env_vars[0]] = "sk-probe-test"

        class _Boom:
            @staticmethod
            def post(*a, **k):
                raise TimeoutError("network down")

        with mock.patch.dict(sys.modules, {"httpx": _Boom}):
            url = P.resolve_base_url("qwen", timeout=0.01)

        assert url == spec.openai_base_url, "must still return something usable"
        assert "qwen" not in P._RESOLVED_BASE, (
            "an unreachable probe cached a region — every later call in this "
            "process is now pinned to a host we have no evidence for"
        )
    finally:
        P._RESOLVED_BASE.clear()
        P._RESOLVED_BASE.update(saved_cache)
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_output_budgets_cover_what_the_roles_actually_emit():
    """Declared budgets set the generation cap (3x), so under-declaring truncates
    the role mid-answer. Measured per-role output from a live run:

      taxonomy_architect  39,647  — measured THREE times and rising on an
                                    unchanged prompt: 23,759, then 38,073 on
                                    live36, then 39,647 on live38 at 94% of cap
      annotator_a         21,975  — measured on live38/deepseek-v4-flash. The
                                    roles INVERTED: annotator_a is now the noisy
                                    one (21,975) and annotator_b the quiet one
                                    (1,792), the reverse of the run this table
                                    was first written from. Reasoning tokens are
                                    billed and capped as output, and which
                                    annotator draws the reasoning model is a
                                    ROUTING decision, so both budgets must fit
                                    the noisier of the two
      referee             19,597  — re-measured on live36/glm-5.2, up from 8,179.
                                    Ten calls died at EXACTLY 24,001 tokens (the
                                    12,000 cap bumped once to 2x) and each
                                    silently discarded 25 adjudications.

    Budgets go stale because the MODEL changes underneath them, not the task.
    Both figures above tripled without a line of prompt changing.
    """
    from qmine.llm.requirements import requirement_for

    measured = {
        "taxonomy_architect": 39_647,
        "annotator_b": 11_910,  # both budgets must fit the noisier role
        "annotator_a": 21_975,
        "referee": 19_597,
    }
    for role, observed in measured.items():
        cap = requirement_for(role).max_output_tokens
        assert cap >= observed * 1.2, (
            f"{role} emitted {observed:,} tokens live but is capped at {cap:,} — "
            "it will truncate, and a truncated structured answer fails to parse"
        )

    # The two annotators do the identical job, so their budgets must not diverge:
    # which one gets the reasoning model is a routing decision made later.
    assert (requirement_for("annotator_a").max_output_tokens
            == requirement_for("annotator_b").max_output_tokens)


def test_retries_are_not_multiplied_by_a_second_layer():
    """live44's maintainer burned 44 minutes and returned zero tokens.

    `_call` retries `max_repair + 1 = 3` times. With the SDK also set to
    `max_retries=2` each of those was 3 HTTP requests, so one logical call was
    **9 requests**. At the maintainer's 292s deadline that is 9 x 292 = 2,628s
    against the 2,638s the log recorded — and `p12_maintain` then reported
    ✔ completed, because the mechanical half of the phase had succeeded.

    A timeout is exactly the case where an SDK retry cannot help: an identical
    request with an identical deadline fails again by construction, and it does
    so INSIDE what our own logs count as a single attempt.
    """
    from qmine.config import LLMConfig, QMineConfig

    assert LLMConfig().max_retries == 0, \
        "SDK retries multiply our own; a timeout then costs 9x its deadline"
    assert QMineConfig().llm.max_retries == 0, "and the default config must carry it"


def test_a_reasoning_role_is_not_timed_at_writing_speed():
    """One global tok/s was wrong by ~5x in both directions.

    Throughput measures how fast a model WRITES; the deadline needs how long it
    takes to ANSWER. A reasoning model spends most of that thinking, which emits
    no output tokens — so thinking time lands in the denominator and never in
    the numerator. Measured on live44: annotator_a 181.7 tok/s against a
    tool-free researcher at 7.4.

    The old constant of 40 gave the researcher 585s for calls that legitimately
    take 850-1,150s, so it timed out, retried, and timed out again.
    """
    from qmine.llm.requirements import ROLE_REQUIREMENTS as R

    fast, slow = R["annotator_a"], R["researcher"]
    assert fast.reasoning in {"light", "standard"} and slow.reasoning in {"strong", "frontier"}

    # The measured floor for a reasoning role, with the 1.3 safety factor.
    assert slow.timeout_seconds >= 1200, \
        f"a reasoning role needs room to think, got {slow.timeout_seconds}s"
    # The reporter's 1,446.5s on live44 had no preceding failure line, so it is a
    # single call and the deadline must clear it. (Elapsed values that FOLLOW a
    # `!!` line are cumulative across attempts and cannot be read as one call —
    # the adversary's "658s" is 244 + 244 + ~170, not a 658s request.)
    assert R["reporter"].timeout_seconds > 1446, \
        f"reporter deadline {R['reporter'].timeout_seconds}s cuts off a measured 1446.5s call"


def test_a_role_is_always_given_time_to_emit_its_own_budget():
    """The timeout used to be a two-step function (180s / 420s) while the caps were
    per-role. Raising the architect's cap to 42,000 tokens to stop it truncating
    left it with 420s to write them — at the ~49 tokens/sec measured on these
    providers that is roughly half the time it needs, so the fix for one failure
    created another. Deriving the timeout from the cap makes that combination
    unreachable."""
    from qmine.llm.requirements import ROLE_REQUIREMENTS, RoleRequirement, requirement_for

    for name in ROLE_REQUIREMENTS:
        q = requirement_for(name)
        implied = q.max_output_tokens / RoleRequirement.THROUGHPUT_TOK_PER_SEC
        assert q.timeout_seconds >= implied, (
            f"{name} may emit {q.max_output_tokens:,} tokens but is cut off after "
            f"{q.timeout_seconds:.0f}s — it needs about {implied:.0f}s"
        )

    # The architect is the case that motivated this: one call, unbounded stakes.
    arch = requirement_for("taxonomy_architect")
    assert arch.timeout_seconds > 420, "still on the old two-step value"
    # ...and nothing waits forever on a hung request.
    assert all(requirement_for(n).timeout_seconds <= 1800 for n in ROLE_REQUIREMENTS)


def test_the_generation_cap_and_the_cost_estimate_are_separate_questions():
    """One field cannot answer both, and conflating them moved a model.

    The CAP must cover the noisiest model that could be routed to a role, or the
    call truncates. The ESTIMATE must reflect what is actually emitted, or the
    role is over-charged — and the router weighs cost, so an inflated estimate
    silently changes which model gets picked. Pricing both annotators at their
    peak (22,000) roughly doubled the pair and moved annotator_b to another lab
    on a number that was not real.
    """
    from qmine.llm.requirements import requirement_for

    a, b = requirement_for("annotator_a"), requirement_for("annotator_b")

    # Measured on live38: the two do identical work and emitted 21,975 vs 1,792.
    assert a.max_output_tokens == b.max_output_tokens, \
        "which annotator draws the reasoning model is decided later; the cap " \
        "must fit either, or whichever gets it truncates"
    assert a.max_output_tokens > 21_975 * 1.2, "the cap must clear the measured peak"
    assert a.output_tokens_per_call < a.max_output_tokens / 2, \
        "the estimate must be the expected cost, not the worst case"
    assert a.output_tokens_per_call == b.output_tokens_per_call, \
        "and symmetric, since routing has not happened when it is computed"


def test_a_pinned_role_still_counts_toward_the_estimated_total(catalog):
    """The `prefer` branch `continue`d past the accumulator.

    Pins land on the highest-VOLUME roles precisely because those are worth
    choosing deliberately — so excluding them understated the total by most of
    the bill. With both annotators pinned the header read "$0.79 per full run"
    against a table showing $36 of annotators on the same screen. An estimate
    that omits its largest line items is worse than none: it is what somebody
    reads to decide whether a model upgrade is affordable.
    """
    labs = ["anthropic", "openai", "deepseek"]
    free = route(catalog, labs)
    pinned = route(catalog, labs, prefer={"annotator_a": "mid-a"})

    a = pinned.assignments["annotator_a"]
    assert a.model == "mid-a" and a.estimated_cost_usd > 0

    # THE invariant, stated so it cannot pass by accident: the headline total is
    # the sum of the per-role estimates printed beside it. Asserting merely that
    # the total exceeds the pinned role's own cost is not decisive — the other
    # roles alone can clear that bar, which is how this survived mutation.
    for plan, label in ((free, "unpinned"), (pinned, "pinned")):
        parts = sum(x.estimated_cost_usd for x in plan.assignments.values())
        assert plan.total_cost_usd == pytest.approx(parts, rel=1e-6), (
            f"{label}: header total {plan.total_cost_usd} != sum of rows {parts}")

    # And the total must actually move when a pin changes the price.
    assert pinned.total_cost_usd != free.total_cost_usd


def test_the_router_says_when_price_could_not_rank_the_candidates(catalog):
    """`cap` is capped at the requirement, so every candidate at or above the
    required tier ties on capability and price decides.

    That is sound only while price tracks capability. Across Chinese labs it does
    not — tier is a PRICE PERCENTILE over the reachable set, and after excluding
    the Western labs not one Chinese model rates `frontier`: `deepseek-v4-pro`,
    `glm-5.2`, `qwen3.8-max` and `kimi-k3` all land in one `strong` band. The
    referee was handed `glm-4.5-airx` over `glm-5.2` on a $0.30/M input
    difference, and measured on the same annotators the cheap one adjudicated at
    near chance (55.1% vs 78.3%).

    The router cannot fix this — it has no capability signal — so it must not
    pretend it made a capability judgement.
    """
    plan = route(catalog, ["anthropic", "openai", "deepseek"])
    noted = [r for r, a in plan.assignments.items()
             if any("cannot rank" in w for w in a.warnings)]
    assert noted, "no role reported a cost-decided choice among same-tier rivals"
    for role in noted:
        a = plan.assignments[role]
        w = next(x for x in a.warnings if "cannot rank" in x)
        assert "Pin this role" in w, "the warning must say what to do about it"
        assert requirement_for(role).blast_radius in ("run", "phase"), (
            "only roles whose errors are expensive are worth this note")


def test_the_referee_is_a_run_blast_radius_role():
    """Its verdicts BECOME the gold set — they train the classifier, define
    kappa, and the rules it drafts ship in the guide the annotators read.

    At `phase` its `cost_sensitivity` is 0.25, and on a candidate set where price
    cannot rank capability that is enough to pick the cheapest model clearing the
    bar. `run` puts it at 0.05.
    """
    req = requirement_for("referee")
    assert req.blast_radius == "run"
    assert req.cost_sensitivity <= 0.05


def test_the_capable_models_list_gates_expensive_roles_and_price_only_breaks_ties(catalog):
    """Capability is STATED, not inferred from price.

    `TIER_PERCENTILES` derives tier from a price percentile and `_score` caps
    capability credit at the requirement, so among candidates at or above the
    bar the cheapest wins. Across the labs this project allows, price does not
    rank capability — not one Chinese model rates `frontier`, and the referee was
    handed an "air" lightweight over `glm-5.2` on $0.30/M of input.

    Removing price was tried and made things worse (it is the only thing keeping
    a 260-call role off an expensive model), so price stays and capability is
    supplied explicitly.
    """
    labs = ["anthropic", "openai", "deepseek"]
    free = route(catalog, labs)
    gated = route(catalog, labs, capable_models=["mid-a"])

    # A run/phase role must come from the list even though it is dearer.
    r = gated.assignments["referee"]
    assert requirement_for("referee").blast_radius == "run"
    assert r.model == "mid-a", f"capability gate ignored; got {r.model}"
    assert any("capable_models" in w for w in r.warnings)
    assert free.assignments["referee"].model != "mid-a", (
        "the fixture must actually differ, or this proves nothing")


def test_a_contained_high_volume_role_is_left_to_price(catalog):
    """The gate is not a blanket upgrade. `l2_interpreter` makes 20 calls and a
    wrong answer costs one cluster's sub-label — cheap-and-adequate is genuinely
    right there, and that is what the cost weighting is for."""
    gated = route(catalog, ["anthropic", "openai", "deepseek"], capable_models=["mid-a"])
    assert requirement_for("l2_interpreter").blast_radius == "contained"
    assert not any("capable_models" in w
                   for w in gated.assignments["l2_interpreter"].warnings)


def test_the_gate_falls_back_loudly_rather_than_leaving_a_role_unserved(catalog):
    """A capable_models list naming nothing reachable must not strand a role."""
    gated = route(catalog, ["anthropic", "openai", "deepseek"],
                  capable_models=["a-model-that-does-not-exist"])
    r = gated.assignments["referee"]
    assert r.model, "the role was left unserved"
    assert any("NO configured `capable_model`" in w for w in r.warnings)


def test_the_adversary_is_a_run_blast_radius_role():
    """Its output is the accuracy estimate the deliverable quotes for the WHOLE
    taxonomy, so an adversary that misses ships false assurance about every
    label. The field said `contained`, contradicting its own rationale, and that
    also kept it outside the capability gate."""
    assert requirement_for("adversary").blast_radius == "run"


def test_annotator_balance_reproduces_both_live_runs():
    """The measurement that would have caught this two runs earlier.

    Same annotator models in both runs; only the referee changed. The first is a
    real capability gap between the annotators, the second is an adjudicator
    deciding at near chance — which LOOKS like parity and is not.
    """
    from types import SimpleNamespace

    from qmine.ops.annotator_balance import annotator_balance

    def rows(a_wins, b_wins):
        out = []
        for i in range(a_wins):
            out.append(SimpleNamespace(adjudicated=True, label_a="A", label_b="B", final="A"))
        for i in range(b_wins):
            out.append(SimpleNamespace(adjudicated=True, label_a="A", label_b="B", final="B"))
        return out

    live38 = annotator_balance(rows(360, 100), "zhipu:glm-5.2")
    assert live38.n_decided == 460
    assert round(live38.a_share, 3) == 0.783
    assert live38.z > 10 and live38.lopsided

    live39 = annotator_balance(rows(253, 206), "zhipu:glm-4.5-airx")
    assert round(live39.a_share, 3) == 0.551
    assert not live39.lopsided, "z=+2.2 on 459 rows is not a systematic gap"
    assert live39.as_record()["referee_model"] == "zhipu:glm-4.5-airx", (
        "the referee id is what separates 'matched' from 'adjudicated at chance'")


def test_annotator_balance_uses_z_not_a_fixed_share():
    """The same 60/40 split is noise on 40 rows and decisive on 400."""
    from types import SimpleNamespace

    from qmine.ops.annotator_balance import annotator_balance

    def rows(a, b):
        return ([SimpleNamespace(adjudicated=True, label_a="A", label_b="B", final="A")] * a
                + [SimpleNamespace(adjudicated=True, label_a="A", label_b="B", final="B")] * b)

    assert not annotator_balance(rows(24, 16)).lopsided        # 60% of 40
    assert annotator_balance(rows(240, 160)).lopsided          # 60% of 400


def test_an_agreed_row_is_not_counted_as_a_win():
    """Rows the annotators agreed on never reached the referee; counting them
    would swamp the contested signal with unanimity."""
    from types import SimpleNamespace

    from qmine.ops.annotator_balance import annotator_balance

    bal = annotator_balance([
        SimpleNamespace(adjudicated=True, label_a="A", label_b="A", final="A"),
        SimpleNamespace(adjudicated=True, label_a="A", label_b="B", final="A"),
        SimpleNamespace(adjudicated=False, label_a="A", label_b="B", final="A"),
        SimpleNamespace(adjudicated=True, label_a="A", label_b="B", final="C"),
    ])
    assert bal.n_contested == 2 and bal.a_won == 1 and bal.neither == 1
    assert bal.b_won == 0


def test_a_model_with_no_published_price_is_flagged_not_costed_at_zero():
    """`blended_cost` returns None for a card with no rate, and `or 0` turns that
    into a confident $0.00 — on `annotator_b` that is ~256 calls reading as free,
    and the spend ledger prices from the same card so the run under-reports too.

    Real: `qwen3.7-plus` resolves to a DIRECT qwen card that publishes no price.
    Same family as pinned roles being excluded from the total — the largest line
    item is the one that goes missing.
    """
    from qmine.llm.catalog import Catalog, ModelCard

    priced = ModelCard(id="has-price", provider="deepseek", input_per_mtok=1.0,
                       output_per_mtok=2.0, context_tokens=200_000,
                       supports_structured_output=True)
    free = ModelCard(id="no-price", provider="deepseek", input_per_mtok=None,
                     output_per_mtok=None, context_tokens=200_000,
                     supports_structured_output=True)
    cat = Catalog(models={f"{c.provider}/{c.id}": c for c in (priced, free)},
                  fetched_at=1.0, sources=["test"])

    plan = route(cat, ["deepseek"], prefer={"annotator_b": "no-price"})
    a = plan.assignments["annotator_b"]
    assert a.model == "no-price"
    assert a.estimated_cost_usd == 0.0
    assert any("NO PRICE" in w for w in a.warnings), (
        "a $0.00 estimate must say the cost is unknown, not free")

    ok = route(cat, ["deepseek"], prefer={"annotator_b": "has-price"})
    assert not any("NO PRICE" in w for w in ok.assignments["annotator_b"].warnings)

    # The RANKED path cannot select one at all: `_eligible` refuses an unpriced
    # card as a likely free tier, preview or meta-endpoint. Only a pin can bring
    # one in, because a pin bypasses eligibility — which is exactly what makes
    # the pin-path warning necessary and a ranked-path one unreachable.
    picked = route(cat, ["deepseek"])
    assert picked.assignments["annotator_b"].model == "has-price", (
        "the ranked path must never select an unpriced card")


def test_the_balance_gate_says_it_is_diagnostic_only():
    """Kept, but demoted — it cannot separate a capability gap from a referee
    deciding at chance, so it must never read as a way to choose a model."""
    import inspect

    from qmine.ops import annotator_balance as mod

    doc = inspect.getdoc(mod) or ""
    assert "diagnostic, not a way to choose models" in doc


def test_the_run_prints_which_model_does_which_job_before_it_starts(catalog):
    """A run used to log one summary line, so the only way to learn that the
    referee was on a lightweight model — or that the observer was the SAME model
    as the architect whose work it reviews — was to read the manifest afterwards.

    Both were discovered here the hard way. The plan is known before the first
    call; printing it is the last cheap moment to stop a misrouted run.
    """
    from qmine.llm.registry import _plan_lines

    plan = route(catalog, ["anthropic", "openai", "deepseek"])
    lines = _plan_lines(plan)
    body = "\n".join(lines)

    assert "role assignments for this run" in lines[0]
    for role in ("referee", "observer", "adversary", "annotator_a", "annotator_b"):
        assert any(line.strip().startswith(role) for line in lines), role
    # The independence property must be stated in the terms the RULE is written
    # in — by lab — because two labs reach you through one gateway and look
    # identical in the model column.
    assert "annotator/referee labs:" in body
    assert "lab=" in body
    # And every router warning has to travel with it, or the table reads as a
    # clean bill of health it did not earn.
    warned = [r for r, a in plan.assignments.items() if a.warnings]
    for role in warned:
        assert f"! {role}:" in body, f"{role}'s warning was dropped from the plan output"


def test_the_plan_output_names_an_unserved_role_rather_than_omitting_it(catalog):
    """A role with no model must not simply be absent from the list — an absence
    reads as 'nothing to say about it'."""
    from qmine.llm.registry import _plan_lines
    from qmine.llm.router import Assignment, RoutingPlan

    plan = RoutingPlan(assignments={
        "referee": Assignment(role="referee", model="", provider="", tier="")})
    assert any("UNSERVED" in line for line in _plan_lines(plan))


def test_a_labs_own_endpoint_beats_a_gateway_reselling_the_same_model():
    """This rule already existed, and I duplicated it before checking.

    `route` runs a "same model, fewer hops" promotion AFTER ranking, scoped to
    the same bare model, with measured evidence behind it: the gateway path
    showed a 6.3x spread between its median and worst call (67.8s to 429.9s)
    against 1.4x direct. It is exercised in `test_lab_independence`.

    A scoring BONUS on top of it is not the same rule — it biases toward models
    whose lab happens to be their own provider, which silently swapped
    `l2_interpreter` from a deepseek model to a qwen one. This test pins the
    promotion's actual shape so the difference stays visible.
    """
    from qmine.llm.catalog import Catalog, ModelCard

    direct = ModelCard(id="acme-1", provider="deepseek", input_per_mtok=1.0,
                       output_per_mtok=2.0, context_tokens=200_000,
                       supports_structured_output=True)
    resold = ModelCard(id="deepseek/acme-1", provider="openrouter", input_per_mtok=0.5,
                       output_per_mtok=1.0, context_tokens=200_000,
                       supports_structured_output=True)
    cat = Catalog(models={f"{c.provider}/{c.id}": c for c in (direct, resold)},
                  fetched_at=1.0, sources=["test"])

    plan = route(cat, ["deepseek", "openrouter"])
    a = plan.assignments["namer"]
    assert a.provider == "deepseek", (
        "the direct route to the SAME bare model must win, even though the "
        "gateway is half the price and therefore scores higher")



def test_the_direct_promotion_does_not_override_a_capability_gate():
    """A direct endpoint for a model nobody judged capable must not beat a
    gateway-served one that IS on the list. The fewer-hops promotion is scoped to
    the SAME bare model, so it cannot substitute a different one."""
    from qmine.llm.catalog import Catalog, ModelCard

    direct_weak = ModelCard(id="weak-1", provider="deepseek", input_per_mtok=0.5,
                            output_per_mtok=1.0, context_tokens=200_000,
                            supports_structured_output=True)
    resold_good = ModelCard(id="zhipu/good-1", provider="openrouter", input_per_mtok=2.0,
                            output_per_mtok=4.0, context_tokens=200_000,
                            supports_structured_output=True)
    cat = Catalog(models={f"{c.provider}/{c.id}": c for c in (direct_weak, resold_good)},
                  fetched_at=1.0, sources=["test"])

    plan = route(cat, ["deepseek", "openrouter"], capable_models=["good-1"])
    a = plan.assignments["referee"]          # blast_radius=run, so gated
    assert a.model == "zhipu/good-1", (
        "the capability gate must bind before the direct-provider preference")


def test_the_default_posture_is_REAL_agents():
    """This pipeline IS a team of agents. With the deterministic stand-in it is a
    function wearing their output shape: a full set of deliverables, every gate
    passing, and nothing in the documents saying no model wrote the prose. So
    routing is the default and offline is a fallback.

    Two things had to be true for that, and neither was:

    * `configs/live.yaml` is the default config and must ask to route;
    * `--provider` defaulted to `"auto"` and was assigned unconditionally, so
      Typer's default overwrote whatever the config said before the file was read
      for anything else.
    """
    import inspect
    from pathlib import Path

    import yaml

    from qmine import cli

    raw = yaml.safe_load(Path("configs/live.yaml").read_text()) or {}
    assert (raw.get("llm") or {}).get("provider") == "router", (
        "the default config must ask for real models")

    sig = inspect.signature(cli.run).parameters["provider"].default
    assert getattr(sig, "default", sig) == "", (
        "a non-empty --provider default overrules the config on every run")

    src = inspect.getsource(cli.run)
    assert "if provider:\n        cfg.llm.provider = provider" in src, (
        "the provider is assigned unconditionally, so the config never wins")


def test_auto_means_whatever_is_reachable_not_anthropic():
    """`auto` asked only `_has_anthropic_credentials()`, so a machine with four
    working providers configured — deepseek, qwen, zhipu, openrouter — resolved
    to the OFFLINE stand-in because none of them was Anthropic. On a project
    whose own live config EXCLUDES Anthropic, `auto` could never route at all.
    """
    import inspect

    from qmine.llm.registry import ModelRegistry

    src = inspect.getsource(ModelRegistry._resolve_provider)
    # Strip comments and docstring text: the fix's own comment names the old
    # helper, and an assertion that matches it tests nothing.
    head = src[:src.index('if requested ==', src.index('auto'))]
    code = "\n".join(ln for ln in head.splitlines()
                     if not ln.strip().startswith(("#", '"', "'")))
    assert "detect()" in code, "auto must ask what is actually reachable"
    assert "_has_anthropic_credentials" not in code, (
        "auto is deciding on one vendor's credentials again")


def test_the_free_commands_stay_free():
    """`demo` is the documented ~4-minute wiring check and `full` the ~25-minute
    offline pass. The default config now routes, so both must say offline
    explicitly or they quietly become paid runs on 8,000 and 50,000 rows.
    """
    import inspect
    from pathlib import Path

    from qmine import cli

    demo_src = inspect.getsource(cli.demo)
    assert "offline=True" in demo_src and 'provider="mock"' in demo_src, (
        "demo would route to real models")

    makefile = Path("Makefile").read_text()
    full = makefile[makefile.index("\nfull:"):]
    full = full[:full.index("\n\n")]
    assert "--offline" in full, "make full would route to real models"


def test_a_config_the_caller_names_still_wins():
    """Defaulting must not shadow an explicit `--config`."""
    import tempfile

    from qmine.cli import _load_config

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        fh.write("llm:\n  temperature: 0.42\n")
        path = fh.name
    cfg = _load_config(path, "k12_zh")
    assert cfg.llm.temperature == 0.42
    assert not cfg.llm.model_overrides, "the default file leaked into an explicit config"


def test_the_preflight_routes_against_the_config_a_run_would_use():
    """`qmine models` is the pre-flight: it exists to show which model each role
    gets before any money is spent. It guarded its config load with
    `if (config or domain)`, so a bare `qmine models` built a stock
    `QMineConfig()` and pre-flighted a configuration no run would ever use —
    listing price-chosen models where the run will use pinned ones.

    That is the exact failure the comment above the line describes; it was left
    in place for the one case where nothing on the command line hints at it.
    """
    import inspect

    from qmine import cli

    src = inspect.getsource(cli.models_cmd)
    call = src[src.index("cfg = _load_config"):]
    call = call[:call.index("\n")]
    assert "if (config or domain)" not in call, (
        "the pre-flight skips the config when no flag is given, so it "
        "pre-flights a different configuration than the run")
    assert "QMineConfig()" not in call


def test_doctor_checks_the_providers_this_project_actually_uses():
    """`doctor` tested ANTHROPIC_API_KEY alone.

    The project routes to DeepSeek, Zhipu and Qwen, so on a fully-configured
    machine it reported "absent → will fall back to the deterministic offline
    stand-in" and said nothing about the keys in use. `_resolve_provider` was
    fixed to consult `detect()` after the same bug; this is the other half.
    """
    import inspect

    from qmine import cli

    src = inspect.getsource(cli.doctor)
    assert "detect()" in src, "doctor must ask which providers are configured"
    assert 'os.environ.get("ANTHROPIC_API_KEY")' not in src, \
        "one vendor's variable is not this project's credential check"


def test_an_override_for_a_suffixed_role_is_actually_planned(catalog):
    """A per-angle override was dead config, and `qmine models` echoed it as live.

    `role_list` came from `ROLE_REQUIREMENTS`, which holds BASE roles only, so
    `researcher_log_reading` was never planned, `role in prefer` never saw it,
    and the entry did nothing. It was found only by routing around a
    deterministic failure and watching three runs use the model it had routed
    away from.

    `requirement_for` resolves a suffixed role to its base requirement and
    `route_for` prefers an exact match over the longest prefix, so planning the
    override's own role makes the specific pin win while unsuffixed siblings keep
    the base assignment.
    """
    from qmine.llm.router import route

    plan = route(catalog, ["zhipu", "deepseek"],
                 prefer={"researcher": "zhipu:glm-5.3-flash",
                         "researcher_log_reading": "deepseek:deepseek-v4-pro"})

    assert "researcher_log_reading" in plan.assignments, \
        "an override must name a role the plan contains, or it is dead config"
    assert "researcher" in plan.assignments, "the base role must survive alongside it"
    a = plan.assignments["researcher_log_reading"]
    assert (a.api_model or a.model) and "deepseek" in f"{a.provider}{a.model}".lower(), \
        f"the specific pin must win for the suffixed role, got {a.provider}/{a.model}"


def test_an_override_naming_no_known_role_warns(catalog, caplog):
    """A typo'd override key resolves to DEFAULT requirements rather than failing.

    `requirement_for` returns a generic requirement for any unknown string, so a
    misspelled key would be planned quietly at the wrong tier. An override that
    resolves to nothing is worse than none, because it looks applied.
    """
    import logging

    from qmine.llm.router import route

    with caplog.at_level(logging.WARNING, logger="qmine.router"):
        route(catalog, ["zhipu"], prefer={"reserchr_typo": "zhipu:glm-5.3-flash"})
    assert any("matches no known role" in r.getMessage() for r in caplog.records), \
        "a typo'd override key must warn, not plan silently"
