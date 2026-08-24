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
    plan = route(catalog, ["anthropic"], prefer={"namer": "my-own-model"})
    assert plan.assignments["namer"].model == "my-own-model"


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
