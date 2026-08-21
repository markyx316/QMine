"""Failover onto the alternates the router already chose.

A live run hit `402 Insufficient Balance` on DeepSeek mid gold-annotation. The
retry loop asked the same dead endpoint three times per batch, dropped the batch,
and did it again — twelve batches, 300 rows — while two usable fallbacks sat in
`run_manifest.json` untouched. The router had planned around the outage and
nothing consulted the plan.
"""

from __future__ import annotations

from qmine.llm.registry import _native_schema_is_broken, _provider_is_unusable

BALANCE = ("APIStatusError: Error code: 402 - {'error': {'message': "
           "'Insufficient Balance', 'type': 'unknown_error'}}")


def test_a_billing_failure_is_not_a_retryable_hiccup():
    assert _provider_is_unusable(BALANCE)
    assert _provider_is_unusable("AuthenticationError: 401 invalid api key")
    assert _provider_is_unusable("Error code: 403 - quota exceeded")


def test_transient_trouble_must_stay_retryable():
    """A busy minute must not permanently abandon a working provider."""
    for err in ("RateLimitError: 429 too many requests",
                "APITimeoutError: request timed out",
                "APIConnectionError: connection reset",
                "InternalServerError: 500"):
        assert not _provider_is_unusable(err), err


def test_a_schema_miss_is_not_a_dead_provider():
    """Opposite remedies: repair the parse, or abandon the endpoint."""
    schema_err = "ValidationError: 1 validation error for ResearchSubmission"
    assert _native_schema_is_broken(schema_err)
    assert not _provider_is_unusable(schema_err)
    assert not _native_schema_is_broken(BALANCE)


def make_registry():
    from qmine.config import QMineConfig
    from qmine.llm.registry import ModelRegistry

    cfg = QMineConfig()
    reg = ModelRegistry(cfg.llm, cache_dir=None, run_cfg=cfg)
    reg._routed = {"annotator_b": ("deepseek", "deepseek-v4-flash"),
                   "annotator_a": ("qwen", "qwen3-next")}
    reg._fallbacks = {"annotator_b": [("qwen", "qwen3-next"), ("zhipu", "zai/glm-4.7-flash")],
                      "annotator_a": [("deepseek", "deepseek-v4-flash")]}
    reg.plan = None   # usage() renders the plan; not what these tests exercise
    return reg


def test_a_dead_provider_routes_to_the_alternate():
    reg = make_registry()
    assert reg.route_for("annotator_b") == ("deepseek", "deepseek-v4-flash")
    reg.mark_provider_dead("deepseek", BALANCE, "annotator_b")
    assert reg.route_for("annotator_b") == ("qwen", "qwen3-next")


def test_it_is_remembered_so_the_outage_is_not_rediscovered_per_call():
    """Eight concurrent batches rediscovering the outage cost three attempts each."""
    reg = make_registry()
    reg.mark_provider_dead("deepseek", BALANCE, "annotator_b")
    for _ in range(8):
        assert reg.route_for("annotator_b")[0] == "qwen"
    assert len(reg.failovers) == 1, "the event is recorded once, not per call"


def test_a_second_outage_walks_further_down_the_list():
    reg = make_registry()
    reg.mark_provider_dead("deepseek", BALANCE, "annotator_b")
    reg.mark_provider_dead("qwen", "401 invalid api key", "annotator_b")
    assert reg.route_for("annotator_b") == ("zhipu", "zai/glm-4.7-flash")


def test_suffixed_roles_find_their_fallbacks():
    """Roles arrive suffixed; an exact-match lookup misses every one of them."""
    reg = make_registry()
    reg._routed = {"namer": ("deepseek", "deepseek-v4-flash")}
    reg._fallbacks = {"namer": [("qwen", "qwen3-next")]}
    reg.mark_provider_dead("deepseek", BALANCE, "namer_3")
    assert reg.route_for("namer_3") == ("qwen", "qwen3-next")


def test_the_run_summary_says_what_actually_answered():
    reg = make_registry()
    reg.mark_provider_dead("deepseek", BALANCE, "annotator_b")
    u = reg.usage()
    assert u["failovers"][0]["provider"] == "deepseek"
    assert u["failovers"][0]["replaced_by"] == ("qwen", "qwen3-next")
    assert "deepseek" in u["dead_providers"]


def test_cached_clients_are_dropped_when_a_provider_dies():
    """A cached LangChain client still points at the dead endpoint."""
    reg = make_registry()
    reg._models["deepseek:fast"] = object()
    reg.mark_provider_dead("deepseek", BALANCE, "annotator_b")
    assert not reg._models


def test_token_counts_in_an_error_do_not_kill_a_provider():
    """A model hit its output cap and the error carried
    `CompletionUsage(completion_tokens=4013, prompt_tokens=8402)`.

    Naive substring matching saw "401" and "402" in those counts, marked the whole
    of Qwen dead, and failed taxonomy research over to a CODING model. A status
    code only counts where a status code appears.
    """
    err = ("LengthFinishReasonError: Could not parse response content as the length "
           "limit was reached - CompletionUsage(completion_tokens=4013, "
           "prompt_tokens=8402, total_tokens=12415)")
    assert not _provider_is_unusable(err)
    assert not _provider_is_unusable("BadRequestError: total_tokens=40213 exceeds context")


def test_real_status_codes_are_still_caught():
    assert _provider_is_unusable("APIStatusError: Error code: 402 - Insufficient Balance")
    assert _provider_is_unusable("Error code: 403 - quota exceeded")
    assert _provider_is_unusable("AuthenticationError: 401 invalid api key")
    assert _provider_is_unusable("PermissionDeniedError: account is suspended")


def test_a_truncation_buys_more_room_not_a_new_provider():
    """Budgets were measured on `glm-4.5-airx`; `glm-5.2` is more verbose and
    overran the researcher's 12,000-token cap on its first call. The remedy is
    room, not a different model — swapping providers abandons a working one for a
    problem it did not have."""
    from qmine.llm.registry import _hit_length_limit, _provider_is_unusable

    err = "LengthFinishReasonError: the length limit was reached"
    assert _hit_length_limit(err)
    assert not _provider_is_unusable(err), "must not be treated as a dead provider"

    for other in ("RateLimitError: 429", "ValidationError: bad schema",
                  "Error code: 402 - Insufficient Balance"):
        assert not _hit_length_limit(other), other


def test_the_raised_cap_actually_reaches_the_client():
    """`_build_routed` reuses one client per (provider, model, timeout, cap). If
    the cap were not in that key, raising it would silently return the cached
    client with the old limit and the retry would truncate again."""
    import inspect

    from qmine.llm.registry import ModelRegistry

    src = inspect.getsource(ModelRegistry._build_routed)
    assert "_length_bump" in src, "the bump must feed the cap"
    assert "{cap}" in src, "and the cap must be part of the client cache key"


def test_one_role_learning_a_models_verbosity_serves_every_role_on_it():
    """Verbosity is a property of the MODEL, not the role.

    Keyed per role, five researchers each paid ~180s and a discarded call to
    discover the same fact about `glm-5.2` — observed identically across two live
    runs, roughly six minutes per run spent learning one thing five times. The
    structured-output switch (`_no_native_schema`) is per-model for exactly this
    reason.
    """
    import inspect

    from qmine.llm.registry import ModelRegistry

    build = inspect.getsource(ModelRegistry._build_routed)
    assert "_length_bump.get(model_id" in build, \
        "the cap must be looked up by MODEL, not by role"

    complete = inspect.getsource(ModelRegistry.complete)
    assert "_length_bump[truncated]" in complete
    assert "self.model_name(tier, role)" in complete, \
        "the bump must be recorded against the model that truncated"


def test_the_bump_is_bounded():
    """A cap that doubles without limit would ask for more than any model emits,
    and the timeout does not grow with it."""
    import inspect

    from qmine.llm.registry import ModelRegistry

    src = inspect.getsource(ModelRegistry.complete)
    assert ", 8)" in src, "the multiplier must be capped"


def test_a_truncation_also_abandons_native_structured_output_for_that_model():
    """More room was only half the remedy.

    Measured four times on `glm-5.2`: the call truncates past 12,000 tokens in
    NATIVE structured-output mode, then completes in ~5,300 on the plain-JSON
    path — the same answer for less than half the tokens. The schema scaffolding
    is what overran the cap, so a truncation is evidence against that mode for
    this model, exactly as an unparseable response already is.
    """
    import inspect

    from qmine.llm.registry import ModelRegistry

    src = inspect.getsource(ModelRegistry.complete)
    i = src.index("_hit_length_limit(last_err)")
    branch = src[i:i + 900]
    assert "_no_native_schema.add(truncated)" in branch, \
        "a truncation must also switch that model to plain-JSON mode"
    assert "_length_bump[truncated]" in branch, "and still grant more room"
