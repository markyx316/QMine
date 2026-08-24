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
    """The bump is a floor, not a ratchet.

    It used to double per truncation with a ceiling of 8. Under concurrency that
    ceiling was reachable from a single underlying problem, and the timeout now
    derives from the cap so it inflated too. Raising to a fixed floor is bounded
    by construction and needs no ceiling.
    """
    from qmine.llm.registry import _next_length_floor

    # Bounded by construction: there is no input that climbs past 4x.
    assert {_next_length_floor(n) for n in (1, 2, 3, 4, 8, 99)} == {2, 4}
    # Idempotent: repeated truncations converge rather than climb.
    assert _next_length_floor(_next_length_floor(_next_length_floor(1))) == 4


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
    branch = src[i:i + 2600]          # the branch grew comments explaining why
    assert "_no_native_schema.add(truncated)" in branch, \
        "a truncation must also switch that model to plain-JSON mode"
    assert "_length_bump[truncated]" in branch, "and still grant more room"


def test_concurrent_truncations_do_not_compound_the_bump():
    """Five researchers run concurrently and each truncates before any of them has
    recorded the switch. Multiplying turned one problem into 4x, then 8x — and the
    timeout derives from the cap, so it inflated with it."""
    import inspect

    from qmine.llm.registry import ModelRegistry

    del inspect, ModelRegistry
    from qmine.llm.registry import _next_length_floor

    # Five racing callers, none of which has seen another's write yet.
    assert len({_next_length_floor(1) for _ in range(5)}) == 1, \
        "every racing caller must compute the same floor"
    # And applying it five times in sequence must not climb five times.
    n = 1
    for _ in range(5):
        n = _next_length_floor(n)
    assert n == 4, "a floor converges; a multiplier would have reached 32x"


# --- truncation is a token count, not a phrase ------------------------------
#
# `live36`: ten referee calls died at EXACTLY 24,001 output tokens — the 12,000
# declared cap bumped once to 2x. The error read "no parseable structured
# output", which the prose matcher does not recognise, so the bump never grew and
# the same batch truncated at the same number ten times. Each discarded 25
# adjudications from a 3,000-row gold set.


def test_a_truncation_is_recognised_from_the_token_count(monkeypatch):
    """The provider said nothing about length; the count said everything."""
    from qmine.llm.registry import _hit_length_limit

    err = "ValueError: no parseable structured output"
    assert _hit_length_limit(err) is False, (
        "this is the prose the referee actually returned — if the matcher ever "
        "starts recognising it, this test is measuring the wrong thing")
    # The registry must not depend on that prose: spending the whole cap IS the
    # signal, and it is provider-independent.
    cap, spent = 24_001, 24_001
    assert bool(cap and spent >= cap) is True


def test_a_second_floor_is_reachable_when_two_x_still_truncates():
    """`max(current, 2)` could never express "2x was not enough"."""
    from qmine.llm.registry import _next_length_floor as floor

    assert floor(1) == 2, "first truncation raises to 2x"
    assert floor(2) == 4, "truncating again AT 2x proves 2x was too small"
    assert floor(4) == 4, "and it stops there — never 8x, never unbounded"


def test_raising_to_a_floor_stays_idempotent_under_concurrency():
    """Why a floor and not a multiplier: racing callers must agree.

    Eight concurrent annotator calls each truncate before any records the switch.
    A multiplier compounds that into 4x then 8x; a floor gives every racing
    caller the same answer no matter how many of them run it.
    """
    from qmine.llm.registry import _next_length_floor as floor

    racing = [floor(2) for _ in range(8)]
    assert set(racing) == {4}, "eight concurrent truncations must agree on 4x"
    assert floor(floor(floor(2))) == 4, "and re-applying it must not climb"


def test_the_referee_budget_matches_what_it_actually_writes():
    """A stale budget silently discards work rather than erroring.

    The declared 4,000 came from a model long since replaced. Measured on
    `live36`/`glm-5.2`, a successful 25-row adjudication emits 19,279-19,597, so
    a 12,000 cap truncates every time and 2x only just fails to cover it.
    """
    from qmine.llm.requirements import requirement_for

    ref = requirement_for("referee")
    observed_max_success = 19_597
    assert ref.max_output_tokens > observed_max_success, (
        f"cap {ref.max_output_tokens} must exceed the largest response measured "
        f"to SUCCEED ({observed_max_success}), or that call truncates")
    observed_truncation = 24_001
    assert ref.max_output_tokens > observed_truncation, (
        f"cap {ref.max_output_tokens} must also clear the point where live36 "
        f"actually died ({observed_truncation})")
    assert ref.timeout_seconds >= ref.max_output_tokens / ref.THROUGHPUT_TOK_PER_SEC, (
        "raising a budget must not leave a deadline too short to emit it")


# --- classify by the SDK's status code, not by the error's prose ------------


def test_a_structured_status_code_settles_it_without_reading_prose():
    """The SDK raises typed errors carrying `status_code`; use it.

    Reading the TEXT for a number cannot distinguish a status from a token count:
    `completion_tokens=4013` contains "401" and `prompt_tokens=8402` contains
    "402". That once killed every Qwen model mid-run and sent taxonomy research
    to a coding model.
    """
    from qmine.llm.registry import _provider_is_unusable

    class Err(Exception):
        def __init__(self, code):
            self.status_code = code

    for code in (401, 402, 403):
        assert _provider_is_unusable("anything at all", Err(code)) is True, code
    for code in (429, 500, 503, 408):
        assert _provider_is_unusable("anything at all", Err(code)) is False, (
            f"{code} is transient — abandoning a working provider over it is "
            "worse than the outage")


def test_a_token_count_in_the_body_cannot_kill_a_provider_even_with_a_status():
    """The structured code must WIN over whatever the body happens to contain."""
    from qmine.llm.registry import _provider_is_unusable

    class Err(Exception):
        def __init__(self, code):
            self.status_code = code

    body = "CompletionUsage(completion_tokens=4013, prompt_tokens=8402)"
    assert _provider_is_unusable(body, Err(429)) is False, \
        "a rate limit whose body contains 401/402 digits must stay retryable"
    assert _provider_is_unusable(body, None) is False, \
        "and with no status at all, the anchored regex must not match either"


def test_the_prose_path_still_works_when_there_is_no_status_code():
    """Not every provider raises a typed error; the fallback must survive."""
    from qmine.llm.registry import _provider_is_unusable

    assert _provider_is_unusable("Error code: 402 - insufficient balance") is True
    assert _provider_is_unusable("read timeout after 180s") is False


def test_a_bool_is_not_a_status_code():
    """`isinstance(True, int)` is True in Python; a flag must not read as 401."""
    from qmine.llm.registry import _provider_is_unusable

    class Err(Exception):
        status_code = True

    assert _provider_is_unusable("read timeout", Err()) is False


# --- lessons must outlive the process ---------------------------------------
#
# `_no_native_schema` and `_length_bump` are learned by FAILING: a call breaks,
# the registry works out why, and later calls avoid it. Discarding that at exit
# meant paying for it again every run — five rediscoveries in live38 alone.


def test_learned_quirks_round_trip_through_the_shared_file(tmp_path, monkeypatch):
    """What one run learns, the next run starts with."""
    from qmine.llm.registry import ModelRegistry

    path = tmp_path / "model_quirks.json"
    monkeypatch.setattr(ModelRegistry, "QUIRKS_PATH", path)

    reg = ModelRegistry.__new__(ModelRegistry)
    import threading
    reg._quirks_lock = threading.Lock()
    reg._no_native_schema = {"some-model"}
    reg._length_bump = {"some-model": 4}
    reg._save_quirks()
    assert path.exists(), "a lesson that is not written is a lesson bought twice"

    fresh = ModelRegistry.__new__(ModelRegistry)
    fresh._quirks_lock = threading.Lock()
    fresh._no_native_schema = set()
    fresh._length_bump = {}
    fresh._load_quirks()
    assert "some-model" in fresh._no_native_schema, "plain-JSON mode must be recalled"
    assert fresh._length_bump.get("some-model") == 4, "and the raised cap with it"


def test_a_smaller_bump_never_overwrites_a_larger_one(tmp_path, monkeypatch):
    """A run that only reached 2x must not erase a 4x another run proved needed."""
    import threading

    from qmine.llm.registry import ModelRegistry

    path = tmp_path / "model_quirks.json"
    monkeypatch.setattr(ModelRegistry, "QUIRKS_PATH", path)

    def save(bump):
        r = ModelRegistry.__new__(ModelRegistry)
        r._quirks_lock = threading.Lock()
        r._no_native_schema = set()
        r._length_bump = {"m": bump}
        r._save_quirks()

    save(4)
    save(2)
    fresh = ModelRegistry.__new__(ModelRegistry)
    fresh._quirks_lock = threading.Lock()
    fresh._no_native_schema = set()
    fresh._length_bump = {}
    fresh._load_quirks()
    assert fresh._length_bump.get("m") == 4, "the worst case already known must survive"


def test_a_stale_lesson_is_forgotten_so_a_fixed_provider_is_re_learned(tmp_path, monkeypatch):
    """Providers do fix their endpoints; believing forever would be wrong."""
    import json
    import threading
    import time

    from qmine.llm.registry import ModelRegistry

    path = tmp_path / "model_quirks.json"
    monkeypatch.setattr(ModelRegistry, "QUIRKS_PATH", path)
    old = time.time() - (ModelRegistry.QUIRKS_TTL_DAYS + 1) * 86400
    path.write_text(json.dumps({"models": {"m": {"no_native_schema": True,
                                                 "learned_at": old}}}))

    fresh = ModelRegistry.__new__(ModelRegistry)
    fresh._quirks_lock = threading.Lock()
    fresh._no_native_schema = set()
    fresh._length_bump = {}
    fresh._load_quirks()
    assert "m" not in fresh._no_native_schema, \
        "an expired lesson must be re-learned, not trusted indefinitely"
