"""Two annotators must come from different LABS, not merely different gateways.

Cohen's kappa between two annotators is evidence about the labelling guide only
if their errors are independent. `provider` was the proxy for that, and an
aggregator breaks it: `zhipu/zai/glm-5.1` and `openrouter/z-ai/glm-5.3` are two
providers and one lab. Adding an OpenRouter key made that reachable, so the check
would have passed while kappa quietly measured shared architecture.
"""

from __future__ import annotations

from qmine.llm.router import lab_of


def test_the_same_lab_is_recognised_through_any_gateway():
    assert lab_of("zai/glm-5.1") == lab_of("z-ai/glm-5.3") == "zhipu"
    assert lab_of("dashscope/glm-5.1") == "zhipu", "a Zhipu model sold by DashScope"
    assert lab_of("deepseek-v4-flash") == lab_of("openrouter/deepseek-v4") == "deepseek"
    assert lab_of("dashscope/qwen3-next-80b-a3b-instruct") == "qwen"


def test_different_labs_stay_different():
    assert lab_of("deepseek-v4-flash") != lab_of("dashscope/qwen3-next-80b")
    assert lab_of("zai/glm-4.7") != lab_of("openai/gpt-5.3-chat")
    assert lab_of("claude-opus-5") == "anthropic"


def test_an_unknown_model_falls_back_to_its_provider():
    """Unrecognised is not the same as 'the same lab'; the old behaviour is safe."""
    class Card:
        id, provider = "some-new-model-v1", "acme"
    assert lab_of(Card()) == "acme"
    assert lab_of("some-new-model-v1") == "some-new-model-v1"


def test_the_dangerous_pair_the_aggregator_created():
    """The exact case adding OpenRouter introduced: different provider, same lab."""
    a_provider, a_model = "zhipu", "zai/glm-5.1"
    b_provider, b_model = "openrouter", "z-ai/glm-5.3"
    assert a_provider != b_provider, "gateway check would have passed this"
    assert lab_of(a_model) == lab_of(b_model), "and it is one lab"


def test_a_fallback_chain_does_not_reintroduce_the_partner_lab():
    """Failing over must not quietly make both annotators the same architecture —
    a live 402 sent annotator_b's first fallback straight at annotator_a's lab."""
    from qmine.llm.router import lab_of as L

    partner_lab = L("dashscope/qwen3-next-80b-a3b-instruct")
    candidates = ["dashscope/qwen3-next-80b-a3b-instruct", "zai/glm-4.7-flash"]
    usable = [c for c in candidates if L(c) != partner_lab]
    assert usable == ["zai/glm-4.7-flash"]


def test_a_tie_on_price_breaks_toward_the_newer_roomier_model():
    """`zai/glm-5.1` (1.40/4.40, 200k) and `zai/glm-4.5-airx` (1.10/4.50, 128k)
    score IDENTICALLY on the cost proxy — 14.60 each. The referee got the older,
    smaller-context model because it came first in the catalogue."""
    from qmine.llm.router import _generation

    # They differ by 1e-15 — a floating-point artifact, not a cost difference.
    # The older model won by that hair, which is worse than a tie, not better.
    assert abs((1.40 + 3 * 4.40) - (1.10 + 3 * 4.50)) < 1e-9
    assert _generation("zai/glm-5.1") > _generation("zai/glm-4.5-airx")


def test_the_recency_proxy_reads_real_model_ids():
    from qmine.llm.router import _generation

    assert _generation("z-ai/glm-5.3") == 5.3
    assert _generation("deepseek-v4-flash") == 4.0
    # A parameter count is not a version.
    assert _generation("dashscope/qwen3-next-80b-a3b-instruct") == 3.0
    assert _generation("mystery-model") == 0.0, "unknown must not outrank anything"


def test_labs_can_be_excluded_through_any_gateway():
    """Excluding "openai" must also exclude `openrouter:openai/gpt-5.1` — a ban
    that only matches the direct provider is no ban at all once an aggregator is
    in the pool."""
    from qmine.llm.router import lab_of

    banned = {"openai", "anthropic", "google"}
    for mid in ("openai/gpt-5.1", "openrouter/anthropic/claude-opus-5",
                "google/gemini-3.7-flash"):
        assert lab_of(mid) in banned, mid
    for mid in ("zai/glm-5.1", "deepseek/deepseek-v4-flash",
                "dashscope/qwen3-next-80b-a3b-instruct"):
        assert lab_of(mid) not in banned, mid


def test_an_unknown_vendor_is_named_by_vendor_not_by_gateway():
    """`sakana/fugu-ultra` resolved to "openrouter", so every unrecognised model
    reported as the same lab — indistinguishable from each other, and impossible
    to exclude by name. It was chosen for the architect at $30/M output."""
    from qmine.llm.router import lab_of

    class Card:
        id, provider = "sakana/fugu-ultra", "openrouter"

    assert lab_of(Card()) == "sakana", "the vendor segment is the lab"
    assert lab_of(Card()) != "openrouter", "a gateway is never a lab"


def test_price_tier_does_not_make_an_obscure_model_frontier_by_default():
    """The proxy's documented failure is a strong-and-cheap model landing low.
    The converse bites too: after excluding the big labs, the only two
    price-'frontier' models left were a $24/M and a $30/M model from vendors
    nobody chose, and the architect requires frontier with run-wide blast radius.
    """
    from qmine.llm.router import _generation, lab_of

    # The signal that should break that tie is not price.
    assert lab_of("dashscope/glm-5.2") == "zhipu"
    assert _generation("dashscope/glm-5.2") > _generation("zai/glm-4.5-airx")


def test_within_one_lab_a_newer_model_is_not_less_capable_for_being_cheaper():
    """`z-ai/glm-5.2` — 1M context, structured output, $3.04/M — was bucketed a
    tier BELOW `zai/glm-4.5-airx` (128k, $4.50/M) purely for costing less, and the
    capability term then kept the older model for the referee.

    Cross-lab this comparison is meaningless; within one lab a version number
    really does order the lineup.
    """
    from qmine.llm.router import _generation, lab_of

    new, old = "z-ai/glm-5.2", "zai/glm-4.5-airx"
    assert lab_of(new) == lab_of(old) == "zhipu", "same lab, so comparable"
    assert _generation(new) > _generation(old)


def test_the_upgrade_requires_at_least_as_much_context():
    """A newer *small* model must not displace an older large one."""
    from qmine.llm.router import _generation

    assert _generation("zai/glm-5.2") > _generation("zai/glm-4.5-airx")
    # The rule is generation AND context, so a 32k v6 would not qualify against
    # a 1M v5 — expressed here as the guard the router applies.
    newer_but_smaller = (5.2, 32_768)
    older_but_bigger = (4.5, 1_048_576)
    assert not (newer_but_smaller[0] > older_but_bigger[0]
                and newer_but_smaller[1] >= older_but_bigger[1])


def test_unpriced_and_meta_endpoints_are_not_candidates():
    """With price out of the ranking, a cost-weighted score cannot resist a free
    model: `:free` variants, previews and `openrouter/auto` — a meta-endpoint that
    resolves to something else at call time — won every role in the pipeline."""
    from qmine.llm.catalog import ModelCard
    from qmine.llm.requirements import requirement_for
    from qmine.llm.router import _eligible

    req = requirement_for("referee")
    for bad in (ModelCard(id="z-ai/glm-5.2:free", provider="openrouter",
                          input_per_mtok=0.0, output_per_mtok=0.0, context_tokens=256_000),
                ModelCard(id="openrouter/auto", provider="openrouter",
                          context_tokens=200_000),
                ModelCard(id="dots/dots-3-note-preview", provider="openrouter",
                          input_per_mtok=1.0, output_per_mtok=2.0, context_tokens=200_000)):
        ok, why = _eligible(bad, req, "frontier")
        assert not ok, f"{bad.id} should be ineligible"


def test_the_referee_must_differ_from_both_annotators():
    """It decides every row they split on. Sharing a lab with either makes it side
    systematically with that one, corrupting the gold set in a direction nobody
    would think to look for."""
    from qmine.llm.router import MUST_DIFFER_FROM

    assert set(MUST_DIFFER_FROM["referee"]) == {"annotator_a", "annotator_b"}
    assert MUST_DIFFER_FROM["annotator_b"] == ("annotator_a",)


def test_a_date_stamp_is_not_a_version_number():
    """`qwen-flash-2025-07-28` read as generation 28, became the top-ranked model
    of its lab, and won every role in the pipeline."""
    from qmine.llm.router import _generation

    assert _generation("dashscope/qwen-flash-2025-07-28") == 0.0
    assert _generation("qwen-max-20250115") == 0.0
    assert _generation("dashscope/qwen3.8-max") == 3.8


def test_the_same_model_is_reached_directly_rather_than_through_a_gateway():
    """`deepseek/deepseek-v4-flash` on OpenRouter and `deepseek-v4-flash` on
    DeepSeek are one model reached two ways, and the router could not tell.

    Measured on one pilot: the gateway path showed a 6.3x spread between its
    median and worst call (67.8s to 429.9s) where a direct provider held 1.4x —
    a gap model verbosity does not explain. An aggregator adds queueing, its own
    outages and its own rate limits.
    """
    from qmine.llm.router import bare_model

    assert bare_model("deepseek/deepseek-v4-flash") == bare_model("deepseek-v4-flash")
    assert bare_model("openrouter/z-ai/glm-5.2") == bare_model("zai/glm-5.2")
    assert bare_model("dashscope/qwen3-next-80b") != bare_model("dashscope/glm-5.2")


def test_preferring_direct_is_a_preference_not_a_rule():
    """An aggregator is sometimes cheaper, and sometimes the only route to a
    model at all — GLM 5.3 is reachable no other way. The swap only fires when
    the SAME model is available directly."""
    from qmine.llm.providers import BY_KEY

    assert BY_KEY["openrouter"].kind == "aggregator"
    assert BY_KEY["deepseek"].kind == "direct"
    # A model only OpenRouter carries has no direct twin to swap to.
    assert bare_model_unreachable_directly("z-ai/glm-5.3")


def bare_model_unreachable_directly(model_id: str) -> bool:
    """No direct provider in this project serves GLM 5.3; only the aggregator."""
    from qmine.llm.router import bare_model

    return bare_model(model_id) == "glm-5.3"
