"""Spend must be measured at the prices that chose the models.

`estimated_cost_usd` charged every token at a hardcoded $3/$15 per million —
frontier rates — while runs were on `deepseek-v4-flash` at $0.44/$1.32 and
`qwen3-next-80b` at $0.15/$1.20. Live32 was reported as $4.68 and cost $0.67.
Every cost figure this project ever quoted was inflated, and the planner — which
uses real catalogue prices — was blamed for the discrepancy.
"""

from __future__ import annotations

from qmine.llm.budget import UsageLedger


def test_spend_is_measured_at_the_routed_models_prices():
    led = UsageLedger()
    led.rates = {"annotator_b": (0.44, 1.32)}
    led.record("annotator_b", input_tokens=1_000_000, output_tokens=1_000_000)
    assert round(led.estimated_cost_usd(), 2) == 1.76          # 0.44 + 1.32
    assert round((1e6 * 3.0 + 1e6 * 15.0) / 1e6, 2) == 18.0    # what it used to say


def test_suffixed_roles_are_priced_by_their_base_role():
    """`researcher_log_reading` is routed as `researcher`. Exact-match lookup
    silently dropped every suffixed agent onto the frontier fallback — which is
    how four researcher roles were mispriced even after the first fix."""
    led = UsageLedger()
    led.rates = {"researcher": (1.10, 4.50)}
    led.record("researcher_log_reading", input_tokens=1_000_000, output_tokens=0)
    assert round(led.estimated_cost_usd(), 2) == 1.10
    assert led.unpriced_roles == []


def test_the_longest_matching_base_wins():
    led = UsageLedger()
    led.rates = {"annotator": (9.0, 9.0), "annotator_b": (0.44, 1.32)}
    led.record("annotator_b", input_tokens=1_000_000, output_tokens=0)
    assert round(led.estimated_cost_usd(), 2) == 0.44


def test_a_role_with_no_published_price_is_named_not_guessed():
    """A fallback rate is a guess; it must not read as a measurement."""
    led = UsageLedger()
    led.rates = {"annotator_a": (0.15, 1.20)}
    led.record("annotator_a", input_tokens=1_000, output_tokens=1_000)
    led.record("mystery_role", input_tokens=1_000, output_tokens=1_000)
    led.estimated_cost_usd()
    assert led.unpriced_roles == ["mystery_role"]


def test_no_prices_at_all_still_produces_a_number_and_says_so():
    """Offline and library use have no routing plan; the ledger must not crash."""
    led = UsageLedger()
    led.record("namer", input_tokens=1_000_000, output_tokens=0)
    assert round(led.estimated_cost_usd(), 2) == 3.0
    assert led.unpriced_roles == ["namer"]
