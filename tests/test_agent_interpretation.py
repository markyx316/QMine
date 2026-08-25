"""Guardrails on agent-written prose in the deliverables.

Agents were doing 3.7% of the work on the bottom-up path (36 of 966 calls on
live38, all in P7 after every decision was already made) and 0% of the report
writing. Bringing them into the deliverable is only safe if a fabricated number
cannot survive to the page, so these tests pin the refusal, not the feature.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from qmine.agents.interpret import interpret
from qmine.agents.verify import check_numbers, extract_claims, fact_sheet

FACTS = {"n_rows": 49999, "leaves": 36, "families": 12, "kappa": 0.8221,
         "unnamed_share": 0.09862, "frag": {"leaves": 2.479, "topdown_l1": 1.319}}


def _deps(outs):
    """A deps stub whose interpreter returns `outs` in order."""
    events = []
    seen = []

    class FakeAgent:
        role = "interpreter"
        model = "test-model"

        def __init__(self, ctx, suffix=""): pass

        def run(self, **kw):
            seen.append(kw)
            o = outs[min(len(seen) - 1, len(outs) - 1)]
            if isinstance(o, Exception):
                raise o
            return SimpleNamespace(reading=o[0], caveats=o[1], unavailable=[])

    return SimpleNamespace(emit=events.append, agent_ctx=lambda: None), events, seen, FakeAgent


def _patch(monkeypatch, FakeAgent):
    import qmine.agents.roles as roles
    monkeypatch.setattr(roles, "InterpreterAgent", FakeAgent)


def test_a_fabricated_number_never_reaches_the_page(monkeypatch):
    """The whole point. An unverifiable number must not ship, ever."""
    deps, events, seen, FA = _deps([("κ 为 0.87, 覆盖 50,000 条。", [])] * 3)
    _patch(monkeypatch, FA)

    got = interpret(deps, "解释一致性", FACTS, max_attempts=3)

    assert not got.ok, "prose containing 0.87 and 50,000 was accepted"
    assert got.as_markdown() == "", "a rejected interpretation must render as nothing"
    assert got.attempts == 3
    assert any("abandoned" in e for e in events), events


def test_a_rejection_tells_the_agent_exactly_what_was_wrong(monkeypatch):
    """Re-asking without the specific failure is intrinsic self-correction.

    Huang et al. (ICLR 2024) find that configuration does not improve reasoning
    and can degrade it; the fix is external feedback, which here means quoting
    the offending numbers back.
    """
    deps, events, seen, FA = _deps([
        ("κ 为 0.87。", []),
        ("κ 为 0.8221, 共 49,999 条。", []),
    ])
    _patch(monkeypatch, FA)

    got = interpret(deps, "解释一致性", FACTS, max_attempts=3)

    assert got.ok and got.attempts == 2
    assert seen[0]["rejected"] == "", "the first attempt cannot have feedback yet"
    assert "0.87" in seen[1]["rejected"], (
        f"the retry did not name the offending number: {seen[1]['rejected']!r}"
    )


def test_a_fabricated_number_in_a_caveat_is_caught_too(monkeypatch):
    """A bullet is not a safer place for an invented number than a paragraph."""
    deps, events, seen, FA = _deps([("共 36 个叶。", ["注意 κ 仅 0.55"])] * 2)
    _patch(monkeypatch, FA)

    got = interpret(deps, "解释叶数", FACTS, max_attempts=2)

    assert not got.ok, "0.55 in a caveat was accepted"


def test_verified_prose_discloses_that_an_agent_wrote_it(monkeypatch):
    """A reader must be able to tell authored text from templated text."""
    deps, events, seen, FA = _deps([("共 36 个叶, 12 个家族。", ["碎裂度 2.479 需与簇数并读"])])
    _patch(monkeypatch, FA)

    got = interpret(deps, "解释交付形状", FACTS)
    md = got.as_markdown()

    assert got.ok
    assert "interpreter@test-model" in md, f"authorship not disclosed: {md}"
    assert "核验" in md, "the verification claim is not stated to the reader"
    assert "2.479" in md


def test_the_fact_sheet_the_agent_sees_is_the_one_the_checker_uses(monkeypatch):
    """A sheet the checker does not share is a guarantee about nothing."""
    deps, events, seen, FA = _deps([("共 36 个叶。", [])])
    _patch(monkeypatch, FA)

    interpret(deps, "q", FACTS)

    sheet = seen[0]["facts"]
    assert sheet == fact_sheet(FACTS)
    for token in ("leaves = 36", "kappa = 0.8221", "frag.topdown_l1 = 1.319"):
        assert token in sheet, f"{token!r} missing from the sheet the agent saw"


def test_section_numbers_are_not_treated_as_claims():
    """Otherwise every 「第 3 阶段」 fails the check and the guardrail gets widened
    until it stops catching anything."""
    assert extract_claims("第 3 阶段, §2.2, 见 1. 概述, 2026 年") == []
    assert check_numbers("详见第 3 阶段与 §2.2。", FACTS).ok


@pytest.mark.parametrize("text,ok", [
    ("49,999 条", True), ("50,000 条", False),      # counts must be exact
    ("κ=0.82", True), ("κ=0.80", False),           # honest rounding vs over-rounding
    ("9.9%", True), ("10%", True), ("12%", False),  # a share may round
    ("36 个叶", True), ("35 个叶", False),
])
def test_rounding_is_allowed_but_rewriting_is_not(text, ok):
    assert check_numbers(text, FACTS).ok is ok
