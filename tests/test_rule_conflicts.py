"""Rule contradictions are MEASURED on the corpus, not inferred from wording.

`_dedupe_rules` compares structurally — identical trigger on an identical class
pair, or 85% text similarity for prose. That misses the case that actually
reaches an annotator: two rules worded differently, firing on overlapping
queries, pointing at different classes.

Text similarity is the wrong instrument and this codebase has the scar: comparing
rendered `when` sentences once shredded 32 of 41 rules on a live run, because two
markers for ONE boundary render as near-identical templates — and a marker pair
pointing opposite ways is exactly what settling a boundary looks like.
"""
from __future__ import annotations

from types import SimpleNamespace

from qmine.ops.rule_conflict import find_conflicts


def R(rid, when, then, classes, trigger=""):
    return SimpleNamespace(id=rid, when=when, then=then, classes=classes, trigger=trigger)


QUERIES = (
    ["买手机推荐"] * 200                       # matches neither
    + ["高中录取分数线是多少"] * 100            # matches BOTH triggers below
    + ["几号放暑假"] * 80                       # matches only A
    + ["学校招生条件"] * 60                     # matches only B
)


def test_two_rules_that_fire_on_the_same_rows_and_disagree_are_found():
    """The live39 case: R018/R019 co-fired on 301 rows and pointed opposite ways.

    Their wording differs, so no text comparison could see it; running both
    triggers over the corpus makes it arithmetic.
    """
    rules = [
        R("A", "holiday or exam timing", "EXAM_HOLIDAY", ["EXAM_HOLIDAY", "SCHOOL_INFO"],
          trigger="分数线|放暑假"),
        R("B", "school identity and admission", "SCHOOL_INFO", ["EXAM_HOLIDAY", "SCHOOL_INFO"],
          trigger="分数线|招生"),
    ]
    rep = find_conflicts(rules, QUERIES, min_rows=25)

    assert len(rep.overlaps) == 1, [o.as_record() for o in rep.overlaps]
    o = rep.overlaps[0]
    assert o.n_both == 100, o.n_both
    assert {o.then_a, o.then_b} == {"EXAM_HOLIDAY", "SCHOOL_INFO"}
    assert o.examples and "分数线" in o.examples[0]


def test_a_legitimate_discriminating_pair_is_not_flagged():
    """Different markers on one boundary pointing opposite ways is the POINT.

    On live39 R006/R007 co-fired on 4 rows out of 49,999 — incidental, not a
    contradiction. Flagging that shape is what destroyed 32 of 41 rules before.
    """
    rules = [
        R("A", "word lists", "WORD_FORMATION", ["WORD_FORMATION", "SENTENCE"], trigger="组词"),
        R("B", "sentence material", "SENTENCE", ["WORD_FORMATION", "SENTENCE"], trigger="造句"),
    ]
    rep = find_conflicts(rules, ["组词大全"] * 100 + ["造句练习"] * 100, min_rows=25)
    assert rep.overlaps == []


def test_rules_that_agree_are_never_a_conflict():
    """Two routes to the same answer is redundancy at worst, not contradiction."""
    rules = [
        R("A", "x", "SAME", ["P", "Q"], trigger="分数线"),
        R("B", "y", "SAME", ["P", "Q"], trigger="分数线|招生"),
    ]
    assert find_conflicts(rules, QUERIES, min_rows=1).overlaps == []


def test_prose_rules_are_not_measurable_and_are_not_guessed_at():
    """No executable predicate means no measurement — and no invented verdict."""
    rules = [
        R("A", "some prose condition", "P", ["P", "Q"]),
        R("B", "different prose condition", "Q", ["P", "Q"]),
    ]
    rep = find_conflicts(rules, QUERIES)
    assert rep.overlaps == []
    assert rep.n_measurable == 0


def test_a_contested_boundary_is_reported_by_its_crowding():
    """live39 put 14 rules on TEXT_INTERPRETATION x WORD_MEANING_LOOKUP, 9 one
    way and 5 the other, 13 of them prose. Their overlap cannot be measured, so
    the count is the only signal — and it says the boundary, not any one rule,
    is the problem."""
    rules = [R(f"r{i}", "prose", "P" if i % 2 else "Q", ["P", "Q"]) for i in range(6)]
    rep = find_conflicts(rules, QUERIES, crowded_at=5)

    assert len(rep.crowded_pairs) == 1
    c = rep.crowded_pairs[0]
    assert c["n_rules"] == 6 and c["n_with_trigger"] == 0
    assert sorted(c["distinct_targets"]) == ["P", "Q"]


def test_a_broken_trigger_regex_is_reported_not_crashed():
    rules = [R("bad", "x", "P", ["P", "Q"], trigger="([unclosed"),
             R("ok", "y", "Q", ["P", "Q"], trigger="分数线")]
    rep = find_conflicts(rules, QUERIES)
    assert rep.bad_regex == ["bad"]
    assert rep.n_measurable == 1


def test_incidental_co_hits_are_below_the_floor():
    """A handful of co-hits is not a contradiction anyone will actually meet."""
    rules = [R("A", "x", "P", ["P", "Q"], trigger="分数线"),
             R("B", "y", "Q", ["P", "Q"], trigger="分数线")]
    assert find_conflicts(rules, QUERIES, min_rows=500).overlaps == []
    assert find_conflicts(rules, QUERIES, min_rows=10).overlaps != []
