"""Holding the referee's rules to the verdicts the referee itself handed down.

77% of live39's shipped rules carry no executable trigger. The obvious reading —
the referee forgot, so ask it harder — is wrong, and acting on it would have made
things worse: its rules describe *semantic* conditions ("when the query is a
proverb and the user wants its moral"), and only 1 of the 80 trigger-less rules
contained an extractable marker. A regex demanded from the other 79 would be a
fabricated predicate, and every overlap measured through it would be an artifact
of the regex rather than a fact about the rules.

What a rule without a predicate can still be held to is the gold set. live39's
`OTHER x TEXT_INTERPRETATION`: the referee ruled TEXT_INTERPRETATION on 15 of 21
rows and drafted five rules sending marker-less queries to OTHER — contradicting,
five times, what it had just done on the rows in front of it.
"""
from __future__ import annotations

from types import SimpleNamespace

from qmine.ops.rule_conflict import (
    NO_MARKER, find_conflicts, rules_against_evidence, validate_trigger,
)


def _rule(rid, then, classes, trigger="", because="", examples=(), when=None):
    return SimpleNamespace(id=rid, then=then, classes=list(classes), trigger=trigger,
                           when=when if when is not None else f"rule {rid}",
                           added_because=because, examples=list(examples))


def _row(idx, a, b, final, adjudicated=True, query=None):
    return SimpleNamespace(idx=idx, query=query if query is not None else f"q{idx}",
                           label_a=a, label_b=b,
                           final=final, adjudicated=adjudicated, agreed=False)


# ------------------------------------------------------------- triggers
def test_a_no_marker_trigger_is_not_a_regex():
    """`re.compile("<no-marker>")` is VALID and matches nothing.

    So the sentinel produced an all-False mask, contributed to no overlap, and
    was still counted in `n_with_executable_trigger` — the tally of what we could
    measure inflated by rules we were measuring as nothing. Its real meaning is
    the negation of its own boundary.
    """
    qs = ["什么意思的查询", "一句没有标记的成语", "读音是什么"]
    marker = _rule("R1", "A", ["A", "B"], trigger="什么意思|读音")
    default = _rule("R2", "B", ["A", "B"], trigger=NO_MARKER)

    masks, bad = _pair_masks_of([marker, default], qs)
    assert bad == []
    assert masks["R1"].tolist() == [True, False, True]
    assert masks["R2"].tolist() == [False, True, False], (
        "the boundary default must fire where NO marker of its own pair fires")


def _pair_masks_of(rules, qs):
    from qmine.ops.rule_conflict import _pair_masks

    return _pair_masks(rules, qs)


def test_a_no_marker_rule_alone_on_its_boundary_is_left_unmeasured():
    """"Carries no marker" with no marker rule to negate has no referent.

    Treating it as "fires on everything" would swamp every overlap on the pair
    with a rule that means nothing — a measurement about the sentinel, not the
    rules.
    """
    masks, _ = _pair_masks_of([_rule("R2", "B", ["A", "B"], trigger=NO_MARKER)], ["x", "y"])
    assert "R2" not in masks


def test_a_trigger_must_fire_on_the_rules_own_evidence():
    """The cheapest way to catch a regex that describes something else.

    Every referee rule records the query whose disagreement created it. A trigger
    that does not match it is not the rule's condition, whatever it looks like.
    """
    qs = ["夸朋友女儿青出于蓝", "什么意思", "读音"]
    ok = validate_trigger("青出于蓝", qs, evidence=["夸朋友女儿青出于蓝"])
    assert ok.ok and ok.n_hits == 1

    wrong = validate_trigger("读音", qs, evidence=["夸朋友女儿青出于蓝"])
    assert not wrong.ok and "own example" in wrong.reason


def test_a_trigger_that_matches_everything_or_nothing_is_refused():
    """Both make every pair of rules look like it overlaps, or none of them."""
    qs = [f"查询第{chr(97 + i)}条" for i in range(20)]
    assert not validate_trigger("查询", qs).ok, "a trigger on 100% of the corpus"
    assert not validate_trigger("完全不存在的词", qs).ok, "a trigger on nothing"
    assert not validate_trigger("[unclosed", qs).ok, "a trigger that does not compile"
    ok = validate_trigger("第a条|第b条", qs)
    assert ok.ok and ok.n_hits == 2, ok


def test_a_rejected_trigger_makes_a_rule_semantic_not_broken():
    """It keeps every other guarantee; it stops claiming a predicate it lacks."""
    qs = [f"q{i}" for i in range(50)]
    rules = [_rule("R1", "A", ["A", "B"], trigger="q", because="disagreement on 'q1'")]
    rows = [_row(i, "A", "B", "A") for i in range(10)]
    rep = rules_against_evidence(rules, rows, qs)
    assert rep.n_lexical == 0 and rep.n_semantic == 1
    assert rep.n_rejected_triggers == 1
    assert rep.rejected[0]["rule"] == "R1"


# ------------------------------------------------------- rules vs verdicts
def test_the_live39_boundary_is_found_by_its_stated_ground_not_by_direction():
    """The live39 finding, on the instrument that survived adversarial review.

    The original check — decisive majority AND most rules pointing the other way —
    reported this boundary and was RIGHT FOR THE WRONG REASON, in three measured
    ways. It was confounded by the drafting rate (5 of 6 minority rows produced a
    rule against 1 of 15 majority rows, which is what a healthy exception set
    looks like). It had no resolution: 15/21 gives a Wilson lower bound of 0.5004
    against a bar of 0.5, so a single row decided whether it reported at all. And
    its magnitude of "five rules" was one drafting template emitted five times.

    What is actually wrong is checkable without any of that: the rules name
    什么意思 / 寓意 / 翻译 as the test, and no row on the boundary carries any of
    them. The direction tally is still recorded — as context, never a verdict.
    """
    pair = ["OTHER", "TEXT_INTERPRETATION"]
    when = "当查询无明确意图标记（如'什么意思'、'寓意'、'翻译'等）时，归入OTHER类。"
    rules = ([_rule("R029", "TEXT_INTERPRETATION", pair,
                    when="当查询为谚语、俗语、诗句或哲理语句时，归为TEXT_INTERPRETATION。",
                    because="disagreement on 'maj0'")]
             + [_rule(f"R{rid}", "OTHER", pair, when=when,
                      because=f"disagreement on 'min{j}'")
                for j, rid in enumerate((98, 100, 132, 142, 151))])
    rows = ([_row(i, *pair, "TEXT_INTERPRETATION", query="善心结善缘,善缘结善果")
             for i in range(15)]
            + [_row(15 + i, *pair, "OTHER", query=f"min{i}") for i in range(6)])
    rows[0].query = "maj0"

    rep = rules_against_evidence(rules, rows, codes=pair)

    assert len(rep.vacuous_grounds) == 1, "the stated ground must be the thing that fires"
    g = rep.vacuous_grounds[0]
    assert g.n_matching == 0 and g.n_rows == 21
    assert sorted(g.rules_citing) == ["R100", "R132", "R142", "R151", "R98"]

    assert rep.contradicted == [], "the direction verdict is retired"
    b = rep.boundaries[0]
    assert len(b.rules_against_majority) == 5, "the tally is still recorded as context"
    assert b.direction_is_confounded, "and its confound is measured alongside it"


def test_one_rule_pointing_at_the_minority_is_an_exception_not_a_defect():
    """The confound this measurement is designed around.

    A rule is conditional: "when <condition>, choose T". A rule carving out a
    genuine minority exception SHOULD point away from the pair's majority. A
    per-rule "disagrees with the majority" score would call every legitimate
    exception a defect — a false finding of exactly the kind this project keeps
    having to unlearn. Aggregating over the boundary is what washes it out.
    """
    pair = ["A", "B"]
    rules = [_rule("R1", "A", pair), _rule("R2", "A", pair), _rule("R3", "B", pair)]
    rows = [_row(i, *pair, "A") for i in range(18)] + [_row(18 + i, *pair, "B") for i in range(3)]

    rep = rules_against_evidence(rules, rows)
    assert rep.contradicted == [], "a lone exception must not be reported as a contradiction"
    assert rep.boundaries[0].decisive


def test_an_evenly_split_boundary_carries_no_direction_to_contradict():
    """The referee split it 11/10. There is nothing for the rules to disagree with."""
    pair = ["A", "B"]
    rules = [_rule(f"R{i}", "B", pair) for i in range(4)]
    rows = [_row(i, *pair, "A") for i in range(11)] + [_row(11 + i, *pair, "B") for i in range(10)]

    rep = rules_against_evidence(rules, rows)
    assert not rep.boundaries[0].decisive
    assert rep.contradicted == []


def test_a_boundary_with_too_few_rows_says_nothing_at_all():
    """A direction read off three rows is noise, whichever way it points."""
    pair = ["A", "B"]
    rep = rules_against_evidence([_rule("R1", "B", pair)],
                                 [_row(i, *pair, "A") for i in range(3)])
    assert rep.boundaries == []


def test_a_row_nobody_adjudicated_is_not_evidence():
    """Agreed rows never reached the referee, so they are not its verdicts."""
    pair = ["A", "B"]
    rows = [_row(i, *pair, "A", adjudicated=False) for i in range(20)]
    assert rules_against_evidence([_rule("R1", "B", pair)], rows).boundaries == []


def test_a_lexical_rule_is_judged_on_the_rows_its_trigger_claims():
    """The conditional test, which escapes the exception confound entirely.

    The trigger says which rows the rule is about, so the referee's verdicts on
    exactly those rows can be read directly — no aggregation needed.
    """
    qs = ["带标记词甲" if i < 4 else f"其他{i}" for i in range(40)]
    pair = ["A", "B"]
    rules = [_rule("R1", "A", pair, trigger="标记词甲", examples=["带标记词甲"])]
    # Rows 0-3 carry the marker; the referee sent them all to B, against the rule.
    rows = ([_row(i, *pair, "B") for i in range(4)]
            + [_row(4 + i, *pair, "A") for i in range(12)])

    rep = rules_against_evidence(rules, rows, qs)
    assert rep.n_lexical == 1
    chk = [c for c in rep.lexical_rules if c["rule"] == "R1"][0]
    assert chk["n_rows_trigger_fired"] == 4
    assert chk["agreement"] == 0.0, "the referee disagreed on every row the rule claims"
    # And the boundary itself is NOT contradicted — the majority went the rule's way.
    assert rep.contradicted == []


def test_nothing_is_deleted_or_rewritten():
    """Withholding is what caused the 32-of-41 disaster.

    The report is a description. A rule set handed in comes back byte-identical.
    """
    pair = ["A", "B"]
    rules = [_rule("R1", "B", pair) for _ in range(4)]
    before = [(r.id, r.then, r.trigger) for r in rules]
    rules_against_evidence(rules, [_row(i, *pair, "A") for i in range(20)])
    assert [(r.id, r.then, r.trigger) for r in rules] == before


def test_find_conflicts_still_measures_overlap_on_the_corpus():
    """The original mechanism, unchanged by the boundary-default rework."""
    qs = ["什么意思啊" for _ in range(60)] + ["读音" for _ in range(60)]
    rules = [_rule("R1", "A", ["A", "B"], trigger="什么意思"),
             _rule("R2", "B", ["A", "B"], trigger="意思")]
    rep = find_conflicts(rules, qs, min_rows=25)
    assert len(rep.overlaps) == 1 and rep.overlaps[0].n_both == 60
    assert rep.n_measurable == 2


# ------------------------------------------------- `then` must BE a class
def test_a_then_holding_a_sentence_is_reduced_to_the_class_it_names():
    """`then` is documented as "the class that wins" and live38 shipped 18 rules
    where it held a whole instruction instead. The old validator asked whether
    the field *mentions* a real class, which a sentence satisfies."""
    from qmine.ops.rule_conflict import normalise_then

    codes = ["JUDGE_LANGUAGE_USAGE", "LOOKUP_CHAR_PRONUNCIATION", "OTHER"]
    r = normalise_then("归 JUDGE_LANGUAGE_USAGE。", codes)
    assert r.is_key and r.code == "JUDGE_LANGUAGE_USAGE"
    assert r.original == "归 JUDGE_LANGUAGE_USAGE。", "the sentence must survive for the rationale"

    assert normalise_then("OTHER", codes).code == "OTHER"


def test_a_then_naming_two_classes_is_refused_rather_than_guessed():
    """`归 A，不归 B` means A, but `有裁决框架的归 A；单纯问 X 的归 B` is two rules
    in one sentence — and nothing mechanical separates them. live38 had 17 of
    these, every one an `X_VS_Y` rule describing BOTH directions of a boundary.
    Picking a side would silently rewrite the rule.
    """
    from qmine.ops.rule_conflict import normalise_then

    codes = ["JUDGE_LANGUAGE_USAGE", "LOOKUP_CHAR_PRONUNCIATION"]
    r = normalise_then("归 JUDGE_LANGUAGE_USAGE，不归 LOOKUP_CHAR_PRONUNCIATION。", codes)
    assert not r.is_key and len(r.found) == 2
    assert "cannot be used as a key" in r.note


def test_a_class_code_is_matched_on_token_boundaries():
    """Two independent defences, tested where only ONE of them can protect.

    Found by mutation: removing either the word-boundary anchor or the
    maximal-match filter left the obvious cases passing, because each rescued
    the other. Redundancy is the right design; a test that cannot tell them
    apart proves neither.
    """
    from qmine.ops.rule_conflict import normalise_then

    # Only the BOUNDARY protects here: nothing contains the false match, so the
    # maximal filter has nothing to strip. Resolving this to OTHER would send a
    # rule to a class its author never named.
    assert normalise_then("这属于 ANOTHER 的情况", ["OTHER", "MATH"]).code is None

    # Only the MAXIMAL filter protects here: with `-` as a separator both codes
    # clear the boundary check, so the containment test is what stops `BAR` from
    # being counted alongside `FOO-BAR` and making a single class look like two.
    r = normalise_then("归 FOO-BAR", ["FOO-BAR", "BAR"])
    assert r.code == "FOO-BAR", r

    # And the ordinary cases both defences agree on.
    assert normalise_then("ANOTHER_THING", ["OTHER", "ANOTHER_THING"]).code == "ANOTHER_THING"
    assert normalise_then("归 WORD_MEANING_LOOKUP",
                          ["WORD_MEANING", "WORD_MEANING_LOOKUP"]).code == "WORD_MEANING_LOOKUP"


def test_a_prose_then_would_be_counted_against_every_boundary_it_touches():
    """THE reason this matters, constructed rather than hoped for.

    The measurement is `rule.then == the referee's majority verdict`. A `then`
    holding a sentence can never equal a class code, so without the class list
    every prose rule counts as pointing AGAINST the evidence — and a boundary
    whose rules all agree with the referee is reported contradicted.

    Measured on live38's real rules: 3 contradicted boundaries WITHOUT the class
    list, all three false, and 0 WITH it.
    """
    pair = ["A", "B"]
    rules = [
        _rule("R1", "A", pair),                                  # agrees with the evidence
        _rule("R2", "归 A，除非查询含标记词才归 B。", pair),          # a sentence, not a key
        _rule("R3", "归 A，除非另有说明才归 B。", pair),              # another sentence
    ]
    rows = [_row(i, *pair, "A") for i in range(20)]

    blind = rules_against_evidence(rules, rows)
    assert len(blind.boundaries[0].rules_against_majority) == 2, (
        "without the class list two sentences are counted as pointing away")

    keyed = rules_against_evidence(rules, rows, codes=["A", "B"])
    b = keyed.boundaries[0]
    assert b.rules_toward_majority == ["R1"] and b.rules_against_majority == []
    assert sorted(b.rules_not_a_key) == ["R2", "R3"], "excluded, and said so"


def test_find_conflicts_does_not_call_two_sentences_a_disagreement():
    """`_dedupe_rules` compares `then` with `==`, and two phrasings of one answer
    read as a CONTRADICTION — observed live on R112 vs R053, where a hallucinated
    code variant cost two valid rules. The overlap measurement must not repeat it.
    """
    qs = ["什么意思啊" for _ in range(60)] + ["读音" for _ in range(60)]
    rules = [_rule("R1", "归 A 类。", ["A", "B"], trigger="什么意思"),
             _rule("R2", "应归入 A。", ["A", "B"], trigger="意思")]

    blind = find_conflicts(rules, qs, min_rows=25)
    assert len(blind.overlaps) == 1, "two phrasings of one answer look like a conflict"

    keyed = find_conflicts(rules, qs, min_rows=25, codes=["A", "B"])
    assert keyed.overlaps == [], "neither resolves to a class, so 'do they disagree' has no answer"
    assert sorted(set(keyed.unkeyed_then)) == ["R1", "R2"]


# ============================================================
# The instrument that replaced counting rule directions
# ============================================================
def test_counting_rule_directions_is_confounded_by_the_drafting_rate():
    """Why the original verdict was retired, encoded so it cannot come back.

    A referee drafts a rule only where it judges the guide to have FAILED, and
    that concentrates on the side that goes against the prevailing pattern.
    Measured on live39's `OTHER x TEXT_INTERPRETATION`: 5 of 6 minority rows
    produced a rule (83%) against 1 of 15 majority rows (7%). So "most of this
    boundary's rules point away from the majority" is the EXPECTED shape of a
    healthy exception set, and a check built on it fires on a guide with no
    defect — this codebase's own named trap, testing a mechanism with a
    distribution it has already filtered.
    """
    pair = ["A", "B"]
    rows = ([_row(i, *pair, "A", query=f"maj{i}") for i in range(15)]
            + [_row(15 + i, *pair, "B", query=f"min{i}") for i in range(6)])
    # A perfectly healthy guide: one rule per minority row, one for the majority.
    rules = ([_rule("R0", "A", pair, because="disagreement on 'maj0'")]
             + [_rule(f"R{i+1}", "B", pair, because=f"disagreement on 'min{i}'")
                for i in range(5)])

    rep = rules_against_evidence(rules, rows, codes=pair)
    b = rep.boundaries[0]
    assert len(b.rules_against_majority) == 5 and len(b.rules_toward_majority) == 1
    assert b.direction_is_confounded, (
        "5-of-6 vs 1-of-15 drafting must be recognised as a confound")
    assert rep.contradicted == [], "the direction verdict must stay retired"


def test_a_stated_ground_that_divides_no_row_is_flagged():
    """The measurement that replaced it, and the live39 case it was built on.

    Five rules named 什么意思 / 寓意 / 翻译 as the discriminator, and NOT ONE of
    the boundary's 21 adjudicated queries contains any of them — including all 15
    the referee ruled the other way. An annotator applying the rule literally is
    routed to the minority class on every row.
    """
    from qmine.ops.rule_conflict import stated_grounds

    pair = ["OTHER", "TEXT_INTERPRETATION"]
    when = "当查询无明确意图标记（如'什么意思'、'寓意'、'翻译'等）时，归入OTHER类。"
    rules = [_rule(f"R{i}", "OTHER", pair, when=when) for i in range(5)]
    rows = ([_row(i, *pair, "TEXT_INTERPRETATION", query="善心结善缘,善缘结善果")
             for i in range(15)]
            + [_row(15 + i, *pair, "OTHER", query="凡事论迹不论心") for i in range(6)])

    g = stated_grounds(rules, rows)[0]
    assert sorted(g.markers) == ["什么意思", "寓意", "翻译"]
    assert g.n_matching == 0 and g.n_rows == 21
    assert not g.separates, "a ground every row falls on one side of divides nothing"


def test_a_stated_ground_that_does_divide_the_boundary_passes():
    """The instrument must not simply flag everything.

    Measured on live39 this is the real contrast: the SAME class pair vocabulary
    separates `TEXT_INTERPRETATION x WORD_MEANING_LOOKUP` (33 of 39 rows carry a
    marker) and separates nothing on `OTHER x TEXT_INTERPRETATION` (0 of 21).
    """
    from qmine.ops.rule_conflict import stated_grounds

    pair = ["A", "B"]
    when = "当查询包含（如'什么意思'、'含义'等）时，归入A。"
    rules = [_rule("R1", "A", pair, when=when)]
    rows = ([_row(i, *pair, "A", query="这个词什么意思") for i in range(10)]
            + [_row(10 + i, *pair, "B", query="一句没有标记的话") for i in range(8)])

    g = stated_grounds(rules, rows)[0]
    assert g.separates and g.n_matching == 10
    assert g.verdicts_when_present == {"A": 10}
    assert g.verdicts_when_absent == {"B": 8}


def test_a_marker_enumeration_is_split_and_a_template_is_refused():
    """`怎么读/意思/翻译/部首` is four markers; testing it whole matches nothing
    and would report a vacuous ground that is really an unusable pattern.
    `X的意思` can never match a query at all."""
    from qmine.ops.rule_conflict import usable_markers

    ok, bad = usable_markers(["怎么读/意思/翻译/部首"])
    assert set(ok) == {"意思", "怎么读", "翻译", "部首"} and bad == []

    ok2, bad2 = usable_markers(["X的意思", "X读yun还是jun"])
    assert ok2 == [] and len(bad2) == 2


def test_an_example_query_is_not_mistaken_for_a_marker():
    """Rules quote two different things in `when` — an enumerated discriminator
    and an example query. Counting the second as a marker makes a boundary look
    like its ground separates when it matched one arbitrary row."""
    from qmine.ops.rule_conflict import usable_markers

    ok, bad = usable_markers(["韩食 唐 韩翃", "4章逐节注解", "同音字50个",
                              "两三个人聚集在一起诗歌", "什么意思"])
    assert ok == ["什么意思"], ok
    assert len(bad) == 4


def test_a_boundary_with_no_usable_marker_gets_no_verdict():
    """Reporting "the ground separates nothing" when we could not build the test
    would be a fact about our extraction, not about the guide — the same
    fabricated-predicate trap `validate_trigger` exists to avoid."""
    from qmine.ops.rule_conflict import stated_grounds

    pair = ["A", "B"]
    rules = [_rule("R1", "A", pair, when="当查询满足'X的意思'时，归入A。")]
    rows = [_row(i, *pair, "A", query=f"q{i}") for i in range(20)]
    assert stated_grounds(rules, rows) == []


def test_a_count_and_the_list_beside_it_describe_the_same_thing():
    """Found live on live40 by the p2b observer, and CONFIRMED by its own check.

    `n_triggers_rejected` was the full count while `rejected_triggers` was
    truncated to 12, so the artifact said 13 next to a list of 12. A reader
    comparing them sees a contradiction and cannot tell whether the count or the
    list is wrong.
    """
    qs = [f"q{chr(97 + i % 26)}{i}" for i in range(400)]
    # 15 rules whose triggers all fire on the whole corpus -> all rejected.
    rules = [_rule(f"R{i}", "A", ["A", "B"], trigger="q") for i in range(15)]
    rows = [_row(i, "A", "B", "A") for i in range(20)]

    rec = rules_against_evidence(rules, rows, qs, codes=["A", "B"]).as_record()
    assert rec["n_triggers_rejected"] == 15
    assert len(rec["rejected_triggers"]) == 12
    assert rec["rejected_triggers_truncated"] == 3, (
        "the artifact must say how many entries were cut")
    assert (len(rec["rejected_triggers"]) + rec["rejected_triggers_truncated"]
            == rec["n_triggers_rejected"])


def test_the_two_unkeyed_then_lists_are_named_for_different_populations():
    """Also found live by the p2b observer.

    `find_conflicts` records the unkeyed rules it met while comparing candidate
    PAIRS; `rules_against_evidence` records every unkeyed rule on a measured
    boundary. Both were called `rules_whose_then_is_not_a_class`, so the two
    artifacts appeared to contradict each other about the same fact.
    """
    qs = ["什么意思啊"] * 60 + ["读音"] * 60
    rules = [_rule("R1", "归 A 类。", ["A", "B"], trigger="什么意思"),
             _rule("R2", "应归入 A。", ["A", "B"], trigger="意思"),
             _rule("R3", "归 A 或 B。", ["A", "B"])]
    rows = [_row(i, "A", "B", "A") for i in range(20)]

    conf = find_conflicts(rules, qs, min_rows=25, codes=["A", "B"]).as_record()
    ev = rules_against_evidence(rules, rows, qs, codes=["A", "B"]).as_record()

    assert "rules_in_compared_pairs_whose_then_is_not_a_class" in conf
    assert "rules_on_measured_boundaries_whose_then_is_not_a_class" in ev
    assert "rules_whose_then_is_not_a_class" not in conf
    assert "rules_whose_then_is_not_a_class" not in ev, (
        "one name over two populations is what made this read as a contradiction")
