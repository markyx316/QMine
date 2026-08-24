"""The loop that redraws a taxonomy the pilot proved is not applicable.

Three live runs printed "these boundaries are broken" and halted, prescribing
nothing; a human redrew by hand. This loop closes that. It also spends money and
decides which taxonomy is delivered, and it is skipped offline — so without these
tests it could only ever be exercised by a paid run.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from qmine.graph.nodes import topdown
from qmine.records import Taxonomy, TaxonomyNode


def tax(*codes: str) -> Taxonomy:
    return Taxonomy(version="v1", nodes=[
        TaxonomyNode(code=c, name=c.lower(), level=1, definition=f"def {c}",
                     user_need="n", positive_examples=[f"{c}-1"], negative_examples=[])
        for c in codes])


def make(deps_events: list[str], *, offline: bool = False, redraws: int = 2):
    cfg = SimpleNamespace(taxonomy=SimpleNamespace(max_taxonomy_redraws=redraws,
                                                   l1_target_range=(2, 25)),
                          domain=SimpleNamespace(domain_notes=""))
    deps = SimpleNamespace(cfg=cfg, emit=deps_events.append,
                           registry=SimpleNamespace(is_offline=offline))
    return deps


def pilot(kappa: float, ceiling: float = 0.9, structural=(("A × B", 6),), sig: bool = True):
    return {"kappa": kappa, "n": 200, "self_consistency_kappa": ceiling,
            "structural_confusions": list(structural), "slack_is_significant": sig}


def install(monkeypatch, redrawn_codes, kappas):
    """Redraw returns `redrawn_codes`; each re-pilot returns the next kappa."""
    seq = iter(kappas)

    class FakeRedraw:
        def __init__(self, ctx): pass
        def run(self, **kw): return SimpleNamespace(nodes=tax(*redrawn_codes).nodes)

    monkeypatch.setattr(topdown, "TaxonomyRedrawAgent", FakeRedraw)
    monkeypatch.setattr(topdown, "_pilot_agreement",
                        lambda deps, ctx, df, t: pilot(next(seq)))


def test_an_improving_redraw_is_kept(monkeypatch):
    ev: list[str] = []
    install(monkeypatch, ["A", "C"], [0.85, 0.85])
    t, p, hist = topdown._redraw_until_stable(make(ev), None, None, tax("A", "B"), pilot(0.70))
    assert p["kappa"] == 0.85, "the improved pilot must be the one the gate sees"
    assert {n.code for n in t.nodes} == {"A", "C"}
    assert hist[0]["kept"] is True
    assert hist[0]["dropped"] == ["B"] and hist[0]["added"] == ["C"]


def test_a_redraw_that_lowers_kappa_is_reverted(monkeypatch):
    """Keeping a redraw because it is newer is how a loop walks a taxonomy downhill."""
    ev: list[str] = []
    install(monkeypatch, ["A", "C"], [0.55])
    before = tax("A", "B")
    t, p, hist = topdown._redraw_until_stable(make(ev), None, None, before, pilot(0.70))
    assert p["kappa"] == 0.70, "the gate must see the BETTER pilot, not the newer one"
    assert {n.code for n in t.nodes} == {"A", "B"}
    assert hist[0]["kept"] is False
    assert any("reverting" in m for m in ev)


def test_the_revert_restores_the_original_definitions(monkeypatch):
    """Filtering the REDRAWN list by the old codes would keep the new definitions
    under the old names — a revert that reverts nothing, and silently."""
    ev: list[str] = []

    class FakeRedraw:
        def __init__(self, ctx): pass
        def run(self, **kw):
            n = tax("A", "B").nodes
            n[0].definition = "REWRITTEN"          # same code, different content
            return SimpleNamespace(nodes=n)

    monkeypatch.setattr(topdown, "TaxonomyRedrawAgent", FakeRedraw)
    monkeypatch.setattr(topdown, "_pilot_agreement", lambda *a: pilot(0.40))
    t, _, _ = topdown._redraw_until_stable(make(ev), None, None, tax("A", "B"), pilot(0.70))
    assert t.nodes[0].definition == "def A", "the original definition must come back"


def test_the_loop_stops_when_there_is_nothing_structural_to_fix(monkeypatch):
    ev: list[str] = []
    install(monkeypatch, ["A", "C"], [0.99])
    _, p, hist = topdown._redraw_until_stable(
        make(ev), None, None, tax("A", "B"), pilot(0.70, structural=()))
    assert hist == [] and p["kappa"] == 0.70, "no structural pairs means nothing to redraw"


def test_the_loop_stops_when_the_slack_is_not_significant(monkeypatch):
    """At the annotator's ceiling there is nothing a redraw can recover."""
    ev: list[str] = []
    install(monkeypatch, ["A", "C"], [0.99])
    _, p, hist = topdown._redraw_until_stable(
        make(ev), None, None, tax("A", "B"), pilot(0.70, sig=False))
    assert hist == []


def test_the_loop_is_skipped_offline(monkeypatch):
    """Re-asking a deterministic stand-in in a different batch order measures its
    batching, not an annotator's reliability — every pair would look structural."""
    ev: list[str] = []
    install(monkeypatch, ["A", "C"], [0.99])
    _, p, hist = topdown._redraw_until_stable(
        make(ev, offline=True), None, None, tax("A", "B"), pilot(0.70))
    assert hist == []


def test_the_loop_is_bounded(monkeypatch):
    ev: list[str] = []
    install(monkeypatch, ["A", "C"], [0.71, 0.72, 0.73, 0.74])
    _, _, hist = topdown._redraw_until_stable(
        make(ev, redraws=2), None, None, tax("A", "B"), pilot(0.70))
    assert len(hist) == 2, "a loop that spends money must have a ceiling"


# --- the gate's reading of the redraw history ------------------------------
#
# `live36` halted with kappa 0.806 against a 0.9193 ceiling because redraw 1
# HELPED (0.781 -> 0.806) and redraw 2 was reverted (0.795). The loop was out of
# moves, but the gate asked `any(kept)` and concluded the remedy was still
# available. An improving redraw made the gate strictly harder to pass than a
# failing one — the opposite of what the remedy is for.


def test_a_redraw_that_helped_still_counts_as_a_spent_remedy(monkeypatch):
    """The live36 shape: helped, then reverted. The loop has no move left."""
    hist = [{"attempt": 1, "kept": True}, {"attempt": 2, "kept": False}]
    assert topdown._remedy_is_exhausted(hist, 2) is True


def test_an_improving_redraw_is_not_penalised_against_a_failing_one():
    """The invariant the defect broke: helping must never cost you the gate."""
    failed = [{"attempt": 1, "kept": False}]
    helped_then_failed = [{"attempt": 1, "kept": True}, {"attempt": 2, "kept": False}]
    assert topdown._remedy_is_exhausted(failed, 2) is True
    assert topdown._remedy_is_exhausted(helped_then_failed, 2) is True, (
        "a run whose agreement improved must not be halted where an identical "
        "run whose agreement fell would proceed")


def test_using_every_allowed_attempt_exhausts_the_remedy():
    """Two kept redraws and no attempts left is still 'no remaining move'."""
    assert topdown._remedy_is_exhausted(
        [{"attempt": 1, "kept": True}, {"attempt": 2, "kept": True}], 2) is True


def test_a_remedy_with_attempts_remaining_is_not_exhausted():
    """A kept redraw with budget left is not a reason to waive the slack."""
    assert topdown._remedy_is_exhausted([{"attempt": 1, "kept": True}], 2) is False


def test_no_redraw_at_all_is_not_an_exhausted_remedy():
    """Offline, or nothing structural to fix — the remedy was never spent."""
    assert topdown._remedy_is_exhausted([], 2) is False


# --- the referee's bisect-on-failure ----------------------------------------
#
# `live36` lost four of its first seven referee batches — 100 adjudications — to
# a truncation bug, because a failed batch was dropped outright. Those rows are
# BY CONSTRUCTION the hardest in the gold set (they are the ones two annotators
# split on), so discarding them strips the difficult cases and every downstream
# number reads optimistically. A failed batch now splits once and retries.


def rows(n: int) -> list[dict]:
    return [{"query": f"q{i}", "label_a": "A", "label_b": "B"} for i in range(n)]


def test_a_healthy_batch_makes_exactly_one_call():
    """The recovery path must cost nothing when nothing is wrong."""
    calls, folded = [], []
    def run(part): calls.append(len(part)); return [f"v{i}" for i in range(len(part))]
    failed, covered = topdown._run_batch_with_bisect(run, lambda v, p: folded.extend(v), rows(25))
    assert calls == [25], "a batch that works must not be split"
    assert (failed, covered) == (0, 25)
    assert len(folded) == 25


def test_a_failed_batch_is_split_and_recovered():
    """The whole batch fails, both halves succeed: nothing is lost."""
    seen = []
    def run(part):
        seen.append(len(part))
        return None if len(part) == 25 else ["v"] * len(part)
    failed, covered = topdown._run_batch_with_bisect(run, lambda v, p: None, rows(25))
    assert seen == [25, 12, 13], "one whole attempt, then each half"
    assert failed == 0
    assert covered == 25, "every row recovered — live36 would have lost all 25"


def test_a_half_that_also_fails_loses_only_its_own_rows():
    """Partial recovery still beats dropping the batch."""
    def run(part):
        if len(part) == 25:
            return None
        return None if len(part) == 12 else ["v"] * len(part)
    failed, covered = topdown._run_batch_with_bisect(run, lambda v, p: None, rows(25))
    assert failed == 1
    assert covered == 13, "the surviving half is kept rather than discarded with the other"


def test_the_split_is_not_recursive():
    """Bounded at two extra calls, so a broken referee cannot cause a retry storm."""
    seen = []
    def run(part):
        seen.append(len(part))
        return None
    failed, covered = topdown._run_batch_with_bisect(run, lambda v, p: None, rows(25))
    assert seen == [25, 12, 13], "exactly three calls, never a fourth"
    assert (failed, covered) == (2, 0)


def test_each_recovered_half_binds_the_next():
    """Folding must happen PER CALL, not once at the end.

    The referee settles boundaries, and a boundary settled in one call must bind
    the next — otherwise the second half re-decides it independently and the rule
    set acquires two rules that fire on the same trigger with opposite answers.
    """
    decided: list[str] = []
    order: list[list[str]] = []
    def run(part):
        order.append(list(decided))      # what this call could see when it ran
        return None if len(part) == 25 else [f"v{len(part)}"]
    topdown._run_batch_with_bisect(run, lambda v, p: decided.extend(v), rows(25))
    assert order[1] == [], "the first half runs before anything is decided"
    assert order[2] == ["v12"], "the SECOND half must see the first half's ruling"


# --- a merge must not leave rules pointing at the class it removed ----------
#
# The redraw replaces NODES only, so the rule set survives a merge intact. On
# live36 gen02 that left 4 of 45 rules dangling and 2 routing straight to the
# deleted code — and both governed a pair the redraw had targeted, so the
# re-pilot that scored the redraw was reading guidance it could not follow on
# exactly the rows the comparison turns on.


def tax_with_rules(codes, rules):
    from qmine.records import AdjudicationRule
    t = tax(*codes)
    return t.model_copy(update={"rules": [
        AdjudicationRule(id=rid, when="w", then=then, rationale="r", classes=list(cls))
        for rid, then, cls in rules]})


def test_a_merge_prunes_the_rules_that_routed_to_the_removed_class(monkeypatch):
    ev: list[str] = []
    start = tax_with_rules(
        ["A", "B", "C"],
        [("R1", "A", ["A", "B"]),      # routes to A, which the redraw removes
         ("R2", "B", ["A", "B"]),      # cites A but routes to B — still followable
         ("R3", "C", ["C"])],
    )
    install(monkeypatch, ["B", "C"], [0.9, 0.9])   # redraw drops A
    t, p, hist = topdown._redraw_until_stable(make(ev), None, None, start, pilot(0.70))

    kept = {r.id for r in t.rules}
    assert "R1" not in kept, "a rule routing to a deleted class is unfollowable"
    assert {"R2", "R3"} <= kept, "rules routing to surviving classes must be kept"
    assert any("merged away by this redraw" in e for e in ev), \
        "pruning guidance must be announced, never silent"


def test_a_redraw_that_drops_nothing_touches_no_rules(monkeypatch):
    """The prune must be scoped to an actual merge."""
    ev: list[str] = []
    start = tax_with_rules(["A", "B"], [("R1", "A", ["A"]), ("R2", "B", ["B"])])
    install(monkeypatch, ["A", "B"], [0.9, 0.9])   # same codes back
    t, p, hist = topdown._redraw_until_stable(make(ev), None, None, start, pilot(0.70))
    assert {r.id for r in t.rules} == {"R1", "R2"}
    assert not any("merged away" in e for e in ev)


# --- the referee batches by class pair --------------------------------------
#
# Chunking by row POSITION forced the whole phase to run sequentially: the same
# boundary could land in batch 1 and batch 5, be decided independently, and leave
# the rule set with two rules that fire on the same trigger and disagree. That
# cost ~10-11 min per batch, ~5-6h for a 3,000-row gold set, making the referee
# the wall-clock bottleneck of the pipeline. Packing each pair entirely into one
# batch removes the hazard at the source.


def dis(a: str, b: str, n: int) -> list[dict]:
    return [{"query": f"{a}/{b}#{i}", "label_a": a, "label_b": b} for i in range(n)]


def pairs_of(batch) -> set:
    return {frozenset((d["label_a"], d["label_b"])) for d in batch}


def flat(groups):
    return [c for g in groups for c in g]


def test_no_class_pair_is_ever_split_across_two_GROUPS():
    """Groups are what run concurrently, so a pair must live in exactly one."""
    rows = dis("A", "B", 9) + dis("C", "D", 7) + dis("E", "F", 3) + dis("G", "H", 2)
    groups = topdown._batch_by_class_pair(rows, target=15)

    seen = {}
    for gi, g in enumerate(groups):
        for chunk in g:
            for p in pairs_of(chunk):
                assert seen.get(p, gi) == gi, (
                    f"pair {sorted(p)} spans groups {seen[p]} and {gi} — two "
                    "concurrent groups would decide the same boundary")
                seen[p] = gi
    assert sum(len(c) for c in flat(groups)) == len(rows), "every row must be batched"


def test_an_oversized_pair_is_split_into_SEQUENTIAL_chunks_in_one_group():
    """It cannot fit in one call, so it is chunked — but stays in one group.

    live38 measured a single pair with 52 rows while 25-row calls were already
    emitting 34,099 tokens and failing to parse. Keeping it whole would hand the
    model a batch larger than the ones already failing.
    """
    rows = dis("BIG", "HUGE", 40) + dis("C", "D", 4)
    groups = topdown._batch_by_class_pair(rows, target=15)

    big = [g for g in groups if frozenset(("BIG", "HUGE")) in pairs_of(g[0])]
    assert len(big) == 1, "the oversized pair must occupy exactly ONE group"
    assert len(big[0]) > 1, "and be split into several sequential chunks"
    assert all(len(c) <= 15 for c in big[0]), "no chunk may exceed the target"
    assert sum(len(c) for c in big[0]) == 40, "without losing a row"


def test_small_pairs_are_packed_so_the_group_count_stays_near_minimum():
    """Pair-completeness must not degenerate into one group per pair."""
    rows = sum((dis(f"X{i}", f"Y{i}", 2) for i in range(10)), [])
    groups = topdown._batch_by_class_pair(rows, target=15)
    assert len(groups) <= 3, f"20 rows over 10 tiny pairs became {len(groups)} groups"


def test_the_referee_prompt_does_not_depend_on_sibling_completion_order():
    """`decided` must not be threaded into concurrent prompts.

    It existed so a later batch would honour an earlier ruling — unnecessary once
    a pair cannot span batches. Passing the running dict under concurrency would
    make each prompt depend on which siblings finished first, so a replay would
    send different prompts and miss its own cache: the resume cascade, recreated
    inside one phase.
    """
    import inspect

    src = inspect.getsource(topdown.p2b_gold)
    statements = {ln.strip() for ln in src.splitlines()}
    assert "decided=list(decided.values())," not in statements, \
        "a race-dependent prompt is not reproducible and will not replay from cache"
    assert "decided=prior," in statements, \
        "the only prior allowed is the within-group one, which is sequential"
    # And that prior must come from the group's OWN pair, not the shared dict.
    assert "prior = (got,) if got else ()" in statements
    assert "if ci and len(group) > 1:" in statements, \
        "a single-chunk group has no earlier ruling to honour"
