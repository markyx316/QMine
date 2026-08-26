"""Turning an agent's claim into a measurement, and keeping it until it is fixed.

live39's `p6_observer` reported a real defect — `n_leaves = 29` against a
`leaves_per_family` summing to 32 — and could do nothing about it. The claim was
arithmetic over an artifact, the pipeline had no way to evaluate it, so the gate
warned, the run finished, and a human verified it by hand the next day.

These tests pin the two halves of the answer. The observer does not get write
access; it gets a way to be **proven right** (`ops/checks`), and a ledger that
refuses to forget a proven finding (`ops/findings`).
"""
from __future__ import annotations

import numpy as np
import pytest

from qmine.ops.checks import evaluate
from qmine.ops.cluster import leaves_per_family
from qmine.ops.findings import FindingLedger, fingerprint

META_BROKEN = {"n_leaves": 29, "leaves_per_family": {"0": 3, "1": 4, "8": 22}}
META_FIXED = {"n_leaves": 29, "leaves_per_family": {"0": 3, "1": 4, "8": 22}}
META_FIXED["n_leaves"] = 29
META_FIXED["leaves_per_family"] = {"0": 3, "1": 4, "8": 22}
LEAF_SUM_CHECK = "sum(hierarchy_meta.leaves_per_family.values()) == hierarchy_meta.n_leaves"


# ------------------------------------------------------------------ checks
def test_the_live39_claim_is_confirmed_by_its_own_check():
    """The exact finding that started this, evaluated instead of believed."""
    broken = {"hierarchy_meta": {"n_leaves": 29,
                                 "leaves_per_family": {"0": 3, "1": 4, "2": 25}}}
    res = evaluate(LEAF_SUM_CHECK, broken)
    assert res.verdict == "confirmed"
    assert res.value is False

    fixed = {"hierarchy_meta": {"n_leaves": 32,
                               "leaves_per_family": {"0": 3, "1": 4, "2": 25}}}
    assert evaluate(LEAF_SUM_CHECK, fixed).verdict == "refuted"


def test_a_check_that_does_not_return_a_boolean_settles_nothing():
    """`sum(...)` returning 32 is not a claim.

    Treating a truthy number as "the assertion held" would silently refute every
    finding whose check was written as an expression rather than a comparison —
    a gap in the checker reported as good news.
    """
    r = evaluate("sum(hierarchy_meta.leaves_per_family.values())",
                 {"hierarchy_meta": META_BROKEN})
    assert r.verdict == "unverifiable" and r.value == 29


def test_a_check_naming_something_absent_is_unverifiable_not_confirmed():
    """A typo in a citation must not read as a proven defect."""
    for expr in ("hierarchy_meta.no_such_key == 3", "nope.n_leaves == 1"):
        assert evaluate(expr, {"hierarchy_meta": META_BROKEN}).verdict == "unverifiable"


@pytest.mark.parametrize("hostile", [
    "__import__('os').system('true')",
    "hierarchy_meta.__class__",
    "(1).__class__.__bases__[0].__subclasses__()",
    "[].__class__.__mro__[1]",
    "open('/etc/passwd').read()",
    "lambda: 1",
    "hierarchy_meta.pop('n_leaves')",
    "exec('x=1')",
])
def test_a_check_expression_cannot_reach_outside_the_artifacts(hostile):
    """The string comes from a model, so the evaluator is the security boundary.

    `a.b` is a DICT LOOKUP, never `getattr` — so `__class__` resolves to a
    missing key rather than to a Python object, and the usual escape chain has
    no first step. Everything here must land on `unverifiable`: no exception
    escapes into the run, and nothing executes.
    """
    art = {"hierarchy_meta": dict(META_BROKEN)}
    assert evaluate(hostile, art).verdict == "unverifiable"
    assert art["hierarchy_meta"] == META_BROKEN, "the artifact was mutated"


# ------------------------------------------------------------------ ledger
def test_a_finding_closes_only_when_its_own_assertion_holds_again(tmp_path):
    """The single automatic exit, and it is a measurement.

    Not a human ticking a box, and not the absence of a re-report — either would
    let a live defect fall quietly out of the ledger.
    """
    led = FindingLedger(tmp_path / "findings.json")
    led.record(phase="p6", severity="blocking", claim="n_leaves disagrees with the breakdown",
               artifact_key="hierarchy_meta", check=LEAF_SUM_CHECK,
               verdict="confirmed", seen_at="live39/gen01")
    assert len(led.confirmed_open) == 1

    still_broken = {"hierarchy_meta": {"n_leaves": 29, "leaves_per_family": {"0": 32}}}
    assert led.recheck(still_broken) == []
    assert len(led.confirmed_open) == 1, "a finding must not age out while it is still true"

    now_fixed = {"hierarchy_meta": {"n_leaves": 32, "leaves_per_family": {"0": 32}}}
    closed = led.recheck(now_fixed, at="live40/gen01")
    assert len(closed) == 1 and closed[0].status == "fixed"
    assert led.confirmed_open == []


def test_a_finding_with_no_check_never_closes_by_itself(tmp_path):
    """An unprovable claim cannot be disproved by rerunning the pipeline."""
    led = FindingLedger(tmp_path / "findings.json")
    led.record(phase="p5", severity="warn", claim="the conclusion does not follow",
               artifact_key="granularity", seen_at="live39/gen01")
    led.recheck({"granularity": {}}, at="live40/gen01")
    assert len(led.open_findings) == 1

    assert not led.waive(led.open_findings[0].id, "   "), "a waiver needs a reason"
    assert led.waive(led.open_findings[0].id, "reviewed 2026-08-26, k was correct")
    assert led.open_findings == []
    assert led.entries[fingerprint("p5", "the conclusion does not follow", "granularity")].resolution


def test_the_same_defect_reworded_is_one_row_not_two(tmp_path):
    """A ledger that grows by a row per run is a ledger nobody reads.

    The observer rewords its sentence every run — different digits, different
    punctuation — so the fingerprint is normalised past exactly those.
    """
    led = FindingLedger(tmp_path / "findings.json")
    led.record(phase="p6", severity="blocking", claim="n_leaves is 29 but the sum is 32.",
               artifact_key="hierarchy_meta", seen_at="g1")
    led.record(phase="p6", severity="blocking", claim="n_leaves is 28, but the sum is 31!",
               artifact_key="hierarchy_meta", seen_at="g2")
    assert len(led.entries) == 1
    assert list(led.entries.values())[0].times_seen == 2


def test_a_waiver_survives_re_sighting_but_a_fixed_finding_does_not(tmp_path):
    """The three closed states are not the same kind of closed.

    `fixed` and `refuted` were decided by a MEASUREMENT, so seeing the defect
    again means the measurement was wrong or the defect regressed — either way
    it must reopen. `waived` was decided by a PERSON who already knows it is
    there and chose not to act; re-flagging it every run would make waiving
    useless and teach the operator to ignore the ledger. So a waiver holds — and
    the panel report still prints it, with its reason, because a waived finding
    must be a recorded decision rather than a disappearance.
    """
    led = FindingLedger(tmp_path / "findings.json")
    f = led.record(phase="p6", severity="warn", claim="c", artifact_key="a", seen_at="g1")
    led.waive(f.id, "reviewed: pre-refinement by design here")
    assert led.open_findings == []

    led.record(phase="p6", severity="warn", claim="c", artifact_key="a", seen_at="g2")
    assert led.open_findings == [], "a human waiver must not be undone by a re-run"
    assert led.entries[f.id].resolution, "the reason must survive for the report"

    g = led.record(phase="p6", severity="warn", claim="d", artifact_key="a",
                   check="a.n == 1", verdict="confirmed", seen_at="g1")
    led.recheck({"a": {"n": 1}}, at="g1")
    assert led.entries[g.id].status == "fixed"
    led.record(phase="p6", severity="warn", claim="d", artifact_key="a", seen_at="g2")
    assert led.entries[g.id].status == "open", "a regression must reopen"


def test_the_ledger_survives_a_corrupt_file(tmp_path):
    """A bookkeeping failure must not take down a run that produced good work."""
    p = tmp_path / "findings.json"
    p.write_text("{not json", encoding="utf-8")
    assert FindingLedger(p).entries == {}


def test_a_ledger_round_trips_through_disk(tmp_path):
    """A new generation inherits findings the way it inherits the LLM cache."""
    p = tmp_path / "findings.json"
    led = FindingLedger(p)
    led.record(phase="p6", severity="blocking", claim="c", artifact_key="a",
               check=LEAF_SUM_CHECK, verdict="confirmed", seen_at="gen01")
    led.save(run_id="live40")
    again = FindingLedger(p)
    assert len(again.confirmed_open) == 1
    assert again.confirmed_open[0].check == LEAF_SUM_CHECK


# ------------------------------------------------------ the defect it found
def test_leaves_per_family_counts_only_leaves_that_survived_refinement():
    """The defect itself, constructed rather than hoped for.

    `family_lut` keeps a row for every leaf the tree EVER had. Leaf 3 here was
    merged away, so it appears in the lookup table and in no label. Grouping over
    the table credits family 1 with two leaves and makes the breakdown sum to one
    more than the tree contains — which is exactly what live39 shipped, three
    times over.
    """
    labels = np.array([0, 0, 1, 1, 2, 2])          # leaf 3 merged away
    fam_lut = np.array([0, 1, 1, 1])               # the tree BEFORE refinement
    got = leaves_per_family(labels, fam_lut)
    assert got == {"0": 1, "1": 2}
    assert sum(got.values()) == len(set(labels.tolist())) == 3

    naive = {}
    for lid, f in enumerate(fam_lut):
        naive[str(int(f))] = naive.get(str(int(f)), 0) + 1
    assert sum(naive.values()) == 4, "the pre-refinement count really does differ"


def test_a_check_that_stops_evaluating_does_not_close_the_finding(tmp_path):
    """An unverifiable re-check is not evidence of a fix.

    Found by mutation: relaxing `recheck` from "the assertion holds again" to
    "the assertion is not failing" let a finding close whenever its check could
    no longer be evaluated. A phase that stops writing the artifact would then
    silently close every finding about it — a gap in the checker reported as
    good news, which is the failure mode the three-valued verdict exists for.
    """
    led = FindingLedger(tmp_path / "findings.json")
    led.record(phase="p6", severity="blocking", claim="the counts disagree",
               artifact_key="hierarchy_meta", check=LEAF_SUM_CHECK,
               verdict="confirmed", seen_at="g1")

    assert led.recheck({}, at="g2") == [], "an absent artifact must not close a finding"
    assert len(led.confirmed_open) == 1

    assert led.recheck({"hierarchy_meta": {"renamed": 1}}, at="g3") == []
    assert len(led.confirmed_open) == 1, "a renamed key must not close a finding either"


def test_the_check_evaluator_never_calls_getattr():
    """A structural assertion about the security boundary.

    `a.b` must be a dict lookup. The moment any path falls back to `getattr`,
    `__class__` resolves to a real Python object and the standard escape chain
    has its first step — and the hostile-expression tests above would still pass,
    because they only probe the paths that happen to be covered. Found by
    mutation: swapping one `return _MISSING` for a `getattr` survived the whole
    suite. This is the invariant, stated where it cannot be missed.
    """
    import inspect
    import textwrap

    import qmine.ops.checks as checks

    body = "\n".join(l for l in inspect.getsource(checks).splitlines()
                     if not l.strip().startswith("#"))
    for forbidden in ("setattr(", "eval(", "exec(", "__import__", "compile(", "globals("):
        assert forbidden not in body, f"{forbidden} in the check evaluator"

    # `getattr` is permitted for ONE thing: dispatching to `self._NodeName`. It
    # must never touch a value that came out of an artifact — that is the step
    # that would turn `a.b` from a dict lookup into Python attribute access.
    uses = [l.strip() for l in body.splitlines() if "getattr(" in l]
    assert uses and all("getattr(self," in u for u in uses), uses

    # Parsed, not grepped: `_lookup`'s own docstring says the word, and a text
    # match on a comment is how a guardrail test starts passing for the wrong
    # reason. Only a real Call node counts.
    import ast

    tree = ast.parse(textwrap.dedent(inspect.getsource(checks._Eval._lookup)))
    called = {n.func.id for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert not called & {"getattr", "setattr", "eval", "exec", "vars", "globals"}, called


@pytest.mark.parametrize("container", [
    {"x": {"n": 1}},                       # dict branch
    {"x": [{"n": 1}]},                     # list branch
    {"x": "a string"},                     # neither
    {"x": 7},
])
def test_no_container_type_leaks_a_python_attribute(container):
    """Each branch of the lookup, probed separately.

    The parametrised hostile tests only ever reached the dict branch, so a hole
    in the fallthrough went undetected. Every branch must refuse.
    """
    for expr in ("x.__class__ == 1", "x.__dict__ == 1", "x.__len__ == 1"):
        assert evaluate(expr, container).verdict == "unverifiable", (expr, container)
