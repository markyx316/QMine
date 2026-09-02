"""The live phase observer, and the two things that keep it honest.

Agents did 3.7% of the bottom-up work on live38 and none of it before P7 — every
representation, algorithm, K and hierarchy decision was made with no agent
looking. This brings a second opinion forward to the phase that makes the
decision, without giving it any authority over the decision.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

from qmine.agents.observe import observe_phase, resolve_key, verified_observations

ARTIFACTS = {
    "granularity": {"triangulation": {"locator": "intent_alignment_ami",
                                      "chosen_family_k": 10},
                    "k_sweep": [{"k": 8, "stability_ari": 0.9191},
                                {"k": 10, "stability_ari": 0.695}]},
    "panel": {"sets": {"leaves": {"metrics": {"n_clusters": 36}}}},
}


def _obs(sev, claim, key, **kw):
    return SimpleNamespace(severity=sev, claim=claim, artifact_key=key,
                           evidence=kw.get("evidence", ""), would_change="",
                           check=kw.get("check", ""), _verdict="unverifiable")


def _raw(*obs):
    return SimpleNamespace(observations=list(obs), checked=["granularity"])


def _deps(root=None):
    ev, gates = [], {}
    root = root or Path(tempfile.mkdtemp())

    def gate(name, phase, **kw):
        # The real `deps.gate` BUILDS and RETURNS a GateResult and registers
        # nothing. A stub that returns None hides the very propagation bug these
        # tests exist to catch, so it returns a record like the real one.
        gates[name] = kw
        return SimpleNamespace(name=name, phase=phase, **kw)

    store = SimpleNamespace(root=root, gen_dir=root / "gen01")
    return (SimpleNamespace(emit=ev.append, gate=gate, agent_ctx=lambda: None,
                            store=store, run_id="t"), ev, gates)


def test_an_uncited_observation_is_dropped_before_anyone_reads_it():
    """A review mixing traceable and untraceable claims costs more than it saves."""
    res = verified_observations(_raw(
        _obs("warn", "K was located by the stability peak, but the artifact says AMI",
             "granularity.triangulation.locator"),
        _obs("blocking", "the embedding is obviously wrong", "vibes.not_a_real_key"),
        _obs("note", "no citation at all", ""),
    ), ARTIFACTS)

    assert len(res.kept) == 1, [o.claim for o in res.kept]
    assert res.kept[0].artifact_key == "granularity.triangulation.locator"
    assert len(res.dropped) == 2
    assert any("not in this phase's artifacts" in why for _, why in res.dropped)


def test_a_claim_about_something_the_observer_was_SHOWN_can_be_cited():
    """The observer is handed decisions and gates, then forbidden to cite them.

    `observe()` builds the prompt with `artifacts` AND `decisions` AND `gates`,
    but resolution used `artifacts` alone. Measured across three runs, 8 of 20
    dropped observations (40%) cited exactly those side channels — content the
    agent was shown and invited to reason about.

    On live44 this deleted the pre-delivery audit's finding that `00_索引.md`
    claims 21 L1 意图类目 where the taxonomy has 20. It was correct, and it was
    the last check before the deliverables shipped.
    """
    from qmine.agents.observe import citable_namespace

    decisions = [{"id": "D003", "choice": "20 L1 intents", "decisive_metrics": []}]
    ns = citable_namespace(ARTIFACTS, decisions=decisions, gates={"p2b_kappa": {"passed": True}})

    res = verified_observations(_raw(
        _obs("warn", "D003 says decided_by=metric but lists no decisive metric",
             "decisions.0.decisive_metrics"),
        _obs("note", "the kappa gate passed", "gates.p2b_kappa.passed"),
    ), ns)

    assert len(res.kept) == 2, [why for _, why in res.dropped]
    # And the SAME claims must still be dropped when the channel was not shown.
    bare = verified_observations(_raw(
        _obs("warn", "D003 says decided_by=metric but lists no decisive metric",
             "decisions.0.decisive_metrics"),
    ), ARTIFACTS)
    assert len(bare.dropped) == 1, "an unshown channel is still uncitable"


def test_widening_the_citation_pool_also_widens_the_CHECK_evaluator():
    """Half this fix is worse than none.

    `verified_observations` passes one mapping to BOTH `resolve_key` and
    `ops.checks.evaluate`. Widening only the citation side would admit a claim
    whose check cannot run — silently demoting a measurable claim to advisory,
    which is the one capability this door exists to provide.
    """
    from qmine.agents.observe import citable_namespace

    ns = citable_namespace(ARTIFACTS, decisions=[{"id": "D003", "decisive_metrics": []}])
    res = verified_observations(_raw(
        _obs("warn", "D003 records no decisive metric",
             "decisions.0.decisive_metrics",
             check="len(decisions[0].decisive_metrics) > 0"),
    ), ns)

    assert res.kept, "the claim must survive citation"
    assert res.check_results, "and its check must have been EVALUATED, not skipped"
    assert res.check_results[0].verdict in {"confirmed", "refuted"}, \
        f"a check over a shown channel must actually run, got {res.check_results[0].verdict!r}"


def test_a_side_channel_never_shadows_a_real_artifact():
    """An artifact named `gates` keeps its meaning."""
    from qmine.agents.observe import citable_namespace

    ns = citable_namespace({"gates": {"real": 1}}, gates={"side": 2})
    assert ns["gates"] == {"real": 1}, "the artifact wins; the side channel is dropped"


def test_a_made_up_severity_is_dropped_rather_than_coerced():
    """Coercing 'critical' to 'blocking' would let the agent invent a halt level."""
    res = verified_observations(_raw(
        _obs("critical", "something", "panel.sets.leaves"),
        _obs("", "something else", "panel.sets.leaves"),
    ), ARTIFACTS)
    assert res.kept == []
    assert all("unknown severity" in why for _, why in res.dropped)


def test_resolve_key_survives_the_shapes_an_agent_actually_writes():
    """Reject uncited claims, not merely badly-punctuated ones."""
    for key in ("granularity.triangulation.locator",
                "`granularity.triangulation.locator`",
                "granularity.triangulation.locator,",
                "granularity[triangulation][locator]",
                "granularity.k_sweep[0].k"):
        found, _ = resolve_key(key, ARTIFACTS)
        assert found, key
    for key in ("", "nope", "granularity.nope", "granularity.k_sweep[9].k"):
        assert not resolve_key(key, ARTIFACTS)[0], key


def _fake(monkeypatch, *obs):
    import qmine.agents.roles as roles

    class FakeObserver:
        def __init__(self, ctx, suffix=""): pass
        def run(self, **kw): return _raw(*obs)

    monkeypatch.setattr(roles, "ObserverAgent", FakeObserver)


def test_a_confirmed_blocking_observation_fails_the_phase_gate(monkeypatch):
    """The output must reach something that ACTS — once it is proven.

    A critic agent once identified the kappa defect before the run that shipped
    it; the finding went to an artifact and nothing consumed it, so it shipped
    anyway. An observation that only lands in JSON is a note to nobody.

    The check is what earns the halt. `k_sweep[1].k == 10` really is not the
    stability peak, and the assertion "the chosen k IS the max-stability k"
    FAILS against the artifact — so this is a measurement, not an opinion.
    """
    _fake(monkeypatch, _obs(
        "blocking", "K=10 is not the stability peak; k=8 is", "granularity.k_sweep",
        check=("granularity.triangulation.chosen_family_k == "
               "granularity.k_sweep[0].k")))
    deps, ev, gates = _deps()

    res = observe_phase(deps, "p5", ARTIFACTS)

    assert len(res.blocking) == 1
    g = gates["p5_observer"]
    assert g["passed"] is False, "a CONFIRMED blocking observation did not fail the gate"
    assert "stability peak" in g["message"]
    assert "CONFIRMED" in g["message"]
    assert g["observed"]["n_blocking_confirmed"] == 1
    assert g["observed"]["confirmed_checks"], "the failing assertion must be recorded"
    assert g["remediation"], "a failing gate must tell the operator what to do"


def test_an_unproven_blocking_observation_warns_but_cannot_halt(monkeypatch):
    """Severity is the agent's confidence. Confidence is not authority.

    This is the line the whole design sits on: `severity` is written by the
    model and audited by nothing, so letting it fail a gate would hand an LLM
    the power to stop a paid run on an unverified hunch — the one thing every
    other guardrail here exists to prevent. An unprovable concern is still
    reported, still filed in the ledger, and still visible to the operator; it
    simply does not get to decide.
    """
    _fake(monkeypatch, _obs("blocking", "the hierarchy looks wrong to me",
                            "granularity.triangulation"))
    deps, ev, gates = _deps()

    res = observe_phase(deps, "p5", ARTIFACTS)

    assert res.blocking == [], "an unverified claim must not count as blocking"
    assert len(res.unverified_blocking) == 1
    g = gates["p5_observer"]
    assert g["passed"] is True
    assert g["observed"]["n_blocking_unverified"] == 1
    assert "no check could settle" in g["message"], g["message"]
    assert "nothing blocking" not in g["message"], (
        "a worried observer that could not prove it must not be summarised as calm")


def test_an_observation_its_own_check_refutes_is_dropped(monkeypatch):
    """The cheapest true finding in the system is an agent falsifying itself.

    The observer supplies the test along with the claim. When the test passes,
    the artifacts are consistent and the claim was wrong — so it is dropped
    before it costs a reader any attention, exactly like an uncited one.
    """
    _fake(monkeypatch,
          _obs("blocking", "n_clusters is missing from the panel",
               "panel.sets.leaves.metrics",
               check="panel.sets.leaves.metrics.n_clusters > 0"),
          _obs("warn", "the locator really is AMI",
               "granularity.triangulation.locator",
               check="granularity.triangulation.locator == 'intent_alignment_ami'"))
    deps, ev, gates = _deps()

    res = observe_phase(deps, "p9", ARTIFACTS)

    assert res.kept == [], [o.claim for o in res.kept]
    assert len(res.dropped) == 2
    assert all("REFUTES" in why for _, why in res.dropped)
    assert gates["p9_observer"]["passed"] is True


def test_a_confirmed_finding_survives_into_the_run_level_ledger(monkeypatch):
    """A finding that lives for one run is one the next run loses.

    live39's p6 observer was right, warned, and would have been forgotten: the
    run summary kept four words and nothing carried them forward. The ledger
    sits at the RUN root beside the LLM cache, so a new generation inherits it.
    """
    from qmine.ops.findings import FINDINGS_FILE, FindingLedger

    root = Path(tempfile.mkdtemp())
    _fake(monkeypatch, _obs(
        "blocking", "K=10 is not the stability peak", "granularity.k_sweep",
        check="granularity.triangulation.chosen_family_k == granularity.k_sweep[0].k"))
    deps, ev, gates = _deps(root)

    observe_phase(deps, "p5", ARTIFACTS)

    led = FindingLedger(root / FINDINGS_FILE)
    assert len(led.confirmed_open) == 1
    f = led.confirmed_open[0]
    assert f.phase == "p5" and f.verdict == "confirmed" and f.blocking
    assert f.check, "the ledger must keep the expression, or it cannot re-check"

    # The same defect, seen again next generation, is ONE row with a history.
    deps2, _, _ = _deps(root)
    observe_phase(deps2, "p5", ARTIFACTS)
    led2 = FindingLedger(root / FINDINGS_FILE)
    assert len(led2.entries) == 1, "the same defect must not become two rows"
    assert led2.entries[f.id].times_seen == 2


def test_an_observer_that_cannot_run_does_not_stop_the_run(monkeypatch):
    """It is a second opinion, not a dependency."""
    import qmine.agents.roles as roles

    class DeadObserver:
        def __init__(self, ctx, suffix=""): pass
        def run(self, **kw): raise RuntimeError("provider down")

    monkeypatch.setattr(roles, "ObserverAgent", DeadObserver)
    deps, ev, gates = _deps()

    res = observe_phase(deps, "p5", ARTIFACTS)

    assert res.kept == [] and res.blocking == []
    assert gates == {}, "a dead observer must not record a passing gate it never earned"
    assert any("unavailable" in e for e in ev), ev


def test_warnings_reach_the_operator_without_halting(monkeypatch):
    import qmine.agents.roles as roles

    class WarnObserver:
        def __init__(self, ctx, suffix=""): pass
        def run(self, **kw):
            return _raw(_obs("warn", "n_clusters describes the pre-governance tree",
                             "panel.sets.leaves.metrics.n_clusters"))

    monkeypatch.setattr(roles, "ObserverAgent", WarnObserver)
    deps, ev, gates = _deps()

    res = observe_phase(deps, "p9", ARTIFACTS)

    assert gates["p9_observer"]["passed"] is True
    assert gates["p9_observer"]["observed"]["n_warn"] == 1
    assert res.warnings and not res.blocking


def test_the_observer_gate_is_handed_back_for_the_node_to_register(monkeypatch):
    """`deps.gate()` RETURNS a GateResult and registers nothing.

    The calling node must place it in the state it returns. The first version of
    `observe_phase` called `deps.gate(...)` and discarded the result, so the
    observer's verdict reached the run log and nothing else — absent from
    `run_summary`, from the report's gate ledger, and from any operator's view.
    A pre-flight caught it: 10 gates recorded where 15 were created.

    That is the precise failure this module's docstring is about, reproduced by
    the module itself. `as_state_gates()` is what a node merges into its return.
    """
    import qmine.agents.roles as roles

    class Quiet:
        def __init__(self, ctx, suffix=""): pass
        def run(self, **kw):
            return SimpleNamespace(observations=[], checked=[])

    monkeypatch.setattr(roles, "ObserverAgent", Quiet)
    deps, ev, gates = _deps()

    res = observe_phase(deps, "p5", ARTIFACTS)

    assert res.gate is not None, "the gate was created and dropped on the floor"
    assert res.as_state_gates() == {"p5_observer": res.gate}
    assert res.gate.name == "p5_observer"


def test_a_dead_observer_hands_back_nothing_to_register(monkeypatch):
    """No gate was earned, so none must appear — not even a passing one."""
    import qmine.agents.roles as roles

    class Dead:
        def __init__(self, ctx, suffix=""): pass
        def run(self, **kw): raise RuntimeError("down")

    monkeypatch.setattr(roles, "ObserverAgent", Dead)
    deps, ev, gates = _deps()

    assert observe_phase(deps, "p5", ARTIFACTS).as_state_gates() == {}


def test_a_decision_can_be_cited_by_its_id_not_only_its_position():
    """med01 dropped two real observations citing `D003.evidence.critic_verdict`.

    `decisions` is a list, so widening the namespace to include it made only
    `decisions.2.evidence...` resolvable — while the id is how the decision
    record prints itself and how an agent naturally refers to one. Position is an
    implementation detail; the id is the name.
    """
    from qmine.agents.observe import citable_namespace

    decisions = [
        {"id": "D001", "question": "encoder?", "choice": "bge-base"},
        {"id": "D003", "choice": "L1 = 22", "evidence": {"critic_verdict": "revise"}},
    ]
    ns = citable_namespace({"panel": {}}, decisions=decisions)

    res = verified_observations(_raw(
        _obs("warn", "D003 records critic_verdict=revise and nothing acted on it",
             "D003.evidence.critic_verdict"),
    ), ns)
    assert res.kept, [why for _, why in res.dropped]

    # Positional citation must keep working — this widens, never replaces.
    res2 = verified_observations(_raw(
        _obs("note", "same claim, positional", "decisions.1.evidence.critic_verdict"),
    ), ns)
    assert res2.kept, [why for _, why in res2.dropped]

    # And an id that does not exist is still uncitable.
    res3 = verified_observations(_raw(
        _obs("warn", "invented decision", "D999.evidence.anything"),
    ), ns)
    assert res3.dropped and not res3.kept


def test_the_observer_is_never_handed_unparseable_json():
    """`json.dumps(x)[:60000]` cut JSON mid-token and logged nothing.

    Measured on med02: the taxonomy artifact alone serialises to 59,959 chars
    against a 60,000 limit, and the payload carries more than the taxonomy — so
    the observer received a fragment on every phase. Its p2a observer reported
    the key cut at `self_consistency_ka…`, and that observation was then DROPPED
    for citing a key the truncation had mangled, which reads as the agent's
    fault rather than ours.

    Whole entries must go, the payload must still parse, and what was withheld
    must be named so the agent does not cite it.
    """
    import inspect
    import json

    from qmine.agents import observe

    src = inspect.getsource(observe.observe_phase)
    assert "__withheld__" in src, "the payload must name what it dropped"
    assert "json.dumps(x, ensure_ascii=False, default=str)[:limit]" not in src, \
        "a raw slice cuts JSON mid-token"

    # Behavioural: rebuild the nested helper and check it holds the contract.
    whole = inspect.getsource(observe.observe_phase)
    i = whole.index("    def _j(x: Any")
    body = "\n".join(l[4:] for l in whole[i:whole.index("\n    try:", i)].splitlines())
    ns = {"json": json, "Any": object}
    exec(body, ns)  # noqa: S102
    out = ns["_j"]({"big": "x" * 40000, "small": {"k": 1}, "other": "y" * 30000}, 20000)
    parsed = json.loads(out)          # must not raise
    assert "small" in parsed, "small artifacts should survive so more keys resolve"
    assert parsed["__withheld__"]["keys"], "and the dropped ones must be named"
