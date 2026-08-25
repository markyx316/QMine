"""The live phase observer, and the two things that keep it honest.

Agents did 3.7% of the bottom-up work on live38 and none of it before P7 — every
representation, algorithm, K and hierarchy decision was made with no agent
looking. This brings a second opinion forward to the phase that makes the
decision, without giving it any authority over the decision.
"""
from __future__ import annotations

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
                           evidence=kw.get("evidence", ""), would_change="")


def _raw(*obs):
    return SimpleNamespace(observations=list(obs), checked=["granularity"])


def _deps():
    ev, gates = [], {}

    def gate(name, phase, **kw):
        gates[name] = kw

    return SimpleNamespace(emit=ev.append, gate=gate, agent_ctx=lambda: None), ev, gates


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


def test_a_blocking_observation_fails_the_phase_gate(monkeypatch):
    """The output must reach something that ACTS.

    A critic agent once identified the kappa defect before the run that shipped
    it; the finding went to an artifact and nothing consumed it, so it shipped
    anyway. An observation that only lands in JSON is a note to nobody.
    """
    import qmine.agents.roles as roles

    class FakeObserver:
        def __init__(self, ctx, suffix=""): pass
        def run(self, **kw):
            return _raw(_obs("blocking", "K=10 is not the stability peak; k=8 is",
                             "granularity.k_sweep"))

    monkeypatch.setattr(roles, "ObserverAgent", FakeObserver)
    deps, ev, gates = _deps()

    res = observe_phase(deps, "p5", ARTIFACTS)

    assert len(res.blocking) == 1
    g = gates["p5_observer"]
    assert g["passed"] is False, "a blocking observation did not fail the gate"
    assert "stability peak" in g["message"]
    assert g["remediation"], "a failing gate must tell the operator what to do"


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
