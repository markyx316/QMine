"""An advisory agent that a provider refuses must not end the run.

`p7_audit` already wrapped the risk sentinel in try/except, with a comment saying in so many
words that losing the finding is acceptable and losing the run is not: on a corpus of Chinese
public figures a Chinese provider returns 400 `contentFilter`, which is exactly the material it
screens. The except branch then bound `SimpleNamespace(findings=[])`, which satisfies the two
lines immediately below it and has no `model_dump` — so fifty lines later the node died anyway
and `ppl-pool5` halted 58 minutes into a paid run, in the same phase, for the second time.

The lesson is not "catch the exception" (it was caught). It is that a fallback VALUE has to
satisfy every use of the real one, and the only test that shows this is one that runs the node
with the agent refusing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qmine.graph.nodes import naming as naming_mod
from qmine.records import LeafNaming


class _Refuses:
    """What a provider content filter looks like from inside the node."""

    def __init__(self, ctx):
        pass

    def run(self, **kw):
        raise RuntimeError("Error code: 400 - {'contentFilter': [{'level': 1}], "
                           "'error': {'code': '1301', 'message': '系统检测到输入或生成内容可能包含敏感内容'}}")


class _Auditor:
    def __init__(self, ctx):
        pass

    def run(self, **kw):
        from qmine.records import TreeAudit

        return TreeAudit(prescriptions=[])


@pytest.fixture
def seeded_deps(deps):
    n, k = 40, 2
    df = pd.DataFrame({"query": [f"人物查询{i}" for i in range(n)]})
    deps.cache_put("corpus", df)
    deps.cache_put("leaf_labels", np.array([i % k for i in range(n)]))
    deps.cache_put("leaf_family", np.zeros(k, dtype=int))
    deps.cache_put("leaf_centroids", np.eye(k, 8, dtype=np.float32))
    return deps


def _namings(k=2, risk_flag=False):
    return [LeafNaming(leaf_id=i, name_zh=f"叶{i}", code=f"L{i}", user_need="需求",
                       coherence=4, risk_flag=risk_flag) for i in range(k)]


def test_a_refused_risk_sentinel_degrades_instead_of_halting_the_run(seeded_deps, monkeypatch):
    monkeypatch.setattr(naming_mod, "RiskSentinelAgent", _Refuses)
    monkeypatch.setattr(naming_mod, "AuditorAgent", _Auditor)
    events: list[str] = []
    seeded_deps.on_event = events.append

    out = naming_mod.p7_audit({"namings": _namings()}, seeded_deps)

    assert isinstance(out, dict), "the node must return, not raise, when the sentinel is refused"
    stored = seeded_deps.store.load("tree_naming")
    assert stored["risk_report"]["findings"] == [], \
        "the fallback must serialise like a RiskReport — this is the line that used to die"
    assert stored["independent_risk_discovery"]["found_without_being_told"] is False, \
        "nobody looked, so 'independently found' is False — not missing, not True"
    assert any("risk sentinel unavailable" in e for e in events), \
        "a sweep that did not happen must be announced, not silently reported as clean"


def test_the_namers_own_flags_still_reach_governance_when_the_sentinel_is_gone(seeded_deps, monkeypatch):
    """The sentinel is the INDEPENDENT sweep. Losing it must not also lose the pre-screen: a
    leaf the blind namer flagged still has to arrive as a prescription."""
    monkeypatch.setattr(naming_mod, "RiskSentinelAgent", _Refuses)
    monkeypatch.setattr(naming_mod, "AuditorAgent", _Auditor)

    out = naming_mod.p7_audit({"namings": _namings(risk_flag=True)}, seeded_deps)

    flagged = [p for p in out.get("prescriptions", []) if p.kind == "flag_risk"]
    assert len(flagged) == 2, f"namer-flagged leaves lost: {out.get('prescriptions')}"
    assert all(p.proposed_by == "namer" for p in flagged), \
        "provenance must say the namer found it, not the sentinel that never ran"
