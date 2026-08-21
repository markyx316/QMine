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
