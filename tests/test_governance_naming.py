"""Nothing written BEFORE p8 governance may be reported as the final tree.

Governance changes the partition after p6 measured it and after p7 named it.
Two separate deliverable defects on live38 came from reading a pre-governance
artifact as though it described what shipped; both are pinned here.

`p7_all_leaves_named` is a BLOCKING gate, and on live38 it passed — "all 29
leaves named". The delivered table carried **36** leaves, and 4,931 rows (9.9%
of 50,000) shipped with an empty name, because p8 governance then executed 6
`split_leaf` and 2 `isolate_leaf` prescriptions and `p10_deliver` builds its
name column from p7's namings plus governance RENAMES only.

The offline stand-in never issues a split, so a pipeline-level assertion passes
whether or not the fix is present — verified by disabling the fix and watching
it still pass. These tests construct the condition instead.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from qmine.graph.nodes import naming as naming_mod


def _deps(namings, n_rows=40):
    events, saved = [], {}
    df = pd.DataFrame({"query": [f"q{i}" for i in range(n_rows)]})
    return SimpleNamespace(
        df=df,
        cfg=SimpleNamespace(
            data=SimpleNamespace(text_column="query"),
            naming=SimpleNamespace(card_center=3, card_random=2, card_edge=2, card_top_ngrams=5),
            seed_metric=0,
        ),
        emit=events.append,
        has=lambda k: k == "tree_naming",
        load=lambda k: {"namings": list(namings), "mean_coherence": 0.7},
        embedding=lambda k: np.eye(n_rows, 8, dtype=np.float32),
        agent_ctx=lambda: SimpleNamespace(),
        store=SimpleNamespace(put_json=lambda k, v, **kw: saved.__setitem__(k, v)),
        cache_put=lambda k, v: saved.__setitem__(k, v),
    ), events, saved


@pytest.fixture
def stub_namer(monkeypatch):
    """Cards and namer both stubbed — this test is about which leaves get asked
    about, not about what the model says."""
    from qmine.ops import cards as cards_mod

    monkeypatch.setattr(
        cards_mod, "build_naming_cards",
        lambda df, labels, H, cents, **kw: [
            SimpleNamespace(leaf_id=i) for i in range(int(labels.max()) + 1)
        ],
    )
    asked = []

    class FakeNamer:
        def __init__(self, ctx, suffix=""): pass

        def run(self, card):
            asked.append(card.leaf_id)
            return SimpleNamespace(model_dump=lambda: {
                "name_zh": f"叶{card.leaf_id}", "code": f"L{card.leaf_id}",
                "user_need": "需求", "coherence": 0.8, "mix_notes": "",
                "risk_flag": False, "risk_reason": "",
            })

    monkeypatch.setattr(naming_mod, "NamerAgent", FakeNamer)
    return asked


def test_a_leaf_that_governance_created_gets_named(stub_namer):
    """The live38 shape: p7 named leaves 0-2, governance split the partition to 5."""
    deps, events, saved = _deps([{"leaf_id": i, "name_zh": f"原{i}"} for i in range(3)])
    labels = np.array([0, 1, 2, 3, 4] * 8)

    naming_mod._name_leaves_governance_created(deps, labels, np.eye(5, 8), 5)

    assert stub_namer == [3, 4], f"asked about {stub_namer}, expected the two new leaves"
    named = {n["leaf_id"] for n in saved["tree_naming"]["namings"]}
    assert named == {0, 1, 2, 3, 4}, f"delivered partition still missing names: {named}"
    assert [n for n in saved["tree_naming"]["namings"] if n["leaf_id"] == 0][0]["name_zh"] == "原0", \
        "renaming an already-named leaf would discard p7's audited name"


def test_an_empty_leaf_is_not_named(stub_namer):
    """A split can leave an id with no rows; naming it would invent a class that
    labels nothing and inflate the delivered leaf count."""
    deps, events, saved = _deps([{"leaf_id": 0, "name_zh": "原0"}])
    labels = np.array([0, 2] * 20)          # leaf 1 exists in the range, holds nothing

    naming_mod._name_leaves_governance_created(deps, labels, np.eye(3, 8), 3)

    assert stub_namer == [2], f"asked about {stub_namer} — empty leaf 1 should be skipped"


def test_nothing_is_rewritten_when_governance_created_nothing(stub_namer):
    """Governance usually only merges. That path must not re-ask the namer, or
    every run pays for 29 redundant calls and p7's audited names are replaced."""
    deps, events, saved = _deps([{"leaf_id": i, "name_zh": f"原{i}"} for i in range(3)])
    labels = np.array([0, 1, 2] * 13 + [0])

    naming_mod._name_leaves_governance_created(deps, labels, np.eye(3, 8), 3)

    assert stub_namer == [], "namer called when there was nothing new to name"
    assert saved == {}, "tree_naming rewritten with no change to record"


def test_a_leaf_the_namer_cannot_name_is_announced_not_silent(monkeypatch):
    """A dropped call must not reproduce the original defect quietly."""
    from qmine.ops import cards as cards_mod

    monkeypatch.setattr(cards_mod, "build_naming_cards",
                        lambda df, labels, H, cents, **kw: [SimpleNamespace(leaf_id=i)
                                                            for i in range(int(labels.max()) + 1)])

    class DeadNamer:
        def __init__(self, ctx, suffix=""): pass

        def run(self, card): raise RuntimeError("provider down")

    monkeypatch.setattr(naming_mod, "NamerAgent", DeadNamer)
    monkeypatch.setattr(naming_mod.time, "sleep", lambda s: None)

    deps, events, saved = _deps([{"leaf_id": 0, "name_zh": "原0"}])
    naming_mod._name_leaves_governance_created(deps, np.array([0, 1] * 20), np.eye(2, 8), 2)

    assert any("could not be named" in e for e in events), f"failure not reported: {events}"


# --------------------------------------------------------------------------
# The reported SHAPE must be the delivered shape.
# --------------------------------------------------------------------------

def _panel(fam_final, leaves, fam_pre=10):
    def m(v):
        return {"metrics": {"n_clusters": {"value": float(v)}}}
    return {"sets": {"families_final": m(fam_final), "leaves": m(leaves),
                     "families_pre_governance": m(fam_pre)}}


def test_the_reported_shape_is_the_delivered_shape():
    """live38 shipped 12 families / 36 leaves and both reports said 10 / 29.

    The Chinese report printed "after blind naming and audit merging: 10 families
    / 29 leaves" three lines above its own metrics table reading 12 and 36 — the
    same document contradicting itself, because `hierarchy_meta` is written in p6.
    """
    from qmine.report._shape import delivered_shape

    stale = {"n_families": 10, "n_leaves": 29}
    assert delivered_shape(_panel(12, 36), stale) == (12, 36), \
        "the pre-governance count won over the panel that measured what shipped"


def test_the_shape_falls_back_when_the_panel_is_absent():
    """A missing panel must degrade to the stale count, never to a crash — the
    report still has to build so the operator can see what else went wrong."""
    from qmine.report._shape import delivered_shape

    assert delivered_shape({}, {"n_families": 10, "n_leaves": 29}) == (10, 29)
    assert delivered_shape({"sets": {}}, {}) == ("?", "?")
    assert delivered_shape({"sets": {"leaves": {}}}, {"n_leaves": 29})[1] == 29


def test_a_split_renames_both_halves_not_just_the_new_one(stub_namer):
    """A split leaves TWO leaves needing a name — the new id and the remnant.

    Naming only the new id was the first version of this fix, and it was not
    enough. On live38 governance split 7 leaves, and in three of them the p7 name
    described the half that LEFT: leaf 8 kept 「汉字拼音查询」 while all 122 of its
    remaining rows are 怎么写 (how-to-write) queries and every pinyin row moved to
    the new leaf; leaf 23 kept 「汉字读音笔顺查询」 with the same inversion. A name
    that actively misdescribes its rows is worse than a missing one, because
    nothing downstream can detect it.

    Measured on the real partitions, retained-fraction picks out exactly the
    seven split sources — 0.34 for leaf 8, 0.25 for leaf 23 — with no access to
    the governance log.
    """
    deps, events, saved = _deps([{"leaf_id": i, "name_zh": f"原{i}"} for i in range(3)], n_rows=40)
    old = np.array([0] * 20 + [1] * 10 + [2] * 10)
    # leaf 0 is split: 6 of its 20 rows move to the new leaf 3.
    new = old.copy()
    new[14:20] = 3

    naming_mod._name_leaves_governance_created(deps, new, np.eye(4, 8), 4, old_labels=old)

    assert set(stub_namer) == {0, 3}, (
        f"asked about {stub_namer}; the split remnant (0) and the new leaf (3) both need names"
    )
    named = {n["leaf_id"]: n["name_zh"] for n in saved["tree_naming"]["namings"]}
    assert named[0] != "原0", "leaf 0 kept the name assigned to the pre-split cluster"
    assert named[1] == "原1" and named[2] == "原2", "untouched leaves were needlessly re-named"


def test_a_leaf_that_only_lost_a_few_rows_keeps_its_name(stub_namer):
    """Re-naming on any change at all would discard p7's audited names every run.

    Reassignment moves a few rows on every governance pass. The bar is a real
    membership change, not a nonzero one — otherwise the fix costs a full naming
    pass per run and throws away names a blind reviewer already scored.
    """
    deps, events, saved = _deps([{"leaf_id": i, "name_zh": f"原{i}"} for i in range(3)], n_rows=60)
    old = np.array([0] * 20 + [1] * 20 + [2] * 20)
    new = old.copy()
    new[19] = 1          # one row of 20 moves: 5% churn, under the 20% bar

    naming_mod._name_leaves_governance_created(deps, new, np.eye(3, 8), 3, old_labels=old)

    assert stub_namer == [], f"re-named on a 5% membership change: {stub_namer}"
    assert saved == {}, "tree_naming rewritten when no name had gone stale"
