"""Regression tests for the durability defects found on the real corpus.

Each of these was a live bug. They share a shape worth naming: state that lived
only in a process's memory looked correct in a single run and silently degraded
across a resume. The graph restored; the reasoning did not.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from qmine.config import DomainProfile, QMineConfig
from qmine.ops.templates import build_groups, group_masks

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_template_masks_rebuild_from_artifacts_not_memory(deps, frame):
    """A resumed run must recover the phrasing families, not silently score zero.

    The original bug: Phase 8 recomputed its metric deltas over an empty mask
    dict after a resume and reported no fragmentation change at all.
    """
    prof = DomainProfile.load(CONFIGS / "domains" / "k12_zh.yaml")
    groups = build_groups(frame, seeds=prof.template_seeds)
    deps.store.put_table("corpus", frame, producer="test")
    deps.store.put_json("template_groups", {"groups": [g.model_dump() for g in groups]},
                        producer="test")

    deps._cache.clear()                       # simulate a fresh process
    masks = deps.template_masks()
    assert masks, "template masks did not survive a cache clear"
    assert all(isinstance(m, np.ndarray) for m in masks.values())


def test_only_trusted_families_judge_representations(frame):
    """A mined marker like 是什么 attaches to every intent and must not vote."""
    prof = DomainProfile.load(CONFIGS / "domains" / "k12_zh.yaml")
    groups = build_groups(
        frame, seeds=prof.template_seeds,
        discovered=[{"affix": "是什么", "side": "suffix"}, {"affix": "有哪些", "side": "suffix"}],
    )
    all_masks = group_masks(groups, frame)
    trusted = group_masks(groups, frame, trusted_only=True)
    assert set(trusted) <= set(all_masks)
    assert not any(name.startswith("suffix:") for name in trusted), sorted(trusted)


def test_taxonomy_rebuilds_from_artifacts(deps):
    from qmine.records import Taxonomy, TaxonomyNode

    tax = Taxonomy(nodes=[TaxonomyNode(code="X", name="查询读音", definition="d")])
    deps.store.put_json("taxonomy", {"taxonomy": tax.model_dump()}, producer="test")
    deps._cache.clear()
    got = deps.taxonomy()
    assert got is not None and got.nodes[0].code == "X"


def test_resume_restores_the_runs_own_config(tmp_path):
    """Rebuilding a default config on resume dropped `reference_label_columns`,
    which armed the blindness firewall with zero terms — the anti-anchoring
    guarantee lapsing without a single error message."""
    cfg = QMineConfig()
    cfg.domain = DomainProfile.load(CONFIGS / "domains" / "k12_zh.yaml")
    cfg.data.reference_label_columns = ["legacy_l1", "legacy_l2"]
    cfg.data.input_path = "whatever.csv"
    path = tmp_path / "config.resolved.yaml"
    cfg.dump(path)

    restored = QMineConfig.load(path)
    assert restored.data.reference_label_columns == ["legacy_l1", "legacy_l2"]
    assert restored.domain.key == "k12_zh"
    assert restored.config_hash == cfg.config_hash


def test_refinement_does_not_oscillate(toy_embedding):
    """Merge and split used to fight: a merge joined two leaves, the split probe
    saw the seam and split them straight back, forever."""
    from qmine.ops.cluster import build_hierarchy, refine

    tree = build_hierarchy(toy_embedding, 5, min_leaf_size=30, min_leaf_fraction=0.0)
    out = refine(toy_embedding, tree["leaf_labels"], tree["leaf_family"],
                 rounds=6, min_leaf_size=30)
    tail = out["history"][-3:]
    if len(tail) == 3:
        churn = [h["merges"] + h["splits"] for h in tail]
        assert not all(c > 0 for c in churn) or out["history"][-1]["moved_fraction"] < 0.02, (
            f"refinement still churning: {out['history']}"
        )


def test_prescription_ids_are_assigned_by_the_pipeline(deps):
    """An agent that invents an id can collide with an existing prescription and
    silently overwrite it in the reducer, since the ledger keys on id."""
    from qmine.records import Prescription
    from qmine.state import merge_prescriptions

    a = [Prescription(id=deps.next_prescription_id(), kind="merge_families", targets=[1, 2])]
    b = [Prescription(id=deps.next_prescription_id(), kind="isolate_leaf", targets=[3])]
    merged = merge_prescriptions(a, b)
    assert len({p.id for p in merged}) == 2, "id collision would have lost a prescription"


def test_artifact_refs_survive_a_checkpoint_round_trip():
    """Our record types must be serialisable by the checkpointer, or a resumed
    run cannot read its own state."""
    from qmine.runner import _serializer
    from qmine.artifacts import ArtifactRef
    from qmine.records import DecisionRecord, GateResult, Prescription

    serde = _serializer()
    for obj in (
        ArtifactRef(name="x", kind="matrix", path="/tmp/x.npy", sha256="abc"),
        DecisionRecord(id="D1", phase="p3", question="q", choice="c", rationale="r"),
        GateResult(name="g", phase="p1"),
        Prescription(id="P1", kind="merge_families"),
    ):
        kind, blob = serde.dumps_typed(obj)
        back = serde.loads_typed((kind, blob))
        assert type(back) is type(obj), f"{type(obj).__name__} did not round-trip"


# -- the referee upgrade protocol ------------------------------------------

def test_promotion_requires_a_significant_win():
    """A model change is not evidence of improvement. A 60/40 split on a small
    sample must not promote — that is what a coin does most afternoons."""
    from qmine.ops.promotion import score_verdicts

    key = {i: "a" for i in range(30)}
    verdicts = [{"row": i, "winner": "a" if i < 18 else "b"} for i in range(30)]
    out = score_verdicts(verdicts, key)
    assert out["new_wins"] == 18 and out["old_wins"] == 12
    assert not out["promote"], out["verdict"]


def test_promotion_accepts_a_decisive_win():
    from qmine.ops.promotion import score_verdicts

    key = {i: "a" for i in range(100)}
    verdicts = [{"row": i, "winner": "a" if i < 75 else "b"} for i in range(100)]
    out = score_verdicts(verdicts, key)
    assert out["promote"] and out["p_value"] < 0.05


def test_promotion_randomises_presentation_order():
    """Presenting the challenger first every time manufactures the very win the
    protocol exists to test — LLM judges have documented position bias."""
    from qmine.ops.promotion import build_blind_matchups

    n = 200
    _, key = build_blind_matchups([f"q{i}" for i in range(n)], ["A"] * n, ["B"] * n, range(n))
    a_side = sum(1 for v in key.values() if v == "a")
    assert 0.35 * n < a_side < 0.65 * n, f"challenger appeared as side A {a_side}/{n} times"


def test_promotion_never_destroys_the_old_labels():
    import pandas as pd

    from qmine.ops.promotion import apply_promotion

    df = pd.DataFrame({"query": ["a", "b", "c"], "label": ["X", "X", "Y"]})
    scoring = {"promote": True, "per_row": [{"row": 0, "outcome": "new"}]}
    out = apply_promotion(df, "label", ["Z", "X", "Y"], scoring)
    assert list(out["label_v1"]) == ["X", "X", "Y"], "old labels were destroyed"
    assert out["label_source"].iloc[0] == "referee"
    assert out["label_final"].iloc[0] == "Z"


# ==========================================================================
# Resume-safety of LLM-dependent phases
# ==========================================================================

def test_recover_rebuilds_from_the_artifact_when_the_cache_is_cold(deps):
    """The bug this guards: a resumed run silently dropped the top-down label column.

    Phase 10 read `_cache["topdown_preds"]`, which is process memory. On a resumed
    run the new process has an empty cache, so the column was simply absent from
    the delivered table — no error, no warning, and a silent violation of the
    rule that both label systems ship side by side.
    """
    import pandas as pd

    deps.store.put_table(
        "topdown_labels",
        pd.DataFrame({"query": ["a", "b", "c"], "l1_pred": ["X", "Y", "X"]}),
        producer="test",
    )
    deps._cache.pop("topdown_preds", None)          # cold process
    got = deps.recover("topdown_preds", "topdown_labels",
                       rebuild=lambda d: d["l1_pred"].to_numpy())
    assert got is not None and list(got) == ["X", "Y", "X"]


def test_recover_prefers_memory_over_disk(deps):
    deps.cache_put("thing", "from-memory")
    assert deps.recover("thing", "nonexistent-artifact") == "from-memory"


def test_recover_returns_default_when_nothing_exists(deps):
    assert deps.recover("missing", "also-missing", default={}) == {}


def test_no_phase_node_reads_process_memory_without_a_fallback():
    """A structural guard: bare `_cache.get` in a phase node is the bug class itself.

    Five separate call sites had it at once, so a lint-style test is more durable
    than fixing them individually and hoping the pattern does not come back.
    """
    import re
    from pathlib import Path

    nodes = Path(__file__).resolve().parents[1] / "src" / "qmine" / "graph" / "nodes"
    offenders = []
    for f in nodes.glob("*.py"):
        lines = f.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            if "_cache.get(" not in line:
                continue
            window = " ".join(lines[i : i + 4])
            if "deps.load" in window or "deps.has" in window or "deps.recover" in window:
                continue
            offenders.append(f"{f.name}:{i + 1}: {line.strip()}")
    assert not offenders, (
        "phase nodes must recover state from artifacts, not process memory, or the "
        "run is not resumable:\n  " + "\n  ".join(offenders)
    )


def test_naming_cards_rebuild_so_a_resumed_run_does_not_name_zero_clusters(deps):
    """A resumed run reaching the naming shards with a cold cache used to name nothing."""
    from qmine.graph.nodes.naming import _recover_cards

    deps.store.put_json(
        "naming_cards",
        {"cards": [{"leaf_id": 0, "size": 10, "share": 0.1, "center_samples": ["q"],
                    "random_samples": [], "edge_samples": [], "top_ngrams": []}]},
        producer="test",
    )
    deps._cache.pop("naming_cards", None)
    cards = _recover_cards(deps)
    assert set(cards) == {0}


# ==========================================================================
# Template coverage diagnosis
# ==========================================================================

def test_overshooting_seeds_are_diagnosed_as_seeds_not_as_mined_groups(frame):
    """Telling a new-domain user "coverage too high" sends them to the wrong knob."""
    from qmine.ops.templates import build_groups, select_groups_for_coverage

    broad = [{"name": "everything", "pattern": ".", "intent_hint": "matches all"}]
    groups = build_groups(frame, seeds=broad, discovered=[])
    _, report = select_groups_for_coverage(groups, frame)
    assert report["in_window"] is False
    assert report["seed_only_coverage"] > 0.4
    assert "SEEDED patterns alone" in report["diagnosis"]


def test_drift_diff_refuses_to_compare_across_domains(tmp_path, deps):
    """An English tree diffed against a Chinese one reports every family as new.

    That output looks like a finding and is pure artifact, so the lookup must not
    return a baseline from another vertical — including an old baseline that
    predates the domain field, since 'unknown' is not 'same'.
    """
    import json

    from qmine.graph.nodes.delivery import _previous_baseline

    root = tmp_path / "runs"
    for rid, dom in [("other-domain", "finance_zh"), ("no-domain-field", None)]:
        d = root / rid / "gen01"
        d.mkdir(parents=True)
        base = {"run_id": rid, "config_hash": "abc", "n_families": 3, "leaf_names": {}}
        if dom:
            base["domain"] = dom
        (d / "maintenance.json").write_text(json.dumps({"baseline": base}), encoding="utf-8")

    deps.cfg.run_root = str(root)
    deps.cfg.domain.key = "k12_zh"
    assert _previous_baseline(deps) is None

    d = root / "same-domain" / "gen01"
    d.mkdir(parents=True)
    (d / "maintenance.json").write_text(json.dumps({"baseline": {
        "run_id": "same-domain", "config_hash": "abc", "domain": "k12_zh",
        "n_families": 3, "leaf_names": {}}}), encoding="utf-8")
    found = _previous_baseline(deps)
    assert found is not None and found["baseline"]["run_id"] == "same-domain"


def test_no_undefined_names_anywhere_in_the_package():
    """A missing import inside a rarely-taken branch is this codebase's recurring bug.

    It has now happened three times — `np` in a Phase 2e helper, `json` in the
    drift lookup, `np` again in the minority-language branch. Each time the code
    imported cleanly, passed every test that did not enter that branch, and then
    failed at runtime deep into a long pipeline. Two of them were additionally
    masked: one by a bare `except Exception`, one by a `warn_only` gate.

    A static check costs milliseconds and catches the whole class, including in
    branches no test exercises.
    """
    import shutil
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    ruff = shutil.which("ruff") or str(Path(sys.executable).parent / "ruff")
    if not Path(ruff).exists():
        import pytest

        pytest.skip("ruff not installed")
    out = subprocess.run(
        [ruff, "check", str(root / "src"), "--select", "F821", "--no-cache"],
        capture_output=True, text=True,
    )
    assert out.returncode == 0, f"undefined names found:\n{out.stdout}"
