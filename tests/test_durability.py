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


# --- resuming the right generation -----------------------------------------
#
# `live36` halted on a gate. The gate's own message says to fix the cause and
# open a new generation — but both resume paths hardcoded generation 1, and the
# LangGraph thread id is `f"{run_id}-gen{generation}"`. So the recommended
# recovery reopened the OLD thread, read `halted=True`, logged "stays halted"
# and exited having done nothing, while gen02 sat untouched.


def test_latest_generation_finds_the_newest(tmp_path):
    from qmine.artifacts import latest_generation

    run = tmp_path / "live36"
    for g in (1, 2, 3):
        (run / f"gen{g:02d}").mkdir(parents=True)
    assert latest_generation(run) == 3


def test_latest_generation_defaults_to_one_for_a_fresh_run(tmp_path):
    from qmine.artifacts import latest_generation

    run = tmp_path / "live99"
    run.mkdir()
    assert latest_generation(run) == 1, "a run with no generations resumes gen01"


def test_a_new_generation_inherits_the_previous_config(tmp_path):
    """`new_generation` opens a directory and a note, not a config.

    Looking for `config.resolved.yaml` only in the current generation found
    nothing, which sent `run --resume` down its "nothing to resume" branch and
    into the guard that refuses an existing run id — a dead end on the exact
    path the halt message recommends.
    """
    from qmine.artifacts import resolved_config_path

    run = tmp_path / "live36"
    (run / "gen01").mkdir(parents=True)
    (run / "gen01" / "config.resolved.yaml").write_text("x: 1")
    (run / "gen02").mkdir()  # opened by new_generation: no config of its own

    found = resolved_config_path(run, 2)
    assert found is not None, "gen02 must fall back to the config it inherited"
    assert found == run / "gen01" / "config.resolved.yaml"


def test_resolved_config_prefers_the_current_generation(tmp_path):
    """A generation that DOES have its own config must not read gen01's."""
    from qmine.artifacts import resolved_config_path

    run = tmp_path / "live36"
    for g in (1, 2):
        (run / f"gen{g:02d}").mkdir(parents=True)
        (run / f"gen{g:02d}" / "config.resolved.yaml").write_text(f"gen: {g}")
    assert resolved_config_path(run, 2) == run / "gen02" / "config.resolved.yaml"


def test_resolved_config_is_absent_when_nothing_was_ever_written(tmp_path):
    from qmine.artifacts import resolved_config_path

    run = tmp_path / "live99"
    (run / "gen01").mkdir(parents=True)
    assert resolved_config_path(run, 1) is None


# --- phase 7 naming: the one blocking gate with no repair path --------------
#
# `p7_all_leaves_named` is blocking and fails on a SINGLE unnamed leaf, and
# `resume` refuses to overturn a gate — so one transient failure among ~60 namer
# calls ends a paid run at phase 7 and forces a new generation. Nothing anywhere
# re-names a lost leaf.


def test_the_namer_retries_before_giving_up_on_a_leaf():
    """`_annotate` retries; `_name` returned None on the first exception.

    The asymmetry mattered because the consequences are opposite: a lost
    annotation batch is excluded and reported, a lost NAMING trips a blocking
    gate that no later phase repairs.
    """
    import inspect

    from qmine.graph.nodes import naming

    src = inspect.getsource(naming.p7_name_shard)
    assert "for attempt in range(3)" in src, "a namer failure must be retried"
    assert "time.sleep" in src, "and backed off between attempts"


def test_the_naming_pool_respects_the_configured_concurrency():
    """The shards run concurrently, so a per-shard pool MULTIPLIES by shard count.

    A hard-coded 4 with 5 shards put 20 calls on one provider while
    `llm.max_concurrency` said 8 — and made that knob inert in exactly the phase
    whose blocking gate has no repair path, so turning it down when a provider
    throttles had no effect here at all.
    """
    import inspect

    from qmine.config import QMineConfig
    from qmine.graph.nodes import naming

    src = inspect.getsource(naming.p7_name_shard)
    assert "min(4, len(leaf_ids))" not in src, "the pool must not be hard-coded"
    assert "max_concurrency" in src, "it must read the configured ceiling"

    cfg = QMineConfig()
    n_shards = max(1, cfg.naming.n_naming_agents)
    per_shard = max(1, cfg.llm.max_concurrency // n_shards)
    assert n_shards * per_shard <= cfg.llm.max_concurrency, (
        f"{n_shards} shards x {per_shard} threads exceeds the "
        f"{cfg.llm.max_concurrency} the operator configured")


def test_every_recovered_artifact_is_actually_written_somewhere():
    """`recover(key, artifact)` is only resume-safe if the artifact EXISTS.

    The sibling guard above catches bare `_cache.get`. This catches the inverse,
    which is just as quiet: `deps.recover("leaf_relabels", "leaf_relabels")` had
    no writer at all, so the fallback never fired and it returned its `{}`
    default — governance renames vanished from the delivered column on every
    resumed run, looking exactly like "there were no renames".

    A `rebuild=` callable is an acceptable substitute: that path reconstructs the
    value from something else on disk.
    """
    import re
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src" / "qmine"
    blob = "\n".join(f.read_text(encoding="utf-8") for f in src.rglob("*.py"))

    offenders = []
    for f in (src / "graph" / "nodes").glob("*.py"):
        text = f.read_text(encoding="utf-8")
        for m in re.finditer(r"recover\(\s*['\"]([^'\"]+)['\"]\s*,\s*['\"]([^'\"]+)['\"]", text):
            key, artifact = m.group(1), m.group(2)
            # a rebuild= in the same call is its own fallback
            tail = text[m.end(): m.end() + 200]
            if "rebuild=" in tail.split(")")[0] or "rebuild=" in tail[:120]:
                continue
            written = re.search(
                rf"put_json\(\s*['\"]{re.escape(artifact)}['\"]"
                rf"|put_table\(\s*['\"]{re.escape(artifact)}['\"]"
                rf"|put_matrix\(\s*['\"]{re.escape(artifact)}['\"]"
                rf"|put_model\(\s*['\"]{re.escape(artifact)}['\"]"
                rf"|register_file\(\s*['\"]{re.escape(artifact)}['\"]",
                blob)
            if not written:
                offenders.append(f"{f.name}: recover(..., {artifact!r}) — nothing writes it")

    assert not offenders, (
        "these recoveries can never fall back, so the value is lost on resume:\n  "
        + "\n  ".join(offenders))


# --- reusing a taxonomy instead of re-deriving one --------------------------


def test_reuse_finds_a_taxonomy_by_run_id_generation_or_path(tmp_path):
    """The root fix for the resume cascade.

    The web-using researchers are not deterministic, so a resumed run re-derives
    a different taxonomy, every annotator prompt changes, and all 3,000 gold rows
    miss the cache the run already paid for.
    """
    import json
    from types import SimpleNamespace

    from qmine.graph.nodes.topdown import _load_taxonomy_for_reuse

    node = {"code": "A", "name": "a", "level": 1, "definition": "d",
            "user_need": "n", "positive_examples": ["x"], "negative_examples": []}
    payload = {"taxonomy": {"version": "v1", "nodes": [node], "rules": []}}
    run = tmp_path / "runX"
    (run / "gen02").mkdir(parents=True)
    (run / "gen02" / "taxonomy.json").write_text(json.dumps(payload), encoding="utf-8")

    deps = SimpleNamespace(cfg=SimpleNamespace(run_root=str(tmp_path)), emit=lambda m: None)
    assert len(_load_taxonomy_for_reuse(deps, "runX").nodes) == 1
    assert len(_load_taxonomy_for_reuse(deps, "runX/gen02").nodes) == 1
    direct = str(run / "gen02" / "taxonomy.json")
    assert len(_load_taxonomy_for_reuse(deps, direct).nodes) == 1


def test_reuse_prefers_the_newest_generation(tmp_path):
    """gen02's taxonomy supersedes gen01's when only a run id is given."""
    import json
    from types import SimpleNamespace

    from qmine.graph.nodes.topdown import _load_taxonomy_for_reuse

    def tx(code):
        return {"taxonomy": {"version": "v1", "rules": [], "nodes": [
            {"code": code, "name": code, "level": 1, "definition": "d",
             "user_need": "n", "positive_examples": ["x"], "negative_examples": []}]}}

    run = tmp_path / "runY"
    for g, code in (("gen01", "OLD"), ("gen02", "NEW")):
        (run / g).mkdir(parents=True)
        (run / g / "taxonomy.json").write_text(json.dumps(tx(code)), encoding="utf-8")

    deps = SimpleNamespace(cfg=SimpleNamespace(run_root=str(tmp_path)), emit=lambda m: None)
    assert _load_taxonomy_for_reuse(deps, "runY").nodes[0].code == "NEW"


def test_reuse_raises_rather_than_silently_re_deriving(tmp_path):
    """Asking to reuse and quietly getting a fresh taxonomy is the worst outcome.

    The run would look like it obeyed, pay for a full architect pass, and miss
    the cache it was pointed at — the exact cascade reuse exists to prevent.
    """
    from types import SimpleNamespace

    import pytest

    from qmine.graph.nodes.topdown import _load_taxonomy_for_reuse

    deps = SimpleNamespace(cfg=SimpleNamespace(run_root=str(tmp_path)), emit=lambda m: None)
    with pytest.raises(FileNotFoundError):
        _load_taxonomy_for_reuse(deps, "nothing_here")


def test_reuse_taxonomy_is_honoured_on_the_resume_path_too():
    """`--resume` returns before the fresh-run setup, so the flag must be applied
    inside that branch.

    Accepted-and-ignored is the worst outcome: the run re-derives a taxonomy from
    non-deterministic web researchers, every annotator prompt changes, and all
    3,000 gold rows miss the cache the flag was pointed at — silently, and at
    full price.
    """
    import inspect

    from qmine import cli

    src = inspect.getsource(cli.run)
    resume_branch = src[src.index("if resume and run_id:"):src.index("nothing to resume")]
    assert "cfg.taxonomy.reuse_taxonomy_from = reuse_taxonomy" in resume_branch, \
        "the resume branch returns early; setting the flag after it is a no-op"


def test_a_new_generation_inherits_the_runs_data_settings():
    """`input_path` comes from a CLI flag, not the config file.

    A generation snapshot built from the config file alone drops it, and the
    resumed run dies in p1 with "config.data.input_path is not set". A new
    generation of the SAME run is by definition the same corpus, so the data
    block must carry over even when the rest of the config is replaced.
    """
    import inspect

    from qmine import cli

    src = inspect.getsource(cli.new_generation_cmd)
    for field in ("input_path", "text_column", "reference_label_columns"):
        assert f"cfg.data.{field}" in src, (
            f"{field} is set at launch, not in the config file — a snapshot that "
            "drops it cannot be resumed")


def test_a_pipeline_exception_is_not_relabelled_as_a_checkpointer_failure(tmp_path):
    """`@contextmanager` throws a WITH-BODY exception back in at the yield.

    `_checkpointer` wrapped a single try/except around its `yield`, so it caught
    the ENTIRE pipeline's failures — not just its own setup — logged them as
    "SQLite checkpointer unavailable", and then fell through to a SECOND `yield`,
    which a generator may not do.

    live42 finished all 17 phases; LangGraph's loop teardown raised; this
    relabelled it "Type is not msgpack serializable: DecisionRecord" — about a
    serializer that encodes `DecisionRecord` perfectly well — and the resulting
    `RuntimeError: generator didn't stop after throw()` replaced the real
    exception and killed the run before `write_summary`. Five of
    `verify_run.py`'s six checks read `run_summary.json`, so a completed run
    scored as though no phase had run at all.
    """
    import pytest

    from qmine.runner import _checkpointer

    boom = ValueError("the pipeline itself failed")
    with pytest.raises(ValueError) as caught:
        with _checkpointer(tmp_path / "ck.sqlite"):
            raise boom
    assert caught.value is boom, (
        "the body's exception was replaced — the original cause is unrecoverable")

    # And the manager still works normally, yielding exactly once.
    seen = []
    with _checkpointer(tmp_path / "ck2.sqlite") as saver:
        seen.append(saver)
    assert len(seen) == 1 and seen[0] is not None


def test_the_memory_store_does_not_swallow_a_pipeline_exception(tmp_path):
    """`open_memory` had the identical defect, and it is the one that actually
    raised on live42: "SQLite store unavailable (generator didn't stop after
    throw())"."""
    import pytest

    from qmine.memory.store import open_memory

    boom = KeyError("phase blew up")
    with pytest.raises(KeyError) as caught:
        with open_memory(tmp_path / "m.sqlite", project="t", domain="d"):
            raise boom
    assert caught.value is boom, "the body's exception was replaced"

    seen = []
    with open_memory(tmp_path / "m2.sqlite", project="t", domain="d") as mem:
        seen.append(mem)
    assert len(seen) == 1 and seen[0] is not None


def test_a_numpy_scalar_in_a_record_cannot_break_checkpointing():
    """A single `numpy.float64` anywhere in a record's free-form fields broke
    LangGraph's checkpoint write — and the error named the pydantic wrapper it
    was nested in, not the scalar:

        TypeError: Type is not msgpack serializable: DecisionRecord

    `DecisionRecord` encodes perfectly well. The encoder's pydantic branch calls
    `model_dump()` and re-encodes the result, and the numpy scalar inside
    `evidence.locator_reach.<ref>.discrimination` is what ormsgpack refuses; the
    OUTER call reports the wrapper's type. That misdirection cost three separate
    investigations, all aimed at the serializer's allowlist — which gates
    DECODING and was never involved.

    Measured: `make demo` wrote 5 checkpoints for 17 phases before this, and 17
    after. A run that cannot checkpoint cannot be resumed, and `qmine render`
    loses its best source of state.

    Nearly every number in this pipeline is computed with numpy, so the guard
    lives at record construction rather than at each call site: one missed
    conversion reintroduces the whole failure with a message pointing elsewhere.
    """
    import numpy as np

    from qmine.records import DecisionRecord, GateResult, MetricRecord
    from qmine.runner import _serializer

    rec = DecisionRecord(
        id="d", phase="p5", question="q", choice="c", rationale="r",
        evidence={"locator_reach": {"legacy_l2": {"discrimination": np.float64(2.72)}},
                  "sweep": np.array([1.5, 2.5]), "k": np.int64(18),
                  "ok": np.bool_(True)},
    )
    ev = rec.evidence
    assert type(ev["locator_reach"]["legacy_l2"]["discrimination"]) is float
    assert type(ev["k"]) is int and type(ev["ok"]) is bool
    assert isinstance(ev["sweep"], list) and type(ev["sweep"][0]) is float

    gate = GateResult(name="g", phase="p", observed={"kappa": np.float64(0.89)},
                      threshold={"min_kappa": np.float64(0.7)})
    assert type(gate.observed["kappa"]) is float

    metric = MetricRecord(name="m", value=1.0, detail={"se": np.float64(0.01)})
    assert type(metric.detail["se"]) is float

    # The whole point: this must encode.
    kind, blob = _serializer().dumps_typed(
        {"decisions": [rec], "gates": {"g": gate}, "metrics": [metric]})
    assert kind == "msgpack" and blob


def test_the_k_locator_returns_a_python_float():
    """`discrimination` is annotated `-> float` and returned a numpy scalar; the
    annotation was the thing that made the leak invisible."""
    import numpy as np

    from qmine.ops.cluster import discrimination

    # The real sweep holds numpy scalars — every metric here is computed with
    # numpy — and `round()` on one returns a numpy scalar. Two things this test
    # needs, both learned by watching it pass under mutation: the values must BE
    # numpy, and the curve must be jagged enough that `noise_floor` is non-zero.
    # A monotonic sweep gives a zero noise floor and takes the `return 0.0`
    # early exit, so the line under test never runs.
    jagged = [0.30, 0.52, 0.34, 0.61, 0.38, 0.66, 0.41, 0.70]
    sweep = [{"k": i + 2, "ami": np.float64(v)} for i, v in enumerate(jagged)]
    out = discrimination(sweep, "ami")
    assert out > 0, "the early zero-noise exit was taken; the test proves nothing"
    assert type(out) is float, f"returned {type(out).__name__}, not float"
