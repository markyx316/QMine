"""End-to-end wiring tests.

These run the real graph over a real (small) slice of the real corpus, entirely
offline. They are slower than the unit tests and worth every second: almost every
bug found while building this system was an integration bug — a phase reading an
artifact a later phase produces, a reducer dropping parallel writes, a schema the
offline stand-in could not fill.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from qmine.config import QMineConfig
from qmine.runner import run_pipeline

DATA = Path(__file__).resolve().parents[1] / "data" / "raw" / "k12_queries_50k.csv"
pytestmark = pytest.mark.skipif(not DATA.exists(), reason="bundled dataset absent")


@pytest.fixture(scope="module")
def completed_run(tmp_path_factory):
    """One offline run, shared by every assertion below."""
    root = tmp_path_factory.mktemp("runs")
    cfg = QMineConfig(fast_mode=True, offline=True, run_root=str(root))
    from qmine.config import DomainProfile

    cfg.domain = DomainProfile.load(Path(__file__).resolve().parents[1] / "configs" / "domains" / "k12_zh.yaml")
    cfg.llm.provider = "mock"
    cfg.data.input_path = str(DATA)
    cfg.data.text_column = "query"
    cfg.data.reference_label_columns = ["legacy_l1", "legacy_l2"]
    # The default 0.98 held-out threshold is calibrated for a real sentence
    # encoder on a full corpus. This fixture runs on 6k rows through the hashing
    # stand-in, which captures surface form only — a structure that genuinely
    # reproduces less sharply. Relaxing the threshold here keeps the test about
    # *wiring*; `test_a_failed_blocking_gate_halts_the_run_and_says_why` covers
    # the gate itself at its real setting.
    cfg.gates.heldout_reproduction = 0.90
    cfg.data.sample_size = 6000
    cfg.taxonomy.gold_sample_size = 80
    cfg.clustering.min_leaf_size = 300
    cfg.clustering.k_sweep = [6, 8, 10]
    cfg.clustering.battery_k = [8]
    cfg.representation.alpha_grid = [0.0, 0.1, 0.5]
    cfg.representation.bakeoff_subsample = 2500
    return run_pipeline(cfg, run_id="pytest-e2e", stream=False)


def test_every_phase_completes(completed_run):
    done = completed_run["state"].get("completed_phases", [])
    for p in ("p0", "p1", "p2a", "p2b", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12"):
        assert p in done, f"{p} missing; halt_reason={completed_run['summary'].get('halt_reason')}"


def test_run_did_not_halt(completed_run):
    s = completed_run["summary"]
    assert not s["halted"], s["halt_reason"]


def test_core_artifacts_exist_on_disk(completed_run):
    gen = Path(completed_run["summary"]["artifact_root"])
    for name in ("run_manifest.json", "data_audit.json", "template_groups.json",
                 "representation.json", "battery.json", "granularity.json",
                 "hierarchy_meta.json", "tree_naming.json", "governance.json",
                 "metrics_panel.json", "labels_full.csv", "deployment.json",
                 "maintenance.json", "Report_BottomUp_Approach.md",
                 "Report_TopDown_Approach.md", "Leaf_Catalogue.md"):
        assert (gen / name).exists(), f"{name} not produced"


def test_state_holds_pointers_not_payloads(completed_run):
    """The design constraint that makes checkpointing viable."""
    for ref in completed_run["state"]["artifacts"].values():
        assert isinstance(ref.path, str)
        assert ref.sha256
        blob = ref.model_dump_json()
        assert len(blob) < 4000, f"{ref.name} reference is {len(blob)} bytes — payload leaked into state"


def test_delivered_table_carries_both_label_systems(completed_run):
    import pandas as pd

    df = pd.read_csv(Path(completed_run["summary"]["artifact_root"]) / "labels_full.csv")
    for col in ("bu_leaf", "bu_family_pre_governance", "bu_family_final", "bu_margin", "bu_ambiguous"):
        assert col in df.columns, col
    assert any(c.startswith("ref_") for c in df.columns), "reference labels not carried through"
    # pre-governance column retained so every merge is reversible
    assert "bu_family_pre_governance" in df.columns


def test_every_prescription_is_settled(completed_run):
    """Principle 6, checked on a real run rather than a fixture."""
    for p in completed_run["state"].get("prescriptions", []):
        assert p.settled, f"{p.id} left {p.status}"
        if p.status == "executed":
            assert p.evidence.get("column"), f"{p.id} executed without an evidence pointer"


def test_panel_rows_share_one_panel_id(completed_run):
    panel = json.loads((Path(completed_run["summary"]["artifact_root"]) / "metrics_panel.json").read_text())
    ids = {s["panel_id"] for s in panel["sets"].values()}
    assert len(ids) == 1, ids


def test_naming_cards_contain_no_label_vocabulary(completed_run):
    """The blindness guarantee, verified against the artifact that was written."""
    gen = Path(completed_run["summary"]["artifact_root"])
    cards = json.loads((gen / "naming_cards.json").read_text())
    legacy = set()
    import pandas as pd

    df = pd.read_csv(DATA)
    legacy |= set(df["legacy_l1"].astype(str).unique())
    legacy |= set(df["legacy_l2"].astype(str).unique())
    blob = json.dumps(cards["cards"], ensure_ascii=False)
    leaked = [t for t in legacy if len(t) >= 4 and t in blob]
    assert not leaked, f"legacy labels reached the naming cards: {leaked[:5]}"


def test_decisions_record_what_was_rejected(completed_run):
    decisions = completed_run["state"].get("decisions", [])
    assert decisions, "no decisions recorded"
    assert any(d.rejected for d in decisions), "nothing was recorded as rejected — the failure history would be empty"


def test_reports_carry_the_provenance_note(completed_run):
    """An offline run must say so in every report it writes."""
    gen = Path(completed_run["summary"]["artifact_root"])
    for name in ("Report_BottomUp_Approach.md", "Report_TopDown_Approach.md"):
        text = (gen / name).read_text()
        assert "offline heuristic" in text.lower() or "NOT by a language model" in text


def test_reports_state_what_the_numbers_do_not_mean(completed_run):
    text = (Path(completed_run["summary"]["artifact_root"]) / "Report_BottomUp_Approach.md").read_text()
    assert "do not mean" in text.lower()
    assert "learnab" in text.lower()


def test_manifest_pins_the_environment(completed_run):
    m = json.loads((Path(completed_run["summary"]["artifact_root"]) / "run_manifest.json").read_text())
    assert m["config_hash"]
    assert m["seed_policy"]["metric"] == 0
    assert m["versions"]["sklearn"] != "not installed"
    assert m["prompt_hashes"], "prompt hashes absent — prompts would not be traceable"


def test_a_failed_blocking_gate_halts_the_run_and_says_why(tmp_path):
    """The gates are not advisory. A failed blocking gate must stop the run
    *and* leave a legible reason — a silent stop is worse than a crash."""
    from qmine.config import DomainProfile

    cfg = QMineConfig(fast_mode=True, offline=True, run_root=str(tmp_path))
    cfg.domain = DomainProfile.load(
        Path(__file__).resolve().parents[1] / "configs" / "domains" / "k12_zh.yaml")
    cfg.llm.provider = "mock"
    cfg.data.input_path = str(DATA)
    cfg.data.sample_size = 2000
    cfg.taxonomy.gold_sample_size = 40
    cfg.clustering.min_leaf_size = 25          # deliberately too fine to reproduce
    cfg.clustering.k_sweep = [12, 20]
    cfg.clustering.battery_k = [12]
    cfg.representation.alpha_grid = [0.0, 0.1]
    cfg.representation.bakeoff_subsample = 1000
    out = run_pipeline(cfg, run_id="gate-halt", stream=False)

    gate = out["state"]["gates"].get("p6_heldout_reproduction")
    assert gate is not None
    if gate.status == "failed":
        assert out["state"]["halted"], "a failed blocking gate did not halt the run"
        assert "p6_heldout_reproduction" in out["state"]["halt_reason"]
        assert "p9" not in out["state"]["completed_phases"], "phases ran past a failed blocking gate"


def test_offline_run_is_reproducible(tmp_path):
    """Same config, same seed, same numbers. Twice."""
    from qmine.config import DomainProfile

    def _run(rid):
        cfg = QMineConfig(fast_mode=True, offline=True, run_root=str(tmp_path))
        cfg.domain = DomainProfile.load(
            Path(__file__).resolve().parents[1] / "configs" / "domains" / "k12_zh.yaml")
        cfg.llm.provider = "mock"
        cfg.data.input_path = str(DATA)
        cfg.data.reference_label_columns = []
        cfg.gates.heldout_reproduction = 0.85
        cfg.data.sample_size = 3000
        cfg.taxonomy.gold_sample_size = 40
        cfg.clustering.min_leaf_size = 300
        cfg.clustering.k_sweep = [6, 8]
        cfg.clustering.battery_k = [8]
        cfg.representation.alpha_grid = [0.0, 0.1]
        cfg.representation.bakeoff_subsample = 1000
        return run_pipeline(cfg, run_id=rid, stream=False)

    a, b = _run("repro-a"), _run("repro-b")
    assert a["state"].get("chosen_alpha") == b["state"].get("chosen_alpha")
    assert a["state"].get("family_k") == b["state"].get("family_k")
    assert a["state"].get("leaf_count") == b["state"].get("leaf_count")
