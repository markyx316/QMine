"""End-to-end wiring tests.

These run the real graph over a real (small) slice of the real corpus, entirely
offline. They are slower than the unit tests and worth every second: almost every
bug found while building this system was an integration bug — a phase reading an
artifact a later phase produces, a reducer dropping parallel writes, a schema the
offline stand-in could not fill.
"""

from __future__ import annotations

#: Deliverables default to Chinese, so the bottom-up report ships under its
#: Chinese name. Tests assert the *contract* (the report exists and states its
#: limits), not the language it happens to be written in.
# The test config runs `report_language: zh`, so every deliverable carries its
# Chinese filename. Until this release only the bottom-up report honoured the
# setting; the other three shipped English on a zh run.
BOTTOMUP_REPORT = "自下而上聚类最终报告.md"
TOPDOWN_REPORT = "自上而下类目体系最终报告.md"
PANEL_REPORT = "统一度量面板.md"
LEAF_CATALOGUE = "叶清单.md"

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
                 "maintenance.json", BOTTOMUP_REPORT,
                 TOPDOWN_REPORT, PANEL_REPORT, LEAF_CATALOGUE):
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
    for name in (BOTTOMUP_REPORT, TOPDOWN_REPORT):
        text = (gen / name).read_text()
        assert (
            "offline heuristic" in text.lower()
            or "NOT by a language model" in text
            or "offline-heuristic" in text
        ), f"{name} does not disclose who produced its judgments"


def test_reports_state_what_the_numbers_do_not_mean(completed_run):
    """The limits section is mandatory in whatever language the report is written.

    Asserted on meaning rather than on an English phrase, so switching
    `report_language` cannot silently drop the section that keeps the report
    honest.
    """
    text = (Path(completed_run["summary"]["artifact_root"]) / BOTTOMUP_REPORT).read_text()
    zh = "这些数字不代表什么" in text
    en = "do not mean" in text.lower()
    assert zh or en, "no limits section"
    # the distillability caveat must be present either way
    assert ("可蒸馏性" in text) or ("learnab" in text.lower())


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


# ==========================================================================
# Deliverable figures
# ==========================================================================

def test_notebook_executes_and_draws_its_figures(completed_run):
    """The notebook is the figure source, so a silent execution failure is a
    silently figure-less report. Assert both halves of that contract."""
    import nbformat

    gen = Path(completed_run["summary"]["artifact_root"])
    nb_path = gen / "自下而上聚类全流程.ipynb"
    assert nb_path.exists(), "Chinese walkthrough notebook not produced"

    nb = nbformat.read(nb_path, as_version=4)
    errors = [o for c in nb.cells for o in (c.get("outputs") or [])
              if o.get("output_type") == "error"]
    assert not errors, (
        f"{len(errors)} cell(s) raised: "
        + "; ".join(f"{o.get('ename')}: {o.get('evalue')}" for o in errors[:3])
    )

    from qmine.report.builder import NOTEBOOK_FIGURES

    # fig4/fig5 project the embedding space, which fast_mode skips; the rest are
    # computed from JSON artifacts and must be present on every run.
    always = {n for n in NOTEBOOK_FIGURES if n not in ("fig4_spaces", "fig5_intent_split")}
    missing = [n for n in always if not (gen / f"{n}.png").exists()]
    assert not missing, f"notebook executed but drew nothing for: {missing}"


def test_report_embeds_figures_that_exist(completed_run):
    """Every `![](...)` in the Chinese report must resolve next to it."""
    import re

    gen = Path(completed_run["summary"]["artifact_root"])
    md = (gen / BOTTOMUP_REPORT).read_text()
    embedded = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", md)
    assert embedded, "report embeds no figures at all"
    dangling = [s for s in embedded if not (gen / s).exists()]
    assert not dangling, f"report points at missing images: {dangling}"


def test_one_figure_per_quantity(completed_run):
    """Regression guard: the notebook and `ops.viz` can each draw the K sweep,
    the alpha decision, the projection and the panel. Shipping both puts two
    different pictures of one number in a single deliverable."""
    from qmine.report.builder import NOTEBOOK_FIGURES

    gen = Path(completed_run["summary"]["artifact_root"])
    refs = completed_run["state"]["artifacts"]
    for filename, slot in NOTEBOOK_FIGURES.items():
        if not (gen / f"{filename}.png").exists() or slot not in refs:
            continue
        assert Path(refs[slot].path).name == f"{filename}.png", (
            f"slot {slot!r} was filled by {Path(refs[slot].path).name} even though the "
            f"notebook produced {filename}.png — both were drawn"
        )


def test_a_declared_blocking_gate_that_never_fires_is_reported(completed_run):
    """A blocking gate that is declared and never emitted is indistinguishable
    from one that passed. `p2a_pilot_agreement` sat in the blocking list, was
    never emitted by any node, and nothing said so."""
    summary = completed_run["summary"]
    assert "declared_gates_never_evaluated" in summary, (
        "run_summary must state which declared blocking gates never ran"
    )
    assert isinstance(summary["declared_gates_never_evaluated"], list)


def test_a_failed_gate_tells_the_operator_what_to_do(completed_run):
    """`remediation` is the only field that says how to fix a failure; it was
    computed on every gate and then dropped from run_summary.json."""
    gates = completed_run["summary"]["gates"]
    assert gates, "no gates recorded"
    assert all("remediation" in g for g in gates.values()), (
        "remediation dropped from the summary — the halt reason ships without the fix"
    )
    assert any(g.get("remediation") for g in gates.values()), "every remediation is empty"


def test_every_authored_rationale_reaches_the_reader_in_the_report_language(completed_run):
    """The audit trail is the reasoning content of the deliverable, so an
    untranslated rationale defeats the reason for including it. This asserts
    coverage rather than eyeballing: any authored prose that reaches a Chinese
    report and has no translation shows up here, so edits to the English cannot
    silently drop a mapping."""
    import re
    from pathlib import Path

    from qmine.report.i18n import prose

    gen = Path(completed_run["summary"]["artifact_root"])
    md = (gen / BOTTOMUP_REPORT).read_text()

    # Three consecutive long lowercase words is running English prose; Chinese
    # reports legitimately carry identifiers, metric names and code spans.
    english = [l for l in md.splitlines()
               if re.search(r"[a-z]{6,}\s+[a-z]{6,}\s+[a-z]{6,}", l)]
    assert not english, (
        "untranslated prose reached the Chinese report:\n  "
        + "\n  ".join(l[:120] for l in english[:6])
    )

    # And the mapping must actually be doing work, not vacuously passing because
    # the report happens to contain no rationales at all.
    assert prose("Low kappa means the guide is ambiguous, whatever follows") != \
        "Low kappa means the guide is ambiguous, whatever follows"
    assert prose("a string nobody has translated") == "a string nobody has translated"


def test_the_delivered_leaves_gate_reaches_the_operator(completed_run):
    """A blocking gate absent from state is a blocking gate that cannot block.

    `p10_delivered_leaves_named` is in `cfg.gates.blocking`, but `p10_deploy`
    called `deps.gate(...)` and discarded the return — and `deps.gate` registers
    nothing, it only builds the record. The gate existed in the log and nowhere
    a person or a test would look.
    """
    gates = completed_run["summary"].get("gates", {})
    assert "p10_delivered_leaves_named" in gates, (
        f"the blocking delivery gate never reached state; recorded: {sorted(gates)}"
    )


def test_every_delivered_leaf_carries_a_name(completed_run):
    """A leaf created AFTER p7 named the partition still reaches the reader.

    `p7_all_leaves_named` is blocking and it passed on live38 — "all 29 leaves
    named" — while the delivered table carried 36 leaves and **4,931 rows (9.9%)
    had an empty name**, because p8 governance ran 6 `split_leaf` and 2
    `isolate_leaf` prescriptions afterwards and `p10_deliver` builds its name
    column from p7's namings plus renames only. A gate placed before the
    operation that invalidates it guarantees nothing.

    This asserts on the table the reader actually receives. NOTE it is a guard for
    LIVE runs only: the offline stand-in never issues a `split_leaf`, so this
    passes with the fix disabled (verified). The load-bearing tests that can
    actually fail are in `test_governance_naming.py`.
    """
    import pandas as pd

    df = pd.read_csv(Path(completed_run["summary"]["artifact_root"]) / "labels_full.csv")
    unnamed = df[df["bu_leaf_name"].isna() | (df["bu_leaf_name"].astype(str).str.strip() == "")]
    assert unnamed.empty, (
        f"{len(unnamed)} of {len(df)} delivered rows have no leaf name "
        f"(leaves {sorted(unnamed['bu_leaf'].unique())[:10]})"
    )
    assert df["bu_leaf"].nunique() == df.groupby("bu_leaf")["bu_leaf_name"].nunique().sum(), \
        "a leaf id maps to more than one name"


def test_the_chinese_report_does_not_contradict_its_own_metrics_table(completed_run):
    """Self-consistency of the shipped document: the executive summary's
    "N 家族 / M 叶" must equal the 簇数 row of the table below it.

    A LIVE-run guard — offline never splits, so pre- and post-governance counts
    coincide and this cannot fail there. `test_the_reported_shape_is_the_delivered_shape`
    is the one that can.
    """
    import re

    md = (Path(completed_run["summary"]["artifact_root"]) / "自下而上聚类最终报告.md").read_text()
    summary = re.search(r"\*\*(\d+) 家族 / (\d+) 叶\*\*", md)
    assert summary, "executive summary no longer states the shape"
    row = re.search(r"^\| 簇数.*?\|\s*(\d+)\s*\|\s*\*\*(\d+)\*\*\s*\|\s*(\d+)\s*\|", md, re.M)
    assert row, "metrics table no longer carries a 簇数 row"
    assert (summary.group(1), summary.group(2)) == (row.group(2), row.group(3)), (
        f"summary says {summary.group(1)}家族/{summary.group(2)}叶 but the table says "
        f"{row.group(2)}/{row.group(3)}"
    )


def test_every_report_is_written_in_the_configured_language(completed_run):
    """`report_language: zh` must reach EVERY deliverable, not just one.

    Only the bottom-up report and the notebook ever branched on the setting.
    `Report_Uniform_Panel.md` shipped 0% Chinese, `Report_TopDown_Approach.md`
    38% and `Leaf_Catalogue.md` 62% — on a run configured `zh`. The top-down
    route is half the methodology and it reached a Chinese reader in English.

    The check is on SECTION HEADINGS, not on the document's character mix. A
    character-share bar cannot separate the two cases: measured on live38 the
    English catalogue was 61.9% Han (its leaf names come from the namer and are
    Chinese) while the genuinely Chinese bottom-up report was only 46.9% (it
    quotes metric names and class codes). Headings are pure authored scaffolding
    with no agent content, and they separate the two perfectly — 0 English
    headings in all four correct reports, 7 / 11 / 14 in the three English ones.
    """
    import re

    gen = Path(completed_run["summary"]["artifact_root"])
    for name in (BOTTOMUP_REPORT, TOPDOWN_REPORT, PANEL_REPORT, LEAF_CATALOGUE):
        english = []
        for line in (gen / name).read_text().splitlines():
            if not re.match(r"^#{1,3} ", line):
                continue
            body = re.sub(r"^#+ ", "", line)
            # A heading of pure codes/numbers is fine; one with real English words
            # and no Han character is untranslated scaffolding.
            if not re.search(r"[\u4e00-\u9fff]", body) and re.search(r"[A-Za-z]{3}", body):
                english.append(body[:60])
        assert not english, (
            f"{name} has {len(english)} untranslated headings: {english[:3]}"
        )


def test_a_family_is_named_after_the_leaves_it_actually_contains(completed_run):
    """The auditor's `family_id` is not the partition's, and joining them broke.

    `tree_naming["audit"]["families"]` numbers its own families — 19 of them on
    live38 — while the partition had 10 pre-governance and 12 final. Looking a
    partition family up by integer id matched a different family for **19 of 19**,
    and the shipped catalogue titled a family of four classical-poetry leaves
    "中考录取分数与学校排名查询" (high-school admission scores). Every family
    heading in the delivered document was wrong.

    The join must go through `leaf_ids`.
    """
    import json

    import numpy as np

    from qmine.report._shape import family_names

    gen = Path(completed_run["summary"]["artifact_root"])
    naming = json.loads((gen / "tree_naming.json").read_text())
    # `*_final.npy` only exists when governance actually changed the partition,
    # which the offline stand-in never does.
    def _arr(final: str, base: str):
        return np.load(gen / (final if (gen / final).exists() else base))

    fam = _arr("leaf_family_final.npy", "leaf_family.npy")
    labels = _arr("leaf_labels_final.npy", "leaf_labels.npy")
    sizes = np.bincount(labels, minlength=len(fam))

    names = family_names(naming, fam, sizes)
    audit = (naming.get("audit") or {}).get("families") or []
    if not audit or not any(f.get("leaf_ids") for f in audit):
        pytest.skip("this run's auditor produced no family groupings to join on")

    assert names, "no family could be named at all"
    # Every name must be traceable to a leaf that is actually in that family.
    by_leaf = {int(l): str(f.get("name_zh") or "")
               for f in audit for l in (f.get("leaf_ids") or [])}
    for fid, label in names.items():
        members = [l for l in by_leaf if l < len(fam) and int(fam[l]) == fid]
        assert members, f"family {fid} was named {label!r} with no member leaves"
        assert any(by_leaf[l] and by_leaf[l] in label for l in members), (
            f"family {fid} is titled {label!r}, which is not the audit name of any "
            f"leaf it contains ({sorted({by_leaf[l] for l in members})})"
        )
    assert len(set(names.values())) == len(names), \
        f"two partition families share a title, so a reader cannot tell them apart: {names}"


def test_the_report_names_the_metric_that_actually_located_k(completed_run):
    """The K label must name the locator, not a metric that only rejects.

    `test_k_is_located_by_intent_alignment_and_only_filtered_by_stability` guards
    the CODE, and the code was right — `locator` reads `intent_alignment_ami`.
    The LABEL contradicted it in three shipped places, because the estimate was
    stored under the key `stability_peak_k`: the report printed
    「稳定性峰 K (主证据)」 and fig1's title read 「定案 K=10 取自稳定性峰」.

    On live38 that was not merely mis-worded, it was false: the true stability
    peak was k=8 (ARI 0.9191) and the delivered k=10 ranked **9th of 14** on
    stability. A reader was told the headline granularity decision rested on the
    one metric the code, the artifact and a passing test all say cannot rank K.
    """
    import json

    gen = Path(completed_run["summary"]["artifact_root"])
    gran = json.loads((gen / "granularity.json").read_text())
    tri = gran.get("triangulation", {})
    locator = str(tri.get("locator", ""))
    assert locator, "the granularity artifact no longer records what located K"

    md = (gen / BOTTOMUP_REPORT).read_text()
    if locator.startswith("intent_alignment_ami"):
        assert "稳定性峰 K" not in md, (
            "the report credits the stability peak for a K located by intent alignment"
        )
        # and the sweep must actually disagree, or the claim would be harmless
        sweep = gran.get("k_sweep") or tri.get("k_sweep") or []
        if sweep and all(r.get("stability_ari") is not None for r in sweep):
            peak = max(sweep, key=lambda r: r["stability_ari"])["k"]
            chosen = tri.get("chosen_family_k")
            if peak != chosen:
                assert "意图对齐" in md, (
                    f"K={chosen} was located by AMI while the stability peak is "
                    f"K={peak}; the report must say which one chose"
                )
