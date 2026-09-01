"""`qmine render` — rebuilding a finished run's deliverables.

Every report is a projection of artifacts already on disk, so re-deriving them
costs only CPU. Without this there was no way to check a report fix against a
real run short of paying for the whole pipeline again — which is how a section
that had NEVER once rendered (`§2.1 L2 子意图`) survived to live42, and how the
reference shelf sat in code for a day with no run producing it.
"""
from __future__ import annotations

import inspect

import pytest


def test_render_never_writes_over_the_generation_it_read():
    """A delivered document is the evidence of what a run said. A re-render that
    overwrote it would destroy the record of the very defect it exists to fix —
    and this project's rule for re-deriving anything is a NEW generation.
    """
    from qmine.runner import render_run

    src = inspect.getsource(render_run)
    assert "new_generation(" in src, "a render must open a new generation"
    body = src[src.index("store = "):]
    assert "src_store" not in body.split("new_generation")[1][:400], (
        "the writing store must not be the source store")


def test_no_agents_switches_the_agents_OFF_rather_than_to_a_stand_in():
    """The offline stand-in returns complete-looking prose that no model wrote.
    In a deliverable that is worse than an honest absence: on the first run of
    this command the auditor produced three refused edits against a file called
    "[offline-heuristic] file" and three findings whose claim was the empty
    string, and wrote them into 交付前审核报告.md.
    """
    from qmine.runner import render_run

    src = inspect.getsource(render_run)
    for gate in ("final_report", "interpret_results", "observe_phases",
                 "delivery_audit"):
        assert f"cfg.{gate} = False" in src, f"{gate} must be switched off"
    # Comments in this block legitimately mention the stand-in; test the CODE.
    block = src.split("if not agents:")[1][:600]
    code = "\n".join(ln for ln in block.splitlines()
                     if not ln.strip().startswith("#"))
    assert "offline" not in code, (
        "switching to the stand-in is not the same as switching the agent off")
    assert "provider" not in code, "the provider must not be rewritten to fake one"


def test_the_pre_delivery_auditor_has_a_gate_at_all():
    """It had none — it ran unconditionally, so a caller asking for no model
    calls could not get them."""
    from qmine.config import QMineConfig
    from qmine.graph.nodes.delivery import p11_report

    assert hasattr(QMineConfig(), "delivery_audit")
    assert 'getattr(deps.cfg, "delivery_audit"' in inspect.getsource(p11_report)


def test_render_loads_the_environment_before_detecting_providers():
    """`--agents --provider router` detected no keys and silently ran the
    OFFLINE stand-in, which passes every check it is given — so the render
    reported "10/10 sections verified" for a report no model had written. Every
    other command that can make a call already calls `_load_env` first.
    """
    from qmine import cli

    src = inspect.getsource(cli.render)
    assert "_load_env()" in src
    assert src.index("_load_env()") < src.index("render_run("), (
        "the environment must be loaded before provider detection runs")


def test_a_rendered_generation_can_itself_be_rendered():
    """`write_summary` runs at the end of a RUN, so a generation produced by
    rendering had no `run_summary.json` — and that file is where `recover_state`
    finds the gate ledger when the checkpoint has no thread for it. Rendering
    gen01→gen02 then gen02→gen03 lost `gates` and `completed_phases` outright,
    and the second render's reports silently shrank.
    """
    from qmine.runner import render_run

    src = inspect.getsource(render_run)
    assert "write_summary(" in src, "a rendered generation needs its own summary"


def test_recover_state_reports_what_it_could_not_find():
    """A narrative written against a starved state does not come out cautious,
    it comes out invented — the same failure the fact-sheet work established. A
    caller about to spend money on agents has to see the hole first.
    """
    from qmine.runner import recover_state

    src = inspect.getsource(recover_state)
    assert "missing" in src
    sig = inspect.signature(recover_state)
    assert sig.return_annotation != inspect.Signature.empty
    assert "list[str]" in str(sig.return_annotation), (
        "the missing channels must be part of the return, not a log line")


@pytest.mark.parametrize("gate", ["final_report", "delivery_audit"])
def test_an_agent_step_that_is_off_still_records_that_it_was_off(gate):
    """A deliverable that is absent because a step was disabled must say so,
    rather than looking like a step that ran and found nothing."""
    from qmine.graph.nodes.delivery import p11_report

    src = inspect.getsource(p11_report)
    assert '"skipped": "disabled"' in src


def test_one_threshold_can_name_several_observed_values():
    """`min_kappa` matches `kappa` AND `self_consistency_kappa`. Pairing only the
    first silently drops the rest — and that regressed
    `_passed_below_threshold`, which stopped flagging a gate whose LATER number
    is the one under its bar. The Chinese 「带保留通过」 prefix that flag adds was
    the only CJK on a line whose message is authored in English, so an
    untranslated gate conclusion reached a Chinese report.
    """
    from qmine.records import GateResult, gate_metric_pairs, paired_gate_metric

    g = GateResult(name="g", phase="p", status="passed",
                   observed={"kappa": 0.89, "self_consistency_kappa": 0.61},
                   threshold={"min_kappa": 0.70})
    pairs = gate_metric_pairs(g)
    assert len(pairs) == 2, f"a threshold naming two observations lost one: {pairs}"
    # The figure plots ONE bar, so the primary is still a single pair.
    assert paired_gate_metric(g) == pairs[0]

    from qmine.report.zh_bottomup import _passed_below_threshold

    assert _passed_below_threshold(g), "0.61 is under the 0.70 bar"


def test_a_boolean_observation_is_not_a_measurement():
    """`isinstance(True, int)` is True in Python, so a gate asserting
    `{"lopsided": True}` contributed a value of 1 and was plotted as if it had
    cleared its bar."""
    from qmine.records import GateResult, gate_metric_pairs

    g = GateResult(name="g", phase="p", status="warned",
                   observed={"lopsided": True}, threshold={"max_lopsided": False})
    assert gate_metric_pairs(g) == []


def test_a_stored_gate_can_be_read_back_into_state():
    """`name` and `phase` are required on GateResult and are not in the
    serialised value — the name is the dict key. Validating the value alone
    raises, which loses the whole gate ledger on a re-render."""
    import json

    from qmine.records import GateResult

    stored = json.loads(GateResult(name="p2b_kappa", phase="p2b",
                                   status="passed").model_dump_json())
    stored.pop("name")
    reread = GateResult.model_validate({"name": "p2b_kappa",
                                        "phase": str(stored.get("phase") or ""),
                                        **stored})
    assert reread.name == "p2b_kappa" and reread.phase == "p2b"


def test_a_render_does_not_overwrite_the_run_s_own_spend_record():
    """`_wire_events` wrote `root/usage.json` on every emit, and a render calls
    it too — so re-rendering live42 replaced that run's record of 702 calls and
    $29.69 with the render's 11 calls and $0.78. Unrecoverably: live42's teardown
    bug meant no `run_summary.json` held a second copy.

    A render's spend is real and worth recording; it belongs in the generation
    the render writes, not over the run's.
    """
    import inspect

    from qmine.runner import _wire_events, render_run

    assert "usage_path" in inspect.signature(_wire_events).parameters
    src = inspect.getsource(render_run)
    call = src[src.index("_wire_events("):]
    call = call[:call.index(")\n")]
    assert "usage_path=" in call, "a render must redirect its usage snapshot"
    assert "gen" in call, "…into its own generation"


def test_a_second_render_opens_a_NEW_generation_not_the_same_one():
    """`ArtifactStore.new_generation` increments from ITSELF, so a store opened at
    the source generation returns `src + 1`. Rendering gen01 twice therefore wrote
    both renders into gen02 — a no-agents pass and an agent pass interleaved in
    one directory, which is an incoherent generation rather than two records.

    The store resolves artifacts across generations, so reading from an old
    generation while writing to the newest is exactly what is wanted.
    """
    import inspect

    from qmine.runner import render_run

    src = inspect.getsource(render_run)
    assert "latest_generation(root) + 1" in src, (
        "the target must be after the LATEST generation, not after the source")
    assert "src_store.new_generation(" not in src, (
        "opening from the source store reintroduces the overwrite")


def test_a_rendered_generation_sees_the_run_s_evidence(tmp_path):
    """A generation is NOT self-contained. Artifacts are written once and
    inherited through `index.jsonl`; only the reports are rewritten. So
    `build_catalogue` reading `gen_dir / name` directly found nothing in a
    generation that did not itself produce the artifact — which is exactly what
    `qmine render` writes.

    The first agent-written report produced this way passed ALL NINE sections and
    described an empty run: `n_named = 0` against 58 named leaves,
    `n_ledger_entries = 0` against 26. Every number it cited really was in its
    fact sheet, so no guardrail could fire — the sheets were empty. A report about
    nothing is worse than one with a marked hole, because nothing marks it.

    Exercised through `build_catalogue`, not through `_read`: the defect was the
    CALL SITE, and a test on the helper alone survives reverting it.
    """
    import types

    from qmine.report.narrative_brief import build_catalogue

    gen = tmp_path / "gen02"
    gen.mkdir()                       # a rendered generation: no artifacts in it

    store_only = {
        "naming_cards": {"cards": [{"leaf_id": i} for i in range(58)]},
        "governance": {"ledger": [{"id": i} for i in range(26)],
                       "execution": {}, "settled": {}},
    }

    class _Store:
        gen_dir = gen

        def has(self, n):
            return n in store_only

        def load(self, n):
            return store_only[n]

    st = _Store()
    deps = types.SimpleNamespace(
        store=st, has=st.has, load=st.load,
        cfg=types.SimpleNamespace(domain=types.SimpleNamespace(key="k"),
                                  report_language="zh"),
        emit=lambda *a, **k: None)

    cat = build_catalogue({}, deps)
    assert cat["naming"].facts["n_named"] == 58, (
        "the catalogue read the generation directory instead of the store, so "
        "the narrative describes an empty run")
    assert cat["governance"].facts["n_ledger_entries"] == 26
