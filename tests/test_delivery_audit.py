"""The one agent allowed to change a deliverable, and the rules that bound it.

Every other agent here describes, because every other agent could be wrong in a
way nothing would catch. This one edits. That is only defensible because the
operation it is given is checkable: an anchored replacement whose anchor is proven
unique, whose numbers come from the artifact it cites, and whose language is
verified before it lands.

These tests are the enforcement. Each one is a way an agent with write access
could ship a mistake, and the rule that stops it.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from qmine.ops.edits import ProposedEdit, apply_edits, validate_edit

ARTIFACTS = {
    "hierarchy_meta": {"n_leaves": 39, "n_families": 11,
                       "leaves_per_family": {"0": 3, "1": 4}},
    "metrics_panel": {"sets": {"leaves": {"metrics": {"stability_ari": 0.8586}}}},
}
DOC = (
    "# 报告\n"
    "本次交付共 **29 叶**, 分布在 10 个家族。\n"
    "稳定性 0.8586。\n"
    "本次交付共 29 叶。\n"
)
TEXTS = {"报告.md": DOC}


def _edit(**kw):
    base = dict(file="报告.md", anchor="本次交付共 **29 叶**, 分布在 10 个家族。",
                replacement="本次交付共 **39 叶**, 分布在 11 个家族。",
                reason="p8 治理改写了树, 文档仍是治理前的数字",
                artifact_key="hierarchy_meta")
    base.update(kw)
    return ProposedEdit(**base)


def test_a_correct_sourced_edit_is_applied(tmp_path):
    """The case the whole mechanism exists to allow."""
    (tmp_path / "报告.md").write_text(DOC, encoding="utf-8")
    res = apply_edits(tmp_path, [_edit()], ARTIFACTS)
    assert len(res.applied) == 1 and not res.refused
    assert "39 叶" in (tmp_path / "报告.md").read_text(encoding="utf-8")


def test_the_pre_edit_file_is_kept(tmp_path):
    """An audited edit differs from a silent one by being reversible."""
    (tmp_path / "报告.md").write_text(DOC, encoding="utf-8")
    apply_edits(tmp_path, [_edit()], ARTIFACTS)
    pre = tmp_path / "报告.pre_audit.md"
    assert pre.exists() and pre.read_text(encoding="utf-8") == DOC


def test_an_anchor_that_is_not_there_is_refused():
    """The agent is correcting text it misread. `assert old in s`, as a rule."""
    ok, why = validate_edit(_edit(anchor="这句话不在文档里"), TEXTS, ARTIFACTS)
    assert not ok and "does not appear" in why


def test_an_ambiguous_anchor_is_refused():
    """`本次交付共 29 叶` occurs twice; the agent cannot know which it meant.

    Replacing the first is a coin flip, and a coin flip in a delivered document
    is exactly the silent no-op that cost this project a debugging cycle.
    """
    ok, why = validate_edit(_edit(anchor="本次交付共"), TEXTS, ARTIFACTS)
    assert not ok and "appears" in why and "unambiguous" in why


def test_a_number_not_in_the_cited_artifact_is_refused():
    """THE load-bearing rule. Without it the agent may write any number at all."""
    ok, why = validate_edit(
        _edit(replacement="本次交付共 **41 叶**, 分布在 11 个家族。"), TEXTS, ARTIFACTS)
    assert not ok and "41" in why and "hierarchy_meta" in why


def test_the_citation_defines_the_pool_not_the_whole_run():
    """`agents.verify` documents its blind spot: a real number for the wrong
    quantity passes when the pool is large. The citation is how the pool is kept
    small — 0.8586 is a genuine value of this run and is NOT in `hierarchy_meta`,
    so an edit citing `hierarchy_meta` may not use it.
    """
    ok, why = validate_edit(
        _edit(replacement="本次交付共 0.8586 叶。"), TEXTS, ARTIFACTS)
    assert not ok and "0.8586" in why

    ok2, _ = validate_edit(
        _edit(anchor="稳定性 0.8586。", replacement="稳定性 0.8586 (叶层)。",
              artifact_key="metrics_panel.sets.leaves.metrics"), TEXTS, ARTIFACTS)
    assert ok2, "the same number IS allowed when the edit cites where it lives"


def test_an_uncited_edit_is_refused():
    """An unsourced correction is a rewrite with a confident tone."""
    ok, why = validate_edit(_edit(artifact_key=""), TEXTS, ARTIFACTS)
    assert not ok and "not an artifact" in why
    ok2, why2 = validate_edit(_edit(artifact_key="vibes.n_leaves"), TEXTS, ARTIFACTS)
    assert not ok2 and "not an artifact" in why2


def test_an_edit_that_deletes_a_number_without_replacing_it_is_refused():
    """Correcting 29 to 39 is a fix. Turning 29 into "若干" removes evidence."""
    ok, why = validate_edit(
        _edit(replacement="本次交付共若干叶, 分布在若干家族。"), TEXTS, ARTIFACTS)
    assert not ok and "removes evidence" in why


def test_an_english_replacement_is_refused_on_a_zh_run():
    """A run configured `zh` shipped English into half its deliverables, and the
    fix for it introduced fresh English of its own. The check is on the edit."""
    ok, why = validate_edit(
        _edit(replacement="delivered 39 leaves across 11 families"), TEXTS, ARTIFACTS)
    assert not ok and "Chinese" in why

    ok2, _ = validate_edit(
        _edit(replacement="本次交付共 **39 叶**。"), TEXTS, ARTIFACTS, language="zh")
    assert ok2


def test_a_bare_number_replacement_is_allowed_in_a_table_cell():
    """The language rule must not block correcting `| 29 |` to `| 39 |`."""
    texts = {"报告.md": "| 叶数 | 29 |\n"}
    ok, why = validate_edit(
        ProposedEdit(file="报告.md", anchor="| 叶数 | 29 |", replacement="| 叶数 | 39 |",
                     reason="治理后的叶数", artifact_key="hierarchy_meta"),
        texts, ARTIFACTS)
    assert ok, why


def test_a_check_that_is_false_against_the_artifacts_refuses_the_edit():
    """This test previously PINNED THE BUG.

    It asserted that a check evaluating TRUE should refuse the edit — the
    observation semantics, applied to an edit, where they are inverted. So the
    suite was actively protecting the defect that rejected four correct
    corrections on live40.

    An edit's check states what the artifacts DO say. True means sourced;
    false means the correction has no basis in the artifact it cites.
    """
    # `n_leaves` really is 39, so a correction to 39 IS sourced.
    ok, why = validate_edit(_edit(check="hierarchy_meta.n_leaves == 39"), TEXTS, ARTIFACTS)
    assert ok, f"a true, sourced check must permit the edit: {why}"

    # A check the artifacts contradict means the edit is inventing its number.
    bad, why2 = validate_edit(_edit(check="hierarchy_meta.n_leaves == 999"), TEXTS, ARTIFACTS)
    assert not bad and "not sourced" in why2



def test_an_edit_with_no_reason_is_refused():
    """An undocumented edit cannot be reviewed, so it cannot be allowed."""
    ok, why = validate_edit(_edit(reason="  "), TEXTS, ARTIFACTS)
    assert not ok and "no reason" in why


@pytest.mark.parametrize("target", [
    "hierarchy_meta.json", "labels_full.csv", "leaf_labels.npy",
    "config.resolved.yaml", "自下而上聚类全流程.ipynb", "../../src/qmine/config.py",
])
def test_only_rendered_prose_can_be_edited(target, tmp_path):
    """A report DESCRIBES a measurement; an artifact IS one.

    Correcting a document changes no measurement. Editing an artifact changes the
    measurement itself, and no agent gets that — including through a filename
    that tries to walk out of the generation directory.
    """
    ok, why = validate_edit(_edit(file=target), TEXTS, ARTIFACTS)
    assert not ok, f"{target} must not be editable"
    assert "editable deliverables" in why or "artifacts, not descriptions" in why


def test_a_refused_edit_leaves_the_file_untouched(tmp_path):
    """Half-applying a batch would leave a document nobody has ever seen whole."""
    (tmp_path / "报告.md").write_text(DOC, encoding="utf-8")
    res = apply_edits(tmp_path, [_edit(replacement="共 **999 叶**。")], ARTIFACTS)
    assert not res.applied and len(res.refused) == 1
    assert (tmp_path / "报告.md").read_text(encoding="utf-8") == DOC
    assert not (tmp_path / "报告.pre_audit.md").exists()


def test_every_refusal_is_recorded_with_the_rule_that_bit(tmp_path):
    """A report showing only successes would be a sales document."""
    (tmp_path / "报告.md").write_text(DOC, encoding="utf-8")
    res = apply_edits(tmp_path, [
        _edit(),
        _edit(anchor="不存在的锚点"),
        _edit(replacement="共 **999 叶**。"),
    ], ARTIFACTS)
    assert len(res.applied) == 1 and len(res.refused) == 2
    assert all(r["why"] for r in res.refused)
    assert res.as_record()["n_refused"] == 2


def test_the_edit_cap_bounds_the_blast_radius(tmp_path):
    """Beyond a point this is a rewrite, which is a different operation."""
    (tmp_path / "报告.md").write_text("\n".join(f"第{i}行 29 叶" for i in range(30)),
                                     encoding="utf-8")
    edits = [ProposedEdit(file="报告.md", anchor=f"第{i}行 29 叶",
                          replacement=f"第{i}行 39 叶", reason="治理后",
                          artifact_key="hierarchy_meta") for i in range(30)]
    res = apply_edits(tmp_path, edits, ARTIFACTS, max_edits=3)
    assert len(res.applied) == 3
    assert any("cap" in r["why"] for r in res.refused)


def test_two_edits_are_validated_against_the_same_original(tmp_path):
    """Otherwise the outcome depends on ordering the auditor never saw.

    Both edits are proposed against the document as it was handed over. An
    anchor made unique — or made ambiguous — by a sibling edit must not silently
    change whether the other one lands.
    """
    (tmp_path / "报告.md").write_text(DOC, encoding="utf-8")
    res = apply_edits(tmp_path, [
        _edit(),
        _edit(anchor="稳定性 0.8586。", replacement="稳定性 0.8586 (叶层, 交叉验证)。",
              artifact_key="metrics_panel.sets.leaves.metrics",
              reason="没有说明这是哪一层的指标"),
    ], ARTIFACTS)
    assert len(res.applied) == 2, res.refused
    out = (tmp_path / "报告.md").read_text(encoding="utf-8")
    assert "39 叶" in out and "叶层, 交叉验证" in out


def test_a_dry_run_changes_nothing(tmp_path):
    (tmp_path / "报告.md").write_text(DOC, encoding="utf-8")
    res = apply_edits(tmp_path, [_edit()], ARTIFACTS, dry_run=True)
    assert len(res.applied) == 1
    assert (tmp_path / "报告.md").read_text(encoding="utf-8") == DOC


def test_the_audit_report_says_so_when_the_audit_did_not_run():
    """Silence is the failure mode. An unaudited document must not look audited."""
    from types import SimpleNamespace

    from qmine.report.zh_audit import build

    deps = SimpleNamespace(store=SimpleNamespace(gen_dir=Path("/tmp/gen01")))
    md = build({"ran": False, "skipped": "ProviderDown"}, {"run_id": "t"}, deps)
    assert "没有经过交付前审核" in md and "ProviderDown" in md


def test_the_audit_report_prints_refusals_next_to_the_edits():
    from types import SimpleNamespace

    from qmine.report.zh_audit import build

    deps = SimpleNamespace(store=SimpleNamespace(gen_dir=Path("/tmp/gen01")))
    md = build({
        "ran": True, "n_applied": 1, "n_refused": 1,
        "applied": [{"file": "a.md", "anchor": "29 叶", "replacement": "39 叶",
                     "reason": "治理后的数字", "artifact_key": "hierarchy_meta"}],
        "refused": [{"file": "a.md", "reason": "想改稳定性", "why": "41 is not in hierarchy_meta"}],
        "dismissed": ["p5 的告警只是 underpowered, 文档已写明"],
    }, {"run_id": "t"}, deps)
    assert "已应用的修改" in md and "被拒绝的修改" in md
    assert "41 is not in hierarchy_meta" in md
    assert "读过并判定无需处理的告警" in md


def test_a_non_markdown_file_is_refused_even_if_it_is_in_scope():
    """Isolates the suffix rule from the whitelist rule.

    Found by mutation: both checks refuse `hierarchy_meta.json`, so removing
    either one alone still passed. Two independent defences is the right design
    — but a test that cannot tell them apart proves neither.
    """
    texts = {"hierarchy_meta.json": '{"n_leaves": 29}'}
    ok, why = validate_edit(
        ProposedEdit(file="hierarchy_meta.json", anchor="29", replacement="39",
                     reason="治理后", artifact_key="hierarchy_meta"),
        texts, ARTIFACTS)
    assert not ok and "artifacts, not descriptions" in why


def test_a_markdown_file_this_run_did_not_produce_is_refused_not_crashed():
    """Isolates the whitelist rule, and pins that it FAILS rather than raises.

    Without it `texts[name]` is a KeyError that takes down p11 after the reports
    are already written — a delivery lost to a bad filename.
    """
    ok, why = validate_edit(_edit(file="不存在的报告.md"), TEXTS, ARTIFACTS)
    assert not ok and "editable deliverables" in why


def test_a_path_that_walks_out_of_the_generation_directory_cannot_reach_a_file(tmp_path):
    """Only the basename is ever used, so `../` reaches nothing."""
    outside = tmp_path.parent / "outside.md"
    outside.write_text("29 叶", encoding="utf-8")
    (tmp_path / "报告.md").write_text(DOC, encoding="utf-8")
    res = apply_edits(tmp_path, [
        ProposedEdit(file="../outside.md", anchor="29 叶", replacement="39 叶",
                     reason="x", artifact_key="hierarchy_meta")], ARTIFACTS)
    assert not res.applied and res.refused
    assert outside.read_text(encoding="utf-8") == "29 叶"


def test_a_sibling_edit_cannot_disambiguate_an_anchor_the_auditor_saw_as_ambiguous(tmp_path):
    """Every edit is judged against the document the auditor was shown.

    Found by mutation: validating against the running text instead let edit 2
    land only because edit 1 happened to run first and removed the other
    occurrence. The auditor never saw that intermediate document, so whether its
    edit applies would depend on an ordering it could not reason about — and the
    same batch would behave differently if the model listed them the other way.
    """
    doc = "共 29 叶 (摘要)。\n共 29 叶\n"
    (tmp_path / "报告.md").write_text(doc, encoding="utf-8")
    res = apply_edits(tmp_path, [
        ProposedEdit(file="报告.md", anchor="共 29 叶 (摘要)。", replacement="共 39 叶 (摘要)。",
                     reason="治理后", artifact_key="hierarchy_meta"),
        ProposedEdit(file="报告.md", anchor="共 29 叶", replacement="共 39 叶",
                     reason="治理后", artifact_key="hierarchy_meta"),
    ], ARTIFACTS)

    assert len(res.applied) == 1
    assert len(res.refused) == 1 and "unambiguous" in res.refused[0]["why"]
    assert (tmp_path / "报告.md").read_text(encoding="utf-8") == "共 39 叶 (摘要)。\n共 29 叶\n"


def test_an_unfixable_finding_goes_through_the_same_verification_as_an_observation(tmp_path):
    """A ledger that accumulates blanks is a ledger nobody reads.

    The first end-to-end run put three empty-claim rows into the ledger, because
    `unfixable` was recorded raw while every other observation had to cite a
    resolving artifact key. Same discipline, same reason: a claim that cannot be
    traced is not evidence.
    """
    from types import SimpleNamespace

    from qmine.agents.observe import verified_observations

    def _o(claim, key, check=""):
        return SimpleNamespace(severity="warn", claim=claim, artifact_key=key,
                               evidence="", would_change="", check=check,
                               _verdict="unverifiable")

    raw = SimpleNamespace(checked=[], observations=[
        _o("", ""),                                           # the blank rows
        _o("树形状不一致", "vibes.not_real"),                    # uncited
        _o("n_leaves 与分解不符", "hierarchy_meta.n_leaves"),     # good
    ])
    res = verified_observations(raw, ARTIFACTS)
    assert len(res.kept) == 1 and res.kept[0].artifact_key == "hierarchy_meta.n_leaves"
    assert len(res.dropped) == 2


def test_an_edits_check_must_be_TRUE_because_it_states_what_the_artifacts_say():
    """The semantics are the OPPOSITE of an observation's, and this was backwards.

    An observation asserts what *should* hold, and the assertion FAILING confirms
    a defect. An edit asserts what the artifacts *do* say — the ground truth the
    document is being aligned to — so the assertion HOLDING is what makes the
    correction well-founded.

    Measured on live40: the auditor wrote
    `adversarial_validation.estimated_accuracy == 0.82` (true) to fix a report
    claiming adversarial accuracy was HIGHER than cross-validation when
    0.82 < 0.8625. The edit was refused for being right, and the wrong claim
    shipped. Three more correct fixes died the same way.
    """
    arts = {"metrics": {"accuracy": 0.82}}
    texts = {"报告.md": "对抗验证准确率 0.90, 高于交叉验证。\n"}

    sourced = ProposedEdit(
        file="报告.md", anchor="对抗验证准确率 0.90, 高于交叉验证。",
        replacement="对抗验证准确率 0.82, 低于交叉验证。",
        reason="artifact 记录 0.82, 原文方向写反",
        artifact_key="metrics", check="metrics.accuracy == 0.82")
    ok, why = validate_edit(sourced, texts, arts)
    assert ok, f"a correctly-sourced edit was refused: {why}"

    unsourced = ProposedEdit(
        file="报告.md", anchor="对抗验证准确率 0.90, 高于交叉验证。",
        replacement="对抗验证准确率 0.82, 低于交叉验证。",
        reason="x", artifact_key="metrics", check="metrics.accuracy == 0.99")
    bad, why2 = validate_edit(unsourced, texts, arts)
    assert not bad and "not sourced" in why2


def test_the_auditor_can_cite_a_gate_and_a_finding_not_only_an_artifact():
    """It is HANDED the gate ledger and the findings ledger and told to cite its
    source — so refusing a gate citation as "unsourced" punished exactly the
    behaviour the prompt asks for.

    Gates live inside `run_summary.json`, not as a top-level artifact key. On
    live40 that rejected 3 of the auditor's 4 correct corrections.
    """
    arts = {
        "hierarchy_meta": {"n_leaves": 39},
        "gates": {"p2b_kappa": {"status": "passed", "observed": {"n": 2982}}},
        "findings": {"abc123": {"phase": "p8", "claim": "x", "evidence": "17"}},
    }
    texts = {"报告.md": "κ 在 2983 行上计算。\n另有 12 条处方。\n"}

    from_gate = ProposedEdit(
        file="报告.md", anchor="κ 在 2983 行上计算。", replacement="κ 在 2982 行上计算。",
        reason="门记录的 n 是 2982", artifact_key="gates.p2b_kappa.observed",
        check="gates.p2b_kappa.observed.n == 2982")
    ok, why = validate_edit(from_gate, texts, arts)
    assert ok, f"a gate citation was refused: {why}"

    from_finding = ProposedEdit(
        file="报告.md", anchor="另有 12 条处方。", replacement="另有 17 条处方。",
        reason="findings 记录 17", artifact_key="findings.abc123")
    ok2, why2 = validate_edit(from_finding, texts, arts)
    assert ok2, f"a finding citation was refused: {why2}"

    # The citation still has to SCOPE the numbers — a gate citation cannot
    # licence a number that lives in a different gate.
    stray = ProposedEdit(
        file="报告.md", anchor="κ 在 2983 行上计算。", replacement="κ 在 39 行上计算。",
        reason="x", artifact_key="gates.p2b_kappa.observed")
    ok3, why3 = validate_edit(stray, texts, arts)
    assert not ok3 and "39" in why3


def test_the_driver_puts_gates_and_findings_into_the_citable_namespace(tmp_path, monkeypatch):
    """Exercised through `audit_deliverables`, not by hand-building the dict.

    Found by mutation: testing `validate_edit` with a namespace I assembled
    myself proved nothing about whether the DRIVER assembles it — and the driver
    is where the bug was. On live40 it handed the auditor the gate ledger, told
    it to cite its source, and then refused every gate citation as unsourced.
    """
    from types import SimpleNamespace

    import qmine.agents.roles as roles
    from qmine.agents.audit_delivery import audit_deliverables

    gen = tmp_path / "gen01"
    gen.mkdir(parents=True)
    (gen / "报告.md").write_text("κ 在 2983 行上计算。\n", encoding="utf-8")

    class FakeAuditor:
        def __init__(self, ctx): pass
        def run(self, **kw):
            return SimpleNamespace(
                edits=[SimpleNamespace(
                    file="报告.md", anchor="κ 在 2983 行上计算。",
                    replacement="κ 在 2982 行上计算。",
                    reason="门 p2b_kappa 记录的 n 是 2982",
                    artifact_key="gates.p2b_kappa.observed",
                    severity="warn", check="gates.p2b_kappa.observed.n == 2982")],
                unfixable=[], dismissed=[], summary="")

    monkeypatch.setattr(roles, "DeliveryAuditorAgent", FakeAuditor)

    deps = SimpleNamespace(
        cfg=SimpleNamespace(audit_delivery=True, smoke_mode=False, report_language="zh"),
        store=SimpleNamespace(root=tmp_path, gen_dir=gen),
        run_id="t", emit=lambda m: None, agent_ctx=lambda: None)
    state = {"gates": {"p2b_kappa": {"status": "passed", "observed": {"n": 2982}}}}

    out = audit_deliverables(state, deps)

    assert out["n_applied"] == 1, (
        f"a gate-cited edit was refused: {[r['why'] for r in out.get('refused', [])]}")
    assert "2982" in (gen / "报告.md").read_text(encoding="utf-8")
