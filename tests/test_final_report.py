"""Guards on the one deliverable Python does not write.

The agent-written final report exists because template-generated documents cannot
carry a through-line: they are assembled section by section, so they read in the
order the code ran. Handing the prose to an agent buys coherence and costs the
guarantee that every sentence was constructed from an artifact. These tests pin
what replaces that guarantee.

The load-bearing one is `test_an_omitted_warning_cannot_be_silently_dropped`.
`check_numbers` is a PRECISION check — every number present must be sourced — and
it is completely silent about omission. A report that states only clean results
and never mentions the gate that passed with slack, or reports the chosen K
without the other values that stood up equally well, passes it perfectly while
misleading the reader entirely. Precision-only grounding is what permits
selective reporting, so coverage is checked separately and on the whole document.
"""
from __future__ import annotations

import pytest

from qmine.agents.narrate import (
    _assign_musts,
    _coverage,
    _disclose_missing,
    _reject,
)
from qmine.report.narrative_brief import Bundle, MustCover, sheet


def _bundle(**facts) -> Bundle:
    return Bundle("b", "测试", "测试用", facts=facts,
                  figures=[("fig1_ksweep.png", "K 扫描")])


# --------------------------------------------------------------- precision


def test_a_fabricated_number_cannot_reach_the_final_report():
    """The whole reason an agent is allowed to write a deliverable at all."""
    b = _bundle(kappa=0.8221, n=600)
    ok = _reject("本次 kappa 为 0.8221, 样本量 600。" * 6, [b], {"fig1_ksweep.png"}, "zh")
    assert ok == [], f"a correctly sourced section was refused: {ok}"

    bad = _reject("本次 kappa 为 0.9134, 样本量 600。" * 6, [b], {"fig1_ksweep.png"}, "zh")
    assert any("不在事实表" in p for p in bad), bad
    assert "0.9134" in " ".join(bad), "the rejection must quote the offending value"


def test_a_number_the_agent_derived_itself_is_refused():
    """The common failure: both inputs are in the sheet, the quotient is not.

    A narrator given 25 leaves and 7 families will write "平均每个家族 3.6 个叶子"
    unless it is stopped, and 3.6 appears in no artifact. It is arithmetically
    right and unsourced, which is exactly the class of claim that has shipped
    wrong before when the two fields turned out to count different populations.
    """
    b = _bundle(n_leaves=25, n_families=7)
    bad = _reject("共 25 个叶子分布在 7 个家族中, 平均每个家族 3.6 个。" * 5,
                  [b], set(), "zh")
    assert any("3.6" in p for p in bad), f"a derived number was allowed through: {bad}"


def test_a_percentage_is_checked_at_the_precision_the_author_wrote():
    """Rounding is allowed; a rounding the artifact does not support is not.

    `_matches` takes its tolerance from the digits the author actually wrote, so
    0.9765 in the sheet licenses "97.65%" exactly, and also "98%" — a bare integer
    percentage is a legitimate rounding of a measured share, which the matcher
    documents as a deliberate choice. What it does not license is 97%, which is
    further from the artifact than one integer step of rounding permits.
    """
    b = _bundle(dominant_share=0.9765)
    assert _reject("汉语查询占 97.65%, 其余语种都在门槛以下。" * 6, [b], set(), "zh") == []
    assert _reject("汉语查询占 98%, 其余语种都在门槛以下。" * 6, [b], set(), "zh") == []
    assert _reject("汉语查询占 97%, 其余语种都在门槛以下。" * 6, [b], set(), "zh") != []
    assert _reject("汉语查询占 97.2%, 其余语种都在门槛以下。" * 6, [b], set(), "zh") != []


# --------------------------------------------------------------- coverage


def test_an_omitted_warning_cannot_be_silently_dropped():
    """Coverage is the check `check_numbers` cannot make.

    Every number in this document is sourced, so the precision check passes it.
    It also never mentions the gate that warned, which is the only thing that
    would stop a reader trusting the result.
    """
    musts = [MustCover("gate:p2b_kappa", "kappa 门带保留通过", ["p2b_kappa"]),
             MustCover("both_routes", "两条路线都要讲", ["自上而下", "自下而上"])]
    clean = "自下而上聚类得到 7 个家族, 自上而下体系有 25 个 L1 类目。"
    missing = _coverage(clean, musts)
    assert [m.id for m in missing] == ["gate:p2b_kappa"], (
        "a document that never mentions the warned gate was reported as complete")

    honest = clean + " 其中 `p2b_kappa` 是带保留通过的。"
    assert _coverage(honest, musts) == []


def test_boilerplate_cannot_satisfy_a_coverage_requirement():
    """Scope is the whole point of where this check is applied.

    The assembled document opens with a provenance banner naming the scripted
    reports — including `自下而上聚类最终报告.md`. Running coverage over the
    assembled text therefore marked "both routes must be explained" as covered
    on a run where no section explained either route: a fixed string the pipeline
    itself writes had satisfied a requirement about what the agent wrote. Only
    authored, accepted sections are searched.
    """
    m = [MustCover("both_routes", "两条路线都要讲", ["自上而下", "自下而上"])]
    boilerplate = ("更细的证据在 `自下而上聚类最终报告.md` 与 "
                   "`自上而下类目体系最终报告.md` 里。")
    assert _coverage(boilerplate, m) == [], "sanity: the anchors are present here"
    # ...which is exactly why the banner must never be part of what is searched.
    authored_but_silent = "本次运行得到 7 个家族, 结构清晰。"
    assert [x.id for x in _coverage(authored_but_silent, m)] == ["both_routes"]


def test_a_coverage_failure_is_disclosed_in_the_document_itself():
    """Logging it is not enough — the reader is the one who needs to know.

    A coverage failure that only reaches the run log is the same omission the
    check exists to catch, moved one step away from the person it misleads.
    """
    text = _disclose_missing([MustCover("gate:x", "x 门警告了", ["x"])])
    assert "gate:x" in text and "x 门警告了" in text
    assert text.strip(), "the disclosure block cannot be empty"


def test_every_must_cover_item_is_routed_to_some_section():
    """The planner is asked to place these; the pipeline does not rely on it.

    An item the planner forgets would simply be absent, and only the final
    whole-document check would notice — after every paid section call was made.
    """
    class _S:
        def __init__(self, i, ev):
            self.id, self.evidence, self.figures = i, ev, []

    class _O:
        sections = [_S("intro", ["corpus"]), _S("k", ["granularity"]),
                    _S("quality", ["gates", "findings"])]

    musts = [MustCover("gate:a", "", ["a"]), MustCover("finding:b", "", ["b"]),
             MustCover("k_tie_set", "", ["7"]), MustCover("both_routes", "", ["x"])]
    routed = _assign_musts(_O(), musts, {})
    assert sum(len(v) for v in routed.values()) == len(musts), (
        f"an item was dropped in routing: {routed}")
    assert "gate:a" in [m.id for m in routed["quality"]]
    assert "k_tie_set" in [m.id for m in routed["k"]]


# --------------------------------------------------------------- other doors


def test_a_figure_the_run_never_produced_is_refused():
    """A plausible filename is the easiest thing in the world to invent."""
    b = _bundle(k=7)
    bad = _reject("如下图所示, K 定在 7。\n\n![对比图](fig9_comparison.png)\n" + "说明。" * 40,
                  [b], {"fig1_ksweep.png"}, "zh")
    assert any("fig9_comparison.png" in p for p in bad), bad


def test_english_prose_is_refused_in_a_chinese_deliverable():
    """Two English paragraphs shipped in live40's Chinese reports.

    The detector that should have caught them asked for three consecutive
    >=6-letter lowercase words, which real English almost never contains — so it
    fired on neither, on the live report or the fixture. This uses the corrected
    one, at generation time rather than only in a test.
    """
    b = _bundle(k=7)
    bad = _reject(
        "The chosen granularity is seven families, which the alignment metric "
        "located and the stability veto did not reject at any point here.",
        [b], set(), "zh")
    assert any("英文" in p for p in bad), bad


def test_a_section_too_short_to_carry_an_argument_is_refused():
    assert any("太短" in p for p in _reject("K 是 7。", [_bundle(k=7)], set(), "zh"))


# --------------------------------------------------------------- fact sheet


def test_the_narrator_sheet_carries_names_not_only_numbers():
    """`verify.fact_sheet` keeps only what `_flatten` keeps, which is numbers.

    A narrator handed that sheet can state that stability was 0.8692 and cannot
    name the encoder that produced it — so it would either omit the name or
    invent one. Strings are safe to add because the numeric guarantee is enforced
    separately, over numbers only.
    """
    from qmine.agents.verify import fact_sheet

    facts = {"chosen_encoder": "BAAI/bge-base-zh-v1.5", "stability_ari": 0.8692}
    assert "BAAI/bge-base-zh-v1.5" not in fact_sheet(facts)
    assert "BAAI/bge-base-zh-v1.5" in sheet(facts)
    assert "0.8692" in sheet(facts)


@pytest.mark.parametrize("bad_id", ["nonexistent", "corpus_v2", ""])
def test_evidence_ids_outside_the_catalogue_are_not_resolvable(bad_id):
    """The catalogue is a closed menu: the narrator cannot ask for evidence the
    run never produced, so it cannot imagine what that evidence would say."""
    catalogue = {"corpus": _bundle(n_rows=50)}
    assert bad_id not in catalogue


# --------------------------------------------------- evidence sufficiency

def test_worked_examples_carry_both_routes_labels_and_include_the_hard_ones(tmp_path):
    """The illustration set is chosen by code, and it is not allowed to flatter.

    Two defects this pins. First, the exemplars were read from
    `deployment.deterministic_exemplars`, which are PHRASING-PATTERN samples
    carrying no labels at all — the bundle built out to zero facts and a section
    asked to walk through labelled examples would have had none. The delivered
    table is the only place a query appears with both routes beside it.

    Second, showing one clean row per family would misrepresent the system. The
    rule therefore also takes the rows with the smallest routing margin — the
    cases it was least sure about — so a reader sees where it is weak by
    construction rather than by the author's generosity.
    """
    import pandas as pd

    from qmine.report.narrative_brief import _exemplars

    gen = tmp_path
    pd.DataFrame({
        "query": [f"q{i}" for i in range(12)],
        "bu_family_final": [0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 1, 2],
        "bu_leaf_name": ["叶A"] * 12,
        "td_l1_name": ["类甲"] * 12,
        "bu_margin": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0],
    }).to_csv(gen / "labels_full.csv", index=False)

    class _D:
        store = type("S", (), {"gen_dir": str(gen)})()

    ex = _exemplars({}, _D(), {"deployment": None})
    assert ex, "no worked examples were produced from the delivered table"
    for e in ex:
        assert "query" in e
    assert any("bu_leaf_name" in e and "td_l1_name" in e for e in ex), (
        "examples must carry BOTH routes' labels — that is what makes them worked")
    # Every family represented...
    fams = {e.get("bu_family_final") for e in ex}
    assert fams >= {0, 1, 2}, f"a family has no example: {fams}"
    # ...and the least-confident rows are in there too.
    assert any(e.get("bu_margin") == 0.0 for e in ex), (
        "only confident rows were shown; the hard cases were dropped")


def test_a_bundle_stays_small_enough_to_be_a_fact_sheet():
    """Scoped is half the rule; this is the half that has a number.

    Read whole, `metrics_panel.json` puts 783 facts into one section — every
    metric wrapped in its panel id, seed, authority, detail and note. That is the
    artifact under a different name, and it buries the handful of values the
    two-route comparison actually turns on.
    """
    from qmine.report.narrative_brief import _panel_grid

    fat = {"sets": {f"subject_{i}": {"metrics": {
        f"m{j}": {"value": float(j), "panel_id": "x" * 12, "seed": 0,
                  "authority": "diagnostic", "detail": {"a": 1}, "note": "n" * 50}
        for j in range(11)}} for i in range(11)}}
    grid = _panel_grid(fat)
    assert set(grid) == {f"subject_{i}" for i in range(11)}
    assert grid["subject_0"]["m3"] == 3.0
    assert len(sheet(grid).splitlines()) == 121, (
        "the grid must be exactly subject x metric — no wrappers")


def test_the_taxonomy_bundle_can_actually_name_a_class():
    """Sufficiency, on the side that induces fabrication.

    The taxonomy sits under `taxonomy.taxonomy`; reading the outer level yielded
    two facts and no class names. A section asked to explain the scheme without
    being able to name one class does not omit the name — it invents one. That is
    the documented failure mode of under-grounded data-to-text generation, and it
    is caused here by the evidence layer, not the writer.
    """
    from qmine.report.narrative_brief import build_catalogue

    class _D:
        store = type("S", (), {"gen_dir": "runs/live40/gen01"})()

    import os
    if not os.path.isdir("runs/live40/gen01"):
        pytest.skip("live40 artifacts not present")
    cat = build_catalogue({"gates": {}, "findings": [], "decisions": []}, _D())
    tax = cat.get("topdown_taxonomy")
    assert tax is not None, "the taxonomy bundle vanished"
    text = sheet(tax.facts)
    assert "classes[0]" in text, "no class is reachable in the sheet"
    assert len(text.splitlines()) > 20, (
        f"the taxonomy sheet is too thin to explain the scheme: {len(text.splitlines())} facts")


def test_a_number_the_agent_was_shown_is_citable():
    """The contract is "use only numbers from this sheet" — so the pool must be
    what the sheet SHOWS.

    `verify._flatten` pools only VALUES, while `sheet()` renders dotted paths. A
    dict keyed by id therefore puts numbers in front of the author that the
    checker refuses:

        execution.splits.32.new_leaf = 49      <- the agent reads "32"

    live42's `governance_and_risk` was rejected three times for citing
    `32, 40, 42, 43, 44, 45` — the leaf ids it had just been shown — and shipped
    as a marked hole. The section was doing exactly what it was told.

    Widening the pool to the sheet's own text must NOT weaken the guarantee: a
    number absent from the rendered sheet is still refused.
    """
    from qmine.agents.verify import check_numbers
    from qmine.report.narrative_brief import citable_numbers, sheet

    facts = {"gov": {"execution": {"splits": {
        "32": {"new_leaf": 49, "sizes": [834, 1368]},
        "40": {"new_leaf": 57},
    }}}}
    rendered = sheet(facts)
    assert "32" in rendered and "40" in rendered, "sanity: the ids are shown"

    # Exercise the path narrate actually takes, not `check_numbers` directly —
    # a test that builds the pool itself cannot notice `_reject` dropping it.
    b = Bundle("gov", "治理", "治理证据", facts=facts["gov"])
    prose = "治理把叶 32 拆成 49, 叶 40 拆成 57, 这是本次交付的实际改动。" * 8
    assert _reject(prose, [b], set(), "zh") == [], (
        "refused ids the sheet displays")

    # The guarantee survives: an id that appears nowhere is still refused.
    invented = "治理还拆了叶 77, 这在任何产物里都查不到, 属于凭空写出的数字。" * 8
    assert any("77" in p for p in _reject(invented, [b], set(), "zh")), (
        "an invented id was accepted — the pool is now too wide")

    # And the original defect reproduces without the sheet pool.
    assert not check_numbers("治理把叶 32 拆成 49。", facts).ok, (
        "the original defect no longer reproduces; this test is stale")


def test_an_empty_section_retry_carries_feedback_instead_of_repeating_the_call():
    """Three identical calls are one call's outcome, paid for three times.

    The empty-return branch appended a rejection and `continue`d WITHOUT setting
    `rejected`, so the retry re-sent a byte-identical prompt to a deterministic-
    enough model and got a byte-identical answer. live42's three empty sections
    (`vector_choice_first`, `two_level_tree`, `samples_and_deployment`) each
    burned all three attempts that way.

    `interpret.py` states the rule this broke: a retry must carry EXTERNAL
    feedback, because re-asking unchanged is intrinsic self-correction — the
    configuration shown not to improve reasoning and sometimes to degrade it.

    Diagnosis note for anyone revisiting this: truncation was the recorded
    suspicion and it is REFUTED. Reproduced against the real model, a 58,246-char
    sheet with 18,246 chars dropped still returned 1,784 chars of prose, and a
    33,462-char sheet returned 1,933. Do not "fix" the budget on this evidence.
    """
    from types import SimpleNamespace

    from qmine.agents import narrate

    seen_rejected: list[str] = []

    class _Agent:
        role = "reporter"

        def __init__(self, *a, **k):
            pass

        def run(self, **kw):
            seen_rejected.append(kw.get("rejected", ""))
            return SimpleNamespace(markdown="", covered=[])

    import qmine.agents.roles as roles

    original = roles.StoryWriterAgent
    roles.StoryWriterAgent = _Agent
    try:
        deps = SimpleNamespace(emit=lambda *_a, **_k: None,
                               agent_ctx=lambda: None)
        b = narrate.Bundle("b", "t", "r", facts={"k": 1})
        res = narrate._write_section(deps, SimpleNamespace(id="s", heading="h", intent="i"),
                                     [b], [], "outline", "", "zh")
    finally:
        roles.StoryWriterAgent = original

    assert not res.ok and res.attempts == narrate.MAX_ATTEMPTS
    # The FIRST call has nothing to report; every later one must.
    assert seen_rejected[0] == ""
    assert all(r.strip() for r in seen_rejected[1:]), (
        f"a retry re-sent an identical prompt: {seen_rejected}")
    assert "markdown" in seen_rejected[1], (
        "the feedback does not name the field that came back empty")


def test_an_attribution_is_supplied_rather_than_trusted():
    """`check_numbers` guarantees NUMBERS. A wrong noun passes it untouched.

    live42's §4 wrote 「交付的 K=18 是参照 phrasing_groups 的粒度锚点」 while
    `legacy_l2` located K and `phrasing_groups` located 10 — two lines after the
    same paragraph listed all three correctly. Every figure in that sentence was
    sourced from the fact sheet; only the subject was wrong.

    Note WHY anchors could not catch it: the anchor for that item is the deciding
    reference's name, and `legacy_l2` does appear in the wrong text — so
    anchor-matching passes. Nothing about "which reference is the subject of this
    claim" is expressible as a substring.

    A general "is this noun right?" check is not achievable, and pattern-matching
    Chinese prose for misattributions fails silently. So for the few claims where
    attribution is load-bearing, the pipeline WRITES the sentence and the section
    must reproduce it verbatim. The model still writes everything around it.
    """
    from qmine.agents.narrate import _coverage
    from qmine.report.narrative_brief import MustCover

    supplied = ("交付的家族层 K=18 由参照系 `legacy_l2` 定位; "
                "各参照系各自定位到的 K 为 legacy_l1→K=30, legacy_l2→K=18, phrasing_groups→K=10。")
    m = MustCover("k_deciding_reference", "写明定位参照系", ["legacy_l2"], verbatim=supplied)

    # The live42 text: anchor present, attribution wrong.
    wrong = "交付的 K=18 是参照 phrasing_groups 的粒度锚点, 不是语料常数。legacy_l2 定位到 18。"
    assert [x.id for x in _coverage(wrong, [m])] == ["k_deciding_reference"], (
        "the misattribution slipped through — the anchor alone cannot see it")

    # And the anchor-only check demonstrably does NOT catch it, which is why the
    # verbatim requirement exists at all.
    anchor_only = MustCover("k", "", ["legacy_l2"])
    assert _coverage(wrong, [anchor_only]) == [], (
        "this test is stale: anchors now catch it and `verbatim` is redundant")

    # Reproducing the supplied sentence satisfies the requirement.
    right = f"经过上面的比较: {supplied} 因此下游必须连同参照系一起读这个 K。"
    assert _coverage(right, [m]) == []

    # A paraphrase is NOT enough — that is the guarantee.
    para = supplied.replace("由参照系", "依据参照系")
    assert _coverage(para, [m]), "a rewritten sentence was accepted as verbatim"


def test_a_negative_number_from_the_sheet_can_actually_be_cited():
    """The defect that emptied most of live42's final report.

    `_NUMBER` had no sign, so `-5.23` was extracted as `+5.23` and reported as
    "not in the fact sheet" against a sheet whose value was exactly `-5.23`. The
    author had copied it correctly, so no retry could ever satisfy the check: the
    section burned all three attempts and shipped as a hole.

    It selects for the sections that matter. Negative numbers are where the
    WARNINGS live — a lopsided annotator split (`z_vs_even = -5.23`), a family
    layer that is less compact than its parts (`template_fragmentation
    = -0.0169`, silhouette `-0.0162`). live42 delivered 3 of 9 sections, and
    governance, audit-and-limits and the panel were among the six lost.

    The sign must survive, and it must still DISCRIMINATE: quoting `+5.23` for a
    fact of `-5.23` is a different claim and must fail.
    """
    b = _bundle(z_vs_even=-5.23, template_fragmentation=-0.0169)
    body = "标注分歧的 `z_vs_even` 为 -5.23, 家族层 `template_fragmentation` 为 -0.0169。" * 4
    assert _reject(body, [b], {"fig1_ksweep.png"}, "zh") == [], (
        "a negative fact quoted exactly as the sheet carries it must ship")

    flipped = _reject("`z_vs_even` 为 5.23。" * 8, [b], {"fig1_ksweep.png"}, "zh")
    assert any("不在事实表" in p for p in flipped), (
        "dropping the sign changes the claim; the check must still catch it")

    # U+2212, which a typographically careful writer may emit.
    assert _reject("`z_vs_even` 为 −5.23。" * 8, [b], {"fig1_ksweep.png"}, "zh") == []


def test_naming_the_model_that_did_the_work_is_not_a_fabrication():
    """`glm-5.2` yielded a phantom claim of `5.2`, and `glm-4.5-airx` one of
    `4.5`, because a hyphen before a digit read as a minus sign that was then
    discarded. live42's `audit_and_limits` was rejected for `5.23, 5.2` in one
    message: the first was the negative-number defect, the second was the
    referee's own model name in the same sentence.

    Provenance is something a report SHOULD state, so stating it must not be
    a rejection. The version digits are not claims about the corpus.
    """
    b = _bundle(n_contested=274)
    body = "争议样本 274 条, 裁判模型是 qwen:glm-5.2, 命名用 `glm-4.5-airx`。" * 4
    assert _reject(body, [b], {"fig1_ksweep.png"}, "zh") == []


def test_a_number_the_writer_was_ORDERED_to_include_is_citable():
    """A must-cover arrives under "必须原样包含这句话" — reproduce this sentence
    exactly. If it carries a number outside the section's bundles, the prompt
    demands what the checker forbids, and the section cannot be written at all.

    That is the same shape as the negative-number defect: an instruction the
    author can satisfy only by failing. The pool must therefore be everything
    the writer was SHOWN, not only the fact sheet.
    """
    b = _bundle(kappa=0.8928)
    must = "K 的并列集合是 2.3436、2.6297、3.044、3.6856, 必须整组报出。"
    body = ("本次 kappa 为 0.8928。" + must) * 3
    assert any("不在事实表" in p for p in
               _reject(body, [b], {"fig1_ksweep.png"}, "zh")), (
        "sanity: without being shown the must-cover, these numbers are refused")
    assert _reject(body, [b], {"fig1_ksweep.png"}, "zh", shown=must) == [], (
        "a number the writer was ordered to reproduce must be citable")


def test_the_rejection_notice_cannot_launder_its_own_numbers():
    """The notice PRINTS the offending values. Pooling it would make every
    rejected number citable on the retry, so the check would pass on its second
    attempt no matter what was written — a guarantee that always succeeds is not
    a guarantee. `_write_section` must never pass `rejected` as `shown`.
    """
    import inspect

    from qmine.agents.narrate import _write_section

    src = inspect.getsource(_write_section)
    call = src[src.index("problems = _reject("):]
    call = call[:call.index(")\n")]
    assert "shown=" in call, "the shown-channels pool must reach _reject"
    assert "rejected" not in call, (
        "the rejection notice quotes the bad numbers; pooling it defeats the check")


def test_the_failure_notice_says_which_of_the_two_things_went_wrong(tmp_path):
    """A hole in the flagship deliverable has to explain itself accurately.

    One notice covered both outcomes and read 「未通过校验」 — did not pass
    validation. For a blank return nothing was ever validated: the writer produced
    no text, which is a different fact about the run and points at a different
    remedy. live42 shipped three sections whose entire explanation was 「空白正文」
    listed three times, which tells a reader neither what happened nor that it was
    one failure repeated rather than three findings.
    """
    import types

    from qmine.agents.narrate import SectionResult, _assemble

    blank = SectionResult(id="s1", heading="空的一节")
    blank.attempts, blank.rejections = 3, ["空白正文 (markdown 字段为空或缺失)"] * 3
    checked = SectionResult(id="s2", heading="被退回的一节")
    checked.attempts, checked.rejections = 3, ["这些数字不在事实表里: 0.9134"] * 2

    outline = types.SimpleNamespace(title="T", thesis="", sections=[])
    deps = types.SimpleNamespace(store=types.SimpleNamespace(gen_dir=tmp_path))
    doc = _assemble(outline, [blank, checked], {"run_id": "r"}, deps, tmp_path)

    body = doc[doc.index("## 1."):doc.index("## 2.")]
    assert "没有返回任何文字" in body, "a blank return must not be called a check failure"
    assert "未通过校验" not in body
    assert body.count("空白正文") <= 1, "one failure repeated is one fact, not three"

    tail = doc[doc.index("## 2."):]
    assert "未通过校验" in tail, "a real check failure must still say so"
    assert tail.count("这些数字不在事实表里: 0.9134") == 1, "de-duplicated, but still shown"
