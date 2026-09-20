"""`mode="fast"` removes checking, never analysis — and cannot hide that it did.

Every test here was written against a specific way fast mode could become
dangerous: a run whose documents read exactly like a full run's while nothing
verified the numbers in them. The two properties that prevent it are that the
analysis is untouched (so the numbers ARE a full run's numbers) and that the
disclosure is generated from the same list that does the skipping (so it cannot
drift out of sync with what actually happened).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qmine.config import QMineConfig
from qmine.report.fast_deliver import _SKIP_MEANING, _banner, demote


def _code_only(src: str) -> str:
    """Source with comment lines removed — see test_measurement_soundness.

    A comment explaining a fix necessarily names the identifiers the fix is
    about, so a static assertion over raw source matches its own explanation and
    passes whatever the code does.
    """
    out = []
    for line in src.splitlines():
        if line.lstrip().startswith("#"):
            continue
        out.append(line.split("  # ")[0] if "  # " in line else line)
    return "\n".join(out)


def _fast(**kw) -> QMineConfig:
    return QMineConfig(mode="fast", offline=True, **kw)


# ==========================================================================
# The analysis
# ==========================================================================

def test_fast_mode_does_not_shrink_the_analysis():
    """Fast mode must produce the SAME numbers as full, only unchecked.

    `smoke_mode` and `mode="fast"` were one flag called `fast_mode`, whose
    documented job was shrinking the grids for a wiring test. A user asking for
    "results, quickly" and getting `alpha_grid=[0.0, 0.1, 0.5]` with a 120-row
    gold set would get a DIFFERENT answer wearing a production label — the worst
    of both. They are separate settings now, and this is the line between them.
    """
    full, fast = QMineConfig(offline=True), _fast()
    for field, obj in (("alpha_grid", "representation"), ("k_sweep", "clustering"),
                       ("battery_k", "clustering"), ("refine_rounds", "clustering")):
        assert getattr(getattr(fast, obj), field) == getattr(getattr(full, obj), field), field
    assert fast.taxonomy.gold_sample_size == full.taxonomy.gold_sample_size
    assert fast.taxonomy.n_researchers == full.taxonomy.n_researchers
    # And the grid WIDENER stays on: dropping it would narrow the search, which
    # is a change to the analysis, not the removal of a check.
    assert fast.propose_grids is True


def test_smoke_mode_and_fast_mode_are_independent():
    """The rename must not have left them coupled through a shared field."""
    smoke = QMineConfig(smoke_mode=True, offline=True)
    assert smoke.representation.alpha_grid == [0.0, 0.1, 0.5]      # shrunken
    assert smoke.mode == "full" and smoke.fast_skipped == []       # still checked
    assert smoke.observe_phases is True
    fast = _fast()
    assert fast.representation.alpha_grid != [0.0, 0.1, 0.5]       # full grid
    assert fast.observe_phases is False                            # not checked


# ==========================================================================
# The disclosure
# ==========================================================================

def test_every_skipped_component_reaches_the_reader():
    """The banner is rendered from `fast_skipped`, so it cannot omit a skip.

    This is the property the whole mode rests on. A hand-written banner would
    describe whatever was true when someone last edited it; this one is a
    function of the list the config validator populates as it turns each
    component off, so the two cannot disagree.
    """
    cfg = _fast()
    assert cfg.fast_skipped, "fast mode turned nothing off — the validator did not run"

    class _Store:
        gen_dir = "/tmp/gen01"

    class _Deps:
        pass

    deps = _Deps()
    deps.cfg, deps.store = cfg, _Store()
    text = "\n".join(_banner({"run_id": "t"}, deps))
    for key in cfg.fast_skipped:
        name = _SKIP_MEANING.get(key, (f"`{key}`", ""))[0]
        assert name in text, f"{key} was skipped and the banner never says so"
    assert "fast" in text.lower()


def test_an_unknown_skip_key_is_still_disclosed():
    """A component skipped without a `_SKIP_MEANING` row must not vanish.

    The tempting implementation is `for k in skips: out.append(_SKIP_MEANING[k])`,
    which raises, and the tempting fix for THAT is `.get(k)` with a `continue` —
    which silently drops the one thing the reader most needs to be told. A future
    skip added to the validator and not to the table degrades to the raw key.
    """
    cfg = _fast()
    cfg.fast_skipped = cfg.fast_skipped + ["some_future_check"]

    class _Store:
        gen_dir = "/tmp/gen01"

    class _Deps:
        pass

    deps = _Deps()
    deps.cfg, deps.store = cfg, _Store()
    assert "some_future_check" in "\n".join(_banner({"run_id": "t"}, deps))


def test_full_mode_never_renders_a_skip_banner():
    assert QMineConfig(offline=True).fast_skipped == []


# ==========================================================================
# The single annotator
# ==========================================================================

def test_one_annotator_yields_an_absent_kappa_not_a_perfect_one():
    """`_annotate_both` must return `None`, never the same labels twice.

    Returning `(labels, labels)` is the change that makes every downstream call
    site work without edits — and it writes kappa 1.000 into
    `gold_agreement.json`, a perfect score for a measurement nobody took. The
    `None` is what forces each caller to say what it does with one reading.
    """
    from qmine.graph.nodes.topdown import _annotate_both, _kappa_str

    calls: list[str] = []

    class _Reg:
        is_offline = True

    class _Ctx:
        cfg = _fast()
        registry = _Reg()

    class _Deps:
        def emit(self, *a, **k):
            pass

    import qmine.graph.nodes.topdown as td

    orig = td._annotate
    td._annotate = lambda ctx, which, *a: (calls.append(which)
                                           or [{"label": "X", "rationale": ""}])
    try:
        a, b = _annotate_both(_Ctx(), ["q"], "", "", "", _Deps())
    finally:
        td._annotate = orig

    assert b is None, "the second annotator must be absent, not a copy of the first"
    assert calls == ["a"], f"exactly one annotator should run, ran {calls}"
    assert _kappa_str({"kappa": None}) == "kappa 未测量 (单标注员)"
    assert _kappa_str({"kappa": 0.9}) == "kappa 0.900"


def test_a_skipped_gate_is_not_a_passed_gate():
    """`deps.gate(skipped=True)` must produce `skipped`, and teach no lesson.

    A fast run's ledger has to be distinguishable from a full run's. If the three
    unmeasurable gates recorded `passed`, `run_summary.json` would show a clean
    sweep for checks that never executed — the exact artefact a reader comparing
    two runs would use to conclude they were equally verified.
    """
    from qmine.records import GateResult

    class _Cfg:
        class gates:
            blocking = ["g"]

    class _D:
        cfg = _Cfg()
        lessons: list = []

        def lesson(self, **kw):
            self.lessons.append(kw)

        def emit(self, *a, **k):
            pass

    from qmine.graph.deps import Deps

    d = _D()
    g = Deps.gate(d, "g", "p2b", passed=False, skipped=True, observed={}, threshold={})
    assert isinstance(g, GateResult) and g.status == "skipped"
    assert g.ok and not g.halts_run, "an unrun check must not halt the run"
    assert d.lessons == [], "an unrun check has taught nothing"


# ==========================================================================
# Composing the documents
# ==========================================================================

def test_demote_leaves_code_fences_alone():
    """`# comment` inside a fence is code, not a heading.

    The composite documents nest whole builder outputs under their own sections,
    which means shifting heading levels. A naive `^#` substitution corrupts every
    Python comment in every fenced example the reader is meant to copy and run.
    """
    src = "# T\n## S\n\n### real\n\n```python\n# a comment\n## another\n```\n\n#### deep\n"
    out = demote(src)
    assert "\n# a comment\n" in out and "\n## another\n" in out
    assert "#### real" in out and "##### deep" in out


def test_demote_never_exceeds_six_levels():
    """A seventh `#` renders as literal text, not a heading."""
    assert demote("###### six\n", drop_title=False).startswith("###### six")


@pytest.mark.parametrize("builder", ["build_topdown", "build_bottomup"])
def test_each_composite_document_opens_with_the_banner(builder, monkeypatch):
    """Both markdown deliverables must carry the disclosure, not just one.

    They are written by separate functions and a reader may be sent either one
    alone, so neither may rely on the other having said it.
    """
    import qmine.report.fast_deliver as fd

    cfg = _fast()

    class _Store:
        gen_dir = "/tmp/gen01"

    class _Deps:
        def emit(self, *a, **k):
            pass

    deps = _Deps()
    deps.cfg, deps.store = cfg, _Store()
    # Every source builder replaced: this asserts the COMPOSITION, and a real
    # builder would need a full pipeline state to say anything.
    monkeypatch.setattr(fd, "_section", lambda *a, **k: "### stub")
    monkeypatch.setattr(fd, "_archive", lambda *a, **k: ["## 原始档案位置"])
    text = getattr(fd, builder)({"run_id": "t"}, deps)
    assert "fast 模式" in text
    for key in cfg.fast_skipped:
        assert _SKIP_MEANING[key][0] in text, f"{builder} omitted {key}"
    assert "原始档案位置" in text, "a deliverable must say where its evidence lives"


# ==========================================================================
# Re-rendering
# ==========================================================================

def test_a_render_cannot_upgrade_a_fast_run_to_a_full_one(tmp_path):
    """`qmine render` must inherit `mode` from the generation it re-derives.

    Found by running it: `render` builds its config from the CLI, where `mode`
    defaults to "full", so re-rendering a fast run produced the THIRTEEN
    full-mode documents — 叶清单.md, 类目清单.md, 统一度量面板.md — with no
    skipped-components banner anywhere in them, describing numbers nothing in
    that run had checked. The one command whose purpose is to re-derive
    deliverables was the one that could strip the disclosure off them.
    """
    from qmine.runner import inherit_mode

    root = tmp_path / "r1"
    (root / "gen01").mkdir(parents=True)
    QMineConfig(mode="fast", offline=True).dump(root / "gen01" / "config.resolved.yaml")

    out = inherit_mode(QMineConfig(offline=True), root, 1)
    assert out.mode == "fast"
    assert out.fast_skipped, "the skip list must travel with the mode"
    assert out.observe_phases is False, "the mode's own switches must be re-applied"


def test_a_render_of_a_full_run_stays_full(tmp_path):
    """The inheritance must not leak the other way."""
    from qmine.runner import inherit_mode

    root = tmp_path / "r2"
    (root / "gen01").mkdir(parents=True)
    QMineConfig(offline=True).dump(root / "gen01" / "config.resolved.yaml")
    assert inherit_mode(QMineConfig(offline=True), root, 1).mode == "full"


def test_a_render_reports_the_skips_the_RUN_made_not_todays(tmp_path):
    """The recorded `fast_skipped` wins over whatever the current code skips.

    A banner rendered a year later must describe the run that produced the
    artifacts. If the validator's list has since gained or lost a component,
    re-validating would silently rewrite history in the deliverable.
    """
    from qmine.runner import inherit_mode

    root = tmp_path / "r3"
    (root / "gen01").mkdir(parents=True)
    cfg = QMineConfig(mode="fast", offline=True)
    cfg.fast_skipped = ["dual_annotation", "a_check_that_no_longer_exists"]
    cfg.dump(root / "gen01" / "config.resolved.yaml")

    out = inherit_mode(QMineConfig(offline=True), root, 1)
    assert out.fast_skipped == ["dual_annotation", "a_check_that_no_longer_exists"]


def test_the_skip_list_survives_revalidation(tmp_path):
    """A resumed fast run must disclose the SAME ten skips a fresh one does.

    Each append used to be conditional on the switch still being on, so validating
    an already-fast config — which is every `--resume`, because it loads the source
    generation's resolved config where `observe_phases` and friends are already
    False — rebuilt a four-entry list. `ppl-pool8` gen03 shipped a banner naming 4
    of 10: dual annotation, the phase observers, the adversarial validation, the
    agent-written report, the pre-delivery audit and the result interpretation all
    read as HAVING RUN. The banner is generated from this list, so a truncated list
    is a false statement in a delivered document.
    """
    fresh = QMineConfig(mode="fast", offline=True)
    again = QMineConfig.model_validate(fresh.model_dump())
    assert again.fast_skipped == fresh.fast_skipped, \
        "re-validating a fast config must not shrink what the banner discloses"
    for name in ("dual_annotation", "phase_observers", "adversarial_validation",
                 "narrative_report", "delivery_audit", "result_interpretation"):
        assert name in again.fast_skipped, f"{name} silently dropped on re-validation"
    # and the switches really are off, both times — the list must not outrun the config
    assert not (again.observe_phases or again.final_report or again.delivery_audit
                or again.interpret_results or again.validate_adversarial)
    assert again.taxonomy.annotators == 1

    # round-tripping through the on-disk config a resume reads must be stable too
    path = tmp_path / "config.resolved.yaml"
    fresh.dump(path)
    assert QMineConfig.load(path).fast_skipped == fresh.fast_skipped


def test_a_render_keeps_the_corpus_it_was_run_on(tmp_path):
    """The re-render must say the domain the RUN used, not `generic`.

    Pre-existing and cosmetic in full mode — every re-rendered document read
    "**领域**: `generic`" because `render` builds its config without `--domain`.
    Fast mode makes it material: the domain key is part of the deliverable
    filename, so a render wrote `generic_自上而下_….md` next to the run's
    `k12_zh_自上而下_….md` — the same document under two names, in one run
    directory.
    """
    from qmine.runner import inherit_mode

    root = tmp_path / "r4"
    (root / "gen01").mkdir(parents=True)
    cfg = QMineConfig(mode="fast", offline=True)
    cfg.domain = cfg.domain.model_copy(update={"key": "medical_zh"})
    cfg.dump(root / "gen01" / "config.resolved.yaml")

    out = inherit_mode(QMineConfig(offline=True), root, 1)
    assert out.domain.key == "medical_zh"
    assert out.mode == "fast", "the domain fix must not have displaced the mode"


# ==========================================================================
# p8's delivered-leaf collision check — two defects found by running `make demo`
# ==========================================================================

def test_the_collision_check_runs_when_governance_changed_nothing():
    """`cents` must be bound on both branches of the governance rewrite.

    It was assigned only inside `if new_labels is not None and not
    array_equal(...)` — the branch that runs when governance actually rewrote
    the partition. On every run where governance changed nothing, the collision
    check raised `UnboundLocalError`, its own `except` swallowed it, and
    `still_colliding` stayed `[]` — so the gate reported "every delivered leaf is
    distinguishable from its siblings by name" having compared nothing.

    Asserted on the SOURCE because reproducing it needs a full p8: the binding
    has to happen before the branch, not inside it.
    """
    import inspect

    from qmine.graph.nodes import naming

    # COMMENTS STRIPPED FIRST. The comment explaining this fix necessarily names
    # `still_colliding` and `cents`, so a slice over raw source finds its own
    # explanation instead of the code — a trap this repo has sprung more than
    # once (`_code_only` in test_measurement_soundness.py exists for it).
    src = _code_only(inspect.getsource(naming.p8_governance))
    body = src[:src.index("still_colliding")]
    bind = body.index("cents = _centroids")
    branch = body.index("if new_labels is not None and not np.array_equal")
    assert bind < branch, (
        "`cents` is bound only inside the governance-rewrote branch; a run where "
        "governance changes nothing will raise UnboundLocalError and the "
        "collision gate will pass on a check that never ran")


def test_the_collision_gate_reaches_the_operator():
    """`deps.gate` BUILDS a gate; a node must return it or it is dropped.

    Verified on a real run: the log printed
    "gate p8_leaves_are_distinguishable: PASSED" while `run_summary.json`'s gate
    list did not contain it. An unreturned gate is invisible to the router, can
    never halt anything, and cannot be read afterwards — the same mistake
    `topdown.py` documents having made once before and found five runs later.
    """
    import inspect

    from qmine.graph.nodes import naming

    src = _code_only(inspect.getsource(naming.p8_governance))
    i = src.index('"p8_leaves_are_distinguishable"')
    assigned = src.rindex("=", 0, src.rindex("deps.gate(", 0, i))
    assert src[assigned - 40:assigned].strip().split()[-1].isidentifier(), \
        "the gate call must be assigned"
    assert "distinguishable_gate.name: distinguishable_gate" in src, \
        "p8 builds p8_leaves_are_distinguishable but never returns it in `gates`"


# ==========================================================================
# Config composition
# ==========================================================================

def test_a_corpus_config_keeps_the_provider_policy_it_extends(tmp_path):
    """`--config` REPLACES the default; `extends:` is how a small file survives it.

    A corpus config saying only "this corpus\'s text column is `original_query`"
    silently discarded the whole of `live.yaml`: the role pins, the excluded
    labs, and the lab-independence requirement that double-blind annotation
    rests on — after which the router picks on price alone and can put both
    annotators on the same lab. `_load_config`\'s docstring calls that the one
    launch mistake nothing catches, and this nearly shipped as a finance run.
    """
    base = tmp_path / "base.yaml"
    base.write_text(
        "llm:\n  provider: router\n  excluded_labs: [openai]\n"
        "  model_overrides: {referee: some-model}\n"
        "taxonomy:\n  n_researchers: 7\n",
        encoding="utf-8")
    child = tmp_path / "corpus.yaml"
    child.write_text(
        "extends: base.yaml\ndata:\n  text_column: original_query\n"
        "taxonomy:\n  n_researchers: 3\n", encoding="utf-8")

    cfg = QMineConfig.load(child)
    assert cfg.data.text_column == "original_query"      # the child's own setting
    assert cfg.llm.excluded_labs == ["openai"]           # inherited
    assert cfg.llm.model_overrides == {"referee": "some-model"}
    assert cfg.taxonomy.n_researchers == 3, "the extending file must win a conflict"


def test_a_missing_extends_target_is_an_error_not_a_silent_default(tmp_path):
    """Silently ignoring it would reintroduce exactly the bug `extends` prevents."""
    child = tmp_path / "c.yaml"
    child.write_text("extends: nope.yaml\ndata:\n  text_column: q\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        QMineConfig.load(child)


def test_the_shipped_finance_config_inherits_the_live_policy():
    """The real file, not a fixture — it is what a paid run will load."""
    from pathlib import Path as _P

    cfgs = _P(__file__).resolve().parents[1] / "configs"
    if not (cfgs / "live_finance.yaml").exists():
        pytest.skip("finance config not present")
    cfg = QMineConfig.load(cfgs / "live_finance.yaml")
    assert cfg.data.text_column == "original_query"
    assert cfg.data.weight_column == "wise_pv"
    assert cfg.llm.provider == "router", "the finance run must still route"
    assert cfg.llm.excluded_labs, "the lab exclusions must survive the extend"


def test_an_unset_cli_flag_does_not_overrule_the_config(tmp_path):
    """Typer\'s DEFAULT is not the user\'s choice, and must not beat a config file.

    `cfg.data.text_column = text_column` ran unconditionally, so the option\'s
    default "query" overwrote a config that said `original_query` — and p1 then
    halted with `KeyError: \'query\'` before reading a single row. Found on an
    offline dry run of the finance corpus, one phase into what would otherwise
    have been a paid run. The same bug had already been found and fixed for
    `provider` eleven lines below, and left standing here.

    `reference_label_columns` had it too, and fails more quietly: the declared
    columns silently become none, and the corpus then looks like one whose
    label-like columns were never declared.
    """
    import inspect

    from qmine import cli

    src = _code_only(inspect.getsource(cli.run))
    assert "if text_column:" in src, \
        "an unset --text-column must leave the config's value alone"
    assert "if reference_columns.strip():" in src, \
        "an unset --reference-columns must leave the config's value alone"
    # And the option itself must be able to express "unset".
    sig = inspect.signature(cli.run).parameters["text_column"]
    assert sig.default.default is None, \
        "--text-column needs a None default, or 'unset' is unrepresentable"


# ==========================================================================
# "Nothing failed" is not "nothing was checked"
# ==========================================================================

def test_no_gate_passes_on_an_empty_evidence_set():
    """`passed=not <collection>` is true when the check found nothing wrong —
    AND when it had nothing to look at. Those are different outcomes.

    Found three times: the p8 collision gate passed after its check raised
    `UnboundLocalError`; `p2b_rules_match_their_evidence` reported "every
    boundary's stated discriminator actually divides its adjudicated rows (0
    tested)" on a fast run, where one annotator means nothing is contested and
    so nothing is adjudicated; and both naming gates would report "all 0 leaves
    named" on an empty partition.

    This pins the shape rather than the three instances, so a fourth gate written
    the same way fails here instead of shipping a green light.
    """
    import inspect
    import re

    from qmine.graph.nodes import delivery, naming, topdown

    # gate name -> the expression that must guard it. Each is either a `skipped=`
    # (the measurement did not happen) or a non-emptiness conjunct in `passed=`
    # (there was nothing to measure over).
    REQUIRED = {
        "p2b_rules_match_their_evidence": "skipped=not ev_report.stated_grounds",
        "p8_leaves_are_distinguishable": "skipped=not collision_check_ran",
        "p10_delivered_leaves_named": "passed=bool(delivered) and not unnamed",
        "p7_all_leaves_named": "passed=n_leaves > 0 and not unnamed",
    }
    sources = {m.__name__: _code_only(inspect.getsource(m))
               for m in (topdown, naming, delivery)}
    joined = "\n".join(sources.values())
    for gate, guard in REQUIRED.items():
        assert f'"{gate}"' in joined, f"{gate} no longer exists — update this test"
        assert guard in joined, (
            f"{gate} must be guarded by `{guard}` — without it the gate reports a "
            "pass when its check found nothing because it looked at nothing")

    # And no NEW gate may use the bare form. A gate added later with
    # `passed=not something` and no guard lands here.
    #
    # The exemption is checked, not assumed: a `skipped=` anywhere in the same
    # `deps.gate(` call counts as the guard, which is why `p2b_kappa` — whose
    # `unsound` is a BOOLEAN (coverage below threshold), not a collection, and
    # whose fast branch already carries `skipped=True` — is not flagged.
    ALLOWED = {
        # No undeclared label-like columns IS the success condition here; there
        # is no "measurement did not run" case to confuse it with.
        "p1_reference_columns_declared",
    }
    unguarded = []
    for m in re.finditer(r'"(p[0-9]+[a-z]?_[a-z_]+)",\s*"[^"]+",\s*\n\s*passed=not \w+,',
                         joined):
        name = m.group(1)
        if name in ALLOWED:
            continue
        call = joined[m.start():m.start() + 1200]
        end = call.find("\n    )")
        if "skipped=" not in (call[:end] if end > 0 else call):
            unguarded.append(name)
    assert not unguarded, (
        f"gate(s) {unguarded} pass on an empty collection with no guard. Either the "
        "empty case genuinely IS the passing condition (like "
        "`p1_reference_columns_declared`, where no undeclared columns is the "
        "success), or the gate needs `skipped=` / a non-emptiness conjunct.")


def test_a_solo_annotator_cannot_invent_a_class():
    """With one reading, nothing contradicts an invented label — so p2b must check.

    Two annotators are self-correcting here almost by accident: an invented code
    rarely matches the other reading, so the row becomes a disagreement and the
    referee's verdict passes through `_snap_label_to_taxonomy`. Solo mode has
    `la == lb` by construction, so `final = la` unconditionally — and on `fin01`
    the query `2246` came back as `UNKNOWN`, a class not in the taxonomy, and
    shipped into the gold set. A one-row phantom class is then dropped from
    cross-validation as "too rare", which reports a malformed label to the reader
    as rarity.
    """
    import inspect

    from qmine.graph.nodes import topdown

    src = _code_only(inspect.getsource(topdown.p2b_gold))
    assert "_snap_label_to_taxonomy(la, _valid_codes)" in src, \
        "the solo path must snap an off-schema label the way the referee path does"
    assert "n_offschema" in src, \
        "rows dropped for an unrepairable label must be counted and reported"
    # And the repair itself must still refuse an invented class rather than guess.
    code, note = topdown._snap_label_to_taxonomy("UNKNOWN", {"LOOKUP_A", "LOOKUP_B"})
    assert not code, f"an invented class must be refused, got {code!r} ({note})"
    fixed, _ = topdown._snap_label_to_taxonomy("LOOKUP_A ", {"LOOKUP_A", "LOOKUP_B"})
    assert fixed == "LOOKUP_A", "a recoverable label must still be repaired"


def test_a_rendered_deliverable_can_still_find_its_evidence():
    """Fast deliverables must resolve artifacts through the STORE, not a path.

    `qmine render` writes into a NEW generation while re-deriving from the old
    one's artifacts. The store resolves that (`ref.generation <= generation`); a
    raw `store.gen_dir / name` does not. Re-rendering `fin01` produced a workbook
    with 8 of its 10 sheets empty and 0 rows in 全量标注, and an 原始档案位置
    table reporting "未生成" for 13 artifacts that were one directory up — a
    deliverable telling the reader all of its evidence was missing, which is
    exactly the promise fast mode makes and must keep.
    """
    import inspect

    from qmine.report import fast_deliver

    src = _code_only(inspect.getsource(fast_deliver))
    assert "def _artifact_path(" in src
    # No raw gen_dir joins for artifact lookup outside the helper itself.
    body = src[src.index("def _archive("):]
    for bad in ('gen / f"{name}{suffix}"', 'gen / "labels_full.csv"',
                'gen / f"{name}.json"'):
        assert bad not in body, (
            f"{bad!r} reads the TARGET generation; a re-render finds nothing there")
    assert "_artifact_path(deps, \"labels_full\")" in src


def test_corner_brackets_are_quotes_too():
    """`『』` must extract like `「」` — the codebase half-knew this.

    `_QUOTED`'s character class listed 「」 and not 『』, while `usable_markers`
    twenty lines below already strips both. A model that writes its
    discriminators as 『走势图/k线』 had every span dropped before extraction, and
    its rules were recorded as carrying no executable trigger. Measured on
    `fin02`: four rules gain usable markers.

    This is NOT the explanation for that run's 52 rejected triggers — those
    described a category (`裸数字代码`) instead of naming a string, which is the
    check doing its job. Keeping the two apart is the point of this test.
    """
    from qmine.ops.rule_conflict import _QUOTED, usable_markers

    assert _QUOTED.findall("查询含『走势图/k线』等词") == ["走势图/k线"]
    assert _QUOTED.findall("查询含「走势图」等词") == ["走势图"]
    ok, _rejected = usable_markers(_QUOTED.findall("含『主连/合约/期货』"))
    assert set(ok) >= {"主连", "合约", "期货"}


def test_the_namer_is_pinned_and_the_pin_is_reachable():
    """The pin must survive `extends:` into a corpus config."""
    from pathlib import Path as _P

    cfgs = _P(__file__).resolve().parents[1] / "configs"
    live = QMineConfig.load(cfgs / "live.yaml")
    assert live.llm.model_overrides.get("namer") == "deepseek-v4-pro"
    if (cfgs / "live_finance.yaml").exists():
        fin = QMineConfig.load(cfgs / "live_finance.yaml")
        assert fin.llm.model_overrides.get("namer") == "deepseek-v4-pro", \
            "the namer pin must survive the extends chain"


# ==========================================================================
# Column binding — the first thing a new corpus gets wrong
# ==========================================================================

def _tiny_export(tmp_path):
    """A file shaped like the house export: constant slice column, pv weights."""
    import pandas as pd

    p = tmp_path / "t.csv"
    pd.DataFrame({"event_day": [1] * 5,
                  "query_1st_category": ["金融"] * 5,
                  "original_query": ["今日金价", "上证指数", "黄金价格", "金价", "国际金价"],
                  "wise_pv": [10, 20, 30, 40, 50]}).to_csv(p, index=False)
    return str(p)


def _p1(path, **over):
    from qmine.graph.nodes.foundation import p1_audit

    cfg = QMineConfig(offline=True)
    cfg.data.input_path = path
    cfg.data.text_column = "original_query"
    cfg.data.weight_column = "wise_pv"
    cfg.data.reference_label_columns = []
    for k, v in over.items():
        setattr(cfg.data, k, v)

    class _D:
        def __init__(self, c):
            self.cfg = c
            self.emitted = []

        def emit(self, m):
            self.emitted.append(m)

    d = _D(cfg)
    try:
        p1_audit({}, d)
    except ValueError:
        raise
    except Exception:
        pass          # later phase logic needs a real Deps; column binding is done
    return d


@pytest.mark.parametrize("field,bad", [
    ("text_column", "originl_query"),
    ("weight_column", "wise_pvv"),
])
def test_a_misnamed_column_stops_the_run_and_names_the_real_ones(field, bad, tmp_path):
    """A column the config names and the file lacks is a config error, not a default.

    Both bound silently: `[c for c in ... if c in raw.columns]` turned a typo into
    "no reference columns", and `if weight_column in raw.columns else None` turned
    one into an UNWEIGHTED run — every metric then counts distinct queries instead
    of traffic, and `population_weighted_accuracy` quietly describes something
    else. Neither said anything, and a run is hours long, so the check has to fire
    before the first paid call and has to print what the file actually offers.
    """
    with pytest.raises(ValueError) as e:
        _p1(_tiny_export(tmp_path), **{field: bad})
    msg = str(e.value)
    assert bad in msg, "the error must name the column that was wrong"
    assert "original_query" in msg, "the error must list the columns the file HAS"


def test_a_reference_column_that_does_not_exist_is_refused(tmp_path):
    with pytest.raises(ValueError) as e:
        _p1(_tiny_export(tmp_path), reference_label_columns=["legacy_l1"])
    assert "legacy_l1" in str(e.value)


def test_a_constant_reference_column_is_dropped_not_used(tmp_path):
    """`query_1st_category` holds one value per file — a slice name, not a label.

    Declared as a reference it hands the K locator a one-class frame against which
    every candidate partition scores identically, which is worse than declaring
    nothing. Dropped with a warning rather than silently, because a user who
    declared it meant something by it.
    """
    d = _p1(_tiny_export(tmp_path), reference_label_columns=["query_1st_category"])
    assert any("query_1st_category" in m and "constant" in m for m in d.emitted), \
        f"expected a dropped-constant warning, got {d.emitted}"


def test_the_house_export_config_chain_resolves():
    """`live_finance` -> `corpus_wise_export` -> `live`: columns once, providers once."""
    from pathlib import Path as _P

    cfgs = _P(__file__).resolve().parents[1] / "configs"
    if not (cfgs / "corpus_wise_export.yaml").exists():
        pytest.skip("house-export config not present")
    c = QMineConfig.load(cfgs / "live_finance.yaml")
    assert c.data.text_column == "original_query"
    assert c.data.weight_column == "wise_pv"
    assert c.data.reference_label_columns == []
    assert c.llm.provider == "router", "the provider policy must survive two extends"
    assert c.llm.model_overrides.get("namer") == "deepseek-v4-pro"


def test_an_empty_query_cell_is_dropped_loudly_not_crashed_on(tmp_path):
    """One empty cell in 20,000 rows must not halt a run at phase 1.

    `raw[text].astype(str)` used to turn NaN into the string "nan"; under pandas
    3.0's string dtype it leaves NA as NA, so a float reached `char_profile` and
    died with `object of type 'float' has no len()`. `edu-pool` halted exactly
    that way on row 17,717 — AFTER the run was launched, which is the expensive
    place to discover it.

    Dropped rather than filled: an empty query is not a query, and "" would be
    embedded, clustered and counted as though someone searched for nothing.
    """
    import pandas as pd

    from qmine.graph.nodes.foundation import p1_audit

    p = tmp_path / "t.csv"
    # 985 as an INT is legitimate here — people search university tiers — so the
    # fix must drop the empty cell without touching the numeric queries.
    pd.DataFrame({"original_query": ["今日金价", None, "上证指数", 985] + ["查询"] * 40,
                  "wise_pv": [1, 2, 3, 4] + [1] * 40}).to_csv(p, index=False)
    cfg = QMineConfig(offline=True)
    cfg.data.input_path = str(p)
    cfg.data.text_column = "original_query"
    cfg.data.weight_column = "wise_pv"
    cfg.data.reference_label_columns = []

    class _D:
        def __init__(self):
            self.cfg = cfg
            self.emitted = []

        def emit(self, m):
            self.emitted.append(m)

    d = _D()
    try:
        p1_audit({}, d)
    except ValueError:
        raise                      # the >5% guard is a different, deliberate path
    except Exception:
        pass                       # later phase logic needs a real Deps
    assert any("empty" in m and "dropped" in m for m in d.emitted), \
        f"the drop must be reported, got {d.emitted}"


def test_a_mostly_empty_text_column_is_refused_outright(tmp_path):
    """5% empty is a broken export, and analysing what survives hides that."""
    import pandas as pd

    from qmine.graph.nodes.foundation import p1_audit

    p = tmp_path / "t.csv"
    pd.DataFrame({"original_query": ["查询"] * 10 + [None] * 10,
                  "wise_pv": [1] * 20}).to_csv(p, index=False)
    cfg = QMineConfig(offline=True)
    cfg.data.input_path = str(p)
    cfg.data.text_column = "original_query"
    cfg.data.reference_label_columns = []

    class _D:
        def __init__(self, c):
            self.cfg = c

        def emit(self, m):
            pass

    with pytest.raises(ValueError, match="export problem"):
        p1_audit({}, _D(cfg))


def test_one_failed_research_angle_does_not_kill_the_phase():
    """A content filter on one of five angles must not end a two-hour paid run.

    The fan-out already treated an angle returning NOTHING as a warning, but an
    angle that RAISED took p2a with it. `ppl-pool` and `film-pool` both died that
    way: `researcher_pragmatic_intents` hit a provider content filter (HTTP 400,
    `contentFilter`, 系统检测到输入或生成内容可能包含不安全或敏感内容) on all three
    attempts. Both corpora contain precisely what a Chinese provider filters —
    人物 mixes serving officials with entertainers, 影视 carries adult and banned
    titles — so this is a property of the corpus, not a fault to halt on.

    Halting is kept for the case that warrants it: NO angle survived, which means
    the role is misconfigured rather than the content awkward.
    """
    import inspect

    from qmine.graph.nodes import topdown

    src = _code_only(inspect.getsource(topdown.p2a_taxonomy))
    assert "_one_safe" in src, "the per-angle call must be wrapped"
    assert "failed_angles" in src, "failures must be recorded, not swallowed"
    assert "if not submissions:" in src, \
        "every angle failing IS fatal — a taxonomy cannot come from nothing"
    # The wrapper must be what the pool maps over, or it protects nothing.
    assert "pool.map(_one_safe, angles)" in src


# ==========================================================================
# Multi-snapshot drift (p10b)
# ==========================================================================

def _pooled_frame():
    """Two snapshots, one label column, deliberately uneven weights."""
    import pandas as pd

    rows = []
    for snap, n_a, n_b in (("20250701", 60, 40), ("20260701", 30, 70)):
        rows += [{"snapshot": snap, "query": f"q{i}", "weight": 10.0, "cls": "A"}
                 for i in range(n_a)]
        rows += [{"snapshot": snap, "query": f"r{i}", "weight": 1.0, "cls": "B"}
                 for i in range(n_b)]
    return pd.DataFrame(rows)


def test_drift_uses_within_snapshot_shares_not_raw_counts():
    """Raw counts report every class as declining when total traffic falls.

    One medical pair fell 9.74M to 5.21M total weight (-47%). A comparison on raw
    weight would show every single class shrinking and say nothing about
    composition, which is the only thing a drift report is for.
    """
    import pandas as pd

    from qmine.ops import drift

    # Both snapshots identical in COMPOSITION; only the second's total weight is
    # halved. `_pooled_frame` deliberately differs between periods, so it cannot
    # isolate the base-rate effect — this needs a frame where nothing but the
    # base rate moves.
    df = pd.DataFrame(
        [{"snapshot": snap, "query": f"q{i}", "weight": w, "cls": c}
         for snap, w in (("20250701", 10.0), ("20260701", 5.0))
         for c, n in (("A", 60), ("B", 40))
         for i in range(n)])
    d = drift.label_drift(df, "cls", "snapshot", "weight")
    assert d["stable"], "two classes present in both snapshots must be comparable"
    for r in d["stable"]:
        assert abs(r["weight_share_delta_pp"]) < 0.001, \
            f"a pure base-rate change must not read as drift: {r}"
        assert abs(r["row_share_delta_pp"]) < 0.001, r

    # ...and the inventory must still SHOW the traffic fall, so a reader can see
    # the base rate moved even though composition did not.
    inv = {r["snapshot"]: r["weight_total"] for r in
           drift.snapshot_inventory(df, "snapshot", "query", "weight")}
    assert inv["20260701"] < inv["20250701"]


def test_drift_separates_emergent_classes_from_shifts():
    """A class present in one snapshot only cannot have a share CHANGE.

    Putting it in the same ranked table as genuine shifts invites reading
    'appeared' as 'grew', and the two need different treatment: an emergent class
    may be new behaviour, or may be behaviour that was always there and too
    sparse to cluster until the pooled corpus gave it enough rows. Observed live:
    a netdisk-piracy class with 2 rows in one snapshot and 189 in the other.
    """
    import pandas as pd

    from qmine.ops import drift

    df = _pooled_frame()
    df = pd.concat([df, pd.DataFrame([{"snapshot": "20260701", "query": f"n{i}",
                                       "weight": 1.0, "cls": "NEW"} for i in range(50)])],
                   ignore_index=True)
    d = drift.label_drift(df, "cls", "snapshot", "weight")
    assert [r["label"] for r in d["emergent"]] == ["NEW"]
    assert "NEW" not in {r["label"] for r in d["stable"]}


def test_drift_purity_check_catches_a_frame_that_is_not_shared():
    """The whole comparison rests on both periods sharing one label frame.

    A group sitting ~entirely in one snapshot was SEPARATED, not compared. Four of
    five real corpora had zero such groups; the fifth (news-driven) had 6 of 25,
    all genuine period-specific events — so a nonzero count is a prompt to look,
    which is why the gate warns rather than blocks.
    """
    import pandas as pd

    from qmine.ops import drift

    df = _pooled_frame()
    clean = drift.snapshot_purity(df, "cls", "snapshot")
    assert clean["n_single_snapshot"] == 0

    df2 = pd.concat([df, pd.DataFrame([{"snapshot": "20260701", "query": f"z{i}",
                                        "weight": 1.0, "cls": "ONLY_B"} for i in range(50)])],
                    ignore_index=True)
    dirty = drift.snapshot_purity(df2, "cls", "snapshot")
    assert dirty["n_single_snapshot"] == 1
    assert dirty["single_snapshot"][0]["label"] == "ONLY_B"


def test_the_drift_phase_is_a_no_op_on_a_single_snapshot():
    """Every run before multi-input had one snapshot; none may change behaviour."""
    import inspect

    from qmine.graph.nodes import delivery

    src = _code_only(inspect.getsource(delivery.p10b_drift))
    assert 'if snap_col not in getattr(df, "columns", [])' in src
    assert "len(tags) < 2" in src, "a one-snapshot corpus must return early"


def test_the_drift_phase_and_document_reach_BOTH_modes():
    """fast mode disables agent prose; the drift analysis must survive it.

    A multi-snapshot run exists FOR the comparison, so losing it to the cheap mode
    would defeat the point. The document is generated from `drift_analysis.json`
    with no model call, which is what lets it ship in both.
    """
    import inspect

    from qmine.config import QMineConfig
    from qmine.graph.build import PHASE_NODES, SEQUENTIAL_TAIL
    from qmine.graph.nodes import delivery

    assert "p10b_drift" in [n for n, _ in PHASE_NODES]
    assert "p10b_drift" in SEQUENTIAL_TAIL
    for m in ("full", "fast"):
        assert "p10b" not in str(QMineConfig(mode=m, offline=True).fast_skipped)
    assert "_drift_document" in _code_only(inspect.getsource(delivery._p11_fast))
    assert "_drift_document" in _code_only(inspect.getsource(delivery.p11_report))


def test_the_snapshot_tag_never_becomes_a_reference_column():
    """Reference columns are the frame the K locator scores against.

    Declaring the snapshot as one asks the clustering to find a K that separates
    2025 from 2026 — meaningless, and the exact opposite of the shared frame the
    comparison depends on.
    """
    import inspect

    from qmine.graph.nodes import foundation
    from qmine.ops import audit

    p1 = _code_only(inspect.getsource(foundation.p1_audit))
    assert "snapshots=snapshots" in p1, "the tag must reach build_frame explicitly"
    assert "reference_labels=ref_labels" in p1
    # and it must be a separate parameter, not folded into reference_labels
    bf = _code_only(inspect.getsource(audit.build_frame))
    assert "snapshots: Sequence[str] | None" in bf
    assert 'df["snapshot"]' in bf


def test_total_variation_is_defined_when_a_class_is_missing_on_one_side():
    """It is summed over the UNION, so an emergent class contributes its full
    share rather than being skipped — which would understate the movement by
    exactly the part that is most interesting."""
    from qmine.ops.drift import _total_variation

    assert _total_variation({"a": 1.0}, {"a": 1.0}) == 0.0
    assert _total_variation({"a": 1.0}, {"b": 1.0}) == 1.0
    # half of |0.6-0.4| + |0.4-0.3| + |0-0.3|
    assert abs(_total_variation({"a": .6, "b": .4}, {"a": .4, "b": .3, "c": .3}) - 0.3) < 1e-9


def test_delta_concentration_separates_one_query_from_many():
    """A class can move because one entity blew up or because the behaviour
    broadened, and the product response is opposite. Measured on 影视: the
    -13.6pp streaming decline spread over 6,201 distinct queries (HHI 0.004),
    while the +9.7pp live-TV rise had one query carrying 23% of it.
    """
    import pandas as pd

    from qmine.ops.drift import _delta_concentration

    one = pd.DataFrame([{"snapshot": "B", "q": "cctv5"} for _ in range(100)]
                       + [{"snapshot": "A", "q": "cctv5"}])
    c = _delta_concentration(one, "snapshot", "q", "A", "B", None)
    assert c["hhi_of_delta"] == 1.0 and c["top1_share_of_delta"] == 1.0

    many = pd.DataFrame([{"snapshot": "B", "q": f"q{i}"} for i in range(100)])
    c2 = _delta_concentration(many, "snapshot", "q", "A", "B", None)
    assert c2["hhi_of_delta"] < 0.02, c2
    assert c2["n_distinct_queries"] == 100


def test_the_concentration_label_names_the_measurement_not_the_conclusion():
    """An earlier draft called the concentrated bucket 「疑似单一事件」 and the first
    real case refuted it: 央视直播 had top1=23%, but its top five deltas were all
    cctv5 phrasings — one ENTITY across many surface forms, not one event. The
    same rule as `test_the_report_does_not_present_a_confirmed_check_as_a_proven_defect`.
    """
    import inspect

    from qmine.report.zh_drift import _conc

    # Comments here legitimately name the phrase they warn against; test the CODE.
    src = _code_only(inspect.getsource(_conc))
    assert "疑似单一事件" not in src, "HHI measures concentration, not event-ness"
    assert _conc({"hhi_of_delta": 0.5, "top1_share_of_delta": 0.6,
                  "n_distinct_queries": 3}).startswith("**变化集中在少数 query**")
    assert _conc(None) == "—" and _conc({}) == "—"


def test_a_family_reaches_the_drift_table_with_a_name_not_a_bare_id():
    """The first real render showed `11` and `15` as the two biggest movers.
    Names are joined through LEAF MEMBERSHIP (`_shape.family_names`), never by
    integer id — an id join mismatched 19 of 19 families on live38.
    """
    import inspect

    from qmine.graph.nodes.delivery import _family_display

    src = _code_only(inspect.getsource(_family_display))
    assert "family_names" in src, "must reuse the leaf-membership join"
    assert "leaf_family_final" in src, "must prefer the DELIVERED partition"

    import pandas as pd
    # naming unavailable must degrade to ids, never raise: a bare-id table beats
    # a p10b that dies.
    class _D:
        has = staticmethod(lambda k: False)
        load = staticmethod(lambda k: (_ for _ in ()).throw(KeyError(k)))
        emit = staticmethod(lambda m: None)
    assert list(_family_display(_D(), pd.Series([11, 15]))) == ["#11", "#15"]


def test_the_snapshot_tag_is_not_reported_as_an_undeclared_legacy_label():
    """It is low-cardinality text by construction, so it tripped the legacy-label
    guard on EVERY pooled run — and that warning tells the operator to declare it
    via `--reference-columns`, which is precisely what
    `test_the_snapshot_tag_never_becomes_a_reference_column` forbids. Found by
    running a real two-snapshot run and reading the gate.
    """
    import pandas as pd

    from qmine.config import QMineConfig
    from qmine.graph.nodes.foundation import _label_like_columns

    cfg = QMineConfig(offline=True)
    raw = pd.DataFrame({"query": [f"q{i}" for i in range(100)],
                        "_snapshot": ["20250701"] * 50 + ["20260701"] * 50,
                        "legacy_cat": ["a", "b"] * 50})
    found = _label_like_columns(raw, cfg)
    assert "_snapshot" not in found, f"the pipeline's own tag is not a legacy label: {found}"
    assert "legacy_cat" in found, "a real legacy label must still be caught"


def test_the_pooling_rationale_states_the_measurement_that_was_actually_taken():
    """The same claim was written into THREE places and was wrong in all three.

    It said two runs over "the same 10,000 rows" shared "0 of 35" class codes.
    Measured: `fin01` and `fin02` are DIFFERENT files (金融 2025-07 and 2026-07)
    producing 20 and 19 classes with zero shared codes. The point survives — the
    codes are `LOOKUP_FX_RATE` vs `FX_RATE_LOOKUP`, parallel but not joinable —
    but the experiment described was not the experiment run, and this one reaches
    a shipped deliverable, where a reader cannot check it against the repo.
    """
    import inspect
    from pathlib import Path

    from qmine.ops import drift
    from qmine.report import zh_drift

    shipped = inspect.getsource(zh_drift)
    cli = Path(inspect.getsourcefile(drift)).parent.parent / "cli.py"
    for where, text in (("the shipped drift report", shipped),
                        ("ops/drift.py", inspect.getsource(drift)),
                        ("cli.py", cli.read_text(encoding="utf-8"))):
        assert "0/35" not in text and "0 of 35" not in text, \
            f"{where} still carries the retracted 0-of-35 figure"
        assert "完全相同的 10,000 行" not in text, \
            f"{where} still claims the two runs used identical rows"
    # and the real measurement must be the one the reader is given
    assert "fin01" in shipped and "fin02" in shipped
    assert "20" in shipped and "19" in shipped


# ==========================================================================
# The README's numbers
# ==========================================================================

def _evidence():
    """The cross-run table, imported rather than shelled out.

    An earlier version ran `tools/run_evidence.py --json /dev/stdout` and parsed
    the output; it silently SKIPPED because the table print and the JSON share
    one stream. A skipped check proves nothing, which is the whole reason this
    file exists.
    """
    import importlib.util
    from pathlib import Path

    import qmine
    root = Path(qmine.__file__).parent.parent.parent
    if not (root / "runs").exists():
        return root, None
    spec = importlib.util.spec_from_file_location(
        "_run_evidence", root / "tools" / "run_evidence.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import os
    cwd = os.getcwd()
    try:
        os.chdir(root)
        return root, mod.collect()
    finally:
        os.chdir(cwd)


def _reader_facing_markdown() -> str:
    """Every hand-written page a reader sees, concatenated.

    Deliberately NOT just README.md. The cross-run tables have already moved once
    (README -> docs/RESULTS.md), and a guard pinned to one filename silently
    stops guarding the moment its content is relocated — it does not fail, it
    passes on an empty search, which is the worse outcome.
    """
    from pathlib import Path

    import qmine
    root = Path(qmine.__file__).parent.parent.parent
    pages = [root / "README.md", *sorted((root / "docs").glob("*.md"))]
    return "\n".join(p.read_text(encoding="utf-8") for p in pages if p.exists())


def test_every_run_the_readme_names_still_exists_with_the_shape_it_claims():
    """The README now carries ~40 numbers read off fourteen runs.

    Every one of them goes stale the moment a run is re-rendered into a new
    generation, and a stale figure in a README is read as current — the same
    failure as the fast-mode banner's hardcoded "13 documents", which was wrong
    in eight places before anyone noticed.

    This asserts only that rows PRESENT in the README agree with
    `tools/run_evidence.py`; it deliberately does not assert the row count, which
    would fail on the fifteenth run rather than catch an error.
    """
    import re

    _, rows = _evidence()
    if not rows:
        pytest.skip("no runs/ directory in this checkout")
    ev = {r["run"]: r for r in rows}

    md = _reader_facing_markdown()
    # rows look like: | `live44` | K12 | 49,999 | 0.8796 | 3,000 | 20 | 53 / 23 | ...
    checked = 0
    for row in re.finditer(r"^\|\s*`([a-z0-9-]+)`\s*\|([^\n]+)$", md, re.M):
        run, rest = row.group(1), row.group(2)
        if run not in ev:
            continue
        cells = [c.strip().replace("**", "") for c in rest.split("|")]
        shape = next((c for c in cells if re.fullmatch(r"\d+ / \d+", c)), None)
        if shape:
            leaves, fams = (int(x) for x in shape.split(" / "))
            assert (leaves, fams) == (ev[run]["leaves"], ev[run]["families"]), (
                f"the docs say {run} delivered {leaves}/{fams}; the artifacts say "
                f"{ev[run]['leaves']}/{ev[run]['families']}")
            checked += 1
        for c in cells:
            if re.fullmatch(r"0\.\d{4}", c) and ev[run]["kappa"] is not None:
                assert abs(float(c) - ev[run]["kappa"]) < 1e-4, (
                    f"the docs quote kappa {c} for {run}; the artifact says {ev[run]['kappa']}")
                checked += 1
    assert checked >= 5, (
        f"only {checked} cells across README.md and docs/*.md matched a known run — "
        "the cross-run tables have gone missing, so nothing is being guarded")


def test_the_readme_never_quotes_a_kappa_for_a_single_annotator_run():
    """Absent is not 1.0 and not 0.0. A fast run has one annotator, so any kappa
    beside one of those run ids would be fabricated."""
    import re

    _, rows = _evidence()
    if not rows:
        pytest.skip("no runs/ directory in this checkout")
    fast = {r["run"] for r in rows if r["kappa"] is None}

    for row in re.finditer(r"^\|\s*`([a-z0-9-]+)`\s*\|([^\n]+)$", _reader_facing_markdown(), re.M):
        if row.group(1) in fast:
            assert not re.search(r"\|\s*\*{0,2}0\.\d{3,4}\*{0,2}\s*\|", row.group(2)), (
                f"{row.group(1)} has ONE annotator — it has no kappa to quote: {row.group(2)[:90]}")


def test_a_backfilled_drift_analysis_equals_what_the_live_phase_wrote():
    """`tools/backfill_drift.py` recomputes p10b for runs that predate the phase.

    It is only trustworthy if it is the SAME computation. `filmdrift` executed
    p10b live on a routed, weighted, two-snapshot corpus, so backfilling it is a
    control: the two artifacts must agree field for field. When this was first
    run they did — zero differences across inventory, churn, all three label axes
    and purity.

    Without this, "backfilled" would be a claim rather than a measurement, and
    five domains' reports would rest on it.
    """
    import json
    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    import qmine
    root = Path(qmine.__file__).parent.parent.parent
    live_p = root / "runs" / "filmdrift" / "gen01" / "drift_analysis.json"
    if not live_p.exists():
        pytest.skip("filmdrift (the live-p10b control) is not in this checkout")

    with tempfile.TemporaryDirectory() as tmp:
        r = subprocess.run([sys.executable, str(root / "tools" / "backfill_drift.py"),
                            "filmdrift", "--out", tmp],
                           cwd=root, capture_output=True, text=True)
        out_p = Path(tmp) / "filmdrift" / "drift_analysis.json"
        assert out_p.exists(), f"backfill wrote nothing (exit {r.returncode}): {r.stderr[-600:]}"
        back = json.loads(out_p.read_text(encoding="utf-8"))
    live = json.loads(live_p.read_text(encoding="utf-8"))

    back.pop("backfilled", None)          # provenance note, absent from a live run
    assert set(live) == set(back), f"key sets differ: {set(live) ^ set(back)}"
    for key in ("inventory", "query_churn", "by_label", "purity", "snapshots"):
        assert live[key] == back[key], (
            f"backfilled `{key}` differs from what the live phase wrote — the "
            f"backfill is not the same computation")


def test_the_translated_analysis_carries_the_same_numbers_as_the_original():
    """A translated report is a second copy of every number, and copies drift.

    `docs/DRIFT_ANALYSIS.zh.md` restates ~310 figures from the English original.
    Editing one and not the other is the obvious failure, and a reader of the
    Chinese version cannot check it against anything. This pins them together.

    Chinese renders large values in 万/亿 rather than M/bn, so those are allowed
    to differ by an explicit, arithmetic-checked mapping — nothing else is.
    """
    import collections
    import re
    from pathlib import Path

    import qmine
    docs = Path(qmine.__file__).parent.parent.parent / "docs"
    en_p, zh_p = docs / "DRIFT_ANALYSIS.md", docs / "DRIFT_ANALYSIS.zh.md"
    if not (en_p.exists() and zh_p.exists()):
        pytest.skip("the drift analysis is not in this checkout")

    def nums(text):
        text = re.sub(r"20\d\d-\d\d-\d\d|20\d\d", " ", text)     # dates are labels
        text = text.replace(",", "").replace("−", "-")
        return collections.Counter(re.findall(r"-?\d+\.\d+|(?<![\d.])\d+(?![\d.])", text))

    #: english token -> the 万/亿 rendering used in the Chinese. Each pair is an
    #: exact unit conversion, verified when the translation was written.
    SCALED = {
        "877.82": "8.7782", "782.06": "7.8206", "154.9": "1549", "20.16": "2016",
        "1466": "1.466", "3660": "3.66", "73": "7.3", "1000000": "100",
        "8.09": "809", "10.90": "1090", "16.45": "1645", "12.50": "1250",
        "9.74": "974", "5.21": "521", "28.50": "2850", "14.80": "1480",
        "9.97": "997", "7.79": "779", "8.04": "804", "11.83": "1183",
        "7.51": "751", "3.92": "392", "10.96": "1096", "17.54": "1754",
        "5.82": "582", "4.62": "462", "2.46": "246", "3.95": "395",
        "2.17": "217", "4.53": "453", "13.70": "1370", "0.32": "32",
        "0.27": "27", "0.70": "70", "0.17": "17", "0.67": "67", "0.80": "80",
        "0.36": "36", "0.13": "13", "696": "69.6",
    }
    en, zh = nums(en_p.read_text(encoding="utf-8")), nums(zh_p.read_text(encoding="utf-8"))
    missing = [v for v in en if v not in zh and SCALED.get(v) not in zh]
    assert not missing, (
        f"{len(missing)} figure(s) in the English analysis have no counterpart in the "
        f"Chinese one — the two have drifted apart: {sorted(missing)[:12]}")

    # and the structure must stay parallel, or a whole section was dropped
    e_txt, z_txt = en_p.read_text(encoding="utf-8"), zh_p.read_text(encoding="utf-8")
    assert (len(re.findall(r"^#{1,3} ", e_txt, re.M))
            == len(re.findall(r"^#{1,3} ", z_txt, re.M))), "heading counts differ"
    assert (len(re.findall(r"^\|", e_txt, re.M))
            == len(re.findall(r"^\|", z_txt, re.M))), "table row counts differ"


def test_the_chinese_analysis_points_at_chinese_figures():
    """An English-labelled figure inside a Chinese report is half-translated.

    The figures carry their own axis labels and footnotes, so `--lang zh` renders
    a second set; the Chinese document must reference those, not the originals.
    """
    import re
    from pathlib import Path

    import qmine
    zh_p = Path(qmine.__file__).parent.parent.parent / "docs" / "DRIFT_ANALYSIS.zh.md"
    if not zh_p.exists():
        pytest.skip("the drift analysis is not in this checkout")
    imgs = re.findall(r'src(?:set)?="([^"]+)"', zh_p.read_text(encoding="utf-8"))
    assert imgs, "no figures referenced at all"
    wrong = [i for i in imgs if "_zh" not in i]
    assert not wrong, f"Chinese report references English figures: {wrong}"
    for i in imgs:
        assert (zh_p.parent / i).exists(), f"missing figure {i}"


# ==========================================================================
# The comparison AXIS: what the two pooled groups differ by
# ==========================================================================
#
# `--input a,b` pools two files and p10b compares them. Everything in
# `ops/drift.py` is axis-agnostic; `report/zh_drift.py` was not, and said so in
# prose: 「不是趋势」, 「同月同日不等于季节可比」, 「时段性事件」. Those are true of
# two dates and false of two SAMPLING STRATA of one period — the AI-assistant
# head (top-N by PV) and tail (random) exports. A reader who believes them
# concludes user behaviour changed when nothing changed at all.

def _drift_payload():
    """A drift_analysis.json exercising every branch of the document."""
    return {
        "snapshots": ["head", "tail"],
        "inventory": [
            {"snapshot": "head", "rows": 100, "distinct_queries": 90, "weight_total": 5000.0},
            {"snapshot": "tail", "rows": 120, "distinct_queries": 110, "weight_total": 7000.0}],
        "query_churn": {"comparable": True, "shared": 40, "jaccard": 0.25,
                        "shared_weight_share_a": 0.5, "shared_weight_share_b": 0.4},
        "by_label": {"td_l1": {
            "comparable": True, "n_classes": 3, "cramers_v": 0.21,
            "total_variation_weight": 0.18, "total_variation_rows": 0.15,
            "n_comparisons": 40,
            "stable": [{"label": "A", "rows_a": 50, "rows_b": 60, "row_share_a": .5,
                        "row_share_b": .5, "weight_share_a": .4, "weight_share_b": .5,
                        "weight_share_delta_pp": 10.0, "z_row_share": 2.4,
                        "delta_concentration": None}],
            "emergent": [{"label": "E", "rows_a": 0, "rows_b": 30, "row_share_a": 0.0,
                          "row_share_b": .25, "weight_share_a": 0.0, "weight_share_b": .1,
                          "weight_share_delta_pp": 10.0, "z_row_share": 3.1,
                          "delta_concentration": None}],
            "receded": [{"label": "R", "rows_a": 30, "rows_b": 0, "row_share_a": .3,
                         "row_share_b": 0.0, "weight_share_a": .1, "weight_share_b": 0.0,
                         "weight_share_delta_pp": -10.0, "z_row_share": -3.1,
                         "delta_concentration": None}],
            "too_thin_to_compare": [{"label": "T"}]}},
        "purity": {"td_l1": {"checked": True, "n_groups": 3, "n_single_snapshot": 1,
                             "share_min": 0.0, "share_median": 0.5, "share_max": 1.0,
                             "single_snapshot": [{"label": "E", "rows": 30,
                                                  "share_of_head": 0.0}]}},
    }


def _render(axis: str) -> str:
    from qmine.report.zh_drift import build

    payload = _drift_payload()

    class _Cfg:
        class data:
            comparison_axis = axis

        class domain:
            key = "ai_assistant_zh"

    class _Deps:
        cfg = _Cfg()

        def load(self, _name):
            return payload

    return build({"run_id": "t01"}, _Deps())


def test_the_time_axis_document_is_unchanged_by_the_axis_parameter():
    """The axis table must be additive. Pinned against the pre-change document.

    Verified once against the original module byte-for-byte over seven payload
    shapes (full / no purity / clean purity / no churn / no labels / empty class
    lists / no snapshot tags) — all identical. These are the sentences that check
    survived, so a later edit to the `time` vocabulary trips here rather than
    silently rewording every existing multi-snapshot run's deliverable.
    """
    doc = _render("time")
    assert doc.startswith("# 快照对比 · 漂移分析")
    assert "同一套标签体系下，两个时间点的差异" in doc
    assert "| 快照 | 行数 | 去重 query | 总流量 |" in doc
    assert "### 2.1 两期都存在的类目" in doc
    assert "### 2.2 新出现的类目" in doc
    assert "### 2.3 消失的类目" in doc
    # the four caveats that only make sense about dates
    assert "- **不是趋势。** 这是两个时间点，不是一条曲线。" in doc
    assert "- **同月同日不等于季节可比。**" in doc
    assert "- **两期的抽样方式必须一致。**" in doc
    assert "它能说某一类涨了或跌了" in doc
    assert "少数几个通常是真实的时段性事件；" in doc, (
        "the purity note's time-specific clause was dropped once while "
        "parameterising this — 16 characters, and nothing else in the document moved")


def test_a_stratum_comparison_is_never_reported_as_change_over_time():
    """Two sampling strata of ONE period must not be described as drift.

    The AI-assistant exports are a top-1000-by-PV head and a random tail of the
    same month. Pooling them is right — one taxonomy has to label both or the two
    sides share no class codes (`fin01`/`fin02`: 20 and 19 classes, zero shared).
    Calling the result 漂移 is not.
    """
    doc = _render("stratum")
    assert doc.startswith("# 分层对比")
    assert "漂移" not in doc, "a stratum comparison must not use the drift vocabulary"
    for forbidden in ("不是趋势", "同月同日", "时段性事件", "两个时间点"):
        assert forbidden not in doc, f"{forbidden!r} describes dates, not strata"
    assert "不是时间变化" in doc, "it must say so positively, not merely omit the claim"
    assert "两层来自同一时间段" in doc


def test_the_stratum_caveats_do_not_warn_against_their_own_design():
    """The inverted caveat — the failure that made this parameterisation necessary.

    On the time axis 「两期的抽样方式必须一致」 is a real warning: differing sampling
    would masquerade as a real change. Between a head and a tail sample the
    differing sampling IS the independent variable, so the same sentence tells the
    reader the document is confounded when it is measuring what it set out to.
    """
    doc = _render("stratum")
    assert "两期的抽样方式必须一致" not in doc
    assert "抽样口径不同正是本报告的自变量" in doc
    # and the estimand limit that IS real here
    assert "不能推回总体" in doc


def test_an_unknown_comparison_axis_degrades_to_time_rather_than_dying():
    """A document whose every number is right must not be lost to a config typo."""
    assert _render("nonsense") == _render("time")


def test_the_stratum_deliverable_is_not_filed_under_the_drift_name():
    """A filename is a claim, and the numbers inside cannot undo it."""
    import inspect

    from qmine.graph.nodes import delivery

    src = _code_only(inspect.getsource(delivery._drift_document))
    assert "comparison_axis" in src
    assert "分层对比_头尾结构差异" in src
    assert "快照对比_漂移分析" in src, "the time axis must keep the name it has always had"


# ==========================================================================
# Preparing a multi-vertical assistant export (`tools/prepare_assistant_corpus`)
# ==========================================================================

def _prep():
    import importlib.util

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "prepare_assistant_corpus", root / "tools" / "prepare_assistant_corpus.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fake_exports(tmp_path):
    """Two categories with deliberately incomparable traffic scales."""
    import pandas as pd

    def frame(rows):
        return pd.DataFrame(rows, columns=["normalized_query", "query_1st_category_new",
                                           "query_2nd_category_new", "search_num"])
    head = frame([
        ("变清晰", "软件", "商用软件", 5_000_000),   # one string, most of the file's PV
        ("去水印", "软件", "商用软件", 1_000_000),
        ("好的", "软件", "商用软件", 100),
        ("好的", "法律", "普法问答", 90),            # SAME string, different category
        ("离婚怎么起诉", "法律", "普法问答", 60),
        ("欠钱不还怎么办", "法律", "普法问答", 50),
    ])
    tail = frame([
        ("图片糊了怎么修", "软件", "商用软件", 2),
        ("合同违约金上限", "法律", "普法问答", 1),
    ])
    h, t = tmp_path / "head.xlsx", tmp_path / "tail.xlsx"
    head.to_excel(h, index=False)
    tail.to_excel(t, index=False)
    return str(h), str(t)


def test_the_prepared_corpus_normalises_traffic_within_each_category(tmp_path):
    """Raw PV is not comparable across categories, so the run must not use it.

    Each category's top-N export has its OWN traffic floor — measured on the real
    corpus, 3 to 419 — so the file is a union of 33 censuses cut at 33 different
    depths. Left raw, two of the 33 categories hold 55% of pooled weight and the
    single string `变清晰` holds 17%, which makes every weighted metric in the run
    mostly a statement about that one string.
    """
    mod = _prep()
    df, audit = mod.build(*_fake_exports(tmp_path), weight_scale=1000.0)

    assert set(df.stratum) == {mod.HEAD_TAG, mod.RANDOM_TAG}, (
        "the stratum tags name the SAMPLING (top1k / random1k); `tail` was a "
        "conclusion about where the rows sit, which is a different claim")
    for (stratum, l1), g in df.groupby(["stratum", "l1"]):
        assert g.pv_norm.sum() == pytest.approx(1000.0), (
            f"{stratum}/{l1} carries {g.pv_norm.sum()}, not the 1000 every "
            f"category must carry for the categories to be comparable")
    head = df[df.stratum == mod.HEAD_TAG]
    raw_soft = head.loc[head.l1 == "软件", "search_num"].sum() / head.search_num.sum()
    norm_soft = head.loc[head.l1 == "软件", "pv_norm"].sum() / head.pv_norm.sum()
    assert raw_soft > 0.99, "the fixture must reproduce the domination it guards against"
    assert norm_soft == pytest.approx(0.5), "two categories, so each must carry half"
    assert audit["l1_categories"] == 2


def test_the_prepared_corpus_keeps_one_string_in_two_categories_apart(tmp_path):
    """Collapsing across categories would move a chip's whole traffic into one.

    `👌 好的，继续吧` appears under 32 of the real corpus's 33 first-level
    categories and carries 2.13M PV. Collapsing to distinct strings would assign
    all of it to whichever category happened to be modal, wrecking that category's
    normalisation and destroying the evidence that the string is not topical.
    Exact duplicate rows are still summed; the category attribution is data.
    """
    mod = _prep()
    df, _ = mod.build(*_fake_exports(tmp_path))
    rows = df[(df["query"] == "好的") & (df.stratum == mod.HEAD_TAG)]
    assert len(rows) == 2, "one row per category, not one row"
    assert set(rows.l1) == {"软件", "法律"}
    assert (rows.n_l1_categories == 2).all(), (
        "the breadth measurement is what makes a content-free string identifiable")
    assert rows.is_ack.all()
    assert not df.loc[df["query"] == "离婚怎么起诉", "is_ack"].any()


def test_the_acknowledgement_pattern_is_anchored_at_both_ends():
    """`好的` is an acknowledgement; `好的，帮我写一份年终总结` is a request.

    A substring match cannot tell them apart, and the family it defines carries
    15% of the real corpus's normalised traffic — so a loose pattern here would
    silently swallow real requests into the one class nobody reads.
    """
    import re

    mod = _prep()
    pat = re.compile(mod.ACK_PATTERN)
    for ack in ("好的", "嗯嗯", "👌 好的，继续吧", "🆗 行，继续吧", "可以", "继续", "OK"):
        assert pat.match(ack), f"{ack!r} is an acknowledgement"
    for real in ("好的，帮我写一份年终总结", "需要准备哪些材料", "继续写下一章",
                 "可以退款吗", "对方不还钱怎么办"):
        assert not pat.match(real), f"{real!r} is a request, not an acknowledgement"


# ==========================================================================
# The vertical crosstab (`tools/vertical_crosstab.py`)
# ==========================================================================

def _crosstab_mod():
    import importlib.util

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "vertical_crosstab", root / "tools" / "vertical_crosstab.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_vertical_crosstab_refuses_the_pre_governance_columns():
    """p8 rewrites the tree, so those columns describe one the run did not deliver.

    Anything shown as final must come from the DELIVERED partition. The refusal is
    in the tool rather than in a comment because the column names differ by one
    suffix and the wrong one produces a table that looks entirely plausible.
    """
    mod = _crosstab_mod()
    assert mod._PRE_GOVERNANCE == {"bu_leaf_pre_governance", "bu_family_pre_governance"}
    import inspect

    src = _code_only(inspect.getsource(mod.main))
    assert "_PRE_GOVERNANCE" in src and "sys.exit" in src


def test_the_crosstab_shares_are_within_category_not_across():
    """Across categories a row count measures the export's allocation, not the corpus.

    This export takes exactly 1,000 rows per category per stratum, so a class that
    is 80% one category partly reflects that quota. Within-category shares are the
    only ones that mean anything, and they must each sum to 1.
    """
    import pandas as pd

    mod = _crosstab_mod()
    df = pd.DataFrame({
        "ref_l1": ["金融"] * 6 + ["医疗"] * 4,
        "bu_family_final": ["A", "A", "A", "B", "B", "C", "A", "B", "B", "B"],
        "w": [10.0, 10, 10, 5, 5, 1, 100, 1, 1, 1],
    })
    ct = mod.crosstab(df, "ref_l1", "bu_family_final", "w")
    for _cat, g in ct.groupby("ref_l1"):
        assert g.row_share_in_category.sum() == pytest.approx(1.0)
        assert g.traffic_share_in_category.sum() == pytest.approx(1.0)
    # 医疗 has 4 rows to 金融's 6; within-category shares must not inherit that
    med = ct[ct.ref_l1 == "医疗"].set_index("bu_family_final")
    assert med.loc["B", "row_share_in_category"] == pytest.approx(0.75)


def test_the_crosstab_names_a_class_breadth_without_calling_it_a_verdict():
    """A class in 32 of 33 categories is not that category's property.

    That is exactly how the acknowledgement family appears, and the number that
    reveals it is breadth. It ships as a measurement — `n_categories` and the top
    category's share — with no verdict column, because whether breadth or
    concentration matters is the reader's call.
    """
    import pandas as pd

    mod = _crosstab_mod()
    df = pd.DataFrame({
        "ref_l1": ["a", "b", "c", "a", "a", "a"],
        "bu_family_final": ["ACK", "ACK", "ACK", "SPECIAL", "SPECIAL", "SPECIAL"],
    })
    ct = mod.crosstab(df, "ref_l1", "bu_family_final", None)
    conc = mod.concentration(ct, "ref_l1", "bu_family_final")
    assert conc.loc["ACK", "n_categories"] == 3
    assert conc.loc["SPECIAL", "n_categories"] == 1
    assert conc.loc["SPECIAL", "top_category_share"] == pytest.approx(1.0)
    assert "verdict" not in conc.columns


def test_the_translated_corpus_guide_carries_the_same_numbers_as_the_original():
    """Second copy of every number, same failure mode as the drift-analysis pair.

    `docs/AI_ASSISTANT_CORPUS.zh.md` restates the measurements that justify each
    preparation decision — the 33 PV floors, the 15.97%, the 17-of-33 that kills
    the population estimate. A reader of the Chinese edition cannot check them
    against anything, so they are pinned to the English here.
    """
    import re
    from pathlib import Path

    import qmine
    docs = Path(qmine.__file__).parent.parent.parent / "docs"
    en_p, zh_p = docs / "AI_ASSISTANT_CORPUS.md", docs / "AI_ASSISTANT_CORPUS.zh.md"
    if not (en_p.exists() and zh_p.exists()):
        pytest.skip("the corpus guide is not in this checkout")

    def nums(text):
        # an ASCII comma after a number is absorbed by the token; 全角 is not
        text = text.replace(",", "")
        return {m for m in re.findall(r"\d+(?:\.\d+)?%?", text) if len(m) > 1}

    #: English M -> the 万 rendering used in the Chinese. Exact conversions.
    SCALED = {"5.80": "580", "5.23": "523", "1.78": "178", "2.13": "213"}
    for eng, chi in SCALED.items():
        assert float(eng) * 1e6 == float(chi) * 1e4, f"{eng}M != {chi}万"

    en = {SCALED.get(x, x) for x in nums(en_p.read_text(encoding="utf-8"))}
    zh = nums(zh_p.read_text(encoding="utf-8"))
    assert not (en - zh), f"figures in the English edition missing from the Chinese: {sorted(en - zh)}"
    assert not (zh - en), f"figures in the Chinese edition missing from the English: {sorted(zh - en)}"

    zh_text = zh_p.read_text(encoding="utf-8")
    # It must land on the stratum deliverable. It DOES quote the drift vocabulary,
    # deliberately — §2 exists to explain why that framing is the wrong one — so
    # the check is on what the document concludes, not on what it mentions.
    assert "分层对比_头尾结构差异.md" in zh_text
    assert "data.comparison_axis: stratum" in zh_text
    # an untranslated Latin word wedged between CJK characters is a translation slip
    assert not re.findall(r"[一-鿿][a-zA-Z]{3,}[一-鿿]", zh_text)


def test_fast_mode_coverage_is_measured_not_asserted():
    """`ai04` reported 「覆盖率 100%」 with 225 of 3,000 gold rows unlabelled.

    The two-annotator path guards this: `n_sub = agree.get("n_submitted",
    agree["n"])`, and `agreement()` supplies `n_submitted`. The single-annotator
    path added for fast mode built its `agree` dict by hand and left the key out,
    so the fallback made `coverage = n / n = 1.0` — structurally, always,
    whatever failed. At 60% coverage it would still have reported 100% and
    passed, which is exactly the failure the comment above that computation was
    written about ("a kappa of 0.813 computed on 199 of 600 rows").

    `n` must also count rows that came back with a REAL label: a lost row is
    filled with the UNLABELED sentinel, so `len(labels_a)` is the SUBMITTED count
    wearing the answered count's name.
    """
    import inspect

    from qmine.graph.nodes.topdown import p2b_gold

    src = _code_only(inspect.getsource(p2b_gold))
    solo = src.split("if solo:", 1)[-1].split("else:", 1)[0]
    assert '"n_submitted"' in solo, (
        "the single-annotator agree dict must carry n_submitted, or the coverage "
        "guard degenerates to 1.0 and cannot fail")
    assert "UNLABELED" in solo, (
        "n must count rows that came back LABELLED, not rows that came back")


def test_the_coverage_fallback_cannot_silently_report_one():
    """The arithmetic itself, on the shape the bug produced."""
    agree_broken = {"n": 3000}                      # what the bug built
    agree_fixed = {"n": 2775, "n_submitted": 3000}  # what it must build

    def coverage(agree):
        n_sub = agree.get("n_submitted", agree["n"]) or 1
        return agree["n"] / n_sub

    assert coverage(agree_broken) == 1.0, "reproduces the defect"
    assert coverage(agree_fixed) == pytest.approx(0.925), "and the fix measures it"
    assert coverage(agree_fixed) < 0.95, (
        "92.5% must be distinguishable from 100% — a run that lost a whole "
        "annotator batch must not read as complete")


def test_a_pooled_run_carries_its_stratum_column_into_the_delivered_labels():
    """Every drift/stratum report ends by pointing at a column that was not written.

    「原始数据: `labels_full.csv`（逐行标签，含分层列）」 is the last line of the
    document, and it was FALSE on all six pooled runs on disk — fin-pool,
    med-pool, edu-pool, film-pool, ppl-pool, filmdrift — as well as `ai04`. None
    of their `labels_full.csv` carries a snapshot column, so a reader who
    followed the pointer to check a share, or to re-slice the comparison, could
    not. The whole document is about that column.

    Additive: `build_frame` only creates `snapshot` when the run pooled several
    inputs, so a single-snapshot run gains no column and its `labels_full.csv` is
    unchanged.
    """
    import inspect

    from qmine.graph.nodes import delivery

    src = _code_only(inspect.getsource(delivery.p10_deploy))
    assert 'if "snapshot" in df.columns:' in src, (
        "the delivered labels must carry the stratum column when the run has one")
    assert 'out["snapshot"] = df["snapshot"]' in src


def test_the_postprocessor_refuses_a_positional_join_it_cannot_verify():
    """`labels_full` is written in corpus order, so the join is by POSITION.

    A positional join that is silently wrong gives every row a different query's
    category — and the table still looks entirely plausible. `backfill_drift`
    refuses a misaligned positional join for the same reason.
    """
    import importlib.util

    import pandas as pd

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "postprocess_assistant_run", root / "tools" / "postprocess_assistant_run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    labels = pd.DataFrame({"query": ["a", "b", "c"], "td_l1": ["X", "Y", "Z"]})
    good = pd.DataFrame({"query": ["a", "b", "c"], "snapshot": ["top1k"] * 3,
                         "l1": ["p", "q", "r"], "weight": [3.0, 2.0, 1.0]})
    out = mod.join_verified(labels, good)
    assert list(out["l1"]) == ["p", "q", "r"]

    shuffled = good.iloc[[1, 0, 2]].reset_index(drop=True)
    with pytest.raises(SystemExit):
        mod.join_verified(labels, shuffled)
    with pytest.raises(SystemExit):
        mod.join_verified(labels, good.head(2))


def test_the_stratum_addendum_will_not_state_an_overlap_it_cannot_compute():
    """`weight` is normalised WITHIN (stratum, category), so it is two scales.

    Comparing one stratum's floor against the other's normalised weights returns
    "100% of the random sample sits above the head floor" — an artefact of the
    normalisation, and the exact incomparability this corpus preparation exists
    to handle. The addendum computes that statistic from RAW counts or says it
    did not compute it.
    """
    import importlib.util

    import pandas as pd

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "postprocess_assistant_run", root / "tools" / "postprocess_assistant_run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    df = pd.DataFrame({"snapshot": ["top1k"] * 3 + ["random1k"] * 3,
                       "l1": ["c"] * 6, "weight": [500.0, 300.0, 200.0, 400.0, 350.0, 250.0]})
    without = mod.stratum_addendum(df, "snapshot", "top1k", "random1k", raw=None)
    assert "本次未计算" in without, "an uncomputable overlap must be declared, not guessed"
    assert "0.00%" not in without and "100.00%" not in without

    raw = pd.DataFrame({"stratum": ["head"] * 3 + ["tail"] * 3, "l1": ["c"] * 6,
                        "search_num": [900, 500, 100, 3, 2, 1]})
    with_raw = mod.stratum_addendum(df, "snapshot", "top1k", "random1k", raw=raw)
    assert "本次未计算" not in with_raw
    assert "0 / 3 = 0.00%" in with_raw, "raw counts give the true, non-overlapping answer"
