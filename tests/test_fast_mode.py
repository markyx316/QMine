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
