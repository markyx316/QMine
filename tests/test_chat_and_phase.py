"""The conversational front door, and the phase that produces the comparison.

The conversation's whole safety property is that a model chooses from a fixed
list and a PROGRAM decides what runs. These pin that boundary, and the two
places the phase node itself has already failed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qmine.chat import CATALOGUE, ChatState, Session, keyword_route
from qmine.chat.intent import ChatPlan, ChatStep, extract_paths, route


class _Console:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def print(self, *a, **_k) -> None:
        self.lines.append(" ".join(str(x) for x in a))

    def rule(self, *a, **_k) -> None:
        self.lines.append("---")

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


# ------------------------------------------------------------------- routing

def test_the_path_extractor_splits_on_cjk_punctuation():
    """With only ASCII separators in the class, "我有三份导出：a.xlsx、b.xlsx"
    matched as ONE path, prefix and all, and the whole sentence was handed to
    `open()`."""
    got = extract_paths("我有三份医疗导出：data/a.xlsx、data/b.xlsx、data/c.xlsx。看看")
    assert got == ["data/a.xlsx", "data/b.xlsx", "data/c.xlsx"]
    assert extract_paths("（x/y.parquet）和「z.csv」") == ["x/y.parquet", "z.csv"]


def test_showing_files_never_escalates_to_a_paid_action(tmp_path):
    """Someone showing you files is not asking you to spend an afternoon's
    compute on them.

    Pinned over several phrasings, including ones that say "run" outright: the
    invariant is not "this one sentence routes to inspect", it is that NO
    keyword route over a message naming files reaches a spending action without
    the person having agreed to it. A narrower test passed while a mutation
    moved the escalation into a different branch.
    """
    phrasings = [
        "这几个文件 a.xlsx, b.xlsx 想跑一下看看",
        "帮我把 a.xlsx b.xlsx 跑起来挖掘",
        "run a.xlsx and b.xlsx now",
        "start mining c.parquet",
        "a.xlsx、b.xlsx 合并清洗然后开始",
        "对比 a.csv 和 b.csv",
    ]
    for msg in phrasings:
        plan = keyword_route(msg, {})
        spending = [s.action for s in plan.steps if CATALOGUE[s.action].spends]
        assert not spending, f"{msg!r} routed straight to {spending}"
        assert plan.steps, f"{msg!r} produced no step at all"
        assert plan.steps[0].action in ("inspect", "plan", "help", "list_runs"), \
            f"{msg!r} opened with {plan.steps[0].action}"


def test_a_request_naming_no_run_asks_instead_of_guessing():
    plan = keyword_route("对比一下", {})
    assert plan.clarify
    assert all(not CATALOGUE[s.action].needs_confirmation for s in plan.steps)


def test_an_action_outside_the_catalogue_cannot_even_be_expressed():
    """Making up a command is the one failure that cannot be recovered from,
    because the person will believe it exists.

    `ChatStep.action` is a Literal over the catalogue, so a model that wants to
    do something outside it fails schema validation at the tool-call layer and
    never reaches the executor. The executor checks again anyway, because a
    guarantee that lives in one place is a guarantee with one bug in it.
    """
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ChatStep(action="rm_rf")  # type: ignore[arg-type]

    c = _Console()
    s = Session(console=c, state=ChatState(run_root="runs"), ask=lambda _q: "y")
    step = ChatStep(action="help")
    object.__setattr__(step, "action", "rm_rf")      # bypass the schema entirely
    out = s._step(step)
    assert out["status"] == "error" and "not an action" in out["detail"]


def test_a_router_that_fails_does_not_end_the_conversation():
    class _Ctx:
        pass

    import qmine.chat.intent as intent

    class _Boom(intent.ChatRouterAgent):
        def __init__(self, _ctx):  # noqa: D107
            pass

        def run(self, **_kw):
            raise RuntimeError("provider down")

    orig = intent.ChatRouterAgent
    intent.ChatRouterAgent = _Boom  # type: ignore[misc]
    try:
        plan, source = route("help", {}, ctx=_Ctx())
    finally:
        intent.ChatRouterAgent = orig  # type: ignore[misc]
    assert "keyword" in source and plan.steps


# -------------------------------------------------------------- confirmation

def test_a_spending_action_stops_for_confirmation():
    """Asking before listing a directory teaches people to type `y` without
    reading, and the one prompt that matters then gets the same reflex."""
    asked: list[str] = []
    c = _Console()
    s = Session(console=c, state=ChatState(run_root="runs"),
                ask=lambda q: asked.append(q) or "n")
    out = s._step(ChatStep(action="run", params={"inputs": "a.csv", "run_id": "x"}))
    assert out["status"] == "declined" and asked
    assert "付费" in asked[0]

    asked.clear()
    s._step(ChatStep(action="list_runs"))
    assert not asked, "a free, reversible action must not ask"


def test_declining_a_confirmation_stops_the_rest_of_the_plan():
    c = _Console()
    s = Session(console=c, state=ChatState(run_root="runs"), ask=lambda _q: "n")
    import qmine.chat.session as sess

    plan = ChatPlan(steps=[ChatStep(action="run", params={"inputs": "a.csv"}),
                           ChatStep(action="compare", params={"run_id": "x"})])
    orig = sess.route
    sess.route = lambda *a, **k: (plan, "test")  # type: ignore[assignment]
    try:
        results = s.handle("跑一下然后对比")
    finally:
        sess.route = orig  # type: ignore[assignment]
    assert [r["action"] for r in results] == ["run"]
    assert results[0]["status"] == "declined"


def test_every_step_prints_the_equivalent_command():
    """An interface you can stop needing is the only honest kind."""
    c = _Console()
    s = Session(console=c, state=ChatState(run_root="runs"), ask=lambda _q: "n")
    s._step(ChatStep(action="compare", params={"run_id": "med-pool8", "axis": "stratum"}))
    assert "qmine compare med-pool8 --axis stratum" in c.text


def test_verify_refuses_to_run_without_a_control():
    """A harness that passes on one run proves nothing about the harness."""
    c = _Console()
    s = Session(console=c, state=ChatState(run_root="runs"), ask=lambda _q: "y")
    out = s._step(ChatStep(action="verify", params={"run_id": "x"}))
    assert out["detail"] == "no control"
    assert "对照" in c.text


# --------------------------------------------------------------- p10c phase

def _pooled_run(deps, n_snap: int = 2) -> Path:
    """Write a pooled generation THROUGH THE REAL STORE.

    A hand-rolled store double is looser than `ArtifactStore` — it was, and the
    phase failed against the real one while passing against the double, which is
    the stub-signature-drift failure this project has already paid for once.
    """
    rng = np.random.default_rng(0)
    n = 300 * n_snap
    snaps = sum(([f"s{i}"] * 300 for i in range(n_snap)), [])
    corpus = pd.DataFrame({"query": [f"查询{i}的内容" for i in range(n)],
                           "weight": rng.integers(1, 100, n), "snapshot": snaps,
                           "row_id": range(n)})
    deps.store.put_table("corpus", corpus, producer="test")
    pd.DataFrame({"query": corpus["query"],
                  "td_l1": rng.choice(["A", "B", "C"], n),
                  "td_l2": rng.choice(["a1", "a2"], n),
                  "bu_leaf": rng.integers(0, 3, n),
                  "bu_family_final": rng.integers(0, 2, n),
                  "td_confidence": rng.random(n),
                  "td_ambiguous": rng.random(n) < .1,
                  "bu_ambiguous": rng.random(n) < .1,
                  "snapshot": snaps}).to_csv(
        Path(deps.store.gen_dir) / "labels_full.csv", index=False)
    return Path(deps.store.gen_dir)


def test_the_pooled_phase_produces_the_comparison_on_a_pooled_run(deps):
    """`deps.df` is a PROPERTY.

    Calling it returned a DataFrame and then tried to call THAT, and the node's
    broad `except` turned a TypeError inside itself into "corpus unavailable" —
    a programming error wearing the costume of a graceful skip, with nothing in
    the log to notice. The phase ran in 0.0s and produced nothing, twice, before
    anything asked it for output.
    """
    from qmine.graph.nodes.delivery import p10c_pooled

    gen = _pooled_run(deps)
    deps.cfg.pooled.bootstrap_draws = 20
    deps.cfg.pooled.null_draws = 10
    out = p10c_pooled({}, deps)
    assert out["completed_phases"] == ["p10c"]
    assert "pooled_comparison" in (out.get("artifacts") or {}), out.get("events")
    assert (gen / "pooled").is_dir()
    assert list((gen / "pooled").glob("*.zh.md")), "a report must be written"
    assert (gen / "pooled" / "tables" / "summary.json").exists()
    # The FILES must be registered too, not merely written. `register_file`
    # validates its kind, and an invalid one raised into the node's own except
    # and reported "files not registered" — documents on disk with no artifact
    # pointing at them, and nothing in p11 to ship.
    for key in ("report_pooled", "workbook_pooled", "workbook_pooled_rows"):
        assert key in out["artifacts"], f"{key} was written but never registered"
        assert Path(out["artifacts"][key].path).exists()


def test_the_pooled_phase_is_a_noop_on_a_single_snapshot_run(deps):
    from qmine.graph.nodes.delivery import p10c_pooled

    gen = _pooled_run(deps, n_snap=1)
    out = p10c_pooled({}, deps)
    assert not out.get("artifacts")
    assert "single snapshot" in " ".join(out["events"])
    assert not (gen / "pooled").exists()


def test_a_failure_in_the_pooled_phase_cannot_halt_a_finished_run(deps):
    """It runs AFTER the labels are delivered and makes no model call. A failure
    here costs a set of extra documents, not the run."""
    from qmine.graph.nodes.delivery import p10c_pooled

    gen = _pooled_run(deps)
    (gen / "labels_full.csv").write_text("query\nmismatched\n", encoding="utf-8")
    out = p10c_pooled({}, deps)
    assert not out.get("halted")
    gate = list((out.get("gates") or {}).values())
    assert gate and gate[0].status in ("warned", "failed")
    assert not gate[0].halts_run


def test_the_comparison_is_analysis_so_fast_mode_keeps_it():
    """`--fast` removes the CHECKING, never the analysis. The comparison makes no
    model call, so a fast run ships the same tables as a full one."""
    from qmine.config import QMineConfig

    full, fast = QMineConfig(), QMineConfig(mode="fast")
    assert fast.pooled.enabled == full.pooled.enabled is True
    assert fast.pooled.bootstrap_draws == full.pooled.bootstrap_draws
    assert not any("pooled" in s or "compar" in s for s in fast.fast_skipped)
    smoke = QMineConfig(smoke_mode=True)
    assert smoke.pooled.bootstrap_draws < full.pooled.bootstrap_draws


def test_the_pooled_phase_sits_between_delivery_and_reporting():
    from qmine.graph.build import PHASE_NODES, SEQUENTIAL_TAIL
    from qmine.state import _PHASE_ORDER

    names = [n for n, _ in PHASE_NODES]
    assert names.index("p10_deploy") < names.index("p10c_pooled") < names.index("p11_report")
    assert "p10c_pooled" in SEQUENTIAL_TAIL
    # An id missing from the canonical order loses to every known value when two
    # concurrent branches write `phase`.
    assert "p10c" in _PHASE_ORDER


def test_the_dashboard_folds_the_sub_phase_onto_its_parent():
    """A new top-level id that is not a suffix of an existing one renders
    nowhere; `p10c` folds onto `p10` the way `p10b` already does."""
    from qmine.ui.live import declared_phase

    assert declared_phase("p10c") == "p10"


@pytest.fixture(autouse=True)
def _quiet_matplotlib():
    import matplotlib

    matplotlib.use("Agg")


def test_a_rendered_generation_does_not_claim_a_comparison_it_does_not_have(deps):
    """A generation inherits its predecessor's artifact index.

    So after a render `deps.has("report_pooled")` is true while the file sits one
    directory up — and the delivered-document index, whose links are relative,
    would point a reader at a path that does not exist. p10c does not run on a
    render, so a rendered generation simply has no comparison.
    """
    from qmine.graph.nodes.delivery import _pooled_refs

    gen = Path(deps.store.gen_dir)
    (gen / "pooled").mkdir(parents=True, exist_ok=True)
    here = gen / "pooled" / "跨快照对比_逐类对照.zh.md"
    here.write_text("x", encoding="utf-8")
    deps.store.register_file("report_pooled", here, "markdown", producer="p10c")
    assert _pooled_refs(deps) == ["report_pooled"]

    nxt = deps.store.new_generation("render")
    deps.store = nxt
    assert deps.has("report_pooled"), "the artifact is still inherited"
    assert _pooled_refs(deps) == [], \
        "a rendered generation must not list a file that is not in it"


def test_the_comparison_does_not_look_for_a_raw_file_column_in_the_mined_frame(deps):
    """`cfg.data.weight_column` names a column of the RAW FILE.

    `build_frame` renamed it to `weight` at p1, so forwarding it to the
    comparison hands `load_frame` a column the frame does not have — every
    snapshot falls back to uniform weight and every "traffic share" silently
    becomes the row share, with nothing in the output saying the number changed
    meaning.
    """
    from qmine.graph.nodes.delivery import _pooled_options

    deps.cfg.data.weight_column = "wise_pv"          # a real corpus config does this
    assert _pooled_options(deps).weight_column is None

    deps.cfg.pooled.weight_column = "pv_norm"        # an explicit override is honoured
    assert _pooled_options(deps).weight_column == "pv_norm"


def test_the_comparison_finds_the_frames_own_weight_column(deps):
    _pooled_run(deps)
    from qmine.pooled.frame import load_frame
    from qmine.pooled.source import DepsSource

    d, man = load_frame(DepsSource(deps), weight_column=None, axis="stratum")
    assert man.uniform_weight == [], "the frame has real weights and they must be used"
    assert d.groupby("snapshot")["pv_norm"].std().gt(0).all(), \
        "uniform pv_norm means the traffic share has silently become the row share"


def test_preparation_only_treats_the_axis_as_declared_when_someone_declared_it():
    """`comparison_axis` defaults to `time`, so forwarding it unconditionally
    made every plan "declared": no inference, no concern saying the axis is an
    assumption, and `confidence: high` for a guess."""
    from qmine.config import QMineConfig

    default = QMineConfig()
    assert "comparison_axis" not in default.data.model_fields_set
    explicit = QMineConfig.model_validate({"data": {"comparison_axis": "stratum"}})
    assert "comparison_axis" in explicit.data.model_fields_set


def test_the_printed_string_check_is_skipped_not_passed_when_nothing_was_quoted(deps):
    """An empty input set is a SKIP, never a PASS.

    "0 hard-rule strings printed" is a measurement only when the report actually
    quoted corpus rows. With nothing quoted the check has not run, and a gate
    reading PASS is indistinguishable from one that examined 2,000 strings.
    """
    from qmine.graph.nodes.delivery import p10c_pooled

    _pooled_run(deps)
    deps.cfg.pooled.bootstrap_draws = 20
    deps.cfg.pooled.null_draws = 10
    # Block everything, so the report can quote nothing.
    deps.cfg.pooled.extra_quote_patterns = [r"."]
    out = p10c_pooled({}, deps)
    gate = out["gates"]["p10c_no_hard_rule_string_printed"]
    assert gate.status == "skipped", f"status was {gate.status}"
    assert "nothing was checked" in gate.message


def test_the_status_view_does_not_call_a_warning_a_failure():
    """A warn-only gate records something to look at. Counting it as a failure
    makes every healthy run look broken and trains people to ignore the count."""
    import json as _json

    c = _Console()
    st = ChatState(run_root="runs")
    s = Session(console=c, state=st, ask=lambda _q: "n")
    root = Path(st.run_root)
    # Drive _do_status against a synthetic summary by pointing run_root at a tmp
    import tempfile

    tmp = Path(tempfile.mkdtemp())
    gen = tmp / "r1" / "gen01"
    gen.mkdir(parents=True)
    (gen / "run_summary.json").write_text(_json.dumps({
        "completed_phases": ["p0"], "halted": False,
        "gates": {"a": {"status": "passed"}, "b": {"status": "warned"},
                  "c": {"status": "skipped"}}}), encoding="utf-8")
    s.state.run_root = str(tmp)
    s._do_status({"run_id": "r1"})
    assert "未通过" not in c.text, "a warning was reported as a failure"
    assert "告警 1 个" in c.text
    assert root is not None
