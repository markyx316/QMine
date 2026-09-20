"""QMine over MCP: what a chat harness may reach, and what it may not.

The surface changes one thing structurally: the agent loop is no longer ours.
A tool the model can see is a tool it can call, so every limit that `qmine chat`
enforced in its own loop has to hold on the server side instead — where it holds
whatever harness is in front of it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qmine.mcp import answers, store
from qmine.mcp.authority import Authority
from qmine.mcp.server import QMineServer


@pytest.fixture(scope="module")
def pooled_run(tmp_path_factory) -> tuple[str, str]:
    """A small finished run with a real cross-snapshot comparison."""
    from qmine.pooled import DirSource, PooledOptions, run_comparison

    root = tmp_path_factory.mktemp("runs")
    gen = root / "demo" / "gen01"
    gen.mkdir(parents=True)
    rng = np.random.default_rng(0)
    n = 400
    snaps = ["A"] * 200 + ["B"] * 200
    # Class C is made entirely of rows the hard rules block, so the
    # "this class has rows but none may be reproduced" path is reachable at all.
    q = ([f"查询{i}的内容" for i in range(300)]
         + [f"我的手机号1381234{i:04d}" for i in range(100)])
    pd.DataFrame({"query": q, "weight": rng.integers(1, 50, n), "snapshot": snaps,
                  "row_id": range(n)}).to_parquet(gen / "corpus.parquet")
    pd.DataFrame({"query": q,
                  "td_l1": ["A"] * 150 + ["B"] * 150 + ["C"] * 100,
                  "td_l2": rng.choice(["x", "y"], n),
                  "bu_leaf": rng.integers(0, 3, n),
                  "bu_family_final": rng.integers(0, 2, n),
                  "td_confidence": rng.random(n),
                  "td_ambiguous": rng.random(n) < .1,
                  "bu_ambiguous": rng.random(n) < .1,
                  "snapshot": snaps}).to_csv(gen / "labels_full.csv", index=False)
    (gen / "run_summary.json").write_text(json.dumps({
        "mode": "fast", "halted": False, "completed_phases": ["p0", "p10c"],
        "llm_usage": {"provider": "routed"}, "fast_skipped": ["dual_annotation"],
        "gates": {"g1": {"status": "passed"}, "g2": {"status": "warned"}}}),
        encoding="utf-8")
    run_comparison(DirSource(gen), PooledOptions(axis="stratum", title="demo",
                                                 n_boot=20, n_null=20))
    return "demo", str(root)


def _srv(root: str, **env) -> QMineServer:
    s = QMineServer(run_root=root)
    s.authority = Authority(write_roots=(Path(root).resolve(),), **env)
    return s


# ------------------------------------------------------------ context budget

def test_the_headline_of_a_study_fits_in_a_conversation(pooled_run):
    """The report runs to hundreds of kilobytes. The failure mode of feeding it
    to a chat model is not a truncation warning — it is an assistant that saw
    the first fraction of one file and answers as if it read the study."""
    rid, root = pooled_run
    qm = _srv(root)
    payload = json.dumps(qm.call("qmine_findings", {"run_id": rid}), ensure_ascii=False)
    report = next(Path(root).glob(f"{rid}/gen01/pooled/*.zh.md"))
    assert len(payload) < 8_000, f"findings payload is {len(payload):,} chars"
    assert len(report.read_text(encoding="utf-8")) > 3 * len(payload), \
        "the fixture is too small to show the point"


def test_a_truncated_table_read_says_how_much_it_left_out(pooled_run):
    """A silently truncated table is read as the whole table, and every share
    computed from it is wrong in a way nothing on the page reveals."""
    rid, root = pooled_run
    out = _srv(root).call("qmine_table", {"run_id": rid, "table": "matrix_td_l1",
                                          "limit": 1})
    assert out["rows_shown"] == 1
    assert out["rows_in_table"] >= 3 and out["truncated"] is True
    assert out["source"].endswith("matrix_td_l1.csv")


def test_a_numpy_scalar_from_a_table_cell_survives_serialisation(pooled_run):
    """Every pandas cell is a numpy scalar, `json.dumps` refuses them, and the
    failure surfaces from inside a tool that worked in a REPL."""
    rid, root = pooled_run
    out = _srv(root).call("qmine_class", {"run_id": rid, "level": "l1", "name": "A"})
    json.dumps(out)                       # must not raise
    assert isinstance(out["rows_in_corpus"], int)
    assert answers.native({"a": np.int64(3), "b": np.float64(1.5),
                           "c": float("nan"), "d": np.bool_(True)}) == {
        "a": 3, "b": 1.5, "c": None, "d": True}


# ------------------------------------------------------------- the guard

def test_a_blocked_string_cannot_leave_through_the_chat_surface(pooled_run):
    """The tables were written under the seven layers — but a chat surface that
    reaches the same data by a different path would be a way around them, and a
    guard with a way around it is not a guard."""
    rid, root = pooled_run
    ref = store.resolve(rid, run_root=root)
    payload = answers.scrub(ref, {"rows": [{"query": "我的手机号13812345678"},
                                           {"query": "十岁苗条小女孩"},
                                           {"query": "乙肝疫苗多久打一次"}]})
    got = [r["query"] for r in payload["rows"]]
    assert got[:2] == [answers.BLOCKED, answers.BLOCKED], got
    assert got[2] == "乙肝疫苗多久打一次", "an ordinary row must pass through"


def test_a_class_whose_rows_are_all_guarded_reports_a_count_not_a_blank(pooled_run):
    """A blank reads as 'no data', which is a different and wrong statement — and
    a guard that silently deletes a section is the grounding-false-positive
    failure mode. Class C in the fixture is entirely rows the hard rules block."""
    rid, root = pooled_run
    out = _srv(root).call("qmine_examples", {"run_id": rid, "level": "l1", "name": "C"})
    assert out["examples"] == [], "a fully guarded class must quote nothing"
    cells = out["cells_with_nothing_quotable"]
    assert cells, "and must still report the rows it has"
    assert all(int(c["该类该快照条数"]) > 0 for c in cells)
    assert "seven-layer quote guard" in out["note"]

    ok = _srv(root).call("qmine_examples", {"run_id": rid, "level": "l1", "name": "A"})
    assert ok["examples"], "an ordinary class must still show real rows"


# ----------------------------------------------------------------- authority

def test_starting_a_paid_run_is_refused_without_permission_from_outside(pooled_run):
    """A token handed back through a tool result is a token the model can read
    and repeat, so authorisation cannot live inside the conversation."""
    _rid, root = pooled_run
    out = _srv(root).call("qmine_start_run", {"inputs": ["a.xlsx"], "run_id": "x"})
    assert out["status"] == "not_run_needs_a_person"
    assert "qmine" in out["run_this_yourself"] and "run" in out["run_this_yourself"]
    assert "QMINE_MCP_ALLOW_SPEND" in out["to_allow_it_here_instead"]

    allowed = _srv(root, allow_spend=True)
    assert allowed.authority.refusal("spend", "qmine_start_run") is None


def test_spending_is_off_unless_the_environment_turns_it_on(monkeypatch):
    """The default has to be read FROM THE ENVIRONMENT, because that is the only
    place a person can set it that the model cannot reach."""
    monkeypatch.delenv("QMINE_MCP_ALLOW_SPEND", raising=False)
    assert Authority.from_env().allow_spend is False
    monkeypatch.setenv("QMINE_MCP_ALLOW_SPEND", "1")
    assert Authority.from_env().allow_spend is True
    monkeypatch.setenv("QMINE_MCP_ALLOW_SPEND", "0")
    assert Authority.from_env().allow_spend is False


def test_a_write_refuses_a_path_outside_the_permitted_roots(pooled_run):
    """An output directory is the argument a model fills in most freely, and the
    study it could overwrite took hours to produce."""
    _rid, root = pooled_run
    qm = _srv(root)
    out = qm.call("qmine_prepare_corpus", {"inputs": ["a.csv"], "out": "/etc/qmine"})
    assert out["error"] == "refused" and "outside the permitted roots" in out["detail"]
    with pytest.raises(PermissionError):
        qm.authority.check_path("../../elsewhere")


def test_reads_never_stop_to_ask(pooled_run):
    """Asking permission to list a directory teaches people to approve without
    reading, and the one prompt that matters then gets the same reflex."""
    _rid, root = pooled_run
    a = _srv(root, allow_write=False).authority
    assert a.refusal("read", "qmine_findings") is None
    assert a.refusal("write", "qmine_build_comparison") is not None


# -------------------------------------------------------------- methodology

def test_an_absence_comes_back_with_its_verdict_not_a_bare_zero(pooled_run):
    """Zero of 950 rows is consistent with a share up to 0.39%. An assistant
    handed the zero alone will say the class does not occur."""
    rid, root = pooled_run
    out = _srv(root).call("qmine_class", {"run_id": rid, "level": "l1", "name": "C"})
    if out.get("absences"):
        assert "判定" in out["absences"][0]
        assert "判定" in out["how_to_read_an_absence"] or "verdict" in out["how_to_read_an_absence"]
    f = _srv(root).call("qmine_findings", {"run_id": rid})
    assert "cannot tell them apart" in f["how_to_read_the_distance"]


def test_a_stratum_axis_is_never_described_as_a_timeline(pooled_run):
    """The computation is axis-agnostic; the prose is not, and one caveat
    inverts. The tool result has to carry the caveat, because the model writes
    the sentence."""
    rid, root = pooled_run
    f = _srv(root).call("qmine_findings", {"run_id": rid})
    assert f["axis"] == "stratum"
    assert "NOT changes over time" in f["how_to_read_the_axis"]


def test_a_fast_run_announces_that_its_kappa_is_absent(pooled_run):
    rid, root = pooled_run
    out = _srv(root).call("qmine_overview", {"run_id": rid})
    assert out["mode"] == "fast"
    assert "ABSENT" in out["read_this_first"]


def test_the_glossary_defines_the_terms_that_mean_something_specific_here():
    for term in ("同源噪声上界", "均衡指数", "缺席判定", "stratum", "fast"):
        assert answers.glossary(term)["meaning"], term
    assert answers.glossary("nonsense")["meaning"] is None


# ---------------------------------------------------------------- the surface

def test_every_declared_tool_is_actually_served(pooled_run):
    """A declared-but-unserved tool is one the model reads about in the
    capabilities listing and can never call."""
    from qmine.mcp.server import build_app

    _rid, root = pooled_run
    app, qm = build_app(root)          # raises if the two lists disagree
    import asyncio

    served = {t.name for t in asyncio.run(app.list_tools())}
    assert served == {t["name"] for t in qm.specs}
    assert len(served) == len(qm.specs)


def test_a_missing_run_says_what_is_there_instead(pooled_run):
    _rid, root = pooled_run
    out = _srv(root).call("qmine_findings", {"run_id": "nope"})
    assert out["error"] == "not_found" and "Known" in out["detail"]


def test_a_run_without_a_comparison_is_told_how_to_get_one(tmp_path):
    gen = tmp_path / "bare" / "gen01"
    gen.mkdir(parents=True)
    (gen / "run_summary.json").write_text("{}", encoding="utf-8")
    out = QMineServer(run_root=str(tmp_path)).call("qmine_findings", {"run_id": "bare"})
    assert out["error"] == "not_found"
    assert "qmine compare" in out["detail"] and "no model call" in out["detail"]


def test_an_omitted_optional_argument_is_not_the_string_None(pooled_run):
    """`str(None)` is the four characters `None`.

    Over MCP an OMITTED optional argument arrives as the key present with a null
    value, so `str(a.get(k, ""))` yields `"None"` and every "did the caller name
    one?" test answers yes — the document tool then looked for a file called
    `None`. A direct call omits the key entirely and takes the `""` default, so
    this is invisible until the real transport carries it.
    """
    from qmine.mcp.server import _opt

    assert _opt({"x": None}, "x") is None
    assert _opt({}, "x") is None
    assert _opt({"x": "  "}, "x") is None
    assert _opt({"x": " a "}, "x") == "a"

    rid, root = pooled_run
    qm = _srv(root)
    # exactly the shape the MCP wrapper builds for `qmine_document(run_id=...)`
    out = qm.call("qmine_document", {"run_id": rid, "document": None,
                                     "section": None, "generation": None})
    assert "error" not in out, out
    assert out["sections"] and out["document"].endswith(".md")

    got = qm.call("qmine_examples", {"run_id": rid, "level": "l1", "name": None,
                                     "snapshot": None, "n": None, "generation": None})
    assert "error" not in got, got


def test_the_default_document_is_the_report_not_the_index(pooled_run):
    """`00_索引.md` sorts ahead of every CJK filename, so "show me the document"
    landed on the index rather than on the study."""
    rid, root = pooled_run
    gen = Path(root) / rid / "gen01"
    (gen / "00_索引.md").write_text("# index\n## a\n", encoding="utf-8")
    out = _srv(root).call("qmine_document", {"run_id": rid, "document": None})
    assert out["document"].endswith(".zh.md"), out["document"]


def test_the_dsh_patch_adds_a_plugin_rather_than_overriding_one():
    """`cordis.patch.yml` is a list of loader PATCHES, not a list of plugins.

    A bare `- id: … name: … config: …` is read as an override of an entry that
    already exists. Verified against dsh 0.1.5-rc.2: it answers
    `patch: entry "mcp-qmine" not found`, warns, skips — and starts a harness
    with no QMine tools in it and nothing in the UI to say why. Adding a plugin
    needs a patch whose `insert` holds the entry and which carries NO `id`, so
    it appends at the top level (`dsh-app-boot`: `else data.push(...insert)`).

    An override IS the right shape for an entry the profile already ships — the
    patch also points `agent-presets` at the QMine preset that way — but only
    then, and an override REPLACES the config object rather than merging into it.
    """
    import yaml

    from qmine.cli import _dsh_config

    doc = yaml.safe_load(_dsh_config("runs"))
    assert isinstance(doc, list)
    inserts = [e for e in doc if "insert" in e]
    assert len(inserts) == 1, "exactly one patch adds a plugin"
    patch = inserts[0]
    assert "id" not in patch, "an `id` on the patch would target a group, not the top level"

    # Every OTHER entry is a bare `id`, which is an override — legitimate, but
    # only for an id the shipped profile already has. dsh warns and SKIPS an
    # override of an id it cannot find, so an entry whose name drifts turns into
    # a silent no-op. This list is what has been verified present with
    # `dsh --profile web --dump-config`; adding to it means checking first.
    SHIPPED_IDS = {"agent-presets", "system-prompt", "mcp-qmine"}
    for entry in doc:
        if "insert" in entry:
            continue
        assert "id" in entry and "name" not in entry, (
            "an override carries only an id and a config; a `name` here would be "
            "read as a new plugin and rejected for having an id")
        assert entry["id"] in SHIPPED_IDS, f'{entry["id"]} is not known to exist'
        assert entry["config"], "an override with no config changes nothing"

    entry = patch["insert"][0]
    assert entry["name"] == "@deepseek-ai/dsh-mcp-client"
    cfg = entry["config"]
    assert cfg["serverName"] == "qmine" and cfg["transport"] == "stdio"
    assert cfg["args"][0] == "mcp"
    # Both of these are NOT the dsh defaults, and both defaults bite:
    # `failOnStartupError: false` starts a harness with zero tools and no error,
    # and 60s is shorter than a comparison rebuild.
    assert cfg["failOnStartupError"] is True
    assert cfg["toolCallTimeoutMs"] >= 300_000


def test_the_generated_config_is_not_pretty_printed(capsys):
    """rich wraps at the terminal width, and a wrapped absolute path in a YAML
    file is a config that parses into the wrong directory."""
    import inspect

    from qmine import cli

    src = inspect.getsource(cli.mcp_cmd)
    body = "\n".join(ln for ln in src.splitlines() if not ln.strip().startswith("#"))
    assert "print(_dsh_config(run_root))" in body
    assert "console.print(_dsh_config" not in body


# --------------------------------------------------------------- live progress

@pytest.fixture
def run_in_flight(tmp_path) -> Path:
    """A run that has started and NOT finished: no `run_summary.json`."""
    root = tmp_path / "flight"
    (root / "gen01").mkdir(parents=True)
    (root / "run.log").write_text(
        "12:23:53 INFO    qmine.graph: P0 foundation — run flight, generation 1\n"
        "12:23:56 INFO    qmine.graph: gate p0_provider: PASSED — real agents\n"
        "12:23:56 INFO    qmine.graph: ✔ p0_foundation completed in 3.2s\n"
        "12:23:56 INFO    qmine.graph: P1 audit — profiling corpus\n"
        "12:24:10 INFO    qmine.graph: gate p1_template_coverage: WARNED — thin\n"
        "12:24:11 INFO    qmine.graph: ✔ p1_audit completed in 15.0s\n"
        "12:24:11 INFO    qmine.graph: P2a taxonomy — five researchers\n",
        encoding="utf-8")
    (root / "usage.json").write_text(json.dumps({
        "calls": 42, "input_tokens": 100, "output_tokens": 20, "errors": 1,
        "elapsed_s": 600, "by_role": {"annotator_a": {"calls": 30},
                                      "researcher": {"calls": 12}}}), encoding="utf-8")
    (root / "index.jsonl").write_text(
        json.dumps({"name": "data_audit", "producer": "p1", "rows": None}) + "\n"
        + json.dumps({"name": "corpus", "producer": "p1", "rows": 20316}) + "\n",
        encoding="utf-8")
    (root / "findings.json").write_text(json.dumps({"n_open": 2, "n_confirmed_open": 0}),
                                        encoding="utf-8")
    (root / "dashboard.html").write_text("<html></html>", encoding="utf-8")
    (root / "gen01" / "data_audit.json").write_text(
        json.dumps({"n_rows": 20316, "median_chars": 11}), encoding="utf-8")
    return root


def test_a_progress_reading_announces_that_it_is_a_snapshot(run_in_flight):
    """The characteristic failure of a polled surface is not a wrong number — it
    is a right number restated ten minutes later as if it were current."""
    from qmine.mcp import progress

    out = progress.read(run_in_flight)
    assert out["as_of"]
    assert "call this tool again" in out["how_to_use_this"]


def test_an_unfinished_run_refuses_to_describe_its_results(run_in_flight):
    """`run_summary.json` is the only file that waits for the end, which is why
    its absence is the test for 'not finished'."""
    from qmine.mcp import progress

    out = progress.read(run_in_flight)
    assert out["finished"] is False
    assert "would be invented" in out["do_not_report_results"]

    qm = QMineServer(run_root=str(run_in_flight.parent))
    st = qm.call("qmine_status", {"run_id": run_in_flight.name})
    assert st["finished"] is False and "do_not_report_results" in st


def test_progress_reads_the_phase_gates_and_spend_off_the_live_files(run_in_flight):
    from qmine.mcp import progress

    out = progress.read(run_in_flight)
    assert out["current_phase"]["phase"] == "P2a"
    assert out["current_phase"]["still_in_it"] is True
    assert [p["phase"] for p in out["phases_completed"]] == ["p0_foundation", "p1_audit"]
    assert out["gates_so_far"]["passed"] == 1
    assert out["gates_so_far"]["warned"] == ["p1_template_coverage"]
    assert out["spend_so_far"]["model_calls"] == 42
    assert out["spend_so_far"]["busiest_roles"][0] == ["annotator_a", 30]
    assert out["open_findings"] == 2
    assert out["live_dashboard"]["path"].endswith("dashboard.html")


def test_progress_says_what_is_already_answerable(run_in_flight):
    """A live surface's job is not only "where is it" but "what can I ask now"."""
    from qmine.mcp import progress

    out = progress.read(run_in_flight)
    subjects = {x["with"] for x in out["what_can_be_answered_now"]}
    assert any("corpus" in s for s in subjects), subjects
    assert not any("taxonomy" in s for s in subjects), "p2a has not written one yet"


def test_a_partial_read_that_is_not_ready_names_the_phase_it_waits_on(run_in_flight):
    """Far better than 'not found': it tells the person what to wait for."""
    from qmine.mcp import progress

    out = progress.partial(run_in_flight, "taxonomy")
    assert out["available"] is False and out["waiting_on_phase"] == "p2a"
    # Every tool named in a hint must be one the model can actually call.
    from qmine.mcp.server import QMineServer

    served = {t["name"] for t in QMineServer(run_root="runs").specs}
    for word in out["note"].split():
        if word.startswith("qmine_"):
            assert word.strip(".,`") in served, f"hint names a tool that does not exist: {word}"
    ready = progress.partial(run_in_flight, "corpus")
    assert ready["available"] is True and ready["fields"]["n_rows"] == 20316


def test_every_partial_read_is_labelled_intermediate(run_in_flight):
    """p8 rewrites the tree and the taxonomy can be redrawn, so a mid-run
    artifact describes the run's state now, not what it will deliver."""
    from qmine.mcp import progress

    out = progress.partial(run_in_flight, "corpus")
    assert "INTERMEDIATE" in out["mid_run_caveat"]
    with pytest.raises(ValueError, match="unknown subject"):
        progress.partial(run_in_flight, "nonsense")


def test_doctor_actually_exercises_the_chat_front_door():
    """The failure that costs a day is a harness that starts fine with no QMine
    tools in it: everything looks healthy and the assistant answers from memory.
    So `doctor` must BUILD the tool surface, not grep for it — a source check
    stays green when the call is removed and only the import is left.
    """
    from typer.testing import CliRunner

    from qmine.cli import app

    out = CliRunner().invoke(app, ["doctor"]).output
    flat = " ".join(out.split())
    for probe in ("mcp SDK", "qmine mcp", "node (for dsh)", "dsh config"):
        assert probe in flat, f"doctor does not report {probe!r}"
    # The `qmine mcp` row must carry a real tool count, which only a successful
    # build_app can produce.
    assert re.search(r"\d+\s+tools", flat), flat[:400]
    assert "spend is" in flat, "doctor must say whether spending is allowed"
