"""Regression tests for the operator's view of a running pipeline.

The dashboard and the run log are how a two-hour, thirty-dollar run is watched
and, afterwards, explained. Neither had a single test, and rendering the panel
against a real ``run.log`` for the first time found five defects in one sitting:
a layout that never split, agent names eaten as markup, a phase row that could
not light up, gate notes cut mid-word, and a halted run still showing ◐.

Every test below pins one of those.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from rich.console import Console

from qmine.ui.live import PHASES, LiveDashboard, declared_phase, parse_log_line

ANSI = re.compile(r"\x1b\[[0-9;]*m")


def render(dash: LiveDashboard, width: int = 175) -> str:
    con = Console(record=True, force_terminal=True, width=width, height=60)
    dash._console = con
    con.print(dash._render())
    return ANSI.sub("", con.export_text())


def feed(dash: LiveDashboard, *lines: str) -> LiveDashboard:
    for line in lines:
        dash._ingest(line)
    return dash


def test_sub_phase_lines_light_up_the_phase_they_belong_to():
    """`P3a`/`P3b`/`P3c` are emitted; the list declares one `p3` row.

    Keying status on the emitted string left the representation phase at ○ for
    the whole run — a completed run finished with a row that had never started.
    """
    assert declared_phase("P3c") == "p3"
    assert declared_phase("P2a") == "p2a"
    assert declared_phase("P99") is None, "an undeclared phase must not create a ghost row"

    dash = feed(LiveDashboard(), "P3a encoder bake-off — 4 candidates",
                "P3c alpha sweep — [0.0, 0.1, 0.5]", "P4 algorithm battery")
    assert dash.status["p3"] == "done", "p3 never completed"
    assert "p3a" not in dash.status, "timings filed under a key nothing renders"


def test_sub_steps_do_not_restart_the_phase_clock():
    """P3 spans three emitted sub-steps; its elapsed time is the whole span."""
    dash = feed(LiveDashboard(), "P3a encoder bake-off")
    first = dash._phase_start["p3"]
    feed(dash, "P3b sparse block", "P3c alpha sweep")
    assert dash._phase_start["p3"] == first


def test_agent_specialties_survive_into_the_activity_feed():
    """Rich read `[log_reading]` as a style tag and swallowed it.

    Three researchers then showed as three identical "researcher" lines, which
    is precisely the distinction an operator is watching the feed for.
    """
    dash = feed(LiveDashboard(), "P2a taxonomy", "  researcher[log_reading] → 1 candidates")
    assert "log_reading" in render(dash)


def test_a_gate_note_reaches_the_screen_whole():
    """The note is the actionable half; it was cut at 70 chars, then cut again."""
    note = ("held-out rows land in the same cluster 92.6% of the time "
            "(95% CI 0.906-0.942, n=800) — below the 98% threshold")
    dash = feed(LiveDashboard(), f"gate p6_heldout_reproduction: FAILED — {note}")
    out = render(dash)
    assert "98% threshold" in out.replace("\n", " ").replace("  ", " ")


def test_a_halted_run_does_not_still_show_a_phase_running():
    """A blocking gate halts by returning, not by raising, so `!!` never fires."""
    dash = feed(LiveDashboard(), "P2a taxonomy — 3 researchers fanning out")
    assert dash.status["p2a"] == "running"
    dash.finish(ok=False)
    assert dash.status["p2a"] == "failed"
    assert dash.timings.get("p2a") is not None, "a halted phase still has an elapsed time"


def test_metric_labels_are_whole_words():
    """The scrape sliced labels mid-token: "held-out structure…" became "out…"."""
    dash = feed(LiveDashboard(), "P9 panel", "  held-out structure reproduction 0.926",
                "  raw agreement 0.792, kappa 0.716, 25 disagreements")
    labels = [lbl for lbl, _ in dash.metrics]
    assert "held-out structure reproduction" in labels
    assert "kappa" in labels
    assert not any(lbl.startswith("out ") for lbl in labels)


def test_the_panel_splits_into_two_panes_on_a_wide_terminal():
    """`Columns(expand=True)` sizes by content, so the panes never sat side by side.

    The phase list rendered alone across the full width and everything else
    stacked below it. The check: on a wide console the first line must contain
    both pane titles.
    """
    dash = feed(LiveDashboard(), "P1 audit — profiling corpus")
    first = render(dash, width=175).splitlines()[0]
    assert "阶段" in first and "QMine" in first, first


def test_the_layout_collapses_rather_than_wrapping_when_narrow():
    dash = feed(LiveDashboard(), "P1 audit — profiling corpus")
    lines = render(dash, width=90).splitlines()
    assert "阶段" in lines[0] and "QMine" not in lines[0]
    assert max(len(line) for line in lines) <= 90


def test_no_phase_label_is_clipped_by_the_pane_width():
    """The width is derived from the labels; CJK glyphs are two cells wide."""
    dash = LiveDashboard()
    out = render(dash, width=175)
    for spec in PHASES:
        assert spec.label_zh in out, f"{spec.label_zh} was wrapped or clipped"


def test_the_run_log_is_written_whatever_the_console_is_doing(tmp_path: Path):
    """Choosing the dashboard used to mean choosing to have no record at all.

    The CLI quieted the *logger*, which gates records before any handler sees
    them, so `<run>/run.log` went silent too and a halted run left nothing but
    whatever six lines the panel happened to be showing.
    """
    from qmine.runner import _run_log

    log = logging.getLogger("qmine")
    log.setLevel(logging.WARNING)          # what a quiet console used to do
    with _run_log(tmp_path):
        log.info("gate p2a_pilot_agreement: FAILED — kappa 0.761")
    text = (tmp_path / "run.log").read_text()
    assert "kappa 0.761" in text
    assert log.level == logging.WARNING, "the caller's level must be restored"


def test_the_log_format_and_the_follower_agree(tmp_path: Path):
    """`qmine watch` reads what `_run_log` writes; nothing else couples them.

    If the formatter changes, the follower does not error — it silently matches
    nothing and shows an empty panel for the whole run, which looks exactly like
    a hung pipeline. This is the only thing holding the two ends together.
    """
    from qmine.runner import _run_log

    lines = ["P2a taxonomy — 3 researchers fanning out",
             "  researcher[log_reading] → 1 candidates",
             "gate p2a_pilot_agreement: FAILED — kappa 0.761 (95% upper 0.814) on 200 queries"]
    log = logging.getLogger("qmine")
    with _run_log(tmp_path):
        for line in lines:
            log.info(line)

    recovered = [parse_log_line(raw) for raw in (tmp_path / "run.log").read_text().splitlines()]
    assert [r for r in recovered if r] == lines


def test_a_follower_shows_the_same_state_as_the_live_panel(tmp_path: Path):
    """Attaching late must reconstruct everything, not just what arrives next."""
    from qmine.runner import _run_log

    log = logging.getLogger("qmine")
    with _run_log(tmp_path):
        log.info("P1 audit — profiling corpus")
        log.info("gate p1_template_coverage: PASSED — 12 phrasing families cover 36.6%")
        log.info("P2a taxonomy — 3 researchers fanning out")

    follower = LiveDashboard(run_id="live31")
    for raw in (tmp_path / "run.log").read_text().splitlines():
        msg = parse_log_line(raw)
        if msg:
            follower._ingest(msg)
    assert follower.status["p1"] == "done"
    assert follower.status["p2a"] == "running"
    assert follower.gates == [("p1_template_coverage", "PASSED",
                               "12 phrasing families cover 36.6%")]


def test_a_tool_loop_reports_the_tokens_it_spent():
    """The tool path recorded a hardcoded zero for every model turn it made.

    Two consequences, the second worse than the first: web-researching agents
    showed `out=0` in `run_summary.json`, and the ledger's output-token ceiling
    could not see the only code path that *iterates* — so the budget guard was
    blind to precisely the thing that can run away.
    """
    from qmine.agents.base import _tool_loop_usage

    class Msg:
        def __init__(self, i, o):
            self.usage_metadata = {"input_tokens": i, "output_tokens": o}

    result = {"messages": [Msg(1200, 300), Msg(1800, 450), object()]}  # a tool call carries none
    assert _tool_loop_usage(result) == {"input_tokens": 3000, "output_tokens": 750}
    assert _tool_loop_usage({}) == {"input_tokens": 0, "output_tokens": 0}
    assert _tool_loop_usage(None) == {"input_tokens": 0, "output_tokens": 0}


def test_gold_annotation_reports_progress_while_it_runs(deps, monkeypatch):
    """120 batches per annotator used to emit nothing at all until they all landed.

    At eight concurrent calls and roughly 150s each that is ~40 minutes of dead
    air per annotator, in the phase that dominates a live run's wall clock. A
    watcher cannot tell that from a hung provider.
    """
    from types import SimpleNamespace

    from qmine.graph.nodes import topdown

    class FakeBatch:
        def __init__(self, chunk):
            self.labels = [SimpleNamespace(query=q, model_dump=lambda q=q: {"label": "X"})
                           for q in chunk]

    class FakeAgent:
        def __init__(self, ctx, suffix=""): pass
        def run(self, *, queries, **kw): return FakeBatch(queries)

    monkeypatch.setattr(topdown, "AnnotatorAgent", FakeAgent)
    ctx = SimpleNamespace(registry=SimpleNamespace(is_offline=False), cfg=deps.cfg)

    seen: list[str] = []
    deps.on_event = seen.append
    queries = [f"q{i}" for i in range(500)]          # 20 batches of 25
    out = topdown._annotate(ctx, "a", queries, "", "", "", deps)

    assert len(out) == len(queries), "every row must still be labelled"
    progress = [m for m in seen if "batches" in m]
    assert progress, f"no progress emitted; saw {seen}"
    assert len(progress) < 20, "one line per batch would flood the panel"
    assert "20/20" not in " ".join(progress), "the final count is already reported by the summary line"


AGENT_LINE = ("  ~ taxonomy_architect ok 704.2s out 41,808 · dashscope/glm-5.1 · "
              "nodes=18, rules=0")


def test_the_panel_shows_what_each_agent_returned():
    """"The agent finished" is not information; "nodes=18, rules=0" is.

    Without this the operator sees phase names for a twelve-minute architect call
    and cannot tell a model thinking from a model stuck, nor — when it lands —
    whether what came back was any good.
    """
    dash = feed(LiveDashboard(), "P2a taxonomy", AGENT_LINE)
    assert dash.agents == [("taxonomy_architect", True, "704.2", "41,808",
                            "dashscope/glm-5.1", "nodes=18, rules=0")]
    out = render(dash)
    assert "taxonomy_architect" in out
    assert "nodes=18" in out, "the return summary must reach the screen"
    assert "41,808" in out, "and what it cost to get it"


def test_a_failed_agent_turn_is_shown_as_failed():
    dash = feed(LiveDashboard(), "P2a taxonomy",
                "  ~ annotator_b !! 12.0s out 0 · deepseek-v4-flash · BadRequestError")
    assert dash.agents[0][1] is False
    assert "✗" in render(dash)


def test_an_agent_line_is_not_also_scraped_as_a_metric():
    """`out 41,808 · … 704.2s` is full of numbers that are not metrics."""
    dash = feed(LiveDashboard(), "P2a taxonomy", AGENT_LINE)
    assert dash.metrics == [], f"agent telemetry leaked into the metrics box: {dash.metrics}"


def test_repeated_activity_lines_collapse_instead_of_flooding():
    """Eight concurrent batches fail identically in the same second.

    Each message wraps to two display lines, so one benign already-handled error
    filled all six activity slots and pushed out the progress it was competing
    with — found by rendering a live run's log, not a recorded clean one.
    """
    err = "role=annotator_a attempt=0 failed (BadRequestError: 400 must contain 'json')"
    dash = feed(LiveDashboard(), "P2b gold", *([err] * 8), "annotator[a] labelled 200/200")
    assert len(dash.activity) <= 3, dash.activity
    assert any("×8" in line for line in dash.activity), dash.activity
    assert any("labelled 200/200" in line for line in dash.activity), \
        "real progress must survive the flood"


def test_the_running_phase_explains_itself():
    """A phase name says where you are, not what a four-minute pause is buying."""
    dash = feed(LiveDashboard(), "P2a taxonomy — 5 researchers fanning out")
    out = render(dash)
    assert "架构师" in out, "the running phase's explanation must be on screen"

    en = feed(LiveDashboard(language="en"), "P2a taxonomy")
    assert "rule writer" in render(en)


def test_every_phase_carries_an_explanation_in_both_languages():
    for spec in PHASES:
        assert spec.why_zh, f"{spec.key} has no Chinese explanation"
        assert spec.why_en, f"{spec.key} has no English explanation"


def test_a_rejected_response_is_billed_even_though_it_was_rejected(cfg, tmp_path):
    """A rejected REQUEST is free; a rejected RESPONSE was generated and billed.

    Recording both as zero made the ledger — and the output-token ceiling that
    reads it — optimistic exactly on the paths that misbehave, which is where a
    runaway shows up first. All three providers reject the first structured-output
    attempt, so this path runs on every model of every run.
    """
    from types import SimpleNamespace

    from qmine.llm.registry import ModelRegistry

    reg = ModelRegistry(cfg.llm, cache_dir=tmp_path, run_cfg=cfg)
    reg._raw.last = SimpleNamespace(
        usage_metadata={"input_tokens": 3540, "output_tokens": 4102})
    usage = getattr(reg._raw.last, "usage_metadata", None) or {}
    reg.ledger.record("researcher", error=True,
                      input_tokens=int(usage["input_tokens"]),
                      output_tokens=int(usage["output_tokens"]))
    snap = reg.ledger.snapshot()
    assert snap["errors"] == 1
    assert snap["output_tokens"] == 4102, "the tokens the provider billed for"


def test_a_tool_loop_turn_is_cached_and_transcribed_like_any_other(cfg, tmp_path):
    """`ToolAgent` bypasses `complete`, and `_store` is the only writer of both
    `llm_cache/` and the `raw_log` that becomes `agent_transcript.json`.

    So the two agents whose claims are least verifiable — they cite pages nobody
    else saw — were the only two leaving no record and no replayable response.
    """
    from qmine.llm.registry import ModelRegistry

    reg = ModelRegistry(cfg.llm, cache_dir=tmp_path, run_cfg=cfg)
    reg.record_external_turn("researcher_literature", "deep", "sys", "usr",
                             {"candidates": ["a", "b"]}, latency=12.0)
    assert list(tmp_path.glob("*.json")), "no cache entry: the turn cannot be replayed"
    assert reg.raw_log, "no transcript entry: nothing records what it returned"
    assert reg.raw_log[-1]["role"] == "researcher_literature"


def test_annotation_progress_is_about_ten_updates_at_any_size(deps, monkeypatch):
    """`len // 10` alone emits once per batch below ten; the pilot printed all 8."""
    from types import SimpleNamespace

    from qmine.graph.nodes import topdown

    class FakeAgent:
        def __init__(self, ctx, suffix=""): pass
        def run(self, *, queries, **kw):
            return SimpleNamespace(labels=[
                SimpleNamespace(query=q, model_dump=lambda q=q: {"label": "X"})
                for q in queries])

    monkeypatch.setattr(topdown, "AnnotatorAgent", FakeAgent)
    ctx = SimpleNamespace(registry=SimpleNamespace(is_offline=False), cfg=deps.cfg)

    for n_rows, batches in ((200, 8), (3000, 120)):
        seen: list[str] = []
        deps.on_event = seen.append
        topdown._annotate(ctx, "a", [f"q{i}" for i in range(n_rows)], "", "", "", deps)
        progress = [m for m in seen if "batches" in m]
        assert len(progress) <= 10, f"{batches} batches emitted {len(progress)} updates"


def test_a_research_angle_that_returns_nothing_is_flagged():
    """One live run had `literature` return zero while the other four returned
    6-12, and nothing anywhere noticed — the phase still announced five
    researchers fanning out and the taxonomy was built from four angles."""
    src = Path("src/qmine/graph/nodes/topdown.py").read_text()
    i = src.index("researcher[{angle['key']}] \u2192")
    window = src[max(0, i - 600):i]
    assert "if not sub.candidates" in window, "an empty angle must warn"


def test_agent_lines_reach_the_log_a_follower_reads(tmp_path: Path):
    """The agents panel was empty for a whole live run while the mechanism worked.

    `_emit` feeds the in-process dashboard; `deps.emit` is what reaches `run.log`.
    The agent stream went only to `_emit`, so `qmine watch` — which reads the file
    — showed nothing. The earlier test checked that `_run_log` and
    `parse_log_line` agree, and never that agent lines get written at all.
    """
    src = Path("src/qmine/runner.py").read_text()
    body = src[src.index("def _agent("):]
    body = body[:body.index("\n    registry.on_call")]
    assert "log.info(" in body, "the agent line never reaches run.log"
    assert "_emit(" in body, "and it must still reach the in-process dashboard"


def test_a_follower_rebuilds_the_agents_panel_from_the_log(tmp_path: Path):
    """End to end: what the runner writes, a watcher must be able to render."""
    import logging as _log

    from qmine.runner import _run_log

    rec = {"role": "annotator_b", "ok": True, "latency_s": 104.6,
           "output_tokens": 19839, "model": "deepseek-v4-flash",
           "returned": "labels=25 · 静夜思的全文 → LOOKUP_POEM_TEXT"}
    line = (f"  ~ {rec['role']} {'ok' if rec['ok'] else '!!'} {rec['latency_s']}s "
            f"out {rec['output_tokens']:,} · {rec['model']} · {rec['returned']}")
    with _run_log(tmp_path):
        _log.getLogger("qmine").info(line)

    follower = LiveDashboard(run_id="live34")
    for raw in (tmp_path / "run.log").read_text().splitlines():
        msg = parse_log_line(raw)
        if msg:
            follower._ingest(msg)
    assert follower.agents, "the follower saw no agent turns"
    role, ok, secs, out, model, ret = follower.agents[0]
    assert (role, ok, out, model) == ("annotator_b", True, "19,839", "deepseek-v4-flash")
    assert "静夜思的全文 → LOOKUP_POEM_TEXT" in ret, "the content sample must survive"
    assert "静夜思的全文" in render(follower)


def test_the_metrics_panel_is_not_empty_during_the_longest_phase():
    """Gold annotation emits 240 lines and not one carries a decimal, so the
    scrape found nothing and the panel sat blank for an hour of a live run."""
    dash = feed(LiveDashboard(), "P2b gold — 3000 queries",
                "  annotator[a] 96/120 batches · 2400 rows")
    dash.usage_fn = lambda: {"calls": 149, "output_tokens": 1_884_783, "errors": 21}
    out = render(dash)
    assert dash.metrics == [], "no decimals in these lines — nothing to scrape"
    assert "149" in out and "1.9M" in out, "vitals must fill the gap"


def test_errors_and_failovers_are_visible_without_reading_the_log():
    """21 errors went unremarked through a whole run; a failover silently changes
    which model produced every number after it."""
    dash = feed(LiveDashboard(), "P2b gold")
    dash.usage_fn = lambda: {"calls": 9, "output_tokens": 100, "errors": 21,
                             "failovers": [{"provider": "deepseek"}]}
    out = render(dash)
    assert "21" in out
    assert "1" in out and ("failover" in out or "切换" in out)


def test_a_rate_is_not_shown_before_it_means_anything():
    dash = feed(LiveDashboard(), "P0 foundation")
    dash.usage_fn = lambda: {"calls": 1, "output_tokens": 1_884_783}
    out = render(dash)
    assert "token/" not in out, "a rate over the first seconds is noise"


def test_the_digest_shows_content_not_just_shape():
    """"labels=25" says work happened; the operator wants to see the work."""
    from qmine.llm.registry import summarize_return

    got = summarize_return({"labels": [{"query": "什么是光合作用",
                                        "label": "EXPLAIN_SCIENCE"}]})
    assert "labels=1" in got and "什么是光合作用" in got and "EXPLAIN_SCIENCE" in got

    # A list of bare ids must not win the sample slot over a readable one.
    got = summarize_return({"nodes": [{"code": "LOOKUP_POEM_TEXT", "name": "查诗词原文"}],
                            "rules": [1, 2, 3]})
    assert "查诗词原文" in got, got


def test_a_config_file_does_not_silently_replace_the_domain_profile():
    """`--config` won outright and dropped `--domain`, so a file holding nothing
    but a provider policy also swapped K12 for the generic profile.

    The symptom was quiet and downstream: template coverage halved from 36.6% to
    19.1% in a DETERMINISTIC phase, because the domain's phrasing seeds had gone
    missing — and fragmentation, intent alignment and the K locator all rest on
    those rows.
    """
    from qmine.cli import _load_config

    cfg = _load_config("configs/live.yaml", "k12_zh")
    assert cfg.domain.key == "k12_zh", "the domain the user asked for must survive"
    assert cfg.domain.template_seeds, "and bring its phrasing seeds with it"
    assert cfg.llm.excluded_labs, "while the config file's own settings still apply"


def test_a_config_that_declares_a_domain_still_wins():
    """Explicit beats inherited — a config file naming its own domain is not
    overridden by the CLI default."""
    import tempfile
    from pathlib import Path as P

    from qmine.cli import _load_config

    with tempfile.TemporaryDirectory() as d:
        f = P(d) / "c.yaml"
        f.write_text("domain:\n  key: ecommerce_en\n  display_name: E-commerce\n")
        cfg = _load_config(str(f), "k12_zh")
    assert cfg.domain.key == "ecommerce_en"


def test_a_tool_loop_turn_is_announced_once():
    """The tool path called `report_call` directly AND `record_external_turn`,
    which reports through `_store` — so every web-researching agent appeared
    twice in the log and twice in the agents panel."""
    import inspect

    from qmine.agents.base import ToolAgent

    # Strip comments — the comment explaining this bug naturally names both calls.
    src = "\n".join(ln for ln in inspect.getsource(ToolAgent.run).splitlines()
                    if not ln.strip().startswith("#"))
    assert src.count("record_external_turn") == 1
    assert "report_call" not in src, "the store path already announces the turn"


def test_a_tool_loop_turn_replays_from_cache(cfg, tmp_path):
    """Written but never read — which made resume impossible.

    The two web-researching agents fetch live pages, so re-running them returns
    different candidates, which changes the architect's prompt, which misses its
    own cache, and so on through every annotation call downstream. Twice today a
    resume that should have replayed ~55 minutes replayed almost none of it.
    """
    from qmine.llm.registry import ModelRegistry

    reg = ModelRegistry(cfg.llm, cache_dir=tmp_path, run_cfg=cfg)
    assert reg.replay_external_turn("researcher_literature", "deep", "sys", "usr") is None

    reg.record_external_turn("researcher_literature", "deep", "sys", "usr",
                             {"candidates": ["a", "b"]}, latency=12.0)
    hit = reg.replay_external_turn("researcher_literature", "deep", "sys", "usr")
    assert hit == {"candidates": ["a", "b"]}
    assert reg.ledger.snapshot()["cache_hits"] == 1

    # A different prompt is a different question and must not replay.
    assert reg.replay_external_turn("researcher_literature", "deep", "sys", "other") is None


def test_the_reader_and_writer_share_one_key_definition(cfg, tmp_path):
    """Two hand-rolled key computations would drift and the cache would go cold
    silently — the failure mode is indistinguishable from 'nothing was cached'."""
    import inspect

    from qmine.llm.registry import ModelRegistry

    for fn in (ModelRegistry.record_external_turn, ModelRegistry.replay_external_turn):
        assert "_external_key(" in inspect.getsource(fn)
