"""Pooling several exports into one corpus, and the limits on what a plan may do.

The preparation step is where a study is most easily invalidated without anyone
noticing: a regex one character too greedy removes a third of a snapshot, and
every table afterwards describes a corpus nobody chose. These pin the bounds the
executor puts on a plan whatever the plan says.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from qmine.prepare import execute, heuristic_plan, profile_input, profile_inputs
from qmine.prepare.agent import diff_plans, validate_plan
from qmine.prepare.execute import PrepError
from qmine.prepare.inspect import is_textual
from qmine.prepare.plan import InputSpec, PrepOp, PrepPlan


def _csv(tmp_path: Path, name: str, rows: list[dict]) -> str:
    p = tmp_path / name
    pd.DataFrame(rows).to_csv(p, index=False)
    return str(p)


def _wrapped(tmp_path: Path, n: int = 200, wrapper: str = "我想咨询") -> str:
    # The character AFTER the wrapper has to vary, or it is genuinely part of
    # the template and the extension rule is right to include it.
    tails = ["头痛", "失眠", "咳嗽", "血压高", "孩子发烧", "胃疼", "皮肤过敏"]
    rows = [{"query": f"{wrapper}{tails[i % len(tails)]}{i}怎么办", "total_pv": 1000 - i,
             "event_day": "20260914"} for i in range(n)]
    rows += [{"query": f"直接问的第{i}个", "total_pv": 5, "event_day": "20260914"}
             for i in range(20)]
    return _csv(tmp_path, "wrapped.csv", rows)


# ---------------------------------------------------------------- inspection

def test_a_text_column_is_found_in_a_pandas_three_string_dtype(tmp_path):
    """pandas 3 reads a text column as `str`, not `object`.

    A `dtype == "object"` test found the query column in every CSV and in none of
    the xlsx exports, so the inspector reported "no column looks like free text"
    on files whose first column is nothing but free text.
    """
    assert is_textual("str") and is_textual("string[pyarrow]") and is_textual("object")
    assert not is_textual("int64") and not is_textual("float64")
    p = _csv(tmp_path, "q.csv", [{"query": f"第{i}个不重复的查询串", "pv": i + 1}
                                 for i in range(60)])
    prof = profile_input(p)
    assert prof.text_candidates and prof.text_candidates[0] == "query"


def test_a_wrapper_prefix_is_reported_at_its_full_length(tmp_path):
    """A scan that stopped at three characters returned `我想咨` for an export
    whose wrapper is `我想咨询`. Stripping three of four leaves a stray `询` on
    every row, and the collision merge that follows then finds nothing to merge.
    """
    prof = profile_input(_wrapped(tmp_path))
    assert prof.repeated_prefixes, "a wrapper on most rows must be reported"
    top = prof.repeated_prefixes[0][0]
    assert top == "我想咨询", f"reported {top!r}, not the full wrapper"


def test_a_head_export_and_a_random_sample_are_told_apart(tmp_path):
    head = _csv(tmp_path, "head.csv",
                [{"q": f"头部{i}", "pv": int(10_000 / (i + 1)) + 1} for i in range(200)])
    rand = _csv(tmp_path, "rand.csv",
                [{"q": f"随机{i}", "pv": 1 + (i * 7) % 5} for i in range(200)])
    a, b = profile_input(head, text_column="q"), profile_input(rand, text_column="q")
    assert a.weight_sorted_desc and "HEAD" in " ".join(a.notes)
    assert not b.weight_sorted_desc


# --------------------------------------------------------------- the planner

def test_the_axis_is_always_stated_as_an_assumption_when_nobody_declared_it(tmp_path):
    """The computation is axis-agnostic; the prose is not, and one caveat
    inverts. Overlap cannot settle it — measured, two random 1w samples of the
    SAME surface a year apart share 0.02% of their strings while two head
    exports of it share 63.3% — so the assumption goes in front of a person."""
    ps = profile_inputs([_csv(tmp_path, "a.csv", [{"query": f"甲{i}", "pv": i + 1}
                                                  for i in range(40)]),
                         _csv(tmp_path, "b.csv", [{"query": f"乙{i}", "pv": i + 2}
                                                  for i in range(40)])])
    inferred = heuristic_plan(ps)
    assert inferred.confidence != "high"
    assert any("对比轴" in c for c in inferred.concerns)
    declared = heuristic_plan(ps, axis="stratum")
    assert declared.comparison_axis == "stratum" and declared.confidence == "high"


def test_the_planner_proposes_stripping_a_wrapper_and_never_dropping_it(tmp_path):
    plan = heuristic_plan(profile_inputs([_wrapped(tmp_path)]), axis="stratum")
    ops = [o.op for o in plan.inputs[0].ops]
    assert "strip_prefix" in ops and "merge_collisions" in ops
    assert "drop_regex" not in ops, "a wrapper is stripped, never used to delete rows"
    assert ops.index("merge_collisions") > ops.index("strip_prefix")


# -------------------------------------------------------------- the executor

def test_an_over_large_removal_is_downgraded_to_a_flag(tmp_path):
    """A rule one character too greedy otherwise removes a third of a snapshot
    silently, and every table afterwards is about a corpus nobody chose.

    Returning an all-False mask is what makes this a REFUSAL and not a warning:
    the rows stay in the mining and the match is still recorded.
    """
    src = _csv(tmp_path, "x.csv", [{"query": f"查询{i}", "pv": 1} for i in range(100)])
    plan = PrepPlan(inputs=[InputSpec(
        path=src, snapshot="s", text_column="query", weight_column="pv",
        ops=[PrepOp(op="drop_regex", name="greedy", pattern=r".",
                    why="deliberately too greedy")])], comparison_axis="stratum")
    rep = execute(plan, tmp_path / "out", max_drop_share=0.25)
    assert rep["n_mined"] == 100, "the refusal must keep every row in the mining"
    ref = rep["inputs"][0]["拒绝执行"]
    assert len(ref) == 1 and ref[0]["share"] == 1.0
    allrows = pd.read_parquet(tmp_path / "out" / "prepared_all_rows.parquet")
    assert allrows["flag_refused_greedy"].all(), "the match is still recorded"


def test_a_removal_below_the_limit_is_obeyed(tmp_path):
    """The guard must not be so broad that no rule can ever remove anything."""
    src = _csv(tmp_path, "x.csv", [{"query": ("广告" if i < 10 else "正常") + f"{i}",
                                    "pv": 1} for i in range(100)])
    plan = PrepPlan(inputs=[InputSpec(
        path=src, snapshot="s", text_column="query", weight_column="pv",
        ops=[PrepOp(op="drop_regex", name="ad", pattern="^广告", why="ads")])],
        comparison_axis="stratum")
    rep = execute(plan, tmp_path / "out", max_drop_share=0.25)
    assert rep["n_mined"] == 90 and not rep["inputs"][0]["拒绝执行"]


def test_stripping_a_wrapper_never_deletes_a_row(tmp_path):
    """The words after the wrapper are the query. A row that is NOTHING but the
    wrapper keeps its original text and falls to the content-free rule instead,
    so no row disappears without a tier saying why."""
    src = _csv(tmp_path, "w.csv",
               [{"query": "我想咨询头痛怎么办", "pv": 5},
                {"query": "我想咨询", "pv": 5}])
    plan = PrepPlan(inputs=[InputSpec(
        path=src, snapshot="s", text_column="query", weight_column="pv",
        ops=[PrepOp(op="strip_prefix", pattern="^我想咨询", why="wrapper")])],
        comparison_axis="stratum")
    rep = execute(plan, tmp_path / "out")
    got = pd.read_parquet(rep["corpus"])
    assert len(got) == 2, "stripping must not lose a row"
    assert set(got["query"]) == {"头痛怎么办", "我想咨询"}
    assert got["wrapper_stripped"].sum() == 1


def test_every_removed_row_ships_with_the_tier_that_removed_it(tmp_path):
    """Cleaning nobody can see is cleaning nobody can check — and the product
    layer these exports carry is precisely what a reader wants to look at."""
    src = _csv(tmp_path, "x.csv", [{"query": "正常查询", "pv": 1},
                                   {"query": "   ", "pv": 1},
                                   {"query": "!!!", "pv": 1}])
    plan = PrepPlan(inputs=[InputSpec(
        path=src, snapshot="s", text_column="query", weight_column="pv",
        ops=[PrepOp(op="trim"), PrepOp(op="drop_empty", name="content_free",
                                       why="no content")])], comparison_axis="stratum")
    rep = execute(plan, tmp_path / "out")
    allrows = pd.read_parquet(tmp_path / "out" / "prepared_all_rows.parquet")
    assert len(allrows) == 3 and rep["n_mined"] == 1
    removed = allrows[~allrows["kept"]]
    assert set(removed["tier"]) == {"content_free"}
    assert rep["removed_by_tier"]


def test_pv_norm_is_computed_after_the_filter_not_before(tmp_path):
    """The denominator is the KEPT traffic of each snapshot, not the exported
    traffic. Normalising first leaves every kept share quietly deflated by
    whatever the cleaning removed."""
    src = _csv(tmp_path, "x.csv", [{"query": "留下的", "pv": 100},
                                   {"query": "!!!", "pv": 900}])
    plan = PrepPlan(inputs=[InputSpec(
        path=src, snapshot="s", text_column="query", weight_column="pv",
        ops=[PrepOp(op="drop_empty", name="content_free", why="no content")])],
        comparison_axis="stratum")
    rep = execute(plan, tmp_path / "out")
    got = pd.read_parquet(rep["corpus"])
    assert got["pv_norm"].sum() == pytest.approx(10_000.0)


def test_two_inputs_with_the_same_snapshot_tag_are_refused(tmp_path):
    """The comparison could not tell them apart afterwards."""
    a = _csv(tmp_path, "a.csv", [{"query": "甲", "pv": 1}])
    b = _csv(tmp_path, "b.csv", [{"query": "乙", "pv": 1}])
    plan = PrepPlan(comparison_axis="stratum", inputs=[
        InputSpec(path=a, snapshot="same", text_column="query"),
        InputSpec(path=b, snapshot="same", text_column="query")])
    with pytest.raises(PrepError, match="same snapshot tag"):
        execute(plan, tmp_path / "out")


def test_a_partly_grouped_plan_is_refused(tmp_path):
    a = _csv(tmp_path, "a.csv", [{"query": "甲", "pv": 1}])
    b = _csv(tmp_path, "b.csv", [{"query": "乙", "pv": 1}])
    plan = PrepPlan(comparison_axis="stratum", inputs=[
        InputSpec(path=a, snapshot="x", group="S", text_column="query"),
        InputSpec(path=b, snapshot="y", text_column="query")])
    with pytest.raises(PrepError, match="group every input or none"):
        execute(plan, tmp_path / "out")


def test_a_plan_that_removes_everything_is_refused(tmp_path):
    """An empty corpus is not a corpus, and the message must name the step."""
    src = _csv(tmp_path, "x.csv", [{"query": "!!!", "pv": 1} for _ in range(5)])
    plan = PrepPlan(inputs=[InputSpec(
        path=src, snapshot="s", text_column="query", weight_column="pv",
        ops=[PrepOp(op="drop_empty", name="content_free", why="no content")])],
        comparison_axis="stratum")
    with pytest.raises(PrepError, match="every row was removed"):
        execute(plan, tmp_path / "out")


def test_a_rebuilt_corpus_must_reproduce_the_snapshots_it_already_delivered(tmp_path):
    """Adding a snapshot to a corpus that has already been mined and reported on
    is the commonest way to invalidate a delivered study without noticing."""
    src = _csv(tmp_path, "x.csv", [{"query": f"查询{i}", "pv": i + 1} for i in range(20)])
    spec = InputSpec(path=src, snapshot="s", text_column="query", weight_column="pv",
                     ops=[PrepOp(op="trim")])
    first = execute(PrepPlan(inputs=[spec], comparison_axis="stratum"), tmp_path / "a")
    again = execute(PrepPlan(inputs=[spec], comparison_axis="stratum"), tmp_path / "b",
                    previous=first["corpus"])
    assert again["superset_check"]["all_reproduced"] is True

    changed = spec.model_copy(update={"ops": [
        PrepOp(op="trim"),
        PrepOp(op="drop_regex", name="cut", pattern="查询1$", why="deliberate change")]})
    third = execute(PrepPlan(inputs=[changed], comparison_axis="stratum"),
                    tmp_path / "c", previous=first["corpus"])
    assert third["superset_check"]["all_reproduced"] is False


# ------------------------------------------------------------ plan validation

def test_a_plan_that_names_an_input_nobody_offered_is_rejected():
    plan = PrepPlan(inputs=[InputSpec(path="/not/offered.csv", snapshot="s",
                                      text_column="query")])
    assert any("offered" in p for p in validate_plan(plan, {"/real/a.csv"}))


def test_a_removal_with_no_stated_reason_is_rejected():
    """A removal whose reason is not written down cannot be reviewed, and an
    unreviewable removal is the one that ships."""
    plan = PrepPlan(inputs=[InputSpec(
        path="a.csv", snapshot="s", text_column="q",
        ops=[PrepOp(op="drop_regex", name="x", pattern="foo", why="")])])
    assert any("no stated reason" in p for p in validate_plan(plan, {"a.csv"}))


def test_a_pattern_that_does_not_compile_names_itself():
    plan = PrepPlan(inputs=[InputSpec(
        path="a.csv", snapshot="s", text_column="q",
        ops=[PrepOp(op="drop_regex", name="x", pattern="(", why="why")])])
    assert any("does not compile" in p for p in validate_plan(plan, {"a.csv"}))


def test_the_difference_between_two_plans_is_reportable():
    """A reviewer reads the DELTA against the mechanical plan, not the plan."""
    base = PrepPlan(inputs=[InputSpec(path="a.csv", snapshot="s", text_column="q",
                                      ops=[PrepOp(op="trim")])])
    other = PrepPlan(comparison_axis="stratum", inputs=[
        InputSpec(path="a.csv", snapshot="t", text_column="q2",
                  ops=[PrepOp(op="trim"), PrepOp(op="drop_empty", why="x")])])
    d = " ".join(diff_plans(base, other))
    assert "axis" in d and "text column" in d and "snapshot" in d and "ops" in d


def test_the_report_records_every_step_and_every_refusal(tmp_path):
    src = _csv(tmp_path, "x.csv", [{"query": f"查询{i}", "pv": 1} for i in range(50)])
    plan = PrepPlan(inputs=[InputSpec(
        path=src, snapshot="s", text_column="query", weight_column="pv",
        ops=[PrepOp(op="trim"), PrepOp(op="drop_empty", name="cf", why="x")])],
        comparison_axis="stratum", rationale="because", concerns=["one thing"])
    execute(plan, tmp_path / "out")
    rep = json.loads((tmp_path / "out" / "prep_report.json").read_text(encoding="utf-8"))
    assert rep["rationale"] == "because" and rep["concerns"] == ["one thing"]
    steps = rep["inputs"][0]["各步"]
    assert [s["op"] for s in steps] == ["trim", "drop_empty", "(总计)"]


def test_a_head_cut_over_a_mostly_empty_weight_column_is_refused(tmp_path):
    """`head_cut` is exempt from the over-large-removal guard because removing a
    lot IS its purpose — which is exactly why it needs its own floor.

    A weight column populated for 4 of 60 rows ranks those four and tiers away
    the other 93%, with no refusal and nothing in the report to notice.
    """
    src = _csv(tmp_path, "h.csv", [{"query": f"q{i}", "pv": 10 - i if i < 4 else None}
                                   for i in range(60)])
    plan = PrepPlan(inputs=[InputSpec(
        path=src, snapshot="s", text_column="query", weight_column="pv",
        ops=[PrepOp(op="head_cut", n=20, name="below_head", why="take the head")])],
        comparison_axis="stratum")
    rep = execute(plan, tmp_path / "out")
    assert rep["n_mined"] == 60, "a refused head cut must remove nothing"
    ref = rep["inputs"][0]["拒绝执行"]
    assert len(ref) == 1 and "populated for only" in ref[0]["reason"]


def test_aggregation_does_not_invent_a_weight_of_zero(tmp_path):
    """pandas sums an all-NaN group to 0.0.

    A weight that does not exist must stay absent: 0.0 reads as "no traffic",
    and it disarmed `head_cut`'s "no usable weight" refusal, which then tiered
    away everything below rank n on a column of zeros.
    """
    src = _csv(tmp_path, "g.csv", [{"query": q} for q in ("a", "a", "b")])
    spec = InputSpec(path=src, snapshot="s", text_column="query", weight_column=None,
                     ops=[PrepOp(op="aggregate_by_query", why="per-day export"),
                          PrepOp(op="head_cut", n=1, name="hc", why="head")])
    rep = execute(PrepPlan(inputs=[spec], comparison_axis="stratum"), tmp_path / "out")
    got = pd.read_parquet(rep["corpus"])
    assert got["pv_raw"].isna().all(), "an absent weight must stay absent"
    assert rep["n_mined"] == 2, "the head cut must have been refused, not applied"


def test_merging_rows_is_accounted_for_rather_than_looking_like_nothing_happened(tmp_path):
    """`merge_collisions` and `aggregate_by_query` change the row count BY DESIGN
    — text and weight are both preserved — so the all-rows table legitimately
    holds fewer rows than the file. Reporting only a removal percentage against
    the post-merge count said "0.0% removed" for a file that went from 3 rows to
    2, which is true and unreadable.
    """
    src = _csv(tmp_path, "m.csv", [{"query": "我想咨询头痛", "pv": 5},
                                   {"query": "我想咨询头痛", "pv": 7},
                                   {"query": "别的问题", "pv": 1}])
    plan = PrepPlan(inputs=[InputSpec(
        path=src, snapshot="s", text_column="query", weight_column="pv",
        ops=[PrepOp(op="strip_prefix", pattern="^我想咨询", why="wrapper"),
             PrepOp(op="merge_collisions", why="stripped collisions")])],
        comparison_axis="stratum")
    rep = execute(plan, tmp_path / "out")
    i = rep["inputs"][0]
    assert i["原始行数"] == 3 and i["处理后行数"] == 2 and i["合并掉的行"] == 1
    assert i["占原始行数%"] == pytest.approx(66.67, abs=0.01)
    allrows = pd.read_parquet(tmp_path / "out" / "prepared_all_rows.parquet")
    assert int(allrows["n_rows_raw"].sum()) == 3, "every input row must be accounted for"
    got = pd.read_parquet(rep["corpus"])
    assert float(got.loc[got["query"] == "头痛", "pv_raw"].iloc[0]) == 12.0, \
        "the merged weight must be the sum, not one of the two"


def test_a_wrapper_is_found_even_when_the_rows_are_short(tmp_path):
    """The extension used to be blocked by the very filter that kept a prefix
    from swallowing the query.

    Counting only strings longer than `k + 2` meant nothing was counted at
    `k = 4` on six-character rows, so `我想咨询头痛` reported its wrapper as
    `我想咨` — one character short, leaving a stray `询` on every row. Growing
    the prefix while it still leaves two characters on the SHORTEST matching row
    does both jobs.
    """
    from qmine.prepare.inspect import _prefixes

    short = pd.Series(["我想咨询头痛"] * 50)
    assert _prefixes(short, 50)[0][0] == "我想咨询", "the wrapper was truncated"
    # and it must not grow into the query itself
    assert len(_prefixes(short, 50)[0][0]) <= 4


def test_a_date_column_is_not_mistaken_for_traffic(tmp_path):
    """`event_day` is numeric, non-constant and often the only other numeric
    column, so it was picked as the weight and `pv_raw` became a sum of dates —
    silently, because the number is plausible."""
    dated = _csv(tmp_path, "d.csv", [{"query": f"查询{i}", "event_day": 20250701 + i % 3}
                                     for i in range(50)])
    assert profile_input(dated).weight_candidates == []
    both = _csv(tmp_path, "e.csv", [{"query": f"查询{i}", "event_day": 20250701,
                                     "wise_pv": 100 - i} for i in range(50)])
    assert profile_input(both).weight_candidates == ["wise_pv"]


def test_a_blank_cell_does_not_enter_the_corpus_as_the_string_nan(tmp_path):
    """`str(float("nan"))` is the three characters `nan`, every one of them
    alphanumeric — so a blank cell read as content, survived `drop_empty`, and
    entered the mining as the literal string "nan"."""
    from qmine.prepare.inspect import has_content

    assert not has_content(float("nan")) and not has_content(None)
    assert not has_content("nan") and has_content("头痛")
    src = _csv(tmp_path, "b.csv", [{"query": "正常查询", "pv": 1},
                                   {"query": None, "pv": 1}])
    plan = PrepPlan(inputs=[InputSpec(
        path=src, snapshot="s", text_column="query", weight_column="pv",
        ops=[PrepOp(op="trim"), PrepOp(op="drop_empty", name="cf", why="no content")])],
        comparison_axis="stratum")
    rep = execute(plan, tmp_path / "out")
    assert rep["n_mined"] == 1
    assert "nan" not in set(pd.read_parquet(rep["corpus"])["query"])


def test_a_missing_weight_is_not_given_a_typical_rows_traffic(tmp_path):
    """`fillna(median)` promotes every blank into the middle of the distribution
    and inflates whatever class the blanks land in."""
    src = _csv(tmp_path, "w.csv", [{"query": "大", "pv": 1000},
                                   {"query": "中", "pv": 10},
                                   {"query": "小", "pv": 1},
                                   {"query": "缺", "pv": None}])
    plan = PrepPlan(inputs=[InputSpec(
        path=src, snapshot="s", text_column="query", weight_column="pv")],
        comparison_axis="stratum")
    rep = execute(plan, tmp_path / "out")
    got = pd.read_parquet(rep["corpus"]).set_index("query")
    assert got.loc["缺", "pv_norm"] == pytest.approx(got.loc["小", "pv_norm"]), \
        "an unknown weight must take the smallest observed one, not the median"
    assert got.loc["缺", "pv_norm"] < got.loc["中", "pv_norm"]


def test_an_input_the_planner_cannot_read_stops_the_run(tmp_path):
    """A dropped input leaves the run comparing fewer snapshots than the person
    asked for, and the concern scrolls past."""
    ok = _csv(tmp_path, "a.csv", [{"query": f"查询{i}", "pv": i + 1} for i in range(40)])
    bad = _csv(tmp_path, "b.csv", [{"n": i, "m": i * 2} for i in range(40)])
    with pytest.raises(ValueError, match="free-text"):
        heuristic_plan(profile_inputs([ok, bad]), axis="stratum")


def test_an_out_of_order_plan_is_rejected():
    """The executor's order is load-bearing and it trusted the plan to get it
    right: a plan that reads a weight before aggregating a per-day export, or
    tiers before cutting a head, produces a corpus that looks fine."""
    plan = PrepPlan(inputs=[InputSpec(
        path="a.csv", snapshot="s", text_column="q",
        ops=[PrepOp(op="head_cut", n=10, why="head"),
             PrepOp(op="aggregate_by_query", why="per-day")])])
    assert any("aggregate_by_query" in p for p in validate_plan(plan, {"a.csv"}))
    good = PrepPlan(inputs=[InputSpec(
        path="a.csv", snapshot="s", text_column="q",
        ops=[PrepOp(op="aggregate_by_query", why="per-day"),
             PrepOp(op="head_cut", n=10, why="head")])])
    assert not [p for p in validate_plan(good, {"a.csv"}) if "come before" in p]
