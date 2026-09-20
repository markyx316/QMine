"""The cross-snapshot comparison, and the defects each guard was written against.

Every test here names a failure that actually happened — in the hand-built study
this package generalises, or in this package while it was being written.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qmine.pooled import classes as C
from qmine.pooled import guards, report, verify
from qmine.pooled.frame import PooledJoinError, load_frame
from qmine.pooled.manifest import SnapshotManifest, build_manifest, default_contrasts
from qmine.pooled.stats import absence_verdict, one_sided_upper, tvd_with_bounds
from qmine.pooled.tables import build_tables


def _frame(n_a: int = 600, n_b: int = 400, seed: int = 0) -> tuple[pd.DataFrame, SnapshotManifest]:
    rng = np.random.default_rng(seed)
    n = n_a + n_b
    df = pd.DataFrame({
        "query": [f"查询{i}的内容" for i in range(n)],
        "snapshot": ["A"] * n_a + ["B"] * n_b,
        "pv": rng.integers(1, 400, n),
        "td_l1": rng.choice(["X", "Y", "Z"], n, p=[.5, .3, .2]),
        "td_l2": rng.choice(["x1", "x2"], n),
        "bu_leaf": rng.integers(0, 4, n),
        "bu_family_final": rng.integers(0, 2, n),
        "td_confidence": rng.random(n),
        "td_ambiguous": rng.random(n) < .1,
        "bu_ambiguous": rng.random(n) < .2,
    })
    man, pv = build_manifest(df, weight_col="pv", axis="stratum")
    df["pv_norm"] = pv
    return df, man


# ------------------------------------------------------------------- the join

def test_the_comparison_refuses_a_join_it_cannot_verify(tmp_path):
    """A text join would be WRONG, not merely risky.

    The same string occurs in several snapshots — that overlap is part of what
    the study measures — so joining on query text binds a row to the wrong
    snapshot. The join is positional, which means it is only valid while the two
    files are in the same order, so both the length and every row's text are
    checked and the analysis refuses rather than producing a plausible table.
    """
    gen = tmp_path / "run" / "gen01"
    gen.mkdir(parents=True)
    pd.DataFrame({"query": ["a", "b", "c"], "snapshot": ["s", "s", "t"],
                  "weight": [1, 2, 3]}).to_parquet(gen / "corpus.parquet")
    pd.DataFrame({"query": ["a", "b"], "td_l1": ["A", "B"]}).to_csv(
        gen / "labels_full.csv", index=False)
    from qmine.pooled.source import DirSource

    with pytest.raises(PooledJoinError, match="positional"):
        load_frame(DirSource(gen))

    pd.DataFrame({"query": ["a", "b", "DIFFERENT"], "td_l1": ["A", "B", "C"]}).to_csv(
        gen / "labels_full.csv", index=False)
    with pytest.raises(PooledJoinError, match="differ in query text"):
        load_frame(DirSource(gen))


def test_a_single_snapshot_run_is_told_so_rather_than_compared(tmp_path):
    """Comparing one snapshot to itself is not an error message anyone can act on."""
    gen = tmp_path / "run" / "gen01"
    gen.mkdir(parents=True)
    pd.DataFrame({"query": ["a", "b"], "weight": [1, 2]}).to_parquet(gen / "corpus.parquet")
    pd.DataFrame({"query": ["a", "b"], "td_l1": ["A", "B"]}).to_csv(
        gen / "labels_full.csv", index=False)
    from qmine.pooled.source import DirSource

    with pytest.raises(PooledJoinError, match="single input"):
        load_frame(DirSource(gen))


# ------------------------------------------------------------------- absence

def test_an_absence_is_shipped_with_its_detectability_not_as_a_zero():
    """Zero of 950 rows is consistent with a share up to 0.39%; zero of 10,000
    with 0.037%. A report that prints the zero and not the bound has stated a
    sample size as if it were a finding."""
    assert one_sided_upper(950) > 10 * one_sided_upper(10_000)
    # The verdict, not the zero, is what prose may quote.
    assert absence_verdict(expected=7.0, p_zero=0.001) == "真缺席"
    assert absence_verdict(expected=1.0, p_zero=0.4).startswith("不可判定")
    assert absence_verdict(expected=4.0, p_zero=0.02) == "偏少但证据弱"


def test_the_absence_table_names_the_expected_count_and_probability():
    df, man = _frame()
    df.loc[df["snapshot"] == "B", "td_l1"] = "X"        # Y and Z vanish from B
    d = C.class_frame(df, "td_l1", {})
    ab = C.absence(d, man)
    assert set(ab["缺席快照"]) == {"B"}
    assert {"期望条数", "P(0)", "该快照上界%", "判定"} <= set(ab.columns)
    assert (ab["期望条数"] > 0).all()


# ---------------------------------------------------------------- noise floor

def test_a_difference_within_the_noise_ceiling_is_not_reported_as_one():
    """Two samples from ONE distribution routinely give a non-zero TVD.

    Reporting the point estimate alone turns resampling noise into a finding —
    which is what two independent reviewers had to compute by hand before the
    ceiling column existed, and it overturned several claims.
    """
    rng = np.random.default_rng(3)
    pool = rng.integers(0, 5, 2000)
    out = tvd_with_bounds(pool[:1000], pool[1000:], 5, tag=("t",), n_boot=120, n_null=80)
    assert out["tvd"] > 0, "two finite samples are never identical"
    assert not out["exceeds_noise"], "same-distribution halves must not read as a difference"

    a = np.zeros(1000, dtype=int)
    b = np.ones(1000, dtype=int)
    far = tvd_with_bounds(a, b, 5, tag=("t2",), n_boot=120, n_null=80)
    assert far["exceeds_noise"], "genuinely disjoint distributions must clear the ceiling"


def test_the_noise_ceiling_is_drawn_at_the_observed_sample_sizes():
    """It used to split each side in HALF and compare n/2 against n/2.

    TVD noise scales as 1/sqrt(n), so halving n inflates the ceiling — measured
    at 1.41x for 1,000 vs 1,000, 1.44x for 10,000 vs 10,316 and **1.80x for 400
    vs 4,000**, worst exactly where the snapshots are most unbalanced. The error
    hides real differences rather than inventing them, which is why it survived;
    `conditional_mix` in this same package already pooled correctly, so two
    computations answered different questions under one column name.
    """
    rng = np.random.default_rng(0)
    na, nb = 400, 4000
    pool = rng.integers(0, 20, na + nb)
    out = tvd_with_bounds(pool[:na], pool[na:], 20, tag=("t", na, nb),
                          n_boot=150, n_null=300)
    # The correct null, computed here independently of the implementation.
    ref = []
    for t in range(300):
        idx = np.random.default_rng(9000 + t).permutation(len(pool))
        p, q = pool[idx[:na]], pool[idx[na:]]
        ref.append(0.5 * np.abs(np.bincount(p, minlength=20) / na
                                - np.bincount(q, minlength=20) / nb).sum())
    assert out["noise_ceiling"] == pytest.approx(float(np.percentile(ref, 95)), rel=0.12), \
        "the ceiling is not the 95th percentile of a size-matched null"
    assert not out["exceeds_noise"], "one distribution split in two is not a difference"

    a = np.concatenate([np.zeros(900, int), np.ones(100, int)])
    b = np.concatenate([np.zeros(700, int), np.ones(300, int)])
    far = tvd_with_bounds(a, b, 20, tag=("t2",), n_boot=150, n_null=300)
    assert far["exceeds_noise"], "a genuine difference must still clear the ceiling"


def test_a_class_only_counts_as_characteristic_when_every_pair_agrees():
    """Raw exclusivity is almost always empty and says nothing about a snapshot.

    The standard here gets HARDER as snapshots are added, which is the property
    that makes it worth reporting.
    """
    df, man = _frame()
    sig = C.signature(C.class_frame(df, "td_l1", {}), man)
    assert sig.empty, "one distribution split in two must yield no characteristic class"


# ------------------------------------------------------------------- grouping

def test_a_partial_group_assignment_produces_no_group_tables():
    """A partial assignment silently drops the unassigned snapshots out of the
    group tables while their rows still count toward the n that is reported —
    the defect that printed 19,997 for a side that had 39,996."""
    man = SnapshotManifest(snapshots=["a", "b", "c"], sizes={"a": 1, "b": 1, "c": 1},
                           surfaces={"a": "S", "b": "T"})
    assert not man.has_surfaces
    man.surfaces["c"] = "T"
    assert man.has_surfaces
    man.surfaces = {"a": "S", "b": "S", "c": "S"}
    assert not man.has_surfaces, "one group is not a split"


def test_the_stratum_axis_does_not_invent_an_order():
    """The computation is axis-agnostic; the prose is not, and one caveat inverts."""
    t = default_contrasts(["p", "q", "r"], "time")
    s = default_contrasts(["p", "q", "r"], "stratum")
    assert [(c.a, c.b) for c in t][:2] == [("p", "q"), ("q", "r")]
    assert len(s) == 3 and all("分层" in c.differs for c in s)


# --------------------------------------------------------------- quote guards

def test_the_quote_guard_compiles_declared_patterns_separately():
    """Joining them into one alternation breaks on an inline global flag.

    Domain profiles legitimately write `(?i)…`, which is only valid at the start
    of an expression. Wrapped as the second branch of an alternation it raises
    `global flags not at the start of the expression`, the whole layer fails to
    compile, and the guard fails OPEN — the worst outcome available to it.
    """
    pats = guards.compile_extra([r"(?i)abc", r"定投", r"(?s)x.y"])
    assert isinstance(pats, list) and len(pats) == 3
    q = pd.Series(["ABC here", "基金定投", "nothing"])
    assert guards._mask(pats, q).tolist() == [True, True, False]
    with pytest.raises(ValueError, match="does not compile"):
        guards.compile_extra(["("])


def test_the_hard_rules_do_not_read_an_adult_age_as_a_minor():
    """`47岁` contains `7岁`. A looser age pattern reads every 47-year-old's
    medical question as a minors-and-sex hit, and a guard that fires on ordinary
    queries gets switched off."""
    g = guards.QuoteGuard()
    d = pd.DataFrame({"query": ["47岁女性胸部胀痛怎么办", "14岁女孩内衣怎么选"],
                      "td_l1": ["A", "A"], "snapshot": ["s", "s"]})
    hits = g.hard_rule_scan(d, printed=set())
    assert list(hits["query"]) == ["14岁女孩内衣怎么选"]


def test_a_risk_flagged_class_is_only_never_quoted_when_quoting_is_the_harm():
    """Blocking every risk-flagged class costs 60%-90% of quotable rows, because
    most risk flags mean "answering this badly is the hazard" rather than "this
    sentence may not be repeated"."""
    tax = {"nodes": [
        {"code": "SEXY", "level": 1, "risk": True, "name": "生成露骨性图像编辑"},
        {"code": "DOSE", "level": 1, "risk": True, "name": "用药剂量咨询",
         "definition": "用户问具体吃多少"},
    ]}
    never = guards.never_quote_classes(tax)
    assert never == {"SEXY"}


def test_a_single_character_never_puts_a_whole_class_out_of_quotation():
    """A bare `裸` matched `裸名称` — "bare name" — inside the definition of
    查询物品或活动的功效与作用, and put **28.6% of a medical corpus** (12,602 rows)
    out of quotation. Those rows' hazard is how a system ANSWERS; quoting
    阿司匹林的功效 harms nobody, and a layer that expensive stops being used.

    Every entry in the lexicon is a compound for this reason.
    """
    innocent = {"nodes": [{
        "code": "EFFICACY", "level": 1, "risk": True,
        "name": "查询物品或活动的功效与作用",
        "definition": "用户给出一个药品、食物或保健品的名称（常为裸名称），"
                      "想知道它有什么功效、主治什么、有何好处或副作用。"}]}
    assert guards.never_quote_classes(innocent) == set(), \
        "a class was silenced by one character inside an ordinary compound"

    real = {"nodes": [
        {"code": "SEXIMG", "level": 1, "risk": True, "name": "生成露骨性图像编辑"},
        {"code": "CRISIS", "level": 1, "risk": True, "name": "自伤或主动致病相关危机"},
        {"code": "PII", "level": 1, "risk": True, "name": "查询他人的个人信息"},
    ]}
    assert guards.never_quote_classes(real) == {"SEXIMG", "CRISIS", "PII"}


def test_a_class_with_no_quotable_row_ships_a_count_not_a_blank():
    """A guard that silently deletes a section is the grounding-false-positive
    failure mode: live42 delivered 3 of 9 sections, 3 lost to a false rejection."""
    df, man = _frame(60, 40)
    d = C.class_frame(df, "td_l1", {})
    ex = C.examples(d, man, pd.Series(False, index=d.index))
    assert len(ex), "a fully-guarded corpus must still produce example ROWS"
    assert (ex["取法"].str.startswith("不引原文")).all()
    assert (ex["该类该快照条数"] > 0).all()


# ----------------------------------------------------------------- rendering

def _table_cell(text: str, heading_contains: str, row_name: str, column: str) -> str:
    """The cell a READER sees, pulled out of the rendered markdown by column name."""
    # The generated table of contents repeats every heading, so search for the
    # HEADING LINE, not the first occurrence of the text.
    start = text.index("### " + heading_contains)
    lines: list[str] = []
    for ln in text[start:].splitlines()[1:]:
        if ln.startswith("|"):
            lines.append(ln)
        elif lines:
            break
    header = [c.strip() for c in lines[0].strip("|").split("|")]
    col = header.index(column)
    for ln in lines[2:]:
        cells = [c.strip() for c in ln.strip("|").split("|")]
        if cells and cells[0] == row_name:
            return cells[col]
    raise AssertionError(f"{row_name!r} not in the table under {heading_contains!r}")


def test_the_same_quantity_is_printed_identically_in_the_table_and_the_card(tmp_path):
    """The table did `.round(2)` and the card did f"{v:.2f}".

    Those disagree on every value the renderer prints with `%g`: the SAME range
    reaches the reader as `1.8` in a table and `1.80` in a card three screens
    later, and nothing on the page says which is the number. The comparison has
    to be between the two RENDERED cells — comparing a card against the
    underlying float passes while the document still contradicts itself, which
    is how this test first shipped unable to fail.
    """
    df, man = _frame(800, 400, seed=11)
    built = build_tables(df, man, {}, guards.QuoteGuard(), tmp_path / "t",
                         n_boot=20, n_null=10, tag="x")
    text = report.build_report(built["frames"], built["summary"], man,
                               title="T", run_id="r", generation="gen01")
    checked = 0
    for name in built["frames"]["matrix_td_l1"]["类目"].astype(str):
        card = [seg for seg in text.split("#### ") if seg.startswith(name + "　")]
        if not card:
            continue
        from_card = card[0].split("极差 ")[1].split("pp")[0]
        from_table = _table_cell(text, "表 自上而下 L1 意图-A", name, "极差pp")
        assert from_card == from_table, (
            f"{name}: the card prints {from_card!r} and the table prints "
            f"{from_table!r} for the same quantity")
        checked += 1
    assert checked >= 2, "the test must actually have compared something"


def test_an_empty_cell_never_reaches_the_reader_as_the_word_nan(tmp_path):
    """A CSV round-trip turns an empty cell into the float nan, and `str(nan)`
    is the four characters `nan` — which shipped as `缺席于 nan`."""
    df, man = _frame(500, 500, seed=5)
    built = build_tables(df, man, {}, guards.QuoteGuard(), tmp_path / "t",
                         n_boot=20, n_null=10, tag="x")
    frames = {k: pd.read_csv(p) for k, p in
              ((f.stem, f) for f in (tmp_path / "t").glob("*.csv"))}
    text = report.build_report(frames, built["summary"], man, title="T",
                               run_id="r", generation="gen01")
    assert " nan" not in text and "nan）" not in text


def test_the_top_ten_membership_has_exactly_one_ordering_authority(tmp_path):
    """The renderer used to re-sort the matrix instead of reading the members
    table. On a tie at rank ten the ten classes displayed were not the ten summed
    in the traffic row — measured 5.25pp apart."""
    src = Path("src/qmine/pooled/report.py").read_text(encoding="utf-8")
    body = src[src.index("def _topn_table"):src.index("def _cards")]
    code = "\n".join(ln for ln in body.splitlines() if not ln.strip().startswith("#"))
    assert "sort_values" not in code, "the renderer must not order the top ten itself"
    assert "topn_members" in src or "mm[" in code


# ------------------------------------------------------------------- verifier

def test_the_verifier_rejects_a_fabricated_number_and_accepts_a_real_one(tmp_path):
    d = tmp_path / "tables"
    d.mkdir()
    pd.DataFrame({"TVD": [0.1473], "n": [20316]}).to_csv(d / "t.csv", index=False)
    (d / "summary.json").write_text(json.dumps({"quotable_rows": 21763}),
                                    encoding="utf-8")
    text = ("<!--NARR:导读-->TVD 是 0.1473，共 20316 行，可引 21763 行；"
            "编的数 99.999。真例「a」，编的例「zzz」。<!--/NARR:导读-->")
    out = verify.check(text, d, corpus={"a"}, quotable={"a"})
    assert out["未匹配数字"] == 1
    assert out["blocks"][0]["未匹配数字"] == ["99.999"]
    assert out["blocks"][0]["语料里不存在"] == ["zzz"]


def test_the_number_pool_refuses_to_be_built_from_an_empty_directory(tmp_path):
    """An empty input set is a SKIP, never a PASS.

    A pool built from a directory that did not exist accepts nothing, so every
    number in the prose reads as unmatched — and a reviewer told that all 200 of
    their figures are wrong stops reading the check.
    """
    with pytest.raises(AssertionError, match="no CSV"):
        verify.number_pool(tmp_path)


def test_a_real_number_is_accepted_in_every_spelling_a_person_writes():
    """The pool once omitted summary.json, so the CORRECT 21763 was flagged while
    a wrong 21,800 coincidentally matched another table. A mechanical check that
    rejects the truth and passes the error is worse than no check."""
    f = verify.forms(0.1473)
    assert {"0.1473", "0.15", "14.73"} <= f
    assert "21,763" in verify.forms(21763.0)


def test_a_named_clinician_is_never_quotable_and_a_speciality_always_is():
    """A named doctor is a private individual's identity, and the rule must not
    depend on which config a `compare` happened to load.

    A run's resolved config FREEZES its domain profile, so a profile-only rule
    would protect future runs and not a re-comparison of an existing one — and
    25 of 26 such rows in med-pool8 were quotable, kept out of the report only
    by the traffic ranking. Measured cost of making it universal: at most 0.11%
    of rows across eight delivered corpora, zero on five.

    A speciality names a kind of doctor, not a person, and must stay quotable or
    every 儿科医生怎么挂号 card goes blank.
    """
    g = guards.QuoteGuard()
    named = ["崔勇医生", "宋阳医生", "新晃县中医院妇科主任唐医生", "胸外科刁明强医生",
             "协和医院主任医师张三医生"]
    generic = ["儿科医生怎么挂号", "急诊科医生", "产科医生", "骨科大夫", "年轻医生",
               "善良的医生", "什么是医生", "医生说要复查"]
    d = pd.DataFrame({"query": named + generic, "td_l1": ["X"] * len(named + generic)})
    ok = g.quotable(d).tolist()
    assert not any(ok[:len(named)]), \
        f"a named clinician stayed quotable: {[q for q, o in zip(named, ok) if o]}"
    assert all(ok[len(named):]), \
        f"a speciality or title was blocked: {[q for q, o in zip(generic, ok[len(named):]) if not o]}"


def test_the_verifier_sees_a_long_quote_and_a_nested_one(tmp_path):
    """`「([^」]{1,80})」` ignored every quotation over eighty characters and
    every one written `『…』` — which is exactly the form the renderer switches
    to for a query that itself contains 「」. The fabricated examples most likely
    to be long or odd were the ones the check skipped.
    """
    d = tmp_path / "t"
    d.mkdir()
    pd.DataFrame({"x": [1]}).to_csv(d / "t.csv", index=False)
    long_fake = "假" * 120
    text = (f"<!--NARR:导读-->短的「真串」，长的「{long_fake}」，"
            "嵌套的『带「」的假串』。<!--/NARR:导读-->")
    out = verify.check(text, d, corpus={"真串"}, quotable={"真串"})
    missing = out["blocks"][0]["语料里不存在"]
    assert long_fake in missing, "a long fabricated quotation was skipped"
    assert "带「」的假串" in missing, "a 『』 quotation was skipped"


def test_a_truncated_quote_still_counts_as_printed():
    """A card that showed 26 characters of a row printed that row.

    The hard-rule scan searches a document for corpus strings, and a prefix
    matches none of them — so a blocked string shown in truncated form was
    reported as "0 printed", which is the one number that layer exists to give.
    """
    corpus = {"这是一条很长的真实查询用来测试前缀匹配是否生效"}
    got = guards.printed_strings([], corpus=corpus)
    assert got == set()
    import tempfile

    p = Path(tempfile.mkdtemp()) / "r.md"
    p.write_text("例子：「这是一条很长的真实查询用来测试」", encoding="utf-8")
    assert guards.printed_strings([p], corpus=corpus) == corpus


def test_two_snapshots_may_not_display_as_the_same_name():
    """Every table keys its columns by the display name, so two snapshots given
    the same one collapse into a single column — the second write wins and the
    counts become whichever snapshot was processed last."""
    df = pd.DataFrame({"snapshot": ["a"] * 3 + ["b"] * 3, "pv": [1, 2, 3, 4, 5, 6],
                       "query": list("abcdef")})
    with pytest.raises(ValueError, match="same name"):
        build_manifest(df, weight_col="pv", labels={"a": "搜索", "b": "搜索"})
    man, _ = build_manifest(df, weight_col="pv", labels={"a": "搜索", "b": "助手"})
    assert man.displays() == ["搜索", "助手"]


def test_a_run_with_only_one_route_still_gets_its_comparison():
    """The cross-route correspondence is a measurement, not a definition. Reading
    the other route's column unconditionally cost a single-route run the entire
    comparison rather than one column of it."""
    df, man = _frame(200, 200)
    del df["bu_leaf"]
    d = C.class_frame(df, "td_l1", {})
    t = C.enrich(C.matrix(d, man), d, "td_l1", {}, pd.Series(True, index=d.index))
    assert len(t) and "主导叶" not in t.columns


def test_a_class_card_quotes_the_whole_row_not_a_fragment(tmp_path):
    """The card used to print 26 characters of a row inside 「」.

    A truncated quote is not a row anyone can look up: the hard-rule scan
    searches the document for corpus strings and a prefix matches none of them,
    so a blocked string printed in truncated form was reported as "0 printed" —
    the one number that layer exists to give.
    """
    long_q = "这是一条特别长的真实查询用来检验卡片会不会把它截断掉再加一些字"
    assert len(long_q) > 26
    n = 120
    df = pd.DataFrame({
        "query": [long_q] + [f"别的查询{i}" for i in range(n - 1)],
        "snapshot": ["A"] * (n // 2) + ["B"] * (n - n // 2),
        "pv": [10_000] + [1] * (n - 1),
        "td_l1": ["X"] * n, "td_l2": ["x1"] * n,
        "bu_leaf": [0] * n, "bu_family_final": [0] * n,
    })
    man, pv = build_manifest(df, weight_col="pv", axis="stratum")
    df["pv_norm"] = pv
    built = build_tables(df, man, {}, guards.QuoteGuard(), tmp_path / "t",
                         n_boot=10, n_null=10, tag="x")
    text = report.build_report(built["frames"], built["summary"], man,
                               title="T", run_id="r", generation="gen01")
    assert f"「{long_q}」" in text, "the card printed a fragment instead of the row"


def test_a_boolean_column_reads_as_yes_or_no_not_as_one_or_zero():
    """`isinstance(True, int)` is True in Python.

    So the integer branch of the table renderer caught every boolean column and
    printed `超出噪声` — whose entire job is to say 是 or 否 — as `1`.
    """
    t = pd.DataFrame({"超出噪声(区间下端)": [True, False], "n": [3, 40000],
                      "x": [1.5, float("nan")], "名": ["甲", "乙"]})
    out = report._md(t)
    assert "| 是 |" in out and "| 否 |" in out
    assert "| 3 |" in out and "| 40,000 |" in out
