"""The reference shelf: 类目清单 / 标注规范与裁定规则 / 家族与叶层级 / 00_索引.

Each test records the gap it was written against. The common shape of all of
them: the run PRODUCED this content and delivered none of it, so the reader had
the argument for the taxonomy and not the taxonomy.
"""
from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pytest

from qmine.report import zh_reference as R


class _Store:
    def __init__(self, tmp_path):
        self.gen_dir = tmp_path


class _Deps:
    def __init__(self, tmp_path, art):
        self.store = _Store(tmp_path)
        self.cfg = types.SimpleNamespace(
            domain=types.SimpleNamespace(key="k12_zh"), report_language="zh")
        self._art = art

    def has(self, n):
        return n in self._art

    def load(self, n):
        return self._art[n]

    def leaf_labels_final(self):
        return self._art["leaf_labels_final"]

    def leaf_family_final(self):
        return self._art["leaf_family_final"]


TAX = {"taxonomy": {
    "labeling_guide": "先看查询是否含明确动作标记, 没有标记的裸字归入噪声类。",
    "nodes": [
        {"code": "char_pronunciation_lookup", "name": "查询字词读音", "level": 1,
         "definition": "用户输入字或词, 希望系统给出拼音。",
         "user_need": "获得正确拼音后, 用户即可满足。",
         "positive_examples": ["瑭怎么读", "炖汤的拼音"],
         "negative_examples": ["界的部首（属于 查询字形结构）"],
         "expected_share": 0.16, "pragmatic_only": False, "risk": False},
        {"code": "navigational", "name": "导航到具体站点", "level": 1,
         "definition": "用户想到达某个官网。", "user_need": "拿到入口即可。",
         "positive_examples": [], "negative_examples": [],
         "expected_share": 0.005, "pragmatic_only": True, "risk": False},
    ],
    "rules": [
        {"id": "r_pron", "when": "出现读音标记", "then": "char_pronunciation_lookup",
         "rationale": "读音标记明确表达发音需求。", "examples": ["行怎么读"],
         "added_in_round": 0, "trigger": "怎么读/拼音"},
        {"id": "R040", "when": "when 查询询问操作步骤", "then": "navigational",
         "rationale": "drafted by the referee to close a gap this disagreement exposed",
         "added_in_round": 1, "added_because": "disagreement on '加减消元法的步骤'"},
    ],
}}

NAMING = {"namings": [
    {"leaf_id": 0, "name_zh": "读音查询", "code": "pron", "user_need": "拿到拼音"},
    {"leaf_id": 1, "name_zh": "释义查询", "code": "mean", "user_need": "拿到释义"},
    {"leaf_id": 2, "name_zh": "导航查询", "code": "nav", "user_need": "拿到入口"},
], "audit": {"families": [
    {"family_id": 7, "name_zh": "字词类", "definition": "查字查词的统称。",
     "leaf_ids": [0, 1]},
]}}


@pytest.fixture
def deps(tmp_path):
    return _Deps(tmp_path, {
        "taxonomy_v2": TAX,
        "tree_naming": NAMING,
        "leaf_labels": True, "leaf_family": True,
        "leaf_labels_final": np.array([0, 0, 0, 1, 1, 2]),
        "leaf_family_final": np.array([0, 0, 1]),      # leaves 0,1 -> fam 0; leaf 2 -> fam 1
        "topdown_labels": pd.DataFrame({
            "query": list("abcdef"),
            "l1_pred": ["char_pronunciation_lookup"] * 4 + ["navigational"] * 2,
        }),
    })


def test_the_labeling_guide_reaches_the_reader_verbatim(deps):
    """`taxonomy.labeling_guide` is the document an annotator would need to
    reproduce the work. Grepped against all six of live42's deliverables and the
    notebook: ZERO occurrences. It existed in the artifact and was delivered
    nowhere, while the reports printed 「139 条规则」 — a count nobody can annotate
    with.
    """
    md = R.build_rules({"run_id": "t"}, deps)
    assert TAX["taxonomy"]["labeling_guide"] in md, "the guide must ship verbatim"


def test_every_adjudication_rule_is_delivered_with_its_reasoning(deps):
    """The reports counted the rules. 33 of live42's 39 first-round rule ids
    appeared in no deliverable at all, so a reader could see that 139 rules
    existed and not one of them.
    """
    md = R.build_rules({"run_id": "t"}, deps)
    for r in TAX["taxonomy"]["rules"]:
        assert f"`{r['id']}`" in md, f"rule {r['id']} was not delivered"
        assert r["when"].removeprefix("when ") in md
    assert "行怎么读" in md, "the worked examples are the usable part of a rule"


def test_a_referee_drafted_rule_is_not_delivered_in_english(deps):
    """`rationale` and `added_because` are OUR hardcoded English, and they carry
    100 of live42's 139 rules — the ones the referee EARNED against real
    disagreements, i.e. the half most worth reading. A Chinese annotation manual
    whose evidence-backed half is English is not usable by its own audience.

    The query inside `added_because` must NOT be translated: it is evidence.
    """
    md = R.build_rules({"run_id": "t"}, deps)
    assert "drafted by the referee" not in md
    assert "disagreement on" not in md
    assert "由裁判补写" in md
    assert "'加减消元法的步骤'" in md, "the disagreeing query is evidence, not prose"
    assert "when 查询询问" not in md, "the referee echoes the field name; strip it"


def test_the_class_catalogue_reports_the_DELIVERED_size(deps):
    """The architect's `expected_share` is a prediction made before seeing the
    corpus. Shipping only that, with no measured count beside it, presents a
    guess as a result. live42 delivered no per-class counts at all.
    """
    md = R.build_classes({"run_id": "t"}, deps)
    assert "查询字词读音" in md and "导航到具体站点" in md
    assert "4" in md and "66.7%" in md, "the measured share must appear"
    assert "16.0%" in md, "the prediction stays, beside the measurement"
    assert "瑭怎么读" in md, "positive examples were never delivered"
    assert "界的部首" in md, "negative examples are the boundary"


def test_a_class_the_clusterer_cannot_see_is_marked_as_such(deps):
    """`pragmatic_only` is the instance of this project's whole thesis — the
    classes bottom-up cannot reach. It must not be a silent field."""
    md = R.build_classes({"run_id": "t"}, deps)
    assert "聚类不可见" in md


def test_the_tree_is_read_from_the_DELIVERED_partition(deps):
    """p8 governance rewrites the tree after p7 names it, so the audit's
    `families` describe one tree and `leaf_family_final` another — live42: 20
    audit families against 24 delivered. Joining them by integer id names a
    family after a different family's leaves, which is how a family of
    classical-poetry leaves once shipped titled 「中考录取分数与学校排名查询」.
    """
    md = R.build_tree({"run_id": "t"}, deps)
    assert "家族 0" in md and "家族 1" in md
    assert "读音查询" in md and "导航查询" in md
    # leaf 2 sits in its own family and has no audit definition; leaves 0/1 do.
    assert "查字查词的统称。" in md


def test_a_tree_the_audit_no_longer_describes_says_so(deps, tmp_path):
    """A mismatch between the audited tree and the delivered one is not a detail
    to smooth over: it is the reason the family definitions below are joined by
    leaf membership rather than by id. Silence there reads as agreement.
    """
    art = dict(deps._art)
    art["tree_naming"] = {**NAMING, "audit": {"families": [
        {"family_id": i, "name_zh": f"f{i}", "definition": "d", "leaf_ids": [i]}
        for i in range(5)]}}                     # 5 audit families, 2 delivered
    md = R.build_tree({"run_id": "t"}, _Deps(tmp_path, art))
    assert "5 个家族" in md and "2 个" in md
    assert "⚠" in md, "the reader must be told the two trees differ"


def test_the_index_lists_only_documents_that_exist(deps, tmp_path):
    """Ten files landed in one directory with no index and no reading order, so
    the first decision a reader makes — which file to open — was the one the
    delivery did not help with. A dead link is worse than no index, so the list
    is built from what is actually on disk.
    """
    (tmp_path / "叶清单.md").write_text("x", encoding="utf-8")
    md = R.build_index({"run_id": "t"}, deps, None)
    assert "叶清单.md" in md
    assert "自下而上聚类最终报告.md" not in md, "a document that does not exist is a dead link"
    assert "pre_audit" in md, "the reader must be told what the snapshots are"


def test_each_catalogue_ships_a_machine_readable_twin(deps):
    """A controlled vocabulary ships both renderings: the person reads the prose,
    the program reads the table, and neither has to parse the other."""
    import csv as _csv
    import io

    for fn, key, n in ((R.classes_csv, "code", 2),
                       (R.rules_csv, "id", 2),
                       (R.tree_csv, "leaf_id", 3)):
        rows = list(_csv.DictReader(io.StringIO(fn({"run_id": "t"}, deps))))
        assert len(rows) == n, f"{fn.__name__} lost rows"
        assert rows[0][key], f"{fn.__name__} has no key column"


def test_a_family_says_where_it_lands_on_the_other_route(deps, tmp_path):
    """`route_crosswalk.csv` is the ONLY artifact that says how the two routes
    line up, and it was named in no deliverable — a delivery whose whole thesis
    is that the routes answer different questions shipped no per-family evidence
    for it.

    It is keyed by delivered family, so it belongs beside the family. A table in
    another file that has to be joined by hand is a table nobody joins.
    """
    art = dict(deps._art)
    art["route_crosswalk"] = pd.DataFrame([
        {"bu_family_final": 0, "n": 5, "td_dominant": "char_pronunciation_lookup",
         "td_dominant_share": 0.911, "td_classes_touched": 12,
         "td_effective_classes": 1.579, "verdict": "routes agree"},
        {"bu_family_final": 1, "n": 1, "td_dominant": "navigational",
         "td_dominant_share": 0.361, "td_classes_touched": 21,
         "td_effective_classes": 8.332, "verdict": "routes disagree"},
    ])
    md = R.build_tree({"run_id": "t"}, _Deps(tmp_path, art))
    assert "routes disagree" in md and "routes agree" in md
    assert "91.1%" in md, "the dominant share is the readable part"
    assert "8.332" in md, "effective classes is what says the routes diverged"


def test_a_missing_crosswalk_does_not_lose_the_tree(deps, tmp_path):
    """The tree is the document; the crosswalk is a section of it. A run that
    never produced the crosswalk must still deliver the tree."""
    art = {k: v for k, v in deps._art.items() if k != "route_crosswalk"}
    md = R.build_tree({"run_id": "t"}, _Deps(tmp_path, art))
    assert "家族 0" in md and "读音查询" in md
    assert "1c." not in md


def test_a_leaf_shows_the_evidence_it_was_named_FROM(tmp_path):
    """`naming_cards.json` holds the exact sample the blind namer was shown — 15
    centroid, 10 random, 5 edge per leaf. None of live42's 1,470 sampled queries
    appeared in any deliverable, so a reader judging whether a name is right
    could not see what it was named from.

    The EDGE samples are the load-bearing half: a name that covers the centre and
    not the edge is too narrow, and that is only visible when both are shown. The
    sampling is mechanical, which is what makes them admissible rather than a
    flattering selection.
    """
    from qmine.report import zh_catalogue as C

    art = {
        "tree_naming": {"namings": [
            {"leaf_id": 0, "name_zh": "暑假放假时间查询", "code": "vac",
             "user_need": "拿到放假日期"}]},
        "leaf_labels": True, "leaf_family": True,
        "leaf_labels_final": np.array([0, 0, 0]),
        "leaf_family_final": np.array([0]),
        "naming_cards": {"cards": [{
            "leaf_id": 0,
            "center_samples": ["2026年中小学放暑假时间", "小学放暑假2026年放假时间"],
            "edge_samples": ["退潮赶海时间表", "目瑙纵歌2026年时间表"],
            "top_ngrams": ["暑假", "放假时间"],
        }]},
    }
    md = C.build({"run_id": "t"}, _Deps(tmp_path, art))
    assert "2026年中小学放暑假时间" in md, "the centroid sample is the name's basis"
    assert "退潮赶海时间表" in md, "the edge sample is where the name is tested"
    assert "暑假" in md
    assert "质心" in md and "边缘" in md, "the reader must know which is which"


def test_a_leaf_with_no_naming_card_still_ships(tmp_path):
    """A leaf p8 governance created after p7 named the tree has no card. It must
    lose its exemplars, not its entry — the delivered partition decides who is
    listed."""
    from qmine.report import zh_catalogue as C

    md = C.build({"run_id": "t"}, _Deps(tmp_path, {
        "tree_naming": {"namings": [
            {"leaf_id": 0, "name_zh": "某叶", "user_need": "x"}]},
        "leaf_labels": True, "leaf_family": True,
        "leaf_labels_final": np.array([0, 0]),
        "leaf_family_final": np.array([0]),
    }))
    assert "某叶" in md and "质心" not in md


def test_an_audit_definition_is_only_shown_when_it_describes_THIS_family(tmp_path):
    """The first version took the first definition found among a family's leaves
    and printed it as 「审计给出的定义」. Measured on live42: 14 of the 24 delivered
    families then carried a definition shared with one or two others — family 8
    (17 leaves, 14,171 rows) and family 10 (1 leaf) got the identical sentence.

    That is this project's own delivered-partition trap. The audit describes the
    20-family PRE-GOVERNANCE tree, so its definitions are not addressed to the
    families that shipped, and a borrowed one presented as this family's is a
    wrong statement rather than a missing one.
    """
    art = {
        "tree_naming": {
            "namings": [{"leaf_id": i, "name_zh": f"叶{i}", "user_need": "u"}
                        for i in range(4)],
            "audit": {"families": [
                # one audit family spanning leaves that governance split apart
                {"family_id": 0, "name_zh": "释义类", "definition": "查词义。",
                 "leaf_ids": [0, 1, 2]},
                # one that maps cleanly onto a single delivered family
                {"family_id": 1, "name_zh": "读音类", "definition": "查读音。",
                 "leaf_ids": [3]},
            ]},
        },
        "leaf_labels": True, "leaf_family": True,
        "leaf_labels_final": np.array([0, 1, 2, 3]),
        # leaves 0,1 -> fam 0 ; leaf 2 -> fam 1 ; leaf 3 -> fam 2
        "leaf_family_final": np.array([0, 0, 1, 2]),
    }
    md = R.build_tree({"run_id": "t"}, _Deps(tmp_path, art))
    # 释义类 backs delivered families 0 AND 1 — neither may claim it.
    assert md.count("查词义。") == 0, "a shared definition was presented as one family's"
    assert "没有专属于本家族的定义" in md
    # 读音类 backs exactly one delivered family, from one audit family.
    assert "查读音。" in md, "a clean 1:1 definition must still be shown"
    assert md.count("**审计给出的定义**") == 1


def test_the_L2_sub_intents_section_is_not_silently_empty(tmp_path):
    """`zh_topdown` read a LIST from `subintents` or `groups`; the artifact
    carries a DICT under `subdivision`, keyed by L1 code. Neither key has ever
    existed, so the guard was always false and the section vanished — while the
    panel's strongest comparative claim (L2 purity 0.7989 against bottom-up
    leaves 0.7797) rests on exactly these groups.
    """
    from qmine.report.zh_topdown import build

    art = {
        "taxonomy_v2": TAX,
        "subintents": {"n_sub_intents": 5, "subdivision": {
            "char_pronunciation_lookup": {"n": 7484, "k": 2, "lift_over_null": 0.121,
                                          "stability_ari": 0.993,
                                          "silhouette_disagrees": False},
            "poem_and_classical_text_lookup": {"n": 4716, "k": 3, "lift_over_null": 0.0666,
                                               "stability_ari": 0.9734,
                                               "silhouette_disagrees": True},
            "navigational": {"n": 393, "k": 1},        # not split — must not be listed
        }},
    }
    d = _Deps(tmp_path, art)
    d.cfg = types.SimpleNamespace(domain=types.SimpleNamespace(key="k12_zh"),
                                  report_language="zh", config_hash="h", seed=0)
    d.registry = types.SimpleNamespace(provenance_note=lambda lang="zh": "",
                                       usage=lambda: {})
    md = build({"run_id": "t", "gates": {}, "decisions": [], "findings": [],
                "observations": []}, d)
    assert "### 2.1 L2 子意图" in md
    assert "char_pronunciation_lookup" in md and "7,484" in md
    assert "没有名字" in md, "an unnamed layer must not read as deliverable labels"
    assert "✓ 反对" in md, "silhouette disagreement is disclosed, not decisive"
    seg = md[md.index("### 2.1"):]
    assert "navigational" not in seg.split("silhouette 是否反对")[1][:400], (
        "a class that was not split has no sub-intents to list")


def _fam_fixture():
    """Two delivered families: one clean, one merged from three sources with a
    governance-created leaf that no audit family covers."""
    naming = {
        "namings": [{"leaf_id": i, "name_zh": f"叶{i}", "user_need": "u"}
                    for i in range(5)],
        "audit": {"families": [
            {"family_id": 0, "name_zh": "释义类", "definition": "d", "leaf_ids": [0]},
            {"family_id": 1, "name_zh": "读音类", "definition": "d", "leaf_ids": [1]},
            {"family_id": 2, "name_zh": "笔顺类", "definition": "d", "leaf_ids": [2]},
            {"family_id": 3, "name_zh": "诗文类", "definition": "d", "leaf_ids": [4]},
        ]},
    }
    # leaves 0,1,2 -> family 0 ; leaf 3 (governance-created, no audit) -> family 0
    # leaf 4 -> family 1
    leaf_family = np.array([0, 0, 0, 0, 1])
    sizes = np.array([50, 30, 10, 10, 100])   # family 0 = 100 rows, family 1 = 100
    return naming, leaf_family, sizes


def test_a_family_label_never_hides_its_denominator():
    """The label read `字词/短语/概念释义查询 等 6 类 (42%)` and was used AS the
    family's name — in headings, table cells, a Mermaid node and a CSV `name`
    column. Four things were wrong at once:

    * `等 N 类` reads in Chinese as "and N others", so 6 was read as 7;
    * the percentage named no referent;
    * its denominator was the NAMED subset, not the family — 42% of 12,844 rather
      than 38% of live42's family 8, which has 14,171;
    * so the three governance-created leaves in that family, 9.4% of it, were
      absent from both the label and its arithmetic.
    """
    from qmine.report._shape import family_names

    naming, leaf_family, sizes = _fam_fixture()
    names = family_names(naming, leaf_family, sizes)

    assert "等" not in names[0], "the ambiguous 等 N 类 form is back"
    assert "%" in names[0]
    # 释义类 is 50 of the family's 100 rows. Against the NAMED subset (90) it
    # would be 56% — the old, wrong denominator.
    assert "50%" in names[0], f"share must be of the family, got {names[0]}"
    assert "56%" not in names[0]
    # A family whose leaves all come from one audit family keeps a plain name.
    assert names[1] == "诗文类"


def test_a_family_with_an_unnamed_leaf_is_not_called_clean():
    """`family_names` branched on the count of NAMED contributors alone, so
    live42's family 1 — 56% of its rows in a governance-created leaf with no
    audit name — was labelled with the one audit name it had, plainly.
    """
    from qmine.report._shape import family_names

    naming = {"namings": [], "audit": {"families": [
        {"family_id": 0, "name_zh": "唯一来源", "leaf_ids": [0]}]}}
    leaf_family = np.array([0, 0])          # leaf 1 has no audit family
    sizes = np.array([44, 56])
    label = family_names(naming, leaf_family, sizes)[0]
    assert label != "唯一来源", "a family that is 56% unaccounted for is not that"
    assert "44%" in label


def test_the_composition_accounts_for_every_row_of_the_family():
    """The shares must sum to 1 across the contributors AND the unnamed
    remainder — that is what makes the denominator checkable."""
    from qmine.report._shape import family_composition

    naming, leaf_family, sizes = _fam_fixture()
    comp = family_composition(naming, leaf_family, sizes)
    c = comp[0]
    assert c["rows"] == 100 and c["unnamed_leaves"] == 1 and c["unnamed_rows"] == 10
    total = sum(sh for _n, _r, sh in c["contributors"]) + c["unnamed_rows"] / c["rows"]
    assert abs(total - 1.0) < 1e-9, f"shares do not close: {total}"
    assert not c["exact"] and comp[1]["exact"]


def test_the_machine_readable_twin_does_not_put_a_label_in_a_name_column(deps):
    """`family_name` carried `混合·主要成分「句子语录查询」38%` — a sentence, a
    percentage and a name in one field. A consumer grouping by it gets a string
    that changes whenever the composition shifts."""
    import csv as _csv
    import io

    rows = list(_csv.DictReader(io.StringIO(R.tree_csv({"run_id": "t"}, deps))))
    assert rows, "no rows"
    for r in rows:
        assert "%" not in r["family_name"], "a label leaked into the name column"
        if r["family_name_is_exact"] == "False":
            assert r["family_name"] == "", "a mixed family has no name of its own"
        assert r["family_label"], "the display label must still be available"


def test_the_notebook_does_not_re_implement_the_family_join():
    """It did, and the two drifted: the notebook printed `X 等 6 类` while the
    reports printed `混合·主要成分「X」38%`, so one run produced two documents that
    disagreed about what a family is called."""
    import inspect

    from qmine.report import zh_notebook

    src = inspect.getsource(zh_notebook)
    assert "等 {len(_r)} 类" not in src, "the duplicate implementation is back"
    assert "from qmine.report._shape import family_names" in src
