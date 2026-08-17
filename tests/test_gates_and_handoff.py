"""Tests for the statistical gate and the external-panel handoff.

Both exist because of failures observed on the real corpus: a blocking gate
fired on 0.978-vs-0.980 with a 1,000-row test set (sampling noise), and the
naming step is the one place where a stronger reviewer than the run's own model
is worth the trouble.
"""

from __future__ import annotations

import json

import pytest

from qmine.memory.context import BlindnessFirewall, BlindnessViolation
from qmine.ops.handoff import coverage_report, export_shards, import_namings
from qmine.ops.stats import proportion_gate, required_n, wilson_interval
from qmine.records import NamingCard, Taxonomy, TaxonomyNode


# -- statistical gates ------------------------------------------------------

def test_small_sample_near_miss_is_underpowered_not_failed():
    """The exact case that halted a real run: 0.978 on n=1000 against 0.98."""
    r = proportion_gate(0.978, 1000, 0.98)
    assert r["verdict"] == "underpowered"
    assert r["passed"] is True
    assert r["blocking_failure"] is False
    assert r["n_needed"] > 1000


def test_same_value_on_a_large_sample_is_a_real_miss():
    r = proportion_gate(0.978, 40000, 0.98)
    assert r["verdict"] == "missed"
    assert r["passed"] is False
    assert r["blocking_failure"] is True


def test_a_clear_pass_is_met():
    r = proportion_gate(0.995, 10000, 0.98)
    assert r["verdict"] == "met"
    assert r["passed"] is True


def test_a_clear_failure_is_missed_even_on_a_small_sample():
    r = proportion_gate(0.70, 500, 0.98)
    assert r["verdict"] == "missed"


def test_wilson_interval_stays_inside_zero_one_near_the_boundary():
    lo, hi = wilson_interval(1000, 1000)
    assert 0.0 <= lo <= hi <= 1.0
    assert hi == 1.0


def test_wilson_narrows_as_n_grows():
    def width(n):
        lo, hi = wilson_interval(int(0.98 * n), n)
        return hi - lo

    assert width(100) > width(1000) > width(10000)


def test_required_n_grows_as_the_gap_shrinks():
    assert required_n(0.979, 0.98) > required_n(0.95, 0.98)
    assert required_n(0.98, 0.98) is None


# -- external panel handoff -------------------------------------------------

@pytest.fixture
def cards():
    return [
        NamingCard(leaf_id=i, size=200, share=0.02,
                   center_samples=[f"查询{i}的拼音"], random_samples=[f"词{i}怎么读"],
                   edge_samples=[f"边缘{i}"], top_ngrams=["的拼音"])
        for i in range(7)
    ]


def test_shards_partition_every_cluster_exactly_once(cards, tmp_path):
    m = export_shards(cards, tmp_path, n_shards=3)
    seen = [lid for s in m["shards"] for lid in s["leaf_ids"]]
    assert sorted(seen) == list(range(7))
    assert len(seen) == len(set(seen))


def test_exported_briefs_reject_a_smuggled_annotation_field(cards, tmp_path):
    """The blindness guarantee must travel with the cards, not stop at the process boundary.

    The check that matters is structural: a card carrying any field outside the
    blind contract is refused whatever it contains. A lexical scan alone would
    not catch a field named `taxonomy_hint` holding a paraphrase.
    """
    fw = BlindnessFirewall()
    smuggled = cards[0].model_dump() | {"legacy_label": "怎么读/读音/拼音"}
    with pytest.raises(BlindnessViolation, match="not part of the blind card contract"):
        fw.assert_card_blind(smuggled)


def test_exported_briefs_are_checked_by_the_same_firewall(cards, tmp_path):
    """export_shards must route every card through the check, not just the in-process namer."""
    calls: list[str] = []

    class _Recording(BlindnessFirewall):
        def assert_card_blind(self, card, *, what="naming card"):
            calls.append(what)
            return super().assert_card_blind(card, what=what)

    export_shards(cards, tmp_path, n_shards=2, firewall=_Recording())
    assert len(calls) == len(cards)


def test_member_queries_are_not_treated_as_leaks(cards, tmp_path):
    """Corpus text is the data, not an annotation.

    A taxonomy named after the phrasing it describes ("的拼音") appears inside
    every member query of the cluster it names. Scanning raw samples for label
    vocabulary would fail every card and force the check to be switched off,
    which is how a guarantee becomes a comment.
    """
    tax = Taxonomy(nodes=[TaxonomyNode(code="PRON", name="的拼音", definition="x")])
    fw = BlindnessFirewall().add_taxonomy(tax)
    m = export_shards(cards, tmp_path, n_shards=2, firewall=fw)
    assert m["n_shards"] == 2
    assert "查询0的拼音" in (tmp_path / "shard_01.md").read_text(encoding="utf-8")


def test_brief_contains_the_data_and_no_answers(cards, tmp_path):
    m = export_shards(cards, tmp_path, n_shards=2)
    text = (tmp_path / "shard_01.md").read_text(encoding="utf-8")
    assert "查询0的拼音" in text
    assert "edge" in text.lower()
    assert "user_need" in text
    assert m["n_shards"] == 2


def test_import_accepts_the_shapes_a_panel_actually_returns():
    flat = import_namings([{"leaf_id": 0, "name_zh": "a", "code": "a", "user_need": "u", "coherence": 5}])
    wrapped = import_namings({"namings": [{"leaf_id": 0, "name_zh": "a", "code": "a",
                                           "user_need": "u", "coherence": 5}]})
    nested = import_namings([[{"leaf_id": 0, "name": "a", "code": "a", "user_need": "u", "coherence": 5}]])
    assert len(flat) == len(wrapped) == len(nested) == 1
    assert nested[0].name_zh == "a"


def test_import_reads_a_file_path(tmp_path):
    p = tmp_path / "verdicts.json"
    p.write_text(json.dumps([{"leaf_id": 3, "name_zh": "x", "code": "x",
                              "user_need": "u", "coherence": 4}]), encoding="utf-8")
    out = import_namings(p)
    assert out[0].leaf_id == 3


def test_import_stamps_provenance():
    out = import_namings([{"leaf_id": 1, "name_zh": "x", "code": "x", "user_need": "u",
                           "coherence": 3}], named_by="opus-panel")
    assert out[0].named_by == "opus-panel"


def test_coverage_report_surfaces_silently_dropped_clusters():
    got = import_namings([{"leaf_id": 0, "name_zh": "x", "code": "x", "user_need": "u", "coherence": 3}])
    cov = coverage_report(got, range(5))
    assert cov["complete"] is False
    assert cov["missing"] == [1, 2, 3, 4]


# -- the offline stand-in must work in any script ---------------------------

def test_offline_namer_extracts_terms_from_english_not_just_chinese():
    """The stand-in used to count only CJK n-grams, so every English cluster came
    back with a placeholder name and 'runs offline on any corpus' was false."""
    from qmine.llm.offline import top_terms

    en = ["how to clean dyson vacuum", "how to reset bosch dishwasher",
          "how to clean shark vacuum", "how to descale philips kettle",
          "how to clean roomba vacuum"]
    terms = top_terms(en)
    assert terms, "no terms extracted from English samples"
    assert any("vacuum" in t or "clean" in t for t in terms)
    assert not any(t in {"how", "to", "the"} for t in terms), "stopwords leaked into terms"


def test_offline_namer_still_handles_chinese():
    from qmine.llm.offline import top_terms

    zh = ["氢怎么读", "钦州的拼音", "木加射读什么", "徜徉怎么读", "臌读什么"]
    assert any("怎么读" in t or "拼音" in t for t in top_terms(zh))


def test_offline_definition_matches_the_corpus_language():
    """A definition sentence in a different script from the data cannot be
    checked against the data by the people who own it."""
    from qmine.llm.offline import synthesize
    from qmine.records import LeafNaming

    def _need(samples):
        card = "## Cluster 1\n" + "\n".join(f"- {s}" for s in samples)
        return LeafNaming.model_validate({**synthesize(card, LeafNaming), "leaf_id": 1}).user_need

    en = _need(["how to clean dyson vacuum", "how to clean shark vacuum",
                "how to clean roomba vacuum", "how to reset bosch vacuum"])
    zh = _need(["氢怎么读", "钦州的拼音", "木加射读什么", "徜徉怎么读"])
    assert "the user asks" in en and "用户" not in en
    assert "用户" in zh
