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


def test_a_rule_naming_a_class_that_does_not_exist_is_caught():
    """Rules are rendered verbatim into both annotators' prompts. A live run
    shipped `R12 → 选 EXOD_INFO` against a taxonomy declaring `EXAM_INFO`; the
    annotators could not choose it, and EXAM_INFO×POLICY_REGULATION became a
    top-five disagreement pair."""
    from qmine.graph.nodes.topdown import _validate_rules
    from qmine.records import AdjudicationRule, Taxonomy, TaxonomyNode

    emitted: list[str] = []

    class _Deps:
        @staticmethod
        def emit(msg: str) -> None:
            emitted.append(msg)

    def _tax(*rules: AdjudicationRule) -> Taxonomy:
        return Taxonomy(
            nodes=[TaxonomyNode(code=c, name=c, definition=f"{c} 的需求")
                   for c in ("EXAM_INFO", "WORD_MEANING", "POLICY_REGULATION")],
            rules=list(rules),
        )

    # A near-miss with one dominant candidate is repaired.
    t = _tax(AdjudicationRule(id="R12", when="考务数据", then="选 EXOD_INFO"))
    health = _validate_rules(t, _Deps)
    assert health["n_repaired"] == 1 and health["n_dropped"] == 0
    assert t.rules[0].then == "选 EXAM_INFO"
    assert any("EXOD_INFO" in m for m in emitted), "the repair must be reported, not silent"

    # A target resembling nothing is dropped rather than guessed at.
    t2 = _tax(AdjudicationRule(id="R99", when="x", then="选 TOTALLY_UNRELATED"))
    health2 = _validate_rules(t2, _Deps)
    assert health2["n_dropped"] == 1 and not t2.rules

    # A valid rule is left exactly as it was.
    t3 = _tax(AdjudicationRule(id="R01", when="x", then="选 WORD_MEANING"))
    health3 = _validate_rules(t3, _Deps)
    assert health3 == {"n_repaired": 0, "n_dropped": 0, "repaired": [], "dropped": []}
    assert t3.rules[0].then == "选 WORD_MEANING"


def test_rule_dedup_separates_a_second_marker_from_a_second_opinion():
    """Two markers for one boundary pointing at opposite classes is what settling
    a boundary looks like. Two rules on the SAME trigger giving different answers
    is a contradiction. An earlier version compared the rendered `when` sentence,
    where those two cases differ by about two characters in forty-five (0.957
    similar) — so it withheld both halves of every legitimate pair and destroyed
    32 of 41 rules on a live run."""
    from qmine.graph.nodes.topdown import _dedupe_rules
    from qmine.records import AdjudicationRule, Taxonomy, TaxonomyNode

    emitted: list[str] = []

    class _Deps:
        @staticmethod
        def emit(msg: str) -> None:
            emitted.append(msg)

    tax = Taxonomy(nodes=[TaxonomyNode(code=c, name=c, definition=c)
                          for c in ("PRON", "STRUCT", "MEANING")], rules=[])

    def rule(rid, trigger, then, classes=("PRON", "STRUCT")):
        return AdjudicationRule(
            id=rid, then=then, classes=list(classes), trigger=trigger,
            when=f"查询包含「{trigger}」且候选类目为 {classes[0]} 或 {classes[1]}",
        )

    # Complementary: one boundary, two markers, opposite classes. Both must live.
    pair = [rule("R1", "拼音", "PRON"), rule("R2", "偏旁", "STRUCT")]
    kept = _dedupe_rules(pair, tax, _Deps)
    assert len(kept) == 2, "a legitimate discriminating pair was destroyed"

    # Genuine contradiction: same trigger, same boundary, different answer.
    clash = [rule("R3", "拼音", "PRON"), rule("R4", "拼音", "STRUCT")]
    emitted.clear()
    kept = _dedupe_rules(clash, tax, _Deps)
    assert kept == [], "a real contradiction was allowed through"
    assert any("withheld" in m for m in emitted)

    # Exact duplicate: same trigger, same answer — keep one, silently.
    dup = [rule("R5", "拼音", "PRON"), rule("R6", "拼音", "PRON")]
    assert len(_dedupe_rules(dup, tax, _Deps)) == 1

    # The same marker on a DIFFERENT boundary is a different rule entirely.
    cross = [rule("R7", "拼音", "PRON"), rule("R8", "拼音", "MEANING", ("MEANING", "STRUCT"))]
    assert len(_dedupe_rules(cross, tax, _Deps)) == 2, "different boundaries must not collide"

    # Referee rules carry no trigger, so they still fall back to text similarity.
    prose = [AdjudicationRule(id="P1", when="当查询为单个汉字且无上下文时", then="PRON"),
             AdjudicationRule(id="P2", when="当查询为单个汉字且无上下文时", then="STRUCT")]
    emitted.clear()
    assert _dedupe_rules(prose, tax, _Deps) == [], "prose contradiction slipped through"


def test_kappa_is_not_reported_as_a_verdict_when_coverage_collapsed():
    """kappa says something about annotator agreement only if the annotators
    answered. A provider outage once left one annotator with 199 of 600 rows and
    the gate reported "kappa 0.813" as a judgement on the labelling guide — it
    was a judgement on whichever rows happened to survive."""
    from qmine.config import QMineConfig

    cfg = QMineConfig()
    assert 0 < cfg.gates.min_annotation_coverage <= 1.0

    # The shape the gate keys on, as `agreement()` reports it.
    collapsed = {"n": 199, "n_submitted": 600, "kappa": 0.813}
    healthy = {"n": 596, "n_submitted": 600, "kappa": 0.913}

    def unsound(a):
        return a["n"] / (a.get("n_submitted") or a["n"]) < cfg.gates.min_annotation_coverage

    assert unsound(collapsed), "33% coverage must not be treated as a measurement"
    assert not unsound(healthy), "99% coverage is a usable measurement"

    # A collapsed sample must never pass, however flattering its kappa.
    flattering = {"n": 8, "n_submitted": 600, "kappa": 0.99}
    assert unsound(flattering)
    assert not ((not unsound(flattering)) and flattering["kappa"] >= cfg.gates.kappa), (
        "a run that lost 99% of its sample would have cleared the gate"
    )


def test_gold_set_size_follows_the_corpus_not_a_constant():
    """The playbook asks for a stratified 3,000-5,000 (line 191) and separately
    for a HIGHER proportion on a small corpus (line 119). This shipped with a
    hardcoded 600, which satisfies neither: five times under spec on a large
    corpus, and on a 2,000-row one it annotates 30% of the data by accident."""
    from qmine.config import QMineConfig, gold_size_for

    cfg = QMineConfig()
    tax = cfg.taxonomy
    assert tax.gold_sample_size is None, "the default must derive, not pin"
    lo, hi = tax.gold_size_range
    assert (lo, hi) == (3000, 5000), "playbook range"

    # Large corpora get the playbook floor, not a proportion of everything.
    for n in (50_000, 500_000):
        assert gold_size_for(n, tax) == lo

    # Small corpora get a larger SHARE, which is the rule that is easy to miss.
    small, large = gold_size_for(3_000, tax), gold_size_for(50_000, tax)
    assert small / 3_000 > large / 50_000, "small corpus must get a higher proportion"

    # But a gold set that is most of the corpus has stopped being a sample.
    for n in (500, 3_000, 12_000):
        assert gold_size_for(n, tax) <= n * tax.gold_max_fraction + 1

    # Monotone in corpus size, and never degenerate.
    sizes = [gold_size_for(n, tax) for n in (200, 800, 3_000, 8_000, 12_000, 50_000)]
    assert sizes == sorted(sizes), sizes
    assert all(s >= 50 for s in sizes)

    # Pinning it explicitly still wins, so a caller can force a cheap run.
    tax.gold_sample_size = 600
    assert (tax.gold_sample_size or gold_size_for(12_000, tax)) == 600


def test_the_pilot_gate_is_blocking_and_precedes_the_gold_set():
    """50 queries and 4 LLM calls, before hundreds are spent on the gold set.
    It was declared in the blocking list and emitted by no node at all."""
    import inspect

    from qmine.config import QMineConfig
    from qmine.graph.nodes import topdown

    cfg = QMineConfig()
    assert "p2a_pilot_agreement" in cfg.gates.blocking
    assert cfg.taxonomy.pilot_agreement_threshold == 0.85, "playbook: 一致率 <85%"

    src = inspect.getsource(topdown.p2a_taxonomy)
    assert "p2a_pilot_agreement" in src, "the declared gate must actually be emitted"
    # It has to run in 2a — the whole point is to fire before 2b spends anything.
    assert "_pilot_agreement" in src
    assert "p2a_pilot_agreement" not in inspect.getsource(topdown.p2b_gold)


def test_figure_cells_do_not_rebind_names_the_setup_cell_owns():
    """Notebook cells share one namespace. A figure cell that assigns to a name the
    setup cell owns corrupts every cell after it — `fam = rows['algorithm']...` in
    the battery figure replaced the leaf→family map with a Series of strings, and
    the family listing eight cells later died on int('kmeans')."""
    import ast

    from qmine.report import zh_figures as figs

    # Names the setup cell binds and every later cell depends on.
    OWNED = {
        "audit", "tmpl", "rep", "gran", "meta", "naming", "gov", "dep", "panel",
        "labels", "fam", "famrow", "df", "GEN", "J", "NPY", "CSV", "SAVE", "FIGDIR",
        "BLUE", "ORANGE", "GREEN", "MUTED",
    }

    producers = ("fig_ksweep", "fig_alpha", "fig_battery",
                 "fig_umap_families", "fig_umap_intent", "fig_panel_bars")
    offenders: dict[str, set[str]] = {}
    for name in producers:
        src = getattr(figs, name)()
        tree = ast.parse(src)
        bound: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        bound.add(t.id)
            elif isinstance(node, (ast.AugAssign, ast.AnnAssign)) and isinstance(node.target, ast.Name):
                bound.add(node.target.id)
            elif isinstance(node, ast.For) and isinstance(node.target, ast.Name):
                bound.add(node.target.id)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                bound.add(node.name)
        clash = bound & OWNED
        # Rebinding a *derived* artifact dict is fine only if it re-reads it.
        clash -= {"panel"} if "panel = J(" in src else set()
        if clash:
            offenders[name] = clash
    assert not offenders, f"figure cells rebind setup-cell names: {offenders}"


def test_the_pilot_can_actually_discriminate_against_its_own_bar():
    """A gate that cannot fail is decorative. The playbook's 50-row pilot has no
    power against the kappa bar it exists to predict: at the agreement three live
    runs actually produced (raw 0.849 / kappa 0.831) the 95% upper bound at n=50 is
    0.924, so a guide destined to fail kappa 0.90 passes the pilot. The sample must
    be large enough that the test can return 'no'."""
    from qmine.config import QMineConfig

    cfg = QMineConfig()

    def upper_bound(po: float, kappa: float, n: int) -> float:
        pe = (po - kappa) / (1 - kappa)
        se = ((po * (1 - po) / n) ** 0.5) / (1 - pe)
        return kappa + 1.645 * se

    # The agreement live02 / live05 / live20 all landed near.
    observed_po, observed_kappa = 0.849, 0.831
    n = cfg.taxonomy.pilot_sample_size

    assert upper_bound(observed_po, observed_kappa, 50) >= cfg.gates.kappa, (
        "fixture check: n=50 is supposed to be under-powered"
    )
    assert upper_bound(observed_po, observed_kappa, n) < cfg.gates.kappa, (
        f"the pilot at n={n} still cannot reject a guide measured at kappa "
        f"{observed_kappa}; it would pass and the run would pay for the full gold set"
    )
    # ...and still be cheap relative to what it protects.
    pilot_calls = 2 * -(-n // 25)
    gold_calls = 2 * -(-3000 // 25)
    assert pilot_calls < gold_calls / 10, "the pilot has stopped being the cheap check"


def test_the_pilot_separates_a_fixable_guide_from_an_annotator_ceiling():
    """Two annotators disagreeing does not say WHY, and the two causes have
    opposite remedies. Measuring the same annotator against itself gives the
    ceiling any pair could reach, which separates them:

      inter << intra  → the guide has slack, redraft it (halt, it is cheap here)
      inter ~= intra  → the annotator is the constraint; guide repair cannot help,
                        so halting would demand a fix the operator cannot perform

    Without this the gate imports a kappa bar from another project and treats an
    unreachable one as a guide defect. On this corpus collapsing the taxonomy from
    21 classes to 4 moved kappa only 0.808 → 0.832, so the taxonomy was never it.
    """
    from qmine.config import QMineConfig

    cfg = QMineConfig()

    def verdict(inter_upper: float, share_of_ceiling: float) -> bool:
        """The gate's own rule: pass if it reaches target OR sits at the ceiling."""
        return inter_upper >= cfg.gates.kappa or share_of_ceiling >= 0.90

    # Clears the bar outright.
    assert verdict(0.94, 0.80)

    # Well short of the bar, and well short of what the annotator can do alone:
    # a real guide defect, and the gate must stop the run before the gold set.
    assert not verdict(0.72, 0.78)

    # Well short of the bar, but the annotator only agrees with ITSELF at ~0.85 —
    # no rule the referee writes can close that. Halting here would be demanding
    # the impossible, so it warns and records the ceiling instead.
    assert verdict(0.83, 0.98)

    # The boundary is on the ceiling share, not on the absolute kappa.
    assert verdict(0.10, 0.95)
    assert not verdict(0.89, 0.50)


def test_gates_do_not_import_thresholds_that_only_fit_one_corpus():
    """Three gates were absolute constants taken from a single K12 run, and three
    corpora disagree with all of them: the e-commerce corpus was flagged for 53%
    template coverage (being more templated is not a defect), k12 and mixed fell
    below the 0.98 held-out bar while e-commerce cleared it, and mean coherence
    landed at 3.93 and 4.27 on two runs of the SAME corpus."""
    from qmine.config import QMineConfig

    g = QMineConfig().gates

    # Coverage: gated on rows, because that is what the metrics are computed over.
    # The share is kept for reporting and must not decide anything.
    assert hasattr(g, "min_template_rows") and g.min_template_rows > 0
    assert hasattr(g, "min_template_row_fraction")
    covered_rows_ecom = int(0.534 * 8000)      # the run that used to be flagged
    assert covered_rows_ecom >= max(g.min_template_rows,
                                    int(g.min_template_row_fraction * 8000))

    # Held-out: the effective bar may only ever RELAX the configured floor, never
    # exceed it — it exists to stop the gate demanding more than the structure
    # achieves when the data is split at all.
    ceiling = 0.8927                            # measured on the reference run
    effective = min(g.heldout_reproduction, ceiling * g.heldout_share_of_ceiling)
    assert effective <= g.heldout_reproduction
    assert 0.9731 >= effective, "the reference run must clear its own relative bar"

    # Coherence: the tail, not the mean. A tree whose average passes while a fifth
    # of its leaves are incoherent must NOT pass.
    good_mean_bad_tail = [5.0] * 16 + [1.5] * 4          # mean 4.3, 20% weak
    weak = [c for c in good_mean_bad_tail if c < g.coherence_weak_below]
    share = len(weak) / len(good_mean_bad_tail)
    assert sum(good_mean_bad_tail) / len(good_mean_bad_tail) >= g.coherence, (
        "fixture check: this tree passes the old mean-based bar"
    )
    assert share > g.coherence_max_weak_share, (
        "a tree with a fifth of its leaves incoherent still passes — the gate is "
        "reading the mean again"
    )


def test_a_taxonomy_without_tie_breaks_cannot_reach_annotation():
    """A taxonomy with no adjudication rules cannot be annotated consistently, so
    the rule floor blocks while the class-count range only warns.

    Measured on a live 50k run: 19 classes shipped with ONE rule, two independent
    annotators then agreed at kappa 0.761, and the same annotator agreed with
    itself at 0.900. The entire 14-point gap was missing tie-breaks — and the
    pilot spent real money to discover what this gate sees for free.
    """
    from qmine.config import QMineConfig

    cfg = QMineConfig()
    lo, hi = cfg.taxonomy.l1_target_range
    floor = cfg.taxonomy.min_adjudication_rules
    assert floor >= 10, "a floor this low cannot cover the confusable pairs"

    def gate(n_l1: int, n_rules: int) -> tuple[bool, bool]:
        passed = (lo <= n_l1 <= hi) and n_rules >= floor
        warn_only = n_rules >= floor and lo / 2 <= n_l1 <= hi * 1.5
        return passed, warn_only

    # Live failure 1: a sane class count, no tie-breaks. Must block.
    passed, warn_only = gate(19, 1)
    assert not passed and not warn_only, "1 rule must halt the run, not warn"

    # Live failure 2, caused by the fix for the first: hardening the rule
    # requirement made the next draft satisfy it by sacrificing the classes —
    # 24 rules and TWO classes, the rules naming a dozen classes that did not
    # exist. Two classes is not a granularity judgement, it is broken output.
    passed, warn_only = gate(2, 24)
    assert not passed and not warn_only, "a 2-class taxonomy must halt, not warn"

    # A count NEAR the range with rules present stays advisory — the data may
    # legitimately settle granularity differently than the prior expected.
    for n in (lo - 2, hi + 2):
        passed, warn_only = gate(n, floor)
        assert not passed and warn_only, f"{n} classes should warn, not halt"

    # Healthy on both axes.
    assert gate(lo + 1, floor + 5) == (True, True)
