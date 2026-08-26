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
    # The BARE code, not the repaired sentence. `then` is documented as "the
    # class that wins" and everything downstream keys on it — `_dedupe_rules`
    # compares it with `==`, `rules_against_evidence` asks whether it equals the
    # referee's verdict. `选 EXAM_INFO` satisfies neither, so the repair path was
    # reintroducing the very shape `normalise_then` exists to remove.
    assert t.rules[0].then == "EXAM_INFO"
    assert "选 EXOD_INFO" in t.rules[0].rationale, "the original wording must survive"
    assert any("EXOD_INFO" in m for m in emitted), "the repair must be reported, not silent"

    # A target resembling nothing is dropped rather than guessed at.
    t2 = _tax(AdjudicationRule(id="R99", when="x", then="选 TOTALLY_UNRELATED"))
    health2 = _validate_rules(t2, _Deps)
    assert health2["n_dropped"] == 1 and not t2.rules

    # A rule that NAMES a real class but is not one is reduced, not repaired.
    # "Valid" used to mean "mentions a declared class", which `选 WORD_MEANING`
    # satisfies while remaining unusable as a key.
    t3 = _tax(AdjudicationRule(id="R01", when="x", then="选 WORD_MEANING"))
    health3 = _validate_rules(t3, _Deps)
    assert health3["n_repaired"] == 0 and health3["n_dropped"] == 0
    assert health3["n_then_reduced_to_code"] == 1
    assert t3.rules[0].then == "WORD_MEANING"
    assert "选 WORD_MEANING" in t3.rules[0].rationale

    # A rule already in the right shape is left byte-identical.
    t4 = _tax(AdjudicationRule(id="R02", when="x", then="WORD_MEANING", rationale="r"))
    health4 = _validate_rules(t4, _Deps)
    assert health4["n_then_reduced_to_code"] == 0 and health4["n_then_ambiguous"] == 0
    assert t4.rules[0].then == "WORD_MEANING" and t4.rules[0].rationale == "r"

    # A rule naming TWO classes ships as guidance but is not treated as a key.
    t5 = _tax(AdjudicationRule(id="R03", when="x",
                               then="有裁决框架的归 EXAM_INFO；否则归 WORD_MEANING。"))
    health5 = _validate_rules(t5, _Deps)
    assert health5["n_then_ambiguous"] == 1 and len(t5.rules) == 1, (
        "it is still readable guidance — dropping it would remove the boundary's help")


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
    # p2b must not EMIT this gate — that is what "precedes the gold set" means.
    # Reading the ceiling p2a already measured is the opposite of moving the gate
    # later: it reuses a cheap measurement instead of paying for it again, and
    # p2b needs it to tell "the guide is ambiguous" from "the bar exceeds what
    # this annotator can do against itself". Assert on the EMISSION, not on any
    # mention of the name.
    p2b_src = inspect.getsource(topdown.p2b_gold)
    assert 'deps.gate(\n        "p2a_pilot_agreement"' not in p2b_src
    assert '"p2a_pilot_agreement", "p2a"' not in p2b_src, \
        "the pilot gate must fire in 2a, before 2b spends anything"
    assert "_pilot_agreement(deps" not in p2b_src, \
        "and 2b must not re-run the pilot either"


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
    The "at the ceiling" side used to be `share_of_ceiling >= 0.90` — a constant
    from one corpus, deciding whether a run halts. It is now a test the pilot can
    answer itself: the slack a redraft could recover is `ceiling - kappa`, and it
    is worth acting on only when it exceeds the error on that difference.
    """
    from qmine.config import QMineConfig

    cfg = QMineConfig()

    def verdict(po: float, k: float, po2: float, k2: float, n: int = 200) -> bool:
        """The gate's own rule, recomputed from the same inputs it uses."""
        pe = (po - k) / (1 - k) if k < 1 else 0.0
        pe2 = (po2 - k2) / (1 - k2) if k2 < 1 else 0.0
        se = ((po * (1 - po) / n) ** 0.5) / max(1e-6, 1 - pe)
        se2 = ((po2 * (1 - po2) / n) ** 0.5) / max(1e-6, 1 - pe2)
        slack_is_real = (k2 - k) > 1.645 * (se ** 2 + se2 ** 2) ** 0.5
        return (k + 1.645 * se) >= cfg.gates.kappa or not slack_is_real

    # Both real halts must still halt. These are the measured pilots.
    assert not verdict(0.785, 0.761, 0.900, 0.8997), "live30 kappa 0.761 vs ceiling 0.900"
    assert not verdict(0.715, 0.688, 0.820, 0.8023), "live31 kappa 0.688 vs ceiling 0.802"

    # Clears the bar outright — ceiling is irrelevant when the target is met.
    assert verdict(0.95, 0.94, 0.96, 0.95)

    # Short of the bar, but the annotator barely beats itself: no rule the referee
    # writes can close that, so halting would demand a fix nobody can perform.
    assert verdict(0.83, 0.795, 0.84, 0.805)

    # And the point of doing it this way: the SAME slack decides differently
    # depending on whether it is bigger than its own error. That is the thing a
    # fixed share threshold could not express.
    assert verdict(0.83, 0.795, 0.845, 0.815, n=200)          # 0.020 ± 0.05 — noise
    assert not verdict(0.80, 0.700, 0.845, 0.815, n=200)      # 0.115 ± 0.05 — real


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


def test_every_adjudication_rule_the_architect_writes_reaches_the_annotator():
    """`TaxonomyNode.adjudication_rules` was write-only for the whole project.

    It is declared to hold rule *ids* cross-referencing the top-level list; the
    models fill it with whole rules; and nothing in `src/` ever read it. So the
    architect spent output tokens on per-class tie-breaks that reached no
    annotator, no referee and no classifier — 55 discarded in one live run, 24 in
    the next, while its shape gate reported "1 adjudication rules".
    """
    from qmine.graph.nodes.topdown import _render_rules
    from qmine.records import AdjudicationRule, Taxonomy, TaxonomyNode

    tax = Taxonomy(
        version="v1",
        nodes=[
            TaxonomyNode(code="A", name="a", level=1, definition="d", user_need="n",
                         adjudication_rules=["A vs B: prefer A when the marker is 读音", "R1"]),
            TaxonomyNode(code="B", name="b", level=1, definition="d", user_need="n",
                         adjudication_rules=["A vs B: prefer A when the marker is 读音"]),
        ],
        rules=[AdjudicationRule(id="R1", when="marker is 拼音", then="A", rationale="r")],
    )
    out = _render_rules(tax)
    assert "[R1]" in out, "structured rules must still render"
    assert "prefer A when the marker is 读音" in out, "node rules must reach the annotator"
    assert out.count("prefer A when the marker is 读音") == 1, "duplicated across nodes"
    assert out.count("R1") == 1, "a bare id is a cross-reference, not a second rule"
    assert out.splitlines()[0].startswith("- [R1]"), "citable ids come first, ahead of free text"


def test_a_halted_pilot_prescribes_a_remedy_and_names_which_kind():
    """Three live halts reported `n_prescriptions: 0` — the operator was told the
    taxonomy was wrong and not which part, nor which of two opposite fixes applied.

    The pilot already holds the answer and was discarding it. A pair one annotator
    cannot reproduce against ITSELF is a boundary not present in the data: merging
    or re-cutting is the only remedy, and a tie-break rule is wasted effort on it.
    A pair two annotators split while each stays self-consistent is precisely what
    a tie-break is for. Replayed on a real pilot the split was 36 queries structural
    against 27 guide — invisible in the aggregate kappa that halted the run.
    """
    import collections

    la = [{"label": x} for x in ["A", "A", "B", "C", "C", "D"]]
    lb = [{"label": x} for x in ["B", "A", "B", "D", "C", "D"]]   # differs at 0 and 3
    back = ["B", "A", "B", "C", "C", "D"]                          # A self-differs at 0, 2

    conf = collections.Counter(" × ".join(sorted((x["label"], y["label"])))
                               for x, y in zip(la, lb) if x["label"] != y["label"])
    structural = collections.Counter(" × ".join(sorted((x["label"], str(v))))
                                     for x, v in zip(la, back) if x["label"] != str(v))
    guide_only = {k: v for k, v in conf.items() if k not in set(structural)}

    assert "A × B" in structural, "A vs B is not reproducible by one annotator"
    assert "C × D" in guide_only, "C vs D is stable within an annotator, unstable across"
    assert "A × B" not in guide_only, "a structural pair must not also be prescribed a rule"


def test_the_architect_is_not_told_to_write_rules_it_does_not_write():
    """The prompt said "You are not writing the adjudication rules" and, twenty
    lines later, "YOU MUST return at least {{MIN_RULES}} adjudication rules".

    It obeyed the second: 41,808 output tokens, rules emitted anyway, and the
    class list starved of the budget it was supposed to receive.
    """
    from pathlib import Path as P

    prompt = P("src/qmine/agents/prompts/architect.md").read_text()
    assert "not writing the adjudication rules" in prompt
    assert "MIN_RULES" not in prompt, "the architect is still ordered to write rules"


def test_the_catchall_instruction_does_not_ask_for_the_defect_it_causes():
    """The prompt required a catch-all "defined by what it *is*, not by what it is
    not". A catch-all named for a content area competes with the named classes for
    the same queries — measured as 25% of one pilot's inter-annotator disagreement.
    A catch-all can only be defined by exclusion."""
    from pathlib import Path as P

    prompt = P("src/qmine/agents/prompts/architect.md").read_text()
    assert "defined by what it *is*, not by what it is not" not in prompt
    assert "none of the above" in prompt
    assert "exclusion" in prompt


def test_a_gate_with_no_remaining_move_proceeds_rather_than_stopping_dead():
    """The slack test exists to trigger a remedy; the pipeline has one.

    Measured on `live35`: kappa 0.844 against a 0.9243 ceiling left 0.080 of
    statistically significant slack, the redraw rewrote the six worst boundaries,
    and kappa FELL to 0.827 — so it was reverted and the slack stood. Halting
    then asks the operator to do by hand what the pipeline just failed at, while
    kappa 0.844 is above the reliability floor and the labels would serve.

    The condition is deliberately narrow: it requires the remedy to have RUN and
    FAILED. A run that never attempted a redraw still halts, because it has a
    move left to make.
    """
    from qmine.config import QMineConfig

    cfg = QMineConfig()
    floor = cfg.gates.annotator_fitness_kappa

    def gate(kappa, upper, ceiling_ok, slack_sig, history):
        attempted = bool(history)
        helped = any(h.get("kept") for h in history)
        exhausted = attempted and not helped
        usable = exhausted and kappa >= floor
        at_ceiling = not slack_sig
        return ceiling_ok and (upper >= cfg.gates.kappa or at_ceiling or usable)

    reverted = [{"kept": False}]
    kept = [{"kept": True}]

    # live35 exactly: good kappa, real slack, redraw tried and reverted.
    assert gate(0.844, 0.888, True, True, reverted)

    # Never attempted — there is still a move to make, so it must halt.
    assert not gate(0.844, 0.888, True, True, [])

    # The redraw HELPED, so the loop stopped for another reason; the slack test
    # still governs and this must halt.
    assert not gate(0.844, 0.888, True, True, kept)

    # Below the reliability floor the labels cannot support conclusions, and no
    # amount of "we tried" changes that.
    assert not gate(0.71, 0.77, True, True, reverted)

    # An unfit annotator halts regardless.
    assert not gate(0.844, 0.888, False, True, reverted)


def test_prescriptions_read_as_limitations_when_the_gate_let_the_run_proceed():
    """A prescription's FORCE depends on what the gate decided.

    Blocked, they are work that must happen. Passed with residual slack, the same
    measurements are a limitation to report — telling an operator to "merge them"
    prescribes work the pipeline already tried, measured, and declined to act on.

    This regressed the moment it was written: the check read
    `getattr(pilot_gate, "passed", False)`, but `GateResult` carries `status`, so
    the default silently chose the false branch and a PASSED gate still printed
    "outstanding actions" on live38.
    """
    from qmine.graph.nodes.topdown import _gate_let_the_run_proceed
    from qmine.records import GateResult

    def g(status):
        return GateResult(name="p2a_pilot_agreement", phase="p2a", status=status)

    assert _gate_let_the_run_proceed(g("passed")) is True
    assert _gate_let_the_run_proceed(g("warned")) is True, \
        "a warned gate is non-blocking — the run proceeds, so these are limitations"
    assert _gate_let_the_run_proceed(g("failed")) is False
    assert _gate_let_the_run_proceed(g("pending")) is False

    # The defect itself: `passed` is not a field, so anything reading it with a
    # default silently reports "the gate did not pass" for a gate that did.
    assert not hasattr(g("passed"), "passed"), (
        "GateResult carries `status`; if a `passed` field is ever added, revisit "
        "every getattr default that was papering over its absence")


# --- the gold set must not silently lose the referee's work -----------------


def test_a_row_nobody_labelled_is_not_agreement():
    """`UNLABELED == UNLABELED` satisfied the equality check.

    `agreement()` already excludes UNLABELED from kappa, so the METRIC was safe
    while the GOLD SET was not: a row both annotators omitted was recorded
    agreed with `final="UNLABELED"`, never reached the referee, and passed p2c's
    non-empty filter into the classifier as a real class.
    """
    import inspect

    from qmine.graph.nodes import topdown

    src = inspect.getsource(topdown.p2b_gold)
    assert "missing = UNLABELED in (la, lb)" in src, \
        "a row missing a label must be excluded, not counted as agreement"
    assert "UNLABELED not in (r.label_a, r.label_b)" in src, \
        "and must not be sent to the referee, which has no second opinion to weigh"


def test_guide_repair_on_a_fresh_sample_keeps_the_refereed_rows():
    """Round 2 annotates DIFFERENT queries, so replacing the list is pure loss.

    It discarded ~3,000 round-1 rows including every referee verdict, and
    substituted a set whose only labelled rows are ones both annotators already
    agreed on — the referee runs before repair and never sees round 2. The gold
    set became agreement-only, i.e. systematically the easy rows, which inflates
    every classifier number computed from it.
    """
    import inspect

    from qmine.config import QMineConfig
    from qmine.graph.nodes import topdown

    assert QMineConfig().taxonomy.repair_on_fresh_sample is True, \
        "the merge path is the default path; if this flips, re-read the branch"

    src = inspect.getsource(topdown.p2b_gold)
    assert "rows = rows + added_rows" in src, \
        "disjoint samples must be merged, never swapped"
    # Match whole statements, not substrings: `rows = repair_meta["rows"]` is a
    # substring of the legitimate `fresh_rows = repair_meta["rows"]`, and a loose
    # `not in` check fails on the correct code.
    statements = {ln.strip() for ln in src.splitlines()}
    assert 'rows = repair_meta["rows"]' not in statements, \
        "the unconditional replacement is what discarded the adjudications"
    assert any("refereed rows kept" in ln for ln in src.splitlines()), \
        "and the count kept must be reported, not assumed"


def test_an_adversary_that_returns_nothing_cannot_report_perfect_accuracy():
    """`n = max(len(verdicts), 1)` made an EMPTY response score 1.000.

    `1 - 0/1` is a perfect accuracy produced by a provider failure. The
    denominator must be the rows that actually got a verdict, the coverage must
    be reported beside it, and no verdicts at all must be undefined — never
    flattering.
    """
    import inspect

    from qmine.graph.nodes import topdown

    src = inspect.getsource(topdown.p2d_validate)
    statements = {ln.strip() for ln in src.splitlines()}
    assert "n = max(len(verdicts), 1)" not in statements, \
        "this is the expression that turned an empty response into 1.000"
    assert "est = None if not n_verdicts else 1 - wrong / n_verdicts" in statements, \
        "no verdicts must mean undefined, not perfect"
    assert "coverage = n_verdicts / max(n_attacked, 1)" in statements, \
        "and the share of attacked rows that answered must travel with the number"

    # The phase must also survive a provider failure rather than crashing the run.
    assert "adversary chunk failed" in src, \
        "an unguarded call here ended the whole run on any blip"


def test_the_annotator_prompt_keeps_the_newest_rules_not_just_the_oldest():
    """Both blocks are APPENDED to, so a head-only trim cuts the newest content.

    The referee's rules arrive via `taxonomy.rules.extend(new_rules)` and the
    guide-repair round appends its 边界裁定 section to the guide — the most
    authoritative lines in each block, written in response to observed
    disagreement, and they sit LAST. Measured: 72 rules render to 8,128 chars,
    90% of the old 9,000 budget, before the referee adds any.
    """
    import inspect

    from qmine.agents.roles import AnnotatorAgent, budget_text

    src = inspect.getsource(AnnotatorAgent.build_user)
    assert "budget_text(rules, 9000)" not in src, \
        "a head-only trim on an appended block discards the referee's own rules"
    for block in ("rules", "guide", "classes"):
        assert f"budget_text({block}," in src and "tail=" in src, \
            f"the {block} block must retain its tail"

    # And the helper must actually keep both ends when asked.
    text = "HEAD" + ("x" * 5000) + "TAIL"
    out = budget_text(text, 1000, tail=200)
    assert out.startswith("HEAD"), "the head must survive"
    assert out.endswith("TAIL"), "and so must the tail — that is the whole point"


def test_adjudication_coverage_counts_only_rows_the_referee_was_given():
    """`not r.agreed` also matches rows excluded for a MISSING label.

    Those never entered `disagreements`, so counting them as unresolved
    subtracts them from a denominator they were never in: live38 reported
    "adjudicated 450/483" while 459 rows were in fact adjudicated. A coverage
    figure has to use the same population on both sides of the fraction.
    """
    import inspect

    from qmine.graph.nodes import topdown

    src = inspect.getsource(topdown.p2b_gold)
    i = src.index("n_unresolved = sum(")
    expr = src[i:i + 260]
    assert "UNLABELED not in (r.label_a, r.label_b)" in expr, \
        "rows the referee never saw must not count against its coverage"


# --- p2b must ask the same question p2a already learned to ask ---------------


def test_the_two_kappa_gates_do_not_contradict_each_other():
    """`live38`: p2a PASSED at kappa 0.824 and p2b HALTED at 0.832.

    Same corpus, same annotators, same 0.9228 ceiling — and the HIGHER number
    blocked, because p2a reasons about the annotator's own self-consistency
    while p2b used a flat 0.90. That bar demands 97.5% of what one annotator
    manages against ITSELF, which is the condition p2a's own comment describes
    as making a gate unpassable by a perfect guide.
    """
    import inspect

    from qmine.graph.nodes import topdown

    src = inspect.getsource(topdown.p2b_gold)
    statements = {ln.strip() for ln in src.splitlines()}
    assert 'passed=(not unsound) and agree["kappa"] >= cfg.gates.kappa,' not in statements, \
        "a flat bar here contradicts p2a on identical data"
    for concept in ("target_above_ceiling", "usable_despite_slack", "annotator_fit"):
        assert concept in src, f"p2b must weigh {concept}, as p2a does"


def test_p2b_still_fails_when_agreement_is_genuinely_poor():
    """The escape hatch must not become an unconditional pass.

    A gate that cannot fail is not a gate — the same invariant the pilot has.
    """
    import inspect

    from qmine.graph.nodes import topdown

    src = inspect.getsource(topdown.p2b_gold)
    # Below the reliability floor, `usable_despite_slack` is False by
    # construction, so a poor kappa still blocks.
    assert 'agree["kappa"] >= cfg.gates.annotator_fitness_kappa' in src, \
        "proceeding on residual slack must still require the reliability floor"
    # And an unfit ANNOTATOR blocks regardless of what the guide work achieved.
    assert "annotator_fit and (" in src, \
        "no amount of guide repair fixes an annotator that cannot reproduce itself"
    assert "(not unsound)" in src, \
        "a measurement on collapsed coverage must still block"


# --- the referee's rules must reach the annotator ---------------------------
#
# There is already a test that the ARCHITECT's rules reach the annotator. The
# REFEREE's were unguarded, and on live38 all 83 of them were silently truncated
# out of the guide-repair round whose entire purpose was to apply them.


def _annotator_budget(block: str) -> tuple[int, int]:
    """The (budget, tail) the annotator prompt applies to a named block.

    Read out of the source rather than hardcoded in each test, so revising a
    budget does not break every test that merely needs to know what it is.
    """
    import inspect
    import re

    from qmine.agents import roles

    src = inspect.getsource(roles.AnnotatorAgent.build_user)
    m = re.search(rf"budget_text\({block},\s*(\d+)(?:,\s*tail=(\d+))?", src)
    assert m, f"no budget_text call found for {block!r}"
    return int(m.group(1)), int(m.group(2) or 0)


def test_every_rule_the_referee_drafts_reaches_the_annotator():
    """83 referee rules pushed the block to 18,496 chars against a 9,000 budget.

    They are APPENDED, so a head-limited budget discards exactly them. Two
    defences, both required: the budget fits the observed set, and rendering is
    newest-round-first so that if it is ever exceeded the rules lost are the
    oldest rather than the freshest.
    """
    import inspect

    from qmine.agents import roles
    from qmine.graph.nodes.topdown import _render_rules
    from qmine.memory.context import budget_text
    from qmine.records import AdjudicationRule, Taxonomy, TaxonomyNode

    nodes = [TaxonomyNode(code=f"C{i}", name=f"c{i}", level=1, definition="d" * 120,
                          user_need="n", positive_examples=["x"], negative_examples=[])
             for i in range(22)]
    old_rules = [AdjudicationRule(id=f"R{i:03d}", when="w" * 60, then="C1",
                                  rationale="r" * 40, classes=["C1", "C2"],
                                  added_in_round=0) for i in range(50)]
    ref_rules = [AdjudicationRule(id=f"REF{i:03d}", when="w" * 60, then="C2",
                                  rationale="r" * 40, classes=["C1", "C2"],
                                  added_in_round=1) for i in range(83)]
    tax = Taxonomy(version="v1", nodes=nodes, rules=old_rules + ref_rules)

    rendered = _render_rules(tax)
    # The budget the annotator prompt actually applies.
    # Assert the PROPERTY, not the literal. A budget is a number that will be
    # revised; what must stay true is that it exceeds what the pipeline actually
    # produces. Pinning "20000" made this test fail the moment the budget was
    # correctly widened.
    budget, tail = _annotator_budget("rules")
    assert budget >= 46_814, (
        f"rules budget {budget:,} is below the 46,814 chars live38 measured — "
        "the referee's rules would be truncated out of the prompt again")

    kept = budget_text(rendered, budget, tail=tail)
    kept_ids = {ln.split("]")[0].strip("- [") for ln in kept.splitlines()
                if ln.startswith("- [")}
    missing = [r.id for r in ref_rules if r.id not in kept_ids]
    assert not missing, (
        f"{len(missing)} of {len(ref_rules)} referee rules never reach the "
        f"annotator — the guide-repair round cannot apply what it cannot see")


def test_referee_rules_survive_first_when_the_budget_is_exceeded():
    """Newest-first ordering is the defence that does not depend on a magic number."""
    from qmine.graph.nodes.topdown import _render_rules
    from qmine.records import AdjudicationRule, Taxonomy, TaxonomyNode

    tax = Taxonomy(
        version="v1",
        nodes=[TaxonomyNode(code="A", name="a", level=1, definition="d",
                            user_need="n", positive_examples=["x"], negative_examples=[])],
        rules=[AdjudicationRule(id="OLD", when="w", then="A", rationale="r",
                                classes=["A"], added_in_round=0),
               AdjudicationRule(id="NEW", when="w", then="A", rationale="r",
                                classes=["A"], added_in_round=2)])
    lines = [ln for ln in _render_rules(tax).splitlines() if ln.startswith("- [")]
    assert lines[0].startswith("- [NEW]"), \
        "the freshest, most evidence-driven rule must not be the one truncated"


def test_p2b_persists_the_taxonomy_the_referee_produced():
    """The referee's rules and the repaired guide were in-memory only.

    `deps.taxonomy()` already prefers `taxonomy_v2`; nothing wrote it, so a
    resumed run recovered the PRE-referee taxonomy and the deliverable reported
    50 rules when the run had 133.
    """
    import inspect

    from qmine.graph.nodes import topdown

    src = inspect.getsource(topdown.p2b_gold)
    assert 'put_json(\n        "taxonomy_v2"' in src or '"taxonomy_v2",' in src, \
        "p2b must persist the taxonomy it actually produced"
    assert '"taxonomy_v2": tax_v2_ref' in src, \
        "and return it as an artifact so downstream and resume can find it"


def test_the_guide_cites_the_referees_rules_instead_of_repeating_them():
    """The annotator was receiving the referee's rules TWICE.

    `_guide_with_decisions` rendered each rule's full text into the 边界裁定
    section, while the same rules were already sent in `## Adjudication rules`.
    On live38 that duplicated ~24,000 chars of every annotation prompt — and
    BOTH copies were then truncated for being oversized, so 75% of the binding
    rulings were cut from the round whose whole purpose was to apply them.
    """
    import itertools

    from qmine.graph.nodes.topdown import _guide_with_decisions
    from qmine.records import AdjudicationRule

    names = [f"LOOKUP_SOMETHING_RATHER_LONG_{i:02d}" for i in range(22)]
    pairs = list(itertools.combinations(names, 2))[:82]
    added, decisions = [], []
    for i, (a, b) in enumerate(pairs):
        added.append(AdjudicationRule(
            id=f"R{500+i}", when="当查询" + "详" * 370, then=a,
            rationale="drafted by the referee", classes=[a, b], added_in_round=1))
        decisions.append({"pair": [a, b], "decided": True})

    out = _guide_with_decisions("原始指南", decisions, added)
    assert "详" * 50 not in out, \
        "the guide must not repeat rule text that the rules block already carries"
    for r in added[:5]:
        assert f"[{r.id}]" in out, "it must cite the rule id so the ruling is traceable"
        assert f"**{r.then}**" in out, "and state which class wins — that is the binding part"

    # It has to FIT, or the binding rulings are what gets cut.
    budget, _tail = _annotator_budget("guide")
    assert len(out) < budget, (
        f"82 binding rulings render to {len(out):,} chars against a {budget:,} "
        "budget — if this ever exceeds it, the rulings are truncated again")


def test_an_undecided_boundary_is_still_reported_as_open():
    """Honest openness beats a fabricated ruling — the pair must still appear."""
    from qmine.graph.nodes.topdown import _guide_with_decisions

    out = _guide_with_decisions("g", [{"pair": ["A", "B"], "decided": False}], [])
    assert "A × B" in out and "开放边界" in out, \
        "a boundary the evidence could not settle must be named, not silently dropped"


def test_a_guide_repair_that_lowers_agreement_is_reverted():
    """The redraw has had this guard since it was built; the repair had none.

    Measured on live38 gen05 — the FIRST run in which the repair's own rules
    actually reached the annotator, earlier ones having truncated them out of the
    prompt: kappa 0.822 -> 0.794, about 3.5 standard errors at n≈2978, and below
    the 0.80 reliability floor. The repair turned a gate that would have passed
    into a halt, and nothing undid it.
    """
    import inspect

    from qmine.graph.nodes import topdown

    src = inspect.getsource(topdown.p2b_gold)
    assert "guide_before_repair = taxonomy.labeling_guide" in src, \
        "the repair can overwrite the guide, so the guide must be snapshotted first"
    assert "if after_k < before_k:" in src, \
        "a repair that lowers agreement must be undone, not merely regretted"
    assert "taxonomy.labeling_guide = guide_before_repair" in src
    assert "guide repair lowered kappa" in src, "and the revert must be announced"
    # The fresh rows are real annotations; only the unproven GUIDE reverts.
    assert "the fresh rows are kept" in src, \
        "reverting the guide must not throw away 3,000 genuine annotations"


def test_the_repair_revert_mirrors_the_redraw_revert():
    """Two loops, one rule: do not adopt a change you cannot show is an improvement."""
    import inspect

    from qmine.graph.nodes import topdown

    redraw = inspect.getsource(topdown._redraw_until_stable)
    gold = inspect.getsource(topdown.p2b_gold)
    assert "reverting to the previous taxonomy" in redraw
    assert "reverting the guide and rules" in gold, \
        "the guide repair needs the same guard the redraw already has"


def test_a_gate_that_passed_with_slack_says_so_to_the_reader():
    """`通过` is not `没有保留`, and the ledger used to hide the difference.

    `p2b_kappa` on live38 rendered as ✅ 通过 with 实测 `kappa`=0.8221 beside
    门槛 `kappa`=0.9 and no explanation — while the gate's own message read
    "PROCEEDING WITH RESIDUAL SLACK … above the 0.8 reliability floor but short of
    0.9; every downstream number …". The caveat was computed on every run and
    printed only for gates that FAILED, so it reached no reader on any run that
    succeeded.
    """
    from types import SimpleNamespace

    from qmine.report.zh_bottomup import _gate_ledger, _passed_below_threshold

    def gate(**kw):
        d = dict(phase="p2b", status="passed", blocking=True, message="", remediation="",
                 observed={}, threshold={})
        d.update(kw)
        return SimpleNamespace(**d)

    slack = gate(observed={"kappa": 0.8221, "n": 2991}, threshold={"kappa": 0.9},
                 message="PROCEEDING WITH RESIDUAL SLACK — 0.822 is short of 0.9")
    assert _passed_below_threshold(slack) is True

    assert _passed_below_threshold(gate(observed={"kappa": 0.93},
                                        threshold={"kappa": 0.9})) is False
    # `max_` thresholds run the other way round.
    assert _passed_below_threshold(gate(observed={"noise_rate": 0.4},
                                        threshold={"max_noise_rate": 0.3})) is True
    assert _passed_below_threshold(gate(observed={"noise_rate": 0.1},
                                        threshold={"max_noise_rate": 0.3})) is False
    # A threshold about one quantity must never be checked against another's value.
    assert _passed_below_threshold(gate(observed={"coherence": 3.59},
                                        threshold={"max_weak_share": 0.15})) is False
    # A failing gate is already visible; it is not a "slack" case.
    assert _passed_below_threshold(gate(status="failed", observed={"kappa": 0.5},
                                        threshold={"kappa": 0.9})) is False

    md = _gate_ledger({"gates": {"p2b_kappa": slack}})
    assert "带保留通过" in md, "the table does not flag a pass that sits under its bar"
    assert "RESIDUAL SLACK" in md, "the gate's own message still does not reach the reader"
