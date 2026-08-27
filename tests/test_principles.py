"""Tests for the playbook principles that this codebase claims to *enforce*.

These are the highest-value tests in the suite. A clustering bug produces worse
numbers; a failure here produces numbers that look fine and are not trustworthy,
which is far more expensive. Each test names the principle it guards.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

import numpy as np
import pytest

from qmine.memory.context import BlindnessFirewall, BlindnessViolation, render_card
from qmine.ops.governance import GovernanceError, apply_merges, assert_all_settled, execute_prescriptions
from qmine.ops.panel import UniformPanel
from qmine.records import METRIC_AUTHORITY, MetricRecord, NamingCard, Prescription, Taxonomy, TaxonomyNode


# -- Principle 3: metrics must not betray the objective --------------------

def test_silhouette_cannot_decide_through_the_panel():
    """An advisory metric must be structurally incapable of selecting a candidate
    *through the panel API*.

    Note the scope. This test passed for months while two code paths chose k by
    silhouette alone — they call `cosine_silhouette` directly and never touch the
    panel, so this guard rail protected a door neither of them walked through.
    `test_local_k_is_not_decided_by_silhouette_alone` covers those.
    """
    panel = UniformPanel(600, subsample=300)
    panel.add_external("a", "silhouette", 0.9)
    panel.add_external("b", "silhouette", 0.1)
    with pytest.raises(ValueError, match="cannot decide"):
        panel.decisive_ranking("silhouette")


def test_local_k_is_not_decided_by_silhouette_alone():
    """Where silhouette DOES rank — choosing k inside one fixed representation —
    it must still clear two tests it cannot itself supply.

    Its bias is a constant offset within a single space, so its variation is
    informative there and it keeps a real vote. What it cannot do is say "this
    group has no structure" (it is undefined at k=1) or notice that its own lead
    is inside its noise. A shipped sub-intent took k=6 at silhouette 0.0749 /
    replay ARI 0.533 over k=2 at 0.0696 / 0.973 — 0.005 of silhouette bought with
    0.44 of reproducibility.
    """
    import numpy as np

    from qmine.ops.cluster import choose_local_k

    rs = np.random.RandomState(0)

    # A group with no structure must not be split at all.
    noise = rs.normal(0, 1, (900, 32))
    noise /= np.linalg.norm(noise, axis=1, keepdims=True)
    flat = choose_local_k(noise, max_k=8, min_size=60)
    assert flat["k"] == 1, f"split structureless data into {flat['k']}: {flat}"
    assert "structureless reference" in flat["rejected_because"]

    # A group with real structure must be split, and at the right granularity.
    cent = np.eye(3, 32)
    real = np.vstack([c + rs.normal(0, 0.18, (300, 32)) for c in cent])
    real /= np.linalg.norm(real, axis=1, keepdims=True)
    good = choose_local_k(real, max_k=8, min_size=60)
    assert good["k"] == 3, f"expected 3 groups, got {good['k']}"
    assert good["chosen"]["lift_over_null"] > 0.02

    # Every candidate is disclosed, including what silhouette alone would have done.
    assert good["candidates"] and all(
        {"k", "silhouette", "stability_ari", "lift_over_null"} <= set(c)
        for c in good["candidates"]
    )
    assert "silhouette_would_have_chosen" in good


def test_decisive_metrics_can_decide():
    panel = UniformPanel(600, subsample=300)
    panel.add_external("a", "stability_ari", 0.5)
    panel.add_external("b", "stability_ari", 0.9)
    assert panel.decisive_ranking("stability_ari")[0]["subject"] == "b"


def test_lower_is_better_metrics_rank_ascending():
    panel = UniformPanel(600, subsample=300)
    panel.add_external("tight", "template_fragmentation", 1.2)
    panel.add_external("shattered", "template_fragmentation", 3.4)
    assert panel.decisive_ranking("template_fragmentation")[0]["subject"] == "tight"


def test_authority_table_covers_every_metric_the_panel_emits():
    for name in ("stability_ari", "template_fragmentation", "silhouette",
                 "heldout_reproduction", "nmi_reference", "ambiguous_rate"):
        assert name in METRIC_AUTHORITY


# -- Principle 5: blind review ---------------------------------------------

def test_firewall_blocks_an_annotation_field_smuggled_onto_a_card():
    """The check that actually stops anchoring: a card may carry member queries
    and n-grams, and nothing else. A `legacy_label` or `taxonomy_hint` field
    cannot pass whatever it contains."""
    fw = BlindnessFirewall().add_reference_labels(["怎么读/读音/拼音"])
    card = {
        "leaf_id": 1, "size": 10, "share": 0.1,
        "center_samples": ["氢怎么读"], "random_samples": [], "edge_samples": [],
        "top_ngrams": ["怎么读"], "length_stats": {},
        "legacy_label": "怎么读/读音/拼音",          # <- the smuggled annotation
    }
    with pytest.raises(BlindnessViolation, match="not part of the blind card contract"):
        fw.assert_card_blind(card)


def test_firewall_does_not_flag_a_domain_word_appearing_in_a_real_query():
    """The false positive that made a lexical-only firewall unusable.

    Good category names come from their domain's own vocabulary, so legacy
    labels and ordinary query words overlap. On the real corpus the legacy label
    "作文" appears inside the genuine query "我的自画像作文350字". Treating that
    row as a leak silently dropped ten clusters from the naming pass.

    A member query cannot anchor a namer — it is the thing being judged.
    """
    fw = BlindnessFirewall().add_reference_labels(["作文", "翻译", "成语"])
    card = NamingCard(
        leaf_id=1, size=900, share=0.02,
        center_samples=["我的自画像作文350字", "六一作文450字左右"],
        random_samples=["三年级作文我的妈妈"], edge_samples=["手抄报模板"],
        top_ngrams=["作文", "0字"],
    )
    rendered = render_card(card, firewall=fw)      # must not raise
    assert "我的自画像作文350字" in rendered


def test_firewall_blocks_legacy_labels():
    fw = BlindnessFirewall().add_reference_labels(["生肖/打一(疑似博彩)", "怎么读/读音/拼音"])
    with pytest.raises(BlindnessViolation):
        fw.assert_blind({"note": "this cluster is 生肖/打一(疑似博彩)"})


def test_firewall_blocks_peer_agent_outputs():
    from qmine.records import LeafNaming

    peer = LeafNaming(leaf_id=1, name_zh="汉字组词查询", code="word_formation",
                      user_need="x", coherence=5)
    fw = BlindnessFirewall().add_peer_outputs([peer])
    with pytest.raises(BlindnessViolation):
        fw.assert_blind({"hint": "probably 汉字组词查询"})


def test_firewall_allows_a_clean_card_through():
    tax = Taxonomy(nodes=[TaxonomyNode(code="PRONUNCIATION", name="读音查询", definition="x")])
    fw = BlindnessFirewall().add_taxonomy(tax)
    clean = NamingCard(leaf_id=1, size=10, share=0.1,
                       center_samples=["氢怎么读", "钦州的拼音"], top_ngrams=["怎么读"])
    assert "氢怎么读" in render_card(clean, firewall=fw)


def test_firewall_still_catches_labels_in_non_corpus_payloads():
    """Exempting corpus text must not disarm the general check — an auditor
    prompt or a memory block carrying a taxonomy name is still a leak."""
    tax = Taxonomy(nodes=[TaxonomyNode(code="PRONUNCIATION", name="读音查询", definition="x")])
    fw = BlindnessFirewall().add_taxonomy(tax)
    with pytest.raises(BlindnessViolation):
        fw.assert_blind({"context_note": "this cluster is probably 读音查询"})


def test_send_payload_is_the_workers_entire_state():
    """The structural half of the anti-anchoring guarantee.

    A Send worker must not be able to read parent state. If this ever regresses
    in LangGraph, blind naming silently stops being blind.
    """
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import Send

    def fan(state):
        return [Send("w", {"cid": i}) for i in range(3)]

    def w(payload):
        return {"seen": [sorted(payload.keys())]}

    g = StateGraph(_SendProbeState)
    g.add_node("start", lambda s: {})
    g.add_node("w", w)
    g.add_conditional_edges("start", fan, ["w"])
    g.add_edge("w", END)
    g.add_edge(START, "start")
    out = g.compile().invoke({"secret_labels": ["LEAK"], "seen": []})
    assert all(keys == ["cid"] for keys in out["seen"]), out["seen"]


class _SendProbeState(TypedDict, total=False):
    """Module-level so the TypedDict's forward references resolve."""

    secret_labels: list[str]
    seen: Annotated[list, operator.add]


# -- Principle 6: governance is executed, not recorded ---------------------

def test_unexecuted_prescription_fails_the_gate():
    ps = [Prescription(id="P1", kind="merge_families", targets=[1, 2])]
    with pytest.raises(GovernanceError, match="never reached the data"):
        assert_all_settled(ps)


def test_executed_and_declined_both_count_as_settled():
    ps = [
        Prescription(id="P1", kind="merge_families", targets=[1, 2], status="executed"),
        Prescription(id="P2", kind="keep_as_is", targets=[3], status="declined",
                     decline_reason="a real distinction"),
    ]
    assert assert_all_settled(ps)["n_total"] == 2


def test_merge_chains_resolve_to_a_root():
    lf = np.arange(12)
    merged, detail = apply_merges(lf, {10: 6, 6: 0})
    assert merged[10] == merged[6] == merged[0]
    assert detail["n_families_after"] == 10


def test_execution_stamps_evidence_on_every_prescription():
    ps = [Prescription(id="P1", kind="merge_families", targets=[1, 2])]
    _, ps, _ = execute_prescriptions(ps, np.arange(6), metrics_before={"m": 1.0},
                                     recompute=lambda f: {"m": 0.5})
    assert ps[0].status == "executed"
    assert ps[0].evidence["column"] == "family_final"
    assert ps[0].evidence["metric_deltas"]["m"] == -0.5


# -- Principle 7: uniform panel, deterministic display ---------------------

def test_panel_refuses_to_mix_measurement_configurations():
    p1 = UniformPanel(1000, subsample=500, seed=0)
    p2 = UniformPanel(1000, subsample=200, seed=7)
    assert p1.panel_id != p2.panel_id
    p1.add_external("a", "silhouette", 0.5)
    p1._sets["b"] = p2.__class__(1000, subsample=200, seed=7).sets().get("x") or __import__(
        "qmine.records", fromlist=["MetricSet"]
    ).MetricSet(subject="b", panel_id=p2.panel_id)
    with pytest.raises(ValueError, match="different panels"):
        p1.comparison_table()


def test_exemplar_selection_is_a_pure_function_of_the_hit_set():
    from qmine.determinism import median_index_exemplar

    hits = [93, 4, 17, 62, 31]
    assert median_index_exemplar(hits) == median_index_exemplar(list(reversed(hits))) == 31


def test_subsampling_is_reproducible():
    from qmine.determinism import deterministic_subsample

    a = deterministic_subsample(10_000, 500, 0)
    b = deterministic_subsample(10_000, 500, 0)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, deterministic_subsample(10_000, 500, 1))


# -- Principle 12: honest reporting ----------------------------------------

def test_distillation_metric_carries_its_own_disclaimer():
    m = MetricRecord.make("distill_accuracy", 0.95, note="measures learnability, not correctness")
    assert m.authority == "diagnostic"
    assert "learnab" in m.note.lower()


def test_panel_footnotes_state_what_the_numbers_do_not_mean():
    """The three caveats must be present AND point the right way.

    This test used to assert the literal string "negatively correlated", which
    froze an inverted footnote in place: the panel told readers fragmentation
    falls as clusters rise, while the sentence immediately after it explained the
    opposite mechanism ("more clusters, more places to fragment into"). Measured
    across live38's own panel rows the correlation is **+0.901 Pearson**.

    Pinning wording rather than meaning is how a wrong caveat survives a test
    suite, so this asserts the direction and rejects the inverted phrasing.
    """
    notes = " ".join(UniformPanel(100, subsample=50).footnotes()).lower()
    assert "advisory" in notes
    assert "learnab" in notes
    assert "correlated with cluster count" in notes, "the fairness caveat is gone"
    assert "positively correlated" in notes
    assert "negatively correlated" not in notes, (
        "fragmentation rises with cluster count; the inverted wording is back"
    )


def test_an_agent_cannot_declare_its_own_prescription_executed():
    """Status and evidence belong to the pipeline, not the proposer.

    A real failure: the auditor returned a prescription already marked
    `executed` with an empty evidence pointer, which is exactly the
    "recommended but never applied" state the Phase 8 gate exists to catch —
    arriving through the gate's own front door.
    """
    from qmine.ops.governance import assert_all_settled

    claimed = Prescription(id="P1", kind="split_leaf", targets=[0], status="executed",
                           executed_at=0.62, evidence={})
    # the ingest path must reset it
    claimed.status, claimed.executed_at, claimed.evidence = "proposed", None, {}
    with pytest.raises(Exception, match="never reached the data"):
        assert_all_settled([claimed])


def test_split_prescriptions_are_executed_or_declined_with_a_reason(toy_embedding):
    """An audit finding must never silently vanish, even when the mechanism to
    apply it is unavailable."""
    import numpy as np

    from qmine.ops.cluster import kmeans_labels
    from qmine.ops.governance import execute_prescriptions

    labels = kmeans_labels(toy_embedding, 3, seed=0)
    fam = np.arange(3)
    ps = [Prescription(id="P1", kind="split_leaf", targets=[0])]

    # with the data available: executed, pointing at the column it changed
    _, done, detail = execute_prescriptions(list(ps), fam, X=toy_embedding, leaf_labels=labels)
    assert done[0].status == "executed"
    assert done[0].evidence["column"] == "bu_leaf"
    assert "leaf_labels" in detail

    # without it: declined, with a reason — never dropped
    _, done2, _ = execute_prescriptions(
        [Prescription(id="P1", kind="split_leaf", targets=[0])], fam)
    assert done2[0].status == "declined" and done2[0].decline_reason


def test_no_prescription_kind_can_slip_through_unhandled():
    """Exhaustive by construction: an unknown kind is declined with a reason,
    never left `proposed` (which halts the run) and never dropped (which is the
    failure Principle 6 names). A `relabel` with no replacement name is the case
    that found this."""
    import numpy as np

    from qmine.ops.governance import assert_all_settled, execute_prescriptions

    ps = [
        Prescription(id="P1", kind="relabel", targets=[2, 4]),                     # no names
        Prescription(id="P2", kind="relabel", targets=[1], target_names=["读音查询"]),
        Prescription(id="P3", kind="keep_as_is", targets=[0]),
    ]
    _, done, detail = execute_prescriptions(ps, np.arange(6))
    assert assert_all_settled(done)["n_total"] == 3
    by_id = {p.id: p for p in done}
    assert by_id["P1"].status == "declined" and by_id["P1"].decline_reason
    assert by_id["P2"].status == "executed"
    assert by_id["P2"].evidence["column"] == "bu_leaf_name"
    assert detail["relabelled"] == {"1": "读音查询"}


def test_the_panel_measures_the_partition_it_labels():
    """`stability_ari` is the metric the methodology calls decisive, and the panel
    attaches it to delivered candidates — leaves after refinement, families after
    governance. It used to be computed as `replay_stability(X, k)`, a function of
    the corpus and the cluster count only, so two structurally different partitions
    with the same k received identical 'stability' and the delivered tree was
    described by a number belonging to a fresh KMeans run."""
    import numpy as np

    from qmine.ops.panel import UniformPanel

    rs = np.random.RandomState(0)
    cent = np.eye(4, 24)
    X = np.vstack([c + rs.normal(0, 0.16, (250, 24)) for c in cent])
    X /= np.linalg.norm(X, axis=1, keepdims=True)

    truth = np.repeat(np.arange(4), 250)
    shuffled = rs.permutation(truth)          # same k, no structure at all

    panel = UniformPanel(len(X), subsample=len(X))
    good = panel.measure("real", X, truth, heldout=False)
    bad = panel.measure("shuffled", X, shuffled, heldout=False)

    g = good.get("stability_ari")
    b = bad.get("stability_ari")
    assert g is not None and b is not None
    assert len(np.unique(truth)) == len(np.unique(shuffled)), "fixture must hold k fixed"
    assert g > b + 0.30, (
        f"a real partition ({g}) and a random one at the same k ({b}) got "
        "indistinguishable stability — the metric is not reading the partition"
    )

    # The old corpus-level quantity is still available, under a name that says so,
    # and it is exactly the thing that CANNOT tell these two apart.
    assert good.get("kmeans_refit_stability") == bad.get("kmeans_refit_stability")


# --- a threshold that only fits one corpus ----------------------------------


def test_l2_visibility_is_judged_against_chance_not_a_flat_number():
    """kNN agreement means something different at every class count.

    With 22 classes a random neighbour agrees ~4.5% of the time; with 2 classes
    ~50%. A flat 0.5 therefore calls a dominant class "geometry-visible" on its
    PRIOR alone, and can never be reached by a small one however cleanly the
    embedding separates it. The bar is now max(floor, 2 x chance), and chance is
    the class's own share.
    """
    import inspect

    from qmine.ops.subintent import geometric_audit

    sig = inspect.signature(geometric_audit)
    assert "chance_multiple" in sig.parameters, \
        "the bar must scale with chance, not sit at an absolute level"

    src = inspect.getsource(geometric_audit)
    # The bar is derived from the corpus's own spread, with a chance-relative
    # floor beneath it. Assert both terms are present rather than a literal
    # expression, which changes whenever the formulation is refined.
    assert "mad_multiple" in src and "np.median" in src, \
        "the bar must come from this corpus's spread, not from a constant"
    assert "chance_multiple * r[\"share_in_subsample\"]" in src, \
        "with a chance-relative floor, so a class cannot pass on its prior alone"
    # The subsample counts must not masquerade as population counts.
    assert '"n_in_subsample"' in src and '"share_in_subsample"' in src, \
        "these are counts from an 8,000-row subsample of a 50k corpus"
    assert '"lift_over_chance"' in src, \
        "the verdict is only checkable if the lift travels with it"


def test_the_l2_bar_comes_from_the_corpus_not_from_a_constant():
    """A flat 0.5 was read off K12 and means nothing elsewhere.

    Measured on live38: 21 classes, kNN agreement 0.25-0.886, and every class
    ran 3.4x-76x above its own share — so a purely chance-relative bar flags
    NOTHING and the audit reports nothing ever. What identifies a class the
    embedding cannot carry is being an outlier against its neighbours, so the
    bar is `median - 1.0 x MAD`. On live38 that is 0.495 and selects exactly the
    five classes the old constant did.
    """
    import numpy as np
    from sklearn.preprocessing import normalize

    from qmine.ops.subintent import geometric_audit

    rng = np.random.default_rng(0)
    dim, per = 8, 50
    X = normalize(np.vstack([
        rng.normal(loc=np.eye(dim)[i] * 12.0, scale=0.05, size=(per, dim))
        for i in range(6)]))
    labels = sum(([f"C{i}"] * per for i in range(6)), [])

    out = geometric_audit(X, labels, sample=len(labels), k=5)
    assert "bar_basis" in out and "MAD" in out["bar_basis"], \
        "the bar must state where it came from"
    flagged = [r for r in out["classes"] if r["verdict"] == "rule-dependent"]
    assert not flagged, (
        "six equally well-separated classes have no outlier, so nothing should be "
        f"called rule-dependent — got {[r['class'] for r in flagged]}. A fixed "
        "quartile would have condemned 25% regardless.")


def test_a_dominant_class_cannot_be_called_visible_on_its_prior_alone():
    """Two classes, one holding 80% of the rows, embedded as pure noise.

    Under a flat 0.5 the majority class clears the bar because most of anyone's
    neighbours belong to it. Judged against chance it cannot.
    """
    import numpy as np

    from qmine.ops.subintent import geometric_audit

    rng = np.random.default_rng(0)
    n = 400
    X = rng.normal(size=(n, 16))          # no structure at all
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    labels = ["BIG"] * int(n * 0.8) + ["SMALL"] * (n - int(n * 0.8))

    out = geometric_audit(X, labels, sample=n, k=5)
    by = {r["class"]: r for r in out["classes"]}
    big = by["BIG"]
    # On noise, kNN agreement for BIG lands near its share (~0.8) — that is chance.
    assert big["knn_agreement"] >= 0.5, "sanity: a flat 0.5 would have passed it"
    assert big["lift_over_chance"] < 2.0, "but it carries no signal over chance"
    assert big["verdict"] == "rule-dependent", (
        "a class the embedding cannot actually separate must not be called "
        "geometry-visible just because it is large")


def test_calibration_is_measured_out_of_fold_like_the_accuracy_beside_it():
    """ECE was computed on the rows the model had just been fitted to.

    It is printed next to an out-of-fold `cv_accuracy`, which invites reading the
    two as comparable when only one was honest — and the report says phase 10
    ROUTES on confidence, so an optimistic calibration figure loosens a live
    threshold rather than merely looking good.
    """
    import numpy as np

    from qmine.ops.classify import train_classifier

    rng = np.random.default_rng(0)
    n = 300
    # Pure noise: a well-calibrated model should be near chance and SAY so.
    X = rng.normal(size=(n, 12))
    y = [("A" if i % 2 else "B") for i in range(n)]

    out = train_classifier(X, y)
    assert out.get("ece_basis") == "out-of-fold", (
        f"ECE basis was {out.get('ece_basis')!r} — it must be measured the same "
        "way as the accuracy it is reported beside")
    assert out["cv_accuracy"] < 0.7, "sanity: noise must not be learnable"


def test_no_translation_key_is_dead():
    """Every PROSE_ZH key must be a prefix of prose something actually authors.

    `prose()` matches with `startswith`, so a key that drifts by one word stops
    matching and the English falls through to a Chinese deliverable. That is how
    "Low coherence means clusters…" sat un-firing while the code authored "Low
    coherence means THOSE clusters…".

    The existing coverage test reads the RENDERED report, so it only sees strings
    the fixture happens to emit — a remediation for a gate that passes never
    renders, and the drift hides. This checks the mapping against the source.
    """
    import pathlib

    from qmine.report.i18n import PROSE_ZH

    src_root = pathlib.Path(__file__).resolve().parents[1] / "src" / "qmine"
    blob = "\n".join(f.read_text(encoding="utf-8")
                     for f in src_root.rglob("*.py"))

    dead = []
    for key in PROSE_ZH:
        # The authored string must START with the key, so the key has to appear
        # in the source immediately after a quote.
        if f'"{key}' in blob or f"'{key}" in blob.replace('"', "'"):
            continue
        dead.append(key)
    # i18n.py itself holds every key, so a key found ONLY there is dead.
    i18n = (src_root / "report" / "i18n.py").read_text(encoding="utf-8")
    dead = [k for k in dead + [k for k in PROSE_ZH if blob.count(f'"{k}') <= i18n.count(f'"{k}')]
            if k in PROSE_ZH]
    dead = sorted(set(dead))
    assert not dead, (
        "these translation keys match no authored prose — the English will reach "
        "the Chinese reader:\n  " + "\n  ".join(dead[:8]))


def test_no_domain_does_not_silently_mean_chinese():
    """`--domain` omitted means "unknown vertical", not "K12 Chinese".

    A bare `QMineConfig()` reported `key="generic"` while carrying
    `language=zh`, `tokenizer=jieba`, Chinese-only bake-off candidates, ZERO risk
    categories and ZERO pragmatic-intent hints — strictly worse than
    `--domain generic`, and silently so on an English corpus.
    """
    from qmine.config import DomainProfile, QMineConfig

    d = QMineConfig().domain
    assert d.key == "generic"
    assert d.language == "multi", "a generic profile must not assume Chinese"
    assert d.tokenizer == "auto", "the tokeniser must be resolved from the corpus"
    assert not any("-zh" in c for c in d.embedding_candidates), d.embedding_candidates

    # And the profile the CLI actually loads carries the universal parts.
    from qmine.cli import CONFIG_DIR

    g = DomainProfile.load(CONFIG_DIR / "domains" / "generic.yaml")
    assert len(g.risk_categories) >= 5, "a generic run must still screen for harm"
    assert g.pragmatic_intents_hint, "the top-down route needs its brief"
    assert g.template_seeds == [], (
        "phrasing families are exactly what differs between verticals — they must "
        "be mined and validated, never assumed"
    )


def test_an_unknown_domain_says_what_exists():
    """A bare FileNotFoundError naming a path inside the package tells a user
    nothing about what they could have typed instead."""
    import pytest as _pytest

    from qmine.cli import _load_domain

    with _pytest.raises(SystemExit) as ei:
        _load_domain("medical_en")
    msg = str(ei.value)
    assert "medical_en" in msg
    assert "k12_zh" in msg and "generic" in msg, f"available profiles not listed: {msg}"
    assert "--domain ./" in msg, "the bring-your-own-profile route is not mentioned"


def test_the_domain_scout_only_runs_when_no_vertical_was_declared():
    """A supplied profile is the operator's statement about their own data and
    outranks a guess from a 300-row sample."""
    import inspect

    from qmine.graph.nodes import foundation

    src = inspect.getsource(foundation.p1_audit)
    assert 'cfg.domain.key == "generic"' in src, (
        "the scout must not second-guess a declared vertical"
    )
    scout = inspect.getsource(foundation._scout_unknown_domain)
    assert "HYPOTHESES ONLY" in scout, "the scout's output must be marked as hypotheses"
    assert "return None" in scout, "a scout that cannot run must not stop the run"


def test_local_k_never_ships_a_pareto_dominated_split():
    """A candidate another admissible candidate beats on BOTH axes cannot win.

    `choose_local_k`'s overrule loop compared `d_sil` against `top` but `d_stab`
    against the RUNNING `pick`, so it depended on iteration order and could settle
    on a candidate that was strictly worse than one it had already seen. live40's
    family 3 shipped k=2 (lift 0.0884, replay ARI 0.9993) while k=3 (0.0994,
    1.0000) dominated it on both — those are the real published numbers below.

    The rule now also ranks on `lift_over_null` rather than raw silhouette.
    Silhouette falls with k on this geometry (Spearman -0.888 on live40's family
    sweep), so ranking k=2 against k=8 on it is the biased comparison; the null is
    computed at the SAME k, which is what removes the k-dependence. The symptom
    was that 5 of 7 families took the minimum admissible k and none took 4-8.
    """
    from qmine.ops.cluster import _rank_local_candidates

    family3 = [
        {"k": 2, "silhouette": 0.0916, "lift_over_null": 0.0884, "stability_ari": 0.9993},
        {"k": 3, "silhouette": 0.0998, "lift_over_null": 0.0994, "stability_ari": 1.0000},
        {"k": 4, "silhouette": 0.0699, "lift_over_null": 0.0709, "stability_ari": 0.9962},
        {"k": 5, "silhouette": 0.0850, "lift_over_null": 0.0862, "stability_ari": 0.9995},
        {"k": 6, "silhouette": 0.0964, "lift_over_null": 0.0969, "stability_ari": 0.9935},
        {"k": 7, "silhouette": 0.1044, "lift_over_null": 0.1058, "stability_ari": 0.8284},
        {"k": 8, "silhouette": 0.1102, "lift_over_null": 0.1099, "stability_ari": 0.7435},
    ]
    pick, _raw = _rank_local_candidates(family3, sil_noise=0.02, stability_gain=0.15)
    assert pick["k"] == 3, f"expected k=3, got k={pick['k']}"

    # And the guarantee itself, stated directly: nothing admissible beats the pick
    # on both axes at once.
    dominated_by = [c for c in family3
                    if c["lift_over_null"] > pick["lift_over_null"]
                    and c["stability_ari"] > pick["stability_ari"]]
    assert not dominated_by, f"shipped a candidate dominated by {dominated_by}"


def test_local_k_ranking_does_not_depend_on_candidate_order():
    """The defect was order-dependence, so the test has to reorder.

    Ranking on a running reference makes the answer a function of how the
    candidate list happens to be sorted — which nothing guarantees.
    """
    import random

    from qmine.ops.cluster import _rank_local_candidates

    cands = [
        {"k": 2, "silhouette": 0.0916, "lift_over_null": 0.0884, "stability_ari": 0.9993},
        {"k": 3, "silhouette": 0.0998, "lift_over_null": 0.0994, "stability_ari": 1.0000},
        {"k": 6, "silhouette": 0.0964, "lift_over_null": 0.0969, "stability_ari": 0.9935},
        {"k": 8, "silhouette": 0.1102, "lift_over_null": 0.1099, "stability_ari": 0.7435},
    ]
    picks = set()
    for seed in range(12):
        shuffled = cands[:]
        random.Random(seed).shuffle(shuffled)
        picks.add(_rank_local_candidates(shuffled, sil_noise=0.02, stability_gain=0.15)[0]["k"])
    assert len(picks) == 1, f"the pick moved with candidate order: {picks}"


def test_local_k_is_ranked_on_lift_over_the_null_not_raw_silhouette():
    """Calibration against a same-k null is what makes the comparison legitimate.

    live40's family 2, real published numbers. Raw silhouette is MAXIMAL at k=2
    (0.0644) and falls monotonically — the small-k attractor in its purest form.
    Lift over a column-shuffled reference computed at the same k peaks at k=3
    (0.0616), because the null scores higher at small k too and subtracting it
    removes precisely that bias.

    Ranking on the raw value gave k=2 for 5 of live40's 7 families — the minimum
    admissible value — and the Phase 7 audit then prescribed splitting 9 of the
    resulting 16 leaves. All 9 pass this function's own null and stability tests on
    replay, so the under-split was real and an agent was compensating for it
    through the one door with no guardrails.

    This case is NOT caught by the Pareto guard: k=2 has the higher stability, so
    neither candidate dominates the other. Only the ranking metric separates them.
    """
    from qmine.ops.cluster import _rank_local_candidates

    family2 = [
        {"k": 2, "silhouette": 0.0644, "lift_over_null": 0.0596, "stability_ari": 1.0},
        {"k": 3, "silhouette": 0.0588, "lift_over_null": 0.0616, "stability_ari": 0.9962},
        {"k": 4, "silhouette": 0.0566, "lift_over_null": 0.0603, "stability_ari": 0.9657},
        {"k": 5, "silhouette": 0.0562, "lift_over_null": 0.0602, "stability_ari": 0.7964},
        {"k": 6, "silhouette": 0.0533, "lift_over_null": 0.0583, "stability_ari": 0.8811},
        {"k": 7, "silhouette": 0.0533, "lift_over_null": 0.0587, "stability_ari": 0.9168},
        {"k": 8, "silhouette": 0.0552, "lift_over_null": 0.0604, "stability_ari": 0.638},
    ]
    pick, raw_top = _rank_local_candidates(family2, sil_noise=0.02, stability_gain=0.15)
    assert raw_top["k"] == 2, "sanity: raw silhouette does peak at the smallest k here"
    assert pick["k"] == 3, (
        f"ranked on the uncalibrated value — got k={pick['k']}, and the "
        "small-k attractor is back")

    # Neither dominates, so the Pareto guard cannot be what saves this.
    assert family2[0]["stability_ari"] > family2[1]["stability_ari"]
    assert family2[1]["lift_over_null"] > family2[0]["lift_over_null"]
