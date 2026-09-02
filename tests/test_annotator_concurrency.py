"""The two gold-set annotators run at the same time.

They are independent BY DESIGN — that is the whole point of having two — so
there was never an ordering dependency between them, only that the code called
one and then the other. Measured on live38: annotator_a ~161s/call, annotator_b
~23s/call, and at the configured batch concurrency that is ~142 min against
~27 min. Sequentially p2b pays their SUM; concurrently it pays their MAX.

The dangerous failure mode is not a crash. It is a SWAP: if a's labels were
returned as b's, kappa would still compute, every gate would still pass, and the
gold set every later phase trains on would be quietly wrong.
"""
from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from qmine.graph.nodes import topdown


def _ctx(offline=False, concurrent=True, max_concurrency=8):
    return SimpleNamespace(
        registry=SimpleNamespace(is_offline=offline),
        cfg=SimpleNamespace(llm=SimpleNamespace(
            max_concurrency=max_concurrency, annotators_concurrent=concurrent)),
    )


def _deps():
    ev = []
    return SimpleNamespace(emit=ev.append), ev


def test_each_annotators_labels_come_back_as_its_own(monkeypatch):
    """A swap would produce a valid kappa on a corrupted gold set."""
    def fake(ctx, which, queries, *a, **k):
        return [{"query": q, "label": f"{which.upper()}_{i}"} for i, q in enumerate(queries)]

    monkeypatch.setattr(topdown, "_annotate", fake)
    deps, _ = _deps()

    la, lb = topdown._annotate_both(_ctx(), ["q0", "q1"], "c", "r", "g", deps)

    assert [x["label"] for x in la] == ["A_0", "A_1"], f"a got {la}"
    assert [x["label"] for x in lb] == ["B_0", "B_1"], f"b got {lb}"


def test_query_order_is_preserved_for_positional_pairing(monkeypatch):
    """kappa pairs the two lists BY INDEX, so both must be in query order.

    `_annotate` already rebuilds its return as `[got.get(q) for q in queries]`
    rather than in completion order; this pins that the concurrent wrapper does
    not reintroduce completion ordering.
    """
    def fake(ctx, which, queries, *a, **k):
        if which == "b":
            time.sleep(0.05)                      # finish out of submission order
        return [{"query": q, "label": which} for q in queries]

    monkeypatch.setattr(topdown, "_annotate", fake)
    deps, _ = _deps()
    qs = [f"q{i}" for i in range(6)]

    la, lb = topdown._annotate_both(_ctx(), qs, "c", "r", "g", deps)

    assert [x["query"] for x in la] == qs
    assert [x["query"] for x in lb] == qs


def test_they_actually_overlap_in_time(monkeypatch):
    """The point of the change. Sequential execution would take the sum."""
    started, lock = [], threading.Lock()
    overlapped = threading.Event()

    def fake(ctx, which, queries, *a, **k):
        with lock:
            started.append(which)
            if len(started) == 2:
                overlapped.set()
        overlapped.wait(timeout=2.0)              # only clears if BOTH started
        return [{"query": q, "label": which} for q in queries]

    monkeypatch.setattr(topdown, "_annotate", fake)
    deps, ev = _deps()
    t0 = time.time()

    topdown._annotate_both(_ctx(), ["q"], "c", "r", "g", deps)

    assert overlapped.is_set(), "the second annotator never started while the first ran"
    assert time.time() - t0 < 2.0, "they ran sequentially"
    assert any("concurrently" in e for e in ev), ev
    assert any("peak 16" in e for e in ev), f"peak concurrency not disclosed: {ev}"


def test_offline_stays_sequential(monkeypatch):
    """The stand-in is deterministic and instant — there is no latency to hide,
    and sequential keeps its logs readable."""
    order = []

    def fake(ctx, which, queries, *a, **k):
        order.append(("start", which))
        order.append(("end", which))
        return [{"query": q, "label": which} for q in queries]

    monkeypatch.setattr(topdown, "_annotate", fake)
    deps, ev = _deps()

    topdown._annotate_both(_ctx(offline=True), ["q"], "c", "r", "g", deps)

    assert order == [("start", "a"), ("end", "a"), ("start", "b"), ("end", "b")]
    assert not any("concurrently" in e for e in ev)


def test_the_flag_turns_it_off(monkeypatch):
    """A provider that rate-limits at 2x concurrency needs an off switch."""
    monkeypatch.setattr(topdown, "_annotate",
                        lambda ctx, which, queries, *a, **k:
                        [{"query": q, "label": which} for q in queries])
    deps, ev = _deps()

    topdown._annotate_both(_ctx(concurrent=False), ["q"], "c", "r", "g", deps)

    assert not any("concurrently" in e for e in ev)


def test_a_failure_names_the_annotator_and_waits_for_the_other(monkeypatch):
    """Letting the first exception escape would still block on the other thread,
    then discard its work and report only one of two possible causes."""
    finished = []

    def fake(ctx, which, queries, *a, **k):
        if which == "a":
            raise RuntimeError("provider down")
        time.sleep(0.05)
        finished.append(which)
        return [{"query": q, "label": which} for q in queries]

    monkeypatch.setattr(topdown, "_annotate", fake)
    deps, _ = _deps()

    with pytest.raises(RuntimeError) as ei:
        topdown._annotate_both(_ctx(), ["q"], "c", "r", "g", deps)

    assert "annotator a" in str(ei.value)
    assert "provider down" in str(ei.value)
    assert finished == ["b"], "the other annotator was abandoned mid-flight"


def test_both_failing_reports_both(monkeypatch):
    def fake(ctx, which, queries, *a, **k):
        raise RuntimeError(f"{which} exploded")

    monkeypatch.setattr(topdown, "_annotate", fake)
    deps, _ = _deps()

    with pytest.raises(RuntimeError) as ei:
        topdown._annotate_both(_ctx(), ["q"], "c", "r", "g", deps)

    msg = str(ei.value)
    assert "a exploded" in msg and "b exploded" in msg, msg


def test_an_empty_batch_is_a_failure_not_a_labelled_batch():
    """`AnnotationBatch.labels` defaults to `[]`, so a model that returns nothing
    usable produces a schema-VALID empty batch: no exception is raised, the retry
    never fires, and 25 rows vanish without even the "batch lost" warning.

    Measured on live43: `qwen3.8-flash` did this on **62 of 128 batches**. The
    distribution was bimodal — 0 or 25, never partial — and the phase reported
    `annotator[b] labelled 1500/3000` with ZERO lost-batch lines, because by its
    own reckoning nothing had failed. Half the gold set lost its second
    annotator, which is the shape of the defect this project already has a rule
    about: a kappa computed on a fraction of its sample and shipped as a result.

    Same family as `SectionDraft.markdown` defaulting to `""`: a permissive
    default turns a failed generation into a successful empty one.
    """
    import inspect

    from qmine.graph.nodes import topdown

    # STRIP COMMENTS FIRST, THEN SLICE. Slicing a fixed byte window and stripping
    # afterwards makes the assertion depend on how much PROSE the function
    # carries: adding a comment block pushed the completeness check past the
    # window and failed a test whose subject had not changed.
    src = inspect.getsource(topdown)
    src = "\n".join(ln for ln in src.splitlines() if not ln.strip().startswith("#"))
    i = src.index("def _one(chunk")
    code = src[i:i + 2200]
    assert "missing" in code and "raise ValueError" in code, (
        "an incomplete batch must raise so the existing retry can run")
    # The check must come BEFORE the success return, or it cannot prevent one.
    assert code.index("raise ValueError") < code.index("return got"), (
        "the completeness check runs after the batch is already accepted")


def test_a_short_batch_is_retried_and_can_recover():
    """The retry loop already existed; the empty batch simply never reached it."""
    import types

    calls = {"n": 0}

    class _Batch:
        def __init__(self, labels):
            self.labels = labels

    class _Label:
        def __init__(self, q):
            self.query = q
        def model_dump(self):
            return {"query": self.query, "label": "x"}

    def fake_run(queries=None, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return _Batch([])                      # the silent empty batch
        return _Batch([_Label(q) for q in queries])

    # Exercise the same shape the node uses: fail once, then succeed.
    agent = types.SimpleNamespace(run=fake_run)
    chunk = ["q1", "q2", "q3"]
    got = {}
    for attempt in range(3):
        batch = agent.run(queries=chunk)
        got = {l.query: l.model_dump() for l in batch.labels}
        if not [q for q in chunk if q not in got]:
            break
    assert calls["n"] == 2, "the empty batch must be retried, not accepted"
    assert len(got) == 3
