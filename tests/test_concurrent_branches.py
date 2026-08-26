"""The two routes run concurrently, and everything that made that safe.

Measured on live39: 38 min of taxonomy design plus 69 min of gold annotation sat
in front of 39 min of bottom-up CPU work that depends on none of it. The graph
now forks at p1 and joins at p2c, which hides the entire bottom-up branch inside
p2b's provider latency.

Concurrency turns three classes of latent bug into real ones, and each test here
is one of them:

* **shape** — langgraph 1.2.11 advances branches in supersteps and a fan-in node
  fires once per incoming edge unless they arrive together;
* **shared mutable state** — `Deps`, the artifact index and the findings ledger
  were all read-modify-write, safe only while the graph was a chain;
* **state channels** — a plain field written by two branches in one superstep is
  a runtime error, not a compile-time one.
"""
from __future__ import annotations

import threading

import pytest

from qmine.graph.build import (
    BOTTOMUP_BRANCH,
    JOIN_NODE,
    PHASE_NODES,
    SEQUENTIAL_TAIL,
    TOPDOWN_BRANCH,
)


# ------------------------------------------------------------------- shape
def test_the_two_branches_stay_the_same_length():
    """A fan-in node fires ONCE PER INCOMING EDGE unless every edge arrives in
    the same superstep — measured on langgraph 1.2.11, not assumed.

    Equal lengths are what make `p2c` receive both at once. `_wrap`'s
    idempotence guard means an imbalance costs wall clock rather than a
    double-trained classifier, but the balance is what makes the schedule good:
    against live39's timings, 2-and-2 costs 107 min where 1-and-3 costs 129.
    """
    assert len(TOPDOWN_BRANCH) == len(BOTTOMUP_BRANCH), (
        f"top-down has {len(TOPDOWN_BRANCH)} nodes, bottom-up {len(BOTTOMUP_BRANCH)} — "
        "an imbalance makes the join fire twice and wastes the overlap")


def test_the_branches_are_disjoint_and_reach_the_join():
    names_td = {n for n, _ in TOPDOWN_BRANCH}
    names_bu = {n for n, _ in BOTTOMUP_BRANCH}
    assert not (names_td & names_bu)
    assert JOIN_NODE not in names_td | names_bu
    assert JOIN_NODE == SEQUENTIAL_TAIL[0]
    listed = {n for n, _ in PHASE_NODES}
    assert names_td | names_bu <= listed, "a branch node must also be a graph node"


def test_the_bottom_up_branch_reads_nothing_the_top_down_branch_writes():
    """The claim the fork rests on, checked against the source rather than
    remembered. `p3_represent` consumes `template_groups`, from p1 — if a future
    edit makes it read the taxonomy or the gold set, the fork becomes a race and
    this fails instead of the run producing quiet nonsense.
    """
    import ast
    import inspect

    import qmine.graph.nodes.bottomup as bu

    TOPDOWN_ARTIFACTS = {"taxonomy", "taxonomy_v2", "gold", "gold_agreement",
                         "topdown_metrics", "topdown_labels", "adversarial_validation"}
    for name, fn in BOTTOMUP_BRANCH:
        src = inspect.getsource(getattr(bu, fn.__name__))
        tree = ast.parse(src.lstrip())
        read = {n.args[0].value for n in ast.walk(tree)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr in {"load", "recover", "has"} and n.args
                and isinstance(n.args[0], ast.Constant)}
        assert not (read & TOPDOWN_ARTIFACTS), (
            f"{name} reads {read & TOPDOWN_ARTIFACTS} — the branches are not independent")
        assert "deps.taxonomy()" not in src, f"{name} reads the taxonomy"


def test_the_naming_phase_stays_downstream_of_the_gold_set():
    """The blindness firewall is armed in p7 from the taxonomy's own vocabulary.

    A p7 running CONCURRENTLY with p2a/p2b is the one dangerous placement: the
    taxonomy would exist but be incomplete, so the firewall would be armed with
    part of the forbidden vocabulary and quietly let the rest through. Before the
    fork it is blind by construction; after p2b it is blind by enforcement.
    Neither is true in between, so p7 must stay in the sequential tail.
    """
    assert "p7_prepare" in SEQUENTIAL_TAIL
    assert "p7_prepare" not in {n for n, _ in TOPDOWN_BRANCH} | {n for n, _ in BOTTOMUP_BRANCH}
    assert SEQUENTIAL_TAIL.index("p7_prepare") > SEQUENTIAL_TAIL.index(JOIN_NODE)


# ------------------------------------------------------- state channels
def test_every_channel_a_forked_branch_writes_has_a_reducer():
    """langgraph refuses a plain field that receives two values in one superstep
    — and it refuses at RUNTIME, on the first step where both branches happen to
    write it. `phase` was exactly that, and the run died at p2b.
    """
    from typing import get_type_hints

    from qmine.state import PipelineState

    hints = get_type_hints(PipelineState, include_extras=True)
    for field in ("phase", "phase_status", "artifacts", "gates", "events",
                  "completed_phases", "decisions", "prescriptions"):
        ann = hints[field]
        assert hasattr(ann, "__metadata__"), (
            f"`{field}` has no reducer; two branches writing it in one superstep "
            "is an InvalidUpdateError at runtime")


def test_the_phase_reducer_is_commutative_and_total():
    """A progress line that jumps backwards between refreshes reads as a stall."""
    from qmine.state import furthest_phase

    assert furthest_phase("p2a", "p3") == furthest_phase("p3", "p2a")
    assert furthest_phase("p2b", "p3") == "p3", "p3 is further along than p2b"
    assert furthest_phase("p6", "p2b") == "p6"
    # Total: unknowns and blanks never raise and never win over a known value.
    assert furthest_phase("", "p3") == "p3"
    assert furthest_phase("p3", "") == "p3"
    assert furthest_phase("nonsense", "p3") == "p3"
    assert furthest_phase("p3", "nonsense") == "p3"
    assert furthest_phase(None, None) == ""


# --------------------------------------------------- shared mutable state
class RecordingLock:
    """An RLock that remembers whether it was ever entered.

    A race test is only trustworthy in one direction: it fails when the code is
    broken *if the interpreter happens to interleave*. Under the GIL a bare
    `+= 1` on a small counter almost never does inside a short test, so the
    stress tests below passed against a deliberately unlocked implementation.

    Asserting that the critical section is actually held is deterministic, and it
    is the invariant we mean. The stress test stays alongside it, because the
    structural check cannot see a critical section that is under the lock but too
    narrow.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self.entries = 0

    def __enter__(self):
        self.entries += 1
        return self._lock.__enter__()

    def __exit__(self, *a):
        return self._lock.__exit__(*a)

    def acquire(self, *a, **kw):
        self.entries += 1
        return self._lock.acquire(*a, **kw)

    def release(self):
        return self._lock.release()


def _bare_deps():
    from qmine.graph.deps import Deps

    deps = Deps.__new__(Deps)
    deps._prescription_seq = 0
    deps._decision_seq = 0
    deps._cache = {}
    deps.on_event = None
    deps._lock = RecordingLock()
    return deps


def test_the_prescription_counter_is_incremented_under_the_lock():
    """Structural, because the stress test alone is not honest.

    p2a and p7_audit both raise prescriptions, and `+= 1` is a
    read-modify-write. A prescription is matched to its executed change BY ID, so
    a collision silently merges two different remedies. Found by mutation:
    removing the lock left the 16-thread stress test below passing.
    """
    deps = _bare_deps()
    deps.next_prescription_id()
    assert deps._lock.entries >= 1, "next_prescription_id does not take the lock"


def test_two_branches_cannot_be_handed_the_same_prescription_id():
    """The stress half. Kept because the structural check cannot see a critical
    section that is held but too narrow — the decision id was incremented under
    the lock and READ outside it, which is still a race."""
    import sys

    from qmine.graph.deps import Deps

    deps = Deps.__new__(Deps)
    deps._prescription_seq = 0
    deps._decision_seq = 0
    deps._lock = threading.RLock()

    got, lock = [], threading.Lock()
    n = 64
    barrier = threading.Barrier(n)
    old_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)          # make an interleaving likely, not lucky
    try:
        def grab():
            barrier.wait()
            for _ in range(20):
                pid = deps.next_prescription_id()
                with lock:
                    got.append(pid)

        ts = [threading.Thread(target=grab) for _ in range(n)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
    finally:
        sys.setswitchinterval(old_interval)
    assert len(set(got)) == len(got), (
        f"{len(got) - len(set(got))} duplicate prescription ids")


def test_the_artifact_index_survives_concurrent_writers(tmp_path):
    """`_register` mutates a dict AND appends a line to `index.jsonl`. Two
    threads interleaving inside one line make `_load_index` skip a real artifact
    on the next resume — a lost artifact that looks like a phase that never ran.
    """
    import json

    import numpy as np

    from qmine.artifacts import ArtifactStore

    store = ArtifactStore(tmp_path)
    real = store._lock
    store._lock = RecordingLock()
    store.put_matrix("probe", np.zeros((1, 1)), producer="t")
    assert store._lock.entries >= 1, "_register does not take the store lock"
    store._lock = real

    barrier = threading.Barrier(12)

    def write(i):
        barrier.wait()
        store.put_matrix(f"m{chr(97 + i)}", np.zeros((2, 2)), producer="t")

    ts = [threading.Thread(target=write, args=(i,)) for i in range(12)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()

    lines = (tmp_path / "index.jsonl").read_text().splitlines()
    for line in lines:
        json.loads(line)                       # every line must parse
    assert len({json.loads(x)["name"] for x in lines}) == 13   # 12 + the probe
    assert len(ArtifactStore(tmp_path).names()) == 13, "a reopened store must see all of them"


@pytest.mark.parametrize("n", [8])
def test_deps_cache_does_not_rebuild_the_same_artifact_twice(tmp_path, n):
    """`if key in cache: ... else: rebuild()` lets two branches both miss."""
    from qmine.graph.deps import Deps

    calls, lock = [], threading.Lock()

    class FakeStore:
        def has(self, name): return True
        def load(self, name): return {"v": 1}

    deps = _bare_deps()
    deps.store = FakeStore()
    deps.recover("probe", "art", rebuild=lambda raw: raw["v"])
    assert deps._lock.entries >= 1, "recover() checks the cache outside the lock"
    deps._lock = threading.RLock()

    def rebuild(raw):
        with lock:
            calls.append(1)
        return raw["v"]

    barrier = threading.Barrier(n)

    def go():
        barrier.wait()
        assert deps.recover("k", "art", rebuild=rebuild) == 1

    ts = [threading.Thread(target=go) for _ in range(n)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert len(calls) == 1, f"rebuilt {len(calls)} times; the cache check is not atomic"


def test_the_decision_id_is_read_inside_the_lock_not_just_incremented():
    """A critical section can be held and still be too narrow.

    Incrementing atomically and then reading the field is a race: a sibling
    branch can bump the counter between the two statements, and both decisions
    ship with the same id. Found by mutation — the RecordingLock check still
    passed, because the lock IS taken; only the read escaped it.

    p2a and p456_tree both record decisions, and they now run concurrently.
    """
    import sys
    from types import SimpleNamespace

    from qmine.graph.deps import Deps

    deps = Deps.__new__(Deps)
    deps._decision_seq = 0
    deps._prescription_seq = 0
    deps._lock = threading.RLock()
    deps.memory = SimpleNamespace(remember_decision=lambda rec: None)
    deps.on_event = None

    ids, guard = [], threading.Lock()
    n = 48
    barrier = threading.Barrier(n)
    old_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        def go():
            barrier.wait()
            for _ in range(10):
                rec = deps.decision(phase="p", question="q", choice="c",
                                    rationale="r", decided_by="metric", evidence={})
                with guard:
                    ids.append(rec.id)

        ts = [threading.Thread(target=go) for _ in range(n)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
    finally:
        sys.setswitchinterval(old_interval)

    assert len(set(ids)) == len(ids), (
        f"{len(ids) - len(set(ids))} duplicate decision ids — the counter is "
        "incremented under the lock but read outside it")


def test_the_fork_can_be_turned_off_and_the_phases_are_unchanged():
    """`concurrent_branches: false` restores the strict chain.

    The escape hatch matters because concurrency is the kind of change that
    fails in an environment rather than in a test — a provider that dislikes two
    pools, a filesystem without atomic appends, a future phase that shares state
    nobody noticed. An operator must be able to get the old schedule back
    without editing the graph.

    It is also how the two schedules are compared: driving the real
    `build_graph` with live39's measured durations gives 174 simulated minutes
    serial against 134.5 forked, and the 39.5-minute difference is exactly the
    bottom-up branch, now hidden inside p2b.
    """
    import types

    from qmine.graph.build import build_graph

    class FakeDeps:
        def emit(self, m): pass
        def recover(self, *a, **kw): return {}
        def has(self, n): return False

    def cfg(concurrent):
        return types.SimpleNamespace(
            gates=types.SimpleNamespace(human_review_points=[]),
            naming=types.SimpleNamespace(n_naming_agents=1),
            concurrent_branches=concurrent)

    forked = build_graph(cfg(True), FakeDeps(), human_review=False)
    serial = build_graph(cfg(False), FakeDeps(), human_review=False)

    # Same phases either way — only the edges differ.
    expected = {n for n, _ in PHASE_NODES} | {"p7_name_shard", "halt"}
    assert set(forked.nodes) - {"__start__"} == expected
    assert set(serial.nodes) - {"__start__"} == expected
