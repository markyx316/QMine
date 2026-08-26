"""A corpus that carries legacy labels and a run that declares none are not the
same run, and every log line looks identical.

`--reference-columns` is a launch flag with an empty default. Omitting it on
live40 changed eight things at once: the gold set and the pilot stopped being
stratified by legacy label, the blindness firewall was armed with fewer forbidden
terms, the legacy-audit researcher returned nothing, the corpus audit lost its
legacy distribution, and the delivered table lost its crosswalk.

Only the researcher said anything, and what it said — "this angle contributed
nothing" — reads as a finding about the corpus rather than about the command.
The run was 22 minutes in before a human noticed.
"""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from qmine.graph.nodes.foundation import _label_like_columns

CFG = SimpleNamespace(data=SimpleNamespace(text_column="query", weight_column="weight"))


def test_legacy_label_columns_are_detected():
    df = pd.DataFrame({
        "query": [f"查询{i}" for i in range(500)],
        "legacy_l1": ["语文", "数学"] * 250,
        "legacy_l2": [f"子类{i % 8}" for i in range(500)],
    })
    assert _label_like_columns(df, CFG) == ["legacy_l1", "legacy_l2"]


def test_the_dtype_check_is_not_object_specific():
    """pandas 3 reads text columns as `str`, not `object`.

    The first version asked `dtype != object` and therefore matched NOTHING on
    the very corpus it was written for — a guard that passes because it cannot
    see, which is the failure mode a guard can least afford. Asking what the
    column is NOT (numeric, boolean) is version-robust.
    """
    df = pd.DataFrame({"query": [f"q{i}" for i in range(200)],
                       "legacy_l1": ["a", "b"] * 100})
    for dtype in ("object", "string", "str"):
        try:
            cast = df.astype({"legacy_l1": dtype})
        except TypeError:
            continue
        assert "legacy_l1" in _label_like_columns(cast, CFG), dtype


def test_free_text_and_numeric_columns_are_not_mistaken_for_labels():
    """A guard that cries wolf gets widened until it stops checking anything."""
    df = pd.DataFrame({
        "query": [f"查询{i}" for i in range(500)],
        "notes": [f"free text number {i}" for i in range(500)],   # high cardinality
        "weight": [1.0] * 500,                                    # numeric
        "row_id": list(range(500)),
    })
    assert _label_like_columns(df, CFG) == []


def test_a_corpus_with_only_a_text_column_reports_nothing():
    df = pd.DataFrame({"query": [f"q{i}" for i in range(100)]})
    assert _label_like_columns(df, CFG) == []
