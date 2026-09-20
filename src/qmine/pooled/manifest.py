"""Who the snapshots are, what to call them, and which pairs the study is about.

The hand-built study kept this in six module-level dictionaries — `SOURCES`,
`SRC_ZH`, `CONTRASTS`, `DOMAINS`, `RUN_ID`, `COHORTS` — and every new corpus had
to be registered in all of them before a single table could be produced. That is
the whole of what stopped the study being a feature.

Everything here is derived from the run's own artifacts instead. A snapshot's
identity is the tag `p0` wrote when it pooled the inputs; its size is a row
count; its weight is whatever weight column the run declared. The only things a
user may supply are **display names** and **which pairs to contrast**, because
those are editorial and nothing in the data determines them — and both have
defaults that work untouched.

WITHIN-SNAPSHOT NORMALISATION IS THE LOAD-BEARING RULE. Snapshots differ in size
by an order of magnitude (10,000-row search exports against 1,000-row assistant
samples), and they differ in what their weight column even measures. A raw count
or a raw PV sum across snapshots is therefore a statement about export sizes, not
about queries. `pv_norm` renormalises each snapshot's weight to 10,000 so that
"traffic share" means share *of that snapshot*, and every share in every table is
within-snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd

#: Each snapshot's weights are renormalised to this total. The value is
#: arbitrary and cancels out of every share; it is fixed so that a `pv_norm`
#: column is comparable between runs and reads as "per ten thousand".
NORM_TOTAL = 10_000.0


@dataclass(frozen=True)
class Contrast:
    """One comparison, and the ONE thing that differs across it.

    The label is not decoration. A pooled run of four snapshots has six pairs and
    most of them differ in more than one respect at once; a reader handed all six
    undifferentiated will attribute a difference to whichever cause comes to mind.
    Naming the varying factor per pair is what makes the table readable, and
    `attributable=False` is how a confounded pair says so out loud.
    """

    a: str
    b: str
    differs: str = ""
    attributable: bool = True


@dataclass
class SnapshotManifest:
    """The snapshots of one pooled run, in the order they were pooled."""

    snapshots: list[str]
    sizes: dict[str, int]
    axis: Literal["time", "stratum"] = "time"
    labels: dict[str, str] = field(default_factory=dict)
    surfaces: dict[str, str] = field(default_factory=dict)
    contrasts: list[Contrast] = field(default_factory=list)
    weight_column: str = "pv_norm"
    #: Snapshots whose weight column was absent or constant, so their "traffic
    #: share" equals their row share. Stated rather than hidden: a uniform weight
    #: is a legitimate answer for an export with no PV, and a reader who is not
    #: told will read those shares as measured traffic.
    uniform_weight: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ naming
    def display(self, snapshot: str) -> str:
        return self.labels.get(snapshot, snapshot)

    def displays(self) -> list[str]:
        return [self.display(s) for s in self.snapshots]

    def surface_of(self, snapshot: str) -> str:
        return self.surfaces.get(snapshot, "")

    @property
    def has_surfaces(self) -> bool:
        """True only when EVERY snapshot is assigned, and to more than one group.

        A partial assignment silently drops the unassigned snapshots out of the
        interface tables — the exact defect the study hit when two new 10,000-row
        random exports were excluded from an n that was then reported as 19,997.
        Either the split covers the corpus or there is no split.
        """
        if len(self.surfaces) != len(self.snapshots):
            return False
        return len({self.surfaces[s] for s in self.snapshots}) > 1

    def surface_groups(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for s in self.snapshots:
            out.setdefault(self.surfaces[s], []).append(s)
        return out

    # ------------------------------------------------------------------ record
    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshots": list(self.snapshots),
            "display": {s: self.display(s) for s in self.snapshots},
            "sizes": dict(self.sizes),
            "axis": self.axis,
            "surfaces": dict(self.surfaces),
            "has_surfaces": self.has_surfaces,
            "weight_column": self.weight_column,
            "uniform_weight": list(self.uniform_weight),
            "normalised_to": NORM_TOTAL,
            "contrasts": [{"a": c.a, "b": c.b, "a_display": self.display(c.a),
                           "b_display": self.display(c.b), "differs": c.differs,
                           "attributable": c.attributable} for c in self.contrasts],
        }


def default_contrasts(snapshots: Sequence[str], axis: str) -> list[Contrast]:
    """Which pairs to put in front of the reader when nobody said.

    Time reads as a sequence, so consecutive pairs are the comparisons and the
    span is the summary. A stratum axis has no order — the snapshots are ways of
    sampling, not moments — so every pair is on an equal footing and all of them
    are shown.
    """
    snaps = list(snapshots)
    if len(snaps) < 2:
        return []
    if axis == "time":
        out = [Contrast(a, b, differs=f"时间（{a} → {b}）")
               for a, b in zip(snaps, snaps[1:])]
        if len(snaps) > 2:
            out.append(Contrast(snaps[0], snaps[-1],
                                differs=f"时间跨度（{snaps[0]} → {snaps[-1]}）"))
        return out
    return [Contrast(a, b, differs="分层（同期，抽样方式不同）")
            for i, a in enumerate(snaps) for b in snaps[i + 1:]]


def normalise_weights(df: pd.DataFrame, snapshot_col: str,
                      weight_col: str | None) -> tuple[pd.Series, list[str]]:
    """Per-snapshot weights renormalised to `NORM_TOTAL`, and who got uniform ones.

    A snapshot whose weights are missing, non-positive or constant gets uniform
    weight — which is honest (its traffic share then equals its row share) and is
    reported, never assumed away.
    """
    out = pd.Series(np.nan, index=df.index, dtype=float)
    uniform: list[str] = []
    for snap, g in df.groupby(snapshot_col, sort=False):
        w = (pd.to_numeric(g[weight_col], errors="coerce")
             if weight_col and weight_col in df.columns else pd.Series(np.nan, index=g.index))
        w = w.where(w > 0)
        if w.notna().sum() == 0 or float(w.max() or 0) == float(w.min() or 0):
            out.loc[g.index] = NORM_TOTAL / max(len(g), 1)
            uniform.append(str(snap))
            continue
        # The smallest observed weight, not the median — a row whose traffic
        # is unknown must not be handed a typical row's traffic.
        w = w.fillna(float(w.min()))
        out.loc[g.index] = w * (NORM_TOTAL / float(w.sum()))
    return out, uniform


def build_manifest(df: pd.DataFrame, *, snapshot_col: str = "snapshot",
                   weight_col: str | None = None, axis: str = "time",
                   labels: dict[str, str] | None = None,
                   surfaces: dict[str, str] | None = None,
                   contrasts: Sequence[Sequence[str]] | None = None
                   ) -> tuple[SnapshotManifest, pd.Series]:
    """Derive the manifest from a pooled frame; return it and the `pv_norm` column.

    Order is the order the snapshots appear in the frame, which is the order the
    inputs were listed — `p0` concatenates them in that order and preserves it.
    That is a better default than sorting, because a user who lists 2025 before
    2026 has already said which way time runs.
    """
    if snapshot_col not in df.columns:
        raise KeyError(
            f"the frame has no {snapshot_col!r} column — this is not a pooled run. "
            f"Columns present: {sorted(df.columns)[:12]}")
    snaps = list(pd.unique(df[snapshot_col].astype(str)))
    sizes = {s: int((df[snapshot_col].astype(str) == s).sum()) for s in snaps}
    pv, uniform = normalise_weights(df, snapshot_col, weight_col)

    declared: list[Contrast] = []
    for row in (contrasts or []):
        a, b = str(row[0]), str(row[1])
        if a not in sizes or b not in sizes:
            raise ValueError(
                f"declared contrast ({a}, {b}) names a snapshot this run does not have. "
                f"Snapshots: {snaps}")
        differs = str(row[2]) if len(row) > 2 else ""
        attributable = bool(row[3]) if len(row) > 3 else True
        declared.append(Contrast(a, b, differs, attributable))

    clean_labels = {k: v for k, v in (labels or {}).items() if k in sizes}
    shown = [clean_labels.get(s, s) for s in snaps]
    if len(set(shown)) != len(shown):
        # Two snapshots that read the same silently collapse into one column of
        # every table — the second write wins and the counts become whichever
        # snapshot was processed last, with nothing saying so.
        dupes = sorted({x for x in shown if shown.count(x) > 1})
        raise ValueError(
            f"two snapshots would display as the same name {dupes}. Every table "
            "keys its columns by the display name, so they would collapse into "
            "one. Give them different names.")
    man = SnapshotManifest(
        snapshots=snaps, sizes=sizes, axis=axis,  # type: ignore[arg-type]
        labels=clean_labels,
        surfaces={k: v for k, v in (surfaces or {}).items() if k in sizes},
        contrasts=declared or default_contrasts(snaps, axis),
        uniform_weight=uniform)
    return man, pv
