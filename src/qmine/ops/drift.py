"""What changed between snapshots of one corpus, measured rather than described.

THE ONE THING THAT MAKES THIS COMPARABLE AT ALL. Two runs of this pipeline do not
produce comparable labels: 金融 2025-07 and 金融 2026-07 run separately (`fin01`,
`fin02`) produced 20 and 19 classes sharing **zero** codes — semantically parallel
(`LOOKUP_FX_RATE` vs `FX_RATE_LOOKUP`) but not joinable — because the taxonomy
architect renames every run and the tree is refitted from scratch. So drift is measured INSIDE one run, over
a pooled corpus, with one taxonomy and one tree labelling every period. Every
figure here is a comparison of two groups of rows that were labelled by the same
labeller — which is what removes "the taxonomy moved" as an explanation and
leaves "the queries moved".

TWO DENOMINATORS, ALWAYS BOTH, NEVER RAW COUNTS.

* **row share** — of the DISTINCT queries in that snapshot. Answers "did the
  variety of things people ask change?"
* **traffic share** — of that snapshot's total weight. Answers "did what people
  actually search change?"

Both are within-snapshot. The snapshots are not the same size in traffic — one
medical pair fell 9.74M to 5.21M (-47%) — so raw weights report every class as
declining and say nothing about composition. The two shares can also move in
OPPOSITE directions, and when they do that is the finding, not an error: a class
gaining queries while losing traffic is fragmenting into long-tail phrasings.

WHAT IS DELIBERATELY NOT COMPUTED HERE.

* No p-value on traffic share. Traffic is the population, not a sample of one;
  a significance test over it is theatre.
* No single "drift score". A scalar invites ranking corpora against each other,
  and the interesting content is always which INTENT moved, never how much
  total movement there was. `total_variation` is reported as an ANCHOR for one
  pair — how much traffic would have to be reassigned to turn A into B — and is
  not comparable across corpora.
* No causal claim. This module can say a class grew; it cannot say why, and the
  report it feeds must not either.

WHAT THIS IS, IN THE FORMAL VOCABULARY. This measures **prior-probability shift
over a fixed taxonomy**: P(class) moved while the labelling rule stayed put. It is
structurally blind to *real concept drift* — the same query string meaning
something new — because ONE architect labelled both periods with one rule set.
That blindness is the price of comparability, and it is the exact reason pooling
works at all; it is stated in the report rather than hidden.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

#: A class this thin in BOTH snapshots is noise, and its share swings wildly on a
#: handful of rows. Reported in its own section rather than ranked with the rest.
MIN_ROWS_FOR_COMPARISON = 30

#: Present in one snapshot and essentially absent from the other. These are the
#: most interesting rows in the table and the most easily misread: a class can be
#: "new" because the behaviour is new, or because the pooled taxonomy finally had
#: enough evidence to name something that was always there but too sparse to
#: cluster. Both readings are reported; the data cannot separate them.
EMERGENCE_RATIO = 0.05


def _z_two_proportion(x1: int, n1: int, x2: int, n2: int) -> float:
    """Standard two-proportion z. Returns 0.0 when it is undefined."""
    if n1 <= 0 or n2 <= 0:
        return 0.0
    p_pool = (x1 + x2) / (n1 + n2)
    se = math.sqrt(p_pool * (1.0 - p_pool) * (1.0 / n1 + 1.0 / n2))
    return 0.0 if se == 0 else ((x1 / n1) - (x2 / n2)) / se


def _cramers_v(counts: np.ndarray) -> float:
    """Effect size for the whole table — how far the two profiles are, 0 to 1.

    Reported INSTEAD of a chi-square p-value. At n=20,000 a chi-square is
    significant on differences far too small to act on, so the p-value carries no
    decision content; the effect size does.
    """
    if counts.size == 0 or counts.sum() == 0:
        return 0.0
    total = counts.sum()
    exp = np.outer(counts.sum(axis=1), counts.sum(axis=0)) / total
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum(np.where(exp > 0, (counts - exp) ** 2 / exp, 0.0))
    k = min(counts.shape) - 1
    return float(math.sqrt(chi2 / (total * k))) if k > 0 and total > 0 else 0.0


def snapshot_inventory(df: pd.DataFrame, snap_col: str, text_col: str,
                       weight_col: str | None) -> list[dict[str, Any]]:
    """Size of each snapshot, before any labelling is involved."""
    out = []
    for tag, g in df.groupby(snap_col, sort=True):
        w = float(g[weight_col].sum()) if weight_col and weight_col in g else float(len(g))
        out.append({"snapshot": str(tag), "rows": int(len(g)),
                    "distinct_queries": int(g[text_col].nunique()),
                    "weight_total": w})
    return out


def query_churn(df: pd.DataFrame, snap_col: str, text_col: str,
                weight_col: str | None) -> dict[str, Any]:
    """How much of the corpus is even the SAME queries between two snapshots.

    This bounds how much of any label-share change can be a demand shift at all.
    Where the shared core carries most of the traffic, drift is mostly volume
    moving between existing asks. Where it does not — a film corpus turned over
    83% of its distinct queries in a year — the composition of the churn is the
    story, and per-title comparison is meaningless.
    """
    tags = sorted(df[snap_col].astype(str).unique())
    if len(tags) != 2:
        return {"comparable": False, "reason": f"{len(tags)} snapshots; churn is defined pairwise"}
    a, b = tags
    qa = set(df.loc[df[snap_col].astype(str) == a, text_col].astype(str))
    qb = set(df.loc[df[snap_col].astype(str) == b, text_col].astype(str))
    shared = qa & qb
    out: dict[str, Any] = {
        "comparable": True, "snapshot_a": a, "snapshot_b": b,
        "distinct_a": len(qa), "distinct_b": len(qb), "shared": len(shared),
        "jaccard": round(len(shared) / len(qa | qb), 4) if (qa | qb) else 0.0,
    }
    if weight_col and weight_col in df:
        for tag, key in ((a, "shared_weight_share_a"), (b, "shared_weight_share_b")):
            m = df[snap_col].astype(str) == tag
            tot = df.loc[m, weight_col].sum() or 1
            out[key] = round(
                float(df.loc[m & df[text_col].astype(str).isin(shared), weight_col].sum() / tot), 4)
    return out


def _total_variation(pa: dict[str, float], pb: dict[str, float]) -> float:
    """Half the L1 distance between two share vectors, over their UNION.

    Reads directly as "this fraction of traffic would have to be reassigned to
    turn A into B" — bounded [0,1], defined when a class is missing from one side
    (its share there is 0), and needing no threshold folklore to interpret.
    """
    keys = set(pa) | set(pb)
    return float(0.5 * sum(abs(pa.get(k, 0.0) - pb.get(k, 0.0)) for k in keys))


def _delta_concentration(g: pd.DataFrame, snap_col: str, text_col: str,
                         a: str, b: str, weight_col: str | None) -> dict[str, Any]:
    """Is this class's movement ONE query, or many?

    A class can double because a single entity blew up for a week, or because the
    whole behaviour broadened — and the product response is opposite in the two
    cases. Concentration on the per-query delta separates them mechanically:
    `top1_share` near 1.0 is one query carrying the entire change (an event),
    a low `hhi` spread over many queries is a broad shift (structural).

    Written for the observed `netdisk piracy 2 -> 189 rows` case, where the
    distinction decides whether anyone should build anything.
    """
    if text_col not in g.columns or g.empty:
        return {}
    ga, gb = g[g[snap_col].astype(str) == a], g[g[snap_col].astype(str) == b]
    col = weight_col if (weight_col and weight_col in g.columns) else None
    ca = ga.groupby(ga[text_col].astype(str))[col].sum() if col else ga[text_col].astype(str).value_counts()
    cb = gb.groupby(gb[text_col].astype(str))[col].sum() if col else gb[text_col].astype(str).value_counts()
    delta = cb.reindex(ca.index.union(cb.index), fill_value=0.0).astype(float) - \
        ca.reindex(ca.index.union(cb.index), fill_value=0.0).astype(float)
    absd = delta.abs()
    tot = float(absd.sum())
    if tot <= 0:
        return {}
    frac = (absd / tot).sort_values(ascending=False)
    return {"n_distinct_queries": int(len(frac)),
            "top1_share_of_delta": round(float(frac.iloc[0]), 4),
            "top5_share_of_delta": round(float(frac.iloc[:5].sum()), 4),
            "hhi_of_delta": round(float((frac ** 2).sum()), 4),
            "top_queries": [{"query": str(q), "delta": round(float(delta[q]), 2)}
                            for q in frac.index[:5]]}


def label_drift(df: pd.DataFrame, label_col: str, snap_col: str,
                weight_col: str | None, text_col: str | None = None) -> dict[str, Any]:
    """Per-class movement between exactly two snapshots.

    `stable` holds classes present in both with enough rows to compare;
    `emergent` and `receded` hold the one-sided ones, which are reported apart
    because a share change is not defined when one side is ~zero.
    """
    tags = sorted(df[snap_col].astype(str).unique())
    if len(tags) != 2:
        return {"comparable": False,
                "reason": f"{len(tags)} snapshots found; this table compares two",
                "snapshots": tags}
    a, b = tags
    ma, mb = df[snap_col].astype(str) == a, df[snap_col].astype(str) == b
    na, nb = int(ma.sum()), int(mb.sum())
    if weight_col and weight_col in df:
        wa = float(df.loc[ma, weight_col].sum()) or 1.0
        wb = float(df.loc[mb, weight_col].sum()) or 1.0
    else:
        wa, wb = float(na) or 1.0, float(nb) or 1.0

    stable, emergent, receded, thin = [], [], [], []
    share_a: dict[str, float] = {}   # traffic share per class, per snapshot —
    share_b: dict[str, float] = {}   # collected over EVERY class, including the
    row_a: dict[str, float] = {}     # emergent and thin ones, because total
    row_b: dict[str, float] = {}     # variation is a whole-distribution figure
    labels = [x for x in df[label_col].dropna().unique()]
    for lab in labels:
        g = df[df[label_col] == lab]
        xa, xb = int((g[snap_col].astype(str) == a).sum()), int((g[snap_col].astype(str) == b).sum())
        if weight_col and weight_col in df:
            pa = float(g.loc[g[snap_col].astype(str) == a, weight_col].sum()) / wa
            pb = float(g.loc[g[snap_col].astype(str) == b, weight_col].sum()) / wb
        else:
            pa, pb = xa / (na or 1), xb / (nb or 1)
        rec = {"label": str(lab), "rows_a": xa, "rows_b": xb,
               "row_share_a": round(xa / (na or 1), 5), "row_share_b": round(xb / (nb or 1), 5),
               "row_share_delta_pp": round((xb / (nb or 1) - xa / (na or 1)) * 100, 3),
               "weight_share_a": round(pa, 5), "weight_share_b": round(pb, 5),
               "weight_share_delta_pp": round((pb - pa) * 100, 3),
               "z_row_share": round(_z_two_proportion(xb, nb, xa, na), 2)}
        share_a[str(lab)], share_b[str(lab)] = pa, pb
        row_a[str(lab)], row_b[str(lab)] = xa / (na or 1), xb / (nb or 1)
        if text_col:
            conc = _delta_concentration(g, snap_col, text_col, a, b, weight_col)
            if conc:
                rec["delta_concentration"] = conc
        total = xa + xb
        minor = min(xa, xb)
        if total >= MIN_ROWS_FOR_COMPARISON and minor / total <= EMERGENCE_RATIO:
            (emergent if xb > xa else receded).append(rec)
        elif xa < MIN_ROWS_FOR_COMPARISON and xb < MIN_ROWS_FOR_COMPARISON:
            thin.append(rec)
        else:
            stable.append(rec)

    counts = np.array([[int(((df[label_col] == lab) & ma).sum()) for lab in labels],
                       [int(((df[label_col] == lab) & mb).sum()) for lab in labels]], dtype=float)
    key = lambda r: -abs(r["weight_share_delta_pp"])  # noqa: E731
    return {
        "comparable": True, "snapshot_a": a, "snapshot_b": b,
        "rows_a": na, "rows_b": nb, "n_classes": len(labels),
        "cramers_v": round(_cramers_v(counts), 4),
        "total_variation_weight": round(_total_variation(share_a, share_b), 4),
        "total_variation_rows": round(_total_variation(row_a, row_b), 4),
        "n_comparisons": len(stable),
        "stable": sorted(stable, key=key),
        "emergent": sorted(emergent, key=key),
        "receded": sorted(receded, key=key),
        "too_thin_to_compare": sorted(thin, key=key),
    }


def snapshot_purity(df: pd.DataFrame, label_col: str, snap_col: str,
                    min_rows: int = MIN_ROWS_FOR_COMPARISON) -> dict[str, Any]:
    """Does any group sit almost entirely in ONE snapshot?

    The premise of the whole comparison is that both periods were labelled in one
    shared frame. A group that is ~100% one snapshot did not get compared — it got
    separated — and if many are, the "shared frame" is a fiction and no share in
    this report means what it appears to. Measured on five real corpora: four had
    ZERO such groups; the fifth was a news-driven people corpus where 6 of 25
    leaves were genuine year-specific events (a personnel announcement, an awards
    ceremony). So a nonzero count is a prompt to look, not a verdict.
    """
    tags = sorted(df[snap_col].astype(str).unique())
    if len(tags) != 2:
        return {"checked": False, "reason": f"{len(tags)} snapshots"}
    a = tags[0]
    shares, sizes = {}, {}
    for lab, g in df.groupby(label_col):
        sizes[str(lab)] = int(len(g))
        shares[str(lab)] = float((g[snap_col].astype(str) == a).mean())
    pure = [{"label": k, "share_of_" + str(a): round(v, 3), "rows": sizes[k]}
            for k, v in shares.items()
            if sizes[k] >= min_rows and (v > 0.95 or v < 0.05)]
    vals = list(shares.values())
    return {"checked": True, "n_groups": len(shares),
            "n_single_snapshot": len(pure), "single_snapshot": pure,
            "share_min": round(min(vals), 3) if vals else None,
            "share_median": round(float(np.median(vals)), 3) if vals else None,
            "share_max": round(max(vals), 3) if vals else None}
