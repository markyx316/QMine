"""Building the evidence packets agents read.

Two packet types, both designed around the same principle: an agent should see
the *hard* cases, not a flattering sample.

A naming card is centre + random + **edge**.  The edge members — those furthest
from the centroid while still assigned to it — are the ones that barely belong,
and including them is the difference between a namer who reports coherence 5 for
everything and one whose coherence scores mean something.

An exemplar is chosen by median index of the hit set, never by anyone's taste.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..determinism import SEED_METRIC, median_index_exemplar, rng
from ..records import NamingCard


def build_naming_cards(
    df: pd.DataFrame,
    labels: np.ndarray,
    X: np.ndarray,
    centroids: np.ndarray,
    *,
    text_col: str = "query",
    n_center: int = 15,
    n_random: int = 10,
    n_edge: int = 5,
    n_ngrams: int = 12,
    seed: int = SEED_METRIC,
) -> list[NamingCard]:
    """One card per cluster.  Contains no label of any kind, by construction."""
    texts = df[text_col].astype(str).to_numpy()
    n_clusters = int(labels.max()) + 1
    sims = X @ centroids.T
    cards: list[NamingCard] = []
    r = rng(seed)

    for c in range(n_clusters):
        idx = np.where(labels == c)[0]
        if idx.size == 0:
            continue
        own = sims[idx, c]
        order = np.argsort(-own)
        center = idx[order[:n_center]]
        edge = idx[order[-n_edge:]] if idx.size > n_center + n_edge else idx[order[-min(n_edge, idx.size) :]]
        pool = np.setdiff1d(idx, np.concatenate([center, edge]), assume_unique=False)
        rand = r.choice(pool, size=min(n_random, pool.size), replace=False) if pool.size else np.array([], dtype=int)
        lens = np.array([len(t) for t in texts[idx]], dtype=float)
        cards.append(
            NamingCard(
                leaf_id=int(c),
                size=int(idx.size),
                share=round(float(idx.size / len(labels)), 5),
                center_samples=[str(t) for t in texts[center]],
                random_samples=[str(t) for t in texts[np.sort(rand)]] if rand.size else [],
                edge_samples=[str(t) for t in texts[edge]],
                top_ngrams=distinctive_ngrams(texts[idx], texts, top=n_ngrams),
                length_stats={
                    "mean": round(float(lens.mean()), 2),
                    "median": float(np.median(lens)),
                    "max": float(lens.max()),
                },
            )
        )
    return cards


def distinctive_ngrams(
    inside: Sequence[str], corpus: Sequence[str], *, top: int = 12, n_range: tuple[int, int] = (2, 4),
    corpus_sample: int = 20000, seed: int = SEED_METRIC,
) -> list[str]:
    """Character n-grams over-represented inside the cluster relative to the corpus.

    A plain frequency list inside a cluster returns the most common substrings of
    the language.  Dividing by the corpus rate is what makes the list *about this
    cluster*.
    """
    def counts(texts: Sequence[str]) -> Counter[str]:
        c: Counter[str] = Counter()
        for t in texts:
            seen = set()
            for n in range(n_range[0], n_range[1] + 1):
                for i in range(len(t) - n + 1):
                    g = t[i : i + n]
                    if g not in seen:
                        seen.add(g)
                        c[g] += 1
        return c

    idx = rng(seed).choice(len(corpus), size=min(corpus_sample, len(corpus)), replace=False)
    outside = counts([corpus[i] for i in idx])
    n_out = max(len(idx), 1)
    inner = counts(inside)
    n_in = max(len(inside), 1)

    scored = []
    for g, c in inner.items():
        if c < max(3, n_in * 0.03):
            continue
        rate_in = c / n_in
        rate_out = outside.get(g, 0) / n_out
        lift = rate_in / (rate_out + 1e-6)
        scored.append((lift * rate_in, g))
    scored.sort(reverse=True)

    out: list[str] = []
    for _, g in scored:
        if any(g in o for o in out):
            continue
        out.append(g)
        if len(out) >= top:
            break
    return out


def cluster_samples(
    df: pd.DataFrame, labels: np.ndarray, *, n: int = 12, text_col: str = "query", seed: int = SEED_METRIC
) -> dict[int, list[str]]:
    """Plain samples per cluster, for the risk sentinel's independent sweep."""
    texts = df[text_col].astype(str).to_numpy()
    r = rng(seed)
    out: dict[int, list[str]] = {}
    for c in range(int(labels.max()) + 1):
        idx = np.where(labels == c)[0]
        if idx.size == 0:
            continue
        pick = r.choice(idx, size=min(n, idx.size), replace=False)
        out[int(c)] = [str(t) for t in texts[np.sort(pick)]]
    return out


def deterministic_exemplars(
    df: pd.DataFrame, masks: dict[str, np.ndarray], *, text_col: str = "query"
) -> list[dict[str, Any]]:
    """One exemplar per pattern, at the median index of its hit set (Principle 7).

    The persuasive force of a worked example comes from the reader knowing you
    could not have picked it.  A rule that is a pure function of the hit set
    delivers that; "here are some queries we thought were illustrative" does not.
    """
    texts = df[text_col].astype(str).to_numpy()
    out = []
    for name, m in masks.items():
        idx = np.where(m)[0]
        if idx.size == 0:
            continue
        pick = median_index_exemplar(idx)
        out.append({
            "pattern": name,
            "n_hits": int(idx.size),
            "exemplar_row": int(pick),
            "exemplar": str(texts[pick]),
            "selection_rule": "median index of the sorted hit set — not chosen by any human or agent",
        })
    return out


def template_spread(
    masks: dict[str, np.ndarray], labels: np.ndarray, *, top: int = 5
) -> dict[str, Any]:
    """Where each phrasing family's members landed.  The auditor's twin detector."""
    out: dict[str, Any] = {}
    for name, m in masks.items():
        if m.sum() < 20:
            continue
        lab = labels[m]
        vc = Counter(int(x) for x in lab)
        total = sum(vc.values())
        out[name] = {
            "n_members": int(total),
            "clusters": [
                {"cluster": c, "n": n, "share": round(n / total, 3)}
                for c, n in vc.most_common(top)
            ],
            "n_clusters_touched": len(vc),
        }
    return out


def centroid_similarity_pairs(centroids: np.ndarray, *, top: int = 40, floor: float = 0.6) -> list[dict[str, Any]]:
    """Most similar cluster pairs — candidate duplicates for the auditor."""
    sim = centroids @ centroids.T
    np.fill_diagonal(sim, -1)
    pairs = []
    n = len(centroids)
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= floor:
                pairs.append({"a": i, "b": j, "cosine": round(float(sim[i, j]), 4)})
    pairs.sort(key=lambda p: -p["cosine"])
    return pairs[:top]
