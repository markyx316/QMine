"""Phase 1 — template mining.  Mine once, use three times.

A *template group* is a set of queries whose phrasing all but guarantees a
shared intent: "…的拼音", "…是什么意思", "how to …", "…股价".  They are cheap to
find and unreasonably useful, because they give us a rare thing in unsupervised
work — a set of pairs we *know* belong together, without anyone having labelled
anything.

That known-together set is then spent three ways:

* **Phase 3c** judges the alpha sweep with it (does this representation keep a
  phrasing family in one cluster, or shatter it?);
* **Phase 9** turns it into the template-fragmentation metric, the direct
  measure of interpretability that replaces silhouette as a decision-maker;
* **Phase 11** draws display exemplars from it, at the median index of the hit
  set, so nobody gets to cherry-pick the demo.

The miner is deterministic: seeds from the domain profile, plus discovered
affixes ranked by a frequency-and-length score, minus anything subsumed by a
stronger sibling.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..determinism import median_index_exemplar
from ..records import TemplateGroup

_PUNCT_EDGE = re.compile(r"^[\W_]+|[\W_]+$")


def _noncapturing(pattern: str) -> str:
    """Turn ``(a|b)`` into ``(?:a|b)``.

    Capturing groups in a membership test are pure noise — pandas warns about
    them on every call and they cost a little work per row. Alternation is the
    natural way to write a phrasing family, so we fix the pattern rather than
    ask profile authors to remember the ``?:``.
    """
    return re.sub(r"\((?!\?)", "(?:", pattern)


def _field(obj: Any, name: str, default: Any = None) -> Any:
    """Read ``name`` from a pydantic model or a plain dict, indifferently.

    Domain profiles arrive as models when loaded through the config layer and as
    dicts when they come straight from YAML in a notebook, and both call sites
    are legitimate.
    """
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)



# --------------------------------------------------------------------------
# Affix discovery
# --------------------------------------------------------------------------

def mine_affixes(
    queries: Sequence[str],
    *,
    max_len: int = 6,
    min_len: int = 2,
    min_count: int = 60,
    top_k: int = 60,
) -> dict[str, list[dict[str, Any]]]:
    """Rank suffixes and prefixes by how much of the corpus they pin down.

    Score is ``count * len``: a long marker that appears often is worth more
    than a short one that appears everywhere, because "的拼音" identifies an
    intent while "的" identifies the Chinese language.
    """
    suf: Counter[str] = Counter()
    pre: Counter[str] = Counter()
    for q in queries:
        s = _PUNCT_EDGE.sub("", q)
        for n in range(min_len, max_len + 1):
            if len(s) > n:  # strict: a marker must not be the entire query
                suf[s[-n:]] += 1
                pre[s[:n]] += 1

    def _rank(c: Counter[str]) -> list[dict[str, Any]]:
        cand = [
            {"affix": a, "count": n, "score": n * len(a)}
            for a, n in c.items()
            if n >= min_count
        ]
        cand.sort(key=lambda d: (-d["score"], d["affix"]))
        return _drop_subsumed(cand, top_k)

    return {"suffixes": _rank(suf), "prefixes": _rank(pre)}


def _drop_subsumed(cands: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    """Keep "的拼音" and drop "拼音" when the longer one explains most of the shorter.

    Without this the top of the list fills with nested fragments of the same
    marker and the group definitions become mush.
    """
    kept: list[dict[str, Any]] = []
    for c in cands:
        redundant = False
        for k in kept:
            a, b = c["affix"], k["affix"]
            if (a in b or b in a) and c["count"] <= k["count"] * 1.3:
                redundant = True
                break
        if not redundant:
            kept.append(c)
        if len(kept) >= top_k:
            break
    return kept


# --------------------------------------------------------------------------
# Group construction and validation
# --------------------------------------------------------------------------

def build_groups(
    df: pd.DataFrame,
    seeds: Sequence[Any] = (),
    discovered: Sequence[dict[str, Any]] = (),
    *,
    text_col: str = "query",
    min_share: float = 0.004,
) -> list[TemplateGroup]:
    """Turn seed regexes and discovered affixes into validated template groups.

    Seeds come from the domain profile (a human's prior about the vertical);
    discovered groups come from the corpus itself.  Both are held to the same
    standard afterwards — a group that matches almost nothing, or that overlaps
    an existing group almost entirely, does not survive.
    """
    q = df[text_col].astype(str)
    n = len(q)
    groups: list[TemplateGroup] = []
    claimed = np.zeros(n, dtype=bool)

    def _add(name: str, pattern: str, hint: str, is_discovered: bool) -> None:
        pattern = _noncapturing(pattern)
        try:
            mask = q.str.contains(pattern, regex=True, na=False).to_numpy()
        except re.error:
            return
        hits = int(mask.sum())
        if hits == 0 or hits / n < min_share:
            return
        # a group that is 85%+ already claimed adds no new pinned-together pairs
        overlap = float((mask & claimed).sum()) / max(hits, 1)
        if overlap > 0.85:
            return
        idx = np.where(mask)[0]
        groups.append(
            TemplateGroup(
                name=name,
                pattern=pattern,
                intent_hint=hint,
                n_hits=hits,
                share=round(hits / n, 5),
                examples=q.iloc[idx[: min(8, idx.size)]].tolist(),
                discovered=is_discovered,
                trusted=not is_discovered,
                median_exemplar_idx=median_index_exemplar(idx),
            )
        )
        claimed[mask] = True

    for s in seeds:
        _add(_field(s, "name"), _field(s, "pattern"), _field(s, "intent_hint", ""), False)
    for d in discovered:
        affix, side = d["affix"], d.get("side", "suffix")
        pattern = f".+{re.escape(affix)}$" if side == "suffix" else f"^{re.escape(affix)}.+"
        _add(f"{side}:{affix}", pattern, d.get("intent_hint", ""), True)

    groups.sort(key=lambda g: -g.n_hits)
    return groups


def coverage(groups: Sequence[TemplateGroup], df: pd.DataFrame, *, text_col: str = "query") -> dict[str, Any]:
    """Union coverage of the corpus, and the Phase 1 quality gate around it.

    The playbook's window is 20-40%.  Below it, the miner has not looked hard
    enough and the fragmentation metric will rest on too little.  Near 100%, the
    groups have become so loose that "same pattern implies same intent" — the
    only reason they are useful — has stopped being true.
    """
    q = df[text_col].astype(str)
    n = len(q)
    union = np.zeros(n, dtype=bool)
    per: list[dict[str, Any]] = []
    for g in groups:
        m = q.str.contains(_noncapturing(g.pattern), regex=True, na=False).to_numpy()
        union |= m
        per.append({"name": g.name, "n": int(m.sum()), "share": round(float(m.mean()), 5)})
    cov = float(union.mean())
    return {
        "n_groups": len(groups),
        "union_coverage": round(cov, 5),
        "per_group": per,
        "in_window": 0.20 <= cov <= 0.40,
        "verdict": (
            "ok" if 0.20 <= cov <= 0.40
            else "too_low: mine more affixes or loosen seeds" if cov < 0.20
            else "too_high: groups are too loose to imply shared intent"
        ),
    }


def group_masks(
    groups: Sequence[TemplateGroup], df: pd.DataFrame, *, text_col: str = "query",
    trusted_only: bool = False,
) -> dict[str, np.ndarray]:
    """Boolean membership per group — the input to fragmentation scoring.

    ``trusted_only`` separates the two jobs these families do. *Coverage* wants
    every family, including loose mined affixes, because it measures how much of
    the corpus we can pin down at all. *Fragmentation* wants only families whose
    contract holds — everything matching is the same intent — because a marker
    like "是什么" spans every intent in the corpus and legitimately lands in eight
    clusters. Scoring it as fragmentation measures the marker's looseness, not
    the representation's quality, and on real data it dominates the average.
    """
    if trusted_only:
        groups = [g for g in groups if g.trusted]
    q = df[text_col].astype(str)
    return {
        g.name: q.str.contains(_noncapturing(g.pattern), regex=True, na=False).to_numpy()
        for g in groups
    }


def template_fragmentation(
    labels: np.ndarray,
    masks: dict[str, np.ndarray],
    *,
    min_group_size: int = 30,
) -> dict[str, Any]:
    """**The interpretability metric.**  1.0 is perfect; higher is worse.

    For one template group, take the distribution ``p`` of its members over
    clusters and compute the *effective number of clusters* it occupies,
    ``exp(H(p))`` where ``H`` is Shannon entropy.  A group living entirely in one
    cluster scores 1.0.  A group split evenly across three scores 3.0.  A group
    that is 85/12/3 scores about 1.5 — which is the point of using perplexity
    rather than a raw cluster count: a handful of stragglers should barely
    register, while a genuine three-way split should be loud.

    Averaged over groups, this is the number that fired the alpha=0.5
    representation in the source project after silhouette had happily approved
    it.  It measures the thing we actually want (does a coherent intent stay
    together?) instead of the thing that correlates with it right up until it
    doesn't (are clusters geometrically tight?).
    """
    per: dict[str, float] = {}
    detail: dict[str, Any] = {}
    for name, mask in masks.items():
        if mask.sum() < min_group_size:
            continue
        lab = labels[mask]
        counts = np.bincount(lab - lab.min() if lab.min() > 0 else lab)
        counts = counts[counts > 0].astype(float)
        p = counts / counts.sum()
        H = float(-(p * np.log(p)).sum())
        eff = float(np.exp(H))
        per[name] = round(eff, 4)
        top = np.sort(p)[::-1][:4]
        detail[name] = {
            "effective_clusters": round(eff, 4),
            "n_members": int(mask.sum()),
            "n_clusters_touched": int(len(counts)),
            "top_shares": [round(float(x), 4) for x in top],
        }
    mean = float(np.mean(list(per.values()))) if per else float("nan")
    return {
        "mean_fragmentation": round(mean, 4),
        "per_group": per,
        "detail": detail,
        "n_groups_scored": len(per),
    }


def select_groups_for_coverage(
    groups: Sequence[TemplateGroup],
    df: pd.DataFrame,
    *,
    text_col: str = "query",
    window: tuple[float, float] = (0.20, 0.40),
    always_keep_seeds: bool = True,
    max_groups: int = 12,
) -> tuple[list[TemplateGroup], dict[str, Any]]:
    """Choose the subset of mined groups that lands coverage inside the window.

    The window exists because both ends are failure modes.  Too little coverage
    and the fragmentation metric rests on a handful of rows; too much and the
    groups have stopped meaning "same intent" — a family matching half the
    corpus is a language detector, not a template.

    Selection is greedy by *specificity* (longest literal pattern first, then
    largest hit count), seeded groups first because a human vouched for their
    precision.  It stops as soon as adding the next group would push union
    coverage past the ceiling, so the result is deterministic and explainable:
    every included group is more specific than every excluded one.
    """
    q = df[text_col].astype(str)
    n = len(q)
    lo, hi = window

    def specificity(g: TemplateGroup) -> tuple[int, int, int]:
        literal = len(re.sub(r"[\\.*+?()\[\]{}|^$]", "", g.pattern))
        return (0 if (always_keep_seeds and not g.discovered) else 1, -literal, -g.n_hits)

    ordered = sorted(groups, key=specificity)
    chosen: list[TemplateGroup] = []
    union = np.zeros(n, dtype=bool)
    skipped: list[dict[str, Any]] = []

    for g in ordered:
        m = q.str.contains(_noncapturing(g.pattern), regex=True, na=False).to_numpy()
        cand = union | m
        cand_cov = float(cand.mean())
        is_seed = not g.discovered
        if cand_cov > hi and not (is_seed and always_keep_seeds):
            skipped.append({"name": g.name, "why": f"would push coverage to {cand_cov:.3f} > {hi}"})
            continue
        if len(chosen) >= max_groups and not is_seed:
            skipped.append({"name": g.name, "why": f"max_groups={max_groups} reached"})
            continue
        chosen.append(g)
        union = cand

    cov = float(union.mean())

    # If the seeded families ALONE overshoot the ceiling, no amount of trimming
    # mined groups can fix it — the profile's patterns are too broad, and saying
    # "coverage too high" without saying that sends the reader to the wrong knob.
    seed_only = np.zeros(n, dtype=bool)
    for g in chosen:
        if not g.discovered:
            seed_only |= q.str.contains(_noncapturing(g.pattern), regex=True, na=False).to_numpy()
    seed_cov = float(seed_only.mean())

    if cov > hi and seed_cov > hi:
        diagnosis = (
            f"the SEEDED patterns alone already match {seed_cov:.1%} of the corpus. Trimming "
            f"mined groups cannot bring this under {hi:.0%} — tighten the regexes in the domain "
            "profile instead. A family matching half the corpus is a language detector, not a "
            "phrasing template, and it will make the fragmentation metric meaningless."
        )
    elif cov > hi:
        diagnosis = (
            f"mined groups pushed coverage past {hi:.0%} (seeds alone: {seed_cov:.1%}); "
            "lower max_groups or raise the specificity floor."
        )
    elif cov < lo:
        diagnosis = (
            f"coverage {cov:.1%} is below {lo:.0%}: too few rows are pinned to a known intent, "
            "so template fragmentation will rest on a small and noisy base. Mine more affixes or "
            "add seed patterns for phrasings you know exist in this vertical."
        )
    else:
        diagnosis = ""

    report = {
        "selected": [g.name for g in chosen],
        "skipped": skipped,
        "union_coverage": round(cov, 5),
        "seed_only_coverage": round(seed_cov, 5),
        "in_window": lo <= cov <= hi,
        "window": [lo, hi],
        "diagnosis": diagnosis,
    }
    return chosen, report
