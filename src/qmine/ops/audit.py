"""Phase 1 — data audit.

The purpose of an audit is to *collect evidence*, not to start designing.  The
playbook is explicit about this trap: the moment you begin sketching categories
while reading the log, you have anchored the taxonomy on the first few thousand
rows you happened to see.  So this module measures and describes; Phase 2 designs.

Everything here is a pure function of the corpus, so the whole audit is a cache
hit on a re-run.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from ..determinism import SEED_METRIC, median_index_exemplar, rng

_HAN = re.compile(r"[一-鿿]")
_LATIN = re.compile(r"[A-Za-z]")
_DIGIT = re.compile(r"[0-9]")
_PUNCT = re.compile(r"[^\w\s一-鿿]")
_SPACE = re.compile(r"\s")


def _noncapturing(pattern: str) -> str:
    """Turn ``(a|b)`` into ``(?:a|b)`` — capturing groups are noise in a membership test."""
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



def char_profile(q: str) -> dict[str, Any]:
    """Per-query surface features.  Cheap, and used again as classifier features."""
    n = max(len(q), 1)
    n_han = len(_HAN.findall(q))
    n_lat = len(_LATIN.findall(q))
    n_dig = len(_DIGIT.findall(q))
    n_punct = len(_PUNCT.findall(q))
    return {
        "len": len(q),
        "n_han": n_han,
        "n_lat": n_lat,
        "n_dig": n_dig,
        "n_punct": n_punct,
        "n_space": len(_SPACE.findall(q)),
        "r_han": n_han / n,
        "has_lat": int(n_lat > 0),
        "has_dig": int(n_dig > 0),
        "has_punct": int(n_punct > 0),
        "has_qmark": int(any(c in q for c in "?？")),
        "is_mixed_script": int(n_han > 0 and n_lat > 0),
    }


def build_frame(
    queries: Sequence[str],
    *,
    reference_labels: dict[str, Sequence[str]] | None = None,
    weights: Sequence[float] | None = None,
) -> pd.DataFrame:
    """The canonical dataframe every later phase reads.

    One row per query, surface features precomputed, reference labels carried
    alongside but never merged into anything — they are for measurement only
    (Principle 1 keeps the two label systems side by side, never overwriting).
    """
    df = pd.DataFrame({"query": list(queries)})
    feats = pd.DataFrame([char_profile(q) for q in df["query"]])
    df = pd.concat([df, feats], axis=1)
    df["weight"] = list(weights) if weights is not None else 1.0
    for name, vals in (reference_labels or {}).items():
        df[name] = list(vals)
    df["row_id"] = np.arange(len(df))
    return df


def audit_corpus(
    df: pd.DataFrame,
    *,
    text_col: str = "query",
    reference_cols: Sequence[str] = (),
    top_n: int = 30,
    seed: int = SEED_METRIC,
) -> dict[str, Any]:
    """The Phase 1 profile: shape, duplication, length, script mix, head, tail."""
    q = df[text_col].astype(str)
    n = len(q)
    lens = q.str.len().to_numpy()
    dup_counts = Counter(q)
    exact_dups = sum(c for c in dup_counts.values() if c > 1) - sum(
        1 for c in dup_counts.values() if c > 1
    )

    def _pct(a: np.ndarray, p: float) -> float:
        return float(np.percentile(a, p)) if len(a) else 0.0

    report: dict[str, Any] = {
        "n_rows": int(n),
        "n_unique": int(q.nunique()),
        "duplicate_rate": round(exact_dups / max(n, 1), 5),
        "length": {
            "mean": round(float(lens.mean()), 2) if n else 0.0,
            "p10": _pct(lens, 10), "p50": _pct(lens, 50),
            "p90": _pct(lens, 90), "p99": _pct(lens, 99),
            "max": int(lens.max()) if n else 0,
        },
        "script_mix": {
            "han_only": round(float((df["r_han"] == 1).mean()), 4),
            "has_latin": round(float(df["has_lat"].mean()), 4),
            "has_digit": round(float(df["has_dig"].mean()), 4),
            "mixed_script": round(float(df["is_mixed_script"].mean()), 4),
            "has_punct": round(float(df["has_punct"].mean()), 4),
        },
        "head_queries": [
            {"query": k, "count": v} for k, v in dup_counts.most_common(top_n) if v > 1
        ][:top_n],
        "char_coverage": _char_coverage(q, top_n=top_n),
    }

    # Reference (legacy) taxonomy audit — is it a skeleton we can inherit, or
    # only a reference? A bucket defined by query *form* rather than intent is
    # the tell, and the playbook found 30% of traffic sitting in those.
    ref: dict[str, Any] = {}
    for col in reference_cols:
        if col not in df.columns:
            continue
        vc = df[col].value_counts()
        ref[col] = {
            "n_classes": int(vc.size),
            "distribution": [
                {"label": str(k), "n": int(v), "share": round(v / n, 4)} for k, v in vc.items()
            ],
            "form_defined_suspects": _form_defined_suspects(vc.index.astype(str).tolist()),
        }
    if ref:
        report["reference_taxonomy"] = ref
    return report


#: Labels containing these are catch-alls: the bucket is defined by "we could
#: not place it", which tells you nothing about what the user wanted.
_CATCHALL_WORDS = ("其他", "其它", "未知", "兜底", "misc", "other", "unknown", "n/a")

#: Labels matching these describe the *shape* of the string — length, script,
#: character class — rather than the need behind it.
_SHAPE_PATTERNS = (
    r"\d+\s*[-~－]?\s*\d*\s*字",      # "5-9字短语", "1-4字查词式"
    r"[长短]句", r"短语",
    r"含\s*(数字|符号|字母|英文)",
    r"(中英|纯英文|多语言)\s*混合?",
    r"[≥≤<>]\s*\d+",
)


def _form_defined_suspects(labels: Iterable[str]) -> list[dict[str, str]]:
    """Labels that describe the *shape* of a query rather than what the user wanted.

    These are the buckets that look like a taxonomy and behave like a landfill.
    Flagging them early tells Phase 2 which parts of a legacy system are
    salvageable structure and which parts are just unsorted traffic.

    Two distinct tells, reported separately because they mean different things:
    a *catch-all* means the taxonomy ran out of ideas, while a *shape* bucket
    means it was never about intent to begin with.
    """
    out: list[dict[str, str]] = []
    for lab in labels:
        low = lab.lower()
        if any(w in low for w in _CATCHALL_WORDS):
            out.append({"label": lab, "reason": "catch-all bucket"})
            continue
        for pat in _SHAPE_PATTERNS:
            if re.search(pat, lab):
                out.append({"label": lab, "reason": f"shape-defined (matches {pat})"})
                break
    return out


def _char_coverage(q: pd.Series, top_n: int = 30) -> dict[str, Any]:
    cats = Counter()
    for s in q.head(20000):
        for ch in s:
            cats[unicodedata.category(ch)] += 1
    total = sum(cats.values()) or 1
    return {k: round(v / total, 4) for k, v in cats.most_common(12)}


def screen_risk(
    df: pd.DataFrame,
    risk_categories: Sequence[Any],
    *,
    text_col: str = "query",
) -> dict[str, Any]:
    """First-pass risk screen (Principle 10).

    Deliberately run *before* clustering, so that Phase 7 can ask the sharper
    question: did an agent that was never told about these patterns find them
    anyway?  A risk family discovered independently is evidence; a risk family
    found because we pointed at it is only bookkeeping.
    """
    q = df[text_col].astype(str)
    out: dict[str, Any] = {"categories": [], "total_flagged": 0}
    flagged_any = np.zeros(len(df), dtype=bool)
    for cat in risk_categories:
        name = _field(cat, "name", str(cat))
        patterns = list(_field(cat, "patterns", []) or [])
        keywords = list(_field(cat, "keywords", []) or [])
        mask = np.zeros(len(df), dtype=bool)
        for pat in patterns:
            # Alternation is the natural way to write a risk pattern, so we fix
            # the capturing groups here rather than ask profile authors to
            # remember `?:` on every one.
            mask |= q.str.contains(_noncapturing(pat), regex=True, na=False).to_numpy()
        for kw in keywords:
            mask |= q.str.contains(re.escape(kw), regex=True, na=False).to_numpy()
        idx = np.where(mask)[0]
        flagged_any |= mask
        out["categories"].append(
            {
                "name": name,
                "n_hits": int(mask.sum()),
                "share": round(float(mask.mean()), 5),
                "policy": _field(cat, "policy", "isolate"),
                "exemplar": q.iloc[median_index_exemplar(idx)] if idx.size else None,
                "samples": q.iloc[idx[: min(8, idx.size)]].tolist() if idx.size else [],
            }
        )
    out["total_flagged"] = int(flagged_any.sum())
    out["total_share"] = round(float(flagged_any.mean()), 5)
    out["flag_mask_indices"] = np.where(flagged_any)[0].tolist()
    return out


def stratified_sample(
    df: pd.DataFrame,
    n: int,
    *,
    strata_cols: Sequence[str] = (),
    seed: int = SEED_METRIC,
) -> np.ndarray:
    """Proportional stratified sample, with tail strata guaranteed at least one row.

    Phase 2b's gold set exists to teach a classifier where the *boundaries* are,
    so a sample that quietly drops the rare shapes is worse than useless.
    """
    if n >= len(df):
        return np.arange(len(df))
    if not strata_cols:
        return np.sort(rng(seed).choice(len(df), n, replace=False))

    # `.reset_index(drop=True)` is load-bearing: `groupby(...).groups` yields index
    # LABELS, while the two early returns above yield POSITIONS. On a default
    # RangeIndex the two coincide, which is why every caller that passes the full
    # corpus has always worked — and why passing a slice (`df.iloc[subset]`, whose
    # labels no longer start at 0) returned indices that overflowed the caller's
    # array. Normalising here makes the return value positional in every branch.
    key = df[list(strata_cols)].astype(str).agg("|".join, axis=1).reset_index(drop=True)
    groups: dict[str, np.ndarray] = {
        str(k): np.asarray(v, dtype=np.int64) for k, v in key.groupby(key).groups.items()
    }
    r = rng(seed)

    # Floor of one per stratum guarantees the tail shapes are represented;
    # the rest is allocated in proportion to stratum size.
    picked: set[int] = set()
    for idx in groups.values():
        picked.add(int(r.choice(idx)))

    remaining = n - len(picked)
    if remaining > 0:
        total_free = sum(max(len(v) - 1, 0) for v in groups.values()) or 1
        for gid, idx in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            free = np.array([i for i in idx if i not in picked], dtype=np.int64)
            if free.size == 0:
                continue
            quota = int(round(remaining * (len(idx) - 1) / total_free))
            take = min(quota, free.size, n - len(picked))
            if take > 0:
                picked.update(int(i) for i in r.choice(free, size=take, replace=False))
            if len(picked) >= n:
                break
        # top up if rounding left us short
        if len(picked) < n:
            pool = np.array([i for i in range(len(df)) if i not in picked], dtype=np.int64)
            if pool.size:
                picked.update(
                    int(i) for i in r.choice(pool, size=min(n - len(picked), pool.size), replace=False)
                )
    return np.sort(np.array(sorted(picked)[:n], dtype=np.int64))
