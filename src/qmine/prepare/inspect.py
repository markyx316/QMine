"""Measure each input file before anyone decides what to do with it.

Everything here is MECHANICAL. No model sees a file until this has run, and the
plan an agent proposes is a plan over these measurements — which is what makes
the plan checkable and what keeps "the agent looked at the data" from meaning
"the agent guessed".

The measurements are chosen to answer the questions a preparation step actually
has to answer: which column is the query, which is traffic, which snapshot is
this, is it a head or a tail sample, does it carry a product's own text rather
than a user's, and does it overlap the other inputs.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

#: A string with no letter, digit or CJK character is not a query. Deliberately
#: not a regex: `\w` is ASCII-only under pandas 3's RE2 and read every Chinese
#: query as empty.
def has_content(s: object) -> bool:
    """True when the cell holds something a person could have typed.

    `str(float("nan"))` is the three characters `nan`, every one of them
    alphanumeric — so a blank cell read as content and survived `drop_empty`,
    entering the mining as the literal string "nan".
    """
    if s is None:
        return False
    if isinstance(s, float) and s != s:      # NaN
        return False
    t = str(s)
    return t.lower() not in ("nan", "none", "<na>", "null") and any(ch.isalnum() for ch in t)


_DATEY = re.compile(r"^(?:19|20)\d{2}[-/]?\d{2}[-/]?\d{2}$")
_HAN = re.compile(r"[一-鿿]")




def is_textual(dtype: str) -> bool:
    """pandas 3 reads a text column as `str`, NOT `object`.

    A `dtype == "object"` test found the query column in every CSV and in none of
    the xlsx exports, so the inspector reported "no column looks like free text"
    on files whose first column is nothing but free text.
    """
    d = str(dtype).lower()
    return d.startswith(("object", "str", "string"))

@dataclass
class ColumnProfile:
    name: str
    dtype: str
    n_missing: int
    n_unique: int
    is_constant: bool
    constant_value: str = ""
    mean_len: float = 0.0
    numeric_share: float = 0.0
    monotone_desc: bool = False
    samples: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class InputProfile:
    path: str
    n_rows: int
    columns: list[ColumnProfile]
    text_candidates: list[str]
    weight_candidates: list[str]
    snapshot_candidates: list[str]
    suggested_tag: str
    duplicate_rate: float = 0.0
    len_median: float = 0.0
    len_p90: float = 0.0
    han_share: float = 0.0
    latin_share: float = 0.0
    content_free_rate: float = 0.0
    weight_sorted_desc: bool = False
    weight_gini: float = 0.0
    repeated_prefixes: list[tuple[str, float]] = field(default_factory=list)
    short_repeated_strings: list[tuple[str, int]] = field(default_factory=list)
    #: Share of THIS file's distinct strings that also occur in each other input.
    #: Recorded because it is informative, and kept out of the axis decision
    #: because it MEASURABLY does not settle it: two random 1w samples of the
    #: same surface a year apart overlap 0.02%, while two head exports of that
    #: same surface overlap 63.3%, and a different surface overlaps 0.00%. The
    #: number separates head from tail, not time from interface.
    overlap: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "columns"}
        d["columns"] = [c.to_dict() for c in self.columns]
        return d

    def brief(self) -> dict[str, Any]:
        """The compact form handed to an agent — measurements, no raw rows."""
        return {
            "path": Path(self.path).name,
            "n_rows": self.n_rows,
            "columns": [{"name": c.name, "dtype": c.dtype, "n_unique": c.n_unique,
                         "is_constant": c.is_constant,
                         "constant_value": c.constant_value[:40],
                         "mean_len": round(c.mean_len, 1),
                         "samples": [s[:40] for s in c.samples[:3]]}
                        for c in self.columns],
            "text_candidates": self.text_candidates,
            "weight_candidates": self.weight_candidates,
            "snapshot_candidates": self.snapshot_candidates,
            "suggested_tag": self.suggested_tag,
            "duplicate_rate": round(self.duplicate_rate, 4),
            "len_median": self.len_median, "len_p90": self.len_p90,
            "han_share": round(self.han_share, 3),
            "latin_share": round(self.latin_share, 3),
            "content_free_rate": round(self.content_free_rate, 4),
            "weight_sorted_desc": self.weight_sorted_desc,
            "weight_gini": round(self.weight_gini, 3),
            "repeated_prefixes": [[p, round(f, 4)] for p, f in self.repeated_prefixes],
            "short_repeated_strings": self.short_repeated_strings[:12],
            "overlap": {k: round(v, 4) for k, v in self.overlap.items()},
            "notes": self.notes,
        }


def read_any(path: str | Path) -> pd.DataFrame:
    p = str(path)
    if p.endswith((".xlsx", ".xls")):
        return pd.read_excel(p)
    if p.endswith(".parquet"):
        return pd.read_parquet(p)
    if p.endswith((".jsonl", ".json")):
        return pd.read_json(p, lines=p.endswith(".jsonl"))
    return pd.read_csv(p)


def _gini(w: np.ndarray) -> float:
    w = np.sort(np.asarray(w, dtype=float))
    w = w[np.isfinite(w) & (w >= 0)]
    n = len(w)
    if n < 2 or w.sum() <= 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * w).sum()) / (n * w.sum()) - (n + 1) / n)


def _profile_column(s: pd.Series, name: str) -> ColumnProfile:
    nn = s.dropna()
    txt = nn.astype(str)
    uniq = int(nn.nunique())
    numeric = pd.to_numeric(nn, errors="coerce")
    mono = False
    if numeric.notna().mean() > 0.95 and len(numeric) > 2:
        mono = bool((numeric.dropna().diff().dropna() <= 0).all())
    return ColumnProfile(
        name=name, dtype=str(s.dtype), n_missing=int(s.isna().sum()), n_unique=uniq,
        is_constant=bool(uniq == 1),
        constant_value=str(nn.iloc[0]) if uniq == 1 and len(nn) else "",
        mean_len=float(txt.str.len().mean()) if len(txt) else 0.0,
        numeric_share=float(numeric.notna().mean()) if len(nn) else 0.0,
        monotone_desc=mono,
        samples=[str(x) for x in nn.head(4).tolist()])


def _prefixes(texts: pd.Series, n_rows: int) -> list[tuple[str, float]]:
    """Leading substrings shared by enough rows to be a product's own wrapper.

    A wrapper is STRIPPED, never deleted: "我想咨询" in front of a real question
    is the product's framing around a user's words, and dropping the row throws
    the words away with it.

    EXTENDED TO ITS FULL LENGTH, AND NO FURTHER. Two failures bound this:

    * A scan that stopped at three characters returned `我想咨` for a wrapper
      that is `我想咨询`; stripping three of four leaves a stray `询` on every
      row and the collision merge that follows finds nothing to merge.
    * Extending without a stop swallows the query itself — on an export where
      every row is `我想咨询头痛`, the "wrapper" becomes the whole string.

    So a candidate grows while it keeps essentially all of its rows AND while it
    still leaves at least two characters on the SHORTEST row it matches. The
    earlier version enforced the second rule by only counting strings longer
    than `k + 2`, which also stopped the growth: for six-character rows nothing
    was counted at `k = 4`, and the wrapper was reported one character short.
    """
    vals = [str(t) for t in texts]
    out: list[tuple[str, float]] = []
    seeds = Counter(t[:3] for t in vals if len(t) > 3)
    for seed, n in seeds.most_common(8):
        if n / max(n_rows, 1) < 0.02:
            continue
        best, best_n = seed, n
        for k in range(4, 17):
            matching = [t for t in vals if t.startswith(best)]
            # A prefix has to leave something behind, or it IS the query.
            if not matching or k + 2 > min(len(t) for t in matching):
                break
            cand = Counter(t[:k] for t in matching if len(t) >= k).most_common(1)
            if not cand:
                break
            pre, cn = cand[0]
            if cn < 0.9 * best_n:
                break
            best, best_n = pre, cn
        if any(best.startswith(q) or q.startswith(best) for q, _ in out):
            continue
        out.append((best, best_n / max(n_rows, 1)))
    out.sort(key=lambda x: (-x[1], -len(x[0])))
    return out[:8]


def profile_input(path: str | Path, *, text_column: str | None = None,
                  others: dict[str, set[str]] | None = None) -> InputProfile:
    df = read_any(path)
    cols = [_profile_column(df[c], str(c)) for c in df.columns]
    n = len(df)

    text_cands = [c.name for c in sorted(cols, key=lambda c: -c.mean_len)
                  if is_textual(c.dtype) and c.n_unique > max(10, 0.2 * n)
                  and c.mean_len >= 2]
    named = [c.name for c in cols if str(c.name).lower() in
             ("query", "queries", "keyword", "search_word", "q", "text", "原始query",
              "query_raw", "original_query", "搜索词", "查询")]
    text_cands = list(dict.fromkeys(named + text_cands))
    text = text_column or (text_cands[0] if text_cands else None)

    #: A DATE IS NOT A WEIGHT. `event_day` is numeric, non-constant and often the
    #: only other numeric column, so it was picked as traffic and `pv_raw`
    #: became a sum of dates — silently, because the number is plausible.
    def _datey(c: ColumnProfile) -> bool:
        if str(c.name).lower() in ("event_day", "date", "dt", "day", "snapshot",
                                   "period", "month", "week", "year"):
            return True
        vals = [str(v) for v in c.samples if str(v).strip()]
        return bool(vals) and all(
            _DATEY.match(v) or re.fullmatch(r"(?:19|20)\d{2}", v) for v in vals)

    weight_cands = [c.name for c in cols
                    if c.numeric_share > 0.9 and not c.is_constant
                    and c.name != text and not _datey(c)
                    and str(c.name).lower() not in ("row_id", "id", "index", "rank",
                                                    "row_id_all", "序号", "排名")]
    named_w = [c for c in weight_cands if any(
        k in str(c).lower() for k in ("pv", "num", "count", "freq", "weight", "cnt", "uv"))]
    weight_cands = list(dict.fromkeys(named_w + weight_cands))

    snap_cands = [c.name for c in cols if c.is_constant] + \
                 [c.name for c in cols if str(c.name).lower() in
                  ("event_day", "date", "dt", "snapshot", "day", "period")]
    snap_cands = list(dict.fromkeys(snap_cands))

    prof = InputProfile(
        path=str(path), n_rows=n, columns=cols, text_candidates=text_cands,
        weight_candidates=weight_cands, snapshot_candidates=snap_cands,
        suggested_tag=_suggest_tag(path, df, cols))
    if text is None:
        prof.notes.append("no column looks like free-text queries — say which one with "
                          "--text-column, or this file cannot be prepared")
        return prof

    t = df[text].astype(str)
    prof.duplicate_rate = float(1 - t.nunique() / max(n, 1))
    lens = t.str.len()
    prof.len_median = float(lens.median())
    prof.len_p90 = float(lens.quantile(.9))
    prof.han_share = float(t.map(lambda s: bool(_HAN.search(s))).mean())
    prof.latin_share = float(t.str.contains(r"[A-Za-z]", regex=True).mean())
    prof.content_free_rate = float((~t.map(has_content)).mean())
    prof.repeated_prefixes = _prefixes(t, n)
    short = Counter(s for s in t if len(s) <= 12)
    prof.short_repeated_strings = [(s, c) for s, c in short.most_common(20) if c >= 3][:20]
    if weight_cands:
        w = pd.to_numeric(df[weight_cands[0]], errors="coerce")
        prof.weight_sorted_desc = bool((w.dropna().diff().dropna() <= 0).all())
        prof.weight_gini = _gini(w.to_numpy())
    if others:
        mine = set(t)
        for name, other in others.items():
            if not other:
                continue
            share = len(mine & other) / max(len(mine), 1)
            prof.overlap[name] = share
            prof.notes.append(
                f"{100 * share:.2f}% of this file's distinct strings also occur in {name}")
    if prof.weight_sorted_desc:
        # SORTED IS THE SIGNAL; CONCENTRATION IS A MODIFIER. A head export is
        # sorted because it was cut at a rank, and a random sample is not — but
        # a head of a flat-tailed distribution has a low Gini, so requiring both
        # missed real head exports and reported nothing at all about them.
        prof.notes.append(
            "weights are sorted descending — this looks like a HEAD (top-N) export, "
            "not a random sample"
            + ("，and highly concentrated" if prof.weight_gini > 0.4
               else "，though the weights are fairly flat"))
    elif weight_cands and not prof.weight_sorted_desc:
        prof.notes.append("weights are not sorted — this looks like a random or "
                          "de-duplicated sample rather than a head export")
    if prof.duplicate_rate > 0.05:
        prof.notes.append(f"{100 * prof.duplicate_rate:.1f}% of rows repeat a string that "
                          "occurs earlier — the export may be per-day rather than per-query")
    return prof


def _suggest_tag(path: str | Path, df: pd.DataFrame, cols: list[ColumnProfile]) -> str:
    """The same rule `p0` uses, so a prepared corpus tags the way a raw one would."""
    for want in ("event_day", "date", "dt", "snapshot"):
        for c in cols:
            if str(c.name).lower() == want and c.is_constant:
                return str(c.constant_value)
    for c in cols:
        if c.is_constant and _DATEY.match(str(c.constant_value)):
            return str(c.constant_value)
    m = re.search(r"(\d{6,8})", Path(path).stem)
    return m.group(1) if m else Path(path).stem


def profile_inputs(paths: list[str], *, text_column: str | None = None) -> list[InputProfile]:
    """Profile every input, and tell each one about the others' strings."""
    texts: dict[str, set[str]] = {}
    for p in paths:
        try:
            df = read_any(p)
        except Exception:  # noqa: BLE001
            continue
        first = profile_input(p, text_column=text_column)
        col = text_column or (first.text_candidates[0] if first.text_candidates else None)
        texts[Path(p).name] = set(df[col].astype(str)) if col and col in df.columns else set()
    out = []
    for p in paths:
        others = {k: v for k, v in texts.items() if k != Path(p).name}
        out.append(profile_input(p, text_column=text_column, others=others))
    return out
