"""Finding a run's results and reading them in BOUNDED pieces.

A finished pooled run leaves a 190KB report, thirty CSVs, two workbooks and
seven figures. None of that can go into a chat context, and the failure mode of
trying is not a truncation warning — it is an assistant that saw the first
2,000 rows of one table and answers as if it had read the study.

So nothing here returns a file. Every reader takes a budget, returns rows under
it, and SAYS WHAT IT LEFT OUT. A caller that needs more asks again with a
narrower filter, which is also how a person would read it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

#: Default row budget for a table read. Chosen so a full payload stays in the
#: low thousands of tokens; every reader reports the true row count beside it.
DEFAULT_LIMIT = 40
MAX_LIMIT = 200
#: Hard character ceiling on any single payload, applied after shaping.
MAX_CHARS = 12_000


class RunNotFound(FileNotFoundError):
    pass


@dataclass(frozen=True)
class RunRef:
    run_id: str
    generation: str
    root: Path

    @property
    def gen_dir(self) -> Path:
        return self.root / self.run_id / self.generation

    @property
    def pooled_dir(self) -> Path:
        return self.gen_dir / "pooled"

    @property
    def tables_dir(self) -> Path:
        return self.pooled_dir / "tables"

    def has_comparison(self) -> bool:
        return (self.tables_dir / "summary.json").is_file()


def list_runs(run_root: str | Path = "runs") -> list[dict[str, Any]]:
    """Every run on disk, with just enough to choose one."""
    root = Path(run_root)
    if not root.is_dir():
        return []
    out = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        gens = sorted((g.name for g in d.glob("gen*") if g.is_dir()), reverse=True)
        if not gens:
            continue
        ref = RunRef(d.name, gens[0], root)
        summ = _read_json(ref.gen_dir / "run_summary.json") or {}
        out.append({
            "run_id": d.name,
            "generations": gens,
            "latest": gens[0],
            "mode": summ.get("mode"),
            "provider": (summ.get("llm_usage") or {}).get("provider"),
            "halted": summ.get("halted"),
            "phases_completed": len(summ.get("completed_phases") or []),
            "has_cross_snapshot_comparison": ref.has_comparison(),
        })
    return out


def resolve(run_id: str, generation: str | None = None,
            run_root: str | Path = "runs") -> RunRef:
    root = Path(run_root)
    d = root / run_id
    if not d.is_dir():
        known = [p.name for p in root.iterdir() if p.is_dir()] if root.is_dir() else []
        raise RunNotFound(f"no run {run_id!r} under {root}. Known: {sorted(known)[:20]}")
    gens = sorted((g.name for g in d.glob("gen*") if g.is_dir()), reverse=True)
    if not gens:
        raise RunNotFound(f"{d} has no generations yet")
    gen = generation or gens[0]
    if gen not in gens:
        raise RunNotFound(f"{run_id} has no {gen}; generations are {gens}")
    return RunRef(run_id, gen, root)


def _read_json(p: Path) -> dict[str, Any] | None:
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


@lru_cache(maxsize=32)
def _cached_csv(path: str, mtime: float) -> pd.DataFrame:
    """Cached on (path, mtime) so a re-run of `compare` is picked up, not served stale."""
    return pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False, na_values=[""])


def read_table(ref: RunRef, name: str) -> pd.DataFrame:
    p = ref.tables_dir / f"{name}.csv"
    if not p.is_file():
        raise FileNotFoundError(
            f"{ref.run_id}/{ref.generation} has no table {name!r}. "
            f"Available: {available_tables(ref)}")
    return _cached_csv(str(p), p.stat().st_mtime)


def available_tables(ref: RunRef) -> list[str]:
    if not ref.tables_dir.is_dir():
        return []
    return sorted(p.stem for p in ref.tables_dir.glob("*.csv"))


def summary(ref: RunRef) -> dict[str, Any]:
    s = _read_json(ref.tables_dir / "summary.json")
    if s is None:
        raise RunNotFound(
            f"{ref.run_id}/{ref.generation} has no cross-snapshot comparison yet. "
            f"Build one with `qmine compare {ref.run_id}` — it makes no model call.")
    return s


def run_summary(ref: RunRef) -> dict[str, Any]:
    return _read_json(ref.gen_dir / "run_summary.json") or {}


def shape(t: pd.DataFrame, *, where: dict[str, Any] | None = None,
          columns: list[str] | None = None, sort: str | None = None,
          descending: bool = True, limit: int = DEFAULT_LIMIT) -> dict[str, Any]:
    """Filter / project / sort / truncate, and REPORT WHAT WAS LEFT OUT.

    The last part is the point. A silently truncated table is read as the whole
    table, and every share computed from it is wrong in a way nothing on the
    page reveals.
    """
    total = len(t)
    notes: list[str] = []
    for col, val in (where or {}).items():
        if col not in t.columns:
            notes.append(f"ignored filter on {col!r}: no such column")
            continue
        series = t[col].astype(str)
        if isinstance(val, (list, tuple)):
            t = t[series.isin([str(v) for v in val])]
        elif isinstance(val, str) and val.startswith(("<", ">")):
            num = pd.to_numeric(t[col], errors="coerce")
            try:
                bound = float(val[1:])
            except ValueError:
                notes.append(f"ignored filter {col}{val}: not a number")
                continue
            t = t[num > bound] if val[0] == ">" else t[num < bound]
        else:
            t = t[series == str(val)]
    matched = len(t)
    if sort and sort in t.columns:
        key = pd.to_numeric(t[sort], errors="coerce")
        t = t.assign(_k=key).sort_values("_k", ascending=not descending,
                                         na_position="last").drop(columns="_k")
    elif sort:
        notes.append(f"ignored sort on {sort!r}: no such column")
    if columns:
        keep = [c for c in columns if c in t.columns]
        missing = [c for c in columns if c not in t.columns]
        if missing:
            notes.append(f"no such column(s): {missing}")
        if keep:
            t = t[keep]
    limit = max(1, min(int(limit), MAX_LIMIT))
    shown = t.head(limit)
    return {
        "rows": json.loads(shown.to_json(orient="records", force_ascii=False)),
        "rows_shown": len(shown),
        "rows_matched": matched,
        "rows_in_table": total,
        "truncated": matched > len(shown),
        "notes": notes,
    }


def clip(payload: dict[str, Any], max_chars: int = MAX_CHARS) -> dict[str, Any]:
    """Last-resort ceiling. Drops ROWS, never keys, and says how many."""
    text = json.dumps(payload, ensure_ascii=False)
    if len(text) <= max_chars:
        return payload
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        payload["_clipped"] = f"payload was {len(text):,} chars, over the {max_chars:,} ceiling"
        return payload
    keep = len(rows)
    while keep > 1 and len(json.dumps({**payload, "rows": rows[:keep]},
                                      ensure_ascii=False)) > max_chars:
        keep = keep * 3 // 4
    payload["rows"] = rows[:keep]
    payload["truncated"] = True
    payload["_clipped"] = (f"{len(rows) - keep} more row(s) dropped to stay under the "
                           f"{max_chars:,}-character ceiling — narrow the filter or "
                           "ask for fewer columns")
    return payload
