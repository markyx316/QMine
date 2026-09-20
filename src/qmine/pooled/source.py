"""Where the comparison gets its inputs, from inside a run or from a finished one.

Two callers need the same analysis: the phase node, which has `deps`, and
`qmine compare`, which has a directory. Rather than write it twice, both build a
`RunSource` and everything downstream is identical.

ARTIFACTS ARE RESOLVED THROUGH THE STORE WHEN THERE IS ONE. A generation
inherits its predecessor's artifacts, and `store.get(name).path` follows that
chain; `gen_dir / f"{name}.json"` does not. A re-render that used the raw path
shipped a workbook with 8 of 10 sheets empty and "未生成" against 13 artifacts
that existed one directory up.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

#: The canonical snapshot column AFTER p1. `cfg.data.snapshot_column` (default
#: `_snapshot`) names the column on the RAW input; `ops.audit.build_frame`
#: renames it to this. Post-p1 code must read this one.
CANONICAL_SNAPSHOT_COL = "snapshot"


class RunSource(Protocol):
    """Enough of a run to compare its snapshots."""

    def gen_dir(self) -> Path: ...
    def table(self, name: str) -> pd.DataFrame | None: ...
    def json(self, name: str) -> dict[str, Any] | None: ...
    def run_id(self) -> str: ...


class DirSource:
    """A finished generation on disk, with fall-back to earlier generations."""

    def __init__(self, gen: Path | str) -> None:
        self._gen = Path(gen)
        if not self._gen.is_dir():
            raise FileNotFoundError(f"{self._gen} is not a generation directory")
        self._run = self._gen.parent

    def gen_dir(self) -> Path:
        return self._gen

    def run_id(self) -> str:
        return self._run.name

    def _find(self, *names: str) -> Path | None:
        """Search this generation, then earlier ones — artifacts are inherited."""
        gens = sorted((p for p in self._run.glob("gen*") if p.is_dir()),
                      key=lambda p: p.name, reverse=True)
        gens = [self._gen] + [g for g in gens if g.name < self._gen.name]
        for g in gens:
            for n in names:
                p = g / n
                if p.is_file():
                    return p
        return None

    def table(self, name: str) -> pd.DataFrame | None:
        p = self._find(f"{name}.csv", f"{name}.parquet")
        if p is None:
            return None
        if p.suffix == ".parquet":
            return pd.read_parquet(p)
        return pd.read_csv(p, encoding="utf-8-sig", keep_default_na=False, na_values=[""])

    def json(self, name: str) -> dict[str, Any] | None:
        p = self._find(f"{name}.json")
        if p is None:
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None


class DepsSource:
    """A run in progress. Everything goes through the artifact store."""

    def __init__(self, deps: Any) -> None:
        self.deps = deps

    def gen_dir(self) -> Path:
        return Path(self.deps.store.gen_dir)

    def run_id(self) -> str:
        return str(getattr(self.deps, "run_id", "") or "")

    def _path(self, name: str) -> Path | None:
        try:
            if not self.deps.store.has(name):
                return None
            p = Path(self.deps.store.get(name).path)
        except (KeyError, AttributeError):
            return None
        return p if p.exists() else None

    def table(self, name: str) -> pd.DataFrame | None:
        p = self._path(name)
        if p is None:
            # labels_full.csv is written by p10 as a plain file in some paths;
            # fall back to the generation directory before giving up.
            p2 = self.gen_dir() / f"{name}.csv"
            if not p2.exists():
                return None
            p = p2
        if p.suffix == ".parquet":
            return pd.read_parquet(p)
        return pd.read_csv(p, encoding="utf-8-sig", keep_default_na=False, na_values=[""])

    def json(self, name: str) -> dict[str, Any] | None:
        try:
            if self.deps.store.has(name):
                obj = self.deps.load(name)
                return obj if isinstance(obj, dict) else None
        except Exception:  # noqa: BLE001 — a missing artifact is not an error here
            pass
        p = self.gen_dir() / f"{name}.json"
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return None
        return None
