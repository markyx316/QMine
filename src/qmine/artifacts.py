"""The artifact store: how a 50k x 768 matrix stays out of the agent's context.

Two ideas carry this module.

**Artifacts are referenced, never carried.**  Graph state holds :class:`ArtifactRef`
objects — a path, a hash, a shape, a one-line description.  The bytes live on
disk.  A checkpoint therefore stays kilobytes wide no matter how large the run
gets, and an agent that needs to *look* at data calls a tool that reads a slice
rather than receiving the whole thing in its prompt.

**Generations are append-only.**  Phase N+1 never overwrites Phase N.  The
playbook earned this rule the hard way: a representation that lost its bake-off
(alpha=0.5) was kept anyway, and its 107 rejected leaves later became the
phrasing-pattern library.  ``gen01``, ``gen02``, ... are directories; nothing
mutates in place.

On top of that sits a content-addressed cache keyed by
``(op_name, params_hash, input_hash)``.  Re-running a pipeline after changing
one alpha value recomputes exactly the nodes downstream of alpha and replays
everything else from cache.
"""

from __future__ import annotations

import json
import os
import threading
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Literal, TypeVar

import numpy as np
from pydantic import BaseModel, Field

from .determinism import hash_array, hash_file, hash_params

ArtifactKind = Literal[
    "table",      # parquet / csv  — one row per query
    "matrix",     # .npy           — dense float matrix
    "sparse",     # .npz           — scipy sparse
    "json",       # .json          — metrics, configs, agent outputs
    "model",      # .joblib        — fitted sklearn estimator
    "figure",     # .png / .svg
    "markdown",   # .md            — report
    "notebook",   # .ipynb
    "text",       # .txt / .log
]

T = TypeVar("T")


class ArtifactRef(BaseModel):
    """A pointer to something on disk, plus everything needed to trust it."""

    name: str = Field(description="Logical name, unique within a generation, e.g. 'emb_base'.")
    kind: ArtifactKind
    path: str = Field(description="Absolute path on disk.")
    sha256: str = Field(description="Content hash (16 hex chars).")
    bytes: int = 0
    shape: list[int] | None = None
    rows: int | None = None
    producer: str = Field(default="", description="Phase / op that created it.")
    params_hash: str = Field(default="", description="Hash of the op parameters.")
    input_hash: str = Field(default="", description="Hash of the op inputs.")
    generation: int = 0
    created_at: float = Field(default_factory=time.time)
    summary: str = Field(default="", description="One line a human (or agent) can read.")

    # -- loading helpers ----------------------------------------------------
    def load(self) -> Any:
        """Read the artifact back into memory, dispatching on ``kind``."""
        p = Path(self.path)
        if self.kind == "matrix":
            return np.load(p)
        if self.kind == "sparse":
            import scipy.sparse as sp

            return sp.load_npz(p)
        if self.kind == "table":
            import pandas as pd

            return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
        if self.kind == "json":
            return json.loads(p.read_text(encoding="utf-8"))
        if self.kind == "model":
            import joblib

            return joblib.load(p)
        return p.read_text(encoding="utf-8")

    def exists(self) -> bool:
        return Path(self.path).exists()


class ArtifactStore:
    """Filesystem-backed artifact store with generations and a memo cache.

    Layout::

        runs/<run_id>/
            manifest.json          # run-level provenance
            gen01/ gen02/ ...      # one directory per generation, append-only
            cache/                 # content-addressed, shared across generations
            index.jsonl            # every artifact ever written, in order
    """

    def __init__(self, root: str | os.PathLike[str], *, generation: int = 1) -> None:
        self.root = Path(root).resolve()
        self.generation = generation
        self.cache_dir = self.root / "cache"
        self.index_path = self.root / "index.jsonl"
        self.gen_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._refs: dict[str, ArtifactRef] = {}
        # The graph runs the top-down and bottom-up branches concurrently, and
        # both write artifacts. `_register` mutates `_refs` AND appends a line to
        # `index.jsonl`; two threads doing that at once can interleave inside one
        # index line, and a corrupt line makes `_load_index` skip a real artifact
        # on the next resume — a lost artifact that looks like a phase that never
        # ran. One lock covers both halves so the dict and the file cannot
        # disagree about what exists.
        self._lock = threading.RLock()
        self._load_index()

    # -- paths --------------------------------------------------------------
    @property
    def gen_dir(self) -> Path:
        return self.root / f"gen{self.generation:02d}"

    def new_generation(self, note: str = "") -> "ArtifactStore":
        """Open the next generation.  The old one is left untouched, forever."""
        nxt = ArtifactStore(self.root, generation=self.generation + 1)
        (nxt.gen_dir / "_why.txt").write_text(
            f"generation {nxt.generation} opened at {time.ctime()}\nreason: {note}\n",
            encoding="utf-8",
        )
        return nxt

    def path_for(self, name: str, suffix: str) -> Path:
        return self.gen_dir / f"{name}{suffix}"

    # -- writing ------------------------------------------------------------
    def put_matrix(self, name: str, arr: np.ndarray, **meta: Any) -> ArtifactRef:
        path = self.path_for(name, ".npy")
        np.save(path, arr)
        return self._register(
            name, "matrix", path, sha256=hash_array(arr), shape=list(arr.shape),
            rows=int(arr.shape[0]), **meta,
        )

    def put_sparse(self, name: str, mat: Any, **meta: Any) -> ArtifactRef:
        import scipy.sparse as sp

        path = self.path_for(name, ".npz")
        sp.save_npz(path, mat)
        return self._register(
            name, "sparse", path, shape=list(mat.shape), rows=int(mat.shape[0]), **meta
        )

    def put_table(self, name: str, df: Any, *, fmt: str = "parquet", **meta: Any) -> ArtifactRef:
        if fmt == "parquet":
            path = self.path_for(name, ".parquet")
            df.to_parquet(path, index=False)
        else:
            path = self.path_for(name, ".csv")
            df.to_csv(path, index=False, encoding="utf-8-sig")
        return self._register(
            name, "table", path, shape=list(df.shape), rows=int(df.shape[0]), **meta
        )

    def put_json(self, name: str, obj: Any, **meta: Any) -> ArtifactRef:
        path = self.path_for(name, ".json")
        path.write_text(
            json.dumps(obj, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8"
        )
        return self._register(name, "json", path, **meta)

    def put_model(self, name: str, model: Any, **meta: Any) -> ArtifactRef:
        import joblib

        path = self.path_for(name, ".joblib")
        joblib.dump(model, path)
        return self._register(name, "model", path, **meta)

    def put_text(self, name: str, text: str, *, kind: ArtifactKind = "text", suffix: str = ".txt", **meta: Any) -> ArtifactRef:
        path = self.path_for(name, suffix)
        path.write_text(text, encoding="utf-8")
        return self._register(name, kind, path, **meta)

    def put_markdown(self, name: str, text: str, **meta: Any) -> ArtifactRef:
        return self.put_text(name, text, kind="markdown", suffix=".md", **meta)

    def put_figure_path(self, name: str) -> Path:
        """Hand a matplotlib figure a path; register it afterwards with :meth:`register_file`."""
        return self.path_for(name, ".png")

    def register_file(self, name: str, path: str | os.PathLike[str], kind: ArtifactKind, **meta: Any) -> ArtifactRef:
        return self._register(name, kind, Path(path), **meta)

    def _register(
        self,
        name: str,
        kind: ArtifactKind,
        path: Path,
        *,
        sha256: str | None = None,
        **meta: Any,
    ) -> ArtifactRef:
        ref = ArtifactRef(
            name=name,
            kind=kind,
            path=str(path),
            sha256=sha256 or hash_file(path),
            bytes=path.stat().st_size if path.exists() else 0,
            generation=self.generation,
            **{k: v for k, v in meta.items() if k in ArtifactRef.model_fields},
        )
        with self._lock:
            self._refs[name] = ref
            with open(self.index_path, "a", encoding="utf-8") as fh:
                fh.write(ref.model_dump_json() + "\n")
        return ref

    # -- reading ------------------------------------------------------------
    def get(self, name: str) -> ArtifactRef:
        with self._lock:
            if name not in self._refs:
                raise KeyError(f"no artifact named {name!r}; have {sorted(self._refs)}")
        return self._refs[name]

    def has(self, name: str) -> bool:
        return name in self._refs and self._refs[name].exists()

    def load(self, name: str) -> Any:
        return self.get(name).load()

    def names(self) -> list[str]:
        return sorted(self._refs)

    def _load_index(self) -> None:
        if not self.index_path.exists():
            return
        for line in self.index_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                ref = ArtifactRef.model_validate_json(line)
            except Exception:
                continue
            # later entries win; only surface artifacts we can still read
            if ref.generation <= self.generation and Path(ref.path).exists():
                self._refs[ref.name] = ref

    # -- memoisation --------------------------------------------------------
    def memoize(
        self,
        op: str,
        params: Any,
        input_hash: str,
        compute: Callable[[], T],
        *,
        loader: Callable[[Path], T] | None = None,
        saver: Callable[[Path, T], None] | None = None,
        suffix: str = ".npy",
    ) -> tuple[T, bool]:
        """Content-addressed memoisation of an expensive op.

        Returns ``(value, was_cached)``.  The cache key is the triple that fully
        determines the result: which op, with which parameters, over which
        inputs.  Change any one and you get a miss; change none and you never
        pay for the recompute.
        """
        key = f"{op}-{hash_params(params)}-{input_hash}"
        blob = self.cache_dir / f"{key}{suffix}"
        _loader = loader or (lambda p: np.load(p))
        _saver = saver or (lambda p, v: np.save(p, v))
        if blob.exists():
            try:
                return _loader(blob), True
            except Exception:
                blob.unlink(missing_ok=True)
        value = compute()
        _saver(blob, value)
        return value, False

    def cache_key_exists(self, op: str, params: Any, input_hash: str, suffix: str = ".npy") -> bool:
        return (self.cache_dir / f"{op}-{hash_params(params)}-{input_hash}{suffix}").exists()

    # -- housekeeping -------------------------------------------------------
    def size_report(self) -> dict[str, Any]:
        total = sum(f.stat().st_size for f in self.root.rglob("*") if f.is_file())
        by_gen: dict[str, int] = {}
        for d in sorted(self.root.glob("gen*")):
            by_gen[d.name] = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
        return {"total_bytes": total, "by_generation": by_gen, "n_artifacts": len(self._refs)}

    def copy_into(self, name: str, dest_dir: str | os.PathLike[str]) -> Path:
        ref = self.get(name)
        dest = Path(dest_dir) / Path(ref.path).name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ref.path, dest)
        return dest


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    if isinstance(obj, set):
        return sorted(obj)
    return str(obj)


def latest_generation(root: str | os.PathLike[str]) -> int:
    """The highest generation directory present under a run, or 1 if none.

    Both resume paths used to hardcode generation 1. After `new-generation` —
    the move the gate-halt message itself tells you to make — that reopened the
    OLD generation's thread, found `halted=True`, logged "stays halted" and
    exited in seconds having done nothing, while the new generation sat
    untouched. The thread id is per-generation, so resuming the wrong number is
    not a near miss; it is a different run.
    """
    p = Path(root)
    gens = [int(d.name[3:]) for d in p.glob("gen[0-9][0-9]")
            if d.is_dir() and d.name[3:].isdigit()]
    return max(gens) if gens else 1


def resolved_config_path(root: str | os.PathLike[str], generation: int) -> Path | None:
    """The newest `config.resolved.yaml` at or below `generation`.

    `new_generation` opens a directory and a note, not a config — so the config
    of gen02 is gen01's. Looking only in the current generation finds nothing and
    sends `run --resume` down its "nothing to resume" path, straight into the
    refuse-an-existing-run-id guard.
    """
    p = Path(root)
    for g in range(generation, 0, -1):
        cand = p / f"gen{g:02d}" / "config.resolved.yaml"
        if cand.exists():
            return cand
    return None


def merge_artifacts(left: dict[str, ArtifactRef], right: dict[str, ArtifactRef]) -> dict[str, ArtifactRef]:
    """Reducer for the ``artifacts`` channel of the graph state.

    Last writer wins per key, which is what we want: a later generation's
    ``emb_hybrid`` supersedes an earlier one *in state* while both remain on
    disk under their own generation directories.
    """
    out = dict(left or {})
    out.update(right or {})
    return out
