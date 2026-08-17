"""Determinism primitives.

Playbook Principle 8 ("everything reproducible") is not a slogan here; it is an
API.  Every stochastic surface in the pipeline draws its seed from this module,
and every artifact is addressed by the hash of the inputs and parameters that
produced it.  Two consequences we rely on downstream:

* re-running any phase with unchanged inputs is a cache hit, not a recompute;
* a metric printed in a report can always be traced back to a seed policy.

The seed policy itself comes from the source K12 project: seed 0 for anything
that feeds a number into a report, seed 42 for anything that only feeds a
picture, and the ordered pair (0, 1) whenever an experiment needs two
independent draws (the replay-stability protocol of Phase 5).
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np

# --------------------------------------------------------------------------
# Seed policy (Phase 0, step 3)
# --------------------------------------------------------------------------

#: Seed for anything whose output appears as a number in a report.
SEED_METRIC = 0
#: Seed for anything whose output appears only as a picture.
SEED_VIZ = 42
#: The two draws used by every replay-stability computation.
SEED_REPLAY: tuple[int, int] = (0, 1)


@dataclass(frozen=True)
class SeedPolicy:
    """The seeds in force for one run.  Serialised into the run manifest."""

    metric: int = SEED_METRIC
    viz: int = SEED_VIZ
    replay: tuple[int, int] = SEED_REPLAY

    def as_dict(self) -> dict[str, Any]:
        return {"metric": self.metric, "viz": self.viz, "replay": list(self.replay)}


def seed_everything(seed: int = SEED_METRIC, *, torch_too: bool = True) -> None:
    """Seed every RNG we might touch.

    ``torch_too`` is honoured only if torch is importable; the pipeline must
    remain runnable on a machine with no deep-learning stack installed.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch_too:
        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():  # pragma: no cover - no CUDA on macOS
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass


@contextmanager
def temporary_seed(seed: int) -> Iterator[None]:
    """Run a block under ``seed`` and restore the previous RNG state after."""
    py_state = random.getstate()
    np_state = np.random.get_state()
    try:
        random.seed(seed)
        np.random.seed(seed)
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


def rng(seed: int = SEED_METRIC) -> np.random.RandomState:
    """A local RandomState.  Preferred over touching the global numpy RNG."""
    return np.random.RandomState(seed)


# --------------------------------------------------------------------------
# Content addressing
# --------------------------------------------------------------------------

def _canonical(obj: Any) -> Any:
    """Convert ``obj`` into something json.dumps will order deterministically."""
    if isinstance(obj, dict):
        return {str(k): _canonical(obj[k]) for k in sorted(obj, key=str)}
    if isinstance(obj, (list, tuple)):
        return [_canonical(v) for v in obj]
    if isinstance(obj, set):
        return sorted(_canonical(v) for v in obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return {"__ndarray__": hash_array(obj), "shape": list(obj.shape), "dtype": str(obj.dtype)}
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if hasattr(obj, "model_dump"):  # pydantic
        return _canonical(obj.model_dump())
    if hasattr(obj, "__dict__"):
        return _canonical({k: v for k, v in vars(obj).items() if not k.startswith("_")})
    return repr(obj)


def hash_params(params: Any, *, length: int = 16) -> str:
    """Stable hash of a parameter structure.  Key half of every cache key."""
    blob = json.dumps(_canonical(params), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:length]


def hash_array(arr: np.ndarray, *, length: int = 16) -> str:
    """Hash of an array's bytes plus its shape/dtype."""
    h = hashlib.sha256()
    h.update(str(arr.shape).encode())
    h.update(str(arr.dtype).encode())
    h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:length]


def hash_file(path: str | os.PathLike[str], *, length: int = 16) -> str:
    """Streaming hash of a file on disk."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:length]


def hash_texts(texts: list[str], *, length: int = 16) -> str:
    """Hash of a corpus.  Order-sensitive on purpose: row order is data."""
    h = hashlib.sha256()
    h.update(str(len(texts)).encode())
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:length]


def deterministic_subsample(n: int, k: int, seed: int = SEED_METRIC) -> np.ndarray:
    """Sorted indices of ``k`` rows out of ``n``, reproducible across machines.

    Every metric in the uniform panel (Phase 9) subsamples through this one
    function so that "same sub-sample" in the panel's contract is enforced by
    construction rather than by discipline.
    """
    if k >= n:
        return np.arange(n)
    return np.sort(rng(seed).choice(n, size=k, replace=False))


def median_index_exemplar(indices: np.ndarray | list[int]) -> int:
    """The median-ranked member of a hit set — Principle 7's anti-cherry-pick rule.

    Given the row indices matching some pattern, return the one sitting at the
    median position of the *sorted* set.  It is a deterministic function of the
    hit set alone, so no one — human or agent — gets to choose the exemplar that
    flatters the result.
    """
    idx = np.sort(np.asarray(list(indices), dtype=np.int64))
    if idx.size == 0:
        raise ValueError("cannot choose an exemplar from an empty hit set")
    return int(idx[idx.size // 2])
