"""Thin re-exports so ops modules do not reach across the package for seeds."""

from __future__ import annotations

from ..determinism import SEED_METRIC, deterministic_subsample


def subsample_indices(n: int, k: int, seed: int = SEED_METRIC):
    return deterministic_subsample(n, k, seed)
