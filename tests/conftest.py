"""Shared fixtures.

Everything here runs offline and deterministically: no network, no API key, no
model download.  A test that needs a GPU or a credential is a test that will be
skipped in CI and therefore is not a test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qmine.artifacts import ArtifactStore  # noqa: E402
from qmine.config import DomainProfile, QMineConfig  # noqa: E402
from qmine.graph.deps import Deps  # noqa: E402
from qmine.llm.registry import ModelRegistry  # noqa: E402
from qmine.memory.context import BlindnessFirewall  # noqa: E402
from qmine.memory.store import QMineMemory  # noqa: E402
from qmine.ops.audit import build_frame  # noqa: E402

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
DATA = Path(__file__).resolve().parents[1] / "data" / "raw" / "k12_queries_50k.csv"


@pytest.fixture(scope="session")
def k12_queries() -> list[str]:
    """A small slice of the real corpus — synthetic strings would not exercise
    the CJK n-gram paths that most of this code is about."""
    if DATA.exists():
        return pd.read_csv(DATA)["query"].astype(str).head(1500).tolist()
    return [f"测试查询{i}的拼音" for i in range(400)] + [f"词{i}是什么意思" for i in range(400)]


@pytest.fixture
def frame(k12_queries) -> pd.DataFrame:
    return build_frame(k12_queries)


@pytest.fixture
def cfg(tmp_path) -> QMineConfig:
    c = QMineConfig(fast_mode=True, offline=True, run_root=str(tmp_path / "runs"))
    c.domain = DomainProfile.load(CONFIGS / "domains" / "k12_zh.yaml")
    c.llm.provider = "mock"
    return c


@pytest.fixture
def deps(cfg, tmp_path) -> Deps:
    from langgraph.store.memory import InMemoryStore

    store = ArtifactStore(tmp_path / "run")
    registry = ModelRegistry(cfg.llm, cache_dir=tmp_path / "llm")
    memory = QMineMemory(InMemoryStore(), project="test", domain="k12_zh")
    return Deps(cfg=cfg, store=store, registry=registry, memory=memory,
                firewall=BlindnessFirewall(), run_id="test-run")


@pytest.fixture
def toy_embedding() -> np.ndarray:
    """Three well-separated blobs on the unit sphere — a structure any correct
    clustering implementation must find, so a failure here is unambiguous."""
    from sklearn.preprocessing import normalize

    rng = np.random.RandomState(0)
    centres = normalize(rng.randn(3, 32))
    X = np.vstack([c + 0.12 * rng.randn(200, 32) for c in centres])
    return normalize(X).astype(np.float32)
