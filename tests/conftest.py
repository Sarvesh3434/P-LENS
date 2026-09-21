"""Shared fixtures: small fast config + network for component tests."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.config import Config
from network.graph import EvolvingNetwork


@pytest.fixture
def cfg() -> Config:
    return Config(
        network_size=12,
        simulation_steps=8,
        initial_topology="ring",
        failure_rate_node=0.0,
        failure_rate_edge=0.0,
        delay_k=2,
        cooldown_steps=2,
        max_mutations_per_step=2,
        mutation_budget_initial=3.0,
        mutation_budget_max=8.0,
    )


@pytest.fixture
def net(cfg: Config) -> EvolvingNetwork:
    rng = np.random.default_rng(0)
    return EvolvingNetwork(cfg, rng)


@pytest.fixture
def demand(cfg: Config) -> np.ndarray:
    """Small deterministic demand matrix."""
    rng = np.random.default_rng(42)
    d = rng.random((cfg.network_size, cfg.network_size)) * 0.1
    np.fill_diagonal(d, 0.0)
    d = d + d.T
    np.fill_diagonal(d, 0.0)
    return d
