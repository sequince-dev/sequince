from __future__ import annotations

from typing import Callable, Dict

import numpy as np
import pytest

from sequince.state import ParticleState
from sequince.steps.base import StepContext
from sequince.utils import _ensure_rng


@pytest.fixture
def rng():
    """Seeded ArrayRNG for deterministic tests."""
    return _ensure_rng(12345)


@pytest.fixture
def simple_particles() -> np.ndarray:
    """Deterministic particle cloud used across tests."""
    return np.array(
        [
            [0.0, 0.0],
            [1.0, -1.0],
            [2.0, 2.0],
            [-1.5, 0.5],
        ],
        dtype=float,
    )


@pytest.fixture
def particle_state(simple_particles: np.ndarray) -> ParticleState:
    """ParticleState with non-uniform log weights for testing."""
    n = simple_particles.shape[0]
    log_weights = np.linspace(-0.1, -0.4, n)
    metadata = {"temperature": 0.0}
    return ParticleState(
        particles=simple_particles, log_weights=log_weights, metadata=metadata
    )


@pytest.fixture
def context_factory() -> Callable[..., StepContext]:
    """Factory returning a fresh StepContext instance."""

    def _factory(
        t: int = 0,
        *,
        observation: object = None,
        shared: Dict[str, object] | None = None,
    ) -> StepContext:
        if shared is None:
            shared = {}
        return StepContext(t=t, observation=observation, shared=shared)

    return _factory


@pytest.fixture
def step_context(context_factory: Callable[..., StepContext]) -> StepContext:
    """Convenience fixture for a default StepContext."""
    return context_factory()
