from __future__ import annotations

from typing import Any, Callable, Dict

import numpy as np
import pytest

from sequince.state import ParticleState
from sequince.steps.base import StepContext
from sequince.utils import resolve_rng

BackendFactoryReturn = tuple[Any, Any, Any, Callable[[Any], Any]]


def _backend_factory(backend: str) -> BackendFactoryReturn:
    if backend == "numpy":
        np = pytest.importorskip("numpy")
        particles = np.array(
            [[0.0, 1.0], [1.0, 2.0], [2.0, 0.5], [-1.0, 1.5]],
            dtype=np.float64,
        )
        log_weights = np.zeros(particles.shape[0], dtype=np.float64)

        def row_sum(arr: Any) -> Any:
            return np.sum(arr, axis=1)

        return np, particles, log_weights, row_sum

    if backend == "torch":
        torch = pytest.importorskip("torch")
        particles = torch.tensor(
            [[0.0, 1.0], [1.0, 2.0], [2.0, 0.5], [-1.0, 1.5]],
            dtype=torch.float32,
        )
        log_weights = torch.zeros(particles.shape[0], dtype=torch.float32)

        def row_sum(arr: Any) -> Any:
            return torch.sum(arr, dim=1)

        return torch, particles, log_weights, row_sum

    if backend == "jax":
        jnp = pytest.importorskip("jax.numpy")
        particles = jnp.array(
            [[0.0, 1.0], [1.0, 2.0], [2.0, 0.5], [-1.0, 1.5]],
            dtype=jnp.float32,
        )
        log_weights = jnp.zeros(particles.shape[0], dtype=jnp.float32)

        def row_sum(arr: Any) -> Any:
            return jnp.sum(arr, axis=1)

        return jnp, particles, log_weights, row_sum

    if backend == "cupy":
        cupy = pytest.importorskip("cupy")
        try:
            cupy.cuda.runtime.getDeviceCount()
        except cupy.cuda.runtime.CUDARuntimeError as exc:
            pytest.skip(f"CuPy runtime unavailable: {exc}")
        particles = cupy.array(
            [[0.0, 1.0], [1.0, 2.0], [2.0, 0.5], [-1.0, 1.5]],
            dtype=cupy.float32,
        )
        log_weights = cupy.zeros(particles.shape[0], dtype=cupy.float32)

        def row_sum(arr: Any) -> Any:
            return cupy.sum(arr, axis=1)

        return cupy, particles, log_weights, row_sum

    raise ValueError(f"Unsupported backend '{backend}'.")


@pytest.fixture
def backend_factory() -> Callable[[str], BackendFactoryReturn]:
    """Return the shared Array API backend constructor."""
    return _backend_factory


@pytest.fixture
def rng():
    """Seeded ArrayRNG for deterministic tests."""
    return resolve_rng(12345)


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
        metadata: Dict[str, object] | None = None,
    ) -> StepContext:
        if shared is None:
            shared = {}
        if metadata is None:
            metadata = {}
        return StepContext(
            t=t, observation=observation, shared=shared, metadata=metadata
        )

    return _factory


@pytest.fixture
def step_context(context_factory: Callable[..., StepContext]) -> StepContext:
    """Convenience fixture for a default StepContext."""
    return context_factory()
