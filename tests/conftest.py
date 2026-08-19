from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict

import numpy as np
import pytest
from orng import RandomGenerator

from sequince.array_api import namespace
from sequince.posterior import TemperedPosterior
from sequince.state import ParticleState
from sequince.steps.base import StepContext
from sequince.utils import resolve_rng


def _standard_normal_log_prob(values: Any, context: Any = None) -> Any:
    del context
    array = np.asarray(values)
    return -0.5 * np.sum(array**2, axis=-1)


def _unit_log_likelihood(values: Any, context: Any = None) -> np.ndarray:
    del context
    return np.ones(np.asarray(values).shape[0])


def _zero_log_likelihood(values: Any, context: Any = None) -> np.ndarray:
    del context
    return np.zeros(np.asarray(values).shape[0])


@dataclass(frozen=True)
class BackendCase:
    name: str
    xp: Any
    particles: Any
    log_weights: Any

    def state(self) -> ParticleState:
        return ParticleState(
            particles=self.particles,
            log_weights=self.log_weights,
        )

    def rng(self, seed: int) -> RandomGenerator:
        return RandomGenerator(backend=self.name, seed=seed)


class AnalyticProposal:
    def __init__(self, n_dim: int, quadratic_scale: float = 0.0) -> None:
        self.n_dim = n_dim
        self.quadratic_scale = quadratic_scale

    def log_prob(self, particles: Any, context: Any = None) -> np.ndarray:
        del context
        values = np.asarray(particles, dtype=float)
        if values.ndim == 1:
            values = values[:, np.newaxis]
        return -self.quadratic_scale * np.sum(values**2, axis=1)

    def sample(self, n_samples: int, context: Any = None) -> np.ndarray:
        del context
        return np.zeros((n_samples, self.n_dim), dtype=float)

    def sample_and_log_prob(
        self, n_samples: int, context: Any = None
    ) -> tuple[np.ndarray, np.ndarray]:
        samples = self.sample(n_samples, context=context)
        return samples, self.log_prob(samples, context=context)


class ShiftFlow:
    xp = np

    def __init__(self, n_dim: int, shift: float = 1.5) -> None:
        self.n_dim = n_dim
        self.shift = shift
        self.fitted_particles: np.ndarray | None = None
        self.fit_kwargs: dict[str, Any] = {}
        self.fit_rng: Any = None

    def fit(
        self,
        values: Any,
        context: Any = None,
        *,
        rng: Any = None,
        **kwargs: Any,
    ) -> ShiftFlow:
        del context
        self.fit_rng = rng
        self.fit_kwargs = dict(kwargs)
        self.fitted_particles = np.asarray(values, dtype=float).copy()
        return self

    def forward(
        self, values: Any, context: Any = None
    ) -> tuple[np.ndarray, np.ndarray]:
        del context
        array = np.asarray(values, dtype=float)
        return array + self.shift, np.zeros(array.shape[0], dtype=float)

    def inverse(
        self, values: Any, context: Any = None
    ) -> tuple[np.ndarray, np.ndarray]:
        del context
        array = np.asarray(values, dtype=float)
        return array - self.shift, np.zeros(array.shape[0], dtype=float)

    def log_prob(self, values: Any, context: Any = None) -> np.ndarray:
        del context
        array = np.asarray(values, dtype=float)
        if array.ndim == 1:
            array = array[:, np.newaxis]
        return -0.5 * np.sum((array - self.shift) ** 2, axis=1)

    def sample(
        self,
        num_samples: int = 1,
        context: Any = None,
        *,
        rng: Any = None,
    ) -> np.ndarray:
        del context, rng
        return np.full((num_samples, self.n_dim), self.shift, dtype=float)

    def state_dict(self) -> dict[str, np.ndarray | None]:
        return {"fitted_particles": self.fitted_particles}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.fitted_particles = np.asarray(state["fitted_particles"])


def _backend_factory(backend: str) -> BackendCase:
    if backend == "numpy":
        np = pytest.importorskip("numpy")
        particles = np.array(
            [[0.0, 1.0], [1.0, 2.0], [2.0, 0.5], [-1.0, 1.5]],
            dtype=np.float64,
        )
        log_weights = np.zeros(particles.shape[0], dtype=np.float64)

        return BackendCase(
            backend, namespace(particles), particles, log_weights
        )

    if backend == "torch":
        torch = pytest.importorskip("torch")
        particles = torch.tensor(
            [[0.0, 1.0], [1.0, 2.0], [2.0, 0.5], [-1.0, 1.5]],
            dtype=torch.float32,
        )
        log_weights = torch.zeros(particles.shape[0], dtype=torch.float32)

        return BackendCase(
            backend, namespace(particles), particles, log_weights
        )

    if backend == "jax":
        jnp = pytest.importorskip("jax.numpy")
        particles = jnp.array(
            [[0.0, 1.0], [1.0, 2.0], [2.0, 0.5], [-1.0, 1.5]],
            dtype=jnp.float32,
        )
        log_weights = jnp.zeros(particles.shape[0], dtype=jnp.float32)

        return BackendCase(
            backend, namespace(particles), particles, log_weights
        )

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

        return BackendCase(
            backend, namespace(particles), particles, log_weights
        )

    raise ValueError(f"Unsupported backend '{backend}'.")


@pytest.fixture
def backend_factory() -> Callable[[str], BackendCase]:
    """Return the shared Array API backend constructor."""
    return _backend_factory


@pytest.fixture(params=["numpy", "torch", "jax", "cupy"])
def backend(request: pytest.FixtureRequest) -> str:
    """Name of each supported Array API backend."""
    return request.param


@pytest.fixture
def standard_normal_log_prob() -> Callable[..., Any]:
    return _standard_normal_log_prob


@pytest.fixture
def unit_log_likelihood() -> Callable[..., np.ndarray]:
    return _unit_log_likelihood


@pytest.fixture
def analytic_proposal_factory() -> type[AnalyticProposal]:
    return AnalyticProposal


@pytest.fixture
def shift_flow_factory() -> type[ShiftFlow]:
    return ShiftFlow


@pytest.fixture
def rng():
    """Seeded RandomGenerator for deterministic tests."""
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
def quadratic_posterior() -> TemperedPosterior:
    """Posterior with non-constant components for mutation tests."""
    return TemperedPosterior(
        log_prior=lambda particles: -np.sum(particles**2, axis=-1),
        log_likelihood=lambda particles: np.sum(particles, axis=-1),
    )


@pytest.fixture
def standard_normal_posterior() -> TemperedPosterior:
    return TemperedPosterior(
        _standard_normal_log_prob,
        _zero_log_likelihood,
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
