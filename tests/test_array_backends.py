from __future__ import annotations

from typing import Any, Callable, Tuple

import pytest
from orng import ArrayRNG

from sequince._array_api import namespace
from sequince.prebuilt_samplers import make_standard_bayesian_smc
from sequince.state import ParticleState

BackendFactoryReturn = Tuple[Any, Any, Any, Callable[[Any], Any]]


def _backend_factory(
    backend: str,
) -> BackendFactoryReturn:
    if backend == "numpy":
        np = pytest.importorskip("numpy")

        dtype = np.float64

        particles = np.array([[0.0, 1.0], [1.0, 2.0]], dtype=dtype)
        log_weights = np.zeros(particles.shape[0], dtype=dtype)

        def row_sum(arr: Any) -> Any:
            return np.sum(arr, axis=1)

        return np, particles, log_weights, row_sum

    if backend == "torch":
        torch = pytest.importorskip("torch")

        dtype = torch.float32
        particles = torch.tensor([[0.0, 1.0], [1.0, 2.0]], dtype=dtype)
        log_weights = torch.zeros(particles.shape[0], dtype=dtype)

        def row_sum(arr: Any) -> Any:
            return torch.sum(arr, dim=1)

        return torch, particles, log_weights, row_sum

    if backend == "jax":
        jnp = pytest.importorskip("jax.numpy")

        dtype = jnp.float32
        particles = jnp.array([[0.0, 1.0], [1.0, 2.0]], dtype=dtype)
        log_weights = jnp.zeros(particles.shape[0], dtype=dtype)

        def row_sum(arr: Any) -> Any:
            return jnp.sum(arr, axis=1)

        return jnp, particles, log_weights, row_sum

    if backend == "cupy":
        cupy = pytest.importorskip("cupy")
        try:
            cupy.cuda.runtime.getDeviceCount()
        except cupy.cuda.runtime.CUDARuntimeError as exc:
            pytest.skip(f"CuPy runtime unavailable: {exc}")

        dtype = cupy.float32
        particles = cupy.array([[0.0, 1.0], [1.0, 2.0]], dtype=dtype)
        log_weights = cupy.zeros(particles.shape[0], dtype=dtype)

        def row_sum(arr: Any) -> Any:
            return cupy.sum(arr, axis=1)

        return cupy, particles, log_weights, row_sum

    raise ValueError(f"Unsupported backend '{backend}'.")


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax", "cupy"])
def test_sequince_runs_with_array_backends(backend: str) -> None:
    out = _backend_factory(backend)
    xp, particles, log_weights, row_sum = out
    backend_ns = namespace(particles)

    initial_state = ParticleState(
        particles=particles,
        log_weights=log_weights,
    )

    smc, _, _ = make_standard_bayesian_smc(
        log_prior=lambda x, ctx: xp.zeros(x.shape[0], dtype=x.dtype),
        log_likelihood=lambda x, ctx: -0.5 * row_sum(x**2),
        mutation="none",
    )

    results = smc.run(
        initial_state,
        num_steps=1,
        rng=ArrayRNG(backend=backend, seed=1234),
    )

    assert len(results) == 1
    final_state = results.final_state

    assert final_state.particles.shape == particles.shape
    assert namespace(final_state.particles) is backend_ns

    samples = results.posterior_samples(
        n_samples=particles.shape[0],
        rng=ArrayRNG(backend=backend, seed=4321),
    )
    assert samples.shape[0] == particles.shape[0]
    assert namespace(samples) is backend_ns

    mean = results.weighted_mean()
    assert namespace(mean) is backend_ns
