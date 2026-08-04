from __future__ import annotations

from typing import Any

import pytest

from sequince._array_api import namespace
from sequince.posterior import TemperedPosterior
from sequince.prebuilt_samplers import make_standard_bayesian_smc
from sequince.state import ParticleState
from sequince.steps.base import StepContext


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax", "cupy"])
def test_sequince_runs_with_array_backends(
    backend: str, backend_factory
) -> None:
    case = backend_factory(backend)
    xp = case.xp
    particles = case.particles
    backend_ns = namespace(particles)

    smc, _, _ = make_standard_bayesian_smc(
        log_prior=lambda x, ctx: xp.zeros(x.shape[0], dtype=x.dtype),
        log_likelihood=lambda x, ctx: -0.5 * xp.sum(x**2, axis=1),
        mutation="none",
    )

    results = smc.run(
        case.state(),
        num_steps=1,
        rng=case.rng(1234),
    )

    assert len(results) == 1
    final_state = results.final_state

    assert final_state.particles.shape == particles.shape
    assert namespace(final_state.particles) is backend_ns

    samples = results.posterior_samples(
        n_samples=particles.shape[0],
        rng=case.rng(4321),
    )
    assert samples.shape[0] == particles.shape[0]
    assert namespace(samples) is backend_ns

    mean = results.weighted_mean()
    assert namespace(mean) is backend_ns


def test_torch_gpu_particles_remain_on_device() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    device = torch.device("cuda", 0)
    dtype = torch.float32
    particles = torch.arange(8, device=device, dtype=dtype).reshape(4, 2)
    log_weights = torch.full((4,), float("-inf"), dtype=dtype, device=device)
    state = ParticleState(particles=particles, log_weights=log_weights)
    xp = namespace(particles)

    normalized, _ = state.normalize()
    assert normalized.log_weights.device == device
    weights = normalized.weights
    assert weights.device == device

    cpu_indices = torch.arange(4, dtype=torch.long)
    resampled = state.resample(cpu_indices)
    assert resampled.particles.device == device
    assert resampled.log_weights.device == device

    context = StepContext(t=0)
    posterior = TemperedPosterior(
        log_prior=lambda x, ctx: xp.sum(x, axis=1),
        log_likelihood=lambda x, ctx: xp.sum(x, axis=1),
    )
    evaluation = posterior.evaluate(state.particles, context)
    log_inc: Any = posterior.log_increment(
        state.particles, context, evaluation=evaluation
    )
    assert log_inc.device == device
