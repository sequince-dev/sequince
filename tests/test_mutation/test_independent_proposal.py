from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from array_api_compat import device as get_device
from orng import RandomGenerator

from sequince import ParticleState
from sequince.array_api import namespace, to_numpy
from sequince.steps import (
    IndependentProposalKernel,
    MetropolisHastingsMutation,
    StepContext,
)
from sequince.steps.mutation.kernels import (
    evaluate_log_transition_ratio,
    get_mutation_kernel,
)
from sequince.typing import Array
from sequince.utils.random import resolve_rng


class NormalProposal:
    def sample(
        self,
        n_samples: int,
        context: StepContext | None = None,
        *,
        rng: RandomGenerator | int | None = None,
        xp: Any | None = None,
    ) -> Array:
        return resolve_rng(rng).normal(size=(n_samples, 2)) * 2

    def log_prob(
        self, particles: Array, context: StepContext | None = None
    ) -> Array:
        xp = namespace(particles)
        return -xp.sum(particles**2, axis=-1) / 8

    def sample_and_log_prob(
        self,
        n_samples: int,
        context: StepContext | None = None,
        *,
        rng: RandomGenerator | int | None = None,
        xp: Any | None = None,
    ) -> tuple[Array, Array]:
        values = self.sample(n_samples, rng=rng, xp=xp)
        return values, self.log_prob(values)


def test_independence_backend(backend, backend_factory) -> None:
    case = backend_factory(backend)
    particles = case.particles
    kernel = IndependentProposalKernel(NormalProposal())
    sample = kernel.sample(particles, case.rng(5))
    ratio = evaluate_log_transition_ratio(kernel, particles, sample)
    assert sample.particles.shape == particles.shape
    assert sample.particles.dtype == particles.dtype
    assert get_device(sample.particles) == get_device(particles)
    assert case.matches_namespace(sample.particles)
    np.testing.assert_allclose(
        to_numpy(ratio),
        to_numpy(
            kernel.proposal.log_prob(particles)
            - kernel.proposal.log_prob(sample.particles)
        ),
        rtol=1e-6,
        atol=1e-6,
    )
    with pytest.raises(ValueError, match="shape"):
        kernel.log_transition_ratio(particles, sample.particles[:1])


def test_support_and_validation() -> None:
    class Bounded(NormalProposal):
        def log_prob(self, particles, context=None):
            return np.where(particles[:, 0] >= 0, -particles[:, 0], -np.inf)

    kernel = IndependentProposalKernel(Bounded())
    ratio = kernel.log_transition_ratio(
        np.array([[-1.0, 0], [1, 0]]), np.array([[1.0, 0], [-1, 0]])
    )
    np.testing.assert_array_equal(ratio, [-np.inf, np.inf])
    with pytest.raises(ValueError, match="shape"):
        kernel.sample(np.zeros((3, 1)), RandomGenerator(seed=5))

    class BadDensity(NormalProposal):
        def log_prob(self, particles, context=None):
            return np.zeros((len(particles), 1))

    with pytest.raises(ValueError, match="densities"):
        IndependentProposalKernel(BadDensity()).sample(
            np.zeros((3, 2)), RandomGenerator(seed=5)
        )


@pytest.mark.parametrize("shape", [(5,), (1, 2), (5, 2, 3)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_independence_particle_shapes(shape, dtype) -> None:
    class ShapedProposal(NormalProposal):
        def sample(self, n_samples, context=None, *, rng=None, xp=None):
            return rng.normal(size=(n_samples, *shape[1:]))

        def log_prob(self, particles, context=None):
            return (
                -np.sum(
                    np.reshape(particles, (len(particles), -1)) ** 2, axis=1
                )
                / 2
            )

    particles = np.ones(shape, dtype=dtype)
    kernel = IndependentProposalKernel(ShapedProposal())
    sample = kernel.sample(particles, RandomGenerator(seed=3))
    assert sample.particles.shape == shape
    assert sample.particles.dtype == dtype
    ratio = evaluate_log_transition_ratio(kernel, particles, sample)
    assert ratio.shape == (shape[0],)
    assert ratio.dtype == dtype


def test_cached_density() -> None:
    class CountedProposal(NormalProposal):
        def __init__(self) -> None:
            self.calls = []

        def log_prob(self, particles, context=None):
            self.calls.append(particles)
            return super().log_prob(particles, context)

    proposal = CountedProposal()
    independent = IndependentProposalKernel(proposal)
    particles = np.ones((10, 2))
    rng = RandomGenerator(seed=22)
    sample = independent.sample(particles, rng)
    evaluate_log_transition_ratio(independent, particles, sample)
    assert len(proposal.calls) == 2  # Sampled density plus current density.
    assert proposal.calls[-1] is particles


def test_known_target(context_factory) -> None:
    rng = RandomGenerator(seed=44)
    state = ParticleState(rng.normal(size=(1500, 2)) * 4)
    step = MetropolisHastingsMutation(
        lambda x: -np.sum(x * x, axis=-1) / 2,
        IndependentProposalKernel(NormalProposal()),
        n_steps=100,
    )
    state, _, _ = step(state, context_factory(), rng)
    np.testing.assert_allclose(np.mean(state.particles, axis=0), 0, atol=0.10)
    np.testing.assert_allclose(np.var(state.particles, axis=0), 1, atol=0.12)


def test_factory() -> None:
    proposal = NormalProposal()
    kernel = get_mutation_kernel("independent-proposal", proposal=proposal)
    assert isinstance(kernel, IndependentProposalKernel)
    assert kernel.proposal is proposal
