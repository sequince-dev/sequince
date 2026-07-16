from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from sequince import ParticleState, TemperedPosterior
from sequince.prebuilt_samplers import make_standard_bayesian_smc
from sequince.steps import (
    AdaptiveTemperedWeighting,
    LoggingStep,
    MetropolisHastingsMutation,
    MiniPCNMutation,
    SystematicResampling,
)
from sequince.steps.base import SMCStep, StepContext
from sequince.steps.mutation import (
    GaussianRandomWalk,
    MixtureMutationKernel,
)

if TYPE_CHECKING:
    from orng import ArrayRNG


def _quadratic_log_prior(particles: np.ndarray, context=None) -> np.ndarray:
    values = np.asarray(particles, dtype=float)
    if values.ndim == 1:
        values = values[:, np.newaxis]
    return -0.5 * np.sum(values**2, axis=1)


def _zero_log_likelihood(particles: np.ndarray, context=None) -> np.ndarray:
    values = np.asarray(particles, dtype=float)
    if values.ndim == 1:
        values = values[:, np.newaxis]
    return np.zeros(values.shape[0], dtype=float)


def test_make_standard_bayesian_smc_builds_default_steps() -> None:
    smc, posterior, metadata = make_standard_bayesian_smc(
        _quadratic_log_prior,
        _zero_log_likelihood,
    )

    assert isinstance(posterior, TemperedPosterior)
    assert metadata == {"max_temperature": 1.0}

    steps = smc._steps  # noqa: SLF001 - inspecting configured steps for testing.
    assert len(steps) == 4
    assert isinstance(steps[0], AdaptiveTemperedWeighting)
    assert isinstance(steps[1], SystematicResampling)
    assert isinstance(steps[2], MiniPCNMutation)
    assert isinstance(steps[3], LoggingStep)


def test_make_standard_bayesian_smc_reaches_max_temperature(rng) -> None:
    max_temperature = 0.5
    smc, _, metadata = make_standard_bayesian_smc(
        _quadratic_log_prior,
        _zero_log_likelihood,
        max_temperature=max_temperature,
        mutation_n_steps=1,
    )

    particles = np.array([[0.0], [1.0], [-1.0], [2.0]], dtype=float)
    n = particles.shape[0]
    initial_state = ParticleState(
        particles=particles,
        log_weights=np.full(n, -np.log(n)),
        metadata={"temperature": 0.0},
    )

    results = smc.run(initial_state, rng=rng, metadata=metadata)

    assert results.states[-1].metadata["temperature"] == pytest.approx(
        max_temperature
    )
    assert metadata["temperature"] == pytest.approx(max_temperature)


def test_make_standard_bayesian_smc_validates_resampling_name() -> None:
    with pytest.raises(ValueError):
        make_standard_bayesian_smc(
            _quadratic_log_prior,
            _zero_log_likelihood,
            resampling="invalid",  # type: ignore[arg-type]
        )


def test_make_standard_bayesian_smc_supports_minipcn_mutation() -> None:
    smc, _, _ = make_standard_bayesian_smc(
        _quadratic_log_prior,
        _zero_log_likelihood,
        mutation="minipcn",
    )

    steps = smc._steps  # noqa: SLF001
    assert isinstance(steps[2], MiniPCNMutation)


def test_make_standard_bayesian_smc_supports_gaussian_random_walk() -> None:
    smc, _, _ = make_standard_bayesian_smc(
        _quadratic_log_prior,
        _zero_log_likelihood,
        mutation="metropolis-hastings",
        mutation_n_steps=3,
        mutation_kwargs={
            "kernel": "gaussian-random-walk",
            "kernel_kwargs": {"scale": 0.25},
            "store_chains": True,
        },
    )

    steps = smc._steps  # noqa: SLF001
    mutation = steps[2]
    assert isinstance(mutation, MetropolisHastingsMutation)
    assert isinstance(mutation._kernel, GaussianRandomWalk)  # noqa: SLF001
    assert mutation._kernel.scale == 0.25  # noqa: SLF001
    assert mutation._n_steps == 3  # noqa: SLF001


def test_metropolis_hastings_requires_kernel() -> None:
    with pytest.raises(ValueError, match="requires a 'kernel'"):
        make_standard_bayesian_smc(
            _quadratic_log_prior,
            _zero_log_likelihood,
            mutation="metropolis-hastings",
        )


def test_make_standard_bayesian_smc_supports_mixture_kernel() -> None:
    smc, _, _ = make_standard_bayesian_smc(
        _quadratic_log_prior,
        _zero_log_likelihood,
        mutation="metropolis-hastings",
        mutation_kwargs={
            "kernel": "mixture",
            "kernel_kwargs": {
                "kernels": [
                    {
                        "kernel": "gaussian-random-walk",
                        "kernel_kwargs": {"scale": 0.5},
                    },
                    {
                        "kernel": "gaussian-random-walk",
                        "kernel_kwargs": {"scale": 0.2},
                    },
                ],
                "weights": [0.8, 0.2],
            },
        },
    )

    mutation = smc._steps[2]  # noqa: SLF001
    assert isinstance(mutation, MetropolisHastingsMutation)
    mixture = mutation._kernel  # noqa: SLF001
    assert isinstance(mixture, MixtureMutationKernel)
    assert isinstance(mixture.kernels[0], GaussianRandomWalk)
    assert isinstance(mixture.kernels[1], GaussianRandomWalk)


def test_make_standard_bayesian_smc_accepts_proposal_distribution() -> None:
    class ZeroProposal:
        def log_prob(self, particles: np.ndarray, context=None) -> np.ndarray:
            values = np.asarray(particles, dtype=float)
            if values.ndim == 1:
                values = values[:, np.newaxis]
            return np.zeros(values.shape[0], dtype=float)

        def sample(self, n_samples: int, context=None) -> np.ndarray:
            return np.zeros((n_samples, 1), dtype=float)

        def sample_and_log_prob(
            self, n_samples: int, context=None
        ) -> tuple[np.ndarray, np.ndarray]:
            samples = self.sample(n_samples, context=context)
            return samples, self.log_prob(samples, context=context)

    _, posterior, _ = make_standard_bayesian_smc(
        _quadratic_log_prior,
        _zero_log_likelihood,
        log_proposal=ZeroProposal(),
    )
    particles = np.array([[0.0], [1.0]], dtype=float)
    values = posterior.log_increment(
        particles,
        StepContext(t=0, metadata={"temperature": 0.0}),
    )

    assert np.allclose(values, _quadratic_log_prior(particles))


def test_make_standard_bayesian_smc_can_disable_mutation() -> None:
    smc, _, _ = make_standard_bayesian_smc(
        _quadratic_log_prior,
        _zero_log_likelihood,
        mutation="none",
    )

    steps = smc._steps  # noqa: SLF001
    assert len(steps) == 3
    assert isinstance(steps[0], AdaptiveTemperedWeighting)
    assert isinstance(steps[1], SystematicResampling)
    assert isinstance(steps[2], LoggingStep)


class IdentityMutation(SMCStep):
    def apply(self, state, context: StepContext, rng: "ArrayRNG"):
        return state


def test_make_standard_bayesian_smc_accepts_custom_mutation_step() -> None:
    custom = IdentityMutation()
    smc, _, _ = make_standard_bayesian_smc(
        _quadratic_log_prior,
        _zero_log_likelihood,
        mutation=custom,
    )

    steps = smc._steps  # noqa: SLF001
    assert steps[-2] is custom


def test_make_standard_bayesian_smc_validates_mutation_name() -> None:
    with pytest.raises(ValueError):
        make_standard_bayesian_smc(
            _quadratic_log_prior,
            _zero_log_likelihood,
            mutation="unknown",  # type: ignore[arg-type]
        )
