from __future__ import annotations

import numpy as np
import pytest

from sequince import ParticleState, TemperedPosterior
from sequince.prebuilt_samplers import make_standard_bayesian_smc
from sequince.steps import (
    AdaptiveTemperedWeighting,
    LoggingStep,
    MiniPCNMutation,
    SystematicResampling,
)
from sequince.steps.base import StepContext


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


def test_make_standard_bayesian_smc_accepts_evolving_target(rng) -> None:
    schedule_calls: list[float] = []

    def target_schedule(temperature: float) -> float:
        schedule_calls.append(temperature)
        return 0.8

    smc, _, metadata = make_standard_bayesian_smc(
        _quadratic_log_prior,
        _zero_log_likelihood,
        target_ess_ratio=target_schedule,
        max_temperature=0.5,
        mutation="none",
    )
    particles = np.array([[0.0], [1.0], [-1.0], [2.0]], dtype=float)
    initial_state = ParticleState(
        particles=particles,
        log_weights=np.full(particles.shape[0], -np.log(particles.shape[0])),
        metadata={"temperature": 0.0},
    )

    result = smc.run(initial_state, rng=rng, metadata=metadata)

    assert schedule_calls
    assert schedule_calls[0] == pytest.approx(0.0)
    assert result.final_state.metadata["target_ess_ratio"] == pytest.approx(
        0.8
    )


def test_make_standard_bayesian_smc_validates_resampling_name() -> None:
    with pytest.raises(ValueError):
        make_standard_bayesian_smc(
            _quadratic_log_prior,
            _zero_log_likelihood,
            resampling="invalid",  # type: ignore[arg-type]
        )


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
