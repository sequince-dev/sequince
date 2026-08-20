"""Integration tests for persistent sampling through the SMC runner."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from sequince import (
    ParticleState,
    TemperedPosterior,
    make_persistent_bayesian_smc,
)
from sequince.steps import (
    PersistParticles,
    StepContext,
)


def test_make_persistent_bayesian_smc_reaches_gaussian_posterior(
    gaussian_log_prior: Callable[..., np.ndarray],
    gaussian_log_likelihood: Callable[..., np.ndarray],
) -> None:
    n_particles = 32
    target_ess = 2 * n_particles
    target = TemperedPosterior(
        log_prior=gaussian_log_prior,
        log_likelihood=gaussian_log_likelihood,
    )
    smc = make_persistent_bayesian_smc(
        target,
        n_particles=n_particles,
        target_ess=target_ess,
        mutation_n_steps=3,
        mutation_kwargs={"step_fn": "tpcn"},
    )
    initial_particles = np.random.default_rng(1234).normal(
        loc=0.0,
        scale=2.0,
        size=(n_particles, 1),
    )
    initial_context = StepContext(t=0, metadata={"temperature": 0.0})
    initial_state = ParticleState(
        initial_particles,
        evaluation=target.prepare_evaluation(
            initial_particles,
            initial_context,
        ),
    )

    result = smc.run(initial_state, rng=5678)

    temperatures = [
        float(outputs[1]["temperature"]) for outputs in result.outputs
    ]
    temperature_deltas = [
        float(outputs[1]["temperature_delta"]) for outputs in result.outputs
    ]
    assert temperature_deltas[:2] == pytest.approx([0.0, 0.0])
    assert temperatures[-1] == pytest.approx(1.0)
    assert result.context.stop_requested
    assert not result.context.stop_immediately

    persist = smc._steps[0]  # noqa: SLF001
    assert isinstance(persist, PersistParticles)
    assert persist.persistent_state is not None
    assert persist.persistent_state.n_batches == len(result)
    assert persist.persistent_state.n_particles == len(result) * n_particles

    final_particles = np.asarray(result.final_state.particles)[:, 0]
    assert np.mean(final_particles) == pytest.approx(1.2, abs=0.35)
    assert np.var(final_particles) == pytest.approx(0.8, abs=0.35)
    analytic_log_evidence = -0.5 * (np.log(2.0 * np.pi * 5.0) + 1.5**2 / 5.0)
    assert float(result.final_state.log_evidence) == pytest.approx(
        analytic_log_evidence,
        abs=0.25,
    )


def test_runner_accumulates_posterior_batches_until_target_total_ess(
    gaussian_log_prior: Callable[..., np.ndarray],
    gaussian_log_likelihood: Callable[..., np.ndarray],
) -> None:
    n_particles = 32
    target_ess = 2 * n_particles
    target_total_ess = 5 * n_particles
    target = TemperedPosterior(
        log_prior=gaussian_log_prior,
        log_likelihood=gaussian_log_likelihood,
    )
    smc = make_persistent_bayesian_smc(
        target,
        n_particles=n_particles,
        target_ess=target_ess,
        target_total_ess=target_total_ess,
        mutation_n_steps=2,
        mutation_kwargs={"step_fn": "tpcn"},
    )
    persist = smc._steps[0]  # noqa: SLF001
    assert isinstance(persist, PersistParticles)
    initial_particles = np.random.default_rng(1234).normal(
        loc=0.0,
        scale=2.0,
        size=(n_particles, 1),
    )
    initial_context = StepContext(t=0, metadata={"temperature": 0.0})
    initial_state = ParticleState(
        initial_particles,
        evaluation=target.prepare_evaluation(
            initial_particles,
            initial_context,
        ),
    )

    result = smc.run(
        initial_state,
        rng=5678,
        metadata={"temperature": 0.0},
    )

    temperature_deltas = [
        float(iteration_output[1]["temperature_delta"])
        for iteration_output in result.outputs
    ]
    temperatures = [
        float(iteration_output[1]["temperature"])
        for iteration_output in result.outputs
    ]
    persistent_ess = [
        float(iteration_output[1]["effective_sample_size"])
        for iteration_output in result.outputs
    ]

    assert temperature_deltas[:2] == pytest.approx([0.0, 0.0])
    assert temperatures[:2] == pytest.approx([0.0, 0.0])
    assert persistent_ess[:2] == pytest.approx([n_particles, target_ess])
    assert any(delta > 0.0 for delta in temperature_deltas[2:])
    assert temperatures[-1] == pytest.approx(1.0)
    assert temperature_deltas[-1] == pytest.approx(0.0)
    assert temperatures.count(1.0) >= 2
    assert persistent_ess[-1] >= target_total_ess
    assert result.context.stop_requested
    assert not result.context.stop_immediately
    assert len(result.outputs[-1]) == 4
    assert result.outputs[-1][2]["resampled"]
    assert result.outputs[-1][3] is not None
    assert persist.persistent_state is not None
    assert persist.persistent_state.n_batches == len(result)
    assert persist.persistent_state.n_particles == len(result) * n_particles
    assert persist.persistent_state.temperatures[-1] == pytest.approx(1.0)
