"""Integration tests for persistent sampling through the SMC runner."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from sequince import ParticleState, SequentialMonteCarlo, TemperedPosterior
from sequince.steps import (
    MiniPCNMutation,
    PersistentAdaptiveTemperedWeighting,
    PersistParticles,
    StepContext,
    SystematicResampling,
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
    persist = PersistParticles()
    smc = SequentialMonteCarlo(
        steps=[
            persist,
            PersistentAdaptiveTemperedWeighting(
                target,
                target_ess=target_ess,
                target_total_ess=target_total_ess,
            ),
            SystematicResampling(
                always_resample=True,
                n_samples=n_particles,
            ),
            MiniPCNMutation(target, n_steps=2, step_fn="tpcn"),
        ]
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
