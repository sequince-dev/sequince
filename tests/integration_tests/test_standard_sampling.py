"""Integration tests for standard sampling through the SMC runner."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from sequince import ParticleState, make_standard_bayesian_smc
from sequince.steps.base import SMCStep


@pytest.mark.parametrize("adaptive_mutation", [None, {"max_steps": 8}])
def test_standard_sampler_reaches_gaussian_posterior(
    adaptive_mutation,
    gaussian_log_prior: Callable[..., np.ndarray],
    gaussian_log_likelihood: Callable[..., np.ndarray],
) -> None:
    n_particles = 256
    sampler, _, metadata = make_standard_bayesian_smc(
        gaussian_log_prior,
        gaussian_log_likelihood,
        mutation_n_steps=3,
        adaptive_mutation=adaptive_mutation,
        logging_frequency=0,
    )
    initial_particles = np.random.default_rng(1234).normal(
        loc=0.0,
        scale=2.0,
        size=(n_particles, 1),
    )
    initial_state = ParticleState(
        initial_particles,
        metadata={"beta": 0.0},
    )

    result = sampler.run(initial_state, rng=5678, metadata=metadata)

    assert result.context.stop_requested
    assert result.context.metadata["beta"] == pytest.approx(1.0)
    np.testing.assert_allclose(result.final_state.weights.sum(), 1.0)

    final_particles = np.asarray(result.final_state.particles)[:, 0]
    final_weights = np.asarray(result.final_state.weights)
    final_mean = np.average(final_particles, weights=final_weights)
    final_variance = np.average(
        (final_particles - final_mean) ** 2,
        weights=final_weights,
    )
    assert final_mean == pytest.approx(1.2, abs=0.25)
    assert final_variance == pytest.approx(0.8, abs=0.25)

    analytic_log_evidence = -0.5 * (np.log(2.0 * np.pi * 5.0) + 1.5**2 / 5.0)
    assert float(result.final_state.log_evidence) == pytest.approx(
        analytic_log_evidence,
        abs=0.2,
    )


@pytest.mark.parametrize(
    ("resample_at_end", "expected_executed"),
    [
        (True, [False, True]),
        (False, [False, False]),
    ],
)
def test_standard_sampler_resamples_and_mutates_at_end(
    standard_normal_log_prob: Callable[..., Any],
    unit_log_likelihood: Callable[..., np.ndarray],
    rng,
    resample_at_end: bool,
    expected_executed: list[bool],
) -> None:
    mutation_calls: list[int] = []

    class RecordingMutation(SMCStep):
        def apply(self, state, context, rng):
            del rng
            mutation_calls.append(context.t)
            return state, context, None

    sampler, _, metadata = make_standard_bayesian_smc(
        standard_normal_log_prob,
        unit_log_likelihood,
        weighting_criterion="cess",
        max_delta=0.5,
        resampling_ess_threshold=None,
        resample_at_end=resample_at_end,
        mutation=RecordingMutation(),
        logging_frequency=0,
    )
    initial_state = ParticleState(
        particles=np.linspace(-1.0, 1.0, 8)[:, np.newaxis],
    )

    result = sampler.run(initial_state, rng=rng, metadata=metadata)

    resampled = [bool(output[1].resampled) for output in result.outputs]
    mutated = [output[2].executed for output in result.outputs]
    assert resampled == expected_executed
    assert mutated == expected_executed
    assert mutation_calls == ([1] if resample_at_end else [])
    assert result.context.stop_requested
    assert result.context.metadata["beta"] == pytest.approx(1.0)
