from __future__ import annotations

import numpy as np
import pytest

from sequince.state import ParticleState
from sequince.steps.base import StepContext
from sequince.steps.mutation import EmceeMutation
from sequince.steps.resampling import (
    MultinomialResampling,
    SystematicResampling,
)
from sequince.steps.weighting import (
    AdaptiveTemperedWeighting,
    FixedTemperedWeighting,
    LogLikelihoodWeighting,
    TemperedWeighting,
)


def test_step_context_request_stop(step_context: StepContext) -> None:
    assert not step_context.stop_requested
    step_context.request_stop()
    assert step_context.stop_requested


def test_log_likelihood_weighting_updates_log_weights(
    particle_state: ParticleState, context_factory, rng
) -> None:
    def log_like(
        particles: np.ndarray,
        context: StepContext,
    ) -> np.ndarray:
        return np.sum(particles, axis=1)

    context = context_factory(t=3, observation=5)
    step = LogLikelihoodWeighting(log_like)
    updated = step(particle_state, context, rng)

    increments = np.sum(particle_state.particles, axis=1)
    expected = particle_state.log_weights + increments
    expected -= np.log(np.sum(np.exp(expected)))
    assert np.allclose(updated.log_weights, expected)


def test_tempered_weighting_scales_log_likelihood(
    particle_state: ParticleState, context_factory, rng
) -> None:
    def log_like(particles: np.ndarray, context: StepContext) -> np.ndarray:
        return np.ones(particles.shape[0])

    def schedule(t: int, context: StepContext) -> float:
        context.static.setdefault("temperatures", []).append(t)
        return 0.5 + 0.1 * t

    context = context_factory(t=2, static={})
    step = TemperedWeighting(log_like, schedule, normalize_after=False)
    updated = step(particle_state, context, rng)

    expected_increment = (0.5 + 0.1 * 2) * np.ones_like(
        particle_state.log_weights
    )
    assert np.allclose(
        updated.log_weights, particle_state.log_weights + expected_increment
    )
    assert context.static["temperatures"] == [2]


def test_fixed_tempered_weighting_updates_metadata_and_context(
    particle_state: ParticleState, context_factory, rng
) -> None:
    def log_like(particles: np.ndarray, context: StepContext) -> np.ndarray:
        return np.ones(particles.shape[0])

    context = context_factory(static={"temperature": 0.0})
    step = FixedTemperedWeighting(
        log_like, temperature_increment=0.25, normalize_after=False
    )
    updated = step(particle_state, context, rng)

    assert updated.metadata["temperature"] == pytest.approx(0.25)
    assert context.static["temperature"] == pytest.approx(0.25)
    assert context.static["temperature_delta"] == pytest.approx(0.25)


def test_adaptive_tempered_weighting_reaches_max_temperature(
    particle_state: ParticleState, context_factory, rng
) -> None:
    def log_like(particles: np.ndarray, context: StepContext) -> np.ndarray:
        return np.ones(particles.shape[0])

    step = AdaptiveTemperedWeighting(
        log_like,
        target_ess_ratio=0.8,
        max_temperature=0.6,
        min_delta=0.05,
        max_bisection_iter=10,
    )

    state = particle_state.with_updates(metadata={"temperature": 0.0})
    shared_static: dict[str, float] = {}

    stop_step = None
    for t in range(20):
        context = context_factory(t=t, static=shared_static)
        state = step(state, context, rng)
        if context.stop_requested:
            stop_step = t
            break

    print(state, shared_static)
    assert stop_step is not None, (
        "Adaptive tempering did not request stop within 20 iterations."
    )
    assert state.metadata["temperature"] == pytest.approx(0.6, abs=1e-6)
    assert shared_static["temperature"] == pytest.approx(
        state.metadata["temperature"]
    )
    assert "temperature_delta" in shared_static


def test_adaptive_tempered_weighting_no_op_when_max_reached(
    particle_state: ParticleState, context_factory, rng
) -> None:
    def log_like(particles: np.ndarray, context: StepContext) -> np.ndarray:
        return np.ones(particles.shape[0])

    step = AdaptiveTemperedWeighting(log_like, max_temperature=1.0)
    state = particle_state.with_updates(metadata={"temperature": 1.0})
    context = context_factory(static={})

    result = step(state, context, rng)

    assert result is state
    assert context.stop_requested
    assert context.static["temperature"] == 1.0
    assert context.static["temperature_delta"] == 0.0


def test_systematic_resampling_skips_when_not_required(
    particle_state: ParticleState, context_factory, rng
) -> None:
    normalized, _ = particle_state.normalize()
    context = context_factory()
    step = SystematicResampling(ess_threshold=0.2, always_resample=False)
    result = step(normalized, context, rng)
    assert result is normalized  # no resampling expected


def test_multinomial_resampling_resamples_particles(
    particle_state: ParticleState, context_factory, rng
) -> None:
    context = context_factory()
    step = MultinomialResampling(always_resample=True)
    resampled = step(particle_state, context, rng)

    assert resampled.particles.shape == particle_state.particles.shape
    assert np.allclose(
        resampled.log_weights,
        np.full_like(
            resampled.log_weights, -np.log(resampled.log_weights.size)
        ),
    )
    assert "ancestors" in resampled.metadata


def test_emcee_mutation_runs_single_step(
    particle_state: ParticleState, context_factory, rng
) -> None:
    pytest.importorskip("emcee")

    def log_prob(position: np.ndarray, context: StepContext) -> float:
        return float(-0.5 * np.sum(position**2))

    context = context_factory(static={"temperature": 1.0})
    step = EmceeMutation(log_prob, n_steps=1)
    mutated = step(particle_state, context, rng)

    assert mutated.particles.shape == particle_state.particles.shape
    assert "emcee_acceptance" in mutated.metadata
    assert mutated.metadata["emcee_steps"] == 1
