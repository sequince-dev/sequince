"""Tests for persistent-sampling importance weights."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from scipy.special import logsumexp

from sequince.array_api import asarray, namespace, to_numpy
from sequince.posterior import TemperedPosterior
from sequince.smc import SequentialMonteCarlo
from sequince.state import ParticleState, PersistentParticleState
from sequince.steps.context import StepContext
from sequince.steps.persistent import (
    PersistentAdaptiveTemperedWeighting,
    PersistentTemperedWeighting,
    PersistParticles,
    persistent_log_mixture_denominator,
    persistent_log_weights,
)


def _reference_denominator(
    log_increment: np.ndarray,
    betas: np.ndarray,
    log_normalizing_constants: np.ndarray,
) -> np.ndarray:
    terms = (
        log_increment[:, None] * betas[None, :]
        - log_normalizing_constants[None, :]
    )
    return logsumexp(terms, axis=1) - np.log(betas.size)


def test_single_initial_target() -> None:
    log_increment = np.array([-2.0, 0.5, 3.0])
    beta = 0.4

    denominator = persistent_log_mixture_denominator(
        log_increment,
        np.array([0.0]),
        np.array([0.0]),
    )
    log_weights = persistent_log_weights(
        log_increment,
        np.array([0.0]),
        np.array([0.0]),
        beta,
    )

    np.testing.assert_allclose(denominator, 0.0)
    np.testing.assert_allclose(log_weights, beta * log_increment)


@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
def test_zero_support_with_initial_target(beta: float) -> None:
    increments = np.array([-np.inf, 0.0])
    betas = np.array([0.0, 1.0])
    normalizers = np.zeros(2)
    with np.errstate(invalid="raise"):
        denominator = persistent_log_mixture_denominator(
            increments, betas, normalizers
        )
        weights = persistent_log_weights(increments, betas, normalizers, beta)
    np.testing.assert_allclose(denominator, [-np.log(2), 0.0])
    np.testing.assert_allclose(
        weights, [np.log(2) if beta == 0 else -np.inf, 0.0]
    )


@pytest.mark.parametrize(
    ("betas", "log_normalizing_constants"),
    [
        ([0.0, 0.3, 0.8], [0.0, 0.0, 0.0]),
        ([0.0, 0.3, 0.8], [0.0, 1.7, -0.4]),
        ([0.0, 0.0, 0.5], [0.0, 0.2, 0.9]),
    ],
    ids=["multiple-targets", "nonzero-evidence", "repeated-betas"],
)
def test_denominator_matches_direct_reference(
    betas: list[float],
    log_normalizing_constants: list[float],
) -> None:
    log_increment = np.array([-1.5, 0.25, 2.0])
    betas_array = np.asarray(betas)
    log_normalizing_constants_array = np.asarray(log_normalizing_constants)

    actual = persistent_log_mixture_denominator(
        log_increment,
        betas_array,
        log_normalizing_constants_array,
    )
    expected = _reference_denominator(
        log_increment,
        betas_array,
        log_normalizing_constants_array,
    )

    np.testing.assert_allclose(actual, expected)


def test_log_weights_match_direct_reference() -> None:
    log_increment = np.array([-1.5, 0.25, 2.0])
    betas = np.array([0.0, 0.2, 0.9])
    log_normalizing_constants = np.array([0.0, 0.6, 1.4])
    beta = 0.7
    expected_denominator = _reference_denominator(
        log_increment,
        betas,
        log_normalizing_constants,
    )

    actual = persistent_log_weights(
        log_increment,
        betas,
        log_normalizing_constants,
        beta,
    )

    np.testing.assert_allclose(
        actual,
        beta * log_increment - expected_denominator,
    )


def test_large_magnitude_values_are_stable() -> None:
    log_increment = np.array([1000.0, -1000.0, 500.0])
    betas = np.array([0.0, 0.5, 1.0])
    log_normalizing_constants = np.array([0.0, 25.0, 100.0])

    actual = persistent_log_mixture_denominator(
        log_increment,
        betas,
        log_normalizing_constants,
    )
    expected = _reference_denominator(
        log_increment,
        betas,
        log_normalizing_constants,
    )

    assert np.all(np.isfinite(actual))
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax", "cupy"])
def test_supported_backends_preserve_namespace_dtype_and_shape(
    backend: str,
    backend_factory,
) -> None:
    case = backend_factory(backend)
    xp = case.xp
    log_increment = case.particles[:, 0]
    betas = asarray([0.0, 0.0, 0.5], xp=xp, like=log_increment)
    log_normalizing_constants = asarray(
        [0.0, 0.2, 0.9], xp=xp, like=log_increment
    )

    denominator = persistent_log_mixture_denominator(
        log_increment,
        betas,
        log_normalizing_constants,
    )
    log_weights = persistent_log_weights(
        log_increment,
        betas,
        log_normalizing_constants,
        0.75,
    )

    expected = _reference_denominator(
        to_numpy(log_increment),
        to_numpy(betas),
        to_numpy(log_normalizing_constants),
    )
    assert denominator.shape == log_increment.shape
    assert log_weights.shape == log_increment.shape
    assert namespace(denominator) is xp
    assert namespace(log_weights) is xp
    assert denominator.dtype == log_increment.dtype
    assert log_weights.dtype == log_increment.dtype
    np.testing.assert_allclose(to_numpy(denominator), expected, rtol=1e-5)


@pytest.mark.parametrize(
    (
        "log_increment",
        "betas",
        "log_normalizing_constants",
        "match",
    ),
    [
        ([1.0], [], [], "must not be empty"),
        ([1.0], [0.0, 0.5], [0.0], "same length"),
        ([[1.0]], [0.0], [0.0], "log_increment"),
        ([1.0], [[0.0]], [0.0], "betas"),
        ([1.0], [0.0], [[0.0]], "log_normalizing_constants"),
    ],
)
def test_invalid_histories_raise_value_error(
    log_increment,
    betas,
    log_normalizing_constants,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        persistent_log_mixture_denominator(
            log_increment,
            betas,
            log_normalizing_constants,
        )


def test_candidate_beta_must_be_scalar() -> None:
    with pytest.raises(ValueError, match="beta must be scalar"):
        persistent_log_weights([1.0], [0.0], [0.0], [0.5])


def _linear_target():
    calls = {"initial": 0, "increment": 0}

    def log_prior(particles):
        calls["initial"] += 1
        xp = namespace(particles)
        return xp.zeros((particles.shape[0],), dtype=particles.dtype)

    def log_likelihood(particles):
        calls["increment"] += 1
        return particles[:, 0]

    return TemperedPosterior(log_prior, log_likelihood), calls


def _evaluated_state(
    target,
    particles,
    *,
    log_normalizing_constant=0.0,
    beta=0.0,
):
    context = StepContext(t=0, metadata={"beta": beta})
    evaluation = target.prepare_evaluation(particles, context)
    return (
        ParticleState(
            particles,
            log_normalizing_constant=log_normalizing_constant,
            evaluation=evaluation,
            metadata={"beta": beta},
        ),
        context,
    )


def test_persist_particles_creates_and_appends_completed_batches(rng) -> None:
    target, _ = _linear_target()
    first, first_context = _evaluated_state(
        target, np.array([[-1.0], [1.0]]), log_normalizing_constant=0.0
    )
    step = PersistParticles()

    one_batch, returned_context, output = step.apply(first, first_context, rng)

    assert type(one_batch) is PersistentParticleState
    assert one_batch.n_batches == 1
    np.testing.assert_allclose(one_batch.betas, [0.0])
    np.testing.assert_allclose(one_batch.log_evidences, [0.0])
    np.testing.assert_array_equal(one_batch.particles, first.particles)
    assert one_batch.evaluation is first.evaluation
    assert returned_context is first_context
    assert output is None

    second, second_context = _evaluated_state(
        target,
        np.array([[2.0], [3.0]]),
        log_normalizing_constant=0.6,
        beta=0.4,
    )
    two_batches, _, _ = step.apply(second, second_context, rng)

    assert two_batches.n_batches == 2
    np.testing.assert_array_equal(
        two_batches.particles[:, 0], [-1.0, 1.0, 2.0, 3.0]
    )
    np.testing.assert_allclose(two_batches.betas, [0.0, 0.4])
    np.testing.assert_allclose(two_batches.log_evidences, [0.0, 0.6])
    assert two_batches.metadata["beta"] == pytest.approx(0.4)
    assert two_batches.evaluation is not None
    np.testing.assert_array_equal(
        two_batches.evaluation["log_likelihood"], two_batches.particles[:, 0]
    )


def test_persist_particles_prepares_missing_target_evaluation(rng) -> None:
    target, calls = _linear_target()
    particles = np.array([[-1.0], [1.0]])
    state = ParticleState(particles)
    context = StepContext(t=0, metadata={"beta": 0.0})

    archived, _, _ = PersistParticles(target).apply(state, context, rng)

    assert archived.evaluation is not None
    assert target.accepts_evaluation(archived.evaluation, context)
    np.testing.assert_array_equal(
        archived.evaluation["log_likelihood"], particles[:, 0]
    )
    assert calls == {"initial": 1, "increment": 1}


def test_first_persistent_weighting_is_direct_and_preserves_history(
    rng,
) -> None:
    target, calls = _linear_target()
    active, context = _evaluated_state(
        target,
        np.array([[-2.0], [0.0], [1.0]]),
        log_normalizing_constant=0.0,
    )
    persisted, _, _ = PersistParticles().apply(active, context, rng)
    persisted = persisted.with_updates(log_normalizing_constant=9.0)
    calls_before = dict(calls)

    weighted, weighted_context, output = PersistentTemperedWeighting(
        target, beta=0.5
    ).apply(persisted, context, rng)

    raw = 0.5 * active.particles[:, 0]
    expected_log_z = logsumexp(raw) - np.log(raw.size)
    np.testing.assert_allclose(weighted.log_weights, raw - logsumexp(raw))
    assert weighted.log_evidence == pytest.approx(expected_log_z)
    assert weighted.log_evidence != pytest.approx(9.0 + expected_log_z)
    np.testing.assert_allclose(np.exp(weighted.log_weights).sum(), 1.0)
    np.testing.assert_allclose(weighted.betas, [0.0])
    np.testing.assert_allclose(weighted.log_evidences, [0.0])
    np.testing.assert_allclose(persisted.log_weights, -np.log(3.0))
    assert persisted.log_evidence == pytest.approx(9.0)
    assert calls == calls_before
    assert weighted.evaluation is not None
    assert float(weighted_context.metadata["beta"]) == pytest.approx(0.5)
    assert float(output["delta_beta"]) == pytest.approx(0.5)
    np.testing.assert_allclose(output["persistent_log_weights"], raw)
    assert "log_weight_increment" not in output


def test_multiple_targets_and_repeated_beta_match_helper(rng) -> None:
    target, _ = _linear_target()
    first, context = _evaluated_state(
        target, np.array([[-1.0], [0.5]]), log_normalizing_constant=0.0
    )
    second, _ = _evaluated_state(
        target,
        np.array([[1.5], [2.0]]),
        log_normalizing_constant=0.4,
        beta=0.5,
    )
    history = PersistentParticleState.from_state(
        first, beta=0.0, log_normalizing_constant=0.0
    ).append(second, beta=0.5, log_normalizing_constant=0.4)
    expected_raw = persistent_log_weights(
        history.particles[:, 0],
        history.betas,
        history.log_evidences,
        0.5,
    )

    weighted, next_context, _ = PersistentTemperedWeighting(
        target, beta=0.5
    ).apply(history, context.with_metadata(beta=0.5), rng)

    expected_log_z = logsumexp(expected_raw) - np.log(len(expected_raw))
    np.testing.assert_allclose(
        weighted.log_weights, expected_raw - logsumexp(expected_raw)
    )
    assert weighted.log_evidence == pytest.approx(expected_log_z)
    np.testing.assert_allclose(weighted.betas, history.betas)
    np.testing.assert_allclose(weighted.log_evidences, history.log_evidences)
    assert float(next_context.metadata["delta_beta"]) == pytest.approx(0.0)


def test_new_target_is_archived_only_after_resampling(rng) -> None:
    target, _ = _linear_target()
    active, context = _evaluated_state(
        target, np.array([[-1.0], [1.0]]), log_normalizing_constant=0.0
    )
    persist = PersistParticles()
    history, _, _ = persist.apply(active, context, rng)
    weighted, next_context, _ = PersistentTemperedWeighting(
        target, beta=0.75
    ).apply(history, context, rng)

    resampled = weighted.resample(np.array([0, 1]))

    assert type(resampled) is ParticleState
    assert type(resampled) is not PersistentParticleState
    np.testing.assert_allclose(history.betas, [0.0])
    np.testing.assert_allclose(history.log_evidences, [0.0])

    updated_history, _, _ = persist.apply(resampled, next_context, rng)

    np.testing.assert_allclose(updated_history.betas, [0.0, 0.75])
    np.testing.assert_allclose(
        updated_history.log_evidences, [0.0, weighted.log_evidence]
    )


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax", "cupy"])
def test_persistent_steps_preserve_backend(
    backend: str, backend_factory, rng
) -> None:
    case = backend_factory(backend)
    target, _ = _linear_target()
    context = StepContext(t=0, metadata={"beta": 0.0})
    evaluation = target.prepare_evaluation(case.particles, context)
    active = case.state().with_updates(evaluation=evaluation)
    history, _, _ = PersistParticles().apply(active, context, rng)

    weighted, _, _ = PersistentTemperedWeighting(target, beta=0.4).apply(
        history, context, rng
    )

    assert namespace(weighted.particles) is case.xp
    assert namespace(weighted.log_weights) is case.xp
    assert namespace(weighted.betas) is case.xp
    assert namespace(weighted.log_evidences) is case.xp
    np.testing.assert_allclose(
        to_numpy(case.xp.sum(case.xp.exp(weighted.log_weights))), 1.0
    )


def test_persist_particles_state_round_trip(rng) -> None:
    target, _ = _linear_target()
    active, context = _evaluated_state(
        target, np.array([[0.0], [1.0]]), log_normalizing_constant=0.2
    )
    original = PersistParticles()
    original.apply(active, context, rng)

    restored = PersistParticles()
    restored.load_state_dict(original.state_dict())

    assert restored.persistent_state is not None
    np.testing.assert_array_equal(
        restored.persistent_state.particles, active.particles
    )
    np.testing.assert_allclose(restored.persistent_state.betas, [0.0])
    np.testing.assert_allclose(restored.persistent_state.log_evidences, [0.2])


def test_adaptive_persistent_weighting_jumps_to_one_and_matches_fixed(
    rng,
) -> None:
    target, _ = _linear_target()
    active, context = _evaluated_state(
        target, np.array([[-0.1], [0.0], [0.1], [0.2]])
    )
    history, _, _ = PersistParticles().apply(active, context, rng)

    adaptive, adaptive_context, output = PersistentAdaptiveTemperedWeighting(
        target, target_ess=3.0
    ).apply(history, context, rng)
    fixed, fixed_context, _ = PersistentTemperedWeighting(
        target, beta=1.0
    ).apply(history, context, rng)

    assert float(output["beta"]) == pytest.approx(1.0)
    assert float(output["effective_sample_size"]) >= 3.0
    assert float(output["target_ess"]) == pytest.approx(3.0)
    assert adaptive_context.stop_requested
    assert adaptive_context.stop_immediately
    np.testing.assert_allclose(adaptive.log_weights, fixed.log_weights)
    np.testing.assert_allclose(adaptive.log_evidence, fixed.log_evidence)
    np.testing.assert_allclose(
        adaptive_context.metadata["beta"],
        fixed_context.metadata["beta"],
    )


def test_adaptive_persistent_weighting_selects_intermediate_beta(
    rng,
) -> None:
    target, _ = _linear_target()
    active, context = _evaluated_state(
        target, np.linspace(-3.0, 3.0, 101)[:, None]
    )
    history, _, _ = PersistParticles().apply(active, context, rng)
    step = PersistentAdaptiveTemperedWeighting(
        target,
        target_ess=60.0,
        beta_tolerance=1e-9,
        max_bisection_iter=60,
    )

    weighted, next_context, output = step.apply(history, context, rng)

    beta = float(output["beta"])
    assert 0.0 < beta < 1.0
    assert float(next_context.metadata["delta_beta"]) == pytest.approx(beta)
    assert float(output["effective_sample_size"]) == pytest.approx(
        60.0, rel=1e-7
    )
    assert weighted.effective_sample_size() == pytest.approx(60.0, rel=1e-7)
    assert not next_context.stop_requested


def test_adaptive_persistent_weighting_repeats_current_beta(
    rng,
) -> None:
    target, _ = _linear_target()
    active, context = _evaluated_state(
        target, np.linspace(-1.0, 1.0, 100)[:, None]
    )
    history, _, _ = PersistParticles().apply(active, context, rng)

    weighted, next_context, output = PersistentAdaptiveTemperedWeighting(
        target, target_ess=200.0
    ).apply(history, context, rng)

    assert float(output["beta"]) == pytest.approx(0.0)
    assert float(output["delta_beta"]) == pytest.approx(0.0)
    assert float(next_context.metadata["delta_beta"]) == pytest.approx(0.0)
    assert weighted.n_particles == 100
    assert not next_context.stop_requested


def test_adaptive_persistent_weighting_accumulates_terminal_batches(
    monkeypatch,
    rng,
) -> None:
    target, _ = _linear_target()
    active, context = _evaluated_state(
        target,
        np.zeros((4, 1)),
        beta=1.0,
    )
    persist = PersistParticles()
    first_history, _, _ = persist.apply(active, context, rng)

    def fail_bisection(*args, **kwargs):
        del args, kwargs
        raise AssertionError("bisection must not run at beta one")

    monkeypatch.setattr(
        "sequince.steps.persistent.bisect_monotone_decreasing",
        fail_bisection,
    )
    step = PersistentAdaptiveTemperedWeighting(
        target,
        target_ess=3.0,
        target_total_ess=6.0,
    )

    first_weighted, first_context, first_output = step.apply(
        first_history,
        context,
        rng,
    )

    assert float(first_output["beta"]) == pytest.approx(1.0)
    assert float(first_output["delta_beta"]) == pytest.approx(0.0)
    assert float(first_output["effective_sample_size"]) == pytest.approx(4.0)
    assert float(first_output["target_total_ess"]) == pytest.approx(6.0)
    assert float(first_weighted.metadata["target_total_ess"]) == pytest.approx(
        6.0
    )
    assert not first_context.stop_requested

    repeated_active = first_weighted.resample(np.arange(4))
    second_history, _, _ = persist.apply(
        repeated_active,
        first_context,
        rng,
    )
    _, second_context, second_output = step.apply(
        second_history,
        first_context,
        rng,
    )

    np.testing.assert_allclose(second_history.betas, [1.0, 1.0])
    assert float(second_output["beta"]) == pytest.approx(1.0)
    assert float(second_output["delta_beta"]) == pytest.approx(0.0)
    assert float(second_output["effective_sample_size"]) == pytest.approx(8.0)
    assert second_context.stop_requested
    assert second_context.stop_immediately


def test_persistent_terminal_weighting_stops_runner_before_later_steps(
    rng,
) -> None:
    target, _ = _linear_target()
    active, _ = _evaluated_state(
        target, np.array([[-0.1], [0.0], [0.1], [0.2]])
    )
    calls = {"resampling": 0, "mutation": 0}

    def record_resampling(state, context, rng):
        del state, context, rng
        calls["resampling"] += 1
        raise AssertionError(
            "resampling must not run after terminal weighting"
        )

    def record_mutation(state, context, rng):
        del state, context, rng
        calls["mutation"] += 1
        raise AssertionError("mutation must not run after terminal weighting")

    smc = SequentialMonteCarlo(
        steps=[
            PersistParticles(),
            PersistentAdaptiveTemperedWeighting(target, target_ess=3.0),
            record_resampling,
            record_mutation,
        ]
    )

    result = smc.run(
        active,
        num_steps=None,
        rng=rng,
        metadata={"beta": 0.0},
    )

    assert isinstance(result.final_state, PersistentParticleState)
    assert result.final_state.metadata["beta"] == pytest.approx(1.0)
    np.testing.assert_allclose(result.final_state.weights.sum(), 1.0)
    assert calls == {"resampling": 0, "mutation": 0}


def test_persistent_runner_accumulates_terminal_batches_until_target_ess(
    rng,
) -> None:
    target, _ = _linear_target()
    active, _ = _evaluated_state(
        target,
        np.zeros((4, 1)),
        beta=1.0,
    )
    calls = {"resampling": 0, "mutation": 0}

    def record_resampling(state, context, rng):
        del rng
        calls["resampling"] += 1
        indices = np.arange(state.batch_size)
        return state.resample(indices), context, None

    def record_mutation(state, context, rng):
        del rng
        calls["mutation"] += 1
        return state, context, None

    smc = SequentialMonteCarlo(
        steps=[
            PersistParticles(),
            PersistentAdaptiveTemperedWeighting(
                target,
                target_ess=3.0,
                target_total_ess=6.0,
            ),
            record_resampling,
            record_mutation,
        ]
    )

    result = smc.run(
        active,
        num_steps=None,
        rng=rng,
        metadata={"beta": 1.0},
    )

    assert len(result) == 2
    assert calls == {"resampling": 1, "mutation": 1}
    assert isinstance(result.final_state, PersistentParticleState)
    assert result.final_state.n_batches == 2
    assert result.final_state.n_particles == 8
    np.testing.assert_allclose(result.final_state.betas, [1.0, 1.0])
    np.testing.assert_allclose(result.final_state.weights.sum(), 1.0)
    assert result.final_state.effective_sample_size() == pytest.approx(8.0)


def test_adaptive_persistent_weighting_advances_after_repeated_zero_batch(
    rng,
) -> None:
    target, _ = _linear_target()
    active, context = _evaluated_state(
        target, np.linspace(-3.0, 3.0, 100)[:, None]
    )
    persist = PersistParticles()
    first_history, _, _ = persist.apply(active, context, rng)
    step = PersistentAdaptiveTemperedWeighting(
        target,
        target_ess=150.0,
        beta_tolerance=1e-9,
        max_bisection_iter=60,
    )

    first_weighted, repeated_context, first_output = step.apply(
        first_history, context, rng
    )
    assert float(first_output["beta"]) == pytest.approx(0.0)

    repeated_active = first_weighted.resample(np.arange(100))
    second_history, _, _ = persist.apply(
        repeated_active, repeated_context, rng
    )
    second_weighted, _, second_output = step.apply(
        second_history, repeated_context, rng
    )

    assert second_history.batch_size == 100
    assert second_history.n_particles == 200
    np.testing.assert_allclose(second_history.betas, [0.0, 0.0])
    assert 0.0 < float(second_output["beta"]) < 1.0
    assert float(second_output["effective_sample_size"]) == pytest.approx(
        150.0, rel=1e-7
    )
    assert second_weighted.effective_sample_size() == pytest.approx(
        150.0, rel=1e-7
    )


def test_adaptive_and_fixed_persistent_weighting_agree_and_preserve_history(
    rng,
) -> None:
    target, _ = _linear_target()
    first, context = _evaluated_state(
        target, np.array([[-2.0], [-0.5], [0.5], [2.0]])
    )
    second, _ = _evaluated_state(
        target,
        np.array([[-1.5], [0.0], [1.0], [2.5]]),
        beta=0.25,
        log_normalizing_constant=0.3,
    )
    history = PersistentParticleState.from_state(
        first, beta=0.0, log_normalizing_constant=0.0
    ).append(second, beta=0.25, log_normalizing_constant=0.3)
    context = context.with_metadata(beta=0.25)
    original_betas = np.array(history.betas, copy=True)
    original_log_evidences = np.array(history.log_evidences, copy=True)

    adaptive, _, output = PersistentAdaptiveTemperedWeighting(
        target,
        target_ess=5.0,
        beta_tolerance=1e-10,
        max_bisection_iter=60,
    ).apply(history, context, rng)
    fixed, _, _ = PersistentTemperedWeighting(
        target, beta=output["beta"]
    ).apply(history, context, rng)

    np.testing.assert_allclose(adaptive.log_weights, fixed.log_weights)
    np.testing.assert_allclose(adaptive.log_evidence, fixed.log_evidence)
    np.testing.assert_allclose(adaptive.betas, original_betas)
    np.testing.assert_allclose(adaptive.log_evidences, original_log_evidences)
    np.testing.assert_allclose(history.betas, original_betas)
    np.testing.assert_allclose(history.log_evidences, original_log_evidences)


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax", "cupy"])
def test_adaptive_persistent_weighting_preserves_backend(
    backend: str, backend_factory, rng
) -> None:
    case = backend_factory(backend)
    target, _ = _linear_target()
    context = StepContext(t=0, metadata={"beta": 0.0})
    evaluation = target.prepare_evaluation(case.particles, context)
    active = case.state().with_updates(evaluation=evaluation)
    history, _, _ = PersistParticles().apply(active, context, rng)

    weighted, _, output = PersistentAdaptiveTemperedWeighting(
        target, target_ess=1.0
    ).apply(history, context, rng)

    assert namespace(weighted.log_weights) is case.xp
    assert namespace(output["effective_sample_size"]) is case.xp
    assert namespace(output["target_ess"]) is case.xp
    assert float(output["beta"]) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"target_ess": 0.0}, "target_ess must be positive"),
        ({"target_total_ess": 0.0}, "target_total_ess must be positive"),
        (
            {"beta_tolerance": 0.0},
            "beta_tolerance must be positive",
        ),
        ({"max_bisection_iter": 0}, "max_bisection_iter must be positive"),
    ],
)
def test_adaptive_persistent_weighting_validates_search_settings(
    kwargs: dict[str, Any], match: str
) -> None:
    target, _ = _linear_target()
    with pytest.raises(ValueError, match=match):
        PersistentAdaptiveTemperedWeighting(target, **kwargs)
