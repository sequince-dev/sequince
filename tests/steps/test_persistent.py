"""Tests for persistent-sampling importance weights."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import logsumexp

from sequince.array_api import asarray, namespace, to_numpy
from sequince.posterior import TemperedPosterior
from sequince.state import ParticleState, PersistentParticleState
from sequince.steps.context import StepContext
from sequince.steps.persistent import (
    PersistentTemperedWeighting,
    PersistParticles,
    persistent_log_mixture_denominator,
    persistent_log_weights,
)


def _reference_denominator(
    log_increment: np.ndarray,
    temperatures: np.ndarray,
    log_evidences: np.ndarray,
) -> np.ndarray:
    terms = (
        log_increment[:, None] * temperatures[None, :] - log_evidences[None, :]
    )
    return logsumexp(terms, axis=1) - np.log(temperatures.size)


def test_single_initial_target() -> None:
    log_increment = np.array([-2.0, 0.5, 3.0])
    temperature = 0.4

    denominator = persistent_log_mixture_denominator(
        log_increment,
        np.array([0.0]),
        np.array([0.0]),
    )
    log_weights = persistent_log_weights(
        log_increment,
        np.array([0.0]),
        np.array([0.0]),
        temperature,
    )

    np.testing.assert_allclose(denominator, 0.0)
    np.testing.assert_allclose(log_weights, temperature * log_increment)


@pytest.mark.parametrize(
    ("temperatures", "log_evidences"),
    [
        ([0.0, 0.3, 0.8], [0.0, 0.0, 0.0]),
        ([0.0, 0.3, 0.8], [0.0, 1.7, -0.4]),
        ([0.0, 0.0, 0.5], [0.0, 0.2, 0.9]),
    ],
    ids=["multiple-targets", "nonzero-evidence", "repeated-temperatures"],
)
def test_denominator_matches_direct_reference(
    temperatures: list[float],
    log_evidences: list[float],
) -> None:
    log_increment = np.array([-1.5, 0.25, 2.0])
    temperatures_array = np.asarray(temperatures)
    log_evidences_array = np.asarray(log_evidences)

    actual = persistent_log_mixture_denominator(
        log_increment,
        temperatures_array,
        log_evidences_array,
    )
    expected = _reference_denominator(
        log_increment,
        temperatures_array,
        log_evidences_array,
    )

    np.testing.assert_allclose(actual, expected)


def test_log_weights_match_direct_reference() -> None:
    log_increment = np.array([-1.5, 0.25, 2.0])
    temperatures = np.array([0.0, 0.2, 0.9])
    log_evidences = np.array([0.0, 0.6, 1.4])
    temperature = 0.7
    expected_denominator = _reference_denominator(
        log_increment,
        temperatures,
        log_evidences,
    )

    actual = persistent_log_weights(
        log_increment,
        temperatures,
        log_evidences,
        temperature,
    )

    np.testing.assert_allclose(
        actual,
        temperature * log_increment - expected_denominator,
    )


def test_large_magnitude_values_are_stable() -> None:
    log_increment = np.array([1000.0, -1000.0, 500.0])
    temperatures = np.array([0.0, 0.5, 1.0])
    log_evidences = np.array([0.0, 25.0, 100.0])

    actual = persistent_log_mixture_denominator(
        log_increment,
        temperatures,
        log_evidences,
    )
    expected = _reference_denominator(
        log_increment,
        temperatures,
        log_evidences,
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
    temperatures = asarray([0.0, 0.0, 0.5], xp=xp, like=log_increment)
    log_evidences = asarray([0.0, 0.2, 0.9], xp=xp, like=log_increment)

    denominator = persistent_log_mixture_denominator(
        log_increment,
        temperatures,
        log_evidences,
    )
    log_weights = persistent_log_weights(
        log_increment,
        temperatures,
        log_evidences,
        0.75,
    )

    expected = _reference_denominator(
        to_numpy(log_increment),
        to_numpy(temperatures),
        to_numpy(log_evidences),
    )
    assert denominator.shape == log_increment.shape
    assert log_weights.shape == log_increment.shape
    assert namespace(denominator) is xp
    assert namespace(log_weights) is xp
    assert denominator.dtype == log_increment.dtype
    assert log_weights.dtype == log_increment.dtype
    np.testing.assert_allclose(to_numpy(denominator), expected, rtol=1e-5)


@pytest.mark.parametrize(
    ("log_increment", "temperatures", "log_evidences", "match"),
    [
        ([1.0], [], [], "must not be empty"),
        ([1.0], [0.0, 0.5], [0.0], "same length"),
        ([[1.0]], [0.0], [0.0], "log_increment"),
        ([1.0], [[0.0]], [0.0], "temperatures"),
        ([1.0], [0.0], [[0.0]], "log_evidences"),
    ],
)
def test_invalid_histories_raise_value_error(
    log_increment,
    temperatures,
    log_evidences,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        persistent_log_mixture_denominator(
            log_increment,
            temperatures,
            log_evidences,
        )


def test_candidate_temperature_must_be_scalar() -> None:
    with pytest.raises(ValueError, match="temperature must be scalar"):
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


def _evaluated_state(target, particles, *, log_evidence=0.0, temperature=0.0):
    context = StepContext(t=0, metadata={"temperature": temperature})
    evaluation = target.prepare_evaluation(particles, context)
    return (
        ParticleState(
            particles,
            log_evidence=log_evidence,
            evaluation=evaluation,
            metadata={"temperature": temperature},
        ),
        context,
    )


def test_persist_particles_creates_and_appends_completed_batches(rng) -> None:
    target, _ = _linear_target()
    first, first_context = _evaluated_state(
        target, np.array([[-1.0], [1.0]]), log_evidence=0.0
    )
    step = PersistParticles()

    one_batch, returned_context, output = step.apply(first, first_context, rng)

    assert type(one_batch) is PersistentParticleState
    assert one_batch.n_batches == 1
    np.testing.assert_allclose(one_batch.temperatures, [0.0])
    np.testing.assert_allclose(one_batch.log_evidences, [0.0])
    np.testing.assert_array_equal(one_batch.particles, first.particles)
    assert one_batch.evaluation is first.evaluation
    assert returned_context is first_context
    assert output is None

    second, second_context = _evaluated_state(
        target,
        np.array([[2.0], [3.0]]),
        log_evidence=0.6,
        temperature=0.4,
    )
    two_batches, _, _ = step.apply(second, second_context, rng)

    assert two_batches.n_batches == 2
    np.testing.assert_array_equal(
        two_batches.particles[:, 0], [-1.0, 1.0, 2.0, 3.0]
    )
    np.testing.assert_allclose(two_batches.temperatures, [0.0, 0.4])
    np.testing.assert_allclose(two_batches.log_evidences, [0.0, 0.6])
    assert two_batches.metadata["temperature"] == pytest.approx(0.4)
    assert two_batches.evaluation is not None
    np.testing.assert_array_equal(
        two_batches.evaluation["log_likelihood"], two_batches.particles[:, 0]
    )


def test_first_persistent_weighting_is_direct_and_preserves_history(
    rng,
) -> None:
    target, calls = _linear_target()
    active, context = _evaluated_state(
        target,
        np.array([[-2.0], [0.0], [1.0]]),
        log_evidence=0.0,
    )
    persisted, _, _ = PersistParticles().apply(active, context, rng)
    persisted = persisted.with_updates(log_evidence=9.0)
    calls_before = dict(calls)

    weighted, weighted_context, output = PersistentTemperedWeighting(
        target, temperature=0.5
    ).apply(persisted, context, rng)

    raw = 0.5 * active.particles[:, 0]
    expected_log_z = logsumexp(raw) - np.log(raw.size)
    np.testing.assert_allclose(weighted.log_weights, raw - logsumexp(raw))
    assert weighted.log_evidence == pytest.approx(expected_log_z)
    assert weighted.log_evidence != pytest.approx(9.0 + expected_log_z)
    np.testing.assert_allclose(np.exp(weighted.log_weights).sum(), 1.0)
    np.testing.assert_allclose(weighted.temperatures, [0.0])
    np.testing.assert_allclose(weighted.log_evidences, [0.0])
    np.testing.assert_allclose(persisted.log_weights, -np.log(3.0))
    assert persisted.log_evidence == pytest.approx(9.0)
    assert calls == calls_before
    assert weighted.evaluation is not None
    assert float(weighted_context.metadata["temperature"]) == pytest.approx(
        0.5
    )
    assert float(output["temperature_delta"]) == pytest.approx(0.5)
    np.testing.assert_allclose(output["persistent_log_weights"], raw)
    assert "log_weight_increment" not in output


def test_multiple_targets_and_repeated_temperature_match_helper(rng) -> None:
    target, _ = _linear_target()
    first, context = _evaluated_state(
        target, np.array([[-1.0], [0.5]]), log_evidence=0.0
    )
    second, _ = _evaluated_state(
        target,
        np.array([[1.5], [2.0]]),
        log_evidence=0.4,
        temperature=0.5,
    )
    history = PersistentParticleState.from_state(
        first, temperature=0.0, log_evidence=0.0
    ).append(second, temperature=0.5, log_evidence=0.4)
    expected_raw = persistent_log_weights(
        history.particles[:, 0],
        history.temperatures,
        history.log_evidences,
        0.5,
    )

    weighted, next_context, _ = PersistentTemperedWeighting(
        target, temperature=0.5
    ).apply(history, context.with_metadata(temperature=0.5), rng)

    expected_log_z = logsumexp(expected_raw) - np.log(len(expected_raw))
    np.testing.assert_allclose(
        weighted.log_weights, expected_raw - logsumexp(expected_raw)
    )
    assert weighted.log_evidence == pytest.approx(expected_log_z)
    np.testing.assert_allclose(weighted.temperatures, history.temperatures)
    np.testing.assert_allclose(weighted.log_evidences, history.log_evidences)
    assert float(next_context.metadata["temperature_delta"]) == pytest.approx(
        0.0
    )


def test_new_target_is_archived_only_after_resampling(rng) -> None:
    target, _ = _linear_target()
    active, context = _evaluated_state(
        target, np.array([[-1.0], [1.0]]), log_evidence=0.0
    )
    persist = PersistParticles()
    history, _, _ = persist.apply(active, context, rng)
    weighted, next_context, _ = PersistentTemperedWeighting(
        target, temperature=0.75
    ).apply(history, context, rng)

    resampled = weighted.resample(np.array([0, 1]))

    assert type(resampled) is ParticleState
    assert type(resampled) is not PersistentParticleState
    np.testing.assert_allclose(history.temperatures, [0.0])
    np.testing.assert_allclose(history.log_evidences, [0.0])

    updated_history, _, _ = persist.apply(resampled, next_context, rng)

    np.testing.assert_allclose(updated_history.temperatures, [0.0, 0.75])
    np.testing.assert_allclose(
        updated_history.log_evidences, [0.0, weighted.log_evidence]
    )


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax", "cupy"])
def test_persistent_steps_preserve_backend(
    backend: str, backend_factory, rng
) -> None:
    case = backend_factory(backend)
    target, _ = _linear_target()
    context = StepContext(t=0, metadata={"temperature": 0.0})
    evaluation = target.prepare_evaluation(case.particles, context)
    active = case.state().with_updates(evaluation=evaluation)
    history, _, _ = PersistParticles().apply(active, context, rng)

    weighted, _, _ = PersistentTemperedWeighting(
        target, temperature=0.4
    ).apply(history, context, rng)

    assert namespace(weighted.particles) is case.xp
    assert namespace(weighted.log_weights) is case.xp
    assert namespace(weighted.temperatures) is case.xp
    assert namespace(weighted.log_evidences) is case.xp
    np.testing.assert_allclose(
        to_numpy(case.xp.sum(case.xp.exp(weighted.log_weights))), 1.0
    )


def test_persist_particles_state_round_trip(rng) -> None:
    target, _ = _linear_target()
    active, context = _evaluated_state(
        target, np.array([[0.0], [1.0]]), log_evidence=0.2
    )
    original = PersistParticles()
    original.apply(active, context, rng)

    restored = PersistParticles()
    restored.load_state_dict(original.state_dict())

    assert restored.persistent_state is not None
    np.testing.assert_array_equal(
        restored.persistent_state.particles, active.particles
    )
    np.testing.assert_allclose(restored.persistent_state.temperatures, [0.0])
    np.testing.assert_allclose(restored.persistent_state.log_evidences, [0.2])
