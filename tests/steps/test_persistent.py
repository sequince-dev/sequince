"""Tests for persistent-sampling importance weights."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import logsumexp

from sequince.array_api import asarray, namespace, to_numpy
from sequince.steps.persistent import (
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
