import os

import numpy as np
import pytest

from sequince.stats import ess_from_log_weights
from sequince.utils import (
    enable_scipy_array_api,
    in_bounds,
    normalize_log_weights,
)


def test_enable_scipy_array_api(monkeypatch):
    monkeypatch.delenv("SCIPY_ARRAY_API")

    with pytest.warns(ImportWarning, match="setting it to '1'"):
        enable_scipy_array_api()

    assert os.environ["SCIPY_ARRAY_API"] == "1"


def test_enable_scipy_array_api_warns_for_conflicting_value(monkeypatch):
    monkeypatch.setenv("SCIPY_ARRAY_API", "0")

    with pytest.warns(RuntimeWarning, match="requires SCIPY_ARRAY_API='1'"):
        enable_scipy_array_api()


def test_normalize_log_weights():
    log_weights = np.log([1.0, 3.0]).tolist()

    normalized, log_norm = normalize_log_weights(log_weights)

    np.testing.assert_allclose(np.exp(normalized), [0.25, 0.75])
    assert np.isclose(log_norm, np.log(4.0))


def test_normalize_log_weights_can_be_jitted():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    normalized, log_norm = jax.jit(normalize_log_weights)(
        jnp.asarray([0.0, -1.0])
    )

    np.testing.assert_allclose(
        np.asarray(normalized),
        np.asarray([-0.3132617, -1.3132617]),
    )
    assert log_norm == pytest.approx(0.3132617)


@pytest.mark.parametrize(
    "log_weights",
    [
        [-np.inf, -np.inf],
        [0.0, np.nan],
        [0.0, np.inf],
    ],
)
def test_normalize_degenerate_log_weights_returns_nonfinite(log_weights):
    normalized, log_norm = normalize_log_weights(log_weights)

    assert not np.isfinite(log_norm)
    assert not np.all(np.isfinite(normalized))


def test_ess_returns_nonfinite_for_degenerate_log_weights():
    assert not np.isfinite(ess_from_log_weights(np.array([-np.inf, -np.inf])))


def test_ess_can_be_jitted():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    ess = jax.jit(ess_from_log_weights)(jnp.asarray([0.0, -1.0]))

    assert ess == pytest.approx(1.6480542)


def test_in_bounds():
    particles = np.array(
        [
            [0.0, -1.0, 0.25],
            [1.0, 2.0, 0.5],
            [-1.5, 0.5, 0.9],
        ],
        dtype=float,
    )
    bounds = np.array(
        [
            [-1.0, -2.0, 0.0],
            [1.0, 2.0, 1.0],
        ],
        dtype=float,
    )
    expected = np.array([True, True, False])
    result = in_bounds(particles, bounds)
    np.testing.assert_array_equal(result, expected)
