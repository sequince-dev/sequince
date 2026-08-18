import os

import numpy as np
import pytest
from orng import RandomGenerator

from sequince.stats import ess_from_log_weights
from sequince.utils import (
    determine_backend_name,
    enable_scipy_array_api,
    in_bounds,
    normalize_log_weights,
    resolve_rng,
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


def test_resolve_rng_defaults_to_numpy():
    assert resolve_rng().backend == "numpy"


def test_resolve_rng_uses_array_namespace():
    jnp = pytest.importorskip("jax.numpy")

    assert resolve_rng(42, xp=jnp).backend == "jax"


def test_resolve_rng_preserves_explicit_generator():
    rng = RandomGenerator(backend="numpy", seed=42)

    assert resolve_rng(rng, xp=np) is rng


def test_resolve_rng_wraps_numpy_generator():
    assert resolve_rng(np.random.default_rng()).backend == "numpy"


def test_resolve_rng_rejects_backend_and_namespace():
    with pytest.raises(ValueError, match="either 'backend' or 'xp'"):
        resolve_rng(backend="numpy", xp=np)


def test_resolve_rng_rejects_unsupported_rng():
    with pytest.raises(TypeError, match="rng must be"):
        resolve_rng("invalid")


@pytest.mark.parametrize(
    "module_name, name",
    [
        ("numpy", "numpy"),
        ("torch", "torch"),
        ("jax", "jax"),
        ("jax.numpy", "jax"),
        ("cupy", "cupy"),
    ],
)
def test_determine_backend_name(module_name, name):
    xp = pytest.importorskip(module_name)
    assert determine_backend_name(xp=xp) == name

    x = xp.array([1.0, 2.0])
    assert determine_backend_name(x=x) == name


def test_determine_backend_name_for_unknown_namespace():
    class UnknownNamespace:
        __name__ = "not_a_real_namespace"

    assert determine_backend_name(xp=UnknownNamespace()) == "unknown"


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
