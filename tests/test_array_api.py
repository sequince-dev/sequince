from __future__ import annotations

import os

import numpy as np
import pytest
from array_api_compat import device

from sequince.array_api import (
    asarray,
    copy,
    determine_backend_name,
    enable_scipy_array_api,
    logsumexp,
    namespace,
    resolve_dtype,
    to_numpy,
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


@pytest.mark.parametrize(
    "module_name, name",
    [
        ("numpy", "numpy"),
        ("torch", "torch"),
        ("jax.numpy", "jax"),
        ("cupy", "cupy"),
    ],
)
def test_determine_backend_name(module_name, name):
    xp = pytest.importorskip(module_name)
    assert determine_backend_name(xp=xp) == name

    x = xp.ones(2)
    assert determine_backend_name(x=x) == name


def test_determine_backend_name_for_unknown_namespace():
    class UnknownNamespace:
        __name__ = "not_a_real_namespace"

    assert determine_backend_name(xp=UnknownNamespace()) == "unknown"


def test_to_numpy_from_supported_backends(backend, backend_factory):
    case = backend_factory(backend)
    xp = case.xp
    particles = case.particles

    result = to_numpy(particles, dtype=np.float64, copy=True)

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64
    np.testing.assert_allclose(
        result, [[0.0, 1.0], [1.0, 2.0], [2.0, 0.5], [-1.0, 1.5]]
    )
    assert resolve_dtype("float32", xp) == xp.float32
    result[0, 0] = 99.0
    assert float(particles[0, 0]) == 0.0


def test_asarray_matches_reference_namespace_device(backend, backend_factory):
    case = backend_factory(backend)
    xp = case.xp
    particles = case.particles
    source = np.arange(np.prod(particles.shape), dtype=np.float64).reshape(
        particles.shape
    )

    result = asarray(source, xp=xp, dtype="float32", like=particles)

    assert namespace(result) is namespace(particles)
    assert device(result) == device(particles)
    assert result.dtype == resolve_dtype("float32", xp)
    np.testing.assert_allclose(to_numpy(result), source)


def test_copy_returns_a_new_array(backend, backend_factory):
    case = backend_factory(backend)
    array = asarray([1.0, 2.0], xp=case.xp)
    copied = copy(array, xp=case.xp)

    assert copied is not array
    np.testing.assert_array_equal(copied, array)
    assert case.matches_namespace(copied)


def test_to_numpy_detaches_torch_gradients():
    torch = pytest.importorskip("torch")
    tensor = torch.tensor([1.0, 2.0], requires_grad=True)

    result = to_numpy(tensor)

    np.testing.assert_allclose(result, [1.0, 2.0])


@pytest.mark.filterwarnings("error")
def test_copy_preserves_torch_gradients() -> None:
    torch = pytest.importorskip("torch")
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    values = x.square()  # Non-leaf tensor.

    copied = copy(values)

    assert copied.data_ptr() != values.data_ptr()
    torch.testing.assert_close(copied, values)
    copied.sum().backward()
    torch.testing.assert_close(x.grad, torch.tensor([2.0, 4.0]))


def test_asarray_converts_between_torch_and_jax():
    torch = pytest.importorskip("torch")
    jnp = pytest.importorskip("jax.numpy")
    torch_values = torch.tensor([1.0, 2.0], requires_grad=True)
    jax_values = jnp.asarray([3.0, 4.0])

    as_jax = asarray(torch_values, xp=jnp)
    as_torch = asarray(jax_values, xp=torch)

    np.testing.assert_allclose(to_numpy(as_jax), [1.0, 2.0])
    np.testing.assert_allclose(to_numpy(as_torch), [3.0, 4.0])


def test_logsumexp_handles_all_negative_infinity():
    values = np.array([-np.inf, -np.inf])

    assert logsumexp(values) == -np.inf


def test_array_like_inputs_default_to_numpy():
    vector = asarray([1, 2], dtype="float32")
    scalar = asarray(1.0)

    assert isinstance(vector, np.ndarray)
    assert vector.dtype == np.float32
    np.testing.assert_array_equal(vector, [1.0, 2.0])
    assert isinstance(scalar, np.ndarray)
    assert np.isclose(logsumexp([0.0, 0.0]), np.log(2.0))
