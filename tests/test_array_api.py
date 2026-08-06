from __future__ import annotations

import numpy as np
import pytest
from array_api_compat import device

from sequince.array_api import (
    asarray,
    logsumexp,
    namespace,
    resolve_dtype,
    to_numpy,
)


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax", "cupy"])
def test_to_numpy_from_supported_backends(backend, backend_factory):
    case = backend_factory(backend)
    xp = case.xp
    particles = case.particles

    result = to_numpy(particles, dtype=np.float64, copy=True)

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64
    np.testing.assert_allclose(result, np.asarray(to_numpy(particles)))
    assert resolve_dtype("float32", xp) is not None


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax", "cupy"])
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


def test_to_numpy_detaches_torch_gradients():
    torch = pytest.importorskip("torch")
    tensor = torch.tensor([1.0, 2.0], requires_grad=True)

    result = to_numpy(tensor)

    np.testing.assert_allclose(result, [1.0, 2.0])


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
