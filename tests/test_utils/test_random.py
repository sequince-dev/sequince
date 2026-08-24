import numpy as np
import pytest
from orng import RandomGenerator

from sequince.utils.random import resolve_rng


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
