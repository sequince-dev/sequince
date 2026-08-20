from __future__ import annotations

import copy
import inspect
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Callable, TypeAlias

import numpy as np
from array_api_compat import (
    is_array_api_obj,
    is_cupy_namespace,
    is_dask_namespace,
    is_jax_namespace,
    is_ndonnx_namespace,
    is_numpy_namespace,
    is_pydata_sparse_namespace,
    is_torch_namespace,
)
from orng import RandomGenerator

from ._scipy_compat import enable_scipy_array_api as _enable_scipy_array_api
from ._typing import Array
from .array_api import asarray, logsumexp, namespace
from .array_api import copy as copy_array
from .exceptions import DegenerateWeightsError

if TYPE_CHECKING:
    from .steps.base import StepContext

RNGLike: TypeAlias = RandomGenerator | np.random.Generator | int | None

IS_NAMESPACE_FUNCTIONS = {
    "numpy": is_numpy_namespace,
    "torch": is_torch_namespace,
    "jax": is_jax_namespace,
    "cupy": is_cupy_namespace,
    "dask": is_dask_namespace,
    "pydata_sparse": is_pydata_sparse_namespace,
    "ndonnx": is_ndonnx_namespace,
}


def array_safe_copy(value: Any) -> Any:
    """Recursively copy a value while preserving array backends.

    Array leaves are copied using :func:`sequince.array_api.copy`. This avoids
    calling :func:`copy.deepcopy` on backend arrays, which is unsupported for
    some arrays such as non-leaf PyTorch tensors. Mappings, lists, and tuples
    are copied recursively; other values use :func:`copy.deepcopy`.

    Parameters
    ----------
    value
        The value to copy.

    Returns
    -------
    Any
        An independent copy of ``value`` where supported by its array backend.
    """
    if is_array_api_obj(value):
        return copy_array(value, xp=namespace(value))
    if isinstance(value, Mapping):
        return {key: array_safe_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [array_safe_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(array_safe_copy(item) for item in value)
    return copy.deepcopy(value)


def pytree_dataclass(*, data_fields, meta_fields=()):
    try:
        import jax
    except ModuleNotFoundError as exc:
        if exc.name not in {"jax", "jaxlib"}:
            raise
        return lambda cls: cls

    return lambda cls: jax.tree_util.register_dataclass(
        cls,
        data_fields=data_fields,
        meta_fields=meta_fields,
    )


def enable_scipy_array_api() -> None:
    """Enable SciPy Array API support; normally done during package import."""
    _enable_scipy_array_api()


def normalize_log_weights(log_weights: Any, *, xp: Any | None = None):
    """Return normalized log weights and their array scalar normalizer."""
    if xp is None:
        values = asarray(log_weights)
        xp = namespace(values)
    else:
        values = asarray(log_weights, xp=xp)
    log_norm = logsumexp(values, xp=xp)
    safe_log_norm = xp.where(xp.isfinite(log_norm), log_norm, 0.0)
    return values - safe_log_norm, log_norm


def _validate_log_normalizer(log_normalizer: Any, *, xp: Any) -> None:
    """Raise for a non-finite normalizer at an eager host boundary."""
    if not bool(xp.isfinite(log_normalizer)):
        raise DegenerateWeightsError(log_normalizer)


def resolve_rng(
    rng: RNGLike = None,
    *,
    backend: str | None = None,
    xp: Any | None = None,
) -> RandomGenerator:
    """Return an ORNG generator for the supplied RNG-like value.

    Parameters
    ----------
    rng
        An RNG-like value, which can be an ORNG RandomGenerator, a NumPy
        Generator, an integer seed, or None (for a default generator).
    backend
        The backend to use for the ORNG RandomGenerator if `rng` is None or an
        integer seed. Ignored if `rng` is an ORNG RandomGenerator or a NumPy
        Generator.
    xp
        An array API namespace to use for the ORNG RandomGenerator if `rng` is
        None or an integer seed. Ignored if `rng` is an ORNG RandomGenerator or
        a NumPy Generator.

    Returns
    -------
    RandomGenerator
        An ORNG RandomGenerator instance corresponding to the provided `rng`
        value, using the specified `backend` or `xp` if necessary.

    Raises
    ------
    TypeError
        If `rng` is not an ORNG RandomGenerator, a NumPy Generator, an integer
        seed, or None.
    ValueError
        If both `backend` and `xp` are specified.
    """
    if isinstance(rng, RandomGenerator):
        return rng
    if isinstance(rng, np.random.Generator):
        return RandomGenerator(backend="numpy", generator=rng)
    if rng is not None and not isinstance(rng, int):
        raise TypeError(
            "rng must be an orng.RandomGenerator, numpy.random.Generator, "
            "an int seed, or None."
        )
    if backend is not None and xp is not None:
        raise ValueError("Specify either 'backend' or 'xp', not both.")
    if xp is not None:
        return RandomGenerator.from_xp(xp, seed=rng)
    return RandomGenerator(backend=backend or "numpy", seed=rng)


def determine_backend_name(
    x: Array | None = None, xp: Any | None = None
) -> str:
    """Determine the backend name from an array or array API module.

    Parameters
    ----------
    x : Array or None
        The array to infer the backend from. If None, xp must be provided.
    xp : Any or None
        The array API module to infer the backend from. If None, x must be provided.

    Returns
    -------
    str
        The name of the backend. If the backend cannot be determined, returns "unknown".
    """
    if x is not None:
        xp = namespace(x)
    if xp is None:
        raise ValueError(
            "Either x or xp must be provided to determine backend."
        )
    for name, is_namespace_fn in IS_NAMESPACE_FUNCTIONS.items():
        if is_namespace_fn(xp):
            return name
    return "unknown"


def call_with_optional_context(
    fn: Callable[..., Any],
    base_args: tuple[Any, ...],
    context: StepContext,
) -> Any:
    """Call fn with base_args and append context only if required."""
    signature = inspect.signature(fn)
    params = signature.parameters.values()
    positional = [
        parameter
        for parameter in params
        if parameter.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    required = sum(
        parameter.default is inspect._empty for parameter in positional
    )
    has_varargs = any(
        parameter.kind == inspect.Parameter.VAR_POSITIONAL
        for parameter in params
    )
    max_positional = float("inf") if has_varargs else len(positional)
    base_count = len(base_args)
    if required <= base_count + 1 <= max_positional:
        return fn(*base_args, context)
    if required <= base_count <= max_positional:
        return fn(*base_args)
    raise TypeError(
        "Function cannot accept the provided positional arguments."
    )


def in_bounds(particles: Array, bounds: Array, xp: Any | None = None) -> Array:
    """Check if particles are within the specified bounds.

    Bounds are assumed to be rectilinear, with shape (2, n_dimensions).


    Parameters
    ----------
    particles
        Array of shape (n_particles, n_dimensions) representing the particles.
    bounds
        Array of shape (2, n_dimensions) where bounds[0] are the lower bounds
        and bounds[1] are the upper bounds for each dimension.

    Returns
    -------
    Array
        Boolean array of shape (n_particles,) indicating whether each particle
        is within bounds.
    """
    if xp is None:
        xp = namespace(particles, bounds)
    if bounds.shape[0] != 2:
        raise ValueError("Bounds array must have shape (2, n_dimensions).")
    if bounds.shape[1] != particles.shape[-1]:
        raise ValueError(
            "Bounds array must have the same number of dimensions as particles."
        )
    return xp.all((particles >= bounds[0]) & (particles <= bounds[1]), axis=-1)
