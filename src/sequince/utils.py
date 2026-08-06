from __future__ import annotations

import inspect
import math
from typing import TYPE_CHECKING, Any, Callable, TypeAlias

import numpy as np
from orng import RandomGenerator

from ._typing import Array
from .array_api import asarray, logsumexp, namespace, to_numpy
from .exceptions import DegenerateWeightsError

if TYPE_CHECKING:
    from .steps.base import StepContext


RNGLike: TypeAlias = RandomGenerator | np.random.Generator | int | None


def normalize_log_weights(log_weights: Any, *, xp: Any | None = None):
    """Return normalized log weights and their finite log normalizer.

    Raises
    ------
    DegenerateWeightsError
        If the log weights have a non-finite log normalizer.
    """
    if xp is None:
        values = asarray(log_weights)
        xp = namespace(values)
    else:
        values = asarray(log_weights, xp=xp)
    log_norm = logsumexp(values, xp=xp)
    log_norm_scalar = float(to_numpy(log_norm))
    if not math.isfinite(log_norm_scalar):
        raise DegenerateWeightsError(log_norm_scalar)
    return values - log_norm_scalar, log_norm_scalar


def resolve_rng(
    rng: RNGLike = None,
    *,
    backend: str | None = None,
) -> RandomGenerator:
    """Return an ORNG generator for the supplied RNG-like value."""
    if isinstance(rng, RandomGenerator):
        return rng
    if isinstance(rng, np.random.Generator):
        return RandomGenerator(backend="numpy", generator=rng)
    if isinstance(rng, int):
        return RandomGenerator(backend=backend or "numpy", seed=rng)
    if rng is None:
        return RandomGenerator(backend=backend or "numpy")
    raise TypeError(
        "rng must be an orng.RandomGenerator, numpy.random.Generator, "
        "an int seed, or None."
    )


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
