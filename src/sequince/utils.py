from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Callable, TypeAlias

import numpy as np
from orng import RandomGenerator

from ._array_api import namespace
from ._typing import Array

if TYPE_CHECKING:
    from .steps.base import StepContext


RNGLike: TypeAlias = RandomGenerator | np.random.Generator | int | None


def resolve_rng(
    rng: RNGLike = None,
    *,
    backend: str | None = None,
) -> RandomGenerator | np.random.Generator:
    """Return the provided RNG or create an ``RandomGenerator`` for the requested backend."""
    if isinstance(rng, RandomGenerator):
        return rng
    if isinstance(rng, np.random.Generator):
        return rng
    if isinstance(rng, int):
        return RandomGenerator(backend=backend or "numpy", seed=rng)
    if rng is None:
        return RandomGenerator(backend=backend or "numpy")
    raise TypeError(
        "rng must be an RandomGenerator, numpy.random.Generator, an int seed, or None."
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
        p
        for p in params
        if p.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    required = sum(1 for p in positional if p.default is inspect._empty)
    if required > len(base_args):
        return fn(*base_args, context)

    has_varargs = any(
        p.kind == inspect.Parameter.VAR_POSITIONAL for p in params
    )
    if not has_varargs and len(base_args) > len(positional):
        raise TypeError(
            "Function cannot accept the provided number of positional arguments."
        )

    try:
        return fn(*base_args)
    except TypeError:
        if has_varargs or len(positional) > len(base_args):
            return fn(*base_args, context)
        raise


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
