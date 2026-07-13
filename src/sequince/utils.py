from __future__ import annotations

import inspect
from collections.abc import Mapping
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


def call_with_optional_args(
    fn: Callable[..., Any],
    base_args: tuple[Any, ...],
    optional_args: tuple[Any, ...] = (),
) -> Any:
    """Call ``fn`` with optional positional arguments when accepted."""
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
    has_varargs = any(
        p.kind == inspect.Parameter.VAR_POSITIONAL for p in params
    )
    base_count = len(base_args)
    max_positional = float("inf") if has_varargs else len(positional)

    for optional_count in range(len(optional_args), -1, -1):
        argument_count = base_count + optional_count
        if required <= argument_count <= max_positional:
            return fn(*base_args, *optional_args[:optional_count])
    raise TypeError(
        "Function cannot accept the provided positional arguments."
    )


def call_with_optional_context(
    fn: Callable[..., Any],
    base_args: tuple[Any, ...],
    context: StepContext,
) -> Any:
    """Call fn with base_args and append context only if required."""
    return call_with_optional_args(fn, base_args, (context,))


def snapshot_rng(rng: RandomGenerator) -> dict[str, Any]:
    """Capture RNG state through ORNG's public state protocol."""
    return rng.state_dict()


def restore_rng(snapshot: Mapping[str, Any]) -> RandomGenerator:
    """Restore an RNG from a snapshot created by :func:`snapshot_rng`."""
    return RandomGenerator.from_state_dict(snapshot)


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
