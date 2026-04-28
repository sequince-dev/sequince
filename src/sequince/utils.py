from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Callable, TypeAlias

import numpy as np
from orng import ArrayRNG

if TYPE_CHECKING:
    from .steps.base import StepContext


RNGLike: TypeAlias = ArrayRNG | np.random.Generator | int | None


def resolve_rng(
    rng: RNGLike = None,
    *,
    backend: str | None = None,
) -> ArrayRNG:
    """Return the provided RNG or create an ``ArrayRNG`` for the requested backend."""
    if isinstance(rng, ArrayRNG):
        return rng
    if isinstance(rng, np.random.Generator):
        return ArrayRNG(backend="numpy", generator=rng)
    if isinstance(rng, int):
        return ArrayRNG(backend=backend or "numpy", seed=rng)
    if rng is None:
        return ArrayRNG(backend=backend or "numpy")
    raise TypeError(
        "rng must be an ArrayRNG, numpy.random.Generator, an int seed, or None."
    )


def _call_with_optional_context(
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
