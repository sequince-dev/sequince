from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Callable, Optional

import numpy as np

if TYPE_CHECKING:
    from .steps.base import StepContext


def _ensure_rng(
    rng: Optional[np.random.Generator] = None,
) -> np.random.Generator:
    """Return the provided RNG or create a default generator."""
    if rng is None:
        return np.random.default_rng()
    return rng


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
