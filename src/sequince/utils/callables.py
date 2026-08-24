from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from ..steps.base import StepContext


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
