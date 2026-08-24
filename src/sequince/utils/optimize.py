from __future__ import annotations

from typing import Any, Callable

from acai import while_loop
from array_api_compat import is_jax_namespace

from .._typing import Array
from ..array_api import namespace


def bisect_monotone_decreasing(
    criterion: Callable[[Array], Array],
    lower: Array,
    upper: Array,
    target: Array,
    *,
    tolerance: Array,
    max_iterations: int,
) -> tuple[Array, Array]:
    """Refine a bracket around a decreasing criterion's target crossing.

    The returned lower bound is the largest tested value whose criterion is at
    least ``target``; the upper bound is the smallest tested value below it.
    Callers own endpoint validation and the policy for selecting a final value
    from the returned bracket.

    The loop uses :func:`acai.while_loop`, preserving structured control flow
    under JAX compilation while retaining eager behavior on other backends.
    """
    xp = namespace(lower, upper, target, tolerance)
    uses_structured_loop = is_jax_namespace(xp)

    def continue_bisection(carry: tuple[Any, Array, Array]) -> Any:
        iteration, low, high = carry
        above_tolerance = high - low > tolerance
        if uses_structured_loop:
            return xp.logical_and(
                iteration < max_iterations,
                above_tolerance,
            )
        return iteration < max_iterations and above_tolerance

    def refine_bracket(
        carry: tuple[Any, Array, Array],
    ) -> tuple[Any, Array, Array]:
        iteration, low, high = carry
        middle = 0.5 * (low + high)
        value = criterion(middle)
        move_low = value >= target
        return (
            iteration + 1,
            xp.where(move_low, middle, low),
            xp.where(move_low, high, middle),
        )

    _, low, high = while_loop(
        continue_bisection,
        refine_bracket,
        (0, lower, upper),
        xp=xp,
    )
    return low, high
