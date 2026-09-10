"""Choose mutation budgets from observed acceptance probabilities."""

from __future__ import annotations

import math

from .....array_api import to_numpy
from ....base import validate_n_steps
from ....context import StepContext
from ...metrics import MutationMetricsOutput


class AcceptanceRatePolicy:
    """Target an expected number of accepted moves per particle.

    Propose ``ceil(target_steps / max(acceptance, min_acceptance))``
    for the next iteration. ``min_acceptance`` (default ``1e-6``) keeps the
    budget finite for zero or very small acceptance rates. Every returned
    budget is clipped to the required ``min_steps`` and ``max_steps`` bounds. A block with no attempted moves
    supplies no acceptance estimate, so its current budget is retained (or
    increased to one when zero) to allow a subsequent measurement.

    Acceptance is estimated from the complete block's accepted and attempted
    counts, including fractional expected counts from recycling algorithms.
    """

    def __init__(
        self,
        target_steps: float,
        *,
        min_steps: int,
        max_steps: int,
        min_acceptance: float = 1e-6,
    ) -> None:
        for name, value in (
            ("target_steps", target_steps),
            ("min_acceptance", min_acceptance),
        ):
            if isinstance(value, bool):
                raise TypeError(f"{name} must be a real number.")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if min_acceptance > 1:
            raise ValueError("min_acceptance must be <= 1.")
        self.min_steps = validate_n_steps(min_steps)
        self.max_steps = validate_n_steps(max_steps)
        if self.max_steps < self.min_steps:
            raise ValueError("max_steps must be >= min_steps.")
        self.target_steps = float(target_steps)
        self.min_acceptance = float(min_acceptance)

    def _bound(self, steps: int) -> int:
        return min(max(steps, self.min_steps), self.max_steps)

    def initialise(self, *, initial_steps: int, context: StepContext) -> int:
        return self._bound(validate_n_steps(initial_steps))

    def update(
        self,
        *,
        current_steps: int,
        metrics: MutationMetricsOutput,
        context: StepContext,
    ) -> int:
        current_steps = validate_n_steps(current_steps)
        if not isinstance(metrics, MutationMetricsOutput):
            raise TypeError(
                "AcceptanceRatePolicy requires MutationMetricsOutput."
            )
        counts = []
        for name in ("n_attempted", "n_accepted"):
            array = to_numpy(getattr(metrics, name))
            if array.size != 1:
                raise TypeError(f"{name} must be scalar.")
            value = float(array.reshape(()))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
            counts.append(value)
        attempted, accepted = counts
        if accepted > attempted:
            raise ValueError("n_accepted must not exceed n_attempted.")
        if attempted == 0:
            return self._bound(max(1, current_steps))
        acceptance = max(accepted / attempted, self.min_acceptance)
        # Integer ratios avoid overflow even for very large finite targets.
        numerator, denominator = self.target_steps.as_integer_ratio()
        a_numerator, a_denominator = acceptance.as_integer_ratio()
        numerator *= a_denominator
        denominator *= a_numerator
        return self._bound((numerator + denominator - 1) // denominator)


__all__ = ["AcceptanceRatePolicy"]
