"""Choose the next mutation budget from completed relative jump distances."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from .....array_api import to_numpy
from ....base import validate_n_steps
from ....context import StepContext
from ...metrics import MutationMetricsOutput


@dataclass(frozen=True)
class RelativeJumpDistancePolicy:
    """Increase or decrease a fixed budget using the fraction of RJD > 1.

    Parameters
    ----------
    target_fraction : float
        Lower threshold for the fraction exceeding the radius, default 0.5.
    factor : float
        Multiplicative budget adjustment, greater than one, default 1.1.
        Below the lower threshold, multiply and round up; above the upper
        threshold, divide and round down. AdaptMutationSteps supplies bounds.
    metric_name : str
        Key containing RelativeJumpDistanceMetric results, default "rjd".

    upper_target_fraction : float or None
        Decrease only above this fraction. None uses target_fraction.
        The inclusive interval between thresholds retains the budget.

    Notes
    -----
    This population-level controller follows the increase/decrease idea in
    Buchner (2024), https://arxiv.org/abs/2402.11936. The factor and aggregation
    are implementation choices, not a convergence guarantee from that paper.
    Decisions affect the next SMC iteration, never stop the current block.
    No measured transitions retains the budget (at least one).
    """

    target_fraction: float = 0.5
    factor: float = 1.1
    metric_name: str = "rjd"
    upper_target_fraction: float | None = None

    def __post_init__(self) -> None:
        if isinstance(self.target_fraction, bool) or not (
            math.isfinite(self.target_fraction)
            and 0 < self.target_fraction < 1
        ):
            raise ValueError("target_fraction must be finite and in (0, 1).")
        if self.upper_target_fraction is not None and (
            isinstance(self.upper_target_fraction, bool)
            or not math.isfinite(self.upper_target_fraction)
            or not self.target_fraction <= self.upper_target_fraction < 1
        ):
            raise ValueError(
                "upper_target_fraction must be finite, >= target_fraction "
                "and < 1."
            )
        if isinstance(self.factor, bool) or not (
            math.isfinite(self.factor) and self.factor > 1
        ):
            raise ValueError("factor must be finite and greater than one.")
        if not isinstance(self.metric_name, str) or not self.metric_name:
            raise ValueError("metric_name must be a nonempty string.")

    def initialise(self, *, initial_steps: int, context: StepContext) -> int:
        """Retain the supplied initial budget."""
        return validate_n_steps(initial_steps)

    def update(
        self,
        *,
        current_steps: int,
        metrics: MutationMetricsOutput,
        context: StepContext,
    ) -> int:
        """Propose an integer budget from completed population diagnostics."""
        current_steps = validate_n_steps(current_steps)
        if float(to_numpy(metrics.n_steps)) == 0:
            return max(1, current_steps)
        values = metrics.values.get(self.metric_name)
        if (
            not isinstance(values, Mapping)
            or "fraction_above_one" not in values
        ):
            raise ValueError(
                f"RelativeJumpDistancePolicy requires {self.metric_name!r} "
                "metrics with fraction_above_one."
            )
        fraction_array = to_numpy(values["fraction_above_one"])
        if fraction_array.size != 1:
            raise ValueError("fraction_above_one must be scalar.")
        fraction = float(fraction_array.reshape(()))
        if not math.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError(
                "fraction_above_one must be finite and in [0, 1]."
            )
        numerator, denominator = self.factor.as_integer_ratio()
        if fraction < self.target_fraction:
            scaled = current_steps * self.factor
            # Fall back to integer arithmetic if float multiplication overflows.
            return max(
                1,
                math.ceil(scaled)
                if math.isfinite(scaled)
                else (current_steps * numerator + denominator - 1)
                // denominator,
            )
        upper = (
            self.target_fraction
            if self.upper_target_fraction is None
            else self.upper_target_fraction
        )
        if fraction > upper:
            return max(1, math.floor(current_steps / self.factor))
        return current_steps


__all__ = ["RelativeJumpDistancePolicy"]
