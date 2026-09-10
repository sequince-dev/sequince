"""Policies for choosing mutation budgets from completed metrics."""

from __future__ import annotations

from typing import Protocol

from ....context import StepContext
from ...metrics import MutationMetricsOutput


class MutationStepPolicy(Protocol):
    """Choose mutation budgets using completed, iteration-level metrics.

    Policies are called only at mutation-block boundaries. In particular, they
    never receive intermediate mutation states or metrics.
    """

    def initialise(self, *, initial_steps: int, context: StepContext) -> int:
        """Choose the budget for the first adaptive iteration."""
        ...

    def update(
        self,
        *,
        current_steps: int,
        metrics: MutationMetricsOutput,
        context: StepContext,
    ) -> int:
        """Choose the proposed budget for the next outer iteration."""
        ...


__all__ = ["MutationStepPolicy"]
