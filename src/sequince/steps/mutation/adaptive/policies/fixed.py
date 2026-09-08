"""Policies for choosing mutation budgets from completed metrics."""

from __future__ import annotations

from ....context import StepContext
from ...metrics import MutationMetricsOutput


class FixedMutationStepPolicy:
    """Keep the mutation budget fixed at its current value."""

    def initialise(self, *, initial_steps: int, context: StepContext) -> int:
        del context
        return initial_steps

    def update(
        self,
        *,
        current_steps: int,
        metrics: MutationMetricsOutput,
        context: StepContext,
    ) -> int:
        del metrics, context
        return current_steps


__all__ = ["FixedMutationStepPolicy"]
