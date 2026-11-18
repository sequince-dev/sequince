"""Policies for choosing mutation budgets from completed metrics."""

from __future__ import annotations

from typing import TypeVar

from ....context import StepContext
from ...metrics import MutationMetricsOutput

_Budget = TypeVar("_Budget")


class FixedMutationStepPolicy:
    """Keep the mutation budget fixed at its current value."""

    def validate_functional_configuration(self) -> None:
        """No additional static configuration is required."""

    def initialise(
        self, *, initial_steps: _Budget, context: StepContext
    ) -> _Budget:
        del context
        return initial_steps

    def update(
        self,
        *,
        current_steps: _Budget,
        metrics: MutationMetricsOutput,
        context: StepContext,
    ) -> _Budget:
        del metrics, context
        return current_steps

    # Fixed decisions have identical scalar and array semantics.
    initialise_functional = initialise
    update_functional = update


__all__ = ["FixedMutationStepPolicy"]
