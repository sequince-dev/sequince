"""Policies for choosing mutation budgets from completed metrics."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .....typing import Array
from ....context import StepContext
from ...metrics import MutationMetricsOutput


class MutationStepPolicy(Protocol):
    """Choose mutation budgets at completed-block boundaries."""

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


@runtime_checkable
class FunctionalMutationStepPolicy(MutationStepPolicy, Protocol):
    """Pure, traceable extension of the eager budget policy.

    Return scalar integer arrays matching the input budget's dtype and device.
    Do not mutate the policy or inputs, convert arrays to host values, or
    change carry structure. Methods return decisions only; validate static
    configuration on the host."""

    def validate_functional_configuration(self) -> None:
        """Reject static settings unsupported by functional decisions."""
        ...

    def initialise_functional(
        self, *, initial_steps: Array, context: StepContext
    ) -> Array:
        """Choose the first budget without modifying any inputs."""
        ...

    def update_functional(
        self,
        *,
        current_steps: Array,
        metrics: MutationMetricsOutput,
        context: StepContext,
    ) -> Array:
        """Choose a pending budget from completed-block metrics."""
        ...


__all__ = ["MutationStepPolicy", "FunctionalMutationStepPolicy"]
