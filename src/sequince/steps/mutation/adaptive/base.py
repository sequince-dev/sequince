"""Shared lifecycle for adaptation after completed mutation blocks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from ...base import FixedLengthMutation, SMCStep, validate_n_steps
from ...context import StepContext
from ..metrics import MutationMetricsOutput

if TYPE_CHECKING:
    from orng import RandomGenerator

    from ....state import ParticleState


class AdaptiveMutationStep(SMCStep, ABC):
    """Base for eager adaptation after a complete mutation block.

    Subclasses prepare context and optionally choose a fixed budget in
    ``_prepare_mutation``, then consume finalized metrics in ``_adapt``.
    The base executes the child once, preserves its output, and forwards child
    checkpoint state. Adaptation cannot interrupt the current mutation block.
    """

    mutation: SMCStep | FixedLengthMutation

    def __init__(self, mutation: SMCStep | FixedLengthMutation) -> None:
        self.mutation = mutation

    def _prepare_mutation(
        self,
        context: StepContext,
        n_steps: int | None,
    ) -> tuple[StepContext, int | None]:
        """Prepare inputs and freeze an optional budget before execution."""
        return context, n_steps

    @abstractmethod
    def _adapt(
        self,
        metrics: MutationMetricsOutput,
        context: StepContext,
        *,
        initial_context: StepContext,
        n_steps: int | None,
    ) -> StepContext:
        """Update adaptation from completed metrics, returning context."""
        raise NotImplementedError

    @staticmethod
    def _completed_metrics(output: Any) -> MutationMetricsOutput:
        metrics = getattr(output, "metrics", None)
        if metrics is None and isinstance(output, Mapping):
            metrics = output.get("metrics")
        if not isinstance(metrics, MutationMetricsOutput):
            raise TypeError(
                "The wrapped mutation must return one complete "
                "MutationMetricsOutput result."
            )
        return metrics

    def _apply(
        self,
        state: ParticleState,
        context: StepContext,
        rng: RandomGenerator,
        *,
        n_steps: int | None,
    ) -> tuple[ParticleState, StepContext, Any]:
        context, n_steps = self._prepare_mutation(context, n_steps)
        initial_context = context
        if n_steps is None:
            if not isinstance(self.mutation, SMCStep):
                raise TypeError(
                    "mutation must be an SMCStep for default execution."
                )
            state, context, output = self.mutation.apply(state, context, rng)
        else:
            if not isinstance(self.mutation, FixedLengthMutation):
                raise TypeError(
                    "mutation must implement apply_with_n_steps()."
                )
            state, context, output = self.mutation.apply_with_n_steps(
                state, context, rng, n_steps=validate_n_steps(n_steps)
            )
        metrics = self._completed_metrics(output)
        context = self._adapt(
            metrics,
            context,
            initial_context=initial_context,
            n_steps=n_steps,
        )
        return state, context, output

    def apply(
        self,
        state: ParticleState,
        context: StepContext,
        rng: RandomGenerator,
    ) -> tuple[ParticleState, StepContext, Any]:
        return self._apply(state, context, rng, n_steps=None)

    def apply_functional(self, state, context, random_state):
        raise NotImplementedError(
            f"{type(self).__name__} supports eager execution only."
        )

    @staticmethod
    def _nested_state(component: Any, name: str) -> dict[str, Any]:
        state_fn = getattr(component, "state_dict", None)
        if not callable(state_fn):
            return {}
        state = state_fn()
        if not isinstance(state, Mapping):
            raise TypeError(f"{name} state_dict() must return a mapping.")
        return dict(state)

    @staticmethod
    def _load_nested_state(component: Any, state: Any, name: str) -> None:
        if not isinstance(state, Mapping):
            raise TypeError(
                f"AdaptiveMutationStep state '{name}' must be a mapping."
            )
        load_fn = getattr(component, "load_state_dict", None)
        if not callable(load_fn):
            if state:
                raise ValueError(f"{name} does not support checkpoint state.")
            return
        load_fn(state)

    def state_dict(self) -> dict[str, Any]:
        """Forward the wrapped mutation's checkpoint state."""
        return {"mutation": self._nested_state(self.mutation, "Mutation")}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore the wrapped mutation's checkpoint state."""
        if "mutation" in state:
            self._load_nested_state(
                self.mutation, state["mutation"], "mutation"
            )


__all__ = ["AdaptiveMutationStep"]
