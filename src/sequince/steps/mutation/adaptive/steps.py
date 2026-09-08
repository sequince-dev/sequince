"""Adapt the number of mutation steps between SMC iterations."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral
from typing import Any

from ...base import FixedLengthMutation, validate_n_steps
from ...context import StepContext
from ..metrics import MutationMetricsOutput
from .base import AdaptiveMutationStep
from .policies import MutationStepPolicy
from .policies.fixed import FixedMutationStepPolicy


class AdaptMutationSteps(AdaptiveMutationStep):
    """Adapt a fixed-length mutation budget between SMC iterations.

    Conditional on the completed SMC history before iteration ``t``, the
    mutation budget is chosen and frozen before mutation begins. The wrapped
    mutation then executes that complete fixed-length block. Its final
    :class:`MutationMetricsOutput` may determine the budget for iteration
    ``t + 1``, but cannot change the current block. Repeated calls with the same
    :attr:`StepContext.iteration` reuse the frozen budget and update the policy
    at most once.

    This wrapper deliberately does not implement trajectory-dependent stopping.
    Budget splitting across fixed rounds (for example, in
    ``RepeatedMutationStep``) remains the responsibility of the wrapped mutation.
    """

    def __init__(
        self,
        mutation: FixedLengthMutation,
        policy: MutationStepPolicy | None = None,
        *,
        initial_steps: int,
        min_steps: int,
        max_steps: int,
    ) -> None:
        if not isinstance(mutation, FixedLengthMutation):
            raise TypeError(
                "mutation must support fixed-length execution via "
                "apply_with_n_steps()."
            )
        self._min_steps = validate_n_steps(min_steps)
        self._max_steps = validate_n_steps(max_steps)
        if self._max_steps < self._min_steps:
            raise ValueError("max_steps must be >= min_steps.")
        self._initial_steps = validate_n_steps(initial_steps)
        if not self._min_steps <= self._initial_steps <= self._max_steps:
            raise ValueError(
                "initial_steps must be within [min_steps, max_steps]."
            )

        resolved_policy = (
            policy if policy is not None else FixedMutationStepPolicy()
        )
        if not callable(
            getattr(resolved_policy, "initialise", None)
        ) or not callable(getattr(resolved_policy, "update", None)):
            raise TypeError("policy must implement initialise() and update().")

        super().__init__(mutation)
        self.policy = resolved_policy
        self._policy_initialised = False
        self._current_iteration: int | None = None
        self._current_n_steps: int | None = None
        self._next_n_steps: int | None = None
        self._last_updated_iteration: int | None = None

    @property
    def current_n_steps(self) -> int | None:
        """The budget frozen for the current or most recent iteration."""
        return self._current_n_steps

    @property
    def next_n_steps(self) -> int | None:
        """The budget proposed for the next outer iteration, if known."""
        return self._next_n_steps

    def _bound_policy_decision(self, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError("Mutation policy decisions must be integers.")
        return min(max(int(value), self._min_steps), self._max_steps)

    def _freeze_budget(self, context: StepContext) -> int:
        if self._current_iteration == context.iteration:
            if self._current_n_steps is None:  # pragma: no cover - state guard
                raise RuntimeError("The current mutation budget is missing.")
            return self._current_n_steps

        if not self._policy_initialised:
            proposed = self.policy.initialise(
                initial_steps=self._initial_steps,
                context=context,
            )
        else:
            if self._next_n_steps is None:  # pragma: no cover - state guard
                raise RuntimeError("The next mutation budget is missing.")
            proposed = self._next_n_steps

        frozen = self._bound_policy_decision(proposed)
        self._policy_initialised = True
        self._current_iteration = context.iteration
        self._current_n_steps = frozen
        return frozen

    def _prepare_mutation(
        self,
        context: StepContext,
        n_steps: int | None,
    ) -> tuple[StepContext, int | None]:
        return context, self._freeze_budget(context)

    def _adapt(
        self,
        metrics: MutationMetricsOutput,
        context: StepContext,
        *,
        initial_context: StepContext,
        n_steps: int | None,
    ) -> StepContext:
        iteration = initial_context.iteration
        if self._last_updated_iteration != iteration:
            assert n_steps is not None
            proposed = self.policy.update(
                current_steps=n_steps,
                metrics=metrics,
                context=context,
            )
            self._next_n_steps = self._bound_policy_decision(proposed)
            self._last_updated_iteration = iteration
        return context

    def state_dict(self) -> dict[str, Any]:
        """Return the frozen/current decision and nested mutable state."""
        return {
            "policy_initialised": self._policy_initialised,
            "current_iteration": self._current_iteration,
            "current_n_steps": self._current_n_steps,
            "next_n_steps": self._next_n_steps,
            "last_updated_iteration": self._last_updated_iteration,
            **super().state_dict(),
            "policy": self._nested_state(self.policy, "Policy"),
        }

    @staticmethod
    def _optional_integer(value: Any, name: str) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(
                f"AdaptMutationSteps state '{name}' must be an integer."
            )
        return int(value)

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore an adaptive decision without consulting the policy again."""
        required = {
            "policy_initialised",
            "current_iteration",
            "current_n_steps",
            "next_n_steps",
            "last_updated_iteration",
            "mutation",
            "policy",
        }
        missing = required.difference(state)
        if missing:
            raise ValueError(
                "AdaptMutationSteps state is missing required keys: "
                + ", ".join(sorted(missing))
            )
        initialised = state["policy_initialised"]
        if not isinstance(initialised, bool):
            raise TypeError(
                "AdaptMutationSteps state 'policy_initialised' must be a boolean."
            )
        current_iteration = self._optional_integer(
            state["current_iteration"], "current_iteration"
        )
        current_n_steps = self._optional_integer(
            state["current_n_steps"], "current_n_steps"
        )
        next_n_steps = self._optional_integer(
            state["next_n_steps"], "next_n_steps"
        )
        last_updated_iteration = self._optional_integer(
            state["last_updated_iteration"], "last_updated_iteration"
        )
        for name, value in (
            ("current_n_steps", current_n_steps),
            ("next_n_steps", next_n_steps),
        ):
            if (
                value is not None
                and not self._min_steps <= value <= self._max_steps
            ):
                raise ValueError(
                    f"AdaptMutationSteps state '{name}' is outside configured bounds."
                )

        super().load_state_dict(state)
        self._load_nested_state(self.policy, state["policy"], "policy")
        self._policy_initialised = initialised
        self._current_iteration = current_iteration
        self._current_n_steps = current_n_steps
        self._next_n_steps = next_n_steps
        self._last_updated_iteration = last_updated_iteration


__all__ = ["AdaptMutationSteps"]
