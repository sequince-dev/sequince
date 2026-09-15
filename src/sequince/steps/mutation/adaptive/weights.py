"""Freeze mixture probabilities within each outer iteration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from orng import RandomGenerator

from ....state import ParticleState
from ...base import SMCStep, validate_n_steps
from ...context import StepContext
from ..mixture import (
    MixtureMutation,
    MixtureMutationOutput,
    normalize_move_weights,
)
from .base import AdaptiveMutationStep
from .steps import AdaptMutationSteps


class MoveWeightPolicy(Protocol):
    """Choose relative move weights using only completed block diagnostics.

    Stateful policies must implement ``state_dict`` and ``load_state_dict``
    for reproducible checkpoints. Decisions follow the mixture's named order.
    """

    def initialise(
        self, *, initial_weights: tuple[float, ...], context: StepContext
    ) -> Sequence[float]:
        """Choose weights before the first block."""
        ...

    def update(
        self,
        *,
        current_weights: tuple[float, ...],
        output: MixtureMutationOutput,
        context: StepContext,
    ) -> Sequence[float]:
        """Propose weights for the next iteration after a completed block."""
        ...


class AdaptMoveWeights(SMCStep):
    """Adapt mixture probabilities at most once per outer iteration.

    Parameters
    ----------
    mutation : MixtureMutation
        Eager mixture receiving an explicit per-call weight override.
    policy : MoveWeightPolicy
        Policy proposing finite nonnegative relative weights with positive sum.

    Notes
    -----
    Repeated calls with the same iteration reuse the frozen decision; only
    the first completed call updates the policy. An outer AdaptMutationSteps
    can supply the total budget through ``apply_with_n_steps``. Child output
    is returned unchanged. Restore with the same configuration and separately
    restore the runner's particle state, context, and RNG state.
    """

    def __init__(
        self, mutation: MixtureMutation, policy: MoveWeightPolicy
    ) -> None:
        if not isinstance(mutation, MixtureMutation):
            raise TypeError("AdaptMoveWeights requires MixtureMutation.")
        if not callable(getattr(policy, "initialise", None)) or not callable(
            getattr(policy, "update", None)
        ):
            raise TypeError("Policy must implement initialise() and update().")
        self.mutation = mutation
        self.policy = policy
        self._current_iteration: int | None = None
        self._last_updated_iteration: int | None = None
        self._current_weights: tuple[float, ...] | None = None
        self._next_weights: tuple[float, ...] | None = None

    @property
    def current_weights(self) -> tuple[float, ...] | None:
        """Probabilities frozen for the current or most recent iteration."""
        return self._current_weights

    @property
    def next_weights(self) -> tuple[float, ...] | None:
        """Probabilities proposed for the next iteration."""
        return self._next_weights

    def apply(
        self, state: ParticleState, context: StepContext, rng: RandomGenerator
    ) -> tuple[ParticleState, StepContext, MixtureMutationOutput]:
        return self.apply_with_n_steps(
            state, context, rng, n_steps=self.mutation.n_steps
        )

    def apply_with_n_steps(
        self,
        state: ParticleState,
        context: StepContext,
        rng: RandomGenerator,
        *,
        n_steps: int,
    ) -> tuple[ParticleState, StepContext, MixtureMutationOutput]:
        """Execute the supplied budget with the iteration's frozen weights."""
        n_steps = validate_n_steps(n_steps)
        iteration = context.iteration
        if self._current_iteration != iteration:
            if self._current_weights is None:
                proposed = self.policy.initialise(
                    initial_weights=self.mutation.weights, context=context
                )
            else:
                if self._next_weights is None:
                    raise RuntimeError(
                        "No valid next weight decision is available."
                    )
                proposed = self._next_weights
            frozen = normalize_move_weights(proposed, self.mutation.names)
            self._current_weights = frozen
            self._current_iteration = iteration
        assert self._current_weights is not None
        state, context, output = self.mutation.apply_with_n_steps(
            state, context, rng, n_steps=n_steps, weights=self._current_weights
        )
        if self._last_updated_iteration != iteration:
            proposed = self.policy.update(
                current_weights=self._current_weights,
                output=output,
                context=context,
            )
            self._next_weights = normalize_move_weights(
                proposed, self.mutation.names
            )
            self._last_updated_iteration = iteration
        return state, context, output

    def state_dict(self) -> dict[str, Any]:
        """Save current/next decisions, iteration markers and nested state."""
        return {
            "current_iteration": self._current_iteration,
            "last_updated_iteration": self._last_updated_iteration,
            "current_weights": self._current_weights,
            "next_weights": self._next_weights,
            "mutation": self.mutation.state_dict(),
            "policy": AdaptiveMutationStep._nested_state(
                self.policy, "Policy"
            ),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore decisions without consulting the policy."""
        required = {
            "current_iteration",
            "last_updated_iteration",
            "current_weights",
            "next_weights",
            "mutation",
            "policy",
        }
        if not required.issubset(state):
            raise ValueError(
                "AdaptMoveWeights checkpoint is missing required keys."
            )
        current = AdaptMutationSteps._optional_integer(
            state["current_iteration"], "current_iteration"
        )
        updated = AdaptMutationSteps._optional_integer(
            state["last_updated_iteration"], "last_updated_iteration"
        )
        weights = []
        for key in ("current_weights", "next_weights"):
            value = state[key]
            weights.append(
                None
                if value is None
                else normalize_move_weights(value, self.mutation.names)
            )
        if (current is None) != (weights[0] is None):
            raise ValueError("Inconsistent current decision in checkpoint.")
        if (updated is None) != (weights[1] is None):
            raise ValueError("Inconsistent next decision in checkpoint.")
        if updated is not None and updated != current:
            raise ValueError("Checkpoint iteration markers disagree.")
        self.mutation.load_state_dict(state["mutation"])
        AdaptiveMutationStep._load_nested_state(
            self.policy, state["policy"], "policy"
        )
        self._current_iteration = current
        self._last_updated_iteration = updated
        self._current_weights, self._next_weights = weights

    def apply_functional(
        self, state: ParticleState, context: StepContext, random_state: Any
    ) -> tuple[ParticleState, StepContext, Any, MixtureMutationOutput]:
        raise NotImplementedError(
            "AdaptMoveWeights supports eager execution only."
        )


__all__ = ["AdaptMoveWeights", "MoveWeightPolicy"]
