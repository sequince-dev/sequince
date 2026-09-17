"""Adapt a DE scale mixture only between outer SMC iterations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from numbers import Integral
from typing import TYPE_CHECKING, Any

from ...base import validate_n_steps
from ...context import StepContext
from ..base import MutationStep
from ..kernels import DifferentialEvolutionKernel
from ..metrics import MutationMetricsOutput
from ..metrics.base import with_metrics
from ..metropolis import RedBlueMetropolisHastingsMutation
from ..repeated import RepeatedMutationStep
from .base import AdaptiveMutationStep
from .policies.de import (
    DEScaleWeightPolicy,
    ESJDScaleWeightPolicy,
    _normalise_weights,
)

if TYPE_CHECKING:
    from orng import RandomGenerator

    from ....state import ParticleState


class AdaptDEScaleWeights(AdaptiveMutationStep):
    """Replace immutable DE weights at the next outer iteration (eager only).

    Parameters
    ----------
    mutation : RedBlueMetropolisHastingsMutation or RepeatedMutationStep
        DE mutation or a complete repeated block whose leaf uses a
        DifferentialEvolutionKernel. Wrap the complete block so weights
        stay fixed across all rounds. Place conditional execution outside.
    policy : DEScaleWeightPolicy or None
        Policy selecting weights from completed metrics. Defaults to
        ESJDScaleWeightPolicy. The multiplier bank and unit-move probability
        remain unchanged.

    Notes
    -----
    Compose as ``AdaptMutationSteps(AdaptDEScaleWeights(mutation), ...)`` to
    adapt both budgets and weights. Repeated calls at the same context
    iteration reuse frozen weights and consult the policy only once. The
    next weights are installed on entering the next iteration, never midway
    through a block. The wrapped kernel configuration is owned by this wrapper;
    do not share the mutation with another adapter or replace its kernel.

    Completed metrics gain a ``de_scale_adaptation`` entry containing
    ``weights`` used for this block and ``next_weights``. Checkpoints retain
    both decisions and nested child/policy state. Reconstruct the same bank,
    kernel parameters and policy configuration before loading a checkpoint.
    """

    def __init__(
        self,
        mutation: RedBlueMetropolisHastingsMutation | RepeatedMutationStep,
        policy: DEScaleWeightPolicy | None = None,
    ) -> None:
        leaf: MutationStep | RepeatedMutationStep = mutation
        while isinstance(leaf, RepeatedMutationStep):
            leaf = leaf.mutation
        if not isinstance(
            leaf, RedBlueMetropolisHastingsMutation
        ) or not isinstance(leaf.kernel, DifferentialEvolutionKernel):
            raise TypeError(
                "AdaptDEScaleWeights requires a DE mutation "
                "or repeated DE block."
            )
        resolved_policy = ESJDScaleWeightPolicy() if policy is None else policy
        if not callable(
            getattr(resolved_policy, "initialise", None)
        ) or not callable(getattr(resolved_policy, "update", None)):
            raise TypeError("policy must implement initialise() and update().")
        super().__init__(mutation)
        self.policy = resolved_policy
        self._leaf = leaf
        self._kernel = leaf.kernel
        assert self._kernel.scale_weights is not None
        self._initial_weights = self._kernel.scale_weights
        self._current_iteration: int | None = None
        self._current_weights: tuple[float, ...] | None = None
        self._next_weights: tuple[float, ...] | None = None
        self._last_updated_iteration: int | None = None

    @property
    def current_weights(self) -> tuple[float, ...] | None:
        """Weights frozen for the current or most recent outer iteration."""
        return self._current_weights

    @property
    def next_weights(self) -> tuple[float, ...] | None:
        """Weights proposed for the next outer iteration."""
        return self._next_weights

    def _validate_weights(self, weights: Any) -> tuple[float, ...]:
        values = _normalise_weights(weights)
        if len(values) != len(self._initial_weights):
            raise ValueError(
                "Policy/checkpoint weights must match the fixed DE scale bank."
            )
        return values

    def _prepare_mutation(
        self,
        context: StepContext,
        n_steps: int | None,
    ) -> tuple[StepContext, int | None]:
        if self._current_iteration != context.iteration:
            proposed = (
                self.policy.initialise(
                    initial_weights=self._initial_weights,
                    context=context,
                )
                if self._current_iteration is None
                else self._next_weights
            )
            weights = self._validate_weights(proposed)
            self._current_weights = weights
            self._current_iteration = context.iteration
        self._leaf.kernel = replace(
            self._kernel, scale_weights=self._current_weights
        )
        return context, n_steps

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
            assert self._current_weights is not None
            proposed = self.policy.update(
                current_weights=self._current_weights,
                metrics=metrics,
                context=context,
            )
            self._next_weights = self._validate_weights(proposed)
            self._last_updated_iteration = iteration
        return context

    def _apply(
        self,
        state: ParticleState,
        context: StepContext,
        rng: RandomGenerator,
        *,
        n_steps: int | None,
    ) -> tuple[ParticleState, StepContext, Any]:
        state, context, output = super()._apply(
            state, context, rng, n_steps=n_steps
        )
        metrics = self._completed_metrics(output)
        metrics = replace(
            metrics,
            values={
                **metrics.values,
                "de_scale_adaptation": {
                    "weights": self._current_weights,
                    "next_weights": self._next_weights,
                },
            },
        )
        return state, context, with_metrics(output, metrics)

    def apply_with_n_steps(
        self,
        state: ParticleState,
        context: StepContext,
        rng: RandomGenerator,
        *,
        n_steps: int,
    ) -> tuple[ParticleState, StepContext, Any]:
        """Run a complete block with a budget supplied by an outer adapter."""
        return self._apply(
            state, context, rng, n_steps=validate_n_steps(n_steps)
        )

    def state_dict(self) -> dict[str, Any]:
        """Return frozen/pending decisions and nested child/policy state."""
        return {
            **super().state_dict(),
            "scale_multipliers": self._kernel.scale_multipliers,
            "initial_weights": self._initial_weights,
            "current_iteration": self._current_iteration,
            "current_weights": self._current_weights,
            "next_weights": self._next_weights,
            "last_updated_iteration": self._last_updated_iteration,
            "policy": self._nested_state(self.policy, "Policy"),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore decisions without rerunning the policy."""
        required = set(self.state_dict())
        if missing := required.difference(state):
            raise ValueError(
                "AdaptDEScaleWeights state is missing required keys: "
                + ", ".join(sorted(missing))
            )
        if tuple(state["scale_multipliers"]) != self._kernel.scale_multipliers:
            raise ValueError(
                "Checkpoint scale bank does not match "
                "the configured DE scale bank."
            )
        iterations = []
        for name in ("current_iteration", "last_updated_iteration"):
            value = state[name]
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or value < 0
            ):
                raise ValueError(
                    f"Checkpoint {name} must be "
                    "a non-negative integer or None."
                )
            iterations.append(None if value is None else int(value))
        current_iteration, last_updated_iteration = iterations
        initial = self._validate_weights(state["initial_weights"])
        current = (
            None
            if state["current_weights"] is None
            else self._validate_weights(state["current_weights"])
        )
        following = (
            None
            if state["next_weights"] is None
            else self._validate_weights(state["next_weights"])
        )
        if (current_iteration is None) != (current is None):
            raise ValueError(
                "Checkpoint current weights and iteration "
                "must both be present or absent."
            )
        if (last_updated_iteration is None) != (following is None) or (
            last_updated_iteration is not None
            and last_updated_iteration != current_iteration
        ):
            raise ValueError(
                "Checkpoint next weights must describe "
                "the completed current iteration."
            )
        super().load_state_dict(state)
        self._load_nested_state(self.policy, state["policy"], "policy")
        self._initial_weights = initial
        self._current_iteration = current_iteration
        self._current_weights = current
        self._next_weights = following
        self._last_updated_iteration = last_updated_iteration
        self._leaf.kernel = replace(
            self._kernel, scale_weights=current or initial
        )


__all__ = ["AdaptDEScaleWeights"]
