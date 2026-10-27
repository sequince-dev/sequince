"""Adapt a top-level proposal mixture only between outer SMC iterations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from numbers import Integral
from typing import TYPE_CHECKING, Any

from ...base import validate_n_steps
from ...context import StepContext
from ..base import MutationStep
from ..kernels import MixtureComplementaryEnsembleKernel
from ..metrics import MutationMetrics, MutationMetricsOutput
from ..metrics.base import with_metrics
from ..metrics.mixture import MixtureComponentMetrics
from ..metropolis import RedBlueMetropolisHastingsMutation
from ..repeated import RepeatedMutationStep
from .base import AdaptiveMutationStep
from .policies.de import _normalise_weights
from .policies.mixture import (
    ESJDMixtureWeightPolicy,
    MixtureWeightPolicy,
)

if TYPE_CHECKING:
    from numpy.typing import ArrayLike
    from orng import RandomGenerator

    from ....state import ParticleState


class AdaptMixtureWeights(AdaptiveMutationStep):
    """Adapt top-level mixture weights at outer SMC iteration boundaries.

    Parameters
    ----------
    mutation : RedBlueMetropolisHastingsMutation or RepeatedMutationStep
        Mixture mutation or complete repeated block. Wrap all repeated rounds
        to score their aggregate metrics while freezing weights throughout.
    policy : MixtureWeightPolicy or None
        Defaults to ESJDMixtureWeightPolicy. Child configurations, including
        DE scale weights, remain fixed.

    Notes
    -----
    Eager only. Calls at the same context iteration retain current weights and
    consult the policy once. Pending weights are installed on entering the
    next iteration. Use AdaptMutationSteps outside this wrapper for adaptive
    budgets and conditional execution outside the complete block.
    The wrapper owns its leaf kernel; do not share or replace it externally.
    MixtureComponentMetrics are installed automatically, preserving other
    metrics. Outputs include mixture_weight_adaptation with current and next
    weights. Checkpoints retain decisions, iteration and nested policy state;
    reconstruct the same components and policy configuration before loading.
    This is outer-iteration adaptation for SMC. Standalone MCMC needs a
    separate adaptation policy with suitable stationarity guarantees.
    """

    def __init__(
        self,
        mutation: RedBlueMetropolisHastingsMutation | RepeatedMutationStep,
        policy: MixtureWeightPolicy | None = None,
    ) -> None:
        leaf: MutationStep | RepeatedMutationStep = mutation
        while isinstance(leaf, RepeatedMutationStep):
            leaf = leaf.mutation
        if not isinstance(
            leaf, RedBlueMetropolisHastingsMutation
        ) or not isinstance(leaf.kernel, MixtureComplementaryEnsembleKernel):
            raise TypeError(
                "AdaptMixtureWeights requires a mixture mutation "
                "or repeated mixture block."
            )
        resolved_policy = (
            ESJDMixtureWeightPolicy() if policy is None else policy
        )
        if not callable(
            getattr(resolved_policy, "initialise", None)
        ) or not callable(getattr(resolved_policy, "update", None)):
            raise TypeError("policy must implement initialise() and update().")
        required = MixtureComponentMetrics(len(leaf.kernel.kernels))
        existing = next(
            (
                metric
                for metric in leaf.metrics
                if metric.name == required.name
            ),
            None,
        )
        if existing is None:
            leaf.metrics = MutationMetrics((*leaf.metrics, required))
        elif not isinstance(existing, MixtureComponentMetrics) or (
            existing.n_components != required.n_components
        ):
            raise ValueError(
                "mixture_components must match the configured component count."
            )
        super().__init__(mutation)
        self.policy = resolved_policy
        self._leaf = leaf
        self._kernel = leaf.kernel
        assert self._kernel.weights is not None
        self._initial_weights = tuple(self._kernel.weights)
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

    def _validate_weights(self, weights: ArrayLike) -> tuple[float, ...]:
        values = _normalise_weights(weights)
        if len(values) != len(self._initial_weights):
            raise ValueError(
                "Policy/checkpoint weights must match "
                "the fixed mixture component count."
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
            assert proposed is not None
            weights = self._validate_weights(proposed)
            self._current_weights = weights
            self._current_iteration = context.iteration
        self._leaf.kernel = replace(
            self._kernel, weights=self._current_weights
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
                "mixture_weight_adaptation": {
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
            "n_components": len(self._kernel.kernels),
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
                "AdaptMixtureWeights state is missing required keys: "
                + ", ".join(sorted(missing))
            )
        if state["n_components"] != len(self._kernel.kernels):
            raise ValueError(
                "Checkpoint component count does not match "
                "the configured mixture."
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
        self._leaf.kernel = replace(self._kernel, weights=current or initial)


__all__ = ["AdaptMixtureWeights"]
