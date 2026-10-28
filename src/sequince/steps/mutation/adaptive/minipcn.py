"""Fit MiniPCN geometry per block and adapt rho between SMC iterations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from numbers import Integral
from typing import TYPE_CHECKING, Any

from ...base import validate_n_steps
from ...context import StepContext
from ..base import MutationStep
from ..metrics import MutationMetricsOutput
from ..metrics.base import with_metrics
from ..minipcn import _MINIPCN_BLOCK_KEY, MiniPCNMutation, _MiniPCNBlock
from ..repeated import RepeatedMutationStep
from .base import AdaptiveMutationStep
from .policies.rho import (
    AcceptanceRateRhoPolicy,
    MiniPCNRhoPolicy,
    _validate_rho,
)

if TYPE_CHECKING:
    from orng import RandomGenerator

    from ....state import ParticleState


class AdaptiveMiniPCNMutation(AdaptiveMutationStep):
    """Adapt MiniPCN proposals between complete blocks (eager execution only).

    Parameters
    ----------
    mutation : MiniPCNMutation or RepeatedMutationStep
        A non-adaptive MiniPCN mutation, optionally inside repeated blocks.
        Wrap the entire repeated block to freeze geometry and rho across all
        rounds. Place conditional execution outside this wrapper.
    policy : MiniPCNRhoPolicy or None, optional
        Selects rho from completed block metrics. Defaults to
        AcceptanceRateRhoPolicy. Stateful policies are checkpointed.
    initial_rho : float, optional
        Initial proposal step size supplied to the policy, overriding the
        child's rho. Default 0.5.

    Notes
    -----
    Requires equal, finite particle weights: place after resampling. Before
    the first nonempty round, fit mu, covariance and (for tPCN) nu using
    MiniPCN's unweighted fitter in latent coordinates. The fitted transform
    and proposal are frozen for the entire block. Transform refit policies
    are consulted once per nonempty block, not once per round.

    Completed acceptance selects rho for the next context iteration. Further
    calls at the same iteration reuse rho and do not update it again; each
    call still fits geometry from its starting population. Empty blocks do
    not consume an adaptation update. Metrics include ``minipcn_adaptation``
    with ``rho`` and ``next_rho``.

    Checkpoints retain rho decisions and the child's transform state. Geometry
    is temporary block state and is fitted again at the next call. Reconstruct
    the same configuration before restoring a checkpoint. No adaptive pilot
    transitions are performed. As with other population-adaptive SMC methods,
    freezing each block is not a claim of exact finite-population invariance.
    """

    def __init__(
        self,
        mutation: MiniPCNMutation | RepeatedMutationStep,
        policy: MiniPCNRhoPolicy | None = None,
        *,
        initial_rho: float = 0.5,
    ) -> None:
        leaf: MutationStep | RepeatedMutationStep = mutation
        while isinstance(leaf, RepeatedMutationStep):
            leaf = leaf.mutation
        if not isinstance(leaf, MiniPCNMutation):
            raise TypeError(
                "AdaptiveMiniPCNMutation requires a MiniPCNMutation "
                "or a repeated MiniPCN block."
            )
        if leaf._kwargs["adaptive"]:
            raise ValueError(
                "The wrapped MiniPCNMutation must use adaptive=False."
            )
        initial_rho = _validate_rho(initial_rho)
        resolved_policy = (
            AcceptanceRateRhoPolicy() if policy is None else policy
        )
        if not callable(
            getattr(resolved_policy, "initialise", None)
        ) or not callable(getattr(resolved_policy, "update", None)):
            raise TypeError("policy must implement initialise() and update().")
        try:
            from minipcn import Sampler
        except ModuleNotFoundError as exc:
            if exc.name != "minipcn":
                raise
            raise ImportError(
                "MiniPCN mutation requires the 'minipcn' extra; install it "
                "with `pip install 'sequince[minipcn]'`."
            ) from exc

        if not hasattr(Sampler, "init_step_state"):
            raise ImportError(
                "AdaptiveMiniPCNMutation requires MiniPCN's proposal-state "
                "API. Install the minipcn revision specified by sequince."
            )
        super().__init__(mutation)
        self._leaf = leaf
        self.initial_rho = float(initial_rho)
        self.policy = resolved_policy
        self._current_rho = self.initial_rho
        self._next_rho = self.initial_rho
        self._current_iteration: int | None = None
        self._last_updated_iteration: int | None = None

    @property
    def current_rho(self) -> float:
        """Step size used for the current or most recent outer iteration."""
        return self._current_rho

    @property
    def next_rho(self) -> float:
        """Step size proposed for the next outer iteration."""
        return self._next_rho

    def _prepare_mutation(
        self, context: StepContext, n_steps: int | None
    ) -> tuple[StepContext, int | None]:
        if _MINIPCN_BLOCK_KEY in context.metadata:
            raise ValueError("Adaptive MiniPCN blocks cannot be nested.")
        if self._current_iteration != context.iteration:
            proposed = (
                self.policy.initialise(
                    initial_rho=self.initial_rho, context=context
                )
                if self._current_iteration is None
                else self._next_rho
            )
            self._current_rho = _validate_rho(proposed)
            self._next_rho = self._current_rho
            self._current_iteration = context.iteration
            self._last_updated_iteration = None
        return context.with_metadata(
            **{_MINIPCN_BLOCK_KEY: _MiniPCNBlock(self._leaf, self.current_rho)}
        ), n_steps

    def _adapt(
        self,
        metrics: MutationMetricsOutput,
        context: StepContext,
        *,
        initial_context: StepContext,
        n_steps: int | None,
    ) -> StepContext:
        iteration = initial_context.iteration
        if (
            self._last_updated_iteration != iteration
            and metrics.n_attempted > 0
        ):
            self._next_rho = _validate_rho(
                self.policy.update(
                    current_rho=self.current_rho,
                    metrics=metrics,
                    context=context,
                )
            )
            self._last_updated_iteration = iteration
        return context.without_metadata(_MINIPCN_BLOCK_KEY)

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
        return (
            state,
            context,
            with_metrics(
                output,
                replace(
                    metrics,
                    values={
                        **metrics.values,
                        "minipcn_adaptation": {
                            "rho": self.current_rho,
                            "next_rho": self.next_rho,
                        },
                    },
                ),
            ),
        )

    def apply_with_n_steps(
        self,
        state: ParticleState,
        context: StepContext,
        rng: RandomGenerator,
        *,
        n_steps: int,
    ) -> tuple[ParticleState, StepContext, Any]:
        """Run a complete block with an externally selected fixed budget."""
        return self._apply(
            state, context, rng, n_steps=validate_n_steps(n_steps)
        )

    def state_dict(self) -> dict[str, Any]:
        """Save completed-block tuning decisions and nested mutation state."""
        return {
            **super().state_dict(),
            "configuration": {
                "initial_rho": self.initial_rho,
                "step_fn": self._leaf._step_fn,
            },
            "policy": self._nested_state(self.policy, "Policy"),
            "current_rho": self.current_rho,
            "next_rho": self.next_rho,
            "current_iteration": self._current_iteration,
            "last_updated_iteration": self._last_updated_iteration,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore tuning without rerunning an adaptation update."""
        expected = self.state_dict()
        if missing := expected.keys() - state.keys():
            raise ValueError(
                "Missing MiniPCN checkpoint keys: "
                + ", ".join(sorted(missing))
            )
        if state["configuration"] != expected["configuration"]:
            raise ValueError(
                "MiniPCN checkpoint configuration does not match."
            )
        for name in ("current_rho", "next_rho"):
            _validate_rho(state[name])
        for name in ("current_iteration", "last_updated_iteration"):
            value = state[name]
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or value < 0
            ):
                raise ValueError(
                    f"Checkpoint {name} must be a nonnegative integer or None."
                )
        iteration = state["current_iteration"]
        updated = state["last_updated_iteration"]
        if updated is not None and updated != iteration:
            raise ValueError(
                "Checkpoint adaptation must describe the current iteration."
            )
        if updated is None and state["next_rho"] != state["current_rho"]:
            raise ValueError(
                "An unadapted checkpoint must retain its current rho."
            )
        if iteration is None and state["current_rho"] != self.initial_rho:
            raise ValueError("A fresh checkpoint must retain initial_rho.")
        self._load_nested_state(self.policy, state["policy"], "policy")
        super().load_state_dict(state)
        self._current_rho = float(state["current_rho"])
        self._next_rho = float(state["next_rho"])
        self._current_iteration = iteration
        self._last_updated_iteration = updated


__all__ = ["AdaptiveMiniPCNMutation"]
