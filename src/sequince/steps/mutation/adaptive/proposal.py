"""Adapt proposal parameters from completed mutation metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ...base import SMCStep, validate_n_steps
from ...context import StepContext
from ..metrics import MutationMetricsOutput
from .base import AdaptiveMutationStep

if TYPE_CHECKING:
    from orng import RandomGenerator

    from ....state import ParticleState


@dataclass(frozen=True)
class AdaptiveGaussianProposal(AdaptiveMutationStep):
    """Adapt proposal scale after a complete mutation block (eager execution).

    The wrapped step must return finalized ``MutationMetricsOutput`` in its
    output. Its output is preserved. The scale in ``context.metadata`` is
    initialized before mutation and updated afterward for the next call using
    ``scale * exp(adaptation_rate * (acceptance - target_acceptance))``.
    Proposal kernels must read ``scale_key`` from the context; this wrapper
    does not modify proposal objects. Wrap an entire ``RepeatedMutationStep``
    to keep the scale fixed across all its rounds. Place conditional execution
    outside this wrapper.

    Runtime scale lives in the context; checkpoint state forwards to the child.
    """

    mutation: SMCStep
    scale_key: str = "proposal_scale"
    initial_scale: float = 1.0
    target_acceptance: float | None = 0.3
    adaptation_rate: float = 0.1
    min_scale: float = 1e-3
    max_scale: float = 10.0

    def __post_init__(self) -> None:
        if not isinstance(self.mutation, SMCStep):
            raise TypeError("mutation must be an SMCStep.")
        if not self.scale_key:
            raise ValueError("scale_key must be nonempty.")
        values = (
            self.initial_scale,
            self.min_scale,
            self.max_scale,
            self.adaptation_rate,
        )
        if not all(np.isfinite(value) for value in values):
            raise ValueError("Scale parameters must be finite.")
        if not 0 < self.min_scale <= self.initial_scale <= self.max_scale:
            raise ValueError(
                "Require 0 < min_scale <= initial_scale <= max_scale."
            )
        if self.adaptation_rate < 0:
            raise ValueError("adaptation_rate must be nonnegative.")
        if (
            self.target_acceptance is not None
            and not 0 <= self.target_acceptance <= 1
        ):
            raise ValueError("target_acceptance must lie in [0, 1].")

    def _prepare_mutation(
        self,
        context: StepContext,
        n_steps: int | None,
    ) -> tuple[StepContext, int | None]:
        scale = float(context.metadata.get(self.scale_key, self.initial_scale))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Proposal scale must be finite and positive.")
        scale = float(np.clip(scale, self.min_scale, self.max_scale))
        context = context.with_metadata(**{self.scale_key: scale})
        return context, n_steps

    def _adapt(
        self,
        metrics: MutationMetricsOutput,
        context: StepContext,
        *,
        initial_context: StepContext,
        n_steps: int | None,
    ) -> StepContext:
        scale = float(initial_context.metadata[self.scale_key])
        if self.target_acceptance is not None and metrics.n_attempted > 0:
            acceptance = float(metrics.acceptance)
            if np.isfinite(acceptance):
                # Bound in log space to avoid overflow for large learning rates.
                log_scale = np.log(scale) + self.adaptation_rate * (
                    acceptance - self.target_acceptance
                )
                scale = float(
                    np.exp(
                        np.clip(
                            log_scale,
                            np.log(self.min_scale),
                            np.log(self.max_scale),
                        )
                    )
                )
        return context.with_metadata(**{self.scale_key: scale})

    def apply_with_n_steps(
        self,
        state: ParticleState,
        context: StepContext,
        rng: RandomGenerator,
        *,
        n_steps: int,
    ) -> tuple[ParticleState, StepContext, Any]:
        return self._apply(
            state, context, rng, n_steps=validate_n_steps(n_steps)
        )


__all__ = ["AdaptiveGaussianProposal"]
