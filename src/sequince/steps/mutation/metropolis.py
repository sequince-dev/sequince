from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, cast

from ..._typing import Array
from ...array_api import asarray, namespace, to_numpy
from ...transforms import ParticleTransform
from ..context import StepContext
from .base import (
    MutationStep,
    TransformRefit,
)
from .kernels import MutationKernel, evaluate_log_transition_ratio

if TYPE_CHECKING:
    from orng import RandomGenerator

    from ...state import ParticleState
    from ...target import SequentialTarget, TargetEvaluation

logger = logging.getLogger(__name__)

MutationKernelFactory = Callable[[Array, StepContext], MutationKernel]


class MetropolisHastingsMutation(MutationStep):
    """Rejuvenate particles with a custom Metropolis-Hastings kernel.

    The kernel operates on the complete particle array, allowing both
    independent and ensemble proposals. All numerical operations remain in the
    input array namespace.
    """

    def __init__(
        self,
        log_prob_fn: Callable[..., Array | float] | "SequentialTarget",
        kernel: MutationKernel | MutationKernelFactory,
        *,
        n_steps: int = 1,
        transform: ParticleTransform | None = None,
        transform_refit: TransformRefit = "always",
        store_chains: bool = False,
    ):
        super().__init__(
            log_prob_fn,
            transform=transform,
            transform_refit=transform_refit,
            store_chains=store_chains,
        )
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1.")
        self._n_steps = int(n_steps)
        if self._is_kernel(kernel):
            self._kernel: MutationKernel | None = cast(MutationKernel, kernel)
            self._kernel_factory: MutationKernelFactory | None = None
        elif callable(kernel):
            self._kernel = None
            self._kernel_factory = cast(MutationKernelFactory, kernel)
        else:
            raise TypeError(
                "kernel must implement the mutation-kernel interface or be "
                "a callable returning one."
            )

    @staticmethod
    def _is_kernel(kernel: object) -> bool:
        has_correction = hasattr(kernel, "log_transition_ratio") or hasattr(
            kernel, "log_density"
        )
        return hasattr(kernel, "sample") and has_correction

    def _resolve_kernel(
        self, particles: Array, context: StepContext
    ) -> MutationKernel:
        if self._kernel_factory is not None:
            kernel = self._kernel_factory(particles, context)
        elif self._kernel is not None:
            kernel = self._kernel
        else:  # pragma: no cover - guarded by __init__
            raise RuntimeError("Mutation kernel is not configured.")
        if not self._is_kernel(kernel):
            raise TypeError(
                "Resolved kernel must implement sample() and either "
                "log_transition_ratio() or log_density()."
            )
        return kernel

    def _evaluate_log_prob(
        self,
        latent_particles: Array,
        transform: ParticleTransform | None,
        context: StepContext,
        evaluation: "TargetEvaluation | None" = None,
    ) -> tuple[Array, "TargetEvaluation | None"]:
        xp = namespace(latent_particles)
        original, log_det = self._inverse_transform(
            latent_particles, transform, context
        )
        result, evaluation = self._evaluate_target(
            original,
            context,
            evaluation=evaluation,
        )
        result = self._combine_with_log_det(result, log_det)
        values = asarray(result, xp=xp, like=latent_particles)
        n_particles = latent_particles.shape[0]
        if values.ndim == 0:
            values = xp.broadcast_to(values, (n_particles,))
        elif values.ndim > 1:
            values = xp.squeeze(values)
        if values.ndim != 1 or values.shape[0] != n_particles:
            raise ValueError(
                "log_prob_fn must return a scalar or one value per particle."
            )
        return values, evaluation

    def apply(
        self,
        state: ParticleState,
        context: StepContext,
        rng: "RandomGenerator",
    ) -> tuple[ParticleState, StepContext, dict[str, Any]]:
        xp = namespace(state.particles)
        transform, context = self._prepare_transform(
            state.particles, context, rng=rng
        )
        current = asarray(
            self._forward_transform(state.particles, transform, context),
            xp=xp,
            like=state.particles,
        )
        n_particles = current.shape[0]
        kernel = self._resolve_kernel(current, context)
        current_log_prob, current_evaluation = self._evaluate_log_prob(
            current,
            transform,
            context,
            state.evaluation,
        )

        accepted_total = 0
        acceptance_by_step: list[float] = []
        chain: list[Array] = []
        for _ in range(self._n_steps):
            proposals = asarray(
                kernel.sample(current, rng), xp=xp, like=current
            )
            if proposals.shape != current.shape:
                raise ValueError(
                    "Mutation kernel must return the same shape as its input."
                )
            proposed_log_prob, proposed_evaluation = self._evaluate_log_prob(
                proposals, transform, context
            )
            transition_ratio = asarray(
                evaluate_log_transition_ratio(kernel, current, proposals),
                xp=xp,
                like=current_log_prob,
            )
            if transition_ratio.shape != (n_particles,):
                raise ValueError(
                    "Mutation kernel must return one transition ratio per "
                    "particle."
                )

            log_acceptance = (
                proposed_log_prob - current_log_prob + transition_ratio
            )
            uniforms = rng.random((n_particles,))
            zeros = xp.zeros_like(log_acceptance)
            accepted = uniforms < xp.exp(xp.minimum(log_acceptance, zeros))

            if current.ndim == 1:
                particle_mask = accepted
            else:
                particle_mask = xp.reshape(
                    accepted,
                    (n_particles,) + (1,) * (current.ndim - 1),
                )
            current = xp.where(particle_mask, proposals, current)
            current_log_prob = xp.where(
                accepted, proposed_log_prob, current_log_prob
            )
            if (
                current_evaluation is not None
                and proposed_evaluation is not None
            ):
                current_evaluation = current_evaluation.select(
                    proposed_evaluation,
                    accepted,
                )

            accepted_count = int(to_numpy(xp.sum(accepted)))
            accepted_total += accepted_count
            acceptance_by_step.append(accepted_count / n_particles)
            if self._store_chains:
                chain.append(current)

        new_particles, _ = self._inverse_transform(current, transform, context)
        acceptance = accepted_total / (n_particles * self._n_steps)
        stored_chain = None
        if self._store_chains:
            stored_chain = self._inverse_chain(
                xp.stack(chain, axis=0),
                transform,
                context,
                like=state.particles,
            )

        context = context.with_metadata(
            metropolis_hastings_acceptance=acceptance
        )
        logger.debug(
            "MetropolisHastingsMutation completed: acceptance=%.3f after %d "
            "steps.",
            acceptance,
            self._n_steps,
        )
        output = {
            "acceptance": acceptance,
            "acceptance_by_step": tuple(acceptance_by_step),
            "steps": self._n_steps,
            "chain": stored_chain,
        }
        return (
            state.replace_particles(
                new_particles,
                evaluation=current_evaluation,
            ),
            context,
            output,
        )
