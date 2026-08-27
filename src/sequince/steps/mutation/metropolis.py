from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, cast

import array_api_extra as xpx

from ..._typing import Array
from ...array_api import asarray, namespace, to_numpy
from ...transforms import ParticleTransform, select_transform_auxiliary
from ..context import StepContext
from .base import (
    MutationStep,
    TransformRefit,
)
from .kernels import MutationKernel, evaluate_log_transition_ratio

if TYPE_CHECKING:
    from orng import RandomGenerator

    from ...state import ParticleState
    from ...target import Target, TargetEvaluation

logger = logging.getLogger(__name__)

MutationKernelFactory = Callable[[Array, StepContext], MutationKernel]
RedBlueKernelFactory = Callable[
    [Array, Array, StepContext],
    MutationKernel,
]


class MetropolisHastingsMutation(MutationStep):
    """Rejuvenate particles with a custom Metropolis-Hastings kernel.

    The kernel operates on the complete particle array, allowing both
    independent and ensemble proposals. All numerical operations remain in the
    input array namespace.
    """

    def __init__(
        self,
        log_prob_fn: Callable[..., Array | float] | "Target",
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
        *,
        auxiliary: Any = None,
    ) -> tuple[Array, "TargetEvaluation | None"]:
        xp = namespace(latent_particles)
        original, log_det = self._inverse_transform(
            latent_particles, transform, context, auxiliary=auxiliary
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
        current, transform_auxiliary = self._forward_transform(
            state.particles, transform, context, rng=rng
        )
        current = asarray(
            current,
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
            auxiliary=transform_auxiliary,
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
                proposals,
                transform,
                context,
                auxiliary=transform_auxiliary,
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

        new_particles, _ = self._inverse_transform(
            current,
            transform,
            context,
            auxiliary=transform_auxiliary,
        )
        acceptance = accepted_total / (n_particles * self._n_steps)
        stored_chain = None
        if self._store_chains:
            stored_chain = self._inverse_chain(
                xp.stack(chain, axis=0),
                transform,
                context,
                auxiliary=transform_auxiliary,
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


class RedBlueMetropolisHastingsMutation(MetropolisHastingsMutation):
    """Metropolis-Hastings updates using complementary particle ensembles.

    Each of ``n_steps`` is one complete red-blue sweep. The population is
    randomly repartitioned, the red group is updated using a kernel built from
    the frozen blue group, and then blue is updated using the accepted red
    population as its frozen complement.
    """

    def __init__(
        self,
        log_prob_fn: Callable[..., Array | float] | "Target",
        kernel_factory: RedBlueKernelFactory,
        *,
        n_steps: int = 1,
        transform: ParticleTransform | None = None,
        transform_refit: TransformRefit = "always",
        store_chains: bool = False,
    ) -> None:
        if not callable(kernel_factory):
            raise TypeError("kernel_factory must be callable.")
        MutationStep.__init__(
            self,
            log_prob_fn,
            transform=transform,
            transform_refit=transform_refit,
            store_chains=store_chains,
        )
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1.")
        self._n_steps = int(n_steps)
        self._red_blue_kernel_factory = kernel_factory

    @staticmethod
    def _split_indices(
        n_particles: int,
        rng: "RandomGenerator",
        *,
        like: Array,
    ) -> tuple[Array, Array]:
        """Randomly split particle indices into two non-empty groups."""
        if n_particles < 2:
            raise ValueError(
                "A red-blue update requires at least two particles."
            )
        xp = namespace(like)
        permutation = asarray(
            rng.choice(n_particles, size=n_particles, replace=False),
            xp=xp,
        )
        split = n_particles // 2
        group_a = permutation[:split]
        group_b = permutation[split:]
        if group_a.shape[0] == 0 or group_b.shape[0] == 0:
            raise ValueError("Red-blue groups must both be non-empty.")
        return group_a, group_b

    def _resolve_red_blue_kernel(
        self,
        active: Array,
        complementary: Array,
        context: StepContext,
    ) -> MutationKernel:
        kernel = self._red_blue_kernel_factory(
            active,
            complementary,
            context,
        )
        if not self._is_kernel(kernel):
            raise TypeError(
                "Resolved kernel must implement sample() and either "
                "log_transition_ratio() or log_density()."
            )
        return kernel

    def _apply_half(
        self,
        current: Array,
        current_log_prob: Array,
        current_evaluation: "TargetEvaluation | None",
        active_indices: Array,
        kernel: MutationKernel,
        transform: ParticleTransform | None,
        transform_auxiliary: Any,
        context: StepContext,
        rng: "RandomGenerator",
    ) -> tuple[Array, Array, "TargetEvaluation | None", int]:
        """Apply one red-blue half-update to the active particle group."""
        xp = namespace(current)
        active = current[active_indices]
        active_log_prob = current_log_prob[active_indices]
        active_auxiliary = select_transform_auxiliary(
            transform_auxiliary,
            active_indices,
            n_particles=current.shape[0],
        )
        proposals = asarray(
            kernel.sample(active, rng),
            xp=xp,
            like=active,
        )
        if proposals.shape != active.shape:
            raise ValueError(
                "Mutation kernel must return the same shape as its input."
            )
        proposed_log_prob, proposed_evaluation = self._evaluate_log_prob(
            proposals,
            transform,
            context,
            auxiliary=active_auxiliary,
        )
        transition_ratio = asarray(
            evaluate_log_transition_ratio(kernel, active, proposals),
            xp=xp,
            like=active_log_prob,
        )
        n_active = active.shape[0]
        if transition_ratio.shape != (n_active,):
            raise ValueError(
                "Mutation kernel must return one transition ratio per particle."
            )

        log_acceptance = proposed_log_prob - active_log_prob + transition_ratio
        uniforms = rng.random((n_active,))
        accepted = uniforms < xp.exp(
            xp.minimum(log_acceptance, xp.zeros_like(log_acceptance))
        )
        particle_mask = accepted
        if current.ndim > 1:
            particle_mask = xp.reshape(
                accepted,
                (n_active,) + (1,) * (current.ndim - 1),
            )
        updated_active = xp.where(particle_mask, proposals, active)
        updated_log_prob = xp.where(
            accepted,
            proposed_log_prob,
            active_log_prob,
        )
        current = xpx.at(current, active_indices).set(
            updated_active,
            copy=True,
            xp=xp,
        )
        current_log_prob = xpx.at(current_log_prob, active_indices).set(
            updated_log_prob,
            copy=True,
            xp=xp,
        )

        if current_evaluation is not None and proposed_evaluation is not None:
            active_evaluation = current_evaluation.take(active_indices).select(
                proposed_evaluation,
                accepted,
            )
            current_evaluation = current_evaluation.replace(
                active_indices,
                active_evaluation,
            )

        accepted_count = int(to_numpy(xp.sum(accepted)))
        return (
            current,
            current_log_prob,
            current_evaluation,
            accepted_count,
        )

    def apply(
        self,
        state: ParticleState,
        context: StepContext,
        rng: "RandomGenerator",
    ) -> tuple[ParticleState, StepContext, dict[str, Any]]:
        xp = namespace(state.particles)
        transform, context = self._prepare_transform(
            state.particles,
            context,
            rng=rng,
        )
        current, transform_auxiliary = self._forward_transform(
            state.particles,
            transform,
            context,
            rng=rng,
        )
        current = asarray(current, xp=xp, like=state.particles)
        n_particles = current.shape[0]
        if n_particles < 2:
            raise ValueError(
                "A red-blue update requires at least two particles."
            )
        current_log_prob, current_evaluation = self._evaluate_log_prob(
            current,
            transform,
            context,
            state.evaluation,
            auxiliary=transform_auxiliary,
        )

        accepted_total = 0
        acceptance_by_step: list[float] = []
        acceptance_by_half: list[tuple[float, float]] = []
        chain: list[Array] = []
        for _ in range(self._n_steps):
            group_a, group_b = self._split_indices(
                n_particles,
                rng,
                like=current,
            )
            kernel_a = self._resolve_red_blue_kernel(
                current[group_a],
                current[group_b],
                context,
            )
            (
                current,
                current_log_prob,
                current_evaluation,
                accepted_a,
            ) = self._apply_half(
                current,
                current_log_prob,
                current_evaluation,
                group_a,
                kernel_a,
                transform,
                transform_auxiliary,
                context,
                rng,
            )
            kernel_b = self._resolve_red_blue_kernel(
                current[group_b],
                current[group_a],
                context,
            )
            (
                current,
                current_log_prob,
                current_evaluation,
                accepted_b,
            ) = self._apply_half(
                current,
                current_log_prob,
                current_evaluation,
                group_b,
                kernel_b,
                transform,
                transform_auxiliary,
                context,
                rng,
            )
            accepted_sweep = accepted_a + accepted_b
            accepted_total += accepted_sweep
            acceptance_by_step.append(accepted_sweep / n_particles)
            acceptance_by_half.append(
                (
                    accepted_a / group_a.shape[0],
                    accepted_b / group_b.shape[0],
                )
            )
            if self._store_chains:
                chain.append(current)

        new_particles, _ = self._inverse_transform(
            current,
            transform,
            context,
            auxiliary=transform_auxiliary,
        )
        acceptance = accepted_total / (n_particles * self._n_steps)
        stored_chain = None
        if self._store_chains:
            stored_chain = self._inverse_chain(
                xp.stack(chain, axis=0),
                transform,
                context,
                auxiliary=transform_auxiliary,
                like=state.particles,
            )
        context = context.with_metadata(
            metropolis_hastings_acceptance=acceptance
        )
        logger.debug(
            "RedBlueMetropolisHastingsMutation completed: acceptance=%.3f "
            "after %d sweeps.",
            acceptance,
            self._n_steps,
        )
        output = {
            "acceptance": acceptance,
            "acceptance_by_step": tuple(acceptance_by_step),
            "acceptance_by_half": tuple(acceptance_by_half),
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
