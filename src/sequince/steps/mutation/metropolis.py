from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Callable, cast

import array_api_extra as xpx
import optree

from ..._typing import Array
from ...array_api import asarray, get_device, namespace, to_numpy
from ...target import EvaluationCounts
from ...transforms import ParticleTransform, select_transform_auxiliary
from ...utils.pytrees import OPTREE_NAMESPACE
from ..base import validate_n_steps
from ..context import StepContext
from .base import (
    MutationStep,
    TransformRefit,
)
from .kernels import (
    ComplementaryEnsembleKernel,
    DifferentialEvolutionKernel,
    GammaRandomization,
    MutationKernel,
    _as_kernel_sample,
    evaluate_log_transition_ratio,
)
from .metrics import (
    MutationMetric,
    MutationStepState,
    MutationTransition,
)
from .metrics.de import DEScaleMetrics

if TYPE_CHECKING:
    from orng import RandomGenerator

    from ...state import ParticleState
    from ...target import Target, TargetEvaluation

logger = logging.getLogger(__name__)

MutationKernelFactory = Callable[[Array, StepContext], MutationKernel]


def _combine_kernel_auxiliary(
    first: Any,
    second: Any,
    n_first: int,
    n_second: int,
    order: Array,
) -> Any:
    """Combine matching particle-aligned auxiliary pytrees in population order.

    Every leaf must be an array whose leading dimension is the active half's
    particle count. None nodes are allowed only at matching tree positions.
    Concatenation and indexing preserve the leaves' backend and device.
    """
    leaves_a, tree_a = optree.tree_flatten(first, namespace=OPTREE_NAMESPACE)
    leaves_b, tree_b = optree.tree_flatten(second, namespace=OPTREE_NAMESPACE)
    if tree_a != tree_b:
        raise ValueError(
            "Kernel auxiliary pytree structures must match between halves."
        )
    combined = []
    for a, b in zip(leaves_a, leaves_b, strict=True):
        if (
            not hasattr(a, "shape")
            or not hasattr(b, "shape")
            or len(a.shape) == 0
            or len(b.shape) == 0
            or a.shape[0] != n_first
            or b.shape[0] != n_second
            or a.shape[1:] != b.shape[1:]
        ):
            raise ValueError(
                "Kernel auxiliary leaves must be particle-aligned arrays with matching trailing shapes."
            )
        xp = namespace(a, b)
        combined.append(xp.concat((a, b), axis=0)[order])
    return optree.tree_unflatten(tree_a, combined)


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
        metrics: tuple[MutationMetric, ...] | list[MutationMetric] = (),
    ):
        super().__init__(
            log_prob_fn,
            transform=transform,
            transform_refit=transform_refit,
            store_chains=store_chains,
            metrics=metrics,
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
        has_correction = (
            hasattr(kernel, "log_transition_ratio")
            or hasattr(kernel, "log_density")
            or hasattr(kernel, "log_transition_ratio_from_sample")
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
    ) -> tuple[Array, "TargetEvaluation | None", EvaluationCounts]:
        xp = namespace(latent_particles)
        original, log_det = self._inverse_transform(
            latent_particles, transform, context, auxiliary=auxiliary
        )
        result, evaluation, counts = self._evaluate_target(
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
        return values, evaluation, counts

    def apply(
        self,
        state: ParticleState,
        context: StepContext,
        rng: "RandomGenerator",
    ) -> tuple[ParticleState, StepContext, dict[str, Any]]:
        return self.apply_with_n_steps(
            state, context, rng, n_steps=self._n_steps
        )

    def apply_with_mutation_state(
        self,
        state: ParticleState,
        context: StepContext,
        rng: "RandomGenerator",
        *,
        n_steps: int,
        mutation_state: MutationStepState | None = None,
    ) -> tuple[
        ParticleState, StepContext, dict[str, Any], MutationStepState | None
    ]:
        n_steps = validate_n_steps(n_steps)
        if n_steps == 0:
            xp = namespace(state.particles)
            mutation_state = (
                self.metrics.initialize(
                    state.particles, state.log_weights, context
                )
                if mutation_state is None
                else self.metrics.prepare(
                    mutation_state, state.particles, context
                )
            )
            empty_chain = (
                xp.empty(
                    (0, *state.particles.shape),
                    dtype=state.particles.dtype,
                    device=get_device(state.particles),
                )
                if self._store_chains
                else None
            )
            return (
                state,
                context,
                {
                    "acceptance_by_step": (),
                    "steps": 0,
                    "chain": empty_chain,
                    "metrics": None,
                },
                mutation_state,
            )
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
        mutation_state = (
            self.metrics.initialize(current, state.log_weights, context)
            if mutation_state is None
            else self.metrics.prepare(mutation_state, current, context)
        )
        kernel = self._resolve_kernel(current, context)
        current_log_prob, current_evaluation, counts = self._evaluate_log_prob(
            current,
            transform,
            context,
            state.evaluation,
            auxiliary=transform_auxiliary,
        )

        accepted_total = 0
        acceptance_by_step: list[float] = []
        chain: list[Array] = []
        for _ in range(n_steps):
            proposal = _as_kernel_sample(kernel.sample(current, rng))
            proposal = replace(
                proposal,
                particles=asarray(proposal.particles, xp=xp, like=current),
            )
            proposals = proposal.particles
            if proposals.shape != current.shape:
                raise ValueError(
                    "Mutation kernel must return the same shape as its input."
                )
            (
                proposed_log_prob,
                proposed_evaluation,
                proposed_counts,
            ) = self._evaluate_log_prob(
                proposals,
                transform,
                context,
                auxiliary=transform_auxiliary,
            )
            counts = counts + proposed_counts
            transition_ratio, transition_counts = (
                evaluate_log_transition_ratio(
                    kernel,
                    current,
                    proposal,
                    return_counts=True,
                )
            )
            counts = counts + transition_counts
            transition_ratio = asarray(
                transition_ratio,
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
            updated = xp.where(particle_mask, proposals, current)
            transition = MutationTransition(
                updated - current, proposal.auxiliary
            )
            current = updated
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
            mutation_state = self.metrics.update(
                mutation_state,
                current,
                accepted,
                context,
                transition=transition,
            )
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
        acceptance = accepted_total / (n_particles * n_steps)
        stored_chain = None
        if self._store_chains:
            stored_chain = self._inverse_chain(
                xp.stack(chain, axis=0),
                transform,
                context,
                auxiliary=transform_auxiliary,
                like=state.particles,
            )

        context = context.add_evaluation_counts(counts).with_metadata(
            metropolis_hastings_acceptance=acceptance
        )
        logger.debug(
            "MetropolisHastingsMutation completed: acceptance=%.3f after %d "
            "steps.",
            acceptance,
            n_steps,
        )
        output = {
            "acceptance_by_step": tuple(acceptance_by_step),
            "steps": n_steps,
            "chain": stored_chain,
            "metrics": None,
        }
        return (
            state.replace_particles(
                new_particles,
                evaluation=current_evaluation,
            ),
            context,
            output,
            mutation_state,
        )


class RedBlueMetropolisHastingsMutation(MetropolisHastingsMutation):
    """Metropolis-Hastings updates using complementary particle ensembles.

    Each of ``n_steps`` is one complete red-blue sweep. The population is
    randomly repartitioned, the red group is updated against the frozen blue
    group, and then blue is updated against the accepted red population. A
    persistent :class:`ComplementaryEnsembleKernel` receives both groups
    directly.
    """

    def __init__(
        self,
        log_prob_fn: Callable[..., Array | float] | "Target",
        kernel: ComplementaryEnsembleKernel,
        *,
        n_steps: int = 1,
        transform: ParticleTransform | None = None,
        transform_refit: TransformRefit = "always",
        store_chains: bool = False,
        metrics: tuple[MutationMetric, ...] | list[MutationMetric] = (),
    ) -> None:
        if not isinstance(kernel, ComplementaryEnsembleKernel):
            raise TypeError("kernel must be a ComplementaryEnsembleKernel.")
        self._complementary_ensemble_kernel = kernel
        if isinstance(kernel, DifferentialEvolutionKernel):
            metrics = (*metrics, DEScaleMetrics(kernel.scale_multipliers))
        MutationStep.__init__(
            self,
            log_prob_fn,
            transform=transform,
            transform_refit=transform_refit,
            store_chains=store_chains,
            metrics=metrics,
        )
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1.")
        self._n_steps = int(n_steps)

    @property
    def kernel(self) -> ComplementaryEnsembleKernel:
        """Fixed complementary kernel; replace between mutation calls."""
        return self._complementary_ensemble_kernel

    @kernel.setter
    def kernel(self, kernel: ComplementaryEnsembleKernel) -> None:
        if not isinstance(kernel, ComplementaryEnsembleKernel):
            raise TypeError("kernel must be a ComplementaryEnsembleKernel.")
        previous = self._complementary_ensemble_kernel
        if isinstance(kernel, DifferentialEvolutionKernel) != isinstance(
            previous, DifferentialEvolutionKernel
        ):
            raise ValueError("Replacement must preserve the kernel family.")
        if isinstance(previous, DifferentialEvolutionKernel) and (
            not isinstance(kernel, DifferentialEvolutionKernel)
            or kernel.scale_multipliers != previous.scale_multipliers
        ):
            raise ValueError("Replacement must preserve the DE scale bank.")
        self._complementary_ensemble_kernel = kernel

    @staticmethod
    def _split_indices(
        n_particles: int,
        rng: "RandomGenerator",
    ) -> tuple[Array, Array]:
        """Randomly split particle indices into two non-empty groups."""
        if n_particles < 2:
            raise ValueError(
                "A red-blue update requires at least two particles."
            )
        permutation = rng.choice(
            n_particles,
            size=n_particles,
            replace=False,
        )
        split = n_particles // 2
        group_a = permutation[:split]
        group_b = permutation[split:]
        if group_a.shape[0] == 0 or group_b.shape[0] == 0:
            raise ValueError("Red-blue groups must both be non-empty.")
        return group_a, group_b

    def _apply_half(
        self,
        current: Array,
        current_log_prob: Array,
        current_evaluation: "TargetEvaluation | None",
        active_indices: Array,
        complementary: Array,
        kernel: ComplementaryEnsembleKernel,
        transform: ParticleTransform | None,
        transform_auxiliary: Any,
        context: StepContext,
        rng: "RandomGenerator",
    ) -> tuple[
        Array,
        Array,
        "TargetEvaluation | None",
        EvaluationCounts,
        int,
        Array,
        Any,
        Array,
    ]:
        """Apply one red-blue half-update to the active particle group."""
        xp = namespace(current)
        active = current[active_indices]
        active_log_prob = current_log_prob[active_indices]
        active_auxiliary = select_transform_auxiliary(
            transform_auxiliary,
            active_indices,
            n_particles=current.shape[0],
        )
        proposal = _as_kernel_sample(kernel.sample(active, complementary, rng))
        proposals = proposal.particles
        if proposals.shape != active.shape:
            raise ValueError(
                "Mutation kernel must return the same shape as its input."
            )
        (
            proposed_log_prob,
            proposed_evaluation,
            counts,
        ) = self._evaluate_log_prob(
            proposals,
            transform,
            context,
            auxiliary=active_auxiliary,
        )
        transition_ratio = evaluate_log_transition_ratio(
            kernel, active, proposal, complementary
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
            counts,
            accepted_count,
            accepted,
            proposal.auxiliary,
            updated_active - active,
        )

    def apply(
        self,
        state: ParticleState,
        context: StepContext,
        rng: "RandomGenerator",
    ) -> tuple[ParticleState, StepContext, dict[str, Any]]:
        return self.apply_with_n_steps(
            state, context, rng, n_steps=self._n_steps
        )

    def apply_with_mutation_state(
        self,
        state: ParticleState,
        context: StepContext,
        rng: "RandomGenerator",
        *,
        n_steps: int,
        mutation_state: MutationStepState | None = None,
    ) -> tuple[
        ParticleState, StepContext, dict[str, Any], MutationStepState | None
    ]:
        kernel = self.kernel  # Snapshot once for the complete mutation call.
        n_steps = validate_n_steps(n_steps)
        if n_steps == 0:
            xp = namespace(state.particles)
            mutation_state = (
                self.metrics.initialize(
                    state.particles, state.log_weights, context
                )
                if mutation_state is None
                else self.metrics.prepare(
                    mutation_state, state.particles, context
                )
            )
            empty_chain = (
                xp.empty(
                    (0, *state.particles.shape),
                    dtype=state.particles.dtype,
                    device=get_device(state.particles),
                )
                if self._store_chains
                else None
            )
            return (
                state,
                context,
                {
                    "acceptance_by_step": (),
                    "acceptance_by_half": (),
                    "steps": 0,
                    "chain": empty_chain,
                    "metrics": None,
                },
                mutation_state,
            )
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
        mutation_state = (
            self.metrics.initialize(current, state.log_weights, context)
            if mutation_state is None
            else self.metrics.prepare(mutation_state, current, context)
        )
        if n_particles < 2:
            raise ValueError(
                "A red-blue update requires at least two particles."
            )
        (
            current_log_prob,
            current_evaluation,
            counts,
        ) = self._evaluate_log_prob(
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
        for _ in range(n_steps):
            accepted_population = xp.zeros_like(
                current_log_prob, dtype=xp.bool
            )
            group_a, group_b = self._split_indices(
                n_particles,
                rng,
            )
            complementary_a = current[group_b]
            (
                current,
                current_log_prob,
                current_evaluation,
                counts_a,
                accepted_a,
                accepted_mask_a,
                auxiliary_a,
                displacement_a,
            ) = self._apply_half(
                current,
                current_log_prob,
                current_evaluation,
                group_a,
                complementary_a,
                kernel,
                transform,
                transform_auxiliary,
                context,
                rng,
            )
            counts = counts + counts_a
            accepted_population = xpx.at(accepted_population, group_a).set(
                accepted_mask_a, copy=True, xp=xp
            )
            complementary_b = current[group_a]
            (
                current,
                current_log_prob,
                current_evaluation,
                counts_b,
                accepted_b,
                accepted_mask_b,
                auxiliary_b,
                displacement_b,
            ) = self._apply_half(
                current,
                current_log_prob,
                current_evaluation,
                group_b,
                complementary_b,
                kernel,
                transform,
                transform_auxiliary,
                context,
                rng,
            )
            counts = counts + counts_b
            accepted_population = xpx.at(accepted_population, group_b).set(
                accepted_mask_b, copy=True, xp=xp
            )
            # Each particle is updated once; restore both halves to population
            # order so all metrics observe the same complete transition.
            order = xp.argsort(xp.concat((group_a, group_b)))
            transition = MutationTransition(
                displacement=xp.concat((displacement_a, displacement_b))[
                    order
                ],
                kernel_auxiliary=_combine_kernel_auxiliary(
                    auxiliary_a,
                    auxiliary_b,
                    group_a.shape[0],
                    group_b.shape[0],
                    order,
                ),
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
            mutation_state = self.metrics.update(
                mutation_state,
                current,
                accepted_population,
                context,
                transition=transition,
            )
            if self._store_chains:
                chain.append(current)

        new_particles, _ = self._inverse_transform(
            current,
            transform,
            context,
            auxiliary=transform_auxiliary,
        )
        acceptance = accepted_total / (n_particles * n_steps)
        stored_chain = None
        if self._store_chains:
            stored_chain = self._inverse_chain(
                xp.stack(chain, axis=0),
                transform,
                context,
                auxiliary=transform_auxiliary,
                like=state.particles,
            )
        context = context.add_evaluation_counts(counts).with_metadata(
            metropolis_hastings_acceptance=acceptance
        )
        logger.debug(
            "RedBlueMetropolisHastingsMutation completed: acceptance=%.3f "
            "after %d sweeps.",
            acceptance,
            n_steps,
        )
        output = {
            "acceptance_by_step": tuple(acceptance_by_step),
            "acceptance_by_half": tuple(acceptance_by_half),
            "steps": n_steps,
            "chain": stored_chain,
            "metrics": None,
        }
        return (
            state.replace_particles(
                new_particles,
                evaluation=current_evaluation,
            ),
            context,
            output,
            mutation_state,
        )


class DifferentialEvolutionMutation(RedBlueMetropolisHastingsMutation):
    """Rejuvenate particles with red-blue differential-evolution proposals.

    Parameters
    ----------
    log_prob_fn
        Function returning the target log probability for each particle, or a
        :class:`~sequince.target.Target` instance.
    n_steps
        Number of complete red-blue sweeps to perform.
    gamma
        Base differential-evolution scale. If ``None``, use
        ``2.38 / sqrt(2 * n_dimensions)``.
    gamma_one_probability
        Probability, independently for each proposal, of replacing the
        ordinary scale with exactly one.
    gamma_randomization
        Scale strategy. ``"none"`` uses the fixed base scale; ``"bilby"``
        multiplies it by an independent ``Gamma(shape=4, scale=0.25)`` draw.
    scale_multipliers : tuple of float
        Positive factors multiplying the ordinary DE scale independently per
        particle. Explicit unit-scale moves override these factors.
    scale_weights : tuple of float or None
        Non-negative mixture weights (normalized internally); None is uniform.
        The kernel is fixed for each call and scale draws are state-independent.
    transform
        Optional coordinate transform applied before mutation.
    transform_refit
        Transform refitting policy: ``"always"``, ``"per-context"``,
        ``"once"``, or a callable receiving the particles and step context.
    store_chains
        Whether to include the particle state after every complete red-blue
        sweep in the step output.
    """

    def __init__(
        self,
        log_prob_fn: Callable[..., Array | float] | "Target",
        *,
        n_steps: int = 1,
        gamma: float | None = None,
        gamma_one_probability: float = 0.1,
        gamma_randomization: GammaRandomization = "none",
        scale_multipliers: tuple[float, ...] = (1.0,),
        scale_weights: tuple[float, ...] | None = None,
        transform: ParticleTransform | None = None,
        transform_refit: TransformRefit = "always",
        store_chains: bool = False,
        metrics: tuple[MutationMetric, ...] | list[MutationMetric] = (),
    ) -> None:
        kernel = DifferentialEvolutionKernel(
            gamma=gamma,
            gamma_one_probability=gamma_one_probability,
            gamma_randomization=gamma_randomization,
            scale_multipliers=scale_multipliers,
            scale_weights=scale_weights,
        )
        super().__init__(
            log_prob_fn,
            kernel,
            n_steps=n_steps,
            transform=transform,
            transform_refit=transform_refit,
            store_chains=store_chains,
            metrics=metrics,
        )
