from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Callable, TypeGuard

import array_api_extra as xpx
import optree
from acai import cond, fori_loop
from array_api_compat import device as get_device
from orng import RandomGenerator

from ...array_api import asarray, namespace, to_numpy
from ...target import EvaluationCounts
from ...transforms import ParticleTransform, select_transform_auxiliary
from ...typing import Array
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
from .metrics.base import with_metrics

if TYPE_CHECKING:
    from ...smc import SMCConfig
    from ...state import ParticleState
    from ...target import Target, TargetEvaluation

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
            self._kernel: MutationKernel | None = kernel
            self._kernel_factory: MutationKernelFactory | None = None
        elif callable(kernel):
            self._kernel = None
            self._kernel_factory = kernel
        else:
            raise TypeError(
                "kernel must implement the mutation-kernel interface or be "
                "a callable returning one."
            )

    @staticmethod
    def _is_kernel(kernel: object) -> TypeGuard[MutationKernel]:
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
            transition = (
                MutationTransition(updated - current, proposal.auxiliary)
                if self.metrics.needs_transition
                else None
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
    directly. DE scale diagnostics are opt-in: configure DEScaleMetrics with
    the kernel's scale_multipliers in metrics, or use AdaptDEScaleWeights,
    which configures the required metric once.
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

    def initialize_functional_metadata(
        self, state: ParticleState, metadata: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        """Initialize acceptance metadata before tracing red-blue updates."""
        xp = namespace(state.particles)
        return {
            **metadata,
            "metropolis_hastings_acceptance": xp.asarray(
                0.0, dtype=state.particles.dtype
            ),
        }

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
        Array,
        Array,
        Any,
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

        accepted_count = xp.astype(xp.sum(accepted), current.dtype)
        return (
            current,
            current_log_prob,
            current_evaluation,
            counts,
            accepted_count,
            accepted,
            proposal.auxiliary,
        )

    def _sweep(
        self,
        current: Array,
        current_log_prob: Array,
        current_evaluation: TargetEvaluation | None,
        mutation_state: MutationStepState,
        *,
        kernel: ComplementaryEnsembleKernel,
        transform: ParticleTransform | None,
        transform_auxiliary: Any,
        context: StepContext,
        rng: RandomGenerator,
    ) -> tuple[
        Array,
        Array,
        TargetEvaluation | None,
        MutationStepState,
        EvaluationCounts,
        Array,
        Array,
    ]:
        """Execute one complete sweep, shared by eager and dynamic execution."""
        xp = namespace(current)
        n_particles = current.shape[0]
        previous = current
        accepted_population = xp.zeros_like(current_log_prob, dtype=xp.bool)
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
        counts = counts_a
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
        transition = None
        if self.metrics.needs_transition:
            # Restore particle-aligned auxiliary to population order.
            order = xp.argsort(xp.concat((group_a, group_b)))
            transition = MutationTransition(
                displacement=current - previous,
                kernel_auxiliary=optree.tree_map(
                    lambda a, b: namespace(a, b).concat((a, b), axis=0)[order],
                    auxiliary_a,
                    auxiliary_b,
                    namespace=OPTREE_NAMESPACE,
                ),
            )
        accepted_sweep = accepted_a + accepted_b
        mutation_state = self.metrics.update(
            mutation_state,
            current,
            accepted_population,
            context,
            transition=transition,
        )
        return (
            current,
            current_log_prob,
            current_evaluation,
            mutation_state,
            counts,
            accepted_sweep,
            xp.stack(
                (accepted_a / group_a.shape[0], accepted_b / group_b.shape[0])
            ),
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

    def _prepare_block(
        self,
        state: ParticleState,
        context: StepContext,
        rng: RandomGenerator,
        mutation_state: MutationStepState | None,
    ) -> tuple[
        Array,
        ParticleTransform | None,
        Any,
        StepContext,
        MutationStepState,
        Array,
        TargetEvaluation | None,
        EvaluationCounts,
    ]:
        """Prepare coordinates, metrics and starting density for a segment."""
        transform, context = self._prepare_transform(
            state.particles, context, rng=rng
        )
        current, auxiliary = self._forward_transform(
            state.particles, transform, context, rng=rng
        )
        current = asarray(current, like=state.particles)
        metrics = self._prepare_block_metrics(
            current, state, context, mutation_state
        )
        if current.shape[0] < 2:
            raise ValueError(
                "A red-blue update requires at least two particles."
            )
        log_prob, evaluation, counts = self._evaluate_log_prob(
            current, transform, context, state.evaluation, auxiliary=auxiliary
        )
        return (
            current,
            transform,
            auxiliary,
            context,
            metrics,
            log_prob,
            evaluation,
            counts,
        )

    def _prepare_block_metrics(
        self,
        current: Array,
        state: ParticleState,
        context: StepContext,
        mutation_state: MutationStepState | None,
    ) -> MutationStepState:
        """Start metrics or continue the enclosing block in current coordinates."""
        return (
            self.metrics.initialize(current, state.log_weights, context)
            if mutation_state is None
            else self.metrics.prepare(mutation_state, current, context)
        )

    def _finish_block(
        self,
        state: ParticleState,
        context: StepContext,
        current: Array,
        evaluation: TargetEvaluation | None,
        counts: EvaluationCounts,
        acceptance: Array | float,
        transform: ParticleTransform | None,
        auxiliary: Any,
    ) -> tuple[ParticleState, StepContext]:
        """Return to physical coordinates and record completed segment work."""
        particles, _ = self._inverse_transform(
            current, transform, context, auxiliary=auxiliary
        )
        context = context.add_evaluation_counts(counts).with_metadata(
            metropolis_hastings_acceptance=acceptance
        )
        return state.replace_particles(
            particles, evaluation=evaluation
        ), context

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
            mutation_state = self._prepare_block_metrics(
                state.particles, state, context, mutation_state
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
        (
            current,
            transform,
            transform_auxiliary,
            context,
            mutation_state,
            current_log_prob,
            current_evaluation,
            counts,
        ) = self._prepare_block(state, context, rng, mutation_state)
        n_particles = current.shape[0]

        accepted_total = 0
        acceptance_by_step: list[float] = []
        acceptance_by_half: list[tuple[float, float]] = []
        chain: list[Array] = []
        for _ in range(n_steps):
            (
                current,
                current_log_prob,
                current_evaluation,
                mutation_state,
                sweep_counts,
                accepted_sweep,
                halves,
            ) = self._sweep(
                current,
                current_log_prob,
                current_evaluation,
                mutation_state,
                kernel=kernel,
                transform=transform,
                transform_auxiliary=transform_auxiliary,
                context=context,
                rng=rng,
            )
            counts = counts + sweep_counts
            accepted_total += accepted_sweep
            acceptance_by_step.append(accepted_sweep / n_particles)
            acceptance_by_half.append((halves[0], halves[1]))
            if self._store_chains:
                chain.append(current)

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
        state, context = self._finish_block(
            state,
            context,
            current,
            current_evaluation,
            counts,
            acceptance,
            transform,
            transform_auxiliary,
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
            state,
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
    noise_scale : float or tuple of float
        Independent Gaussian noise standard deviation in mutation coordinates,
        either shared or one value per dimension. Must be finite and
        non-negative. Default is ``1e-5``; zero disables noise and its RNG draws.
        With a transform, specify the scale in transformed coordinates.
    transform
        Optional coordinate transform applied before mutation.
    transform_refit
        Transform refitting policy: ``"always"``, ``"per-context"``,
        ``"once"``, or a callable receiving the particles and step context.
    store_chains
        Whether to include the particle state after every complete red-blue
        sweep in the step output.
    metrics : sequence of MutationMetric
        Optional diagnostics. For per-scale DE totals, include
        DEScaleMetrics(scale_multipliers). AdaptDEScaleWeights configures
        these diagnostics automatically.
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
        noise_scale: float | tuple[float, ...] = 1e-5,
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
            noise_scale=noise_scale,
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

    def _validate_dynamic_budget(
        self, state: ParticleState, *, max_steps: int
    ) -> None:
        """Validate static dynamic-execution requirements before compilation."""
        validate_n_steps(max_steps)
        if state.particles.ndim != 2 or state.particles.shape[0] < 4:
            raise ValueError(
                "Functional DE requires a 2D population of at least four particles."
            )
        if callable(self._transform_refit):
            raise NotImplementedError(
                "Callable transform_refit is not functional."
            )

    def validate_functional_configuration(
        self,
        config: SMCConfig,
        state: ParticleState,
        *,
        max_steps: int | None = None,
    ) -> None:
        """Validate the effective capacity supplied by an enclosing adapter."""
        self._validate_dynamic_budget(
            state, max_steps=self.n_steps if max_steps is None else max_steps
        )

    def apply_functional(
        self, state: ParticleState, context: StepContext, random_state: Any
    ) -> tuple[ParticleState, StepContext, Any, Any]:
        state, context, random_state, output, metric_state = (
            self.apply_dynamic_budget(
                state,
                context,
                random_state,
                n_steps=namespace(state.particles).asarray(
                    self.n_steps, dtype="int32"
                ),
                max_steps=self.n_steps,
            )
        )
        return (
            state,
            context,
            random_state,
            with_metrics(output, self.metrics.result(metric_state, context)),
        )

    def apply_dynamic_budget(
        self,
        state: ParticleState,
        context: StepContext,
        random_state: Any,
        *,
        n_steps: Array,
        max_steps: int,
        mutation_state: MutationStepState | None = None,
    ) -> tuple[
        ParticleState, StepContext, Any, dict[str, Any], MutationStepState
    ]:
        """Run complete sweeps with fixed-capacity diagnostics under JAX.

        Parameters
        ----------
        state : ParticleState
            Initialized particles with shape (n_particles, n_dimensions),
            floating dtype, and any shape-complete target cache.
        context : StepContext
            Initialized acceptance and optional functional transform metadata.
        random_state : object
            Backend-native state returned by ``orng.RandomGenerator.state()``,
            threaded explicitly through executed sweeps. For JAX this is a
            PRNG key array. Pass the state rather than the ORNG generator
            object; ORNG wrappers are reconstructed internally for draws.
        n_steps : array, shape ()
            Integer budget in [0, max_steps]. Bounds are a caller precondition;
            compiled runtime values are not checked.
        max_steps : int
            Static non-negative capacity bounding the executed sweep count.
        mutation_state : MutationStepState or None
            Continuing block metrics, or None to initialize a new block.

        Returns
        -------
        state : ParticleState
            Updated original-coordinate particles and target cache.
        context : StepContext
            Updated transform, acceptance metadata and evaluation counters.
        random_state : object
            Advanced backend-native ORNG state, unchanged when the budget is
            zero. For JAX this is a PRNG key array.
        output : dict
            ``steps`` is the scalar executed count. Only this many leading rows
            of ``acceptance_by_step`` (max_steps,) and ``acceptance_by_half``
            (max_steps, 2) are valid. Optional ``chain`` has shape (max_steps,
            n_particles, n_dimensions) in original coordinates. Unused rows are
            zero. No chain buffer is allocated when storage is disabled.
        mutation_state : MutationStepState
            Accumulated metrics, independent of padded histories, for the block
            owner to finalize once. ``output["metrics"]`` remains None.

        Notes
        -----
        Call functional metadata and state/context initialization before JIT.
        One-time initialization may fit transforms and populate target caches.
        Block execution with a zero budget performs no transform preparation,
        target evaluation or RNG draw. Transforms remain fixed per segment.
        """
        self._validate_dynamic_budget(state, max_steps=max_steps)
        xp = namespace(state.particles)
        if n_steps.shape != () or not xp.isdtype(n_steps.dtype, "integral"):
            raise TypeError("n_steps must be a scalar integer array.")
        zeros = xp.zeros((max_steps,), dtype=state.particles.dtype)
        halves = xp.zeros((max_steps, 2), dtype=state.particles.dtype)
        chain = (
            xp.zeros(
                (max_steps, *state.particles.shape),
                dtype=state.particles.dtype,
            )
            if self.store_chains
            else None
        )

        def execute(_: None) -> tuple[Any, ...]:
            rng = RandomGenerator.from_xp(xp, generator=random_state)
            (
                current,
                transform,
                auxiliary,
                prepared,
                metrics,
                log_prob,
                evaluation,
                counts,
            ) = self._prepare_block(state, context, rng, mutation_state)
            accepted_before = metrics.n_accepted

            def sweep(index: Array, carry: tuple[Any, ...]) -> tuple[Any, ...]:
                (
                    current,
                    log_prob,
                    evaluation,
                    metrics,
                    counts,
                    key,
                    history,
                    half_history,
                    chains,
                ) = carry
                rng = RandomGenerator.from_xp(xp, generator=key)
                (
                    current,
                    log_prob,
                    evaluation,
                    metrics,
                    work,
                    acceptance,
                    half,
                ) = self._sweep(
                    current,
                    log_prob,
                    evaluation,
                    metrics,
                    kernel=self.kernel,
                    transform=transform,
                    transform_auxiliary=auxiliary,
                    context=prepared,
                    rng=rng,
                )
                if self.store_chains:
                    original, _ = self._inverse_transform(
                        current, transform, prepared, auxiliary=auxiliary
                    )
                    chains = chains.at[index].set(original)
                return (
                    current,
                    log_prob,
                    evaluation,
                    metrics,
                    counts + work,
                    rng.state(),
                    history.at[index].set(acceptance / current.shape[0]),
                    half_history.at[index].set(half),
                    chains,
                )

            (
                current,
                _,
                evaluation,
                metrics,
                counts,
                key,
                history,
                half_history,
                chains,
            ) = fori_loop(
                0,
                n_steps,
                sweep,
                (
                    current,
                    log_prob,
                    evaluation,
                    metrics,
                    counts,
                    rng.state(),
                    zeros,
                    halves,
                    chain,
                ),
                xp=xp,
            )
            acceptance = xp.astype(
                metrics.n_accepted - accepted_before, current.dtype
            ) / (current.shape[0] * n_steps)
            updated, prepared = self._finish_block(
                state,
                prepared,
                current,
                evaluation,
                counts,
                acceptance,
                transform,
                auxiliary,
            )
            return (
                updated,
                prepared,
                key,
                history,
                half_history,
                chains,
                metrics,
            )

        def skip(_: None) -> tuple[Any, ...]:
            return (
                state,
                context,
                random_state,
                zeros,
                halves,
                chain,
                self._prepare_block_metrics(
                    state.particles, state, context, mutation_state
                ),
            )

        result = (
            skip(None)
            if max_steps == 0
            else cond(
                n_steps > 0,
                execute,
                skip,
                None,
                xp=xp,
            )
        )
        state, context, key, history, half_history, chain, metrics = result
        return (
            state,
            context,
            key,
            {
                "acceptance_by_step": history,
                "acceptance_by_half": half_history,
                "steps": n_steps,
                "chain": chain,
                "metrics": None,
            },
            metrics,
        )
