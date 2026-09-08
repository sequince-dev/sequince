from __future__ import annotations

import logging
import math
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from array_api_compat import device as get_device

from .._typing import Array
from ..array_api import asarray, namespace
from ..target import EvaluationCounts, Target
from ..utils.callables import call_with_optional_context
from .base import SMCStep, StepContext
from .mutation import (
    MutationKernel,
    MutationMetricsOutput,
    evaluate_log_transition_ratio,
)

if TYPE_CHECKING:
    from orng import RandomGenerator

    from ..state import ParticleState

logger = logging.getLogger(__name__)


MutationKernelFactory = Callable[[Array, StepContext], MutationKernel]


class WasteFreeResampleMove(SMCStep):
    """Combine resampling and mutation using waste-free recycling of proposals.

    Completed ``MutationMetricsOutput`` report proposal rounds and attempted
    proposals. Since recycling integrates over accept/reject decisions,
    ``n_accepted`` is the sum of acceptance probabilities (an expected count),
    and ``acceptance`` is their mean. This supports ``AdaptiveGaussianProposal``
    without introducing additional random draws.

    Examples
    --------
    >>> from sequince.posterior import TemperedPosterior
    >>> from sequince.steps.mutation import GaussianRandomWalk
    >>> from sequince.steps import WasteFreeResampleMove
    >>> tempered_post = TemperedPosterior(log_likelihood, log_prior)
    >>> kernel = GaussianRandomWalk(scale=0.5)
    >>> step = WasteFreeResampleMove(
    ...     log_prob_fn=tempered_post,
    ...     kernel=kernel,
    ...     resample_ancestors=True,
    ...     proposals_per_particle=3,
    ... )
    >>> smc = SequentialMonteCarlo(steps=[..., step, ...])

    References
    ----------
    .. [1] Dau, H., & Chopin, N. (2021). Waste-free Sequential Monte Carlo. arXiv:2011.02328


    Parameters
    ----------
    log_prob_fn:
        Callable returning log probabilities of the tempered target, or an
        object exposing a ``log_prob`` method (e.g. ``Target``).
    kernel:
        Mutation kernel supplying forward and backward log densities.
    resample_ancestors:
        When ``True`` the step draws ancestors proportional to the current
        weights before proposing new particles (default). Set to ``False`` to
        operate directly on the existing particle set.
    proposals_per_particle:
        Number of proposals generated for each ancestor within a single
        application of the step. Values greater than one produce multiple
        proposal rounds whose acceptance contributions are recycled.
    """

    def __init__(
        self,
        log_prob_fn: Callable[..., Any] | Target,
        kernel: MutationKernel | MutationKernelFactory,
        *,
        resample_ancestors: bool = True,
        proposals_per_particle: int = 1,
    ):
        if isinstance(log_prob_fn, Target):
            self._target: Target | None = log_prob_fn
            self._log_prob_fn: Callable[..., Any] | None = None
        else:
            self._target = None
            self._log_prob_fn = log_prob_fn
        has_correction = hasattr(kernel, "log_transition_ratio") or hasattr(
            kernel, "log_density"
        )
        if hasattr(kernel, "sample") and has_correction:
            self._kernel: MutationKernel | None = kernel  # type: ignore[assignment]
            self._kernel_factory: MutationKernelFactory | None = None
        elif callable(kernel):
            self._kernel = None
            self._kernel_factory = kernel  # type: ignore[assignment]
        else:
            raise TypeError(
                "kernel must implement MutationKernel or be a callable returning one."
            )
        self._resample = resample_ancestors
        self._n_proposals = int(proposals_per_particle)
        if self._n_proposals < 1:
            raise ValueError("proposals_per_particle must be >= 1.")

    def apply(
        self,
        state: "ParticleState",
        context: StepContext,
        rng: "RandomGenerator",
    ) -> tuple["ParticleState", StepContext, dict[str, Any]]:
        n = state.log_weights.shape[0]
        if n == 0:
            raise ValueError(
                "WasteFreeResampleMove requires at least one particle."
            )

        xp = namespace(state.particles, state.log_weights)
        particles = asarray(state.particles, xp=xp)
        squeeze_last = False
        if particles.ndim == 1:
            particles = xp.reshape(particles, (n, 1))
            squeeze_last = True

        if self._resample:
            weights = state.weights
            ancestor_indices = rng.choice(n, size=n, replace=True, p=weights)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "WasteFreeResampleMove drew ancestors with effective ESS %.4f.",
                    state.effective_sample_size(),
                )
        else:
            ancestor_indices = xp.arange(n, device=get_device(particles))

        parent_particles = particles[ancestor_indices]
        parent_evaluation = (
            None
            if state.evaluation is None
            else state.evaluation.take(ancestor_indices)
        )

        log_prob_parents, parent_evaluation, counts = self._evaluate_log_prob(
            parent_particles,
            context,
            parent_evaluation,
        )
        kernel = self._resolve_kernel(parent_particles, context)

        log_base_weight = -math.log(n)
        remaining_log_mass = xp.full(
            (n,),
            log_base_weight,
            dtype=log_prob_parents.dtype,
            device=get_device(log_prob_parents),
        )

        candidate_particles = [parent_particles]
        candidate_evaluations = (
            [parent_evaluation] if parent_evaluation is not None else []
        )
        proposal_log_weights: list[Array] = []
        acceptance_history: list[Array] = []

        for _ in range(self._n_proposals):
            proposals = kernel.sample(parent_particles, rng)
            proposals = asarray(proposals, xp=xp, like=particles)
            if proposals.shape != parent_particles.shape:
                raise ValueError(
                    "Mutation kernel must return the same shape as the input particles."
                )

            (
                log_prob_proposals,
                proposal_evaluation,
                proposal_counts_delta,
            ) = self._evaluate_log_prob(proposals, context)
            counts = counts + proposal_counts_delta
            if proposal_evaluation is not None:
                candidate_evaluations.append(proposal_evaluation)
            transition_ratio, transition_counts = (
                evaluate_log_transition_ratio(
                    kernel,
                    parent_particles,
                    proposals,
                    return_counts=True,
                )
            )
            counts = counts + transition_counts
            transition_ratio = asarray(
                transition_ratio,
                xp=xp,
                like=particles,
            )

            log_r = log_prob_proposals - log_prob_parents + transition_ratio
            log_r = xp.clip(log_r, -1e12, 1e12)
            zeros = xp.full(
                log_r.shape,
                0.0,
                dtype=log_r.dtype,
                device=get_device(log_r),
            )
            log_accept = xp.minimum(log_r, zeros)
            accept_prob = xp.exp(log_accept)
            acceptance_history.append(accept_prob)

            proposal_log_mass = remaining_log_mass + xp.log(accept_prob)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                remaining_log_mass = remaining_log_mass + xp.log1p(
                    -accept_prob
                )

            proposal_log_weights.append(proposal_log_mass)
            candidate_particles.append(proposals)

        candidate_log_weights = [remaining_log_mass] + proposal_log_weights

        combined_particles = xp.concat(tuple(candidate_particles), axis=0)
        combined_log_weights = xp.concat(
            tuple(xp.reshape(log_w, (-1,)) for log_w in candidate_log_weights),
            axis=0,
        )

        max_log = xp.max(combined_log_weights)
        weights = xp.exp(combined_log_weights - max_log)
        weight_sum = xp.sum(weights)
        weight_sum_scalar = float(weight_sum)
        if not math.isfinite(weight_sum_scalar) or weight_sum_scalar <= 0.0:
            raise RuntimeError(
                "WasteFreeResampleMove produced invalid candidate weights."
            )
        weights = xp.reshape(weights / weight_sum, (-1,))

        selected = rng.choice(
            combined_particles.shape[0],
            size=n,
            replace=True,
            p=weights,
        )

        new_particles = combined_particles[selected]
        new_evaluation = None
        if candidate_evaluations:
            from ..target import TargetEvaluation

            new_evaluation = TargetEvaluation.concatenate(
                candidate_evaluations
            ).take(selected)
        if squeeze_last:
            new_particles = new_particles[:, 0]

        selected_ancestor_indices = ancestor_indices[selected % n]
        metrics = MutationMetricsOutput(
            n_steps=xp.asarray(self._n_proposals),
            n_attempted=xp.asarray(n * self._n_proposals),
            n_accepted=xp.sum(xp.concat(tuple(acceptance_history), axis=0)),
        )
        mean_acceptance = float(metrics.acceptance)
        acceptance_by_round = [
            float(xp.mean(values)) for values in acceptance_history
        ]
        selected_proposals = int(xp.sum(selected >= n))
        selected_parents = int(xp.sum(selected < n))
        proposal_counts = []
        for round_index in range(self._n_proposals):
            low = n * (round_index + 1)
            high = n * (round_index + 2)
            count = int(xp.sum((selected >= low) & (selected < high)))
            proposal_counts.append(count)

        context = context.add_evaluation_counts(counts).with_metadata(
            waste_free_acceptance=mean_acceptance,
        )

        log_weight_template = asarray(state.log_weights, xp=xp)
        log_weights = xp.full(
            (n,),
            -math.log(n),
            dtype=log_weight_template.dtype,
            device=get_device(log_weight_template),
        )
        output = {
            "metrics": metrics,
            "ancestors": selected_ancestor_indices,
            "acceptance": mean_acceptance,
            "acceptance_by_round": tuple(acceptance_by_round),
            "selected_proposals": selected_proposals,
            "selected_parents": selected_parents,
            "selected_proposals_by_round": tuple(proposal_counts),
            "proposal_repeats": self._n_proposals,
            "total_proposals": n * self._n_proposals,
        }
        return (
            state.replace_particles(
                new_particles,
                evaluation=new_evaluation,
                log_weights=log_weights,
                initial_ancestor_indices=state.initial_ancestor_indices[
                    selected_ancestor_indices
                ],
            ),
            context,
            output,
        )

    def _evaluate_log_prob(
        self,
        particles: Array,
        context: StepContext,
        evaluation=None,
    ):
        xp = namespace(particles)
        log_prob_count = EvaluationCounts(
            log_target_evaluations=xp.asarray(particles.shape[0])
        )
        if self._target is None:
            if self._log_prob_fn is None:  # pragma: no cover
                raise RuntimeError(
                    "Mutation log probability is not configured."
                )
            values = call_with_optional_context(
                self._log_prob_fn,
                (particles,),
                context,
            )
            counts = log_prob_count
        else:
            evaluation, counts = self._target.prepare_evaluation(
                particles,
                context,
                evaluation=evaluation,
                return_counts=True,
            )
            values = self._target.log_prob(
                particles,
                context,
                evaluation=evaluation,
            )
        arr = asarray(values, xp=xp, like=particles)
        if arr.ndim == 0:
            arr = xp.broadcast_to(arr, (particles.shape[0],))
        elif arr.ndim > 1:
            squeezed = xp.squeeze(arr)
            if squeezed.ndim != 1:
                raise ValueError(
                    "log_prob_fn must return a scalar or 1-D array per particle."
                )
            arr = squeezed
        if arr.shape[0] != particles.shape[0]:
            raise ValueError("log_prob_fn must return one value per particle.")
        return arr, evaluation, counts

    def _resolve_kernel(
        self,
        parent_particles: Array,
        context: StepContext,
    ) -> MutationKernel:
        if self._kernel_factory is not None:
            kernel = self._kernel_factory(parent_particles, context)
        else:
            if self._kernel is None:
                raise RuntimeError(
                    "Kernel factory did not return a valid MutationKernel."
                )
            kernel = self._kernel
        has_correction = hasattr(kernel, "log_transition_ratio") or hasattr(
            kernel, "log_density"
        )
        if not hasattr(kernel, "sample") or not has_correction:
            raise TypeError(
                "Resolved kernel must implement sample() and either "
                "log_transition_ratio() or log_density()."
            )
        return kernel
