from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from .._array_api import to_numpy
from .._typing import Array
from ..mutation_kernels import MutationKernel
from ..utils import call_with_optional_context
from .base import SMCStep, StepContext

if TYPE_CHECKING:
    from orng import ArrayRNG

    from ..state import ParticleState

logger = logging.getLogger(__name__)


MutationKernelFactory = Callable[[Array, StepContext], MutationKernel]


class WasteFreeResampleMove(SMCStep):
    """Combine resampling and mutation using waste-free recycling of proposals.

    Examples
    --------
    >>> from sequince.posterior import TemperedPosterior
    >>> from sequince.mutation_kernels import GaussianRandomWalk
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
        object exposing a ``log_prob`` method (e.g. ``TemperedPosterior``).
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
        log_prob_fn: Callable[..., Any],
        kernel: MutationKernel | MutationKernelFactory,
        *,
        resample_ancestors: bool = True,
        proposals_per_particle: int = 1,
    ):
        if hasattr(log_prob_fn, "log_prob"):
            posterior = log_prob_fn

            def _wrapped_log_prob(
                particles: Array,
                context: StepContext,
            ) -> Any:
                log_proposal_override = context.metadata.get(
                    "log_proposal_override"
                )
                return posterior.log_prob(
                    particles,
                    context,
                    log_proposal_override=log_proposal_override,
                )

            self._log_prob_fn = _wrapped_log_prob
        else:
            self._log_prob_fn = log_prob_fn
        if hasattr(kernel, "sample") and hasattr(kernel, "log_density"):
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
        rng: "ArrayRNG",
    ) -> "ParticleState":
        n = state.log_weights.shape[0]
        if n == 0:
            raise ValueError(
                "WasteFreeResampleMove requires at least one particle."
            )

        particles = np.asarray(state.particles, dtype=float)
        squeeze_last = False
        if particles.ndim == 1:
            particles = particles[:, np.newaxis]
            squeeze_last = True

        if self._resample:
            weights = state.weights
            ancestor_indices = to_numpy(
                rng.choice(n, size=n, replace=True, p=weights)
            ).astype(int)
            logger.debug(
                "WasteFreeResampleMove drew ancestors with effective ESS %.4f.",
                float(state.effective_sample_size()),
            )
        else:
            ancestor_indices = np.arange(n, dtype=int)

        parent_particles = particles[ancestor_indices]

        log_prob_parents = self._evaluate_log_prob(parent_particles, context)
        kernel = self._resolve_kernel(parent_particles, context)

        log_base_weight = -np.log(n)
        remaining_log_mass = np.full(n, log_base_weight, dtype=float)

        candidate_particles = [parent_particles]
        proposal_log_weights: list[Array] = []
        acceptance_history: list[Array] = []

        for _ in range(self._n_proposals):
            proposals = kernel.sample(parent_particles, rng)
            if proposals.shape != parent_particles.shape:
                raise ValueError(
                    "Mutation kernel must return the same shape as the input particles."
                )

            log_prob_proposals = self._evaluate_log_prob(proposals, context)
            log_forward = kernel.log_density(parent_particles, proposals)
            log_backward = kernel.log_density(proposals, parent_particles)

            log_r = (
                log_prob_proposals
                + log_backward
                - log_prob_parents
                - log_forward
            )
            np.clip(log_r, a_min=-1e12, a_max=1e12, out=log_r)
            log_accept = np.minimum(log_r, 0.0)
            accept_prob = np.exp(log_accept)
            acceptance_history.append(accept_prob)

            with np.errstate(divide="ignore"):
                proposal_log_mass = remaining_log_mass + np.log(accept_prob)
                remaining_log_mass = remaining_log_mass + np.log1p(
                    -accept_prob
                )

            proposal_log_weights.append(proposal_log_mass)
            candidate_particles.append(proposals)

        candidate_log_weights = [remaining_log_mass] + proposal_log_weights

        combined_particles = np.concatenate(candidate_particles, axis=0)
        combined_log_weights = np.concatenate(
            [log_w.reshape(-1) for log_w in candidate_log_weights],
            axis=0,
        )

        max_log = np.max(combined_log_weights)
        weights = np.exp(combined_log_weights - max_log)
        weight_sum = np.sum(weights)
        if not np.isfinite(weight_sum) or weight_sum <= 0.0:
            raise RuntimeError(
                "WasteFreeResampleMove produced invalid candidate weights."
            )
        weights = (weights / weight_sum).reshape(-1)

        selected = to_numpy(
            rng.choice(
                combined_particles.shape[0],
                size=n,
                replace=True,
                p=weights,
            )
        ).astype(int)

        new_particles = combined_particles[selected]
        if squeeze_last:
            new_particles = new_particles[:, 0]

        metadata = dict(state.metadata)
        metadata["ancestors"] = ancestor_indices
        mean_acceptance = (
            float(np.mean(np.concatenate(acceptance_history)))
            if acceptance_history
            else float("nan")
        )
        metadata["waste_free_acceptance"] = mean_acceptance
        acceptance_by_round = [
            float(np.mean(values)) for values in acceptance_history
        ]
        metadata["waste_free_acceptance_by_round"] = acceptance_by_round
        metadata["waste_free_selected_proposals"] = int(np.sum(selected >= n))
        metadata["waste_free_selected_parents"] = int(np.sum(selected < n))
        proposal_counts = []
        for round_index in range(self._n_proposals):
            low = n * (round_index + 1)
            high = n * (round_index + 2)
            count = int(np.sum((selected >= low) & (selected < high)))
            proposal_counts.append(count)
        metadata["waste_free_selected_proposals_by_round"] = proposal_counts
        metadata["waste_free_proposal_repeats"] = self._n_proposals

        context.metadata["waste_free_acceptance"] = metadata[
            "waste_free_acceptance"
        ]
        context.metadata["waste_free_acceptance_by_round"] = (
            acceptance_by_round
        )
        context.metadata["waste_free_selected_proposals"] = metadata[
            "waste_free_selected_proposals"
        ]
        context.metadata["waste_free_total_proposals"] = n * self._n_proposals

        log_weights = np.full(n, -np.log(n), dtype=float)
        return state.with_updates(
            particles=new_particles,
            log_weights=log_weights,
            metadata=metadata,
        )

    def _evaluate_log_prob(
        self,
        particles: Array,
        context: StepContext,
    ) -> Array:
        values = call_with_optional_context(
            self._log_prob_fn,
            (particles,),
            context,
        )
        arr = np.asarray(values, dtype=float)
        if arr.ndim == 0:
            arr = np.full(particles.shape[0], float(arr))
        elif arr.ndim > 1:
            squeezed = np.squeeze(arr)
            if squeezed.ndim != 1:
                raise ValueError(
                    "log_prob_fn must return a scalar or 1-D array per particle."
                )
            arr = squeezed
        if arr.shape[0] != particles.shape[0]:
            raise ValueError("log_prob_fn must return one value per particle.")
        return arr

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
        if not hasattr(kernel, "sample") or not hasattr(kernel, "log_density"):
            raise TypeError(
                "Resolved kernel must implement sample() and log_density()."
            )
        return kernel
