from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ..._typing import Array
from ...array_api import asarray, namespace
from ...target import EvaluationCounts
from ...transforms import ParticleTransform
from ...utils.pytrees import pytree_dataclass
from ..base import validate_n_steps
from ..context import StepContext
from ..mutation_diagnostics import (
    empty_mutation_diagnostics,
    finish_mutation_diagnostics,
    prepare_mutation_diagnostics,
)
from .base import (
    MutationStep,
    TransformRefit,
)

if TYPE_CHECKING:
    from orng import RandomGenerator

    from ...state import ParticleState
    from ...target import Target

logger = logging.getLogger(__name__)


@pytree_dataclass(
    data_fields=("acceptance", "chain", "diagnostics"),
    meta_fields=("step_fn", "steps"),
)
@dataclass(frozen=True)
class MiniPCNOutput(Mapping[str, Any]):
    """Fixed-structure MiniPCN diagnostics returned by a mutation step."""

    acceptance: Any
    step_fn: str
    steps: int
    chain: Any
    diagnostics: Any = None

    def __getitem__(self, key: str) -> Any:
        if key not in {
            "acceptance",
            "step_fn",
            "steps",
            "chain",
            "diagnostics",
        }:
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self):
        return iter(("acceptance", "step_fn", "steps", "chain", "diagnostics"))

    def __len__(self) -> int:
        return 5


class MiniPCNMutation(MutationStep):
    """Run MiniPCN MCMC moves to rejuvenate particles.

    By default uses the 'tpcn' step function and targets an acceptance rate of 0.234.
    Also sets 'verbose' to False by default.
    """

    def __init__(
        self,
        log_prob_fn: Callable[..., Array | float] | "Target",
        *,
        n_steps: int = 1,
        step_fn: str = "tpcn",
        verbose: bool = False,
        transform: ParticleTransform | None = None,
        transform_refit: TransformRefit = "always",
        store_chains: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            log_prob_fn,
            transform=transform,
            transform_refit=transform_refit,
            store_chains=store_chains,
        )
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1.")
        self._n_steps = n_steps
        self._step_fn = step_fn
        self._verbose = verbose
        self._kwargs = kwargs
        self._kwargs.setdefault("target_acceptance_rate", 0.234)
        self._kwargs.setdefault("adaptive", False)

    def apply(
        self,
        state: ParticleState,
        context: StepContext,
        rng: "RandomGenerator",
    ) -> tuple[ParticleState, StepContext, MiniPCNOutput]:
        return self.apply_with_n_steps(
            state, context, rng, n_steps=self._n_steps
        )

    def apply_with_n_steps(
        self,
        state: ParticleState,
        context: StepContext,
        rng: "RandomGenerator",
        *,
        n_steps: int,
    ) -> tuple[ParticleState, StepContext, MiniPCNOutput]:
        n_steps = validate_n_steps(n_steps)
        if n_steps == 0:
            xp = namespace(state.particles)
            diagnostics, context = empty_mutation_diagnostics(
                state.particles, state.log_weights, context
            )
            chain = (
                xp.empty(
                    (0, *state.particles.shape), dtype=state.particles.dtype
                )
                if self._store_chains
                else None
            )
            return (
                state,
                context,
                MiniPCNOutput(
                    acceptance=0.0,
                    step_fn=self._step_fn,
                    steps=0,
                    chain=chain,
                    diagnostics=diagnostics,
                ),
            )
        (
            sampler,
            latent_particles,
            transform,
            transform_auxiliary,
            context,
        ) = self._prepare_sampler(state, context, rng=rng)

        chain, history = sampler.sample(
            latent_particles,
            n_steps=n_steps,
            verbose=self._verbose,
            rng=rng,
        )

        acceptance = float(np.mean(history.acceptance_rate))
        return self._complete_mutation(
            state,
            context,
            latent_last=chain[-1],
            latent_chain=chain[-n_steps:] if self._store_chains else None,
            diagnostic_chain=chain[-n_steps:],
            latent_particles=latent_particles,
            transform=transform,
            transform_auxiliary=transform_auxiliary,
            acceptance=acceptance,
            n_steps=n_steps,
        )

    def apply_functional(
        self,
        state: ParticleState,
        context: StepContext,
        random_state: Any,
    ) -> tuple[ParticleState, StepContext, Any, MiniPCNOutput]:
        """Run MiniPCN while explicitly threading backend random state."""
        from orng import RandomGenerator

        transform_rng = RandomGenerator.from_xp(
            namespace(random_state), generator=random_state
        )
        (
            sampler,
            latent_particles,
            transform,
            transform_auxiliary,
            context,
        ) = self._prepare_sampler(state, context, rng=transform_rng)
        random_state = transform_rng.state()
        return_last_only = not self._store_chains
        chain, history, next_random_state = sampler.sample_functional(
            latent_particles,
            n_steps=self._n_steps,
            rng_state=random_state,
            verbose=False,
            return_last_only=return_last_only,
        )
        xp = namespace(latent_particles)
        acceptance_rate = xp.asarray(history.acceptance_rate)
        acceptance = xp.mean(acceptance_rate)
        latent_last = chain if return_last_only else chain[-1]
        latent_chain = None if return_last_only else chain[-self._n_steps :]
        updated_state, context, output = self._complete_mutation(
            state,
            context,
            latent_last=latent_last,
            latent_chain=latent_chain,
            diagnostic_chain=None,
            latent_particles=latent_particles,
            transform=transform,
            transform_auxiliary=transform_auxiliary,
            acceptance=acceptance,
            n_steps=self._n_steps,
        )
        return updated_state, context, next_random_state, output

    def _prepare_sampler(
        self,
        state: ParticleState,
        context: StepContext,
        *,
        rng: Any = None,
    ) -> tuple[Any, Array, ParticleTransform | None, Any, StepContext]:
        """Prepare the latent particles and MiniPCN sampler."""
        from minipcn import Sampler

        xp = namespace(state.particles)
        transform, context = self._prepare_transform(
            state.particles, context, rng=rng
        )
        latent_particles, transform_auxiliary = self._forward_transform(
            state.particles, transform, context, rng=rng
        )

        def log_prob_fn(position):
            backend_position = asarray(position, xp=xp, like=latent_particles)
            original, log_det = self._inverse_transform(
                backend_position,
                transform,
                context,
                auxiliary=transform_auxiliary,
            )
            result, _, _ = self._evaluate_target(original, context)
            return self._combine_with_log_det(result, log_det)

        dims = latent_particles.shape[1] if latent_particles.ndim > 1 else 1
        sampler = Sampler(
            log_prob_fn=log_prob_fn,
            step_fn=self._step_fn,
            dims=dims,
            xp=xp,
            **self._kwargs,
        )
        return (
            sampler,
            latent_particles,
            transform,
            transform_auxiliary,
            context,
        )

    def _complete_mutation(
        self,
        state: ParticleState,
        context: StepContext,
        *,
        latent_last: Array,
        latent_chain: Array | None,
        diagnostic_chain: Array | None,
        latent_particles: Array,
        transform: ParticleTransform | None,
        transform_auxiliary: Any,
        acceptance: Any,
        n_steps: int,
    ) -> tuple[ParticleState, StepContext, MiniPCNOutput]:
        """Map MiniPCN results back to particle space and build diagnostics."""
        xp = namespace(latent_particles)
        latent_last = asarray(latent_last, xp=xp, like=latent_particles)
        new_particles, _ = self._inverse_transform(
            latent_last,
            transform,
            context,
            auxiliary=transform_auxiliary,
        )
        _, final_evaluation, final_counts = self._evaluate_target(
            new_particles, context
        )
        counts = EvaluationCounts(
            log_target_evaluations=(
                final_counts.log_target_evaluations * (n_steps + 2)
            ),
            log_likelihood_evaluations=(
                final_counts.log_likelihood_evaluations * (n_steps + 2)
            ),
            log_proposal_evaluations=(
                final_counts.log_proposal_evaluations * (n_steps + 2)
            ),
            log_increment_evaluations=(
                final_counts.log_increment_evaluations * (n_steps + 2)
            ),
        )
        context = context.add_evaluation_counts(counts)

        stored_chain = None
        if latent_chain is not None:
            stored_chain = self._inverse_chain(
                latent_chain,
                transform,
                context,
                auxiliary=transform_auxiliary,
                like=latent_particles,
            )

        diagnostics = None
        if diagnostic_chain is not None:
            movement, context, owns_accumulator = prepare_mutation_diagnostics(
                latent_particles, state.log_weights, context
            )
            previous = latent_particles
            for positions in diagnostic_chain:
                position_matrix = (
                    positions[:, None] if positions.ndim == 1 else positions
                )
                previous_matrix = (
                    previous[:, None] if previous.ndim == 1 else previous
                )
                accepted = xp.any(
                    position_matrix != previous_matrix,
                    axis=tuple(range(1, position_matrix.ndim)),
                )
                movement.add_transition(positions, accepted)
                previous = positions
            diagnostics, context = finish_mutation_diagnostics(
                movement, context, owns_accumulator=owns_accumulator
            )

        logger.debug(
            "Iteration %d: MiniPCNMutation completed after %d steps.",
            context.t,
            n_steps,
        )
        output = MiniPCNOutput(
            acceptance=acceptance,
            step_fn=self._step_fn,
            steps=n_steps,
            chain=stored_chain,
            diagnostics=diagnostics,
        )
        return (
            state.replace_particles(
                new_particles,
                evaluation=final_evaluation,
            ),
            context,
            output,
        )
