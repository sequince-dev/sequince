from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ..._typing import Array
from ...array_api import asarray, namespace
from ...transforms import ParticleTransform
from ...utils import pytree_dataclass
from ..context import StepContext
from .base import (
    MutationStep,
    TransformRefit,
)

if TYPE_CHECKING:
    from orng import RandomGenerator

    from ...state import ParticleState
    from ...target import SequentialTarget

logger = logging.getLogger(__name__)


@pytree_dataclass(
    data_fields=("acceptance", "chain"),
    meta_fields=("step_fn", "steps"),
)
@dataclass(frozen=True)
class MiniPCNOutput(Mapping[str, Any]):
    """Fixed-structure MiniPCN diagnostics returned by a mutation step."""

    acceptance: Any
    step_fn: str
    steps: int
    chain: Any

    def __getitem__(self, key: str) -> Any:
        if key not in {"acceptance", "step_fn", "steps", "chain"}:
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self):
        return iter(("acceptance", "step_fn", "steps", "chain"))

    def __len__(self) -> int:
        return 4


class MiniPCNMutation(MutationStep):
    """Run MiniPCN MCMC moves to rejuvenate particles.

    By default uses the 'tpcn' step function and targets an acceptance rate of 0.234.
    Also sets 'verbose' to False by default.
    """

    def __init__(
        self,
        log_prob_fn: Callable[..., Array | float] | "SequentialTarget",
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

    def apply(
        self,
        state: ParticleState,
        context: StepContext,
        rng: "RandomGenerator",
    ) -> tuple[ParticleState, StepContext, MiniPCNOutput]:
        sampler, latent_particles, transform, context = self._prepare_sampler(
            state, context
        )

        chain, history = sampler.sample(
            latent_particles,
            n_steps=self._n_steps,
            verbose=self._verbose,
            rng=rng,
        )

        acceptance = float(np.mean(history.acceptance_rate))
        return self._complete_mutation(
            state,
            context,
            latent_last=chain[-1],
            latent_chain=chain[-self._n_steps :]
            if self._store_chains
            else None,
            latent_particles=latent_particles,
            transform=transform,
            acceptance=acceptance,
        )

    def functional_step(
        self,
        state: ParticleState,
        context: StepContext,
        random_state: Any,
    ) -> tuple[ParticleState, StepContext, Any, MiniPCNOutput]:
        """Run MiniPCN while explicitly threading backend random state."""
        sampler, latent_particles, transform, context = self._prepare_sampler(
            state, context
        )
        return_last_only = not self._store_chains
        chain, history, next_random_state = sampler.sample_functional(
            latent_particles,
            n_steps=self._n_steps,
            rng_state=random_state,
            verbose=False,
            return_last_only=return_last_only,
        )
        xp = namespace(history.acceptance_rate)
        acceptance = xp.mean(history.acceptance_rate)
        latent_last = chain if return_last_only else chain[-1]
        latent_chain = None if return_last_only else chain[-self._n_steps :]
        updated_state, context, output = self._complete_mutation(
            state,
            context,
            latent_last=latent_last,
            latent_chain=latent_chain,
            latent_particles=latent_particles,
            transform=transform,
            acceptance=acceptance,
        )
        return updated_state, context, next_random_state, output

    def _prepare_sampler(
        self,
        state: ParticleState,
        context: StepContext,
    ) -> tuple[Any, Array, ParticleTransform | None, StepContext]:
        """Prepare the latent particles and MiniPCN sampler."""
        from minipcn import Sampler

        xp = namespace(state.particles)
        transform, context = self._prepare_transform(state.particles, context)
        latent_particles = self._forward_transform(
            state.particles, transform, context
        )

        def log_prob_fn(position):
            backend_position = asarray(position, xp=xp, like=latent_particles)
            original, log_det = self._inverse_transform(
                backend_position, transform, context
            )
            result, _ = self._evaluate_target(original, context)
            return self._combine_with_log_det(result, log_det)

        dims = latent_particles.shape[1] if latent_particles.ndim > 1 else 1
        sampler = Sampler(
            log_prob_fn=log_prob_fn,
            step_fn=self._step_fn,
            dims=dims,
            xp=xp,
            **self._kwargs,
        )
        return sampler, latent_particles, transform, context

    def _complete_mutation(
        self,
        state: ParticleState,
        context: StepContext,
        *,
        latent_last: Array,
        latent_chain: Array | None,
        latent_particles: Array,
        transform: ParticleTransform | None,
        acceptance: Any,
    ) -> tuple[ParticleState, StepContext, MiniPCNOutput]:
        """Map MiniPCN results back to particle space and build diagnostics."""
        xp = namespace(latent_particles)
        latent_last = asarray(latent_last, xp=xp, like=latent_particles)
        new_particles, _ = self._inverse_transform(
            latent_last, transform, context
        )
        _, final_evaluation = self._evaluate_target(new_particles, context)

        stored_chain = None
        if latent_chain is not None:
            stored_chain = self._inverse_chain(
                latent_chain,
                transform,
                context,
                like=latent_particles,
            )

        logger.debug(
            "MiniPCNMutation completed after %d steps.",
            self._n_steps,
        )
        output = MiniPCNOutput(
            acceptance=acceptance,
            step_fn=self._step_fn,
            steps=self._n_steps,
            chain=stored_chain,
        )
        return (
            state.replace_particles(
                new_particles,
                evaluation=final_evaluation,
            ),
            context,
            output,
        )
