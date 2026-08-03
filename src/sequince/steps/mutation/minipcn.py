from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

import numpy as np

from ..._array_api import asarray, namespace
from ..._typing import Array
from ...transforms import ParticleTransform
from ..context import LOG_PROPOSAL_OVERRIDE_KEY, StepContext
from .base import (
    MutationStep,
    TransformRefit,
)

if TYPE_CHECKING:
    from orng import RandomGenerator

    from ...posterior import TemperedPosterior
    from ...state import ParticleState

logger = logging.getLogger(__name__)


class MiniPCNMutation(MutationStep):
    """Run MiniPCN MCMC moves to rejuvenate particles.

    By default uses the 'tpcn' step function and targets an acceptance rate of 0.234.
    Also sets 'verbose' to False by default.
    """

    def __init__(
        self,
        log_prob_fn: Callable[..., Array | float] | "TemperedPosterior",
        *,
        n_steps: int = 1,
        step_fn: str = "tpcn",
        verbose: bool = False,
        transform: ParticleTransform | None = None,
        transform_refit: TransformRefit = "always",
        store_chains: bool = False,
        log_proposal_override_key: str = LOG_PROPOSAL_OVERRIDE_KEY,
        **kwargs,
    ):
        super().__init__(
            log_prob_fn,
            transform=transform,
            transform_refit=transform_refit,
            store_chains=store_chains,
            log_proposal_override_key=log_proposal_override_key,
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
    ):
        from minipcn import Sampler

        xp = namespace(state.particles)
        transform = self._prepare_transform(state.particles, context)
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

        chain, history = sampler.sample(
            latent_particles,
            n_steps=self._n_steps,
            verbose=self._verbose,
            rng=rng,
        )

        latent_last = asarray(chain[-1], xp=xp, like=latent_particles)
        new_particles, _ = self._inverse_transform(
            latent_last, transform, context
        )
        _, final_evaluation = self._evaluate_target(new_particles, context)

        metadata = dict(state.metadata)
        metadata["minipcn_acceptance"] = float(
            np.mean(history.acceptance_rate)
        )
        metadata["minipcn_step_fn"] = self._step_fn
        metadata["minipcn_steps"] = self._n_steps

        if self._store_chains:
            metadata["mutation_chains"] = chain

        acceptance = metadata["minipcn_acceptance"]
        logger.debug(
            "MiniPCNMutation completed: acceptance=%.3f after %d steps.",
            acceptance,
            self._n_steps,
        )
        return state.replace_particles(
            new_particles,
            evaluation=final_evaluation,
            metadata=metadata,
        )
