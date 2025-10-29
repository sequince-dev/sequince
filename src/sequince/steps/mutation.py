from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Optional

import emcee
import numpy as np

from .base import SMCStep, StepContext

if TYPE_CHECKING:
    from ..posterior import TemperedPosterior
    from ..state import ParticleState

logger = logging.getLogger(__name__)


def _call_log_prob(
    fn: Callable[..., float],
    position: np.ndarray,
    context: StepContext,
) -> float:
    try:
        return fn(position, context)
    except TypeError:
        return fn(position)


class EmceeMutation(SMCStep):
    """Run emcee ensemble MCMC moves to rejuvenate particles."""

    def __init__(
        self,
        log_prob_fn: Callable[..., float] | "TemperedPosterior",
        *,
        n_steps: int = 1,
        move: Optional[emcee.moves.Move] = None,
        run_kwargs: dict = None,
        **kwargs,
    ):
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1.")
        if hasattr(log_prob_fn, "log_prob"):
            self._log_prob_fn = log_prob_fn.log_prob  # type: ignore[union-attr]
        else:
            self._log_prob_fn = log_prob_fn  # type: ignore[assignment]
        self._n_steps = n_steps
        self._move = move or emcee.moves.StretchMove()
        self._run_kwargs = run_kwargs or {}
        self._kwargs = kwargs

    def apply(
        self,
        state: ParticleState,
        context: StepContext,
        rng: np.random.Generator,
    ) -> ParticleState:
        particles = np.asarray(state.particles)
        if particles.ndim == 1:
            initial_positions = particles[:, np.newaxis]
            squeeze_last = True
        else:
            initial_positions = particles
            squeeze_last = False

        n_walkers, ndim = initial_positions.shape
        logger.debug(
            "EmceeMutation starting: %d walkers, %d dimensions, %d steps.",
            n_walkers,
            ndim,
            self._n_steps,
        )

        def log_prob(position: np.ndarray) -> float:
            return _call_log_prob(self._log_prob_fn, position, context)

        sampler = emcee.EnsembleSampler(
            n_walkers,
            ndim,
            log_prob,
            moves=self._move,
            **self._kwargs,
        )

        state_result = sampler.run_mcmc(
            initial_positions,
            self._n_steps,
            progress=False,
            **self._run_kwargs,
        )
        new_positions = state_result.coords

        if squeeze_last:
            new_particles = new_positions[:, 0]
        else:
            new_particles = new_positions

        metadata = dict(state.metadata)
        metadata["emcee_acceptance"] = float(
            np.mean(sampler.acceptance_fraction)
        )
        metadata["emcee_steps"] = self._n_steps
        acceptance = metadata["emcee_acceptance"]
        logger.debug(
            "EmceeMutation completed: acceptance=%.3f after %d steps.",
            acceptance,
            self._n_steps,
        )
        return state.with_updates(particles=new_particles, metadata=metadata)
