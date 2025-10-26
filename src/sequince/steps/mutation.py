from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

import emcee
import numpy as np

from .base import SMCStep, StepContext

if TYPE_CHECKING:
    from ..state import ParticleState


def _call_log_prob(
    fn: Callable[..., float],
    position: np.ndarray,
    context: StepContext,
) -> float:
    try:
        return float(fn(position, context))
    except TypeError:
        return float(fn(position))


class EmceeMutation(SMCStep):
    """Run emcee ensemble MCMC moves to rejuvenate particles."""

    def __init__(
        self,
        log_prob_fn: Callable[..., float],
        *,
        n_steps: int = 1,
        move: Optional[emcee.moves.Move] = None,
    ):
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1.")
        self._log_prob_fn = log_prob_fn
        self._n_steps = n_steps
        self._move = move or emcee.moves.StretchMove()

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

        def log_prob(position: np.ndarray) -> float:
            return _call_log_prob(self._log_prob_fn, position, context)

        sampler = emcee.EnsembleSampler(
            n_walkers,
            ndim,
            log_prob,
            moves=self._move,
            vectorize=False,
            # seed=int(rng.integers(0, np.iinfo(np.uint32).max)),
        )

        state_result = sampler.run_mcmc(
            initial_positions, self._n_steps, progress=False
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

        return state.with_updates(particles=new_particles, metadata=metadata)
