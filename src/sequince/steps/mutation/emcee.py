from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from ..._array_api import asarray, namespace, to_numpy
from ..._typing import Array
from ...transforms import ParticleTransform
from ..context import StepContext
from .base import (
    MutationStep,
    TransformRefit,
    as_scalar_log_prob,
    as_vector_log_prob,
    call_log_prob,
)

if TYPE_CHECKING:
    from orng import RandomGenerator

    from ...posterior import TemperedPosterior
    from ...state import ParticleState

logger = logging.getLogger(__name__)

EMCEE_MOVES = {
    "de": "DEMove",
    "de-snooker": "DESnookerMove",
    "gaussian": "GaussianMove",
    "kde": "KDEMove",
    "stretch": "StretchMove",
    "walk": "WalkMove",
}


def get_emcee_move(name: str, **kwargs: Any) -> Any:
    """Construct a named emcee move."""
    import emcee

    try:
        class_name = EMCEE_MOVES[name]
    except KeyError as exc:
        valid = ", ".join(sorted(EMCEE_MOVES))
        raise ValueError(
            f"Unknown emcee move '{name}'. Valid options are: {valid}."
        ) from exc
    return getattr(emcee.moves, class_name)(**kwargs)


def resolve_emcee_moves(spec: Any) -> Any:
    """Resolve an emcee move object, named move, preset, or weighted schedule."""
    if spec is None:
        return None
    if isinstance(spec, str) and spec == "bilby-de":
        return [
            (get_emcee_move("de", gamma0=None, sigma=1e-5), 0.9),
            (get_emcee_move("de", gamma0=1.0, sigma=1e-5), 0.1),
        ]
    if isinstance(spec, str) or isinstance(spec, Mapping):
        return _resolve_emcee_move_spec(spec)
    if not isinstance(spec, Sequence):
        return spec

    weighted_moves = [_resolve_weighted_move(item) for item in spec]
    weights = [weight for _, weight in weighted_moves]
    if any(not math.isfinite(weight) or weight < 0.0 for weight in weights):
        raise ValueError("Emcee move weights must be finite and non-negative.")
    if sum(weights) <= 0.0:
        raise ValueError("At least one emcee move weight must be positive.")
    return weighted_moves


def _resolve_weighted_move(spec: Any) -> tuple[Any, float]:
    if isinstance(spec, Mapping):
        config = dict(spec)
        weight = float(config.pop("weight", 1.0))
        return _resolve_emcee_move_spec(config), weight
    if (
        isinstance(spec, Sequence)
        and not isinstance(spec, str)
        and len(spec) == 2
        and isinstance(spec[1], Real)
    ):
        return _resolve_emcee_move_spec(spec[0]), float(spec[1])
    return _resolve_emcee_move_spec(spec), 1.0


def _resolve_emcee_move_spec(spec: Any) -> Any:
    if isinstance(spec, str):
        return get_emcee_move(spec)
    if not isinstance(spec, Mapping):
        return spec

    config = dict(spec)
    try:
        name = config.pop("move")
    except KeyError as exc:
        raise ValueError(
            "Emcee move mappings require a 'move' option."
        ) from exc
    move_kwargs = dict(config.pop("move_kwargs", {}) or {})
    if config:
        raise ValueError(
            f"Unknown emcee move options: {', '.join(sorted(config))}."
        )
    return get_emcee_move(name, **move_kwargs)


class EmceeMutation(MutationStep):
    """Run emcee ensemble MCMC moves to rejuvenate particles.

    Parameters
    ----------
    log_prob_fn
        Function to compute the log-probability of a particle position, or a
        `TemperedPosterior` instance.
    n_steps
        Number of emcee steps to perform.
    move
        An emcee Move instance, named move configuration, weighted move list,
        or preset to use for proposing new positions. If ``None`` (default),
        uses ``StretchMove``.
    run_kwargs
        Additional keyword arguments to pass to the ``emcee.EnsembleSampler.run_mcmc``
        method.
    transform
        A `ParticleTransform` instance to apply to the particles before mutation.
    store_chains
        Whether to store the entire mutation chain in the metadata. Default is
        False.
    **kwargs
        Additional keyword arguments to pass to the ``emcee.EnsembleSampler``
    """

    def __init__(
        self,
        log_prob_fn: Callable[..., Array | float] | "TemperedPosterior",
        *,
        n_steps: int = 1,
        move: Any = None,
        run_kwargs: dict[str, Any] | None = None,
        transform: ParticleTransform | None = None,
        transform_refit: TransformRefit = "always",
        store_chains: bool = False,
        **kwargs,
    ):
        import emcee

        super().__init__(
            log_prob_fn,
            transform=transform,
            transform_refit=transform_refit,
            store_chains=store_chains,
        )
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1.")
        self._n_steps = n_steps
        self._move = resolve_emcee_moves(move) or emcee.moves.StretchMove()
        self._run_kwargs: dict[str, Any] = run_kwargs or {}
        self._kwargs: dict[str, Any] = kwargs

    def apply(
        self,
        state: ParticleState,
        context: StepContext,
        rng: "RandomGenerator",
    ) -> ParticleState:
        import emcee

        xp = namespace(state.particles)
        transform = self._prepare_transform(state.particles, context)
        latent_particles = self._forward_transform(
            state.particles, transform, context
        )
        particles = to_numpy(latent_particles)
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

        vectorized = bool(self._kwargs.get("vectorize", False))

        if vectorized:

            def _log_prob_vectorized(positions: np.ndarray) -> np.ndarray:
                backend_positions = asarray(
                    positions, xp=xp, like=latent_particles
                )
                original, log_det = self._inverse_transform(
                    backend_positions, transform, context
                )
                result = call_log_prob(self._log_prob_fn, original, context)
                result = self._combine_with_log_det(result, log_det)
                return as_vector_log_prob(result, backend_positions.shape[0])

            log_prob_callable: Callable[[np.ndarray], Any] = (
                _log_prob_vectorized
            )

        else:

            def _log_prob_scalar(position: np.ndarray) -> float:
                backend_position = asarray(
                    position, xp=xp, like=latent_particles
                )
                original, log_det = self._inverse_transform(
                    backend_position, transform, context
                )
                result = call_log_prob(self._log_prob_fn, original, context)
                result = self._combine_with_log_det(result, log_det)
                return as_scalar_log_prob(result)

            log_prob_callable = _log_prob_scalar

        sampler = emcee.EnsembleSampler(
            n_walkers,
            ndim,
            log_prob_callable,
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
            result_np = new_positions[:, 0]
        else:
            result_np = new_positions

        latent_result = asarray(result_np, xp=xp, like=latent_particles)
        new_particles, _ = self._inverse_transform(
            latent_result, transform, context
        )

        metadata = dict(state.metadata)
        metadata["emcee_acceptance"] = float(
            np.mean(sampler.acceptance_fraction)
        )
        metadata["emcee_steps"] = self._n_steps
        acceptance = metadata["emcee_acceptance"]
        if self._store_chains:
            chains = sampler.get_chain(flat=False, discard=0, thin=1)
            metadata["mutation_chains"] = chains

        logger.debug(
            "EmceeMutation completed: acceptance=%.3f after %d steps.",
            acceptance,
            self._n_steps,
        )
        return state.with_updates(particles=new_particles, metadata=metadata)
