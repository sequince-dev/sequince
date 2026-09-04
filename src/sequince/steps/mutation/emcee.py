from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from ..._typing import Array
from ...array_api import asarray, namespace, to_numpy
from ...target import EvaluationCounts, TargetEvaluation
from ...transforms import (
    ParticleTransform,
    has_particle_aligned_auxiliary,
)
from ..base import validate_n_steps
from ..context import StepContext
from .base import (
    MutationStep,
    TransformRefit,
    as_scalar_log_prob,
    as_vector_log_prob,
)

if TYPE_CHECKING:
    from orng import RandomGenerator

    from ...state import ParticleState
    from ...target import Target

logger = logging.getLogger(__name__)

EMCEE_MOVES = {
    "de": "DEMove",
    "de-snooker": "DESnookerMove",
    "gaussian": "GaussianMove",
    "kde": "KDEMove",
    "stretch": "StretchMove",
    "walk": "WalkMove",
}


def _evaluation_blob(
    evaluation: TargetEvaluation,
    index: int,
) -> dict[str, np.ndarray]:
    """Convert one particle's target components to an emcee blob."""
    return {
        name: np.array(to_numpy(values[index]), copy=True)
        for name, values in evaluation.components.items()
    }


def _evaluation_from_blobs(
    blobs: Any,
    *,
    target: Target,
    particles: Array,
    context: StepContext,
) -> TargetEvaluation:
    """Reassemble the final particle-aligned evaluation from emcee blobs."""
    entries = list(np.asarray(blobs, dtype=object).reshape(-1))
    if not entries or not all(isinstance(entry, Mapping) for entry in entries):
        raise RuntimeError("emcee did not return target-evaluation blobs.")

    component_names = tuple(entries[0])
    if any(tuple(entry) != component_names for entry in entries[1:]):
        raise RuntimeError(
            "emcee returned inconsistent target-evaluation blobs."
        )

    xp = namespace(particles)
    evaluation = TargetEvaluation(
        components={
            name: xp.stack(
                tuple(
                    asarray(entry[name], xp=xp, like=particles)
                    for entry in entries
                ),
                axis=0,
            )
            for name in component_names
        }
    )
    return target.bind_evaluation(evaluation, context)


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

    Transforms requiring walker-specific auxiliary data are currently
    unsupported. emcee evaluates internally selected walker subsets, but its
    standard log-probability callback does not expose the original walker
    selection or context. Such auxiliary is rejected because Sequince cannot
    guarantee that the correct entry would be supplied to each walker.

    Parameters
    ----------
    log_prob_fn
        Function to compute the log-probability of a particle position, or a
        `Target` instance.
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
        log_prob_fn: Callable[..., Array | float] | "Target",
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
        self._run_kwargs: dict[str, Any] = dict(run_kwargs or {})
        self._run_kwargs.setdefault("progress", False)
        self._kwargs: dict[str, Any] = kwargs

    def apply(
        self,
        state: ParticleState,
        context: StepContext,
        rng: "RandomGenerator",
    ) -> tuple[ParticleState, StepContext, dict[str, object]]:
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
    ) -> tuple[ParticleState, StepContext, dict[str, object]]:
        n_steps = validate_n_steps(n_steps)
        if n_steps == 0:
            xp = namespace(state.particles)
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
                {
                    "acceptance": 0.0,
                    "steps": 0,
                    "chain": chain,
                },
            )
        import emcee

        xp = namespace(state.particles)
        transform, context = self._prepare_transform(
            state.particles, context, rng=rng
        )
        latent_particles, transform_auxiliary = self._forward_transform(
            state.particles, transform, context, rng=rng
        )
        if has_particle_aligned_auxiliary(
            transform_auxiliary,
            n_particles=state.particles.shape[0],
        ):
            raise NotImplementedError(
                "The transform requires walker-specific auxiliary data, but "
                "emcee evaluates internally selected walker subsets and its "
                "standard log-probability callback does not expose the "
                "original walker selection or context. Sequince therefore "
                "cannot guarantee that the correct auxiliary entry is "
                "supplied. Use a batch-preserving mutation implementation."
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
            n_steps,
        )

        vectorized = bool(self._kwargs.get("vectorize", False))
        counts = EvaluationCounts.zeros(xp=xp)

        if vectorized:

            def _log_prob_vectorized(positions: np.ndarray) -> Any:
                nonlocal counts
                backend_positions = asarray(
                    positions, xp=xp, like=latent_particles
                )
                original, log_det = self._inverse_transform(
                    backend_positions,
                    transform,
                    context,
                    auxiliary=transform_auxiliary,
                )
                result, evaluation, delta = self._evaluate_target(
                    original, context
                )
                counts = counts + delta
                result = self._combine_with_log_det(result, log_det)
                log_prob = as_vector_log_prob(
                    result, backend_positions.shape[0]
                )
                if self._target is None:
                    return log_prob

                if evaluation is None:  # pragma: no cover - target contract
                    raise RuntimeError("Target did not return an evaluation.")
                return [
                    (log_prob[index], _evaluation_blob(evaluation, index))
                    for index in range(backend_positions.shape[0])
                ]

            log_prob_callable: Callable[[np.ndarray], Any] = (
                _log_prob_vectorized
            )

        else:

            def _log_prob_scalar(position: np.ndarray) -> Any:
                nonlocal counts
                backend_position = asarray(
                    position, xp=xp, like=latent_particles
                )
                evaluation_position = (
                    xp.expand_dims(backend_position, axis=0)
                    if self._target is not None
                    else backend_position
                )
                original, log_det = self._inverse_transform(
                    evaluation_position,
                    transform,
                    context,
                    auxiliary=transform_auxiliary,
                )
                result, evaluation, delta = self._evaluate_target(
                    original, context
                )
                counts = counts + delta
                result = self._combine_with_log_det(result, log_det)
                log_prob = as_scalar_log_prob(result)
                if evaluation is None:
                    return log_prob
                return log_prob, _evaluation_blob(evaluation, 0)

            log_prob_callable = _log_prob_scalar

        sampler_kwargs = dict(self._kwargs)
        if self._target is not None:
            sampler_kwargs["blobs_dtype"] = object
        sampler = emcee.EnsembleSampler(
            n_walkers,
            ndim,
            log_prob_callable,
            moves=self._move,
            **sampler_kwargs,
        )
        emcee_seed = int(
            float(to_numpy(rng.random((1,)))[0]) * np.iinfo(np.uint32).max
        )
        sampler.random_state = np.random.RandomState(emcee_seed).get_state()

        state_result = sampler.run_mcmc(
            initial_positions,
            n_steps,
            **self._run_kwargs,
        )
        new_positions = state_result.coords

        if squeeze_last:
            result_np = new_positions[:, 0]
        else:
            result_np = new_positions

        latent_result = asarray(result_np, xp=xp, like=latent_particles)
        new_particles, _ = self._inverse_transform(
            latent_result,
            transform,
            context,
            auxiliary=transform_auxiliary,
        )
        final_evaluation = None
        if self._target is not None:
            final_evaluation = _evaluation_from_blobs(
                state_result.blobs,
                target=self._target,
                particles=new_particles,
                context=context,
            )
        context = context.add_evaluation_counts(counts)

        acceptance = float(np.mean(sampler.acceptance_fraction))
        stored_chain = None
        if self._store_chains:
            chains = sampler.get_chain(flat=False, discard=0, thin=1)
            stored_chain = self._inverse_chain(
                chains,
                transform,
                context,
                auxiliary=transform_auxiliary,
                like=latent_particles,
            )

        logger.debug(
            "EmceeMutation completed: acceptance=%.3f after %d steps.",
            acceptance,
            n_steps,
        )
        output = {
            "acceptance": acceptance,
            "steps": n_steps,
            "chain": stored_chain,
        }
        return (
            state.replace_particles(
                new_particles,
                evaluation=final_evaluation,
            ),
            context,
            output,
        )
