from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, cast

import numpy as np

from .._array_api import asarray, namespace, to_numpy
from .._typing import Array
from ..transforms import ParticleTransform
from .base import SMCStep, StepContext

if TYPE_CHECKING:
    import emcee
    from orng import ArrayRNG

    from ..posterior import TemperedPosterior
    from ..state import ParticleState

logger = logging.getLogger(__name__)


def _call_log_prob(
    fn: LogProbFn,
    position: Array,
    context: StepContext,
) -> Any:
    try:
        return fn(position, context)
    except TypeError:
        return fn(position)


LogProbFn = Callable[..., Array | float]


def _as_scalar_log_prob(value: Any) -> float:
    array = to_numpy(value)
    if array.ndim == 0:
        return float(array)
    if array.size == 1:
        return float(array.reshape(()))
    raise TypeError(
        "Log-probability function must return a scalar when vectorization is disabled."
    )


def _as_vector_log_prob(value: Any, length: int) -> np.ndarray:
    array = to_numpy(value)
    return array.reshape(length)


class MutationStep(SMCStep):
    def __init__(
        self,
        log_prob_fn: Callable[..., Array | float] | "TemperedPosterior",
        *,
        transform: ParticleTransform | None = None,
        store_chains: bool = False,
    ):
        super().__init__()
        if hasattr(log_prob_fn, "log_prob"):
            posterior = log_prob_fn

            def _wrapped_log_prob(
                particles: Array,
                context: StepContext,
            ) -> Array | float:
                log_proposal_override = context.metadata.get(
                    "log_proposal_override"
                )
                return posterior.log_prob(
                    particles,
                    context,
                    log_proposal_override=log_proposal_override,
                )

            self._log_prob_fn = cast(LogProbFn, _wrapped_log_prob)
        else:
            self._log_prob_fn = cast(LogProbFn, log_prob_fn)
        self._transform = transform
        self._store_chains = store_chains
        self._chain: Array | None = None

    def _prepare_transform(
        self, particles: Array, context: StepContext
    ) -> ParticleTransform | None:
        if self._transform is None:
            return None
        xp = namespace(particles)
        transform = self._transform.clone()
        transform.fit(asarray(particles, xp=xp), context=context)
        return transform

    @staticmethod
    def _forward_transform(
        values: Array,
        transform: ParticleTransform | None,
        context: StepContext,
    ):
        if transform is None:
            return values
        xp = namespace(values)
        arr = asarray(values, xp=xp)
        latent, _ = transform.forward(arr, context=context)
        return asarray(latent, xp=xp)

    @staticmethod
    def _inverse_transform(
        values: Array,
        transform: ParticleTransform | None,
        context: StepContext,
    ):
        if transform is None:
            return values, None
        xp = namespace(values)
        arr = asarray(values, xp=xp)
        output, log_det = transform.inverse(arr, context=context)
        return asarray(output, xp=xp), asarray(log_det, xp=xp)

    @staticmethod
    def _combine_with_log_det(result: Any, log_det: Array | None):
        if log_det is None:
            return result
        xp = namespace(result, log_det)
        result_arr = xp.asarray(result)
        log_det_arr = xp.asarray(log_det)
        return result_arr + log_det_arr


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
        An emcee Move instance to use for proposing new positions.  If ``None``
        (default) uses the ``StretchMove``.
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
        move: emcee.moves.Move | None = None,
        run_kwargs: dict[str, Any] | None = None,
        transform: ParticleTransform | None = None,
        store_chains: bool = False,
        **kwargs,
    ):
        import emcee

        super().__init__(
            log_prob_fn, transform=transform, store_chains=store_chains
        )
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1.")
        self._n_steps = n_steps
        self._move = move or emcee.moves.StretchMove()
        self._run_kwargs: dict[str, Any] = run_kwargs or {}
        self._kwargs: dict[str, Any] = kwargs

    def apply(
        self,
        state: ParticleState,
        context: StepContext,
        rng: "ArrayRNG",
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
                result = _call_log_prob(self._log_prob_fn, original, context)
                result = self._combine_with_log_det(result, log_det)
                return _as_vector_log_prob(result, backend_positions.shape[0])

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
                result = _call_log_prob(self._log_prob_fn, original, context)
                result = self._combine_with_log_det(result, log_det)
                return _as_scalar_log_prob(result)

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


class MiniPCNMutation(MutationStep):
    """Run MiniPCN MCMC moves to rejuvenate particles.

    By default uses the 'tpcn' step function and targets an acceptance rate of 0.234.
    Also sets 'verbose' to False by default.

    Parameters
    ----------
    log_prob_fn
        Function to compute the log-probability of a particle position, or a
        `TemperedPosterior` instance.
    n_steps
        Number of MiniPCN steps to perform.
    step_fn
        Step function to use. See `minipcn` documentation for available options.
    verbose
        Whether to enable verbose output from the MiniPCN sampler.
    transform
        A `ParticleTransform` instance to apply to the particles before mutation.
    store_chains
        Whether to store the entire mutation chain in the metadata. Default is
        False.
    **kwargs
        Additional keyword arguments to pass to the `minipcn.Sampler` constructor.
    """

    def __init__(
        self,
        log_prob_fn: Callable[..., Array | float] | "TemperedPosterior",
        *,
        n_steps: int = 1,
        step_fn: str = "tpcn",
        verbose: bool = False,
        transform: ParticleTransform | None = None,
        store_chains: bool = False,
        **kwargs,
    ):
        super().__init__(
            log_prob_fn, transform=transform, store_chains=store_chains
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
        rng: "ArrayRNG",
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
            result = _call_log_prob(self._log_prob_fn, original, context)
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
        return state.with_updates(particles=new_particles, metadata=metadata)


def get_mutation_step(
    mutation: str | SMCStep | None,
    posterior: "TemperedPosterior",
    transform: ParticleTransform | None = None,
    mutation_n_steps: int = 10,
    mutation_kwargs: dict[str, Any] | None = None,
) -> SMCStep | None:
    kwargs = dict(mutation_kwargs or {})
    if transform is not None:
        kwargs["transform"] = transform
    if isinstance(mutation, SMCStep):
        mutation_step = mutation
    elif mutation == "emcee":
        run_kwargs = kwargs.pop("run_kwargs", {})
        mutation_step = EmceeMutation(
            posterior,
            n_steps=mutation_n_steps,
            run_kwargs=run_kwargs,
            **kwargs,
        )
    elif mutation == "minipcn":
        mutation_step = MiniPCNMutation(
            posterior,
            n_steps=mutation_n_steps,
            **kwargs,
        )
    elif mutation == "none":
        mutation_step = None
    else:
        raise ValueError(
            "mutation must be 'emcee', 'minipcn', 'none', or an SMCStep instance."
        )
    return mutation_step
