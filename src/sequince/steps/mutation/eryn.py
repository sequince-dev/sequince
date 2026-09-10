from __future__ import annotations

import logging
from contextlib import contextmanager
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
from array_api_compat import is_cupy_namespace

from ..._typing import Array
from ...array_api import asarray, namespace, to_numpy
from ...target import EvaluationCounts
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
from .metrics import (
    MutationMetric,
    MutationStepState,
)

if TYPE_CHECKING:
    from orng import RandomGenerator

    from ...state import ParticleState
    from ...target import Target

logger = logging.getLogger(__name__)


@contextmanager
def _seed_eryn_rng(seed: int, *, cupy: Any = None):
    """Isolate the global RNGs used by Eryn's CPU and GPU moves."""
    numpy_state = np.random.get_state()
    cupy_state = None
    try:
        np.random.seed(seed)
        if cupy is not None:
            cupy_state = cupy.random.get_random_state()
            # Replace rather than reseed the caller's mutable CuPy state.
            cupy.random.set_random_state(cupy.random.RandomState(seed))
        yield
    finally:
        np.random.set_state(numpy_state)
        if cupy_state is not None:
            cupy.random.set_random_state(cupy_state)


class _ZeroLogPrior:
    """Eryn prior that returns zero log probability for any input.

    The actual prior is handled by Sequince's target.
    """

    def __init__(self, ndim: int):
        self.key_order = list(range(ndim))

    def logpdf(self, values):
        return np.zeros(values.shape[:-1])


class ErynMutation(MutationStep):
    """Fixed-dimensional mutation step using Eryn.

    Parameters
    ----------
    log_prob_fn : callable or Target
        Function to compute the log-probability of a particle position, or a
        `Target` instance.
    n_steps : int, optional
        Number of mutation steps to perform. Default is 1.
    move : Eryn move object, optional
        Eryn move to use for the mutation. If not provided, a default
        `StretchMove` is used.
    run_kwargs : dict, optional
        Additional keyword arguments to pass to the Eryn sampler's `sample`
        method. Default is None.
    transform : ParticleTransform, optional
        Transform to apply to the particles before mutation. Default is None.
    transform_refit : TransformRefit, optional
        Strategy for refitting the transform. Default is "always".
    store_chains : bool, optional
        Whether to store the full chain of mutated particles. Default is False.
    metrics : list of MutationMetric, optional
        Metrics to compute during mutation. Default is an empty list.
    vectorize : bool, optional
        Whether to vectorize the log-probability evaluation. Default is True.
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
        metrics: tuple[MutationMetric, ...] | list[MutationMetric] = (),
        vectorize: bool = True,
    ):
        from eryn.moves import StretchMove

        super().__init__(
            log_prob_fn,
            transform=transform,
            transform_refit=transform_refit,
            store_chains=store_chains,
            metrics=metrics,
        )
        n_steps = validate_n_steps(n_steps)
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1.")
        self._n_steps = n_steps
        self._move = StretchMove() if move is None else move
        self._run_kwargs: dict[str, Any] = dict(run_kwargs or {})
        self._run_kwargs.setdefault("progress", False)
        if self._run_kwargs.get("thin_by", 1) != 1 or not self._run_kwargs.get(
            "store", True
        ):
            raise ValueError(
                "Eryn mutation bookkeeping requires unthinned transitions "
                "(thin_by=1) and store=True."
            )
        unsupported = set(self._run_kwargs) - {
            "progress",
            "tune",
            "skip_initial_state_check",
            "thin_by",
            "store",
        }
        if unsupported:
            raise ValueError(f"Unsupported Eryn sample options: {unsupported}")
        self._vectorize = vectorize

    def apply(
        self,
        state: ParticleState,
        context: StepContext,
        rng: "RandomGenerator",
    ) -> tuple[ParticleState, StepContext, dict[str, object]]:
        return self.apply_with_n_steps(
            state, context, rng, n_steps=self._n_steps
        )

    def apply_with_mutation_state(
        self,
        state: ParticleState,
        context: StepContext,
        rng: "RandomGenerator",
        *,
        n_steps: int,
        mutation_state: MutationStepState | None = None,
    ) -> tuple[
        ParticleState,
        StepContext,
        dict[str, object],
        MutationStepState | None,
    ]:
        n_steps = validate_n_steps(n_steps)
        if n_steps == 0:
            xp = namespace(state.particles)
            mutation_state = (
                self.metrics.initialize(
                    state.particles, state.log_weights, context
                )
                if mutation_state is None
                else self.metrics.prepare(
                    mutation_state, state.particles, context
                )
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
                {
                    "steps": 0,
                    "chain": chain,
                    "metrics": None,
                },
                mutation_state,
            )
        from eryn.ensemble import EnsembleSampler

        xp = namespace(state.particles)
        use_gpu = is_cupy_namespace(xp)
        cupy_backend: Any = None
        if use_gpu:
            import cupy

            cupy_backend = cupy

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
                "eryn evaluates internally selected walker subsets and its "
                "standard log-probability callback does not expose the "
                "original walker selection or context. Sequince therefore "
                "cannot guarantee that the correct auxiliary entry is "
                "supplied. Use a batch-preserving mutation implementation."
            )
        # Eryn requires host coordinates even for GPU-enabled moves. The
        # callback below restores the particle backend before evaluation.
        particles = to_numpy(latent_particles)
        if particles.ndim == 1:
            initial_positions = particles[:, np.newaxis]
            squeeze_last = True
        else:
            initial_positions = particles
            squeeze_last = False

        n_walkers, ndim = initial_positions.shape
        mutation_state = (
            self.metrics.initialize(
                asarray(initial_positions, xp=xp, like=latent_particles),
                state.log_weights,
                context,
            )
            if mutation_state is None
            else self.metrics.prepare(
                mutation_state,
                asarray(initial_positions, xp=xp, like=latent_particles),
                context,
            )
        )
        logger.debug(
            "ErynMutation starting: %d walkers, %d dimensions, %d steps.",
            n_walkers,
            ndim,
            n_steps,
        )

        counts = EvaluationCounts.zeros(xp=xp)

        def log_prob_callable(positions):
            nonlocal counts
            backend_positions = asarray(
                positions, xp=xp, like=latent_particles
            )
            if not self._vectorize and self._target is not None:
                backend_positions = xp.expand_dims(backend_positions, axis=0)
            original, log_det = self._inverse_transform(
                backend_positions,
                transform,
                context,
                auxiliary=transform_auxiliary,
            )
            result, _, delta = self._evaluate_target(original, context)
            counts = counts + delta
            result = self._combine_with_log_det(result, log_det)
            if self._vectorize:
                return as_vector_log_prob(result, positions.shape[0])
            return as_scalar_log_prob(result)

        sampler = EnsembleSampler(
            n_walkers,
            ndim,
            log_prob_callable,
            priors={"model_0": _ZeroLogPrior(ndim)},
            moves=deepcopy(self._move),
            vectorize=self._vectorize,
        )
        for move in sampler.moves:
            if hasattr(move, "use_gpu"):
                move.use_gpu = use_gpu
            if hasattr(move, "return_gpu"):
                move.return_gpu = False
        eryn_seed = int(
            float(to_numpy(rng.random((1,)))[0]) * np.iinfo(np.uint32).max
        )
        sampler.random_state = np.random.RandomState(eryn_seed).get_state()

        previous_accepted = np.zeros(n_walkers)
        # Red/blue splitting uses NumPy's global RNG; GPU stretch proposals
        # use CuPy's. Seed both independently of sampler.random_state.
        with _seed_eryn_rng(eryn_seed, cupy=cupy_backend):
            for state_result in sampler.sample(
                initial_positions[None, :, None, :],
                iterations=n_steps,
                **self._run_kwargs,
            ):
                cumulative_accepted = sampler.backend.accepted[0]
                accepted = cumulative_accepted > previous_accepted
                mutation_state = self.metrics.update(
                    mutation_state,
                    asarray(
                        state_result.branches_coords["model_0"][
                            0, :, 0, :
                        ].copy(),
                        xp=xp,
                        like=latent_particles,
                    ),
                    asarray(accepted, xp=xp, like=latent_particles),
                    context,
                )
                previous_accepted = cumulative_accepted.copy()
        new_positions = state_result.branches_coords["model_0"][0, :, 0, :]

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
            final_evaluation, delta = self._target.prepare_evaluation(
                new_particles,
                context,
                return_counts=True,
            )
            counts = counts + delta
        context = context.add_evaluation_counts(counts)

        acceptance = float(np.mean(sampler.acceptance_fraction))
        stored_chain = None
        if self._store_chains:
            stored_chain = self._inverse_chain(
                sampler.get_chain()["model_0"][:, 0, :, 0, :],
                transform,
                context,
                auxiliary=transform_auxiliary,
                like=latent_particles,
            )

        if stored_chain is not None and squeeze_last:
            stored_chain = stored_chain[..., 0]

        logger.debug(
            "ErynMutation completed: acceptance=%.3f after %d steps.",
            acceptance,
            n_steps,
        )
        output = {
            "steps": n_steps,
            "chain": stored_chain,
            "metrics": None,
        }
        return (
            state.replace_particles(
                new_particles,
                evaluation=final_evaluation,
            ),
            context,
            output,
            mutation_state,
        )
