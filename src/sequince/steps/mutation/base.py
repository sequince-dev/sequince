from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Callable, Literal, cast

import numpy as np

from ..._typing import Array
from ...array_api import asarray, namespace, to_numpy
from ...target import EvaluationCounts, Target
from ...transforms import ParticleTransform, _repeat_transform_auxiliary
from ...utils.copying import array_safe_copy
from ..base import SMCStep
from ..context import StepContext
from .metrics import (
    MutationMetric,
    MutationMetrics,
    MutationStepState,
)
from .metrics.base import with_metrics

if TYPE_CHECKING:
    from ...state import ParticleState
    from ...target import TargetEvaluation

LogProbFn = Callable[..., Array | float]
PARTICLE_TRANSFORM_KEY = "particle_transform"
_FUNCTIONAL_TRANSFORM_MARKER_KEY = "_functional_particle_transform"
TransformRefit = (
    Literal["always", "per-context", "once"]
    | Callable[[Array, StepContext], bool]
)


def call_log_prob(
    fn: LogProbFn,
    position: Array,
    context: StepContext,
) -> Any:
    try:
        return fn(position, context)
    except TypeError:
        return fn(position)


def as_scalar_log_prob(value: Any) -> float:
    array = to_numpy(value)
    if array.ndim == 0:
        return float(array)
    if array.size == 1:
        return float(array.reshape(()))
    raise TypeError(
        "Log-probability function must return a scalar when vectorization is disabled."
    )


def as_vector_log_prob(value: Any, length: int) -> np.ndarray:
    array = to_numpy(value)
    return array.reshape(length)


class MutationStep(SMCStep):
    """Base class for mutations that optionally operate in transformed space.

    ``transform_refit`` controls whether a cloned transform is fitted for every
    mutation call, once per outer :class:`StepContext`, or only once for the
    lifetime of this mutation step. A callable can instead request fits based
    on the particles and context. It is evaluated once per mutation
    application, including the initial fit, and may consume or reset a trigger
    stored in the context.
    """

    def __init__(
        self,
        log_prob_fn: Callable[..., Array | float] | "Target",
        *,
        transform: ParticleTransform | None = None,
        transform_refit: TransformRefit = "always",
        store_chains: bool = False,
        metrics: tuple[MutationMetric, ...] | list[MutationMetric] = (),
    ):
        super().__init__()
        self.metrics = MutationMetrics(metrics)
        self._target = log_prob_fn if isinstance(log_prob_fn, Target) else None
        self._log_prob_fn = (
            None if self._target is not None else cast(LogProbFn, log_prob_fn)
        )
        if not callable(transform_refit) and not isinstance(
            transform_refit, str
        ):
            raise TypeError("transform_refit must be a mode or callable.")
        if isinstance(transform_refit, str) and transform_refit not in {
            "always",
            "per-context",
            "once",
        }:
            raise ValueError(
                "transform_refit must be 'always', 'per-context', "
                "'once', or a callable."
            )
        self._transform = transform
        self._transform_refit = transform_refit
        self._prepared_transform: ParticleTransform | None = None
        self._prepared_context_key: tuple[int, int] | None = None
        self._store_chains = store_chains
        self._chain: Array | None = None

    _n_steps: int

    @property
    def n_steps(self) -> int:
        """Default transition count for one standalone application."""
        return self._n_steps

    def apply_with_mutation_state(
        self,
        state: "ParticleState",
        context: StepContext,
        rng: Any,
        *,
        n_steps: int,
        mutation_state: MutationStepState | None = None,
    ) -> tuple["ParticleState", StepContext, Any, MutationStepState | None]:
        """Run transitions within a block without finalizing metrics.

        Pass ``mutation_state=None`` to initialize a new block; otherwise pass
        the state returned by the preceding call. The state holds individual
        metric states and cumulative acceptance bookkeeping. It must belong
        to the same metric configuration and mutation coordinate space.

        Returns ``(state, context, output, mutation_state)``. The output's
        metrics remain unset, including on the first and zero-step calls.
        Concrete mutations prepare existing metrics on entry, update state for each
        transition, and return it explicitly without calling metric.result().
        Runtime state must never be stored on the mutation object.

        Use ``apply`` or ``apply_with_n_steps`` for a complete standalone block.
        RepeatedMutationStep uses this method to continue across rounds and
        finalizes metrics once after the complete block.
        """
        raise NotImplementedError

    def apply_with_n_steps(self, state, context, rng, *, n_steps):
        """Run a complete fixed-length block and finalize its configured metrics."""
        state, context, output, mutation_state = (
            self.apply_with_mutation_state(
                state, context, rng, n_steps=n_steps, mutation_state=None
            )
        )
        return (
            state,
            context,
            with_metrics(output, self.metrics.result(mutation_state, context)),
        )

    @property
    def store_chains(self) -> bool:
        """Whether full mutation chains are included in step outputs."""
        return self._store_chains

    @store_chains.setter
    def store_chains(self, value: bool) -> None:
        self._store_chains = bool(value)

    def state_dict(self) -> dict[str, Any]:
        """Return the fitted transform and the context identifying its fit."""
        return {
            "prepared_transform": array_safe_copy(self._prepared_transform),
            "prepared_context_key": self._prepared_context_key,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore the transform cache; legacy empty state clears the cache."""
        transform = state.get("prepared_transform")
        context_key = state.get("prepared_context_key")
        if transform is not None and not isinstance(
            transform, ParticleTransform
        ):
            raise TypeError("Prepared transform must be a ParticleTransform.")
        if context_key is not None and (
            not isinstance(context_key, tuple) or len(context_key) != 2
        ):
            raise TypeError("Prepared context key must be a pair of indices.")
        if (transform is None) != (context_key is None):
            raise ValueError(
                "Prepared transform and context key must be paired."
            )
        self._prepared_transform = array_safe_copy(transform)
        self._prepared_context_key = context_key

    def _evaluate_target(
        self,
        particles: Array,
        context: StepContext,
        *,
        evaluation: "TargetEvaluation | None" = None,
    ) -> tuple[Any, "TargetEvaluation | None", EvaluationCounts]:
        xp = namespace(particles)
        n_particles = 1 if particles.ndim == 1 else particles.shape[0]
        log_prob_count = EvaluationCounts(
            log_target_evaluations=xp.asarray(n_particles)
        )
        if self._target is None:
            if self._log_prob_fn is None:  # pragma: no cover
                raise RuntimeError("Mutation target is not configured.")
            return (
                call_log_prob(self._log_prob_fn, particles, context),
                None,
                log_prob_count,
            )
        evaluation, counts = self._target.prepare_evaluation(
            particles,
            context,
            evaluation=evaluation,
            return_counts=True,
        )
        return (
            self._target.log_prob(
                particles,
                context,
                evaluation=evaluation,
            ),
            evaluation,
            counts,
        )

    def initialize_functional_state_and_context(
        self,
        state: "ParticleState",
        context: StepContext,
    ) -> tuple["ParticleState", StepContext]:
        """Initialize cached target data and functional transform state."""
        if self._target is not None:
            evaluation, counts = self._target.prepare_evaluation(
                state.particles,
                context,
                evaluation=state.evaluation,
                return_counts=True,
            )
            state = state.with_updates(evaluation=evaluation)
            context = context.add_evaluation_counts(counts)
        return self._initialize_functional_transform(state, context)

    def _initialize_functional_transform(
        self,
        state: "ParticleState",
        context: StepContext,
    ) -> tuple["ParticleState", StepContext]:
        """Add a fitted, shape-complete transform to the functional carry."""
        if self._transform is None:
            return state, context
        if callable(self._transform_refit):
            raise NotImplementedError(
                "Callable transform_refit policies are not supported by the "
                "functional API. Use 'always', 'per-context', or 'once'."
            )
        if (
            self._transform_refit == "once"
            and _FUNCTIONAL_TRANSFORM_MARKER_KEY in context.metadata
            and PARTICLE_TRANSFORM_KEY in context.metadata
        ):
            return state, context
        transform = self._fit_transform(state.particles, context)
        context = context.with_metadata(
            **{
                PARTICLE_TRANSFORM_KEY: transform,
                _FUNCTIONAL_TRANSFORM_MARKER_KEY: (),
            }
        )
        return state, context

    def _fit_transform(
        self,
        particles: Array,
        context: StepContext,
        *,
        rng: Any = None,
    ) -> ParticleTransform:
        if self._transform is None:  # pragma: no cover - guarded by callers
            raise RuntimeError("Particle transform is not configured.")
        xp = namespace(particles)
        return self._transform.fit(
            asarray(particles, xp=xp),
            context=context,
            rng=rng,
        )

    def _prepare_functional_transform(
        self,
        particles: Array,
        context: StepContext,
        *,
        rng: Any = None,
    ) -> tuple[ParticleTransform, StepContext]:
        transform = context.metadata[PARTICLE_TRANSFORM_KEY]
        if self._transform_refit in {"always", "per-context"}:
            transform = self._fit_transform(particles, context, rng=rng)
        context = context.with_metadata(**{PARTICLE_TRANSFORM_KEY: transform})
        return transform, context

    def _prepare_transform(
        self,
        particles: Array,
        context: StepContext,
        *,
        rng: Any = None,
    ) -> tuple[ParticleTransform | None, StepContext]:
        if self._transform is None:
            return None, context
        if _FUNCTIONAL_TRANSFORM_MARKER_KEY in context.metadata:
            return self._prepare_functional_transform(
                particles, context, rng=rng
            )
        triggered = (
            bool(self._transform_refit(particles, context))
            if callable(self._transform_refit)
            else False
        )
        context_key = (context.t, context.next_observation_index)
        should_fit = (
            self._transform_refit == "always"
            or self._prepared_transform is None
            or triggered
            or (
                self._transform_refit == "per-context"
                and self._prepared_context_key != context_key
            )
        )
        if should_fit:
            transform = self._fit_transform(particles, context, rng=rng)
            self._prepared_transform = transform
            self._prepared_context_key = context_key
        else:
            cached_transform = self._prepared_transform
            if cached_transform is None:  # pragma: no cover - guarded above
                raise RuntimeError("Prepared transform cache is empty.")
            transform = cached_transform
        context = context.with_metadata(**{PARTICLE_TRANSFORM_KEY: transform})
        return transform, context

    @staticmethod
    def _forward_transform(
        values: Array,
        transform: ParticleTransform | None,
        context: StepContext,
        *,
        rng: Any = None,
    ):
        if transform is None:
            return values, None
        xp = namespace(values)
        arr = asarray(values, xp=xp)
        latent, _, auxiliary = transform.forward(arr, context=context, rng=rng)
        return asarray(latent, xp=xp), auxiliary

    @staticmethod
    def _inverse_transform(
        values: Array,
        transform: ParticleTransform | None,
        context: StepContext,
        *,
        auxiliary: Any = None,
        rng: Any = None,
    ):
        if transform is None:
            return values, None
        xp = namespace(values)
        arr = asarray(values, xp=xp)
        output, log_det = transform.inverse(
            arr, context=context, auxiliary=auxiliary, rng=rng
        )
        return asarray(output, xp=xp), asarray(log_det, xp=xp)

    def _inverse_chain(
        self,
        chain: Array,
        transform: ParticleTransform | None,
        context: StepContext,
        *,
        auxiliary: Any = None,
        like: Array,
    ) -> Array:
        """Map a stored mutation chain back into particle coordinates.

        The leading chain axis represents mutation steps, not particles. Flatten
        the step and particle axes into one batch and repeat particle-aligned
        auxiliary leaves for each step before performing one batched inverse.
        """
        xp = namespace(like)
        latent_chain = asarray(chain, xp=xp, like=like)
        if transform is None:
            return latent_chain
        n_steps, n_particles = latent_chain.shape[:2]
        flat_chain = xp.reshape(
            latent_chain,
            (n_steps * n_particles, *latent_chain.shape[2:]),
        )
        flat_auxiliary = _repeat_transform_auxiliary(
            auxiliary,
            n_steps,
            n_particles=n_particles,
        )
        flat_particles, _ = self._inverse_transform(
            flat_chain,
            transform,
            context,
            auxiliary=flat_auxiliary,
        )
        particle_chain = xp.reshape(
            flat_particles,
            (n_steps, n_particles, *flat_particles.shape[1:]),
        )
        return asarray(particle_chain, xp=xp, like=like)

    @staticmethod
    def _combine_with_log_det(result: Any, log_det: Array | None):
        if log_det is None:
            return result
        xp = namespace(result, log_det)
        result_arr = xp.asarray(result)
        log_det_arr = xp.asarray(log_det)
        return result_arr + log_det_arr
