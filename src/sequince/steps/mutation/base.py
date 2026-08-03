from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Callable, Literal, cast

import numpy as np

from ..._array_api import asarray, namespace, to_numpy
from ..._typing import Array
from ...transforms import ParticleTransform
from ..base import SMCStep
from ..context import LOG_PROPOSAL_OVERRIDE_KEY, StepContext

if TYPE_CHECKING:
    from ...posterior import PosteriorEvaluation, TemperedPosterior

LogProbFn = Callable[..., Array | float]
PARTICLE_TRANSFORM_KEY = "particle_transform"
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
        log_prob_fn: Callable[..., Array | float] | "TemperedPosterior",
        *,
        transform: ParticleTransform | None = None,
        transform_refit: TransformRefit = "always",
        store_chains: bool = False,
        log_proposal_override_key: str = LOG_PROPOSAL_OVERRIDE_KEY,
    ):
        super().__init__()
        self._posterior = (
            log_prob_fn
            if hasattr(log_prob_fn, "evaluate")
            and hasattr(log_prob_fn, "log_prob")
            else None
        )
        self._log_prob_fn = (
            None
            if self._posterior is not None
            else cast(LogProbFn, log_prob_fn)
        )
        self._log_proposal_override_key = log_proposal_override_key
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
        self._prepared_context: StepContext | None = None
        self._store_chains = store_chains
        self._chain: Array | None = None
        self._evaluation_cache: dict[bytes, PosteriorEvaluation] = {}

    def _evaluate_target(
        self,
        particles: Array,
        context: StepContext,
        *,
        evaluation: "PosteriorEvaluation | None" = None,
    ) -> tuple[Any, "PosteriorEvaluation | None"]:
        if self._posterior is None:
            if self._log_prob_fn is None:  # pragma: no cover
                raise RuntimeError("Mutation target is not configured.")
            return call_log_prob(self._log_prob_fn, particles, context), None
        log_proposal_override = context.metadata.get(
            self._log_proposal_override_key
        )
        if evaluation is None:
            evaluation = self._posterior.evaluate(
                particles,
                context,
                log_proposal_override=log_proposal_override,
            )
        return (
            self._posterior.log_prob(
                particles,
                context,
                evaluation=evaluation,
                log_proposal_override=log_proposal_override,
            ),
            evaluation,
        )

    @staticmethod
    def _particle_rows(particles: Array) -> np.ndarray:
        values = to_numpy(particles)
        return values[np.newaxis, ...] if values.ndim == 1 else values

    def _cache_evaluation(
        self,
        particles: Array,
        evaluation: "PosteriorEvaluation | None",
    ) -> None:
        if evaluation is None:
            return
        for index, row in enumerate(self._particle_rows(particles)):
            self._evaluation_cache[row.tobytes()] = evaluation.take(
                slice(index, index + 1)
            )

    def _cached_evaluation(
        self, particles: Array
    ) -> "PosteriorEvaluation | None":
        if self._posterior is None:
            return None
        from ...posterior import PosteriorEvaluation

        items = []
        for row in self._particle_rows(particles):
            item = self._evaluation_cache.get(row.tobytes())
            if item is None:
                return None
            items.append(item)
        return PosteriorEvaluation.concatenate(items)

    def _evaluate_target_cached(
        self,
        particles: Array,
        context: StepContext,
    ) -> tuple[Any, "PosteriorEvaluation | None"]:
        evaluation = self._cached_evaluation(particles)
        result, evaluation = self._evaluate_target(
            particles,
            context,
            evaluation=evaluation,
        )
        self._cache_evaluation(particles, evaluation)
        return result, evaluation

    @property
    def store_chains(self) -> bool:
        return self._store_chains

    @store_chains.setter
    def store_chains(self, value: bool) -> None:
        self._store_chains = bool(value)

    def state_dict(self) -> dict[str, Any]:
        if self._prepared_transform is None:
            return {}
        return {
            "prepared_transform": self._prepared_transform.state_dict(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        prepared_state = state.get("prepared_transform")
        if prepared_state is None:
            self._prepared_transform = None
            self._prepared_context = None
            return
        if not isinstance(prepared_state, Mapping):
            raise TypeError(
                "MutationStep prepared transform state must be a mapping."
            )
        if self._transform is None:
            raise ValueError(
                "Checkpoint contains transform state but no transform is "
                "configured."
            )
        prepared_transform = self._transform.clone()
        prepared_transform.load_state_dict(prepared_state)
        self._prepared_transform = prepared_transform
        self._prepared_context = None

    def _prepare_transform(
        self, particles: Array, context: StepContext
    ) -> ParticleTransform | None:
        if self._transform is None:
            return None
        triggered = (
            bool(self._transform_refit(particles, context))
            if callable(self._transform_refit)
            else False
        )
        should_fit = (
            self._transform_refit == "always"
            or self._prepared_transform is None
            or triggered
            or (
                self._transform_refit == "per-context"
                and self._prepared_context is not context
            )
        )
        if should_fit:
            xp = namespace(particles)
            transform = self._transform.clone()
            transform.fit(asarray(particles, xp=xp), context=context)
            self._prepared_transform = transform
            self._prepared_context = context
        else:
            cached_transform = self._prepared_transform
            if cached_transform is None:  # pragma: no cover - guarded above
                raise RuntimeError("Prepared transform cache is empty.")
            transform = cached_transform
        context.metadata[PARTICLE_TRANSFORM_KEY] = transform
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

    def _inverse_chain(
        self,
        chain: Array,
        transform: ParticleTransform | None,
        context: StepContext,
        *,
        like: Array,
    ) -> Array:
        """Map a stored mutation chain back into particle coordinates."""
        xp = namespace(like)
        latent_chain = asarray(chain, xp=xp, like=like)
        if transform is None:
            return latent_chain
        original_shape = latent_chain.shape
        flattened = xp.reshape(latent_chain, (-1, original_shape[-1]))
        particles, _ = self._inverse_transform(flattened, transform, context)
        return xp.reshape(particles, original_shape)

    @staticmethod
    def _combine_with_log_det(result: Any, log_det: Array | None):
        if log_det is None:
            return result
        xp = namespace(result, log_det)
        result_arr = xp.asarray(result)
        log_det_arr = xp.asarray(log_det)
        return result_arr + log_det_arr
