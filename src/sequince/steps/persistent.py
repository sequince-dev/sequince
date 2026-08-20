"""Persistent particle archival and fixed-temperature importance weighting."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import TYPE_CHECKING, Any

from .._typing import Array, ArrayLike
from ..array_api import asarray, logsumexp, namespace
from ..state import ParticleState, PersistentParticleState
from ..target import TemperedTarget
from ..utils import _validate_log_normalizer
from .base import SMCStep
from .context import StepContext

if TYPE_CHECKING:
    from orng import RandomGenerator

__all__ = [
    "PersistParticles",
    "PersistentTemperedWeighting",
    "persistent_log_mixture_denominator",
    "persistent_log_weights",
]


def _persistent_arrays(
    log_increment: Array,
    temperatures: ArrayLike,
    log_evidences: Array,
    *,
    xp: Any | None,
) -> tuple[Array, Array, Array, Any]:
    """Convert and validate the arrays used by persistent weighting."""
    increments = asarray(log_increment, xp=xp)
    xp = namespace(increments)
    temperatures = asarray(
        temperatures,
        xp=xp,
        dtype=increments.dtype,
        like=increments,
    )
    log_evidences = asarray(
        log_evidences,
        xp=xp,
        dtype=increments.dtype,
        like=increments,
    )

    if increments.ndim != 1:
        raise ValueError("log_increment must be one-dimensional.")
    if temperatures.ndim != 1:
        raise ValueError("temperatures must be one-dimensional.")
    if log_evidences.ndim != 1:
        raise ValueError("log_evidences must be one-dimensional.")
    if temperatures.shape[0] == 0:
        raise ValueError("temperature history must not be empty.")
    if temperatures.shape[0] != log_evidences.shape[0]:
        raise ValueError(
            "temperatures and log_evidences must have the same length."
        )

    return increments, temperatures, log_evidences, xp


def persistent_log_mixture_denominator(
    log_increment: Array,
    temperatures: ArrayLike,
    log_evidences: Array,
    *,
    xp: Any | None = None,
) -> Array:
    """Compute the log density of the equally weighted historical mixture.

    The common initial-target density is cancelled, leaving, for particle
    ``j``, ``logmeanexp(temperatures * log_increment[j] - log_evidences)``.
    """
    increments, temperatures, log_evidences, xp = _persistent_arrays(
        log_increment,
        temperatures,
        log_evidences,
        xp=xp,
    )
    terms = (
        increments[:, None] * temperatures[None, :] - log_evidences[None, :]
    )
    n_targets = asarray(
        temperatures.shape[0],
        xp=xp,
        dtype=increments.dtype,
        like=increments,
    )
    return logsumexp(terms, axis=1, xp=xp) - xp.log(n_targets)


def persistent_log_weights(
    log_increment: Array,
    temperatures: ArrayLike,
    log_evidences: Array,
    temperature: ArrayLike,
    *,
    xp: Any | None = None,
) -> Array:
    """Compute unnormalised persistent importance weights."""
    increments = asarray(log_increment, xp=xp)
    xp = namespace(increments)
    candidate_temperature = asarray(
        temperature,
        xp=xp,
        dtype=increments.dtype,
        like=increments,
    )
    if candidate_temperature.ndim != 0:
        raise ValueError("temperature must be scalar.")

    log_denominator = persistent_log_mixture_denominator(
        increments,
        temperatures,
        log_evidences,
        xp=xp,
    )
    return candidate_temperature * increments - log_denominator


class PersistParticles(SMCStep):
    """Retain each completed active particle population in a persistent archive."""

    def __init__(self) -> None:
        self.persistent_state: PersistentParticleState | None = None

    def apply(
        self,
        state: ParticleState,
        context: StepContext,
        rng: "RandomGenerator",
    ) -> tuple[PersistentParticleState, StepContext, None]:
        del rng
        if isinstance(state, PersistentParticleState):
            raise TypeError(
                "PersistParticles requires an active ParticleState."
            )

        # Match BaseTemperedStep: context is authoritative, active-state metadata
        # is the fallback, and an uninitialized run starts at beta = 0.
        temperature = context.metadata.get(
            "temperature", state.metadata.get("temperature", 0.0)
        )
        if self.persistent_state is None:
            archived = PersistentParticleState.from_state(
                state,
                temperature=temperature,
                log_evidence=state.log_evidence,
            )
        else:
            archived = self.persistent_state.append(
                state,
                temperature=temperature,
                log_evidence=state.log_evidence,
            )
        self.persistent_state = archived
        return archived, context, None

    def state_dict(self) -> dict[str, Any]:
        """Return the archive in checkpoint-friendly mapping form."""
        return {
            "persistent_state": (
                None
                if self.persistent_state is None
                else self.persistent_state.to_dict()
            )
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore an archive produced by :meth:`state_dict`."""
        payload = state.get("persistent_state")
        if payload is None:
            self.persistent_state = None
            return
        if not isinstance(payload, Mapping):
            raise TypeError("PersistParticles state must contain a mapping.")
        self.persistent_state = PersistentParticleState.from_dict(payload)


class PersistentTemperedWeighting(SMCStep):
    """Weight a persistent particle pool directly toward a fixed temperature."""

    def __init__(
        self,
        target: TemperedTarget,
        *,
        temperature: ArrayLike,
        refresh_components: Collection[str] = (),
    ) -> None:
        if not isinstance(target, TemperedTarget):
            raise TypeError(
                "PersistentTemperedWeighting requires a TemperedTarget."
            )
        self._target = target
        self._temperature = temperature
        self._refresh_components = frozenset(refresh_components)

    def apply(
        self,
        state: ParticleState,
        context: StepContext,
        rng: "RandomGenerator",
    ) -> tuple[PersistentParticleState, StepContext, dict[str, Any]]:
        del rng
        if not isinstance(state, PersistentParticleState):
            raise TypeError(
                "PersistentTemperedWeighting requires a "
                "PersistentParticleState."
            )

        evaluation = self._target.prepare_evaluation(
            state.particles,
            context,
            evaluation=state.evaluation,
            refresh_components=self._refresh_components,
        )
        log_increment = self._target.log_increment(
            state.particles,
            context,
            evaluation=evaluation,
        )
        xp = namespace(state.log_weights)
        log_increment = asarray(log_increment, xp=xp, like=state.log_weights)
        if log_increment.ndim == 0:
            log_increment = xp.broadcast_to(
                log_increment, state.log_weights.shape
            )
        if log_increment.shape != state.log_weights.shape:
            raise ValueError(
                "Target log increment must return one value per particle."
            )

        unnormalized = persistent_log_weights(
            log_increment,
            state.temperatures,
            state.log_evidences,
            self._temperature,
            xp=xp,
        )
        log_norm = logsumexp(unnormalized, xp=xp)
        _validate_log_normalizer(log_norm, xp=xp)
        particle_count = asarray(
            state.n_particles,
            xp=xp,
            dtype=unnormalized.dtype,
            like=unnormalized,
        )
        log_evidence = log_norm - xp.log(particle_count)
        normalized = unnormalized - log_norm

        current_temperature = context.metadata.get(
            "temperature", state.metadata.get("temperature", 0.0)
        )
        temperature = asarray(
            self._temperature,
            xp=xp,
            dtype=unnormalized.dtype,
            like=unnormalized,
        )
        if temperature.ndim != 0:
            raise ValueError("temperature must be scalar.")
        temperature_delta = temperature - asarray(
            current_temperature,
            xp=xp,
            dtype=unnormalized.dtype,
            like=unnormalized,
        )
        context = context.with_metadata(
            temperature=temperature,
            temperature_delta=temperature_delta,
        )
        metadata = dict(state.metadata)
        metadata.update(
            temperature=temperature,
            temperature_delta=temperature_delta,
        )
        weighted = state.with_updates(
            log_weights=normalized,
            log_evidence=log_evidence,
            evaluation=evaluation,
            metadata=metadata,
        )
        return (
            weighted,
            context,
            {
                "temperature": temperature,
                "temperature_delta": temperature_delta,
                "persistent_log_weights": unnormalized,
            },
        )
