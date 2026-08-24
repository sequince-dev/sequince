"""Persistent particle archival and tempered importance weighting.

See: https://arxiv.org/abs/2407.20722
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import TYPE_CHECKING, Any

from .._typing import Array, ArrayLike
from ..array_api import asarray, logsumexp, namespace
from ..state import ParticleState, PersistentParticleState
from ..stats import ess_from_log_weights
from ..target import TemperedTarget
from ..utils import _validate_log_normalizer, bisect_monotone_decreasing
from .base import SMCStep
from .context import StepContext

if TYPE_CHECKING:
    from orng import RandomGenerator

__all__ = [
    "PersistParticles",
    "PersistentAdaptiveTemperedWeighting",
    "PersistentTemperedWeighting",
    "persistent_log_mixture_denominator",
    "persistent_log_weights",
]


def _persistent_arrays(
    log_increment: Array,
    temperatures: ArrayLike,
    log_normalizing_constants: Array,
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
    log_normalizing_constants = asarray(
        log_normalizing_constants,
        xp=xp,
        dtype=increments.dtype,
        like=increments,
    )

    if increments.ndim != 1:
        raise ValueError("log_increment must be one-dimensional.")
    if temperatures.ndim != 1:
        raise ValueError("temperatures must be one-dimensional.")
    if log_normalizing_constants.ndim != 1:
        raise ValueError("log_normalizing_constants must be one-dimensional.")
    if temperatures.shape[0] == 0:
        raise ValueError("temperature history must not be empty.")
    if temperatures.shape[0] != log_normalizing_constants.shape[0]:
        raise ValueError(
            "temperatures and log_normalizing_constants must have the same "
            "length."
        )

    return increments, temperatures, log_normalizing_constants, xp


def persistent_log_mixture_denominator(
    log_increment: Array,
    temperatures: ArrayLike,
    log_normalizing_constants: Array,
    *,
    xp: Any | None = None,
) -> Array:
    """Compute the log density of the equally weighted historical mixture.

    The common initial-target density is cancelled, leaving, for particle
    ``j``, ``logmeanexp(temperatures * log_increment[j] -
    log_normalizing_constants)``.
    """
    increments, temperatures, log_normalizing_constants, xp = (
        _persistent_arrays(
            log_increment,
            temperatures,
            log_normalizing_constants,
            xp=xp,
        )
    )
    terms = (
        increments[:, None] * temperatures[None, :]
        - log_normalizing_constants[None, :]
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
    log_normalizing_constants: Array,
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
        log_normalizing_constants,
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
                log_normalizing_constant=state.log_normalizing_constant,
            )
        else:
            archived = self.persistent_state.append(
                state,
                temperature=temperature,
                log_normalizing_constant=state.log_normalizing_constant,
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


class BasePersistentTemperedWeighting(SMCStep):
    """Shared target evaluation and weighting for persistent tempering."""

    def __init__(
        self,
        target: TemperedTarget,
        *,
        refresh_components: Collection[str] = (),
    ) -> None:
        if not isinstance(target, TemperedTarget):
            raise TypeError(
                f"{type(self).__name__} requires a TemperedTarget."
            )
        self._target = target
        self._refresh_components = frozenset(refresh_components)

    def _require_persistent_state(
        self, state: ParticleState
    ) -> PersistentParticleState:
        if not isinstance(state, PersistentParticleState):
            raise TypeError(
                f"{type(self).__name__} requires a PersistentParticleState."
            )
        return state

    def _prepare_log_increment(
        self,
        state: PersistentParticleState,
        context: StepContext,
    ) -> tuple[Any, Array]:
        """Prepare one cached target evaluation and its log increment."""
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
        return evaluation, log_increment

    def _weight(
        self,
        state: PersistentParticleState,
        evaluation: Any,
        log_increment: Array,
        temperature: ArrayLike,
        context: StepContext,
        *,
        log_denominator: Array | None = None,
    ) -> tuple[PersistentParticleState, StepContext, dict[str, Any]]:
        """Weight the persistent archive at a chosen temperature.

        Normalize the resulting importance weights, calculate the direct
        persistent normalizing-constant estimate, and update the state and
        context temperature metadata. A precomputed mixture denominator may be
        supplied by adaptive temperature selection.
        """
        xp = namespace(state.log_weights)
        temperature = asarray(
            temperature,
            xp=xp,
            dtype=log_increment.dtype,
            like=log_increment,
        )
        if temperature.ndim != 0:
            raise ValueError("temperature must be scalar.")
        if log_denominator is None:
            log_denominator = persistent_log_mixture_denominator(
                log_increment,
                state.temperatures,
                state.log_normalizing_constants,
                xp=xp,
            )
        unnormalized = temperature * log_increment - log_denominator
        log_norm = logsumexp(unnormalized, xp=xp)
        _validate_log_normalizer(log_norm, xp=xp)
        particle_count = asarray(
            state.n_particles,
            xp=xp,
            dtype=unnormalized.dtype,
            like=unnormalized,
        )
        log_normalizing_constant = log_norm - xp.log(particle_count)
        normalized = unnormalized - log_norm
        effective_sample_size = ess_from_log_weights(unnormalized)

        current_temperature = context.metadata.get(
            "temperature", state.metadata.get("temperature", 0.0)
        )
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
            effective_sample_size=effective_sample_size,
        )
        weighted = state.with_updates(
            log_weights=normalized,
            log_normalizing_constant=log_normalizing_constant,
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
                "effective_sample_size": effective_sample_size,
            },
        )


class PersistentTemperedWeighting(BasePersistentTemperedWeighting):
    """Weight a persistent particle pool directly toward a fixed temperature."""

    def __init__(
        self,
        target: TemperedTarget,
        *,
        temperature: ArrayLike,
        refresh_components: Collection[str] = (),
    ) -> None:
        super().__init__(target, refresh_components=refresh_components)
        self._temperature = temperature

    def apply(
        self,
        state: ParticleState,
        context: StepContext,
        rng: "RandomGenerator",
    ) -> tuple[PersistentParticleState, StepContext, dict[str, Any]]:
        del rng
        state = self._require_persistent_state(state)
        evaluation, log_increment = self._prepare_log_increment(state, context)
        return self._weight(
            state,
            evaluation,
            log_increment,
            self._temperature,
            context,
        )


class PersistentAdaptiveTemperedWeighting(BasePersistentTemperedWeighting):
    """Choose a persistent target temperature using absolute archive ESS.

    The historical deterministic-mixture density is evaluated once, then the
    temperature is bisected over the interval from the current temperature to
    one. If the requested ESS cannot be attained while advancing, the current
    temperature is repeated so that another batch can be added to the archive.
    Once temperature one is selected, the weighting requests an immediate
    runner stop. When ``target_total_ess`` is configured, temperature-one
    batches are instead accumulated until the complete archive reaches that
    ESS.

    Parameters
    ----------
    target:
        Tempered target defining the persistent importance weights.
    target_ess:
        Desired effective sample size while selecting temperatures below one.
    target_total_ess:
        Optional effective sample size required before stopping at temperature
        one. If omitted, stop when temperature one is first selected.
    temperature_tolerance:
        Stop bisection once the temperature bracket is no wider than this.
    max_bisection_iter:
        Maximum number of bisection iterations.
    refresh_components:
        Cached target-evaluation components to refresh before weighting.
    """

    def __init__(
        self,
        target: TemperedTarget,
        *,
        target_ess: float = 512,
        target_total_ess: float | None = None,
        temperature_tolerance: float = 1e-6,
        max_bisection_iter: int = 30,
        refresh_components: Collection[str] = (),
    ) -> None:
        super().__init__(target, refresh_components=refresh_components)
        if not target_ess > 0.0:
            raise ValueError("target_ess must be positive.")
        if target_total_ess is not None and not target_total_ess > 0.0:
            raise ValueError("target_total_ess must be positive.")
        if not temperature_tolerance > 0.0:
            raise ValueError("temperature_tolerance must be positive.")
        if max_bisection_iter <= 0:
            raise ValueError("max_bisection_iter must be positive.")
        self._target_ess = target_ess
        self._target_total_ess = target_total_ess
        self._temperature_tolerance = temperature_tolerance
        self._max_bisection_iter = max_bisection_iter

    def apply(
        self,
        state: ParticleState,
        context: StepContext,
        rng: "RandomGenerator",
    ) -> tuple[PersistentParticleState, StepContext, dict[str, Any]]:
        del rng
        state = self._require_persistent_state(state)
        evaluation, log_increment = self._prepare_log_increment(state, context)
        xp = namespace(state.log_weights)
        log_denominator = persistent_log_mixture_denominator(
            log_increment,
            state.temperatures,
            state.log_normalizing_constants,
            xp=xp,
        )
        current_temperature = float(
            context.metadata.get(
                "temperature", state.metadata.get("temperature", 0.0)
            )
        )
        if not 0.0 <= current_temperature <= 1.0:
            raise ValueError("current temperature must be in [0, 1].")

        def effective_sample_size(temperature: Array) -> Array:
            candidate_log_weights = (
                temperature * log_increment - log_denominator
            )
            return ess_from_log_weights(candidate_log_weights)

        selected_terminal = current_temperature == 1.0
        if selected_terminal:
            temperature = 1.0
        else:
            current = asarray(
                current_temperature,
                xp=xp,
                dtype=log_increment.dtype,
                like=log_increment,
            )
            terminal = asarray(
                1.0,
                xp=xp,
                dtype=log_increment.dtype,
                like=log_increment,
            )
            target = asarray(
                self._target_ess,
                xp=xp,
                dtype=log_increment.dtype,
                like=log_increment,
            )
            terminal_ess = float(effective_sample_size(terminal))
            if terminal_ess >= self._target_ess:
                temperature = 1.0
                selected_terminal = True
            else:
                current_ess = float(effective_sample_size(current))
                if current_ess < self._target_ess:
                    temperature = current_temperature
                else:
                    tolerance = asarray(
                        self._temperature_tolerance,
                        xp=xp,
                        dtype=log_increment.dtype,
                        like=log_increment,
                    )
                    low, _ = bisect_monotone_decreasing(
                        effective_sample_size,
                        current,
                        terminal,
                        target,
                        tolerance=tolerance,
                        max_iterations=self._max_bisection_iter,
                    )
                    temperature = float(low)

        weighted, context, output = self._weight(
            state,
            evaluation,
            log_increment,
            temperature,
            context,
            log_denominator=log_denominator,
        )
        target_ess = asarray(
            self._target_ess,
            xp=xp,
            dtype=weighted.log_weights.dtype,
            like=weighted.log_weights,
        )
        metadata = dict(weighted.metadata)
        metadata["target_ess"] = target_ess
        weighted = weighted.with_updates(metadata=metadata)
        context = context.with_metadata(target_ess=target_ess)
        output["target_ess"] = target_ess
        if self._target_total_ess is not None:
            target_total_ess = asarray(
                self._target_total_ess,
                xp=xp,
                dtype=weighted.log_weights.dtype,
                like=weighted.log_weights,
            )
            metadata = dict(weighted.metadata)
            metadata["target_total_ess"] = target_total_ess
            weighted = weighted.with_updates(metadata=metadata)
            context = context.with_metadata(target_total_ess=target_total_ess)
            output["target_total_ess"] = target_total_ess

        if selected_terminal and (
            self._target_total_ess is None
            or float(output["effective_sample_size"]) >= self._target_total_ess
        ):
            context = context.request_stop_immediately()
        return weighted, context, output

    def apply_functional(
        self,
        state: ParticleState,
        context: StepContext,
        random_state: Any,
    ) -> tuple[ParticleState, StepContext, Any, dict[str, Any]]:
        """Reject functional execution because the archive grows dynamically."""
        del state, context, random_state
        raise NotImplementedError(
            "Persistent adaptive tempering supports eager execution only."
        )
