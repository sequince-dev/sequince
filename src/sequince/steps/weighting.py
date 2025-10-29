from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Optional

import numpy as np
from scipy.special import logsumexp

from ..utils import _call_with_optional_context
from .base import SMCStep, StepContext

if TYPE_CHECKING:
    from ..state import ParticleState


LogLikelihoodFn = Callable[..., np.ndarray]
TemperatureSchedule = Callable[..., float]

logger = logging.getLogger(__name__)


def _as_float_array(value: Any) -> np.ndarray:
    """Convert a value to a 1d float array."""
    array = np.asarray(value, dtype=float)
    if array.ndim != 1:
        raise ValueError("Weight updates must be one-dimensional.")
    return array


def _ess_ratio_from_log_weights(log_weights: np.ndarray) -> float:
    """Return ESS ratio (ESS / n) given log weights."""
    log_norm = logsumexp(log_weights)
    if np.isfinite(log_norm):
        weights = np.exp(log_weights - log_norm)
    else:
        n = log_weights.shape[0]
        weights = np.full(n, 1.0 / n)
    ess = 1.0 / np.sum(np.square(weights))
    return ess / log_weights.shape[0]


class LogLikelihoodWeighting(SMCStep):
    """Increment log weights using a user-supplied log-likelihood function.

    The callable can have signature ``(particles,)`` or
    ``(particles, context)``; the ``StepContext`` instance is
    passed automatically when required.
    """

    def __init__(
        self,
        log_likelihood_fn: Callable[..., Any],
        *,
        normalize_after: bool = True,
    ):
        self._log_likelihood_fn = log_likelihood_fn
        self._normalize_after = normalize_after

    def apply(
        self,
        state: ParticleState,
        context: StepContext,
        rng: np.random.Generator,
    ) -> ParticleState:
        log_like = _call_with_optional_context(
            self._log_likelihood_fn,
            (state.particles,),
            context,
        )
        increments = _as_float_array(log_like)
        if increments.shape[0] != state.log_weights.shape[0]:
            raise ValueError(
                "Log-likelihood function must return one value per particle."
            )

        updated_state = state.with_updates(
            log_weights=state.log_weights + increments
        )
        if self._normalize_after:
            updated_state, _ = updated_state.normalize()
        logger.debug(
            "LogLikelihoodWeighting: mean increment %.6f (normalize_after=%s).",
            float(np.mean(increments)),
            self._normalize_after,
        )
        return updated_state


class TemperedWeighting(SMCStep):
    """Apply tempering by scaling log-likelihood increments with a temperature schedule.

    Both the log-likelihood callable and the temperature schedule accept the
    current ``StepContext`` when they require it.
    """

    def __init__(
        self,
        log_likelihood_fn: LogLikelihoodFn,
        temperature_schedule: TemperatureSchedule,
        *,
        normalize_after: bool = True,
        clip_temperature: Optional[tuple[float, float]] = None,
    ):
        self._log_likelihood_fn = log_likelihood_fn
        self._schedule = temperature_schedule
        self._normalize_after = normalize_after
        self._clip = clip_temperature

    def apply(
        self,
        state: ParticleState,
        context: StepContext,
        rng: np.random.Generator,
    ) -> ParticleState:
        temperature = float(
            _call_with_optional_context(
                self._schedule,
                (context.t,),
                context,
            )
        )
        if self._clip is not None:
            lo, hi = self._clip
            temperature = float(np.clip(temperature, lo, hi))

        log_like = _call_with_optional_context(
            self._log_likelihood_fn,
            (state.particles,),
            context,
        )
        increments = _as_float_array(log_like) * temperature
        if increments.shape[0] != state.log_weights.shape[0]:
            raise ValueError(
                "Temperature schedule must align with particle count."
            )

        logger.debug(
            "TemperedWeighting: temperature %.4f applied; mean increment %.6f.",
            temperature,
            float(np.mean(increments)),
        )
        updated_state = state.with_updates(
            log_weights=state.log_weights + increments
        )
        if self._normalize_after:
            updated_state, _ = updated_state.normalize()
        return updated_state


class FixedTemperedWeighting(SMCStep):
    """Apply tempering with a fixed temperature increment."""

    def __init__(
        self,
        log_likelihood_fn: LogLikelihoodFn,
        *,
        temperature_increment: float,
        normalize_after: bool = True,
    ):
        if temperature_increment < 0.0:
            raise ValueError("temperature_increment must be non-negative.")
        self._log_likelihood_fn = log_likelihood_fn
        self._temperature_increment = temperature_increment
        self._normalize_after = normalize_after

    def apply(
        self,
        state: ParticleState,
        context: StepContext,
        rng: np.random.Generator,
    ) -> ParticleState:
        log_like = _call_with_optional_context(
            self._log_likelihood_fn,
            (state.particles,),
            context,
        )
        increments = _as_float_array(log_like) * self._temperature_increment
        if increments.shape[0] != state.log_weights.shape[0]:
            raise ValueError(
                "Log-likelihood function must return one value per particle."
            )
        metadata = dict(state.metadata)
        current_temp = metadata.get("temperature", 0.0)
        metadata["temperature"] = current_temp + self._temperature_increment
        updated_state = state.with_updates(
            log_weights=state.log_weights + increments, metadata=metadata
        )
        context.static["temperature"] = metadata["temperature"]
        context.static["temperature_delta"] = self._temperature_increment
        if self._normalize_after:
            updated_state, _ = updated_state.normalize()
        logger.debug(
            "FixedTemperedWeighting: increment %.4f -> temperature %.4f.",
            self._temperature_increment,
            metadata["temperature"],
        )
        return updated_state


class AdaptiveTemperedWeighting(SMCStep):
    """Adaptively choose the next temperature using an ESS target."""

    def __init__(
        self,
        log_likelihood_fn: LogLikelihoodFn,
        *,
        target_ess_ratio: float = 0.7,
        max_temperature: float = 1.0,
        min_delta: float = 1e-4,
        normalize_after: bool = True,
        max_bisection_iter: int = 30,
    ):
        if not (0.0 < target_ess_ratio <= 1.0):
            raise ValueError("target_ess_ratio must be in (0, 1].")
        if max_temperature <= 0.0:
            raise ValueError("max_temperature must be positive.")

        self._log_likelihood_fn = log_likelihood_fn
        self._target_ess_ratio = target_ess_ratio
        self._max_temperature = max_temperature
        self._min_delta = min_delta
        self._normalize_after = normalize_after
        self._max_bisection_iter = max_bisection_iter

    def apply(
        self,
        state: ParticleState,
        context: StepContext,
        rng: np.random.Generator,
    ) -> ParticleState:
        current_temp = state.metadata.get("temperature", 0.0)
        remaining = self._max_temperature - current_temp
        if remaining <= 0.0:
            context.static["temperature"] = current_temp
            context.static["temperature_delta"] = 0.0
            context.request_stop()
            return state

        log_like = _call_with_optional_context(
            self._log_likelihood_fn,
            (state.particles,),
            context,
        )
        increments = _as_float_array(log_like)

        if np.allclose(increments, 0.0):
            delta = remaining
        else:
            delta = self._find_delta(state.log_weights, increments, remaining)

        new_log_weights = state.log_weights + delta * increments
        metadata = dict(state.metadata)
        metadata["temperature"] = current_temp + delta
        metadata["temperature_delta"] = delta

        updated_state = state.with_updates(
            log_weights=new_log_weights, metadata=metadata
        )
        if self._normalize_after:
            updated_state, _ = updated_state.normalize()

        context.static["temperature"] = metadata["temperature"]
        context.static["temperature_delta"] = delta
        new_ess_ratio = _ess_ratio_from_log_weights(new_log_weights)
        logger.debug(
            "AdaptiveTemperedWeighting: ΔT=%.6f, temperature=%.4f, ESS ratio=%.4f.",
            delta,
            metadata["temperature"],
            new_ess_ratio,
        )
        if metadata["temperature"] >= self._max_temperature or np.isclose(
            metadata["temperature"], self._max_temperature
        ):
            logger.debug(
                "AdaptiveTemperedWeighting reached max temperature %.3f; requesting stop.",
                self._max_temperature,
            )
            context.request_stop()
        return updated_state

    def _find_delta(
        self,
        base_log_weights: np.ndarray,
        increments: np.ndarray,
        max_delta: float,
    ) -> float:
        low = 0.0
        high = max_delta

        base_ratio = _ess_ratio_from_log_weights(base_log_weights)
        if base_ratio <= self._target_ess_ratio:
            return max(self._min_delta, min(high, self._min_delta))

        hi_ratio = _ess_ratio_from_log_weights(
            base_log_weights + high * increments
        )
        if hi_ratio >= self._target_ess_ratio:
            return high

        for _ in range(self._max_bisection_iter):
            mid = 0.5 * (low + high)
            if high - low <= self._min_delta:
                break
            ratio = _ess_ratio_from_log_weights(
                base_log_weights + mid * increments
            )
            if ratio >= self._target_ess_ratio:
                low = mid
            else:
                high = mid

        delta = max(0.5 * (low + high), self._min_delta)
        return min(delta, max_delta)
