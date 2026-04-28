from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any, Callable

from .._array_api import asarray, full, logsumexp, namespace, to_numpy
from .._typing import Array
from ..utils import _call_with_optional_context
from .base import SMCStep, StepContext

if TYPE_CHECKING:
    try:
        from array_api_compat.typing import Array
    except Exception:  # pragma: no cover
        from typing import Any as Array
    from orng import ArrayRNG

    from ..posterior import TemperedPosterior
    from ..state import ParticleState
else:  # pragma: no cover
    from typing import Any as Array

    from orng import ArrayRNG


IncrementFn = Callable[..., Array | float]
TemperatureSchedule = Callable[..., float]

logger = logging.getLogger(__name__)


def _as_1d_array(value: Any, *, xp, like: Array | None = None) -> Array:
    """Convert a value to a 1d array in the provided namespace."""
    arr = asarray(value, xp=xp, like=like)
    if arr.ndim == 0:
        return xp.reshape(arr, (1,))
    if arr.ndim != 1:
        raise ValueError("Weight updates must be one-dimensional.")
    return arr


def _ess_ratio_from_log_weights(log_weights: Any) -> float:
    """Return ESS ratio (ESS / n) given log weights."""
    xp = namespace(log_weights)
    lw = asarray(log_weights, xp=xp)
    log_norm = logsumexp(lw, xp=xp)
    log_norm_scalar = float(log_norm)
    if math.isfinite(log_norm_scalar):
        weights = xp.exp(lw - log_norm_scalar)
    else:
        n = lw.shape[0]
        weights = full((n,), 1.0 / n, xp=xp, dtype=lw.dtype, like=lw)
    ess = 1.0 / float(to_numpy(xp.sum(xp.square(weights))))
    return ess / lw.shape[0]


def _resolve_increment_fn(
    increment: IncrementFn | "TemperedPosterior",
) -> IncrementFn:
    if hasattr(increment, "log_increment"):
        posterior = increment  # type: ignore[assignment]

        def _wrapped(particles: Array, context: StepContext) -> Array | float:
            log_proposal_override = context.metadata.get(
                "log_proposal_override"
            )
            # type: ignore[attr-defined]
            return posterior.log_increment(  # type: ignore[union-attr]
                particles,
                context,
                log_proposal_override=log_proposal_override,
            )

        return _wrapped
    if callable(increment):
        return increment
    raise TypeError("log_increment_fn must be callable or TemperedPosterior.")


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
        rng: "ArrayRNG",
    ) -> ParticleState:
        xp = namespace(state.log_weights)
        log_like = _call_with_optional_context(
            self._log_likelihood_fn,
            (state.particles,),
            context,
        )
        increments = _as_1d_array(log_like, xp=xp, like=state.log_weights)
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
            float(to_numpy(xp.sum(increments) / increments.shape[0])),
            self._normalize_after,
        )
        return updated_state


class BaseTemperedStep(SMCStep):
    """Base class for tempered weighting steps.

    This provides common functionality for tempering steps that apply log-weight
    increments scaled by a temperature schedule (e.g. ``TemperedWeighting`` and
    ``AdaptiveTemperedWeighting``). It does not implement the logic for determining
    the temperature schedule itself, which is left to subclasses.
    """

    def __init__(
        self,
        log_increment_fn: IncrementFn | "TemperedPosterior",
        *,
        normalize_after: bool = True,
    ):
        self._log_increment_fn = _resolve_increment_fn(log_increment_fn)
        self._normalize_after = normalize_after

    def initialize_metadata(
        self, state: ParticleState, context: StepContext
    ) -> ParticleState:
        """Initialize metadata related to tempering."""
        if "temperature" not in context.metadata:
            logger.debug("Initializing tempering metadata in context to 0.0.")
            context.metadata["temperature"] = 0.0
            state_metadata = dict(state.metadata)
            state_metadata["temperature"] = 0.0
            state = state.with_updates(metadata=state_metadata)
        if "temperature_delta" not in context.metadata:
            context.metadata["temperature_delta"] = 0.0
        return state

    def apply(self, state, context, rng):
        raise NotImplementedError(
            "BaseTemperedStep subclasses must implement apply()."
        )


class TemperedWeighting(BaseTemperedStep):
    """Apply tempering by scaling log-weight increments with a temperature schedule.

    Both the increment callable and the temperature schedule accept the
    current ``StepContext`` when they require it.
    """

    def __init__(
        self,
        log_increment_fn: IncrementFn | "TemperedPosterior",
        temperature_schedule: TemperatureSchedule,
        *,
        normalize_after: bool = True,
        clip_temperature: tuple[float, float] | None = None,
    ):
        super().__init__(log_increment_fn, normalize_after=normalize_after)
        self._schedule = temperature_schedule
        self._clip = clip_temperature

    def apply(
        self,
        state: ParticleState,
        context: StepContext,
        rng: "ArrayRNG",
    ) -> ParticleState:
        state = self.initialize_metadata(state, context)
        temperature = float(
            _call_with_optional_context(
                self._schedule,
                (context.t,),
                context,
            )
        )
        if self._clip is not None:
            lo, hi = self._clip
            temperature = float(max(lo, min(hi, temperature)))

        log_increments = _call_with_optional_context(
            self._log_increment_fn,
            (state.particles,),
            context,
        )
        xp = namespace(state.log_weights)
        increments = (
            _as_1d_array(log_increments, xp=xp, like=state.log_weights)
            * temperature
        )
        if increments.shape[0] != state.log_weights.shape[0]:
            raise ValueError(
                "Increment function must return one value per particle."
            )

        logger.debug(
            "TemperedWeighting: temperature %.4f applied; mean increment %.6f.",
            temperature,
            float(to_numpy(xp.sum(increments) / increments.shape[0])),
        )
        updated_state = state.with_updates(
            log_weights=state.log_weights + increments
        )
        if self._normalize_after:
            updated_state, _ = updated_state.normalize()
        return updated_state


class FixedTemperedWeighting(BaseTemperedStep):
    """Apply tempering with a fixed temperature increment."""

    def __init__(
        self,
        log_increment_fn: IncrementFn | "TemperedPosterior",
        *,
        temperature_increment: float,
        normalize_after: bool = True,
    ):
        if temperature_increment < 0.0:
            raise ValueError("temperature_increment must be non-negative.")
        super().__init__(log_increment_fn, normalize_after=normalize_after)
        self._temperature_increment = temperature_increment

    def apply(
        self,
        state: ParticleState,
        context: StepContext,
        rng: "ArrayRNG",
    ) -> ParticleState:
        state = self.initialize_metadata(state, context)
        log_increments = _call_with_optional_context(
            self._log_increment_fn,
            (state.particles,),
            context,
        )
        xp = namespace(state.log_weights)
        increments = (
            _as_1d_array(log_increments, xp=xp, like=state.log_weights)
            * self._temperature_increment
        )
        if increments.shape[0] != state.log_weights.shape[0]:
            raise ValueError(
                "Increment function must return one value per particle."
            )
        metadata = dict(state.metadata)
        current_temp = metadata.get("temperature", 0.0)
        metadata["temperature"] = current_temp + self._temperature_increment
        updated_state = state.with_updates(
            log_weights=state.log_weights + increments, metadata=metadata
        )
        context.metadata["temperature"] = metadata["temperature"]
        context.metadata["temperature_delta"] = self._temperature_increment
        if self._normalize_after:
            updated_state, _ = updated_state.normalize()
        logger.debug(
            "FixedTemperedWeighting: increment %.4f -> temperature %.4f.",
            self._temperature_increment,
            metadata["temperature"],
        )
        return updated_state


class AdaptiveTemperedWeighting(BaseTemperedStep):
    """Adaptively choose the next temperature using an ESS target.

    The temperature increment is chosen such that the effective sample size
    (ESS) ratio after weighting reaches the specified target value
    (``target_ess_ratio``), up to the maximum temperature (``max_temperature``).
    If the maximum temperature is reached, the SMC run is requested to stop.

    Parameters
    ----------
    log_increment_fn:
        A callable or ``TemperedPosterior`` providing the log-weight increments.
    target_ess_ratio:
        The target ESS ratio (ESS / n) to achieve after weighting; must be in
        the interval (0, 1].
    max_temperature:
        The maximum temperature to reach; once this temperature is reached,
        the SMC run is requested to stop.
    min_delta:
        The minimum temperature increment to apply at each step.
    normalize_after:
        Whether to normalize the particle weights after updating.
    max_bisection_iter:
        The maximum number of bisection iterations to perform when finding
        the temperature increment.
    """

    def __init__(
        self,
        log_increment_fn: IncrementFn | "TemperedPosterior",
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

        super().__init__(log_increment_fn, normalize_after=normalize_after)

        self._target_ess_ratio = target_ess_ratio
        self._max_temperature = max_temperature
        self._min_delta = min_delta
        self._max_bisection_iter = max_bisection_iter

    def apply(
        self,
        state: ParticleState,
        context: StepContext,
        rng: "ArrayRNG",
    ) -> ParticleState:
        state = self.initialize_metadata(state, context)
        current_temp = context.metadata.get("temperature", 0.0)
        remaining = self._max_temperature - current_temp
        if remaining <= 0.0:
            context.metadata["temperature"] = current_temp
            context.metadata["temperature_delta"] = 0.0
            logger.debug(
                "AdaptiveTemperedWeighting reached max temperature %.3f; requesting stop.",
                self._max_temperature,
            )
            context.request_stop()
            return state

        xp = namespace(state.log_weights)
        log_increments = _call_with_optional_context(
            self._log_increment_fn,
            (state.particles,),
            context,
        )
        increments = _as_1d_array(
            log_increments, xp=xp, like=state.log_weights
        )

        if _max_abs(increments, xp=xp) <= 1e-12:
            delta = remaining
        else:
            delta = self._find_delta(state.log_weights, increments, remaining)

        new_log_weights = state.log_weights + delta * increments
        metadata = dict(state.metadata)
        metadata["temperature"] = current_temp + delta

        updated_state = state.with_updates(
            log_weights=new_log_weights, metadata=metadata
        )
        if self._normalize_after:
            updated_state, _ = updated_state.normalize()

        context.metadata["temperature"] = metadata["temperature"]
        context.metadata["temperature_delta"] = delta
        new_ess_ratio = _ess_ratio_from_log_weights(new_log_weights)
        logger.debug(
            "AdaptiveTemperedWeighting: ΔT=%.6f, temperature=%.4f, ESS ratio=%.4f.",
            delta,
            metadata["temperature"],
            new_ess_ratio,
        )
        if (
            metadata["temperature"] >= self._max_temperature
            or abs(metadata["temperature"] - self._max_temperature) <= 1e-9
        ):
            logger.debug(
                "AdaptiveTemperedWeighting reached max temperature %.3f; requesting stop.",
                self._max_temperature,
            )
            context.request_stop()
        return updated_state

    def _find_delta(
        self,
        base_log_weights: Array,
        increments: Array,
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


def _max_abs(array: Array, *, xp) -> float:
    return float(to_numpy(xp.max(xp.abs(array))))
