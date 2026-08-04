from __future__ import annotations

import logging
import math
from collections.abc import Collection
from typing import TYPE_CHECKING, Any, Callable, Literal, cast

from .._typing import Array
from ..array_api import asarray, logsumexp, namespace, to_numpy
from ..stats import (
    cess_ratio_from_log_weights_and_increments,
    ess_from_log_weights,
    ess_ratio_from_log_weights,
)
from ..target import SequentialTarget, TemperedTarget
from ..utils import call_with_optional_context
from .base import SMCStep, StepContext

if TYPE_CHECKING:
    from orng import RandomGenerator

    from ..state import ParticleState


IncrementFn = Callable[..., Array | float]
TemperatureSchedule = Callable[..., float]
TargetESSRatio = float | Callable[..., float]

logger = logging.getLogger(__name__)


def _tempering_diagnostics(
    base_log_weights: Array,
    log_increments: Array,
    *,
    delta: float,
    terminal_delta: float,
) -> dict[str, float]:
    """Return diagnostics available while applying a tempering increment."""
    xp = namespace(base_log_weights, log_increments)
    base_log_norm = logsumexp(base_log_weights, xp=xp)
    normalized_base_log_weights = base_log_weights - base_log_norm
    updated_log_weights = base_log_weights + delta * log_increments

    log_evidence_increment = float(
        logsumexp(updated_log_weights, xp=xp) - base_log_norm
    )
    ess = ess_from_log_weights(updated_log_weights)
    terminal_ess = ess_from_log_weights(
        base_log_weights + terminal_delta * log_increments
    )

    scaled_log_ratios = delta * log_increments
    max_log_ratio = xp.max(scaled_log_ratios)
    scaled_ratios = xp.exp(scaled_log_ratios - max_log_ratio)
    base_weights = xp.exp(normalized_base_log_weights)
    mean_ratio = xp.sum(base_weights * scaled_ratios)
    mean_ratio_scalar = float(to_numpy(mean_ratio))
    if mean_ratio_scalar == 0.0 or not math.isfinite(mean_ratio_scalar):
        log_evidence_increment_variance = math.nan
    else:
        centered = scaled_ratios - mean_ratio
        log_evidence_increment_variance = float(
            to_numpy(
                xp.sum(xp.square(base_weights) * xp.square(centered))
                / xp.square(mean_ratio)
            )
        )

    return {
        "log_evidence_increment": log_evidence_increment,
        "log_evidence_increment_variance": (log_evidence_increment_variance),
        "ess_before_resampling": ess,
        "ess_at_max_temperature": terminal_ess,
    }


def evolving_target_ess_ratio(
    start: float,
    end: float,
    *,
    rate: float = 1.0,
    max_temperature: float = 1.0,
) -> Callable[[float], float]:
    """Return a power-law schedule for an evolving target ESS ratio.

    The schedule evolves from ``start`` to ``end`` as the temperature moves
    from zero to ``max_temperature``. Temperatures outside that interval are
    clipped to the corresponding endpoint. Values of ``rate`` greater than
    one delay the transition, while values below one accelerate it.

    Parameters
    ----------
    start:
        Target ESS ratio at zero temperature; must be in ``(0, 1]``.
    end:
        Target ESS ratio at ``max_temperature``; must be in ``(0, 1]``.
    rate:
        Positive exponent controlling the rate of transition.
    max_temperature:
        Positive temperature at which the schedule reaches ``end``.
    """
    if not 0.0 < start <= 1.0:
        raise ValueError("start must be in (0, 1].")
    if not 0.0 < end <= 1.0:
        raise ValueError("end must be in (0, 1].")
    if not rate > 0.0:
        raise ValueError("rate must be positive.")
    if not max_temperature > 0.0:
        raise ValueError("max_temperature must be positive.")

    def schedule(temperature: float) -> float:
        progress = min(max(temperature / max_temperature, 0.0), 1.0)
        return start + (end - start) * progress**rate

    return schedule


def _weights_as_1d_array(
    value: Any, *, xp, like: Array | None = None
) -> Array:
    """Convert weights to a 1d array in the provided namespace.

    Raises
    ------
    ValueError
        If the input cannot be converted to a 1d array.
    """
    arr = asarray(value, xp=xp, like=like)
    if arr.ndim == 0:
        return xp.reshape(arr, (1,))
    if arr.ndim != 1:
        raise ValueError("Weight updates must be one-dimensional.")
    return arr


def _resolve_increment_fn(
    increment: IncrementFn,
) -> IncrementFn:
    if callable(increment):
        return increment
    raise TypeError("log_increment_fn must be callable.")


class IncrementalWeighting(SMCStep):
    """Apply a complete sequential-target log-weight increment.

    A callable can have signature ``(particles,)`` or
    ``(particles, context)``. A ``SequentialTarget`` supplies its current
    complete increment and participates in evaluation caching.
    """

    def __init__(
        self,
        log_increment_fn: IncrementFn | SequentialTarget,
        *,
        normalize_after: bool = True,
    ):
        self._target = (
            log_increment_fn
            if isinstance(log_increment_fn, SequentialTarget)
            else None
        )
        self._log_increment_fn = (
            None
            if self._target is not None
            else _resolve_increment_fn(cast(IncrementFn, log_increment_fn))
        )
        self._normalize_after = normalize_after

    def apply(
        self,
        state: ParticleState,
        context: StepContext,
        rng: "RandomGenerator",
    ) -> ParticleState:
        xp = namespace(state.log_weights)
        if self._target is None:
            if self._log_increment_fn is None:  # pragma: no cover
                raise RuntimeError("Sequential increment is not configured.")
            increment = call_with_optional_context(
                self._log_increment_fn,
                (state.particles,),
                context,
            )
        else:
            evaluation = self._target.prepare_evaluation(
                state.particles,
                context,
                evaluation=state.evaluation,
            )
            increment = self._target.log_increment(
                state.particles,
                context,
                evaluation=evaluation,
            )
            if evaluation is not state.evaluation:
                state = state.with_updates(evaluation=evaluation)
        increments = _weights_as_1d_array(
            increment, xp=xp, like=state.log_weights
        )
        if increments.shape[0] != state.log_weights.shape[0]:
            raise ValueError(
                "Increment function must return one value per particle."
            )

        updated_state = state.with_updates(
            log_weights=state.log_weights + increments
        )
        if self._normalize_after:
            updated_state, _ = updated_state.normalize()
        logger.debug(
            "IncrementalWeighting: mean increment %.6f (normalize_after=%s).",
            float(to_numpy(xp.sum(increments) / increments.shape[0])),
            self._normalize_after,
        )
        return updated_state


class LogLikelihoodWeighting(IncrementalWeighting):
    """Backward-compatible name for callable likelihood increments."""


class BaseTemperedStep(SMCStep):
    """Base class for tempered weighting steps.

    This provides common functionality for tempering steps that apply log-weight
    increments scaled by a temperature schedule (e.g. ``TemperedWeighting`` and
    ``AdaptiveTemperedWeighting``). It does not implement the logic for determining
    the temperature schedule itself, which is left to subclasses.
    """

    def __init__(
        self,
        log_increment_fn: IncrementFn | "TemperedTarget",
        *,
        normalize_after: bool = True,
        refresh_components: Collection[str] = (),
    ):
        self._target = (
            log_increment_fn
            if isinstance(log_increment_fn, TemperedTarget)
            else None
        )
        self._log_increment_fn = (
            None
            if self._target is not None
            else _resolve_increment_fn(cast(IncrementFn, log_increment_fn))
        )
        self._normalize_after = normalize_after
        self._refresh_components = frozenset(refresh_components)

    def evaluate_log_increment(
        self,
        state: ParticleState,
        context: StepContext,
    ) -> tuple[Array | float, ParticleState]:
        """Return increments and attach a missing target evaluation."""
        if self._target is None:
            if self._log_increment_fn is None:  # pragma: no cover
                raise RuntimeError("Tempering increment is not configured.")
            return (
                call_with_optional_context(
                    self._log_increment_fn,
                    (state.particles,),
                    context,
                ),
                state,
            )
        evaluation = self._target.prepare_evaluation(
            state.particles,
            context,
            evaluation=state.evaluation,
            refresh_components=self._refresh_components,
        )
        if evaluation is not state.evaluation:
            state = state.with_updates(evaluation=evaluation)
        return (
            self._target.log_increment(
                state.particles,
                context,
                evaluation=evaluation,
            ),
            state,
        )

    def initialize_metadata(
        self, state: ParticleState, context: StepContext
    ) -> ParticleState:
        """Initialize metadata related to tempering."""
        if "temperature" not in context.metadata:
            initial_temperature = float(state.metadata.get("temperature", 0.0))
            logger.debug(
                "Initializing tempering metadata in context to %.4f.",
                initial_temperature,
            )
            context.metadata["temperature"] = initial_temperature
            state_metadata = dict(state.metadata)
            state_metadata["temperature"] = initial_temperature
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
        log_increment_fn: IncrementFn | "TemperedTarget",
        temperature_schedule: TemperatureSchedule,
        *,
        normalize_after: bool = True,
        clip_temperature: tuple[float, float] | None = None,
        refresh_components: Collection[str] = (),
    ):
        super().__init__(
            log_increment_fn,
            normalize_after=normalize_after,
            refresh_components=refresh_components,
        )
        self._schedule = temperature_schedule
        self._clip = clip_temperature

    def apply(
        self,
        state: ParticleState,
        context: StepContext,
        rng: "RandomGenerator",
    ) -> ParticleState:
        state = self.initialize_metadata(state, context)
        current_temperature = float(context.metadata.get("temperature", 0.0))
        target_temperature = float(
            call_with_optional_context(
                self._schedule,
                (context.t,),
                context,
            )
        )
        if self._clip is not None:
            lo, hi = self._clip
            target_temperature = float(max(lo, min(hi, target_temperature)))
        if target_temperature < current_temperature:
            raise ValueError(
                "temperature_schedule produced a temperature smaller than the "
                "current temperature."
            )
        temperature_delta = target_temperature - current_temperature

        log_increments, state = self.evaluate_log_increment(state, context)
        xp = namespace(state.log_weights)
        increments = (
            _weights_as_1d_array(log_increments, xp=xp, like=state.log_weights)
            * temperature_delta
        )
        if increments.shape[0] != state.log_weights.shape[0]:
            raise ValueError(
                "Increment function must return one value per particle."
            )

        logger.debug(
            "TemperedWeighting: temperature %.4f -> %.4f applied; mean increment %.6f.",
            current_temperature,
            target_temperature,
            float(to_numpy(xp.sum(increments) / increments.shape[0])),
        )
        metadata = dict(state.metadata)
        metadata["temperature"] = target_temperature
        updated_state = state.with_updates(
            log_weights=state.log_weights + increments,
            metadata=metadata,
        )
        if self._normalize_after:
            updated_state, _ = updated_state.normalize()
        context.metadata["temperature"] = target_temperature
        context.metadata["temperature_delta"] = temperature_delta
        return updated_state


class FixedTemperedWeighting(BaseTemperedStep):
    """Apply tempering with a fixed temperature increment."""

    def __init__(
        self,
        log_increment_fn: IncrementFn | "TemperedTarget",
        *,
        temperature_increment: float,
        normalize_after: bool = True,
        refresh_components: Collection[str] = (),
    ):
        if temperature_increment < 0.0:
            raise ValueError("temperature_increment must be non-negative.")
        super().__init__(
            log_increment_fn,
            normalize_after=normalize_after,
            refresh_components=refresh_components,
        )
        self._temperature_increment = temperature_increment

    def apply(
        self,
        state: ParticleState,
        context: StepContext,
        rng: "RandomGenerator",
    ) -> ParticleState:
        state = self.initialize_metadata(state, context)
        log_increments, state = self.evaluate_log_increment(state, context)
        xp = namespace(state.log_weights)
        increments = (
            _weights_as_1d_array(log_increments, xp=xp, like=state.log_weights)
            * self._temperature_increment
        )
        if increments.shape[0] != state.log_weights.shape[0]:
            raise ValueError(
                "Increment function must return one value per particle."
            )
        metadata = dict(state.metadata)
        current_temp = metadata.get("temperature", 0.0)
        target_temperature = current_temp + self._temperature_increment
        max_temperature = float(context.metadata.get("max_temperature", 1.0))
        metadata.update(
            _tempering_diagnostics(
                state.log_weights,
                _weights_as_1d_array(
                    log_increments, xp=xp, like=state.log_weights
                ),
                delta=self._temperature_increment,
                terminal_delta=max(max_temperature - current_temp, 0.0),
            )
        )
        metadata["temperature"] = target_temperature
        metadata["temperature_delta"] = self._temperature_increment
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
    """Adaptively choose the next temperature using an ESS-like target.

    The temperature increment is chosen such that the effective sample size
    (ESS) ratio after weighting, or the conditional ESS (CESS) ratio implied by
    a proposed increment, reaches the specified target value
    (``target_ess_ratio``), up to the maximum temperature
    (``max_temperature``).
    If the maximum temperature is reached while advancing the schedule, the SMC
    run is requested to stop after the current iteration finishes.

    Parameters
    ----------
    log_increment_fn:
        A callable or ``TemperedTarget`` providing log-weight increments.
    target_ess_ratio:
        The target ESS ratio (ESS / n) to achieve after weighting, or a
        callable that returns the target for the current temperature. The
        callable may optionally also accept the current ``StepContext``. Each
        resolved target must be in the interval (0, 1].
    max_temperature:
        The maximum temperature to reach; once this temperature is reached,
        the SMC run is requested to stop after the current iteration.
    min_delta:
        The minimum temperature increment to apply at each step.
    max_delta:
        Optional maximum temperature increment to apply at each step.
    normalize_after:
        Whether to normalize the particle weights after updating.
    max_bisection_iter:
        The maximum number of bisection iterations to perform when finding
        the temperature increment.
    criterion:
        Criterion used when selecting the temperature increment. ``"ess"``
        targets the ESS ratio after weighting. ``"cess"`` targets the
        conditional ESS ratio induced by the proposed incremental weights.

    Notes
    -----
    The returned particle state and current step context record
    ``temperature``, ``temperature_delta``, ``ess_ratio``, and ``cess_ratio``.
    Active updates also record ``target_ess_ratio``.
    """

    def __init__(
        self,
        log_increment_fn: IncrementFn | "TemperedTarget",
        *,
        target_ess_ratio: TargetESSRatio = 0.7,
        max_temperature: float = 1.0,
        min_delta: float = 1e-4,
        max_delta: float | None = None,
        normalize_after: bool = True,
        max_bisection_iter: int = 30,
        criterion: Literal["ess", "cess"] = "ess",
        refresh_components: Collection[str] = (),
    ):
        if not callable(target_ess_ratio) and not (
            0.0 < target_ess_ratio <= 1.0
        ):
            raise ValueError("target_ess_ratio must be in (0, 1].")
        if max_temperature <= 0.0:
            raise ValueError("max_temperature must be positive.")
        if min_delta < 0.0:
            raise ValueError("min_delta must be non-negative.")
        if max_delta is not None and max_delta <= 0.0:
            raise ValueError("max_delta must be positive.")
        if max_delta is not None and max_delta < min_delta:
            raise ValueError(
                "max_delta must be greater than or equal to min_delta."
            )
        if criterion not in {"ess", "cess"}:
            raise ValueError("criterion must be either 'ess' or 'cess'.")

        super().__init__(
            log_increment_fn,
            normalize_after=normalize_after,
            refresh_components=refresh_components,
        )

        self._target_ess_ratio = target_ess_ratio
        self._max_temperature = max_temperature
        self._min_delta = min_delta
        self._max_delta = max_delta
        self._max_bisection_iter = max_bisection_iter
        self._criterion = criterion

    def apply(
        self,
        state: ParticleState,
        context: StepContext,
        rng: "RandomGenerator",
    ) -> ParticleState:
        state = self.initialize_metadata(state, context)
        current_temp = context.metadata.get("temperature", 0.0)
        remaining = self._max_temperature - current_temp
        if remaining <= 0.0:
            ess_ratio = ess_ratio_from_log_weights(state.log_weights)
            metadata = dict(state.metadata)
            metadata.update(
                temperature=current_temp,
                temperature_delta=0.0,
                ess_ratio=ess_ratio,
                cess_ratio=1.0,
            )
            context.metadata["temperature"] = current_temp
            context.metadata["temperature_delta"] = 0.0
            context.metadata["ess_ratio"] = ess_ratio
            context.metadata["cess_ratio"] = 1.0
            logger.debug(
                "AdaptiveTemperedWeighting already at max temperature %.3f; requesting immediate stop.",
                self._max_temperature,
            )
            context.request_stop_immediately()
            return state.with_updates(metadata=metadata)
        allowed_delta = (
            remaining
            if self._max_delta is None
            else min(remaining, self._max_delta)
        )

        xp = namespace(state.log_weights)
        log_increments, state = self.evaluate_log_increment(state, context)
        increments = _weights_as_1d_array(
            log_increments, xp=xp, like=state.log_weights
        )
        target_ess_ratio = self._resolve_target_ess_ratio(
            current_temp, context
        )

        if _max_abs(increments, xp=xp) <= 1e-12:
            delta = allowed_delta
        else:
            delta = self._find_delta(
                state.log_weights,
                increments,
                allowed_delta,
                target_ess_ratio=target_ess_ratio,
            )

        new_log_weights = state.log_weights + delta * increments
        new_ess_ratio = ess_ratio_from_log_weights(new_log_weights)
        new_cess_ratio = cess_ratio_from_log_weights_and_increments(
            state.log_weights,
            delta * increments,
        )
        metadata = dict(state.metadata)
        metadata.update(
            _tempering_diagnostics(
                state.log_weights,
                increments,
                delta=delta,
                terminal_delta=remaining,
            )
        )
        metadata["temperature"] = current_temp + delta
        metadata["temperature_delta"] = delta
        metadata["target_ess_ratio"] = target_ess_ratio
        metadata["ess_ratio"] = new_ess_ratio
        metadata["cess_ratio"] = new_cess_ratio

        updated_state = state.with_updates(
            log_weights=new_log_weights, metadata=metadata
        )
        if self._normalize_after:
            updated_state, _ = updated_state.normalize()

        context.metadata["temperature"] = metadata["temperature"]
        context.metadata["temperature_delta"] = delta
        context.metadata["target_ess_ratio"] = target_ess_ratio
        context.metadata["ess_ratio"] = new_ess_ratio
        context.metadata["cess_ratio"] = new_cess_ratio
        logger.debug(
            "AdaptiveTemperedWeighting: ΔT=%.6f, temperature=%.4f, "
            "target %s ratio=%.4f, ESS ratio=%.4f.",
            delta,
            metadata["temperature"],
            self._criterion.upper(),
            target_ess_ratio,
            new_ess_ratio,
        )
        if (
            metadata["temperature"] >= self._max_temperature
            or abs(metadata["temperature"] - self._max_temperature) <= 1e-9
        ):
            logger.debug(
                "AdaptiveTemperedWeighting reached max temperature %.3f; requesting stop after iteration.",
                self._max_temperature,
            )
            context.request_stop()
        return updated_state

    def _resolve_target_ess_ratio(
        self,
        current_temperature: float,
        context: StepContext,
    ) -> float:
        target = self._target_ess_ratio
        if callable(target):
            resolved_target = call_with_optional_context(
                target,
                (current_temperature,),
                context,
            )
        else:
            resolved_target = target
        target_value = float(resolved_target)
        if not 0.0 < target_value <= 1.0:
            raise ValueError(
                "target_ess_ratio must resolve to a value in (0, 1]."
            )
        return target_value

    def _find_delta(
        self,
        base_log_weights: Array,
        increments: Array,
        max_delta: float,
        *,
        target_ess_ratio: float,
    ) -> float:
        low = 0.0
        high = max_delta

        ratio_fn = (
            self._ess_ratio_for_delta
            if self._criterion == "ess"
            else self._cess_ratio_for_delta
        )

        base_ratio = ratio_fn(base_log_weights, increments, 0.0)
        if base_ratio <= target_ess_ratio:
            return max(self._min_delta, min(high, self._min_delta))

        hi_ratio = ratio_fn(base_log_weights, increments, high)
        if hi_ratio >= target_ess_ratio:
            return high

        for _ in range(self._max_bisection_iter):
            mid = 0.5 * (low + high)
            if high - low <= self._min_delta:
                break
            ratio = ratio_fn(base_log_weights, increments, mid)
            if ratio >= target_ess_ratio:
                low = mid
            else:
                high = mid

        delta = max(0.5 * (low + high), self._min_delta)
        return min(delta, max_delta)

    @staticmethod
    def _ess_ratio_for_delta(
        base_log_weights: Array,
        increments: Array,
        delta: float,
    ) -> float:
        return ess_ratio_from_log_weights(
            base_log_weights + delta * increments
        )

    @staticmethod
    def _cess_ratio_for_delta(
        base_log_weights: Array,
        increments: Array,
        delta: float,
    ) -> float:
        return cess_ratio_from_log_weights_and_increments(
            base_log_weights,
            delta * increments,
        )


def _max_abs(array: Array, *, xp) -> float:
    return float(to_numpy(xp.max(xp.abs(array))))
