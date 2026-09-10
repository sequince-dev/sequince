from __future__ import annotations

import logging
import math
from collections.abc import Collection, Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Callable, Literal, cast

from .._typing import Array
from ..array_api import asarray, logsumexp, namespace, to_numpy
from ..state import _validate_log_normalizer, normalize_log_weights
from ..stats import (
    cess_ratio_from_log_weights_and_increments,
    ess_from_log_weights,
    ess_ratio_from_log_weights,
)
from ..target import (
    EvaluationCounts,
    SequentialTarget,
    TemperedTarget,
)
from ..utils.callables import call_with_optional_context
from ..utils.optimize import bisect_monotone_decreasing
from .base import SMCStep, StepContext

if TYPE_CHECKING:
    from orng import RandomGenerator

    from ..state import ParticleState


IncrementFn = Callable[..., Array | float]
BetaSchedule = Callable[..., float]
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

    log_normalizing_constant_increment = float(
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
        log_normalizing_constant_increment_variance = math.nan
    else:
        centered = scaled_ratios - mean_ratio
        log_normalizing_constant_increment_variance = float(
            to_numpy(
                xp.sum(xp.square(base_weights) * xp.square(centered))
                / xp.square(mean_ratio)
            )
        )

    return {
        "log_normalizing_constant_increment": (
            log_normalizing_constant_increment
        ),
        "log_normalizing_constant_increment_variance": (
            log_normalizing_constant_increment_variance
        ),
        "ess_before_resampling": ess,
        "ess_at_max_beta": terminal_ess,
    }


def evolving_target_ess_ratio(
    start: float,
    end: float,
    *,
    rate: float = 1.0,
    max_beta: float = 1.0,
) -> Callable[[float], float]:
    """Return a power-law schedule for an evolving target ESS ratio.

    The schedule evolves from ``start`` to ``end`` as the beta moves
    from zero to ``max_beta``. Betas outside that interval are
    clipped to the corresponding endpoint. Values of ``rate`` greater than
    one delay the transition, while values below one accelerate it.

    Parameters
    ----------
    start:
        Target ESS ratio at zero beta; must be in ``(0, 1]``.
    end:
        Target ESS ratio at ``max_beta``; must be in ``(0, 1]``.
    rate:
        Positive exponent controlling the rate of transition.
    max_beta:
        Positive beta at which the schedule reaches ``end``.
    """
    if not 0.0 < start <= 1.0:
        raise ValueError("start must be in (0, 1].")
    if not 0.0 < end <= 1.0:
        raise ValueError("end must be in (0, 1].")
    if not rate > 0.0:
        raise ValueError("rate must be positive.")
    if not max_beta > 0.0:
        raise ValueError("max_beta must be positive.")

    def schedule(beta: float) -> float:
        progress = min(max(beta / max_beta, 0.0), 1.0)
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


def _apply_log_weight_increment(
    state: ParticleState,
    log_increment: Array | float,
    *,
    normalize: bool = True,
    validate_normalizer: bool = True,
) -> tuple[ParticleState, Any | None]:
    """Apply an incremental log importance weight to a particle state.

    Functional callers can defer normalizer validation to an eager boundary so
    that this operation remains compatible with traced array scalars.
    """
    updated_state = state.with_updates(
        log_weights=state.log_weights + log_increment
    )
    if not normalize:
        return updated_state, None

    if validate_normalizer:
        updated_state, log_norm = updated_state.normalize()
    else:
        xp = namespace(updated_state.log_weights)
        normalized, log_norm = normalize_log_weights(
            updated_state.log_weights,
            xp=xp,
        )
        updated_state = updated_state.with_updates(log_weights=normalized)
    return (
        updated_state.with_updates(
            log_normalizing_constant=(
                updated_state.log_normalizing_constant + log_norm
            )
        ),
        log_norm,
    )


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

    def initialize_functional_state_and_context(
        self,
        state: ParticleState,
        context: StepContext,
    ) -> tuple[ParticleState, StepContext]:
        """Attach a cached target evaluation before functional compilation."""
        if self._target is None:
            return state, context
        evaluation, counts = self._target.prepare_evaluation(
            state.particles,
            context,
            evaluation=state.evaluation,
            components={"log_increment"},
            return_counts=True,
        )
        return (
            state.with_updates(evaluation=evaluation),
            context.add_evaluation_counts(counts),
        )

    def apply(
        self,
        state: ParticleState,
        context: StepContext,
        rng: "RandomGenerator",
    ) -> tuple[ParticleState, StepContext, dict[str, Any]]:
        xp = namespace(state.log_weights)
        log_increment_counts = EvaluationCounts(
            log_increment_evaluations=xp.asarray(state.particles.shape[0])
        )
        if self._target is None:
            if self._log_increment_fn is None:  # pragma: no cover
                raise RuntimeError("Sequential increment is not configured.")
            increment = call_with_optional_context(
                self._log_increment_fn,
                (state.particles,),
                context,
            )
        else:
            evaluation, counts = self._target.prepare_evaluation(
                state.particles,
                context,
                evaluation=state.evaluation,
                components={"log_increment"},
                return_counts=True,
            )
            context = context.add_evaluation_counts(counts)
            increment = self._target.log_increment(
                state.particles,
                context,
                evaluation=evaluation,
            )
            if evaluation is not state.evaluation:
                state = state.with_updates(evaluation=evaluation)
        context = context.add_evaluation_counts(log_increment_counts)
        increments = _weights_as_1d_array(
            increment, xp=xp, like=state.log_weights
        )
        if increments.shape[0] != state.log_weights.shape[0]:
            raise ValueError(
                "Increment function must return one value per particle."
            )

        updated_state, _ = _apply_log_weight_increment(
            state,
            increments,
            normalize=self._normalize_after,
        )
        logger.debug(
            "IncrementalWeighting: mean increment %.6f (normalize_after=%s).",
            float(to_numpy(xp.sum(increments) / increments.shape[0])),
            self._normalize_after,
        )
        return updated_state, context, {"log_weight_increment": increments}


class LogLikelihoodWeighting(IncrementalWeighting):
    """Backward-compatible name for callable likelihood increments."""


class BaseTemperedStep(SMCStep):
    """Base class for tempered weighting steps.

    This provides common functionality for tempering steps that apply log-weight
    increments scaled by a beta schedule (e.g. ``TemperedWeighting`` and
    ``AdaptiveTemperedWeighting``). It does not implement the logic for determining
    the beta schedule itself, which is left to subclasses.
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

    def initialize_functional_state_and_context(
        self,
        state: ParticleState,
        context: StepContext,
    ) -> tuple[ParticleState, StepContext]:
        """Attach a cached target evaluation before functional compilation."""
        if self._target is None:
            return state, context
        evaluation, counts = self._target.prepare_evaluation(
            state.particles,
            context,
            evaluation=state.evaluation,
            components={"log_increment"},
            return_counts=True,
            refresh_components=self._refresh_components,
        )
        return (
            state.with_updates(evaluation=evaluation),
            context.add_evaluation_counts(counts),
        )

    def evaluate_log_increment(
        self,
        state: ParticleState,
        context: StepContext,
    ) -> tuple[Array | float, ParticleState, StepContext]:
        """Return increments and attach a missing target evaluation."""
        xp = namespace(state.log_weights)
        log_increment_counts = EvaluationCounts(
            log_increment_evaluations=xp.asarray(state.particles.shape[0])
        )
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
                context.add_evaluation_counts(log_increment_counts),
            )
        evaluation, counts = self._target.prepare_evaluation(
            state.particles,
            context,
            evaluation=state.evaluation,
            components={"log_increment"},
            return_counts=True,
            refresh_components=self._refresh_components,
        )
        context = context.add_evaluation_counts(counts + log_increment_counts)
        if evaluation is not state.evaluation:
            state = state.with_updates(evaluation=evaluation)
        return (
            self._target.log_increment(
                state.particles,
                context,
                evaluation=evaluation,
            ),
            state,
            context,
        )

    def initialize_metadata(
        self, state: ParticleState, context: StepContext
    ) -> tuple[ParticleState, StepContext]:
        """Initialize metadata related to tempering."""
        if "beta" not in context.metadata:
            initial_beta = float(state.metadata.get("beta", 0.0))
            logger.debug(
                "Initializing tempering metadata in context to %.4f.",
                initial_beta,
            )
            context = context.with_metadata(beta=initial_beta)
            state_metadata = dict(state.metadata)
            state_metadata["beta"] = initial_beta
            state = state.with_updates(metadata=state_metadata)
        if "delta_beta" not in context.metadata:
            context = context.with_metadata(delta_beta=0.0)
        return state, context

    def initialize_functional_metadata(
        self,
        state: ParticleState,
        metadata: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Initialize fixed-shape tempering metadata on the array backend."""
        xp = namespace(state.log_weights)
        initialized = dict(metadata)
        beta = initialized.get(
            "beta",
            state.metadata.get("beta", 0.0),
        )
        initialized["beta"] = asarray(
            beta,
            xp=xp,
            like=state.log_weights,
        )
        initialized["delta_beta"] = asarray(
            initialized.get("delta_beta", 0.0),
            xp=xp,
            like=state.log_weights,
        )
        return initialized

    def _prepare_log_increments(
        self,
        state: ParticleState,
        context: StepContext,
    ) -> tuple[Array, ParticleState, StepContext, Any]:
        """Evaluate and validate one log increment per particle."""
        xp = namespace(state.log_weights)
        log_increments, state, context = self.evaluate_log_increment(
            state, context
        )
        increments = _weights_as_1d_array(
            log_increments,
            xp=xp,
            like=state.log_weights,
        )
        if increments.shape[0] != state.log_weights.shape[0]:
            raise ValueError(
                "Increment function must return one value per particle."
            )
        return increments, state, context, xp

    def _update_log_weights(
        self,
        state: ParticleState,
        increments: Array,
    ) -> tuple[ParticleState, Any | None]:
        """Apply increments and optionally normalize without leaving the backend."""
        return _apply_log_weight_increment(
            state,
            increments,
            normalize=self._normalize_after,
            validate_normalizer=False,
        )

    def apply(self, state, context, rng):
        raise NotImplementedError(
            "BaseTemperedStep subclasses must implement apply()."
        )


class TemperedWeighting(BaseTemperedStep):
    """Apply tempering by scaling log-weight increments with a beta schedule.

    Both the increment callable and the beta schedule accept the
    current ``StepContext`` when they require it.
    """

    def __init__(
        self,
        log_increment_fn: IncrementFn | "TemperedTarget",
        beta_schedule: BetaSchedule,
        *,
        normalize_after: bool = True,
        clip_beta: tuple[float, float] | None = None,
        refresh_components: Collection[str] = (),
    ):
        super().__init__(
            log_increment_fn,
            normalize_after=normalize_after,
            refresh_components=refresh_components,
        )
        self._schedule = beta_schedule
        self._clip = clip_beta

    def apply(
        self,
        state: ParticleState,
        context: StepContext,
        rng: "RandomGenerator",
    ) -> tuple[ParticleState, StepContext, dict[str, Any]]:
        state, context = self.initialize_metadata(state, context)
        current_beta = float(context.metadata.get("beta", 0.0))
        target_beta = float(
            call_with_optional_context(
                self._schedule,
                (context.t,),
                context,
            )
        )
        if self._clip is not None:
            lo, hi = self._clip
            target_beta = float(max(lo, min(hi, target_beta)))
        if target_beta < current_beta:
            raise ValueError(
                "beta_schedule produced a beta smaller than the current beta."
            )
        delta_beta = target_beta - current_beta

        log_increments, state, context = self.evaluate_log_increment(
            state, context
        )
        xp = namespace(state.log_weights)
        increments = (
            _weights_as_1d_array(log_increments, xp=xp, like=state.log_weights)
            * delta_beta
        )
        if increments.shape[0] != state.log_weights.shape[0]:
            raise ValueError(
                "Increment function must return one value per particle."
            )

        logger.debug(
            "TemperedWeighting: beta %.4f -> %.4f applied; mean increment %.6f.",
            current_beta,
            target_beta,
            float(to_numpy(xp.sum(increments) / increments.shape[0])),
        )
        metadata = dict(state.metadata)
        metadata["beta"] = target_beta
        metadata["delta_beta"] = delta_beta
        updated_state, _ = _apply_log_weight_increment(
            state.with_updates(metadata=metadata),
            increments,
            normalize=self._normalize_after,
        )
        context = context.with_metadata(
            beta=target_beta,
            delta_beta=delta_beta,
        )
        output = {
            "beta": target_beta,
            "delta_beta": delta_beta,
            "log_weight_increment": increments,
        }
        return updated_state, context, output


class FixedTemperedWeighting(BaseTemperedStep):
    """Apply tempering with a fixed beta increment."""

    def __init__(
        self,
        log_increment_fn: IncrementFn | "TemperedTarget",
        *,
        beta_increment: float,
        normalize_after: bool = True,
        refresh_components: Collection[str] = (),
    ):
        if beta_increment < 0.0:
            raise ValueError("beta_increment must be non-negative.")
        super().__init__(
            log_increment_fn,
            normalize_after=normalize_after,
            refresh_components=refresh_components,
        )
        self._beta_increment = beta_increment

    def apply(
        self,
        state: ParticleState,
        context: StepContext,
        rng: "RandomGenerator",
    ) -> tuple[ParticleState, StepContext, dict[str, Any]]:
        del rng
        state, context = self.initialize_metadata(state, context)
        base_state = state
        current_beta = context.metadata.get("beta", 0.0)
        updated_state, context, output, raw_increments, log_norm = (
            self._transition(state, context)
        )
        if log_norm is not None:
            _validate_log_normalizer(
                log_norm,
                xp=namespace(updated_state.log_weights),
            )

        metadata = dict(state.metadata)
        max_beta = float(context.metadata.get("max_beta", 1.0))
        metadata.update(
            _tempering_diagnostics(
                base_state.log_weights,
                raw_increments,
                delta=self._beta_increment,
                terminal_delta=max(
                    max_beta - current_beta,
                    0.0,
                ),
            )
        )
        metadata["beta"] = output["beta"]
        metadata["delta_beta"] = self._beta_increment
        updated_state = updated_state.with_updates(metadata=metadata)
        logger.debug(
            "FixedTemperedWeighting: increment %.4f -> beta %.4f.",
            self._beta_increment,
            metadata["beta"],
        )
        return updated_state, context, output

    def apply_functional(
        self,
        state: ParticleState,
        context: StepContext,
        random_state: Any,
    ) -> tuple[ParticleState, StepContext, Any, dict[str, Any]]:
        """Apply fixed tempering without host scalar conversions."""
        state, context, output, _, _ = self._transition(state, context)
        return state, context, random_state, output

    def _transition(
        self,
        state: ParticleState,
        context: StepContext,
    ) -> tuple[
        ParticleState,
        StepContext,
        dict[str, Any],
        Array,
        Any | None,
    ]:
        """Apply the backend-native fixed-beta transition."""
        current_beta = context.metadata.get(
            "beta", state.metadata.get("beta", 0.0)
        )
        log_increments, state, context, xp = self._prepare_log_increments(
            state, context
        )
        increments = log_increments * self._beta_increment
        target_beta = current_beta + self._beta_increment
        updated_state, log_norm = self._update_log_weights(
            state,
            increments,
        )
        context = context.with_metadata(
            beta=target_beta,
            delta_beta=self._beta_increment,
        )
        output = {
            "beta": target_beta,
            "delta_beta": self._beta_increment,
            "log_weight_increment": increments,
        }
        return updated_state, context, output, log_increments, log_norm


class AdaptiveTemperedWeighting(BaseTemperedStep):
    """Adaptively choose the next beta using an ESS-like target.

    The beta increment is chosen such that the effective sample size
    (ESS) ratio after weighting, or the conditional ESS (CESS) ratio implied by
    a proposed increment, reaches the specified target value
    (``target_ess_ratio``), up to the maximum beta
    (``max_beta``).
    If the maximum beta is reached while advancing the schedule, the SMC
    run is requested to stop after the current iteration finishes.

    Parameters
    ----------
    log_increment_fn:
        A callable or ``TemperedTarget`` providing log-weight increments.
    target_ess_ratio:
        The target ESS ratio (ESS / n) to achieve after weighting, or a
        callable that returns the target for the current beta. The
        callable may optionally also accept the current ``StepContext``. Each
        resolved target must be in the interval (0, 1].
    max_beta:
        The maximum beta to reach; once this beta is reached,
        the SMC run is requested to stop after the current iteration.
    min_delta:
        The minimum beta increment to apply at each step.
    max_delta:
        Optional maximum beta increment to apply at each step.
    normalize_after:
        Whether to normalize the particle weights after updating.
    max_bisection_iter:
        The maximum number of bisection iterations to perform when finding
        the beta increment.
    criterion:
        Criterion used when selecting the beta increment. ``"ess"``
        targets the ESS ratio after weighting. ``"cess"`` targets the
        conditional ESS ratio induced by the proposed incremental weights.

    Notes
    -----
    The returned particle state and current step context record
    ``beta``, ``delta_beta``, ``ess_ratio``, and ``cess_ratio``.
    Active updates also record ``target_ess_ratio``.
    """

    def __init__(
        self,
        log_increment_fn: IncrementFn | "TemperedTarget",
        *,
        target_ess_ratio: TargetESSRatio = 0.7,
        max_beta: float = 1.0,
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
        if max_beta <= 0.0:
            raise ValueError("max_beta must be positive.")
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
        self._max_beta = max_beta
        self._min_delta = min_delta
        self._max_delta = max_delta
        self._max_bisection_iter = max_bisection_iter
        self._criterion = criterion

    def initialize_functional_metadata(
        self,
        state: ParticleState,
        metadata: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Initialize fixed-shape adaptive tempering metadata."""
        initialized = dict(
            super().initialize_functional_metadata(state, metadata)
        )
        xp = namespace(state.log_weights)
        target = initialized.get("target_ess_ratio", self._target_ess_ratio)
        if callable(target):
            target = 1.0
        initialized.update(
            target_ess_ratio=asarray(
                target,
                xp=xp,
                like=state.log_weights,
            ),
            ess_ratio=asarray(
                initialized.get("ess_ratio", 1.0),
                xp=xp,
                like=state.log_weights,
            ),
            cess_ratio=asarray(
                initialized.get("cess_ratio", 1.0),
                xp=xp,
                like=state.log_weights,
            ),
        )
        return initialized

    def apply(
        self,
        state: ParticleState,
        context: StepContext,
        rng: "RandomGenerator",
    ) -> tuple[ParticleState, StepContext, dict[str, Any]]:
        del rng
        state, context = self.initialize_metadata(state, context)
        base_state = state
        (
            updated_state,
            context,
            output,
            raw_increments,
            remaining,
            log_norm,
        ) = self._transition(state, context, validate_target=True)
        if log_norm is not None:
            _validate_log_normalizer(
                log_norm,
                xp=namespace(updated_state.log_weights),
            )

        metadata = dict(updated_state.metadata)
        metadata.update(
            _tempering_diagnostics(
                base_state.log_weights,
                raw_increments,
                delta=output["delta_beta"],
                terminal_delta=remaining,
            )
        )
        metadata.update(
            beta=output["beta"],
            delta_beta=output["delta_beta"],
            target_ess_ratio=output["target_ess_ratio"],
            ess_ratio=output["ess_ratio"],
            cess_ratio=output["cess_ratio"],
        )
        updated_state = updated_state.with_updates(metadata=metadata)

        if context.stop_immediately:
            logger.debug(
                "Iteration %d: AdaptiveTemperedWeighting already at max beta %.3f; "
                "requesting immediate stop.",
                context.t,
                self._max_beta,
            )
        else:
            logger.debug(
                "Iteration %d: AdaptiveTemperedWeighting: Δβ=%.6f, beta=%.4f, "
                "target %s ratio=%.4f, ESS ratio=%.4f.",
                context.t,
                output["delta_beta"],
                output["beta"],
                self._criterion.upper(),
                output["target_ess_ratio"],
                output["ess_ratio"],
            )
            if context.stop_requested:
                logger.debug(
                    "Iteration %d: AdaptiveTemperedWeighting reached max beta %.3f; "
                    "requesting stop after iteration.",
                    context.t,
                    self._max_beta,
                )
        return updated_state, context, output

    def apply_functional(
        self,
        state: ParticleState,
        context: StepContext,
        random_state: Any,
    ) -> tuple[ParticleState, StepContext, Any, dict[str, Any]]:
        """Apply adaptive tempering while explicitly threading random state."""
        state, context, output, _, _, _ = self._transition(state, context)
        return state, context, random_state, output

    def _transition(
        self,
        state: ParticleState,
        context: StepContext,
        *,
        validate_target: bool = False,
    ) -> tuple[
        ParticleState,
        StepContext,
        dict[str, Any],
        Array,
        Array,
        Any | None,
    ]:
        """Apply the backend-native adaptive-beta transition."""
        xp = namespace(state.log_weights)
        current_beta = asarray(
            context.metadata.get("beta", state.metadata.get("beta", 0.0)),
            xp=xp,
            like=state.log_weights,
        )
        zero = asarray(0.0, xp=xp, like=state.log_weights)
        max_beta = asarray(
            self._max_beta,
            xp=xp,
            like=state.log_weights,
        )
        remaining = xp.maximum(
            max_beta - current_beta,
            zero,
        )
        allowed_delta = remaining
        if self._max_delta is not None:
            allowed_delta = xp.minimum(
                allowed_delta,
                asarray(self._max_delta, xp=xp, like=state.log_weights),
            )

        increments, state, context, xp = self._prepare_log_increments(
            state, context
        )
        target_ess_ratio = self._resolve_target_ess_ratio(
            current_beta,
            context,
            xp=xp,
            like=state.log_weights,
        )
        if validate_target:
            target_value = float(target_ess_ratio)
            if not 0.0 < target_value <= 1.0:
                raise ValueError(
                    "target_ess_ratio must resolve to a value in (0, 1]."
                )

        selected_delta = self._find_delta(
            state.log_weights,
            increments,
            allowed_delta,
            target_ess_ratio=target_ess_ratio,
        )
        selected_delta = xp.where(
            xp.max(xp.abs(increments)) <= 1e-12,
            allowed_delta,
            selected_delta,
        )
        already_at_max = remaining <= 0.0
        delta = xp.where(already_at_max, zero, selected_delta)
        scaled_increments = delta * increments
        new_log_weights = state.log_weights + scaled_increments
        new_ess_ratio = ess_ratio_from_log_weights(new_log_weights)
        new_cess_ratio = cess_ratio_from_log_weights_and_increments(
            state.log_weights,
            scaled_increments,
        )
        new_beta = xp.minimum(
            current_beta + delta,
            max_beta,
        )
        reached_max = new_beta >= self._max_beta - 1e-9

        updated_state, log_norm = self._update_log_weights(
            state,
            scaled_increments,
        )
        context = context.with_metadata(
            beta=new_beta,
            delta_beta=delta,
            target_ess_ratio=target_ess_ratio,
            ess_ratio=new_ess_ratio,
            cess_ratio=new_cess_ratio,
        )
        stop_requested = asarray(
            context.stop_requested,
            xp=xp,
            dtype=bool,
            like=state.log_weights,
        )
        stop_immediately = asarray(
            context.stop_immediately,
            xp=xp,
            dtype=bool,
            like=state.log_weights,
        )
        context = replace(
            context,
            stop_requested=xp.logical_or(
                stop_requested,
                reached_max,
            ),
            stop_immediately=xp.logical_or(
                stop_immediately,
                already_at_max,
            ),
        )
        output = {
            "beta": new_beta,
            "delta_beta": delta,
            "target_ess_ratio": target_ess_ratio,
            "ess_ratio": new_ess_ratio,
            "cess_ratio": new_cess_ratio,
            "log_weight_increment": scaled_increments,
        }
        return updated_state, context, output, increments, remaining, log_norm

    def _resolve_target_ess_ratio(
        self,
        current_beta: Array,
        context: StepContext,
        *,
        xp: Any,
        like: Array,
    ) -> Array:
        target = self._target_ess_ratio
        if callable(target):
            target = call_with_optional_context(
                target,
                (current_beta,),
                context,
            )
        return asarray(target, xp=xp, like=like)

    def _find_delta(
        self,
        base_log_weights: Array,
        increments: Array,
        max_delta: Array,
        *,
        target_ess_ratio: Array,
    ) -> Array:
        """Find a beta increment with fixed-count bisection."""
        xp = namespace(base_log_weights, increments, max_delta)
        low = xp.zeros_like(max_delta)
        high = max_delta
        min_delta = asarray(
            self._min_delta,
            xp=xp,
            like=base_log_weights,
        )

        ratio_fn = (
            self._ess_ratio_for_delta
            if self._criterion == "ess"
            else self._cess_ratio_for_delta
        )

        base_ratio = ratio_fn(base_log_weights, increments, low)
        high_ratio = ratio_fn(base_log_weights, increments, high)
        low, high = bisect_monotone_decreasing(
            lambda delta: ratio_fn(base_log_weights, increments, delta),
            low,
            high,
            target_ess_ratio,
            tolerance=min_delta,
            max_iterations=self._max_bisection_iter,
        )

        minimum_delta = xp.minimum(max_delta, min_delta)
        bisected_delta = xp.minimum(
            xp.maximum(0.5 * (low + high), min_delta),
            max_delta,
        )
        return xp.where(
            base_ratio <= target_ess_ratio,
            minimum_delta,
            xp.where(
                high_ratio >= target_ess_ratio,
                max_delta,
                bisected_delta,
            ),
        )

    @staticmethod
    def _ess_ratio_for_delta(
        base_log_weights: Array,
        increments: Array,
        delta: Array,
    ) -> Array:
        xp = namespace(base_log_weights, increments)
        # A zero step leaves weights unchanged even outside the next target's
        # support. Mask before multiplication to avoid 0 * (-inf).
        active_increments = xp.where(
            delta == 0, xp.zeros_like(increments), increments
        )
        return ess_ratio_from_log_weights(
            base_log_weights + delta * active_increments
        )

    @staticmethod
    def _cess_ratio_for_delta(
        base_log_weights: Array,
        increments: Array,
        delta: Array,
    ) -> Array:
        xp = namespace(base_log_weights, increments)
        active_increments = xp.where(
            delta == 0, xp.zeros_like(increments), increments
        )
        return cess_ratio_from_log_weights_and_increments(
            base_log_weights,
            delta * active_increments,
        )
