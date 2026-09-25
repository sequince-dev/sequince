"""Eager likelihood-constrained SMC steps and shell evidence bookkeeping."""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Callable, ClassVar

import numpy as np
from orng import RandomGenerator

from .._typing import Array
from ..array_api import copy as array_copy
from ..array_api import logsumexp, namespace, to_numpy
from ..state import (
    ParticleState,
    _validate_log_normalizer,
    normalize_log_weights,
)
from ..target import (
    EvaluationCounts,
    Target,
    TargetEvaluation,
    _evaluate_component,
)
from .base import SMCStep
from .context import StepContext

LikelihoodThresholdSchedule = Callable[
    [Array, ParticleState, StepContext], float
]


def _nested_sampling_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Copy execution metadata, supplying defaults for standalone steps."""
    values = metadata.get("nested_sampling", {})
    if not isinstance(values, Mapping):
        raise TypeError("nested_sampling metadata must be a mapping.")
    return {
        "threshold": -math.inf,
        "next_threshold": None,
        "is_terminal": False,
        "log_prior_volume_increment": 0.0,
        "log_prior_volume": 0.0,
        "log_nested_evidence": -math.inf,
        "n_surviving": 0,
        "survival_fraction": 1.0,
        "stop_reason": None,
        **values,
    }


def _record_nested_sampling_snapshot(
    state: ParticleState, context: StepContext
) -> ParticleState:
    """Record a completed level independently of execution metadata.

    Weighting records nonterminal levels before mutation; terminal shell
    integration records the final level before stopping.
    """
    return state.with_updates(
        metadata={
            **state.metadata,
            "nested_sampling": dict(context.metadata["nested_sampling"]),
        }
    )


@dataclass(frozen=True)
class NestedSamplingLevel:
    """Diagnostics for a shell; prior volume is measured before discarding it."""

    output_key: ClassVar[str] = "nested_sampling_level"
    iteration: int
    shell_lower_threshold: float
    shell_upper_threshold: float | None
    log_prior_volume: float
    log_shell_expectation: float
    log_evidence_increment: float
    log_evidence: float
    is_terminal: bool
    n_surviving: int
    survival_fraction: float


class LikelihoodConstrainedPrior(Target):
    """A static prior and likelihood with a context-dependent hard constraint.

    Parameters
    ----------
    log_prior_fn, log_likelihood_fn : callable
        Return one log density per particle, accepting particles and optionally
        a context. The density functions must remain unchanged between levels;
        their cached values are reused when only the threshold changes.
    inclusive : bool, optional
        Retain likelihoods equal to the threshold as well as those above it.
    """

    def __init__(
        self,
        log_prior_fn: Callable[..., Any],
        log_likelihood_fn: Callable[..., Any],
        *,
        inclusive: bool = False,
    ) -> None:
        self._components = {
            "log_prior": log_prior_fn,
            "log_likelihood": log_likelihood_fn,
        }
        self.inclusive = inclusive

    def evaluation_components(self, context: StepContext) -> Collection[str]:
        return self._components.keys()

    def evaluate(
        self,
        particles: Array,
        context: StepContext,
        *,
        components: Collection[str] | None = None,
        evaluation: TargetEvaluation | None = None,
    ) -> TargetEvaluation:
        return self._evaluate(
            particles, context, components=components, evaluation=evaluation
        )[0]

    def _evaluate(
        self,
        particles: Array,
        context: StepContext,
        *,
        components: Collection[str] | None = None,
        evaluation: TargetEvaluation | None = None,
        refresh: bool = False,
    ) -> tuple[TargetEvaluation, EvaluationCounts]:
        if evaluation is not None:
            self.validate_evaluation(evaluation, particles, context)
        requested = set(self._components if components is None else components)
        if requested.difference(self._components):
            raise ValueError("Unknown constrained-prior evaluation component.")
        values = {} if evaluation is None else dict(evaluation.components)
        n = 1 if particles.ndim == 1 else particles.shape[0]
        for name in requested:
            value = _evaluate_component(
                self._components[name], particles, context
            )
            if value.shape != (n,):
                raise ValueError(
                    "Density functions must return one value per particle."
                )
            xp = namespace(value)
            if bool(xp.any(xp.isnan(value) | (value == math.inf))):
                raise ValueError(
                    "Log densities must not contain NaN or positive infinity."
                )
            values[name] = value
        return self.bind_evaluation(
            TargetEvaluation(values), context
        ), EvaluationCounts(
            log_target_evaluations=n if requested else 0,
            log_likelihood_evaluations=n
            if "log_likelihood" in requested
            else 0,
        )

    def log_prob(
        self,
        particles: Array,
        context: StepContext,
        *,
        evaluation: TargetEvaluation | None = None,
    ) -> Array | float:
        if evaluation is not None:
            self.validate_evaluation(evaluation, particles, context)
        evaluation = self.prepare_evaluation(
            particles, context, evaluation=evaluation
        )
        runtime = context.metadata.get("nested_sampling", {})
        threshold = runtime.get("threshold", -math.inf)
        values = evaluation["log_likelihood"]
        mask = values >= threshold if self.inclusive else values > threshold
        # The initial target is the full prior, including zero-likelihood mass.
        if threshold == -math.inf and runtime.get("next_threshold") is None:
            mask = namespace(values).ones_like(values, dtype=bool)
        result = namespace(values).where(
            mask, evaluation["log_prior"], -math.inf
        )
        return result[0] if particles.ndim == 1 else result


def _prepare_likelihood(
    target: LikelihoodConstrainedPrior,
    state: ParticleState,
    context: StepContext,
) -> tuple[ParticleState, StepContext, Array]:
    evaluation, counts = target.prepare_evaluation(
        state.particles,
        context,
        evaluation=state.evaluation,
        components={"log_likelihood"},
        return_counts=True,
    )
    if evaluation is not state.evaluation:
        state = state.with_updates(evaluation=evaluation)
    return (
        state,
        context.add_evaluation_counts(counts),
        evaluation["log_likelihood"],
    )


def _normalized_weights(state: ParticleState) -> Array:
    xp = namespace(state.log_weights)
    weights, norm = normalize_log_weights(state.log_weights, xp=xp)
    _validate_log_normalizer(norm, xp=xp)
    return weights


@dataclass(frozen=True)
class FixedLikelihoodThresholdSchedule:
    """Use a strictly increasing sequence of log-likelihood thresholds.

    Exhaustion records all remaining mass as a terminal shell. Positive
    infinity is also an explicit terminal threshold.
    """

    thresholds: Sequence[float]

    def __post_init__(self) -> None:
        values = tuple(float(x) for x in self.thresholds)
        if (
            not values
            or any(math.isnan(x) for x in values)
            or any(b <= a for a, b in zip(values, values[1:]))
        ):
            raise ValueError(
                "thresholds must be nonempty and strictly increasing without NaN."
            )
        object.__setattr__(self, "thresholds", values)

    def __call__(
        self,
        log_likelihoods: Array,
        state: ParticleState,
        context: StepContext,
    ) -> float:
        if context.t >= len(self.thresholds):
            raise StopIteration
        return float(self.thresholds[context.t])


@dataclass(frozen=True)
class QuantileLikelihoodThresholdSchedule:
    """Select an observed likelihood at the weighted quantile.

    Equal likelihoods are discarded together with a strict constraint. This
    can remove more than the requested fraction on a plateau.
    """

    quantile: float = 0.5

    def __post_init__(self) -> None:
        if not 0 <= self.quantile < 1:
            raise ValueError("quantile must lie in [0, 1).")

    def __call__(
        self,
        log_likelihoods: Array,
        state: ParticleState,
        context: StepContext,
    ) -> float:
        values = np.asarray(to_numpy(log_likelihoods))
        weights = np.exp(to_numpy(_normalized_weights(state)))
        active = weights > 0
        values, weights = values[active], weights[active]
        order = np.argsort(values)
        index = min(
            int(
                np.searchsorted(
                    np.cumsum(weights[order]), self.quantile, side="left"
                )
            ),
            len(order) - 1,
        )
        return float(values[order[index]])


class WorstParticleLikelihoodThresholdSchedule(
    QuantileLikelihoodThresholdSchedule
):
    """Discard the lowest likelihood among particles with positive weight."""

    def __init__(self) -> None:
        super().__init__(quantile=0.0)


class _ConstraintStep(SMCStep):
    def apply_functional(
        self, state: ParticleState, context: StepContext, random_state: Any
    ) -> tuple[ParticleState, StepContext, Any, Any]:
        raise NotImplementedError(
            "NS-SMC currently supports eager execution only."
        )


class LikelihoodThresholdSelection(_ConstraintStep):
    """Select a threshold or finalize using an estimated remaining-evidence ratio.

    Schedules must implement ``schedule(log_likelihoods, state, context)`` and
    return a float. ``StopIteration`` finalizes the remaining shell; other
    exceptions propagate to the caller.

    ``evidence_tolerance`` compares the current estimate X * E[L] with the
    accumulated evidence. It is a stopping heuristic, not a rigorous bound on
    unobserved modes. Supply a known log_likelihood_upper_bound to use X * Lmax
    instead (X itself is estimated), or disable it for fixed-schedule experiments.
    """

    def __init__(
        self,
        target: LikelihoodConstrainedPrior,
        *,
        threshold_schedule: LikelihoodThresholdSchedule | None = None,
        evidence_tolerance: float | None = None,
        log_likelihood_upper_bound: float | None = None,
    ) -> None:
        if evidence_tolerance is not None and not 0 < evidence_tolerance < 1:
            raise ValueError("evidence_tolerance must lie in (0, 1).")
        self.target = target
        self.schedule = (
            threshold_schedule or QuantileLikelihoodThresholdSchedule()
        )
        self.evidence_tolerance = evidence_tolerance
        if log_likelihood_upper_bound is not None and not math.isfinite(
            log_likelihood_upper_bound
        ):
            raise ValueError("log_likelihood_upper_bound must be finite.")
        self.log_likelihood_upper_bound = log_likelihood_upper_bound

    def apply(
        self, state: ParticleState, context: StepContext, rng: RandomGenerator
    ) -> tuple[ParticleState, StepContext, None]:
        runtime = _nested_sampling_metadata(context.metadata)
        if runtime["is_terminal"]:
            raise ValueError("Cannot continue a finalized NS-SMC run.")
        state, context, values = _prepare_likelihood(
            self.target, state, context
        )
        weights = _normalized_weights(state)
        xp = namespace(values)
        if self.log_likelihood_upper_bound is not None:
            if bool(xp.any(values > self.log_likelihood_upper_bound)):
                raise ValueError(
                    "Observed likelihood exceeds log_likelihood_upper_bound."
                )
            remaining = (
                runtime["log_prior_volume"] + self.log_likelihood_upper_bound
            )
        else:
            remaining = runtime["log_prior_volume"] + float(
                logsumexp(weights + values)
            )
        terminal = (
            self.evidence_tolerance is not None
            and math.isfinite(runtime["log_nested_evidence"])
            and remaining
            <= runtime["log_nested_evidence"]
            + math.log(self.evidence_tolerance)
        )
        reason = "evidence_tolerance" if terminal else None
        if terminal:
            threshold = math.inf
        else:
            try:
                threshold = float(self.schedule(values, state, context))
            except StopIteration:
                threshold = math.inf
                reason = "schedule_exhausted"
        if math.isnan(threshold) or threshold < runtime["threshold"]:
            raise ValueError(
                "Likelihood thresholds must be nondecreasing and not NaN."
            )
        active = xp.isfinite(weights)
        survivors = (
            values >= threshold
            if self.target.inclusive
            else values > threshold
        ) & active
        # A sampled plateau (including a constant likelihood) is integrated in
        # full rather than passed to a resampler with zero total weight.
        if threshold == math.inf or not bool(xp.any(survivors)):
            terminal = True
            reason = reason or "no_survivors"
        elif (
            threshold == runtime["threshold"]
            and runtime["next_threshold"] is not None
        ):
            raise ValueError(
                "Likelihood threshold must increase between levels."
            )
        elif isinstance(
            self.schedule, QuantileLikelihoodThresholdSchedule
        ) and not bool(xp.any(active & ~survivors)):
            raise ValueError(
                "Adaptive threshold made no progress; use a strict constraint."
            )
        runtime["next_threshold"] = threshold
        runtime["is_terminal"] = terminal
        runtime["stop_reason"] = reason
        context = context.with_metadata(nested_sampling=runtime)
        return state, context, None


class ShellEvidenceStep(_ConstraintStep):
    """Integrate a disjoint likelihood shell and optionally retain its particles."""

    def __init__(
        self,
        target: LikelihoodConstrainedPrior,
        *,
        store_shell_particles: bool = True,
    ) -> None:
        self.target = target
        self.store_shell_particles = store_shell_particles

    def apply(
        self, state: ParticleState, context: StepContext, rng: RandomGenerator
    ) -> tuple[ParticleState, StepContext, dict[str, Any]]:
        state, context, values = _prepare_likelihood(
            self.target, state, context
        )
        runtime = _nested_sampling_metadata(context.metadata)
        if runtime["next_threshold"] is None:
            raise ValueError(
                "Select a likelihood threshold before recording a shell."
            )
        xp = namespace(values)
        weights = _normalized_weights(state)
        if runtime["is_terminal"]:
            mask = xp.ones_like(values, dtype=bool)
        elif self.target.inclusive:
            mask = values < runtime["next_threshold"]
        else:
            mask = values <= runtime["next_threshold"]
        mask = mask & xp.isfinite(weights)
        shell_weights = xp.where(mask, weights + values, -math.inf)
        log_expectation = float(logsumexp(shell_weights))
        log_increment = runtime["log_prior_volume"] + log_expectation
        runtime["log_nested_evidence"] = float(
            np.logaddexp(runtime["log_nested_evidence"], log_increment)
        )
        n_surviving = int(xp.sum(xp.isfinite(weights) & ~mask))
        survival_fraction = float(xp.sum(xp.where(mask, 0.0, xp.exp(weights))))
        level = NestedSamplingLevel(
            iteration=int(context.t),
            shell_lower_threshold=runtime["threshold"],
            shell_upper_threshold=None
            if runtime["is_terminal"]
            else runtime["next_threshold"],
            log_prior_volume=runtime["log_prior_volume"],
            log_shell_expectation=log_expectation,
            log_evidence_increment=log_increment,
            log_evidence=runtime["log_nested_evidence"],
            is_terminal=runtime["is_terminal"],
            n_surviving=n_surviving,
            survival_fraction=survival_fraction,
        )
        output: dict[str, Any] = {
            NestedSamplingLevel.output_key: asdict(level)
        }
        if self.store_shell_particles:
            output["particles"] = array_copy(state.particles[mask])
            output["log_weights"] = array_copy(
                shell_weights[mask] + runtime["log_prior_volume"]
            )
        if runtime["is_terminal"]:
            context = context.request_stop_immediately()
        context = context.with_metadata(nested_sampling=runtime)
        if runtime["is_terminal"]:
            state = _record_nested_sampling_snapshot(state, context)
        return state, context, output


class LikelihoodThresholdWeighting(_ConstraintStep):
    """Normalize survivor weights and accumulate constrained-prior mass."""

    def __init__(self, target: LikelihoodConstrainedPrior) -> None:
        self.target = target

    def apply(
        self, state: ParticleState, context: StepContext, rng: RandomGenerator
    ) -> tuple[ParticleState, StepContext, None]:
        state, context, values = _prepare_likelihood(
            self.target, state, context
        )
        runtime = _nested_sampling_metadata(context.metadata)
        threshold = runtime["next_threshold"]
        if threshold is None or runtime["is_terminal"]:
            raise ValueError(
                "Select a nonterminal likelihood threshold before weighting."
            )
        mask = (
            values >= threshold
            if self.target.inclusive
            else values > threshold
        )
        xp = namespace(values)
        weights, norm = normalize_log_weights(
            xp.where(mask, _normalized_weights(state), -math.inf), xp=xp
        )
        _validate_log_normalizer(norm, xp=xp)
        runtime["threshold"] = threshold
        runtime["n_surviving"] = int(xp.sum(xp.isfinite(weights)))
        runtime["log_prior_volume_increment"] = float(norm)
        runtime["log_prior_volume"] += float(norm)
        runtime["survival_fraction"] = math.exp(float(norm))
        state = state.with_updates(
            log_weights=weights,
            log_normalizing_constant=runtime["log_prior_volume"],
        )
        context = context.with_metadata(nested_sampling=runtime)
        state = _record_nested_sampling_snapshot(state, context)
        return state, context, None


class LogPriorMassStoppingCriterion(_ConstraintStep):
    """Finalize the remaining shell when prior mass falls below the given limit.

    Unlike the original prototype, stopping includes the surviving particles'
    evidence and posterior contributions before requesting an immediate stop.
    """

    def __init__(
        self,
        min_log_prior_volume: float,
        target: LikelihoodConstrainedPrior,
        *,
        store_shell_particles: bool = True,
    ) -> None:
        if (
            not math.isfinite(min_log_prior_volume)
            or min_log_prior_volume >= 0
        ):
            raise ValueError(
                "min_log_prior_volume must be finite and negative."
            )
        self.minimum = min_log_prior_volume
        self.shell = ShellEvidenceStep(
            target, store_shell_particles=store_shell_particles
        )

    def apply(
        self, state: ParticleState, context: StepContext, rng: RandomGenerator
    ) -> tuple[ParticleState, StepContext, Any]:
        runtime = _nested_sampling_metadata(context.metadata)
        if runtime["log_prior_volume"] > self.minimum:
            return state, context, None
        runtime["is_terminal"] = True
        runtime["next_threshold"] = math.inf
        runtime["stop_reason"] = "prior_mass"
        context = context.with_metadata(nested_sampling=runtime)
        return self.shell(state, context, rng)
