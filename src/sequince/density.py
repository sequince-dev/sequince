"""Callable-based sequential and tempered target densities."""

from __future__ import annotations

from collections.abc import Collection, Hashable
from numbers import Real
from typing import Callable

from .steps.context import StepContext
from .target import (
    EvaluationCounts,
    SequentialTarget,
    Target,
    TargetComponentFn,
    TargetEvaluation,
    TemperedTarget,
    _evaluate_component,
)
from .typing import Array


class _DensityComponents(Target):
    _components: dict[str, TargetComponentFn]
    _evaluation_key_fn: Callable[[StepContext], Hashable | None] | None

    def evaluation_key(self, context: StepContext) -> Hashable | None:
        if self._evaluation_key_fn is None:
            return None
        return self._evaluation_key_fn(context)

    def evaluation_components(self, context):
        return set(self._components)

    def _evaluate(
        self,
        particles,
        context,
        *,
        components=None,
        evaluation=None,
        refresh=False,
    ):
        requested = set(
            self.evaluation_components(context)
            if components is None
            else components
        )
        result = self.evaluate(
            particles, context, components=requested, evaluation=evaluation
        )
        n = 1 if particles.ndim == 1 else particles.shape[0]
        density_work = bool(requested.difference({"log_increment"}))
        return result, EvaluationCounts(
            log_target_evaluations=n if density_work else 0,
        )

    def evaluate(
        self,
        particles: Array,
        context: StepContext,
        *,
        components: Collection[str] | None = None,
        evaluation: TargetEvaluation | None = None,
    ) -> TargetEvaluation:
        if evaluation is not None:
            self.validate_evaluation(evaluation, particles, context)
        requested = (
            set(self.evaluation_components(context))
            if components is None
            else set(components)
        )
        unknown = requested.difference(self._components)
        if unknown:
            raise ValueError(
                f"Unknown evaluation components: {sorted(unknown)}"
            )
        values = {} if evaluation is None else dict(evaluation.components)
        for name in requested:
            values[name] = _evaluate_component(
                self._components[name], particles, context
            )
        return self.bind_evaluation(TargetEvaluation(values), context)

    def _value(self, name, particles, context, evaluation):
        if evaluation is not None:
            self.validate_evaluation(evaluation, particles, context)
        evaluation = self.prepare_evaluation(
            particles, context, components={name}, evaluation=evaluation
        )
        values = evaluation[name]
        return values[0] if particles.ndim == 1 else values


class SequentialDensity(_DensityComponents, SequentialTarget):
    """Sequential target defined by density and increment callables.

    ``log_prob`` returns the current target log density for MCMC;
    ``log_increment`` returns the complete weight increment. Both accept
    particles and optionally context. Weighting calls only the increment.
    MCMC evaluates and caches ``log_prob`` separately.

    For context-dependent densities, supply ``evaluation_key(context)`` to
    identify the stage relevant to cached density values, or explicitly refresh
    the evaluation when the density changes. Keys must be hashable; compiled
    execution requires static keys.
    """

    def __init__(
        self,
        log_prob: TargetComponentFn,
        log_increment: TargetComponentFn,
        *,
        evaluation_key: Callable[[StepContext], Hashable | None] | None = None,
    ) -> None:
        self._components = {
            "log_prob": log_prob,
            "log_increment": log_increment,
        }
        self._evaluation_key_fn = evaluation_key

    def evaluation_components(self, context):
        return {"log_prob"}

    def log_prob(self, particles, context, *, evaluation=None):
        return self._value("log_prob", particles, context, evaluation)

    def log_increment(self, particles, context, *, evaluation=None):
        return self._value("log_increment", particles, context, evaluation)


class TemperedDensity(_DensityComponents, TemperedTarget):
    """Tempered target defined by an initial density and a log-density ratio.

    ``log_initial`` and ``log_density_ratio`` accept particles and optionally
    context. The target density is ``log_initial + beta * log_density_ratio``.
    Weighting evaluates only the increment, reusing it from a compatible MCMC
    evaluation when available. MCMC evaluates and caches both components.

    ``evaluation_key`` has the same meaning as for ``SequentialDensity``;
    beta alone does not invalidate these endpoint components.
    """

    def __init__(
        self,
        log_initial: TargetComponentFn,
        log_density_ratio: TargetComponentFn,
        *,
        evaluation_key: Callable[[StepContext], Hashable | None] | None = None,
    ) -> None:
        self._components = {
            "log_initial": log_initial,
            "log_increment": log_density_ratio,
        }
        self._evaluation_key_fn = evaluation_key

    def log_initial(self, particles, context, *, evaluation=None):
        return self._value("log_initial", particles, context, evaluation)

    def log_terminal(self, particles, context, *, evaluation=None):
        if evaluation is None:
            evaluation = self.evaluate(particles, context)
        return self.log_initial(
            particles, context, evaluation=evaluation
        ) + self.log_increment(particles, context, evaluation=evaluation)

    def log_increment(self, particles, context, *, evaluation=None):
        return self._value("log_increment", particles, context, evaluation)

    def log_prob(self, particles, context, *, evaluation=None):
        if evaluation is None:
            evaluation = self.evaluate(particles, context)
        initial = self.log_initial(particles, context, evaluation=evaluation)
        beta = context.metadata.get("beta", 0.0)
        if isinstance(beta, Real) and beta == 0.0:
            return initial
        return initial + beta * self.log_increment(
            particles, context, evaluation=evaluation
        )
