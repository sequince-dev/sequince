"""Metric interfaces and collection lifecycle."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from typing import Any, Protocol

from ....array_api import asarray, namespace
from ....typing import Array
from ...context import StepContext
from .state import MutationMetricsOutput, MutationStepState, MutationTransition


class MutationMetric(Protocol):
    """Static metric configuration with explicit, fixed-structure pytree state.

    ``update`` must be pure: (old state, transition) -> new state. Never
    mutate Python attributes or input arrays. Arrays, tuples, dictionaries and
    registered pytree dataclasses are all valid states. Shapes and dtypes must
    remain fixed under JIT. ``accepted`` is a per-particle boolean mask.

    An optional ``prepare(state, particles, context)`` method prepares existing
    state for the next mutation segment, preserving accumulated measurements.
    It must return state without mutating its input; absent means no-op.

    Every update accepts the keyword-only ``transition`` argument. Metrics
    needing displacement and kernel auxiliary declare ``needs_transition = True``;
    otherwise mutations may omit this work and pass None. The flag is optional
    and defaults to False. Auxiliary data remains opaque to the collection.
    """

    @property
    def name(self) -> str: ...

    def initialize(
        self, particles: Array, log_weights: Array, context: StepContext
    ) -> Any: ...
    def update(
        self,
        state: Any,
        particles: Array,
        accepted: Array,
        context: StepContext,
        *,
        transition: MutationTransition | None = None,
    ) -> Any: ...
    def result(self, state: Any, context: StepContext) -> Any: ...


@dataclass(frozen=True, init=False)
class MutationMetrics:
    """Static metric collection with pure operations on MutationStepState.

    Initialize once per block, update once per population transition, and
    compute results only at the complete block boundary. Preparing the next
    segment preserves accumulated measurements. The collection holds
    configuration only; runtime state is always returned explicitly.
    """

    _metrics: tuple[MutationMetric, ...]
    needs_transition: bool

    def __init__(self, metrics: Sequence[MutationMetric] = ()) -> None:
        configured = tuple(metrics)
        names = [metric.name for metric in configured]
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError(
                "Mutation metric names must be non-empty strings."
            )
        if len(set(names)) != len(names):
            raise ValueError(
                "Mutation metric names must be unique (duplicate name)."
            )
        object.__setattr__(self, "_metrics", configured)
        object.__setattr__(
            self,
            "needs_transition",
            any(
                getattr(metric, "needs_transition", False)
                for metric in configured
            ),
        )

    def __iter__(self) -> Iterator[MutationMetric]:
        return iter(self._metrics)

    def __len__(self) -> int:
        return len(self._metrics)

    def initialize(
        self,
        particles: Array,
        log_weights: Array,
        context: StepContext,
    ) -> MutationStepState:
        """Initialize metric state and zero acceptance counters for a new block."""
        xp = namespace(particles)
        return MutationStepState(
            tuple(
                metric.initialize(particles, log_weights, context)
                for metric in self._metrics
            ),
            xp.asarray(0),
            xp.asarray(0),
            xp.asarray(0),
        )

    def prepare(
        self,
        metric_state: MutationStepState,
        particles: Array,
        context: StepContext,
    ) -> MutationStepState:
        """Prepare existing metric state for the next segment, retaining totals."""
        return replace(
            metric_state,
            states=tuple(
                metric.prepare(state, particles, context)
                if hasattr(metric, "prepare")
                else state
                for metric, state in zip(
                    self._metrics, metric_state.states, strict=True
                )
            ),
        )

    def update(
        self,
        metric_state: MutationStepState,
        particles: Array,
        accepted: Array,
        context: StepContext,
        *,
        transition: MutationTransition | None = None,
    ) -> MutationStepState:
        """Record one population transition without changing the input state."""
        xp = namespace(particles)
        return replace(
            metric_state,
            states=tuple(
                metric.update(
                    state,
                    particles,
                    accepted,
                    context,
                    transition=transition,
                )
                for metric, state in zip(
                    self._metrics,
                    metric_state.states,
                    strict=True,
                )
            ),
            n_steps=metric_state.n_steps + 1,
            n_attempted=metric_state.n_attempted + particles.shape[0],
            n_accepted=metric_state.n_accepted
            + xp.sum(asarray(accepted, xp=xp)),
        )

    def result(
        self,
        metric_state: MutationStepState | None,
        context: StepContext,
    ) -> MutationMetricsOutput | None:
        """Collect final results at the outermost block boundary."""
        if metric_state is None:
            return None
        return MutationMetricsOutput(
            metric_state.n_steps,
            metric_state.n_attempted,
            metric_state.n_accepted,
            {
                metric.name: metric.result(state, context)
                for metric, state in zip(
                    self._metrics, metric_state.states, strict=True
                )
            },
        )


def with_metrics(output, metrics):
    if isinstance(output, dict):
        return {**output, "metrics": metrics}
    return replace(output, metrics=metrics)


__all__ = ["MutationMetric", "MutationMetrics"]
