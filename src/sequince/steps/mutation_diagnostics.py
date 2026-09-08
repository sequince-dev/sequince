"""Functional diagnostics and extensible metrics for fixed mutation blocks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from .._typing import Array
from ..array_api import asarray, logsumexp, namespace
from ..utils.pytrees import pytree_dataclass
from .context import StepContext


class MutationMetric(Protocol):
    """Static metric configuration with explicit, fixed-structure pytree state.

    ``update`` must be pure: (old state, transition) -> new state. Never
    mutate Python attributes or input arrays. Arrays, tuples, dictionaries and
    registered pytree dataclasses are all valid states. Shapes and dtypes must
    remain fixed under JIT. ``accepted`` is a per-particle boolean mask.

    An optional ``reset_reference(state, particles, context)`` method may
    return a new reference after a non-MCMC operation; absent means no-op.
    """

    name: str

    def initialize(
        self, particles: Array, log_weights: Array, context: StepContext
    ) -> Any: ...
    def update(
        self,
        state: Any,
        particles: Array,
        accepted: Array,
        context: StepContext,
    ) -> Any: ...
    def result(self, state: Any, context: StepContext) -> Any: ...


@pytree_dataclass(
    data_fields=("n_steps", "n_attempted", "n_accepted", "metrics")
)
@dataclass(frozen=True)
class MutationDiagnostics:
    """Core bookkeeping and named results for one complete mutation block."""

    n_steps: Any
    n_attempted: Any
    n_accepted: Any
    metrics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def acceptance(self) -> Any:
        accepted = asarray(self.n_accepted)
        xp = namespace(accepted)
        return accepted / xp.maximum(xp.asarray(self.n_attempted), 1)


@pytree_dataclass(
    data_fields=(
        "last_particles",
        "inverse_variance",
        "sum_squared_jump",
        "n_steps",
    )
)
@dataclass(frozen=True)
class ESJDState:
    last_particles: Array
    inverse_variance: Array
    sum_squared_jump: Array
    n_steps: Array


class ESJDMetric:
    """Mean squared jump per particle, dimension and transition.

    Uses mutation coordinates and frozen weighted diagonal pre-block scaling.
    Degenerate variances (<= machine epsilon) are replaced by one.
    """

    name = "esjd"

    def initialize(
        self, particles: Array, log_weights: Array, context: StepContext
    ) -> ESJDState:
        xp = namespace(particles)
        values = xp.reshape(particles, (particles.shape[0], -1))
        weights = xp.exp(log_weights - logsumexp(log_weights, xp=xp))
        mean = xp.sum(values * weights[:, None], axis=0)
        variance = xp.sum(xp.square(values - mean) * weights[:, None], axis=0)
        variance = xp.where(
            variance > xp.finfo(values.dtype).eps,
            variance,
            xp.ones_like(variance),
        )
        return ESJDState(
            values,
            1.0 / variance,
            xp.asarray(0.0, dtype=values.dtype),
            xp.asarray(0),
        )

    def update(
        self,
        state: ESJDState,
        particles: Array,
        accepted: Array,
        context: StepContext,
    ) -> ESJDState:
        xp = namespace(particles)
        values = xp.reshape(particles, state.last_particles.shape)
        jump = xp.sum(
            xp.square(values - state.last_particles) * state.inverse_variance
        )
        return replace(
            state,
            last_particles=values,
            sum_squared_jump=state.sum_squared_jump + jump,
            n_steps=state.n_steps + 1,
        )

    def result(self, state: ESJDState, context: StepContext) -> Array:
        xp = namespace(state.sum_squared_jump)
        n, d = state.last_particles.shape
        return state.sum_squared_jump / xp.maximum(n * d * state.n_steps, 1)

    def reset_reference(
        self, state: ESJDState, particles: Array, context: StepContext
    ) -> ESJDState:
        xp = namespace(particles)
        return replace(
            state,
            last_particles=xp.reshape(particles, state.last_particles.shape),
        )


def validate_metrics(
    metrics: Sequence[MutationMetric],
) -> tuple[MutationMetric, ...]:
    configured = tuple(metrics)
    names = [metric.name for metric in configured]
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("Mutation metric names must be non-empty strings.")
    if len(set(names)) != len(names):
        raise ValueError(
            "Mutation metric names must be unique (duplicate name)."
        )
    return configured


@pytree_dataclass(
    data_fields=("states", "n_steps", "n_attempted", "n_accepted")
)
@dataclass(frozen=True)
class MutationStepState:
    """Immutable pytree state for continuing a mutation block.

    Normally obtained from ``apply_with_mutation_state`` by passing ``None``
    for the initial state, then passed unchanged into the next call. ``states``
    holds individual metric states in configured metric order; the counters
    accumulate across the complete block. This is runtime state, not finalized
    diagnostics. Its pytree structure, leaf shapes and dtypes must remain fixed
    inside compiled loops.
    """

    states: tuple[Any, ...]
    n_steps: Any
    n_attempted: Any
    n_accepted: Any


def prepare_metrics(
    metrics: Sequence[MutationMetric],
    metric_state: MutationStepState | None,
    particles: Array,
    log_weights: Array,
    context: StepContext,
) -> MutationStepState:
    """Initialize once, or refresh references at a participant boundary."""
    if metric_state is not None:
        # Each participant may have crossed a symmetry or coordinate operation.
        # Reset only references; the original scaling and totals remain frozen.
        return replace(
            metric_state,
            states=tuple(
                metric.reset_reference(state, particles, context)
                if hasattr(metric, "reset_reference")
                else state
                for metric, state in zip(
                    metrics, metric_state.states, strict=True
                )
            ),
        )
    xp = namespace(particles)
    return MutationStepState(
        tuple(
            metric.initialize(particles, log_weights, context)
            for metric in metrics
        ),
        xp.asarray(0),
        xp.asarray(0),
        xp.asarray(0),
    )


def update_metrics(
    metrics: Sequence[MutationMetric],
    metric_state: MutationStepState,
    particles: Array,
    accepted: Array,
    context: StepContext,
) -> MutationStepState:
    """Record one population transition without changing any input state."""
    xp = namespace(particles)
    return replace(
        metric_state,
        states=tuple(
            metric.update(state, particles, accepted, context)
            for metric, state in zip(metrics, metric_state.states, strict=True)
        ),
        n_steps=metric_state.n_steps + 1,
        n_attempted=metric_state.n_attempted + particles.shape[0],
        n_accepted=metric_state.n_accepted + xp.sum(asarray(accepted, xp=xp)),
    )


def finish_metrics(
    metrics: Sequence[MutationMetric],
    metric_state: MutationStepState | None,
    context: StepContext,
) -> MutationDiagnostics | None:
    """Finalize only at the outermost block boundary."""
    if metric_state is None:
        return None
    return MutationDiagnostics(
        metric_state.n_steps,
        metric_state.n_attempted,
        metric_state.n_accepted,
        {
            metric.name: metric.result(state, context)
            for metric, state in zip(metrics, metric_state.states, strict=True)
        },
    )


def with_diagnostics(output, diagnostics):
    if isinstance(output, dict):
        return {**output, "diagnostics": diagnostics}
    return replace(output, diagnostics=diagnostics)


__all__ = [
    "MutationMetric",
    "MutationDiagnostics",
    "MutationStepState",
    "ESJDMetric",
    "ESJDState",
]
