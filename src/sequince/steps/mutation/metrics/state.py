"""Immutable runtime state and completed mutation metrics."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ....array_api import asarray, namespace
from ....utils.pytrees import pytree_dataclass


@pytree_dataclass(
    data_fields=("n_steps", "n_attempted", "n_accepted", "values")
)
@dataclass(frozen=True)
class MutationMetricsOutput:
    """Core bookkeeping and named results for one complete mutation block.

    ``n_accepted`` counts accepted proposals for sampled transitions. Recycling
    algorithms integrating over accept/reject decisions report the expected
    count instead, which may be fractional.
    """

    n_steps: Any
    n_attempted: Any
    n_accepted: Any
    values: Mapping[str, Any] = field(default_factory=dict)

    @property
    def acceptance(self) -> Any:
        accepted = asarray(self.n_accepted)
        xp = namespace(accepted)
        return accepted / xp.maximum(xp.asarray(self.n_attempted), 1)


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
    metrics. Its pytree structure, leaf shapes and dtypes must remain fixed
    inside compiled loops.
    """

    states: tuple[Any, ...]
    n_steps: Any
    n_attempted: Any
    n_accepted: Any


__all__ = ["MutationMetricsOutput", "MutationStepState"]
