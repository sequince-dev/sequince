from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ._array_api import copy as array_copy
from ._array_api import namespace
from ._typing import Array

if TYPE_CHECKING:
    from .steps.context import StepContext


@dataclass(frozen=True)
class TargetEvaluation:
    """Named, particle-aligned values evaluated by an annealed target."""

    components: Mapping[str, Array]

    def __post_init__(self) -> None:
        components = dict(self.components)
        if not components:
            raise ValueError("Target evaluation must contain a component.")
        arrays = list(components.values())
        if any(array.ndim == 0 for array in arrays):
            raise ValueError(
                "Target evaluation arrays must be particle-aligned."
            )
        n_particles = arrays[0].shape[0]
        if any(array.shape[0] != n_particles for array in arrays):
            raise ValueError(
                "Target evaluation arrays must have the same leading dimension."
            )
        object.__setattr__(self, "components", components)

    def __contains__(self, component: str) -> bool:
        return component in self.components

    def __getitem__(self, component: str) -> Array:
        return self.components[component]

    @property
    def n_particles(self) -> int:
        """Number of particle-aligned entries in the evaluation."""
        return next(iter(self.components.values())).shape[0]

    def validate_for(self, particles: Array) -> None:
        """Validate that this evaluation has one entry per supplied particle."""
        n_particles = 1 if particles.ndim == 1 else particles.shape[0]
        if self.n_particles != n_particles:
            raise ValueError(
                "Target evaluation has "
                f"{self.n_particles} entries but received "
                f"{n_particles} particles."
            )

    def copy(self) -> TargetEvaluation:
        """Return a detached copy preserving array backends."""
        return type(self)(
            components={
                key: array_copy(value, xp=namespace(value))
                for key, value in self.components.items()
            },
        )

    def take(self, indices: Array) -> TargetEvaluation:
        """Select particle evaluations using ancestor indices."""
        return type(self)(
            components={
                key: value[indices] for key, value in self.components.items()
            },
        )

    @classmethod
    def concatenate(
        cls, evaluations: Collection[TargetEvaluation]
    ) -> TargetEvaluation:
        """Concatenate evaluations from multiple particle batches."""
        items = list(evaluations)
        if not items:
            raise ValueError("No target evaluations to concatenate.")
        component_keys = items[0].components.keys()
        if any(item.components.keys() != component_keys for item in items[1:]):
            raise ValueError("Target evaluations have different components.")
        return cls(
            components={
                key: namespace(items[0][key]).concat(
                    tuple(item[key] for item in items)
                )
                for key in component_keys
            },
        )

    def select(
        self,
        proposed: TargetEvaluation,
        accepted: Array,
    ) -> TargetEvaluation:
        """Select proposed values for accepted particles."""
        if self.components.keys() != proposed.components.keys():
            raise ValueError("Target evaluations have different components.")

        def choose(current: Array, candidate: Array) -> Array:
            xp = namespace(current, candidate, accepted)
            mask = accepted
            if current.ndim > 1:
                mask = xp.reshape(
                    accepted,
                    (accepted.shape[0],) + (1,) * (current.ndim - 1),
                )
            return xp.where(mask, candidate, current)

        return type(self)(
            components={
                key: choose(value, proposed[key])
                for key, value in self.components.items()
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {"components": dict(self.components)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TargetEvaluation:
        components = payload.get("components")
        if not isinstance(components, Mapping):
            raise TypeError(
                "Target evaluation payload must contain a component mapping."
            )
        return cls(components=dict(components))


class AnnealedTarget(ABC):
    """Interface for targets following a geometric annealing path."""

    @abstractmethod
    def evaluate(
        self,
        particles: Array,
        context: StepContext,
        *,
        components: Collection[str] | None = None,
        evaluation: TargetEvaluation | None = None,
        **kwargs: Any,
    ) -> TargetEvaluation:
        """Evaluate or refresh named components of the target."""

    def refresh(
        self,
        particles: Array,
        evaluation: TargetEvaluation,
        context: StepContext,
        *,
        components: Collection[str],
        **kwargs: Any,
    ) -> TargetEvaluation:
        """Refresh named components for an unchanged particle population."""
        return self.evaluate(
            particles,
            context,
            components=components,
            evaluation=evaluation,
            **kwargs,
        )

    @abstractmethod
    def log_prob(
        self,
        particles: Array,
        context: StepContext,
        *,
        evaluation: TargetEvaluation | None = None,
        **kwargs: Any,
    ) -> Array | float:
        """Return log density at the current annealing parameter."""

    @abstractmethod
    def log_increment(
        self,
        particles: Array,
        context: StepContext,
        *,
        evaluation: TargetEvaluation | None = None,
        **kwargs: Any,
    ) -> Array | float:
        """Return the statistic multiplied by an annealing increment."""
