from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from ._array_api import copy as array_copy
from ._array_api import namespace
from ._typing import Array

if TYPE_CHECKING:
    from .steps.context import StepContext


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TargetEvaluation:
    """Named, particle-aligned values evaluated by an annealed target."""

    components: Mapping[str, Array]
    target_id: str | None = None

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

    def with_target_id(self, target_id: str) -> TargetEvaluation:
        """Return this evaluation associated with an annealed target."""
        if self.target_id == target_id:
            return self
        return type(self)(components=self.components, target_id=target_id)

    def copy(self) -> TargetEvaluation:
        """Return a detached copy preserving array backends."""
        return type(self)(
            components={
                key: array_copy(value, xp=namespace(value))
                for key, value in self.components.items()
            },
            target_id=self.target_id,
        )

    def take(self, indices: Array) -> TargetEvaluation:
        """Select particle evaluations using ancestor indices."""
        return type(self)(
            components={
                key: value[indices] for key, value in self.components.items()
            },
            target_id=self.target_id,
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
        target_id = items[0].target_id
        if any(item.target_id != target_id for item in items[1:]):
            raise ValueError("Target evaluations belong to different targets.")
        return cls(
            components={
                key: namespace(items[0][key]).concat(
                    tuple(item[key] for item in items)
                )
                for key in component_keys
            },
            target_id=target_id,
        )

    def select(
        self,
        proposed: TargetEvaluation,
        accepted: Array,
    ) -> TargetEvaluation:
        """Select proposed values for accepted particles."""
        if self.components.keys() != proposed.components.keys():
            raise ValueError("Target evaluations have different components.")
        if self.target_id != proposed.target_id:
            raise ValueError("Target evaluations belong to different targets.")

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
            target_id=self.target_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "components": dict(self.components),
            "target_id": self.target_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TargetEvaluation:
        components = payload.get("components")
        if not isinstance(components, Mapping):
            raise TypeError(
                "Target evaluation payload must contain a component mapping."
            )
        target_id = payload.get("target_id")
        if target_id is not None and not isinstance(target_id, str):
            raise TypeError("Target evaluation target_id must be a string.")
        return cls(components=dict(components), target_id=target_id)


class AnnealedTarget(ABC):
    """Interface for targets following a geometric annealing path."""

    _evaluation_id: str

    def __new__(cls, *args: Any, **kwargs: Any) -> AnnealedTarget:
        instance = super().__new__(cls)
        object.__setattr__(instance, "_evaluation_id", uuid4().hex)
        return instance

    @property
    def evaluation_id(self) -> str:
        """Opaque identity used to validate cached target evaluations."""
        return self._evaluation_id

    def accepts_evaluation(self, evaluation: TargetEvaluation) -> bool:
        """Return whether an evaluation was produced by this target."""
        return evaluation.target_id == self.evaluation_id

    def bind_evaluation(
        self, evaluation: TargetEvaluation
    ) -> TargetEvaluation:
        """Associate an evaluation returned by this target with its identity."""
        return evaluation.with_target_id(self.evaluation_id)

    def prepare_evaluation(
        self,
        particles: Array,
        context: StepContext,
        *,
        evaluation: TargetEvaluation | None = None,
        refresh_components: Collection[str] = (),
        **kwargs: Any,
    ) -> TargetEvaluation:
        """Return a compatible evaluation, refreshing it when required."""
        if evaluation is None:
            prepared = self.evaluate(particles, context, **kwargs)
        elif not self.accepts_evaluation(evaluation):
            logger.debug(
                "Discarding evaluation from target %s; current %s target is %s.",
                evaluation.target_id,
                type(self).__name__,
                self.evaluation_id,
            )
            prepared = self.evaluate(particles, context, **kwargs)
        elif refresh_components:
            evaluation.validate_for(particles)
            prepared = self.refresh(
                particles,
                evaluation,
                context,
                components=refresh_components,
                **kwargs,
            )
        else:
            evaluation.validate_for(particles)
            return evaluation
        prepared.validate_for(particles)
        return self.bind_evaluation(prepared)

    def validate_evaluation(
        self,
        evaluation: TargetEvaluation,
        particles: Array,
    ) -> None:
        """Validate particle alignment and ownership of an evaluation."""
        evaluation.validate_for(particles)
        if not self.accepts_evaluation(evaluation):
            raise ValueError(
                "Target evaluation belongs to a different target."
            )

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
