from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Collection, Hashable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable
from uuid import uuid4

from ._typing import Array
from .array_api import asarray, namespace
from .array_api import copy as array_copy
from .utils import call_with_optional_context

if TYPE_CHECKING:
    from .steps.context import StepContext


logger = logging.getLogger(__name__)

TargetComponentFn = Callable[..., Array | float]


def _evaluate_component(
    component: TargetComponentFn,
    particles: Array,
    context: StepContext,
) -> Array:
    """Evaluate a target component as one value per particle."""
    values = call_with_optional_context(component, (particles,), context)
    xp = namespace(particles, values)
    array = asarray(values, xp=xp, like=particles)
    if array.ndim == 0:
        n_particles = 1 if particles.ndim == 1 else particles.shape[0]
        array = xp.broadcast_to(array, (n_particles,))
    if array.ndim > 1:
        array = xp.reshape(array, (array.shape[0],))
    return array


@dataclass(frozen=True)
class TargetEvaluation:
    """Named, particle-aligned values evaluated by an annealed target."""

    components: Mapping[str, Array]
    target_id: str | None = None
    evaluation_key: Hashable | None = None

    def __post_init__(self) -> None:
        components = dict(self.components)
        if not components:
            raise ValueError("Target evaluation must contain a component.")
        if self.evaluation_key is not None and not isinstance(
            self.evaluation_key, Hashable
        ):
            raise TypeError("Target evaluation key must be hashable.")
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

    def with_target(
        self,
        target_id: str,
        evaluation_key: Hashable | None,
    ) -> TargetEvaluation:
        """Return this evaluation associated with a sequential target state."""
        if (
            self.target_id == target_id
            and self.evaluation_key == evaluation_key
        ):
            return self
        return type(self)(
            components=self.components,
            target_id=target_id,
            evaluation_key=evaluation_key,
        )

    def copy(self) -> TargetEvaluation:
        """Return a detached copy preserving array backends."""
        return type(self)(
            components={
                key: array_copy(value, xp=namespace(value))
                for key, value in self.components.items()
            },
            target_id=self.target_id,
            evaluation_key=self.evaluation_key,
        )

    def take(self, indices: Array) -> TargetEvaluation:
        """Select particle evaluations using ancestor indices."""
        return type(self)(
            components={
                key: value[indices] for key, value in self.components.items()
            },
            target_id=self.target_id,
            evaluation_key=self.evaluation_key,
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
        evaluation_key = items[0].evaluation_key
        if any(item.evaluation_key != evaluation_key for item in items[1:]):
            raise ValueError(
                "Target evaluations belong to different target stages."
            )
        return cls(
            components={
                key: namespace(items[0][key]).concat(
                    tuple(item[key] for item in items)
                )
                for key in component_keys
            },
            target_id=target_id,
            evaluation_key=evaluation_key,
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
        if self.evaluation_key != proposed.evaluation_key:
            raise ValueError(
                "Target evaluations belong to different target stages."
            )

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
            evaluation_key=self.evaluation_key,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "components": dict(self.components),
            "target_id": self.target_id,
            "evaluation_key": self.evaluation_key,
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
        evaluation_key = payload.get("evaluation_key")
        if evaluation_key is not None and not isinstance(
            evaluation_key, Hashable
        ):
            raise TypeError("Target evaluation key must be hashable.")
        return cls(
            components=dict(components),
            target_id=target_id,
            evaluation_key=evaluation_key,
        )


class SequentialTarget(ABC):
    """Interface for a sequence of targets and their log-weight increments."""

    _evaluation_id: str

    def __new__(cls, *args: Any, **kwargs: Any) -> SequentialTarget:
        instance = super().__new__(cls)
        object.__setattr__(instance, "_evaluation_id", uuid4().hex)
        return instance

    @property
    def evaluation_id(self) -> str:
        """Opaque identity used to validate cached target evaluations."""
        return self._evaluation_id

    def evaluation_key(
        self,
        context: StepContext,
    ) -> Hashable | None:
        """Return the state of this target relevant to cached evaluations."""
        return None

    def accepts_evaluation(
        self,
        evaluation: TargetEvaluation,
        context: StepContext,
    ) -> bool:
        """Return whether an evaluation was produced by this target."""
        return (
            evaluation.target_id == self.evaluation_id
            and evaluation.evaluation_key == self.evaluation_key(context)
        )

    def bind_evaluation(
        self,
        evaluation: TargetEvaluation,
        context: StepContext,
    ) -> TargetEvaluation:
        """Associate an evaluation returned by this target with its identity."""
        return evaluation.with_target(
            self.evaluation_id,
            self.evaluation_key(context),
        )

    def prepare_evaluation(
        self,
        particles: Array,
        context: StepContext,
        *,
        evaluation: TargetEvaluation | None = None,
        refresh_components: Collection[str] = (),
    ) -> TargetEvaluation:
        """Return a compatible evaluation, refreshing it when required."""
        if evaluation is None:
            prepared = self.evaluate(particles, context)
        elif not self.accepts_evaluation(evaluation, context):
            logger.debug(
                "Discarding evaluation from target %s at key %r; "
                "current %s target is %s at key %r.",
                evaluation.target_id,
                evaluation.evaluation_key,
                type(self).__name__,
                self.evaluation_id,
                self.evaluation_key(context),
            )
            prepared = self.evaluate(particles, context)
        elif refresh_components:
            evaluation.validate_for(particles)
            prepared = self.refresh(
                particles,
                evaluation,
                context,
                components=refresh_components,
            )
        else:
            evaluation.validate_for(particles)
            return evaluation
        prepared.validate_for(particles)
        return self.bind_evaluation(prepared, context)

    def validate_evaluation(
        self,
        evaluation: TargetEvaluation,
        particles: Array,
        context: StepContext,
    ) -> None:
        """Validate particle alignment and ownership of an evaluation."""
        evaluation.validate_for(particles)
        if not self.accepts_evaluation(evaluation, context):
            raise ValueError(
                "Target evaluation belongs to a different target or stage."
            )

    @abstractmethod
    def evaluate(
        self,
        particles: Array,
        context: StepContext,
        *,
        components: Collection[str] | None = None,
        evaluation: TargetEvaluation | None = None,
    ) -> TargetEvaluation:
        """Evaluate or refresh named components of the target."""

    def refresh(
        self,
        particles: Array,
        evaluation: TargetEvaluation,
        context: StepContext,
        *,
        components: Collection[str],
    ) -> TargetEvaluation:
        """Refresh named components for an unchanged particle population."""
        return self.evaluate(
            particles,
            context,
            components=components,
            evaluation=evaluation,
        )

    @abstractmethod
    def log_prob(
        self,
        particles: Array,
        context: StepContext,
        *,
        evaluation: TargetEvaluation | None = None,
    ) -> Array | float:
        """Return the current target log density."""

    @abstractmethod
    def log_increment(
        self,
        particles: Array,
        context: StepContext,
        *,
        evaluation: TargetEvaluation | None = None,
    ) -> Array | float:
        """Return the log-weight increment to the next target."""


class TemperedTarget(SequentialTarget):
    """Target interpolating geometrically between two endpoint densities."""

    @abstractmethod
    def log_initial(
        self,
        particles: Array,
        context: StepContext,
        *,
        evaluation: TargetEvaluation | None = None,
    ) -> Array | float:
        """Return the initial endpoint log density."""

    @abstractmethod
    def log_terminal(
        self,
        particles: Array,
        context: StepContext,
        *,
        evaluation: TargetEvaluation | None = None,
    ) -> Array | float:
        """Return the terminal endpoint log density."""

    def log_prob(
        self,
        particles: Array,
        context: StepContext,
        *,
        evaluation: TargetEvaluation | None = None,
    ) -> Array | float:
        """Return the interpolated log density at the current temperature."""
        if evaluation is None:
            evaluation = self.evaluate(particles, context)
        self.validate_evaluation(evaluation, particles, context)
        beta = float(context.metadata.get("temperature", 0.0))
        if beta == 0.0:
            return self.log_initial(particles, context, evaluation=evaluation)
        if beta == 1.0:
            return self.log_terminal(particles, context, evaluation=evaluation)
        initial = self.log_initial(particles, context, evaluation=evaluation)
        terminal = self.log_terminal(particles, context, evaluation=evaluation)
        return (1.0 - beta) * initial + beta * terminal

    def log_increment(
        self,
        particles: Array,
        context: StepContext,
        *,
        evaluation: TargetEvaluation | None = None,
    ) -> Array | float:
        """Return the terminal-to-initial log-density ratio."""
        if evaluation is None:
            evaluation = self.evaluate(particles, context)
        self.validate_evaluation(evaluation, particles, context)
        initial = self.log_initial(particles, context, evaluation=evaluation)
        terminal = self.log_terminal(particles, context, evaluation=evaluation)
        return terminal - initial
