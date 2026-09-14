from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Collection, Hashable, Mapping
from dataclasses import dataclass
from numbers import Real
from typing import TYPE_CHECKING, Any, Callable, Literal, Self, cast, overload
from uuid import uuid4

import array_api_extra as xpx
import numpy as np

from ._typing import Array
from .array_api import asarray, namespace
from .array_api import copy as array_copy
from .utils.callables import call_with_optional_context
from .utils.pytrees import pytree_dataclass

if TYPE_CHECKING:
    from .steps.context import StepContext


logger = logging.getLogger(__name__)

TargetComponentFn = Callable[..., Array | float]


@pytree_dataclass(
    data_fields=(
        "log_target_evaluations",
        "log_likelihood_evaluations",
        "log_proposal_evaluations",
        "log_increment_evaluations",
    )
)
@dataclass(frozen=True)
class EvaluationCounts:
    """Backend-compatible counters for probabilistic model evaluations.

    The fields count work required to evaluate the log target density, log
    likelihood, log proposal, and log increment. Log-target work includes any
    refreshed target-density component, while log-likelihood work records only
    calls to ``log_likelihood``. Reusing cached components does not count again.
    """

    log_target_evaluations: Any = 0
    log_likelihood_evaluations: Any = 0
    log_proposal_evaluations: Any = 0
    log_increment_evaluations: Any = 0

    def __add__(self, other: EvaluationCounts) -> EvaluationCounts:
        if not isinstance(other, EvaluationCounts):
            return NotImplemented
        return type(self)(
            log_target_evaluations=(
                self.log_target_evaluations + other.log_target_evaluations
            ),
            log_likelihood_evaluations=(
                self.log_likelihood_evaluations
                + other.log_likelihood_evaluations
            ),
            log_proposal_evaluations=(
                self.log_proposal_evaluations + other.log_proposal_evaluations
            ),
            log_increment_evaluations=(
                self.log_increment_evaluations
                + other.log_increment_evaluations
            ),
        )

    @classmethod
    def zeros(cls, *, xp: Any | None = None) -> EvaluationCounts:
        """Return zero counts, optionally as scalars from an array namespace."""
        if xp is None:
            return cls()
        zero = xp.asarray(0)
        return cls(zero, zero, zero, zero)


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


@pytree_dataclass(
    data_fields=("components",),
    meta_fields=("target_id", "evaluation_key"),
)
@dataclass(frozen=True)
class TargetEvaluation:
    """Named, particle-aligned values evaluated by a target."""

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
        if any(len(array.shape) == 0 for array in arrays):
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
        """Return this evaluation associated with a target state."""
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

    def replace(
        self,
        indices: Array,
        replacement: TargetEvaluation,
    ) -> TargetEvaluation:
        """Return an evaluation with indexed particle rows replaced."""
        if self.components.keys() != replacement.components.keys():
            raise ValueError("Target evaluations have different components.")
        if self.target_id != replacement.target_id:
            raise ValueError("Target evaluations belong to different targets.")
        if self.evaluation_key != replacement.evaluation_key:
            raise ValueError(
                "Target evaluations belong to different target stages."
            )
        if indices.ndim != 1:
            raise ValueError(
                "Target evaluation indices must be one-dimensional."
            )
        if replacement.n_particles != indices.shape[0]:
            raise ValueError(
                "Replacement evaluation must contain one entry per index."
            )

        return type(self)(
            components={
                key: xpx.at(value, indices).set(
                    replacement[key],
                    copy=True,
                    xp=namespace(value),
                )
                for key, value in self.components.items()
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

    def encode_for_hdf5(self) -> dict[str, Any]:
        """Return the versioned portable HDF5 payload."""
        from .utils.hdf5 import schema_header

        _validate_portable_evaluation_key(
            self.evaluation_key, "evaluation_key"
        )
        return {
            **schema_header("target_evaluation"),
            "components": dict(self.components),
            "target_id": self.target_id,
            "evaluation_key": self.evaluation_key,
        }

    @classmethod
    def _decode_from_hdf5(cls, payload: Mapping[str, Any]) -> Self:
        """Reconstruct an evaluation from its logical HDF5 payload."""
        from .utils.hdf5 import validate_schema

        validate_schema(
            payload,
            kinds={"target_evaluation"},
            required={"components", "target_id", "evaluation_key"},
        )
        _validate_portable_evaluation_key(
            payload["evaluation_key"], "evaluation_key"
        )
        return cast(Self, cls.from_dict(payload))

    def save(
        self, h5_file_or_group: Any, path: str = "target_evaluation"
    ) -> None:
        """Save this evaluation below ``path`` in an open HDF5 handle."""
        from .utils.hdf5 import recursively_save_to_h5_file

        recursively_save_to_h5_file(
            h5_file_or_group, path, self.encode_for_hdf5()
        )

    @classmethod
    def load(
        cls, h5_file_or_group: Any, path: str = "target_evaluation"
    ) -> Self:
        """Load an evaluation from ``path`` in an open HDF5 handle."""
        from .utils.hdf5 import load_from_h5_file

        return cls._decode_from_hdf5(load_from_h5_file(h5_file_or_group, path))


def _validate_portable_evaluation_key(value: Any, path: str) -> None:
    if value is None or isinstance(value, (str, bytes, bool, np.bool_)):
        return
    if isinstance(value, (int, float, complex, np.number)):
        return
    if isinstance(value, tuple):
        for index, child in enumerate(value):
            _validate_portable_evaluation_key(child, f"{path}/{index}")
        return
    raise TypeError(
        f"Unsupported target evaluation key type {type(value).__name__} "
        f"at {path}."
    )


class Target(ABC):
    """Interface for evaluating a target distribution."""

    _evaluation_id: str

    def __new__(cls, *args: Any, **kwargs: Any) -> Target:
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

    def evaluation_components(
        self, context: StepContext
    ) -> Collection[str] | None:
        """Components needed for a full density evaluation, when known.

        These are the components used to compute ``log_prob``, such as during
        MCMC. They need not include every component the target supports:
        ``log_increment`` is included only when the density itself requires it.
        Other components can still be requested explicitly through
        ``prepare_evaluation(components=...)``.

        When ``components=None``, preparation uses this declaration to complete
        a partial cache and retain only these components. Returning None
        preserves the default full-evaluation contract for custom targets.
        """
        return None

    @overload
    def prepare_evaluation(
        self,
        particles: Array,
        context: StepContext,
        *,
        evaluation: TargetEvaluation | None = None,
        components: Collection[str] | None = None,
        refresh_components: Collection[str] = (),
        return_counts: Literal[False] = False,
    ) -> TargetEvaluation: ...

    @overload
    def prepare_evaluation(
        self,
        particles: Array,
        context: StepContext,
        *,
        evaluation: TargetEvaluation | None = None,
        components: Collection[str] | None = None,
        refresh_components: Collection[str] = (),
        return_counts: Literal[True],
    ) -> tuple[TargetEvaluation, EvaluationCounts]: ...

    def prepare_evaluation(
        self,
        particles: Array,
        context: StepContext,
        *,
        evaluation: TargetEvaluation | None = None,
        components: Collection[str] | None = None,
        refresh_components: Collection[str] = (),
        return_counts: bool = False,
    ) -> TargetEvaluation | tuple[TargetEvaluation, EvaluationCounts]:
        """Return a compatible evaluation, optionally with new work counts.

        ``components`` requests named values, filling any missing cached values.
        Full requests use ``evaluation_components`` when declared. ``refresh_components``
        requests recomputation when a compatible cached evaluation is supplied.
        """
        xp = namespace(particles)
        required = (
            self.evaluation_components(context)
            if components is None
            else components
        )
        if evaluation is None:
            prepared, counts = self._evaluate(
                particles,
                context,
                components=components,
            )
        elif not self.accepts_evaluation(evaluation, context):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Discarding evaluation from target %s at key %r; "
                    "current %s target is %s at key %r.",
                    evaluation.target_id,
                    evaluation.evaluation_key,
                    type(self).__name__,
                    self.evaluation_id,
                    self.evaluation_key(context),
                )
            prepared, counts = self._evaluate(
                particles,
                context,
                components=components,
            )
        else:
            evaluation.validate_for(particles)
            missing = (
                set()
                if required is None
                else set(required).difference(evaluation.components)
            )
            needed = missing | set(refresh_components)
            if refresh_components and components is not None:
                needed.update(components)
            if not needed:
                prepared = evaluation
                counts = EvaluationCounts.zeros(xp=xp)
            else:
                prepared, counts = self._evaluate(
                    particles,
                    context,
                    evaluation=evaluation,
                    components=needed,
                    refresh=bool(refresh_components),
                )
        if (
            components is None
            and required is not None
            and set(prepared.components).difference(required)
        ):
            prepared = TargetEvaluation(
                {name: prepared[name] for name in required}
            )
        prepared.validate_for(particles)
        prepared = self.bind_evaluation(prepared, context)
        return (prepared, counts) if return_counts else prepared

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

    def _evaluate(
        self,
        particles: Array,
        context: StepContext,
        *,
        components: Collection[str] | None = None,
        evaluation: TargetEvaluation | None = None,
        refresh: bool = False,
    ) -> tuple[TargetEvaluation, EvaluationCounts]:
        """Evaluate the target and report newly incurred work.

        Custom targets default to one ``log_prob`` evaluation per particle.
        They may override this method when they can classify their component
        work more precisely.
        """
        if refresh:
            if evaluation is None or components is None:
                raise ValueError(
                    "Refreshing requires an evaluation and component names."
                )
            result = self.refresh(
                particles,
                evaluation,
                context,
                components=components,
            )
        else:
            result = self.evaluate(
                particles,
                context,
                components=components,
                evaluation=evaluation,
            )
        xp = namespace(particles)
        n_particles = 1 if particles.ndim == 1 else particles.shape[0]
        return result, EvaluationCounts(
            log_target_evaluations=xp.asarray(n_particles)
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


class SequentialTarget(Target):
    """Target that also defines sequential importance-weight increments."""

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
        """Return the interpolated log density at the current beta."""
        if evaluation is None:
            evaluation = self.evaluate(particles, context)
        self.validate_evaluation(evaluation, particles, context)
        beta = context.metadata.get("beta", 0.0)
        if isinstance(beta, Real):
            if beta == 0.0:
                return self.log_initial(
                    particles, context, evaluation=evaluation
                )
            if beta == 1.0:
                return self.log_terminal(
                    particles, context, evaluation=evaluation
                )
        initial = self.log_initial(particles, context, evaluation=evaluation)
        terminal = self.log_terminal(particles, context, evaluation=evaluation)
        xp = namespace(particles)
        beta = asarray(beta, xp=xp, like=particles)
        initial = xp.where(beta == 1.0, 0.0, initial)
        terminal = xp.where(beta == 0.0, 0.0, terminal)
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
