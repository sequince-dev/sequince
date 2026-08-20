from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Real
from typing import TYPE_CHECKING, Any, MutableMapping

from array_api_compat import device as get_device

from ._typing import Array, ArrayLike
from .array_api import asarray, namespace
from .array_api import copy as array_copy
from .utils import (
    _validate_log_normalizer,
    array_safe_copy,
    normalize_log_weights,
    pytree_dataclass,
)

if TYPE_CHECKING:
    from .target import TargetEvaluation


@pytree_dataclass(
    data_fields=(
        "particles",
        "log_weights",
        "initial_ancestor_indices",
        "log_evidence",
        "evaluation",
        "metadata",
    )
)
@dataclass(init=False)
class ParticleState:
    """Container for particle locations, log weights, and auxiliary information."""

    particles: Array
    log_weights: Array
    initial_ancestor_indices: Array
    log_evidence: float = 0.0
    evaluation: TargetEvaluation | None = None
    metadata: MutableMapping[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        particles: Array,
        log_weights: Array | Real | None = None,
        log_evidence: float = 0.0,
        initial_ancestor_indices: Array | None = None,
        evaluation: TargetEvaluation | None = None,
        metadata: MutableMapping[str, Any] | None = None,
    ) -> None:
        if particles.shape[0] == 0:
            raise ValueError("Particles array cannot be empty.")

        self.particles = particles
        self.log_evidence = log_evidence
        self.evaluation = evaluation
        self.metadata = dict(metadata or {})
        n = particles.shape[0]
        if initial_ancestor_indices is None:
            xp = namespace(self.particles)
            initial_ancestor_indices = xp.arange(
                n,
                device=get_device(self.particles),
            )
        if initial_ancestor_indices.shape != (n,):
            raise ValueError(
                "Initial ancestor indices must match the particle population."
            )
        self.initial_ancestor_indices = initial_ancestor_indices
        if (
            evaluation is not None
            and evaluation.n_particles != particles.shape[0]
        ):
            raise ValueError(
                "Target evaluation must match the particle population."
            )

        if log_weights is None or isinstance(log_weights, Real):
            xp = namespace(self.particles)
            value = (
                float(log_weights) if log_weights is not None else -math.log(n)
            )
            log_weights = xp.full(
                (n,), value, device=get_device(self.particles)
            )
        self.log_weights = log_weights

    def copy(self) -> "ParticleState":
        """Return a copy of the state preserving the array backend."""
        xp = namespace(
            self.particles,
            self.log_weights,
            self.initial_ancestor_indices,
        )
        return ParticleState(
            particles=array_copy(self.particles, xp=xp),
            log_weights=array_copy(self.log_weights, xp=xp),
            log_evidence=self.log_evidence,
            initial_ancestor_indices=array_copy(
                self.initial_ancestor_indices,
                xp=xp,
            ),
            evaluation=(
                None if self.evaluation is None else self.evaluation.copy()
            ),
            metadata=array_safe_copy(self.metadata),
        )

    def normalize(self) -> tuple["ParticleState", float]:
        """Return a new state with normalized log weights and normalizer."""
        state = self.copy()
        xp = namespace(state.log_weights)
        state.log_weights, log_norm = normalize_log_weights(
            state.log_weights,
            xp=xp,
        )
        _validate_log_normalizer(log_norm, xp=xp)
        return state, log_norm

    @property
    def weights(self) -> Array:
        """Return normalized weights."""
        xp = namespace(self.log_weights)
        normalized, log_norm = normalize_log_weights(self.log_weights, xp=xp)
        _validate_log_normalizer(log_norm, xp=xp)
        return xp.exp(normalized)

    def effective_sample_size(self) -> float:
        """Compute the effective sample size (ESS)."""
        xp = namespace(self.log_weights)
        w = asarray(self.weights, xp=xp)
        return 1.0 / float(xp.sum(xp.square(w)))

    def with_updates(self, **kwargs: Any) -> "ParticleState":
        """Return a copy of the state with selective updates applied."""
        updated = self.copy()
        if "particles" in kwargs and "evaluation" not in kwargs:
            updated.evaluation = None
        for key, value in kwargs.items():
            setattr(updated, key, value)
        return updated

    def replace_particles(
        self,
        particles: Array,
        *,
        evaluation: TargetEvaluation | None = None,
        **updates: Any,
    ) -> "ParticleState":
        """Replace particles and explicitly provide their evaluation."""
        return self.with_updates(
            particles=particles,
            evaluation=evaluation,
            **updates,
        )

    def resample(self, indices: Array) -> "ParticleState":
        """Return a resampled state using the provided ancestor indices."""
        xp = namespace(self.particles, indices)
        particles = asarray(self.particles, xp=xp)
        new_particles = particles[indices]
        n = indices.shape[0]
        log_weight_template = asarray(self.log_weights, xp=xp)
        log_weights = xp.full(
            (n,),
            -math.log(n),
            dtype=log_weight_template.dtype,
            device=get_device(log_weight_template),
        )
        return ParticleState(
            particles=new_particles,
            log_weights=log_weights,
            log_evidence=self.log_evidence,
            initial_ancestor_indices=asarray(
                self.initial_ancestor_indices,
                xp=xp,
            )[indices],
            evaluation=(
                None
                if self.evaluation is None
                else self.evaluation.take(indices)
            ),
            metadata=self.metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a detached dictionary representation of this state."""
        copied = self.copy()
        return {
            "particles": copied.particles,
            "log_weights": copied.log_weights,
            "log_evidence": copied.log_evidence,
            "initial_ancestor_indices": copied.initial_ancestor_indices,
            "evaluation": (
                None
                if copied.evaluation is None
                else copied.evaluation.to_dict()
            ),
            "metadata": copied.metadata,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ParticleState":
        """Reconstruct a state from :meth:`to_dict` output."""
        if not isinstance(payload, Mapping):
            raise TypeError("ParticleState payload must be a mapping.")
        missing = {
            key
            for key in (
                "particles",
                "log_weights",
                "log_evidence",
                "initial_ancestor_indices",
                "metadata",
            )
            if key not in payload
        }
        if missing:
            raise ValueError(
                "ParticleState payload is missing required keys: "
                + ", ".join(sorted(missing))
            )
        evaluation_payload = payload.get("evaluation")
        if evaluation_payload is not None:
            from .target import TargetEvaluation

            evaluation = TargetEvaluation.from_dict(evaluation_payload)
        else:
            evaluation = None
        return cls(
            particles=payload["particles"],
            log_weights=payload["log_weights"],
            log_evidence=float(payload["log_evidence"]),
            initial_ancestor_indices=payload["initial_ancestor_indices"],
            evaluation=evaluation,
            metadata=array_safe_copy(payload["metadata"]),
        )


@pytree_dataclass(
    data_fields=(
        "particles",
        "log_weights",
        "initial_ancestor_indices",
        "log_evidence",
        "evaluation",
        "metadata",
        "temperatures",
        "log_evidences",
    ),
    meta_fields=("batch_size",),
)
@dataclass(init=False, kw_only=True)
class PersistentParticleState(ParticleState):
    """Equal-sized particle batches retained from historical targets.

    ``log_evidences`` contains evidence estimates for targets which generated
    archived batches.  The scalar ``log_evidence`` follows the ordinary
    :class:`ParticleState` interface and may instead describe a new target toward
    which the archive has just been weighted.
    """

    temperatures: Array
    log_evidences: Array
    batch_size: int

    def __init__(
        self,
        particles: Array,
        temperatures: ArrayLike,
        log_evidences: ArrayLike,
        batch_size: int,
        *,
        log_weights: Array | None = None,
        log_evidence: Any | None = None,
        initial_ancestor_indices: Array | None = None,
        evaluation: TargetEvaluation | None = None,
        metadata: MutableMapping[str, Any] | None = None,
    ) -> None:
        xp = namespace(particles)
        temperatures = asarray(temperatures, xp=xp, like=particles)
        log_evidences = asarray(log_evidences, xp=xp, like=particles)

        if particles.shape[0] == 0:
            raise ValueError("Particles array cannot be empty.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if temperatures.ndim != 1:
            raise ValueError("temperatures must be one-dimensional.")
        if log_evidences.ndim != 1:
            raise ValueError("log_evidences must be one-dimensional.")
        if temperatures.shape[0] != log_evidences.shape[0]:
            raise ValueError(
                "temperatures and log_evidences must have the same length."
            )

        n_particles = particles.shape[0]
        if n_particles % batch_size != 0:
            raise ValueError("Particle count must be divisible by batch_size.")
        n_batches = n_particles // batch_size
        if temperatures.shape[0] != n_batches:
            raise ValueError(
                "Temperature and evidence histories must contain one entry "
                "per particle batch."
            )

        if log_weights is None:
            log_weights = xp.full(
                (n_particles,),
                -math.log(n_particles),
                device=get_device(particles),
            )
        else:
            log_weights = asarray(log_weights, xp=xp, like=particles)
            if log_weights.shape != (n_particles,):
                raise ValueError(
                    "Log weights must match the persistent particle population."
                )
        if log_evidence is None:
            log_evidence = log_evidences[-1]
        super().__init__(
            particles=particles,
            log_weights=log_weights,
            log_evidence=log_evidence,
            initial_ancestor_indices=initial_ancestor_indices,
            evaluation=evaluation,
            metadata=metadata,
        )
        self.temperatures = temperatures
        self.log_evidences = log_evidences
        self.batch_size = batch_size

    def __len__(self) -> int:
        """Return the total number of retained particles."""
        return self.particles.shape[0]

    @property
    def n_particles(self) -> int:
        """Return the total number of retained particles."""
        return len(self)

    @property
    def n_batches(self) -> int:
        """Return the number of historical particle batches."""
        return self.temperatures.shape[0]

    @classmethod
    def from_state(
        cls,
        state: ParticleState,
        *,
        temperature: ArrayLike,
        log_evidence: ArrayLike,
    ) -> "PersistentParticleState":
        """Create a one-batch history from an active particle state."""
        xp = namespace(state.particles)
        temperature_history = cls._history_entry(
            temperature,
            name="temperature",
            xp=xp,
            like=state.particles,
        )
        evidence_history = cls._history_entry(
            log_evidence,
            name="log_evidence",
            xp=xp,
            like=state.particles,
        )
        return cls(
            particles=state.particles,
            temperatures=temperature_history,
            log_evidences=evidence_history,
            batch_size=state.particles.shape[0],
            initial_ancestor_indices=state.initial_ancestor_indices,
            evaluation=state.evaluation,
            metadata=array_safe_copy(state.metadata),
        )

    @staticmethod
    def _history_entry(
        value: ArrayLike,
        *,
        name: str,
        xp: Any,
        like: Array,
    ) -> Array:
        entry = asarray(value, xp=xp, like=like)
        if entry.ndim != 0:
            raise ValueError(f"{name} must be scalar.")
        return xp.reshape(entry, (1,))

    def copy(self) -> "PersistentParticleState":
        """Return an independent copy preserving the persistent structure."""
        xp = namespace(
            self.particles,
            self.log_weights,
            self.initial_ancestor_indices,
            self.temperatures,
            self.log_evidences,
        )
        copied = type(self)(
            particles=array_copy(self.particles, xp=xp),
            temperatures=array_copy(self.temperatures, xp=xp),
            log_evidences=array_copy(self.log_evidences, xp=xp),
            batch_size=self.batch_size,
            log_weights=array_copy(self.log_weights, xp=xp),
            log_evidence=self.log_evidence,
            initial_ancestor_indices=array_copy(
                self.initial_ancestor_indices,
                xp=xp,
            ),
            evaluation=(
                None if self.evaluation is None else self.evaluation.copy()
            ),
            metadata=array_safe_copy(self.metadata),
        )
        return copied

    def with_updates(self, **kwargs: Any) -> "PersistentParticleState":
        """Return a copy with non-structural fields updated.

        Historical particles and their batch metadata can only change through
        :meth:`append`. Persistent weighting steps may update ``log_weights``;
        callers may also replace compatible evaluations or metadata.
        """
        allowed = {"log_weights", "log_evidence", "evaluation", "metadata"}
        protected = sorted(set(kwargs).difference(allowed))
        if protected:
            raise ValueError(
                "Persistent structure cannot be changed with with_updates: "
                + ", ".join(protected)
            )
        if "log_weights" in kwargs:
            xp = namespace(self.particles)
            log_weights = asarray(
                kwargs["log_weights"], xp=xp, like=self.log_weights
            )
            if log_weights.shape != (self.n_particles,):
                raise ValueError(
                    "Log weights must match the persistent particle population."
                )
            kwargs["log_weights"] = log_weights
        if "evaluation" in kwargs:
            evaluation = kwargs["evaluation"]
            if (
                evaluation is not None
                and evaluation.n_particles != self.n_particles
            ):
                raise ValueError(
                    "Target evaluation must match the persistent particle "
                    "population."
                )
        updated = self.copy()
        for key, value in kwargs.items():
            setattr(updated, key, value)
        return updated

    def append(
        self,
        state: ParticleState,
        *,
        temperature: ArrayLike,
        log_evidence: ArrayLike,
    ) -> "PersistentParticleState":
        """Return a new history with one equal-sized batch appended.

        Ancestor indices are concatenated without offsets because every batch
        retains its genealogy relative to the same initial active population.
        """
        if state.particles.shape[0] != self.batch_size:
            raise ValueError(
                "Appended particle batch must match batch_size "
                f"({self.batch_size})."
            )
        xp = namespace(self.particles)
        new_particles = asarray(state.particles, xp=xp, like=self.particles)
        particles = xp.concat((self.particles, new_particles), axis=0)
        new_ancestors = asarray(
            state.initial_ancestor_indices,
            xp=xp,
            like=self.initial_ancestor_indices,
        )
        initial_ancestor_indices = xp.concat(
            (self.initial_ancestor_indices, new_ancestors),
            axis=0,
        )
        temperature_entry = self._history_entry(
            temperature,
            name="temperature",
            xp=xp,
            like=self.temperatures,
        )
        evidence_entry = self._history_entry(
            log_evidence,
            name="log_evidence",
            xp=xp,
            like=self.log_evidences,
        )
        temperatures = xp.concat(
            (self.temperatures, temperature_entry), axis=0
        )
        log_evidences = xp.concat((self.log_evidences, evidence_entry), axis=0)

        if self.evaluation is None and state.evaluation is None:
            evaluation = None
        elif self.evaluation is None or state.evaluation is None:
            raise ValueError(
                "Cannot append states with inconsistent target evaluations."
            )
        else:
            evaluation = type(self.evaluation).concatenate(
                (self.evaluation, state.evaluation)
            )

        return type(self)(
            particles=particles,
            temperatures=temperatures,
            log_evidences=log_evidences,
            batch_size=self.batch_size,
            log_evidence=log_evidence,
            initial_ancestor_indices=initial_ancestor_indices,
            evaluation=evaluation,
            metadata=array_safe_copy(state.metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a detached dictionary including persistent history."""
        payload = super().to_dict()
        copied = self.copy()
        payload.update(
            temperatures=copied.temperatures,
            log_evidences=copied.log_evidences,
            batch_size=copied.batch_size,
        )
        return payload

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "PersistentParticleState":
        """Reconstruct a persistent state from :meth:`to_dict` output."""
        missing = {
            key
            for key in ("temperatures", "log_evidences", "batch_size")
            if key not in payload
        }
        if missing:
            raise ValueError(
                "PersistentParticleState payload is missing required keys: "
                + ", ".join(sorted(missing))
            )
        base = ParticleState.from_dict(payload)
        return cls(
            particles=base.particles,
            temperatures=payload["temperatures"],
            log_evidences=payload["log_evidences"],
            batch_size=int(payload["batch_size"]),
            log_weights=base.log_weights,
            log_evidence=base.log_evidence,
            initial_ancestor_indices=base.initial_ancestor_indices,
            evaluation=base.evaluation,
            metadata=base.metadata,
        )
