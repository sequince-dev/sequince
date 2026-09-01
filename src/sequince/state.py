from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Real
from typing import TYPE_CHECKING, Any, MutableMapping, Self, cast

from array_api_compat import device as get_device

from ._typing import Array, ArrayLike
from .array_api import asarray, logsumexp, namespace
from .array_api import copy as array_copy
from .exceptions import DegenerateWeightsError
from .utils.copying import array_safe_copy
from .utils.pytrees import pytree_dataclass

if TYPE_CHECKING:
    from .target import TargetEvaluation

_CHECKPOINT_STATE_KIND_KEY = "_sequince_state_kind"


def normalize_log_weights(log_weights: Any, *, xp: Any | None = None):
    """Return normalized log weights and their array scalar normalizer."""
    if xp is None:
        values = asarray(log_weights)
        xp = namespace(values)
    else:
        values = asarray(log_weights, xp=xp)
    log_norm = logsumexp(values, xp=xp)
    safe_log_norm = xp.where(xp.isfinite(log_norm), log_norm, 0.0)
    return values - safe_log_norm, log_norm


def _validate_log_normalizer(log_normalizer: Any, *, xp: Any) -> None:
    """Raise for a non-finite normalizer at an eager host boundary."""
    if not bool(xp.isfinite(log_normalizer)):
        raise DegenerateWeightsError(log_normalizer)


@pytree_dataclass(
    data_fields=(
        "particles",
        "log_weights",
        "initial_ancestor_indices",
        "log_normalizing_constant",
        "evaluation",
        "metadata",
    )
)
@dataclass(init=False)
class ParticleState:
    """Container for particle locations, log weights, and auxiliary information.

    ``log_normalizing_constant`` is the accumulated estimate of ``log Z`` for
    the SMC target sequence. For Bayesian posterior SMC this is the log
    evidence; for particle filtering it is the log marginal likelihood of the
    observation sequence.
    """

    particles: Array
    log_weights: Array
    initial_ancestor_indices: Array
    log_normalizing_constant: float = 0.0
    evaluation: TargetEvaluation | None = None
    metadata: MutableMapping[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        particles: Array,
        log_weights: Array | Real | None = None,
        log_normalizing_constant: float = 0.0,
        initial_ancestor_indices: Array | None = None,
        evaluation: TargetEvaluation | None = None,
        metadata: MutableMapping[str, Any] | None = None,
    ) -> None:
        if particles.shape[0] == 0:
            raise ValueError("Particles array cannot be empty.")

        self.particles = particles
        self.log_normalizing_constant = log_normalizing_constant
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

    @property
    def log_evidence(self) -> float:
        """Alias for ``log_normalizing_constant``."""
        return self.log_normalizing_constant

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
            log_normalizing_constant=self.log_normalizing_constant,
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
        """Return a new state and its incremental log normalizing constant."""
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
            log_normalizing_constant=self.log_normalizing_constant,
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
            "log_normalizing_constant": copied.log_normalizing_constant,
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
                "log_normalizing_constant",
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
            log_normalizing_constant=float(
                payload["log_normalizing_constant"]
            ),
            initial_ancestor_indices=payload["initial_ancestor_indices"],
            evaluation=evaluation,
            metadata=array_safe_copy(payload["metadata"]),
        )

    def _encode_hdf5_fields(self) -> dict[str, Any]:
        """Encode fields shared by built-in particle-state schemas."""
        from .target import TargetEvaluation
        from .utils.hdf5 import nested

        evaluation = self.evaluation
        return {
            "particles": self.particles,
            "log_weights": self.log_weights,
            "log_normalizing_constant": self.log_normalizing_constant,
            "initial_ancestor_indices": self.initial_ancestor_indices,
            "evaluation": (
                None
                if evaluation is None
                else nested(TargetEvaluation.encode_for_hdf5(evaluation))
            ),
            "metadata": dict(self.metadata),
        }

    def _encode_for_checkpoint(self) -> dict[str, Any]:
        """Encode an ordinary state for the resumable checkpoint format."""
        if type(self) is not ParticleState:
            raise TypeError(
                "Unsupported ParticleState subtype in checkpoint: "
                f"{type(self).__module__}.{type(self).__qualname__}."
            )
        return {
            _CHECKPOINT_STATE_KIND_KEY: "particle_state",
            **self.to_dict(),
        }

    @classmethod
    def _decode_from_checkpoint(cls, payload: Mapping[str, Any]) -> Self:
        """Decode an ordinary state or dispatch a persistent checkpoint state."""
        values = dict(payload)
        kind = values.pop(_CHECKPOINT_STATE_KIND_KEY, None)
        if kind == "persistent_particle_state":
            return cast(
                Self, PersistentParticleState._decode_from_checkpoint(payload)
            )
        if kind != "particle_state":
            raise ValueError(f"Unsupported checkpoint state kind {kind!r}.")
        return ParticleState.from_dict(values)  # type: ignore[return-value]

    def encode_for_hdf5(self) -> dict[str, Any]:
        """Return a versioned, portable particle-state payload."""
        from .utils.hdf5 import schema_header

        if type(self) is not ParticleState:
            raise TypeError(
                "Unsupported ParticleState subtype for HDF5 saving: "
                f"{type(self).__module__}.{type(self).__qualname__}."
            )
        return {
            **schema_header("particle_state"),
            **self._encode_hdf5_fields(),
        }

    @classmethod
    def _decode_from_hdf5(cls, payload: Mapping[str, Any]) -> Self:
        """Decode and dispatch a built-in state payload."""
        from .target import TargetEvaluation
        from .utils.hdf5 import validate_schema

        if cls is not ParticleState:
            raise TypeError(
                "Unsupported ParticleState subtype for HDF5 loading: "
                f"{cls.__module__}.{cls.__qualname__}."
            )
        required = {
            "particles",
            "log_weights",
            "log_normalizing_constant",
            "initial_ancestor_indices",
            "evaluation",
            "metadata",
        }
        kind = payload.get("_sequince_kind")
        if kind == "persistent_particle_state":
            if cls is not ParticleState:
                raise ValueError(
                    f"Unexpected Sequince HDF5 kind {kind!r}; expected "
                    "particle_state."
                )
            return cast(
                Self, PersistentParticleState._decode_from_hdf5(payload)
            )
        validate_schema(payload, kinds={"particle_state"}, required=required)
        values = {key: payload[key] for key in required}
        evaluation_payload = values["evaluation"]
        if evaluation_payload is not None:
            TargetEvaluation._decode_from_hdf5(evaluation_payload)
        return ParticleState.from_dict(values)  # type: ignore[return-value]

    def save(self, h5_file_or_group: Any, path: str = "state") -> None:
        """Save this state below ``path`` in an open HDF5 handle."""
        from .utils.hdf5 import recursively_save_to_h5_file

        recursively_save_to_h5_file(
            h5_file_or_group, path, self.encode_for_hdf5()
        )

    @classmethod
    def load(cls, h5_file_or_group: Any, path: str = "state") -> Self:
        """Load a built-in state from ``path`` in an open HDF5 handle."""
        from .utils.hdf5 import load_from_h5_file

        return cls._decode_from_hdf5(load_from_h5_file(h5_file_or_group, path))


@pytree_dataclass(
    data_fields=(
        "particles",
        "log_weights",
        "initial_ancestor_indices",
        "log_normalizing_constant",
        "evaluation",
        "metadata",
        "betas",
        "log_normalizing_constants",
    ),
    meta_fields=("batch_size",),
)
@dataclass(init=False, kw_only=True)
class PersistentParticleState(ParticleState):
    """Equal-sized particle batches retained from historical targets.

    ``log_normalizing_constants`` contains normalizing-constant estimates for
    targets which generated archived batches. The scalar
    ``log_normalizing_constant`` follows the ordinary :class:`ParticleState`
    interface and may instead describe a new target toward which the archive
    has just been weighted.
    """

    betas: Array
    log_normalizing_constants: Array
    batch_size: int

    def __init__(
        self,
        particles: Array,
        betas: ArrayLike,
        log_normalizing_constants: ArrayLike,
        batch_size: int,
        *,
        log_weights: Array | None = None,
        log_normalizing_constant: Any | None = None,
        initial_ancestor_indices: Array | None = None,
        evaluation: TargetEvaluation | None = None,
        metadata: MutableMapping[str, Any] | None = None,
    ) -> None:
        xp = namespace(particles)
        betas = asarray(betas, xp=xp, like=particles)
        normalizing_constants = asarray(
            log_normalizing_constants, xp=xp, like=particles
        )

        if particles.shape[0] == 0:
            raise ValueError("Particles array cannot be empty.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if betas.ndim != 1:
            raise ValueError("betas must be one-dimensional.")
        if normalizing_constants.ndim != 1:
            raise ValueError(
                "log_normalizing_constants must be one-dimensional."
            )
        if betas.shape[0] != normalizing_constants.shape[0]:
            raise ValueError(
                "betas and log_normalizing_constants must have the "
                "same length."
            )

        n_particles = particles.shape[0]
        if n_particles % batch_size != 0:
            raise ValueError("Particle count must be divisible by batch_size.")
        n_batches = n_particles // batch_size
        if betas.shape[0] != n_batches:
            raise ValueError(
                "beta and normalizing-constant histories must contain "
                "one entry per particle batch."
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
        if log_normalizing_constant is None:
            log_normalizing_constant = normalizing_constants[-1]
        super().__init__(
            particles=particles,
            log_weights=log_weights,
            log_normalizing_constant=log_normalizing_constant,
            initial_ancestor_indices=initial_ancestor_indices,
            evaluation=evaluation,
            metadata=metadata,
        )
        self.betas = betas
        self.log_normalizing_constants = normalizing_constants
        self.batch_size = batch_size

    @property
    def log_evidence(self) -> float:
        """Alias for ``log_normalizing_constant``."""
        return self.log_normalizing_constant

    @property
    def log_evidences(self) -> Array:
        """Alias for ``log_normalizing_constants``."""
        return self.log_normalizing_constants

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
        return self.betas.shape[0]

    @classmethod
    def from_state(
        cls,
        state: ParticleState,
        *,
        beta: ArrayLike,
        log_normalizing_constant: ArrayLike,
    ) -> "PersistentParticleState":
        """Create a one-batch history from an active particle state."""
        xp = namespace(state.particles)
        beta_history = cls._history_entry(
            beta,
            name="beta",
            xp=xp,
            like=state.particles,
        )
        normalizing_constant_history = cls._history_entry(
            log_normalizing_constant,
            name="log_normalizing_constant",
            xp=xp,
            like=state.particles,
        )
        return cls(
            particles=state.particles,
            betas=beta_history,
            log_normalizing_constants=normalizing_constant_history,
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
            self.betas,
            self.log_normalizing_constants,
        )
        copied = type(self)(
            particles=array_copy(self.particles, xp=xp),
            betas=array_copy(self.betas, xp=xp),
            log_normalizing_constants=array_copy(
                self.log_normalizing_constants, xp=xp
            ),
            batch_size=self.batch_size,
            log_weights=array_copy(self.log_weights, xp=xp),
            log_normalizing_constant=self.log_normalizing_constant,
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
        allowed = {
            "log_weights",
            "log_normalizing_constant",
            "evaluation",
            "metadata",
        }
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
        beta: ArrayLike,
        log_normalizing_constant: ArrayLike,
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
        beta_entry = self._history_entry(
            beta,
            name="beta",
            xp=xp,
            like=self.betas,
        )
        normalizing_constant_entry = self._history_entry(
            log_normalizing_constant,
            name="log_normalizing_constant",
            xp=xp,
            like=self.log_normalizing_constants,
        )
        betas = xp.concat((self.betas, beta_entry), axis=0)
        log_normalizing_constants = xp.concat(
            (self.log_normalizing_constants, normalizing_constant_entry),
            axis=0,
        )

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
            betas=betas,
            log_normalizing_constants=log_normalizing_constants,
            batch_size=self.batch_size,
            log_normalizing_constant=log_normalizing_constant,
            initial_ancestor_indices=initial_ancestor_indices,
            evaluation=evaluation,
            metadata=array_safe_copy(state.metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a detached dictionary including persistent history."""
        payload = super().to_dict()
        copied = self.copy()
        payload.update(
            betas=copied.betas,
            log_normalizing_constants=copied.log_normalizing_constants,
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
            for key in (
                "betas",
                "log_normalizing_constants",
                "batch_size",
            )
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
            betas=payload["betas"],
            log_normalizing_constants=payload["log_normalizing_constants"],
            batch_size=int(payload["batch_size"]),
            log_weights=base.log_weights,
            log_normalizing_constant=base.log_normalizing_constant,
            initial_ancestor_indices=base.initial_ancestor_indices,
            evaluation=base.evaluation,
            metadata=base.metadata,
        )

    def encode_for_hdf5(self) -> dict[str, Any]:
        """Return a versioned payload including persistent history."""
        from .utils.hdf5 import schema_header

        if type(self) is not PersistentParticleState:
            raise TypeError(
                "Unsupported PersistentParticleState subtype for HDF5 saving: "
                f"{type(self).__module__}.{type(self).__qualname__}."
            )
        return {
            **schema_header("persistent_particle_state"),
            **self._encode_hdf5_fields(),
            "betas": self.betas,
            "log_normalizing_constants": self.log_normalizing_constants,
            "batch_size": self.batch_size,
        }

    def _encode_for_checkpoint(self) -> dict[str, Any]:
        """Encode persistent history for the resumable checkpoint format."""
        if type(self) is not PersistentParticleState:
            raise TypeError(
                "Unsupported PersistentParticleState subtype in checkpoint: "
                f"{type(self).__module__}.{type(self).__qualname__}."
            )
        return {
            _CHECKPOINT_STATE_KIND_KEY: "persistent_particle_state",
            **self.to_dict(),
        }

    @classmethod
    def _decode_from_checkpoint(cls, payload: Mapping[str, Any]) -> Self:
        """Decode persistent history from a checkpoint payload."""
        values = dict(payload)
        kind = values.pop(_CHECKPOINT_STATE_KIND_KEY, None)
        if kind != "persistent_particle_state":
            raise ValueError(f"Unsupported checkpoint state kind {kind!r}.")
        return cast(Self, cls.from_dict(values))

    @classmethod
    def _decode_from_hdf5(cls, payload: Mapping[str, Any]) -> Self:
        """Decode a persistent-state payload without flattening its subtype."""
        from .target import TargetEvaluation
        from .utils.hdf5 import validate_schema

        required = {
            "particles",
            "log_weights",
            "log_normalizing_constant",
            "initial_ancestor_indices",
            "evaluation",
            "metadata",
            "betas",
            "log_normalizing_constants",
            "batch_size",
        }
        validate_schema(
            payload,
            kinds={"persistent_particle_state"},
            required=required,
        )
        values = {key: payload[key] for key in required}
        evaluation_payload = values["evaluation"]
        if evaluation_payload is not None:
            TargetEvaluation._decode_from_hdf5(evaluation_payload)
        return cast(Self, cls.from_dict(values))
