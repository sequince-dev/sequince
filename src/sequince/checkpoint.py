from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote

import numpy as np
from orng import RandomGenerator

from .array_api import namespace
from .result import SMCResult
from .state import ParticleState
from .steps.context import StepContext
from .utils.copying import array_safe_copy
from .utils.random import resolve_rng

CHECKPOINT_VERSION = 1
_HDF_KIND = "_sequince_kind"
_HDF_PYTHON_TYPE = "_sequince_python_type"


def _rng_state_dict(rng: Any) -> dict[str, Any]:
    """Serialize either a mutable generator or backend-native random state."""
    try:
        generator = resolve_rng(rng)
    except TypeError:
        generator = RandomGenerator.from_xp(
            namespace(rng),
            generator=rng,
        )
    return generator.state_dict()


def _pickle_bytes(value: Any, protocol: int) -> np.ndarray:
    """Return a pickled value as an HDF5-compatible byte array."""
    return np.frombuffer(pickle.dumps(value, protocol=protocol), dtype="u1")


def _write_pickled_hdf_value(
    parent: Any, name: str, value: Any, protocol: int
) -> Any:
    """Write one unsupported leaf as a tagged pickle dataset."""
    if name in parent:
        del parent[name]
    node = parent.create_dataset(name, data=_pickle_bytes(value, protocol))
    node.attrs[_HDF_KIND] = "pickle"
    return node


def _write_hdf_value(parent: Any, name: str, value: Any, protocol: int) -> Any:
    """Write mappings, sequences, and native leaves to HDF5 recursively."""
    if name in parent:
        del parent[name]

    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            return _write_pickled_hdf_value(parent, name, value, protocol)
        node = parent.create_group(name)
        node.attrs[_HDF_KIND] = "mapping"
        for key, item in value.items():
            child_name = quote(key, safe="") or "%"
            _write_hdf_value(node, child_name, item, protocol)
        return node

    if isinstance(value, (list, tuple)):
        node = parent.create_group(name)
        node.attrs[_HDF_KIND] = "list" if isinstance(value, list) else "tuple"
        for index, item in enumerate(value):
            _write_hdf_value(node, f"{index:08d}", item, protocol)
        return node

    if value is None:
        node = parent.create_group(name)
        node.attrs[_HDF_KIND] = "none"
        return node

    if isinstance(value, str):
        from .utils.hdf5 import _create_string_dataset

        node = _create_string_dataset(parent, name, value)
        node.attrs[_HDF_KIND] = "string"
        return node

    if isinstance(value, bytes):
        node = parent.create_dataset(
            name, data=np.frombuffer(value, dtype="u1")
        )
        node.attrs[_HDF_KIND] = "bytes"
        return node

    if isinstance(value, (np.ndarray, np.generic)):
        if value.dtype.hasobject:
            return _write_pickled_hdf_value(parent, name, value, protocol)
        try:
            node = parent.create_dataset(name, data=value)
        except (TypeError, ValueError):
            return _write_pickled_hdf_value(parent, name, value, protocol)
        node.attrs[_HDF_KIND] = "numpy"
        return node

    if isinstance(value, (bool, int, float, complex)):
        try:
            node = parent.create_dataset(name, data=value)
        except (OverflowError, TypeError, ValueError):
            return _write_pickled_hdf_value(parent, name, value, protocol)
        node.attrs[_HDF_KIND] = "scalar"
        node.attrs[_HDF_PYTHON_TYPE] = type(value).__name__
        return node

    return _write_pickled_hdf_value(parent, name, value, protocol)


def _read_hdf_value(node: Any) -> Any:
    """Reconstruct a value written by :func:`_write_hdf_value`."""
    from .utils.hdf5 import _decode_hdf5_text, _read_string_dataset

    if _HDF_KIND not in node.attrs:
        raise ValueError(f"HDF5 node {node.name!r} has no Sequince type tag.")
    kind = _decode_hdf5_text(node.attrs[_HDF_KIND])

    if kind == "mapping":
        return {
            "" if name == "%" else unquote(name): _read_hdf_value(node[name])
            for name in node
        }

    if kind in ("list", "tuple"):
        items = [_read_hdf_value(node[name]) for name in sorted(node.keys())]
        return items if kind == "list" else tuple(items)

    if kind == "none":
        return None
    if kind == "string":
        return _read_string_dataset(node)
    if kind == "bytes":
        return node[...].tobytes()
    if kind == "numpy":
        return node[()]
    if kind == "scalar":
        scalar_value = node[()]
        if isinstance(scalar_value, np.generic):
            scalar_value = scalar_value.item()
        scalar_type = _decode_hdf5_text(node.attrs[_HDF_PYTHON_TYPE])
        scalar_constructors = {
            "bool": bool,
            "int": int,
            "float": float,
            "complex": complex,
        }
        return scalar_constructors[scalar_type](scalar_value)
    if kind == "pickle":
        return pickle.loads(node[...].tobytes())
    raise ValueError(f"Unsupported Sequince HDF5 node kind {kind!r}.")


class CheckpointWriter(ABC):
    """Base class for persisting committed SMC checkpoints."""

    requires_detached_checkpoint = True
    """Whether the writer may retain the checkpoint after ``write`` returns."""

    def __init__(
        self,
        *,
        every: int = 1,
        save_rng: bool = True,
    ) -> None:
        if every <= 0:
            raise ValueError("every must be positive.")
        self.every = int(every)
        self.save_rng = save_rng

    def is_due(self, next_iteration: int) -> bool:
        """Return whether a checkpoint is due after this iteration."""
        return next_iteration % self.every == 0

    def write(self, checkpoint: "SMCCheckpoint") -> None:
        """Apply this writer's options and persist the checkpoint."""
        prepared = replace(
            checkpoint,
            rng_state=checkpoint.rng_state if self.save_rng else None,
        )
        self._write(prepared)

    @abstractmethod
    def _write(self, checkpoint: "SMCCheckpoint") -> None:
        """Persist a checkpoint prepared for this writer."""


@dataclass(frozen=True)
class StepState:
    """Checkpoint state associated with one configured SMC step."""

    index: int
    module: str
    qualname: str
    state: dict[str, Any]

    @classmethod
    def from_step(cls, index: int, step: Any) -> "StepState":
        state_fn = getattr(step, "state_dict", None)
        state = {} if state_fn is None else state_fn()
        if not isinstance(state, Mapping):
            raise TypeError(
                f"Step {index} ({type(step).__name__}) state_dict() must "
                "return a mapping."
            )
        step_type = type(step)
        return cls(
            index=index,
            module=step_type.__module__,
            qualname=step_type.__qualname__,
            state=array_safe_copy(dict(state)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "module": self.module,
            "qualname": self.qualname,
            "state": array_safe_copy(self.state),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StepState":
        missing = {
            key
            for key in ("index", "module", "qualname", "state")
            if key not in payload
        }
        if missing:
            raise ValueError(
                "Step state is missing required keys: "
                + ", ".join(sorted(missing))
            )
        state = payload["state"]
        if not isinstance(state, Mapping):
            raise TypeError("Step state 'state' must be a mapping.")
        module = payload["module"]
        qualname = payload["qualname"]
        if not isinstance(module, str) or not isinstance(qualname, str):
            raise TypeError("Step state module and qualname must be strings.")
        return cls(
            index=int(payload["index"]),
            module=module,
            qualname=qualname,
            state=array_safe_copy(dict(state)),
        )


def capture_step_states(steps: Sequence[Any]) -> tuple[StepState, ...]:
    """Capture mutable state from a configured step pipeline."""
    return tuple(
        StepState.from_step(index, step) for index, step in enumerate(steps)
    )


def restore_steps(steps: Sequence[Any], states: Sequence[StepState]) -> None:
    """Restore step state after validating pipeline identity and order."""
    if len(steps) != len(states):
        raise ValueError(
            "Checkpoint step pipeline length does not match the configured "
            f"pipeline ({len(states)} != {len(steps)})."
        )
    for index, (step, saved) in enumerate(zip(steps, states, strict=True)):
        step_type = type(step)
        identity = (step_type.__module__, step_type.__qualname__)
        saved_identity = (saved.module, saved.qualname)
        if saved.index != index or saved_identity != identity:
            raise ValueError(
                f"Checkpoint step {index} is "
                f"{saved.module}.{saved.qualname}, but the configured step is "
                f"{step_type.__module__}.{step_type.__qualname__}."
            )
        load_fn = getattr(step, "load_state_dict", None)
        if load_fn is None and saved.state:
            raise ValueError(
                f"Step {index} ({step_type.__name__}) has checkpoint "
                "state but does not implement load_state_dict()."
            )
    for step, saved in zip(steps, states, strict=True):
        load_fn = getattr(step, "load_state_dict", None)
        if load_fn is not None:
            load_fn(array_safe_copy(saved.state))


@dataclass(frozen=True)
class SMCCheckpoint:
    """Detached end-of-iteration state for resuming an SMC run."""

    states: tuple[ParticleState, ...]
    next_iteration: int
    next_observation_index: int
    metadata: dict[str, Any]
    outputs: tuple[tuple[Any, ...], ...]
    step_states: tuple[StepState, ...]
    rng_state: dict[str, Any] | None = None
    observation: Any = None
    initial_state: ParticleState | None = None
    version: int = CHECKPOINT_VERSION

    def __post_init__(self) -> None:
        if not self.states:
            raise ValueError("SMCCheckpoint requires at least one state.")

    @property
    def state(self) -> ParticleState:
        """Return the current state represented by this checkpoint."""
        return self.states[-1]

    @classmethod
    def from_runtime(
        cls,
        state: ParticleState,
        context: StepContext,
        result: SMCResult,
        steps: Sequence[Any],
        rng: Any | None = None,
        *,
        detach: bool = True,
    ) -> "SMCCheckpoint":
        """Capture a checkpoint after a completed SMC iteration."""
        states = tuple(result.states)
        if not states:
            raise ValueError(
                "Checkpoint result must contain at least one state."
            )
        if states[-1] is not state:
            raise ValueError(
                "Checkpoint state must be the final state in the result."
            )
        states = tuple(item.copy() if detach else item for item in states)
        metadata = {
            key: value
            for key, value in context.metadata.items()
            if key != "particle_transform"
        }
        return cls(
            states=states,
            next_iteration=context.t + 1,
            next_observation_index=context.next_observation_index,
            metadata=(array_safe_copy(metadata) if detach else dict(metadata)),
            outputs=(
                array_safe_copy(tuple(result.outputs))
                if detach
                else tuple(result.outputs)
            ),
            step_states=capture_step_states(steps),
            rng_state=None if rng is None else _rng_state_dict(rng),
            observation=(
                array_safe_copy(context.observation)
                if detach
                else context.observation
            ),
            initial_state=(
                None
                if result.initial_state is None
                else (
                    result.initial_state.copy()
                    if detach
                    else result.initial_state
                )
            ),
        )

    def _runtime_dict(self) -> dict[str, Any]:
        """Return detached runtime fields without materializing states."""
        return {
            "version": self.version,
            "next_iteration": self.next_iteration,
            "next_observation_index": self.next_observation_index,
            "metadata": array_safe_copy(self.metadata),
            "outputs": array_safe_copy(self.outputs),
            "step_states": [item.to_dict() for item in self.step_states],
            "rng_state": array_safe_copy(self.rng_state),
            "observation": array_safe_copy(self.observation),
            "initial_state": (
                None
                if self.initial_state is None
                else self.initial_state._encode_for_checkpoint()
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a detached plain-Python checkpoint representation."""
        payload = self._runtime_dict()
        payload["states"] = [
            item._encode_for_checkpoint() for item in self.states
        ]
        return payload

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        copy_values: bool = True,
    ) -> "SMCCheckpoint":
        """Reconstruct a checkpoint from :meth:`to_dict` output."""
        if not isinstance(payload, Mapping):
            raise TypeError("SMCCheckpoint payload must be a mapping.")
        required = {
            "version",
            "states",
            "next_iteration",
            "next_observation_index",
            "metadata",
            "outputs",
            "step_states",
        }
        missing = required.difference(payload)
        if missing:
            raise ValueError(
                "SMCCheckpoint payload is missing required keys: "
                + ", ".join(sorted(missing))
            )
        version = payload["version"]
        if version != CHECKPOINT_VERSION:
            raise ValueError(
                f"Unsupported SMC checkpoint version {version!r}; expected "
                f"{CHECKPOINT_VERSION}."
            )
        rng_state = payload.get("rng_state")
        if rng_state is not None and not isinstance(rng_state, Mapping):
            raise TypeError(
                "SMCCheckpoint 'rng_state' must be a mapping or None."
            )
        if not isinstance(payload["metadata"], Mapping):
            raise TypeError("SMCCheckpoint 'metadata' must be a mapping.")
        if not isinstance(payload["outputs"], Sequence) or isinstance(
            payload["outputs"], (str, bytes)
        ):
            raise TypeError("SMCCheckpoint 'outputs' must be a sequence.")
        step_payload = payload["step_states"]
        if not isinstance(step_payload, Sequence) or isinstance(
            step_payload, (str, bytes)
        ):
            raise TypeError("SMCCheckpoint 'step_states' must be a sequence.")
        states_payload = payload["states"]
        if (
            not isinstance(states_payload, Sequence)
            or isinstance(states_payload, (str, bytes))
            or not states_payload
        ):
            raise TypeError(
                "SMCCheckpoint 'states' must be a non-empty sequence."
            )
        initial_state_payload = payload.get("initial_state")
        if initial_state_payload is not None and not isinstance(
            initial_state_payload, Mapping
        ):
            raise TypeError(
                "SMCCheckpoint 'initial_state' must be a mapping or None."
            )
        next_iteration = int(payload["next_iteration"])
        next_observation_index = int(payload["next_observation_index"])
        if next_iteration < 0 or next_observation_index < 0:
            raise ValueError(
                "Checkpoint iteration and observation indices must be "
                "non-negative."
            )

        def owned(value):
            return array_safe_copy(value) if copy_values else value

        return cls(
            states=tuple(
                ParticleState._decode_from_checkpoint(owned(item))
                for item in states_payload
            ),
            next_iteration=next_iteration,
            next_observation_index=next_observation_index,
            metadata=array_safe_copy(dict(payload["metadata"])),
            outputs=tuple(
                tuple(iteration) for iteration in payload["outputs"]
            ),
            step_states=tuple(
                StepState.from_dict(item) for item in step_payload
            ),
            rng_state=(
                None if rng_state is None else array_safe_copy(dict(rng_state))
            ),
            observation=array_safe_copy(payload.get("observation")),
            initial_state=(
                None
                if initial_state_payload is None
                else ParticleState._decode_from_checkpoint(
                    owned(initial_state_payload)
                )
            ),
            version=CHECKPOINT_VERSION,
        )

    def restore_rng(self) -> Any | None:
        """Return a live ORNG generator restored from this checkpoint."""
        if self.rng_state is None:
            return None
        return RandomGenerator.from_state_dict(self.rng_state)

    def result(
        self, shared_context: Mapping[str, Any] | None = None
    ) -> SMCResult:
        """Return the result states represented by this checkpoint."""
        context = StepContext(
            t=self.next_iteration - 1,
            next_observation_index=self.next_observation_index,
            observation=array_safe_copy(self.observation),
            shared=dict(shared_context or {}),
            metadata=array_safe_copy(self.metadata),
        )
        return SMCResult(
            states=list(self.states),
            context=context,
            initial_state=self.initial_state,
            outputs=array_safe_copy(self.outputs),
        )

    def save_pickle(
        self,
        file_path: str | Path,
        protocol: int | None = None,
    ) -> None:
        """Save this checkpoint as a versioned pickle payload."""
        protocol = (
            pickle.HIGHEST_PROTOCOL if protocol is None else int(protocol)
        )
        with Path(file_path).open("wb") as checkpoint_file:
            pickle.dump(self.to_dict(), checkpoint_file, protocol=protocol)

    @classmethod
    def load_pickle(cls, file_path: str | Path) -> "SMCCheckpoint":
        """Load a trusted checkpoint from a pickle file."""
        with Path(file_path).open("rb") as checkpoint_file:
            payload = pickle.load(checkpoint_file)
        return cls.from_dict(payload, copy_values=False)

    def save_hdf5(
        self,
        h5_file,
        path: str = "checkpoint",
        dsetname: str = "runtime",
        protocol: int | None = None,
        *,
        replace_states: bool = False,
    ) -> None:
        """Save runtime state and append result states in HDF5."""
        protocol = (
            pickle.HIGHEST_PROTOCOL if protocol is None else int(protocol)
        )
        group = h5_file.require_group(path)
        runtime_payload = self._runtime_dict()

        states_group = group.require_group("states")
        if replace_states:
            del group["states"]
            states_group = group.require_group("states")
        existing = len(states_group)
        if existing > len(self.states):
            del group["states"]
            states_group = group.require_group("states")
            existing = 0
        for index in range(existing, len(self.states)):
            _write_hdf_value(
                states_group,
                f"{index:08d}",
                self.states[index]._encode_for_checkpoint(),
                protocol,
            )

        temporary_name = f"{dsetname}.tmp"
        _write_hdf_value(group, temporary_name, runtime_payload, protocol)
        if dsetname in group:
            del group[dsetname]
        group.move(temporary_name, dsetname)

    @classmethod
    def load_hdf5(
        cls,
        h5_file,
        path: str = "checkpoint",
        dsetname: str = "runtime",
    ) -> "SMCCheckpoint":
        """Load a checkpoint from its recursive HDF5 representation."""
        group = h5_file[path]
        checkpoint_payload = _read_hdf_value(group[dsetname])
        if not isinstance(checkpoint_payload, dict):
            raise TypeError("HDF5 checkpoint runtime must be a mapping.")
        states_group = group["states"]
        checkpoint_payload["states"] = [
            _read_hdf_value(states_group[name])
            for name in sorted(states_group)
        ]
        return cls.from_dict(checkpoint_payload, copy_values=False)


class HDF5CheckpointWriter(CheckpointWriter):
    """Persist committed SMC checkpoints to HDF5 at a fixed cadence."""

    requires_detached_checkpoint = False

    def __init__(
        self,
        file_path: str | Path,
        *,
        every: int = 1,
        path: str = "checkpoint",
        dsetname: str = "runtime",
        save_rng: bool = True,
    ) -> None:
        super().__init__(
            every=every,
            save_rng=save_rng,
        )
        file_path = Path(file_path)
        if not file_path.name.lower().endswith((".h5", ".hdf5")):
            raise ValueError(
                "Checkpoint file must be an HDF5 file (.h5 or .hdf5)."
            )
        self.file_path = file_path
        self.path = path
        self.dsetname = dsetname
        self._last_iteration: int | None = None
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

    def _write(self, checkpoint: SMCCheckpoint) -> None:
        """Write a committed checkpoint to the configured HDF5 location."""
        import h5py

        replace_states = (
            self._last_iteration is None
            or checkpoint.next_iteration <= self._last_iteration
        )
        with h5py.File(self.file_path, "a") as h5_file:
            checkpoint.save_hdf5(
                h5_file,
                path=self.path,
                dsetname=self.dsetname,
                replace_states=replace_states,
            )
        self._last_iteration = checkpoint.next_iteration


class PickleCheckpointWriter(CheckpointWriter):
    """Atomically persist committed SMC checkpoints as pickle files."""

    requires_detached_checkpoint = False

    def __init__(
        self,
        file_path: str | Path,
        *,
        every: int = 1,
        save_rng: bool = True,
        protocol: int | None = None,
    ) -> None:
        super().__init__(
            every=every,
            save_rng=save_rng,
        )
        file_path = Path(file_path)
        if file_path.suffix.lower() not in (".pkl", ".pickle"):
            raise ValueError(
                "Checkpoint file must be a pickle file (.pkl or .pickle)."
            )
        self.file_path = file_path
        self.protocol = protocol
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

    def _write(self, checkpoint: SMCCheckpoint) -> None:
        """Write a committed checkpoint using an atomic file replacement."""
        temporary_path = self.file_path.with_name(self.file_path.name + ".tmp")
        checkpoint.save_pickle(temporary_path, protocol=self.protocol)
        temporary_path.replace(self.file_path)
