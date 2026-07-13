from __future__ import annotations

import copy
import pickle
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np

from .result import SMCResult
from .state import ParticleState
from .steps.context import StepContext
from .utils import resolve_rng, restore_rng, snapshot_rng

CHECKPOINT_VERSION = 1
_HDF_KIND = "_sequince_kind"
_HDF_KEY_KIND = "_sequince_key_kind"
_HDF_KEY = "_sequince_key"
_HDF_SCALAR_TYPE = "_sequince_scalar_type"


def _attr_text(value: Any) -> str:
    """Normalize an HDF5 text attribute to ``str``."""
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


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
    """Recursively write Python containers using native HDF5 structures."""
    if name in parent:
        del parent[name]

    if isinstance(value, Mapping):
        node = parent.create_group(name)
        node.attrs[_HDF_KIND] = "mapping"
        for index, (key, item) in enumerate(value.items()):
            child_name = (
                quote(key, safe="") or "%"
                if isinstance(key, str)
                else f"@{index:08d}"
            )
            child = _write_hdf_value(node, child_name, item, protocol)
            if isinstance(key, str):
                try:
                    child.attrs[_HDF_KEY] = key
                    child.attrs[_HDF_KEY_KIND] = "string"
                    continue
                except (TypeError, ValueError):
                    if _HDF_KEY in child.attrs:
                        del child.attrs[_HDF_KEY]
            child.attrs[_HDF_KEY] = np.void(
                pickle.dumps(key, protocol=protocol)
            )
            child.attrs[_HDF_KEY_KIND] = "pickle"
        return node

    if isinstance(value, (list, tuple, set, frozenset)):
        node = parent.create_group(name)
        container_kind = {
            list: "list",
            tuple: "tuple",
            set: "set",
            frozenset: "frozenset",
        }
        node.attrs[_HDF_KIND] = container_kind[type(value)]
        for index, item in enumerate(value):
            _write_hdf_value(node, f"{index:08d}", item, protocol)
        return node

    if value is None:
        node = parent.create_group(name)
        node.attrs[_HDF_KIND] = "none"
        return node

    if isinstance(value, str):
        import h5py

        node = parent.create_dataset(
            name, data=value, dtype=h5py.string_dtype(encoding="utf-8")
        )
        node.attrs[_HDF_KIND] = "string"
        return node

    if isinstance(value, bytes):
        node = parent.create_dataset(
            name, data=np.frombuffer(value, dtype="u1")
        )
        node.attrs[_HDF_KIND] = "bytes"
        return node

    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            return _write_pickled_hdf_value(parent, name, value, protocol)
        try:
            node = parent.create_dataset(name, data=value)
        except (TypeError, ValueError):
            return _write_pickled_hdf_value(parent, name, value, protocol)
        node.attrs[_HDF_KIND] = "array"
        return node

    if isinstance(value, np.generic):
        try:
            node = parent.create_dataset(name, data=value)
        except (TypeError, ValueError):
            return _write_pickled_hdf_value(parent, name, value, protocol)
        node.attrs[_HDF_KIND] = "numpy_scalar"
        return node

    if isinstance(value, (bool, int, float, complex)):
        try:
            node = parent.create_dataset(name, data=value)
        except OverflowError:
            if isinstance(value, int):
                node = parent.create_dataset(
                    name,
                    data=str(value),
                    dtype="S" + str(len(str(value))),
                )
                node.attrs[_HDF_KIND] = "bigint"
                return node
            return _write_pickled_hdf_value(parent, name, value, protocol)
        except (TypeError, ValueError):
            return _write_pickled_hdf_value(parent, name, value, protocol)
        node.attrs[_HDF_KIND] = "scalar"
        node.attrs[_HDF_SCALAR_TYPE] = type(value).__name__
        return node

    return _write_pickled_hdf_value(parent, name, value, protocol)


def _read_hdf_value(node: Any) -> Any:
    """Reconstruct a value written by :func:`_write_hdf_value`."""
    if _HDF_KIND not in node.attrs:
        raise ValueError(f"HDF5 node {node.name!r} has no Sequince type tag.")
    kind = _attr_text(node.attrs[_HDF_KIND])

    if kind == "mapping":
        mapping_value: dict[Any, Any] = {}
        for child_name in sorted(node.keys()):
            child = node[child_name]
            key_kind = _attr_text(child.attrs[_HDF_KEY_KIND])
            if key_kind == "string":
                key = child.attrs[_HDF_KEY]
                if isinstance(key, bytes):
                    key = key.decode("utf-8")
            elif key_kind == "pickle":
                key = pickle.loads(bytes(child.attrs[_HDF_KEY]))
            else:
                raise ValueError(
                    f"Unsupported HDF5 mapping key kind {key_kind!r}."
                )
            mapping_value[key] = _read_hdf_value(child)
        return mapping_value

    if kind in ("list", "tuple", "set", "frozenset"):
        items = [_read_hdf_value(node[name]) for name in sorted(node.keys())]
        container_constructors = {
            "list": list,
            "tuple": tuple,
            "set": set,
            "frozenset": frozenset,
        }
        return container_constructors[kind](items)

    if kind == "none":
        return None
    if kind == "string":
        string_value = node[()]
        return (
            string_value.decode("utf-8")
            if isinstance(string_value, bytes)
            else str(string_value)
        )
    if kind == "bytes":
        return node[...].tobytes()
    if kind == "array":
        return node[...]
    if kind == "numpy_scalar":
        return node[()]
    if kind == "bigint":
        return int(node[()].decode("ascii"))
    if kind == "scalar":
        scalar_value = node[()]
        if isinstance(scalar_value, np.generic):
            scalar_value = scalar_value.item()
        scalar_type = _attr_text(node.attrs[_HDF_SCALAR_TYPE])
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

    def __init__(
        self,
        *,
        every: int = 1,
        save_rng: bool = True,
        include_history: bool = True,
    ) -> None:
        if every <= 0:
            raise ValueError("every must be positive.")
        self.every = int(every)
        self.save_rng = save_rng
        self.include_history = include_history

    def is_due(self, next_iteration: int) -> bool:
        """Return whether a checkpoint is due after this iteration."""
        return next_iteration % self.every == 0

    def write(self, checkpoint: "SMCCheckpoint") -> None:
        """Apply this writer's options and persist the checkpoint."""
        prepared = replace(
            checkpoint,
            rng_state=checkpoint.rng_state if self.save_rng else None,
            history=checkpoint.history if self.include_history else None,
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
            state=copy.deepcopy(dict(state)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "module": self.module,
            "qualname": self.qualname,
            "state": copy.deepcopy(self.state),
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
            state=copy.deepcopy(dict(state)),
        )


def snapshot_steps(steps: Sequence[Any]) -> tuple[StepState, ...]:
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
            load_fn(copy.deepcopy(saved.state))


@dataclass(frozen=True)
class SMCCheckpoint:
    """Detached end-of-iteration state for resuming an SMC run."""

    state: ParticleState
    next_iteration: int
    next_observation_index: int
    metadata: dict[str, Any]
    outputs: dict[str, Any]
    step_states: tuple[StepState, ...]
    rng_state: dict[str, Any] | None = None
    history: tuple[ParticleState, ...] | None = None
    version: int = CHECKPOINT_VERSION

    @classmethod
    def from_runtime(
        cls,
        state: ParticleState,
        context: StepContext,
        result: SMCResult,
        steps: Sequence[Any],
        rng: Any | None = None,
        *,
        include_history: bool = True,
    ) -> "SMCCheckpoint":
        """Capture a checkpoint after a completed SMC iteration."""
        state_copy = ParticleState.from_snapshot(state.snapshot())
        history = None
        if include_history:
            history = tuple(
                ParticleState.from_snapshot(item.snapshot())
                for item in result.states
            )
        return cls(
            state=state_copy,
            next_iteration=context.t + 1,
            next_observation_index=context.next_observation_index,
            metadata=copy.deepcopy(
                {
                    key: value
                    for key, value in context.metadata.items()
                    if key != "particle_transform"
                }
            ),
            outputs=copy.deepcopy(dict(context.outputs)),
            step_states=snapshot_steps(steps),
            rng_state=None if rng is None else snapshot_rng(resolve_rng(rng)),
            history=history,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a detached plain-Python checkpoint representation."""
        return {
            "version": self.version,
            "state": self.state.snapshot(),
            "next_iteration": self.next_iteration,
            "next_observation_index": self.next_observation_index,
            "metadata": copy.deepcopy(self.metadata),
            "outputs": copy.deepcopy(self.outputs),
            "step_states": [item.to_dict() for item in self.step_states],
            "rng_state": copy.deepcopy(self.rng_state),
            "history": (
                None
                if self.history is None
                else [item.snapshot() for item in self.history]
            ),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SMCCheckpoint":
        """Reconstruct a checkpoint from :meth:`to_dict` output."""
        if not isinstance(payload, Mapping):
            raise TypeError("SMCCheckpoint payload must be a mapping.")
        required = {
            "version",
            "state",
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
        for key in ("metadata", "outputs"):
            if not isinstance(payload[key], Mapping):
                raise TypeError(f"SMCCheckpoint '{key}' must be a mapping.")
        step_payload = payload["step_states"]
        if not isinstance(step_payload, Sequence) or isinstance(
            step_payload, (str, bytes)
        ):
            raise TypeError("SMCCheckpoint 'step_states' must be a sequence.")
        history_payload = payload.get("history")
        if history_payload is not None and (
            not isinstance(history_payload, Sequence)
            or isinstance(history_payload, (str, bytes))
        ):
            raise TypeError(
                "SMCCheckpoint 'history' must be a sequence or None."
            )
        next_iteration = int(payload["next_iteration"])
        next_observation_index = int(payload["next_observation_index"])
        if next_iteration < 0 or next_observation_index < 0:
            raise ValueError(
                "Checkpoint iteration and observation indices must be "
                "non-negative."
            )
        return cls(
            state=ParticleState.from_snapshot(copy.deepcopy(payload["state"])),
            next_iteration=next_iteration,
            next_observation_index=next_observation_index,
            metadata=copy.deepcopy(dict(payload["metadata"])),
            outputs=copy.deepcopy(dict(payload["outputs"])),
            step_states=tuple(
                StepState.from_dict(item) for item in step_payload
            ),
            rng_state=(
                None if rng_state is None else copy.deepcopy(dict(rng_state))
            ),
            history=(
                None
                if history_payload is None
                else tuple(
                    ParticleState.from_snapshot(copy.deepcopy(item))
                    for item in history_payload
                )
            ),
            version=version,
        )

    def restore_rng(self) -> Any | None:
        """Return a live ORNG generator restored from this checkpoint."""
        if self.rng_state is None:
            return None
        return restore_rng(self.rng_state)

    def result(
        self, shared_context: Mapping[str, Any] | None = None
    ) -> SMCResult:
        """Return the result history represented by this checkpoint."""
        states = list(self.history or (self.state,))
        context = StepContext(
            t=self.next_iteration - 1,
            next_observation_index=self.next_observation_index,
            shared=dict(shared_context or {}),
            metadata=copy.deepcopy(self.metadata),
            outputs=copy.deepcopy(self.outputs),
        )
        return SMCResult(states=states, context=context)

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
        return cls.from_dict(payload)

    def save_hdf5(
        self,
        h5_file,
        path: str = "checkpoint",
        dsetname: str = "runtime",
        protocol: int | None = None,
        *,
        replace_history: bool = False,
    ) -> None:
        """Save runtime state and append result history in HDF5."""
        protocol = (
            pickle.HIGHEST_PROTOCOL if protocol is None else int(protocol)
        )
        group = h5_file.require_group(path)
        runtime_payload = self.to_dict()
        runtime_payload.pop("history")

        if self.history is None:
            if "history" in group:
                del group["history"]
            runtime_payload["history_length"] = 0
        else:
            history_group = group.require_group("history")
            if replace_history:
                del group["history"]
                history_group = group.require_group("history")
            existing = len(history_group)
            if existing > len(self.history):
                del group["history"]
                history_group = group.require_group("history")
                existing = 0
            for index in range(existing, len(self.history)):
                _write_hdf_value(
                    history_group,
                    f"{index:08d}",
                    self.history[index].snapshot(),
                    protocol,
                )
            history_group.attrs["length"] = len(self.history)
            runtime_payload["history_length"] = len(self.history)

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
        history_length = int(checkpoint_payload.pop("history_length", 0))
        if history_length:
            if "history" not in group:
                raise ValueError("Checkpoint history is missing from HDF5.")
            history_group = group["history"]
            checkpoint_payload["history"] = [
                _read_hdf_value(history_group[f"{index:08d}"])
                for index in range(history_length)
            ]
        return cls.from_dict(checkpoint_payload)


class HDF5CheckpointWriter(CheckpointWriter):
    """Persist committed SMC checkpoints to HDF5 at a fixed cadence."""

    def __init__(
        self,
        file_path: str | Path,
        *,
        every: int = 1,
        path: str = "checkpoint",
        dsetname: str = "runtime",
        save_rng: bool = True,
        include_history: bool = True,
    ) -> None:
        super().__init__(
            every=every,
            save_rng=save_rng,
            include_history=include_history,
        )
        file_path = Path(file_path)
        if not file_path.name.lower().endswith((".h5", ".hdf5")):
            raise ValueError(
                "Checkpoint file must be an HDF5 file (.h5 or .hdf5)."
            )
        self.file_path = file_path
        self.path = path
        self.dsetname = dsetname
        self._has_written = False
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

    def _write(self, checkpoint: SMCCheckpoint) -> None:
        """Write a committed checkpoint to the configured HDF5 location."""
        import h5py

        with h5py.File(self.file_path, "a") as h5_file:
            checkpoint.save_hdf5(
                h5_file,
                path=self.path,
                dsetname=self.dsetname,
                replace_history=not self._has_written,
            )
        self._has_written = True


class PickleCheckpointWriter(CheckpointWriter):
    """Atomically persist committed SMC checkpoints as pickle files."""

    def __init__(
        self,
        file_path: str | Path,
        *,
        every: int = 1,
        save_rng: bool = True,
        include_history: bool = True,
        protocol: int | None = None,
    ) -> None:
        super().__init__(
            every=every,
            save_rng=save_rng,
            include_history=include_history,
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
