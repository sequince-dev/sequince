"""HDF5 storage and logical-schema helpers for Sequince."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Complex, Integral, Real
from typing import Any

import h5py
import numpy as np

# Private attributes preserve distinctions that HDF5's group/dataset types lose.
_NODE_TYPE = "_sequince_hdf5_node_type"
_PYTHON_TYPE = "_sequince_hdf5_python_type"
_NUMPY_DTYPE = "_sequince_hdf5_numpy_dtype"

# Zero-padding makes lexicographic HDF5 key order match numeric sequence order.
_SEQUENCE_WIDTH = 8

SCHEMA_VERSION = 1
KIND_KEY = "_sequince_kind"
SCHEMA_KEY = "_sequince_schema_version"
VERSION_KEY = "_sequince_version"


def schema_header(
    kind: str, *, include_version: bool = True
) -> dict[str, Any]:
    """Build the common logical header for a Sequince HDF5 payload."""
    from .. import __version__

    payload: dict[str, Any] = {
        KIND_KEY: kind,
        SCHEMA_KEY: SCHEMA_VERSION,
    }
    if include_version:
        payload[VERSION_KEY] = __version__
    return payload


def validate_schema(
    payload: Mapping[str, Any],
    *,
    kinds: set[str],
    required: set[str],
    optional: set[str] | frozenset[str] = frozenset(),
) -> str:
    """Validate the common header and fields of a logical HDF5 payload."""
    if not isinstance(payload, Mapping):
        raise TypeError("Sequince HDF5 payload must be a mapping.")
    kind = payload.get(KIND_KEY)
    if kind not in kinds:
        expected = ", ".join(sorted(kinds))
        raise ValueError(
            f"Unexpected Sequince HDF5 kind {kind!r}; expected {expected}."
        )
    version = payload.get(SCHEMA_KEY)
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported {kind} schema version {version!r}; expected "
            f"{SCHEMA_VERSION}."
        )
    missing = required.difference(payload)
    if missing:
        raise ValueError(
            f"{kind} payload is missing required fields: "
            + ", ".join(sorted(missing))
        )
    allowed = required | optional | {KIND_KEY, SCHEMA_KEY, VERSION_KEY}
    unexpected = set(payload).difference(allowed)
    if unexpected:
        raise ValueError(
            f"{kind} payload has unexpected fields: "
            + ", ".join(sorted(unexpected))
        )
    return kind


def nested(payload: dict[str, Any]) -> dict[str, Any]:
    """Remove root-only producer information from a nested domain payload."""
    payload.pop(VERSION_KEY, None)
    return payload


def _decode_hdf5_text(value: Any) -> str:
    """Normalize an HDF5 string value or attribute to ``str``."""
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _create_string_dataset(parent: Any, name: str, value: str) -> Any:
    """Create one scalar UTF-8 HDF5 string dataset."""
    return parent.create_dataset(
        name, data=value, dtype=h5py.string_dtype(encoding="utf-8")
    )


def _read_string_dataset(node: Any) -> str:
    """Read one scalar UTF-8 HDF5 string dataset."""
    return _decode_hdf5_text(node[()])


def _validate_handle(handle: Any, *, writable: bool) -> None:
    if not isinstance(handle, (h5py.File, h5py.Group)):
        raise TypeError("Expected an open h5py.File or h5py.Group.")
    if not bool(handle.id.valid):
        raise ValueError("HDF5 file or group is closed.")
    if writable and handle.file.mode == "r":
        raise OSError("HDF5 file is not writable.")


def _validate_key(key: Any, *, logical_path: str) -> str:
    if not isinstance(key, str):
        raise TypeError(
            f"HDF5 mapping key at {logical_path} must be a string."
        )
    if not key or "/" in key or "\x00" in key or key in {".", ".."}:
        raise ValueError(
            f"Invalid HDF5 mapping key {key!r} at {logical_path}."
        )
    return key


def _path_parts(path: str) -> tuple[str, ...]:
    if not isinstance(path, str):
        raise TypeError("HDF5 path must be a string.")
    if not path:
        raise ValueError("HDF5 path must be non-empty.")
    if path.startswith("/"):
        raise ValueError("HDF5 path must be relative.")
    parts = tuple(path.split("/"))
    for index, part in enumerate(parts):
        _validate_key(part, logical_path=f"path component {index}")
    return parts


def _is_array_candidate(value: Any) -> bool:
    return isinstance(value, np.ndarray) or (
        hasattr(value, "shape") and hasattr(value, "dtype")
    )


def _validate_array_dtype(dtype: np.dtype[Any], logical_path: str) -> None:
    if dtype.hasobject or dtype.fields is not None or dtype.kind in "MmU":
        raise TypeError(
            f"Unsupported array dtype {dtype!s} at {logical_path}."
        )
    if dtype.kind not in "biufcS":
        raise TypeError(
            f"Unsupported array dtype {dtype!s} at {logical_path}."
        )


def _numpy_dtype(dtype: Any, logical_path: str) -> np.dtype[Any]:
    try:
        return np.dtype(dtype)
    except (TypeError, ValueError):
        name = str(dtype).rsplit(".", 1)[-1]
        try:
            return np.dtype(name)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"Unsupported array dtype {dtype!r} at {logical_path}."
            ) from exc


def _validate_value_tree(
    value: Any, logical_path: str, active: set[int]
) -> None:
    """Validate structure and leaves without materializing array data."""
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active:
            raise ValueError(f"Cyclic value at {logical_path}.")
        active.add(identity)
        try:
            for key, child in value.items():
                child_key = _validate_key(key, logical_path=logical_path)
                _validate_value_tree(
                    child, f"{logical_path}/{child_key}", active
                )
        finally:
            active.remove(identity)
        return
    if isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in active:
            raise ValueError(f"Cyclic value at {logical_path}.")
        active.add(identity)
        try:
            for index, child in enumerate(value):
                _validate_value_tree(child, f"{logical_path}/{index}", active)
        finally:
            active.remove(identity)
        return
    if value is None or isinstance(value, (str, bytes, bool)):
        return
    if isinstance(value, np.generic):
        _validate_array_dtype(value.dtype, logical_path)
        return
    if isinstance(value, (Integral, Real, Complex)):
        return
    if _is_array_candidate(value):
        dtype = _numpy_dtype(value.dtype, logical_path)
        _validate_array_dtype(dtype, logical_path)
        return
    raise TypeError(
        f"Unsupported value of type {type(value).__name__} at {logical_path}."
    )


def _tag(node: Any, kind: str) -> Any:
    node.attrs[_NODE_TYPE] = kind
    return node


def _write_value(
    parent: Any, name: str, value: Any, logical_path: str
) -> None:
    if isinstance(value, Mapping):
        group = _tag(parent.create_group(name), "mapping")
        for key, child in value.items():
            _write_value(group, key, child, f"{logical_path}/{key}")
        return
    if isinstance(value, (list, tuple)):
        kind = "list" if isinstance(value, list) else "tuple"
        group = _tag(parent.create_group(name), kind)
        for index, child in enumerate(value):
            child_name = f"{index:0{_SEQUENCE_WIDTH}d}"
            _write_value(group, child_name, child, f"{logical_path}/{index}")
        return
    if value is None:
        _tag(parent.create_group(name), "none")
        return
    if isinstance(value, str):
        node = _create_string_dataset(parent, name, value)
        _tag(node, "string")
        return
    if isinstance(value, bytes):
        _tag(
            parent.create_dataset(
                name, data=np.frombuffer(value, dtype=np.uint8)
            ),
            "bytes",
        )
        return
    if isinstance(value, np.generic):
        node = _tag(parent.create_dataset(name, data=value), "numpy_scalar")
        node.attrs[_NUMPY_DTYPE] = value.dtype.str
        return
    if isinstance(value, (bool, Integral, Real, Complex)):
        node = _tag(parent.create_dataset(name, data=value), "python_scalar")
        if isinstance(value, bool):
            scalar_type = "bool"
        elif isinstance(value, Integral):
            scalar_type = "int"
        elif isinstance(value, Real):
            scalar_type = "float"
        else:
            scalar_type = "complex"
        node.attrs[_PYTHON_TYPE] = scalar_type
        return

    from ..array_api import to_numpy

    try:
        array = to_numpy(value)
        _validate_array_dtype(array.dtype, logical_path)
        _tag(parent.create_dataset(name, data=array), "array")
    except Exception as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        exc.add_note(f"while writing HDF5 value at {logical_path}")
        raise


def recursively_save_to_h5_file(
    h5_file_or_group: Any,
    path: str,
    dictionary: Mapping[str, Any],
) -> None:
    """Save a portable mapping below an open HDF5 file or group."""
    _validate_handle(h5_file_or_group, writable=True)
    parts = _path_parts(path)
    if not isinstance(dictionary, Mapping):
        raise TypeError("The HDF5 root value must be a mapping.")
    _validate_value_tree(dictionary, path, set())

    parent = h5_file_or_group
    for component in parts[:-1]:
        if component in parent and not hasattr(parent[component], "keys"):
            raise FileExistsError(
                f"HDF5 parent path component {component!r} is not a group."
            )
        parent = parent.require_group(component)
    target = parts[-1]
    if target in parent:
        raise FileExistsError(f"HDF5 path {path!r} already exists.")
    try:
        _write_value(parent, target, dictionary, path)
    except BaseException as exc:
        if target in parent:
            try:
                del parent[target]
            except Exception as cleanup_error:
                if isinstance(exc, Exception):
                    exc.add_note(
                        "failed to clean up incomplete HDF5 path "
                        f"{path!r}: {cleanup_error}"
                    )
        if isinstance(exc, Exception):
            exc.add_note(f"while saving HDF5 path {path!r}")
        raise


def _attr_text(node: Any, name: str) -> str:
    if name not in node.attrs:
        raise ValueError(
            f"HDF5 node {node.name!r} is missing attribute {name!r}."
        )
    return _decode_hdf5_text(node.attrs[name])


def _require_group(node: Any, kind: str) -> None:
    if not isinstance(node, h5py.Group):
        raise ValueError(f"HDF5 {kind} node {node.name!r} must be a group.")


def _require_dataset(node: Any, kind: str) -> None:
    if not isinstance(node, h5py.Dataset):
        raise ValueError(f"HDF5 {kind} node {node.name!r} must be a dataset.")


def _read_value(node: Any) -> Any:
    kind = _attr_text(node, _NODE_TYPE)
    if kind == "mapping":
        _require_group(node, kind)
        mapping_values = {}
        for name in node:
            _validate_key(name, logical_path=node.name)
            mapping_values[name] = _read_value(node[name])
        return mapping_values
    if kind in {"list", "tuple"}:
        _require_group(node, kind)
        expected = [
            f"{index:0{_SEQUENCE_WIDTH}d}" for index in range(len(node))
        ]
        if sorted(node.keys()) != expected:
            raise ValueError(
                f"HDF5 sequence node {node.name!r} has non-contiguous indices."
            )
        sequence_values = [_read_value(node[name]) for name in expected]
        return sequence_values if kind == "list" else tuple(sequence_values)
    if kind == "none":
        _require_group(node, kind)
        if len(node):
            raise ValueError(f"HDF5 None node {node.name!r} must be empty.")
        return None
    if kind == "string":
        _require_dataset(node, kind)
        if node.shape != ():
            raise ValueError(f"Malformed HDF5 string node {node.name!r}.")
        return _read_string_dataset(node)
    if kind == "bytes":
        _require_dataset(node, kind)
        if node.dtype != np.dtype(np.uint8) or node.ndim != 1:
            raise ValueError(f"Malformed HDF5 bytes node {node.name!r}.")
        return node[...].tobytes()
    if kind == "python_scalar":
        _require_dataset(node, kind)
        if node.shape != ():
            raise ValueError(
                f"Malformed HDF5 Python scalar node {node.name!r}."
            )
        scalar_type = _attr_text(node, _PYTHON_TYPE)
        constructors = {
            "bool": bool,
            "int": int,
            "float": float,
            "complex": complex,
        }
        if scalar_type not in constructors:
            raise ValueError(
                f"Unsupported Python scalar type {scalar_type!r} at {node.name!r}."
            )
        return constructors[scalar_type](node[()])
    if kind == "numpy_scalar":
        _require_dataset(node, kind)
        dtype = np.dtype(_attr_text(node, _NUMPY_DTYPE))
        if node.shape != () or node.dtype != dtype:
            raise ValueError(f"Malformed NumPy scalar node {node.name!r}.")
        return node[()]
    if kind == "array":
        _require_dataset(node, kind)
        _validate_array_dtype(node.dtype, node.name)
        return np.asarray(node[...])
    raise ValueError(f"Unsupported HDF5 node type {kind!r} at {node.name!r}.")


def load_from_h5_file(h5_file_or_group: Any, path: str) -> dict[str, Any]:
    """Load a portable mapping below an open HDF5 file or group."""
    _validate_handle(h5_file_or_group, writable=False)
    parts = _path_parts(path)
    node = h5_file_or_group
    for component in parts:
        if not isinstance(node, h5py.Group):
            raise KeyError(f"HDF5 path {path!r} does not exist.")
        if component not in node:
            raise KeyError(f"HDF5 path {path!r} does not exist.")
        node = node[component]
    value = _read_value(node)
    if not isinstance(value, dict):
        raise TypeError(f"HDF5 root at {path!r} is not a mapping.")
    return value
