from __future__ import annotations

import math
from typing import Any

import numpy as _np
from array_api_compat import (
    array_namespace as _array_namespace,
)
from array_api_compat import device as get_device
from array_api_compat import (
    is_cupy_namespace,
    is_jax_array,
    is_jax_namespace,
    is_numpy_namespace,
    is_torch_array,
    is_torch_namespace,
    to_device,
)

__all__ = [
    "asarray",
    "copy",
    "logsumexp",
    "namespace",
    "resolve_dtype",
    "to_numpy",
]


def namespace(*arrays: Any):
    """Return the array API namespace associated with the provided arrays."""
    filtered = [array for array in arrays if array is not None]
    if not filtered:
        return _np
    return _array_namespace(*filtered)


def _namespace_or_numpy(array: Any):
    """Infer an array namespace, defaulting array-like inputs to NumPy."""
    try:
        return namespace(array)
    except (TypeError, ValueError):
        return _np


def _dtype_name(dtype: Any) -> str:
    """Return a backend-independent name for a dtype-like object."""
    if isinstance(dtype, str):
        name = dtype
    elif getattr(dtype, "name", None):
        name = dtype.name
    elif hasattr(dtype, "__name__"):
        name = dtype.__name__
    else:
        name = str(dtype)
        if name.startswith("<class '") and name.endswith("'>"):
            name = name.split("'")[1]
        if name.startswith("dtype(") and name.endswith(")"):
            name = name[6:-1].strip("'\" ")
    return name.split(".")[-1].strip(" '\"<>").lower()


def resolve_dtype(dtype: Any | str | None, xp: Any) -> Any | None:
    """Resolve a dtype-like value for an array API namespace."""
    if dtype is None or xp is None:
        return dtype
    if is_torch_namespace(xp) and str(dtype).startswith("torch."):
        return dtype

    name = _dtype_name(dtype)
    if not name:
        raise ValueError(f"Could not infer a dtype name from {dtype!r}.")
    resolved = getattr(xp, name, None)
    if resolved is not None:
        return resolved
    try:
        return xp.dtype(name)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Unknown dtype {dtype!r} for namespace "
            f"{getattr(xp, '__name__', xp)!r}."
        ) from exc


def _same_namespace(first: Any, second: Any) -> bool:
    if first is second:
        return True
    predicates = (
        is_numpy_namespace,
        is_torch_namespace,
        is_jax_namespace,
        is_cupy_namespace,
    )
    return any(
        predicate(first) and predicate(second) for predicate in predicates
    )


def _cross_backend_array(array: Any, xp: Any) -> Any:
    """Convert between namespaces, preferring zero-copy DLPack paths."""
    if is_numpy_namespace(xp):
        return to_numpy(array)
    if is_jax_array(array) and is_torch_namespace(xp):
        import torch

        return torch.utils.dlpack.from_dlpack(array)
    if is_torch_array(array) and is_jax_namespace(xp):
        import jax.dlpack

        return jax.dlpack.from_dlpack(array.detach())

    from_dlpack = getattr(xp, "from_dlpack", None)
    if callable(from_dlpack) and hasattr(array, "__dlpack__"):
        try:
            return from_dlpack(array)
        except (BufferError, RuntimeError, TypeError):
            pass
    return xp.asarray(to_numpy(array))


def asarray(
    array: Any,
    *,
    xp: Any | None = None,
    dtype: Any | str | None = None,
    like: Any | None = None,
):
    """Convert an array while preserving the requested namespace and device."""
    if xp is None:
        xp = (
            namespace(like) if like is not None else _namespace_or_numpy(array)
        )
    resolved_dtype = resolve_dtype(dtype, xp)
    try:
        source_namespace = namespace(array)
    except (TypeError, ValueError):
        source_namespace = None

    if source_namespace is not None and not _same_namespace(
        source_namespace, xp
    ):
        converted = _cross_backend_array(array, xp)
        result = xp.asarray(converted, dtype=resolved_dtype)
    else:
        result = xp.asarray(array, dtype=resolved_dtype)
    if like is None:
        return result
    return to_device(result, get_device(like))


def copy(array: Any, *, xp: Any | None = None):
    """Copy an array without changing its namespace or device."""
    if xp is None:
        xp = _namespace_or_numpy(array)
    arr = asarray(array, xp=xp)
    if hasattr(xp, "copy"):
        try:
            return xp.copy(arr)  # type: ignore[attr-defined]
        except AttributeError:  # pragma: no cover - namespace lacks copy
            pass
    if hasattr(arr, "clone"):
        return arr.clone()
    return xp.asarray(arr)


def _numpy_result(
    array: Any,
    *,
    dtype: Any | None,
    copy: bool | None,
) -> _np.ndarray:
    if copy is None:
        return _np.asarray(array, dtype=dtype)
    return _np.array(array, dtype=dtype, copy=copy)


def to_numpy(
    array: Any,
    *,
    dtype: Any | None = None,
    copy: bool | None = None,
) -> _np.ndarray:
    """Detach an array, move it to CPU, and convert it to NumPy."""
    if is_torch_array(array):
        return _numpy_result(
            array.detach().cpu().numpy(), dtype=dtype, copy=copy
        )
    try:
        array = to_device(array, "cpu")
    except (AttributeError, NotImplementedError, TypeError, ValueError):
        if hasattr(array, "get"):
            array = array.get()
        elif hasattr(array, "cpu"):
            array = array.cpu()
    if hasattr(array, "numpy"):
        array = array.numpy()
    return _numpy_result(array, dtype=dtype, copy=copy)


def logsumexp(array: Any, *, xp: Any | None = None):
    if xp is None:
        xp = _namespace_or_numpy(array)
    arr = asarray(array, xp=xp)
    max_value = xp.max(arr)
    if not math.isfinite(float(to_numpy(max_value))):
        return max_value
    stable = xp.exp(arr - max_value)
    return xp.log(xp.sum(stable)) + max_value
