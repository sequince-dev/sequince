from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as _np
from array_api_compat import array_namespace as _array_namespace
from array_api_compat import device as get_device
from array_api_compat import to_device


def _preferred_device(*arrays: Any):
    """Return the first available device descriptor from the provided arrays."""
    for arr in arrays:
        if arr is None:
            continue
        return get_device(arr)
    return None


def _move_to_device(array: Any, device: Any):
    """Move ``array`` onto ``device`` when supported by the backend."""
    if device is None:
        return array
    return to_device(array, device)


def _match_device(array: Any, *references: Any) -> Any:
    """Return ``array`` residing on the first device detected in ``references``."""
    return _move_to_device(array, _preferred_device(*references))


def namespace(*arrays: Any):
    """Return the array API namespace associated with the provided arrays."""
    filtered = [arr for arr in arrays if arr is not None]
    if not filtered:
        return _np
    return _array_namespace(*filtered)


def asarray(array: Any, *, xp=None, like=None):
    xp = xp or namespace(array, like)
    arr = xp.asarray(array)
    return _match_device(arr, like, array)


def copy(array: Any, *, xp=None):
    xp = xp or namespace(array)
    arr = asarray(array, xp=xp)
    if hasattr(xp, "copy"):
        try:
            return xp.copy(arr)  # type: ignore[attr-defined]
        except AttributeError:  # pragma: no cover - namespace lacks copy
            pass
    if hasattr(arr, "clone"):
        return arr.clone()
    return xp.asarray(arr)


def to_numpy(array: Any) -> _np.ndarray:
    """Convert an arbitrary array API object to a NumPy array."""
    if isinstance(array, _np.ndarray):
        return array
    if hasattr(array, "detach"):
        array = array.detach()
    if hasattr(array, "cpu"):
        array = array.cpu()
    if hasattr(array, "numpy"):
        return _np.asarray(array.numpy())
    if hasattr(array, "tolist"):
        return _np.asarray(array.tolist())
    return _np.asarray(array)


def full(
    shape: Iterable[int],
    fill_value: Any,
    *,
    xp=None,
    dtype=None,
    like=None,
):
    xp = xp or namespace(fill_value, like)
    arr = xp.full(tuple(shape), fill_value, dtype=dtype)
    return _match_device(arr, like)


def arange(*range_args: Any, xp=None, dtype=None, like=None):
    xp = xp or namespace(like)
    arr = xp.arange(*range_args, dtype=dtype)
    return _match_device(arr, like)


def logsumexp(array: Any, *, xp=None):
    xp = xp or namespace(array)
    arr = asarray(array, xp=xp)
    max_val = xp.max(arr)
    stable = xp.exp(arr - max_val)
    return xp.log(xp.sum(stable)) + max_val


def normalize_log_weights(log_weights: Any, *, xp=None):
    xp = xp or namespace(log_weights)
    lw = asarray(log_weights, xp=xp)
    log_norm = logsumexp(lw, xp=xp)
    log_norm_scalar = float(log_norm)
    if math.isfinite(log_norm_scalar):
        normalized = lw - log_norm_scalar
        return normalized, log_norm_scalar, True
    n = lw.shape[0]
    fill_value = -math.log(n)
    return full((n,), fill_value, xp=xp, dtype=lw.dtype, like=lw), 0.0, False
