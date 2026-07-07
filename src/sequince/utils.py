from __future__ import annotations

import copy
import inspect
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, TypeAlias

import h5py
import numpy as np
from array_api_compat import is_torch_namespace
from orng import ArrayRNG, RandomGenerator

from ._array_api import namespace
from ._typing import Array

if TYPE_CHECKING:
    from .steps.base import StepContext

logger = logging.getLogger(__name__)

RNGLike: TypeAlias = RandomGenerator | np.random.Generator | int | None


def resolve_rng(
    rng: RNGLike = None,
    *,
    backend: str | None = None,
) -> RandomGenerator | np.random.Generator:
    """Return the provided RNG or create a generator for the requested backend."""
    if isinstance(rng, RandomGenerator):
        return rng
    if isinstance(rng, np.random.Generator):
        return rng
    if isinstance(rng, int):
        return RandomGenerator(backend=backend or "numpy", seed=rng)
    if rng is None:
        return RandomGenerator(backend=backend or "numpy")
    raise TypeError(
        "rng must be an RandomGenerator, numpy.random.Generator, an int seed, or None."
    )


def call_with_optional_args(
    fn: Callable[..., Any],
    base_args: tuple[Any, ...],
    optional_args: tuple[Any, ...] = (),
) -> Any:
    """Call fn with optional positional arguments when its signature allows it."""
    signature = inspect.signature(fn)
    params = signature.parameters.values()

    positional = [
        p
        for p in params
        if p.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    required = sum(1 for p in positional if p.default is inspect._empty)
    has_varargs = any(
        p.kind == inspect.Parameter.VAR_POSITIONAL for p in params
    )

    base_count = len(base_args)
    full_count = base_count + len(optional_args)
    max_positional = float("inf") if has_varargs else len(positional)

    if required <= full_count <= max_positional:
        return fn(*base_args, *optional_args)
    if required <= base_count <= max_positional:
        return fn(*base_args)
    raise TypeError(
        "Function cannot accept the provided positional arguments."
    )


def _snapshot_numpy_generator_state(
    rng: np.random.Generator,
) -> dict[str, Any]:
    return {
        "kind": "numpy.random.Generator",
        "bit_generator_state": copy.deepcopy(rng.bit_generator.state),
    }


def _restore_numpy_generator(state: Mapping[str, Any]) -> np.random.Generator:
    try:
        bit_generator_state = copy.deepcopy(state["bit_generator_state"])
    except KeyError as exc:
        raise ValueError(
            "NumPy RNG snapshot is missing 'bit_generator_state'."
        ) from exc
    rng = np.random.default_rng()
    rng.bit_generator.state = bit_generator_state
    return rng


def _snapshot_orng_state(backend: str, state: Any) -> dict[str, Any]:
    normalized_backend = backend.lower()
    if normalized_backend == "numpy":
        generator = state
        if isinstance(generator, np.random.Generator):
            return {
                "kind": "numpy",
                "bit_generator_state": copy.deepcopy(
                    generator.bit_generator.state
                ),
            }
    if normalized_backend in {"torch", "pytorch"}:
        generator = state
        if hasattr(generator, "get_state"):
            generator_device = getattr(generator, "device", None)
            return {
                "kind": "torch",
                "generator_state": generator.get_state().clone(),
                "device": None
                if generator_device is None
                else str(generator_device),
            }
    if normalized_backend == "cupy":
        generator = state
        if hasattr(generator, "bit_generator"):
            return {
                "kind": "cupy",
                "bit_generator_state": copy.deepcopy(
                    generator.bit_generator.state
                ),
            }
    if normalized_backend == "jax":
        xp = namespace(state)
        return {
            "kind": "jax",
            "key": xp.asarray(state),
        }
    return {
        "kind": normalized_backend,
        "state": copy.deepcopy(state),
    }


def _restore_orng_backend_state(
    backend: str,
    state: Mapping[str, Any],
) -> Any:
    normalized_backend = backend.lower()
    state_kind = state.get("kind")
    if normalized_backend == "numpy":
        if state_kind != "numpy":
            raise ValueError(f"Expected NumPy ORNG state, got '{state_kind}'.")
        rng = np.random.default_rng()
        rng.bit_generator.state = copy.deepcopy(state["bit_generator_state"])
        return rng
    if normalized_backend in {"torch", "pytorch"}:
        if state_kind != "torch":
            raise ValueError(f"Expected Torch ORNG state, got '{state_kind}'.")
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Restoring a torch-backed ORNG RNG requires 'torch'."
            ) from exc
        device = state.get("device") or "cpu"
        generator = torch.Generator(device=device)
        generator_state = state["generator_state"]
        if hasattr(generator_state, "clone"):
            generator_state = generator_state.clone()
        generator.set_state(generator_state)
        return generator
    if normalized_backend == "cupy":
        if state_kind != "cupy":
            raise ValueError(f"Expected CuPy ORNG state, got '{state_kind}'.")
        try:
            import cupy as cp
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Restoring a cupy-backed ORNG RNG requires 'cupy'."
            ) from exc
        generator = cp.random.default_rng()
        generator.bit_generator.state = copy.deepcopy(
            state["bit_generator_state"]
        )
        return generator
    if normalized_backend == "jax":
        if state_kind != "jax":
            raise ValueError(f"Expected JAX ORNG state, got '{state_kind}'.")
        return state["key"]
    if "state" not in state:
        raise ValueError(
            f"Unsupported ORNG RNG snapshot payload for backend '{backend}'."
        )
    return copy.deepcopy(state["state"])


def snapshot_rng(rng: RandomGenerator | np.random.Generator) -> dict[str, Any]:
    """Capture the current RNG state in a plain Python mapping."""
    if isinstance(rng, np.random.Generator):
        return _snapshot_numpy_generator_state(rng)
    if isinstance(rng, RandomGenerator):
        _, backend_state = rng.to_functional(pure=False)
        return {
            "kind": "orng.RandomGenerator",
            "class_name": type(rng).__name__,
            "backend": rng.backend,
            "device": None if rng.device is None else str(rng.device),
            "state": _snapshot_orng_state(rng.backend, backend_state),
        }
    raise TypeError(
        "rng must be a numpy.random.Generator or orng.RandomGenerator."
    )


def restore_rng(
    snapshot: Mapping[str, Any],
) -> RandomGenerator | np.random.Generator:
    """Restore an RNG from a snapshot created by :func:`snapshot_rng`."""
    if not isinstance(snapshot, Mapping):
        raise TypeError("RNG snapshot must be a mapping.")
    kind = snapshot.get("kind")
    if kind == "numpy.random.Generator":
        return _restore_numpy_generator(snapshot)
    if kind == "orng.RandomGenerator":
        try:
            backend = snapshot["backend"]
            state = snapshot["state"]
        except KeyError as exc:
            raise ValueError(
                "ORNG RNG snapshot must include 'backend' and 'state'."
            ) from exc
        if not isinstance(state, Mapping):
            raise TypeError("ORNG RNG snapshot 'state' must be a mapping.")
        rng_type = (
            ArrayRNG
            if snapshot.get("class_name") == "ArrayRNG"
            else RandomGenerator
        )
        generator = _restore_orng_backend_state(backend, state)
        device = snapshot.get("device")
        kwargs: dict[str, Any] = {
            "backend": backend,
            "generator": generator,
        }
        if device is not None:
            kwargs["device"] = device
        return rng_type(**kwargs)
    raise ValueError(f"Unsupported RNG snapshot kind '{kind}'.")


def call_with_optional_context(
    fn: Callable[..., Any],
    base_args: tuple[Any, ...],
    context: StepContext,
) -> Any:
    """Call fn with base_args and append context only if required."""
    return call_with_optional_args(fn, base_args, (context,))


def in_bounds(particles: Array, bounds: Array, xp: Any | None = None) -> Array:
    """Check if particles are within the specified bounds.

    Bounds are assumed to be rectilinear, with shape (2, n_dimensions).


    Parameters
    ----------
    particles
        Array of shape (n_particles, n_dimensions) representing the particles.
    bounds
        Array of shape (2, n_dimensions) where bounds[0] are the lower bounds
        and bounds[1] are the upper bounds for each dimension.

    Returns
    -------
    Array
        Boolean array of shape (n_particles,) indicating whether each particle
        is within bounds.
    """
    if xp is None:
        xp = namespace(particles, bounds)
    if bounds.shape[0] != 2:
        raise ValueError("Bounds array must have shape (2, n_dimensions).")
    if bounds.shape[1] != particles.shape[-1]:
        raise ValueError(
            "Bounds array must have the same number of dimensions as particles."
        )
    return xp.all((particles >= bounds[0]) & (particles <= bounds[1]), axis=-1)


def _dtype_to_name(dtype: Any | str | None) -> str | None:
    if dtype is None:
        return None
    if isinstance(dtype, str):
        name = dtype
    elif hasattr(dtype, "name") and getattr(dtype, "name"):
        name = dtype.name
    elif hasattr(dtype, "__name__"):
        name = dtype.__name__
    else:
        text = str(dtype)
        if text.startswith("<class '") and text.endswith("'>"):
            text = text.split("'")[1]
        if text.startswith("dtype(") and text.endswith(")"):
            text = text[6:-1].strip("'\" ")
        name = text
    name = name.split(".")[-1]
    return name.strip(" '\"<>").lower()


def resolve_dtype(dtype: Any | str | None, xp: Any) -> Any | None:
    if dtype is None or xp is None:
        return dtype
    if isinstance(dtype, str):
        dtype_name = _dtype_to_name(dtype)
        if is_torch_namespace(xp):
            resolved = getattr(xp, dtype_name, None)
            if resolved is None:
                raise ValueError(
                    f"Unknown dtype '{dtype}' for namespace {xp.__name__}"
                )
            return resolved
        try:
            return xp.dtype(dtype_name)
        except (AttributeError, TypeError, ValueError):
            resolved = getattr(xp, dtype_name, None)
            if resolved is not None:
                return resolved
            raise ValueError(
                f"Unknown dtype '{dtype}' for namespace "
                f"{getattr(xp, '__name__', xp)}"
            )
    if is_torch_namespace(xp):
        return dtype
    try:
        return xp.dtype(dtype)
    except (AttributeError, TypeError, ValueError):
        return dtype


def encode_dtype(xp: Any, dtype: Any) -> dict[str, Any] | None:
    if dtype is None:
        return None
    return {
        "__dtype__": True,
        "xp": xp.__name__,
        "dtype": _dtype_to_name(dtype),
    }


def decode_dtype(xp: Any, encoded_dtype: Any) -> Any:
    if isinstance(encoded_dtype, dict) and encoded_dtype.get("__dtype__"):
        if encoded_dtype["xp"] != xp.__name__:
            raise ValueError(
                f"Encoded dtype xp {encoded_dtype['xp']} does not match "
                f"current xp {xp.__name__}"
            )
        if is_torch_namespace(xp):
            return getattr(xp, encoded_dtype["dtype"].split(".")[-1])
        return xp.dtype(encoded_dtype["dtype"].split(".")[-1])
    return encoded_dtype


def to_numpy(value: Any) -> Any:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def encode_for_hdf5(value: Any) -> Any:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (int, float, str, bool)):
        return value
    if isinstance(value, (list, tuple)):
        if all(isinstance(v, str) for v in value):
            dt = h5py.string_dtype(encoding="utf-8")
            return np.array(value, dtype=dt)
        return [encode_for_hdf5(v) for v in value]
    if isinstance(value, dict):
        if not value:
            return "__empty_dict__"
        return {k: encode_for_hdf5(v) for k, v in value.items()}
    if value is None:
        return "__none__"
    return value


def decode_from_hdf5(value: Any) -> Any:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        if value == "__none__":
            return None
        if value == "__empty_dict__":
            return {}
        return value
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return value.item()
        if value.dtype.kind in {"S", "O", "U"}:
            try:
                return value.astype(str).tolist()
            except Exception:
                return value
        return value
    if isinstance(value, list):
        return [decode_from_hdf5(v) for v in value]
    if isinstance(value, tuple):
        return tuple(decode_from_hdf5(v) for v in value)
    if isinstance(value, dict):
        return {
            (
                k.decode("utf-8") if isinstance(k, bytes) else k
            ): decode_from_hdf5(v)
            for k, v in value.items()
        }
    return value


def recursively_save_to_h5_file(
    h5_file, path: str, dictionary: dict[str, Any]
) -> None:
    """Save a nested dictionary to an HDF5 group using flattened keys."""
    group = h5_file.require_group(path)

    def _save_flattened(g, prefix: str, mapping: dict[str, Any]) -> None:
        for key, value in mapping.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                _save_flattened(g, full_key, value)
                continue
            if full_key in g:
                del g[full_key]
            try:
                g.create_dataset(full_key, data=encode_for_hdf5(value))
            except (TypeError, ValueError) as error:
                logger.warning(
                    "Falling back to string encoding for key %s: %s",
                    full_key,
                    error,
                )
                dt = h5py.string_dtype(encoding="utf-8")
                g.create_dataset(full_key, data=np.array(str(value), dtype=dt))

    _save_flattened(group, "", dictionary)


def load_from_h5_file(h5_file, path: str) -> dict[str, Any]:
    """Load a nested dictionary saved by ``recursively_save_to_h5_file``."""
    group = h5_file[path]
    result: dict[str, Any] = {}
    for key, dataset in group.items():
        parts = key.split(".")
        cursor = result
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = decode_from_hdf5(dataset[()])
    return result
