"""Coordinate transforms used to manipulate particle states during mutation."""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import array_api_extra as xpx
import numpy as np

from ._array_api import asarray, namespace
from .flows import NormalizingFlow, clone_flow, flow_asarray
from .utils import load_from_h5_file, recursively_save_to_h5_file, to_numpy

logger = logging.getLogger(__name__)


def _to_index(columns):
    """Normalize a column specification into a tuple of integer indices.

    Parameters
    ----------
    columns : Sequence[int] | int | None
        Either a single column index, an iterable of indices, or ``None`` to
        indicate that every column should be used.

    Returns
    -------
    tuple[int] | None
        Tuple of integer indices, or ``None`` when all columns are implied.
    """
    if columns is None:
        return None
    if hasattr(columns, "ndim"):
        arr = np.asarray(columns)
        if arr.ndim == 0:
            return (int(arr.item()),)
        return tuple(int(i) for i in arr.tolist())
    if isinstance(columns, Sequence) and not isinstance(columns, (str, bytes)):
        return tuple(int(i) for i in columns)
    return (int(columns),)


def _replace_columns(array, columns, values):
    """Return a copy of ``array`` where ``columns`` are replaced by ``values``.

    Parameters
    ----------
    array : ArrayLike
        Original data.
    columns : Sequence[int] | None
        Indices to be replaced. When ``None`` the entire array is replaced.
    values : ArrayLike
        Replacement values aligned with the selected columns.

    Returns
    -------
    ArrayLike
        Array that contains ``values`` at the requested columns.
    """
    xp = namespace(array, values)
    idx = columns
    if idx is None:
        return values
    return xpx.at(array, (..., idx)).set(values, copy=True, xp=xp)


def _log_det_entries(value, arr, xp):
    """Broadcast a scalar log-determinant to match the particle batch size.

    Parameters
    ----------
    value : ArrayLike
        Scalar log-determinant value.
    arr : ArrayLike
        Reference array whose leading dimension defines the batch size.
    xp : module
        Array API namespace associated with ``arr``.

    Returns
    -------
    ArrayLike
        A vector of log-determinant entries, one per particle.
    """
    if arr.ndim == 1 or arr.shape[0] == 1:
        return xp.asarray(value)
    return xp.full((arr.shape[0],), value, dtype=arr.dtype)


def _encode_bounds(
    bounds: Mapping[int, tuple[float, float]],
) -> dict[str, list[float]]:
    return {
        str(key): [float(lo), float(hi)] for key, (lo, hi) in bounds.items()
    }


def _decode_bounds(
    bounds: Mapping[str, Any],
) -> dict[int, tuple[float, float]]:
    return {int(key): tuple(value) for key, value in bounds.items()}


class ParticleTransform:
    """Minimal interface for coordinate transforms applied to particle states.

    Methods
    -------
    fit(particles, context=None)
        Optionally adapt internal statistics using the provided particle cloud.
    forward(particles, context=None)
        Map particles into a transformed space and return log-Jacobian values.
    inverse(values, context=None)
        Undo the transform applied by :meth:`forward`.
    """

    def clone(
        self,
    ) -> "ParticleTransform":  # pragma: no cover - overridden when needed
        # Default implementation assumes the transform is stateless.
        import copy

        return copy.deepcopy(self)

    def fit(self, particles, context=None):
        """Fit the transform to the supplied ``particles`` and return ``self``."""
        return self

    def forward(self, particles, context=None):
        """Return the identity mapping plus zero log-determinants."""
        xp = namespace(particles)
        arr = asarray(particles, xp=xp)
        zeros = xp.zeros(arr.shape[0], dtype=arr.dtype)
        return arr, zeros

    def inverse(self, values, context=None):
        """Return the identity mapping plus zero log-determinants."""
        xp = namespace(values)
        arr = asarray(values, xp=xp)
        zeros = xp.zeros(arr.shape[0], dtype=arr.dtype)
        return arr, zeros

    def config_dict(self) -> dict[str, Any]:
        """Return constructor arguments required to reconstruct the transform."""
        return {}

    def save(self, h5_file, path: str = "particle_transform") -> None:
        """Serialize the transform into an HDF5 group."""
        group = h5_file.require_group(path)
        group.attrs["class"] = self.__class__.__name__
        group.attrs["module"] = self.__class__.__module__
        recursively_save_to_h5_file(group, "config", self.config_dict())
        self._save_state(group)

    @classmethod
    def load(
        cls,
        h5_file,
        path: str = "particle_transform",
        strict: bool = False,
    ) -> "ParticleTransform":
        """Load a transform serialized with :meth:`save`."""
        group = h5_file[path]
        class_name = group.attrs["class"]
        module_name = group.attrs.get("module", __name__)

        transform_cls = cls
        if class_name != cls.__name__ or module_name != cls.__module__:
            if strict:
                raise ValueError(
                    f"Expected class {cls.__module__}.{cls.__name__}, got "
                    f"{module_name}.{class_name}."
                )
            module = importlib.import_module(module_name)
            transform_cls = getattr(module, class_name)

        return transform_cls._load_from_group(group)

    @classmethod
    def _load_from_group(cls, group) -> "ParticleTransform":
        config = (
            load_from_h5_file(group, "config") if "config" in group else {}
        )
        obj = cls(**config)
        obj._load_state(group)
        return obj

    def _save_state(self, group) -> None:
        """Hook for persisting fitted state."""

    def _load_state(self, group) -> None:
        """Hook for restoring fitted state."""


class IdentityTransform(ParticleTransform):
    """No-op transform that simply returns the input particles."""


class FlowTransform(ParticleTransform):
    """Adapt a normalizing flow to the ``ParticleTransform`` interface."""

    def __init__(self, flow: NormalizingFlow):
        self._flow = flow

    @property
    def flow(self) -> NormalizingFlow:
        return self._flow

    def clone(self) -> "FlowTransform":
        return FlowTransform(clone_flow(self._flow))

    def reset(self) -> "FlowTransform":
        reset = getattr(self._flow, "reset", None)
        if not callable(reset):
            raise AttributeError("Underlying flow does not implement reset().")
        reset()
        return self

    def fit(self, particles, context=None, reset: bool = False):
        if reset:
            self.reset()
        backend_particles = flow_asarray(self._flow, particles)
        logger.debug(
            "Fitting flow transform to %d particles.",
            backend_particles.shape[0],
        )
        self._flow.fit(backend_particles, context=context)
        return self

    def forward(self, particles, context=None):
        backend_particles = flow_asarray(self._flow, particles)
        return self._flow.forward(backend_particles, context=context)

    def inverse(self, values, context=None):
        backend_values = flow_asarray(self._flow, values)
        return self._flow.inverse(backend_values, context=context)

    def _save_state(self, group) -> None:
        self._flow.save(group, "flow")

    @classmethod
    def _load_from_group(cls, group) -> "FlowTransform":
        flow_group = group["flow"]
        module = importlib.import_module(flow_group.attrs["module"])
        flow_cls = getattr(module, flow_group.attrs["class"])
        flow = flow_cls.load(group, "flow")
        return cls(flow)


@dataclass
class AffineTransform(ParticleTransform):
    """Standardize coordinates using batch mean and variance statistics.

    Parameters
    ----------
    eps : float, default=1e-6
        Small positive value added to the variance to avoid division by zero.
    columns : Sequence[int] | None, optional
        Indices that should be standardized. When ``None`` every column is
        transformed jointly.
    """

    eps: float = 1e-6
    columns: Sequence[int] | None = None
    _mean: object | None = field(init=False, default=None)
    _scale: object | None = field(init=False, default=None)
    _column_index: Sequence[int] | None = field(init=False, default=None)

    def __post_init__(self):
        self._column_index = _to_index(self.columns)

    def fit(self, particles, context=None):
        xp = namespace(particles)
        arr = asarray(particles, xp=xp)
        idx = self._column_index
        focus = arr if idx is None else arr[..., idx]
        mean = xp.mean(focus, axis=0)
        var = xp.var(focus, axis=0)
        scale = xp.sqrt(var + self.eps)
        self._mean = mean
        self._scale = scale
        return self

    def forward(self, particles, context=None):
        xp = namespace(particles)
        arr = asarray(particles, xp=xp)
        mean = asarray(self._mean, xp=xp, like=arr)
        scale = asarray(self._scale, xp=xp, like=arr)
        if self._column_index is None:
            centered = arr - mean
            scaled = centered / scale
            log_det = -xp.sum(xp.log(scale))
            log_det_vec = _log_det_entries(log_det, arr, xp)
            return scaled, log_det_vec
        subset = arr[..., self._column_index]
        centered = subset - mean
        scaled_subset = centered / scale
        scaled = _replace_columns(arr, self._column_index, scaled_subset)
        log_det = -xp.sum(xp.log(scale))
        log_det_vec = _log_det_entries(log_det, arr, xp)
        return scaled, log_det_vec

    def inverse(self, values, context=None):
        xp = namespace(values)
        arr = asarray(values, xp=xp)
        mean = asarray(self._mean, xp=xp, like=arr)
        scale = asarray(self._scale, xp=xp, like=arr)
        if self._column_index is None:
            restored = arr * scale + mean
            log_det = xp.sum(xp.log(scale))
            log_det_vec = _log_det_entries(log_det, arr, xp)
            return restored, log_det_vec
        subset = arr[..., self._column_index]
        restored_subset = subset * scale + mean
        restored = _replace_columns(arr, self._column_index, restored_subset)
        log_det = xp.sum(xp.log(scale))
        log_det_vec = _log_det_entries(log_det, arr, xp)
        return restored, log_det_vec

    def config_dict(self) -> dict[str, Any]:
        return {
            "eps": float(self.eps),
            "columns": list(self.columns)
            if self.columns is not None
            else None,
        }

    def _save_state(self, group) -> None:
        recursively_save_to_h5_file(
            group,
            "state",
            {
                "mean": None if self._mean is None else to_numpy(self._mean),
                "scale": None
                if self._scale is None
                else to_numpy(self._scale),
            },
        )

    def _load_state(self, group) -> None:
        state = load_from_h5_file(group, "state") if "state" in group else {}
        mean = state.get("mean")
        scale = state.get("scale")
        self._mean = None if mean is None else np.asarray(mean)
        self._scale = None if scale is None else np.asarray(scale)


class BoundedToUnitHypercubeTransform(ParticleTransform):
    """Linearly rescale specified coordinates into the unit hypercube.

    Parameters
    ----------
    bounds : Mapping[int, tuple[float, float]]
        Dictionary mapping column indices to ``(lower, upper)`` bounds. Each
        lower bound must be strictly smaller than the corresponding upper bound.
    """

    def __init__(self, bounds: Mapping[int, tuple[float, float]]):
        self._bounds, self._indices = self._normalize_bounds(bounds)

    @staticmethod
    def _normalize_bounds(bounds: Mapping[int, tuple[float, float]]):
        """Validate ``bounds`` and normalize indices to integers."""
        if not bounds:
            raise ValueError("bounds must contain at least one entry.")
        normalized = {}
        indices = []
        for idx, (lo, hi) in bounds.items():
            if not hi > lo:
                raise ValueError(
                    f"Upper bound must exceed lower bound for index {idx}."
                )
            i = int(idx)
            normalized[i] = (float(lo), float(hi))
            indices.append(i)
        return normalized, tuple(indices)

    def _bounds_arrays(self, xp, dtype):
        """Return arrays of lower bounds and widths aligned with ``xp``."""
        idx = list(self._indices)
        if not idx:
            return idx, None, None, None
        lower = xp.asarray([self._bounds[i][0] for i in idx], dtype=dtype)
        upper = xp.asarray([self._bounds[i][1] for i in idx], dtype=dtype)
        width = upper - lower
        return idx, lower, upper, width

    def forward(self, particles, context=None):
        xp = namespace(particles)
        arr = asarray(particles, xp=xp)
        idx, lower, _, width = self._bounds_arrays(xp, arr.dtype)
        if not idx or lower is None or width is None:
            zeros = xp.zeros(arr.shape[0], dtype=arr.dtype)
            return arr, zeros
        subset = arr[..., idx]
        scaled_subset = (subset - lower) / width
        updated = _replace_columns(arr, idx, scaled_subset)
        log_det = -xp.sum(xp.log(width))
        log_det_vec = _log_det_entries(log_det, arr, xp)
        return updated, log_det_vec

    def inverse(self, values, context=None):
        xp = namespace(values)
        arr = asarray(values, xp=xp)
        idx, lower, _, width = self._bounds_arrays(xp, arr.dtype)
        if not idx or lower is None or width is None:
            zeros = xp.zeros(arr.shape[0], dtype=arr.dtype)
            return arr, zeros
        subset = arr[..., idx]
        restored_subset = subset * width + lower
        updated = _replace_columns(arr, idx, restored_subset)
        log_det = xp.sum(xp.log(width))
        log_det_vec = _log_det_entries(log_det, arr, xp)
        return updated, log_det_vec

    def config_dict(self) -> dict[str, Any]:
        return {"bounds": _encode_bounds(self._bounds)}

    @classmethod
    def _load_from_group(cls, group) -> "BoundedToUnitHypercubeTransform":
        config = (
            load_from_h5_file(group, "config") if "config" in group else {}
        )
        config["bounds"] = _decode_bounds(config["bounds"])
        return cls(**config)


class BoundedToUnboundedTransform(BoundedToUnitHypercubeTransform):
    """Send bounded coordinates to the real line via an independent logit.

    Parameters
    ----------
    bounds : Mapping[int, tuple[float, float]]
        Dictionary of coordinate bounds, as defined in
        :class:`BoundedToUnitHypercubeTransform`.
    eps : float, default=1e-6
        Clamp parameter that keeps values a safe distance away from 0 and 1
        before applying the logit.
    """

    def __init__(
        self, bounds: Mapping[int, tuple[float, float]], *, eps: float = 1e-6
    ):
        super().__init__(bounds)
        self._eps = eps

    def forward(self, particles, context=None):
        scaled, base_log_det = super().forward(particles, context=context)
        idx = list(self._indices)
        if not idx:
            return scaled, base_log_det
        xp = namespace(scaled, base_log_det)
        arr = asarray(scaled, xp=xp)
        subset = arr[..., idx]
        u = xp.clip(subset, self._eps, 1 - self._eps)
        transformed_subset = xp.log(u) - xp.log1p(-u)
        updated = _replace_columns(arr, idx, transformed_subset)
        jac = -xp.log(u) - xp.log1p(-u)
        base_log_det = xp.asarray(base_log_det, dtype=updated.dtype)
        log_abs_det = base_log_det + xp.sum(jac, axis=-1)
        return updated, log_abs_det

    def inverse(self, values, context=None):
        idx = list(self._indices)
        if not idx:
            return super().inverse(values, context=context)
        xp = namespace(values)
        arr = asarray(values, xp=xp)
        subset = arr[..., idx]
        u = 1.0 / (1.0 + xp.exp(-subset))
        cube_values = _replace_columns(arr, idx, u)
        restored, base_log_det = super().inverse(cube_values, context=context)
        xp = namespace(restored, base_log_det, u)
        base_log_det = xp.asarray(base_log_det, dtype=restored.dtype)
        jac = xp.sum(xp.log(u) + xp.log1p(-u), axis=-1)
        log_abs_det = base_log_det + jac
        return restored, log_abs_det

    def config_dict(self) -> dict[str, Any]:
        return {
            "bounds": _encode_bounds(self._bounds),
            "eps": float(self._eps),
        }

    @classmethod
    def _load_from_group(cls, group) -> "BoundedToUnboundedTransform":
        config = (
            load_from_h5_file(group, "config") if "config" in group else {}
        )
        config["bounds"] = _decode_bounds(config["bounds"])
        return cls(**config)


class PeriodicBoundaryTransform(ParticleTransform):
    """Wrap selected coordinates so they stay within ``[lower, upper)``.

    Parameters
    ----------
    bounds : Mapping[int, tuple[float, float]]
        Map from index to ``(lower, upper)`` bounds defining the periodic
        interval for each coordinate.
    """

    def __init__(self, bounds: Mapping[int, tuple[float, float]]):
        if not bounds:
            raise ValueError("bounds must contain at least one entry.")
        self._bounds = dict(bounds)
        self._indices = tuple(self._bounds.keys())
        for idx, (lo, hi) in self._bounds.items():
            if not hi > lo:
                raise ValueError(
                    f"Upper bound must exceed lower bound for index {idx}."
                )

    def forward(self, particles, context=None):
        xp = namespace(particles)
        arr = asarray(particles, xp=xp)
        idx = list(self._indices)
        if idx:
            lower = xp.asarray(
                [self._bounds[i][0] for i in idx], dtype=arr.dtype
            )
            upper = xp.asarray(
                [self._bounds[i][1] for i in idx], dtype=arr.dtype
            )
            width = upper - lower
            subset = arr[..., idx]
            shifted = subset - lower
            wrapped_subset = (
                lower + shifted - width * xp.floor(shifted / width)
            )
            arr = _replace_columns(arr, idx, wrapped_subset)
        zeros = xp.zeros(arr.shape[0], dtype=arr.dtype)
        return arr, zeros

    def inverse(self, values, context=None):
        return self.forward(values, context=context)

    def config_dict(self) -> dict[str, Any]:
        return {"bounds": _encode_bounds(self._bounds)}

    @classmethod
    def _load_from_group(cls, group) -> "PeriodicBoundaryTransform":
        config = (
            load_from_h5_file(group, "config") if "config" in group else {}
        )
        config["bounds"] = _decode_bounds(config["bounds"])
        return cls(**config)


class CompositeTransform(ParticleTransform):
    """Compose multiple transforms while tracking cumulative log-Jacobians.

    Parameters
    ----------
    transforms : Sequence[ParticleTransform]
        Ordered sequence of transforms to apply. The forward pass executes
        transforms from first to last, while the inverse pass applies them in
        reverse.
    """

    def __init__(self, transforms: Sequence[ParticleTransform]):
        if not transforms:
            raise ValueError(
                "CompositeTransform requires at least one transform."
            )
        self._transforms = [t.clone() for t in transforms]

    def clone(self):
        return CompositeTransform([t.clone() for t in self._transforms])

    def fit(self, particles, context=None):
        data = particles
        for transform in self._transforms:
            transform.fit(data, context=context)
            data, _ = transform.forward(data, context=context)
        return self

    def forward(self, particles, context=None):
        data = particles
        total_log_det = None
        for transform in self._transforms:
            data, log_det = transform.forward(data, context=context)
            xp = namespace(data)
            if total_log_det is None:
                total_log_det = xp.asarray(log_det, dtype=data.dtype)
            else:
                total_log_det = xp.asarray(total_log_det) + xp.asarray(log_det)
        if total_log_det is None:
            xp = namespace(data)
            total_log_det = xp.zeros(
                data.shape[0], dtype=asarray(data, xp=xp).dtype
            )
        return data, total_log_det

    def inverse(self, values, context=None):
        data = values
        total_log_det = None
        for transform in reversed(self._transforms):
            data, log_det = transform.inverse(data, context=context)
            xp = namespace(data)
            if total_log_det is None:
                total_log_det = xp.asarray(
                    log_det, dtype=asarray(data, xp=xp).dtype
                )
            else:
                total_log_det = xp.asarray(total_log_det) + xp.asarray(log_det)
        if total_log_det is None:
            xp = namespace(data)
            total_log_det = xp.zeros(
                data.shape[0], dtype=asarray(data, xp=xp).dtype
            )
        return data, total_log_det

    def _save_state(self, group) -> None:
        transforms_group = group.require_group("transforms")
        for name in list(transforms_group.keys()):
            del transforms_group[name]
        for index, transform in enumerate(self._transforms):
            transform.save(transforms_group, str(index))

    @classmethod
    def _load_from_group(cls, group) -> "CompositeTransform":
        transforms_group = group["transforms"]
        loaded = [
            ParticleTransform.load(transforms_group, str(index))
            for index in sorted(map(int, transforms_group.keys()))
        ]
        obj = cls.__new__(cls)
        obj._transforms = loaded
        return obj


_transform_registry = {
    "identity": IdentityTransform,
    "affine": AffineTransform,
    "bounded_to_unit_hypercube": BoundedToUnitHypercubeTransform,
    "bounded_to_unbounded": BoundedToUnboundedTransform,
    "periodic_boundary": PeriodicBoundaryTransform,
}


def get_prebuilt_transform(
    name: str,
    **kwargs,
) -> ParticleTransform:
    """Return a prebuilt particle transform by name.

    Parameters
    ----------
    name : str
        Name of the desired transform. Supported options are
        ``"identity"``, ``"affine"``, ``"bounded_to_unit_hypercube"``,
        ``"bounded_to_unbounded"``, and ``"periodic_boundary"``.
    **kwargs :
        Additional keyword arguments passed to the transform constructor.

    Returns
    -------
    ParticleTransform
        The requested particle transform instance.
    """
    try:
        transform_cls = _transform_registry[name.lower()]
    except KeyError as exc:
        valid = ", ".join(sorted(_transform_registry))
        raise ValueError(
            f"Unknown transform '{name}'. Valid options are: {valid}."
        ) from exc

    return transform_cls(**kwargs)
