"""Coordinate transforms used to manipulate particle states during mutation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

import array_api_extra as xpx

from .array_api import asarray, namespace, to_numpy
from .flows import (
    NormalizingFlow,
    clone_flow,
    flow_asarray,
    flow_state_dict,
    load_flow_state_dict,
)
from .utils import pytree_dataclass

logger = logging.getLogger(__name__)

Bounds = (
    Mapping[int, tuple[float, float]]
    | tuple[tuple[int, tuple[float, float]], ...]
)


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
        """Return a transform fitted to the supplied ``particles``."""
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


@pytree_dataclass(data_fields=())
@dataclass(frozen=True)
class IdentityTransform(ParticleTransform):
    """No-op transform that simply returns the input particles."""


@dataclass(frozen=True)
class FlowTransform(ParticleTransform):
    """Adapt a normalizing flow to the ``ParticleTransform`` interface."""

    flow: NormalizingFlow
    fit_kwargs: Mapping[str, Any] | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        object.__setattr__(self, "fit_kwargs", dict(self.fit_kwargs or {}))

    def clone(self) -> "FlowTransform":
        return replace(self, flow=clone_flow(self.flow))

    def state_dict(self):
        return {"flow": flow_state_dict(self.flow)}

    def load_state_dict(self, state):
        flow_state = state.get("flow", {})
        if not isinstance(flow_state, Mapping):
            raise TypeError("FlowTransform state 'flow' must be a mapping.")
        load_flow_state_dict(self.flow, flow_state)

    def fit(self, particles, context=None):
        flow = clone_flow(self.flow)
        backend_particles = flow_asarray(flow, particles)
        logger.debug(
            "Fitting flow transform to %d particles.",
            backend_particles.shape[0],
        )
        fitted_flow = flow.fit(
            backend_particles,
            context=context,
            **(self.fit_kwargs or {}),
        )
        if fitted_flow is None:
            fitted_flow = flow
        return replace(self, flow=fitted_flow)

    @staticmethod
    def _restore_namespace(values, like):
        xp = namespace(like)
        try:
            if namespace(values) is xp:
                return asarray(values, xp=xp, like=like)
        except TypeError:
            pass
        return asarray(to_numpy(values), xp=xp, like=like)

    def forward(self, particles, context=None):
        backend_particles = flow_asarray(self.flow, particles)
        values, log_det = self.flow.forward(backend_particles, context=context)
        return (
            self._restore_namespace(values, particles),
            self._restore_namespace(log_det, particles),
        )

    def inverse(self, values, context=None):
        backend_values = flow_asarray(self.flow, values)
        particles, log_det = self.flow.inverse(backend_values, context=context)
        return (
            self._restore_namespace(particles, values),
            self._restore_namespace(log_det, values),
        )


@pytree_dataclass(
    data_fields=("_mean", "_scale"),
    meta_fields=("eps", "columns"),
)
@dataclass(frozen=True)
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
    _mean: object | None = field(default=None, repr=False)
    _scale: object | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "columns", _to_index(self.columns))

    def fit(self, particles, context=None):
        xp = namespace(particles)
        arr = asarray(particles, xp=xp)
        idx = self.columns
        focus = arr if idx is None else arr[..., idx]
        mean = xp.mean(focus, axis=0)
        var = xp.var(focus, axis=0)
        scale = xp.sqrt(var + self.eps)
        return replace(self, _mean=mean, _scale=scale)

    def forward(self, particles, context=None):
        xp = namespace(particles, self._mean, self._scale)
        arr = asarray(particles, xp=xp)
        if self.columns is None:
            centered = arr - self._mean
            scaled = centered / self._scale
            log_det = -xp.sum(xp.log(self._scale))
            log_det_vec = _log_det_entries(log_det, arr, xp)
            return scaled, log_det_vec
        subset = arr[..., self.columns]
        centered = subset - self._mean
        scaled_subset = centered / self._scale
        scaled = _replace_columns(arr, self.columns, scaled_subset)
        log_det = -xp.sum(xp.log(self._scale))
        log_det_vec = _log_det_entries(log_det, arr, xp)
        return scaled, log_det_vec

    def inverse(self, values, context=None):
        xp = namespace(values, self._mean, self._scale)
        arr = asarray(values, xp=xp)
        if self.columns is None:
            restored = arr * self._scale + self._mean
            log_det = xp.sum(xp.log(self._scale))
            log_det_vec = _log_det_entries(log_det, arr, xp)
            return restored, log_det_vec
        subset = arr[..., self.columns]
        restored_subset = subset * self._scale + self._mean
        restored = _replace_columns(arr, self.columns, restored_subset)
        log_det = xp.sum(xp.log(self._scale))
        log_det_vec = _log_det_entries(log_det, arr, xp)
        return restored, log_det_vec


@pytree_dataclass(data_fields=(), meta_fields=("bounds",))
@dataclass(frozen=True)
class BoundedToUnitHypercubeTransform(ParticleTransform):
    """Linearly rescale specified coordinates into the unit hypercube.

    Parameters
    ----------
    bounds : Mapping[int, tuple[float, float]]
        Dictionary mapping column indices to ``(lower, upper)`` bounds. Each
        lower bound must be strictly smaller than the corresponding upper bound.
    """

    bounds: Bounds

    def __post_init__(self) -> None:
        normalized, _ = self._normalize_bounds(self.bounds)
        object.__setattr__(self, "bounds", tuple(normalized.items()))

    @property
    def _bounds(self) -> dict[int, tuple[float, float]]:
        return dict(self.bounds)

    @property
    def _indices(self) -> tuple[int, ...]:
        return tuple(self._bounds)

    @staticmethod
    def _normalize_bounds(bounds: Bounds):
        """Validate ``bounds`` and normalize indices to integers."""
        if not bounds:
            raise ValueError("bounds must contain at least one entry.")
        normalized = {}
        indices = []
        items = bounds.items() if isinstance(bounds, Mapping) else bounds
        for idx, (lo, hi) in items:
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


@pytree_dataclass(data_fields=(), meta_fields=("bounds", "eps"))
@dataclass(frozen=True)
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

    eps: float = field(default=1e-6, kw_only=True)

    def forward(self, particles, context=None):
        scaled, base_log_det = super().forward(particles, context=context)
        idx = list(self._indices)
        if not idx:
            return scaled, base_log_det
        xp = namespace(scaled, base_log_det)
        arr = asarray(scaled, xp=xp)
        subset = arr[..., idx]
        u = xp.clip(subset, self.eps, 1 - self.eps)
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


@pytree_dataclass(data_fields=(), meta_fields=("bounds",))
@dataclass(frozen=True)
class PeriodicBoundaryTransform(ParticleTransform):
    """Wrap selected coordinates so they stay within ``[lower, upper)``.

    Parameters
    ----------
    bounds : Mapping[int, tuple[float, float]]
        Map from index to ``(lower, upper)`` bounds defining the periodic
        interval for each coordinate.
    """

    bounds: Bounds

    def __post_init__(self) -> None:
        if not self.bounds:
            raise ValueError("bounds must contain at least one entry.")
        normalized = []
        items = (
            self.bounds.items()
            if isinstance(self.bounds, Mapping)
            else self.bounds
        )
        for idx, (lo, hi) in items:
            if not hi > lo:
                raise ValueError(
                    f"Upper bound must exceed lower bound for index {idx}."
                )
            normalized.append((int(idx), (float(lo), float(hi))))
        object.__setattr__(self, "bounds", tuple(normalized))

    @property
    def _bounds(self) -> dict[int, tuple[float, float]]:
        return dict(self.bounds)

    @property
    def _indices(self) -> tuple[int, ...]:
        return tuple(self._bounds)

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


@pytree_dataclass(data_fields=("transforms",))
@dataclass(frozen=True)
class CompositeTransform(ParticleTransform):
    """Compose multiple transforms while tracking cumulative log-Jacobians.

    Parameters
    ----------
    transforms : Sequence[ParticleTransform]
        Ordered sequence of transforms to apply. The forward pass executes
        transforms from first to last, while the inverse pass applies them in
        reverse.
    """

    transforms: Sequence[ParticleTransform]

    def __post_init__(self) -> None:
        if not self.transforms:
            raise ValueError(
                "CompositeTransform requires at least one transform."
            )
        object.__setattr__(self, "transforms", tuple(self.transforms))

    def clone(self):
        return CompositeTransform([t.clone() for t in self.transforms])

    def fit(self, particles, context=None):
        data = particles
        fitted_transforms = []
        for transform in self.transforms:
            fitted = transform.fit(data, context=context)
            fitted_transforms.append(fitted)
            data, _ = fitted.forward(data, context=context)
        return replace(self, transforms=tuple(fitted_transforms))

    def forward(self, particles, context=None):
        data = particles
        total_log_det = None
        for transform in self.transforms:
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
        for transform in reversed(self.transforms):
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
