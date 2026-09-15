"""Coordinate transforms used to manipulate particle states during mutation."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import Any, Literal, Mapping, Protocol, Sequence, cast

import array_api_extra as xpx
import optree
from array_api_compat import is_array_api_obj
from scipy.special import ndtr, ndtri
from scipy.stats import circmean

from ._typing import Array
from .array_api import asarray, namespace, to_numpy
from .flows import (
    NormalizingFlow,
    clone_flow,
    flow_asarray,
    flow_state_dict,
    load_flow_state_dict,
)
from .utils.pytrees import OPTREE_NAMESPACE, pytree_dataclass
from .utils.random import RNGLike, resolve_rng

logger = logging.getLogger(__name__)

Bounds = (
    Mapping[int, tuple[float, float]]
    | tuple[tuple[int, tuple[float, float]], ...]
)


def _map_auxiliary_pytree(value: Any, fn):
    """Map array leaves in an Optree pytree."""
    return optree.tree_map(
        lambda leaf: fn(leaf) if is_array_api_obj(leaf) else leaf,
        value,
        namespace=OPTREE_NAMESPACE,
    )


def select_transform_auxiliary(
    auxiliary: Any,
    selection: Any,
    *,
    n_particles: int,
) -> Any:
    """Apply a particle selection to particle-aligned auxiliary array leaves.

    Every array leaf satisfying ``leaf.ndim > 0`` and
    ``leaf.shape[0] == n_particles`` is particle-aligned and receives
    ``selection``. Other leaves are returned unchanged; this shape convention
    assigns no additional semantics to them.
    """

    def select_leaf(leaf):
        if leaf.ndim > 0 and leaf.shape[0] == n_particles:
            return leaf[selection]
        return leaf

    return _map_auxiliary_pytree(auxiliary, select_leaf)


def _repeat_transform_auxiliary(
    auxiliary: Any,
    repeats: int,
    *,
    n_particles: int,
) -> Any:
    """Repeat particle-aligned leaves for a flattened chain batch."""

    def repeat_leaf(leaf):
        if leaf.ndim == 0 or leaf.shape[0] != n_particles:
            return leaf
        xp = namespace(leaf)
        expanded = xp.broadcast_to(leaf, (repeats, *leaf.shape))
        return xp.reshape(
            expanded,
            (repeats * n_particles, *leaf.shape[1:]),
        )

    return _map_auxiliary_pytree(auxiliary, repeat_leaf)


def has_particle_aligned_auxiliary(
    auxiliary: Any,
    *,
    n_particles: int,
) -> bool:
    """Return whether an auxiliary pytree contains particle-aligned arrays."""
    leaves = optree.tree_leaves(
        auxiliary,
        namespace=OPTREE_NAMESPACE,
    )
    return any(
        is_array_api_obj(leaf)
        and leaf.ndim > 0
        and leaf.shape[0] == n_particles
        for leaf in leaves
    )


class UnitIntervalMapping(Protocol):
    """Mapping between the open unit interval and the real line.

    Implementations return the elementwise log absolute determinant alongside
    transformed values. They must preserve the Array API namespace of their
    input.
    """

    def forward(self, values):
        """Map values in ``(0, 1)`` to the real line."""

    def inverse(self, values):
        """Map values on the real line back to ``(0, 1)``."""


def _validate_mapping_eps(eps: float) -> None:
    if not 0.0 < eps < 0.5:
        raise ValueError("eps must lie strictly between 0 and 0.5.")


@pytree_dataclass(data_fields=(), meta_fields=("eps",))
@dataclass(frozen=True)
class LogitMapping:
    """Map the unit interval to the real line with a stabilized logit.

    Parameters
    ----------
    eps : float, default=1e-6
        Clamp values to ``[eps, 1 - eps]`` before applying the logit.
    """

    eps: float = 1e-6

    def __post_init__(self) -> None:
        _validate_mapping_eps(self.eps)

    def forward(self, values):
        xp = namespace(values)
        arr = asarray(values, xp=xp)
        u = xp.clip(arr, self.eps, 1 - self.eps)
        transformed = xp.log(u) - xp.log1p(-u)
        log_abs_det = -xp.log(u) - xp.log1p(-u)
        return transformed, log_abs_det

    def inverse(self, values):
        xp = namespace(values)
        arr = asarray(values, xp=xp)
        limit = math.log((1.0 - self.eps) / self.eps)
        z = xp.clip(arr, -limit, limit)

        zero = xp.zeros_like(z)
        log_term = xp.log1p(xp.exp(-xp.abs(z)))

        softplus_negative_z = xp.maximum(-z, zero) + log_term
        restored = xp.exp(-softplus_negative_z)

        softplus_z = xp.maximum(z, zero) + log_term
        log_abs_det = -softplus_negative_z - softplus_z

        return restored, log_abs_det


@pytree_dataclass(data_fields=(), meta_fields=("eps",))
@dataclass(frozen=True)
class CauchitMapping:
    """Map the unit interval to the real line with a Cauchy quantile.

    Parameters
    ----------
    eps : float, default=1e-6
        Clamp values to ``[eps, 1 - eps]`` before applying the quantile.
    """

    eps: float = 1e-6

    def __post_init__(self) -> None:
        _validate_mapping_eps(self.eps)

    def forward(self, values):
        xp = namespace(values)
        arr = asarray(values, xp=xp)
        u = xp.clip(arr, self.eps, 1 - self.eps)
        pi = xp.asarray(math.pi, dtype=u.dtype)
        angle = pi * (u - 0.5)
        transformed = xp.tan(angle)
        log_abs_det = xp.log(pi) - 2.0 * xp.log(xp.abs(xp.cos(angle)))
        return transformed, log_abs_det

    def inverse(self, values):
        xp = namespace(values)
        arr = asarray(values, xp=xp)
        limit = math.tan(math.pi * (0.5 - self.eps))
        z = xp.clip(arr, -limit, limit)
        pi = xp.asarray(math.pi, dtype=z.dtype)
        restored = xp.atan(z) / pi + 0.5
        log_abs_det = -xp.log(pi) - xp.log1p(z * z)
        return restored, log_abs_det


@pytree_dataclass(data_fields=(), meta_fields=("eps",))
@dataclass(frozen=True)
class ProbitMapping:
    """Map the unit interval to the real line with a normal quantile.

    Parameters
    ----------
    eps : float, default=1e-6
        Clamp values to ``[eps, 1 - eps]`` before applying the quantile.
    """

    eps: float = 1e-6

    def __post_init__(self) -> None:
        _validate_mapping_eps(self.eps)

    def forward(self, values):
        xp = namespace(values)
        arr = asarray(values, xp=xp)
        u = xp.clip(arr, self.eps, 1 - self.eps)
        transformed = ndtri(u)
        log_normalization = 0.5 * math.log(2.0 * math.pi)
        log_abs_det = 0.5 * transformed * transformed + log_normalization
        return transformed, log_abs_det

    def inverse(self, values):
        xp = namespace(values)
        arr = asarray(values, xp=xp)
        limit = float(ndtri(1.0 - self.eps))
        z = xp.clip(arr, -limit, limit)
        restored = ndtr(z)
        log_normalization = 0.5 * math.log(2.0 * math.pi)
        log_abs_det = -0.5 * z * z - log_normalization
        return restored, log_abs_det


_unit_interval_mapping_registry: dict[str, type[UnitIntervalMapping]] = {
    "logit": LogitMapping,
    "cauchit": CauchitMapping,
    "probit": ProbitMapping,
}


def get_unit_interval_mapping(
    mapping: str | UnitIntervalMapping,
) -> UnitIntervalMapping:
    """Resolve a built-in mapping name or validate a mapping strategy object."""
    if isinstance(mapping, str):
        try:
            mapping_cls = _unit_interval_mapping_registry[mapping.lower()]
        except KeyError as exc:
            valid = ", ".join(sorted(_unit_interval_mapping_registry))
            raise ValueError(
                f"Unknown unit-interval mapping '{mapping}'. "
                f"Valid options are: {valid}."
            ) from exc
        return mapping_cls()

    if not callable(getattr(mapping, "forward", None)) or not callable(
        getattr(mapping, "inverse", None)
    ):
        raise TypeError(
            "mapping must be a registered name or an object providing "
            "forward() and inverse()."
        )
    return cast(UnitIntervalMapping, mapping)


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
        return asarray(value, xp=xp, like=arr)
    # Broadcast on the particle device without extracting a host scalar.
    value = asarray(value, xp=xp, dtype=arr.dtype, like=arr)
    return xp.broadcast_to(value, (arr.shape[0],))


class ParticleTransform:
    """Coordinate transform interface for particle mutation.

    Methods
    -------
    fit(particles, context=None, *, rng=None)
        Optionally adapt internal statistics using the provided particle cloud.
    forward(particles, context=None, *, rng=None)
        Return transformed particles, log-Jacobian values, and auxiliary data.
    inverse(values, context=None, *, auxiliary=None, rng=None)
        Undo the transform applied by :meth:`forward`.
    """

    def clone(
        self,
    ) -> "ParticleTransform":  # pragma: no cover - overridden when needed
        # Default implementation assumes the transform is stateless.
        import copy

        return copy.deepcopy(self)

    def fit(self, particles, context=None, *, rng: RNGLike = None):
        """Return a transform fitted to the supplied ``particles``."""
        del rng
        return self

    def forward(self, particles, context=None, *, rng: RNGLike = None):
        """Return the identity mapping, zero log-determinants, and no auxiliary."""
        del rng
        xp = namespace(particles)
        arr = asarray(particles, xp=xp)
        zeros = xp.zeros_like(arr[..., 0])
        return arr, zeros, None

    def inverse(
        self,
        values,
        context=None,
        *,
        auxiliary=None,
        rng: RNGLike = None,
    ):
        """Return the identity mapping plus zero log-determinants."""
        del auxiliary, rng
        xp = namespace(values)
        arr = asarray(values, xp=xp)
        zeros = xp.zeros_like(arr[..., 0])
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
        fit_kwargs = dict(self.fit_kwargs or {})
        reserved = {"context", "rng"}.intersection(fit_kwargs)
        if reserved:
            names = ", ".join(sorted(reserved))
            raise ValueError(
                f"fit_kwargs must not override transform arguments: {names}."
            )
        object.__setattr__(self, "fit_kwargs", fit_kwargs)

    def clone(self) -> "FlowTransform":
        return replace(self, flow=clone_flow(self.flow))

    def state_dict(self):
        return {"flow": flow_state_dict(self.flow)}

    def load_state_dict(self, state):
        flow_state = state.get("flow", {})
        if not isinstance(flow_state, Mapping):
            raise TypeError("FlowTransform state 'flow' must be a mapping.")
        load_flow_state_dict(self.flow, flow_state)

    def fit(self, particles, context=None, *, rng: RNGLike = None):
        flow = clone_flow(self.flow)
        backend_particles = flow_asarray(flow, particles)
        logger.debug(
            "Fitting flow transform to %d particles.",
            backend_particles.shape[0],
        )
        fitted_flow = flow.fit(
            backend_particles,
            context=context,
            rng=rng,
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

    def forward(self, particles, context=None, *, rng: RNGLike = None):
        del rng
        backend_particles = flow_asarray(self.flow, particles)
        values, log_det = self.flow.forward(backend_particles, context=context)
        return (
            self._restore_namespace(values, particles),
            self._restore_namespace(log_det, particles),
            None,
        )

    def inverse(
        self, values, context=None, *, auxiliary=None, rng: RNGLike = None
    ):
        del auxiliary, rng
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

    def fit(self, particles, context=None, *, rng: RNGLike = None):
        del rng
        xp = namespace(particles)
        arr = asarray(particles, xp=xp)
        idx = self.columns
        focus = arr if idx is None else arr[..., idx]
        mean = xp.mean(focus, axis=0)
        var = xp.var(focus, axis=0)
        scale = xp.sqrt(var + self.eps)
        return replace(self, _mean=mean, _scale=scale)

    def forward(self, particles, context=None, *, rng: RNGLike = None):
        del rng
        xp = namespace(particles, self._mean, self._scale)
        arr = asarray(particles, xp=xp)
        if self.columns is None:
            centered = arr - self._mean
            scaled = centered / self._scale
            log_det = -xp.sum(xp.log(self._scale))
            log_det_vec = _log_det_entries(log_det, arr, xp)
            return scaled, log_det_vec, None
        subset = arr[..., self.columns]
        centered = subset - self._mean
        scaled_subset = centered / self._scale
        scaled = _replace_columns(arr, self.columns, scaled_subset)
        log_det = -xp.sum(xp.log(self._scale))
        log_det_vec = _log_det_entries(log_det, arr, xp)
        return scaled, log_det_vec, None

    def inverse(
        self, values, context=None, *, auxiliary=None, rng: RNGLike = None
    ):
        del auxiliary, rng
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


@pytree_dataclass(
    data_fields=("_mean", "_covariance", "_cholesky"),
    meta_fields=("eps",),
)
@dataclass(frozen=True)
class WhiteningTransform(ParticleTransform):
    """Whiten a complete Euclidean particle vector with an affine transform.

    The transform is fitted to a particle population and maps its coordinates
    according to ``z = L**-1 (x - mean)``, where ``L L.T`` is a regularized
    estimate of the population covariance.  It is intended to be composed after
    any problem-specific transforms which unwrap periodic coordinates or
    otherwise make the representation suitable for Euclidean linear algebra.

    Parameters
    ----------
    eps : float, default=1e-6
        Relative diagonal regularization. Each variance receives ``eps`` times
        its own scale. Exactly constant directions receive a smaller floor tied
        to the largest variance and the precision of the array dtype. Values
        below machine precision are promoted to machine precision.
    """

    eps: float = 1e-6
    _mean: object | None = field(default=None, repr=False)
    _covariance: object | None = field(default=None, repr=False)
    _cholesky: object | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not math.isfinite(self.eps) or self.eps <= 0.0:
            raise ValueError("eps must be finite and strictly positive.")

    def fit(self, particles, context=None, *, rng: RNGLike = None):
        """Return a transform fitted to the unweighted particle population."""
        del rng
        xp = namespace(particles)
        arr = asarray(particles, xp=xp)
        if arr.ndim != 2:
            raise ValueError(
                "WhiteningTransform.fit requires a 2D particle array."
            )

        mean = xp.mean(arr, axis=0)
        centered = arr - mean
        covariance = xp.matmul(xp.permute_dims(centered, (1, 0)), centered)
        covariance = covariance / xp.asarray(arr.shape[0], dtype=arr.dtype)
        covariance = 0.5 * (covariance + xp.permute_dims(covariance, (1, 0)))

        variances = xp.linalg.diagonal(covariance)
        largest_variance = xp.max(variances)
        unit = xp.asarray(1.0, dtype=arr.dtype)
        covariance_scale = xp.where(
            largest_variance > 0, largest_variance, unit
        )
        dtype_eps = xp.asarray(xp.finfo(arr.dtype).eps, dtype=arr.dtype)
        variance_floor = dtype_eps * covariance_scale
        relative_eps = xp.maximum(
            xp.asarray(self.eps, dtype=arr.dtype), dtype_eps
        )
        regularizer = relative_eps * xp.maximum(variances, variance_floor)
        covariance = covariance + regularizer * xp.eye(
            arr.shape[1], dtype=arr.dtype
        )
        cholesky = xp.linalg.cholesky(covariance)
        return replace(
            self,
            _mean=mean,
            _covariance=covariance,
            _cholesky=cholesky,
        )

    def _require_fitted(self) -> tuple[object, object]:
        if self._mean is None or self._cholesky is None:
            raise RuntimeError(
                "WhiteningTransform must be fitted before it can be applied."
            )
        return self._mean, self._cholesky

    def forward(self, particles, context=None, *, rng: RNGLike = None):
        del rng
        mean, cholesky = self._require_fitted()
        xp = namespace(particles, mean, cholesky)
        arr = asarray(particles, xp=xp)
        centered = arr - mean
        flat = xp.reshape(centered, (-1, centered.shape[-1]))
        latent_t = xp.linalg.solve(cholesky, xp.permute_dims(flat, (1, 0)))
        latent = xp.reshape(xp.permute_dims(latent_t, (1, 0)), arr.shape)
        log_det = -xp.sum(xp.log(xp.linalg.diagonal(cholesky)))
        return latent, _log_det_entries(log_det, arr, xp), None

    def inverse(
        self, values, context=None, *, auxiliary=None, rng: RNGLike = None
    ):
        del auxiliary, rng
        mean, cholesky = self._require_fitted()
        xp = namespace(values, mean, cholesky)
        arr = asarray(values, xp=xp)
        flat = xp.reshape(arr, (-1, arr.shape[-1]))
        restored = xp.matmul(flat, xp.permute_dims(cholesky, (1, 0))) + mean
        restored = xp.reshape(restored, arr.shape)
        log_det = xp.sum(xp.log(xp.linalg.diagonal(cholesky)))
        return restored, _log_det_entries(log_det, arr, xp)


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

    def forward(self, particles, context=None, *, rng: RNGLike = None):
        del rng
        xp = namespace(particles)
        arr = asarray(particles, xp=xp)
        idx, lower, _, width = self._bounds_arrays(xp, arr.dtype)
        if not idx or lower is None or width is None:
            zeros = xp.zeros(arr.shape[0], dtype=arr.dtype)
            return arr, zeros, None
        subset = arr[..., idx]
        scaled_subset = (subset - lower) / width
        updated = _replace_columns(arr, idx, scaled_subset)
        log_det = -xp.sum(xp.log(width))
        log_det_vec = _log_det_entries(log_det, arr, xp)
        return updated, log_det_vec, None

    def inverse(
        self, values, context=None, *, auxiliary=None, rng: RNGLike = None
    ):
        del auxiliary, rng
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


@pytree_dataclass(data_fields=(), meta_fields=("bounds", "mapping"))
@dataclass(frozen=True)
class BoundedToUnboundedTransform(BoundedToUnitHypercubeTransform):
    """Send bounded coordinates to the real line independently.

    Parameters
    ----------
    bounds : Mapping[int, tuple[float, float]]
        Dictionary of coordinate bounds, as defined in
        :class:`BoundedToUnitHypercubeTransform`.
    mapping : str | UnitIntervalMapping, default="logit"
        Built-in mapping name or an object implementing ``forward`` and
        ``inverse``. Built-in options are ``"logit"``, ``"cauchit"``, and
        ``"probit"``.
    """

    mapping: str | UnitIntervalMapping = field(default="logit", kw_only=True)

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(
            self, "mapping", get_unit_interval_mapping(self.mapping)
        )

    def forward(self, particles, context=None, *, rng: RNGLike = None):
        scaled, base_log_det, _ = super().forward(
            particles, context=context, rng=rng
        )
        idx = list(self._indices)
        if not idx:
            return scaled, base_log_det, None
        xp = namespace(scaled, base_log_det)
        arr = asarray(scaled, xp=xp)
        mapping = cast(UnitIntervalMapping, self.mapping)
        transformed_subset, jac = mapping.forward(arr[..., idx])
        updated = _replace_columns(arr, idx, transformed_subset)
        base_log_det = xp.asarray(base_log_det, dtype=updated.dtype)
        log_abs_det = base_log_det + xp.sum(jac, axis=-1)
        return updated, log_abs_det, None

    def inverse(
        self, values, context=None, *, auxiliary=None, rng: RNGLike = None
    ):
        idx = list(self._indices)
        if not idx:
            return super().inverse(
                values, context=context, auxiliary=auxiliary, rng=rng
            )
        xp = namespace(values)
        arr = asarray(values, xp=xp)
        mapping = cast(UnitIntervalMapping, self.mapping)
        u, jac = mapping.inverse(arr[..., idx])
        cube_values = _replace_columns(arr, idx, u)
        restored, base_log_det = super().inverse(
            cube_values, context=context, auxiliary=auxiliary, rng=rng
        )
        xp = namespace(restored, base_log_det, u)
        base_log_det = xp.asarray(base_log_det, dtype=restored.dtype)
        log_abs_det = base_log_det + xp.sum(jac, axis=-1)
        return restored, log_abs_det


@pytree_dataclass(
    data_fields=("_center",), meta_fields=("bounds", "centering")
)
@dataclass(frozen=True)
class PeriodicBoundaryTransform(ParticleTransform):
    """Wrap selected periodic coordinates, optionally centering their mean.

    Parameters
    ----------
    bounds : Mapping[int, tuple[float, float]]
        Map from index to ``(lower, upper)`` bounds defining the periodic
        interval for each coordinate.
    centering : {None, "circular-mean", "angular-mean"}, optional
        With ``None`` (default), only wrap into the configured interval.
        With ``"circular-mean"``, first call :meth:`fit` to estimate the
        unweighted circular mean of each selected coordinate. Forward values
        are offsets from that mean in ``[-width / 2, width / 2)``; inverse
        values are restored to ``[lower, upper)``. Uses
        :func:`scipy.stats.circmean`; when the mean resultant vector is zero,
        the center is implementation-defined, as in SciPy.
        ``"angular-mean"`` is an alias for ``"circular-mean"``. Underscore
        spellings (``"circular_mean"`` and ``"angular_mean"``) are also accepted.
    """

    bounds: Bounds
    centering: (
        Literal[
            "circular-mean", "angular-mean", "circular_mean", "angular_mean"
        ]
        | None
    ) = field(default=None, kw_only=True)
    _center: Array | None = field(default=None, repr=False, kw_only=True)

    def __post_init__(self) -> None:
        if self.centering in ("angular-mean", "angular_mean", "circular_mean"):
            object.__setattr__(self, "centering", "circular-mean")
        if self.centering not in (None, "circular-mean"):
            raise ValueError(
                "centering must be None, 'circular-mean', or 'angular-mean'."
            )
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

    @cached_property
    def _bounds(self) -> dict[int, tuple[float, float]]:
        return dict(self.bounds)

    @cached_property
    def _indices(self) -> tuple[int, ...]:
        return tuple(self._bounds)

    @cached_property
    def _lower(self) -> tuple[float, ...]:
        return tuple(lo for lo, _ in self._bounds.values())

    @cached_property
    def _upper(self) -> tuple[float, ...]:
        return tuple(hi for _, hi in self._bounds.values())

    def fit(
        self, particles: Array, context: Any = None, *, rng: RNGLike = None
    ) -> PeriodicBoundaryTransform:
        """Return a fitted copy using a cloud of shape (n_particles, n_dims).

        Circular-mean fitting requires a nonempty floating-point cloud and
        uses the configured interval width as the period.
        """
        del context, rng
        if self.centering is None:
            return self
        xp = namespace(particles)
        arr = asarray(particles, xp=xp)
        if arr.ndim != 2 or arr.shape[0] == 0:
            raise ValueError("fit requires a nonempty 2D particle cloud.")
        lower = asarray(self._lower, like=arr, dtype=arr.dtype)
        upper = asarray(self._upper, like=arr, dtype=arr.dtype)
        width = upper - lower
        unit_values = (arr[..., list(self._indices)] - lower) / width
        mean = circmean(unit_values, low=0.0, high=1.0, axis=0)
        center = lower + width * asarray(mean, like=arr, dtype=arr.dtype)
        return replace(self, _center=center)

    def forward(self, particles, context=None, *, rng: RNGLike = None):
        del rng
        xp = namespace(particles)
        arr = asarray(particles, xp=xp)
        idx = list(self._indices)
        if idx:
            lower = asarray(self._lower, like=arr, dtype=arr.dtype)
            upper = asarray(self._upper, like=arr, dtype=arr.dtype)
            width = upper - lower
            subset = arr[..., idx]
            shifted = subset - lower
            if self.centering is not None:
                if self._center is None:
                    raise ValueError("Call fit before using circular-mean.")
                shifted = subset - asarray(self._center, like=arr) + width / 2
                lower = -width / 2
            wrapped_subset = (
                lower + shifted - width * xp.floor(shifted / width)
            )
            arr = _replace_columns(arr, idx, wrapped_subset)
        zeros = xp.zeros_like(arr[..., 0])
        return arr, zeros, None

    def inverse(
        self, values, context=None, *, auxiliary=None, rng: RNGLike = None
    ):
        del auxiliary
        if self.centering is not None:
            if self._center is None:
                raise ValueError("Call fit before using circular-mean.")
            xp = namespace(values)
            arr = asarray(values, xp=xp)
            idx = list(self._indices)
            values = _replace_columns(
                arr, idx, arr[..., idx] + asarray(self._center, like=arr)
            )
        xp = namespace(values)
        arr = asarray(values, xp=xp)
        idx = list(self._indices)
        lower = asarray(self._lower, like=arr, dtype=arr.dtype)
        upper = asarray(self._upper, like=arr, dtype=arr.dtype)
        width = upper - lower
        shifted = arr[..., idx] - lower
        wrapped = lower + shifted - width * xp.floor(shifted / width)
        restored = _replace_columns(arr, idx, wrapped)
        return restored, xp.zeros_like(arr[..., 0])


@pytree_dataclass(data_fields=("transforms",))
@dataclass(frozen=True)
class CompositeTransform(ParticleTransform):
    """Compose multiple transforms while tracking cumulative log-Jacobians.

    Forward auxiliary data is a tuple with exactly one entry per component,
    including ``None`` entries.  Inversion validates this shape and supplies
    entries to their components in reverse transform order.  A supplied RNG is
    resolved once and advanced sequentially across components, which avoids
    reusing functional random keys.

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

    def fit(self, particles, context=None, *, rng: RNGLike = None):
        data = particles
        component_rng = (
            None if rng is None else resolve_rng(rng, xp=namespace(particles))
        )
        fitted_transforms = []
        for transform in self.transforms:
            fitted = transform.fit(data, context=context, rng=component_rng)
            fitted_transforms.append(fitted)
            data, _, _ = fitted.forward(
                data, context=context, rng=component_rng
            )
        return replace(self, transforms=tuple(fitted_transforms))

    def forward(self, particles, context=None, *, rng: RNGLike = None):
        data = particles
        component_rng = (
            None if rng is None else resolve_rng(rng, xp=namespace(particles))
        )
        total_log_det = None
        auxiliary = []
        for transform in self.transforms:
            data, log_det, transform_auxiliary = transform.forward(
                data, context=context, rng=component_rng
            )
            auxiliary.append(transform_auxiliary)
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
        return data, total_log_det, tuple(auxiliary)

    def inverse(
        self,
        values,
        context=None,
        *,
        auxiliary=None,
        rng: RNGLike = None,
    ):
        component_rng = (
            None if rng is None else resolve_rng(rng, xp=namespace(values))
        )
        component_auxiliary: Sequence[Any]
        if auxiliary is None:
            component_auxiliary = (None,) * len(self.transforms)
        elif not isinstance(auxiliary, (tuple, list)):
            raise TypeError(
                "CompositeTransform auxiliary must be a tuple or list with "
                "one entry per component transform."
            )
        elif len(auxiliary) != len(self.transforms):
            raise ValueError(
                "CompositeTransform auxiliary has "
                f"{len(auxiliary)} entries; expected {len(self.transforms)}."
            )
        else:
            component_auxiliary = auxiliary
        data = values
        total_log_det = None
        for transform, transform_auxiliary in zip(
            reversed(self.transforms),
            reversed(component_auxiliary),
            strict=True,
        ):
            data, log_det = transform.inverse(
                data,
                context=context,
                auxiliary=transform_auxiliary,
                rng=component_rng,
            )
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
    "whitening": WhiteningTransform,
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
        ``"identity"``, ``"affine"``, ``"whitening"``,
        ``"bounded_to_unit_hypercube"``, ``"bounded_to_unbounded"``, and
        ``"periodic_boundary"``.
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
