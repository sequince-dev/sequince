"""Importance-weight calculations for persistent sampling."""

from __future__ import annotations

from typing import Any

from .._typing import Array, ArrayLike
from ..array_api import asarray, logsumexp, namespace

__all__ = [
    "persistent_log_mixture_denominator",
    "persistent_log_weights",
]


def _persistent_arrays(
    log_increment: Array,
    temperatures: ArrayLike,
    log_evidences: Array,
    *,
    xp: Any | None,
) -> tuple[Array, Array, Array, Any]:
    """Convert and validate the arrays used by persistent weighting."""
    increments = asarray(log_increment, xp=xp)
    xp = namespace(increments)
    temperatures = asarray(
        temperatures,
        xp=xp,
        dtype=increments.dtype,
        like=increments,
    )
    log_evidences = asarray(
        log_evidences,
        xp=xp,
        dtype=increments.dtype,
        like=increments,
    )

    if increments.ndim != 1:
        raise ValueError("log_increment must be one-dimensional.")
    if temperatures.ndim != 1:
        raise ValueError("temperatures must be one-dimensional.")
    if log_evidences.ndim != 1:
        raise ValueError("log_evidences must be one-dimensional.")
    if temperatures.shape[0] == 0:
        raise ValueError("temperature history must not be empty.")
    if temperatures.shape[0] != log_evidences.shape[0]:
        raise ValueError(
            "temperatures and log_evidences must have the same length."
        )

    return increments, temperatures, log_evidences, xp


def persistent_log_mixture_denominator(
    log_increment: Array,
    temperatures: ArrayLike,
    log_evidences: Array,
    *,
    xp: Any | None = None,
) -> Array:
    """Compute the log density of the equally weighted historical mixture.

    The common initial-target density is cancelled, leaving, for particle
    ``j``, ``logmeanexp(temperatures * log_increment[j] - log_evidences)``.
    """
    increments, temperatures, log_evidences, xp = _persistent_arrays(
        log_increment,
        temperatures,
        log_evidences,
        xp=xp,
    )
    terms = (
        increments[:, None] * temperatures[None, :] - log_evidences[None, :]
    )
    n_targets = asarray(
        temperatures.shape[0],
        xp=xp,
        dtype=increments.dtype,
        like=increments,
    )
    return logsumexp(terms, axis=1, xp=xp) - xp.log(n_targets)


def persistent_log_weights(
    log_increment: Array,
    temperatures: ArrayLike,
    log_evidences: Array,
    temperature: ArrayLike,
    *,
    xp: Any | None = None,
) -> Array:
    """Compute unnormalised persistent importance weights."""
    increments = asarray(log_increment, xp=xp)
    xp = namespace(increments)
    candidate_temperature = asarray(
        temperature,
        xp=xp,
        dtype=increments.dtype,
        like=increments,
    )
    if candidate_temperature.ndim != 0:
        raise ValueError("temperature must be scalar.")

    log_denominator = persistent_log_mixture_denominator(
        increments,
        temperatures,
        log_evidences,
        xp=xp,
    )
    return candidate_temperature * increments - log_denominator
