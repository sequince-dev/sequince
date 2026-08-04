from __future__ import annotations

import math

from ._typing import Array
from .array_api import asarray, full, logsumexp, namespace, to_numpy


def ess_from_log_weights(log_weights: Array) -> float:
    """Return the effective sample size (ESS) given log weights."""
    xp = namespace(log_weights)
    lw = asarray(log_weights, xp=xp)
    log_norm = logsumexp(lw, xp=xp)
    log_norm_scalar = float(log_norm)
    if math.isfinite(log_norm_scalar):
        weights = xp.exp(lw - log_norm_scalar)
    else:
        n = lw.shape[0]
        weights = full((n,), 1.0 / n, xp=xp, dtype=lw.dtype, like=lw)
    ess = 1.0 / float(to_numpy(xp.sum(xp.square(weights))))
    return ess


def ess_ratio_from_log_weights(log_weights: Array) -> float:
    """Return ESS ratio (ESS / n) given log weights."""
    ess = ess_from_log_weights(log_weights)
    return ess / log_weights.shape[0]


def cess_ratio_from_log_weights_and_increments(
    log_weights: Array,
    log_increments: Array,
) -> float:
    """Return the conditional ESS ratio induced by log increments.

    The quantity computed is

        (sum_i W_i g_i)^2 / sum_i W_i (g_i)^2

    where W_i are the normalized current weights and g_i are the proposed
    incremental weights.

    See e.g. Eq 13 in arXiv:1612.06468 for more details.
    """
    xp = namespace(log_weights, log_increments)
    lw = asarray(log_weights, xp=xp)
    li = asarray(log_increments, xp=xp, like=lw)

    log_norm = logsumexp(lw, xp=xp)
    normalized_lw = lw - log_norm

    log_num = 2.0 * logsumexp(normalized_lw + li, xp=xp)
    log_den = logsumexp(normalized_lw + 2.0 * li, xp=xp)
    return float(to_numpy(xp.exp(log_num - log_den)))
