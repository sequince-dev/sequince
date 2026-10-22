from __future__ import annotations

from .array_api import asarray, logsumexp, namespace
from .state import normalize_log_weights
from .typing import Array


def ess_from_log_weights(log_weights: Array) -> Array:
    """Return the effective sample size (ESS) as an array scalar."""
    xp = namespace(log_weights)
    lw = asarray(log_weights, xp=xp)
    normalized_lw, _ = normalize_log_weights(lw, xp=xp)
    weights = xp.exp(normalized_lw)
    squared_weight_sum = xp.sum(xp.square(weights))
    safe_sum = xp.where(squared_weight_sum > 0.0, squared_weight_sum, 1.0)
    ess = 1.0 / safe_sum
    return xp.where(squared_weight_sum > 0.0, ess, xp.nan)


def ess_ratio_from_log_weights(log_weights: Array) -> Array:
    """Return ESS ratio (ESS / n) as an array scalar."""
    ess = ess_from_log_weights(log_weights)
    return ess / log_weights.shape[0]


def cess_ratio_from_log_weights_and_increments(
    log_weights: Array,
    log_increments: Array,
) -> Array:
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
    return xp.exp(log_num - log_den)
