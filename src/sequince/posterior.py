from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np

from .utils import _call_with_optional_context

if TYPE_CHECKING:
    from .steps.base import StepContext

ArrayLikeFn = Callable[..., np.ndarray]


def _evaluate(
    fn: ArrayLikeFn,
    particles: np.ndarray,
    context: "StepContext",
) -> np.ndarray:
    """Call function with optional context and return a 1d float array."""
    values = _call_with_optional_context(fn, (particles,), context)
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        array = np.full(particles.shape[0], float(array), dtype=float)
    return np.atleast_1d(array).astype(float, copy=False)


@dataclass(frozen=True)
class TemperedPosterior:
    """Wrapper exposing posterior building blocks for SMC workflows.

    Parameters
    ----------
    log_prior:
        Callable returning ``log p(x)`` for each particle.
    log_likelihood:
        Callable returning ``log l(x)`` for each particle.
    log_proposal:
        Optional callable returning ``log q(x)``. When supplied the intermediate
        targets follow the bridge ``q(x)^{1-β} [p(x)L(x)]^{β}``.
    """

    log_prior: ArrayLikeFn
    log_likelihood: ArrayLikeFn
    log_proposal: Optional[ArrayLikeFn] = None

    def log_increment(
        self,
        particles: np.ndarray,
        context: "StepContext",
    ) -> np.ndarray:
        """Return the log-weight increment required by the tempering step."""
        log_like = _evaluate(self.log_likelihood, particles, context)
        if self.log_proposal is None:
            if particles.ndim == 1:
                return float(log_like[0])
            return log_like
        log_prior = _evaluate(self.log_prior, particles, context)
        log_prop = _evaluate(self.log_proposal, particles, context)
        bridge = log_prior + log_like - log_prop
        if particles.ndim == 1:
            return float(bridge[0])
        return bridge

    def log_prob(
        self,
        particles: np.ndarray,
        context: "StepContext",
    ) -> np.ndarray:
        """Return the tempered log posterior evaluated at the current β."""
        beta = float(context.shared.get("temperature", 0.0))
        log_prior = _evaluate(self.log_prior, particles, context)
        log_like = _evaluate(self.log_likelihood, particles, context)
        if self.log_proposal is None:
            posterior = log_prior + beta * log_like
        else:
            log_prop = _evaluate(self.log_proposal, particles, context)
            posterior = (1.0 - beta) * log_prop + beta * (log_prior + log_like)
        if particles.ndim == 1:
            return float(posterior[0])
        return posterior
