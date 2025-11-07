from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .._array_api import arange, asarray, namespace, to_numpy
from .._typing import Array
from .base import SMCStep, StepContext

if TYPE_CHECKING:
    from orng import ArrayRNG

    from ..state import ParticleState

logger = logging.getLogger(__name__)


class ResamplingStep(SMCStep):
    """Base class for resampling steps with an optional ESS trigger.

    Parameters
    ----------
    ess_threshold:
        Resample when the effective sample size (ESS) ratio falls below this
        threshold.  If ``None`` (default) resampling is never triggered based on
        ESS.
    always_resample:
        When ``True`` always resample regardless of ESS (overrides
        ``ess_threshold``).
    """

    def __init__(
        self,
        *,
        ess_threshold: float | None = 0.5,
        always_resample: bool = False,
    ):
        self._ess_threshold = ess_threshold
        self._always_resample = always_resample

    def apply(
        self,
        state: ParticleState,
        context: StepContext,
        rng: "ArrayRNG",
    ) -> ParticleState:
        ess = state.effective_sample_size()
        n = state.log_weights.shape[0]
        ess_ratio = ess / n if n else 0.0
        if not self._should_resample(state):
            logger.debug(
                "%s skipped resampling (ESS ratio=%.4f).",
                self.__class__.__name__,
                ess_ratio,
            )
            return state
        logger.debug(
            "%s performing resampling (ESS ratio=%.4f).",
            self.__class__.__name__,
            ess_ratio,
        )
        weights = state.weights
        indices = self._choose_ancestors(weights, rng)
        return state.resample(indices)

    def _choose_ancestors(self, weights: Array, rng: "ArrayRNG") -> Array:
        raise NotImplementedError

    def _should_resample(self, state: ParticleState) -> bool:
        if self._always_resample:
            return True
        if self._ess_threshold is None:
            return False
        ess = state.effective_sample_size()
        n = state.log_weights.shape[0]
        return ess / n < self._ess_threshold


class MultinomialResampling(ResamplingStep):
    """Draw ancestors with replacement according to the weights."""

    def _choose_ancestors(self, weights: Array, rng: "ArrayRNG") -> Array:
        xp = namespace(weights)
        probs = asarray(weights, xp=xp)
        n = probs.shape[0]
        draws = rng.choice(n, size=n, p=probs)
        return asarray(draws, xp=xp, like=probs)


class StratifiedResampling(ResamplingStep):
    """Stratified resampling for reduced variance."""

    def _choose_ancestors(self, weights: Array, rng: "ArrayRNG") -> Array:
        xp = namespace(weights)
        probs = asarray(weights, xp=xp)
        n = probs.shape[0]
        offsets = arange(n, xp=xp, dtype=probs.dtype, like=probs)
        jitter = asarray(rng.random((n,)), xp=xp, like=probs)
        positions = (offsets + jitter) / n
        cumulative = _cumulative_sum(probs, xp=xp)
        return _searchsorted(cumulative, positions, xp=xp)


class SystematicResampling(ResamplingStep):
    """Systematic resampling for reduced variance."""

    def _choose_ancestors(self, weights: Array, rng: "ArrayRNG") -> Array:
        xp = namespace(weights)
        probs = asarray(weights, xp=xp)
        n = probs.shape[0]
        start = asarray(rng.random(), xp=xp, like=probs)
        offsets = arange(n, xp=xp, dtype=probs.dtype, like=probs)
        positions = (start + offsets) / n
        cumulative = _cumulative_sum(probs, xp=xp)
        return _searchsorted(cumulative, positions, xp=xp)


def _cumulative_sum(array: Array, *, xp):
    if hasattr(xp, "cumsum"):
        return xp.cumsum(array)
    return asarray(_np_cumsum(array), xp=xp, like=array)


def _np_cumsum(array: Array):
    return to_numpy(array).cumsum()


def _searchsorted(cumulative: Array, positions: Array, *, xp):
    if hasattr(xp, "searchsorted"):
        return xp.searchsorted(cumulative, positions, side="right")
    indices = to_numpy(cumulative)
    pos = to_numpy(positions)
    result = indices.searchsorted(pos, side="right")
    return asarray(result, xp=xp, like=cumulative)
