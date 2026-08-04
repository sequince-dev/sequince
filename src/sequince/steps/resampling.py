from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import array_api_extra as xpx

from .._typing import Array
from ..array_api import arange, asarray, namespace
from .base import SMCStep, StepContext

if TYPE_CHECKING:
    from orng import RandomGenerator

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
        rng: "RandomGenerator",
    ) -> ParticleState:
        ess = state.effective_sample_size()
        n = state.log_weights.shape[0]
        ess_ratio = ess / n if n else 0.0
        context.metadata["ess"] = ess
        if not self._should_resample(state):
            logger.debug(
                "%s skipped resampling (ESS ratio=%.4f).",
                self.__class__.__name__,
                ess_ratio,
            )
            context.metadata["resampled"] = False
            return state
        logger.debug(
            "%s performing resampling (ESS ratio=%.4f).",
            self.__class__.__name__,
            ess_ratio,
        )
        weights = state.weights
        indices = self._choose_ancestors(weights, rng)
        context.metadata["resampled"] = True
        return state.resample(indices)

    def _choose_ancestors(
        self, weights: Array, rng: "RandomGenerator"
    ) -> Array:
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

    def _choose_ancestors(
        self, weights: Array, rng: "RandomGenerator"
    ) -> Array:
        xp = namespace(weights)
        probs = asarray(weights, xp=xp)
        n = probs.shape[0]
        return rng.choice(n, size=n, p=probs)


class StratifiedResampling(ResamplingStep):
    """Stratified resampling for reduced variance."""

    def _choose_ancestors(
        self, weights: Array, rng: "RandomGenerator"
    ) -> Array:
        xp = namespace(weights)
        probs = asarray(weights, xp=xp)
        n = probs.shape[0]
        offsets = arange(n, xp=xp, dtype=probs.dtype, like=probs)
        jitter = rng.random((n,))
        positions = (offsets + jitter) / n
        cumulative = xp.cumulative_sum(probs)
        return xpx.searchsorted(cumulative, positions, side="right", xp=xp)


class SystematicResampling(ResamplingStep):
    """Systematic resampling for reduced variance."""

    def _choose_ancestors(
        self, weights: Array, rng: "RandomGenerator"
    ) -> Array:
        xp = namespace(weights)
        probs = asarray(weights, xp=xp)
        n = probs.shape[0]
        start = rng.random()
        offsets = arange(n, xp=xp, dtype=probs.dtype, like=probs)
        positions = (start + offsets) / n
        cumulative = xp.cumulative_sum(probs)
        return xpx.searchsorted(cumulative, positions, side="right", xp=xp)
