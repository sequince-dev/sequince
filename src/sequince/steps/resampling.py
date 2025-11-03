from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from .base import SMCStep, StepContext

if TYPE_CHECKING:
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
        rng: np.random.Generator,
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

    def _choose_ancestors(
        self, weights: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
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
        self, weights: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        n = weights.shape[0]
        return np.asarray(rng.choice(n, size=n, p=weights))


class StratifiedResampling(ResamplingStep):
    """Stratified resampling for reduced variance."""

    def _choose_ancestors(
        self, weights: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        n = weights.shape[0]
        positions = (np.arange(n) + rng.random(n)) / n
        cumulative = np.cumsum(weights)
        return np.searchsorted(cumulative, positions, side="right")


class SystematicResampling(ResamplingStep):
    """Systematic resampling for reduced variance."""

    def _choose_ancestors(
        self, weights: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        n = weights.shape[0]
        start = rng.random()
        positions = (start + np.arange(n)) / n
        cumulative = np.cumsum(weights)
        return np.searchsorted(cumulative, positions, side="right")
