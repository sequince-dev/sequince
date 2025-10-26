from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from .base import SMCStep, StepContext

if TYPE_CHECKING:
    from ..state import ParticleState


class ResamplingStep(SMCStep):
    """Base class for resampling steps with an optional ESS trigger."""

    def __init__(
        self,
        *,
        ess_threshold: Optional[float] = 0.5,
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
        if not self._should_resample(state):
            return state
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
        return rng.choice(n, size=n, p=weights)


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
