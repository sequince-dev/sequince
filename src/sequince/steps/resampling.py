from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import array_api_extra as xpx
from array_api_compat import device as get_device

from .._typing import Array
from ..array_api import asarray, namespace
from ..utils import normalize_log_weights
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
    ) -> tuple[ParticleState, StepContext, dict[str, object]]:
        ess = state.effective_sample_size()
        n = state.log_weights.shape[0]
        ess_ratio = ess / n if n else 0.0
        output: dict[str, object] = {"ess": ess, "resampled": False}
        if not self._should_resample(state):
            logger.debug(
                "%s skipped resampling (ESS ratio=%.4f).",
                self.__class__.__name__,
                ess_ratio,
            )
            context = context.with_metadata(ess=ess, resampled=False)
            return state, context, output
        logger.debug(
            "%s performing resampling (ESS ratio=%.4f).",
            self.__class__.__name__,
            ess_ratio,
        )
        weights = state.weights
        indices = self._choose_ancestors(weights, rng)
        context = context.with_metadata(ess=ess, resampled=True)
        output["resampled"] = True
        return state.resample(indices), context, output

    def _choose_ancestors(
        self, weights: Array, rng: "RandomGenerator"
    ) -> Array:
        raise NotImplementedError

    def initialize_functional_metadata(
        self,
        state: ParticleState,
        metadata: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Initialize fixed-shape resampling metadata."""
        xp = namespace(state.log_weights)
        initialized = dict(metadata)
        initialized["ess"] = asarray(
            initialized.get("ess", state.log_weights.shape[0]),
            xp=xp,
            like=state.log_weights,
        )
        initialized["resampled"] = asarray(
            initialized.get("resampled", False),
            xp=xp,
            dtype=bool,
            like=state.log_weights,
        )
        return initialized

    def apply_functional(
        self,
        state: ParticleState,
        context: StepContext,
        random_state: object,
    ) -> tuple[ParticleState, StepContext, object, dict[str, object]]:
        """Resample while explicitly threading RNG state.

        The initial compiled subset requires unconditional resampling.  A
        traced ESS threshold needs a conditional RNG transition and is not
        supported yet.
        """
        if not self._always_resample:
            raise NotImplementedError(
                "functional resampling currently requires always_resample=True"
            )

        from orng import RandomGenerator

        xp = namespace(state.log_weights, random_state)
        normalized, _ = normalize_log_weights(
            state.log_weights,
            xp=xp,
        )
        weights = xp.exp(normalized)
        ess = 1.0 / xp.sum(xp.square(weights))
        rng = RandomGenerator.from_xp(xp, generator=random_state)
        indices = self._choose_ancestors(weights, rng)
        updated = state.resample(indices, record_ancestors=False)
        context = context.with_metadata(ess=ess, resampled=xp.asarray(True))
        output = {
            "ess": ess,
            "resampled": xp.asarray(True),
        }
        return updated, context, rng.state(), output

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
        offsets = xp.arange(n, dtype=probs.dtype, device=get_device(probs))
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
        offsets = xp.arange(n, dtype=probs.dtype, device=get_device(probs))
        positions = (start + offsets) / n
        cumulative = xp.cumulative_sum(probs)
        return xpx.searchsorted(cumulative, positions, side="right", xp=xp)
