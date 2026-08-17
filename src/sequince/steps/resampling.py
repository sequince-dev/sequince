from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import array_api_extra as xpx
from array_api_compat import device as get_device

from .._typing import Array
from ..array_api import asarray, namespace
from ..utils import normalize_log_weights, pytree_dataclass
from .base import SMCStep, StepContext

if TYPE_CHECKING:
    from orng import RandomGenerator

    from ..state import ParticleState

logger = logging.getLogger(__name__)


@pytree_dataclass(
    data_fields=("ess", "resampled", "ancestors"),
    meta_fields=("resampling_scheme",),
)
@dataclass(frozen=True)
class ResamplingOutput(Mapping[str, Any]):
    """Fixed-structure resampling output with a static scheme label."""

    ess: float | Array
    resampled: bool | Array
    ancestors: Array
    resampling_scheme: str

    def __getitem__(self, key: str) -> Any:
        if key not in {
            "ess",
            "resampled",
            "ancestors",
            "resampling_scheme",
        }:
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self):
        return iter(("ess", "resampled", "ancestors", "resampling_scheme"))

    def __len__(self) -> int:
        return 4


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

    resampling_scheme: ClassVar[str] = "unknown"

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
    ) -> tuple[ParticleState, StepContext, ResamplingOutput]:
        ess = state.effective_sample_size()
        n = state.log_weights.shape[0]
        ess_ratio = ess / n if n else 0.0
        xp = namespace(state.log_weights)
        identity_ancestors = xp.arange(
            n,
            device=get_device(state.log_weights),
        )
        if not self._should_resample(state):
            logger.debug(
                "%s skipped resampling (ESS ratio=%.4f).",
                self.__class__.__name__,
                ess_ratio,
            )
            context = context.with_metadata(ess=ess, resampled=False)
            return (
                state,
                context,
                ResamplingOutput(
                    ess=ess,
                    resampled=False,
                    ancestors=identity_ancestors,
                    resampling_scheme=self.resampling_scheme,
                ),
            )
        logger.debug(
            "%s performing resampling (ESS ratio=%.4f).",
            self.__class__.__name__,
            ess_ratio,
        )
        weights = state.weights
        indices = self._choose_ancestors(weights, rng)
        context = context.with_metadata(ess=ess, resampled=True)
        return (
            state.resample(indices),
            context,
            ResamplingOutput(
                ess=ess,
                resampled=True,
                ancestors=indices,
                resampling_scheme=self.resampling_scheme,
            ),
        )

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
            dtype=state.log_weights.dtype,
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
    ) -> tuple[ParticleState, StepContext, object, ResamplingOutput]:
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
        updated = state.resample(indices)
        context = context.with_metadata(ess=ess, resampled=xp.asarray(True))
        output = ResamplingOutput(
            ess=ess,
            resampled=xp.asarray(True),
            ancestors=indices,
            resampling_scheme=self.resampling_scheme,
        )
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

    resampling_scheme = "multinomial"

    def _choose_ancestors(
        self, weights: Array, rng: "RandomGenerator"
    ) -> Array:
        xp = namespace(weights)
        probs = asarray(weights, xp=xp)
        n = probs.shape[0]
        return rng.choice(n, size=n, p=probs)


class StratifiedResampling(ResamplingStep):
    """Stratified resampling for reduced variance."""

    resampling_scheme = "stratified"

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

    resampling_scheme = "systematic"

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
