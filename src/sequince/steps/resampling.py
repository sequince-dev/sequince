from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
from typing import TYPE_CHECKING, Any, ClassVar

import array_api_extra as xpx
from array_api_compat import device as get_device

from .._typing import Array
from ..array_api import asarray, namespace
from ..state import normalize_log_weights
from ..utils.pytrees import pytree_dataclass
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
    n_samples:
        Number of particles to draw when resampling. If ``None``, draw the same
        number as are present in the input state.
    resample_at_end:
        If ``True`` (default) will always resample when the inverse
        temperature β reaches 1.0, regardless of ESS. If ``False``, the final
        resampling may be skipped if ``always_resample`` is ``False`` and the
        ESS ratio is above ``ess_threshold``.
    """

    resampling_scheme: ClassVar[str] = "unknown"

    def __init__(
        self,
        *,
        ess_threshold: float | None = 0.5,
        always_resample: bool = False,
        n_samples: int | None = None,
        resample_at_end: bool = True,
    ):
        if n_samples is None:
            self._n_samples = None
        else:
            if isinstance(n_samples, bool) or not isinstance(
                n_samples, Integral
            ):
                raise TypeError(
                    "n_samples must be a positive integer or None."
                )
            if n_samples <= 0:
                raise ValueError("n_samples must be a positive integer.")
            self._n_samples = int(n_samples)
        self._ess_threshold = ess_threshold
        self._always_resample = always_resample
        self._resample_at_end = resample_at_end

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
        if not self._should_resample(state, context):
            logger.debug(
                "Iteration %d: %s skipped resampling (ESS ratio=%.4f).",
                context.t,
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
            "Iteration %d: %s performing resampling (ESS ratio=%.4f).",
            context.t,
            self.__class__.__name__,
            ess_ratio,
        )
        weights = state.weights
        n_samples = self._n_samples if self._n_samples is not None else n
        indices = self._choose_ancestors(weights, n_samples, rng)
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
        self, weights: Array, n_samples: int, rng: "RandomGenerator"
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
        n_particles = state.log_weights.shape[0]
        n_samples = (
            self._n_samples if self._n_samples is not None else n_particles
        )
        indices = self._choose_ancestors(weights, n_samples, rng)
        updated = state.resample(indices)
        context = context.with_metadata(ess=ess, resampled=xp.asarray(True))
        output = ResamplingOutput(
            ess=ess,
            resampled=xp.asarray(True),
            ancestors=indices,
            resampling_scheme=self.resampling_scheme,
        )
        return updated, context, rng.state(), output

    def _should_resample(
        self, state: ParticleState, context: StepContext
    ) -> bool:
        if self._always_resample:
            return True
        if self._resample_at_end and context.stop_requested:
            logger.debug(
                "Iteration %d: %s resampling at end of run (ESS ratio=%.4f).",
                context.t,
                self.__class__.__name__,
                state.effective_sample_size() / state.log_weights.shape[0],
            )
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
        self, weights: Array, n_samples: int, rng: "RandomGenerator"
    ) -> Array:
        xp = namespace(weights)
        probs = asarray(weights, xp=xp)
        n_particles = probs.shape[0]
        return rng.choice(n_particles, size=n_samples, p=probs)


class StratifiedResampling(ResamplingStep):
    """Stratified resampling for reduced variance."""

    resampling_scheme = "stratified"

    def _choose_ancestors(
        self, weights: Array, n_samples: int, rng: "RandomGenerator"
    ) -> Array:
        xp = namespace(weights)
        probs = asarray(weights, xp=xp)
        offsets = xp.arange(
            n_samples, dtype=probs.dtype, device=get_device(probs)
        )
        jitter = rng.random((n_samples,))
        positions = (offsets + jitter) / n_samples
        cumulative = xp.cumulative_sum(probs)
        return xpx.searchsorted(cumulative, positions, side="right", xp=xp)


class SystematicResampling(ResamplingStep):
    """Systematic resampling for reduced variance."""

    resampling_scheme = "systematic"

    def _choose_ancestors(
        self, weights: Array, n_samples: int, rng: "RandomGenerator"
    ) -> Array:
        xp = namespace(weights)
        probs = asarray(weights, xp=xp)
        start = rng.random()
        offsets = xp.arange(
            n_samples, dtype=probs.dtype, device=get_device(probs)
        )
        positions = (start + offsets) / n_samples
        cumulative = xp.cumulative_sum(probs)
        return xpx.searchsorted(cumulative, positions, side="right", xp=xp)
