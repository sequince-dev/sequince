"""Persistent particle archival and tempered importance weighting.

See: https://arxiv.org/abs/2407.20722
"""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping
from numbers import Integral
from typing import TYPE_CHECKING, Any

from .._typing import Array, ArrayLike
from ..array_api import asarray, logsumexp, namespace
from ..state import (
    ParticleState,
    PersistentParticleState,
    _validate_log_normalizer,
)
from ..stats import ess_from_log_weights
from ..target import EvaluationCounts, Target, TemperedTarget
from ..utils.optimize import bisect_monotone_decreasing
from .base import SMCStep
from .context import StepContext
from .weighting import _safe_tempered_log_increment

if TYPE_CHECKING:
    from orng import RandomGenerator

__all__ = [
    "PersistParticles",
    "PersistentAdaptiveTemperedWeighting",
    "PersistentTemperedWeighting",
    "persistent_log_mixture_denominator",
    "persistent_log_weights",
    "expected_unique_count",
]


def expected_unique_count(weights: Array, n_active: int) -> Array:
    """Return multinomial expected unique entries for normalized weights.

    Parameters
    ----------
    weights : array, shape (n_archive,)
        Nonnegative, normalized floating-point weights (zeros are allowed).
    n_active : int
        Positive number of draws with replacement.

    Returns
    -------
    array
        Scalar expected unique count, on the input backend and device.
        This is only a diversity heuristic for systematic resampling.
    """
    if isinstance(n_active, bool) or not isinstance(n_active, Integral):
        raise TypeError("n_active must be a positive integer.")
    if n_active < 1:
        raise ValueError("n_active must be positive.")
    xp = namespace(weights)
    # Avoid log1p(-1), including warnings on eager NumPy backends.
    certain = weights >= 1
    safe = xp.where(certain, xp.zeros_like(weights), weights)
    probabilities = -xp.expm1(n_active * xp.log1p(-safe))
    return xp.sum(xp.where(certain, xp.ones_like(weights), probabilities))


def _persistent_arrays(
    log_increment: Array,
    betas: ArrayLike,
    log_normalizing_constants: Array,
    *,
    xp: Any | None,
) -> tuple[Array, Array, Array, Any]:
    """Convert and validate the arrays used by persistent weighting."""
    increments = asarray(log_increment, xp=xp)
    xp = namespace(increments)
    betas = asarray(
        betas,
        xp=xp,
        dtype=increments.dtype,
        like=increments,
    )
    log_normalizing_constants = asarray(
        log_normalizing_constants,
        xp=xp,
        dtype=increments.dtype,
        like=increments,
    )

    if increments.ndim != 1:
        raise ValueError("log_increment must be one-dimensional.")
    if betas.ndim != 1:
        raise ValueError("betas must be one-dimensional.")
    if log_normalizing_constants.ndim != 1:
        raise ValueError("log_normalizing_constants must be one-dimensional.")
    if betas.shape[0] == 0:
        raise ValueError("beta history must not be empty.")
    if betas.shape[0] != log_normalizing_constants.shape[0]:
        raise ValueError(
            "betas and log_normalizing_constants must have the same length."
        )

    return increments, betas, log_normalizing_constants, xp


def persistent_log_mixture_denominator(
    log_increment: Array,
    betas: ArrayLike,
    log_normalizing_constants: Array,
    *,
    xp: Any | None = None,
) -> Array:
    """Compute the log density of the equally weighted historical mixture.

    The common initial-target density is cancelled, leaving, for particle
    ``j``, ``logmeanexp(betas * log_increment[j] -
    log_normalizing_constants)``.
    """
    increments, betas, log_normalizing_constants, xp = _persistent_arrays(
        log_increment,
        betas,
        log_normalizing_constants,
        xp=xp,
    )
    terms = (
        _safe_tempered_log_increment(increments[:, None], betas[None, :])
        - log_normalizing_constants[None, :]
    )
    n_targets = asarray(
        betas.shape[0],
        xp=xp,
        dtype=increments.dtype,
        like=increments,
    )
    return logsumexp(terms, axis=1, xp=xp) - xp.log(n_targets)


def persistent_log_weights(
    log_increment: Array,
    betas: ArrayLike,
    log_normalizing_constants: Array,
    beta: ArrayLike,
    *,
    xp: Any | None = None,
) -> Array:
    """Compute unnormalised persistent importance weights."""
    increments = asarray(log_increment, xp=xp)
    xp = namespace(increments)
    candidate_beta = asarray(
        beta,
        xp=xp,
        dtype=increments.dtype,
        like=increments,
    )
    if candidate_beta.ndim != 0:
        raise ValueError("beta must be scalar.")

    log_denominator = persistent_log_mixture_denominator(
        increments,
        betas,
        log_normalizing_constants,
        xp=xp,
    )
    return (
        _safe_tempered_log_increment(increments, candidate_beta)
        - log_denominator
    )


class PersistParticles(SMCStep):
    """Archive populations in equal-sized batches.

    ``n_particles`` fixes the active batch size. The first state may contain
    any positive multiple of that size, with equal weights at beta zero.
    All initial batches are archived in one call. If omitted, infer the batch
    size from the first state, preserving the one-batch interface.
    """

    def __init__(
        self, target: Target | None = None, *, n_particles: int | None = None
    ) -> None:
        if target is not None and not isinstance(target, Target):
            raise TypeError("target must be a Target instance or None.")
        if n_particles is not None:
            if isinstance(n_particles, bool) or not isinstance(
                n_particles, Integral
            ):
                raise TypeError("n_particles must be a positive integer.")
            if n_particles <= 0:
                raise ValueError("n_particles must be positive.")
        self._n_particles = n_particles
        self._target = target
        self.persistent_state: PersistentParticleState | None = None

    def apply(
        self,
        state: ParticleState,
        context: StepContext,
        rng: "RandomGenerator",
    ) -> tuple[PersistentParticleState, StepContext, None]:
        del rng
        if isinstance(state, PersistentParticleState):
            raise TypeError(
                "PersistParticles requires an active ParticleState."
            )
        batch_size = self._n_particles or state.particles.shape[0]
        size = state.particles.shape[0]
        # Match BaseTemperedStep: context is authoritative, active-state metadata
        # is the fallback, and an uninitialized run starts at beta = 0.
        beta = context.metadata.get("beta", state.metadata.get("beta", 0.0))
        if self.persistent_state is None:
            if size < batch_size or size % batch_size:
                raise ValueError(
                    "Initial population must be a positive multiple of n_particles."
                )
            if size > batch_size:
                xp = namespace(state.log_weights)
                if float(beta) != 0.0:
                    raise ValueError(
                        "Larger initial populations require beta=0."
                    )
                if not bool(xp.all(state.log_weights == state.log_weights[0])):
                    raise ValueError("Initial draws must have equal weights.")
        elif size != self.persistent_state.batch_size:
            raise ValueError(
                "Active population must match the archive batch size."
            )
        if self._target is not None:
            evaluation, counts = self._target.prepare_evaluation(
                state.particles,
                context,
                evaluation=state.evaluation,
                return_counts=True,
            )
            state = state.with_updates(evaluation=evaluation)
            context = context.add_evaluation_counts(counts)

        if self.persistent_state is None:
            xp = namespace(state.particles)
            n_batches = size // batch_size
            archived = PersistentParticleState(
                particles=state.particles,
                batch_size=batch_size,
                betas=asarray([beta] * n_batches, xp=xp, like=state.particles),
                log_normalizing_constants=asarray(
                    [state.log_normalizing_constant] * n_batches,
                    xp=xp,
                    like=state.particles,
                ),
                initial_ancestor_indices=state.initial_ancestor_indices,
                evaluation=state.evaluation,
                metadata=dict(state.metadata),
            )
        else:
            archived = self.persistent_state.append(
                state,
                beta=beta,
                log_normalizing_constant=state.log_normalizing_constant,
            )
        self.persistent_state = archived
        return archived, context, None

    def state_dict(self) -> dict[str, Any]:
        """Return the archive in checkpoint-friendly mapping form."""
        payload: dict[str, Any] = {
            "persistent_state": (
                None
                if self.persistent_state is None
                else self.persistent_state.to_dict()
            )
        }
        if self._target is not None:
            payload["target_evaluation_id"] = self._target.evaluation_id
        return payload

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore an archive produced by :meth:`state_dict`."""
        if "initialization" in state:
            raise ValueError(
                "Staged-initialization checkpoints are no longer supported."
            )
        evaluation_id = state.get("target_evaluation_id")
        if evaluation_id is not None:
            if self._target is None:
                raise ValueError(
                    "Checkpoint requires PersistParticles to have a target."
                )
            if not isinstance(evaluation_id, str):
                raise TypeError(
                    "PersistParticles target_evaluation_id must be a string."
                )
            object.__setattr__(self._target, "_evaluation_id", evaluation_id)
        payload = state.get("persistent_state")
        if payload is None:
            self.persistent_state = None
            return
        if not isinstance(payload, Mapping):
            raise TypeError("PersistParticles state must contain a mapping.")
        archived = PersistentParticleState.from_dict(payload)
        if (
            self._n_particles is not None
            and archived.batch_size != self._n_particles
        ):
            raise ValueError(
                "Checkpoint archive batch size does not match n_particles."
            )
        self.persistent_state = archived


class BasePersistentTemperedWeighting(SMCStep):
    """Shared target evaluation and weighting for persistent tempering."""

    def __init__(
        self,
        target: TemperedTarget,
        *,
        refresh_components: Collection[str] = (),
    ) -> None:
        if not isinstance(target, TemperedTarget):
            raise TypeError(
                f"{type(self).__name__} requires a TemperedTarget."
            )
        self._target = target
        self._refresh_components = frozenset(refresh_components)

    def _require_persistent_state(
        self, state: ParticleState
    ) -> PersistentParticleState:
        if not isinstance(state, PersistentParticleState):
            raise TypeError(
                f"{type(self).__name__} requires a PersistentParticleState."
            )
        return state

    def _prepare_log_increment(
        self,
        state: PersistentParticleState,
        context: StepContext,
    ) -> tuple[Any, Array, Any]:
        """Prepare one cached target evaluation and its log increment."""
        evaluation, counts = self._target.prepare_evaluation(
            state.particles,
            context,
            evaluation=state.evaluation,
            refresh_components=self._refresh_components,
            return_counts=True,
        )
        log_increment = self._target.log_increment(
            state.particles,
            context,
            evaluation=evaluation,
        )
        xp = namespace(state.log_weights)
        counts = counts + EvaluationCounts(
            log_increment_evaluations=xp.asarray(state.particles.shape[0])
        )
        log_increment = asarray(log_increment, xp=xp, like=state.log_weights)
        if log_increment.ndim == 0:
            log_increment = xp.broadcast_to(
                log_increment, state.log_weights.shape
            )
        if log_increment.shape != state.log_weights.shape:
            raise ValueError(
                "Target log increment must return one value per particle."
            )
        return evaluation, log_increment, counts

    def _weight(
        self,
        state: PersistentParticleState,
        evaluation: Any,
        log_increment: Array,
        beta: ArrayLike,
        context: StepContext,
        *,
        log_denominator: Array | None = None,
    ) -> tuple[PersistentParticleState, StepContext, dict[str, Any]]:
        """Weight the persistent archive at a chosen beta.

        Normalize the resulting importance weights, calculate the direct
        persistent normalizing-constant estimate, and update the state and
        context beta metadata. A precomputed mixture denominator may be
        supplied by adaptive beta selection.
        """
        xp = namespace(state.log_weights)
        beta = asarray(
            beta,
            xp=xp,
            dtype=log_increment.dtype,
            like=log_increment,
        )
        if beta.ndim != 0:
            raise ValueError("beta must be scalar.")
        if log_denominator is None:
            log_denominator = persistent_log_mixture_denominator(
                log_increment,
                state.betas,
                state.log_normalizing_constants,
                xp=xp,
            )
        unnormalized = (
            _safe_tempered_log_increment(log_increment, beta) - log_denominator
        )
        log_norm = logsumexp(unnormalized, xp=xp)
        _validate_log_normalizer(log_norm, xp=xp)
        particle_count = asarray(
            state.n_particles,
            xp=xp,
            dtype=unnormalized.dtype,
            like=unnormalized,
        )
        log_normalizing_constant = log_norm - xp.log(particle_count)
        normalized = unnormalized - log_norm
        effective_sample_size = ess_from_log_weights(unnormalized)

        current_beta = context.metadata.get(
            "beta", state.metadata.get("beta", 0.0)
        )
        delta_beta = beta - asarray(
            current_beta,
            xp=xp,
            dtype=unnormalized.dtype,
            like=unnormalized,
        )
        context = context.with_metadata(
            beta=beta,
            delta_beta=delta_beta,
        )
        metadata = dict(state.metadata)
        metadata.update(
            beta=beta,
            delta_beta=delta_beta,
            effective_sample_size=effective_sample_size,
        )
        weighted = state.with_updates(
            log_weights=normalized,
            log_normalizing_constant=log_normalizing_constant,
            evaluation=evaluation,
            metadata=metadata,
        )
        return (
            weighted,
            context,
            {
                "beta": beta,
                "delta_beta": delta_beta,
                "persistent_log_weights": unnormalized,
                "effective_sample_size": effective_sample_size,
            },
        )


class PersistentTemperedWeighting(BasePersistentTemperedWeighting):
    """Weight a persistent particle pool directly toward a fixed beta."""

    def __init__(
        self,
        target: TemperedTarget,
        *,
        beta: ArrayLike,
        refresh_components: Collection[str] = (),
    ) -> None:
        super().__init__(target, refresh_components=refresh_components)
        self._beta = beta

    def apply(
        self,
        state: ParticleState,
        context: StepContext,
        rng: "RandomGenerator",
    ) -> tuple[PersistentParticleState, StepContext, dict[str, Any]]:
        del rng
        state = self._require_persistent_state(state)
        evaluation, log_increment, counts = self._prepare_log_increment(
            state, context
        )
        context = context.add_evaluation_counts(counts)
        return self._weight(
            state,
            evaluation,
            log_increment,
            self._beta,
            context,
        )


class PersistentAdaptiveTemperedWeighting(BasePersistentTemperedWeighting):
    """Choose a persistent target beta using absolute archive ESS.

    The historical deterministic-mixture density is evaluated once, then the
    beta is bisected over the interval from the current beta to 1. If the
    requested ESS cannot be attained while advancing, the current
    beta is repeated so that another batch can be added to the archive.
    Once beta = 1 is selected, the weighting requests a
    runner stop. When ``target_total_ess`` is configured, beta = 1
    batches are instead accumulated until the complete archive reaches that
    ESS.

    Parameters
    ----------
    target:
        Tempered target defining the persistent importance weights.
    target_ess:
        Desired effective sample size while selecting betas below 1.
    target_total_ess:
        Optional effective sample size required before stopping at beta = 1.
        If omitted, stop when beta = 1 is first selected.
    dynamic : bool, optional
        Adapt the integer ESS threshold for the next iteration using expected
        multinomial unique entries. Defaults to False. The fixed reference is
        set by the initial threshold; the terminal target never adapts.
    resample_at_end : bool, optional
        If True, let downstream resampling and mutation generate a beta-one
        batch before checking termination on the next archived, weighted state.
        The terminal ESS requirement is checked after that batch is included.
        Defaults to False, returning the terminal weighted archive immediately.
    beta_tolerance:
        Stop bisection once the beta bracket is no wider than this.
    max_bisection_iter:
        Maximum number of bisection iterations.
    refresh_components:
        Cached target-evaluation components to refresh before weighting.
    """

    _dynamic: bool
    _resample_at_end: bool
    _initial_target_ess: float
    _target_ess: float
    _target_total_ess: float | None
    _beta_tolerance: float
    _max_bisection_iter: int

    def __init__(
        self,
        target: TemperedTarget,
        *,
        target_ess: float = 512,
        target_total_ess: float | None = None,
        dynamic: bool = False,
        resample_at_end: bool = False,
        beta_tolerance: float = 1e-6,
        max_bisection_iter: int = 30,
        refresh_components: Collection[str] = (),
    ) -> None:
        super().__init__(target, refresh_components=refresh_components)
        if isinstance(target_ess, bool) or isinstance(target_total_ess, bool):
            raise TypeError("ESS targets must be numbers, not booleans.")
        if not math.isfinite(target_ess) or not target_ess > 0.0:
            raise ValueError("target_ess must be positive.")
        if target_total_ess is not None and (
            not math.isfinite(target_total_ess) or not target_total_ess > 0.0
        ):
            raise ValueError("target_total_ess must be positive.")
        if not math.isfinite(beta_tolerance) or not beta_tolerance > 0.0:
            raise ValueError("beta_tolerance must be positive.")
        if isinstance(max_bisection_iter, bool) or not isinstance(
            max_bisection_iter, Integral
        ):
            raise TypeError("max_bisection_iter must be an integer.")
        if max_bisection_iter <= 0:
            raise ValueError("max_bisection_iter must be positive.")
        if not isinstance(dynamic, bool):
            raise TypeError("dynamic must be a bool.")
        if not isinstance(resample_at_end, bool):
            raise TypeError("resample_at_end must be a bool.")
        if dynamic and target_ess < 1:
            raise ValueError("Dynamic target_ess must be at least one.")
        self._dynamic = dynamic
        self._resample_at_end = resample_at_end
        self._initial_target_ess = int(target_ess) if dynamic else target_ess
        self._target_ess = self._initial_target_ess
        self._target_total_ess = target_total_ess
        self._beta_tolerance = beta_tolerance
        self._max_bisection_iter = max_bisection_iter

    def apply(
        self,
        state: ParticleState,
        context: StepContext,
        rng: "RandomGenerator",
    ) -> tuple[PersistentParticleState, StepContext, dict[str, Any]]:
        del rng
        state = self._require_persistent_state(state)
        evaluation, log_increment, counts = self._prepare_log_increment(
            state, context
        )
        context = context.add_evaluation_counts(counts)
        xp = namespace(state.log_weights)
        log_denominator = persistent_log_mixture_denominator(
            log_increment,
            state.betas,
            state.log_normalizing_constants,
            xp=xp,
        )
        current_beta = float(
            context.metadata.get("beta", state.metadata.get("beta", 0.0))
        )
        if not 0.0 <= current_beta <= 1.0:
            raise ValueError("current beta must be in [0, 1].")

        def effective_sample_size(beta: Array) -> Array:
            candidate_log_weights = (
                _safe_tempered_log_increment(log_increment, beta)
                - log_denominator
            )
            return ess_from_log_weights(candidate_log_weights)

        selected_terminal = current_beta == 1.0
        if selected_terminal:
            beta = 1.0
        else:
            current = asarray(
                current_beta,
                xp=xp,
                dtype=log_increment.dtype,
                like=log_increment,
            )
            terminal = asarray(
                1.0,
                xp=xp,
                dtype=log_increment.dtype,
                like=log_increment,
            )
            target = asarray(
                self._target_ess,
                xp=xp,
                dtype=log_increment.dtype,
                like=log_increment,
            )
            terminal_ess = float(effective_sample_size(terminal))
            if terminal_ess >= self._target_ess:
                beta = 1.0
                selected_terminal = True
            else:
                current_ess = float(effective_sample_size(current))
                if current_ess < self._target_ess:
                    beta = current_beta
                else:
                    tolerance = asarray(
                        self._beta_tolerance,
                        xp=xp,
                        dtype=log_increment.dtype,
                        like=log_increment,
                    )
                    low, _ = bisect_monotone_decreasing(
                        effective_sample_size,
                        current,
                        terminal,
                        target,
                        tolerance=tolerance,
                        max_iterations=self._max_bisection_iter,
                    )
                    beta = float(low)

        weighted, context, output = self._weight(
            state,
            evaluation,
            log_increment,
            beta,
            context,
            log_denominator=log_denominator,
        )
        target_ess = asarray(
            self._target_ess,
            xp=xp,
            dtype=weighted.log_weights.dtype,
            like=weighted.log_weights,
        )
        metadata = dict(weighted.metadata)
        metadata["target_ess"] = target_ess
        weighted = weighted.with_updates(metadata=metadata)
        context = context.with_metadata(target_ess=target_ess)
        output["target_ess"] = target_ess
        if self._dynamic:
            n_active = state.batch_size
            unique_fraction = (
                float(expected_unique_count(weighted.weights, n_active))
                / n_active
            )
            initial = self._initial_target_ess
            reference = (
                1.0
                if initial == 1
                else -initial * math.expm1(n_active * math.log1p(-1 / initial))
            ) / n_active
            if unique_fraction < 0.95 * reference:
                self._target_ess = int(self._target_ess / unique_fraction)
            elif unique_fraction > min(1.05 * reference, 1.0):
                self._target_ess = max(
                    1, int(self._target_ess * unique_fraction)
                )
            diagnostics = dict(
                next_target_ess=self._target_ess,
                expected_unique_fraction=unique_fraction,
                reference_unique_fraction=reference,
            )
            output.update(diagnostics)
            context = context.with_metadata(**diagnostics)
            weighted = weighted.with_updates(
                metadata={**weighted.metadata, **diagnostics}
            )
        if self._target_total_ess is not None:
            target_total_ess = asarray(
                self._target_total_ess,
                xp=xp,
                dtype=weighted.log_weights.dtype,
                like=weighted.log_weights,
            )
            metadata = dict(weighted.metadata)
            metadata["target_total_ess"] = target_total_ess
            weighted = weighted.with_updates(metadata=metadata)
            context = context.with_metadata(target_total_ess=target_total_ess)
            output["target_total_ess"] = target_total_ess

        if (
            selected_terminal
            # A newly selected beta of one still needs its resample/mutate
            # cycle. On the next iteration, persistence includes that batch,
            # and the stopping test uses the complete reweighted archive.
            and (not self._resample_at_end or current_beta == 1.0)
            and (
                self._target_total_ess is None
                or float(output["effective_sample_size"])
                >= self._target_total_ess
            )
        ):
            context = context.request_stop_immediately()
        return weighted, context, output

    def state_dict(self) -> dict[str, Any]:
        """Save the next threshold and the fixed feedback reference."""
        return dict(
            dynamic=self._dynamic,
            target_ess=self._target_ess,
            initial_target_ess=self._initial_target_ess,
            target_total_ess=self._target_total_ess,
            resample_at_end=self._resample_at_end,
        )

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Load matching configuration; legacy empty state is static only."""
        if not state:
            if self._resample_at_end:
                raise ValueError(
                    "Checkpoint weighting configuration mismatch."
                )
            if self._dynamic:
                raise ValueError(
                    "Legacy static checkpoint cannot resume dynamically."
                )
            return
        for key, value in self.state_dict().items():
            saved = (
                state.get(key, False)
                if key == "resample_at_end"
                else state.get(key)
            )
            if key != "target_ess" and saved != value:
                raise ValueError(
                    "Checkpoint weighting configuration mismatch."
                )
        threshold = state["target_ess"]
        if (
            not math.isfinite(threshold)
            or threshold <= 0
            or isinstance(threshold, bool)
            or (self._dynamic and threshold != int(threshold))
        ):
            raise ValueError("Invalid checkpoint target_ess.")
        self._target_ess = threshold

    def apply_functional(
        self,
        state: ParticleState,
        context: StepContext,
        random_state: Any,
    ) -> tuple[ParticleState, StepContext, Any, dict[str, Any]]:
        """Reject functional execution because the archive grows dynamically."""
        del state, context, random_state
        raise NotImplementedError(
            "Persistent adaptive tempering supports eager execution only."
        )
