"""Relative jump distance in fixed, already-whitened mutation coordinates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from array_api_compat import device as get_device
from orng import RandomGenerator

from ....array_api import asarray, namespace, to_numpy
from ....typing import Array
from ....utils.pytrees import pytree_dataclass
from ...base import validate_n_steps
from ...context import StepContext
from .state import MutationTransition


def _pair_median_reference_radius(
    particles: Array, *, n_pairs: int, seed: int
) -> Array:
    """Median of independent empirical pair distances, including self pairs."""
    xp = namespace(particles)
    if particles.shape[0] < 2:
        raise ValueError("Pair-median radius requires at least two particles.")
    if not bool(to_numpy(xp.all(xp.isfinite(particles)))):
        raise ValueError("Pair-median radius requires finite particles.")
    rng = RandomGenerator.from_xp(xp, seed=seed, device=get_device(particles))
    indices = rng.choice(particles.shape[0], size=(2, n_pairs), replace=True)
    differences = particles[indices[0]] - particles[indices[1]]
    distances = xp.sort(xp.sqrt(xp.sum(differences**2, axis=1)))
    # Average the central distances for an even sample count.
    radius = (distances[(n_pairs - 1) // 2] + distances[n_pairs // 2]) / 2
    value = float(to_numpy(radius))
    if not math.isfinite(value) or value <= 0:
        raise ValueError(
            "Pair-median radius is not finite and positive; use a fixed "
            "reference_radius or increase n_pairs for a sparse population."
        )
    return radius


@pytree_dataclass(data_fields=("start", "end", "radius"))
@dataclass(frozen=True)
class _RJDState:
    start: Array
    end: Array
    radius: Array


@dataclass(frozen=True)
class RelativeJumpDistanceMetric:
    """Measure endpoint movement relative to a population reference distance.

    Parameters
    ----------
    reference_radius : float or {"pair_median"}
        Positive fixed distance in whitened coordinates, default 1. The default
        is an approximation; whitening alone does not imply this radius.
        "pair_median" estimates the median distance between independent
        empirical draws with replacement, including self pairs, once at block
        initialization. This estimator does not perform clustering.
    name : str
        Result key in MutationMetricsOutput.values, default "rjd".
    seed : int
        Independent diagnostic seed, default 0. Reset on each initialization
        for reproducible radii without consuming the mutation RNG.
    n_pairs : int
        Number of sampled pairs for "pair_median", default 4096.

    Notes
    -----
    Input must already be whitened. Particle identities and the coordinate
    system must remain fixed across all segments of a block: do not reorder,
    resample, or refit transforms between segments. Endpoint displacement
    includes intervening steps when they preserve this interpretation.
    Weights are not used: diagnostics describe the equally counted chains.

    Outputs are scalar arrays: fraction_above_one, geometric_mean and
    reference_radius. A zero-distance chain makes the geometric mean zero.
    Fixed radii support functional/JIT initialization. Pair-median
    initialization is eager and uses O(n_pairs * D) memory; subsequent metric
    updates support JIT. Only two populations and the frozen radius are retained.
    It is an SMC heuristic, not Buchner's neighbour-radius estimator.
    No mode consumes mutation RNG draws or adds target evaluations.

    Based on Buchner (2024), https://arxiv.org/abs/2402.11936.
    """

    reference_radius: float | Literal["pair_median"] = 1.0
    name: str = "rjd"
    seed: int = 0
    n_pairs: int = 4096

    def __post_init__(self) -> None:
        if isinstance(self.reference_radius, str):
            if self.reference_radius != "pair_median":
                raise ValueError("Unknown reference_radius mode.")
        elif isinstance(self.reference_radius, bool) or (
            not math.isfinite(self.reference_radius)
            or self.reference_radius <= 0
        ):
            raise ValueError("reference_radius must be finite and positive.")
        if validate_n_steps(self.n_pairs) == 0:
            raise ValueError("n_pairs must be positive.")
        validate_n_steps(self.seed)

    def initialize(
        self, particles: Array, log_weights: Array, context: StepContext
    ) -> _RJDState:
        """Freeze the starting population and radius for one complete block."""
        xp = namespace(particles)
        if particles.ndim != 2 or min(particles.shape) == 0:
            raise ValueError("RJD requires a nonempty 2D particle array.")
        if not xp.isdtype(particles.dtype, "real floating"):
            raise TypeError("RJD requires floating-point particles.")
        if self.reference_radius == "pair_median":
            radius = _pair_median_reference_radius(
                particles, n_pairs=self.n_pairs, seed=self.seed
            )
        else:
            radius = asarray(
                self.reference_radius, like=particles, dtype=particles.dtype
            )
        return _RJDState(particles, particles, radius)

    def prepare(
        self, state: _RJDState, particles: Array, context: StepContext
    ) -> _RJDState:
        """Continue the same block without resetting its start or radius."""
        if particles.shape != state.start.shape:
            raise ValueError("RJD population shape must remain fixed.")
        return _RJDState(state.start, particles, state.radius)

    def update(
        self,
        state: _RJDState,
        particles: Array,
        accepted: Array,
        context: StepContext,
        *,
        transition: MutationTransition | None = None,
    ) -> _RJDState:
        """Retain the latest endpoint, including unchanged rejected chains."""
        return self.prepare(state, particles, context)

    def result(
        self, state: _RJDState, context: StepContext
    ) -> dict[str, Array]:
        """Return endpoint RJD summaries without retaining particle histories."""
        xp = namespace(state.end)
        rjd = xp.sqrt(xp.sum((state.end - state.start) ** 2, axis=1))
        rjd = rjd / state.radius
        positive = rjd > 0
        # Avoid log(0) warnings; restore the exact geometric mean afterwards.
        log_rjd = xp.log(xp.where(positive, rjd, xp.ones_like(rjd)))
        geometric_mean = xp.where(
            xp.all(positive), xp.exp(xp.mean(log_rjd)), 0.0
        )
        return {
            "fraction_above_one": xp.where(
                xp.all(xp.isfinite(rjd)),
                xp.mean(xp.astype(rjd > 1, rjd.dtype)),
                xp.nan,
            ),
            "geometric_mean": xp.where(
                xp.all(xp.isfinite(rjd)), geometric_mean, xp.nan
            ),
            "reference_radius": state.radius,
        }


__all__ = ["RelativeJumpDistanceMetric"]
