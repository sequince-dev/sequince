"""Kernel-independent expected squared jump distance."""

from __future__ import annotations

from dataclasses import dataclass

from ....array_api import asarray, namespace
from ....typing import Array
from ...context import StepContext
from .state import MutationTransition


def _squared_jump_distances(displacement: Array, accepted: Array) -> Array:
    """Return one squared jump per particle, including zeros for rejection.

    Flatten trailing coordinate axes and retain the input backend and dtype.
    Both overall ESJD and component-wise metrics use this definition.
    """
    xp = namespace(displacement)
    displacement = xp.reshape(displacement, (displacement.shape[0], -1))
    return xp.where(accepted, xp.sum(displacement**2, axis=1), 0.0)


@dataclass(frozen=True)
class ESJDMetric:
    """Mean squared displacement per attempted particle transition.

    Rejections contribute zero. Distances use mutation coordinates (latent
    coordinates when a transform is configured), without additional whitening
    or dimension normalization. This measures individual transitions, not the
    displacement between a complete block's starting and ending populations.

    ``name`` defaults to "esjd"; the result is a scalar array. State retains
    one previous population and scalar totals, without a chain history or
    additional target evaluations. Optional transition displacements are used
    when supplied; otherwise consecutive populations supply the displacement.
    Preparing a new segment resets the previous population but preserves totals,
    excluding moves between segments. Coordinates must use a consistent metric
    for the accumulated average to be meaningful.
    """

    name: str = "esjd"

    def initialize(
        self, particles: Array, log_weights: Array, context: StepContext
    ) -> tuple[Array, Array, Array]:
        """Initialize the previous population and scalar totals."""
        xp = namespace(particles)
        if particles.ndim < 1 or particles.shape[0] == 0:
            raise ValueError("ESJD requires a nonempty particle population.")
        if not xp.isdtype(particles.dtype, "real floating"):
            raise TypeError("ESJD requires floating-point particles.")
        return (
            particles,
            asarray(0.0, like=particles, dtype=particles.dtype),
            asarray(0, like=particles, dtype=xp.asarray(0).dtype),
        )

    def prepare(
        self,
        state: tuple[Array, Array, Array],
        particles: Array,
        context: StepContext,
    ) -> tuple[Array, Array, Array]:
        """Preserve totals while excluding inter-segment particle changes."""
        return particles, state[1], state[2]

    def update(
        self,
        state: tuple[Array, Array, Array],
        particles: Array,
        accepted: Array,
        context: StepContext,
        *,
        transition: MutationTransition | None = None,
    ) -> tuple[Array, Array, Array]:
        """Accumulate squared movement and count all attempted transitions."""
        xp = namespace(particles)
        displacement = (
            particles - state[0]
            if transition is None
            else transition.displacement
        )
        jumps = _squared_jump_distances(displacement, accepted)
        return (
            particles,
            state[1] + xp.sum(jumps),
            state[2] + particles.shape[0],
        )

    def result(
        self, state: tuple[Array, Array, Array], context: StepContext
    ) -> Array:
        """Return ESJD, or zero when no transitions were recorded."""
        xp = namespace(state[1])
        denominator = xp.astype(
            xp.maximum(state[2], xp.ones_like(state[2])), state[1].dtype
        )
        return state[1] / denominator


__all__ = ["ESJDMetric"]
