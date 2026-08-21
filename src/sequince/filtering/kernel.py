"""Transition kernels for sequential importance sampling."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from .._typing import Array

if TYPE_CHECKING:
    from orng import RandomGenerator

    from ..steps.context import StepContext


class ImportanceKernel(Protocol):
    """Define a particle transition and its importance-weight correction."""

    def sample(
        self,
        particles: Array,
        context: StepContext,
        rng: RandomGenerator,
    ) -> Array:
        """Sample the next particle population from the supplied parents."""

    def log_increment(
        self,
        previous_particles: Array,
        particles: Array,
        context: StepContext,
    ) -> Array | float:
        """Return the transition's importance log-weight increment."""


__all__ = ["ImportanceKernel"]
