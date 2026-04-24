from __future__ import annotations

from typing import TYPE_CHECKING

from ..utils import resolve_rng
from .context import StepContext

if TYPE_CHECKING:
    from orng import ArrayRNG

    from ..state import ParticleState


class SMCStep:
    """Base class for composable SMC steps."""

    def __call__(
        self,
        state: "ParticleState",
        context: StepContext,
        rng: "ArrayRNG" | int | None = None,
    ) -> "ParticleState":
        return self.apply(state, context, resolve_rng(rng))

    def apply(
        self,
        state: "ParticleState",
        context: StepContext,
        rng: "ArrayRNG",
    ) -> "ParticleState":
        raise NotImplementedError("SMCStep subclasses must implement apply().")
