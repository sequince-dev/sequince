from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

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

    def state_dict(self) -> dict[str, Any]:
        """Return mutable state required by subsequent SMC iterations."""
        return {}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore state produced by :meth:`state_dict`."""
        if state:
            raise ValueError(
                f"{type(self).__name__} does not support checkpoint state."
            )
