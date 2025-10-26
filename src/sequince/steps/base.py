from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, MutableMapping, Optional

import numpy as np

from ..utils import _ensure_rng

if TYPE_CHECKING:
    from ..state import ParticleState


@dataclass
class StepContext:
    """Runtime context passed to each SMC step."""

    t: int
    observation: Any = None
    static: MutableMapping[str, Any] = field(default_factory=dict)
    stop_requested: bool = False

    def request_stop(self) -> None:
        """Signal that the SMC runner should halt after this iteration."""
        self.stop_requested = True


class SMCStep:
    """Base class for composable SMC steps."""

    def __call__(
        self,
        state: "ParticleState",
        context: StepContext,
        rng: Optional[np.random.Generator] = None,
    ) -> "ParticleState":
        return self.apply(state, context, _ensure_rng(rng))

    def apply(
        self,
        state: "ParticleState",
        context: StepContext,
        rng: np.random.Generator,
    ) -> "ParticleState":
        raise NotImplementedError("SMCStep subclasses must implement apply().")
