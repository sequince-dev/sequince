from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np

from .base import SMCStep, StepContext

if TYPE_CHECKING:
    from ..state import ParticleState


class ContinueWhile(SMCStep):
    """Ensure the SMC loop continues while a predicate holds.

    This step clears any stop requests raised by earlier steps when the supplied
    predicate evaluates to ``True`` for the current ``ParticleState`` and
    ``StepContext``.  When the predicate returns ``False`` the stop flag remains
    untouched, allowing the run to terminate.
    """

    def __init__(
        self,
        predicate: Callable[["ParticleState", StepContext], bool],
    ):
        self._predicate = predicate

    def apply(
        self,
        state: "ParticleState",
        context: StepContext,
        rng: np.random.Generator,
    ) -> "ParticleState":
        if self._predicate(state, context):
            context.stop_requested = False
        return state
