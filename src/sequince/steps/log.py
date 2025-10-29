import logging
from typing import TYPE_CHECKING, Callable

import numpy as np

from .base import SMCStep

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..state import ParticleState
    from .base import StepContext


class LoggingStep(SMCStep):
    """A simple step that logs a message at each iteration."""

    def __init__(
        self,
        str_fn: Callable[["ParticleState", "StepContext"], str],
        level: int = logging.INFO,
        frequency: int = 1,
    ):
        super().__init__()
        self._str_fn = str_fn
        self._level = level
        self._frequency = frequency

    def apply(
        self,
        state: "ParticleState",
        context: "StepContext",
        rng: "np.random.Generator",
    ) -> "ParticleState":
        if context.t % self._frequency != 0:
            return state
        message = self._str_fn(state, context)
        logger.log(self._level, message)
        return state
