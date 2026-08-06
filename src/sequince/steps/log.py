import logging
from typing import TYPE_CHECKING, Callable

from .base import SMCStep

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from orng import RandomGenerator

    from ..state import ParticleState
    from .base import StepContext


class LoggingStep(SMCStep):
    """A simple step that logs a message at each iteration.

    Examples
    --------
    >>> def log_fn(state, context):
    ...     return f"Iteration {context.t}: ESS={state.effective_sample_size():.2f}"
    >>> step = LoggingStep(str_fn=log_fn, level=logging.INFO, frequency=1)
    >>> smc = SequentialMonteCarlo(steps=[..., step, ...])

    Parameters
    ----------
    str_fn:
        A function that takes the current ``ParticleState`` and ``StepContext``
        and returns a string message to log.
    level:
        The logging level to use (e.g., ``logging.INFO``).
    frequency:
        How often to log the message (in terms of SMC iterations).
    """

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
        rng: "RandomGenerator",
    ) -> tuple["ParticleState", "StepContext", None]:
        if context.t % self._frequency != 0:
            return state, context, None
        message = self._str_fn(state, context)
        logger.log(self._level, message)
        return state, context, None
