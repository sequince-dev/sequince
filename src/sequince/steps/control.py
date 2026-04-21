from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

from .base import SMCStep, StepContext

if TYPE_CHECKING:
    from orng import ArrayRNG

    from ..state import ParticleState

logger = logging.getLogger(__name__)


class ContinueWhile(SMCStep):
    """Ensure the SMC loop continues while a predicate holds.

    This step clears any stop requests raised by earlier steps when the supplied
    predicate evaluates to ``True`` for the current ``ParticleState`` and
    ``StepContext``.  When the predicate returns ``False`` the stop flag remains
    untouched, allowing the run to terminate.

    Parameters
    ----------
    predicate:
        A callable receiving the current ``ParticleState`` and ``StepContext``
        and returning a boolean indicating whether to continue the SMC loop.
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
        rng: "ArrayRNG",
    ) -> "ParticleState":
        if self._predicate(state, context):
            context.stop_requested = False
            logger.debug(
                "ContinueWhile cleared stop flag at iteration %d.", context.t
            )
        else:
            logger.debug(
                "ContinueWhile left stop flag set at iteration %d.", context.t
            )
        return state


class PeriodicStep(SMCStep):
    """Apply a step every N iterations.

    This step applies the supplied ``step`` only on iterations where the current
    iteration number is a multiple of ``n``.  On other iterations, it simply
    returns the input state unchanged.

    Parameters
    ----------
    step:
        The step to apply every N iterations.
    n:
        The number of iterations between applications of the step.
    offset:
        An optional offset to apply to the iteration number before checking for
        divisibility.  This allows you to, for example, apply a step every 5
        iterations starting from iteration 2 by setting ``n=5`` and ``offset=2``.
    """

    def __init__(self, step: SMCStep, n: int, offset: int = 0):
        self._step = step
        self._n = n
        self._offset = offset

    def apply(
        self,
        state: "ParticleState",
        context: StepContext,
        rng: "ArrayRNG",
    ) -> "ParticleState":
        if (context.t - self._offset) % self._n == 0:
            logger.debug(
                "Applying step at iteration %d (every %d iterations with offset %d).",
                context.t,
                self._n,
                self._offset,
            )
            return self._step.apply(state, context, rng)
        else:
            logger.debug(
                "Skipping step at iteration %d (not a multiple of %d with offset %d).",
                context.t,
                self._n,
                self._offset,
            )
            return state
