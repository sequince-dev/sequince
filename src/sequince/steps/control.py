from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Sequence

from .base import SMCStep, StepContext

if TYPE_CHECKING:
    from orng import RandomGenerator

    from ..state import ParticleState

logger = logging.getLogger(__name__)


class ContinueWhile(SMCStep):
    """Ensure the SMC loop continues while a predicate holds.

    This step clears any stop requests raised by earlier steps when the supplied
    predicate evaluates to ``True`` for the current ``ParticleState`` and
    ``StepContext``.  When the predicate returns ``False`` the stop flags remain
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
        rng: "RandomGenerator",
    ) -> "ParticleState":
        if self._predicate(state, context):
            context.clear_stop_request()
            logger.debug(
                "ContinueWhile cleared stop flags at iteration %d.", context.t
            )
        else:
            logger.debug(
                "ContinueWhile left stop flags set at iteration %d.", context.t
            )
        return state


class StopIf(SMCStep):
    """Request the SMC loop to stop when a predicate holds.

    This step sets a stop request flag in the context when the supplied predicate
    evaluates to ``True`` for the current ``ParticleState`` and ``StepContext``.
    When the predicate returns ``False`` the stop flag remains untouched, allowing
    the run to continue.

    Parameters
    ----------
    predicate:
        A callable receiving the current ``ParticleState`` and ``StepContext``
        and returning a boolean indicating whether to request stopping the SMC
        loop.
    immediate:
        When ``True``, halt the current iteration immediately. When ``False``,
        stop after the current iteration finishes.
    """

    def __init__(
        self,
        predicate: Callable[["ParticleState", StepContext], bool],
        *,
        immediate: bool = False,
    ):
        self._predicate = predicate
        self._immediate = immediate

    def apply(
        self,
        state: "ParticleState",
        context: StepContext,
        rng: "RandomGenerator",
    ) -> "ParticleState":
        if self._predicate(state, context):
            if self._immediate:
                context.request_stop_immediately()
            else:
                context.request_stop()
            logger.debug(
                "StopIf set stop flags at iteration %d (immediate=%s).",
                context.t,
                self._immediate,
            )
        else:
            logger.debug(
                "StopIf left stop flag cleared at iteration %d.", context.t
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
        self.step = step
        self.n = n
        self.offset = offset

    def apply(
        self,
        state: "ParticleState",
        context: StepContext,
        rng: "RandomGenerator",
    ) -> "ParticleState":
        if (context.t - self.offset) % self.n == 0:
            logger.debug(
                "Applying step at iteration %d (every %d iterations with offset %d).",
                context.t,
                self.n,
                self.offset,
            )
            return self.step.apply(state, context, rng)
        else:
            logger.debug(
                "Skipping step at iteration %d (not a multiple of %d with offset %d).",
                context.t,
                self.n,
                self.offset,
            )
            return state


class RepeatStep(SMCStep):
    """Repeat a step or sequence of steps a fixed number of times."""

    def __init__(self, step: SMCStep, n: int):
        if n < 1:
            raise ValueError("n must be at least one.")
        self.step = step
        self.n = int(n)

    def apply(self, state, context, rng):
        for _ in range(self.n):
            state = self.step(state, context, rng)

            if context.metadata.get("stop_immediately", False):
                break

        return state


class SequentialStep(SMCStep):
    """Apply several SMC steps in order.

    This is useful for grouping multiple steps together into a single step,
    which can then be used in other contexts (e.g., as part of a
    `RepeatStep` or `PeriodicStep`).
    """

    def __init__(self, steps: Sequence[SMCStep]):
        self.steps = tuple(steps)

    def apply(self, state, context, rng):
        for step in self.steps:
            state = step(state, context, rng)

            if context.metadata.get("stop_immediately", False):
                break

        return state
