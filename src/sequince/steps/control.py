from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Sequence, cast

from .base import FixedLengthMutation, SMCStep, StepContext, validate_n_steps

if TYPE_CHECKING:
    from orng import RandomGenerator

    from ..state import ParticleState

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConditionalStepOutput:
    """Output from a step that may skip its child."""

    executed: bool
    child_output: Any = None


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
    ) -> tuple["ParticleState", StepContext, None]:
        if self._predicate(state, context):
            context = context.clear_stop_request()
            logger.debug(
                "ContinueWhile cleared stop flags at iteration %d.", context.t
            )
        else:
            logger.debug(
                "ContinueWhile left stop flags set at iteration %d.", context.t
            )
        return state, context, None


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
    ) -> tuple["ParticleState", StepContext, None]:
        if self._predicate(state, context):
            if self._immediate:
                context = context.request_stop_immediately()
            else:
                context = context.request_stop()
            logger.debug(
                "StopIf set stop flags at iteration %d (immediate=%s).",
                context.t,
                self._immediate,
            )
        else:
            logger.debug(
                "StopIf left stop flag cleared at iteration %d.", context.t
            )
        return state, context, None


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
    ) -> tuple["ParticleState", StepContext, ConditionalStepOutput]:
        if (context.t - self.offset) % self.n == 0:
            logger.debug(
                "Applying step at iteration %d (every %d iterations with offset %d).",
                context.t,
                self.n,
                self.offset,
            )
            state, context, output = self.step.apply(state, context, rng)
            return state, context, ConditionalStepOutput(True, output)
        else:
            logger.debug(
                "Skipping step at iteration %d (not a multiple of %d with offset %d).",
                context.t,
                self.n,
                self.offset,
            )
            return state, context, ConditionalStepOutput(False)

    def state_dict(self) -> dict[str, Any]:
        return {"step": self.step.state_dict()}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        nested_state = state.get("step", {})
        if not isinstance(nested_state, Mapping):
            raise TypeError("PeriodicStep state 'step' must be a mapping.")
        self.step.load_state_dict(nested_state)


class RepeatStep(SMCStep):
    """Repeat a step or sequence of steps a fixed number of times."""

    def __init__(self, step: SMCStep, n: int):
        if n < 1:
            raise ValueError("n must be at least one.")
        self.step = step
        self.n = int(n)

    def apply(self, state, context, rng):
        outputs = []
        for _ in range(self.n):
            state, context, output = self.step(state, context, rng)
            outputs.append(output)

            if context.stop_immediately:
                break

        return state, context, tuple(outputs)

    def apply_with_n_steps(self, state, context, rng, *, n_steps: int):
        """Distribute a total budget while executing every fixed repetition."""
        n_steps = validate_n_steps(n_steps)
        if not isinstance(self.step, FixedLengthMutation):
            raise TypeError(
                "RepeatStep's child does not support fixed-length mutation."
            )
        quotient, remainder = divmod(n_steps, self.n)
        outputs = []
        for round_index in range(self.n):
            round_steps = quotient + (round_index < remainder)
            state, context, output = self.step.apply_with_n_steps(
                state, context, rng, n_steps=round_steps
            )
            outputs.append(output)
        return state, context, tuple(outputs)


class SequentialStep(SMCStep):
    """Apply several SMC steps in order.

    This is useful for grouping multiple steps together into a single step,
    which can then be used in other contexts (e.g., as part of a
    `RepeatStep` or `PeriodicStep`).
    """

    def __init__(self, steps: Sequence[SMCStep]):
        self.steps = tuple(steps)

    def apply(self, state, context, rng):
        outputs = []
        for step in self.steps:
            state, context, output = step(state, context, rng)
            outputs.append(output)

            if context.stop_immediately:
                break

        return state, context, tuple(outputs)

    def apply_with_n_steps(self, state, context, rng, *, n_steps: int):
        """Forward a budget and execute the complete fixed sequence."""
        n_steps = validate_n_steps(n_steps)
        capable = [
            index
            for index, step in enumerate(self.steps)
            if isinstance(step, FixedLengthMutation)
        ]
        if len(capable) != 1:
            raise TypeError(
                "SequentialStep fixed-length execution requires exactly one "
                "fixed-length mutation child."
            )
        mutation_index = capable[0]
        outputs = []
        for index, step in enumerate(self.steps):
            if index == mutation_index:
                fixed_step = cast(FixedLengthMutation, step)
                output_state = fixed_step.apply_with_n_steps(
                    state, context, rng, n_steps=n_steps
                )
            else:
                output_state = step(state, context, rng)
            state, context, output = output_state
            outputs.append(output)
        return state, context, tuple(outputs)
