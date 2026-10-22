from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Callable, Sequence

from acai import cond
from array_api_compat import is_jax_namespace

from .._typing import Array
from ..array_api import namespace
from .base import SMCStep, StepContext

if TYPE_CHECKING:
    from orng import RandomGenerator

    from ..state import ParticleState

logger = logging.getLogger(__name__)


def apply_functional_if(
    predicate: bool | Array,
    apply: Callable[
        [ParticleState, StepContext, Any],
        tuple[ParticleState, StepContext, Any, Any],
    ],
    state: ParticleState,
    context: StepContext,
    random_state: Any,
) -> tuple[ParticleState, StepContext, Any, Any]:
    """Conditionally execute a functional step through backend control flow.

    Parameters
    ----------
    predicate : bool or scalar array
        Execute the step when true; otherwise preserve the entire input carry.
    apply : callable
        Pure, traceable ``apply(state, context, random_state)`` operation.
    state : ParticleState
        Initialized particle state, including any evaluation cache.
    context : StepContext
        Initialized context with all metadata required by the operation.
    random_state : object
        Explicit backend random state.

    Returns
    -------
    state, context, random_state, output : tuple
        Executed step results, or unchanged inputs. Skipped output is None on
        eager backends. On JAX it contains zero array placeholders in the step's
        output structure, with static fields unchanged.

    Notes
    -----
    Uses acai.cond: eager backends execute only the selected branch. On JAX,
    abstract evaluation traces ``apply`` even when skipped, but does not execute
    transitions or consume random draws. Both branches must preserve the carry's
    pytree structure, shapes, and dtypes. Initialize missing leaves with the
    functional initialization hooks before calling this helper. Output leaves
    must support JAX zero construction; placeholders are not step diagnostics.
    PyTorch execution is eager; structured torch.compile is not guaranteed.
    """
    xp = namespace(state.particles)
    carry = (state, context, random_state)
    empty = None
    if is_jax_namespace(xp):
        import jax

        abstract = jax.eval_shape(apply, *carry)
        if jax.tree.structure(abstract[:3]) != jax.tree.structure(carry):
            raise ValueError(
                "Conditional step changes its carry structure. Initialize all "
                "state, context, and evaluation-cache leaves before guarding it."
            )
        empty = jax.tree.map(
            lambda leaf: xp.zeros(leaf.shape, leaf.dtype), abstract[3]
        )
    return cond(
        predicate,
        lambda carry: apply(*carry),
        lambda carry: (*carry, empty),
        carry,
        xp=xp,
    )


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

    def apply_functional(
        self, state: ParticleState, context: StepContext, random_state: Any
    ) -> tuple[ParticleState, StepContext, Any, None]:
        """Evaluate a traceable predicate without consuming random state."""
        xp = namespace(state.particles)
        stop = xp.asarray(self._predicate(state, context), dtype=bool)
        context = replace(
            context,
            stop_requested=xp.where(stop, True, context.stop_requested),
            stop_immediately=xp.where(
                stop, self._immediate, context.stop_immediately
            ),
        )
        return state, context, random_state, None

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

    def state_dict(self) -> dict[str, Any]:
        return {"step": self.step.state_dict()}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        nested_state = state.get("step", {})
        if not isinstance(nested_state, Mapping):
            raise TypeError("RepeatStep state 'step' must be a mapping.")
        self.step.load_state_dict(nested_state)


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

    def state_dict(self) -> dict[str, Any]:
        return {"steps": [step.state_dict() for step in self.steps]}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if "steps" not in state:
            # Checkpoints created before nested SequentialStep state was
            # supported contain the inherited empty state.
            return
        nested_states = state.get("steps", [])
        if not isinstance(nested_states, Sequence) or isinstance(
            nested_states, (str, bytes)
        ):
            raise TypeError("SequentialStep state 'steps' must be a sequence.")
        if len(nested_states) != len(self.steps):
            raise ValueError(
                "SequentialStep checkpoint state does not match its children."
            )
        for step, nested_state in zip(self.steps, nested_states, strict=True):
            if not isinstance(nested_state, Mapping):
                raise TypeError(
                    "SequentialStep child state must be a mapping."
                )
            step.load_state_dict(nested_state)
