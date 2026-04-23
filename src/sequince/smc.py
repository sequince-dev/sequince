from __future__ import annotations

import logging
import copy
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    Iterator,
    List,
    Mapping,
    MutableMapping,
    Sequence,
)

from .result import SMCResult
from .state import ParticleState
from .steps.base import StepContext
from .utils import resolve_rng

if TYPE_CHECKING:
    import numpy as np
    from orng import ArrayRNG

logger = logging.getLogger(__name__)


class SequentialMonteCarlo:
    """Composable Sequential Monte Carlo runner.

    Users supply an ordered collection of SMC steps that will be called in sequence at
    each iteration. Steps can mutate the state or return a new one.

    Parameters
    ----------
    steps:
        A sequence of callables implementing SMC steps.  Each callable should accept
        three arguments: the current ``ParticleState``, the current ``StepContext``,
        and an ``orng.ArrayRNG`` instance, and return a new ``ParticleState``.
    callbacks:
        Optional sequence of callables that will be invoked at the end of each iteration
        with the current state, context, and an ``SMCResult`` containing all states
        produced so far.  Callbacks can be used for logging, diagnostics, or custom
        stopping criteria that require access to the full history of states.
    auto_normalize:
        If ``True`` (default) automatically normalize the particle weights at the end
        of each iteration.
    """

    def __init__(
        self,
        steps: Sequence[
            Callable[
                [ParticleState, StepContext, "ArrayRNG"],
                ParticleState,
            ]
        ],
        *,
        callbacks: Sequence[
            Callable[[ParticleState, StepContext, SMCResult], None]
        ]
        | None = None,
        auto_normalize: bool = True,
    ):
        self._steps = list(steps)
        self._callbacks = list(callbacks) if callbacks is not None else []
        self._auto_normalize = auto_normalize

    def run(
        self,
        initial_state: ParticleState,
        observations: Iterable[Any] | None = None,
        num_steps: int | None = None,
        rng: "ArrayRNG" | np.random.Generator | int | None = None,
        shared_context: Mapping[str, Any] | None = None,
        metadata: MutableMapping[str, Any] | None = None,
        outputs: MutableMapping[str, Any] | None = None,
    ) -> SMCResult:
        """Run the SMC algorithm.

        Parameters
        ----------
        initial_state:
            Initial particle state to start the SMC run.
        observations:
            Optional iterable of observations to provide to each iteration via
            the context. If not provided, the context observation will be
            ``None`` for all iterations.
        num_steps:
            Optional maximum number of iterations to run. If not provided, the
            SMC loop will continue until the observations are exhausted
            (if provided) or indefinitely if not.
        rng:
            Optional random number generator or seed for reproducibility. If
            not provided, a new RNG will be created.
        shared_context:
            Optional dictionary of shared context values that will be passed to
            each step via the context. This can be used to provide global
            information that steps may need access to.
        metadata:
            Optional mutable dictionary for storing metadata that can be used
            for communication between steps and callbacks.
        outputs:
            Optional mutable dictionary for storing outputs from steps.
        """
        rng = resolve_rng(rng)
        shared_context = shared_context or {}
        metadata = metadata or {}
        outputs = outputs or {}

        logger.info(
            "Starting SMC run with %d steps (auto_normalize=%s)",
            len(self._steps),
            self._auto_normalize,
        )
        if num_steps is not None:
            logger.debug(
                "Run constrained to at most %d iterations.", num_steps
            )
        if observations is not None:
            logger.debug("Observations iterator supplied to SMC run.")

        produced_states: List[ParticleState] = []
        state = initial_state

        observation_iterator = (
            iter(observations) if observations is not None else None
        )

        t = 0
        steps_remaining = num_steps if num_steps is not None else None

        while steps_remaining is None or steps_remaining > 0:
            logger.debug("Beginning iteration %d.", t)
            if observation_iterator is not None:
                try:
                    observation = next(observation_iterator)
                except StopIteration:
                    if steps_remaining is None:
                        break
                    observation = None
            else:
                observation = None

            if steps_remaining is not None:
                steps_remaining -= 1
                logger.debug("Iterations remaining: %d", steps_remaining)

            context = StepContext(
                t=t,
                observation=observation,
                shared=shared_context,
                metadata=metadata,
                outputs=outputs,
            )
            for step in self._steps:
                step_name = getattr(step, "__name__", step.__class__.__name__)
                logger.debug("Iteration %d: applying step %s.", t, step_name)
                state = step(state, context, rng)
                if context.metadata["stop_requested"]:
                    logger.info(
                        "Stop requested by step %s at iteration %d; terminating SMC run.",
                        step_name,
                        t,
                    )
                    break
            if self._auto_normalize:
                pre_log_evidence = state.log_evidence
                state, _ = state.normalize()
                logger.debug(
                    "Iteration %d: normalized weights (log_evidence=%.6f -> %.6f).",
                    t,
                    pre_log_evidence,
                    state.log_evidence,
                )
            produced_states.append(state)

            for callback in self._callbacks:
                callback_name = getattr(
                    callback, "__name__", callback.__class__.__name__
                )
                logger.debug(
                    "Iteration %d: invoking callback %s.", t, callback_name
                )
                callback(
                    state,
                    context,
                    SMCResult(states=produced_states, context=context),
                )

            if context.metadata["stop_requested"]:
                logger.info(
                    "Stop requested after iteration %d; terminating SMC run.",
                    t,
                )
                break

            t += 1

        logger.info(
            "Finished SMC run after %d iterations; produced %d states.",
            len(produced_states),
            len(produced_states),
        )
        return SMCResult(states=produced_states, context=context)
