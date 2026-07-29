from __future__ import annotations

import copy
import logging
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from .checkpoint import (
    CheckpointWriter,
    SMCCheckpoint,
    StepState,
    capture_step_states,
    restore_steps,
)
from .result import SMCResult
from .state import ParticleState
from .steps.base import StepContext
from .utils import resolve_rng

if TYPE_CHECKING:
    import numpy as np
    from orng import RandomGenerator

logger = logging.getLogger(__name__)


@dataclass
class SMCRuntime:
    """Prepared SMC runtime state for fresh or resumed execution."""

    state: ParticleState
    produced_states: list[ParticleState]
    initial_state: ParticleState | None
    t: int
    next_observation_index: int
    shared_context: dict[str, Any]
    metadata: MutableMapping[str, Any]
    outputs: MutableMapping[str, Any]
    rng: Any


class SequentialMonteCarlo:
    """Composable Sequential Monte Carlo runner.

    Users supply an ordered collection of SMC steps that will be called in sequence at
    each iteration. Steps can mutate the state or return a new one.

    Parameters
    ----------
    steps:
        A sequence of callables implementing SMC steps.  Each callable should accept
        three arguments: the current ``ParticleState``, the current ``StepContext``,
        and an ``orng.RandomGenerator`` instance, and return a new ``ParticleState``.
    callbacks:
        Optional sequence of callables that will be invoked at the end of each iteration
        with the current state, context, and an ``SMCResult`` containing all states
        produced so far.  Callbacks can be used for logging, diagnostics, or custom
        stopping criteria that require access to the full history of states.
    checkpoint_writers:
        Optional persistence hooks invoked with a committed checkpoint after all
        ordinary callbacks have completed.
    auto_normalize:
        If ``True`` (default) automatically normalize the particle weights at the end
        of each iteration.
    """

    def __init__(
        self,
        steps: Sequence[
            Callable[
                [ParticleState, StepContext, "RandomGenerator"],
                ParticleState,
            ]
        ],
        *,
        callbacks: Sequence[Callable[..., None]] | None = None,
        checkpoint_writers: Sequence[CheckpointWriter] | None = None,
        auto_normalize: bool = True,
    ):
        self._steps = list(steps)
        self._callbacks = list(callbacks) if callbacks is not None else []
        self._checkpoint_writers = list(checkpoint_writers or ())
        self._auto_normalize = auto_normalize

    def state_dict(self) -> dict[str, Any]:
        """Return mutable state for the configured step pipeline."""
        return {
            "steps": [
                item.to_dict() for item in capture_step_states(self._steps)
            ]
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore mutable state for the configured step pipeline."""
        step_payload = state.get("steps", [])
        if not isinstance(step_payload, Sequence):
            raise TypeError("SMC runner state 'steps' must be a sequence.")
        restore_steps(
            self._steps,
            [StepState.from_dict(item) for item in step_payload],
        )

    def run(
        self,
        initial_state: ParticleState | None = None,
        observations: Iterable[Any] | None = None,
        num_steps: int | None = None,
        rng: "RandomGenerator" | np.random.Generator | int | None = None,
        shared_context: Mapping[str, Any] | None = None,
        metadata: MutableMapping[str, Any] | None = None,
        outputs: MutableMapping[str, Any] | None = None,
        runtime: SMCRuntime | None = None,
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
        runtime:
            Optional prepared runtime returned by :meth:`resume_from`.
        """
        if runtime is not None:
            if initial_state is not None:
                raise ValueError(
                    "run() accepts either initial_state or runtime, not both."
                )
            if any(
                value is not None
                for value in (rng, shared_context, metadata, outputs)
            ):
                raise ValueError(
                    "run(runtime=...) does not accept rng, shared_context, "
                    "metadata, or outputs."
                )
        else:
            if initial_state is None:
                raise ValueError(
                    "run() requires either initial_state or runtime."
                )
            runtime = SMCRuntime(
                state=initial_state.copy(),
                produced_states=[],
                initial_state=initial_state.copy(),
                t=0,
                next_observation_index=0,
                shared_context=dict(shared_context or {}),
                metadata=metadata or {},
                outputs=outputs or {},
                rng=resolve_rng(rng),
            )

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
        if (
            runtime.next_observation_index > 0
            and observations is not None
            and not isinstance(observations, Sequence)
        ):
            raise ValueError(
                "Resuming with observations requires an indexable sequence or None."
            )

        observation_sequence = (
            observations if isinstance(observations, Sequence) else None
        )
        observation_iterator = (
            iter(observations)
            if observations is not None and observation_sequence is None
            else None
        )
        if (
            runtime.next_observation_index > 0
            and observation_sequence is None
            and observations is not None
        ):
            raise ValueError(
                "Resumed runtime only supports observations=None or sequence observations."
            )

        steps_remaining = num_steps if num_steps is not None else None
        produced_states = list(runtime.produced_states)
        state = runtime.state
        t = runtime.t
        next_observation_index = runtime.next_observation_index
        shared_context = runtime.shared_context
        metadata = runtime.metadata
        outputs = runtime.outputs
        rng = runtime.rng
        context = None

        while steps_remaining is None or steps_remaining > 0:
            logger.debug("Beginning iteration %d.", t)
            if observation_sequence is not None:
                if next_observation_index < len(observation_sequence):
                    observation = observation_sequence[next_observation_index]
                    next_observation_index += 1
                elif steps_remaining is None:
                    break
                else:
                    observation = None
            elif observation_iterator is not None:
                try:
                    observation = next(observation_iterator)
                    next_observation_index += 1
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
                next_observation_index=next_observation_index,
                observation=observation,
                shared=shared_context,
                metadata=metadata,
                outputs=outputs,
            )
            for step in self._steps:
                step_name = getattr(step, "__name__", step.__class__.__name__)
                logger.debug("Iteration %d: applying step %s.", t, step_name)
                state = step(state, context, rng)
                if context.metadata.get("stop_immediately", False):
                    logger.info(
                        "Immediate stop requested by step %s at iteration %d; terminating iteration early.",
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
            result = SMCResult(
                states=list(produced_states),
                context=context,
                initial_state=runtime.initial_state,
            )

            for callback in self._callbacks:
                callback_name = getattr(
                    callback, "__name__", callback.__class__.__name__
                )
                logger.debug(
                    "Iteration %d: invoking callback %s.", t, callback_name
                )
                callback(state, context, result)

            next_iteration = context.t + 1
            checkpoint_writers = [
                writer
                for writer in self._checkpoint_writers
                if writer.is_due(next_iteration)
            ]
            if checkpoint_writers:
                include_history = any(
                    writer.include_history for writer in checkpoint_writers
                )
                save_rng = any(
                    writer.save_rng for writer in checkpoint_writers
                )
                checkpoint = SMCCheckpoint.from_runtime(
                    state=state,
                    context=context,
                    result=result,
                    steps=self._steps,
                    rng=rng if save_rng else None,
                    include_history=include_history,
                )
                for writer in checkpoint_writers:
                    writer.write(checkpoint)

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
        if context is None or not produced_states:
            raise ValueError("SMC run produced no states.")
        return SMCResult(
            states=list(produced_states),
            context=context,
            initial_state=runtime.initial_state,
        )

    def resume_from(
        self,
        checkpoint: SMCCheckpoint | Mapping[str, Any],
        *,
        rng: "RandomGenerator" | np.random.Generator | int | None = None,
        shared_context: Mapping[str, Any] | None = None,
    ) -> SMCRuntime:
        """Reconstruct runtime state from a checkpoint for use with :meth:`run`."""
        checkpoint_obj = (
            checkpoint
            if isinstance(checkpoint, SMCCheckpoint)
            else SMCCheckpoint.from_dict(checkpoint)
        )
        restore_steps(self._steps, checkpoint_obj.step_states)
        restored_rng = checkpoint_obj.restore_rng()
        state = checkpoint_obj.state.copy()
        history = checkpoint_obj.history or (checkpoint_obj.state,)
        return SMCRuntime(
            state=state,
            produced_states=[item.copy() for item in history],
            initial_state=(
                None
                if checkpoint_obj.initial_state is None
                else checkpoint_obj.initial_state.copy()
            ),
            t=checkpoint_obj.next_iteration,
            next_observation_index=checkpoint_obj.next_observation_index,
            shared_context=dict(shared_context or {}),
            metadata=copy.deepcopy(checkpoint_obj.metadata),
            outputs=copy.deepcopy(checkpoint_obj.outputs),
            rng=restored_rng if restored_rng is not None else resolve_rng(rng),
        )
