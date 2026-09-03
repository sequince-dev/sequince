from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Callable, TypeAlias

from .array_api import namespace
from .checkpoint import (
    CheckpointWriter,
    SMCCheckpoint,
    StepState,
    capture_step_states,
    restore_steps,
)
from .result import SMCResult
from .state import ParticleState, normalize_log_weights
from .steps.base import StepContext
from .steps.context import EVALUATION_COUNTS_KEY
from .target import EvaluationCounts
from .utils.copying import array_safe_copy
from .utils.random import resolve_rng

if TYPE_CHECKING:
    import numpy as np
    from orng import RandomGenerator

logger = logging.getLogger(__name__)

IterationCallable: TypeAlias = Callable[
    [ParticleState, StepContext, Any],
    tuple[ParticleState, StepContext, Any, tuple[Any, ...]],
]


@dataclass
class SMCRuntime:
    """Prepared SMC runtime state for fresh or resumed execution.

    ``rng`` contains either an ``orng.RandomGenerator`` for eager execution or
    an explicit backend random state for functional execution.
    """

    state: ParticleState
    produced_states: list[ParticleState]
    initial_state: ParticleState | None
    t: int
    next_observation_index: int
    shared_context: Mapping[str, Any]
    metadata: Mapping[str, Any]
    evaluation_counts: EvaluationCounts
    outputs: list[tuple[Any, ...]]
    rng: Any


class SequentialMonteCarlo:
    """Composable Sequential Monte Carlo runner.

    Users supply an ordered collection of SMC steps that will be called in
    sequence at each iteration. Steps return the updated particle state,
    context, and an explicit output describing the completed work.

    Parameters
    ----------
    steps:
        A sequence of callables implementing SMC steps. Each callable should
        accept the current ``ParticleState``, current ``StepContext``, and an
        ``orng.RandomGenerator``, and return ``(state, context, output)``.
    callbacks:
        Optional callbacks invoked after each completed iteration with the
        current state, context, and partial ``SMCResult``. A callback may
        return an updated context, or ``None`` when observational.
    checkpoint_writers:
        Optional persistence hooks invoked with a committed checkpoint after
        callbacks have completed.
    auto_normalize:
        If ``True``, normalize particle weights after each iteration.
    """

    def __init__(
        self,
        steps: Sequence[
            Callable[
                [ParticleState, StepContext, "RandomGenerator"],
                tuple[ParticleState, StepContext, Any],
            ]
        ],
        *,
        callbacks: Sequence[Callable[..., StepContext | None]] | None = None,
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

    def iteration(
        self,
        state: ParticleState,
        context: StepContext,
        rng: "RandomGenerator",
    ) -> tuple[
        ParticleState,
        StepContext,
        "RandomGenerator",
        tuple[Any, ...],
    ]:
        """Apply one eager SMC iteration.

        The returned RNG is the same mutable generator supplied by the caller.
        Returning it gives eager and functional iterations a common internal
        calling convention.
        """
        outputs = []
        for step in self._steps:
            step_name = getattr(step, "__name__", type(step).__name__)
            logger.debug(
                "Iteration %d: applying step %s.",
                context.t,
                step_name,
            )
            state, context, output = step(state, context, rng)
            outputs.append(output)

            if context.stop_immediately:
                logger.info(
                    "Immediate stop requested by step %s at iteration %d; "
                    "terminating iteration early.",
                    step_name,
                    context.t,
                )
                break

        if self._auto_normalize:
            pre_log_normalizing_constant = state.log_normalizing_constant
            state, log_norm = state.normalize()
            state.log_normalizing_constant += log_norm
            logger.debug(
                "Iteration %d: normalized weights "
                "(log_normalizing_constant=%.6f -> %.6f).",
                context.t,
                pre_log_normalizing_constant,
                state.log_normalizing_constant,
            )

        return state, context, rng, tuple(outputs)

    def functional_iteration(
        self,
        state: ParticleState,
        context: StepContext,
        random_state: Any,
    ) -> tuple[ParticleState, StepContext, Any, tuple[Any, ...]]:
        """Apply one pure, explicitly RNG-threaded SMC iteration.

        Notes
        -----
        This method deliberately does not inspect ``stop_immediately`` between
        steps. Runtime-dependent early interruption of a compiled iteration
        would require JAX control flow and fixed skipped-output structures.
        Stop requests are therefore handled by the host loop after a complete
        functional iteration.
        """
        xp = namespace(state.log_weights)
        outputs = []

        for step in self._steps:
            if not hasattr(step, "apply_functional"):
                raise TypeError(
                    f"{type(step).__name__} does not provide "
                    "apply_functional()."
                )

            state, context, random_state, output = step.apply_functional(
                state,
                context,
                random_state,
            )
            outputs.append(output)

        if self._auto_normalize:
            normalized, log_norm = normalize_log_weights(
                state.log_weights,
                xp=xp,
            )
            state = state.with_updates(
                log_weights=normalized,
                log_normalizing_constant=(
                    state.log_normalizing_constant + log_norm
                ),
            )

        return state, context, random_state, tuple(outputs)

    def run(
        self,
        initial_state: ParticleState | None = None,
        observations: Iterable[Any] | None = None,
        num_steps: int | None = None,
        rng: "RandomGenerator" | np.random.Generator | int | None = None,
        shared_context: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        outputs: Sequence[tuple[Any, ...]] | None = None,
        runtime: SMCRuntime | None = None,
    ) -> SMCResult:
        """Run the SMC algorithm using a mutable random generator."""
        runtime = self._prepare_runtime(
            method_name="run",
            initial_state=initial_state,
            runtime=runtime,
            supplied_rng=rng,
            resolved_rng=(resolve_rng(rng) if runtime is None else None),
            shared_context=shared_context,
            metadata=metadata,
            outputs=outputs,
        )

        result, _ = self._run(
            runtime=runtime,
            observations=observations,
            num_steps=num_steps,
            run_iteration=self.iteration,
        )
        return result

    def initialize_functional(
        self,
        initial_state: ParticleState,
        context: StepContext,
    ) -> tuple[ParticleState, StepContext]:
        """Return shape-complete state and context for functional execution."""
        initialized: Mapping[str, Any] = dict(context.metadata)
        for step in self._steps:
            initializer = getattr(step, "initialize_functional_metadata", None)
            if callable(initializer):
                initialized = initializer(initial_state, initialized)
        context = replace(context, metadata=initialized)

        state = initial_state
        for step in self._steps:
            initializer = getattr(
                step,
                "initialize_functional_state_and_context",
                None,
            )
            if callable(initializer):
                state, context = initializer(state, context)
        return state, context

    def run_functional(
        self,
        initial_state: ParticleState | None = None,
        observations: Iterable[Any] | None = None,
        num_steps: int | None = None,
        random_state: Any = None,
        shared_context: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        outputs: Sequence[tuple[Any, ...]] | None = None,
        runtime: SMCRuntime | None = None,
        iteration: IterationCallable | None = None,
    ) -> tuple[SMCResult, Any]:
        """Run the SMC algorithm with explicit random-state threading.

        ``iteration`` may be an externally transformed version of
        :meth:`functional_iteration`, for example
        ``jax.jit(smc.functional_iteration)``.

        Returns
        -------
        result, random_state:
            The completed SMC result and final backend-native random state.
        """
        if runtime is None and random_state is None:
            raise ValueError(
                "run_functional() requires random_state or runtime."
            )

        runtime = self._prepare_runtime(
            method_name="run_functional",
            initial_state=initial_state,
            runtime=runtime,
            supplied_rng=random_state,
            resolved_rng=random_state,
            shared_context=shared_context,
            metadata=metadata,
            outputs=outputs,
        )

        if iteration is None:
            iteration = self.functional_iteration

        return self._run(
            runtime=runtime,
            observations=observations,
            num_steps=num_steps,
            run_iteration=iteration,
            initialize_functional=True,
        )

    def _prepare_runtime(
        self,
        *,
        method_name: str,
        initial_state: ParticleState | None,
        runtime: SMCRuntime | None,
        supplied_rng: Any,
        resolved_rng: Any,
        shared_context: Mapping[str, Any] | None,
        metadata: Mapping[str, Any] | None,
        outputs: Sequence[tuple[Any, ...]] | None,
    ) -> SMCRuntime:
        """Validate inputs and prepare fresh or resumed runtime state."""
        if runtime is not None:
            if initial_state is not None:
                raise ValueError(
                    f"{method_name}() accepts either initial_state or "
                    "runtime, not both."
                )
            if any(
                value is not None
                for value in (
                    supplied_rng,
                    shared_context,
                    metadata,
                    outputs,
                )
            ):
                raise ValueError(
                    f"{method_name}(runtime=...) does not accept random "
                    "state, shared_context, metadata, or outputs."
                )
            return runtime

        if initial_state is None:
            raise ValueError(
                f"{method_name}() requires either initial_state or runtime."
            )

        runtime_metadata = dict(metadata or {})
        initial_counts = initial_state.metadata.get(
            EVALUATION_COUNTS_KEY, EvaluationCounts()
        )
        if not isinstance(initial_counts, EvaluationCounts):
            raise TypeError(
                "Initial state evaluation_counts metadata must be "
                "EvaluationCounts."
            )

        runtime_state = initial_state.copy()
        runtime_state.metadata.pop(EVALUATION_COUNTS_KEY, None)

        return SMCRuntime(
            state=runtime_state,
            produced_states=[],
            initial_state=runtime_state.copy(),
            t=0,
            next_observation_index=0,
            shared_context=(
                shared_context if shared_context is not None else {}
            ),
            metadata=runtime_metadata,
            evaluation_counts=initial_counts,
            outputs=list(outputs or ()),
            rng=resolved_rng,
        )

    def _run(
        self,
        *,
        runtime: SMCRuntime,
        observations: Iterable[Any] | None,
        num_steps: int | None,
        run_iteration: IterationCallable,
        initialize_functional: bool = False,
    ) -> tuple[SMCResult, Any]:
        """Run common host-side orchestration for eager or functional mode."""
        logger.info(
            "Starting SMC run with %d steps (auto_normalize=%s)",
            len(self._steps),
            self._auto_normalize,
        )
        if num_steps is not None:
            logger.debug(
                "Run constrained to at most %d iterations.",
                num_steps,
            )
        if observations is not None:
            logger.debug("Observations iterator supplied to SMC run.")

        if (
            runtime.next_observation_index > 0
            and observations is not None
            and not isinstance(observations, Sequence)
        ):
            raise ValueError(
                "Resuming with observations requires an indexable sequence "
                "or None."
            )

        observation_sequence = (
            observations if isinstance(observations, Sequence) else None
        )
        observation_iterator = (
            iter(observations)
            if observations is not None and observation_sequence is None
            else None
        )

        steps_remaining = num_steps
        produced_states = list(runtime.produced_states)
        state = runtime.state
        t = runtime.t
        next_observation_index = runtime.next_observation_index
        shared_context = runtime.shared_context
        metadata = runtime.metadata
        evaluation_counts = runtime.evaluation_counts
        outputs = list(runtime.outputs)
        execution_state = runtime.rng
        context: StepContext | None = None

        while steps_remaining is None or steps_remaining > 0:
            logger.debug("Iteration %d.", t)

            (
                observation,
                next_observation_index,
                exhausted,
            ) = self._next_observation(
                observation_sequence=observation_sequence,
                observation_iterator=observation_iterator,
                next_observation_index=next_observation_index,
                allow_missing=steps_remaining is not None,
            )
            if exhausted:
                break

            if steps_remaining is not None:
                steps_remaining -= 1
                logger.debug(
                    "Iterations remaining: %d",
                    steps_remaining,
                )

            context = StepContext(
                t=t,
                next_observation_index=next_observation_index,
                observation=observation,
                shared=shared_context,
                metadata=metadata,
                evaluation_counts=evaluation_counts,
            )

            if initialize_functional:
                state, context = self.initialize_functional(state, context)
                initialize_functional = False

            (
                state,
                context,
                execution_state,
                iteration_outputs,
            ) = run_iteration(
                state,
                context,
                execution_state,
            )

            outputs.append(iteration_outputs)
            produced_states.append(state)

            result = SMCResult(
                states=list(produced_states),
                context=context,
                initial_state=runtime.initial_state,
                outputs=tuple(outputs),
            )

            context = self._run_callbacks(
                state=state,
                context=context,
                result=result,
            )

            # Rebuild the result with any callback-updated context.
            result = SMCResult(
                states=result.states,
                context=context,
                initial_state=result.initial_state,
                outputs=result.outputs,
            )

            next_iteration = context.t + 1
            checkpoint_writers = [
                writer
                for writer in self._checkpoint_writers
                if writer.is_due(next_iteration)
            ]

            if checkpoint_writers:
                save_rng = any(
                    writer.save_rng for writer in checkpoint_writers
                )
                detach_checkpoint = any(
                    writer.requires_detached_checkpoint
                    for writer in checkpoint_writers
                )
                checkpoint = SMCCheckpoint.from_runtime(
                    state=state,
                    context=context,
                    result=result,
                    steps=self._steps,
                    rng=execution_state if save_rng else None,
                    detach=detach_checkpoint,
                )
                for writer in checkpoint_writers:
                    writer.write(checkpoint)

            metadata = context.metadata
            evaluation_counts = context.evaluation_counts
            if context.stop_requested:
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

        return (
            SMCResult(
                states=list(produced_states),
                context=context,
                initial_state=runtime.initial_state,
                outputs=tuple(outputs),
            ),
            execution_state,
        )

    @staticmethod
    def _next_observation(
        *,
        observation_sequence: Sequence[Any] | None,
        observation_iterator: Any,
        next_observation_index: int,
        allow_missing: bool,
    ) -> tuple[Any, int, bool]:
        """Return the next observation and whether input is exhausted.

        When ``allow_missing`` is true, exhaustion produces ``None`` and the
        run may continue until its explicit ``num_steps`` limit. Otherwise,
        exhaustion terminates the run.
        """
        if observation_sequence is not None:
            if next_observation_index < len(observation_sequence):
                observation = observation_sequence[next_observation_index]
                return observation, next_observation_index + 1, False

            if allow_missing:
                return None, next_observation_index, False

            return None, next_observation_index, True

        if observation_iterator is not None:
            try:
                observation = next(observation_iterator)
            except StopIteration:
                if allow_missing:
                    return None, next_observation_index, False
                return None, next_observation_index, True

            return observation, next_observation_index + 1, False

        return None, next_observation_index, False

    def _run_callbacks(
        self,
        *,
        state: ParticleState,
        context: StepContext,
        result: SMCResult,
    ) -> StepContext:
        """Run callbacks and propagate explicitly returned context updates."""
        for callback in self._callbacks:
            callback_name = getattr(
                callback,
                "__name__",
                callback.__class__.__name__,
            )
            logger.debug(
                "Iteration %d: invoking callback %s.",
                context.t,
                callback_name,
            )
            updated_context = callback(state, context, result)
            if updated_context is not None:
                context = updated_context

        return context

    def resume_from(
        self,
        checkpoint: SMCCheckpoint | Mapping[str, Any],
        *,
        rng: "RandomGenerator" | np.random.Generator | int | None = None,
        shared_context: Mapping[str, Any] | None = None,
    ) -> SMCRuntime:
        """Reconstruct eager runtime state from a checkpoint."""
        checkpoint_obj = self._coerce_checkpoint(checkpoint)
        restore_steps(self._steps, checkpoint_obj.step_states)

        restored_rng = checkpoint_obj.restore_rng()
        return self._runtime_from_checkpoint(
            checkpoint_obj,
            shared_context=shared_context,
            rng=(
                restored_rng if restored_rng is not None else resolve_rng(rng)
            ),
        )

    def resume_functional_from(
        self,
        checkpoint: SMCCheckpoint | Mapping[str, Any],
        *,
        random_state: Any = None,
        shared_context: Mapping[str, Any] | None = None,
    ) -> SMCRuntime:
        """Reconstruct functional runtime state from a checkpoint."""
        checkpoint_obj = self._coerce_checkpoint(checkpoint)
        restore_steps(self._steps, checkpoint_obj.step_states)

        restored_rng = checkpoint_obj.restore_rng()
        if restored_rng is not None:
            _, restored_random_state = restored_rng.to_functional()
        elif random_state is not None:
            restored_random_state = random_state
        else:
            raise ValueError(
                "Checkpoint does not contain RNG state; provide "
                "random_state when resuming functional execution."
            )

        return self._runtime_from_checkpoint(
            checkpoint_obj,
            shared_context=shared_context,
            rng=restored_random_state,
        )

    @staticmethod
    def _coerce_checkpoint(
        checkpoint: SMCCheckpoint | Mapping[str, Any],
    ) -> SMCCheckpoint:
        return (
            checkpoint
            if isinstance(checkpoint, SMCCheckpoint)
            else SMCCheckpoint.from_dict(checkpoint)
        )

    @staticmethod
    def _runtime_from_checkpoint(
        checkpoint: SMCCheckpoint,
        *,
        shared_context: Mapping[str, Any] | None,
        rng: Any,
    ) -> SMCRuntime:
        states = [item.copy() for item in checkpoint.states]
        state = states[-1]
        restored_metadata = array_safe_copy(checkpoint.metadata)

        return SMCRuntime(
            state=state,
            produced_states=states,
            initial_state=(
                None
                if checkpoint.initial_state is None
                else checkpoint.initial_state.copy()
            ),
            t=checkpoint.next_iteration,
            next_observation_index=checkpoint.next_observation_index,
            shared_context=(
                shared_context if shared_context is not None else {}
            ),
            metadata=restored_metadata,
            evaluation_counts=checkpoint.evaluation_counts,
            outputs=list(array_safe_copy(checkpoint.outputs)),
            rng=rng,
        )
