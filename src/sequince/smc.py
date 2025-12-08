from __future__ import annotations

import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    Iterator,
    List,
    MutableMapping,
    Sequence,
)

from ._array_api import asarray, namespace
from ._typing import Array
from .result import MetaSMCResult, SMCResult
from .state import ParticleState
from .steps.base import StepContext
from .utils import _call_with_optional_context, _ensure_rng

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
        shared_context: MutableMapping[str, Any] | None = None,
    ) -> SMCResult:
        rng = _ensure_rng(rng)
        shared_context = shared_context or {}

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
                t=t, observation=observation, shared=shared_context
            )
            for step in self._steps:
                step_name = getattr(step, "__name__", step.__class__.__name__)
                logger.debug("Iteration %d: applying step %s.", t, step_name)
                state = step(state, context, rng)
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
        return SMCResult(states=produced_states, context=context)


class MetaSequentialMonteCarlo:
    """Bridge a sequence of targets using SMC style reweighting.

    The class reweights particles by the ratio between consecutive target
    densities and optionally runs a standard ``SequentialMonteCarlo`` instance
    after each reweighting step to rejuvenate the particle set.

    Parameters
    ----------
    targets:
        An ordered sequence of callables evaluating the (unnormalized)
        log-density of each target. Each callable should accept
        ``(particles,)`` or ``(particles, context)``; the provided
        ``StepContext`` is passed when required. The incremental weight for
        stage ``k`` is ``log_target_k - log_target_(k-1)``.
    target_generator:
        Callable with signature ``(state, context)`` returning the next target
        function, or ``None`` when no further targets are available. Use this
        for open-ended or state-dependent sequences. Mutually exclusive with
        ``targets``; at least one must be provided.
    inner_runner:
        Optional ``SequentialMonteCarlo`` instance applied within each stage
        after reweighting.
    inner_steps:
        Steps used to build an internal ``SequentialMonteCarlo`` runner when
        ``inner_runner`` is not supplied.
    model_transition:
        Optional callable that mutates the particle state between stages. It is
        invoked as ``model_transition(state, shared_context, rng)`` after each
        stage completes (and before the next reweighting). This is useful for
        transdimensional workflows that need to append new latent variables or
        otherwise transform particles before evaluating the next target.
    callbacks:
        Optional callbacks attached when an internal runner is constructed.
    normalize_after_reweight:
        Whether to normalize particle weights immediately after each ratio
        update (recommended so log-evidence is accumulated correctly).
    reset_temperature_between_models:
        When ``True`` (default), reset any temperature bookkeeping in the shared
        context and particle metadata after each ``model_transition`` so users do
        not need to do this manually.
    """

    def __init__(
        self,
        targets: Sequence[Callable[..., Any]] | None = None,
        target_generator: Callable[
            [ParticleState, StepContext], Callable[..., Any] | None
        ]
        | None = None,
        *,
        inner_runner: SequentialMonteCarlo | None = None,
        inner_steps: Sequence[
            Callable[
                [ParticleState, StepContext, "ArrayRNG"],
                ParticleState,
            ]
        ]
        | None = None,
        model_transition: Callable[
            [ParticleState, MutableMapping[str, Any], "ArrayRNG"],
            ParticleState,
        ]
        | None = None,
        callbacks: Sequence[
            Callable[[ParticleState, StepContext, SMCResult], None]
        ]
        | None = None,
        normalize_after_reweight: bool = True,
        reset_temperature_between_models: bool = True,
    ):
        exclusive = sum(x is not None for x in (inner_runner, inner_steps))
        if exclusive > 1:
            raise ValueError(
                "Provide either inner_runner or inner_steps, not both."
            )

        if targets is None and target_generator is None:
            raise ValueError(
                "Provide either targets or target_generator to MetaSequentialMonteCarlo."
            )
        if targets is not None and target_generator is not None:
            raise ValueError(
                "Specify only one of targets or target_generator, not both."
            )

        if isinstance(targets, Sequence) and len(targets) == 0:
            raise ValueError(
                "MetaSequentialMonteCarlo requires at least one target."
            )

        self._targets: Sequence[Callable[..., Any]] | None = (
            list(targets) if isinstance(targets, Sequence) else None
        )
        self._next_target_fn: (
            Callable[[ParticleState, StepContext], Callable[..., Any] | None]
            | None
        ) = target_generator

        self._normalize_after_reweight = normalize_after_reweight
        self._callbacks = list(callbacks) if callbacks is not None else []
        self._model_transition: (
            Callable[
                [ParticleState, MutableMapping[str, Any], "ArrayRNG"],
                ParticleState,
            ]
            | None
        ) = model_transition
        self._reset_temperature_between_models = (
            reset_temperature_between_models
        )

        self._runner: SequentialMonteCarlo | None = None
        if inner_runner is not None:
            self._runner = inner_runner
        elif inner_steps is not None:
            self._runner = SequentialMonteCarlo(
                inner_steps,
                callbacks=callbacks,
            )

    def run(
        self,
        initial_state: ParticleState,
        *,
        rng: "ArrayRNG" | np.random.Generator | int | None = None,
        shared_context: MutableMapping[str, Any] | None = None,
        num_steps: int | Sequence[int | None] | None = None,
    ) -> MetaSMCResult:
        rng = _ensure_rng(rng)
        shared_context = shared_context or {}

        if self._targets is not None:
            logger.info(
                "Starting MetaSequentialMonteCarlo over %d targets.",
                len(self._targets),
            )
            target_iterator: Iterator[Callable[..., Any]] | None = iter(
                self._targets
            )
        else:
            logger.info(
                "Starting MetaSequentialMonteCarlo with dynamic targets."
            )
            target_iterator = None

        state = initial_state
        shared_context["meta_step"] = 0
        context = StepContext(t=0, shared=shared_context)

        current_target = self._get_next_target(
            state, context, iterator=target_iterator
        )
        if current_target is None:
            raise ValueError(
                "MetaSequentialMonteCarlo requires at least one target."
            )

        results: list[SMCResult] = []
        previous_target: Callable[..., Any] | None = None
        step_idx = 0

        while True:
            logger.debug("MetaSMC: entering step %d.", step_idx)

            if previous_target is not None:
                state = self._apply_reweight(
                    state,
                    context,
                    current_target=current_target,
                    previous_target=previous_target,
                )

            stage_steps = self._resolve_num_steps(num_steps, step_idx)
            if self._runner is not None and stage_steps != 0:
                result = self._runner.run(
                    state,
                    rng=rng,
                    shared_context=shared_context,
                    num_steps=stage_steps,
                )
                state = result.final_state
            else:
                result = SMCResult(states=[state], context=context)
                for callback in self._callbacks:
                    callback(state, context, result)

            results.append(result)

            if (
                self._targets is not None
                and step_idx >= len(self._targets) - 1
            ):
                break

            pre_transition_state = state
            shared_snapshot: MutableMapping[str, Any] | None = None
            if self._model_transition is not None:
                shared_snapshot = dict(shared_context)
                state = self._model_transition(state, shared_context, rng)
                if self._reset_temperature_between_models:
                    shared_context["temperature"] = 0.0
                    shared_context["temperature_delta"] = 0.0
                    metadata = dict(state.metadata)
                    metadata["temperature"] = 0.0
                    metadata.pop("temperature_delta", None)
                    state = state.with_updates(metadata=metadata)

            next_step_idx = step_idx + 1
            shared_context["meta_step"] = next_step_idx
            next_context = StepContext(t=0, shared=shared_context)
            next_target = self._get_next_target(
                state, next_context, iterator=target_iterator
            )
            if next_target is None:
                if (
                    self._model_transition is not None
                    and shared_snapshot is not None
                ):
                    shared_context.clear()
                    shared_context.update(shared_snapshot)
                    state = pre_transition_state
                else:
                    shared_context["meta_step"] = step_idx
                break

            previous_target = current_target
            current_target = next_target
            context = next_context
            step_idx += 1

        logger.info(
            "Finished MetaSequentialMonteCarlo; produced %d meta steps.",
            len(results),
        )
        return MetaSMCResult(results=results)

    def _apply_reweight(
        self,
        state: ParticleState,
        context: StepContext,
        *,
        current_target: Callable[..., Any],
        previous_target: Callable[..., Any],
    ) -> ParticleState:
        current_log = self._evaluate_target(current_target, state, context)
        previous_log = self._evaluate_target(previous_target, state, context)
        increments = current_log - previous_log

        xp = namespace(increments, state.log_weights)
        mean_increment = float(xp.sum(increments) / increments.shape[0])
        logger.debug(
            "MetaSMC reweighting: mean Δlogw=%.6f (step %s).",
            mean_increment,
            context.shared.get("meta_step"),
        )

        updated = state.with_updates(
            log_weights=state.log_weights + increments
        )
        if self._normalize_after_reweight:
            updated, _ = updated.normalize()

        metadata = dict(updated.metadata)
        metadata["meta_step"] = context.shared.get("meta_step")
        return updated.with_updates(metadata=metadata)

    def _evaluate_target(
        self,
        fn: Callable[..., Any],
        state: ParticleState,
        context: StepContext,
    ) -> Array:
        values = _call_with_optional_context(fn, (state.particles,), context)
        xp = namespace(values, state.particles)
        arr = asarray(values, xp=xp, like=state.log_weights)
        if arr.ndim == 0:
            arr = xp.reshape(arr, (1,))
        if arr.ndim != 1:
            arr = xp.reshape(arr, (arr.shape[0],))
        if arr.shape[0] != state.log_weights.shape[0]:
            raise ValueError(
                "Target function must return one value per particle."
            )
        return arr

    def _get_next_target(
        self,
        state: ParticleState,
        context: StepContext,
        *,
        iterator: Iterator[Callable[..., Any]] | None,
    ) -> Callable[..., Any] | None:
        if self._next_target_fn is not None:
            return self._next_target_fn(state, context)

        if iterator is None:
            raise RuntimeError("No target source configured.")

        try:
            return next(iterator)
        except StopIteration:
            return None

    @staticmethod
    def _resolve_num_steps(
        num_steps: int | Sequence[int | None] | None,
        stage_idx: int,
    ) -> int | None:
        if num_steps is None:
            return None
        if isinstance(num_steps, int):
            return num_steps
        try:
            return num_steps[stage_idx]
        except IndexError as exc:
            raise ValueError(
                "num_steps sequence is shorter than the number of targets."
            ) from exc
