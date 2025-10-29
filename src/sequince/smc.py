from __future__ import annotations

from typing import (
    Any,
    Callable,
    Iterable,
    List,
    MutableMapping,
    Optional,
    Sequence,
)

import numpy as np

from .result import SMCResults
from .state import ParticleState
from .steps.base import StepContext
from .utils import _ensure_rng


class SequentialMonteCarlo:
    """Composable Sequential Monte Carlo runner.

    Users supply an ordered collection of SMC steps that will be called in sequence at
    each iteration. Steps can mutate the state or return a new one.
    """

    def __init__(
        self,
        steps: Sequence[
            Callable[
                [ParticleState, StepContext, np.random.Generator],
                ParticleState,
            ]
        ],
        auto_normalize: bool = True,
    ):
        self._steps = list(steps)
        self._auto_normalize = auto_normalize

    def run(
        self,
        initial_state: ParticleState,
        observations: Optional[Iterable[Any]] = None,
        num_steps: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        static_context: Optional[MutableMapping[str, Any]] = None,
    ) -> SMCResults:
        rng = _ensure_rng(rng)
        static_context = static_context or {}

        produced_states: List[ParticleState] = []
        state = initial_state

        observation_iterator = (
            iter(observations) if observations is not None else None
        )

        t = 0
        steps_remaining = num_steps if num_steps is not None else None

        while steps_remaining is None or steps_remaining > 0:
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

            context = StepContext(
                t=t, observation=observation, static=static_context
            )
            for step in self._steps:
                state = step(state, context, rng)
            if self._auto_normalize:
                state, _ = state.normalize()
            produced_states.append(state)

            if context.stop_requested:
                break

            t += 1

        return SMCResults(states=produced_states)
