from __future__ import annotations

from typing import Any, Callable

from orng import RandomGenerator

from .._typing import Array
from ..array_api import namespace
from ..state import ParticleState
from .base import SMCStep
from .context import StepContext
from .weighting import _apply_log_weight_increment, _weights_as_1d_array

TransitionFn = Callable[..., tuple[Array, Any]]
TransitionIncrementFn = Callable[..., Array | float]


class ImportanceTransition(SMCStep):
    """Propagate particles and apply a joint importance-weight correction.

    ``transition`` is called as ``transition(previous_particles, context, rng)``
    and must return ``(new_particles, auxiliary)``. ``log_increment`` is then
    called as ``log_increment(previous_particles, new_particles, auxiliary,
    context)``. Use ``None`` as the auxiliary value when no transition
    diagnostics are needed.

    The transition is one-to-one: it must preserve the particle population
    size. Moving particles invalidates any cached target evaluation while
    preserving their existing ancestry indices.
    """

    def __init__(
        self,
        transition: TransitionFn,
        log_increment: TransitionIncrementFn,
        *,
        normalize_after: bool = True,
    ) -> None:
        if not callable(transition):
            raise TypeError("transition must be callable.")
        if not callable(log_increment):
            raise TypeError("log_increment must be callable.")
        self._transition = transition
        self._log_increment = log_increment
        self._normalize_after = normalize_after

    def apply(
        self,
        state: ParticleState,
        context: StepContext,
        rng: "RandomGenerator",
    ) -> tuple[ParticleState, StepContext, dict[str, Any]]:
        previous_particles = state.particles
        new_particles, auxiliary = self._transition(
            previous_particles,
            context,
            rng,
        )
        if new_particles.shape[0] != state.log_weights.shape[0]:
            raise ValueError(
                "Transition must return one particle per input particle."
            )

        increment = self._log_increment(
            previous_particles,
            new_particles,
            auxiliary,
            context,
        )
        xp = namespace(state.log_weights)
        increments = _weights_as_1d_array(
            increment,
            xp=xp,
            like=state.log_weights,
        )
        if increments.shape[0] != state.log_weights.shape[0]:
            raise ValueError(
                "Increment function must return one value per particle."
            )

        moved_state = state.replace_particles(new_particles)
        updated_state, _ = _apply_log_weight_increment(
            moved_state,
            increments,
            normalize=self._normalize_after,
        )
        return (
            updated_state,
            context,
            {
                "log_weight_increment": increments,
                "auxiliary": auxiliary,
            },
        )
