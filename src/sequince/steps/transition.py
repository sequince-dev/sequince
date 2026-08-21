from __future__ import annotations

from typing import Any

from orng import RandomGenerator

from ..array_api import namespace
from ..filtering import ImportanceKernel
from ..state import ParticleState
from .base import SMCStep
from .context import StepContext
from .weighting import _apply_log_weight_increment, _weights_as_1d_array


class ImportanceTransition(SMCStep):
    """Propagate particles and apply a joint importance-weight correction.

    The supplied :class:`~sequince.filtering.ImportanceKernel` samples one
    child per parent and computes the associated importance log-weight
    increment from the parent and child populations.

    The transition is one-to-one: it must preserve the particle population
    size. Moving particles invalidates any cached target evaluation while
    preserving their existing ancestry indices.
    """

    def __init__(
        self,
        kernel: ImportanceKernel,
        *,
        normalize_after: bool = True,
    ) -> None:
        if not callable(getattr(kernel, "sample", None)) or not callable(
            getattr(kernel, "log_increment", None)
        ):
            raise TypeError(
                "kernel must implement sample() and log_increment()."
            )
        self._kernel = kernel
        self._normalize_after = normalize_after

    @property
    def kernel(self) -> ImportanceKernel:
        """The transition kernel applied by this step."""
        return self._kernel

    def apply(
        self,
        state: ParticleState,
        context: StepContext,
        rng: "RandomGenerator",
    ) -> tuple[ParticleState, StepContext, dict[str, Any]]:
        previous_particles = state.particles
        new_particles = self._kernel.sample(
            previous_particles,
            context,
            rng,
        )
        if new_particles.shape[0] != state.log_weights.shape[0]:
            raise ValueError(
                "Transition must return one particle per input particle."
            )

        increment = self._kernel.log_increment(
            previous_particles,
            new_particles,
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
            },
        )
