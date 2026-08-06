from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from orng import RandomGenerator

from ..array_api import namespace
from ..utils import resolve_rng
from .context import StepContext

if TYPE_CHECKING:
    from orng import RandomGenerator

    from ..state import ParticleState


class SMCStep:
    """Base class for steps with explicit context and output returns."""

    def __call__(
        self,
        state: "ParticleState",
        context: StepContext,
        rng: "RandomGenerator" | int | None = None,
    ) -> tuple["ParticleState", StepContext, Any]:
        return self.apply(state, context, resolve_rng(rng))

    def apply(
        self,
        state: "ParticleState",
        context: StepContext,
        rng: "RandomGenerator",
    ) -> tuple["ParticleState", StepContext, Any]:
        raise NotImplementedError("SMCStep subclasses must implement apply().")

    def functional_step(
        self,
        state: "ParticleState",
        context: StepContext,
        random_state: Any,
    ) -> tuple["ParticleState", StepContext, Any, Any]:
        """Apply the step while explicitly threading backend random state.

        The generator is local to this call, making the external transition
        pure when :meth:`apply` is itself free of mutation. Backend selection is
        resolved from the random state's array namespace during tracing. This
        RNG boundary is suitable for JAX transformations, but it does not make
        Python control flow over tracers, host scalar conversions, mutation,
        dynamic dictionaries or outputs, shape-changing operations, non-JAX
        third-party code, host logging or callbacks, traced-value exceptions,
        or NumPy operations on tracers JIT-compatible.
        """
        rng = RandomGenerator.from_xp(
            namespace(random_state),
            generator=random_state,
        )
        state, context, output = self.apply(state, context, rng)
        new_random_state = rng.state()
        return state, context, new_random_state, output

    def state_dict(self) -> dict[str, Any]:
        """Return mutable state required by subsequent SMC iterations."""
        return {}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore state produced by :meth:`state_dict`."""
        if state:
            raise ValueError(
                f"{type(self).__name__} does not support checkpoint state."
            )
