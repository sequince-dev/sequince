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

    def apply_functional(
        self,
        state: "ParticleState",
        context: StepContext,
        random_state: Any,
    ) -> tuple["ParticleState", StepContext, Any, Any]:
        """Apply the functional step that.

        This returns the updated state, context, next random state, and any
        additional outputs.

        This method is intended for use in functional SMC iterations where the
        random state is explicitly threaded through the computation.
        """
        rng = RandomGenerator.from_xp(
            namespace(random_state),
            generator=random_state,
        )
        state, context, output = self.apply(state, context, rng)
        new_random_state = rng.state()
        return state, context, new_random_state, output

    def initialize_functional_metadata(
        self,
        state: "ParticleState",
        metadata: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Return metadata with fields required by ``apply_functional``."""
        del state
        return metadata

    def initialize_functional_state_and_context(
        self,
        state: "ParticleState",
        context: StepContext,
    ) -> tuple["ParticleState", StepContext]:
        """Return shape-complete inputs for ``apply_functional``.

        Steps whose functional output adds array leaves should initialize
        matching leaves here so one compiled signature can be reused from the
        first iteration onward.
        """
        return state, context

    def state_dict(self) -> dict[str, Any]:
        """Return mutable state required by subsequent SMC iterations."""
        return {}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore state produced by :meth:`state_dict`."""
        if state:
            raise ValueError(
                f"{type(self).__name__} does not support checkpoint state."
            )
