from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from orng import RandomGenerator

from ..array_api import namespace
from ..utils.random import resolve_rng
from .context import StepContext

if TYPE_CHECKING:
    from orng import RandomGenerator

    from ..state import ParticleState


def validate_n_steps(n_steps: int) -> int:
    """Validate a caller-supplied fixed mutation length."""
    if isinstance(n_steps, bool) or not isinstance(n_steps, Integral):
        raise TypeError("n_steps must be an integer.")
    value = int(n_steps)
    if value < 0:
        raise ValueError("n_steps must be >= 0.")
    return value


@runtime_checkable
class FixedLengthMutation(Protocol):
    """Capability for mutation-like steps accepting a per-call transition count.

    ``n_steps`` is a host-side integer. Backends such as JAX therefore require it
    to be static while tracing or compiling a call.
    """

    def apply_with_n_steps(
        self,
        state: "ParticleState",
        context: StepContext,
        rng: Any,
        *,
        n_steps: int,
    ) -> tuple["ParticleState", StepContext, Any]: ...


class SMCStep:
    """Base class for steps with explicit context and output returns.

    Functional steps request immediate stopping through StepContext. The runner
    conditionally skips later steps, which must have initialized, fixed-structure
    functional carries under JAX.
    """

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
        """Apply a functional step to particle step.

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
