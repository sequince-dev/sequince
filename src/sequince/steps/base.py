from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from orng import RandomGenerator

from ..array_api import namespace
from ..typing import Array
from ..utils.random import resolve_rng
from .context import StepContext

if TYPE_CHECKING:
    from orng import RandomGenerator

    from ..smc import SMCConfig
    from ..state import ParticleState
    from .mutation.metrics import MutationMetrics, MutationStepState


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


@runtime_checkable
class DynamicBudgetMutation(Protocol):
    """Functional mutation with a runtime budget and static history capacity.

    The scalar integer array budget must lie in [0, max_steps]. Initialize the
    functional state/context and metadata before tracing. The returned metric
    state continues an enclosing block; only its owner finalizes metrics.
    ``random_state`` is the backend-native state returned by
    ``orng.RandomGenerator.state()``; for JAX it is a PRNG key array. Functional
    calls accept and return that state, while eager methods accept the ORNG
    generator object. Configuration validation accepts an optional ``max_steps``
    override so enclosing adapters validate the effective history capacity rather
    than the child's default sweep count.
    """

    @property
    def metrics(self) -> MutationMetrics: ...

    def initialize_functional_metadata(
        self, state: ParticleState, metadata: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...

    def initialize_functional_state_and_context(
        self, state: ParticleState, context: StepContext
    ) -> tuple[ParticleState, StepContext]: ...

    def validate_functional_configuration(
        self,
        config: SMCConfig,
        state: ParticleState,
        *,
        max_steps: int | None = None,
    ) -> None: ...

    def apply_dynamic_budget(
        self,
        state: ParticleState,
        context: StepContext,
        random_state: Any,
        *,
        n_steps: Array,
        max_steps: int,
        mutation_state: MutationStepState | None = None,
    ) -> tuple[ParticleState, StepContext, Any, Any, MutationStepState]: ...


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
        random state is explicitly threaded through the computation. Pass the
        backend-native state returned by ``orng.RandomGenerator.state()``;
        for JAX this is a PRNG key array. An ORNG wrapper is reconstructed
        internally, and its advanced state is returned to the caller.
        """
        rng = RandomGenerator.from_xp(
            namespace(random_state),
            generator=random_state,
        )
        state, context, output = self.apply(state, context, rng)
        new_random_state = rng.state()
        return state, context, new_random_state, output

    def validate_functional_configuration(
        self, config: SMCConfig, state: ParticleState
    ) -> None:
        """Validate host-side pipeline settings before initializing any step.

        Parameters
        ----------
        config : SMCConfig
            Read-only runner configuration, including the complete pipeline.
        state : ParticleState
            Initial particle state, including the population shape.

        Notes
        -----
        Override to reject incompatible pipeline settings. Metadata and cache
        preparation belong in the existing functional initialization hooks.
        """

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
