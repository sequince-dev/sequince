"""Flow interfaces used by Sequince transforms and proposals."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Protocol

from ._typing import Array

if TYPE_CHECKING:
    from orng import RandomGenerator

    from .steps.context import StepContext


class NormalizingFlow(Protocol):
    """Minimal interface for normalizing flows used by Sequince."""

    xp: Any

    def fit(
        self,
        particles: Array,
        context: StepContext | None = None,
        rng: RandomGenerator | None = None,
        **kwargs: Any,
    ) -> "NormalizingFlow": ...

    def forward(
        self, particles: Array, context: StepContext | None = None
    ) -> tuple[Array, Array]:
        """Map particles into a transformed space and return log-Jacobians."""
        ...

    def inverse(
        self, particles: Array, context: StepContext | None = None
    ) -> tuple[Array, Array]:
        """Undo the transform applied by :meth:`forward`."""
        ...

    def log_prob(
        self, particles: Array, context: StepContext | None = None
    ) -> Array:
        """Evaluate the flow log-density in the original particle space."""
        ...

    def sample(
        self,
        num_samples: int = 1,
        context: StepContext | None = None,
        rng: RandomGenerator | None = None,
    ) -> Array:
        """Draw samples from the flow distribution."""
        ...


def flow_asarray(flow: NormalizingFlow, array: Any):
    """Cast ``array`` to the backend array type expected by ``flow``."""
    return flow.xp.asarray(array)


def flow_state_dict(flow: NormalizingFlow) -> dict[str, Any]:
    """Return persistent flow state when the adapter exposes it."""
    state_fn = getattr(flow, "state_dict", None)
    if state_fn is None:
        return {}
    state = state_fn()
    if not isinstance(state, Mapping):
        raise TypeError("Flow state_dict() must return a mapping.")
    return copy.deepcopy(dict(state))


def load_flow_state_dict(
    flow: NormalizingFlow, state: Mapping[str, Any]
) -> None:
    """Restore persistent flow state when supported by the adapter."""
    load_fn = getattr(flow, "load_state_dict", None)
    if load_fn is None:
        if state:
            raise ValueError(
                f"{type(flow).__name__} does not support checkpoint state."
            )
        return
    load_fn(copy.deepcopy(dict(state)))


def clone_flow(flow: NormalizingFlow) -> NormalizingFlow:
    """Return a fresh copy of ``flow``.

    External flow adapters can expose ``_init_args`` to control how the clone is
    reconstructed; otherwise ``deepcopy`` is used as a fallback.
    """

    init_args = getattr(flow, "_init_args", None)
    if isinstance(init_args, dict):
        cloned_args = copy.deepcopy(dict(init_args))
        extra_kwargs = cloned_args.pop("kwargs", None)
        if extra_kwargs is not None:
            if not isinstance(extra_kwargs, dict):
                raise TypeError(
                    "Expected flow._init_args['kwargs'] to be a mapping."
                )
            cloned_args = {**extra_kwargs, **cloned_args}
        return flow.__class__(**cloned_args)
    return copy.deepcopy(flow)
