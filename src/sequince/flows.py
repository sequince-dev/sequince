"""Flow interfaces used by Sequince transforms and proposals."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, Protocol

from ._typing import Array

if TYPE_CHECKING:
    from .steps.context import StepContext

FLOW_FIT_OUTPUT_KEY = "flow_fit"


class NormalizingFlow(Protocol):
    """Minimal interface for normalizing flows used by Sequince."""

    xp: Any

    def fit(
        self,
        particles: Array,
        context: StepContext | None = None,
        **kwargs: Any,
    ) -> "NormalizingFlow":
        """Adapt the flow in place.

        Implementations may record fit diagnostics in ``context.outputs`` under
        :data:`FLOW_FIT_OUTPUT_KEY`.
        """
        ...

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
        self, num_samples: int = 1, context: StepContext | None = None
    ) -> Array:
        """Draw samples from the flow distribution."""
        ...


def flow_asarray(flow: NormalizingFlow, array: Any):
    """Cast ``array`` to the backend array type expected by ``flow``."""
    return flow.xp.asarray(array)


def record_flow_fit_output(
    context: "StepContext | None",
    diagnostics: dict[str, Any],
    *,
    key: str = FLOW_FIT_OUTPUT_KEY,
) -> dict[str, Any]:
    """Record flow-fit diagnostics in ``context.outputs`` when available."""
    if context is None:
        return diagnostics
    record_output = getattr(context, "record_output", None)
    if callable(record_output):
        record_output(key, diagnostics)
        return diagnostics
    outputs = getattr(context, "outputs", None)
    if outputs is not None:
        outputs.setdefault(key, []).append(diagnostics)
    return diagnostics


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
