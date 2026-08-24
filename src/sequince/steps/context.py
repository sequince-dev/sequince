from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from ..utils.pytrees import pytree_dataclass

LOG_PROPOSAL_OVERRIDE_KEY = "log_proposal_override"
SMC_TRANSFORMATION_KEY = "smc_transformation"


@pytree_dataclass(
    data_fields=(
        "t",
        "next_observation_index",
        "observation",
        "shared",
        "metadata",
        "stop_requested",
        "stop_immediately",
    )
)
@dataclass(frozen=True)
class StepContext:
    """Runtime context passed to each SMC step."""

    t: int
    next_observation_index: int = 0
    observation: Any = None
    shared: Mapping[str, Any] = field(default_factory=dict)
    """Dictionary for shared user-defined context across steps.

    Should not be modified by steps directly; use ``metadata`` for step-to-step communication instead.
    """
    metadata: Mapping[str, Any] = field(default_factory=dict)
    """Dictionary for step metadata that can be used for communication between steps and callbacks."""
    stop_requested: bool = False
    stop_immediately: bool = False

    @property
    def iteration(self) -> int:
        """Alias for ``t``."""
        return self.t

    def with_metadata(self, **updates: Any) -> StepContext:
        """Return a context with copy-on-write metadata updates."""
        return replace(self, metadata={**self.metadata, **updates})

    def request_stop(self) -> StepContext:
        """Signal that the SMC runner should halt after this iteration."""
        return replace(self, stop_requested=True, stop_immediately=False)

    def request_stop_immediately(self) -> StepContext:
        """Signal that the SMC runner should halt immediately."""
        return replace(self, stop_requested=True, stop_immediately=True)

    def clear_stop_request(self) -> StepContext:
        """Clear any pending stop request."""
        return replace(self, stop_requested=False, stop_immediately=False)
