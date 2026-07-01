from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from frozendict import frozendict

from .._typing import MetaData, Outputs, SharedContext

if TYPE_CHECKING:
    pass


LOG_PROPOSAL_OVERRIDE_KEY = "log_proposal_override"
SMC_TRANSFORMATION_KEY = "smc_transformation"


@dataclass
class StepContext:
    """Runtime context passed to each SMC step."""

    t: int
    observation: Any = None
    shared: SharedContext = field(default_factory=dict)
    """Dictionary for shared user-defined context across steps.

    Should not be modified by steps directly; use ``metadata`` for step-to-step communication instead.
    """
    metadata: MetaData = field(default_factory=dict)
    """Dictionary for step metadata that can be used for communication between steps and callbacks."""
    outputs: Outputs = field(default_factory=dict)
    """Dictionary for storing outputs from steps."""

    @property
    def iteration(self) -> int:
        """Alias for ``t``."""
        return self.t

    def __post_init__(self):
        if "stop_requested" not in self.metadata:
            self.metadata["stop_requested"] = False
        if "stop_immediately" not in self.metadata:
            self.metadata["stop_immediately"] = False
        self.shared = frozendict(self.shared)

    def request_stop(self) -> None:
        """Signal that the SMC runner should halt after this iteration."""
        self.metadata["stop_requested"] = True
        self.metadata["stop_immediately"] = False

    def request_stop_immediately(self) -> None:
        """Signal that the SMC runner should halt immediately."""
        self.metadata["stop_requested"] = True
        self.metadata["stop_immediately"] = True

    def clear_stop_request(self) -> None:
        """Clear any pending stop request."""
        self.metadata["stop_requested"] = False
        self.metadata["stop_immediately"] = False

    def record_output(self, key: str, value: Any) -> None:
        """Store an output value that can be accessed by callbacks."""
        self.outputs.setdefault(key, []).append(value)
