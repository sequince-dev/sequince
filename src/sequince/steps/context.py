from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

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

    def __post_init__(self):
        if "stop_requested" not in self.metadata:
            self.metadata["stop_requested"] = False

    def request_stop(self) -> None:
        """Signal that the SMC runner should halt after this iteration."""
        self.metadata["stop_requested"] = True

    def record_output(self, key: str, value: Any) -> None:
        """Store an output value that can be accessed by callbacks."""
        self.outputs.setdefault(key, []).append(value)
