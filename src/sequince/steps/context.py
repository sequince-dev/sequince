from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from ..target import EvaluationCounts
from ..utils.pytrees import pytree_dataclass

LOG_PROPOSAL_OVERRIDE_KEY = "log_proposal_override"
EVALUATION_COUNTS_KEY = "evaluation_counts"
SMC_TRANSFORMATION_KEY = "smc_transformation"
RESULT_HDF5_OMITTED_METADATA_KEYS = frozenset(
    {
        "particle_transform",
        "_functional_particle_transform",
        SMC_TRANSFORMATION_KEY,
        LOG_PROPOSAL_OVERRIDE_KEY,
    }
)


@pytree_dataclass(
    data_fields=(
        "t",
        "next_observation_index",
        "observation",
        "shared",
        "metadata",
        "evaluation_counts",
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
    evaluation_counts: EvaluationCounts = field(
        default_factory=EvaluationCounts
    )
    stop_requested: bool = False
    stop_immediately: bool = False

    @property
    def iteration(self) -> int:
        """Alias for ``t``."""
        return self.t

    def with_metadata(self, **updates: Any) -> StepContext:
        """Return a context with copy-on-write metadata updates."""
        return replace(self, metadata={**self.metadata, **updates})

    def add_evaluation_counts(self, counts: EvaluationCounts) -> StepContext:
        """Return a context with newly incurred work added to its counters."""
        return replace(
            self,
            evaluation_counts=self.evaluation_counts + counts,
        )

    def request_stop(self) -> StepContext:
        """Signal that the SMC runner should halt after this iteration."""
        return replace(self, stop_requested=True, stop_immediately=False)

    def request_stop_immediately(self) -> StepContext:
        """Signal that the SMC runner should halt immediately."""
        return replace(self, stop_requested=True, stop_immediately=True)

    def clear_stop_request(self) -> StepContext:
        """Clear any pending stop request."""
        return replace(self, stop_requested=False, stop_immediately=False)

    def _encode_for_result_hdf5(self) -> dict[str, Any]:
        """Encode this context as a non-resumable analysis snapshot.

        Live runtime values in ``shared`` and ``observation`` are omitted, as
        are Sequince-owned transformation and proposal entries at the top level
        of ``metadata``.
        """
        from ..utils.hdf5 import schema_header

        metadata = {
            key: value
            for key, value in self.metadata.items()
            if key not in RESULT_HDF5_OMITTED_METADATA_KEYS
        }
        return {
            **schema_header("step_context", include_version=False),
            "t": self.t,
            "next_observation_index": self.next_observation_index,
            "metadata": metadata,
            "evaluation_counts": {
                "log_target_evaluations": (
                    self.evaluation_counts.log_target_evaluations
                ),
                "log_proposal_evaluations": (
                    self.evaluation_counts.log_proposal_evaluations
                ),
                "log_increment_evaluations": (
                    self.evaluation_counts.log_increment_evaluations
                ),
            },
            "stop_requested": self.stop_requested,
            "stop_immediately": self.stop_immediately,
        }

    @classmethod
    def _decode_from_result_hdf5(
        cls, payload: Mapping[str, Any]
    ) -> StepContext:
        """Decode a result analysis snapshot with safe runtime defaults."""
        from ..utils.hdf5 import validate_schema

        validate_schema(
            payload,
            kinds={"step_context"},
            required={
                "t",
                "next_observation_index",
                "metadata",
                "stop_requested",
                "stop_immediately",
            },
            optional={"evaluation_counts"},
        )
        metadata = payload["metadata"]
        if not isinstance(metadata, Mapping):
            raise TypeError("Saved StepContext metadata must be a mapping.")
        return cls(
            t=int(payload["t"]),
            next_observation_index=int(payload["next_observation_index"]),
            observation=None,
            shared={},
            metadata=dict(metadata),
            evaluation_counts=EvaluationCounts(
                **dict(payload.get("evaluation_counts", {}))
            ),
            stop_requested=bool(payload["stop_requested"]),
            stop_immediately=bool(payload["stop_immediately"]),
        )
