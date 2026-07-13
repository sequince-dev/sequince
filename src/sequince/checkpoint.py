from __future__ import annotations

import copy
import pickle
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from .result import SMCResult
from .state import ParticleState
from .steps.context import StepContext
from .utils import restore_rng, snapshot_rng


@dataclass(frozen=True)
class SMCCheckpoint:
    """Snapshot of the SMC runtime for checkpointing and resuming."""

    state: ParticleState
    context: StepContext
    result: SMCResult
    rng_state: dict[str, Any] | None = None

    @classmethod
    def from_runtime(
        cls,
        state: ParticleState,
        context: StepContext,
        result: SMCResult,
        rng: Any | None = None,
    ) -> "SMCCheckpoint":
        """Capture the current SMC runtime state."""
        context_copy = StepContext.from_snapshot(context.snapshot())
        state_copy = ParticleState.from_snapshot(state.snapshot())
        result_copy = SMCResult(
            states=[
                ParticleState.from_snapshot(result_state.snapshot())
                for result_state in result.states
            ],
            context=context_copy,
        )
        return cls(
            state=state_copy,
            context=context_copy,
            result=result_copy,
            rng_state=None if rng is None else snapshot_rng(rng),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a plain Python mapping representation of the checkpoint."""
        return {
            "state": self.state.snapshot(),
            "context": self.context.snapshot(),
            "result": self.result.snapshot(),
            "rng_state": copy.deepcopy(self.rng_state),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SMCCheckpoint":
        """Reconstruct a checkpoint from :meth:`to_dict` output."""
        if not isinstance(payload, Mapping):
            raise TypeError("SMCCheckpoint payload must be a mapping.")
        missing = {
            key for key in ("state", "context", "result") if key not in payload
        }
        if missing:
            raise ValueError(
                "SMCCheckpoint payload is missing required keys: "
                + ", ".join(sorted(missing))
            )
        rng_state = payload.get("rng_state")
        if rng_state is not None and not isinstance(rng_state, Mapping):
            raise TypeError(
                "SMCCheckpoint 'rng_state' must be a mapping or None."
            )
        return cls(
            state=ParticleState.from_snapshot(copy.deepcopy(payload["state"])),
            context=StepContext.from_snapshot(
                copy.deepcopy(payload["context"])
            ),
            result=SMCResult.from_snapshot(copy.deepcopy(payload["result"])),
            rng_state=(
                None if rng_state is None else copy.deepcopy(dict(rng_state))
            ),
        )

    def restore_rng(self) -> Any | None:
        """Return a live RNG restored from the stored RNG snapshot."""
        if self.rng_state is None:
            return None
        return restore_rng(self.rng_state)

    def save_hdf5(
        self,
        h5_file,
        path: str = "checkpoint",
        dsetname: str = "state",
        protocol: int | None = None,
    ) -> None:
        """Save the checkpoint to an HDF5 dataset as a pickled blob."""
        protocol = (
            pickle.HIGHEST_PROTOCOL if protocol is None else int(protocol)
        )
        group = h5_file.require_group(path)
        if dsetname in group:
            del group[dsetname]
        payload = pickle.dumps(self.to_dict(), protocol=protocol)
        group.create_dataset(dsetname, data=np.frombuffer(payload, dtype="u1"))

    @classmethod
    def load_hdf5(
        cls,
        h5_file,
        path: str = "checkpoint",
        dsetname: str = "state",
    ) -> "SMCCheckpoint":
        """Load a checkpoint from a pickled HDF5 dataset."""
        payload = h5_file[path][dsetname][...].tobytes()
        return cls.from_dict(pickle.loads(payload))
