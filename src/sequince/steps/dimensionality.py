from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .._array_api import asarray, namespace
from .base import SMCStep, StepContext

if TYPE_CHECKING:
    from orng import ArrayRNG

    from .._typing import Array
    from ..state import ParticleState

logger = logging.getLogger(__name__)


@dataclass
class SMCTransformation:
    """Transformation between successive SMC targets."""

    def forward(
        self,
        state: "ParticleState",
        context: StepContext,
        rng: "ArrayRNG",
    ) -> "ParticleState":
        raise NotImplementedError

    def inverse(
        self,
        particles: "Array",
        context: StepContext,
    ) -> "Array":
        raise NotImplementedError

    def log_auxiliary_density(
        self,
        particles: "Array",
        context: StepContext,
    ) -> "Array | float":
        xp = namespace(particles)
        arr = asarray(particles, xp=xp)
        return xp.zeros(arr.shape[0], dtype=arr.dtype)

    def log_jacobian(
        self,
        particles: "Array",
        context: StepContext,
    ) -> "Array | float":
        xp = namespace(particles)
        arr = asarray(particles, xp=xp)
        return xp.zeros(arr.shape[0], dtype=arr.dtype)


class ExpandDimensions(SMCStep):
    """Apply an SMC transformation that increases the particle dimension."""

    def __init__(
        self,
        transformation: SMCTransformation,
        *,
        expected_block_size: int | None = None,
        metadata_key: str = "dimensions",
    ):
        self._transformation = transformation
        self._expected_block_size = expected_block_size
        self._metadata_key = metadata_key

    def apply(
        self,
        state: "ParticleState",
        context: StepContext,
        rng: "ArrayRNG",
    ) -> "ParticleState":
        xp = namespace(state.particles)
        current_particles = asarray(state.particles, xp=xp)
        if current_particles.ndim == 1:
            current_particles = xp.reshape(
                current_particles, (current_particles.shape[0], 1)
            )

        expanded_state = self._transformation.forward(state, context, rng)
        expanded_particles = asarray(expanded_state.particles, xp=xp)
        if expanded_particles.ndim == 1:
            expanded_particles = xp.reshape(
                expanded_particles, (expanded_particles.shape[0], 1)
            )

        if expanded_particles.shape[0] != current_particles.shape[0]:
            raise ValueError(
                "Transformation must preserve the number of particles."
            )

        delta = expanded_particles.shape[1] - current_particles.shape[1]
        if delta < 0:
            raise ValueError(
                "ExpandDimensions requires a transformation that increases "
                "or preserves dimensionality."
            )
        if (
            self._expected_block_size is not None
            and delta != self._expected_block_size
        ):
            raise ValueError(
                "Transformation changed dimensionality by "
                f"{delta} (expected {self._expected_block_size})."
            )

        total_dims = expanded_particles.shape[1]
        state_metadata = dict(expanded_state.metadata)
        state_metadata[self._metadata_key] = total_dims

        context.metadata[self._metadata_key] = total_dims
        context.metadata["added_dimensions"] = context.metadata.get(
            "added_dimensions", 0
        ) + max(delta, 0)
        context.metadata["last_dimension_delta"] = delta
        context.metadata["smc_transformation"] = self._transformation

        logger.info(
            "ExpandDimensions changed dimensionality by %d (total=%d).",
            delta,
            total_dims,
        )

        return expanded_state.with_updates(metadata=state_metadata)
