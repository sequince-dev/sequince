from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from .._array_api import asarray, namespace
from ..utils import _call_with_optional_context
from .base import SMCStep
from .context import (
    LOG_PROPOSAL_OVERRIDE_KEY,
    SMC_TRANSFORMATION_KEY,
    StepContext,
)

if TYPE_CHECKING:
    from orng import ArrayRNG

    from .._typing import Array
    from ..posterior import TemperedPosterior
    from ..state import ParticleState

logger = logging.getLogger(__name__)

MetadataUpdateFn = Callable[
    ["ParticleState", StepContext, "ArrayRNG"],
    dict[str, Any] | None,
]


@dataclass
class SMCTransformation:
    """Transformation between successive SMC targets."""

    def forward(
        self,
        state: "ParticleState",
        context: StepContext,
        rng: "ArrayRNG",
    ) -> "ParticleState":
        return state

    def inverse(
        self,
        particles: "Array",
        context: StepContext,
    ) -> "Array":
        return particles

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


@dataclass
class MetadataTransformation(SMCTransformation):
    """Same-space transformation that only updates metadata."""

    state_metadata_fn: MetadataUpdateFn | None = None
    context_metadata_fn: MetadataUpdateFn | None = None

    def forward(
        self,
        state: "ParticleState",
        context: StepContext,
        rng: "ArrayRNG",
    ) -> "ParticleState":
        state_metadata = dict(state.metadata)
        if self.state_metadata_fn is not None:
            state_updates = self.state_metadata_fn(state, context, rng) or {}
            state_metadata.update(state_updates)
        if self.context_metadata_fn is not None:
            context_updates = (
                self.context_metadata_fn(state, context, rng) or {}
            )
            context.metadata.update(context_updates)
        return state.with_updates(metadata=state_metadata)


class TransformationStep(SMCStep):
    """Apply a general SMC transformation and expose it to later steps."""

    def __init__(
        self,
        transformation: SMCTransformation,
        *,
        posterior: "TemperedPosterior | None" = None,
        transformation_key: str = SMC_TRANSFORMATION_KEY,
        log_proposal_override_key: str = LOG_PROPOSAL_OVERRIDE_KEY,
    ):
        self._transformation = transformation
        self._posterior = posterior
        self._transformation_key = transformation_key
        self._log_proposal_override_key = log_proposal_override_key

    def apply(
        self,
        state: "ParticleState",
        context: StepContext,
        rng: "ArrayRNG",
    ) -> "ParticleState":
        transformed_state = self._transformation.forward(state, context, rng)
        context.metadata[self._transformation_key] = self._transformation
        if self._posterior is not None:
            base_log_proposal = context.metadata.get(
                self._log_proposal_override_key,
                self._posterior.log_proposal,
            )
            if base_log_proposal is not None:
                transformation = self._transformation

                def _transformed_log_proposal(
                    particles,
                    inner_context,
                ):
                    old_particles = transformation.inverse(
                        particles, inner_context
                    )
                    values = _call_with_optional_context(
                        transformation.log_auxiliary_density,
                        (particles,),
                        inner_context,
                    )
                    base_values = _call_with_optional_context(
                        base_log_proposal,
                        (old_particles,),
                        inner_context,
                    )
                    jacobian = _call_with_optional_context(
                        transformation.log_jacobian,
                        (particles,),
                        inner_context,
                    )
                    return base_values + values - jacobian

                context.metadata[self._log_proposal_override_key] = (
                    _transformed_log_proposal
                )
        return transformed_state


class ExpandDimensions(TransformationStep):
    """Apply an SMC transformation that increases the particle dimension."""

    def __init__(
        self,
        transformation: SMCTransformation,
        *,
        expected_block_size: int | None = None,
        metadata_key: str = "dimensions",
        added_dimensions_key: str = "added_dimensions",
        dimension_delta_key: str = "last_dimension_delta",
        posterior: "TemperedPosterior | None" = None,
        transformation_key: str = SMC_TRANSFORMATION_KEY,
        log_proposal_override_key: str = LOG_PROPOSAL_OVERRIDE_KEY,
    ):
        super().__init__(
            transformation,
            posterior=posterior,
            transformation_key=transformation_key,
            log_proposal_override_key=log_proposal_override_key,
        )
        self._expected_block_size = expected_block_size
        self._metadata_key = metadata_key
        self._added_dimensions_key = added_dimensions_key
        self._dimension_delta_key = dimension_delta_key

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

        expanded_state = super().apply(state, context, rng)
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
        state_metadata[self._added_dimensions_key] = state_metadata.get(
            self._added_dimensions_key, 0
        ) + max(delta, 0)
        state_metadata[self._dimension_delta_key] = delta

        context.metadata[self._metadata_key] = total_dims
        context.metadata[self._added_dimensions_key] = context.metadata.get(
            self._added_dimensions_key, 0
        ) + max(delta, 0)
        context.metadata[self._dimension_delta_key] = delta

        logger.info(
            "ExpandDimensions changed dimensionality by %d (total=%d).",
            delta,
            total_dims,
        )

        return expanded_state.with_updates(metadata=state_metadata)
