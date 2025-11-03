from __future__ import annotations

import inspect
import logging
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from .base import SMCStep, StepContext

if TYPE_CHECKING:
    from ..state import ParticleState


ComponentSampler = Callable[..., np.ndarray]

logger = logging.getLogger(__name__)


class DimensionalityExpansion(SMCStep):
    """Append new latent dimensions to each particle.

    The supplied ``component_sampler`` is responsible for generating the new
    coordinates for every particle.  The sampler receives the current particle
    state as its first positional argument and can optionally request the
    calling ``StepContext`` and the RNG via keyword parameters named
    ``context`` and ``rng`` respectively.

    Parameters
    ----------
    component_sampler:
        Callable returning an ``(n_particles, k)`` array containing the new
        coordinates to append.  A 1-D array of length ``n_particles`` is also
        accepted and treated as ``k == 1``.
    expected_block_size:
        Optional integrity check enforcing the number of appended dimensions.
    metadata_key:
        Metadata entry updated with the total dimensionality after expansion.
    context_key:
        Optional ``StepContext.shared`` entry mirroring the current
        dimensionality for downstream steps.
    context_delta_key:
        Optional ``StepContext.shared`` entry receiving the number of
        dimensions added by the latest expansion.
    reset_temperature:
        When ``True`` (default) the step resets any temperature bookkeeping in
        the particle metadata and shared context to zero, enabling a fresh
        annealing cycle after expanding the latent space.
    """

    def __init__(
        self,
        component_sampler: ComponentSampler,
        *,
        expected_block_size: int | None = None,
        metadata_key: str = "dimensions",
        context_key: str | None = "dimensions",
        context_delta_key: str | None = "dimension_delta",
        reset_temperature: bool = True,
    ):
        self._component_sampler = component_sampler
        self._expected_block_size = expected_block_size
        self._metadata_key = metadata_key
        self._context_key = context_key
        self._context_delta_key = context_delta_key
        self._reset_temperature = reset_temperature

    def apply(
        self,
        state: ParticleState,
        context: StepContext,
        rng: np.random.Generator,
    ) -> ParticleState:
        current_particles = np.asarray(state.particles)
        if current_particles.ndim == 1:
            current_particles = current_particles[:, np.newaxis]

        new_block = self._invoke_sampler(state, context, rng)
        new_block = np.asarray(new_block)
        if new_block.ndim == 1:
            new_block = new_block[:, np.newaxis]

        if new_block.shape[0] != current_particles.shape[0]:
            raise ValueError(
                "Component sampler must return one row per particle."
            )
        if self._expected_block_size is not None:
            logger.debug(
                "DimensionalityExpansion: expecting block size %d.",
                self._expected_block_size,
            )
            if new_block.shape[1] != self._expected_block_size:
                raise ValueError(
                    "Component sampler returned unexpected block size: "
                    f"{new_block.shape[1]} (expected "
                    f"{self._expected_block_size})."
                )

        expanded_particles = np.concatenate(
            [current_particles, new_block], axis=1
        )
        metadata = dict(state.metadata)
        total_dims = expanded_particles.shape[1]
        metadata[self._metadata_key] = total_dims
        metadata["added_dimensions"] = (
            metadata.get("added_dimensions", 0) + new_block.shape[1]
        )
        metadata["last_dimension_delta"] = new_block.shape[1]

        if self._reset_temperature:
            metadata["temperature"] = 0.0
            metadata.pop("temperature_delta", None)
            context.shared["temperature"] = 0.0
            context.shared["temperature_delta"] = 0.0
            log_evidence = 0.0
            logger.debug(
                "DimensionalityExpansion reset temperature after adding %d dimensions.",
                new_block.shape[1],
            )
        else:
            log_evidence = state.log_evidence

        if self._context_key is not None:
            context.shared[self._context_key] = total_dims
        if self._context_delta_key is not None:
            context.shared[self._context_delta_key] = new_block.shape[1]

        logger.info(
            "DimensionalityExpansion appended %d dims (total=%d).",
            new_block.shape[1],
            total_dims,
        )

        return state.with_updates(
            particles=expanded_particles,
            metadata=metadata,
            log_evidence=log_evidence,
        )

    def _invoke_sampler(
        self,
        state: ParticleState,
        context: StepContext,
        rng: np.random.Generator,
    ) -> np.ndarray:
        sampler = self._component_sampler
        signature = inspect.signature(sampler)
        params = signature.parameters

        kwargs: dict[str, Any] = {}
        if "context" in params:
            kwargs["context"] = context
        if "rng" in params:
            kwargs["rng"] = rng

        positional_params = [
            param
            for param in params.values()
            if param.kind
            in {
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            }
        ]

        args: list[Any] = []
        if positional_params:
            first_name = positional_params[0].name
            if first_name in {"particles", "coords", "positions"}:
                args.append(state.particles)
            else:
                args.append(state)

        logger.debug(
            "Invoking component sampler %s for dimensionality expansion.",
            getattr(sampler, "__name__", sampler.__class__.__name__),
        )
        return sampler(*args, **kwargs)
