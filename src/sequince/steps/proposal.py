from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .base import SMCStep, StepContext

if TYPE_CHECKING:
    from orng import RandomGenerator

    from ..state import ParticleState


@dataclass(frozen=True)
class AdaptGaussianProposal(SMCStep):
    """Adapt Gaussian proposal parameters between SMC steps.

    This step leaves the particle state untouched while updating entries inside
    ``StepContext.metadata`` so downstream proposal kernels can react to the
    latest diagnostics.

    Parameters
    ----------
    scale_key:
        Name of the entry in ``context.metadata`` storing the scalar proposal
        scale. If the key is missing the ``initial_scale`` is injected.
    initial_scale:
        Default scale used when the key is absent.
    target_acceptance:
        Desired acceptance probability used to adapt the scale. Set to ``None``
        to disable acceptance-based adaptation.
    adaptation_rate:
        Multiplicative learning rate applied to the scale update. Values around
        ``0.05``–``0.2`` typically work well.
    min_scale, max_scale:
        Bounds applied to the adapted scale to guard against degeneracy.
    acceptance_metadata_key:
        Metadata field read from the particle state for the last acceptance
        statistic (defaults to ``"waste_free_acceptance"``).
    acceptance_context_key:
        Fallback key looked up in ``context.metadata`` when the metadata entry is
        missing.
    scale_metadata_key:
        Optional metadata entry that mirrors the adapted scale for inspection.
    covariance_key:
        When supplied, the empirical covariance of the current particles is
        stored at this ``context.metadata`` entry after applying diagonal jitter.
    mean_key:
        Optional entry saving the empirical particle mean alongside the
        covariance estimate.
    covariance_jitter:
        Diagonal jitter added to the covariance estimate for numerical
        stability.
    """

    scale_key: str = "waste_free_scale"
    initial_scale: float = 1.0
    target_acceptance: float | None = 0.3
    adaptation_rate: float = 0.1
    min_scale: float = 1e-3
    max_scale: float = 10.0
    acceptance_metadata_key: str = "waste_free_acceptance"
    acceptance_context_key: str | None = "waste_free_acceptance"
    scale_metadata_key: str | None = "waste_free_scale"
    covariance_key: str | None = None
    mean_key: str | None = None
    covariance_jitter: float = 1e-6

    def apply(
        self,
        state: "ParticleState",
        context: StepContext,
        rng: "RandomGenerator",
    ) -> tuple["ParticleState", StepContext, dict[str, object]]:
        del rng  # Step does not require randomness.

        scale = float(context.metadata.get(self.scale_key, self.initial_scale))
        acceptance = state.metadata.get(self.acceptance_metadata_key)
        if acceptance is None and self.acceptance_context_key is not None:
            acceptance = context.metadata.get(self.acceptance_context_key)
        if (
            acceptance is not None
            and self.target_acceptance is not None
            and np.isfinite(acceptance)
        ):
            delta = float(acceptance) - float(self.target_acceptance)
            scale *= float(np.exp(self.adaptation_rate * delta))
        scale = float(np.clip(scale, self.min_scale, self.max_scale))
        context_updates: dict[str, object] = {self.scale_key: scale}

        updated_state = state
        if self.scale_metadata_key is not None:
            metadata = dict(state.metadata)
            metadata[self.scale_metadata_key] = scale
            updated_state = state.with_updates(metadata=metadata)

        if self.covariance_key is not None:
            particles = np.asarray(state.particles, dtype=float)
            particles_2d = (
                particles[:, np.newaxis]
                if particles.ndim == 1
                else particles.reshape(particles.shape[0], -1)
            )
            n, dim = particles_2d.shape
            if n <= 1:
                cov = np.eye(dim, dtype=float)
                mean = (
                    particles_2d[0] if n == 1 else np.zeros(dim, dtype=float)
                )
            else:
                centered = particles_2d - np.mean(particles_2d, axis=0)
                cov = centered.T @ centered / max(n - 1, 1)
                mean = np.mean(particles_2d, axis=0)
            cov = np.asarray(cov, dtype=float)
            if not np.all(np.isfinite(cov)):
                cov = np.eye(dim, dtype=float)
            cov += float(self.covariance_jitter) * np.eye(dim, dtype=float)
            context_updates[self.covariance_key] = cov
            if self.mean_key is not None:
                context_updates[self.mean_key] = mean

        context = context.with_metadata(**context_updates)
        return updated_state, context, context_updates
