from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any,
    MutableMapping,
)

import numpy as np
from scipy.special import logsumexp


@dataclass
class ParticleState:
    """Container for particle locations, log weights, and auxiliary information."""

    particles: np.ndarray
    log_weights: np.ndarray
    log_evidence: float = 0.0
    metadata: MutableMapping[str, Any] = field(default_factory=dict)

    def copy(self) -> "ParticleState":
        """Return a copy of the state."""
        return ParticleState(
            particles=np.array(self.particles, copy=True),
            log_weights=np.array(self.log_weights, copy=True),
            log_evidence=float(self.log_evidence),
            metadata=dict(self.metadata),
        )

    def normalize(self) -> tuple["ParticleState", float]:
        """Return a new state with normalized log weights and updated evidence."""
        state = self.copy()
        log_norm = logsumexp(state.log_weights)
        if np.isfinite(log_norm):
            state.log_weights = state.log_weights - log_norm
            state.log_evidence += log_norm
        else:
            n = state.log_weights.shape[0]
            state.log_weights = np.full(n, -np.log(n))
        return state, log_norm

    @property
    def weights(self) -> np.ndarray:
        """Return normalized weights."""
        log_norm = logsumexp(self.log_weights)
        if np.isfinite(log_norm):
            return np.exp(self.log_weights - log_norm)
        n = self.log_weights.shape[0]
        return np.full(n, 1.0 / n)

    def effective_sample_size(self) -> float:
        """Compute the effective sample size (ESS)."""
        w = self.weights
        return 1.0 / np.sum(np.square(w))

    def with_updates(self, **kwargs: Any) -> "ParticleState":
        """Return a copy of the state with selective updates applied."""
        updated = self.copy()
        for key, value in kwargs.items():
            setattr(updated, key, value)
        return updated

    def resample(self, indices: np.ndarray) -> "ParticleState":
        """Return a resampled state using the provided ancestor indices."""
        new_particles = np.asarray(self.particles)[indices]
        n = len(indices)
        log_weights = np.full(n, -np.log(n), dtype=float)
        return ParticleState(
            particles=new_particles,
            log_weights=log_weights,
            log_evidence=self.log_evidence,
            metadata=dict(self.metadata, ancestors=indices),
        )
