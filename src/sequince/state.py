from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, MutableMapping

from ._array_api import (
    asarray,
    full,
    namespace,
)
from ._array_api import (
    copy as array_copy,
)
from ._array_api import (
    logsumexp as xp_logsumexp,
)
from ._typing import Array


@dataclass
class ParticleState:
    """Container for particle locations, log weights, and auxiliary information."""

    particles: Array
    log_weights: Array | float | int | None = None
    log_evidence: float = 0.0
    metadata: MutableMapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.log_weights, None | float | int):
            xp = namespace(self.particles)
            n = self.particles.shape[0]
            value = (
                float(self.log_weights)
                if self.log_weights is not None
                else -math.log(n)
            )
            self.log_weights = full(
                (n,), value, xp=xp, dtype=float, like=self.particles
            )

    def copy(self) -> "ParticleState":
        """Return a copy of the state preserving the array backend."""
        xp = namespace(self.particles, self.log_weights)
        return ParticleState(
            particles=array_copy(self.particles, xp=xp),
            log_weights=array_copy(self.log_weights, xp=xp),
            log_evidence=float(self.log_evidence),
            metadata=dict(self.metadata),
        )

    def normalize(self) -> tuple["ParticleState", float]:
        """Return a new state with normalized log weights and updated evidence."""
        state = self.copy()
        xp = namespace(state.log_weights)
        log_weights = asarray(state.log_weights, xp=xp)
        log_norm = xp_logsumexp(log_weights, xp=xp)
        log_norm_scalar = float(log_norm)
        if math.isfinite(log_norm_scalar):
            state.log_weights = log_weights - log_norm_scalar
            state.log_evidence += log_norm_scalar
        else:
            n = state.log_weights.shape[0]
            state.log_weights = full(
                (n,),
                -math.log(n),
                xp=xp,
                dtype=log_weights.dtype,
                like=log_weights,
            )
        return state, float(log_norm)

    @property
    def weights(self) -> Array:
        """Return normalized weights."""
        xp = namespace(self.log_weights)
        log_weights = asarray(self.log_weights, xp=xp)
        log_norm = xp_logsumexp(log_weights, xp=xp)
        log_norm_scalar = float(log_norm)
        if math.isfinite(log_norm_scalar):
            return xp.exp(log_weights - log_norm_scalar)
        n = self.log_weights.shape[0]
        return full(
            (n,), 1.0 / n, xp=xp, dtype=log_weights.dtype, like=log_weights
        )

    def effective_sample_size(self) -> float:
        """Compute the effective sample size (ESS)."""
        xp = namespace(self.log_weights)
        w = asarray(self.weights, xp=xp)
        return 1.0 / float(xp.sum(xp.square(w)))

    def with_updates(self, **kwargs: Any) -> "ParticleState":
        """Return a copy of the state with selective updates applied."""
        updated = self.copy()
        for key, value in kwargs.items():
            setattr(updated, key, value)
        return updated

    def resample(self, indices: Array) -> "ParticleState":
        """Return a resampled state using the provided ancestor indices."""
        xp = namespace(self.particles, indices)
        particles = asarray(self.particles, xp=xp)
        idx = asarray(indices, xp=xp, like=particles)
        new_particles = particles[idx]
        n = idx.shape[0]
        log_weight_template = asarray(self.log_weights, xp=xp)
        log_weights = full(
            (n,),
            -math.log(n),
            xp=xp,
            dtype=log_weight_template.dtype,
            like=log_weight_template,
        )
        return ParticleState(
            particles=new_particles,
            log_weights=log_weights,
            log_evidence=self.log_evidence,
            metadata=dict(self.metadata, ancestors=idx),
        )
