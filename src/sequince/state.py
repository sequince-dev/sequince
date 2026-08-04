from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Real
from typing import TYPE_CHECKING, Any, MutableMapping

from ._typing import Array
from .array_api import (
    asarray,
    full,
    namespace,
)
from .array_api import (
    copy as array_copy,
)
from .array_api import (
    logsumexp as xp_logsumexp,
)

if TYPE_CHECKING:
    from .target import TargetEvaluation


@dataclass(init=False)
class ParticleState:
    """Container for particle locations, log weights, and auxiliary information."""

    particles: Array
    log_weights: Array
    log_evidence: float = 0.0
    evaluation: TargetEvaluation | None = None
    metadata: MutableMapping[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        particles: Array,
        log_weights: Array | Real | None = None,
        log_evidence: float = 0.0,
        evaluation: TargetEvaluation | None = None,
        metadata: MutableMapping[str, Any] | None = None,
    ) -> None:
        if len(particles) == 0:
            raise ValueError("Particles array cannot be empty.")

        self.particles = particles
        self.log_evidence = log_evidence
        self.evaluation = evaluation
        self.metadata = dict(metadata or {})
        if (
            evaluation is not None
            and evaluation.n_particles != particles.shape[0]
        ):
            raise ValueError(
                "Target evaluation must match the particle population."
            )

        if log_weights is None or isinstance(log_weights, Real):
            xp = namespace(self.particles)
            n = self.particles.shape[0]
            value = (
                float(log_weights) if log_weights is not None else -math.log(n)
            )
            log_weights = full(
                (n,), value, xp=xp, dtype=float, like=self.particles
            )
        self.log_weights = log_weights

    def copy(self) -> "ParticleState":
        """Return a copy of the state preserving the array backend."""
        xp = namespace(self.particles, self.log_weights)
        return ParticleState(
            particles=array_copy(self.particles, xp=xp),
            log_weights=array_copy(self.log_weights, xp=xp),
            log_evidence=float(self.log_evidence),
            evaluation=(
                None if self.evaluation is None else self.evaluation.copy()
            ),
            metadata=copy.deepcopy(self.metadata),
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
        if "particles" in kwargs and "evaluation" not in kwargs:
            updated.evaluation = None
        for key, value in kwargs.items():
            setattr(updated, key, value)
        return updated

    def replace_particles(
        self,
        particles: Array,
        *,
        evaluation: TargetEvaluation | None = None,
        **updates: Any,
    ) -> "ParticleState":
        """Replace particles and explicitly provide their evaluation."""
        return self.with_updates(
            particles=particles,
            evaluation=evaluation,
            **updates,
        )

    def resample(self, indices: Array) -> "ParticleState":
        """Return a resampled state using the provided ancestor indices."""
        xp = namespace(self.particles, indices)
        particles = asarray(self.particles, xp=xp)
        new_particles = particles[indices]
        n = indices.shape[0]
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
            evaluation=(
                None
                if self.evaluation is None
                else self.evaluation.take(indices)
            ),
            metadata=dict(self.metadata, ancestors=indices),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a detached dictionary representation of this state."""
        copied = self.copy()
        return {
            "particles": copied.particles,
            "log_weights": copied.log_weights,
            "log_evidence": copied.log_evidence,
            "evaluation": (
                None
                if copied.evaluation is None
                else copied.evaluation.to_dict()
            ),
            "metadata": copied.metadata,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ParticleState":
        """Reconstruct a state from :meth:`to_dict` output."""
        if not isinstance(payload, Mapping):
            raise TypeError("ParticleState payload must be a mapping.")
        missing = {
            key
            for key in ("particles", "log_weights", "log_evidence", "metadata")
            if key not in payload
        }
        if missing:
            raise ValueError(
                "ParticleState payload is missing required keys: "
                + ", ".join(sorted(missing))
            )
        evaluation_payload = payload.get("evaluation")
        if evaluation_payload is not None:
            from .target import TargetEvaluation

            evaluation = TargetEvaluation.from_dict(evaluation_payload)
        else:
            evaluation = None
        return cls(
            particles=payload["particles"],
            log_weights=payload["log_weights"],
            log_evidence=float(payload["log_evidence"]),
            evaluation=evaluation,
            metadata=copy.deepcopy(payload["metadata"]),
        )
