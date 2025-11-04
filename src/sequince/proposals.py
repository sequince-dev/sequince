from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from ._array_api import asarray, namespace
from ._typing import Array

if TYPE_CHECKING:
    from orng import ArrayRNG
else:  # pragma: no cover - runtime fallback
    from orng import ArrayRNG  # type: ignore


class ProposalKernel(Protocol):
    """Protocol defining the interface required by waste-free moves."""

    def sample(
        self,
        particles: Array,
        rng: ArrayRNG,
    ) -> Array:
        """Draw proposal particles corresponding to the supplied parents."""

    def log_density(
        self,
        from_particles: Array,
        to_particles: Array,
    ) -> Array:
        """Return log transition densities q(to | from) for each particle."""


@dataclass(frozen=True)
class GaussianRandomWalk:
    """Isotropic Gaussian random-walk proposal."""

    scale: float = 1.0

    def __post_init__(self) -> None:
        if self.scale <= 0.0:
            raise ValueError("GaussianRandomWalk.scale must be positive.")

    def sample(
        self,
        particles: Array,
        rng: ArrayRNG,
    ) -> Array:
        xp = namespace(particles)
        arr = asarray(particles, xp=xp)
        noise = asarray(
            rng.normal(
                loc=0.0,
                scale=self.scale,
                size=arr.shape,
            ),
            xp=xp,
        )
        return arr + noise

    def log_density(
        self,
        from_particles: Array,
        to_particles: Array,
    ) -> Array:
        xp = namespace(from_particles, to_particles)
        start = asarray(from_particles, xp=xp)
        end = asarray(to_particles, xp=xp)
        if start.shape != end.shape:
            raise ValueError(
                "Particles passed to log_density must share shape."
            )
        diff = end - start
        if diff.ndim == 1:
            diff = xp.reshape(diff, (diff.shape[0], 1))
        dim = diff.shape[1]
        var = float(self.scale) ** 2
        norm_const = -0.5 * dim * math.log(2.0 * math.pi * var)
        exponent = -0.5 * xp.sum((diff**2) / var, axis=1)
        return exponent + norm_const
