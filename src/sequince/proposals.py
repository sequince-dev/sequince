from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class ProposalKernel(Protocol):
    """Protocol defining the interface required by waste-free moves."""

    def sample(
        self,
        particles: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Draw proposal particles corresponding to the supplied parents."""

    def log_density(
        self,
        from_particles: np.ndarray,
        to_particles: np.ndarray,
    ) -> np.ndarray:
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
        particles: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        arr = np.asarray(particles, dtype=float)
        noise = rng.normal(
            loc=0.0,
            scale=self.scale,
            size=arr.shape,
        )
        return arr + noise

    def log_density(
        self,
        from_particles: np.ndarray,
        to_particles: np.ndarray,
    ) -> np.ndarray:
        start = np.asarray(from_particles, dtype=float)
        end = np.asarray(to_particles, dtype=float)
        if start.shape != end.shape:
            raise ValueError(
                "Particles passed to log_density must share shape."
            )
        diff = end - start
        if diff.ndim == 1:
            diff = diff[:, np.newaxis]
        dim = diff.shape[1]
        var = float(self.scale) ** 2
        norm_const = -0.5 * dim * np.log(2.0 * np.pi * var)
        exponent = -0.5 * np.sum((diff**2) / var, axis=1)
        return norm_const + exponent
