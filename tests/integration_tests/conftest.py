"""Shared fixtures for integration tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

PRIOR_MEAN = 0.0
PRIOR_STD = 2.0
OBSERVATION = 1.5
LIKELIHOOD_STD = 1.0


@pytest.fixture
def gaussian_log_prior() -> Callable[..., np.ndarray]:
    """Return the normalized Gaussian prior density from the example model."""

    def log_prior(
        particles: np.ndarray,
        context: Any | None = None,
    ) -> np.ndarray:
        del context
        x = particles[:, 0]
        residual = (x - PRIOR_MEAN) / PRIOR_STD
        return -0.5 * residual**2 - np.log(PRIOR_STD * np.sqrt(2.0 * np.pi))

    return log_prior


@pytest.fixture
def gaussian_log_likelihood() -> Callable[..., np.ndarray]:
    """Return the normalized likelihood from the Gaussian example model."""

    def log_likelihood(
        particles: np.ndarray,
        context: Any | None = None,
    ) -> np.ndarray:
        del context
        x = particles[:, 0]
        residual = (OBSERVATION - x) / LIKELIHOOD_STD
        return -0.5 * residual**2 - np.log(
            LIKELIHOOD_STD * np.sqrt(2.0 * np.pi)
        )

    return log_likelihood
