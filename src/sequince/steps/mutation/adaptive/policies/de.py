"""Stage-boundary policies for a fixed bank of DE scale multipliers."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Protocol

import numpy as np

from .....array_api import to_numpy
from ....context import StepContext
from ...metrics import MutationMetricsOutput


def _normalise_weights(weights: Sequence[float]) -> tuple[float, ...]:
    """Validate and normalize configuration on the host at a stage boundary."""
    values = np.asarray(weights, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(
            "Scale weights must be a nonempty one-dimensional sequence."
        )
    if (
        not np.all(np.isfinite(values))
        or np.any(values < 0)
        or not np.any(values > 0)
    ):
        raise ValueError(
            "Scale weights must be finite, non-negative "
            "and have positive total."
        )
    values = values / values.max()
    return tuple(float(value) for value in values / values.sum())


class DEScaleWeightPolicy(Protocol):
    """Choose categorical weights only from completed mutation blocks."""

    def initialise(
        self,
        *,
        initial_weights: tuple[float, ...],
        context: StepContext,
    ) -> tuple[float, ...]: ...

    def update(
        self,
        *,
        current_weights: tuple[float, ...],
        metrics: MutationMetricsOutput,
        context: StepContext,
    ) -> tuple[float, ...]: ...


@dataclass(frozen=True)
class ESJDScaleWeightPolicy:
    """Favor DE scales with larger squared displacement per attempted move.

    Parameters
    ----------
    learning_rate : float
        Fraction of the proposed distribution used in each update, in (0, 1].
        The remainder retains the current weights.
    exploration_probability : float
        Fraction of each proposed distribution assigned uniformly, in (0, 1].
        Initialization mixes the supplied weights with this uniform fraction
        too, ensuring every component can be measured. Every weight is then at
        least ``exploration_probability / n_components``.
    min_attempts : int
        Require this many attempts in *every* ordinary component before
        updating. Sparse blocks and blocks with zero total movement retain
        the current distribution.

    Notes
    -----
    Scores are squared accepted-jump sums divided by attempts (rejections
    contribute zero). Explicit unit moves, component -1, are excluded; their
    probability is not adapted. Scores use mutation coordinates, including
    any configured transform. They are not normalized for coordinate scale
    or likelihood cost. Equal-cost proposals make ESJD a movement-per-cost
    proxy, not a convergence diagnostic. Host conversion occurs only after
    the mutation block has completed.
    """

    learning_rate: float = 0.5
    exploration_probability: float = 0.1
    min_attempts: int = 20

    def __post_init__(self) -> None:
        for name in ("learning_rate", "exploration_probability"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not math.isfinite(value)
                or not 0 < value <= 1
            ):
                raise ValueError(f"{name} must be finite and in (0, 1].")
        if isinstance(self.min_attempts, bool) or not isinstance(
            self.min_attempts, Integral
        ):
            raise TypeError("min_attempts must be a positive integer.")
        if self.min_attempts < 1:
            raise ValueError("min_attempts must be a positive integer.")

    def _explore(self, weights: np.ndarray) -> np.ndarray:
        return (
            1 - self.exploration_probability
        ) * weights + self.exploration_probability / len(weights)

    def initialise(
        self,
        *,
        initial_weights: tuple[float, ...],
        context: StepContext,
    ) -> tuple[float, ...]:
        weights = np.asarray(_normalise_weights(initial_weights))
        return tuple(float(w) for w in self._explore(weights))

    def update(
        self,
        *,
        current_weights: tuple[float, ...],
        metrics: MutationMetricsOutput,
        context: StepContext,
    ) -> tuple[float, ...]:
        weights = _normalise_weights(current_weights)
        n_components = len(weights)
        values = metrics.values.get("de_scales")
        if not isinstance(values, Mapping):
            raise ValueError(
                "ESJDScaleWeightPolicy requires de_scales metrics."
            )
        arrays = {}
        for name in ("component", "n_attempted", "squared_jump_distance"):
            if name not in values:
                raise ValueError(f"de_scales metrics are missing {name}.")
            array = np.asarray(to_numpy(values[name]), dtype=float)
            if array.shape != (n_components + 1,) or not np.all(
                np.isfinite(array)
            ):
                raise ValueError(
                    f"de_scales {name} must contain one finite value "
                    "per component, including unit moves."
                )
            arrays[name] = array
        if not np.array_equal(arrays["component"], [*range(n_components), -1]):
            raise ValueError(
                "de_scales component identities do not match the scale bank."
            )
        attempts = arrays["n_attempted"]
        jumps = arrays["squared_jump_distance"]
        if np.any(attempts < 0) or np.any(attempts != np.floor(attempts)):
            raise ValueError(
                "Per-scale attempts must be non-negative integers."
            )
        if np.any(jumps < 0) or np.any((attempts == 0) & (jumps != 0)):
            raise ValueError(
                "Per-scale jump sums must be non-negative "
                "and zero without attempts."
            )
        if np.any(attempts[:-1] < self.min_attempts):
            return weights
        scores = jumps[:-1] / attempts[:-1]
        if scores.max() == 0:
            return weights
        # Scaling first keeps normalization finite even for large jump sums.
        scores = scores / scores.max()
        proposed = self._explore(scores / scores.sum())
        updated = (1 - self.learning_rate) * np.asarray(
            weights
        ) + self.learning_rate * proposed
        return _normalise_weights(updated)


__all__ = ["DEScaleWeightPolicy", "ESJDScaleWeightPolicy"]
