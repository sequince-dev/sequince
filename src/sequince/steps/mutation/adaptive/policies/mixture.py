"""Outer-iteration policies for top-level mixture probabilities."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
from typing import Protocol

import numpy as np

from .....array_api import to_numpy
from ....context import StepContext
from ...metrics import MutationMetricsOutput
from .de import _normalise_weights


class MixtureWeightPolicy(Protocol):
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
class ESJDMixtureWeightPolicy:
    """Favor components with larger squared displacement per attempted move.

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
        Require this many attempts in *every* mixture component before
        updating. Sparse blocks and blocks with zero total movement retain
        the current distribution.

    Notes
    -----
    Scores are squared accepted-jump sums divided by attempts (rejections
    contribute zero). Only top-level component probabilities are adapted;
    internal DE scale probabilities remain fixed. Scores use mutation
    coordinates, including any configured transform. They are not normalized
    for coordinate scale or likelihood cost. Equal-cost proposals make ESJD
    a movement-per-cost proxy, not a convergence diagnostic. Host conversion
    occurs only after the mutation block has completed.
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
        values = metrics.values.get("mixture_components")
        if not isinstance(values, Mapping):
            raise ValueError(
                "ESJDMixtureWeightPolicy requires mixture_components metrics."
            )
        arrays = {}
        for name in ("component", "n_attempted", "squared_jump_distance"):
            if name not in values:
                raise ValueError(
                    f"mixture_components metrics are missing {name}."
                )
            array = np.asarray(to_numpy(values[name]), dtype=float)
            if array.shape != (n_components,) or not np.all(
                np.isfinite(array)
            ):
                raise ValueError(
                    f"mixture_components {name} must contain one finite value "
                    "per mixture component."
                )
            arrays[name] = array
        if not np.array_equal(arrays["component"], list(range(n_components))):
            raise ValueError(
                "mixture_components identities do not match the mixture."
            )
        attempts = arrays["n_attempted"]
        jumps = arrays["squared_jump_distance"]
        if np.any(attempts < 0) or np.any(attempts != np.floor(attempts)):
            raise ValueError(
                "Per-component attempts must be non-negative integers."
            )
        if np.any(jumps < 0) or np.any((attempts == 0) & (jumps != 0)):
            raise ValueError(
                "Per-component jump sums must be non-negative "
                "and zero without attempts."
            )
        if np.any(attempts < self.min_attempts):
            return weights
        scores = jumps / attempts
        if scores.max() == 0:
            return weights
        # Scaling first keeps normalization finite even for large jump sums.
        scores = scores / scores.max()
        proposed = self._explore(scores / scores.sum())
        updated = (1 - self.learning_rate) * np.asarray(
            weights
        ) + self.learning_rate * proposed
        return _normalise_weights(updated)


__all__ = ["MixtureWeightPolicy", "ESJDMixtureWeightPolicy"]
