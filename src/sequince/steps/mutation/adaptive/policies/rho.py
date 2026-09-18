"""Policies selecting MiniPCN step sizes from completed mutation blocks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Protocol

import numpy as np

from .....array_api import to_numpy
from ....context import StepContext
from ...metrics import MutationMetricsOutput


def _validate_rho(value: Any) -> float:
    """Validate the kernel's step-size domain independently of policy bounds."""
    array = to_numpy(value)
    if array.size != 1:
        raise TypeError("rho must be scalar.")
    rho = float(array.reshape(()))
    if not np.isfinite(rho) or not 0 < rho < 1:
        raise ValueError(
            "rho must be finite and strictly between zero and one."
        )
    return rho


class MiniPCNRhoPolicy(Protocol):
    """Choose rho before a block and from its completed metrics.

    Decisions must be scalar, finite and strictly between zero and one.
    Stateful policies may supply ``state_dict`` and ``load_state_dict``;
    the wrapper forwards these when checkpointing. Policies never see
    intermediate rounds or mutate the sampler.
    """

    def initialise(self, *, initial_rho: float, context: StepContext) -> float:
        """Choose the first iteration's step size."""
        ...

    def update(
        self,
        *,
        current_rho: float,
        metrics: MutationMetricsOutput,
        context: StepContext,
    ) -> float:
        """Choose the next iteration's step size from one complete block."""
        ...


@dataclass(frozen=True)
class AcceptanceRateRhoPolicy:
    """Tune rho toward a target acceptance using a bounded multiplicative update.

    Parameters
    ----------
    target_acceptance : float, optional
        Desired acceptance fraction. Default 0.234.
    adaptation_rate : float, optional
        Rate in ``rho * exp(rate * (acceptance - target))``. Default 0.1.
        Zero retains rho exactly after initial clipping.
    min_rho, max_rho : float, optional
        Bounds strictly between zero and one. Defaults 0.001 and 0.99.
        Initial rho is clipped to these bounds.

    Notes
    -----
    Higher acceptance increases the next block's proposal step size. Updates
    use finalized accepted and attempted counts; fractional counts are valid.
    Empty blocks retain rho. This policy has no evolving state; checkpoint
    configuration is validated when restoring into a reconstructed policy.
    """

    target_acceptance: float = 0.234
    adaptation_rate: float = 0.1
    min_rho: float = 1e-3
    max_rho: float = 0.99

    def __post_init__(self) -> None:
        if not all(np.isfinite(value) for value in asdict(self).values()):
            raise ValueError("Rho policy parameters must be finite.")
        if not 0 < self.min_rho <= self.max_rho < 1:
            raise ValueError("Require 0 < min_rho <= max_rho < 1.")
        if not 0 <= self.target_acceptance <= 1 or self.adaptation_rate < 0:
            raise ValueError(
                "Require target_acceptance in [0, 1] and adaptation_rate >= 0."
            )

    def initialise(self, *, initial_rho: float, context: StepContext) -> float:
        return float(
            np.clip(_validate_rho(initial_rho), self.min_rho, self.max_rho)
        )

    def update(
        self,
        *,
        current_rho: float,
        metrics: MutationMetricsOutput,
        context: StepContext,
    ) -> float:
        rho = self.initialise(initial_rho=current_rho, context=context)
        counts = []
        for name in ("n_attempted", "n_accepted"):
            array = to_numpy(getattr(metrics, name))
            if array.size != 1:
                raise TypeError(f"{name} must be scalar.")
            value = float(array.reshape(()))
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
            counts.append(value)
        attempted, accepted = counts
        if accepted > attempted:
            raise ValueError("n_accepted must not exceed n_attempted.")
        if attempted == 0 or self.adaptation_rate == 0:
            return rho
        log_rho = np.log(rho) + self.adaptation_rate * (
            accepted / attempted - self.target_acceptance
        )
        # Clip before exponentiation to avoid overflow, and afterward to
        # keep rounding at a boundary inside the configured interval.
        return float(
            np.clip(
                np.exp(
                    np.clip(
                        log_rho, np.log(self.min_rho), np.log(self.max_rho)
                    )
                ),
                self.min_rho,
                self.max_rho,
            )
        )

    def state_dict(self) -> dict[str, Any]:
        return asdict(self)

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if dict(state) != self.state_dict():
            raise ValueError(
                "Rho policy checkpoint configuration does not match."
            )


__all__ = ["MiniPCNRhoPolicy", "AcceptanceRateRhoPolicy"]
