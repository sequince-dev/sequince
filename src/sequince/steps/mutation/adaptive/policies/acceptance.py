"""Choose mutation budgets from observed acceptance probabilities."""

from __future__ import annotations

import math

from .....array_api import asarray, namespace, to_numpy
from .....typing import Array, ArrayLike
from ....base import validate_n_steps
from ....context import StepContext
from ...metrics import MutationMetricsOutput


class AcceptanceRatePolicy:
    """Target expected accepted moves per particle using completed counts.

    Choose ``ceil(target_steps / max(acceptance, min_acceptance))``, clipped
    to configured bounds. With zero attempts, retain the current budget or
    recover to one, subject to bounds. Fractional accepted counts are supported.
    Both APIs divide in floating point before the ceiling and require
    ``max_steps <= 2**24`` for safe integer conversion."""

    def __init__(
        self,
        target_steps: float,
        *,
        min_steps: int,
        max_steps: int,
        min_acceptance: float = 1e-6,
    ) -> None:
        for name, value in (
            ("target_steps", target_steps),
            ("min_acceptance", min_acceptance),
        ):
            if isinstance(value, bool):
                raise TypeError(f"{name} must be a real number.")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if min_acceptance > 1:
            raise ValueError("min_acceptance must be <= 1.")
        self.min_steps = validate_n_steps(min_steps)
        self.max_steps = validate_n_steps(max_steps)
        if self.max_steps < self.min_steps:
            raise ValueError("max_steps must be >= min_steps.")
        self.validate_functional_configuration()
        self.target_steps = float(target_steps)
        self.min_acceptance = float(min_acceptance)
        self._functional_target = min(self.target_steps, self.max_steps)
        # Bound on the host before the ceiling, including ratios that
        # overflow to infinity. Positive targets always require at least one.
        self._zero_acceptance_steps = self._bound(
            max(
                1,
                math.ceil(
                    min(
                        self.target_steps / self.min_acceptance, self.max_steps
                    )
                ),
            )
        )

    def _bound(self, steps: int) -> int:
        return min(max(steps, self.min_steps), self.max_steps)

    def validate_functional_configuration(self) -> None:
        """Require bounds exactly representable in float32 and int32."""
        if self.max_steps > 2**24:
            raise ValueError(
                "AcceptanceRatePolicy max_steps must be <= 2**24."
            )

    def initialise_functional(
        self, *, initial_steps: Array, context: StepContext
    ) -> Array:
        xp = namespace(initial_steps)
        return xp.clip(initial_steps, self.min_steps, self.max_steps)

    def update_functional(
        self,
        *,
        current_steps: Array,
        metrics: MutationMetricsOutput,
        context: StepContext,
    ) -> Array:
        """Choose a bounded budget from scalar completed-block counts.

        Counts must be finite with ``0 <= accepted <= attempted``. Compiled code
        checks shapes only; callers must validate values. Eager calculations use
        host float64; other backend precisions may round differently."""
        return self._steps_from_counts(
            current_steps=current_steps,
            n_attempted=metrics.n_attempted,
            n_accepted=metrics.n_accepted,
        )

    def _steps_from_counts(
        self,
        *,
        current_steps: Array,
        n_attempted: ArrayLike,
        n_accepted: ArrayLike,
    ) -> Array:
        """Compute a bounded budget directly from scalar count inputs."""
        xp = namespace(current_steps)
        accepted = asarray(n_accepted, like=current_steps)
        attempted = asarray(n_attempted, like=current_steps)
        if accepted.shape != () or attempted.shape != ():
            raise TypeError("Mutation counts must be scalar arrays.")
        dtype = xp.result_type(
            accepted.dtype,
            attempted.dtype,
            asarray(0.0, like=current_steps).dtype,
        )
        accepted = xp.astype(accepted, dtype)
        attempted = xp.astype(attempted, dtype)
        one = xp.ones_like(attempted)
        acceptance = accepted / xp.where(attempted > 0, attempted, one)
        # Cap the target on the host before materializing it: even a finite
        # Python float may overflow the backend's floating dtype.
        target = asarray(
            self._functional_target,
            like=accepted,
            dtype=dtype,
        )
        # ceil(target / max(rate, floor)) is the smaller of the ordinary
        # decision and the zero-acceptance decision. Use the host-prepared
        # safeguard even when tiny configuration values underflow together.
        denominator = xp.maximum(acceptance, target / max(1, self.max_steps))
        ratio = target / xp.where(denominator > 0, denominator, one)
        safeguard = asarray(
            self._zero_acceptance_steps, like=accepted, dtype=dtype
        )
        proposed = xp.where(
            acceptance > 0,
            xp.minimum(xp.ceil(ratio), safeguard),
            safeguard,
        )
        proposed = xp.clip(
            xp.maximum(proposed, one), self.min_steps, self.max_steps
        )
        # Bound before conversion to prevent overflow in the integer cast.
        proposed = xp.astype(proposed, current_steps.dtype)
        recovery = xp.clip(
            xp.maximum(current_steps, xp.ones_like(current_steps)),
            self.min_steps,
            self.max_steps,
        )
        return xp.where(attempted > 0, proposed, recovery)

    def initialise(self, *, initial_steps: int, context: StepContext) -> int:
        return self._bound(validate_n_steps(initial_steps))

    def update(
        self,
        *,
        current_steps: int,
        metrics: MutationMetricsOutput,
        context: StepContext,
    ) -> int:
        current_steps = validate_n_steps(current_steps)
        if not isinstance(metrics, MutationMetricsOutput):
            raise TypeError(
                "AcceptanceRatePolicy requires MutationMetricsOutput."
            )
        counts = []
        for name in ("n_attempted", "n_accepted"):
            array = to_numpy(getattr(metrics, name))
            if array.size != 1:
                raise TypeError(f"{name} must be scalar.")
            value = float(array.reshape(()))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
            counts.append(value)
        attempted, accepted = counts
        if accepted > attempted:
            raise ValueError("n_accepted must not exceed n_attempted.")
        return int(
            self._steps_from_counts(
                current_steps=asarray(
                    self._bound(current_steps), dtype="int32"
                ),
                n_attempted=asarray(attempted),
                n_accepted=asarray(accepted),
            )
        )


__all__ = ["AcceptanceRatePolicy"]
