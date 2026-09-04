"""Minimal diagnostics for fixed-length mutation blocks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..array_api import asarray, logsumexp, namespace
from ..utils.pytrees import pytree_dataclass

MUTATION_DIAGNOSTICS_ACCUMULATOR_KEY = "_mutation_diagnostics_accumulator"
MUTATION_DIAGNOSTICS_RESET_REFERENCE_KEY = (
    "_mutation_diagnostics_reset_reference"
)


@pytree_dataclass(
    data_fields=("n_accepted", "normalised_esjd_per_step"),
    meta_fields=("n_steps", "n_attempted"),
)
@dataclass(frozen=True)
class MutationDiagnostics:
    """Diagnostics for one complete continuous-MCMC mutation block."""

    n_steps: int
    n_attempted: int
    n_accepted: Any
    normalised_esjd_per_step: Any

    @classmethod
    def empty(cls, particles: Any) -> MutationDiagnostics:
        """Return zero diagnostics in the particles' array namespace."""
        xp = namespace(particles)
        return cls(0, 0, xp.asarray(0), xp.asarray(0.0))

    @property
    def acceptance(self) -> Any:
        """Fraction of attempted proposals accepted, or zero without attempts."""
        if self.n_attempted == 0:
            return 0.0
        return self.n_accepted / self.n_attempted


class MutationDiagnosticsAccumulator:
    """Stream standardised squared jumps for one mutation block."""

    def __init__(self, particles: Any = None, log_weights: Any = None):
        self._initialized = False
        if particles is not None:
            self.initialize(particles, log_weights)

    @property
    def initialized(self) -> bool:
        """Whether the first mutation participant has fixed the scaling."""
        return self._initialized

    def initialize(self, particles: Any, log_weights: Any) -> None:
        """Fix scaling and reference coordinates at the first participant."""
        if self._initialized:
            return
        xp = namespace(particles)
        values = self._matrix(asarray(particles, xp=xp))
        weights = asarray(log_weights, xp=xp)
        weights = xp.exp(weights - logsumexp(weights, xp=xp))
        mean = xp.sum(values * weights[:, None], axis=0)
        variance = xp.sum(xp.square(values - mean) * weights[:, None], axis=0)
        safe_variance = xp.where(
            variance > xp.finfo(values.dtype).eps,
            variance,
            xp.ones_like(variance),
        )

        self.last_particles = values
        self.inverse_variance = 1.0 / safe_variance
        self.sum_squared_jump = xp.asarray(0.0, dtype=values.dtype)
        self.n_steps = 0
        self.n_accepted = xp.asarray(0)
        self.n_particles = values.shape[0]
        self.n_dimensions = values.shape[1]
        self._initialized = True

    @staticmethod
    def _matrix(values: Any) -> Any:
        xp = namespace(values)
        return xp.reshape(values, (values.shape[0], -1))

    def reset_reference(self, particles: Any) -> None:
        """Exclude an intervening non-MCMC operation from the next jump."""
        if not self._initialized:  # pragma: no cover - lifecycle guard
            raise RuntimeError(
                "Mutation diagnostics have not been initialized."
            )
        xp = namespace(particles)
        self.last_particles = self._matrix(asarray(particles, xp=xp))

    def add_transition(self, particles: Any, accepted: Any) -> None:
        """Record one continuous-MCMC population transition."""
        if not self._initialized:  # pragma: no cover - lifecycle guard
            raise RuntimeError(
                "Mutation diagnostics have not been initialized."
            )
        xp = namespace(particles)
        current = self._matrix(asarray(particles, xp=xp))
        self.sum_squared_jump = self.sum_squared_jump + xp.sum(
            xp.square(current - self.last_particles) * self.inverse_variance
        )
        self.n_steps += 1
        self.n_accepted = self.n_accepted + xp.sum(asarray(accepted, xp=xp))
        self.last_particles = current

    def finish(self) -> MutationDiagnostics:
        """Return the completed fixed-block diagnostics."""
        if not self._initialized:  # pragma: no cover - lifecycle guard
            raise RuntimeError(
                "Mutation diagnostics have not been initialized."
            )
        n_attempted = self.n_particles * self.n_steps
        normalizer = n_attempted * self.n_dimensions
        esjd = (
            self.sum_squared_jump / normalizer
            if normalizer
            else self.sum_squared_jump
        )
        return MutationDiagnostics(
            n_steps=self.n_steps,
            n_attempted=n_attempted,
            n_accepted=self.n_accepted,
            normalised_esjd_per_step=esjd,
        )


def prepare_mutation_diagnostics(
    particles: Any, log_weights: Any, context: Any
) -> tuple[MutationDiagnosticsAccumulator, Any, bool]:
    """Prepare diagnostics and report whether this mutation owns them."""
    owns_accumulator = (
        MUTATION_DIAGNOSTICS_ACCUMULATOR_KEY not in context.metadata
    )
    accumulator = context.metadata.get(MUTATION_DIAGNOSTICS_ACCUMULATOR_KEY)
    if accumulator is None:
        accumulator = MutationDiagnosticsAccumulator(particles, log_weights)
        context = context.with_metadata(
            **{MUTATION_DIAGNOSTICS_ACCUMULATOR_KEY: accumulator}
        )
    elif not accumulator.initialized:
        accumulator.initialize(particles, log_weights)
    elif context.metadata.get(MUTATION_DIAGNOSTICS_RESET_REFERENCE_KEY, False):
        accumulator.reset_reference(particles)
    if MUTATION_DIAGNOSTICS_RESET_REFERENCE_KEY in context.metadata:
        context = context.without_metadata(
            MUTATION_DIAGNOSTICS_RESET_REFERENCE_KEY
        )
    return accumulator, context, owns_accumulator


def begin_mutation_diagnostics(context: Any) -> tuple[Any, bool]:
    """Begin an outer block without choosing its mutation coordinate space."""
    owns_accumulator = (
        MUTATION_DIAGNOSTICS_ACCUMULATOR_KEY not in context.metadata
    )
    if owns_accumulator:
        context = context.with_metadata(
            **{
                MUTATION_DIAGNOSTICS_ACCUMULATOR_KEY: (
                    MutationDiagnosticsAccumulator()
                )
            }
        )
    return context, owns_accumulator


def empty_mutation_diagnostics(
    particles: Any, log_weights: Any, context: Any
) -> tuple[MutationDiagnostics | None, Any]:
    """Return standalone empty diagnostics or join an enclosing block."""
    if MUTATION_DIAGNOSTICS_ACCUMULATOR_KEY not in context.metadata:
        return MutationDiagnostics.empty(particles), context
    accumulator, context, owns_accumulator = prepare_mutation_diagnostics(
        particles, log_weights, context
    )
    return finish_mutation_diagnostics(
        accumulator,
        context,
        owns_accumulator=owns_accumulator,
    )


def finish_mutation_diagnostics(
    accumulator: MutationDiagnosticsAccumulator,
    context: Any,
    *,
    owns_accumulator: bool,
) -> tuple[MutationDiagnostics | None, Any]:
    """Finalise and remove diagnostics only for the owning mutation block."""
    if not owns_accumulator:
        return None, context
    diagnostics = accumulator.finish()
    context = context.without_metadata(
        MUTATION_DIAGNOSTICS_ACCUMULATOR_KEY,
        MUTATION_DIAGNOSTICS_RESET_REFERENCE_KEY,
    )
    return diagnostics, context


def request_mutation_diagnostics_reference_reset(context: Any) -> Any:
    """Exclude an intervening non-MCMC operation from the next transition."""
    if MUTATION_DIAGNOSTICS_ACCUMULATOR_KEY not in context.metadata:
        return context
    return context.with_metadata(
        **{MUTATION_DIAGNOSTICS_RESET_REFERENCE_KEY: True}
    )


__all__ = [
    "MUTATION_DIAGNOSTICS_ACCUMULATOR_KEY",
    "MUTATION_DIAGNOSTICS_RESET_REFERENCE_KEY",
    "MutationDiagnostics",
    "MutationDiagnosticsAccumulator",
    "begin_mutation_diagnostics",
    "empty_mutation_diagnostics",
    "finish_mutation_diagnostics",
    "prepare_mutation_diagnostics",
    "request_mutation_diagnostics_reference_reset",
]
