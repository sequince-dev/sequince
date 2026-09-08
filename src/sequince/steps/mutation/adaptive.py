"""Adaptation after completed mutation blocks."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from ...array_api import to_numpy
from ..base import FixedLengthMutation, SMCStep, validate_n_steps
from ..context import StepContext
from ..mutation_diagnostics import MutationDiagnostics

if TYPE_CHECKING:
    from orng import RandomGenerator

    from ...state import ParticleState


class MutationStepPolicy(Protocol):
    """Choose mutation budgets using completed, iteration-level diagnostics.

    Policies are called only at mutation-block boundaries. In particular, they
    never receive intermediate mutation states or diagnostics.
    """

    def initialise(self, *, initial_steps: int, context: StepContext) -> int:
        """Choose the budget for the first adaptive iteration."""
        ...

    def update(
        self,
        *,
        current_steps: int,
        diagnostics: MutationDiagnostics,
        context: StepContext,
    ) -> int:
        """Choose the proposed budget for the next outer iteration."""
        ...


class FixedMutationStepPolicy:
    """Keep the mutation budget fixed at its current value."""

    def initialise(self, *, initial_steps: int, context: StepContext) -> int:
        del context
        return initial_steps

    def update(
        self,
        *,
        current_steps: int,
        diagnostics: MutationDiagnostics,
        context: StepContext,
    ) -> int:
        del diagnostics, context
        return current_steps


class ESJDPolicy:
    """Adapt mutation length from the previous block's normalised ESJD.

    The proposed multiplicative change is

    ``(target_esjd / esjd) ** damping``.

    The ratio is first clipped to
    ``[1 / max_change_factor, max_change_factor]``, then multiplied by the
    current length and rounded upward. Absolute ``min_steps`` and ``max_steps``
    bounds remain the responsibility of :class:`AdaptMutationSteps`.

    Parameters
    ----------
    target_esjd:
        Positive target normalised ESJD per transition.
    damping:
        Power-law damping in ``(0, 1]``. The default ``0.5`` uses a square-root
        response to reduce sensitivity to noisy block diagnostics.
    max_change_factor:
        Finite factor of at least one limiting the increase or decrease in one
        update. The default prevents more than a doubling or halving.
    """

    def __init__(
        self,
        target_esjd: float,
        *,
        damping: float = 0.5,
        max_change_factor: float = 2.0,
    ) -> None:
        self.target_esjd = self._positive_finite(target_esjd, "target_esjd")
        self.damping = self._positive_finite(damping, "damping")
        if self.damping > 1.0:
            raise ValueError("damping must be <= 1.")
        self.max_change_factor = self._positive_finite(
            max_change_factor, "max_change_factor"
        )
        if self.max_change_factor < 1.0:
            raise ValueError("max_change_factor must be >= 1.")

    @staticmethod
    def _positive_finite(value: float, name: str) -> float:
        if isinstance(value, bool):
            raise TypeError(f"{name} must be a real number.")
        try:
            scalar = float(value)
        except (TypeError, ValueError) as error:
            raise TypeError(f"{name} must be a real number.") from error
        if not math.isfinite(scalar) or scalar <= 0.0:
            raise ValueError(f"{name} must be finite and > 0.")
        return scalar

    def initialise(self, *, initial_steps: int, context: StepContext) -> int:
        del context
        return initial_steps

    @staticmethod
    def _esjd(diagnostics: MutationDiagnostics) -> float:
        if not isinstance(diagnostics, MutationDiagnostics):
            raise TypeError(
                "ESJDPolicy requires completed MutationDiagnostics."
            )
        value = diagnostics.metrics.get("esjd")
        if value is None:
            raise ValueError("MutationDiagnostics is missing esjd.")
        array = to_numpy(value)
        if array.size != 1:
            raise TypeError("esjd must be a scalar.")
        try:
            esjd = float(array.reshape(()))
        except (TypeError, ValueError) as error:
            raise TypeError("esjd must be a real scalar.") from error
        if not math.isfinite(esjd):
            raise ValueError("esjd must be finite.")
        if esjd < 0.0:
            raise ValueError("esjd must be >= 0.")
        return esjd

    def update(
        self,
        *,
        current_steps: int,
        diagnostics: MutationDiagnostics,
        context: StepContext,
    ) -> int:
        del context
        current_steps = validate_n_steps(current_steps)
        esjd = self._esjd(diagnostics)
        if esjd == 0.0:
            # Multiplication cannot escape a zero budget, so propose one step in
            # that degenerate case. The wrapper still applies its absolute bounds.
            return max(1, math.ceil(current_steps * self.max_change_factor))

        log_limit = math.log(self.max_change_factor)
        log_ratio = self.damping * (
            math.log(self.target_esjd) - math.log(esjd)
        )
        bounded_ratio = math.exp(min(max(log_ratio, -log_limit), log_limit))
        return math.ceil(current_steps * bounded_ratio)


class AdaptiveMutationStep(SMCStep, ABC):
    """Base for eager adaptation after a complete mutation block.

    Subclasses prepare context and optionally choose a fixed budget in
    ``_prepare_mutation``, then consume finalized diagnostics in ``_adapt``.
    The base executes the child once, preserves its output, and forwards child
    checkpoint state. Adaptation cannot interrupt the current mutation block.
    """

    mutation: SMCStep | FixedLengthMutation

    def __init__(self, mutation: SMCStep | FixedLengthMutation) -> None:
        self.mutation = mutation

    def _prepare_mutation(
        self,
        context: StepContext,
        n_steps: int | None,
    ) -> tuple[StepContext, int | None]:
        """Prepare inputs and freeze an optional budget before execution."""
        return context, n_steps

    @abstractmethod
    def _adapt(
        self,
        diagnostics: MutationDiagnostics,
        context: StepContext,
        *,
        initial_context: StepContext,
        n_steps: int | None,
    ) -> StepContext:
        """Update adaptation from completed diagnostics, returning context."""
        raise NotImplementedError

    @staticmethod
    def _completed_diagnostics(output: Any) -> MutationDiagnostics:
        diagnostics = getattr(output, "diagnostics", None)
        if diagnostics is None and isinstance(output, Mapping):
            diagnostics = output.get("diagnostics")
        if not isinstance(diagnostics, MutationDiagnostics):
            raise TypeError(
                "The wrapped mutation must return one complete "
                "MutationDiagnostics result."
            )
        return diagnostics

    def _apply(
        self,
        state: ParticleState,
        context: StepContext,
        rng: RandomGenerator,
        *,
        n_steps: int | None,
    ) -> tuple[ParticleState, StepContext, Any]:
        context, n_steps = self._prepare_mutation(context, n_steps)
        initial_context = context
        if n_steps is None:
            if not isinstance(self.mutation, SMCStep):
                raise TypeError(
                    "mutation must be an SMCStep for default execution."
                )
            state, context, output = self.mutation.apply(state, context, rng)
        else:
            if not isinstance(self.mutation, FixedLengthMutation):
                raise TypeError(
                    "mutation must implement apply_with_n_steps()."
                )
            state, context, output = self.mutation.apply_with_n_steps(
                state, context, rng, n_steps=validate_n_steps(n_steps)
            )
        diagnostics = self._completed_diagnostics(output)
        context = self._adapt(
            diagnostics,
            context,
            initial_context=initial_context,
            n_steps=n_steps,
        )
        return state, context, output

    def apply(
        self,
        state: ParticleState,
        context: StepContext,
        rng: RandomGenerator,
    ) -> tuple[ParticleState, StepContext, Any]:
        return self._apply(state, context, rng, n_steps=None)

    def apply_functional(self, state, context, random_state):
        raise NotImplementedError(
            f"{type(self).__name__} supports eager execution only."
        )

    @staticmethod
    def _nested_state(component: Any, name: str) -> dict[str, Any]:
        state_fn = getattr(component, "state_dict", None)
        if not callable(state_fn):
            return {}
        state = state_fn()
        if not isinstance(state, Mapping):
            raise TypeError(f"{name} state_dict() must return a mapping.")
        return dict(state)

    @staticmethod
    def _load_nested_state(component: Any, state: Any, name: str) -> None:
        if not isinstance(state, Mapping):
            raise TypeError(
                f"AdaptiveMutationStep state '{name}' must be a mapping."
            )
        load_fn = getattr(component, "load_state_dict", None)
        if not callable(load_fn):
            if state:
                raise ValueError(f"{name} does not support checkpoint state.")
            return
        load_fn(state)

    def state_dict(self) -> dict[str, Any]:
        """Forward the wrapped mutation's checkpoint state."""
        return {"mutation": self._nested_state(self.mutation, "Mutation")}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore the wrapped mutation's checkpoint state."""
        if "mutation" in state:
            self._load_nested_state(
                self.mutation, state["mutation"], "mutation"
            )


class AdaptMutationSteps(AdaptiveMutationStep):
    """Adapt a fixed-length mutation budget between SMC iterations.

    Conditional on the completed SMC history before iteration ``t``, the
    mutation budget is chosen and frozen before mutation begins. The wrapped
    mutation then executes that complete fixed-length block. Its final
    :class:`MutationDiagnostics` may determine the budget for iteration
    ``t + 1``, but cannot change the current block. Repeated calls with the same
    :attr:`StepContext.iteration` reuse the frozen budget and update the policy
    at most once.

    This wrapper deliberately does not implement trajectory-dependent stopping.
    Budget splitting across fixed rounds (for example, in
    ``RepeatedMutationStep``) remains the responsibility of the wrapped mutation.
    """

    def __init__(
        self,
        mutation: FixedLengthMutation,
        policy: MutationStepPolicy | None = None,
        *,
        initial_steps: int,
        min_steps: int,
        max_steps: int,
    ) -> None:
        if not isinstance(mutation, FixedLengthMutation):
            raise TypeError(
                "mutation must support fixed-length execution via "
                "apply_with_n_steps()."
            )
        self._min_steps = validate_n_steps(min_steps)
        self._max_steps = validate_n_steps(max_steps)
        if self._max_steps < self._min_steps:
            raise ValueError("max_steps must be >= min_steps.")
        self._initial_steps = validate_n_steps(initial_steps)
        if not self._min_steps <= self._initial_steps <= self._max_steps:
            raise ValueError(
                "initial_steps must be within [min_steps, max_steps]."
            )

        resolved_policy = (
            policy if policy is not None else FixedMutationStepPolicy()
        )
        if not callable(
            getattr(resolved_policy, "initialise", None)
        ) or not callable(getattr(resolved_policy, "update", None)):
            raise TypeError("policy must implement initialise() and update().")

        super().__init__(mutation)
        self.policy = resolved_policy
        self._policy_initialised = False
        self._current_iteration: int | None = None
        self._current_n_steps: int | None = None
        self._next_n_steps: int | None = None
        self._last_updated_iteration: int | None = None

    @property
    def current_n_steps(self) -> int | None:
        """The budget frozen for the current or most recent iteration."""
        return self._current_n_steps

    @property
    def next_n_steps(self) -> int | None:
        """The budget proposed for the next outer iteration, if known."""
        return self._next_n_steps

    def _bound_policy_decision(self, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError("Mutation policy decisions must be integers.")
        return min(max(int(value), self._min_steps), self._max_steps)

    def _freeze_budget(self, context: StepContext) -> int:
        if self._current_iteration == context.iteration:
            if self._current_n_steps is None:  # pragma: no cover - state guard
                raise RuntimeError("The current mutation budget is missing.")
            return self._current_n_steps

        if not self._policy_initialised:
            proposed = self.policy.initialise(
                initial_steps=self._initial_steps,
                context=context,
            )
        else:
            if self._next_n_steps is None:  # pragma: no cover - state guard
                raise RuntimeError("The next mutation budget is missing.")
            proposed = self._next_n_steps

        frozen = self._bound_policy_decision(proposed)
        self._policy_initialised = True
        self._current_iteration = context.iteration
        self._current_n_steps = frozen
        return frozen

    def _prepare_mutation(
        self,
        context: StepContext,
        n_steps: int | None,
    ) -> tuple[StepContext, int | None]:
        return context, self._freeze_budget(context)

    def _adapt(
        self,
        diagnostics: MutationDiagnostics,
        context: StepContext,
        *,
        initial_context: StepContext,
        n_steps: int | None,
    ) -> StepContext:
        iteration = initial_context.iteration
        if self._last_updated_iteration != iteration:
            assert n_steps is not None
            proposed = self.policy.update(
                current_steps=n_steps,
                diagnostics=diagnostics,
                context=context,
            )
            self._next_n_steps = self._bound_policy_decision(proposed)
            self._last_updated_iteration = iteration
        return context

    def state_dict(self) -> dict[str, Any]:
        """Return the frozen/current decision and nested mutable state."""
        return {
            "policy_initialised": self._policy_initialised,
            "current_iteration": self._current_iteration,
            "current_n_steps": self._current_n_steps,
            "next_n_steps": self._next_n_steps,
            "last_updated_iteration": self._last_updated_iteration,
            **super().state_dict(),
            "policy": self._nested_state(self.policy, "Policy"),
        }

    @staticmethod
    def _optional_integer(value: Any, name: str) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(
                f"AdaptMutationSteps state '{name}' must be an integer."
            )
        return int(value)

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore an adaptive decision without consulting the policy again."""
        required = {
            "policy_initialised",
            "current_iteration",
            "current_n_steps",
            "next_n_steps",
            "last_updated_iteration",
            "mutation",
            "policy",
        }
        missing = required.difference(state)
        if missing:
            raise ValueError(
                "AdaptMutationSteps state is missing required keys: "
                + ", ".join(sorted(missing))
            )
        initialised = state["policy_initialised"]
        if not isinstance(initialised, bool):
            raise TypeError(
                "AdaptMutationSteps state 'policy_initialised' must be a boolean."
            )
        current_iteration = self._optional_integer(
            state["current_iteration"], "current_iteration"
        )
        current_n_steps = self._optional_integer(
            state["current_n_steps"], "current_n_steps"
        )
        next_n_steps = self._optional_integer(
            state["next_n_steps"], "next_n_steps"
        )
        last_updated_iteration = self._optional_integer(
            state["last_updated_iteration"], "last_updated_iteration"
        )
        for name, value in (
            ("current_n_steps", current_n_steps),
            ("next_n_steps", next_n_steps),
        ):
            if (
                value is not None
                and not self._min_steps <= value <= self._max_steps
            ):
                raise ValueError(
                    f"AdaptMutationSteps state '{name}' is outside configured bounds."
                )

        super().load_state_dict(state)
        self._load_nested_state(self.policy, state["policy"], "policy")
        self._policy_initialised = initialised
        self._current_iteration = current_iteration
        self._current_n_steps = current_n_steps
        self._next_n_steps = next_n_steps
        self._last_updated_iteration = last_updated_iteration


@dataclass(frozen=True)
class AdaptGaussianProposal(AdaptiveMutationStep):
    """Adapt proposal scale after a complete mutation block (eager execution).

    The wrapped step must return finalized ``MutationDiagnostics`` in its
    output. Its output is preserved. The scale in ``context.metadata`` is
    initialized before mutation and updated afterward for the next call using
    ``scale * exp(adaptation_rate * (acceptance - target_acceptance))``.
    Proposal kernels must read ``scale_key`` from the context; this wrapper
    does not modify proposal objects. Wrap an entire ``RepeatedMutationStep``
    to keep the scale fixed across all its rounds. Place conditional execution
    outside this wrapper.

    Runtime scale lives in the context; checkpoint state forwards to the child.
    """

    mutation: SMCStep
    scale_key: str = "proposal_scale"
    initial_scale: float = 1.0
    target_acceptance: float | None = 0.3
    adaptation_rate: float = 0.1
    min_scale: float = 1e-3
    max_scale: float = 10.0

    def __post_init__(self) -> None:
        if not isinstance(self.mutation, SMCStep):
            raise TypeError("mutation must be an SMCStep.")
        if not self.scale_key:
            raise ValueError("scale_key must be nonempty.")
        values = (
            self.initial_scale,
            self.min_scale,
            self.max_scale,
            self.adaptation_rate,
        )
        if not all(np.isfinite(value) for value in values):
            raise ValueError("Scale parameters must be finite.")
        if not 0 < self.min_scale <= self.initial_scale <= self.max_scale:
            raise ValueError(
                "Require 0 < min_scale <= initial_scale <= max_scale."
            )
        if self.adaptation_rate < 0:
            raise ValueError("adaptation_rate must be nonnegative.")
        if (
            self.target_acceptance is not None
            and not 0 <= self.target_acceptance <= 1
        ):
            raise ValueError("target_acceptance must lie in [0, 1].")

    def _prepare_mutation(
        self,
        context: StepContext,
        n_steps: int | None,
    ) -> tuple[StepContext, int | None]:
        scale = float(context.metadata.get(self.scale_key, self.initial_scale))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Proposal scale must be finite and positive.")
        scale = float(np.clip(scale, self.min_scale, self.max_scale))
        context = context.with_metadata(**{self.scale_key: scale})
        return context, n_steps

    def _adapt(
        self,
        diagnostics: MutationDiagnostics,
        context: StepContext,
        *,
        initial_context: StepContext,
        n_steps: int | None,
    ) -> StepContext:
        scale = float(initial_context.metadata[self.scale_key])
        if self.target_acceptance is not None and diagnostics.n_attempted > 0:
            acceptance = float(diagnostics.acceptance)
            if np.isfinite(acceptance):
                # Bound in log space to avoid overflow for large learning rates.
                log_scale = np.log(scale) + self.adaptation_rate * (
                    acceptance - self.target_acceptance
                )
                scale = float(
                    np.exp(
                        np.clip(
                            log_scale,
                            np.log(self.min_scale),
                            np.log(self.max_scale),
                        )
                    )
                )
        return context.with_metadata(**{self.scale_key: scale})

    def apply_with_n_steps(
        self,
        state: ParticleState,
        context: StepContext,
        rng: RandomGenerator,
        *,
        n_steps: int,
    ) -> tuple[ParticleState, StepContext, Any]:
        return self._apply(
            state, context, rng, n_steps=validate_n_steps(n_steps)
        )


__all__ = [
    "AdaptGaussianProposal",
    "AdaptiveMutationStep",
    "AdaptMutationSteps",
    "ESJDPolicy",
    "FixedMutationStepPolicy",
    "MutationStepPolicy",
]
