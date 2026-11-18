"""Adapt the number of mutation steps between SMC iterations."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, fields, replace
from numbers import Integral
from typing import TYPE_CHECKING, Any, TypeVar

from acai import cond

from ....array_api import asarray, namespace
from ....typing import Array
from ...base import (
    DynamicBudgetMutation,
    FixedLengthMutation,
    validate_n_steps,
)
from ...context import StepContext
from ..metrics import MutationMetricsOutput
from ..metrics.base import with_metrics
from .base import AdaptiveMutationStep
from .policies import MutationStepPolicy
from .policies.base import FunctionalMutationStepPolicy
from .policies.fixed import FixedMutationStepPolicy
from .state import MutationBudgetState

if TYPE_CHECKING:
    from ....smc import SMCConfig
    from ....state import ParticleState


_Scalar = TypeVar("_Scalar", int, Array)


class AdaptMutationSteps(AdaptiveMutationStep):
    """Adapt fixed mutation budgets using completed SMC history.

    Freeze each iteration's budget before mutation. Completed-block metrics
    propose the next budget; repeated calls reuse the frozen budget and
    update the policy at most once per iteration."""

    def __init__(
        self,
        mutation: FixedLengthMutation,
        policy: MutationStepPolicy | None = None,
        *,
        initial_steps: int,
        min_steps: int,
        max_steps: int,
        metadata_key: str = "mutation_budget",
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
        if not isinstance(metadata_key, str) or not metadata_key:
            raise ValueError("metadata_key must be a nonempty string.")
        self._metadata_key = metadata_key
        self.policy = resolved_policy
        self._budget = MutationBudgetState[int].initial(
            self._initial_steps, -1
        )

    @property
    def metadata_key(self) -> str:
        """Stable carry identity; use distinct keys for multiple adapters."""
        return self._metadata_key

    def _functional_policy(self) -> FunctionalMutationStepPolicy:
        if not isinstance(self.policy, FunctionalMutationStepPolicy):
            raise NotImplementedError(
                "Policy must implement FunctionalMutationStepPolicy for "
                "functional budget operations. Eager policies remain supported."
            )
        return self.policy

    def initialize_functional_metadata(
        self, state: ParticleState, metadata: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        """Allocate a scalar int32 carry before compilation, preserving resumes.

        Requires ``max_steps <= 2**24``. Resume with the same metadata key and
        static configuration; eager state is independent."""
        if isinstance(self.mutation, DynamicBudgetMutation):
            metadata = self.mutation.initialize_functional_metadata(
                state, metadata
            )
        policy = self._functional_policy()
        policy.validate_functional_configuration()
        if self._max_steps > 2**24:
            raise ValueError("Functional budget max_steps must be <= 2**24.")
        if self.metadata_key in metadata:
            carry = metadata[self.metadata_key]
            if not isinstance(carry, MutationBudgetState):
                raise TypeError("Budget metadata must be MutationBudgetState.")
            for field in fields(carry):
                value = getattr(carry, field.name)
                if (
                    getattr(value, "shape", None) != ()
                    or value.dtype != namespace(value).int32
                ):
                    raise TypeError(
                        "Budget carry leaves must be scalar int32."
                    )
            return metadata
        like = state.log_weights

        def scalar(value: int) -> Array:
            return asarray(value, like=like, dtype="int32")

        carry = MutationBudgetState[Array].initial(
            scalar(self._initial_steps), scalar(-1)
        )
        return {**metadata, self.metadata_key: carry}

    def _budget_state(
        self, context: StepContext
    ) -> MutationBudgetState[Array]:
        carry = context.metadata.get(self.metadata_key)
        if not isinstance(carry, MutationBudgetState):
            raise TypeError(
                "Initialize functional budget metadata before compilation."
            )
        return carry

    def prepare_budget_functional(
        self, context: StepContext
    ) -> tuple[StepContext, Array]:
        """Freeze the budget before a block, promoting a pending decision.

        Parameters
        ----------
        context : StepContext
            Initialized context with a nonnegative scalar int32 iteration.

        Returns
        -------
        context : StepContext
            New context containing the prepared carry.
        n_steps : array, shape ()
            Frozen int32 budget, reused for every call in this iteration.
        """
        policy = self._functional_policy()
        carry = self._budget_state(context)
        xp = namespace(carry.current_n_steps)
        iteration = asarray(
            context.iteration, like=carry.current_iteration, dtype="int32"
        )

        prepared = self._prepare_budget(
            carry,
            iteration,
            initialise=lambda: policy.initialise_functional(
                initial_steps=carry.current_n_steps, context=context
            ),
            xp=xp,
        )
        return (
            context.with_metadata(**{self.metadata_key: prepared}),
            prepared.current_n_steps,
        )

    def update_budget_functional(
        self, metrics: MutationMetricsOutput, context: StepContext
    ) -> StepContext:
        """Store a pending budget from completed metrics, at most once.

        Parameters
        ----------
        metrics : MutationMetricsOutput
            Final metrics for the complete mutation block.
        context : StepContext
            Prepared context identifying that block's iteration.

        Returns
        -------
        StepContext
            New context with a pending budget; the current budget is unchanged.
            Compiled runtime values are not validated."""
        policy = self._functional_policy()
        carry = self._budget_state(context)
        xp = namespace(carry.current_n_steps)

        updated = self._update_budget(
            carry,
            propose=lambda: policy.update_functional(
                current_steps=carry.current_n_steps,
                metrics=metrics,
                context=context,
            ),
            xp=xp,
        )
        return context.with_metadata(**{self.metadata_key: updated})

    def _dynamic_child(self) -> DynamicBudgetMutation:
        if not isinstance(self.mutation, DynamicBudgetMutation):
            raise NotImplementedError(
                "Child supports eager execution only; functional execution requires DynamicBudgetMutation."
            )
        return self.mutation

    def validate_functional_configuration(
        self, config: SMCConfig, state: ParticleState
    ) -> None:
        self._dynamic_child().validate_functional_configuration(
            config, state, max_steps=self._max_steps
        )

    def initialize_functional_state_and_context(
        self, state: ParticleState, context: StepContext
    ) -> tuple[ParticleState, StepContext]:
        return self._dynamic_child().initialize_functional_state_and_context(
            state, context
        )

    def apply_functional(
        self, state: ParticleState, context: StepContext, random_state: Any
    ) -> tuple[ParticleState, StepContext, Any, Any]:
        """Execute the frozen budget and adapt from this completed block only."""
        child = self._dynamic_child()
        context, n_steps = self.prepare_budget_functional(context)
        state, context, random_state, output, metric_state = (
            child.apply_dynamic_budget(
                state,
                context,
                random_state,
                n_steps=n_steps,
                max_steps=self._max_steps,
            )
        )
        metrics = child.metrics.result(metric_state, context)
        output = with_metrics(output, metrics)
        context = self.update_budget_functional(
            self._completed_metrics(output), context
        )
        return state, context, random_state, output

    @property
    def current_n_steps(self) -> int:
        """The frozen budget, or initial_steps before the first preparation."""
        return self._budget.current_n_steps

    @property
    def next_n_steps(self) -> int:
        """The pending budget, initially equal to initial_steps."""
        return self._budget.next_n_steps

    def _bound_policy_decision(
        self, value: int | Array, *, xp: Any | None = None
    ) -> int | Array:
        if xp is not None:
            return xp.clip(value, self._min_steps, self._max_steps)
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError("Mutation policy decisions must be integers.")
        return min(max(int(value), self._min_steps), self._max_steps)

    def _prepare_budget(
        self,
        carry: MutationBudgetState[_Scalar],
        iteration: _Scalar,
        *,
        initialise: Callable[[], _Scalar],
        xp: Any | None = None,
    ) -> MutationBudgetState[_Scalar]:
        """Choose once, clip, and freeze; promote pending on advancement."""

        control_xp = namespace() if xp is None else xp

        def prepare(_: None) -> MutationBudgetState[_Scalar]:
            budget = cond(
                carry.current_iteration >= 0,
                lambda _: carry.next_n_steps,
                lambda _: initialise(),
                None,
                xp=control_xp,
            )
            budget = self._bound_policy_decision(budget, xp=xp)
            return replace(
                carry,
                current_iteration=iteration,
                current_n_steps=budget,
                next_n_steps=budget,
            )

        return cond(
            carry.current_iteration == iteration,
            lambda _: carry,
            prepare,
            None,
            xp=control_xp,
        )

    def _update_budget(
        self,
        carry: MutationBudgetState[_Scalar],
        *,
        propose: Callable[[], _Scalar],
        xp: Any | None = None,
    ) -> MutationBudgetState[_Scalar]:
        """Accept the first completed block's bounded pending decision."""

        def update(_: None) -> MutationBudgetState[_Scalar]:
            return replace(
                carry,
                next_n_steps=self._bound_policy_decision(propose(), xp=xp),
                last_updated_iteration=carry.current_iteration,
            )

        return cond(
            (carry.current_iteration >= 0)
            & (carry.last_updated_iteration != carry.current_iteration),
            update,
            lambda _: carry,
            None,
            xp=namespace() if xp is None else xp,
        )

    def _freeze_budget(self, context: StepContext) -> int:
        self._budget = self._prepare_budget(
            self._budget,
            context.iteration,
            initialise=lambda: self.policy.initialise(
                initial_steps=self._budget.current_n_steps, context=context
            ),
        )
        return self._budget.current_n_steps

    def _prepare_mutation(
        self,
        context: StepContext,
        n_steps: int | None,
    ) -> tuple[StepContext, int | None]:
        return context, self._freeze_budget(context)

    def _adapt(
        self,
        metrics: MutationMetricsOutput,
        context: StepContext,
        *,
        initial_context: StepContext,
        n_steps: int | None,
    ) -> StepContext:
        del initial_context, n_steps
        carry = self._budget
        self._budget = self._update_budget(
            carry,
            propose=lambda: self.policy.update(
                current_steps=carry.current_n_steps,
                metrics=metrics,
                context=context,
            ),
        )
        return context

    def state_dict(self) -> dict[str, Any]:
        """Return the complete eager budget carry and nested mutable state."""
        return {
            "budget": asdict(self._budget),
            **super().state_dict(),
            "policy": self._nested_state(self.policy, "Policy"),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore the eager budget and nested state from the current schema.

        Legacy checkpoints are not supported."""
        required = {"budget", "mutation", "policy"}
        if missing := required.difference(state):
            raise ValueError(
                "AdaptMutationSteps state is missing required keys: "
                + ", ".join(sorted(missing))
            )
        budget = state["budget"]
        names = {field.name for field in fields(MutationBudgetState)}
        if not isinstance(budget, Mapping) or set(budget) != names:
            raise ValueError(
                "Budget state must contain all four carry fields."
            )
        values = {}
        for name, value in budget.items():
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"Budget state '{name}' must be an integer.")
            values[name] = int(value)
        carry = MutationBudgetState[int](**values)
        for value in (carry.current_n_steps, carry.next_n_steps):
            if not self._min_steps <= value <= self._max_steps:
                raise ValueError("Budget state is outside configured bounds.")
        if not -1 <= carry.last_updated_iteration <= carry.current_iteration:
            raise ValueError(
                "Budget state iteration markers are inconsistent."
            )
        super().load_state_dict(state)
        self._load_nested_state(self.policy, state["policy"], "policy")
        self._budget = carry


__all__ = ["AdaptMutationSteps"]
