"""Complete mutation blocks with explicit rounds and metric ownership."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from ...utils.pytrees import pytree_dataclass
from ..base import SMCStep, validate_n_steps
from ..context import StepContext
from .base import MutationStep
from .metrics import (
    MutationMetrics,
    MutationMetricsOutput,
    MutationStepState,
)

if TYPE_CHECKING:
    from ...state import ParticleState


@pytree_dataclass(
    data_fields=("round_outputs", "after_round_outputs", "metrics")
)
@dataclass(frozen=True)
class RepeatedMutationStepOutput:
    """Mutation and post-round outputs plus one complete diagnostic record."""

    round_outputs: tuple[Any, ...]
    after_round_outputs: tuple[Any, ...]
    metrics: MutationMetricsOutput | None


class RepeatedMutationStep(SMCStep):
    """Run fixed mutation rounds with one shared collection of metric states.

    ``n_steps`` is the total default budget. If omitted, use the child's
    configured length times ``n_rounds``. ``apply_with_n_steps`` overrides the
    total for a single call. Budgets are divided evenly, with the remainder
    assigned to the first rounds.

    ``after_round`` is an ordinary SMCStep, executed after every round,
    including zero-step rounds and the final round. It never receives metric
    state. The mutation refreshes references on entering the following round
    without recording the intervening displacement or changing frozen scaling.

    Metrics are configured on the mutation child. Runtime state is local to
    the block call and passed explicitly through the rounds. Nested blocks
    share the same state and only the outermost block finalizes metrics.
    Fixed schedules execute all rounds even if a child requests stopping;
    the stop flags remain available to the enclosing SMC runner.
    """

    def __init__(
        self,
        mutation: MutationStep | RepeatedMutationStep,
        *,
        n_rounds: int = 1,
        n_steps: int | None = None,
        after_round: SMCStep | None = None,
    ) -> None:
        if not isinstance(mutation, (MutationStep, RepeatedMutationStep)):
            raise TypeError(
                "mutation must be a MutationStep or RepeatedMutationStep."
            )
        if after_round is not None and not isinstance(after_round, SMCStep):
            raise TypeError("after_round must be an SMCStep or None.")
        self.n_rounds = validate_n_steps(n_rounds)
        if self.n_rounds == 0:
            raise ValueError("n_rounds must be at least one.")
        self.mutation = mutation
        self.after_round = after_round
        self._n_steps = (
            mutation.n_steps * self.n_rounds
            if n_steps is None
            else validate_n_steps(n_steps)
        )

    @property
    def n_steps(self) -> int:
        """Default total number of transitions across all rounds."""
        return self._n_steps

    @property
    def metrics(self) -> MutationMetrics:
        """Static metric configuration owned by the mutation child."""
        return self.mutation.metrics

    def apply(self, state, context, rng):
        return self.apply_with_n_steps(
            state, context, rng, n_steps=self.n_steps
        )

    def apply_with_n_steps(self, state, context, rng, *, n_steps: int):
        state, context, output, mutation_state = (
            self.apply_with_mutation_state(
                state, context, rng, n_steps=n_steps, mutation_state=None
            )
        )
        return (
            state,
            context,
            replace(
                output,
                metrics=self.metrics.result(mutation_state, context),
            ),
        )

    def apply_with_mutation_state(
        self,
        state: ParticleState,
        context: StepContext,
        rng: Any,
        *,
        n_steps: int,
        mutation_state: MutationStepState | None = None,
    ) -> tuple[
        ParticleState,
        StepContext,
        RepeatedMutationStepOutput,
        MutationStepState | None,
    ]:
        """Run all rounds and return mutation state without finalizing it.

        ``None`` starts a new block. An existing state continues an enclosing
        block, preserving its metric scaling and cumulative bookkeeping.
        """
        n_steps = validate_n_steps(n_steps)
        quotient, remainder = divmod(n_steps, self.n_rounds)
        round_outputs = []
        after_round_outputs = []
        for index in range(self.n_rounds):
            state, context, output, mutation_state = (
                self.mutation.apply_with_mutation_state(
                    state,
                    context,
                    rng,
                    n_steps=quotient + (index < remainder),
                    mutation_state=mutation_state,
                )
            )
            round_outputs.append(output)
            after_output = None
            if self.after_round is not None:
                state, context, after_output = self.after_round(
                    state, context, rng
                )
            after_round_outputs.append(after_output)
        return (
            state,
            context,
            RepeatedMutationStepOutput(
                tuple(round_outputs), tuple(after_round_outputs), None
            ),
            mutation_state,
        )

    def initialize_functional_state_and_context(self, state, context):
        state, context = self.mutation.initialize_functional_state_and_context(
            state, context
        )
        if self.after_round is not None:
            state, context = (
                self.after_round.initialize_functional_state_and_context(
                    state, context
                )
            )
        return state, context

    def state_dict(self) -> dict[str, Any]:
        return {
            "mutation": self.mutation.state_dict(),
            "after_round": None
            if self.after_round is None
            else self.after_round.state_dict(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        mutation_state = state["mutation"]
        after_state = state["after_round"]
        if not isinstance(mutation_state, Mapping):
            raise TypeError(
                "RepeatedMutationStep mutation state must be a mapping."
            )
        if self.after_round is None:
            if after_state is not None:
                raise ValueError(
                    "RepeatedMutationStep checkpoint contains an after_round step."
                )
        elif not isinstance(after_state, Mapping):
            raise TypeError(
                "RepeatedMutationStep after_round state must be a mapping."
            )
        self.mutation.load_state_dict(mutation_state)
        if self.after_round is not None:
            self.after_round.load_state_dict(after_state)
