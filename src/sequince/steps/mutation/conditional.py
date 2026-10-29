from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from ..base import FixedLengthMutation, SMCStep, validate_n_steps
from ..context import StepContext
from ..control import ConditionalStepOutput

if TYPE_CHECKING:
    from orng import RandomGenerator

    from ...state import ParticleState

logger = logging.getLogger(__name__)


class MutateIfResampled(SMCStep):
    """Apply a mutation step only when the preceding step resampled.

    This step reads the ``resampled`` flag written to the current
    :class:`StepContext` by Sequince's resampling steps. The resulting
    execution decision is returned explicitly in the step output.

    Parameters
    ----------
    mutation:
        Mutation step to apply after resampling.
    """

    def __init__(self, mutation: SMCStep) -> None:
        self.mutation = mutation

    def apply(
        self,
        state: ParticleState,
        context: StepContext,
        rng: "RandomGenerator",
    ) -> tuple[ParticleState, StepContext, ConditionalStepOutput]:
        mutation_skipped = not bool(context.metadata.get("resampled", False))
        child_output = None
        if mutation_skipped:
            logger.debug(
                "Iteration %d: Skipping mutation step because no resampling occurred.",
                context.t,
            )
            updated = state
        else:
            logger.debug(
                "Iteration %d: Applying mutation step because resampling occurred.",
                context.t,
            )
            updated, context, child_output = self.mutation(state, context, rng)
        return (
            updated,
            context,
            ConditionalStepOutput(not mutation_skipped, child_output),
        )

    def apply_with_n_steps(
        self,
        state: ParticleState,
        context: StepContext,
        rng: "RandomGenerator",
        *,
        n_steps: int,
    ) -> tuple[ParticleState, StepContext, ConditionalStepOutput]:
        n_steps = validate_n_steps(n_steps)
        if not isinstance(self.mutation, FixedLengthMutation):
            raise TypeError(
                "MutateIfResampled's child does not support fixed-length mutation."
            )
        if not bool(context.metadata.get("resampled", False)):
            return state, context, ConditionalStepOutput(False)
        updated, context, output = self.mutation.apply_with_n_steps(
            state, context, rng, n_steps=n_steps
        )
        return updated, context, ConditionalStepOutput(True, output)

    def state_dict(self) -> dict[str, Any]:
        """Preserve the child mutation's adaptive and proposal state."""
        return {"mutation": self.mutation.state_dict()}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore nested state, accepting older empty wrapper checkpoints."""
        if "mutation" not in state:
            return
        nested = state["mutation"]
        if not isinstance(nested, Mapping):
            raise TypeError(
                "MutateIfResampled state 'mutation' must be a mapping."
            )
        self.mutation.load_state_dict(nested)
