from __future__ import annotations

from typing import TYPE_CHECKING

from ..base import SMCStep
from ..context import StepContext
from ..control import ConditionalStepOutput

if TYPE_CHECKING:
    from orng import RandomGenerator

    from ...state import ParticleState


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
            updated = state
        else:
            updated, context, child_output = self.mutation(state, context, rng)
        return (
            updated,
            context,
            ConditionalStepOutput(not mutation_skipped, child_output),
        )
