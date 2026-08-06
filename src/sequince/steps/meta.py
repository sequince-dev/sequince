"""Meta SMC steps that run another SMC loop in the same target space."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from orng import RandomGenerator

from ..state import ParticleState
from .base import SMCStep
from .context import StepContext

logger = logging.getLogger(__name__)

META_SMC_OUTPUT_KEY = "smc_mutation_results"


class MetaSMCStep(SMCStep):
    """SMC step that runs another SMC loop in the same target space."""

    def __init__(
        self,
        smc_runner,
        posterior,
        *,
        record_outputs=False,
        run_kwargs=None,
        output_key=META_SMC_OUTPUT_KEY,
    ):
        self.runner = smc_runner
        self.run_kwargs = run_kwargs or {}
        self.posterior = posterior
        self.record_outputs = record_outputs
        self.output_key = output_key

    def apply(
        self,
        state: "ParticleState",
        context: StepContext,
        rng: "RandomGenerator",
    ) -> tuple["ParticleState", StepContext, dict[str, Any] | None]:
        if self.runner is None:
            raise ValueError("MetaSMCStep requires an SMC runner.")

        input_state = state
        logger.debug(
            "Running meta SMC loop with %d particles.",
            len(input_state.particles),
        )
        # TODO: is there a better way to do this? Avoid copying?
        metadata = dict(context.metadata)
        result = self.runner.run(
            input_state,
            rng=rng,
            shared_context=context.shared,
            metadata=metadata,
            **self.run_kwargs,
        )
        final_context = result.context
        override_key = self.posterior.log_proposal_override_key
        output_log_proposal = self.posterior.freeze_log_prob(
            final_context,
            log_proposal_override=final_context.metadata.get(override_key),
        )
        context = context.with_metadata(**{override_key: output_log_proposal})
        output = {self.output_key: result} if self.record_outputs else None
        # The retained evaluation was computed using the inner loop's proposal.
        # Installing the terminal density as the next proposal changes the
        # target components required by the outer context, so that evaluation
        # must not be reused by a subsequent weighting or mutation step.
        return (
            result.final_state.with_updates(evaluation=None),
            context,
            output,
        )

    def state_dict(self) -> dict[str, Any]:
        return {"runner": self.runner.state_dict()}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        runner_state = state.get("runner", {})
        if not isinstance(runner_state, Mapping):
            raise TypeError("MetaSMCStep state 'runner' must be a mapping.")
        self.runner.load_state_dict(runner_state)


__all__ = [
    "META_SMC_OUTPUT_KEY",
    "MetaSMCStep",
]
