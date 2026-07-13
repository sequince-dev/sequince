"""Nested SMC steps."""

from __future__ import annotations

import copy
import logging
from collections.abc import Mapping
from typing import Any

from orng import RandomGenerator

from ..state import ParticleState
from .base import SMCStep
from .context import LOG_PROPOSAL_OVERRIDE_KEY, StepContext

logger = logging.getLogger(__name__)

NESTED_SMC_OUTPUT_KEY = "smc_mutation_results"


class NestedSMCStep(SMCStep):
    """SMC step that runs another SMC loop in the same target space."""

    def __init__(
        self,
        smc_runner,
        posterior,
        *,
        record_outputs=False,
        run_kwargs=None,
        output_key=NESTED_SMC_OUTPUT_KEY,
        log_proposal_override_key: str = LOG_PROPOSAL_OVERRIDE_KEY,
    ):
        self.runner = smc_runner
        self.run_kwargs = run_kwargs or {}
        self.posterior = posterior
        self.record_outputs = record_outputs
        self.output_key = output_key
        self.log_proposal_override_key = log_proposal_override_key

    def apply(
        self,
        state: "ParticleState",
        context: StepContext,
        rng: "RandomGenerator",
    ) -> "ParticleState":
        if self.runner is None:
            raise ValueError("NestedSMCStep requires an SMC runner.")

        input_state = state
        logger.debug(
            "Running nested SMC loop with %d particles.",
            len(input_state.particles),
        )
        # TODO: is there a better way to do this? Avoid copying?
        metadata = copy.deepcopy(context.metadata)
        metadata.pop("stop_requested", None)
        metadata.pop("stop_immediately", None)
        result = self.runner.run(
            input_state,
            rng=rng,
            shared_context=context.shared,
            metadata=metadata,
            **self.run_kwargs,
        )
        final_context = result.context
        output_log_proposal = self.posterior.freeze_log_prob(
            final_context,
            log_proposal_override=final_context.metadata.get(
                self.log_proposal_override_key
            ),
        )
        context.metadata[self.log_proposal_override_key] = output_log_proposal
        if self.record_outputs:
            context.record_output(self.output_key, result)
        return result.final_state

    def state_dict(self) -> dict[str, Any]:
        return {"runner": self.runner.state_dict()}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        runner_state = state.get("runner", {})
        if not isinstance(runner_state, Mapping):
            raise TypeError("NestedSMCStep state 'runner' must be a mapping.")
        self.runner.load_state_dict(runner_state)
