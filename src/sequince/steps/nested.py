"""Nested SMC steps."""

from __future__ import annotations

import copy
import logging
from typing import Any, Callable

from orng import ArrayRNG

from ..state import ParticleState
from ..utils import _call_with_optional_context
from .base import SMCStep, StepContext

logger = logging.getLogger(__name__)


class NestedSMCStep(SMCStep):
    """SMC step that runs another SMC loop in the same target space."""

    def __init__(
        self,
        smc_runner,
        posterior,
        *,
        record_outputs=False,
        run_kwargs=None,
        output_key="smc_mutation_results",
    ):
        self.runner = smc_runner
        self.run_kwargs = run_kwargs or {}
        self.posterior = posterior
        self.record_outputs = record_outputs
        self.output_key = output_key

    def make_input_log_proposal(
        self,
        context: StepContext,
    ) -> Callable[..., Any] | None:
        base_log_proposal = getattr(self.posterior, "log_proposal", None)
        transformation = context.metadata.get("smc_transformation")
        if base_log_proposal is None or transformation is None:
            return base_log_proposal

        def _transformed_log_proposal(
            particles,
            inner_context,
        ):
            old_particles = transformation.inverse(particles, inner_context)
            values = _call_with_optional_context(
                base_log_proposal,
                (old_particles,),
                inner_context,
            )
            values = values + _call_with_optional_context(
                transformation.log_auxiliary_density,
                (particles,),
                inner_context,
            )
            values = values - _call_with_optional_context(
                transformation.log_jacobian,
                (particles,),
                inner_context,
            )
            return values

        return _transformed_log_proposal

    def make_output_log_proposal(
        self,
        context: StepContext,
    ) -> Callable[..., Any] | None:
        return self.posterior.freeze_log_prob(context)

    def apply(
        self,
        state: "ParticleState",
        context: StepContext,
        rng: "ArrayRNG",
    ) -> "ParticleState":
        if self.runner is None:
            raise ValueError("NestedSMCStep requires an SMC runner.")

        input_log_proposal = self.make_input_log_proposal(context)
        if input_log_proposal is not None:
            self.posterior.update_log_proposal(input_log_proposal)

        input_state = state
        logger.debug(
            "Running nested SMC loop with %d particles.",
            len(input_state.particles),
        )
        metadata = copy.deepcopy(context.metadata)
        metadata.pop("stop_requested", None)
        result = self.runner.run(
            input_state,
            rng=rng,
            shared_context=context.shared,
            metadata=metadata,
            **self.run_kwargs,
        )
        final_context = result.context
        output_log_proposal = self.make_output_log_proposal(final_context)
        if output_log_proposal is not None:
            self.posterior.update_log_proposal(output_log_proposal)
        if self.record_outputs:
            context.record_output(self.output_key, result)
        return result.final_state
