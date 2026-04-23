"""Steps for Meta SMC algorithms."""

from __future__ import annotations

import copy
import logging

from orng import ArrayRNG

from ..state import ParticleState
from .base import SMCStep, StepContext

logger = logging.getLogger(__name__)


class NestedSMCStep(SMCStep):
    """SMC step that uses another SMC loop.

    Parameters
    ----------
    runner:
        An instance of ``SequentialMonteCarlo`` that will be used to run the
        nested SMC loop at each iteration.
    posterior:
        The posterior distribution object that will be updated with the log
        proposal from the nested SMC at each iteration.
    evidence_mode:
        Determines how the log-evidence from the nested SMC is handled. If
        "cumulative" (default), the log-evidence in the state will accumulate
        the evidence from all mutation steps. If "incremental", the log-evidence
        in the state will be reset to 0 before each mutation SMC run, so it will
        only reflect the evidence from the current mutation step.
    record_outputs:
        If ``True``, the full result of each mutation SMC run will be recorded
        in the context outputs under the key specified by ``output_key``. This
        can be useful for diagnostics or analysis of the mutation SMC runs.
    output_key:
        The key under which to record the mutation SMC results in the context
        outputs if ``record_outputs`` is ``True``. Default is
        "smc_mutation_results".
    run_kwargs:
        Additional keyword arguments to pass to the SMC runner's ``run`` method.
    """

    def __init__(
        self,
        smc_runner,
        posterior,
        *,
        evidence_mode="cumulative",
        record_outputs=False,
        run_kwargs=None,
        output_key="smc_mutation_results",
    ):
        self.runner = smc_runner
        self.run_kwargs = run_kwargs or {}
        self.base_posterior = posterior
        self.posterior = posterior
        self.record_outputs = record_outputs
        self.output_key = output_key

        if evidence_mode not in ("cumulative", "incremental"):
            raise ValueError(
                f"Invalid evidence_mode: {evidence_mode}. Must be 'cumulative' or 'incremental'."
            )

        self.evidence_mode = evidence_mode

    def apply(
        self,
        state: "ParticleState",
        context: StepContext,
        rng: "ArrayRNG",
    ) -> "ParticleState":
        if self.runner is None:
            raise ValueError("SMCMutation requires an SMC runner.")

        if self.evidence_mode == "incremental":
            # Reset the log-evidence in the state to 0 before running the mutation SMC, so that it only captures the incremental evidence from the mutation steps
            state = state.with_updates(log_evidence=0.0)

        logger.debug(
            "Running mutation SMC loop with %d particles.",
            len(state.particles),
        )
        metadata = copy.deepcopy(context.metadata)
        metadata.pop(
            "stop_requested", None
        )  # Ensure mutation SMC doesn't inherit stop request
        logger.debug("Mutation SMC context metadata: %s", context.metadata)
        result = self.runner.run(
            state,
            rng=rng,
            shared_context=context.shared,
            metadata=metadata,
            **self.run_kwargs,
        )
        final_context = result.context
        log_q_next = self.posterior.freeze_log_prob(final_context)
        self.posterior = self.posterior.with_log_proposal(log_q_next)
        if self.record_outputs:
            context.record_output(self.output_key, result)
        return result.final_state
