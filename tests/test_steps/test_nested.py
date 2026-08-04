from __future__ import annotations

from unittest.mock import Mock

import numpy as np

from sequince.posterior import TemperedPosterior
from sequince.result import SMCResult
from sequince.state import ParticleState
from sequince.steps.context import LOG_PROPOSAL_OVERRIDE_KEY
from sequince.steps.nested import NestedSMCStep


def test_nested_smc_invalidates_evaluation_after_changing_proposal(
    context_factory,
    rng,
) -> None:
    particles = np.asarray([[0.0], [1.0]])
    posterior = TemperedPosterior(
        log_prior=lambda values, context: np.zeros(len(values)),
        log_likelihood=lambda values, context: -values[:, 0],
    )
    inner_context = context_factory(metadata={"temperature": 1.0})
    evaluation = posterior.evaluate(particles, inner_context)
    final_state = ParticleState(particles, evaluation=evaluation)
    runner = Mock()
    runner.run.return_value = SMCResult([final_state], inner_context)
    outer_context = context_factory(metadata={"temperature": 0.0})

    updated = NestedSMCStep(runner, posterior)(
        ParticleState(particles),
        outer_context,
        rng,
    )

    assert updated.evaluation is None
    assert callable(outer_context.metadata[LOG_PROPOSAL_OVERRIDE_KEY])

    next_evaluation = posterior.prepare_evaluation(
        updated.particles,
        outer_context,
        evaluation=updated.evaluation,
    )
    assert "log_proposal" in next_evaluation
    np.testing.assert_allclose(
        posterior.log_increment(
            updated.particles,
            outer_context,
            evaluation=next_evaluation,
        ),
        0.0,
    )
