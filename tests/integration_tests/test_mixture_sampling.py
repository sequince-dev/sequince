"""Known-target checks for fixed and between-iteration adaptive mixtures."""

import numpy as np
import pytest
from orng import RandomGenerator

from sequince.state import ParticleState
from sequince.steps import (
    AdaptMoveWeights,
    AdaptMutationSteps,
    JumpDistancePerEvaluation,
    MixtureMutation,
    StepContext,
)
from sequince.steps.mutation import (
    CallableMutationKernel,
    GaussianRandomWalk,
    MetropolisHastingsMutation,
)
from sequince.steps.mutation.adaptive.policies.acceptance import (
    AcceptanceRatePolicy,
)


@pytest.mark.parametrize("mode", ["fixed", "length", "joint"])
def test_asymmetric_normal_target(mode):
    def target(x):
        return -0.5 * np.sum(x**2, axis=-1)

    # Independent N(0.7, 1) proposals are asymmetric. Omitting this correction
    # would shift the stationary mean and fail the accuracy check.
    independent = CallableMutationKernel(
        lambda x, rng: rng.normal(size=x.shape) + 0.7,
        log_transition_ratio_fn=lambda x, y: -0.5
        * np.sum((x - 0.7) ** 2 - (y - 0.7) ** 2, axis=-1),
    )
    mixture = MixtureMutation(
        {
            "walk": MetropolisHastingsMutation(
                target, GaussianRandomWalk(scale=0.4)
            ),
            "independent": MetropolisHastingsMutation(target, independent),
        },
        n_steps=8,
    )
    mutation = mixture
    if mode == "joint":
        mutation = AdaptMoveWeights(mixture, JumpDistancePerEvaluation())
    if mode != "fixed":
        mutation = AdaptMutationSteps(
            mutation,
            AcceptanceRatePolicy(4, min_steps=2, max_steps=20),
            initial_steps=8,
            min_steps=2,
            max_steps=20,
        )
    rng = RandomGenerator(seed=731)
    state = ParticleState(
        particles=np.full((8000, 1), 4.0), log_weights=np.zeros(8000)
    )
    cost = 0
    for t in range(8):
        state, context, output = mutation(state, StepContext(t), rng)
        cost += int(output.evaluation_counts.log_target_evaluations)
        assert (
            context.evaluation_counts.log_target_evaluations
            == output.evaluation_counts.log_target_evaluations
        )
        assert (
            output.evaluation_counts.log_target_evaluations
            == 2 * 8000 * output.metrics.n_steps
        )
    # About four Monte Carlo standard errors for independent target draws.
    assert abs(np.mean(state.particles)) < 0.05
    assert abs(np.var(state.particles) - 1) < 0.07
    assert cost > 0
    np.testing.assert_array_equal(state.log_weights, np.zeros(8000))
