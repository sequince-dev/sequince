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


@pytest.mark.parametrize("step_fn", ["pcn", "tpcn"])
@pytest.mark.parametrize("adaptive", [False, True])
def test_minipcn_mixture_normal_target(step_fn: str, adaptive: bool) -> None:
    from sequince._typing import Array
    from sequince.steps.mutation import MiniPCNMutation

    def target(particles: Array) -> Array:
        return -0.5 * np.sum(particles**2, axis=-1)

    mixture = MixtureMutation(
        {
            "walk": MetropolisHastingsMutation(
                target, GaussianRandomWalk(scale=0.4)
            ),
            "minipcn": MiniPCNMutation(target, step_fn=step_fn, rho=0.8),
        },
        n_steps=12,
    )
    mutation = mixture
    if adaptive:
        mutation = AdaptMutationSteps(
            AdaptMoveWeights(mixture, JumpDistancePerEvaluation()),
            AcceptanceRatePolicy(6, min_steps=4, max_steps=20),
            initial_steps=12,
            min_steps=4,
            max_steps=20,
        )
    # A shifted initial population exercises the asymmetric MiniPCN proposal
    # correction: its fitted reference is initially different from the target.
    particles = np.random.default_rng(93).normal(size=(6000, 2)) + 1.0
    state = ParticleState(particles)
    rng = RandomGenerator(seed=82)
    for iteration in range(8):
        state, _, output = mutation(state, StepContext(iteration), rng)
        assert output.evaluation_counts.log_target_evaluations == (
            6000
            * (
                2 * output.moves["walk"].selections
                + 3 * output.moves["minipcn"].selections
            )
        )
    # Roughly four standard errors for independent standard-normal draws.
    np.testing.assert_allclose(np.mean(state.particles, axis=0), 0, atol=0.055)
    np.testing.assert_allclose(
        np.cov(state.particles.T), np.eye(2), atol=0.075
    )
