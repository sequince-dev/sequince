import numpy as np
import pytest

from sequince.steps.base import SMCStep
from sequince.steps.mutation import AdaptMutationSteps, MutationMetricsOutput
from sequince.steps.mutation.adaptive.policies.acceptance import (
    AcceptanceRatePolicy,
)


@pytest.mark.parametrize(
    "accepted,expected", [(100, 3), (50, 5), (25, 10), (0, 25)]
)
def test_acceptance_budget(accepted, expected, context_factory):
    policy = AcceptanceRatePolicy(
        2.5, min_steps=1, max_steps=100, min_acceptance=0.1
    )
    assert (
        policy.update(
            current_steps=10,
            metrics=MutationMetricsOutput(10, 100, accepted),
            context=context_factory(),
        )
        == expected
    )


@pytest.mark.parametrize("current,expected", [(0, 1), (5, 5)])
def test_no_attempts_retains_budget(current, expected, context_factory):
    assert (
        AcceptanceRatePolicy(2, min_steps=1, max_steps=100).update(
            current_steps=current,
            metrics=MutationMetricsOutput(0, 0, 0),
            context=context_factory(),
        )
        == expected
    )


@pytest.mark.parametrize("target", [0, -1, np.nan, np.inf])
def test_invalid_target(target):
    with pytest.raises(ValueError):
        AcceptanceRatePolicy(target, min_steps=1, max_steps=100)


@pytest.mark.parametrize("floor", [0, -1, 2, np.nan, np.inf])
def test_invalid_floor(floor):
    with pytest.raises(ValueError):
        AcceptanceRatePolicy(
            2, min_steps=1, max_steps=100, min_acceptance=floor
        )


@pytest.mark.parametrize(
    "attempted,accepted",
    [(10, -1), (10, 11), (0, 1), (np.inf, 1), (10, np.nan)],
)
def test_invalid_counts(attempted, accepted, context_factory):
    with pytest.raises(ValueError):
        AcceptanceRatePolicy(2, min_steps=1, max_steps=100).update(
            current_steps=2,
            metrics=MutationMetricsOutput(2, attempted, accepted),
            context=context_factory(),
        )


def test_budget_is_lagged_bounded_and_updated_once(
    particle_state, context_factory, rng
):
    class Mutation(SMCStep):
        def __init__(self):
            self.lengths = []

        def apply_with_n_steps(self, state, context, rng, *, n_steps):
            self.lengths.append(n_steps)
            acceptance = 0 if context.t == 0 else 1
            attempted = n_steps * len(state.particles)
            return (
                state,
                context,
                {
                    "metrics": MutationMetricsOutput(
                        n_steps, attempted, acceptance * attempted
                    )
                },
            )

    mutation = Mutation()
    adaptive = AdaptMutationSteps(
        mutation,
        AcceptanceRatePolicy(2.5, min_steps=3, max_steps=20),
        initial_steps=4,
        min_steps=0,
        max_steps=100,
    )
    for t in (0, 0, 1, 2):
        adaptive(particle_state, context_factory(t=t), rng)
    assert mutation.lengths == [4, 4, 20, 3]


@pytest.mark.parametrize(
    "accepted,expected", [(100, 4), (50, 6), (1, 12), (0, 12)]
)
def test_policy_clips_move_count(accepted, expected, context_factory):
    policy = AcceptanceRatePolicy(target_steps=3, min_steps=4, max_steps=12)
    assert (
        policy.update(
            current_steps=8,
            metrics=MutationMetricsOutput(8, 100, accepted),
            context=context_factory(),
        )
        == expected
    )


@pytest.mark.parametrize("steps,expected", [(0, 4), (8, 8), (100, 12)])
def test_policy_bounds_initial_and_unmeasured_budget(
    steps, expected, context_factory
):
    policy = AcceptanceRatePolicy(target_steps=3, min_steps=4, max_steps=12)
    context = context_factory()
    assert policy.initialise(initial_steps=steps, context=context) == expected
    assert (
        policy.update(
            current_steps=steps,
            metrics=MutationMetricsOutput(0, 0, 0),
            context=context,
        )
        == expected
    )


@pytest.mark.parametrize(
    "minimum,maximum,error",
    [
        (-1, 10, ValueError),
        (5, 4, ValueError),
        (1.5, 10, TypeError),
        (1, True, TypeError),
    ],
)
def test_invalid_move_bounds(minimum, maximum, error):
    with pytest.raises(error):
        AcceptanceRatePolicy(3, min_steps=minimum, max_steps=maximum)
