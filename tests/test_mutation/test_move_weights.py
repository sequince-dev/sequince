import copy

import numpy as np
import pytest
from orng import RandomGenerator
from test_mixture import make_mixture

from sequince.steps.mutation import AdaptMutationSteps
from sequince.steps.mutation.adaptive.weights import AdaptMoveWeights


class AlternatingPolicy:
    def __init__(self):
        self.calls = 0

    def initialise(self, *, initial_weights, context):
        return [1, 0]

    def update(self, *, current_weights, output, context):
        self.calls += 1
        return current_weights[::-1]

    def state_dict(self):
        return {"calls": self.calls}

    def load_state_dict(self, state):
        self.calls = state["calls"]


def test_frozen_weights_and_length_composition(
    particle_state, context_factory
):
    policy = AlternatingPolicy()
    weights = AdaptMoveWeights(make_mixture(), policy)
    adaptive = AdaptMutationSteps(
        weights, initial_steps=3, min_steps=1, max_steps=10
    )
    for t, expected in ((0, (1, 0)), (0, (1, 0)), (1, (0, 1))):
        _, _, out = adaptive(particle_state, context_factory(t=t), 2)
        assert out.weights == expected
        assert out.metrics.n_steps == 3
    assert policy.calls == 2
    assert weights.mutation.weights == (0.5, 0.5)


def test_checkpoint_resume(particle_state, context_factory):
    wrapper = AdaptMoveWeights(make_mixture(), AlternatingPolicy())
    rng = RandomGenerator(seed=56)
    state, _, _ = wrapper(particle_state, context_factory(t=0), rng)
    saved = copy.deepcopy(wrapper.state_dict())
    rng_state = copy.deepcopy(rng.state_dict())
    restored = AdaptMoveWeights(make_mixture(), AlternatingPolicy())
    restored.load_state_dict(saved)
    other_rng = RandomGenerator.from_state_dict(rng_state)
    for t in (0, 1, 2):
        a, _, out = wrapper(state, context_factory(t=t), rng)
        b, _, other = restored(state, context_factory(t=t), other_rng)
        np.testing.assert_array_equal(a.particles, b.particles)
        assert out.moves == other.moves
        assert out.weights == other.weights
        assert restored.policy.calls == wrapper.policy.calls
        state = a


@pytest.mark.parametrize("decision", [[0, 0], [np.nan, 1], [-1, 1], [1]])
def test_invalid_policy_decisions(decision, particle_state, context_factory):
    class InvalidPolicy(AlternatingPolicy):
        def update(self, **kwargs):
            return decision

    wrapper = AdaptMoveWeights(make_mixture(), InvalidPolicy())
    with pytest.raises(ValueError):
        wrapper(particle_state, context_factory(), 1)
    assert wrapper.next_weights is None
