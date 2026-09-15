import numpy as np
import pytest

from sequince.steps.mutation import (
    GaussianRandomWalk,
    MetropolisHastingsMutation,
)
from sequince.steps.mutation.mixture import MixtureMutation


def log_prob(x):
    return -0.5 * (x**2).sum(axis=-1)


def make_mixture(weights=None):
    return MixtureMutation(
        {
            name: MetropolisHastingsMutation(
                log_prob, GaussianRandomWalk(scale=scale), n_steps=99
            )
            for name, scale in (("small", 0.1), ("large", 2.0))
        },
        weights,
        n_steps=20,
    )


def test_seeded_selection_and_exact_budget(particle_state, context_factory):
    mixture = make_mixture()
    a, _, out = mixture(particle_state, context_factory(), 42)
    b, _, other = mixture(particle_state, context_factory(), 42)
    np.testing.assert_array_equal(a.particles, b.particles)
    assert out.metrics.n_steps == 20
    assert sum(move.selections for move in out.moves.values()) == 20
    assert all(0 < move.selections < 20 for move in out.moves.values())
    assert out.metrics.n_attempted == 20 * len(a.particles)
    assert out.moves == other.moves


def test_zero_steps(particle_state, context_factory, rng):
    state, context, output = make_mixture().apply_with_n_steps(
        particle_state, context_factory(), rng, n_steps=0
    )
    assert state is particle_state
    assert output.metrics.n_attempted == 0
    assert output.metrics.acceptance == 0
    assert all(move.selections == 0 for move in output.moves.values())
    assert context.evaluation_counts.log_target_evaluations == 0


@pytest.mark.parametrize(
    "weights", [[], [0, 0], [-1, 1], [np.nan, 1], [np.inf, 1]]
)
def test_invalid_weights(weights):
    with pytest.raises(ValueError):
        make_mixture(weights)


def test_override_and_functional_rejection(
    particle_state, context_factory, rng
):
    mixture = make_mixture()
    _, _, output = mixture.apply_with_n_steps(
        particle_state, context_factory(), rng, n_steps=4, weights=[0, 1]
    )
    assert output.moves["large"].selections == 4
    assert mixture.weights == (0.5, 0.5)
    with pytest.raises(NotImplementedError, match="eager"):
        mixture.apply_functional(particle_state, context_factory(), None)


def test_counts_and_adaptive_length(particle_state, context_factory, rng):
    from sequince.steps.mutation import AdaptMutationSteps
    from sequince.steps.mutation.adaptive.policies.acceptance import (
        AcceptanceRatePolicy,
    )
    from sequince.target import EvaluationCounts

    mixture = make_mixture()
    adaptive = AdaptMutationSteps(
        mixture,
        AcceptanceRatePolicy(2, min_steps=1, max_steps=100),
        initial_steps=5,
        min_steps=1,
        max_steps=100,
    )
    initial = context_factory().add_evaluation_counts(
        EvaluationCounts(log_target_evaluations=123)
    )
    _, context, output = adaptive(particle_state, initial, rng)
    total = output.evaluation_counts.log_target_evaluations
    assert total == 2 * 5 * len(particle_state.particles)
    assert context.evaluation_counts.log_target_evaluations == total + 123
    assert (
        sum(
            x.evaluation_counts.log_target_evaluations
            for x in output.moves.values()
        )
        == total
    )
    assert output.metrics.n_accepted == sum(
        x.metrics.n_accepted for x in output.moves.values()
    )
    assert adaptive.next_n_steps == min(
        100, int(np.ceil(2 / max(float(output.metrics.acceptance), 1e-6)))
    )


def test_checkpoint_and_order(particle_state, context_factory):
    import copy

    from orng import RandomGenerator

    mixture = make_mixture()
    rng = RandomGenerator(seed=17)
    state, context, _ = mixture(particle_state, context_factory(), rng)
    checkpoint = copy.deepcopy(mixture.state_dict())
    saved_rng = RandomGenerator.from_state_dict(
        copy.deepcopy(rng.state_dict())
    )
    a, _, out_a = mixture(state, context, rng)
    restored = make_mixture()
    restored.load_state_dict(checkpoint)
    b, _, out_b = restored(state, context, saved_rng)
    np.testing.assert_array_equal(a.particles, b.particles)
    assert out_a == out_b
    checkpoint["names"] = tuple(reversed(checkpoint["names"]))
    with pytest.raises(ValueError, match="order"):
        restored.load_state_dict(checkpoint)
