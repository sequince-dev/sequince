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
