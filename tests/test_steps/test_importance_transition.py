from __future__ import annotations

import numpy as np
import pytest

from sequince.array_api import namespace
from sequince.state import ParticleState
from sequince.steps import ImportanceTransition
from sequince.target import TargetEvaluation


def test_importance_transition_represents_bootstrap_filter(
    context_factory,
    rng,
) -> None:
    particles = np.array([[-1.0], [0.5], [2.0]])
    state = ParticleState(particles)
    context = context_factory(observation=1.25)

    def transition(previous, context, rng):
        del context
        return previous + np.asarray(rng.normal(size=previous.shape)), None

    def log_observation_prob(current, observation):
        return -0.5 * np.square(current[:, 0] - observation)

    def log_increment(previous, current, auxiliary, context):
        del previous, auxiliary
        return log_observation_prob(current, context.observation)

    updated, returned_context, output = ImportanceTransition(
        transition=transition,
        log_increment=log_increment,
        normalize_after=False,
    )(state, context, rng)

    expected_increment = log_observation_prob(
        updated.particles,
        context.observation,
    )
    np.testing.assert_allclose(
        updated.log_weights,
        state.log_weights + expected_increment,
    )
    np.testing.assert_allclose(
        output["log_weight_increment"],
        expected_increment,
    )
    assert output["auxiliary"] is None
    assert returned_context is context


def test_importance_transition_increment_uses_old_new_and_auxiliary(
    context_factory,
    rng,
) -> None:
    particles = np.array([[-1.0], [0.5], [2.0]])
    log_weights = np.array([-0.4, -1.2, -0.8])
    state = ParticleState(particles, log_weights=log_weights)
    context = context_factory(metadata={"scale": 0.25})

    def transition(previous, context, rng):
        del rng
        proposed = previous + context.metadata["scale"]
        log_proposal = -np.square(proposed[:, 0] - previous[:, 0])
        return proposed, log_proposal

    def log_increment(previous, current, log_proposal, context):
        return (
            previous[:, 0]
            + 2.0 * current[:, 0]
            - log_proposal
            + context.metadata["scale"]
        )

    updated, returned_context, output = ImportanceTransition(
        transition=transition,
        log_increment=log_increment,
    )(state, context, rng)

    expected_increment = log_increment(
        particles,
        particles + 0.25,
        output["auxiliary"],
        context,
    )
    unnormalized = log_weights + expected_increment
    expected_log_norm = np.logaddexp.reduce(unnormalized)
    np.testing.assert_allclose(
        updated.log_weights,
        unnormalized - expected_log_norm,
    )
    assert updated.log_evidence == pytest.approx(expected_log_norm)
    np.testing.assert_allclose(
        output["log_weight_increment"],
        expected_increment,
    )
    assert returned_context is context


def test_importance_transition_invalidates_evaluation_and_preserves_ancestry(
    context_factory,
    rng,
) -> None:
    particles = np.arange(6.0).reshape(3, 2)
    ancestors = np.array([4, 2, 7])
    state = ParticleState(
        particles,
        initial_ancestor_indices=ancestors,
        evaluation=TargetEvaluation(components={"cached": np.arange(3.0)}),
    )
    context = context_factory()

    updated, returned_context, _ = ImportanceTransition(
        transition=lambda previous, context, rng: (previous + 1.0, None),
        log_increment=lambda previous, current, auxiliary, context: np.zeros(
            previous.shape[0]
        ),
    )(state, context, rng)

    assert updated.evaluation is None
    np.testing.assert_array_equal(
        updated.initial_ancestor_indices,
        ancestors,
    )
    assert returned_context is context


def test_importance_transition_supports_scalar_increment_for_one_particle(
    context_factory,
    rng,
) -> None:
    state = ParticleState(np.array([[2.0]]), log_weights=np.array([0.0]))

    updated, _, output = ImportanceTransition(
        transition=lambda previous, context, rng: (previous + 1.0, None),
        log_increment=lambda previous, current, auxiliary, context: 0.75,
        normalize_after=False,
    )(state, context_factory(), rng)

    np.testing.assert_allclose(updated.log_weights, np.array([0.75]))
    assert output["log_weight_increment"].shape == (1,)


def test_importance_transition_validates_population_and_increment_shapes(
    context_factory,
    rng,
) -> None:
    state = ParticleState(np.zeros((3, 1)))
    context = context_factory()

    wrong_population = ImportanceTransition(
        transition=lambda previous, context, rng: (previous[:2], None),
        log_increment=lambda previous, current, auxiliary, context: np.zeros(
            2
        ),
    )
    with pytest.raises(ValueError, match="one particle per input particle"):
        wrong_population(state, context, rng)

    wrong_increment = ImportanceTransition(
        transition=lambda previous, context, rng: (previous + 1.0, None),
        log_increment=lambda previous, current, auxiliary, context: np.zeros(
            2
        ),
    )
    with pytest.raises(ValueError, match="one value per particle"):
        wrong_increment(state, context, rng)


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax", "cupy"])
def test_importance_transition_preserves_array_backend(
    backend,
    backend_factory,
    context_factory,
) -> None:
    case = backend_factory(backend)
    xp = case.xp

    def transition(previous, context, rng):
        del context
        return previous + rng.normal(size=previous.shape), None

    def log_increment(previous, current, auxiliary, context):
        del previous, auxiliary, context
        return xp.sum(current, axis=1)

    updated, _, output = ImportanceTransition(
        transition=transition,
        log_increment=log_increment,
    )(case.state(), context_factory(), case.rng(1234))

    assert namespace(updated.particles) is case.xp
    assert namespace(updated.log_weights) is case.xp
    assert namespace(output["log_weight_increment"]) is case.xp
