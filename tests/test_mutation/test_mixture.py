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
    assert out_a.moves == out_b.moves
    assert out_a.weights == out_b.weights
    checkpoint["names"] = tuple(reversed(checkpoint["names"]))
    with pytest.raises(ValueError, match="order"):
        restored.load_state_dict(checkpoint)


def test_displacement_uses_frozen_common_scale(
    particle_state, context_factory
):
    from sequince.steps.mutation import CallableMutationKernel

    def flat(x):
        return np.zeros(len(x))

    def move_by(amount):
        return MetropolisHastingsMutation(
            flat,
            CallableMutationKernel(lambda x, rng: x + amount, symmetric=True),
        )

    # Deterministic translations isolate diagnostics; this is not a sampling
    # correctness test (the proposal deliberately has no reverse support).
    particles = np.array([[0.0, 7.0], [2.0, 7.0], [0.0, 7.0], [2.0, 7.0]])
    mixture = MixtureMutation(
        {"one": move_by(1), "two": move_by(2)}, n_steps=10
    )
    _, _, output = mixture(
        particle_state.replace_particles(particles), context_factory(), 5
    )
    np.testing.assert_array_equal(output.coordinate_scale, [1.0, 1.0])
    for name, amount in (("one", 1), ("two", 2)):
        move = output.moves[name]
        assert (
            move.scaled_squared_displacement == move.selections * 8 * amount**2
        )


def test_displacement_zero_for_rejections(particle_state, context_factory):
    from sequince.steps.mutation import CallableMutationKernel

    def only_origin(x):
        return np.where(np.all(x == 0, axis=-1), 0.0, -np.inf)

    child = MetropolisHastingsMutation(
        only_origin,
        CallableMutationKernel(lambda x, rng: x + 1, symmetric=True),
    )
    mixture = MixtureMutation({"rejected": child}, n_steps=3)
    state = particle_state.replace_particles(
        np.zeros_like(particle_state.particles)
    )
    _, _, output = mixture(state, context_factory(), 2)
    assert output.moves["rejected"].scaled_squared_displacement == 0


def test_backends(backend, backend_factory, context_factory):
    from sequince.array_api import namespace
    from sequince.steps import DifferentialEvolutionMutation

    case = backend_factory(backend)

    def target(x):
        return -0.5 * namespace(x).sum(x**2, axis=-1)

    mixture = MixtureMutation(
        {
            "walk": MetropolisHastingsMutation(
                target, GaussianRandomWalk(scale=0.5)
            ),
            "de": DifferentialEvolutionMutation(target),
        },
        n_steps=8,
    )
    state, _, out = mixture(case.state(), context_factory(), case.rng(5))
    assert case.matches_namespace(state.particles)
    assert state.particles.dtype == case.particles.dtype
    assert case.matches_namespace(out.coordinate_scale)
    assert out.metrics.n_attempted == 8 * len(case.particles)
    assert all(move.selections for move in out.moves.values())


def test_interleaving_and_metric_isolation(particle_state, context_factory):
    from sequince.steps.mutation import CallableMutationKernel

    order = []
    results = []

    class Metric:
        name = "custom"

        def __init__(self, label):
            self.label = label

        def initialize(self, particles, log_weights, context):
            return 0

        def prepare(self, state, particles, context):
            return state

        def update(self, state, particles, accepted, context):
            return state + 1

        def result(self, state, context):
            results.append(self.label)
            return self.label, state

    def kernel(label):
        def sample(x, rng):
            order.append(label)
            return x

        return CallableMutationKernel(sample, symmetric=True)

    mixture = MixtureMutation(
        {
            name: MetropolisHastingsMutation(
                log_prob, kernel(name), metrics=[Metric(name)]
            )
            for name in ("a", "b")
        },
        n_steps=30,
    )
    _, _, output = mixture(particle_state, context_factory(), 33)
    assert sum(a != b for a, b in zip(order, order[1:])) > 2
    assert results == ["a", "b"]
    assert output.metrics.values == {}
    for name, move in output.moves.items():
        assert move.metrics.values["custom"] == (name, order.count(name))


@pytest.mark.parametrize(
    "invalid", ["empty", "name", "target", "factory", "child", "duplicate"]
)
def test_child_validation(invalid):
    moves = make_mixture().moves
    if invalid == "empty":
        moves = {}
    elif invalid == "name":
        moves[" "] = moves.pop("small")
    elif invalid == "target":
        moves["small"] = MetropolisHastingsMutation(
            lambda x: log_prob(x), GaussianRandomWalk()
        )
    elif invalid == "factory":
        moves["small"] = MetropolisHastingsMutation(
            log_prob, lambda x, context: GaussianRandomWalk()
        )
    elif invalid == "child":
        moves["small"] = object()
    elif invalid == "duplicate":
        moves["small"] = moves["large"]
    with pytest.raises((ValueError, TypeError)):
        MixtureMutation(moves)
