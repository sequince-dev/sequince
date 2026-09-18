"""Endpoint RJD and lagged step-count adaptation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from orng import RandomGenerator

from sequince import ParticleState
from sequince.array_api import asarray, get_device, namespace, to_numpy
from sequince.steps import (
    AdaptMutationSteps,
    RelativeJumpDistanceMetric,
    RelativeJumpDistancePolicy,
    RepeatedMutationStep,
)
from sequince.steps.context import StepContext
from sequince.steps.mutation import (
    CallableMutationKernel,
    GaussianRandomWalk,
    MetropolisHastingsMutation,
    MutationMetricsOutput,
)


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
def test_endpoint_distance_and_rejections(
    backend: str, backend_factory: Any
) -> None:
    case = backend_factory(backend)
    metric = RelativeJumpDistanceMetric(reference_radius=2)
    context = StepContext(t=0)
    initial = metric.initialize(case.particles, case.log_weights, context)
    jump = asarray([[6, 8], [0, 0], [0, 2], [0, 4]], like=case.particles)
    accepted = asarray(
        [True, False, True, True], like=case.particles, dtype=case.xp.bool
    )
    first = metric.update(initial, case.particles + jump, accepted, context)
    result = metric.result(first, context)
    assert float(to_numpy(result["fraction_above_one"])) == 0.5
    assert float(to_numpy(result["geometric_mean"])) == 0
    for value in result.values():
        assert namespace(value) is case.xp
        assert get_device(value) == get_device(case.particles)
        assert value.dtype == case.particles.dtype
    # Returning to the start must erase RJD, despite a long path travelled.
    final = metric.update(first, case.particles, accepted, context)
    assert (
        float(to_numpy(metric.result(final, context)["fraction_above_one"]))
        == 0
    )
    np.testing.assert_array_equal(
        to_numpy(initial.end), to_numpy(case.particles)
    )


@pytest.mark.parametrize("radius", [0, -1, np.nan, np.inf, True])
def test_invalid_radius(radius: float) -> None:
    with pytest.raises(ValueError, match="reference_radius"):
        RelativeJumpDistanceMetric(reference_radius=radius)


def test_geometric_mean_and_fixed_radius() -> None:
    context = StepContext(t=0)
    metric = RelativeJumpDistanceMetric(reference_radius=2)
    start = np.zeros((2, 1))
    state = metric.initialize(start, np.zeros(2), context)
    state = metric.update(
        state, np.array([[1.0], [8.0]]), np.ones(2, bool), context
    )
    assert metric.result(state, context)["geometric_mean"] == pytest.approx(
        np.sqrt(2)
    )


@pytest.mark.parametrize("fraction,expected", [(0, 11), (0.5, 10), (1, 9)])
def test_budget_direction(fraction: float, expected: int) -> None:
    policy = RelativeJumpDistancePolicy()
    metrics = MutationMetricsOutput(
        10, 40, 20, {"rjd": {"fraction_above_one": fraction}}
    )
    assert (
        policy.update(
            current_steps=10, metrics=metrics, context=StepContext(t=0)
        )
        == expected
    )


@pytest.mark.parametrize("fraction", [np.nan, np.inf, -0.1, 1.1, [0.2, 0.3]])
def test_invalid_fraction(fraction: Any) -> None:
    with pytest.raises(ValueError, match="fraction_above_one"):
        RelativeJumpDistancePolicy().update(
            current_steps=3,
            metrics=MutationMetricsOutput(
                3, 6, 0, {"rjd": {"fraction_above_one": fraction}}
            ),
            context=StepContext(t=0),
        )


def test_no_measurement_and_missing_metric() -> None:
    policy = RelativeJumpDistancePolicy()
    context = StepContext(t=0)
    assert (
        policy.update(
            current_steps=0,
            metrics=MutationMetricsOutput(0, 0, 0),
            context=context,
        )
        == 1
    )
    with pytest.raises(ValueError, match="requires"):
        policy.update(
            current_steps=1,
            metrics=MutationMetricsOutput(1, 2, 0),
            context=context,
        )


def _flat(particles: Any, context: StepContext) -> Any:
    return namespace(particles).zeros_like(particles[:, 0])


def _adaptive() -> AdaptMutationSteps:
    return AdaptMutationSteps(
        MetropolisHastingsMutation(
            _flat,
            CallableMutationKernel(lambda x, rng: x + 0.2, symmetric=True),
            metrics=[RelativeJumpDistanceMetric(reference_radius=1)],
        ),
        RelativeJumpDistancePolicy(factor=2),
        initial_steps=2,
        min_steps=1,
        max_steps=6,
    )


def test_lagged_frozen_budget_bounds_and_checkpoint() -> None:
    adaptive = _adaptive()
    initial = ParticleState(np.zeros((4, 1)))
    lengths = []
    for t in (0, 0, 1, 2):
        _, _, output = adaptive(
            initial, StepContext(t=t), RandomGenerator(seed=1)
        )
        lengths.append(output["steps"])
    assert lengths == [2, 2, 4, 6]
    assert adaptive.next_n_steps == 3
    restored = _adaptive()
    restored.load_state_dict(adaptive.state_dict())
    for step in (adaptive, restored):
        _, _, output = step(initial, StepContext(t=3), RandomGenerator(seed=2))
        assert output["steps"] == 3
        assert step.next_n_steps == 6


def test_repeated_segments_preserve_start_and_zero_steps() -> None:
    mutation = MetropolisHastingsMutation(
        _flat,
        CallableMutationKernel(lambda x, rng: x + 1, symmetric=True),
        metrics=[RelativeJumpDistanceMetric(reference_radius=1)],
    )
    initial = ParticleState(np.zeros((3, 1)))
    step = RepeatedMutationStep(mutation, n_rounds=4)
    for n_steps in (0, 2, 5):
        _, _, output = step.apply_with_n_steps(
            initial, StepContext(t=0), RandomGenerator(seed=1), n_steps=n_steps
        )
        assert output.metrics.values["rjd"]["geometric_mean"] == pytest.approx(
            n_steps
        )


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
@pytest.mark.parametrize("radius", [1.0, "pair_median"])
def test_metric_preserves_gaussian_sampling_and_rng(
    backend: str, backend_factory: Any, radius: Any
) -> None:
    case = backend_factory(backend)
    results = []
    for metrics in ([], [RelativeJumpDistanceMetric(reference_radius=radius)]):
        calls = []

        def target(x: Any, context: StepContext) -> Any:
            calls.append(1)
            return -0.5 * case.xp.sum(x**2, axis=1)

        mutation = MetropolisHastingsMutation(
            target, GaussianRandomWalk(scale=0.5), n_steps=4, metrics=metrics
        )
        rng = case.rng(23)
        final, _, output = mutation(case.state(), StepContext(t=0), rng)
        results.append((final.particles, rng.random((5,)), len(calls)))
        if metrics:
            assert output["metrics"].values["rjd"]["reference_radius"] > 0
    for left, right in zip(results[0], results[1], strict=True):
        np.testing.assert_array_equal(to_numpy(left), to_numpy(right))


def test_fixed_radius_metric_in_jax_loop() -> None:
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    from sequince.steps.mutation.metrics import MutationMetrics

    metrics = MutationMetrics([RelativeJumpDistanceMetric(reference_radius=2)])
    context = StepContext(t=0)

    @jax.jit
    def run(particles: Any) -> Any:
        initial = metrics.initialize(particles, jnp.zeros(3), context)
        final = jax.lax.fori_loop(
            0,
            4,
            lambda i, state: metrics.update(
                state, particles + i + 1, jnp.ones(3, bool), context
            ),
            initial,
        )
        return metrics.result(final, context)

    result = run(jnp.zeros((3, 2))).values["rjd"]
    np.testing.assert_allclose(result["geometric_mean"], np.sqrt(8), rtol=1e-6)
    assert float(result["fraction_above_one"]) == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_fraction": 0},
        {"target_fraction": 1},
        {"target_fraction": np.nan},
        {"target_fraction": True},
        {"factor": 1},
        {"factor": np.inf},
        {"factor": np.nan},
        {"factor": True},
        {"metric_name": ""},
    ],
)
def test_invalid_policy_configuration(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        RelativeJumpDistancePolicy(**kwargs)


def test_rejected_mutation_proposes_larger_next_budget() -> None:
    mutation = MetropolisHastingsMutation(
        lambda x, c: np.where(x[:, 0] == 0, 0, -np.inf),
        CallableMutationKernel(lambda x, rng: x + 1, symmetric=True),
        metrics=[RelativeJumpDistanceMetric()],
    )
    step = AdaptMutationSteps(
        mutation,
        RelativeJumpDistancePolicy(factor=2),
        initial_steps=3,
        min_steps=1,
        max_steps=5,
    )
    initial = ParticleState(np.zeros((4, 1)))
    final, _, output = step(initial, StepContext(t=0), RandomGenerator(seed=1))
    np.testing.assert_array_equal(final.particles, initial.particles)
    assert output["steps"] == 3
    assert output["metrics"].values["rjd"]["fraction_above_one"] == 0
    assert step.next_n_steps == 5


def test_adaptive_gaussian_target() -> None:
    # Start at stationarity; adapt across blocks and check known moments.
    particles = np.random.default_rng(14).normal(size=(4096, 2))
    step = AdaptMutationSteps(
        MetropolisHastingsMutation(
            lambda x, c: -0.5 * np.sum(x**2, axis=1),
            GaussianRandomWalk(scale=0.7),
            metrics=[RelativeJumpDistanceMetric()],
        ),
        RelativeJumpDistancePolicy(),
        initial_steps=8,
        min_steps=1,
        max_steps=20,
    )
    state = ParticleState(particles)
    rng = RandomGenerator(seed=18)
    for t in range(10):
        state, _, _ = step(state, StepContext(t=t), rng)
    # About four standard errors for the mean and variance with 4096 chains.
    np.testing.assert_allclose(np.mean(state.particles, axis=0), 0, atol=0.07)
    np.testing.assert_allclose(np.var(state.particles, axis=0), 1, atol=0.1)


def test_nonfinite_distances_are_not_reported_as_good_movement() -> None:
    context = StepContext(t=0)
    metric = RelativeJumpDistanceMetric()
    state = metric.initialize(np.zeros((2, 1)), np.zeros(2), context)
    state = metric.update(
        state, np.array([[np.nan], [2.0]]), np.ones(2, bool), context
    )
    result = metric.result(state, context)
    assert np.isnan(result["fraction_above_one"])
    assert np.isnan(result["geometric_mean"])


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
def test_red_blue_endpoint_distance(
    backend: str, backend_factory: Any
) -> None:
    from sequince.steps.mutation import (
        DifferentialEvolutionKernel,
        RedBlueMetropolisHastingsMutation,
    )

    case = backend_factory(backend)
    metric = RelativeJumpDistanceMetric()
    mutation = RedBlueMetropolisHastingsMutation(
        lambda x, c: -0.5 * case.xp.sum(x**2, axis=1),
        DifferentialEvolutionKernel(gamma=0.5),
        n_steps=4,
        metrics=[metric],
    )
    final, _, output = mutation(case.state(), StepContext(t=0), case.rng(18))
    distances = np.linalg.norm(
        to_numpy(final.particles - case.particles), axis=1
    )
    result = output["metrics"].values["rjd"]
    assert float(to_numpy(result["fraction_above_one"])) == np.mean(
        distances > 1
    )
    expected = (
        0.0 if np.any(distances == 0) else np.exp(np.mean(np.log(distances)))
    )
    assert float(to_numpy(result["geometric_mean"])) == pytest.approx(
        expected, rel=1e-6
    )


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_pair_median_backend_and_scaling(
    backend: str, dtype: str, backend_factory: Any
) -> None:
    case = backend_factory(backend)
    if backend == "jax" and dtype == "float64":
        import jax

        if not jax.config.x64_enabled:
            pytest.skip("JAX x64 is disabled")
    particles = case.xp.astype(case.particles, getattr(case.xp, dtype))
    metric = RelativeJumpDistanceMetric(reference_radius="pair_median", seed=4)
    context = StepContext(t=0)
    initial = metric.initialize(particles, case.log_weights, context)
    repeated = metric.initialize(particles, case.log_weights, context)
    scaled = metric.initialize(3 * particles + 5, case.log_weights, context)
    np.testing.assert_allclose(
        to_numpy(repeated.radius), to_numpy(initial.radius)
    )
    np.testing.assert_allclose(
        to_numpy(scaled.radius), 3 * to_numpy(initial.radius), rtol=1e-6
    )
    assert namespace(initial.radius) is case.xp
    assert initial.radius.dtype == particles.dtype
    assert get_device(initial.radius) == get_device(particles)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"reference_radius": "invalid"},
        {"reference_radius": "bootstrap"},
        {"n_bootstrap": 30},
        {"seed": -1},
    ],
)
def test_invalid_radius_configuration(kwargs: dict[str, Any]) -> None:
    with pytest.raises((ValueError, TypeError)):
        RelativeJumpDistanceMetric(**kwargs)


@pytest.mark.parametrize("n_pairs", [4095, 4096])
def test_pair_median_matches_independent_empirical_reference(
    n_pairs: int,
) -> None:
    particles = np.random.default_rng(12).normal(size=(100, 3))
    metric = RelativeJumpDistanceMetric(
        reference_radius="pair_median", n_pairs=n_pairs, seed=11
    )
    context = StepContext(t=0)
    state = metric.initialize(particles, np.zeros(len(particles)), context)
    indices = RandomGenerator(seed=11).choice(
        len(particles), size=(2, n_pairs), replace=True
    )
    expected = np.median(
        np.linalg.norm(particles[indices[0]] - particles[indices[1]], axis=1)
    )
    assert float(state.radius) == pytest.approx(expected)
    # Approximate median calibrates near half of all independent empirical pairs.
    all_pairs = np.linalg.norm(particles[:, None] - particles[None, :], axis=2)
    assert np.mean(all_pairs > state.radius) == pytest.approx(0.5, abs=0.03)
    continued = metric.prepare(state, particles * 10, context)
    assert continued.radius is state.radius
    assert continued.start is state.start


@pytest.mark.parametrize(
    "particles",
    [
        np.ones((1, 2)),
        np.ones((4, 2)),
        np.full((4, 2), np.nan),
        np.array([[0.0], [0.0], [0.0], [1.0]]),
    ],
)
def test_pair_median_invalid_population(particles: np.ndarray) -> None:
    with pytest.raises(ValueError, match="Pair-median"):
        RelativeJumpDistanceMetric(reference_radius="pair_median").initialize(
            particles, np.zeros(len(particles)), StepContext(t=0)
        )


@pytest.mark.parametrize("n_pairs", [0, -1, 1.5, True])
def test_pair_median_invalid_count(n_pairs: Any) -> None:
    with pytest.raises((ValueError, TypeError)):
        RelativeJumpDistanceMetric(
            reference_radius="pair_median", n_pairs=n_pairs
        )


@pytest.mark.parametrize(
    "fraction,expected",
    [
        (0.39, 111),
        (0.4, 100),
        (0.425, 100),
        (0.45, 100),
        (0.46, 90),
    ],
)
def test_budget_hold_band(fraction: float, expected: int) -> None:
    policy = RelativeJumpDistancePolicy(
        target_fraction=0.4, upper_target_fraction=0.45, factor=1.1
    )
    actual = policy.update(
        current_steps=100,
        metrics=MutationMetricsOutput(
            100, 1000, 500, {"rjd": {"fraction_above_one": fraction}}
        ),
        context=StepContext(t=0),
    )
    assert actual == expected


@pytest.mark.parametrize("upper", [0.39, 1, np.inf, np.nan, True])
def test_invalid_budget_hold_band(upper: float) -> None:
    with pytest.raises(ValueError, match="upper_target_fraction"):
        RelativeJumpDistancePolicy(
            target_fraction=0.4, upper_target_fraction=upper
        )
