from __future__ import annotations

import numpy as np
import pytest

from sequince import SequentialDensity, TemperedDensity, TemperedPosterior
from sequince.state import ParticleState
from sequince.steps import FixedTemperedWeighting, StepContext
from sequince.target import EvaluationCounts


@pytest.mark.parametrize("single", [True, False])
@pytest.mark.parametrize("beta", [0.0, 0.4, 1.0])
def test_tempered_density_endpoints_and_probability(single, beta):
    target = TemperedDensity(
        lambda x: -np.sum(x**2, axis=-1), lambda x, ctx: 2.0
    )
    particles = (
        np.array([1.0, 2.0]) if single else np.array([[1.0, 2.0], [2.0, 3.0]])
    )
    context = StepContext(t=0, metadata={"beta": beta})
    initial = -np.sum(particles**2, axis=-1)
    np.testing.assert_allclose(target.log_initial(particles, context), initial)
    np.testing.assert_allclose(
        target.log_terminal(particles, context), initial + 2.0
    )
    np.testing.assert_allclose(
        target.log_prob(particles, context), initial + beta * 2.0
    )
    np.testing.assert_allclose(target.log_increment(particles, context), 2.0)


def test_sequential_density_observation_cache_key():
    calls = []

    def log_prob(x, context):
        calls.append(context.observation)
        return -np.sum(x**2, axis=-1) + context.observation

    target = SequentialDensity(
        log_prob,
        lambda x, ctx: ctx.observation,
        evaluation_key=lambda ctx: ctx.observation,
    )
    particles = np.zeros((3, 1))
    context = StepContext(t=0, observation=1)
    evaluation = target.prepare_evaluation(particles, context)
    assert (
        target.prepare_evaluation(particles, context, evaluation=evaluation)
        is evaluation
    )
    next_context = StepContext(t=1, observation=2)
    fresh = target.prepare_evaluation(
        particles, next_context, evaluation=evaluation
    )
    np.testing.assert_allclose(
        target.log_prob(particles, next_context, evaluation=fresh), 2.0
    )
    assert calls == [1, 2]
    with pytest.raises(ValueError, match="different target or stage"):
        target.log_prob(particles, next_context, evaluation=evaluation)


def test_tempered_increment_reuses_density_cache_and_refreshes(rng):
    calls = []

    def increment(x, ctx):
        calls.append(ctx.t)
        return float(ctx.t + 1)

    target = TemperedDensity(lambda x: 0.0, increment)
    particles = np.zeros((4, 1))
    context = StepContext(t=0)
    state = ParticleState(
        particles, evaluation=target.evaluate(particles, context)
    )
    _, context, _ = FixedTemperedWeighting(target, beta_increment=0.25).apply(
        state, context, rng
    )
    assert calls == [0]
    assert context.evaluation_counts == EvaluationCounts(
        log_increment_evaluations=4
    )
    _, _, output = FixedTemperedWeighting(
        target, beta_increment=0.25, refresh_components={"log_increment"}
    ).apply(state, StepContext(t=1), rng)
    assert calls == [0, 1]
    np.testing.assert_allclose(output["log_weight_increment"], 0.5)


def test_posterior_direct_increment_leaves_log_prob_and_frozen_density_unchanged():
    calls = []

    def increment(x, context):
        calls.append(None)
        return 42.0

    target = TemperedPosterior(
        lambda x: -1.0,
        lambda x: -2.0,
        lambda x: -4.0,
        log_density_ratio=increment,
    )
    particles = np.zeros((3, 1))
    context = StepContext(t=0, metadata={"beta": 0.5})
    np.testing.assert_allclose(target.log_increment(particles, context), 42.0)
    np.testing.assert_allclose(target.log_prob(particles, context), -3.5)
    np.testing.assert_allclose(
        target.freeze_log_prob(context)(particles), -3.5
    )
    assert len(calls) == 1


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax", "cupy"])
def test_density_backends(backend, backend_factory):
    case = backend_factory(backend)
    xp = case.xp
    target = TemperedDensity(lambda x: -xp.sum(x**2, axis=-1), lambda x: 2.0)
    context = StepContext(t=0, metadata={"beta": 0.5})
    result = target.log_prob(case.particles, context)
    assert case.matches_namespace(result)
    if backend == "jax":
        import jax

        gradient = jax.jit(jax.grad(lambda x: target.log_prob(x, context)))(
            case.particles[0]
        )
        np.testing.assert_allclose(
            np.asarray(gradient), -2.0 * np.asarray(case.particles[0])
        )


@pytest.mark.parametrize("target_kind", ["posterior", "tempered"])
def test_compiled_density_pipeline_with_mcmc(target_kind):
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    pytest.importorskip("blackjax")
    from sequince import SequentialMonteCarlo
    from sequince.jax import compile_functional_iteration
    from sequince.steps import BlackJAXMutation

    def initial(x):
        return -0.5 * jnp.sum(x**2, axis=-1)

    def increment(x):
        return -0.5 * jnp.sum((x - 1.0) ** 2, axis=-1)

    target = (
        TemperedDensity(initial, increment)
        if target_kind == "tempered"
        else TemperedPosterior(initial, increment, log_density_ratio=increment)
    )
    smc = SequentialMonteCarlo(
        steps=[
            FixedTemperedWeighting(target, beta_increment=0.5),
            BlackJAXMutation(
                target,
                n_steps=1,
                algorithm="random-walk",
                algorithm_kwargs={"sigma": 0.1},
            ),
        ]
    )
    state = ParticleState(jnp.zeros((4, 1)))
    key = jax.random.key(42)
    iteration = compile_functional_iteration(smc, state, key)
    result, _ = smc.run_functional(
        state, random_state=key, iteration=iteration, num_steps=2
    )
    assert result.final_state.evaluation is not None
    assert np.isfinite(float(result.final_state.log_evidence))
    assert float(result.context.metadata["beta"]) == 1.0


def test_density_constructor_keywords():
    particles = np.zeros((3, 1))
    context = StepContext(t=0)
    sequential = SequentialDensity(
        log_prob=lambda x: -1.0, log_increment=lambda x: 2.0
    )
    tempered = TemperedDensity(
        log_initial=lambda x: -1.0, log_density_ratio=lambda x: 2.0
    )
    posterior = TemperedPosterior(
        log_prior=lambda x: -1.0,
        log_likelihood=lambda x: 0.0,
        log_density_ratio=lambda x: 2.0,
    )
    for target in (sequential, tempered, posterior):
        np.testing.assert_allclose(
            target.log_increment(particles, context), 2.0
        )


def test_custom_step_can_request_increment_directly():
    from unittest.mock import Mock

    density = Mock(side_effect=AssertionError("Unused density"))
    increment = Mock(return_value=np.array([1.0, 2.0]))
    particles = np.zeros((2, 1))
    context = StepContext(t=0)
    for target in (
        SequentialDensity(density, increment),
        TemperedDensity(density, increment),
        TemperedPosterior(density, density, log_density_ratio=increment),
    ):
        np.testing.assert_allclose(
            target.log_increment(particles, context), [1.0, 2.0]
        )
    assert increment.call_count == 3
    density.assert_not_called()


@pytest.mark.parametrize("kind", ["posterior", "tempered", "sequential"])
def test_increment_cache_promotes_to_mcmc_density(kind):
    calls = {"density": 0, "increment": 0}

    def density(x):
        calls["density"] += 1
        return -np.sum(x**2, axis=-1)

    def increment(x):
        calls["increment"] += 1
        return np.ones(x.shape[0])

    if kind == "posterior":
        target = TemperedPosterior(
            lambda x: 0.0, density, log_density_ratio=increment
        )
    elif kind == "tempered":
        target = TemperedDensity(density, increment)
    else:
        target = SequentialDensity(density, increment)
    particles = np.zeros((3, 1))
    context = StepContext(t=0)
    partial = target.prepare_evaluation(
        particles, context, components={"log_increment"}
    )
    assert set(partial.components) == {"log_increment"}
    assert calls == {"density": 0, "increment": 1}
    full, counts = target.prepare_evaluation(
        particles, context, evaluation=partial, return_counts=True
    )
    assert calls == {"density": 1, "increment": 1}
    assert set(full.components) == set(target.evaluation_components(context))
    assert counts.log_target_evaluations == 3
    target.log_prob(particles, context, evaluation=full)
    assert calls == {"density": 1, "increment": 1}


def test_posterior_refresh_recomputes_derived_increment(rng):
    calls = []

    def likelihood(x, context):
        calls.append(context.t)
        return np.full(x.shape[0], context.t + 1.0)

    target = TemperedPosterior(lambda x: 0.0, likelihood)
    particles = np.zeros((3, 1))
    context = StepContext(t=0)
    cached = target.prepare_evaluation(
        particles, context, components={"log_increment"}
    )
    state = ParticleState(particles, evaluation=cached)
    updated, _, output = FixedTemperedWeighting(
        target, beta_increment=0.25, refresh_components={"log_likelihood"}
    ).apply(state, StepContext(t=1), rng)
    np.testing.assert_allclose(output["log_weight_increment"], 0.5)
    np.testing.assert_allclose(updated.evaluation["log_increment"], 2.0)
    assert calls == [0, 1]
