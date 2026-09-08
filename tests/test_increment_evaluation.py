from __future__ import annotations

import numpy as np
import pytest

from sequince import SequentialDensity, TemperedDensity, TemperedPosterior
from sequince.state import ParticleState
from sequince.steps import (
    AdaptiveTemperedWeighting,
    FixedTemperedWeighting,
    IncrementalWeighting,
    StepContext,
)
from sequince.target import EvaluationCounts, TargetEvaluation


def make_step(kind, target, **kwargs):
    if kind == "incremental":
        return IncrementalWeighting(target, **kwargs)
    if kind == "fixed":
        return FixedTemperedWeighting(target, beta_increment=0.25, **kwargs)
    return AdaptiveTemperedWeighting(target, target_ess_ratio=0.8, **kwargs)


@pytest.mark.parametrize("kind", ["incremental", "fixed", "adaptive"])
@pytest.mark.parametrize(
    "target_kind", ["posterior", "tempered", "sequential"]
)
def test_direct_callbacks_skip_density_preparation(kind, target_kind, rng):
    if target_kind == "sequential" and kind != "incremental":
        return  # A sequential density has no geometric tempering contract.
    calls = []

    def density(x):
        raise AssertionError("Weighting must not evaluate the full density")

    def increment(x, context):
        calls.append(context.t)
        return -np.sum(x**2, axis=-1)

    if target_kind == "posterior":
        target = TemperedPosterior(
            density, density, log_density_ratio=increment
        )
    elif target_kind == "tempered":
        target = TemperedDensity(density, increment)
    else:
        target = SequentialDensity(density, increment)
    particles = np.arange(4.0).reshape(4, 1)
    state = ParticleState(
        particles,
        evaluation=TargetEvaluation({"old": np.zeros(4)}, target_id="other"),
    )
    step = make_step(kind, target)
    state, context = step.initialize_functional_state_and_context(
        state, StepContext(t=0)
    )
    assert calls == [0]
    assert set(state.evaluation.components) == {"log_increment"}
    updated, context, _ = step.apply(state, context, rng)
    expected, _, _ = make_step(kind, increment).apply(
        ParticleState(particles), StepContext(t=0), rng
    )
    np.testing.assert_allclose(updated.log_weights, expected.log_weights)
    assert calls == [0, 0]
    assert "log_increment" in updated.evaluation
    assert context.evaluation_counts == EvaluationCounts(
        log_increment_evaluations=4
    )


@pytest.mark.parametrize("kind", ["incremental", "fixed", "adaptive"])
def test_default_posterior_keeps_cached_evaluations(kind, rng):
    calls = []

    def likelihood(x):
        calls.append(None)
        return -np.sum(x**2, axis=-1)

    target = TemperedPosterior(lambda x: 0.0, likelihood)
    step = make_step(kind, target)
    state, context = step.initialize_functional_state_and_context(
        ParticleState(np.zeros((4, 1))), StepContext(t=0)
    )
    updated, context, _ = step.apply(state, context, rng)
    assert len(calls) == 1
    assert updated.evaluation is not None
    assert context.evaluation_counts == EvaluationCounts(4, 4, 0, 4)


@pytest.mark.parametrize("target_kind", ["posterior", "tempered"])
@pytest.mark.parametrize("kind", ["fixed", "adaptive"])
def test_direct_callbacks_compile_without_cache(kind, target_kind):
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    def density(x):
        raise AssertionError("Unnecessary density evaluation")

    def increment(x):
        return -jnp.sum(x**2, axis=-1)

    target = (
        TemperedDensity(density, increment)
        if target_kind == "tempered"
        else TemperedPosterior(density, density, log_density_ratio=increment)
    )
    step = make_step(kind, target)
    state = ParticleState(jnp.arange(4.0).reshape(4, 1))
    context = StepContext(t=0)
    state, context = step.initialize_functional_state_and_context(
        state, context
    )
    context = context.with_metadata(
        **step.initialize_functional_metadata(state, context.metadata)
    )
    transition = jax.jit(step.apply_functional)
    updated, next_context, key, _ = transition(
        state, context, jax.random.key(1)
    )
    assert "log_increment" in updated.evaluation
    assert int(next_context.evaluation_counts.log_target_evaluations) == 0
    again, final_context, _, _ = transition(updated, next_context, key)
    assert "log_increment" in again.evaluation
    assert int(final_context.evaluation_counts.log_increment_evaluations) == 8


@pytest.mark.parametrize(
    "target_kind", ["posterior", "tempered", "sequential"]
)
def test_mcmc_evaluates_full_density_after_direct_weighting(target_kind, rng):
    from sequince.steps.mutation import (
        CallableMutationKernel,
        MetropolisHastingsMutation,
    )

    calls = []

    def density(x):
        calls.append(None)
        return -np.sum(x**2, axis=-1)

    if target_kind == "posterior":
        target = TemperedPosterior(
            lambda x: 0.0, density, log_density_ratio=lambda x: 0.0
        )
    elif target_kind == "tempered":
        target = TemperedDensity(density, lambda x: 0.0)
    else:
        target = SequentialDensity(density, lambda x: 0.0)
    state, context, _ = IncrementalWeighting(target).apply(
        ParticleState(np.zeros((4, 1))), StepContext(t=0), rng
    )
    assert not calls
    mutation = MetropolisHastingsMutation(
        target,
        CallableMutationKernel(lambda x, rng: x + 0.1, symmetric=True),
        n_steps=1,
    )
    updated, context, _ = mutation.apply(state, context, rng)
    assert len(calls) == 2
    assert updated.evaluation is not None
    assert context.evaluation_counts.log_target_evaluations == 8
    assert context.evaluation_counts.log_increment_evaluations == 4
