"""Runtime budgets preserve the eager DE kernel and complete-block semantics."""

from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from orng import RandomGenerator

from sequince.state import ParticleState
from sequince.steps import (
    AdaptMutationSteps,
    DifferentialEvolutionMutation,
    ESJDMetric,
    RelativeJumpDistanceMetric,
    RepeatedMutationStep,
    StepContext,
)
from sequince.steps.base import SMCStep
from sequince.steps.mutation.adaptive.policies.acceptance import (
    AcceptanceRatePolicy,
)
from sequince.steps.mutation.metrics.de import DEScaleMetrics
from sequince.target import Target, TargetEvaluation
from sequince.transforms import AffineTransform

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


def gaussian(x: Any, context: StepContext) -> Any:
    return -jnp.sum(x * x, axis=1) / 2


class CachedGaussian(Target):
    def evaluate(self, particles, context, **kwargs):
        return TargetEvaluation({"log_prob": gaussian(particles, context)})

    def log_prob(self, particles, context, *, evaluation=None):
        return evaluation["log_prob"]


def initialize(step: Any, state: ParticleState) -> tuple[Any, Any]:
    context = StepContext(
        t=jnp.asarray(0, dtype=jnp.int32),
        metadata=step.initialize_functional_metadata(state, {}),
    )
    state, context = step.initialize_functional_state_and_context(
        state, context
    )
    # Convert scalar counters too so scan signatures match the first call.
    return jax.tree.map(jnp.asarray, (state, context))


def assert_tree_close(left: Any, right: Any) -> None:
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for a, b in zip(
        jax.tree.leaves(left), jax.tree.leaves(right), strict=True
    ):
        np.testing.assert_allclose(a, b, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("store", [False, True])
@pytest.mark.parametrize("transformed,cached", [(False, False), (True, True)])
def test_dynamic_matches_eager_and_reuses_compilation(
    store, transformed, cached
) -> None:
    step = DifferentialEvolutionMutation(
        CachedGaussian() if cached else gaussian,
        transform=AffineTransform() if transformed else None,
        store_chains=store,
        scale_multipliers=(0.3, 1.0),
        gamma_randomization="bilby",
        metrics=[
            DEScaleMetrics((0.3, 1.0)),
            ESJDMetric(),
            RelativeJumpDistanceMetric(reference_radius=1.0),
        ],
    )
    state, context = initialize(
        step,
        ParticleState(
            jnp.asarray(np.random.default_rng(8).normal(size=(9, 2)))
        ),
    )
    traces = []

    @jax.jit
    def run(state, context, key, budget):
        traces.append(1)
        return step.apply_dynamic_budget(
            state, context, key, n_steps=budget, max_steps=5
        )

    key = jax.random.key(9)
    for budget in (0, 1, 3, 5, 0):
        rng = RandomGenerator.from_xp(jnp, generator=key)
        eager_state, eager_context, eager, eager_metrics = (
            step.apply_with_mutation_state(state, context, rng, n_steps=budget)
        )
        updated, actual_context, actual_key, output, metrics = run(
            state, context, key, jnp.asarray(budget, dtype=jnp.int32)
        )
        assert_tree_close(updated, eager_state)
        assert_tree_close(
            actual_context.evaluation_counts, eager_context.evaluation_counts
        )
        np.testing.assert_array_equal(
            jax.random.key_data(actual_key), jax.random.key_data(rng.state())
        )
        assert_tree_close(
            step.metrics.result(metrics, actual_context),
            step.metrics.result(eager_metrics, eager_context),
        )
        assert int(output["steps"]) == budget
        for name in ("acceptance_by_step", "acceptance_by_half"):
            if budget:
                np.testing.assert_allclose(
                    output[name][:budget], np.asarray(eager[name])
                )
            np.testing.assert_array_equal(output[name][budget:], 0)
        if store:
            assert output["chain"].shape == (5, 9, 2)
            np.testing.assert_allclose(
                output["chain"][:budget], eager["chain"], rtol=2e-5, atol=2e-5
            )
            np.testing.assert_array_equal(output["chain"][budget:], 0)
        else:
            assert output["chain"] is None
        if budget == 0:
            assert_tree_close(actual_context, context)
            np.testing.assert_array_equal(
                jax.random.key_data(actual_key), jax.random.key_data(key)
            )
        # Feed results straight back into the same executable: no warm-up call
        # may change a context/cache leaf's shape, dtype or weak scalar type.
        state, context, key = updated, actual_context, actual_key
    assert len(traces) == 1


class Negate(SMCStep):
    def apply(self, state, context, rng):
        return (
            state.replace_particles(-state.particles),
            replace(
                context,
                next_observation_index=context.next_observation_index + 1,
            ),
            None,
        )

    def apply_functional(self, state, context, key):
        state, context, output = self.apply(state, context, None)
        return state, context, key, output


@pytest.mark.parametrize("budget", [0, 1, 2, 5, 7])
@pytest.mark.parametrize("transformed", [False, True])
def test_repeated_matches_eager_with_zero_rounds(budget, transformed) -> None:
    child = DifferentialEvolutionMutation(
        gaussian,
        transform=AffineTransform() if transformed else None,
        metrics=[
            ESJDMetric(),
            RelativeJumpDistanceMetric(reference_radius=1.0),
        ],
    )
    step = RepeatedMutationStep(child, n_rounds=3, after_round=Negate())
    state, context = initialize(
        step, ParticleState(jax.random.normal(jax.random.key(6), (9, 2)))
    )
    capacity = 2 if budget <= 2 else 7
    key = jax.random.key(8)
    rng = RandomGenerator.from_xp(jnp, generator=key)
    expected, expected_context, expected_output = step.apply_with_n_steps(
        state, context, rng, n_steps=budget
    )
    actual, actual_context, actual_key, output, metrics = jax.jit(
        lambda s, c, k, n: step.apply_dynamic_budget(
            s, c, k, n_steps=n, max_steps=capacity
        )
    )(state, context, key, jnp.asarray(budget, dtype=jnp.int32))
    assert_tree_close(actual, expected)
    assert_tree_close(actual_context, expected_context)
    assert_tree_close(
        step.metrics.result(metrics, actual_context), expected_output.metrics
    )
    np.testing.assert_array_equal(
        jax.random.key_data(actual_key), jax.random.key_data(rng.state())
    )
    assert [int(o["steps"]) for o in output.round_outputs] == [
        budget // 3 + (i < budget % 3) for i in range(3)
    ]
    assert all(o["metrics"] is None for o in output.round_outputs)
    assert int(actual_context.next_observation_index) == 3
    assert [
        o["acceptance_by_step"].shape[0] for o in output.round_outputs
    ] == [capacity // 3 + (i < capacity % 3) for i in range(3)]


@pytest.mark.parametrize("initial_steps", [0, 1])
def test_adaptation_scan_and_repeated_calls(initial_steps: int) -> None:
    child = RepeatedMutationStep(
        DifferentialEvolutionMutation(gaussian), n_rounds=2
    )
    step = AdaptMutationSteps(
        child,
        AcceptanceRatePolicy(3, min_steps=0, max_steps=6),
        initial_steps=initial_steps,
        min_steps=0,
        max_steps=6,
    )
    state, context = initialize(
        step, ParticleState(jax.random.normal(jax.random.key(6), (21, 2)))
    )

    def body(carry, iteration):
        state, context, key = carry
        context = replace(context, t=iteration)
        state, context, key, output = step.apply_functional(
            state, context, key
        )
        first = context.metadata[step.metadata_key]
        state, context, key, repeated = step.apply_functional(
            state, context, key
        )
        return (state, context, key), (
            output.metrics.n_steps,
            repeated.metrics.n_steps,
            first,
            context.metadata[step.metadata_key],
        )

    carry = (state, context, jax.random.key(9))
    actual, (budgets, repeated, first, second) = jax.jit(
        lambda c: jax.lax.scan(body, c, jnp.arange(5, dtype=jnp.int32))
    )(carry)
    np.testing.assert_array_equal(budgets, repeated)
    np.testing.assert_array_equal(budgets[1:], first.next_n_steps[:-1])
    assert_tree_close(first, second)
    assert int(budgets[0]) == initial_steps
    assert np.any(np.asarray(budgets[1:]) != initial_steps)
    expected = carry
    for i in range(5):
        expected, _ = body(expected, jnp.asarray(i, dtype=jnp.int32))
    assert_tree_close(actual[:2], expected[:2])
    np.testing.assert_array_equal(
        jax.random.key_data(actual[2]), jax.random.key_data(expected[2])
    )


@pytest.mark.parametrize("capacity", [0, 8])
def test_skipped_target_and_rng(capacity) -> None:
    calls = []

    def target(x, context):
        jax.debug.callback(lambda _: calls.append(1), x[0, 0], ordered=True)
        return gaussian(x, context)

    step = DifferentialEvolutionMutation(target)
    state, context = initialize(step, ParticleState(jnp.ones((5, 2))))
    key = jax.random.key(2)
    result = jax.jit(
        lambda n: step.apply_dynamic_budget(
            state, context, key, n_steps=n, max_steps=capacity
        )
    )(jnp.asarray(0, dtype=jnp.int32))
    jax.block_until_ready(result)
    jax.effects_barrier()
    assert calls == []
    assert_tree_close(result[0], state)
    np.testing.assert_array_equal(
        jax.random.key_data(result[2]), jax.random.key_data(key)
    )
    assert result[3]["acceptance_by_step"].shape == (capacity,)


def test_gaussian_stationarity() -> None:
    # Start at stationarity: 5000 particles allow mean SE ~0.014 and variance
    # SE ~0.020. Tolerances 0.07/0.10 allow five marginal SE plus weak ensemble
    # dependence. An incorrect target or acceptance rule changes these moments.
    state = ParticleState(jax.random.normal(jax.random.key(10), (5000, 2)))
    step = DifferentialEvolutionMutation(
        gaussian, scale_multipliers=(0.3, 1.0)
    )
    state, context = initialize(step, state)
    result = jax.jit(
        lambda s, c, k: step.apply_dynamic_budget(
            s, c, k, n_steps=jnp.asarray(20, dtype=jnp.int32), max_steps=25
        )
    )(state, context, jax.random.key(11))
    np.testing.assert_allclose(
        jnp.mean(result[0].particles, axis=0), 0, atol=0.07
    )
    np.testing.assert_allclose(
        jnp.cov(result[0].particles.T), np.eye(2), atol=0.10
    )


def test_nested_rounds_and_continuing_state() -> None:
    child = DifferentialEvolutionMutation(gaussian, metrics=[ESJDMetric()])
    step = RepeatedMutationStep(
        RepeatedMutationStep(child, n_rounds=2), n_rounds=2
    )
    state, context = initialize(
        step, ParticleState(jnp.arange(18.0).reshape(9, 2) / 8)
    )
    key = jax.random.key(8)
    rng = RandomGenerator.from_xp(jnp, generator=key)
    expected, expected_context, expected_output = step.apply_with_n_steps(
        state, context, rng, n_steps=3
    )
    result = jax.jit(
        lambda s, c, k: step.apply_dynamic_budget(
            s, c, k, n_steps=jnp.array(3, dtype=jnp.int32), max_steps=5
        )
    )(state, context, key)
    assert_tree_close(result[0], expected)
    assert_tree_close(
        step.metrics.result(result[4], result[1]), expected_output.metrics
    )
    assert_tree_close(result[1], expected_context)
    np.testing.assert_array_equal(
        jax.random.key_data(result[2]), jax.random.key_data(rng.state())
    )


def test_validation_and_initialization_forwarding() -> None:
    from sequince.steps.mutation import (
        GaussianRandomWalk,
        MetropolisHastingsMutation,
    )

    state = ParticleState(jnp.ones((5, 2)))
    unsupported = AdaptMutationSteps(
        MetropolisHastingsMutation(gaussian, GaussianRandomWalk()),
        initial_steps=1,
        min_steps=0,
        max_steps=5,
    )
    with pytest.raises(NotImplementedError, match="DynamicBudgetMutation"):
        unsupported.initialize_functional_state_and_context(
            state, StepContext(t=0)
        )

    class EagerPost(SMCStep):
        pass

    unsupported = AdaptMutationSteps(
        RepeatedMutationStep(
            DifferentialEvolutionMutation(gaussian), after_round=EagerPost()
        ),
        initial_steps=1,
        min_steps=0,
        max_steps=5,
    )
    with pytest.raises(NotImplementedError, match="after_round"):
        initialize(unsupported, state)
    with pytest.raises(ValueError, match="four particles"):
        initialize(
            DifferentialEvolutionMutation(gaussian),
            ParticleState(jnp.ones((3, 2))),
        )
    with pytest.raises(NotImplementedError, match="Callable transform_refit"):
        initialize(
            DifferentialEvolutionMutation(
                gaussian,
                transform=AffineTransform(),
                transform_refit=lambda p, c: True,
            ),
            state,
        )


def test_no_optional_metrics_and_no_padding_work() -> None:
    calls = []

    def target(x, context):
        jax.debug.callback(lambda _: calls.append(1), x[0, 0], ordered=True)
        return gaussian(x, context)

    step = DifferentialEvolutionMutation(
        target, noise_scale=0.0, gamma_one_probability=1.0
    )
    state, context = initialize(
        step, ParticleState(jnp.arange(18.0).reshape(9, 2) / 8)
    )
    key = jax.random.key(8)
    result = jax.jit(
        lambda s, c, k: step.apply_dynamic_budget(
            s, c, k, n_steps=jnp.array(2, dtype=jnp.int32), max_steps=8
        )
    )(state, context, key)
    jax.block_until_ready(result)
    jax.effects_barrier()
    assert (
        len(calls) == 5
    )  # Initial population, then two half-targets per sweep.
    assert int(result[1].evaluation_counts.log_target_evaluations) == 27
    assert result[4].states == ()
    assert int(result[4].n_attempted) == 18
    assert result[3]["chain"] is None
    np.testing.assert_array_equal(result[3]["acceptance_by_step"][2:], 0)


def test_compiled_adapter_pipeline_signature() -> None:
    from sequince import SequentialMonteCarlo
    from sequince.jax import compile_functional_iteration

    step = AdaptMutationSteps(
        DifferentialEvolutionMutation(
            CachedGaussian(), transform=AffineTransform()
        ),
        AcceptanceRatePolicy(3, min_steps=0, max_steps=6),
        initial_steps=1,
        min_steps=0,
        max_steps=6,
    )
    sampler = SequentialMonteCarlo([step])
    state = ParticleState(jax.random.normal(jax.random.key(6), (21, 2)))
    key = jax.random.key(9)
    compiled = compile_functional_iteration(sampler, state, key)
    result, _ = sampler.run_functional(
        state, random_state=key, iteration=compiled, num_steps=4
    )
    assert len(result) == 4
    carry = result.context.metadata[step.metadata_key]
    assert int(carry.current_iteration) == 3
    assert int(carry.last_updated_iteration) == 3
    assert int(carry.current_n_steps) > 1
