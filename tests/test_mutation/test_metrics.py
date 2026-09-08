from __future__ import annotations

import numpy as np
import pytest

from sequince import ParticleState
from sequince.steps.base import SMCStep
from sequince.steps.control import SequentialStep
from sequince.steps.mutation import (
    CallableMutationKernel,
    MetropolisHastingsMutation,
    MutationStepState,
    RepeatedMutationStep,
    RepeatedMutationStepOutput,
)


def _flat_log_prob(particles, context):
    del context
    return np.zeros(particles.shape[0])


def _add(value):
    return CallableMutationKernel(
        lambda particles, rng: particles + value,
        symmetric=True,
    )


class NegateStep(SMCStep):
    def apply(self, state, context, rng):
        del rng
        return state.replace_particles(-state.particles), context, None


def test_deterministic_mutation_has_expected_counts(
    context_factory, rng
) -> None:
    state = ParticleState(np.array([[-1.0, 0.0], [1.0, 0.0]]))
    mutation = MetropolisHastingsMutation(
        _flat_log_prob,
        _add(np.array([1.0, 2.0])),
        n_steps=2,
        metrics=[CountMetric()],
    )

    updated, _, output = mutation(state, context_factory(), rng)
    metrics = output["metrics"]

    assert metrics.n_steps == 2
    assert metrics.n_attempted == 4
    assert metrics.n_accepted == 4
    assert metrics.acceptance == pytest.approx(1.0)
    # Initial variances are (1, 0); the zero-variance coordinate uses scale 1.
    # Each particle moves (1, 2), hence (1 + 4) / two dimensions = 2.5.
    np.testing.assert_allclose(updated.particles, state.particles + [2.0, 4.0])


def test_rejected_proposals_count_as_attempts(context_factory, rng) -> None:
    state = ParticleState(np.zeros((3, 1)))

    def target(particles, context):
        del context
        return np.where(particles[:, 0] == 0.0, 0.0, -np.inf)

    mutation = MetropolisHastingsMutation(
        target, _add(1.0), n_steps=2, metrics=[CountMetric()]
    )
    updated, _, output = mutation(state, context_factory(), rng)
    metrics = output["metrics"]

    assert metrics.n_steps == 2
    assert metrics.n_attempted == 6
    assert metrics.n_accepted == 0
    assert metrics.acceptance == 0.0
    np.testing.assert_array_equal(updated.particles, state.particles)


def test_partial_acceptance_is_counted_per_particle(
    context_factory, rng
) -> None:
    state = ParticleState(np.array([[-1.0], [1.0]]))

    def target(particles, context):
        del context
        return np.where(particles[:, 0] <= 1.0, 0.0, -np.inf)

    _, _, output = MetropolisHastingsMutation(
        target, _add(1.0), metrics=[CountMetric()]
    )(state, context_factory(), rng)
    metrics = output["metrics"]

    assert metrics.n_attempted == 2
    assert metrics.n_accepted == 1
    assert metrics.acceptance == pytest.approx(0.5)


def test_zero_step_mutation_reports_zero_diagnostics(
    particle_state, context_factory, rng
) -> None:
    mutation = MetropolisHastingsMutation(
        _flat_log_prob, _add(1.0), metrics=[CountMetric()]
    )

    updated, _, output = mutation.apply_with_n_steps(
        particle_state, context_factory(), rng, n_steps=0
    )
    metrics = output["metrics"]

    assert metrics.n_steps == 0
    assert metrics.n_attempted == 0
    assert metrics.n_accepted == 0
    assert metrics.acceptance == 0.0
    np.testing.assert_array_equal(updated.particles, particle_state.particles)


def test_repeated_rounds_share_metric_state(context_factory, rng) -> None:
    state = ParticleState(np.array([[-1.0], [1.0]]))
    mutation = MetropolisHastingsMutation(
        _flat_log_prob, _add(1.0), metrics=[CountMetric()]
    )
    repeated = RepeatedMutationStep(
        mutation, after_round=SequentialStep([NegateStep()]), n_rounds=2
    )

    updated, context, outputs = repeated.apply_with_n_steps(
        state, context_factory(), rng, n_steps=2
    )
    metrics = outputs.metrics

    assert isinstance(outputs, RepeatedMutationStepOutput)
    assert all(
        round_output["metrics"] is None
        for round_output in outputs.round_outputs
    )
    assert metrics.n_steps == 2
    assert metrics.n_attempted == 4
    assert metrics.n_accepted == 4
    assert "_mutation_diagnostics_accumulator" not in context.metadata
    # Both mutation and symmetry rounds still execute exactly as before.
    np.testing.assert_array_equal(updated.particles, state.particles)


def test_repeated_mutation_returns_one_complete_block_diagnostic(
    context_factory, rng
) -> None:
    state = ParticleState(np.array([[-1.0], [1.0]]))
    repeated = RepeatedMutationStep(
        MetropolisHastingsMutation(
            _flat_log_prob, _add(1.0), metrics=[CountMetric()]
        ),
        n_rounds=3,
    )

    updated, context, outputs = repeated.apply_with_n_steps(
        state, context_factory(), rng, n_steps=5
    )

    assert [output["steps"] for output in outputs.round_outputs] == [2, 2, 1]
    assert all(output["metrics"] is None for output in outputs.round_outputs)
    assert outputs.metrics.n_steps == 5
    assert outputs.metrics.n_attempted == 10
    assert outputs.metrics.n_accepted == 10
    assert "_mutation_diagnostics_accumulator" not in context.metadata
    np.testing.assert_array_equal(updated.particles, state.particles + 5.0)


def test_repeated_mutation_budget_smaller_than_round_count(
    context_factory, rng
) -> None:
    state = ParticleState(np.array([[-1.0], [1.0]]))
    repeated = RepeatedMutationStep(
        MetropolisHastingsMutation(
            _flat_log_prob, _add(1.0), metrics=[CountMetric()]
        ),
        n_rounds=4,
    )

    updated, _, outputs = repeated.apply_with_n_steps(
        state, context_factory(), rng, n_steps=2
    )

    assert [output["steps"] for output in outputs.round_outputs] == [
        1,
        1,
        0,
        0,
    ]
    assert all(output["metrics"] is None for output in outputs.round_outputs)
    assert outputs.metrics.n_steps == 2
    assert outputs.metrics.n_attempted == 4
    assert outputs.metrics.n_accepted == 4
    np.testing.assert_array_equal(updated.particles, state.particles + 2.0)


def test_nested_repeats_only_finalise_the_outer_block(
    context_factory, rng
) -> None:
    state = ParticleState(np.array([[-1.0], [1.0]]))
    inner = RepeatedMutationStep(
        MetropolisHastingsMutation(
            _flat_log_prob, _add(1.0), metrics=[CountMetric()]
        ),
        n_rounds=2,
    )
    outer = RepeatedMutationStep(inner, n_rounds=2)

    updated, context, outputs = outer.apply_with_n_steps(
        state, context_factory(), rng, n_steps=4
    )

    assert outputs.metrics.n_steps == 4
    assert outputs.metrics.n_attempted == 8
    assert all(
        inner_output.metrics is None for inner_output in outputs.round_outputs
    )
    assert all(
        child_output["metrics"] is None
        for inner_output in outputs.round_outputs
        for child_output in inner_output.round_outputs
    )
    assert "_mutation_diagnostics_accumulator" not in context.metadata
    np.testing.assert_array_equal(updated.particles, state.particles + 4.0)


class CountMetric:
    """User metric with a plain dictionary/tuple pytree, no core inheritance."""

    def __init__(self, name="count", factor=1):
        self.name = name
        self.factor = factor

    def initialize(self, particles, log_weights, context):
        from sequince.array_api import namespace

        xp = namespace(particles)
        return {"totals": (xp.asarray(0), xp.asarray(0))}

    def update(self, state, particles, accepted, context):
        from sequince.array_api import namespace

        xp = namespace(particles)
        transitions, accepts = state["totals"]
        return {
            "totals": (transitions + self.factor, accepts + xp.sum(accepted))
        }

    def result(self, state, context):
        return state["totals"]


def test_metric_lifecycle_and_independence(context_factory, rng):
    from unittest.mock import Mock

    first = CountMetric()
    second = CountMetric("scaled_count", 7)
    for metric in (first, second):
        for method in ("initialize", "update", "result"):
            setattr(metric, method, Mock(wraps=getattr(metric, method)))
    mutation = MetropolisHastingsMutation(
        _flat_log_prob, _add(1.0), metrics=[first, second]
    )
    state = ParticleState(np.array([[-1.0], [1.0]]))
    repeat = RepeatedMutationStep(
        mutation, after_round=SequentialStep([NegateStep()]), n_rounds=3
    )
    _, _, output = repeat.apply_with_n_steps(
        state, context_factory(), rng, n_steps=5
    )
    for metric in (first, second):
        assert metric.initialize.call_count == 1
        assert metric.update.call_count == 5
        assert metric.result.call_count == 1
    assert output.metrics.values["count"] == (5, 10)
    assert output.metrics.values["scaled_count"] == (35, 10)


def test_no_metrics_and_sampling_invariance(context_factory):
    from sequince.utils.random import resolve_rng

    state = ParticleState(np.array([[-1.0], [1.0]]))
    from sequince.steps.mutation import GaussianRandomWalk

    outputs = []
    for metrics in ([], [CountMetric()]):
        mutation = MetropolisHastingsMutation(
            lambda p, c: -np.sum(p**2, axis=1),
            GaussianRandomWalk(scale=0.3),
            n_steps=4,
            store_chains=True,
            metrics=metrics,
        )
        result, context, output = mutation(
            state, context_factory(), resolve_rng(42)
        )
        outputs.append((result, context, output))
    assert outputs[0][2]["metrics"].values == {}
    np.testing.assert_array_equal(
        outputs[0][0].particles, outputs[1][0].particles
    )
    np.testing.assert_array_equal(
        outputs[0][2]["chain"], outputs[1][2]["chain"]
    )
    assert (
        outputs[0][2]["metrics"].n_accepted
        == outputs[1][2]["metrics"].n_accepted
    )
    assert outputs[0][1].metadata == outputs[1][1].metadata


def test_duplicate_metric_names_rejected():
    with pytest.raises(ValueError, match="unique.*duplicate"):
        MetropolisHastingsMutation(
            _flat_log_prob, _add(1), metrics=[CountMetric(), CountMetric()]
        )


def test_custom_metric_under_jit():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from sequince.steps.context import StepContext
    from sequince.steps.mutation import BlackJAXMutation

    state = ParticleState(jnp.array([[-1.0], [1.0]]))
    context = StepContext(t=0)
    mutation = BlackJAXMutation(
        lambda p, c: -jnp.sum(p**2) / 2,
        algorithm="random-walk",
        algorithm_kwargs={"sigma": 0.3},
        n_steps=4,
        metrics=[CountMetric()],
        store_chains=True,
    )

    @jax.jit
    def run(state, key):
        return mutation.apply_functional(state, context, key)

    result, _, next_key, output = run(state, jax.random.key(42))
    metrics = output.metrics
    assert metrics.n_steps == 4
    assert metrics.n_attempted == 8
    assert metrics.values["count"][0] == 4
    assert metrics.values["count"][1] == metrics.n_accepted
    assert np.isfinite(metrics.acceptance)
    plain = BlackJAXMutation(
        lambda p, c: -jnp.sum(p**2) / 2,
        algorithm="random-walk",
        algorithm_kwargs={"sigma": 0.3},
        n_steps=4,
    )
    plain_result, _, plain_key, _ = jax.jit(
        lambda s, k: plain.apply_functional(s, context, k)
    )(state, jax.random.key(42))
    np.testing.assert_array_equal(result.particles, plain_result.particles)
    np.testing.assert_array_equal(
        jax.random.key_data(next_key), jax.random.key_data(plain_key)
    )


def test_zero_block_metric_lifecycle(context_factory, rng):
    from unittest.mock import Mock

    metric = CountMetric()
    for name in ("initialize", "update", "result"):
        setattr(metric, name, Mock(wraps=getattr(metric, name)))
    step = RepeatedMutationStep(
        MetropolisHastingsMutation(_flat_log_prob, _add(1), metrics=[metric]),
        n_rounds=3,
    )
    state = ParticleState(np.array([[-1.0], [1.0]]))
    _, _, output = step.apply_with_n_steps(
        state, context_factory(), rng, n_steps=0
    )
    assert metric.initialize.call_count == 1
    assert metric.update.call_count == 0
    assert metric.result.call_count == 1
    assert output.metrics.values["count"] == (0, 0)
    assert output.metrics.acceptance == 0


def test_blocked_mutation_metrics_share_state(context_factory, rng):
    from sequince.steps.mutation import (
        BlockedMetropolisHastingsMutation,
        MetropolisBlock,
    )

    step = BlockedMetropolisHastingsMutation(
        _flat_log_prob,
        [
            MetropolisBlock("first", _add(1), 2),
            MetropolisBlock("second", _add(2), 1),
        ],
        metrics=[CountMetric()],
    )
    state = ParticleState(np.array([[-1.0], [1.0]]))
    _, _, output = step(state, context_factory(), rng)
    assert output["metrics"].values["count"] == (3, 6)


def test_minipcn_functional_metrics_preserve_sampling():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from sequince.steps.context import StepContext
    from sequince.steps.mutation import MiniPCNMutation

    state = ParticleState(jax.random.normal(jax.random.key(1), (16, 2)))
    context = StepContext(t=jnp.asarray(0))
    key = jax.random.key(42)
    results = []
    for metrics in ([], [CountMetric()]):
        step = MiniPCNMutation(
            lambda p, c: -0.5 * jnp.sum(p**2, axis=-1),
            n_steps=3,
            metrics=metrics,
        )
        results.append(jax.jit(step.apply_functional)(state, context, key))
    np.testing.assert_array_equal(
        results[0][0].particles, results[1][0].particles
    )
    np.testing.assert_array_equal(
        jax.random.key_data(results[0][2]), jax.random.key_data(results[1][2])
    )
    output = results[1][3]
    assert output.chain is None
    assert output.metrics.values["count"][0] == 3
    assert output.metrics.n_attempted == 48


def test_emcee_metrics_count_accepted_zero_displacement(context_factory, rng):
    emcee = pytest.importorskip("emcee")
    from sequince.steps.mutation import EmceeMutation

    class IdentityMove(emcee.moves.Move):
        def propose(self, model, state):
            return state, np.ones(state.coords.shape[0], dtype=bool)

    state = ParticleState(np.arange(8, dtype=float)[:, None])
    step = EmceeMutation(
        lambda p, c: 0.0,
        n_steps=3,
        move=IdentityMove(),
        metrics=[CountMetric()],
    )
    _, _, output = step(state, context_factory(), rng)
    assert output["metrics"].values["count"] == (3, 24)
    assert output["metrics"].acceptance == 1


def test_custom_dataclass_and_array_state_in_jax_loop():
    from dataclasses import dataclass, replace
    from typing import Any

    from sequince.steps.context import StepContext
    from sequince.steps.mutation.metrics import (
        MutationMetrics,
    )
    from sequince.utils.pytrees import pytree_dataclass

    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    @pytree_dataclass(data_fields=("total",))
    @dataclass(frozen=True)
    class UserState:
        total: Any

    class ArrayMetric:
        name = "array"

        def initialize(self, particles, log_weights, context):
            return jnp.zeros_like(particles)

        def update(self, state, particles, accepted, context):
            return state + particles

        def result(self, state, context):
            return state

    class DataclassMetric(ArrayMetric):
        name = "dataclass"

        def initialize(self, particles, log_weights, context):
            return UserState(jnp.zeros_like(particles))

        def update(self, state, particles, accepted, context):
            return replace(state, total=state.total + particles)

        def result(self, state, context):
            return state.total

    particles = jnp.ones((2, 1))
    context = StepContext(t=0)
    metrics = MutationMetrics([ArrayMetric(), DataclassMetric()])

    @jax.jit
    def run():
        initial = metrics.initialize(particles, jnp.zeros(2), context)
        final = jax.lax.fori_loop(
            0,
            3,
            lambda i, metric_state: metrics.update(
                metric_state,
                particles,
                jnp.ones(2, dtype=bool),
                context,
            ),
            initial,
        )
        return initial, metrics.result(final, context)

    initial, result = run()
    np.testing.assert_array_equal(initial.states[0], 0)
    np.testing.assert_array_equal(initial.states[1].total, 0)
    np.testing.assert_array_equal(result.values["array"], 3)
    np.testing.assert_array_equal(result.values["dataclass"], 3)


@pytest.mark.parametrize(
    "run_kwargs", [{"thin_by": 2}, {"thin": 2}, {"store": False}]
)
def test_emcee_rejects_hidden_transitions(run_kwargs):
    from sequince.steps.mutation import EmceeMutation

    with pytest.raises(ValueError, match="unthinned transitions"):
        EmceeMutation(_flat_log_prob, run_kwargs=run_kwargs)


@pytest.mark.parametrize("resampled", [False, True])
def test_conditional_mutation_in_shared_block(resampled, context_factory, rng):
    from unittest.mock import Mock

    from sequince.steps.mutation import MutateIfResampled

    metric = CountMetric()
    metric.initialize = Mock(wraps=metric.initialize)
    metric.result = Mock(wraps=metric.result)
    mutation = MetropolisHastingsMutation(
        _flat_log_prob, _add(1), metrics=[metric]
    )
    step = MutateIfResampled(RepeatedMutationStep(mutation, n_rounds=3))
    state = ParticleState(np.array([[-1.0], [1.0]]))
    context = context_factory().with_metadata(resampled=resampled)
    _, _, output = step.apply_with_n_steps(state, context, rng, n_steps=4)
    assert output.executed == resampled
    output = output.child_output
    if resampled:
        assert output.metrics.values["count"] == (4, 8)
        assert metric.initialize.call_count == metric.result.call_count == 1
        assert all(child["metrics"] is None for child in output.round_outputs)
    else:
        assert output is None
        assert metric.initialize.call_count == metric.result.call_count == 0


def test_block_requires_an_explicit_mutation_child():
    class BudgetOnlyMutation(SMCStep):
        def apply_with_n_steps(self, state, context, rng, *, n_steps):
            pytest.fail("Invalid mutation must not execute")

    with pytest.raises(
        TypeError, match="MutationStep or RepeatedMutationStep"
    ):
        RepeatedMutationStep(BudgetOnlyMutation())


def test_repeated_block_returns_structured_results_under_jit():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from orng import RandomGenerator

    from sequince.steps.context import StepContext
    from sequince.steps.mutation import BlackJAXMutation

    step = RepeatedMutationStep(
        BlackJAXMutation(
            lambda p, c: -0.5 * jnp.sum(p**2),
            algorithm="random-walk",
            algorithm_kwargs={"sigma": 0.1},
            metrics=[CountMetric()],
        ),
        n_rounds=2,
    )

    @jax.jit
    def run(particles, key):
        return step.apply_with_n_steps(
            ParticleState(particles),
            StepContext(t=0),
            RandomGenerator(backend="jax", generator=key),
            n_steps=3,
        )

    _, _, output = run(jnp.array([[-1.0], [1.0]]), jax.random.key(42))
    assert [child.steps for child in output.round_outputs] == [2, 1]
    assert all(child.metrics is None for child in output.round_outputs)
    assert output.metrics.n_steps == 3
    assert output.metrics.values["count"][0] == 3


def test_public_continuation_preserves_state_without_finalizing(
    context_factory,
):
    from unittest.mock import Mock

    from sequince.steps.mutation import GaussianRandomWalk
    from sequince.utils.random import resolve_rng

    counter = CountMetric()
    counter.initialize = Mock(wraps=counter.initialize)
    counter.result = Mock(wraps=counter.result)
    mutation = MetropolisHastingsMutation(
        lambda p, c: -np.sum(p**2, axis=1),
        GaussianRandomWalk(scale=0.5),
        metrics=[counter],
        store_chains=True,
    )
    initial = ParticleState(np.array([[-1.0], [1.0]]))
    rng = resolve_rng(42)
    first, context, first_output, first_state = (
        mutation.apply_with_mutation_state(
            initial, context_factory(), rng, n_steps=2, mutation_state=None
        )
    )
    final, context, final_output, final_state = (
        mutation.apply_with_mutation_state(
            first, context, rng, n_steps=4, mutation_state=first_state
        )
    )
    unchanged, _, zero_output, zero_state = mutation.apply_with_mutation_state(
        final, context, rng, n_steps=0, mutation_state=final_state
    )

    assert counter.initialize.call_count == 1
    assert counter.result.call_count == 0
    assert first_output["metrics"] is None
    assert final_output["metrics"] is None
    assert zero_output["metrics"] is None
    assert isinstance(first_state, MutationStepState)
    assert isinstance(final_state, MutationStepState)
    assert first_state.n_steps == 2
    assert final_state.n_steps == zero_state.n_steps == 6
    np.testing.assert_array_equal(unchanged.particles, final.particles)

    complete, _, output = mutation.apply_with_n_steps(
        initial, context_factory(), resolve_rng(42), n_steps=6
    )
    assert counter.initialize.call_count == 2
    assert counter.result.call_count == 1
    np.testing.assert_array_equal(final.particles, complete.particles)
    np.testing.assert_array_equal(
        np.concatenate([first_output["chain"], final_output["chain"]]),
        output["chain"],
    )
    assert final_state.n_accepted == output["metrics"].n_accepted


def test_metric_collection_prepare_preserves_accumulation(context_factory):
    from sequince.steps.mutation.metrics import MutationMetrics

    class SegmentMetric(CountMetric):
        name = "segments"

        def __init__(self):
            super().__init__("segments")

        def initialize(self, particles, log_weights, context):
            return {"totals": (0, 0), "segments": 0}

        def update(self, state, particles, accepted, context):
            return {
                **super().update(state, particles, accepted, context),
                "segments": state["segments"],
            }

        def prepare(self, state, particles, context):
            return {**state, "segments": state["segments"] + 1}

        def result(self, state, context):
            return state

    metrics = MutationMetrics([SegmentMetric(), CountMetric()])
    context = context_factory()
    particles = np.zeros((2, 1))
    initial = metrics.initialize(particles, np.zeros(2), context)
    first = metrics.update(initial, particles, np.ones(2, dtype=bool), context)
    prepared = metrics.prepare(first, particles + 100, context)
    final = metrics.update(
        prepared, particles, np.ones(2, dtype=bool), context
    )
    result = metrics.result(final, context)
    assert result.n_steps == 2
    assert result.n_accepted == result.n_attempted == 4
    assert result.values["segments"] == {"totals": (2, 4), "segments": 1}
    assert result.values["count"] == (2, 4)
    assert first.states[0]["segments"] == 0
    assert initial.n_steps == 0
