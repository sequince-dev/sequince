from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
import pytest
from orng import RandomGenerator

from sequince.steps.base import SMCStep

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


class FunctionalParticleState(NamedTuple):
    particles: Any


class FunctionalContext(NamedTuple):
    total: Any


class TwoDrawStep(SMCStep):
    """Small fixed-structure step used to isolate the functional RNG boundary."""

    def apply(self, state, context, rng):
        first = rng.normal(size=state.particles.shape)
        second = rng.normal(size=state.particles.shape)
        updated_state = FunctionalParticleState(
            particles=state.particles + first + second
        )
        updated_context = FunctionalContext(
            total=context.total + jnp.sum(first + second)
        )
        return updated_state, updated_context, (first, second)


def _inputs():
    return (
        FunctionalParticleState(jnp.zeros((4,))),
        FunctionalContext(jnp.asarray(0.0)),
        jax.random.key(123),
    )


def _assert_trees_equal(left, right) -> None:
    left_leaves = jax.tree.leaves(left)
    right_leaves = jax.tree.leaves(right)
    assert len(left_leaves) == len(right_leaves)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        if hasattr(left_leaf, "dtype") and jax.dtypes.issubdtype(
            left_leaf.dtype, jax.dtypes.prng_key
        ):
            left_leaf = jax.random.key_data(left_leaf)
            right_leaf = jax.random.key_data(right_leaf)
        np.testing.assert_array_equal(left_leaf, right_leaf)


def _keys_equal(left, right) -> bool:
    return np.array_equal(
        jax.random.key_data(left),
        jax.random.key_data(right),
    )


def test_functional_step_reproduces_and_advances_random_state() -> None:
    step = TwoDrawStep()
    state, context, key = _inputs()

    first = step.apply_functional(state, context, key)
    repeated = step.apply_functional(state, context, key)
    advanced = step.apply_functional(state, context, first[2])

    _assert_trees_equal(first, repeated)
    assert not _keys_equal(first[2], key)
    assert not np.array_equal(advanced[3][0], first[3][0])
    assert not np.array_equal(first[3][0], first[3][1])
    assert context.total == 0.0
    assert not hasattr(step, "rng")
    assert not hasattr(step, "random_state")


def test_apply_and_functional_step_agree_from_equivalent_jax_key() -> None:
    step = TwoDrawStep()
    state, context, key = _inputs()
    rng = RandomGenerator(backend="jax", generator=key)

    applied_state, applied_context, applied_output = step.apply(
        state, context, rng
    )
    functional_state, functional_context, functional_key, functional_output = (
        step.apply_functional(state, context, key)
    )

    _assert_trees_equal(applied_state, functional_state)
    _assert_trees_equal(applied_context, functional_context)
    _assert_trees_equal(applied_output, functional_output)
    state_method = getattr(rng, "state", None)
    expected_key = (
        state_method() if state_method is not None else rng.to_functional()[1]
    )
    assert _keys_equal(functional_key, expected_key)


def test_functional_step_can_be_jitted_without_leaking_tracers() -> None:
    step = TwoDrawStep()
    state, context, key = _inputs()
    compiled_step = jax.jit(step.apply_functional)

    first = compiled_step(state, context, key)
    advanced = compiled_step(state, context, first[2])
    repeated = compiled_step(state, context, key)

    _assert_trees_equal(first, repeated)
    assert not _keys_equal(first[2], key)
    assert not np.array_equal(advanced[3][0], first[3][0])
    assert jax.tree.structure(first) == jax.tree.structure(advanced)
    assert not vars(step)


@pytest.mark.parametrize("execute", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
def test_conditional_functional_step(execute: bool, compiled: bool) -> None:
    from sequince import ParticleState
    from sequince.steps import StepContext, apply_functional_if
    from sequince.target import EvaluationCounts, TargetEvaluation

    calls = []

    def apply(state, context, key):
        jax.debug.callback(lambda: calls.append(True))
        key, draw_key = jax.random.split(key)
        draws = jax.random.normal(draw_key, state.particles.shape)
        return (
            state.with_updates(
                particles=state.particles + draws, evaluation=state.evaluation
            ),
            context.add_evaluation_counts(
                EvaluationCounts(log_target_evaluations=4)
            ),
            key,
            {"draws": draws, "optional": None, "nested": (jnp.asarray(1),)},
        )

    state = ParticleState(
        jnp.zeros((4, 1)),
        evaluation=TargetEvaluation({"cached": jnp.arange(4.0)}),
    )
    context = StepContext(t=0)
    key = jax.random.key(42)

    def conditional(predicate, state, context, key):
        return apply_functional_if(predicate, apply, state, context, key)

    operation = jax.jit(conditional) if compiled else conditional
    updated, updated_context, next_key, output = operation(
        jnp.asarray(execute), state, context, key
    )
    jax.effects_barrier()
    assert calls == ([True] if execute else [])
    assert output["optional"] is None
    assert output["draws"].shape == state.particles.shape
    np.testing.assert_array_equal(
        updated.evaluation["cached"], state.evaluation["cached"]
    )
    if execute:
        assert updated_context.evaluation_counts.log_target_evaluations == 4
        assert not _keys_equal(key, next_key)
        np.testing.assert_allclose(updated.particles, output["draws"])
    else:
        _assert_trees_equal(
            (updated, updated_context, next_key), (state, context, key)
        )
        np.testing.assert_array_equal(
            output["draws"], jnp.zeros_like(state.particles)
        )
        assert output["nested"][0] == 0


@pytest.mark.parametrize("immediate", [False, True])
@pytest.mark.parametrize("stop", [False, True])
def test_functional_stopif_guards_later_steps(
    immediate: bool, stop: bool
) -> None:
    from sequince import ParticleState
    from sequince.jax import compile_functional_iteration
    from sequince.smc import SequentialMonteCarlo
    from sequince.steps.control import StopIf

    class Draw(SMCStep):
        def apply_functional(self, state, context, key):
            key, draw_key = jax.random.split(key)
            return (
                state.with_updates(
                    particles=state.particles
                    + jax.random.normal(draw_key, state.particles.shape)
                ),
                context,
                key,
                jnp.asarray(1),
            )

    state = ParticleState(jnp.zeros((4, 1)))
    key = jax.random.key(42)
    sampler = SequentialMonteCarlo(
        [
            StopIf(
                lambda state, context: context.t == (0 if stop else 1),
                immediate=immediate,
            ),
            Draw(),
            Draw(),
        ],
        auto_normalize=False,
    )
    iteration = compile_functional_iteration(sampler, state, key)
    result, final_key = sampler.run_functional(
        state, random_state=key, num_steps=1, iteration=iteration
    )
    skipped = stop and immediate
    assert _keys_equal(key, final_key) == skipped
    assert bool(result.context.stop_requested) == stop
    assert bool(result.context.stop_immediately) == skipped
    assert [int(output) for output in result.outputs[0][1:]] == (
        [0, 0] if skipped else [1, 1]
    )
    if skipped:
        np.testing.assert_array_equal(
            result.final_state.particles, state.particles
        )


@pytest.mark.parametrize("prefix_steps", [0, 2])
def test_custom_functional_stopper_without_declaration(
    prefix_steps: int,
) -> None:
    from sequince import ParticleState
    from sequince.smc import SequentialMonteCarlo
    from sequince.steps import StepContext

    class Stop(SMCStep):
        def apply_functional(self, state, context, key):
            return state, context.request_stop_immediately(), key, None

    class Change(SMCStep):
        def apply_functional(self, state, context, key):
            return (
                state.with_updates(particles=state.particles + 1),
                context,
                key,
                None,
            )

    state = ParticleState(jnp.zeros((4, 1)))
    sampler = SequentialMonteCarlo(
        [*[Change() for _ in range(prefix_steps)], Stop(), Change()],
        auto_normalize=False,
    )
    updated, context, _, _ = jax.jit(sampler.functional_iteration)(
        state, StepContext(t=0), jax.random.key(42)
    )
    np.testing.assert_array_equal(
        updated.particles, state.particles + prefix_steps
    )
    assert context.stop_immediately


def test_conditional_functional_step_requires_initialized_carry() -> None:
    from sequince import ParticleState
    from sequince.steps import StepContext, apply_functional_if

    def adds_metadata(state, context, key):
        return state, context.with_metadata(new=jnp.asarray(1)), key, None

    with pytest.raises(ValueError, match="carry structure"):
        apply_functional_if(
            jnp.asarray(False),
            adds_metadata,
            ParticleState(jnp.zeros((4, 1))),
            StepContext(t=0),
            jax.random.key(42),
        )
