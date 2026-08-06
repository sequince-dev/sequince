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
        if jax.dtypes.issubdtype(left_leaf.dtype, jax.dtypes.prng_key):
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

    first = step.functional_step(state, context, key)
    repeated = step.functional_step(state, context, key)
    advanced = step.functional_step(state, context, first[2])

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
        step.functional_step(state, context, key)
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
    compiled_step = jax.jit(step.functional_step)

    first = compiled_step(state, context, key)
    advanced = compiled_step(state, context, first[2])
    repeated = compiled_step(state, context, key)

    _assert_trees_equal(first, repeated)
    assert not _keys_equal(first[2], key)
    assert not np.array_equal(advanced[3][0], first[3][0])
    assert jax.tree.structure(first) == jax.tree.structure(advanced)
    assert not vars(step)
