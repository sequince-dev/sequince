from __future__ import annotations

import numpy as np
import pytest

from sequince import ParticleState, make_standard_bayesian_smc
from sequince.jax import compile_functional_iteration
from sequince.steps import StepContext
from sequince.steps.mutation import MiniPCNMutation

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


def _standard_normal_log_prob(particles, context):
    del context
    return -0.5 * jnp.sum(particles**2, axis=-1)


@pytest.mark.parametrize("step_fn", ["pcn", "tpcn"])
@pytest.mark.parametrize("store_chains", [False, True])
def test_functional_step_can_be_jitted(
    step_fn: str,
    store_chains: bool,
) -> None:
    step = MiniPCNMutation(
        _standard_normal_log_prob,
        n_steps=2,
        step_fn=step_fn,
        store_chains=store_chains,
    )
    state = ParticleState(jax.random.normal(jax.random.key(1), shape=(16, 2)))
    context = StepContext(t=jnp.asarray(0))
    key = jax.random.key(42)
    functional_step = jax.jit(step.apply_functional)

    updated, next_context, next_key, output = functional_step(
        state, context, key
    )
    replayed, _, replayed_key, replayed_output = functional_step(
        state, context, key
    )

    assert updated.particles.shape == state.particles.shape
    assert next_context.t == context.t
    assert output.step_fn == step_fn
    assert output.steps == 2
    assert 0.0 <= output.acceptance <= 1.0
    if store_chains:
        assert output.chain.shape == (2, *state.particles.shape)
    else:
        assert output.chain is None
    np.testing.assert_array_equal(updated.particles, replayed.particles)
    np.testing.assert_array_equal(
        output.acceptance, replayed_output.acceptance
    )
    np.testing.assert_array_equal(
        jax.random.key_data(next_key),
        jax.random.key_data(replayed_key),
    )
    assert not np.array_equal(
        jax.random.key_data(key),
        jax.random.key_data(next_key),
    )


@pytest.mark.parametrize("step_fn", ["pcn", "tpcn"])
def test_functional_smc_with_minipcn_reaches_max_temperature(
    step_fn: str,
) -> None:
    def log_prior(particles, context):
        del context
        return -0.5 * jnp.sum(particles**2, axis=-1)

    def log_likelihood(particles, context):
        del context
        return -0.5 * jnp.sum((particles - 1.0) ** 2, axis=-1)

    smc, _, metadata = make_standard_bayesian_smc(
        log_prior,
        log_likelihood,
        target_ess_ratio=0.8,
        always_resample=True,
        mutation="minipcn",
        mutation_n_steps=2,
        mutation_kwargs={"step_fn": step_fn},
        particle_transform=None,
        logging_frequency=0,
    )
    state = ParticleState(jax.random.normal(jax.random.key(1), shape=(32, 2)))
    key = jax.random.key(42)
    iteration = compile_functional_iteration(
        smc,
        state,
        key,
        metadata=metadata,
    )

    result, next_key = smc.run_functional(
        state,
        num_steps=20,
        random_state=key,
        metadata=metadata,
        iteration=iteration,
    )

    assert result.context.metadata["temperature"] == pytest.approx(1.0)
    assert result.context.stop_requested
    assert len(result) < 20
    assert result.final_state.particles.shape == state.particles.shape
    assert not np.array_equal(
        jax.random.key_data(key),
        jax.random.key_data(next_key),
    )
