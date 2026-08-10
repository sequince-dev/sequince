from __future__ import annotations

import numpy as np
import pytest
from orng import RandomGenerator

from sequince import make_standard_bayesian_smc
from sequince.posterior import TemperedPosterior
from sequince.state import ParticleState
from sequince.steps.context import StepContext
from sequince.steps.mutation import BlackJAXMutation
from sequince.utils import resolve_rng

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


def _posterior() -> TemperedPosterior:
    return TemperedPosterior(
        log_prior=lambda x, ctx: -0.5 * jnp.sum(x**2, axis=-1),
        log_likelihood=lambda x, ctx: -0.5 * jnp.sum((x - 1.0) ** 2, axis=-1),
    )


def _state() -> ParticleState:
    particles = jnp.asarray(
        [[-1.0, 0.5], [0.0, -0.5], [0.5, 0.5], [1.5, -1.0]]
    )
    return ParticleState(particles=particles)


@pytest.mark.parametrize(
    ("algorithm", "algorithm_kwargs"),
    [
        (
            "nuts",
            {"step_size": 0.1, "inverse_mass_matrix": jnp.ones(2)},
        ),
        (
            "hmc",
            {
                "step_size": 0.1,
                "inverse_mass_matrix": jnp.ones(2),
                "num_integration_steps": 2,
            },
        ),
        ("random-walk", {"sigma": 0.1}),
    ],
)
def test_blackjax_mutates_jax_particles(algorithm, algorithm_kwargs) -> None:
    mutation = BlackJAXMutation(
        _posterior(),
        algorithm=algorithm,
        algorithm_kwargs=algorithm_kwargs,
        n_steps=2,
        store_chains=True,
    )
    context = StepContext(t=0, metadata={"temperature": 0.5})

    result, _, output = mutation(_state(), context, resolve_rng(42))

    assert result.particles.shape == (4, 2)
    assert isinstance(result.particles, jax.Array)
    assert result.evaluation is not None
    assert output["algorithm"] == algorithm
    assert output["steps"] == 2
    assert 0.0 <= output["acceptance"] <= 1.0
    assert "mutation_chains" not in result.metadata
    assert output["chain"].shape == (2, 4, 2)
    assert np.all(np.isfinite(np.asarray(result.particles)))


def test_blackjax_rejects_non_jax_particles() -> None:
    mutation = BlackJAXMutation(_posterior(), algorithm="random-walk")
    state = ParticleState(particles=np.zeros((4, 2)))

    with pytest.raises(TypeError, match="JAX-backed"):
        mutation(state, StepContext(t=0), resolve_rng(42))


def test_blackjax_consumes_jax_rng_key() -> None:
    initial_key = jax.random.key(42)
    expected_key, expected_state = jax.random.split(initial_key)
    rng = RandomGenerator(backend="jax", generator=initial_key)

    key = BlackJAXMutation._jax_key(rng)

    np.testing.assert_array_equal(
        jax.random.key_data(key),
        jax.random.key_data(expected_key),
    )
    np.testing.assert_array_equal(
        jax.random.key_data(rng.state()),
        jax.random.key_data(expected_state),
    )


def test_blackjax_jax_key_consumption_can_be_jitted() -> None:
    def consume_key(random_state):
        rng = RandomGenerator(backend="jax", generator=random_state)
        key = BlackJAXMutation._jax_key(rng)
        return key, rng.state()

    key, next_state = jax.jit(consume_key)(jax.random.key(42))

    assert not np.array_equal(
        jax.random.key_data(key),
        jax.random.key_data(next_state),
    )


def test_blackjax_array_boundary_accepts_dynamic_temperature_under_jit() -> (
    None
):
    mutation = BlackJAXMutation(
        _posterior(),
        algorithm="random-walk",
        algorithm_kwargs={"sigma": 0.1},
        n_steps=2,
    )

    @jax.jit
    def compiled_step(particles, temperature, random_state):
        state = ParticleState(particles=particles)
        context = StepContext(t=0, metadata={"temperature": temperature})
        updated, _, next_state, output = mutation.apply_functional(
            state,
            context,
            random_state,
        )
        return updated.particles, next_state, output["acceptance"]

    particles, next_state, acceptance = compiled_step(
        _state().particles,
        jnp.asarray(0.5),
        jax.random.key(42),
    )

    assert particles.shape == (4, 2)
    assert acceptance.shape == ()
    assert not np.array_equal(
        jax.random.key_data(next_state),
        jax.random.key_data(jax.random.key(42)),
    )


def test_blackjax_runs_in_standard_smc() -> None:
    posterior = _posterior()
    smc, _, metadata = make_standard_bayesian_smc(
        posterior.log_prior,
        posterior.log_likelihood,
        adaptive=False,
        num_steps=1,
        mutation="blackjax",
        mutation_n_steps=2,
        mutation_kwargs={
            "algorithm": "random-walk",
            "algorithm_kwargs": {"sigma": 0.1},
        },
        particle_transform=None,
        logging_frequency=0,
    )

    result = smc.run(_state(), num_steps=1, rng=42, metadata=metadata)

    assert result.final_state.particles.shape == (4, 2)
    assert result.outputs[0][-1]["algorithm"] == "random-walk"
    assert result.final_state.evaluation is not None


def test_blackjax_validates_configuration() -> None:
    with pytest.raises(ValueError, match="n_steps"):
        BlackJAXMutation(_posterior(), n_steps=0)
    with pytest.raises(ValueError, match="algorithm must be"):
        BlackJAXMutation(_posterior(), algorithm="invalid")
