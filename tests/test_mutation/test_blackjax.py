from __future__ import annotations

import numpy as np
import pytest
from orng import RandomGenerator

from sequince import make_standard_bayesian_smc
from sequince.posterior import TemperedPosterior
from sequince.state import ParticleState
from sequince.steps.context import StepContext
from sequince.steps.mutation import BlackJAXMutation
from sequince.transforms import ParticleTransform
from sequince.utils.random import resolve_rng

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


class ParticleAlignedShiftTransform(ParticleTransform):
    def forward(self, particles, context=None, *, rng=None):
        del context, rng
        offsets = 10.0 * jnp.arange(particles.shape[0])[:, None]
        return particles - offsets, jnp.zeros(particles.shape[0]), offsets

    def inverse(self, values, context=None, *, auxiliary=None, rng=None):
        del context, rng
        if auxiliary is None:
            raise ValueError("particle-aligned offsets are required")
        return values + auxiliary, jnp.zeros_like(values[..., 0])


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
    context = StepContext(t=0, metadata={"beta": 0.5})

    result, _, output = mutation(_state(), context, resolve_rng(42))

    assert result.particles.shape == (4, 2)
    assert isinstance(result.particles, jax.Array)
    assert result.evaluation is not None
    assert output["algorithm"] == algorithm
    assert output["steps"] == 2
    assert 0.0 <= output["metrics"].acceptance <= 1.0
    assert "mutation_chains" not in result.metadata
    assert output["chain"].shape == (2, 4, 2)
    assert "acceptance" not in output
    if algorithm in {"nuts", "hmc"}:
        assert 0 <= output.mean_acceptance_probability <= 1
        assert output["num_integration_steps"].shape == (2, 4)
        assert np.all(np.asarray(output["num_integration_steps"]) > 0)
    else:
        assert output["num_integration_steps"] is None
    assert np.all(np.isfinite(np.asarray(result.particles)))


def test_blackjax_rejects_non_jax_particles() -> None:
    mutation = BlackJAXMutation(_posterior(), algorithm="random-walk")
    state = ParticleState(particles=np.zeros((4, 2)))

    with pytest.raises(TypeError, match="JAX-backed"):
        mutation(state, StepContext(t=0), resolve_rng(42))


def test_blackjax_selects_particle_aligned_auxiliary() -> None:
    mutation = BlackJAXMutation(
        _posterior(),
        algorithm="random-walk",
        algorithm_kwargs={"sigma": 0.01},
        transform=ParticleAlignedShiftTransform(),
    )

    @jax.jit
    def compiled_step(particles, random_state):
        state = ParticleState(particles=particles)
        context = StepContext(t=0, metadata={"beta": jnp.asarray(0.5)})
        result, _, next_state, _ = mutation.apply_functional(
            state, context, random_state
        )
        return result.particles, next_state

    particles, next_state = compiled_step(
        _state().particles, jax.random.key(42)
    )

    assert particles.shape == _state().particles.shape
    assert np.all(np.isfinite(np.asarray(particles)))
    assert not np.array_equal(
        jax.random.key_data(next_state),
        jax.random.key_data(jax.random.key(42)),
    )


def test_blackjax_pairs_unique_auxiliary_with_particle_index() -> None:
    observed_single_particles: list[float] = []

    def record_original_position(position) -> None:
        values = np.asarray(position)
        if values.ndim == 1:
            observed_single_particles.append(float(values[0]))

    def log_prob(position):
        jax.debug.callback(record_original_position, position)
        return -0.5 * jnp.sum(position**2, axis=-1)

    mutation = BlackJAXMutation(
        log_prob,
        algorithm="random-walk",
        algorithm_kwargs={"sigma": 0.0},
        n_steps=2,
        transform=ParticleAlignedShiftTransform(),
        store_chains=True,
    )

    @jax.jit
    def compiled_step(particles, random_state):
        state = ParticleState(particles=particles)
        result, _, next_state, output = mutation.apply_functional(
            state, StepContext(t=0), random_state
        )
        return result.particles, next_state, output.chain

    result, _, chain = compiled_step(_state().particles, jax.random.key(42))
    result.block_until_ready()

    expected = set(np.asarray(_state().particles[:, 0]).tolist())
    assert expected.issubset(set(observed_single_particles))
    np.testing.assert_allclose(
        chain,
        jnp.stack([_state().particles, _state().particles]),
        atol=1e-6,
    )


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


def test_blackjax_array_boundary_accepts_dynamic_beta_under_jit() -> None:
    posterior = TemperedPosterior(
        lambda x: 0.0,
        lambda x: jnp.where(x[..., 0] < 0, -jnp.inf, 0.0),
    )
    mutation = BlackJAXMutation(
        posterior,
        algorithm="random-walk",
        algorithm_kwargs={"sigma": 1.0},
        n_steps=1,
    )

    @jax.jit
    def compiled_step(beta):
        updated, _, _, output = mutation.apply_functional(
            ParticleState(jnp.zeros((32, 1))),
            StepContext(t=0, metadata={"beta": beta}),
            jax.random.key(42),
        )
        return updated.particles, output.metrics.acceptance

    proposals, prior_acceptance = compiled_step(jnp.asarray(0.0))
    constrained, posterior_acceptance = compiled_step(jnp.asarray(1.0))
    # At beta=0 every proposal is accepted. At beta=1 negative proposals
    # have zero density and must remain at their starting point, zero.
    assert np.any(np.asarray(proposals) < 0)
    assert np.any(np.asarray(proposals) > 0)
    np.testing.assert_array_equal(constrained, jnp.maximum(proposals, 0.0))
    assert float(prior_acceptance) == 1.0
    assert float(posterior_acceptance) == np.mean(np.asarray(proposals) >= 0)


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
