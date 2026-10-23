from __future__ import annotations

import h5py
import numpy as np
import pytest

from sequince import (
    MarkovChainMonteCarlo,
    MCMCResult,
    ParticleState,
    make_mcmc,
)
from sequince.checkpoint import HDF5CheckpointWriter, SMCCheckpoint
from sequince.steps.mutation import (
    CallableMutationKernel,
    MetropolisHastingsMutation,
    MutationStep,
    get_mutation_step,
)
from sequince.transforms import AffineTransform


def make_runner(target, *, n_steps=4, checkpoint_writers=None, transform=None):
    return make_mcmc(
        target,
        mutation="metropolis-hastings",
        n_steps=n_steps,
        mutation_kwargs={
            "kernel": "gaussian-random-walk",
            "kernel_kwargs": {"scale": 0.1},
        },
        particle_transform=transform,
        checkpoint_writers=checkpoint_writers,
    )


@pytest.mark.parametrize("transformed", [False, True])
def test_mcmc_returns_chain_in_particle_coordinates(transformed: bool) -> None:
    # Mean=12, scale=2: a unit latent move must be two particle units.
    particles = np.array([[10.0], [14.0]])
    mutation = MetropolisHastingsMutation(
        lambda x: np.zeros(len(x)),
        CallableMutationKernel(lambda x, rng: x + 1, symmetric=True),
        n_steps=4,
        transform=AffineTransform(eps=0.0) if transformed else None,
    )
    result = MarkovChainMonteCarlo(mutation).run(
        ParticleState(particles), rng=123
    )

    stride = 2 if transformed else 1
    expected = particles[None, ...] + stride * np.arange(1, 5)[:, None, None]
    assert isinstance(result, MCMCResult)
    np.testing.assert_allclose(result.chain, expected)
    np.testing.assert_allclose(result.final_state.particles, expected[-1])
    np.testing.assert_allclose(
        result.samples(burn_in=1, thin=2), expected[[1, 3]].reshape(-1, 1)
    )
    np.testing.assert_allclose(
        result.samples(burn_in=1, thin=2, flat=False), expected[[1, 3]]
    )
    assert "mutation_chains" not in result.final_state.metadata
    assert result.diagnostics["metrics"].acceptance == 1.0
    assert result.diagnostics["steps"] == 4


def test_mcmc_checkpoint_resume_concatenates_chains(
    simple_particles,
    tmp_path,
    standard_normal_log_prob,
):
    checkpoint_path = tmp_path / "mcmc.h5"
    writer = HDF5CheckpointWriter(checkpoint_path)
    state = ParticleState(simple_particles)

    first = make_runner(
        standard_normal_log_prob,
        n_steps=2,
        checkpoint_writers=[writer],
    ).run(state, rng=789)

    with h5py.File(checkpoint_path, "r") as h5_file:
        checkpoint = SMCCheckpoint.load_hdf5(h5_file)

    resumed_runner = make_runner(standard_normal_log_prob, n_steps=2)
    runtime = resumed_runner.resume_from(checkpoint)
    resumed = resumed_runner.run(runtime=runtime)

    reference = make_runner(standard_normal_log_prob, n_steps=4).run(
        state, rng=789
    )
    np.testing.assert_array_equal(resumed.chain, reference.chain)
    np.testing.assert_array_equal(
        resumed.final_state.particles, reference.final_state.particles
    )
    assert first.chain.shape == (2, len(simple_particles), 2)
    assert resumed.chain.shape == (4, len(simple_particles), 2)
    np.testing.assert_allclose(resumed.chain[:2], first.chain)


def test_minipcn_chain_contains_requested_number_of_steps(
    simple_particles, standard_normal_log_prob
):
    runner = make_mcmc(
        standard_normal_log_prob,
        mutation="minipcn",
        n_steps=3,
    )

    result = runner.run(ParticleState(simple_particles), rng=321)

    assert result.chain.shape == (3, len(simple_particles), 2)


def test_direct_mutation_step_is_configured_to_store_chains(
    simple_particles, standard_normal_log_prob
):
    mutation = get_mutation_step(
        "metropolis-hastings",
        target=standard_normal_log_prob,
        mutation_n_steps=2,
        mutation_kwargs={
            "kernel": "gaussian-random-walk",
            "kernel_kwargs": {"scale": 0.1},
        },
    )
    assert isinstance(mutation, MutationStep)

    runner = MarkovChainMonteCarlo(mutation)
    result = runner.run(ParticleState(simple_particles), rng=654)

    assert mutation.store_chains
    assert result.chain.shape == (2, len(simple_particles), 2)
