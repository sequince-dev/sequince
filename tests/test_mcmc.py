from __future__ import annotations

import h5py
import numpy as np

from sequince import (
    MarkovChainMonteCarlo,
    MCMCResult,
    ParticleState,
    make_mcmc,
)
from sequince.checkpoint import HDF5CheckpointWriter, SMCCheckpoint
from sequince.steps.mutation import MutationStep, get_mutation_step
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


def test_mcmc_returns_chain_and_processed_samples(
    simple_particles, standard_normal_log_prob
):
    state = ParticleState(simple_particles)

    result = make_runner(standard_normal_log_prob, n_steps=4).run(
        state, rng=123
    )

    assert isinstance(result, MCMCResult)
    assert "mutation_chains" not in result.final_state.metadata
    assert result.chain.shape == (4, len(simple_particles), 2)
    assert result.samples(burn_in=1, thin=2).shape == (
        2 * len(simple_particles),
        2,
    )
    np.testing.assert_allclose(result.chain[-1], result.final_state.particles)
    assert "acceptance" in result.diagnostics
    assert result.diagnostics["steps"] == 4


def test_mcmc_stores_chain_in_particle_coordinates(
    simple_particles, standard_normal_log_prob
):
    state = ParticleState(simple_particles)

    result = make_runner(
        standard_normal_log_prob,
        n_steps=3,
        transform=AffineTransform(),
    ).run(state, rng=456)

    np.testing.assert_allclose(result.chain[-1], result.final_state.particles)
    assert np.max(np.abs(result.chain)) > 0.5


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
