from __future__ import annotations

import numpy as np

from sequince.state import ParticleState
from sequince.steps.mutation import (
    CallableMutationKernel,
    GaussianRandomWalk,
    MetropolisHastingsMutation,
)


def test_metropolis_hastings_mutation_accepts_symmetric_proposals(
    particle_state: ParticleState,
    context_factory,
    rng,
) -> None:
    kernel = CallableMutationKernel(
        lambda particles, inner_rng: particles + 0.5,
        symmetric=True,
    )
    step = MetropolisHastingsMutation(
        lambda particles, context: np.zeros(particles.shape[0]),
        kernel,
        n_steps=2,
        store_chains=True,
    )

    updated = step(particle_state, context_factory(), rng)

    np.testing.assert_allclose(
        updated.particles, particle_state.particles + 1.0
    )
    np.testing.assert_allclose(updated.log_weights, particle_state.log_weights)
    assert updated.metadata["metropolis_hastings_acceptance"] == 1.0
    assert updated.metadata["metropolis_hastings_acceptance_by_step"] == [
        1.0,
        1.0,
    ]
    assert updated.metadata["mutation_chains"].shape == (2, 4, 2)


def test_metropolis_hastings_mutation_applies_transition_ratio(
    particle_state: ParticleState,
    context_factory,
    rng,
) -> None:
    kernel = CallableMutationKernel(
        lambda particles, inner_rng: particles + 1.0,
        log_transition_ratio_fn=lambda start, end: np.full(
            start.shape[0], -np.inf
        ),
    )
    step = MetropolisHastingsMutation(
        lambda particles, context: 0.0,
        kernel,
    )

    updated = step(particle_state, context_factory(), rng)

    np.testing.assert_allclose(updated.particles, particle_state.particles)
    assert updated.metadata["metropolis_hastings_acceptance"] == 0.0


def test_metropolis_hastings_mutation_supports_kernel_factory(
    particle_state: ParticleState,
    context_factory,
    rng,
) -> None:
    context = context_factory(metadata={"proposal_scale": 0.2})

    def factory(particles, inner_context):
        inner_context.metadata["factory_shape"] = particles.shape
        return GaussianRandomWalk(
            scale=inner_context.metadata["proposal_scale"]
        )

    step = MetropolisHastingsMutation(
        lambda particles, inner_context: np.zeros(particles.shape[0]),
        factory,
    )

    updated = step(particle_state, context, rng)

    assert updated.particles.shape == particle_state.particles.shape
    assert context.metadata["factory_shape"] == particle_state.particles.shape
