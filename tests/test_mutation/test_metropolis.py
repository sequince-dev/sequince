from __future__ import annotations

import numpy as np

from sequince.posterior import TemperedPosterior
from sequince.state import ParticleState
from sequince.steps.context import StepContext
from sequince.steps.mutation import (
    CallableMutationKernel,
    GaussianRandomWalk,
    MetropolisHastingsMutation,
)
from sequince.transforms import ParticleTransform


class SignFoldingTransform(ParticleTransform):
    def forward(self, particles, context=None, *, rng=None):
        del context, rng
        values = np.asarray(particles)
        signs = np.where(values < 0, -1.0, 1.0)
        return np.abs(values), np.zeros(values.shape[0]), signs

    def inverse(self, values, context=None, *, auxiliary=None, rng=None):
        del context, rng
        if auxiliary is None:
            raise ValueError("auxiliary signs are required")
        values = np.asarray(values)
        return values * auxiliary, np.zeros(values.shape[0])


def test_metropolis_tracks_target_and_proposal_work(rng) -> None:
    n_particles = 4
    target = TemperedPosterior(
        log_prior=lambda values: -np.sum(values**2, axis=1),
        log_likelihood=lambda values: -np.sum((values - 1.0) ** 2, axis=1),
    )
    step = MetropolisHastingsMutation(
        target,
        GaussianRandomWalk(scale=0.1),
        n_steps=2,
    )

    _, context, _ = step.apply(
        ParticleState(np.zeros((n_particles, 2))),
        StepContext(t=0, metadata={"temperature": 0.5}),
        rng,
    )

    assert context.evaluation_counts.target_evaluations == 12
    assert context.evaluation_counts.proposal_evaluations == 0


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

    updated, _, output = step(particle_state, context_factory(), rng)

    np.testing.assert_allclose(
        updated.particles, particle_state.particles + 1.0
    )
    np.testing.assert_allclose(updated.log_weights, particle_state.log_weights)
    assert output["acceptance"] == 1.0
    assert output["acceptance_by_step"] == (1.0, 1.0)
    assert output["steps"] == 2
    assert "mutation_chains" not in updated.metadata
    assert output["chain"].shape == (2, 4, 2)


def test_metropolis_keeps_forward_auxiliary_through_mutation(
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
        transform=SignFoldingTransform(),
        store_chains=True,
    )

    updated, _, output = step(particle_state, context_factory(), rng)

    signs = np.where(particle_state.particles < 0, -1.0, 1.0)
    expected = (np.abs(particle_state.particles) + 0.5) * signs
    np.testing.assert_allclose(updated.particles, expected)
    np.testing.assert_allclose(output["chain"][0], expected)


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

    updated, _, output = step(particle_state, context_factory(), rng)

    np.testing.assert_allclose(updated.particles, particle_state.particles)
    assert output["acceptance"] == 0.0


def test_metropolis_hastings_mutation_supports_kernel_factory(
    particle_state: ParticleState,
    context_factory,
    rng,
) -> None:
    context = context_factory(metadata={"proposal_scale": 0.2})
    factory_shapes = []

    def factory(particles, inner_context):
        factory_shapes.append(particles.shape)
        return GaussianRandomWalk(
            scale=inner_context.metadata["proposal_scale"]
        )

    step = MetropolisHastingsMutation(
        lambda particles, inner_context: np.zeros(particles.shape[0]),
        factory,
    )

    updated, context, _ = step(particle_state, context, rng)

    assert updated.particles.shape == particle_state.particles.shape
    assert factory_shapes == [particle_state.particles.shape]
