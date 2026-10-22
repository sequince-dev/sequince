from __future__ import annotations

import numpy as np
import pytest

from sequince.posterior import TemperedPosterior
from sequince.state import ParticleState
from sequince.steps.context import StepContext
from sequince.steps.mutation import (
    CallableMutationKernel,
    GaussianRandomWalk,
    MetropolisHastingsMutation,
)
from sequince.transforms import WhiteningTransform


def test_fixed_length_override_is_per_call_and_keeps_default(
    particle_state: ParticleState,
    context_factory,
    rng,
) -> None:
    attempts = 0

    def proposal(particles, inner_rng):
        nonlocal attempts
        attempts += 1
        return particles

    step = MetropolisHastingsMutation(
        lambda particles: np.zeros(particles.shape[0]),
        CallableMutationKernel(proposal, symmetric=True),
        n_steps=2,
    )

    _, _, first = step.apply_with_n_steps(
        particle_state, context_factory(), rng, n_steps=4
    )
    _, _, second = step.apply_with_n_steps(
        particle_state, context_factory(), rng, n_steps=np.int64(1)
    )
    _, _, default = step(particle_state, context_factory(), rng)

    assert attempts == 7
    assert (first["steps"], second["steps"], default["steps"]) == (4, 1, 2)
    assert step._n_steps == 2  # noqa: SLF001


def test_fixed_length_zero_is_noop_without_kernel_calls(
    particle_state: ParticleState,
    context_factory,
    rng,
) -> None:
    attempts = 0

    def proposal(particles, inner_rng):
        nonlocal attempts
        attempts += 1
        return particles

    step = MetropolisHastingsMutation(
        lambda particles: np.zeros(particles.shape[0]),
        CallableMutationKernel(proposal, symmetric=True),
        store_chains=True,
    )
    context = context_factory()
    updated, returned_context, output = step.apply_with_n_steps(
        particle_state, context, rng, n_steps=0
    )

    assert updated is particle_state
    assert returned_context is context
    assert attempts == 0
    assert output["steps"] == 0
    assert output["chain"].shape == (0, *particle_state.particles.shape)


@pytest.mark.parametrize("value", [-1, 1.5, "2", True])
def test_fixed_length_rejects_invalid_values(
    particle_state: ParticleState,
    context_factory,
    rng,
    value,
) -> None:
    step = MetropolisHastingsMutation(
        lambda particles: np.zeros(particles.shape[0]),
        CallableMutationKernel(
            lambda particles, inner_rng: particles, symmetric=True
        ),
    )
    error = ValueError if value == -1 else TypeError
    with pytest.raises(error, match="n_steps"):
        step.apply_with_n_steps(
            particle_state, context_factory(), rng, n_steps=value
        )


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

    assert context.evaluation_counts.log_target_evaluations == 12
    assert context.evaluation_counts.log_likelihood_evaluations == 12
    assert context.evaluation_counts.log_proposal_evaluations == 0


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
    assert output["metrics"].acceptance == 1.0
    assert output["acceptance_by_step"] == (1.0, 1.0)
    assert output["steps"] == 2
    assert "mutation_chains" not in updated.metadata
    assert output["chain"].shape == (2, 4, 2)


def test_metropolis_keeps_forward_auxiliary_through_mutation(
    sign_folding_transform_factory,
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
        transform=sign_folding_transform_factory(),
        store_chains=True,
    )

    updated, _, output = step(particle_state, context_factory(), rng)

    signs = np.where(particle_state.particles < 0, -1.0, 1.0)
    expected = (np.abs(particle_state.particles) + 0.5) * signs
    np.testing.assert_allclose(updated.particles, expected)
    np.testing.assert_allclose(output["chain"][0], expected)


def test_metropolis_whitening_keeps_target_in_particle_coordinates(
    particle_state: ParticleState,
    context_factory,
    rng,
) -> None:
    evaluated = []

    def log_prob(particles, context):
        evaluated.append(np.asarray(particles).copy())
        return np.zeros(particles.shape[0])

    kernel = CallableMutationKernel(
        lambda particles, inner_rng: particles + 0.25,
        symmetric=True,
    )
    step = MetropolisHastingsMutation(
        log_prob,
        kernel,
        transform=WhiteningTransform(),
    )

    updated, _, output = step(particle_state, context_factory(), rng)

    fitted = WhiteningTransform().fit(particle_state.particles)
    latent, _, _ = fitted.forward(particle_state.particles)
    expected, _ = fitted.inverse(latent + 0.25)
    np.testing.assert_allclose(
        evaluated[0], particle_state.particles, atol=1e-12
    )
    np.testing.assert_allclose(evaluated[1], expected, atol=1e-12)
    np.testing.assert_allclose(updated.particles, expected, atol=1e-12)
    assert output["metrics"].acceptance == 1.0


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

    updated, context, output = step(particle_state, context_factory(), rng)

    np.testing.assert_allclose(updated.particles, particle_state.particles)
    assert output["metrics"].acceptance == 0.0
    assert context.evaluation_counts.log_proposal_evaluations == 0


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
