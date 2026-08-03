from __future__ import annotations

import numpy as np

from sequince.posterior import TemperedPosterior
from sequince.state import ParticleState
from sequince.steps.base import StepContext
from sequince.steps.mutation import MiniPCNMutation
from sequince.transforms import AffineTransform


def test_minipcn_mutation_runs_single_step(
    particle_state: ParticleState, context_factory, rng
) -> None:
    def log_prob(position: np.ndarray, context: StepContext) -> np.ndarray:
        return -0.5 * np.sum(position**2, axis=-1)

    context = context_factory(shared={"temperature": 1.0})
    step = MiniPCNMutation(log_prob, n_steps=1)
    mutated = step(particle_state, context, rng)

    assert mutated.particles.shape == particle_state.particles.shape
    assert "minipcn_acceptance" in mutated.metadata
    assert mutated.metadata["minipcn_steps"] == 1


def test_minipcn_mutation_evaluates_final_population(
    particle_state: ParticleState, context_factory, rng
) -> None:
    posterior = TemperedPosterior(
        log_prior=lambda particles: -np.sum(particles**2, axis=-1),
        log_likelihood=lambda particles: np.sum(particles, axis=-1),
    )
    context = context_factory(metadata={"temperature": 1.0})

    mutated = MiniPCNMutation(posterior, n_steps=1)(
        particle_state, context, rng
    )

    assert mutated.evaluation is not None
    expected = posterior.evaluate(mutated.particles, context)
    np.testing.assert_allclose(
        mutated.evaluation["log_prior"], expected["log_prior"]
    )
    np.testing.assert_allclose(
        mutated.evaluation["log_likelihood"], expected["log_likelihood"]
    )


def test_minipcn_mutation_with_transform(
    particle_state: ParticleState, context_factory, rng
) -> None:
    def log_prob(position: np.ndarray, context: StepContext) -> float:
        return float(-0.5 * np.sum(position**2))

    context = context_factory(shared={"temperature": 1.0})
    transform = AffineTransform(columns=[1])
    transform.fit(particle_state.particles)
    step = MiniPCNMutation(log_prob, n_steps=3, transform=transform)
    mutated = step(particle_state, context, rng)

    assert mutated.particles.shape == particle_state.particles.shape
    assert "minipcn_acceptance" in mutated.metadata
    assert mutated.metadata["minipcn_steps"] == 3
