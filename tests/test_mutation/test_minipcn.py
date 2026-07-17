from __future__ import annotations

import numpy as np

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
