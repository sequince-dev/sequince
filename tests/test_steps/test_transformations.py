from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from sequince.posterior import TemperedPosterior
from sequince.state import ParticleState
from sequince.steps import (
    ExpandDimensions,
    MetadataTransformation,
    SMCTransformation,
    TransformationStep,
)
from sequince.steps.base import StepContext

if TYPE_CHECKING:
    from orng import RandomGenerator


def test_dimensionality_expansion_appends_block(
    particle_state: ParticleState, context_factory, rng
) -> None:
    class AppendingTransformation(SMCTransformation):
        def forward(
            self,
            state: ParticleState,
            context: StepContext,
            rng: "RandomGenerator",
        ) -> ParticleState:
            n = state.particles.shape[0]
            new_block = np.full((n, 2), fill_value=0.5)
            particles = np.concatenate([state.particles, new_block], axis=1)
            return state.with_updates(particles=particles)

    context = context_factory(metadata={"temperature": 0.8})
    initial_state = particle_state.with_updates(
        metadata=dict(particle_state.metadata, temperature=0.8)
    )

    step = ExpandDimensions(
        AppendingTransformation(),
        expected_block_size=2,
        metadata_key="dims",
        added_dimensions_key="dims_added",
        dimension_delta_key="dims_delta",
    )
    expanded = step(initial_state, context, rng)

    assert expanded.particles.shape[1] == initial_state.particles.shape[1] + 2
    assert np.allclose(expanded.particles[:, -2:], 0.5)
    assert expanded.metadata["dims"] == expanded.particles.shape[1]
    assert expanded.metadata["dims_added"] == 2
    assert expanded.metadata["dims_delta"] == 2
    assert context.metadata["dims"] == expanded.particles.shape[1]
    assert context.metadata["dims_added"] == 2
    assert context.metadata["dims_delta"] == 2
    assert expanded.metadata["temperature"] == 0.8
    assert context.metadata["temperature"] == 0.8
    assert "temperature_delta" not in context.metadata


def test_transformation_step_composes_proposal_distribution_density(
    particle_state: ParticleState, context_factory, rng
) -> None:
    class QuadraticProposal:
        def log_prob(self, particles: np.ndarray, context=None) -> np.ndarray:
            values = np.atleast_2d(particles)
            return -0.5 * np.sum(values**2, axis=1)

        def sample(self, n_samples: int, context=None) -> np.ndarray:
            return np.zeros((n_samples, particle_state.particles.shape[1]))

        def sample_and_log_prob(
            self, n_samples: int, context=None
        ) -> tuple[np.ndarray, np.ndarray]:
            samples = self.sample(n_samples, context=context)
            return samples, self.log_prob(samples, context=context)

    class ScalingTransformation(SMCTransformation):
        def forward(
            self,
            state: ParticleState,
            context: StepContext,
            rng: "RandomGenerator",
        ) -> ParticleState:
            return state.with_updates(particles=2.0 * state.particles)

        def inverse(
            self, particles: np.ndarray, context: StepContext
        ) -> np.ndarray:
            return particles / 2.0

        def log_auxiliary_density(
            self, particles: np.ndarray, context: StepContext
        ) -> np.ndarray:
            return np.full(particles.shape[0], -0.25)

        def log_jacobian(
            self, particles: np.ndarray, context: StepContext
        ) -> np.ndarray:
            dimensions = particles.shape[1] if particles.ndim > 1 else 1
            return np.full(particles.shape[0], dimensions * np.log(2.0))

    posterior = TemperedPosterior(
        log_prior=lambda particles: np.zeros(particles.shape[0]),
        log_likelihood=lambda particles: np.zeros(particles.shape[0]),
        log_proposal=QuadraticProposal(),
    )
    context = context_factory()
    step = TransformationStep(ScalingTransformation(), posterior=posterior)

    once_transformed = step(particle_state, context, rng)
    twice_transformed = step(once_transformed, context, rng)
    transformed_log_proposal = context.metadata["log_proposal_override"]

    base_particles = twice_transformed.particles / 4.0
    expected = -0.5 * np.sum(
        np.atleast_2d(base_particles) ** 2, axis=1
    ) + 2.0 * (-0.25 - particle_state.particles.shape[1] * np.log(2.0))
    actual = transformed_log_proposal(twice_transformed.particles, context)

    assert np.allclose(actual, expected)


def test_metadata_transformation_updates_state_and_context(
    particle_state: ParticleState, context_factory, rng
) -> None:
    context = context_factory(metadata={"stage": 0})

    transformation = MetadataTransformation(
        state_metadata_fn=lambda state, context, rng: {
            "stage": context.metadata["stage"] + 1
        },
        context_metadata_fn=lambda state, context, rng: {
            "stage": context.metadata["stage"] + 1
        },
    )

    transformed = transformation.forward(particle_state, context, rng)

    assert transformed.metadata["stage"] == 1
    assert context.metadata["stage"] == 1
