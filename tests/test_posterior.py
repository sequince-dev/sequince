from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from sequince.posterior import TemperedPosterior, draw_additional_samples
from sequince.proposals import FlowProposal
from sequince.result import SMCResult
from sequince.state import ParticleState


def test_tempered_posterior_accepts_proposal_distribution_object(
    context_factory,
) -> None:
    class QuadraticProposal:
        def log_prob(self, particles: np.ndarray, context=None) -> np.ndarray:
            values = np.asarray(particles, dtype=float)
            if values.ndim == 1:
                values = values[:, np.newaxis]
            return -0.25 * np.sum(values**2, axis=1)

        def sample(self, n_samples: int, context=None) -> np.ndarray:
            return np.zeros((n_samples, 1), dtype=float)

        def sample_and_log_prob(
            self, n_samples: int, context=None
        ) -> tuple[np.ndarray, np.ndarray]:
            samples = self.sample(n_samples, context=context)
            return samples, self.log_prob(samples, context=context)

    posterior = TemperedPosterior(
        log_prior=lambda particles, context: np.zeros(particles.shape[0]),
        log_likelihood=lambda particles, context: np.ones(particles.shape[0]),
        log_proposal=QuadraticProposal(),
    )
    particles = np.array([[0.0], [2.0]], dtype=float)
    context = context_factory(metadata={"temperature": 0.25})

    values = posterior.log_prob(particles, context)

    expected = 0.75 * np.array([0.0, -1.0]) + 0.25 * np.array([1.0, 1.0])
    assert np.allclose(values, expected)


def test_flow_proposal_delegates_sampling_and_density(context_factory) -> None:
    class DummyFlow:
        xp = np

        def __init__(self, shift=0.75):
            self.shift = shift
            self.fitted_particles = None

        def fit(self, particles, context=None):
            self.fitted_particles = np.asarray(particles, dtype=float).copy()
            return self

        def forward(self, particles, context=None):
            arr = np.asarray(particles, dtype=float)
            return arr + self.shift, np.zeros(arr.shape[0], dtype=float)

        def inverse(self, particles, context=None):
            arr = np.asarray(particles, dtype=float)
            return arr - self.shift, np.zeros(arr.shape[0], dtype=float)

        def log_prob(self, particles, context=None):
            arr = np.asarray(particles, dtype=float)
            if arr.ndim == 1:
                arr = arr[:, np.newaxis]
            return -0.5 * np.sum((arr - self.shift) ** 2, axis=1)

        def sample(self, num_samples: int = 1, context=None):
            return np.full((num_samples, 2), self.shift, dtype=float)

    training_particles = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=float)
    proposal = FlowProposal(DummyFlow()).fit(
        training_particles,
        context=context_factory(),
    )
    samples = proposal.sample(3)
    sampled_particles, sampled_log_prob = proposal.sample_and_log_prob(3)

    np.testing.assert_allclose(
        getattr(proposal.flow, "fitted_particles"), training_particles
    )
    np.testing.assert_allclose(samples, np.full((3, 2), 0.75))
    np.testing.assert_allclose(sampled_particles, samples)
    np.testing.assert_allclose(
        sampled_log_prob,
        proposal.log_prob(sampled_particles, context=context_factory()),
    )


def test_draw_additional_samples_respects_terminal_weights(
    context_factory,
    rng,
) -> None:
    particles = np.array([[0.0], [1.0], [2.0]], dtype=float)
    state = ParticleState(
        particles=particles,
        log_weights=np.array([0.0, -1.0e9, -1.0e9]),
        metadata={"temperature": 1.0},
    )
    result = SMCResult([state], context_factory(metadata={"temperature": 1.0}))

    updated_state, updated_result = draw_additional_samples(
        result=result,
        rng=rng,
        n_particles=8,
        mutation="none",
        posterior=MagicMock(spec=TemperedPosterior),
    )

    assert updated_state.particles.shape == (8, 1)
    assert np.all(updated_state.particles == 0.0)
    assert updated_state.log_weights.shape == (8,)
    assert np.allclose(updated_state.weights, np.full(8, 1.0 / 8.0))
    assert updated_result.final_state is updated_state
