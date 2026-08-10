from __future__ import annotations

import math

import numpy as np
import pytest

from sequince import (
    TemperedPosterior,
    draw_initial_state,
    draw_mcmc_initial_state,
)
from sequince.steps import FixedTemperedWeighting
from sequince.steps.context import StepContext


class DeterministicProposal:
    def __init__(self, particles):
        self.particles = np.asarray(particles, dtype=float)
        self.sample_calls = 0
        self.log_prob_calls = 0

    def sample(self, n_samples, context=None):
        raise AssertionError(
            "draw_initial_state should use sample_and_log_prob"
        )

    def log_prob(self, particles, context=None):
        self.log_prob_calls += 1
        return np.zeros(len(particles))

    def sample_and_log_prob(self, n_samples, context=None):
        self.sample_calls += 1
        return self.particles.copy(), np.zeros(n_samples)


def test_draw_initial_state_retains_evaluation_without_recomputing_proposal():
    proposal = DeterministicProposal([[0.0], [1.0], [2.0]])
    posterior = TemperedPosterior(
        log_prior=lambda particles, context: -particles[:, 0],
        log_likelihood=lambda particles, context: -2 * particles[:, 0],
        log_proposal=proposal,
    )

    state = draw_initial_state(proposal, 3, posterior=posterior)

    assert proposal.sample_calls == 1
    assert proposal.log_prob_calls == 0
    assert state.evaluation is not None
    np.testing.assert_allclose(
        state.evaluation["log_prior"], [0.0, -1.0, -2.0]
    )
    np.testing.assert_allclose(
        state.evaluation["log_likelihood"], [0.0, -2.0, -4.0]
    )
    np.testing.assert_allclose(state.evaluation["log_proposal"], 0.0)


def test_draw_initial_state_retains_zero_density_points_for_evidence():
    proposal = DeterministicProposal([[0.0], [1.0]])
    posterior = TemperedPosterior(
        log_prior=lambda particles, context: np.zeros(len(particles)),
        log_likelihood=lambda particles, context: np.array([0.0, -np.inf]),
        log_proposal=proposal,
    )
    context = StepContext(t=0, metadata={"max_temperature": 1.0})
    state = draw_initial_state(
        proposal,
        2,
        posterior=posterior,
        context=context,
    )

    weighted, _, _ = FixedTemperedWeighting(
        posterior,
        temperature_increment=1.0,
    )(state, context, np.random.default_rng(123))

    assert len(state.particles) == 2
    assert state.evaluation is not None
    assert np.isneginf(state.evaluation["log_likelihood"][1])
    assert weighted.log_evidence == pytest.approx(-math.log(2.0))
    np.testing.assert_allclose(weighted.weights, [1.0, 0.0])


def test_draw_initial_state_rejects_invalid_proposal_output():
    proposal = DeterministicProposal([[0.0], [np.inf]])

    with pytest.raises(ValueError, match="non-finite particles"):
        draw_initial_state(proposal, 2)


def test_draw_mcmc_initial_state_refills_rejected_particles():
    class BatchedProposal(DeterministicProposal):
        def __init__(self):
            super().__init__([[-1.0], [0.0], [-2.0]])
            self.batches = [
                np.array([[-1.0], [0.0], [-2.0]]),
                np.array([[1.0], [2.0], [3.0]]),
            ]

        def sample_and_log_prob(self, n_samples, context=None):
            batch = self.batches[self.sample_calls]
            self.sample_calls += 1
            return batch.copy(), np.zeros(n_samples)

    proposal = BatchedProposal()

    state = draw_mcmc_initial_state(
        proposal,
        3,
        lambda particles: np.where(particles[:, 0] < 0.0, -np.inf, 0.0),
    )

    np.testing.assert_allclose(state.particles[:, 0], [0.0, 1.0, 2.0])
    assert proposal.sample_calls == 2


@pytest.mark.parametrize("invalid", [np.nan, np.inf])
def test_draw_mcmc_initial_state_raises_on_target_numerical_errors(invalid):
    proposal = DeterministicProposal([[0.0], [1.0]])

    with pytest.raises(ValueError, match=r"NaN or \+inf"):
        draw_mcmc_initial_state(
            proposal,
            2,
            lambda particles: np.array([0.0, invalid]),
        )


def test_draw_mcmc_initial_state_has_bounded_rejection():
    proposal = DeterministicProposal([[-1.0], [-2.0]])

    with pytest.raises(RuntimeError, match="after 2 proposal batches"):
        draw_mcmc_initial_state(
            proposal,
            2,
            lambda particles: np.full(len(particles), -np.inf),
            max_batches=2,
        )

    assert proposal.sample_calls == 2
