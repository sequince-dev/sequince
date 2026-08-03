from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from sequince.posterior import (
    TemperedPosterior,
    draw_additional_samples,
)
from sequince.proposals import FlowProposal
from sequince.result import SMCResult
from sequince.state import ParticleState
from sequince.target import AnnealedTarget, TargetEvaluation


def test_tempered_posterior_implements_annealed_target() -> None:
    target = TemperedPosterior(
        log_prior=lambda particles: np.zeros(particles.shape[0]),
        log_likelihood=lambda particles: np.zeros(particles.shape[0]),
    )

    assert isinstance(target, AnnealedTarget)


def test_targets_receive_distinct_automatic_evaluation_ids() -> None:
    first = TemperedPosterior(
        log_prior=lambda particles: np.zeros(particles.shape[0]),
        log_likelihood=lambda particles: np.zeros(particles.shape[0]),
    )
    second = TemperedPosterior(
        log_prior=first.log_prior,
        log_likelihood=first.log_likelihood,
    )

    assert len(first.evaluation_id) == 32
    assert first.evaluation_id != second.evaluation_id


def test_target_replaces_evaluation_from_another_target(
    context_factory,
    caplog,
) -> None:
    calls = {"first": 0, "second": 0}

    def likelihood(name, value):
        def evaluate(particles, context):
            calls[name] += 1
            return np.full(particles.shape[0], value)

        return evaluate

    first = TemperedPosterior(
        log_prior=lambda particles: np.zeros(particles.shape[0]),
        log_likelihood=likelihood("first", 1.0),
    )
    second = TemperedPosterior(
        log_prior=lambda particles: np.zeros(particles.shape[0]),
        log_likelihood=likelihood("second", 2.0),
    )
    particles = np.zeros((3, 1))
    context = context_factory(metadata={"temperature": 0.5})

    first_evaluation = first.evaluate(particles, context)
    with caplog.at_level("DEBUG", logger="sequince.target"):
        second_evaluation = second.prepare_evaluation(
            particles,
            context,
            evaluation=first_evaluation,
        )

    assert calls == {"first": 1, "second": 1}
    assert second_evaluation.target_id == second.evaluation_id
    np.testing.assert_allclose(second_evaluation["log_likelihood"], 2.0)
    assert "Discarding evaluation from target" in caplog.text
    assert first.evaluation_id in caplog.text
    assert second.evaluation_id in caplog.text
    with pytest.raises(ValueError, match="different target"):
        second.log_prob(
            particles,
            context,
            evaluation=first_evaluation,
        )


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

    evaluation = posterior.evaluate(particles, context)
    values = posterior.log_prob(particles, context, evaluation=evaluation)

    expected = 0.75 * np.array([0.0, -1.0]) + 0.25 * np.array([1.0, 1.0])
    assert np.allclose(values, expected)


def test_target_evaluation_refreshes_only_requested_components(
    context_factory,
) -> None:
    calls = {"prior": 0, "likelihood": 0, "proposal": 0}

    def component(name, value):
        def evaluate(particles, context):
            calls[name] += 1
            return np.full(particles.shape[0], value)

        return evaluate

    posterior = TemperedPosterior(
        log_prior=component("prior", -1.0),
        log_likelihood=component("likelihood", -2.0),
        log_proposal=component("proposal", -3.0),
    )
    particles = np.zeros((4, 2))
    context = context_factory(metadata={"temperature": 0.5})

    evaluation = posterior.evaluate(particles, context)
    refreshed = posterior.refresh(
        particles,
        evaluation,
        context,
        components={"log_likelihood"},
    )
    posterior.log_increment(particles, context, evaluation=refreshed)
    posterior.log_prob(particles, context, evaluation=refreshed)

    assert calls == {"prior": 1, "likelihood": 2, "proposal": 1}


def test_tempered_posterior_evaluates_directly_from_particles(
    context_factory,
) -> None:
    calls = {"prior": 0, "likelihood": 0}

    def log_prior(particles, context):
        calls["prior"] += 1
        return np.full(particles.shape[0], -1.0)

    def log_likelihood(particles, context):
        calls["likelihood"] += 1
        return np.full(particles.shape[0], -2.0)

    posterior = TemperedPosterior(log_prior, log_likelihood)
    particles = np.zeros((3, 1))
    context = context_factory(metadata={"temperature": 0.5})

    values = posterior.log_prob(particles, context)

    np.testing.assert_allclose(values, -2.0)
    assert calls == {"prior": 1, "likelihood": 1}

    calls.update(prior=0, likelihood=0)
    increments = posterior.log_increment(particles, context)

    np.testing.assert_allclose(increments, -2.0)
    assert calls == {"prior": 1, "likelihood": 1}


def test_target_evaluation_selects_and_serializes() -> None:
    current = TargetEvaluation(
        components={
            "log_prior": np.array([1.0, 2.0]),
            "log_likelihood": np.array([3.0, 4.0]),
            "score": np.array([5.0, 6.0]),
        },
        target_id="target-a",
    )
    proposed = TargetEvaluation(
        components={
            "log_prior": np.array([10.0, 20.0]),
            "log_likelihood": np.array([30.0, 40.0]),
            "score": np.array([50.0, 60.0]),
        },
        target_id="target-a",
    )

    selected = current.select(proposed, np.array([True, False]))
    restored = TargetEvaluation.from_dict(selected.to_dict())

    np.testing.assert_allclose(restored["log_prior"], [10.0, 2.0])
    np.testing.assert_allclose(restored["log_likelihood"], [30.0, 4.0])
    np.testing.assert_allclose(restored["score"], [50.0, 6.0])
    assert restored.target_id == "target-a"


def test_target_evaluation_rejects_combining_different_targets() -> None:
    current = TargetEvaluation(
        components={"value": np.array([1.0, 2.0])},
        target_id="target-a",
    )
    proposed = TargetEvaluation(
        components={"value": np.array([3.0, 4.0])},
        target_id="target-b",
    )

    with pytest.raises(ValueError, match="different targets"):
        current.select(proposed, np.array([True, False]))
    with pytest.raises(ValueError, match="different targets"):
        TargetEvaluation.concatenate([current, proposed])


def test_target_evaluation_rejects_different_particle_count(
    context_factory,
) -> None:
    posterior = TemperedPosterior(
        log_prior=lambda particles, context: np.zeros(particles.shape[0]),
        log_likelihood=lambda particles, context: np.zeros(particles.shape[0]),
    )
    evaluation = posterior.evaluate(
        np.zeros((2, 1)),
        context_factory(),
    )
    particles = np.zeros((3, 1))
    context = context_factory(metadata={"temperature": 0.5})

    with pytest.raises(ValueError, match="2 entries.*3 particles"):
        posterior.log_prob(particles, context, evaluation=evaluation)
    with pytest.raises(ValueError, match="2 entries.*3 particles"):
        posterior.log_increment(particles, context, evaluation=evaluation)
    with pytest.raises(ValueError, match="2 entries.*3 particles"):
        posterior.refresh(
            particles,
            evaluation,
            context,
            components={"log_likelihood"},
        )


def test_flow_proposal_delegates_sampling_and_density(context_factory) -> None:
    class DummyFlow:
        def __init__(self, shift: float = 0.75) -> None:
            self.xp: Any = np
            self.shift = shift
            self.fitted_particles: Any = None

        def fit(self, particles, context=None, **kwargs: Any) -> "DummyFlow":
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

        def state_dict(self):
            return {"fitted_particles": self.fitted_particles}

        def load_state_dict(self, state):
            self.fitted_particles = np.asarray(state["fitted_particles"])

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
    restored_proposal = FlowProposal(DummyFlow())
    restored_proposal.load_state_dict(proposal.state_dict())
    np.testing.assert_allclose(
        getattr(restored_proposal.flow, "fitted_particles"), training_particles
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


def test_draw_additional_samples_supports_metropolis_hastings(
    context_factory,
    rng,
) -> None:
    particles = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=float)
    state = ParticleState(
        particles=particles,
        metadata={"temperature": 1.0},
    )
    result = SMCResult([state], context_factory(metadata={"temperature": 1.0}))
    posterior = TemperedPosterior(
        log_prior=lambda values, context: np.zeros(values.shape[0]),
        log_likelihood=lambda values, context: np.zeros(values.shape[0]),
    )

    updated_state, updated_result = draw_additional_samples(
        result=result,
        rng=rng,
        n_particles=8,
        mutation="metropolis-hastings",
        mutation_n_steps=2,
        mutation_kwargs={
            "kernel": "gaussian-random-walk",
            "kernel_kwargs": {"scale": 0.1},
        },
        particle_transform=None,
        posterior=posterior,
    )

    assert updated_state.particles.shape == (8, 1)
    assert updated_state.metadata["metropolis_hastings_steps"] == 2
    assert updated_result.final_state is updated_state
