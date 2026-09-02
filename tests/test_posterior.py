from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from sequince.array_api import get_device, namespace, to_numpy
from sequince.posterior import (
    TemperedPosterior,
    draw_additional_samples,
)
from sequince.proposals import FlowProposal
from sequince.result import SMCResult
from sequince.state import ParticleState
from sequince.steps.context import StepContext
from sequince.target import (
    EvaluationCounts,
    TargetEvaluation,
    TemperedTarget,
)


def test_tempered_posterior_implements_tempered_target() -> None:
    target = TemperedPosterior(
        log_prior=lambda particles: np.zeros(particles.shape[0]),
        log_likelihood=lambda particles: np.zeros(particles.shape[0]),
    )

    assert isinstance(target, TemperedTarget)


def test_tempered_posterior_reports_proposal_component_evaluations() -> None:
    particles = np.zeros((5, 2))
    context = StepContext(t=0, metadata={"temperature": 0.5})
    target = TemperedPosterior(
        log_prior=lambda values: np.zeros(values.shape[0]),
        log_likelihood=lambda values: np.ones(values.shape[0]),
        log_proposal=lambda values: np.full(values.shape[0], -1.0),
    )

    evaluation, counts = target.prepare_evaluation(
        particles, context, return_counts=True
    )

    assert counts == EvaluationCounts(
        log_target_evaluations=5,
        log_proposal_evaluations=5,
    )

    cached, cached_counts = target.prepare_evaluation(
        particles,
        context,
        evaluation=evaluation,
        return_counts=True,
    )
    assert cached is evaluation
    assert cached_counts == EvaluationCounts.zeros(xp=np)

    _, refreshed_counts = target.prepare_evaluation(
        particles,
        context,
        evaluation=evaluation,
        refresh_components={"log_likelihood"},
        return_counts=True,
    )
    assert refreshed_counts == EvaluationCounts(log_target_evaluations=5)


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
    context = context_factory(metadata={"beta": 0.5})

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


def test_annealed_posterior_accepts_proposal_distribution_object(
    context_factory,
    analytic_proposal_factory,
) -> None:
    posterior = TemperedPosterior(
        log_prior=lambda particles, context: np.zeros(particles.shape[0]),
        log_likelihood=lambda particles, context: np.ones(particles.shape[0]),
        log_proposal=analytic_proposal_factory(n_dim=1, quadratic_scale=0.25),
    )
    particles = np.array([[0.0], [2.0]], dtype=float)
    context = context_factory(metadata={"beta": 0.25})

    evaluation = posterior.evaluate(particles, context)
    values = posterior.log_prob(particles, context, evaluation=evaluation)

    expected = 0.75 * np.array([0.0, -1.0]) + 0.25 * np.array([1.0, 1.0])
    assert np.allclose(values, expected)


def test_tempered_posterior_uses_configured_proposal_override_key(
    context_factory,
) -> None:
    posterior = TemperedPosterior(
        log_prior=lambda particles: np.zeros(particles.shape[0]),
        log_likelihood=lambda particles: np.zeros(particles.shape[0]),
        log_proposal=lambda particles: np.full(particles.shape[0], -1.0),
        log_proposal_override_key="custom_proposal_override",
    )
    particles = np.zeros((3, 1))
    context = context_factory(
        metadata={
            "custom_proposal_override": lambda particles: np.full(
                particles.shape[0], -2.0
            )
        }
    )

    evaluation = posterior.evaluate(particles, context)

    np.testing.assert_allclose(evaluation["log_proposal"], -2.0)
    np.testing.assert_allclose(
        posterior.log_initial(particles, context, evaluation=evaluation), -2.0
    )


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
    context = context_factory(metadata={"beta": 0.5})

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
    context = context_factory(metadata={"beta": 0.5})

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
        evaluation_key="stage-a",
    )
    proposed = TargetEvaluation(
        components={
            "log_prior": np.array([10.0, 20.0]),
            "log_likelihood": np.array([30.0, 40.0]),
            "score": np.array([50.0, 60.0]),
        },
        target_id="target-a",
        evaluation_key="stage-a",
    )

    selected = current.select(proposed, np.array([True, False]))
    restored = TargetEvaluation.from_dict(selected.to_dict())

    np.testing.assert_allclose(restored["log_prior"], [10.0, 2.0])
    np.testing.assert_allclose(restored["log_likelihood"], [30.0, 4.0])
    np.testing.assert_allclose(restored["score"], [50.0, 6.0])
    assert restored.target_id == "target-a"
    assert restored.evaluation_key == "stage-a"


def test_target_evaluation_replaces_rows_across_backends(
    backend,
    backend_factory,
) -> None:
    case = backend_factory(backend)
    xp = namespace(case.particles)
    indices = xp.asarray(
        [3, 1],
        dtype=xp.int32,
        device=get_device(case.particles),
    )
    scores = xp.arange(
        case.particles.shape[0],
        dtype=case.particles.dtype,
        device=get_device(case.particles),
    )
    current = TargetEvaluation(
        components={"vectors": case.particles, "scores": scores},
        target_id="target-a",
        evaluation_key="stage-a",
    )
    replacement = TargetEvaluation(
        components={
            "vectors": case.particles[indices] + 10,
            "scores": scores[indices] + 10,
        },
        target_id="target-a",
        evaluation_key="stage-a",
    )
    original = to_numpy(case.particles).copy()

    updated = current.replace(indices, replacement)

    expected = original.copy()
    expected[[3, 1]] += 10
    np.testing.assert_array_equal(to_numpy(updated["vectors"]), expected)
    np.testing.assert_array_equal(to_numpy(current["vectors"]), original)
    np.testing.assert_array_equal(
        to_numpy(updated["vectors"])[[0, 2]], original[[0, 2]]
    )
    assert updated["vectors"].shape == case.particles.shape
    assert updated["vectors"].dtype == case.particles.dtype
    assert get_device(updated["vectors"]) == get_device(case.particles)
    assert case.matches_namespace(updated["vectors"])
    assert updated.target_id == current.target_id
    assert updated.evaluation_key == current.evaluation_key


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
        current.replace(np.array([0, 1]), proposed)
    with pytest.raises(ValueError, match="different targets"):
        TargetEvaluation.concatenate([current, proposed])


def test_target_evaluation_rejects_combining_different_stages() -> None:
    current = TargetEvaluation(
        components={"value": np.array([1.0, 2.0])},
        target_id="target-a",
        evaluation_key="stage-a",
    )
    proposed = TargetEvaluation(
        components={"value": np.array([3.0, 4.0])},
        target_id="target-a",
        evaluation_key="stage-b",
    )

    with pytest.raises(ValueError, match="different target stages"):
        current.select(proposed, np.array([True, False]))
    with pytest.raises(ValueError, match="different target stages"):
        current.replace(np.array([0, 1]), proposed)
    with pytest.raises(ValueError, match="different target stages"):
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
    context = context_factory(metadata={"beta": 0.5})

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


def test_flow_proposal_delegates_sampling_and_density(
    context_factory, shift_flow_factory
) -> None:
    training_particles = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=float)
    proposal = FlowProposal(shift_flow_factory(n_dim=2, shift=0.75)).fit(
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
    restored_proposal = FlowProposal(shift_flow_factory(n_dim=2, shift=0.75))
    restored_proposal.load_state_dict(proposal.state_dict())
    np.testing.assert_allclose(
        getattr(restored_proposal.flow, "fitted_particles"), training_particles
    )


def test_flow_proposal_passes_configured_fit_kwargs(
    context_factory, shift_flow_factory, rng
) -> None:
    flow = shift_flow_factory(n_dim=2, shift=0.75)
    proposal = FlowProposal(flow, fit_kwargs={"marker": "configured"})

    proposal.fit(np.ones((3, 2)), context=context_factory(), rng=rng)

    assert getattr(proposal.flow, "fit_kwargs") == {"marker": "configured"}
    assert getattr(proposal.flow, "fit_rng") is rng


def test_flow_proposal_converts_joint_sample_output_to_requested_namespace(
    context_factory, shift_flow_factory
) -> None:
    torch = pytest.importorskip("torch")
    proposal = FlowProposal(shift_flow_factory(n_dim=2, shift=0.75))

    particles, log_prob = proposal.sample_and_log_prob(
        3,
        context=context_factory(),
        xp=torch,
    )

    assert isinstance(particles, torch.Tensor)
    assert isinstance(log_prob, torch.Tensor)


def test_draw_additional_samples_respects_terminal_weights(
    context_factory,
    rng,
) -> None:
    particles = np.array([[0.0], [1.0], [2.0]], dtype=float)
    state = ParticleState(
        particles=particles,
        log_weights=np.array([0.0, -1.0e9, -1.0e9]),
        metadata={"beta": 1.0},
    )
    result = SMCResult([state], context_factory(metadata={"beta": 1.0}))

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
        metadata={"beta": 1.0},
    )
    result = SMCResult([state], context_factory(metadata={"beta": 1.0}))
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
    assert updated_result.outputs[-1][0]["steps"] == 2
    assert updated_result.final_state is updated_state
