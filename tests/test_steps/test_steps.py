"""Tests for composable SMC steps."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pytest
from orng import RandomGenerator

from sequince.posterior import TemperedPosterior
from sequince.state import ParticleState
from sequince.stats import (
    cess_ratio_from_log_weights_and_increments,
    ess_ratio_from_log_weights,
)
from sequince.steps import (
    AdaptiveGaussianProposal,
    ContinueWhile,
    MutateIfResampled,
)
from sequince.steps.base import SMCStep, StepContext
from sequince.steps.control import PeriodicStep
from sequince.steps.weighting import (
    AdaptiveTemperedWeighting,
    FixedTemperedWeighting,
    IncrementalWeighting,
    LogLikelihoodWeighting,
    TemperedWeighting,
    _apply_log_weight_increment,
    evolving_target_ess_ratio,
)
from sequince.target import SequentialTarget, Target, TargetEvaluation

if TYPE_CHECKING:
    from orng import RandomGenerator


def test_step_context_request_stop(step_context: StepContext) -> None:
    stopped = step_context.request_stop()
    assert not step_context.stop_requested
    assert stopped.stop_requested
    assert not stopped.stop_immediately


def test_step_context_request_stop_immediately(
    step_context: StepContext,
) -> None:
    stopped = step_context.request_stop_immediately()
    assert stopped.stop_requested
    assert stopped.stop_immediately


def test_step_context_is_frozen_and_updates_metadata_copy_on_write() -> None:
    shared = {"data": object()}
    unchanged = np.arange(3)
    context = StepContext(
        t=0,
        shared=shared,
        metadata={"unchanged": unchanged, "replaced": 1},
    )

    updated = context.with_metadata(replaced=2, added=3)

    with pytest.raises(FrozenInstanceError):
        setattr(context, "t", 1)
    assert updated is not context
    assert updated.shared is shared
    assert updated.metadata["unchanged"] is unchanged
    assert updated.metadata["replaced"] == 2
    assert updated.metadata["added"] == 3
    assert context.metadata == {"unchanged": unchanged, "replaced": 1}


def test_log_likelihood_weighting_updates_log_weights(
    particle_state: ParticleState, context_factory, rng
) -> None:
    def log_like(
        particles: np.ndarray,
        context: StepContext,
    ) -> np.ndarray:
        return np.sum(particles, axis=1)

    context = context_factory(t=3, observation=5)
    step = LogLikelihoodWeighting(log_like)
    updated, context, output = step(particle_state, context, rng)

    increments = np.sum(particle_state.particles, axis=1)
    expected = particle_state.log_weights + increments
    expected_log_norm = np.logaddexp.reduce(expected)
    expected -= np.log(np.sum(np.exp(expected)))
    assert np.allclose(updated.log_weights, expected)
    assert updated.log_evidence == pytest.approx(
        particle_state.log_evidence + expected_log_norm
    )
    np.testing.assert_allclose(output["log_weight_increment"], increments)
    assert context.evaluation_counts.log_increment_evaluations == len(
        particle_state.particles
    )


def test_apply_log_weight_increment_accepts_a_scalar() -> None:
    state = ParticleState(
        particles=np.arange(6.0).reshape(3, 2),
        log_weights=np.log(np.array([0.2, 0.3, 0.5])),
        log_normalizing_constant=1.25,
    )

    updated, log_norm = _apply_log_weight_increment(state, 0.7)

    np.testing.assert_allclose(updated.log_weights, state.log_weights)
    assert log_norm == pytest.approx(0.7)
    assert updated.log_evidence == pytest.approx(1.95)


def test_apply_log_weight_increment_normalizes_vector() -> None:
    state = ParticleState(
        particles=np.arange(6.0).reshape(3, 2),
        log_weights=np.log(np.array([0.2, 0.3, 0.5])),
        log_normalizing_constant=-0.4,
    )
    increment = np.array([-0.3, 0.2, 0.8])
    unnormalized = state.log_weights + increment
    expected_log_norm = np.logaddexp.reduce(unnormalized)

    updated, log_norm = _apply_log_weight_increment(state, increment)

    np.testing.assert_allclose(
        updated.log_weights,
        unnormalized - expected_log_norm,
    )
    assert np.sum(updated.weights) == pytest.approx(1.0)
    assert log_norm == pytest.approx(expected_log_norm)
    assert updated.log_evidence == pytest.approx(-0.4 + expected_log_norm)


def test_apply_log_weight_increment_can_defer_normalization() -> None:
    state = ParticleState(
        particles=np.arange(6.0).reshape(3, 2),
        log_weights=np.array([-0.1, -0.2, -0.3]),
        log_normalizing_constant=0.6,
    )
    increment = np.array([0.4, -0.2, 0.1])

    updated, log_norm = _apply_log_weight_increment(
        state,
        increment,
        normalize=False,
    )

    np.testing.assert_allclose(
        updated.log_weights,
        state.log_weights + increment,
    )
    assert updated.log_evidence == pytest.approx(state.log_evidence)
    assert log_norm is None


def test_tempered_weighting_scales_log_likelihood(
    particle_state: ParticleState,
    context_factory,
    rng,
    unit_log_likelihood,
) -> None:
    def schedule(t: int, context: StepContext) -> float:
        return 0.5 + 0.1 * t

    context = context_factory(t=2, metadata={})
    step = TemperedWeighting(
        unit_log_likelihood, schedule, normalize_after=False
    )
    updated, context, output = step(particle_state, context, rng)

    expected_increment = (0.5 + 0.1 * 2) * np.ones_like(
        particle_state.log_weights
    )
    assert np.allclose(
        updated.log_weights, particle_state.log_weights + expected_increment
    )
    assert updated.metadata["beta"] == pytest.approx(0.7)
    assert updated.metadata["delta_beta"] == pytest.approx(0.7)
    assert context.metadata["beta"] == pytest.approx(0.7)
    assert context.metadata["delta_beta"] == pytest.approx(0.7)
    assert output["beta"] == pytest.approx(0.7)
    assert output["delta_beta"] == pytest.approx(0.7)
    assert context.evaluation_counts.log_increment_evaluations == len(
        particle_state.particles
    )


def test_fixed_tempered_weighting_updates_metadata_and_context(
    particle_state: ParticleState,
    context_factory,
    rng,
    unit_log_likelihood,
) -> None:
    context = context_factory(metadata={"beta": 0.0})
    step = FixedTemperedWeighting(
        unit_log_likelihood,
        beta_increment=0.25,
        normalize_after=False,
    )
    updated, context, _ = step(particle_state, context, rng)

    assert updated.metadata["beta"] == pytest.approx(0.25)
    assert context.metadata["beta"] == pytest.approx(0.25)
    assert context.metadata["delta_beta"] == pytest.approx(0.25)
    assert updated.metadata["delta_beta"] == pytest.approx(0.25)
    assert updated.metadata[
        "log_normalizing_constant_increment"
    ] == pytest.approx(0.25)
    assert updated.metadata["log_normalizing_constant_increment_variance"] == (
        pytest.approx(0.0)
    )
    assert updated.metadata["ess_before_resampling"] == pytest.approx(
        particle_state.effective_sample_size()
    )
    assert updated.metadata["ess_at_max_beta"] == pytest.approx(
        particle_state.effective_sample_size()
    )


def test_tempering_diagnostics_support_weighted_particles(
    particle_state: ParticleState, context_factory, rng
) -> None:
    normalized, _ = particle_state.normalize()
    log_increments = np.array([-1.0, 0.0, 0.5, 1.5])
    context = context_factory(metadata={"beta": 0.0, "max_beta": 1.0})
    step = FixedTemperedWeighting(
        lambda particles: log_increments,
        beta_increment=0.4,
    )

    updated, context, _ = step(normalized, context, rng)

    base_weights = normalized.weights
    scaled_ratios = np.exp(0.4 * log_increments - np.max(0.4 * log_increments))
    mean_ratio = np.sum(base_weights * scaled_ratios)
    expected_variance = (
        np.sum(base_weights**2 * (scaled_ratios - mean_ratio) ** 2)
        / mean_ratio**2
    )
    terminal_log_weights = normalized.log_weights + log_increments
    terminal_weights = np.exp(
        terminal_log_weights - np.logaddexp.reduce(terminal_log_weights)
    )
    expected_terminal_ess = 1.0 / np.sum(terminal_weights**2)

    assert updated.metadata[
        "log_normalizing_constant_increment_variance"
    ] == pytest.approx(expected_variance)
    assert updated.metadata["ess_at_max_beta"] == pytest.approx(
        expected_terminal_ess
    )


def test_tempered_weighting_reuses_attached_evaluation(
    particle_state: ParticleState, context_factory, rng
) -> None:
    calls = {"prior": 0, "likelihood": 0}

    def log_prior(particles, context):
        calls["prior"] += 1
        return np.zeros(particles.shape[0])

    def log_likelihood(particles, context):
        calls["likelihood"] += 1
        return np.ones(particles.shape[0])

    posterior = TemperedPosterior(log_prior, log_likelihood)
    context = context_factory(metadata={"beta": 0.0})
    evaluation = posterior.evaluate(particle_state.particles, context)
    state = particle_state.with_updates(evaluation=evaluation)
    step = FixedTemperedWeighting(
        posterior,
        beta_increment=0.25,
    )

    state, context, _ = step(state, context, rng)
    state, context, _ = step(state, context, rng)

    assert state.evaluation is not None
    assert calls == {"prior": 1, "likelihood": 1}
    assert context.evaluation_counts.log_target_evaluations == 0
    assert context.evaluation_counts.log_increment_evaluations == 2 * len(
        particle_state.particles
    )


def test_incremental_weighting_accepts_a_general_sequential_target(
    particle_state: ParticleState, context_factory, rng
) -> None:
    class LinearTarget(SequentialTarget):
        def evaluate(
            self,
            particles,
            context,
            *,
            components=None,
            evaluation=None,
        ):
            requested = (
                {"base", "log_increment"} if components is None else components
            )
            values = {} if evaluation is None else dict(evaluation.components)
            if "base" in requested:
                values["base"] = -np.sum(particles**2, axis=1)
            if "log_increment" in requested:
                values["log_increment"] = particles[:, 0]
            return TargetEvaluation(values)

        def log_increment(self, particles, context, *, evaluation=None):
            evaluation = evaluation or self.evaluate(particles, context)
            return evaluation["log_increment"]

        def log_prob(self, particles, context, *, evaluation=None):
            evaluation = evaluation or self.evaluate(particles, context)
            beta = float(context.metadata.get("beta", 0.0))
            return evaluation["base"] + beta * evaluation["log_increment"]

    context = context_factory(metadata={"beta": 0.0})
    initial_log_weights = particle_state.log_weights.copy()

    target = LinearTarget()
    invalid_tempered_target: Any = target
    with pytest.raises(TypeError, match="log_increment_fn must be callable"):
        FixedTemperedWeighting(
            invalid_tempered_target,
            beta_increment=0.25,
        )

    updated, context, _ = IncrementalWeighting(
        target,
        normalize_after=False,
    )(particle_state, context, rng)

    np.testing.assert_allclose(
        updated.log_weights,
        initial_log_weights + particle_state.particles[:, 0],
    )
    assert updated.log_evidence == pytest.approx(particle_state.log_evidence)
    assert updated.evaluation is not None
    assert updated.evaluation.components.keys() == {"log_increment"}
    assert context.evaluation_counts.log_increment_evaluations == len(
        particle_state.particles
    )


def test_incremental_weighting_rejects_nonsequential_target() -> None:
    class DensityTarget(Target):
        def evaluate(
            self,
            particles,
            context,
            *,
            components=None,
            evaluation=None,
        ):
            del components, evaluation
            return self.bind_evaluation(
                TargetEvaluation({"log_prob": -np.sum(particles**2, axis=1)}),
                context,
            )

        def log_prob(self, particles, context, *, evaluation=None):
            evaluation = evaluation or self.evaluate(particles, context)
            return evaluation["log_prob"]

    nonsequential_target: Any = DensityTarget()
    with pytest.raises(TypeError, match="log_increment_fn must be callable"):
        IncrementalWeighting(nonsequential_target)


def test_tempered_weighting_can_refresh_only_likelihood(
    particle_state: ParticleState, context_factory, rng
) -> None:
    calls = {"prior": 0, "likelihood": 0}

    def log_prior(particles, context):
        calls["prior"] += 1
        return np.zeros(particles.shape[0])

    def log_likelihood(particles, context):
        calls["likelihood"] += 1
        return np.full(particles.shape[0], context.t + 1.0)

    posterior = TemperedPosterior(log_prior, log_likelihood)
    initial_context = context_factory(t=0, metadata={"beta": 0.0})
    state = particle_state.with_updates(
        evaluation=posterior.evaluate(
            particle_state.particles,
            initial_context,
        )
    )
    step = FixedTemperedWeighting(
        posterior,
        beta_increment=0.25,
        refresh_components={"log_likelihood"},
    )

    updated, _, _ = step(
        state,
        context_factory(t=1, metadata={"beta": 0.0}),
        rng,
    )

    assert calls == {"prior": 1, "likelihood": 2}
    evaluation = updated.evaluation
    assert evaluation is not None
    np.testing.assert_allclose(evaluation["log_likelihood"], 2.0)


def test_evolving_target_ess_ratio_uses_clipped_beta_progress() -> None:
    schedule = evolving_target_ess_ratio(
        0.2,
        0.8,
        rate=2.0,
        max_beta=2.0,
    )

    assert schedule(-1.0) == pytest.approx(0.2)
    assert schedule(1.0) == pytest.approx(0.35)
    assert schedule(2.0) == pytest.approx(0.8)
    assert schedule(3.0) == pytest.approx(0.8)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"start": 0.0, "end": 0.8}, "start must be"),
        ({"start": 0.2, "end": 1.1}, "end must be"),
        ({"start": 0.2, "end": 0.8, "rate": 0.0}, "rate must be"),
        (
            {"start": 0.2, "end": 0.8, "max_beta": 0.0},
            "max_beta must be",
        ),
    ],
)
def test_evolving_target_ess_ratio_validates_arguments(
    kwargs: dict[str, float], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        evolving_target_ess_ratio(**kwargs)


def test_adaptive_tempered_weighting_reaches_max_beta(
    particle_state: ParticleState,
    context_factory,
    rng,
    unit_log_likelihood,
) -> None:
    step = AdaptiveTemperedWeighting(
        unit_log_likelihood,
        target_ess_ratio=0.8,
        max_beta=0.6,
        min_delta=0.05,
        max_bisection_iter=10,
    )

    state = particle_state.with_updates(metadata={"beta": 0.0})

    stop_step = None
    for t in range(20):
        context = context_factory(t=t)
        state, context, _ = step(state, context, rng)
        if context.stop_requested:
            stop_step = t
            break

    assert stop_step is not None, (
        "Adaptive tempering did not request stop within 20 iterations."
    )
    assert state.metadata["beta"] == pytest.approx(0.6, abs=1e-6)
    assert context.metadata["beta"] == pytest.approx(state.metadata["beta"])
    assert "delta_beta" in context.metadata
    assert context.stop_requested
    assert not context.stop_immediately


@pytest.mark.parametrize("criterion", ["ess", "cess"])
def test_adaptive_tempered_weighting_supports_evolving_target(
    particle_state: ParticleState,
    context_factory,
    rng,
    criterion: Literal["ess", "cess"],
) -> None:
    schedule_calls: list[tuple[float, int]] = []

    def log_like(particles: np.ndarray, context: StepContext) -> np.ndarray:
        return np.linspace(-2.0, 2.0, particles.shape[0])

    def target_schedule(beta: float, context: StepContext) -> float:
        schedule_calls.append((beta, context.t))
        return 0.95

    state = particle_state.with_updates(metadata={"beta": 0.0})
    fixed_context = context_factory(t=4)
    scheduled_context = context_factory(t=4)
    fixed_step = AdaptiveTemperedWeighting(
        log_like,
        target_ess_ratio=0.8,
        criterion=criterion,
        min_delta=1e-8,
        max_bisection_iter=60,
    )
    scheduled_step = AdaptiveTemperedWeighting(
        log_like,
        target_ess_ratio=target_schedule,
        criterion=criterion,
        min_delta=1e-8,
        max_bisection_iter=60,
    )

    _, fixed_context, _ = fixed_step(state, fixed_context, rng)
    updated, scheduled_context, _ = scheduled_step(
        state, scheduled_context, rng
    )

    assert schedule_calls == [(0.0, 4)]
    assert (
        scheduled_context.metadata["delta_beta"]
        < fixed_context.metadata["delta_beta"]
    )
    assert updated.metadata["target_ess_ratio"] == pytest.approx(0.95)
    assert scheduled_context.metadata["target_ess_ratio"] == pytest.approx(
        0.95
    )
    delta = scheduled_context.metadata["delta_beta"]
    increments = log_like(state.particles, scheduled_context)
    expected_ess_ratio = ess_ratio_from_log_weights(
        state.log_weights + delta * increments
    )
    expected_cess_ratio = cess_ratio_from_log_weights_and_increments(
        state.log_weights, delta * increments
    )
    for metadata in (scheduled_context.metadata, updated.metadata):
        assert metadata["delta_beta"] == pytest.approx(delta)
        assert metadata["ess_ratio"] == pytest.approx(expected_ess_ratio)
        assert metadata["cess_ratio"] == pytest.approx(expected_cess_ratio)


@pytest.mark.parametrize("invalid_target", [0.0, 1.1, np.nan])
def test_adaptive_tempered_weighting_validates_scheduled_target(
    particle_state: ParticleState,
    context_factory,
    rng,
    invalid_target: float,
) -> None:
    step = AdaptiveTemperedWeighting(
        lambda particles: np.arange(particles.shape[0], dtype=float),
        target_ess_ratio=lambda beta: invalid_target,
    )

    with pytest.raises(
        ValueError,
        match="target_ess_ratio must resolve to a value in",
    ):
        step(particle_state, context_factory(), rng)


def test_adaptive_tempered_weighting_no_op_when_max_reached(
    particle_state: ParticleState,
    context_factory,
    rng,
    unit_log_likelihood,
) -> None:
    step = AdaptiveTemperedWeighting(unit_log_likelihood, max_beta=1.0)
    state = particle_state.with_updates(metadata={"beta": 1.0})
    context = context_factory(metadata={"beta": 1.0})

    result_state, context, output = step(state, context, rng)

    assert np.array_equal(result_state.particles, state.particles)
    assert result_state.metadata["beta"] == 1.0
    assert result_state.metadata["delta_beta"] == 0.0
    assert result_state.metadata["ess_ratio"] == pytest.approx(
        ess_ratio_from_log_weights(state.log_weights)
    )
    assert result_state.metadata["cess_ratio"] == 1.0
    assert context.stop_requested
    assert context.stop_immediately
    assert context.metadata["beta"] == 1.0
    assert context.metadata["delta_beta"] == 0.0
    assert context.metadata["ess_ratio"] == pytest.approx(
        ess_ratio_from_log_weights(state.log_weights)
    )
    assert context.metadata["cess_ratio"] == 1.0
    assert output["target_ess_ratio"] == pytest.approx(0.7)
    np.testing.assert_array_equal(
        output["log_weight_increment"],
        np.zeros_like(state.log_weights),
    )


def test_terminal_adaptive_weighting_with_zero_support(rng) -> None:
    state = ParticleState(np.zeros((4, 1)))
    increments = np.array([-np.inf, 0.0, 0.0, 0.0])
    step = AdaptiveTemperedWeighting(lambda _: increments)
    with np.errstate(invalid="raise"):
        updated, context, output = step(
            state, StepContext(t=0, metadata={"beta": 1.0}), rng
        )
    np.testing.assert_allclose(updated.log_weights, state.log_weights)
    assert updated.log_normalizing_constant == pytest.approx(0.0)
    assert context.stop_immediately
    assert output["delta_beta"] == 0.0
    assert updated.metadata["ess_at_max_beta"] == pytest.approx(4.0)


def test_tempered_posterior_custom_increment_is_used_by_weighting(
    particle_state: ParticleState, context_factory, rng
) -> None:
    class CustomPosterior(TemperedPosterior):
        def log_increment(
            self,
            particles: np.ndarray,
            context: StepContext,
            *,
            evaluation=None,
        ) -> np.ndarray:
            return np.full(particles.shape[0], 2.0)

    posterior = CustomPosterior(
        log_prior=lambda particles, context: np.zeros(particles.shape[0]),
        log_likelihood=lambda particles, context: np.ones(particles.shape[0]),
        log_proposal_override_key="custom_proposal_override",
    )

    def proposal_override(particles):
        return np.zeros(particles.shape[0])

    context = context_factory(
        metadata={
            "beta": 0.0,
            "custom_proposal_override": proposal_override,
        }
    )
    step = FixedTemperedWeighting(
        posterior,
        beta_increment=0.25,
        normalize_after=False,
    )

    updated, context, _ = step(particle_state, context, rng)

    assert context.metadata["custom_proposal_override"] is proposal_override
    expected = particle_state.log_weights + 0.5
    assert np.allclose(updated.log_weights, expected)


@pytest.mark.parametrize("resampled", [False, True])
def test_mutate_if_resampled(
    particle_state: ParticleState,
    context_factory,
    rng,
    resampled: bool,
) -> None:
    class TrackingMutation(SMCStep):
        def apply(self, state, context, rng):
            context = context.with_metadata(mutation_called=True)
            return (
                state.with_updates(particles=state.particles + 1.0),
                context,
                "called",
            )

    context = context_factory(metadata={"resampled": resampled})
    step = MutateIfResampled(TrackingMutation())
    updated, context, output = step(particle_state, context, rng)

    assert "mutation_skipped" not in context.metadata
    assert "mutation_skipped" not in updated.metadata
    assert context.metadata.get("mutation_called", False) is resampled
    assert output.executed is resampled
    expected_particles = (
        particle_state.particles + 1.0
        if resampled
        else particle_state.particles
    )
    assert np.array_equal(updated.particles, expected_particles)


@pytest.mark.parametrize("resampled", [False, True])
def test_mutate_if_resampled_forwards_fixed_length_only_when_executed(
    particle_state: ParticleState,
    context_factory,
    rng,
    resampled: bool,
) -> None:
    class FixedMutation(SMCStep):
        def __init__(self):
            self.lengths = []

        def apply(self, state, context, rng):
            return self.apply_with_n_steps(state, context, rng, n_steps=1)

        def apply_with_n_steps(self, state, context, rng, *, n_steps):
            self.lengths.append(n_steps)
            return state, context, {"steps": n_steps}

    mutation = FixedMutation()
    step = MutateIfResampled(mutation)
    context = context_factory(metadata={"resampled": resampled})

    _, _, output = step.apply_with_n_steps(
        particle_state, context, rng, n_steps=5
    )

    assert mutation.lengths == ([5] if resampled else [])
    assert output.executed is resampled
    if resampled:
        assert output.child_output["steps"] == 5


def test_adapt_gaussian_proposal_updates_scale(
    particle_state: ParticleState,
    context_factory,
    rng,
) -> None:
    from sequince.steps import MutationMetricsOutput

    class Mutation(SMCStep):
        def apply(self, state, context, rng):
            assert context.metadata["proposal_scale"] == 0.5
            return (
                state,
                context,
                {"metrics": MutationMetricsOutput(1, 10, 6)},
            )

    state = particle_state
    context = context_factory(metadata={"waste_free_acceptance": 0.0})

    adapter = AdaptiveGaussianProposal(
        Mutation(),
        initial_scale=0.5,
        target_acceptance=0.3,
        adaptation_rate=0.2,
        scale_key="proposal_scale",
        min_scale=0.05,
        max_scale=5.0,
    )

    updated, context, _ = adapter(state, context, rng)

    assert updated is state
    assert context.metadata["proposal_scale"] == pytest.approx(
        0.5 * np.exp(0.06)
    )


def test_continue_while_overrides_stop_request(
    particle_state: ParticleState, context_factory, rng
) -> None:
    predicate_calls: list[tuple[float, int]] = []

    def predicate(state: ParticleState, context: StepContext) -> bool:
        beta = float(state.metadata.get("beta", 0.0))
        predicate_calls.append((beta, context.t))
        return beta < 1.0

    step = ContinueWhile(predicate)

    context = context_factory(t=0)
    context = context.request_stop_immediately()
    state = particle_state.with_updates(metadata={"beta": 0.5})

    result, context, _ = step(state, context, rng)

    assert result is state
    assert not context.stop_requested
    assert not context.stop_immediately

    context_late = context_factory(t=1)
    context_late = context_late.request_stop_immediately()
    late_state = particle_state.with_updates(metadata={"beta": 1.0})

    final_result, context_late, _ = step(late_state, context_late, rng)

    assert final_result is late_state
    assert context_late.stop_requested
    assert context_late.stop_immediately
    assert predicate_calls == [(0.5, 0), (1.0, 1)]


def test_step_every_n_iterations_applies_step_conditionally(
    particle_state: ParticleState, context_factory, rng
) -> None:
    applied_iterations: list[int] = []

    class RecordingStep(SMCStep):
        def apply(
            self,
            state: ParticleState,
            context: StepContext,
            rng: "RandomGenerator",
        ) -> tuple[ParticleState, StepContext, dict[str, object]]:
            applied_iterations.append(context.t)
            return state, context, {}

    recording_step = RecordingStep()
    step = PeriodicStep(recording_step, n=3, offset=1)

    for t in range(10):
        context = context_factory(t=t)
        step(particle_state, context, rng)

    assert applied_iterations == [1, 4, 7]


@pytest.mark.parametrize(
    "accepted,target,expected",
    [(0, 0.3, 0.1), (10, 0.3, 2.0), (10, None, 0.5)],
)
def test_proposal_adaptation_bounds_and_disabled(
    accepted, target, expected, particle_state, context_factory, rng
):
    from sequince.steps import MutationMetricsOutput

    output = {
        "metrics": MutationMetricsOutput(1, 10, accepted),
        "extra": object(),
    }

    class Mutation(SMCStep):
        def apply(self, state, context, rng):
            return state, context, output

    wrapper = AdaptiveGaussianProposal(
        Mutation(),
        initial_scale=0.5,
        target_acceptance=target,
        adaptation_rate=1e6,
        min_scale=0.1,
        max_scale=2.0,
    )
    _, context, result = wrapper(particle_state, context_factory(), rng)
    assert result is output
    assert context.metadata["proposal_scale"] == pytest.approx(expected)


def test_proposal_adaptation_requires_completed_metrics(
    particle_state, context_factory, rng
):
    class Mutation(SMCStep):
        def apply(self, state, context, rng):
            return state, context, {"metrics": None}

    with pytest.raises(TypeError, match="complete MutationMetricsOutput"):
        AdaptiveGaussianProposal(Mutation())(
            particle_state, context_factory(), rng
        )
