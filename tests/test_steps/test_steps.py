"""Tests for composable SMC steps."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pytest
from orng import RandomGenerator

from sequince.array_api import namespace
from sequince.posterior import TemperedPosterior
from sequince.state import ParticleState
from sequince.stats import (
    cess_ratio_from_log_weights_and_increments,
    ess_ratio_from_log_weights,
)
from sequince.steps import (
    AdaptGaussianProposal,
    ContinueWhile,
    MutateIfResampled,
    WasteFreeResampleMove,
)
from sequince.steps.base import SMCStep, StepContext
from sequince.steps.control import PeriodicStep
from sequince.steps.mutation import GaussianRandomWalk
from sequince.steps.resampling import (
    MultinomialResampling,
    StratifiedResampling,
    SystematicResampling,
)
from sequince.steps.weighting import (
    AdaptiveTemperedWeighting,
    FixedTemperedWeighting,
    IncrementalWeighting,
    LogLikelihoodWeighting,
    TemperedWeighting,
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
    expected -= np.log(np.sum(np.exp(expected)))
    assert np.allclose(updated.log_weights, expected)
    np.testing.assert_allclose(output["log_weight_increment"], increments)


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
    assert output["temperature"] == pytest.approx(0.7)


def test_fixed_tempered_weighting_updates_metadata_and_context(
    particle_state: ParticleState,
    context_factory,
    rng,
    unit_log_likelihood,
) -> None:
    context = context_factory(metadata={"temperature": 0.0})
    step = FixedTemperedWeighting(
        unit_log_likelihood,
        temperature_increment=0.25,
        normalize_after=False,
    )
    updated, context, _ = step(particle_state, context, rng)

    assert updated.metadata["temperature"] == pytest.approx(0.25)
    assert context.metadata["temperature"] == pytest.approx(0.25)
    assert context.metadata["temperature_delta"] == pytest.approx(0.25)
    assert updated.metadata["temperature_delta"] == pytest.approx(0.25)
    assert updated.metadata["log_evidence_increment"] == pytest.approx(0.25)
    assert updated.metadata["log_evidence_increment_variance"] == (
        pytest.approx(0.0)
    )
    assert updated.metadata["ess_before_resampling"] == pytest.approx(
        particle_state.effective_sample_size()
    )
    assert updated.metadata["ess_at_max_temperature"] == pytest.approx(
        particle_state.effective_sample_size()
    )


def test_tempering_diagnostics_support_weighted_particles(
    particle_state: ParticleState, context_factory, rng
) -> None:
    normalized, _ = particle_state.normalize()
    log_increments = np.array([-1.0, 0.0, 0.5, 1.5])
    context = context_factory(
        metadata={"temperature": 0.0, "max_temperature": 1.0}
    )
    step = FixedTemperedWeighting(
        lambda particles: log_increments,
        temperature_increment=0.4,
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
        "log_evidence_increment_variance"
    ] == pytest.approx(expected_variance)
    assert updated.metadata["ess_at_max_temperature"] == pytest.approx(
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
    context = context_factory(metadata={"temperature": 0.0})
    evaluation = posterior.evaluate(particle_state.particles, context)
    state = particle_state.with_updates(evaluation=evaluation)
    step = FixedTemperedWeighting(
        posterior,
        temperature_increment=0.25,
    )

    state, context, _ = step(state, context, rng)
    state, context, _ = step(state, context, rng)

    assert state.evaluation is not None
    assert calls == {"prior": 1, "likelihood": 1}


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
                {"base", "increment"} if components is None else components
            )
            values = {} if evaluation is None else dict(evaluation.components)
            if "base" in requested:
                values["base"] = -np.sum(particles**2, axis=1)
            if "increment" in requested:
                values["increment"] = particles[:, 0]
            return TargetEvaluation(values)

        def log_increment(self, particles, context, *, evaluation=None):
            evaluation = evaluation or self.evaluate(particles, context)
            return evaluation["increment"]

        def log_prob(self, particles, context, *, evaluation=None):
            evaluation = evaluation or self.evaluate(particles, context)
            beta = float(context.metadata.get("temperature", 0.0))
            return evaluation["base"] + beta * evaluation["increment"]

    context = context_factory(metadata={"temperature": 0.0})
    initial_log_weights = particle_state.log_weights.copy()

    target = LinearTarget()
    invalid_tempered_target: Any = target
    with pytest.raises(TypeError, match="log_increment_fn must be callable"):
        FixedTemperedWeighting(
            invalid_tempered_target,
            temperature_increment=0.25,
        )

    updated, context, _ = IncrementalWeighting(
        target,
        normalize_after=False,
    )(particle_state, context, rng)

    np.testing.assert_allclose(
        updated.log_weights,
        initial_log_weights + particle_state.particles[:, 0],
    )
    assert updated.evaluation is not None
    assert updated.evaluation.components.keys() == {"base", "increment"}


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
    initial_context = context_factory(t=0, metadata={"temperature": 0.0})
    state = particle_state.with_updates(
        evaluation=posterior.evaluate(
            particle_state.particles,
            initial_context,
        )
    )
    step = FixedTemperedWeighting(
        posterior,
        temperature_increment=0.25,
        refresh_components={"log_likelihood"},
    )

    updated, _, _ = step(
        state,
        context_factory(t=1, metadata={"temperature": 0.0}),
        rng,
    )

    assert calls == {"prior": 1, "likelihood": 2}
    evaluation = updated.evaluation
    assert evaluation is not None
    np.testing.assert_allclose(evaluation["log_likelihood"], 2.0)


def test_evolving_target_ess_ratio_uses_clipped_temperature_progress() -> None:
    schedule = evolving_target_ess_ratio(
        0.2,
        0.8,
        rate=2.0,
        max_temperature=2.0,
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
            {"start": 0.2, "end": 0.8, "max_temperature": 0.0},
            "max_temperature must be",
        ),
    ],
)
def test_evolving_target_ess_ratio_validates_arguments(
    kwargs: dict[str, float], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        evolving_target_ess_ratio(**kwargs)


def test_adaptive_tempered_weighting_reaches_max_temperature(
    particle_state: ParticleState,
    context_factory,
    rng,
    unit_log_likelihood,
) -> None:
    step = AdaptiveTemperedWeighting(
        unit_log_likelihood,
        target_ess_ratio=0.8,
        max_temperature=0.6,
        min_delta=0.05,
        max_bisection_iter=10,
    )

    state = particle_state.with_updates(metadata={"temperature": 0.0})

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
    assert state.metadata["temperature"] == pytest.approx(0.6, abs=1e-6)
    assert context.metadata["temperature"] == pytest.approx(
        state.metadata["temperature"]
    )
    assert "temperature_delta" in context.metadata
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

    def target_schedule(temperature: float, context: StepContext) -> float:
        schedule_calls.append((temperature, context.t))
        return 0.95

    state = particle_state.with_updates(metadata={"temperature": 0.0})
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
        scheduled_context.metadata["temperature_delta"]
        < fixed_context.metadata["temperature_delta"]
    )
    assert updated.metadata["target_ess_ratio"] == pytest.approx(0.95)
    assert scheduled_context.metadata["target_ess_ratio"] == pytest.approx(
        0.95
    )
    delta = scheduled_context.metadata["temperature_delta"]
    increments = log_like(state.particles, scheduled_context)
    expected_ess_ratio = ess_ratio_from_log_weights(
        state.log_weights + delta * increments
    )
    expected_cess_ratio = cess_ratio_from_log_weights_and_increments(
        state.log_weights, delta * increments
    )
    for metadata in (scheduled_context.metadata, updated.metadata):
        assert metadata["temperature_delta"] == pytest.approx(delta)
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
        target_ess_ratio=lambda temperature: invalid_target,
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
    step = AdaptiveTemperedWeighting(unit_log_likelihood, max_temperature=1.0)
    state = particle_state.with_updates(metadata={"temperature": 1.0})
    context = context_factory(metadata={"temperature": 1.0})

    result_state, context, output = step(state, context, rng)

    assert np.array_equal(result_state.particles, state.particles)
    assert result_state.metadata["temperature"] == 1.0
    assert result_state.metadata["temperature_delta"] == 0.0
    assert result_state.metadata["ess_ratio"] == pytest.approx(
        ess_ratio_from_log_weights(state.log_weights)
    )
    assert result_state.metadata["cess_ratio"] == 1.0
    assert context.stop_requested
    assert context.stop_immediately
    assert context.metadata["temperature"] == 1.0
    assert context.metadata["temperature_delta"] == 0.0
    assert context.metadata["ess_ratio"] == pytest.approx(
        ess_ratio_from_log_weights(state.log_weights)
    )
    assert context.metadata["cess_ratio"] == 1.0
    assert output["target_ess_ratio"] == pytest.approx(0.7)
    np.testing.assert_array_equal(
        output["log_weight_increment"],
        np.zeros_like(state.log_weights),
    )


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
            "temperature": 0.0,
            "custom_proposal_override": proposal_override,
        }
    )
    step = FixedTemperedWeighting(
        posterior,
        temperature_increment=0.25,
        normalize_after=False,
    )

    updated, context, _ = step(particle_state, context, rng)

    assert context.metadata["custom_proposal_override"] is proposal_override
    expected = particle_state.log_weights + 0.5
    assert np.allclose(updated.log_weights, expected)


def test_systematic_resampling_skips_when_not_required(
    particle_state: ParticleState, context_factory, rng
) -> None:
    normalized, _ = particle_state.normalize()
    context = context_factory()
    step = SystematicResampling(ess_threshold=0.2, always_resample=False)
    result, context, output = step(normalized, context, rng)
    np.testing.assert_array_equal(result.particles, normalized.particles)
    np.testing.assert_array_equal(result.log_weights, normalized.log_weights)
    np.testing.assert_array_equal(
        output["ancestors"], np.arange(normalized.particles.shape[0])
    )
    assert output["resampled"] is False


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


def test_multinomial_resampling_resamples_particles(
    particle_state: ParticleState, context_factory, rng
) -> None:
    context = context_factory()
    step = MultinomialResampling(always_resample=True)
    resampled, context, output = step(particle_state, context, rng)

    assert resampled.particles.shape == particle_state.particles.shape
    assert np.allclose(
        resampled.log_weights,
        np.full_like(
            resampled.log_weights, -np.log(resampled.log_weights.size)
        ),
    )
    assert "ancestors" not in resampled.metadata
    assert "ancestors" in output
    assert output["resampled"] is True
    assert output["resampling_scheme"] == "multinomial"


@pytest.mark.parametrize(
    "step_type",
    [MultinomialResampling, StratifiedResampling, SystematicResampling],
)
@pytest.mark.parametrize(
    ("n_particles", "n_samples", "expected_size"),
    [(12, None, 12), (100, 25, 25), (20, 50, 50)],
)
def test_resampling_supports_different_output_population_sizes(
    step_type,
    n_particles: int,
    n_samples: int | None,
    expected_size: int,
    context_factory,
    rng,
) -> None:
    particles = np.column_stack(
        (np.arange(n_particles), -np.arange(n_particles))
    )
    evaluation = TargetEvaluation(
        components={"cached": np.arange(n_particles) + 1000}
    )
    state = ParticleState(
        particles=particles,
        log_weights=np.linspace(-2.0, 0.0, n_particles),
        evaluation=evaluation,
    )
    kwargs = {} if n_samples is None else {"n_samples": n_samples}

    updated, _, output = step_type(always_resample=True, **kwargs)(
        state, context_factory(), rng
    )

    ancestors = output["ancestors"]
    assert updated.particles.shape == (expected_size, 2)
    assert updated.log_weights.shape == (expected_size,)
    assert ancestors.shape == (expected_size,)
    np.testing.assert_array_equal(updated.particles, particles[ancestors])
    np.testing.assert_array_equal(updated.initial_ancestor_indices, ancestors)
    assert updated.evaluation is not None
    np.testing.assert_array_equal(
        updated.evaluation["cached"], evaluation["cached"][ancestors]
    )
    np.testing.assert_allclose(updated.log_weights, -np.log(expected_size))


@pytest.mark.parametrize(
    ("n_samples", "error"),
    [
        (0, ValueError),
        (-1, ValueError),
        (1.5, TypeError),
        ("2", TypeError),
        (True, TypeError),
    ],
)
def test_resampling_rejects_invalid_n_samples(n_samples, error) -> None:
    with pytest.raises(error, match="n_samples must be a positive integer"):
        SystematicResampling(n_samples=n_samples)


def test_skipped_resampling_outputs_identity_ancestors(
    particle_state: ParticleState, context_factory, rng
) -> None:
    state = particle_state
    step = MultinomialResampling(ess_threshold=None)

    updated, context, output = step(state, context_factory(), rng)

    np.testing.assert_array_equal(
        output["ancestors"], np.arange(state.particles.shape[0])
    )
    assert "ancestors" not in updated.metadata
    np.testing.assert_array_equal(updated.particles, state.particles)
    np.testing.assert_array_equal(updated.log_weights, state.log_weights)
    assert context.metadata["resampled"] is False
    assert output["resampled"] is False
    assert output["resampling_scheme"] == "multinomial"


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax", "cupy"])
@pytest.mark.parametrize(
    "step_type",
    [MultinomialResampling, StratifiedResampling, SystematicResampling],
)
def test_resampling_preserves_array_backend(
    backend: str,
    step_type,
    backend_factory,
    context_factory,
) -> None:
    case = backend_factory(backend)
    particles = case.particles
    backend_ns = namespace(particles)

    updated, _, output = step_type(always_resample=True, n_samples=6)(
        case.state(),
        context_factory(),
        case.rng(1234),
    )

    assert updated.particles.shape == (6, particles.shape[1])
    assert namespace(updated.particles) is backend_ns
    assert namespace(updated.log_weights) is backend_ns
    assert namespace(output["ancestors"]) is backend_ns
    assert output["resampling_scheme"] == step_type.resampling_scheme


def test_adapt_gaussian_proposal_updates_scale(
    particle_state: ParticleState,
    context_factory,
    rng,
) -> None:
    state = particle_state
    context = context_factory(metadata={"waste_free_acceptance": 0.6})

    adapter = AdaptGaussianProposal(
        initial_scale=0.5,
        target_acceptance=0.3,
        adaptation_rate=0.2,
        scale_key="proposal_scale",
        min_scale=0.05,
        max_scale=5.0,
    )

    updated, context, _ = adapter(state, context, rng)

    assert updated is state
    assert context.metadata["proposal_scale"] > 0.5


def test_adapt_gaussian_proposal_adds_covariance(
    particle_state: ParticleState,
    context_factory,
    rng,
) -> None:
    context = context_factory(metadata={})
    adapter = AdaptGaussianProposal(
        covariance_key="proposal_cov",
        mean_key="proposal_mean",
        covariance_jitter=1e-5,
    )

    _, context, _ = adapter(particle_state, context, rng)

    cov = context.metadata["proposal_cov"]
    mean = context.metadata["proposal_mean"]
    dims = particle_state.particles.shape[1]
    assert cov.shape == (dims, dims)
    assert np.all(np.isfinite(cov))
    assert mean.shape == (dims,)


def test_waste_free_resample_move_returns_uniform_weights(
    particle_state: ParticleState,
    context_factory,
    rng,
    standard_normal_log_prob,
) -> None:
    context = context_factory(shared={"temperature": 1.0})
    proposal = GaussianRandomWalk(scale=0.5)
    step = WasteFreeResampleMove(standard_normal_log_prob, proposal)

    updated, context, output = step(particle_state, context, rng)

    n = particle_state.particles.shape[0]
    assert np.allclose(updated.log_weights, -np.log(n))
    assert "waste_free_acceptance" not in updated.metadata
    assert output["proposal_repeats"] == 1
    total_selected = output["selected_parents"] + output["selected_proposals"]
    assert total_selected == n
    assert len(output["acceptance_by_round"]) == 1


def test_waste_free_resample_move_handles_one_dimensional_particles(
    rng,
) -> None:
    particles = np.array([-1.0, -0.2, 0.5, 1.3])
    n = particles.size
    state = ParticleState(
        particles=particles,
        log_weights=np.full(n, -np.log(n)),
        metadata={"temperature": 0.0},
    )

    def log_prob(particles: np.ndarray, context: StepContext) -> np.ndarray:
        pts = np.asarray(particles, dtype=float)
        return -0.5 * (pts**2)

    context = StepContext(t=0, shared={})
    proposal = GaussianRandomWalk(scale=0.3)
    step = WasteFreeResampleMove(
        log_prob,
        proposal,
        resample_ancestors=False,
    )

    updated, context, output = step(state, context, rng)
    assert updated.particles.shape == particles.shape
    assert np.allclose(updated.log_weights, -np.log(n))
    assert len(output["acceptance_by_round"]) == 1
    assert output["proposal_repeats"] == 1


def test_waste_free_resample_move_multiple_proposals(
    particle_state: ParticleState,
    context_factory,
    rng,
    standard_normal_log_prob,
) -> None:
    context = context_factory(shared={"temperature": 1.0})
    step = WasteFreeResampleMove(
        standard_normal_log_prob,
        GaussianRandomWalk(scale=0.4),
        proposals_per_particle=3,
    )

    updated, context, output = step(particle_state, context, rng)

    n = particle_state.particles.shape[0]
    assert np.allclose(updated.log_weights, -np.log(n))
    assert output["proposal_repeats"] == 3
    assert len(output["acceptance_by_round"]) == 3
    assert len(output["selected_proposals_by_round"]) == 3
    assert (
        sum(output["selected_proposals_by_round"])
        == output["selected_proposals"]
    )
    assert output["total_proposals"] == n * 3


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax", "cupy"])
def test_waste_free_resample_move_preserves_array_backend(
    backend: str,
    backend_factory,
    context_factory,
) -> None:
    case = backend_factory(backend)
    xp = case.xp
    particles = case.particles
    backend_ns = namespace(particles)

    def log_prob(values, context: StepContext):
        return -0.5 * xp.sum(values**2, axis=1)

    step = WasteFreeResampleMove(
        log_prob,
        GaussianRandomWalk(scale=0.4),
        proposals_per_particle=2,
    )
    updated, _, output = step(
        case.state(),
        context_factory(shared={"temperature": 1.0}),
        case.rng(1234),
    )

    assert namespace(updated.particles) is backend_ns
    assert namespace(updated.log_weights) is backend_ns
    assert namespace(output["ancestors"]) is backend_ns
    assert updated.particles.shape == particles.shape
    assert updated.log_weights.shape == case.log_weights.shape


def test_waste_free_resample_move_supports_dynamic_proposal(
    particle_state: ParticleState,
    context_factory,
    rng,
    standard_normal_log_prob,
) -> None:
    context = context_factory(metadata={"proposal_scale": 0.15})
    factory_scales = []

    def factory(parents: np.ndarray, ctx: StepContext) -> GaussianRandomWalk:
        scale = float(ctx.metadata.get("proposal_scale", 0.15))
        factory_scales.append(scale)
        return GaussianRandomWalk(scale=scale)

    adapter = AdaptGaussianProposal(
        scale_key="proposal_scale",
        initial_scale=0.15,
        target_acceptance=None,
    )

    # First update ensures the scale key is present.
    _, context, _ = adapter(particle_state, context, rng)

    step = WasteFreeResampleMove(
        standard_normal_log_prob,
        factory,
        proposals_per_particle=2,
    )

    _, context, _ = step(particle_state, context, rng)

    assert factory_scales
    assert all(
        scale == context.metadata["proposal_scale"] for scale in factory_scales
    )  # noqa: PLR2004


def test_continue_while_overrides_stop_request(
    particle_state: ParticleState, context_factory, rng
) -> None:
    predicate_calls: list[tuple[float, int]] = []

    def predicate(state: ParticleState, context: StepContext) -> bool:
        temperature = float(state.metadata.get("temperature", 0.0))
        predicate_calls.append((temperature, context.t))
        return temperature < 1.0

    step = ContinueWhile(predicate)

    context = context_factory(t=0)
    context = context.request_stop_immediately()
    state = particle_state.with_updates(metadata={"temperature": 0.5})

    result, context, _ = step(state, context, rng)

    assert result is state
    assert not context.stop_requested
    assert not context.stop_immediately

    context_late = context_factory(t=1)
    context_late = context_late.request_stop_immediately()
    late_state = particle_state.with_updates(metadata={"temperature": 1.0})

    final_result, context_late, _ = step(late_state, context_late, rng)

    assert final_result is late_state
    assert context_late.stop_requested
    assert context_late.stop_immediately
    assert predicate_calls == [(0.5, 0), (1.0, 1)]


def test_step_every_n_iterations_applies_step_conditionally(
    particle_state: ParticleState, context_factory, rng
) -> None:
    applied_iterations: list[int] = []

    class RecordingStep(AdaptGaussianProposal):
        def apply(
            self,
            state: ParticleState,
            context: StepContext,
            rng: "RandomGenerator",
        ) -> tuple[ParticleState, StepContext, dict[str, object]]:
            applied_iterations.append(context.t)
            return state, context, {}

    recording_step = RecordingStep(initial_scale=0.1, target_acceptance=None)
    step = PeriodicStep(recording_step, n=3, offset=1)

    for t in range(10):
        context = context_factory(t=t)
        step(particle_state, context, rng)

    assert applied_iterations == [1, 4, 7]
