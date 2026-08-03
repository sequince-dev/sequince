"""Tests for composable SMC steps."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pytest
from orng import RandomGenerator

from sequince._array_api import namespace
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
    LogLikelihoodWeighting,
    TemperedWeighting,
    evolving_target_ess_ratio,
)

if TYPE_CHECKING:
    from orng import RandomGenerator


def test_step_context_request_stop(step_context: StepContext) -> None:
    assert step_context.metadata["stop_requested"] is False
    assert step_context.metadata["stop_immediately"] is False
    step_context.request_stop()
    assert step_context.metadata["stop_requested"] is True
    assert step_context.metadata["stop_immediately"] is False


def test_step_context_request_stop_immediately(
    step_context: StepContext,
) -> None:
    step_context.request_stop_immediately()
    assert step_context.metadata["stop_requested"] is True
    assert step_context.metadata["stop_immediately"] is True


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
    updated = step(particle_state, context, rng)

    increments = np.sum(particle_state.particles, axis=1)
    expected = particle_state.log_weights + increments
    expected -= np.log(np.sum(np.exp(expected)))
    assert np.allclose(updated.log_weights, expected)


def test_tempered_weighting_scales_log_likelihood(
    particle_state: ParticleState, context_factory, rng
) -> None:
    def log_like(particles: np.ndarray, context: StepContext) -> np.ndarray:
        return np.ones(particles.shape[0])

    def schedule(t: int, context: StepContext) -> float:
        context.metadata.setdefault("temperatures", []).append(t)
        return 0.5 + 0.1 * t

    context = context_factory(t=2, metadata={})
    step = TemperedWeighting(log_like, schedule, normalize_after=False)
    updated = step(particle_state, context, rng)

    expected_increment = (0.5 + 0.1 * 2) * np.ones_like(
        particle_state.log_weights
    )
    assert np.allclose(
        updated.log_weights, particle_state.log_weights + expected_increment
    )
    assert context.metadata["temperatures"] == [2]


def test_fixed_tempered_weighting_updates_metadata_and_context(
    particle_state: ParticleState, context_factory, rng
) -> None:
    def log_like(particles: np.ndarray, context: StepContext) -> np.ndarray:
        return np.ones(particles.shape[0])

    context = context_factory(metadata={"temperature": 0.0})
    step = FixedTemperedWeighting(
        log_like, temperature_increment=0.25, normalize_after=False
    )
    updated = step(particle_state, context, rng)

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

    updated = step(normalized, context, rng)

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

    state = step(state, context, rng)
    state = step(state, context, rng)

    assert state.evaluation is not None
    assert calls == {"prior": 1, "likelihood": 1}


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

    updated = step(
        state,
        context_factory(t=1, metadata={"temperature": 0.0}),
        rng,
    )

    assert calls == {"prior": 1, "likelihood": 2}
    evaluation = updated.evaluation
    assert evaluation is not None
    np.testing.assert_allclose(evaluation.log_likelihood, 2.0)


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
    particle_state: ParticleState, context_factory, rng
) -> None:
    def log_like(particles: np.ndarray, context: StepContext) -> np.ndarray:
        return np.ones(particles.shape[0])

    step = AdaptiveTemperedWeighting(
        log_like,
        target_ess_ratio=0.8,
        max_temperature=0.6,
        min_delta=0.05,
        max_bisection_iter=10,
    )

    state = particle_state.with_updates(metadata={"temperature": 0.0})

    stop_step = None
    for t in range(20):
        context = context_factory(t=t)
        state = step(state, context, rng)
        if context.metadata["stop_requested"]:
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
    assert context.metadata["stop_requested"] is True
    assert context.metadata["stop_immediately"] is False


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

    fixed_step(state, fixed_context, rng)
    updated = scheduled_step(state, scheduled_context, rng)

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
    particle_state: ParticleState, context_factory, rng
) -> None:
    def log_like(particles: np.ndarray, context: StepContext) -> np.ndarray:
        return np.ones(particles.shape[0])

    step = AdaptiveTemperedWeighting(log_like, max_temperature=1.0)
    state = particle_state.with_updates(metadata={"temperature": 1.0})
    context = context_factory(metadata={"temperature": 1.0})

    result_state = step(state, context, rng)

    assert np.array_equal(result_state.particles, state.particles)
    assert result_state.metadata["temperature"] == 1.0
    assert result_state.metadata["temperature_delta"] == 0.0
    assert result_state.metadata["ess_ratio"] == pytest.approx(
        ess_ratio_from_log_weights(state.log_weights)
    )
    assert result_state.metadata["cess_ratio"] == 1.0
    assert context.metadata["stop_requested"] is True
    assert context.metadata["stop_immediately"] is True
    assert context.metadata["temperature"] == 1.0
    assert context.metadata["temperature_delta"] == 0.0
    assert context.metadata["ess_ratio"] == pytest.approx(
        ess_ratio_from_log_weights(state.log_weights)
    )
    assert context.metadata["cess_ratio"] == 1.0


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
            log_proposal_override=None,
        ) -> np.ndarray:
            context.metadata["custom_increment_called"] = True
            context.metadata["received_proposal_override"] = (
                log_proposal_override
            )
            return np.full(particles.shape[0], 2.0)

    posterior = CustomPosterior(
        log_prior=lambda particles, context: np.zeros(particles.shape[0]),
        log_likelihood=lambda particles, context: np.ones(particles.shape[0]),
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
        log_proposal_override_key="custom_proposal_override",
    )

    updated = step(particle_state, context, rng)

    assert context.metadata["custom_increment_called"] is True
    assert context.metadata["received_proposal_override"] is proposal_override
    expected = particle_state.log_weights + 0.5
    assert np.allclose(updated.log_weights, expected)


def test_systematic_resampling_skips_when_not_required(
    particle_state: ParticleState, context_factory, rng
) -> None:
    normalized, _ = particle_state.normalize()
    context = context_factory()
    step = SystematicResampling(ess_threshold=0.2, always_resample=False)
    result = step(normalized, context, rng)
    assert result is normalized  # no resampling expected


@pytest.mark.parametrize("resampled", [False, True])
def test_mutate_if_resampled(
    particle_state: ParticleState,
    context_factory,
    rng,
    resampled: bool,
) -> None:
    class TrackingMutation(SMCStep):
        def apply(self, state, context, rng):
            context.metadata["mutation_called"] = True
            return state.with_updates(particles=state.particles + 1.0)

    context = context_factory(metadata={"resampled": resampled})
    step = MutateIfResampled(TrackingMutation())
    updated = step(particle_state, context, rng)

    assert context.metadata["mutation_skipped"] is not resampled
    assert updated.metadata["mutation_skipped"] is not resampled
    assert context.metadata.get("mutation_called", False) is resampled
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
    resampled = step(particle_state, context, rng)

    assert resampled.particles.shape == particle_state.particles.shape
    assert np.allclose(
        resampled.log_weights,
        np.full_like(
            resampled.log_weights, -np.log(resampled.log_weights.size)
        ),
    )
    assert "ancestors" in resampled.metadata


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
    _, particles, log_weights, _ = backend_factory(backend)
    state = ParticleState(particles=particles, log_weights=log_weights)
    backend_ns = namespace(particles)

    updated = step_type(always_resample=True)(
        state,
        context_factory(),
        RandomGenerator(backend=backend, seed=1234),
    )

    assert namespace(updated.particles) is backend_ns
    assert namespace(updated.log_weights) is backend_ns
    assert namespace(updated.metadata["ancestors"]) is backend_ns


def test_adapt_gaussian_proposal_updates_scale(
    particle_state: ParticleState,
    context_factory,
    rng,
) -> None:
    metadata = dict(particle_state.metadata)
    metadata["waste_free_acceptance"] = 0.6
    state = particle_state.with_updates(metadata=metadata)
    context = context_factory(metadata=metadata)

    adapter = AdaptGaussianProposal(
        initial_scale=0.5,
        target_acceptance=0.3,
        adaptation_rate=0.2,
        scale_key="proposal_scale",
        min_scale=0.05,
        max_scale=5.0,
        scale_metadata_key="proposal_scale",
    )

    updated = adapter(state, context, rng)

    assert updated.metadata["proposal_scale"] > 0.5
    assert (
        context.metadata["proposal_scale"]
        == updated.metadata["proposal_scale"]
    )


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

    adapter(particle_state, context, rng)

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
) -> None:
    def log_prob(particles: np.ndarray, context: StepContext) -> np.ndarray:
        pts = np.asarray(particles, dtype=float)
        if pts.ndim == 1:
            pts = pts[:, np.newaxis]
        return -0.5 * np.sum(pts**2, axis=1)

    context = context_factory(shared={"temperature": 1.0})
    proposal = GaussianRandomWalk(scale=0.5)
    step = WasteFreeResampleMove(log_prob, proposal)

    updated = step(particle_state, context, rng)

    n = particle_state.particles.shape[0]
    assert np.allclose(updated.log_weights, -np.log(n))
    assert "waste_free_acceptance" in updated.metadata
    assert updated.metadata["waste_free_proposal_repeats"] == 1
    assert len(updated.metadata["waste_free_acceptance_by_round"]) == 1
    total_selected = (
        updated.metadata["waste_free_selected_parents"]
        + updated.metadata["waste_free_selected_proposals"]
    )
    assert total_selected == n


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

    updated = step(state, context, rng)
    assert updated.particles.shape == particles.shape
    assert np.allclose(updated.log_weights, -np.log(n))
    assert len(updated.metadata["waste_free_acceptance_by_round"]) == 1
    assert updated.metadata["waste_free_proposal_repeats"] == 1


def test_waste_free_resample_move_multiple_proposals(
    particle_state: ParticleState,
    context_factory,
    rng,
) -> None:
    def log_prob(particles: np.ndarray, context: StepContext) -> np.ndarray:
        pts = np.asarray(particles, dtype=float)
        if pts.ndim == 1:
            pts = pts[:, np.newaxis]
        return -0.5 * np.sum(pts**2, axis=1)

    context = context_factory(shared={"temperature": 1.0})
    step = WasteFreeResampleMove(
        log_prob,
        GaussianRandomWalk(scale=0.4),
        proposals_per_particle=3,
    )

    updated = step(particle_state, context, rng)

    n = particle_state.particles.shape[0]
    assert np.allclose(updated.log_weights, -np.log(n))
    assert updated.metadata["waste_free_proposal_repeats"] == 3
    assert len(updated.metadata["waste_free_acceptance_by_round"]) == 3
    assert len(updated.metadata["waste_free_selected_proposals_by_round"]) == 3
    assert (
        sum(updated.metadata["waste_free_selected_proposals_by_round"])
        == updated.metadata["waste_free_selected_proposals"]
    )
    assert context.metadata["waste_free_total_proposals"] == n * 3


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax", "cupy"])
def test_waste_free_resample_move_preserves_array_backend(
    backend: str,
    backend_factory,
    context_factory,
) -> None:
    xp, particles, log_weights, row_sum = backend_factory(backend)
    state = ParticleState(particles=particles, log_weights=log_weights)
    backend_ns = namespace(particles)

    def log_prob(values, context: StepContext):
        return -0.5 * row_sum(values**2)

    step = WasteFreeResampleMove(
        log_prob,
        GaussianRandomWalk(scale=0.4),
        proposals_per_particle=2,
    )
    updated = step(
        state,
        context_factory(shared={"temperature": 1.0}),
        RandomGenerator(backend=backend, seed=1234),
    )

    assert namespace(updated.particles) is backend_ns
    assert namespace(updated.log_weights) is backend_ns
    assert namespace(updated.metadata["ancestors"]) is backend_ns
    assert updated.particles.shape == particles.shape
    assert updated.log_weights.shape == log_weights.shape


def test_waste_free_resample_move_supports_dynamic_proposal(
    particle_state: ParticleState,
    context_factory,
    rng,
) -> None:
    def log_prob(particles: np.ndarray, context: StepContext) -> np.ndarray:
        pts = np.asarray(particles, dtype=float)
        return -0.5 * np.sum(np.atleast_2d(pts) ** 2, axis=-1)

    context = context_factory(metadata={"proposal_scale": 0.15})

    def factory(parents: np.ndarray, ctx: StepContext) -> GaussianRandomWalk:
        scale = float(ctx.metadata.get("proposal_scale", 0.15))
        ctx.metadata.setdefault("factory_scales", []).append(scale)
        return GaussianRandomWalk(scale=scale)

    adapter = AdaptGaussianProposal(
        scale_key="proposal_scale",
        initial_scale=0.15,
        target_acceptance=None,
    )

    # First update ensures the scale key is present.
    adapter(particle_state, context, rng)

    step = WasteFreeResampleMove(
        log_prob,
        factory,
        proposals_per_particle=2,
    )

    step(particle_state, context, rng)

    assert context.metadata["factory_scales"]
    assert all(
        scale == context.metadata["proposal_scale"]
        for scale in context.metadata["factory_scales"]
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
    context.request_stop_immediately()
    state = particle_state.with_updates(metadata={"temperature": 0.5})

    result = step(state, context, rng)

    assert result is state
    assert not context.metadata["stop_requested"]
    assert not context.metadata["stop_immediately"]

    context_late = context_factory(t=1)
    context_late.request_stop_immediately()
    late_state = particle_state.with_updates(metadata={"temperature": 1.0})

    final_result = step(late_state, context_late, rng)

    assert final_result is late_state
    assert context_late.metadata["stop_requested"] is True
    assert context_late.metadata["stop_immediately"] is True
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
        ) -> ParticleState:
            applied_iterations.append(context.t)
            return state

    recording_step = RecordingStep(initial_scale=0.1, target_acceptance=None)
    step = PeriodicStep(recording_step, n=3, offset=1)

    for t in range(10):
        context = context_factory(t=t)
        step(particle_state, context, rng)

    assert applied_iterations == [1, 4, 7]
