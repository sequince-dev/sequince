from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from sequince.posterior import TemperedPosterior
from sequince.state import ParticleState
from sequince.steps import (
    AdaptGaussianProposal,
    ContinueWhile,
    ExpandDimensions,
    MetadataTransformation,
    SMCTransformation,
    WasteFreeResampleMove,
)
from sequince.steps.base import StepContext
from sequince.steps.control import PeriodicStep
from sequince.steps.mutation import GaussianRandomWalk
from sequince.steps.resampling import (
    MultinomialResampling,
    SystematicResampling,
)
from sequince.steps.weighting import (
    AdaptiveTemperedWeighting,
    FixedTemperedWeighting,
    LogLikelihoodWeighting,
    TemperedWeighting,
)

if TYPE_CHECKING:
    from orng import ArrayRNG


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


def test_adaptive_tempered_weighting_no_op_when_max_reached(
    particle_state: ParticleState, context_factory, rng
) -> None:
    def log_like(particles: np.ndarray, context: StepContext) -> np.ndarray:
        return np.ones(particles.shape[0])

    step = AdaptiveTemperedWeighting(log_like, max_temperature=1.0)
    state = particle_state.with_updates(metadata={"temperature": 1.0})
    context = context_factory(metadata={"temperature": 1.0})

    result_state = step(state, context, rng)

    expected_state = state.with_updates(
        metadata=dict(state.metadata, temperature=1.0)
    )

    assert np.array_equal(result_state.particles, expected_state.particles)
    assert result_state.metadata == expected_state.metadata
    assert context.metadata["stop_requested"] is True
    assert context.metadata["stop_immediately"] is True
    assert context.metadata["temperature"] == 1.0
    assert context.metadata["temperature_delta"] == 0.0


def test_tempered_posterior_custom_increment_is_used_by_weighting(
    particle_state: ParticleState, context_factory, rng
) -> None:
    class CustomPosterior(TemperedPosterior):
        def log_increment(
            self,
            particles: np.ndarray,
            context: StepContext,
            *,
            log_proposal_override=None,
        ) -> np.ndarray:
            context.metadata["custom_increment_called"] = True
            return np.full(particles.shape[0], 2.0)

    posterior = CustomPosterior(
        log_prior=lambda particles, context: np.zeros(particles.shape[0]),
        log_likelihood=lambda particles, context: np.ones(particles.shape[0]),
    )
    context = context_factory(metadata={"temperature": 0.0})
    step = FixedTemperedWeighting(
        posterior,
        temperature_increment=0.25,
        normalize_after=False,
    )

    updated = step(particle_state, context, rng)

    assert context.metadata["custom_increment_called"] is True
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


def test_dimensionality_expansion_appends_block(
    particle_state: ParticleState, context_factory, rng
) -> None:
    class AppendingTransformation(SMCTransformation):
        def forward(
            self,
            state: ParticleState,
            context: StepContext,
            rng: "ArrayRNG",
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
            rng: "ArrayRNG",
        ) -> ParticleState:
            applied_iterations.append(context.t)
            return state

    recording_step = RecordingStep(initial_scale=0.1, target_acceptance=None)
    step = PeriodicStep(recording_step, n=3, offset=1)

    for t in range(10):
        context = context_factory(t=t)
        step(particle_state, context, rng)

    assert applied_iterations == [1, 4, 7]
