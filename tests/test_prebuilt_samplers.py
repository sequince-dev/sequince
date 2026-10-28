from __future__ import annotations

from typing import Literal

import numpy as np
import pytest
from orng import RandomGenerator

from sequince import ParticleState, PersistentParticleState, TemperedPosterior
from sequince.prebuilt_samplers import (
    make_nested_sampling_smc,
    make_persistent_bayesian_smc,
    make_standard_bayesian_smc,
)
from sequince.steps import (
    AdaptiveTemperedWeighting,
    DifferentialEvolutionMutation,
    FixedTemperedWeighting,
    LoggingStep,
    MiniPCNMutation,
    MutateIfResampled,
    PersistentAdaptiveTemperedWeighting,
    PersistParticles,
    SystematicResampling,
)
from sequince.steps.base import SMCStep, StepContext


def _quadratic_log_prior(particles: np.ndarray, context=None) -> np.ndarray:
    values = np.asarray(particles, dtype=float)
    if values.ndim == 1:
        values = values[:, np.newaxis]
    return -0.5 * np.sum(values**2, axis=1)


def _zero_log_likelihood(particles: np.ndarray, context=None) -> np.ndarray:
    values = np.asarray(particles, dtype=float)
    if values.ndim == 1:
        values = values[:, np.newaxis]
    return np.zeros(values.shape[0], dtype=float)


def test_make_standard_bayesian_smc_builds_default_steps() -> None:
    smc, posterior, metadata = make_standard_bayesian_smc(
        _quadratic_log_prior,
        _zero_log_likelihood,
    )

    assert isinstance(posterior, TemperedPosterior)
    assert metadata == {"max_beta": 1.0}

    steps = smc._steps  # noqa: SLF001 - inspecting configured steps for testing.
    assert len(steps) == 4
    assert isinstance(steps[0], AdaptiveTemperedWeighting)
    assert isinstance(steps[1], SystematicResampling)
    assert isinstance(steps[2], DifferentialEvolutionMutation)
    assert isinstance(steps[3], LoggingStep)


def test_make_standard_bayesian_smc_reaches_max_beta(rng) -> None:
    max_beta = 0.5
    smc, _, metadata = make_standard_bayesian_smc(
        _quadratic_log_prior,
        _zero_log_likelihood,
        max_beta=max_beta,
        mutation_n_steps=1,
    )

    particles = np.array([[0.0], [1.0], [-1.0], [2.0]], dtype=float)
    n = particles.shape[0]
    initial_state = ParticleState(
        particles=particles,
        log_weights=np.full(n, -np.log(n)),
        metadata={"beta": 0.0},
    )

    results = smc.run(initial_state, rng=rng, metadata=metadata)

    assert results.states[-1].metadata["beta"] == pytest.approx(max_beta)
    assert results.final_state.evaluation is not None
    assert results.context.metadata["beta"] == pytest.approx(max_beta)


def test_make_standard_bayesian_smc_accepts_evolving_target(rng) -> None:
    schedule_calls: list[float] = []

    def target_schedule(beta: float) -> float:
        schedule_calls.append(beta)
        return 0.8

    smc, _, metadata = make_standard_bayesian_smc(
        _quadratic_log_prior,
        _zero_log_likelihood,
        target_ess_ratio=target_schedule,
        max_beta=0.5,
        mutation="none",
    )
    particles = np.array([[0.0], [1.0], [-1.0], [2.0]], dtype=float)
    initial_state = ParticleState(
        particles=particles,
        log_weights=np.full(particles.shape[0], -np.log(particles.shape[0])),
        metadata={"beta": 0.0},
    )

    result = smc.run(initial_state, rng=rng, metadata=metadata)

    assert schedule_calls
    assert schedule_calls[0] == pytest.approx(0.0)
    assert result.final_state.metadata["target_ess_ratio"] == pytest.approx(
        0.8
    )


def test_make_standard_bayesian_smc_supports_fixed_schedule(rng) -> None:
    smc, _, metadata = make_standard_bayesian_smc(
        _quadratic_log_prior,
        _zero_log_likelihood,
        adaptive=False,
        num_steps=4,
        mutation="none",
    )
    particles = np.array([[0.0], [1.0], [-1.0], [2.0]], dtype=float)

    result = smc.run(
        ParticleState(particles),
        num_steps=4,
        rng=rng,
        metadata=metadata,
    )

    assert isinstance(smc._steps[0], FixedTemperedWeighting)  # noqa: SLF001
    assert result.final_state.metadata["beta"] == pytest.approx(1.0)


def test_make_standard_bayesian_smc_requires_fixed_step_count() -> None:
    with pytest.raises(ValueError, match="num_steps must be positive"):
        make_standard_bayesian_smc(
            _quadratic_log_prior,
            _zero_log_likelihood,
            adaptive=False,
        )


@pytest.mark.parametrize("runtime_steps", [None, 20])
@pytest.mark.parametrize("schedule_steps", [2, 10])
def test_fixed_schedule_stops_and_resamples_at_end(
    runtime_steps: int | None, schedule_steps: int, rng
) -> None:
    smc, _, metadata = make_standard_bayesian_smc(
        _quadratic_log_prior,
        lambda particles: -(particles[:, 0] ** 2),
        adaptive=False,
        num_steps=schedule_steps,
        weighting_criterion="cess",
        resampling_ess_threshold=None,
        mutation="none",
        always_mutate=True,
        logging_frequency=0,
    )
    result = smc.run(
        ParticleState(np.arange(4.0)[:, None]),
        num_steps=runtime_steps,
        rng=rng,
        metadata=metadata,
    )
    assert len(result.states) == schedule_steps
    assert result.context.stop_requested
    assert result.context.metadata["beta"] == 1.0
    assert result.context.metadata["resampled"]
    np.testing.assert_allclose(result.final_state.log_weights, -np.log(4))


def test_make_standard_bayesian_smc_validates_resampling_name() -> None:
    with pytest.raises(ValueError):
        make_standard_bayesian_smc(
            _quadratic_log_prior,
            _zero_log_likelihood,
            resampling="invalid",  # type: ignore[arg-type]
        )


def test_make_standard_bayesian_smc_forwards_resample_at_end() -> None:
    smc, _, _ = make_standard_bayesian_smc(
        _quadratic_log_prior,
        _zero_log_likelihood,
        resample_at_end=False,
        logging_frequency=0,
    )

    resampling = smc._steps[1]  # noqa: SLF001
    assert isinstance(resampling, SystematicResampling)
    assert not resampling._resample_at_end  # noqa: SLF001


@pytest.mark.parametrize(
    ("weighting_criterion", "always_mutate", "expected_type"),
    [
        ("ess", None, DifferentialEvolutionMutation),
        ("cess", None, MutateIfResampled),
        ("cess", True, DifferentialEvolutionMutation),
    ],
)
def test_make_standard_bayesian_smc_configures_conditional_mutation(
    weighting_criterion: Literal["ess", "cess"],
    always_mutate: bool | None,
    expected_type: type[SMCStep],
) -> None:
    smc, _, _ = make_standard_bayesian_smc(
        _quadratic_log_prior,
        _zero_log_likelihood,
        weighting_criterion=weighting_criterion,
        always_mutate=always_mutate,
        logging_frequency=0,
    )

    mutation = smc._steps[2]  # noqa: SLF001
    assert isinstance(mutation, expected_type)
    if isinstance(mutation, MutateIfResampled):
        assert isinstance(mutation.mutation, DifferentialEvolutionMutation)


def test_make_standard_bayesian_smc_wraps_custom_mutation() -> None:
    class CustomMutation(SMCStep):
        def apply(self, state, context, rng):
            return state, context, None

    custom_mutation = CustomMutation()

    smc, _, _ = make_standard_bayesian_smc(
        _quadratic_log_prior,
        _zero_log_likelihood,
        weighting_criterion="cess",
        mutation=custom_mutation,
        logging_frequency=0,
    )

    mutation = smc._steps[2]  # noqa: SLF001
    assert isinstance(mutation, MutateIfResampled)
    assert mutation.mutation is custom_mutation


def test_standard_smc_rejects_redundant_conditional_mutation() -> None:
    with pytest.raises(ValueError, match="always_mutate=False"):
        make_standard_bayesian_smc(
            _quadratic_log_prior,
            _zero_log_likelihood,
            always_resample=True,
            always_mutate=False,
        )


@pytest.mark.parametrize("weighting_criterion", ["ess", "cess"])
@pytest.mark.parametrize("always_resample", [None, True, False])
@pytest.mark.parametrize("always_mutate", [None, True, False])
def test_standard_smc_without_mutation_accepts_mutation_flags(
    weighting_criterion, always_resample, always_mutate
) -> None:
    kwargs = dict(
        mutation="none",
        weighting_criterion=weighting_criterion,
        always_resample=always_resample,
        always_mutate=always_mutate,
        logging_frequency=0,
    )
    if weighting_criterion == "ess" and always_resample is False:
        with pytest.raises(
            ValueError, match="always_resample cannot be False"
        ):
            make_standard_bayesian_smc(
                _quadratic_log_prior, _zero_log_likelihood, **kwargs
            )
        return

    smc, _, metadata = make_standard_bayesian_smc(
        _quadratic_log_prior, _zero_log_likelihood, **kwargs
    )
    particles = np.arange(4.0)[:, None]
    result = smc.run(ParticleState(particles), metadata=metadata, rng=123)

    assert result.context.metadata["beta"] == pytest.approx(1.0)
    assert len(result.outputs[-1]) == 2
    np.testing.assert_allclose(result.final_state.weights, 0.25)
    assert result.log_evidence() == pytest.approx(0.0)


def test_make_standard_bayesian_smc_accepts_proposal_distribution(
    analytic_proposal_factory,
) -> None:
    _, posterior, _ = make_standard_bayesian_smc(
        _quadratic_log_prior,
        _zero_log_likelihood,
        log_proposal=analytic_proposal_factory(n_dim=1),
    )
    particles = np.array([[0.0], [1.0]], dtype=float)
    context = StepContext(t=0, metadata={"beta": 0.0})
    evaluation = posterior.evaluate(particles, context)
    values = posterior.log_increment(particles, context, evaluation=evaluation)

    assert np.allclose(values, _quadratic_log_prior(particles))


def test_make_persistent_bayesian_smc_builds_fixed_size_pipeline() -> None:
    n_particles = 17
    target_ess = 29
    target_total_ess = 41

    smc, posterior, metadata = make_persistent_bayesian_smc(
        _quadratic_log_prior,
        _zero_log_likelihood,
        n_particles=n_particles,
        target_ess=target_ess,
        target_total_ess=target_total_ess,
    )

    assert isinstance(posterior, TemperedPosterior)
    assert metadata == {}

    steps = smc._steps  # noqa: SLF001 - inspecting configured steps for testing.
    assert len(steps) == 4
    assert isinstance(steps[0], PersistParticles)
    assert isinstance(steps[1], PersistentAdaptiveTemperedWeighting)
    assert isinstance(steps[2], SystematicResampling)
    assert isinstance(steps[3], DifferentialEvolutionMutation)
    assert steps[1]._target_ess == target_ess  # noqa: SLF001
    assert steps[1]._target_total_ess == target_total_ess  # noqa: SLF001
    assert steps[2]._n_samples == n_particles  # noqa: SLF001
    assert steps[2]._always_resample  # noqa: SLF001


def test_make_persistent_bayesian_smc_defaults_target_ess_to_two_batches() -> (
    None
):
    n_particles = 17

    smc, _, _ = make_persistent_bayesian_smc(
        _quadratic_log_prior,
        _zero_log_likelihood,
        n_particles=n_particles,
    )

    weighting = smc._steps[1]  # noqa: SLF001
    assert isinstance(weighting, PersistentAdaptiveTemperedWeighting)
    assert weighting._target_ess == 2 * n_particles  # noqa: SLF001


def test_make_persistent_bayesian_smc_logs_archive_progress(
    rng,
    caplog,
) -> None:
    n_particles = 5
    smc, posterior, metadata = make_persistent_bayesian_smc(
        _quadratic_log_prior,
        _zero_log_likelihood,
        n_particles=n_particles,
        target_ess=2 * n_particles,
        mutation="none",
    )
    particles = np.linspace(-1.0, 1.0, n_particles)[:, None]
    context = StepContext(t=0, metadata={"beta": 0.0})
    initial_state = ParticleState(
        particles,
        evaluation=posterior.prepare_evaluation(particles, context),
    )

    with caplog.at_level("INFO", logger="sequince.prebuilt_samplers"):
        smc.run(initial_state, metadata=metadata, rng=rng)

    assert "Persistent SMC iteration 0" in caplog.messages[0]
    terminal_message = caplog.messages[-1]
    assert "Persistent SMC iteration 2" in terminal_message
    assert "Beta=1" in terminal_message
    assert "Batches=3" in terminal_message
    assert f"Archive size={3 * n_particles}" in terminal_message
    assert (
        f"Persistent ESS={3 * n_particles:.1f}/{2 * n_particles:.1f}"
        in terminal_message
    )


def test_make_persistent_bayesian_smc_returns_weighted_persistent_population(
    rng,
) -> None:
    n_particles = 5
    smc, posterior, metadata = make_persistent_bayesian_smc(
        _quadratic_log_prior,
        _zero_log_likelihood,
        n_particles=n_particles,
        target_ess=2 * n_particles,
        mutation="none",
        resample_at_end=False,
    )
    particles = np.linspace(-1.0, 1.0, n_particles)[:, None]
    context = StepContext(t=0, metadata={"beta": 0.0})
    initial_state = ParticleState(
        particles,
        evaluation=posterior.prepare_evaluation(particles, context),
    )

    result = smc.run(initial_state, metadata=metadata, rng=rng)

    assert isinstance(result.final_state, PersistentParticleState)
    assert result.final_state.particles.shape == (2 * n_particles, 1)
    assert result.final_state.metadata["beta"] == pytest.approx(1.0)
    np.testing.assert_allclose(result.final_state.weights.sum(), 1.0)
    persist = smc._steps[0]  # noqa: SLF001
    assert isinstance(persist, PersistParticles)
    assert persist.persistent_state is not None
    assert (
        result.final_state.n_particles == persist.persistent_state.n_particles
    )


@pytest.mark.parametrize("target_total_ess", [None, 12.0])
def test_persistent_default_finishes_terminal_mutation(
    target_total_ess: float | None,
) -> None:
    terminal_calls: list[ParticleState] = []

    class RecordMutation(SMCStep):
        def apply(
            self,
            state: ParticleState,
            context: StepContext,
            rng: RandomGenerator,
        ) -> tuple[ParticleState, StepContext, None]:
            if float(context.metadata["beta"]) == 1.0:
                terminal_calls.append(state)
            return state, context, None

    smc, _, metadata = make_persistent_bayesian_smc(
        _quadratic_log_prior,
        _zero_log_likelihood,
        n_particles=4,
        target_ess=8,
        target_total_ess=target_total_ess,
        mutation=RecordMutation(),
        logging_frequency=0,
    )
    result = smc.run(
        ParticleState(np.arange(4.0)[:, None]), metadata=metadata, rng=123
    )
    assert len(terminal_calls) >= 1
    assert result.context.stop_requested
    assert result.context.stop_immediately
    assert isinstance(result.final_state, PersistentParticleState)
    n_batches = len(result)
    assert result.final_state.particles.shape == (4 * n_batches, 1)
    np.testing.assert_allclose(result.final_state.weights, 1 / (4 * n_batches))
    np.testing.assert_allclose(
        result.final_state.particles[-4:], terminal_calls[-1].particles
    )
    assert result.final_state.betas[-1] == pytest.approx(1.0)
    assert result.final_state.log_evidence == pytest.approx(0.0)
    required_ess = target_total_ess or 8
    assert result.outputs[-1][1]["effective_sample_size"] >= required_ess


@pytest.mark.parametrize("auto_normalize", [False, True])
def test_persistent_final_batch_reweights_complete_archive(
    auto_normalize: bool,
) -> None:
    terminal_batches: list[ParticleState] = []

    class ShiftMutation(SMCStep):
        def apply(
            self,
            state: ParticleState,
            context: StepContext,
            rng: RandomGenerator,
        ) -> tuple[ParticleState, StepContext, None]:
            # A deterministic marker verifies that the post-mutation batch is
            # included, rather than its resampled parents. This is not MCMC.
            state = state.replace_particles(state.particles + 0.5)
            terminal_batches.append(state)
            return state, context, None

    def log_likelihood(particles: np.ndarray) -> np.ndarray:
        return -0.5 * particles[:, 0] ** 2

    sampler, posterior, _ = make_persistent_bayesian_smc(
        _quadratic_log_prior,
        log_likelihood,
        n_particles=4,
        target_ess=1,
        mutation=ShiftMutation(),
        auto_normalize=auto_normalize,
        logging_frequency=0,
    )
    initial = np.linspace(-1, 2, 4)[:, None]
    result = sampler.run(ParticleState(initial), rng=123)
    final = result.final_state
    assert isinstance(final, PersistentParticleState)
    assert len(terminal_batches) == 1
    batch = terminal_batches[0]
    np.testing.assert_array_equal(final.particles[:4], initial)
    np.testing.assert_array_equal(final.particles[4:], batch.particles)
    np.testing.assert_array_equal(final.betas, [0, 1])
    np.testing.assert_array_equal(
        final.initial_ancestor_indices[4:], batch.initial_ancestor_indices
    )
    log_z = np.log(np.mean(np.exp(log_likelihood(initial))))
    np.testing.assert_allclose(final.log_normalizing_constants, [0, log_z])
    likelihood = np.exp(log_likelihood(final.particles))
    mixture = (1 + likelihood / np.exp(log_z)) / 2
    raw_weights = likelihood / mixture
    np.testing.assert_allclose(final.weights, raw_weights / raw_weights.sum())
    assert final.log_evidence == pytest.approx(np.log(raw_weights.mean()))
    assert final.evaluation is not None
    expected_evaluation = posterior.prepare_evaluation(
        final.particles, result.context
    )
    for name in expected_evaluation.components:
        np.testing.assert_allclose(
            final.evaluation[name], expected_evaluation[name]
        )


@pytest.mark.parametrize(
    ("target_total_ess", "expected_batches"), [(3, 2), (9, 2), (14, 3)]
)
def test_persistent_checks_terminal_ess_after_mutation(
    target_total_ess: float, expected_batches: int
) -> None:
    mutated_batches: list[ParticleState] = []

    class RecordMutation(SMCStep):
        def apply(
            self,
            state: ParticleState,
            context: StepContext,
            rng: RandomGenerator,
        ) -> tuple[ParticleState, StepContext, None]:
            assert not context.stop_requested
            assert float(context.metadata["beta"]) == 1.0
            mutated_batches.append(state)
            return state, context, None

    sampler, _, _ = make_persistent_bayesian_smc(
        _quadratic_log_prior,
        _zero_log_likelihood,
        n_particles=5,
        target_ess=2,
        target_total_ess=target_total_ess,
        mutation=RecordMutation(),
        logging_frequency=0,
    )
    result = sampler.run(ParticleState(np.arange(5.0)[:, None]), rng=123)
    final = result.final_state
    assert isinstance(final, PersistentParticleState)
    assert final.n_batches == expected_batches
    assert len(mutated_batches) == expected_batches - 1
    assert final.effective_sample_size() >= target_total_ess
    assert result.context.stop_immediately
    np.testing.assert_array_equal(
        final.particles[-5:], mutated_batches[-1].particles
    )
    np.testing.assert_array_equal(final.betas[1:], 1.0)


@pytest.mark.parametrize("resample_at_end", [False, True])
def test_persistent_zero_support_preserves_evidence(
    rng, resample_at_end: bool
) -> None:
    smc, _, metadata = make_persistent_bayesian_smc(
        _zero_log_likelihood,
        lambda p: np.where(p[:, 0] > 0, 0.0, -np.inf),
        n_particles=4,
        target_ess=2,
        mutation="none",
        resample_at_end=resample_at_end,
        logging_frequency=0,
    )
    result = smc.run(
        ParticleState(np.arange(4.0)[:, None]),
        num_steps=2,
        rng=rng,
        metadata=metadata,
    )
    assert result.context.stop_requested
    assert result.context.metadata["beta"] == 1.0
    n_supported = 7 if resample_at_end else 3
    np.testing.assert_allclose(
        result.final_state.weights, [0] + [1 / n_supported] * n_supported
    )
    assert result.final_state.log_evidence == pytest.approx(np.log(0.75))


@pytest.mark.parametrize(
    ("resampling_kwargs", "message"),
    [
        ({"always_resample": False}, "always_resample=True"),
        ({"n_samples": 8}, "n_samples to equal n_particles"),
    ],
)
def test_make_persistent_bayesian_smc_rejects_incompatible_resampling(
    resampling_kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        make_persistent_bayesian_smc(
            _quadratic_log_prior,
            _zero_log_likelihood,
            n_particles=4,
            resampling_kwargs=resampling_kwargs,
        )


def test_make_persistent_bayesian_smc_accepts_custom_resampling_step() -> None:
    resampling = SystematicResampling(always_resample=True, n_samples=4)

    smc, _, _ = make_persistent_bayesian_smc(
        _quadratic_log_prior,
        _zero_log_likelihood,
        n_particles=4,
        resampling=resampling,
    )

    assert smc._steps[2] is resampling  # noqa: SLF001


def test_make_persistent_bayesian_smc_rejects_custom_resampling_step() -> None:
    resampling = SystematicResampling(always_resample=True, n_samples=5)

    with pytest.raises(ValueError, match="n_samples=4"):
        make_persistent_bayesian_smc(
            _quadratic_log_prior,
            _zero_log_likelihood,
            n_particles=4,
            resampling=resampling,
        )


@pytest.mark.parametrize(
    "factory",
    [
        make_standard_bayesian_smc,
        make_persistent_bayesian_smc,
        make_nested_sampling_smc,
    ],
)
@pytest.mark.parametrize("mutation", [None, "minipcn"])
def test_prebuilt_mutation_default_and_explicit_minipcn(
    factory, mutation: str | None
) -> None:
    kwargs = {} if mutation is None else {"mutation": mutation}
    sampler, _, _ = factory(
        _quadratic_log_prior, _zero_log_likelihood, **kwargs
    )
    expected = (
        DifferentialEvolutionMutation if mutation is None else MiniPCNMutation
    )
    assert any(isinstance(step, expected) for step in sampler.config.steps)
