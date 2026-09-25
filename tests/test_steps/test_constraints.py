from __future__ import annotations

import math

import h5py
import numpy as np
import pytest

from sequince import (
    NestedSamplingSMCResult,
    ParticleState,
    make_nested_sampling_smc,
)
from sequince.array_api import namespace, to_numpy
from sequince.checkpoint import HDF5CheckpointWriter, SMCCheckpoint
from sequince.result import SMCResult
from sequince.steps.constraints import (
    FixedLikelihoodThresholdSchedule,
    LikelihoodConstrainedPrior,
    LikelihoodThresholdSelection,
    LikelihoodThresholdWeighting,
    QuantileLikelihoodThresholdSchedule,
    ShellEvidenceStep,
)
from sequince.steps.context import StepContext


def prior(x):
    return namespace(x).zeros_like(x[..., 0])


def likelihood(x):
    return x[..., 0]


def make_sampler(**kwargs):
    options = dict(
        mutation="none",
        always_resample=False,
        resampling_ess_threshold=None,
        logging_frequency=0,
        evidence_tolerance=None,
    )
    options.update(kwargs)
    return make_nested_sampling_smc(prior, likelihood, **options)


def run(values=(1.0, 2.0, 3.0, 4.0), weights=None, **kwargs):
    sampler, _, metadata = make_sampler(**kwargs)
    state = ParticleState(
        np.log(np.array(values))[:, None],
        log_weights=None if weights is None else np.log(weights),
    )
    return NestedSamplingSMCResult(
        sampler.run(state, metadata=metadata, rng=42, num_steps=100)
    )


@pytest.mark.parametrize("inclusive", [False, True])
@pytest.mark.parametrize("weights", [None, np.array([0.1, 0.2, 0.3, 0.4])])
def test_shell_partition_and_posterior(inclusive, weights):
    result = run(
        weights=weights,
        threshold_schedule=[math.log(2.0)],
        threshold_inclusive=inclusive,
    )
    base = np.ones(4) / 4 if weights is None else weights
    expected = base * np.arange(1.0, 5.0)
    assert np.exp(result.log_evidence) == pytest.approx(expected.sum())
    samples, actual = result.posterior_samples()
    np.testing.assert_allclose(np.exp(samples[:, 0]), np.arange(1.0, 5.0))
    np.testing.assert_allclose(actual, expected / expected.sum())
    assert result.complete
    assert result.levels()[0].n_surviving == (3 if inclusive else 2)
    assert result.summary()["final_log_prior_volume"] == pytest.approx(
        math.log(base[1 if inclusive else 2 :].sum())
    )


def test_prior_mass_stop_integrates_tail():
    result = run(min_log_prior_volume=-0.5)
    assert len(result) == 1
    assert len(result.levels()) == 2
    assert result.log_evidence == pytest.approx(math.log(2.5))
    assert result.summary()["final_log_prior_volume"] == pytest.approx(
        math.log(0.5)
    )
    assert result.summary()["stop_reason"] == "prior_mass"
    assert result.levels()[-1].is_terminal
    np.testing.assert_allclose(
        result.posterior_samples()[1], [0.1, 0.2, 0.3, 0.4]
    )


@pytest.mark.parametrize("schedule", [None, "worst"])
def test_constant_likelihood_is_terminal(schedule):
    result = run(values=(3.0, 3.0, 3.0, 3.0), threshold_schedule=schedule)
    assert len(result) == 1
    assert result.log_evidence == pytest.approx(math.log(3.0))
    np.testing.assert_allclose(result.posterior_samples()[1], [0.25] * 4)


def test_tied_likelihoods_are_partitioned_once():
    result = run(values=(1.0, 1.0, 2.0, 2.0))
    assert result.log_evidence == pytest.approx(math.log(1.5))
    assert result.complete
    assert np.exp(result.posterior_samples()[0][:, 0]).shape == (4,)


def test_zero_likelihood_and_empty_shells():
    with np.errstate(divide="ignore"):
        result = run(
            values=(0.0, 0.0, 1.0, 3.0), threshold_schedule=[-math.inf, -10.0]
        )
    assert result.log_evidence == pytest.approx(0.0)
    assert result.levels()[0].log_evidence_increment == -math.inf
    assert result.levels()[1].log_evidence_increment == -math.inf
    assert not any(math.isnan(level.log_evidence) for level in result.levels())
    np.testing.assert_allclose(
        result.posterior_samples()[1], [0.0, 0.0, 0.25, 0.75]
    )


def test_zero_evidence_has_no_posterior():
    with np.errstate(divide="ignore"):
        result = run(values=(0.0, 0.0, 0.0, 0.0))
    assert result.complete
    assert result.log_evidence == -math.inf
    with pytest.raises(ValueError, match="undefined"):
        result.posterior_samples()


def test_limit_suspends_without_false_completion():
    sampler, _, metadata = make_sampler(threshold_schedule=[0.0, 1.0])
    result = NestedSamplingSMCResult(
        sampler.run(
            ParticleState(np.array([[-1.0], [0.5], [2.0]])),
            metadata=metadata,
            num_steps=1,
        )
    )
    assert not result.complete
    with pytest.raises(ValueError, match="incomplete"):
        result.posterior_samples()


def test_evidence_only_mode():
    result = run(store_shell_particles=False)
    assert result.log_evidence == pytest.approx(math.log(2.5))
    with pytest.raises(ValueError, match="store_shell_particles"):
        result.posterior_samples()


def test_evidence_tolerance_records_tail():
    result = run(
        values=(1.0, 2.0, 3.0, 4.0),
        threshold_schedule=[math.log(3.0), math.log(3.5)],
        evidence_tolerance=0.9,
    )
    assert result.summary()["stop_reason"] == "evidence_tolerance"
    assert result.log_evidence == pytest.approx(math.log(2.5))


def test_cached_likelihood_counts():
    calls = []

    def counted(x):
        calls.append(len(x))
        return likelihood(x)

    sampler, _, metadata = make_nested_sampling_smc(
        prior,
        counted,
        mutation="none",
        always_resample=False,
        resampling_ess_threshold=None,
        logging_frequency=0,
        evidence_tolerance=None,
    )
    result = sampler.run(
        ParticleState(np.log(np.arange(1.0, 5.0))[:, None]),
        metadata=metadata,
        num_steps=10,
    )
    assert calls == [4]
    assert (
        int(result.context.evaluation_counts.log_likelihood_evaluations) == 4
    )


def test_target_cache_survives_threshold_changes():
    target = LikelihoodConstrainedPrior(prior, likelihood)
    context = StepContext(t=0)
    particles = np.array([[0.0], [2.0]])
    evaluation = target.prepare_evaluation(particles, context)
    changed = context.with_metadata(nested_sampling={"threshold": 1.0})
    np.testing.assert_allclose(
        target.log_prob(particles, changed, evaluation=evaluation),
        [-math.inf, 0.0],
    )
    other = LikelihoodConstrainedPrior(prior, likelihood)
    with pytest.raises(ValueError, match="different target"):
        other.log_prob(particles, context, evaluation=evaluation)
    assert target.log_prob(particles[1], changed) == 0.0


def test_zero_likelihood_excluded_after_initial_constraint():
    target = LikelihoodConstrainedPrior(prior, likelihood)
    particles = np.array([[-math.inf], [0.0]])
    np.testing.assert_allclose(
        target.log_prob(particles, StepContext(t=0)), [0.0, 0.0]
    )
    context = StepContext(
        t=1,
        metadata={
            "nested_sampling": {
                "threshold": -math.inf,
                "next_threshold": -math.inf,
            }
        },
    )
    np.testing.assert_allclose(
        target.log_prob(particles, context), [-math.inf, 0.0]
    )


@pytest.mark.parametrize("invalid", [math.nan, math.inf])
def test_invalid_likelihood_rejected(invalid):
    sampler, _, metadata = make_sampler()
    with pytest.raises(ValueError, match="NaN or positive infinity"):
        sampler.run(
            ParticleState(np.array([[invalid], [1.0]])),
            metadata=metadata,
            num_steps=1,
        )


@pytest.mark.parametrize(
    "thresholds", [[], [0.0, -1.0], [1.0, 1.0], [math.nan]]
)
def test_invalid_fixed_schedule(thresholds):
    with pytest.raises(ValueError, match="strictly increasing"):
        FixedLikelihoodThresholdSchedule(thresholds)


@pytest.mark.parametrize("error", [TypeError, ValueError, RuntimeError])
def test_schedule_error_propagates(error):
    def broken(values, state, context):
        raise error("user schedule failed")

    with pytest.raises(error, match="user schedule failed"):
        run(threshold_schedule=broken)


def test_schedule_receives_likelihoods_state_and_context() -> None:
    # Argument names have no significance; all three values are positional.
    def schedule(values, population, runtime):
        np.testing.assert_array_equal(values, population.particles[:, 0])
        assert runtime.t == 0
        assert population.particles.shape == (4, 1)
        raise StopIteration

    result = run(threshold_schedule=schedule)
    assert result.log_evidence == pytest.approx(math.log(2.5))
    assert result.summary()["stop_reason"] == "schedule_exhausted"


@pytest.mark.parametrize(
    "schedule", [lambda values: math.inf, lambda values, context: math.inf]
)
def test_schedule_requires_three_arguments(schedule) -> None:
    with pytest.raises(TypeError):
        run(threshold_schedule=schedule)


def test_quantile_ignores_zero_weight_particles():
    state = ParticleState(
        np.arange(3.0)[:, None],
        log_weights=np.array([-math.inf, -math.inf, 0.0]),
    )
    assert (
        QuantileLikelihoodThresholdSchedule()(
            np.arange(3.0), state, StepContext(t=0)
        )
        == 2.0
    )


def test_functional_non_jax_rejected_explicitly():
    step = LikelihoodThresholdSelection(
        LikelihoodConstrainedPrior(prior, likelihood)
    )
    with pytest.raises(NotImplementedError, match="JAX"):
        step.initialize_functional_metadata(ParticleState(np.ones((4, 1))), {})


@pytest.mark.parametrize("method", ["weights", "multinomial", "rejection"])
def test_posterior_sampling_methods(method):
    samples, weights = run().posterior_samples(method=method, rng=42)
    assert samples.shape[0] == weights.shape[0]
    assert weights.sum() == pytest.approx(1.0)


def test_portable_result_roundtrip(tmp_path):
    original = run()
    with h5py.File(tmp_path / "result.h5", "w") as f:
        original.result.save(f, strict=True)
    with h5py.File(tmp_path / "result.h5", "r") as f:
        loaded = NestedSamplingSMCResult(SMCResult.load(f))
    assert loaded.summary() == original.summary()
    np.testing.assert_array_equal(
        loaded.threshold_history(), original.threshold_history()
    )
    np.testing.assert_array_equal(
        loaded.log_prior_volume_history(), original.log_prior_volume_history()
    )
    for actual, expected in zip(
        loaded.posterior_samples(), original.posterior_samples()
    ):
        np.testing.assert_allclose(actual, expected)


def test_checkpoint_resume_with_new_target(tmp_path):
    path = tmp_path / "checkpoint.h5"
    options = dict(
        threshold_schedule=[math.log(2.0), math.log(3.0)], always_resample=True
    )
    initial = ParticleState(np.log(np.arange(1.0, 5.0))[:, None])
    sampler, _, metadata = make_sampler(**options)
    expected = NestedSamplingSMCResult(
        sampler.run(initial, metadata=metadata, rng=123)
    )
    sampler, _, metadata = make_sampler(
        **options, checkpoint_writers=[HDF5CheckpointWriter(path)]
    )
    sampler.run(initial, metadata=metadata, rng=123, num_steps=1)
    with h5py.File(path, "r") as f:
        checkpoint = SMCCheckpoint.load_hdf5(f)
    sampler, _, _ = make_sampler(**options)
    actual = NestedSamplingSMCResult(
        sampler.run(runtime=sampler.resume_from(checkpoint))
    )
    assert actual.summary() == expected.summary()
    for a, b in zip(actual.posterior_samples(), expected.posterior_samples()):
        np.testing.assert_allclose(a, b)


def test_backend_shell_partition(backend, backend_factory):
    case = backend_factory(backend)
    xp = case.xp
    particles = xp.asarray([[0.0], [1.0], [2.0], [3.0]])
    sampler, _, metadata = make_sampler(threshold_schedule=[1.0])
    result = NestedSamplingSMCResult(
        sampler.run(
            ParticleState(particles), metadata=metadata, rng=case.rng(42)
        )
    )
    samples, weights = result.posterior_samples()
    assert case.matches_namespace(samples)
    assert case.matches_namespace(weights)
    np.testing.assert_allclose(
        to_numpy(weights),
        np.exp(np.arange(4.0)) / np.exp(np.arange(4.0)).sum(),
        rtol=1e-6,
    )


@pytest.mark.parametrize("mutation", ["minipcn", "differential-evolution"])
def test_gaussian_evidence_and_posterior(mutation):
    def log_prior(x):
        return -0.5 * np.sum(x**2, axis=-1) - 0.5 * math.log(2 * math.pi)

    sampler, _, metadata = make_nested_sampling_smc(
        log_prior,
        lambda x: -0.5 * np.sum(x**2, axis=-1),
        mutation=mutation,
        mutation_n_steps=10,
        logging_frequency=0,
    )
    initial = ParticleState(np.random.default_rng(42).normal(size=(1024, 1)))
    result = NestedSamplingSMCResult(
        sampler.run(initial, metadata=metadata, rng=123, num_steps=100)
    )
    assert result.complete
    assert result.log_evidence == pytest.approx(-0.5 * math.log(2), abs=0.08)
    samples, weights = result.posterior_samples()
    assert np.sum(weights * samples[:, 0]) == pytest.approx(0.0, abs=0.08)
    assert np.sum(weights * samples[:, 0] ** 2) == pytest.approx(0.5, abs=0.08)
    assert result.context.evaluation_counts.log_likelihood_evaluations > 1024


def test_repeated_custom_threshold_rejected():
    with pytest.raises(ValueError, match="must increase"):
        run(threshold_schedule=lambda values, state, context: math.log(2.0))


@pytest.mark.parametrize(
    "options",
    [
        {"auto_normalize": True},
        {"threshold_schedule": "unknown"},
        {"threshold_quantile": 1.0},
        {"threshold_schedule": "worst", "threshold_quantile": 0.5},
        {"threshold_inclusive": True},
        {"min_log_prior_volume": 0.0},
        {"evidence_tolerance": 0.0},
    ],
)
def test_invalid_configuration(options):
    with pytest.raises(ValueError):
        make_sampler(**options)


@pytest.mark.parametrize("mutation", ["minipcn", "differential-evolution"])
def test_backend_mutation(backend, backend_factory, mutation):
    case = backend_factory(backend)
    xp = case.xp

    def gaussian(x):
        return -0.5 * xp.sum(x**2, axis=-1)

    sampler, _, metadata = make_nested_sampling_smc(
        gaussian,
        gaussian,
        mutation=mutation,
        mutation_n_steps=5,
        logging_frequency=0,
    )
    particles = xp.asarray(np.random.default_rng(42).normal(size=(128, 1)))
    result = NestedSamplingSMCResult(
        sampler.run(
            ParticleState(particles),
            metadata=metadata,
            num_steps=100,
            rng=case.rng(42),
        )
    )
    assert result.complete
    assert result.log_evidence == pytest.approx(-0.5 * math.log(2), abs=0.15)
    assert case.matches_namespace(result.posterior_samples()[0])


def test_likelihood_bound_prevents_premature_evidence_stop():
    # The sampled survivors suggest little remaining evidence, but a known
    # larger likelihood maximum requires exploring the next level first.
    options = dict(
        threshold_schedule=[math.log(3.0), math.log(3.5)],
        evidence_tolerance=0.9,
    )
    unbounded = run(**options)
    bounded = run(**options, log_likelihood_upper_bound=math.log(100.0))
    assert len(unbounded) == 2
    assert len(bounded) == 3
    assert bounded.summary()["stop_reason"] == "schedule_exhausted"
    assert bounded.log_evidence == pytest.approx(math.log(2.5))


def test_likelihood_bound_checked_against_observations():
    with pytest.raises(ValueError, match="exceeds"):
        run(log_likelihood_upper_bound=math.log(2.0))


def test_result_conversion_helper():
    result = run()
    assert (
        result.result.as_nested_sampling_result().summary() == result.summary()
    )


def test_context_owns_runtime_and_state_records_completed_levels() -> None:
    target = LikelihoodConstrainedPrior(prior, likelihood)
    state = ParticleState(
        np.log(np.arange(1.0, 5.0))[:, None],
        metadata={"nested_sampling": {"is_terminal": True}},
    )
    selection = LikelihoodThresholdSelection(
        target,
        threshold_schedule=FixedLikelihoodThresholdSchedule([math.log(2)]),
    )
    selected, context, _ = selection(state, StepContext(t=0))
    # Stale state snapshots must not override execution context.
    assert not context.metadata["nested_sampling"]["is_terminal"]
    assert selected.metadata == state.metadata
    selected_context = context
    shelled, context, _ = ShellEvidenceStep(target)(selected, context)
    assert (
        selected_context.metadata["nested_sampling"]["log_nested_evidence"]
        == -math.inf
    )
    assert shelled.metadata == state.metadata
    weighted, context, _ = LikelihoodThresholdWeighting(target)(
        shelled, context
    )
    snapshot = weighted.metadata["nested_sampling"].copy()
    assert snapshot == context.metadata["nested_sampling"]
    np.testing.assert_allclose(
        target.log_prob(weighted.particles, context),
        [-math.inf, -math.inf, 0.0, 0.0],
    )
    weighted_context = context
    selected, context, _ = selection(
        weighted, StepContext(t=1, metadata=context.metadata)
    )
    assert weighted_context.metadata["nested_sampling"] == snapshot
    terminal, context, _ = ShellEvidenceStep(target)(selected, context)
    assert (
        terminal.metadata["nested_sampling"]
        == context.metadata["nested_sampling"]
    )
    assert terminal.metadata["nested_sampling"]["is_terminal"]
    assert weighted.metadata["nested_sampling"] == snapshot
    context.metadata["nested_sampling"]["threshold"] = 100.0
    assert (
        terminal.metadata["nested_sampling"]["threshold"]
        == snapshot["threshold"]
    )
