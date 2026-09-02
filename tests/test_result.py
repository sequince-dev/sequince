import warnings

import numpy as np
import pytest

from sequince.result import MetaSMCResult, SMCResult
from sequince.state import ParticleState
from sequince.steps.context import StepContext
from sequince.steps.meta import META_SMC_OUTPUT_KEY
from sequince.target import EvaluationCounts


@pytest.fixture
def dummy_results(particle_state, context_factory) -> SMCResult:
    return SMCResult(
        [
            particle_state,
        ]
        * 10,
        context_factory(),
    )


def test_smc_results_length(dummy_results: SMCResult) -> None:
    assert len(dummy_results) == 10


def test_smc_results_indexing(
    dummy_results: SMCResult, particle_state
) -> None:
    for i in range(10):
        assert dummy_results[i] == particle_state

    assert dummy_results.final_state == particle_state


def test_smc_result_exposes_evaluation_counts(particle_state) -> None:
    counts = EvaluationCounts(
        log_target_evaluations=12,
        log_proposal_evaluations=3,
    )
    result = SMCResult(
        [particle_state],
        StepContext(t=0, evaluation_counts=counts),
    )

    assert result.evaluation_counts is counts


def test_smc_results_append_state(
    dummy_results: SMCResult, particle_state
) -> None:
    updated = dummy_results.append_state(particle_state)

    assert len(updated) == len(dummy_results) + 1
    assert updated.final_state == particle_state
    assert len(dummy_results) == 10


def test_smc_results_summary(dummy_results: SMCResult) -> None:
    summary = dummy_results.summary()

    assert summary["final_log_normalizing_constant"] == pytest.approx(0.0)
    assert "final_log_evidence" not in summary


def test_smc_result_normalizing_constant_alias(
    particle_state: ParticleState, context_factory
) -> None:
    state = particle_state.with_updates(log_normalizing_constant=1.75)
    result = SMCResult([state], context_factory())

    assert result.log_normalizing_constant() == pytest.approx(1.75)
    assert result.log_evidence() == result.log_normalizing_constant()
    np.testing.assert_array_equal(
        result.log_evidence_history(),
        result.log_normalizing_constant_history(),
    )


@pytest.fixture
def weighted_results(context_factory) -> SMCResult:
    state = ParticleState(
        particles=np.arange(4.0)[:, None],
        log_weights=np.log([0.1, 0.2, 0.3, 0.4]),
    )
    return SMCResult([state], context_factory())


def test_posterior_samples_returns_final_particles_and_weights(
    weighted_results: SMCResult,
) -> None:
    samples, weights = weighted_results.posterior_samples()

    np.testing.assert_array_equal(
        samples, weighted_results.final_state.particles
    )
    np.testing.assert_allclose(weights, [0.1, 0.2, 0.3, 0.4])
    np.testing.assert_allclose(weights.sum(), 1.0)


def test_posterior_samples_downselects_and_renormalizes_weights(
    weighted_results: SMCResult,
) -> None:
    samples, weights = weighted_results.posterior_samples(
        n_samples=2,
        rng=1234,
        method="weights",
    )

    assert samples.shape == (2, 1)
    assert np.unique(samples, axis=0).shape[0] == 2
    selected = samples[:, 0].astype(int)
    expected = np.array([0.1, 0.2, 0.3, 0.4])[selected]
    np.testing.assert_allclose(weights, expected / expected.sum())


def test_posterior_samples_multinomial_is_reproducible_and_equal_weighted(
    weighted_results: SMCResult,
) -> None:
    first_samples, first_weights = weighted_results.posterior_samples(
        n_samples=20,
        rng=1234,
        method="multinomial",
    )
    second_samples, second_weights = weighted_results.posterior_samples(
        n_samples=20,
        rng=1234,
        method="multinomial",
    )

    np.testing.assert_array_equal(first_samples, second_samples)
    np.testing.assert_array_equal(first_weights, second_weights)
    np.testing.assert_allclose(first_weights, np.full(20, 1.0 / 20.0))


def test_posterior_samples_rejection_returns_unique_equal_weight_samples(
    context_factory,
) -> None:
    state = ParticleState(particles=np.arange(4.0)[:, None])
    result = SMCResult([state], context_factory())

    samples, weights = result.posterior_samples(
        n_samples=2,
        rng=1234,
        method="rejection",
    )

    assert samples.shape == (2, 1)
    assert np.unique(samples, axis=0).shape[0] == 2
    np.testing.assert_allclose(weights, [0.5, 0.5])


@pytest.mark.parametrize("method", ["weights", "rejection"])
def test_posterior_samples_rejects_too_many_unique_samples(
    weighted_results: SMCResult,
    method: str,
) -> None:
    with pytest.raises(ValueError, match="cannot exceed the particle count"):
        weighted_results.posterior_samples(
            n_samples=5,
            method=method,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("method", ["weights", "rejection", "multinomial"])
def test_posterior_samples_requires_positive_sample_count(
    weighted_results: SMCResult,
    method: str,
) -> None:
    with pytest.raises(ValueError, match="n_samples must be positive"):
        weighted_results.posterior_samples(
            n_samples=0,
            method=method,  # type: ignore[arg-type]
        )


def test_posterior_samples_rejects_unknown_method(
    weighted_results: SMCResult,
) -> None:
    with pytest.raises(ValueError, match="Unknown posterior sampling method"):
        weighted_results.posterior_samples(method="unknown")  # type: ignore[arg-type]


def test_log_normalizing_constant_variance_dispatches_to_incremental(
    particle_state, context_factory
) -> None:
    states = [
        particle_state.with_updates(
            metadata={"log_normalizing_constant_increment_variance": variance}
        )
        for variance in (0.1, 0.2)
    ]
    result = SMCResult(states, context_factory())

    assert result.log_normalizing_constant_variance() == pytest.approx(0.3)
    assert result.log_normalizing_constant_variance(
        method="incremental"
    ) == pytest.approx(0.3)
    assert result.log_normalizing_constant_error() == pytest.approx(
        np.sqrt(0.3)
    )
    assert result.log_evidence_variance() == pytest.approx(0.3)
    assert result.log_evidence_error() == pytest.approx(np.sqrt(0.3))


def test_lee_whiteley_log_evidence_variance(context_factory) -> None:
    ancestors = (
        np.array([0, 0, 2, 3]),
        np.array([0, 1, 1, 3]),
    )
    initial_ancestor_indices = np.arange(4)
    states = []
    for indices in ancestors:
        initial_ancestor_indices = initial_ancestor_indices[indices]
        states.append(
            ParticleState(
                particles=np.zeros((4, 1)),
                initial_ancestor_indices=initial_ancestor_indices,
            )
        )
    outputs = tuple(
        (
            {
                "ess": 4.0,
                "resampled": True,
                "resampling_scheme": "multinomial",
                "ancestors": indices,
            },
        )
        for indices in ancestors
    )
    result = SMCResult(states, context_factory(), outputs=outputs)

    assert result.log_evidence_variance(
        method="lee-whiteley"
    ) == pytest.approx(1.0 / 9.0)
    assert result.log_evidence_error(method="lee-whiteley") == pytest.approx(
        1.0 / 3.0
    )


def test_lee_whiteley_requires_resampling_every_iteration(
    context_factory,
) -> None:
    state = ParticleState(particles=np.zeros((4, 1)))
    result = SMCResult(
        [state],
        context_factory(),
        outputs=(
            (
                {
                    "ess": 4.0,
                    "resampled": False,
                    "resampling_scheme": "systematic",
                    "ancestors": np.arange(4),
                },
            ),
        ),
    )

    with pytest.raises(ValueError, match="resampling at every iteration"):
        result.log_evidence_variance(method="lee-whiteley")


def test_lee_whiteley_requires_ancestor_indices(context_factory) -> None:
    state = ParticleState(particles=np.zeros((4, 1)))
    result = SMCResult(
        [state],
        context_factory(),
        outputs=(({"ess": 4.0, "resampled": True},),),
    )

    with pytest.raises(ValueError, match="recorded ancestor indices"):
        result.log_evidence_variance(method="lee-whiteley")


@pytest.mark.parametrize(
    ("scheme", "warning_match"),
    [("systematic", "systematic"), (None, "unknown")],
)
def test_lee_whiteley_warns_for_non_multinomial_or_unknown_resampling(
    context_factory,
    scheme: str | None,
    warning_match: str,
) -> None:
    state = ParticleState(
        particles=np.zeros((4, 1)),
        initial_ancestor_indices=np.array([0, 0, 2, 3]),
    )
    output: dict[str, object] = {
        "ess": 4.0,
        "resampled": True,
        "ancestors": np.array([0, 0, 2, 3]),
    }
    if scheme is not None:
        output["resampling_scheme"] = scheme
    result = SMCResult(
        [state],
        context_factory(),
        outputs=((output,),),
    )

    with pytest.warns(UserWarning, match=warning_match) as warning_records:
        result.log_evidence_variance(method="lee-whiteley")

    assert len(warning_records) == 1


def test_fixed_lag_lee_whiteley_warns_for_non_multinomial_resampling(
    context_factory,
) -> None:
    state = ParticleState(particles=np.zeros((4, 1)))
    result = SMCResult(
        [state],
        context_factory(),
        outputs=(
            (
                {
                    "ess": 4.0,
                    "resampled": True,
                    "resampling_scheme": "stratified",
                    "ancestors": np.array([0, 0, 2, 3]),
                },
            ),
        ),
    )

    with pytest.warns(UserWarning, match="stratified") as warning_records:
        result.log_evidence_variance(
            method="fixed-lag-lee-whiteley",
            lag=1,
        )

    assert len(warning_records) == 1


def test_fixed_lag_lee_whiteley_counts_resampling_events(
    context_factory,
) -> None:
    ancestors = (
        np.array([0, 0, 2, 3]),
        None,
        np.array([0, 1, 1, 3]),
        np.array([0, 0, 2, 3]),
    )
    states = [ParticleState(particles=np.zeros((4, 1))) for _ in ancestors]
    outputs = tuple(
        (
            {
                "ess": 4.0,
                "resampled": indices is not None,
                "resampling_scheme": "multinomial",
                "ancestors": (np.arange(4) if indices is None else indices),
            },
        )
        for indices in ancestors
    )
    result = SMCResult(states, context_factory(), outputs=outputs)

    assert result.log_evidence_variance(
        method="fixed-lag-lee-whiteley", lag=0
    ) == pytest.approx(0.0)
    assert result.log_evidence_variance(
        method="fixed-lag-lee-whiteley", lag=1
    ) == pytest.approx(-1.0 / 9.0)
    assert result.log_evidence_variance(
        method="fixed-lag-lee-whiteley", lag=3
    ) == pytest.approx(-5.0 / 27.0)
    with pytest.raises(ValueError, match="variance is negative"):
        result.log_evidence_error(method="fixed-lag-lee-whiteley", lag=1)

    diagnostics = result.fixed_lag_lee_whiteley_diagnostics(lag=1)
    assert diagnostics.requested_lag == 1
    assert diagnostics.traced_resampling_events == 1
    assert diagnostics.total_resampling_events == 3
    assert diagnostics.distinct_ancestors == 3
    assert diagnostics.family_ess == pytest.approx(8.0 / 3.0)


def test_fixed_lag_lee_whiteley_full_lag_matches_lee_whiteley(
    context_factory,
) -> None:
    ancestors = (
        np.array([0, 0, 2, 3]),
        np.array([0, 1, 1, 3]),
    )
    initial_ancestor_indices = np.arange(4)
    states = []
    for indices in ancestors:
        initial_ancestor_indices = initial_ancestor_indices[indices]
        states.append(
            ParticleState(
                particles=np.zeros((4, 1)),
                initial_ancestor_indices=initial_ancestor_indices,
            )
        )
    outputs = tuple(
        (
            {
                "ess": 4.0,
                "resampled": True,
                "resampling_scheme": "multinomial",
                "ancestors": indices,
            },
        )
        for indices in ancestors
    )
    result = SMCResult(states, context_factory(), outputs=outputs)

    fixed_lag = result.log_evidence_variance(
        method="fixed-lag-lee-whiteley",
        lag=100,
    )
    assert fixed_lag == pytest.approx(
        result.log_evidence_variance(method="lee-whiteley")
    )


@pytest.mark.parametrize("lag", [-1, 1.5, True])
def test_fixed_lag_lee_whiteley_rejects_invalid_lag(
    dummy_results: SMCResult,
    lag,
) -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        dummy_results.log_evidence_variance(
            method="fixed-lag-lee-whiteley",
            lag=lag,
        )


def test_fixed_lag_lee_whiteley_requires_lag(dummy_results: SMCResult) -> None:
    with pytest.raises(ValueError, match="requires a non-negative lag"):
        dummy_results.log_evidence_variance(method="fixed-lag-lee-whiteley")


def test_fixed_lag_lee_whiteley_handles_no_resampling_and_terminal_weights(
    context_factory,
) -> None:
    state = ParticleState(
        particles=np.zeros((2, 1)),
        log_weights=np.log([0.75, 0.25]),
    )
    result = SMCResult(
        [state],
        context_factory(),
        outputs=(
            (
                {
                    "ess": 1.6,
                    "resampled": False,
                    "resampling_scheme": "systematic",
                },
                None,
            ),
        ),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        variance = result.log_evidence_variance(
            method="fixed-lag-lee-whiteley", lag=4
        )

    assert variance == pytest.approx(0.25)
    assert result.log_evidence_error(
        method="fixed-lag-lee-whiteley", lag=4
    ) == pytest.approx(0.5)
    diagnostics = result.fixed_lag_lee_whiteley_diagnostics(lag=4)
    assert diagnostics.traced_resampling_events == 0
    assert diagnostics.total_resampling_events == 0
    assert diagnostics.distinct_ancestors == 2
    assert diagnostics.family_ess == pytest.approx(1.6)


def test_log_normalizing_constant_variance_rejects_unknown_method(
    dummy_results: SMCResult,
) -> None:
    with pytest.raises(
        ValueError, match="Unknown log normalizing-constant variance method"
    ):
        dummy_results.log_normalizing_constant_variance(
            method="unknown"  # type: ignore[arg-type]
        )


def test_get_meta_smc_results(dummy_results: SMCResult) -> None:
    dummy_results = SMCResult(
        states=dummy_results.states,
        context=dummy_results.context,
        outputs=(((None, {META_SMC_OUTPUT_KEY: dummy_results}),),),
    )

    meta_result = dummy_results.get_meta_results()

    assert isinstance(meta_result, MetaSMCResult)


def test_smc_results_plot_metadata_history(dummy_results: SMCResult) -> None:
    import matplotlib.pyplot as plt

    ax = dummy_results.plot_metadata_history("beta", default=0.0)
    plt.close(ax.get_figure())


def test_smc_results_plot_corner(dummy_results: SMCResult) -> None:
    import matplotlib.pyplot as plt

    ax = dummy_results.plot_corner()
    plt.close(ax.get_figure())
