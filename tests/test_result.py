import numpy as np
import pytest

from sequince.result import MetaSMCResult, SMCResult
from sequince.state import ParticleState
from sequince.steps.meta import META_SMC_OUTPUT_KEY


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


def test_smc_results_append_state(
    dummy_results: SMCResult, particle_state
) -> None:
    updated = dummy_results.append_state(particle_state)

    assert len(updated) == len(dummy_results) + 1
    assert updated.final_state == particle_state
    assert len(dummy_results) == 10


def test_smc_results_summary(dummy_results: SMCResult) -> None:
    dummy_results.summary()


def test_log_evidence_variance_dispatches_to_incremental(
    particle_state, context_factory
) -> None:
    states = [
        particle_state.with_updates(
            metadata={"log_evidence_increment_variance": variance}
        )
        for variance in (0.1, 0.2)
    ]
    result = SMCResult(states, context_factory())

    assert result.log_evidence_variance() == pytest.approx(0.3)
    assert result.log_evidence_variance(method="incremental") == pytest.approx(
        0.3
    )
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
        ({"ess": 4.0, "resampled": True, "ancestors": indices},)
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
        ({"ess": 4.0, "resampled": True, "ancestors": indices},)
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
        outputs=(({"ess": 1.6, "resampled": False}, None),),
    )

    assert result.log_evidence_variance(
        method="fixed-lag-lee-whiteley", lag=4
    ) == pytest.approx(0.25)
    assert result.log_evidence_error(
        method="fixed-lag-lee-whiteley", lag=4
    ) == pytest.approx(0.5)
    diagnostics = result.fixed_lag_lee_whiteley_diagnostics(lag=4)
    assert diagnostics.traced_resampling_events == 0
    assert diagnostics.total_resampling_events == 0
    assert diagnostics.distinct_ancestors == 2
    assert diagnostics.family_ess == pytest.approx(1.6)


def test_log_evidence_variance_rejects_unknown_method(
    dummy_results: SMCResult,
) -> None:
    with pytest.raises(
        ValueError, match="Unknown log-evidence variance method"
    ):
        dummy_results.log_evidence_variance(method="unknown")  # type: ignore[arg-type]


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

    ax = dummy_results.plot_metadata_history("temperature", default=0.0)
    plt.close(ax.get_figure())


def test_smc_results_plot_corner(dummy_results: SMCResult) -> None:
    import matplotlib.pyplot as plt

    ax = dummy_results.plot_corner()
    plt.close(ax.get_figure())
