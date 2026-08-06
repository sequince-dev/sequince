import pytest

from sequince.result import MetaSMCResult, SMCResult
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
