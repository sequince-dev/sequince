import pytest

from sequince.result import SMCResults


@pytest.fixture
def dummy_results(particle_state):
    return SMCResults(
        [
            particle_state,
        ]
        * 10
    )


def test_smc_results_length(dummy_results: SMCResults) -> None:
    assert len(dummy_results) == 10


def test_smc_results_indexing(
    dummy_results: SMCResults, particle_state
) -> None:
    for i in range(10):
        assert dummy_results[i] == particle_state

    assert dummy_results.final_state == particle_state


def test_smc_results_summary(dummy_results: SMCResults) -> None:
    dummy_results.summary()


def test_smc_results_plot_metadata_history(dummy_results: SMCResults) -> None:
    import matplotlib.pyplot as plt

    ax = dummy_results.plot_metadata_history("temperature", default=0.0)
    plt.close(ax.get_figure())


def test_smc_results_plot_corner(dummy_results: SMCResults) -> None:
    import matplotlib.pyplot as plt

    ax = dummy_results.plot_corner()
    plt.close(ax.get_figure())
