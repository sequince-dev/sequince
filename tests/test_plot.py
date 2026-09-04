from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

from sequince.plot import plot_chains, plot_corner


@pytest.fixture
def chain() -> np.ndarray:
    return np.arange(4 * 3 * 2, dtype=float).reshape(4, 3, 2)


@pytest.fixture
def weights(simple_particles: np.ndarray) -> np.ndarray:
    values = np.arange(1, simple_particles.shape[0] + 1, dtype=float)
    return values / values.sum()


@pytest.fixture
def corner_patch(monkeypatch):
    import corner

    calls: dict[str, object] = {}

    def fake_corner(particles, labels, **kwargs):
        calls["particles"] = particles
        calls["labels"] = labels
        calls["kwargs"] = kwargs
        fig, _ = plt.subplots()
        return fig

    monkeypatch.setattr(corner, "corner", fake_corner)
    return calls


def test_plot_chains_raises_for_non_3d_input() -> None:
    with pytest.raises(ValueError, match="Chains must be a 3D array"):
        plot_chains(np.zeros((4, 2)))


def test_plot_chains_returns_figure_for_selected_parameter(
    chain: np.ndarray,
) -> None:
    fig = plot_chains(
        chain,
        n_chains=2,
        parameter_inds=[1],
        parameter_labels=["beta"],
    )

    assert fig is not None
    assert len(fig.axes) == 1
    assert len(fig.axes[0].lines) == 2
    assert fig.axes[0].get_ylabel() == "beta"
    assert fig.axes[0].get_xlabel() == "Mutation iteration"
    assert fig.axes[0].get_xlim() == (0.0, chain.shape[0] - 1)
    np.testing.assert_array_equal(
        fig.axes[0].lines[0].get_ydata(),
        chain[:, 0, 1],
    )

    plt.close(fig)


def test_plot_chains_saves_figure(chain: np.ndarray, tmp_path) -> None:
    filename = tmp_path / "chain.png"

    result = plot_chains(chain, filename=filename)

    assert result is None
    assert filename.exists()


def test_plot_chains_marks_chain_splits(chain: np.ndarray) -> None:
    fig = plot_chains(
        chain,
        n_chains=1,
        split_indices=[2],
        split_kwargs={"color": "red"},
    )

    assert fig is not None
    for axis in fig.axes:
        assert len(axis.lines) == 2
        split_line = axis.lines[1]
        np.testing.assert_array_equal(split_line.get_xdata(), [1.5, 1.5])
        assert split_line.get_color() == "red"

    plt.close(fig)


def test_plot_corner_passes_selected_particles_and_labels(
    simple_particles: np.ndarray,
    weights: np.ndarray,
    corner_patch,
) -> None:
    fig = plot_corner(
        simple_particles,
        weights=weights,
        parameter_inds=[0],
        parameter_labels=["mass"],
        include_weights=False,
        bins=16,
    )

    assert fig is not None
    np.testing.assert_array_equal(
        corner_patch["particles"],
        simple_particles[:, [0]],
    )
    assert corner_patch["labels"] == ["mass"]
    assert corner_patch["kwargs"]["weights"] is None
    assert corner_patch["kwargs"]["bins"] == 16

    plt.close(fig)


def test_plot_corner_with_real_corner(
    simple_particles: np.ndarray,
    weights: np.ndarray,
) -> None:
    pytest.importorskip("corner")

    fig = plot_corner(
        simple_particles,
        weights=weights,
        parameter_labels=["x", "y"],
    )

    assert fig is not None
    assert len(fig.axes) > 0

    plt.close(fig)


def test_plot_corner_saves_figure_with_default_labels(
    simple_particles: np.ndarray,
    weights: np.ndarray,
    corner_patch,
    tmp_path,
) -> None:
    filename = tmp_path / "corner.png"

    result = plot_corner(simple_particles, weights=weights, filename=filename)

    assert result is None
    assert filename.exists()
    assert corner_patch["labels"] == ["$x_0$", "$x_1$"]
    np.testing.assert_array_equal(corner_patch["kwargs"]["weights"], weights)
