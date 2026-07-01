"""Plotting utilities for sequince."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from ._array_api import namespace, to_numpy
from ._typing import Array


def plot_chains(
    chains: Array,
    n_chains: int | None = None,
    filename: str | Path | None = None,
    parameter_labels: Sequence[str] | None = None,
    parameter_inds: Sequence[int] | None = None,
    **kwargs,
):
    """Plot MCMC chains.

    Parameters
    ----------
    chains : Array
        The MCMC chains to plot, of shape (n_steps, n_chains, n_params).
    n_chains : int, optional
        The number of chains to plot. If None, all chains are plotted.
    filename : str | Path, optional
        If provided, the plot will be saved to this filename. If None, the plot will be returned as a matplotlib figure.
    **kwargs : additional keyword arguments
        Additional keyword arguments to pass to the plot function.
    """

    if chains.ndim != 3:
        raise ValueError(
            "Chains must be a 3D array of shape (n_steps, n_chains, n_params)"
        )

    if parameter_inds is not None:
        chains = chains[:, :, parameter_inds]

    n_params = chains.shape[2]

    if n_chains is None:
        n_chains = chains.shape[1]

    fig, axs = plt.subplots(
        n_params, 1, figsize=(8, 2 * n_params), sharex=True
    )
    if n_params == 1:
        axs = [axs]

    kwargs.setdefault("lw", 0.5)
    kwargs.setdefault("alpha", 0.7)

    if parameter_labels is None:
        parameter_labels = [f"$x_{i}$" for i in range(n_params)]

    its = np.arange(chains.shape[0])

    for i in range(n_params):
        for j in range(n_chains):
            axs[i].plot(its, chains[:, j, i], **kwargs)
        axs[i].set_ylabel(parameter_labels[i])
    axs[-1].set_xlabel("Mutation iteration")
    axs[-1].set_xlim(0, chains.shape[0] - 1)

    if filename is not None:
        plt.savefig(filename)
        plt.close(fig)
    else:
        return fig


def plot_corner(
    particles: Array,
    weights: Array | None = None,
    parameter_inds: Sequence[int] | None = None,
    parameter_labels: Sequence[str] | None = None,
    include_weights: bool = True,
    filename: str | Path | None = None,
    **kwargs,
):
    """Plot a corner plot of the particles.

    Parameters
    ----------
    particles : Array
        The particles to plot, of shape (n_particles, n_dimensions).
    weights : Array, optional
        The weights of the particles, of shape (n_particles,). If None, uniform weights are assumed.
    parameter_inds : Sequence[int], optional
        Indices of the parameters to plot. If None, all parameters are plotted.
    parameter_labels : Sequence[str], optional
        Labels for the parameters. If None, default labels are used.
    include_weights : bool, optional
        Whether to include weights in the corner plot. Default is True.
    filename : str | Path, optional
        If provided, the plot will be saved to this filename. If None, the plot will be returned as a matplotlib figure.
    **kwargs : additional keyword arguments
        Additional keyword arguments to pass to the corner plot function.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The corner plot figure, if :code:`filename` is not provided.
    """
    from corner import corner

    xp = namespace(particles)
    particles = to_numpy(xp.atleast_2d(particles))

    default_kwargs = dict(
        bins=32,
        smooth=0.9,
        color="#EFBF35",
        truth_color="#6B8E23",
        quantiles=[0.16, 0.5, 0.84],
        levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.0)),
        plot_density=True,
        plot_datapoints=True,
        fill_contours=True,
        show_titles=True,
        hist_kwargs=dict(density=True, linewidth=2),
        weights=weights if include_weights else None,
    )

    if parameter_inds is not None:
        particles = particles[:, parameter_inds]

    if parameter_labels is None:
        parameter_labels = [f"$x_{i}$" for i in range(particles.shape[1])]

    default_kwargs.update(kwargs)

    fig = corner(particles, labels=parameter_labels, **default_kwargs)
    if filename is not None:
        fig.savefig(filename)
        plt.close(fig)
    else:
        return fig
