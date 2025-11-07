from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Sequence

import numpy as np
from numpy.random import Generator as NumPyGenerator

from ._array_api import asarray, namespace, to_numpy
from ._typing import Array
from .state import ParticleState
from .utils import _ensure_rng

if TYPE_CHECKING:
    from orng import ArrayRNG

    from .steps.base import StepContext


def _as_2d(array: Any) -> Any:
    """Ensure particle arrays are 2-D with rows corresponding to samples."""
    xp = namespace(array)
    arr = asarray(array, xp=xp)
    if arr.ndim == 1:
        arr = xp.reshape(arr, (arr.shape[0], 1))
    return arr


@dataclass(frozen=True)
class SMCResults:
    """Lightweight container exposing diagnostics for a completed SMC run."""

    states: Sequence[ParticleState]
    context: StepContext

    def __post_init__(self) -> None:
        if not self.states:
            raise ValueError("SMCResults requires at least one ParticleState.")

    @staticmethod
    def _prepare_axis(ax: Any) -> Any:
        if ax is not None:
            return ax
        import matplotlib.pyplot as plt  # Imported lazily to avoid hard dependency at import time.

        _, ax = plt.subplots()
        return ax

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, index: int) -> ParticleState:
        return self.states[index]

    @property
    def final_state(self) -> ParticleState:
        """Return the last state produced by the run."""
        return self.states[-1]

    def log_evidence_history(self) -> Array:
        """Return the log evidence trajectory."""
        reference = self.final_state.log_weights if self.states else None
        xp = namespace(reference)
        values = [float(state.log_evidence) for state in self.states]
        return asarray(values, xp=xp, like=reference)

    def effective_sample_sizes(self) -> Array:
        """Return the effective sample size (ESS) for every iteration."""
        reference = self.final_state.log_weights if self.states else None
        xp = namespace(reference)
        values = [
            float(state.effective_sample_size()) for state in self.states
        ]
        return asarray(values, xp=xp, like=reference)

    def effective_sample_size_ratios(self) -> Array:
        """Return ESS / n_particles for each iteration."""
        reference = self.final_state.log_weights if self.states else None
        xp = namespace(reference)
        values = [
            float(state.effective_sample_size() / state.particles.shape[0])
            for state in self.states
        ]
        return asarray(values, xp=xp, like=reference)

    def metadata_history(
        self, key: str, *, default: Any = math.nan
    ) -> Array | list[Any]:
        """Return an array containing a metadata field pulled from each state."""
        values = [state.metadata.get(key, default) for state in self.states]
        if isinstance(default, (int, float)):
            reference = self.final_state.log_weights if self.states else None
            xp = namespace(reference)
            return asarray(values, xp=xp, like=reference)
        return values

    def summary(self) -> dict[str, Any]:
        """Return a concise dictionary of high-level diagnostics."""
        ess = self.effective_sample_sizes()
        ess_ratio = self.effective_sample_size_ratios()
        log_evidence = self.log_evidence_history()
        summary_dict: dict[str, Any] = {
            "iterations": len(self),
            "final_ess": float(to_numpy(ess)[-1]),
            "final_ess_ratio": float(to_numpy(ess_ratio)[-1]),
            "final_log_evidence": float(to_numpy(log_evidence)[-1]),
        }

        temperature_history = self.metadata_history(
            "temperature", default=math.nan
        )
        summary_dict["final_temperature"] = float(
            to_numpy(temperature_history)[-1]
        )
        return summary_dict

    def posterior_samples(
        self,
        n_samples: int | None = None,
        *,
        rng: "ArrayRNG" | NumPyGenerator | int | None = None,
        with_replacement: bool = True,
    ) -> Array:
        """Draw samples from the weighted posterior represented by the final state."""
        if n_samples is None:
            return _as_2d(self.final_state.particles)
        if n_samples <= 0:
            raise ValueError("n_samples must be positive.")
        rng_obj: ArrayRNG = _ensure_rng(rng)
        final_state = self.final_state
        particles = _as_2d(final_state.particles)
        xp = namespace(particles, final_state.weights)
        weights = asarray(final_state.weights, xp=xp)
        indices = asarray(
            rng_obj.choice(
                particles.shape[0],
                size=n_samples,
                replace=with_replacement,
                p=weights,
            ),
            xp=xp,
            like=particles,
        )
        return particles[indices]

    def weighted_mean(
        self,
        *,
        state_index: int = -1,
    ) -> Array:
        """Return the weighted mean of the specified state (defaults to final)."""
        state = self.states[state_index]
        particles = _as_2d(state.particles)
        xp = namespace(particles)
        weights = asarray(state.weights, xp=xp)
        weights = xp.reshape(weights, (weights.shape[0], 1))
        return xp.sum(weights * particles, axis=0)

    def weighted_covariance(
        self,
        *,
        state_index: int = -1,
        ddof: int = 0,
    ) -> Array:
        """Return the weighted covariance matrix for the specified state."""
        state = self.states[state_index]
        particles = _as_2d(state.particles)
        xp = namespace(particles)
        weights = asarray(state.weights, xp=xp)
        mean = asarray(
            self.weighted_mean(state_index=state_index),
            xp=xp,
            like=particles,
        )
        centered = particles - mean
        weights_column = xp.reshape(weights, (weights.shape[0], 1))
        weighted = centered * weights_column
        cov = xp.matmul(xp.transpose(centered), weighted)
        weight_sum = float(to_numpy(xp.sum(weights)))
        if ddof != 0:
            weight_sum = max(weight_sum - ddof, 1e-12)
        return cov / weight_sum

    def apply_weighted(
        self,
        fn: Callable[[Any], Any],
        *,
        state_index: int = -1,
    ) -> Array:
        """Evaluate a function under the weighted empirical distribution."""
        state = self.states[state_index]
        particles = _as_2d(state.particles)
        xp = namespace(particles)
        values = asarray(fn(particles), xp=xp, like=particles)
        weights = asarray(state.weights, xp=xp)
        if values.ndim == 1:
            return xp.sum(weights * values)
        reshape_dims = (weights.shape[0],) + (1,) * (values.ndim - 1)
        expanded_weights = xp.reshape(weights, reshape_dims)
        return xp.sum(expanded_weights * values, axis=0)

    def plot_log_evidence(self, *, ax: Any = None, **plot_kwargs: Any) -> Any:
        """Plot the log evidence trajectory."""
        x = to_numpy(range(len(self)))
        y = to_numpy(self.log_evidence_history())
        axis = self._prepare_axis(ax)
        axis.plot(x, y, **plot_kwargs)
        axis.set_xlabel("Iteration")
        axis.set_ylabel("Log evidence")
        axis.set_title("Log evidence trajectory")
        return axis

    def plot_effective_sample_size(
        self,
        *,
        normalize: bool = False,
        ax: Any = None,
        **plot_kwargs: Any,
    ) -> Any:
        """Plot ESS or the ESS ratio across iterations."""
        x = to_numpy(range(len(self)))
        if normalize:
            y = to_numpy(self.effective_sample_size_ratios())
            ylabel = "ESS / n"
            title = "Effective sample size ratio"
        else:
            y = to_numpy(self.effective_sample_sizes())
            ylabel = "ESS"
            title = "Effective sample size"
        axis = self._prepare_axis(ax)
        axis.plot(x, y, **plot_kwargs)
        axis.set_xlabel("Iteration")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        return axis

    def plot_metadata_history(
        self,
        key: str,
        *,
        ax: Any = None,
        default: Any = math.nan,
        **plot_kwargs: Any,
    ) -> Any:
        """Plot the evolution of a metadata entry stored in each state."""
        x = to_numpy(range(len(self)))
        y = to_numpy(self.metadata_history(key, default=default))
        axis = self._prepare_axis(ax)
        axis.plot(x, y, **plot_kwargs)
        axis.set_xlabel("Iteration")
        axis.set_ylabel(key)
        axis.set_title(f"Metadata '{key}' over iterations")
        return axis

    def plot_corner(
        self,
        *,
        state_index: int = -1,
        include_weights: bool = False,
        parameter_inds: Sequence[int] | None = None,
        filename: str | None = None,
        **corner_kwargs: Any,
    ) -> Any:
        """
        Plot a corner plot of the particles in the specified state.

        Parameters
        ----------
        state_index
            Index of the state to plot (default is -1, the final state).
        include_weights
            If ``True``, include particle weights in the corner plot.
        parameter_inds
            Indices of the parameters to include in the corner plot.
            If ``None`` (default) all parameters are included.
        filename
            If provided, the plot is saved to this file instead of being returned.
        corner_kwargs
            Additional keyword arguments passed to the `corner` function.

        Returns
        -------
        fig
            The corner plot figure, if `filename` is not provided.
        """
        import matplotlib.pyplot as plt
        from corner import corner

        state = self.states[state_index]
        particles = _as_2d(state.particles)

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
            weights=state.weights if include_weights else None,
        )

        if parameter_inds is not None:
            particles = particles[:, parameter_inds]

        if corner_kwargs:
            default_kwargs.update(corner_kwargs)

        fig = corner(particles, **default_kwargs)
        if filename is not None:
            fig.savefig(filename)
            plt.close(fig)
        else:
            return fig
