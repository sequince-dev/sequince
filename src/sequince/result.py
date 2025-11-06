from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from .state import ParticleState
from .utils import _ensure_rng


def _as_2d(array: np.ndarray) -> np.ndarray:
    """Ensure particle arrays are 2-D with rows corresponding to samples."""
    arr = np.asarray(array)
    if arr.ndim == 1:
        return arr[:, np.newaxis]
    return arr


@dataclass(frozen=True)
class SMCResults:
    """Lightweight container exposing diagnostics for a completed SMC run."""

    states: Sequence[ParticleState]

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

    def log_evidence_history(self) -> np.ndarray:
        """Return the log evidence trajectory."""
        return np.array([float(state.log_evidence) for state in self.states])

    def effective_sample_sizes(self) -> np.ndarray:
        """Return the effective sample size (ESS) for every iteration."""
        return np.array(
            [float(state.effective_sample_size()) for state in self.states]
        )

    def effective_sample_size_ratios(self) -> np.ndarray:
        """Return ESS / n_particles for each iteration."""
        return np.array(
            [
                float(state.effective_sample_size() / state.particles.shape[0])
                for state in self.states
            ]
        )

    def metadata_history(
        self, key: str, *, default: Any = np.nan
    ) -> np.ndarray:
        """Return an array containing a metadata field pulled from each state."""
        return np.array(
            [state.metadata.get(key, default) for state in self.states],
            dtype=float if isinstance(default, (int, float)) else object,
        )

    def summary(self) -> dict[str, Any]:
        """Return a concise dictionary of high-level diagnostics."""
        ess = self.effective_sample_sizes()
        ess_ratio = self.effective_sample_size_ratios()
        log_evidence = self.log_evidence_history()
        summary_dict: dict[str, Any] = {
            "iterations": len(self),
            "final_ess": ess[-1],
            "final_ess_ratio": ess_ratio[-1],
            "final_log_evidence": log_evidence[-1],
        }

        temperature_history = self.metadata_history(
            "temperature", default=np.nan
        )
        summary_dict["final_temperature"] = temperature_history[-1]
        return summary_dict

    def posterior_samples(
        self,
        n_samples: int | None = None,
        *,
        rng: np.random.Generator | int | None = None,
        with_replacement: bool = True,
    ) -> np.ndarray:
        """Draw samples from the weighted posterior represented by the final state."""
        if n_samples is None:
            return _as_2d(self.final_state.particles)
        if n_samples <= 0:
            raise ValueError("n_samples must be positive.")
        rng = _ensure_rng(rng)
        final_state = self.final_state
        particles = _as_2d(final_state.particles)
        weights = np.asarray(final_state.weights, dtype=float)
        indices = rng.choice(
            particles.shape[0],
            size=n_samples,
            replace=with_replacement,
            p=weights,
        )
        return particles[indices]

    def weighted_mean(
        self,
        *,
        state_index: int = -1,
    ) -> np.ndarray:
        """Return the weighted mean of the specified state (defaults to final)."""
        state = self.states[state_index]
        particles = _as_2d(state.particles)
        weights = state.weights[:, np.newaxis]
        return np.sum(weights * particles, axis=0)

    def weighted_covariance(
        self,
        *,
        state_index: int = -1,
        ddof: int = 0,
    ) -> np.ndarray:
        """Return the weighted covariance matrix for the specified state."""
        state = self.states[state_index]
        particles = _as_2d(state.particles)
        weights = np.asarray(state.weights, dtype=float)
        mean = self.weighted_mean(state_index=state_index)
        centered = particles - mean
        cov = centered.T @ (centered * weights[:, np.newaxis])
        weight_sum = weights.sum()
        if ddof != 0:
            weight_sum = max(weight_sum - ddof, np.finfo(float).eps)
        return cov / weight_sum

    def apply_weighted(
        self,
        fn: Callable[[np.ndarray], np.ndarray],
        *,
        state_index: int = -1,
    ) -> np.ndarray:
        """Evaluate a function under the weighted empirical distribution."""
        state = self.states[state_index]
        particles = _as_2d(state.particles)
        values = np.asarray(fn(particles))
        weights = state.weights
        if values.ndim == 1:
            return np.sum(weights * values)
        return np.tensordot(weights, values, axes=(0, 0))

    def plot_log_evidence(self, *, ax: Any = None, **plot_kwargs: Any) -> Any:
        """Plot the log evidence trajectory."""
        x = np.arange(len(self))
        y = self.log_evidence_history()
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
        x = np.arange(len(self))
        if normalize:
            y = self.effective_sample_size_ratios()
            ylabel = "ESS / n"
            title = "Effective sample size ratio"
        else:
            y = self.effective_sample_sizes()
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
        default: Any = np.nan,
        **plot_kwargs: Any,
    ) -> Any:
        """Plot the evolution of a metadata entry stored in each state."""
        x = np.arange(len(self))
        y = self.metadata_history(key, default=default)
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
