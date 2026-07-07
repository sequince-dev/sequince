from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
from numpy.random import Generator as NumPyGenerator

from ._array_api import asarray, namespace, to_numpy
from ._typing import Array
from .state import ParticleState
from .steps.context import StepContext
from .steps.nested import NESTED_SMC_OUTPUT_KEY
from .utils import resolve_rng

if TYPE_CHECKING:
    from orng import ArrayRNG


def _prepare_axis(ax: Any, **subplot_kwargs: Any) -> Any:
    if ax is not None:
        return ax
    import matplotlib.pyplot as plt  # Imported lazily to avoid hard dependency at import time.

    _, ax = plt.subplots(**subplot_kwargs)
    return ax


@dataclass(frozen=True)
class SMCResult:
    """Lightweight container exposing diagnostics for a completed SMC run."""

    states: Sequence[ParticleState]
    context: StepContext

    def __post_init__(self) -> None:
        if not self.states:
            raise ValueError("SMCResults requires at least one ParticleState.")

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, index: int) -> ParticleState:
        return self.states[index]

    def append_state(self, state: ParticleState) -> SMCResult:
        """Return a new result with one additional state appended."""
        return SMCResult(
            states=[*self.states, state],
            context=self.context,
        )

    def snapshot(self) -> dict[str, Any]:
        """Return a detached snapshot suitable for checkpoint payloads."""
        return {
            "states": [state.snapshot() for state in self.states],
            "context": self.context.snapshot(),
        }

    @classmethod
    def from_snapshot(cls, payload: Mapping[str, Any]) -> "SMCResult":
        """Reconstruct a result from :meth:`snapshot` output."""
        if not isinstance(payload, Mapping):
            raise TypeError("SMCResult snapshot must be a mapping.")
        missing = {key for key in ("states", "context") if key not in payload}
        if missing:
            raise ValueError(
                "SMCResult snapshot is missing required keys: "
                + ", ".join(sorted(missing))
            )
        states_payload = payload["states"]
        if not isinstance(states_payload, Sequence):
            raise TypeError("SMCResult snapshot 'states' must be a sequence.")
        return cls(
            states=[
                ParticleState.from_snapshot(state_payload)
                for state_payload in states_payload
            ],
            context=StepContext.from_snapshot(
                copy.deepcopy(payload["context"])
            ),
        )

    def get_nested_results(
        self,
        key: str = NESTED_SMC_OUTPUT_KEY,
    ) -> NestedSMCResult | None:
        """Return a NestedSMCResult if the specified key is present in the
        context outputs.
        """
        nested_results = self.context.outputs.get(key)
        if nested_results is not None:
            return NestedSMCResult(results=nested_results)
        return None

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
        final_state = self.final_state
        xp = namespace(final_state.particles, final_state.weights)
        particles = xp.atleast_2d(final_state.particles)
        if n_samples is None:
            n_samples = particles.shape[0]
        if n_samples <= 0:
            raise ValueError("n_samples must be positive.")
        rng_obj: ArrayRNG = resolve_rng(rng)
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
        xp = namespace(state.particles)
        particles = xp.atleast_2d(state.particles)
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
        xp = namespace(state.particles)
        particles = xp.atleast_2d(state.particles)
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
        xp = namespace(state.particles)
        particles = xp.atleast_2d(state.particles)
        xp = namespace(particles)
        values = asarray(fn(particles), xp=xp, like=particles)
        weights = asarray(state.weights, xp=xp)
        if values.ndim == 1:
            return xp.sum(weights * values)
        reshape_dims = (weights.shape[0],) + (1,) * (values.ndim - 1)
        expanded_weights = xp.reshape(weights, reshape_dims)
        return xp.sum(expanded_weights * values, axis=0)

    def plot_log_evidence(
        self, *, ax: Any = None, label_axes: bool = True, **plot_kwargs: Any
    ) -> Any:
        """Plot the log evidence trajectory."""
        x = to_numpy(range(len(self)))
        y = to_numpy(self.log_evidence_history())
        axis = _prepare_axis(ax)
        axis.plot(x, y, **plot_kwargs)
        if label_axes:
            axis.set_xlabel("Iteration")
            axis.set_ylabel("Log evidence")
            axis.set_title("Log evidence trajectory")
        return axis

    def plot_effective_sample_size(
        self,
        *,
        normalize: bool = False,
        ax: Any = None,
        label_axes: bool = True,
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
        axis = _prepare_axis(ax)
        axis.plot(x, y, **plot_kwargs)
        if label_axes:
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
        label_axes: bool = True,
        **plot_kwargs: Any,
    ) -> Any:
        """Plot the evolution of a metadata entry stored in each state."""
        x = to_numpy(range(len(self)))
        y = to_numpy(self.metadata_history(key, default=default))
        axis = _prepare_axis(ax)
        axis.plot(x, y, **plot_kwargs)
        if label_axes:
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
        filename: str | Path | None = None,
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
        from .plot import plot_corner

        state = self.states[state_index]
        return plot_corner(
            state.particles,
            weights=state.weights if include_weights else None,
            parameter_inds=parameter_inds,
            filename=filename,
            **corner_kwargs,
        )

    def plot_trace(
        self,
        *,
        ax: Any = None,
        parameter_inds: Sequence[int] | None = None,
        parameter_labels: Sequence[str] | None = None,
        filename: str | Path | None = None,
        kind: str = "violin",
        **plot_kwargs: Any,
    ) -> Any:
        """
        Plot a trace plot of the particles as a function of state index.

        Parameters
        ----------
        ax
            Axis on which to plot.  If ``None``, a new figure and axis are created.
        parameter_inds
            Indices of the parameters to include in the trace plot.
            If ``None`` (default) all parameters are included.
        parameter_labels
            Labels for the parameters to include in the trace plot.
            If ``None`` (default) generic labels `{x_i}` are used.
        filename
            If provided, the plot is saved to this file instead of being returned.
        kind
            The kind of trace plot to create.  Currently only "violin" or "scatter" are supported.
        plot_kwargs
            Additional keyword arguments passed to the `ax.plot` function.

        Returns
        -------
        ax
            The trace plot axis, if `filename` is not provided.
        """
        import matplotlib.pyplot as plt

        assert kind in ("violin", "scatter"), (
            "Only 'violin' and 'scatter' kinds are supported."
        )

        default_kwargs: dict[str, Any] = dict()

        if kind == "scatter":
            default_kwargs.update(
                dict(
                    marker=",",
                    linestyle="",
                )
            )
        elif kind == "violin":
            default_kwargs.update(
                dict(
                    side="high",
                    widths=2,
                    showextrema=False,
                )
            )

        if plot_kwargs:
            default_kwargs.update(plot_kwargs)

        all_particles = np.array(
            [np.atleast_2d(to_numpy(state.particles)) for state in self.states]
        )  # (n_iterations, n_particles, n_parameters)
        n_iterations = len(all_particles)

        if parameter_inds is None:
            parameter_inds = list(range(all_particles.shape[2]))

        if parameter_labels is None:
            parameter_labels = [f"x_{i}" for i in parameter_inds]

        assert len(parameter_inds) == len(parameter_labels)

        axis = _prepare_axis(
            ax,
            nrows=len(parameter_inds),
            figsize=(5, 3 * len(parameter_inds)),
            sharex=True,
        )

        if len(parameter_inds) == 1:
            axis = [axis]

        if kind == "scatter":
            x_iter = np.repeat(np.arange(n_iterations), all_particles.shape[1])
            all_particles = np.array(all_particles)

        for i, param_ind in enumerate(parameter_inds):
            if kind == "scatter":
                axis[i].plot(
                    x_iter,
                    all_particles[:, :, param_ind].flatten(),
                    **default_kwargs,
                )
            elif kind == "violin":
                axis[i].violinplot(
                    all_particles[:, :, param_ind].T,
                    **default_kwargs,
                )

            axis[i].set_ylabel(parameter_labels[i])

        axis[-1].set_xlabel("Iteration")

        if filename is not None:
            plt.savefig(filename, bbox_inches="tight")
            plt.close()
        else:
            return ax.get_figure()


@dataclass(frozen=True)
class NestedSMCResult:
    """Aggregate results from a SMC run with a nested SMC step."""

    results: Sequence[SMCResult]

    def __post_init__(self) -> None:
        if not self.results:
            raise ValueError(
                "NestedSMCResult requires at least one nested step result."
            )

    def __len__(self) -> int:
        return len(self.results)

    def __getitem__(self, index: int) -> SMCResult:
        return self.results[index]

    @property
    def final_result(self) -> SMCResult:
        """Return the final meta-step result."""
        return self.results[-1]

    @property
    def final_state(self) -> ParticleState:
        """Return the final particle state from the last meta step."""
        return self.final_result.final_state

    @property
    def all_states(self) -> list[ParticleState]:
        """Flatten particle states across all meta steps in execution order."""
        return [state for result in self.results for state in result.states]

    def log_evidence_history(self) -> Array:
        """Return the log evidence trajectory across all meta steps."""
        reference = self.final_state.log_weights
        xp = namespace(reference)
        values = [float(state.log_evidence) for state in self.all_states]
        return asarray(values, xp=xp, like=reference)

    def plot_log_evidence(
        self,
        *,
        ax: Any = None,
        label_axes: bool = True,
        cmap: str = "viridis",
        **plot_kwargs: Any,
    ) -> Any:
        """Plot the log evidence trajectory across all meta steps."""
        import matplotlib.pyplot as plt

        axis = _prepare_axis(ax)
        x0 = 0.0
        colors = plt.get_cmap(cmap)(np.linspace(0, 1, len(self.results)))
        for result, color in zip(self.results, colors):
            x = np.arange(x0, x0 + len(result))
            y = to_numpy(result.log_evidence_history())
            axis.plot(x, y, color=color, **plot_kwargs)
            x0 += len(result)
        if label_axes:
            axis.set_xlabel("Iteration")
            axis.set_ylabel("Log evidence")
            axis.set_title("Log evidence trajectory across meta steps")
        return axis

    def plot_metadata_history(
        self,
        key: str,
        *,
        ax: Any = None,
        default: Any = math.nan,
        label_axes: bool = True,
        cmap: str = "viridis",
        **plot_kwargs: Any,
    ) -> Any:
        """Plot metadata history across all meta steps."""
        import matplotlib.pyplot as plt

        axis = _prepare_axis(ax)
        x0 = 0.0

        colors = plt.get_cmap(cmap)(np.linspace(0, 1, len(self.results)))
        for result, color in zip(self.results, colors):
            x = np.arange(x0, x0 + len(result))
            y = to_numpy(result.metadata_history(key, default=default))
            axis.plot(x, y, color=color, **plot_kwargs)
            x0 += len(result)
        if label_axes:
            axis.set_xlabel("Iteration")
            axis.set_ylabel(key)
            axis.set_title(f"Metadata '{key}' over iterations")
        return axis

    def summary(self) -> dict[str, Any]:
        """Return a concise dictionary of high-level diagnostics."""
        summary_dict: dict[str, Any] = {
            "meta_steps": len(self.results),
            "total_iterations": len(self.all_states),
            "final_log_evidence": float(self.final_state.log_evidence),
        }
        try:
            ess = float(self.final_state.effective_sample_size())
            summary_dict["final_ess"] = ess
            summary_dict["final_ess_ratio"] = float(
                ess / self.final_state.particles.shape[0]
            )
        except Exception:
            pass
        return summary_dict
