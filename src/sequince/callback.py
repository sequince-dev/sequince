from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .result import SMCResults
    from .steps.base import StepContext


class SMCCallback:
    """Base class for composable SMC callbacks."""

    def __call__(
        self,
        results: "SMCResults",
        context: "StepContext",
    ) -> None:
        self.apply(results, context)

    def apply(
        self,
        results: "SMCResults",
        context: "StepContext",
    ) -> None:
        raise NotImplementedError(
            "SMCCallback subclasses must implement apply()."
        )


class TracePlot(SMCCallback):
    """
    Callback to plot 1-d particle locations every SMC iteration.

    Parameters
    ----------
    parameter_inds:
        List of parameter indices to plot.  If ``None`` (default) plot all parameters.
    outdir:
        Directory to save the trace plots.  If ``None`` (default) saves to the current
        working directory.
    filename:
        Filename for the trace plot.  Default is ``"trace.png"``.
    plot_kwargs:
        Additional keyword arguments to pass to the `SMCResults.plot_trace` method.
    """

    def __init__(
        self,
        parameter_inds: list[int] | None = None,
        parameter_labels: list[str] | None = None,
        outdir: str | None = None,
        filename: str = "trace.png",
        plot_kwargs: dict[str, Any] | None = None,
    ):
        self.parameter_inds = parameter_inds
        self.parameter_labels = parameter_labels

        if parameter_labels is not None and parameter_inds is not None:
            assert len(parameter_labels) == len(parameter_inds), (
                "parameter_labels and parameter_inds must have the same length."
            )

        self.outdir = outdir or "."
        self.filename = filename
        self.full_path = str(
            (
                Path(self.outdir) / Path(self.filename)
                if self.outdir
                else Path(self.filename)
            )
        )
        self.plot_kwargs = plot_kwargs or {}

        Path(self.outdir).mkdir(parents=True, exist_ok=True)

    def apply(
        self,
        results: "SMCResults",
        context: "StepContext",
    ) -> None:
        results.plot_trace(
            parameter_inds=self.parameter_inds,
            filename=self.full_path,
            **self.plot_kwargs,
        )


class StatePlot(SMCCallback):
    """
    Callback to plot user-specified state metadata every SMC iteration.

    Parameters
    ----------
    metadata_keys:
        List of metadata keys to plot. At least one key must be specified.
    outdir:
        Directory to save the state plots.  If ``None`` (default) saves to the current
        working directory.
    filename:
        Filename for the state plot.  Default is ``"state.png"``.
    plot_kwargs:
        Additional keyword arguments to pass to the `SMCResults.plot_metadata_history` method.
    """

    def __init__(
        self,
        metadata_keys: list[str],
        outdir: str | None = None,
        filename: str = "state.png",
        plot_kwargs: dict[str, Any] | None = None,
    ):
        assert len(metadata_keys) > 0, (
            "At least one metadata key must be specified."
        )
        self.metadata_keys = metadata_keys
        self.outdir = outdir or "."
        self.filename = filename
        self.full_path = str(
            (
                Path(self.outdir) / Path(self.filename)
                if self.outdir
                else Path(self.filename)
            )
        )
        self.plot_kwargs = plot_kwargs or {}
        Path(self.outdir).mkdir(parents=True, exist_ok=True)

    def apply(
        self,
        results: "SMCResults",
        context: "StepContext",
    ) -> None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(
            nrows=len(self.metadata_keys),
            sharex=True,
            figsize=(5, 3 * len(self.metadata_keys)),
        )

        plotting_fn = {
            "log_evidence": results.plot_log_evidence,
            "ess": results.plot_effective_sample_size,
        }

        for i, key in enumerate(self.metadata_keys):
            plot_func: Any = plotting_fn.get(
                key,
                lambda *args, **kwargs: results.plot_metadata_history(
                    key, *args, **kwargs
                ),
            )
            plot_func(
                ax=ax[i],
                label_axes=False,
                **self.plot_kwargs,
            )

            ax[i].set_ylabel(key)

        ax[-1].set_xlabel("SMC Iteration")

        plt.tight_layout()
        fig.savefig(self.full_path)
        plt.close(fig)
