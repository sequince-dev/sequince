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
        outdir: str | None = None,
        filename: str = "trace.png",
        plot_kwargs: dict[str, Any] | None = None,
    ):
        self.parameter_inds = parameter_inds
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
