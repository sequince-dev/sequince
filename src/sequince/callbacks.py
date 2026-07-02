from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .steps.mutation import PARTICLE_TRANSFORM_KEY

if TYPE_CHECKING:
    from .result import SMCResult
    from .state import ParticleState
    from .steps.base import StepContext


logger = logging.getLogger(__name__)


class SMCCallback:
    """Base class for composable SMC callbacks."""

    def __call__(
        self,
        state: "ParticleState",
        context: "StepContext",
        results: "SMCResult",
    ) -> None:
        self.apply(state, context, results)

    def apply(
        self,
        state: "ParticleState",
        context: "StepContext",
        results: "SMCResult",
    ) -> None:
        raise NotImplementedError(
            "SMCCallback subclasses must implement apply()."
        )


class PlotCallback(SMCCallback):
    """
    Base class for SMC callbacks that generate plots.

    Parameters
    ----------
    outdir:
        Directory to save the plots.  If ``None`` (default) saves to the current
        working directory.
    filename:
        Filename for the plot.
    plot_kwargs:
        Additional keyword arguments to pass to the plotting method.
    """

    def __init__(
        self,
        outdir: str | None = None,
        filename: str = "plot.png",
        plot_kwargs: dict[str, Any] | None = None,
    ):
        self.outdir = Path(outdir or ".")
        self.filename = Path(filename)
        self.full_path = self.outdir / self.filename
        self.plot_kwargs = plot_kwargs or {}

        Path(self.outdir).mkdir(parents=True, exist_ok=True)


class TracePlot(PlotCallback):
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
        super().__init__(outdir, filename, plot_kwargs)

        self.parameter_inds = parameter_inds
        self.parameter_labels = parameter_labels

        if parameter_labels is not None and parameter_inds is not None:
            if len(parameter_labels) != len(parameter_inds):
                raise ValueError(
                    "parameter_labels and parameter_inds must have the same length."
                )

    def apply(
        self,
        state: "ParticleState",
        context: "StepContext",
        results: "SMCResult",
    ) -> None:
        results.plot_trace(
            parameter_inds=self.parameter_inds,
            parameter_labels=self.parameter_labels,
            filename=self.full_path,
            **self.plot_kwargs,
        )


class IntermediateParticlesPlot(PlotCallback):
    """Plot the intermediate particles at each SMC iteration.

    Plots a corner-style plot of the particles at each SMC iteration.

    Parameters
    ----------
    outdir:
        Directory to save the intermediate particle plots.  If ``None``
        (default) saves to the current working directory.
    base_filename:
        Base filename for the intermediate particle plots.  The iteration number
        will be appended to the base filename.  Default is ``"particles"``.
    plot_kwargs:
        Additional keyword arguments to pass to the `plot_corner` function.
    """

    def __init__(
        self,
        outdir: str | None = None,
        base_filename: str = "particles",
        parameter_inds: list[int] | None = None,
        parameter_labels: list[str] | None = None,
        plot_kwargs: dict[str, Any] | None = None,
    ):
        filename = f"{base_filename}_0.png"
        self.base_filename = base_filename
        self.parameter_labels = parameter_labels
        self.parameter_inds = parameter_inds
        super().__init__(outdir, filename, plot_kwargs)

    def apply(
        self,
        state: "ParticleState",
        context: "StepContext",
        results: "SMCResult",
    ) -> None:
        from .plot import plot_corner

        filename = f"{self.base_filename}_{context.iteration}.png"
        self.full_path = self.outdir / filename
        logger.debug(f"Plotting intermediate particles to {self.full_path}")
        plot_corner(
            state.particles,
            filename=self.full_path,
            **self.plot_kwargs,
        )


class StatePlot(PlotCallback):
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
        super().__init__(outdir, filename, plot_kwargs)
        if len(metadata_keys) == 0:
            raise ValueError("At least one metadata key must be specified.")
        self.metadata_keys = metadata_keys

    def apply(
        self,
        state: "ParticleState",
        context: "StepContext",
        results: "SMCResult",
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


class MutationChainPlot(PlotCallback):
    """
    Callback to plot the chains from the mutation step of the SMC algorithm.

    Parameters
    ----------
    outdir:
        Directory to save the mutation chains plots.  If ``None`` (default) saves
        to the current working directory.
    base_filename:
        Base filename for the mutation chains plots.  The iteration number will
        be appended to the base filename.  Default is ``"chain"``.
    plot_kwargs:
        Additional keyword arguments to pass to the `plot_chain` function.
    """

    def __init__(
        self,
        outdir: str | None = None,
        base_filename: str = "chain",
        parameter_inds: list[int] | None = None,
        parameter_labels: list[str] | None = None,
        plot_kwargs: dict[str, Any] | None = None,
        metadata_key: str = "mutation_chains",
    ):
        self.base_filename = base_filename
        self.parameter_labels = parameter_labels
        self.parameter_inds = parameter_inds
        self.metadata_key = metadata_key
        filename = f"{base_filename}_0.png"
        super().__init__(outdir, filename, plot_kwargs)

    def apply(
        self,
        state: "ParticleState",
        context: "StepContext",
        results: "SMCResult",
    ) -> None:
        chains = state.metadata.get(self.metadata_key)
        if chains is not None:
            filename = f"{self.base_filename}_{context.iteration}.png"
            self.full_path = self.outdir / filename
            logger.debug(f"Plotting mutation chains to {self.full_path}")
            from .plot import plot_chains

            plot_chains(
                chains,
                filename=self.full_path,
                parameter_labels=self.parameter_labels,
                parameter_inds=self.parameter_inds,
                **self.plot_kwargs,
            )
        else:
            logger.debug("No mutation chains found in state metadata.")


class PlotParticleTransform(PlotCallback):
    """
    Callback to plot the active particle transform at each SMC iteration.

    Parameters
    ----------
    outdir:
        Directory to save the active particle transform plots.  If ``None``
        (default) saves to the current working directory.
    base_filename:
        Base filename for the active particle transform plots. The iteration
        number is appended to the base filename. Default is
        :code:``"particle_transform"``.
    metadata_key:
        Key in the state metadata where the active particle transform is stored.
        Default is :code:``"particle_transform"``.
    plot_kwargs:
        Additional keyword arguments to pass to the `plot_transform` function.
    """

    def __init__(
        self,
        outdir: str | None = None,
        base_filename: str = "particle_transform",
        metadata_key: str = PARTICLE_TRANSFORM_KEY,
        plot_kwargs: dict[str, Any] | None = None,
        parameter_inds: list[int] | None = None,
        parameter_labels: list[str] | None = None,
    ):
        self.base_filename = base_filename
        self.metadata_key = metadata_key
        self.parameter_inds = parameter_inds
        self.parameter_labels = parameter_labels
        filename = f"{base_filename}_0.png"
        super().__init__(outdir, filename, plot_kwargs)

    def apply(
        self,
        state: "ParticleState",
        context: "StepContext",
        results: "SMCResult",
    ) -> None:
        transform = context.metadata.get(self.metadata_key)
        if transform is not None:
            filename = f"{self.base_filename}_{context.iteration}.png"
            self.full_path = self.outdir / filename
            logger.debug(f"Plotting particle transform to {self.full_path}")

            particles = state.particles
            transformed_particles = transform.forward(
                particles, context=context
            )[0]

            from .plot import plot_corner

            plot_corner(
                transformed_particles,
                filename=self.full_path,
                parameter_inds=self.parameter_inds,
                parameter_labels=self.parameter_labels,
                **self.plot_kwargs,
            )
        else:
            logger.debug("No particle transform found in state metadata.")
