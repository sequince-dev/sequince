from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .steps.mutation import PARTICLE_TRANSFORM_KEY

if TYPE_CHECKING:
    from .result import SMCResult
    from .state import ParticleState
    from .steps.base import StepContext


logger = logging.getLogger(__name__)


class SMCCallback:
    """Base class for composable SMC callbacks.

    Parameters
    ----------
    every:
        Run the callback every `every` iterations. Default is 1 (run every
        iteration).
    """

    def __init__(self, every: int = 1):
        if every < 1:
            raise ValueError("every must be a positive integer.")
        self.every = every
        self._last_called: int | None = None

    def __call__(
        self,
        state: "ParticleState",
        context: "StepContext",
        results: "SMCResult",
    ) -> "StepContext" | None:
        """Run the callback and optionally return an updated context."""
        if (
            self._last_called is None
            or context.iteration - self._last_called >= self.every
        ):
            self._last_called = context.iteration
            logger.debug(
                "Iteration %d: applying callback %s",
                context.iteration,
                self.__class__.__name__,
            )
            return self.apply(state, context, results)
        else:
            logger.debug(
                "Iteration %d: skipping callback %s (every=%d)",
                context.iteration,
                self.__class__.__name__,
                self.every,
            )
            return None

    def apply(
        self,
        state: "ParticleState",
        context: "StepContext",
        results: "SMCResult",
    ) -> "StepContext" | None:
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
    every:
        Run the callback every `every` iterations. Default is 1 (run every
        iteration).
    """

    def __init__(
        self,
        outdir: str | None = None,
        filename: str = "plot.png",
        plot_kwargs: dict[str, Any] | None = None,
        every: int = 1,
    ):
        super().__init__(every=every)
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
    every:
        Run the callback every `every` iterations. Default is 1 (run every
        iteration).
    """

    def __init__(
        self,
        parameter_inds: list[int] | None = None,
        parameter_labels: list[str] | None = None,
        outdir: str | None = None,
        filename: str = "trace.png",
        plot_kwargs: dict[str, Any] | None = None,
        every: int = 1,
    ):
        super().__init__(outdir, filename, plot_kwargs, every=every)

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
    every:
        Run the callback every `every` iterations. Default is 1 (run every
        iteration).
    """

    def __init__(
        self,
        outdir: str | None = None,
        base_filename: str = "particles",
        parameter_inds: list[int] | None = None,
        parameter_labels: list[str] | None = None,
        plot_kwargs: dict[str, Any] | None = None,
        every: int = 1,
    ):
        filename = f"{base_filename}_0.png"
        self.base_filename = base_filename
        self.parameter_labels = parameter_labels
        self.parameter_inds = parameter_inds
        super().__init__(outdir, filename, plot_kwargs, every=every)

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
    every:
        Run the callback every `every` iterations. Default is 1 (run every
        iteration).
    """

    def __init__(
        self,
        metadata_keys: list[str],
        outdir: str | None = None,
        filename: str = "state.png",
        plot_kwargs: dict[str, Any] | None = None,
        every: int = 1,
    ):
        super().__init__(outdir, filename, plot_kwargs, every=every)
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
            "log_normalizing_constant": results.plot_log_normalizing_constant,
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
    step_index:
        Index of the mutation step in the current iteration's outputs.
    output_key:
        Key containing the chain in the selected step output.
    every:
        Run the callback every `every` iterations. Default is 1 (run every
        iteration).
    """

    def __init__(
        self,
        outdir: str | None = None,
        base_filename: str = "chain",
        parameter_inds: list[int] | None = None,
        parameter_labels: list[str] | None = None,
        plot_kwargs: dict[str, Any] | None = None,
        step_index: int = -1,
        output_key: str = "chain",
        every: int = 1,
    ):
        self.base_filename = base_filename
        self.parameter_labels = parameter_labels
        self.parameter_inds = parameter_inds
        self.step_index = step_index
        self.output_key = output_key
        self.every = every
        filename = f"{base_filename}_0.png"
        super().__init__(outdir, filename, plot_kwargs)

    def apply(
        self,
        state: "ParticleState",
        context: "StepContext",
        results: "SMCResult",
    ) -> None:
        del state
        chains = None
        if results.outputs:
            iteration_outputs = results.outputs[-1]
            try:
                output = iteration_outputs[self.step_index]
            except IndexError:
                output = None
            if isinstance(output, Mapping):
                chains = output.get(self.output_key)
        if isinstance(chains, Mapping):
            from .array_api import asarray, namespace

            chain_values = list(chains.values())
            if chain_values:
                xp = namespace(*chain_values)
                chains = xp.concat(
                    [
                        asarray(chain, xp=xp, like=chain_values[0])
                        for chain in chain_values
                    ],
                    axis=0,
                )
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
            logger.debug(
                "No mutation chain found in the selected step output."
            )


class ParticleTransformPlot(PlotCallback):
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
    every:
        Run the callback every `every` iterations. Default is 1 (run every
        iteration).
    """

    def __init__(
        self,
        outdir: str | None = None,
        base_filename: str = "particle_transform",
        metadata_key: str = PARTICLE_TRANSFORM_KEY,
        plot_kwargs: dict[str, Any] | None = None,
        parameter_inds: list[int] | None = None,
        parameter_labels: list[str] | None = None,
        every: int = 1,
    ):
        self.base_filename = base_filename
        self.metadata_key = metadata_key
        self.parameter_inds = parameter_inds
        self.parameter_labels = parameter_labels
        filename = f"{base_filename}_0.png"
        super().__init__(outdir, filename, plot_kwargs, every=every)

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
            transformed_particles, _, _ = transform.forward(
                particles, context=context
            )

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
