from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from sequince.callbacks import (
    IntermediateParticlesPlot,
    MutationChainPlot,
    ParticleTransformPlot,
    SMCCallback,
    StatePlot,
    TracePlot,
)
from sequince.result import SMCResult
from sequince.state import ParticleState
from sequince.steps.context import StepContext
from sequince.steps.control import ConditionalStepOutput


def test_smc_callback_every(monkeypatch) -> None:
    state = ParticleState(np.zeros((1, 1)))
    result = SMCResult(states=[state], context=StepContext(t=0))
    callback = SMCCallback(every=2)
    called_at = []

    monkeypatch.setattr(
        callback,
        "apply",
        lambda state, context, results: called_at.append(context.iteration),
    )

    for iteration in range(5):
        callback(state, StepContext(t=iteration), result)

    assert called_at == [0, 2, 4]


def test_mutation_chain_plot_reads_chain_from_step_output(
    tmp_path,
    monkeypatch,
) -> None:
    chain = np.zeros((2, 4, 3))
    state = ParticleState(np.zeros((4, 3)))
    context = StepContext(t=0)
    result = SMCResult(
        states=[state],
        context=context,
        outputs=(({"chain": chain},),),
    )
    plotted = {}

    def record_plot(values, **kwargs):
        plotted["chain"] = values
        plotted["filename"] = kwargs["filename"]

    monkeypatch.setattr("sequince.plot.plot_chains", record_plot)
    callback = MutationChainPlot(outdir=tmp_path)

    callback(state, context, result)

    assert plotted["chain"] is chain
    assert plotted["filename"] == tmp_path / "chain_0.png"


def test_mutation_chain_plot_reads_repeated_sequential_step_outputs(
    tmp_path,
    monkeypatch,
) -> None:
    first_chain = np.zeros((2, 4, 3))
    second_chain = np.ones((2, 4, 3))
    repeated_output = (
        ({"chain": first_chain}, {"symmetry_mode": 1}),
        ({"chain": second_chain}, {"symmetry_mode": 2}),
    )
    state = ParticleState(np.zeros((4, 3)))
    context = StepContext(t=3)
    result = SMCResult(
        states=[state],
        context=context,
        outputs=((repeated_output,),),
    )
    plotted = {}

    monkeypatch.setattr(
        "sequince.plot.plot_chains",
        lambda values, **kwargs: plotted.update(chain=values, **kwargs),
    )

    MutationChainPlot(outdir=tmp_path)(state, context, result)

    np.testing.assert_array_equal(
        plotted["chain"], np.concatenate([first_chain, second_chain])
    )
    assert plotted["split_indices"] == [2]
    assert plotted["filename"] == tmp_path / "chain_3.png"


def test_mutation_chain_plot_reads_conditional_output_before_later_steps(
    tmp_path,
    monkeypatch,
) -> None:
    chain = np.ones((2, 4, 3))
    mutation_output = ConditionalStepOutput(
        executed=True,
        child_output=((({"chain": chain}, {"symmetry_mode": 1}),),),
    )
    state = ParticleState(np.zeros((4, 3)))
    context = StepContext(t=2)
    result = SMCResult(
        states=[state],
        context=context,
        outputs=((mutation_output, None),),
    )
    plotted = {}

    monkeypatch.setattr(
        "sequince.plot.plot_chains",
        lambda values, **kwargs: plotted.update(chain=values, **kwargs),
    )

    MutationChainPlot(outdir=tmp_path)(state, context, result)

    np.testing.assert_array_equal(plotted["chain"], chain)


def test_mutation_chain_plot_respects_every(tmp_path, monkeypatch) -> None:
    chain = np.zeros((2, 4, 3))
    state = ParticleState(np.zeros((4, 3)))
    result = SMCResult(
        states=[state],
        context=StepContext(t=0),
        outputs=(({"chain": chain},),),
    )
    plot_chains = MagicMock()
    monkeypatch.setattr("sequince.plot.plot_chains", plot_chains)
    callback = MutationChainPlot(outdir=tmp_path, every=2)

    for iteration in range(4):
        callback(state, StepContext(t=iteration), result)

    assert plot_chains.call_count == 2


def test_trace_plot(tmp_path) -> None:
    state = ParticleState(np.zeros((4, 3)))
    context = StepContext(t=2)
    result = MagicMock(spec=SMCResult)
    callback = TracePlot(
        parameter_inds=[0, 2],
        parameter_labels=["a", "c"],
        outdir=tmp_path,
        plot_kwargs={"alpha": 0.5},
    )

    callback(state, context, result)

    result.plot_trace.assert_called_once_with(
        parameter_inds=[0, 2],
        parameter_labels=["a", "c"],
        filename=tmp_path / "trace.png",
        alpha=0.5,
    )


def test_intermediate_particles_plot(
    tmp_path,
    monkeypatch,
) -> None:
    particles = np.arange(12).reshape(4, 3)
    state = ParticleState(particles)
    context = StepContext(t=3)
    result = MagicMock(spec=SMCResult)
    plot_corner = MagicMock()
    monkeypatch.setattr("sequince.plot.plot_corner", plot_corner)
    callback = IntermediateParticlesPlot(
        outdir=tmp_path,
        base_filename="cloud",
        parameter_inds=[1],
        parameter_labels=["b"],
        plot_kwargs={"bins": 10},
    )

    callback(state, context, result)

    plot_corner.assert_called_once_with(
        particles,
        filename=tmp_path / "cloud_3.png",
        parameter_inds=[1],
        parameter_labels=["b"],
        bins=10,
    )


def test_state_plot(tmp_path, monkeypatch) -> None:
    state = ParticleState(np.zeros((1, 1)))
    context = StepContext(t=0)
    result = MagicMock(spec=SMCResult)
    figure = MagicMock()
    axis = MagicMock()
    monkeypatch.setattr(
        "matplotlib.pyplot.subplots", lambda **kwargs: (figure, axis)
    )
    callback = StatePlot(
        ["acceptance"],
        outdir=tmp_path,
        plot_kwargs={"color": "black"},
    )

    callback(state, context, result)

    result.plot_metadata_history.assert_called_once_with(
        "acceptance",
        ax=axis,
        label_axes=False,
        color="black",
    )
    axis.set_ylabel.assert_called_once_with("acceptance")
    axis.set_xlabel.assert_called_once_with("SMC Iteration")
    figure.savefig.assert_called_once_with(tmp_path / "state.png")


def test_particle_transform_plot(
    tmp_path,
    monkeypatch,
) -> None:
    particles = np.zeros((4, 2))
    transformed_particles = np.ones((4, 2))
    transform = MagicMock()
    transform.forward.return_value = (transformed_particles, None, None)
    state = ParticleState(particles)
    context = StepContext(t=4, metadata={"particle_transform": transform})
    result = MagicMock(spec=SMCResult)
    plot_corner = MagicMock()
    monkeypatch.setattr("sequince.plot.plot_corner", plot_corner)
    callback = ParticleTransformPlot(
        outdir=tmp_path,
        parameter_inds=[0],
        parameter_labels=["x"],
        plot_kwargs={"bins": 8},
    )

    callback(state, context, result)

    transform.forward.assert_called_once_with(particles, context=context)
    plot_corner.assert_called_once_with(
        transformed_particles,
        filename=tmp_path / "particle_transform_4.png",
        parameter_inds=[0],
        parameter_labels=["x"],
        bins=8,
    )
