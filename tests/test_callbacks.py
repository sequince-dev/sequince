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
    TransformedMutationChainPlot,
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


def test_transformed_mutation_chain_plot_transforms_repeated_chains(
    tmp_path,
    monkeypatch,
) -> None:
    first_chain = np.zeros((2, 4, 3))
    second_chain = np.ones((3, 4, 3))
    repeated_output = (
        ({"chain": first_chain}, {"symmetry_mode": 1}),
        ({"chain": second_chain}, {"symmetry_mode": 2}),
    )
    transform = MagicMock()
    transform.forward.side_effect = lambda values, **kwargs: (
        values + 2.0,
        np.zeros(len(values)),
        None,
    )
    state = ParticleState(np.zeros((4, 3)))
    context = StepContext(t=3, metadata={"particle_transform": transform})
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

    TransformedMutationChainPlot(outdir=tmp_path)(state, context, result)

    expected = np.concatenate([first_chain + 2.0, second_chain + 2.0])
    np.testing.assert_array_equal(plotted["chain"], expected)
    assert plotted["split_indices"] == [2]
    assert plotted["filename"] == tmp_path / "transformed_chain_3.png"
    assert [
        call.args[0].shape for call in transform.forward.call_args_list
    ] == [
        (8, 3),
        (12, 3),
    ]


def test_transformed_mutation_chain_plot_skips_without_transform(
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
    plot_chains = MagicMock()
    monkeypatch.setattr("sequince.plot.plot_chains", plot_chains)

    TransformedMutationChainPlot(outdir=tmp_path)(state, context, result)

    plot_chains.assert_not_called()


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


def test_adaptive_mutation_plot_completed_metrics(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from matplotlib.figure import Figure

    from sequince.callbacks import AdaptiveMutationPlot
    from sequince.steps.mutation.metrics import MutationMetricsOutput

    saved = []
    savefig = Figure.savefig

    def record(fig, *args, **kwargs):
        saved.append(fig)
        return savefig(fig, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", record)
    state = ParticleState(np.zeros((4, 1)))
    context = StepContext(t=3)
    result = SMCResult(
        states=[state] * 4,
        context=context,
        outputs=(
            ({"metrics": MutationMetricsOutput(8, 32, 16)},),
            (
                SimpleNamespace(
                    child_output=SimpleNamespace(
                        metrics=MutationMetricsOutput(4, 16, 12)
                    )
                ),
            ),
            ({"metrics": MutationMetricsOutput(0, 0, 0)},),
            (None,),
        ),
    )
    callback = AdaptiveMutationPlot(tmp_path, step_index=0, target_steps=3)
    callback(state, context, result)
    assert callback.full_path.is_file()
    axes = saved[0].axes
    np.testing.assert_allclose(axes[0].lines[0].get_ydata(), [8, 4, 0, np.nan])
    np.testing.assert_allclose(
        axes[1].lines[0].get_ydata(), [0.5, 0.75, np.nan, np.nan]
    )
    np.testing.assert_allclose(axes[2].lines[0].get_ydata(), [4, 3, 0, np.nan])
    np.testing.assert_allclose(axes[2].lines[1].get_ydata(), [3, 3])


def test_adaptive_mutation_plot_empty_history(tmp_path):
    from sequince.callbacks import AdaptiveMutationPlot

    state = ParticleState(np.zeros((1, 1)))
    context = StepContext(t=0)
    callback = AdaptiveMutationPlot(tmp_path)
    callback(state, context, SMCResult(states=[state], context=context))
    assert not callback.full_path.exists()
