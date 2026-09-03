from __future__ import annotations

import numpy as np

from sequince.callbacks import MutationChainPlot, SMCCallback
from sequince.result import SMCResult
from sequince.state import ParticleState
from sequince.steps.context import StepContext


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
