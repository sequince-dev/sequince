from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from sequince.exceptions import DegenerateWeightsError
from sequince.smc import SequentialMonteCarlo
from sequince.state import ParticleState
from sequince.steps.base import SMCStep, StepContext
from sequince.steps.weighting import (
    AdaptiveTemperedWeighting,
    FixedTemperedWeighting,
)
from sequince.target import EvaluationCounts

if TYPE_CHECKING:
    from orng import RandomGenerator


class StopAfterNStep(SMCStep):
    def __init__(self, limit: int):
        self._limit = limit

    def apply(
        self,
        state: ParticleState,
        context: StepContext,
        rng: "RandomGenerator",
    ) -> tuple[ParticleState, StepContext, int]:
        count = state.metadata.get("calls", 0) + 1
        metadata = dict(state.metadata)
        metadata["calls"] = count
        if count >= self._limit:
            context = context.request_stop()
        return state.with_updates(metadata=metadata), context, count


class RecordObservationStep(SMCStep):
    def apply(
        self,
        state: ParticleState,
        context: StepContext,
        rng: "RandomGenerator",
    ) -> tuple[ParticleState, StepContext, object]:
        metadata = dict(state.metadata)
        metadata.setdefault("observations", []).append(context.observation)
        return (
            state.with_updates(metadata=metadata),
            context,
            context.observation,
        )


class RecordStep(SMCStep):
    def __init__(self, name: str):
        self._name = name

    def apply(
        self,
        state: ParticleState,
        context: StepContext,
        rng: "RandomGenerator",
    ) -> tuple[ParticleState, StepContext, str]:
        metadata = dict(state.metadata)
        history = list(metadata.get("executed_steps", []))
        history.append(self._name)
        metadata["executed_steps"] = history
        return state.with_updates(metadata=metadata), context, self._name


class RecordingCallback:
    def __init__(self) -> None:
        self.records: list[dict[str, object]] = []

    def __call__(
        self,
        state: ParticleState,
        context: StepContext,
        result,
    ) -> None:
        self.records.append(
            {
                "iteration": context.t,
                "state_iteration": state.metadata.get("iteration"),
                "result_length": len(result),
                "result_final_iteration": result.final_state.metadata.get(
                    "iteration"
                ),
            }
        )


def test_smc_respects_stop_request(particle_state: ParticleState, rng) -> None:
    smc = SequentialMonteCarlo(steps=[StopAfterNStep(limit=3)])
    states = smc.run(particle_state, num_steps=None, rng=rng)

    assert len(states) == 3
    assert states[-1].metadata["calls"] == 3
    initial_state = states.initial_state
    assert initial_state is not None
    assert initial_state is not particle_state
    np.testing.assert_allclose(
        initial_state.particles,
        particle_state.particles,
    )


def test_smc_consumes_initial_evaluation_counts(
    particle_state: ParticleState, rng
) -> None:
    counts = EvaluationCounts(
        log_target_evaluations=5,
        log_proposal_evaluations=5,
    )
    particle_state.metadata["evaluation_counts"] = counts
    smc = SequentialMonteCarlo(steps=[StopAfterNStep(limit=1)])

    result = smc.run(particle_state, rng=rng)

    assert result.evaluation_counts == counts
    assert "evaluation_counts" not in result.initial_state.metadata
    assert "evaluation_counts" not in result.final_state.metadata


def test_smc_processes_observations_in_order(
    particle_state: ParticleState, rng
) -> None:
    observations = ["a", "b", "c"]
    smc = SequentialMonteCarlo(steps=[RecordObservationStep()])

    states = smc.run(
        particle_state,
        observations=observations,
        num_steps=len(observations),
        rng=rng,
    )

    assert len(states) == len(observations)
    assert states[-1].metadata["observations"] == observations


def test_smc_deferred_stop_finishes_current_iteration(
    particle_state: ParticleState, rng
) -> None:
    class DeferredStopStep(SMCStep):
        def apply(
            self,
            state: ParticleState,
            context: StepContext,
            rng: "RandomGenerator",
        ) -> tuple[ParticleState, StepContext, None]:
            return state, context.request_stop(), None

    smc = SequentialMonteCarlo(
        steps=[
            DeferredStopStep(),
            RecordStep("after_stop"),
        ]
    )
    result = smc.run(particle_state, num_steps=5, rng=rng)

    assert len(result) == 1
    assert result.final_state.metadata["executed_steps"] == ["after_stop"]
    assert result.outputs == ((None, "after_stop"),)


def test_smc_immediate_stop_skips_remaining_steps(
    particle_state: ParticleState, rng
) -> None:
    class ImmediateStopStep(SMCStep):
        def apply(
            self,
            state: ParticleState,
            context: StepContext,
            rng: "RandomGenerator",
        ) -> tuple[ParticleState, StepContext, None]:
            return state, context.request_stop_immediately(), None

    smc = SequentialMonteCarlo(
        steps=[
            ImmediateStopStep(),
            RecordStep("should_not_run"),
        ]
    )
    result = smc.run(particle_state, num_steps=5, rng=rng)

    assert len(result) == 1
    assert "executed_steps" not in result.final_state.metadata
    assert result.outputs == ((None,),)


def test_smc_with_adaptive_tempering_stops_at_max_beta(
    particle_state: ParticleState, rng, unit_log_likelihood
) -> None:
    adaptive = AdaptiveTemperedWeighting(
        unit_log_likelihood,
        target_ess_ratio=0.8,
        max_beta=0.5,
        min_delta=0.05,
        max_bisection_iter=8,
    )

    initial_state = particle_state.with_updates(metadata={"beta": 0.0})
    metadata = {"beta": 0.0}

    smc = SequentialMonteCarlo(
        steps=[adaptive, RecordStep("resampling"), RecordStep("mutation")]
    )
    states = smc.run(initial_state, num_steps=None, rng=rng, metadata=metadata)

    assert states[-1].metadata["beta"] == pytest.approx(0.5, abs=1e-6)
    assert states.context.metadata["beta"] == pytest.approx(
        states[-1].metadata["beta"]
    )
    assert states[-1].metadata["executed_steps"] == [
        "resampling",
        "mutation",
    ]


@pytest.mark.parametrize("invalid", [-np.inf, np.inf, np.nan])
@pytest.mark.parametrize("functional", [False, True])
def test_runner_rejects_degenerate_weights(
    invalid: float, functional: bool
) -> None:
    callbacks = []
    smc = SequentialMonteCarlo(
        [
            FixedTemperedWeighting(
                lambda _: np.full(4, invalid), beta_increment=1
            )
        ],
        callbacks=[lambda *args: callbacks.append(args)],
    )
    state = ParticleState(np.zeros((4, 1)))
    with pytest.raises(DegenerateWeightsError):
        if functional:
            smc.run_functional(state, num_steps=1, random_state=np.array([0]))
        else:
            smc.run(state, num_steps=1, rng=0)
    assert not callbacks


def test_smc_invokes_callbacks_with_incremental_results(
    particle_state: ParticleState, rng
) -> None:
    def annotate_iteration(
        state: ParticleState, context: StepContext, rng: "RandomGenerator"
    ) -> tuple[ParticleState, StepContext, int]:
        metadata = dict(state.metadata)
        history = list(metadata.get("history", []))
        history.append(context.t)
        metadata["history"] = history
        metadata["iteration"] = context.t
        return state.with_updates(metadata=metadata), context, context.t

    callback = RecordingCallback()
    smc = SequentialMonteCarlo(
        steps=[annotate_iteration],
        callbacks=[callback],
    )

    result = smc.run(particle_state, num_steps=3, rng=rng)

    assert [record["iteration"] for record in callback.records] == [0, 1, 2]
    assert [record["result_length"] for record in callback.records] == [
        1,
        2,
        3,
    ]
    assert [record["state_iteration"] for record in callback.records] == [
        0,
        1,
        2,
    ]
    assert [
        record["result_final_iteration"] for record in callback.records
    ] == [
        0,
        1,
        2,
    ]
    assert len(result) == 3
    assert result.final_state.metadata["iteration"] == 2
    assert result.outputs == ((0,), (1,), (2,))


def test_callback_can_request_stop(particle_state: ParticleState, rng) -> None:
    class StopViaCallback:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(
            self,
            state: ParticleState,
            context: StepContext,
            result,
        ) -> StepContext:
            self.calls += 1
            return context.request_stop()

    def identity_step(
        state: ParticleState, context: StepContext, rng: "RandomGenerator"
    ) -> tuple[ParticleState, StepContext, None]:
        return state, context, None

    callback = StopViaCallback()
    smc = SequentialMonteCarlo(
        steps=[identity_step],
        callbacks=[callback],
    )

    result = smc.run(particle_state, num_steps=5, rng=rng)

    assert callback.calls == 1
    assert len(result) == 1
