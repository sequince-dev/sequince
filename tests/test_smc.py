from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from sequince.smc import SequentialMonteCarlo
from sequince.state import ParticleState
from sequince.steps.base import SMCStep, StepContext
from sequince.steps.weighting import AdaptiveTemperedWeighting

if TYPE_CHECKING:
    from orng import ArrayRNG


class StopAfterNStep(SMCStep):
    def __init__(self, limit: int):
        self._limit = limit

    def apply(
        self,
        state: ParticleState,
        context: StepContext,
        rng: "ArrayRNG",
    ) -> ParticleState:
        count = state.metadata.get("calls", 0) + 1
        metadata = dict(state.metadata)
        metadata["calls"] = count
        if count >= self._limit:
            context.request_stop()
        return state.with_updates(metadata=metadata)


class RecordObservationStep(SMCStep):
    def apply(
        self,
        state: ParticleState,
        context: StepContext,
        rng: "ArrayRNG",
    ) -> ParticleState:
        metadata = dict(state.metadata)
        metadata.setdefault("observations", []).append(context.observation)
        return state.with_updates(metadata=metadata)


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


def test_smc_with_adaptive_tempering_stops_at_max_temperature(
    particle_state: ParticleState, rng
) -> None:
    def log_like(particles: np.ndarray, context: StepContext) -> np.ndarray:
        return np.ones(particles.shape[0])

    adaptive = AdaptiveTemperedWeighting(
        log_like,
        target_ess_ratio=0.8,
        max_temperature=0.5,
        min_delta=0.05,
        max_bisection_iter=8,
    )

    initial_state = particle_state.with_updates(metadata={"temperature": 0.0})
    metadata = {"temperature": 0.0}

    smc = SequentialMonteCarlo(steps=[adaptive])
    states = smc.run(initial_state, num_steps=None, rng=rng, metadata=metadata)

    assert states[-1].metadata["temperature"] == pytest.approx(0.5, abs=1e-6)
    assert metadata["temperature"] == pytest.approx(
        states[-1].metadata["temperature"]
    )


def test_smc_invokes_callbacks_with_incremental_results(
    particle_state: ParticleState, rng
) -> None:
    def annotate_iteration(
        state: ParticleState, context: StepContext, rng: "ArrayRNG"
    ) -> ParticleState:
        metadata = dict(state.metadata)
        history = list(metadata.get("history", []))
        history.append(context.t)
        metadata["history"] = history
        metadata["iteration"] = context.t
        return state.with_updates(metadata=metadata)

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


def test_callback_can_request_stop(particle_state: ParticleState, rng) -> None:
    class StopViaCallback:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(
            self,
            state: ParticleState,
            context: StepContext,
            result,
        ) -> None:
            self.calls += 1
            context.request_stop()

    def identity_step(
        state: ParticleState, context: StepContext, rng: "ArrayRNG"
    ) -> ParticleState:
        return state

    callback = StopViaCallback()
    smc = SequentialMonteCarlo(
        steps=[identity_step],
        callbacks=[callback],
    )

    result = smc.run(particle_state, num_steps=5, rng=rng)

    assert callback.calls == 1
    assert len(result) == 1
