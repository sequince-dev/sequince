from __future__ import annotations

import numpy as np
import pytest

from sequince.smc import SequentialMonteCarlo
from sequince.state import ParticleState
from sequince.steps.base import SMCStep, StepContext
from sequince.steps.weighting import AdaptiveTemperedWeighting


class StopAfterNStep(SMCStep):
    def __init__(self, limit: int):
        self._limit = limit

    def apply(
        self,
        state: ParticleState,
        context: StepContext,
        rng: np.random.Generator,
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
        rng: np.random.Generator,
    ) -> ParticleState:
        metadata = dict(state.metadata)
        metadata.setdefault("observations", []).append(context.observation)
        return state.with_updates(metadata=metadata)


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
    static_context = {"temperature": 0.0}

    smc = SequentialMonteCarlo(steps=[adaptive])
    states = smc.run(
        initial_state, num_steps=None, rng=rng, static_context=static_context
    )

    assert states[-1].metadata["temperature"] == pytest.approx(0.5, abs=1e-6)
    assert static_context["temperature"] == pytest.approx(
        states[-1].metadata["temperature"]
    )
