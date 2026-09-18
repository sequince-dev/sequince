from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest
from orng import RandomGenerator

from sequince import ParticleState
from sequince.steps import (
    AcceptanceRateRhoPolicy,
    AdaptiveMiniPCNMutation,
    MiniPCNMutation,
    MutationMetricsOutput,
    RepeatedMutationStep,
    StepContext,
)


@pytest.mark.parametrize("accepted, expected", [(0, 0.4), (10, 0.6)])
def test_policy_bounds_large_updates(accepted: int, expected: float) -> None:
    policy = AcceptanceRateRhoPolicy(
        adaptation_rate=1e300, min_rho=0.4, max_rho=0.6
    )
    assert (
        policy.update(
            current_rho=0.5,
            metrics=MutationMetricsOutput(1, 10, accepted),
            context=StepContext(t=0),
        )
        == expected
    )


def test_empty_disabled_fractional_counts_and_initial_clipping() -> None:
    context = StepContext(t=0)
    policy = AcceptanceRateRhoPolicy(min_rho=0.2, max_rho=0.8)
    assert policy.initialise(initial_rho=0.1, context=context) == 0.2
    assert (
        policy.update(
            current_rho=0.5,
            metrics=MutationMetricsOutput(0, 0, 0),
            context=context,
        )
        == 0.5
    )
    fractional = MutationMetricsOutput(1, 10, 2.5)
    assert policy.update(
        current_rho=0.5,
        metrics=fractional,
        context=context,
    ) == pytest.approx(0.5 * np.exp(0.1 * (0.25 - 0.234)))
    assert (
        AcceptanceRateRhoPolicy(adaptation_rate=0).update(
            current_rho=0.123,
            metrics=fractional,
            context=context,
        )
        == 0.123
    )


@pytest.mark.parametrize(
    "attempted, accepted",
    [
        (-1, 0),
        (1, -1),
        (1, 2),
        (0, 1),
        (np.nan, 0),
        (1, np.inf),
    ],
)
def test_invalid_counts_rejected(attempted: float, accepted: float) -> None:
    with pytest.raises(ValueError):
        AcceptanceRateRhoPolicy().update(
            current_rho=0.5,
            metrics=MutationMetricsOutput(1, attempted, accepted),
            context=StepContext(t=0),
        )


class RecordingPolicy:
    def __init__(self) -> None:
        self.initialisations = 0
        self.budgets: list[int] = []

    def initialise(self, *, initial_rho: float, context: StepContext) -> float:
        self.initialisations += 1
        return initial_rho / 2

    def update(
        self,
        *,
        current_rho: float,
        metrics: MutationMetricsOutput,
        context: StepContext,
    ) -> float:
        self.budgets.append(int(metrics.n_steps))
        return 0.7

    def state_dict(self) -> dict[str, Any]:
        return {
            "initialisations": self.initialisations,
            "budgets": list(self.budgets),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.initialisations = state["initialisations"]
        self.budgets = list(state["budgets"])


def make(policy: Any) -> AdaptiveMiniPCNMutation:
    return AdaptiveMiniPCNMutation(
        RepeatedMutationStep(
            MiniPCNMutation(
                lambda x: -0.5 * np.sum(x * x, axis=1), step_fn="pcn"
            ),
            n_rounds=3,
            n_steps=7,
        ),
        policy=policy,
    )


def test_custom_policy_block_boundary_and_checkpoint() -> None:
    state = ParticleState(np.random.default_rng(4).normal(size=(64, 2)))
    policy = RecordingPolicy()
    wrapper = make(policy)
    wrapper.apply_with_n_steps(state, StepContext(t=0), None, n_steps=0)
    assert policy.budgets == []
    assert policy.initialisations == 1
    wrapper(state, StepContext(t=0), RandomGenerator(seed=9))
    assert policy.budgets == [
        7
    ]  # One complete block, not three round updates.
    assert wrapper.current_rho == 0.25
    assert wrapper.next_rho == 0.7
    restored_policy = RecordingPolicy()
    restored = make(restored_policy)
    restored.load_state_dict(json.loads(json.dumps(wrapper.state_dict())))
    restored(state, StepContext(t=0), RandomGenerator(seed=9))
    assert restored_policy.initialisations == 1
    assert restored_policy.budgets == [7]
    expected, _, _ = wrapper(state, StepContext(t=1), RandomGenerator(seed=10))
    actual, _, _ = restored(state, StepContext(t=1), RandomGenerator(seed=10))
    np.testing.assert_array_equal(expected.particles, actual.particles)
    assert restored.current_rho == 0.7
    assert restored_policy.budgets == [7, 7]


@pytest.mark.parametrize("value", [0, 1, np.nan, np.inf, [0.1, 0.2]])
@pytest.mark.parametrize("phase", ["initialise", "update"])
def test_invalid_policy_decisions(value: Any, phase: str) -> None:
    policy = RecordingPolicy()
    setattr(policy, phase, lambda **kwargs: value)
    with pytest.raises((ValueError, TypeError)):
        make(policy)(
            ParticleState(np.random.default_rng(4).normal(size=(64, 2))),
            StepContext(t=0),
            RandomGenerator(seed=9),
        )


def test_policy_configuration_checkpoint_mismatch_and_invalid_interface() -> (
    None
):
    saved = make(AcceptanceRateRhoPolicy()).state_dict()
    with pytest.raises(ValueError, match="policy checkpoint configuration"):
        make(AcceptanceRateRhoPolicy(adaptation_rate=0.2)).load_state_dict(
            saved
        )
    with pytest.raises(TypeError, match="initialise.*update"):
        make(object())
