from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from orng import RandomGenerator
from test_ensemble_mixture import mixture

from sequince import ParticleState
from sequince.array_api import to_numpy
from sequince.steps import (
    AdaptMixtureWeights,
    AdaptMutationSteps,
    ESJDMixtureWeightPolicy,
    MixtureComponentMetrics,
    MutationMetricsOutput,
    MutationTransition,
    RedBlueMetropolisHastingsMutation,
    RepeatedMutationStep,
    StepContext,
)
from sequince.steps.mutation.adaptive.policies.acceptance import (
    AcceptanceRatePolicy,
)


def metrics(
    attempts=(100, 100), jumps=(100.0, 900.0)
) -> MutationMetricsOutput:
    return MutationMetricsOutput(
        1,
        sum(attempts),
        0,
        {
            "mixture_components": {
                "component": np.arange(2),
                "n_attempted": np.asarray(attempts),
                "squared_jump_distance": np.asarray(jumps),
            }
        },
    )


def test_scores_and_exploration(context_factory) -> None:
    policy = ESJDMixtureWeightPolicy(
        learning_rate=1, exploration_probability=0.2
    )
    context = context_factory()
    current = policy.initialise(initial_weights=(1, 0), context=context)
    np.testing.assert_allclose(current, [0.9, 0.1])
    updated = policy.update(
        current_weights=current, metrics=metrics(), context=context
    )
    np.testing.assert_allclose(updated, [0.18, 0.82])
    np.testing.assert_allclose(
        policy.update(
            current_weights=current,
            metrics=metrics((200, 100), (200.0, 900.0)),
            context=context,
        ),
        updated,
    )
    damped = replace(policy, learning_rate=0.5).update(
        current_weights=current,
        metrics=metrics(jumps=(0.0, 900.0)),
        context=context,
    )
    np.testing.assert_allclose(damped, [0.5, 0.5])
    for _ in range(20):
        current = policy.update(
            current_weights=current,
            metrics=metrics(jumps=(0.0, 900.0)),
            context=context,
        )
        assert min(current) >= 0.1 - 1e-15


@pytest.mark.parametrize(
    "attempts,jumps",
    [((0, 0), (0, 0)), ((100, 1), (100, 10)), ((100, 100), (0, 0))],
)
def test_sparse_or_zero(context_factory, attempts, jumps) -> None:
    assert ESJDMixtureWeightPolicy().update(
        current_weights=(0.3, 0.7),
        metrics=metrics(attempts, jumps),
        context=context_factory(),
    ) == (0.3, 0.7)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(learning_rate=0),
        dict(exploration_probability=0),
        dict(exploration_probability=np.nan),
        dict(min_attempts=True),
        dict(min_attempts=0),
    ],
)
def test_invalid_policy(kwargs) -> None:
    with pytest.raises((ValueError, TypeError)):
        ESJDMixtureWeightPolicy(**kwargs)


@pytest.mark.parametrize(
    "attempts,jumps",
    [
        ((-1, 100), (0, 2)),
        ((1.5, 100), (1, 2)),
        ((100, 100), (-1, 2)),
        ((0, 100), (1, 2)),
        ((100, 100), (np.inf, 2)),
    ],
)
def test_invalid_statistics(context_factory, attempts, jumps) -> None:
    with pytest.raises(ValueError):
        ESJDMixtureWeightPolicy().update(
            current_weights=(0.5, 0.5),
            metrics=metrics(attempts, jumps),
            context=context_factory(),
        )


def test_component_totals_and_rejections(
    backend, backend_factory, context_factory
) -> None:
    case = backend_factory(backend)
    xp = case.xp
    context = context_factory()
    metric = MixtureComponentMetrics(2)
    state = metric.initialize(case.particles, case.log_weights, context)
    state = metric.update(
        state,
        case.particles,
        xp.asarray([True, False, True, False]),
        context,
        transition=MutationTransition(
            case.particles, {"mixture_component": xp.asarray([0, 0, 1, 1])}
        ),
    )
    result = metric.result(state, context)
    np.testing.assert_array_equal(to_numpy(result["n_attempted"]), [2, 2])
    np.testing.assert_array_equal(to_numpy(result["n_accepted"]), [1, 1])
    jumps = np.sum(to_numpy(case.particles)[[0, 2]] ** 2, axis=1)
    np.testing.assert_allclose(
        to_numpy(result["squared_jump_distance"]), jumps
    )
    np.testing.assert_allclose(to_numpy(result["esjd"]), jumps / 2)


class SwitchPolicy:
    def __init__(self) -> None:
        self.calls = 0

    def initialise(
        self, *, initial_weights: tuple[float, ...], context: StepContext
    ) -> tuple[float, ...]:
        return initial_weights

    def update(
        self,
        *,
        current_weights: tuple[float, ...],
        metrics: MutationMetricsOutput,
        context: StepContext,
    ) -> tuple[float, ...]:
        self.calls += 1
        return (0.0, 1.0)

    def state_dict(self) -> dict[str, int]:
        return {"calls": self.calls}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.calls = state["calls"]


def make_mutation(target=None, **kwargs) -> RedBlueMetropolisHastingsMutation:
    return RedBlueMetropolisHastingsMutation(
        (lambda x: np.zeros(len(x))) if target is None else target,
        mixture((1.0, 0.0)),
        n_steps=4,
        **kwargs,
    )


def test_freeze_rounds_iterations_and_policy_state(context_factory) -> None:
    observed = []

    def target(x):
        observed.append(mutation.kernel.weights)
        return np.zeros(len(x))

    mutation = make_mutation(target)
    children = mutation.kernel.kernels
    policy = SwitchPolicy()
    wrapper = AdaptMixtureWeights(
        RepeatedMutationStep(mutation, n_rounds=3, n_steps=7), policy
    )
    state = ParticleState(np.arange(20.0).reshape(10, 2))
    for t, expected in [(0, (1.0, 0.0)), (0, (1.0, 0.0)), (1, (0.0, 1.0))]:
        observed.clear()
        _, _, output = wrapper(
            state, context_factory(t=t), RandomGenerator(seed=8)
        )
        assert all(w == expected for w in observed)
        assert output.metrics.n_steps == 7
        np.testing.assert_array_equal(
            output.metrics.values["mixture_components"]["n_attempted"],
            np.array(expected) * 70,
        )
        assert mutation.kernel.kernels == children
    assert policy.calls == 2
    restored_policy = SwitchPolicy()
    restored = AdaptMixtureWeights(
        RepeatedMutationStep(make_mutation(), n_rounds=3, n_steps=7),
        restored_policy,
    )
    restored.load_state_dict(json.loads(json.dumps(wrapper.state_dict())))
    restored(state, context_factory(t=1), RandomGenerator(seed=8))
    assert restored_policy.calls == 2
    restored(state, context_factory(t=2), RandomGenerator(seed=8))
    assert restored_policy.calls == 3


def make_adaptive() -> AdaptMutationSteps:
    return AdaptMutationSteps(
        AdaptMixtureWeights(
            make_mutation(), ESJDMixtureWeightPolicy(min_attempts=1)
        ),
        AcceptanceRatePolicy(2, min_steps=1, max_steps=8),
        initial_steps=4,
        min_steps=1,
        max_steps=8,
    )


def test_checkpoint_resume_and_budget(context_factory) -> None:
    state = ParticleState(np.random.default_rng(3).normal(size=(40, 2)))
    original = make_adaptive()
    # Also exercise checkpoints before the first call.
    original.load_state_dict(original.state_dict())
    state, _, _ = original(state, context_factory(), RandomGenerator(seed=4))
    restored = make_adaptive()
    restored.load_state_dict(json.loads(json.dumps(original.state_dict())))
    for t in (0, 1, 2):
        expected, _, a = original(
            state, context_factory(t=t), RandomGenerator(seed=5 + t)
        )
        actual, _, b = restored(
            state, context_factory(t=t), RandomGenerator(seed=5 + t)
        )
        np.testing.assert_array_equal(actual.particles, expected.particles)
        assert a["metrics"].n_steps == b["metrics"].n_steps
        assert (
            a["metrics"].values["mixture_weight_adaptation"]
            == b["metrics"].values["mixture_weight_adaptation"]
        )
        for key in a["metrics"].values["mixture_components"]:
            np.testing.assert_array_equal(
                a["metrics"].values["mixture_components"][key],
                b["metrics"].values["mixture_components"][key],
            )
        state = expected


@pytest.mark.parametrize(
    "change",
    [
        {"n_components": 3},
        {"next_weights": (1.0,)},
        {"current_iteration": True},
        {"last_updated_iteration": 5},
        {"next_weights": None},
        {"current_weights": (0.0, 0.0)},
    ],
)
def test_checkpoint_validation(context_factory, change) -> None:
    wrapper = AdaptMixtureWeights(make_mutation())
    wrapper(
        ParticleState(np.arange(20.0).reshape(10, 2)),
        context_factory(),
        RandomGenerator(seed=1),
    )
    with pytest.raises((ValueError, TypeError)):
        wrapper.load_state_dict({**wrapper.state_dict(), **change})


def test_empty_stage_and_eager_only(context_factory) -> None:
    wrapper = AdaptMixtureWeights(make_mutation())
    state = ParticleState(np.arange(20.0).reshape(10, 2))
    _, _, output = wrapper.apply_with_n_steps(
        state, context_factory(), RandomGenerator(seed=1), n_steps=0
    )
    assert output["metrics"].n_attempted == 0
    assert wrapper.current_weights == wrapper.next_weights
    with pytest.raises(NotImplementedError, match="eager"):
        wrapper.apply_functional(state, context_factory(), None)


def test_backend_adaptation(backend, backend_factory, context_factory) -> None:
    case = backend_factory(backend)
    step = make_mutation(lambda x: -case.xp.sum(x * x, axis=-1) / 2)
    wrapper = AdaptMixtureWeights(
        step, ESJDMixtureWeightPolicy(min_attempts=1)
    )
    for t in range(2):
        state, _, output = wrapper(
            case.state(), context_factory(t=t), case.rng(11 + t)
        )
        assert case.matches_namespace(state.particles)
        assert case.matches_namespace(
            output["metrics"].values["mixture_components"]["esjd"]
        )
        assert sum(wrapper.next_weights) == pytest.approx(1)
