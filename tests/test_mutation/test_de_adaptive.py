from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
from orng import RandomGenerator

from sequince import ParticleState
from sequince.steps import (
    AdaptDEScaleWeights,
    AdaptMutationSteps,
    DifferentialEvolutionMutation,
    ESJDScaleWeightPolicy,
    MutationMetricsOutput,
    RepeatedMutationStep,
)
from sequince.steps.mutation.adaptive.policies.acceptance import (
    AcceptanceRatePolicy,
)


def metrics(
    attempts=(100, 100, 100), jumps=(100.0, 900.0, 1e9)
) -> MutationMetricsOutput:
    return MutationMetricsOutput(
        10,
        sum(attempts),
        100,
        {
            "de_scales": {
                "component": np.array([0, 1, -1]),
                "n_attempted": np.asarray(attempts),
                "squared_jump_distance": np.asarray(jumps),
            }
        },
    )


def test_policy_uses_esjd_not_acceptance_or_unit_jumps(
    context_factory,
) -> None:
    policy = ESJDScaleWeightPolicy(
        learning_rate=1, exploration_probability=0.2
    )
    updated = policy.update(
        current_weights=(0.5, 0.5),
        metrics=metrics(),
        context=context_factory(),
    )
    np.testing.assert_allclose(updated, [0.18, 0.82])
    # Same ESJD with different attempted counts must give the same weights.
    np.testing.assert_allclose(
        policy.update(
            current_weights=(0.5, 0.5),
            metrics=metrics((200, 100, 0), (200, 900, 0)),
            context=context_factory(),
        ),
        updated,
    )


def test_smoothing_and_exploration_floor(context_factory) -> None:
    policy = ESJDScaleWeightPolicy(
        learning_rate=0.5, exploration_probability=0.2
    )
    context = context_factory()
    current = policy.initialise(initial_weights=(1.0, 0.0), context=context)
    np.testing.assert_allclose(current, [0.9, 0.1])
    updated = policy.update(
        current_weights=current,
        metrics=metrics(jumps=(0, 900, 0)),
        context=context,
    )
    np.testing.assert_allclose(updated, [0.5, 0.5])
    for _ in range(30):
        updated = policy.update(
            current_weights=updated,
            metrics=metrics(jumps=(0, 900, 0)),
            context=context,
        )
        assert min(updated) >= 0.1 - 1e-15
        assert sum(updated) == pytest.approx(1)


@pytest.mark.parametrize(
    "attempts,jumps",
    [
        ((0, 0, 100), (0, 0, 100)),
        ((100, 1, 0), (50, 10, 0)),
        ((100, 100, 100), (0, 0, 100)),
    ],
)
def test_sparse_or_stalled_blocks_retain_weights(
    context_factory, attempts, jumps
) -> None:
    updated = ESJDScaleWeightPolicy().update(
        current_weights=(0.3, 0.7),
        metrics=metrics(attempts, jumps),
        context=context_factory(),
    )
    np.testing.assert_allclose(updated, [0.3, 0.7])


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(learning_rate=0),
        dict(learning_rate=np.nan),
        dict(learning_rate=1.1),
        dict(exploration_probability=0),
        dict(exploration_probability=np.inf),
        dict(min_attempts=0),
        dict(min_attempts=True),
        dict(min_attempts=1.5),
    ],
)
def test_invalid_policy_configuration(kwargs) -> None:
    with pytest.raises((TypeError, ValueError)):
        ESJDScaleWeightPolicy(**kwargs)


@pytest.mark.parametrize(
    "attempts,jumps",
    [
        ((-1, 10, 0), (0, 10, 0)),
        ((1.5, 10, 0), (1, 10, 0)),
        ((100, 100, 0), (np.nan, 1, 0)),
        ((100, 100, 0), (-1, 1, 0)),
        ((100, 100, 0), (1, 1, 3)),
        ((100, 100, 0), (np.inf, 1, 0)),
    ],
)
def test_invalid_diagnostics_are_rejected(
    context_factory, attempts, jumps
) -> None:
    with pytest.raises(ValueError):
        ESJDScaleWeightPolicy().update(
            current_weights=(0.5, 0.5),
            metrics=metrics(attempts, jumps),
            context=context_factory(),
        )


def test_missing_or_misidentified_metrics_are_rejected(
    context_factory,
) -> None:
    for value in (
        {},
        {"de_scales": {}},
        {
            "de_scales": {
                **metrics().values["de_scales"],
                "component": np.array([1, 0, -1]),
            }
        },
    ):
        with pytest.raises(ValueError):
            ESJDScaleWeightPolicy().update(
                current_weights=(0.5, 0.5),
                metrics=replace(metrics(), values=value),
                context=context_factory(),
            )


class SwitchPolicy:
    def __init__(self) -> None:
        self.calls = 0

    def initialise(self, *, initial_weights, context):
        return initial_weights

    def update(self, *, current_weights, metrics, context):
        self.calls += 1
        return (0.0, 1.0)

    def state_dict(self):
        return {"calls": self.calls}

    def load_state_dict(self, state):
        self.calls = state["calls"]


def make_mutation(target=None, **kwargs) -> DifferentialEvolutionMutation:
    return DifferentialEvolutionMutation(
        (lambda x: np.zeros(len(x))) if target is None else target,
        gamma_one_probability=0,
        scale_multipliers=(0.2, 1.0),
        scale_weights=(1.0, 0.0),
        n_steps=4,
        **kwargs,
    )


def test_weights_freeze_across_rounds_and_repeated_iteration(
    context_factory,
) -> None:
    observed = []

    def target(x):
        observed.append(mutation.kernel.scale_weights)
        return np.zeros(len(x))

    mutation = make_mutation(target)
    policy = SwitchPolicy()
    wrapper = AdaptDEScaleWeights(
        RepeatedMutationStep(mutation, n_rounds=3, n_steps=7), policy
    )
    state = ParticleState(np.arange(20.0).reshape(10, 2))
    for t, expected in [(0, (1.0, 0.0)), (0, (1.0, 0.0)), (1, (0.0, 1.0))]:
        observed.clear()
        _, _, output = wrapper(
            state, context_factory(t=t), RandomGenerator(seed=8)
        )
        assert all(w == expected for w in observed)
        diagnostics = output.metrics.values["de_scale_adaptation"]
        assert diagnostics["weights"] == expected
        assert diagnostics["next_weights"] == (0.0, 1.0)
        assert output.metrics.n_steps == 7
        np.testing.assert_array_equal(
            output.metrics.values["de_scales"]["n_attempted"],
            [70 * expected[0], 70 * expected[1], 0],
        )
    assert policy.calls == 2
    assert mutation.kernel.gamma_one_probability == 0


def make_adaptive():
    weights = AdaptDEScaleWeights(
        make_mutation(), ESJDScaleWeightPolicy(min_attempts=1)
    )
    return AdaptMutationSteps(
        weights,
        AcceptanceRatePolicy(2, min_steps=1, max_steps=8),
        initial_steps=4,
        min_steps=1,
        max_steps=8,
    )


def test_both_controls_checkpoint_and_seeded_resume(context_factory) -> None:
    state = ParticleState(np.random.default_rng(5).normal(size=(40, 2)))
    original = make_adaptive()
    state, _, first = original(
        state, context_factory(t=0), RandomGenerator(seed=6)
    )
    checkpoint = json.loads(json.dumps(original.state_dict()))
    assert original.next_n_steps == 2
    assert first["metrics"].n_steps == 4
    restored = make_adaptive()
    restored.load_state_dict(checkpoint)
    expected, _, a = original(
        state, context_factory(t=1), RandomGenerator(seed=7)
    )
    actual, _, b = restored(
        state, context_factory(t=1), RandomGenerator(seed=7)
    )
    np.testing.assert_array_equal(actual.particles, expected.particles)
    assert a["metrics"].n_steps == b["metrics"].n_steps == 2
    assert (
        a["metrics"].values["de_scale_adaptation"]
        == b["metrics"].values["de_scale_adaptation"]
    )
    for name in a["metrics"].values["de_scales"]:
        np.testing.assert_array_equal(
            a["metrics"].values["de_scales"][name],
            b["metrics"].values["de_scales"][name],
        )


def test_checkpoint_replay_does_not_readapt(context_factory) -> None:
    policy = SwitchPolicy()
    original = AdaptDEScaleWeights(make_mutation(), policy)
    state = ParticleState(np.arange(20.0).reshape(10, 2))
    original(state, context_factory(t=3), RandomGenerator(seed=4))
    restored_policy = SwitchPolicy()
    restored = AdaptDEScaleWeights(make_mutation(), restored_policy)
    restored.load_state_dict(original.state_dict())
    _, _, output = restored(
        state, context_factory(t=3), RandomGenerator(seed=4)
    )
    assert restored_policy.calls == 1
    assert output["metrics"].values["de_scale_adaptation"]["weights"] == (
        1.0,
        0.0,
    )
    restored(state, context_factory(t=4), RandomGenerator(seed=5))
    assert restored.current_weights == (0.0, 1.0)
    assert restored_policy.calls == 2


@pytest.mark.parametrize(
    "change",
    [
        {"next_weights": (1.0,)},
        {"scale_multipliers": (0.1, 1.0)},
        {"current_iteration": True},
        {"current_weights": (0.0, 0.0)},
        {"last_updated_iteration": 1},
        {"next_weights": None},
    ],
)
def test_invalid_checkpoint_rejected(context_factory, change) -> None:
    wrapper = AdaptDEScaleWeights(make_mutation())
    wrapper(
        ParticleState(np.arange(20.0).reshape(10, 2)),
        context_factory(t=0),
        RandomGenerator(seed=1),
    )
    with pytest.raises((TypeError, ValueError)):
        wrapper.load_state_dict({**wrapper.state_dict(), **change})


def test_empty_stage_preserves_weights(context_factory) -> None:
    wrapper = AdaptDEScaleWeights(make_mutation())
    _, _, output = wrapper.apply_with_n_steps(
        ParticleState(np.arange(20.0).reshape(10, 2)),
        context_factory(),
        RandomGenerator(seed=1),
        n_steps=0,
    )
    assert output["metrics"].n_attempted == 0
    assert wrapper.current_weights == wrapper.next_weights


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
def test_backend_adaptation(backend, backend_factory, context_factory) -> None:
    case = backend_factory(backend)
    mutation = make_mutation(lambda x: -case.xp.sum(x * x, axis=1))
    wrapper = AdaptDEScaleWeights(
        mutation, ESJDScaleWeightPolicy(min_attempts=1)
    )
    for t in range(2):
        state, _, output = wrapper(
            case.state(), context_factory(t=t), case.rng(11 + t)
        )
        assert case.matches_namespace(state.particles)
        assert case.matches_namespace(
            output["metrics"].values["de_scales"]["esjd"]
        )
        assert sum(wrapper.next_weights) == pytest.approx(1)


def test_correlated_gaussian_remains_stable_across_adaptation(
    context_factory,
) -> None:
    covariance = np.array([[1.0, 0.7], [0.7, 2.0]])
    inverse = np.linalg.inv(covariance)
    state = ParticleState(
        np.random.default_rng(14).multivariate_normal(
            [0.0, 0.0], covariance, size=6000
        )
    )
    mutation = make_mutation(
        lambda x: -0.5 * np.sum((x @ inverse) * x, axis=1)
    )
    adaptive = AdaptMutationSteps(
        AdaptDEScaleWeights(mutation),
        AcceptanceRatePolicy(3, min_steps=4, max_steps=20),
        initial_steps=8,
        min_steps=4,
        max_steps=20,
    )
    for t in range(4):
        state, _, _ = adaptive(
            state, context_factory(t=t), RandomGenerator(seed=15 + t)
        )
    np.testing.assert_allclose(state.particles.mean(axis=0), 0.0, atol=0.07)
    np.testing.assert_allclose(
        np.cov(state.particles.T), covariance, atol=0.10
    )


@pytest.mark.parametrize(
    "weights", [(1.0,), (0.0, 0.0), (-1.0, 2.0), (np.nan, 1.0)]
)
def test_bad_policy_decisions_are_rejected(context_factory, weights) -> None:
    class BadPolicy(SwitchPolicy):
        def update(self, **kwargs):
            return weights

    wrapper = AdaptDEScaleWeights(make_mutation(), BadPolicy())
    with pytest.raises(ValueError):
        wrapper(
            ParticleState(np.arange(20.0).reshape(10, 2)),
            context_factory(),
            RandomGenerator(seed=2),
        )


def test_fresh_checkpoint_and_unit_only_stage(context_factory) -> None:
    mutation = make_mutation()
    mutation.kernel = replace(mutation.kernel, gamma_one_probability=1)
    wrapper = AdaptDEScaleWeights(mutation)
    checkpoint = wrapper.state_dict()
    wrapper.load_state_dict(checkpoint)
    _, _, output = wrapper(
        ParticleState(np.arange(20.0).reshape(10, 2)),
        context_factory(),
        RandomGenerator(seed=2),
    )
    assert wrapper.current_weights == wrapper.next_weights
    np.testing.assert_array_equal(
        output["metrics"].values["de_scales"]["n_attempted"], [0, 0, 40]
    )
    assert mutation.kernel.gamma_one_probability == 1


def test_singleton_and_functional_rejection(context_factory) -> None:
    wrapper = AdaptDEScaleWeights(
        DifferentialEvolutionMutation(lambda x: np.zeros(len(x)))
    )
    state = ParticleState(np.arange(20.0).reshape(10, 2))
    wrapper(state, context_factory(), RandomGenerator(seed=2))
    assert wrapper.current_weights == wrapper.next_weights == (1.0,)
    with pytest.raises(NotImplementedError, match="eager"):
        wrapper.apply_functional(state, context_factory(), None)


def test_constructor_rejects_non_de_mutation() -> None:
    from sequince.steps import MetropolisHastingsMutation
    from sequince.steps.mutation import GaussianRandomWalk

    with pytest.raises(TypeError, match="DE mutation"):
        AdaptDEScaleWeights(
            MetropolisHastingsMutation(
                lambda x: np.zeros(len(x)), GaussianRandomWalk()
            )
        )
