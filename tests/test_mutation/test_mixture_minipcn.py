"""MiniPCN preparation, interleaving and bookkeeping regression tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from orng import RandomGenerator

from sequince._typing import Array
from sequince.array_api import namespace, to_numpy
from sequince.state import ParticleState
from sequince.steps import (
    AdaptMoveWeights,
    AdaptMutationSteps,
    JumpDistancePerEvaluation,
    MixtureMutation,
    StepContext,
)
from sequince.steps.mutation import (
    CallableMutationKernel,
    GaussianRandomWalk,
    MetropolisHastingsMutation,
    MiniPCNMutation,
)
from sequince.steps.mutation.adaptive.policies.acceptance import (
    AcceptanceRatePolicy,
)


def log_target(particles: Array) -> Array:
    return -0.5 * namespace(particles).sum(particles**2, axis=-1)


def make_mixture(step_fn: str = "tpcn", **kwargs: Any) -> MixtureMutation:
    return MixtureMutation(
        {
            "walk": MetropolisHastingsMutation(
                log_target, GaussianRandomWalk(scale=0.4)
            ),
            "minipcn": MiniPCNMutation(
                log_target, step_fn=step_fn, n_steps=99, **kwargs
            ),
        },
        n_steps=12,
    )


def initial_state() -> ParticleState:
    return ParticleState(np.random.default_rng(21).normal(size=(128, 2)))


@pytest.mark.parametrize("step_fn", ["pcn", "tpcn"])
def test_preparation_once_from_block_entry(
    step_fn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import minipcn.step

    cls = minipcn.step.PCNStep if step_fn == "pcn" else minipcn.step.TPCNStep
    original_init = cls.init_state
    original_propose = cls.propose
    fits = []
    proposals = []

    def init(step: Any, particles: Array) -> Any:
        fits.append(np.array(particles, copy=True))
        return original_init(step, particles)

    def propose(
        step: Any, prepared: Any, rng_state: Any, particles: Array
    ) -> Any:
        proposals.append(prepared)
        return original_propose(step, prepared, rng_state, particles)

    monkeypatch.setattr(cls, "init_state", init)
    monkeypatch.setattr(cls, "propose", propose)
    mixture = make_mixture(step_fn)
    state = initial_state()
    updated, _, output = mixture(state, StepContext(0), 22)
    assert len(fits) == 1
    np.testing.assert_array_equal(fits[0], state.particles)
    count = output.moves["minipcn"].selections
    assert 1 < count < 12
    assert len(proposals) == count
    assert all(prepared.cov is proposals[0].cov for prepared in proposals)
    assert [prepared.iteration for prepared in proposals] == list(range(count))
    assert output.moves["minipcn"].metrics.n_attempted == count * 128
    # A subsequent block fits afresh; no block-local proposal leaks into it.
    mixture(updated, StepContext(1), 23)
    assert len(fits) == 2
    np.testing.assert_array_equal(fits[1], updated.particles)


@pytest.mark.parametrize("step_fn", ["pcn", "tpcn"])
def test_one_transition_matches_standalone(step_fn: str) -> None:
    state = initial_state()
    child = MiniPCNMutation(log_target, step_fn=step_fn)
    mixture = MixtureMutation({"minipcn": child})
    actual, _, out = mixture(state, StepContext(0), 3)
    rng = RandomGenerator(seed=3)
    rng.choice(1, size=(1,), p=(1.0,))
    expected, _, standalone = child(state, StepContext(0), rng)
    np.testing.assert_array_equal(actual.particles, expected.particles)
    assert out.metrics.n_accepted == standalone.metrics.n_accepted
    assert out.evaluation_counts.log_target_evaluations == 3 * 128


@pytest.mark.parametrize("n_steps,weights", [(0, (1, 1)), (4, (1, 0))])
def test_zero_budget_or_weight_does_not_prepare(
    n_steps: int, weights: tuple[int, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("An inactive MiniPCN child was prepared.")

    monkeypatch.setattr(MiniPCNMutation, "_initialize_mixture_state", fail)
    _, _, out = make_mixture().apply_with_n_steps(
        initial_state(),
        StepContext(0),
        RandomGenerator(seed=4),
        n_steps=n_steps,
        weights=weights,
    )
    assert out.moves["minipcn"].selections == 0
    assert out.moves["minipcn"].evaluation_counts.log_target_evaluations == 0


def test_reject_internal_adaptation() -> None:
    with pytest.raises(ValueError, match="adaptive=False"):
        make_mixture(adaptive=True)


def test_actual_target_calls_and_particle_identity() -> None:
    evaluated = 0

    def target(particles: Array) -> Array:
        nonlocal evaluated
        evaluated += particles.shape[0]
        return log_target(particles)

    mixture = make_mixture()
    for child in mixture.moves.values():
        child._log_prob_fn = target
    state = initial_state()
    updated, context, output = mixture(state, StepContext(0), 9)
    assert evaluated == output.evaluation_counts.log_target_evaluations
    assert evaluated == context.evaluation_counts.log_target_evaluations
    assert evaluated == sum(
        move.evaluation_counts.log_target_evaluations
        for move in output.moves.values()
    )
    np.testing.assert_array_equal(updated.log_weights, state.log_weights)
    np.testing.assert_array_equal(
        updated.initial_ancestor_indices, state.initial_ancestor_indices
    )


@pytest.mark.parametrize("step_fn", ["pcn", "tpcn"])
def test_joint_checkpoint_resume(step_fn: str) -> None:
    def make() -> AdaptMutationSteps:
        return AdaptMutationSteps(
            AdaptMoveWeights(
                make_mixture(step_fn), JumpDistancePerEvaluation()
            ),
            AcceptanceRatePolicy(3, min_steps=2, max_steps=20),
            initial_steps=8,
            min_steps=2,
            max_steps=20,
        )

    mutation = make()
    rng = RandomGenerator(seed=45)
    state, _, _ = mutation(initial_state(), StepContext(0), rng)
    saved = mutation.state_dict()
    restored = make()
    restored.load_state_dict(saved)
    other_rng = RandomGenerator.from_state_dict(rng.state_dict())
    for iteration in (0, 1, 2):
        actual, _, out = mutation(state, StepContext(iteration), rng)
        expected, _, other = restored(state, StepContext(iteration), other_rng)
        np.testing.assert_array_equal(actual.particles, expected.particles)
        assert out.moves == other.moves
        assert out.weights == other.weights
        assert out.metrics.n_steps == other.metrics.n_steps
        state = actual


@pytest.mark.parametrize("step_fn", ["pcn", "tpcn"])
def test_array_backends(
    step_fn: str, backend: str, backend_factory: Callable[[str], Any]
) -> None:
    case = backend_factory(backend)
    rng = case.rng(6)
    state = ParticleState(rng.normal(size=(128, 2)))
    updated, _, output = make_mixture(step_fn)(state, StepContext(0), rng)
    assert case.matches_namespace(updated.particles)
    assert updated.particles.dtype == state.particles.dtype
    assert np.all(np.isfinite(to_numpy(updated.particles)))
    assert output.metrics.n_attempted == 12 * 128
    assert output.moves["minipcn"].selections > 0


@pytest.mark.parametrize("step_fn", ["pcn", "tpcn"])
def test_metric_state_and_actual_interleaving(step_fn: str) -> None:
    order = []

    class CountMetric:
        name = "count"

        def initialize(
            self, particles: Array, log_weights: Array, context: StepContext
        ) -> int:
            return 0

        def update(
            self,
            state: int,
            particles: Array,
            accepted: Array,
            context: StepContext,
        ) -> int:
            order.append("minipcn")
            return state + 1

        def result(self, state: int, context: StepContext) -> int:
            return state

    def sample(particles: Array, rng: RandomGenerator) -> Array:
        order.append("walk")
        return particles

    mixture = MixtureMutation(
        {
            "walk": MetropolisHastingsMutation(
                log_target, CallableMutationKernel(sample, symmetric=True)
            ),
            "minipcn": MiniPCNMutation(
                log_target, step_fn=step_fn, metrics=[CountMetric()]
            ),
        },
        n_steps=20,
    )
    _, _, output = mixture(initial_state(), StepContext(0), 10)
    assert sum(a != b for a, b in zip(order, order[1:])) > 2
    move = output.moves["minipcn"]
    assert move.metrics.values["count"] == move.selections
    assert output.moves["walk"].metrics.values == {}


def test_target_cache_and_likelihood_costs() -> None:
    from sequince.posterior import TemperedPosterior

    likelihood_calls = 0

    def likelihood(particles: Array) -> Array:
        nonlocal likelihood_calls
        likelihood_calls += particles.shape[0]
        return -0.5 * np.sum((particles - 0.5) ** 2, axis=1)

    target = TemperedPosterior(log_prior=log_target, log_likelihood=likelihood)
    mixture = MixtureMutation(
        {
            "walk": MetropolisHastingsMutation(
                target, GaussianRandomWalk(scale=0.4)
            ),
            "minipcn": MiniPCNMutation(target, step_fn="pcn"),
        },
        n_steps=10,
    )
    state, context, out = mixture(
        initial_state(), StepContext(0, metadata={"temperature": 0.5}), 16
    )
    assert out.moves[
        "minipcn"
    ].evaluation_counts.log_likelihood_evaluations == (
        3 * 128 * out.moves["minipcn"].selections
    )
    assert out.evaluation_counts.log_likelihood_evaluations == likelihood_calls
    assert out.evaluation_counts.log_target_evaluations == likelihood_calls
    assert (
        context.evaluation_counts.log_likelihood_evaluations
        == likelihood_calls
    )
    assert state.evaluation is not None
    np.testing.assert_allclose(
        state.evaluation["log_likelihood"], likelihood(state.particles)
    )
