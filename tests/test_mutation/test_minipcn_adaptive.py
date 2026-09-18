from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from minipcn import Sampler
from orng import RandomGenerator

from sequince import ParticleState, SequentialMonteCarlo
from sequince.array_api import namespace, to_numpy
from sequince.checkpoint import HDF5CheckpointWriter, SMCCheckpoint
from sequince.steps import (
    AcceptanceRateRhoPolicy,
    AdaptiveMiniPCNMutation,
    AdaptMutationSteps,
    MiniPCNMutation,
    RepeatedMutationStep,
    StepContext,
)
from sequince.steps.base import SMCStep
from sequince.steps.mutation.adaptive.policies.acceptance import (
    AcceptanceRatePolicy,
)
from sequince.steps.mutation.minipcn import _MINIPCN_BLOCK_KEY
from sequince.transforms import WhiteningTransform


def log_prob(x: Any, context: Any = None) -> Any:
    return -0.5 * namespace(x).sum(x**2, axis=-1)


def population() -> ParticleState:
    return ParticleState(np.random.default_rng(31).normal(size=(128, 2)) + 1)


def make_wrapper(**kwargs: Any) -> AdaptiveMiniPCNMutation:
    return AdaptiveMiniPCNMutation(
        MiniPCNMutation(log_prob, step_fn="pcn", n_steps=4),
        initial_rho=kwargs.pop("initial_rho", 0.5),
        policy=AcceptanceRateRhoPolicy(**kwargs),
    )


@pytest.mark.parametrize("step_fn", ["pcn", "tpcn"])
@pytest.mark.parametrize("transformed", [False, True])
def test_repeated_block_matches_fixed_sampler(
    step_fn: str, transformed: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = population()
    fitted = []
    original = Sampler.init_step_state

    def record_fit(self: Sampler, x: Any) -> Any:
        result = original(self, x)
        fitted.append(result)
        return result

    monkeypatch.setattr(Sampler, "init_step_state", record_fit)
    mutation = MiniPCNMutation(
        log_prob,
        step_fn=step_fn,
        n_steps=7,
        store_chains=True,
        transform=WhiteningTransform() if transformed else None,
    )
    wrapper = AdaptiveMiniPCNMutation(
        RepeatedMutationStep(mutation, n_rounds=3, n_steps=7),
        initial_rho=0.4,
    )
    actual, context, output = wrapper(
        state, StepContext(t=0), RandomGenerator(seed=4)
    )
    assert len(fitted) == 1
    assert _MINIPCN_BLOCK_KEY not in context.metadata
    # A standalone frozen kernel with the identical fit and RNG must yield
    # the exact same chain, even though the wrapper split it into rounds.
    transform = mutation._prepared_transform
    x = (
        state.particles
        if transform is None
        else transform.forward(state.particles)[0]
    )

    def target(z: Any) -> Any:
        if transform is None:
            return log_prob(z)
        particles, log_det = transform.inverse(z)
        return log_prob(particles) + log_det

    sampler = Sampler(target, step_fn, dims=2, adaptive=False)
    chain, history = sampler.sample(
        x, 7, rng=RandomGenerator(seed=4), step_state=fitted[0], verbose=False
    )
    expected = (
        chain
        if transform is None
        else np.stack([transform.inverse(z)[0] for z in chain])
    )
    np.testing.assert_allclose(actual.particles, expected[-1], atol=1e-13)
    np.testing.assert_allclose(
        np.concatenate([r.chain for r in output.round_outputs]),
        expected[1:],
        atol=1e-13,
    )
    acceptance = np.mean(history.acceptance_rate)
    assert output.metrics.acceptance == pytest.approx(acceptance)
    assert output.metrics.n_steps == 7
    assert wrapper.current_rho == 0.4
    assert wrapper.next_rho == pytest.approx(
        0.4 * np.exp(0.1 * (acceptance - 0.234))
    )
    assert context.evaluation_counts.log_target_evaluations == 128 * (
        7 + 2 * 3
    )
    np.testing.assert_array_equal(actual.log_weights, state.log_weights)
    assert actual.log_normalizing_constant == state.log_normalizing_constant


def test_geometry_refits_and_rho_updates_only_next_iteration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fitted = []
    original = Sampler.init_step_state

    def record_fit(self: Sampler, x: Any) -> Any:
        result = original(self, x)
        fitted.append(result)
        return result

    monkeypatch.setattr(Sampler, "init_step_state", record_fit)
    wrapper = make_wrapper()
    state = population()
    wrapper(state, StepContext(t=0), RandomGenerator(seed=3))
    next_rho = wrapper.next_rho
    moved = state.replace_particles(state.particles + 4)
    wrapper(moved, StepContext(t=0), RandomGenerator(seed=4))
    assert wrapper.next_rho == next_rho
    assert wrapper.current_rho == 0.5
    wrapper(moved, StepContext(t=1), RandomGenerator(seed=5))
    assert wrapper.current_rho == next_rho
    np.testing.assert_allclose(fitted[1].mu, fitted[0].mu + 4)
    assert [float(f.rho) for f in fitted] == [0.5, 0.5, next_rho]


def test_budget_adapter_checkpoint_resume_and_same_iteration_replay() -> None:
    def make() -> AdaptMutationSteps:
        return AdaptMutationSteps(
            make_wrapper(),
            AcceptanceRatePolicy(2, min_steps=1, max_steps=8),
            initial_steps=4,
            min_steps=1,
            max_steps=8,
        )

    wrapper = make()
    state, _, _ = wrapper(
        population(), StepContext(t=0), RandomGenerator(seed=2)
    )
    saved = json.loads(json.dumps(wrapper.state_dict()))
    restored = make()
    restored.load_state_dict(saved)
    for t in (0, 1, 2):
        expected, _, a = wrapper(
            state, StepContext(t=t), RandomGenerator(seed=3 + t)
        )
        actual, _, b = restored(
            state, StepContext(t=t), RandomGenerator(seed=3 + t)
        )
        np.testing.assert_array_equal(actual.particles, expected.particles)
        assert a["metrics"].n_steps == b["metrics"].n_steps
        assert a["metrics"].values == b["metrics"].values
        state = expected


def test_zero_steps_do_not_fit_draw_or_consume_adaptation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper = make_wrapper()
    state = population()

    def fail(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("Empty blocks must not fit or draw random numbers.")

    with monkeypatch.context() as patch:
        patch.setattr(Sampler, "init_step_state", fail)
        actual, context, output = wrapper.apply_with_n_steps(
            state, StepContext(t=0), None, n_steps=0
        )
    assert actual is state
    assert output.metrics.n_attempted == 0
    assert wrapper.current_rho == wrapper.next_rho == 0.5
    assert _MINIPCN_BLOCK_KEY not in context.metadata
    assert context.evaluation_counts.log_target_evaluations == 0
    wrapper(state, StepContext(t=0), RandomGenerator(seed=3))
    assert wrapper.next_rho != 0.5


@pytest.mark.parametrize("weights", [[0, -1], [0, -np.inf], [np.nan, np.nan]])
def test_nonuniform_or_invalid_weights_rejected(weights: list[float]) -> None:
    state = ParticleState(
        np.array([[0.0], [1.0]]), log_weights=np.array(weights)
    )
    with pytest.raises(ValueError, match="Resample before mutation"):
        make_wrapper()(state, StepContext(t=0), RandomGenerator(seed=2))


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
@pytest.mark.parametrize("step_fn", ["pcn", "tpcn"])
def test_backend_preserved(
    backend: str, backend_factory: Any, step_fn: str
) -> None:
    case = backend_factory(backend)
    wrapper = AdaptiveMiniPCNMutation(
        MiniPCNMutation(log_prob, step_fn=step_fn)
    )
    state = ParticleState(
        case.xp.asarray(population().particles, dtype=case.particles.dtype)
    )
    actual, _, output = wrapper(state, StepContext(t=0), case.rng(5))
    assert case.matches_namespace(actual.particles)
    assert actual.particles.dtype == state.particles.dtype
    assert np.isfinite(to_numpy(actual.particles)).all()
    assert output.metrics.n_steps == 1


@pytest.mark.parametrize("step_fn", ["pcn", "tpcn"])
def test_known_gaussian_remains_stable(step_fn: str) -> None:
    covariance = np.array([[1.0, 0.7], [0.7, 2.0]])
    inverse = np.linalg.inv(covariance)
    state = ParticleState(
        np.random.default_rng(14).multivariate_normal(
            [0.0, 0.0], covariance, size=6000
        )
    )
    wrapper = AdaptiveMiniPCNMutation(
        MiniPCNMutation(
            lambda x: -0.5 * np.sum((x @ inverse) * x, axis=1),
            step_fn=step_fn,
            n_steps=4,
        )
    )
    for t in range(4):
        state, _, _ = wrapper(
            state, StepContext(t=t), RandomGenerator(seed=15 + t)
        )
    # Monte Carlo tolerances for 6000 particles, including off-diagonal moments.
    np.testing.assert_allclose(state.particles.mean(axis=0), 0.0, atol=0.07)
    np.testing.assert_allclose(
        np.cov(state.particles.T), covariance, atol=0.10
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"initial_rho": 0},
        {"min_rho": 0},
        {"max_rho": 1},
        {"adaptation_rate": -1},
        {"adaptation_rate": np.nan},
        {"target_acceptance": 1.1},
        {"initial_rho": np.inf},
    ],
)
def test_invalid_configuration(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        make_wrapper(**kwargs)


def test_invalid_child_and_functional_execution() -> None:
    with pytest.raises(TypeError, match="requires a MiniPCNMutation"):
        AdaptiveMiniPCNMutation(object())
    with pytest.raises(ValueError, match="adaptive=False"):
        AdaptiveMiniPCNMutation(MiniPCNMutation(log_prob, adaptive=True))
    with pytest.raises(NotImplementedError, match="eager"):
        make_wrapper().apply_functional(population(), StepContext(t=0), None)


@pytest.mark.parametrize(
    "change",
    [
        {"current_rho": 0},
        {"next_rho": np.nan},
        {"current_iteration": True},
        {"last_updated_iteration": 2},
        {"configuration": {}},
    ],
)
def test_invalid_checkpoint(change: dict[str, Any]) -> None:
    wrapper = make_wrapper()
    wrapper(population(), StepContext(t=0), RandomGenerator(seed=3))
    with pytest.raises(ValueError):
        wrapper.load_state_dict({**wrapper.state_dict(), **change})


def test_fresh_checkpoint_and_rho_bounds() -> None:
    wrapper = make_wrapper(adaptation_rate=1000)
    wrapper.load_state_dict(json.loads(json.dumps(wrapper.state_dict())))
    wrapper(population(), StepContext(t=0), RandomGenerator(seed=3))
    assert wrapper.next_rho == pytest.approx(wrapper.policy.max_rho)
    # Force low acceptance through the same completed-metrics adaptation hook.
    from sequince.steps.mutation.metrics import MutationMetricsOutput

    context, _ = wrapper._prepare_mutation(StepContext(t=1), None)
    wrapper._adapt(
        MutationMetricsOutput(1, 100, 0),
        context,
        initial_context=context,
        n_steps=1,
    )
    assert wrapper.next_rho == pytest.approx(wrapper.policy.min_rho)


def test_interleaved_moves_reuse_geometry_and_refresh_target() -> None:
    proposals = []
    evaluated = []

    class Reflect(SMCStep):
        def apply(self, state, context, rng):
            proposals.append(context.metadata[_MINIPCN_BLOCK_KEY].proposal)
            return state.replace_particles(-state.particles), context, None

    def target(x: Any, context: StepContext) -> Any:
        evaluated.append(np.array(x, copy=True))
        return log_prob(x)

    child = MiniPCNMutation(target, step_fn="pcn", store_chains=True)
    wrapper = AdaptiveMiniPCNMutation(
        RepeatedMutationStep(
            RepeatedMutationStep(child, n_rounds=2, after_round=Reflect()),
            n_rounds=2,
        )
    )
    state, _, output = wrapper(
        population(), StepContext(t=0), RandomGenerator(seed=4)
    )
    assert len(proposals) == 4
    assert all(proposal is proposals[0] for proposal in proposals)
    assert output.metrics.n_steps == 4
    # Each one-step round evaluates start, proposal and final particles.
    for index in range(1, 4):
        np.testing.assert_array_equal(
            evaluated[index * 3], -evaluated[index * 3 - 1]
        )
    np.testing.assert_array_equal(state.particles, -evaluated[-1])


@pytest.mark.parametrize("refit", ["always", "once"])
def test_runner_hdf5_checkpoint_restores_rng_transform_and_adaptation(
    tmp_path: Path, refit: str
) -> None:
    import h5py

    def make(*writers: Any) -> SequentialMonteCarlo:
        return SequentialMonteCarlo(
            steps=[
                AdaptiveMiniPCNMutation(
                    MiniPCNMutation(
                        log_prob,
                        n_steps=3,
                        step_fn="pcn",
                        transform=WhiteningTransform(),
                        transform_refit=refit,
                    )
                )
            ],
            checkpoint_writers=writers,
        )

    state = population()
    full = make().run(state, num_steps=4, rng=RandomGenerator(seed=9))
    path = tmp_path / "adaptive.h5"
    make(HDF5CheckpointWriter(path, every=1)).run(
        state, num_steps=2, rng=RandomGenerator(seed=9)
    )
    with h5py.File(path, "r") as handle:
        checkpoint = SMCCheckpoint.load_hdf5(handle)
    runner = make()
    resumed = runner.run(runtime=runner.resume_from(checkpoint), num_steps=2)
    for expected, actual in zip(full.states, resumed.states, strict=True):
        np.testing.assert_array_equal(actual.particles, expected.particles)
        np.testing.assert_array_equal(actual.log_weights, expected.log_weights)
        assert actual.log_evidence == expected.log_evidence
    assert _MINIPCN_BLOCK_KEY not in resumed.context.metadata
