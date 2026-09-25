"""NS-SMC estimator and terminal transitions across execution modes."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
from orng import RandomGenerator

from sequince import ParticleState, make_nested_sampling_smc
from sequince.checkpoint import HDF5CheckpointWriter, SMCCheckpoint
from sequince.result import SMCResult
from sequince.steps.context import StepContext

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
from sequince.jax import compile_functional_iteration  # noqa: E402


def make_sampler(**kwargs: Any) -> Any:
    options = dict(
        mutation="none",
        particle_transform=None,
        logging_frequency=0,
        evidence_tolerance=None,
        threshold_schedule=[1.0, 2.0],
    )
    options.update(kwargs)
    return make_nested_sampling_smc(
        lambda x: jnp.zeros_like(x[..., 0]), lambda x: x[..., 0], **options
    )


def run_mode(mode: str, state: ParticleState, **kwargs: Any) -> Any:
    sampler, _, metadata = make_sampler(**kwargs)
    key = jax.random.key(42)
    if mode == "eager":
        rng = RandomGenerator(backend="jax", generator=key)
        result = sampler.run(state, metadata=metadata, rng=rng, num_steps=20)
        return result.as_nested_sampling_result(), rng.state()
    iteration = (
        compile_functional_iteration(sampler, state, key, metadata=metadata)
        if mode == "compiled"
        else None
    )
    result, key = sampler.run_functional(
        state,
        metadata=metadata,
        random_state=key,
        num_steps=20,
        iteration=iteration,
    )
    return result.as_nested_sampling_result(), key


def assert_results(actual: Any, expected: Any) -> None:
    assert actual.complete == expected.complete
    assert actual.stop_reason == expected.stop_reason
    assert actual.log_evidence == pytest.approx(
        float(expected.log_evidence), abs=2e-6
    )
    assert len(actual.levels()) == len(expected.levels())
    for name in (
        "threshold_history",
        "log_prior_volume_history",
        "log_evidence_history",
        "survival_fraction_history",
        "surviving_particles_history",
    ):
        np.testing.assert_allclose(
            getattr(actual, name)(), getattr(expected, name)(), atol=2e-6
        )
    for a, b in zip(
        actual.posterior_samples(), expected.posterior_samples(), strict=True
    ):
        np.testing.assert_allclose(a, b, atol=2e-6)
    np.testing.assert_allclose(
        actual.final_state.particles, expected.final_state.particles, atol=2e-6
    )


@pytest.mark.parametrize("mode", ["functional", "compiled"])
@pytest.mark.parametrize("inclusive", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_fixed_partition(mode: str, inclusive: bool, weighted: bool) -> None:
    state = ParticleState(
        jnp.arange(4.0)[:, None],
        log_weights=(
            jnp.log(jnp.asarray([0.1, 0.2, 0.3, 0.4])) if weighted else None
        ),
    )
    options = dict(threshold_inclusive=inclusive)
    expected, expected_key = run_mode("eager", state, **options)
    actual, key = run_mode(mode, state, **options)
    assert_results(actual, expected)
    np.testing.assert_array_equal(
        jax.random.key_data(key), jax.random.key_data(expected_key)
    )
    for shell in actual._shell_outputs():
        assert shell["particles"].shape == (4, 1)
        assert shell["log_weights"].shape == (4,)
        assert np.all(
            np.isneginf(
                np.asarray(shell["log_weights"])[~np.asarray(shell["mask"])]
            )
        )


@pytest.mark.parametrize("mode", ["functional", "compiled"])
@pytest.mark.parametrize(
    "options",
    [
        dict(min_log_prior_volume=-0.5),
        dict(evidence_tolerance=0.9, threshold_schedule=[2.0, 2.5]),
        dict(threshold_schedule=[math.inf]),
    ],
)
def test_terminal_rng_and_skipped_mutation(
    mode: str, options: dict[str, Any]
) -> None:
    mutation = dict(
        mutation="blackjax",
        mutation_n_steps=1,
        mutation_kwargs={
            "algorithm": "random-walk",
            "algorithm_kwargs": {"sigma": 0.1},
        },
    )
    state = ParticleState(jnp.arange(4.0)[:, None])
    expected, expected_key = run_mode("eager", state, **options, **mutation)
    actual, key = run_mode(mode, state, **options, **mutation)
    assert_results(actual, expected)
    np.testing.assert_array_equal(
        jax.random.key_data(key), jax.random.key_data(expected_key)
    )
    if "min_log_prior_volume" in options:
        assert len(actual) == 1
        assert len(actual.levels()) == 2
        np.testing.assert_array_equal(
            jax.random.key_data(key), jax.random.key_data(jax.random.key(42))
        )
        expected_evidence = jnp.log(jnp.mean(jnp.exp(state.particles[:, 0])))
        assert actual.log_evidence == pytest.approx(
            float(expected_evidence), abs=1e-6
        )
    terminal_mutation = actual.result.outputs[-1][-1]
    assert int(terminal_mutation.metrics.n_attempted) == 0
    assert (
        actual.context.evaluation_counts.log_likelihood_evaluations
        == expected.context.evaluation_counts.log_likelihood_evaluations
    )


@pytest.mark.parametrize("mode", ["functional", "compiled"])
@pytest.mark.parametrize(
    "values,weights,schedule",
    [
        ([0.0, 0.0, 1.0, 1.0], None, None),
        ([2.0, 2.0, 2.0, 2.0], None, "worst"),
        ([-math.inf, -math.inf, 0.0, 2.0], None, [-math.inf, -10.0]),
        ([0.0, 1.0, 2.0, 3.0], [-math.inf, -math.inf, 0.0, 0.0], "worst"),
        ([0.0, 1.0, 2.0, 3.0], [-math.inf, -2.0, -1.0, 0.0], None),
    ],
)
def test_adaptive_ties_and_zero_weights(
    mode: str, values: list[float], weights: Any, schedule: Any
) -> None:
    state = ParticleState(
        jnp.asarray(values)[:, None],
        log_weights=(None if weights is None else jnp.asarray(weights)),
    )
    expected, _ = run_mode("eager", state, threshold_schedule=schedule)
    actual, _ = run_mode(mode, state, threshold_schedule=schedule)
    assert_results(actual, expected)


@pytest.mark.parametrize("store", [False, True])
def test_cache_initialization_and_stable_signature(store: bool) -> None:
    calls = []

    def likelihood(x: Any) -> Any:
        calls.append(x.shape)
        return x[..., 0]

    sampler, target, metadata = make_nested_sampling_smc(
        lambda x: jnp.zeros_like(x[..., 0]),
        likelihood,
        threshold_schedule=[-1.0, 1.0, 2.0],
        evidence_tolerance=None,
        store_shell_particles=store,
        mutation="none",
        logging_frequency=0,
    )
    initial = ParticleState(
        jnp.arange(4.0)[:, None],
        metadata={"nested_sampling": {"is_terminal": True}},
    )
    state, context = sampler.initialize_functional(
        initial, StepContext(t=0, metadata=metadata)
    )
    assert len(calls) == 1
    assert not context.metadata["nested_sampling"]["is_terminal"]
    key = jax.random.key(42)
    compiled = compile_functional_iteration(
        sampler, state, key, metadata=context.metadata
    )
    result, _ = sampler.run_functional(
        state,
        random_state=key,
        metadata=context.metadata,
        iteration=compiled,
        num_steps=10,
    )
    assert len(calls) == 1
    assert result.final_state.evaluation.target_id == target.evaluation_id
    assert (
        context.metadata["nested_sampling"]["log_nested_evidence"] == -math.inf
    )
    assert (
        result.states[0].metadata["nested_sampling"]
        is not result.context.metadata["nested_sampling"]
    )
    ns = result.as_nested_sampling_result()
    if not store:
        assert all("particles" not in o for o in ns._shell_outputs())
        with pytest.raises(ValueError, match="store_shell_particles"):
            ns.posterior_samples()
    else:
        assert ns.posterior_samples()[0].shape[1] == 1


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("mutation", ["none", "blackjax"])
def test_checkpoint_resume_new_target_and_portable_result(
    tmp_path: Path, compiled: bool, mutation: str
) -> None:
    options = dict(mutation=mutation, particle_transform="affine")
    if mutation == "blackjax":
        options["mutation_kwargs"] = {
            "algorithm": "random-walk",
            "algorithm_kwargs": {"sigma": 0.1},
        }
    state = ParticleState(jnp.arange(4.0)[:, None])
    expected, expected_key = run_mode(
        "compiled" if compiled else "functional", state, **options
    )
    path = tmp_path / "checkpoint.h5"
    sampler, _, metadata = make_sampler(
        checkpoint_writers=[HDF5CheckpointWriter(path)], **options
    )
    partial, _ = sampler.run_functional(
        state, metadata=metadata, random_state=jax.random.key(42), num_steps=1
    )
    with pytest.raises(ValueError, match="incomplete"):
        partial.as_nested_sampling_result().posterior_samples()
    with h5py.File(path, "r") as handle:
        checkpoint = SMCCheckpoint.load_hdf5(handle)
    sampler, target, _ = make_sampler(**options)
    runtime = sampler.resume_functional_from(checkpoint)
    iteration = (
        compile_functional_iteration(
            sampler, runtime.state, runtime.rng, metadata=runtime.metadata
        )
        if compiled
        else None
    )
    result, key = sampler.run_functional(
        runtime=runtime, num_steps=20, iteration=iteration
    )
    assert result.final_state.evaluation.target_id == target.evaluation_id
    assert_results(result.as_nested_sampling_result(), expected)
    np.testing.assert_array_equal(
        jax.random.key_data(key), jax.random.key_data(expected_key)
    )
    with h5py.File(tmp_path / "result.h5", "w") as handle:
        result.save(handle, strict=True)
    with h5py.File(tmp_path / "result.h5", "r") as handle:
        loaded = SMCResult.load(handle).as_nested_sampling_result()
    assert_results(loaded, expected)


@pytest.mark.parametrize("mode", ["functional", "compiled"])
def test_zero_evidence(mode: str) -> None:
    result, key = run_mode(
        mode,
        ParticleState(jnp.full((4, 1), -math.inf)),
        threshold_schedule=None,
    )
    assert result.complete
    assert result.log_evidence == -math.inf
    np.testing.assert_array_equal(
        jax.random.key_data(key), jax.random.key_data(jax.random.key(42))
    )
    with pytest.raises(ValueError, match="undefined"):
        result.posterior_samples()


@pytest.mark.parametrize("mode", ["functional", "compiled"])
@pytest.mark.parametrize("invalid", [math.nan, math.inf])
def test_invalid_cached_likelihood(mode: str, invalid: float) -> None:
    with pytest.raises(ValueError, match="NaN or positive infinity"):
        run_mode(mode, ParticleState(jnp.asarray([[invalid], [0.0]])))


@pytest.mark.parametrize("mode", ["functional", "compiled"])
@pytest.mark.parametrize(
    "schedule,message",
    [
        (lambda values, state, context: jnp.asarray(math.nan), "not NaN"),
        (
            lambda values, state, context: jnp.where(
                context.t == 0, 0.0, -1.0
            ),
            "nondecreasing",
        ),
        (lambda values, state, context: jnp.asarray(0.0), "increase"),
    ],
)
def test_invalid_schedules(mode: str, schedule: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        run_mode(
            mode,
            ParticleState(jnp.arange(4.0)[:, None]),
            threshold_schedule=schedule,
        )


def test_uncompiled_schedule_stopiteration_and_exceptions() -> None:
    calls = []

    def schedule(
        values: Any, state: ParticleState, context: StepContext
    ) -> float:
        calls.append(int(context.t))
        if context.t == 1:
            raise StopIteration
        return 1.0

    result, _ = run_mode(
        "functional",
        ParticleState(jnp.arange(4.0)[:, None]),
        threshold_schedule=schedule,
    )
    assert calls == [0, 1]
    assert result.stop_reason == "schedule_exhausted"
    for error in (TypeError, ValueError, RuntimeError):

        def broken(values: Any, state: Any, context: Any) -> float:
            raise error("schedule failed")

        with pytest.raises(error, match="schedule failed"):
            run_mode(
                "functional",
                ParticleState(jnp.arange(4.0)[:, None]),
                threshold_schedule=broken,
            )


@pytest.mark.parametrize(
    "options,message",
    [
        ({"always_resample": False}, "unconditional"),
        ({"resampling_kwargs": {"n_samples": 8}}, "population"),
        ({"logging_frequency": 1}, "logging_frequency=0"),
        ({"mutation": "emcee"}, "EmceeMutation"),
        ({"mutation": "eryn"}, "ErynMutation"),
        (
            {"mutation": "differential-evolution"},
            "DifferentialEvolutionMutation",
        ),
    ],
)
def test_unsupported_configurations(
    options: dict[str, Any], message: str
) -> None:
    if options.get("mutation") == "eryn":
        pytest.importorskip("eryn")
    with pytest.raises(NotImplementedError, match=message):
        run_mode(
            "functional", ParticleState(jnp.arange(4.0)[:, None]), **options
        )


@pytest.mark.parametrize("mode", ["functional", "compiled"])
def test_bound_validation_and_evidence_stop(mode: str) -> None:
    state = ParticleState(jnp.arange(4.0)[:, None])
    with pytest.raises(ValueError, match="exceeds"):
        run_mode(mode, state, log_likelihood_upper_bound=2.0)
    options = dict(
        evidence_tolerance=0.9,
        threshold_schedule=[2.0, 2.5],
        log_likelihood_upper_bound=10.0,
    )
    expected, _ = run_mode("eager", state, **options)
    actual, _ = run_mode(mode, state, **options)
    assert_results(actual, expected)
    assert len(actual) == 3


@pytest.mark.parametrize("mutation", ["blackjax", "minipcn"])
@pytest.mark.parametrize("mode", ["eager", "functional", "compiled"])
def test_gaussian_reference(mode: str, mutation: str) -> None:
    def gaussian(x: Any) -> Any:
        return -0.5 * jnp.sum(x**2, axis=-1)

    sampler, _, metadata = make_nested_sampling_smc(
        gaussian,
        gaussian,
        mutation=mutation,
        mutation_n_steps=5,
        mutation_kwargs=(
            {"algorithm": "random-walk", "algorithm_kwargs": {"sigma": 0.2}}
            if mutation == "blackjax"
            else {}
        ),
        logging_frequency=0,
    )
    state = ParticleState(
        jnp.asarray(np.random.default_rng(42).normal(size=(256, 1)))
    )
    key = jax.random.key(42)
    if mode == "eager":
        result = sampler.run(
            state,
            metadata=metadata,
            rng=RandomGenerator(backend="jax", generator=key),
            num_steps=30,
        )
    else:
        iteration = (
            compile_functional_iteration(
                sampler, state, key, metadata=metadata
            )
            if mode == "compiled"
            else None
        )
        result, _ = sampler.run_functional(
            state,
            random_state=key,
            metadata=metadata,
            num_steps=30,
            iteration=iteration,
        )
    ns = result.as_nested_sampling_result()
    assert ns.complete
    assert ns.log_evidence == pytest.approx(-0.5 * math.log(2), abs=0.15)
    samples, weights = ns.posterior_samples()
    assert float(jnp.sum(weights * samples[:, 0])) == pytest.approx(
        0.0, abs=0.15
    )
    assert float(jnp.sum(weights * samples[:, 0] ** 2)) == pytest.approx(
        0.5, abs=0.15
    )


def test_terminal_branch_does_not_execute_mutation() -> None:
    from sequince.steps.base import SMCStep

    calls = []

    class ObservedMutation(SMCStep):
        def apply_functional(self, state: Any, context: Any, key: Any) -> Any:
            jax.debug.callback(lambda t: calls.append(int(t)), context.t)
            return state, context, jax.random.split(key)[0], None

    state = ParticleState(jnp.arange(4.0)[:, None])
    result, _ = run_mode(
        "compiled",
        state,
        mutation=ObservedMutation(),
        threshold_schedule=[1.0],
    )
    jax.effects_barrier()
    assert result.complete
    assert calls == [0]


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_compiled_dtypes_and_terminal_cache(dtype: str) -> None:
    with jax.enable_x64():
        state = ParticleState(
            jnp.arange(4.0, dtype=dtype)[:, None],
            log_weights=jnp.full((4,), -math.log(4), dtype=dtype),
        )
        sampler, _, metadata = make_sampler(threshold_schedule=[math.inf])
        prepared, context = sampler.initialize_functional(
            state, StepContext(t=0, metadata=metadata)
        )
        key = jax.random.key(42)
        iteration = compile_functional_iteration(
            sampler, prepared, key, metadata=context.metadata
        )
        result, final_key = sampler.run_functional(
            prepared,
            metadata=context.metadata,
            random_state=key,
            num_steps=1,
            iteration=iteration,
        )
        final = result.final_state
        assert final.log_weights.dtype == getattr(jnp, dtype)
        for name in (
            "particles",
            "log_weights",
            "initial_ancestor_indices",
            "log_normalizing_constant",
        ):
            np.testing.assert_array_equal(
                getattr(final, name), getattr(prepared, name)
            )
        for name in prepared.evaluation.components:
            np.testing.assert_array_equal(
                final.evaluation[name], prepared.evaluation[name]
            )
        np.testing.assert_array_equal(
            jax.random.key_data(final_key), jax.random.key_data(key)
        )


@pytest.mark.parametrize("mode", ["functional", "compiled"])
def test_weighted_shells_without_resampling(mode: str) -> None:
    from sequince.smc import SequentialMonteCarlo
    from sequince.steps.constraints import (
        FixedLikelihoodThresholdSchedule,
        LikelihoodConstrainedPrior,
        LikelihoodThresholdSelection,
        LikelihoodThresholdWeighting,
        ShellEvidenceStep,
    )

    target = LikelihoodConstrainedPrior(
        lambda x: jnp.zeros_like(x[..., 0]), lambda x: x[..., 0]
    )
    sampler = SequentialMonteCarlo(
        [
            LikelihoodThresholdSelection(
                target,
                threshold_schedule=FixedLikelihoodThresholdSchedule([1.0]),
            ),
            ShellEvidenceStep(target),
            LikelihoodThresholdWeighting(target),
        ],
        auto_normalize=False,
    )
    weights = jnp.asarray([0.1, 0.2, 0.3, 0.4])
    state = ParticleState(
        jnp.arange(4.0)[:, None], log_weights=jnp.log(weights)
    )
    key = jax.random.key(42)
    iteration = (
        compile_functional_iteration(sampler, state, key)
        if mode == "compiled"
        else None
    )
    result, final_key = sampler.run_functional(
        state, random_state=key, num_steps=10, iteration=iteration
    )
    ns = result.as_nested_sampling_result()
    unnormalized = weights * jnp.exp(state.particles[:, 0])
    assert ns.log_evidence == pytest.approx(
        float(jnp.log(jnp.sum(unnormalized))), abs=1e-6
    )
    samples, posterior_weights = ns.posterior_samples()
    np.testing.assert_array_equal(samples, state.particles)
    np.testing.assert_allclose(
        posterior_weights, unnormalized / unnormalized.sum(), atol=1e-7
    )
    np.testing.assert_array_equal(
        jax.random.key_data(final_key), jax.random.key_data(key)
    )
