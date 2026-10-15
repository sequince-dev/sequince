from __future__ import annotations

from dataclasses import replace
from typing import Literal

import numpy as np
import pytest
from orng import RandomGenerator

from sequince import ParticleState
from sequince.checkpoint import CheckpointWriter, SMCCheckpoint
from sequince.steps import (
    AdaptiveTemperedWeighting,
    FixedTemperedWeighting,
    StepContext,
)
from sequince.utils.optimize import bisect_monotone_decreasing

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
compile_functional_iteration = pytest.importorskip(
    "sequince.jax"
).compile_functional_iteration


@pytest.mark.parametrize("n_particles", [16, 17])
def test_compiled_differential_evolution_replays(n_particles: int) -> None:
    from sequince import make_standard_bayesian_smc
    from sequince._typing import Array

    def prior(particles: Array, context: StepContext) -> Array:
        return -0.5 * jnp.sum(particles**2, axis=-1) - jnp.log(2 * jnp.pi)

    def likelihood(particles: Array, context: StepContext) -> Array:
        return -0.5 * jnp.sum(particles**2, axis=-1)

    initial = ParticleState(
        jax.random.normal(jax.random.key(10), (n_particles, 2))
    )
    key = jax.random.key(20)
    smc, _, metadata = make_standard_bayesian_smc(
        log_prior=prior,
        log_likelihood=likelihood,
        adaptive=False,
        num_steps=2,
        always_resample=True,
        mutation="differential-evolution",
        mutation_n_steps=2,
        particle_transform="affine",
        logging_frequency=0,
    )
    iteration = compile_functional_iteration(
        smc, initial, key, metadata=metadata
    )
    result, final_key = smc.run_functional(
        initial, random_state=key, metadata=metadata, iteration=iteration
    )
    replay, replay_key = smc.run_functional(
        initial, random_state=key, metadata=metadata, iteration=iteration
    )
    np.testing.assert_array_equal(
        result.final_state.particles, replay.final_state.particles
    )
    np.testing.assert_array_equal(
        jax.random.key_data(final_key), jax.random.key_data(replay_key)
    )
    assert not np.array_equal(
        jax.random.key_data(key), jax.random.key_data(final_key)
    )
    assert result.final_state.particles.shape == (n_particles, 2)
    assert result.final_state.particles.dtype == initial.particles.dtype
    assert float(result.context.metadata["beta"]) == pytest.approx(1.0)
    acceptance = float(
        result.context.metadata["metropolis_hastings_acceptance"]
    )
    assert 0.0 <= acceptance <= 1.0


@pytest.mark.parametrize("invalid", [-np.inf, np.inf, np.nan])
@pytest.mark.parametrize("resample", [False, True])
def test_jitted_runner_rejects_invalid_second_iteration(
    invalid: float, resample: bool
) -> None:
    from sequince import DegenerateWeightsError
    from sequince.smc import SequentialMonteCarlo
    from sequince.steps import SystematicResampling

    def increment(particles, context):
        return jnp.full(
            particles.shape[0], jnp.where(context.t == 0, 0.0, invalid)
        )

    callbacks = []
    steps = [FixedTemperedWeighting(increment, beta_increment=0.5)]
    if resample:
        steps.append(SystematicResampling(always_resample=True))
    smc = SequentialMonteCarlo(
        steps, callbacks=[lambda *args: callbacks.append(args)]
    )
    state = ParticleState(jnp.zeros((4, 1)))
    key = jax.random.key(0)
    iteration = compile_functional_iteration(smc, state, key)
    with pytest.raises(DegenerateWeightsError):
        smc.run_functional(
            state, num_steps=2, random_state=key, iteration=iteration
        )
    assert len(callbacks) == 1


def test_monotone_bisection_uses_jittable_control_flow() -> None:
    def solve():
        return bisect_monotone_decreasing(
            lambda value: 1.0 - value,
            jnp.asarray(0.0),
            jnp.asarray(1.0),
            jnp.asarray(0.3),
            tolerance=jnp.asarray(1e-6),
            max_iterations=40,
        )

    low, high = jax.jit(solve)()

    assert float(low) <= 0.7 <= float(high)
    assert float(high - low) <= 1e-6


class CheckpointRecorder(CheckpointWriter):
    def __init__(self) -> None:
        super().__init__(save_rng=True)
        self.checkpoints: list[SMCCheckpoint] = []

    def _write(self, checkpoint: SMCCheckpoint) -> None:
        self.checkpoints.append(checkpoint)


def _fixed_blackjax_sampler(
    *, checkpoint_writers=None, particle_transform=None
):
    from sequince import make_standard_bayesian_smc

    def log_prior(x, context):
        return -0.5 * jnp.sum(x**2, axis=-1)

    def log_likelihood(x, context):
        return -0.5 * jnp.sum((x - 1.0) ** 2, axis=-1)

    return make_standard_bayesian_smc(
        log_prior,
        log_likelihood,
        adaptive=False,
        num_steps=2,
        mutation="blackjax",
        mutation_n_steps=1,
        mutation_kwargs={
            "algorithm": "random-walk",
            "algorithm_kwargs": {"sigma": 0.1},
        },
        particle_transform=particle_transform,
        logging_frequency=0,
        checkpoint_writers=checkpoint_writers,
    )


def _context() -> StepContext:
    return StepContext(
        t=jnp.asarray(0),
        metadata={
            "max_beta": jnp.asarray(1.0),
            "beta": jnp.asarray(0.0),
            "delta_beta": jnp.asarray(0.0),
            "ess": jnp.asarray(4.0),
            "resampled": jnp.asarray(False),
        },
    )


def _adaptive_context(*, beta=0.0) -> StepContext:
    return StepContext(
        t=jnp.asarray(0),
        metadata={
            "beta": jnp.asarray(beta),
            "delta_beta": jnp.asarray(0.0),
            "target_ess_ratio": jnp.asarray(0.8),
            "ess_ratio": jnp.asarray(1.0),
            "cess_ratio": jnp.asarray(1.0),
            "ess": jnp.asarray(4.0),
            "resampled": jnp.asarray(False),
        },
    )


def _adaptive_sampler(*, criterion: Literal["ess", "cess"] = "ess"):
    from sequince import make_standard_bayesian_smc

    def log_prior(x, context):
        return -0.5 * jnp.sum(x**2, axis=-1)

    def log_likelihood(x, context):
        return jnp.linspace(-2.0, 2.0, x.shape[0])

    return make_standard_bayesian_smc(
        log_prior,
        log_likelihood,
        target_ess_ratio=0.8,
        max_beta=0.6,
        min_delta=1e-6,
        weighting_criterion=criterion,
        always_resample=True,
        mutation="none",
        particle_transform=None,
        logging_frequency=0,
    )


def test_functional_metadata_is_initialized_by_the_step_pipeline() -> None:
    from sequince import make_standard_bayesian_smc

    smc, _, metadata = make_standard_bayesian_smc(
        lambda x, context: -0.5 * jnp.sum(x**2, axis=-1),
        lambda x, context: jnp.linspace(-1.0, 1.0, x.shape[0]),
        target_ess_ratio=lambda beta: 0.8,
        always_resample=True,
        mutation="none",
        particle_transform=None,
        logging_frequency=0,
    )
    custom = jnp.asarray(7.0)
    state = ParticleState(
        jnp.zeros((4, 1)),
        metadata={"beta": jnp.asarray(0.25)},
    )

    _, initialized_context = smc.initialize_functional(
        state,
        StepContext(
            t=0,
            metadata={**metadata, "custom": custom},
        ),
    )
    initialized = initialized_context.metadata

    assert initialized["custom"] is custom
    assert initialized["max_beta"] == pytest.approx(1.0)
    assert initialized["beta"] == pytest.approx(0.25)
    assert initialized["delta_beta"] == pytest.approx(0.0)
    assert initialized["target_ess_ratio"] == pytest.approx(1.0)
    assert initialized["ess_ratio"] == pytest.approx(1.0)
    assert initialized["cess_ratio"] == pytest.approx(1.0)
    assert initialized["ess"] == pytest.approx(4.0)
    assert not initialized["resampled"]


def test_fixed_functional_metadata_does_not_require_adaptive_target() -> None:
    smc, _, _ = _fixed_blackjax_sampler()

    _, initialized_context = smc.initialize_functional(
        ParticleState(jnp.zeros((4, 2))),
        StepContext(t=0),
    )
    initialized = initialized_context.metadata

    assert "target_ess_ratio" not in initialized
    assert "ess_ratio" not in initialized
    assert "cess_ratio" not in initialized
    assert initialized["beta"] == pytest.approx(0.0)
    assert initialized["ess"] == pytest.approx(4.0)


@pytest.mark.parametrize("adaptive", [False, True])
def test_guarded_weighting_reuses_density_cache(adaptive: bool) -> None:
    from sequince import TemperedPosterior
    from sequince.steps import apply_functional_if

    calls = []

    def likelihood(particles):
        calls.append(True)
        return -jnp.sum(particles**2, axis=-1)

    target = TemperedPosterior(lambda x: jnp.zeros(x.shape[0]), likelihood)
    step = (
        AdaptiveTemperedWeighting(target, target_ess_ratio=0.8)
        if adaptive
        else FixedTemperedWeighting(target, beta_increment=0.5)
    )
    context = _adaptive_context() if adaptive else _context()
    particles = jnp.arange(4.0).reshape(4, 1)
    state = ParticleState(
        particles, evaluation=target.evaluate(particles, context)
    )
    expected, _, _ = step.apply(state, context, RandomGenerator(backend="jax"))

    def guarded(state, context, key):
        return apply_functional_if(
            jnp.asarray(True), step.apply_functional, state, context, key
        )

    updated, _, _, _ = jax.jit(guarded)(state, context, jax.random.key(42))
    assert calls == [True]
    assert set(updated.evaluation.components) == {
        "log_prior",
        "log_likelihood",
    }
    for name, value in state.evaluation.components.items():
        np.testing.assert_array_equal(updated.evaluation[name], value)
    np.testing.assert_allclose(updated.log_weights, expected.log_weights)


def test_run_functional_initializes_metadata_automatically() -> None:
    smc, _, metadata = _fixed_blackjax_sampler()
    custom = jnp.asarray(7.0)

    result, _ = smc.run_functional(
        ParticleState(jnp.zeros((4, 2))),
        num_steps=2,
        random_state=jax.random.key(42),
        metadata={**metadata, "custom": custom},
        iteration=jax.jit(smc.functional_iteration),
    )

    assert result.context.metadata["custom"] == pytest.approx(7.0)
    assert result.context.metadata["beta"] == pytest.approx(1.0)


def test_compile_functional_iteration_lowers_one_stable_signature() -> None:
    smc, _, metadata = _fixed_blackjax_sampler()
    state = ParticleState(jnp.zeros((4, 2)))
    key = jax.random.key(42)

    iteration = compile_functional_iteration(
        smc,
        state,
        key,
        metadata=metadata,
    )
    assert isinstance(iteration, jax.stages.Compiled)
    result, final_key = smc.run_functional(
        state,
        num_steps=2,
        random_state=key,
        metadata=metadata,
        iteration=iteration,
    )

    assert len(result) == 2
    assert result.final_state.evaluation is not None
    np.testing.assert_array_equal(state.particles, jnp.zeros((4, 2)))
    np.testing.assert_array_equal(
        jax.random.key_data(key),
        jax.random.key_data(jax.random.key(42)),
    )
    assert not np.array_equal(
        jax.random.key_data(key),
        jax.random.key_data(final_key),
    )


def test_compile_functional_iteration_rejects_unstable_carry() -> None:
    from sequince.smc import SequentialMonteCarlo
    from sequince.steps.base import SMCStep

    class AddsStateMetadata(SMCStep):
        def apply_functional(self, state, context, random_state):
            state = state.with_updates(metadata={"added": jnp.asarray(1.0)})
            return state, context, random_state, {}

    smc = SequentialMonteCarlo([AddsStateMetadata()])

    with pytest.raises(ValueError, match="changes its carry signature"):
        compile_functional_iteration(
            smc,
            ParticleState(jnp.zeros((4, 2))),
            jax.random.key(42),
        )


def test_compiled_functional_iteration_supports_affine_transform() -> None:
    from sequince.transforms import AffineTransform

    smc, _, metadata = _fixed_blackjax_sampler(particle_transform="affine")
    state = ParticleState(jnp.arange(8.0).reshape(4, 2))
    key = jax.random.key(42)

    iteration = compile_functional_iteration(
        smc,
        state,
        key,
        metadata=metadata,
    )
    result, _ = smc.run_functional(
        state,
        num_steps=2,
        random_state=key,
        metadata=metadata,
        iteration=iteration,
    )

    transform = result.context.metadata["particle_transform"]
    assert isinstance(transform, AffineTransform)
    assert np.asarray(transform._mean).shape == (2,)
    assert np.asarray(transform._scale).shape == (2,)


def test_functional_iteration_can_be_jitted() -> None:
    smc, _, _ = _fixed_blackjax_sampler()
    state = ParticleState(jnp.zeros((4, 2)))
    key = jax.random.key(42)

    state, context = smc.initialize_functional(state, _context())
    updated, context, next_key, outputs = jax.jit(smc.functional_iteration)(
        state, context, key
    )

    assert updated.particles.shape == (4, 2)
    assert updated.initial_ancestor_indices.shape == (4,)
    assert context.metadata["beta"] == pytest.approx(0.5)
    assert len(outputs) == 3
    assert outputs[1]["ancestors"].shape == (4,)
    assert outputs[1]["resampling_scheme"] == "systematic"
    assert "ancestors" not in updated.metadata
    assert outputs[-1]["algorithm"] == "random-walk"
    assert context.evaluation_counts.log_target_evaluations == 16
    assert context.evaluation_counts.log_likelihood_evaluations == 16
    assert context.evaluation_counts.log_proposal_evaluations == 0
    assert context.evaluation_counts.log_increment_evaluations == 4
    assert not np.array_equal(
        jax.random.key_data(key),
        jax.random.key_data(next_key),
    )


def test_functional_iteration_with_affine_transform_can_be_jitted() -> None:
    from sequince.transforms import AffineTransform

    smc, _, metadata = _fixed_blackjax_sampler(
        particle_transform=AffineTransform()
    )
    state = ParticleState(jnp.zeros((4, 2)))
    state, context = smc.initialize_functional(
        state,
        StepContext(
            t=jnp.asarray(0),
            metadata=metadata,
        ),
    )
    iteration = jax.jit(smc.functional_iteration)

    state, context, key, _ = iteration(state, context, jax.random.key(42))
    state, context, _, _ = iteration(
        state,
        replace(context, t=context.t + 1),
        key,
    )
    jax.block_until_ready(state)

    assert isinstance(context.metadata["particle_transform"], AffineTransform)


def test_functional_iteration_matches_host_iteration() -> None:
    smc, _, _ = _fixed_blackjax_sampler()
    state = ParticleState(jnp.zeros((4, 2)))
    context = _context()
    key = jax.random.key(42)
    rng = RandomGenerator(backend="jax", generator=key)

    expected = smc.run(
        state,
        num_steps=1,
        rng=rng,
        metadata=context.metadata,
    )
    state, context = smc.initialize_functional(state, context)
    actual_state, actual_context, actual_key, _ = jax.jit(
        smc.functional_iteration
    )(state, context, key)

    np.testing.assert_allclose(
        actual_state.particles,
        expected.final_state.particles,
    )
    np.testing.assert_allclose(
        actual_state.log_weights,
        expected.final_state.log_weights,
    )
    np.testing.assert_allclose(
        actual_state.log_evidence,
        expected.final_state.log_evidence,
    )
    np.testing.assert_array_equal(
        actual_state.initial_ancestor_indices,
        expected.final_state.initial_ancestor_indices,
    )
    np.testing.assert_allclose(
        actual_context.metadata["beta"],
        expected.context.metadata["beta"],
    )
    np.testing.assert_array_equal(
        jax.random.key_data(actual_key),
        jax.random.key_data(rng.state()),
    )


def test_functional_iteration_replays_from_same_random_state() -> None:
    smc, _, _ = _fixed_blackjax_sampler()
    state = ParticleState(jnp.zeros((4, 2)))
    state, context = smc.initialize_functional(state, _context())
    key = jax.random.key(42)
    iteration = jax.jit(smc.functional_iteration)

    first = iteration(state, context, key)
    second = iteration(state, context, key)

    np.testing.assert_array_equal(first[0].particles, second[0].particles)
    np.testing.assert_array_equal(
        jax.random.key_data(first[2]),
        jax.random.key_data(second[2]),
    )


def test_functional_iteration_runs_under_scan() -> None:
    smc, posterior, _ = _fixed_blackjax_sampler()
    context = _context()
    particles = jnp.zeros((4, 2))
    evaluation = posterior.prepare_evaluation(particles, context)
    state = ParticleState(particles, evaluation=evaluation)

    def body(carry, unused):
        del unused
        state, context, key = carry
        state, context, key, outputs = smc.functional_iteration(
            state,
            context,
            key,
        )
        context = replace(context, t=context.t + 1)
        return (state, context, key), outputs

    (state, context, _), _ = jax.jit(
        lambda carry: jax.lax.scan(body, carry, None, length=2)
    )((state, context, jax.random.key(42)))

    assert state.particles.shape == (4, 2)
    assert context.metadata["beta"] == pytest.approx(1.0)


def test_run_functional_returns_advanced_random_state() -> None:
    smc, _, _ = _fixed_blackjax_sampler()
    state = ParticleState(jnp.zeros((4, 2)))
    key = jax.random.key(42)

    result, final_key = smc.run_functional(
        state,
        num_steps=2,
        random_state=key,
        metadata=_context().metadata,
        iteration=jax.jit(smc.functional_iteration),
    )

    assert len(result) == 2
    assert result.context.metadata["beta"] == pytest.approx(1.0)
    assert not np.array_equal(
        jax.random.key_data(key),
        jax.random.key_data(final_key),
    )


def test_run_functional_replays_from_same_random_state() -> None:
    smc, _, _ = _fixed_blackjax_sampler()
    state = ParticleState(jnp.zeros((4, 2)))
    key = jax.random.key(42)
    iteration = jax.jit(smc.functional_iteration)

    first_result, first_key = smc.run_functional(
        state,
        num_steps=2,
        random_state=key,
        metadata=_context().metadata,
        iteration=iteration,
    )
    second_result, second_key = smc.run_functional(
        state,
        num_steps=2,
        random_state=key,
        metadata=_context().metadata,
        iteration=iteration,
    )

    np.testing.assert_array_equal(
        first_result.final_state.particles,
        second_result.final_state.particles,
    )
    np.testing.assert_array_equal(
        jax.random.key_data(first_key),
        jax.random.key_data(second_key),
    )


def test_run_functional_can_continue_from_returned_random_state() -> None:
    state = ParticleState(jnp.zeros((4, 2)))
    key = jax.random.key(42)

    full_smc, _, _ = _fixed_blackjax_sampler()
    full_result, full_key = full_smc.run_functional(
        state,
        num_steps=2,
        random_state=key,
        metadata=_context().metadata,
        iteration=jax.jit(full_smc.functional_iteration),
    )

    split_smc, _, _ = _fixed_blackjax_sampler()
    first_result, next_key = split_smc.run_functional(
        state,
        num_steps=1,
        random_state=key,
        metadata=_context().metadata,
        iteration=jax.jit(split_smc.functional_iteration),
    )
    resumed_result, resumed_key = split_smc.run_functional(
        first_result.final_state,
        num_steps=1,
        random_state=next_key,
        metadata=first_result.context.metadata,
        iteration=jax.jit(split_smc.functional_iteration),
    )

    np.testing.assert_allclose(
        resumed_result.final_state.particles,
        full_result.final_state.particles,
    )
    np.testing.assert_array_equal(
        jax.random.key_data(resumed_key),
        jax.random.key_data(full_key),
    )


def test_run_functional_checkpoint_resume_matches_full_run() -> None:
    state = ParticleState(jnp.zeros((4, 2)))
    key = jax.random.key(42)

    full_smc, _, _ = _fixed_blackjax_sampler()
    full_result, full_key = full_smc.run_functional(
        state,
        num_steps=2,
        random_state=key,
        metadata=_context().metadata,
        iteration=jax.jit(full_smc.functional_iteration),
    )

    recorder = CheckpointRecorder()
    first_smc, _, _ = _fixed_blackjax_sampler(checkpoint_writers=[recorder])
    first_smc.run_functional(
        state,
        num_steps=1,
        random_state=key,
        metadata=_context().metadata,
        iteration=jax.jit(first_smc.functional_iteration),
    )

    resumed_smc, _, _ = _fixed_blackjax_sampler()
    runtime = resumed_smc.resume_functional_from(
        recorder.checkpoints[-1].to_dict()
    )
    resumed_result, resumed_key = resumed_smc.run_functional(
        runtime=runtime,
        num_steps=1,
        iteration=jax.jit(resumed_smc.functional_iteration),
    )

    np.testing.assert_allclose(
        resumed_result.final_state.particles,
        full_result.final_state.particles,
    )
    np.testing.assert_array_equal(
        jax.random.key_data(resumed_key),
        jax.random.key_data(full_key),
    )


def test_functional_resume_preserves_once_transform() -> None:
    from sequince.smc import SequentialMonteCarlo
    from sequince.steps.mutation import BlackJAXMutation
    from sequince.transforms import AffineTransform

    def make_runner(*writers: CheckpointWriter) -> SequentialMonteCarlo:
        return SequentialMonteCarlo(
            [
                BlackJAXMutation(
                    lambda p: -0.5 * jnp.sum(p**2),
                    algorithm="random-walk",
                    algorithm_kwargs={"sigma": 0.5},
                    transform=AffineTransform(),
                    transform_refit="once",
                )
            ],
            checkpoint_writers=writers,
        )

    state = ParticleState(jnp.arange(8.0).reshape(4, 2))
    key = jax.random.key(42)
    full_runner = make_runner()
    full, full_key = full_runner.run_functional(
        state,
        num_steps=4,
        random_state=key,
        iteration=jax.jit(full_runner.functional_iteration),
    )
    recorder = CheckpointRecorder()
    first_runner = make_runner(recorder)
    first_runner.run_functional(
        state,
        num_steps=2,
        random_state=key,
        iteration=jax.jit(first_runner.functional_iteration),
    )
    resumed_runner = make_runner()
    resumed, resumed_key = resumed_runner.run_functional(
        runtime=resumed_runner.resume_functional_from(
            recorder.checkpoints[-1].to_dict()
        ),
        num_steps=2,
        iteration=jax.jit(resumed_runner.functional_iteration),
    )

    np.testing.assert_array_equal(
        full.final_state.particles, resumed.final_state.particles
    )
    np.testing.assert_array_equal(
        full.context.metadata["particle_transform"]._scale,
        resumed.context.metadata["particle_transform"]._scale,
    )
    np.testing.assert_array_equal(
        jax.random.key_data(full_key), jax.random.key_data(resumed_key)
    )


def test_functional_resampling_requires_fixed_control_flow() -> None:
    from sequince.steps import SystematicResampling

    step = SystematicResampling(always_resample=False, ess_threshold=0.5)

    with pytest.raises(NotImplementedError, match="always_resample=True"):
        step.apply_functional(
            ParticleState(jnp.zeros((4, 1))),
            StepContext(t=0),
            jax.random.key(42),
        )


def test_fixed_tempering_uses_same_eager_and_functional_transition() -> None:
    step = FixedTemperedWeighting(
        lambda particles: jnp.linspace(-1.0, 1.0, particles.shape[0]),
        beta_increment=0.25,
    )
    state = ParticleState(jnp.zeros((4, 1)))
    context = _context()
    key = jax.random.key(42)

    expected_state, expected_context, expected_output = step(
        state,
        context,
        RandomGenerator(backend="jax", generator=key),
    )
    updated, next_context, _, output = jax.jit(step.apply_functional)(
        state,
        context,
        key,
    )

    np.testing.assert_allclose(updated.log_weights, expected_state.log_weights)
    np.testing.assert_allclose(
        updated.log_evidence,
        expected_state.log_evidence,
    )
    np.testing.assert_allclose(
        next_context.metadata["beta"],
        expected_context.metadata["beta"],
    )
    assert output.keys() == expected_output.keys()
    np.testing.assert_allclose(
        output["log_weight_increment"],
        expected_output["log_weight_increment"],
    )


@pytest.mark.parametrize("criterion", ["ess", "cess"])
def test_adaptive_tempering_functional_step_can_be_jitted(
    criterion: Literal["ess", "cess"],
) -> None:
    increments = jnp.linspace(-2.0, 2.0, 4)
    step = AdaptiveTemperedWeighting(
        lambda particles: increments,
        target_ess_ratio=0.8,
        max_beta=0.6,
        min_delta=1e-6,
        criterion=criterion,
    )
    state = ParticleState(jnp.zeros((4, 1)))
    context = _adaptive_context()
    key = jax.random.key(42)
    expected_state, expected_context, _ = step(
        state,
        context,
        RandomGenerator(backend="jax", generator=key),
    )

    updated, next_context, next_key, output = jax.jit(step.apply_functional)(
        state, context, key
    )

    assert output["delta_beta"] > 0.0
    assert output["beta"] <= 0.6
    assert output["log_weight_increment"].shape == (4,)
    np.testing.assert_allclose(
        next_context.metadata["beta"],
        output["beta"],
    )
    np.testing.assert_allclose(
        updated.log_weights,
        expected_state.log_weights,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        output["delta_beta"],
        expected_context.metadata["delta_beta"],
        atol=1e-6,
    )
    np.testing.assert_array_equal(
        jax.random.key_data(next_key),
        jax.random.key_data(key),
    )
    assert updated.log_weights.shape == (4,)


def test_adaptive_tempering_terminal_output_has_fixed_structure() -> None:
    step = AdaptiveTemperedWeighting(
        lambda particles: jnp.linspace(-2.0, 2.0, particles.shape[0]),
        max_beta=0.6,
    )
    state = ParticleState(jnp.zeros((4, 1)))
    key = jax.random.key(42)
    functional_step = jax.jit(step.apply_functional)

    _, _, _, active_output = functional_step(
        state,
        _adaptive_context(),
        key,
    )
    updated, context, _, terminal_output = functional_step(
        state,
        _adaptive_context(beta=0.6),
        key,
    )

    assert active_output.keys() == terminal_output.keys()
    assert terminal_output["delta_beta"] == pytest.approx(0.0)
    np.testing.assert_array_equal(
        terminal_output["log_weight_increment"],
        jnp.zeros(4),
    )
    np.testing.assert_array_equal(updated.log_weights, state.log_weights)
    assert context.stop_requested
    assert context.stop_immediately


def test_adaptive_callable_target_and_bounds_are_jittable() -> None:
    def target(beta, context):
        return 0.9 - 0.1 * beta + 0.0 * context.t

    step = AdaptiveTemperedWeighting(
        lambda particles: jnp.zeros(particles.shape[0]),
        target_ess_ratio=target,
        max_beta=0.6,
        max_delta=0.2,
    )

    _, context, _, output = jax.jit(step.apply_functional)(
        ParticleState(jnp.zeros((4, 1))),
        _adaptive_context(),
        jax.random.key(42),
    )

    assert output["target_ess_ratio"] == pytest.approx(0.9)
    assert output["delta_beta"] == pytest.approx(0.2)
    assert context.metadata["beta"] == pytest.approx(0.2)
    np.testing.assert_array_equal(
        output["log_weight_increment"],
        jnp.zeros(4),
    )


@pytest.mark.parametrize("criterion", ["ess", "cess"])
def test_adaptive_functional_run_reaches_max_beta(
    criterion: Literal["ess", "cess"],
) -> None:
    smc, _, metadata = _adaptive_sampler(criterion=criterion)
    state = ParticleState(jnp.zeros((4, 1)))

    result, _ = smc.run_functional(
        state,
        num_steps=20,
        random_state=jax.random.key(42),
        metadata={**metadata, **_adaptive_context().metadata},
        iteration=jax.jit(smc.functional_iteration),
    )

    assert result.context.metadata["beta"] == pytest.approx(0.6)
    assert result.context.stop_requested
    assert len(result) < 20
