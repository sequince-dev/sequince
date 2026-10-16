"""Dynamic feedback, population initialization, and resume regressions."""

from __future__ import annotations

from typing import Any

import h5py
import numpy as np
import pytest

from sequince import ParticleState, make_persistent_bayesian_smc
from sequince.array_api import asarray
from sequince.checkpoint import (
    HDF5CheckpointWriter,
    PickleCheckpointWriter,
    SMCCheckpoint,
)
from sequince.posterior import TemperedPosterior
from sequince.state import PersistentParticleState
from sequince.steps import StepContext
from sequince.steps.persistent import (
    PersistentAdaptiveTemperedWeighting,
    expected_unique_count,
)


def zero(x: Any, context: Any = None) -> Any:
    return x[:, 0] * 0


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax", "cupy"])
@pytest.mark.parametrize(
    "weights, draws, expected",
    [
        ([1.0, 0.0, 0.0], 20, 1.0),
        ([0.0, 0.0], 20, 0.0),
        ([0.25] * 4, 4, 4 * (1 - 0.75**4)),
        ([1.0e-20] * 4, 100, 4.0e-18),
    ],
)
def test_unique_count(
    backend: str,
    backend_factory: Any,
    weights: list[float],
    draws: int,
    expected: float,
) -> None:
    case = backend_factory(backend)
    w = asarray(
        weights, xp=case.xp, like=case.particles, dtype=case.particles.dtype
    )
    with np.errstate(all="raise"):
        actual = expected_unique_count(w, draws)
    assert case.matches_namespace(actual)
    assert actual.dtype == w.dtype
    assert float(actual) == pytest.approx(expected, rel=2.0e-6, abs=1.0e-25)


@pytest.mark.parametrize(
    "size, direction", [(4, "up"), (8, "same"), (32, "down")]
)
def test_feedback_and_next_iteration(size: int, direction: str) -> None:
    target = TemperedPosterior(zero, zero)
    step = PersistentAdaptiveTemperedWeighting(
        target,
        target_ess=8.9,
        target_total_ess=1000,
        dynamic=True,
    )
    archive = PersistentParticleState.from_state(
        ParticleState(np.zeros((4, 1))),
        beta=0.0,
        log_normalizing_constant=0.0,
    )
    for _ in range(size // 4 - 1):
        archive = archive.append(
            ParticleState(np.zeros((4, 1))),
            beta=0.0,
            log_normalizing_constant=0.0,
        )
    _, _, output = step(archive, StepContext(t=0), rng=1)
    fraction = size * (1 - (1 - 1 / size) ** 4) / 4
    expected = {"up": int(8 / fraction), "same": 8, "down": int(8 * fraction)}[
        direction
    ]
    assert float(output["target_ess"]) == 8
    assert output["next_target_ess"] == expected
    assert output["expected_unique_fraction"] == pytest.approx(fraction)
    _, _, following = step(archive, StepContext(t=1), rng=1)
    assert float(following["target_ess"]) == expected
    assert float(following["target_total_ess"]) == 1000
    assert (
        following["reference_unique_fraction"]
        == output["reference_unique_fraction"]
    )


@pytest.mark.parametrize(
    "setting", ["target_ess", "target_total_ess", "beta_tolerance"]
)
@pytest.mark.parametrize("value", [0.0, -1.0, np.inf, np.nan])
def test_invalid_thresholds(setting: str, value: float) -> None:
    with pytest.raises(ValueError):
        PersistentAdaptiveTemperedWeighting(
            TemperedPosterior(zero, zero), **{setting: value}
        )


def test_legacy_checkpoint_policy() -> None:
    static = PersistentAdaptiveTemperedWeighting(TemperedPosterior(zero, zero))
    static.load_state_dict({})
    dynamic = PersistentAdaptiveTemperedWeighting(
        TemperedPosterior(zero, zero),
        dynamic=True,
    )
    with pytest.raises(ValueError, match="Legacy"):
        dynamic.load_state_dict({})
    with pytest.raises(ValueError, match="mismatch"):
        dynamic.load_state_dict(static.state_dict())


def test_checkpoint_preserves_terminal_resampling_choice() -> None:
    target = TemperedPosterior(zero, zero)
    weighted = PersistentAdaptiveTemperedWeighting(target)
    resampled = PersistentAdaptiveTemperedWeighting(
        target, resample_at_end=True
    )
    resampled.load_state_dict(resampled.state_dict())
    with pytest.raises(ValueError, match="mismatch"):
        resampled.load_state_dict(weighted.state_dict())
    with pytest.raises(ValueError, match="mismatch"):
        weighted.load_state_dict(resampled.state_dict())
    legacy = weighted.state_dict()
    del legacy["resample_at_end"]
    weighted.load_state_dict(legacy)
    with pytest.raises(ValueError, match="mismatch"):
        resampled.load_state_dict(legacy)
    with pytest.raises(ValueError, match="mismatch"):
        resampled.load_state_dict({})


@pytest.mark.parametrize("resample_at_end", [False, True])
@pytest.mark.parametrize("checkpoint_format", ["pickle", "hdf5"])
@pytest.mark.parametrize("dynamic", [False, True])
def test_initialization_resume(
    tmp_path: Any,
    dynamic: bool,
    checkpoint_format: str,
    resample_at_end: bool,
) -> None:
    draws = np.arange(16, dtype=float).reshape(-1, 1)
    initial = ParticleState(draws)

    def build(writers: Any = ()) -> Any:
        return make_persistent_bayesian_smc(
            zero,
            zero,
            n_particles=4,
            target_ess=8,
            target_total_ess=24,
            dynamic=dynamic,
            mutation="none",
            logging_frequency=0,
            checkpoint_writers=writers,
            resample_at_end=resample_at_end,
        )[0]

    reference = build().run(initial, rng=45, num_steps=30)
    path = tmp_path / (
        "init.pkl" if checkpoint_format == "pickle" else "init.h5"
    )
    writer = (
        PickleCheckpointWriter
        if checkpoint_format == "pickle"
        else HDF5CheckpointWriter
    )
    partial = build([writer(path, every=1)]).run(
        initial,
        rng=45,
        num_steps=2,
    )
    assert (
        float(partial.context.evaluation_counts.log_likelihood_evaluations)
        == 16
    )
    assert not partial.context.stop_requested
    assert partial.outputs[0][2] is not None
    sampler = build()
    if checkpoint_format == "pickle":
        checkpoint = SMCCheckpoint.load_pickle(path)
    else:
        with h5py.File(path, "r") as file:
            checkpoint = SMCCheckpoint.load_hdf5(file)
    resumed = sampler.run(
        runtime=sampler.resume_from(checkpoint),
        num_steps=30,
    )
    np.testing.assert_array_equal(
        resumed.final_state.particles, reference.final_state.particles
    )
    np.testing.assert_array_equal(
        resumed.final_state.log_weights, reference.final_state.log_weights
    )
    archive = sampler._steps[0].persistent_state
    np.testing.assert_array_equal(archive.betas[:4], 0.0)
    np.testing.assert_array_equal(archive.particles[:16], draws)
    assert resumed.context.stop_requested
    assert float(resumed.outputs[-1][1]["effective_sample_size"]) >= 24
    assert (
        float(resumed.context.evaluation_counts.log_likelihood_evaluations)
        == 16
    )
    assert float(resumed.final_state.log_evidence) == pytest.approx(0.0)


@pytest.mark.parametrize("size", [4, 8, 16, 20])
@pytest.mark.parametrize("dynamic", [False, True])
def test_inferred_initial_population(size: int, dynamic: bool) -> None:
    sampler, _, _ = make_persistent_bayesian_smc(
        zero,
        zero,
        n_particles=4,
        target_ess=100,
        dynamic=dynamic,
        mutation="none",
        logging_frequency=0,
    )
    draws = np.arange(size, dtype=float)[:, None]
    result = sampler.run(ParticleState(draws), rng=1, num_steps=1)
    archive = sampler._steps[0].persistent_state
    assert archive.batch_size == 4
    assert archive.n_batches == size // 4
    np.testing.assert_array_equal(archive.particles, draws)
    np.testing.assert_array_equal(archive.betas, 0.0)
    assert result.final_state.particles.shape == (4, 1)
    assert (
        float(result.context.evaluation_counts.log_likelihood_evaluations)
        == size
    )
    assert "initialization" not in sampler._steps[0].state_dict()


def test_proposal_initialization() -> None:
    # Proposal is N(0,1); posterior proportional to N(1,1). The initial
    # density must remain the proposal even with a larger beta=0 archive.
    def prior(x: Any, context: Any = None) -> Any:
        return -0.5 * (x[:, 0] - 1.0) ** 2

    def proposal(x: Any, context: Any = None) -> Any:
        return -0.5 * x[:, 0] ** 2

    draws = np.random.default_rng(1).normal(size=(4000, 1))
    sampler, _, _ = make_persistent_bayesian_smc(
        prior,
        zero,
        log_proposal=proposal,
        n_particles=1000,
        target_ess=1,
        mutation="none",
        logging_frequency=0,
    )
    result = sampler.run(ParticleState(draws), rng=1, num_steps=10)
    assert result.context.stop_requested
    assert np.average(
        result.final_state.particles[:, 0], weights=result.final_state.weights
    ) == pytest.approx(1.0, abs=0.1)
    assert float(result.final_state.log_evidence) == pytest.approx(
        0.0, abs=0.1
    )


@pytest.mark.parametrize("split", [1, 3])
def test_dynamic_gaussian_moments_evidence_and_resume(
    tmp_path: Any,
    split: int,
) -> None:
    """Exercise checkpoints after initial archival and later mutations."""

    def gaussian_log_prior(x: Any, context: Any = None) -> Any:
        return -0.5 * (x[:, 0] / 2) ** 2 - np.log(2 * np.sqrt(2 * np.pi))

    def gaussian_log_likelihood(x: Any, context: Any = None) -> Any:
        return -0.5 * (x[:, 0] - 1.5) ** 2 - 0.5 * np.log(2 * np.pi)

    n = 128
    initial = ParticleState(
        np.random.default_rng(1234).normal(
            scale=2.0,
            size=(4 * n, 1),
        )
    )

    def build(writers: Any = ()) -> Any:
        return make_persistent_bayesian_smc(
            gaussian_log_prior,
            gaussian_log_likelihood,
            n_particles=n,
            dynamic=True,
            target_total_ess=8 * n,
            mutation_n_steps=3,
            mutation_kwargs={"step_fn": "tpcn"},
            logging_frequency=0,
            checkpoint_writers=writers,
            resample_at_end=False,
        )[0]

    reference = build().run(initial, rng=5678, num_steps=100)
    path = tmp_path / "gaussian.pkl"
    build([PickleCheckpointWriter(path, every=1)]).run(
        initial,
        rng=5678,
        num_steps=split,
    )
    sampler = build()
    resumed = sampler.run(
        runtime=sampler.resume_from(SMCCheckpoint.load_pickle(path)),
        num_steps=100,
    )
    assert resumed.context.stop_requested
    final = resumed.final_state
    np.testing.assert_array_equal(
        final.particles, reference.final_state.particles
    )
    np.testing.assert_array_equal(
        final.log_weights, reference.final_state.log_weights
    )
    x, w = final.particles[:, 0], final.weights
    mean = w @ x
    assert mean == pytest.approx(1.2, abs=0.12)
    assert w @ (x - mean) ** 2 == pytest.approx(0.8, abs=0.12)
    logz = -0.5 * (np.log(2 * np.pi * 5.0) + 1.5**2 / 5.0)
    assert float(final.log_evidence) == pytest.approx(logz, abs=0.12)
    assert float(final.effective_sample_size()) >= 8 * n
    for a, b in zip(resumed.outputs, reference.outputs):
        assert float(a[1]["target_ess"]) == float(b[1]["target_ess"])
        assert a[1].get("next_target_ess") == b[1].get("next_target_ess")


@pytest.mark.parametrize("size", [1, 3, 5, 19])
def test_invalid_initial_population_size(size: int) -> None:
    sampler, _, _ = make_persistent_bayesian_smc(
        zero,
        zero,
        n_particles=4,
        mutation="none",
    )
    with pytest.raises(ValueError, match="positive multiple"):
        sampler.run(ParticleState(np.zeros((size, 1))), rng=1, num_steps=1)


@pytest.mark.parametrize(
    "metadata,weights", [({"beta": 0.5}, None), ({}, [0.0] * 7 + [-1.0])]
)
def test_invalid_large_initial_population(metadata: Any, weights: Any) -> None:
    sampler, _, _ = make_persistent_bayesian_smc(
        zero,
        zero,
        n_particles=4,
        mutation="none",
    )
    with pytest.raises(ValueError, match="beta=0|equal weights"):
        sampler.run(
            ParticleState(
                np.zeros((8, 1)),
                log_weights=None if weights is None else np.asarray(weights),
                metadata=metadata,
            ),
            rng=1,
            num_steps=1,
        )


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax", "cupy"])
def test_backend_initialization_and_feedback(
    backend: str, backend_factory: Any
) -> None:
    case = backend_factory(backend)
    initial = ParticleState(case.particles)
    sampler, _, _ = make_persistent_bayesian_smc(
        zero,
        zero,
        n_particles=2,
        target_ess=2,
        dynamic=True,
        mutation="none",
        logging_frequency=0,
    )
    result = sampler.run(initial, rng=case.rng(11), num_steps=10)
    assert result.context.stop_requested
    assert case.matches_namespace(result.final_state.particles)
    np.testing.assert_allclose(
        float(result.final_state.log_evidence), 0.0, atol=1.0e-6
    )


def test_static_default_is_identical_to_explicit_false() -> None:
    initial = ParticleState(np.arange(4, dtype=float)[:, None])
    results = []
    for kwargs in ({}, {"dynamic": False}):
        sampler, _, _ = make_persistent_bayesian_smc(
            zero,
            zero,
            n_particles=4,
            mutation="none",
            logging_frequency=0,
            **kwargs,
        )
        results.append(sampler.run(initial, rng=42, num_steps=10))
    np.testing.assert_array_equal(
        results[0].final_state.particles, results[1].final_state.particles
    )
    assert "persistent_initializing" not in results[0].context.metadata
    assert len(results[0]) == len(results[1])


def test_small_dynamic_threshold_remains_positive() -> None:
    step = PersistentAdaptiveTemperedWeighting(
        TemperedPosterior(zero, zero),
        target_ess=1,
        dynamic=True,
    )
    archive = PersistentParticleState.from_state(
        ParticleState(np.zeros((4, 1))),
        beta=0.0,
        log_normalizing_constant=0.0,
    )
    _, _, output = step(archive, StepContext(t=0), rng=1)
    assert output["next_target_ess"] == 1
    assert output["reference_unique_fraction"] == 0.25
