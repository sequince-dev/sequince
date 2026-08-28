"""Integration tests for persistent sampling through the SMC runner."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from sequince import (
    ParticleState,
    PersistentParticleState,
    make_persistent_bayesian_smc,
)
from sequince.checkpoint import PickleCheckpointWriter, SMCCheckpoint
from sequince.smc import SequentialMonteCarlo
from sequince.steps import (
    PersistParticles,
    StepContext,
)


def test_make_persistent_bayesian_smc_reaches_gaussian_posterior(
    gaussian_log_prior: Callable[..., np.ndarray],
    gaussian_log_likelihood: Callable[..., np.ndarray],
) -> None:
    n_particles = 32
    target_ess = 2 * n_particles
    smc, target, metadata = make_persistent_bayesian_smc(
        gaussian_log_prior,
        gaussian_log_likelihood,
        n_particles=n_particles,
        target_ess=target_ess,
        mutation_n_steps=3,
        mutation_kwargs={"step_fn": "tpcn"},
    )
    initial_particles = np.random.default_rng(1234).normal(
        loc=0.0,
        scale=2.0,
        size=(n_particles, 1),
    )
    initial_context = StepContext(t=0, metadata={"beta": 0.0})
    initial_state = ParticleState(
        initial_particles,
        evaluation=target.prepare_evaluation(
            initial_particles,
            initial_context,
        ),
    )

    result = smc.run(initial_state, metadata=metadata, rng=5678)

    betas = [float(outputs[1]["beta"]) for outputs in result.outputs]
    delta_betas = [
        float(outputs[1]["delta_beta"]) for outputs in result.outputs
    ]
    assert delta_betas[:2] == pytest.approx([0.0, 0.0])
    assert betas[-1] == pytest.approx(1.0)
    assert result.context.stop_requested
    assert result.context.stop_immediately
    assert isinstance(result.final_state, PersistentParticleState)
    np.testing.assert_allclose(result.final_state.weights.sum(), 1.0)

    persist = smc._steps[0]  # noqa: SLF001
    assert isinstance(persist, PersistParticles)
    assert persist.persistent_state is not None
    assert persist.persistent_state.n_batches == len(result)
    assert persist.persistent_state.n_particles == len(result) * n_particles

    final_particles = np.asarray(result.final_state.particles)[:, 0]
    final_weights = np.asarray(result.final_state.weights)
    final_mean = np.average(final_particles, weights=final_weights)
    final_variance = np.average(
        (final_particles - final_mean) ** 2,
        weights=final_weights,
    )
    assert final_mean == pytest.approx(1.2, abs=0.35)
    assert final_variance == pytest.approx(0.8, abs=0.35)
    analytic_log_evidence = -0.5 * (np.log(2.0 * np.pi * 5.0) + 1.5**2 / 5.0)
    assert float(result.final_state.log_evidence) == pytest.approx(
        analytic_log_evidence,
        abs=0.25,
    )


def test_runner_accumulates_posterior_batches_until_target_total_ess(
    gaussian_log_prior: Callable[..., np.ndarray],
    gaussian_log_likelihood: Callable[..., np.ndarray],
) -> None:
    n_particles = 32
    target_ess = 2 * n_particles
    target_total_ess = 5 * n_particles
    smc, target, metadata = make_persistent_bayesian_smc(
        gaussian_log_prior,
        gaussian_log_likelihood,
        n_particles=n_particles,
        target_ess=target_ess,
        target_total_ess=target_total_ess,
        mutation_n_steps=2,
        mutation_kwargs={"step_fn": "tpcn"},
    )
    persist = smc._steps[0]  # noqa: SLF001
    assert isinstance(persist, PersistParticles)
    initial_particles = np.random.default_rng(1234).normal(
        loc=0.0,
        scale=2.0,
        size=(n_particles, 1),
    )
    initial_context = StepContext(t=0, metadata={"beta": 0.0})
    initial_state = ParticleState(
        initial_particles,
        evaluation=target.prepare_evaluation(
            initial_particles,
            initial_context,
        ),
    )

    result = smc.run(
        initial_state,
        rng=5678,
        metadata=metadata,
    )

    delta_betas = [
        float(iteration_output[1]["delta_beta"])
        for iteration_output in result.outputs
    ]
    betas = [
        float(iteration_output[1]["beta"])
        for iteration_output in result.outputs
    ]
    persistent_ess = [
        float(iteration_output[1]["effective_sample_size"])
        for iteration_output in result.outputs
    ]

    assert delta_betas[:2] == pytest.approx([0.0, 0.0])
    assert betas[:2] == pytest.approx([0.0, 0.0])
    assert persistent_ess[:2] == pytest.approx([n_particles, target_ess])
    assert any(delta > 0.0 for delta in delta_betas[2:])
    assert betas[-1] == pytest.approx(1.0)
    assert delta_betas[-1] == pytest.approx(0.0)
    assert betas.count(1.0) >= 2
    assert persistent_ess[-1] >= target_total_ess
    assert result.context.stop_requested
    assert result.context.stop_immediately
    assert len(result.outputs[-1]) == 2
    assert isinstance(result.final_state, PersistentParticleState)
    np.testing.assert_allclose(result.final_state.weights.sum(), 1.0)
    assert persist.persistent_state is not None
    assert persist.persistent_state.n_batches == len(result)
    assert persist.persistent_state.n_particles == len(result) * n_particles
    assert persist.persistent_state.betas[-1] == pytest.approx(1.0)
    assert (
        result.final_state.n_particles == persist.persistent_state.n_particles
    )


def test_persistent_checkpoint_resume_matches_uninterrupted_run(
    tmp_path,
) -> None:
    n_particles = 32
    target_ess = 2 * n_particles
    target_total_ess = 5 * n_particles

    def log_prior(particles, context=None):
        del context
        values = np.asarray(particles)
        return -0.5 * np.sum((values / 2.0) ** 2, axis=1)

    def log_likelihood(particles, context=None):
        del context
        values = np.asarray(particles)
        return -0.5 * np.sum((values - 1.5) ** 2, axis=1)

    initial_particles = np.random.default_rng(1234).normal(
        loc=0.0,
        scale=2.0,
        size=(n_particles, 1),
    )
    initial_state = ParticleState(initial_particles)

    def make_sampler(
        *,
        checkpoint_writers=(),
        callbacks=(),
    ) -> SequentialMonteCarlo:
        sampler, _, _ = make_persistent_bayesian_smc(
            log_prior,
            log_likelihood,
            n_particles=n_particles,
            target_ess=target_ess,
            target_total_ess=target_total_ess,
            mutation_n_steps=2,
            mutation_kwargs={"step_fn": "tpcn"},
            checkpoint_writers=checkpoint_writers,
            callbacks=callbacks,
        )
        return sampler

    reference = make_sampler().run(
        initial_state,
        rng=5678,
        metadata={"beta": 0.0},
    )
    first_nonterminal_beta_one = next(
        index
        for index, outputs in enumerate(reference.outputs)
        if float(outputs[1]["beta"]) == pytest.approx(1.0)
        and float(outputs[1]["effective_sample_size"]) < target_total_ess
    )
    split_after = first_nonterminal_beta_one + 1

    checkpoint_path = tmp_path / "persistent.pkl"
    writer = PickleCheckpointWriter(checkpoint_path, every=1)
    partial_sampler = make_sampler(checkpoint_writers=[writer])
    partial = partial_sampler.run(
        initial_state,
        num_steps=split_after,
        rng=5678,
        metadata={"beta": 0.0},
    )
    checkpoint = SMCCheckpoint.load_pickle(checkpoint_path)
    partial_persist = partial_sampler._steps[0]  # noqa: SLF001
    assert isinstance(partial_persist, PersistParticles)
    assert partial_persist.persistent_state is not None
    n_archived_batches = partial_persist.persistent_state.n_batches
    checkpointed_active = checkpoint.state.copy()

    assert checkpoint.next_iteration == split_after
    assert n_archived_batches == split_after
    assert not isinstance(checkpointed_active, PersistentParticleState)
    assert float(partial.context.metadata["beta"]) == pytest.approx(1.0)
    assert float(partial.outputs[-1][1]["effective_sample_size"]) < (
        target_total_ess
    )
    assert len(partial.outputs[-1]) == 4

    first_resumed_archive: dict[str, object] = {}
    resumed_steps: dict[str, PersistParticles] = {}

    def inspect_first_resumed_archive(state, context, result):
        del state, context, result
        if first_resumed_archive:
            return
        archive = resumed_steps["persist"].persistent_state
        assert archive is not None
        first_resumed_archive["n_batches"] = archive.n_batches
        first_resumed_archive["particles"] = np.array(
            archive.particles,
            copy=True,
        )

    resumed_sampler = make_sampler(callbacks=[inspect_first_resumed_archive])
    resumed_persist = resumed_sampler._steps[0]  # noqa: SLF001
    assert isinstance(resumed_persist, PersistParticles)
    resumed_steps["persist"] = resumed_persist
    runtime = resumed_sampler.resume_from(checkpoint)
    resumed = resumed_sampler.run(runtime=runtime)

    assert first_resumed_archive["n_batches"] == n_archived_batches + 1
    first_particles = np.asarray(first_resumed_archive["particles"])
    np.testing.assert_allclose(
        first_particles[-n_particles:],
        checkpointed_active.particles,
    )
    archived_batches = first_particles.reshape(
        n_archived_batches + 1,
        n_particles,
        -1,
    )
    assert (
        sum(
            np.array_equal(batch, checkpointed_active.particles)
            for batch in archived_batches
        )
        == 1
    )

    reference_final = reference.final_state
    resumed_final = resumed.final_state
    assert isinstance(reference_final, PersistentParticleState)
    assert isinstance(resumed_final, PersistentParticleState)
    assert resumed_final.batch_size == reference_final.batch_size
    assert resumed_final.n_batches == reference_final.n_batches
    assert resumed_final.n_particles == reference_final.n_particles
    np.testing.assert_allclose(
        resumed_final.particles, reference_final.particles
    )
    np.testing.assert_allclose(
        resumed_final.log_weights,
        reference_final.log_weights,
    )
    np.testing.assert_allclose(
        resumed_final.betas,
        reference_final.betas,
    )
    np.testing.assert_allclose(
        resumed_final.log_evidences,
        reference_final.log_evidences,
    )
    np.testing.assert_array_equal(
        resumed_final.initial_ancestor_indices,
        reference_final.initial_ancestor_indices,
    )
    assert resumed_final.log_evidence == pytest.approx(
        reference_final.log_evidence
    )
    assert resumed_final.evaluation is not None
    assert reference_final.evaluation is not None
    assert resumed_final.evaluation.components.keys() == (
        reference_final.evaluation.components.keys()
    )
    for component in reference_final.evaluation.components:
        np.testing.assert_allclose(
            resumed_final.evaluation[component],
            reference_final.evaluation[component],
        )

    reference_betas = np.asarray(
        [float(outputs[1]["beta"]) for outputs in reference.outputs]
    )
    resumed_betas = np.asarray(
        [float(outputs[1]["beta"]) for outputs in resumed.outputs]
    )
    np.testing.assert_allclose(resumed_betas, reference_betas)
    assert np.count_nonzero(np.isclose(reference_betas, 1.0)) >= 2
    assert len(resumed_final.betas) == len(reference_final.betas)
    assert len(resumed_final.log_evidences) == len(
        reference_final.log_evidences
    )

    for key in (
        "beta",
        "delta_beta",
        "target_ess",
        "target_total_ess",
    ):
        np.testing.assert_allclose(
            resumed.context.metadata[key],
            reference.context.metadata[key],
        )
        np.testing.assert_allclose(
            resumed_final.metadata[key],
            reference_final.metadata[key],
        )

    assert reference.context.stop_immediately
    assert resumed.context.stop_immediately

    reference_mean = np.average(
        reference_final.particles,
        axis=0,
        weights=reference_final.weights,
    )
    resumed_mean = np.average(
        resumed_final.particles,
        axis=0,
        weights=resumed_final.weights,
    )
    np.testing.assert_allclose(resumed_mean, reference_mean)
