from __future__ import annotations

import numpy as np
import pytest

from sequince import ParticleState
from sequince.steps.base import SMCStep
from sequince.steps.control import RepeatStep, SequentialStep
from sequince.steps.mutation import (
    CallableMutationKernel,
    MetropolisHastingsMutation,
)


def _flat_log_prob(particles, context):
    del context
    return np.zeros(particles.shape[0])


def _add(value):
    return CallableMutationKernel(
        lambda particles, rng: particles + value,
        symmetric=True,
    )


class NegateStep(SMCStep):
    def apply(self, state, context, rng):
        del rng
        return state.replace_particles(-state.particles), context, None


def test_deterministic_esjd_has_expected_counts_and_normalisation(
    context_factory, rng
) -> None:
    state = ParticleState(np.array([[-1.0, 0.0], [1.0, 0.0]]))
    mutation = MetropolisHastingsMutation(
        _flat_log_prob,
        _add(np.array([1.0, 2.0])),
        n_steps=2,
    )

    updated, _, output = mutation(state, context_factory(), rng)
    diagnostics = output["diagnostics"]

    assert diagnostics.n_steps == 2
    assert diagnostics.n_attempted == 4
    assert diagnostics.n_accepted == 4
    assert diagnostics.acceptance == pytest.approx(1.0)
    # Initial variances are (1, 0); the zero-variance coordinate uses scale 1.
    # Each particle moves (1, 2), hence (1 + 4) / two dimensions = 2.5.
    assert diagnostics.normalised_esjd_per_step == pytest.approx(2.5)
    np.testing.assert_allclose(updated.particles, state.particles + [2.0, 4.0])


def test_rejected_proposals_count_as_attempts_and_zero_esjd(
    context_factory, rng
) -> None:
    state = ParticleState(np.zeros((3, 1)))

    def target(particles, context):
        del context
        return np.where(particles[:, 0] == 0.0, 0.0, -np.inf)

    mutation = MetropolisHastingsMutation(target, _add(1.0), n_steps=2)
    updated, _, output = mutation(state, context_factory(), rng)
    diagnostics = output["diagnostics"]

    assert diagnostics.n_steps == 2
    assert diagnostics.n_attempted == 6
    assert diagnostics.n_accepted == 0
    assert diagnostics.acceptance == 0.0
    assert diagnostics.normalised_esjd_per_step == 0.0
    np.testing.assert_array_equal(updated.particles, state.particles)


def test_partial_acceptance_is_counted_per_particle(
    context_factory, rng
) -> None:
    state = ParticleState(np.array([[-1.0], [1.0]]))

    def target(particles, context):
        del context
        return np.where(particles[:, 0] <= 1.0, 0.0, -np.inf)

    _, _, output = MetropolisHastingsMutation(target, _add(1.0))(
        state, context_factory(), rng
    )
    diagnostics = output["diagnostics"]

    assert diagnostics.n_attempted == 2
    assert diagnostics.n_accepted == 1
    assert diagnostics.acceptance == pytest.approx(0.5)
    assert diagnostics.normalised_esjd_per_step == pytest.approx(0.5)


def test_weighted_pre_mutation_variance_is_frozen(
    context_factory, rng
) -> None:
    state = ParticleState(
        np.array([[0.0], [2.0]]),
        log_weights=np.log([0.75, 0.25]),
    )
    mutation = MetropolisHastingsMutation(
        _flat_log_prob,
        _add(np.sqrt(0.75)),
        n_steps=3,
    )

    _, _, output = mutation(state, context_factory(), rng)

    # The weighted initial variance is 0.75. Every transition therefore has
    # unit standardised squared displacement even as the population moves.
    assert output["diagnostics"].normalised_esjd_per_step == pytest.approx(1.0)


def test_zero_variance_dimensions_use_unit_scaling(
    context_factory, rng
) -> None:
    state = ParticleState(np.zeros((4, 2)))
    mutation = MetropolisHastingsMutation(
        _flat_log_prob,
        _add(np.array([1.0, 2.0])),
    )

    _, _, output = mutation(state, context_factory(), rng)

    assert np.isfinite(output["diagnostics"].normalised_esjd_per_step)
    assert output["diagnostics"].normalised_esjd_per_step == pytest.approx(2.5)


def test_zero_step_mutation_reports_zero_diagnostics(
    particle_state, context_factory, rng
) -> None:
    mutation = MetropolisHastingsMutation(_flat_log_prob, _add(1.0))

    updated, _, output = mutation.apply_with_n_steps(
        particle_state, context_factory(), rng, n_steps=0
    )
    diagnostics = output["diagnostics"]

    assert diagnostics.n_steps == 0
    assert diagnostics.n_attempted == 0
    assert diagnostics.n_accepted == 0
    assert diagnostics.acceptance == 0.0
    assert diagnostics.normalised_esjd_per_step == 0.0
    np.testing.assert_array_equal(updated.particles, particle_state.particles)


def test_repeated_rounds_share_scaling_and_exclude_symmetry(
    context_factory, rng
) -> None:
    state = ParticleState(np.array([[-1.0], [1.0]]))
    mutation = MetropolisHastingsMutation(_flat_log_prob, _add(1.0))
    repeated = RepeatStep(SequentialStep([mutation, NegateStep()]), n=2)

    updated, context, outputs = repeated.apply_with_n_steps(
        state, context_factory(), rng, n_steps=2
    )
    diagnostics = outputs.diagnostics

    assert isinstance(outputs, tuple)
    assert all(
        round_output[0]["diagnostics"] is None for round_output in outputs
    )
    assert diagnostics.n_steps == 2
    assert diagnostics.n_attempted == 4
    assert diagnostics.n_accepted == 4
    assert diagnostics.normalised_esjd_per_step == pytest.approx(1.0)
    assert "_mutation_diagnostics_accumulator" not in context.metadata
    # Both mutation and symmetry rounds still execute exactly as before.
    np.testing.assert_array_equal(updated.particles, state.particles)


def test_repeated_mutation_returns_one_complete_block_diagnostic(
    context_factory, rng
) -> None:
    state = ParticleState(np.array([[-1.0], [1.0]]))
    repeated = RepeatStep(
        MetropolisHastingsMutation(_flat_log_prob, _add(1.0)), n=3
    )

    updated, context, outputs = repeated.apply_with_n_steps(
        state, context_factory(), rng, n_steps=5
    )

    assert [output["steps"] for output in outputs] == [2, 2, 1]
    assert all(output["diagnostics"] is None for output in outputs)
    assert outputs.diagnostics.n_steps == 5
    assert outputs.diagnostics.n_attempted == 10
    assert outputs.diagnostics.n_accepted == 10
    assert outputs.diagnostics.normalised_esjd_per_step == pytest.approx(1.0)
    assert "_mutation_diagnostics_accumulator" not in context.metadata
    np.testing.assert_array_equal(updated.particles, state.particles + 5.0)


def test_repeated_mutation_budget_smaller_than_round_count(
    context_factory, rng
) -> None:
    state = ParticleState(np.array([[-1.0], [1.0]]))
    repeated = RepeatStep(
        MetropolisHastingsMutation(_flat_log_prob, _add(1.0)), n=4
    )

    updated, _, outputs = repeated.apply_with_n_steps(
        state, context_factory(), rng, n_steps=2
    )

    assert [output["steps"] for output in outputs] == [1, 1, 0, 0]
    assert all(output["diagnostics"] is None for output in outputs)
    assert outputs.diagnostics.n_steps == 2
    assert outputs.diagnostics.n_attempted == 4
    assert outputs.diagnostics.n_accepted == 4
    assert outputs.diagnostics.normalised_esjd_per_step == pytest.approx(1.0)
    np.testing.assert_array_equal(updated.particles, state.particles + 2.0)


def test_nested_repeats_only_finalise_the_outer_block(
    context_factory, rng
) -> None:
    state = ParticleState(np.array([[-1.0], [1.0]]))
    inner = RepeatStep(
        MetropolisHastingsMutation(_flat_log_prob, _add(1.0)), n=2
    )
    outer = RepeatStep(inner, n=2)

    updated, context, outputs = outer.apply_with_n_steps(
        state, context_factory(), rng, n_steps=4
    )

    assert outputs.diagnostics.n_steps == 4
    assert outputs.diagnostics.n_attempted == 8
    assert all(inner_output.diagnostics is None for inner_output in outputs)
    assert all(
        child_output["diagnostics"] is None
        for inner_output in outputs
        for child_output in inner_output
    )
    assert "_mutation_diagnostics_accumulator" not in context.metadata
    np.testing.assert_array_equal(updated.particles, state.particles + 4.0)
