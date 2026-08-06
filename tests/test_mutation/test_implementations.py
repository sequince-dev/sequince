from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from sequince.posterior import TemperedPosterior
from sequince.state import ParticleState
from sequince.steps.mutation import EmceeMutation, MiniPCNMutation
from sequince.transforms import AffineTransform

MUTATION_IMPLEMENTATIONS = [
    pytest.param(EmceeMutation, "emcee", id="emcee"),
    pytest.param(MiniPCNMutation, "minipcn", id="minipcn"),
]

TRANSFORM_IMPLEMENTATIONS = [
    pytest.param(EmceeMutation, "emcee", 1, 0, id="emcee"),
    pytest.param(MiniPCNMutation, "minipcn", 3, 1, id="minipcn"),
]


@pytest.mark.parametrize(
    ("mutation_type", "metadata_prefix"), MUTATION_IMPLEMENTATIONS
)
def test_mutation_runs_single_step(
    mutation_type: type[Any],
    metadata_prefix: str,
    particle_state: ParticleState,
    context_factory,
    rng,
    standard_normal_log_prob,
) -> None:
    context = context_factory(shared={"temperature": 1.0})
    mutated, context, _ = mutation_type(standard_normal_log_prob, n_steps=1)(
        particle_state, context, rng
    )

    assert mutated.particles.shape == particle_state.particles.shape
    assert f"{metadata_prefix}_acceptance" in mutated.metadata
    assert mutated.metadata[f"{metadata_prefix}_steps"] == 1


@pytest.mark.parametrize(
    ("mutation_type", "metadata_prefix"), MUTATION_IMPLEMENTATIONS
)
def test_mutation_evaluates_final_population(
    mutation_type: type[Any],
    metadata_prefix: str,
    quadratic_posterior: TemperedPosterior,
    particle_state: ParticleState,
    context_factory,
    rng,
) -> None:
    context = context_factory(metadata={"temperature": 1.0})
    mutated, context, _ = mutation_type(quadratic_posterior, n_steps=1)(
        particle_state, context, rng
    )

    assert mutated.evaluation is not None
    expected = quadratic_posterior.evaluate(mutated.particles, context)
    for component in ("log_prior", "log_likelihood"):
        np.testing.assert_allclose(
            mutated.evaluation[component], expected[component]
        )
    assert f"{metadata_prefix}_acceptance" in mutated.metadata


@pytest.mark.parametrize(
    ("mutation_type", "metadata_prefix", "n_steps", "column"),
    TRANSFORM_IMPLEMENTATIONS,
)
def test_mutation_with_transform(
    mutation_type: type[Any],
    metadata_prefix: str,
    n_steps: int,
    column: int,
    particle_state: ParticleState,
    context_factory,
    rng,
    standard_normal_log_prob,
) -> None:
    context = context_factory(shared={"temperature": 1.0})
    transform = AffineTransform(columns=[column])
    transform.fit(particle_state.particles)
    mutated, context, _ = mutation_type(
        standard_normal_log_prob,
        n_steps=n_steps,
        transform=transform,
    )(particle_state, context, rng)

    assert mutated.particles.shape == particle_state.particles.shape
    assert f"{metadata_prefix}_acceptance" in mutated.metadata
    assert mutated.metadata[f"{metadata_prefix}_steps"] == n_steps
