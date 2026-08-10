from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from sequince.posterior import TemperedPosterior
from sequince.state import ParticleState
from sequince.steps.mutation import EmceeMutation, MiniPCNMutation
from sequince.transforms import AffineTransform

MUTATION_IMPLEMENTATIONS = [
    pytest.param(EmceeMutation, id="emcee"),
    pytest.param(MiniPCNMutation, id="minipcn"),
]

TRANSFORM_IMPLEMENTATIONS = [
    pytest.param(EmceeMutation, 1, 0, id="emcee"),
    pytest.param(MiniPCNMutation, 3, 1, id="minipcn"),
]


@pytest.mark.parametrize("mutation_type", MUTATION_IMPLEMENTATIONS)
def test_mutation_runs_single_step(
    mutation_type: type[Any],
    particle_state: ParticleState,
    context_factory,
    rng,
    standard_normal_log_prob,
) -> None:
    context = context_factory(shared={"temperature": 1.0})
    mutated, context, output = mutation_type(
        standard_normal_log_prob,
        n_steps=1,
        store_chains=True,
    )(particle_state, context, rng)

    assert mutated.particles.shape == particle_state.particles.shape
    assert "acceptance" in output
    assert output["steps"] == 1
    assert "mutation_chains" not in mutated.metadata
    assert output["chain"].shape[0] == 1


@pytest.mark.parametrize("mutation_type", MUTATION_IMPLEMENTATIONS)
def test_mutation_evaluates_final_population(
    mutation_type: type[Any],
    quadratic_posterior: TemperedPosterior,
    particle_state: ParticleState,
    context_factory,
    rng,
) -> None:
    context = context_factory(metadata={"temperature": 1.0})
    mutated, context, output = mutation_type(quadratic_posterior, n_steps=1)(
        particle_state, context, rng
    )

    assert mutated.evaluation is not None
    expected = quadratic_posterior.evaluate(mutated.particles, context)
    for component in ("log_prior", "log_likelihood"):
        np.testing.assert_allclose(
            mutated.evaluation[component], expected[component]
        )
    assert "acceptance" in output


@pytest.mark.parametrize(
    ("mutation_type", "n_steps", "column"),
    TRANSFORM_IMPLEMENTATIONS,
)
def test_mutation_with_transform(
    mutation_type: type[Any],
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
    mutated, context, output = mutation_type(
        standard_normal_log_prob,
        n_steps=n_steps,
        transform=transform,
    )(particle_state, context, rng)

    assert mutated.particles.shape == particle_state.particles.shape
    assert "acceptance" in output
    assert output["steps"] == n_steps
