from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from sequince.posterior import TemperedPosterior
from sequince.steps.base import SMCStep, StepContext
from sequince.steps.mutation import (
    GaussianRandomWalk,
    MetropolisHastingsMutation,
    MiniPCNMutation,
    MixtureMutationKernel,
    get_mutation_step,
)

if TYPE_CHECKING:
    from orng import RandomGenerator


def _posterior() -> TemperedPosterior:
    def log_prior(particles: np.ndarray, context=None) -> np.ndarray:
        return -0.5 * np.sum(particles**2, axis=-1)

    def log_likelihood(particles: np.ndarray, context=None) -> np.ndarray:
        return np.zeros(particles.shape[0])

    return TemperedPosterior(log_prior, log_likelihood)


def test_get_mutation_step_constructs_minipcn() -> None:
    mutation = get_mutation_step("minipcn", _posterior())

    assert isinstance(mutation, MiniPCNMutation)


def test_get_mutation_step_passes_transform_refit() -> None:
    mutation = get_mutation_step(
        "minipcn",
        _posterior(),
        mutation_kwargs={"transform_refit": "per-context"},
    )

    assert isinstance(mutation, MiniPCNMutation)
    assert mutation._transform_refit == "per-context"  # noqa: SLF001


def test_get_mutation_step_constructs_gaussian_random_walk() -> None:
    mutation = get_mutation_step(
        "metropolis-hastings",
        _posterior(),
        mutation_n_steps=3,
        mutation_kwargs={
            "kernel": "gaussian-random-walk",
            "kernel_kwargs": {"scale": 0.25},
            "store_chains": True,
        },
    )

    assert isinstance(mutation, MetropolisHastingsMutation)
    assert isinstance(mutation._kernel, GaussianRandomWalk)  # noqa: SLF001
    assert mutation._kernel.scale == 0.25  # noqa: SLF001
    assert mutation._n_steps == 3  # noqa: SLF001


def test_get_mutation_step_requires_metropolis_hastings_kernel() -> None:
    with pytest.raises(ValueError, match="requires a 'kernel'"):
        get_mutation_step("metropolis-hastings", _posterior())


def test_get_mutation_step_constructs_mixture_kernel() -> None:
    mutation = get_mutation_step(
        "metropolis-hastings",
        _posterior(),
        mutation_kwargs={
            "kernel": "mixture",
            "kernel_kwargs": {
                "kernels": [
                    {
                        "kernel": "gaussian-random-walk",
                        "kernel_kwargs": {"scale": 0.5},
                    },
                    {
                        "kernel": "gaussian-random-walk",
                        "kernel_kwargs": {"scale": 0.2},
                    },
                ],
                "weights": [0.8, 0.2],
            },
        },
    )

    assert isinstance(mutation, MetropolisHastingsMutation)
    mixture = mutation._kernel  # noqa: SLF001
    assert isinstance(mixture, MixtureMutationKernel)
    assert isinstance(mixture.kernels[0], GaussianRandomWalk)
    assert isinstance(mixture.kernels[1], GaussianRandomWalk)


def test_get_mutation_step_can_disable_mutation() -> None:
    mutation = get_mutation_step("none", _posterior())

    assert mutation is None


class IdentityMutation(SMCStep):
    def apply(self, state, context: StepContext, rng: "RandomGenerator"):
        return state


def test_get_mutation_step_preserves_custom_step() -> None:
    custom = IdentityMutation()

    assert get_mutation_step(custom, _posterior()) is custom


def test_get_mutation_step_validates_mutation_name() -> None:
    with pytest.raises(ValueError, match="mutation must be"):
        get_mutation_step("unknown", _posterior())  # type: ignore[arg-type]
