from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from sequince.steps.base import SMCStep, StepContext
from sequince.steps.mutation import (
    BlackJAXMutation,
    GaussianRandomWalk,
    MetropolisHastingsMutation,
    MiniPCNMutation,
    MixtureMutationKernel,
    get_mutation_step,
)

if TYPE_CHECKING:
    from orng import RandomGenerator


def test_get_mutation_step_constructs_blackjax(
    standard_normal_posterior,
) -> None:
    mutation = get_mutation_step(
        "blackjax",
        standard_normal_posterior,
        mutation_n_steps=3,
        mutation_kwargs={
            "algorithm": "random-walk",
            "algorithm_kwargs": {"sigma": 0.25},
        },
    )

    assert isinstance(mutation, BlackJAXMutation)
    assert mutation.algorithm_name == "random-walk"
    assert mutation._n_steps == 3  # noqa: SLF001


def test_get_mutation_step_constructs_minipcn(
    standard_normal_posterior,
) -> None:
    mutation = get_mutation_step("minipcn", standard_normal_posterior)

    assert isinstance(mutation, MiniPCNMutation)


def test_get_mutation_step_passes_transform_refit(
    standard_normal_posterior,
) -> None:
    mutation = get_mutation_step(
        "minipcn",
        standard_normal_posterior,
        mutation_kwargs={"transform_refit": "per-context"},
    )

    assert isinstance(mutation, MiniPCNMutation)
    assert mutation._transform_refit == "per-context"  # noqa: SLF001


def test_get_mutation_step_constructs_gaussian_random_walk(
    standard_normal_posterior,
) -> None:
    mutation = get_mutation_step(
        "metropolis-hastings",
        standard_normal_posterior,
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


def test_get_mutation_step_requires_metropolis_hastings_kernel(
    standard_normal_posterior,
) -> None:
    with pytest.raises(ValueError, match="requires a 'kernel'"):
        get_mutation_step("metropolis-hastings", standard_normal_posterior)


def test_get_mutation_step_constructs_mixture_kernel(
    standard_normal_posterior,
) -> None:
    mutation = get_mutation_step(
        "metropolis-hastings",
        standard_normal_posterior,
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


def test_get_mutation_step_can_disable_mutation(
    standard_normal_posterior,
) -> None:
    mutation = get_mutation_step("none", standard_normal_posterior)

    assert mutation is None


class IdentityMutation(SMCStep):
    def apply(self, state, context: StepContext, rng: "RandomGenerator"):
        return state, context, None


def test_get_mutation_step_preserves_custom_step(
    standard_normal_posterior,
) -> None:
    custom = IdentityMutation()

    assert get_mutation_step(custom, standard_normal_posterior) is custom


def test_get_mutation_step_validates_mutation_name(
    standard_normal_posterior,
) -> None:
    with pytest.raises(ValueError, match="mutation must be"):
        get_mutation_step("unknown", standard_normal_posterior)  # type: ignore[arg-type]


def test_factory_does_not_expose_proposal_rejuvenation(
    standard_normal_posterior,
) -> None:
    with pytest.raises(ValueError, match="mutation must be"):
        get_mutation_step(  # type: ignore[arg-type]
            "proposal",
            standard_normal_posterior,
        )
