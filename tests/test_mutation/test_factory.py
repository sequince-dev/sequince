from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from sequince.steps import DifferentialEvolutionMutation as PublicDEMutation
from sequince.steps.base import SMCStep, StepContext
from sequince.steps.mutation import (
    BlackJAXMutation,
    DifferentialEvolutionKernel,
    DifferentialEvolutionMutation,
    GaussianRandomWalk,
    MetropolisHastingsMutation,
    MiniPCNMutation,
    MixtureMutationKernel,
    RedBlueMetropolisHastingsMutation,
    get_mutation_step,
)
from sequince.steps.mutation.metropolis import (
    DifferentialEvolutionMutation as DirectDEMutation,
)

if TYPE_CHECKING:
    from orng import RandomGenerator


def test_differential_evolution_mutation_is_publicly_importable() -> None:
    assert PublicDEMutation is DifferentialEvolutionMutation
    assert DirectDEMutation is DifferentialEvolutionMutation


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


def test_get_mutation_step_constructs_differential_evolution(
    standard_normal_posterior,
    particle_state,
    context_factory,
    rng,
) -> None:
    mutation = get_mutation_step(
        "differential-evolution",
        standard_normal_posterior,
        mutation_n_steps=2,
        mutation_kwargs={
            "gamma": 0.7,
            "gamma_one_probability": 0.0,
            "gamma_randomization": "bilby",
        },
    )

    assert isinstance(mutation, DifferentialEvolutionMutation)
    kernel = mutation._complementary_ensemble_kernel  # noqa: SLF001
    assert isinstance(kernel, DifferentialEvolutionKernel)
    assert kernel.gamma == 0.7
    assert kernel.gamma_randomization == "bilby"
    updated, _, output = mutation(particle_state, context_factory(), rng)
    assert updated.particles.shape == particle_state.particles.shape
    assert output["steps"] == 2


def test_get_mutation_step_constructs_named_red_blue_kernel(
    standard_normal_posterior,
) -> None:
    mutation = get_mutation_step(
        "red-blue",
        standard_normal_posterior,
        mutation_n_steps=3,
        mutation_kwargs={
            "kernel": "differential-evolution",
            "kernel_kwargs": {"gamma": 0.5},
        },
    )

    assert isinstance(mutation, RedBlueMetropolisHastingsMutation)
    kernel = mutation._complementary_ensemble_kernel  # noqa: SLF001
    assert isinstance(kernel, DifferentialEvolutionKernel)
    assert kernel.gamma == 0.5
    assert mutation._n_steps == 3  # noqa: SLF001


def test_factory_rejects_kernel_from_wrong_registry(
    standard_normal_posterior,
) -> None:
    with pytest.raises(ValueError, match="Unknown mutation kernel"):
        get_mutation_step(
            "metropolis-hastings",
            standard_normal_posterior,
            mutation_kwargs={"kernel": "differential-evolution"},
        )
    with pytest.raises(ValueError, match="Unknown complementary-ensemble"):
        get_mutation_step(
            "red-blue",
            standard_normal_posterior,
            mutation_kwargs={"kernel": "gaussian-random-walk"},
        )


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


@pytest.mark.parametrize("name", ["minipcn", "differential-evolution"])
def test_factory_adapts_completed_budget(
    name, standard_normal_posterior, particle_state, context_factory, rng
) -> None:
    from sequince.steps.mutation.adaptive.steps import AdaptMutationSteps

    step = get_mutation_step(
        name,
        standard_normal_posterior,
        mutation_n_steps=2,
        adaptive_mutation={
            "min_steps": 1,
            "max_steps": 4,
            "target_accepted_moves": 100.0,
        },
    )
    assert isinstance(step, AdaptMutationSteps)
    state, _, output = step(particle_state, context_factory(t=0), rng)
    assert output["metrics"].n_steps == 2
    assert step.next_n_steps == 4
    _, _, output = step(state, context_factory(t=1), rng)
    assert output["metrics"].n_steps == 4


@pytest.mark.parametrize(
    "config,error,message",
    [
        (True, TypeError, "mapping"),
        ({"rho": 0.5}, ValueError, "Unknown"),
        ({"enabled": 1}, TypeError, "boolean"),
        ({"min_steps": True}, TypeError, "integer"),
        ({"max_steps": 2.5}, TypeError, "integer"),
        ({"min_steps": -1}, ValueError, ">= 0"),
        ({"min_steps": 5, "max_steps": 2}, ValueError, ">= min_steps"),
        ({"target_accepted_moves": "5"}, TypeError, "real number"),
        ({"target_accepted_moves": False}, TypeError, "real number"),
        ({"target_accepted_moves": float("nan")}, ValueError, "finite"),
        ({"target_accepted_moves": 0}, ValueError, "positive"),
        ({"min_steps": 20}, ValueError, "mutation_n_steps"),
        ({"enabled": False, "max_steps": -1}, ValueError, ">= 0"),
    ],
)
def test_factory_invalid_adaptation(
    config, error, message, standard_normal_posterior
) -> None:
    with pytest.raises(error, match=message):
        get_mutation_step(
            "minipcn", standard_normal_posterior, adaptive_mutation=config
        )


@pytest.mark.parametrize(
    "mutation", [None, "none", "emcee", "red-blue", IdentityMutation()]
)
def test_factory_unsupported_adaptation(mutation, standard_normal_posterior):
    with pytest.raises(ValueError, match="requires named"):
        get_mutation_step(
            mutation,
            standard_normal_posterior,
            adaptive_mutation={"enabled": True},
        )


@pytest.mark.parametrize("name", ["minipcn", "differential-evolution", "none"])
def test_factory_disabled_adaptation(name, standard_normal_posterior) -> None:
    plain = get_mutation_step(name, standard_normal_posterior)
    disabled = get_mutation_step(
        name, standard_normal_posterior, adaptive_mutation={"enabled": False}
    )
    assert type(disabled) is type(plain)
