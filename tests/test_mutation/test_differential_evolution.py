from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from orng import RandomGenerator

from sequince.array_api import namespace
from sequince.state import ParticleState
from sequince.steps.mutation import (
    DifferentialEvolutionKernel,
    RedBlueMetropolisHastingsMutation,
)
from sequince.steps.mutation.kernels import (
    _sample_donor_indices,
    _sample_effective_gamma,
)


class StubRandomGenerator:
    def __init__(
        self,
        *,
        choices: list[np.ndarray],
        gamma: np.ndarray | None = None,
        random: np.ndarray | None = None,
    ) -> None:
        self.choices = list(choices)
        self.gamma_values = gamma
        self.random_values = random

    def choice(self, *args: Any, **kwargs: Any) -> np.ndarray:
        del args, kwargs
        return self.choices.pop(0)

    def gamma(self, *args: Any, **kwargs: Any) -> np.ndarray:
        del args, kwargs
        assert self.gamma_values is not None
        return self.gamma_values

    def random(self, *args: Any, **kwargs: Any) -> np.ndarray:
        del args, kwargs
        assert self.random_values is not None
        return self.random_values


def _controlled_rng(
    donor_j: list[int],
    donor_k_raw: list[int],
    *,
    factors: list[float] | None = None,
) -> StubRandomGenerator:
    return StubRandomGenerator(
        choices=[np.asarray(donor_j), np.asarray(donor_k_raw)],
        gamma=None if factors is None else np.asarray(factors),
    )


def test_fixed_rng_replays_proposals() -> None:
    active = np.arange(12.0).reshape(6, 2)
    complementary = np.arange(8.0).reshape(4, 2)
    kernel = DifferentialEvolutionKernel(gamma_randomization="bilby")

    first = kernel.sample(active, complementary, RandomGenerator(seed=42))
    replayed = kernel.sample(active, complementary, RandomGenerator(seed=42))

    np.testing.assert_array_equal(first, replayed)


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
def test_shape_dtype_and_backend_are_preserved(
    backend, backend_factory
) -> None:
    case = backend_factory(backend)
    kernel = DifferentialEvolutionKernel(gamma_randomization="bilby")

    proposals = kernel.sample(
        case.particles[:2],
        case.particles,
        case.rng(123),
    )

    assert proposals.shape == case.particles[:2].shape
    assert proposals.dtype == case.particles.dtype
    assert namespace(proposals) is namespace(case.particles)


def test_complementary_population_requires_at_least_two_particles() -> None:
    kernel = DifferentialEvolutionKernel()
    with pytest.raises(ValueError, match="at least two complementary"):
        kernel.sample(
            np.zeros((2, 2)),
            np.zeros((1, 2)),
            RandomGenerator(seed=1),
        )


@pytest.mark.parametrize(
    ("complementary", "active", "message"),
    [
        (np.zeros(2), np.zeros((2, 2)), "2D complementary"),
        (np.zeros((2, 2)), np.zeros(2), "2D active"),
        (np.zeros((2, 3)), np.zeros((2, 2)), "parameter dimension"),
    ],
)
def test_particle_shapes_are_validated(complementary, active, message) -> None:
    kernel = DifferentialEvolutionKernel()
    with pytest.raises(ValueError, match=message):
        kernel.sample(active, complementary, RandomGenerator(seed=2))


def test_donor_indices_are_distinct_uniform_ordered_pairs() -> None:
    n_complementary = 4
    n_samples = 120_000
    donor_j, donor_k = _sample_donor_indices(
        n_samples,
        n_complementary,
        RandomGenerator(seed=91),
    )
    donor_j = np.asarray(donor_j)
    donor_k = np.asarray(donor_k)

    assert np.all((0 <= donor_j) & (donor_j < n_complementary))
    assert np.all((0 <= donor_k) & (donor_k < n_complementary))
    assert np.all(donor_j != donor_k)

    counts = np.zeros((n_complementary, n_complementary), dtype=int)
    np.add.at(counts, (donor_j, donor_k), 1)
    ordered_counts = counts[~np.eye(n_complementary, dtype=bool)]
    expected = n_samples / (n_complementary * (n_complementary - 1))
    assert np.max(np.abs(ordered_counts - expected)) < 0.05 * expected


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
def test_donor_indices_are_backend_integer_index_arrays(
    backend,
    backend_factory,
) -> None:
    case = backend_factory(backend)
    donor_j, donor_k = _sample_donor_indices(
        8,
        case.particles.shape[0],
        case.rng(81),
    )

    assert case.xp.isdtype(donor_j.dtype, "integral")
    assert case.xp.isdtype(donor_k.dtype, "integral")
    assert namespace(donor_j) is namespace(case.particles)
    assert namespace(donor_k) is namespace(case.particles)
    assert case.particles[donor_j].shape == (8, case.particles.shape[1])
    assert case.particles[donor_k].shape == (8, case.particles.shape[1])


def test_donors_come_only_from_complementary_population() -> None:
    active = np.full((3, 1), 100.0)
    complementary = np.array([[0.0], [2.0], [5.0]])
    kernel = DifferentialEvolutionKernel(gamma=1.0, gamma_one_probability=0.0)

    proposals = kernel.sample(active, complementary, RandomGenerator(seed=4))
    displacements = set((proposals - active)[:, 0])
    complementary_differences = {
        float(left - right)
        for left in complementary[:, 0]
        for right in complementary[:, 0]
        if left != right
    }

    assert displacements <= complementary_differences


def test_default_gamma_uses_latent_dimension() -> None:
    active = np.zeros((2, 2))
    complementary = np.array([[0.0, 0.0], [1.0, 2.0]])
    kernel = DifferentialEvolutionKernel(gamma_one_probability=0.0)
    rng = _controlled_rng([1, 1], [0, 0])

    proposals = kernel.sample(  # type: ignore[arg-type]
        active,
        complementary,
        rng,
    )

    gamma_base = 2.38 / np.sqrt(2 * active.shape[1])
    np.testing.assert_allclose(
        proposals,
        active + gamma_base * (complementary[1] - complementary[0]),
    )


def test_standard_mode_uses_explicit_gamma_exactly_without_jitter() -> None:
    active = np.array([[10.0, 20.0], [30.0, 40.0]])
    complementary = np.array([[1.0, 2.0], [4.0, 8.0], [9.0, 7.0]])
    kernel = DifferentialEvolutionKernel(
        gamma=0.7,
        gamma_one_probability=0.0,
        gamma_randomization="none",
    )
    rng = _controlled_rng([0, 2], [1, 0])

    proposals = kernel.sample(  # type: ignore[arg-type]
        active,
        complementary,
        rng,
    )

    expected = active + 0.7 * np.stack(
        [
            complementary[0] - complementary[2],
            complementary[2] - complementary[0],
        ]
    )
    np.testing.assert_array_equal(proposals, expected)


def test_bilby_mode_uses_controlled_gamma_factors_exactly() -> None:
    active = np.array([[10.0], [20.0]])
    complementary = np.array([[1.0], [4.0], [9.0]])
    kernel = DifferentialEvolutionKernel(
        gamma=0.7,
        gamma_one_probability=0.0,
        gamma_randomization="bilby",
    )
    rng = _controlled_rng([0, 2], [1, 0], factors=[0.5, 1.5])

    proposals = kernel.sample(  # type: ignore[arg-type]
        active,
        complementary,
        rng,
    )

    expected = active + np.array([[0.7 * 0.5], [0.7 * 1.5]]) * np.stack(
        [
            complementary[0] - complementary[2],
            complementary[2] - complementary[0],
        ]
    )
    np.testing.assert_array_equal(proposals, expected)


def test_bilby_gamma_factor_has_mean_one() -> None:
    factors = _sample_effective_gamma(
        100_000,
        1.0,
        0.0,
        "bilby",
        RandomGenerator(seed=7),
        like=np.zeros(1),
    )

    assert np.mean(factors) == pytest.approx(1.0, abs=0.015)


@pytest.mark.parametrize("gamma", [0.0, -1.0, np.nan, np.inf])
def test_invalid_gamma_fails(gamma: float) -> None:
    with pytest.raises(ValueError, match="finite and strictly positive"):
        DifferentialEvolutionKernel(gamma=gamma)


def test_invalid_randomization_strategy_fails_clearly() -> None:
    with pytest.raises(ValueError, match="'none', 'bilby'"):
        DifferentialEvolutionKernel(
            gamma_randomization="foo",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("probability", [-0.1, 1.1, np.nan, np.inf])
def test_invalid_gamma_one_probability_fails(probability: float) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        DifferentialEvolutionKernel(gamma_one_probability=probability)


def test_gamma_one_probability_boundaries_override_ordinary_scale() -> None:
    like = np.zeros(4)
    never = _sample_effective_gamma(
        4,
        0.3,
        0.0,
        "none",
        StubRandomGenerator(choices=[]),  # type: ignore[arg-type]
        like=like,
    )
    always = _sample_effective_gamma(
        4,
        0.3,
        1.0,
        "bilby",
        StubRandomGenerator(choices=[]),  # type: ignore[arg-type]
        like=like,
    )

    np.testing.assert_array_equal(never, 0.3)
    np.testing.assert_array_equal(always, 1.0)


def test_intermediate_gamma_one_probability_is_per_proposal() -> None:
    scales = _sample_effective_gamma(
        50_000,
        0.3,
        0.25,
        "none",
        RandomGenerator(seed=71),
        like=np.zeros(1),
    )

    assert np.mean(scales == 1.0) == pytest.approx(0.25, abs=0.01)


def test_transition_ratio_is_zero_per_active_particle() -> None:
    current = np.zeros((5, 2), dtype=np.float32)
    proposed = np.ones_like(current)
    complementary = np.zeros((2, 2), dtype=np.float32)
    kernel = DifferentialEvolutionKernel()

    ratio = kernel.log_transition_ratio(
        current,
        proposed,
        complementary,
    )

    assert ratio.shape == (5,)
    assert ratio.dtype == current.dtype
    np.testing.assert_array_equal(ratio, 0.0)


@pytest.mark.parametrize("gamma_randomization", ["none", "bilby"])
def test_red_blue_integration_uses_sequential_frozen_complements(
    gamma_randomization,
    context_factory,
) -> None:
    particles = np.arange(12.0).reshape(6, 2) / 4.0
    kernel = DifferentialEvolutionKernel(
        gamma=0.4,
        gamma_one_probability=0.0,
        gamma_randomization=gamma_randomization,
    )

    mutation = RedBlueMetropolisHastingsMutation(
        lambda values, context: np.zeros(values.shape[0]),
        kernel,
    )
    updated, _, _ = mutation(
        ParticleState(particles),
        context_factory(),
        RandomGenerator(seed=25),
    )

    assert updated.particles.shape == particles.shape
    assert updated.particles.shape[0] == particles.shape[0]
    assert mutation._complementary_ensemble_kernel is kernel  # noqa: SLF001
