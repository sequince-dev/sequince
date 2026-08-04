from __future__ import annotations

import numpy as np
import pytest
from orng import RandomGenerator

from sequince._array_api import namespace
from sequince.steps.base import StepContext
from sequince.steps.mutation import (
    AdaptiveMaskedGaussianRandomWalk,
    CallableMutationKernel,
    GaussianRandomWalk,
    MaskedGaussianRandomWalk,
    MetropolisHastingsMutation,
    MixtureMutationKernel,
    evaluate_log_transition_ratio,
    get_mutation_kernel,
)


def test_callable_symmetric_kernel(rng) -> None:
    particles = np.arange(6, dtype=float).reshape(3, 2)
    kernel = CallableMutationKernel(
        lambda values, inner_rng: values + 1.0,
        symmetric=True,
    )

    proposals = kernel.sample(particles, rng)
    ratio = kernel.log_transition_ratio(particles, proposals)

    np.testing.assert_allclose(proposals, particles + 1.0)
    np.testing.assert_allclose(ratio, 0.0)


def test_callable_kernel_computes_ratio_from_density(rng) -> None:
    particles = np.array([[0.0], [1.0], [2.0]])

    def log_density(start, end):
        return end[:, 0] - 2.0 * start[:, 0]

    kernel = CallableMutationKernel(
        lambda values, inner_rng: values + 1.0,
        log_density_fn=log_density,
    )
    proposals = kernel.sample(particles, rng)

    expected = log_density(proposals, particles) - log_density(
        particles, proposals
    )
    np.testing.assert_allclose(
        kernel.log_transition_ratio(particles, proposals), expected
    )


def test_callable_kernel_requires_one_transition_correction() -> None:
    def proposal(values, rng):
        return values

    with pytest.raises(ValueError, match="exactly one"):
        CallableMutationKernel(proposal)
    with pytest.raises(ValueError, match="exactly one"):
        CallableMutationKernel(
            proposal,
            symmetric=True,
            log_transition_ratio_fn=lambda start, end: start[:, 0],
        )


def test_legacy_density_kernel_transition_ratio(rng) -> None:
    particles = np.array([[0.0], [1.0], [2.0]])
    proposals = particles + 0.5
    kernel = GaussianRandomWalk(scale=0.3)

    ratio = evaluate_log_transition_ratio(kernel, particles, proposals)

    np.testing.assert_allclose(ratio, 0.0)


def test_get_mutation_kernel_constructs_named_kernel() -> None:
    kernel = get_mutation_kernel("gaussian-random-walk", scale=0.25)

    assert isinstance(kernel, GaussianRandomWalk)
    assert kernel.scale == 0.25


def test_get_mutation_kernel_validates_name() -> None:
    with pytest.raises(ValueError, match="Unknown mutation kernel"):
        get_mutation_kernel("unknown")


def test_mixture_kernel_selects_component_for_complete_sweep(rng) -> None:
    particles = np.arange(6, dtype=float).reshape(3, 2)
    first = CallableMutationKernel(
        lambda values, inner_rng: values + 1.0,
        symmetric=True,
    )
    second = CallableMutationKernel(
        lambda values, inner_rng: values + 100.0,
        symmetric=True,
    )
    kernel = MixtureMutationKernel([first, second], weights=[1.0, 0.0])

    proposals = kernel.sample(particles, rng)

    np.testing.assert_allclose(proposals, particles + 1.0)
    np.testing.assert_allclose(
        kernel.log_transition_ratio(particles, proposals),
        0.0,
    )
    assert kernel.selected_kernel is first


def test_mixture_kernel_requires_sample_before_transition_ratio() -> None:
    component = CallableMutationKernel(
        lambda values, inner_rng: values,
        symmetric=True,
    )
    kernel = MixtureMutationKernel([component])

    with pytest.raises(RuntimeError, match=r"sample\(\) must be called"):
        kernel.log_transition_ratio(np.zeros((3, 1)), np.zeros((3, 1)))


@pytest.mark.parametrize(
    ("kernels", "weights", "message"),
    [
        ([], None, "at least one kernel"),
        ([GaussianRandomWalk()], [1.0, 1.0], "must match"),
        ([GaussianRandomWalk()], [-1.0], "non-negative"),
        ([GaussianRandomWalk()], [0.0], "positive weight"),
    ],
)
def test_mixture_kernel_validates_configuration(
    kernels,
    weights,
    message,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        MixtureMutationKernel(kernels, weights=weights)


def test_get_mutation_kernel_builds_recursive_mixture() -> None:
    kernel = get_mutation_kernel(
        "mixture",
        kernels=[
            {
                "kernel": "gaussian-random-walk",
                "kernel_kwargs": {"scale": 0.25},
            },
            {
                "kernel": "gaussian-random-walk",
                "kernel_kwargs": {"scale": 0.1},
            },
        ],
        weights=[3.0, 1.0],
    )

    assert isinstance(kernel, MixtureMutationKernel)
    assert isinstance(kernel.kernels[0], GaussianRandomWalk)
    assert isinstance(kernel.kernels[1], GaussianRandomWalk)
    assert kernel.weights == (0.75, 0.25)


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax", "cupy"])
def test_gaussian_random_walk_mutation_preserves_backend(
    backend: str,
    backend_factory,
) -> None:
    case = backend_factory(backend)
    particles = case.particles
    backend_ns = namespace(particles)
    step = MetropolisHastingsMutation(
        lambda values, context: -0.5 * case.row_sum(values**2),
        GaussianRandomWalk(scale=0.1),
        n_steps=2,
        store_chains=True,
    )

    updated = step(
        case.state(),
        StepContext(t=0),
        case.rng(2468),
    )

    assert updated.particles.shape == particles.shape
    assert namespace(updated.particles) is backend_ns
    assert namespace(updated.metadata["mutation_chains"]) is backend_ns


def test_masked_gaussian_changes_only_active_coordinates(rng) -> None:
    particles = np.arange(20, dtype=float).reshape(5, 4)
    kernel = MaskedGaussianRandomWalk(indices=(1, 3), scale=0.2)

    proposals = kernel.sample(particles, rng)

    assert np.array_equal(proposals[:, (0, 2)], particles[:, (0, 2)])
    assert np.any(proposals[:, (1, 3)] != particles[:, (1, 3)])
    np.testing.assert_allclose(
        kernel.log_transition_ratio(particles, proposals),
        0.0,
    )


def test_masked_gaussian_uses_block_covariance() -> None:
    particles = np.zeros((20_000, 3))
    covariance = np.array([[1.0, 0.7], [0.7, 2.0]])
    kernel = MaskedGaussianRandomWalk(
        indices=(0, 2),
        covariance=covariance,
        scale=0.5,
    )

    proposals = kernel.sample(particles, RandomGenerator(seed=1234))

    empirical = np.cov(proposals[:, (0, 2)], rowvar=False)
    np.testing.assert_allclose(empirical, 0.25 * covariance, rtol=0.04)
    assert np.array_equal(proposals[:, 1], particles[:, 1])


def test_adaptive_gaussian_estimates_active_covariance() -> None:
    particles = np.array(
        [
            [-1.0, 10.0, -2.0],
            [0.0, 20.0, 0.0],
            [1.0, 30.0, 2.0],
        ]
    )
    factory = AdaptiveMaskedGaussianRandomWalk(
        indices=(0, 2),
        scale=1.0,
        regularization=1e-4,
    )

    kernel = factory(particles, StepContext(t=0))

    expected = np.cov(particles[:, (0, 2)], rowvar=False)
    expected += 1e-4 * np.eye(2)
    np.testing.assert_allclose(kernel.covariance, expected)
    assert kernel.indices == (0, 2)
    assert kernel.scale == 1.0


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax", "cupy"])
def test_mixture_mutation_kernel_preserves_backend(
    backend: str,
    backend_factory,
) -> None:
    case = backend_factory(backend)
    particles = case.particles
    backend_ns = namespace(particles)
    kernel = get_mutation_kernel(
        "mixture",
        kernels=[
            {
                "kernel": "gaussian-random-walk",
                "kernel_kwargs": {"scale": 0.2},
            },
            {
                "kernel": "gaussian-random-walk",
                "kernel_kwargs": {"scale": 0.1},
            },
        ],
        weights=[1.0, 0.0],
    )

    proposals = kernel.sample(
        particles,
        case.rng(1357),
    )

    assert proposals.shape == particles.shape
    assert namespace(proposals) is backend_ns
