from __future__ import annotations

import numpy as np
import pytest
from orng import ArrayRNG

from sequince._array_api import namespace
from sequince.state import ParticleState
from sequince.steps.base import StepContext
from sequince.steps.mutation import (
    CallableMutationKernel,
    GaussianRandomWalk,
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
    _, particles, log_weights, row_sum = backend_factory(backend)
    backend_ns = namespace(particles)
    state = ParticleState(particles=particles, log_weights=log_weights)
    step = MetropolisHastingsMutation(
        lambda values, context: -0.5 * row_sum(values**2),
        GaussianRandomWalk(scale=0.1),
        n_steps=2,
        store_chains=True,
    )

    updated = step(
        state,
        StepContext(t=0),
        ArrayRNG(backend=backend, seed=2468),
    )

    assert updated.particles.shape == particles.shape
    assert namespace(updated.particles) is backend_ns
    assert namespace(updated.metadata["mutation_chains"]) is backend_ns


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax", "cupy"])
def test_mixture_mutation_kernel_preserves_backend(
    backend: str,
    backend_factory,
) -> None:
    _, particles, _, _ = backend_factory(backend)
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
        ArrayRNG(backend=backend, seed=1357),
    )

    assert proposals.shape == particles.shape
    assert namespace(proposals) is backend_ns
