from __future__ import annotations

from collections.abc import Sequence
from dataclasses import FrozenInstanceError
from typing import Any

import numpy as np
import optree
import pytest
from array_api_compat import device as get_device
from orng import RandomGenerator

from sequince import ParticleState
from sequince.array_api import namespace, to_numpy
from sequince.steps import StepContext
from sequince.steps.mutation.kernels import (
    CallableMutationKernel,
    DifferentialEvolutionKernel,
    IndependentProposalKernel,
    MixtureComplementaryEnsembleKernel,
    evaluate_log_transition_ratio,
    get_complementary_ensemble_kernel,
)
from sequince.steps.mutation.metropolis import (
    RedBlueMetropolisHastingsMutation,
)
from sequince.typing import Array
from sequince.utils.random import resolve_rng


class NormalProposal:
    def sample(
        self,
        n_samples: int,
        context: StepContext | None = None,
        *,
        rng: RandomGenerator | int | None = None,
        xp: Any | None = None,
    ) -> Array:
        return resolve_rng(rng).normal(size=(n_samples, 2)) * 2

    def log_prob(
        self, particles: Array, context: StepContext | None = None
    ) -> Array:
        xp = namespace(particles)
        return -xp.sum(particles**2, axis=-1) / 8

    def sample_and_log_prob(
        self,
        n_samples: int,
        context: StepContext | None = None,
        *,
        rng: RandomGenerator | int | None = None,
        xp: Any | None = None,
    ) -> tuple[Array, Array]:
        values = self.sample(n_samples, rng=rng, xp=xp)
        return values, self.log_prob(values)


def mixture(
    weights: Sequence[float] = (0.5, 0.5),
) -> MixtureComplementaryEnsembleKernel:
    return MixtureComplementaryEnsembleKernel(
        (
            DifferentialEvolutionKernel(),
            IndependentProposalKernel(NormalProposal()),
        ),
        weights,
    )


def test_independence_backend(backend, backend_factory) -> None:
    case = backend_factory(backend)
    particles = case.particles
    kernel = IndependentProposalKernel(NormalProposal())
    sample = kernel.sample(particles, case.rng(5))
    ratio = evaluate_log_transition_ratio(kernel, particles, sample)
    assert sample.particles.shape == particles.shape
    assert sample.particles.dtype == particles.dtype
    assert get_device(sample.particles) == get_device(particles)
    assert case.matches_namespace(sample.particles)
    np.testing.assert_allclose(
        to_numpy(ratio),
        to_numpy(
            kernel.proposal.log_prob(particles)
            - kernel.proposal.log_prob(sample.particles)
        ),
        rtol=1e-6,
        atol=1e-6,
    )
    with pytest.raises(ValueError, match="shape"):
        kernel.log_transition_ratio(particles, sample.particles[:1])


def test_support_and_validation() -> None:
    class Bounded(NormalProposal):
        def log_prob(self, particles, context=None):
            return np.where(particles[:, 0] >= 0, -particles[:, 0], -np.inf)

    kernel = IndependentProposalKernel(Bounded())
    ratio = kernel.log_transition_ratio(
        np.array([[-1.0, 0], [1, 0]]), np.array([[1.0, 0], [-1, 0]])
    )
    np.testing.assert_array_equal(ratio, [-np.inf, np.inf])
    with pytest.raises(ValueError, match="shape"):
        kernel.sample(np.zeros((3, 1)), RandomGenerator(seed=5))

    class BadDensity(NormalProposal):
        def log_prob(self, particles, context=None):
            return np.zeros((len(particles), 1))

    with pytest.raises(ValueError, match="densities"):
        IndependentProposalKernel(BadDensity()).sample(
            np.zeros((3, 2)), RandomGenerator(seed=5)
        )


@pytest.mark.parametrize("weights", [(1.0, 0.0), (0.0, 1.0), (0.4, 0.6)])
def test_dispatch_and_auxiliary_backend(
    backend, backend_factory, weights
) -> None:
    case = backend_factory(backend)
    kernel = mixture(weights)
    active = case.particles[:2]
    complement = case.particles[2:]
    sample = kernel.sample(active, complement, case.rng(22))
    second = kernel.sample(case.particles[2:], active, case.rng(33))
    assert optree.tree_structure(sample.auxiliary) == optree.tree_structure(
        second.auxiliary
    )
    assert sample.particles.dtype == active.dtype
    assert case.matches_namespace(sample.particles)
    component = to_numpy(sample.auxiliary["mixture_component"])
    ratio = to_numpy(
        evaluate_log_transition_ratio(kernel, active, sample, complement)
    )
    expected = to_numpy(
        NormalProposal().log_prob(active)
        - NormalProposal().log_prob(sample.particles)
    )
    np.testing.assert_allclose(
        ratio, np.where(component == 1, expected, 0), rtol=1e-6, atol=1e-6
    )
    for i, child in enumerate(sample.auxiliary["children"]):
        np.testing.assert_array_equal(
            to_numpy(child["selected"]), component == i
        )
        for leaf in optree.tree_leaves(child["diagnostics"]):
            assert leaf.shape == (2,)
            assert np.all(to_numpy(leaf)[component != i] == 0)
    if weights[1] == 0:
        diagnostics = sample.auxiliary["children"][0]["diagnostics"]
        donors = to_numpy(complement)
        gamma = 2.38 / 2
        multiplier = np.where(
            to_numpy(diagnostics["scale_component"]) == -1, 1.0, gamma
        )
        expected = to_numpy(active) + multiplier[:, None] * (
            donors[to_numpy(diagnostics["donor_j"])]
            - donors[to_numpy(diagnostics["donor_k"])]
        )
        np.testing.assert_allclose(
            to_numpy(sample.particles), expected, atol=1e-6
        )


def test_selection_and_order() -> None:
    seen = []

    def shift(particles, rng):
        seen.append(particles.copy())
        return particles + 3

    child = CallableMutationKernel(
        shift, log_transition_ratio_fn=lambda a, b: a[:, 0]
    )
    kernel = MixtureComplementaryEnsembleKernel((child, child), (0.2, 0.8))
    particles = np.arange(40000.0).reshape(-1, 2)
    sample = kernel.sample(particles, particles[:2], RandomGenerator(seed=54))
    ids = sample.auxiliary["mixture_component"]
    assert abs(np.mean(ids == 0) - 0.2) < 0.015
    assert abs(np.mean((ids[:-1] == 0) & (ids[1:] == 0)) - 0.04) < 0.01
    np.testing.assert_array_equal(sample.particles, particles + 3)
    np.testing.assert_array_equal(
        sample.auxiliary["correction"], particles[:, 0]
    )
    for i in range(2):
        np.testing.assert_array_equal(seen[i], particles[ids == i])


@pytest.mark.parametrize(
    "weights", [(), (0, 0), (-1, 2), (np.nan, 1), (np.inf, 1), (1,)]
)
def test_invalid_weights(weights) -> None:
    with pytest.raises(ValueError):
        mixture(weights)


def test_immutable_and_factory() -> None:
    kernel = mixture()
    with pytest.raises(FrozenInstanceError):
        kernel.weights = (1, 0)
    built = get_complementary_ensemble_kernel(
        "mixture",
        kernels=[
            "differential-evolution",
            {
                "kernel": "independent-proposal",
                "kernel_kwargs": {"proposal": NormalProposal()},
            },
        ],
        weights=[1, 2],
    )
    assert built.weights == (1 / 3, 2 / 3)


def test_known_target(context_factory) -> None:
    rng = RandomGenerator(seed=44)
    state = ParticleState(rng.normal(size=(1500, 2)) * 4)
    step = RedBlueMetropolisHastingsMutation(
        lambda x: -np.sum(x * x, axis=-1) / 2,
        mixture(),
        n_steps=100,
    )
    state, _, _ = step(state, context_factory(), rng)
    np.testing.assert_allclose(np.mean(state.particles, axis=0), 0, atol=0.10)
    np.testing.assert_allclose(np.var(state.particles, axis=0), 1, atol=0.12)


def test_sample_aware_complement_and_custom_diagnostics(
    context_factory,
) -> None:
    from sequince.steps.mutation import (
        ComplementaryEnsembleKernel,
        KernelSample,
        MixtureComponentMetrics,
    )

    complements = []

    class ShiftComplement(ComplementaryEnsembleKernel):
        def sample(self, particles, complementary_particles, rng):
            complements.append(complementary_particles.copy())
            return KernelSample(particles + 1, {"start": particles.copy()})

        def log_transition_ratio_from_sample(self, start, sample, complement):
            # This verifies that the cached child sample and complement reach
            # the sample-aware correction, without a second proposal draw.
            np.testing.assert_array_equal(sample.auxiliary["start"], start)
            np.testing.assert_array_equal(complement, complements[-1])
            return np.full(len(start), -np.inf)

    kernel = MixtureComplementaryEnsembleKernel(
        (ShiftComplement(), IndependentProposalKernel(NormalProposal())),
        (1.0, 0.0),
        auxiliary_templates=(
            {"start": np.zeros(2)},
            {"log_proposal": np.asarray(0.0)},
        ),
    )
    original = np.arange(14.0).reshape(7, 2)
    step = RedBlueMetropolisHastingsMutation(
        lambda x: np.zeros(len(x)),
        kernel,
        metrics=[MixtureComponentMetrics(2)],
    )
    state, _, output = step(
        ParticleState(original), context_factory(), RandomGenerator(seed=20)
    )
    # Every proposal is rejected; the second half must see the original first
    # half as its complement, with no in-place mutation of either array.
    np.testing.assert_array_equal(state.particles, original)
    np.testing.assert_array_equal(
        np.sort(np.concatenate(complements)[:, 0]), original[:, 0]
    )
    np.testing.assert_array_equal(
        output["metrics"].values["mixture_components"]["n_attempted"], [7, 0]
    )
    np.testing.assert_array_equal(
        output["metrics"].values["mixture_components"][
            "squared_jump_distance"
        ],
        [0, 0],
    )


def test_child_shape_and_template_validation() -> None:
    from sequince.steps.mutation import KernelSample

    class DiagnosticChild:
        def sample(self, particles, rng):
            return KernelSample(
                particles, {"value": np.zeros((len(particles), 2))}
            )

        def log_transition_ratio(self, start, end):
            return np.zeros(len(start))

    particles = np.zeros((4, 2))
    for templates in (None, ({"value": np.asarray(0.0)},)):
        kernel = MixtureComplementaryEnsembleKernel(
            (DiagnosticChild(),), auxiliary_templates=templates
        )
        with pytest.raises(ValueError, match="template"):
            kernel.sample(particles, particles, RandomGenerator(seed=2))
    for child in (
        CallableMutationKernel(lambda x, rng: x[:1], symmetric=True),
        CallableMutationKernel(
            lambda x, rng: x,
            log_transition_ratio_fn=lambda a, b: np.zeros((len(a), 1)),
        ),
    ):
        with pytest.raises(ValueError, match="shape"):
            MixtureComplementaryEnsembleKernel((child,)).sample(
                particles, particles, RandomGenerator(seed=2)
            )


@pytest.mark.parametrize("shape", [(5,), (1, 2), (5, 2, 3)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_independence_particle_shapes(shape, dtype) -> None:
    class ShapedProposal(NormalProposal):
        def sample(self, n_samples, context=None, *, rng=None, xp=None):
            return rng.normal(size=(n_samples, *shape[1:]))

        def log_prob(self, particles, context=None):
            return (
                -np.sum(
                    np.reshape(particles, (len(particles), -1)) ** 2, axis=1
                )
                / 2
            )

    particles = np.ones(shape, dtype=dtype)
    kernel = IndependentProposalKernel(ShapedProposal())
    sample = kernel.sample(particles, RandomGenerator(seed=3))
    assert sample.particles.shape == shape
    assert sample.particles.dtype == dtype
    ratio = evaluate_log_transition_ratio(kernel, particles, sample)
    assert ratio.shape == (shape[0],)
    assert ratio.dtype == dtype


def test_factory_errors() -> None:
    with pytest.raises(ValueError, match="kernels"):
        get_complementary_ensemble_kernel("mixture")
    with pytest.raises(ValueError, match="kernel"):
        get_complementary_ensemble_kernel("mixture", kernels=[{}])


def test_cached_density_and_unselected_component() -> None:
    class CountedProposal(NormalProposal):
        def __init__(self) -> None:
            self.calls = []

        def log_prob(self, particles, context=None):
            self.calls.append(particles)
            return super().log_prob(particles, context)

    proposal = CountedProposal()
    independent = IndependentProposalKernel(proposal)
    particles = np.ones((10, 2))
    rng = RandomGenerator(seed=22)
    sample = independent.sample(particles, rng)
    evaluate_log_transition_ratio(independent, particles, sample)
    assert len(proposal.calls) == 2  # Sampled density plus current density.
    assert proposal.calls[-1] is particles
    proposal.calls.clear()
    MixtureComplementaryEnsembleKernel(
        (DifferentialEvolutionKernel(), independent), (1.0, 0.0)
    ).sample(particles, particles, rng)
    assert proposal.calls == []
