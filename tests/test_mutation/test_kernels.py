from __future__ import annotations

import numpy as np
import pytest
from orng import RandomGenerator

from sequince.array_api import namespace
from sequince.steps.base import StepContext
from sequince.steps.mutation import (
    AdaptiveMaskedGaussianRandomWalk,
    CallableMutationKernel,
    DifferentialEvolutionKernel,
    GaussianRandomWalk,
    KernelSample,
    MaskedGaussianRandomWalk,
    MetropolisHastingsMutation,
    MixtureMutationKernel,
    evaluate_log_transition_ratio,
    get_complementary_ensemble_kernel,
    get_mutation_kernel,
)


def test_callable_symmetric_kernel(rng) -> None:
    particles = np.arange(6, dtype=float).reshape(3, 2)
    kernel = CallableMutationKernel(
        lambda values, inner_rng: values + 1.0,
        symmetric=True,
    )

    proposal_state = kernel.sample(particles, rng)
    proposals = proposal_state
    ratio = evaluate_log_transition_ratio(kernel, particles, proposal_state)

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
    proposal_state = kernel.sample(particles, rng)
    proposals = proposal_state

    expected = log_density(proposals, particles) - log_density(
        particles, proposals
    )
    np.testing.assert_allclose(
        evaluate_log_transition_ratio(kernel, particles, proposal_state),
        expected,
    )
    np.testing.assert_allclose(
        evaluate_log_transition_ratio(kernel, particles, proposal_state),
        expected,
    )
    _, counts = evaluate_log_transition_ratio(
        kernel, particles, proposal_state, return_counts=True
    )
    assert counts.log_proposal_evaluations == 0


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
    proposal_state = KernelSample(particles + 0.5)
    kernel = GaussianRandomWalk(scale=0.3)

    ratio = evaluate_log_transition_ratio(kernel, particles, proposal_state)

    np.testing.assert_allclose(ratio, 0.0)


def test_get_mutation_kernel_constructs_named_kernel() -> None:
    kernel = get_mutation_kernel("gaussian-random-walk", scale=0.25)

    assert isinstance(kernel, GaussianRandomWalk)
    assert kernel.scale == 0.25


def test_get_mutation_kernel_validates_name() -> None:
    with pytest.raises(ValueError, match="Unknown mutation kernel"):
        get_mutation_kernel("unknown")


def test_get_complementary_ensemble_kernel_constructs_named_kernel() -> None:
    kernel = get_complementary_ensemble_kernel(
        "differential-evolution",
        gamma=0.7,
    )

    assert isinstance(kernel, DifferentialEvolutionKernel)
    assert kernel.gamma == 0.7


def test_kernel_registries_reject_incompatible_kernel_kinds() -> None:
    with pytest.raises(ValueError, match="Unknown mutation kernel"):
        get_mutation_kernel("differential-evolution")
    with pytest.raises(ValueError, match="Unknown complementary-ensemble"):
        get_complementary_ensemble_kernel("gaussian-random-walk")


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

    proposal_state = kernel.sample(particles, rng)
    proposals = proposal_state.particles

    np.testing.assert_allclose(proposals, particles + 1.0)
    np.testing.assert_allclose(
        evaluate_log_transition_ratio(kernel, particles, proposal_state),
        0.0,
    )
    _, counts = evaluate_log_transition_ratio(
        kernel, particles, proposal_state, return_counts=True
    )
    assert counts.log_proposal_evaluations == 0
    assert proposal_state.auxiliary["mixture_component"] == 0
    assert not hasattr(kernel, "selected_kernel")


def test_mixture_kernel_requires_component_metadata() -> None:
    component = CallableMutationKernel(
        lambda values, inner_rng: values,
        symmetric=True,
    )
    kernel = MixtureMutationKernel([component])

    with pytest.raises(KeyError, match="mixture_component"):
        kernel.log_transition_ratio_from_sample(
            np.zeros((3, 1)), KernelSample(np.zeros((3, 1)), {})
        )


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
    xp = case.xp
    particles = case.particles
    backend_ns = namespace(particles)
    step = MetropolisHastingsMutation(
        lambda values, context: -0.5 * xp.sum(values**2, axis=1),
        GaussianRandomWalk(scale=0.1),
        n_steps=2,
        store_chains=True,
    )

    updated, _, output = step(
        case.state(),
        StepContext(t=0),
        case.rng(2468),
    )

    assert updated.particles.shape == particles.shape
    assert namespace(updated.particles) is backend_ns
    assert "mutation_chains" not in updated.metadata
    assert namespace(output["chain"]) is backend_ns


def test_masked_gaussian_changes_only_active_coordinates(rng) -> None:
    particles = np.arange(20, dtype=float).reshape(5, 4)
    kernel = MaskedGaussianRandomWalk(indices=(1, 3), scale=0.2)

    proposal_state = kernel.sample(particles, rng)
    proposals = proposal_state

    assert np.array_equal(proposals[:, (0, 2)], particles[:, (0, 2)])
    assert np.any(proposals[:, (1, 3)] != particles[:, (1, 3)])
    np.testing.assert_allclose(
        evaluate_log_transition_ratio(kernel, particles, proposal_state),
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

    proposal_state = kernel.sample(particles, RandomGenerator(seed=1234))
    proposals = proposal_state

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

    proposal_state = kernel.sample(
        particles,
        case.rng(1357),
    )
    proposals = proposal_state.particles

    assert proposals.shape == particles.shape
    assert namespace(proposals) is backend_ns


def test_mixture_correction_uses_saved_proposal_including_nested_mixtures():
    class MetadataKernel:
        def __init__(self, correction):
            self.correction = correction

        def sample(self, particles, rng):
            return KernelSample(
                particles + 1,
                {"correction": np.full(len(particles), self.correction)},
            )

        def log_transition_ratio_from_sample(self, particles, proposal):
            return proposal.auxiliary["correction"]

    class SelectComponent:
        def __init__(self, index):
            self.index = index

        def choice(self, *args, **kwargs):
            return np.array([self.index])

    inner = MixtureMutationKernel([MetadataKernel(-2), MetadataKernel(-7)])
    outer = MixtureMutationKernel([inner])
    particles = np.zeros((3, 1))
    saved = outer.sample(particles, SelectComponent(0))
    # A later draw must not change the correction of an earlier proposal.
    inner.sample(particles, SelectComponent(1))
    assert not hasattr(inner, "_selected_kernel")
    np.testing.assert_array_equal(
        outer.log_transition_ratio_from_sample(particles, saved), [-2, -2, -2]
    )


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
def test_proposal_state_pytree_roundtrip(backend, backend_factory):
    import optree

    from sequince.array_api import to_numpy
    from sequince.utils.pytrees import OPTREE_NAMESPACE

    case = backend_factory(backend)
    proposal = DifferentialEvolutionKernel().sample(
        case.particles, case.particles, case.rng(77)
    )
    leaves, structure = optree.tree_flatten(
        proposal, namespace=OPTREE_NAMESPACE
    )
    restored = optree.tree_unflatten(structure, leaves)
    assert isinstance(restored, KernelSample)
    assert namespace(restored.particles) is case.xp
    np.testing.assert_array_equal(
        to_numpy(restored.particles), to_numpy(proposal.particles)
    )
    for key in proposal.auxiliary:
        np.testing.assert_array_equal(
            to_numpy(restored.auxiliary[key]),
            to_numpy(proposal.auxiliary[key]),
        )
    if backend == "jax":
        import jax

        compiled = jax.jit(lambda state: state)(proposal)
        np.testing.assert_array_equal(
            to_numpy(compiled.auxiliary["scale_component"]),
            to_numpy(proposal.auxiliary["scale_component"]),
        )


def test_standard_mh_passes_proposal_metadata_to_correction_and_metrics(
    context_factory, rng
):
    from sequince.state import ParticleState

    class Kernel:
        def sample(self, particles, rng):
            return KernelSample(
                particles + 2,
                {"correction": np.array([0.0, -np.inf, 0.0, -np.inf])},
            )

        def log_transition_ratio_from_sample(self, particles, proposal):
            return proposal.auxiliary["correction"]

    class Metric:
        name = "jump_sum"
        needs_transition = True

        def initialize(self, particles, log_weights, context):
            return 0.0

        def update(
            self, state, particles, accepted, context, *, transition=None
        ):
            np.testing.assert_array_equal(
                transition.kernel_auxiliary["correction"],
                [0.0, -np.inf, 0.0, -np.inf],
            )
            return state + np.sum(transition.displacement**2)

        def result(self, state, context):
            return state

    mutation = MetropolisHastingsMutation(
        lambda particles, context: np.zeros(4),
        Kernel(),
        n_steps=2,
        metrics=[Metric()],
    )
    updated, _, output = mutation(
        ParticleState(np.zeros((4, 1))), context_factory(), rng
    )
    np.testing.assert_array_equal(updated.particles[:, 0], [4, 0, 4, 0])
    assert output["metrics"].values["jump_sum"] == 16
    assert output["metrics"].n_accepted == 4


@pytest.mark.parametrize("structured", [False, True])
@pytest.mark.parametrize(
    "kind", ["ordinary", "complementary", "mixture", "callable"]
)
def test_kernel_sample_accepts_arrays_or_state(
    structured, kind, context_factory, rng
):
    from sequince.state import ParticleState
    from sequince.steps.mutation import (
        RedBlueMetropolisHastingsMutation,
        SymmetricComplementaryEnsembleKernel,
        SymmetricMutationKernel,
    )

    def draw(particles):
        proposed = particles + 1
        return (
            KernelSample(proposed, {"tag": np.full(len(particles), 7)})
            if structured
            else proposed
        )

    class Ordinary(SymmetricMutationKernel):
        def sample(self, particles, rng):
            return draw(particles)

        def log_transition_ratio_from_sample(self, particles, proposal):
            assert isinstance(proposal, KernelSample)
            assert bool(proposal.auxiliary) == structured
            return super().log_transition_ratio(particles, proposal.particles)

    class Complementary(SymmetricComplementaryEnsembleKernel):
        def sample(self, particles, complement, rng):
            return draw(particles)

        def log_transition_ratio_from_sample(
            self, particles, proposal, complement
        ):
            assert isinstance(proposal, KernelSample)
            assert bool(proposal.auxiliary) == structured
            return super().log_transition_ratio(
                particles, proposal.particles, complement
            )

    kernel = {
        "ordinary": Ordinary(),
        "complementary": Complementary(),
        "mixture": MixtureMutationKernel([Ordinary()]),
        "callable": CallableMutationKernel(
            lambda particles, rng: particles + 1, symmetric=True
        ),
    }[kind]
    mutation_type = (
        RedBlueMetropolisHastingsMutation
        if kind == "complementary"
        else MetropolisHastingsMutation
    )
    mutation = mutation_type(
        lambda particles, context: np.zeros(len(particles)), kernel
    )
    initial = ParticleState(np.zeros((6, 2)))
    updated, _, _, runtime = mutation.apply_with_mutation_state(
        initial, context_factory(), rng, n_steps=2
    )
    np.testing.assert_array_equal(updated.particles, np.full((6, 2), 2))
    assert not hasattr(runtime, "displacement")
    assert not hasattr(runtime, "kernel_auxiliary")


def test_sample_hook_precedes_array_hook_and_preserves_identity() -> None:
    from sequince.steps.mutation.kernels import _as_kernel_sample

    particles = np.zeros((3, 2))
    wrapped = _as_kernel_sample(particles)
    assert wrapped.particles is particles
    assert wrapped.auxiliary is None
    sample = KernelSample(particles + 1, {"correction": np.arange(3.0)})

    class Kernel:
        def log_transition_ratio(self, start, end):
            raise AssertionError("Array hook must not be called")

        def log_transition_ratio_from_sample(self, start, received):
            assert received is sample
            return received.auxiliary["correction"]

    assert (
        evaluate_log_transition_ratio(Kernel(), particles, sample)
        is sample.auxiliary["correction"]
    )


def test_array_correction_receives_exact_particle_array() -> None:
    particles = np.zeros((3, 2))
    sample = KernelSample(particles + 1, {"unused": np.arange(3)})

    class Kernel:
        def log_transition_ratio(self, start, end):
            assert start is particles
            assert end is sample.particles
            return np.zeros(3)

    np.testing.assert_array_equal(
        evaluate_log_transition_ratio(Kernel(), particles, sample), 0
    )


@pytest.mark.parametrize("complementary", [False, True])
def test_sample_auxiliary_preserves_rng_and_target_calls(
    complementary, context_factory
) -> None:
    from orng import RandomGenerator

    from sequince.state import ParticleState
    from sequince.steps.mutation import RedBlueMetropolisHastingsMutation
    from sequince.steps.mutation.kernels import (
        SymmetricComplementaryEnsembleKernel,
        SymmetricMutationKernel,
    )

    results = []
    for structured in (False, True):
        calls = {"sample": 0, "target": 0, "random": 0, "choice": 0}

        class RNG:
            def __init__(self):
                self.inner = RandomGenerator(seed=7)

            def random(self, *args, **kwargs):
                calls["random"] += 1
                return self.inner.random(*args, **kwargs)

            def choice(self, *args, **kwargs):
                calls["choice"] += 1
                return self.inner.choice(*args, **kwargs)

        def draw(particles):
            calls["sample"] += 1
            end = particles + 0.1
            return KernelSample(end, {"x": particles}) if structured else end

        class Ordinary(SymmetricMutationKernel):
            def sample(self, particles, rng):
                return draw(particles)

        class Complementary(SymmetricComplementaryEnsembleKernel):
            def sample(self, particles, complement, rng):
                return draw(particles)

        def target(particles, context):
            calls["target"] += 1
            return -np.sum(particles**2, axis=1)

        mutation_type = (
            RedBlueMetropolisHastingsMutation
            if complementary
            else MetropolisHastingsMutation
        )
        mutation = mutation_type(
            target, Complementary() if complementary else Ordinary(), n_steps=2
        )
        updated, _, _ = mutation.apply(
            ParticleState(np.zeros((6, 2))), context_factory(), RNG()
        )
        results.append((calls, updated.particles))
    assert results[0][0] == results[1][0]
    assert results[0][0]["sample"] == (4 if complementary else 2)
    assert results[0][0]["random"] == (4 if complementary else 2)
    np.testing.assert_array_equal(results[0][1], results[1][1])


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
@pytest.mark.parametrize(
    "kernel",
    [GaussianRandomWalk(0.2), MaskedGaussianRandomWalk((0,), scale=0.2)],
)
def test_gaussian_samples_remain_arrays(
    kernel, backend, backend_factory
) -> None:
    from sequince.array_api import get_device
    from sequince.steps.mutation.kernels import _as_kernel_sample

    case = backend_factory(backend)
    particles = case.xp.astype(
        case.particles, case.xp.asarray(np.array(0.0)).dtype
    )
    proposal = kernel.sample(particles, case.rng(12))
    assert not isinstance(proposal, KernelSample)
    assert proposal.shape == case.particles.shape
    assert proposal.dtype == particles.dtype
    assert namespace(proposal) is case.xp
    assert get_device(proposal) == get_device(case.particles)
    assert _as_kernel_sample(proposal).particles is proposal
