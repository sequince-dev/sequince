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

    first = kernel.sample(
        active, complementary, RandomGenerator(seed=42)
    ).particles
    replayed = kernel.sample(
        active, complementary, RandomGenerator(seed=42)
    ).particles

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
    ).particles

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
        ).particles


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
        kernel.sample(active, complementary, RandomGenerator(seed=2)).particles


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

    proposals = kernel.sample(
        active, complementary, RandomGenerator(seed=4)
    ).particles
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
    ).particles

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
    ).particles

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
    ).particles

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


@pytest.mark.parametrize("randomization", ["none", "bilby"])
@pytest.mark.parametrize("probability", [0.0, 0.3, 1.0])
def test_singleton_matches_legacy_rng_exactly(
    randomization, probability
) -> None:
    from dataclasses import replace

    active = np.arange(40.0).reshape(20, 2)
    complement = np.arange(16.0).reshape(8, 2)
    kernel = DifferentialEvolutionKernel(
        gamma=0.7,
        gamma_one_probability=probability,
        gamma_randomization=randomization,
    )
    legacy_rng = RandomGenerator(seed=23)
    j, k = _sample_donor_indices(20, 8, legacy_rng)
    gamma = _sample_effective_gamma(
        20,
        0.7,
        probability,
        randomization,
        legacy_rng,
        like=active,
    )
    expected = active + gamma[:, None] * (complement[j] - complement[k])
    for candidate in (kernel, replace(kernel, scale_weights=(1.0,))):
        rng = RandomGenerator(seed=23)
        np.testing.assert_array_equal(
            candidate.sample(active, complement, rng).particles, expected
        )
        # Compare the next draw too, including the early unit-only branch.
        reference = RandomGenerator(seed=23)
        _sample_donor_indices(20, 8, reference)
        _sample_effective_gamma(
            20, 0.7, probability, randomization, reference, like=active
        )
        np.testing.assert_array_equal(rng.random((5,)), reference.random((5,)))


@pytest.mark.parametrize(
    ("multipliers", "weights", "message"),
    [
        ((), (), "nonempty"),
        ((1.0,), (), "matching"),
        ((0.0,), (1.0,), "positive"),
        ((-1.0,), (1.0,), "positive"),
        ((np.nan,), (1.0,), "finite"),
        ((np.inf,), (1.0,), "finite"),
        ((1.0,), (-1.0,), "non-negative"),
        ((1.0,), (np.inf,), "finite"),
        ((1.0,), (np.nan,), "finite"),
        ((1.0,), (0.0,), "positive"),
    ],
)
def test_invalid_scale_mixtures(multipliers, weights, message) -> None:
    with pytest.raises(ValueError, match=message):
        DifferentialEvolutionKernel(
            scale_multipliers=multipliers, scale_weights=weights
        )


def test_mixture_metadata_frequencies_and_independence() -> None:
    kernel = DifferentialEvolutionKernel(
        gamma=0.4,
        gamma_one_probability=0.2,
        scale_multipliers=(0.1, 0.5, 1.0),
        scale_weights=(1.0, 3.0, 6.0),
    )
    active = np.zeros((100_000, 1))
    complement = np.array([[0.0], [1.0]])
    proposal = kernel.sample(active, complement, RandomGenerator(seed=19))
    proposed, metadata = proposal.particles, proposal.auxiliary
    ids = metadata["scale_component"]
    frequencies = np.array([np.mean(ids == i) for i in (-1, 0, 1, 2)])
    np.testing.assert_allclose(
        frequencies, [0.2, 0.08, 0.24, 0.48], atol=0.006
    )
    # Adjacent particles have independent component choices, not one shared draw.
    for i in (-1, 0, 1, 2):
        assert np.mean((ids[::2] == i) & (ids[1::2] == i)) == pytest.approx(
            np.mean(ids == i) ** 2, abs=0.005
        )
    expected_gamma = np.where(
        ids == -1, 1.0, 0.4 * metadata["scale_multiplier"]
    )
    np.testing.assert_allclose(np.abs(proposed[:, 0]), expected_gamma)
    np.testing.assert_array_equal(
        kernel.log_transition_ratio(active, proposed, complement),
        0,
    )
    proposal = kernel.sample(active, complement, RandomGenerator(seed=19))
    replay, replay_metadata = proposal.particles, proposal.auxiliary
    np.testing.assert_array_equal(replay, proposed)
    np.testing.assert_array_equal(replay_metadata["scale_component"], ids)


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
def test_multiscale_backend_metrics(
    backend, backend_factory, context_factory
) -> None:
    from array_api_compat import device

    from sequince.array_api import to_numpy

    case = backend_factory(backend)
    kernel = DifferentialEvolutionKernel(
        scale_multipliers=(0.2, 0.5, 1.0),
        scale_weights=(1.0, 2.0, 1.0),
        gamma_randomization="bilby",
    )
    proposal = kernel.sample(case.particles, case.particles, case.rng(9))
    proposed, metadata = proposal.particles, proposal.auxiliary
    assert proposed.dtype == case.particles.dtype
    assert device(proposed) == device(case.particles)
    assert namespace(metadata["scale_component"]) is case.xp
    mutation = RedBlueMetropolisHastingsMutation(
        lambda x, context: -case.xp.sum(x * x, axis=1),
        kernel,
        n_steps=2,
    )
    _, _, output = mutation(case.state(), context_factory(), case.rng(21))
    result = output["metrics"].values["de_scales"]
    assert (
        np.sum(to_numpy(result["n_attempted"])) == 2 * case.particles.shape[0]
    )
    assert namespace(result["squared_jump_distance"]) is case.xp


def test_scale_diagnostics_exact_attribution(context_factory) -> None:
    from sequince.steps.mutation.metrics.de import DEScaleMetrics

    metric = DEScaleMetrics((0.25, 1.0))
    context = context_factory()
    state = metric.initialize(np.zeros((4, 2)), np.zeros(4), context)
    from sequince.steps.mutation.metrics import MutationTransition

    transition = MutationTransition(
        displacement=np.array(
            [[3.0, 4.0], [0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]
        ),
        kernel_auxiliary={"scale_component": np.array([0, 1, 0, -1])},
    )
    state = metric.update(
        state,
        np.zeros((4, 2)),
        np.array([True, False, True, True]),
        context,
        transition=transition,
    )
    result = metric.result(state, context)
    np.testing.assert_array_equal(result["n_attempted"], [2, 1, 1])
    np.testing.assert_array_equal(result["n_accepted"], [2, 0, 1])
    np.testing.assert_array_equal(result["squared_jump_distance"], [26, 0, 4])
    np.testing.assert_array_equal(result["esjd"], [13, 0, 4])
    np.testing.assert_array_equal(result["component"], [0, 1, -1])


def test_stage_replacement_through_adaptive_wrapper(context_factory) -> None:
    from dataclasses import replace

    from sequince.steps.mutation.adaptive.base import AdaptiveMutationStep

    kernel = DifferentialEvolutionKernel(
        gamma_one_probability=0,
        scale_multipliers=(0.2, 1.0),
        scale_weights=(1.0, 0.0),
    )
    mutation = RedBlueMetropolisHastingsMutation(
        lambda x, context: np.zeros(len(x)), kernel, n_steps=3
    )

    class ReplaceAfterStage(AdaptiveMutationStep):
        def _adapt(self, metrics, context, **kwargs):
            mutation.kernel = replace(kernel, scale_weights=(0.0, 1.0))
            return context

    wrapper = ReplaceAfterStage(mutation)
    state = ParticleState(np.arange(12.0).reshape(6, 2))
    state, context, first = wrapper(
        state, context_factory(), RandomGenerator(seed=4)
    )
    _, _, second = wrapper(state, context, RandomGenerator(seed=8))
    np.testing.assert_array_equal(
        first["metrics"].values["de_scales"]["n_attempted"], [18, 0, 0]
    )
    np.testing.assert_array_equal(
        second["metrics"].values["de_scales"]["n_attempted"], [0, 18, 0]
    )
    assert kernel.scale_weights == (1.0, 0.0)


def test_kernel_snapshot_survives_replacement_during_target_call(
    context_factory,
) -> None:
    from dataclasses import replace

    kernel = DifferentialEvolutionKernel(
        gamma_one_probability=0,
        scale_multipliers=(0.2, 1.0),
        scale_weights=(1.0, 0.0),
    )

    def target(x, context):
        mutation.kernel = replace(kernel, scale_weights=(0.0, 1.0))
        return np.zeros(len(x))

    mutation = RedBlueMetropolisHastingsMutation(target, kernel, n_steps=3)
    _, _, output = mutation(
        ParticleState(np.arange(12.0).reshape(6, 2)),
        context_factory(),
        RandomGenerator(seed=7),
    )
    np.testing.assert_array_equal(
        output["metrics"].values["de_scales"]["n_attempted"], [18, 0, 0]
    )


@pytest.mark.parametrize("randomization", ["none", "bilby"])
def test_multiscale_correlated_gaussian_stationarity(
    context_factory, randomization
) -> None:
    covariance = np.array([[1.0, 0.8], [0.8, 2.0]])
    mean = np.array([1.0, -2.0])
    inverse = np.linalg.inv(covariance)
    particles = np.random.default_rng(50).multivariate_normal(
        mean, covariance, size=6000
    )
    kernel = DifferentialEvolutionKernel(
        scale_multipliers=(0.1, 0.5, 1.0),
        scale_weights=(1.0, 2.0, 3.0),
        gamma_randomization=randomization,
    )
    mutation = RedBlueMetropolisHastingsMutation(
        lambda x, context: -0.5
        * np.sum(((x - mean) @ inverse) * (x - mean), axis=1),
        kernel,
        n_steps=12,
    )
    updated, _, _ = mutation(
        ParticleState(particles), context_factory(), RandomGenerator(seed=51)
    )
    np.testing.assert_allclose(
        np.mean(updated.particles, axis=0), mean, atol=0.06
    )
    np.testing.assert_allclose(
        np.cov(updated.particles.T), covariance, atol=0.10
    )


def test_metadata_correction_and_jump_metrics(context_factory) -> None:
    class CorrectedTestKernel(DifferentialEvolutionKernel):
        def log_transition_ratio_from_sample(
            self, current, proposal, complement
        ):
            proposed, metadata = proposal.particles, proposal.auxiliary
            # A dummy correction proves metadata reaches MH acceptance intact.
            np.testing.assert_array_equal(
                proposed - current,
                (
                    complement[metadata["donor_j"]]
                    - complement[metadata["donor_k"]]
                )
                * metadata["scale_multiplier"][:, None],
            )
            return np.where(metadata["scale_component"] == 0, 0.0, -np.inf)

    kernel = CorrectedTestKernel(
        gamma=1.0, gamma_one_probability=0, scale_multipliers=(0.5, 1.0)
    )
    mutation = RedBlueMetropolisHastingsMutation(
        lambda x, context: np.zeros(len(x)),
        kernel,
        store_chains=True,
        n_steps=3,
    )
    particles = np.arange(40.0).reshape(20, 2)
    _, _, output = mutation(
        ParticleState(particles), context_factory(), RandomGenerator(seed=17)
    )
    metrics = output["metrics"].values["de_scales"]
    assert metrics["n_attempted"][1] > 0
    assert metrics["n_accepted"][1] == 0
    assert metrics["squared_jump_distance"][1] == 0
    assert metrics["n_accepted"][0] == metrics["n_attempted"][0]
    chain = np.concatenate([particles[None], output["chain"]], axis=0)
    assert metrics["squared_jump_distance"].sum() == pytest.approx(
        np.sum(np.diff(chain, axis=0) ** 2)
    )


def test_unit_only_mixture_ignores_all_factors() -> None:
    kernel = DifferentialEvolutionKernel(
        gamma=0.3,
        gamma_one_probability=1.0,
        gamma_randomization="bilby",
        scale_multipliers=(0.1, 0.5),
        scale_weights=(0.0, 1.0),
    )
    rng = _controlled_rng([1, 1], [0, 0])
    proposal = kernel.sample(np.zeros((2, 1)), np.array([[0.0], [2.0]]), rng)
    proposed, metadata = proposal.particles, proposal.auxiliary
    np.testing.assert_array_equal(proposed, 2.0)
    np.testing.assert_array_equal(metadata["scale_component"], -1)
    np.testing.assert_array_equal(metadata["scale_multiplier"], 1.0)


def test_bilby_factors_multiply_selected_scale() -> None:
    kernel = DifferentialEvolutionKernel(
        gamma=0.4,
        gamma_one_probability=0,
        gamma_randomization="bilby",
        scale_multipliers=(0.25, 0.5),
    )
    rng = StubRandomGenerator(
        choices=[np.array([1, 1]), np.array([0, 0]), np.array([0, 1])],
        gamma=np.array([0.5, 1.5]),
    )
    proposal = kernel.sample(np.zeros((2, 1)), np.array([[0.0], [2.0]]), rng)
    proposed, metadata = proposal.particles, proposal.auxiliary
    np.testing.assert_allclose(
        proposed[:, 0], 0.4 * np.array([0.5, 1.5]) * np.array([0.25, 0.5]) * 2
    )
    np.testing.assert_array_equal(metadata["scale_component"], [0, 1])


def test_scale_configuration_copies_inputs_and_normalizes() -> None:
    from dataclasses import FrozenInstanceError

    multipliers = [0.5, 1.0]
    weights = [1e308, 1e308]
    kernel = DifferentialEvolutionKernel(
        scale_multipliers=multipliers, scale_weights=weights
    )
    multipliers[0] = -1
    weights[0] = 0
    assert kernel.scale_multipliers == (0.5, 1.0)
    assert kernel.scale_weights == (0.5, 0.5)
    with pytest.raises(FrozenInstanceError):
        kernel.scale_weights = (1.0, 0.0)


def test_scale_metrics_accumulate_across_segments_and_empty_stage(
    context_factory,
) -> None:
    from sequince.steps.mutation import DifferentialEvolutionMutation

    mutation = DifferentialEvolutionMutation(
        lambda x, context: np.zeros(len(x)),
        gamma_one_probability=0,
        scale_multipliers=(0.5, 1.0),
        scale_weights=(0.0, 1.0),
    )
    particles = ParticleState(np.arange(12.0).reshape(6, 2))
    context = context_factory()
    _, _, empty = mutation.apply_with_n_steps(
        particles, context, RandomGenerator(seed=3), n_steps=0
    )
    np.testing.assert_array_equal(
        empty["metrics"].values["de_scales"]["n_attempted"], 0
    )
    state, context, _, metrics_state = mutation.apply_with_mutation_state(
        particles, context, RandomGenerator(seed=3), n_steps=2
    )
    _, context, _, metrics_state = mutation.apply_with_mutation_state(
        state,
        context,
        RandomGenerator(seed=4),
        n_steps=3,
        mutation_state=metrics_state,
    )
    result = mutation.metrics.result(metrics_state, context)
    np.testing.assert_array_equal(
        result.values["de_scales"]["n_attempted"], [0, 30, 0]
    )
    np.testing.assert_array_equal(
        result.values["de_scales"]["n_accepted"], [0, 30, 0]
    )
    assert result.n_steps == 5


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_mixture_preserves_numpy_precision(dtype) -> None:
    active = np.zeros((10, 2), dtype=dtype)
    kernel = DifferentialEvolutionKernel(scale_multipliers=(0.25, 1.0))
    proposal = kernel.sample(
        active, np.ones((3, 2), dtype=dtype), RandomGenerator(seed=10)
    )
    proposed, metadata = proposal.particles, proposal.auxiliary
    assert proposed.dtype == dtype
    assert metadata["scale_multiplier"].dtype == dtype


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
def test_transition_state_is_in_particle_order(
    backend, backend_factory, context_factory
) -> None:
    from sequince.array_api import to_numpy

    case = backend_factory(backend)

    class CaptureTransitions:
        name = "transitions"

        def initialize(self, particles, log_weights, context):
            return particles, ()

        def update(
            self, state, particles, accepted, context, *, transition=None
        ):
            previous, history = state
            assert transition is not None
            displacement = transition.displacement
            np.testing.assert_allclose(
                to_numpy(displacement), to_numpy(particles - previous)
            )
            np.testing.assert_array_equal(
                to_numpy(displacement)[~to_numpy(accepted)], 0
            )
            return particles, (
                *history,
                (displacement, transition.kernel_auxiliary, accepted),
            )

        def result(self, state, context):
            return state[1]

    # Unequal halves expose dropped particles and ordering mistakes.
    initial = ParticleState(
        case.xp.concat((case.particles, case.particles + 0.3))[:-1]
    )
    mutation = RedBlueMetropolisHastingsMutation(
        lambda x, context: -case.xp.sum(x * x, axis=1),
        DifferentialEvolutionKernel(
            scale_multipliers=(0.2, 1.0), gamma_one_probability=0.3
        ),
        n_steps=4,
        metrics=[CaptureTransitions()],
    )
    final, context, _, runtime = mutation.apply_with_mutation_state(
        initial, context_factory(), case.rng(21), n_steps=4
    )
    results = mutation.metrics.result(runtime, context)
    history = results.values["transitions"]
    assert len(history) == 4
    expected_jumps = np.zeros(3)
    expected_attempts = np.zeros(3)
    expected_accepted = np.zeros(3)
    for displacement, metadata, accepted in history:
        components = to_numpy(metadata["scale_component"])
        jumps = np.sum(to_numpy(displacement) ** 2, axis=1)
        for index, component in enumerate((0, 1, -1)):
            selected = components == component
            expected_attempts[index] += np.sum(selected)
            expected_accepted[index] += np.sum(to_numpy(accepted)[selected])
            expected_jumps[index] += np.sum(jumps[selected])
    scales = results.values["de_scales"]
    np.testing.assert_allclose(
        to_numpy(scales["squared_jump_distance"]), expected_jumps, rtol=1e-6
    )
    np.testing.assert_array_equal(
        to_numpy(scales["n_attempted"]), expected_attempts
    )
    np.testing.assert_array_equal(
        to_numpy(scales["n_accepted"]), expected_accepted
    )
    _, _, _, prepared = mutation.apply_with_mutation_state(
        final, context, case.rng(22), n_steps=0, mutation_state=runtime
    )
    assert not hasattr(prepared, "displacement")
    assert not hasattr(prepared, "kernel_auxiliary")
    assert prepared.n_steps == 4
    # Preparing another segment does not mutate the previous state.
    assert not hasattr(runtime, "displacement")


def test_transition_metrics_fixed_jax_loop(context_factory) -> None:
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    from sequince.steps.mutation.metrics import (
        MutationMetrics,
        MutationTransition,
    )
    from sequince.steps.mutation.metrics.de import DEScaleMetrics

    particles = jnp.zeros((4, 2))
    context = context_factory()
    metrics = MutationMetrics([DEScaleMetrics((1.0,))])
    initial = metrics.initialize(particles, jnp.zeros(4), context)
    accepted = jnp.array([True, False, True, False])
    transition = MutationTransition(
        jnp.where(accepted[:, None], jnp.ones_like(particles), 0),
        {"scale_component": jnp.zeros(4, dtype=jnp.int32)},
    )

    @jax.jit
    def accumulate(state):
        return jax.lax.fori_loop(
            0,
            3,
            lambda _, current: metrics.update(
                current, particles, accepted, context, transition=transition
            ),
            state,
        )

    final = accumulate(initial)
    result = metrics.result(final, context).values["de_scales"]
    np.testing.assert_array_equal(result["n_attempted"], [12, 0])
    np.testing.assert_array_equal(result["n_accepted"], [6, 0])
    np.testing.assert_array_equal(result["squared_jump_distance"], [12, 0])
    assert not hasattr(final, "displacement")
    assert jax.tree_util.tree_structure(final) == jax.tree_util.tree_structure(
        initial
    )
