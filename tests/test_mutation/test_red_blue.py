from __future__ import annotations

import numpy as np
import pytest
from orng import RandomGenerator

from sequince.array_api import namespace
from sequince.posterior import TemperedPosterior
from sequince.state import ParticleState
from sequince.steps import RedBlueMetropolisHastingsMutation
from sequince.steps.mutation import (
    ComplementaryEnsembleKernel,
    KernelSample,
    SymmetricComplementaryEnsembleKernel,
)
from sequince.steps.mutation.metropolis import (
    RedBlueMetropolisHastingsMutation as DirectRedBlueMutation,
)
from sequince.transforms import ParticleTransform


class ShiftKernel(SymmetricComplementaryEnsembleKernel):
    def __init__(self, amount: float = 1.0) -> None:
        self.amount = amount

    def sample(self, active, complementary, rng):
        del complementary, rng
        return KernelSample(active + self.amount)


class GaussianKernel(SymmetricComplementaryEnsembleKernel):
    def __init__(self, scale: float) -> None:
        self.scale = scale

    def sample(self, active, complementary, rng):
        del complementary
        return KernelSample(
            active
            + rng.normal(
                loc=0.0,
                scale=self.scale,
                size=active.shape,
            )
        )


class AcceptFirstHalfKernel(ComplementaryEnsembleKernel):
    def __init__(self, amount: float) -> None:
        self.amount = amount
        self.active_groups: list[np.ndarray] = []

    def sample(self, active, complementary, rng):
        del complementary, rng
        self.active_groups.append(np.asarray(active).copy())
        return KernelSample(active + self.amount)

    def log_transition_ratio(self, current, proposed, complementary):
        del proposed, complementary
        if len(self.active_groups) == 1:
            return np.zeros(current.shape[0])
        return np.full(current.shape[0], -np.inf)


def _shift_kernel(amount: float = 1.0) -> ShiftKernel:
    return ShiftKernel(amount)


def test_red_blue_is_publicly_importable() -> None:
    assert RedBlueMetropolisHastingsMutation is DirectRedBlueMutation


@pytest.mark.parametrize("n_particles", [2, 3, 4, 5, 10])
def test_split_is_a_random_partition(n_particles: int) -> None:
    first = RedBlueMetropolisHastingsMutation._split_indices(
        n_particles,
        RandomGenerator(backend="numpy", seed=41),
    )
    replayed = RedBlueMetropolisHastingsMutation._split_indices(
        n_particles,
        RandomGenerator(backend="numpy", seed=41),
    )

    np.testing.assert_array_equal(first[0], replayed[0])
    np.testing.assert_array_equal(first[1], replayed[1])
    assert first[0].shape[0] == n_particles // 2
    assert first[1].shape[0] == n_particles - n_particles // 2
    assert set(np.asarray(first[0])).isdisjoint(np.asarray(first[1]))
    assert sorted(np.concatenate(first).tolist()) == list(range(n_particles))


def test_persistent_red_blue_kernel_receives_each_frozen_complement(
    flat_log_prob,
    context_factory,
) -> None:
    class RecordingKernel(SymmetricComplementaryEnsembleKernel):
        def __init__(self) -> None:
            self.calls: list[tuple[np.ndarray, np.ndarray]] = []

        def sample(self, active, complementary, rng):
            del rng
            self.calls.append(
                (np.asarray(active).copy(), np.asarray(complementary).copy())
            )
            return KernelSample(active + 10.0)

    particles = np.arange(6.0).reshape(6, 1)
    kernel = RecordingKernel()
    mutation = RedBlueMetropolisHastingsMutation(flat_log_prob, kernel)

    updated, _, _ = mutation(
        ParticleState(particles),
        context_factory(),
        RandomGenerator(backend="numpy", seed=12),
    )

    first_active, first_complement = kernel.calls[0]
    second_active, second_complement = kernel.calls[1]
    np.testing.assert_array_equal(second_active, first_complement)
    np.testing.assert_array_equal(second_complement, first_active + 10.0)
    np.testing.assert_array_equal(updated.particles, particles + 10.0)
    assert mutation._complementary_ensemble_kernel is kernel  # noqa: SLF001


def test_persistent_red_blue_kernel_transition_ratio_is_applied(
    flat_log_prob,
    context_factory,
) -> None:
    class RejectingKernel(ComplementaryEnsembleKernel):
        def sample(self, active, complementary, rng):
            del complementary, rng
            return KernelSample(active + 10.0)

        def log_transition_ratio(self, current, proposed, complementary):
            del proposed, complementary
            return np.full(current.shape[0], -np.inf)

    particles = np.arange(6.0).reshape(6, 1)
    mutation = RedBlueMetropolisHastingsMutation(
        flat_log_prob,
        RejectingKernel(),
    )

    updated, returned_context, output = mutation(
        ParticleState(particles),
        context_factory(),
        RandomGenerator(backend="numpy", seed=12),
    )

    np.testing.assert_array_equal(updated.particles, particles)
    assert output["metrics"].acceptance == 0.0
    assert returned_context.evaluation_counts.log_target_evaluations == 12
    assert returned_context.evaluation_counts.log_proposal_evaluations == 0


def test_fixed_rng_replays_stochastic_red_blue_result(context_factory) -> None:
    particles = np.arange(12.0).reshape(6, 2) / 10.0

    def make_mutation():
        return RedBlueMetropolisHastingsMutation(
            lambda values, context: -0.5 * np.sum(values**2, axis=1),
            GaussianKernel(scale=0.3),
            n_steps=3,
            store_chains=True,
        )

    first, _, first_output = make_mutation()(
        ParticleState(particles),
        context_factory(),
        RandomGenerator(backend="numpy", seed=55),
    )
    replayed, _, replayed_output = make_mutation()(
        ParticleState(particles),
        context_factory(),
        RandomGenerator(backend="numpy", seed=55),
    )

    np.testing.assert_array_equal(first.particles, replayed.particles)
    np.testing.assert_array_equal(
        first_output["chain"], replayed_output["chain"]
    )
    assert (
        first_output["acceptance_by_half"]
        == replayed_output["acceptance_by_half"]
    )


def test_only_active_particles_are_evaluated(context_factory) -> None:
    evaluation_sizes: list[int] = []

    def target(particles, context):
        del context
        evaluation_sizes.append(particles.shape[0])
        return np.zeros(particles.shape[0])

    mutation = RedBlueMetropolisHastingsMutation(
        target,
        _shift_kernel(),
        n_steps=2,
    )
    _, returned_context, _ = mutation(
        ParticleState(np.zeros((5, 2))),
        context_factory(),
        RandomGenerator(backend="numpy", seed=3),
    )

    assert evaluation_sizes == [5, 2, 3, 2, 3]
    assert sum(evaluation_sizes[1:3]) == 5
    assert sum(evaluation_sizes[3:]) == 5
    assert returned_context.evaluation_counts.log_target_evaluations == 15
    assert returned_context.evaluation_counts.log_proposal_evaluations == 0


def test_acceptance_updates_only_the_active_half(
    flat_log_prob, context_factory
) -> None:
    particles = np.arange(6.0).reshape(6, 1)
    kernel = AcceptFirstHalfKernel(10.0)
    mutation = RedBlueMetropolisHastingsMutation(flat_log_prob, kernel)
    updated, _, output = mutation(
        ParticleState(particles),
        context_factory(),
        RandomGenerator(backend="numpy", seed=8),
    )

    expected = particles.copy()
    first_indices = kernel.active_groups[0][:, 0].astype(int)
    expected[first_indices] += 10.0
    np.testing.assert_array_equal(updated.particles, expected)
    assert output["metrics"].acceptance == 0.5
    assert output["acceptance_by_step"] == (0.5,)
    assert output["acceptance_by_half"] == ((1.0, 0.0),)


def test_cached_target_evaluation_is_merged(context_factory) -> None:
    posterior = TemperedPosterior(
        log_prior=lambda particles, context: np.zeros(particles.shape[0]),
        log_likelihood=lambda particles, context: particles[:, 0],
    )
    context = context_factory(metadata={"temperature": 0.0})
    state = ParticleState(np.arange(5.0).reshape(5, 1))
    state = state.with_updates(
        evaluation=posterior.evaluate(state.particles, context)
    )
    mutation = RedBlueMetropolisHastingsMutation(
        posterior,
        AcceptFirstHalfKernel(2.0),
    )

    updated, _, _ = mutation(
        state,
        context,
        RandomGenerator(backend="numpy", seed=9),
    )

    assert updated.evaluation is not None
    np.testing.assert_array_equal(
        updated.evaluation["log_likelihood"],
        updated.particles[:, 0],
    )


def test_cached_target_cost_is_one_population_per_sweep(
    context_factory,
) -> None:
    calls: dict[str, list[int]] = {"prior": [], "likelihood": []}

    def log_prior(particles, context):
        del context
        calls["prior"].append(particles.shape[0])
        return np.zeros(particles.shape[0])

    def log_likelihood(particles, context):
        del context
        calls["likelihood"].append(particles.shape[0])
        return np.zeros(particles.shape[0])

    posterior = TemperedPosterior(log_prior, log_likelihood)
    context = context_factory(metadata={"temperature": 0.5})
    state = ParticleState(np.zeros((5, 2)))
    state = state.with_updates(
        evaluation=posterior.evaluate(state.particles, context)
    )
    calls = {"prior": [], "likelihood": []}
    mutation = RedBlueMetropolisHastingsMutation(
        posterior,
        _shift_kernel(),
        n_steps=2,
    )

    _, returned_context, _ = mutation(
        state,
        context,
        RandomGenerator(backend="numpy", seed=21),
    )

    assert calls["prior"] == [2, 3, 2, 3]
    assert calls["likelihood"] == [2, 3, 2, 3]
    assert sum(calls["prior"]) == 2 * state.particles.shape[0]
    assert sum(calls["likelihood"]) == 2 * state.particles.shape[0]
    assert returned_context.evaluation_counts.log_target_evaluations == 10
    assert returned_context.evaluation_counts.log_likelihood_evaluations == 10
    assert returned_context.evaluation_counts.log_proposal_evaluations == 0


class StructuredAuxiliaryTransform(ParticleTransform):
    def __init__(self) -> None:
        self.fit_calls = 0
        self.forward_calls = 0
        self.inverse_batch_sizes: list[int] = []

    def fit(self, particles, context=None, *, rng=None):
        del particles, context, rng
        self.fit_calls += 1
        return self

    def forward(self, particles, context=None, *, rng=None):
        del context, rng
        self.forward_calls += 1
        values = np.asarray(particles)
        auxiliary = {
            "nested": {
                "offset": np.arange(values.shape[0], dtype=values.dtype)[
                    :, None
                ]
            },
            "shared": np.asarray(1.0),
        }
        return (
            values - auxiliary["nested"]["offset"],
            np.zeros(values.shape[0]),
            auxiliary,
        )

    def inverse(self, values, context=None, *, auxiliary=None, rng=None):
        del context, rng
        assert auxiliary is not None
        offsets = auxiliary["nested"]["offset"]
        self.inverse_batch_sizes.append(offsets.shape[0])
        values = np.asarray(values) + offsets
        return values, np.zeros(values.shape[0])


def test_structured_transform_auxiliary_is_sliced(context_factory) -> None:
    transform = StructuredAuxiliaryTransform()
    evaluated = []

    def target(particles, context):
        evaluated.append(particles.copy())
        return np.zeros(len(particles))

    mutation = RedBlueMetropolisHastingsMutation(
        target, _shift_kernel(), transform=transform
    )
    particles = 10 * np.arange(5, dtype=np.float32)[:, None]

    updated, _, _ = mutation(
        ParticleState(particles),
        context_factory(),
        RandomGenerator(backend="numpy", seed=5),
    )

    np.testing.assert_array_equal(evaluated[0], particles)
    proposals = np.concatenate(evaluated[1:])[:, 0]
    np.testing.assert_array_equal(np.sort(proposals), particles[:, 0] + 1)
    np.testing.assert_array_equal(updated.particles, particles + 1)
    assert updated.particles.dtype == particles.dtype


def test_transform_is_prepared_once_per_apply(
    flat_log_prob, context_factory
) -> None:
    transform = StructuredAuxiliaryTransform()
    mutation = RedBlueMetropolisHastingsMutation(
        flat_log_prob,
        _shift_kernel(),
        transform=transform,
        n_steps=3,
    )

    mutation(
        ParticleState(np.zeros((6, 1))),
        context_factory(),
        RandomGenerator(backend="numpy", seed=13),
    )

    assert transform.fit_calls == 1
    assert transform.forward_calls == 1


def test_store_chains_records_complete_sweeps(
    flat_log_prob, context_factory
) -> None:
    particles = np.zeros((5, 2), dtype=np.float32)
    mutation = RedBlueMetropolisHastingsMutation(
        flat_log_prob,
        _shift_kernel(),
        n_steps=3,
        store_chains=True,
    )

    updated, _, output = mutation(
        ParticleState(particles),
        context_factory(),
        RandomGenerator(backend="numpy", seed=2),
    )

    assert output["chain"].shape == (3, *particles.shape)
    np.testing.assert_array_equal(output["chain"][-1], updated.particles)
    np.testing.assert_array_equal(updated.particles, particles + 3.0)


def test_shape_dtype_namespace_and_particle_count_are_preserved(
    backend,
    backend_factory,
    context_factory,
) -> None:
    case = backend_factory(backend)

    def target(particles, context):
        del context
        xp = namespace(particles)
        return xp.zeros(
            (particles.shape[0],),
            dtype=particles.dtype,
            device=case.particles.device,
        )

    mutation = RedBlueMetropolisHastingsMutation(
        target,
        _shift_kernel(),
        store_chains=True,
    )
    updated, _, output = mutation(
        case.state(),
        context_factory(),
        case.rng(20),
    )

    assert updated.particles.shape == case.particles.shape
    assert updated.particles.dtype == case.particles.dtype
    assert case.matches_namespace(updated.particles)
    assert output["chain"].shape == (1, *case.particles.shape)
    assert case.matches_namespace(output["chain"])


def test_gaussian_target_is_stable_under_symmetric_kernel(
    context_factory,
) -> None:
    initial_rng = np.random.default_rng(123)
    particles = initial_rng.normal(size=(4000, 1))
    mutation = RedBlueMetropolisHastingsMutation(
        lambda values, context: -0.5 * np.sum(values**2, axis=1),
        GaussianKernel(scale=0.5),
        n_steps=4,
    )

    updated, _, _ = mutation(
        ParticleState(particles),
        context_factory(),
        RandomGenerator(backend="numpy", seed=17),
    )

    assert np.mean(updated.particles) == pytest.approx(0.0, abs=0.06)
    assert np.std(updated.particles) == pytest.approx(1.0, abs=0.06)


def test_red_blue_requires_two_particles(
    flat_log_prob, context_factory
) -> None:
    with pytest.raises(ValueError, match="n_steps"):
        RedBlueMetropolisHastingsMutation(
            flat_log_prob,
            _shift_kernel(),
            n_steps=0,
        )

    mutation = RedBlueMetropolisHastingsMutation(
        flat_log_prob,
        _shift_kernel(),
    )

    with pytest.raises(ValueError, match="at least two particles"):
        mutation(
            ParticleState(np.zeros((1, 2))),
            context_factory(),
            RandomGenerator(backend="numpy", seed=1),
        )


def test_red_blue_requires_complementary_ensemble_kernel(
    flat_log_prob,
) -> None:
    with pytest.raises(TypeError, match="ComplementaryEnsembleKernel"):
        RedBlueMetropolisHastingsMutation(
            flat_log_prob,
            object(),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
def test_sample_only_correction_and_nested_transition_particle_order(
    flat_log_prob, backend, backend_factory, context_factory
) -> None:
    from sequince.array_api import get_device, to_numpy

    case = backend_factory(backend)
    xp = case.xp

    class Kernel(ComplementaryEnsembleKernel):
        def sample(self, particles, complement, rng):
            return KernelSample(
                particles + 10,
                {
                    "nested": (particles[:, 0], {"original": particles}),
                    "none": None,
                },
            )

        def log_transition_ratio_from_sample(
            self, particles, sample, complement
        ):
            assert sample.auxiliary["nested"][1]["original"] is particles
            return xp.where(
                sample.auxiliary["nested"][0] % 2 == 0, 0.0, -xp.inf
            )

    class Metric:
        name = "sweeps"
        needs_transition = True

        def initialize(self, particles, weights, context):
            return 0

        def update(
            self, state, particles, accepted, context, *, transition=None
        ):
            ids, nested = transition.kernel_auxiliary["nested"]
            assert transition.kernel_auxiliary["none"] is None
            assert nested["original"].dtype == particles.dtype
            assert get_device(nested["original"]) == get_device(particles)
            np.testing.assert_array_equal(to_numpy(ids), np.arange(7))
            np.testing.assert_array_equal(
                to_numpy(nested["original"][:, 0]), to_numpy(ids)
            )
            np.testing.assert_array_equal(
                to_numpy(accepted), to_numpy(ids % 2 == 0)
            )
            np.testing.assert_array_equal(
                to_numpy(transition.displacement[:, 0]),
                np.where(to_numpy(accepted), 10, 0),
            )
            return state + 1

        def result(self, state, context):
            return state

    mutation = RedBlueMetropolisHastingsMutation(
        flat_log_prob, Kernel(), metrics=[Metric()]
    )
    _, _, output = mutation(
        ParticleState(xp.arange(7, dtype=case.particles.dtype)[:, None]),
        context_factory(),
        case.rng(17),
    )
    assert output["metrics"].values["sweeps"] == 1
