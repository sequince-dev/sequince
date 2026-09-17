"""Verify chain retention, Metropolis transitions and SMC bookkeeping."""

import numpy as np
import pytest

from sequince import ParticleState, TemperedPosterior
from sequince.array_api import namespace
from sequince.steps import AdaptiveGaussianProposal, WasteFreeResampleMove
from sequince.steps.mutation import (
    CallableMutationKernel,
    GaussianRandomWalk,
    KernelSample,
    MetropolisHastingsMutation,
)


def waste_free_mh(target, kernel, *, chain_length):
    return WasteFreeResampleMove(
        MetropolisHastingsMutation(target, kernel), chain_length=chain_length
    )


class ScriptedRNG:
    """Control ancestry and accept/reject draws without Monte Carlo tolerances."""

    def __init__(self, ancestors, uniforms=()):
        self.ancestors = np.asarray(ancestors)
        self.uniforms = iter(uniforms)
        self.resampling_calls = 0
        self.input_weights = None

    def choice(self, n, *, size, replace, p):
        assert size == len(self.ancestors)
        assert replace
        assert n == len(p)
        self.resampling_calls += 1
        self.input_weights = np.asarray(p)
        return self.ancestors

    def random(self, shape):
        values = np.asarray(next(self.uniforms))
        assert values.shape == shape
        return values


def test_retains_starts_and_successive_states_with_ancestry(context_factory):
    state = ParticleState(
        np.arange(6.0)[:, None],
        log_weights=np.log(np.arange(1, 7) / 21),
        log_normalizing_constant=2.5,
        initial_ancestor_indices=np.arange(6) + 10,
    )
    starts_seen = []

    def sample(particles, rng):
        starts_seen.append(particles.copy())
        return particles + 1

    step = waste_free_mh(
        lambda p: np.zeros(p.shape[0]),
        CallableMutationKernel(sample, symmetric=True),
        chain_length=3,
    )
    rng = ScriptedRNG([4, 1], [[0.1, 0.9], [0.3, 0.7]])
    result, _, output = step.apply(state, context_factory(), rng)

    np.testing.assert_array_equal(result.particles[:, 0], [4, 1, 5, 2, 6, 3])
    np.testing.assert_array_equal(starts_seen[1][:, 0], [5, 2])
    np.testing.assert_array_equal(output["ancestors"], [4, 1, 4, 1, 4, 1])
    np.testing.assert_array_equal(
        result.initial_ancestor_indices, [14, 11] * 3
    )
    np.testing.assert_allclose(result.weights, np.full(6, 1 / 6))
    np.testing.assert_allclose(rng.input_weights, state.weights)
    assert rng.resampling_calls == 1  # No final resampling of retained states.
    assert result.log_evidence == 2.5
    assert output["n_chains"] == 2
    assert output["chain_length"] == 3
    assert output["acceptance_by_step"] == (1.0, 1.0)
    assert output["metrics"].n_steps == 2
    assert output["metrics"].n_attempted == 4
    assert output["metrics"].n_accepted == 4


def test_rejections_retain_repeats_and_use_start_cache(context_factory):
    calls = []

    def log_likelihood(particles):
        calls.append(particles.copy())
        return -np.log(2) * particles[:, 0]

    target = TemperedPosterior(
        log_prior=lambda p: np.zeros(p.shape[0]),
        log_likelihood=log_likelihood,
    )
    state = ParticleState(np.zeros((6, 1)))
    context = context_factory(metadata={"beta": 1.0})
    evaluation = target.prepare_evaluation(state.particles, context)
    state = state.with_updates(evaluation=evaluation)
    calls.clear()
    step = waste_free_mh(
        target,
        CallableMutationKernel(lambda p, r: p + 1, symmetric=True),
        chain_length=3,
    )
    # alpha=1/2: first chain accepts then rejects, second rejects then accepts.
    rng = ScriptedRNG([0, 1], [[0.1, 0.9], [0.9, 0.1]])
    result, context, output = step.apply(state, context, rng)

    np.testing.assert_array_equal(result.particles[:, 0], [0, 0, 1, 0, 1, 1])
    assert len(calls) == 2  # Starting evaluations reused, one batch per step.
    assert result.evaluation is None  # Endpoint cache cannot cover all states.
    assert context.evaluation_counts.log_likelihood_evaluations == 4
    assert context.evaluation_counts.log_target_evaluations == 4
    assert output["metrics"].n_accepted == 2
    assert output["metrics"].acceptance == 0.5
    assert output["acceptance_by_step"] == (0.5, 0.5)


def test_asymmetric_proposal_uses_hastings_correction(context_factory):
    # q(1|0)=0.8 and q(0|1)=0.2 for a uniform two-state target.
    # Script proposals flipping the states; alpha(0,1)=1/4, alpha(1,0)=1.
    class FlipKernel:
        def sample(self, particles, rng):
            return KernelSample(1 - particles)

        def log_density(self, source, destination):
            return np.where(source[:, 0] == 0, np.log(0.8), np.log(0.2))

    kernel = FlipKernel()
    step = waste_free_mh(lambda p: 0.0, kernel, chain_length=2)
    state = ParticleState(np.array([[0.0], [1.0], [0.0], [1.0]]))
    rng = ScriptedRNG([0, 1], [[0.5, 0.5]])
    result, context, output = step.apply(state, context_factory(), rng)
    np.testing.assert_array_equal(result.particles[:, 0], [0, 1, 0, 0])
    assert output["metrics"].n_accepted == 1
    # Analytic transition corrections carry no charged proposal evaluations.
    assert context.evaluation_counts.log_proposal_evaluations == 0


@pytest.mark.parametrize("shape", [(6,), (6, 2), (6, 2, 3)])
def test_preserves_particle_shape(shape, context_factory, rng):
    step = waste_free_mh(
        lambda p: np.zeros(p.shape[0]),
        GaussianRandomWalk(scale=0.2),
        chain_length=3,
    )
    state = ParticleState(np.zeros(shape))
    result, _, _ = step(state, context_factory(), rng)
    assert result.particles.shape == shape


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax", "cupy"])
def test_preserves_array_backend(backend, backend_factory, context_factory):
    case = backend_factory(backend)
    xp = case.xp
    step = waste_free_mh(
        lambda p: -0.5 * xp.sum(p**2, axis=1),
        GaussianRandomWalk(scale=0.4),
        chain_length=2,
    )
    result, _, output = step(case.state(), context_factory(), case.rng(123))
    for array in (result.particles, result.log_weights, output["ancestors"]):
        assert namespace(array) is namespace(case.particles)
    assert result.particles.shape == case.particles.shape


def test_chain_length_one_only_resamples(context_factory):
    def fail(*args):
        pytest.fail(
            "Pure resampling must not evaluate the target or build a kernel"
        )

    step = waste_free_mh(fail, fail, chain_length=1)
    state = ParticleState(np.arange(4.0))
    rng = ScriptedRNG([3, 3, 1, 2])
    result, _, output = step.apply(state, context_factory(), rng)
    np.testing.assert_array_equal(result.particles, [3, 3, 1, 2])
    assert output["metrics"].n_attempted == 0
    assert output["acceptance_by_step"] == ()


def test_supports_column_log_prob(context_factory, rng):
    step = waste_free_mh(
        lambda p: -0.5 * p**2, GaussianRandomWalk(), chain_length=np.int64(2)
    )
    result, _, output = step(
        ParticleState(np.zeros((4, 1))), context_factory(), rng
    )
    assert result.particles.shape == (4, 1)
    assert output["n_chains"] == 2


@pytest.mark.parametrize("length", [0, -1, True, np.bool_(True), 1.5, "2"])
def test_invalid_chain_length(length):
    error = ValueError if length in (0, -1) else TypeError
    with pytest.raises(error, match="chain_length"):
        waste_free_mh(lambda p: 0.0, GaussianRandomWalk(), chain_length=length)


def test_chain_length_must_divide_population(context_factory, rng):
    step = waste_free_mh(lambda p: 0.0, GaussianRandomWalk(), chain_length=3)
    with pytest.raises(ValueError, match="divisible"):
        step(ParticleState(np.zeros((4, 1))), context_factory(), rng)


def test_obsolete_arguments_are_not_reinterpreted():
    with pytest.raises(TypeError, match="proposals_per_particle"):
        waste_free_mh(
            lambda p: 0.0,
            GaussianRandomWalk(),
            chain_length=2,
            proposals_per_particle=2,
        )


def test_adaptation_uses_realized_acceptance_and_freezes_kernel(
    context_factory, rng
):
    factory_calls = []

    def factory(starts, context):
        factory_calls.append(
            (starts.shape, context.metadata["proposal_scale"])
        )
        return CallableMutationKernel(lambda p, r: p + 1, symmetric=True)

    # alpha=1 everywhere: adaptation can be checked without random tolerances.
    step = AdaptiveGaussianProposal(
        waste_free_mh(lambda p: 0.0, factory, chain_length=3),
        initial_scale=0.7,
        target_acceptance=0.3,
        adaptation_rate=0.2,
    )
    state = ParticleState(np.zeros((6, 1)))
    state, context, output = step(state, context_factory(), rng)
    assert output["metrics"].n_attempted == 4
    assert output["metrics"].n_accepted == 4
    expected_scale = 0.7 * np.exp(0.2 * (1.0 - 0.3))
    assert context.metadata["proposal_scale"] == pytest.approx(expected_scale)
    step(state, context, rng)
    assert [shape for shape, _ in factory_calls] == [(2, 1), (2, 1)]
    assert [scale for _, scale in factory_calls] == pytest.approx(
        [0.7, expected_scale]
    )


@pytest.mark.parametrize("chain_length", [1, 2])
def test_callable_drops_cache_only_when_mutating(
    chain_length, context_factory, rng
):
    target = TemperedPosterior(
        log_prior=lambda p: np.zeros(p.shape[0]),
        log_likelihood=lambda p: -(p[:, 0] ** 2),
    )
    state = ParticleState(np.arange(4.0)[:, None])
    context = context_factory(metadata={"beta": 1.0})
    state = state.with_updates(
        evaluation=target.prepare_evaluation(state.particles, context)
    )
    step = waste_free_mh(
        lambda p: 0.0, GaussianRandomWalk(), chain_length=chain_length
    )
    result, _, _ = step(state, context, rng)
    if chain_length == 1:
        np.testing.assert_allclose(
            result.evaluation["log_likelihood"], -(result.particles[:, 0] ** 2)
        )
    else:
        assert result.evaluation is None


@pytest.mark.parametrize("bad_part", ["particles", "log_prob", "ratio"])
def test_rejects_invalid_kernel_and_target_shapes(
    bad_part, context_factory, rng
):
    class Kernel:
        def sample(self, particles, rng):
            return KernelSample(
                particles[:1] if bad_part == "particles" else particles
            )

        def log_transition_ratio(self, source, destination):
            return np.zeros((2, 2)) if bad_part == "ratio" else np.zeros(2)

    step = waste_free_mh(
        lambda p: np.zeros((2, 2)) if bad_part == "log_prob" else np.zeros(2),
        Kernel(),
        chain_length=2,
    )
    with pytest.raises(ValueError):
        step(ParticleState(np.zeros((4, 1))), context_factory(), rng)


@pytest.mark.parametrize("store_chains", [False, True])
def test_runs_one_complete_block_and_restores_configuration(
    store_chains, context_factory, rng
):
    calls = []

    def factory(particles, context):
        calls.append(particles.copy())
        return CallableMutationKernel(lambda p, r: p + 1, symmetric=True)

    mutation = MetropolisHastingsMutation(
        lambda p: 0.0, factory, n_steps=17, store_chains=store_chains
    )
    step = WasteFreeResampleMove(mutation, chain_length=4)
    result, _, output = step(
        ParticleState(np.zeros((8, 1))), context_factory(), rng
    )
    assert len(calls) == 1
    assert mutation.n_steps == 17
    assert mutation.store_chains is store_chains
    np.testing.assert_array_equal(
        result.particles.reshape(4, 2)[:, 0], [0, 1, 2, 3]
    )
    assert output["mutation"]["chain"].shape == (3, 2, 1)
    assert output["metrics"] is output["mutation"]["metrics"]


@pytest.mark.parametrize(
    "failure", ["exception", "missing_chain", "bad_shape", "missing_metrics"]
)
def test_rejects_invalid_mutation_output_and_restores_storage(
    failure, context_factory, rng
):
    from sequince.steps.mutation import MutationStep

    class BrokenMutation(MutationStep):
        def apply_with_n_steps(self, state, context, rng, *, n_steps):
            if failure == "exception":
                raise RuntimeError("failed mutation")
            return (
                state,
                context,
                {
                    "chain": None
                    if failure == "missing_chain"
                    else np.zeros(
                        (1, 2, 1) if failure == "bad_shape" else (2, 2, 1)
                    ),
                    "metrics": None,
                },
            )

    mutation = BrokenMutation(lambda p: 0.0)
    step = WasteFreeResampleMove(mutation, chain_length=3)
    error = (
        RuntimeError
        if failure == "exception"
        else TypeError
        if failure == "missing_metrics"
        else ValueError
    )
    with pytest.raises(error):
        step(ParticleState(np.zeros((6, 1))), context_factory(), rng)
    assert mutation.store_chains is False


def test_custom_gibbs_mutation_and_checkpoint_forwarding(context_factory, rng):
    from sequince.steps.mutation import MutationMetricsOutput, MutationStep

    class GaussianGibbs(MutationStep):
        def __init__(self):
            super().__init__(lambda p: -0.5 * np.sum(p**2, axis=1))
            self.calls = 0

        def apply_with_n_steps(self, state, context, rng, *, n_steps):
            self.calls += 1
            assert self.store_chains
            # Independent normal coordinates: an exact full Gibbs sweep.
            chain = rng.normal(size=(n_steps, *state.particles.shape))
            attempts = n_steps * len(state.particles)
            return (
                state.replace_particles(chain[-1]),
                context,
                {
                    "chain": chain,
                    "metrics": MutationMetricsOutput(
                        n_steps, attempts, attempts
                    ),
                    "gibbs_sweeps": n_steps,
                },
            )

        def state_dict(self):
            return {"calls": self.calls}

        def load_state_dict(self, state):
            self.calls = state["calls"]

    mutation = GaussianGibbs()
    step = WasteFreeResampleMove(mutation, chain_length=4)
    result, _, output = step(
        ParticleState(np.zeros((8, 2))), context_factory(), rng
    )
    assert result.particles.shape == (8, 2)
    np.testing.assert_array_equal(
        result.particles[2:], output["mutation"]["chain"].reshape(6, 2)
    )
    assert output["mutation"]["gibbs_sweeps"] == 3
    assert output["acceptance_by_step"] is None
    restored = WasteFreeResampleMove(GaussianGibbs(), chain_length=4)
    restored.load_state_dict(step.state_dict())
    assert restored.mutation.calls == 1


@pytest.mark.parametrize("algorithm", ["hmc", "nuts"])
def test_blackjax_retains_full_chains_and_diagnostics(
    algorithm, context_factory
):
    jnp = pytest.importorskip("jax.numpy")
    pytest.importorskip("blackjax")
    from orng import RandomGenerator

    from sequince.steps.mutation import BlackJAXMutation

    kwargs = {"step_size": 0.2, "inverse_mass_matrix": jnp.ones(2)}
    if algorithm == "hmc":
        kwargs["num_integration_steps"] = 3
    mutation = BlackJAXMutation(
        lambda p: -0.5 * jnp.sum(p**2, axis=-1),
        algorithm=algorithm,
        algorithm_kwargs=kwargs,
        n_steps=12,
    )
    step = WasteFreeResampleMove(mutation, chain_length=4)
    state = ParticleState(jnp.zeros((16, 2)))
    result, _, output = step(
        state, context_factory(), RandomGenerator(backend="jax", seed=123)
    )
    assert namespace(result.particles) is namespace(state.particles)
    np.testing.assert_array_equal(
        result.particles[4:],
        np.asarray(output["mutation"]["chain"]).reshape(12, 2),
    )
    assert output["metrics"].n_steps == 3
    assert output["metrics"].n_attempted == 12
    assert output["mutation"]["algorithm"] == algorithm
    assert output["mutation"]["num_integration_steps"].shape == (3, 4)
    assert mutation.n_steps == 12
    assert mutation.store_chains is False


def test_minipcn_retains_full_chains(context_factory, rng):
    pytest.importorskip("minipcn")
    from sequince.steps.mutation import MiniPCNMutation

    mutation = MiniPCNMutation(
        lambda p: -0.5 * np.sum(p**2, axis=-1), n_steps=12
    )
    step = WasteFreeResampleMove(mutation, chain_length=4)
    state = ParticleState(np.random.default_rng(12).normal(size=(64, 2)))
    result, _, output = step(state, context_factory(), rng)
    assert result.particles.shape == (64, 2)
    np.testing.assert_array_equal(
        result.particles[16:], output["mutation"]["chain"].reshape(48, 2)
    )
    assert output["metrics"].n_steps == 3
    assert mutation.store_chains is False


def test_transformed_mutation_retains_original_coordinates(
    context_factory, rng
):
    from sequince.transforms import ParticleTransform

    class ShiftTransform(ParticleTransform):
        def __init__(self):
            self.fits = 0

        def fit(self, particles, context=None, *, rng=None):
            self.fits += 1
            return self

        def forward(self, particles, context=None, *, rng=None):
            return particles - 10, np.zeros(len(particles)), None

        def inverse(self, values, context=None, *, auxiliary=None, rng=None):
            return values + 10, np.zeros(len(values))

    transform = ShiftTransform()
    mutation = MetropolisHastingsMutation(
        lambda p: 0.0,
        CallableMutationKernel(lambda p, r: p + 1, symmetric=True),
        transform=transform,
    )
    step = WasteFreeResampleMove(mutation, chain_length=4)
    result, _, _ = step(
        ParticleState(np.zeros((8, 1))), context_factory(), rng
    )
    np.testing.assert_array_equal(
        result.particles.reshape(4, 2)[:, 0], [0, 1, 2, 3]
    )
    assert transform.fits == 1
