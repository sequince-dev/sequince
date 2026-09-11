from __future__ import annotations

import numpy as np
import pytest
from orng import RandomGenerator

from sequince import ParticleState
from sequince.steps import ErynMutation, RepeatedMutationStep
from sequince.steps.mutation import get_mutation_step
from sequince.transforms import ParticleTransform

pytest.importorskip("eryn")


@pytest.mark.parametrize("vectorize", [False, True])
def test_target_cache_counts_and_reproducibility(
    particle_state, context_factory, standard_normal_posterior, vectorize
):
    step = get_mutation_step(
        "eryn",
        standard_normal_posterior,
        mutation_n_steps=3,
        mutation_kwargs={"vectorize": vectorize, "store_chains": True},
    )
    assert isinstance(step, ErynMutation)
    initial_rng = np.random.get_state()
    first, context, output = step(
        particle_state,
        context_factory(metadata={"temperature": 0.4}),
        RandomGenerator(seed=2468),
    )
    second, _, replay = step(
        particle_state,
        context_factory(metadata={"temperature": 0.4}),
        RandomGenerator(seed=2468),
    )
    np.testing.assert_array_equal(first.particles, second.particles)
    np.testing.assert_array_equal(output["chain"], replay["chain"])
    np.testing.assert_array_equal(output["chain"][-1], first.particles)
    np.testing.assert_array_equal(
        first.log_weights, particle_state.log_weights
    )
    assert not np.array_equal(first.particles, particle_state.particles)
    assert output["chain"].shape == (3, *particle_state.particles.shape)
    assert output["metrics"].n_steps == 3
    n = len(particle_state.particles)
    assert output["metrics"].n_attempted == 3 * n
    assert 0 < output["metrics"].acceptance <= 1
    assert context.evaluation_counts.log_target_evaluations == 5 * n
    expected = standard_normal_posterior.evaluate(first.particles, context)
    for key in ("log_prior", "log_likelihood"):
        np.testing.assert_allclose(first.evaluation[key], expected[key])
    after_rng = np.random.get_state()
    for first_part, last_part in zip(initial_rng, after_rng):
        np.testing.assert_equal(first_part, last_part)


@pytest.mark.parametrize("shape", [(20,), (20, 1), (20, 2)])
def test_repeated_metrics_and_zero_budget(shape, context_factory):
    state = ParticleState(np.random.default_rng(7).normal(size=shape))
    step = ErynMutation(
        lambda x: -0.5 * np.sum(x * x, axis=-1),
        n_steps=2,
        store_chains=True,
    )
    unchanged, _, zero = step.apply_with_n_steps(
        state,
        context_factory(),
        RandomGenerator(seed=7),
        n_steps=0,
    )
    assert unchanged is state
    assert zero["chain"].shape == (0, *shape)
    assert zero["metrics"].n_attempted == 0
    block = RepeatedMutationStep(step, n_rounds=3)
    result, _, output = block(
        state, context_factory(), RandomGenerator(seed=7)
    )
    assert result.particles.shape == shape
    assert output.metrics.n_steps == 6
    assert output.metrics.n_attempted == 6 * shape[0]
    for row in output.round_outputs:
        assert row["chain"].shape == (2, *shape)


class ExponentialTransform(ParticleTransform):
    def forward(self, particles, context=None, *, rng=None):
        return np.log(particles), -np.log(particles).sum(axis=-1), None

    def inverse(self, values, context=None, *, auxiliary=None, rng=None):
        return np.exp(values), values.sum(axis=-1)


@pytest.mark.parametrize("vectorize", [False, True])
def test_transform_jacobian_matches_latent_density(vectorize, context_factory):
    latent = np.random.default_rng(6).normal(size=(20, 2))
    # An exponential density in x becomes exp(z) * exp(-exp(z)) in z.
    transformed = ErynMutation(
        lambda x: -np.sum(x, axis=-1),
        transform=ExponentialTransform(),
        n_steps=4,
        vectorize=vectorize,
        store_chains=True,
    )
    direct = ErynMutation(
        lambda z: np.sum(z - np.exp(z), axis=-1),
        n_steps=4,
        vectorize=vectorize,
        store_chains=True,
    )
    result, _, output = transformed(
        ParticleState(np.exp(latent)),
        context_factory(),
        RandomGenerator(seed=12),
    )
    expected, _, reference = direct(
        ParticleState(latent),
        context_factory(),
        RandomGenerator(seed=12),
    )
    np.testing.assert_allclose(result.particles, np.exp(expected.particles))
    np.testing.assert_allclose(output["chain"], np.exp(reference["chain"]))


@pytest.mark.parametrize(
    "run_kwargs",
    [
        {"thin_by": 2},
        {"store": False},
        {"iterations": 10},
    ],
)
def test_invalid_sample_options(run_kwargs):
    with pytest.raises(ValueError):
        ErynMutation(lambda x: 0, run_kwargs=run_kwargs)


@pytest.mark.parametrize("n_steps", [0, -1, True, 1.5])
def test_invalid_budget(n_steps):
    with pytest.raises((ValueError, TypeError)):
        ErynMutation(lambda x: 0, n_steps=n_steps)


def test_custom_move(context_factory):
    from eryn.moves import GaussianMove

    move = GaussianMove({"model_0": np.eye(2) * 0.1})
    state = ParticleState(np.zeros((10, 2)))
    result, _, output = ErynMutation(
        lambda x: np.zeros(x.shape[0]),
        move=move,
        n_steps=3,
    )(state, context_factory(), RandomGenerator(seed=7))
    assert output["metrics"].acceptance == 1
    assert output["chain"] is None
    assert np.any(result.particles != 0)


def test_rng_restored_after_target_failure(particle_state, context_factory):
    def fail(x):
        raise RuntimeError("target failed")

    before = np.random.get_state()
    with pytest.raises(RuntimeError, match="target failed"):
        ErynMutation(fail)(
            particle_state, context_factory(), RandomGenerator(seed=3)
        )
    for previous, current in zip(before, np.random.get_state()):
        np.testing.assert_equal(previous, current)


@pytest.mark.parametrize("vectorize", [False, True])
def test_tempered_target_matches_callable(vectorize, context_factory):
    from sequince.posterior import TemperedPosterior

    target = TemperedPosterior(
        log_prior=lambda x: -0.5 * np.sum(x * x, axis=-1),
        log_likelihood=lambda x: -np.sum((x - 2) ** 2, axis=-1),
        log_proposal=lambda x: -np.sum((x + 1) ** 2, axis=-1),
    )
    state = ParticleState(np.random.default_rng(3).normal(size=(20, 2)))
    context = context_factory(metadata={"temperature": 0.3})
    actual, _, _ = ErynMutation(target, n_steps=5, vectorize=vectorize)(
        state,
        context,
        RandomGenerator(seed=4),
    )
    expected, _, _ = ErynMutation(
        lambda x: target.log_prob(np.atleast_2d(x), context),
        n_steps=5,
        vectorize=vectorize,
    )(state, context, RandomGenerator(seed=4))
    np.testing.assert_array_equal(actual.particles, expected.particles)


def test_particle_auxiliary_rejected(particle_state, context_factory):
    class WalkerTransform(ParticleTransform):
        def forward(self, values, context=None, *, rng=None):
            return values, np.zeros(len(values)), np.arange(len(values))

    with pytest.raises(NotImplementedError, match="walker subsets"):
        ErynMutation(lambda x: 0, transform=WalkerTransform())(
            particle_state,
            context_factory(),
            RandomGenerator(seed=4),
        )


@pytest.mark.parametrize("backend_name", ["numpy", "jax", "torch"])
def test_array_backend_preserved(
    backend_name, backend_factory, context_factory
):
    from eryn.moves import GaussianMove

    case = backend_factory(backend_name)
    step = ErynMutation(
        lambda x: -case.xp.sum(x * x, axis=-1),
        vectorize=True,
        store_chains=True,
        move=GaussianMove({"model_0": np.eye(2) * 0.1}),
    )
    result, _, output = step(case.state(), context_factory(), case.rng(4))
    assert case.matches_namespace(result.particles)
    assert case.matches_namespace(output["chain"])
    assert result.particles.dtype == case.particles.dtype


@pytest.fixture(params=["numpy", "cupy"])
def eryn_xp(request):
    if request.param == "numpy":
        return np
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("No CUDA device available")
        cp.zeros(1)
    except cp.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CUDA unavailable: {exc}")
    return cp


@pytest.mark.parametrize("vectorize", [False, True])
@pytest.mark.parametrize("custom_move", [False, True])
def test_numpy_and_cupy_execution(
    eryn_xp, vectorize, custom_move, context_factory, monkeypatch
):
    from eryn.moves import StretchMove

    from sequince.array_api import namespace, to_numpy
    from sequince.posterior import TemperedPosterior

    xp = eryn_xp
    use_gpu = xp is not np
    proposals = []
    original_proposal = StretchMove.get_proposal

    def check_proposal(self, *args, **kwargs):
        assert self.use_gpu is use_gpu
        assert self.xp is xp
        assert not self.return_gpu
        proposals.append(True)
        return original_proposal(self, *args, **kwargs)

    monkeypatch.setattr(StretchMove, "get_proposal", check_proposal)

    def log_density(x):
        assert isinstance(x, xp.ndarray)
        return -0.5 * xp.sum(x * x, axis=-1)

    class ShiftTransform(ParticleTransform):
        def forward(self, particles, context=None, *, rng=None):
            assert isinstance(particles, xp.ndarray)
            return particles - 1, xp.zeros(particles.shape[0]), None

        def inverse(self, values, context=None, *, auxiliary=None, rng=None):
            assert isinstance(values, xp.ndarray)
            return values + 1, xp.zeros_like(values[..., 0])

    target = TemperedPosterior(
        log_prior=log_density, log_likelihood=log_density
    )
    particles = xp.asarray(np.random.default_rng(24).normal(size=(20, 2)))
    state = ParticleState(particles)
    move = [(StretchMove(a=1.5), 0.4), (StretchMove(a=2.5), 0.6)]
    step = ErynMutation(
        target,
        n_steps=3,
        vectorize=vectorize,
        store_chains=True,
        move=move if custom_move else None,
        transform=ShiftTransform(),
    )
    before = xp.random.get_random_state() if use_gpu else np.random.get_state()
    results = []
    for _ in range(2):
        results.append(
            step(
                state,
                context_factory(metadata={"temperature": 0.4}),
                RandomGenerator(seed=7),
            )
        )
    first, context, output = results[0]
    second, _, replay = results[1]
    assert proposals
    assert namespace(first.particles) is namespace(particles)
    assert namespace(output["chain"]) is namespace(particles)
    np.testing.assert_array_equal(
        to_numpy(first.particles), to_numpy(second.particles)
    )
    np.testing.assert_array_equal(
        to_numpy(output["chain"]), to_numpy(replay["chain"])
    )
    np.testing.assert_array_equal(
        to_numpy(output["chain"][-1]), to_numpy(first.particles)
    )
    np.testing.assert_array_equal(
        to_numpy(state.particles), to_numpy(particles)
    )
    expected = target.evaluate(first.particles, context)
    for component in ("log_prior", "log_likelihood"):
        assert isinstance(first.evaluation[component], xp.ndarray)
        np.testing.assert_allclose(
            to_numpy(first.evaluation[component]),
            to_numpy(expected[component]),
        )
    assert output["metrics"].n_attempted == 60
    assert 0 < output["metrics"].acceptance <= 1
    assert context.evaluation_counts.log_target_evaluations == 100
    # Runtime backend selection must not mutate user-supplied moves.
    assert all(not item.use_gpu for item, _ in move)
    if use_gpu:
        assert xp.random.get_random_state() is before
    else:
        for previous, current in zip(before, np.random.get_state()):
            np.testing.assert_equal(previous, current)


def test_cpu_overrides_gpu_move_configuration(context_factory):
    from eryn.moves import StretchMove

    move = StretchMove(use_gpu=True, return_gpu=True)
    state = ParticleState(np.random.default_rng(3).normal(size=(20, 2)))
    result, _, _ = ErynMutation(
        lambda x: -0.5 * np.sum(x * x, axis=-1),
        move=move,
    )(state, context_factory(), RandomGenerator(seed=4))
    assert isinstance(result.particles, np.ndarray)
    assert move.use_gpu and move.return_gpu


@pytest.mark.parametrize("fail", [False, True])
def test_cupy_rng_is_replaced_and_restored_without_reseeding_caller(fail):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from sequince.steps.mutation.eryn import _seed_eryn_rng

    original = Mock()
    temporary = Mock()
    random = Mock()
    random.get_random_state.return_value = original
    random.RandomState.return_value = temporary
    before = np.random.get_state()

    def run():
        with _seed_eryn_rng(2468, cupy=SimpleNamespace(random=random)):
            random.set_random_state.assert_called_once_with(temporary)
            if fail:
                raise RuntimeError("proposal failed")

    if fail:
        with pytest.raises(RuntimeError, match="proposal failed"):
            run()
    else:
        run()
    random.RandomState.assert_called_once_with(2468)
    random.set_random_state.assert_called_with(original)
    assert random.set_random_state.call_count == 2
    original.seed.assert_not_called()
    for previous, current in zip(before, np.random.get_state()):
        np.testing.assert_equal(previous, current)
