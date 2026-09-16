"""Numerical and state invariants for the optional Pyro mutation."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from sequince import ParticleState, TemperedPosterior
from sequince.steps import PyroNUTSMutation, StepContext
from sequince.steps.mutation import get_mutation_step
from sequince.transforms import AffineTransform, ParticleTransform
from sequince.utils.random import resolve_rng


def test_without_pyro() -> None:
    script = """
import sys
sys.modules['pyro'] = None
import sequince
from sequince.steps.mutation import PyroNUTSMutation, get_mutation_step
assert get_mutation_step('differential-evolution', lambda x: 0) is not None
try:
    PyroNUTSMutation(lambda x: 0)
except ImportError as error:
    assert "sequince[pyro]" in str(error)
else:
    raise AssertionError('Pyro import should fail')
"""
    subprocess.run([sys.executable, "-c", script], check=True)


@pytest.fixture
def torch():
    pytest.importorskip("pyro")
    return pytest.importorskip("torch")


@pytest.fixture
def target(torch):
    return TemperedPosterior(
        lambda x: -0.5 * (x**2).sum(-1),
        lambda x: -0.5 * ((x - 1) ** 2).sum(-1),
    )


@pytest.mark.parametrize("beta", [0.0, 0.35, 1.0])
def test_potential(torch, target, beta) -> None:
    x = torch.tensor([0.2, -0.8], dtype=torch.float64, requires_grad=True)
    context = StepContext(t=0, metadata={"beta": beta})
    potential, _ = PyroNUTSMutation(target)._potential(x, context)
    expected = 0.5 * (x**2).sum() + beta * 0.5 * ((x - 1) ** 2).sum()
    torch.testing.assert_close(potential, expected)
    (gradient,) = torch.autograd.grad(potential, x)
    torch.testing.assert_close(gradient, x + beta * (x - 1))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_steps": 0},
        {"n_steps": True},
        {"n_steps": 1.2},
        {"max_tree_depth": 0},
        {"max_tree_depth": False},
        {"max_tree_depth": 2.5},
        {"step_size": 0},
        {"step_size": -1},
        {"step_size": float("nan")},
        {"step_size": float("inf")},
        {"step_size": True},
        {"use_multinomial_sampling": 1},
        {"store_chains": 1},
    ],
)
def test_invalid_configuration(kwargs) -> None:
    with pytest.raises((ValueError, TypeError)):
        PyroNUTSMutation(lambda x: x, **kwargs)


@pytest.mark.parametrize(
    "kind,match",
    [
        ("float", "Torch tensor"),
        ("numpy", "Torch tensor"),
        ("detached", "detached"),
        ("vector", "scalar"),
        ("dtype", "dtype and device"),
        ("unrelated", "gradients"),
        ("nonfinite", "finite"),
    ],
)
def test_invalid_targets(torch, kind, match) -> None:
    def log_prob(x):
        if kind == "float":
            return 1.0
        if kind == "numpy":
            return np.array(1.0)
        if kind == "detached":
            return x.sum().detach()
        if kind == "vector":
            return x
        if kind == "dtype":
            return x.sum().float()
        if kind == "unrelated":
            return torch.tensor(1.0, dtype=x.dtype, requires_grad=True)
        return x.sqrt().sum()

    state = ParticleState(torch.tensor([[-1.0, 0.2]], dtype=torch.float64))
    rng_before = torch.random.get_rng_state().clone()
    with pytest.raises((ValueError, TypeError), match=match):
        PyroNUTSMutation(log_prob).apply(state, StepContext(t=0), 12)
    assert torch.equal(torch.random.get_rng_state(), rng_before)


def test_non_torch(torch, target) -> None:
    with pytest.raises(TypeError, match="Torch-backed"):
        PyroNUTSMutation(target).apply(
            ParticleState(np.ones((2, 1))), StepContext(t=0), 1
        )
    with pytest.raises(TypeError, match="floating-point"):
        PyroNUTSMutation(target).apply(
            ParticleState(torch.ones((2, 1), dtype=torch.int64)),
            StepContext(t=0),
            1,
        )


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("multinomial", [False, True])
def test_mutation(torch, target, dtype, multinomial) -> None:
    particles = torch.tensor(
        [[0.2, -0.1], [0.2, -0.1], [-0.8, 0.4]], dtype=getattr(torch, dtype)
    )
    context = StepContext(t=3, metadata={"beta": 0.4, "other": "preserved"})
    stale = target.prepare_evaluation(particles + 100, context)
    state = ParticleState(particles, evaluation=stale, metadata={"other": [1]})
    weights = state.log_weights.clone()
    mutation = PyroNUTSMutation(
        target,
        n_steps=3,
        step_size=0.3,
        max_tree_depth=4,
        store_chains=True,
        use_multinomial_sampling=multinomial,
    )
    before = torch.random.get_rng_state().clone()
    with torch.no_grad():
        result, updated, output = mutation.apply(state, context, 123)
    assert torch.equal(before, torch.random.get_rng_state())
    assert result.particles.shape == particles.shape
    assert result.particles.dtype == particles.dtype
    assert result.particles.device == particles.device
    assert not result.particles.requires_grad
    assert torch.isfinite(result.particles).all()
    assert not torch.equal(result.particles, particles)
    assert not torch.equal(result.particles[0], result.particles[1])
    assert torch.equal(result.log_weights, weights)
    assert torch.equal(state.particles, particles)
    assert state.evaluation is stale
    assert result.metadata == state.metadata
    assert torch.equal(
        result.initial_ancestor_indices, state.initial_ancestor_indices
    )
    assert updated.metadata == context.metadata
    for name in ("log_prior", "log_likelihood"):
        torch.testing.assert_close(
            result.evaluation[name],
            target.prepare_evaluation(result.particles, context)[name],
        )
        assert not result.evaluation[name].requires_grad
    assert output["chain"].shape == (3, *particles.shape)
    assert output["metrics"].n_steps == 3
    assert output["metrics"].n_attempted == 9
    assert 0 <= output["pyro_acceptance_rate"] <= 1
    assert output["divergences"] >= 0
    assert updated.evaluation_counts.log_likelihood_evaluations > 3
    repeated = mutation.apply(state, context, 123)[0]
    different = mutation.apply(state, context, 124)[0]
    assert torch.equal(result.particles, repeated.particles)
    assert not torch.equal(result.particles, different.particles)


def test_affine_transform(torch, target) -> None:
    particles = torch.tensor([[0.0], [4.0]], dtype=torch.float64)
    context = StepContext(t=0, metadata={"beta": 0.5})
    transform = AffineTransform(eps=0).fit(particles)
    z, forward_det, _ = transform.forward(particles)
    restored, inverse_det = transform.inverse(z)
    torch.testing.assert_close(
        z, torch.tensor([[-1.0], [1.0]], dtype=particles.dtype)
    )
    torch.testing.assert_close(restored, particles)
    torch.testing.assert_close(
        inverse_det, torch.full((2,), np.log(2), dtype=particles.dtype)
    )
    torch.testing.assert_close(forward_det, -inverse_det)
    position = z[0].clone().requires_grad_()
    potential, _ = PyroNUTSMutation(target)._potential(
        position, context, transform
    )
    expected = -target.log_prob(position * 2 + 2, context) - np.log(2)
    torch.testing.assert_close(potential, expected)
    (gradient,) = torch.autograd.grad(potential, position)
    assert torch.isfinite(gradient).all()
    result, _, output = PyroNUTSMutation(
        target, transform=AffineTransform(), store_chains=True
    ).apply(ParticleState(particles), context, 1)
    torch.testing.assert_close(output["chain"][-1], result.particles)


class DetachedTransform(ParticleTransform):
    def inverse(self, values, context=None, *, auxiliary=None, rng=None):
        return values.detach(), values.sum() * 0


def test_detached_transform(torch, target) -> None:
    state = ParticleState(torch.ones((2, 1), dtype=torch.float64))
    with pytest.raises(ValueError, match="Transform inverse is detached"):
        PyroNUTSMutation(target, transform=DetachedTransform()).apply(
            state, StepContext(t=0), 1
        )


def test_frozen_beta_and_counts(torch) -> None:
    beta = torch.tensor(0.5, dtype=torch.float64)
    seen = []

    def density(x, context):
        seen.append(context.metadata["beta"].clone())
        beta.fill_(0.9)
        return -(x**2).sum(-1) * context.metadata["beta"]

    mutation = PyroNUTSMutation(density, max_tree_depth=2)
    state = ParticleState(torch.ones((2, 1), dtype=torch.float64))
    _, context, _ = mutation.apply(
        state, StepContext(t=0, metadata={"beta": beta}), 3
    )
    assert all(value == 0.5 for value in seen)
    # One batch evaluation at the end accounts for both particles.
    assert context.evaluation_counts.log_target_evaluations == len(seen) + 1


def test_rng_and_block_protocol(torch, target) -> None:
    state = ParticleState(torch.ones((2, 1), dtype=torch.float64))
    context = StepContext(t=0)
    mutation = get_mutation_step("pyro-nuts", target, mutation_n_steps=2)
    rng = resolve_rng(42, backend="torch")
    first, context, _, accumulated = mutation.apply_with_mutation_state(
        state, context, rng, n_steps=1
    )
    _, context, _, accumulated = mutation.apply_with_mutation_state(
        first, context, rng, n_steps=2, mutation_state=accumulated
    )
    assert accumulated.n_steps == 3
    same, _, output = mutation.apply_with_n_steps(
        state, context, rng, n_steps=0
    )
    assert same is state
    assert output["metrics"].n_steps == 0


def test_cuda(torch, target) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    x = torch.tensor(
        [0.1, 0.2], device="cuda", dtype=torch.float64, requires_grad=True
    )
    mutation = PyroNUTSMutation(target, max_tree_depth=3)
    context = StepContext(t=0, metadata={"beta": 0.5})
    potential, _ = mutation._potential(x, context)
    (gradient,) = torch.autograd.grad(potential, x)
    assert potential.is_cuda and gradient.is_cuda
    before = torch.cuda.get_rng_state().clone()
    result, _, _ = mutation.apply(
        ParticleState(x.detach().repeat(2, 1)), context, 42
    )
    assert result.particles.is_cuda
    assert torch.equal(before, torch.cuda.get_rng_state())
    assert result.particles.dtype == x.dtype


def test_standard_smc(torch, target) -> None:
    from sequince import make_standard_bayesian_smc

    sampler, _, metadata = make_standard_bayesian_smc(
        target.log_prior,
        target.log_likelihood,
        mutation="pyro-nuts",
        mutation_n_steps=1,
        mutation_kwargs={"step_size": 0.2, "max_tree_depth": 3},
        max_delta=0.5,
        logging_frequency=0,
    )
    initial = ParticleState(
        torch.linspace(-2, 2, 16, dtype=torch.float64)[:, None]
    )
    with torch.no_grad():
        result = sampler.run(initial, rng=12, metadata=metadata)
    assert result.context.metadata["beta"] == 1
    assert torch.isfinite(result.final_state.particles).all()
    assert not result.final_state.particles.requires_grad


def test_unit_space_logit(torch) -> None:
    from sequince.transforms import BoundedToUnboundedTransform

    transform = BoundedToUnboundedTransform({0: (0.0, 1.0)})
    x = torch.tensor([0.25], dtype=torch.float64)
    z, forward_jacobian, _ = transform.forward(x)
    torch.testing.assert_close(z, torch.logit(x))
    restored, jacobian = transform.inverse(z)
    torch.testing.assert_close(restored, x)
    torch.testing.assert_close(jacobian, torch.log(x * (1 - x)).sum())
    torch.testing.assert_close(forward_jacobian, -jacobian)
    # Uniform unit-space prior has no additional physical-space prior factor.
    target = TemperedPosterior(
        lambda u: 0 * u.sum(-1), lambda u: -u.square().sum(-1)
    )
    position = z.detach().requires_grad_()
    mutation = PyroNUTSMutation(target, transform=transform)
    context = StepContext(t=0, metadata={"beta": 0.4})
    potential, _ = mutation._potential(position, context, transform)
    expected = 0.4 * x.square().sum() - torch.log(x * (1 - x)).sum()
    torch.testing.assert_close(potential, expected)
    (gradient,) = torch.autograd.grad(potential, position)
    torch.testing.assert_close(
        gradient, 0.8 * x.square() * (1 - x) - (1 - 2 * x)
    )


def test_exact_transition_lifecycle(torch, target, monkeypatch) -> None:
    import pyro.infer.mcmc
    from pyro.infer.mcmc import NUTS

    calls = []

    class RecordedNUTS(NUTS):
        def __init__(self, **kwargs):
            assert kwargs["adapt_step_size"] is False
            assert kwargs["adapt_mass_matrix"] is False
            assert kwargs["step_size"] == 0.2
            calls.append("new")
            super().__init__(**kwargs)

        def setup(self, warmup_steps, *args, **kwargs):
            assert warmup_steps == 0
            calls.append("setup")
            return super().setup(warmup_steps, *args, **kwargs)

        def sample(self, params):
            calls.append("sample")
            return super().sample(params)

        def cleanup(self):
            calls.append("cleanup")
            return super().cleanup()

    monkeypatch.setattr(pyro.infer.mcmc, "NUTS", RecordedNUTS)
    mutation = PyroNUTSMutation(
        target, n_steps=2, step_size=0.2, max_tree_depth=2
    )
    mutation.apply(
        ParticleState(torch.ones((3, 1), dtype=torch.float64)),
        StepContext(t=0),
        4,
    )
    assert calls == ["new", "setup", "sample", "sample", "cleanup"] * 3


def test_periodic_transform_rejected() -> None:
    from sequince.transforms import (
        CompositeTransform,
        PeriodicBoundaryTransform,
    )

    periodic = PeriodicBoundaryTransform({0: (0, 1)})
    for transform in (periodic, CompositeTransform([periodic])):
        with pytest.raises(ValueError, match="bijection"):
            PyroNUTSMutation(lambda x: -x.sum(), transform=transform)
