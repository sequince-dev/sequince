"""Tests for particle resampling steps."""

from __future__ import annotations

import numpy as np
import pytest

from sequince.array_api import asarray, get_device, namespace
from sequince.state import ParticleState
from sequince.steps.resampling import (
    MultinomialResampling,
    StratifiedResampling,
    SystematicResampling,
)
from sequince.target import TargetEvaluation
from sequince.utils.random import resolve_rng


@pytest.mark.parametrize(
    ("step_type", "seed"),
    [(SystematicResampling, 82), (StratifiedResampling, 93)],
)
def test_float32_resampling_roundoff(step_type, seed, context_factory) -> None:
    n = 1000
    state = ParticleState(
        np.arange(n, dtype=np.float32)[:, None],
        log_weights=np.full(n, -np.log(n), dtype=np.float32),
    )
    updated, _, output = step_type(always_resample=True)(
        state, context_factory(), resolve_rng(seed)
    )
    assert updated.particles.shape == (n, 1)
    assert np.all((output.ancestors >= 0) & (output.ancestors < n))


@pytest.mark.parametrize(
    "step_type", [SystematicResampling, StratifiedResampling]
)
@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_rounded_endpoint_does_not_select_zero_weight_particle(
    step_type, backend: str, dtype: str, backend_factory
) -> None:
    case = backend_factory(backend)
    xp = case.xp
    probs = asarray(
        [0.25, 0.75, 0.0, 0.0],
        xp=xp,
        dtype=getattr(xp, dtype),
        like=case.particles,
    )

    class EndpointRNG:
        def random(self, size=()):
            return xp.full(
                size,
                1.0 - xp.finfo(probs.dtype).eps / 2,
                dtype=probs.dtype,
                device=get_device(probs),
            )

    ancestors = step_type()._choose_ancestors(probs, 4, EndpointRNG())
    assert namespace(ancestors) is namespace(probs)
    assert get_device(ancestors) == get_device(probs)
    assert bool(xp.all((ancestors >= 0) & (ancestors < 2)))


def test_systematic_resampling_skips_when_not_required(
    particle_state: ParticleState, context_factory, rng
) -> None:
    normalized, _ = particle_state.normalize()
    context = context_factory()
    step = SystematicResampling(ess_threshold=0.2, always_resample=False)

    result, context, output = step(normalized, context, rng)

    np.testing.assert_array_equal(result.particles, normalized.particles)
    np.testing.assert_array_equal(result.log_weights, normalized.log_weights)
    np.testing.assert_array_equal(
        output["ancestors"], np.arange(normalized.particles.shape[0])
    )
    assert output["resampled"] is False


@pytest.mark.parametrize(
    ("resample_at_end", "expected_resampled"),
    [(True, True), (False, False)],
)
def test_systematic_resampling_respects_resample_at_end(
    particle_state: ParticleState,
    context_factory,
    rng,
    resample_at_end: bool,
    expected_resampled: bool,
) -> None:
    context = context_factory().request_stop()
    step = SystematicResampling(
        ess_threshold=None,
        always_resample=False,
        resample_at_end=resample_at_end,
    )

    result, context, output = step(particle_state, context, rng)

    assert context.stop_requested
    assert context.metadata["resampled"] is expected_resampled
    assert output["resampled"] is expected_resampled
    if expected_resampled:
        np.testing.assert_allclose(
            result.log_weights,
            np.full_like(result.log_weights, -np.log(result.log_weights.size)),
        )
    else:
        np.testing.assert_array_equal(
            result.particles, particle_state.particles
        )
        np.testing.assert_array_equal(
            result.log_weights, particle_state.log_weights
        )
        np.testing.assert_array_equal(
            output["ancestors"], np.arange(particle_state.particles.shape[0])
        )


@pytest.mark.parametrize(
    "step_type",
    [MultinomialResampling, StratifiedResampling, SystematicResampling],
)
def test_resampling_selects_particles_by_weight(
    step_type, context_factory, rng
) -> None:
    probabilities = np.array([0.0, 0.25, 0.75, 0.0])
    with np.errstate(divide="ignore"):
        state = ParticleState(
            np.arange(4.0)[:, None], log_weights=np.log(probabilities)
        )
    n = 4096 if step_type is MultinomialResampling else 8

    resampled, _, output = step_type(always_resample=True, n_samples=n)(
        state, context_factory(), rng
    )

    ancestors = output["ancestors"]
    counts = np.bincount(ancestors, minlength=4)
    assert counts[0] == counts[3] == 0
    if step_type is MultinomialResampling:
        # Hoeffding + union bound: P(any error > .05) < 2e-8 for 4 bins.
        np.testing.assert_allclose(
            counts / n, probabilities, atol=0.05, rtol=0
        )
    else:
        # Each CDF interval contains an integer number of strata.
        np.testing.assert_array_equal(counts, [0, 2, 6, 0])
    np.testing.assert_array_equal(
        resampled.particles, state.particles[ancestors]
    )
    np.testing.assert_allclose(resampled.log_weights, -np.log(n))
    assert "ancestors" not in resampled.metadata
    assert output["resampled"] is True
    assert output["resampling_scheme"] == step_type.resampling_scheme


@pytest.mark.parametrize(
    "step_type",
    [MultinomialResampling, StratifiedResampling, SystematicResampling],
)
@pytest.mark.parametrize(
    ("n_particles", "n_samples", "expected_size"),
    [(12, None, 12), (100, 25, 25), (20, 50, 50)],
)
def test_resampling_supports_different_output_population_sizes(
    step_type,
    n_particles: int,
    n_samples: int | None,
    expected_size: int,
    context_factory,
    rng,
) -> None:
    particles = np.column_stack(
        (np.arange(n_particles), -np.arange(n_particles))
    )
    evaluation = TargetEvaluation(
        components={"cached": np.arange(n_particles) + 1000}
    )
    state = ParticleState(
        particles=particles,
        log_weights=np.linspace(-2.0, 0.0, n_particles),
        evaluation=evaluation,
    )
    kwargs = {} if n_samples is None else {"n_samples": n_samples}

    updated, _, output = step_type(always_resample=True, **kwargs)(
        state, context_factory(), rng
    )

    ancestors = output["ancestors"]
    assert updated.particles.shape == (expected_size, 2)
    assert updated.log_weights.shape == (expected_size,)
    assert ancestors.shape == (expected_size,)
    np.testing.assert_array_equal(updated.particles, particles[ancestors])
    np.testing.assert_array_equal(updated.initial_ancestor_indices, ancestors)
    assert updated.evaluation is not None
    np.testing.assert_array_equal(
        updated.evaluation["cached"], evaluation["cached"][ancestors]
    )
    np.testing.assert_allclose(updated.log_weights, -np.log(expected_size))


@pytest.mark.parametrize(
    ("n_samples", "error"),
    [
        (0, ValueError),
        (-1, ValueError),
        (1.5, TypeError),
        ("2", TypeError),
        (True, TypeError),
    ],
)
def test_resampling_rejects_invalid_n_samples(n_samples, error) -> None:
    with pytest.raises(error, match="n_samples must be a positive integer"):
        SystematicResampling(n_samples=n_samples)


def test_skipped_resampling_outputs_identity_ancestors(
    particle_state: ParticleState, context_factory, rng
) -> None:
    state = particle_state
    step = MultinomialResampling(ess_threshold=None)

    updated, context, output = step(state, context_factory(), rng)

    np.testing.assert_array_equal(
        output["ancestors"], np.arange(state.particles.shape[0])
    )
    assert "ancestors" not in updated.metadata
    np.testing.assert_array_equal(updated.particles, state.particles)
    np.testing.assert_array_equal(updated.log_weights, state.log_weights)
    assert context.metadata["resampled"] is False
    assert output["resampled"] is False
    assert output["resampling_scheme"] == "multinomial"


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax", "cupy"])
@pytest.mark.parametrize(
    "step_type",
    [MultinomialResampling, StratifiedResampling, SystematicResampling],
)
def test_resampling_preserves_array_backend(
    backend: str,
    step_type,
    backend_factory,
    context_factory,
) -> None:
    case = backend_factory(backend)
    particles = case.particles
    backend_ns = namespace(particles)

    updated, _, output = step_type(always_resample=True, n_samples=6)(
        case.state(),
        context_factory(),
        case.rng(1234),
    )

    assert updated.particles.shape == (6, particles.shape[1])
    assert namespace(updated.particles) is backend_ns
    assert namespace(updated.log_weights) is backend_ns
    assert namespace(output["ancestors"]) is backend_ns
    assert output["resampling_scheme"] == step_type.resampling_scheme
