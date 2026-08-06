from __future__ import annotations

import numpy as np
import pytest

from sequince.array_api import namespace, to_numpy
from sequince.state import ParticleState
from sequince.steps.base import StepContext
from sequince.steps.mutation import (
    BlockedMetropolisHastingsMutation,
    MaskedGaussianRandomWalk,
    MetropolisBlock,
)


def test_blocked_metropolis_updates_each_block_and_records_diagnostics(
    rng,
) -> None:
    particles = np.zeros((16, 4))
    state = ParticleState(
        particles=particles,
        log_weights=np.full(16, -np.log(16)),
    )
    context = StepContext(t=0)
    mutation = BlockedMetropolisHastingsMutation(
        lambda values, inner_context: np.zeros(values.shape[0]),
        blocks=[
            MetropolisBlock(
                "intrinsic",
                MaskedGaussianRandomWalk((0, 1), scale=0.1),
                n_steps=2,
            ),
            MetropolisBlock(
                "extrinsic",
                MaskedGaussianRandomWalk((2, 3), scale=0.1),
                n_steps=3,
            ),
        ],
        store_chains=True,
    )

    updated, context, output = mutation(state, context, rng)

    assert np.any(updated.particles[:, :2] != particles[:, :2])
    assert np.any(updated.particles[:, 2:] != particles[:, 2:])
    assert updated.metadata["blocked_metropolis_hastings_acceptance"] == {
        "intrinsic": 1.0,
        "extrinsic": 1.0,
    }
    assert updated.metadata["blocked_metropolis_hastings_steps"] == {
        "intrinsic": 2,
        "extrinsic": 3,
    }
    assert updated.metadata["blocked_metropolis_hastings_chains"][
        "intrinsic"
    ].shape == (2, 16, 4)
    assert updated.metadata["blocked_metropolis_hastings_chains"][
        "extrinsic"
    ].shape == (3, 16, 4)
    assert context.metadata["blocked_metropolis_hastings_acceptance"] == {
        "intrinsic": 1.0,
        "extrinsic": 1.0,
    }
    assert (
        output["acceptance"]
        == context.metadata["blocked_metropolis_hastings_acceptance"]
    )


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax", "cupy"])
def test_masked_gaussian_preserves_backend(
    backend: str,
    backend_factory,
) -> None:
    case = backend_factory(backend)
    particles = case.particles
    kernel = MaskedGaussianRandomWalk(indices=(1,), scale=0.1)

    proposals = kernel.sample(
        particles,
        case.rng(4321),
    )

    assert namespace(proposals) is namespace(particles)
    assert np.array_equal(to_numpy(proposals)[:, 0], to_numpy(particles)[:, 0])


def test_masked_gaussian_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="active indices"):
        MaskedGaussianRandomWalk(indices=())
    with pytest.raises(ValueError, match="unique"):
        MaskedGaussianRandomWalk(indices=(0, 0))
    with pytest.raises(ValueError, match="positive"):
        MaskedGaussianRandomWalk(indices=(0,), scale=-1.0)
    with pytest.raises(ValueError, match="shape"):
        MaskedGaussianRandomWalk(indices=(0, 1), covariance=np.eye(3))


def test_blocked_metropolis_requires_unique_block_names() -> None:
    block = MetropolisBlock(
        "same",
        MaskedGaussianRandomWalk(indices=(0,)),
    )
    with pytest.raises(ValueError, match="names must be unique"):
        BlockedMetropolisHastingsMutation(
            lambda values: np.zeros(values.shape[0]),
            [block, block],
        )
