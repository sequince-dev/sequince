from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from orng import RandomGenerator

from sequince import ParticleState
from sequince.steps import ProposalRejuvenation
from sequince.steps.base import StepContext
from sequince.steps.mutation import GenerativeProposal
from sequince.transforms import ParticleTransform


class RecordingTransform(ParticleTransform):
    def __init__(self) -> None:
        self.fit_rng = None

    def fit(self, particles, context=None, *, rng=None):
        del particles, context
        self.fit_rng = rng
        return self


class EmpiricalGaussianProposal:
    """Minimal non-GMM proposal with no diagnostics or checkpoint hooks."""

    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.fit_rng = None

    def fit(self, particles, context=None, *, rng=None):
        del context
        self.fit_rng = rng
        values = np.asarray(particles)
        self.mean = np.mean(values, axis=0)
        self.scale = np.std(values, axis=0) + 1e-6
        return self

    def sample(self, n_samples, context=None, *, rng=None):
        del context
        if rng is None or self.mean is None or self.scale is None:
            raise RuntimeError("Proposal must be fitted and supplied an RNG.")
        return self.mean + self.scale * np.asarray(
            rng.normal(size=(n_samples, self.mean.shape[0]))
        )


def test_rejuvenates_with_minimal_generative_proposal(
    standard_normal_posterior,
) -> None:
    particles = np.linspace(-2.0, 2.0, 100)[:, None]
    context = StepContext(t=2, metadata={"temperature": 0.5})
    state = ParticleState(
        particles,
        log_weights=np.linspace(-2.0, 0.0, len(particles)),
        log_evidence=1.25,
    )
    proposal = EmpiricalGaussianProposal()
    transform = RecordingTransform()
    step = ProposalRejuvenation(
        standard_normal_posterior, proposal, transform=transform
    )

    rng = RandomGenerator(seed=7)
    updated, returned_context, output = step.apply(state, context, rng)

    assert returned_context is context
    assert updated.particles.shape == particles.shape
    assert not np.array_equal(updated.particles, particles)
    assert updated.log_weights == pytest.approx(-np.log(len(particles)))
    assert updated.log_evidence == pytest.approx(1.25)
    assert updated.evaluation is not None
    assert output == {
        "proposal": "EmpiricalGaussianProposal",
        "approximate": True,
    }
    assert step.state_dict() == {}
    assert proposal.fit_rng is rng
    assert transform.fit_rng is rng


def test_requires_fit_and_sample(standard_normal_posterior) -> None:
    with pytest.raises(TypeError, match=r"fit\(\) and RNG-aware sample"):
        ProposalRejuvenation(
            standard_normal_posterior,
            cast(GenerativeProposal, object()),
        )
