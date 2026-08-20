from __future__ import annotations

import numpy as np
import pytest

from sequince import DegenerateWeightsError
from sequince.state import ParticleState
from sequince.target import TargetEvaluation


def test_copy_creates_independent_state(particle_state: ParticleState) -> None:
    particle_state.metadata["nested"] = {"values": [1.0]}
    copied = particle_state.copy()
    copied.particles[0, 0] = 99.0
    copied.log_weights[0] = -5.0
    copied.initial_ancestor_indices[0] = 2
    copied.metadata["temperature"] = 10.0
    copied.metadata["nested"]["values"].append(2.0)

    assert not np.array_equal(copied.particles, particle_state.particles)
    assert not np.array_equal(copied.log_weights, particle_state.log_weights)
    np.testing.assert_array_equal(
        particle_state.initial_ancestor_indices,
        np.arange(particle_state.particles.shape[0]),
    )
    assert particle_state.metadata["temperature"] == 0.0
    assert particle_state.metadata["nested"] == {"values": [1.0]}


def test_copy_handles_non_leaf_torch_metadata() -> None:
    torch = pytest.importorskip("torch")
    value = torch.tensor(0.0, requires_grad=True).exp()
    state = ParticleState(
        particles=torch.zeros((2, 1)),
        metadata={"nested": [value]},
    )

    copied = state.copy()

    assert torch.equal(copied.metadata["nested"][0], value)
    assert copied.metadata["nested"][0] is not value


def test_normalize_updates_log_weights(
    particle_state: ParticleState,
) -> None:
    normalized, log_norm = particle_state.normalize()

    expected_log_norm = np.log(np.sum(np.exp(particle_state.log_weights)))
    assert pytest.approx(log_norm) == expected_log_norm
    assert (
        pytest.approx(normalized.log_evidence) == particle_state.log_evidence
    )

    weights = np.exp(normalized.log_weights)
    assert pytest.approx(np.sum(weights)) == 1.0


def test_weights_rejects_non_finite_log_norm() -> None:
    log_weights = np.array([-np.inf, -np.inf, -np.inf])
    state = ParticleState(particles=np.zeros((3, 1)), log_weights=log_weights)

    with pytest.raises(DegenerateWeightsError, match="zero weight"):
        _ = state.weights
    with pytest.raises(DegenerateWeightsError, match="zero weight"):
        state.normalize()


def test_state_resamples_and_serializes_target_evaluation() -> None:
    evaluation = TargetEvaluation(
        components={
            "log_prior": np.array([1.0, 2.0, 3.0]),
            "log_likelihood": np.array([4.0, 5.0, 6.0]),
        }
    )
    state = ParticleState(
        particles=np.array([[1.0], [2.0], [3.0]]),
        evaluation=evaluation,
    )

    resampled = state.resample(np.array([2, 0, 2]))
    restored = ParticleState.from_dict(resampled.to_dict())

    np.testing.assert_allclose(restored.particles[:, 0], [3.0, 1.0, 3.0])
    np.testing.assert_array_equal(restored.initial_ancestor_indices, [2, 0, 2])
    assert restored.evaluation is not None
    np.testing.assert_allclose(
        restored.evaluation["log_likelihood"],
        [6.0, 4.0, 6.0],
    )


def test_effective_sample_size_matches_expected(
    particle_state: ParticleState,
) -> None:
    state = particle_state.normalize()[0]
    ess = state.effective_sample_size()
    weights = state.weights
    expected_ess = 1.0 / np.sum(np.square(weights))
    assert pytest.approx(ess) == expected_ess


def test_with_updates_returns_new_state(particle_state: ParticleState) -> None:
    updated = particle_state.with_updates(log_evidence=5.0)
    assert updated.log_evidence == 5.0
    assert particle_state.log_evidence == 0.0


def test_resample_returns_equal_weights_and_preserves_metadata(
    particle_state: ParticleState, rng
) -> None:
    n = len(particle_state.log_weights)
    indices = np.asarray(rng.choice(n, size=n, replace=True))
    resampled = particle_state.resample(indices)

    assert np.allclose(
        resampled.log_weights, np.full(len(indices), -np.log(len(indices)))
    )
    assert "ancestors" not in resampled.metadata
    assert resampled.metadata.get(
        "temperature"
    ) == particle_state.metadata.get("temperature")
    assert resampled.log_evidence == particle_state.log_evidence


def test_resample_composes_initial_ancestor_indices() -> None:
    state = ParticleState(particles=np.zeros((4, 1)))

    first = state.resample(np.array([0, 0, 2, 3]))
    second = first.resample(np.array([0, 1, 1, 3]))

    np.testing.assert_array_equal(first.initial_ancestor_indices, [0, 0, 2, 3])
    np.testing.assert_array_equal(
        second.initial_ancestor_indices, [0, 0, 0, 3]
    )


def test_initial_ancestor_indices_must_match_particles() -> None:
    with pytest.raises(ValueError, match="must match the particle population"):
        ParticleState(
            particles=np.zeros((4, 1)),
            initial_ancestor_indices=np.arange(3),
        )
