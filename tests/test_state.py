from __future__ import annotations

import numpy as np
import pytest

from sequince.state import ParticleState


def test_copy_creates_independent_state(particle_state: ParticleState) -> None:
    particle_state.metadata["nested"] = {"values": [1.0]}
    copied = particle_state.copy()
    copied.particles[0, 0] = 99.0
    copied.log_weights[0] = -5.0
    copied.metadata["temperature"] = 10.0
    copied.metadata["nested"]["values"].append(2.0)

    assert not np.array_equal(copied.particles, particle_state.particles)
    assert not np.array_equal(copied.log_weights, particle_state.log_weights)
    assert particle_state.metadata["temperature"] == 0.0
    assert particle_state.metadata["nested"] == {"values": [1.0]}


def test_normalize_updates_log_weights_and_evidence(
    particle_state: ParticleState,
) -> None:
    normalized, log_norm = particle_state.normalize()

    expected_log_norm = np.log(np.sum(np.exp(particle_state.log_weights)))
    assert pytest.approx(log_norm) == expected_log_norm
    assert (
        pytest.approx(normalized.log_evidence)
        == particle_state.log_evidence + log_norm
    )

    weights = np.exp(normalized.log_weights)
    assert pytest.approx(np.sum(weights)) == 1.0


def test_weights_handles_non_finite_log_norm() -> None:
    log_weights = np.array([-np.inf, -np.inf, -np.inf])
    state = ParticleState(particles=np.zeros((3, 1)), log_weights=log_weights)
    weights = state.weights
    assert np.allclose(weights, np.full(3, 1 / 3))


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


def test_resample_returns_equal_weights_and_track_ancestors(
    particle_state: ParticleState, rng
) -> None:
    n = len(particle_state.log_weights)
    indices = np.asarray(rng.choice(n, size=n, replace=True))
    resampled = particle_state.resample(indices)

    assert np.allclose(
        resampled.log_weights, np.full(len(indices), -np.log(len(indices)))
    )
    assert np.array_equal(resampled.metadata["ancestors"], indices)
    assert resampled.metadata.get(
        "temperature"
    ) == particle_state.metadata.get("temperature")
    assert resampled.log_evidence == particle_state.log_evidence
