from __future__ import annotations

import numpy as np
import pytest

from sequince import DegenerateWeightsError
from sequince.array_api import namespace, to_numpy
from sequince.state import (
    ParticleState,
    PersistentParticleState,
    normalize_log_weights,
)
from sequince.target import TargetEvaluation


@pytest.mark.parametrize("shape", [(3, 1), (2,), (3, 3), ()])
def test_rejects_malformed_log_weights(shape: tuple[int, ...]) -> None:
    particles = np.zeros((3, 2))
    weights = np.zeros(shape)
    with pytest.raises(ValueError, match="Log weights must match"):
        ParticleState(particles, log_weights=weights)
    state = ParticleState(particles)
    with pytest.raises(ValueError, match="Log weights must match"):
        state.with_updates(log_weights=weights)
    assert state.log_weights.shape == (3,)


def test_normalize_log_weights():
    log_weights = np.log([1.0, 3.0]).tolist()

    normalized, log_norm = normalize_log_weights(log_weights)

    np.testing.assert_allclose(np.exp(normalized), [0.25, 0.75])
    assert np.isclose(log_norm, np.log(4.0))


def test_normalize_log_weights_can_be_jitted():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    normalized, log_norm = jax.jit(normalize_log_weights)(
        jnp.asarray([0.0, -1.0])
    )

    np.testing.assert_allclose(
        np.asarray(normalized),
        np.asarray([-0.3132617, -1.3132617]),
    )
    assert log_norm == pytest.approx(0.3132617)


@pytest.mark.parametrize(
    "log_weights",
    [
        [-np.inf, -np.inf],
        [0.0, np.nan],
        [0.0, np.inf],
    ],
)
def test_normalize_degenerate_log_weights_returns_nonfinite(log_weights):
    normalized, log_norm = normalize_log_weights(log_weights)

    assert not np.isfinite(log_norm)
    assert not np.all(np.isfinite(normalized))


def test_copy_creates_independent_state(particle_state: ParticleState) -> None:
    particle_state.metadata["nested"] = {"values": [1.0]}
    copied = particle_state.copy()
    copied.particles[0, 0] = 99.0
    copied.log_weights[0] = -5.0
    copied.initial_ancestor_indices[0] = 2
    copied.metadata["beta"] = 10.0
    copied.metadata["nested"]["values"].append(2.0)

    assert not np.array_equal(copied.particles, particle_state.particles)
    assert not np.array_equal(copied.log_weights, particle_state.log_weights)
    np.testing.assert_array_equal(
        particle_state.initial_ancestor_indices,
        np.arange(particle_state.particles.shape[0]),
    )
    assert particle_state.metadata["beta"] == 0.0
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
        pytest.approx(normalized.log_normalizing_constant)
        == particle_state.log_normalizing_constant
    )
    assert normalized.log_evidence == normalized.log_normalizing_constant

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
    payload = resampled.to_dict()
    assert "log_normalizing_constant" in payload
    assert "log_evidence" not in payload
    restored = ParticleState.from_dict(payload)

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
    updated = particle_state.with_updates(log_normalizing_constant=5.0)
    assert updated.log_normalizing_constant == 5.0
    assert updated.log_evidence == 5.0
    assert particle_state.log_normalizing_constant == 0.0


def test_constructor_accepts_log_normalizing_constant() -> None:
    state = ParticleState(
        particles=np.zeros((2, 1)),
        log_normalizing_constant=1.25,
    )

    assert state.log_normalizing_constant == pytest.approx(1.25)
    assert state.log_evidence == pytest.approx(1.25)


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
    assert resampled.metadata.get("beta") == particle_state.metadata.get(
        "beta"
    )
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


def _persistent_batch(value: float, n: int = 3) -> ParticleState:
    particles = np.full((n, 1), value)
    evaluation = TargetEvaluation(
        components={"log_likelihood": np.full(n, 10.0 + value)},
        target_id="target",
        evaluation_key="stage",
    )
    return ParticleState(
        particles=particles,
        evaluation=evaluation,
        metadata={"source": "active"},
    )


def test_persistent_state_constructs_one_batch() -> None:
    state = _persistent_batch(1.0)

    history = PersistentParticleState.from_state(
        state,
        beta=0.25,
        log_normalizing_constant=1.5,
    )

    assert len(history) == 3
    assert history.n_particles == 3
    assert history.n_batches == 1
    assert history.batch_size == 3
    np.testing.assert_array_equal(history.particles, state.particles)
    np.testing.assert_allclose(history.betas, [0.25])
    np.testing.assert_allclose(history.log_evidences, [1.5])
    np.testing.assert_allclose(history.log_normalizing_constants, [1.5])
    assert history.log_evidence == pytest.approx(1.5)
    assert history.log_normalizing_constant == pytest.approx(1.5)
    np.testing.assert_allclose(history.log_weights, -np.log(3))
    assert history.evaluation is state.evaluation


def test_persistent_state_appends_multiple_batches_in_order() -> None:
    first = _persistent_batch(0.0)
    history = PersistentParticleState.from_state(
        first,
        beta=0.0,
        log_normalizing_constant=0.0,
    )

    two_batches = history.append(
        _persistent_batch(1.0),
        beta=0.0,
        log_normalizing_constant=0.4,
    )
    three_batches = two_batches.append(
        _persistent_batch(2.0),
        beta=0.25,
        log_normalizing_constant=1.1,
    )

    assert history.n_batches == 1
    assert len(history) == 3
    assert two_batches.n_batches == 2
    assert len(two_batches) == 6
    assert three_batches.n_batches == 3
    assert len(three_batches) == 9
    np.testing.assert_array_equal(
        three_batches.particles[:, 0],
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
    )
    np.testing.assert_allclose(three_batches.betas, [0.0, 0.0, 0.25])
    np.testing.assert_allclose(three_batches.log_evidences, [0.0, 0.4, 1.1])
    assert three_batches.log_evidence == pytest.approx(1.1)
    np.testing.assert_allclose(three_batches.log_weights, -np.log(9))


def test_persistent_state_preserves_evaluation_alignment() -> None:
    history = PersistentParticleState.from_state(
        _persistent_batch(0.0),
        beta=0.0,
        log_normalizing_constant=0.0,
    ).append(
        _persistent_batch(2.0),
        beta=0.5,
        log_normalizing_constant=0.7,
    )

    assert history.evaluation is not None
    np.testing.assert_array_equal(
        history.evaluation["log_likelihood"],
        [10.0, 10.0, 10.0, 12.0, 12.0, 12.0],
    )
    np.testing.assert_array_equal(
        history.evaluation["log_likelihood"],
        10.0 + history.particles[:, 0],
    )


def test_persistent_state_rejects_different_batch_size() -> None:
    history = PersistentParticleState.from_state(
        _persistent_batch(0.0),
        beta=0.0,
        log_normalizing_constant=0.0,
    )

    with pytest.raises(ValueError, match="must match batch_size"):
        history.append(
            _persistent_batch(1.0, n=2),
            beta=0.5,
            log_normalizing_constant=0.2,
        )


@pytest.mark.parametrize("history_has_evaluation", [False, True])
def test_persistent_state_rejects_inconsistent_evaluations(
    history_has_evaluation: bool,
) -> None:
    evaluated = _persistent_batch(0.0)
    unevaluated = ParticleState(particles=np.zeros((3, 1)))
    history_state = evaluated if history_has_evaluation else unevaluated
    appended_state = unevaluated if history_has_evaluation else evaluated
    history = PersistentParticleState.from_state(
        history_state,
        beta=0.0,
        log_normalizing_constant=0.0,
    )

    with pytest.raises(ValueError, match="inconsistent target evaluations"):
        history.append(
            appended_state,
            beta=0.5,
            log_normalizing_constant=0.2,
        )


def test_persistent_state_append_does_not_mutate_history() -> None:
    history = PersistentParticleState.from_state(
        _persistent_batch(0.0),
        beta=0.0,
        log_normalizing_constant=0.0,
    )
    original_particles = history.particles.copy()
    original_betas = history.betas.copy()
    original_evidences = history.log_evidences.copy()

    appended = history.append(
        _persistent_batch(1.0),
        beta=0.5,
        log_normalizing_constant=0.2,
    )

    assert appended is not history
    np.testing.assert_array_equal(history.particles, original_particles)
    np.testing.assert_array_equal(history.betas, original_betas)
    np.testing.assert_array_equal(history.log_evidences, original_evidences)


@pytest.mark.parametrize(
    "field",
    [
        "particles",
        "initial_ancestor_indices",
        "betas",
        "log_normalizing_constants",
        "batch_size",
    ],
)
def test_persistent_state_with_updates_rejects_structural_fields(
    field: str,
) -> None:
    history = PersistentParticleState.from_state(
        _persistent_batch(0.0),
        beta=0.0,
        log_normalizing_constant=0.0,
    )

    with pytest.raises(ValueError, match="Persistent structure cannot"):
        history.with_updates(**{field: 1.0})


def test_persistent_state_scalar_log_evidence_can_differ_from_history() -> (
    None
):
    history = PersistentParticleState.from_state(
        _persistent_batch(0.0),
        beta=0.0,
        log_normalizing_constant=0.7,
    )

    weighted = history.with_updates(log_normalizing_constant=2.0)

    assert history.log_evidence == pytest.approx(0.7)
    assert weighted.log_evidence == pytest.approx(2.0)
    np.testing.assert_allclose(weighted.log_evidences, [0.7])


def test_persistent_state_constructor_accepts_scalar_normalizing_constant() -> (
    None
):
    history = PersistentParticleState(
        particles=np.zeros((2, 1)),
        betas=[0.0],
        log_normalizing_constants=[0.7],
        batch_size=2,
        log_normalizing_constant=2.0,
    )

    assert history.log_evidence == pytest.approx(2.0)
    np.testing.assert_allclose(history.log_evidences, [0.7])


def test_persistent_state_with_updates_accepts_log_weights() -> None:
    history = PersistentParticleState.from_state(
        _persistent_batch(0.0),
        beta=0.0,
        log_normalizing_constant=0.0,
    )

    updated = history.with_updates(log_weights=np.array([0.0, -1.0, -2.0]))

    assert type(updated) is PersistentParticleState
    np.testing.assert_allclose(updated.log_weights, [0.0, -1.0, -2.0])
    np.testing.assert_allclose(history.log_weights, -np.log(3))


def test_normalizing_persistent_state_preserves_type_and_structure() -> None:
    history = PersistentParticleState.from_state(
        _persistent_batch(0.0),
        beta=0.0,
        log_normalizing_constant=0.4,
    ).with_updates(log_weights=np.array([0.0, -1.0, -2.0]))

    normalized, log_norm = history.normalize()

    assert type(normalized) is PersistentParticleState
    assert log_norm == pytest.approx(np.log(np.exp([0.0, -1.0, -2.0]).sum()))
    np.testing.assert_allclose(np.exp(normalized.log_weights).sum(), 1.0)
    np.testing.assert_allclose(normalized.betas, history.betas)
    np.testing.assert_allclose(normalized.log_evidences, history.log_evidences)
    assert normalized.batch_size == history.batch_size
    assert normalized.log_evidence == pytest.approx(0.4)


def test_persistent_state_concatenates_shared_initial_ancestry() -> None:
    first = ParticleState(
        particles=np.zeros((3, 1)),
        initial_ancestor_indices=np.array([2, 0, 1]),
    )
    second = ParticleState(
        particles=np.ones((3, 1)),
        initial_ancestor_indices=np.array([1, 1, 0]),
    )
    history = PersistentParticleState.from_state(
        first,
        beta=0.0,
        log_normalizing_constant=0.0,
    ).append(
        second,
        beta=0.5,
        log_normalizing_constant=0.2,
    )

    np.testing.assert_array_equal(
        history.initial_ancestor_indices,
        [2, 0, 1, 1, 1, 0],
    )
    resampled = history.resample(np.array([3, 0]))
    np.testing.assert_array_equal(resampled.initial_ancestor_indices, [1, 2])


def test_resampling_persistent_state_returns_active_state() -> None:
    history = PersistentParticleState.from_state(
        _persistent_batch(0.0),
        beta=0.0,
        log_normalizing_constant=0.0,
    ).append(
        _persistent_batch(2.0),
        beta=0.5,
        log_normalizing_constant=0.7,
    )

    resampled = history.resample(np.array([5, 0, 3]))

    assert type(resampled) is ParticleState
    assert len(resampled.particles) == 3
    np.testing.assert_array_equal(resampled.particles[:, 0], [2.0, 0.0, 2.0])
    np.testing.assert_allclose(resampled.log_weights, -np.log(3))
    assert resampled.evaluation is not None
    np.testing.assert_array_equal(
        resampled.evaluation["log_likelihood"],
        [12.0, 10.0, 12.0],
    )


@pytest.mark.parametrize(
    (
        "betas",
        "log_normalizing_constants",
        "batch_size",
        "match",
    ),
    [
        ([0.0], [0.0], 3, "Particles array cannot be empty"),
        ([0.0], [0.0], 0, "must be positive"),
        ([[0.0]], [0.0], 3, "betas must be one-dimensional"),
        (
            [0.0],
            [[0.0]],
            3,
            "log_normalizing_constants must be one-dimensional",
        ),
        ([0.0, 0.5], [0.0], 3, "must have the same length"),
        ([0.0], [0.0], 2, "must be divisible"),
        ([0.0, 0.5], [0.0, 0.2], 3, "one entry per particle batch"),
    ],
)
def test_persistent_state_validates_structure(
    betas,
    log_normalizing_constants,
    batch_size: int,
    match: str,
) -> None:
    n_particles = 0 if match == "Particles array cannot be empty" else 3
    with pytest.raises(ValueError, match=match):
        PersistentParticleState(
            particles=np.zeros((n_particles, 1)),
            betas=np.asarray(betas),
            log_normalizing_constants=np.asarray(log_normalizing_constants),
            batch_size=batch_size,
        )


def test_persistent_state_integer_history_still_has_floating_weights() -> None:
    history = PersistentParticleState(
        particles=np.zeros((3, 1)),
        betas=np.array([0]),
        log_normalizing_constants=np.array([0]),
        batch_size=3,
    )

    assert np.issubdtype(history.log_weights.dtype, np.floating)
    np.testing.assert_allclose(history.log_weights, -np.log(3))


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax", "cupy"])
def test_persistent_state_preserves_array_backend(
    backend: str,
    backend_factory,
) -> None:
    case = backend_factory(backend)
    xp = case.xp
    first = case.state()
    second = ParticleState(
        particles=case.particles + 10.0,
        log_weights=case.log_weights,
    )

    history = PersistentParticleState.from_state(
        first,
        beta=0.0,
        log_normalizing_constant=0.0,
    ).append(
        second,
        beta=0.5,
        log_normalizing_constant=0.8,
    )

    assert namespace(history.particles) is xp
    assert namespace(history.log_weights) is xp
    assert namespace(history.betas) is xp
    assert namespace(history.log_evidences) is xp
    assert history.particles.shape[0] == 2 * case.particles.shape[0]
    np.testing.assert_allclose(
        to_numpy(history.log_weights),
        -np.log(history.n_particles),
    )
    np.testing.assert_allclose(to_numpy(history.betas), [0.0, 0.5])
    np.testing.assert_allclose(to_numpy(history.log_evidences), [0.0, 0.8])


def test_persistent_state_accepts_absent_evaluations() -> None:
    state = ParticleState(particles=np.zeros((2, 1)))
    history = PersistentParticleState.from_state(
        state,
        beta=0.0,
        log_normalizing_constant=0.0,
    ).append(
        state.with_updates(particles=np.ones((2, 1))),
        beta=0.5,
        log_normalizing_constant=0.3,
    )

    assert history.evaluation is None
    np.testing.assert_array_equal(
        history.particles[:, 0], [0.0, 0.0, 1.0, 1.0]
    )


def test_persistent_state_supports_jax_pytree_round_trip() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    history = PersistentParticleState.from_state(
        ParticleState(particles=jnp.zeros((2, 1))),
        beta=0.0,
        log_normalizing_constant=0.0,
    ).with_updates(log_weights=jnp.asarray([-0.25, -1.5]))

    leaves, structure = jax.tree.flatten(history)
    restored = jax.tree.unflatten(structure, leaves)

    assert type(restored) is PersistentParticleState
    assert restored.n_batches == 1
    np.testing.assert_allclose(to_numpy(restored.log_weights), [-0.25, -1.5])
