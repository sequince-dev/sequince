import numpy as np
import pytest

from sequince.transforms import (
    AffineTransform,
    BoundedToUnboundedTransform,
    BoundedToUnitHypercubeTransform,
    CompositeTransform,
    FlowTransform,
    PeriodicBoundaryTransform,
)


@pytest.fixture
def particles():
    return np.array(
        [
            [0.0, -1.0, 0.25],
            [1.0, 2.0, 0.5],
            [-2.0, 0.5, 0.9],
        ],
        dtype=float,
    )


def test_affine_all_columns(particles):
    transform = AffineTransform()
    transform.fit(particles)
    latent, log_det = transform.forward(particles)
    restored, restore_log_det = transform.inverse(latent)
    np.testing.assert_allclose(restored, particles)
    np.testing.assert_allclose(log_det + restore_log_det, 0.0)
    np.testing.assert_allclose(np.mean(latent, axis=0), 0.0, atol=1e-12)


def test_affine_subset(particles):
    transform = AffineTransform(columns=[0, 2])
    transform.fit(particles)
    latent, _ = transform.forward(particles)
    restored, _ = transform.inverse(latent)
    np.testing.assert_allclose(restored, particles)
    assert np.allclose(latent[..., 1], particles[..., 1])
    assert not np.allclose(latent[..., 0], particles[..., 0])


def test_bounded_to_unbounded_round_trip(particles):
    bounds = {2: (0.0, 1.0)}
    transform = BoundedToUnboundedTransform(bounds)
    latent, log_det = transform.forward(particles)
    restored, restore_log_det = transform.inverse(latent)
    np.testing.assert_allclose(restored, particles, atol=1e-12)
    np.testing.assert_allclose(log_det + restore_log_det, 0.0, atol=1e-10)


def test_bounded_to_unit_hypercube_round_trip(particles):
    bounded_particles = np.array(
        [
            [0.0, -1.0, 0.25],
            [1.0, 2.0, 0.5],
            [-0.5, 0.5, 0.9],
        ],
        dtype=float,
    )
    bounds = {0: (-1.0, 2.0), 2: (0.0, 1.0)}
    transform = BoundedToUnitHypercubeTransform(bounds)
    latent, log_det = transform.forward(bounded_particles)
    assert np.all(latent[..., 0] >= 0.0 - 1e-12)
    assert np.all(latent[..., 0] <= 1.0 + 1e-12)
    assert np.all(latent[..., 2] >= 0.0 - 1e-12)
    assert np.all(latent[..., 2] <= 1.0 + 1e-12)
    expected_log_det = -np.log(3.0)
    np.testing.assert_allclose(log_det, expected_log_det)
    restored, restore_log_det = transform.inverse(latent)
    np.testing.assert_allclose(restored, bounded_particles, atol=1e-12)
    np.testing.assert_allclose(log_det + restore_log_det, 0.0, atol=1e-12)


def test_periodic_boundary_transform(particles):
    bounds = {1: (-np.pi, np.pi)}
    transform = PeriodicBoundaryTransform(bounds)
    latent, log_det = transform.forward(particles)
    assert np.allclose(log_det, 0.0)
    assert np.all(latent[..., 1] >= -np.pi) and np.all(latent[..., 1] < np.pi)
    restored, _ = transform.inverse(latent)
    np.testing.assert_allclose(restored, latent)


def test_composite_transform(particles):
    bounds = {2: (0.0, 1.0)}
    composite = CompositeTransform(
        [
            AffineTransform(columns=[0]),
            BoundedToUnboundedTransform(bounds),
        ]
    )
    composite.fit(particles)
    latent, log_det = composite.forward(particles)
    restored, restore_log_det = composite.inverse(latent)
    np.testing.assert_allclose(restored, particles)
    np.testing.assert_allclose(log_det + restore_log_det, 0.0, atol=1e-10)


def test_flow_transform_round_trip_and_clone(particles):
    class ShiftFlow:
        xp = np

        def __init__(self, shift=1.5):
            self.shift = shift
            self.fitted_particles = None

        def fit(self, values, context=None):
            self.fitted_particles = np.asarray(values, dtype=float).copy()
            return self

        def forward(self, values, context=None):
            arr = np.asarray(values, dtype=float)
            return arr + self.shift, np.zeros(arr.shape[0], dtype=float)

        def inverse(self, values, context=None):
            arr = np.asarray(values, dtype=float)
            return arr - self.shift, np.zeros(arr.shape[0], dtype=float)

        def log_prob(self, values, context=None):
            arr = np.asarray(values, dtype=float)
            if arr.ndim == 1:
                arr = arr[:, np.newaxis]
            return -0.5 * np.sum(arr**2, axis=1)

        def sample(self, num_samples: int = 1, context=None):
            return np.full((num_samples, 3), self.shift, dtype=float)

    transform = FlowTransform(ShiftFlow())
    transform.fit(particles)
    latent, log_det = transform.forward(particles)
    restored, restore_log_det = transform.inverse(latent)

    np.testing.assert_allclose(restored, particles)
    np.testing.assert_allclose(log_det + restore_log_det, 0.0)
    np.testing.assert_allclose(
        getattr(transform.flow, "fitted_particles"), particles
    )

    cloned = transform.clone()
    assert isinstance(cloned, FlowTransform)
    assert cloned is not transform
    assert cloned.flow is not transform.flow


def test_flow_transform_passes_fit_kwargs(particles):
    class RecordingFlow:
        xp = np

        def __init__(self):
            self.fit_options = None

        def fit(self, values, context=None, **kwargs):
            self.fit_options = kwargs
            return self

        def forward(self, values, context=None):
            return values, np.zeros(values.shape[0])

        def inverse(self, values, context=None):
            return values, np.zeros(values.shape[0])

    transform = FlowTransform(
        RecordingFlow(),
        fit_kwargs={"n_epochs": 7, "batch_size": 2},
    )

    transform.fit(particles)

    assert transform.flow.fit_options == {"n_epochs": 7, "batch_size": 2}
    assert transform.fit_kwargs == {"n_epochs": 7, "batch_size": 2}


def test_flow_transform_preserves_input_namespace(particles):
    torch = pytest.importorskip("torch")
    torch_api = pytest.importorskip("array_api_compat.torch")

    class TorchFlow:
        xp = torch_api

        def fit(self, values, context=None, **kwargs):
            return self

        def forward(self, values, context=None):
            values = torch.asarray(values)
            return values + 1.0, torch.zeros(values.shape[0])

        def inverse(self, values, context=None):
            values = torch.asarray(values)
            return values - 1.0, torch.zeros(values.shape[0])

    transform = CompositeTransform(
        [AffineTransform(), FlowTransform(TorchFlow())]
    )
    transform.fit(particles)

    latent, log_det = transform.forward(particles)
    restored, inverse_log_det = transform.inverse(latent)

    assert isinstance(latent, np.ndarray)
    assert isinstance(log_det, np.ndarray)
    np.testing.assert_allclose(restored, particles)
    np.testing.assert_allclose(log_det + inverse_log_det, 0.0)
