import numpy as np
import pytest

from sequince.transforms import (
    AffineTransform,
    BoundedToUnboundedTransform,
    BoundedToUnitHypercubeTransform,
    CompositeTransform,
    FlowTransform,
    IdentityTransform,
    PeriodicBoundaryTransform,
)


@pytest.mark.parametrize(
    "transform",
    [IdentityTransform(), PeriodicBoundaryTransform({1: (-np.pi, np.pi)})],
)
def test_transform_vector_log_det_is_scalar(transform):
    _, log_det = transform.forward(np.array([0.0, 1.0, 2.0]))
    assert np.ndim(log_det) == 0


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
    fitted = transform.fit(particles)
    assert fitted is not transform
    assert transform._mean is None
    assert transform._scale is None
    transform = fitted
    latent, log_det = transform.forward(particles)
    restored, restore_log_det = transform.inverse(latent)
    np.testing.assert_allclose(restored, particles)
    np.testing.assert_allclose(log_det + restore_log_det, 0.0)
    np.testing.assert_allclose(np.mean(latent, axis=0), 0.0, atol=1e-12)


def test_affine_subset(particles):
    transform = AffineTransform(columns=[0, 2])
    transform = transform.fit(particles)
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


def test_bounded_to_unbounded_inverse_is_finite_in_extreme_tails():
    transform = BoundedToUnboundedTransform(
        {0: (-5.0, 5.0), 1: (-5.0, 5.0)},
        eps=1e-6,
    )

    particles, log_det = transform.inverse(
        np.array([[-100.0, 100.0]], dtype=np.float32)
    )

    assert np.all(np.isfinite(particles))
    assert np.all(np.isfinite(log_det))
    assert np.all(particles > -5.0)
    assert np.all(particles < 5.0)


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
    fitted = composite.fit(particles)
    assert fitted is not composite
    composite = fitted
    latent, log_det = composite.forward(particles)
    restored, restore_log_det = composite.inverse(latent)
    np.testing.assert_allclose(restored, particles)
    np.testing.assert_allclose(log_det + restore_log_det, 0.0, atol=1e-10)


def test_flow_transform_round_trip_and_clone(particles, shift_flow_factory):
    transform = FlowTransform(shift_flow_factory(n_dim=3))
    fitted = transform.fit(particles)
    assert fitted is not transform
    assert transform.flow.fitted_particles is None
    transform = fitted
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

    restored_transform = FlowTransform(shift_flow_factory(n_dim=3))
    restored_transform.load_state_dict(transform.state_dict())
    np.testing.assert_allclose(
        restored_transform.flow.fitted_particles, particles
    )


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

    fitted = transform.fit(particles)

    assert transform.flow.fit_options is None
    assert fitted.flow.fit_options == {"n_epochs": 7, "batch_size": 2}
    assert fitted.fit_kwargs == {"n_epochs": 7, "batch_size": 2}


def test_flow_transform_preserves_input_namespace(particles):
    torch_api = pytest.importorskip("array_api_compat.torch")

    class TorchFlow:
        xp = torch_api

        def fit(self, values, context=None, **kwargs):
            return self

        def forward(self, values, context=None):
            values = self.xp.asarray(values)
            return values + 1.0, self.xp.zeros(values.shape[0])

        def inverse(self, values, context=None):
            values = self.xp.asarray(values)
            return values - 1.0, self.xp.zeros(values.shape[0])

    transform = CompositeTransform(
        [AffineTransform(), FlowTransform(TorchFlow())]
    )
    transform = transform.fit(particles)

    latent, log_det = transform.forward(particles)
    restored, inverse_log_det = transform.inverse(latent)

    assert isinstance(latent, np.ndarray)
    assert isinstance(log_det, np.ndarray)
    np.testing.assert_allclose(restored, particles, atol=1e-12)
    np.testing.assert_allclose(log_det + inverse_log_det, 0.0)


def test_builtin_transforms_are_jittable_dataclass_pytrees():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    particles = jnp.asarray([[0.2, 1.0], [0.8, 3.0]])
    transform = CompositeTransform(
        [
            BoundedToUnboundedTransform({0: (0.0, 1.0)}),
            AffineTransform(columns=[1]),
        ]
    )
    fitted = jax.jit(lambda current, values: current.fit(values))(
        transform, particles
    )
    latent, log_det = jax.jit(lambda current, values: current.forward(values))(
        fitted, particles
    )
    restored, inverse_log_det = jax.jit(
        lambda current, values: current.inverse(values)
    )(fitted, latent)

    np.testing.assert_allclose(restored, particles, atol=1e-6)
    np.testing.assert_allclose(log_det + inverse_log_det, 0.0, atol=1e-6)
