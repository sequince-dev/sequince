import numpy as np
import pytest

from sequince.array_api import namespace, to_numpy
from sequince.transforms import (
    AffineTransform,
    BoundedToUnboundedTransform,
    BoundedToUnitHypercubeTransform,
    CauchitMapping,
    CompositeTransform,
    FlowTransform,
    IdentityTransform,
    LogitMapping,
    PeriodicBoundaryTransform,
    ProbitMapping,
)


@pytest.fixture
def particles(backend, backend_factory):
    case = backend_factory(backend)
    return case.xp.asarray(
        [
            [0.0, -1.0, 0.25],
            [1.0, 2.0, 0.5],
            [-0.5, 0.5, 0.9],
        ],
        dtype=case.particles.dtype,
    )


def assert_allclose(actual, expected, **kwargs):
    np.testing.assert_allclose(
        to_numpy(actual),
        to_numpy(expected),
        **kwargs,
    )


def assert_preserves_namespace(result, particles):
    assert namespace(result) is namespace(particles)


@pytest.mark.parametrize(
    "transform",
    [IdentityTransform(), PeriodicBoundaryTransform({1: (-np.pi, np.pi)})],
)
def test_transform_vector_log_det_is_scalar(transform):
    _, log_det = transform.forward(np.array([0.0, 1.0, 2.0]))
    assert np.ndim(log_det) == 0


def test_affine_all_columns(particles):
    xp = namespace(particles)
    transform = AffineTransform()
    fitted = transform.fit(particles)
    assert fitted is not transform
    assert transform._mean is None
    assert transform._scale is None
    transform = fitted
    latent, log_det = transform.forward(particles)
    restored, restore_log_det = transform.inverse(latent)
    assert_preserves_namespace(latent, particles)
    assert_preserves_namespace(log_det, particles)
    assert_allclose(restored, particles, atol=1e-6)
    assert_allclose(log_det + restore_log_det, 0.0, atol=1e-6)
    assert_allclose(xp.mean(latent, axis=0), 0.0, atol=1e-6)


def test_affine_subset(particles):
    transform = AffineTransform(columns=[0, 2])
    transform = transform.fit(particles)
    latent, _ = transform.forward(particles)
    restored, _ = transform.inverse(latent)
    assert_preserves_namespace(latent, particles)
    assert_allclose(restored, particles, atol=1e-6)
    assert_allclose(latent[..., 1], particles[..., 1])
    assert not np.allclose(
        to_numpy(latent[..., 0]), to_numpy(particles[..., 0])
    )


def test_bounded_to_unbounded_inverse_is_stable_in_extreme_tails():
    transform = BoundedToUnboundedTransform({0: (-5.0, 5.0), 1: (-5.0, 5.0)})

    particles, log_det = transform.inverse(
        np.array([[-100.0, 100.0]], dtype=np.float32)
    )

    assert np.all(np.isfinite(particles))
    assert np.all(np.isfinite(log_det))
    assert np.all(particles > -5.0)
    assert np.all(particles < 5.0)


@pytest.mark.parametrize(
    "mapping",
    [
        "cauchit",
        CauchitMapping(),
        "probit",
        ProbitMapping(),
        "logit",
        LogitMapping(),
    ],
)
def test_bounded_to_unbounded_round_trip(particles, mapping):
    transform = BoundedToUnboundedTransform({2: (0.0, 1.0)}, mapping=mapping)
    latent, log_det = transform.forward(particles)
    restored, restore_log_det = transform.inverse(latent)

    assert_preserves_namespace(latent, particles)
    assert_allclose(restored, particles, atol=1e-6)
    assert_allclose(log_det + restore_log_det, 0.0, atol=1e-5)


def test_bounded_to_unbounded_accepts_custom_mapping_strategy(particles):
    class ShiftedLogit:
        def __init__(self, shift):
            self.shift = shift
            self.base = LogitMapping()

        def forward(self, values):
            transformed, log_det = self.base.forward(values)
            return transformed + self.shift, log_det

        def inverse(self, values):
            return self.base.inverse(values - self.shift)

    transform = BoundedToUnboundedTransform(
        {2: (0.0, 1.0)}, mapping=ShiftedLogit(2.0)
    )
    latent, log_det = transform.forward(particles)
    restored, restore_log_det = transform.inverse(latent)

    assert_preserves_namespace(latent, particles)
    assert_allclose(restored, particles, atol=1e-6)
    assert_allclose(log_det + restore_log_det, 0.0, atol=1e-5)


def test_bounded_to_unbounded_rejects_unknown_mapping():
    with pytest.raises(ValueError, match="Unknown unit-interval mapping"):
        BoundedToUnboundedTransform({0: (0.0, 1.0)}, mapping="unknown")


@pytest.mark.parametrize(
    "mapping_cls", [LogitMapping, CauchitMapping, ProbitMapping]
)
@pytest.mark.parametrize("eps", [0.0, 0.5])
def test_unit_interval_mapping_validates_eps(mapping_cls, eps):
    with pytest.raises(ValueError, match="eps must lie"):
        mapping_cls(eps=eps)


@pytest.mark.parametrize(
    "mapping_cls", [LogitMapping, CauchitMapping, ProbitMapping]
)
def test_unit_interval_mapping_owns_eps(mapping_cls):
    mapping = mapping_cls(eps=0.1)
    transform = BoundedToUnboundedTransform({0: (0.0, 1.0)}, mapping=mapping)

    restored, log_det = transform.inverse(np.array([[-1e6]]))

    assert transform.mapping is mapping
    np.testing.assert_allclose(restored, [[0.1]], atol=1e-12)
    assert np.all(np.isfinite(log_det))


def test_probit_mapping_known_values():
    mapping = ProbitMapping()

    transformed, log_det = mapping.forward(np.array([0.5]))
    restored, inverse_log_det = mapping.inverse(transformed)

    np.testing.assert_allclose(transformed, [0.0], atol=1e-12)
    np.testing.assert_allclose(restored, [0.5], atol=1e-12)
    np.testing.assert_allclose(
        log_det, [0.5 * np.log(2.0 * np.pi)], atol=1e-12
    )
    np.testing.assert_allclose(log_det + inverse_log_det, 0.0, atol=1e-12)


def test_bounded_to_unit_hypercube_round_trip(particles):
    bounded_particles = particles
    bounds = {0: (-1.0, 2.0), 2: (0.0, 1.0)}
    transform = BoundedToUnitHypercubeTransform(bounds)
    latent, log_det = transform.forward(bounded_particles)
    assert np.all(to_numpy(latent[..., 0]) >= 0.0 - 1e-6)
    assert np.all(to_numpy(latent[..., 0]) <= 1.0 + 1e-6)
    assert np.all(to_numpy(latent[..., 2]) >= 0.0 - 1e-6)
    assert np.all(to_numpy(latent[..., 2]) <= 1.0 + 1e-6)
    expected_log_det = -np.log(3.0)
    assert_preserves_namespace(latent, particles)
    assert_allclose(log_det, expected_log_det, atol=1e-6)
    restored, restore_log_det = transform.inverse(latent)
    assert_allclose(restored, bounded_particles, atol=1e-6)
    assert_allclose(log_det + restore_log_det, 0.0, atol=1e-6)


def test_periodic_boundary_transform(particles):
    bounds = {1: (-np.pi, np.pi)}
    transform = PeriodicBoundaryTransform(bounds)
    latent, log_det = transform.forward(particles)
    assert_preserves_namespace(latent, particles)
    assert_allclose(log_det, 0.0)
    assert np.all(to_numpy(latent[..., 1]) >= -np.pi)
    assert np.all(to_numpy(latent[..., 1]) < np.pi)
    restored, _ = transform.inverse(latent)
    assert_allclose(restored, latent)


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
    assert_preserves_namespace(latent, particles)
    assert_allclose(restored, particles, atol=1e-6)
    assert_allclose(log_det + restore_log_det, 0.0, atol=1e-5)


def test_flow_transform_round_trip_and_clone(particles, shift_flow_factory):
    transform = FlowTransform(shift_flow_factory(n_dim=3))
    fitted = transform.fit(particles)
    assert fitted is not transform
    assert transform.flow.fitted_particles is None
    transform = fitted
    latent, log_det = transform.forward(particles)
    restored, restore_log_det = transform.inverse(latent)

    assert_preserves_namespace(latent, particles)
    assert_allclose(restored, particles, atol=1e-6)
    assert_allclose(log_det + restore_log_det, 0.0, atol=1e-6)
    assert_allclose(getattr(transform.flow, "fitted_particles"), particles)

    cloned = transform.clone()
    assert isinstance(cloned, FlowTransform)
    assert cloned is not transform
    assert cloned.flow is not transform.flow

    restored_transform = FlowTransform(shift_flow_factory(n_dim=3))
    restored_transform.load_state_dict(transform.state_dict())
    assert_allclose(restored_transform.flow.fitted_particles, particles)


def test_flow_transform_passes_rng_and_fit_kwargs(particles, rng):
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

    fitted = transform.fit(particles, rng=rng)

    assert transform.flow.fit_options is None
    assert fitted.flow.fit_options == {
        "rng": rng,
        "n_epochs": 7,
        "batch_size": 2,
    }
    assert fitted.fit_kwargs == {"n_epochs": 7, "batch_size": 2}


@pytest.mark.parametrize("reserved", ["context", "rng"])
def test_flow_transform_rejects_reserved_fit_kwargs(
    shift_flow_factory, reserved
):
    with pytest.raises(ValueError, match=reserved):
        FlowTransform(
            shift_flow_factory(n_dim=2), fit_kwargs={reserved: object()}
        )


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

    assert_preserves_namespace(latent, particles)
    assert_preserves_namespace(log_det, particles)
    assert_allclose(restored, particles, atol=1e-6)
    assert_allclose(log_det + inverse_log_det, 0.0, atol=1e-6)


@pytest.mark.parametrize(
    "mapping", [LogitMapping(), CauchitMapping(), ProbitMapping()]
)
def test_builtin_transforms_are_jittable_dataclass_pytrees(mapping):
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    particles = jnp.asarray([[0.2, 1.0], [0.8, 3.0]])
    transform = CompositeTransform(
        [
            BoundedToUnboundedTransform({0: (0.0, 1.0)}, mapping=mapping),
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
