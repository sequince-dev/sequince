from dataclasses import dataclass

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
    WhiteningTransform,
    has_particle_aligned_auxiliary,
    select_transform_auxiliary,
)
from sequince.utils.pytrees import pytree_dataclass


class SignFoldingTransform(IdentityTransform):
    """Test-only many-to-one transform with explicit branch information."""

    def forward(self, particles, context=None, *, rng=None):
        del context, rng
        xp = namespace(particles)
        values = xp.asarray(particles)
        signs = xp.where(
            values < 0, -xp.ones_like(values), xp.ones_like(values)
        )
        return xp.abs(values), xp.zeros_like(values[..., 0]), signs

    def inverse(self, values, context=None, *, auxiliary=None, rng=None):
        del context
        xp = namespace(values)
        values = xp.asarray(values)
        if auxiliary is None:
            if rng is None:
                raise ValueError(
                    "SignFoldingTransform inverse requires auxiliary or rng."
                )
            choices = xp.asarray(rng.choice(2, size=values.shape))
            auxiliary = 2 * choices - 1
        return (
            values * xp.asarray(auxiliary),
            xp.zeros_like(values[..., 0]),
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
    _, log_det, auxiliary = transform.forward(np.array([0.0, 1.0, 2.0]))
    assert np.ndim(log_det) == 0
    assert auxiliary is None


def test_affine_all_columns(particles):
    xp = namespace(particles)
    transform = AffineTransform()
    fitted = transform.fit(particles)
    assert fitted is not transform
    assert transform._mean is None
    assert transform._scale is None
    transform = fitted
    latent, log_det, auxiliary = transform.forward(particles)
    restored, restore_log_det = transform.inverse(latent)
    assert auxiliary is None
    assert_preserves_namespace(latent, particles)
    assert_preserves_namespace(log_det, particles)
    assert_allclose(restored, particles, atol=1e-6)
    assert_allclose(log_det + restore_log_det, 0.0, atol=1e-6)
    assert_allclose(xp.mean(latent, axis=0), 0.0, atol=1e-6)


def test_affine_subset(particles):
    transform = AffineTransform(columns=[0, 2])
    transform = transform.fit(particles)
    latent, _, _ = transform.forward(particles)
    restored, _ = transform.inverse(latent)
    assert_preserves_namespace(latent, particles)
    assert_allclose(restored, particles, atol=1e-6)
    assert_allclose(latent[..., 1], particles[..., 1])
    assert not np.allclose(
        to_numpy(latent[..., 0]), to_numpy(particles[..., 0])
    )


def test_whitening_round_trip_and_covariance(particles):
    xp = namespace(particles)
    particles = xp.asarray(
        [
            [-2.0, 0.0, 1.0],
            [-1.0, 2.0, -1.0],
            [0.0, -1.0, 3.0],
            [1.0, 1.0, 0.0],
            [2.0, -2.0, -2.0],
            [3.0, 3.0, 2.0],
        ],
        dtype=particles.dtype,
    )
    transform = WhiteningTransform().fit(particles)

    latent, log_det, auxiliary = transform.forward(particles)
    restored, restore_log_det = transform.inverse(latent)
    centered = latent - xp.mean(latent, axis=0)
    covariance = xp.matmul(xp.permute_dims(centered, (1, 0)), centered)
    covariance = covariance / particles.shape[0]

    assert auxiliary is None
    assert_preserves_namespace(latent, particles)
    assert_allclose(restored, particles, atol=2e-5)
    assert_allclose(log_det + restore_log_det, 0.0, atol=1e-6)
    assert_allclose(xp.mean(latent, axis=0), 0.0, atol=2e-5)
    assert_allclose(covariance, xp.eye(particles.shape[1]), atol=2e-5)


@pytest.mark.parametrize(
    "particles",
    [
        np.ones((8, 3)),
        np.column_stack([np.linspace(-1.0, 1.0, 8), np.zeros(8), np.zeros(8)]),
        np.column_stack(
            [
                np.linspace(-1.0, 1.0, 8),
                np.linspace(-1.0, 1.0, 8) + 1e-12,
            ]
        ),
    ],
)
def test_whitening_handles_degenerate_covariance(particles):
    transform = WhiteningTransform().fit(particles)

    latent, log_det, _ = transform.forward(particles)
    restored, restore_log_det = transform.inverse(latent)

    assert np.all(np.isfinite(latent))
    assert np.all(np.isfinite(log_det))
    assert np.all(np.isfinite(restore_log_det))
    assert_allclose(restored, particles, atol=1e-8)


def test_whitening_requires_fit():
    with pytest.raises(RuntimeError, match="must be fitted"):
        WhiteningTransform().forward(np.zeros((2, 2)))


def test_whitening_validates_configuration():
    with pytest.raises(ValueError, match="strictly positive"):
        WhiteningTransform(eps=0.0)
    with pytest.raises(ValueError, match="2D particle array"):
        WhiteningTransform().fit(np.zeros(2))


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
    latent, log_det, _ = transform.forward(particles)
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
    latent, log_det, _ = transform.forward(particles)
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
    latent, log_det, _ = transform.forward(bounded_particles)
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


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_periodic_circular_mean(particles, dtype: str) -> None:
    xp = namespace(particles)
    cloud = xp.asarray(
        [[9.8, 42.0, 1.8], [2.2, -7.0, -1.8]],
        dtype=getattr(xp, dtype),
    )
    transform = PeriodicBoundaryTransform(
        {2: (-2.0, 2.0), 0: (2.0, 10.0)}, centering="circular_mean"
    )
    fitted = transform.fit(cloud)
    assert fitted is not transform
    assert transform._center is None
    latent, log_det, auxiliary = fitted.forward(cloud)
    assert auxiliary is None
    assert_preserves_namespace(latent, cloud)
    assert latent.dtype == cloud.dtype
    assert_allclose(latent, [[-0.2, 42, -0.2], [0.2, -7, 0.2]], atol=2e-6)
    restored, inverse_log_det = fitted.inverse(latent)
    assert_allclose(restored, cloud, atol=2e-6)
    assert_allclose(log_det + inverse_log_det, 0.0)
    vector, vector_log_det, _ = fitted.forward(cloud[0])
    assert vector_log_det.ndim == 0
    assert_allclose(vector, latent[0], atol=2e-6)
    # Inverse also wraps proposals that cross multiple periods.
    displaced = latent + xp.asarray([24.0, 0.0, -12.0], dtype=cloud.dtype)
    wrapped, _ = fitted.inverse(displaced)
    assert_allclose(wrapped, cloud, atol=3e-6)


def test_periodic_circular_mean_degenerate(particles) -> None:
    xp = namespace(particles)
    cloud = xp.asarray([[0.0], [0.5]], dtype=particles.dtype)
    fitted = PeriodicBoundaryTransform(
        {0: (0.0, 1.0)}, centering="circular_mean"
    ).fit(cloud)
    latent, _, _ = fitted.forward(cloud)
    assert np.all(np.isfinite(to_numpy(latent)))
    restored, _ = fitted.inverse(latent)
    assert_allclose(restored, cloud, atol=1e-6)


def test_periodic_circular_mean_validation() -> None:
    with pytest.raises(ValueError, match="centering"):
        PeriodicBoundaryTransform({0: (0, 1)}, centering="invalid")
    transform = PeriodicBoundaryTransform(
        {0: (0, 1)}, centering="circular_mean"
    )
    for method in (transform.forward, transform.inverse):
        with pytest.raises(ValueError, match="Call fit"):
            method(np.array([[0.2]]))
    for cloud in (np.empty((0, 1)), np.array([0.2])):
        with pytest.raises(ValueError, match="nonempty 2D"):
            transform.fit(cloud)


@pytest.mark.parametrize(
    "centering",
    ["circular-mean", "angular-mean", "circular_mean", "angular_mean"],
)
def test_periodic_circular_mean_jit(centering: str) -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    cloud = jnp.asarray([[0.95], [0.05]])
    transform = PeriodicBoundaryTransform({0: (0.0, 1.0)}, centering=centering)
    fitted = jax.jit(lambda t, x: t.fit(x))(transform, cloud)
    latent, _, _ = jax.jit(lambda t, x: t.forward(x))(fitted, cloud)
    restored, _ = jax.jit(lambda t, x: t.inverse(x))(fitted, latent)
    assert_allclose(latent, [[-0.05], [0.05]], atol=1e-6)
    assert_allclose(restored, cloud, atol=1e-6)


@pytest.mark.parametrize(
    "centering",
    ["circular-mean", "angular-mean", "circular_mean", "angular_mean"],
)
def test_periodic_angular_mean_alias(particles, centering: str) -> None:
    bounds = {0: (-1.0, 2.0), 2: (0.0, 1.0)}
    circular = PeriodicBoundaryTransform(
        bounds, centering="circular_mean"
    ).fit(particles)
    angular = PeriodicBoundaryTransform(bounds, centering=centering).fit(
        particles
    )
    assert angular.centering == "circular-mean"
    expected, expected_log_det, _ = circular.forward(particles)
    actual, actual_log_det, _ = angular.forward(particles)
    assert_allclose(actual, expected)
    assert_allclose(actual_log_det, expected_log_det)
    restored, _ = angular.inverse(actual)
    assert_allclose(restored, particles, atol=1e-6)


def test_periodic_boundary_transform(particles):
    bounds = {1: (-np.pi, np.pi)}
    transform = PeriodicBoundaryTransform(bounds)
    latent, log_det, _ = transform.forward(particles)
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
    latent, log_det, auxiliary = composite.forward(particles)
    restored, restore_log_det = composite.inverse(latent, auxiliary=auxiliary)
    assert auxiliary == (None, None)
    assert_preserves_namespace(latent, particles)
    assert_allclose(restored, particles, atol=1e-6)
    assert_allclose(log_det + restore_log_det, 0.0, atol=1e-5)


def test_whitening_composes_after_folding():
    particles = np.array([[-3.0, 1.0], [2.0, 2.0], [-1.0, 4.0]])
    composite = CompositeTransform(
        [SignFoldingTransform(), WhiteningTransform()]
    ).fit(particles)

    latent, log_det, auxiliary = composite.forward(particles)
    restored, restore_log_det = composite.inverse(latent, auxiliary=auxiliary)

    assert_allclose(restored, particles, atol=1e-10)
    assert_allclose(log_det + restore_log_det, 0.0, atol=1e-10)


def test_auxiliary_interprets_mutated_transformed_values():
    transform = SignFoldingTransform()
    particles = np.array([[-2.0], [3.0]])

    folded, _, auxiliary = transform.forward(particles)
    reconstructed, _ = transform.inverse(folded + 1.0, auxiliary=auxiliary)

    np.testing.assert_array_equal(reconstructed, [[-3.0], [4.0]])


def test_auxiliary_from_overlapping_forward_calls_is_independent():
    transform = SignFoldingTransform()
    first = np.array([[-2.0], [3.0]])
    second = np.array([[4.0], [-5.0]])

    folded_first, _, auxiliary_first = transform.forward(first)
    folded_second, _, auxiliary_second = transform.forward(second)
    restored_first, _ = transform.inverse(
        folded_first, auxiliary=auxiliary_first
    )
    restored_second, _ = transform.inverse(
        folded_second, auxiliary=auxiliary_second
    )

    np.testing.assert_array_equal(restored_first, first)
    np.testing.assert_array_equal(restored_second, second)


def test_inverse_can_define_transform_specific_rng_fallback(rng):
    transform = SignFoldingTransform()
    values = np.ones((8, 1))

    deterministic, _ = transform.inverse(
        values, auxiliary=-np.ones_like(values)
    )
    sampled, _ = transform.inverse(values, rng=rng)

    np.testing.assert_array_equal(deterministic, -values)
    assert sampled.shape == values.shape
    assert set(np.unique(sampled)).issubset({-1.0, 1.0})
    with pytest.raises(ValueError, match="requires auxiliary or rng"):
        transform.inverse(values)


def test_rng_fallback_preserves_functional_backend(
    particles, backend, backend_factory
):
    case = backend_factory(backend)
    transform = SignFoldingTransform()
    xp = namespace(particles)
    values = xp.ones_like(particles)

    sampled, _ = transform.inverse(values, rng=case.rng(7))

    assert_preserves_namespace(sampled, particles)
    assert set(np.unique(to_numpy(sampled))).issubset({-1.0, 1.0})


def test_composite_preserves_component_auxiliary_positionally():
    composite = CompositeTransform(
        [IdentityTransform(), SignFoldingTransform()]
    )
    particles = np.array([[-2.0], [3.0]])

    folded, _, auxiliary = composite.forward(particles)
    restored, _ = composite.inverse(folded, auxiliary=auxiliary)

    assert auxiliary[0] is None
    np.testing.assert_array_equal(auxiliary[1], [[-1.0], [1.0]])
    np.testing.assert_array_equal(restored, particles)


def test_composite_preserves_two_nontrivial_auxiliary_entries():
    composite = CompositeTransform(
        [SignFoldingTransform(), SignFoldingTransform()]
    )
    particles = np.array([[-2.0], [3.0]])

    folded, _, auxiliary = composite.forward(particles)
    restored, _ = composite.inverse(folded + 1.0, auxiliary=auxiliary)

    assert len(auxiliary) == 2
    np.testing.assert_array_equal(auxiliary[0], [[-1.0], [1.0]])
    np.testing.assert_array_equal(auxiliary[1], [[1.0], [1.0]])
    np.testing.assert_array_equal(restored, [[-3.0], [4.0]])


@pytest.mark.parametrize("auxiliary", [(), (None,), (None, None, None)])
def test_composite_rejects_incorrectly_sized_auxiliary(auxiliary):
    composite = CompositeTransform(
        [IdentityTransform(), SignFoldingTransform()]
    )

    with pytest.raises(ValueError, match="expected 2"):
        composite.inverse(np.ones((2, 1)), auxiliary=auxiliary)


def test_composite_rejects_malformed_auxiliary():
    composite = CompositeTransform(
        [IdentityTransform(), SignFoldingTransform()]
    )

    with pytest.raises(TypeError, match="tuple or list"):
        composite.inverse(np.ones((2, 1)), auxiliary={"sign": 1})


def test_composite_advances_one_rng_across_components():
    draws = []

    class RecordingRandomInverse(IdentityTransform):
        def inverse(self, values, context=None, *, auxiliary=None, rng=None):
            del context, auxiliary
            draw = rng.choice(2, size=np.asarray(values).shape)
            draws.append(np.asarray(draw))
            return values, np.zeros(np.asarray(values).shape[0])

    composite = CompositeTransform(
        [RecordingRandomInverse(), RecordingRandomInverse()]
    )

    composite.inverse(np.ones((64, 1)), rng=7)

    assert len(draws) == 2
    assert not np.array_equal(draws[0], draws[1])


@pytest.mark.parametrize(
    ("selection", "expected_signs"),
    [
        pytest.param(np.array([2, 0]), [[-1.0], [-1.0]], id="integer-indices"),
        pytest.param(
            np.array([True, False, True]),
            [[-1.0], [-1.0]],
            id="boolean-mask",
        ),
    ],
)
def test_select_transform_auxiliary_maps_nested_particle_leaves(
    selection, expected_signs
):
    @pytree_dataclass(data_fields=("signs", "other"))
    @dataclass(frozen=True)
    class Auxiliary:
        signs: object
        other: object

    auxiliary = {
        "component": Auxiliary(
            signs=np.array([[-1.0], [1.0], [-1.0]]),
            other=np.array([10.0, 20.0]),
        ),
        "label": "fold",
    }

    selected = select_transform_auxiliary(auxiliary, selection, n_particles=3)

    np.testing.assert_array_equal(selected["component"].signs, expected_signs)
    np.testing.assert_array_equal(selected["component"].other, [10.0, 20.0])
    assert selected["label"] == "fold"
    assert has_particle_aligned_auxiliary(auxiliary, n_particles=3)


def test_select_transform_auxiliary_supports_standard_pytrees():
    aligned = np.array([10, 20, 30])
    auxiliary = (
        aligned,
        [None, {"aligned": aligned[:, None], "label": "mode"}],
        np.array(7),
    )

    selected = select_transform_auxiliary(
        auxiliary, np.array([2, 0]), n_particles=3
    )

    np.testing.assert_array_equal(selected[0], [30, 10])
    assert isinstance(selected, tuple)
    assert isinstance(selected[1], list)
    assert selected[1][0] is None
    np.testing.assert_array_equal(selected[1][1]["aligned"], [[30], [10]])
    assert selected[1][1]["label"] == "mode"
    np.testing.assert_array_equal(selected[2], np.array(7))
    assert select_transform_auxiliary(None, 0, n_particles=3) is None


@pytest.mark.parametrize(
    ("selection_kind", "expected"),
    [
        pytest.param("scalar", [1.0, 2.0, 0.5], id="scalar-integer"),
        pytest.param(
            "indices",
            [[-0.5, 0.5, 0.9], [0.0, -1.0, 0.25]],
            id="integer-array",
        ),
        pytest.param(
            "mask",
            [[0.0, -1.0, 0.25], [-0.5, 0.5, 0.9]],
            id="boolean-mask",
        ),
    ],
)
def test_select_transform_auxiliary_uses_backend_native_indexing(
    particles, selection_kind, expected
):
    xp = namespace(particles)
    if selection_kind == "scalar":
        selection = xp.asarray(1)
    elif selection_kind == "indices":
        selection = xp.asarray([2, 0])
    else:
        selection = xp.asarray([True, False, True])

    selected = select_transform_auxiliary(
        particles, selection, n_particles=particles.shape[0]
    )

    assert_preserves_namespace(selected, particles)
    np.testing.assert_allclose(to_numpy(selected), expected, rtol=1e-6)


def test_flow_transform_round_trip_and_clone(particles, shift_flow_factory):
    transform = FlowTransform(shift_flow_factory(n_dim=3))
    fitted = transform.fit(particles)
    assert fitted is not transform
    assert transform.flow.fitted_particles is None
    transform = fitted
    latent, log_det, _ = transform.forward(particles)
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

    latent, log_det, auxiliary = transform.forward(particles)
    restored, inverse_log_det = transform.inverse(latent)
    assert auxiliary == (None, None)

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
    latent, log_det, auxiliary = jax.jit(
        lambda current, values: current.forward(values)
    )(fitted, particles)
    restored, inverse_log_det = jax.jit(
        lambda current, values: current.inverse(values)
    )(fitted, latent)

    assert auxiliary == (None, None)
    np.testing.assert_allclose(restored, particles, atol=1e-6)
    np.testing.assert_allclose(log_det + inverse_log_det, 0.0, atol=1e-6)
