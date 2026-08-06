import numpy as np

from sequince.utils import in_bounds, normalize_log_weights


def test_normalize_log_weights():
    log_weights = np.log([1.0, 3.0]).tolist()

    normalized, log_norm, valid = normalize_log_weights(log_weights)

    np.testing.assert_allclose(np.exp(normalized), [0.25, 0.75])
    assert np.isclose(log_norm, np.log(4.0))
    assert valid


def test_normalize_degenerate_log_weights_to_uniform():
    normalized, log_norm, valid = normalize_log_weights(
        np.array([-np.inf, -np.inf])
    )

    np.testing.assert_allclose(np.exp(normalized), [0.5, 0.5])
    assert log_norm == 0.0
    assert not valid


def test_in_bounds():
    particles = np.array(
        [
            [0.0, -1.0, 0.25],
            [1.0, 2.0, 0.5],
            [-1.5, 0.5, 0.9],
        ],
        dtype=float,
    )
    bounds = np.array(
        [
            [-1.0, -2.0, 0.0],
            [1.0, 2.0, 1.0],
        ],
        dtype=float,
    )
    expected = np.array([True, True, False])
    result = in_bounds(particles, bounds)
    np.testing.assert_array_equal(result, expected)
