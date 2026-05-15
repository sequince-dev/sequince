import numpy as np

from sequince.utils import in_bounds


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
